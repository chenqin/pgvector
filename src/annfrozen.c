#include "postgres.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "access/table.h"
#include "access/tableam.h"
#include "access/visibilitymap.h"
#include "access/xlog.h"
#include "access/xloginsert.h"
#include "ann.h"
#include "ann/filter.h"
#include "access/stratnum.h"
#include "catalog/pg_opfamily_d.h"
#include "catalog/pg_operator_d.h"
#include "catalog/pg_trigger.h"
#include "catalog/pg_type_d.h"
#include "commands/defrem.h"
#include "commands/trigger.h"
#include "common/hashfn.h"
#include "common/relpath.h"
#include "executor/spi.h"
#include "executor/tuptable.h"
#include "funcapi.h"
#include "miscadmin.h"
#include "utils/acl.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "utils/rls.h"
#include "utils/tuplesort.h"
#include "utils/tuplestore.h"
#include "utils/varlena.h"

/* Bytes of a page's contents available to array elements. */
#define ANN_ARRAY_SPACE (BLCKSZ - MAXALIGN(SizeOfPageHeaderData) - MAXALIGN(sizeof(AnnPageOpaqueData)))
#define ANN_FROZEN_SEED UINT64CONST(0x616e6e5f66726f7a)

StaticAssertDecl(MAXALIGN(SizeOfPageHeaderData) + MAXALIGN(sizeof(AnnMetaPageData)) +
				 AnnFrozenMetaSize(ANN_FROZEN_MAX_COLUMNS) <=
				 BLCKSZ - MAXALIGN(sizeof(AnnPageOpaqueData)),
				 "frozen ann metapage does not fit");

/*
 * The guard keeps a frozen table frozen: a BEFORE statement trigger on
 * INSERT, UPDATE, DELETE and TRUNCATE that always fails. Frozen builds
 * require it, so the rows the GPU index was built from stay the table's rows.
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(ann_frozen_guard);
Datum
ann_frozen_guard(PG_FUNCTION_ARGS)
{
	TriggerData *trigdata = (TriggerData *) fcinfo->context;

	if (!CALLED_AS_TRIGGER(fcinfo))
		elog(ERROR, "ann_frozen_guard: not called by trigger manager");
	ereport(ERROR,
			(errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
			 errmsg("table \"%s\" is frozen for ann indexes",
					RelationGetRelationName(trigdata->tg_relation)),
			 errhint("Drop its frozen ann indexes and its ann_frozen_guard trigger to change it.")));
	PG_RETURN_NULL();
}

/* The guard trigger's OID, or InvalidOid when the table has no valid guard. */
static Oid
FindGuard(Relation heap)
{
	TriggerDesc *desc = heap->trigdesc;

	if (desc == NULL)
		return InvalidOid;
	for (int i = 0; i < desc->numtriggers; i++)
	{
		Trigger    *trigger = &desc->triggers[i];
		int16		type = trigger->tgtype;
		FmgrInfo	flinfo;

		if (trigger->tgenabled != TRIGGER_FIRES_ALWAYS || trigger->tgnattr > 0 ||
			trigger->tgqual != NULL || TRIGGER_FOR_ROW(type) || !TRIGGER_FOR_BEFORE(type) ||
			!TRIGGER_FOR_INSERT(type) || !TRIGGER_FOR_UPDATE(type) ||
			!TRIGGER_FOR_DELETE(type) || !TRIGGER_FOR_TRUNCATE(type))
			continue;
		fmgr_info(trigger->tgfoid, &flinfo);
		if (flinfo.fn_addr == ann_frozen_guard)
			return trigger->tgoid;
	}
	return InvalidOid;
}

/*
 * ann_freeze(table): install the guard trigger, firing even under
 * session_replication_role = replica. Returns false if it is already there.
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(ann_freeze);
Datum
ann_freeze(PG_FUNCTION_ARGS)
{
	Oid			relid = PG_GETARG_OID(0);
	Relation	heap = table_open(relid, ShareRowExclusiveLock);
	char	   *table;
	char	   *function;

	if (heap->rd_rel->relkind != RELKIND_RELATION)
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("\"%s\" is not a table", RelationGetRelationName(heap))));
	if (OidIsValid(FindGuard(heap)))
	{
		table_close(heap, NoLock);
		PG_RETURN_BOOL(false);
	}
	table = quote_qualified_identifier(get_namespace_name(RelationGetNamespace(heap)),
									   RelationGetRelationName(heap));
	function = quote_qualified_identifier(get_namespace_name(get_func_namespace(fcinfo->flinfo->fn_oid)),
										  "ann_frozen_guard");
	table_close(heap, NoLock);
	if (SPI_connect() != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed");
	if (SPI_execute(psprintf("CREATE TRIGGER ann_frozen_guard BEFORE INSERT OR UPDATE OR DELETE OR TRUNCATE "
							 "ON %s FOR EACH STATEMENT EXECUTE FUNCTION %s()", table, function),
					false, 0) != SPI_OK_UTILITY ||
		SPI_execute(psprintf("ALTER TABLE %s ENABLE ALWAYS TRIGGER ann_frozen_guard", table),
					false, 0) != SPI_OK_UTILITY)
		elog(ERROR, "could not create the ann_frozen_guard trigger");
	SPI_finish();
	PG_RETURN_BOOL(true);
}

/* Array writers fill one page at a time, holding it exclusively. */
typedef struct ArrayWriter
{
	Relation	index;
	AnnArray   *array;
	bool		reserved;		/* pages exist already */
	Buffer		buf;
	uint32		used;			/* elements on the current page */
	uint64		written;
} ArrayWriter;

static void
InitArray(AnnArray *array, uint32 width, uint64 count)
{
	array->width = width;
	array->perPage = ANN_ARRAY_SPACE / width;
	array->count = count;
	array->pages = (count + array->perPage - 1) / array->perPage;
	array->start = InvalidBlockNumber;
}

/* Append an empty ann page to the index; returned locked. */
static Buffer
NewPage(Relation index)
{
	Buffer		buf = ReadBufferExtended(index, MAIN_FORKNUM, P_NEW, RBM_NORMAL, NULL);
	Page		page;

	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	page = BufferGetPage(buf);
	PageInit(page, BufferGetPageSize(buf), sizeof(AnnPageOpaqueData));
	AnnPageGetOpaque(page)->page_id = ANN_PAGE_ID;
	return buf;
}

/*
 * Extend the index by an array's pages, so arrays whose lengths are known can
 * be filled side by side while another array is appended after them.
 */
static void
ReserveArray(Relation index, AnnArray *array)
{
	array->start = RelationGetNumberOfBlocks(index);
	for (BlockNumber p = 0; p < array->pages; p++)
	{
		Buffer		buf = NewPage(index);

		if (BufferGetBlockNumber(buf) != array->start + p)
			elog(ERROR, "unexpected ann index extension");
		MarkBufferDirty(buf);
		UnlockReleaseBuffer(buf);
	}
}

static void
WriterBegin(ArrayWriter *writer, Relation index, AnnArray *array, bool reserved)
{
	writer->index = index;
	writer->array = array;
	writer->reserved = reserved;
	writer->buf = InvalidBuffer;
	writer->used = 0;
	writer->written = 0;
	if (!reserved)
	{
		array->start = RelationGetNumberOfBlocks(index);
		array->pages = 0;
		array->count = 0;
	}
}

/* pd_lower covers the elements, so full-page images keep them. */
static void
WriterFinishPage(ArrayWriter *writer)
{
	Page		page = BufferGetPage(writer->buf);

	((PageHeader) page)->pd_lower = MAXALIGN(SizeOfPageHeaderData) + writer->used * writer->array->width;
	MarkBufferDirty(writer->buf);
	UnlockReleaseBuffer(writer->buf);
	writer->buf = InvalidBuffer;
}

/* The next element's slot, zeroed by PageInit. */
static char *
WriterNext(ArrayWriter *writer)
{
	AnnArray   *array = writer->array;
	char	   *slot;

	if (BufferIsValid(writer->buf) && writer->used == array->perPage)
		WriterFinishPage(writer);
	if (!BufferIsValid(writer->buf))
	{
		BlockNumber page = writer->written / array->perPage;

		if (writer->reserved)
		{
			if (page >= array->pages)
				elog(ERROR, "ann array overflow");
			writer->buf = ReadBuffer(writer->index, array->start + page);
			LockBuffer(writer->buf, BUFFER_LOCK_EXCLUSIVE);
			AnnCheckPage(BufferGetPage(writer->buf));
		}
		else
		{
			writer->buf = NewPage(writer->index);
			if (BufferGetBlockNumber(writer->buf) != array->start + page)
				elog(ERROR, "unexpected ann index extension");
			array->pages++;
		}
		writer->used = 0;
	}
	slot = PageGetContents(BufferGetPage(writer->buf)) + writer->used * array->width;
	writer->used++;
	writer->written++;
	return slot;
}

static void
WriterEnd(ArrayWriter *writer)
{
	if (BufferIsValid(writer->buf))
		WriterFinishPage(writer);
	if (!writer->reserved)
		writer->array->count = writer->written;
	else if (writer->written != writer->array->count)
		elog(ERROR, "ann array has " UINT64_FORMAT " of " UINT64_FORMAT " elements",
			 writer->written, writer->array->count);
}

/*
 * Read-only mappings of frozen index files, per backend, by relfilenode.
 * Every backend maps the same files, so the operating system keeps one copy
 * of the pages in host memory and reads take no buffer pins or locks.
 * Frozen pages never change after the build, which flushed them, so the
 * files hold their final content; TID pages, which VACUUM can still
 * change, always go through shared buffers. During recovery (a standby
 * replays into shared buffers, not the files) nothing is mapped.
 */
typedef struct AnnFrozenMap
{
	RelFileLocator locator;		/* hash key */
	int			nsegments;		/* 0 when the files could not be mapped */
	char	  **bases;
	BlockNumber blocks;
} AnnFrozenMap;

bool		ann_frozen_mmap = true;
static HTAB *frozenMaps = NULL;
static Oid	buildingIndex = InvalidOid; /* reads during its own build use buffers */

static const AnnFrozenMap *
MapIndex(Relation index)
{
	AnnFrozenMap *map;
	bool		found;
	char	   *path;
	MemoryContext oldCtx;

	if (!ann_frozen_mmap || RecoveryInProgress() || index->rd_rel->relpersistence != RELPERSISTENCE_PERMANENT ||
		AnnFrozenMeta(index) == NULL || !AnnFrozenFlushed(index))
		return NULL;
	if (frozenMaps == NULL)
	{
		HASHCTL		ctl;

		ctl.keysize = sizeof(RelFileLocator);
		ctl.entrysize = sizeof(AnnFrozenMap);
		frozenMaps = hash_create("ann frozen index mappings", 16, &ctl, HASH_ELEM | HASH_BLOBS);
	}
	map = hash_search(frozenMaps, &index->rd_locator, HASH_ENTER, &found);
	if (found)
		return map->nsegments > 0 ? map : NULL;
	map->nsegments = 0;
	map->blocks = RelationGetNumberOfBlocks(index);
	oldCtx = MemoryContextSwitchTo(TopMemoryContext);
	map->bases = palloc0(sizeof(char *) * ((map->blocks + RELSEG_SIZE - 1) / RELSEG_SIZE + 1));
	MemoryContextSwitchTo(oldCtx);
	path = relpathperm(index->rd_locator, MAIN_FORKNUM);
	for (BlockNumber first = 0; first < map->blocks; first += RELSEG_SIZE)
	{
		int			segment = first / RELSEG_SIZE;
		char	   *file = segment == 0 ? path : psprintf("%s.%d", path, segment);
		size_t		bytes = (size_t) Min(map->blocks - first, RELSEG_SIZE) * BLCKSZ;
		struct stat st;
		int			fd = open(file, O_RDONLY | PG_BINARY, 0);
		void	   *base = MAP_FAILED;

		if (fd >= 0 && fstat(fd, &st) == 0 && (size_t) st.st_size >= bytes)
			base = mmap(NULL, bytes, PROT_READ, MAP_SHARED, fd, 0);
		if (fd >= 0)
			close(fd);
		if (base == MAP_FAILED)
		{
			elog(DEBUG1, "could not map \"%s\"; frozen ann index pages are read through shared buffers", file);
			for (int s = 0; s < segment; s++)
				(void) munmap(map->bases[s], (size_t) Min(map->blocks - (BlockNumber) s * RELSEG_SIZE,
														  RELSEG_SIZE) * BLCKSZ);
			return NULL;
		}
		map->bases[segment] = base;
	}
	map->nsegments = (map->blocks + RELSEG_SIZE - 1) / RELSEG_SIZE;
	return map->nsegments > 0 ? map : NULL;
}

static Page
MappedPage(const AnnFrozenMap *map, BlockNumber block)
{
	return (Page) (map->bases[block / RELSEG_SIZE] + (size_t) (block % RELSEG_SIZE) * BLCKSZ);
}

/* The array's first and last pages in the files are the finished ones. */
static bool
MappedArrayValid(const AnnFrozenMap *map, const AnnArray *array)
{
	BlockNumber ends[2] = {array->start, array->start + array->pages - 1};

	if (array->pages == 0)
		return true;
	if (array->start + array->pages > map->blocks)
		return false;
	for (int i = 0; i < 2; i++)
	{
		Page		page = MappedPage(map, ends[i]);
		uint64		count = i == 0 && array->pages > 1 ? array->perPage :
			array->count - (uint64) (array->pages - 1) * array->perPage;

		if (PageIsNew(page) || PageGetSpecialSize(page) != MAXALIGN(sizeof(AnnPageOpaqueData)) ||
			AnnPageGetOpaque(page)->page_id != ANN_PAGE_ID ||
			((PageHeader) page)->pd_lower != MAXALIGN(SizeOfPageHeaderData) + count * array->width)
			return false;
	}
	return true;
}

void
AnnArrayReaderInit(AnnArrayReader *reader, Relation index, const AnnArray *array,
				   BufferAccessStrategy strategy)
{
	const AnnFrozenMap *map;
	AnnFrozenMetaData *meta;

	reader->index = index;
	reader->array = array;
	reader->strategy = strategy;
	reader->buf = InvalidBuffer;
	reader->page = InvalidBlockNumber;
	reader->locked = false;
	reader->map = NULL;
	if (RelationGetRelid(index) != buildingIndex && (map = MapIndex(index)) != NULL && (meta = AnnFrozenMeta(index)) != NULL &&
		array->start != meta->tids.start && MappedArrayValid(map, array))
		reader->map = map;
}

/*
 * Pin and share-lock one page of the array; returns its first element and
 * sets count to the elements on it. The lock lasts until the next call or
 * AnnArrayUnlockPage.
 */
char *
AnnArrayLockPage(AnnArrayReader *reader, BlockNumber page, uint32 *count)
{
	const AnnArray *array = reader->array;
	Page		p;

	if (page >= array->pages)
		elog(ERROR, "ann array page %u is out of range", page);
	if (reader->map)
	{
		if (count)
			*count = page + 1 < array->pages ? array->perPage :
				(uint32) (array->count - (uint64) page * array->perPage);
		return PageGetContents(MappedPage(reader->map, array->start + page));
	}
	AnnArrayUnlockPage(reader);
	if (reader->page != page)
	{
		if (BufferIsValid(reader->buf))
			ReleaseBuffer(reader->buf);
		reader->buf = ReadBufferExtended(reader->index, MAIN_FORKNUM, array->start + page,
										 RBM_NORMAL, reader->strategy);
		reader->page = page;
	}
	LockBuffer(reader->buf, BUFFER_LOCK_SHARE);
	reader->locked = true;
	p = BufferGetPage(reader->buf);
	AnnCheckPage(p);
	if (count)
		*count = page + 1 < array->pages ? array->perPage :
			(uint32) (array->count - (uint64) page * array->perPage);
	return PageGetContents(p);
}

void
AnnArrayUnlockPage(AnnArrayReader *reader)
{
	if (reader->locked)
	{
		LockBuffer(reader->buf, BUFFER_LOCK_UNLOCK);
		reader->locked = false;
	}
}

/* Copy elements first .. first + n - 1 to out. */
void
AnnArrayReadRange(AnnArrayReader *reader, uint64 first, uint64 n, void *out)
{
	const AnnArray *array = reader->array;
	char	   *target = out;

	if (first > array->count || n > array->count - first)
		elog(ERROR, "ann array element " UINT64_FORMAT " is out of range", first + n);
	while (n > 0)
	{
		uint32		offset = first % array->perPage;
		uint64		take = Min(n, array->perPage - offset);
		char	   *data = AnnArrayLockPage(reader, first / array->perPage, NULL);

		memcpy(target, data + (size_t) offset * array->width, take * array->width);
		AnnArrayUnlockPage(reader);
		target += take * array->width;
		first += take;
		n -= take;
	}
}

void
AnnArrayRead(AnnArrayReader *reader, uint64 element, void *out)
{
	AnnArrayReadRange(reader, element, 1, out);
}

void
AnnArrayReaderEnd(AnnArrayReader *reader)
{
	AnnArrayUnlockPage(reader);
	if (BufferIsValid(reader->buf))
		ReleaseBuffer(reader->buf);
	reader->buf = InvalidBuffer;
	reader->page = InvalidBlockNumber;
}

/* Whether the index has the frozen layout, from its metapage. */
bool
AnnIsFrozen(Relation index)
{
	return AnnFrozenMeta(index) != NULL;
}

/*
 * A frozen index never changes after its build, so its metadata and the
 * description of its predicate columns are kept with the relcache entry
 * (rd_amcache, freed when the entry is invalidated, as by REINDEX).
 */
typedef struct AnnIndexCache
{
	AnnFrozenMetaData *meta;	/* NULL for a mutable index */
	bool		flushed;		/* the build flushed the pages to the files */
	bool		columnsReady;
	AnnPredicateColumns columns;
} AnnIndexCache;

static AnnFrozenMetaData *ReadFrozenMeta(Relation index, bool *flushed);

static AnnIndexCache *
IndexCache(Relation index)
{
	if (index->rd_amcache == NULL)
	{
		MemoryContext oldCtx = MemoryContextSwitchTo(index->rd_indexcxt);
		AnnIndexCache *cache = palloc0(sizeof(AnnIndexCache));

		cache->meta = ReadFrozenMeta(index, &cache->flushed);
		MemoryContextSwitchTo(oldCtx);
		index->rd_amcache = cache;
	}
	return (AnnIndexCache *) index->rd_amcache;
}

bool
AnnFrozenFlushed(Relation index)
{
	return IndexCache(index)->flushed;
}

/* The cached frozen metadata (read-only), or NULL for a mutable index. */
AnnFrozenMetaData *
AnnFrozenMeta(Relation index)
{
	return IndexCache(index)->meta;
}

/* The cached predicate columns slot; *ready says whether it is filled. */
AnnPredicateColumns *
AnnFrozenColumnsCache(Relation index, bool **ready)
{
	AnnIndexCache *cache = IndexCache(index);

	*ready = &cache->columnsReady;
	return &cache->columns;
}

/* A copy of the frozen metadata; NULL for a mutable index when missingOk. */
AnnFrozenMetaData *
AnnFrozenGetMeta(Relation index, bool missingOk)
{
	AnnFrozenMetaData *cached = AnnFrozenMeta(index);
	AnnFrozenMetaData *meta = NULL;

	if (cached)
	{
		meta = palloc(AnnFrozenMetaSize(ANN_FROZEN_MAX_COLUMNS));
		memcpy(meta, cached, AnnFrozenMetaSize(ANN_FROZEN_MAX_COLUMNS));
	}
	else if (!missingOk)
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("\"%s\" is not a frozen ann index", RelationGetRelationName(index)),
				 errhint("Create it WITH (frozen = true).")));
	return meta;
}

/* The metapage's frozen metadata, or NULL; allocated in the current context. */
static AnnFrozenMetaData *
ReadFrozenMeta(Relation index, bool *flushed)
{
	Buffer		buf = ReadBuffer(index, ANN_METAPAGE_BLKNO);
	Page		page;
	AnnFrozenMetaData *meta = NULL;

	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	AnnCheckMetaPage(page);
	if (AnnVersionFrozen(AnnPageGetMeta(page)->version))
	{
		AnnFrozenMetaData *stored = AnnPageGetFrozen(page);
		bool		bitsets = AnnPageGetMeta(page)->version == ANN_FROZEN_VERSION;

		if (stored->ncolumns > ANN_FROZEN_MAX_COLUMNS)
			ereport(ERROR, (errcode(ERRCODE_INDEX_CORRUPTED), errmsg("invalid frozen ann index metapage")));
		meta = palloc0(AnnFrozenMetaSize(ANN_FROZEN_MAX_COLUMNS));
		memcpy(meta, stored, bitsets ? AnnFrozenMetaSize(stored->ncolumns) : AnnFrozenMetaSizeV2(stored->ncolumns));
	}
	*flushed = meta != NULL && AnnPageGetMeta(page)->insertPage == ANN_FROZEN_FLUSHED;
	UnlockReleaseBuffer(buf);
	return meta;
}

/*
 * Collation of an index column (0-based) from its table column: INCLUDE
 * columns have no collation in the index itself.
 */
Oid
AnnFrozenColumnCollation(Relation index, int column)
{
	Oid			type;
	int32		typmod;
	Oid			collation;

	get_atttypetypmodcoll(index->rd_index->indrelid, index->rd_index->indkey.values[column],
						  &type, &typmod, &collation);
	return collation;
}

/* Current table column name of an index column (1-based). */
const char *
AnnFrozenColumnName(Relation index, int attno)
{
	AttrNumber	heapAttno = index->rd_index->indkey.values[attno - 1];

	if (heapAttno <= 0)
		elog(ERROR, "frozen ann index column %d is not a table column", attno);
	return get_attname(index->rd_index->indrelid, heapAttno, false);
}

typedef struct FrozenBuildState
{
	int			dims;
	int			ncolumns;		/* INCLUDE columns after the key */
	int			bitsetMin;		/* reloption: bitset pages from this many rows */
	Oid			keyType;
	bool		text[ANN_FROZEN_MAX_COLUMNS];	/* dictionary-coded columns */
	Oid			collations[ANN_FROZEN_MAX_COLUMNS];
	TupleDesc	sortDesc;
	TupleTableSlot *slot;
	Tuplesortstate *sort;
	uint64		rows;
	MemoryContext tmpCtx;
} FrozenBuildState;

static int64
KeyValue(Datum value, Oid type)
{
	if (type == INT2OID)
		return DatumGetInt16(value);
	if (type == INT4OID)
		return DatumGetInt32(value);
	return DatumGetInt64(value);
}

bool
AnnFrozenAttributeType(Oid type)
{
	return type == INT2OID || type == INT4OID || type == INT8OID || type == BOOLOID || type == DATEOID ||
		AnnFrozenTextType(type);
}

/* A text column's string for a code, as a text datum. */
Datum
AnnFrozenTextDatum(Relation index, const AnnFrozenMetaData *meta, int column, int32 code)
{
	const AnnFrozenColumnExtra *extra = &AnnFrozenGetExtra(meta)[column];
	AnnArrayReader starts;
	AnnArrayReader strings;
	uint32		range[2];
	text	   *result;

	AnnArrayReaderInit(&starts, index, &extra->starts, NULL);
	AnnArrayReaderInit(&strings, index, &extra->strings, NULL);
	AnnArrayReadRange(&starts, (uint64) code, 2, range);
	result = palloc(VARHDRSZ + range[1] - range[0]);
	SET_VARSIZE(result, VARHDRSZ + range[1] - range[0]);
	if (range[1] > range[0])
		AnnArrayReadRange(&strings, range[0], range[1] - range[0], VARDATA(result));
	AnnArrayReaderEnd(&starts);
	AnnArrayReaderEnd(&strings);
	return PointerGetDatum(result);
}

/*
 * First code whose string sorts at or after the value (after it when
 * upper), in the column's collation: a binary search over the dictionary.
 */
int64
AnnFrozenTextSearch(Relation index, const AnnFrozenMetaData *meta, int column, Oid collation,
					text *value, bool upper)
{
	const AnnFrozenColumnExtra *extra = &AnnFrozenGetExtra(meta)[column];
	AnnArrayReader starts;
	AnnArrayReader strings;
	int64		low = 0;
	int64		high = extra->starts.count > 0 ? (int64) extra->starts.count - 1 : 0;
	char	   *buffer = NULL;
	uint32		capacity = 0;

	AnnArrayReaderInit(&starts, index, &extra->starts, NULL);
	AnnArrayReaderInit(&strings, index, &extra->strings, NULL);
	while (low < high)
	{
		int64		middle = low + (high - low) / 2;
		uint32		range[2];
		int			cmp;

		AnnArrayReadRange(&starts, middle, 2, range);
		if (range[1] - range[0] > capacity)
		{
			capacity = Max(range[1] - range[0], 64);
			buffer = buffer ? repalloc(buffer, capacity) : palloc(capacity);
		}
		if (range[1] > range[0])
			AnnArrayReadRange(&strings, range[0], range[1] - range[0], buffer);
		cmp = varstr_cmp(buffer ? buffer : "", range[1] - range[0], VARDATA_ANY(value), VARSIZE_ANY_EXHDR(value),
						 collation);
		if (cmp < 0 || (upper && cmp == 0))
			low = middle + 1;
		else
			high = middle;
	}
	AnnArrayReaderEnd(&starts);
	AnnArrayReaderEnd(&strings);
	return low;
}

/* The int32 an attribute stores: the integer, 0/1 for bool, days for date. */
int32
AnnFrozenAttributeValue(Datum value, bool isnull, Oid type)
{
	int64		result;

	if (isnull)
		return ANN_FROZEN_NULL;
	switch (type)
	{
		case INT2OID:
			result = DatumGetInt16(value);
			break;
		case INT8OID:
			result = DatumGetInt64(value);
			break;
		case BOOLOID:
			result = DatumGetBool(value) ? 1 : 0;
			break;
		default:
			result = DatumGetInt32(value);	/* integer, date */
			break;
	}
	if (result <= ANN_FROZEN_NULL || result > PG_INT32_MAX)
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("frozen ann index attribute value " INT64_FORMAT " is out of range", result),
				 errdetail("Attributes hold values above %d; %d stands for NULL, and -infinity dates are not stored.",
						   ANN_FROZEN_NULL, ANN_FROZEN_NULL)));
	return (int32) result;
}

Datum
AnnFrozenAttributeDatum(int32 value, Oid type)
{
	switch (type)
	{
		case INT2OID:
			return Int16GetDatum((int16) value);
		case INT8OID:
			return Int64GetDatum((int64) value);
		case BOOLOID:
			return BoolGetDatum(value != 0);
		default:
			return Int32GetDatum(value);
	}
}

/* Key, heap TID, vector and attributes of each row, sorted by key. */
static void
FrozenBuildCallback(Relation index, ItemPointer tid, Datum *values, bool *isnull,
					bool tupleIsAlive, void *state)
{
	FrozenBuildState *bs = state;
	TupleTableSlot *slot = bs->slot;
	MemoryContext oldCtx = MemoryContextSwitchTo(bs->tmpCtx);
	Vector	   *vector;

	if (!tupleIsAlive)
		ereport(ERROR,
				(errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
				 errmsg("table has row versions that are not visible to everyone"),
				 errhint("Run VACUUM (FREEZE) on the table, then create the index.")));
	if (isnull[0] || isnull[1])
		ereport(ERROR,
				(errcode(ERRCODE_NOT_NULL_VIOLATION),
				 errmsg("frozen ann indexes do not accept NULL %s", isnull[0] ? "vectors" : "keys")));
	vector = DatumGetVector(values[0]);
	if (vector->dim != bs->dims)
		ereport(ERROR, (errcode(ERRCODE_DATA_EXCEPTION), errmsg("different vector dimensions")));
	ExecClearTuple(slot);
	slot->tts_values[0] = Int64GetDatum(KeyValue(values[1], bs->keyType));
	slot->tts_values[1] = Int64GetDatum(((int64) ItemPointerGetBlockNumber(tid) << 16) |
										ItemPointerGetOffsetNumber(tid));
	slot->tts_values[2] = PointerGetDatum(vector);
	memset(slot->tts_isnull, 0, sizeof(bool) * (3 + bs->ncolumns));
	for (int c = 0; c < bs->ncolumns; c++)
	{
		if (bs->text[c])
		{
			/* Text goes through the sort; its code is assigned afterwards. */
			slot->tts_isnull[3 + c] = isnull[2 + c];
			slot->tts_values[3 + c] = isnull[2 + c] ? (Datum) 0 :
				PointerGetDatum(PG_DETOAST_DATUM_PACKED(values[2 + c]));
		}
		else
			slot->tts_values[3 + c] = Int32GetDatum(AnnFrozenAttributeValue(values[2 + c], isnull[2 + c],
																			TupleDescAttr(RelationGetDescr(index), 2 + c)->atttypid));
	}
	ExecStoreVirtualTuple(slot);
	tuplesort_puttupleslot(bs->sort, slot);
	bs->rows++;
	MemoryContextSwitchTo(oldCtx);
	MemoryContextReset(bs->tmpCtx);
}

/*
 * The contract: a permanent table that cannot change (the guard trigger),
 * whose rows are all visible to everyone (all-frozen visibility map), indexed
 * on plain columns with an integer key first among the INCLUDE columns.
 */
static Oid
CheckFrozenTable(Relation heap, Relation index, IndexInfo *indexInfo)
{
	TupleDesc	desc = RelationGetDescr(index);
	BlockNumber blocks;
	BlockNumber visible;
	BlockNumber frozen;
	Oid			guard;
	Oid			type;

	if (heap->rd_rel->relpersistence != RELPERSISTENCE_PERMANENT)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("frozen ann indexes require a permanent table")));
	if (indexInfo->ii_Predicate != NIL || indexInfo->ii_Expressions != NIL ||
		indexInfo->ii_IndexAttrNumbers[0] == 0)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("frozen ann indexes cannot be partial or on expressions")));
	if (indexInfo->ii_NumIndexAttrs < 2)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_OBJECT_DEFINITION),
				 errmsg("frozen ann indexes need the table's key as their first INCLUDE column"),
				 errhint("For example: INCLUDE (id, tenant).")));
	type = TupleDescAttr(desc, 1)->atttypid;
	if (type != INT2OID && type != INT4OID && type != INT8OID)
		ereport(ERROR,
				(errcode(ERRCODE_DATATYPE_MISMATCH),
				 errmsg("the key of a frozen ann index must be smallint, integer or bigint")));
	for (int i = 2; i < desc->natts; i++)
	{
		type = TupleDescAttr(desc, i)->atttypid;
		if (AnnFrozenTextType(type) && OidIsValid(AnnFrozenColumnCollation(index, i)) &&
			!get_collation_isdeterministic(AnnFrozenColumnCollation(index, i)))
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("text INCLUDE column \"%s\" of a frozen ann index needs a deterministic collation",
							NameStr(TupleDescAttr(desc, i)->attname))));
		if (!AnnFrozenAttributeType(type))
			ereport(ERROR,
					(errcode(ERRCODE_DATATYPE_MISMATCH),
					 errmsg("INCLUDE column \"%s\" of a frozen ann index must be smallint, integer, bigint, boolean, date, text or varchar",
							NameStr(TupleDescAttr(desc, i)->attname))));
	}
	guard = FindGuard(heap);
	if (!OidIsValid(guard))
		ereport(ERROR,
				(errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
				 errmsg("table \"%s\" is not frozen", RelationGetRelationName(heap)),
				 errhint("Run SELECT ann_freeze('%s') and VACUUM (FREEZE) on it, then create the index.",
						 RelationGetRelationName(heap))));
	blocks = RelationGetNumberOfBlocks(heap);
	visibilitymap_count(heap, &visible, &frozen);

	/*
	 * Bulk loads extend the table ahead of use, leaving all-zero pages that
	 * VACUUM never marks; they hold no rows, so they count as frozen.
	 */
	if (frozen < blocks)
	{
		Buffer		vmbuf = InvalidBuffer;
		BufferAccessStrategy strategy = GetAccessStrategy(BAS_BULKREAD);

		for (BlockNumber blkno = 0; blkno < blocks && frozen < blocks; blkno++)
		{
			Buffer		buf;

			if (VM_ALL_FROZEN(heap, blkno, &vmbuf))
				continue;
			buf = ReadBufferExtended(heap, MAIN_FORKNUM, blkno, RBM_NORMAL, strategy);
			LockBuffer(buf, BUFFER_LOCK_SHARE);
			if (PageIsNew(BufferGetPage(buf)))
				frozen++;
			UnlockReleaseBuffer(buf);
		}
		if (BufferIsValid(vmbuf))
			ReleaseBuffer(vmbuf);
		FreeAccessStrategy(strategy);
	}
	if (frozen < blocks)
		ereport(ERROR,
				(errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
				 errmsg("%u of %u pages of table \"%s\" are not frozen",
						blocks - frozen, blocks, RelationGetRelationName(heap)),
				 errhint("Run VACUUM (FREEZE) on the table again, then create the index. VACUUM leaves pages "
						 "unfrozen while other sessions hold them, or until asynchronous commits are flushed.")));
	return guard;
}

/*
 * Postings of one column from its (value, ordinal) sort: ordinals grouped by
 * value, then the distinct values and where each one's postings start.
 */
static void
WritePostings(Relation index, Tuplesortstate *sort, AnnFrozenColumn *column, AnnFrozenColumnExtra *bitsets,
			  int64 bitsetMin, uint64 rows)
{
	ArrayWriter postings;
	ArrayWriter writer;
	uint64		capacity = 1024;
	uint64		nvalues = 0;
	uint64		position = 0;
	int32	   *values = palloc(capacity * sizeof(int32));
	uint32	   *starts = palloc(capacity * sizeof(uint32));
	Datum		datum;
	bool		isnull;

	tuplesort_performsort(sort);
	InitArray(&column->postings, sizeof(uint32), 0);
	WriterBegin(&postings, index, &column->postings, false);
#if PG_VERSION_NUM >= 150000
	while (tuplesort_getdatum(sort, true, false, &datum, &isnull, NULL))
#else
	while (tuplesort_getdatum(sort, true, &datum, &isnull, NULL))
#endif
	{
		uint64		packed = (uint64) DatumGetInt64(datum);
		int32		value = (int32) (uint32) (packed >> 32);
		uint32		ordinal = (uint32) packed;

		CHECK_FOR_INTERRUPTS();
		if (nvalues == 0 || value != values[nvalues - 1])
		{
			if (nvalues == capacity)
			{
				capacity *= 2;
				values = repalloc_huge(values, capacity * sizeof(int32));
				starts = repalloc_huge(starts, capacity * sizeof(uint32));
			}
			values[nvalues] = value;
			starts[nvalues++] = (uint32) position;
		}
		memcpy(WriterNext(&postings), &ordinal, sizeof(ordinal));
		position++;
	}
	WriterEnd(&postings);
	if (position != rows)
		elog(ERROR, "ann postings have " UINT64_FORMAT " of " UINT64_FORMAT " rows", position, rows);

	InitArray(&column->dict, sizeof(int32), 0);
	WriterBegin(&writer, index, &column->dict, false);
	for (uint64 i = 0; i < nvalues; i++)
		memcpy(WriterNext(&writer), &values[i], sizeof(int32));
	WriterEnd(&writer);

	InitArray(&column->offsets, sizeof(uint32), 0);
	WriterBegin(&writer, index, &column->offsets, false);
	for (uint64 i = 0; i <= nvalues; i++)
	{
		uint32		start = i < nvalues ? starts[i] : (uint32) position;

		memcpy(WriterNext(&writer), &start, sizeof(uint32));
	}
	WriterEnd(&writer);

	/*
	 * Bitset pages for the values holding at least bitsetMin rows, from the
	 * postings just written; their positions follow.
	 */
	InitArray(&bitsets->words, sizeof(uint32), 0);
	WriterBegin(&writer, index, &bitsets->words, false);
	{
		size_t		nwords = (rows + 31) / 32;
		uint32	   *words = MemoryContextAllocHuge(CurrentMemoryContext, Max(nwords, 1) * sizeof(uint32));
		uint32	   *codes = palloc(sizeof(uint32) * Max(nvalues, 1));
		uint32	   *chunk = palloc(sizeof(uint32) * 8192);
		uint32		ncodes = 0;
		AnnArrayReader reader;

		AnnArrayReaderInit(&reader, index, &column->postings, NULL);
		for (uint64 v = 0; v < nvalues; v++)
		{
			uint64		first = starts[v];
			uint64		end = v + 1 < nvalues ? starts[v + 1] : position;

			if (end - first < (uint64) bitsetMin)
				continue;
			memset(words, 0, nwords * sizeof(uint32));
			for (uint64 at = first; at < end; at += 8192)
			{
				uint64		n = Min(end - at, 8192);

				CHECK_FOR_INTERRUPTS();
				AnnArrayReadRange(&reader, at, n, chunk);
				for (uint64 i = 0; i < n; i++)
					words[chunk[i] / 32] |= (uint32) 1 << (chunk[i] % 32);
			}
			for (size_t w = 0; w < nwords; w++)
				memcpy(WriterNext(&writer), &words[w], sizeof(uint32));
			codes[ncodes++] = (uint32) v;
		}
		AnnArrayReaderEnd(&reader);
		WriterEnd(&writer);
		InitArray(&bitsets->codes, sizeof(uint32), 0);
		WriterBegin(&writer, index, &bitsets->codes, false);
		for (uint32 i = 0; i < ncodes; i++)
			memcpy(WriterNext(&writer), &codes[i], sizeof(uint32));
		WriterEnd(&writer);
		pfree(words);
		pfree(codes);
		pfree(chunk);
	}
	pfree(values);
	pfree(starts);
}

/*
 * A text column from its (string, ordinal) sort in the column's collation:
 * distinct strings get codes 0, 1, ... in that order (NULL stays
 * ANN_FROZEN_NULL), so ranges of strings are ranges of codes. Writes the
 * strings and their offsets, fills the reserved value pages with each
 * row's code, and feeds the codes to the column's postings sort.
 */
static void
WriteDictionary(Relation index, TupleDesc textDesc, Tuplesortstate *textSort, Tuplesortstate *sort,
				AnnFrozenColumn *column, AnnFrozenColumnExtra *extra, uint64 rows)
{
	TupleTableSlot *slot = MakeSingleTupleTableSlot(textDesc, &TTSOpsMinimalTuple);
	int32	   *codes = MemoryContextAllocHuge(CurrentMemoryContext, Max(rows, 1) * sizeof(int32));
	uint64		capacity = 1024;
	uint32	   *starts = palloc(capacity * sizeof(uint32));
	uint64		nstrings = 0;
	uint64		bytes = 0;
	text	   *previous = NULL;
	ArrayWriter writer;

	tuplesort_performsort(textSort);
	InitArray(&extra->strings, 1, 0);
	WriterBegin(&writer, index, &extra->strings, false);
	while (tuplesort_gettupleslot(textSort, true, true, slot, NULL))
	{
		bool		isnull;
		Datum		value = slot_getattr(slot, 1, &isnull);
		uint64		ordinal = (uint64) DatumGetInt64(slot_getattr(slot, 2, &isnull));
		text	   *string;
		int			length;

		CHECK_FOR_INTERRUPTS();
		if (slot_attisnull(slot, 1))
		{
			codes[ordinal] = ANN_FROZEN_NULL;
			continue;
		}
		string = DatumGetTextPP(value);
		length = VARSIZE_ANY_EXHDR(string);
		if (previous == NULL || VARSIZE_ANY_EXHDR(previous) != length ||
			memcmp(VARDATA_ANY(previous), VARDATA_ANY(string), length) != 0)
		{
			if (nstrings + 1 >= capacity)
			{
				capacity *= 2;
				starts = repalloc_huge(starts, capacity * sizeof(uint32));
			}
			if (bytes + length > PG_UINT32_MAX)
				ereport(ERROR,
						(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
						 errmsg("frozen ann index text dictionaries hold at most 4 GB")));
			starts[nstrings++] = (uint32) bytes;
			for (int i = 0; i < length; i++)
				*WriterNext(&writer) = VARDATA_ANY(string)[i];
			bytes += length;
			if (previous)
				pfree(previous);
			previous = (text *) PG_DETOAST_DATUM_COPY(PointerGetDatum(string));
		}
		codes[ordinal] = (int32) (nstrings - 1);
	}
	WriterEnd(&writer);
	starts[nstrings] = (uint32) bytes;
	InitArray(&extra->starts, sizeof(uint32), 0);
	WriterBegin(&writer, index, &extra->starts, false);
	for (uint64 i = 0; i <= nstrings; i++)
		memcpy(WriterNext(&writer), &starts[i], sizeof(uint32));
	WriterEnd(&writer);

	/* Codes per row into the reserved value pages, and to the postings. */
	WriterBegin(&writer, index, &column->values, true);
	for (uint64 i = 0; i < rows; i++)
	{
		memcpy(WriterNext(&writer), &codes[i], sizeof(int32));
		tuplesort_putdatum(sort, Int64GetDatum((int64) codes[i] * INT64CONST(4294967296) + (int64) i), false);
	}
	WriterEnd(&writer);
	ExecDropSingleTupleTableSlot(slot);
	pfree(codes);
	pfree(starts);
}

/*
 * Lay out the sorted rows: TIDs, keys and attribute values fill reserved
 * pages side by side while vectors are appended after them; each attribute's
 * inverted index follows. The generation hashes every key, vector and value
 * in ordinal order.
 */
static void
WriteArrays(Relation index, FrozenBuildState *bs, AnnFrozenMetaData *meta)
{
	TupleTableSlot *slot = MakeSingleTupleTableSlot(bs->sortDesc, &TTSOpsMinimalTuple);
	ArrayWriter tids;
	ArrayWriter keys;
	ArrayWriter vectors;
	ArrayWriter *values = palloc0(sizeof(ArrayWriter) * Max(bs->ncolumns, 1));
	Tuplesortstate **sorts = palloc0(sizeof(Tuplesortstate *) * Max(bs->ncolumns, 1));
	Tuplesortstate **textSorts = palloc0(sizeof(Tuplesortstate *) * Max(bs->ncolumns, 1));
	TupleDesc	textDesc;
	TupleTableSlot *textSlot;
	int			workMem = Max(maintenance_work_mem / Max(bs->ncolumns, 1), 1024);
	uint32		header[2] = {bs->dims, bs->ncolumns};
	uint64		hash = hash_bytes_extended((const unsigned char *) header, sizeof(header), ANN_FROZEN_SEED);
	uint64		ordinal = 0;
	int64		previous = 0;

	InitArray(&meta->tids, sizeof(ItemPointerData), bs->rows);
	InitArray(&meta->keys, sizeof(int64), bs->rows);
	ReserveArray(index, &meta->tids);
	ReserveArray(index, &meta->keys);
	textDesc = CreateTemplateTupleDesc(2);
	TupleDescInitEntry(textDesc, 1, "value", TEXTOID, -1, 0);
	TupleDescInitEntry(textDesc, 2, "ordinal", INT8OID, -1, 0);
#if PG_VERSION_NUM >= 190000
	TupleDescFinalize(textDesc);
#endif
	textSlot = MakeSingleTupleTableSlot(textDesc, &TTSOpsVirtual);
	for (int c = 0; c < bs->ncolumns; c++)
	{
		InitArray(&meta->columns[c].values, sizeof(int32), bs->rows);
		ReserveArray(index, &meta->columns[c].values);
		if (!bs->text[c])
			WriterBegin(&values[c], index, &meta->columns[c].values, true);
		else
		{
			/* (string, ordinal) in the column's collation, NULLs first */
			AttrNumber	attNums[2] = {1, 2};
			Oid			operators[2] = {get_opfamily_member(TEXT_BTREE_FAM_OID, TEXTOID, TEXTOID,
															BTLessStrategyNumber), Int8LessOperator};
			Oid			collations[2] = {bs->collations[c], InvalidOid};
			bool		nullsFirst[2] = {true, false};

			textSorts[c] = tuplesort_begin_heap(textDesc, 2, attNums, operators, collations, nullsFirst,
												workMem, NULL, false);
		}
		sorts[c] = tuplesort_begin_datum(INT8OID, Int8LessOperator, InvalidOid, false,
										 workMem, NULL, false);
	}
	WriterBegin(&tids, index, &meta->tids, true);
	WriterBegin(&keys, index, &meta->keys, true);
	InitArray(&meta->vectors, VECTOR_SIZE(bs->dims), 0);
	WriterBegin(&vectors, index, &meta->vectors, false);

	while (tuplesort_gettupleslot(bs->sort, true, false, slot, NULL))
	{
		MemoryContext oldCtx = MemoryContextSwitchTo(bs->tmpCtx);
		bool		isnull;
		int64		key = DatumGetInt64(slot_getattr(slot, 1, &isnull));
		int64		tid = DatumGetInt64(slot_getattr(slot, 2, &isnull));
		Vector	   *vector = DatumGetVector(slot_getattr(slot, 3, &isnull));
		int32		row[ANN_FROZEN_MAX_COLUMNS];
		ItemPointerData item;

		CHECK_FOR_INTERRUPTS();
		if (ordinal > 0 && key <= previous)
			ereport(ERROR,
					(errcode(ERRCODE_UNIQUE_VIOLATION),
					 errmsg("key " INT64_FORMAT " appears more than once", key),
					 errdetail("The key of a frozen ann index must be unique.")));
		ItemPointerSet(&item, (BlockNumber) (tid >> 16), (OffsetNumber) (tid & 0xFFFF));
		memcpy(WriterNext(&tids), &item, sizeof(item));
		memcpy(WriterNext(&keys), &key, sizeof(key));
		memcpy(WriterNext(&vectors), vector, VECTOR_SIZE(bs->dims));
		hash = hash_bytes_extended((const unsigned char *) &key, sizeof(key), hash);
		hash = hash_bytes_extended((const unsigned char *) vector->x, sizeof(float) * bs->dims, hash);
		for (int c = 0; c < bs->ncolumns; c++)
		{
			Datum		value = slot_getattr(slot, 4 + c, &isnull);

			if (bs->text[c])
			{
				/* Hash the string; it gets its code from the collation sort. */
				ExecClearTuple(textSlot);
				textSlot->tts_values[0] = value;
				textSlot->tts_isnull[0] = isnull;
				textSlot->tts_values[1] = Int64GetDatum((int64) ordinal);
				textSlot->tts_isnull[1] = false;
				ExecStoreVirtualTuple(textSlot);
				tuplesort_puttupleslot(textSorts[c], textSlot);
				row[c] = isnull ? ANN_FROZEN_NULL : 0;
				if (!isnull)
				{
					text	   *string = DatumGetTextPP(value);

					hash = hash_bytes_extended((const unsigned char *) VARDATA_ANY(string),
											   VARSIZE_ANY_EXHDR(string), hash);
				}
				continue;
			}
			row[c] = DatumGetInt32(value);
			memcpy(WriterNext(&values[c]), &row[c], sizeof(int32));
			/* value * 2^32 + ordinal sorts by value, then ordinal */
			tuplesort_putdatum(sorts[c], Int64GetDatum((int64) row[c] * INT64CONST(4294967296) + (int64) ordinal),
							   false);
		}
		if (bs->ncolumns > 0)
			hash = hash_bytes_extended((const unsigned char *) row, sizeof(int32) * bs->ncolumns, hash);
		previous = key;
		ordinal++;
		MemoryContextSwitchTo(oldCtx);
		MemoryContextReset(bs->tmpCtx);
	}
	WriterEnd(&tids);
	WriterEnd(&keys);
	WriterEnd(&vectors);
	for (int c = 0; c < bs->ncolumns; c++)
		if (!bs->text[c])
			WriterEnd(&values[c]);
	ExecDropSingleTupleTableSlot(slot);
	for (int c = 0; c < bs->ncolumns; c++)
		if (bs->text[c])
		{
			WriteDictionary(index, textDesc, textSorts[c], sorts[c], &meta->columns[c],
							&AnnFrozenGetExtra(meta)[c], ordinal);
			tuplesort_end(textSorts[c]);
		}
	ExecDropSingleTupleTableSlot(textSlot);
	meta->rows = ordinal;
	meta->generation = hash_bytes_extended((const unsigned char *) &ordinal, sizeof(ordinal), hash);
	for (int c = 0; c < bs->ncolumns; c++)
	{
		WritePostings(index, sorts[c], &meta->columns[c], &AnnFrozenGetExtra(meta)[c],
					  bs->bitsetMin < 0 ? Max(ordinal / 32, 1) : Max(bs->bitsetMin, 1), ordinal);
		tuplesort_end(sorts[c]);
	}
}

/* Rewrite block 0 as a frozen metapage; pd_lower covers the directory. */
static void
WriteFrozenMeta(Relation index, const AnnFrozenMetaData *frozen)
{
	Buffer		buf = ReadBuffer(index, ANN_METAPAGE_BLKNO);
	Page		page;
	AnnFrozenMetaData *stored;

	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	page = BufferGetPage(buf);
	AnnCheckMetaPage(page);
	AnnPageGetMeta(page)->version = ANN_FROZEN_VERSION;
	AnnPageGetMeta(page)->insertPage = ANN_FROZEN_FLUSHED;
	stored = AnnPageGetFrozen(page);
	memcpy(stored, frozen, AnnFrozenMetaSize(frozen->ncolumns));
	((PageHeader) page)->pd_lower = ((char *) stored + AnnFrozenMetaSize(frozen->ncolumns)) - (char *) page;
	MarkBufferDirty(buf);
	UnlockReleaseBuffer(buf);
}

IndexBuildResult *
AnnFrozenBuild(Relation heap, Relation index, IndexInfo *indexInfo)
{
	FrozenBuildState bs = {0};
	IndexBuildResult *result = palloc0_object(IndexBuildResult);
	TupleDesc	indexDesc = RelationGetDescr(index);
	AttrNumber	attNums[] = {1};
	Oid			operators[] = {Int8LessOperator};
	Oid			collations[] = {InvalidOid};
	bool		nullsFirst[] = {false};
	AnnFrozenMetaData *meta;
	Oid			guard;

	AnnCheckIndex(index);
	guard = CheckFrozenTable(heap, index, indexInfo);
	buildingIndex = RelationGetRelid(index);
	AnnInitMetaPage(index, MAIN_FORKNUM);
	bs.dims = TupleDescAttr(indexDesc, 0)->atttypmod;
	bs.bitsetMin = index->rd_options ? ((AnnOptions *) index->rd_options)->bitsetMin : -1;
	bs.ncolumns = indexDesc->natts - 2;
	bs.keyType = TupleDescAttr(indexDesc, 1)->atttypid;
	bs.sortDesc = CreateTemplateTupleDesc(3 + bs.ncolumns);
	TupleDescInitEntry(bs.sortDesc, 1, "key", INT8OID, -1, 0);
	TupleDescInitEntry(bs.sortDesc, 2, "tid", INT8OID, -1, 0);
	TupleDescInitEntry(bs.sortDesc, 3, "vector", AnnVectorTypeOid(), -1, 0);
	for (int c = 0; c < bs.ncolumns; c++)
	{
		bs.text[c] = AnnFrozenTextType(TupleDescAttr(indexDesc, 2 + c)->atttypid);
		bs.collations[c] = AnnFrozenColumnCollation(index, 2 + c);
		TupleDescInitEntry(bs.sortDesc, 4 + c, "value", bs.text[c] ? TEXTOID : INT4OID, -1, 0);
	}
#if PG_VERSION_NUM >= 190000
	TupleDescFinalize(bs.sortDesc);
#endif
	bs.slot = MakeSingleTupleTableSlot(bs.sortDesc, &TTSOpsVirtual);
	bs.sort = tuplesort_begin_heap(bs.sortDesc, 1, attNums, operators, collations, nullsFirst,
								   maintenance_work_mem, NULL, false);
	bs.tmpCtx = AllocSetContextCreate(CurrentMemoryContext, "Ann frozen build temporary context",
									  ALLOCSET_DEFAULT_SIZES);
	result->heap_tuples = table_index_build_scan(heap, index, indexInfo, true, true,
												 FrozenBuildCallback, &bs, NULL);
	if (bs.rows >= PG_UINT32_MAX)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("frozen ann indexes hold at most %u rows", PG_UINT32_MAX - 1)));
	tuplesort_performsort(bs.sort);

	meta = palloc0(AnnFrozenMetaSize(ANN_FROZEN_MAX_COLUMNS));
	meta->guard = guard;
	meta->ncolumns = bs.ncolumns;
	WriteArrays(index, &bs, meta);
	tuplesort_end(bs.sort);
	ExecDropSingleTupleTableSlot(bs.slot);
	WriteFrozenMeta(index, meta);
	result->index_tuples = meta->rows;
	if (RelationNeedsWAL(index))
		log_newpage_range(index, MAIN_FORKNUM, 0, RelationGetNumberOfBlocks(index), true);
	/* The files now hold the final pages: backends may map them. */
	FlushRelationBuffers(index);
	buildingIndex = InvalidOid;
	if (index->rd_amcache)
	{
		pfree(index->rd_amcache);
		index->rd_amcache = NULL;
	}
	MemoryContextDelete(bs.tmpCtx);
	return result;
}

/* Frozen pages as an ann/filter.h source. */
typedef struct FrozenSource
{
	Relation	index;
	const AnnFrozenMetaData *meta;
	const AnnFrozenColumnExtra *bitsets;
	const char *names[ANN_FROZEN_MAX_COLUMNS];
	AnnArrayReader dict[ANN_FROZEN_MAX_COLUMNS];
	AnnArrayReader offsets[ANN_FROZEN_MAX_COLUMNS];
	AnnArrayReader postings[ANN_FROZEN_MAX_COLUMNS];
	AnnArrayReader codes[ANN_FROZEN_MAX_COLUMNS];
	AnnArrayReader words[ANN_FROZEN_MAX_COLUMNS];
} FrozenSource;

static int
SourceColumn(void *arg, const char *name)
{
	FrozenSource *source = arg;

	for (uint32 c = 0; c < source->meta->ncolumns; c++)
		if (strcmp(source->names[c], name) == 0)
			return (int) c;
	return -1;
}

static uint64_t
SourceCount(void *arg, int column)
{
	return ((FrozenSource *) arg)->meta->columns[column].dict.count;
}

static int32_t
SourceValue(void *arg, int column, uint64_t i)
{
	int32		value;

	AnnArrayRead(&((FrozenSource *) arg)->dict[column], i, &value);
	return value;
}

static uint64_t
SourceStart(void *arg, int column, uint64_t i)
{
	uint32		start;

	AnnArrayRead(&((FrozenSource *) arg)->offsets[column], i, &start);
	return start;
}

static int
SourcePostings(void *arg, int column, uint64_t first, uint64_t n, uint32_t *out)
{
	CHECK_FOR_INTERRUPTS();
	AnnArrayReadRange(&((FrozenSource *) arg)->postings[column], first, n, out);
	return 1;
}

static int
SourceBitset(void *arg, int column, uint64_t i, uint32_t *words)
{
	FrozenSource *source = arg;
	const AnnFrozenColumnExtra *bitsets = &source->bitsets[column];
	uint64		nwords = (source->meta->rows + 31) / 32;
	int64		low = 0;
	int64		high = (int64) bitsets->codes.count - 1;

	while (low <= high)
	{
		int64		middle = low + (high - low) / 2;
		uint32		code;

		AnnArrayRead(&source->codes[column], middle, &code);
		if (code == i)
		{
			AnnArrayReadRange(&source->words[column], (uint64) middle * nwords, nwords, words);
			return 1;
		}
		if (code < i)
			low = middle + 1;
		else
			high = middle - 1;
	}
	return 0;
}

static void *
SourceAlloc(void *arg, size_t size)
{
	return MemoryContextAllocHuge(CurrentMemoryContext, size);
}

static void
SourceRelease(void *arg, void *pointer)
{
	pfree(pointer);
}

/*
 * Evaluate a predicate program over the frozen pages: the same evaluator,
 * over the same ordinals, as the worker. Returns an ANN_FILTER_* code.
 */
int
AnnFrozenEvaluate(Relation index, const AnnFrozenMetaData *meta, const unsigned char *program,
				  size_t size, uint32 *words, uint64 *count)
{
	FrozenSource *state = palloc0(sizeof(FrozenSource));
	AnnFilterSource source;
	int			code;

	state->index = index;
	state->meta = meta;
	state->bitsets = AnnFrozenGetExtra(meta);
	for (uint32 c = 0; c < meta->ncolumns; c++)
	{
		state->names[c] = AnnFrozenColumnName(index, 3 + c);
		AnnArrayReaderInit(&state->dict[c], index, &meta->columns[c].dict, NULL);
		AnnArrayReaderInit(&state->offsets[c], index, &meta->columns[c].offsets, NULL);
		AnnArrayReaderInit(&state->postings[c], index, &meta->columns[c].postings, NULL);
		AnnArrayReaderInit(&state->codes[c], index, &state->bitsets[c].codes, NULL);
		AnnArrayReaderInit(&state->words[c], index, &state->bitsets[c].words, NULL);
	}
	memset(&source, 0, sizeof(source));
	source.arg = state;
	source.rows = meta->rows;
	source.column = SourceColumn;
	source.count = SourceCount;
	source.value = SourceValue;
	source.start = SourceStart;
	source.postings = SourcePostings;
	source.bitset = SourceBitset;
	source.alloc = SourceAlloc;
	source.release = SourceRelease;
	code = AnnFilterEvaluate(&source, program, size, words, count);
	for (uint32 c = 0; c < meta->ncolumns; c++)
	{
		AnnArrayReaderEnd(&state->dict[c]);
		AnnArrayReaderEnd(&state->offsets[c]);
		AnnArrayReaderEnd(&state->postings[c]);
		AnnArrayReaderEnd(&state->codes[c]);
		AnnArrayReaderEnd(&state->words[c]);
	}
	pfree(state);
	return code;
}

void
AnnFrozenEvaluateOrError(Relation index, const AnnFrozenMetaData *meta, const char *program,
						 size_t size, uint32 *words, uint64 *count)
{
	int			code = AnnFrozenEvaluate(index, meta, (const unsigned char *) program, size, words, count);

	if (code == ANN_FILTER_UNKNOWN_COLUMN)
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_COLUMN),
				 errmsg("predicate names a column that is not an INCLUDE attribute of \"%s\"",
						RelationGetRelationName(index))));
	if (code == ANN_FILTER_UNKNOWN_BITMAP)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("stored filter ids in predicates are evaluated on GPU nodes only")));
	if (code != ANN_FILTER_OK)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid ann predicate program (code %d)", code)));
}

/* Ordinal of a key (binary search over the key pages), or -1. */
int64
AnnFrozenKeyOrdinal(Relation index, const AnnFrozenMetaData *meta, int64 key)
{
	AnnArrayReader reader;
	int64		low = 0;
	int64		high = (int64) meta->keys.count - 1;
	int64		found = -1;

	AnnArrayReaderInit(&reader, index, &meta->keys, NULL);
	while (low <= high)
	{
		int64		middle = low + (high - low) / 2;
		int64		value;

		AnnArrayRead(&reader, middle, &value);
		if (value == key)
		{
			found = middle;
			break;
		}
		if (value < key)
			low = middle + 1;
		else
			high = middle - 1;
	}
	AnnArrayReaderEnd(&reader);
	return found;
}

/*
 * The table's valid frozen ann index, opened, with this generation (any
 * generation when 0); NULL if there is none.
 */
Relation
AnnFrozenIndexOf(Oid relid, uint64 generation, LOCKMODE lockmode)
{
	Oid			am = get_am_oid("ann", false);
	Relation	heap = table_open(relid, AccessShareLock);
	List	   *indexes = RelationGetIndexList(heap);
	ListCell   *lc;
	Relation	result = NULL;

	foreach(lc, indexes)
	{
		Relation	index = index_open(lfirst_oid(lc), lockmode);

		if (index->rd_rel->relam == am && index->rd_index->indisvalid)
		{
			AnnFrozenMetaData *meta = AnnFrozenGetMeta(index, true);

			if (meta && (generation == 0 || meta->generation == generation))
			{
				result = index;
				break;
			}
		}
		index_close(index, lockmode);
	}
	table_close(heap, AccessShareLock);
	return result;
}

/*
 * SQL inspection of a frozen index requires SELECT on its table, and refuses
 * tables with row-level security, since rows come back without the policy.
 */
static Relation
OpenFrozenIndex(Oid relid)
{
	Relation	index = index_open(relid, AccessShareLock);
	Oid			heapOid = index->rd_index->indrelid;
	AclResult	aclresult;

	if (index->rd_rel->relam != get_am_oid("ann", false))
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("\"%s\" is not an ann index", RelationGetRelationName(index))));
	aclresult = pg_class_aclcheck(heapOid, GetUserId(), ACL_SELECT);
	if (aclresult != ACLCHECK_OK)
		aclcheck_error(aclresult, OBJECT_TABLE, get_rel_name(heapOid));
	if (check_enable_rls(heapOid, InvalidOid, false) == RLS_ENABLED)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("frozen ann index functions do not support row-level security")));
	return index;
}

/*
 * ann_frozen_info(index): generation, rows, dimensions, key column, attribute
 * columns with their distinct values, and pages.
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(ann_frozen_info);
Datum
ann_frozen_info(PG_FUNCTION_ARGS)
{
	Relation	index = OpenFrozenIndex(PG_GETARG_OID(0));
	AnnFrozenMetaData *meta = AnnFrozenGetMeta(index, false);
	TupleDesc	tupdesc;
	Datum		values[7];
	bool		nulls[7] = {0};
	Datum	   *names = palloc(sizeof(Datum) * Max(meta->ncolumns, 1));
	Datum	   *distinct = palloc(sizeof(Datum) * Max(meta->ncolumns, 1));
	HeapTuple	tuple;

	if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
		elog(ERROR, "return type must be a row type");
	for (uint32 c = 0; c < meta->ncolumns; c++)
	{
		names[c] = CStringGetTextDatum(AnnFrozenColumnName(index, 3 + c));
		distinct[c] = Int64GetDatum((int64) meta->columns[c].dict.count);
	}
	values[0] = CStringGetTextDatum(psprintf("%016llx", (unsigned long long) meta->generation));
	values[1] = Int64GetDatum((int64) meta->rows);
	values[2] = Int32GetDatum(TupleDescAttr(RelationGetDescr(index), 0)->atttypmod);
	values[3] = CStringGetTextDatum(AnnFrozenColumnName(index, 2));
	values[4] = PointerGetDatum(construct_array(names, meta->ncolumns, TEXTOID, -1, false, TYPALIGN_INT));
	values[5] = PointerGetDatum(construct_array(distinct, meta->ncolumns, INT8OID, sizeof(int64),
												FLOAT8PASSBYVAL, TYPALIGN_DOUBLE));
	values[6] = Int64GetDatum((int64) RelationGetNumberOfBlocks(index));
	tuple = heap_form_tuple(tupdesc, values, nulls);
	index_close(index, AccessShareLock);
	PG_RETURN_DATUM(HeapTupleGetDatum(tuple));
}

static int
Int32Compare(const void *a, const void *b)
{
	int32		x = *(const int32 *) a;
	int32		y = *(const int32 *) b;

	return (x > y) - (x < y);
}

static int
Uint32Compare(const void *a, const void *b)
{
	uint32		x = *(const uint32 *) a;
	uint32		y = *(const uint32 *) b;

	return (x > y) - (x < y);
}

/* Position of value in the column's distinct values, or -1. */
static int64
FindValue(AnnArrayReader *dict, int32 value)
{
	int64		low = 0;
	int64		high = (int64) dict->array->count - 1;

	while (low <= high)
	{
		int64		middle = low + (high - low) / 2;
		int32		found;

		AnnArrayRead(dict, middle, &found);
		if (found == value)
			return middle;
		if (found < value)
			low = middle + 1;
		else
			high = middle - 1;
	}
	return -1;
}

/*
 * ann_frozen_keys(index, column, values): keys of the rows whose column holds
 * one of the values, in key order. The inverted index (distinct values,
 * postings) gives ordinals; the forward index gives their keys.
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(ann_frozen_keys);
Datum
ann_frozen_keys(PG_FUNCTION_ARGS)
{
	Relation	index = OpenFrozenIndex(PG_GETARG_OID(0));
	char	   *name = text_to_cstring(PG_GETARG_TEXT_PP(1));
	ArrayType  *array = PG_GETARG_ARRAYTYPE_P(2);
	ReturnSetInfo *rsinfo = (ReturnSetInfo *) fcinfo->resultinfo;
	AnnFrozenMetaData *meta = AnnFrozenGetMeta(index, false);
	AnnFrozenColumn *column = NULL;
	AnnArrayReader dict;
	AnnArrayReader offsets;
	AnnArrayReader postings;
	AnnArrayReader keys;
	Datum	   *elements;
	bool	   *elementNulls;
	int			nelements;
	int32	   *wanted;
	int			nwanted = 0;
	uint32	   *ordinals = NULL;
	uint64		nordinals = 0;
	TupleDesc	tupdesc;
	Tuplestorestate *store;
	MemoryContext oldCtx;

	if (rsinfo == NULL || !IsA(rsinfo, ReturnSetInfo) || !(rsinfo->allowedModes & SFRM_Materialize))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("set-valued function called in context that cannot accept a set")));
	for (uint32 c = 0; c < meta->ncolumns && column == NULL; c++)
		if (strcmp(AnnFrozenColumnName(index, 3 + c), name) == 0)
			column = &meta->columns[c];
	if (column == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_COLUMN),
				 errmsg("frozen ann index \"%s\" has no INCLUDE column \"%s\" after its key",
						RelationGetRelationName(index), name)));
	if (ARR_NDIM(array) > 1)
		ereport(ERROR, (errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR), errmsg("array must be one-dimensional")));
	deconstruct_array(array, INT4OID, sizeof(int32), true, TYPALIGN_INT, &elements, &elementNulls, &nelements);
	wanted = palloc(sizeof(int32) * Max(nelements, 1));
	for (int i = 0; i < nelements; i++)
		if (!elementNulls[i] && DatumGetInt32(elements[i]) != ANN_FROZEN_NULL)
			wanted[nwanted++] = DatumGetInt32(elements[i]);
	if (nwanted > 1)
	{
		int			unique = 1;

		qsort(wanted, nwanted, sizeof(int32), Int32Compare);
		for (int i = 1; i < nwanted; i++)
			if (wanted[i] != wanted[unique - 1])
				wanted[unique++] = wanted[i];
		nwanted = unique;
	}

	AnnArrayReaderInit(&dict, index, &column->dict, NULL);
	AnnArrayReaderInit(&offsets, index, &column->offsets, NULL);
	AnnArrayReaderInit(&postings, index, &column->postings, NULL);
	AnnArrayReaderInit(&keys, index, &meta->keys, NULL);
	for (int pass = 0; pass < 2; pass++)
	{
		uint64		filled = 0;

		/* First count the postings, then copy them. */
		for (int i = 0; i < nwanted; i++)
		{
			int64		code = FindValue(&dict, wanted[i]);
			uint32		range[2];

			if (code < 0)
				continue;
			AnnArrayReadRange(&offsets, code, 2, range);
			if (pass == 1)
				AnnArrayReadRange(&postings, range[0], range[1] - range[0], ordinals + filled);
			filled += range[1] - range[0];
		}
		if (pass == 0)
		{
			nordinals = filled;
			ordinals = MemoryContextAllocHuge(CurrentMemoryContext, sizeof(uint32) * Max(nordinals, 1));
		}
	}
	if (nwanted > 1)
		qsort(ordinals, nordinals, sizeof(uint32), Uint32Compare);

	oldCtx = MemoryContextSwitchTo(rsinfo->econtext->ecxt_per_query_memory);
	tupdesc = CreateTemplateTupleDesc(1);
	TupleDescInitEntry(tupdesc, 1, "id", INT8OID, -1, 0);
#if PG_VERSION_NUM >= 190000
	TupleDescFinalize(tupdesc);
#endif
	store = tuplestore_begin_heap(true, false, work_mem);
	MemoryContextSwitchTo(oldCtx);
	for (uint64 i = 0; i < nordinals; i++)
	{
		int64		key;
		Datum		value;
		bool		isnull = false;

		CHECK_FOR_INTERRUPTS();
		AnnArrayRead(&keys, ordinals[i], &key);
		value = Int64GetDatum(key);
		tuplestore_putvalues(store, tupdesc, &value, &isnull);
	}
	AnnArrayReaderEnd(&dict);
	AnnArrayReaderEnd(&offsets);
	AnnArrayReaderEnd(&postings);
	AnnArrayReaderEnd(&keys);
	index_close(index, AccessShareLock);
	rsinfo->returnMode = SFRM_Materialize;
	rsinfo->setResult = store;
	rsinfo->setDesc = tupdesc;
	return (Datum) 0;
}

/*
 * ann_frozen_filter(index, filter): rows of a filter over the index's
 * INCLUDE attributes, evaluated from its postings and bitset pages, as
 * (eligible rows, bitmap over ordinals).
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(ann_frozen_filter);
Datum
ann_frozen_filter(PG_FUNCTION_ARGS)
{
	Relation	index = OpenFrozenIndex(PG_GETARG_OID(0));
	char	   *filter = text_to_cstring(PG_GETARG_TEXT_PP(1));
	AnnFrozenMetaData *meta = AnnFrozenGetMeta(index, false);
	AnnPredicateColumns columns;
	StringInfoData program;
	TupleDesc	tupdesc;
	Datum		values[2];
	bool		nulls[2] = {false, false};
	size_t		nwords = (meta->rows + 31) / 32;
	bytea	   *bitmap = palloc(VARHDRSZ + nwords * 4);
	uint64		eligible;

	if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
		elog(ERROR, "return type must be a row type");
	AnnPredicateColumnsInit(index, &columns);
	initStringInfo(&program);
	if (!AnnPredicateFromText(index->rd_index->indrelid, filter, &columns, &program))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("filter \"%s\" is not a predicate over the INCLUDE attributes of \"%s\"",
						filter, RelationGetRelationName(index))));
	SET_VARSIZE(bitmap, VARHDRSZ + nwords * 4);
	AnnFrozenEvaluateOrError(index, meta, program.data, program.len, (uint32 *) VARDATA(bitmap), &eligible);
	index_close(index, AccessShareLock);
	values[0] = Int64GetDatum((int64) eligible);
	values[1] = PointerGetDatum(bitmap);
	PG_RETURN_DATUM(HeapTupleGetDatum(heap_form_tuple(tupdesc, values, nulls)));
}
