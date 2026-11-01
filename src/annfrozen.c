#include "postgres.h"

#include "access/table.h"
#include "access/tableam.h"
#include "access/visibilitymap.h"
#include "access/xloginsert.h"
#include "ann.h"
#include "catalog/pg_operator_d.h"
#include "catalog/pg_trigger.h"
#include "catalog/pg_type_d.h"
#include "commands/defrem.h"
#include "commands/trigger.h"
#include "common/hashfn.h"
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

void
AnnArrayReaderInit(AnnArrayReader *reader, Relation index, const AnnArray *array,
				   BufferAccessStrategy strategy)
{
	reader->index = index;
	reader->array = array;
	reader->strategy = strategy;
	reader->buf = InvalidBuffer;
	reader->page = InvalidBlockNumber;
	reader->locked = false;
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
static void
ReadRange(AnnArrayReader *reader, uint64 first, uint64 n, void *out)
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
	ReadRange(reader, element, 1, out);
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
	Buffer		buf = ReadBuffer(index, ANN_METAPAGE_BLKNO);
	bool		frozen;

	LockBuffer(buf, BUFFER_LOCK_SHARE);
	AnnCheckMetaPage(BufferGetPage(buf));
	frozen = AnnPageGetMeta(BufferGetPage(buf))->version == ANN_FROZEN_VERSION;
	UnlockReleaseBuffer(buf);
	return frozen;
}

/* A copy of the frozen metadata; NULL for a mutable index when missingOk. */
AnnFrozenMetaData *
AnnFrozenGetMeta(Relation index, bool missingOk)
{
	Buffer		buf = ReadBuffer(index, ANN_METAPAGE_BLKNO);
	Page		page;
	AnnFrozenMetaData *meta = NULL;

	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	AnnCheckMetaPage(page);
	if (AnnPageGetMeta(page)->version == ANN_FROZEN_VERSION)
	{
		AnnFrozenMetaData *stored = AnnPageGetFrozen(page);

		if (stored->ncolumns > ANN_FROZEN_MAX_COLUMNS)
			ereport(ERROR, (errcode(ERRCODE_INDEX_CORRUPTED), errmsg("invalid frozen ann index metapage")));
		meta = palloc(AnnFrozenMetaSize(ANN_FROZEN_MAX_COLUMNS));
		memcpy(meta, stored, AnnFrozenMetaSize(stored->ncolumns));
	}
	UnlockReleaseBuffer(buf);
	if (meta == NULL && !missingOk)
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("\"%s\" is not a frozen ann index", RelationGetRelationName(index)),
				 errhint("Create it WITH (frozen = true).")));
	return meta;
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
	Oid			keyType;
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

static int32
AttributeValue(Datum value, bool isnull, Oid type)
{
	int32		result;

	if (isnull)
		return ANN_FROZEN_NULL;
	result = type == INT2OID ? DatumGetInt16(value) : DatumGetInt32(value);
	if (result == ANN_FROZEN_NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NUMERIC_VALUE_OUT_OF_RANGE),
				 errmsg("frozen ann indexes reserve the value %d for NULL", ANN_FROZEN_NULL)));
	return result;
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
	for (int c = 0; c < bs->ncolumns; c++)
		slot->tts_values[3 + c] = Int32GetDatum(AttributeValue(values[2 + c], isnull[2 + c],
															   TupleDescAttr(RelationGetDescr(index), 2 + c)->atttypid));
	memset(slot->tts_isnull, 0, sizeof(bool) * (3 + bs->ncolumns));
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
		if (type != INT2OID && type != INT4OID)
			ereport(ERROR,
					(errcode(ERRCODE_DATATYPE_MISMATCH),
					 errmsg("INCLUDE column \"%s\" of a frozen ann index must be smallint or integer",
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
	if (frozen < blocks)
		ereport(ERROR,
				(errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
				 errmsg("%u of %u pages of table \"%s\" are not frozen",
						blocks - frozen, blocks, RelationGetRelationName(heap)),
				 errhint("Run VACUUM (FREEZE) on the table, then create the index. Rows committed with "
						 "synchronous_commit = off may need a second VACUUM once their commit is flushed.")));
	return guard;
}

/*
 * Postings of one column from its (value, ordinal) sort: ordinals grouped by
 * value, then the distinct values and where each one's postings start.
 */
static void
WritePostings(Relation index, Tuplesortstate *sort, AnnFrozenColumn *column, uint64 rows)
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
	pfree(values);
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
	int			workMem = Max(maintenance_work_mem / Max(bs->ncolumns, 1), 1024);
	uint32		header[2] = {bs->dims, bs->ncolumns};
	uint64		hash = hash_bytes_extended((const unsigned char *) header, sizeof(header), ANN_FROZEN_SEED);
	uint64		ordinal = 0;
	int64		previous = 0;

	InitArray(&meta->tids, sizeof(ItemPointerData), bs->rows);
	InitArray(&meta->keys, sizeof(int64), bs->rows);
	ReserveArray(index, &meta->tids);
	ReserveArray(index, &meta->keys);
	for (int c = 0; c < bs->ncolumns; c++)
	{
		InitArray(&meta->columns[c].values, sizeof(int32), bs->rows);
		ReserveArray(index, &meta->columns[c].values);
		WriterBegin(&values[c], index, &meta->columns[c].values, true);
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
			row[c] = DatumGetInt32(slot_getattr(slot, 4 + c, &isnull));
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
		WriterEnd(&values[c]);
	ExecDropSingleTupleTableSlot(slot);
	meta->rows = ordinal;
	meta->generation = hash_bytes_extended((const unsigned char *) &ordinal, sizeof(ordinal), hash);
	for (int c = 0; c < bs->ncolumns; c++)
	{
		WritePostings(index, sorts[c], &meta->columns[c], ordinal);
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
	AnnInitMetaPage(index, MAIN_FORKNUM);
	bs.dims = TupleDescAttr(indexDesc, 0)->atttypmod;
	bs.ncolumns = indexDesc->natts - 2;
	bs.keyType = TupleDescAttr(indexDesc, 1)->atttypid;
	bs.sortDesc = CreateTemplateTupleDesc(3 + bs.ncolumns);
	TupleDescInitEntry(bs.sortDesc, 1, "key", INT8OID, -1, 0);
	TupleDescInitEntry(bs.sortDesc, 2, "tid", INT8OID, -1, 0);
	TupleDescInitEntry(bs.sortDesc, 3, "vector", AnnVectorTypeOid(), -1, 0);
	for (int c = 0; c < bs.ncolumns; c++)
		TupleDescInitEntry(bs.sortDesc, 4 + c, "value", INT4OID, -1, 0);
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
	MemoryContextDelete(bs.tmpCtx);
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
			ReadRange(&offsets, code, 2, range);
			if (pass == 1)
				ReadRange(&postings, range[0], range[1] - range[0], ordinals + filled);
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
