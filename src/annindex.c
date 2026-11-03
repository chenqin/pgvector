#include "postgres.h"

#include "access/table.h"
#include "ann.h"
#include "catalog/namespace.h"
#include "catalog/pg_type_d.h"
#include "commands/defrem.h"
#include "commands/extension.h"
#include "common/hashfn.h"
#include "executor/spi.h"
#include "fmgr.h"
#include "funcapi.h"
#include "access/htup_details.h"
#include "portability/instr_time.h"
#include "miscadmin.h"
#include "utils/acl.h"
#include "utils/rls.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "utils/lsyscache.h"
#include "utils/regproc.h"
#include "utils/rel.h"
#include "utils/tuplestore.h"
#include "vector.h"

#define ANN_OP_LOAD_INDEX 11
#define ANN_OP_INDEX_SEARCH_PACKED 14
#define ANN_OP_INDEX_INFO 15
#define ANN_OP_INDEX_SEARCH_MATCH 17
#define ANN_ATTRIBUTE_MAX_VALUES 64
#define ANN_OP_INDEX_BITMAP 18
#define ANN_OP_INDEX_SEARCH_BITMAP 19
#define ANN_OP_BITMAP_PUT 20
#define ANN_OP_INDEX_SEARCH_CACHED 21
#define ANN_BITMAP_TIMEOUT_MS (60 * 1000)
#define ANN_INDEX_NAME_MAX 63
#define ANN_INDEX_MAX_K 10000
#define ANN_LOAD_TIMEOUT_MS (30 * 60 * 1000)
#define ANN_KEY_FETCH 65536


/* What the worker reports for a loaded index. */
typedef struct AnnIndexInfo
{
	uint64		rows;
	char		source[256];
	char		key[64];
	char		vector[64];
	char		generation[17]; /* frozen index the files came from, or "-" */
} AnnIndexInfo;

/* Catalog row: where an index's vectors live on this host. */
typedef struct AnnCatalogEntry
{
	char	   *source;			/* quoted relation name */
	char	   *key;			/* quoted column names */
	char	   *vector;
	uint64		generation;		/* frozen index generation, 0 if none */
} AnnCatalogEntry;

static void
Put32(StringInfo buffer, uint32 value)
{
	appendBinaryStringInfo(buffer, (char *) &value, sizeof(value));
}

static const char *
CheckName(text *name)
{
	char	   *value = text_to_cstring(name);

	if (strlen(value) == 0 || strlen(value) > ANN_INDEX_NAME_MAX)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("ann index name must have 1 to %d bytes", ANN_INDEX_NAME_MAX)));
	return value;
}

/* Networked GPU nodes when configured, otherwise the local worker socket. */
bool
AnnIndexCall(uint16 operation, const char *payload, size_t length, int timeoutMs,
			 uint32 *status, char **reply, uint32 *replyLength)
{
	instr_time	start;
	long		delayUs = 500;

	INSTR_TIME_SET_CURRENT(start);
	for (;;)
	{
		instr_time	now;
		bool		ok;

		if (AnnRemoteConfigured())
			ok = AnnRemoteCall(operation, payload, length, timeoutMs, status, reply, replyLength);
		else
			ok = AnnWorkerCall(operation, payload, length, timeoutMs, status, reply, replyLength);

		/*
		 * A full admission queue is back-pressure: wait with jittered
		 * exponential backoff (0.5 to 16 ms) and retry, for up to
		 * ann.worker_overload_wait_ms. A request too large for the worker's
		 * whole budget still fails once that time is up.
		 */
		if (!ok || *status != ANN_STATUS_OVERLOADED)
			return ok;
		INSTR_TIME_SET_CURRENT(now);
		INSTR_TIME_SUBTRACT(now, start);
		if (INSTR_TIME_GET_MILLISEC(now) + delayUs / 1000.0 >= ann_worker_overload_wait_ms)
			return ok;
		pg_usleep(delayUs / 2 + random() % (delayUs / 2 + 1));
		CHECK_FOR_INTERRUPTS();
		if (delayUs < 16000)
			delayUs *= 2;
	}
}

static void
CheckWorker(bool ok, uint32 status, const char *name)
{
	if (!ok)
		ereport(ERROR,
				(errcode(ERRCODE_CONNECTION_FAILURE),
				 errmsg("ann worker is unavailable"),
				 errhint("Set ann.worker_address or ann.worker_socket to a running ann-worker.")));
	if (status == ANN_STATUS_OK)
		return;
	if (status == ANN_STATUS_OVERLOADED)
		ereport(ERROR,
				(errcode(ERRCODE_INSUFFICIENT_RESOURCES),
				 errmsg("ann worker is overloaded")));
	if (status == ANN_STATUS_STALE)
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_OBJECT),
				 errmsg("ann index \"%s\" is not loaded, or is already loaded", name)));
	if (status == ANN_STATUS_UNSUPPORTED)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("ann worker backend does not serve loaded indexes")));
	if (status == ANN_STATUS_GENERATION)
		ereport(ERROR,
				(errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
				 errmsg("ann index \"%s\" on the GPU nodes is another generation than this host's catalog", name),
				 errhint("Run ann_index_load or ann_index_attach for the current export.")));
	ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
			 errmsg("ann worker rejected the request for index \"%s\" (status %u)", name, status)));
}

static void
ParseInfo(const char *reply, AnnIndexInfo *info)
{
	unsigned long long rows;

	const char *generation;

	strcpy(info->key, "-");
	strcpy(info->vector, "-");
	strcpy(info->generation, "-");
	if (sscanf(reply, "rows %llu dimensions %*u shards %*u dtype %*s source %255s key %63s vector %63s",
			   &rows, info->source, info->key, info->vector) < 2)
		elog(ERROR, "unexpected ann worker reply: %s", reply);
	info->rows = rows;
	if ((generation = strstr(reply, " generation ")) != NULL &&
		sscanf(generation, " generation %16s", info->generation) != 1)
		elog(ERROR, "unexpected ann worker reply: %s", reply);
}

/* Schema-qualified catalog name, in the extension's schema. */
static char *
CatalogName(FunctionCallInfo fcinfo)
{
	char	   *schema = get_namespace_name(get_func_namespace(fcinfo->flinfo->fn_oid));

	return psprintf("%s.ann_index_catalog", quote_identifier(schema));
}

/* Schema-qualified bitmap table name, in the extension's schema. */
static char *
BitmapTableName(FunctionCallInfo fcinfo)
{
	char	   *schema = get_namespace_name(get_func_namespace(fcinfo->flinfo->fn_oid));

	return psprintf("%s.ann_filter_bitmap", quote_identifier(schema));
}

static int64 PushAllBitmaps(FunctionCallInfo fcinfo, const char *name);
static void PushTombstones(FunctionCallInfo fcinfo, const char *name, const char *source);

/*
 * A filter's cache id: a hash of the index name and the filter's text. Every
 * PostgreSQL host computes the same id for the same filter, so hosts sharing
 * GPU nodes share cached bitmaps instead of overwriting each other's.
 */
static int64
FilterId(const char *name, uint64 generation, const char *filter)
{
	StringInfoData key;

	initStringInfo(&key);
	appendBinaryStringInfo(&key, name, strlen(name) + 1);
	/* Files of a frozen index: a rebuilt table gets new ids. */
	if (generation != 0)
		appendBinaryStringInfo(&key, (char *) &generation, sizeof(generation));
	appendStringInfoString(&key, filter);
	return (int64) hash_bytes_extended((const unsigned char *) key.data, key.len, 0x616e6e5f62697473);
}

static char *
QualifiedSource(const char *source)
{
	List	   *names = stringToQualifiedNameList(source
#if PG_VERSION_NUM >= 160000
												  ,NULL
#endif
		);

	return NameListToQuotedString(names);
}

/*
 * The index is immutable: refuse one whose source table on this host no
 * longer has the row count it was built from. Rebuild after changing the
 * table.
 */
static void
CheckSource(const AnnIndexInfo *info)
{
	char	   *query;
	bool		isnull;
	int64		count;

	if (strcmp(info->source, "-") == 0)
		return;
	query = psprintf("SELECT count(*) FROM %s", QualifiedSource(info->source));
	if (SPI_connect() != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed");
	if (SPI_execute(query, true, 1) != SPI_OK_SELECT || SPI_processed != 1)
		elog(ERROR, "could not count rows of %s", info->source);
	count = DatumGetInt64(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));
	SPI_finish();
	if (count != (int64) info->rows)
		ereport(ERROR,
				(errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
				 errmsg("ann index is stale: %s has " INT64_FORMAT " rows, index has " UINT64_FORMAT,
						info->source, count, info->rows),
				 errhint("Rebuild the index with scripts/ann/build_cuvs_index.py.")));
}

/*
 * Files exported from a frozen index carry its generation: require a frozen
 * ann index on the source table with the same content, so the GPU index and
 * this host's table hold the same rows in the same order.
 */
static void
CheckGeneration(const AnnIndexInfo *info)
{
	Oid			am = get_am_oid("ann", false);
	List	   *names;
	Relation	heap;
	List	   *indexes;
	ListCell   *lc;
	bool		found = false;

	if (strcmp(info->generation, "-") == 0)
		return;
	if (strcmp(info->source, "-") == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("ann index directory has a generation but no source table")));
	names = stringToQualifiedNameList(info->source
#if PG_VERSION_NUM >= 160000
									  ,NULL
#endif
		);
	heap = table_openrv(makeRangeVarFromNameList(names), AccessShareLock);
	indexes = RelationGetIndexList(heap);
	foreach(lc, indexes)
	{
		Relation	index = index_open(lfirst_oid(lc), AccessShareLock);

		if (index->rd_rel->relam == am && index->rd_index->indisvalid)
		{
			AnnFrozenMetaData *meta = AnnFrozenGetMeta(index, true);

			found = meta != NULL &&
				strcmp(psprintf("%016llx", (unsigned long long) meta->generation), info->generation) == 0;
		}
		index_close(index, AccessShareLock);
		if (found)
			break;
	}
	table_close(heap, AccessShareLock);
	if (!found)
		ereport(ERROR,
				(errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
				 errmsg("ann index directory generation %s matches no frozen ann index of %s",
						info->generation, info->source),
				 errhint("Export the table's frozen index again with ann_index_export and rebuild the directory.")));
}

/*
 * Generation of each loaded index name, as this backend last read it from
 * the catalog. Requests carry it; a node serving another generation answers
 * ANN_STATUS_GENERATION, and the entry is read again.
 */
typedef struct GenerationEntry
{
	char		name[ANN_INDEX_NAME_MAX + 1];
	uint64		generation;
} GenerationEntry;

static HTAB *generations = NULL;

/* Whether the catalog has its generation column (0.9.1 and later). */
static bool
CatalogHasGeneration(FunctionCallInfo fcinfo)
{
	bool		found;

	if (SPI_connect() != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed");
	if (SPI_execute(psprintf("SELECT 1 FROM pg_attribute WHERE attrelid = '%s'::regclass "
							 "AND attname = 'generation' AND NOT attisdropped", CatalogName(fcinfo)),
					true, 1) != SPI_OK_SELECT)
		elog(ERROR, "could not read the ann index catalog");
	found = SPI_processed == 1;
	SPI_finish();
	return found;
}

static uint64
ParseGeneration(const char *value)
{
	return value && strlen(value) == 16 ? strtou64(value, NULL, 16) : 0;
}

static uint64
CachedGeneration(FunctionCallInfo fcinfo, const char *name, bool refresh)
{
	GenerationEntry *entry;
	bool		found;
	Oid			types[1] = {TEXTOID};
	Datum		values[1];
	uint64		generation = 0;

	if (generations == NULL)
	{
		HASHCTL		ctl;

		ctl.keysize = ANN_INDEX_NAME_MAX + 1;
		ctl.entrysize = sizeof(GenerationEntry);
		generations = hash_create("ann index generations", 64, &ctl, HASH_ELEM | HASH_STRINGS);
	}
	entry = hash_search(generations, name, HASH_FIND, NULL);
	if (entry && !refresh)
		return entry->generation;
	values[0] = CStringGetTextDatum(name);
	if (SPI_connect() != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed");
	/* to_jsonb tolerates a catalog from before the generation column. */
	if (SPI_execute_with_args(psprintf("SELECT to_jsonb(c) ->> 'generation' FROM %s c WHERE name = $1",
									   CatalogName(fcinfo)),
							  1, types, values, NULL, true, 1) != SPI_OK_SELECT)
		elog(ERROR, "could not read the ann index catalog");
	if (SPI_processed == 1)
		generation = ParseGeneration(SPI_getvalue(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1));
	SPI_finish();
	entry = hash_search(generations, name, HASH_ENTER, &found);
	entry->generation = generation;
	return generation;
}

/*
 * IndexCall for a named index, tagged with its generation. When the nodes
 * serve another generation, the catalog is read again and the call retried
 * once. Returns the generation used in *used.
 */
static bool
TaggedCall(FunctionCallInfo fcinfo, const char *name, uint16 operation, const char *payload, size_t length,
		   int timeoutMs, uint32 *status, char **reply, uint32 *replyLength, uint64 *used)
{
	uint64		generation = CachedGeneration(fcinfo, name, false);
	bool		ok = false;

	for (int attempt = 0; attempt < 2; attempt++)
	{
		uint64		fresh;

		ann_request_generation = generation;
		PG_TRY();
		{
			ok = AnnIndexCall(operation, payload, length, timeoutMs, status, reply, replyLength);
		}
		PG_FINALLY();
		{
			ann_request_generation = 0;
		}
		PG_END_TRY();
		if (!ok || *status != ANN_STATUS_GENERATION || attempt == 1)
			break;
		fresh = CachedGeneration(fcinfo, name, true);
		if (fresh == generation)
			break;
		generation = fresh;
	}
	if (used)
		*used = generation;
	return ok;
}

/* Record where the index's rows live, so this host can route exact scans. */
static void
RecordCatalog(FunctionCallInfo fcinfo, const char *name, const AnnIndexInfo *info)
{
	Oid			types[6] = {TEXTOID, TEXTOID, TEXTOID, TEXTOID, INT8OID, TEXTOID};
	Datum		values[6];
	char		nulls[6] = {' ', ' ', ' ', ' ', ' ', ' '};
	bool		generation = CatalogHasGeneration(fcinfo);

	if (strcmp(info->source, "-") == 0 || strcmp(info->key, "-") == 0 || strcmp(info->vector, "-") == 0)
		return;
	values[0] = CStringGetTextDatum(name);
	values[1] = CStringGetTextDatum(QualifiedSource(info->source));
	values[2] = CStringGetTextDatum(info->key);
	values[3] = CStringGetTextDatum(info->vector);
	values[4] = Int64GetDatum((int64) info->rows);
	values[5] = CStringGetTextDatum(info->generation);
	if (strcmp(info->generation, "-") == 0)
		nulls[5] = 'n';
	if (SPI_connect() != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed");
	if (SPI_execute_with_args(generation ?
							  psprintf("INSERT INTO %s (name, source, key_column, vector_column, rows, generation) "
									   "VALUES ($1, $2::regclass, $3, $4, $5, $6) ON CONFLICT (name) DO UPDATE "
									   "SET source = EXCLUDED.source, key_column = EXCLUDED.key_column, "
									   "vector_column = EXCLUDED.vector_column, rows = EXCLUDED.rows, "
									   "generation = EXCLUDED.generation, loaded_at = now()", CatalogName(fcinfo)) :
							  psprintf("INSERT INTO %s (name, source, key_column, vector_column, rows) "
									   "VALUES ($1, $2::regclass, $3, $4, $5) ON CONFLICT (name) DO UPDATE "
									   "SET source = EXCLUDED.source, key_column = EXCLUDED.key_column, "
									   "vector_column = EXCLUDED.vector_column, rows = EXCLUDED.rows, "
									   "loaded_at = now()", CatalogName(fcinfo)),
							  generation ? 6 : 5, types, values, nulls, false, 0) != SPI_OK_INSERT)
		elog(ERROR, "could not record ann index \"%s\"", name);
	SPI_finish();
	(void) CachedGeneration(fcinfo, name, true);
}

static bool
LookupCatalog(FunctionCallInfo fcinfo, const char *name, AnnCatalogEntry *entry)
{
	Oid			types[1] = {TEXTOID};
	Datum		values[1];
	bool		found;

	values[0] = CStringGetTextDatum(name);
	if (SPI_connect() != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed");
	if (SPI_execute_with_args(psprintf("SELECT source::text, key_column::text, vector_column::text, "
									   "to_jsonb(c) ->> 'generation' FROM %s c WHERE name = $1", CatalogName(fcinfo)),
							  1, types, values, NULL, true, 1) != SPI_OK_SELECT)
		elog(ERROR, "could not read the ann index catalog");
	found = SPI_processed == 1;
	if (found)
	{
		MemoryContext old = MemoryContextSwitchTo(fcinfo->flinfo->fn_mcxt);

		entry->source = pstrdup(SPI_getvalue(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1));
		entry->key = pstrdup(quote_identifier(SPI_getvalue(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 2)));
		entry->vector = pstrdup(quote_identifier(SPI_getvalue(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 3)));
		entry->generation = ParseGeneration(SPI_getvalue(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 4));
		MemoryContextSwitchTo(old);
	}
	SPI_finish();
	return found;
}

static void
RequireSuperuser(const char *action)
{
	if (!superuser())
		ereport(ERROR,
				(errcode(ERRCODE_INSUFFICIENT_PRIVILEGE),
				 errmsg("must be superuser to %s an ann index", action)));
}

/*
 * ann_index_load(name, path): load an index directory into the worker, or
 * into every node of ann.worker_address. Workers read the files, so the path
 * is resolved on each GPU node.
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(ann_index_load);
Datum
ann_index_load(PG_FUNCTION_ARGS)
{
	const char *name = CheckName(PG_GETARG_TEXT_PP(0));
	char	   *path = text_to_cstring(PG_GETARG_TEXT_PP(1));
	StringInfoData payload;
	uint32		status = 0;
	char	   *reply = NULL;
	uint32		replyLength;
	AnnIndexInfo info;
	bool		ok;

	RequireSuperuser("load");
	initStringInfo(&payload);
	Put32(&payload, strlen(name));
	appendBinaryStringInfo(&payload, name, strlen(name));
	appendBinaryStringInfo(&payload, path, strlen(path));
	if (AnnRemoteConfigured())
		ok = AnnRemoteBroadcast(ANN_OP_LOAD_INDEX, payload.data, payload.len, ANN_LOAD_TIMEOUT_MS,
								&status, &reply, &replyLength) > 0 || status != PG_UINT32_MAX;
	else
		ok = AnnWorkerCall(ANN_OP_LOAD_INDEX, payload.data, payload.len, ANN_LOAD_TIMEOUT_MS,
						   &status, &reply, &replyLength);
	CheckWorker(ok, status, name);
	ParseInfo(reply, &info);
	CheckSource(&info);
	CheckGeneration(&info);
	RecordCatalog(fcinfo, name, &info);
	if (strcmp(info.source, "-") != 0)
		PushTombstones(fcinfo, name, QualifiedSource(info.source));

	/* Freshly started nodes have no bitmaps: cache the stored ones. */
	{
		int64		pushed = PushAllBitmaps(fcinfo, name);

		if (pushed > 0)
			ereport(NOTICE,
					(errmsg("cached %lld ann filter bitmaps of index \"%s\" on the GPU nodes",
							(long long) pushed, name)));
	}
	PG_RETURN_INT64((int64) info.rows);
}

/*
 * ann_index_attach(name): use an index another host already loaded on the
 * GPU nodes. Verifies this host's copy of the source table, then records it.
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(ann_index_attach);
Datum
ann_index_attach(PG_FUNCTION_ARGS)
{
	const char *name = CheckName(PG_GETARG_TEXT_PP(0));
	uint32		status = 0;
	char	   *reply = NULL;
	uint32		replyLength;
	AnnIndexInfo info;
	bool		ok;

	RequireSuperuser("attach");
	ok = AnnIndexCall(ANN_OP_INDEX_INFO, name, strlen(name), 0, &status, &reply, &replyLength);
	CheckWorker(ok, status, name);
	ParseInfo(reply, &info);
	CheckSource(&info);
	CheckGeneration(&info);
	RecordCatalog(fcinfo, name, &info);
	if (strcmp(info.source, "-") != 0)
		PushTombstones(fcinfo, name, QualifiedSource(info.source));

	/* Freshly started nodes have no bitmaps: cache the stored ones. */
	{
		int64		pushed = PushAllBitmaps(fcinfo, name);

		if (pushed > 0)
			ereport(NOTICE,
					(errmsg("cached %lld ann filter bitmaps of index \"%s\" on the GPU nodes",
							(long long) pushed, name)));
	}
	PG_RETURN_INT64((int64) info.rows);
}

static int
KeyCompare(const void *a, const void *b)
{
	int64		x = *(const int64 *) a;
	int64		y = *(const int64 *) b;

	return x < y ? -1 : x > y;
}

/* Sort when needed and drop duplicates; returns the new count. */
static int
SortKeys(int64 *keys, int nkeys)
{
	int			n = 0;

	for (int i = 1; i < nkeys; i++)
		if (keys[i] < keys[i - 1])
		{
			qsort(keys, nkeys, sizeof(int64), KeyCompare);
			break;
		}
	for (int i = 0; i < nkeys; i++)
		if (i == 0 || keys[i] != keys[i - 1])
			keys[n++] = keys[i];
	return n;
}

static void
PutVarint(StringInfo buffer, uint64 value)
{
	char		bytes[10];
	int			n = 0;

	do
	{
		bytes[n] = (char) (value & 0x7f);
		value >>= 7;
		if (value)
			bytes[n] |= (char) 0x80;
		n++;
	} while (value);
	appendBinaryStringInfo(buffer, bytes, n);
}

static void
InitResult(FunctionCallInfo fcinfo, TupleDesc *tupdesc, Tuplestorestate **store)
{
	ReturnSetInfo *rsinfo = (ReturnSetInfo *) fcinfo->resultinfo;
	MemoryContext oldcontext;

	if (rsinfo == NULL || !IsA(rsinfo, ReturnSetInfo) ||
		!(rsinfo->allowedModes & SFRM_Materialize))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("set-valued function called in context that cannot accept a set")));
	if (get_call_result_type(fcinfo, NULL, tupdesc) != TYPEFUNC_COMPOSITE)
		elog(ERROR, "return type must be a row type");
	oldcontext = MemoryContextSwitchTo(rsinfo->econtext->ecxt_per_query_memory);
	*tupdesc = CreateTupleDescCopy(*tupdesc);
	*store = tuplestore_begin_heap(true, false, work_mem);
	MemoryContextSwitchTo(oldcontext);
	rsinfo->returnMode = SFRM_Materialize;
	rsinfo->setResult = *store;
	rsinfo->setDesc = *tupdesc;
}

/* Check a worker search reply and append its (key, distance) hits. */
static void
StoreHits(bool ok, uint32 status, const char *name, const char *reply, uint32 replyLength,
		  TupleDesc tupdesc, Tuplestorestate *store)
{
	uint32		count;

	CheckWorker(ok, status, name);
	if (replyLength < 8 || (count = *(uint32 *) reply) != (replyLength - 8) / 12 ||
		(Size) count * 12 + 8 != replyLength)
		elog(ERROR, "malformed ann worker search reply");
	for (uint32 i = 0; i < count; i++)
	{
		Datum		values[2];
		bool		nulls[2] = {false, false};
		int64		key;
		float		distance;

		memcpy(&key, reply + 8 + (Size) i * 12, sizeof(key));
		memcpy(&distance, reply + 16 + (Size) i * 12, sizeof(distance));
		values[0] = Int64GetDatum(key);
		values[1] = Float4GetDatum(distance);
		tuplestore_putvalues(store, tupdesc, values, nulls);
	}
}

/*
 * Ask a GPU node for the top k, restricted to sorted unique keys when
 * filtered. Keys travel as a zigzag varint first key and varint gaps.
 */
static void
SearchGpu(FunctionCallInfo fcinfo, const char *name, Vector *query, int32 k, bool filtered,
		  const int64 *keys, int nkeys, TupleDesc tupdesc, Tuplestorestate *store)
{
	StringInfoData payload;
	StringInfoData packed;
	uint32		status = 0;
	char	   *reply = NULL;
	uint32		replyLength;
	bool		ok;

	initStringInfo(&packed);
	for (int i = 0; i < nkeys; i++)
	{
		if (i == 0)
			PutVarint(&packed, ((uint64) keys[0] << 1) ^ (uint64) (keys[0] >> 63));
		else
			PutVarint(&packed, (uint64) keys[i] - (uint64) keys[i - 1]);
	}
	initStringInfo(&payload);
	Put32(&payload, strlen(name));
	appendBinaryStringInfo(&payload, name, strlen(name));
	Put32(&payload, query->dim);
	Put32(&payload, k);
	Put32(&payload, filtered ? (uint32) nkeys : PG_UINT32_MAX);
	Put32(&payload, packed.len);
	appendBinaryStringInfo(&payload, (char *) query->x, sizeof(float) * query->dim);
	appendBinaryStringInfo(&payload, packed.data, packed.len);
	ok = TaggedCall(fcinfo, name, ANN_OP_INDEX_SEARCH_PACKED, payload.data, payload.len, 0, &status, &reply,
					&replyLength, NULL);
	StoreHits(ok, status, name, reply, replyLength, tupdesc, store);
}

/*
 * Exact top k on this host with the SQL operator, over either eligible keys
 * or a filter predicate. The query plan is PostgreSQL's own.
 */
static void
SearchLocal(FunctionCallInfo fcinfo, const AnnCatalogEntry *entry, Datum query, int32 k,
			const char *filter, const int64 *keys, int nkeys, TupleDesc tupdesc, Tuplestorestate *store)
{
	Oid			types[3];
	Datum		values[3];
	char	   *sql;
	SPIPlanPtr	plan;

	types[0] = get_fn_expr_argtype(fcinfo->flinfo, 1);
	values[0] = query;
	types[1] = INT4OID;
	values[1] = Int32GetDatum(k);
	if (filter)
		sql = psprintf("SELECT %s::bigint, (%s <-> $1)::real FROM %s WHERE (%s) ORDER BY 2, 1 LIMIT $2",
					   entry->key, entry->vector, entry->source, filter);
	else
	{
		Datum	   *elements = palloc(sizeof(Datum) * Max(nkeys, 1));

		for (int i = 0; i < nkeys; i++)
			elements[i] = Int64GetDatum(keys[i]);
		types[2] = INT8ARRAYOID;
		values[2] = PointerGetDatum(construct_array(elements, nkeys, INT8OID, 8, FLOAT8PASSBYVAL, TYPALIGN_DOUBLE));
		sql = psprintf("SELECT %s::bigint, (%s <-> $1)::real FROM %s WHERE %s = ANY ($3) ORDER BY 2, 1 LIMIT $2",
					   entry->key, entry->vector, entry->source, entry->key);
	}
	if (SPI_connect() != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed");

	/* One prepared read-only statement: a filter cannot smuggle in writes. */
	plan = SPI_prepare(sql, filter ? 2 : 3, types);
	if (plan == NULL)
		elog(ERROR, "could not prepare ann local search: %s", SPI_result_code_string(SPI_result));
	if (SPI_execute_plan(plan, values, NULL, true, 0) != SPI_OK_SELECT)
		elog(ERROR, "ann local search failed");
	for (uint64 i = 0; i < SPI_processed; i++)
	{
		Datum		row[2];
		bool		nulls[2];

		row[0] = SPI_getbinval(SPI_tuptable->vals[i], SPI_tuptable->tupdesc, 1, &nulls[0]);
		row[1] = SPI_getbinval(SPI_tuptable->vals[i], SPI_tuptable->tupdesc, 2, &nulls[1]);
		tuplestore_putvalues(store, tupdesc, row, nulls);
	}
	SPI_finish();
}

static int32
CheckK(int32 k)
{
	if (k < 1 || k > ANN_INDEX_MAX_K)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("k must be between 1 and %d", ANN_INDEX_MAX_K)));
	return k;
}

/*
 * ann_index_search(name, query, k, filter): top-k (id, distance) from a
 * loaded index. A non-NULL filter restricts results to those keys, so any
 * SQL predicate can drive hybrid search; NULL elements are ignored. Filters
 * smaller than ann.gpu_min_eligible are scored exactly on this host when the
 * index is in the catalog.
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(ann_index_search);
Datum
ann_index_search(PG_FUNCTION_ARGS)
{
	TupleDesc	tupdesc;
	Tuplestorestate *store;
	const char *name;
	int32		k;
	int64	   *keys = NULL;
	int			nkeys = 0;
	bool		filtered = !PG_ARGISNULL(3);
	AnnCatalogEntry entry;

	InitResult(fcinfo, &tupdesc, &store);
	if (PG_ARGISNULL(0) || PG_ARGISNULL(1) || PG_ARGISNULL(2))
		return (Datum) 0;
	name = CheckName(PG_GETARG_TEXT_PP(0));
	k = CheckK(PG_GETARG_INT32(2));
	if (filtered)
	{
		ArrayType  *array = PG_GETARG_ARRAYTYPE_P(3);
		Datum	   *values;
		bool	   *nulls;
		int			n;

		deconstruct_array(array, INT8OID, 8, FLOAT8PASSBYVAL, TYPALIGN_DOUBLE, &values, &nulls, &n);
		keys = palloc(sizeof(int64) * Max(n, 1));
		for (int i = 0; i < n; i++)
			if (!nulls[i])
				keys[nkeys++] = DatumGetInt64(values[i]);

		/* No eligible keys means no rows; skip the round trip. */
		if (nkeys == 0)
			return (Datum) 0;
		nkeys = SortKeys(keys, nkeys);
		if (nkeys < ann_gpu_min_eligible && LookupCatalog(fcinfo, name, &entry))
		{
			SearchLocal(fcinfo, &entry, PG_GETARG_DATUM(1), k, NULL, keys, nkeys, tupdesc, store);
			return (Datum) 0;
		}
	}
	SearchGpu(fcinfo, name, PG_GETARG_VECTOR_P(1), k, filtered, keys, nkeys, tupdesc, store);
	return (Datum) 0;
}

/* Eligible keys collected straight from the executor, without SPI tuples. */
typedef struct KeyReceiver
{
	DestReceiver pub;
	int64	   *keys;			/* in the caller's context, which outlives SPI */
	int			nkeys;
	int			capacity;
} KeyReceiver;

static bool
KeyReceive(TupleTableSlot *slot, DestReceiver *self)
{
	KeyReceiver *receiver = (KeyReceiver *) self;
	bool		isnull;
	Datum		value = slot_getattr(slot, 1, &isnull);

	if (isnull)
		return true;
	if (receiver->nkeys == receiver->capacity)
	{
		if (receiver->capacity > INT_MAX / 2)
			ereport(ERROR,
					(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
					 errmsg("ann hybrid filter matches too many rows")));
		receiver->capacity *= 2;
		receiver->keys = repalloc_huge(receiver->keys, sizeof(int64) * receiver->capacity);
	}
	receiver->keys[receiver->nkeys++] = DatumGetInt64(value);
	return true;
}

static void
KeyStartup(DestReceiver *self, int operation, TupleDesc typeinfo)
{
}

static void
KeyShutdown(DestReceiver *self)
{
}

static void
KeyDestroy(DestReceiver *self)
{
}

/*
 * Run the filter once and return every eligible key, in the caller's memory
 * context. Rows reach the receiver as executor slots, so no per-row SPI
 * tuple is built or copied.
 */
static int64 *
CollectKeys(const AnnCatalogEntry *entry, const char *filter, int *nkeys)
{
	KeyReceiver receiver = {0};
	SPIPlanPtr	plan;

	receiver.pub.receiveSlot = KeyReceive;
	receiver.pub.rStartup = KeyStartup;
	receiver.pub.rShutdown = KeyShutdown;
	receiver.pub.rDestroy = KeyDestroy;
	receiver.pub.mydest = DestNone;
	receiver.capacity = 1024;
	receiver.keys = palloc(sizeof(int64) * receiver.capacity);
	if (SPI_connect() != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed");

	/* One prepared read-only statement: a filter cannot smuggle in writes. */
	plan = SPI_prepare(psprintf("SELECT %s::bigint FROM %s WHERE (%s)", entry->key, entry->source, filter), 0, NULL);
	if (plan == NULL)
		elog(ERROR, "could not prepare the ann hybrid filter: %s", SPI_result_code_string(SPI_result));
	if (list_length(SPI_plan_get_plan_sources(plan)) != 1)
		ereport(ERROR,
				(errcode(ERRCODE_SYNTAX_ERROR),
				 errmsg("ann hybrid filter must be a single boolean expression")));
#if PG_VERSION_NUM >= 140000
	{
		SPIExecuteOptions options = {0};
		int			result;

		options.read_only = true;
		options.dest = &receiver.pub;
		result = SPI_execute_plan_extended(plan, &options);

		/* SPI reports a SELECT into a DestNone receiver as a utility. */
		if (result != SPI_OK_SELECT && result != SPI_OK_UTILITY)
			elog(ERROR, "could not evaluate the ann hybrid filter");
	}
#else
	{
		Portal		portal = SPI_cursor_open(NULL, plan, NULL, NULL, true);

		for (;;)
		{
			SPI_cursor_fetch(portal, true, ANN_KEY_FETCH);
			if (SPI_processed == 0)
				break;
			for (uint64 i = 0; i < SPI_processed; i++)
			{
				bool		isnull;
				Datum		value = SPI_getbinval(SPI_tuptable->vals[i], SPI_tuptable->tupdesc, 1, &isnull);

				if (isnull)
					continue;
				if (receiver.nkeys == receiver.capacity)
				{
					if (receiver.capacity > INT_MAX / 2)
						ereport(ERROR,
								(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
								 errmsg("ann hybrid filter matches too many rows")));
					receiver.capacity *= 2;
					receiver.keys = repalloc_huge(receiver.keys, sizeof(int64) * receiver.capacity);
				}
				receiver.keys[receiver.nkeys++] = DatumGetInt64(value);
			}
			SPI_freetuptable(SPI_tuptable);
			CHECK_FOR_INTERRUPTS();
		}
		SPI_cursor_close(portal);
	}
#endif
	SPI_finish();
	*nkeys = receiver.nkeys;
	return receiver.keys;
}

/*
 * ann_hybrid_search(name, query, k, filter): hybrid search where this host
 * evaluates the SQL filter on the index's source table, once. Filters with
 * fewer than ann.gpu_min_eligible rows run exactly here; larger ones send
 * their keys to a GPU node. The filter is a SQL boolean expression over the
 * source table, run read-only with the caller's privileges, like any dynamic
 * SQL; quote values with format().
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(ann_hybrid_search);
Datum
ann_hybrid_search(PG_FUNCTION_ARGS)
{
	TupleDesc	tupdesc;
	Tuplestorestate *store;
	const char *name;
	int32		k;
	char	   *filter;
	AnnCatalogEntry entry;
	int64	   *keys;
	int			nkeys;

	InitResult(fcinfo, &tupdesc, &store);
	if (PG_ARGISNULL(0) || PG_ARGISNULL(1) || PG_ARGISNULL(2) || PG_ARGISNULL(3))
		return (Datum) 0;
	name = CheckName(PG_GETARG_TEXT_PP(0));
	k = CheckK(PG_GETARG_INT32(2));
	filter = text_to_cstring(PG_GETARG_TEXT_PP(3));
	if (!LookupCatalog(fcinfo, name, &entry))
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_OBJECT),
				 errmsg("ann index \"%s\" is not in this database's catalog", name),
				 errhint("Run ann_index_load or ann_index_attach on this host.")));
	keys = CollectKeys(&entry, filter, &nkeys);
	if (nkeys == 0)
		return (Datum) 0;
	if (nkeys < ann_gpu_min_eligible)
	{
		/*
		 * Few rows: score them here, exactly. Rerunning the filter lets the
		 * plan use whatever index serves it, with no index on the key.
		 */
		SearchLocal(fcinfo, &entry, PG_GETARG_DATUM(1), k, filter, NULL, 0, tupdesc, store);
		return (Datum) 0;
	}
	nkeys = SortKeys(keys, nkeys);
	SearchGpu(fcinfo, name, PG_GETARG_VECTOR_P(1), k, true, keys, nkeys, tupdesc, store);
	return (Datum) 0;
}

/*
 * ann_attribute_search(name, query, k, attribute, values): top-k rows whose
 * index attribute (an int32 column stored with the index, see
 * build_cuvs_index.py --attribute) is one of values. The GPU node resolves
 * the filter from its own inverted column, so this host sends only the
 * predicate: no SQL scan, no key list.
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(ann_attribute_search);
Datum
ann_attribute_search(PG_FUNCTION_ARGS)
{
	TupleDesc	tupdesc;
	Tuplestorestate *store;
	const char *name;
	char	   *attribute;
	Vector	   *query;
	int32		k;
	ArrayType  *array;
	Datum	   *elements;
	bool	   *nulls;
	int			n;
	int			nvalues = 0;
	StringInfoData payload;
	StringInfoData values;
	uint32		status = 0;
	char	   *reply = NULL;
	uint32		replyLength;
	bool		ok;

	InitResult(fcinfo, &tupdesc, &store);
	if (PG_ARGISNULL(0) || PG_ARGISNULL(1) || PG_ARGISNULL(2) || PG_ARGISNULL(3) || PG_ARGISNULL(4))
		return (Datum) 0;
	name = CheckName(PG_GETARG_TEXT_PP(0));
	query = PG_GETARG_VECTOR_P(1);
	k = CheckK(PG_GETARG_INT32(2));
	attribute = text_to_cstring(PG_GETARG_TEXT_PP(3));
	if (strlen(attribute) == 0 || strlen(attribute) > 63)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("ann attribute name must be 1 to 63 bytes")));
	array = PG_GETARG_ARRAYTYPE_P(4);
	deconstruct_array(array, INT4OID, 4, true, TYPALIGN_INT, &elements, &nulls, &n);
	initStringInfo(&values);
	for (int i = 0; i < n; i++)
		if (!nulls[i])
		{
			Put32(&values, (uint32) DatumGetInt32(elements[i]));
			nvalues++;
		}
	if (nvalues == 0)
		return (Datum) 0;
	if (nvalues > ANN_ATTRIBUTE_MAX_VALUES)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("ann attribute search takes at most %d values", ANN_ATTRIBUTE_MAX_VALUES)));
	initStringInfo(&payload);
	Put32(&payload, strlen(name));
	appendBinaryStringInfo(&payload, name, strlen(name));
	Put32(&payload, query->dim);
	Put32(&payload, k);
	Put32(&payload, strlen(attribute));
	appendBinaryStringInfo(&payload, attribute, strlen(attribute));
	Put32(&payload, nvalues);
	appendBinaryStringInfo(&payload, values.data, values.len);
	appendBinaryStringInfo(&payload, (char *) query->x, sizeof(float) * query->dim);
	ok = TaggedCall(fcinfo, name, ANN_OP_INDEX_SEARCH_MATCH, payload.data, payload.len, 0, &status, &reply,
					&replyLength, NULL);
	StoreHits(ok, status, name, reply, replyLength, tupdesc, store);
	return (Datum) 0;
}

/*
 * Cache one stored bitmap on every GPU node (or the local worker) under its
 * id. Returns the number of nodes that took it; *failed counts nodes that
 * refused or were unreachable, and *status holds the last such status
 * (PG_UINT32_MAX when unreachable).
 */
static int
PushBitmap(const char *name, uint64 generation, int64 id, bytea *bitmap, int *failed, uint32 *status)
{
	StringInfoData payload;
	char	   *reply = NULL;
	uint32		replyLength;
	int			accepted;

	initStringInfo(&payload);
	Put32(&payload, strlen(name));
	appendBinaryStringInfo(&payload, name, strlen(name));
	appendBinaryStringInfo(&payload, (char *) &id, sizeof(id));
	Put32(&payload, VARSIZE_ANY_EXHDR(bitmap) / 4);
	appendBinaryStringInfo(&payload, VARDATA_ANY(bitmap), VARSIZE_ANY_EXHDR(bitmap));
	*status = PG_UINT32_MAX;
	ann_request_generation = generation;
	PG_TRY();
	{
		if (AnnRemoteConfigured())
			accepted = AnnRemoteBroadcastEach(ANN_OP_BITMAP_PUT, payload.data, payload.len, ANN_BITMAP_TIMEOUT_MS,
											  failed, status);
		else
		{
			accepted = AnnWorkerCall(ANN_OP_BITMAP_PUT, payload.data, payload.len, ANN_BITMAP_TIMEOUT_MS,
									 status, &reply, &replyLength) && *status == ANN_STATUS_OK;
			*failed = !accepted;
		}
	}
	PG_FINALLY();
	{
		ann_request_generation = 0;
	}
	PG_END_TRY();
	pfree(payload.data);
	return accepted;
}

/*
 * Push every stored bitmap of an index, one row at a time so memory stays
 * at one bitmap. Returns the number pushed; failures raise warnings.
 */
static int64
PushAllBitmaps(FunctionCallInfo fcinfo, const char *name)
{
	Oid			types[1] = {TEXTOID};
	Datum		values[1];
	SPIPlanPtr	plan;
	Portal		portal;
	int64		pushed = 0;
	int64		failed = 0;
	int			nodeFailures;
	uint32		status = 0;
	uint64		generation = CachedGeneration(fcinfo, name, true);

	values[0] = CStringGetTextDatum(name);
	if (SPI_connect() != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed");
	plan = SPI_prepare(psprintf("SELECT id, bitmap, filter FROM %s WHERE index_name = $1 ORDER BY id",
								BitmapTableName(fcinfo)), 1, types);
	if (plan == NULL)
		elog(ERROR, "could not read ann filter bitmaps");
	portal = SPI_cursor_open(NULL, plan, values, NULL, true);
	for (;;)
	{
		bool		isnull;
		int64		id;

		SPI_cursor_fetch(portal, true, 1);
		if (SPI_processed == 0)
			break;
		id = DatumGetInt64(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));

		/* Bitmaps built for another generation would select other rows. */
		if (id != FilterId(name, generation, SPI_getvalue(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 3)))
		{
			SPI_freetuptable(SPI_tuptable);
			continue;
		}
		if (PushBitmap(name, generation, id,
					   DatumGetByteaPP(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 2, &isnull)),
					   &nodeFailures, &status) > 0)
			pushed++;
		if (nodeFailures)
			failed++;
		SPI_freetuptable(SPI_tuptable);
		CHECK_FOR_INTERRUPTS();
	}
	SPI_cursor_close(portal);
	SPI_finish();
	if (failed)
		ereport(WARNING,
				(errmsg("could not push %lld ann filter bitmaps of index \"%s\" to every GPU node (worker status %d)",
						(long long) failed, name, (int) status),
				 errhint("Searches push missing bitmaps on demand; ann_filter_bitmap_push retries all.")));
	return pushed;
}

/*
 * For files exported from a frozen index, a filter over its INCLUDE
 * attributes is evaluated from the index's postings and bitset pages, in
 * the ordinals the GPU node uses, instead of by a SQL scan of the table.
 * Returns false when the filter needs SQL.
 */
static bool
FrozenBitmap(const char *source, uint64 generation, const char *filter, bytea **bitmap, uint64 *eligible)
{
	Oid			relid;
	Relation	index;
	AnnFrozenMetaData *meta;
	AnnPredicateColumns columns;
	StringInfoData program;
	size_t		nwords;

	if (generation == 0)
		return false;
	relid = DatumGetObjectId(DirectFunctionCall1(regclassin, CStringGetDatum(source)));
	if ((index = AnnFrozenIndexOf(relid, generation, AccessShareLock)) == NULL)
		return false;
	AnnPredicateColumnsInit(index, &columns);
	initStringInfo(&program);
	if (!AnnPredicateFromText(relid, filter, &columns, &program))
	{
		index_close(index, AccessShareLock);
		return false;
	}
	meta = AnnFrozenGetMeta(index, false);
	nwords = (meta->rows + 31) / 32;
	*bitmap = palloc(VARHDRSZ + nwords * 4);
	SET_VARSIZE(*bitmap, VARHDRSZ + nwords * 4);
	AnnFrozenEvaluateOrError(index, meta, program.data, program.len, (uint32 *) VARDATA(*bitmap), eligible,
							 false);
	index_close(index, AccessShareLock);
	return true;
}

/*
 * ann_filter_bitmap_build(name, filter): evaluate a SQL filter on the
 * index's source table once, have the GPU node turn the matching keys into
 * a bitmap over the index's rows, store it in ann_filter_bitmap under the
 * filter's text, and cache it on every GPU node under its id. Indexes are
 * immutable, so a bitmap stays valid for the index it was built for.
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(ann_filter_bitmap_build);
Datum
ann_filter_bitmap_build(PG_FUNCTION_ARGS)
{
	const char *name = CheckName(PG_GETARG_TEXT_PP(0));
	char	   *filter = text_to_cstring(PG_GETARG_TEXT_PP(1));
	AnnCatalogEntry entry;
	int64	   *keys;
	int			nkeys;
	StringInfoData packed;
	StringInfoData payload;
	uint32		status = 0;
	char	   *reply = NULL;
	uint32		replyLength;
	uint64		eligible;
	bytea	   *bitmap;
	Oid			types[5] = {TEXTOID, TEXTOID, INT8OID, BYTEAOID, INT8OID};
	Datum		values[5];
	int64		id;
	bool		ok;
	int			failures;
	uint64		generation;

	if (!superuser())
		ereport(ERROR,
				(errcode(ERRCODE_INSUFFICIENT_PRIVILEGE),
				 errmsg("must be superuser to build ann filter bitmaps")));
	if (!LookupCatalog(fcinfo, name, &entry))
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_OBJECT),
				 errmsg("ann index \"%s\" is not in this database's catalog", name),
				 errhint("Run ann_index_load or ann_index_attach on this host.")));
	generation = CachedGeneration(fcinfo, name, true);
	if (FrozenBitmap(entry.source, generation, filter, &bitmap, &eligible))
		goto store;
	keys = CollectKeys(&entry, filter, &nkeys);
	nkeys = SortKeys(keys, nkeys);
	initStringInfo(&packed);
	for (int i = 0; i < nkeys; i++)
	{
		if (i == 0)
			PutVarint(&packed, ((uint64) keys[0] << 1) ^ (uint64) (keys[0] >> 63));
		else
			PutVarint(&packed, (uint64) keys[i] - (uint64) keys[i - 1]);
	}
	initStringInfo(&payload);
	Put32(&payload, strlen(name));
	appendBinaryStringInfo(&payload, name, strlen(name));
	Put32(&payload, nkeys);
	Put32(&payload, packed.len);
	appendBinaryStringInfo(&payload, packed.data, packed.len);
	ok = TaggedCall(fcinfo, name, ANN_OP_INDEX_BITMAP, payload.data, payload.len, 0, &status, &reply,
					&replyLength, NULL);
	CheckWorker(ok, status, name);
	if (replyLength < 8 || (replyLength - 8) % 4 != 0)
		elog(ERROR, "malformed ann worker bitmap reply");
	memcpy(&eligible, reply, sizeof(eligible));
	bitmap = palloc(VARHDRSZ + replyLength - 8);
	SET_VARSIZE(bitmap, VARHDRSZ + replyLength - 8);
	memcpy(VARDATA(bitmap), reply + 8, replyLength - 8);

store:
	values[0] = CStringGetTextDatum(name);
	values[1] = CStringGetTextDatum(filter);
	values[2] = Int64GetDatum((int64) eligible);
	values[3] = PointerGetDatum(bitmap);
	id = FilterId(name, generation, filter);
	values[4] = Int64GetDatum(id);
	if (SPI_connect() != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed");
	if (SPI_execute_with_args(psprintf("INSERT INTO %s (index_name, filter, eligible, bitmap, id) VALUES ($1, $2, $3, $4, $5) "
									   "ON CONFLICT (index_name, filter) DO UPDATE SET eligible = excluded.eligible, "
									   "bitmap = excluded.bitmap, built_at = now()",
									   BitmapTableName(fcinfo)),
							  5, types, values, NULL, false, 0) != SPI_OK_INSERT)
		elog(ERROR, "could not store the ann filter bitmap");
	SPI_finish();

	/* The row is the source of truth; searches push it again if needed. */
	PushBitmap(name, generation, id, bitmap, &failures, &status);
	if (failures)
		ereport(WARNING,
				(errmsg("could not cache ann filter bitmap %lld on every GPU node (worker status %d)",
						(long long) id, (int) status),
				 errhint("Searches push missing bitmaps on demand.")));
	PG_RETURN_INT64((int64) eligible);
}

/*
 * ann_filter_bitmap_push(name): cache every stored bitmap of an index on
 * every GPU node, for example after nodes restart. ann_index_load does this
 * itself. Returns the number pushed.
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(ann_filter_bitmap_push);
Datum
ann_filter_bitmap_push(PG_FUNCTION_ARGS)
{
	const char *name = CheckName(PG_GETARG_TEXT_PP(0));

	if (!superuser())
		ereport(ERROR,
				(errcode(ERRCODE_INSUFFICIENT_PRIVILEGE),
				 errmsg("must be superuser to push ann filter bitmaps")));
	PG_RETURN_INT64(PushAllBitmaps(fcinfo, name));
}

/*
 * ann_bitmap_search(name, query, k, filter): top k among the rows of a
 * filter built with ann_filter_bitmap_build, matched by its exact text. Only
 * the filter's id travels with the query: GPU nodes cache the bitmaps. A
 * node without it (restarted, or loaded by another host) answers
 * ANN_MISSING; the bitmap is then pushed from the table and the search
 * retried once.
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(ann_bitmap_search);
Datum
ann_bitmap_search(PG_FUNCTION_ARGS)
{
	TupleDesc	tupdesc;
	Tuplestorestate *store;
	const char *name;
	char	   *filter;
	Vector	   *query;
	int32		k;
	StringInfoData payload;
	Oid			types[2] = {TEXTOID, TEXTOID};
	Datum		values[2];
	uint32		status = 0;
	char	   *reply = NULL;
	uint32		replyLength;
	int64		id = 0;
	bool		ok;
	bool		found;
	uint64		generation = 0;

	InitResult(fcinfo, &tupdesc, &store);
	if (PG_ARGISNULL(0) || PG_ARGISNULL(1) || PG_ARGISNULL(2) || PG_ARGISNULL(3))
		return (Datum) 0;
	name = CheckName(PG_GETARG_TEXT_PP(0));
	query = PG_GETARG_VECTOR_P(1);
	k = CheckK(PG_GETARG_INT32(2));
	filter = text_to_cstring(PG_GETARG_TEXT_PP(3));
	values[0] = CStringGetTextDatum(name);
	values[1] = CStringGetTextDatum(filter);
	if (SPI_connect() != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed");
	if (SPI_execute_with_args(psprintf("SELECT id FROM %s WHERE index_name = $1 AND filter = $2",
									   BitmapTableName(fcinfo)),
							  2, types, values, NULL, true, 1) != SPI_OK_SELECT)
		elog(ERROR, "could not read ann filter bitmaps");
	found = SPI_processed == 1;
	if (found)
	{
		bool		isnull;

		id = DatumGetInt64(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));
	}
	SPI_finish();
	if (!found)
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_OBJECT),
				 errmsg("no ann filter bitmap for index \"%s\" and filter \"%s\"", name, filter),
				 errhint("Run ann_filter_bitmap_build first.")));

	initStringInfo(&payload);
	Put32(&payload, strlen(name));
	appendBinaryStringInfo(&payload, name, strlen(name));
	Put32(&payload, query->dim);
	Put32(&payload, k);
	appendBinaryStringInfo(&payload, (char *) &id, sizeof(id));
	appendBinaryStringInfo(&payload, (char *) query->x, sizeof(float) * query->dim);
	ok = TaggedCall(fcinfo, name, ANN_OP_INDEX_SEARCH_CACHED, payload.data, payload.len, 0, &status, &reply,
					&replyLength, &generation);
	/* A generation cached before the index was (re)loaded: read it again. */
	if (id != FilterId(name, generation, filter) &&
		id == FilterId(name, CachedGeneration(fcinfo, name, true), filter))
		ok = TaggedCall(fcinfo, name, ANN_OP_INDEX_SEARCH_CACHED, payload.data, payload.len, 0, &status,
						&reply, &replyLength, &generation);
	if (id != FilterId(name, generation, filter))
		ereport(ERROR,
				(errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
				 errmsg("ann filter bitmap for index \"%s\" and filter \"%s\" was built for another generation",
						name, filter),
				 errhint("Run ann_filter_bitmap_build again.")));
	if (ok && status == ANN_STATUS_MISSING)
	{
		Oid			idType[1] = {INT8OID};
		Datum		idValue[1] = {Int64GetDatum(id)};
		bool		pushed = false;
		uint32		pushStatus = 0;
		int			failures;

		if (SPI_connect() != SPI_OK_CONNECT)
			elog(ERROR, "SPI_connect failed");
		if (SPI_execute_with_args(psprintf("SELECT bitmap FROM %s WHERE id = $1", BitmapTableName(fcinfo)),
								  1, idType, idValue, NULL, true, 1) == SPI_OK_SELECT && SPI_processed == 1)
		{
			bool		isnull;

			pushed = PushBitmap(name, generation, id,
								DatumGetByteaPP(SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull)),
								&failures, &pushStatus) > 0;
		}
		SPI_finish();
		if (!pushed)
			ereport(ERROR,
					(errcode(ERRCODE_CONNECTION_FAILURE),
					 errmsg("could not push ann filter bitmap %lld to the GPU nodes (worker status %d)",
							(long long) id, (int) pushStatus)));
		ok = TaggedCall(fcinfo, name, ANN_OP_INDEX_SEARCH_CACHED, payload.data, payload.len, 0, &status,
						&reply, &replyLength, NULL);
	}
	if (ok && status == ANN_STATUS_MISSING)
		ereport(ERROR,
				(errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
				 errmsg("ann filter bitmap %lld is not cached on the GPU node", (long long) id)));
	StoreHits(ok, status, name, reply, replyLength, tupdesc, store);
	return (Datum) 0;
}

/* (eligible, bitmap) as a composite result. */
static Datum
FilterResult(FunctionCallInfo fcinfo, uint64 eligible, bytea *bitmap)
{
	TupleDesc	tupdesc;
	Datum		values[2];
	bool		nulls[2] = {false, false};

	if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
		elog(ERROR, "return type must be a row type");
	values[0] = Int64GetDatum((int64) eligible);
	values[1] = PointerGetDatum(bitmap);
	PG_RETURN_DATUM(HeapTupleGetDatum(heap_form_tuple(tupdesc, values, nulls)));
}

/*
 * ann_index_filter(name, filter): the rows the GPU node selects for a filter
 * over the attributes of an index exported from a frozen index, evaluated
 * there (operation 23) from its own postings. ann_frozen_filter evaluates
 * the same program from the frozen pages; the bitmaps are equal.
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(ann_index_filter);
Datum
ann_index_filter(PG_FUNCTION_ARGS)
{
	const char *name = CheckName(PG_GETARG_TEXT_PP(0));
	char	   *filter = text_to_cstring(PG_GETARG_TEXT_PP(1));
	AnnCatalogEntry entry;
	Oid			relid;
	Relation	index;
	AnnPredicateColumns columns;
	StringInfoData program;
	StringInfoData payload;
	uint32		status = 0;
	char	   *reply = NULL;
	uint32		replyLength;
	uint64		eligible;
	bytea	   *bitmap;

	if (!LookupCatalog(fcinfo, name, &entry) || entry.generation == 0)
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_OBJECT),
				 errmsg("ann index \"%s\" is not an export of a frozen index in this database's catalog", name)));
	relid = DatumGetObjectId(DirectFunctionCall1(regclassin, CStringGetDatum(entry.source)));
	/* Like reading the table: SELECT on it, and no row-level security. */
	{
		AclResult	aclresult = pg_class_aclcheck(relid, GetUserId(), ACL_SELECT);

		if (aclresult != ACLCHECK_OK)
			aclcheck_error(aclresult, OBJECT_TABLE, get_rel_name(relid));
		if (check_enable_rls(relid, InvalidOid, false) == RLS_ENABLED)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("ann_index_filter does not support row-level security")));
	}
	if ((index = AnnFrozenIndexOf(relid, entry.generation, AccessShareLock)) == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
				 errmsg("%s has no frozen ann index with the generation of \"%s\"", entry.source, name)));
	AnnPredicateColumnsInit(index, &columns);
	initStringInfo(&program);
	if (!AnnPredicateFromText(relid, filter, &columns, &program))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("filter \"%s\" is not a predicate over the index's INCLUDE attributes", filter)));
	index_close(index, AccessShareLock);
	initStringInfo(&payload);
	Put32(&payload, strlen(name));
	appendBinaryStringInfo(&payload, name, strlen(name));
	appendBinaryStringInfo(&payload, program.data, program.len);
	CheckWorker(TaggedCall(fcinfo, name, ANN_OP_INDEX_PREDICATE, payload.data, payload.len, 0, &status, &reply,
						   &replyLength, NULL), status, name);
	if (replyLength < 8 || (replyLength - 8) % 4 != 0)
		elog(ERROR, "malformed ann worker predicate reply");
	memcpy(&eligible, reply, sizeof(eligible));
	bitmap = palloc(VARHDRSZ + replyLength - 8);
	SET_VARSIZE(bitmap, VARHDRSZ + replyLength - 8);
	memcpy(VARDATA(bitmap), reply + 8, replyLength - 8);
	return FilterResult(fcinfo, eligible, bitmap);
}

/*
 * Name of the loaded index exported from a table's frozen index with this
 * generation, the most recently loaded one, or NULL. Read from the catalog
 * in the extension's schema.
 */
char *
AnnCatalogNameFor(Oid relid, uint64 generation)
{
	Oid			types[2] = {OIDOID, TEXTOID};
	Datum		values[2];
	char	   *name = NULL;
	char	   *schema = get_namespace_name(get_extension_schema(get_extension_oid("vector", false)));
	MemoryContext caller = CurrentMemoryContext;

	values[0] = ObjectIdGetDatum(relid);
	values[1] = CStringGetTextDatum(psprintf("%016llx", (unsigned long long) generation));
	if (SPI_connect() != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed");
	if (SPI_execute_with_args(psprintf("SELECT name FROM %s.ann_index_catalog c WHERE source = $1 "
									   "AND to_jsonb(c) ->> 'generation' = $2 ORDER BY loaded_at DESC LIMIT 1",
									   quote_identifier(schema)),
							  2, types, values, NULL, true, 1) == SPI_OK_SELECT && SPI_processed == 1)
	{
		char	   *found = SPI_getvalue(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1);

		name = MemoryContextStrdup(caller, found);
	}
	SPI_finish();
	return name;
}

/*
 * Tombstones (ann_frozen_tombstone): deleted rows of a frozen table, by
 * key, that searches leave out until the table is rebuilt without them.
 * Sent to every worker of this host for a loaded name, in batches; a worker
 * ignores keys its index doesn't hold.
 */
static void
SendTombstones(const char *name, const int64 *keys, int count)
{
	for (int first = 0; first < count; first += ANN_TOMBSTONE_BATCH)
	{
		int			n = Min(count - first, ANN_TOMBSTONE_BATCH);
		StringInfoData payload;
		uint32		length = strlen(name);
		uint32		status = 0;
		char	   *reply = NULL;
		uint32		replyLength = 0;
		bool		ok;

		initStringInfo(&payload);
		appendBinaryStringInfo(&payload, (char *) &length, 4);
		appendBinaryStringInfo(&payload, name, length);
		length = (uint32) n;
		appendBinaryStringInfo(&payload, (char *) &length, 4);
		appendBinaryStringInfo(&payload, (char *) (keys + first), sizeof(int64) * n);
		if (AnnRemoteConfigured())
			ok = AnnRemoteBroadcast(ANN_OP_DELETE_KEYS, payload.data, payload.len, ann_worker_timeout_ms,
									&status, &reply, &replyLength) >= 0 && status == ANN_STATUS_OK;
		else
			ok = AnnWorkerCall(ANN_OP_DELETE_KEYS, payload.data, payload.len, ann_worker_timeout_ms,
							   &status, &reply, &replyLength) && status == ANN_STATUS_OK;
		if (!ok)
			ereport(ERROR,
					(errcode(ERRCODE_CONNECTION_FAILURE),
					 errmsg("could not send deleted rows of ann index \"%s\" to the ann worker (status %u)", name,
							status)));
	}
}

static char *
TombstoneTable(FunctionCallInfo fcinfo)
{
	char	   *schema = get_namespace_name(get_func_namespace(fcinfo->flinfo->fn_oid));

	return psprintf("%s.ann_frozen_tombstone", quote_identifier(schema));
}

/* After a load or attach: the table's tombstones to the loaded name. */
static void
PushTombstones(FunctionCallInfo fcinfo, const char *name, const char *source)
{
	Oid			types[1] = {TEXTOID};
	Datum		values[1] = {CStringGetTextDatum(source)};
	int64	   *keys;
	int			count;

	if (SPI_connect() != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed");
	if (SPI_execute_with_args(psprintf("SELECT key FROM %s WHERE source = $1::regclass", TombstoneTable(fcinfo)),
							  1, types, values, NULL, true, 0) != SPI_OK_SELECT)
		elog(ERROR, "could not read the ann tombstones");
	count = (int) SPI_processed;
	keys = MemoryContextAlloc(TopTransactionContext, sizeof(int64) * Max(count, 1));
	for (int i = 0; i < count; i++)
	{
		bool		isnull;

		keys[i] = DatumGetInt64(SPI_getbinval(SPI_tuptable->vals[i], SPI_tuptable->tupdesc, 1, &isnull));
	}
	SPI_finish();
	if (count > 0)
		SendTombstones(name, keys, count);
}

/*
 * ann_frozen_delete(table, keys): delete rows of a frozen table by key
 * without touching it: they're recorded in ann_frozen_tombstone and sent to
 * every loaded export of the table, and the frozen route leaves them out.
 * Other scans of the table still see them; rebuild the table without them
 * (then clear their tombstones) to remove them for good. Returns how many
 * were newly recorded.
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(ann_frozen_delete);
Datum
ann_frozen_delete(PG_FUNCTION_ARGS)
{
	Oid			table = PG_GETARG_OID(0);
	ArrayType  *array = PG_GETARG_ARRAYTYPE_P(1);
	Oid			types[2] = {OIDOID, INT8ARRAYOID};
	Datum		values[2] = {ObjectIdGetDatum(table), PointerGetDatum(array)};
	int64		recorded;
	List	   *names = NIL;
	Datum	   *elements;
	bool	   *nulls;
	int			count;
	int64	   *keys;
	int			nkeys = 0;
	ListCell   *cell;

	RequireSuperuser("delete from");
	if (!AnnFrozenTableFrozen(table))
		ereport(ERROR,
				(errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
				 errmsg("\"%s\" is not a frozen table", get_rel_name(table)),
				 errhint("Delete from it as usual.")));
	deconstruct_array(array, INT8OID, 8, true, TYPALIGN_DOUBLE, &elements, &nulls, &count);
	keys = palloc(sizeof(int64) * Max(count, 1));
	for (int i = 0; i < count; i++)
		if (!nulls[i])
			keys[nkeys++] = DatumGetInt64(elements[i]);
	if (SPI_connect() != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed");
	if (SPI_execute_with_args(psprintf("INSERT INTO %s (source, key) SELECT $1, k FROM unnest($2) k "
									   "WHERE k IS NOT NULL ON CONFLICT DO NOTHING", TombstoneTable(fcinfo)),
							  2, types, values, NULL, false, 0) != SPI_OK_INSERT)
		elog(ERROR, "could not record the ann tombstones");
	recorded = (int64) SPI_processed;
	if (SPI_execute_with_args(psprintf("SELECT name FROM %s WHERE source = $1", CatalogName(fcinfo)),
							  1, types, values, NULL, false, 0) != SPI_OK_SELECT)
		elog(ERROR, "could not read the ann index catalog");
	for (uint64 i = 0; i < SPI_processed; i++)
		names = lappend(names, MemoryContextStrdup(TopTransactionContext,
												   SPI_getvalue(SPI_tuptable->vals[i], SPI_tuptable->tupdesc, 1)));
	SPI_finish();
	foreach(cell, names)
		SendTombstones(lfirst(cell), keys, nkeys);
	AnnFrozenTombstonesChanged();
	PG_RETURN_INT64(recorded);
}
