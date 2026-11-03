#include "postgres.h"

#include "access/table.h"
#include "ann.h"
#include "ann/filter.h"
#include "catalog/pg_type_d.h"
#include "common/hashfn.h"
#include "executor/spi.h"
#include "fmgr.h"
#include "funcapi.h"
#include "lib/stringinfo.h"
#include "miscadmin.h"
#include "portability/instr_time.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "utils/datum.h"
#include "utils/hsearch.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "utils/rel.h"
#include "utils/tuplestore.h"
#include "vector.h"

/*
 * Collections: one table's rows split into parts, each part a loaded index
 * on its own GPU nodes (replicas), searched together by one coordinator
 * call. A part is usually built from the frozen exports of several
 * partitions (members) and holds one CAGRA shard per GPU of its nodes. The
 * coordinator holds no rows: it builds the predicate program once, sends it
 * with the query to one replica of every part that can match (operation 22,
 * all parts in flight at once, over TCP or RDMA), and merges their top k.
 * When a collection is placed by an attribute, each part records the range
 * of that attribute it holds, and a filter that pins the attribute reaches
 * only the parts whose range it can match.
 */

#define ANN_OP_LOAD_INDEX 11
#define ANN_OP_INDEX_INFO 15
#define ANN_OP_UNLOAD_INDEX 26
#define ANN_COLLECTION_LOAD_TIMEOUT_MS (30 * 60 * 1000)
#define ANN_COLLECTION_MAX_PARTS 64
#define ANN_COLLECTION_REFRESH_MS 10000
#define ANN_COLLECTION_PROGRAMS 4096
#define ANN_COLLECTION_NAME 256

typedef struct CollectionPart
{
	int32		part;
	char	   *index;			/* loaded index name on its nodes */
	char	   *nodes;			/* comma-separated host:port replicas */
	uint64		generation;
	bool		ranged;			/* holds placement values low .. high only */
	int64		low;
	int64		high;
} CollectionPart;

typedef struct ProgramEntry
{
	uint64		hash;			/* of the filter text */
	char	   *filter;
	char	   *program;
	int			length;
	SPIPlanPtr	deltaPlan[2];	/* the delta query with this filter, without
								 * and with the attribute columns, or NULL */
} ProgramEntry;

typedef struct Collection
{
	char		name[ANN_COLLECTION_NAME];	/* hash key */
	bool		valid;
	MemoryContext context;		/* everything below */
	MemoryContext programContext;	/* cached programs */
	Oid			source;
	int			nattributes;
	char	  **attributes;
	int			placement;		/* attribute position, or -1 */
	int			nparts;
	CollectionPart *parts;
	AnnPredicateColumns columns;
	HTAB	   *programs;
	int			nprograms;
	Oid			delta;			/* table of rows not yet in a part, or 0 */
	char	   *deltaKey;
	char	   *deltaVector;
	SPIPlanPtr	deltaAll[2];	/* the delta query without a filter */
	instr_time	loaded;
} Collection;

static HTAB *collections = NULL;

typedef struct Hit
{
	int64		key;
	float		distance;
	Datum	   *values;			/* the collection's attributes, when asked */
	bool	   *nulls;
} Hit;

/* Schema-qualified name of a catalog table, in the extension's schema. */
static char *
CatalogTable(FunctionCallInfo fcinfo, const char *table)
{
	char	   *schema = get_namespace_name(get_func_namespace(fcinfo->flinfo->fn_oid));

	return psprintf("%s.%s", quote_identifier(schema), table);
}

static uint64
ParseGenerationText(const char *value)
{
	return value && strlen(value) == 16 ? strtou64(value, NULL, 16) : 0;
}

/* SPI plans live outside the collection's memory: free them before it goes. */
static void
FreeDeltaPlans(Collection *collection)
{
	HASH_SEQ_STATUS status;
	ProgramEntry *entry;

	for (int rows = 0; rows < 2; rows++)
	{
		if (collection->deltaAll[rows])
			SPI_freeplan(collection->deltaAll[rows]);
		collection->deltaAll[rows] = NULL;
	}
	if (collection->programs == NULL)
		return;
	hash_seq_init(&status, collection->programs);
	while ((entry = hash_seq_search(&status)) != NULL)
		for (int rows = 0; rows < 2; rows++)
			if (entry->deltaPlan[rows])
			{
				SPI_freeplan(entry->deltaPlan[rows]);
				entry->deltaPlan[rows] = NULL;
			}
}

static void
CreateProgramCache(Collection *collection)
{
	HASHCTL		ctl;

	memset(&ctl, 0, sizeof(ctl));
	ctl.keysize = sizeof(uint64);
	ctl.entrysize = sizeof(ProgramEntry);
	ctl.hcxt = collection->programContext;
	collection->programs = hash_create("ann collection programs", 256, &ctl,
									   HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);
	collection->nprograms = 0;
}

/* Read a collection's catalog rows into its cache entry. */
static Collection *
LoadCollection(FunctionCallInfo fcinfo, const char *name)
{
	Collection *collection;
	bool		found;
	Oid			types[1] = {TEXTOID};
	Datum		values[1];
	MemoryContext context;
	MemoryContext old;
	Relation	rel;

	if (strlen(name) >= ANN_COLLECTION_NAME)
		ereport(ERROR,
				(errcode(ERRCODE_NAME_TOO_LONG),
				 errmsg("ann collection name is too long")));
	if (collections == NULL)
	{
		HASHCTL		ctl;

		memset(&ctl, 0, sizeof(ctl));
		ctl.keysize = ANN_COLLECTION_NAME;
		ctl.entrysize = sizeof(Collection);
		collections = hash_create("ann collections", 16, &ctl, HASH_ELEM | HASH_STRINGS);
	}

	context = AllocSetContextCreate(CacheMemoryContext, "ann collection", ALLOCSET_SMALL_SIZES);
	values[0] = CStringGetTextDatum(name);
	if (SPI_connect() != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed");
	if (SPI_execute_with_args(psprintf("SELECT source::oid, attributes, placement, delta::oid, delta_key, delta_vector "
									   "FROM %s WHERE name = $1",
									   CatalogTable(fcinfo, "ann_collection")),
							  1, types, values, NULL, true, 1) != SPI_OK_SELECT)
		elog(ERROR, "could not read the ann collection catalog");
	if (SPI_processed != 1)
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_OBJECT),
				 errmsg("ann collection \"%s\" does not exist", name)));

	collection = hash_search(collections, name, HASH_ENTER, &found);
	if (found && collection->valid)
	{
		FreeDeltaPlans(collection);
		MemoryContextDelete(collection->context);
	}
	collection->programs = NULL;
	collection->deltaAll[0] = collection->deltaAll[1] = NULL;
	collection->valid = false;
	collection->context = context;
	old = MemoryContextSwitchTo(context);
	{
		HeapTuple	tuple = SPI_tuptable->vals[0];
		TupleDesc	desc = SPI_tuptable->tupdesc;
		bool		isnull;
		Datum		attributes;
		char	   *placement;
		Datum	   *elements;
		bool	   *nulls;
		int			count;

		collection->source = DatumGetObjectId(SPI_getbinval(tuple, desc, 1, &isnull));
		attributes = SPI_getbinval(tuple, desc, 2, &isnull);
		deconstruct_array(DatumGetArrayTypeP(attributes), TEXTOID, -1, false, TYPALIGN_INT,
						  &elements, &nulls, &count);
		collection->nattributes = count;
		collection->attributes = palloc0(sizeof(char *) * Max(count, 1));
		for (int i = 0; i < count; i++)
			collection->attributes[i] = nulls[i] ? pstrdup("") : TextDatumGetCString(elements[i]);
		placement = SPI_getvalue(tuple, desc, 3);
		{
			bool		deltaNull;
			Datum		delta = SPI_getbinval(tuple, desc, 4, &deltaNull);

			collection->delta = deltaNull ? InvalidOid : DatumGetObjectId(delta);
			collection->deltaKey = SPI_getvalue(tuple, desc, 5);
			collection->deltaVector = SPI_getvalue(tuple, desc, 6);
		}
		collection->placement = -1;
		for (int i = 0; placement && i < count; i++)
			if (strcmp(collection->attributes[i], placement) == 0)
				collection->placement = i;
	}
	MemoryContextSwitchTo(old);

	if (SPI_execute_with_args(psprintf("SELECT part, index_name, nodes, generation, placement_low, placement_high "
									   "FROM %s WHERE collection = $1 ORDER BY part",
									   CatalogTable(fcinfo, "ann_collection_part")),
							  1, types, values, NULL, true, 0) != SPI_OK_SELECT)
		elog(ERROR, "could not read the ann collection catalog");
	if (SPI_processed > ANN_COLLECTION_MAX_PARTS)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("ann collection \"%s\" has more than %d parts", name, ANN_COLLECTION_MAX_PARTS)));
	old = MemoryContextSwitchTo(context);
	collection->nparts = (int) SPI_processed;
	collection->parts = palloc0(sizeof(CollectionPart) * Max(collection->nparts, 1));
	for (int i = 0; i < collection->nparts; i++)
	{
		HeapTuple	tuple = SPI_tuptable->vals[i];
		TupleDesc	desc = SPI_tuptable->tupdesc;
		CollectionPart *part = &collection->parts[i];
		bool		lowNull;
		bool		highNull;
		bool		isnull;
		Datum		low;
		Datum		high;

		part->part = DatumGetInt32(SPI_getbinval(tuple, desc, 1, &isnull));
		part->index = SPI_getvalue(tuple, desc, 2);
		part->nodes = SPI_getvalue(tuple, desc, 3);
		part->generation = ParseGenerationText(SPI_getvalue(tuple, desc, 4));
		low = SPI_getbinval(tuple, desc, 5, &lowNull);
		high = SPI_getbinval(tuple, desc, 6, &highNull);
		part->ranged = collection->placement >= 0 && !lowNull && !highNull;
		part->low = lowNull ? 0 : DatumGetInt32(low);
		part->high = highNull ? 0 : DatumGetInt32(high);
	}
	MemoryContextSwitchTo(old);
	SPI_finish();

	/* Column types come from the table the filters are written against. */
	rel = table_open(collection->source, AccessShareLock);
	old = MemoryContextSwitchTo(context);
	PG_TRY();
	{
		AnnPredicateColumnsForTable(rel, collection->nattributes, collection->attributes, &collection->columns);
	}
	PG_FINALLY();
	{
		MemoryContextSwitchTo(old);
		table_close(rel, AccessShareLock);
	}
	PG_END_TRY();

	collection->programContext = AllocSetContextCreate(context, "ann collection programs", ALLOCSET_DEFAULT_SIZES);
	CreateProgramCache(collection);
	INSTR_TIME_SET_CURRENT(collection->loaded);
	collection->valid = true;
	return collection;
}

static Collection *
GetCollection(FunctionCallInfo fcinfo, const char *name, bool refresh)
{
	Collection *collection = NULL;

	if (collections != NULL && !refresh)
	{
		collection = hash_search(collections, name, HASH_FIND, NULL);
		if (collection && collection->valid)
		{
			instr_time	now;

			INSTR_TIME_SET_CURRENT(now);
			INSTR_TIME_SUBTRACT(now, collection->loaded);
			if (INSTR_TIME_GET_MILLISEC(now) < ANN_COLLECTION_REFRESH_MS)
				return collection;
		}
	}
	return LoadCollection(fcinfo, name);
}

/* Drop this backend's copy of a collection whose catalog it just changed. */
static void
ForgetCollection(const char *name)
{
	Collection *collection;

	if (collections == NULL || strlen(name) >= ANN_COLLECTION_NAME)
		return;
	collection = hash_search(collections, name, HASH_FIND, NULL);
	if (collection == NULL)
		return;
	if (collection->valid)
	{
		FreeDeltaPlans(collection);
		MemoryContextDelete(collection->context);
	}
	hash_search(collections, name, HASH_REMOVE, NULL);
}

/* A filter's program, translated once per backend and cached. */
static void
ProgramFor(Collection *collection, const char *filter, char **program, int *length)
{
	uint64		hash;
	ProgramEntry *entry;
	bool		found;
	StringInfoData built;

	if (filter == NULL || filter[0] == '\0')
	{
		static char all = ANN_FILTER_ALL;

		*program = &all;
		*length = 1;
		return;
	}
	hash = hash_bytes_extended((const unsigned char *) filter, (int) strlen(filter), 0);
	entry = hash_search(collection->programs, &hash, HASH_FIND, NULL);
	if (entry && strcmp(entry->filter, filter) == 0)
	{
		*program = entry->program;
		*length = entry->length;
		return;
	}
	initStringInfo(&built);
	if (!AnnPredicateFromText(collection->source, filter, &collection->columns, &built))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("filter \"%s\" is not a predicate over the attributes of ann collection \"%s\"",
						filter, collection->name),
				 errhint("Use comparisons, IN, BETWEEN, IS NULL, AND, OR and NOT over the collection's attributes with constants.")));
	if (collection->nprograms >= ANN_COLLECTION_PROGRAMS)
	{
		FreeDeltaPlans(collection);
		MemoryContextReset(collection->programContext);
		CreateProgramCache(collection);
	}
	entry = hash_search(collection->programs, &hash, HASH_ENTER, &found);
	if (found && strcmp(entry->filter, filter) != 0)
	{
		/* A hash collision: use the program without caching it. */
		*program = built.data;
		*length = built.len;
		return;
	}
	if (!found)
	{
		entry->deltaPlan[0] = entry->deltaPlan[1] = NULL;
		entry->filter = MemoryContextStrdup(collection->programContext, filter);
		entry->program = MemoryContextAlloc(collection->programContext, Max(built.len, 1));
		memcpy(entry->program, built.data, built.len);
		entry->length = built.len;
		collection->nprograms++;
	}
	pfree(built.data);
	*program = entry->program;
	*length = entry->length;
}

/*
 * The values of the placement attribute a program can match, as sorted
 * disjoint ranges, by running the program over ranges instead of rows:
 * leaves on the placement attribute give their values, every other leaf
 * all values, AND intersects and OR unites. NOT gives all values (the
 * complement of a leaf's rows can hold any value). The answer may include
 * values that match no row, never exclude one that does.
 */
typedef struct RangeSet
{
	bool		all;
	int			count;
	int64	   *low;
	int64	   *high;
} RangeSet;

static RangeSet
RangeAll(void)
{
	RangeSet	set = {true, 0, NULL, NULL};

	return set;
}

static int
RangeCompare(const void *a, const void *b)
{
	int64		x = ((const int64 *) a)[0];
	int64		y = ((const int64 *) b)[0];

	return (x > y) - (x < y);
}

/* Sort and merge ranges given as low, high pairs. */
static RangeSet
RangeNormalize(int64 *pairs, int count)
{
	RangeSet	set = {false, 0, NULL, NULL};

	qsort(pairs, count, sizeof(int64) * 2, RangeCompare);
	set.low = palloc(sizeof(int64) * Max(count, 1));
	set.high = palloc(sizeof(int64) * Max(count, 1));
	for (int i = 0; i < count; i++)
	{
		int64		low = pairs[2 * i];
		int64		high = pairs[2 * i + 1];

		if (low > high)
			continue;
		if (set.count > 0 && low <= set.high[set.count - 1] + 1)
			set.high[set.count - 1] = Max(set.high[set.count - 1], high);
		else
		{
			set.low[set.count] = low;
			set.high[set.count] = high;
			set.count++;
		}
	}
	return set;
}

static RangeSet
RangeUnion(RangeSet a, RangeSet b)
{
	int64	   *pairs;
	int			n = 0;

	if (a.all || b.all)
		return RangeAll();
	pairs = palloc(sizeof(int64) * 2 * Max(a.count + b.count, 1));
	for (int i = 0; i < a.count; i++)
	{
		pairs[n++] = a.low[i];
		pairs[n++] = a.high[i];
	}
	for (int i = 0; i < b.count; i++)
	{
		pairs[n++] = b.low[i];
		pairs[n++] = b.high[i];
	}
	return RangeNormalize(pairs, n / 2);
}

static RangeSet
RangeIntersect(RangeSet a, RangeSet b)
{
	RangeSet	set = {false, 0, NULL, NULL};
	int			i = 0;
	int			j = 0;

	if (a.all)
		return b;
	if (b.all)
		return a;
	set.low = palloc(sizeof(int64) * Max(a.count + b.count, 1));
	set.high = palloc(sizeof(int64) * Max(a.count + b.count, 1));
	while (i < a.count && j < b.count)
	{
		int64		low = Max(a.low[i], b.low[j]);
		int64		high = Min(a.high[i], b.high[j]);

		if (low <= high)
		{
			set.low[set.count] = low;
			set.high[set.count] = high;
			set.count++;
		}
		if (a.high[i] < b.high[j])
			i++;
		else
			j++;
	}
	return set;
}

/* False when the program is malformed (then every part is searched). */
static bool
PlacementRanges(const char *program, int size, const char *placement, RangeSet *result)
{
	RangeSet	stack[ANN_FILTER_MAX_DEPTH + 2];
	int			depth = 0;
	int			at = 0;

	while (at < size)
	{
		uint8		op = (uint8) program[at++];
		RangeSet	set = RangeAll();

		if (op == ANN_FILTER_VALUES || op == ANN_FILTER_RANGE || op == ANN_FILTER_NULL)
		{
			uint8		nameLength;
			const char *name;
			bool		negate;
			bool		mine;

			if (at + 1 > size)
				return false;
			nameLength = (uint8) program[at++];
			if (at + nameLength + 1 > size)
				return false;
			name = program + at;
			at += nameLength;
			negate = program[at++] != 0;
			mine = !negate && strlen(placement) == nameLength && memcmp(name, placement, nameLength) == 0;
			if (op == ANN_FILTER_VALUES)
			{
				uint32		n;
				int64	   *pairs;

				if (at + 4 > size)
					return false;
				memcpy(&n, program + at, 4);
				at += 4;
				if ((size_t) at + (size_t) n * 4 > (size_t) size)
					return false;
				if (mine)
				{
					pairs = palloc(sizeof(int64) * 2 * Max(n, 1));
					for (uint32 v = 0; v < n; v++)
					{
						int32		value;

						memcpy(&value, program + at + v * 4, 4);
						pairs[2 * v] = pairs[2 * v + 1] = value;
					}
					set = RangeNormalize(pairs, (int) n);
				}
				at += (int) n * 4;
			}
			else if (op == ANN_FILTER_RANGE)
			{
				int32		low;
				int32		high;

				if (at + 8 > size)
					return false;
				memcpy(&low, program + at, 4);
				memcpy(&high, program + at + 4, 4);
				at += 8;
				if (mine)
				{
					int64		pair[2] = {low, high};

					set = RangeNormalize(pair, 1);
				}
			}
		}
		else if (op == ANN_FILTER_BITMAP)
			at += 8;
		else if (op == ANN_FILTER_AND || op == ANN_FILTER_OR)
		{
			if (depth < 2)
				return false;
			set = op == ANN_FILTER_AND ? RangeIntersect(stack[depth - 2], stack[depth - 1]) :
				RangeUnion(stack[depth - 2], stack[depth - 1]);
			depth -= 2;
		}
		else if (op == ANN_FILTER_NOT)
		{
			if (depth < 1)
				return false;
			depth--;
		}
		else if (op == ANN_FILTER_NONE)
			set.all = false;
		else if (op != ANN_FILTER_ALL)
			return false;
		if (at > size || depth > ANN_FILTER_MAX_DEPTH)
			return false;
		stack[depth++] = set;
	}
	if (depth != 1)
		return false;
	*result = stack[0];
	return true;
}

static bool
PartMatches(const CollectionPart *part, const RangeSet *ranges)
{
	if (!part->ranged || ranges->all)
		return true;
	for (int i = 0; i < ranges->count; i++)
		if (ranges->low[i] <= part->high && ranges->high[i] >= part->low)
			return true;
	return false;
}

/*
 * The exact search over a collection's delta (rows not yet in a part): the
 * filter as SQL over the delta table, prepared once per filter.
 */
typedef struct DeltaSearch
{
	SPIPlanPtr	plan;
	Datum		query;
	int			k;
	int			nattributes;	/* attribute columns after key and distance */
	MemoryContext context;		/* the hits' values outlive SPI */
	Hit		   *hits;
	int			nhits;
} DeltaSearch;

/* Plans of a filter whose program is not cached. */
static SPIPlanPtr scratchPlan[2] = {NULL, NULL};

/*
 * The delta query for a filter: key, distance and, for rows, the
 * collection's attribute columns, cast to the source table's types.
 */
static SPIPlanPtr
DeltaPlanFor(Collection *collection, const char *filter, Oid vectorType, bool rows)
{
	SPIPlanPtr *slot = &collection->deltaAll[rows];
	bool		scratch = false;
	Oid			types[2] = {vectorType, INT4OID};
	char	   *table;
	StringInfoData columns;
	char	   *statement;
	SPIPlanPtr	plan;

	if (filter && filter[0] != '\0')
	{
		uint64		hash = hash_bytes_extended((const unsigned char *) filter, (int) strlen(filter), 0);
		ProgramEntry *entry = hash_search(collection->programs, &hash, HASH_FIND, NULL);

		scratch = !(entry && strcmp(entry->filter, filter) == 0);
		slot = scratch ? &scratchPlan[rows] : &entry->deltaPlan[rows];
	}
	if (*slot && !scratch)
		return *slot;
	initStringInfo(&columns);
	for (int a = 0; rows && a < collection->nattributes; a++)
		appendStringInfo(&columns, ", (%s)::%s", quote_identifier(collection->attributes[a]),
						 format_type_be(collection->columns.types[a]));
	table = quote_qualified_identifier(get_namespace_name(get_rel_namespace(collection->delta)),
									   get_rel_name(collection->delta));
	statement = psprintf("SELECT (%s)::bigint, (%s <-> $1)::real%s FROM %s%s%s ORDER BY %s <-> $1 LIMIT $2",
						 quote_identifier(collection->deltaKey), quote_identifier(collection->deltaVector),
						 columns.data, table, filter && filter[0] ? " WHERE " : "",
						 filter && filter[0] ? filter : "", quote_identifier(collection->deltaVector));
	if (SPI_connect() != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed");
	if ((plan = SPI_prepare(statement, 2, types)) == NULL)
		elog(ERROR, "could not prepare the delta search of ann collection \"%s\": %s", collection->name,
			 SPI_result_code_string(SPI_result));
	if (SPI_keepplan(plan) != 0)
		elog(ERROR, "could not keep the delta search of ann collection \"%s\"", collection->name);
	SPI_finish();
	if (scratch && *slot)
		SPI_freeplan(*slot);
	*slot = plan;
	return plan;
}

/* AnnRemoteFanoutWhile's callback: the delta's best k while the nodes search. */
static void
DeltaQuery(void *arg)
{
	DeltaSearch *delta = arg;
	Datum		values[2] = {delta->query, Int32GetDatum(delta->k)};

	if (SPI_connect() != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed");
	if (SPI_execute_plan(delta->plan, values, NULL, true, delta->k) != SPI_OK_SELECT)
		elog(ERROR, "the delta search of an ann collection failed");
	for (uint64 i = 0; i < SPI_processed && delta->nhits < delta->k; i++)
	{
		HeapTuple	tuple = SPI_tuptable->vals[i];
		TupleDesc	desc = SPI_tuptable->tupdesc;
		bool		keyNull;
		bool		distanceNull;
		Datum		key = SPI_getbinval(tuple, desc, 1, &keyNull);
		Datum		distance = SPI_getbinval(tuple, desc, 2, &distanceNull);
		Hit		   *hit = &delta->hits[delta->nhits];

		if (keyNull || distanceNull)
			continue;
		hit->key = DatumGetInt64(key);
		hit->distance = DatumGetFloat4(distance);
		hit->values = NULL;
		hit->nulls = NULL;
		if (delta->nattributes > 0)
		{
			MemoryContext old = MemoryContextSwitchTo(delta->context);

			hit->values = palloc(sizeof(Datum) * delta->nattributes);
			hit->nulls = palloc(sizeof(bool) * delta->nattributes);
			for (int a = 0; a < delta->nattributes; a++)
			{
				Form_pg_attribute attr = TupleDescAttr(desc, 2 + a);
				Datum		value = SPI_getbinval(tuple, desc, 3 + a, &hit->nulls[a]);

				hit->values[a] = hit->nulls[a] ? (Datum) 0 : datumCopy(value, attr->attbyval, attr->attlen);
			}
			MemoryContextSwitchTo(old);
		}
		delta->nhits++;
	}
	SPI_finish();
}

static int
HitCompare(const void *a, const void *b)
{
	const Hit  *x = a;
	const Hit  *y = b;

	if (x->distance != y->distance)
		return x->distance < y->distance ? -1 : 1;
	return (x->key > y->key) - (x->key < y->key);
}

static void
InitSetResult(FunctionCallInfo fcinfo, TupleDesc *tupdesc, Tuplestorestate **store)
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

static void
CheckPartStatus(const Collection *collection, const CollectionPart *part, const AnnRemoteRequest *request)
{
	if (!request->ok)
		ereport(ERROR,
				(errcode(ERRCODE_CONNECTION_FAILURE),
				 errmsg("no node of ann collection \"%s\" part %d answered within %d ms",
						collection->name, part->part, ann_worker_timeout_ms),
				 errdetail("Nodes: %s.", part->nodes)));
	if (request->status == ANN_STATUS_OVERLOADED)
		ereport(ERROR,
				(errcode(ERRCODE_INSUFFICIENT_RESOURCES),
				 errmsg("the nodes of ann collection \"%s\" part %d are overloaded", collection->name,
						part->part)));
	if (request->status == ANN_STATUS_UNSUPPORTED)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("the nodes of ann collection \"%s\" part %d do not search by predicate",
						collection->name, part->part)));
	ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
			 errmsg("ann collection \"%s\" part %d (index \"%s\") failed with status %u", collection->name,
					part->part, part->index, request->status)));
}

/*
 * ann_collection_search(collection, query, k, filter): the k nearest rows
 * over every part of a collection that can match the filter, as (key,
 * distance).
 */
/* The column definition list of ann_collection_search_rows: key, distance, then attributes. */
static void
CheckRowsResult(TupleDesc tupdesc)
{
	if (tupdesc->natts < 2 || TupleDescAttr(tupdesc, 0)->atttypid != INT8OID ||
		TupleDescAttr(tupdesc, 1)->atttypid != FLOAT4OID)
		ereport(ERROR,
				(errcode(ERRCODE_DATATYPE_MISMATCH),
				 errmsg("ann_collection_search_rows returns key bigint, distance real, then attribute columns"),
				 errhint("Use a column definition list: AS r(key bigint, distance real, tenant integer, ...).")));
}

/* Each result column after key and distance names an attribute, of its type. */
static int *
RowsMap(const Collection *collection, TupleDesc tupdesc)
{
	int		   *map = palloc0(sizeof(int) * tupdesc->natts);

	for (int c = 2; c < tupdesc->natts; c++)
	{
		Form_pg_attribute attr = TupleDescAttr(tupdesc, c);
		int			a;

		for (a = 0; a < collection->nattributes; a++)
			if (strcmp(collection->attributes[a], NameStr(attr->attname)) == 0)
				break;
		if (a == collection->nattributes)
			ereport(ERROR,
					(errcode(ERRCODE_UNDEFINED_COLUMN),
					 errmsg("\"%s\" is not an attribute of ann collection \"%s\"", NameStr(attr->attname),
							collection->name)));
		if (attr->atttypid != collection->columns.types[a])
			ereport(ERROR,
					(errcode(ERRCODE_DATATYPE_MISMATCH),
					 errmsg("attribute \"%s\" of ann collection \"%s\" is %s, not %s", NameStr(attr->attname),
							collection->name, format_type_be(collection->columns.types[a]),
							format_type_be(attr->atttypid))));
		map[c] = a;
	}
	return map;
}

/*
 * The search behind ann_collection_search and ann_collection_search_rows.
 * With rows, the result columns after key and distance name collection
 * attributes (a column definition list); the nodes return those values with
 * each hit (ANN_PREDICATE_ATTRIBUTES) and the delta query selects them.
 */
static Datum
CollectionSearch(FunctionCallInfo fcinfo, bool rows)
{
	char	   *name;
	Vector	   *query;
	int32		k;
	char	   *filter = NULL;
	TupleDesc	tupdesc;
	Tuplestorestate *store;
	Collection *collection;
	char	   *program;
	int			programLength;
	RangeSet	ranges = RangeAll();
	AnnRemoteRequest requests[ANN_COLLECTION_MAX_PARTS];
	int			selected[ANN_COLLECTION_MAX_PARTS]; /* a request's first part */
	int			members[ANN_COLLECTION_MAX_PARTS];	/* parts per request */
	int			nselected;
	Hit		   *hits = NULL;
	int			nhits = 0;
	int			returned = 0;
	DeltaSearch delta;
	int		   *map = NULL;		/* result column -> attribute */
	int			nattributes = 0;	/* attributes per hit, when rows */

	if (PG_ARGISNULL(0) || PG_ARGISNULL(1) || PG_ARGISNULL(2))
		PG_RETURN_NULL();
	name = text_to_cstring(PG_GETARG_TEXT_PP(0));
	query = PG_GETARG_VECTOR_P(1);
	k = PG_GETARG_INT32(2);
	if (!PG_ARGISNULL(3))
		filter = text_to_cstring(PG_GETARG_TEXT_PP(3));
	if (k < 1 || k > 10000)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("k must be between 1 and 10000")));
	InitSetResult(fcinfo, &tupdesc, &store);
	memset(&delta, 0, sizeof(delta));
	if (rows)
		CheckRowsResult(tupdesc);

	for (int attempt = 0;; attempt++)
	{
		bool		stale = false;

		collection = GetCollection(fcinfo, name, attempt > 0);
		if (rows)
		{
			map = RowsMap(collection, tupdesc);
			nattributes = collection->nattributes;
		}
		ProgramFor(collection, filter, &program, &programLength);
		if (collection->placement >= 0 &&
			!PlacementRanges(program, programLength, collection->attributes[collection->placement], &ranges))
			ranges = RangeAll();

		/*
		 * One request per replica set: the parts that can match, grouped by
		 * their node list, so a node holding several parts answers them in
		 * one request (operation 25); a lone part is operation 22.
		 */
		nselected = 0;
		for (int i = 0; i < collection->nparts; i++)
		{
			int			r;

			if (!PartMatches(&collection->parts[i], &ranges))
				continue;
			for (r = 0; r < nselected; r++)
				if (strcmp(collection->parts[selected[r]].nodes, collection->parts[i].nodes) == 0)
					break;
			if (r == nselected)
			{
				selected[nselected] = i;
				members[nselected++] = 0;
			}
			members[r]++;
		}
		for (int r = 0; r < nselected; r++)
		{
			StringInfoData payload;
			uint32		header[4];

			initStringInfo(&payload);
			if (members[r] > 1)
			{
				uint32		count = (uint32) members[r];

				appendBinaryStringInfo(&payload, (char *) &count, 4);
				for (int i = selected[r]; i < collection->nparts; i++)
				{
					CollectionPart *part = &collection->parts[i];
					uint32		length = strlen(part->index);

					if (!PartMatches(part, &ranges) || strcmp(part->nodes, collection->parts[selected[r]].nodes) != 0)
						continue;
					appendBinaryStringInfo(&payload, (char *) &length, 4);
					appendBinaryStringInfo(&payload, part->index, length);
					appendBinaryStringInfo(&payload, (char *) &part->generation, 8);
				}
			}
			else
			{
				uint32		length = strlen(collection->parts[selected[r]].index);

				appendBinaryStringInfo(&payload, (char *) &length, 4);
				appendBinaryStringInfo(&payload, collection->parts[selected[r]].index, length);
			}
			header[0] = query->dim;
			header[1] = (uint32) k;
			header[2] = rows ? ANN_PREDICATE_ATTRIBUTES : 0;	/* keys, not ordinals */
			header[3] = (uint32) programLength;
			appendBinaryStringInfo(&payload, (char *) header, sizeof(header));
			appendBinaryStringInfo(&payload, program, programLength);
			appendBinaryStringInfo(&payload, (char *) query->x, sizeof(float) * query->dim);
			memset(&requests[r], 0, sizeof(AnnRemoteRequest));
			requests[r].nodes = collection->parts[selected[r]].nodes;
			requests[r].operation = members[r] > 1 ? ANN_OP_COLLECTION_SEARCH : ANN_OP_INDEX_SEARCH_PREDICATE;
			requests[r].generation = members[r] > 1 ? 0 : collection->parts[selected[r]].generation;
			requests[r].payload = payload.data;
			requests[r].length = payload.len;
		}
		if (nselected == 0 && !OidIsValid(collection->delta))
			break;

		/* The delta's exact search runs while the nodes search. */
		memset(&delta, 0, sizeof(delta));
		nhits = 0;
		if (OidIsValid(collection->delta))
		{
			delta.plan = DeltaPlanFor(collection, filter, get_fn_expr_argtype(fcinfo->flinfo, 1), rows);
			delta.query = PointerGetDatum(query);
			delta.k = k;
			delta.nattributes = nattributes;
			delta.context = CurrentMemoryContext;
			delta.hits = palloc(sizeof(Hit) * k);
		}
		(void) AnnRemoteFanoutWhile(requests, nselected, ann_worker_timeout_ms,
									OidIsValid(collection->delta) ? DeltaQuery : NULL, &delta);

		hits = palloc(sizeof(Hit) * ((Size) nselected * k + k));
		nhits = 0;
		for (int r = 0; r < nselected; r++)
		{
			AnnRemoteRequest *request = &requests[r];
			CollectionPart *part = &collection->parts[selected[r]];
			uint32		count;

			if (request->ok && (request->status == ANN_STATUS_GENERATION || request->status == ANN_STATUS_STALE) &&
				attempt == 0)
			{
				/* The nodes moved to another generation: reread the catalog. */
				stale = true;
				continue;
			}
			if (!request->ok || request->status != ANN_STATUS_OK)
				CheckPartStatus(collection, part, request);
			if (request->replyLength < 16 || (count = *(uint32 *) request->reply) > (uint32) k ||
				(Size) count * 12 + 16 + (rows ? 4 + (Size) count * nattributes * 4 : 0) != request->replyLength)
				elog(ERROR, "malformed reply from ann collection \"%s\" part %d", name, part->part);
			if (rows && *(uint32 *) (request->reply + 16 + (Size) count * 12) != (uint32) nattributes)
				ereport(ERROR,
						(errcode(ERRCODE_DATA_CORRUPTED),
						 errmsg("ann collection \"%s\" part %d has %u attributes on its nodes, not %d",
								name, part->part, *(uint32 *) (request->reply + 16 + (Size) count * 12),
								nattributes)));
			for (uint32 i = 0; i < count; i++)
			{
				memcpy(&hits[nhits].key, request->reply + 16 + (Size) i * 12, 8);
				memcpy(&hits[nhits].distance, request->reply + 24 + (Size) i * 12, 4);
				hits[nhits].values = NULL;
				hits[nhits].nulls = NULL;
				if (rows)
				{
					const char *codes = request->reply + 20 + (Size) count * 12 + (Size) i * nattributes * 4;

					hits[nhits].values = palloc(sizeof(Datum) * Max(nattributes, 1));
					hits[nhits].nulls = palloc(sizeof(bool) * Max(nattributes, 1));
					for (int a = 0; a < nattributes; a++)
					{
						int32		code;

						memcpy(&code, codes + (Size) a * 4, 4);
						hits[nhits].nulls[a] = code == ANN_FROZEN_NULL;
						hits[nhits].values[a] = hits[nhits].nulls[a] ? (Datum) 0 :
							AnnFrozenAttributeDatum(code, collection->columns.types[a]);
					}
				}
				nhits++;
			}
		}
		if (!stale)
			break;
	}

	for (int i = 0; i < delta.nhits; i++)
		hits[nhits++] = delta.hits[i];	/* room for k was allocated */
	if (nhits > 1)
		qsort(hits, nhits, sizeof(Hit), HitCompare);

	/*
	 * A row moving from the delta into a rebuilt part can briefly be in both:
	 * return each key once, at its nearest.
	 */
	for (int i = 0; i < nhits && returned < k; i++)
	{
		Datum	   *values = palloc(sizeof(Datum) * tupdesc->natts);
		bool	   *nulls = palloc0(sizeof(bool) * tupdesc->natts);
		bool		seen = false;

		for (int j = 0; j < i && !seen; j++)
			seen = hits[j].key == hits[i].key;
		if (seen)
			continue;
		values[0] = Int64GetDatum(hits[i].key);
		values[1] = Float4GetDatum(hits[i].distance);
		for (int c = 2; rows && c < tupdesc->natts; c++)
		{
			values[c] = hits[i].values[map[c]];
			nulls[c] = hits[i].nulls[map[c]];
		}
		tuplestore_putvalues(store, tupdesc, values, nulls);
		returned++;
	}
	return (Datum) 0;
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(ann_collection_search);
Datum
ann_collection_search(PG_FUNCTION_ARGS)
{
	return CollectionSearch(fcinfo, false);
}

/*
 * ann_collection_search_rows(collection, query, k, filter): as
 * ann_collection_search, with the hits' attribute values from the nodes, so
 * a query needing only attributes doesn't fetch rows from the shard hosts:
 *
 *   SELECT * FROM ann_collection_search_rows('docs', $1, 10, 'tenant = 7')
 *       AS r(key bigint, distance real, category integer, tenant integer);
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(ann_collection_search_rows);
Datum
ann_collection_search_rows(PG_FUNCTION_ARGS)
{
	return CollectionSearch(fcinfo, true);
}

/*
 * ann_collection_create(name, source, attributes, placement): a collection
 * over a table (on a coordinator, usually partitioned with foreign
 * partitions). Filters are written against source; attributes name the
 * columns the parts' indexes were built with; placement, when set, is the
 * attribute the parts are split by.
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(ann_collection_create);
Datum
ann_collection_create(PG_FUNCTION_ARGS)
{
	text	   *name;
	Oid			source;
	ArrayType  *attributes;
	char	   *placement = PG_ARGISNULL(3) ? NULL : text_to_cstring(PG_GETARG_TEXT_PP(3));
	Datum	   *elements;
	bool	   *nulls;
	int			count;
	char	  **names;
	bool		placed = placement == NULL;
	Relation	rel;
	AnnPredicateColumns columns;
	Oid			types[4] = {TEXTOID, OIDOID, TEXTARRAYOID, TEXTOID};
	Datum		values[4];
	char		valueNulls[4] = {' ', ' ', ' ', placement ? ' ' : 'n'};

	if (PG_ARGISNULL(0) || PG_ARGISNULL(1) || PG_ARGISNULL(2))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("name, source and attributes are required")));
	name = PG_GETARG_TEXT_PP(0);
	source = PG_GETARG_OID(1);
	attributes = PG_GETARG_ARRAYTYPE_P(2);
	deconstruct_array(attributes, TEXTOID, -1, false, TYPALIGN_INT, &elements, &nulls, &count);
	names = palloc(sizeof(char *) * Max(count, 1));
	for (int i = 0; i < count; i++)
	{
		if (nulls[i])
			ereport(ERROR,
					(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
					 errmsg("ann collection attributes cannot be null")));
		names[i] = TextDatumGetCString(elements[i]);
		placed |= placement && strcmp(names[i], placement) == 0;
	}
	if (!placed)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("placement attribute \"%s\" is not one of the collection's attributes", placement)));
	rel = table_open(source, AccessShareLock);
	AnnPredicateColumnsForTable(rel, count, names, &columns);
	table_close(rel, AccessShareLock);

	values[0] = PointerGetDatum(name);
	values[1] = ObjectIdGetDatum(source);
	values[2] = PointerGetDatum(attributes);
	values[3] = placement ? CStringGetTextDatum(placement) : (Datum) 0;
	if (SPI_connect() != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed");
	if (SPI_execute_with_args(psprintf("INSERT INTO %s (name, source, attributes, placement) "
									   "VALUES ($1, $2::oid::regclass, $3, $4)",
									   CatalogTable(fcinfo, "ann_collection")),
							  4, types, values, valueNulls, false, 0) != SPI_OK_INSERT)
		elog(ERROR, "could not record the ann collection");
	SPI_finish();
	PG_RETURN_VOID();
}

/* Ask one node for a loaded index's rows and generation (operation 15). */
static void
IndexInfoOn(const char *node, const char *index, uint64 *rows, char *generation)
{
	AnnRemoteRequest request;
	unsigned long long count;
	const char *at;

	memset(&request, 0, sizeof(request));
	request.nodes = node;
	request.operation = ANN_OP_INDEX_INFO;
	request.payload = index;
	request.length = strlen(index);
	(void) AnnRemoteFanout(&request, 1, ann_worker_timeout_ms);
	if (!request.ok)
		ereport(ERROR,
				(errcode(ERRCODE_CONNECTION_FAILURE),
				 errmsg("ann node %s is unavailable", node)));
	if (request.status != ANN_STATUS_OK)
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_OBJECT),
				 errmsg("ann index \"%s\" is not loaded on node %s (status %u)", index, node, request.status)));
	if (sscanf(request.reply, "rows %llu", &count) != 1)
		elog(ERROR, "unexpected ann node reply: %s", request.reply);
	*rows = count;
	strcpy(generation, "-");
	if ((at = strstr(request.reply, " generation ")) != NULL && sscanf(at, " generation %16s", generation) != 1)
		elog(ERROR, "unexpected ann node reply: %s", request.reply);
}

/*
 * ann_collection_load(collection, part, index_name, path, nodes,
 * placement_low, placement_high): load a part's index directory on every
 * node of its replica list (or, with a NULL path, attach an index already
 * loaded there), check that all of them serve the same generation, and
 * record the part. Returns the part's rows.
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(ann_collection_load);
Datum
ann_collection_load(PG_FUNCTION_ARGS)
{
	text	   *collection;
	int32		part;
	char	   *index;
	char	   *nodes;
	char		generation[17] = "-";
	uint64		rows = 0;
	char	   *list;
	char	   *save = NULL;
	int			nnodes = 0;
	Oid			types[7] = {TEXTOID, INT4OID, TEXTOID, TEXTOID, TEXTOID, INT4OID, INT4OID};
	Datum		values[7];
	char		valueNulls[7] = {' ', ' ', ' ', ' ', ' ', ' ', ' '};

	if (PG_ARGISNULL(0) || PG_ARGISNULL(1) || PG_ARGISNULL(2) || PG_ARGISNULL(4))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("collection, part, index name and nodes are required")));
	collection = PG_GETARG_TEXT_PP(0);
	part = PG_GETARG_INT32(1);
	index = text_to_cstring(PG_GETARG_TEXT_PP(2));
	nodes = text_to_cstring(PG_GETARG_TEXT_PP(4));
	if (PG_ARGISNULL(5) != PG_ARGISNULL(6))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("give both placement_low and placement_high, or neither")));

	if (!PG_ARGISNULL(3))
	{
		char	   *path = text_to_cstring(PG_GETARG_TEXT_PP(3));
		StringInfoData payload;
		uint32		length = strlen(index);
		uint32		status = 0;
		char	   *reply = NULL;
		uint32		replyLength = 0;

		initStringInfo(&payload);
		appendBinaryStringInfo(&payload, (char *) &length, 4);
		appendBinaryStringInfo(&payload, index, length);
		appendBinaryStringInfo(&payload, path, strlen(path));
		if (AnnRemoteBroadcastTo(nodes, ANN_OP_LOAD_INDEX, payload.data, payload.len,
								 ANN_COLLECTION_LOAD_TIMEOUT_MS, &status, &reply, &replyLength) < 0)
			ereport(ERROR,
					(errcode(ERRCODE_CONNECTION_FAILURE),
					 errmsg("could not load \"%s\" as ann index \"%s\" on nodes %s (status %u)", path, index,
							nodes, status),
					 errhint("A node answers status 3 when an index of that name is already loaded.")));
	}

	/* Every replica must serve the same generation. */
	list = pstrdup(nodes);
	for (char *node = strtok_r(list, ", ", &save); node; node = strtok_r(NULL, ", ", &save))
	{
		uint64		nodeRows;
		char		nodeGeneration[17];

		IndexInfoOn(node, index, &nodeRows, nodeGeneration);
		if (nnodes > 0 && (nodeRows != rows || strcmp(nodeGeneration, generation) != 0))
			ereport(ERROR,
					(errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
					 errmsg("the nodes of ann collection part %d serve different versions of \"%s\"", part, index)));
		rows = nodeRows;
		strcpy(generation, nodeGeneration);
		nnodes++;
	}
	if (nnodes == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("no nodes given for ann collection part %d", part)));

	values[0] = PointerGetDatum(collection);
	values[1] = Int32GetDatum(part);
	values[2] = CStringGetTextDatum(index);
	values[3] = CStringGetTextDatum(nodes);
	values[4] = CStringGetTextDatum(generation);
	values[5] = PG_ARGISNULL(5) ? (Datum) 0 : PG_GETARG_DATUM(5);
	values[6] = PG_ARGISNULL(6) ? (Datum) 0 : PG_GETARG_DATUM(6);
	if (PG_ARGISNULL(5))
		valueNulls[5] = valueNulls[6] = 'n';
	if (SPI_connect() != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed");
	if (SPI_execute_with_args(psprintf("INSERT INTO %s (collection, part, index_name, nodes, generation, rows, "
									   "placement_low, placement_high) VALUES ($1, $2, $3, $4, $5, %llu, $6, $7) "
									   "ON CONFLICT (collection, part) DO UPDATE SET index_name = excluded.index_name, "
									   "nodes = excluded.nodes, generation = excluded.generation, rows = excluded.rows, "
									   "placement_low = excluded.placement_low, placement_high = excluded.placement_high, "
									   "loaded_at = now()",
									   CatalogTable(fcinfo, "ann_collection_part"), (unsigned long long) rows),
							  7, types, values, valueNulls, false, 0) != SPI_OK_INSERT)
		elog(ERROR, "could not record ann collection part %d", part);
	SPI_finish();
	ForgetCollection(text_to_cstring(collection));
	PG_RETURN_INT64((int64) rows);
}

/*
 * ann_collection_delta(collection, delta, key_column, vector_column): the
 * table holding a collection's rows not yet frozen into a part (NULL for
 * none). Searches score it exactly with the same filter, while the nodes
 * search, and merge it in. Keep it small: re-freeze its rows into their
 * partitions and rebuild those parts on a schedule.
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(ann_collection_delta);
Datum
ann_collection_delta(PG_FUNCTION_ARGS)
{
	Oid			types[4] = {TEXTOID, OIDOID, TEXTOID, TEXTOID};
	Datum		values[4];
	char		nulls[4] = {' ', ' ', ' ', ' '};

	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("collection is required")));
	values[0] = PG_GETARG_DATUM(0);
	values[1] = PG_ARGISNULL(1) ? (Datum) 0 : PG_GETARG_DATUM(1);
	values[2] = PG_ARGISNULL(2) ? CStringGetTextDatum("id") : PG_GETARG_DATUM(2);
	values[3] = PG_ARGISNULL(3) ? CStringGetTextDatum("embedding") : PG_GETARG_DATUM(3);
	if (PG_ARGISNULL(1))
		nulls[1] = nulls[2] = nulls[3] = 'n';
	if (SPI_connect() != SPI_OK_CONNECT)
		elog(ERROR, "SPI_connect failed");
	if (SPI_execute_with_args(psprintf("UPDATE %s SET delta = $2::oid::regclass, delta_key = $3, delta_vector = $4 "
									   "WHERE name = $1", CatalogTable(fcinfo, "ann_collection")),
							  4, types, values, nulls, false, 0) != SPI_OK_UPDATE)
		elog(ERROR, "could not record the delta of the ann collection");
	if (SPI_processed != 1)
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_OBJECT),
				 errmsg("ann collection \"%s\" does not exist", TextDatumGetCString(values[0]))));
	SPI_finish();
	ForgetCollection(TextDatumGetCString(values[0]));
	PG_RETURN_VOID();
}

/*
 * ann_index_unload(index_name, nodes): unload a loaded index from every
 * node of a host:port list (ann.worker_address when NULL), as after a
 * collection part was replaced: each node frees it once the searches using
 * it are done. Returns the nodes that had it loaded.
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(ann_index_unload);
Datum
ann_index_unload(PG_FUNCTION_ARGS)
{
	char	   *index;
	char	   *nodes;
	char	   *save = NULL;
	StringInfoData payload;
	uint32		length;
	int32		unloaded = 0;

	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("index name is required")));
	index = text_to_cstring(PG_GETARG_TEXT_PP(0));
	nodes = PG_ARGISNULL(1) ? pstrdup(ann_worker_address ? ann_worker_address : "") :
		text_to_cstring(PG_GETARG_TEXT_PP(1));
	length = strlen(index);
	initStringInfo(&payload);
	appendBinaryStringInfo(&payload, (char *) &length, 4);
	appendBinaryStringInfo(&payload, index, length);
	for (char *node = strtok_r(nodes, ", ", &save); node; node = strtok_r(NULL, ", ", &save))
	{
		AnnRemoteRequest request;

		memset(&request, 0, sizeof(request));
		request.nodes = node;
		request.operation = ANN_OP_UNLOAD_INDEX;
		request.payload = payload.data;
		request.length = payload.len;
		(void) AnnRemoteFanout(&request, 1, ann_worker_timeout_ms);
		if (!request.ok)
			ereport(WARNING,
					(errmsg("ann node %s is unavailable; index \"%s\" stays loaded there", node, index)));
		else if (request.status == ANN_STATUS_OK)
			unloaded++;
	}
	PG_RETURN_INT32(unloaded);
}
