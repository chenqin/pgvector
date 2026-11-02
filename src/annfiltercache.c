#include "postgres.h"

#include "ann.h"
#include "common/hashfn.h"
#include "fmgr.h"
#include "funcapi.h"
#include "access/htup_details.h"
#include "miscadmin.h"
#include "utils/guc.h"

#if PG_VERSION_NUM >= 170000
#include "lib/dshash.h"
#include "port/atomics.h"
#include "storage/dsm_registry.h"
#include "storage/lwlock.h"
#include "utils/dsa.h"
#endif

/*
 * Predicate results shared by every backend of the cluster: a frozen index
 * never changes within a generation, so the rows a program selects are
 * evaluated once and copied out by later queries, ann_frozen_filter and
 * ann_filter_bitmap_build. The entries live in a DSA area found through the
 * DSM registry (PostgreSQL 17 and later), so no preloading is needed, and
 * are bounded by ann.frozen_filter_cache_mb with least-recently-used
 * eviction. The key includes the index's generation, so a rebuilt index
 * never sees an older generation's rows.
 */
int			ann_frozen_filter_cache_mb = 4096;

#if PG_VERSION_NUM >= 170000

typedef struct FilterCacheKey
{
	Oid			index;
	uint32		length;			/* program bytes */
	uint64		generation;
	uint64		hash;			/* of the program */
} FilterCacheKey;

typedef struct FilterCacheEntry
{
	FilterCacheKey key;
	dsa_pointer program;		/* the program bytes, compared on a hit */
	dsa_pointer words;
	uint64		nwords;
	uint64		eligible;
	pg_atomic_uint64 used;		/* clock value of the last use */
} FilterCacheEntry;

typedef struct FilterCacheControl
{
	int			tranche;
	dsa_handle	area;
	dshash_table_handle table;
	pg_atomic_uint64 clock;
	pg_atomic_uint64 bytes;
	pg_atomic_uint64 entries;
	pg_atomic_uint64 hits;
	pg_atomic_uint64 misses;
	pg_atomic_uint64 evictions;
	pg_atomic_uint32 evicting;	/* one eviction pass at a time */
} FilterCacheControl;

static FilterCacheControl *control = NULL;
static dsa_area *area = NULL;
static dshash_table *table = NULL;

static dshash_parameters
Parameters(int tranche)
{
	dshash_parameters params = {
		sizeof(FilterCacheKey), sizeof(FilterCacheEntry),
		dshash_memcmp, dshash_memhash, dshash_memcpy, tranche
	};

	return params;
}

static void
InitControl(void *pointer)
{
	FilterCacheControl *init = pointer;
	dsa_area   *newArea;
	dshash_table *newTable;
	dshash_parameters params;

	init->tranche = LWLockNewTrancheId();
	LWLockRegisterTranche(init->tranche, "ann_filter_cache");
	newArea = dsa_create(init->tranche);
	dsa_pin(newArea);
	params = Parameters(init->tranche);
	newTable = dshash_create(newArea, &params, NULL);
	init->area = dsa_get_handle(newArea);
	init->table = dshash_get_hash_table_handle(newTable);
	pg_atomic_init_u64(&init->clock, 0);
	pg_atomic_init_u64(&init->bytes, 0);
	pg_atomic_init_u64(&init->entries, 0);
	pg_atomic_init_u64(&init->hits, 0);
	pg_atomic_init_u64(&init->misses, 0);
	pg_atomic_init_u64(&init->evictions, 0);
	pg_atomic_init_u32(&init->evicting, 0);
	dshash_detach(newTable);
	dsa_detach(newArea);
}

/* Attach this backend to the cache; false when it is disabled. */
static bool
Attach(void)
{
	bool		found;
	dshash_parameters params;
	MemoryContext oldCtx;

	if (ann_frozen_filter_cache_mb <= 0)
		return false;
	if (table)
		return true;
	oldCtx = MemoryContextSwitchTo(TopMemoryContext);
	control = GetNamedDSMSegment("ann_filter_cache", sizeof(FilterCacheControl), InitControl, &found);
	LWLockRegisterTranche(control->tranche, "ann_filter_cache");
	area = dsa_attach(control->area);
	dsa_pin_mapping(area);
	params = Parameters(control->tranche);
	table = dshash_attach(area, &params, control->table, NULL);
	MemoryContextSwitchTo(oldCtx);
	return true;
}

static FilterCacheKey
MakeKey(Oid index, uint64 generation, const char *program, size_t size)
{
	FilterCacheKey key;

	memset(&key, 0, sizeof(key));
	key.index = index;
	key.length = (uint32) size;
	key.generation = generation;
	key.hash = hash_bytes_extended((const unsigned char *) program, (int) size, 0x616e6e5f63616368);
	return key;
}

/* Copy a cached result into words; false on a miss. */
bool
AnnFilterCacheGet(Oid index, uint64 generation, const char *program, size_t size, uint32 *words,
				  size_t nwords, uint64 *eligible)
{
	FilterCacheKey key;
	FilterCacheEntry *entry;
	bool		hit = false;

	if (!Attach())
		return false;
	key = MakeKey(index, generation, program, size);
	entry = dshash_find(table, &key, false);
	if (entry)
	{
		if (entry->nwords == nwords &&
			memcmp(dsa_get_address(area, entry->program), program, size) == 0)
		{
			memcpy(words, dsa_get_address(area, entry->words), nwords * sizeof(uint32));
			*eligible = entry->eligible;
			pg_atomic_write_u64(&entry->used, pg_atomic_add_fetch_u64(&control->clock, 1));
			hit = true;
		}
		dshash_release_lock(table, entry);
	}
	pg_atomic_fetch_add_u64(hit ? &control->hits : &control->misses, 1);
	return hit;
}

typedef struct Victim
{
	FilterCacheKey key;
	uint64		used;
} Victim;

static int
VictimCompare(const void *a, const void *b)
{
	uint64		x = ((const Victim *) a)->used;
	uint64		y = ((const Victim *) b)->used;

	return (x > y) - (x < y);
}

static uint64
EntryBytes(uint64 nwords, size_t length)
{
	return nwords * sizeof(uint32) + length + sizeof(FilterCacheEntry);
}

/* Drop the least recently used entries until the cache is at 90% of its budget. */
static void
Evict(uint64 budget)
{
	dshash_seq_status status;
	FilterCacheEntry *entry;
	Victim	   *victims;
	uint64		n = 0;
	uint64		capacity = 1024;
	uint64		target = budget / 10 * 9;

	if (pg_atomic_exchange_u32(&control->evicting, 1) != 0)
		return;
	victims = palloc(sizeof(Victim) * capacity);
	dshash_seq_init(&status, table, false);
	while ((entry = dshash_seq_next(&status)) != NULL)
	{
		if (n == capacity)
		{
			capacity *= 2;
			victims = repalloc_huge(victims, sizeof(Victim) * capacity);
		}
		victims[n].key = entry->key;
		victims[n].used = pg_atomic_read_u64(&entry->used);
		n++;
	}
	dshash_seq_term(&status);
	qsort(victims, n, sizeof(Victim), VictimCompare);
	for (uint64 i = 0; i < n && pg_atomic_read_u64(&control->bytes) > target; i++)
	{
		dsa_pointer words;
		dsa_pointer program;
		uint64		bytes;

		if ((entry = dshash_find(table, &victims[i].key, true)) == NULL)
			continue;
		words = entry->words;
		program = entry->program;
		bytes = EntryBytes(entry->nwords, entry->key.length);
		dshash_delete_entry(table, entry);
		dsa_free(area, words);
		dsa_free(area, program);
		pg_atomic_fetch_sub_u64(&control->bytes, bytes);
		pg_atomic_fetch_sub_u64(&control->entries, 1);
		pg_atomic_fetch_add_u64(&control->evictions, 1);
	}
	pfree(victims);
	pg_atomic_write_u32(&control->evicting, 0);
}

/*
 * Reserve bytes of the budget, evicting when they don't fit; false when they
 * still don't (another backend is evicting, or the rest is in use). Reserving
 * before allocating keeps concurrent puts from overshooting the budget.
 */
static bool
Reserve(uint64 bytes, uint64 budget)
{
	if (pg_atomic_add_fetch_u64(&control->bytes, bytes) <= budget)
		return true;
	pg_atomic_fetch_sub_u64(&control->bytes, bytes);
	Evict(budget - bytes);
	if (pg_atomic_add_fetch_u64(&control->bytes, bytes) <= budget)
		return true;
	pg_atomic_fetch_sub_u64(&control->bytes, bytes);
	return false;
}

/* Cache a program's result; results larger than the whole budget are not kept. */
void
AnnFilterCachePut(Oid index, uint64 generation, const char *program, size_t size, const uint32 *words,
				  size_t nwords, uint64 eligible)
{
	uint64		budget = (uint64) ann_frozen_filter_cache_mb * 1024 * 1024;
	uint64		bytes = EntryBytes(nwords, size);
	FilterCacheKey key;
	FilterCacheEntry *entry;
	dsa_pointer wordsPointer;
	dsa_pointer programPointer;
	bool		found;

	if (!Attach() || bytes > budget || !Reserve(bytes, budget))
		return;
	wordsPointer = dsa_allocate_extended(area, Max(nwords, 1) * sizeof(uint32), DSA_ALLOC_HUGE | DSA_ALLOC_NO_OOM);
	programPointer = dsa_allocate_extended(area, Max(size, 1), DSA_ALLOC_NO_OOM);
	if (!DsaPointerIsValid(wordsPointer) || !DsaPointerIsValid(programPointer))
	{
		if (DsaPointerIsValid(wordsPointer))
			dsa_free(area, wordsPointer);
		if (DsaPointerIsValid(programPointer))
			dsa_free(area, programPointer);
		pg_atomic_fetch_sub_u64(&control->bytes, bytes);
		return;
	}
	memcpy(dsa_get_address(area, wordsPointer), words, nwords * sizeof(uint32));
	memcpy(dsa_get_address(area, programPointer), program, size);
	key = MakeKey(index, generation, program, size);
	entry = dshash_find_or_insert(table, &key, &found);
	if (found)
	{
		/* someone else cached it first */
		dshash_release_lock(table, entry);
		dsa_free(area, wordsPointer);
		dsa_free(area, programPointer);
		pg_atomic_fetch_sub_u64(&control->bytes, bytes);
		return;
	}
	entry->program = programPointer;
	entry->words = wordsPointer;
	entry->nwords = nwords;
	entry->eligible = eligible;
	pg_atomic_init_u64(&entry->used, pg_atomic_add_fetch_u64(&control->clock, 1));
	dshash_release_lock(table, entry);
	pg_atomic_fetch_add_u64(&control->entries, 1);
}

#else

bool
AnnFilterCacheGet(Oid index, uint64 generation, const char *program, size_t size, uint32 *words,
				  size_t nwords, uint64 *eligible)
{
	return false;
}

void
AnnFilterCachePut(Oid index, uint64 generation, const char *program, size_t size, const uint32 *words,
				  size_t nwords, uint64 eligible)
{
}

#endif

/*
 * ann_frozen_cache(): the shared filter cache's entries, bytes, hits,
 * misses and evictions since the cluster started (zeros when it is off).
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(ann_frozen_cache);
Datum
ann_frozen_cache(PG_FUNCTION_ARGS)
{
	TupleDesc	tupdesc;
	Datum		values[5] = {0};
	bool		nulls[5] = {0};

	if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
		elog(ERROR, "return type must be a row type");
#if PG_VERSION_NUM >= 170000
	if (Attach())
	{
		values[0] = Int64GetDatum((int64) pg_atomic_read_u64(&control->entries));
		values[1] = Int64GetDatum((int64) pg_atomic_read_u64(&control->bytes));
		values[2] = Int64GetDatum((int64) pg_atomic_read_u64(&control->hits));
		values[3] = Int64GetDatum((int64) pg_atomic_read_u64(&control->misses));
		values[4] = Int64GetDatum((int64) pg_atomic_read_u64(&control->evictions));
	}
	else
#endif
		for (int i = 0; i < 5; i++)
			values[i] = Int64GetDatum(0);
	PG_RETURN_DATUM(HeapTupleGetDatum(heap_form_tuple(tupdesc, values, nulls)));
}

void
AnnFilterCacheInit(void)
{
	DefineCustomIntVariable("ann.frozen_filter_cache_mb",
							"Shared memory for predicate results over frozen ann indexes (0 disables).",
							NULL, &ann_frozen_filter_cache_mb, 4096, 0, INT_MAX / 2, PGC_SUSET,
							GUC_UNIT_MB, NULL, NULL, NULL);
}
