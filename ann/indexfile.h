#ifndef ANN_INDEXFILE_H
#define ANN_INDEXFILE_H

#include "backend.h"

#include <pthread.h>

/*
 * Immutable on-disk index produced by scripts/ann/build_cuvs_index.py:
 *
 *   manifest      text, one "key value..." pair per line (see ann/README.md)
 *   keys.i64      rows sorted unique little-endian int64 table keys
 *   vectors.f32   rows x dimensions little-endian FP32 vectors, key order
 *   shard-N.cagra optional cuVS CAGRA graph and dataset for one key range
 *
 * Global ordinal i is the row with the i-th smallest key. Shards cover
 * contiguous ordinal ranges, so key to shard mapping is a binary search.
 */
#define ANN_INDEX_MAX_SHARDS 64
#define ANN_INDEX_NAME_MAX 63
#define ANN_INDEX_MAX_ATTRIBUTES 8
#define ANN_INDEX_MAX_VALUES 64

/*
 * Optional quantized copy of the vectors for Blackwell scans (manifest line
 * "quant nvfp4" or "quant fp8"), next to the exact vectors.f32:
 *
 *   nvfp4: vectors.nvfp4  rows x blocks x 8 bytes, two E2M1 values per byte
 *                         (even dimension in the low nibble), blocks = dims/16
 *                         rounded up; scales.e4m3 rows x blocks E4M3 block
 *                         scales; value = e2m1 * block scale * row scale
 *   fp8:   vectors.e4m3   rows x dims (rounded up to 16) E4M3 bytes;
 *                         value = e4m3 * row scale
 *   both:  rowscale.f32   rows FP32 row scales
 */
#define ANN_QUANT_NONE 0
#define ANN_QUANT_NVFP4 1
#define ANN_QUANT_FP8 2

typedef struct AnnIndexShard
{
	uint64_t	first;			/* first global ordinal */
	uint32_t	rows;
	char		file[256];		/* CAGRA file name inside the directory */
} AnnIndexShard;

/*
 * Rows of one attribute value: a slice of the attribute's key and ordinal
 * arrays, both sorted, and, for values matching many rows, one eligibility
 * bitset per shard, built once at load so graph searches need no per-query
 * bitset work.
 */
typedef struct AnnIndexPosting
{
	int32_t		value;
	uint64_t	offset;
	uint64_t	count;
	uint32_t  **bitsets;		/* nshards bitsets, or NULL */
} AnnIndexPosting;

/* An int32 column stored per row in key order, inverted at load. */
typedef struct AnnIndexAttribute
{
	char		name[64];
	char		file[256];
	uint32_t	nvalues;
	AnnIndexPosting *postings;	/* sorted by value */
	int64_t    *keys;			/* rows grouped by value, key order within */
	uint64_t   *ordinals;
	const int32_t *column;		/* the value of each row, by ordinal */
	const void *mapped;			/* the column file's mapping */
	size_t		columnBytes;
	/* text attributes: the dictionary the codes index (manifest "dictionary") */
	char		stringsFile[256];
	char		startsFile[256];
	uint32_t	nstrings;
	uint32_t   *starts;			/* nstrings + 1 offsets into strings */
	char	   *strings;
	/* AnnIndexRemap: the codes rewritten into a collection's dictionary */
	int32_t    *remap;			/* the map, by part code; NULL until remapped */
	int32_t    *ownedColumn;	/* the remapped column (column points here) */
	AnnIndexPosting *oldPostings;	/* the arrays before the remap: searches */
	int64_t    *oldKeys;		/* that started then may still read them */
	uint64_t   *oldOrdinals;
	uint32_t	oldValues;
} AnnIndexAttribute;

/*
 * A filter bitmap pushed by PostgreSQL (ann_filter_bitmap) and cached under
 * its id: the bitmap over global ordinals, one cuVS bitset per shard
 * (pointing into it when the shard starts on a 32-row boundary), and, for
 * filters the exact routes take, the matching ordinals and keys. Searches
 * hold a reference, so a replaced entry is freed by its last user.
 */
typedef struct AnnIndexBitmap
{
	uint64_t	id;
	int			kind;			/* ANN_BITMAP_PUSHED or ANN_BITMAP_PREDICATE */
	uint64_t	count;
	uint32_t   *words;			/* NULL for a sparse predicate result: its
								 * ordinals are the only form kept */
	uint32_t   *bitsets[ANN_INDEX_MAX_SHARDS];
	uint32_t   *owned[ANN_INDEX_MAX_SHARDS];
	int64_t    *keys;			/* NULL for filters only the graph route takes */
	uint64_t   *ordinals;
	void	  **device;			/* backend copies of the bitsets, or NULL */
	void	  **deviceRows;		/* backend copies of each shard's rows, or
								 * NULL */
	uint64_t	deviceBytes;	/* device memory those copies hold */
	uint64_t	hostBytes;		/* host memory the entry holds */
	uint64_t	used;			/* last use, for predicate eviction */
	int			references;
	struct AnnIndexBitmap *next;
} AnnIndexBitmap;

/*
 * Backend hooks for cached bitmaps: upload copies the per-shard bitsets to
 * device memory before the bitmap is published (failure leaves device NULL,
 * and searches copy them per query); release frees those copies.
 */
typedef struct AnnIndexFile AnnIndexFile;

typedef struct AnnIndexBitmapHooks
{
	void		(*upload) (AnnIndexFile *file, AnnIndexBitmap *bitmap);
	void		(*release) (AnnIndexFile *file, AnnIndexBitmap *bitmap);
} AnnIndexBitmapHooks;

extern AnnIndexBitmapHooks AnnIndexBitmapDevice;

#define ANN_INDEX_BITMAP_BUCKETS 1024

struct AnnIndexFile
{
	char		name[ANN_INDEX_NAME_MAX + 1];
	char		directory[1024];
	char		source[256];	/* table the index was built from, or empty */
	char		key[64];		/* key column name in source, or empty */
	char		vector[64];		/* vector column name in source, or empty */
	char		dtype[16];		/* CAGRA dataset type: float32 or float16 */
	char		generation[17]; /* frozen index content hash, or empty */
	uint64_t	generationValue;	/* the same as a number, 0 if none */
	uint32_t	dimensions;
	uint64_t	rows;
	uint32_t	nshards;
	AnnIndexShard shards[ANN_INDEX_MAX_SHARDS];
	uint32_t	nattributes;
	AnnIndexAttribute attributes[ANN_INDEX_MAX_ATTRIBUTES];
	int			quant;			/* ANN_QUANT_* */
	uint32_t	quantRowBytes;
	uint32_t	quantBlocks;	/* NVFP4 scale blocks per row */
	const uint8_t *quantData;	/* mapped */
	const uint8_t *quantScales;
	const float *rowScales;
	size_t		quantDataBytes;
	size_t		quantScalesBytes;
	size_t		rowScalesBytes;
	int			quantActive;	/* set by a backend scanning the copy */
	pthread_mutex_t bitmapMutex;
	AnnIndexBitmap *bitmaps[ANN_INDEX_BITMAP_BUCKETS];
	uint64_t	nbitmaps;
	uint64_t	bitmapBytes;
	uint64_t	bitmapDeviceBytes;
	uint64_t	clock;			/* stamps predicate entries' last use */
	uint64_t	predicateEntries;	/* cached predicate results */
	uint64_t	predicateBytes; /* host memory they hold */
	uint64_t	predicateDeviceBytes;	/* device memory they hold */
	uint64_t	predicateEvictions;
	uint64_t	pendingBytes;	/* its queued and running searches' charge,
								 * under the scheduler's mutex */
	uint32_t   *deleted;		/* tombstones: a bit per ordinal, set only */
	uint64_t	ndeleted;		/* bits set (atomic reads) */
	const int64_t *keys;		/* mapped */
	const float *vectors;		/* mapped */
	size_t		keysBytes;		/* mapped or reserved bytes, for munmap */
	size_t		vectorsBytes;
	void	   *device;			/* backend-owned device state */
};

/* Worker-wide tuning shared by index backends; set once before serving. */
typedef struct AnnIndexOptions
{
	uint32_t	itopk;			/* CAGRA internal top-k for unfiltered queries */
	uint32_t	filtered_itopk; /* ceiling for filtered queries */
	uint32_t	exact_limit;	/* filtered queries this selective run exact */
	int			persistent;		/* CAGRA persistent kernels */
	int			replicate;		/* full shard set on every GPU */
	uint32_t	gather_limit;	/* filtered queries this selective run a GPU
								 * gather scan */
	uint64_t	bitset_min;		/* attribute values with more rows get
								 * prebuilt bitsets */
	uint32_t	filtered_algo;	/* bitset CAGRA: 0 auto, 1 single, 2 multi CTA */
	uint32_t	filtered_width; /* bitset CAGRA search width, 0 default */
	uint32_t	filtered_iterations;	/* bitset CAGRA max iterations, 0 auto */
	int			device_bitmaps; /* keep cached bitmaps' bitsets on the GPUs */
	uint32_t	memory_pool;	/* RMM pool per GPU, max % of memory; 0 for
								 * stream-ordered allocation */
	uint32_t	bitset_gather_limit;	/* gather limit for filters with a
										 * ready bitset */
	uint32_t	group_gather_limit; /* largest filter a group of same-filter
									 * queries scans together; cached bitmaps
									 * up to it keep their rows */
	uint32_t	index_copies;	/* most GPU copies any loaded index has; set
								 * by the backend, read by the scheduler */
	int			quantized_scan; /* scan quantized copies on Blackwell GPUs */
	uint32_t	quant_slack;	/* extra candidates re-scored in FP32 */
	uint32_t	predicate_cache;	/* most predicate results per index, 0 for
									 * no cap */
	uint64_t	predicate_cache_bytes;	/* host memory for predicate results
										 * per index; 0 for a quarter of RAM */
	uint64_t	predicate_device_bytes; /* device memory for their GPU copies
										 * per index */
	int			prewarm;		/* read index files into memory at load */
	int			lock_memory;	/* mlock them */
	int			plain_allocator;	/* --device-allocator plain: cuVS
									 * allocates with cudaMalloc, no pool
									 * (last: the initializer is positional) */
} AnnIndexOptions;

extern AnnIndexOptions AnnIndexConfig;

const char *AnnIndexQuantName(int quant);

AnnStatus	AnnIndexFileOpen(const char *name, const char *directory, AnnIndexFile **result);
void		AnnIndexFileClose(AnnIndexFile *file);
void		AnnIndexHold(AnnIndexFile *file);
AnnStatus	AnnIndexUnload(const char *name);

/*
 * Text attributes of indexes serving a collection: the part's dictionary
 * (u32 count, count + 1 u32 offsets, the strings' bytes, then u32 1 and the
 * count collection codes if remapped, else u32 0), and rewriting the
 * part's codes into the collection's (map[part code] = collection code, one
 * per dictionary string). A remap is done once, before searches use the
 * index's predicates; repeating it with the same map is allowed.
 */
AnnStatus	AnnIndexDictionary(AnnIndexFile *file, const char *attribute, unsigned char **reply, size_t *length);

/*
 * Tombstones: rows deleted from a frozen index's table, by key, until the
 * index is rebuilt without them. Searches never return them: predicate
 * results leave them out (cached results are dropped when rows are
 * deleted), and hits of every route are checked. Keys the index doesn't
 * hold are ignored. *added counts the rows newly deleted.
 */
AnnStatus	AnnIndexDelete(AnnIndexFile *file, const int64_t *keys, uint32_t count, uint64_t *added);

static inline int
AnnIndexIsDeleted(const AnnIndexFile *file, uint64_t ordinal)
{
	const uint32_t *deleted = __atomic_load_n(&file->deleted, __ATOMIC_ACQUIRE);

	return deleted && ordinal < file->rows &&
		((__atomic_load_n(&deleted[ordinal / 32], __ATOMIC_RELAXED) >> (ordinal % 32)) & 1);
}
AnnStatus	AnnIndexRemap(AnnIndexFile *file, const char *attribute, const int32_t *map, uint32_t count);
int			AnnIndexRemapped(const AnnIndexFile *file);

/* Frees a loaded index's backend state (device memory) when it is unloaded. */
extern void (*AnnIndexDeviceDestroy) (void *device);

/*
 * Evaluate a predicate program (ann/filter.h) over the index's attribute
 * columns and cached bitmaps into words ((rows + 31) / 32 of them).
 * Returns an ANN_FILTER_* code.
 */
void		AnnIndexBitmapWords(const AnnIndexFile *file, const AnnIndexBitmap *bitmap, uint32_t *words);
int			AnnIndexEvaluate(AnnIndexFile *file, const unsigned char *program, size_t size,
							 uint32_t *words, uint64_t *count);

/*
 * Cache a predicate's result under its id like a pushed bitmap. Predicate
 * results share a memory budget per index (predicate_cache_bytes of host
 * memory, predicate_device_bytes of device memory, and optionally at most
 * predicate_cache entries): past it, the least recently used ones are
 * dropped down to 90% of the budget. Pushed bitmaps are never dropped.
 */
AnnStatus	AnnIndexPredicatePut(AnnIndexFile *file, uint64_t id, const uint32_t *words, size_t nwords,
								 uint64_t expandLimit, AnnIndexBitmap **result);

/* Totals over loaded indexes, for the stats reply. */
void		AnnIndexPredicateStats(uint64_t *entries, uint64_t *bytes, uint64_t *deviceBytes,
								   uint64_t *evictions);

/* Registry of loaded indexes; lookups take a reference released by Put. */
AnnStatus	AnnIndexRegister(AnnIndexFile *file);
AnnIndexFile *AnnIndexGet(const char *name);
void		AnnIndexPut(AnnIndexFile *file);

/*
 * Rows matching attribute IN (values): sorted keys and ordinals, and the
 * prebuilt per-shard bitsets when exactly one value has them. A single value
 * returns slices of the index (*owned = 0); several are merged into new
 * arrays the caller frees (*owned = 1).
 */
AnnStatus	AnnIndexMatch(const AnnIndexFile *file, const char *attribute, const int32_t *values,
						  uint32_t nvalues, const int64_t **keys, const uint64_t **ordinals,
						  uint64_t *count, uint32_t *const **bitsets, int *owned);

/*
 * Cache a filter bitmap of (rows + 31) / 32 words under id, replacing any
 * entry with that id. Filters of at most expandLimit rows also get their
 * ordinals and keys.
 */
/*
 * Pushed filters (operation 20, ids PostgreSQL computes) and predicate
 * results (operation 22, ids hashed from the program) are separate kinds:
 * an id names an entry only together with its kind, so a pushed bitmap can
 * never stand in for a predicate's rows. With result, the new entry comes
 * back with a reference the caller releases.
 */
#define ANN_BITMAP_PUSHED 0
#define ANN_BITMAP_PREDICATE 1
AnnStatus	AnnIndexBitmapPut(AnnIndexFile *file, uint64_t id, int kind, const uint32_t *words, size_t nwords,
							  uint64_t expandLimit, AnnIndexBitmap **result);

/* A referenced cached bitmap, or NULL; release it with AnnIndexBitmapRelease. */
AnnIndexBitmap *AnnIndexBitmapGet(AnnIndexFile *file, uint64_t id, int kind);
void		AnnIndexBitmapRelease(AnnIndexFile *file, AnnIndexBitmap *bitmap);

/* Map sorted unique keys to sorted global ordinals; returns the count found. */
uint64_t	AnnIndexOrdinals(const AnnIndexFile *file, const int64_t *keys, uint32_t nkeys,
							 uint64_t *ordinals);

/* Exact search over all rows, or over sorted global ordinals when given. */
void		AnnIndexExact(const AnnIndexFile *file, AnnIndexJob *job,
						  const uint64_t *ordinals, uint64_t nordinals);

/* Merge sorted hit lists pairwise until one top-k list remains; buffers hold k. */
uint32_t	AnnIndexTreeMerge(AnnIndexHit **lists, uint32_t *counts, uint32_t nlists,
							  uint32_t k, AnnIndexHit *output);

#endif
