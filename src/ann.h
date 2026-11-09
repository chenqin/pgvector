#ifndef ANN_H
#define ANN_H

#include "postgres.h"

#include "access/amapi.h"
#include "access/genam.h"
#include "access/itup.h"
#include "access/relscan.h"
#include "nodes/execnodes.h"
#include "nodes/extensible.h"
#include "storage/bufmgr.h"
#include "utils/rel.h"
#include "catalog/pg_type_d.h"
#include "vector.h"

#define ANN_MAGIC_NUMBER 0xA991A991
#define ANN_VERSION 1			/* mutable layout: index tuples */
#define ANN_FROZEN_VERSION 5	/* frozen layout: arrays by ordinal; 3 adds
								 * bitset pages and text dictionaries to 2, 4
								 * the attribute names, 5 integer[] attributes */
#define ANN_FROZEN_VERSION_SINGLE 4 /* written when no attribute is an array */
#define AnnVersionFrozen(version) ((version) >= 2 && (version) <= ANN_FROZEN_VERSION)
#define ANN_PAGE_ID 0xFF92
#define ANN_METAPAGE_BLKNO 0
#define ANN_MAX_DIM 2000
#define ANN_DISTANCE_PROC 1
#define ANN_NPROCS 1

typedef struct AnnPageOpaqueData
{
	uint16		page_id;
} AnnPageOpaqueData;

typedef struct AnnMetaPageData
{
	uint32		magicNumber;
	uint32		version;
	uint32		dimensions;
	BlockNumber insertPage;
} AnnMetaPageData;

#define AnnPageGetMeta(page) ((AnnMetaPageData *) PageGetContents(page))
#define AnnPageGetOpaque(page) ((AnnPageOpaqueData *) PageGetSpecialPointer(page))

typedef struct AnnOptions
{
	int32		vl_len_;		/* varlena header (do not touch directly!) */
	bool		frozen;			/* build the immutable layout */
	int			bitsetMin;		/* values with more rows get bitset pages; -1
								 * for rows / 32 */
	int			vectors;		/* frozen: ANN_FROZEN_VECTORS_* */
} AnnOptions;

/*
 * How a frozen index keeps the vectors (reloption vectors): full vector
 * values, FP16 halves (2 bytes a dimension, for the CPU routes only: exports
 * and returned columns read the heap's exact vectors), or none, read from
 * the heap by TID. The vector array's width tells them apart: the vector
 * size, 2 * dims, or ANN_FROZEN_HEAP_WIDTH with no pages.
 */
#define ANN_FROZEN_VECTORS_FLOAT 0
#define ANN_FROZEN_VECTORS_HALF 1
#define ANN_FROZEN_VECTORS_HEAP 2
#define ANN_FROZEN_HEAP_WIDTH 1

/*
 * Frozen indexes (WITH (frozen = true)) are built once from a table that
 * cannot change, and store every structure as a fixed-width array over
 * consecutive pages. Element i of an array lives on block
 * start + i / perPage at offset (i % perPage) * width of the page contents.
 * Ordinal i is the row with the i-th smallest key, the numbering the GPU
 * worker uses for the same rows.
 */
typedef struct AnnArray
{
	BlockNumber start;
	BlockNumber pages;
	uint32		width;			/* bytes per element */
	uint32		perPage;
	uint64		count;
} AnnArray;

/*
 * An INCLUDE column after the key: forward values and its inverted index.
 * For an integer[] column, values holds every row's elements in ordinal
 * order (each row's as stored, a NULL array as one ANN_FROZEN_NULL), the
 * extra directory's starts where each row's begin, and the postings the
 * distinct (value, ordinal) pairs, so a row is under each of its values.
 */
typedef struct AnnFrozenColumn
{
	AnnArray	values;			/* int32 per ordinal, ANN_FROZEN_NULL for NULL */
	AnnArray	postings;		/* uint32 ordinals grouped by value, ascending */
	AnnArray	dict;			/* distinct int32 values, ascending */
	AnnArray	offsets;		/* uint32 per value plus one: its postings */
} AnnFrozenColumn;

/*
 * Version 3 directory per column: prebuilt bitsets of its most frequent
 * values, and for text columns the dictionary its int32 codes index.
 */
typedef struct AnnFrozenColumnExtra
{
	AnnArray	codes;			/* uint32 distinct-value positions, ascending */
	AnnArray	words;			/* (rows + 31) / 32 uint32 words per code */
	AnnArray	strings;		/* text: the distinct strings' bytes, in
								 * collation order (code 0 first) */
	AnnArray	starts;			/* text: uint32 offset of each code's string,
								 * plus the end; integer[]: of each row's
								 * first element, plus the end */
} AnnFrozenColumnExtra;

typedef struct AnnFrozenMetaData
{
	uint64		rows;
	uint64		generation;		/* hash of the indexed content */
	Oid			guard;			/* guard trigger at build time */
	uint32		ncolumns;		/* INCLUDE columns after the key */
	AnnArray	tids;			/* ItemPointerData per ordinal */
	AnnArray	keys;			/* int64 per ordinal, strictly increasing */
	AnnArray	vectors;		/* Vector varlena per ordinal, or halves, or
								 * none (see ANN_FROZEN_VECTORS_*) */
	AnnFrozenColumn columns[FLEXIBLE_ARRAY_MEMBER];
} AnnFrozenMetaData;

#define ANN_FROZEN_MAX_COLUMNS (INDEX_MAX_KEYS - 2)
#define ANN_FROZEN_NULL PG_INT32_MIN
#define AnnPageGetFrozen(page) \
	((AnnFrozenMetaData *) ((char *) PageGetContents(page) + MAXALIGN(sizeof(AnnMetaPageData))))
#define AnnFrozenMetaSizeV2(ncolumns) \
	(offsetof(AnnFrozenMetaData, columns) + (ncolumns) * sizeof(AnnFrozenColumn))
#define AnnFrozenMetaSizeV3(ncolumns) \
	(AnnFrozenMetaSizeV2(ncolumns) + (ncolumns) * sizeof(AnnFrozenColumnExtra))
#define AnnFrozenMetaSize(ncolumns) (AnnFrozenMetaSizeV3(ncolumns) + (ncolumns) * sizeof(NameData))
/* The extra directory follows the columns, then the attribute names. */
#define AnnFrozenGetExtra(meta) ((AnnFrozenColumnExtra *) &(meta)->columns[(meta)->ncolumns])
/*
 * Attribute names as of the build: predicate programs and exports use them,
 * so a later column rename can't make a program name another column.
 */
#define AnnFrozenGetNames(meta) ((NameData *) &AnnFrozenGetExtra(meta)[(meta)->ncolumns])

/*
 * Frozen metapages whose build flushed every page to the files say so in
 * insertPage (unused by frozen indexes); only those are read through
 * read-only mappings of the files. VACUUM's first TID invalidation turns
 * PRISTINE into FLUSHED, after which the table is never immutable again.
 */
#define ANN_FROZEN_FLUSHED ((BlockNumber) 0xFFFFFFFE)
/* Flushed build whose TIDs have never been invalidated by VACUUM. */
#define ANN_FROZEN_PRISTINE ((BlockNumber) 0xFFFFFFFD)

struct AnnFrozenMap;

/*
 * Reads of an array, one page at a time: through a read-only mapping of the
 * index files when the array never changes (everything but the TIDs), else
 * pinned and share-locked in shared buffers.
 */
typedef struct AnnArrayReader
{
	Relation	index;
	const AnnArray *array;
	BufferAccessStrategy strategy;
	Buffer		buf;
	BlockNumber page;			/* page of the array pinned in buf */
	bool		locked;
	const struct AnnFrozenMap *map;
} AnnArrayReader;

extern const CustomPathMethods AnnCustomPathMethods;
extern const CustomScanMethods AnnCustomScanMethods;
void		AnnInit(void);

extern char *ann_worker_socket;
extern int	ann_worker_chunk_rows;
extern int	ann_worker_pipeline_depth;
extern int	ann_worker_timeout_ms;
extern int	ann_worker_overload_wait_ms;
extern int	ann_worker_shared_memory_mb;
void		AnnWorkerInit(void);
bool		AnnWorkerBegin(void);
void		AnnWorkerEnd(void);
float	   *AnnWorkerSharedBuffer(size_t *size);
uint64		AnnWorkerSubmit(uint32 rows, uint32 dimensions, uint32 k,
							const float *query, const float *vectors);
bool		AnnWorkerReady(void);
extern char *ann_worker_address;
extern char *ann_worker_auth_token;
extern int	ann_gpu_min_eligible;
extern int	ann_worker_heartbeat_ms;
extern int	ann_worker_retry_ms;
extern char *ann_worker_rdma;
extern int	ann_worker_rdma_spin_us;
void		AnnRemoteInit(void);
bool		AnnRemoteConfigured(void);
int			AnnRemoteNodeCount(void);
void		AnnRemoteNextNode(void);
bool		AnnRemoteCall(uint16 operation, const char *payload, size_t length, int timeoutMs,
						  uint32 *status, char **reply, uint32 *replyLength);
int			AnnRemoteBroadcast(uint16 operation, const char *payload, size_t length, int timeoutMs,
							   uint32 *status, char **reply, uint32 *replyLength);

/* One request of a fan-out (AnnRemoteFanout): to one of the replicas in nodes. */
typedef struct AnnRemoteRequest
{
	const char *nodes;			/* comma-separated host:port replicas */
	uint16		operation;
	uint64		generation;		/* request header generation */
	const char *payload;
	size_t		length;
	/* results */
	bool		ok;
	uint32		status;
	char	   *reply;
	uint32		replyLength;
	/* internal */
	bool		done;
	uint64		tried;			/* replicas tried, by position */
	uint64		rotation;
} AnnRemoteRequest;

int			AnnRemoteFanout(AnnRemoteRequest *requests, int count, int timeoutMs);
int			AnnRemoteFanoutWhile(AnnRemoteRequest *requests, int count, int timeoutMs,
								 void (*whileWaiting) (void *), void *arg);
int			AnnRemoteBroadcastTo(const char *nodes, uint16 operation, const char *payload, size_t length,
								 int timeoutMs, uint32 *status, char **reply, uint32 *replyLength);
int			AnnRemoteBroadcastEach(uint16 operation, const char *payload, size_t length, int timeoutMs,
								   int *failed, uint32 *status);
bool		AnnWorkerCall(uint16 operation, const char *payload, size_t length, int timeoutMs,
						  uint32 *status, char **reply, uint32 *replyLength);
bool		AnnWorkerReceive(uint64 *request, uint32 *status, uint32 **ids, uint32 *count);

Oid			AnnVectorTypeOid(void);
void		AnnCheckPage(Page page);
void		AnnCheckIndex(Relation index);
void		AnnInitMetaPage(Relation index, ForkNumber forkNum);
void		AnnCheckMetaPage(Page page);
void		AnnInsertTuple(Relation index, Datum *values, bool *isnull,
						   ItemPointer tid, bool building);

void		AnnInitOptions(void);
bool		AnnFrozenOption(Relation index);
bool		AnnIsFrozen(Relation index);
AnnFrozenMetaData *AnnFrozenGetMeta(Relation index, bool missingOk);
AnnFrozenMetaData *AnnFrozenMeta(Relation index);
bool		AnnFrozenFlushed(Relation index);
IndexBuildResult *AnnFrozenBuild(Relation heap, Relation index, IndexInfo *indexInfo);
void		AnnArrayReaderInit(AnnArrayReader *reader, Relation index, const AnnArray *array,
							   BufferAccessStrategy strategy);
void		AnnArrayReaderInitExtended(AnnArrayReader *reader, Relation index, const AnnArray *array,
									   BufferAccessStrategy strategy, bool immutable);
bool		AnnFrozenImmutable(Relation heap, Relation index);
bool		AnnFrozenIndexImmutable(Relation index);
char	   *AnnArrayLockPage(AnnArrayReader *reader, BlockNumber page, uint32 *count);
void		AnnArrayUnlockPage(AnnArrayReader *reader);
void		AnnArrayRead(AnnArrayReader *reader, uint64 element, void *out);
void		AnnArrayPrefetch(AnnArrayReader *reader, uint64 element);
void		AnnArrayReaderEnd(AnnArrayReader *reader);
void		AnnFrozenReadTid(AnnArrayReader *reader, Relation heap, uint64 ordinal,
							 ItemPointerData *tid, bool *visible, Buffer *vmbufs, int nvmbufs);
bool		AnnFrozenVisibleTids(Relation index, Relation heap, const AnnFrozenMetaData *meta,
								 const uint64 *ordinals, uint64 count, ItemPointerData *tids,
								 bool *visible, Buffer *vmbufs, int nvmbufs);
const char *AnnFrozenColumnName(Relation index, int column);
Oid			AnnFrozenColumnCollation(Relation index, int column);
void		AnnArrayReadRange(AnnArrayReader *reader, uint64 first, uint64 n, void *out);
int			AnnFrozenEvaluate(Relation index, const AnnFrozenMetaData *meta, const unsigned char *program,
							  size_t size, uint32 *words, uint64 *count);
void		AnnFrozenEvaluateOrError(Relation index, const AnnFrozenMetaData *meta, const char *program,
									 size_t size, uint32 *words, uint64 *count, bool cache);
int64		AnnFrozenKeyOrdinal(Relation index, const AnnFrozenMetaData *meta, int64 key);
int			AnnFrozenVectorMode(Relation index, const AnnFrozenMetaData *meta);

/* Ordinal i's vector, as an exact or (FP16 pages) rounded vector value. */
typedef struct AnnFrozenVectorReader
{
	int			mode;			/* ANN_FROZEN_VECTORS_*, or HEAP when forced */
	int			dims;
	AnnArrayReader pages;		/* vector or half pages */
	AnnArrayReader tids;		/* heap reads */
	Relation	heap;
	struct TupleTableSlot *slot;
	AttrNumber	attno;			/* the vector's table column */
	struct Vector *vector;		/* the result, valid until the next read */
	char	   *raw;
} AnnFrozenVectorReader;
void		AnnFrozenVectorReaderInit(AnnFrozenVectorReader *reader, Relation index, const AnnFrozenMetaData *meta,
									  BufferAccessStrategy strategy, bool exact);
struct Vector *AnnFrozenVectorRead(AnnFrozenVectorReader *reader, uint64 ordinal);
void		AnnFrozenVectorReaderEnd(AnnFrozenVectorReader *reader);

/*
 * A collection's dictionary for a text attribute: codes are assigned as
 * strings first appear (append-only), so unlike a frozen index's they don't
 * follow the collation; sorted gives the collation order for comparisons.
 */
typedef struct AnnTextDictionary
{
	int			count;
	text	  **sorted;			/* in the column's collation */
	int32	   *codes;			/* code of each sorted string */
	text	  **byCode;			/* string of each code, NULL for gaps */
	int32		ncodes;
} AnnTextDictionary;

/* Predicates over a frozen index's attribute columns (annpred.c). */
typedef struct AnnPredicateColumns
{
	int			ncolumns;
	AttrNumber	attnos[INDEX_MAX_KEYS]; /* table columns */
	Oid			types[INDEX_MAX_KEYS];
	Oid			families[INDEX_MAX_KEYS];	/* default btree operator families */
	Oid			collations[INDEX_MAX_KEYS];
	char	   *names[INDEX_MAX_KEYS];
	Relation	index;			/* for text dictionaries */
	AnnFrozenMetaData *meta;
	AnnTextDictionary *dictionaries[INDEX_MAX_KEYS];	/* collections: text
														 * columns' codes */
} AnnPredicateColumns;

/*
 * Attribute columns hold int32 values: smallint, integer, bigint, bool,
 * date, and codes of a dictionary for text and varchar; integer[] columns
 * of frozen indexes hold any number per row.
 */
#define AnnFrozenTextType(type) ((type) == TEXTOID || (type) == VARCHAROID)
/* Multi-valued attributes: integer[], each element an int32 value. */
#define AnnFrozenMultiType(type) ((type) == INT4ARRAYOID)
bool		AnnFrozenAttributeType(Oid type);
Datum		AnnFrozenTextDatum(Relation index, const AnnFrozenMetaData *meta, int column, int32 code);
Datum		AnnFrozenArrayDatum(AnnArrayReader *values, AnnArrayReader *starts, uint64 ordinal, bool *isnull);
int64		AnnFrozenTextSearch(Relation index, const AnnFrozenMetaData *meta, int column, Oid collation,
								text *value, bool upper);
int32		AnnFrozenAttributeValue(Datum value, bool isnull, Oid type);
Datum		AnnFrozenAttributeDatum(int32 value, Oid type);

void		AnnPredicateColumnsInit(Relation index, AnnPredicateColumns *columns);
void		AnnPredicateColumnsForTable(Relation rel, int count, char **names, AnnPredicateColumns *columns);
bool		AnnFrozenCachedColumns(Relation index, AnnPredicateColumns *columns);
void		AnnFrozenStoreColumns(Relation index, const AnnPredicateColumns *columns);
extern int	ann_collection_text_range_limit;
extern uint64 ann_collection_program_epoch;
void		AnnCollectionTextRangeLimitAssign(int value, void *extra);

/* Program bytes bound a range's codes: 4 each, up to ANN_FILTER_MAX_PROGRAM. */
#define ANN_TEXT_RANGE_MAX 250000
bool		AnnFrozenTableFrozen(Oid relid);
void		AnnFrozenTombstonesChanged(void);
bool		AnnPredicateSupported(List *clauses, Index varno, const AnnPredicateColumns *columns);
void		AnnPredicateBuild(List *clauses, Index varno, const AnnPredicateColumns *columns,
							  PlanState *parent, ExprContext *econtext, StringInfo program);
bool		AnnPredicateFromText(Oid relid, const char *filter, const AnnPredicateColumns *columns,
								 StringInfo program);
Relation	AnnFrozenIndexOf(Oid relid, uint64 generation, LOCKMODE lockmode);

/* Statuses shared with ann/backend.h. */
#define ANN_STATUS_OK 0
#define ANN_STATUS_INVALID 1
#define ANN_STATUS_UNSUPPORTED 2
#define ANN_STATUS_STALE 3
#define ANN_STATUS_OVERLOADED 7
#define ANN_STATUS_MISSING 11
#define ANN_STATUS_GENERATION 12

#define ANN_OP_INDEX_SEARCH_PREDICATE 22
#define ANN_OP_COLLECTION_SEARCH 25
#define ANN_OP_INDEX_PREDICATE 23
#define ANN_OP_DELETE_KEYS 29
#define ANN_TOMBSTONE_BATCH 100000	/* keys per delete request */
#define ANN_OP_INDEX_SEARCH_BATCH 30
#define ANN_PREDICATE_ORDINALS 1
#define ANN_PREDICATE_SNAPSHOT 8	/* coordinator applies snapshot-visible
									 * tombstones */
#define ANN_PREDICATE_ATTRIBUTES 2

bool		AnnIndexCall(uint16 operation, const char *payload, size_t length, int timeoutMs,
						 uint32 *status, char **reply, uint32 *replyLength);
char	   *AnnCatalogNameFor(Oid relid, uint64 generation);

/* Generation sent in request headers; 0 for untagged requests. */
extern uint64 ann_request_generation;
extern bool ann_enable_frozen_search;
extern int	ann_frozen_filter_cache_mb;
void		AnnFilterCacheInit(void);
bool		AnnFilterCacheGet(Oid index, uint64 generation, const char *program, size_t size, uint32 *words,
							  size_t nwords, uint64 *eligible);
void		AnnFilterCachePut(Oid index, uint64 generation, const char *program, size_t size,
							  const uint32 *words, size_t nwords, uint64 eligible);
extern bool ann_frozen_mmap;
extern bool ann_frozen_immutable;
extern bool ann_frozen_complete_short;
void		AnnFrozenScanInit(void);
IndexBulkDeleteResult *AnnFrozenBulkDelete(IndexVacuumInfo *info, IndexBulkDeleteResult *stats,
										   IndexBulkDeleteCallback callback, void *callback_state);

IndexBuildResult *annbuild(Relation heap, Relation index, IndexInfo *indexInfo);
void		annbuildempty(Relation index);
bool		anninsert(Relation index, Datum *values, bool *isnull, ItemPointer tid,
					  Relation heap, IndexUniqueCheck checkUnique, bool indexUnchanged,
					  IndexInfo *indexInfo);
IndexScanDesc annbeginscan(Relation index, int nkeys, int norderbys);
void		annrescan(IndexScanDesc scan, ScanKey keys, int nkeys,
					  ScanKey orderbys, int norderbys);
bool		anngettuple(IndexScanDesc scan, ScanDirection dir);
void		annendscan(IndexScanDesc scan);
IndexBulkDeleteResult *annbulkdelete(IndexVacuumInfo *info, IndexBulkDeleteResult *stats,
									 IndexBulkDeleteCallback callback, void *callback_state);
IndexBulkDeleteResult *annvacuumcleanup(IndexVacuumInfo *info, IndexBulkDeleteResult *stats);

#endif
