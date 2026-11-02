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
#define ANN_FROZEN_VERSION 4	/* frozen layout: arrays by ordinal; 3 adds
								 * bitset pages and text dictionaries to 2, 4
								 * the attribute names */
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
} AnnOptions;

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

/* An INCLUDE column after the key: forward values and its inverted index. */
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
								 * plus the end */
} AnnFrozenColumnExtra;

typedef struct AnnFrozenMetaData
{
	uint64		rows;
	uint64		generation;		/* hash of the indexed content */
	Oid			guard;			/* guard trigger at build time */
	uint32		ncolumns;		/* INCLUDE columns after the key */
	AnnArray	tids;			/* ItemPointerData per ordinal */
	AnnArray	keys;			/* int64 per ordinal, strictly increasing */
	AnnArray	vectors;		/* Vector varlena per ordinal */
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
 * read-only mappings of the files.
 */
#define ANN_FROZEN_FLUSHED ((BlockNumber) 0xFFFFFFFE)

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
char	   *AnnArrayLockPage(AnnArrayReader *reader, BlockNumber page, uint32 *count);
void		AnnArrayUnlockPage(AnnArrayReader *reader);
void		AnnArrayRead(AnnArrayReader *reader, uint64 element, void *out);
void		AnnArrayReaderEnd(AnnArrayReader *reader);
const char *AnnFrozenColumnName(Relation index, int column);
Oid			AnnFrozenColumnCollation(Relation index, int column);
void		AnnArrayReadRange(AnnArrayReader *reader, uint64 first, uint64 n, void *out);
int			AnnFrozenEvaluate(Relation index, const AnnFrozenMetaData *meta, const unsigned char *program,
							  size_t size, uint32 *words, uint64 *count);
void		AnnFrozenEvaluateOrError(Relation index, const AnnFrozenMetaData *meta, const char *program,
									 size_t size, uint32 *words, uint64 *count, bool cache);
int64		AnnFrozenKeyOrdinal(Relation index, const AnnFrozenMetaData *meta, int64 key);

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
} AnnPredicateColumns;

/*
 * Attribute columns hold int32 values: smallint, integer, bigint, bool,
 * date, and codes of a dictionary for text and varchar.
 */
#define AnnFrozenTextType(type) ((type) == TEXTOID || (type) == VARCHAROID)
bool		AnnFrozenAttributeType(Oid type);
Datum		AnnFrozenTextDatum(Relation index, const AnnFrozenMetaData *meta, int column, int32 code);
int64		AnnFrozenTextSearch(Relation index, const AnnFrozenMetaData *meta, int column, Oid collation,
								text *value, bool upper);
int32		AnnFrozenAttributeValue(Datum value, bool isnull, Oid type);
Datum		AnnFrozenAttributeDatum(int32 value, Oid type);

void		AnnPredicateColumnsInit(Relation index, AnnPredicateColumns *columns);
void		AnnPredicateColumnsForTable(Relation rel, int count, char **names, AnnPredicateColumns *columns);
bool		AnnFrozenCachedColumns(Relation index, AnnPredicateColumns *columns);
void		AnnFrozenStoreColumns(Relation index, const AnnPredicateColumns *columns);
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
#define ANN_OP_INDEX_PREDICATE 23
#define ANN_PREDICATE_ORDINALS 1

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
