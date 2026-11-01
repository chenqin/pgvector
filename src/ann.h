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
#include "vector.h"

#define ANN_MAGIC_NUMBER 0xA991A991
#define ANN_VERSION 1			/* mutable layout: index tuples */
#define ANN_FROZEN_VERSION 2	/* frozen layout: arrays by ordinal */
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
#define AnnFrozenMetaSize(ncolumns) \
	(offsetof(AnnFrozenMetaData, columns) + (ncolumns) * sizeof(AnnFrozenColumn))

/* Sequential reads of an array, one pinned page at a time. */
typedef struct AnnArrayReader
{
	Relation	index;
	const AnnArray *array;
	BufferAccessStrategy strategy;
	Buffer		buf;
	BlockNumber page;			/* page of the array pinned in buf */
	bool		locked;
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
IndexBuildResult *AnnFrozenBuild(Relation heap, Relation index, IndexInfo *indexInfo);
void		AnnArrayReaderInit(AnnArrayReader *reader, Relation index, const AnnArray *array,
							   BufferAccessStrategy strategy);
char	   *AnnArrayLockPage(AnnArrayReader *reader, BlockNumber page, uint32 *count);
void		AnnArrayUnlockPage(AnnArrayReader *reader);
void		AnnArrayRead(AnnArrayReader *reader, uint64 element, void *out);
void		AnnArrayReaderEnd(AnnArrayReader *reader);
const char *AnnFrozenColumnName(Relation index, int column);
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
