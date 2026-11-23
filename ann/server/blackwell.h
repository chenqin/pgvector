#ifndef ANN_BLACKWELL_H
#define ANN_BLACKWELL_H

#include "indexfile.h"

#include <stdint.h>

/*
 * Blackwell (compute capability 10.x or 12.x) scan module for the cuVS
 * worker. It keeps an index's quantized copy (NVFP4 or FP8, see indexfile.h)
 * in each shard's GPU memory and scores filtered rows from it with kernels
 * compiled for sm_100a or sm_120a, which decode E2M1 and E4M3 with the hardware conversion
 * instructions (cvt.rn.f16x2.e2m1x2, cvt.rn.f16x2.e4m3x2). Callers select
 * k + slack candidates from those scores and re-score them in FP32, so
 * returned distances stay exact; graph search stays with cuVS CAGRA.
 */
/* Row formats the module scores, besides the quantized copies. */
#define ANN_BLACKWELL_HALF 3	/* a CAGRA FP16 dataset, read in place */
#define ANN_BLACKWELL_FLOAT 4	/* a CAGRA FP32 dataset, read in place */

typedef struct AnnBlackwellShard
{
	int			quant;			/* ANN_QUANT_* or ANN_BLACKWELL_*, NONE when
								 * not available */
	int			owned;			/* data is the module's own copy */
	uint8_t    *data;			/* the shard's rows on the device */
	uint8_t    *scales;
	float	   *rowScales;
	uint32_t	rowBytes;
	uint32_t	blocks;
	size_t		bytes;			/* device memory held */
	uint8_t    *packed;			/* NVFP4: the lane-contiguous copy the fused
								 * scan reads */
	float	   *norms;			/* NVFP4: (row scale, |x|^2) per row */
} AnnBlackwellShard;

/* Whether the module runs on this device: compute capability 10.x or 12.x and NVRTC. */
int			AnnBlackwellReady(int device);

/* Whether the fused scan (AnnBlackwellScoreBelow) has every input it reads for this shard. */
int			AnnBlackwellFuseReady(const AnnBlackwellShard *shard);

/*
 * An FP8 (E4M3, row-scaled) copy of FP32 rows on a device, made at load for
 * re-scoring candidates there (--rescore fp8-gpu); 0 on failure.
 */
int			AnnBlackwellUploadFp8(const float *vectors, uint64_t rows, uint32_t dims, int device,
								  AnnBlackwellShard *out);

/* Copy a shard's quantized rows to the current device; 0 on failure. */
int			AnnBlackwellUpload(const AnnIndexFile *file, uint32_t shard, int device, AnnBlackwellShard *out);

/*
 * Score a shard's CAGRA dataset in place (FP16 or FP32 rows of dims
 * elements, stride elements apart) for grouped searches; 0 when the module
 * is not available on the device.
 */
int			AnnBlackwellView(const void *dataset, int64_t stride, int half, uint32_t dims, int device,
							 AnnBlackwellShard *out);
void		AnnBlackwellFree(AnnBlackwellShard *shard);

/*
 * Squared L2 distance of each listed shard row (UINT32_MAX scores +inf) to
 * an FP32 device query, written to out, on a CUDA stream of that device.
 */
int			AnnBlackwellScore(const AnnBlackwellShard *shard, int device, const uint32_t *rows, unsigned n,
							  const float *query, int dims, float *out, void *stream);

/*
 * The same for nq queries at once, each over its own rows: query q's are
 * rows[offsets[q]] .. rows[offsets[q + 1] - 1] (offsets on the device, nq + 1
 * of them), scored into out at the same positions. NVFP4 copies only.
 */
int			AnnBlackwellScoreSegments(const AnnBlackwellShard *shard, int device, const uint32_t *rows,
									  unsigned n, const uint32_t *offsets, int nq, const float *queries,
									  int dims, float *out, void *stream);

/*
 * Squared L2 distances of the listed rows to nq FP32 device queries on the
 * tensor cores, written to out as a [nq][n] matrix for the caller's own
 * selection (large k); nq is at most AnnBlackwellGroupLimit(dims).
 */
int			AnnBlackwellScoreMatrix(const AnnBlackwellShard *shard, int device, const uint32_t *rows,
									unsigned n, const float *queries, unsigned nq, int dims,
									float *out, void *stream);

/*
 * Score the listed rows for nq queries as AnnBlackwellScoreMatrix does, but
 * keep only distances at or under each query's bound (bounds[q * 32 + 31]):
 * they are appended, with their positions in rows, to candD and candR
 * (cap per query, counts[q] counting every kept row, also past the cap;
 * the caller zeroes counts). No score matrix is written.
 */
int			AnnBlackwellScoreBelow(const AnnBlackwellShard *shard, int device, const uint32_t *rows,
								   unsigned n, const float *queries, unsigned nq, int dims, const float *bounds,
								   float *candD, uint32_t *candR, uint32_t *counts, unsigned cap, void *stream);

/* rows[c * n / s] for c < s into out (a spread sample of a row list). */
int			AnnBlackwellStrideRows(int device, const uint32_t *rows, unsigned n, unsigned s, uint32_t *out,
								   void *stream);

/*
 * Large-k candidates for nq queries over the listed rows: a sample of the
 * rows bounds about four times `wanted` candidates per query, which the
 * tensor-core scan keeps (up to the returned capacity per query, counting
 * them all). The counts land in pinned host memory on the stream; after a
 * sync, AnnBlackwellCopyCandidates copies `take[q]` of each query's (distance,
 * position in rows) pairs and AnnBlackwellCandidateRows reads them. Every row
 * scoring at most the wanted-th best is a candidate when a query's count is
 * between `wanted` and the capacity. AnnBlackwellCandidateCap is that capacity
 * for n rows, or 0 when the rows are too few for a sample to pay; the same 0
 * comes back when a launch fails.
 */
unsigned	AnnBlackwellCandidateCap(unsigned n, unsigned wanted);
unsigned	AnnBlackwellCandidates(const AnnBlackwellShard *shard, int device, unsigned slot, const uint32_t *rows,
								   unsigned n, const float *queries, unsigned nq, int dims, unsigned wanted,
								   void *stream);
const unsigned *AnnBlackwellCandidateCounts(int device, unsigned slot);
int			AnnBlackwellCopyCandidates(int device, unsigned slot, unsigned nq, unsigned cap, const unsigned *take,
									   void *stream);
void		AnnBlackwellCandidateRows(int device, unsigned slot, unsigned q, unsigned cap, float **distances,
									  unsigned **rows);

/*
 * Fused search: score the listed rows for nq queries (contiguous FP32 on
 * the device) and select each query's best ANN_BLACKWELL_TOP in one launch,
 * then copy them to pinned host memory, all on the stream. A single query
 * uses eight threads a row; several use one thread a row, so every row is
 * read once for all of them. nq is at most AnnBlackwellGroupLimit(dims),
 * never above ANN_BLACKWELL_GROUP_MAX. Slot separates concurrent searches
 * of one caller thread on a device (one per shard). After the stream
 * completes, AnnBlackwellResults gives query q's positions in the row list
 * (UINT32_MAX past the end) and their quantized distances, ascending.
 */
#define ANN_BLACKWELL_TOP 32
#define ANN_BLACKWELL_GROUP_MAX 64
/*
 * Grouped scans of at least ANN_BLACKWELL_FUSE_ROWS rows (and at most
 * ANN_BLACKWELL_FUSE_DIMS dimensions) take the fused route: bounds from an
 * ANN_BLACKWELL_SAMPLE-row sample, then only rows under them kept.
 */
#define ANN_BLACKWELL_SAMPLE 8192
#define ANN_BLACKWELL_FUSE_ROWS 262144
#define ANN_BLACKWELL_FUSE_DIMS 512
/* Opt-in dynamic shared memory for one scan block's queries. */
#define ANN_BLACKWELL_SHARED_MAX (160 * 1024)
int			AnnBlackwellGroupLimit(int dims);
int			AnnBlackwellSearch(const AnnBlackwellShard *shard, int device, unsigned slot, const uint32_t *rows,
							   unsigned n, const float *queries, unsigned nq, int dims, void *stream);
void		AnnBlackwellResults(int device, unsigned slot, unsigned q, const uint32_t **positions,
								const float **distances);

#endif
