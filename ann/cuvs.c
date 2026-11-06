/*
 * Resident CAGRA and gathered searches share loaded shard data and
 * per-lane streams; grouped gathers can score, select and rerank matrices.
 */
#include "backend.h"
#include "indexfile.h"
#include "nvrtc.h"
#include "blackwell.h"

/* The driver API header has enumerators that are not strict ISO C. */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#include <cuda.h>
#pragma GCC diagnostic pop
#include <cuda_runtime_api.h>
#include <cuvs/neighbors/brute_force.h>
#include <cuvs/neighbors/cagra.h>
#include <cuvs/selection/select_k.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <time.h>
#include "replica.h"
#include "select_batch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct AnnCuvsState
{
	cuvsResources_t resources;
	cuvsCagraIndex_t graph;
	cuvsDataset_t dataset;
	cuvsCagraIndexParams_t build;
	cuvsCagraSearchParams_t search;
	int			device;
} AnnCuvsState;

/* Error paths synchronize before releasing objects referenced by queued work. */
#define CuvsCall(call) \
	do { \
		if ((call) != CUVS_SUCCESS) { \
			const char *message = cuvsGetLastErrorText(); \
			fprintf(stderr, "cuVS: %s\n", message ? message : "operation failed"); \
			goto done; \
		} \
	} while (0)

#define CudaCall(call) \
	do { \
		cudaError_t error = (call); \
		if (error != cudaSuccess) { \
			fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(error)); \
			goto done; \
		} \
	} while (0)

static DLManagedTensor
			Tensor(void *data, DLDeviceType deviceType, int device,
				   uint8_t code, int dimensions, int64_t *shape);

/*
 * Gathered search: every request carries its own eligible vectors, so each
 * job is an exact brute-force search on a GPU. A lane is one worker thread;
 * lanes are spread round-robin across visible devices and keep their own
 * resources, stream, and device buffers.
 */
#define ANN_CUVS_MIN_ROWS 2048	/* smaller jobs are cheaper on the CPU */
#define ANN_CUVS_MAX_K 1024
#define ANN_CUVS_ERROR 2e-3		/* bound on fp32 relative distance error */

typedef struct AnnCuvsLane
{
	int			ready;			/* 0 unset, 1 usable, -1 no device */
	int			device;
	cuvsResources_t resources;
	float	   *vectors;		/* device: every job's rows, then queries */
	int64_t    *neighbors;		/* device */
	float	   *distances;		/* device */
	int64_t    *hostNeighbors;	/* pinned */
	float	   *hostDistances;	/* pinned */
	size_t		vectorCapacity; /* floats */
	size_t		resultCapacity; /* candidates */
} AnnCuvsLane;

static _Thread_local AnnCuvsLane lane;
static atomic_uint nextLane;
static atomic_ullong deviceJobs;

static int
LaneReady(void)
{
	int			devices = 0;

	if (lane.ready)
		return lane.ready > 0;
	lane.ready = -1;
	if (cudaGetDeviceCount(&devices) != cudaSuccess || devices <= 0)
		return 0;
	lane.device = (int) (atomic_fetch_add(&nextLane, 1) % (unsigned) devices);
	if (cudaSetDevice(lane.device) != cudaSuccess ||
		(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync) != cudaSuccess &&
		 cudaGetLastError() != cudaErrorSetOnActiveProcess) ||
		cuvsResourcesCreate(&lane.resources) != CUVS_SUCCESS)
		return 0;
	lane.ready = 1;
	return 1;
}

/* Grow lane buffers geometrically; buffers are reused across submissions. */
static int
Reserve(size_t floats, size_t candidates)
{
	if (floats > lane.vectorCapacity)
	{
		if (lane.vectors)
			(void) cudaFree(lane.vectors);
		lane.vectorCapacity = 0;
		floats += floats / 2;
		if (cudaMalloc((void **) &lane.vectors, floats * sizeof(float)) != cudaSuccess)
		{
			lane.vectors = NULL;
			return 0;
		}
		lane.vectorCapacity = floats;
	}
	if (candidates > lane.resultCapacity)
	{
		(void) cudaFree(lane.neighbors);
		(void) cudaFree(lane.distances);
		(void) cudaFreeHost(lane.hostNeighbors);
		(void) cudaFreeHost(lane.hostDistances);
		lane.neighbors = NULL;
		lane.distances = NULL;
		lane.hostNeighbors = NULL;
		lane.hostDistances = NULL;
		lane.resultCapacity = 0;
		candidates += candidates / 2;
		if (cudaMalloc((void **) &lane.neighbors, candidates * sizeof(int64_t)) != cudaSuccess ||
			cudaMalloc((void **) &lane.distances, candidates * sizeof(float)) != cudaSuccess ||
			cudaMallocHost((void **) &lane.hostNeighbors, candidates * sizeof(int64_t)) != cudaSuccess ||
			cudaMallocHost((void **) &lane.hostDistances, candidates * sizeof(float)) != cudaSuccess)
			return 0;
		lane.resultCapacity = candidates;
	}
	return 1;
}

static int	CandidateCompare(const void *left, const void *right);

/*
 * Re-score GPU candidates in double precision and keep rows within the
 * gathered tolerance of the k-th exact distance. A row the GPU did not return
 * has an fp32 distance of at least the last returned one; if that bound
 * cannot exclude it from the threshold, the caller scores the job on the CPU.
 */
static int
FinishJob(AnnGatheredJob *job, const int64_t *neighbors, const float *distances, uint32_t count)
{
	double		threshold;
	uint32_t	kept = 0;

	for (uint32_t i = 0; i < count; i++)
	{
		const float *vector;
		double		squared = 0;

		if (neighbors[i] < 0 || neighbors[i] >= job->rows)
			return 0;
		vector = job->vectors + (size_t) neighbors[i] * job->dimensions;
		for (uint32_t d = 0; d < job->dimensions; d++)
		{
			double		delta = (double) vector[d] - job->query[d];

			squared += delta * delta;
		}
		job->results[i] = (AnnCandidate)
		{
			(uint32_t) neighbors[i], sqrt(squared)
		};
	}
	qsort(job->results, count, sizeof(*job->results), CandidateCompare);
	threshold = job->results[job->k - 1].distance;
	threshold += ANN_GATHERED_TOLERANCE * fabs(threshold) + 1e-6;
	if (count < job->rows &&
		(double) distances[count - 1] * (1 - ANN_CUVS_ERROR) - 1e-6 <= threshold)
		return 0;
	while (kept < count && job->results[kept].distance <= threshold)
		kept++;
	job->nresults = kept;
	job->status = ANN_OK;
	return 1;
}

static uint32_t
Candidates(const AnnGatheredJob *job)
{
	uint32_t	count = job->k * 2 + 64;

	if (lane.ready <= 0 || job->metric != 0 || job->k == 0 ||
		job->rows < ANN_CUVS_MIN_ROWS || job->k > ANN_CUVS_MAX_K / 2)
		return 0;
	if (count > ANN_CUVS_MAX_K)
		count = ANN_CUVS_MAX_K;
	return count < job->rows ? count : job->rows;
}

/*
 * Queue copies, build, search, and result copies for every job on the lane's
 * stream, then synchronize once. Jobs the GPU skips or cannot settle are
 * scored by the exact CPU kernel.
 */
static void
SearchGathered(AnnGatheredJob *jobs, uint32_t count)
{
	size_t		floats = 0;
	size_t		candidates = 0;
	size_t		vectorOffset = 0;
	size_t		queryOffset = 0;
	size_t		resultOffset = 0;
	cudaStream_t stream;
	cuvsBruteForceIndex_t *indexes = calloc(count, sizeof(*indexes));
	int			synced;

	(void) LaneReady();
	for (uint32_t i = 0; i < count; i++)
	{
		uint32_t	n = Candidates(&jobs[i]);

		jobs[i].status = ANN_PENDING;
		if (n == 0)
			continue;
		floats += ((size_t) jobs[i].rows + 1) * jobs[i].dimensions;
		queryOffset += (size_t) jobs[i].rows * jobs[i].dimensions;
		candidates += n;
	}
	if (candidates == 0 || indexes == NULL || !Reserve(floats, candidates) ||
		cuvsStreamGet(lane.resources, &stream) != CUVS_SUCCESS)
		goto cpu;
	for (uint32_t i = 0; i < count; i++)
	{
		AnnGatheredJob *job = &jobs[i];
		uint32_t	n = Candidates(job);
		int64_t		dataShape[2] = {job->rows, job->dimensions};
		int64_t		queryShape[2] = {1, job->dimensions};
		int64_t		resultShape[2] = {1, n};
		float	   *data = lane.vectors + vectorOffset;
		float	   *query = lane.vectors + queryOffset;
		DLManagedTensor dataTensor;
		DLManagedTensor queryTensor;
		DLManagedTensor neighborTensor;
		DLManagedTensor distanceTensor;
		cuvsBruteForceIndex_t index = NULL;
		cuvsFilter	filter = {0, NO_FILTER};

		if (n == 0)
			continue;
		vectorOffset += (size_t) job->rows * job->dimensions;
		queryOffset += job->dimensions;
		dataTensor = Tensor(data, kDLCUDA, lane.device, kDLFloat, 2, dataShape);
		queryTensor = Tensor(query, kDLCUDA, lane.device, kDLFloat, 2, queryShape);
		neighborTensor = Tensor(lane.neighbors + resultOffset, kDLCUDA, lane.device, kDLInt, 2, resultShape);
		neighborTensor.dl_tensor.dtype.bits = 64;
		distanceTensor = Tensor(lane.distances + resultOffset, kDLCUDA, lane.device, kDLFloat, 2, resultShape);
		if (cudaMemcpyAsync(data, job->vectors, (size_t) job->rows * job->dimensions * sizeof(float),
							cudaMemcpyHostToDevice, stream) == cudaSuccess &&
			cudaMemcpyAsync(query, job->query, job->dimensions * sizeof(float),
							cudaMemcpyHostToDevice, stream) == cudaSuccess &&
			cuvsBruteForceIndexCreate(&index) == CUVS_SUCCESS &&
			cuvsBruteForceBuild(lane.resources, &dataTensor, L2SqrtUnexpanded, 0, index) == CUVS_SUCCESS &&
			cuvsBruteForceSearch(lane.resources, index, &queryTensor, &neighborTensor,
								 &distanceTensor, filter) == CUVS_SUCCESS &&
			cudaMemcpyAsync(lane.hostNeighbors + resultOffset, lane.neighbors + resultOffset,
							n * sizeof(int64_t), cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
			cudaMemcpyAsync(lane.hostDistances + resultOffset, lane.distances + resultOffset,
							n * sizeof(float), cudaMemcpyDeviceToHost, stream) == cudaSuccess)
			job->status = ANN_BACKEND_ERROR;	/* queued; settled after sync */
		else
		{
			const char *message = cuvsGetLastErrorText();

			fprintf(stderr, "cuVS gathered search: %s\n", message ? message : "failed");
		}

		/* An index references its dataset; destroy it after the sync. */
		indexes[i] = index;
		resultOffset += n;
	}
	synced = cuvsStreamSync(lane.resources) == CUVS_SUCCESS;
	resultOffset = 0;
	for (uint32_t i = 0; i < count; i++)
	{
		uint32_t	n = Candidates(&jobs[i]);

		if (indexes[i])
			(void) cuvsBruteForceIndexDestroy(indexes[i]);
		if (n == 0)
			continue;
		if (!(synced && jobs[i].status == ANN_BACKEND_ERROR &&
			  FinishJob(&jobs[i], lane.hostNeighbors + resultOffset,
						lane.hostDistances + resultOffset, n)))
			jobs[i].status = ANN_PENDING;
		else
			atomic_fetch_add(&deviceJobs, 1);
		resultOffset += n;
	}
cpu:
	free(indexes);
	for (uint32_t i = 0; i < count; i++)
		if (jobs[i].status != ANN_OK)
			AnnGatheredExact(&jobs[i]);
}

static DLManagedTensor
Tensor(void *data, DLDeviceType deviceType, int device,
	   uint8_t code, int dimensions, int64_t *shape)
{
	DLManagedTensor tensor = {0};

	tensor.dl_tensor.data = data;
	tensor.dl_tensor.device.device_type = deviceType;
	tensor.dl_tensor.device.device_id = device;
	tensor.dl_tensor.ndim = dimensions;
	tensor.dl_tensor.dtype = (DLDataType)
	{
		code, 32, 1
	};
	tensor.dl_tensor.shape = shape;
	return tensor;
}

static void
Destroy(AnnIndex *index)
{
	AnnCuvsState *state = index->backend_state;

	if (state)
	{
		if (state->resources)
			(void) cuvsStreamSync(state->resources);
		if (state->graph)
			(void) cuvsCagraIndexDestroy(state->graph);
		if (state->dataset)
			(void) cuvsDatasetDestroy(state->dataset);
		if (state->search)
			(void) cuvsCagraSearchParamsDestroy(state->search);
		if (state->build)
			(void) cuvsCagraIndexParamsDestroy(state->build);
		if (state->resources)
			(void) cuvsResourcesDestroy(state->resources);
		free(state);
	}
	AnnCpuBackend()->destroy(index);
}

static AnnStatus
Build(AnnIndex *index, uint64_t generation, uint32_t rows,
	  uint32_t dimensions, const float *vectors)
{
	AnnIndex	next = {0};
	AnnCuvsState *state;
	AnnStatus	status;
	int64_t		shape[2] = {rows, dimensions};
	DLManagedTensor source;
	uint16_t	major;
	uint16_t	minor;
	uint16_t	patch;

	/* Conservative initial envelope for the chosen graph degrees. */
	if (rows < 128)
		return ANN_UNSUPPORTED;
	status = AnnCpuBackend()->build(&next, generation, rows, dimensions, vectors);
	if (status != ANN_OK)
		return status;
	state = calloc(1, sizeof(*state));
	if (state == NULL)
	{
		AnnCpuBackend()->destroy(&next);
		return ANN_NOMEM;
	}
	next.backend_state = state;
	status = ANN_BACKEND_ERROR;
	cuvsSetLogLevel(CUVS_LOG_LEVEL_OFF);
	CuvsCall(cuvsVersionGet(&major, &minor, &patch));
	if (major != 26 || minor != 10)
	{
		fprintf(stderr, "cuVS: expected 26.10, found %u.%u.%u\n", major, minor, patch);
		status = ANN_UNSUPPORTED;
		goto done;
	}
	CuvsCall(cuvsResourcesCreate(&state->resources));
	CuvsCall(cuvsDeviceIdGet(state->resources, &state->device));
	CuvsCall(cuvsCagraIndexParamsCreate(&state->build));
	CuvsCall(cuvsCagraSearchParamsCreate(&state->search));
	CuvsCall(cuvsCagraIndexCreate(&state->graph));
	state->build->metric = L2Expanded;
	state->build->graph_degree = 32;
	state->build->intermediate_graph_degree = 64;
	state->search->max_queries = 1;
	state->search->persistent = false;
	source = Tensor(next.vectors, kDLCPU, 0, kDLFloat, 2, shape);
	CuvsCall(cuvsDatasetMakePadded(state->resources, &source,
								   CUVS_DATASET_MEM_TYPE_DEVICE, &state->dataset));
	CuvsCall(cuvsCagraBuild(state->resources, state->build, state->dataset, state->graph));
	CuvsCall(cuvsStreamSync(state->resources));
	/* Publish only after all construction work has completed successfully. */
	Destroy(index);
	*index = next;
	memset(&next, 0, sizeof(next));
	status = ANN_OK;
done:
	Destroy(&next);
	return status;
}

static int
IdCompare(const void *left, const void *right)
{
	uint32_t	a = *(const uint32_t *) left;
	uint32_t	b = *(const uint32_t *) right;

	return a < b ? -1 : a > b;
}

static int
CandidateCompare(const void *left, const void *right)
{
	const AnnCandidate *a = left;
	const AnnCandidate *b = right;

	if (a->distance != b->distance)
		return a->distance < b->distance ? -1 : 1;
	return a->id < b->id ? -1 : a->id > b->id;
}

static AnnStatus
Search(const AnnIndex *index, const float *query, uint32_t metric,
	   const uint32_t *ids, uint32_t nids, uint32_t k,
	   AnnCandidate *results, uint32_t *nresults)
{
	AnnCuvsState *state = index->backend_state;
	float	   *deviceQuery = NULL;
	float	   *deviceDistances = NULL;
	uint32_t   *deviceNeighbors = NULL;
	uint32_t   *deviceMask = NULL;
	uint32_t   *mask = NULL;
	uint32_t   *neighbors = NULL;
	size_t		words = ((size_t) index->rows + 31) / 32;
	int64_t		queryShape[2] = {1, index->dimensions};
	int64_t		resultShape[2];
	int64_t		maskShape[1] = {(int64_t) words};
	DLManagedTensor queryTensor;
	DLManagedTensor neighborTensor;
	DLManagedTensor distanceTensor;
	DLManagedTensor maskTensor;
	cuvsFilter	filter = {0};
	AnnStatus	status = ANN_BACKEND_ERROR;

	*nresults = 0;
	if (metric != 0)
		return ANN_UNSUPPORTED;
	if (k > 1024)
		return ANN_UNSUPPORTED;
	for (uint32_t d = 0; d < index->dimensions; d++)
		if (!isfinite(query[d]))
			return ANN_INVALID;
	for (uint32_t i = 0; ids && i < nids; i++)
		if (ids[i] >= index->rows || (i && ids[i] <= ids[i - 1]))
			return ANN_INVALID;
	if (ids && k > nids)
		k = nids;
	if (k == 0)
		return ANN_OK;
	if (state == NULL)
		return ANN_STALE;
	resultShape[0] = 1;
	resultShape[1] = k;
	neighbors = malloc((size_t) k * sizeof(*neighbors));
	if (neighbors == NULL)
		return ANN_NOMEM;
	CudaCall(cudaMalloc((void **) &deviceQuery, index->dimensions * sizeof(float)));
	CudaCall(cudaMalloc((void **) &deviceNeighbors, (size_t) k * sizeof(uint32_t)));
	CudaCall(cudaMalloc((void **) &deviceDistances, (size_t) k * sizeof(float)));
	CudaCall(cudaMemset(deviceNeighbors, 0xff, (size_t) k * sizeof(uint32_t)));
	CudaCall(cudaMemcpy(deviceQuery, query, index->dimensions * sizeof(float), cudaMemcpyHostToDevice));
	queryTensor = Tensor(deviceQuery, kDLCUDA, state->device, kDLFloat, 2, queryShape);
	neighborTensor = Tensor(deviceNeighbors, kDLCUDA, state->device, kDLUInt, 2, resultShape);
	distanceTensor = Tensor(deviceDistances, kDLCUDA, state->device, kDLFloat, 2, resultShape);
	filter.type = NO_FILTER;
	if (ids)
	{
		mask = calloc(words, sizeof(*mask));
		if (mask == NULL)
		{
			status = ANN_NOMEM;
			goto done;
		}
		for (uint32_t i = 0; i < nids; i++)
			mask[ids[i] / 32] |= UINT32_C(1) << (ids[i] % 32);
		CudaCall(cudaMalloc((void **) &deviceMask, words * sizeof(uint32_t)));
		CudaCall(cudaMemcpy(deviceMask, mask, words * sizeof(uint32_t), cudaMemcpyHostToDevice));
		maskTensor = Tensor(deviceMask, kDLCUDA, state->device, kDLUInt, 1, maskShape);
		filter.type = BITSET;
		filter.addr = (uintptr_t) &maskTensor;
	}
	state->search->itopk_size = k < 128 ? 128 : ((size_t) k + 31) / 32 * 32;
	CuvsCall(cuvsCagraSearch(state->resources, state->search, state->graph,
							 &queryTensor, &neighborTensor, &distanceTensor, filter));
	CuvsCall(cuvsStreamSync(state->resources));
	CudaCall(cudaMemcpy(neighbors, deviceNeighbors, (size_t) k * sizeof(uint32_t), cudaMemcpyDeviceToHost));
	for (uint32_t i = 0; i < k; i++)
	{
		uint32_t	id = neighbors[i];
		double		squared = 0;

		if (id == UINT32_MAX)
			continue;
		if (id >= index->rows || (ids && !bsearch(&id, ids, nids, sizeof(*ids), IdCompare)))
			goto done;
		for (uint32_t j = 0; j < *nresults; j++)
			if (results[j].id == id)
				goto done;

		/*
		 * Canonical reranking fixes score precision, not approximate
		 * membership.
		 */
		for (uint32_t d = 0; d < index->dimensions; d++)
		{
			double		delta = (double) index->vectors[(size_t) id * index->dimensions + d] - query[d];

			squared += delta * delta;
		}
		results[*nresults] = (AnnCandidate)
		{
			id, sqrt(squared)
		};
		(*nresults)++;
	}
	qsort(results, *nresults, sizeof(*results), CandidateCompare);
	status = ANN_OK;
done:
	(void) cuvsStreamSync(state->resources);
	if (deviceMask)
		(void) cudaFree(deviceMask);
	if (deviceDistances)
		(void) cudaFree(deviceDistances);
	if (deviceNeighbors)
		(void) cudaFree(deviceNeighbors);
	if (deviceQuery)
		(void) cudaFree(deviceQuery);
	free(neighbors);
	free(mask);
	return status;
}

/*
 * Loaded indexes: one CAGRA graph per shard file, each on GPU (shard modulo
 * devices). Lanes search with their own per-device resources and buffers.
 */
#define ANN_CUVS_MAX_DEVICES ANN_REPLICA_MAX_DEVICES
#define ANN_CUVS_WORKSPACE_BYTES (64 * 1024 * 1024)

typedef struct AnnCuvsShard
{
	int			device;
	cuvsResources_t resources;	/* load-time owner of the index memory */
	cuvsCagraIndex_t index;
	cuvsDataset_t dataset;
	const void *rows;			/* device view of the CAGRA dataset */
	int64_t    *keys;			/* device copy of the shard's sorted keys */
	int64_t		stride;			/* elements between rows */
	float	   *rerank;			/* FP32 dataset for exact GPU reranking */
	int64_t		rerankStride;
	size_t		rerankBytes;	/* owned bytes, zero when reusing FP32 CAGRA
								 * rows */
	AnnBlackwellShard quant;	/* quantized rows for Blackwell scans */
} AnnCuvsShard;

/*
 * Shard s of copy c is shards[c * nshards + s]. Without replication there is
 * one copy and shard s lives on GPU s mod devices; with it, every GPU holds a
 * full copy. A batch reserves its home or a less loaded GPU copy.
 */
typedef struct AnnCuvsRoute
{
	atomic_uint active;
	atomic_uint_fast64_t started;
	atomic_uint_fast64_t finished;
	atomic_uint_fast64_t queries;
	atomic_uint_fast64_t migrations;
	atomic_uint_fast64_t failures;
	atomic_uint_fast64_t reserved_ns;
} AnnCuvsRoute;

typedef struct AnnCuvsIndex
{
	uint32_t	nshards;
	uint32_t	copies;
	int			fp16;
	int			quantized;		/* every shard copy has its quantized rows */
	int			grouped;		/* every shard copy can be scored in groups */
	AnnCuvsShard *shards;
	AnnCuvsRoute routing[ANN_CUVS_MAX_DEVICES];
	atomic_uint_fast64_t matrix_queries;
	atomic_uint_fast64_t matrix_peak_score_bytes;
	atomic_uint_fast64_t gpu_rerank_candidates;
	atomic_uint_fast64_t cost_queries[4];
} AnnCuvsIndex;

/* Keep the full lane ordinal: wrapping before modulo copies biases 3/6/7 GPUs. */
static _Thread_local int64_t laneHome = -1;
static atomic_uint nextHome;
static _Thread_local uint32_t laneCopy;

/* -1 uses deployment defaults; 0/1 is a batch-local calibrated decision. */
static _Thread_local int laneRerank = -1;
static _Thread_local uint32_t laneProfileItopk;
static _Thread_local uint32_t laneProfileSlack;
static _Alignas(64)
AnnReplicaRouter replicaRouter;

static uint64_t
ReplicaNow(void)
{
	struct timespec now;

	(void) clock_gettime(CLOCK_MONOTONIC, &now);
	return (uint64_t) now.tv_sec * UINT64_C(1000000000) + (uint64_t) now.tv_nsec;
}

/* One reservation spans a batch's device work and host reranking. */
static uint32_t
ReplicaAcquire(AnnCuvsIndex *state, uint32_t queries, uint64_t now)
{
	int			devices[ANN_CUVS_MAX_DEVICES];
	uint32_t	copy;

	if (laneHome < 0)
		laneHome = (int64_t) atomic_fetch_add(&nextHome, 1);
	for (uint32_t c = 0; c < state->copies; c++)
		devices[c] = state->shards[(size_t) c * state->nshards].device;
	copy = AnnReplicaAcquire(&replicaRouter, devices, state->copies, (uint64_t) laneHome,
							 AnnIndexConfig.replica_routing, AnnIndexConfig.replica_load_slack, now);
	if (state->copies == 1)
	{
		uint32_t	seen = UINT32_C(1) << devices[0];

		/* A sharded, unreplicated batch occupies all its devices. */
		for (uint32_t s = 0; s < state->nshards; s++)
		{
			int			d = state->shards[s].device;

			if (!(seen & (UINT32_C(1) << d)))
			{
				(void) atomic_fetch_add_explicit(&replicaRouter.devices[d].active, 1, memory_order_relaxed);
				seen |= UINT32_C(1) << d;
			}
		}
	}
	(void) atomic_fetch_add_explicit(&state->routing[copy].active, 1, memory_order_relaxed);
	(void) atomic_fetch_add_explicit(&state->routing[copy].started, 1, memory_order_relaxed);
	(void) atomic_fetch_add_explicit(&state->routing[copy].queries, queries, memory_order_relaxed);
	if (copy != (uint64_t) laneHome % state->copies)
		(void) atomic_fetch_add_explicit(&state->routing[copy].migrations, 1, memory_order_relaxed);
	laneCopy = copy;
	return copy;
}

static void
ReplicaRelease(AnnCuvsIndex *state, uint32_t copy, AnnIndexJob **jobs, uint32_t queries, uint64_t started)
{
	uint32_t	seen = 0;
	int			failed = 0;
	uint64_t	finished = ReplicaNow();
	uint64_t	elapsed = finished - started;

	for (uint32_t j = 0; j < queries; j++)
		if (jobs[j]->status != ANN_OK)
			failed = 1;
	for (uint32_t s = 0; s < state->nshards; s++)
	{
		int			d = state->shards[(size_t) copy * state->nshards + s].device;

		if (!(seen & (UINT32_C(1) << d)))
		{
			AnnReplicaSample(&replicaRouter, d, elapsed, finished);
			AnnReplicaRelease(&replicaRouter, d);
			seen |= UINT32_C(1) << d;
		}
	}
	(void) atomic_fetch_add_explicit(&state->routing[copy].reserved_ns, elapsed, memory_order_relaxed);
	(void) atomic_fetch_add_explicit(&state->routing[copy].failures, (uint64_t) failed, memory_order_relaxed);
	(void) atomic_fetch_add_explicit(&state->routing[copy].finished, 1, memory_order_relaxed);
	(void) atomic_fetch_sub_explicit(&state->routing[copy].active, 1, memory_order_relaxed);
}

/* Independent counters are approximate while searches run; no scheduler lock. */
static size_t
ReplicaStats(const AnnIndexFile *file, char *data, size_t capacity)
{
	const AnnCuvsIndex *state = file->device;
	size_t		used;
	int			written;

	if (state == NULL)
		return 0;
	written = snprintf(data, capacity, "{\"policy\":\"%s\",\"slack\":%u,\"copies\":%u,\"matrix_queries\":%llu,\"matrix_peak_score_bytes\":%llu,\"gpu_rerank_candidates\":%llu,\"cost_queries\":[%llu,%llu,%llu,%llu],\"replicas\":[",
					   AnnIndexConfig.replica_routing ? "load" : "home", AnnIndexConfig.replica_load_slack, state->copies,
					   (unsigned long long) atomic_load_explicit(&state->matrix_queries, memory_order_relaxed),
					   (unsigned long long) atomic_load_explicit(&state->matrix_peak_score_bytes, memory_order_relaxed),
					   (unsigned long long) atomic_load_explicit(&state->gpu_rerank_candidates, memory_order_relaxed),
					   (unsigned long long) atomic_load_explicit(&state->cost_queries[0], memory_order_relaxed),
					   (unsigned long long) atomic_load_explicit(&state->cost_queries[1], memory_order_relaxed),
					   (unsigned long long) atomic_load_explicit(&state->cost_queries[2], memory_order_relaxed),
					   (unsigned long long) atomic_load_explicit(&state->cost_queries[3], memory_order_relaxed));
	if (written < 0 || (size_t) written >= capacity)
		return 0;
	used = (size_t) written;
	for (uint32_t c = 0; c < state->copies; c++)
	{
		const AnnCuvsRoute *r = &state->routing[c];
		int			device = state->shards[(size_t) c * state->nshards].device;
		uint32_t	deviceMask = 0;
		uint64_t	rerankBytes = 0;
		int			rerankReady = 1;

		for (uint32_t s = 0; s < state->nshards; s++)
		{
			const AnnCuvsShard *shard = &state->shards[(size_t) c * state->nshards + s];

			deviceMask |= UINT32_C(1) << shard->device;
			rerankBytes += shard->rerankBytes;
			rerankReady &= shard->rerank != NULL;
		}

		written = snprintf(data + used, capacity - used,
						   "%s{\"copy\":%u,\"device\":%d,\"device_mask\":%u,\"rerank_bytes\":%llu,\"rerank_ready\":%s,\"active\":%u,\"device_active\":%u,"
						   "\"started\":%llu,\"finished\":%llu,\"queries\":%llu,\"migrations\":%llu,"
						   "\"failures\":%llu,\"reserved_ns\":%llu,\"device_batch_ns\":%llu}", c ? "," : "", c, device, deviceMask, (unsigned long long) rerankBytes, rerankReady ? "true" : "false",
						   atomic_load_explicit(&r->active, memory_order_relaxed),
						   atomic_load_explicit(&replicaRouter.devices[device].active, memory_order_relaxed),
						   (unsigned long long) atomic_load_explicit(&r->started, memory_order_relaxed),
						   (unsigned long long) atomic_load_explicit(&r->finished, memory_order_relaxed),
						   (unsigned long long) atomic_load_explicit(&r->queries, memory_order_relaxed),
						   (unsigned long long) atomic_load_explicit(&r->migrations, memory_order_relaxed),
						   (unsigned long long) atomic_load_explicit(&r->failures, memory_order_relaxed),
						   (unsigned long long) atomic_load_explicit(&r->reserved_ns, memory_order_relaxed),
						   (unsigned long long) AnnReplicaPredictCost(&replicaRouter.devices[device], ReplicaNow()));
		if (written < 0 || (size_t) written >= capacity - used)
			return 0;
		used += (size_t) written;
	}
	if (capacity - used < 3)
		return 0;
	memcpy(data + used, "]}", 3);
	return used + 2;
}

typedef struct AnnCuvsDevice
{
	cuvsResources_t resources;
	cuvsCagraSearchParams_t params;
	void	   *queries;
	uint32_t   *neighbors;
	float	   *distances;
	uint32_t   *bitset;
	uint32_t   *hostNeighbors;	/* pinned */
	float	   *hostDistances;	/* pinned */
	int64_t    *gatherKeys;		/* gather scan: filter keys of one shard */
	uint32_t   *gatherRows;		/* their shard rows, UINT32_MAX if absent */
	float	   *gatherScores;	/* squared distance per eligible row */
	float	   *gatherBest;		/* selected distances */
	int64_t    *gatherIndex;	/* selected positions in gatherKeys */
	int64_t    *hostKeys;		/* pinned */
	int64_t    *hostIndex;		/* pinned */
	size_t		gatherCount;
	size_t		gatherTop;
	size_t		queryBytes;
	size_t		resultCount;
	size_t		bitsetWords;
	float	   *matrixScores,
			   *matrixCandidates,
			   *matrixValues;
	int64_t    *matrixIndices,
			   *matrixRows;
	float	   *matrixHostValues;
	int64_t    *matrixHostRows;
	float	   *rerankQueries;
	size_t		rerankQueryCap;
	size_t		matrixScoreCap,
				matrixCandidateCap,
				matrixValueCap;
	size_t		matrixIndexCap,
				matrixRowCap,
				matrixHostValueCap,
				matrixHostRowCap;
} AnnCuvsDevice;

static _Thread_local AnnCuvsDevice laneDevices[ANN_CUVS_MAX_DEVICES];

static uint16_t
FloatToHalf(float value)
{
	uint32_t	bits;
	uint32_t	sign;
	int32_t		exponent;
	uint32_t	mantissa;

	memcpy(&bits, &value, sizeof(bits));
	sign = (bits >> 16) & 0x8000;
	exponent = (int32_t) ((bits >> 23) & 0xff) - 127 + 15;
	mantissa = bits & 0x7fffff;
	if (exponent >= 31)
		return (uint16_t) (sign | 0x7c00);
	if (exponent <= 0)
	{
		if (exponent < -10)
			return (uint16_t) sign;
		mantissa |= 0x800000;
		{
			uint32_t	shift = (uint32_t) (14 - exponent);
			uint32_t	half = mantissa >> shift;
			uint32_t	rest = mantissa & ((1U << shift) - 1);
			uint32_t	middle = 1U << (shift - 1);

			if (rest > middle || (rest == middle && (half & 1)))
				half++;

			return (uint16_t) (sign | half);
		}
	}
	{
		uint32_t	half = sign | ((uint32_t) exponent << 10) | (mantissa >> 13);
		uint32_t	rest = mantissa & 0x1fff;

		if (rest > 0x1000 || (rest == 0x1000 && (half & 1)))
			half++;
		return (uint16_t) half;
	}
}

static int
GrowDevice(void **pointer, size_t *capacity, size_t needed, size_t unit, size_t floor, cudaStream_t stream)
{
	if (needed <= *capacity)
		return 1;

	/*
	 * Lane buffers grow rarely: to twice the need, never below a floor, and
	 * stream-ordered, because cudaFree and cudaMalloc synchronize the whole
	 * device and stalled every lane while batches grew under load (nsys: 99
	 * ms per cudaFree on average, 1.7 s at worst).
	 */
	if (*pointer)
		(void) cudaFreeAsync(*pointer, stream);
	*pointer = NULL;
	*capacity = 0;
	needed = needed * 2 > floor ? needed * 2 : floor;
	if (cudaMallocAsync(pointer, needed * unit, stream) != cudaSuccess)
	{
		*pointer = NULL;
		return 0;
	}
	*capacity = needed;
	return 1;
}

/* Pinned host buffers: the same growth, so cudaFreeHost's device sync is rare. */
static int
GrowPinned(void **pointer, size_t *capacity, size_t needed, size_t unit, size_t floor)
{
	if (needed <= *capacity)
		return 1;
	if (*pointer)
		(void) cudaFreeHost(*pointer);
	*pointer = NULL;
	*capacity = 0;
	needed = needed * 2 > floor ? needed * 2 : floor;
	if (cudaMallocHost(pointer, needed * unit) != cudaSuccess)
	{
		*pointer = NULL;
		return 0;
	}
	*capacity = needed;
	return 1;
}

#define ANN_LANE_QUERY_FLOOR (1024 * 1024)	/* bytes */
#define ANN_LANE_RESULT_FLOOR 65536 /* (neighbor, distance) pairs */
#define ANN_LANE_BITSET_FLOOR 32768 /* words: 1M rows */
#define ANN_LANE_ROW_FLOOR 262144	/* gather rows */

/* Lazily create this lane's state on a device and size its buffers. */
static AnnCuvsDevice *
LaneDevice(int device, size_t queryBytes, size_t results, size_t bitsetWords)
{
	AnnCuvsDevice *lane = &laneDevices[device];
	cudaStream_t stream;

	if (cudaSetDevice(device) != cudaSuccess)
		return NULL;

	/*
	 * A per-lane workspace pool keeps concurrent searches from serializing on
	 * device-wide allocation.
	 */
	if (lane->resources == 0 &&
		(cuvsResourcesCreate(&lane->resources) != CUVS_SUCCESS ||
		 cuvsResourcesSetWorkspacePool(lane->resources, ANN_CUVS_WORKSPACE_BYTES) != CUVS_SUCCESS ||
		 cuvsCagraSearchParamsCreate(&lane->params) != CUVS_SUCCESS))
		return NULL;
	if (cuvsStreamGet(lane->resources, &stream) != CUVS_SUCCESS ||
		!GrowDevice(&lane->queries, &lane->queryBytes, queryBytes, 1, ANN_LANE_QUERY_FLOOR, stream) ||
		!GrowDevice((void **) &lane->bitset, &lane->bitsetWords, bitsetWords, sizeof(uint32_t),
					ANN_LANE_BITSET_FLOOR, stream))
		return NULL;
	if (results > lane->resultCount)
	{
		size_t		count = results * 2 > ANN_LANE_RESULT_FLOOR ? results * 2 : ANN_LANE_RESULT_FLOOR;
		size_t		have = lane->resultCount;
		size_t		hostCount = lane->resultCount;

		lane->resultCount = 0;
		if (!GrowDevice((void **) &lane->neighbors, &have, count, sizeof(uint32_t), 0, stream))
			return NULL;
		have = hostCount;
		if (!GrowDevice((void **) &lane->distances, &have, count, sizeof(float), 0, stream))
			return NULL;
		have = hostCount;
		if (!GrowPinned((void **) &lane->hostNeighbors, &have, count, sizeof(uint32_t), 0))
			return NULL;
		have = hostCount;
		if (!GrowPinned((void **) &lane->hostDistances, &have, count, sizeof(float), 0))
			return NULL;
		lane->resultCount = have;
	}
	return lane;
}

typedef struct LoadArgs
{
	AnnIndexFile *file;
	AnnCuvsIndex *state;
	uint32_t	shard;			/* shard in the manifest */
	uint32_t	slot;			/* entry in state->shards */
	int			device;
	int			threaded;
	AnnStatus	status;
} LoadArgs;

static void *
LoadShard(void *argument)
{
	LoadArgs   *args = argument;
	AnnCuvsShard *shard = &args->state->shards[args->slot];
	char		path[1400];
	int64_t		dims = 0,
				size = 0;

	args->status = ANN_BACKEND_ERROR;
	shard->device = args->device;
	snprintf(path, sizeof(path), "%s/%s", args->file->directory, args->file->shards[args->shard].file);
	if (cudaSetDevice(shard->device) != cudaSuccess ||
		cuvsResourcesCreate(&shard->resources) != CUVS_SUCCESS ||
		cuvsCagraIndexCreate(&shard->index) != CUVS_SUCCESS ||
		cuvsCagraDeserializeGraphAndDataset(shard->resources, path, shard->index,
											&shard->dataset) != CUVS_SUCCESS ||
		cuvsStreamSync(shard->resources) != CUVS_SUCCESS)
	{
		const char *message = cuvsGetLastErrorText();

		fprintf(stderr, "cuVS: cannot load %s: %s\n", path, message ? message : "failed");
		return NULL;
	}
	if (cuvsCagraIndexGetDims(shard->index, &dims) != CUVS_SUCCESS ||
		cuvsCagraIndexGetSize(shard->index, &size) != CUVS_SUCCESS ||
		dims != args->file->dimensions || size != args->file->shards[args->shard].rows)
	{
		fprintf(stderr, "cuVS: %s has %lld rows x %lld dims, manifest says %u x %u\n", path,
				(long long) size, (long long) dims, args->file->shards[args->shard].rows,
				args->file->dimensions);
		args->status = ANN_INVALID;
		return NULL;
	}

	/*
	 * The gather scan reads rows straight from the index's device dataset,
	 * whose rows may be padded.
	 */
	{
		DLManagedTensor view = {0};

		if (cuvsCagraIndexGetDataset(shard->index, &view) == CUVS_SUCCESS &&
			view.dl_tensor.ndim == 2 && view.dl_tensor.device.device_type == kDLCUDA &&
			view.dl_tensor.dtype.bits == (strcmp(args->file->dtype, "float16") == 0 ? 16 : 32))
		{
			shard->rows = view.dl_tensor.data;
			shard->stride = view.dl_tensor.strides ? view.dl_tensor.strides[0] : view.dl_tensor.shape[1];
		}
		if (view.deleter)
			view.deleter(&view);
	}
	if (cudaMalloc((void **) &shard->keys, (size_t) args->file->shards[args->shard].rows * sizeof(int64_t)) != cudaSuccess ||
		cudaMemcpy(shard->keys, args->file->keys + args->file->shards[args->shard].first,
				   (size_t) args->file->shards[args->shard].rows * sizeof(int64_t), cudaMemcpyHostToDevice) != cudaSuccess)
	{
		if (shard->keys)
			(void) cudaFree(shard->keys);
		shard->keys = NULL;
	}

	/*
	 * On Blackwell GPUs the gather scan reads the quantized copy instead: 4
	 * or 8 bits per dimension rather than 16 or 32.
	 */
	if (args->file->quant != ANN_QUANT_NONE && AnnIndexConfig.quantized_scan)
		(void) AnnBlackwellUpload(args->file, args->shard, shard->device, &shard->quant);
	else if (AnnIndexConfig.quantized_scan && shard->rows)
	{
		/* Grouped searches can read the FP16 or FP32 dataset in place. */
		(void) AnnBlackwellView(shard->rows, shard->stride, strcmp(args->file->dtype, "float16") == 0,
								args->file->dimensions, shard->device, &shard->quant);
	}
	if (AnnIndexConfig.gpu_rerank)
	{
		if (strcmp(args->file->dtype, "float32") == 0 && shard->rows)
		{
			shard->rerank = (float *) shard->rows;
			shard->rerankStride = shard->stride;
		}
		else
		{
			size_t		bytes = (size_t) args->file->shards[args->shard].rows * args->file->dimensions * sizeof(float);

			if (cudaMalloc((void **) &shard->rerank, bytes) == cudaSuccess &&
				cudaMemcpy(shard->rerank, args->file->vectors +
						   args->file->shards[args->shard].first * args->file->dimensions,
						   bytes, cudaMemcpyHostToDevice) == cudaSuccess)
			{
				shard->rerankBytes = bytes;
				shard->rerankStride = args->file->dimensions;
			}
			else
			{
				if (shard->rerank)
					(void) cudaFree(shard->rerank);
				shard->rerank = NULL;
				(void) cudaGetLastError();
				fprintf(stderr, "cuVS: FP32 rerank copy unavailable on device %d shard %u; using host reranking\n",
						shard->device, args->shard);
			}
		}
	}
	args->status = ANN_OK;
	return NULL;
}

static void
DestroyIndexState(AnnCuvsIndex *state)
{
	if (state == NULL)
		return;
	for (uint32_t s = 0; state->shards && s < state->nshards * state->copies; s++)
	{
		AnnCuvsShard *shard = &state->shards[s];

		(void) cudaSetDevice(shard->device);
		if (shard->quant.quant != ANN_QUANT_NONE)
			AnnBlackwellFree(&shard->quant);
		if (shard->keys)
			(void) cudaFree(shard->keys);
		if (shard->rerankBytes)
			(void) cudaFree(shard->rerank);
		if (shard->index)
			(void) cuvsCagraIndexDestroy(shard->index);
		if (shard->dataset)
			(void) cuvsDatasetDestroy(shard->dataset);
		if (shard->resources)
			(void) cuvsResourcesDestroy(shard->resources);
	}
	free(state->shards);
	free(state);
}

/*
 * AnnIndexDeviceDestroy: an unloaded index's shards and device arrays. Its
 * cached bitmaps were freed into the default pool, which keeps freed blocks
 * (see UploadStream): trim that. Memory cuVS freed into the RMM pool stays
 * with this process for later loads; with --device-allocator plain it goes
 * back to the device at once. An unload costs one device-wide
 * sync per device it used, so its frees land before the trim.
 */
static void
DestroyDevice(void *device)
{
	AnnCuvsIndex *state = device;
	int			used[ANN_INDEX_MAX_SHARDS] = {0};

	for (uint32_t s = 0; state && state->shards && s < state->nshards * state->copies; s++)
		if (state->shards[s].device >= 0 && state->shards[s].device < ANN_INDEX_MAX_SHARDS)
			used[state->shards[s].device] = 1;
	DestroyIndexState(state);
	for (int d = 0; d < ANN_INDEX_MAX_SHARDS; d++)
	{
		cudaMemPool_t pool;

		if (!used[d] || cudaSetDevice(d) != cudaSuccess)
			continue;
		(void) cudaDeviceSynchronize();
		if (cudaDeviceGetDefaultMemPool(&pool, d) == cudaSuccess)
			(void) cudaMemPoolTrimTo(pool, 0);
	}
}

/*
 * Cached bitmaps are copied and freed on a non-blocking stream per device,
 * stream-ordered: cudaMalloc and cudaFree would wait for every lane's work
 * on the device, so each new predicate would stall all searches.
 */
static pthread_mutex_t uploadMutex = PTHREAD_MUTEX_INITIALIZER;
static cudaStream_t uploadStreams[ANN_INDEX_MAX_SHARDS];

static cudaStream_t
UploadStream(int device)
{
	cudaStream_t stream;

	(void) pthread_mutex_lock(&uploadMutex);
	if (uploadStreams[device] == NULL)
	{
		cudaMemPool_t pool;
		uint64_t	keep = UINT64_MAX;

		if (cudaStreamCreateWithFlags(&uploadStreams[device], cudaStreamNonBlocking) != cudaSuccess)
			uploadStreams[device] = NULL;
		/* Keep freed bitmap memory in the pool instead of unmapping it. */
		else if (cudaDeviceGetDefaultMemPool(&pool, device) == cudaSuccess)
			(void) cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &keep);
	}
	stream = uploadStreams[device];
	(void) pthread_mutex_unlock(&uploadMutex);
	return stream;
}

static int
UploadArray(int device, void **target, const void *source, size_t size)
{
	cudaStream_t stream;

	*target = NULL;
	if (cudaSetDevice(device) != cudaSuccess || (stream = UploadStream(device)) == NULL)
		return 0;
	if (cudaMallocAsync(target, size, stream) != cudaSuccess)
	{
		*target = NULL;
		return 0;
	}
	return cudaMemcpyAsync(*target, source, size, cudaMemcpyHostToDevice, stream) == cudaSuccess &&
		cudaStreamSynchronize(stream) == cudaSuccess;
}

/* Shards load in parallel, one thread each; any failure rejects the load. */
static void
FreeDeviceArrays(const AnnCuvsIndex *state, void **arrays)
{
	for (size_t i = 0; arrays && i < (size_t) state->copies * state->nshards; i++)
		if (arrays[i])
		{
			cudaStream_t stream;

			(void) cudaSetDevice(state->shards[i].device);
			if ((stream = UploadStream(state->shards[i].device)) != NULL)
				(void) cudaFreeAsync(arrays[i], stream);
		}
	free(arrays);
}

/*
 * Cached filter bitmaps: copy each shard's bitset to every device holding
 * that shard, once, so filtered CAGRA reads it in place, and, when the
 * bitmap's rows are known, each shard's rows as u32 so the gather scans
 * upload nothing per query. A failure frees what was copied and leaves
 * that form on the per-query path.
 */
static void
UploadBitmap(AnnIndexFile *file, AnnIndexBitmap *bitmap)
{
	AnnCuvsIndex *state = file->device;
	size_t		n = (size_t) state->copies * state->nshards;
	void	  **device = NULL;
	void	  **rows = NULL;
	uint32_t   *staged = NULL;
	uint64_t	start[ANN_INDEX_MAX_SHARDS + 1];
	uint64_t	bytes = 0;
	int			ok = 1;

	/*
	 * Keep on the GPUs only what this filter's routes read: bitsets for
	 * filtered CAGRA, taken by filters past the bitset gather limit or whose
	 * rows are not kept, and rows for the gather scans, which start past the
	 * exact limit. Smaller filters run exact on the CPU; a route that finds
	 * no device copy copies the filter per query.
	 */
	int			needBitsets = bitmap->words != NULL &&
		(AnnCostMatches(AnnIndexCostProfile, file->rows, file->dimensions, file->generationValue) ||
		 bitmap->ordinals == NULL || bitmap->count > AnnIndexConfig.bitset_gather_limit);
	int			needRows = bitmap->ordinals != NULL && bitmap->count > AnnIndexConfig.exact_limit;

	if (needBitsets)
	{
		device = calloc(n, sizeof(void *));
		ok = device != NULL;
		for (size_t i = 0; ok && i < n; i++)
		{
			uint32_t	s = (uint32_t) (i % state->nshards);
			size_t		size = ((size_t) file->shards[s].rows + 31) / 32 * sizeof(uint32_t);

			ok = UploadArray(state->shards[i].device, &device[i], bitmap->bitsets[s], size);
			bytes += size;
		}
		if (!ok)
		{
			fprintf(stderr, "cuVS: could not keep filter bitmap %llu on the GPUs; it is copied per query\n",
					(unsigned long long) bitmap->id);
			FreeDeviceArrays(state, device);
			return;
		}
		bitmap->device = device;
		bitmap->deviceBytes = bytes;
	}
	if (!needRows || bitmap->count == 0)
		return;

	/* Rows are sorted: split them at shard boundaries, then relative rows. */
	start[0] = 0;
	start[state->nshards] = bitmap->count;
	for (uint32_t s = 1; s < state->nshards; s++)
	{
		uint64_t	low = start[s - 1],
					high = bitmap->count;

		while (low < high)
		{
			uint64_t	middle = low + (high - low) / 2;

			if (bitmap->ordinals[middle] < file->shards[s].first)
				low = middle + 1;
			else
				high = middle;
		}
		start[s] = low;
	}
	rows = calloc(n, sizeof(void *));
	staged = malloc(bitmap->count * sizeof(uint32_t));
	ok = rows != NULL && staged != NULL;
	for (uint32_t s = 0; ok && s < state->nshards; s++)
		for (uint64_t i = start[s]; i < start[s + 1]; i++)
			staged[i] = (uint32_t) (bitmap->ordinals[i] - file->shards[s].first);
	for (size_t i = 0; ok && i < n; i++)
	{
		uint32_t	s = (uint32_t) (i % state->nshards);
		size_t		size = (start[s + 1] - start[s]) * sizeof(uint32_t);

		if (size == 0)
			continue;
		ok = UploadArray(state->shards[i].device, &rows[i], staged + start[s], size);
		bytes += size;
	}
	free(staged);
	if (ok)
	{
		bitmap->deviceRows = rows;
		bitmap->deviceBytes = bytes;
		return;
	}
	fprintf(stderr, "cuVS: could not keep filter bitmap %llu's rows on the GPUs; they are uploaded per query\n",
			(unsigned long long) bitmap->id);
	FreeDeviceArrays(state, rows);
}

static void
ReleaseBitmap(AnnIndexFile *file, AnnIndexBitmap *bitmap)
{
	AnnCuvsIndex *state = file->device;

	if (state)
	{
		FreeDeviceArrays(state, bitmap->device);
		FreeDeviceArrays(state, bitmap->deviceRows);
	}
	bitmap->device = NULL;
	bitmap->deviceRows = NULL;
	bitmap->deviceBytes = 0;
}

static AnnStatus
LoadIndex(const char *name, const char *directory)
{
	AnnIndexFile *file;
	AnnCuvsIndex *state;
	LoadArgs   *args;
	pthread_t  *threads;
	uint32_t	copies;
	uint32_t	total;
	int			devices = 0;
	AnnStatus	status = AnnIndexFileOpen(name, directory, &file);

	if (status != ANN_OK)
		return status;
	AnnIndexBitmapDevice.upload = UploadBitmap;
	AnnIndexBitmapDevice.release = ReleaseBitmap;
	AnnIndexDeviceDestroy = DestroyDevice;
	if (file->nshards == 0 || cudaGetDeviceCount(&devices) != cudaSuccess || devices <= 0)
	{
		/* No CAGRA shards or devices: serve exact results from the files. */
		if ((status = AnnIndexRegister(file)) != ANN_OK)
			AnnIndexFileClose(file);
		return status;
	}
	if (devices > ANN_CUVS_MAX_DEVICES)
		devices = ANN_CUVS_MAX_DEVICES;
	state = calloc(1, sizeof(*state));
	copies = AnnIndexConfig.replicate ? (uint32_t) devices : 1;
	total = file->nshards * copies;
	args = calloc(total, sizeof(*args));
	threads = calloc(total, sizeof(*threads));
	if (state)
		state->shards = calloc(total, sizeof(*state->shards));
	if (state == NULL || state->shards == NULL || args == NULL || threads == NULL)
	{
		if (state)
			free(state->shards);
		free(state);
		free(args);
		free(threads);
		AnnIndexFileClose(file);
		return ANN_NOMEM;
	}
	state->nshards = file->nshards;
	state->copies = copies;
	atomic_init(&state->matrix_queries, 0);
	atomic_init(&state->matrix_peak_score_bytes, 0);
	atomic_init(&state->gpu_rerank_candidates, 0);
	for (int mode = 0; mode < 4; mode++)
		atomic_init(&state->cost_queries[mode], 0);
	for (uint32_t c = 0; c < copies; c++)
	{
		atomic_init(&state->routing[c].active, 0);
		atomic_init(&state->routing[c].started, 0);
		atomic_init(&state->routing[c].finished, 0);
		atomic_init(&state->routing[c].queries, 0);
		atomic_init(&state->routing[c].migrations, 0);
		atomic_init(&state->routing[c].failures, 0);
		atomic_init(&state->routing[c].reserved_ns, 0);
	}

	state->fp16 = strcmp(file->dtype, "float16") == 0;
	cuvsSetLogLevel(CUVS_LOG_LEVEL_OFF);

	/*
	 * Blocking synchronization lets waiting lanes sleep instead of spinning a
	 * core each, leaving CPUs to PostgreSQL. It must precede context
	 * creation. Allocations outside lane pools (cuVS allocates temporaries
	 * per filtered search) use stream-ordered allocation, which avoids
	 * device-wide syncs, or with --memory-pool an RMM pool per GPU, which
	 * also avoids the driver allocator's process-wide locks. The pool (the
	 * default, --memory-pool 80) reserves a quarter of its cap up front (21%
	 * of each GPU) and never shrinks, so loads take that much and unloads
	 * return nothing to the device: three collection layouts held 99.6 GB per
	 * GPU against 37.6 GB with --device-allocator plain (cudaMalloc), which
	 * costs unfiltered throughput (about a quarter lower) but sizes memory to
	 * the indexes and returns it on unload.
	 */
	for (int d = 0; d < devices; d++)
		if (cudaSetDevice(d) != cudaSuccess ||
			(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync) != cudaSuccess &&
			 cudaGetLastError() != cudaErrorSetOnActiveProcess) ||
			(AnnIndexConfig.plain_allocator ? cuvsRMMMemoryResourceReset() :
			 AnnIndexConfig.memory_pool ?
			 cuvsRMMPoolMemoryResourceEnable((int) AnnIndexConfig.memory_pool / 4 + 1,
											 (int) AnnIndexConfig.memory_pool, false) :
			 cuvsRMMAsyncMemoryResourceEnable()) != CUVS_SUCCESS)
			fprintf(stderr, "cuVS: memory resource unavailable on device %d\n", d);
	for (uint32_t s = 0; s < total; s++)
	{
		uint32_t	copy = s / file->nshards;
		uint32_t	shard = s % file->nshards;

		args[s] = (LoadArgs)
		{
			file, state, shard, s, copies > 1 ? (int) copy : (int) (shard % (uint32_t) devices),
				1, ANN_BACKEND_ERROR
		};
		if (pthread_create(&threads[s], NULL, LoadShard, &args[s]) != 0)
		{
			args[s].threaded = 0;
			LoadShard(&args[s]);
		}
	}
	for (uint32_t s = 0; s < total; s++)
	{
		if (args[s].threaded)
			(void) pthread_join(threads[s], NULL);
		if (args[s].status != ANN_OK && status == ANN_OK)
			status = args[s].status;
	}
	free(args);
	free(threads);
	if (status == ANN_OK && AnnIndexConfig.quantized_scan)
	{
		state->grouped = 1;
		for (uint32_t s = 0; s < total; s++)
			state->grouped &= state->shards[s].quant.quant != ANN_QUANT_NONE;
		if (state->grouped && file->quant == ANN_QUANT_NONE)
			fprintf(stderr, "Blackwell module: grouped %s searches for index %s read the dataset in place\n",
					file->dtype, name);
	}
	if (status == ANN_OK && file->quant != ANN_QUANT_NONE && AnnIndexConfig.quantized_scan)
	{
		size_t		bytes = 0;

		state->quantized = 1;
		for (uint32_t s = 0; s < total; s++)
		{
			state->quantized &= state->shards[s].quant.quant != ANN_QUANT_NONE;
			bytes += state->shards[s].quant.bytes;
		}
		file->quantActive = state->quantized;
		fprintf(stderr, "Blackwell module: %s gather scan for index %s %s (%.1f MB of quantized rows on the GPUs)\n",
				AnnIndexQuantName(file->quant), name, state->quantized ? "on" : "off: not every GPU is ready",
				(double) bytes / 1048576.0);
	}
	if (status == ANN_OK && state->copies > AnnIndexConfig.index_copies)
		AnnIndexConfig.index_copies = state->copies;
	file->device = state;
	if (status == ANN_OK)
		status = AnnIndexRegister(file);
	if (status != ANN_OK)
	{
		DestroyIndexState(state);
		AnnIndexFileClose(file);
	}
	return status;
}

static int
HitCompare(const void *left, const void *right)
{
	const AnnIndexHit *a = left;
	const AnnIndexHit *b = right;

	if (a->distance != b->distance)
		return a->distance < b->distance ? -1 : 1;
	return a->key < b->key ? -1 : a->key > b->key;
}

static float
ExactDistance(const AnnIndexFile *file, uint64_t row, const float *query)
{
	const float *vector = file->vectors + row * file->dimensions;
	double		sum = 0;

	for (uint32_t d = 0; d < file->dimensions; d++)
	{
		double		delta = (double) vector[d] - query[d];

		sum += delta * delta;
	}
	return (float) sqrt(sum);
}

/* Host memory hints only; keep ExactDistance's arithmetic unchanged. */
static void
PrefetchExactVector(const AnnIndexFile *file, uint64_t row)
{
#if defined(__GNUC__) || defined(__clang__)
	const float *vector;

	if (row >= file->rows)
		return;
	vector = file->vectors + row * file->dimensions;
	for (uint32_t d = 0; d < file->dimensions; d += 16)
		__builtin_prefetch(vector + d, 0, 3);
#else
	(void) file;
	(void) row;
#endif
}

/*
 * Turn one shard's CAGRA neighbours into a sorted hit list, re-scoring each
 * with FP32 vectors so FP16 datasets still return exact distances. Rows
 * outside the eligibility bitset (when given) and padding are dropped.
 */
static uint32_t
ShardHits(const AnnIndexFile *file, uint32_t s, const uint32_t *neighbors, uint32_t n,
		  const float *query, const uint32_t *bitset, AnnIndexHit *hits, const float *exact)
{
	uint32_t	count = 0;

	for (uint32_t i = 0; i < n; i++)
	{
		uint32_t	local = neighbors[i];
		uint32_t	distance = AnnIndexConfig.rerank_prefetch;

		if (!exact && distance && n - i > distance)
		{
			uint32_t	future = neighbors[i + distance];

			if (future < file->shards[s].rows)
				PrefetchExactVector(file, file->shards[s].first + future);
		}

		if (local >= file->shards[s].rows ||
			(bitset && !(bitset[local / 32] & (UINT32_C(1) << (local % 32)))))
			continue;
		hits[count].key = file->keys[file->shards[s].first + local];
		hits[count].ordinal = file->shards[s].first + local;
		hits[count].distance = exact ? exact[i] : ExactDistance(file, file->shards[s].first + local, query);
		count++;
	}
	qsort(hits, count, sizeof(*hits), HitCompare);
	return count;
}

static DLManagedTensor
QueryTensor(void *data, int device, int half, int64_t *shape)
{
	DLManagedTensor tensor = Tensor(data, kDLCUDA, device, kDLFloat, 2, shape);

	if (half)
		tensor.dl_tensor.dtype.bits = 16;
	return tensor;
}

/* Copy queries to the device, converting to FP16 for half datasets. */
static int
UploadQueries(AnnCuvsDevice *lane, const AnnIndexJob *const *jobs, uint32_t m, uint32_t dims,
			  int half, cudaStream_t stream, void *host)
{
	size_t		unit = half ? 2 : 4;

	for (uint32_t j = 0; j < m; j++)
		for (uint32_t d = 0; d < dims; d++)
		{
			if (half)
				((uint16_t *) host)[(size_t) j * dims + d] = FloatToHalf(jobs[j]->query[d]);
			else
				((float *) host)[(size_t) j * dims + d] = jobs[j]->query[d];
		}
	return cudaMemcpyAsync(lane->queries, host, (size_t) m * dims * unit, cudaMemcpyHostToDevice,
						   stream) == cudaSuccess;
}

static int	RerankDevice(const AnnCuvsShard *shard, uint32_t dims, const uint32_t *rows,
						 uint32_t eligible, uint32_t nrows, void *positions, uint32_t top,
						 uint32_t nq, const float *queries, float *out, cudaStream_t stream, int wide);

static uint32_t
RoundTopK(uint32_t value)
{
	value = (value + 31) / 32 * 32;
	return value < 32 ? 32 : value;
}

/*
 * Search every shard for a set of queries, filtered by one bitset per shard
 * when given (single query). Returns 0 on any device error.
 */
static int
SearchShards(const AnnIndexFile *file, AnnCuvsIndex *state, const AnnIndexJob *const *jobs,
			 uint32_t m, uint32_t k, uint32_t itopk, uint32_t **bitsets, void *const *deviceBitsets,
			 const int *useShard, AnnIndexHit *lists, uint32_t *counts)
{
	uint32_t	dims = file->dimensions;
	size_t		unit = state->fp16 ? 2 : 4;
	void	   *host = malloc((size_t) m * dims * unit);
	int			uploaded[ANN_CUVS_MAX_DEVICES] = {0};
	int			exactUploaded[ANN_CUVS_MAX_DEVICES] = {0};
	int			exactGpu[ANN_INDEX_MAX_SHARDS] = {0};
	float	   *exactHost = AnnIndexConfig.gpu_rerank && laneRerank != 0 && state->fp16 ? malloc((size_t) m * dims * sizeof(float)) : NULL;
	size_t		slot[ANN_INDEX_MAX_SHARDS];
	size_t		bitsetSlot[ANN_INDEX_MAX_SHARDS];
	size_t		perDevice[ANN_CUVS_MAX_DEVICES] = {0};
	size_t		bitsetWords[ANN_CUVS_MAX_DEVICES] = {0};
	AnnCuvsShard *shards;
	uint32_t	copy;
	int			ok = host != NULL;

	if (exactHost)
		for (uint32_t j = 0; j < m; j++)
			memcpy(exactHost + (size_t) j * dims, jobs[j]->query, dims * sizeof(float));
	copy = laneCopy;
	shards = state->shards + (size_t) copy * state->nshards;

	/*
	 * Each shard gets m * k result slots, and its own bitset slot, in its
	 * device's buffers, so no shard waits for another before launching.
	 */
	for (uint32_t s = 0; s < state->nshards; s++)
	{
		int			device = shards[s].device;

		slot[s] = perDevice[device];
		bitsetSlot[s] = bitsetWords[device];
		if (!useShard[s])
			continue;
		perDevice[device] += (size_t) m * k;
		if (bitsets)
			bitsetWords[device] += ((size_t) file->shards[s].rows + 31) / 32;
	}
	for (uint32_t s = 0; ok && s < state->nshards; s++)
	{
		AnnCuvsShard *shard = &shards[s];
		size_t		words = ((size_t) file->shards[s].rows + 31) / 32;
		AnnCuvsDevice *lane;
		cudaStream_t stream;
		int64_t		queryShape[2] = {m, dims};
		int64_t		resultShape[2] = {m, k};
		int64_t		bitsetShape[1] = {(int64_t) words};
		DLManagedTensor queries,
					neighbors,
					distances,
					mask;
		cuvsFilter	filter = {0, NO_FILTER};

		if (!useShard[s])
			continue;
		lane = LaneDevice(shard->device, (size_t) m * dims * unit, perDevice[shard->device],
						  bitsets ? bitsetWords[shard->device] : 1);
		if (lane == NULL || cuvsStreamGet(lane->resources, &stream) != CUVS_SUCCESS)
		{
			ok = 0;
			break;
		}
		if (!uploaded[shard->device] &&
			!(uploaded[shard->device] = UploadQueries(lane, jobs, m, dims, state->fp16, stream, host)))
		{
			ok = 0;
			break;
		}
		lane->params->itopk_size = itopk > laneProfileItopk ? itopk : laneProfileItopk;
		lane->params->persistent = AnnIndexConfig.persistent != 0;

		/* Lane parameters are shared: set every filtered knob each time. */
		lane->params->algo = AUTO;
		lane->params->search_width = 1;
		lane->params->max_iterations = 0;
		if (bitsets)
		{
			if (AnnIndexConfig.filtered_algo == 1)
				lane->params->algo = SINGLE_CTA;
			else if (AnnIndexConfig.filtered_algo == 2)
				lane->params->algo = MULTI_CTA;
			if (AnnIndexConfig.filtered_width)
				lane->params->search_width = AnnIndexConfig.filtered_width;
			lane->params->max_iterations = AnnIndexConfig.filtered_iterations;
		}
		if (lane->params->algo != MULTI_CTA && lane->params->itopk_size > 512)
			lane->params->itopk_size = 512;
		queries = QueryTensor(lane->queries, shard->device, state->fp16, queryShape);
		neighbors = Tensor(lane->neighbors + slot[s], kDLCUDA, shard->device, kDLUInt, 2, resultShape);
		distances = Tensor(lane->distances + slot[s], kDLCUDA, shard->device, kDLFloat, 2, resultShape);
		if (bitsets && deviceBitsets && deviceBitsets[(size_t) copy * state->nshards + s])
		{
			/* Cached on this shard's device when the bitmap was pushed. */
			mask = Tensor(deviceBitsets[(size_t) copy * state->nshards + s], kDLCUDA, shard->device,
						  kDLUInt, 1, bitsetShape);
			filter.type = BITSET;
			filter.addr = (uintptr_t) &mask;
		}
		else if (bitsets)
		{
			if (cudaMemcpyAsync(lane->bitset + bitsetSlot[s], bitsets[s], words * sizeof(uint32_t),
								cudaMemcpyHostToDevice, stream) != cudaSuccess)
			{
				ok = 0;
				break;
			}
			mask = Tensor(lane->bitset + bitsetSlot[s], kDLCUDA, shard->device, kDLUInt, 1, bitsetShape);
			filter.type = BITSET;
			filter.addr = (uintptr_t) &mask;
		}
		if (cuvsCagraSearch(lane->resources, lane->params, shard->index, &queries, &neighbors,
							&distances, filter) != CUVS_SUCCESS)
		{
			ok = 0;
			break;
		}
		if (shard->rerank && laneRerank != 0 && (!state->fp16 || exactHost))
		{
			const float *exactQueries = lane->queries;
			int			ready = 1;

			if (state->fp16)
			{
				ready = GrowDevice((void **) &lane->rerankQueries, &lane->rerankQueryCap,
								   (size_t) m * dims, sizeof(float), 0, stream);
				if (ready && !exactUploaded[shard->device])
				{
					ready = cudaMemcpyAsync(lane->rerankQueries, exactHost, (size_t) m * dims * sizeof(float),
											cudaMemcpyHostToDevice, stream) == cudaSuccess;
					exactUploaded[shard->device] = ready;
				}
				exactQueries = lane->rerankQueries;
			}
			if (ready && RerankDevice(shard, dims, NULL, file->shards[s].rows, file->shards[s].rows,
									  lane->neighbors + slot[s], k, m, exactQueries,
									  lane->distances + slot[s], stream, 0))
			{
				exactGpu[s] = 1;
				if (cudaMemcpyAsync(lane->hostDistances + slot[s], lane->distances + slot[s],
									(size_t) m * k * sizeof(float), cudaMemcpyDeviceToHost, stream) != cudaSuccess)
				{
					ok = 0;
					break;
				}
			}
		}
		if (cudaMemcpyAsync(lane->hostNeighbors + slot[s], lane->neighbors + slot[s],
							(size_t) m * k * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream) != cudaSuccess)
		{
			const char *message = cuvsGetLastErrorText();

			fprintf(stderr, "cuVS index search: %s\n", message ? message : "failed");
			ok = 0;
			break;
		}
	}
	for (int d = 0; d < ANN_CUVS_MAX_DEVICES; d++)
		if (laneDevices[d].resources && perDevice[d] &&
			(cudaSetDevice(d) != cudaSuccess || cuvsStreamSync(laneDevices[d].resources) != CUVS_SUCCESS))
			ok = 0;
	for (uint32_t s = 0; ok && s < state->nshards; s++)
		for (uint32_t j = 0; j < m; j++)
		{
			AnnIndexHit *hits = lists + ((size_t) j * state->nshards + s) * k;

			counts[j * state->nshards + s] = useShard[s] ?
				ShardHits(file, s, laneDevices[shards[s].device].hostNeighbors + slot[s] + (size_t) j * k,
						  k, jobs[j]->query, bitsets ? bitsets[s] : NULL, hits, exactGpu[s] ?
						  laneDevices[shards[s].device].hostDistances + slot[s] + (size_t) j * k : NULL) : 0;
		}
	if (ok)
		for (uint32_t s = 0; s < state->nshards; s++)
			if (exactGpu[s])
				(void) atomic_fetch_add_explicit(&state->gpu_rerank_candidates, (uint64_t) m * k, memory_order_relaxed);
	free(exactHost);
	free(host);
	return ok;
}

/* Merge per-shard lists for query j into its job's results. */
static void
MergeJob(AnnIndexJob *job, AnnIndexHit *lists, uint32_t *counts, uint32_t nshards, uint32_t k)
{
	AnnIndexHit *heads[ANN_INDEX_MAX_SHARDS];

	for (uint32_t s = 0; s < nshards; s++)
		heads[s] = lists + (size_t) s * k;
	job->nresults = AnnIndexTreeMerge(heads, counts, nshards, job->k, job->results);
	job->status = ANN_OK;
}

/*
 * Gather scan: exact filtered search over only the eligible rows, read from
 * the CAGRA dataset already in device memory. One warp scores one row, then
 * cuVS selects the top k. Cost follows the eligible count, not the shard
 * size, so it beats a mostly-masked graph walk for selective filters (an
 * inverted posting list scanned on the GPU). The kernel is compiled with
 * NVRTC on first use, so building the worker needs no CUDA compiler.
 */
static const char GatherSource[] =
"__device__ __forceinline__ float load(const float *p) { return *p; }\n"
"__device__ __forceinline__ float load(const unsigned short *p)\n"
"{ float f; unsigned short h = *p; asm(\"cvt.f32.f16 %0, %1;\" : \"=f\"(f) : \"h\"(h)); return f; }\n"
"template <typename T> __device__ void gather(const T *data, long long stride, const unsigned *rows,\n"
"    unsigned n, const float *query, int dims, float *out)\n"
"{\n"
"  unsigned i = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;\n"
"  int lane = threadIdx.x & 31;\n"
"  if (i >= n) return;\n"
"  if (rows[i] == 0xffffffffu) { if (lane == 0) out[i] = __int_as_float(0x7f800000); return; }\n"
"  const T *v = data + (long long) rows[i] * stride;\n"
"  float s = 0.0f;\n"
"  for (int d = lane; d < dims; d += 32) { float t = load(v + d) - query[d]; s += t * t; }\n"
"  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);\n"
"  if (lane == 0) out[i] = s;\n"
"}\n"
"extern \"C\" __global__ void lookup(const long long *index, unsigned rows, const long long *keys,\n"
"    unsigned n, unsigned *out)\n"
"{\n"
"  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"  if (i >= n) return;\n"
"  long long key = keys[i];\n"
"  unsigned low = 0, high = rows;\n"
"  while (low < high) { unsigned m = low + (high - low) / 2; if (index[m] < key) low = m + 1; else high = m; }\n"
"  out[i] = low < rows && index[low] == key ? low : 0xffffffffu;\n"
"}\n"
"extern \"C\" __global__ void gather_f32(const float *d, long long st, const unsigned *r, unsigned n,\n"
"    const float *q, int dims, float *o) { gather(d, st, r, n, q, dims, o); }\n"
"extern \"C\" __global__ void gather_f16(const unsigned short *d, long long st, const unsigned *r,\n"
"    unsigned n, const float *q, int dims, float *o) { gather(d, st, r, n, q, dims, o); }\n";

static const char RerankSource[] =
"template <typename T> __device__ void rerank(const float *data, long long stride, const unsigned *rows,\n"
"    unsigned eligible, unsigned nr, T *positions, unsigned top, unsigned nq, const float *queries, int dims, float *out)\n"
"{\n"
"  unsigned i = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;\n"
"  int lane = threadIdx.x & 31; if (i >= top * nq) return;\n"
"  long long p = (long long) positions[i];\n"
"  unsigned row = p < 0 || (unsigned long long)p >= eligible ? 0xffffffffu : rows ? rows[p] : (unsigned)p;\n"
"  if (row >= nr) { if (!lane) { positions[i] = (T)-1; out[i] = __int_as_float(0x7f800000); } return; }\n"
"  const float *v = data + (long long) row * stride, *q = queries + (i / top) * dims;\n"
"  double sum = 0.; for (int d = lane; d < dims; d += 32) { double x = (double)v[d] - q[d]; sum += x * x; }\n"
"  for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);\n"
"  if (!lane) { positions[i] = (T)row; out[i] = (float)sqrt(sum); }\n"
"}\n"
"extern \"C\" __global__ void rerank_i64(const float *v, long long st, const unsigned *r, unsigned ne, unsigned nr,\n"
"  long long *p, unsigned top, unsigned nq, const float *q, int d, float *o) { rerank(v, st, r, ne, nr, p, top, nq, q, d, o); }\n"
"extern \"C\" __global__ void rerank_u32(const float *v, long long st, const unsigned *r, unsigned ne, unsigned nr,\n"
"  unsigned *p, unsigned top, unsigned nq, const float *q, int d, float *o) { rerank(v, st, r, ne, nr, p, top, nq, q, d, o); }\n";

#define ANN_GATHER_BLOCK 256
#define ANN_GATHER_SLACK 32		/* extra FP16 candidates re-scored in FP32 */
#define ANN_EXACT_FALLBACK 20000	/* CPU exact limit when gather cannot run */

typedef struct GatherModule
{
	atomic_int	state;			/* 0 untried/compiling, 1 ready, -1
								 * unavailable */
	CUcontext	context;
	CUmodule	module;
	CUfunction	lookup;
	CUfunction	f32;
	CUfunction	f16;
	CUfunction	rerank64;
	CUfunction	rerank32;
} GatherModule;

static GatherModule gatherModules[ANN_CUVS_MAX_DEVICES];
static pthread_mutex_t gatherMutex = PTHREAD_MUTEX_INITIALIZER;

/* Compile and load the gather kernels on a device once; NULL if unavailable. */
static GatherModule *
GatherKernels(int device)
{
	GatherModule *module = &gatherModules[device];
	char		arch[32];
	char	   *image = NULL;
	char	   *source = NULL;
	size_t		size = 0;
	int			major = 0,
				minor = 0;
	CUdevice	handle;

	if (atomic_load_explicit(&module->state, memory_order_acquire) != 0)
		return atomic_load_explicit(&module->state, memory_order_acquire) > 0 ? module : NULL;
	(void) pthread_mutex_lock(&gatherMutex);
	if (atomic_load_explicit(&module->state, memory_order_relaxed) != 0)
		goto unlock;
	if (cuInit(0) != CUDA_SUCCESS || cuDeviceGet(&handle, device) != CUDA_SUCCESS ||
		cuDevicePrimaryCtxRetain(&module->context, handle) != CUDA_SUCCESS ||
		cuCtxSetCurrent(module->context) != CUDA_SUCCESS ||
		cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess ||
		cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess)
		goto fail;
	snprintf(arch, sizeof(arch), "sm_%d%d", major, minor);
	source = malloc(strlen(GatherSource) + strlen(RerankSource) + 1);
	if (source == NULL)
		goto fail;
	strcpy(source, GatherSource);
	strcat(source, RerankSource);
	if ((image = AnnNvrtcCompile(source, "cuVS gather kernel", arch, &size)) == NULL ||
		cuModuleLoadData(&module->module, image) != CUDA_SUCCESS ||
		cuModuleGetFunction(&module->lookup, module->module, "lookup") != CUDA_SUCCESS ||
		cuModuleGetFunction(&module->f32, module->module, "gather_f32") != CUDA_SUCCESS ||
		cuModuleGetFunction(&module->f16, module->module, "gather_f16") != CUDA_SUCCESS ||
		cuModuleGetFunction(&module->rerank64, module->module, "rerank_i64") != CUDA_SUCCESS ||
		cuModuleGetFunction(&module->rerank32, module->module, "rerank_u32") != CUDA_SUCCESS)
		goto fail;
	atomic_store_explicit(&module->state, 1, memory_order_release);
	goto unlock;
fail:
	atomic_store_explicit(&module->state, -1, memory_order_release);
	fprintf(stderr, "cuVS: gather scan unavailable on device %d; filtered queries use CAGRA\n", device);
unlock:
	free(source);
	free(image);
	(void) pthread_mutex_unlock(&gatherMutex);
	return atomic_load_explicit(&module->state, memory_order_acquire) > 0 ? module : NULL;
}

/* Size this lane's gather buffers on its current device. */
static int
GatherBuffers(AnnCuvsDevice *lane, size_t rows, size_t top)
{
	cudaStream_t stream;

	if (cuvsStreamGet(lane->resources, &stream) != CUVS_SUCCESS)
		return 0;
	if (rows > lane->gatherCount)
	{
		size_t		count = rows * 2 > ANN_LANE_ROW_FLOOR ? rows * 2 : ANN_LANE_ROW_FLOOR;
		size_t		have;

		have = lane->gatherCount;
		if (!GrowDevice((void **) &lane->gatherKeys, &have, count, sizeof(int64_t), 0, stream))
			return 0;
		have = lane->gatherCount;
		if (!GrowDevice((void **) &lane->gatherRows, &have, count, sizeof(uint32_t), 0, stream))
			return 0;
		have = lane->gatherCount;
		if (!GrowDevice((void **) &lane->gatherScores, &have, count, sizeof(float), 0, stream))
			return 0;
		have = lane->gatherCount;
		if (!GrowPinned((void **) &lane->hostKeys, &have, count, sizeof(int64_t), 0))
			return 0;
		lane->gatherCount = have;
	}
	if (top > lane->gatherTop)
	{
		size_t		count = top * 2 > ANN_LANE_RESULT_FLOOR ? top * 2 : ANN_LANE_RESULT_FLOOR;
		size_t		have;

		have = lane->gatherTop;
		if (!GrowDevice((void **) &lane->gatherBest, &have, count, sizeof(float), 0, stream))
			return 0;
		have = lane->gatherTop;
		if (!GrowDevice((void **) &lane->gatherIndex, &have, count, sizeof(int64_t), 0, stream))
			return 0;
		have = lane->gatherTop;
		if (!GrowPinned((void **) &lane->hostIndex, &have, count, sizeof(int64_t), 0))
			return 0;
		lane->gatherTop = have;
	}
	return 1;
}

static int
RerankDevice(const AnnCuvsShard *shard, uint32_t dims, const uint32_t *rows,
			 uint32_t eligible, uint32_t nrows, void *positions, uint32_t top,
			 uint32_t nq, const float *queries, float *out, cudaStream_t stream, int wide)
{
	GatherModule *module = GatherKernels(shard->device);
	int			dimensions = (int) dims;
	long long	stride = shard->rerankStride;
	unsigned	blocks = (unsigned) (((uint64_t) nq * top * 32 + ANN_GATHER_BLOCK - 1) / ANN_GATHER_BLOCK);
	void	   *args[] = {(void *) &shard->rerank, &stride, (void *) &rows, &eligible, &nrows,
	&positions, &top, &nq, (void *) &queries, &dimensions, &out};

	return module && cuCtxSetCurrent(module->context) == CUDA_SUCCESS &&
		cuLaunchKernel(wide ? module->rerank64 : module->rerank32, blocks, 1, 1,
					   ANN_GATHER_BLOCK, 1, 1, 0, (CUstream) stream, args, NULL) == CUDA_SUCCESS;
}

static atomic_ullong gatherRepairs;

/*
 * Exact top k of one filtered query by gather scan. Keys are sorted and
 * unique; each shard's slice goes to its device, where a lookup kernel finds
 * each key's row in the shard's key array (misses score +inf), the gather
 * kernel scores the rows, and cuVS selects the best. Every shard launches
 * before any waits; the per-shard lists are re-scored in FP32 and merged.
 * Returns 0 when the gather scan cannot run, before any result is written.
 */
static int
SearchGather(const AnnIndexFile *file, AnnCuvsIndex *state, AnnIndexJob *job)
{
	uint32_t	dims = file->dimensions;
	uint32_t	k = job->k;

	/*
	 * Quantized and FP16 scores rank approximately: take extra candidates and
	 * let the FP32 re-score below decide.
	 */
	uint32_t	slack = state->quantized ? AnnIndexConfig.quant_slack : state->fp16 ? ANN_GATHER_SLACK : 0;

	/*
	 * On Blackwell the module's fused kernel scores and selects in one
	 * launch, returning its best ANN_BLACKWELL_TOP; all of them are
	 * re-scored, so small k gets that many candidates.
	 */
	int			fused = state->quantized && k + 8 <= ANN_BLACKWELL_TOP;
	int64_t		fusedFound[ANN_BLACKWELL_TOP];
	const int64_t *keys = job->filter;
	uint64_t	start[ANN_INDEX_MAX_SHARDS + 1];
	uint64_t	eligible[ANN_INDEX_MAX_SHARDS];
	uint32_t	wanted[ANN_INDEX_MAX_SHARDS];
	size_t		keySlot[ANN_INDEX_MAX_SHARDS];
	size_t		topSlot[ANN_INDEX_MAX_SHARDS];
	size_t		perDeviceKeys[ANN_CUVS_MAX_DEVICES] = {0};
	size_t		perDeviceTop[ANN_CUVS_MAX_DEVICES] = {0};
	int			uploaded[ANN_CUVS_MAX_DEVICES] = {0};
	uint32_t	counts[ANN_INDEX_MAX_SHARDS] = {0};
	AnnIndexHit *lists = NULL;
	AnnCuvsShard *shards;
	uint32_t	copy;
	int			ok = 1;

	if (fused)
		slack = ANN_BLACKWELL_TOP - k;
	copy = laneCopy;
	shards = state->shards + (size_t) copy * state->nshards;

	/* Split the keys at each shard's first key. */
	start[0] = 0;
	start[state->nshards] = job->nfilter;
	for (uint32_t s = 1; s < state->nshards; s++)
	{
		int64_t		boundary = file->keys[file->shards[s].first];
		uint64_t	low = start[s - 1],
					high = job->nfilter;

		while (low < high)
		{
			uint64_t	middle = low + (high - low) / 2;

			if (keys[middle] < boundary)
				low = middle + 1;
			else
				high = middle;
		}
		start[s] = low;
	}
	for (uint32_t s = 0; s < state->nshards; s++)
	{
		int			device = shards[s].device;

		eligible[s] = start[s + 1] - start[s];
		wanted[s] = (uint32_t) ((uint64_t) k + slack < eligible[s] ? (uint64_t) k + slack : eligible[s]);
		keySlot[s] = perDeviceKeys[device];
		topSlot[s] = perDeviceTop[device];
		perDeviceKeys[device] += eligible[s];
		perDeviceTop[device] += wanted[s];
		if (eligible[s] && (shards[s].rows == NULL || shards[s].keys == NULL || GatherKernels(device) == NULL))
			return 0;
	}
	lists = malloc(sizeof(AnnIndexHit) * (size_t) (k + slack) * state->nshards);
	if (lists == NULL)
		return 0;
	for (uint32_t s = 0; ok && s < state->nshards; s++)
	{
		AnnCuvsShard *shard = &shards[s];
		GatherModule *module = &gatherModules[shard->device];
		unsigned	count = (unsigned) eligible[s];
		unsigned	shardRows = file->shards[s].rows;
		long long	stride = shard->stride;
		int			dimensions = (int) dims;
		int64_t		scoreShape[2] = {1, (int64_t) eligible[s]};
		int64_t		topShape[2] = {1, wanted[s]};
		DLManagedTensor scores,
					best,
					index;
		AnnCuvsDevice *lane;
		cudaStream_t stream;
		int64_t    *wantedKeys;
		uint32_t   *rows;
		float	   *queries;
		float	   *out;
		void	   *lookup[5];
		void	   *gather[7];
		unsigned	lookupBlocks;
		unsigned	gatherBlocks;

		if (eligible[s] == 0)
			continue;
		lane = LaneDevice(shard->device, dims * sizeof(float), 1, 1);
		if (lane == NULL || cuvsStreamGet(lane->resources, &stream) != CUVS_SUCCESS ||
			!GatherBuffers(lane, perDeviceKeys[shard->device], perDeviceTop[shard->device]) ||
			cuCtxSetCurrent(module->context) != CUDA_SUCCESS)
		{
			ok = 0;
			break;
		}
		if (!uploaded[shard->device] &&
			!(uploaded[shard->device] = cudaMemcpyAsync(lane->queries, job->query, dims * sizeof(float),
														cudaMemcpyHostToDevice, stream) == cudaSuccess))
		{
			ok = 0;
			break;
		}
		wantedKeys = lane->gatherKeys + keySlot[s];
		rows = lane->gatherRows + keySlot[s];
		queries = lane->queries;
		out = lane->gatherScores + keySlot[s];
		lookup[0] = &shard->keys;
		lookup[1] = &shardRows;
		lookup[2] = &wantedKeys;
		lookup[3] = &count;
		lookup[4] = &rows;
		gather[0] = (void *) &shard->rows;
		gather[1] = &stride;
		gather[2] = &rows;
		gather[3] = &count;
		gather[4] = &queries;
		gather[5] = &dimensions;
		gather[6] = &out;
		lookupBlocks = (unsigned) ((eligible[s] + ANN_GATHER_BLOCK - 1) / ANN_GATHER_BLOCK);
		gatherBlocks = (unsigned) ((eligible[s] * 32 + ANN_GATHER_BLOCK - 1) / ANN_GATHER_BLOCK);
		scores = Tensor(out, kDLCUDA, shard->device, kDLFloat, 2, scoreShape);
		best = Tensor(lane->gatherBest + topSlot[s], kDLCUDA, shard->device, kDLFloat, 2, topShape);
		index = Tensor(lane->gatherIndex + topSlot[s], kDLCUDA, shard->device, kDLInt, 2, topShape);
		index.dl_tensor.dtype.bits = 64;
		/* A cached bitmap's rows are already on this device. */
		if (job->deviceRows && job->deviceRows[(size_t) copy * state->nshards + s])
		{
			rows = job->deviceRows[(size_t) copy * state->nshards + s];
			if (fused ? !AnnBlackwellSearch(&shard->quant, shard->device, s, rows, count, queries, 1, dimensions, stream) :
				((state->quantized ?
				  !AnnBlackwellScore(&shard->quant, shard->device, rows, count, queries, dimensions, out, stream) :
				  cuLaunchKernel(state->fp16 ? module->f16 : module->f32, gatherBlocks, 1, 1,
								 ANN_GATHER_BLOCK, 1, 1, 0, (CUstream) stream, gather, NULL) != CUDA_SUCCESS) ||
				 cuvsSelectK(lane->resources, &scores, &best, &index) != CUVS_SUCCESS ||
				 cudaMemcpyAsync(lane->hostIndex + topSlot[s], lane->gatherIndex + topSlot[s],
								 wanted[s] * sizeof(int64_t), cudaMemcpyDeviceToHost, stream) != cudaSuccess))
			{
				fprintf(stderr, "cuVS gather scan over cached rows failed on device %d\n", shard->device);
				ok = 0;
			}
			continue;
		}
		if (fused)
		{
			int			ready;

			/*
			 * Known ordinals (cached bitmaps, attributes) become shard rows
			 * here: 4 bytes a row, and no key lookup on the device.
			 */
			if (job->ordinals)
			{
				uint32_t   *staged = (uint32_t *) (void *) (lane->hostKeys + keySlot[s]);
				uint64_t	first = file->shards[s].first;

				for (uint64_t i = 0; i < eligible[s]; i++)
					staged[i] = (uint32_t) (job->ordinals[start[s] + i] - first);
				ready = cudaMemcpyAsync(rows, staged, eligible[s] * sizeof(uint32_t), cudaMemcpyHostToDevice,
										stream) == cudaSuccess;
			}
			else
			{
				memcpy(lane->hostKeys + keySlot[s], keys + start[s], eligible[s] * sizeof(int64_t));
				ready = cudaMemcpyAsync(wantedKeys, lane->hostKeys + keySlot[s], eligible[s] * sizeof(int64_t),
										cudaMemcpyHostToDevice, stream) == cudaSuccess &&
					cuLaunchKernel(module->lookup, lookupBlocks, 1, 1, ANN_GATHER_BLOCK, 1, 1, 0,
								   (CUstream) stream, lookup, NULL) == CUDA_SUCCESS;
			}
			if (!ready || !AnnBlackwellSearch(&shard->quant, shard->device, s, rows, count, queries, 1, dimensions, stream))
			{
				fprintf(stderr, "Blackwell module: fused search failed on device %d\n", shard->device);
				ok = 0;
			}
			continue;
		}
		memcpy(lane->hostKeys + keySlot[s], keys + start[s], eligible[s] * sizeof(int64_t));
		if (cudaMemcpyAsync(wantedKeys, lane->hostKeys + keySlot[s], eligible[s] * sizeof(int64_t),
							cudaMemcpyHostToDevice, stream) != cudaSuccess ||
			cuLaunchKernel(module->lookup, lookupBlocks, 1, 1, ANN_GATHER_BLOCK, 1, 1, 0,
						   (CUstream) stream, lookup, NULL) != CUDA_SUCCESS ||
			(state->quantized ?
			 !AnnBlackwellScore(&shard->quant, shard->device, rows, count, queries, dimensions, out, stream) :
			 cuLaunchKernel(state->fp16 ? module->f16 : module->f32, gatherBlocks, 1, 1,
							ANN_GATHER_BLOCK, 1, 1, 0, (CUstream) stream, gather, NULL) != CUDA_SUCCESS) ||
			cuvsSelectK(lane->resources, &scores, &best, &index) != CUVS_SUCCESS ||
			cudaMemcpyAsync(lane->hostIndex + topSlot[s], lane->gatherIndex + topSlot[s],
							wanted[s] * sizeof(int64_t), cudaMemcpyDeviceToHost, stream) != cudaSuccess)
		{
			const char *message = cuvsGetLastErrorText();

			fprintf(stderr, "cuVS gather scan: %s\n", message ? message : "failed");
			ok = 0;
		}
	}
	for (int d = 0; d < ANN_CUVS_MAX_DEVICES; d++)
		if (laneDevices[d].resources && perDeviceKeys[d] &&
			(cudaSetDevice(d) != cudaSuccess || cuvsStreamSync(laneDevices[d].resources) != CUVS_SUCCESS))
			ok = 0;

	/*
	 * Map the selected keys back to rows on the host: k binary searches. Keys
	 * missing from the index scored +inf and are dropped here.
	 */
	for (uint32_t s = 0; ok && s < state->nshards; s++)
	{
		const int64_t *found = laneDevices[shards[s].device].hostIndex + topSlot[s];
		AnnIndexHit *hits = lists + (size_t) s * (k + slack);
		uint64_t	first = file->shards[s].first;

		if (fused && eligible[s])
		{
			const uint32_t *positions;
			const float *unused;

			AnnBlackwellResults(shards[s].device, s, 0, &positions, &unused);
			for (uint32_t i = 0; i < ANN_BLACKWELL_TOP; i++)
				fusedFound[i] = positions[i] == UINT32_MAX ? -1 : (int64_t) positions[i];
			found = fusedFound;
		}

		/*
		 * Under heavy load, rare hybrid results have lost one true neighbour
		 * of a close pair, as if the selection returned a candidate twice.
		 * Distinct selected positions are cheap to check; a shard that fails
		 * is scored again exactly on the CPU.
		 */
		if (!AnnIndexPositionsDistinct(found, wanted[s]))
		{
			uint64_t   *ordinals = malloc(sizeof(uint64_t) * (eligible[s] ? eligible[s] : 1));
			AnnIndexJob part = *job;

			if (atomic_fetch_add(&gatherRepairs, 1) < 100)
				fprintf(stderr, "cuVS gather scan: repeated candidates in shard %u; scoring it exactly\n", s);
			if (ordinals == NULL)
			{
				ok = 0;
				break;
			}
			part.results = hits;
			part.k = k;
			AnnIndexExact(file, &part, ordinals,
						  AnnIndexOrdinals(file, keys + start[s], (uint32_t) eligible[s], ordinals));
			counts[s] = part.nresults;
			free(ordinals);
			continue;
		}
		for (uint32_t i = 0; i < wanted[s]; i++)
		{
			int64_t		key;
			uint64_t	low = first,
						high = first + file->shards[s].rows;
			uint32_t	distance = AnnIndexConfig.rerank_prefetch;

			if (distance && job->ordinals && wanted[s] - i > distance)
			{
				int64_t		future = found[i + distance];

				if (future >= 0 && (uint64_t) future < eligible[s])
					PrefetchExactVector(file, job->ordinals[start[s] + (uint64_t) future]);
			}

			if (found[i] < 0 || (uint64_t) found[i] >= eligible[s])
				continue;
			key = keys[start[s] + (uint64_t) found[i]];
			if (job->ordinals)
			{
				/* Cached filters already pair each key with its ordinal. */
				low = job->ordinals[start[s] + (uint64_t) found[i]];
				if (low < first || low >= high)
					continue;
			}
			else
				while (low < high)
				{
					uint64_t	middle = low + (high - low) / 2;

					if (file->keys[middle] < key)
						low = middle + 1;
					else
						high = middle;
				}
			if (low == first + file->shards[s].rows || file->keys[low] != key)
				continue;
			hits[counts[s]].key = key;
			hits[counts[s]].ordinal = low;
			hits[counts[s]].distance = ExactDistance(file, low, job->query);
			counts[s]++;
		}
		qsort(hits, counts[s], sizeof(*hits), HitCompare);
		if (counts[s] > k)
			counts[s] = k;
	}
	if (ok)
	{
		/* MergeJob expects lists k apart. */
		for (uint32_t s = 1; slack && s < state->nshards; s++)
			memmove(lists + (size_t) s * k, lists + (size_t) s * (k + slack), sizeof(AnnIndexHit) * counts[s]);
		MergeJob(job, lists, counts, state->nshards, k);
		job->route = ANN_ROUTE_EXACT;
	}
	else
		job->status = ANN_BACKEND_ERROR;
	free(lists);
	return 1;
}

/*
 * Filtered query: shards whose eligible slice is small are scanned exactly;
 * the rest run CAGRA with a bitset, with a larger internal top-k when the
 * filter is selective. All per-shard lists are tree-merged.
 */
/* Overfetch before FP32 reranking: rescoring cannot recover omitted neighbors. */
static uint32_t
FilteredCandidates(uint32_t k)
{
	uint32_t	slack = AnnIndexConfig.filtered_slack > laneProfileSlack ?
		AnnIndexConfig.filtered_slack : laneProfileSlack;

	/* cuVS' filtered result selector supports at most 1024 output columns. */
	if (k >= 1024)
		return k;
	return k + (slack < 1024 - k ? slack : 1024 - k);
}

static void
SearchFiltered(const AnnIndexFile *file, AnnCuvsIndex *state, AnnIndexJob *job)
{
	uint64_t   *ordinals = NULL;
	const uint64_t *known;
	uint64_t	n;
	uint32_t	k = job->k;
	uint32_t   *bitsets[ANN_INDEX_MAX_SHARDS] = {0};
	int			prebuilt = job->bitsets != NULL;
	int			useShard[ANN_INDEX_MAX_SHARDS] = {0};
	uint32_t	counts[ANN_INDEX_MAX_SHARDS] = {0};
	AnnIndexHit *lists = NULL;
	const AnnIndexJob *jobs[1] = {job};
	uint64_t	lo = 0;
	uint32_t	graphShards = 0;
	double		selectivity;
	uint32_t	itopk;
	uint64_t	exactLimit = AnnIndexConfig.exact_limit;

	/*
	 * A broad filter with prebuilt bitsets runs filtered CAGRA on every
	 * shard, with no per-row work here, whether or not its rows are known.
	 */
	if (job->bitsets != NULL &&
		((job->ordinals == NULL && job->filter == NULL) || job->nfilter > AnnIndexConfig.bitset_gather_limit))
	{
		uint32_t	graphCounts[ANN_INDEX_MAX_SHARDS] = {0};
		uint32_t	selected = FilteredCandidates(k);
		AnnIndexHit *graphLists = malloc(sizeof(AnnIndexHit) * (size_t) selected * state->nshards);

		for (uint32_t s = 0; s < state->nshards; s++)
		{
			useShard[s] = 1;
			bitsets[s] = job->bitsets[s];
		}
		itopk = RoundTopK(selected) > AnnIndexConfig.itopk ? RoundTopK(selected) : AnnIndexConfig.itopk;
		if ((double) job->nfilter / (double) file->rows < 0.25 && AnnIndexConfig.filtered_itopk > itopk)
			itopk = AnnIndexConfig.filtered_itopk;
		if (graphLists == NULL ||
			!SearchShards(file, state, jobs, 1, selected, itopk, bitsets, job->deviceBitsets, useShard, graphLists,
						  graphCounts))
			job->status = ANN_BACKEND_ERROR;
		else
		{
			MergeJob(job, graphLists, graphCounts, state->nshards, selected);
			job->route = ANN_ROUTE_FILTERED;
		}
		free(graphLists);
		return;
	}

	/*
	 * Mid-sized filters go to the GPU gather scan, which maps keys to rows on
	 * the device, so no per-key work happens here. With a ready bitset,
	 * filtered CAGRA overtakes the gather scan at a smaller filter than when
	 * the bitset must be built per query.
	 */
	if (job->nfilter > AnnIndexConfig.exact_limit &&
		job->nfilter <= (job->bitsets ? AnnIndexConfig.bitset_gather_limit : AnnIndexConfig.gather_limit))
	{
		if (SearchGather(file, state, job))
			return;

		/*
		 * Without the gather scan, CPU scans beat a mostly-masked graph walk
		 * up to a larger size.
		 */
		if (exactLimit < ANN_EXACT_FALLBACK)
			exactLimit = ANN_EXACT_FALLBACK;
	}
	/* Attribute filters arrive with their ordinals, and maybe bitsets. */
	if (job->ordinals == NULL)
		ordinals = malloc(sizeof(uint64_t) * (job->nfilter ? job->nfilter : 1));
	lists = malloc(sizeof(AnnIndexHit) * (size_t) k * state->nshards);
	job->status = ANN_NOMEM;
	if ((job->ordinals == NULL && ordinals == NULL) || lists == NULL)
		goto done;
	if (job->ordinals)
	{
		known = job->ordinals;
		n = job->nfilter;
	}
	else
	{
		n = AnnIndexOrdinals(file, job->filter, job->nfilter, ordinals);
		known = ordinals;
	}
	if (n <= exactLimit)
	{
		AnnIndexExact(file, job, known, n);
		goto done;
	}
	for (uint32_t s = 0; s < state->nshards; s++)
	{
		uint64_t	first = file->shards[s].first;
		uint64_t	end = first + file->shards[s].rows;
		uint64_t	start = lo;

		while (lo < n && known[lo] < end)
			lo++;
		if (lo - start <= exactLimit / state->nshards)
		{
			AnnIndexJob part = *job;

			part.results = lists + (size_t) s * k;
			AnnIndexExact(file, &part, known + start, lo - start);
			counts[s] = part.nresults;
			continue;
		}
		if (prebuilt)
			bitsets[s] = job->bitsets[s];
		else
		{
			bitsets[s] = calloc(((size_t) file->shards[s].rows + 31) / 32, sizeof(uint32_t));
			if (bitsets[s] == NULL)
				goto done;
			for (uint64_t i = start; i < lo; i++)
				bitsets[s][(known[i] - first) / 32] |= UINT32_C(1) << ((known[i] - first) % 32);
		}
		useShard[s] = 1;
		graphShards++;
	}
	if (graphShards)
	{
		uint32_t	graphCounts[ANN_INDEX_MAX_SHARDS] = {0};
		AnnIndexHit *graphLists = malloc(sizeof(AnnIndexHit) * (size_t) k * state->nshards);

		selectivity = (double) n / (double) file->rows;

		itopk = RoundTopK(k);
		if (itopk < AnnIndexConfig.itopk)
			itopk = AnnIndexConfig.itopk;
		if (selectivity < 0.25)
			itopk = AnnIndexConfig.filtered_itopk > itopk ? AnnIndexConfig.filtered_itopk : itopk;
		if (graphLists == NULL ||
			!SearchShards(file, state, jobs, 1, k, itopk, bitsets, prebuilt ? job->deviceBitsets : NULL, useShard,
						  graphLists, graphCounts))
		{
			free(graphLists);
			job->status = ANN_BACKEND_ERROR;
			goto done;
		}
		for (uint32_t s = 0; s < state->nshards; s++)
			if (useShard[s])
			{
				memcpy(lists + (size_t) s * k, graphLists + (size_t) s * k, sizeof(AnnIndexHit) * graphCounts[s]);
				counts[s] = graphCounts[s];
			}
		free(graphLists);
	}
	MergeJob(job, lists, counts, state->nshards, k);
	job->route = graphShards ? ANN_ROUTE_FILTERED : ANN_ROUTE_EXACT;
done:
	for (uint32_t s = 0; !prebuilt && s < state->nshards; s++)
		free(bitsets[s]);
	free(ordinals);
	free(lists);
}

/*
 * Unfiltered jobs of one index share each CAGRA submission: all queries go
 * to every shard at once with the batch's largest k.
 */
static void
SearchUnfiltered(const AnnIndexFile *file, AnnCuvsIndex *state, AnnIndexJob **jobs, uint32_t m)
{
	uint32_t	k = 0;
	int			useShard[ANN_INDEX_MAX_SHARDS];
	AnnIndexHit *lists;
	uint32_t   *counts;
	uint32_t	itopk;

	for (uint32_t j = 0; j < m; j++)
		if (jobs[j]->k > k)
			k = jobs[j]->k;
	itopk = RoundTopK(k) > AnnIndexConfig.itopk ? RoundTopK(k) : AnnIndexConfig.itopk;
	for (uint32_t s = 0; s < state->nshards; s++)
		useShard[s] = 1;
	lists = malloc(sizeof(AnnIndexHit) * (size_t) m * state->nshards * k);
	counts = calloc((size_t) m * state->nshards, sizeof(uint32_t));
	if (lists == NULL || counts == NULL ||
		!SearchShards(file, state, (const AnnIndexJob *const *) jobs, m, k, itopk, NULL, NULL, useShard, lists, counts))
	{
		for (uint32_t j = 0; j < m; j++)
			jobs[j]->status = ANN_BACKEND_ERROR;
	}
	else
		for (uint32_t j = 0; j < m; j++)
		{
			MergeJob(jobs[j], lists + (size_t) j * state->nshards * k, counts + (size_t) j * state->nshards,
					 state->nshards, k);
			jobs[j]->route = ANN_ROUTE_ANN;
		}
	free(lists);
	free(counts);
}

/*
 * Jobs over the same filter share its row arrays (the lane groups them, see
 * server.c); names are already equal here.
 */
static int
SameFilterJob(const AnnIndexJob *a, const AnnIndexJob *b)
{
	return b->filtered && (b->ordinals != NULL || b->bitsets != NULL) &&
		a->nfilter == b->nfilter && a->ordinals == b->ordinals && a->bitsets == b->bitsets;
}

/*
 * Filtered CAGRA for m jobs over one filter: one submission with the shared
 * bitsets, as SearchUnfiltered batches unfiltered jobs, instead of one
 * single-query search per job.
 */
static void
SearchFilteredGroupGraph(const AnnIndexFile *file, AnnCuvsIndex *state, AnnIndexJob **jobs, uint32_t m)
{
	uint32_t	k = 0;
	uint32_t	itopk;
	int			useShard[ANN_INDEX_MAX_SHARDS];
	uint32_t   *bitsets[ANN_INDEX_MAX_SHARDS];
	AnnIndexHit *lists;
	uint32_t   *counts;

	for (uint32_t j = 0; j < m; j++)
		if (jobs[j]->k > k)
			k = jobs[j]->k;
	k = FilteredCandidates(k);
	for (uint32_t s = 0; s < state->nshards; s++)
	{
		useShard[s] = 1;
		bitsets[s] = jobs[0]->bitsets[s];
	}
	itopk = RoundTopK(k) > AnnIndexConfig.itopk ? RoundTopK(k) : AnnIndexConfig.itopk;
	if ((double) jobs[0]->nfilter / (double) file->rows < 0.25 && AnnIndexConfig.filtered_itopk > itopk)
		itopk = AnnIndexConfig.filtered_itopk;
	lists = malloc(sizeof(AnnIndexHit) * (size_t) m * state->nshards * k);
	counts = calloc((size_t) m * state->nshards, sizeof(uint32_t));
	if (lists == NULL || counts == NULL ||
		!SearchShards(file, state, (const AnnIndexJob *const *) jobs, m, k, itopk, bitsets,
					  jobs[0]->deviceBitsets, useShard, lists, counts))
	{
		for (uint32_t j = 0; j < m; j++)
			jobs[j]->status = ANN_BACKEND_ERROR;
	}
	else
		for (uint32_t j = 0; j < m; j++)
		{
			MergeJob(jobs[j], lists + (size_t) j * state->nshards * k, counts + (size_t) j * state->nshards,
					 state->nshards, k);
			jobs[j]->route = ANN_ROUTE_FILTERED;
		}
	free(lists);
	free(counts);
}

/*
 * Gather scan of m jobs over one filter with the Blackwell module: each
 * shard reads its eligible rows once for all the queries and selects every
 * query's best ANN_BLACKWELL_TOP, which the host re-scores in FP32. Rows
 * come from the device cache when the filter has one, else are uploaded
 * once per shard. Groups larger than the module takes at once go in turns.
 * Returns 0, with no result written, when the group cannot take this path
 * or a launch failed; the caller then searches job by job.
 */
static int
SearchGatherGroup(const AnnIndexFile *file, AnnCuvsIndex *state, AnnIndexJob **jobs, uint32_t m)
{
	uint32_t	dims = file->dimensions;
	const uint64_t *ordinals = jobs[0]->ordinals;
	uint32_t	n = jobs[0]->nfilter;
	uint32_t	limit = (uint32_t) AnnBlackwellGroupLimit((int) dims);
	uint64_t	start[ANN_INDEX_MAX_SHARDS + 1];
	size_t		keySlot[ANN_INDEX_MAX_SHARDS];
	size_t		perDeviceKeys[ANN_CUVS_MAX_DEVICES] = {0};
	uint32_t	copy;
	AnnCuvsShard *shards;
	AnnIndexHit *lists;
	uint32_t   *counts;
	float	   *hostQueries;
	int			ok = 1;

	if (!state->grouped || ordinals == NULL || limit < 2)
		return 0;
	for (uint32_t j = 0; j < m; j++)
		if (jobs[j]->k + 8 > ANN_BLACKWELL_TOP)
			return 0;
	copy = laneCopy;
	shards = state->shards + (size_t) copy * state->nshards;

	/* Rows are sorted: split them at shard boundaries. */
	start[0] = 0;
	start[state->nshards] = n;
	for (uint32_t s = 1; s < state->nshards; s++)
	{
		uint64_t	low = start[s - 1],
					high = n;

		while (low < high)
		{
			uint64_t	middle = low + (high - low) / 2;

			if (ordinals[middle] < file->shards[s].first)
				low = middle + 1;
			else
				high = middle;
		}
		start[s] = low;
	}
	for (uint32_t s = 0; s < state->nshards; s++)
	{
		keySlot[s] = perDeviceKeys[shards[s].device];
		perDeviceKeys[shards[s].device] += start[s + 1] - start[s];
		if (start[s + 1] > start[s] && shards[s].quant.quant == ANN_QUANT_NONE)
			return 0;
	}
	lists = malloc(sizeof(AnnIndexHit) * (size_t) m * state->nshards * ANN_BLACKWELL_TOP);
	counts = calloc((size_t) m * state->nshards, sizeof(uint32_t));
	hostQueries = malloc(sizeof(float) * (size_t) limit * dims);
	if (lists == NULL || counts == NULL || hostQueries == NULL)
	{
		free(lists);
		free(counts);
		free(hostQueries);
		return 0;
	}
	for (uint32_t first = 0; ok && first < m; first += limit)
	{
		uint32_t	q = m - first < limit ? m - first : limit;
		int			uploaded[ANN_CUVS_MAX_DEVICES] = {0};

		for (uint32_t j = 0; j < q; j++)
			memcpy(hostQueries + (size_t) j * dims, jobs[first + j]->query, sizeof(float) * dims);
		for (uint32_t s = 0; ok && s < state->nshards; s++)
		{
			AnnCuvsShard *shard = &shards[s];
			uint32_t	eligible = (uint32_t) (start[s + 1] - start[s]);
			AnnCuvsDevice *lane;
			cudaStream_t stream;
			const uint32_t *rows = NULL;

			if (eligible == 0)
				continue;
			lane = LaneDevice(shard->device, (size_t) q * dims * sizeof(float), 1, 1);
			if (lane == NULL || cuvsStreamGet(lane->resources, &stream) != CUVS_SUCCESS ||
				!GatherBuffers(lane, perDeviceKeys[shard->device], 1))
			{
				ok = 0;
				break;
			}
			if (!uploaded[shard->device] &&
				!(uploaded[shard->device] = cudaMemcpyAsync(lane->queries, hostQueries, (size_t) q * dims * sizeof(float),
															cudaMemcpyHostToDevice, stream) == cudaSuccess))
			{
				ok = 0;
				break;
			}
			if (jobs[0]->deviceRows && jobs[0]->deviceRows[(size_t) copy * state->nshards + s])
				rows = jobs[0]->deviceRows[(size_t) copy * state->nshards + s];
			else
			{
				uint32_t   *staged = (uint32_t *) (void *) (lane->hostKeys + keySlot[s]);
				uint64_t	shardFirst = file->shards[s].first;

				for (uint32_t i = 0; i < eligible; i++)
					staged[i] = (uint32_t) (ordinals[start[s] + i] - shardFirst);
				rows = lane->gatherRows + keySlot[s];
				if (cudaMemcpyAsync(lane->gatherRows + keySlot[s], staged, eligible * sizeof(uint32_t),
									cudaMemcpyHostToDevice, stream) != cudaSuccess)
				{
					ok = 0;
					break;
				}
			}
			if (!AnnBlackwellSearch(&shard->quant, shard->device, s, rows, eligible, lane->queries, q, (int) dims, stream))
			{
				fprintf(stderr, "Blackwell module: grouped search of %u queries failed on device %d\n", q, shard->device);
				ok = 0;
			}
		}
		for (int d = 0; d < ANN_CUVS_MAX_DEVICES; d++)
			if (laneDevices[d].resources && uploaded[d] &&
				(cudaSetDevice(d) != cudaSuccess || cuvsStreamSync(laneDevices[d].resources) != CUVS_SUCCESS))
				ok = 0;

		/* Re-score each query's candidates in FP32; rows are known. */
		for (uint32_t s = 0; ok && s < state->nshards; s++)
		{
			uint32_t	eligible = (uint32_t) (start[s + 1] - start[s]);

			if (eligible == 0)
				continue;
			for (uint32_t j = 0; j < q; j++)
			{
				AnnIndexJob *job = jobs[first + j];
				AnnIndexHit *hits = lists + ((size_t) (first + j) * state->nshards + s) * ANN_BLACKWELL_TOP;
				uint32_t   *found = &counts[(size_t) (first + j) * state->nshards + s];
				const uint32_t *positions;
				const float *unused;
				int64_t		distinct[ANN_BLACKWELL_TOP];

				AnnBlackwellResults(shards[s].device, s, j, &positions, &unused);
				for (uint32_t i = 0; i < ANN_BLACKWELL_TOP; i++)
					distinct[i] = positions[i] == UINT32_MAX ? -1 : (int64_t) positions[i];
				if (!AnnIndexPositionsDistinct(distinct, ANN_BLACKWELL_TOP))
				{
					AnnIndexJob part = *job;

					if (atomic_fetch_add(&gatherRepairs, 1) < 100)
						fprintf(stderr, "Blackwell module: repeated candidates in shard %u; scoring it exactly\n", s);
					part.results = hits;
					AnnIndexExact(file, &part, ordinals + start[s], eligible);
					*found = part.nresults;
					continue;
				}
				for (uint32_t i = 0; i < ANN_BLACKWELL_TOP; i++)
				{
					uint64_t	row;

					if (positions[i] == UINT32_MAX || positions[i] >= eligible)
						continue;
					row = ordinals[start[s] + positions[i]];
					hits[*found].key = file->keys[row];
					hits[*found].ordinal = row;
					hits[*found].distance = ExactDistance(file, row, job->query);
					(*found)++;
				}
				qsort(hits, *found, sizeof(*hits), HitCompare);
				if (*found > job->k)
					*found = job->k;
			}
		}
	}
	if (ok)
		for (uint32_t j = 0; j < m; j++)
		{
			MergeJob(jobs[j], lists + (size_t) j * state->nshards * ANN_BLACKWELL_TOP,
					 counts + (size_t) j * state->nshards, state->nshards, ANN_BLACKWELL_TOP);
			jobs[j]->route = ANN_ROUTE_EXACT;
		}
	free(lists);
	free(counts);
	free(hostQueries);

	/*
	 * A failed launch leaves no result written: the caller searches job by
	 * job.
	 */
	return ok;
}

static int
MatrixScores(AnnCuvsDevice *lane, size_t needed, cudaStream_t stream)
{
	size_t		maximum = (size_t) (AnnIndexConfig.matrix_max_bytes / sizeof(float));
	size_t		capacity;

	if (needed > maximum)
		return 0;
	if (needed <= lane->matrixScoreCap)
		return 1;
	capacity = lane->matrixScoreCap * 2;
	if (capacity < needed)
		capacity = needed;
	if (capacity > maximum)
		capacity = maximum;
	if (lane->matrixScores)
		(void) cudaFreeAsync(lane->matrixScores, stream);
	lane->matrixScores = NULL;
	lane->matrixScoreCap = 0;
	if (cudaMallocAsync((void **) &lane->matrixScores, capacity * sizeof(float), stream) != cudaSuccess)
		return 0;
	lane->matrixScoreCap = capacity;
	return 1;
}

/* Independently sized buffers also recover cleanly from a partial allocation failure. */
static int
MatrixBuffers(AnnCuvsDevice *lane, size_t scores, size_t candidates, size_t output, size_t host,
			  cudaStream_t stream)
{
	return MatrixScores(lane, scores, stream) &&
		GrowDevice((void **) &lane->matrixCandidates, &lane->matrixCandidateCap, candidates, sizeof(float), 0, stream) &&
		GrowDevice((void **) &lane->matrixIndices, &lane->matrixIndexCap, candidates, sizeof(int64_t), 0, stream) &&
		GrowDevice((void **) &lane->matrixValues, &lane->matrixValueCap, output, sizeof(float), 0, stream) &&
		GrowDevice((void **) &lane->matrixRows, &lane->matrixRowCap, output, sizeof(int64_t), 0, stream) &&
		GrowPinned((void **) &lane->matrixHostValues, &lane->matrixHostValueCap, host, sizeof(float), 0) &&
		GrowPinned((void **) &lane->matrixHostRows, &lane->matrixHostRowCap, host, sizeof(int64_t), 0);
}

/*
 * General top-k: score a matrix once, select k+slack for every query, then
 * rerank in FP32 on device when resident, otherwise with the host vectors.
 * A shared filter is uploaded once per shard; GPU buffers are reused on its
 * stream and each shard copies results to its own pinned output segment.
 */
static int
SearchGatherMatrix(const AnnIndexFile *file, AnnCuvsIndex *state, AnnIndexJob **jobs, uint32_t m)
{
	uint32_t	dims = file->dimensions;
	uint32_t	limit = (uint32_t) AnnBlackwellGroupLimit((int) dims);
	uint32_t	maxk = 0;
	uint32_t	slack = state->quantized ? AnnIndexConfig.quant_slack : ANN_GATHER_SLACK;
	uint32_t	capacity;
	uint64_t	start[ANN_INDEX_MAX_SHARDS + 1];
	uint32_t	selected[ANN_INDEX_MAX_SHARDS];
	int			gpu[ANN_INDEX_MAX_SHARDS];
	const uint64_t *ordinals = jobs[0]->ordinals;
	AnnCuvsShard *shards = state->shards + (size_t) laneCopy * state->nshards;
	AnnIndexHit *lists;
	AnnIndexHit *scratch;
	uint32_t   *counts;
	float	   *hostQueries;
	uint64_t	reranked = 0;
	int			ok = 1;

	if (!state->grouped || ordinals == NULL || limit < 2 || m == 0)
		return 0;
	for (uint32_t j = 0; j < m; j++)
		if (jobs[j]->k > maxk)
			maxk = jobs[j]->k;
	if (maxk + 8 <= ANN_BLACKWELL_TOP)
		return 0;
	capacity = (uint32_t) ((uint64_t) maxk + slack < jobs[0]->nfilter ?
						   (uint64_t) maxk + slack : jobs[0]->nfilter);
	if (capacity == 0)
		return 0;
	start[0] = 0;
	start[state->nshards] = jobs[0]->nfilter;
	for (uint32_t s = 1; s < state->nshards; s++)
	{
		uint64_t	low = start[s - 1],
					high = jobs[0]->nfilter;

		while (low < high)
		{
			uint64_t	mid = low + (high - low) / 2;

			if (ordinals[mid] < file->shards[s].first)
				low = mid + 1;
			else
				high = mid;
		}
		start[s] = low;
	}
	for (uint32_t s = 0; s < state->nshards; s++)
	{
		uint64_t	n = start[s + 1] - start[s];
		uint64_t	fit;

		if (!n)
			continue;
		if (shards[s].quant.quant == ANN_QUANT_NONE)
			return 0;
		fit = AnnIndexConfig.matrix_max_bytes / (n * sizeof(float));
		if (!fit)
			return 0;
		if (fit < limit)
			limit = (uint32_t) fit;
	}
	lists = malloc(sizeof(AnnIndexHit) * (size_t) m * state->nshards * maxk);
	scratch = malloc(sizeof(AnnIndexHit) * capacity);
	counts = calloc((size_t) m * state->nshards, sizeof(uint32_t));
	hostQueries = malloc(sizeof(float) * (size_t) limit * dims);
	if (lists == NULL || scratch == NULL || counts == NULL || hostQueries == NULL)
	{
		free(lists);
		free(scratch);
		free(counts);
		free(hostQueries);
		return 0;
	}
	for (uint32_t first = 0; ok && first < m; first += limit)
	{
		uint32_t	q = m - first < limit ? m - first : limit;
		int			uploaded[ANN_CUVS_MAX_DEVICES] = {0};
		int			used[ANN_CUVS_MAX_DEVICES] = {0};

		for (uint32_t j = 0; j < q; j++)
			memcpy(hostQueries + (size_t) j * dims, jobs[first + j]->query, sizeof(float) * dims);
		for (uint32_t s = 0; ok && s < state->nshards; s++)
		{
			AnnCuvsShard *shard = &shards[s];
			uint32_t	n = (uint32_t) (start[s + 1] - start[s]);
			uint32_t	candidates = capacity < n ? capacity : n;
			uint32_t	outk = maxk < candidates ? maxk : candidates;
			AnnCuvsDevice *lane;
			cudaStream_t stream;
			const uint32_t *rows;
			size_t		hostOffset = (size_t) s * limit * capacity;

			selected[s] = 0;
			gpu[s] = 0;
			if (n == 0)
				continue;
			lane = LaneDevice(shard->device, (size_t) q * dims * sizeof(float), 1, 1);
			if (lane == NULL || cuvsStreamGet(lane->resources, &stream) != CUVS_SUCCESS ||
				!GatherBuffers(lane, jobs[0]->nfilter, 1) ||
				!MatrixBuffers(lane, (size_t) q * n, (size_t) q * candidates,
							   (size_t) q * outk, (size_t) state->nshards * limit * capacity, stream))
			{
				ok = 0;
				break;
			}
			used[shard->device] = 1;
			if (!uploaded[shard->device])
			{
				if (cudaMemcpyAsync(lane->queries, hostQueries, (size_t) q * dims * sizeof(float),
									cudaMemcpyHostToDevice, stream) != cudaSuccess)
				{
					ok = 0;
					break;
				}
				uploaded[shard->device] = 1;
			}
			rows = jobs[0]->deviceRows ? jobs[0]->deviceRows[(size_t) laneCopy * state->nshards + s] : NULL;
			if (rows == NULL)
			{
				uint32_t   *staged = (uint32_t *) (void *) (lane->hostKeys + start[s]);

				for (uint32_t i = 0; i < n; i++)
					staged[i] = (uint32_t) (ordinals[start[s] + i] - file->shards[s].first);
				if (cudaMemcpyAsync(lane->gatherRows + start[s], staged, (size_t) n * sizeof(uint32_t),
									cudaMemcpyHostToDevice, stream) != cudaSuccess)
				{
					ok = 0;
					break;
				}
				rows = lane->gatherRows + start[s];
			}
			{
				uint_fast64_t peak = atomic_load_explicit(&state->matrix_peak_score_bytes, memory_order_relaxed);
				uint_fast64_t bytes = (uint64_t) q * n * sizeof(float);

				while (peak < bytes && !atomic_compare_exchange_weak_explicit(&state->matrix_peak_score_bytes,
																			  &peak, bytes, memory_order_relaxed, memory_order_relaxed))
					;
			}
			if (!AnnBlackwellScoreMatrix(&shard->quant, shard->device, rows, n, lane->queries, q,
										 (int) dims, lane->matrixScores, stream) ||
				AnnSelectBatch(lane->resources, lane->matrixScores, NULL, lane->matrixCandidates,
							   lane->matrixIndices, q, n, candidates, 0) != CUVS_SUCCESS)
			{
				ok = 0;
				break;
			}
			if (shard->rerank && laneRerank != 0)
			{
				if (!RerankDevice(shard, dims, rows, n, file->shards[s].rows, lane->matrixIndices,
								  candidates, q, lane->queries, lane->matrixCandidates, stream, 1) ||
					AnnSelectBatch(lane->resources, lane->matrixCandidates, lane->matrixIndices,
								   lane->matrixValues, lane->matrixRows, q, candidates, outk, 1) != CUVS_SUCCESS ||
					cudaMemcpyAsync(lane->matrixHostRows + hostOffset, lane->matrixRows,
									(size_t) q * outk * sizeof(int64_t), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
					cudaMemcpyAsync(lane->matrixHostValues + hostOffset, lane->matrixValues,
									(size_t) q * outk * sizeof(float), cudaMemcpyDeviceToHost, stream) != cudaSuccess)
				{
					ok = 0;
					break;
				}
				selected[s] = outk;
				gpu[s] = 1;
				reranked += (uint64_t) q * candidates;
			}
			else
			{
				if (cudaMemcpyAsync(lane->matrixHostRows + hostOffset, lane->matrixIndices,
									(size_t) q * candidates * sizeof(int64_t), cudaMemcpyDeviceToHost, stream) != cudaSuccess)
				{
					ok = 0;
					break;
				}
				selected[s] = candidates;
			}
		}

		/*
		 * Even failed submissions must finish before pageable query storage
		 * is reused/freed.
		 */
		for (int d = 0; d < ANN_CUVS_MAX_DEVICES; d++)
			if (used[d] && (cudaSetDevice(d) != cudaSuccess || cuvsStreamSync(laneDevices[d].resources) != CUVS_SUCCESS))
				ok = 0;
		for (uint32_t s = 0; ok && s < state->nshards; s++)
			for (uint32_t j = 0; selected[s] && j < q; j++)
			{
				AnnIndexJob *job = jobs[first + j];
				AnnIndexHit *hits = lists + ((size_t) (first + j) * state->nshards + s) * maxk;
				uint32_t   *found = &counts[(size_t) (first + j) * state->nshards + s];
				AnnCuvsDevice *lane = &laneDevices[shards[s].device];
				const int64_t *positions = lane->matrixHostRows + (size_t) s * limit * capacity + (size_t) j * selected[s];
				const float *distances = lane->matrixHostValues + (size_t) s * limit * capacity + (size_t) j * selected[s];
				uint32_t	n = (uint32_t) (start[s + 1] - start[s]);
				int			ordered = 1;

				if (!AnnIndexPositionsDistinct(positions, selected[s]))
				{
					AnnIndexJob part = *job;

					part.results = hits;
					AnnIndexExact(file, &part, ordinals + start[s], n);
					*found = part.nresults;
					continue;
				}
				for (uint32_t i = 0; i < selected[s]; i++)
				{
					int64_t		pos = positions[i];
					uint64_t	row;
					AnnIndexHit hit;

					if (pos < 0 || (uint64_t) pos >= (gpu[s] ? file->shards[s].rows : n))
						continue;
					row = gpu[s] ? file->shards[s].first + (uint64_t) pos : ordinals[start[s] + (uint64_t) pos];
					hit.key = file->keys[row];
					hit.ordinal = row;
					hit.distance = gpu[s] ? distances[i] : ExactDistance(file, row, job->query);
					if (gpu[s] && isnan(hit.distance))
						hit.distance = ExactDistance(file, row, job->query);
					if (!gpu[s])
					{
						scratch[(*found)++] = hit;
						ordered = 0;
					}
					else
					{
						if (*found && HitCompare(hits + *found - 1, &hit) > 0)
							ordered = 0;
						hits[(*found)++] = hit;
					}
				}
				if (!gpu[s])
				{
					qsort(scratch, *found, sizeof(*scratch), HitCompare);
					if (*found > maxk)
						*found = maxk;
					memcpy(hits, scratch, sizeof(*hits) * *found);
				}
				else if (!ordered)
					qsort(hits, *found, sizeof(*hits), HitCompare);
				if (*found > job->k)
					*found = job->k;
			}
	}
	if (ok)
	{
		for (uint32_t j = 0; j < m; j++)
		{
			MergeJob(jobs[j], lists + (size_t) j * state->nshards * maxk,
					 counts + (size_t) j * state->nshards, state->nshards, maxk);
			jobs[j]->route = ANN_ROUTE_EXACT;
		}
		(void) atomic_fetch_add_explicit(&state->matrix_queries, m, memory_order_relaxed);
		(void) atomic_fetch_add_explicit(&state->gpu_rerank_candidates, reranked, memory_order_relaxed);
	}
	free(lists);
	free(scratch);
	free(counts);
	free(hostQueries);
	return ok;
}

/*
 * Filtered jobs over one filter. Selective filters run exactly, one by one.
 * On a quantized index, several jobs scan the filter's rows together on the
 * tensor cores up to --group-gather-limit (exact; 82k QPS of 10% filters on
 * one B200 against 53k for batched CAGRA); otherwise mid-sized filters take
 * the gather scan one by one and broad ones with prebuilt bitsets run one
 * batched filtered CAGRA search.
 */
/* Profile hints use actual eligibility and batch shape, with no queue/RPC lock. */
static int
CalibratedFilteredGroup(const AnnIndexFile *file, AnnCuvsIndex *state, AnnIndexJob **jobs, uint32_t m)
{
	double		us[ANN_COST_MODES];
	uint32_t	k = 0;
	int			gpu = AnnIndexConfig.gpu_rerank;
	int			scan = AnnIndexConfig.matrix_gather && state->grouped && jobs[0]->ordinals != NULL;
	int			graph = jobs[0]->bitsets != NULL;
	int			mode;

	if (AnnIndexCostProfile == NULL || file->quant != ANN_QUANT_NVFP4 || file->nshards != 1 || !state->fp16 ||
		AnnIndexConfig.filtered_algo != 2 || AnnIndexConfig.quant_slack != 256 || AnnIndexConfig.itopk != 512 ||
		AnnIndexConfig.filtered_itopk != 1024 || AnnIndexConfig.filtered_width != 4 ||
		AnnIndexConfig.filtered_iterations != 0 || AnnIndexConfig.persistent || AnnIndexConfig.filtered_slack != 0 ||
		AnnIndexConfig.matrix_max_bytes != (UINT64_C(64) << 20))
		return 0;
	for (uint32_t j = 0; j < m; j++)
		if (jobs[j]->k > k)
			k = jobs[j]->k;
	for (uint32_t s = 0; s < state->nshards; s++)
		gpu = gpu && state->shards[(size_t) laneCopy * state->nshards + s].rerank != NULL;
	if (!AnnCostEstimate(AnnIndexCostProfile, file->rows, file->dimensions, file->generationValue,
						 jobs[0]->nfilter, k, m, us) || (mode = AnnCostBest(us, scan, graph, gpu)) < 0)
		return 0;
	laneRerank = mode & 1;
	if (mode < ANN_COST_GRAPH_HOST)
	{
		int			ok = SearchGatherMatrix(file, state, jobs, m);

		laneRerank = -1;
		if (ok)
			(void) atomic_fetch_add_explicit(&state->cost_queries[mode], m, memory_order_relaxed);
		return ok;
	}
	laneProfileItopk = AnnIndexCostProfile->graph_itopk;
	laneProfileSlack = AnnIndexCostProfile->graph_slack;
	SearchFilteredGroupGraph(file, state, jobs, m);
	laneProfileItopk = 0;
	laneProfileSlack = 0;
	laneRerank = -1;
	(void) atomic_fetch_add_explicit(&state->cost_queries[mode], m, memory_order_relaxed);
	return 1;
}

static void
SearchFilteredGroup(const AnnIndexFile *file, AnnCuvsIndex *state, AnnIndexJob **jobs, uint32_t m)
{
	uint32_t	n = jobs[0]->nfilter;

	if (CalibratedFilteredGroup(file, state, jobs, m))
		return;

	if (n > AnnIndexConfig.exact_limit && n <= AnnIndexConfig.group_gather_limit)
	{
		if (m > 1 && SearchGatherGroup(file, state, jobs, m))
			return;
		if (AnnIndexConfig.matrix_gather && SearchGatherMatrix(file, state, jobs, m))
			return;
	}
	if (m > 1 && jobs[0]->bitsets != NULL && n > AnnIndexConfig.bitset_gather_limit)
	{
		SearchFilteredGroupGraph(file, state, jobs, m);
		return;
	}
	for (uint32_t j = 0; j < m; j++)
		SearchFiltered(file, state, jobs[j]);
}

static void
SearchIndex(AnnIndexJob *jobs, uint32_t count)
{
	AnnIndexJob **group = malloc(sizeof(*group) * (count ? count : 1));
	AnnIndexJob **filtered = malloc(sizeof(*filtered) * (count ? count : 1));
	AnnIndexJob **members = malloc(sizeof(*members) * (count ? count : 1));
	char	   *done = calloc(count ? count : 1, 1);

	for (uint32_t i = 0; i < count; i++)
	{
		AnnIndexFile *file;
		AnnCuvsIndex *state;
		uint32_t	m = 0;
		uint32_t	f = 0;

		if (done[i])
			continue;
		file = AnnIndexGet(jobs[i].name);
		state = file ? file->device : NULL;
		for (uint32_t j = i; j < count; j++)
		{
			if (done[j] || strcmp(jobs[j].name, jobs[i].name) != 0)
				continue;
			done[j] = 1;
			jobs[j].nresults = 0;
			if (file == NULL)
				jobs[j].status = ANN_STALE;
			else if (jobs[j].dimensions != file->dimensions)
				jobs[j].status = ANN_INVALID;
			else if (state == NULL)
			{
				/* Files without CAGRA shards are served exactly. */
				uint64_t   *ordinals = NULL;
				uint64_t	n = 0;

				if (jobs[j].filtered && jobs[j].ordinals)
					AnnIndexExact(file, &jobs[j], jobs[j].ordinals, jobs[j].nfilter);
				else
				{
					if (jobs[j].filtered && (ordinals = malloc(sizeof(uint64_t) * (jobs[j].nfilter + 1))))
						n = AnnIndexOrdinals(file, jobs[j].filter, jobs[j].nfilter, ordinals);
					if (jobs[j].filtered && ordinals == NULL)
						jobs[j].status = ANN_NOMEM;
					else
						AnnIndexExact(file, &jobs[j], ordinals, n);
				}
				free(ordinals);
			}
			else if (jobs[j].filtered)
				filtered[f++] = &jobs[j];
			else
				group[m++] = &jobs[j];
		}

		/* Same-filter jobs, as the lane queued them, share one search. */
		for (uint32_t a = 0; a < f; a++)
		{
			uint32_t	g = 1;

			if (filtered[a] == NULL)
				continue;
			members[0] = filtered[a];
			for (uint32_t b = a + 1; b < f; b++)
				if (filtered[b] && SameFilterJob(members[0], filtered[b]))
				{
					members[g++] = filtered[b];
					filtered[b] = NULL;
				}
			{
				uint64_t	started = ReplicaNow();
				uint32_t	copy = ReplicaAcquire(state, g, started);

				SearchFilteredGroup(file, state, members, g);
				ReplicaRelease(state, copy, members, g, started);
			}
		}
		if (m && group && file && state)
		{
			uint64_t	started = ReplicaNow();
			uint32_t	copy = ReplicaAcquire(state, m, started);

			SearchUnfiltered(file, state, group, m);
			ReplicaRelease(state, copy, group, m, started);
		}
		if (file)
			AnnIndexPut(file);
	}
	free(group);
	free(filtered);
	free(members);
	free(done);
}

static uint64_t
DeviceJobs(void)
{
	return atomic_load(&deviceJobs);
}

/*
 * AnnBackend.lost: a sticky error on any device. These errors poison the
 * context: every later call on the device returns them.
 */
static const char *
Lost(void)
{
	static char why[160];
	int			devices = 0;

	if (cudaGetDeviceCount(&devices) != cudaSuccess)
		return "no CUDA devices";
	for (int d = 0; d < devices && d < ANN_INDEX_MAX_SHARDS; d++)
	{
		cudaError_t error;

		if (cudaSetDevice(d) != cudaSuccess)
			error = cudaGetLastError();
		else
			error = cudaDeviceSynchronize();
		switch (error)
		{
			case cudaErrorIllegalAddress:
			case cudaErrorLaunchFailure:
			case cudaErrorHardwareStackError:
			case cudaErrorIllegalInstruction:
			case cudaErrorMisalignedAddress:
			case cudaErrorInvalidAddressSpace:
			case cudaErrorInvalidPc:
			case cudaErrorAssert:
			case cudaErrorECCUncorrectable:
			case cudaErrorDevicesUnavailable:
				snprintf(why, sizeof(why), "device %d: %s", d, cudaGetErrorString(error));
				return why;
			default:
				break;
		}
	}
	return NULL;
}

const AnnBackend *
AnnCuvsBackend(void)
{
	static const AnnBackend backend = {
		"cuvs", 1, 7, "cagra", "fp32", 0, 128, 1024, Build, Search, Destroy,
		SearchGathered, DeviceJobs, LoadIndex, SearchIndex, Lost
	};

	AnnIndexRoutingStats = ReplicaStats;
	return &backend;
}
