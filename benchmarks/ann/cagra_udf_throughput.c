/*
 * Saturated throughput of filtered FP32 CAGRA on one GPU: T threads, each
 * with its own cuVS resources (stream) and buffers, search the shared index
 * back to back for a fixed time. Run one process per GPU (each loads its own
 * replica) and add the totals. Recall is measured by cagra_udf_probe; this
 * uses the same search settings and the estimated filtering_rate it validated.
 *
 *   cagra_udf_throughput INDEX_DIR CASE MODE [threads] [seconds] [batch] [k] [itopk] [slack]
 *
 * CASE: cat50 | cat10 | cat1 | mixed. MODE: bitset (one cached device bitset
 * per batch; for mixed, one cached bitset and one search per query) | udf |
 * udfgroup (mixed: one UDF search per group of queries with the same window
 * size, each with its own filtering_rate, instead of one at the batch's
 * highest rate).
 * Prints one JSON line.
 */
#include <cuda_runtime.h>
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/cagra.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "cagra_udf.h"
#include "filter.h"
#include "udf.h"

#define CHECK(x) do { if ((x) != CUVS_SUCCESS) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cuvsGetLastErrorText()); exit(1); } } while (0)
#define CUDA(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1); } } while (0)

#define MAX_BATCH 64

static uint64_t rows;
static uint32_t dims = 128;
static const float *vectors;
static const int32_t *columns[2];
static int32_t *dColumns[2];
static cuvsCagraIndex_t sharedIndex;
static uint32_t nwords;
static uint32_t batch = 32,
			k = 500,
			itopk = 4096,
			slack = 256,
			sel;
static int	mixed,
			useUdf,
			grouped;
static uint32_t groupStart[11];	/* udfgroup: query slots of each window size */
static float groupRate[10];
static double seconds = 20;
static atomic_uint_fast64_t totalQueries;
static atomic_int ready;
static volatile int go,
			stop;

typedef struct
{
	unsigned char b[256];
	size_t		n;
} Program;

typedef struct
{
	AnnUdfStep	steps[256];
	int32_t		values[256];
	uint32_t	starts[MAX_BATCH + 1];
	uint32_t	nsteps,
				nvalues;
} Batch;

static Batch programs;
static float rate;
static uint32_t *dBitsets[MAX_BATCH];	/* per query (mixed) or [0] */

static const void *
Map(const char *dir, const char *name, size_t *size)
{
	char		path[1024];
	struct stat st;
	int			fd;
	void	   *p;

	snprintf(path, sizeof(path), "%s/%s", dir, name);
	if ((fd = open(path, O_RDONLY)) < 0 || fstat(fd, &st) != 0)
	{
		perror(path);
		exit(1);
	}
	p = mmap(NULL, st.st_size, PROT_READ, MAP_SHARED | MAP_POPULATE, fd, 0);
	close(fd);
	if (p == MAP_FAILED)
		exit(1);
	*size = st.st_size;
	return p;
}

static double
Now(void)
{
	struct timespec t;

	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec + t.tv_nsec / 1e9;
}

static void
Put(Program *p, uint64_t v, int bytes)
{
	for (int i = 0; i < bytes; i++)
		p->b[p->n++] = (unsigned char) (v >> (8 * i));
}

static void
CategoryRange(Program *p, int32_t lo, int32_t hi)
{
	Put(p, ANN_FILTER_RANGE, 1);
	Put(p, 8, 1);
	memcpy(p->b + p->n, "category", 8);
	p->n += 8;
	Put(p, 0, 1);
	Put(p, (uint32_t) lo, 4);
	Put(p, (uint32_t) hi, 4);
}

static int
Column(void *arg, const char *name)
{
	(void) arg;
	return strcmp(name, "category") == 0 ? 0 : strcmp(name, "tenant") == 0 ? 1 : -1;
}

/* Rows per category value, for AnnUdfEstimate. */
static uint64_t categoryCount[100];

static uint64_t
CountRows(void *arg, int column, int32_t low, int32_t high)
{
	uint64_t	n = 0;

	(void) arg;
	if (column != 0)
		return 0;
	for (int32_t v = low < 0 ? 0 : low; v <= high && v < 100; v++)
		n += categoryCount[v];
	return n;
}

static void
BuildBitset(uint32_t q, uint32_t *dBitset)
{
	uint32_t   *words = calloc(nwords, 4);

#pragma omp parallel for schedule(static)
	for (uint64_t w = 0; w < nwords; w++)
	{
		uint32_t	bits = 0;

		for (uint32_t i = 0; i < 32 && w * 32 + i < rows; i++)
			if (AnnUdfMatch(programs.steps + programs.starts[q], programs.starts[q + 1] - programs.starts[q],
							programs.values, columns, w * 32 + i))
				bits |= 1u << i;
		words[w] = bits;
	}
	CUDA(cudaMemcpy(dBitset, words, sizeof(uint32_t) * nwords, cudaMemcpyHostToDevice));
	free(words);
}

static void
BitsetSearch(cuvsResources_t res, cuvsCagraSearchParams_t params, float *dQueries, uint32_t *dNeighbors,
			 float *dDistances, uint32_t *dBitset, uint32_t q0, uint32_t m)
{
	int64_t		qs[2] = {m, dims},
				rs[2] = {m, sel},
				bs[1] = {nwords};
	DLManagedTensor tq,
				tn,
				td,
				tb;
	cuvsFilter	f;
	DLDevice	device = {kDLCUDA, 0};
	DLDataType	f32 = {kDLFloat, 32, 1};
	DLDataType	u32 = {kDLUInt, 32, 1};

	memset(&tq, 0, sizeof(tq));
	tq.dl_tensor.data = dQueries + (size_t) q0 * dims;
	tq.dl_tensor.device = device;
	tq.dl_tensor.ndim = 2;
	tq.dl_tensor.dtype = f32;
	tq.dl_tensor.shape = qs;
	tn = tq;
	tn.dl_tensor.data = dNeighbors + (size_t) q0 * sel;
	tn.dl_tensor.dtype = u32;
	tn.dl_tensor.shape = rs;
	td = tq;
	td.dl_tensor.data = dDistances + (size_t) q0 * sel;
	td.dl_tensor.shape = rs;
	tb = tq;
	tb.dl_tensor.data = dBitset;
	tb.dl_tensor.ndim = 1;
	tb.dl_tensor.dtype = u32;
	tb.dl_tensor.shape = bs;
	f.type = BITSET;
	f.addr = (uintptr_t) &tb;
	CHECK(cuvsCagraSearch(res, params, sharedIndex, &tq, &tn, &td, f));
}

static void *
Worker(void *arg)
{
	uint32_t	id = (uint32_t) (uintptr_t) arg;
	cuvsResources_t res;
	cuvsCagraSearchParams_t params;
	float	   *dQueries,
			   *dDistances;
	uint32_t   *dNeighbors;
	AnnUdfParams *dParams;
	AnnUdfParams *dGroupParams;
	AnnUdfStep *dSteps;
	int32_t    *dValues;
	uint32_t   *dStarts;
	float	   *queries = malloc(sizeof(float) * MAX_BATCH * dims);
	uint64_t	done = 0;

	CUDA(cudaSetDevice(0));
	CHECK(cuvsResourcesCreate(&res));
	CHECK(cuvsCagraSearchParamsCreate(&params));
	params->itopk_size = itopk;
	params->algo = MULTI_CTA;
	params->search_width = 4;
	params->persistent = false;
	params->max_queries = mixed && !useUdf ? 1 : batch;
	CUDA(cudaMalloc((void **) &dQueries, sizeof(float) * MAX_BATCH * dims));
	CUDA(cudaMalloc((void **) &dNeighbors, sizeof(uint32_t) * MAX_BATCH * 1024));
	CUDA(cudaMalloc((void **) &dDistances, sizeof(float) * MAX_BATCH * 1024));
	CUDA(cudaMalloc((void **) &dParams, sizeof(AnnUdfParams)));
	CUDA(cudaMalloc((void **) &dGroupParams, sizeof(AnnUdfParams) * 10));
	CUDA(cudaMalloc((void **) &dSteps, sizeof(programs.steps)));
	CUDA(cudaMalloc((void **) &dValues, sizeof(programs.values)));
	CUDA(cudaMalloc((void **) &dStarts, sizeof(programs.starts)));
	CUDA(cudaMemcpy(dSteps, programs.steps, sizeof(programs.steps), cudaMemcpyHostToDevice));
	CUDA(cudaMemcpy(dValues, programs.values, sizeof(programs.values), cudaMemcpyHostToDevice));
	CUDA(cudaMemcpy(dStarts, programs.starts, sizeof(programs.starts), cudaMemcpyHostToDevice));
	{
		AnnUdfParams p;

		memset(&p, 0, sizeof(p));
		p.columns[0] = dColumns[0];
		p.columns[1] = dColumns[1];
		p.steps = dSteps;
		p.values = dValues;
		p.starts = dStarts;
		CUDA(cudaMemcpy(dParams, &p, sizeof(p), cudaMemcpyHostToDevice));
		/* A group's query ids start at 0: its starts begin at its first slot. */
		for (int g = 0; g < 10; g++)
		{
			p.starts = dStarts + groupStart[g];
			CUDA(cudaMemcpy(dGroupParams + g, &p, sizeof(p), cudaMemcpyHostToDevice));
		}
	}
	/* Each thread its own query set: dataset rows plus noise. */
	{
		unsigned	seed = 1000 + id;

		for (uint32_t q = 0; q < MAX_BATCH; q++)
		{
			uint64_t	r = ((uint64_t) rand_r(&seed) * 7919u + q * 104729u) % rows;

			for (uint32_t d = 0; d < dims; d++)
				queries[q * dims + d] = vectors[r * dims + d] + ((float) rand_r(&seed) / RAND_MAX - 0.5f) * 0.05f;
		}
	}
	CUDA(cudaMemcpy(dQueries, queries, sizeof(float) * MAX_BATCH * dims, cudaMemcpyHostToDevice));

	/* Warm up (JIT, pool), then wait for every thread. */
	for (int w = 0; w < 3; w++)
	{
		if (useUdf && grouped)
			for (int g = 0; g < 10; g++)
			{
				uint32_t	first = groupStart[g],
							n = groupStart[g + 1] - groupStart[g];

				if (n)
					CHECK(AnnCagraUdfSearch(res, params, sharedIndex, dQueries + (size_t) first * dims, n, dims,
											dNeighbors + (size_t) first * sel, dDistances + (size_t) first * sel,
											sel, dGroupParams + g, groupRate[g]));
			}
		else if (useUdf)
			CHECK(AnnCagraUdfSearch(res, params, sharedIndex, dQueries, batch, dims, dNeighbors, dDistances,
									sel, dParams, rate));
		else
			for (uint32_t q = 0; q < (mixed ? batch : 1); q++)
				BitsetSearch(res, params, dQueries, dNeighbors, dDistances, dBitsets[mixed ? q : 0],
							 mixed ? q : 0, mixed ? 1 : batch);
		CHECK(cuvsStreamSync(res));
	}
	atomic_fetch_add(&ready, 1);
	while (!go)
		usleep(100);
	while (!stop)
	{
		if (useUdf && grouped)
			for (int g = 0; g < 10; g++)
			{
				uint32_t	first = groupStart[g],
							n = groupStart[g + 1] - groupStart[g];

				if (n)
					CHECK(AnnCagraUdfSearch(res, params, sharedIndex, dQueries + (size_t) first * dims, n, dims,
											dNeighbors + (size_t) first * sel, dDistances + (size_t) first * sel,
											sel, dGroupParams + g, groupRate[g]));
			}
		else if (useUdf)
			CHECK(AnnCagraUdfSearch(res, params, sharedIndex, dQueries, batch, dims, dNeighbors, dDistances,
									sel, dParams, rate));
		else
			for (uint32_t q = 0; q < (mixed ? batch : 1); q++)
				BitsetSearch(res, params, dQueries, dNeighbors, dDistances, dBitsets[mixed ? q : 0],
							 mixed ? q : 0, mixed ? 1 : batch);
		CHECK(cuvsStreamSync(res));
		done += batch;
	}
	atomic_fetch_add(&totalQueries, done);
	free(queries);
	return NULL;
}

int
main(int argc, char **argv)
{
	const char *dir,
			   *caseName;
	uint32_t	threads;
	size_t		size;
	char		path[1024];
	cuvsResources_t res;
	cuvsDataset_t dataset = NULL;
	pthread_t	tids[64];
	double		start,
				elapsed;

	if (argc < 4)
	{
		fprintf(stderr, "usage: %s INDEX_DIR CASE MODE [threads] [seconds] [batch] [k] [itopk] [slack]\n", argv[0]);
		return 2;
	}
	dir = argv[1];
	caseName = argv[2];
	grouped = strcmp(argv[3], "udfgroup") == 0;
	useUdf = strcmp(argv[3], "udf") == 0 || grouped;
	threads = argc > 4 ? (uint32_t) atoi(argv[4]) : 8;
	seconds = argc > 5 ? atof(argv[5]) : 20;
	batch = argc > 6 ? (uint32_t) atoi(argv[6]) : 32;
	k = argc > 7 ? (uint32_t) atoi(argv[7]) : 500;
	itopk = argc > 8 ? (uint32_t) atoi(argv[8]) : 4096;
	slack = argc > 9 ? (uint32_t) atoi(argv[9]) : 256;
	if (threads < 1 || threads > 64 || batch < 1 || batch > MAX_BATCH)
		return 2;
	sel = k + slack < 1024 ? k + slack : (k >= 1024 ? k : 1024);
	mixed = strcmp(caseName, "mixed") == 0;

	vectors = Map(dir, "vectors.f32", &size);
	rows = size / (dims * 4);
	columns[0] = Map(dir, "attr-category.i32", &size);
	columns[1] = Map(dir, "attr-tenant.i32", &size);
	nwords = (uint32_t) ((rows + 31) / 32);
	for (uint64_t r = 0; r < rows; r++)
		if (columns[0][r] >= 0 && columns[0][r] < 100)
			categoryCount[columns[0][r]]++;

	/* One program per query slot; uniform cases repeat the same one. */
	for (uint32_t q = 0; q < batch; q++)
	{
		Program		p = {.n = 0};
		double		match;
		float		r;

		if (strcmp(caseName, "cat50") == 0)
			CategoryRange(&p, 0, 49);
		else if (strcmp(caseName, "cat10") == 0)
			CategoryRange(&p, 0, 9);
		else if (strcmp(caseName, "cat1") == 0)
			CategoryRange(&p, 7, 7);
		else
		{
			/* 1..10 categories, contiguous slots of each width (udfgroup). */
			int32_t		lo = (int32_t) ((q * 37) % 90);

			CategoryRange(&p, lo, lo + (int32_t) (q * 10 / batch));
		}
		programs.starts[q] = programs.nsteps;
		if (AnnUdfCompile(p.b, p.n, Column, NULL, programs.steps, 256, &programs.nsteps, programs.values, 256,
						  &programs.nvalues) != ANN_FILTER_OK)
			return 1;
		programs.starts[q + 1] = programs.nsteps;
		match = AnnUdfEstimate(programs.steps + programs.starts[q], 1, programs.values, CountRows, NULL, rows);
		r = (float) (1.0 - match / (double) rows);
		if (r > rate)
			rate = r;
		if (r > groupRate[q * 10 / batch])
			groupRate[q * 10 / batch] = r > 0.999f ? 0.999f : r;
	}
	for (int g = 0; g <= 10; g++)
		groupStart[g] = (uint32_t) ((g * batch + 9) / 10);
	if (rate > 0.999f)
		rate = 0.999f;

	cuvsSetLogLevel(CUVS_LOG_LEVEL_OFF);
	/* As the worker's --memory-pool: cuVS temporaries from an RMM pool. */
	CHECK(cuvsRMMPoolMemoryResourceEnable(10, 40, false));
	CHECK(cuvsResourcesCreate(&res));
	CHECK(cuvsCagraIndexCreate(&sharedIndex));
	snprintf(path, sizeof(path), "%s/shard-0.cagra", dir);
	CHECK(cuvsCagraDeserializeGraphAndDataset(res, path, sharedIndex, &dataset));
	CHECK(cuvsStreamSync(res));
	for (int c = 0; c < 2; c++)
	{
		CUDA(cudaMalloc((void **) &dColumns[c], sizeof(int32_t) * rows));
		CUDA(cudaMemcpy(dColumns[c], columns[c], sizeof(int32_t) * rows, cudaMemcpyHostToDevice));
	}
	if (!useUdf)
		for (uint32_t q = 0; q < (mixed ? batch : 1); q++)
		{
			CUDA(cudaMalloc((void **) &dBitsets[q], sizeof(uint32_t) * nwords));
			BuildBitset(q, dBitsets[q]);
		}

	for (uint32_t t = 0; t < threads; t++)
		pthread_create(&tids[t], NULL, Worker, (void *) (uintptr_t) t);
	while (atomic_load(&ready) < (int) threads)
		usleep(1000);
	start = Now();
	go = 1;
	usleep((useconds_t) (seconds * 1e6));
	stop = 1;
	for (uint32_t t = 0; t < threads; t++)
		pthread_join(tids[t], NULL);
	elapsed = Now() - start;
	printf("{\"case\":\"%s\",\"mode\":\"%s\",\"threads\":%u,\"batch\":%u,\"k\":%u,\"itopk\":%u,\"candidates\":%u,"
		   "\"filtering_rate\":%.4f,\"seconds\":%.2f,\"queries\":%llu,\"qps\":%.0f}\n",
		   caseName, grouped ? "udfgroup" : useUdf ? "udf" : (mixed ? "bitset_cached_per_query" : "bitset_cached"), threads, batch, k, itopk,
		   sel, useUdf ? rate : 0.0, elapsed, (unsigned long long) atomic_load(&totalQueries),
		   atomic_load(&totalQueries) / elapsed);
	return 0;
}
