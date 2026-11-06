/*
 * CAGRA filter UDF probe: loads one FP32 CAGRA shard through the cuVS C API
 * (as the worker does) and compares, on the same graph and search settings,
 * a device-resident bitset filter with the JIT-LTO predicate UDF of
 * ann/cagra_udf.cpp. Reports cold (JIT compile) and warm latency, recall
 * against an exact scan of the eligible rows, and whether every returned row
 * satisfies its query's predicate.
 *
 *   cagra_udf_probe INDEX_DIR [k] [itopk] [slack] [queries]
 *
 * Output is one JSON object per line.
 */
#include <cuda_runtime.h>
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/cagra.h>
#include <fcntl.h>
#include <math.h>
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

static uint64_t rows;
static uint32_t dims;
static const float *vectors;
static const int32_t *columns[2];	/* category, tenant */

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
	{
		perror("mmap");
		exit(1);
	}
	*size = st.st_size;
	return p;
}

static double
Now(void)
{
	struct timespec t;

	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

/* Program builders (ann/filter.h encoding). */
typedef struct
{
	unsigned char b[4096];
	size_t		n;
} Program;

static void
Put(Program *p, uint64_t v, int bytes)
{
	for (int i = 0; i < bytes; i++)
		p->b[p->n++] = (unsigned char) (v >> (8 * i));
}

static void
Leaf(Program *p, int op, const char *name, int negate)
{
	Put(p, op, 1);
	Put(p, strlen(name), 1);
	memcpy(p->b + p->n, name, strlen(name));
	p->n += strlen(name);
	Put(p, negate, 1);
}

static void
RangeLeaf(Program *p, const char *name, int32_t lo, int32_t hi)
{
	Leaf(p, ANN_FILTER_RANGE, name, 0);
	Put(p, (uint32_t) lo, 4);
	Put(p, (uint32_t) hi, 4);
}

static void
Values(Program *p, const char *name, const int32_t *v, uint32_t n)
{
	Leaf(p, ANN_FILTER_VALUES, name, 0);
	Put(p, n, 4);
	for (uint32_t i = 0; i < n; i++)
		Put(p, (uint32_t) v[i], 4);
}

/* Sorted copies of the columns, for AnnUdfEstimate. */
static int32_t *sorted[2];

static int
I32Cmp(const void *a, const void *b)
{
	int32_t		x = *(const int32_t *) a,
				y = *(const int32_t *) b;

	return (x > y) - (x < y);
}

static uint64_t
Lower(const int32_t *v, uint64_t n, int64_t x)
{
	uint64_t	lo = 0,
				hi = n;

	while (lo < hi)
	{
		uint64_t	mid = lo + (hi - lo) / 2;

		if (v[mid] < x)
			lo = mid + 1;
		else
			hi = mid;
	}
	return lo;
}

static uint64_t
CountRows(void *arg, int column, int32_t low, int32_t high)
{
	(void) arg;
	return Lower(sorted[column], rows, (int64_t) high + 1) - Lower(sorted[column], rows, low);
}

static int
Column(void *arg, const char *name)
{
	(void) arg;
	return strcmp(name, "category") == 0 ? 0 : strcmp(name, "tenant") == 0 ? 1 : -1;
}

typedef struct
{
	AnnUdfStep	steps[1024];
	int32_t		values[1024];
	uint32_t	starts[65];
	uint32_t	nsteps,
				nvalues;
} Batch;

static void
AddProgram(Batch *b, uint32_t q, const Program *p)
{
	b->starts[q] = b->nsteps;
	if (AnnUdfCompile(p->b, p->n, Column, NULL, b->steps, 1024, &b->nsteps, b->values, 1024, &b->nvalues) != ANN_FILTER_OK)
	{
		fprintf(stderr, "compile failed\n");
		exit(1);
	}
	b->starts[q + 1] = b->nsteps;
}

typedef struct
{
	float		d;
	uint32_t	id;
} Hit;

static int
HitCmp(const void *a, const void *b)
{
	const Hit  *x = a,
			   *y = b;

	return (x->d > y->d) - (x->d < y->d);
}

/* Exact top k over rows matching the query's steps; returns ids sorted. */
static void
Truth(const Batch *b, uint32_t q, const float *query, uint32_t k, uint32_t *out)
{
	Hit		   *heap = malloc(sizeof(Hit) * (rows / 4 + 1024));
	size_t		n = 0,
				cap = rows / 4 + 1024;
	float		bound = INFINITY;

#pragma omp parallel
	{
		Hit		   *local = malloc(sizeof(Hit) * 4 * k);
		size_t		ln = 0;

#pragma omp for schedule(static)
		for (uint64_t r = 0; r < rows; r++)
		{
			const float *v;
			float		d = 0;

			if (!AnnUdfMatch(b->steps + b->starts[q], b->starts[q + 1] - b->starts[q], b->values,
							 columns, r))
				continue;
			v = vectors + r * dims;
			for (uint32_t i = 0; i < dims; i++)
			{
				float		t = v[i] - query[i];

				d += t * t;
			}
			local[ln].d = d;
			local[ln++].id = (uint32_t) r;
			if (ln == 4 * k)
			{
				qsort(local, ln, sizeof(Hit), HitCmp);
				ln = k;
			}
		}
		qsort(local, ln, sizeof(Hit), HitCmp);
		if (ln > k)
			ln = k;
#pragma omp critical
		{
			memcpy(heap + n, local, ln * sizeof(Hit));
			n += ln;
		}
		free(local);
	}
	(void) cap;
	(void) bound;
	qsort(heap, n, sizeof(Hit), HitCmp);
	for (uint32_t i = 0; i < k; i++)
		out[i] = i < n ? heap[i].id : UINT32_MAX;
	free(heap);
}

static int
U32Cmp(const void *a, const void *b)
{
	uint32_t	x = *(const uint32_t *) a,
				y = *(const uint32_t *) b;

	return (x > y) - (x < y);
}

/* Recall of the first k of got (by distance order) against truth. */
static double
Recall(const uint32_t *got, uint32_t gotN, const uint32_t *truth, uint32_t k)
{
	uint32_t   *t = malloc(sizeof(uint32_t) * k);
	uint32_t	hits = 0,
				valid = 0;

	memcpy(t, truth, sizeof(uint32_t) * k);
	qsort(t, k, sizeof(uint32_t), U32Cmp);
	for (uint32_t i = 0; i < gotN && valid < k; i++)
	{
		if (got[i] >= rows)
			continue;
		valid++;
		if (bsearch(&got[i], t, k, sizeof(uint32_t), U32Cmp))
			hits++;
	}
	free(t);
	return (double) hits / k;
}

/* One bitset-filtered search of m queries starting at query q0. */
static void
BitsetSearch(cuvsResources_t res, cuvsCagraSearchParams_t params, cuvsCagraIndex_t index,
			 float *dQueries, uint32_t *dNeighbors, float *dDistances, uint32_t *dBitset,
			 uint32_t q0, uint32_t m, uint32_t sel, uint32_t nwords)
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
	CHECK(cuvsCagraSearch(res, params, index, &tq, &tn, &td, f));
}

/* Query q's bitset, built on the host and copied to dBitset. */
static void BuildBitset(const void *batch, uint32_t q, uint32_t *dBitset, uint32_t nwords,
						cudaStream_t stream);

static int
DoubleCmp(const void *a, const void *b)
{
	double		x = *(const double *) a,
				y = *(const double *) b;

	return (x > y) - (x < y);
}

static void
BuildBitset(const void *batch, uint32_t q, uint32_t *dBitset, uint32_t nwords, cudaStream_t stream)
{
	const Batch *b = batch;
	uint32_t   *words = calloc(nwords, 4);

#pragma omp parallel for schedule(static)
	for (uint64_t w = 0; w < nwords; w++)
	{
		uint32_t	bits = 0;

		for (uint32_t i = 0; i < 32 && w * 32 + i < rows; i++)
			if (AnnUdfMatch(b->steps + b->starts[q], b->starts[q + 1] - b->starts[q], b->values, columns,
							w * 32 + i))
				bits |= 1u << i;
		words[w] = bits;
	}
	CUDA(cudaMemcpyAsync(dBitset, words, sizeof(uint32_t) * nwords, cudaMemcpyHostToDevice, stream));
	CUDA(cudaStreamSynchronize(stream));
	free(words);
}

int
main(int argc, char **argv)
{
	const char *dir;
	uint32_t	k = argc > 2 ? atoi(argv[2]) : 500;
	uint32_t	itopk = argc > 3 ? atoi(argv[3]) : 4096;
	uint32_t	slack = argc > 4 ? atoi(argv[4]) : 256;
	uint32_t	nq = argc > 5 ? atoi(argv[5]) : 32;

	if (nq < 1 || nq > 64)
		nq = 32;
	uint32_t	sel;
	char		path[1024];
	size_t		size;
	cuvsResources_t res;
	cuvsCagraIndex_t index;
	cuvsDataset_t dataset = NULL;
	cuvsCagraSearchParams_t params;
	cudaStream_t stream;
	float	   *dQueries;
	uint32_t   *dNeighbors;
	float	   *dDistances;
	uint32_t   *dBitset;
	static uint32_t *dBitsets[64];
	AnnUdfParams *dParams;
	AnnUdfParams *dGroupParams;
	AnnUdfStep *dSteps;
	int32_t    *dValues;
	uint32_t   *dStarts;
	int32_t    *dColumns[2];
	float	   *queries;
	uint32_t   *hNeighbors;
	uint32_t	nwords;
	double		t0;

	if (argc < 2)
	{
		fprintf(stderr, "usage: %s INDEX_DIR [k] [itopk] [slack] [queries]\n", argv[0]);
		return 2;
	}
	dir = argv[1];
	sel = k + slack < 1024 ? k + slack : (k >= 1024 ? k : 1024);
	vectors = Map(dir, "vectors.f32", &size);
	dims = 128;
	rows = size / (dims * 4);
	columns[0] = Map(dir, "attr-category.i32", &size);
	columns[1] = Map(dir, "attr-tenant.i32", &size);
	nwords = (uint32_t) ((rows + 31) / 32);
	for (int c = 0; c < 2; c++)
	{
		sorted[c] = malloc(sizeof(int32_t) * rows);
		memcpy(sorted[c], columns[c], sizeof(int32_t) * rows);
		qsort(sorted[c], rows, sizeof(int32_t), I32Cmp);
	}

	cuvsSetLogLevel(CUVS_LOG_LEVEL_OFF);
	CHECK(cuvsResourcesCreate(&res));
	CHECK(cuvsCagraIndexCreate(&index));
	CHECK(cuvsCagraSearchParamsCreate(&params));
	CHECK(cuvsStreamGet(res, &stream));
	snprintf(path, sizeof(path), "%s/shard-0.cagra", dir);
	t0 = Now();
	CHECK(cuvsCagraDeserializeGraphAndDataset(res, path, index, &dataset));
	CHECK(cuvsStreamSync(res));
	fprintf(stderr, "loaded %llu rows in %.1f s\n", (unsigned long long) rows, (Now() - t0) / 1e3);

	params->itopk_size = itopk;
	params->algo = MULTI_CTA;
	params->search_width = 4;
	params->max_iterations = 0;
	params->persistent = false;

	CUDA(cudaMalloc((void **) &dQueries, sizeof(float) * 64 * dims));
	CUDA(cudaMalloc((void **) &dNeighbors, sizeof(uint32_t) * 64 * 1024));
	CUDA(cudaMalloc((void **) &dDistances, sizeof(float) * 64 * 1024));
	CUDA(cudaMalloc((void **) &dBitset, sizeof(uint32_t) * nwords));
	for (uint32_t q = 0; q < nq; q++)
		CUDA(cudaMalloc((void **) &dBitsets[q], sizeof(uint32_t) * nwords));
	CUDA(cudaMalloc((void **) &dParams, sizeof(AnnUdfParams)));
	CUDA(cudaMalloc((void **) &dGroupParams, sizeof(AnnUdfParams) * 10));
	CUDA(cudaMalloc((void **) &dSteps, sizeof(AnnUdfStep) * 1024));
	CUDA(cudaMalloc((void **) &dValues, sizeof(int32_t) * 1024));
	CUDA(cudaMalloc((void **) &dStarts, sizeof(uint32_t) * 65));
	for (int c = 0; c < 2; c++)
	{
		CUDA(cudaMalloc((void **) &dColumns[c], sizeof(int32_t) * rows));
		CUDA(cudaMemcpy(dColumns[c], columns[c], sizeof(int32_t) * rows, cudaMemcpyHostToDevice));
	}
	{
		AnnUdfParams p;

		memset(&p, 0, sizeof(p));
		p.columns[0] = dColumns[0];
		p.columns[1] = dColumns[1];
		p.steps = dSteps;
		p.values = dValues;
		p.starts = dStarts;
		p.rowBase = 0;
		CUDA(cudaMemcpy(dParams, &p, sizeof(p), cudaMemcpyHostToDevice));
		/* udf_grouped: a group's query ids start at its first slot. */
		for (int g = 0; g < 10; g++)
		{
			p.starts = dStarts + (g * nq + 9) / 10;
			CUDA(cudaMemcpy(dGroupParams + g, &p, sizeof(p), cudaMemcpyHostToDevice));
		}
	}

	/* Queries: dataset rows with deterministic noise. */
	queries = malloc(sizeof(float) * 64 * dims);
	srand(129);
	for (uint32_t q = 0; q < 64; q++)
	{
		uint64_t	r = ((uint64_t) rand() * 7919u + q * 104729u) % rows;

		for (uint32_t d = 0; d < dims; d++)
			queries[q * dims + d] = vectors[r * dims + d] + ((float) rand() / RAND_MAX - 0.5f) * 0.05f;
	}
	CUDA(cudaMemcpy(dQueries, queries, sizeof(float) * 64 * dims, cudaMemcpyHostToDevice));
	hNeighbors = malloc(sizeof(uint32_t) * 64 * 1024);

	/*
	 * Cases: uniform predicates (one program for all queries, where a cached
	 * bitset applies) and a mixed batch (each query its own predicate).
	 */
	const char *caseNames[] = {"cat_50pct", "cat_10pct", "cat_1pct", "cat10pct_and_tenant50pct", "mixed_1to10pct"};
	double		caseSel[] = {0.5, 0.1, 0.01, 0.05, -1};

	for (int c = 0; c < 5; c++)
	{
		Batch	   *b = calloc(1, sizeof(Batch));
		uint32_t   *truth = malloc(sizeof(uint32_t) * nq * k);
		uint32_t	batches[] = {1, 16, nq};

		(void) caseSel;

		for (uint32_t q = 0; q < nq; q++)
		{
			Program		p = {.n = 0};

			if (c == 0)
				RangeLeaf(&p, "category", 0, 49);
			else if (c == 1)
				RangeLeaf(&p, "category", 0, 9);
			else if (c == 2)
			{
				int32_t		v = 7;

				Values(&p, "category", &v, 1);
			}
			else if (c == 3)
			{
				RangeLeaf(&p, "category", 10, 19);
				RangeLeaf(&p, "tenant", 0, 499);
				Put(&p, ANN_FILTER_AND, 1);
			}
			else
			{
				/* Query q: category in a q-dependent window of 1..10 values. */
				int32_t		lo = (int32_t) ((q * 37) % 90);

				/* 1..10 categories, contiguous slots of each width (udf_grouped). */
				RangeLeaf(&p, "category", lo, lo + (int32_t) (q * 10 / nq));
			}
			AddProgram(b, q, &p);
		}
		t0 = Now();
		for (uint32_t q = 0; q < nq; q++)
			Truth(b, q, queries + (size_t) q * dims, k, truth + (size_t) q * k);
		fprintf(stderr, "%s: truth in %.1f s\n", caseNames[c], (Now() - t0) / 1e3);
		CUDA(cudaMemcpy(dSteps, b->steps, sizeof(AnnUdfStep) * b->nsteps, cudaMemcpyHostToDevice));
		CUDA(cudaMemcpy(dValues, b->values, sizeof(int32_t) * (b->nvalues ? b->nvalues : 1), cudaMemcpyHostToDevice));
		CUDA(cudaMemcpy(dStarts, b->starts, sizeof(uint32_t) * (nq + 1), cudaMemcpyHostToDevice));
		if (c == 4)
			for (uint32_t q = 0; q < nq; q++)
				BuildBitset(b, q, dBitsets[q], nwords, stream);

		for (int bi = 0; bi < 3; bi++)
		{
			uint32_t	m = batches[bi];

			if (m > nq || (bi > 0 && m <= batches[bi - 1]))
				continue;

			/*
			 * filtering_rate from the attribute statistics: the batch's most
			 * selective predicate sets it, so no query's search is too small.
			 */
			float		rate = 0;

			for (uint32_t q = 0; q < m; q++)
			{
				double		match = AnnUdfEstimate(b->steps + b->starts[q], b->starts[q + 1] - b->starts[q],
												   b->values, CountRows, NULL, rows);
				float		r = (float) (1.0 - match / (double) rows);

				if (r > rate)
					rate = r;
			}
			if (rate > 0.999f)
				rate = 0.999f;
			/* A selectivity estimate 10x too broad. */
			float		lowRate = 1.0f - (1.0f - rate) * 10.0f;

			if (lowRate < 0)
				lowRate = 0;

			/* udf_grouped: each window width's slots, searched at its own rate. */
			uint32_t	groupStart[11];
			float		groupRate[10] = {0};

			for (int g = 0; g <= 10; g++)
			{
				groupStart[g] = (uint32_t) ((g * nq + 9) / 10);
				if (groupStart[g] > m)
					groupStart[g] = m;
			}
			for (uint32_t q = 0; q < m; q++)
			{
				double		match = AnnUdfEstimate(b->steps + b->starts[q], b->starts[q + 1] - b->starts[q],
												   b->values, CountRows, NULL, rows);
				float		r = (float) (1.0 - match / (double) rows);
				int			g = (int) (q * 10 / nq);

				if (r > groupRate[g])
					groupRate[g] = r > 0.999f ? 0.999f : r;
			}

			/*
			 * 0 bitset (cached if uniform, else built per query), 1 udf, 2
			 * udf without a rate, 3 cached bitsets per query (mixed only), 4
			 * udf with lowRate, 5 udf grouped by window width (mixed only).
			 */
			for (int mode = 0; mode < 6; mode++)
			{
				double		times[15];
				double		evalMs = 0;
				double		recall = 0,
							worst = 1;
				uint64_t	invalid = 0,
							returned = 0;
				double		cold = 0;
				int			uniform = c != 4;
				const char *modeName = mode == 0 ? (uniform ? "bitset_cached" : "bitset_per_query") :
					mode == 1 ? "udf" : mode == 2 ? "udf_no_rate" : mode == 3 ? "bitset_cached_per_query" :
					mode == 4 ? "udf_rate_10x_broad" : "udf_grouped";
				double		usedRate = mode == 1 ? rate : mode == 2 ? -1.0 : mode == 4 ? lowRate : 0.0;

				if ((mode == 3 || mode == 5) && uniform)
					continue;

				/* Per-query searches are planned for one query each. */
				params->max_queries = mode == 3 || (mode == 0 && !uniform) ? 1 : m;
				for (int it = -3; it < 15; it++)
				{
					double		start = Now();

					if (mode == 0)
					{
						/*
						 * Bitset: one search per distinct predicate. For the
						 * mixed case each query needs its own bitset, which
						 * is built on the host here and timed separately.
						 */
						uint32_t	groups = uniform ? 1 : m;

						for (uint32_t g = 0; g < groups; g++)
						{
							uint32_t	q0 = uniform ? 0 : g;

							if (it == -3 || !uniform)
							{
								double		e0 = Now();

								BuildBitset(b, q0, dBitset, nwords, stream);
								if (it >= 0)
									evalMs += Now() - e0;
							}
							BitsetSearch(res, params, index, dQueries, dNeighbors, dDistances, dBitset, q0,
										 uniform ? m : 1, sel, nwords);
							if (!uniform)
								CHECK(cuvsStreamSync(res));
						}
					}
					else if (mode == 3)
					{
						/*
						 * Each query's bitset already on the device (as
						 * cached bitmaps with --device-bitmaps): one search
						 * per query, queued without waiting, one sync at the
						 * end.
						 */
						for (uint32_t q = 0; q < m; q++)
							BitsetSearch(res, params, index, dQueries, dNeighbors, dDistances,
										 dBitsets[q], q, 1, sel, nwords);
					}
					else if (mode == 5)
						for (int g = 0; g < 10; g++)
						{
							uint32_t	first = groupStart[g],
										n = groupStart[g + 1] - groupStart[g];

							if (n)
								CHECK(AnnCagraUdfSearch(res, params, index, dQueries + (size_t) first * dims, n, dims,
														dNeighbors + (size_t) first * sel, dDistances + (size_t) first * sel,
														sel, dGroupParams + g, groupRate[g]));
						}
					else
						CHECK(AnnCagraUdfSearch(res, params, index, dQueries, m, dims, dNeighbors, dDistances,
												sel, dParams, mode == 1 ? rate : mode == 4 ? lowRate : -1.0f));
					CHECK(cuvsStreamSync(res));
					if (mode == 1 && it == -3)
						cold = Now() - start;
					if (it >= 0)
						times[it] = Now() - start;
				}
				CUDA(cudaMemcpy(hNeighbors, dNeighbors, sizeof(uint32_t) * m * sel, cudaMemcpyDeviceToHost));
				for (uint32_t q = 0; q < m; q++)
				{
					double		r = Recall(hNeighbors + (size_t) q * sel, sel, truth + (size_t) q * k, k);

					recall += r;
					if (r < worst)
						worst = r;
					for (uint32_t i = 0; i < sel; i++)
					{
						uint32_t	id = hNeighbors[(size_t) q * sel + i];

						if (id >= rows)
							continue;
						returned++;
						if (!AnnUdfMatch(b->steps + b->starts[q], b->starts[q + 1] - b->starts[q], b->values,
										 columns, id))
							invalid++;
					}
				}
				qsort(times, 15, sizeof(double), DoubleCmp);
				printf("{\"case\":\"%s\",\"mode\":\"%s\",\"batch\":%u,\"k\":%u,\"itopk\":%u,\"candidates\":%u,"
					   "\"median_ms\":%.3f,\"p90_ms\":%.3f,\"host_bitset_ms_per_iter\":%.3f,\"cold_ms\":%.1f,"
					   "\"filtering_rate\":%.4f,\"mean_recall\":%.4f,\"min_recall\":%.4f,\"returned\":%llu,\"ineligible\":%llu}\n",
					   caseNames[c], modeName, m, k, itopk,
					   sel, times[7], times[13], evalMs / 15, cold, usedRate, recall / m, worst,
					   (unsigned long long) returned, (unsigned long long) invalid);
				fflush(stdout);
			}
		}
		free(b);
		free(truth);
	}
	return 0;
}
