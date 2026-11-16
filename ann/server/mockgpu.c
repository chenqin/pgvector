#define _GNU_SOURCE

#include "backend.h"
#include "indexfile.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*
 * Stand-in for an accelerator adapter. Each search_gathered call models one
 * device submission: a fixed launch latency plus host-to-device transfer time,
 * then exact CPU scoring for the results. The waits use no CPU, so batching
 * amortizes launch cost the way it would on a device. It never claims real
 * accelerator performance; it only exercises scheduling and pipelining.
 */
static uint64_t launchNs = 100000;
static double bytesPerNs = 12.0;
static uint64_t lostAfter = 0;	/* ANN_MOCK_GPU_LOST_AFTER */
static uint64_t evaluateNs = 0; /* ANN_MOCK_GPU_EVALUATE_US */
static uint64_t uploadNs = 0;	/* ANN_MOCK_GPU_UPLOAD_US */
static uint64_t evaluatedNs = 0;	/* ANN_MOCK_GPU_EVALUATED_US */
static uint64_t canonicalNs = 0;	/* ANN_MOCK_GPU_CANONICAL_US */
static void (*cpuSearchIndex) (AnnIndexJob *jobs, uint32_t count);
static int	deviceStandIn;		/* what loaded indexes' device points at */
static uint64_t submissions = 0;	/* atomic */

static void
Wait(uint64_t ns)
{
	struct timespec ts = {(time_t) (ns / 1000000000), (long) (ns % 1000000000)};

	while (ns && nanosleep(&ts, &ts) != 0)
		;
}

static void
SearchGathered(AnnGatheredJob *jobs, uint32_t count)
{
	double		bytes = 0;

	for (uint32_t i = 0; i < count; i++)
		bytes += ((double) jobs[i].rows + 1) * jobs[i].dimensions * sizeof(float);
	Wait(launchNs + (uint64_t) (bytes / bytesPerNs));
	for (uint32_t i = 0; i < count; i++)
		AnnGatheredExact(&jobs[i]);
	if (lostAfter && __atomic_add_fetch(&submissions, 1, __ATOMIC_RELAXED) > lostAfter)
		for (uint32_t i = 0; i < count; i++)
			jobs[i].status = ANN_BACKEND_ERROR;
}

/*
 * ANN_MOCK_GPU_EVALUATE_US=N with --gpu-predicates on: every predicate
 * evaluation takes N microseconds longer (tests). The device declines it
 * after the wait, so the host evaluates it exactly.
 */
static int
Evaluate(AnnIndexFile *file, const unsigned char *program, size_t size, uint32_t *words, uint64_t *count)
{
	(void) file;
	(void) program;
	(void) size;
	(void) words;
	(void) count;
	Wait(evaluateNs);
	return 0;
}

/*
 * ANN_MOCK_GPU_EVALUATED_US=N: every host evaluation, once it has read the
 * index's arrays and before its result is used, is logged to stderr and
 * then takes N microseconds longer (tests: a remap from another connection
 * lands inside an evaluation).
 */
static void
Evaluated(AnnIndexFile *file)
{
	fprintf(stderr, "mock-gpu: evaluated %s\n", file->name);
	Wait(evaluatedNs);
}

/*
 * ANN_MOCK_GPU_CANONICAL_US=N: every canonicalisation, once it has read the
 * index's arrays and before the program is looked up or evaluated, is
 * logged to stderr and then takes N microseconds longer (tests: a remap
 * from another connection lands between a search's canonical form and its
 * evaluation).
 */
static void
Canonical(AnnIndexFile *file)
{
	fprintf(stderr, "mock-gpu: canonical %s\n", file->name);
	Wait(canonicalNs);
}

/*
 * ANN_MOCK_GPU_UPLOAD_US=N: loaded indexes have device state and cached
 * bitmaps fake device copies, each upload taking N microseconds (tests).
 * Uploads, and the copies each search job carries, are logged to stderr;
 * the searches themselves stay the CPU's.
 */
static void
Upload(AnnIndexFile *file, AnnIndexBitmap *bitmap)
{
	Wait(uploadNs);
	bitmap->device = calloc(ANN_INDEX_MAX_SHARDS, sizeof(void *));
	bitmap->deviceBytes = bitmap->device ? 64 : 0;
	fprintf(stderr, "mock-gpu: uploaded %s %016llx to %p\n", file->name, (unsigned long long) bitmap->id,
			(void *) bitmap->device);
}

static void
Release(AnnIndexFile *file, AnnIndexBitmap *bitmap)
{
	(void) file;
	free(bitmap->device);
	bitmap->device = NULL;
}

static AnnStatus
LoadIndex(const char *name, const char *directory)
{
	AnnIndexFile *file;
	AnnStatus	status = AnnIndexFileOpen(name, directory, &file);

	if (status != ANN_OK)
		return status;
	file->device = &deviceStandIn;	/* never dereferenced: no destroy hook */
	if ((status = AnnIndexRegister(file)) != ANN_OK)
	{
		file->device = NULL;
		AnnIndexFileClose(file);
	}
	return status;
}

static void
SearchIndex(AnnIndexJob *jobs, uint32_t count)
{
	for (uint32_t i = 0; i < count; i++)
		if (jobs[i].filtered)
			fprintf(stderr, "mock-gpu: searched %s with %p\n", jobs[i].name, (void *) jobs[i].deviceBitsets);
	cpuSearchIndex(jobs, count);
}

/* ANN_MOCK_GPU_LOST_AFTER=N: the device is lost after N submissions (tests). */
static const char *
Lost(void)
{
	return lostAfter && __atomic_load_n(&submissions, __ATOMIC_RELAXED) > lostAfter ? "simulated: ANN_MOCK_GPU_LOST_AFTER" : NULL;
}

const AnnBackend *
AnnMockGpuBackend(void)
{
	static AnnBackend backend;

	/* Called once from the worker's main thread before any service starts. */
	if (backend.name == NULL)
	{
		const char *launch = getenv("ANN_MOCK_GPU_LAUNCH_US");
		const char *bandwidth = getenv("ANN_MOCK_GPU_GBPS");

		if (launch)
			launchNs = strtoull(launch, NULL, 10) * 1000;
		if (bandwidth && strtod(bandwidth, NULL) > 0)
			bytesPerNs = strtod(bandwidth, NULL);
		if (getenv("ANN_MOCK_GPU_LOST_AFTER"))
			lostAfter = strtoull(getenv("ANN_MOCK_GPU_LOST_AFTER"), NULL, 10);
		if (getenv("ANN_MOCK_GPU_EVALUATE_US"))
		{
			evaluateNs = strtoull(getenv("ANN_MOCK_GPU_EVALUATE_US"), NULL, 10) * 1000;
			AnnIndexDeviceEvaluate = Evaluate;
		}
		if (getenv("ANN_MOCK_GPU_CANONICAL_US"))
		{
			canonicalNs = strtoull(getenv("ANN_MOCK_GPU_CANONICAL_US"), NULL, 10) * 1000;
			AnnIndexCanonicalHook = Canonical;
		}
		if (getenv("ANN_MOCK_GPU_EVALUATED_US"))
		{
			evaluatedNs = strtoull(getenv("ANN_MOCK_GPU_EVALUATED_US"), NULL, 10) * 1000;
			AnnIndexEvaluatedHook = Evaluated;
		}
		backend = *AnnCpuBackend();
		backend.name = "mock-gpu";
		backend.storage = "fp32-mock-device";
		backend.search_gathered = SearchGathered;
		backend.lost = Lost;
		if (getenv("ANN_MOCK_GPU_UPLOAD_US"))
		{
			uploadNs = strtoull(getenv("ANN_MOCK_GPU_UPLOAD_US"), NULL, 10) * 1000;
			AnnIndexBitmapDevice.upload = Upload;
			AnnIndexBitmapDevice.release = Release;
			cpuSearchIndex = backend.search_index;
			backend.load_index = LoadIndex;
			backend.search_index = SearchIndex;
		}
	}
	return &backend;
}
