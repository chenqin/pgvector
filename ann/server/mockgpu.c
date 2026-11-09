#include "backend.h"

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
		backend = *AnnCpuBackend();
		backend.name = "mock-gpu";
		backend.storage = "fp32-mock-device";
		backend.search_gathered = SearchGathered;
		backend.lost = Lost;
	}
	return &backend;
}
