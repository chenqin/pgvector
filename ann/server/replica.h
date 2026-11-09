#ifndef ANN_REPLICA_H
#define ANN_REPLICA_H

#include <stdatomic.h>
#include <stdint.h>

#define ANN_REPLICA_MAX_DEVICES 16
#define ANN_REPLICA_INITIAL_NS UINT64_C(100000)

/* Each GPU's counter occupies its own cache line. Callers align the router. */
typedef struct AnnReplicaLoad
{
	atomic_uint_fast64_t batch_ns;
	atomic_uint_fast64_t sampled_ns;
	atomic_uint active;
	unsigned char padding[64 - sizeof(atomic_uint) - 2 * sizeof(atomic_uint_fast64_t)];
} AnnReplicaLoad;

typedef struct AnnReplicaRouter
{
	AnnReplicaLoad devices[ANN_REPLICA_MAX_DEVICES];
} AnnReplicaRouter;

static inline uint64_t
AnnReplicaCost(const AnnReplicaLoad *load)
{
	uint64_t	cost = atomic_load_explicit(&load->batch_ns, memory_order_relaxed);

	return cost ? cost : ANN_REPLICA_INITIAL_NS;
}

/* Idle estimates decay so a temporarily slow GPU gets sampled again. */
static inline uint64_t
AnnReplicaPredictCost(const AnnReplicaLoad *load, uint64_t now)
{
	uint64_t	cost = AnnReplicaCost(load);
	uint64_t	sampled = atomic_load_explicit(&load->sampled_ns, memory_order_relaxed);

	if (sampled && now > sampled && cost > ANN_REPLICA_INITIAL_NS)
	{
		uint64_t	seconds = (now - sampled) / UINT64_C(1000000000);

		if (seconds > 63)
			seconds = 63;
		cost = ANN_REPLICA_INITIAL_NS + ((cost - ANN_REPLICA_INITIAL_NS) >> seconds);
	}
	return cost;
}

/*
 * Reserve a batch using active reservations and their observed service time.
 * Each lane scans from home, retaining its stream locality for close scores.
 * Slack is measured in batches at the chosen GPU's estimated service time.
 * The estimates are shared across indexes and include host reranking; they
 * are hints under concurrent arrivals, without a scheduler mutex.
 */
static inline uint32_t
AnnReplicaAcquire(AnnReplicaRouter *router, const int *devices, uint32_t copies,
				  uint64_t home, int loadAware, uint32_t slack, uint64_t now)
{
	uint32_t	preferred = (uint32_t) (home % copies);
	uint32_t	best = preferred;
	const AnnReplicaLoad *load = &router->devices[devices[preferred]];
	uint64_t	bestCost = AnnReplicaPredictCost(load, now);
	uint64_t	preferredScore = (1 + (uint64_t) atomic_load_explicit(&load->active, memory_order_relaxed)) * bestCost;
	uint64_t	bestScore = preferredScore;

	if (loadAware)
		for (uint32_t offset = 1; offset < copies; offset++)
		{
			uint32_t	copy = (preferred + offset) % copies;
			uint64_t	cost;
			uint64_t	score;

			load = &router->devices[devices[copy]];
			cost = AnnReplicaPredictCost(load, now);
			score = (1 + (uint64_t) atomic_load_explicit(&load->active, memory_order_relaxed)) * cost;
			if (score < bestScore)
			{
				best = copy;
				bestCost = cost;
				bestScore = score;
			}
		}
	if (preferredScore - bestScore <= (uint64_t) slack * bestCost)
		best = preferred;
	(void) atomic_fetch_add_explicit(&router->devices[devices[best]].active, 1, memory_order_relaxed);
	return best;
}

/* Bound cold stream initialization and scheduler outliers; smooth at 1/16. */
static inline void
AnnReplicaSample(AnnReplicaRouter *router, int device, uint64_t elapsed, uint64_t now)
{
	atomic_uint_fast64_t *target = &router->devices[device].batch_ns;
	uint_fast64_t old = atomic_load_explicit(target, memory_order_relaxed);
	uint_fast64_t next;

	do
	{
		uint64_t	base = old ? old : ANN_REPLICA_INITIAL_NS;
		uint64_t	sampled = atomic_load_explicit(&router->devices[device].sampled_ns, memory_order_relaxed);
		uint64_t	sample = elapsed;

		if (sampled && now > sampled && base > ANN_REPLICA_INITIAL_NS)
		{
			uint64_t	seconds = (now - sampled) / UINT64_C(1000000000);

			if (seconds > 63)
				seconds = 63;
			base = ANN_REPLICA_INITIAL_NS + ((base - ANN_REPLICA_INITIAL_NS) >> seconds);
		}

		if (sample > base * 8)
			sample = base * 8;
		if (sample < base / 8)
			sample = base / 8;
		next = sample > base ? base + (sample - base) / 16 : base - (base - sample) / 16;
	} while (!atomic_compare_exchange_weak_explicit(target, &old, next, memory_order_relaxed, memory_order_relaxed));
	atomic_store_explicit(&router->devices[device].sampled_ns, now, memory_order_relaxed);
}

static inline void
AnnReplicaRelease(AnnReplicaRouter *router, int device)
{
	(void) atomic_fetch_sub_explicit(&router->devices[device].active, 1, memory_order_relaxed);
}

#endif
