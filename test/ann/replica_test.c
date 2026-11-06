#include "replica.h"

#include <assert.h>
#include <pthread.h>
#include <stdio.h>

_Static_assert(sizeof(AnnReplicaLoad) == 64, "each load occupies one cache line");
static _Alignas(64)
AnnReplicaRouter router;
static const int devices[6] = {0, 1, 2, 3, 4, 5};

static void *
Concurrent(void *arg)
{
	uint64_t	home = (uint64_t) (uintptr_t) arg;

	for (unsigned int i = 0; i < 100000; i++)
	{
		uint32_t	copy = AnnReplicaAcquire(&router, devices, 6, home, 1, 1, 0);

		assert(copy < 6);
		assert(atomic_load(&router.devices[devices[copy]].active) > 0);
		AnnReplicaRelease(&router, devices[copy]);
	}
	return NULL;
}

int
main(void)
{
	uint32_t	picked[64];
	pthread_t	threads[16];

	assert(atomic_is_lock_free(&router.devices[0].active));
	assert(atomic_is_lock_free(&router.devices[0].batch_ns));
	/* Exact least-active selection, including a non-power-of-two GPU count. */
	for (uint32_t i = 0; i < 6; i++)
	{
		picked[i] = AnnReplicaAcquire(&router, devices, 6, 0, 1, 0, 0);
		assert(picked[i] == i);
	}
	for (uint32_t i = 0; i < 6; i++)
		AnnReplicaRelease(&router, devices[picked[i]]);
	/* The locality slack tolerates one extra batch; a hotter home spills. */
	picked[0] = AnnReplicaAcquire(&router, devices, 6, 0, 1, 1, 0);
	picked[1] = AnnReplicaAcquire(&router, devices, 6, 0, 1, 1, 0);
	picked[2] = AnnReplicaAcquire(&router, devices, 6, 0, 1, 1, 0);
	assert(picked[0] == 0 && picked[1] == 0 && picked[2] == 1);
	for (uint32_t i = 0; i < 3; i++)
		AnnReplicaRelease(&router, devices[picked[i]]);
	/* Home mode provides a control and must preserve the full lane ordinal. */
	for (uint32_t i = 0; i < 64; i++)
		picked[i] = AnnReplicaAcquire(&router, devices, 6, i, 0, 1, 0);
	for (uint32_t d = 0; d < 6; d++)
		assert(atomic_load(&router.devices[d].active) == (d < 4 ? 11 : 10));
	for (uint32_t i = 0; i < 64; i++)
		AnnReplicaRelease(&router, devices[picked[i]]);
	/* Loads shared by other indexes must influence the same router. */
	for (uint32_t i = 0; i < 8; i++)
		picked[i] = AnnReplicaAcquire(&router, devices, 1, 0, 0, 1, 0);
	picked[8] = AnnReplicaAcquire(&router, devices, 6, 0, 1, 1, 0);
	assert(picked[8] != 0);
	for (uint32_t i = 0; i < 9; i++)
		AnnReplicaRelease(&router, devices[picked[i]]);
	/* A slow device must shed work even when reservation counts are equal. */
	atomic_store(&router.devices[0].batch_ns, ANN_REPLICA_INITIAL_NS * 8);
	picked[0] = AnnReplicaAcquire(&router, devices, 6, 0, 1, 1, 0);
	assert(picked[0] != 0);
	AnnReplicaRelease(&router, devices[picked[0]]);
	for (uint32_t i = 0; i < 100; i++)
		AnnReplicaSample(&router, 0, ANN_REPLICA_INITIAL_NS, 0);
	assert(AnnReplicaCost(&router.devices[0]) < ANN_REPLICA_INITIAL_NS * 2);
	atomic_store(&router.devices[0].batch_ns, 0);
	/* Old slow estimates must permit probes and recovery, without traffic. */
	atomic_store(&router.devices[0].batch_ns, ANN_REPLICA_INITIAL_NS * 8);
	atomic_store(&router.devices[0].sampled_ns, 1000);
	picked[0] = AnnReplicaAcquire(&router, devices, 6, 0, 1, 1, UINT64_C(4000001000));
	assert(picked[0] == 0);
	AnnReplicaRelease(&router, devices[picked[0]]);
	AnnReplicaSample(&router, 0, ANN_REPLICA_INITIAL_NS, UINT64_C(4000001000));
	assert(AnnReplicaCost(&router.devices[0]) < ANN_REPLICA_INITIAL_NS * 2);
	atomic_store(&router.devices[0].batch_ns, 0);
	atomic_store(&router.devices[0].sampled_ns, 0);
	/* Concurrent reservations must leave no load behind, without a mutex. */
	for (uintptr_t i = 0; i < 16; i++)
		assert(pthread_create(&threads[i], NULL, Concurrent, (void *) i) == 0);
	for (uint32_t i = 0; i < 16; i++)
		assert(pthread_join(threads[i], NULL) == 0);
	for (uint32_t d = 0; d < 6; d++)
		assert(atomic_load(&router.devices[d].active) == 0);
	puts("replica routing: passed");
	return 0;
}
