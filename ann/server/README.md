# ANN server

The ANN server is the worker PostgreSQL sends vector searches to: `ann-worker` (CPU, no dependencies) and `ann-worker-cuvs` (NVIDIA GPUs through cuVS 26.10). It loads index directories built by the [index builder](../builder/README.md) and answers searches over a Unix socket, TCP or RDMA. It can run on the database host or on separate GPU nodes.

| Interface | Document |
| --- | --- |
| Requests from PostgreSQL (or any client) | [PROTOCOL.md](PROTOCOL.md) |
| Index directories it loads | [INDEX_FORMAT.md](../common/INDEX_FORMAT.md) |
| Predicate programs in filtered searches | [filter.h](../common/filter.h) |

Runtime code here is C. Python is used only by the tests and benchmarks under `test/ann/` and `benchmarks/ann/`. The server compiles `../common` (the filter language and cost model it shares with the PostgreSQL extension) and nothing else outside this directory.

## Build and test

A C11 compiler, POSIX threads and Make, on Linux (macOS for development). Windows isn't supported.

```sh
make -C ann/server                       # ann-worker and libannbatch.a
make -C ann/server check
# GPU hosts: also ann-worker-cuvs (pip SDK paths: see docs/ann-cuvs-index.md)
make -C ann/server USE_CUVS=1 CUVS_PREFIX=/path/to/libcuvs CUDA_HOME=/path/to/cuda
make -C ann/server check-cuvs CUVS_PREFIX=/path/to/libcuvs
```

`USE_RDMA=1` adds RDMA search frames over InfiniBand or RoCE verbs ([guide](../../docs/ann-networked.md#rdma-infiniband-roce)). `USE_CUVS` defaults to 0, so hosts without a GPU SDK build and test every CPU path. Both builds serve loaded indexes: the CPU worker scans them exactly, which makes it the test reference, and the cuVS worker searches CAGRA shards on all GPUs ([guide](../../docs/ann-cuvs-index.md)). `make check` uses Python to drive the protocol and scheduler tests; live PostgreSQL and GPU tests are gated.

## Deploy

```sh
make -C ann/server USE_CUVS=1 CUVS_PREFIX=... CUDA_HOME=...
sudo make -C ann/server install PREFIX=/usr/local        # ann-worker, and ann-worker-cuvs if built
```

`install` copies the binaries to `$(DESTDIR)$(PREFIX)/bin`. `ann-worker-cuvs` needs the cuVS 26.10 and CUDA runtime libraries at run time (`LD_LIBRARY_PATH` for pip installs) and optionally `libnvrtc.so` for the GPU gather scan.

```sh
# A GPU node serving PostgreSQL hosts on the network
head -c 32 /dev/urandom | base64 > /etc/ann/token && chmod 600 /etc/ann/token
ann-worker-cuvs --backend cuvs --listen-tcp 0.0.0.0:7700 --auth-token-file /etc/ann/token --lanes 64

# On the database host, over a Unix socket
ann-worker-cuvs --backend cuvs --listen /run/ann/ann.sock --lanes 64
```

```sql
ALTER SYSTEM SET ann.worker_address = 'gpu1:7700';   -- or ann.worker_socket = '/run/ann/ann.sock'
ALTER SYSTEM SET ann.worker_auth_token = '...';
SELECT pg_reload_conf();
SELECT ann_index_load('docs', '/data/ann/docs');      -- a builder output directory on the server's host
```

Run it under a supervisor that restarts it (systemd `Restart=always`): after a GPU context is lost the server exits with status 70 so the driver reclaims its memory, and clients fail over meanwhile. The worker options are listed in the [cuVS guide](../../docs/ann-cuvs-index.md#b200-throughput-options) and the [networked guide](../../docs/ann-networked.md).

## C batching and admission API

[batching.h](batching.h) exposes the native API; [batching.c](batching.c) implements scheduling, resource accounting, deadlines, cancellation, and request/result ownership. There is no Python scheduler or runtime client dependency.

```c
#include "batching.h"

int main(void)
{
    float vectors[] = {0, 1, 1, 1, 2, 1};
    float query[] = {1, 1};
    AnnBatchOptions options;
    AnnBatcher *batcher;
    AnnRequest *request;
    uint32_t count;

    AnnBatchDefaultOptions(&options);
    if (AnnBatchCreate(AnnCpuBackend(), 1, 3, 2, vectors,
                       &options, &batcher) != ANN_OK)
        return 1;
    if (AnnBatchSubmit(batcher, query, 1, 0, NULL, 0,
                       30000000, &request) != ANN_OK)
    {
        AnnBatchClose(batcher);
        return 1;
    }
    if (AnnRequestWait(request, 30000000) == ANN_OK)
    {
        const AnnCandidate *results = AnnRequestResults(request, &count);
        /* Consume results[0..count); storage belongs to request. */
        (void) results;
    }
    AnnRequestRelease(request);
    AnnBatchClose(batcher);
    return 0;
}
```

Link with `-Iann/server -Iann/common -Lann/server -lannbatch -pthread -lm`. The scheduler owns a canonical index copy by default. The worker uses `AnnBatchCreateBorrowed()` to share its immutable index; it closes the scheduler before unloading or replacing that index. Failed replacement builds preserve the old generation.

Admission returns `ANN_OVERLOADED` without queueing when request count, bytes, or estimated work exceed capacity. The C byte charge includes the request structure, query, filter, and reserved result array. It excludes the resident index, allocator overhead, bounded scheduler metadata, input wire frames, and results retained by callers after completion. Work is estimated as eligible count times dimension. Limits apply to queued and running work; timed-out running requests retain reservations until execution finishes.

Compatible jobs share metric, effective k, filter presence, and a coarse selectivity class. The oldest request anchors selection. The dispatcher groups jobs up to row/byte/work limits or the collection delay. CPU distance evaluation remains serial; there is no claimed GPU kernel batching or automatic cross-process aggregation. CPU worker search frames use this C scheduler directly. External protocol frames are still read serially.

A monotonic-clock watcher expires queued and running requests. Queued cancellation removes the job; running cancellation is not allowed. `AnnBatchClose()` cancels queued work, marks unfinished results canceled, waits for backend work to finish, and releases the index. It does not forcibly terminate a C thread or preempt a GPU kernel. Request handles remain valid until `AnnRequestRelease()`, even after service close. Caller-held results are released with their handle.

Lifecycle rules: coordinate producer threads before closing the service; do not use a batcher handle after `AnnBatchClose()`. Do not release a request handle while another caller is using that same handle. Backend descriptors must outlive the service. Worker threads must never call PostgreSQL backend APIs; future backend integration uses PostgreSQL's memory, lock, interrupt, and resource mechanisms in the backend process.

## Protocol

The wire protocol, statuses and every operation are in [PROTOCOL.md](PROTOCOL.md).

## Test references and benchmarks

[test/ann/support/](../../test/ann/support/) contains Python-only protocol clients and a frozen-table SQL oracle. Snapshot checksum/save/load helpers there are test conveniences; production canonical export/import is implemented in C, while a durable C artifact-file format remains pending.

```sh
python3 -m pip install -r test/ann/requirements-postgres.txt
ANN_PG_DSN='dbname=your_test_database' make -C ann/server check-postgres
python3 benchmarks/ann/postgres.py --rows 10000 --dimensions 32 > /tmp/ann-baseline.sql
psql -X -v ON_ERROR_STOP=1 -f /tmp/ann-baseline.sql
python3 benchmarks/ann/search.py --mode shared --batch 16
python3 benchmarks/ann/batching.py --requests 1000 --sequential
python3 benchmarks/ann/batching.py --requests 1000 --pending 1024 --batch-rows 64
```

The batching launcher runs [a C benchmark](../../benchmarks/ann/batch.c); search and scheduling execute entirely in C. It reports successful/offered/rejected counts, elapsed time, and batch occupancy. The test-client search benchmark measures IPC and uses an independent distance oracle. Neither benchmark includes SQL eligibility or establishes production QPS. The PostgreSQL fixtures create temporary or uniquely named test objects and require the corresponding database permissions.

C tests cover batching/filter isolation, input copying, budgets, queued/running deadlines, concurrent producers, early handle release, shutdown during execution, backend errors, and result lifetime after close. Python protocol tests independently validate metrics and filters. Sanitizer checks:

```sh
make -C ann/server clean
make -C ann/server CFLAGS='-O1 -g -std=c11 -Wall -Wextra -Werror -pedantic -fsanitize=address,undefined -fno-omit-frame-pointer' check
```

## cuVS integration

`cuvs.c` is compiled and validated against cuVS 26.10 on NVIDIA B200 GPUs (`make -C ann/server check-cuvs`):

- **Gathered search** serves the [pipelined CustomScan](../../docs/ann-worker-pipeline.md) with exact GPU brute force.
- **Loaded indexes** serve sharded CAGRA search across all GPUs, with bitset filters and an exact route for selective filters ([guide](../../docs/ann-cuvs-index.md)).
- **GPU failures and memory.** A search that fails in the backend makes the worker check every device for a lost context (a sticky CUDA error: illegal address, launch failure, ECC, assert and the like). A lost context fails every later call and keeps the process's device and pinned memory, so the worker logs the device and error and exits with status 70 without running exit handlers: the driver takes the memory back, clients see the connection close and fail over to other replicas, and a supervisor (systemd `Restart=always`) starts it again. `ANN_MOCK_GPU_LOST_AFTER=N` makes the `mock-gpu` backend lose its device after N submissions, for tests. Unloading an index (operation 26) frees its shards once their searches finish; with the default RMM pool the freed device memory stays with the worker for later loads, and `--device-allocator plain` returns it to the device ([trade-off](../../docs/ann-cuvs-index.md#b200-throughput-options)).

The stdin-mode resident CAGRA build (operations 1–4 with `--backend cuvs`) is still the older unvalidated draft. Use loaded indexes instead. hipVS and NVFP4 adapters are not implemented: cuVS 26.10 CAGRA supports FP32, FP16, int8 and uint8 datasets, and no FP4 or FP8 path.

C PostgreSQL worker integration is implemented: `--listen PATH` serves pipelined gathered searches (operations 7–10) to `AnnFilteredSearch`, with a `mock-gpu` backend standing in for accelerators ([guide](../../docs/ann-worker-pipeline.md)). Sustained-load validation remains. Python remains test/benchmark tooling only.

### Replica routing

With `--replicate on`, `--replica-routing home` keeps each worker lane on its
home GPU. `--replica-routing load` compares active search batches and a
smoothed observation of their service time across GPUs. It retains home
when the estimate differs by no more than `--replica-load-slack N` batches
(default 1). These settings are worker-wide. Load counters include searches
of other loaded indexes and unreplicated indexes spanning several GPUs.

Reservations span device work and host reranking, so the time estimate is
not CUDA kernel time. It reflects the recent workload mix as well as device
contention. Cold initialization and timing outliers are bounded before the
1/16 moving-average update. Idle slow estimates decay toward the initial
100 microseconds, halving the difference once a second, so the router can
probe and reuse a GPU after temporary contention. Counter operations use
C11 atomics; replica choice takes no scheduler mutex.

Operation 31 exposes per-index counts and global per-device load estimates.
`device_mask` names every GPU a copy spans; `device` is its first shard's
GPU. Snapshots are approximate while work runs. `frozen_stress.py` records
before/after snapshots when `--index-name` is supplied, outside the measured
workload. Home routing remains the default while performance tuning continues.

### Generalized matrix selection and GPU reranking

The cuVS build now requires the C++20 SDK headers and `libcuvs` as well as
the C API. RAFT, RMM and CCCL headers must be available; the Makefile finds
standard sibling pip packages automatically, or `CUVS_CXXFLAGS` can supply
include paths for another SDK layout. The CPU-only build is unchanged.

`--matrix-gather on` enables Blackwell tensor-core matrix scoring and cuVS
batched selection for larger k, including k=500. `--group-gather-limit ROWS`
sets the largest shared filter it scans; the existing fused top-32 path is
retained for small k. `--matrix-max-mb N` (default 64, range 1–1024) bounds
one lane's input score matrix. Query batches split into tiles to respect it.
This bound excludes selector workspace, output buffers and resident vectors.

`--gpu-rerank on` reranks CAGRA candidates using resident original FP32
vectors. Matrix gathers select extra candidates, rerank on device, then
select each query's final top-k with row-ID payloads. Accumulation uses
double precision, matching the host rerank arithmetic. FP32 CAGRA datasets
are reused; other datasets need an additional FP32 copy, about 5.12 GB per
GPU for 10M × 128 dimensions. If a copy cannot be allocated, that shard
retains host reranking. Both switches default to off; GPU reranking adds resident memory and matrix
gathers require the Blackwell scorer. The measured B200 configuration enables
both and uses `--group-gather-limit 500000`, `--replica-routing home`,
64 lanes and batches up to 32. The 500k cutoff sends broader filters to CAGRA.

Routing statistics include `matrix_queries`, `matrix_peak_score_bytes`,
`gpu_rerank_candidates`, and per-copy `rerank_ready`/`rerank_bytes`, so a test
can verify which path actually ran and how much resident memory was added.
The benchmark comparison is documented in
[the k=500 profile](../../docs/ann-k500-profile.md).

`--cost-profile PATH` enables the optional [measured cost model](../../docs/ann-cost-model.md).
It chooses among NVFP4/CAGRA and CPU/GPU reranking using actual filter
cardinality and batch shape. Profiles are explicit deployment configuration;
the default remains the configured threshold policy when no profile is loaded.

`--filtered-slack N` (default 0) overfetches extra bitset-filtered CAGRA
candidates before FP32 reranking, with total overfetched output bounded at
1024 candidates. Calibrated batches use the profile's measured frontier and
overfetch budget. Kernel modules publish their ready state atomically;
concurrent cold callers wait for initialization instead of falling back.

FP32 CAGRA searches use the graph's FP32 distances directly, without CPU or
GPU rescoring. The worker converts squared L2 to Euclidean distance before
merging shards and returning SQL results. This is automatic for indexes built with `--dtype float32`, regardless of
`--gpu-rerank`. FP16 graphs also use their own distances unless
`--fp16-exact on` rescores them in FP32; without it, GPU reranking of
quantized gather candidates reads the FP16 dataset in place (see
[FP16 distances](../../docs/ann-cuvs-index.md#fp16-distances)). Optional
`--filtered-slack` still requests more graph candidates, which can change
traversal and recall; FP32 scores are used directly to select the final k.
Native scores can differ from PostgreSQL's distance arithmetic by FP32
rounding, especially near zero. Graph traversal remains approximate.
Statistics expose `cagra_fp32_queries`, `cagra_fp16_queries` and `graph_cpu_rerank_candidates`,
alongside `gpu_rerank_candidates`, to verify the selected path. A resident
`rerank_ready` buffer may still be used by scans of the same index.

The current v5 cost profile was calibrated for FP16 CAGRA with rescoring;
it does not price native FP32 CAGRA. Clear `ann.frozen_cost_profile` in sessions
using an FP32 rebuild of the same export generation until it has a suitable
profile. Worker dispatch already excludes FP32 graphs from that calibration.
See the [FP32 CAGRA guide](../../docs/ann-fp32-cagra.md) for setup and validation.
