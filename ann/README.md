# Native C search worker and scheduler

Runtime code in this directory is C. Python is used only by test helpers and benchmark launchers under `test/ann/` and `benchmarks/ann/`. The [native C PostgreSQL AM](../docs/ann-native-index.md) provides an exact L2 baseline; an [opt-in exact CustomScan](../docs/ann-filtered-search.md) scores eligible vectors, optionally [pipelined to this worker](../docs/ann-worker-pipeline.md), and [loaded cuVS indexes](../docs/ann-cuvs-index.md) serve sharded multi-GPU ANN and hybrid search from SQL; the Python frozen-table implementation is retained solely as a test reference, not an application API.

## Build and test

A C11 compiler, POSIX threads, and Make are required for the standalone worker on Linux/macOS. PostgreSQL's main build is independent of this experimental worker. Windows is not a supported worker platform.

```sh
make -C ann
make -C ann check
# GPU hosts only: also build ann-worker-cuvs (pip SDK paths: see docs/ann-cuvs-index.md)
make -C ann USE_CUVS=1 CUVS_PREFIX=/path/to/libcuvs CUDA_HOME=/path/to/cuda
make -C ann check-cuvs CUVS_PREFIX=/path/to/libcuvs
```

`USE_RDMA=1` adds RDMA search frames over InfiniBand or RoCE verbs for networked workers ([guide](../docs/ann-networked.md#rdma-infiniband-roce)). `USE_CUVS` defaults to 0, so hosts without a GPU SDK build and test every CPU path. Loaded indexes are served by both builds. The CPU worker scans them exactly, which makes it the test reference; the cuVS worker searches CAGRA shards on all GPUs ([guide](../docs/ann-cuvs-index.md)).

`make` builds the C executable and `libannbatch.a`, without Python or a GPU SDK. `make check` uses Python to drive C protocol and scheduler tests. Live PostgreSQL and GPU tests are explicitly gated.

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

Link with `-Iann -Lann -lannbatch -pthread -lm`. The scheduler owns a canonical index copy by default. The worker uses `AnnBatchCreateBorrowed()` to share its immutable index; it closes the scheduler before unloading or replacing that index. Failed replacement builds preserve the old generation.

Admission returns `ANN_OVERLOADED` without queueing when request count, bytes, or estimated work exceed capacity. The C byte charge includes the request structure, query, filter, and reserved result array. It excludes the resident index, allocator overhead, bounded scheduler metadata, input wire frames, and results retained by callers after completion. Work is estimated as eligible count times dimension. Limits apply to queued and running work; timed-out running requests retain reservations until execution finishes.

Compatible jobs share metric, effective k, filter presence, and a coarse selectivity class. The oldest request anchors selection. The dispatcher groups jobs up to row/byte/work limits or the collection delay. CPU distance evaluation remains serial; there is no claimed GPU kernel batching or automatic cross-process aggregation. CPU worker search frames use this C scheduler directly. External protocol frames are still read serially.

A monotonic-clock watcher expires queued and running requests. Queued cancellation removes the job; running cancellation is not allowed. `AnnBatchClose()` cancels queued work, marks unfinished results canceled, waits for backend work to finish, and releases the index. It does not forcibly terminate a C thread or preempt a GPU kernel. Request handles remain valid until `AnnRequestRelease()`, even after service close. Caller-held results are released with their handle.

Lifecycle rules: coordinate producer threads before closing the service; do not use a batcher handle after `AnnBatchClose()`. Do not release a request handle while another caller is using that same handle. Backend descriptors must outlive the service. Worker threads must never call PostgreSQL backend APIs; future backend integration uses PostgreSQL's memory, lock, interrupt, and resource mechanisms in the backend process.

## Protocol

The C worker uses version 1, 32-byte little-endian headers: magic `PVAN`, u16 version, u16 operation, u64 request ID, u64 generation, u32 payload size, u32 status. Frames are capped at 64 MiB. Generation zero is reserved, and successful builds increase generation monotonically per process/session.

| Operation | Request | Success result |
| --- | --- | --- |
| 0 capabilities | Empty | JSON describing compiled backend capabilities |
| 1 build | u32 rows, dimensions; row-major FP32 vectors | Empty |
| 2 search | u32 queries, dimensions, k, metric, filter; vectors; optional sparse allowlists | Per query: u32 count and `(u32 ordinal, FP64 distance)` records |
| 3 unload | Empty | Empty |
| 4 export | Empty | Canonical build payload, not a graph artifact |
| 5 diagnostic probe | Bounded bytes | Empty; never a search result |
| 6 index info | Empty | u32 rows and dimensions |
| 7–10 | Gathered and shared-memory search for the [pipelined CustomScan](../docs/ann-worker-pipeline.md) (`--listen` mode) | |
| 11 load index | u32 name length, name, directory path | Text `rows N dimensions D shards S dtype T source NAME` |
| 12 index search | u32 name length, name, u32 dimensions, k, filter count (UINT32_MAX = none); FP32 query; i64 keys | u32 count, u32 route (0 graph, 1 filtered graph, 2 exact), then `(i64 key, f32 distance)` per hit |
| 13–15 | Network auth, packed-filter index search, index info ([networked guide](../docs/ann-networked.md)) | |
| — | Indexes built with `--quantize nvfp4\|fp8` report `quant KIND quant_scan on\|off` in load and index-info replies ([Blackwell module](../docs/ann-blackwell.md)) | |
| 8 stats | | JSON also carries `groups`, `grouped_jobs` (same-filter searches scanned together) and `held` (lane pickups that waited for a filter's lanes to free up); see [same-filter grouping](../docs/ann-gpu-filters.md#same-filter-grouping) | |
| 17–21 | Attribute search, bitmap build, bitmap search, bitmap cache put, cached bitmap search ([GPU-side filters](../docs/ann-gpu-filters.md)) | |
| 25 | Search several loaded indexes (parts of a collection on this node) with one program and query: u32 parts, per part (name, u64 generation), then operation 22's fields after the name (no ordinals flag; with the attributes flag every part must have the same attributes). Each part runs as an operation 22 job; the reply is the merged top k as operation 22's, eligible rows summed ([collections](../docs/ann-collections.md#how-a-search-runs)) | |
| 26 | Unload a loaded index: u32 name length, name. The name is free at once (another index can be loaded under it); the index is freed once the searches using it finish. Stale when it isn't loaded | |
| 27 | A loaded index's dictionary for a text attribute: u32 name length, name, u32 attribute length, attribute | u32 count, count + 1 u32 offsets, the strings' bytes, then u32 1 and count i32 collection codes if remapped, else u32 0 |
| 28 | Map a text attribute's codes to a collection's: the same payload, then u32 count and count i32 codes, one per dictionary string. The node rewrites the attribute's column and postings; done once, before predicates are cached (the same map again is accepted). Operations 22 and 25 with flag 4 answer stale for an index whose text attributes aren't mapped | |
| 29 | Delete rows by key (tombstones): u32 name length, name, u32 count, i64 keys. Keys the index doesn't hold are ignored. Deleted rows leave predicate results (cached ones are dropped), unfiltered searches over-fetch past them, and every route's hits are checked; they stay until the index is unloaded | u64 rows newly deleted, u64 rows deleted in all |
| 30 | Query matrix: u32 name length, name, u32 dimensions, k, flags, program length, query count (1–64), predicate program, row-major FP32 queries. One generation and predicate serve every query. Flag 1 returns ordinals; flag 8 retains tombstoned candidates so the coordinator can apply its own snapshot-visible tombstones. Other flags are rejected | u32 query count; per query u32 hit count, u32 route, u64 eligible, then (i64 key or u64 ordinal, f32 distance) per hit in distance order |
| 31 | Routing statistics for one loaded index: UTF-8 index name, without a length prefix | JSON policy, slack, replica/device bindings, active reservations, started/finished batches, queries, migrations, failed batches, reserved nanoseconds, and estimated device batch time. Requires a backend with routing statistics; control requests use the ordinary socket or TCP connection |
| 22–23 | Search by predicate program and evaluate a predicate program ([filter.h](filter.h)), cached under the program's hash ([frozen indexes](../docs/ann-frozen-index.md#worker-operations-and-caching)); [collections](../docs/ann-collections.md) send operation 22 to one replica of every part at once and merge the replies | |

Metric values: 0 L2, 1 negative inner product, 2 cosine. Filter values: 0 none, 1 shared IDs, 2 IDs per query. Allowlists are u32 count followed by strictly increasing unique u32 ordinals; an empty list differs from no filter. Capability masks use `1 << enum_value`. The CPU backend uses FP32 input and double accumulation.

Statuses 0–5 are success, invalid, unsupported, stale, allocation failure, and backend failure. The C scheduling API adds 6 pending, 7 overloaded, 8 timed out, 9 canceled, and 10 closed; loaded indexes add 11 missing (a cached filter id) and 12 generation (the request's header names another generation of the index than the one loaded). Search errors have empty protocol payloads. The PostgreSQL client enforces `ann.worker_timeout_ms`, and server mode sets a 10 s reply write timeout; timeouts in the Python test client are test infrastructure, not a shipped C client feature.

## Test references and benchmarks

[test/ann/support/](../test/ann/support/) contains Python-only protocol clients and a frozen-table SQL oracle. Snapshot checksum/save/load helpers there are test conveniences; production canonical export/import is implemented in C, while a durable C artifact-file format remains pending.

```sh
python3 -m pip install -r test/ann/requirements-postgres.txt
ANN_PG_DSN='dbname=your_test_database' make -C ann check-postgres
python3 benchmarks/ann/postgres.py --rows 10000 --dimensions 32 > /tmp/ann-baseline.sql
psql -X -v ON_ERROR_STOP=1 -f /tmp/ann-baseline.sql
python3 benchmarks/ann/search.py --mode shared --batch 16
python3 benchmarks/ann/batching.py --requests 1000 --sequential
python3 benchmarks/ann/batching.py --requests 1000 --pending 1024 --batch-rows 64
```

The batching launcher runs [a C benchmark](../benchmarks/ann/batch.c); search and scheduling execute entirely in C. It reports successful/offered/rejected counts, elapsed time, and batch occupancy. The test-client search benchmark measures IPC and uses an independent distance oracle. Neither benchmark includes SQL eligibility or establishes production QPS. The PostgreSQL fixtures create temporary or uniquely named test objects and require the corresponding database permissions.

C tests cover batching/filter isolation, input copying, budgets, queued/running deadlines, concurrent producers, early handle release, shutdown during execution, backend errors, and result lifetime after close. Python protocol tests independently validate metrics and filters. Sanitizer checks:

```sh
make -C ann clean
make -C ann CFLAGS='-O1 -g -std=c11 -Wall -Wextra -Werror -pedantic -fsanitize=address,undefined -fno-omit-frame-pointer' check
```

## cuVS integration

`cuvs.c` is compiled and validated against cuVS 26.10 on NVIDIA B200 GPUs (`make -C ann check-cuvs`):

- **Gathered search** serves the [pipelined CustomScan](../docs/ann-worker-pipeline.md) with exact GPU brute force.
- **Loaded indexes** serve sharded CAGRA search across all GPUs, with bitset filters and an exact route for selective filters ([guide](../docs/ann-cuvs-index.md)).
- **GPU failures and memory.** A search that fails in the backend makes the worker check every device for a lost context (a sticky CUDA error: illegal address, launch failure, ECC, assert and the like). A lost context fails every later call and keeps the process's device and pinned memory, so the worker logs the device and error and exits with status 70 without running exit handlers: the driver takes the memory back, clients see the connection close and fail over to other replicas, and a supervisor (systemd `Restart=always`) starts it again. `ANN_MOCK_GPU_LOST_AFTER=N` makes the `mock-gpu` backend lose its device after N submissions, for tests. Unloading an index (operation 26) frees its shards once their searches finish; with the default RMM pool the freed device memory stays with the worker for later loads, and `--device-allocator plain` returns it to the device ([trade-off](../docs/ann-cuvs-index.md#b200-throughput-options)).

The stdin-mode resident CAGRA build (operations 1–4 with `--backend cuvs`) is still the older unvalidated draft. Use loaded indexes instead. hipVS and NVFP4 adapters are not implemented: cuVS 26.10 CAGRA supports FP32, FP16, int8 and uint8 datasets, and no FP4 or FP8 path.

C PostgreSQL worker integration is implemented: `--listen PATH` serves pipelined gathered searches (operations 7–10) to `AnnFilteredSearch`, with a `mock-gpu` backend standing in for accelerators ([guide](../docs/ann-worker-pipeline.md)). Sustained-load validation remains. Python remains test/benchmark tooling only.

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
[the k=500 profile](../docs/ann-k500-profile.md).

`--cost-profile PATH` enables the optional [measured cost model](../docs/ann-cost-model.md).
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
[FP16 distances](../docs/ann-cuvs-index.md#fp16-distances)). Optional
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
See the [FP32 CAGRA guide](../docs/ann-fp32-cagra.md) for setup and validation.
