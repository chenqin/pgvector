# CPU and GPU profiling of frozen k=500 searches

This follow-up profiles the [10M × 128 B200 workload](ann-k500-stress.md)
after the visibility-map pin improvement. The main comparison uses eight GPUs,
32 worker lanes, batches up to 32, 256 SQL clients, generic prepared plans,
`ann.frozen_setup_cost = 1000`, and RAM-backed data. All 500 IDs are returned
per search. Narrow filters use GPU gather search; the 10% filter uses CAGRA.

## Methods and limits

* GNU gprofng timer sampling at 10 ms collected PostgreSQL CPU stacks while
  running the 10% workload for 20 seconds. The private test server was briefly
  restarted under the profiler with descendant collection, then restored to
  normal startup. Hardware perf sampling is blocked by the host's paranoid
  level of 4; no host security settings were changed.
* Nsight Systems 2026.3.2 traced CUDA APIs, kernels and transfers on a separate
  worker loaded with the same index and filters. Each filter had about six
  seconds of CUDA capture inside an 18-second load run. CPU and context-switch
  sampling in Nsight were disabled. GPU busy fractions below are unions of
  kernel intervals per device divided by the global kernel capture window,
  rather than sums of overlapping kernel durations or SM occupancy.
* The initial worker CPU profile ran alongside Nsight and therefore includes
  CUPTI overhead. A separate follow-up gprofng worker profile after the changes
  ran without Nsight, with 15 seconds of each filter. CPU profiles include
  startup/load work and are diagnostic, not throughput measurements.
* Throughput measurements ran separately without either profiler. Main SQL
  runs lasted 60 seconds per filter; repeats lasted 30 seconds. Native-worker
  controls lasted 20 seconds per filter. The native control uses the same
  filter/query distributions but different query/filter pairings, bypasses SQL
  and tuple visibility processing, and validates full 500-hit replies.

Raw captures, commands and reports are retained on the GPU node under
`~/pgvector-test.tKp5o0/profile-next/`. The temporary profiling workers and
catalog registrations were removed afterward; the normal instance remains up.

## What consumed the time

PostgreSQL's CPU profile attributed about 68.8% of CPU time to
`AnnArrayReadRange`, including about 37.7% at its copy call and 20.6% in buffer
reads. `PinBuffer` consumed 9.3% exclusive CPU time, `LWLockAttemptLock` 8.2%,
and `AnnCheckPage` 6.4%. These percentages overlap when inclusive and exclusive
measurements are mixed; they must not be added. Immutable vector materialization
and mutable TID buffer access remain significant after VM pin reuse.

GPU tracing showed different costs for different selectivities:

| Filter | Main kernel work | Share of summed kernel time | Kernel busy fraction across GPUs |
| --- | --- | ---: | ---: |
| 0.1% | Radix top-k, one-block path | 97.1% | 34.5–35.6% |
| 1% | Radix top-k passes | 79.0% | 15.9–16.9% |
| 10% | CAGRA `search_multi_cta` | 94.6% | 67.1–76.6% |

These are pre-worker-optimization captures. At 0.1%, the selection kernel
averaged about 89.6 µs while quantized scoring averaged 2.7 µs. At 1%, selection
dominated scoring as well. This identifies k=500 selection as a GPU optimization
target, but the low GPU busy fractions also showed that host work limited
submission. It would be misleading to call either narrow case simply GPU-bound.

The worker CPU profile and annotated disassembly exposed two avoidable costs:

1. Candidate validation checked every selected position against every previous
   position. The quantized path can select k plus 256 slack candidates: 756
   positions mean 285,390 pair comparisons per shard. The inner loop alone
   accounted for roughly 13% of the initial worker CPU samples.
2. Each selected key was binary-searched in the full shard's key array even
   when the cached filter already supplied its matching global ordinal.

CUDA memcpy APIs accounted for much of traced API time, while actual GPU
transfer durations were small relative to kernels. That is a host/API and
synchronization investigation target, not evidence that PCIe bandwidth was
saturated; tracing overhead also affects these API durations.

## Implemented changes and throughput

`AnnIndexPositionsDistinct` now uses an expected-linear-time hash set for larger
candidate lists, bounded stack scratch for normal k values, and the original
comparison loop for small lists or allocation failure. It preserves candidate
order, ignores negative padding positions, and keeps the existing duplicate
repair behavior. Cached-filter searches now read the supplied ordinal directly,
retaining shard-bound and key checks; requests without cached ordinals still
use binary search. Neither change adjusts search parameters or reranking.

| Filter | Before, SQL QPS | After, SQL QPS | Gain | After p99 ms | After native-worker QPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.1% | 33,276 | 63,707 | 91.5% | 6.12 | 89,531 |
| 1% | 29,507 | 49,998 | 69.4% | 7.92 | 73,978 |
| 10% | 65,053 | 64,869 | −0.3% | 6.73 | 94,968 |

The before figures include VM pin reuse. Narrow before runs lasted 30 seconds;
the broad before figure is from a 60-second run. Thirty-second repeats after
the worker changes reached 64,982 QPS and 50,480 QPS at 0.1% and 1% respectively.
All 14,154,495 SQL transactions across the main after runs and repeats succeeded.
The direct controls reported zero errors. Monitoring reported no errors and
zero sampled physical reads.

### More lanes after removing the CPU work

With the optimized worker, 64 lanes improved all three filters in 30-second
trials, so the configuration was confirmed with 60-second runs:

| Filter | 32 lanes SQL QPS | 64 lanes SQL QPS | 64-lane SQL p99 ms | 64 lanes native-worker QPS |
| --- | ---: | ---: | ---: | ---: |
| 0.1% | 63,707 | 97,131 | 4.12 | 154,669 |
| 1% | 49,998 | 80,059 | 5.22 | 130,477 |
| 10% | 64,869 | 70,045 | 5.84 | 94,513 |

This is a configuration gain on top of the code changes, not a change in
search accuracy or route. Sixty-four lanes are retained on the test node.
The cost is more CPU: PostgreSQL used about 112 and 96 cores on the narrow
filters, with about 65 worker cores, versus 53/45 PostgreSQL cores and 33 worker
cores with 32 lanes. The 32-lane setting remains more CPU-efficient. All other
worker settings, k, clients, query distributions and result sizes stayed the
same. These observations do not establish a benefit from raising lanes further.

All 22,174,724 SQL transactions in the 64-lane trials and sustained runs
succeeded, with no monitor errors and zero sampled physical reads. Native
controls reported no errors. At this lane count, SQL reaches about 63%, 61%
and 74% of native-worker throughput, respectively. This is consistent with
significant PostgreSQL execution/memory overhead and host contention, but the
gap also includes query-vector fetching, protocol and worker batch occupancy;
it is not a direct attribution of all lost throughput to one CPU function.
The broad SQL run used about 78 PostgreSQL and 56 worker cores. Repeated
accuracy checks at 64 lanes returned 500 unique hits in all nine samples:
recall was 1.0 for the six narrow samples and 0.992–0.996 for the broad samples.

Reproduce the retained worker by changing only `--lanes 32` to `--lanes 64`
in the adaptive command from the stress report; keep `--max-batch 32`,
`--filtered-itopk 1024`, `--filtered-width 4`, `--exact-limit 0`,
`--gather-limit 4000000`, `--bitset-gather-limit 500000` and
`--group-gather-limit 0`. Load and precompute the same 210 predicates before
timing. Run the main SQL comparison with:

```sh
python benchmarks/ann/frozen_stress.py --socket "$SOCKET" --worker-pid "$WORKER_PID" \
  --output results --label worker-opt-lanes64-sustained --clients 256 \
  --seconds 60 --k 500 --setup-cost 1000 --plan-cache-mode force_generic_plan
```

The follow-up worker CPU profile reduced distinct-position checking to 0.53%
exclusive CPU time. Filtered-search postprocessing and FP32-vector reranking
remain substantial; the disassembly's largest hotspot is the reranking loop's
vector load/conversion. `cuvsStreamSync` accounted for about 19% inclusive CPU
time, and `sched_yield` 7.8% exclusive, so CUDA completion polling also deserves
attention. These categories overlap.

Three exact SQL comparisons per filter returned 500 unique hits. Recall@500
was 1.0 for all six narrow samples and 0.994–0.996 for the three broad samples.
This is a small accuracy check, not a recall guarantee. Search parameters and
the FP32 reranking calculation were unchanged.

One experiment was rejected: pointing tuple vectors directly into mmap rather
than copying them produced 64,251 QPS for the broad filter versus the 65,053
baseline. It showed no throughput benefit and was reverted. A copy hotspot can
reflect memory-read latency that moves into the consuming function when the
copy is removed.

## Remaining opportunities

1. **Reduce PostgreSQL's per-hit memory and buffer work.** With 32 lanes, SQL
   reaches about 68–71% of the refreshed native-worker rates. Test lookahead
   prefetch of immutable vectors and page-local result materialization, retaining
   distance order, TID locking and current visibility checks. An explicit SQL
   batch sharing one predicate and snapshot can also amortize planning and IPC,
   but batching alone does not remove the dominant per-result reads and copies.
   Count vector searches, not just statements.
2. **Reduce worker reranking and synchronization cost.** Evaluate vectorized or
   GPU-batched FP32 reranking with tie/accuracy validation. Test persistent
   pinned query staging against the current pageable query uploads, and compare
   blocking completion waits with the observed polling path. Blocking device
   scheduling is already enabled; setting it again is not an optimization.
3. **Optimize top-k for k=500.** Narrow searches spend most device time selecting
   results rather than scoring vectors. A suitable batched or hierarchical
   selection implementation is a more direct target than a faster score kernel.
   The existing grouped Blackwell top-32 path cannot serve k=500.
4. **Tune broad CAGRA only with recall checks.** Traversal dominates broad-filter
   GPU time. Internal top-k, search width and batching can trade recall or
   latency for throughput; the measurements above retain itopk 1024 and width 4.

The code passed 15 GPU tests, all 27 frozen tests, all 16 SQL regressions and
eight native unit scenarios. The new selection check covers padding, duplicates,
zero and maximum positions, stack/heap sizes, input preservation and 1,000
randomized comparisons against the reference implementation. AddressSanitizer
and UndefinedBehaviorSanitizer also passed that scenario on macOS.

Full metrics, kernel summaries, CPU reports and accuracy samples are in the
[result artifact](../benchmarks/ann/results/frozen-k500-cpu-gpu-profile.json).

For reproduction, the key collector commands were:

```sh
# Launch the private test server under timer profiling, then drive it separately.
gprofng collect app -p 10 -a off -F on -o postgres-cpu.er postgres -D "$PGDATA"
gprofng display text -limit 60 -functions postgres-cpu.er
gprofng display text -limit 100 -calltree postgres-cpu.er

# Start a separate worker session, load/precompute the same index, then capture.
nsys launch --session-new=frozen-gpu --trace=cuda,nvtx ann/ann-worker-cuvs ...
nsys start --session=frozen-gpu --sample=none --cpuctxsw=none --output=gpu-capture
# Run the workload during the capture, then stop and summarize.
nsys stop --session=frozen-gpu
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum gpu-capture.nsys-rep
```

## Prefetch follow-up, 2026-11-05

Two read-side changes were retained after testing on the same 10M-row,
eight-B200 dataset with 64 lanes, batch up to 32, k=500 and 256 SQL clients:

* `ann.frozen_prefetch_distance = 4` hints immutable mapped vectors, keys and
  attributes four result rows ahead. `0` disables it; the allowed range is
  0–64. The scan still materializes rows in distance order and reads mutable
  TIDs and current visibility bits normally. With mmap disabled the hint is
  a no-op. Startup lookahead is bounded by the actual result count.
* Worker `--rerank-prefetch 8` hints host FP32 vectors eight candidates ahead
  during CAGRA and cached-ordinal gather reranking. `0` disables it; the range
  is 0–64. Candidate bounds are checked before issuing hints. The distance
  calculation, candidate count, repair checks and GPU selection are unchanged.

Four alternating 60-second broad-filter runs isolated the SQL hint with worker
prefetch disabled. Disabled controls reached 70,179 and 70,045 QPS; lookahead
four reached 71,516 and 71,613 QPS, approximately a 2.1% gain on average. The
initial 15-second sweep also checked lookaheads 1, 8 and 16, with similar broad
rates; four was retained as the smaller consistently successful lookahead.

The main comparison ran each filter for 60 seconds with both hints disabled,
then with SQL lookahead four and worker lookahead eight:

| Filter | Disabled QPS | Enabled QPS | Gain | Disabled p99 ms | Enabled p99 ms | Enabled native-worker QPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1% | 96,267 | 101,686 | 5.6% | 4.18 | 4.43 | 170,133 |
| 1% | 78,234 | 97,611 | 24.8% | 5.37 | 4.37 | 170,692 |
| 10% | 69,985 | 74,482 | 6.4% | 5.83 | 5.13 | 96,426 |

The 0.1% tail latency increased by 0.24 ms despite higher throughput and lower
median latency; these changes do not improve every latency percentile. Both
controls are exposed so latency-sensitive workloads can disable either hint.
The main comparison completed 31,063,512 searches without failure. Across all
22 SQL tuning and comparison runs, 61,707,118 transactions succeeded, with no
monitor errors and zero sampled physical read bytes. Native controls ran for
20 seconds per filter and reported no errors. The main comparison is one pass
per setting/filter, so the observed differences are not a production guarantee.

All nine recall samples returned 500 unique hits. Recall@500 was 1.0 on the six
narrow samples and 0.994–0.998 on the broad samples. Tests passed: 15 GPU cases,
28 frozen cases on Linux, 27 frozen cases plus one GPU skip on macOS, 16 SQL
regressions and eight native unit scenarios. The new boundary case exercises
lookahead 0/1/4/64, empty and short results, and a fresh mmap-disabled backend.
Both platforms were clean-built; pgindent and diff checks passed.

The installed cuVS C++ selection API was also probed with 10,000 input scores
and k=756 (500 plus the existing 256 quantization slack). Auto and both
11-bit radix variants took about 59.3 µs in that standalone microtest; 8-bit
radix took 163.1 µs. The installed warp variants rejected k=756; WarpAuto
reported a maximum k of 256. Supported variants passed unique-position and
selection-threshold checks. No selector change was retained, and no C++ shim
or new dependency was added to the production worker. This microtest used
synthetic scores and does not establish performance on other distributions.
Custom generalized-k selection and GPU-batched reranking remain future work.

To reproduce the main comparison, start/load/precompute the worker twice with
the same adaptive settings above, using `--rerank-prefetch 0` for the control
and `--rerank-prefetch 8` for the enabled run:

```sh
# Control worker: --rerank-prefetch 0
python benchmarks/ann/frozen_stress.py --socket "$SOCKET" --worker-pid "$WORKER_PID" \
  --output results --label prefetch-control60 --clients 256 --seconds 60 \
  --k 500 --prefetch-distance 0
# Enabled worker: --rerank-prefetch 8
python benchmarks/ann/frozen_stress.py --socket "$SOCKET" --worker-pid "$WORKER_PID" \
  --output results --label prefetch-enabled60 --clients 256 --seconds 60 \
  --k 500 --prefetch-distance 4
```

The retained defaults are SQL lookahead four and worker lookahead eight. Raw
logs and selector probe sources remain in `~/pgvector-test.tKp5o0/prefetch-test/`
on the node. Full measurements, source hashes and accuracy checks are in the
[prefetch result artifact](../benchmarks/ann/results/frozen-k500-prefetch.json).

## Multi-vector SQL follow-up, 2026-11-05

`ann_frozen_search_batch` now sends 1–64 vectors and one shared predicate in
one worker request, with separate top-k lists. PostgreSQL pins the source
index generation, checks SELECT permissions and RLS, and uses one statement
snapshot for visibility and soft deletions. Snapshot searches retain worker
tombstone candidates and use a separate, budgeted predicate-cache namespace.
A short result after visibility filtering is filled by exact SQL. This also
handles soft deletes committed by another session after a repeatable-read
snapshot was taken.

The first implementation repeatedly switched a single visibility-map buffer
pin. Retaining 32 pins, as the ordinary frozen route does, and removing an
unnecessary vector parse from benchmark query setup improved the sweep.
These two adjustments were measured together. Catalog lookup strings now
live in the caller's memory context instead of accumulating in the function
context across prepared-statement executions.

Eight B200s, 64 worker lanes, 64 PostgreSQL clients, 32 pgbench threads,
10M × 128 dimensions, k=500, 15 seconds per point, all 500 keys returned per
vector; no failures or physical disk reads:

| SQL path | 0.1% vectors/sec | 1% vectors/sec | 10% vectors/sec | 10% transaction p99 |
|---|---:|---:|---:|---:|
| Ordinary frozen scan | 52,075 | 50,214 | 34,328 | 3.12 ms |
| Batch of 1 | 54,725 | 51,791 | 35,436 | 2.50 ms |
| Batch of 4 | 93,028 | 89,027 | 62,489 | 5.47 ms |
| Batch of 8 | 103,213 | 99,837 | 76,629 | 9.64 ms |
| Batch of 16 | 109,102 | 104,357 | 86,600 | 16.88 ms |

Each batch transaction includes all its vectors and results; its latency is
not a single-vector latency. `qps` in the result files counts transactions,
while `vector_qps` counts searches. For the 10% filter, the 16-vector run
used about 40.6 PostgreSQL cores and 46.8 worker cores, versus 26.6 and 30.4
for the 64-client ordinary scan. PostgreSQL CPU per search fell from about
775 to 469 microseconds. GPU utilization remained high in both runs; that
metric alone does not identify a compute or bandwidth limit.

These are short comparisons at 64 clients. The earlier 256-client scan
measurements in this document are a different concurrency point; longer
saturation sweeps remain pending. The raw initial and tuned results, CPU,
latency and GPU samples are in
[`frozen-k500-matrix.json`](../benchmarks/ann/results/frozen-k500-matrix.json).
The measurements precede final checks restricting the snapshot flag to
matrix requests and handling short approximate results; neither changes
the normal full-result benchmark path.

Example against the private GPU-node deployment after sourcing its env.sh:

```sh
python benchmarks/ann/frozen_stress.py \
  --socket "$task_root/run/k500.sock" \
  --index-name frozen_stress --batch-queries 8 \
  --clients 64 --threads 32 --seconds 60 --k 500 \
  --output "$task_root/replication-next/batch-confirm" --label matrix8
```

This completes a first implementation of the SQL batch path. Page-local
materialization still regresses and remains disabled by default. The
load-aware routing follow-up below implements and benchmarks replica choice;
generalized k=500 GPU selection/reranking remains unfinished. Existing k=500 gather jobs continue using the existing selector.

Validation: all 16 cuVS tests passed (including the new k=500 query-matrix
test), all 30 frozen tests passed across CPU/GPU runs, and seven worker
predicate-cache tests passed. A final build with the leased-generation check
also passed the GPU k=500 matrix test. Its five-second 10% smoke run returned
85,521 vectors/sec with zero failures; it is recorded separately from the
15-second sweep. Local fresh-install and 0.9.0-to-0.9.3 upgrade checks expose
the new SQL function. The private GPU-node worker remains running.

## Load-aware replica routing follow-up, 2026-11-05

The worker now supports `--replica-routing home|load` and
`--replica-load-slack N` (default 1). The load policy compares active batch
reservations weighted by a 1/16 moving average of their service time, keeping
a lane on its home GPU when estimates are close. Counters are shared across
loaded indexes, include unreplicated multi-GPU searches, and use C11 atomics
without the scheduler mutex. Service time includes host reranking; it is not
a kernel timing or a direct GPU utilization measurement.

Operation 31 returns per-index replica counts and timing; the benchmark
records snapshots before and after each measurement when `--index-name` is
supplied. They include device bindings, active reservations, started/finished
batches, query counts, migrations, failures, total reserved time, and the
current service-time estimate. This made a startup problem visible: the
initial load policy retained 173 ms and 39 ms estimates on two GPUs and sent
neither any work during a scan measurement. The corrected policy ages idle
estimates toward 100 microseconds, halving the excess each second, and uses
the decayed estimate when a returning GPU is sampled. It recovered all eight
replicas in the follow-up runs.

Eight replicated B200 GPUs, 64 worker lanes, 10M × 128 dimensions, k=500,
10% filter, all 500 keys returned per vector; 15-second measurements after
warm-up, no failures:

| Condition | Workload | Home vectors/sec | Load vectors/sec |
|---|---|---:|---:|
| Uniform | Scan, 256 clients | 72,797 | 71,991 |
| Uniform | Batch of 16, 64 clients | 89,176 | 86,912 |
| GPU 0 contended | Scan, 256 clients | 69,786 | 69,166 |
| GPU 0 contended | Batch of 16, 64 clients | 85,528 | 84,410 |

Contention used an independent 4096 × 4096 FP16 GEMM loop on GPU 0 for
55 seconds, with the measurements taken while it was running. In the batch
case, load routing sent GPU 0 about 52.7k queries against home's 93.6k,
shifting work to the other seven replicas. It did not improve overall
throughput. The global queue already allows faster home lanes to consume
more work, which is consistent with this result. Scan p99 under contention
was 6.85 ms for home and 6.53 ms for load in this short sample; longer repeats
would be needed to establish a latency advantage.

Both policies used all eight replicas again in a ten-second recovery scan
after the contender exited. Waiting for it to exit leaves a short idle
interval, so this is not an immediate-recovery latency measurement. All
reservations were released, started/finished counts matched, and replica
failure counters were zero at the sampled endpoints. Raw measurements and
routing snapshots are in
[`frozen-k500-routing.json`](../benchmarks/ann/results/frozen-k500-routing.json).
The contention generator is
[`gpu_contention.py`](../benchmarks/ann/gpu_contention.py).

Home remains the default and the private GPU-node deployment was restored
to it. Load-aware routing is available for deployments where this heuristic
helps. The core test exercises 1.6 million concurrent reservations,
non-power-of-two GPU counts, overloaded homes, slow-device avoidance and
idle recovery. All eleven GPU index tests, including a two-GPU k=500 replica
test, passed with the corrected policy. Thirty frozen tests also passed
locally (one GPU-only case skipped), along with eight server tests and seven
predicate-cache tests. Generalized k=500 GPU selection/reranking is implemented in the follow-up
below; the page-local materializer still requires tuning.

## Generalized GPU selection/reranking, 2026-11-05

The worker now has a C++20 bridge to cuVS matrix selection, a Blackwell
score-matrix entry point, bounded query tiling, and optional GPU reranking
from the original FP32 vectors with double-precision accumulation. The
large-k path keeps candidates on device through a second selection carrying
row IDs. CAGRA searches can use the same resident vectors for reranking;
CPU reranking remains available when the copy is disabled or unavailable.
The switches are `--matrix-gather on`, `--gpu-rerank on`,
`--group-gather-limit ROWS` and `--matrix-max-mb N`.

Eight B200s, 64 worker lanes, 10M × 128 dimensions, k=500, 15 seconds per
point. The SQL rows below use 64 clients and 16 vectors per transaction,
returning all 500 keys per vector:

| Configuration | 0.1% vectors/sec | 1% vectors/sec | 10% vectors/sec |
|---|---:|---:|---:|
| Control | 108,127 | 105,780 | 88,848 |
| GPU reranking, graph for broad filters | 107,934 | 103,918 | 101,491 |
| Matrix gather up to 4M eligible rows + GPU reranking | 137,008 | 130,513 | 96,702 |

For the 10% filter at 256 clients, ordinary SQL scans reached 72,695, 76,501
and 80,170 QPS respectively. Native worker throughput was 96,331, 96,977
and 114,389 QPS. All measured points completed without errors. Resident
FP32 copies added 40.96 GB over the eight GPUs (5.12 GB per replica).
These measurements favor matrix gathers for the narrower filters and
CAGRA plus GPU reranking for the broad SQL batch workload. The combined
500k-row cutoff was subsequently measured below. Raw data, CPU/GPU samples and path counters are in
[`frozen-k500-gpu-selection.json`](../benchmarks/ann/results/frozen-k500-gpu-selection.json).

The executed 13-test GPU index suite passed, including matrix sizes 1, 3,
16 and 64, all four dataset formats, CPU/device reranking, a bounded-score
matrix case, and CAGRA rerank precision/filter checks. The expanded candidate-list tests for every format also passed in the
combined validation run.

The materializer also has a third revision: it caches only TIDs and their
visibility proof in page order, then reads immutable values when emitting
rows. This reuses the ordinary vector scratch and per-tuple text context,
removing per-row vector allocations and complete slot copies. The local
frozen suite passed with a 64-row window (30 tests, one GPU-only skip).
It remains disabled by default until the GPU-node sweep confirms a gain.

The combined 500k-row cutoff completed its first sweep successfully:

| Filter | SQL batch of 16, 64 clients | Ordinary scan, 256 clients | Native worker, 256 clients |
|---|---:|---:|---:|
| 0.1% | 138,065 vectors/sec | 125,332 QPS | 261,366 QPS |
| 1% | 131,528 vectors/sec | 113,033 QPS | 260,914 QPS |
| 10% | 101,509 vectors/sec | 76,275 QPS | 96,931 QPS |

These are 15-second points with no failures. The native protocol sends one
vector per request; SQL batches share a predicate across 16 vectors, so the
worker's batching opportunities differ. Nine sampled queries returned 500
unique rows each. Recall@500 was 1.0 for the narrow filters and
0.990–0.998 for the 10% filter. The expanded 13-test GPU suite and all 30
frozen tests passed on the node before this run.

The third materializer revision still did not win: at 256 clients on the
10% filter, its 16/64/256-row windows reached 75,723/75,640/75,765 QPS,
against bookend controls of 76,399 and 76,469. It stays off. The remaining
materialization work targets the mutable-TID buffer accesses directly;
merely sorting a small window does not provide enough locality here.
The combined results, materializer sweep and selected configuration are
included in the GPU-selection JSON above. Longer confirmation runs and the
remaining materialization work are still pending.


### Mapped TID validation

The fourth materializer revision removes the per-result TID buffer pin and
content lock for unchanged frozen indexes. It validates a bounded window
of mapped TIDs and VM bits under one shared metapage lock, then emits values
in result order. VACUUM WAL-logs clearing the unchanged-TID marker before
invalidating any TID; after physical deletion, reads use live buffers until
REINDEX. No-op VACUUM preserves the fast path. Old backends must be restarted
before building indexes with the new marker. This is a bounded read barrier,
not an entirely lock-free PostgreSQL executor.

The 10M-row index was reindexed after restarting the private PostgreSQL
instance; its generation stayed unchanged and its GPU copy stayed loaded.
All 31 frozen tests, including the GPU case and cursor/VACUUM barrier test,
and all 12 filtered tests passed on the node. Local full ANN checks and all
16 upstream pg_regress cases passed. A local immediate-shutdown/restart
check after physical deletion and VACUUM recovered the cleared marker from
WAL and returned the same 500 live rows as the exact SQL oracle.

Initial 20-second measurements:

| Filter | Covering scan, 256 clients | SQL batch16, 64 clients |
|---|---:|---:|
| 0.1% | 139,163 QPS | 241,622 vectors/sec |
| 1% | 130,306 QPS | 228,787 vectors/sec |
| 10% | 83,940 QPS | 133,320 vectors/sec |

On the broad-filter sweep, controls at window 0 reached 75,670 and 75,257 QPS.
Window 64 reached 84,151 QPS and window 256 reached 84,192 QPS. Window 16 fell to
55,942 QPS, indicating that frequent access to the shared metadata barrier
can erase the gain. The selected default is 64. The sorting fallback from
earlier prototypes was removed; legacy indexes use direct buffered reads.

PostgreSQL CPU use on the broad case fell from about 84 cores to 54 cores,
while worker use stayed near 48 cores. SQL batch16 used about 42/45/25
PostgreSQL cores across the three filters, versus 54/55/41 before mapped
TID validation, while completing more vectors. GPU utilization samples were
92–100% during SQL batches, and physical read samples remained zero.
These are utilization samples, not new kernel-level Nsight attributions.
The ordinary scan still evaluates SQL projections and vector distances;
the SQL batch API avoids much of that per-row executor work.

Raw CPU/GPU samples, worker path counters and the initial sweep are in
[`frozen-k500-mapped-tids.json`](../benchmarks/ann/results/frozen-k500-mapped-tids.json).
Final-build confirmation completed with the selected 64-row default:

| Filter | Unbatched scan control | 64-row scan | SQL batch16, 64 clients | Batch p99 |
|---|---:|---:|---:|---:|
| 0.1% | 126,739 QPS | 141,125 QPS | 244,512 vectors/sec | 5.43 ms |
| 1% | 115,052 QPS | 134,249 QPS | 231,078 vectors/sec | 5.64 ms |
| 10% | 77,644 QPS | 85,577 QPS | 133,589 vectors/sec | 10.73 ms |

Optimized points lasted 60 seconds each; the initial scan controls lasted
30 seconds. A second 60-second broad control reached 78,230 QPS, followed by
another 60-second window 64 run at 85,563 QPS, a 9.4% gain. Batch latency covers
all 16 vectors. The confirmation completed 43,333,444 SQL transactions
(77,588,404 vectors).
Every point had zero query failures and zero monitor errors;
all physical-read samples were zero. The complete confirmation series is
included in the mapped-TID result artifact. The worker was then reloaded
from the final build with the same configuration and validated again.
Its executable hash matched the built binary; all eight replicas reported
GPU reranking ready and the PostgreSQL default was 64. A 15-second deployed
batch16 smoke reached 240,716 / 228,927 / 133,273 vectors/sec with zero failures.

To reproduce the optimized B200 policy, build with `USE_CUVS=1`, load the
same index on eight GPUs with replication, and use these worker options:

```sh
ann/ann-worker-cuvs --backend cuvs --listen "$SOCKET" \
  --lanes 64 --max-batch 32 --replicate on --itopk 512 \
  --filtered-itopk 1024 --filtered-algo multi --filtered-width 4 \
  --exact-limit 0 --gather-limit 4000000 --bitset-gather-limit 500000 \
  --group-gather-limit 500000 --gpu-rerank on --matrix-gather on \
  --memory-pool 40 --predicate-cache-mb 8192 --predicate-device-mb 16384 \
  --max-pending-mb 8192 --rerank-prefetch 8 --replica-routing home \
  --replica-load-slack 1
```

After loading and warming the same 210 predicates, run the SQL workloads:

```sh
python benchmarks/ann/frozen_stress.py --socket "$SOCKET" --worker-pid "$WORKER_PID" \
  --output results --label mapped-scan --clients 256 --seconds 60 --k 500 \
  --index-name frozen_stress --materialize-batch 64
python benchmarks/ann/frozen_stress.py --socket "$SOCKET" --worker-pid "$WORKER_PID" \
  --output results --label mapped-batch16 --clients 64 --seconds 60 --k 500 \
  --index-name frozen_stress --batch-queries 16
```

The table and query fixtures are the ones from the stress guide. The data
and index files reside on the node's RAM-backed filesystem. GPU reranking
adds 5.12 GB per replica for this dataset; its flags stay opt-in. The
64-row scan visibility window is the new PostgreSQL default. Home routing
stays the worker default because the measured adaptive policy did not
increase throughput, including under the tested single-GPU contention.
