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
