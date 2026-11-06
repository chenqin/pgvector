# Throughput of 90%-rejection frozen searches at k = 500

Measured on 2026-11-06 on the [k=500 stress deployment](ann-k500-stress.md): one
host with eight B200 GPUs and 224 logical CPUs, PostgreSQL 17.6, cuVS 26.10, the
10M × 128 `frozen_stress` table (FP16 CAGRA graph with an NVFP4 copy, replicated
on all eight GPUs). The workload is the 10% filter, `category BETWEEN d*10 AND
d*10+9` with ten values of `d`, so 90% of rows are rejected. Every search returns
500 keys. pgbench runs on the same host as PostgreSQL and the worker.

The result: **780.4k vectors/s through PostgreSQL** with the multi-vector SQL
batch function on an immutable table, against 86.2k for the deployed
configuration's ordinary SQL scan (9.1×), and **177.3k QPS** for ordinary
single-vector SQL (2.1×). Mean recall@500 stayed at 0.9998 (lowest single query
0.994). The goal was at least 90%.

## Method

* An experimental worker ran on its own socket next to the deployed one, with the
  same index files and the same 210 precomputed predicates; sessions chose it with
  `ann.worker_socket`. `benchmarks/ann/frozen_stress.py` drove 30-second pgbench
  runs (the final one 60 s) with generic plans and the 64-row materialization window.
* Recall: `benchmarks/ann/frozen_recall.py` compares SQL results with exact FP64
  top 500s computed on a GPU from the index's original FP32 vectors. Batch runs check
  640 vectors through `ann_frozen_search_batch` in batches like the benchmark's (so
  the worker groups them as under load); single-vector runs check 40 queries.
* Profiles: Nsight Systems kernel summaries of the worker, gprofng samples of the
  worker's CPU, and gdb stack samples of PostgreSQL backends (`perf` is not
  permitted on the host; backends were made traceable with a session-preloaded
  module calling `prctl(PR_SET_PTRACER)`).
* `vector_qps` counts query vectors per second; for single-vector SQL it equals
  transactions per second. Raw run summaries are in
  [`benchmarks/ann/results/frozen-10pct-throughput/`](../benchmarks/ann/results/frozen-10pct-throughput/).

## Bottlenecks and changes, in order

| Step | Change | vectors/s | Limit found next |
| --- | --- | ---: | --- |
| Deployed, ordinary SQL, 256 clients | NVFP4 matrix gather (cost profile), 64 lanes | 86.2k | GPUs 91% busy; groups of 2.3 queries; NVFP4 scoring is per group |
| Ordinary SQL, 32 lanes, 512 clients | Fewer lanes, more clients: groups of 8.5 | 126.8k | PostgreSQL 108 cores |
| SQL batch of 16 vectors, 128 clients | `ann_frozen_search_batch`: groups of 32 | 169.7k | GPU: radix top-756 of 1M scores per query is 72% of kernel time |
| `--matrix-prune on` | Sampled cutoff, select from ~4k kept scores | 311.3k | GPU 84–92%; per-vector queue lock in the worker |
| Queue a batch's jobs together | One lock and one wake-up per batch request | 397.9k | PostgreSQL 120 cores: a tuple per hit |
| `ann_frozen_search_batch_arrays` | One row of arrays per vector | 480.1k (523.0k at 64 vectors) | Worker: per-job reference and accounting locks |
| Settle references per batch | Counted holds/releases, one lock per batch | 697.0k (726.2k at 96 lanes) | Host CPU: PostgreSQL ~143 + worker ~45 + pgbench of 224 cores |
| Final, 60 s | 96 lanes, 64-vector array batches, 448 clients | 745.8k | Host CPU; GPUs 79% busy |
| With the Matcha patches, `ann.frozen_immutable = on` | Batch function skips TID and visibility checks on an immutable table | **780.4k** | Per-lane pipeline: mid-pipeline sync, host result handling; GPUs 80%, ~57 host cores idle |

The graph route was no alternative here: forced FP16 CAGRA reached 85.8k with FP16
distances and 84.4k with FP32 reranking (both GPU-bound at 96%), mean recall 0.974,
and one query at 0.34. Reranking did not change recall without overfetch.

Options that did not help: `--filter-lanes 1 --active-lanes 16` (the per-filter
limit is multiplied by the eight copies), `--delay-us 200` (−5%), the RMM async
allocator instead of the pool (169.4k against 169.7k), 64-query matrix batches
before pruning, and passing kept counts to `select_k` as row lengths (no change:
its one-block radix kernel was bound by its launch shape, one block per query).

## What changed in the code

* **Pruned matrix selection** (`--matrix-prune`, `ann/cuvs.c`): after scoring a
  group's eligible rows, sample 8,192 scores per query, take the rank-th smallest
  as a cutoff (rank chosen so about four times the k + slack candidates pass, at
  least 32), compact the scores under it, and sort each query's kept list in one
  block's shared memory. A query keeping too few or too many falls back to the full
  radix select, so the candidates equal the full selection's. At 10% (1M rows) the
  cutoff keeps about 3,900 scores; at 1% (100k rows) about 3,000.
* **Grouped submission** (`ann/server.c`): a batch request's jobs are queued under
  one lock and one lane is woken per `--max-batch` of them, instead of a lock and a
  wake-up per vector, which also split groups across lanes.
* **Settled accounting** (`ann/server.c`, `ann/indexfile.c`): a lane returns a
  finished batch's pending bytes and drops its index and bitmap references once per
  batch with counted holds and releases; batch submission takes them once per
  request.
* **Array results** (`src/annindex.c`): `ann_frozen_search_batch_arrays` returns
  `(query_no, ids bigint[], distances real[])`, one row per vector, sharing the
  row-per-hit function's visibility checks and local refill.
* **FP16 rerank in place** ([FP16 distances](ann-cuvs-index.md#fp16-distances)):
  without `--fp16-exact`, quantized gathers on FP16 indexes rerank from the FP16
  dataset; at equal throughput (169.5k against 169.4k) it frees 5.12 GB per GPU.

## With the Matcha patches

The Matcha patch series (scan-only indexes, rescore modes, immutable frozen
tables, multi-valued attributes) was applied under this work. On the merged tree
the batch path measured 733.4k vectors/s with its scan route on and 735.9k with
`--scan-routes off` (both groups take the pruned matrix gather), and ordinary SQL
157.6k–163.1k across `--rescore fp32|off|fp16|fp8-gpu` and `--scan-routes off`:
PostgreSQL-bound, so the routing options do not matter for this workload.

`ann.frozen_immutable = on` (the table unchanged since its build, both guards
firing always) maps TID pages and drops the visibility checks and the recheck.
Ordinary SQL rose from 161.3k to 177.3k QPS (PostgreSQL 145 → 117 cores). The
batch function did not use it at first; it now skips TID reads and visibility
checks for an immutable table too (tombstones still apply), and reached 780.4k
vectors/s with PostgreSQL at 108 cores. More lanes (128) or clients (560) did not
add throughput.

## Formats and routes

The node also holds an FP32 CAGRA build of the same export (same generation), so
PostgreSQL's frozen route accepts it; an FP32 + NVFP4 variant links the FP16
index's NVFP4 files beside it (same vectors, same order). All runs use the merged
worker, `ann.frozen_immutable = on`, 10% filters and k=500. Gather is the pruned
matrix gather (final configuration); graph is filtered CAGRA (`--gather-limit 0
--bitset-gather-limit 0 --group-gather-limit 0`), FP16 or FP32 native distances.

| Index | Gather, 64-vector arrays | Graph, 64-vector arrays | Gather, single-vector SQL |
| --- | ---: | ---: | ---: |
| FP16 CAGRA + NVFP4 | 764.5k (0.9998 / 0.994) | 177.0k (0.968 / 0.138) | 176.0k (0.9999 / 0.998) |
| FP32 CAGRA, no quantized copy | 725.5k (1.0 / 1.0) | 174.1k (0.968 / 0.128) | 177.1k (1.0 / 1.0) |
| FP32 CAGRA + NVFP4 | 779.9k (0.9999 / 0.994) | 174.1k (0.968 / 0.132) | 171.5k (0.9999 / 0.998) |

vectors/s, with mean / lowest recall@500. FP16 and FP32 CAGRA graphs perform the
same on this route (1.7% apart, the same recall), and both are GPU-bound at 96%.
The FP32 gather scores its rows in place on the tensor cores and reranks in FP32:
exact recall for 5% less throughput than with the NVFP4 copy.

### Crossover by k, selectivity and batch shape

| Case | Gather | Graph | Recall, gather / graph (lowest query) |
| --- | ---: | ---: | --- |
| 10%, k=500, single-vector SQL | 177.1k | 136.9k | 0.998 / 0.338 |
| 10%, k=100, 64-vector arrays | 1,006.2k | 325.7k | 0.990 / 0.430 |
| 10%, k=100, single-vector SQL | 211.1k | 212.3k | 0.990 / 0.730 |
| 1%, k=500, 64-vector arrays | 928.4k | 25.4k | 0.998 / 0.890 |

With grouped queries the gather scan beats the graph from 1% to 10% matching rows,
at k=100 and k=500: the crossover lies above 10%. Only single-vector SQL at k=100
ties, where both routes are limited outside the GPUs. The graph's weakest queries
are far below 90% at 10% filters with these settings (filtered itopk 1024, width 4),
even when its mean exceeds 0.97.

### Sweep over matching rows and batch size (k=500)

20-second points, FP16 CAGRA + NVFP4, `ann.frozen_immutable = on`. b1 is
single-vector SQL (32 lanes, 512 clients); b16 and b32 are array batches (96
lanes, 256 and 320 clients). vectors/s, with the lowest single-query recall@500
(all means are at least 0.94).

| Matching rows | Gather b1 | Graph b1 | Gather b16 | Graph b16 | Gather b32 | Graph b32 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10k | 180.3k (0.998) | 2.8k (0.994) | 883.8k (0.998) | 2.8k (0.988) | 949.5k (0.998) | 2.8k (0.992) |
| 50k | 158.3k (1.0) | 11.6k (0.99) | 796.3k (0.998) | 12.2k (0.954) | 885.2k (0.998) | 12.2k (0.95) |
| 100k | 166.2k (0.998) | 23.7k (0.982) | 771.5k (0.998) | 25.2k (0.894) | 870.8k (0.998) | 25.3k (0.888) |
| 500k | 171.0k (0.998) | 87.1k (0.28) | 716.1k (0.996) | 97.5k (0.136) | 829.1k (0.996) | 98.5k (0.13) |
| 1M | 175.8k (0.998) | 133.7k (0.30) | 573.3k (0.996) | 173.1k (0.136) | 758.8k (0.996) | 176.5k (0.14) |
| 2M | 131.5k (0.996) | 164.4k (0.364) | 339.1k (0.992) | 331.4k (0.202) | 450.5k (0.992) | 345.4k (0.19) |

At 4M rows (40 categories, just over the 4M gather limit) both configurations ran
the graph: 171.6k, 490.5k and 766.3k for b1, b16 and b32, lowest recall 0.29–0.53.

* The gather scan wins at every batch size up to 1M matching rows, by 1.3× to
  340×, with recall at or above 0.996 for every query. Graph search costs grow as
  filters narrow (cuVS's internal top-k grows with the filtering rate).
* At 2M rows the graph overtakes the gather for single-vector SQL (164k against
  131k) and ties at b16; the gather still wins at b32. The graph's mean recall stays
  above 0.97, but its weakest queries fall to 0.19–0.36.
* So the crossover is at about 2M rows for single queries and above 2M for
  batches, at k=500. A calibrated FP32 cost profile would only change routes there;
  with the graph's weak queries at these settings, the gather limit of 4M rows is the
  safer rule for a per-query recall target, and the profile was not recalibrated.
* Rate grouping for the graph route (Phase 3 of the earlier plan) matters where one
  graph search serves several selectivities: measured on mixed UDF batches, one
  search per rate group gives 28.7k q/s on 8 GPUs against 11.3k at the batch's
  highest rate and 22.0k with per-query bitsets, at the same recall
  ([details](ann-cagra-filter-udf.md#grouping-a-mixed-batch-by-filtering-rate)).
  The worker's bitset graph route already searches one filter (one rate) per call.

### The same sweep at k = 100

| Matching rows | Gather b1 | Graph b1 | Gather b16 | Graph b16 | Gather b32 | Graph b32 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10k | 214.7k (0.99) | 10.7k (0.99) | 1,493.3k (0.99) | 10.9k (0.94) | 2,056.2k (0.99) | 11.0k (0.93) |
| 50k | 197.0k (0.99) | 36.2k (0.97) | 1,394.4k (0.99) | 40.8k (0.64) | 1,852.1k (0.99) | 41.1k (0.65) |
| 100k | 194.1k (0.99) | 59.9k (0.96) | 1,434.8k (0.99) | 67.2k (0.23) | 1,897.0k (0.99) | 67.7k (0.24) |
| 500k | 197.3k (0.99) | 160.0k (0.66) | 1,356.7k (0.99) | 193.2k (0.29) | 1,724.2k (0.99) | 196.1k (0.30) |
| 1M | 207.9k (0.99) | 215.8k (0.66) | 853.4k (0.99) | 315.1k (0.38) | 994.1k (0.99) | 325.9k (0.44) |
| 2M | 163.5k (1.0) | 215.1k (0.90) | 451.8k (0.99) | 440.5k (0.73) | 288.4k (0.99) | 460.3k (0.68) |

At k=100 the crossover moves down, as the cost argument predicts (graph work grows
with k / (1 − r)): single-vector SQL ties at 1M rows and the graph wins at 2M;
batches tie at 2M (b16). The gather's 2M b32 point varies between runs: five reruns from
fresh workers gave 295.1k once (the worker at 7 cores, GPUs 62% busy: a stall, not a
limit) and 489.2k–490.7k four times, above the graph's 460.3k; b16 repeated at
449.0k. Below 1M rows the gather is 1.2× to 190× faster with every query at
0.99 or better.

### Cost-profile routing against the limits

The deployed FP16 profile ([cost model](ann-cost-model.md)) requires its calibrated
settings (64 MiB matrices) and, with this change, `--fp16-exact on`. Same worker and
settings otherwise, k=500, with and without `--cost-profile`:

| Matching rows | Limits b1 | Profile b1 | Limits b32 | Profile b32 | Profile's choice |
| ---: | ---: | ---: | ---: | ---: | --- |
| 100k | 159.9k | 156.7k | 864.3k | 866.9k | NVFP4 scan + GPU rerank |
| 1M | 176.2k | 171.4k | 668.2k | 665.5k | NVFP4 scan + GPU rerank |
| 2M | 129.9k | 92.2k | 206.4k | 247.8k | mostly CAGRA + GPU rerank |

Recall was the same (lowest query 0.992 or better). The profile matches the
limits up to 1M rows; at 2M it trades 29% of single-query throughput for 20% more
batched throughput.

**FP32 calibration.** `calibrate_cost.py` ran against the FP32 CAGRA + NVFP4 index
(GPU 3, 95% recall target, frontier 4096, 256 extra graph candidates; 63 cells,
[profile](../benchmarks/ann/results/frozen-10pct-throughput/fp32-cost-profile/)). At
k=500 its scan beats the graph up to 1M matching rows at every batch size (0.43 ms
against 0.79 ms for one search at 1M) and loses from 5M (1.13 ms against 0.75 ms).
The worker now applies a profile to an FP32 index (FP16 indexes still need
`--fp16-exact on`). On that index, same settings, with and without it:

| Matching rows | Limits b1 | Profile b1 | Limits b32 | Profile b32 |
| ---: | ---: | ---: | ---: | ---: |
| 100k | 161.7k | 161.5k | 871.7k | 873.0k |
| 1M | 171.7k | 168.9k | 682.8k | 685.5k |
| 2M | 130.3k | 93.3k | 206.9k | 205.5k |

Recall is identical (lowest query 0.992 or better). Up to 1M rows the profile
chooses the NVFP4 scan, as the limits do; at 2M it sends single searches to the
graph (isolated latency favours it, saturated throughput does not) and loses 28%.
For saturated throughput the 4M gather limit is the better rule here.

At 4M matching rows (gather limit raised to 8M so the gather takes them) the graph
wins at k=100: 221.7k against 31.6k for single-vector SQL, 884.1k against 45.8k
(b16) and 1,205.9k against 45.8k (b32), lowest recall 0.85–0.99. At that size the
gather's cutoff would keep more than the sort holds, so it selects from all 4M
scores. With the default 4M limit these searches take the graph anyway.

### Distinct predicates

`mixed10` draws each search's range from 91 overlapping 10% windows
(`category BETWEEN a AND a+9`) instead of the workload's 10. Array batches share
one predicate per call and barely change (749.1k against 758.8k at b32, lowest
recall 0.996; the graph route gives 175.8k with 0.156). Single-vector SQL loses 11%
(156.5k against 175.8k): same-filter groups shrink. That is the case a per-query
predicate pipeline would address.

### Mixed-filter groups (`--mixed-groups`)

Phase 2 of the earlier plan, built in the worker without a protocol change: a lane
whose same-filter group is short also takes queued searches over other cached
filters, and those whose filter appears once in the batch share one segmented
gather. Their device row lists are laid end to end, a segmented NVFP4 kernel scores
each row against its own query, each query keeps its scores under its own sampled
cutoff (every score for lists up to 16,384 rows) and sorts them in shared memory,
and the candidates are reranked in FP32 on the GPU. Single-vector SQL, k=500, 512
clients, 32 lanes, `ann.frozen_immutable = on`:

| Workload | Off | On | Lowest recall |
| --- | ---: | ---: | --- |
| `unique1`: 991 distinct 1% tenant windows | 101.2k | **146.9k (+45%)** | 0.998 / 0.998 |
| `mixed10`: 91 overlapping 10% windows | 158.4k | 119.0k (−25%) | 0.996 / 0.996 |
| `10pct`: 10 ranges | 173.7k | 172.1k | 0.998 / 0.998 |

When predicates rarely repeat, same-filter batches average 1.5 searches and the GPUs
are busy with per-search launches (93%); mixed groups average 6 and leave them at
69%. When predicates repeat, the same-filter matrix gather scores a shared row list
once per group, while a segmented gather scores every search's rows separately, and
pulling other filters into a lane splits the groups other lanes would form. Mixing
every job, not only singletons, measured 80.9k on `mixed10`. The option stays off by
default. Unlike Phase 2 as planned, candidates come from the cached predicates' row
lists rather than from postings evaluated on the GPU, and predicates are not
compiled per vector (operation 30 still shares one).

## Final configuration

```sh
ann/ann-worker-cuvs --backend cuvs --listen "$SOCKET" \
  --lanes 96 --max-batch 64 --replicate on --itopk 512 --filtered-itopk 1024 \
  --filtered-algo multi --filtered-width 4 --exact-limit 0 \
  --gather-limit 4000000 --bitset-gather-limit 4000000 --group-gather-limit 4000000 \
  --memory-pool 40 --predicate-cache-mb 8192 --predicate-device-mb 16384 \
  --max-pending-mb 8192 --rerank-prefetch 8 --replica-routing home \
  --gpu-rerank on --matrix-gather on --matrix-max-mb 512 --matrix-prune on

python benchmarks/ann/frozen_stress.py --socket "$SOCKET" --worker-pid "$WORKER_PID" \
  --output results --label arrays --clients 448 --seconds 60 --k 500 --workloads 10pct \
  --materialize-batch 64 --index-name frozen_stress --batch-queries 64 --batch-arrays \
  --immutable --plan-cache-mode force_generic_plan
```

No cost profile is loaded: limits route 1M-row groups to the matrix gather. For
ordinary single-vector SQL, 32 lanes and 512 clients did best (161.3k QPS, p99 6.2
ms; PostgreSQL used 151 cores, the worker 25, GPUs 70% busy).

| Path | vectors/s | Mean / lowest recall@500 | p99 |
| --- | ---: | --- | ---: |
| Deployed, ordinary SQL, 256 clients | 86,244 | 0.9999 / 0.998 | 5.5 ms |
| Final worker, ordinary SQL, 512 clients | 161,296 | 0.9999 / 0.998 | 6.2 ms |
| Final worker, `ann_frozen_search_batch`, 64 vectors | 398,345 | 0.9998 / 0.994 | 113 ms per call |
| Final worker, `ann_frozen_search_batch_arrays`, 64 vectors | 745,766 | 0.9998 / 0.994 | 52 ms per call |
| Merged tree, ordinary SQL, `ann.frozen_immutable = on` | 177,261 | 0.9999 / 0.998 | 5.3 ms |
| Merged tree, arrays, 64 vectors, `ann.frozen_immutable = on` | 780,404 | 0.9998 / 0.994 | 46 ms per call |

Batch latency covers all 64 vectors of a call. Latency was not optimized.

## Limits and next steps

* **The host CPU is the limit.** PostgreSQL, the worker and pgbench share 224
  cores. PostgreSQL's remaining work per call is reading keys and TIDs and checking
  visibility (about 15% of backend samples), text output of the id arrays (about
  11%), and parsing the text `vector[]` argument (about 4%). Clients on other hosts,
  binary result format, and binary query vectors would each reduce it.
* **The GPUs have headroom**: 79% busy at the final point. NVFP4 scoring of the
  1M eligible rows per group and the final top-500 after reranking are now the
  largest kernels.
* **Ordinary SQL** remains PostgreSQL-bound at about 1 ms of backend CPU per
  search: the per-row executor path, not the GPU.
* `--matrix-prune` is opt-in. The GPU index tests cover it with exact-result checks;
  it should be validated on other datasets before it becomes the default.
* The deployed worker and its cost profile were left unchanged. The test node's
  PostgreSQL now runs this branch's `vector.so`, with
  `ann_frozen_search_batch_arrays` created by hand in `gpu_ann`; the previous
  library is kept beside the experiment files.
