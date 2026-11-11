# Muse fan-out at production size on one 8× B200 host

How many Muse requests per second one GPU host serves over TCP when it
holds a production-size slice of the ads index, and what it took to get
there. Measured 2026-11-09 on 8× NVIDIA B200 (183 GB each), 224 CPUs, 2 TB
RAM, with three other workers of other test trees resident on the GPUs
(28–46 GB per GPU), the load generator on the same host over loopback.

## Workload

A Muse request fans out to 6 ANN sub-queries (operation 22, search by
predicate program): the captured sample's shapes, k = 300/600/1000, `pg_ids`
OVERLAP filters matching 4% (one shape 0.53%) of the rows, one duplicate
pair per request (the CPC and CPM rankers ask the same group the same
question). The captured sample has no unfiltered sub-query, so neither does
the test. Requests come from `benchmarks/ann/muse_frames.py emit`
(`--unfiltered-share 0`, 64 filter variants per shape, 200,000 distinct
requests) and `benchmarks/ann/muse_direct.c` (`--processes`, open-loop
`--rate`). Recall is exact, from the index files (`muse_frames.py check`,
180 sub-queries): the gate was 0.95.

A 20,000-request file inflated every number by 20–30%: with more clients
than requests the same request is in flight twice and the worker merges it
(`merged` ran at 37–48% instead of the genuine 16.5%). Use a request file
larger than the client count.

## Index

Synthetic scale-up of the 9,862-item seed (`benchmarks/ann/scale_export.py`,
copies placed at half the 5th-neighbour distance, attributes inherited so
filter selectivity holds) to a production 1/8 corpus shard: 37 groups
(19 × 65 dims, 18 × 33) × 46.5M rows, 1.72B vectors, FP8 scan codes with
fp16 CAGRA datasets, built by `benchmarks/ann/scale_pipeline.sh`. The graph
build fails on 33-dim groups and graph searches return garbage on this data
(thousands of copies per seed make every node's neighbours its own clique),
so the graphs are useless here: the winning route never reads them, and the
final layout is scan-only (`--scan-only`, 588 GB on disk, 50 min to build). With the 8-shard index (834 GB) kept beside it the test host's disk is full (44 GB free of 1.8 TB).

Load time is 9 minutes with `load.py` sending 8 loads at once (59 minutes
one at a time); `--prewarm off` keeps `vectors.f32` on disk, which the GPU
route never reads.

## Results

Honest request file, recall ≥ 0.98 unless noted, closed loop unless noted.

| Layout | Worker settings | Clients | Muse req/s | ANN q/s | p99 |
| --- | --- | ---: | ---: | ---: | ---: |
| 8 shards × 1 copy, fp16 + graphs (first run) | lanes 48, slack 50% | 8192 | 35.5k | 213k | 382 ms |
| same | lanes 96 | 16384 | 63.1k | 379k | 457 ms |
| same | + `--shard-topk-share 25 --slack-ratio 5` | 24576 | 93.4k | 561k | 294 ms |
| same, 11,954 distinct filters instead of 2,482 | same | 24576 | 53.9k | 323k | 503 ms |
| **4 scan-only shards × 2 copies** (`--replicate 2`) | lanes 96, share 40, slack 5 | 8192 | 120k | 721k | 81 ms |
| same | | 16384 | 134k | 805k | 145 ms |
| same | | 24576 | **137k** | 821k | 202 ms |
| same, open loop | | 16384 | 50k at p99 45 ms; 60k at 48; 70k at 51 | | |
| same, no rerank (`--rescore off`), recall 0.89 | | 24576 | 90k | 538k | 306 ms |
| same, `--mixed-groups on` | | 24576 | 57k | 340k | 481 ms |
| same, 192 lanes | | 24576 | 105k | 631k | 284 ms |

Recall at the final settings: mean 0.9986, min 0.98 (key sets), 0.9992 by
distance. GPU memory: 8 × 1 copy with graphs 80–88 GB per GPU; 4 × 2
scan-only with fp16 rerank 100–103 GB per GPU (fp16 48.5, codes 29.5, keys
3.4, predicate rows ~9, lane buffers ~10); without rerank 54–57 GB. Full
replication of the shard would need ~640 GB per GPU.

Per-GPU slice replicated 8× (5.8M rows per group, 1 shard, what each GPU
holds in production) ran at 126k with the small request file (37% merged),
so about 100k honest: the 4 × 2 layout on the full shard matches it.

## One shard per group (2026-11-09, later the same day)

Each batch of the 4 × 2 layout cost about 1 ms in its lane whatever it
scanned (a 9.6 times wider filter added 4%), plus 30–60 µs a query merging
the four shard lists. So each group became one shard living whole on one
GPU, with 2 copies on a pair of GPUs (`--replicate 2`; the index files are
the same, hard-linked under a manifest with `shards 1`). GPU memory is
unchanged, 107–109 GB per GPU.

One shard has no per-shard over-fetch, so recall needs more slack: 50%
gave 0.977 mean / 0.752 worst, 100% 0.988 / 0.855, 300% (4,000 candidates
at k = 1000, `ANN_PRUNE_CAP` raised to 32,768 for it) 0.998 / 0.986. The
rows below use 300%.

| Build | Peak Muse req/s | p99 at 120k/s | 140k/s | 160k/s |
| --- | ---: | ---: | ---: | ---: |
| One scheduler queue, 16 driver calls a batch | 176k | – | 68 ms | 118 ms |
| 12 scheduler queues, 11 driver calls | 210k | 45 ms | 50 ms | 59 ms |
| **4 scheduler queues (the default at 96 lanes)** | **214k** (1.29M ANN/s) | 40 ms | **44 ms** | 59 ms |
| 64 lanes, 8 queues | 170k | 65 ms | 84 ms | 125 ms |
| 48 lanes, 6 queues | 154k | 82 ms | 179 ms | – |

The rate held under a 50 ms p99 is about 150k Muse requests/s, against
65k for 4 × 2. With one queue the GPUs were 60–67% busy and lanes waited
on the scheduler mutex (45% of samples); with the queues split the GPUs
run at 98–100% and batches are 16 searches instead of 40.

What changed in the worker for it: per-index device placement for
`--replicate N`; scheduler shards (`--scheduler-shards`); `bound_cut`
replacing three launches and one result copy replacing four; equal
distances in a result list fixed in place (a host sort of the whole list
ran for about half the lists, 17% of lane CPU); a duplicate check over
runs of equal distances instead of a hash; `--scan-skip-tail`.

### Smaller K

The request's K (300/600/1000) is the leaf's eager top-K; the root keeps
60–552 a ranker. With K sized for half the candidates surviving the root's
filters (30–182 a ranker; `benchmarks/ann` frames rewritten in place), the
same worker held a 50 ms p99 to about 250k requests/s and peaked at 424k
(2.5M ANN searches/s, event-driven client). Whether those K are acceptable
needs the root's survival and depth-used statistics.

### Load generator

`muse_direct --epoll` drives all of a process's clients from one thread.
It matches the thread mode against a mock server and lifted the peak where
the client was the limit (small K); at k = 1000 one thread a process
drains 8 GB/s of replies and reads about 7 ms higher at p99, so latency
points at large K use the thread mode.

## What moved the number (3k → 137k)

1. Scan-only groups were excluded from the GPU matrix route and fell to
   host re-scoring at 6–12 ms a query (lanes 98% on the CPU): removed the
   exclusion (scan-only shards upload an fp16 rerank copy, padded to 8 dims
   so the fused rerank kernels apply).
2. Unfiltered k > 512 on a graph needs multi-CTA search; single-CTA
   rejected it and every such query failed.
3. Multi-shard pruned selection synced per shard, serializing the 8 GPUs:
   the deferred path now applies whenever a copy's shards are on distinct
   devices.
4. The pruned selection's sample cutoff used cuVS `select_k` (167 µs a
   batch): the shared-memory radix select (`prune_select_s`) does it in 50.
   The same kernel selects candidates directly for tiles under ~50k rows.
5. The fused rerank+top-k kernels never ran for 65/33 dims (`dims % 8`):
   query loads masked for the tail and alignment-checked.
6. Readers took the scheduler mutex per frame (61 cores, 54% blocked):
   each epoll round's searches are queued under one lock; pending-byte
   budgets became atomics; a lane releases its filter at its next pickup.
7. Replies went out as two writes: one `writev`.
8. Identical searches in a batch run once (`--merge-duplicates on`, 1 in 6).
9. Each shard returns a share of k to the merge (`--shard-topk-share`),
   which was half the lane CPU with 8 lists of 1000; slack 50% → 5%.
10. `--replicate N` copies with shards spread over GPUs: 4 × 2 halves the
    fan-out and the per-batch fixed cost of 8 × 1.

## What did not

- More lanes (192) or waking a lane per filter group: smaller batches.
- `--mixed-groups on`: the multi-shard matrix route only takes same-filter
  batches, so mixed batches fell to the per-job path.
- No rerank: recall 0.89, and host key lookups made it slower.
- NVFP4 codes: larger on the GPU than FP8 here (121 vs 84 B/row) and worse
  ordering at single-shard scale; FP8 throughout.

## Where the limits are now

At the peak the GPUs run at 92–97%. Below it the latency floor is the
batching: batches are per filter, the 2,482 test filters are equally
popular, and a 4-GPU batch costs ~2.4 ms in a lane whatever its size, so at
20k requests/s a request waits ~25 ms for enough same-filter company. Real
traffic with hot filters batches better; traffic with far more distinct
filters batches worse (54k at 11,954 filters). The section above took the
first lever (a cheaper batch, by putting each group on one GPU). 2 shards
× 4 copies with rerank does not fit: 163 GB of index a GPU before any
working memory.

## Graph route with a bitset filter (2026-11-10)

The same shard served from fp16 CAGRA graphs instead of the scan, to
compare the two ways of applying a `pg_ids` filter. Both start the same
way: the predicate is evaluated once on the host from the worker's posting
lists and cached per distinct filter. The scan route sends the result to
the GPU as a row list and scans those rows once for a whole same-filter
batch; the graph route sends it as a bitset, and each search walks the
graph on its own, keeping only the nodes that pass.

The graphs are useless on the tree above, so this ran on a rebuild with
the copies spread 8 times wider (`scale_export.py --bandwidth 4`, against
the default 0.5): 37 groups × 46.5M rows, one shard each, fp16 graphs and
FP8 codes, 952 GB. Every group built, the 33-dim ones included. The
requests use the smaller K (30–182); recall is from 120 checked
sub-queries.

| | Scan, K 300–1000 | Scan, K 30–182 | Graph, K 30–182 |
| --- | ---: | ---: | ---: |
| Peak Muse req/s | 214k | 424k | 10.9k |
| ANN searches/s at the peak | 1.29M | 2.5M | 65.5k |
| Rate held under a 50 ms p99 | ~150k | ~250k | > 8k |
| Recall mean / lowest | 0.998 / 0.986 | not checked | 0.72 / 0.34 |
| Spread of the copies (bandwidth) | 0.5 | 0.5 | 4 |

The graph route held a 4.5 ms p99 at 8k requests/s; the next step
offered, 16k, saturated it at 10.9k. At the same K the scan peaks at about
39 times the graph route, and the graph route's recall is far below the
0.95 gate. The trees differ in spread, so the ratio is indicative: graph
searches return garbage on the scan's tree, and the scan was not rerun on
this one.

Where the graph route's time goes (nsys over 5 s at 65k searches/s):
`search_multi_cta` is 95.8% of GPU kernel time, one launch per search at
450 µs on average. Batches averaged 1.7 searches. The worker used 10.6 cores,
lanes spent 59% of samples in `cudaStreamSynchronize` and readers sat in
`epoll_wait`, and the GPUs were 94–98% busy: GPU-bound.

The allocator set the first limit:

| Allocator | Peak Muse req/s | p99 at 2k/s | p99 at 8k/s |
| --- | ---: | ---: | ---: |
| `--device-allocator plain` | 3.0k | 843 ms | saturated |
| `--memory-pool 0` (stream-ordered) | 10.9k | 80 ms | 308 ms |
| `--memory-pool 95` (RMM pool) | 10.9k | 2.9 ms | 4.5 ms |

With the plain allocator two thirds of lane samples waited on the
driver's lock in `cuMemAlloc` and `cuMemFree`. Stream-ordered allocation
lifts the peak and the pool also removes the latency. `--memory-pool 60`
ran out of pool memory loading several groups' graphs; 95 fits, at
115–123 GB a GPU.

The walk length sets the cost. A worker with two of the groups (g0 and
g19, 2 copies each on 4 GPUs, the same K, multi-CTA,
`--filtered-itopk 512`, width 4):

| `--filtered-iterations` | ANN searches/s | Recall mean / lowest |
| ---: | ---: | ---: |
| 8 | 204k | 0.31 / 0.07 |
| 16 | 108k | 0.52 / 0.10 |
| 32 | 51k | 0.67 / 0.20 |
| 64 | 25.6k | 0.81 / 0.30 |
| 128 | 13.3k | 0.90 / 0.57 |
| 0 (automatic) | 41.4–41.8k | 0.71 / 0.27 |

Iterations times searches/s stays at 1.6–1.7M. The shortfall from full
recall roughly halves per doubling (0.33, 0.19, 0.10 from 32 to 128), so
0.95 would take about 256 iterations: about 6.7k searches/s on this
worker and, scaled by the full worker's ratio at the automatic setting
(65.5k against about 41.6k), about 1.75k Muse requests/s on the shard.
That is an extrapolation, not a measurement.

At the automatic setting, `--filtered-itopk` 256 or 512 and
`--filtered-width` 1, 2 or 4 all gave 41.4–41.8k at recall 0.71, and
single-CTA gave 3.0–6.5k at recall 0.18–0.19. At today's K a longer list
did raise recall: 0.80, 0.85 and 0.90 at 1024, 2048 and 4096 (plain
allocator, not rerun with the pool).

With 0.5–4% of rows passing, most nodes a walk reaches fail the bitset,
so collecting K passing neighbours takes many iterations, and every
search pays for its own walk; the scan pays once per same-filter batch.
This test had no unfiltered or broad-filter traffic, so where the graph
route overtakes the scan is not measured. Real embeddings, without
thousands of close copies per seed, may need shorter walks.

Graph worker flags (on top of the `wk.sh` defaults):

```
--replicate 2 --lanes 48 --max-batch 32 --filter-lanes 16 --memory-pool 95
--quantized-scan off --matrix-gather off --gpu-rerank off --scan-routes off
--exact-limit 0 --gather-limit 0 --bitset-gather-limit 0 --group-gather-limit 0
--itopk 512 --filtered-algo multi --filtered-itopk 512 --filtered-width 4
--merge-duplicates on --predicate-device-mb 4096 --prewarm off
```

## Graph route walking passing rows only (2026-11-11)

cuVS' filtered search scores every neighbour of a node it expands, keeps
rejected rows on its list as stepping stones and lengthens the list by the
filter's pass rate. The `chenqin/cuvs` fork's `main` (27.02 plus
`PASSING_TWO_HOP`) walks the passing rows instead, which is what makes the
scan cheap: a node's children are its neighbours that pass the bitset and,
through each neighbour that does not, that neighbour's passing neighbours;
rejected rows are never scored; the walk starts from random passing rows;
each query brings its own bitset, so one search serves a batch of different
filters (`--filtered-expansion two-hop`, `--filtered-samplings`,
`--mixed-graph on` with `--mixed-groups on`).

The tree matters more than the search. A simulation of the walk on the
index files (exact answers, the graph as built) showed the bandwidth-4 tree
above is close to structureless: local intrinsic dimension 23 (33-dim
groups) and 48 (65-dim) against about 10 for the seed rows, and a
32-neighbour overlap between neighbours of 0.04-0.08 against 0.35. Any walk
on it needs thousands of expansions. `scale_export.py --kernel tangent
--k 16 --bandwidth 1` keeps both figures (9.5 and 0.36), and the shard was
rebuilt with it (37 groups x 46.5M rows, 952 GB, 25 minutes; the
bandwidth-4 tree keeps only g0 and g19). On the seed rows a passing row's
nearest neighbour passes 26-58% of the time against a 0.4-3.8% base rate,
so real filters are clustered, as the tangent tree's are.

Two groups (g0, g19), 4 GPUs, 2 copies, K 30-182, 200,000 requests, recall
from 120 checked sub-queries:

| Search | Peak Muse req/s | Recall mean / lowest | p99 |
| --- | ---: | ---: | --- |
| cuVS' filtered search (itopk 512, automatic iterations) | 10.6k | 0.53 / 0.00 | 16 ms at 8k/s |
| two-hop, itopk 1024, 20 iterations, 8 samplings | 57.9k | 0.954 / 0.68 | 3.8 ms at 40k/s |
| the same with `--mixed-graph on` | 58.3k | 0.954 / 0.68 | 2.7 ms at 40k/s |
| FP8 scan | 283k | 0.999 / 0.97 | 4.2 ms at 120k/s |

The whole 1/8 shard on 8 GPUs, 2 copies, `--mixed-graph on`, one worker
kept loaded while the settings changed (10 s per point; itopk / 32 walkers
a query). Every point ran the GPUs at 88-100%.

| itopk, iterations, samplings | Lanes, batch | Peak Muse req/s | Recall mean / lowest | p99 at 60k/s | at 80k/s |
| --- | --- | ---: | ---: | ---: | ---: |
| 1024, 20, 8 | 128, 32 | 114k | 0.929 / 0.43 | 4.0 ms | 6.0 ms |
| 1536, 12, 8 | 128, 32 | 131k | 0.945 / 0.62 | 1.9 ms | 3.0 ms |
| 1536, 16, 4 | 192, 64 | 147k | 0.944 / 0.46 | 3.5 ms | 7.3 ms |
| **1536, 20, 4** | 192, 64 | **125k** | **0.957** / 0.70 | 5.4 ms | not held |
| 1536, 20, 4 | 128, 32 | 114k | 0.957 / 0.67 | 3.3 ms | not held |
| 1536, 16, 8 | 192, 64 | 119k | 0.961 / 0.82 | 4.3 ms | not held |
| 1536, 20, 8 | 128, 32 | 95k | 0.970 / 0.83 | 7.2 ms | 22 ms |
| 3072, 10, 4 | 128, 32 | 100k | 0.963 / 0.68 | 3.4 ms | 11 ms |

A second pass around the best point, 20 s closed loop and 15 s per rate, 192
lanes and batch 64 (256 lanes peaked the same with a higher p99 at every
rate):

| itopk, iterations, samplings, block size | Peak Muse req/s | Recall | p99 at 60k/s | 75k | 90k | 105k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **1536, 18, 4, 64** | **139k** | **0.951** | 7.2 ms | 7.9 ms | 8.8 ms | **10.3 ms** |
| 1536, 20, 4, 64 | 128k | 0.957 | 8.0 ms | 8.9 ms | 10.2 ms | 12.5 ms |
| 1536, 18, 4, automatic | 136k | 0.951 | 4.1 ms | 11.7 ms | 21.9 ms | 28.6 ms |
| 1536, 16, 8, automatic | 120k | 0.961 | 4.3 ms | 13.4 ms | 26.1 ms | 38.4 ms |
| 1536, 16, 4, automatic | 149k | 0.944 | 3.4 ms | 4.8 ms | 15.3 ms | 21.3 ms |

`--filtered-block-size 64` costs a few milliseconds at 60k/s and keeps the
p99 flat above it. At the 0.95 gate the shard serves 128-139k Muse
requests/s and holds a 10-13 ms p99 at 105k/s; the earlier graph result was 21.4k at recall
0.72 on the bandwidth-4 tree. More walkers buy recall more cheaply than
more iterations; a thread-block size of 64 and the kernel variants tried on
one GPU (fewer second-hop entries, skipping the second hop when the first
finds enough) moved throughput and recall together. "Not held" points
saturated in a 10-second open-loop run and are not rerun. The scan was not
measured on the rebuilt shard: with graphs and codes both loaded, two copies
did not fit beside the other tenants.

The scan stays about 5 times faster for these filters (283k against 58k on
the two groups): the search kernel spends about 21 us of a GPU per query,
the scan about 3, because a filter passing under 4% leaves it under 2M rows
to score once per batch. The walk's cost does not grow with the filter, the
scan's does, so wider filters and unfiltered searches are the graph's.

## Reproducing

Build: `BUILDER_ARGS=--scan-only benchmarks/ann/scale_pipeline.sh OUT 4715 4 fp8`
(the seed index directories and the group catalog as its environment
describes). Load: `benchmarks/ann/muse_direct.c` compiled with
`cc -O2 -std=c11 -pthread`; `benchmarks/ann/ann_profile.sh LABEL SECONDS`
profiles a window (GPU utilisation, kernel stage times, gdb stack
sampling, lock waits, server counters). The host's `~/b200dev/wk.sh NAME
PORT ROOT [flags]` wraps the worker start and the parallel load; the
final worker flags (one shard per group; for 4 × 2 use `--shard-topk-share 40
--slack-ratio 5` instead of the slack and skip-tail flags):

```
--replicate 2 --lanes 96 --readers 64 --filter-lanes 16 --max-batch 64
--matrix-gather on --matrix-prune on --gpu-rerank on --matrix-max-mb 512
--slack-ratio 300 --quant-slack 16 --merge-duplicates on --scan-skip-tail on
--gather-limit 4000000 --bitset-gather-limit 4000000 --group-gather-limit 4000000
--memory-pool 60 --predicate-cache-mb 8192 --predicate-device-mb 2048 --prewarm off
```
