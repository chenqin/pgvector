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

The 1/8 shard on 8 GPUs, 2 copies, K 30-182, 200,000 requests, the
released worker (60 s closed loop, 30 s per rate, all 37 groups, no errors,
GPUs 85-100% busy at the peak):

| Search | Peak Muse req/s | ANN queries/s | Recall mean / lowest | p99 at 105k/s | 135k | 150k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **two-hop graph search** | **170k** | 1.02M | 0.953 / 0.14 | 4.5 ms | 8.4 ms | 23.8 ms |
| FP8 scan | 388k | 2.33M | 0.998 / 0.97 | 21 ms (100k) | 24 ms (150k) | 47 ms (350k) |
| cuVS' own filtered search | 21.4k | 128k | 0.72 / 0.34 | saturated | | |

The last row is the earlier result on the bandwidth-4 tree. On the tangent
tree it was measured on two groups only (g0 and g19, 4 GPUs): 10.6k Muse
requests/s at recall 0.53, against 283k at 0.999 for the scan.

```
--filtered-expansion two-hop --filtered-itopk 1536 --filtered-iterations 18
--filtered-samplings 2 --filtered-block-size 64 --mixed-groups on --mixed-graph on
--lanes 128 --max-batch 64 --replicate 2 --memory-pool 95
```

itopk / 32 walkers search each query (48 here), each for the given number
of iterations, starting from the nearest of `--filtered-samplings` random
passing rows per slot.

Recall is from 300 checked requests: 1,732 sub-queries with rows to find,
over every group, exact answers from the index files, each setting measured
twice and agreeing to 0.001. Their query vectors are drawn like the load
file's, from rows of the bandwidth-4 distribution, which sit as far from
the item rows as the six captured queries do (nearest seed row at 0.95
against 0.77-1.17); vectors drawn from the tangent tree's own rows sit on
top of them (0.44) and score 0.02-0.03 higher. The 120 sub-queries used
elsewhere in this document cover 28 groups and gave 0.950 for a setting
that scores 0.961 here.

How the setting was chosen. One worker stayed loaded while the settings
changed (128 lanes, block size 64, 30 s closed loop and 20 s per rate a
point): 64 settings from 24 to 96 walkers, 8 to 39 iterations and 2 to 8
samplings, then all 108 combinations of 1 to 3 samplings, 40 to 64 walkers
and 14 to 24 iterations with 10 points in between.

| itopk, iterations, samplings | Recall | Peak Muse req/s | p99 at 105k/s | 120k |
| --- | ---: | ---: | ---: | ---: |
| 1024, 10, 4 | 0.873 | 275k | 4.1 ms | 5.8 ms |
| 1280, 14, 4 | 0.932 | 197k | 4.1 ms | 5.5 ms |
| 1536, 12, 4 | 0.939 | 180k | 4.0 ms | 5.0 ms |
| 1536, 14, 4 | 0.949 | 164k | 4.7 ms | 5.4 ms |
| 1408, 16, 4 | 0.950 | 164k | 4.9 ms | 5.8 ms |
| 1536, 18, 2 | 0.953 | 160k | 4.5 ms | 5.6 ms |
| 1536, 16, 4 | 0.956 | 150k | 5.4 ms | 6.9 ms |
| 1536, 18, 4 | 0.961 | 137k | 6.4 ms | 9.9 ms |
| 2048, 18, 4 | 0.974 | 97k | not held | not held |

About 10k requests/s buy 0.005 of recall around the gate. Of the 118
settings of the second pass 83 reached 0.948 and the best of them peaked at
162-170k whatever the mix of walkers, iterations and samplings; 24 or 32
walkers run at 250-280k and stop at 0.78-0.91. More walkers buy recall more
cheaply than more iterations, and 2 samplings instead of 4 cost 0.008 of
recall for 17% of throughput. Every setting leaves a few queries near 0.1,
where no walker started in the right cluster. The released worker measures
a little above the tuning runs (170k against 160k for the chosen setting;
1408, 16, 4 gave 172k at 0.950, exactly on the gate).

Lanes, batch and block size at the chosen setting, released worker:

| Lanes, batch | Peak Muse req/s | p99 at 105k/s | 135k | 150k |
| --- | ---: | ---: | ---: | ---: |
| **128, 64** | **170k** | 4.5 ms | 8.4 ms | 23.8 ms |
| 160, 64 | 167k | 6.2 ms | 9.0 ms | 23.2 ms |
| 192, 64 | 169k | 8.0 ms | 11.1 ms | 24.4 ms |
| 128, 32 | 156k | 4.6 ms | 8.1 ms | 40.7 ms |

`--filtered-block-size 64` keeps the p99 flat under load (10 ms against 29
at 105k/s with cuVS' own choice, at the same peak, measured with 4
samplings and 192 lanes). Kernel variants tried on one GPU, fewer
second-hop entries and skipping the second hop when the first finds enough,
moved throughput and recall together.

The FP8 scan ran on a graph-free view of the same files (hard links under
manifests without a graph, 2 copies, 96 lanes, slack 300%): with graphs and
codes both loaded, two copies do not fit beside the other tenants. It
peaks at 2.3 times the graph search and its p99 is three to five times
higher below 135k/s, where it waits to fill same-filter batches and the
graph search does not. The scan scores every row a filter passes, once per
batch, and a filter passing under 4% leaves it under 2M rows; the walk's
cost does not grow with the filter, so wider filters and unfiltered
searches are the graph's. Where the two cross is not measured.

Not covered: K 300-1000 with the two-hop search; filters passing under
0.4%; any filter but a `pg_ids` list. The requests derive from one captured
request, so the spread over groups and the filters' popularity are the
generator's.

## Related reading

Pinterest Engineering posts on the serving stack this test stands in for.
They were located by search and read in summary only; none describes Muse,
the 8-leaf fan-out or `pg_ids` filtering.

- [Beyond Two Towers: Re-architecting the Serving Stack for Next-Gen Ads Lightweight Ranking Models (Part 1)](https://medium.com/pinterest-engineering/beyond-two-towers-re-architecting-the-serving-stack-for-next-gen-ads-lightweight-ranking-models-1992f2b76cbb):
  the earlier stack streamed scores for about 100K documents from the GPU
  to the CPU, which dropped 99% of them after filtering; filtering and
  top-k moved onto the GPU. The same idea as the scan and the two-hop
  search here: apply the filter before paying for the row.
- [GPU-Serving Two-Tower Models for Lightweight Ads Engagement Prediction](https://medium.com/pinterest-engineering/gpu-serving-two-tower-models-for-lightweight-ads-engagement-prediction-5a0ffb442f3b):
  GPU serving for the lightweight ranking stage the test's rankers belong
  to.
- [Unlocking Efficient Ad Retrieval: Offline Approximate Nearest Neighbors in Pinterest Ads](https://medium.com/pinterest-engineering/unlocking-efficient-ad-retrieval-offline-approximate-nearest-neighbors-in-pinterest-ads-6fccc131ac14):
  online ANN growing expensive with the ad inventory, and the move from
  HNSW to IVF for a larger index. The captured request asks for IVF indexes
  throughout.
- [Ads Candidate Generation using Behavioral Sequence Modeling](https://medium.com/pinterest-engineering/ads-candidate-generation-using-behavioral-sequence-modeling-f9077ee1325d):
  candidate generators fetch a top-K by ANN, are capped to a quota and
  blended, the over-fetch behind K 300-1000 here.
- [How we scaled the size of Pinterest's ad corpus by 60x](https://medium.com/pinterest-engineering/how-we-scaled-the-size-of-pinterests-ad-corpus-by-60x-d6d5bfa6bf16)
  and [Redesigning Pinterest's Ad Serving Systems with Zero Downtime (part 2)](https://medium.com/pinterest-engineering/redesigning-pinterests-ad-serving-systems-with-zero-downtime-part-2-3e607bbba9f0):
  corpus size and the ad serving stack around retrieval.
- [Manas: A high performing customized search system](https://medium.com/pinterest-engineering/manas-a-high-performing-customized-search-system-cf189f6ca40f):
  the search platform's leaf and root serving.

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
