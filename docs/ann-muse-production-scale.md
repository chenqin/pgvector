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
final layout is scan-only (`--scan-only`, 588 GB on disk, 50 min to build).

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
8. Identical searches in a batch run once (`--merge-duplicates`, 1 in 6).
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
filters batches worse (54k at 11,954 filters). The levers left are a cheaper
small batch (fewer launches per tile, CUDA graphs), mixed-filter batching
on the multi-shard matrix route, and, on a host without other tenants,
2 shards × 4 copies with rerank (about 160 GB per GPU).

## Reproducing

Build: `BUILDER_ARGS=--scan-only benchmarks/ann/scale_pipeline.sh OUT 4715 4 fp8`
(the seed index directories and the group catalog as its environment
describes). Load: `benchmarks/ann/muse_direct.c` compiled with
`cc -O2 -std=c11 -pthread`; `benchmarks/ann/ann_profile.sh LABEL SECONDS`
profiles a window (GPU utilisation, kernel stage times, gdb stack
sampling, lock waits, server counters). The host's `~/b200dev/wk.sh NAME
PORT ROOT [flags]` wraps the worker start and the parallel load; the
final worker flags:

```
--replicate 2 --lanes 96 --readers 64 --filter-lanes 16 --max-batch 64
--matrix-gather on --matrix-prune on --gpu-rerank on --matrix-max-mb 512
--shard-topk-share 40 --slack-ratio 5 --quant-slack 16
--gather-limit 4000000 --bitset-gather-limit 4000000 --group-gather-limit 4000000
--memory-pool 60 --predicate-cache-mb 8192 --predicate-device-mb 2048 --prewarm off
```
