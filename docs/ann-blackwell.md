# Blackwell module: quantized filtered search on B200

The cuVS worker keeps using cuVS CAGRA for graph search. On Blackwell GPUs (compute capability 10.x, such as the B200, or 12.x, such as the RTX PRO 4500), filtered searches that go to the GPU gather scan use the worker's Blackwell module instead:

- **Quantized index.** The builder writes a 4-bit (NVFP4) or 8-bit (FP8 E4M3) copy of the vectors next to the exact FP32 ones. The worker keeps it in each shard's GPU memory.
- **Blackwell kernels.** The worker compiles the module with NVRTC for the device's Blackwell-specific target (`sm_100a` or `sm_120a`). On 12.x devices each grouped kernel's dynamic shared memory is capped at the device's opt-in limit (99 KB rather than B200's 227 KB) less its static shared memory, so wide vectors take smaller query groups. It decodes with Blackwell's conversion instructions (`cvt.rn.f16x2.e2m1x2` for FP4 and `cvt.rn.f16x2.e4m3x2` for FP8), which were checked bit-exact against a host decoder on the B200s.
- **Fused search.** One kernel launch scores the eligible rows and selects the best 32, replacing the gather path's separate key lookup, scoring kernel, `cuvsSelectK` and copies.
- **Exact results.** The worker re-scores the 32 candidates with the FP32 vectors and returns the top k, so distances match `<->`.

PostgreSQL, the SQL functions and the network protocol are unchanged: a GPU node loads a quantized index like any other, and `ann_index_search`, `ann_hybrid_search`, `ann_attribute_search` and `ann_bitmap_search` use the module whenever their filter reaches the gather scan.

## Building a quantized index

```sh
python3 scripts/ann/build_cuvs_index.py --dsn 'dbname=app' --table public.docs \
    --key id --vector embedding --out /data/ann/docs --quantize nvfp4    # or fp8
```

| File | Content |
| --- | --- |
| `vectors.nvfp4` (NVFP4) | Per row, `ceil(dims / 16) * 8` bytes: two E2M1 values per byte, the even dimension in the low nibble, dimensions zero padded to a multiple of 16 |
| `scales.e4m3` (NVFP4) | Per row, one E4M3 scale per 16 dimensions, rounded up so no value clips |
| `vectors.e4m3` (FP8) | Per row, dimensions rounded up to 16 bytes of E4M3 values |
| `rowscale.f32` (both) | One FP32 scale per row |

A value decodes as `e2m1 × block scale × row scale` (NVFP4, the two-level scaling of the NVFP4 format) or `e4m3 × row scale` (FP8). The manifest gets a line `quant nvfp4` or `quant fp8`, and `vectors.f32` and the CAGRA shards are written as usual. The builder uses CuPy when it's installed, else NumPy, and both produce identical bytes. Quantizing 10M × 128 took 5 s.

| 10M × 128 | Bytes per row | GPU memory per index copy |
| --- | --- | --- |
| FP32 (`vectors.f32`) | 512 | 5.1 GB |
| FP8 | 128 + 4 | 1.26 GB |
| NVFP4 | 64 + 8 + 4 | 0.72 GB |

The CPU worker loads quantized indexes too. It validates the files and keeps answering exactly from `vectors.f32`.

## In the worker

At load, each shard's slice of the quantized copy goes to the GPU holding that shard, or to every GPU with `--replicate on`. If every GPU is compute capability 10.x or 12.x and NVRTC compiles the module for its target, the index's gather scans use the module. The worker's log says so (`Blackwell module: nvfp4 gather scan for index docs on (724.8 MB of quantized rows on the GPUs)`), and so do the replies to load and index info (`quant nvfp4 quant_scan on`). On other GPUs, or with `--quantized-scan off`, the copy isn't uploaded and the scan reads the CAGRA dataset as before.

**Fused search kernel.** For each query and shard:
1. The worker uploads the eligible rows. Filters that arrive with their ordinals (cached bitmaps, attribute values) become 4-byte shard rows on the host. Key lists are uploaded as keys and found by the existing lookup kernel.
2. One launch scores and selects. Eight threads score a row, each taking 16 dimensions with one 8-byte (NVFP4) or 16-byte (FP8) load and, for NVFP4, one block-scale decode. A warp scores four rows a step and keeps its best 32 in registers, merging new rows with warp-level bitonic sorts only when one beats both its own 32nd best and the best 32nd best any warp of its block has published. The last block to finish merges every block's list, its eight warps in parallel.
3. The worker copies back the 32 positions, re-scores them in FP32, and merges shards as before. It checks the positions for repeats, as on every gather scan.

Small lists get 128-row blocks, so enough warps hide the scattered row loads; larger ones 512-row blocks, up to four per SM, so the final merge stays short. The path serves k ≤ 24 (at least 8 extra candidates); larger k uses the earlier path with `cuvsSelectK` and `--quant-slack` extra candidates (default 256).

**Grouped search on the tensor cores.** When several queued searches share a cached filter ([same-filter grouping](ann-gpu-filters.md#same-filter-grouping)), the module scores the filter's rows for up to 16 queries at once with `mma.sync` (m16n8k16, FP16 inputs, FP32 accumulation). Each warp takes 16 rows at a time and loads their quantized bytes and block scales up front. The FP4 and FP8 conversion instructions produce the A fragments directly: a `cvt.rn.f16x2.e2m1x2` result is two neighbouring dimensions of one row, which is exactly how the fragment lays out k. Block scales are folded into the fragments with `mul.rn.f16x2`; the B fragments are two 32-bit loads of the half-precision queries from shared memory, where the earlier scalar kernel needed 64 `LDS.128` per 16 dimensions. Distances come from ‖x‖² + ‖q‖² − 2x·q, with ‖x‖² summed from the same half values, and go to a [queries × rows] matrix. A batched top-32 kernel then scans each query's row of the matrix in about 37 slices (one query per block, one score per lane, a register list per warp merged only when a score beats the warp's 32nd best or the block's bound), and a merge kernel folds the slices. All candidates are re-scored in FP32.

| Rows | 1 query (fused kernel) | 2 queries | 8 | 16 |
| --- | --- | --- | --- | --- |
| 10k | 17 µs | 23 µs (12 per query) | 23 (2.9) | 23 (1.4) |
| 100k | 41 µs | 41 (21) | 46 (5.8) | 50 (3.1) |
| 1M | 156 µs | 189 (95) | 237 (30) | 250 (16) |

The score kernel itself takes 136 µs for 16 queries over 1M rows (8.5 µs per query) and 21 µs over 100k, about the time the single-query kernel spends gathering one query's rows: the multiply is free, the gather is the cost. The selection takes most of the rest at 1M rows.

The same kernel reads FP16 and FP32 CAGRA datasets in place (`score_mma_half`, `score_mma_float`), so grouped searches on indexes without a quantized copy take the tensor-core route too: an FP16 pair is already an A fragment, and an FP32 pair becomes one with `cvt.rn.f16x2.f32`. Dimensions past `dims` (a dataset row's padding) are masked. Single queries on those indexes keep the FP32/FP16 gather scan.

**Wider passes: 32 and 64 queries.** The scan comes in three widths, 2, 4 or 8 tiles of 8 queries (`score_mma_nvfp4`, `_32`, `_64`, and the same for FP8, FP16 and FP32 rows): a warp loads and decodes its 16 rows once and runs one `mma.sync` per tile. Groups take the narrowest width that holds them, up to 64 queries, as long as the queries' FP16 copies fit in shared memory (the wider kernels opt in to up to 160 KB per block; at 1,024 dimensions 64 queries use 128 KB). `ANN_BLACKWELL_GROUP=16` or `32` in the worker's environment caps the width. Per query, against groups split into passes of 16 (one B200, NVFP4, `benchmarks/ann/blackwell_bench.py time`):

| Dimensions | Rows | 16 per pass | 32 per pass | 64 per pass |
| --- | --- | --- | --- | --- |
| 128 | 100k | 3.59 µs | 2.37 | 1.83 (1.96×) |
| 128 | 1M | 15.9 µs | 12.3 | 10.5 (1.51×) |
| 1,024 | 100k | 10.2 µs | 8.25 | 7.17 (1.43×) |
| 1,024 | 1M | 78.1 µs | 47.6 | 42.5 (1.84×) |

Not 4×: at 128 dimensions the top-32 selection over the score matrix is half of a pass (5.5–8 µs per query at 1M rows) and does not shrink with width; at 1,024 dimensions the scan itself only halves per query (72.5 to 36.9 µs at 1M rows), because a 64-query pass issues about 32M `mma.sync` per 1M rows, roughly 113 TFLOPS of FP16, while reading rows at only about 245 GB/s: it is bound by the legacy `mma.sync` path, not by memory. Block sizes of 512 threads and more blocks per SM changed per-query times by ±15% in both directions and were not kept. The selection is now fused into the scan for large groups (below); the remaining step is Blackwell's native block-scaled FP4 or `tcgen05` tensor-core path, which would remove the FP16 conversion and multiply 4× more per instruction. A narrower group run at a wider width costs more (16 queries at width 64: 1.4–2× the width-16 time), which is why the width follows the group. End to end on the [collections](ann-collections.md) cluster (16M × 1,024, four 2-GPU nodes, TCP), same-filter groups averaged about 5 searches, so wide passes ran rarely: the 2×2 layout's 10% filters went from 62.8k QPS (p99 26.6 ms, groups capped at 16) to 69.4k (p99 19.9 ms); 1% filters were unchanged within noise (143k, and 197–200k on the full layout). The gain grows with group size: hot filters at high load, and large eligible sets as at a billion rows.

**The fused scan's defaults** (NVFP4): each shard also gets a lane-contiguous copy of its rows (each lane of a quad loads one 16-byte word per row; `ANN_FUSE_LAYOUT=0` drops it) and a (row scale, ‖x‖²) pair per row made at upload (`ANN_FUSE_ROWNORM=0`), and each lane keeps its hits in a private shared-memory list flushed once per 16-row tile (`ANN_FUSE_APPEND_MODE=0`). The kernel then fits 80 registers and runs 3 blocks per SM: 176 → 122 µs for 64 queries over 1M rows at 128 dimensions, for about 72 extra bytes a row on the GPU. The scan also runs a kernel compiled for 128 dimensions (`ANN_FUSE_TUNE=42`; 0 restores the generic kernel) and stages queries four floats at a time (`ANN_FUSE_QVEC=3`), and the stages after it use a shared-memory radix select with a fused rerank and final top-k (`ANN_POST=452`; 0 restores the earlier kernels): 122 → 88 µs for the scan, 80 → 38 µs after it. Measurements are in the [exploration notes](ann-filter-throughput-exploration.md).

**Fused selection.** A group scanning at least 262,144 rows of at most 512 dimensions no longer writes a score matrix. The module first scores a spread sample of 8,192 of the rows (every n / 8,192-th, so a list sorted by key, which may follow tenants, still gives a fair sample) and takes each query's 32nd best there: no row of the query's true best 32 can be farther. The scan (`fuse_mma_*`, the same tensor-core code) then appends a row to the query's candidate list only when it is at most that bound, and `select32c` picks the best 32 from the candidates. The lists have room for every row, so a poor sample costs time, never correctness. About 32 × n / 8,192 rows qualify per query (3,900 at 1M rows). Per query, one B200:

| Dimensions | Rows | 16 per pass, matrix / fused | 64 per pass, matrix / fused |
| --- | --- | --- | --- |
| 128 | 300k | 6.58 / 6.12 µs | 4.37 / 2.67 µs |
| 128 | 1M | 16.5 / 11.9 µs | 10.5 / 6.01 µs (1.75×) |
| 1,024 | 1M | 80.3 / 67.2 µs | 42.5 / 40.8 µs |

Below 262,144 rows the sample costs more than the matrix it saves (at 100k rows, 16 per pass: 3.59 against 4.62 µs), and at 1,024 dimensions the scan dominates (300k rows: 25.5 against 28.6 µs), hence the limits; appends are atomic, so a smaller sample (more candidates) was slower too (1M rows, 64 per pass: 17.3 µs with a 1,024-row sample). `ANN_BLACKWELL_FUSE=off` turns it off. End to end on the 10M × 128 frozen index (one B200, 24 worker cores, two PostgreSQL hosts with 512 clients each, frozen route): 10% filters (1M rows each) went from 87.6k to 125.6k QPS (p99 16.2 to 12.2 ms), three hot 10% ranges from 91.6k to 140.0k, and 1% filters (100k rows, below the limit) stayed at 258–262k. Under that load 99.99% of searches ran grouped (2.03M of 2.03M, groups of about 16), and 320 concurrent answers for one 10% filter, checked against exact SQL meanwhile, were all exact.

**Correctness gates** (`benchmarks/ann/blackwell_bench.py check`, the loop [AutoKernel](https://github.com/RightNow-AI/autokernel) runs for PyTorch kernels, here over the worker's own NVRTC source compiled through CuPy): scores within 2% of float64 distances to the dequantized rows (relative to ‖x‖² + ‖q‖²) at 64, 128, 200 and 1,024 dimensions, 1 to 20,000 rows and 1 to 64 queries, every width; each query's 32 candidates contain its exact top 10; rows absent from the index score +∞; three runs are bitwise identical (the fused path's candidates arrive in any order, so its results are compared as sorted sets). Every case runs with and without the fused selection, up to 50,000 rows. A deliberately wrong reference (rows scaled by 1.1) fails the score gate (error 0.19).

**How it got there.** The first multi-query kernel (one thread per row, FP32 FMAs against queries in shared memory, per-warp best-32 lists with warp shuffles) plateaued at 55–60 µs per query at 1M rows whatever the group size. Its SASS showed 1,300 `WARPSYNC.COLLECTIVE` pairs (shuffles and ballots the compiler could not prove convergent), and ablations split its 928 µs at 16 queries into 524 µs of compute (bound by the 64 `LDS.128` feeding 256 multiply-adds per chunk) and 392 µs of selection. The selection was structural: 4,736 warps each saw only 211 rows, so no warp's list ever filled and the sort-and-merge path ran on almost every tile. Moving the multiply to tensor cores removed the compute term; moving selection out of the scoring kernel into long slices (37 instead of 4,736 lists per query) removed the other. An intermediate design with insertion into shared-memory lists was 3× slower still: sixteen lanes' lists 32 words apart hit one bank.

## Measurements

10M × 128 rows, k = 10, one B200.

**Kernel stages, per query** (one GPU stream, CUDA events, 200 repetitions; random eligible rows):

| Stage (µs) | 10k rows (0.1%) | 100k rows (1%) | 1M rows (10%) |
| --- | --- | --- | --- |
| Earlier path: upload keys | 4.8 | 18.0 | 147 |
| Earlier path: look up rows | 8.3 | 8.2 | 27 |
| Earlier path: score (FP32) | 5.4 | 14.4 | 168 |
| Earlier path: `cuvsSelectK` | 23.7 | 33.2 | 68 |
| Earlier path: copy result | 3.1 | 3.0 | 3 |
| **Earlier path, total** | **45** | **77** | **413** |
| Fused: upload 4-byte rows | 4.0 | 10.5 | 75 |
| Fused: score and select (NVFP4) | 15.4 | 40.9 | 156 |
| Fused: copy 32 results | 3.2 | 3.1 | 3 |
| **Fused NVFP4, total** | **23 (2.0×)** | **55 (1.4×)** | **234 (1.8×)** |

FP8 runs within 2% of NVFP4: the scan is bound by scattered row loads and latency, not bytes. Scoring alone (without selection) took 10.6 µs for 100k rows and 82 µs for 1M from NVFP4, against 14.4 and 168 µs from FP32. Both formats found the exact top 10 inside their top 32 at every size.

**Recall**, measured straight against the worker (key lists from 40 queries per filter size, exact SQL ground truth): 1.000 for FP32, FP8 and NVFP4 at 0.1%, 1% and 10%, with the fused path's 22 extra candidates and also with only 16 on the earlier path.

**End to end, single queries** (cached bitmaps, `ann_bitmap_search`, 512 clients per PostgreSQL host, QPS; before same-filter grouping, which the next section and [the grouping guide](ann-gpu-filters.md#same-filter-grouping) measure):

| Workload | 1 GPU, FP32 | 1 GPU, FP8 | 1 GPU, NVFP4 | 8 GPUs replicated, FP32 | 8 GPUs, FP8 | 8 GPUs, NVFP4 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.1% filter | 76,894 | **107,788** (+40%) | 106,860 | 255,008 | **281,936** (+11%) | 282,668 |
| 1% filter | 35,782 | **51,438** (+44%) | 48,525 | 142,191 | **175,569** (+23%) | 175,216 |
| 10% filter | 16,054 | 15,942 | 15,941 | 107,875 | 107,088 | 106,715 |

The 1-GPU runs (28 worker cores) are driven by one 64-core PostgreSQL host; the 8-GPU runs (96 worker cores, `--replicate on`) by two, and report their sum. All use default settings, with no route flags.
- **0.1% and 1% filters take the fused path.** Their cached bitmaps hold ordinals, so the worker uploads 4-byte rows and skips the key lookup and `cuvsSelectK`.
- **10% filters are unchanged.** Above `--bitset-gather-limit` (500,000 rows) a cached filter runs filtered CAGRA, which the module doesn't touch. Forcing them onto the fused gather scan (`--bitset-gather-limit 2000000`) gave 10.1–10.5k QPS on one GPU, against 16k on CAGRA and 5.0k on the FP32 gather scan.
- **FP8 and NVFP4 perform alike.** The 7× smaller NVFP4 copy is the choice when GPU memory matters: 0.72 GB instead of 5.1 GB per copy of 10M × 128.

Sending key lists straight to the worker (`ann_index_search` with keys) showed no gain: those requests carry 0.8–8 MB of keys each, and moving them over TCP, not the GPU, is the limit there.

**End to end, grouped** (same-filter grouping on, defaults; one GPU and one host, or 8 GPUs with the index replicated and two hosts; QPS): 10% filters 82,058 on one GPU (FP32 index on batched CAGRA: 47,112) and 320,528 on eight; 1% filters 284,012 and 312,812; skewed 1% traffic (10 hot filters) 339,541 and 327,887. All exact in the concurrent checks (4.6M answers).

## Not used, and why

- **Blackwell's new tensor-core instructions (`tcgen05`, block-scaled FP4/FP8 MMA).** The grouped search uses the warp-level `mma.sync` with FP16 inputs, which Blackwell still supports, after converting FP4/FP8 to half. The multiply is already free at that point (136 µs of 250 at 1M rows × 16 queries is the row gather), so the native FP4 MMA would save the conversions but not change the bound. A single filtered query stays matrix-vector work on the scalar kernel.
- **Triton.** It could have prototyped these kernels; it would have to ship as an ahead-of-time CUBIN loaded by the C worker, since the query path stays C. Not needed in the end: the kernels are about 300 lines of CUDA compiled at run time with NVRTC.
- **TMA, thread-block clusters, the 228 KB shared memory, NVLink 5.** Nothing measured here is limited by what they address.
- **cuVS graph search on quantized data.** cuVS 26.10 has no FP4 or FP8 CAGRA path, so broad filters (above `--bitset-gather-limit`) still run filtered CAGRA on the FP32 or FP16 dataset.

## Tests

- `test/ann/test_index.py` (CPU worker): an index with a `quant nvfp4` or `quant fp8` manifest line loads, reports `quant_scan off`, and answers exactly; truncated quantized files and unknown kinds are rejected.
- `test/ann/test_cuvs_index.py` (cuVS): indexes built with `--quantize nvfp4` and `--quantize fp8` report `quant_scan on` on compute capability 10.x GPUs; filtered searches through the module have recall ≥ 0.99, exact distances, and no row outside the filter.
