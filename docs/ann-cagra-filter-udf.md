# CAGRA filter UDFs for FP32 search

cuVS 26.10's C++ API accepts `filtering::udf_filter`: CUDA source for a
`__device__ bool f(query_id, source_id, filter_data)` predicate that cuVS
compiles with NVRTC and links into CAGRA's sample filter by JIT-LTO. The worker
uses the C API, which only takes bitsets. This experiment adds a C++ bridge
and measures filter UDFs against bitsets on the 10M-row FP32 graph. The worker
does not route queries to it yet.

| File | Role |
| --- | --- |
| `ann/udf.{h,c}` | Compiles a predicate program (`ann/filter.h`) into postfix steps, evaluates one row on the CPU (`AnnUdfMatch`) and estimates matching rows (`AnnUdfEstimate`) |
| `ann/cagra_udf.{h,cpp}` | `AnnCagraUdfSearch`: FP32 CAGRA search with a fixed device interpreter as the UDF |
| `test/ann/udf_test.c` | 3,000 random programs (NULLs, negated leaves, NOT, AND/OR) select the same rows as `AnnFilterEvaluate` |
| `benchmarks/ann/cagra_udf_probe.c` | Bitset vs UDF benchmark on one FP32 shard |

## Design

**One compiled source.** Generating CUDA per predicate would compile once
per predicate shape. Instead, the UDF is a fixed interpreter for the steps that
`AnnUdfCompile` produces, which are VALUES/RANGE/NULL leaves over int32
attribute columns, with NOT, AND, OR, ALL and NONE. cuVS keys its fragment
cache on the source text, so the interpreter is compiled once per process.
Predicates are passed only as data: an `AnnUdfParams` in device memory with
the column pointers, the steps, the sorted VALUES lists and `starts[q]`. Each
query in a batch can therefore have its own predicate. Leaves follow
`AnnFilterEvaluate`'s semantics: comparisons never match NULL, and a negated
leaf matches the non-NULL rows that the leaf does not.

**Reaching the C++ index.** `cuvsCagraIndex::addr` points at cuVS's private
index box, not at a `cagra::index`. Its first member is the index pointer and
its second the dataset layout (see `cagra_c_api_index_ptr` in cuVS's
`c/src/neighbors/cagra.cpp`). The bridge reads those two fields and accepts
only an FP32 device-padded index. The worker already refuses cuVS versions
other than 26.10, which keeps that mirror valid. Search parameters are copied
the same way as cuVS's `convert_c_search_params`.

**`filtering_rate` is required.** Given a bitset, cuVS computes the rejected
fraction itself and widens its search. A UDF is opaque, so when no rate is
given cuVS assumes 0. The caller must supply the rate: `AnnUdfEstimate` uses
exact counts for single leaves and assumes leaves are independent otherwise.
In a mixed batch, the most selective query sets the rate.

**Limits.** BITMAP leaves (stored filters) are not compiled
(`ANN_UDF_UNSUPPORTED`). Columns must be resident on the GPU: 4 bytes per row
per attribute, 40 MB per column at 10M rows. The UDF decides which rows can be
returned. It does not change graph traversal or the distance computation.

## Benchmark

One B200 (driver 580.126.09), cuVS 26.10, the 10M × 128 FP32 stress graph
(`fp32-native/index`), 32 queries (dataset rows plus noise), multi-CTA,
search width 4. Each case has 3 warmups and 15 timed searches, reporting the
median time per batch. Recall is measured against an exact scan of the
eligible rows. `category` takes values 0–99 at 1% each; `tenant` takes values
0–999 at 0.1% each. Every case uses graph search, as the `fp32-native`
endpoint does. The worker would send 1% filters to a gather scan instead, and
the probe doesn't compare against that route.

| Mode | Meaning |
| --- | --- |
| `bitset_cached` | Uniform predicate, with its bitset already on the device (as `--device-bitmaps`), one search for the batch |
| `bitset_cached_per_query` | Mixed batch: every query's bitset already on the device; one search per query, queued without waiting, one sync at the end |
| `bitset_per_query` | Mixed batch: each query also builds its bitset on the host (OpenMP) and waits for its search |
| `udf` | One UDF search for the whole batch, with the estimated `filtering_rate` |
| `udf_no_rate` | The same search with `filtering_rate = -1` |
| `udf_rate_10x_broad` | The same search with a rate that assumes 10× as many matching rows |

No mode returned an ineligible row in any run. The full results are in
[`benchmarks/ann/results/cagra-udf/`](../benchmarks/ann/results/cagra-udf/).

### Uniform predicates (median ms per batch)

| Case | Batch | k=10 bitset | k=10 UDF | k=500 bitset | k=500 UDF | k=500 recall (bitset / UDF) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| category 0–49 (50%) | 1 | 0.426 | 0.396 | 0.476 | 0.453 | 1.000 / 1.000 |
| | 32 | 0.718 | 0.661 | 1.846 | 1.858 | 1.000 / 1.000 |
| category 0–9 (10%) | 1 | 0.443 | 0.407 | 0.659 | 0.618 | 1.000 / 1.000 |
| | 32 | 0.823 | 0.795 | 4.412 | 4.502 | 1.000 / 1.000 |
| category = 7 (1%) | 1 | 0.442 | 0.405 | 1.348 | 1.342 | 1.000 / 1.000 |
| | 32 | 1.779 | 1.799 | 22.761 | 23.669 | 0.9994 / 0.9994 |
| category 10–19 AND tenant 0–499 (5%) | 1 | 0.410 | 0.379 | 0.681 | 0.676 | 1.000 / 1.000 |
| | 32 | 1.181 | 1.225 | 6.569 | 7.122 | 0.9981 / 0.9987 |

The k=500 runs use itopk 4096 with 256 extra candidates, the FP32 test
endpoint's settings. The k=10 runs use itopk 512 with no extra candidates;
recall was 1.000 in every k=10 row.

### Mixed batch (each query its own predicate)

Query *q* has category in a *q*-dependent window of 1–10 values (1–10% of
rows). Median ms per batch, with mean recall in parentheses:

| k | Batch | Per-query host bitsets | Cached bitsets per query | UDF | UDF, rate 10× broad | UDF, no rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 1 | 1.309 | 0.427 | 0.405 (1.000) | 0.387 (1.000) | 0.317 (0.50) |
| 10 | 16 | 21.245 | 7.397 | 1.257 (1.000) | 0.695 (1.000) | 0.362 (0.91) |
| 10 | 32 | 41.957 | 14.795 | 1.794 (1.000) | 0.792 (1.000) | 0.548 (0.95) |
| 500 | 1 | 2.836 | 1.352 | 1.336 (1.000) | 0.611 (0.68) | 0.364 (0.17) |
| 500 | 16 | 36.643 | 13.078 | 12.301 (1.000) | 2.555 (0.93) | 0.744 (0.59) |
| 500 | 32 | 73.280 | 26.145 | 23.616 (0.9998) | 4.493 (0.94) | 1.224 (0.61) |

Both bitset modes had recall of 0.998–1.000.

### Misestimated selectivity (uniform 1%, k=500)

| Batch | Correct rate 0.99 | Rate 0.9 (10× broad) | No rate |
| ---: | ---: | ---: | ---: |
| 1 | 1.342 ms, recall 1.000 | 0.616 ms, recall 0.664 | 0.367 ms, recall 0.180 |
| 32 | 23.669 ms, recall 0.9994 | 4.523 ms, recall 0.593 | 1.239 ms, recall 0.154 |

### Saturated throughput, 8 GPU replicas (k=500, batch 32)

`benchmarks/ann/cagra_udf_throughput.c` runs one process per GPU, each with its
own full copy of the index and 4 threads with their own streams, and an RMM
pool as with `--memory-pool`. It searches back to back for 15 s. One stream
already reached 94% of the saturated rate (cat10 UDF: 8,031 q/s with 1
thread, 8,559 with 8). The search settings and filtering rates are the ones
the probe validated for recall (0.998 or better). These are cuVS calls only,
with no worker, PostgreSQL or network in the path.

| Predicate | 1 GPU bitset | 1 GPU UDF | 8 GPUs bitset | 8 GPUs UDF |
| --- | ---: | ---: | ---: | ---: |
| category 0–49 (50%) | 21,257 | 21,557 | 170,039 | 172,461 |
| category 0–9 (10%) | 8,582 | 8,396 | 68,643 | 67,156 |
| category = 7 (1%) | 1,472 | 1,427 | 11,785 | 11,411 |
| mixed 1–10%, own predicate per query | 2,734 (cached bitset per query) | 1,417 | 21,879 | 11,331 |

Replicas scale linearly: 8.0× across all cases, with every GPU within 0.6%
of the others. Under load, the mixed-batch result reverses. Per-query bitset
searches overlap across streams, and each one uses its own query's
filtering rate. The UDF batch pays the most selective query's 0.99 rate for
all 32 queries, so it delivers half the throughput. Single-stream latency
(1.1× in the UDF's favour) hides this.

### Compilation

The first UDF search in a process took 36–586 ms. The longest time was the
first process on the host; later processes reused the CUDA driver's
ComputeCache (`~/.nv/ComputeCache`). Inside a process, new predicates and new
batches caused no further compilation, because the source never changes. Each
case's first search cost the same as a warm one.

## Findings

- **Uniform predicates: about the same speed.** Against a cached device
  bitset, the UDF is 0–9% faster for single queries (cuVS does not count bits
  in a bitset) and up to 8% slower at batch 16–32 (the interpreter loads per
  candidate). Recall is the same. Cached bitsets stay the right route for
  repeated filters at large batch sizes. The UDF needs no `rows / 8` bytes of
  device memory per filter.
- **Mixed predicates: the gain depends on k.** Against bitsets already
  cached per query, one UDF search for 32 queries is 8.2× faster at k=10
  (1.8 against 14.8 ms) but only 1.1× faster at k=500 (23.6 against
  26.1 ms). At k=10 the batch saves per-launch overhead. At k=500 with a 0.99
  rate, search effort dominates either way. Without cached bitsets, building
  them per query adds 27–45 ms per batch of 32, which the UDF avoids
  completely. At saturation (8 GPUs), per-query cached bitsets give 21,879 q/s
  against 11,331 for an unbucketed UDF batch at k=500. A mixed batch has to be
  split by filtering rate before the UDF pays off at large k.
- **The selectivity estimate decides recall at large k.** Without a rate,
  recall at 1% selectivity was 0.15–0.18 (k=500), with fewer than k rows
  returned. An estimate 10× too broad gave 0.59–0.66. At k=10 a 10× error cost
  little (0.997). `AnnUdfEstimate` assumes independent columns, so correlated
  predicates can make it too broad. The search's expense follows the rate too:
  a mixed batch costs as much as its most selective query, the same 23.6 ms
  as a uniform 1% batch. Bucketing a batch's queries by rate, and widening the
  estimate for multi-column predicates, are the next things to measure.

### Grouping a mixed batch by filtering rate

`udfgroup` (throughput tool) and `udf_grouped` (probe) split a mixed batch into
its window widths, laid out in contiguous query slots, and run one UDF search per
width at that width's filtering rate, instead of one search at the batch's highest
rate. k=500, FP32 graph, frontier 4096, 256 extra candidates:

| Mixed 1–10% batch of 32 | Latency (1 GPU, 1 stream) | 8 GPUs, saturated | Recall (mean / lowest) |
| --- | ---: | ---: | --- |
| Bitsets cached per query, a search each | 26.0 ms | 22.0k q/s | 0.9998 / 0.998 |
| One UDF search at the highest rate | 23.6 ms | 11.3k q/s | 0.9998 / 0.994 |
| One UDF search per rate group | **16.8 ms** | **28.7k q/s** | 0.9997 / 0.994 |

Grouping by rate is 2.5× the single-rate batch and 30% above per-query bitsets
under load, at the same recall. It reverses the finding above that per-query
bitsets beat the UDF under load: what cost throughput was searching every query at
the most selective one's rate, not the UDF. Raw results:
[`rategroup-probe.txt`](../benchmarks/ann/results/cagra-udf/rategroup-probe.txt),
[`rategroup-throughput.txt`](../benchmarks/ann/results/cagra-udf/rategroup-throughput.txt).

## Next steps for routing

1. Upload the attribute columns to each GPU at load, next to the dataset.
2. Add a batch operation with one program per query. Operation 30 shares one
   program across the batch.
3. For FP32 indexes, route a batch whose queries' estimated selectivity is
   above the gather limit to `AnnCagraUdfSearch`, grouped by `filtering_rate`
   bucket. Keep cached bitmaps for repeated filters, gather scans for
   selective ones, and BITMAP programs on the bitset path.
4. Calibrate a cost-profile entry for the route. The FP16 profile does not
   apply to FP32.
