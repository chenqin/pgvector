# Native FP32 CAGRA

FP32 CAGRA indexes use their returned distances directly. The worker does not
run a CPU or GPU distance-rescoring pass after graph search. FP16 CAGRA also
uses its own distances unless the worker runs with `--fp16-exact on`
([FP16 distances](ann-cuvs-index.md#fp16-distances)); compressed-vector scans
retain rescoring.

```mermaid
flowchart LR
    A[PostgreSQL query] --> B[Scalar filter bitset]
    B --> C[FP32 CAGRA graph search]
    C --> D[Candidate IDs and squared L2 scores]
    D --> E[Square root and shard merge]
    E --> F[Visibility checks and final top k]
    F --> G[Content and metadata lookup]
    G --> H[Enriched SQL response]
```

## Selection and distances

Build the graph from original FP32 vectors:

```sh
python ann/builder/build_cuvs_index.py \
  --from-export "$EXPORT_DIR" --out "$FP32_INDEX_DIR" --dtype float32 --shards 1
```

The graph branch is automatic for a `dtype float32` manifest. Both
`--gpu-rerank on` and `--gpu-rerank off` use native graph scores. The same
index can still use filtered gather scans; force graph search in an experiment
with `--exact-limit 0 --gather-limit 0 --bitset-gather-limit 0
--group-gather-limit 0`.

Loading also verifies the serialized graph's floating-point datatype against
the manifest. A mislabeled FP16 graph is rejected before it can use this path.

Our builder uses squared Euclidean distance. The worker takes the square root
of valid CAGRA scores, clamping negative roundoff to zero, to return Euclidean
distances in the protocol and SQL. It keeps eligibility checks, invalid-ID
removal, deterministic distance/key sorting and shard merging. FP32 rounding can
differ from PostgreSQL's distance arithmetic, especially for nearly identical
vectors. Native FP32 scores are not a promise of bit-identical CPU results.

`--filtered-slack` can still request extra graph candidates. FP32 candidates
already have their distances; the worker selects the final k without rescoring
their vectors. Increasing the graph frontier, search width, iteration budget
or candidate count changes search effort and can affect recall. Graph traversal
remains approximate even when distance storage and computation use FP32.

## Verification

Routing statistics expose:

- `cagra_fp32_queries`: queries answered using native FP32 CAGRA distances.
- `graph_cpu_rerank_candidates`: candidate distances recomputed on the CPU for
  graph searches.
- `gpu_rerank_candidates`: candidates rescored on the GPU, including scan paths.

On an isolated worker forced to use FP32 graph search, the first counter rises
while both rescoring counters stay zero. A `rerank_ready` pointer can still
exist because gather scans can use the same FP32 dataset; it does not mean
CAGRA performed a rescoring pass.

The GPU regression tests cover four-shard filtered and unfiltered searches,
single queries, batches, GPU-rerank enabled/disabled, recall, distance units,
ordering and zero rescoring counters. The FP16 and quantized regression tests
also remain active.

## Cost profiles and benchmark

The existing v5 profile describes NVFP4 scanning and FP16 CAGRA with rescoring.
Worker dispatch excludes FP32 CAGRA from that profile. PostgreSQL does not yet
have a dataset-dtype capability in its catalog: when testing an FP32 rebuild
of the same content generation, clear the old planner profile in that session:

```sql
SET ann.frozen_cost_profile = '';
```

A native FP32 cost profile requires a separate calibration; do not reuse the
FP16 coefficients as a measurement of this path.

`benchmarks/ann/benchmark_fp32.py` compares an older worker doing CPU/GPU
rescoring with the new worker using native scores, all on the same FP32 graph
and configured candidate budget:

```sh
python benchmarks/ann/benchmark_fp32.py \
  --index-dir "$FP32_INDEX_DIR" --baseline-worker "$OLD_WORKER" \
  --truth "$TRAINING_TRUTH_JSON" --output fp32-results \
  --itopk 4096 --slack 256
```

The benchmark uses the 10M-row stress schema, k=500, one GPU and one worker
lane, batch sizes 1/16/32, three warmups and fifteen timed samples. Worker
round-trip timing excludes Python result decoding. Recall uses original-vector
FP64 references for every query in the request. The script records quality;
its completion marker does not imply that every case met a recall target.
These measurements describe isolated latency, not client SQL QPS or saturated
throughput.

## B200 results

[Raw results](../benchmarks/ann/results/frozen-fp32-native/manifest.json) retain
the no-overfetch 4096/8192-frontier runs, a search-budget probe and the final
4096-frontier/256-extra-candidate comparison. Representative final medians:

| Matching rows | Batch | CPU rescore (ms) | GPU rescore (ms) | Native FP32 (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 100,001 | 1 | 1.592 | 1.543 | 1.502 |
| 1,000,019 | 16 | 4.330 | 3.587 | 3.629 |
| 1,000,019 | 32 | 7.554 | 5.617 | 5.531 |
| 10,000,000 | 1 | 0.699 | 0.656 | 0.635 |
| 10,000,000 | 16 | 3.178 | 1.995 | 1.771 |
| 10,000,000 | 32 | 4.328 | 2.568 | 2.266 |

The median speedup across all twelve points was 1.09x against CPU rescoring
and approximately 1.00x against GPU rescoring. Search dominates selective
filters; removing an already cheap GPU rescoring pass does not guarantee a
large latency gain. These are sequential mode comparisons without confidence
intervals, not a saturated-throughput claim.

All twelve native request cases had correct cardinality and at least 95% mean
recall on the sampled vectors. The weakest individual query was 53.8%: this
is not a per-query recall guarantee. Without candidate overfetch, some broad
cases fell below 95% mean recall even at frontier 8192. Keeping more CAGRA
candidates changes traversal effort; it does not require a rescoring stage.

The final native worker processed 3,528 benchmark vectors with zero CPU graph
rescoring and zero GPU rescoring candidates. The SQL smoke test returned 500
rows with maximum Euclidean-distance error below 1.8e-6 against PostgreSQL.

## GPU test endpoint

A separate eight-lane FP32 worker is loaded on GPU 7 at
`/home/nvadmin/pgvector-test.tKp5o0/run/fp32-native.sock`, using the same 10M-row
source and export generation. It forces graph search with frontier 4096,
filtered width 4 and 256 extra candidates. The default database worker and its
FP16 cost profile are unchanged. To use the FP32 endpoint in a test session:

```sql
LOAD 'vector';
SET ann.worker_socket = '/home/nvadmin/pgvector-test.tKp5o0/run/fp32-native.sock';
SET ann.frozen_cost_profile = '';
SELECT s.*
FROM frozen_stress_queries q
CROSS JOIN LATERAL ann_frozen_search_batch(
  'frozen_stress', ARRAY[q.v::vector], 500, 'category BETWEEN 0 AND 9'
) s
WHERE q.q = 129;
```

Deployment metadata and logs are under
`/home/nvadmin/pgvector-test.tKp5o0/fp32-native/`. The endpoint is experimental;
its measured recall and latency depend on the query, filter and search budget.
