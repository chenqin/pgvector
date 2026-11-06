# Measured frozen-search cost model

The optional cost profile connects PostgreSQL planning and worker route
selection. It compares NVFP4 scanning and CAGRA, each with CPU or GPU FP32
reranking. The current B200 calibration targets **95% recall@k** on its
reference samples. This is a measured ANN target, not a per-query guarantee.

PostgreSQL uses `ann.frozen_cost_profile = '/absolute/path/profile.tsv'`.
The worker uses `--cost-profile /absolute/path/profile.tsv`. Both settings
are opt-in. Loading a profile does not make a worker RPC during planning.
PostgreSQL checks its cached catalog registration for the matching export.
Load the profile before preparing statements; use `DISCARD PLANS` after
replacing it. Restart the worker to load a replacement worker profile.

## Calibration

The simulation runs real requests against isolated, single-GPU workers on
the existing 10M × 128 NVFP4/FP16-CAGRA export. It changes filter cardinality,
k=50/500/1000, and batch size 1/16/32. Each point has two warmups and seven
timed samples. CPU-only PostgreSQL controls use the same predicates and
vectors, with frozen routing disabled. Btrees on `(tenant,category)` and
`(category)` give local execution a selective alternative to a table scan.
The worker load uses the binary protocol and does not register its temporary
loaded name in the database catalog or change the deployed worker setting.

CAGRA calibration uses a 4,096-entry frontier and overfetches up to 256
extra candidates before FP32 reranking. When overfetching, the total output
is capped at 1,024 candidates for the current cuVS selector. This improves recall before the
final k is selected. The original threshold policy retains zero overfetch
unless `--filtered-slack N` is explicitly set.

The profile retains median times in microseconds per request for four
routes and local SQL. It also fits a nonnegative SQL setup/return overhead
from measurements against the deployed service. Empty predicates are kept
in the raw results but excluded from the model grid. Correlated predicates
with the same cardinality are combined; a route remains excluded if a
corresponding calibration sample failed cardinality or recall qualification.

Interpolation checks use different vectors, intermediate filter sizes, k=250/750
and batch sizes 4/8. Exact references use original FP32 vectors widened to
FP64, never NVFP4 scores or CAGRA candidates. The reference is cross-checked
against PostgreSQL. Those vectors also informed the CAGRA budget tuning; a separate final validation
uses vectors starting at 129. Runtime validation reports mean recall across the
vectors of each request, plus the minimum individual recall and cardinality.

```sh
CUDA_VISIBLE_DEVICES=7 python benchmarks/ann/cost_truth.py \
  --index-dir "$INDEX_DIR" --query-start 1 --calibration-predicates \
  --output calibration/training-truth.json
CUDA_VISIBLE_DEVICES=7 python benchmarks/ann/cost_truth.py \
  --index-dir "$INDEX_DIR" --output calibration/holdout-truth.json
python benchmarks/ann/calibrate_cost.py --index-dir "$INDEX_DIR" \
  --service-socket "$SOCKET" --gpu 7 --min-recall .95 --output calibration \
  --graph-itopk 4096 --graph-slack 256 \
  --truth-training calibration/training-truth.json \
  --truth-holdout calibration/holdout-truth.json
cc -O2 -std=c11 -Iann benchmarks/ann/cost_predict.c -lm -o /tmp/cost-predict
printf '100000 500 16\n' | /tmp/cost-predict calibration/profile.tsv
python benchmarks/ann/validate_cost.py --index-dir "$INDEX_DIR" \
  --profile "$PWD/calibration/profile.tsv" --truth calibration/holdout-truth.json \
  --output validation
CUDA_VISIBLE_DEVICES=7 python benchmarks/ann/cost_truth.py \
  --index-dir "$INDEX_DIR" --query-start 129 --output calibration/unseen-truth.json
python benchmarks/ann/validate_cost.py --index-dir "$INDEX_DIR" --query-start 129 \
  --profile "$PWD/calibration/profile.tsv" --truth calibration/unseen-truth.json \
  --output validation-unseen
```

These scripts target the stress guide's table/query schema. `--resume` on
the calibrator rebuilds a profile from saved samples without rerunning the
workers. `--continue-run` collects missing points after a stopped run, using
the saved local controls and exact references. Raw timing and qualification
samples remain available for review.

## How planning uses the profile

Profiles contain bounded grids, rather than a universal scan coefficient:
GPU selection changes algorithms with k and batch size, and CAGRA cost is
not linear in matching rows. Interpolation uses log latency and log
matching-row/k/batch coordinates. It never extrapolates or interpolates a
route through an unqualified sample.

The planner uses PostgreSQL's estimated matching rows and the known LIMIT.
It estimates the best worker time plus calibrated SQL overhead, then
converts that time to the query's local planner cost units using the
measured local SQL time. The local cost includes the sort and LIMIT that
upper planning has not yet added. Normal tuple/projection costs remain.
`EXPLAIN` exposes the estimated route, search/local times, matching rows and
calibration recall target. In the SQL validation, narrow correlated predicates
used local btree-plus-sort plans; intermediate filters chose NVFP4 with CPU
or GPU reranking, and broad filters chose CAGRA. The worker may revise the
planned route when actual cardinality or batch shape differs.

Unsupported projections, joins, unknown LIMITs, missing loaded exports,
wrong generations/dimensions/row counts and points outside the measured
grid retain the existing cost model. `ann.frozen_setup_cost` controls that
legacy branch. Estimated selectivity can still be inaccurate, particularly
for correlated attributes and generic range parameters.

## How execution uses the profile

The worker makes a fresh decision from the **actual** predicate cardinality
and the batch it has assembled. It checks matrix/bitset availability and
whether FP32 reranking vectors are resident. A batch-local choice selects
CPU or GPU reranking without changing shared configuration. Predicate
caching keeps the representations needed by the measured routes; its
existing memory budgets and eviction still apply. Routing statistics expose
`cost_queries` in scan-CPU, scan-GPU, CAGRA-CPU, CAGRA-GPU order.

The current worker calibration is for one-shard NVFP4 exports with FP16
CAGRA, deployment itopk=512, filtered-itopk=1024, multi-CTA width=4, quant-slack=256,
zero explicit filtered iterations, persistent search off, filtered-slack=0,
and a 64 MiB score-matrix cap. The profile records its own larger CAGRA
frontier budget, applied only to calibrated CAGRA batches, so quality and
cost are measured with that budget. Different deployment settings use the existing worker
routing rules. A missing FP32 device copy excludes GPU reranking and keeps
the CPU option available.

Recalibrate for different hardware, SDKs, graph builds, dimensions, candidate
budgets or cache conditions. A matching content generation alone does not
prove that those other deployment conditions are unchanged. Warm isolated
latency and saturated throughput are different objectives; both are tested
before selecting a deployed profile.


## B200 measurements

The retained [results and raw samples](../benchmarks/ann/results/frozen-cost-b200-95/manifest.json)
contain 372 route measurements and a 63-cell grid. The fitted SQL overhead
is **257.12 µs per call + 0.615 µs per returned row**. For k=500, representative
single-GPU request medians are:

| Matching rows | Batch | NVFP4 + GPU rerank (µs) | CAGRA + GPU rerank (µs) |
| ---: | ---: | ---: | ---: |
| 10,000 | 1 | 260 | 6,076 |
| 1,000,019 | 1 | 404 | 689 |
| 5,000,099 | 1 | 983 | 866 |
| 10,000,000 | 1 | 1,818 | 682 |
| 1,000,019 | 16 | 1,534 | 3,187 |
| 5,000,099 | 16 | 7,776 | 2,038 |
| 10,000,000 | 16 | 26,099 | 1,971 |

The crossover depends on batch size and k, rather than one filter-rate
threshold. On 12 intermediate cases the model covered all cases, with
22.24% median relative timing error, zero median route regret and 48.45%
maximum regret against the fastest measured qualified route. These are
latency estimates with meaningful error, not a guaranteed optimal route.

The first 90-case route validation reached at least **99.23% mean recall**
per request, with correct cardinality in every case. The lowest individual
query recall was **80.67%**. The original fixed policy's lowest request mean
was 83.8%. On the final independent vectors (129–160), all 90 calibrated
cases passed cardinality and 95% mean recall: the lowest request mean was
**99.24%**, and the lowest individual query was **96.27%**. The target and qualification apply to request means; applications
requiring 95% for each query need stronger qualification or exact search.

End-to-end SQL stress used 8 replicas, 64 worker lanes, k=500, a warm tmpfs
index and 15-second measurement windows. Each SQL batch request contains
16 vectors; batch QPS below counts SQL calls, not individual vectors.

| Filter | SQL clients / batch | Baseline SQL QPS | Profile SQL QPS | Baseline / profile p99 (ms) |
| --- | --- | ---: | ---: | ---: |
| 0.1% | 256 / 1 | 145,124 | 143,660 | 2.788 / 2.802 |
| 1% | 256 / 1 | 133,360 | 134,731 | 3.591 / 3.687 |
| 10% | 256 / 1 | 83,909 | 85,542 | 4.903 / 5.707 |
| 0.1% | 64 / 16 | 15,346 | 15,071 | 5.395 / 5.560 |
| 1% | 64 / 16 | 14,428 | 14,346 | 5.691 / 5.696 |
| 10% | 64 / 16 | 8,333 | 8,316 | 10.836 / 13.361 |

There were zero query failures. Batch vector throughput was 241,137,
229,543 and 133,056 vectors/second with the profile. All measured calibrated
stress requests chose NVFP4 with GPU reranking. Overall throughput is
essentially unchanged, and the 10% case has worse tail latency; these single
windows do not establish statistical significance. Worker CPU usage also
rose, so isolated service latency does not predict saturated efficiency.
The profile remains opt-in. These results support cost-based route selection
and recall qualification, but do not establish a saturated throughput gain.
