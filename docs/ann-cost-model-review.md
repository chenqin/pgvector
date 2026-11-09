# Cost-model audit

Reviewed implementation and retained B200 results at `fae61f4`.
This audit adds no planner or worker routing changes. It corrects a validation
measurement boundary for future runs; existing benchmark artifacts are retained.
It also isolates calibration and fixed-plan controls from a profile enabled by
the database or role defaults.

## Current scope

The model chooses one of four complete search pipelines:

- NVFP4 scan followed by CPU FP32 rescoring.
- NVFP4 scan followed by GPU FP32 rescoring.
- FP16 CAGRA followed by CPU FP32 rescoring.
- FP16 CAGRA followed by GPU FP32 rescoring.

It estimates total request latency from matching rows, requested k and assembled
batch size. It does not separately estimate traversal, candidate selection,
rescoring, copies, queueing or PostgreSQL materialization. CAGRA's calibrated
frontier is 4096; candidate overfetch is 256, capped at 1024 total output
candidates for k below 1024. At k=500 both calibrated pipelines rescore up to
756 candidates per query per shard. The frontier is not the rescore count.

## 1. Isolated latency is the wrong sole objective at saturation

Calibration uses one GPU, one worker lane and warm caches. Deployment uses
eight GPUs and 64 lanes. `CalibratedFilteredGroup` selects minimum isolated
latency without pricing contention, queue depth, CPU service demand, or GPU
memory bandwidth occupied by concurrent scans.

At the 10% filter the measured policy changes CAGRA to NVFP4:

| Workload | Baseline SQL QPS | Profile SQL QPS | Baseline/profile p99 ms |
| --- | ---: | ---: | ---: |
| Single vector, 256 clients | 83,909 | 85,542 | 4.903 / 5.707 |
| 16 vectors, 64 clients | 8,333 | 8,316 | 10.836 / 13.361 |

Model worker CPU use rose from 47.85 to 57.71 cores in the single-vector case
and from 44.80 to 60.71 cores in the batch case. These single 15-second windows
do not isolate the cause or establish significance. They do demonstrate that
the lowest isolated latency is insufficient evidence of higher saturated QPS.
At 0.1% and 1%, both policies use NVFP4; the profile adds no new main GPU route.

Next experiment: calibrate CPU time, GPU time and bandwidth/queue effects
alongside latency, with repeated interleaved baseline/model runs. Record batch
shape and selected pipeline per request. Choose a throughput policy using those
service demands, with a latency policy retained for lightly loaded operation.
Do not infer queueing coefficients from these twelve stress windows alone.

## 2. Recall qualification creates a k=501 routing cliff

The grid contains k=50, 500 and 1000. A failed corner excludes a route throughout
the surrounding interpolation interval. Exact replay of the shared C model on
the saved profile, for 10M matching rows, gives:

| Batch | k=500 selected/predicted ms | k=501 selected/predicted ms |
| ---: | --- | --- |
| 1 | CAGRA GPU / 0.682 | NVFP4 CPU / 1.791 |
| 16 | CAGRA GPU / 1.971 | NVFP4 GPU / 26.100 |
| 32 | CAGRA GPU / 2.233 | NVFP4 GPU / 51.984 |

These are model predictions, not new latency measurements at k=501. CAGRA's
k=1000 mean recalls were 88.50%, 93.95% and 94.48% for batches 1/16/32;
excluding those samples is justified. It does not establish that k=501 fails.
The 1024-output cap also shrinks effective overfetch from 256 at k=500 to 24
at k=1000, introducing a real algorithmic change that a smooth k coordinate
cannot represent well.

Next experiment: add k near 500/501, 750, 768/769 and 1000, explicitly recording
actual candidate count. Qualify separate frontier/candidate-budget variants
and select the cheapest variant meeting the target. Preserve conservative
qualification until those points are measured.

## 3. Planner conversion depends on the competing local plan

The planner currently computes:

```
T = estimated_worker_us + 257.117499 + min(k, estimated_rows) * 0.615052
frozen_startup_cost = current_best_local_cost * T / calibrated_local_us
```

It then adds normal per-row PostgreSQL costs. `current_best_local_cost` includes
the local sort and LIMIT. The calibrated local microseconds are tied to the
benchmark's predicates, projection, indexes and cache state.

A new btree can reduce current local planner cost while calibrated local time
remains unchanged. The formula then reduces frozen startup cost too, even
though GPU work has not become cheaper. This weakens the intended comparison
against the improved local plan. Other projections or more expensive residual
clauses also need different SQL overheads. The fitted return overhead overlaps
with some of the standard row work added again; the exact overlap needs a
component measurement before changing the formula.

Next change: use a fixed calibrated conversion between microseconds and planner
cost units, independent of which competing path currently wins. Keep worker
setup and PostgreSQL per-row work distinct. Validate crossover plans before and
after adding/removing a selective index, and for different projections.

## 4. Validation and calibration timed different work

`calibrate_cost.py` stops the clock after receiving the worker response, before
Python decodes its records. The original `validate_cost.py` stopped after that
decoding. Direct comparison therefore included Python overhead in only one
side. In the retained independent validation, the median actual/predicted ratio
was 1.26; the maximum was 3.17. Those ratios cannot all be attributed to worker
prediction error.

The validation script now records comparable worker round-trip `us`, separate
`decode_us`, and decode-inclusive `client_us`, with raw samples and timing
version 2. Historical `validation-native.json` and `unseen-native.json` still
contain decode-inclusive `us`; they have not been rewritten or remeasured.
The earlier 22.24% median timing error and 48.45% maximum route regret came from
12 calibration holdout points with the same original timing scope on both
sides, and are unaffected by this correction.

Calibration now explicitly clears the session planner profile, and validation
clears it for the fixed-policy SQL phase. Previously both could inherit a
database/role profile, contaminating a future recalibration or its baseline.
The retained runs preceded database-level profile activation and are unaffected.
The timing fix was syntax-checked and the decoder exercised locally; the GPU
validation has not been rerun with the new timing fields.

## 5. Eligibility and quality are deployment assumptions

The worker checks generation, dimensions, row count, route representations,
FP32 device residency and selected deployment knobs. The planner assumes both
scan/graph and CPU/GPU choices are available; a catalog registration is not a
live readiness check. A backend/configuration mismatch can therefore make the
planner optimistic even when execution safely uses a different route.

Requests outside the measured grid or incompatible worker settings use legacy
routing. The profile's 95% label is consequently a sampled mean-recall target,
not an enforced guarantee on all requests or individual queries. Matching-row
count alone also cannot describe how a filter's IDs are distributed across the
graph, which can affect traversal and recall.

Next change: publish an immutable worker capability/profile fingerprint outside
planning, cache it in PostgreSQL, and expose fallback reasons plus per-request
route/budget statistics. Calibrate multiple filter distributions, not only
cardinality. Avoid a planner RPC on every query.

## 6. Two vector fields with AND are outside this model

The frozen path selects one vector ordering expression and requires supported
scalar predicates for its bitset program. It does not compose two ANN indexes.
Both the session and worker also hold one generation-bound profile, so distinct
exports need a profile registry before their costs can be compared together.
The current model has no costs or quality qualification for these alternatives:

- Search field A, then compute field B distances for those candidates.
- Search B, then compute A distances and restore the requested final order.
- Search both, intersect candidate IDs, and rescore survivors.
- Scan both vector fields over a selective metadata subset.

Choosing among them needs candidate overlap and conditional selectivity, not
just each field's matching-row estimate. For an A-first plan, the useful yield
is approximately `candidates_A * P(B passes | A candidate)`; substituting an
independence assumption can severely underfetch. Intersecting two fixed top-k
candidate lists is not exhaustive AND-threshold search. Adaptive expansion and
joint recall validation are necessary for an approximate implementation.

## Evidence and reproduction

Raw calibration, quality, plans and stress records are in
[the retained B200 results](../benchmarks/ann/results/frozen-cost-b200-95/manifest.json).
The matching implementation is in [cost.h](../ann/common/cost.h),
[annfrozenscan.c](../src/annfrozenscan.c), and [cuvs.c](../ann/server/cuvs.c).

```sh
cc -O2 -std=c11 -Iann benchmarks/ann/cost_predict.c -lm -o /tmp/cost-predict
printf '10000000 500 32\n10000000 501 32\n' | /tmp/cost-predict \
  benchmarks/ann/results/frozen-cost-b200-95/profile.tsv
```

Priorities are consistent timing first, denser qualification around candidate
budget changes second, then independent planner conversion and load-aware
costs. Multi-vector plans require a separate extension of the search and
quality model.
