# Filtered ANN throughput: exploration roadmap

Status: this roadmap comes from a read-only review of commit 4f18cfe (branch `matcha-patches`). Every number comes from `docs/ann-10pct-throughput.md`, from `benchmarks/ann/results/frozen-10pct-throughput/runs.jsonl`, or from the code citations given. A number marked *estimate* was worked out during the review and was not measured. Sections 1 to 6 describe the state at 4f18cfe; what was built afterwards, and what it measured, is under [Results so far](#results-so-far).

**Built as of 2026-11-07 (4a9e2c4):** P01 as `--residual-masks` (default on), P07 (ordinals-only entries), P08 step 1 (lane rows expanded from bitsets), P03's key half (result keys gathered on the GPU), P04 (no candidate fill, no mid-batch sync, final top-k in shared memory), P02 (keys for immutable batch searches, with result arrays packed in place), experiment 4 (the cached point was bound by the lanes' host work, including RAFT's yield-polling stream sync), and harness prerequisites 1-2. The cached `10pct` arrays point went from 775.6k to 973.5k vectors/s, and `distinct10` from 156.2k to 765.2k vectors/s and from 4.55k to 98.6k QPS. Not built: G1a, G4a, G3a, P14 and the rest of P03; the selection stage (47% of GPU time) is the next target.

Benchmark case: 10M rows x 128 dims, a predicate keeping 10% of rows, k=500, 8 B200 GPUs, recall floor 0.90. Recall is 0.99 or better at every measured point today.

## 1. Where the time goes today

**Cached predicates (10pct, 64-vector arrays, immutable).**
- Throughput: 775-783k vectors/s, about 12.2k calls/s (`mg-immutable2-b64`: 780,404 vectors/s, 12,194 calls/s).
- CPU: PostgreSQL uses 108.5 logical cores, about 8.9 ms of backend CPU per call (278 ns per returned id). The worker uses 50.8 cores, about 4.2 ms per batch.
- GPU: 79.9% busy (NVML). Dividing busy time by the batch rate gives about 524 us per 64-query batch (*estimate*). NVML counts time when any kernel is running, with about 12 lanes sharing each GPU. It is not exclusive kernel time, and no per-stage timing of `SearchGatherMatrix` exists.
- Host:
  - `docs/ann-k500-stress.md:7` gives 224 logical CPUs on two Xeon Platinum 8570 sockets. The 112-physical-core figure is inferred from the CPU model, not measured.
  - Nothing records which logical CPUs are idle.
  - pgbench CPU is not in `runs.jsonl`. `frozen_stress.py` does record it (`Monitor.record('client', ...)` at `:97-100`, returned at `:141`, written to the per-tag JSON at `:337`). `baseline-deployed.json:318-326` shows 6.73 client cores at 86.2k tx/s.
- **The limiter is not identified.** Two hypotheses are open:
  - **H1: physical cores are saturated** (SMT co-run slowdown). Some runs fit it. Across the 96-lane cached points, PostgreSQL CPU per call goes from 8.73 to 9.07 ms (+3.9%) as total logical load rises from 153.5 to 160.6 cores. That is close to run-to-run noise. The larger rise (9.70-9.77 ms) comes only from the 128-lane / 560-client runs, which also change lane and client counts, so they are confounded.
  - **H2: a worker/GPU per-lane pipeline limit.** `docs/ann-10pct-throughput.md:46` names "mid-pipeline sync, host result handling" with "~57 host cores idle" at 780.4k. Lane wall time rising with lane count (7.9 ms at 96 lanes, 10.8 ms at 128) also fits 12-16 lanes queueing on each GPU behind about 0.5 ms batches and two blocking host syncs (`cuvs.c:3784, 4014`; `cudaDeviceScheduleBlockingSync` at `cuvs.c:104-105`).

  More concurrency did not help under either hypothesis: 128 lanes gave 759.3k at 81.8% GPU busy, and 560 clients added nothing. Every item in this review that targets the cached point was cut by its reviewers to single digits, because nobody knows which hypothesis holds.
- Other cached points:
  - **Single-vector SQL:** 106-117k QPS at 96 lanes and 256 clients (GPUs 94-96%, 1.5-1.8 queries per batch); 177.3k at 32 lanes and 512 clients (PostgreSQL 117.1 cores, 0.66 ms per query, GPUs 71.1%). Gather b1 at 32 lanes is flat across eligible-row counts: 180.3k at 10k rows, 175.8k at 1M.
  - **k=1500:** 254.6k vectors/s (PostgreSQL 125.7 cores, GPUs 51.2%, 28.52 queries per batch).
  - **unique1:** about 930k vectors/s (PostgreSQL 145.8 cores, GPUs 47.9%).
  - **0.1% tenant, 32-vector arrays:** 949.5k vectors/s (PostgreSQL 152.5 cores, GPUs 66.6%).
  - **Broad filters:**
    - 2M rows, k=500: gather b32 450.5k, lowest recall 0.992-0.996.
    - 4M rows, k=500: only the graph was run, 766.3k b32, with lowest-query recall 0.29-0.53.
    - 4M rows, k=100: graph 221.7k (b1, lowest 0.99), 884.1k (b16, 0.87), 1,205.9k (b32, 0.85). The gather at 4M disables pruning and selects over all scores: 31.6k b1, 45.8k b16/b32, measured at k=100 only.

**Uncached predicates.**
- Canonicalisation now folds fresh10 to 773.7k.
- Truly distinct predicates stay slow:
  - distinct10 (`category BETWEEN a AND a+9 AND tenant <> w`): 156.2k vectors/s, 2,441 calls/s, 4.55k QPS on SQL. Hit rate 0.0073. In 30 s: 74,649 evaluations and 74,355 evictions.
  - mix10: 248.8k vectors/s.
- CPU per call on distinct10:
  - The worker uses 91.4 cores, about 37 ms of worker CPU per call (134.5 cores, about 30 ms per call on SQL).
  - PostgreSQL uses 39.4 / 2,441 = 16 ms per call, against 8.7 ms on the cached path. fresh10 with `--gpu-predicates` shows 15 ms. The cause is unexplained.
- The only uncached profile is fresh10 with gprofng. It was taken on build 7501378, before the a0d4a81 device-leak fix and before canonicalisation. It puts 58.6% of worker CPU in `AnnIndexBitmapPut`, of which 33.8% is the `file->keys[ordinal]` line, and 27.5% in lane row staging. Applying these shares to distinct10 at 4f18cfe is an assumption: no distinct10 profile exists.
- Nothing visible is saturated on distinct10:
  - worker 91.4 + PostgreSQL 39.4 of 224 logical cores;
  - GPUs 39.2%;
  - mean call latency (Little's law) 183 ms, p99 2.5 s.

  The summed CPU per call (about 46-53 ms) includes work done in parallel on several lanes and reader threads, so it is not critical-path time. The gap between the two is therefore not a measured wait. What could add latency is still worth measuring: `bitmapMutex` held across the eviction qsort, the pageable D2H in `DevicePredicate`, host memory bandwidth, and queueing.

## 2. Recommended changes

| Rank | Id | Title | Bottleneck | Workload helped | Expected gain after review | Effort | Recall risk | Cheapest validation |
|---|---|---|---|---|---|---|---|---|
| 1 | P01 | Cache the base conjunct only; mask the residual conjuncts on the GPU inside the scan | Worker CPU (entry build, churn, staging) | distinct10, mix10 and similar AND/NOT predicates | distinct10 arrays 156.2k -> ~600-750k vectors/s (*estimate*). Only the ceiling is measured: 764k in `fin-mixed10-arrays-gpoff`. The low end applies if PostgreSQL's uncached 16 ms per call does not fall back to 8.7 ms. mix10 248.8k -> ~650-750k (*estimate*). distinct10 SQL 4.55k -> ~65-95k QPS (*estimate*; assumes about 1 query per batch on GPUs that are already busy, unmeasured). Worker 91.4 -> ~48-65 cores (*estimate*). Cached 10pct: 0 | L (possibly L-XL) | Low, provided every grouping site and fallback applies the residual (no test covers this today) | 30-line hack: look up only the category conjunct and ignore the residual (wrong answers, right throughput). Run distinct10 arrays for 30 s and expect about 700k |
| 2 | P07 | Drop host `keys[]` from predicate cache entries | Worker CPU (possibly host memory bandwidth, unmeasured) | Predicates that cannot be decomposed (any uncached result) | distinct10 arrays 1.07-1.5x, most likely 1.1-1.4x: ~170-220k vectors/s (*estimate*). SQL 4.55k -> ~5.0-6.5k QPS (*estimate*). Host bytes per 10% entry 17.25 -> 9.25 MB | S, with a longer change list than proposed | None | Skip the `keys[]` build in `BuildBitmap`, split the `server.c:2131` gate on `ordinals`, fix `BitmapBytes`, keep the matrix route. Run distinct10 for 30 s and record worker CPU per predicate, mean latency, memory bandwidth and lock waits |
| 3 | G1a | Clustered worker export by (category, tenant) with range-form predicate jobs | Worker CPU (evaluation, entry build, staging, device entries) | Uncached predicates on the declared cluster key (distinct10, mix10 on the benchmark data) | distinct10 arrays 156.2k -> ~440-775k vectors/s (*estimate*): 440k if PostgreSQL stays at 16 ms per call, 775k if it falls to 8.7 ms. mix10 to the same ceiling. distinct10 SQL 4.55k -> ~10-70k QPS (*estimate*; reviewers disagree, unmeasured). No device predicate memory for cluster-key predicates. Cached 10pct, fresh10: 0 | L for the range-job core; the full permutation is larger than stated | None | Range jobs (steps 3-4) on a table whose key order equals cluster order and whose joint (category, tenant) distribution matches the benchmark. Run distinct10/mix10 arrays and check that the worker falls to ~50 cores. Kill below 400k with the worker under 60 cores |
| 4 | G4a | Per-vector predicates in one batch call (`vector[]` + template + `params int4[]`), served by a composite-confined segmented gather | PostgreSQL per-search cost, protocol | Multi-tenant traffic where each vector carries its own small predicate (off-benchmark) | Distinct 1% windows per vector: 146.9k (best measured single-vector SQL, `--mixed-groups on`) -> ~180-370k vectors/s (*estimate*; depends on the unmeasured fixed cost of the segmented route). PostgreSQL CPU per vector ~160 us vs 530-660 us on single-vector SQL (measured on one-predicate arrays). The 0.1% projection (570-970k) is unsupported. Benchmark case: 0 | L (or more) | Low | Zero-code GPU anchor: rerun `u-unique1-on` with 8-16 lanes and 1,024 clients so mixed groups reach 32-64, and compute GPU-us per query at m = 6..64. If m=64 stays above ~25 us/query, the 1% case dies |

Items 1-3 target the uncached regime. Item 4 targets a workload the benchmark does not contain. No surviving proposal materially raises the cached 10pct point. Section 6 lists the measurements that should come before any build aimed at that point.

## 3. Recommended changes in detail

### P01: base conjunct plus residual mask on the GPU

**Mechanism.**
- After `AnnIndexCanonical` in `PredicateBitmap` (`ann/server.c:2253-2313`), split a top-level AND into a BASE and a RESIDUAL:
  - BASE: one single-column run, which is today's cached entry, or a postings slice.
  - RESIDUAL: the remaining leaves.
- Only the BASE is looked up and cached. The job carries the residual program and its hash.
- The lane masks residual-rejected rows on the GPU in one of two ways:
  - (a) copy the base's device rows into `lane->gatherRows` (a 4 MB D2D) and overwrite rejected entries with `0xffffffff`;
  - (b) test the residual inside `matrix_compact`.
- Rows set to `0xffffffff` already score +inf in three places: `score_mma` (`ann/blackwell.c:338-343, 468`), the gather kernel (`cuvs.c:2392`) and rerank (`cuvs.c:2555`).
- `matrix_compact` keeps only scores at or under the sampled bound (`cuvs.c:2436-2452`), so masked rows never reach the candidate set.
- The residual itself can run on the existing UDF interpreter (`DevicePredicate`, `cuvs.c:1615-1697`) over device-resident columns (`PredicateDeviceColumn`, `cuvs.c:1594-1612`).
- Masked predicates create no cache entry.

**Evidence.**
- distinct10's per-call cost is entry construction and churn: 2,441 calls/s, worker 91.4 cores, 74,355 evictions in 30 s.
- The target rate is already measured. The same 91 category windows, cached, run at 11,942 calls/s = 764k with the worker at 48.6 cores (`fin-mixed10-arrays-gpoff`). fresh10 with canonicalisation runs at 773.7k.
- What the mask adds (*estimates*, unmeasured):
  - a D2D copy of about 2-3 us;
  - one kernel over 1M threads of about 10-20 us;
  - against about 500 us of GPU time per batch, itself an NVML-derived *estimate*.

  Other per-call costs are not costed: residual compile, larger `FilterEntry` keys, and warming 91 bases.

**What the reviewers corrected.**
- **Base selection is inverted.** `tenant <> w` canonicalises to a K_SET over the complement, about 9.99M postings (`canonical.c:378-396`). "Largest postings count" would pick it as base and fail its own 25% retention gate. The base must be the smallest-count conjunct.
- **The residual is larger than claimed.** It removes about 1,000-1,430 rows of a 1M base, not about 100. Only 7,000 of the 100,000 (category, tenant) cells are non-empty, about 1,428 rows each (`cuvs_load.py:112-113`). Host binary search for these rows is about 0.5-1 ms per group, so mask on the device.
- **Several proposed steps are unnecessary:**
  - scaling `ANN_PRUNE_RANK` (the sampled cutoff already sorts +inf last);
  - the starvation check (validation step 1);
  - an eligible-count atomic (the base count is harmless at k << 1M).
- **The canonicaliser needs work.** `AnnFilterCanonical` returns serialized bytes and frees its tree (`canonical.c:720-788`). The split needs a re-parse or a new API.
- **Grouping is keyed on entry pointers in more places than proposed.** All of these need the residual hash:
  - `FilterFor` and `SameFilterJob`;
  - `Shareable` (`server.c:814`);
  - `FilterBucket` (835-844);
  - `Append` (929-966);
  - `SettleAdd` (609-636);
  - the lane same-filter pull (1079-1086).

  Otherwise a group is masked with `jobs[0]`'s residual and returns wrong rows. No unit test exercises these sites (`server.c` is not linked into `test/ann/*`), so "the result is exact" holds only once the sites are covered.
- **Single-vector SQL estimate lowered from ~100k to 65-95k QPS.** Distinct (base, residual) pairs never share a group, so the per-query mask lands on GPUs that are already busy. This figure is unmeasured.
- Attribution: the 58%/27% split comes from the fresh10 profile on an older build, not from distinct10.

**Hidden costs.**
- New `AnnIndexJob` fields require `make clean`.
- Fallbacks need the residual: `AnnIndexPositionsDistinct` -> `AnnIndexExact` (`cuvs.c:4027-4035`) and `SearchGatherMixed` must apply it, or exclude residual jobs.
- The split must be gated to search operations. Op 25, precompute and PutBitmap must keep caching the full predicate.
- Residual retention is estimated from marginal per-value counts, which cannot see cross-column correlation. A residual far more selective than estimated lands in the full `select_k` fallback (`cuvs.c:3789-3797`), which is correct but slow.
- Repeating ANDs with 25-50% retention would never be cached, a regression outside distinct10/mix10. They need promotion on repeat sightings.
- The 91 bases may hold up to about 3.8 GB of device memory (*estimate*). That assumes every base gets the deferred upload to all 8 copies at 42 MB each. The `--predicate-device-mb` setting in force in the benchmark is not cited here.

**Dependencies.** Requires canonical predicates on (the default), an nshards = 1 NVFP4 index and `--matrix-prune on`. For distinct10/mix10 it subsumes P07, P08 and P10.

### P07: drop host `keys[]` from predicate entries

**Mechanism.**
- `BuildBitmap` (`ann/indexfile.c:1015-1034`) reads `file->keys[ordinal]` for every set bit, the 33.8% line of the fresh10 profile.
- No matrix-route consumer reads an entry's `keys[]`. `SearchGatherMatrix` uses ordinals and `deviceRows`, and hits come from `file->keys[row]` (`cuvs.c:4046`).
- Build only `ordinals[]` for computed predicates. Keep `keys[]` for key-list filters that arrive through `SubmitIndex`.

**Evidence.**
- Per predicate: 91.4 cores / 2,488 evaluations per second = 36.7 ms. Removing 33.8% gives 24.3 ms, at best 1.51x, if the fresh10 shares carry over.
- At 10% density the loop streams the whole 80 MB key array per predicate. The implied rate is about 200-240 GB/s (*estimate*, computed from assumed stream sizes). Host memory bandwidth on distinct10 has never been measured. If bandwidth is a hidden limit, this change is the right lever.

**What the reviewers corrected.**
- **1.51x is a ceiling, not an estimate.** The worker is not saturated (about 90 logical cores idle). `hit-distinct10-second-arrays` used the same CPU per call on 24% fewer cores and delivered 19% less throughput. If whatever holds calls back stays, the gain is about 7%. A later review put distinct10 at +20-45% (*estimate*), inside the same band.
- **`SearchGather`'s non-fused path reads keys unconditionally.** This covers k=500, `--matrix-gather off`, scan-only indexes and every `SearchGatherMatrix` decline:
  - staging at `cuvs.c:3012-3016`;
  - the repair path at 3077-3078;
  - the hit key at 3091.

  These must move to ordinals. A sentinel filter pointer would crash these routes.
- **The `job->filter` host-keys route gate** (`server.c:2131-2135`, `cuvs.c:3231, 5056`) must become a flag. Audit `AnnIndexExact` and `SearchGatherGroupLarge` for `keys[]` use.
- **`server.c:2131-2134` couples the two fields.** It sets `ordinals` only inside `if (bitmap->keys)`, so the condition must be split.
- **`BitmapBytes` (`indexfile.c:921-923`) must charge `count*8` under `if (bitmap->ordinals)`.** Otherwise entries go uncharged, and with `--gpu-predicates on` at 0.7% hits the host cache never evicts.
- The proposed `rowsKnown` field is redundant: `ordinals != NULL` already serves as the flag.
- Cache capacity (475 -> about 865 entries) has no throughput effect on these workloads.
- The optional u32-rows step is effort M and should not ride on this change.

**Dependencies.** None. It is the first step of the uncached chain and stays useful for predicates that neither P01 nor G1a can decompose.

### G1a: clustered worker export with range-form predicate jobs

**Mechanism.**
1. **Export permutation.** Write the worker files (vectors, NVFP4 codes and scales, keys, attribute columns) in argsort order of a declared cluster key, for example `(category, tenant, key)`. This goes in `src/annexport.c` (`:147-175, 241, 250`) or in `scripts/ann/build_cuvs_index.py` before quantization; NVFP4 is per row, so quantizing the permuted rows gives byte-identical rows. The PostgreSQL frozen index stays key-ordered. Side files:
   - `ordinal.u32`: the original ordinal of each row;
   - `keyorder.u32`: rows sorted by key;
   - a (category, tenant) cell directory;
   - a manifest line that makes old workers refuse the directory.

   The CAGRA graph is rebuilt on the permuted data.
2. **Load.** `InvertAttribute` (`indexfile.c:397-440`) is unchanged. For the primary cluster column, the postings ordinals become identity runs.
3. **Range jobs.** After canonicalisation, a K_SET on the primary column, optionally ANDed with one K_SET or NOT K_SET on the secondary column (`canonical.c:245-262, 680-704`), maps directly to base ranges plus a short exclusion list. Past ~4096 entries, or below an average of 64 rows per range, the job falls back to the bitmap path.

   A range job skips evaluation, `AnnIndexBitmapGet/Put`, `BuildBitmap`, `UploadBitmap` and `ExpandRows`. Its 64 jobs share one refcounted ranges object. `FilterBucket`/`FilterFor` and `Shareable` (`server.c:817, 835-885`) key on that pointer.
4. **GPU step.** A tiny device kernel fills `gatherRows` from the ranges, replacing the host staging loop at `cuvs.c:3947-3961`. `score_mma`, `PrunedSelect` and `RerankDevice` work unchanged.
5. **Result mapping.** Replies carrying ordinals write `ordinal.u32[row]`. Under P02, PostgreSQL never sees worker rows.
6. **Sites that assume key-sorted rows** must be rewritten (see the corrections below).

**Evidence.**
- On the benchmark data every 10% window spans all 10M rows: `tenant = id % 1000`, `category = (id // 7) % 100` (`benchmarks/ann/cuvs_load.py:112-113`).
- The work removed is the fresh10 profile's 58.6% + 27.5%. Taking out ~86% of the 37 ms of worker CPU per call leaves about the cached-path 4-5 ms, so the worker stops being the limit and PostgreSQL becomes it.

**What the reviewers corrected.**
- Worker CPU per new predicate is about 37 ms per call on arrays and 30 ms on SQL, not 15-25 ms.
- **The PostgreSQL ceiling is not established.** Uncached runs show 15-16 ms of PostgreSQL CPU per call, against 8.7 ms cached. The 775k end assumes the cached figure returns. The range is 440-775k (one reviewer: 450-700k).
- **Fragmentation.** Range jobs bypass the cache, so the per-filter lane cap of 16 and the idle-share split (`server.c:1061-1086`) can fragment 64-query requests. fresh10 showed 22.5 queries per batch, and each fragment rescans 1M rows.
- **Cell shape.** Cells are not uniform: 7,000 non-empty cells of about 1,428 rows. Each category holds 70 tenants and each tenant appears in 7 categories. `tenant <> w` removes 0-3 cells inside a 10-category window.
- **The proposed validation loader does not reproduce the benchmark distribution.** Its formula puts every tenant in every category. A like-for-like test table must keep each row's attribute values and only reassign keys in cluster order. Even then it measures a different layout, not the permutation design that the existing table needs.
- **Key-sorted sites missing from the proposal:**
  - the load-time strictly-increasing check (`indexfile.c:1649-1655`);
  - `AnnIndexOrdinals` (`indexfile.c:1943+`) and its callers (`cuvs.c:3081, 3311, 5105`, `server.c:1804`, `cpu.c:335`);
  - the `AnnIndexTreeMerge` disjoint-key contract (`indexfile.c:2148-2151`);
  - `cuvs.c:521` bsearch and the `cuvs.c:2813` gather assumption;
  - `OrdinalOf` and `AnnIndexDelete` (`indexfile.c:1227-1273`);
  - the key-list shard split and lookup (`cuvs.c:574, 2404, 2859-2878, 3103-3121`);
  - `server.c:1599`.
- **Ordinal-space inputs from PostgreSQL also need remapping:**
  - pushed bitmaps (`src/annindex.c:1078-1100`, `server.c:2028`, a 10M-bit scatter per push);
  - ops 18/19;
  - ordinal replies on op 22 as well as op 30.
- Only the primary column's postings become identity runs. The secondary column keeps its full ordinals and keys.
- After a collection dictionary remap (`indexfile.c:760-795`) value-index runs no longer map to contiguous rows.

**Hidden costs.**
- Only predicates on the declared cluster-key prefix benefit, and the key was chosen to fit the benchmark's predicate columns.
- About 80 MB more host memory per index.
- A graph rebuild and requantisation whenever the key changes. Valid only for frozen/immutable tables.
- A new job kind carried through the scheduler and every route. Graph routes above 2M rows still need bitsets built from the ranges.
- A second evaluation path that must match the bitmap path exactly.
- Test binaries and recall scripts that assume worker row == PostgreSQL ordinal must be updated.

**Relation to P01.** Both attack the same 86%. P01 covers any decomposable AND. G1a additionally removes the base evaluation, the cache entry and the device rows, but only for cluster-key predicates. One reviewer estimates this unique increment at 15-45% of a GPU at about 12k new predicates/s (*estimate*). Build it after P01, and only if the deployment's predicates match a stable cluster key.

### G4a: per-vector predicates in one batch call

**Mechanism.**
- **SQL.** Add an overload `ann_frozen_search_batch_arrays(name, vector[], k, filter text, params int4[])`, where the filter has one placeholder (for example `tenant = $1`).
- **Protocol.** A new op-30 flag `ANN_PREDICATE_PER_PART` carries one program per part, about 1.5 KB extra per 64-vector call.
- **Worker lookup.** The worker canonicalises and looks up each part (`PredicateBitmap`). Parts with the same canonical hash share one entry reference.
- **Scheduling.** The composite's jobs are queued as one unit. A lane pulls only jobs of the same composite, which avoids the cross-lane splitting behind part of the -25% `mixed10` result.
- **Routing.**
  - Same-filter parts go to `SearchGatherMatrix`.
  - Singletons (up to `ANN_MIXED_MAX` 64, `cuvs.c:4094`) go to one `SearchGatherMixed` batch, gated by a per-job mix bit in place of the global `--mixed-groups`.
  - For parts without device rows, rows are staged from host ordinals.
  - Parts at or under `--exact-limit` stay on `AnnIndexExact`.

**Evidence.**
- Today op 30 validates one program for all parts (`server.c:2575-2578`) and does one lookup per request (`2600-2604`).
- `MixedEligible` requires the global config and `deviceRows` (`cuvs.c:4099-4106`).
- PostgreSQL cost per vector on arrays is 157-160 us (`fin-unique1-arrays-gpoff`, `s-0.1pct-gather-b32`), against 0.53-0.66 ms on single-vector SQL.
- GPU anchors:
  - `u-unique1-off`: 101.2k at 1.49 queries per batch, GPU 92.8%.
  - `u-unique1-on`: 146.9k at 6.04 queries per batch, GPU 69.3%, PostgreSQL 88.1 cores.

  These give 37.7 GPU-us per query for the segmented route at m=6.

**What the reviewers corrected.**
- **The headline 2.0-2.5x is optimistic.** It assumes the segmented route shares the matrix route's fixed cost of about 105 us per batch, which is unmeasured. With no shared fixed cost the bound is about 212k at 100% GPU. Adjusted range: ~180-370k against 146.9k (*estimate*).
- **The 0.1% projection** rests on an assumed 5-10 us per segment. The only 0.1% point (`s-0.1pct-gather-b1`) is 32 GPU-us per query on the matrix route, with 4.27-query same-filter groups.
- **Confining mixing to one composite** removes only the group-splitting half of the `mixed10` loss. Separate per-part scoring remains, so per-vector distinct 10% windows (1M rows each) would still lose to the matrix route.
- **Byte-patching parameters into a compiled program is not valid in general.** `src/annpred.c` produces a value-dependent layout: `ANN_FILTER_NONE` when low > high (`:171-183`), AND clauses sorted by bytes (`:948`), VALUES sorted and deduplicated (`:451, :575`). `$1+9` needs evaluation per vector. Bind a Param list and rerun Translate per vector instead (a few us).
- The unique1 SQL anchors ran at 32 lanes / 512 clients. `fin-unique1-sql-gpoff` (116.8k) is a different configuration.

**Hidden costs.**
- **Protocol skew:** `server.c:2575` rejects unknown flags. `src/annremote.c` (`:359-367`), RDMA and collections must learn the layout.
- **Misses:** up to 64 per request, evaluated serially on one reader thread under the single-flight table (`server.c:2283-2300`).
- **Staging:** host-ordinal staging is the same loop that took 27.5% on fresh10.
- **Scheduling:** a new lane scheduling mode that interacts with `--filter-lanes`, the idle-share caps (`server.c:972-997, 1061-1086`) and `Settle`.
- **Cache budget:** about 1.6 GB host / 3.2 GB device for 991 distinct 1% windows, under the same LRU budgets.
- **Benchmark tooling:** a multi-tenant mode in `frozen_stress.py` and per-vector exact top-k in `frozen_recall.py` are needed before any gain can be shown.

**Dependencies.** Gains stack with P02/P11 (per-id PostgreSQL cost) and P03.

## 4. Contested (one reviewer refuted, one did not)

| Id | Title | Disagreement | Adjusted gain range | Effort / risk |
|---|---|---|---|---|
| P02 | Return keys instead of ordinals under `ann.frozen_immutable` | Both lenses agree the change is correct and cheap. The evidence lens refutes it on gain: under 10% at the target. The feasibility lens keeps it as a prerequisite for P03/P11 and as banked PostgreSQL capacity. A later check confirmed the path: the flags are one line (`src/annindex.c:1887`), every route sets `hit.key` (`cuvs.c:2056, 3123, 3657, 4046, 4303, 4934`; `indexfile.c:2120`), and the `server.c:526` composite rewrite touches `.ordinal` only when attributes are requested, which op 30 never does | Backend CPU -7..14% (*estimate*; the post-immutable share of `AnnArrayRead` is unmeasured, and the 15% profile share predates immutable and bundles TID/visibility work). 10pct +3-6% (*estimate*, assuming freed cores convert at 0.5-0.65) or +0-4%. unique1 +4-8%. k=1500 +2-12% | S / none. Replace the `ordinal >= rows` guard with a key sanity check |
| P03 | Worker writes the reply in wire layout from device-gathered keys | The evidence lens: the result loop is at most 0.71% of worker CPU in the fresh10 profile (about 11 ns per hit, not 30-60 ns); the 128-lane null suggests the lane host phase is off the critical path; k=1500 is not lane-bound. The feasibility lens: it can be built, step 0 is nearly free, the gain is small, and a profile should come first | k=500 +0..8% (*estimate*). k=1500 capped near ~310k by PostgreSQL's per-id rate. Step 0 alone frees about 1-8 worker cores | M-L: needs shard keys on every GPU (may be NULL after a failed upload, `cuvs.c:1036-1043`), a host fallback for CPU-scored rows (`cuvs.c:4047-4050`), and a rework of `CompositeAnswer`/`BatchAnswer` / none |
| P04 | Replace both cuVS `select_k` calls, drop `matrix_fill`, remove the mid-pipeline sync | Both lenses agree the mechanism holds: fill writes 12.6 MB nobody reads, and the sync exists only for host count checks. They disagree on translation: about +3-7% (evidence) versus about +5-10% (feasibility, 780k -> ~820-860k) | Only select_k#2 (about 59 us upper bound, measured on a different shape) and fill are well evidenced. select_k#1 at k=32 is likely already a warp-sort kernel (~0-5 us saved). Fixing the sort size at 8192 matches today's behaviour but would regress filters of 1.5-2M rows | M / none |
| P14 | Single-vector SQL: the custom scan projects the worker's distance | The plan shape (no `CUSTOMPATH_SUPPORT_PROJECTION`, a Result node recomputing `emb <-> $1`) is confirmed in code but not by EXPLAIN. The evidence lens: much of the 0.66 ms per query is fixed per-call work, the vector read is already prefetched, and the host is not shown to be PostgreSQL-saturated, so 5-12% (177k -> ~186-198k). The feasibility lens gives 13-25% (~200-220k) | Arrays path 0 | M (towards M-L: slot layout rewrite) / none for ordering. Semantic risk: projected distances become approximate worker values unless the substitution is limited to resjunk uses |
| G2b | Re-rank the CPU-saving items under physical-core accounting (P11 production part ≈ P02 > P03 at 780k; P07 for uncached only; stop CPU work once GPU busy passes ~92%) | The evidence lens refutes. G2b changes no code. Its premise (freed logical cores worth f ≈ 0.5) is unmeasured and conflicts with `docs/ann-10pct-throughput.md:46`. Its base is internally inconsistent: 112 core-equivalents for all roles implies f ≈ 0.65-0.7. The feasibility lens keeps the order as defensible and verified the P02 path (see the P02 row) | 0 on its own. Components (*estimates*): P02 +3-6%; P11 production part (catalog, program and tombstone caching, in-place reply parse) +1.5-4%, with the text-I/O and setup-statement share treated as a harness artifact; P03 0-6%. Combined production-relevant ~830-880k, under a GPU cap of ~880-930k that is itself an NVML-based *estimate* | M (planning) / none |
| G3a | Scale the prune sample with eligible rows: `sample = clamp(pow2ceil(n*32/4096), 8192, 65536)`, so the pruned NVFP4 gather stays valid from 1.5M to 8M rows at k≥300 | The evidence lens refutes for the target: at n=1M the formula returns today's sample (8192) and rank (32), so nothing changes. The feasibility lens accepts the mechanism (a few lines: `sample` is already a parameter everywhere) for broad filters | 10pct: 0. 2M rows k=500: +0-5% from a sort shrinking from 16384 to 8192 (*estimate*; kept-count spread puts the 64-query max near 5.5k). 4M rows k=500: pruned gather ~150-265k b32 (*estimate*). It would be the only route there with lowest-query recall ≥0.90 (graph 0.29-0.53). The unpruned k=500 baseline at 4M was never measured; the k=100 analogue is 45.8k | S for the prune change. The routing split by k does not exist: gather limits are global (`cuvs.c:3186, 3284, 5046`; `server.c:1901, 1983`). `SearchGatherMixed` has its own fixed-sample copy (`cuvs.c:4109-4123, 4154`) / none |

G3a validation note: `matrix_prune_fallbacks` cannot detect the 4M disablement. The early guard return at `cuvs.c:3755-3757` does not increment it. Check `matrix_pruned_queries` (`cuvs.c:3816`) against the number of queries searched. An offline CuPy check of the kept-count distribution at S = 8192/16384/32768 needs no worker change.

## 5. Rejected (two refutations)

- **P05: fused candidate scan, CUDA graphs and double buffering.** The fused scan measured 6.01 us per query (385 us per batch). That figure is for the top-32 shape (`docs/ann-blackwell.md`, Fused selection table), not k=500 with wanted=756, so the -13%..+8% net is an extrapolation. Parts B and C are contradicted by the 128-lane null. The fused route is reachable today with `--matrix-gather off` under `ANN_KERNEL_TIMING=1`, which measures the real k=500 cost for free.
- **P06: request-whole same-filter pickup.** Fragmentation happens only when lanes are idle, so lane savings barely reach throughput: about 1% at k=1500 and an unverified 0-8% on distinct10. The staging it saves is removed by row-staging fixes anyway.
- **P08: stage from bitset, GPU-resident rows, peer copy.** Its CPU-proportional model fails because about 90 logical cores sit idle. Step 1 is worth about 3-25% and only about +5-10% on top of P07. Steps 2-3 rewrite twelve or more route gates, the evaluation hook and publication (L-XL). Superseded by P01. Keep its cheap probe: u32 rows in `BuildBitmap`, so staging becomes a memcpy.
- **P09: one worker process per GPU.** CUDA contexts are already per device with 12 lanes each. The shared mutexes showed 0.18% in the fresh10 profile; there is no worker profile at the cached point. TCP becomes a confound. Per-process caches would push mix10 towards distinct10's rate. Expected -5..+5%.
- **P10: attribute postings as implicit row slices.** Every single-column workload is already cached at the PostgreSQL-bound rate. mix10's ceiling is about 250-260k, the existing 8 GB figure. Its only value is as P01's base (or G1a's identity runs); op 17 is partial prior art.
- **P11: binary client, packed ids, cached per-call work.** The target is not shown to be PostgreSQL-bound (+0-5%). The setup statement does no `vector_out` (the queries column is text). The change breaks comparability with every `runs.jsonl` row. Most of its share is text I/O and the setup statement, which are harness costs. The production part (caching catalog, program and tombstones; parsing the reply in place) is about +1.5-4% (*estimate*; see G2b). It is real on unique1 and k=1500, which are not the benchmark case.
- **P12: PostgreSQL scale-out over TCP plus NUMA pinning.** The transport is already implemented. The gain at the target is 0-3%. The GPU-to-socket topology is not documented in the repository, so the pinning target is unspecified.
- **P13: 128-query groups, then tcgen05 FP4.** A fit puts fixed per-call backend work at about 0.43 ms of 8.75 ms (4.9%), worth about +2.5%. The fit comes from three pre-immutable runs that also differ in lanes, clients and GPU busy, so it is not a clean measurement. The step 2 roofline misreads the 1,024-dim mma bound. At 128 dims, the reviewers judged the gather (~136 us per 1M rows) to be the bound.
- **P15: GPU-phase lane cap per device.** A semaphore after pickup cannot regroup jobs a lane already holds. Little's law rules out groups of 8.5 at 256 clients. At best it equals `--lanes 32`.
- **P16: NVFP4 order without rerank.** The sort-size shrink is false because the rank is clamped at 32. Recall is unmeasured, and no measured workload is GPU-bound: 0 throughput.
- **P17: CUDA graphs and double-buffered lanes standalone.** Contradicted by the 128-lane null. Graph launches save 0.4-0.9 cores. Expected 0-2%.
- **P18: 64-dim PCA scoring copy.** The 128-dim scan is not the binding limit: the kernel would save about 20-35% and the ceiling would rise 8-20%, on a ceiling that does not bind. Recall cannot be validated on the synthetic 16-d latent data.
- **G1b: range-input `score_mma` streaming contiguous clustered rows.** 0 throughput at every measured point. The 64-wide pass is bound by the mma.sync issue path, not memory (`docs/ann-blackwell.md:68`). The "gather is the cost" remark (`:55`) is about the 16-query and single-query kernels. NVFP4 codes are already read as full sectors (`blackwell.c:375-380`); only the scale and rowscale loads are amplified. It depends on G1a. Keep its cheap probe if G1a is built: time the existing `AnnBlackwellScoreMatrix` on a scattered 10% rows list against an iota list. Drop the kernel variant below 1.2x.
- **G2a: SMT/physical-core discriminator as a lever.** 0 by construction. Its supporting fit leans on the two 128-lane points. `perf stat` is reportedly blocked on the benchmark host (gprofng timer samples only). A CPU burner also slows the lanes' blocking-sync wakeups, which biases the test towards H1. Its cheap parts are kept as experiment 4 below.
- **G3b: size the filtered CAGRA frontier to the filtering rate.** 0 on the benchmark: 1M rows already run at itopk 1024, and the gather wins at k=500.
  - **The zero-code validation cannot run.** `--itopk` is clamped to 512 (`ann/worker.c:332-333`), and the 0.25 gate blocks `--filtered-itopk` at 40%.
  - **The baseline is wrong.** At 4M rows, k=100, b1 already meets the floor (221.7k, lowest 0.99). Under the default 4M gather limit these searches already take the graph.
  - **The ratio model of recall is contradicted:** the 10%/k=500 UDF row at ratio 0.82 has recall 1.000.
  - **Costs:** raising the frontier would cut 4M throughput from 884k-1.2M to about 450-800k (*estimate*). `WidenShortJob` caps at 1024 (`cuvs.c:2309, 2340`). The cost profile would need recalibration.
- **G3c: cluster-probed two-stage gather for single-vector SQL.** Its baseline (96 lanes, 115.5k) is not the best configuration. 32 lanes / 512 clients gives 177.3k with GPUs at 71.1%, PostgreSQL-bound, and gather b1 is flat from 10k to 1M rows. The proposal's own delivered range (170-200k) is -4% to +13% over that. `SearchGatherMixed` refuses m<2 and handles one contiguous segment per job (`cuvs.c:4147-4210`). Cluster-ordered renumbering changes ordinals shared with PostgreSQL. Recall at 3-6% probed is unvalidated on real data.
- **G4b: fused FP16 exact route for ≤16,384-row segments.** Lists under `ANN_PRUNE_MIN_ROWS` = 65536 already skip pruning (`cuvs.c:2527, 3755`), so the kernel chain it replaces does not run there. `s-0.1pct-gather-b32` already reaches 949.5k, PostgreSQL-bound at GPU 66.6%, which is the outcome the proposal predicts. The benchmark table has 1,000 tenants, not 10,000 (`docs/ann-frozen-index.md:290`). Sparse entries are about 160 KB, not 17 MB.
- **G4c: host exact route for tiny tenant x category conjunctions.** The exact route already exists for ≤`exact_limit` filters (`cuvs.c:3315-3317, 5101-5109`), with no device rows (`cuvs.c:1276`). Host evaluation already intersects sorted lists (`filter.c:408`). A transient mode exists (`AnnIndexBitmapTransient`, `indexfile.c:1068`). Tiny entries do not trigger the eviction qsort (`indexfile.c:1415-1416`). The simpler fixes are to return short lists without filling the 1.25 MB words and to raise `ANN_INDEX_BITMAP_BUCKETS` (`indexfile.h:152`). 0 on the benchmark.

## Results so far

**P01 falsifier (experiment 1): passed.** A throwaway build dropped the `tenant` operand from the canonical AND, so `distinct10` was served from its category window alone (wrong answers by design; the recall check uses `mixed10` predicates and cannot see the difference). Same settings as the final configuration with `--gpu-predicates on`, 30 s:

| Run | Arrays, vectors/s | SQL, QPS | PostgreSQL cores | Worker cores | GPU busy |
| --- | ---: | ---: | ---: | ---: | ---: |
| `distinct10` (`hit-distinct10-all`) | 156.2k | 4.55k | 39.4 | 91.4 | 39% |
| `distinct10`, residual ignored (`p01f3-*`) | 769.9k | 105.3k | 107.0 | 48.2 | 78% |
| `mixed10`, same build (`p01f3-mixed10-arrays`) | 773.8k | — | 105.5 | 48.0 | 78% |

So the ceiling P01 targets is real, and PostgreSQL's 16 ms per call on `distinct10` is not caused by the predicate: with the worker doing cached work it returns to 8.9 ms per call. The real P01 still pays for the residual mask, and through single-vector SQL each query carries its own residual, so batches would not group the way they do here (1.52 queries per batch). A first attempt that split the AND on the raw program kept only `category >= a` for right-nested programs and sent most searches to the graph route (`p01f2-*`, 220k); splitting after canonicalisation fixed that.

**P07 built: predicate entries keep ordinals only.** `BuildBitmap` no longer reads `file->keys` for each row of a new predicate result (pushed bitmaps still keep keys), `SearchGather` splits, stages and maps results from ordinals, and the route gates accept ordinals as known rows. Same settings, 30 s:

| Workload | Before | After (`p07-*`) | Change |
| --- | ---: | ---: | ---: |
| `distinct10` arrays | 156.2k vectors/s | 237.6k | +52% |
| `distinct10` SQL | 4.55k QPS | 6.18k | +36% |
| `mix10` arrays | 248.8k | 435.3k | +75% |
| `10pct` arrays (cached) | 775.6k | 776.8k | 0% |

Recall 0.9998 mean, 0.994-0.996 lowest. The gain is at the top of the reviewed range (1.1-1.4x on `distinct10`) and larger on `mix10`, whose repeating half now evicts less: an entry is 9.25 MB instead of 17.25 MB. The worker is still CPU-bound on `distinct10` (110-136 cores).

**Lane rows from the bitset on the GPU (P08 step 1).** A `distinct10` profile on the current build (gprofng, 30 s of 64-vector arrays) put 31% of the worker's CPU in `SearchGatherMatrix` turning an entry's ~1M 64-bit ordinals into 32-bit shard rows on the host for every search (`cuvs.c`, the `staged[i] = ordinals[...]` loop), 27% in `BuildBitmap`, and 9% in the per-hit `file->keys[row]` lookup. With `--gpu-predicates`, a lane that finds no device rows now copies the shard bitset (rows / 8 bytes) and expands it into rows on its own stream with the expand kernels (`ExpandLaneRows`); sparse entries without bitsets still stage from ordinals.

| Workload | P07 | + lane expansion (`lex-*`) | Change |
| --- | ---: | ---: | ---: |
| `distinct10` arrays | 237.6k vectors/s | 363.9k | +53% |
| `distinct10` SQL | 6.18k QPS | 10.15k | +64% |
| `mix10` arrays | 435.3k | 507.8k | +17% |
| `10pct` arrays (cached) | 776.8k | 771.9k | 0% (noise) |

Recall 0.9998 mean, 0.994-0.996 lowest. Together with P07, `distinct10` is 2.3x its 156.2k start through arrays and 2.2x through SQL. `BuildBitmap` (building ordinals for every new entry) is now the largest worker cost on this workload; P01 removes it for decomposable predicates.

**P01 built: `--residual-masks` (default off).** The canonicaliser splits an AND (`AnnFilterCanonicalSplit`): operands on one single-valued column whose complement holds at most 2^18 rows become exclusions, the operand removing the most rows always stays in the base, and the base is cached like any predicate. The server turns the exclusions into the base rows they remove (`AnnIndexExcludedRows`: the excluded values' postings, kept where the base has them), attaches them to each job and subtracts them from the eligible count. The matrix gather sets those rows' scores to +inf right after scoring (`mask_scores`: one thread per (query, row) pair, a binary search of the shard's sorted rows), after checking that every query keeps k plus the slack; every other route, and the gather's repair fallback, first turns the job into one over its own rows (`AnnIndexJobDropExcluded`). Searches sharing a base group by its rows, so single-vector SQL batches as on cached predicates. The base must take the gather routes (ordinals known, within the group gather limit); otherwise the whole predicate is evaluated as before. Tests: 20,000 random programs (about 3,700 splits) whose base minus exclusions equals the program's rows, under eleven seeds; a GPU test comparing ids, distances and eligible counts with masks on and off, with query vectors taken from the excluded tenant; recall measured against an exact `distinct10` reference (`frozen_recall.py --workload distinct10`).

| Workload | Lane expansion | + residual masks (`rm-*`) | Change | Recall (mean / lowest) |
| --- | ---: | ---: | ---: | --- |
| `distinct10` arrays | 363.9k vectors/s | 546.5k | +50% | 0.9998 / 0.996 (exact `distinct10` reference) |
| `distinct10` SQL | 10.15k QPS | 35.8k | 3.5x | 0.9999 / 0.998 |
| `mix10` arrays | 507.8k | 660.3k | +30% | 0.9998 / 0.996 |
| `10pct` arrays (cached, no AND) | 771.9k | 761.2k | -1.4% (within run spread) | 0.9998 / 0.994 |

From the start of this work, `distinct10` went 156.2k -> 546.5k vectors/s (3.5x) and 4.55k -> 35.8k QPS (7.9x). The falsifier's ceiling is 769.9k and 105.3k; the worker is the limit again (94 cores on arrays, 138 on SQL), most likely the per-search exclusion list (each search copies and binary-searches about 10,000 rows) and canonicalising the program twice.

**Residual masks, cheaper exclusions.** Profiles of the `rm-*` runs put 78% of the worker's CPU (SQL) and 27% (arrays) in `AnnIndexExcludedRows`, which binary-searched each of a search's ~10,000 excluded rows in the 1M-row base, and 13% (arrays) in the per-hit exclusion check. Membership is now one bit test in the base entry's bitmap (binary search only for entries without one), and the per-hit check is one search instead of two:

| Workload | `rm-*` | `rm2-*` | Falsifier ceiling | Recall (mean / lowest) |
| --- | ---: | ---: | ---: | --- |
| `distinct10` arrays | 546.5k vectors/s | 674.7k | 769.9k | 0.9998 / 0.996 |
| `distinct10` SQL | 35.8k QPS | 96.6k | 105.3k | 0.9999 / 0.998 |
| `mix10` arrays | 660.3k | 741.9k | — | 0.9998 / 0.996 |

`distinct10` single-vector SQL is now GPU-bound (95.6% busy, as cached predicates are); the arrays point has PostgreSQL at 109 cores, its cached level, and the worker at 75 cores against 48 on cached work. Since the start of this work: `distinct10` 156.2k -> 674.7k vectors/s (4.3x) and 4.55k -> 96.6k QPS (21x).

**The limiter at the cached point (experiment 4): per-lane host work, not saturated cores.** On the cached `10pct` arrays point, 167 of 224 logical CPUs were busy (PostgreSQL 108.6, worker 50.0, pgbench 8.3; 112 physical cores). Three measurements on the same build:

| Experiment | Vectors/s | Reading |
| --- | ---: | --- |
| Baseline (`h1-base-arrays`) | 760.8k | |
| + 32 busy threads (`h1-burn32-arrays`) | 736.0k (-3.3%) | took most of the ~53 idle logical CPUs at little cost |
| + 64 busy threads (`h1-burn64-arrays`) | 689.8k (-9.3%) | oversubscribed host |
| Lanes spin-wait on the GPU (`h2-spin-arrays`) | 735.8k | +32 worker cores, no gain: completion wake-ups are not the limit |
| Lanes yield-wait (`h2-yield-arrays`) | 736.8k | same |

A worker profile at this point (none existed) put about 35% of its CPU in stream synchronisation, about 20% in the per-hit key lookup (`file->keys[row]`, ~32,000 random reads per 64-query batch), 18% in `BatchAnswer` serialising replies byte by byte, and 7% in the duplicate-position check. Two changes cut that host work: the matrix gather now fetches result keys on the GPU from the shard's device keys (`gather_keys`), and `Read`/`Write` copy fields on little-endian hosts:

| Workload | Before | After (`dk-*`) | Change |
| --- | ---: | ---: | ---: |
| `10pct` arrays (cached) | 760.8-776.8k vectors/s | 806.2k | +4-6% (previous best 782.6k) |
| `distinct10` arrays | 674.7k | 692.9k | +2.7% |
| `10pct` single-vector SQL | 113.0k QPS | 113.0k | 0% |

Worker CPU fell from about 50 to 45 cores and GPU busy rose from 78% to 83%, so the lanes' host work per batch is on the critical path (H2). Next on that path: the stream-synchronisation spin (35%), the remaining reply serialisation, and the duplicate-position check.

**Lanes sleep while their GPU work runs.** The stream-synchronisation share in that profile was busy waiting: `cuvsStreamSync` goes through RAFT's `interruptible::synchronize`, which polls `cudaStreamQuery` and calls `std::this_thread::yield()` in a loop whatever the device's scheduling flag (the earlier spin and yield experiments changed only the flag, so they could not show this). Lanes now wait with `cudaStreamSynchronize` on their stream, which honours `cudaDeviceScheduleBlockingSync`:

| Workload | `dk-*` | `ls-*` | Worker cores |
| --- | ---: | ---: | --- |
| `10pct` arrays (cached) | 806.2k vectors/s | 833.1k (+3.3%) | 45.0 -> 26.7 |
| `10pct` SQL | 113.0k QPS | 119.6k (+5.9%) | 68.7 -> 35.4 |
| `distinct10` arrays | 692.9k | 711.1k (+2.6%) | 71.4 -> 58.3 |
| `distinct10` SQL | 96.6k | 98.6k | 74.9 -> 43.1 |

Recall unchanged. The worker now uses about a quarter of the host's busy CPU at the cached point; PostgreSQL (104-112 cores) is again the largest consumer.

**Defaults.** `--gpu-predicates` and `--residual-masks` are now on by default (`off` restores the earlier behaviour). With neither flag given: `distinct10` arrays 699.3k vectors/s and cached `10pct` arrays 829.9k (`def-*`), within run-to-run variation of the explicit runs above.

**P04, first half: no candidate fill, no mid-pipeline sync.** The pruned selection filled the whole q x cap candidate buffer with +inf before compacting (12.6 MB of writes a 64-query batch that the sort never reads; it reads each query's kept entries only), and synchronised the stream mid-batch so the host could check the kept counts and size the sort. The fill is now a memset of the counts; on one-shard indexes a sort kernel sizes itself from the count on the device (shared memory launched for the cap) and the counts are checked after the batch's final sync, a failure returning the group to the other routes. Both workloads pruned every matrix query with no fallback:

| Workload | `ls-*` | `ns-*` | Change |
| --- | ---: | ---: | ---: |
| `10pct` arrays (cached) | 833.1k vectors/s | 890.7k | +6.9% |
| `distinct10` arrays | 711.1k | 724.6k | +1.9% |

Recall unchanged. GPU busy rose to 91.4% and PostgreSQL to 138 cores at the cached point.

**P04, second half: the final top-k in shared memory.** After reranking, each query's k plus slack candidates (756 at k=500) went through a cuVS `select_k` launch. One block per query now sorts them, padded to the next power of two, in shared memory (`final_topk`; cuVS still takes lists over 4,096):

| Workload | `ns-*` | `ft-*` | Change |
| --- | ---: | ---: | ---: |
| `10pct` arrays (cached) | 890.7k vectors/s | 929.9k | +4.4% |
| `10pct` SQL | 119.6k QPS | 138.2k | +15.5% |
| `distinct10` arrays | 724.6k | 712.8k | -1.6% (within run spread; worker-bound) |

Recall unchanged. At the cached point PostgreSQL is at 146.5 cores, the worker at 32.8 and the GPUs at 90%.

**PostgreSQL per-result work (P02 and in-place arrays).** With the worker at 33 cores, PostgreSQL became the largest consumer at the cached point (146.5 cores at 929.9k). Backend stack samples at that point (gdb, 120 samples) spent 61% waiting for the worker and, of the rest, about 12% writing the id arrays as text, 9% reading each hit's key from the index's mapped key array (`AnnArrayReadRange`), 8% building the result arrays (`palloc0`, `construct_md_array`) and 4% parsing the `vector[]` argument. Two changes in `ann_frozen_search_batch_arrays` and `ann_frozen_search_batch`: under `ann.frozen_immutable` the request asks the worker for keys instead of ordinals (no TIDs are needed, and the worker reads keys on the GPU), and the result arrays are written in place instead of through `construct_array`:

| Workload | `ft-*` | `pk-*` | PostgreSQL cores |
| --- | ---: | ---: | --- |
| `10pct` arrays (cached) | 929.9k vectors/s | 973.5k (+4.7%) | 146.5 -> 82.5 |
| `distinct10` arrays | 712.8k | 765.2k (+7.4%) | 124.8 -> 84.6 |

Recall unchanged. The GPUs are now 92% busy at the cached point, with PostgreSQL at 82.5 cores and the worker at 26.6.

**Harness (prerequisites 1-2).** `frozen_stress.py` now records the worker's server statistics (operation 8: predicate hits, evaluations, residual masks) before and after each run, beside the routing statistics; per-role CPU including pgbench's was already in each run's JSON. With `ANN_KERNEL_TIMING=1` the matrix gather records six events on the first shard's stream and prints per-batch averages every 10 s. At 96 lanes (12 per GPU) those times include other lanes' kernels; with one lane per GPU (64 clients, 64-query batches over 1M rows) they are the stages' own GPU time:

| Stage | us per batch | Share |
| --- | ---: | ---: |
| NVFP4 scoring | 319 | 45% |
| Residual mask | 3 | 0.4% |
| Selection (sample, cutoff, compact, sort) | 331-346 | 47% |
| FP32 rerank | 20 | 3% |
| Final top-k | 22.5 | 3% |

The selection stage costs as much as scoring 64M rows; its bitonic sort runs one block per query (64 of 148 SMs) over about 4,096 kept scores, and the cutoff uses a cuVS `select_k` launch. That is the next GPU target.

**A faster selection.** Timed inside, the selection was 37 us for the cutoff (sampling plus `select_k`), 174 us compacting the scores under it (reading 256 MB a batch at about 1.5 TB/s) and 118.5 us sorting the kept scores. The rerank and the final top-k need the best candidates as a set, not in order, so a radix select replaces the sort (`matrix_prune_select`: four 8-bit histogram passes over the scores' order-preserving bits find the k-th smallest, then one pass writes the keys below it and enough of those equal to it), and the compaction reads four scores a thread with one atomic a warp (`matrix_compact4`, for rows of n % 4 == 0). One lane per GPU: sort 118.5 -> 38.8 us, compact 174.0 -> 157.6 us, selection 339 -> 243 us a batch.

| Workload | `pk-*` | `rs-*` | Change |
| --- | ---: | ---: | ---: |
| `10pct` arrays (cached) | 973.5k vectors/s | 1,042.6k | +7.1% |
| `10pct` SQL (two repeats) | 138.2k QPS | 139.4k | +0.9% |
| `distinct10` arrays | 765.2k | 825.0k | +7.8% |

Recall unchanged, no prune fallbacks. The compaction is still well below memory bandwidth and, with scoring (319 us), is the largest GPU cost left; fusing the cutoff test into the scoring kernel would remove the 256 MB written and read again per batch.

**Fused selection: no score matrix.** For tiles without residual masks on one-shard indexes, the matrix gather now scores the same spread sample of rows the matrix path samples (8,192 rows), takes each query's cutoff from it, then scores every row with the module's fused kernel, which keeps only distances at or under the cutoff (`AnnBlackwellScoreBelow`), and radix-selects them; the candidates equal those the matrix path compacts. `ANN_MATRIX_FUSE=off` restores the matrix. One lane per GPU: scoring plus selection 564 -> 441 us a batch (-22%).

| Workload | `rs-*` | `fu-*` | GPU busy |
| --- | ---: | ---: | --- |
| `10pct` arrays (cached) | 1,042.6k vectors/s | 1,033.3k | 93.1% -> 77.5% |
| `10pct` SQL | 139.4k QPS | 140.5k | 95.4% -> 95.7% |
| `distinct10` arrays (masked tiles keep the matrix) | 825.0k | 814.9k | 77.5% -> 73.7% |

Recall unchanged, no prune fallbacks. Throughput is flat: at the cached point the GPUs are no longer the limit (77.5% busy) and the host is (PostgreSQL 97-128 cores, the worker 34-44 and pgbench about 9 of 224). The fused path is kept for its GPU headroom; residual-masked tiles could use it once the mask applies to the candidate lists instead of a score matrix.

## 6. Suggested order

**Harness and test prerequisites.** The first four are needed to read the experiments; items 5-6 gate the P01 and P07 builds.
1. **`frozen_stress.py`:** copy client CPU (already in the per-tag JSON), op-8 and op-31 statistics before and after each run, and per-role cores into `runs.jsonl`.
2. **A cudaEvent stage-timing mode for `SearchGatherMatrix`.** `ANN_KERNEL_TIMING` covers only the large-k path (`cuvs.c:4387-4440`).
3. **A fake-device or CPU backend** so `BuildBitmap`, the grouping keys (`Shareable`, `FilterBucket`, `SameFilterJob`) and lane pickup can be unit-tested.
4. **An exact-result GPU fixture with at least 1024 x candidates eligible rows,** so the rank==32 prune branch is exercised. Current fixtures give rank ~1743.
5. **A distinct10 correctness check for P01 that compares ids, not only recall.** Gates the P01 build.
6. **Unit coverage for the `BuildBitmap` keys/ordinals change.** Gates P07.

**Experiments first: cheap, no build or under 30 lines, high information.**
1. **P01 falsifier.** Serve distinct10 from the category base only and ignore the residual. If it does not reach about 700k vectors/s, P01 dies before any kernel work. The result also bounds G1a, since both remove the same worker cost.
2. **P07 minimal prototype** on the matrix route, with the `server.c:2131` gate split and `BitmapBytes` fixed. Record:
   - worker CPU per predicate;
   - mean latency;
   - memory bandwidth;
   - `bitmapMutex` wait.

   Add u32 shard rows in `BuildBitmap` as a second cell; this is P08's probe. Take a distinct10 gprofng profile on the current build so the fresh10 shares stop standing in for it.
3. **Explain the distinct10 latency and the PostgreSQL 16 ms vs 8.7 ms per call gap:**
   - lock waits on `bitmapMutex` around the eviction qsort;
   - the pageable 1.25 MB D2H plus sync in `DevicePredicate` (`cuvs.c:1670-1680`);
   - host memory bandwidth;
   - whether the worker's 80 MB key streams slow the backends.

   This decides whether CPU-share arithmetic means anything for any uncached item, and which end of the P01/G1a range applies.
4. **Identify the limiter at the cached 780k point (H1 vs H2).** Nothing in this review moves it until it is known.
   - (a) Recompute every `runs.jsonl` point with client CPU, from the per-tag JSON files, if they still exist.
   - (b) Run a baseline and a baseline plus `stress-ng --cpu 32` spread over both sockets. Remember the burner also slows blocking-sync wakeups.
   - (c) Run pgbench from another host.
   - (d) Take a worker gprofng profile at the cached point (none exists).
   - (e) Add cudaEvent stage timing inside `SearchGatherMatrix`.
   - (f) Record `--trace` finished-to-reply times.
   - (g) Sample backend stacks with gdb under immutable, for the post-immutable share of `AnnArrayRead`.
   - (h) Try the one-line `ScheduleYield` experiment (`cuvs.c:104`).
   - (i) Take a 10 s nsys capture of one GPU.

   These outputs gate P02, P03, P04 and the G2b ordering.
5. **Free side measurements:**
   - fused route timing via `--matrix-gather off` with `ANN_KERNEL_TIMING=1` (closes P05);
   - SQL at 96 lanes and 512 clients (closes P15);
   - EXPLAIN VERBOSE of the single-vector statement (gates P14);
   - G3a's offline kept-count check at 2M/4M rows;
   - G4a's zero-code mixed-groups anchor at 8-16 lanes, if multi-tenant traffic matters.

**Builds.**
- **P07 (S).** Land it first: cheap, recall-neutral, and it carries the extra `SearchGather` fixes the reviewers listed.
- **P01 (L).** Build it if experiment 1 passes and the id-level distinct10 check exists. Start with mask option (a). Then run `frozen_recall.py --workload distinct10 --batch 64`; recall must match the exact reference (0.9998 / 0.996). It subsumes P07, P08 and P10 for distinct10/mix10. P07 still pays for predicates that cannot be decomposed.
- **G1a (L).** Only if deployment predicates match a stable cluster key and the step 3 measurements show evaluation or device entries still cost after P01. First build range jobs on a like-for-like table whose key order equals cluster order. Build the permutation (steps 1, 5, 6) only after that table reaches 400k or more.
- **P02 (S).** Worth landing for its small backend saving; add a key sanity check in place of the dropped `ordinal >= rows` guard. It is the precondition for P03 and P11 if experiment 4 shows that PostgreSQL per-id work, or host result handling, binds.
- **G3a (S).** Only if broad filters (≥2M rows at k ≥ 300) matter. It needs k-aware gather limits and the matching change in `SearchGatherMixed`.
- **G4a (L).** Only for multi-tenant deployments, after the GPU anchor, with Param rebinding instead of byte patching.
- **P04.** Only if the stage timing shows `select_k#2` plus fill worth at least about 30-40 us per batch and the GPU shows up as a co-limiter.
- **P14.** Only for deployments that serve single-vector SQL, after EXPLAIN and a profile.
- **Compounding.** On host-bound points (k=1500, unique1), P02, P03 and P11 compound conditionally. On the cached 10pct point no combination has measured support yet.

## Method

This document came from a structured read-only review:
- Parallel code readers first mapped the PostgreSQL side, the worker, the GPU backend and the existing measurements.
- Seven proposers then drafted changes from those maps.
- Each proposal was checked by two independent skeptics, one on evidence and one on feasibility. Each skeptic either refuted the proposal or adjusted its gain, effort and hidden costs.
- A completeness pass then listed gaps and unverified claims. A second round of proposals answered the gaps and went through the same two-skeptic check.

Proposals with no refutation are recommended, those with one are contested, and those with two are rejected. No code was changed, built or run during the review.
