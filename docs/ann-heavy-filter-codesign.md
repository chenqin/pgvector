# Heavy filters: Andromeda's co-design, read for the ANN server

2026-11-14. What Meta's Andromeda does when every request is filtered and no
filter result can be reused, and what the same design comes to on the GPU
worker in `ann/server`.

Nothing was measured for this note. The GPU host was unreachable, so every
figure about this server comes from the existing docs or from reading the
code, and many are arithmetic on those. Every statement about Andromeda's
internals is a guess and is marked as one.

## Summary

- **Heavy filter** here means: every search carries a filter; it passes a few
  percent of the rows or less; the predicate is a long id list; results must
  be exactly filtered; and filters need not repeat.
- **The bar: 150k Muse requests/s sustained** (900k ANN searches/s at six a
  request) with p99 under 50 ms. The plan keeps only routes measured at that
  rate or estimated to reach it. Everything else is listed under
  [Dropped: below 150k](#dropped-below-150k).
- **Andromeda looks built for filters that never repeat.** It puts matching,
  scoring and top-K on one GPU, makes the matcher ten times faster than the
  scorer, and scores only eligible ads (all stated). The reason is a guess:
  every request is a different set of user values, so a match result cannot
  be reused.
- **This server has that shape only while filters recur.** The measured 388k
  requests/s rests on about 2,500 recurring filters: each is evaluated once,
  its rows stay on the GPU, and 16 to 40 searches share one scan.
- **With a new filter on every search, the limits come in this order:** host
  evaluation on a reader thread, the hand-off of the result to the GPU, one
  launch chain per search, and only then row reads. An evaluation costs 20
  to 26 ms, measured (42 ms with 64 at once), so the first limit is a few
  hundred requests/s per reader.
- **There is a hard limit too.** Without shared row reads a request costs six
  scans of its own rows. For filters passing about 2.8% or more, no scan-side
  design recovers the 388k; the ceiling is about 133k requests/s at 85% of an
  assumed memory bandwidth. At 0.4% the same arithmetic allows several
  hundred thousand.
- **So the heavy-filter co-design is not a faster scan.** In order: make
  every stage's work follow the eligible set and not the group; keep the
  eligible set on the GPU from the moment it is computed; let searches with
  different filters share launches; then raise rows per second for one
  query.
- **Measured at 150k on the Muse shard (2026-11-16):** with moves 1, 2, 3
  and 11 in, a 0.1% miss share holds p99 at 26.5 ms (219 ms before), and
  deletes at up to 48 a second leave p99 at 23.7 ms (8.8 s and 109k before).
  A 0.5% miss share or more still collapses throughput. See
  [Measured on the GPU host](#measured-on-the-gpu-host).

## Two regimes

| | Recurring filters (what the Muse test measures) | Non-recurring filters |
| --- | --- | --- |
| Filter evaluation | Once per filter, then a cache hit | On the host for every search |
| Result on the GPU | Cached row list or bitset | Copied or uploaded per search |
| Sharing | One scan serves 16 to 40 searches | None |
| Measured | Scan 388k requests/s at recall 0.998 (K 30 to 182), 214k (K 300 to 1000); two-hop graph 170k at 0.953 | Nothing |

Real traffic sits between the two, at some share of searches that miss the
cache. That share is the first production statistic to get.

## How heavy filtering shapes Andromeda

**Stated in the Oct 2026 post:**
- Index matching evaluates "advertiser-defined targeting logic across tens
  of feature categories", "preserving the full fidelity of production
  targeting semantics", at over 100 billion ads/s.
- The fused scoring kernel "scores every eligible ad", at over 10 billion
  ads/s and about 85% of peak memory bandwidth.
- Matching, scoring and top-K sit on one GPU server, because "data movement
  between pipeline components — not computation — accounted for a
  significant fraction of end-to-end latency".
- Matching "can be re-run across multiple retrieval pathways without becoming
  a bottleneck".

**Guessed, with confidence:**

| Property | Guess | Confidence |
| --- | --- | --- |
| Reuse | Nothing is cached: a match result is good for one request. This is inferred from how ad targeting works in general, not from the posts. Alternative: users with the same values share a match | Medium |
| Matcher | Exact and bit-parallel: bit rows per (category, value), postings for sparse values, a validity bit last | Medium (postings: low) |
| No ANN index | Exhaustive scoring of the eligible set, so a selective filter cannot cost recall | High (stated) |
| Hand-off | One bit per ad, turned into positions the kernel gathers from; never leaves the GPU | Medium |
| Updates | With no clusters or graph, an update is a record write and a few bit flips | Medium |

**What it borrows and what heavy filtering changes.** The left column is the
ordinary GPU retrieval design (Meta's own SilverTorch is the published
example); the right is the likely ads form, all guesses.

| Ordinary design | What heavy, exact filtering requires | Likely form in Andromeda | Confidence |
| --- | --- | --- | --- |
| Hashed (Bloom) signature, false positives tolerated | A wrong exclusion is permanent and no later stage sees it | Exact bit rows; nothing hashed under NOT | Medium |
| The request carries the predicate | The ad carries it; the request carries the user's values | Rows keyed by the request's values | High for the direction |
| Cluster probing, then filter the probed clusters | A selective filter must not cost recall | No ANN structure; score every eligible ad | High (stated) |
| One dot product per item per query | A richer score at no extra memory traffic | Tiles of ads against the user's experts: one read of an ad serves every expert | Medium |
| Fused scoring that reads each item once | Kept | One kernel from gather to score | High (stated) |
| A mask of one bit per item between filter and scorer | Kept | Mask, then positions | Medium |
| Index rebuilt and republished | Ads change all day | A validity bit; new ads in free slots or a fresh segment | Medium for the bit, low for the rest |

**Calibrated to this server's shape** (one group of 46.5M rows, 1.3M eligible
rows a search; arithmetic on stated rates):
- A matcher that examines every row at Andromeda's stated rate takes 465 us
  per search, against 13 us to score the eligible rows at the bandwidth
  ceiling. Matching would be 96% of a search.
- So with a few percent eligible, matching work has to follow postings or
  passing rows, not the group. Whether Andromeda's matcher does is not
  known.
- Andromeda's stated pipeline rate, at six searches of 1.3M eligible rows a
  request, is about 128 requests/s per server. The post offers no way round
  paying for every eligible row per request. It tells what binds, not what
  rate to expect.

## Where this server stands

Requests per second per host. Only the first row is measured.

| Case | Requests/s | Binding step | Basis |
| --- | --- | --- | --- |
| Recurring filters | 388k (K 30 to 182) | GPU | Measured (`docs/ann-muse-production-scale.md:296`) |
| Every filter new, as the code stands | 270 to 640 | Host evaluation on 64 reader threads | Arithmetic on 20 to 40 ms per evaluation, which is one comment in an untracked script |
| Every filter new, evaluation free | at least 10k to 14k; perhaps 33k | One launch chain per search | A floor from the 214k peak; the upper figure is a weak fit |
| 1% of searches miss | 27k to 64k | Host evaluation | Same arithmetic |
| 10% of searches miss | 2.7k to 6.4k | Host evaluation | Same arithmetic |
| Ceiling, lone scans at 2.8% | about 133k | Memory bandwidth | 85% of an assumed 8 TB/s on 52-byte rows |
| Ceiling, lone scans at 0.4% | about 930k | Memory bandwidth | Same |

### What a Muse request carries

From the request generator (`benchmarks/ann/muse_frames.py:100-116`); the
capture itself is not in the repo.

- **Within a request**, searches that shared a filter in the capture share
  one drawn filter, each with its own group and query embedding. One pair is
  identical: the CPC and CPM rankers ask the same group the same question.
- **Across requests**, a filter is one of 64 variants and the embeddings are
  always new. So a (group, filter) pair recurs with different embeddings,
  which is what lets one scan serve many searches.
- The documented pair counts (2,482 = 37 × 64 + 19 × 6) fit a capture with
  two distinct filters: a wide list used on every group and a narrow list on
  the 65-dimension groups. That count is an inference.
- The filter cache is per group (`ann/server/indexfile.c:1102`). The same
  filter sent to several groups in one request is evaluated and cached once
  per group.

So a request needs up to five evaluations where two distinct filters would
do, if groups hold the same documents in the same key order. On the test
index they may: every group has the same row count and the exporter gives a
document the same key in every group
(`benchmarks/ann/scale_export.py:9-12`). Whether production groups do is not
in the docs.

### What a search pays when its filter is new

1. **Host evaluation on a reader thread.** The GPU evaluator refuses the Muse
   leaf (`ann/server/udf.c:57-59`), so `OVERLAP` lists are evaluated from host
   postings in `PredicateBitmap` (`ann/server/server.c:2697-2757`). A reader
   serves many connections, and they all wait.
2. **The hand-off.** The result is uploaded at the filter's second sighting,
   on the reader (`ann/server/server.c:2720-2721`). Until then a lane copies
   it per search.
3. **A launch chain of its own.** A lone Muse search runs the 16-column
   tensor-core tile with one query, in 8 or 9 kernel launches
   (`ann/server/cuvs.c:5466-5477`, `:7034-7037`).
4. **Its own row reads.** Nothing is shared.

Two things in the code make it worse than it need be:
- **The identical pair.** A Muse request carries two identical searches. The
  second is a cache hit, so it counts as a second sighting and uploads the
  result to every copy on the reader. The duplicate merge then compares
  device pointers and does not merge the pair (`ann/server/server.c:1205`).
  This is a reading of the code, not a run.
- **A delete turns recurring filters into new ones.** It drops every cached
  filter of its group (`ann/server/indexfile.c:1288-1308`).

## The two-hop graph route and its cache

The two-hop walk is the one route whose cost does not follow the number of
passing rows, and it batches searches with different filters. Its kernel
needs only the words of a bitset per query. Everything that depends on a
filter recurring is in the worker.

- **First sighting.** No device bitset exists, so the job cannot join a mixed
  batch (`MixedGraphJob`, `ann/server/cuvs.c:4719-4721`). It is searched with
  its own filter's jobs only, and the lane copies the 5.81 MB bitset to the
  GPU for that search (`ann/server/cuvs.c:2726-2737`).
- **Second sighting.** The reader uploads the bitset to each copy.
- **After that.** Jobs carry device pointers and ride mixed batches of up to
  64 (`ann/server/cuvs.c:2697-2709`).
- **Host words stay needed.** Every returned row is re-checked against the
  host bitset (`ShardHits`, `ann/server/cuvs.c:2454-2478`).

Inside the kernel, per query at the released setting (48 walkers, 18
iterations): an estimated 70k to 140k bitset word loads at random positions,
at most 29k dataset rows and 2k to 7k graph rows. By cache lines the bitset
costs about as much as the dataset rows.

Caches inside cuVS on this path:

| Cache | Scope | Note |
| --- | --- | --- |
| Search plan buffers | Rebuilt on every call | Five allocations; one (`num_executed_iterations`) is never used, and the traversed hash is allocated at four times its used size |
| Dataset descriptor | Per lane, LRU of 100 | A miss costs one small kernel launch |
| Filter payload | Process-wide, never freed, keyed by bitset pointer and device | Every search takes a global mutex and a stream wait; in per-query mode the entry is looked up and discarded |
| JIT kernel launcher | By kernel type | Linked at first use |

The filter payload cache can be bypassed in the fork by passing the words
pointer directly (`search_multi_cta_kernel_launcher_jit.cuh:101`, `:58-61`).

With a new filter on every search this route also stops at 270 to 530
requests/s, on host evaluation. Of four ways costed to remove that
dependence, only one has an estimate that reaches 150k: a bitmap over
classes of rows with identical `pg_ids` sets, estimated at 107k to 170k if
classes are few. Per search it costs a few thousand lookups on the host and
a bitmap of kilobytes to megabytes; it keeps the host re-check and an exact
eligible count; its start rows would be drawn from classes. The estimate
comes from one analysis that no reviewer checked. It depends on one unknown,
how many distinct value sets real rows have. The test index has at most
9,862 by construction, so it cannot answer. The other three are listed under
[Dropped: below 150k](#dropped-below-150k).

The walk's evidence is thin for this role. The 170k at recall 0.953 was
measured with every filter cached, at K 30 to 182, on a synthetic tree where
75% to 84% of a passing row's neighbours pass by construction; on the seed
rows that share is 26% to 58%. It is a second route until it is measured on
real labels.

## Direction: what holds 150k

### Routes

| Route | Status at 150k requests/s | Condition |
| --- | --- | --- |
| FP8 scan, filter in the request, filters recurring | Measured. K 30 to 182: peak 388k, p99 under 50 ms to about 250k, recall 0.998. K 300 to 1000: peak 214k, p99 44 ms at 140k and 59 ms at 160k, so about 150k sustained | Searches that miss the filter cache stay rare (below) |
| Two-hop graph walk, filter in the request, filters recurring | Measured at K 30 to 182: peak 170k, p99 23.8 ms at 150k, recall 0.953 with a lowest query at 0.14 | As above; recall accepted; graph and FP16 rows fit; not measured at K 300 to 1000 |
| Unfiltered graph search with Muse's join kept (Option 1 below) | Not measured. Inferred to be the same order as two-hop (170k at a list of 1,536), and unaffected by filter repeats | Keep only if a run at K 300 to 1000 holds 150k |
| Two-hop with a class bitmap, for filters that never repeat | Estimated 107k to 170k, unchecked | Few distinct `pg_ids` sets in production |

**The miss budget.** At 150k requests/s the worker sees 900k searches/s. A
miss is evaluated on the host at 20 to 40 ms (one unverified note). On the
64 reader threads that is 1,600 to 3,200 misses/s, so the measured routes
hold 150k only while fewer than 0.18% to 0.36% of searches miss. With
evaluation moved to a pool of 200 cores (move 2) the budget is 5,000 to
10,000 misses/s, 0.56% to 1.1%. Above that, only the routes whose cost does
not depend on filter repeats can hold 150k.

### Moves that keep the measured routes at 150k

| # | Move | Where | Why it matters at 150k | Gate |
| --- | --- | --- | --- | --- |
| 1 | Deletes without the cache flush | `AnnIndexDelete` (`ann/server/indexfile.c:1288-1308`); reply path (`ann/server/server.c:756-767`, `:806`) | Every delete makes a group's filters misses; one delete per group per second exceeds the miss budget | Deletes per group per second. No host for the host half |
| 2 | Misses off the reader threads; a request's identical searches merged before the lookup | `PredicateBitmap` (`ann/server/server.c:2697-2757`); `AnnIndexBitmapUploadDeferred` (`ann/server/indexfile.c:1122-1156`); `SameSearch` (`ann/server/server.c:1199-1209`) | Raises the miss budget about threefold and stops a miss from stalling a reader's other connections | None for the pool. No host |
| 3 | A host evaluator whose work follows the result | `FromRuns` (`ann/common/filter.c:178-244`); `AnnIndexEvaluate` and `BuildBitmap` (`ann/server/indexfile.c:1317-1352`, `:971-1074`) | Raises the miss budget in proportion to the speed-up | `predicate_eval_us` per miss on Muse programs. No host |
| 4 | Fix the wide end of the fused selection: it fails above 9.0% of a group, and in bands from 6.9% at K 1000 | `FusedEligible` (`ann/server/cuvs.c:5423`); `PrunedSelect` (`:5161-5162`) | Searches there fall to a slower path | Share of production filters above about 7% |
| 11 | One filter result for every group that holds the same documents | The cache key (`ann/server/indexfile.c:1102-1112`); `PredicateBitmap` | Cuts a request's evaluations from five to two, which raises the miss budget 2.5 times | Whether production groups hold the same documents in the same key order. No host for the host half |

Not code, and the largest lever on record: size K to what the root keeps
(214k to 424k requests/s on the same tree).

Move 1 is not safe as a host-only change that simply removes the flush:
stale device row lists and bitsets would still feed the routes, and replies
would come back short. The host half must return an entry's dead rows on top
of K from inside the existing candidates, correct the entry's count, and
keep the flush as a fallback, until a device mask exists. Raising K itself
is not an option at K 1000, where candidates are already 4,000 against a
limit of 4,096.

### Unproven routes that could reach 150k

| # | Move | Where | Gate |
| --- | --- | --- | --- |
| 8 | Two-hop for filters that never repeat, with a class bitmap per query | `MixedGraphJob` (`ann/server/cuvs.c:4713-4725`); `SearchShards` (`:2697-2737`); a class-word mode and start draw in the cuVS fork | Move 9's statistics pass; recall on real labels |
| 9 | Class statistics now | The loader's multi-valued log line (`ann/server/indexfile.c:669-671`) | Mean class of at least 16 rows on real data |
| 12 | A post-filter route, Muse's join done inside the server: an unfiltered search over-fetches, then each candidate's `pg_ids` is checked exactly | Keep each row's `pg_ids` after load (`ann/server/indexfile.c:662-667`); a candidate check before the final top-K; routing in `SearchFilteredGroup` (`ann/server/cuvs.c:7002-7050`) | The unfiltered graph run at K 300 to 1000 holds 150k |

### Dropped: below 150k

| Item | Best figure | Why it cannot reach 150k |
| --- | --- | --- |
| Any route with a new filter on every search, as the code stands | 270 to 640 requests/s (estimate) | Host evaluation |
| Scans with no shared reads (filters that never repeat), today | 10k to 14k, perhaps 33k (estimate) | One launch chain per search |
| Scans with no shared reads, at the bandwidth ceiling, filters of 2.8% or more | about 133k (arithmetic) | Each request reads its own rows six times |
| Move 5: postings on the GPU, scattered into a mask per search | 13k to 73k (estimate, for two-hop) | One scatter per passing row per search |
| Move 6: one launch chain for many (filter, query) segments | Serves lone scans, capped as above | Same cap as the scans it serves |
| Move 7: a kernel for one query per row read | Same | Same |
| Move 10: a dense pass with each query's bit tested at emit | 2.8 useful queries per row at 4% (arithmetic) | Gains too little at Muse filter widths |
| First sighting joins a mixed batch with its host words uploaded | 4k to 11.5k (estimate) | Host evaluation and a 5.81 MB upload per search |
| Filter tested inside the walk from each row's values | 85k to 115k (estimate) | The walk is 1.5 to 2 times slower |
| cuVS's own filtered CAGRA | 21.4k (measured) | Recall 0.72; walks rejected rows |

### Measure first on the GPU host

1. **The miss budget.** Request files with 0.1%, 0.5%, 1% and 2% of searches
   on filters that never repeat, at 150k requests/s. Read `predicate_eval_us`,
   `predicate_evaluations` and p99. It replaces the 20 to 40 ms and gives the
   real budget. No code.
2. **Option 1.** An unfiltered graph search at K 300, 600 and 1000 on the
   Muse shard, at 150k requests/s. No code.
3. **Two-hop at the K the root keeps,** with recall on real embeddings and
   labels if they can be loaded.

### Measured on the GPU host

2026-11-16, the 1/8 shard scan view (37 groups of 46.5M rows), K 30 to 182,
the documented Muse scan flags, `muse_direct` at 150,000 requests/s. Misses
come from `muse_direct --unique-share F`, which adds an absent id to the
filter of a share F of searches; deletes are single keys spread over the
groups. p99 in ms, with the throughput where it fell short:

| Run | `aa6276e` (before) | Moves 1, 2, 3, 11 | Same, `--predicate-resolvers 64` | Same, deletes off the readers (`2830df1`) |
| --- | --- | --- | --- | --- |
| Every filter cached | 23.7 | 24.5 | 22.8 | 23.3 |
| Closed-loop peak | 397k | 394k | 394k | |
| 0.1% misses | 219 | 123 | 26.5 | |
| 0.5% misses | 14,220 at 75k | 13,450 at 78k | 13,080 at 82k | |
| 1% and 2% misses | 17,860 to 20,980 at 25k to 45k | the same | the same | |
| 24 deletes a second | 8,830 at 109k | 1,019 | 769 | 23.3 |
| 48 deletes a second | | | | 23.7 |

- **The cost of a miss:** 20 to 26 ms per evaluation, 42 ms with 64
  resolvers evaluating at once. The pool holds 150k up to a miss share of
  about 0.1% to 0.3%; above that the host path is the limit, as predicted.
- **Deletes patch, they do not flush:** on a full cache every touched entry
  was patched and none re-evaluated. What remained was the reader: a delete
  patched on the reader that read it (up to 700 ms), and every connection on
  that reader waited, which `--trace` showed as time before the frame was
  read. Deletes now run on their own threads.
- **Not yet run:** option 1 (unfiltered graph at K 300 to 1000) and two-hop
  at the root's K.

### Production statistics that decide the rest

1. **The share of searches that miss**, i.e. whose (group, filter) was not
   seen recently, against the 0.18% to 1.1% budget; how many distinct filters
   a request carries; whether its groups hold the same documents (move 11).
2. **Filter anatomy.** The distribution of passing share, especially above
   about 7%; distinct `pg_ids` sets and their sizes (moves 8 and 9).
3. **Deletes per group per second, the K the root keeps, and whether recall
   0.95 with a low tail is acceptable for any traffic.**

## Integrating with Muse's join

How Muse filters, as described by the team (not from code): a leaf runs an
ANN top-K with no filter and a lookup of the request's `pg_ids` in its
inverted index, then keeps the overlap. The test in this repo does something
else: it sends the filter inside the ANN request (operation 22), so it
measures pre-filtering. Muse's own recall against the exact filtered answer
is unknown.

The root keeps 60 to 552 of K = 300 to 1000 per ranker
(`docs/ann-muse-production-scale.md:107-109`), 20% to 55%. If those are the
survivors of the join, that fits the seed-row figure that a passing row's
nearest neighbour passes 26% to 58% of the time. Not confirmed.

| | Option 1: the server replaces Muse's ANN | Option 2: the filter moves into the server | Option 3: both, routed per search |
| --- | --- | --- | --- |
| Call | Operation 22 with the match-all program, same K (`ann/server/server.c:2905-2920`) | Operation 22 with a `pg_ids OVERLAP` program, K = what the root keeps | Either |
| Muse's join | Kept | Not needed | Kept for one route |
| Index files | Keys = Muse document ids, vectors | Plus `pg_ids` as a multi-valued column | Plus `pg_ids` kept per row (move 12) |
| At 150k | Not measured; unfiltered search at K 300 to 1000 must take the graph route, whose candidates stop at 1,024 | Scan 388k to 424k requests/s at K 30 to 182, recall 0.998; two-hop 170k at 0.953 with p99 4.5 ms at 105k; filters cached | Nothing; not built |
| If filters don't repeat | Unaffected: every search batches with every other | Holds 150k only while misses stay within the budget above | Unaffected |
| Results | Today's overlap | The true top-K among passing rows | Either |

Which is likely fastest:
- **If filters repeat**, Option 2: it scores only passing rows, shares scans,
  and asks for fewer results. Within it, the scan carries the peak and
  two-hop gives the lower latency below it.
- **If they mostly don't**, Option 1, if its unfiltered run holds 150k.
  Two-hop with a class bitmap (move 8) is the only filter-in-server route
  estimated to reach 150k for filters that never repeat, and only if
  production has few distinct `pg_ids` sets.

Rollout: export one group; run Option 1 in shadow and compare survivors per
ranker with today's ANN (`benchmarks/ann/muse_compare.py` compares two
workers and would need a Muse-side adapter); run Option 2 in shadow at the K
the root keeps; then choose or route by repeat rate and pass rate.

Open questions for the Muse side: where its ANN runs today; whether the join
runs per leaf or at the root; how `pg_ids` and documents are updated; whether
document ids fit an i64.

## Considered and not pursued

| Idea | Reason |
| --- | --- |
| Bloom signatures | Results must be exact; over a long `OVERLAP` list the false passes exceed the true ones, and under NOT a false positive drops a true row |
| A matcher that examines every row | 465 us per search at Andromeda's own stated rate |
| Cluster probing in place of the scan | Recall 0.987 needed 13% of the eligible rows on SIFT1M; unmeasured on Muse rows |
| Rows reordered so a filter's rows form runs | Not pursued until class statistics exist. The earlier evidence against it was taken at 64 queries a pass and does not speak to a lone scan |
| Int8 for speed | An 8-bit tensor-core form saved nothing in the kernel |
| Upload a new result at its first sighting, for recurring filters | Reverses a measured fix (`docs/ann-10pct-throughput.md:286-300`) |

## Not verified

- The list length (about 2,500 ids) and the mean filter width (about 2.8%)
  each rest on one comment in an untracked plotting script. The cost of a
  miss is now measured (above).
- The memory bandwidth figures (about 8 TB/s for B200, from a secondary
  source) and the 32-byte fetch unit.
- Every capacity figure for non-recurring filters, every per-query load
  count for the walk, and every estimate in the options table.
- The posts were read through extracts; their quotations were not re-checked
  against the pages.

## Background

Sources: the Andromeda posts of
[Dec 2024](https://engineering.fb.com/2024/12/02/production-engineering/meta-andromeda-advantage-automation-next-gen-personalized-ads-retrieval-engine/)
and
[Oct 2026](https://engineering.fb.com/2026/10/28/ml-applications/meta-andromeda-next-gen-ads-retrieval-gpu-co-design/);
for bandwidth, a
[Jarvislabs spec table](https://jarvislabs.ai/ai-faqs/nvidia-b200-specs).

Meta's SilverTorch
([post](https://engineering.fb.com/2026/05/26/ml-applications/silvertorch-index-as-model-new-retrieval-paradigm-recommendation-systems/),
arXiv 2511.14881, [repo](https://github.com/meta-recsys/silvertorch)) was
also read and is not the direction here. It serves content recommendation
with a Bloom filter whose false positives later ranking removes, an Int8 IVF
search and a learned re-scorer. Its filter is inexact and its ANN is cluster
probing, both ruled out above. The one idea from it considered here, a mask bit
per item tested inside the scoring kernel, was dropped as below 150k (moves
6 and 10).
