# Frozen ann indexes: forward, inverted and GPU ANN over one numbering

A frozen `ann` index is built once from a table that can no longer change. In one pass it stores three indexes that number rows the same way: ordinal *i* is the row with the *i*-th smallest key.

| Index | What the frozen pages hold | Used by |
| --- | --- | --- |
| Forward (ordinal → row) | heap TID, key, vector, INCLUDE attribute values | the frozen search route, index-only scans, `ann_frozen_keys`, the export |
| Inverted (value → ordinals) | per attribute: distinct values, offsets, postings, bitsets of frequent values, and a dictionary for text | predicate programs evaluated in PostgreSQL, `ann_frozen_filter`, `ann_frozen_keys`, `ann_filter_bitmap_build` |
| ANN | exported to the cuVS builder, which adds CAGRA shards and quantized copies | the GPU worker, through the frozen route and the `ann_*_search` functions |

The GPU worker already numbers rows by key rank (`keys.i64` is sorted), so the export is byte-for-byte what `scripts/ann/build_cuvs_index.py` writes from SQL. A filter evaluated on either side selects the same bits: PostgreSQL (from the pages) and the worker (from its postings) run the same evaluator, `ann/filter.c`. A content hash (the generation) in the index, the exported manifest, the catalog and every request header ties the GPU index to the table.

This implements the [plan](ann-immutable-indexes-plan.md), phases 1 to 4.

## Usage

```sql
-- 1. load the data, then make the table immutable
COPY docs FROM '/data/docs.csv' (FORMAT csv);
SELECT ann_freeze('docs');      -- installs the ann_frozen_guard trigger
VACUUM (FREEZE) docs;           -- every page all-frozen in the visibility map

-- 2. build: the first INCLUDE column is the key, the rest are attributes
CREATE INDEX docs_ann ON docs USING ann (embedding vector_l2_ops)
    INCLUDE (id, tenant, category, lang) WITH (frozen = true);

-- 3. GPU index: export the pages, build CAGRA from them, load (the generation is checked)
SELECT ann_index_export('docs_ann', '/data/ann/docs-export');
```

```sh
python3 scripts/ann/build_cuvs_index.py --from-export /data/ann/docs-export \
    --out /data/ann/docs --shards 8 --quantize nvfp4
```

```sql
SELECT ann_index_load('docs', '/data/ann/docs');

-- 4. plain SQL: the planner sends the predicate to the GPU node
SELECT id, tenant, embedding <-> $1 AS distance
FROM docs
WHERE tenant IN (7, 9) AND category BETWEEN 10 AND 19 AND NOT lang = 'de'
ORDER BY embedding <-> $1 LIMIT 10;
--  Limit
--    ->  Custom Scan (AnnFrozenSearch) on docs
--          Search Route: frozen index predicates on GPU worker "docs"
--          Columns: index pages when all-visible
--          Filter Rows: 41873   Heap Fetches: 0
```

The other functions:

```sql
SELECT * FROM ann_frozen_info('docs_ann');            -- generation, rows, attributes, distinct values, pages
SELECT * FROM ann_frozen_keys('docs_ann', 'tenant', ARRAY[7, 9]);       -- keys of rows with tenant 7 or 9
SELECT eligible, bitmap FROM ann_frozen_filter('docs_ann', 'tenant = 7 AND category < 50');  -- from pages
SELECT eligible, bitmap FROM ann_index_filter('docs', 'tenant = 7 AND category < 50');       -- on the GPU node
SELECT ann_filter_bitmap_build('docs', 'category IN (1, 2)');   -- stored filter, evaluated from pages
```

## The freeze contract

`CREATE INDEX ... WITH (frozen = true)` checks, in order:

- **A permanent table, plain columns.** No unlogged or temporary tables, no partial or expression indexes.
- **Column types.** The vector column has dimensions. The first INCLUDE column (the key) is `smallint`, `integer` or `bigint`. Attributes (the other INCLUDE columns) are `smallint`, `integer`, `bigint` (values that fit an `integer`), `boolean`, `date` (not `-infinity`), `text` or `varchar`. Every attribute is stored as an int32, the type the GPU worker filters on: integers as themselves, booleans as 0/1, dates as days, and text as codes of a dictionary sorted in the column's collation. `enum` and other types are refused.
- **The guard.** A `BEFORE INSERT OR UPDATE OR DELETE OR TRUNCATE ... FOR EACH STATEMENT` trigger, enabled `ALWAYS`, that runs `ann_frozen_guard()`. The build identifies it by the function it calls, not by its name. `ann_freeze(table)` creates it and returns `false` if it's already there. Because it's a statement trigger set to `ALWAYS`, it also stops `COPY`, `MERGE`, writes under `session_replication_role = replica`, and statements that match no rows. Load the data before calling `ann_freeze`, since `TRUNCATE` (and so `COPY ... FREEZE` into a truncated table) is blocked too.
- **All-frozen pages.** Every heap page must be all-frozen in the visibility map, so every indexed row version is visible to every snapshot and no dead or in-progress version takes an ordinal. All-zero pages, which bulk loads (`COPY`, `CREATE TABLE AS`) leave at the end of a table and VACUUM never marks, count as frozen: they hold no rows. VACUUM leaves pages unfrozen while another session holds them, or until asynchronous commits (`synchronous_commit = off`) are flushed; the error says how many pages are missing, and another `VACUUM (FREEZE)` fixes it.
- **Rows.** No NULL keys or vectors, unique keys, fewer than 2^32 rows. NULL attributes are stored as `-2147483648`, which is therefore rejected as a real value. The builder writes NULL attributes the same way.

After the build, `aminsert` refuses every insert ("cannot insert into frozen ann index"), even if the guard is dropped. To change the table, either drop the index and the guard, write, then freeze, vacuum and build again (a new generation, and the old GPU directory no longer loads), or add rows to a delta table next to it (see [new rows](#new-rows-a-frozen-base-and-a-delta)).

**Dump and restore.** `pg_dump` writes indexes before triggers, and restores load rows with plain `COPY`, so restoring a frozen table's `CREATE INDEX` fails: the guard doesn't exist yet and no page is frozen. The guard is restored right after it. Then run `VACUUM (FREEZE)` and the `CREATE INDEX` again; the generation is the same, so existing GPU directories still load. The guard is per table: a partitioned table needs `ann_freeze` and a frozen index on each frozen partition.

If the guard is dropped and rows are deleted anyway, VACUUM doesn't move any ordinal: it overwrites the deleted rows' TIDs with an invalid TID, and scans skip them, so heap slots can be reused safely. The GPU index still holds those rows until it is rebuilt; the frozen route skips them, and joins back to the table drop them, as for any loaded index.

## Page layout

Block 0 is the metapage (layout version 3; version 2 indexes, without bitset pages or dictionaries, still read). After the existing header it holds the row count, the generation, the guard trigger's OID, and a directory of arrays. Every structure is an array of fixed-width elements over consecutive pages. Element *i* of an array is on block `start + i / perPage` at byte `(i % perPage) × width` of the page contents, with 8,160 bytes per page. Each page's `pd_lower` covers its elements, so full-page WAL images keep them.

| Array | Element | Per page | 10M rows, 128 dims |
| --- | --- | --- | --- |
| TIDs | `ItemPointerData` (6 bytes) | 1,360 | 7,353 pages |
| Keys | `int64` | 1,020 | 9,804 pages |
| Attribute values, per column | `int32` (text: its code) | 2,040 | 4,902 pages |
| Vectors | a complete `Vector` (8-byte header + 4 × dims) | 15 | 666,667 pages |
| Postings, per column | `uint32` ordinals, grouped by value, ascending within a value | 2,040 | 4,902 pages |
| Distinct values, per column | `int32`, ascending | 2,040 | 1 page |
| Offsets, per column | `uint32` start of each value's postings, plus the end | 2,040 | 1 page |
| Bitsets, per column | `uint32` words, `(rows + 31) / 32` per value holding at least `bitset_min` rows (default rows / 32), and the values' positions | 2,040 | 154 pages per value |
| Dictionary, per text column | the distinct strings' bytes in collation order, and a `uint32` offset per code | 8,160 bytes | |

Vectors are stored as whole `Vector` values, so the distance function reads them in place. A value lookup is a binary search over the distinct values, two offsets, then one contiguous run of postings; the keys of those ordinals come from the key pages. A range of values is a contiguous run of codes and so of postings. For text, a value's code comes from a binary search of the dictionary with the column's collation, and string ranges are code ranges because the codes follow the collation.

**Build.** `table_index_build_scan` feeds `(key, TID, vector, attributes)` to a tuplesort on the key (`maintenance_work_mem`). The sorted output assigns ordinals: TIDs, keys and attribute values fill pages reserved for them, while vectors are appended after them, all in one pass. Each attribute also feeds a datum sort of `value × 2^32 + ordinal`, which yields its postings in order; distinct values, offsets and bitset pages follow. A text column first goes through a `(string, ordinal)` sort in its collation, which assigns codes, writes the dictionary and fills its value pages. The build memory beyond the sorts is the distinct values and offsets of one column at a time (plus one int32 per row for a text column). The pages go through shared buffers and are WAL-logged with `log_newpage_range`, like other `ann` builds.

**Generation.** `hash_bytes_extended` chained over every key, vector and attribute value (text: its bytes) in ordinal order, plus the dimensions, column count and row count. The same rows in another table, inserted in another order, get the same generation. `REINDEX` keeps it, and any changed value changes it.

## Predicates

The planner, `ann_frozen_filter`, `ann_index_filter` and `ann_filter_bitmap_build` turn a predicate over the attribute columns into a predicate program ([ann/filter.h](../ann/filter.h)): a postfix list of leaves and operators.

| SQL | Program |
| --- | --- |
| `a = v`, `a IN (...)`, `a = ANY(array)` | `VALUES a v...` |
| `a <> v`, `a NOT IN (...)`, `a <> ALL(array)` | negated `VALUES` (rows whose `a` is NULL never match, as in SQL) |
| `a < v`, `<=`, `>`, `>=`, `BETWEEN` | `RANGE a low high` |
| `a IS [NOT] NULL` | `NULL a`, negated |
| `flag`, `NOT flag`, `flag IS [NOT] TRUE/FALSE/UNKNOWN` | `VALUES flag 1`, `VALUES flag 0`, with `NULL flag` where SQL includes NULLs |
| `AND`, `OR`, `NOT` | `AND`, `OR`; `NOT` is pushed down to the leaves |
| constants out of range, `= NULL`, `IN ()` | `NONE`; `<> ALL('{}')` is `ALL` |

Operands are constants, parameters (including generic plans) or stable expressions without columns; integer operands of any width compare with integer attributes, `date` with `date`, `boolean` with `boolean`, `text` with `text`/`varchar` in the column's collation. A predicate that uses any other column, operator, function of a column, collation or volatile function isn't translated: the planner keeps its ordinary plans, and `ann_filter_bitmap_build` runs the filter as SQL, as before.

The evaluator keeps each intermediate set as a sorted list while it holds fewer than rows / 64 ordinals, and as a bitset otherwise. Lists intersect and merge in one pass; bitsets combine word by word; a prebuilt bitset replaces a frequent value's postings.

## The frozen search route

`ann.enable_frozen_search` (default on) adds a path for every query over a table with a valid frozen index whose first `ORDER BY` key is the distance from its vector column to a constant or parameter, when every restriction on the table translates to a program. The path sits at the scan level and carries the distance ordering, so it also serves partitions and `UNION ALL` branches under PostgreSQL's MergeAppend (see [new rows](#new-rows-a-frozen-base-and-a-delta)). EXPLAIN shows it as `Custom Scan (AnnFrozenSearch)`.

At execution:

1. The scan finds the loaded export of its index's generation: `ann_index_catalog` rows whose `source` is the table and whose `generation` matches, the most recently loaded one. The name is cached per backend, and read again after a failed call (and, when none was loaded, at most once a second).
2. It builds the program, evaluates the query vector and sends both in operation 22, with the generation in the request header and `ANN_PREDICATE_ORDINALS`. The worker evaluates the program (or finds it cached), searches the rows it selects on its usual route (exact, GPU gather, grouped tensor-core scan, filtered CAGRA; plain ANN for a program matching every row), and returns ordinals and distances.
3. Each ordinal's TID comes from the TID pages. When every column the query reads is in the index (the vector, the key, the attributes) and the heap page is all-visible, the row is built from the index pages (`Heap Fetches: 0`); otherwise it's fetched from the heap with the query's snapshot. The table's quals are rechecked on each row.
4. The first request asks for `LIMIT + OFFSET` rows (100 without a constant limit). If the consumer wants more, the scan asks again for four times as many and skips the rows it already returned.

Without a worker for the generation (none loaded, the nodes unreachable, or a worker without operation 22), the scan evaluates the program from the pages and scores the selected rows on the CPU from the vector pages: exact, the complete ordering at once. EXPLAIN ANALYZE reports `Filter Rows` (the program's rows), `Heap Fetches`, `Worker Calls` and `Local Evaluations`.

## Worker operations and caching

| Operation | Payload | Reply |
| --- | --- | --- |
| 22 search by predicate | u32 name length, name, u32 dimensions, k, flags (1: ordinals), program length, program; FP32 query | u32 count, u32 route, u64 eligible rows, then (i64 key or ordinal, f32 distance) per hit |
| 23 evaluate predicate | u32 name length, name, program | u64 eligible rows, then the bitmap as u32 words |

The worker evaluates a program over its attribute postings (`attr-*.i32`, inverted at load) and caches the rows under the program's hash, as a bitmap entry like a pushed one (operation 20): per-shard bitsets, rows for the exact and gather routes, and device copies. Repeated predicates then cost a hash lookup, and searches over the same predicate group into one scan like any cached filter. A program being evaluated by one connection makes others wait for it rather than evaluate it again. Device copies are allocated and freed stream-ordered on a stream per device, so a new predicate doesn't stall the searches running there. At most `--predicate-cache N` (default 1,024) predicate results stay cached per index; the oldest is dropped first. The stats reply (operation 8) adds `predicate_hits`, `predicate_evaluations` and `predicate_eval_us`; operation 15 adds `predicates on`.

## Export

`ann_index_export(index, directory)` writes `manifest` (with `shards 0` and `generation G`), `keys.i64`, `vectors.f32` without the `Vector` headers, and one `attr-COLUMN.i32` per attribute: integers as themselves, booleans as 0/1, dates as days, text as its dictionary codes (so `ann_attribute_search` on a text attribute takes codes; the frozen route translates strings itself). Names that a manifest can't hold (spaces, slashes, more than 63 bytes) are refused. The directory must be absolute and new or empty, and the manifest is written last with `durable_rename`. It needs superuser or `pg_write_server_files`, plus `SELECT` on the table. `build_cuvs_index.py --from-export` hard-links the files into the index directory (copying across file systems), keeps the export's `source`, `key`, `vector`, `attribute` and `generation` lines, and adds shards, `dtype` and `quant`; one export can feed several builds.

`ann_frozen_info`, `ann_frozen_keys` and `ann_frozen_filter` need `SELECT` on the table and refuse tables with row-level security, since they return rows without applying policies.

## Generations on the GPU nodes

- **Catalog.** `ann_index_load` and `ann_index_attach` require a valid frozen index on the manifest's `source` table with the manifest's generation (directories built from SQL have none and load as before), and record the generation in `ann_index_catalog.generation` (added by the 0.9.1 upgrade).
- **Requests.** PostgreSQL puts the generation of the index it means in the request header's generation field: the frozen route uses its index's, and the `ann_*_search` functions the catalog's (cached per backend). A worker serving another generation of that name answers status 12 (`ANN_GENERATION`); the functions read the catalog again and retry once, and the route falls back to the pages.
- **Stored filters.** A filter's id hashes its index name, its text and, for exported frozen indexes, the generation. A rebuilt table's filters get new ids, so the GPU nodes never answer with a bitmap over another generation's ordinals: `ann_index_load` pushes only the stored rows of the current generation, and `ann_bitmap_search` refuses a stored row built for another one (rebuild it with `ann_filter_bitmap_build`).

## New rows: a frozen base and a delta

A frozen table can't take new rows, but a table next to it can. Make both partitions of one table (or branches of a `UNION ALL` view): the frozen base with its frozen index, and a delta that takes inserts, with an ordinary index or none.

```sql
CREATE TABLE docs (id bigint, tenant int, category int, embedding vector(128)) PARTITION BY RANGE (id);
CREATE TABLE docs_delta PARTITION OF docs DEFAULT;
ALTER TABLE docs ATTACH PARTITION docs_base FOR VALUES FROM (0) TO (10000001);   -- frozen, indexed
INSERT INTO docs ...;   -- new keys land in docs_delta

SELECT id FROM docs WHERE tenant = 7 ORDER BY embedding <-> $1 LIMIT 10;
--  Limit
--    ->  Merge Append
--          Sort Key: ((docs.embedding <-> $1))
--          ->  Custom Scan (AnnFrozenSearch) on docs_base   (GPU worker, generation G1)
--          ->  Sort -> Seq Scan on docs_delta                (exact)
```

PostgreSQL merges the two ordered inputs under one `LIMIT`, with its own MVCC for the delta, so the answer equals exact SQL over the whole table whenever the base's route is exact (the CPU route, and the worker's exact and gather routes; filtered CAGRA is approximate, as everywhere).

**Publishing a new generation** folds the delta in without a gap:

1. Build the next base: `CREATE TABLE docs_base2 AS SELECT * FROM docs` (base and delta), then `ann_freeze`, `VACUUM (FREEZE)`, `CREATE INDEX ... WITH (frozen = true)`: generation G2.
2. Export, build and `ann_index_load('docs_g2', ...)`. The catalog now has a row for G2. Queries keep using G1, because the route looks up the worker by the generation of the table it scans.
3. In one transaction, detach `docs_base` and the delta, attach `docs_base2` and a new empty delta (or delete the moved rows from the old delta). Queries planned after it scan G2 with the G2 worker; no request can reach a bitmap of the other generation, since ids and headers carry it.
4. Drop the old base table: its index and guard trigger go with it. The worker keeps the old name loaded until it restarts.

## Measurements

Each PostgreSQL host (clusters A and B: 64 cores, 32 GB shared buffers) ran pgbench with 512 clients. The worker ran on GPU 0 (28 cores) or all eight GPUs with replication (96 cores). The data is the 10M × 128 benchmark table (`tenant`: 1,000 values, 0.1% each; `category`: 100 values, 1% each), copied into a permanent table, frozen, and indexed with `INCLUDE (id, tenant, category)`. The export was built into one NVFP4 CAGRA shard. Query vectors come from a 1,000-row table. Every query is plain SQL: `SELECT id FROM f10m WHERE <predicate> ORDER BY emb <-> $1 LIMIT 10`.

**Build and evaluation**

| Step | Time |
| --- | --- |
| `ALTER EXTENSION vector UPDATE TO '0.9.1'` on the 0.9.0 benchmark databases | immediate |
| `CREATE INDEX ... WITH (frozen = true)`, 10M rows, 2 attributes (two hosts, independently) | 29.3 s and 32.5 s; 5.5 GB; the same generation on both |
| `ann_frozen_filter`: `tenant = 7` / `category = 7` / `category BETWEEN 10 AND 19` / `tenant BETWEEN 100 AND 199 AND category < 50` / `NOT (category < 50) OR tenant IN (1, 2, 3)` | 1.8 ms / 0.8 ms / 13 ms / 22 ms / 6.8 ms (10k / 100k / 1M / 500k / 5M rows) |
| `ann_index_export` | 7.7 s |
| `build_cuvs_index.py --from-export --shards 1 --quantize nvfp4` | 31 s |
| A new predicate on the worker: evaluation, caching and device upload | 18 ms on average (1,136 predicates) |

**Throughput: one B200, one PostgreSQL host**

| Workload | Frozen route (plain SQL) | Function on the same index |
| --- | --- | --- |
| 0.1%: `tenant = t` (1,000 distinct predicates) | 108k QPS, p99 6.8 ms | `ann_attribute_search`: 127k |
| 1%: `category = c` | 149k, p99 5.9 ms | `ann_attribute_search`: 151k |
| 10%: `category BETWEEN 10d AND 10d + 9` | 81k, p99 11.6 ms | `ann_attribute_search` with 10 values: 543 (it merges 1M ordinals per query) |
| 10%, three hot ranges | 78k | |
| 5%: `tenant BETWEEN ... AND category < 50` | 131k | no function expresses it |
| no filter | 165k | `ann_index_search`: 188k |

TWO_HOST_AND_EIGHT_GPU_RESULTS

**Profile.**

- **PostgreSQL.** gdb samples of backends running the route (perf is blocked on this host) spent 42 of 48 samples at 0.1%, and 25 of 48 unfiltered, in `poll` inside `AnnRemoteCall`, waiting for the worker. Almost all the rest was the client protocol. Building the program, the catalog lookup (cached) and rows from the pages didn't appear. The route's remaining gap to the functions (−15% at 0.1%, −12% unfiltered, none at 1%) is executor work per query on the PostgreSQL host, which the pgbench clients share.
- **GPU.** nsys traces of the worker show the same kernels per search as the attribute route (`search_nvfp4` 17.1 µs against 16.4 µs).
- **Fixed during the measurements.** A first run reached 1,947 QPS at 0.1%. 512 clients missing the same new predicate evaluated it concurrently: 20,680 evaluations at 478 ms each, mostly synchronous `cudaMalloc` and `cudaMemcpy` of device copies, which wait for every search on the device. After single-flight evaluation, a 1,024-entry default cache and stream-ordered uploads, each distinct predicate is evaluated once in 18 ms. That gave 0.1% filters 108k QPS.

**Correctness at scale.** The two hosts built the same generation independently. Filtered results through the route equal exact SQL on the gather routes, and ANN recall@10 through the loaded export was 1.0. Crash recovery was checked by building an index, stopping the cluster with `pg_ctl -m immediate` before a checkpoint, and comparing a 20,000-row ordered scan, a postings lookup and the generation after WAL replay: identical to before the crash and to plain SQL.

## Tests

`test/ann/test_frozen.py` (`ANN_PG_DSN`, superuser) covers:

- **Contract.** Every refusal (INCLUDE without `frozen`, no key, no guard, unfrozen pages, column types, partial index, duplicate and NULL keys, NULL vectors, out-of-range values, unlogged tables). The guard stops INSERT, UPDATE, DELETE, TRUNCATE and COPY, including in replica mode.
- **Scans.** The access method's ordered scans equal sequential exact SQL, as index-only scans with `Heap Fetches: 0`, or as index scans when a column isn't in the index.
- **Predicates.** Twenty predicate shapes (equality, `<>`, IN and NOT IN with NULLs and empty arrays, ranges, BETWEEN, AND/OR/NOT, bigint constants out of range, TRUE/FALSE) select exactly SQL's rows from the pages, with and without bitset pages. Booleans (including IS [NOT] TRUE/UNKNOWN), dates, bigint, text and varchar (ranges under the column's collation, values missing from the dictionary) do too, and another collation isn't translated.
- **The route.** Every shape through the frozen route equals SQL with the route off, rows from pages without heap fetches, columns outside the index from the heap, a generic plan with a parameter `LIMIT` that refills, and unsupported predicates keeping ordinary plans.
- **Base and delta.** A partitioned table with a frozen base and a delta (inserts land in the delta; the base refuses them) answers through Merge Append over the route, equal to SQL.
- **The worker.** With the CPU worker: the worker's bitmaps (operation 23) equal the pages', word for word, for every shape. Routed queries go to the worker and equal SQL. `ann_filter_bitmap_build` builds from the pages and `ann_bitmap_search` equals SQL.
- **Generations.** A catalog naming another generation makes the worker refuse tagged requests (status 12, an error from the functions after a retry) and sends the route to the pages. A new base generation, loaded under a new name, is found by the route. Dropping the old base removes its guard.
- **Postings, generation stability, VACUUM after a dropped guard, export byte identity, and loading**, as before; **cuVS** (`FrozenCuvsTests`, `ANN_CUVS_TEST=1`): an export built into 2 CAGRA shards plus an NVFP4 copy serves filtered searches equal to SQL and ANN recall of at least 0.9.

```sh
ANN_PG_DSN='host=/tmp dbname=postgres' python3 test/ann/test_frozen.py
ANN_CUVS_TEST=1 ANN_PG_DSN='host=/tmp dbname=postgres' python3 test/ann/test_frozen.py FrozenCuvsTests
```

`make -C ann check`, `check-postgres` and `check-cuvs` include them.
