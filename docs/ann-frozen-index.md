# Frozen ann indexes: forward, inverted and GPU ANN over one numbering

A frozen `ann` index is built once from a table that can no longer change. In one pass it stores three indexes that number rows the same way: ordinal *i* is the row with the *i*-th smallest key.

| Index | What the frozen pages hold | Used by |
| --- | --- | --- |
| Forward (ordinal → row) | heap TID, key, vector and INCLUDE attribute values | index-only scans, `ann_frozen_keys`, the export |
| Inverted (value → ordinals) | per attribute: distinct values, offsets, postings | `ann_frozen_keys` |
| ANN | exported to the cuVS builder, which adds CAGRA shards and quantized copies | `ann_index_search`, `ann_attribute_search`, `ann_bitmap_search` on the GPU worker |

The GPU worker already numbers rows by key rank (`keys.i64` is sorted), so the export is byte-for-byte what `scripts/ann/build_cuvs_index.py` writes from SQL, and a filter evaluated on either side selects the same bits. A content hash (the generation) in the index and in the exported manifest lets `ann_index_load` prove that the GPU index and the table hold the same rows.

This implements phases 1 and 2 and the inverted pages of phase 3 of the [plan](ann-immutable-indexes-plan.md). Not yet: the shared predicate evaluator for `AND`/`OR`/ranges, operation 22, prebuilt bitset pages, ordinals in worker replies, and, as a todo for later, phase 4's frozen-base-plus-delta union (until then a changed table is refrozen and rebuilt).

## Usage

```sql
-- 1. load the data, then make the table immutable
COPY docs FROM '/data/docs.csv' (FORMAT csv);
SELECT ann_freeze('docs');      -- installs the ann_frozen_guard trigger
VACUUM (FREEZE) docs;           -- every page all-frozen in the visibility map

-- 2. build: the first INCLUDE column is the key, the rest are attributes
CREATE INDEX docs_ann ON docs USING ann (embedding vector_l2_ops)
    INCLUDE (id, tenant, category) WITH (frozen = true);

SELECT * FROM ann_frozen_info('docs_ann');
--     generation    |   rows   | dimensions | key_column |    attributes     | distinct_values | pages
-- 2c814523d6f919a0  | 10000000 |        128 | id         | {tenant,category} | {1000,100}      | 703437

-- inverted then forward: keys of the rows with tenant 7 or 9, in key order
SELECT * FROM ann_frozen_keys('docs_ann', 'tenant', ARRAY[7, 9]);

-- exact ordered scans from the vector pages, index-only when the columns are in the index
SELECT id, tenant, embedding <-> $1 FROM docs ORDER BY embedding <-> $1 LIMIT 10;
--   Index Only Scan using docs_ann on docs ... Heap Fetches: 0

-- 3. GPU index: export the pages, build CAGRA from them, load with a generation check
SELECT ann_index_export('docs_ann', '/data/ann/docs-export');
```

```sh
python3 scripts/ann/build_cuvs_index.py --from-export /data/ann/docs-export \
    --out /data/ann/docs --shards 8 --quantize nvfp4
```

```sql
SELECT ann_index_load('docs', '/data/ann/docs');
SELECT * FROM ann_attribute_search('docs', $1, 10, 'tenant', ARRAY[7]);
```

`--from-export` hard-links `keys.i64`, `vectors.f32` and the `attr-*.i32` files into the new directory (copying across file systems), keeps the export's `source`, `key`, `vector`, `attribute` and `generation` lines, and adds shards, `dtype` and `quant`. One export can feed several builds (FP32, FP16, NVFP4). Copy the directory to remote GPU nodes as before.

## The freeze contract

`CREATE INDEX ... WITH (frozen = true)` checks, in order:

- **A permanent table, plain columns.** No unlogged or temporary tables, no partial or expression indexes.
- **Column types.** The vector column has dimensions; the first INCLUDE column (the key) is `smallint`, `integer` or `bigint`; further INCLUDE columns are `smallint` or `integer`, the attribute type the GPU worker filters on.
- **The guard.** A `BEFORE INSERT OR UPDATE OR DELETE OR TRUNCATE ... FOR EACH STATEMENT` trigger, enabled `ALWAYS`, that runs `ann_frozen_guard()`. The build identifies it by the function it calls, not by its name. `ann_freeze(table)` creates it and returns `false` if it's already there. Because it's a statement trigger set to `ALWAYS`, it also stops `COPY`, `MERGE`, writes under `session_replication_role = replica`, and statements that match no rows. Load the data and use `COPY ... FREEZE` before calling `ann_freeze`, since `TRUNCATE` is blocked too.
- **All-frozen pages.** Every heap page must be all-frozen in the visibility map, so every indexed row version is visible to every snapshot, and no dead or in-progress version takes an ordinal. VACUUM leaves pages unfrozen while another session holds them (a concurrent autoanalyze left 18 of 714,304 pages on the 10M table), and can't mark pages committed with `synchronous_commit = off` until their commit is flushed. The error says how many pages are missing; run `VACUUM (FREEZE)` again.
- **Rows.** No NULL keys or vectors, unique keys, fewer than 2^32 rows. NULL attributes are stored as `-2147483648`, which is therefore rejected as a real value. The builder writes NULL attributes the same way.

After the build, `aminsert` refuses every insert ("cannot insert into frozen ann index"), even if the guard is dropped. To change the table: `DROP INDEX`, `DROP TRIGGER ann_frozen_guard ON docs`, write, then freeze, vacuum and build again. That gives a new generation, and the old GPU directory no longer loads.

**Dump and restore.** `pg_dump` writes indexes before triggers, and restores load rows with plain `COPY`, so restoring a frozen table's `CREATE INDEX` fails: the guard doesn't exist yet and no page is frozen. The guard is restored right after it. Then run `VACUUM (FREEZE)` and the `CREATE INDEX` again; the generation is the same, so existing GPU directories still load. The guard is per table: a partitioned table needs `ann_freeze` and a frozen index on each partition.

If the guard is dropped and rows are deleted anyway, VACUUM doesn't move any ordinal: it overwrites the deleted rows' TIDs with an invalid TID, and scans skip them, so heap slots can be reused safely. The GPU index still holds those rows until it is rebuilt, and joins back to the table drop them, as for any loaded index.

## Page layout

Block 0 is the metapage (layout version 2). After the existing header it holds the row count, the generation, the guard trigger's OID, and a directory of arrays. Every structure is an array of fixed-width elements over consecutive pages. Element *i* of an array is on block `start + i / perPage` at byte `(i % perPage) × width` of the page contents, with 8,160 bytes per page. Each page's `pd_lower` covers its elements, so full-page WAL images keep them.

| Array | Element | Per page | 10M rows, 128 dims |
| --- | --- | --- | --- |
| TIDs | `ItemPointerData` (6 bytes) | 1,360 | 7,353 pages |
| Keys | `int64` | 1,020 | 9,804 pages |
| Attribute values, per column | `int32` | 2,040 | 4,902 pages |
| Vectors | a complete `Vector` (8-byte header + 4 × dims) | 15 | 666,667 pages |
| Postings, per column | `uint32` ordinals, grouped by value, ascending within a value | 2,040 | 4,902 pages |
| Distinct values, per column | `int32`, ascending | 2,040 | 1 page |
| Offsets, per column | `uint32` start of each value's postings, plus the end | 2,040 | 1 page |

Vectors are stored as whole `Vector` values, so the distance function reads them in place. A value lookup is a binary search over the distinct values, two offsets, then one contiguous run of postings; the keys of those ordinals come from the key pages.

**Build.** `table_index_build_scan` feeds `(key, TID, vector, attributes)` to a tuplesort on the key (`maintenance_work_mem`). The sorted output assigns ordinals: TIDs, keys and attribute values fill pages reserved for them, while vectors are appended after them, all in one pass. Each attribute also feeds a datum sort of `value × 2^32 + ordinal`, which yields its postings in order; distinct values and offsets follow them. The build memory beyond the sorts is the distinct values and offsets of one column at a time. The pages go through shared buffers and are WAL-logged with `log_newpage_range`, like other `ann` builds.

**Generation.** `hash_bytes_extended` chained over every key, vector and attribute value in ordinal order, plus the dimensions, column count and row count. The same rows in another table, inserted in another order, get the same generation. `REINDEX` keeps it, and any changed value changes it.

## Loading with a generation check

`ann_index_export` writes `manifest` (with `shards 0` and `generation G`), `keys.i64`, `vectors.f32` without the `Vector` headers, and one `attr-COLUMN.i32` per attribute. Names that a manifest can't hold (spaces, slashes, more than 63 bytes) are refused. The directory must be absolute and new or empty, and the manifest is written last with `durable_rename`. It needs superuser or `pg_write_server_files`, plus `SELECT` on the table.

The worker reads `generation` from the manifest and reports it in its load and index-info replies. `ann_index_load` and `ann_index_attach` then require a valid frozen `ann` index on the manifest's `source` table with that generation, in addition to the row-count check. Directories without a generation (built from SQL) load as before.

`ann_frozen_info` and `ann_frozen_keys` need `SELECT` on the table and refuse tables with row-level security, since they return rows without applying policies.

## Measurements

One B200 host, PostgreSQL 17.6 (64 GB shared buffers, 8 GB `maintenance_work_mem`), the 10M × 128 benchmark table copied into a permanent table with `tenant` (1,000 values, 0.1% each) and `category` (100 values, 1% each):

| Step | Time |
| --- | --- |
| `VACUUM (FREEZE)` of the freshly loaded table (5.6 GB) | 3.8 s |
| `CREATE INDEX ... WITH (frozen = true)`, 2 attributes | 28.6 s; 5.5 GB index |
| `ann_frozen_keys`: 0.1% / 1% / 10% of rows | 4 ms / 13 ms / 163 ms (10k / 100k / 1M keys); a `count(*)` over the table takes 561 ms |
| `ann_index_export` | 4.9 s |
| `build_cuvs_index.py --from-export --shards 8 --quantize nvfp4` | 23 s |
| `ann_index_load` with the generation check | 8.6 s |
| Exact index-only scan, `LIMIT 10` (all 10M distances on the CPU) | 3.0 s |

Through the loaded export, `ann_attribute_search` matched exact SQL over the same postings on all 10 tenant and 5 category queries (GPU gather route), and `ann_index_search` had recall@10 of 1.0. The exact scan is the CPU fallback; filtered and ANN queries belong on the GPU worker.

Crash safety was checked by building an index, stopping the cluster with `pg_ctl -m immediate` before a checkpoint, and comparing a 20,000-row ordered scan, a postings lookup and the generation after WAL replay: identical to before the crash and to plain SQL.

## Tests

`test/ann/test_frozen.py` (`ANN_PG_DSN`, superuser) covers:

- **Contract.** Every refusal (INCLUDE without `frozen`, no key, no guard, unfrozen pages, column types, partial index, duplicate and NULL keys, NULL vectors, the reserved value, unlogged tables). The guard stops INSERT, UPDATE, DELETE, TRUNCATE and COPY, including in replica mode.
- **Scans.** Ordered scans equal sequential exact SQL, with and without filters, as index-only scans with `Heap Fetches: 0`, or as index scans when a column isn't in the index. Every row comes back once.
- **Postings.** `ann_frozen_keys` equals `WHERE column = ANY(values)` for single values, lists with duplicates and NULLs, absent values, and NULL attributes.
- **Generation.** Stable across `REINDEX` and across tables holding the same rows; changed by a single value.
- **VACUUM after a dropped guard.** Inserts still fail. Deleted rows disappear from scans after VACUUM.
- **Export.** Files equal the builder's `--no-cagra --attribute` output byte for byte; non-empty and relative directories are refused. With the CPU worker: export, then `--from-export`, then `ann_index_load`; `ann_attribute_search` equals SQL; after the table is changed and refrozen, the old directory is refused by generation.
- **cuVS** (`FrozenCuvsTests`, `ANN_CUVS_TEST=1`). An export built into 2 CAGRA shards plus an NVFP4 copy loads into `ann-worker-cuvs`; filtered searches equal SQL and ANN recall is at least 0.9.

```sh
ANN_PG_DSN='host=/tmp dbname=postgres' python3 test/ann/test_frozen.py
ANN_CUVS_TEST=1 ANN_PG_DSN='host=/tmp dbname=postgres' python3 test/ann/test_frozen.py FrozenCuvsTests
```

`make -C ann check`, `check-postgres` and `check-cuvs` include them.
