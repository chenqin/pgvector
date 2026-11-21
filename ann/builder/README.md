# ANN index builder

Builds an index directory for the [ANN server](../server/README.md) from a PostgreSQL table or from `ann_index_export` output. It runs offline, on any host that can reach the database (or read the export) and write where the server will read. Nothing here runs on the query path.

## Interface

| | |
| --- | --- |
| Input | A PostgreSQL table with an integer key and a `vector` column (`--dsn`, `--table`), or one or more export directories (`--from-export`), as described in [INDEX_FORMAT.md](../common/INDEX_FORMAT.md#exports) |
| Output | A new index directory (`--out`, must be empty) in [index format version 1](../common/INDEX_FORMAT.md) |
| Consumer | The server, through `SELECT ann_index_load(name, dir)` or protocol operation 11 |

The builder and the server share no code: the directory format is their only contract. CAGRA shard files are written by the cuVS Python package in `requirements-gpu.txt` and must be in a format the server's cuVS reads. The server builds against a source build of the `chenqin/cuvs` fork's `main` ([build flag](../../docs/ann-cuvs-index.md#build-flag)), not a pip cuVS.

## Install

```sh
python3 -m venv /opt/ann-builder && . /opt/ann-builder/bin/activate
pip install -r ann/builder/requirements-gpu.txt   # CAGRA shards; needs an NVIDIA GPU, CUDA 13
pip install -r ann/builder/requirements.txt       # or: no graphs (--no-cagra), no GPU
cp ann/builder/build_cuvs_index.py /opt/ann-builder/bin/
```

The builder is one file, `build_cuvs_index.py`; it imports nothing else from this repository. Tested with Python 3.12.

## Use

```sh
# From a table: one CAGRA shard per visible GPU, FP16 graphs
python3 build_cuvs_index.py --dsn 'dbname=app' --table public.docs --key id --vector embedding \
    --out /data/ann/docs --dtype float16

# With attribute columns the server can filter on, and an FP8 copy that cuVS' CAGRA
# filtered scan scores filtered rows from (FP8 needs FP16 graphs)
python3 build_cuvs_index.py --dsn 'dbname=app' --table public.docs --key id --vector embedding \
    --attribute tenant --attribute lang --dtype float16 --quantize fp8 --out /data/ann/docs-q8

# No graphs: every search scans the FP8 copy (the server keeps FP16 rows made at load)
python3 build_cuvs_index.py --dsn 'dbname=app' --table public.docs --key id --vector embedding \
    --attribute tenant --scan-only --quantize fp8 --shards 4 --out /data/ann/docs-scan

# From a frozen index's export (SELECT ann_index_export('docs_ann', '/data/ann/docs-export'))
python3 build_cuvs_index.py --from-export /data/ann/docs-export --out /data/ann/docs
```

| Option | Default | Effect |
| --- | --- | --- |
| `--dsn`, `--table` | | Source table (`schema.table`) |
| `--key`, `--vector` | `id`, `embedding` | Key and vector columns |
| `--from-export DIR` | | Build from an export instead; repeatable for disjoint key ranges (one collection part) |
| `--out DIR` | required | New or empty output directory |
| `--shards N` | visible GPUs | Contiguous key-range shards; shard N builds on GPU N modulo the GPU count, in parallel |
| `--dtype float32\|float16` | float32 | CAGRA dataset precision; graph builds with `--quantize fp8` need `float16` |
| `--graph-degree`, `--intermediate-graph-degree` | 32, 64 | CAGRA graph parameters |
| `--attribute COLUMN` | | An integer column written as `attr-COLUMN.i32`; repeatable. Text and `integer[]` attributes come from an export |
| `--quantize fp8\|nvfp4` | | Also write a quantized copy: `fp8` for cuVS' CAGRA filtered scan, which re-scores its candidates from FP16 rows, so the builder refuses it beside `float32` graphs; `nvfp4` for the server's Blackwell module (compute capability 10.x or 12.x) |
| `--scan-only` | | No graphs: shards served by quantized scans only (needs `--quantize`; any `--dtype`). With `fp8` the server holds each shard's rows in FP16 on its GPU for the scan's re-scoring |
| `--no-cagra` | | Keys, vectors and attributes only, for CPU servers and tests |
| `--limit N` | | Export at most N rows (testing) |

`python3 build_cuvs_index.py --help` describes each option in full. Quantizing 10M × 128 took 5 s; a 100,000 × 64 CAGRA shard took 2 s on one B200.

## Deploy the output

Copy or mount the directory where the server can read it, then load it from SQL:

```sql
SELECT ann_index_load('docs', '/data/ann/docs');   -- path on the server's host
```

Directories are immutable. To change an index, build a new directory and load it under a new name.
