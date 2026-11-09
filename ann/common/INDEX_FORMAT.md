# ANN index directory format, version 1

An index directory is the contract between its writers and the ANN server:

| Writer | What it writes |
| --- | --- |
| [`ann/builder/build_cuvs_index.py`](../builder/README.md) | A servable index: keys, vectors, CAGRA shards, optional quantized copies and attributes |
| `ann_index_export(index, dir)` in PostgreSQL (`src/annexport.c`) | An export: a frozen `ann` index's keys, vectors and attribute columns, with `shards 0`. The builder's `--from-export` input |

The server's reader is `ann/server/indexfile.c` (`ParseManifest`, `AnnIndexFileOpen`); what it accepts is the definition. `ann_index_load(name, dir)` sends the path to the server (operation 11, [PROTOCOL.md](../server/PROTOCOL.md)), so the path must be valid on the server's host.

## Rules

- **Immutable.** A directory is never modified after its manifest exists. Writers refuse a non-empty output directory, and changed data means a new directory loaded under a new name.
- **Manifest last.** Writers write `manifest.tmp` and rename it to `manifest` after every other file, so a partial directory never loads.
- **Little-endian**, no headers in data files: a file's size follows from the manifest.
- **Row order is key order.** Row *i* of every per-row file is the *i*-th smallest key. Keys are strictly increasing, which the server checks at load.
- **File names** in the manifest are plain names in the directory (no `/`).

## Manifest

A text file, one `key value...` line each. Lines starting with `#` and blank lines are ignored; any other unknown or malformed line makes the load fail.

| Line | Required | Meaning |
| --- | --- | --- |
| `pgvector-ann-index 1` | yes | Format version |
| `dimensions D` | yes | 1 to the server's maximum dimensions |
| `rows N` | yes | N > 0 |
| `metric l2` | | The only metric |
| `dtype float32\|float16` | | The CAGRA dataset's precision (default float32). `vectors.f32` is always FP32 |
| `source SCHEMA.TABLE` | | The PostgreSQL table the rows came from; `ann_index_load` checks its row count |
| `key COLUMN`, `vector COLUMN` | | The table's key and vector columns |
| `shards S` | | Shard count. 0 marks an export, which the server doesn't serve |
| `shard I FIRST ROWS FILE` | | Shard I (0, 1, … in order) covers rows FIRST to FIRST + ROWS − 1; shards are contiguous and cover every row. FILE is a CAGRA file, or `-` for a shard without a graph (`--scan-only`) |
| `attribute NAME FILE` | | An attribute column (up to 32): FILE holds one i32 per row, NULL as −2147483648 |
| `dictionary NAME STRINGS STARTS` | | Makes attribute NAME a text attribute: its i32 values are codes into this dictionary |
| `multivalued NAME OFFSETS` | | Makes attribute NAME an integer array: FILE holds every row's elements and OFFSETS their bounds |
| `quant nvfp4\|fp8` | | A quantized copy of the vectors is present |
| `generation G` | | 16 lowercase hex digits: the frozen index generation the rows came from. The server answers status 12 to requests naming another generation |
| `# member SOURCE FIRST ROWS GENERATION` | | A comment: one export the builder combined into this index |

The cuVS server refuses an index that mixes graph and graphless shards, or whose shards are all `-` without `quant`.

## Files

| File | Present | Content |
| --- | --- | --- |
| `keys.i64` | always | N strictly increasing i64 keys |
| `vectors.f32` | always | N × D FP32, row-major. Memory-mapped by the server for exact routes and re-scoring |
| `shard-I.cagra` | per graph shard | cuVS 26.10 CAGRA serialization of the shard's rows with its dataset (`cagra.save(..., include_dataset=True)`). Neighbour ids are rows within the shard |
| `attr-NAME.i32` | per attribute | N i32, or for `multivalued` every row's elements in row order |
| `attr-NAME.offsets` | per `multivalued` | N + 1 u32: row *i*'s elements are `[offsets[i], offsets[i+1])`; starts at 0, non-decreasing |
| `attr-NAME.strings`, `attr-NAME.starts` | per `dictionary` | The strings' bytes concatenated; C + 1 u32 starts, so string *c* is `strings[starts[c]:starts[c+1]]` |
| `vectors.nvfp4` | `quant nvfp4` | Per row `ceil(D / 16) × 8` bytes: two E2M1 values a byte, the even dimension in the low nibble, zero padded to a multiple of 16 dimensions |
| `scales.e4m3` | `quant nvfp4` | Per row one E4M3 scale per 16 dimensions |
| `vectors.e4m3` | `quant fp8` | Per row D rounded up to 16 E4M3 bytes, zero padded |
| `rowscale.f32` | `quant` | One FP32 scale per row. A value decodes as `e2m1 × block scale × row scale` (NVFP4) or `e4m3 × row scale` (FP8) |

## Exports

`ann_index_export` writes `pgvector-ann-index 1`, `dimensions`, `rows`, `metric l2`, `dtype float32`, `source`, `key`, `vector`, `shards 0`, the attribute lines and `generation`, with `keys.i64`, `vectors.f32` and the attribute files: everything above except shards and quantized copies. `build_cuvs_index.py --from-export DIR` (repeatable for several partitions with disjoint keys) links or copies those files and writes the shards. Combined exports get a generation hashed from the members' and one `# member` line each.

## Compatibility

Version 1 adds features only as new optional lines and files; a server that doesn't know a line refuses the directory rather than misread it. CAGRA files are tied to the cuVS version (26.10): rebuild the shards when the server's cuVS changes.
