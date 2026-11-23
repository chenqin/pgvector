# ANN server protocol

This is the interface between the PostgreSQL extension (or any other client) and the ANN server (`ann-worker`, `ann-worker-cuvs`). The extension's client is `src/annremote.c` (TCP and RDMA) and `src/annworker.c` (Unix socket). The server's code is `server.c` and `worker.c`.

The server reads index directories in the format described in [INDEX_FORMAT.md](../common/INDEX_FORMAT.md). Predicate programs use the encoding in [filter.h](../common/filter.h).

## Transports

| Transport | Started with | Who connects | Rules |
| --- | --- | --- | --- |
| stdin/stdout | no `--listen` or `--listen-tcp` | test harnesses, one client | Operations 0–6, one frame at a time |
| Unix socket | `--listen PATH` | PostgreSQL on the same host | Every operation; no authentication (file permissions apply) |
| TCP | `--listen-tcp HOST:PORT --auth-token-file FILE` | PostgreSQL hosts on the network | The first frame must be operation 13 with the token (16+ bytes), within 10 s, or the connection closes. Operations 9 and 10 (shared memory) are refused. Keepalive (10 s idle, 3 probes) |
| RDMA | `--rdma DEVICE[:PORT]` with `--listen-tcp` | PostgreSQL hosts with InfiniBand/RoCE | Set up over an authenticated TCP connection with operation 16; search frames up to the server's RDMA frame size then travel over RDMA, everything else over TCP |

`--listen` and `--listen-tcp` can be used together or separately.

## Frames

Every request and reply is one frame: a 32-byte header, then the payload. All integers are little-endian.

| Offset | Size | Field |
| --- | --- | --- |
| 0 | 4 | Magic `PVAN` |
| 4 | 2 | Version: 1 |
| 6 | 2 | Operation |
| 8 | 8 | Request ID, echoed in the reply |
| 16 | 8 | Generation: for loaded indexes built from a frozen export, the index generation the client expects (0: any) |
| 24 | 4 | Payload bytes (at most 64 MiB) |
| 28 | 4 | Status: 0 in requests; the result in replies |

Replies to one connection can arrive in any order; match them by request ID. A failed request has an empty payload and a nonzero status.

## Statuses

| Status | Meaning | Client action |
| --- | --- | --- |
| 0 | OK | |
| 1 | Invalid request | Report the error |
| 2 | Unsupported by this backend | Report the error |
| 3 | Stale: the index isn't loaded (or is already loaded, for a load) | Reload or report |
| 4 | Out of memory | Report the error |
| 5 | Backend failure | Retry on another node |
| 7 | Overloaded: the admission budget (`--max-pending-mb`) is full | Back off and retry, or try another node |
| 11 | Missing: a cached filter id isn't on this node | Push the bitmap (operation 20) and retry |
| 12 | Generation: the loaded index is another generation than the header names | Reload or report |

Statuses 6 (pending), 8 (timed out), 9 (cancelled) and 10 (closed) are used inside the server's C scheduling API (`batching.h`) and don't appear on the wire.

## Operations

Field notation: `name` is a u32 length followed by that many UTF-8 bytes. Distances are L2 unless a metric says otherwise. Routes in replies: 0 graph, 1 filtered graph, 2 exact.

### Resident index (stdin mode and tests)

| Op | Name | Request | Reply |
| --- | --- | --- | --- |
| 0 | Capabilities | Empty | JSON: compiled backend capabilities (`exact`, `metrics_mask`, `algorithm`, …) |
| 1 | Build | u32 rows, dimensions; row-major FP32 vectors | Empty |
| 2 | Search | u32 queries, dimensions, k, metric (0 L2, 1 negative inner product, 2 cosine), filter (0 none, 1 shared IDs, 2 IDs per query); vectors; allowlists (u32 count, strictly increasing u32 ordinals) | Per query: u32 count, then (u32 ordinal, f64 distance) |
| 3 | Unload | Empty | Empty |
| 4 | Export | Empty | The canonical build payload |
| 5 | Probe | Bounded bytes | Empty. Used as a heartbeat |
| 6 | Info | Empty | u32 rows, dimensions |

### Pipelined row scoring (Unix socket; [guide](../../docs/ann-worker-pipeline.md))

| Op | Name | Request | Reply |
| --- | --- | --- | --- |
| 7 | Gathered search | u32 rows, dimensions, k, metric; FP32 query; FP32 row-major vectors | u32 count, then (u32 row, f64 distance) by distance |
| 8 | Statistics | Empty | JSON: jobs, batches, groups, grouped jobs, rejections, pending bytes, RDMA counters, … |
| 9 | Attach shared memory | Segment name `/pgann.*` | Empty; the segment is mapped read-only for this connection |
| 10 | Shared search | u32 rows, dimensions, k, metric; u64 byte offset; FP32 query | As operation 7 |

### Loaded indexes ([guide](../../docs/ann-cuvs-index.md))

| Op | Name | Request | Reply |
| --- | --- | --- | --- |
| 11 | Load | `name`, directory path | Text: `rows N dimensions D shards S dtype T source NAME`, plus `quant KIND quant_scan on\|off` for quantized indexes |
| 12 | Search by keys | `name`, u32 dimensions, k, key count (UINT32_MAX: no filter); FP32 query; i64 keys | u32 count, u32 route, then (i64 key, f32 distance) per hit |
| 13 | Authenticate | Token bytes | Empty. TCP only, first frame |
| 14 | Search by packed keys | `name`, u32 dimensions, k, key count (UINT32_MAX: none), u32 packed bytes; FP32 query; keys as a zigzag varint first key then varint gaps | As operation 12 |
| 15 | Index info | Index name (no length prefix) | As operation 11, plus the `key` and `vector` columns |
| 16 | RDMA hello | The client's 32-byte queue pair address | u32 largest RDMA frame, the server's 32-byte queue pair address, device name |
| 26 | Unload | `name` | Empty. The name is free at once; the index is freed when its searches finish |
| 29 | Delete keys | `name`, u32 count, i64 keys | u64 rows newly deleted, u64 rows deleted in all |
| 31 | Routing statistics | Index name (no length prefix) | JSON: replica bindings, batches, queries, device load estimates |

### Filters resolved on the server ([guide](../../docs/ann-gpu-filters.md))

| Op | Name | Request | Reply |
| --- | --- | --- | --- |
| 17 | Attribute search | `name`, u32 dimensions, k, attribute `name`, u32 value count; i32 values; FP32 query | As operation 12 |
| 18 | Build bitmap | `name`, u32 key count, u32 packed bytes, varint keys (as operation 14) | u64 eligible rows, then the bitmap as u32 words |
| 19 | Bitmap search | `name`, u32 dimensions, k, bitmap words; FP32 query; the bitmap | As operation 12 |
| 20 | Cache bitmap | `name`, u64 filter id, u32 words; the bitmap | Empty; replaces a bitmap with the same id |
| 21 | Cached bitmap search | `name`, u32 dimensions, k, u64 filter id; FP32 query | As operation 12, or status 11 |

### Predicate programs ([frozen indexes](../../docs/ann-frozen-index.md#worker-operations-and-caching), [collections](../../docs/ann-collections.md))

Programs are [filter.h](../common/filter.h) bytecode over the index's attribute columns. Results are cached under the program's hash.

| Op | Name | Request | Reply |
| --- | --- | --- | --- |
| 22 | Search by predicate | `name`, u32 dimensions, k, flags (1 ordinals instead of keys; 2 append each hit's attribute values; 4 attribute values are collection codes, so text attributes must be remapped with operation 28), program length, program; FP32 query | u32 count, u32 route, u64 eligible rows, then (i64 key or ordinal, f32 distance) per hit; with flag 2, u32 attributes then i32 per hit and attribute (NULL as INT32_MIN) |
| 23 | Evaluate predicate | `name`, program | u64 eligible rows, then the bitmap as u32 words |
| 25 | Collection search | u32 parts, per part (`name`, u64 generation), then operation 22's fields after the name | As operation 22, merged over the parts, eligible rows summed |
| 27 | Dictionary | `name`, attribute `name` | u32 count, count + 1 u32 offsets, the strings, then u32 1 and count i32 collection codes if remapped, else u32 0 |
| 28 | Remap text codes | Operation 27's payload, then u32 count and count i32 codes | Empty. Done once per index, before predicates are cached |
| 30 | Search batch | `name`, u32 dimensions, k, flags (1 ordinals, 8 keep tombstoned candidates), program length, query count (1–64), program, row-major FP32 queries | u32 query count; per query u32 hit count, u32 route, u64 eligible, then (i64 key or u64 ordinal, f32 distance) by distance |

Operation 24 is unused.

## Batching and admission

Every connection feeds one queue per server. Lanes (`--lanes`) take the oldest eligible request and whatever else is queued, up to `--max-batch`, and submit them together; searches over one cached filter are scanned together. `--delay-us` holds a short batch for more requests only while no lane is idle, counted from the oldest request's arrival. Queued and running work is bounded by `--max-pending-mb` (status 7 past it) and per index by `--index-pending-share`.

## Compatibility

The version field is 1. New operations get new numbers; existing request and reply layouts don't change. The PostgreSQL extension and the server should be built from the same release: the extension checks only that operations it sends are answered, not a server version.
