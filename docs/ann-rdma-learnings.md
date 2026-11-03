# RDMA learnings

What we learned running coordinator → GPU node search traffic over InfiniBand RDMA, against TCP, on one 8× B200 host whose two ConnectX-7 ports (`mlx5_2` for the coordinators, `mlx5_0` for the workers) reach each other through the fabric. Setup and the full results table are in [collections](ann-collections.md#measured-16m--1024-on-one-b200-host-split-into-nodes); the transport itself is in [networked GPU nodes](ann-networked.md#rdma-infiniband-roce).

## Summary

- **RDMA wins latency, not throughput, at this scale.** Pinned tenant queries: 223–238k QPS at p99 1.6–1.8 ms over RDMA, against 224k at p99 5–11 ms over TCP. Queries that reach several nodes lose throughput over RDMA (4-part layout: 48k against 100k).
- **The comparison flatters TCP.** Coordinators and nodes share one host, so TCP runs over loopback: a kernel memory copy that never touches the network. RDMA goes out one adapter and back in the other.
- **The binding cost is active queue pairs.** Every backend holds one reliable-connection (RC) queue pair per node it uses, and the adapters' message rate and latency fall as thousands are active. CPU, the GPUs and the worker's receive path were not the limit.
- **One replica per part over RDMA** (backends no longer rotate over replicas) cut queue pairs 4× on the replicated layout: tenant 172k → 195k QPS at 1,024 clients, 1% filters 94k → 179k.

## How we found it

Per-request timestamps at both ends, joined by request ID (IDs carry the backend's process ID; both ends use the realtime clock of the same host):

```sql
SET ann.worker_trace_sample = 64;   -- coordinator: /tmp/ann-fanout-PID.trace
```
```sh
ann-worker-cuvs ... --trace /tmp/ann-node0.trace --trace-sample 64
python3 benchmarks/ann/fanout_trace.py /tmp/ann-fanout-*.trace /tmp/ann-node*.trace
```

Full layout (one part, four replicas), `tenant = t`, 1,024 clients, mean microseconds per request:

| Stage | TCP | RDMA |
| --- | --- | --- |
| send → worker reads the frame | 80 | 1,618 |
| decode, program, queue | 78 | 15 |
| queued → taken by a search lane | 37 | 256 |
| search (batch and GPU) | 207 | 174 |
| reply posted | 52 | 2 |
| reply posted → coordinator sees it | 312 | 1,345 |
| **send → seen** | **766** | **3,409** |

The GPU and the worker were fast; the time was on the wire, both ways. Then by load (RDMA, rotating replicas):

| Clients | QPS | wire in / out (mean µs) |
| --- | --- | --- |
| 2 | 3.5k | 135 / 29 (p50 29 / 19) |
| 32 | 73k | 61 / 37 |
| 256 | 206k | 265 / 248 |
| 1,024 | 172k | 1,618 / 1,345 |

Wire time is tens of microseconds when idle and grows with concurrency, and throughput *falls* past 256 clients. Something saturates that isn't CPU or GPU.

## What it was not

| Hypothesis | Test | Result |
| --- | --- | --- |
| Receiver not ready (no posted receive) | adapter counters `rnr_nak_retry_err`, `out_of_buffer` over a run | 0 on both adapters |
| Worker wake-up latency | worker RDMA lanes spinning (`--rdma-poll spin`) instead of sleeping | wire times unchanged (1.7 ms / 2.1 ms) |
| Coordinator CPU | per-backend user and system time from `/proc` | 56 of 128 cores; 326 µs per query against 444 µs over TCP |
| CPU starvation | run-queue wait from `schedstat` | low on nodes and backends |
| Worker ingress | decode stage; 4 vs 8 RDMA lanes; separate peer-table lock | decode 15 µs; lanes and lock made no difference |
| Batch hold-off | `--delay-us 200` | no change |
| Missed wakeups | poll timeouts in the coordinator's waits | 0.03% |

Coordinators spinning 5 ms instead of sleeping cut wire time to 64 / 100 µs, but QPS stayed at 173k: spinning backends took the cores, and the queue moved from the wire to the run queue. The bench is closed-loop, so the queue builds wherever the bottleneck is; that's why latency at one stage alone can mislead.

## What it was

**Queue-pair scaling.** perftest between the same two adapters, 4,352-byte sends:

| Queue pairs | Messages per second | MB/s |
| --- | --- | --- |
| 1 | 0.56M | 1,161 |
| 64 | 0.61M | 1,262 |
| 1,024 | 0.61M | 1,275 |
| 4,096 | 0.44M | 917 |

With rotating replicas, 1,024 backends × 4 nodes = 4,096 active queue pairs on each adapter, right where the rate drops; the adapter keeps per-connection state and loses its cache of it. Over TCP the adapter isn't involved here at all, and the kernel multiplexes sockets for free.

Two fabric facts made each message expensive:

- **Active MTU 512 bytes** (set by the subnet manager; the ports support 4,096). A 1,024-dimension FP32 query with its filter is about 4.3 KB, so about nine packets per request; the counters showed about 12 packets per request including replies and acknowledgements.
- **About 10 Gb/s per route** in perftest, though both links run at 100 Gb/s (2× HDR). Only `mlx5_0` ↔ `mlx5_2` routed at all (`mlx5_1` and `mlx5_3` didn't, checked with `ibv_rc_pingpong`); the connection probe keeps a backend on TCP when a pair doesn't route.

And a consequence for the GPUs: requests reached the nodes at half the rate, so a GPU batch held 3.2 searches over RDMA against 7.3 over TCP, and same-filter grouping had less to group. GPUs were 58–81% busy over RDMA, 96–99% over TCP.

## What helped

| Change | Effect |
| --- | --- |
| One shared device context per backend | RDMA setup 35–50 ms → 9–14 ms per node |
| Deadline counted from the first send, not from connecting | first searches no longer time out during setup |
| Connection probe after the hello | unroutable adapter pairs fall back to TCP instead of hanging |
| Worker lanes waiting on completion events (`--rdma-poll wait`) | 2×2 unfiltered 87k → 111–114k; full 1% filters 94k → 139k (spinning lanes took cores from search lanes) |
| One replica per part over RDMA (by process ID; the next only on failure) | full layout: tenant 172k → 195k at 1,024 clients (223k at 256, p99 1.8 ms), unfiltered 197k → 201k, 1% filters → 179k |

## Guidance

- **Use RDMA where a query makes one node request** (replicated whole copies, pinned tenants): same throughput as TCP, 3–6× lower p99.
- **Use TCP for layouts that fan out to several nodes,** until queue pairs are shared; every node a query reaches is another queue pair per backend.
- **Keep queue pairs to about a thousand per adapter:** pool coordinator connections (backends × nodes reached = queue pairs), and prefer layouts where each query reaches few nodes.
- **Check the fabric first:** `ibv_devinfo` (active MTU), `ibstat` (rate), perftest between the actual endpoints, and `ibv_rc_pingpong` for which pairs route. An MTU of 512 multiplies per-request packets by about four against 4,096.
- **Measure per stage, not per query:** under a closed-loop benchmark the queue moves to the bottleneck, so only per-stage timing at both ends shows where it is.

## Open

- A transport that shares connections across processes (DCT or XRC on ConnectX), or a per-coordinator proxy holding few queue pairs, so fan-out layouts stop paying per backend.
- Active MTU 4,096 (a subnet manager change): about two packets per request instead of nine.
- Smaller frames: FP16 query vectors (half the bytes), several queries per frame.
- A real two-host measurement, where TCP also crosses the network and pays its kernel costs.
- Adapter profiling (on-chip queue-pair cache misses) to confirm the mechanism; here it is inferred from the traces, the perftest drop and the gain from fewer queue pairs.
