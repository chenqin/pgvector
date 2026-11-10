#!/usr/bin/env python3
"""Stress PostgreSQL ANN serving with Muse-shaped requests: several filtered ANN
sub-requests per request, run in parallel, from an ANN capture.

A Muse SCORE_BLEND request runs its corpus sub-requests in parallel (one per
ranker: L1 CPC/CPM, high utility, conversion, ...), each an ANN search with its
own index, k and filter; the request takes as long as its slowest sub-request.
The capture (ann-capture-*.json, written by the Matcha capture tool) holds that
translation: per sub-request the ANN group served, k, the query vector and the
filter as SQL over the frozen index's attributes with its terms.

    python3 benchmarks/ann/muse_stress.py --dsn "$DSN" --capture capture.json \\
        --mode mixed --clients 1,8,32,64 --seconds 20 --output results.jsonl

Modes:
  replay  the captured sub-requests as they are (their vectors, groups, filters)
  mixed   each sub-request keeps its tag, k and filter shape, but its group is
          drawn from every loaded group of the same dimension, its vector from
          that group's rows plus noise, and its pg_ids terms from a pool of
          --distinct-filters random subsets of the captured terms; a share of
          sub-requests (--unfiltered-share) run without a filter

Every --recall-every-th request is kept and checked against exact SQL
(ann.enable_frozen_search = off) after the timed window, so the check does not
load the run. --socket adds the worker's statistics (operation 8) before and
after each level. Needs psycopg 3; run against a test database.
"""

import argparse
import asyncio
import json
import math
import multiprocessing
import random
import socket
import struct
import time

import psycopg


def control(path, operation, payload=b''):
    """A worker control request (operation 8: statistics); {} when unavailable."""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(2)
            connection.connect(path)
            connection.sendall(b'PVAN' + struct.pack('<HHQQII', 1, operation, 1, 0, len(payload), 0) + payload)
            data = b''
            while len(data) < 32:
                data += connection.recv(32 - len(data)) or b'!'
            size, status = struct.unpack('<II', data[24:32])
            body = b''
            while len(body) < size:
                chunk = connection.recv(size - len(body))
                if not chunk:
                    break
                body += chunk
            return json.loads(body) if status == 0 else {}
    except (OSError, ValueError):
        return {}


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    return round(values[min(len(values) - 1, int(math.ceil(p / 100 * len(values))) - 1)], 3)


class Workload:
    """The captured sub-requests and what mixed mode draws from."""

    def __init__(self, args, conn):
        capture = json.load(open(args.capture))
        cur = conn.cursor()
        cur.execute(f"SELECT stem, table_name, dimension, worker_index FROM {args.group_table} "
                    f"WHERE worker_index IS NOT NULL")
        found = cur.fetchall()
        self.groups = {stem: (table, dim) for stem, table, dim, _ in found}
        self.worker_index = {table: index for _, table, _, index in found}
        cur.execute(f"SELECT fid, code FROM {args.schema}.terms WHERE fid IS NOT NULL")
        codes = dict(cur.fetchall())
        self.schema = args.schema
        self.subs = []
        for q in capture['queries']:
            if q['repeat'] != 0:
                continue
            if q['group'] not in self.groups:
                raise SystemExit(f"capture group {q['group']} is not loaded here")
            clauses = q['filter']['clauses']
            mapped = []
            for clause in clauses:
                values = [t['code'] if t.get('code') is not None else codes.get(t.get('fid')) for t in clause['terms']]
                mapped.append(sorted({v for v in values if v is not None}))
            self.subs.append({'tag': q['tag'], 'k': q['k'], 'group': q['group'], 'vector': q['vector'],
                              'sql': q['filter']['sql'], 'columns': [c['column'] for c in clauses],
                              'ops': [c['op'] for c in clauses], 'codes': mapped})
        if not self.subs:
            raise SystemExit('the capture has no sub-requests')
        self.mode = args.mode
        self.rng = random.Random(args.seed)
        self.noise = args.noise
        self.unfiltered = args.unfiltered_share
        # Mixed mode: groups by served dimension, a pool of row vectors per group, filter variants per sub-request.
        self.by_dim = {}
        for stem, (table, dim) in self.groups.items():
            self.by_dim.setdefault(dim, []).append(table)
        self.pools = {}
        if self.mode == 'mixed':
            for table, _ in self.groups.values():
                cur.execute(f"SELECT embedding::text FROM {self.schema}.{table} TABLESAMPLE SYSTEM (50) LIMIT 256")
                self.pools[table] = [[float(x) for x in r[0].strip('[]').split(',')] for r in cur.fetchall()]
            self.variants = []
            for sub in self.subs:
                pool = []
                for _ in range(args.distinct_filters):
                    codes_ = []
                    for op, values in zip(sub['ops'], sub['codes']):
                        if op == 'overlap' and len(values) > 1:
                            size = max(1, int(len(values) * self.rng.uniform(0.25, 1.0)))
                            codes_.append(sorted(self.rng.sample(values, size)))
                        else:
                            codes_.append(values)
                    pool.append(codes_)
                self.variants.append(pool)

    def statement(self, sub, table, codes, filtered):
        where, params = 'true', {}
        if filtered:
            where = sub['sql']
            for i, (column, op, values) in enumerate(zip(sub['columns'], sub['ops'], codes)):
                name = f'p{i}'
                if op == 'overlap':
                    where = where.replace('%s::int4[]', f'%({name})s::int4[]', 1)
                    params[name] = values
                else:
                    where = where.replace(f'{column} = %s', f'{column} = %({name})s', 1)
                    params[name] = values[0] if values else -2147483648
        sql = (f"SELECT local_id FROM {self.schema}.{table} WHERE {where} "
               f"ORDER BY embedding <-> %(v)s::vector LIMIT %(k)s")
        return sql, params

    def request(self):
        """One Muse-shaped request: a (tag, sql, params) per sub-request."""
        out = []
        self.last_ann = []
        for i, sub in enumerate(self.subs):
            table, dim = self.groups[sub['group']]
            vector, codes, filtered = sub['vector'], sub['codes'], True
            if self.mode == 'mixed':
                table = self.rng.choice(self.by_dim[dim])
                base = self.rng.choice(self.pools[table]) if self.pools[table] else vector
                vector = [x + self.rng.gauss(0, self.noise) for x in base]
                codes = self.rng.choice(self.variants[i])
                filtered = self.rng.random() >= self.unfiltered
            sql, params = self.statement(sub, table, codes, filtered)
            params['v'] = '[' + ','.join(f'{x:.7g}' for x in vector) + ']'
            params['k'] = sub['k']
            out.append((sub['tag'], sql, params))
            self.last_ann.append((self.worker_index[table], vector, sub, codes, filtered))
        return out

    def frames(self):
        """The last request's sub-requests as worker operation-22 frames (search by predicate program)."""
        out = []
        for index, vector, sub, codes, filtered in self.last_ann:
            program = b''
            if filtered:
                for column, op, values in zip(sub['columns'], sub['ops'], codes):
                    name = column.encode()
                    leaf = OVERLAP if op == 'overlap' else VALUES
                    program += struct.pack('<BB', leaf, len(name)) + name + struct.pack('<BI', 0, len(values))
                    program += struct.pack(f'<{len(values)}i', *values)
                program += bytes([AND]) * (len(sub['columns']) - 1)
            else:
                program = bytes([ALL])
            name = index.encode()
            payload = struct.pack('<I', len(name)) + name + struct.pack('<IIII', len(vector), sub['k'], 0, len(program))
            payload += program + struct.pack(f'<{len(vector)}f', *vector)
            out.append(b'PVAN' + struct.pack('<HHQQII', 1, 22, 1, 0, len(payload), 0) + payload)
        return out


# ann/common/filter.h opcodes
VALUES, AND, ALL, OVERLAP = 1, 5, 8, 10


def emit(args, workload):
    """Write --emit-requests Muse requests as operation-22 frames for benchmarks/ann/muse_direct.c:
    u32 requests, u32 sub-requests, then per sub-request u32 frame size and the frame. A sidecar
    FILE.check.jsonl keeps the first requests' SQL, to check the direct answers against."""
    with open(args.emit_ann, 'wb') as out, open(args.emit_ann + '.check.jsonl', 'w') as check:
        out.write(struct.pack('<II', args.emit_requests, len(workload.subs)))
        for r in range(args.emit_requests):
            request = workload.request()
            for frame in workload.frames():
                out.write(struct.pack('<I', len(frame)) + frame)
            if r < 100:
                check.write(json.dumps([[sql, params, ann[0], ann[2]['k']] for (_, sql, params), ann in
                                        zip(request, workload.last_ann)]) + '\n')


async def session(dsn, settings):
    conn = await psycopg.AsyncConnection.connect(dsn, autocommit=True)
    async with conn.cursor() as cur:
        for setting in settings:
            await cur.execute(setting)
    return conn


async def collect(args, workload, clients):
    """Run `clients` Muse clients in this process; raw measurements."""
    settings = ['SET ann.enable_frozen_search = on']
    if args.force_worker:
        settings.append('SET ann.frozen_setup_cost = 0')
    nsubs = len(workload.subs)
    conns = []
    for _ in range(clients):
        group = []
        for _ in range(nsubs):
            group.append(await session(args.dsn, settings))
        conns.append(group)
    stop = time.monotonic() + args.warmup + args.seconds
    measure_from = time.monotonic() + args.warmup
    request_ms, sub_ms, errors, samples = [], {}, [], []
    counter = [0, 0]

    async def one(conn, tag, sql, params):
        start = time.perf_counter()
        async with conn.cursor() as cur:
            await cur.execute(sql, params)
            rows = await cur.fetchall()
        return tag, (time.perf_counter() - start) * 1000, [r[0] for r in rows]

    async def client(c):
        while time.monotonic() < stop:
            request = workload.request()
            start = time.perf_counter()
            results = await asyncio.gather(*(one(conns[c][i], *request[i]) for i in range(nsubs)),
                                           return_exceptions=True)
            elapsed = (time.perf_counter() - start) * 1000
            if time.monotonic() < measure_from:
                continue
            counter[0] += 1
            failed = [r for r in results if isinstance(r, BaseException)]
            if failed:
                errors.append(str(failed[0])[:200])
                continue
            request_ms.append(elapsed)
            counter[1] += len(results)
            for tag, ms, _ in results:
                sub_ms.setdefault(tag, []).append(ms)
            if args.recall_every and counter[0] % args.recall_every == 0:
                samples.append([(sql, params, ids) for (_, sql, params), (_, _, ids) in zip(request, results)])

    await asyncio.gather(*(client(c) for c in range(clients)))
    for group in conns:
        for conn in group:
            await conn.close()
    return {'request_ms': request_ms, 'sub_ms': sub_ms, 'errors': errors, 'samples': samples,
            'subrequests': counter[1]}


def process_main(args, clients, seed, start_at):
    """One client process: its own workload (and connections), started together with the others."""
    args.seed = seed
    with psycopg.connect(args.dsn, autocommit=True) as conn:
        workload = Workload(args, conn)
    time.sleep(max(0.0, start_at - time.time()))
    return asyncio.run(collect(args, workload, clients))


async def recall_check(args, samples):
    """Recall of kept sub-requests against exact SQL, after the timed window."""
    recalls = []
    exact = await session(args.dsn, ['SET ann.enable_frozen_search = off'])
    async with exact.cursor() as cur:
        for request in samples:
            for sql, params, ids in request:
                await cur.execute(sql, params)
                rows = await cur.fetchall()
                truth = [r[0] for r in rows]
                recalls.append(len(set(ids) & set(truth)) / len(truth) if truth else 1.0)
    await exact.close()
    return recalls


def run_level(args, clients):
    needed = clients * args.nsubs + 2
    with psycopg.connect(args.dsn, autocommit=True) as conn:
        limit = int(conn.execute("SHOW max_connections").fetchone()[0])
        used = conn.execute("SELECT count(*) FROM pg_stat_activity WHERE backend_type = 'client backend'").fetchone()[0]
    if needed + used > limit:
        return {'mode': args.mode, 'clients': clients, 'error':
                f'needs {needed} connections ({clients} clients x {args.nsubs} sub-requests), '
                f'{limit - used} free under max_connections {limit}'}
    processes = max(1, min(args.processes, clients))
    shares = [clients // processes + (1 if i < clients % processes else 0) for i in range(processes)]
    start_at = time.time() + 2 + 0.2 * processes  # after every process has built its workload
    before = control(args.socket, 8) if args.socket else {}
    with multiprocessing.get_context('spawn').Pool(processes) as pool:
        parts = pool.starmap(process_main, [(args, n, args.seed + 1000 * i, start_at) for i, n in enumerate(shares)])
    after = control(args.socket, 8) if args.socket else {}
    request_ms = [x for p in parts for x in p['request_ms']]
    sub_ms = {}
    for p in parts:
        for tag, values in p['sub_ms'].items():
            sub_ms.setdefault(tag, []).extend(values)
    errors = [e for p in parts for e in p['errors']]
    recalls = asyncio.run(recall_check(args, [s for p in parts for s in p['samples']]))
    delta = {key: after[key] - before.get(key, 0) for key in after
             if isinstance(after.get(key), (int, float)) and isinstance(before.get(key), (int, float))}
    return {
        'mode': args.mode, 'clients': clients, 'processes': processes, 'seconds': args.seconds,
        'requests': len(request_ms), 'requests_per_s': round(len(request_ms) / args.seconds, 1),
        'subrequests_per_s': round(sum(p['subrequests'] for p in parts) / args.seconds, 1),
        'errors': len(errors), 'first_error': errors[0] if errors else None,
        'request_ms': {p: percentile(request_ms, p) for p in (50, 95, 99)},
        'sub_ms': {tag: {'p50': percentile(v, 50), 'p99': percentile(v, 99), 'n': len(v)}
                   for tag, v in sorted(sub_ms.items())},
        'recall': {'checked': len(recalls), 'mean': round(sum(recalls) / len(recalls), 4) if recalls else None,
                   'min': round(min(recalls), 4) if recalls else None},
        'worker': delta,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dsn', required=True)
    parser.add_argument('--capture', required=True, help='ann-capture-*.json')
    parser.add_argument('--mode', choices=['replay', 'mixed'], default='mixed')
    parser.add_argument('--clients', default='1,8,32')
    parser.add_argument('--seconds', type=float, default=20)
    parser.add_argument('--warmup', type=float, default=3)
    parser.add_argument('--schema', default='seg_ads', help='schema of the group tables and terms')
    parser.add_argument('--group-table', default='matcha_pgvector.ann_group')
    parser.add_argument('--distinct-filters', type=int, default=64,
                        help='mixed: pg_ids variants per sub-request (1: every request repeats one filter)')
    parser.add_argument('--unfiltered-share', type=float, default=0.0, help='mixed: share of unfiltered sub-requests')
    parser.add_argument('--noise', type=float, default=0.01, help='mixed: Gaussian noise added to sampled vectors')
    parser.add_argument('--force-worker', action='store_true',
                        help='SET ann.frozen_setup_cost = 0 (small segments otherwise plan locally)')
    parser.add_argument('--recall-every', type=int, default=200, help='check every Nth request against exact SQL; 0: off')
    parser.add_argument('--socket', help='worker socket, for statistics before and after')
    parser.add_argument('--processes', type=int, default=8,
                        help='client processes the clients are spread over (one Python process saturates first)')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--output', help='append one JSON line per level')
    parser.add_argument('--emit-ann', metavar='FILE',
                        help='write the requests as worker frames for muse_direct.c instead of running')
    parser.add_argument('--emit-requests', type=int, default=20000)
    args = parser.parse_args()
    with psycopg.connect(args.dsn, autocommit=True) as conn:
        workload = Workload(args, conn)
    args.nsubs = len(workload.subs)
    if args.emit_ann:
        emit(args, workload)
        print(json.dumps({'emitted': args.emit_requests, 'subrequests': args.nsubs, 'file': args.emit_ann}))
        return
    print(json.dumps({'subrequests': [(s['tag'], s['k'], s['group'].split('.')[1]) for s in workload.subs],
                      'groups': len(workload.groups)}))
    for clients in [int(c) for c in args.clients.split(',')]:
        result = run_level(args, clients)
        print(json.dumps(result), flush=True)
        if args.output:
            with open(args.output, 'a') as out:
                out.write(json.dumps(result) + '\n')


if __name__ == '__main__':
    main()
