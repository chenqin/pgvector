#!/usr/bin/env python3
"""Compare two ANN workers on the same Muse requests: latency, recall, overlap.

Sends each checked sub-request of a muse_frames.py frames file (operation 22,
the frame as written) to worker A and to worker B, one at a time on an idle
worker, and computes the exact answer from the index files (keys.i64,
vectors.f32, attr-*.i32, multi-valued offsets), so the two routes are judged
on the same queries, filters and k:

    muse_compare.py --frames F --index-root ROOT --capture C.json \\
        --a 127.0.0.1:7821:cagra-fp16 --b 127.0.0.1:7822:fp8-scan \\
        --token-file token --requests 6

Prints one table for the first request (one line per Muse ranker), then the
same columns averaged per ranker over --requests requests, as text and, with
--json, as one JSON line. Each sub-request is sent once to warm the filter
cache (a new filter is evaluated on the host first) and then timed
--repeats times; the median is reported.
"""

import argparse
import json
import os
import socket
import statistics
import struct
import time

import numpy as np


def manifest(directory):
    fields = {}
    for line in open(os.path.join(directory, 'manifest')):
        parts = line.split()
        if parts and not parts[0].startswith('#'):
            fields.setdefault(parts[0], []).append(parts[1:])
    return fields


class Worker:
    def __init__(self, spec, token):
        host, port, self.label = spec.split(':', 2)
        self.sock = socket.create_connection((host, int(port)))
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        status, _ = self.call(b'PVAN' + struct.pack('<HHQQII', 1, 13, 1, 0, len(token), 0) + token)
        if status:
            raise SystemExit(f'{self.label}: authentication failed')

    def recv(self, n):
        data = bytearray()
        while len(data) < n:
            chunk = self.sock.recv(n - len(data))
            if not chunk:
                raise EOFError('connection closed')
            data += chunk
        return bytes(data)

    def call(self, frame):
        self.sock.sendall(frame)
        header = self.recv(32)
        size, status = struct.unpack('<II', header[24:32])
        return status, self.recv(size)

    def search(self, frame, repeats):
        self.call(frame)                       # warm: filter evaluation, upload
        times, body, status = [], b'', 0
        for _ in range(repeats):
            start = time.perf_counter()
            status, body = self.call(frame)
            times.append((time.perf_counter() - start) * 1000)
        if status:
            return {'status': status, 'ms': statistics.median(times), 'keys': [], 'route': -1}
        count, route, eligible = struct.unpack('<IIQ', body[:16])
        keys = [struct.unpack_from('<q', body, 16 + 12 * i)[0] for i in range(count)]
        return {'status': 0, 'ms': statistics.median(times), 'keys': keys, 'route': route, 'eligible': eligible}


class Truth:
    """Exact filtered top-k from an index directory's files."""

    def __init__(self, root):
        self.root, self.tables = root, {}

    def table(self, name):
        if name not in self.tables:
            d = os.path.join(self.root, name)
            f = manifest(d)
            rows, dims = int(f['rows'][0][0]), int(f['dimensions'][0][0])
            self.tables[name] = {
                'keys': np.fromfile(os.path.join(d, 'keys.i64'), dtype='<i8'),
                'vectors': np.memmap(os.path.join(d, 'vectors.f32'), dtype='<f4', mode='r', shape=(rows, dims)),
                'dir': d, 'attrs': {a[0]: a[1] for a in f.get('attribute', [])},
                'multi': {m[0]: m[1] for m in f.get('multivalued', [])}, 'columns': {}}
        return self.tables[name]

    def column(self, t, name):
        if name not in t['columns']:
            values = np.fromfile(os.path.join(t['dir'], t['attrs'][name]), dtype='<i4')
            offsets = None
            if name in t['multi']:
                offsets = np.fromfile(os.path.join(t['dir'], t['multi'][name]), dtype='<u4').astype(np.int64)
            t['columns'][name] = (values, offsets)
        return t['columns'][name]

    def top(self, spec):
        t = self.table(spec['table'])
        mask = np.ones(len(t['keys']), dtype=bool)
        if spec['filtered']:
            for column, codes in zip(spec['columns'], spec['codes']):
                values, offsets = self.column(t, column)
                hit = np.isin(values, np.asarray(codes, dtype=np.int32))
                if offsets is not None:
                    counts = np.add.reduceat(np.append(hit, False).astype(np.int64), offsets[:-1])
                    counts[offsets[1:] == offsets[:-1]] = 0
                    mask &= counts > 0
                else:
                    mask &= hit
        rows = np.flatnonzero(mask)
        q = np.asarray(spec['vector'], dtype=np.float32)
        dist = ((t['vectors'][rows] - q) ** 2).sum(1)
        k = min(spec['k'], len(rows))
        best = rows[np.argpartition(dist, k - 1)[:k]] if k else rows[:0]
        return set(t['keys'][best].tolist()), int(len(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--frames', required=True)
    parser.add_argument('--index-root', required=True)
    parser.add_argument('--capture', help='ANN capture, for the rankers\' tags')
    parser.add_argument('--a', required=True, help='HOST:PORT:LABEL')
    parser.add_argument('--b', required=True, help='HOST:PORT:LABEL')
    parser.add_argument('--token-file', required=True)
    parser.add_argument('--requests', type=int, default=6)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    token = open(args.token_file).read().strip().encode()
    workers = [Worker(args.a, token), Worker(args.b, token)]
    tags = []
    if args.capture:
        tags = [q['tag'].split(':', 1)[-1] for q in json.load(open(args.capture))['queries'] if q['repeat'] == 0]
    truth = Truth(args.index_root)
    stream = open(args.frames, 'rb')
    nreq, nsub = struct.unpack('<II', stream.read(8))
    rows = []
    for r, line in enumerate(open(args.frames + '.check.jsonl')):
        if r >= args.requests:
            break
        for s, spec in enumerate(json.loads(line)):
            frame = stream.read(struct.unpack('<I', stream.read(4))[0])
            exact, eligible = truth.top(spec)
            row = {'request': r, 'sub': s, 'tag': tags[s] if s < len(tags) else f'sub{s}', 'table': spec['table'],
                   'k': spec['k'], 'eligible': eligible}
            got = []
            for w in workers:
                res = w.search(frame, args.repeats)
                found = set(res['keys'])
                row[w.label] = {'ms': round(res['ms'], 2), 'recall': round(len(found & exact) / len(exact), 3) if exact else 1.0,
                                'returned': len(res['keys']), 'route': res['route'], 'status': res['status']}
                got.append(found)
            row['overlap'] = round(len(got[0] & got[1]) / max(1, min(len(got[0]), len(got[1]))), 3)
            rows.append(row)
    la, lb = workers[0].label, workers[1].label
    head = f"{'ranker':34s} {'group':5s} {'k':>4s} {'eligible':>9s} | {la + ' ms':>12s} {'recall':>6s} | {lb + ' ms':>12s} {'recall':>6s} | overlap"
    def line(name, table, k, elig, a_ms, a_rc, b_ms, b_rc, ov):
        return f'{name[:34]:34s} {table:5s} {k:4d} {elig:9d} | {a_ms:12.2f} {a_rc:6.3f} | {b_ms:12.2f} {b_rc:6.3f} | {ov:.3f}'
    print(f'Request 0 (one Muse request, {nsub} rankers), single searches on idle workers:')
    print(head)
    for row in rows:
        if row['request'] == 0:
            print(line(row['tag'], row['table'], row['k'], row['eligible'], row[la]['ms'], row[la]['recall'],
                       row[lb]['ms'], row[lb]['recall'], row['overlap']))
    print(f'\nMean per ranker over {args.requests} requests:')
    print(head)
    summary = []
    for s in range(nsub):
        sub = [row for row in rows if row['sub'] == s]
        if not sub:
            continue
        m = lambda f: statistics.mean(f(x) for x in sub)
        entry = {'tag': sub[0]['tag'], 'k': sub[0]['k'], 'eligible': round(m(lambda x: x['eligible'])),
                 la: {'ms': m(lambda x: x[la]['ms']), 'recall': m(lambda x: x[la]['recall'])},
                 lb: {'ms': m(lambda x: x[lb]['ms']), 'recall': m(lambda x: x[lb]['recall'])},
                 'overlap': m(lambda x: x['overlap'])}
        summary.append(entry)
        print(line(entry['tag'], '*', entry['k'], entry['eligible'], entry[la]['ms'], entry[la]['recall'],
                   entry[lb]['ms'], entry[lb]['recall'], entry['overlap']))
    # A Muse request waits for its slowest ranker.
    for w in (la, lb):
        per = [max(row[w]['ms'] for row in rows if row['request'] == r) for r in range(args.requests)]
        rec = statistics.mean(row[w]['recall'] for row in rows)
        print(f'{w}: request latency (slowest ranker, idle worker) mean {statistics.mean(per):.2f} ms, '
              f'recall mean {rec:.3f}; routes {sorted({row[w]["route"] for row in rows})}')
    if args.json:
        print(json.dumps({'rows': rows, 'summary': summary}))


if __name__ == '__main__':
    main()
