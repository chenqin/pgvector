#!/usr/bin/env python3
"""Muse-shaped ANN requests for a GPU node, without PostgreSQL.

Mocks what PostgreSQL would send for a Muse request's ANN sub-requests: one
worker operation 22 (search by predicate program) per sub-request, built from
an ANN capture (each sub-request's tag, k, group, vector and pg_ids filter),
the group catalog and term codes (JSON, dumped once from the segment), and the
index directories themselves (dimensions, row vectors to draw queries from).

    # requests for benchmarks/ann/muse_direct.c
    muse_frames.py emit --capture C.json --groups groups.json --terms terms.json \\
        --index-root /data/scale100/i8 --out muse.frames --requests 20000
    # exact recall of the first requests' answers, from the index files
    muse_frames.py check --frames muse.frames --index-root /data/scale100/i8 \\
        --host 127.0.0.1 --port 7700 --token-file token

Mixed requests (the default) keep each sub-request's tag, k and filter shape:
the group is drawn from every group of its dimension, the vector from that
group's rows plus noise, the pg_ids terms from a pool of --distinct-filters
random subsets of the captured terms, and --unfiltered-share run without a
filter. Sub-requests that share a group and vector in the capture (rankers
asking the same group with the same user embedding) keep sharing the drawn
group and vector, and those sharing a filter keep sharing its drawn terms, so
a request's duplicate sub-requests stay duplicates; --independent draws every
sub-request on its own. Worker index names are the group tables (load them
under those names).
"""

import argparse
import json
import os
import random
import socket
import struct

import numpy as np

VALUES, AND, ALL, OVERLAP = 1, 5, 8, 10   # ann/common/filter.h


def manifest(directory):
    fields = {}
    for line in open(os.path.join(directory, 'manifest')):
        parts = line.split()
        if parts and not parts[0].startswith('#'):
            fields.setdefault(parts[0], []).append(parts[1:])
    return fields


def program(columns, ops, codes):
    out = b''
    for column, op, values in zip(columns, ops, codes):
        name = column.encode()
        out += struct.pack('<BB', OVERLAP if op == 'overlap' else VALUES, len(name)) + name
        out += struct.pack('<BI', 0, len(values)) + struct.pack(f'<{len(values)}i', *values)
    return out + bytes([AND]) * (len(columns) - 1)


def frame(index, vector, k, prog):
    name = index.encode()
    payload = struct.pack('<I', len(name)) + name + struct.pack('<IIII', len(vector), k, 0, len(prog))
    payload += prog + np.asarray(vector, dtype='<f4').tobytes()
    return b'PVAN' + struct.pack('<HHQQII', 1, 22, 1, 0, len(payload), 0) + payload


def emit(args):
    rng = random.Random(args.seed)
    nrng = np.random.default_rng(args.seed)
    capture = json.load(open(args.capture))
    groups = {g['stem']: g for g in json.load(open(args.groups))}
    terms = {int(k): v for k, v in json.load(open(args.terms)).items()}
    by_dim = {}
    for g in groups.values():
        by_dim.setdefault(g['dimension'], []).append(g['table'])
    vectors = {}
    for g in groups.values():
        d = os.path.join(args.index_root, g['table'])
        f = manifest(d)
        rows, dims = int(f['rows'][0][0]), int(f['dimensions'][0][0])
        vectors[g['table']] = np.memmap(os.path.join(d, 'vectors.f32'), dtype='<f4', mode='r', shape=(rows, dims))
    subs, variants_by_filter = [], {}
    for q in capture['queries']:
        if q['repeat'] != 0:
            continue
        clauses = q['filter']['clauses']
        codes = [sorted({t['code'] if t.get('code') is not None else terms.get(t.get('fid')) for t in c['terms']}
                        - {None}) for c in clauses]
        fkey = json.dumps(q['filter'], sort_keys=True)
        if fkey not in variants_by_filter:
            variants_by_filter[fkey] = [
                [sorted(rng.sample(v, max(1, int(len(v) * rng.uniform(0.25, 1.0)))))
                 if c['op'] == 'overlap' and len(v) > 1 else v for c, v in zip(clauses, codes)]
                for _ in range(args.distinct_filters)]
        subs.append({'tag': q['tag'], 'k': q['k'], 'group': groups[q['group']], 'vector': q['vector'],
                     'columns': [c['column'] for c in clauses], 'ops': [c['op'] for c in clauses],
                     'codes': codes, 'variants': variants_by_filter[fkey], 'fkey': fkey,
                     'vkey': (q['group'], json.dumps(q['vector']))})
    with open(args.out, 'wb') as out, open(args.out + '.check.jsonl', 'w') as check:
        out.write(struct.pack('<II', args.requests, len(subs)))
        for r in range(args.requests):
            specs, drawn_vectors, drawn_filters = [], {}, {}
            for i, sub in enumerate(subs):
                table, vector, codes, filtered = sub['group']['table'], sub['vector'], sub['codes'], True
                if args.mode == 'mixed':
                    vkey = i if args.independent else sub['vkey']
                    fkey = i if args.independent else sub['fkey']
                    if vkey not in drawn_vectors:
                        t = rng.choice(by_dim[sub['group']['dimension']])
                        rows = vectors[t]
                        drawn_vectors[vkey] = (t, (rows[nrng.integers(len(rows))] +
                                                   nrng.normal(0, args.noise, rows.shape[1])).tolist())
                    if fkey not in drawn_filters:
                        drawn_filters[fkey] = (rng.choice(sub['variants']), rng.random() >= args.unfiltered_share)
                    table, vector = drawn_vectors[vkey]
                    codes, filtered = drawn_filters[fkey]
                prog = program(sub['columns'], sub['ops'], codes) if filtered else bytes([ALL])
                f = frame(table, vector, sub['k'], prog)
                out.write(struct.pack('<I', len(f)) + f)
                specs.append({'table': table, 'k': sub['k'], 'vector': vector, 'filtered': filtered,
                              'columns': sub['columns'], 'ops': sub['ops'], 'codes': codes})
            if r < args.check_requests:
                check.write(json.dumps(specs) + '\n')
    print(json.dumps({'requests': args.requests, 'subrequests': len(subs), 'frames': args.out,
                      'tags': [(s['tag'], s['k']) for s in subs]}))


def recv(sock, n):
    data = b''
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise EOFError('connection closed')
        data += chunk
    return data


def call(sock, frame_bytes):
    sock.sendall(frame_bytes)
    header = recv(sock, 32)
    size, status = struct.unpack('<II', header[24:32])
    return status, recv(sock, size)


def check(args):
    """Exact answers from the index files (keys, vectors, attributes) for the first requests."""
    token = open(args.token_file).read().strip().encode()
    sock = socket.create_connection((args.host, args.port))
    status, _ = call(sock, b'PVAN' + struct.pack('<HHQQII', 1, 13, 1, 0, len(token), 0) + token)
    if status:
        raise SystemExit('authentication failed')
    files = {}

    def table_files(table):
        if table not in files:
            d = os.path.join(args.index_root, table)
            f = manifest(d)
            rows, dims = int(f['rows'][0][0]), int(f['dimensions'][0][0])
            attrs = {a[0]: a[1] for a in f.get('attribute', [])}
            multi = {m[0]: m[1] for m in f.get('multivalued', [])}
            files[table] = (np.fromfile(os.path.join(d, 'keys.i64'), dtype='<i8'),
                            np.memmap(os.path.join(d, 'vectors.f32'), dtype='<f4', mode='r', shape=(rows, dims)),
                            d, attrs, multi)
        return files[table]

    columns = {}

    def column_values(table, column):
        """An attribute column's values and, for a multi-valued column, its row offsets; loaded once."""
        if (table, column) not in columns:
            _, _, d, attrs, multi = table_files(table)
            values = np.fromfile(os.path.join(d, attrs[column]), dtype='<i4')
            offsets = (np.fromfile(os.path.join(d, multi[column]), dtype='<u4').astype(np.int64)
                       if column in multi else None)
            columns[table, column] = (values, offsets)
        return columns[table, column]

    def eligible(table, spec):
        keys, *_ = table_files(table)
        mask = np.ones(len(keys), dtype=bool)
        if not spec['filtered']:
            return mask
        for column, op, codes in zip(spec['columns'], spec['ops'], spec['codes']):
            values, offsets = column_values(table, column)
            hit = np.isin(values, np.asarray(codes, dtype=np.int32))
            if offsets is not None:
                counts = np.add.reduceat(np.append(hit, False).astype(np.int64), offsets[:-1])
                counts[offsets[1:] == offsets[:-1]] = 0
                mask &= counts > 0
            else:
                mask &= hit
        return mask

    stream = open(args.frames, 'rb')
    nreq, nsub = struct.unpack('<II', stream.read(8))
    recalls, by_distance, routes = [], [], {}
    for line in open(args.frames + '.check.jsonl'):
        for spec in json.loads(line):
            size = struct.unpack('<I', stream.read(4))[0]
            frame_bytes = stream.read(size)
            if args.only_table and spec['table'] not in args.only_table.split(','):
                continue
            status, body = call(sock, frame_bytes)
            if status:
                raise SystemExit(f'status {status} for {spec["table"]}')
            count, route, _ = struct.unpack('<IIQ', body[:16])
            routes[route] = routes.get(route, 0) + 1
            hits = np.array([struct.unpack_from('<q', body, 16 + 12 * i)[0] for i in range(count)], dtype=np.int64)
            got = set(hits.tolist())
            keys, vectors, *_ = table_files(spec['table'])
            rows = np.flatnonzero(eligible(spec['table'], spec))
            q = np.asarray(spec['vector'], dtype=np.float32)
            dist = ((vectors[rows] - q) ** 2).sum(1)
            order = np.argsort(dist, kind='stable')[:spec['k']]
            truth = set(keys[rows[order]].tolist())
            recalls.append(len(got & truth) / len(truth) if truth else float(not got))
            # By distance: a hit counts when its exact distance is within the true k-th distance (ties
            # between near-duplicate rows make a hit as good as the true one it displaced).
            if count and truth:
                hit_rows = np.minimum(np.searchsorted(keys, hits), len(keys) - 1)
                hit_dist = np.where(keys[hit_rows] == hits, ((vectors[hit_rows] - q) ** 2).sum(1), np.inf)
                by_distance.append(float(np.mean(hit_dist <= dist[order[-1]] * (1 + 1e-4))))
            else:
                by_distance.append(recalls[-1])
            if args.verbose:
                print(json.dumps({'table': spec['table'], 'k': spec['k'], 'filtered': spec['filtered'],
                                  'eligible': int(len(rows)), 'returned': count, 'route': route,
                                  'recall': round(recalls[-1], 4), 'recall_by_distance': round(by_distance[-1], 4)}))
    print(json.dumps({'checked': len(recalls), 'routes': routes, 'recall_mean': round(float(np.mean(recalls)), 4),
                      'recall_min': round(float(np.min(recalls)), 4),
                      'below_0.99': int(sum(r < 0.99 for r in recalls)),
                      'recall_by_distance_mean': round(float(np.mean(by_distance)), 4)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    e = commands.add_parser('emit')
    e.add_argument('--capture', required=True)
    e.add_argument('--groups', required=True)
    e.add_argument('--terms', required=True)
    e.add_argument('--index-root', required=True)
    e.add_argument('--out', required=True)
    e.add_argument('--requests', type=int, default=20000)
    e.add_argument('--mode', choices=['mixed', 'replay'], default='mixed')
    e.add_argument('--distinct-filters', type=int, default=64)
    e.add_argument('--unfiltered-share', type=float, default=0.1)
    e.add_argument('--noise', type=float, default=0.01)
    e.add_argument('--check-requests', type=int, default=30)
    e.add_argument('--independent', action='store_true',
                   help='draw every sub-request on its own (no duplicates within a request)')
    e.add_argument('--seed', type=int, default=7)
    c = commands.add_parser('check')
    c.add_argument('--frames', required=True)
    c.add_argument('--index-root', required=True)
    c.add_argument('--host', default='127.0.0.1')
    c.add_argument('--port', type=int, default=7700)
    c.add_argument('--token-file', required=True)
    c.add_argument('--verbose', action='store_true', help='one line per checked sub-request')
    c.add_argument('--only-table', help='comma-separated tables: check only their sub-requests (a worker with a subset loaded)')
    args = parser.parse_args()
    emit(args) if args.command == 'emit' else check(args)


if __name__ == '__main__':
    main()
