#!/usr/bin/env python3
"""Recall@k of frozen SQL searches against exact FP64 references.

Samples (query, predicate) pairs of a frozen_stress.py workload, computes the
exact top k of each from the index directory's original FP32 vectors on one
GPU (cached in --truth), then runs the same SQL statement the benchmark runs
through the given worker socket and reports recall per pair and overall.

    python benchmarks/ann/frozen_recall.py --index-dir DIR --socket SOCK \
        --truth /dev/shm/truth-10pct.json --workload 10pct --samples 40
"""

import argparse
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import psycopg

PREDICATES = {
    '0.1pct': ('tenant', lambda v: (v, v), 'tenant = {v}', 100),
    '1pct': ('category', lambda v: (v, v), 'category = {v}', 100),
    '10pct': ('category', lambda v: (v * 10, v * 10 + 9), 'category BETWEEN {lo} AND {hi}', 10),
    'rows50k': ('tenant', lambda v: (v * 5, v * 5 + 4), 'tenant BETWEEN {lo} AND {hi}', 200),
    'rows500k': ('category', lambda v: (v * 5, v * 5 + 4), 'category BETWEEN {lo} AND {hi}', 20),
    'rows2m': ('category', lambda v: (v * 20, v * 20 + 19), 'category BETWEEN {lo} AND {hi}', 5),
    'rows4m': ('category', lambda v: (v * 40, v * 40 + 39), 'category BETWEEN {lo} AND {hi}', 2),
    'mixed10': ('category', lambda v: (v, v + 9), 'category BETWEEN {lo} AND {hi}', 91),
}


def truth(args, pairs, queries):
    path = Path(args.truth)
    cached = json.loads(path.read_text()) if path.exists() else {}
    missing = [p for p in pairs if f'{p[0]}:{p[1]}:{args.workload}:{args.k}' not in cached]
    if missing:
        import cupy as cp
        directory = Path(args.index_dir)
        keys = np.fromfile(directory / 'keys.i64', dtype='<i8')
        vectors = np.memmap(directory / 'vectors.f32', dtype='<f4', mode='r').reshape(len(keys), -1)
        column, bounds, _, _ = PREDICATES[args.workload]
        values = np.fromfile(directory / f'attr-{column}.i32', dtype='<i4')
        for value in sorted({p[1] for p in missing}):
            lo, hi = bounds(value)
            rows = np.flatnonzero((values >= lo) & (values <= hi))
            eligible = cp.asarray(vectors[rows], dtype=cp.float64)
            for q, v in [p for p in missing if p[1] == value]:
                query = cp.asarray(np.asarray(queries[q], dtype=np.float64))
                distance = ((eligible - query) ** 2).sum(axis=1)
                top = cp.asnumpy(cp.argsort(distance)[:args.k])
                cached[f'{q}:{v}:{args.workload}:{args.k}'] = [int(keys[rows[i]]) for i in top]
            del eligible
        path.write_text(json.dumps(cached))
    return cached


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dsn', default=os.environ.get('ANN_PG_DSN'))
    parser.add_argument('--index-dir', required=True)
    parser.add_argument('--socket', required=True)
    parser.add_argument('--truth', required=True)
    parser.add_argument('--workload', choices=sorted(PREDICATES), default='10pct')
    parser.add_argument('--samples', type=int, default=40)
    parser.add_argument('--k', type=int, default=500)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--table', default='public.frozen_stress')
    parser.add_argument('--queries', default='public.frozen_stress_queries')
    parser.add_argument('--batch', type=int, default=0, metavar='N',
                        help='search N consecutive query vectors per ann_frozen_search_batch call, '
                             'as frozen_stress.py --batch-queries does (the worker groups them)')
    parser.add_argument('--index-name', default='frozen_stress')
    parser.add_argument('--immutable', action='store_true', help='with ann.frozen_immutable = on')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    column, bounds, template, choices = PREDICATES[args.workload]
    random.seed(args.seed)
    width = max(args.batch, 1)
    pairs = [(random.randint(1, 1001 - width), random.randrange(choices))
             for _ in range(args.samples)]
    checked = [(q + i, v) for q, v in pairs for i in range(width)]
    with psycopg.connect(args.dsn, autocommit=True) as db:
        queries = {q: json.loads(v) for q, v in db.execute(f'SELECT q, v FROM {args.queries}')}
        reference = truth(args, checked, queries)
        for setting, value in [('ann.worker_socket', args.socket), ('ann.enable_frozen_search', 'on'),
                               ('ann.frozen_mmap', 'on'), ('ann.frozen_setup_cost', '1000'), ('jit', 'off')] + \
                [('ann.frozen_immutable', 'on')] * args.immutable:
            db.execute('SELECT set_config(%s, %s, false)', (setting, value))
        rows = []
        started = time.monotonic()
        for q, v in pairs:
            lo, hi = bounds(v)
            predicate = template.format(v=v, lo=lo, hi=hi)
            if args.batch:
                vectors = '{' + ','.join('"' + json.dumps(queries[q + i]) + '"' for i in range(width)) + '}'
                found = {i: [] for i in range(width)}
                for number, key in db.execute('SELECT query_no, id FROM ann_frozen_search_batch(%s, %s::vector[], %s, %s)',
                                              (args.index_name, vectors, args.k, predicate)):
                    found[number - 1 if 1 <= number <= width else number].append(key)
                results = [(q + i, found.get(i, [])) for i in range(width)]
            else:
                results = [(q, [r[0] for r in db.execute(
                    f'SELECT id FROM {args.table} WHERE {predicate} ORDER BY emb <-> %s::vector LIMIT {args.k}',
                    (json.dumps(queries[q]),))])]
            for query, ids in results:
                want = set(reference[f'{query}:{v}:{args.workload}:{args.k}'])
                rows.append(dict(query=query, value=v, returned=len(ids), unique=len(set(ids)),
                                 recall=len(set(ids) & want) / args.k))
        recalls = [r['recall'] for r in rows]
        summary = dict(workload=args.workload, k=args.k, samples=len(rows), batch=args.batch, socket=args.socket,
                       mean_recall=float(np.mean(recalls)), min_recall=float(np.min(recalls)),
                       short=sum(r['returned'] < args.k for r in rows), seconds=time.monotonic() - started)
    print(json.dumps(summary), flush=True)
    if args.output:
        args.output.write_text(json.dumps(dict(summary=summary, rows=rows), indent=2) + '\n')


if __name__ == '__main__':
    main()
