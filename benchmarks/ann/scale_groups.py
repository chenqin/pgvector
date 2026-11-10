#!/usr/bin/env python3
"""Scale a segment's ANN group tables by a factor, keeping their distributions.

Each group table (row_id, local_id, ..., m0 int[], embedding vector) gets
FACTOR - 1 synthetic copies of every row next to the original:

- local_id: copy j of document d is j * ID_STRIDE + d in every group, so a
  synthetic document is the same document across groups.
- Attributes and m0 (pg_ids) are the source row's, so every filter keeps the
  share of rows it matches.
- The embedding is sampled around the source row's (a kernel density
  estimate): moved by BANDWIDTH times the distance to its K-th nearest
  neighbour in that group, in a random direction, then renormalized to unit
  length; a trailing cosine-as-L2 coordinate is recomputed from the group's
  augment_norm. Clusters and local density follow the source, FACTOR times
  denser.

New tables go to --target-schema with the source's frozen ann index; terms
and the group catalog are copied there (worker_index renamed), ready for
ann_index_export and the builder.

    python3 benchmarks/ann/scale_groups.py --dsn "$DSN" --factor 10
"""

import argparse
import io
import time

import numpy as np
import psycopg


def knn_distance(x, k, chunk=2048):
    """Distance from each row to its k-th nearest other row."""
    sq = (x * x).sum(1)
    out = np.empty(len(x), dtype=np.float32)
    for i in range(0, len(x), chunk):
        d2 = sq[i:i + chunk, None] + sq[None, :] - 2 * x[i:i + chunk] @ x.T
        np.maximum(d2, 0, out=d2)
        d2[np.arange(len(d2)), np.arange(i, i + len(d2))] = np.inf
        out[i:i + chunk] = np.sqrt(np.partition(d2, k - 1, axis=1)[:, k - 1])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dsn', required=True)
    parser.add_argument('--source-schema', default='seg_ads')
    parser.add_argument('--target-schema', default='seg_ads10')
    parser.add_argument('--group-table', default='matcha_pgvector.ann_group')
    parser.add_argument('--factor', type=int, default=10)
    parser.add_argument('--k', type=int, default=5, help='neighbour whose distance sets the bandwidth')
    parser.add_argument('--bandwidth', type=float, default=0.5)
    parser.add_argument('--id-stride', type=int, default=100000, help='above the largest local_id')
    parser.add_argument('--groups', help='comma-separated tables (default: every group)')
    parser.add_argument('--seed', type=int, default=11)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    src, dst = args.source_schema, args.target_schema
    conn = psycopg.connect(args.dsn, autocommit=True)
    cur = conn.cursor()
    cur.execute(f"SELECT table_name, dimension, augment_norm FROM {args.group_table} "
                f"WHERE worker_index IS NOT NULL ORDER BY table_name")
    groups = cur.fetchall()
    if args.groups:
        wanted = set(args.groups.split(','))
        groups = [g for g in groups if g[0] in wanted]
    cur.execute(f"CREATE SCHEMA IF NOT EXISTS {dst}")
    cur.execute(f"DROP TABLE IF EXISTS {dst}.terms")
    cur.execute(f"CREATE TABLE {dst}.terms AS TABLE {src}.terms")
    cur.execute(f"DROP TABLE IF EXISTS {dst}.ann_group")
    cur.execute(f"CREATE TABLE {dst}.ann_group AS SELECT * FROM {args.group_table}")
    cur.execute(f"UPDATE {dst}.ann_group SET worker_index = '{dst}_' || table_name, rows = rows * {args.factor}")
    for table, dim, augment in groups:
        started = time.time()
        cur.execute(f"SELECT max(local_id) FROM {src}.{table}")
        if cur.fetchone()[0] >= args.id_stride:
            raise SystemExit(f'{src}.{table}: local_id reaches --id-stride {args.id_stride}')
        cur.execute(f"SELECT string_agg(quote_ident(attname), ',' ORDER BY attnum) FROM pg_attribute "
                    f"WHERE attrelid = '{src}.{table}'::regclass AND attnum > 0 AND NOT attisdropped "
                    f"AND attname NOT IN ('row_id', 'local_id', 'embedding')")
        others = cur.fetchone()[0]
        cur.execute(f"SELECT row_id, local_id, {others}, embedding::text FROM {src}.{table} ORDER BY row_id")
        rows = cur.fetchall()
        n = len(rows)
        vectors = np.array([[float(v) for v in r[-1].strip('[]').split(',')] for r in rows], dtype=np.float32)
        base = vectors[:, :dim]
        radius = args.bandwidth * knn_distance(base, args.k)
        cur.execute(f"DROP TABLE IF EXISTS {dst}.{table}")
        cur.execute(f"CREATE TABLE {dst}.{table} (LIKE {src}.{table} INCLUDING DEFAULTS)")
        cur.execute(f"SELECT pg_get_indexdef(indexrelid) FROM pg_index WHERE indrelid = '{src}.{table}'::regclass")
        indexdefs = [r[0] for r in cur.fetchall()]
        with cur.copy(f"COPY {dst}.{table} (row_id, local_id, {others}, embedding) FROM STDIN") as copy:
            for j in range(args.factor):
                if j == 0:
                    synthetic = vectors
                else:
                    step = rng.standard_normal(base.shape).astype(np.float32)
                    step *= (radius / np.linalg.norm(step, axis=1))[:, None]
                    moved = base + step
                    moved /= np.linalg.norm(moved, axis=1)[:, None]
                    synthetic = np.zeros_like(vectors)
                    synthetic[:, :dim] = moved
                    if vectors.shape[1] > dim:
                        synthetic[:, dim] = np.sqrt(max(augment * augment - 1.0, 0.0))
                for i, r in enumerate(rows):
                    copy.write_row([j * n + r[0], j * args.id_stride + r[1], *r[2:-1],
                                    '[' + ','.join(f'{x:.7g}' for x in synthetic[i]) + ']'])
        # A frozen ann index needs a frozen table.
        cur.execute(f"SELECT ann_freeze('{dst}.{table}')")
        cur.execute(f"VACUUM (FREEZE) {dst}.{table}")
        for indexdef in indexdefs:
            cur.execute(indexdef.replace(f' ON {src}.', f' ON {dst}.', 1))
        cur.execute(f"ANALYZE {dst}.{table}")
        print(f'{dst}.{table}: {n * args.factor} rows from {n} (dim {dim}, median bandwidth '
              f'{float(np.median(radius)):.4f}) in {time.time() - started:.1f}s', flush=True)


if __name__ == '__main__':
    main()
