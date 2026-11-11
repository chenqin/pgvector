#!/usr/bin/env python3
"""Scale ANN index directories by a factor without PostgreSQL, keeping their distributions.

Reads each group's seed index directory (keys.i64, vectors.f32, attr-*.i32 and
multi-valued attr-*.offsets, as ann/common/INDEX_FORMAT.md describes) and writes
an export directory FACTOR times larger, which build_cuvs_index.py --from-export
turns into a worker index:

- keys: copy j of key d is j * KEY_STRIDE + d, the same in every group, so a
  synthetic document is the same document across groups; keys stay sorted.
- attributes and multi-valued columns (pg_ids) are the seed row's, so every
  filter keeps the share of rows it matches.
- vectors are sampled around the seed row's (a kernel density estimate),
  renormalized to unit length; a trailing cosine-as-L2 coordinate is
  recomputed from the group's augment_norm. --kernel picks the sample:
  shell (the default) moves a copy by BANDWIDTH times the distance to its
  K-th nearest seed neighbour in a random direction; tangent moves it by a
  random Gaussian mix of the offsets to its K nearest seed neighbours, scaled
  by BANDWIDTH / sqrt(K), so copies stay in the directions the seed's
  neighbours lie in.

The shell kernel does not keep the seed's local structure at a large factor,
which graph searches depend on (scans do not). On a 33-dim group whose seed
has local intrinsic dimension 10 and a 32-neighbour overlap of 0.35 between
neighbours, shell at bandwidth 4 gave dimension 23 and overlap 0.08 (every
row about equally far from every other), and at 0.5 each seed's copies are
nearer to each other than to anything else. --kernel tangent --k 16
--bandwidth 1 kept both (9.5 and 0.36). Use it for anything that walks a
graph.

    python3 benchmarks/ann/scale_export.py --seed-root SEED/build/seg_ads \\
        --groups groups.json --factor 100 --out /data/scale100/export

--only g3,g25 exports those groups alone (scale_pipeline.sh exports, builds and
deletes one group at a time). A group's output directory must not exist yet.
Output is written seed copy by seed copy through large buffers, so memory stays
at the seed's size whatever the factor; at factor 4715 (46.5M rows) a 65-dim
group is 16.8 GB and takes about 45 s on one core.
"""

import argparse
import concurrent.futures
import json
import os
import time

import numpy as np

BUFFER = 16 << 20   # bytes of output buffered per file: seed-sized pieces become few large writes


def knn(x, k, chunk=2048):
    """Each row's k nearest other rows, nearest first, and the distance to the k-th."""
    sq = (x * x).sum(1)
    out = np.empty(len(x), dtype=np.float32)
    nearest = np.empty((len(x), k), dtype=np.int64)
    for i in range(0, len(x), chunk):
        d2 = sq[i:i + chunk, None] + sq[None, :] - 2 * x[i:i + chunk] @ x.T
        np.maximum(d2, 0, out=d2)
        d2[np.arange(len(d2)), np.arange(i, i + len(d2))] = np.inf
        part = np.argpartition(d2, k - 1, axis=1)[:, :k]
        order = np.argsort(np.take_along_axis(d2, part, 1), axis=1)
        nearest[i:i + chunk] = np.take_along_axis(part, order, 1)
        out[i:i + chunk] = np.sqrt(np.take_along_axis(d2, nearest[i:i + chunk, -1:], 1)[:, 0])
    return nearest, out


def manifest(path):
    lines = [line.rstrip('\n') for line in open(path) if line.strip()]
    return lines, dict(line.split(' ', 1) for line in lines if not line.startswith('#'))


def scale(args, group):
    started = time.time()
    table, dim, augment = group['table'], group['dimension'], group['augment']
    src = os.path.join(args.seed_root, table, 'index')
    dst = os.path.join(args.out, table)
    lines, fields = manifest(os.path.join(src, 'manifest'))
    rows, dims = int(fields['rows']), int(fields['dimensions'])
    keys = np.fromfile(os.path.join(src, 'keys.i64'), dtype='<i8')
    if keys.max() >= args.key_stride:
        raise SystemExit(f'{table}: keys reach --key-stride {args.key_stride}')
    vectors = np.fromfile(os.path.join(src, 'vectors.f32'), dtype='<f4').reshape(rows, dims)
    rng = np.random.default_rng(args.seed + int(table.lstrip('g') or 0))
    base = vectors[:, :dim]
    nearest, kth = knn(base, args.k)
    if args.kernel == 'tangent':
        offsets = base[nearest] - base[:, None, :]      # [rows, k, dim]
        weight = np.float32(args.bandwidth / np.sqrt(args.k))
    else:
        radius = args.bandwidth * kth
    try:
        os.makedirs(dst, exist_ok=False)
    except FileExistsError:
        raise SystemExit(f'{table}: output directory {dst} already exists; remove it or choose another --out')
    with open(os.path.join(dst, 'keys.i64'), 'wb', BUFFER) as out:
        for j in range(args.factor):
            out.write((keys + j * args.key_stride).astype('<i8').data)
    with open(os.path.join(dst, 'vectors.f32'), 'wb', BUFFER) as out:
        synthetic = np.zeros(vectors.shape, dtype='<f4')
        if dims > dim:
            synthetic[:, dim] = np.sqrt(max(augment * augment - 1.0, 0.0))
        for j in range(args.factor):
            if j == 0:
                out.write(vectors.data)
                continue
            if args.kernel == 'tangent':
                mix = rng.standard_normal((rows, args.k), dtype=np.float32) * weight
                step = np.einsum('rk,rkd->rd', mix, offsets)
            else:
                step = rng.standard_normal(base.shape).astype(np.float32)
                step *= (radius / np.linalg.norm(step, axis=1))[:, None]
            moved = base + step
            moved /= np.linalg.norm(moved, axis=1)[:, None]
            synthetic[:, :dim] = moved
            out.write(synthetic.data)
    multi = {line.split()[1]: line.split()[2] for line in lines if line.startswith('multivalued ')}
    attributes = [line.split()[1:3] for line in lines if line.startswith('attribute ')]
    for name, file in attributes:
        column = np.fromfile(os.path.join(src, file), dtype='<i4')
        if name in multi:
            # Row starts are u32 on disk: add each copy's base in int64, and refuse a total the last
            # offset (the element count) cannot hold.
            offsets = np.fromfile(os.path.join(src, multi[name]), dtype='<u4').astype(np.int64)
            if len(column) * args.factor >= 2 ** 32:
                raise SystemExit(f'{table}.{name}: more than 2^32 elements')
            with open(os.path.join(dst, multi[name]), 'wb', BUFFER) as out:
                for j in range(args.factor):
                    out.write((offsets[:-1] + j * len(column)).astype('<u4').data)
                out.write(np.array([len(column) * args.factor], dtype='<u4').data)
        with open(os.path.join(dst, file), 'wb', BUFFER) as out:
            for _ in range(args.factor):
                out.write(column.data)
    out_lines = []
    for line in lines:
        key = line.split(' ', 1)[0]
        if key == 'rows':
            line = f'rows {rows * args.factor}'
        elif key == 'source':
            line = f'source {args.source_schema}.{table}'
        elif key in ('shards', 'shard', 'quant') or line.startswith('#'):
            continue
        elif key == 'generation':
            line = f'generation {(int(fields["generation"], 16) ^ args.factor) & (2 ** 64 - 1):016x}'
        elif key == 'dtype':
            line = 'dtype float32'
        out_lines.append(line)
    out_lines.insert(out_lines.index(next(l for l in out_lines if l.startswith('vector '))) + 1, 'shards 0')
    with open(os.path.join(dst, 'manifest'), 'w') as out:
        out.write('\n'.join(out_lines) + '\n')
    return f'{table}: {rows * args.factor} rows from {rows} (dim {dim}) in {time.time() - started:.1f}s'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seed-root', required=True, help='directory of <table>/index seed directories')
    parser.add_argument('--groups', required=True, help='JSON list of {stem, table, dimension, augment}')
    parser.add_argument('--factor', type=int, required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--source-schema', default='seg_ads_scaled')
    parser.add_argument('--kernel', choices=('shell', 'tangent'), default='shell',
                        help='how a copy is moved from its seed row (see above)')
    parser.add_argument('--k', type=int, default=5, help='seed neighbours the kernel uses')
    parser.add_argument('--bandwidth', type=float, default=0.5)
    parser.add_argument('--key-stride', type=int, default=100000)
    parser.add_argument('--jobs', type=int, default=16)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--only', metavar='TABLE[,TABLE...]',
                        help='export only these group tables (default: every group of --groups)')
    args = parser.parse_args()
    groups = json.load(open(args.groups))
    if args.only is not None:
        only = [table for table in args.only.split(',') if table]
        unknown = sorted(set(only) - {group['table'] for group in groups})
        if unknown or not only:
            raise SystemExit(f'--only: no group table named {", ".join(unknown) or "(empty)"} in {args.groups}')
        groups = [group for group in groups if group['table'] in only]
    existing = [group['table'] for group in groups if os.path.lexists(os.path.join(args.out, group['table']))]
    if existing:
        raise SystemExit(f'output directory exists for {", ".join(existing)} under {args.out}; '
                         'remove them or choose another --out (nothing was written)')
    os.makedirs(args.out, exist_ok=True)
    with concurrent.futures.ProcessPoolExecutor(min(args.jobs, len(groups))) as pool:
        for line in pool.map(scale, [args] * len(groups), groups):
            print(line, flush=True)


if __name__ == '__main__':
    main()
