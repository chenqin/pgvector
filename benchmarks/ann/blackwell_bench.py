#!/usr/bin/env python3
"""Correctness gates and timing for the Blackwell module's grouped tensor-core scan.

    python3 benchmarks/ann/blackwell_bench.py check            # every gate, every width
    python3 benchmarks/ann/blackwell_bench.py time --dims 128,1024 --rows 100000,1000000

Compiles the worker's own kernel source (ann/blackwell.c, the BlackwellSource
strings) with NVRTC for sm_100a through CuPy, as the worker does, quantizes
clustered vectors with the index builder's NVFP4 code, and runs the scan
(score_mma_nvfp4[_32|_64]), select32 and merge32 with the worker's launch
geometry. A kernel change is kept only if every gate passes and it is
faster, the loop AutoKernel (github.com/RightNow-AI/autokernel) runs for
PyTorch kernels:

  1. smoke: one small case, scores within tolerance
  2. sweep: dims x rows x queries, including sizes that are not multiples of 16
  3. recall: the 32 candidates of every query contain its exact top 10
  4. edges: absent rows (UINT32_MAX) score +inf; one row; one query
  5. determinism: three runs give bitwise identical results

Scores are compared with float64 distances to the dequantized rows; the
kernel works in FP16 (queries and dequantized values) with FP32 sums, so the
tolerance is relative to |x|^2 + |q|^2. Offline tooling only (Linux, a
compute capability 10.x GPU, CuPy).
"""

import argparse
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'ann'))
from build_cuvs_index import E2M1, _e4m3_table, quantize_nvfp4  # noqa: E402

ABSENT = 0xFFFFFFFF
WIDTHS = {16: '', 32: '_32', 64: '_64'}


def kernel_source():
    text = open(os.path.join(ROOT, 'ann', 'blackwell.c')).read()
    body = text[text.index('BlackwellSource[] = {'):text.index('NULL};')]
    pieces = re.findall(r'"((?:[^"\\]|\\.)*)"', body)
    return ''.join(bytes(p, 'utf-8').decode('unicode_escape') for p in pieces)


class Scan:
    def __init__(self):
        import cupy as cp

        self.cp = cp
        self.module = cp.RawModule(code=kernel_source(), options=('-arch=sm_100a',), backend='nvrtc')
        self.sms = cp.cuda.Device().attributes['MultiProcessorCount']
        self.functions = {}
        self.fused = {}
        for width, suffix in WIDTHS.items():
            for table, prefix in ((self.functions, 'score_mma_nvfp4'), (self.fused, 'fuse_mma_nvfp4')):
                function = self.module.get_function(prefix + suffix)
                if width > 16:
                    function.max_dynamic_shared_size_bytes = 160 * 1024
                table[width] = function
        self.select = self.module.get_function('select32')
        self.selectc = self.module.get_function('select32c')
        self.merge = self.module.get_function('merge32')

    def index(self, vectors):
        cp = self.cp
        packed, codes, row = quantize_nvfp4(cp.asarray(vectors), cp)
        rows, dims = vectors.shape
        blocks = (dims + 15) // 16
        table = cp.asnumpy(_e4m3_table(np))
        nibbles = np.stack([cp.asnumpy(packed) & 15, cp.asnumpy(packed) >> 4], axis=2).reshape(rows, blocks * 16)
        values = np.asarray(E2M1)[nibbles & 7] * np.where(nibbles & 8, -1.0, 1.0)
        scale = table[cp.asnumpy(codes)].astype(np.float64) * cp.asnumpy(row)[:, None].astype(np.float64)
        dequantized = (values.reshape(rows, blocks, 16) * scale[:, :, None]).reshape(rows, blocks * 16)[:, :dims]
        return {'data': cp.ascontiguousarray(packed), 'scales': cp.ascontiguousarray(codes),
                'rowScales': cp.asarray(row, dtype=cp.float32), 'rowBytes': blocks * 8, 'blocks': blocks,
                'dims': dims, 'dequantized': dequantized}

    def run(self, index, rows, queries, width, fused=False):
        """Scores [nq][n] (unfused) and the merged best 32 (distances, positions) per query."""
        launch, results = (self.prepare_fused if fused else self.prepare)(index, rows, queries, width)
        launch()
        return results

    def prepare_fused(self, index, rows, queries, width, sample=8192, slices=8):
        """Bounds from a sample of the rows, then the fused scan and a selection over candidates."""
        cp = self.cp
        n, nq, dims = len(rows), queries.shape[0], index['dims']
        s = min(sample, n)
        # A spread sample (every (n / s)-th row), as the worker's stride_rows takes.
        picked = rows[(np.arange(s, dtype=np.uint64) * n // s).astype(np.int64)]
        bound_launch, (_, bounds, _) = self.prepare(index, picked, queries, width)
        rows_d = cp.asarray(rows, dtype=cp.uint32)
        queries_d = cp.asarray(queries, dtype=cp.float32)
        cand_d = cp.empty((nq, n), dtype=cp.float32)
        cand_r = cp.empty((nq, n), dtype=cp.uint32)
        counts = cp.zeros(nq, dtype=cp.uint32)
        sgrid = min((n + 127) // 128, self.sms * 4)
        shared = width * ((dims + 15) & ~15) * 2
        part_d = cp.empty(slices * nq * 32, dtype=cp.float32)
        part_r = cp.empty(slices * nq * 32, dtype=cp.uint32)
        out_d = cp.empty(nq * 32, dtype=cp.float32)
        out_r = cp.empty(nq * 32, dtype=cp.uint32)

        def launch():
            bound_launch()
            counts.fill(0)
            self.fused[width]((sgrid,), (256,), (
                index['data'], index['scales'], index['rowScales'], np.uint32(index['rowBytes']),
                np.uint32(index['blocks']), rows_d, np.uint32(n), queries_d, np.int32(nq), np.int32(dims),
                bounds, cand_d, cand_r, counts), shared_mem=shared)
            self.selectc((slices, nq), (256,), (cand_d, cand_r, counts, np.uint32(n), part_d, part_r))
            self.merge((nq,), (32,), (part_d, part_r, np.uint32(slices), out_d, out_r))

        self.last_counts = counts
        return launch, (None, out_d.reshape(nq, 32), out_r.reshape(nq, 32))

    def prepare(self, index, rows, queries, width):
        """Buffers allocated once; returns (launch, (scores, distances, positions))."""
        cp = self.cp
        n, nq, dims = len(rows), queries.shape[0], index['dims']
        rows_d = cp.asarray(rows, dtype=cp.uint32)
        queries_d = cp.asarray(queries, dtype=cp.float32)
        scores = cp.empty((nq, n), dtype=cp.float32)
        max_blocks = self.sms * 4
        sgrid = min((n + 127) // 128, max_blocks)
        cap = max(self.sms // 4, 1)
        slices = max(min((n + 8191) // 8192, cap), 1)
        per_slice = (n + slices - 1) // slices
        shared = width * ((dims + 15) & ~15) * 2
        part_d = cp.empty(slices * nq * 32, dtype=cp.float32)
        part_r = cp.empty(slices * nq * 32, dtype=cp.uint32)
        out_d = cp.empty(nq * 32, dtype=cp.float32)
        out_r = cp.empty(nq * 32, dtype=cp.uint32)
        def launch():
            self.functions[width]((sgrid,), (256,), (
                index['data'], index['scales'], index['rowScales'], np.uint32(index['rowBytes']),
                np.uint32(index['blocks']), rows_d, np.uint32(n), queries_d, np.int32(nq), np.int32(dims), scores),
                shared_mem=shared)
            self.select((slices, nq), (256,), (scores, np.uint32(n), np.uint32(per_slice), part_d, part_r))
            self.merge((nq,), (32,), (part_d, part_r, np.uint32(slices), out_d, out_r))

        return launch, (scores, out_d.reshape(nq, 32), out_r.reshape(nq, 32))


def sample(rng, n, dims):
    shared = np.random.default_rng(12345)
    centroids = (shared.normal(size=(1024, 16)) * 2).astype(np.float32)
    projection = (shared.normal(size=(16, dims)) / 4).astype(np.float32)
    latent = centroids[rng.integers(0, 1024, n)] + rng.normal(size=(n, 16)).astype(np.float32)
    return (latent @ projection + rng.normal(scale=0.05, size=(n, dims))).astype(np.float32)


def check_case(scan, dims, index_rows, n, nq, width, absent=0, seed=0, fused=False):
    rng = np.random.default_rng(seed)
    index = scan.index(sample(rng, index_rows, dims))
    rows = np.sort(rng.choice(index_rows, size=min(n, index_rows), replace=False)).astype(np.uint32)
    rows = np.concatenate([rows, np.full(absent, ABSENT, dtype=np.uint32)])
    queries = sample(np.random.default_rng(seed + 1), nq, dims)
    scores, out_d, out_r = (None if a is None else scan.cp.asnumpy(a)
                            for a in scan.run(index, rows, queries, width, fused))
    present = rows != ABSENT
    x = index['dequantized'][rows[present]]
    q = queries.astype(np.float64)
    exact = (x * x).sum(1)[None, :] + (q * q).sum(1)[:, None] - 2 * q @ x.T
    scale = (x * x).sum(1)[None, :] + (q * q).sum(1)[:, None]
    problems = []
    if fused:
        # The fused scan returns no matrix: check the returned distances instead.
        for j in range(nq):
            ok = out_r[j] != ABSENT
            got = out_r[j][ok]
            if (~present[got]).any():
                problems.append('an absent row was returned')
                break
            where = np.searchsorted(positions_all := np.flatnonzero(present), got)
            e = np.abs(out_d[j][ok] - exact[j][where]) / np.maximum(scale[j][where], 1e-6)
            if e.size and e.max() > 2e-2:
                problems.append(f'returned distance error {e.max():.3g}')
                break
    else:
        error = np.abs(scores[:, present] - exact) / np.maximum(scale, 1e-6)
        if error.max() > 2e-2:
            problems.append(f'score error {error.max():.3g}')
        if absent and not np.isinf(scores[:, ~present]).all():
            problems.append('absent rows did not score +inf')
    positions = np.flatnonzero(present)
    for j in range(nq):
        top = positions[np.argsort(exact[j], kind='stable')[:min(10, len(positions))]]
        if not set(top.tolist()) <= set(out_r[j].tolist()):
            # Ties at the 10th distance may swap; require the distances to be covered.
            bound = np.sort(exact[j])[min(10, len(positions)) - 1]
            missing = [p for p in top if p not in set(out_r[j].tolist())]
            if any(exact[j][np.searchsorted(positions, p)] < bound * (1 - 1e-6) for p in missing):
                problems.append(f'query {j} lost an exact top-10 row')
                break
    return problems


def check(args):
    scan = Scan()
    failures = 0
    stages = [('smoke', [(128, 2000, 1000, 16)]),
              ('sweep', [(d, 60000, n, nq) for d in (64, 128, 200, 1024) for n in (1, 15, 777, 20000, 50000)
                         for nq in (1, 2, 15, 16, 17, 31, 32, 33, 47, 64)]),
              ('edges', [(128, 5000, 300, 64), (1024, 5000, 300, 33)])]
    for stage, cases in stages:
        for dims, index_rows, n, nq in cases:
            for width in WIDTHS:
                if nq > width or (width == 16 and nq > 16):
                    continue
                if width * ((dims + 15) & ~15) * 2 > 160 * 1024:
                    continue
                for fused in (False, True):
                    problems = check_case(scan, dims, index_rows, n, nq, width,
                                          absent=37 if stage == 'edges' else 0, fused=fused)
                    if problems:
                        failures += 1
                        print(f'FAIL {stage} dims={dims} rows={n} queries={nq} width={width} fused={fused}: '
                              f'{"; ".join(problems)}')
        print(f'{stage}: done', flush=True)
    # determinism
    rng = np.random.default_rng(7)
    index = scan.index(sample(rng, 50000, 128))
    rows = np.sort(rng.choice(50000, 20000, replace=False)).astype(np.uint32)
    queries = sample(np.random.default_rng(8), 64, 128)
    runs = [tuple(scan.cp.asnumpy(a).tobytes() for a in scan.run(index, rows, queries, 64)) for _ in range(3)]
    if len(set(runs)) != 1:
        failures += 1
        print('FAIL determinism: runs differ')
    # Fused candidates arrive in any order: the sets and distances must still agree.
    runs = [tuple(np.sort(scan.cp.asnumpy(a), axis=1).tobytes() for a in scan.run(index, rows, queries, 64, True)[1:])
            for _ in range(3)]
    if len(set(runs)) != 1:
        failures += 1
        print('FAIL determinism (fused): runs differ')
    print('determinism: done')
    print('PASS' if failures == 0 else f'{failures} failures')
    return failures == 0


def time_widths(args):
    scan = Scan()
    cp = scan.cp
    print(f'{"dims":>5} {"rows":>9} {"queries":>8} {"width":>6} {"us/pass":>9} {"us/query":>9}')
    for dims in [int(d) for d in args.dims.split(',')]:
        rng = np.random.default_rng(1)
        index_rows = max(int(r) for r in args.rows.split(','))
        index = scan.index(sample(rng, index_rows, dims))
        for n in [int(r) for r in args.rows.split(',')]:
            rows = np.sort(rng.choice(index_rows, size=n, replace=False)).astype(np.uint32)
            for nq in [int(q) for q in args.queries.split(',')]:
                for width in WIDTHS:
                    if nq > width or width * ((dims + 15) & ~15) * 2 > 160 * 1024:
                        continue
                    passes = (nq + width - 1) // width
                    queries = sample(np.random.default_rng(2), nq, dims)
                    prepare = scan.prepare_fused if args.fused else scan.prepare
                    launches = [prepare(index, rows, queries[p * width:(p + 1) * width], width)[0]
                                for p in range(passes)]
                    for launch in launches * 3:
                        launch()
                    start, stop = cp.cuda.Event(), cp.cuda.Event()
                    start.record()
                    for _ in range(args.repeat):
                        for launch in launches:
                            launch()
                    stop.record()
                    stop.synchronize()
                    us = cp.cuda.get_elapsed_time(start, stop) * 1000 / args.repeat
                    print(f'{dims:>5} {n:>9} {nq:>8} {width:>6} {us / passes:>9.1f} {us / nq:>9.2f}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('check')
    t = commands.add_parser('time')
    t.add_argument('--dims', default='128,1024')
    t.add_argument('--rows', default='100000,1000000')
    t.add_argument('--queries', default='16,32,64')
    t.add_argument('--repeat', type=int, default=50)
    t.add_argument('--fused', action='store_true', help='sample bounds, fused scan, select over candidates')
    args = parser.parse_args()
    if args.command == 'check':
        sys.exit(0 if check(args) else 1)
    time_widths(args)


if __name__ == '__main__':
    main()
