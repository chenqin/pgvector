#!/usr/bin/env python3
"""Compare old FP32 rescoring and native FP32 CAGRA on the same graph.

Uses the frozen stress schema and per-vector references from cost_truth.py.
Measures isolated worker round-trip latency, not saturated SQL throughput.
Temporary workers are loaded through the protocol without catalog changes.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import time

import psycopg

from calibrate_cost import ROOT, decode, range_program, request
from test_index import load_frame
from test_predicate_cache import search_batch_frame, search_predicate_frame, values_program
from frozen_stress import routing_stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dsn', default=os.environ.get('ANN_PG_DSN'),
                        required=not os.environ.get('ANN_PG_DSN'))
    parser.add_argument('--index-dir', required=True)
    parser.add_argument('--baseline-worker', type=Path, required=True)
    parser.add_argument('--truth', type=Path, required=True, help='training references for vectors 1..32')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', default='7')
    parser.add_argument('--repeats', type=int, default=15)
    parser.add_argument('--itopk', type=int, default=4096)
    parser.add_argument('--slack', type=int, default=0)
    args = parser.parse_args()
    if args.repeats < 3 or not 32 <= args.itopk <= 65536 or args.itopk % 32 or not 0 <= args.slack <= 10000:
        parser.error('repeats must be >=3, itopk a multiple of 32 in 32..65536, and slack in 0..10000')
    args.baseline_worker = args.baseline_worker.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'results.json').exists():
        parser.error('use a new output directory')
    truth = {''.join(k.split()): v for k, v in json.loads(args.truth.read_text()).items()}
    with psycopg.connect(args.dsn) as db:
        vectors = [json.loads(r[0]) for r in db.execute(
            'SELECT v FROM frozen_stress_queries WHERE q BETWEEN 1 AND 32 ORDER BY q')]
    if len(vectors) != 32:
        raise RuntimeError('requires query vectors 1..32')
    cases = [('tenant = 7', values_program('tenant', [7]))]
    cases += [(f'category BETWEEN 0 AND {hi}', range_program('category', 0, hi)) for hi in (0, 9, 99)]
    modes = [('cpu_rescore', args.baseline_worker, 'off'),
             ('gpu_rescore', args.baseline_worker, 'on'),
             ('native_fp32', ROOT/'ann/ann-worker-cuvs', 'on')]
    rows, snapshots, commands = [], {}, {}
    for label, executable, rerank in modes:
        path = args.output/f'{label}.sock'
        command = [str(executable), '--backend', 'cuvs', '--listen', str(path),
                   '--replicate', 'off', '--lanes', '1', '--max-batch', '32',
                   '--exact-limit', '0', '--gather-limit', '0', '--bitset-gather-limit', '0',
                   '--group-gather-limit', '0', '--filtered-slack', str(args.slack),
                   '--itopk', str(args.itopk), '--filtered-itopk', str(args.itopk),
                   '--filtered-algo', 'multi', '--filtered-width', '4',
                   '--gpu-rerank', rerank, '--memory-pool', '20',
                   '--predicate-cache-mb', '2048', '--predicate-device-mb', '8192']
        commands[label] = command
        with (args.output/f'{label}.log').open('w') as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, env=dict(os.environ,
                                       CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS='1'))
        try:
            deadline = time.monotonic()+120
            while not path.exists():
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(f'{label} failed to start')
                time.sleep(.05)
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
                conn.settimeout(120)
                conn.connect(str(path))
                request(conn, load_frame(1, 'fp32_benchmark', args.index_dir))
                for predicate, program in cases:
                    for batch in (1, 16, 32):
                        packet = (search_predicate_frame(2, 'fp32_benchmark', vectors[0], 500, program)
                                  if batch == 1 else search_batch_frame(
                                      2, 'fp32_benchmark', vectors[:batch], 500, program))
                        samples = []
                        for rep in range(args.repeats+3):
                            start = time.perf_counter_ns()
                            payload = request(conn, packet)
                            elapsed = (time.perf_counter_ns()-start)/1000
                            result = decode(payload, batch)
                            if rep >= 3:
                                samples.append(elapsed)
                        eligible = result[0][2]
                        recalls = [len(set(hits) & set(truth[''.join(predicate.split())][j][:500])) /
                                   min(500, eligible) for j, (hits, _, _) in enumerate(result)]
                        cardinality = all(len(hits) == min(500, eligible) and
                                          len(set(hits)) == len(hits) for hits, _, _ in result)
                        assert cardinality, (label, predicate, batch)
                        row = dict(mode=label, predicate=predicate, matching=eligible, k=500,
                                   batch=batch, us=statistics.median(samples), samples_us=samples,
                                   timing_scope='worker_roundtrip', recall=statistics.mean(recalls),
                                   min_recall=min(recalls), query_recalls=recalls, cardinality=cardinality)
                        rows.append(row)
                        (args.output/'results.json').write_text(json.dumps(rows, indent=2)+'\n')
                        print(json.dumps(row | {'samples_us': None, 'query_recalls': None}), flush=True)
            snapshots[label] = routing_stats(str(path), 'fp32_benchmark')
            if label == 'native_fp32':
                assert snapshots[label]['cagra_fp32_queries'] > 0
                assert snapshots[label]['graph_cpu_rerank_candidates'] == 0
                assert snapshots[label]['gpu_rerank_candidates'] == 0
        finally:
            process.terminate()
            try:
                process.wait(30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        (args.output/'stats.json').write_text(json.dumps(snapshots, indent=2)+'\n')
    (args.output/'config.json').write_text(json.dumps(dict(
        index_dir=args.index_dir, gpu=args.gpu, itopk=args.itopk, filtered_width=4, slack=args.slack,
        k=500, warmups=3, repeats=args.repeats, commands=commands,
        note='One GPU, one lane, warm isolated latency; same FP32 graph and candidate budget.'), indent=2)+'\n')
    print('fp32 benchmark complete', flush=True)


if __name__ == '__main__':
    main()
