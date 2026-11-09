#!/usr/bin/env python3
"""Calibrate frozen search on a loaded index using isolated worker processes.

Measures real requests, checks cardinality/recall, and retains raw samples.
Never changes the database's worker setting or registers temporary indexes.
"""
import argparse
import json
import math
import os
from pathlib import Path
import socket
import statistics
import struct
import subprocess
import sys
import time

import psycopg
from psycopg import sql

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'test/ann'))
from test_server import frame, receive  # noqa: E402
from test_index import load_frame  # noqa: E402
from test_predicate_cache import values_program, search_batch_frame, search_predicate_frame  # noqa: E402


def range_program(column, lo, hi):
    return bytes([2, len(column)]) + column.encode() + struct.pack('<Bii', 0, lo, hi)


def predicates():
    tenant = values_program('tenant', [7])
    return [(f'tenant = 7 AND category BETWEEN 0 AND {hi}', tenant + range_program('category', 0, hi) + b'\x05')
            for hi in [0, 4, 9]] + [('tenant = 7', tenant)] + [
                (f'category BETWEEN 0 AND {hi}', range_program('category', 0, hi))
                for hi in [0, 4, 9, 49, 99]]


def request(connection, packet):
    connection.sendall(packet)
    _, _, status, payload = receive(connection)
    if status:
        raise RuntimeError(f'worker status {status}')
    return payload


def decode(payload, batch):
    offset = 4 if batch > 1 else 0
    if batch > 1 and struct.unpack_from('<I', payload)[0] != batch:
        raise RuntimeError('wrong batch count')
    out = []
    for _ in range(batch):
        count, route, eligible = struct.unpack_from('<IIQ', payload, offset)
        offset += 16
        keys = [struct.unpack_from('<qf', payload, offset + i * 12)[0] for i in range(count)]
        offset += count * 12
        out.append((keys, route, eligible))
    if offset != len(payload):
        raise RuntimeError('invalid result size')
    return out


def finalize(args, rows, local, total, dims, generation, cases):
    for row in rows:
        row['qualified'] = row['cardinality'] and row['recall'] >= args.min_recall
    cells = []
    for ci in range(len(cases)):
        for k in [50, 500, 1000]:
            for batch in [1, 16, 32]:
                selected = [r for r in rows if r['case']==ci and r['k']==k and r['batch']==batch]
                matching = selected[0]['matching']
                values = {r['mode']:r['us'] if r['qualified'] else -1 for r in selected}
                cpu = next(r['us'] for r in local if r['case']==ci and r['k']==k)
                cells.append(dict(matching=matching,k=k,batch=batch,local_us=cpu,**values))
    # Correlated attributes can produce empty or duplicate cardinalities.
    # Keep every raw sample, but the interpolation grid needs unique positive cells.
    grouped = {}
    for c in cells:
        if c['matching']:
            grouped.setdefault((c['matching'], c['k'], c['batch']), []).append(c)
    cells = []
    for (matching, k, batch), group in grouped.items():
        cell = dict(matching=matching, k=k, batch=batch)
        for mode in ['scan_host','scan_gpu','graph_host','graph_gpu','local_us']:
            values = [c[mode] for c in group]
            cell[mode] = statistics.median(values) if all(v > 0 for v in values) else -1
        cells.append(cell)
    setup_us = tuple_us = 0.0
    overhead = []
    if args.service_socket:
        import numpy as np
        for reference in local:
            if reference['case'] >= len(cases):
                continue
            mode = 'scan_gpu' if next(r['matching'] for r in rows if r['case']==reference['case']) <= 500000 else 'graph_gpu'
            timing = next(r for r in rows if r['case']==reference['case'] and r['k']==reference['k'] and r['batch']==1 and r['mode']==mode)
            if timing['matching'] and mode == 'scan_gpu':
                overhead.append(dict(k=reference['k'], us=max(0,reference['sql_us']-timing['us'])))
        ks = [50,500,1000]
        matrix = np.array([[1,k] for k in ks],dtype=float)
        target = np.array([statistics.median(r['us'] for r in overhead if r['k']==k) for k in ks])
        unconstrained = np.linalg.lstsq(matrix,target,rcond=None)[0]
        candidates = [np.array([target.mean(),0]), np.array([0,np.dot(matrix[:,1],target)/np.dot(matrix[:,1],matrix[:,1])])]
        if np.all(unconstrained >= 0):
            candidates.append(unconstrained)
        coefficients = min(candidates,key=lambda c:np.sum((matrix@c-target)**2))
        setup_us,tuple_us = map(float,coefficients)
    profile = dict(version=5,graph_slack=args.graph_slack,graph_itopk=args.graph_itopk,rows=total,dimensions=dims,generation=generation,min_recall=args.min_recall,
                   setup_us=setup_us,tuple_us=tuple_us,overhead_samples=overhead,
                   note='Warm, one-GPU one-lane isolated service latency in microseconds per request; local controls one SQL request.',cells=cells)
    (args.output/'profile.json').write_text(json.dumps(profile,indent=2)+'\n')
    (args.output/'profile.tsv').write_text(f'ANN_COST_PROFILE 5 {total} {dims} {generation} {len(cells)} {setup_us:.6f} {tuple_us:.6f} {args.min_recall:.6f} {args.graph_itopk} {args.graph_slack}\n'+
        ''.join(f"{c['matching']} {c['k']} {c['batch']} {c['scan_host']:.6f} {c['scan_gpu']:.6f} {c['graph_host']:.6f} {c['graph_gpu']:.6f} {c['local_us']:.6f}\n" for c in cells))
    (args.output/'complete.json').write_text(json.dumps(dict(completed=time.time(),points=len(rows)))+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dsn', default=os.environ.get('ANN_PG_DSN'), required=not os.environ.get('ANN_PG_DSN'))
    p.add_argument('--index-dir', required=True)
    p.add_argument('--table', default='public.frozen_stress')
    p.add_argument('--queries', default='public.frozen_stress_queries')
    p.add_argument('--frozen-index', default='public.frozen_stress_ann')
    p.add_argument('--gpu', default='7')
    p.add_argument('--service-socket', help='optional deployed worker for SQL overhead measurements')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=7)
    p.add_argument('--continue-run', action='store_true', help='collect missing points using existing samples and exact references')
    p.add_argument('--resume', action='store_true', help='finalize already collected samples without rerunning workers')
    p.add_argument('--min-recall', type=float, default=.95)
    p.add_argument('--graph-itopk', type=int, default=4096)
    p.add_argument('--graph-slack', type=int, default=256)
    p.add_argument('--truth-training',type=Path)
    p.add_argument('--truth-holdout',type=Path)
    args = p.parse_args()
    if args.repeats < 3 or not 0 < args.min_recall <= 1 or args.graph_itopk < 32 or args.graph_itopk > 65536 or args.graph_itopk % 32 or args.graph_slack < 0 or args.graph_slack > 10000:
        p.error('repeats must be >=3 and min-recall in (0,1]')
    if args.resume and args.continue_run:
        p.error('--resume and --continue-run are mutually exclusive')
    if not args.resume and (not args.truth_training or not args.truth_holdout):
        p.error('per-vector --truth-training and --truth-holdout are required')
    args.output.mkdir(parents=True, exist_ok=True)
    config = dict(graph_itopk=args.graph_itopk, graph_slack=args.graph_slack, min_recall=args.min_recall, gpu=args.gpu, repeats=args.repeats, index_dir=args.index_dir)
    config_path = args.output / 'run-config.json'
    if args.resume or args.continue_run:
        if not config_path.exists() or json.loads(config_path.read_text()) != config:
            p.error('saved run configuration differs; use the original settings or a new output directory')
    elif (args.output/'samples.json').exists():
        p.error('samples already exist; use --resume, --continue-run or a new output directory')
    (args.output / 'controller.pid').write_text(str(os.getpid()))
    config_path.write_text(json.dumps(config,indent=2)+'\n')
    table = sql.Identifier(*args.table.split('.'))
    queries_table = sql.Identifier(*args.queries.split('.'))
    rows = json.loads((args.output/'samples.json').read_text()) if args.continue_run else []
    local = json.loads((args.output/'local.json').read_text()) if args.continue_run else []
    cases = predicates()
    truth_training={''.join(k.split()):v for k,v in json.loads(args.truth_training.read_text()).items()} if args.truth_training else None
    truth_holdout={''.join(k.split()):v for k,v in json.loads(args.truth_holdout.read_text()).items()} if args.truth_holdout else None
    if args.resume:
        rows = json.loads((args.output/'samples.json').read_text())
        local = json.loads((args.output/'local.json').read_text())
        with psycopg.connect(args.dsn) as db:
            generation,total,dims = db.execute('SELECT generation,rows,dimensions FROM ann_frozen_info(%s::regclass)',(args.frozen_index,)).fetchone()
        finalize(args,rows,local,total,dims,generation,cases)
        return

    holdouts = [('tenant = 7 AND category BETWEEN 0 AND 1', values_program('tenant', [7]) + range_program('category', 0, 1) + b'\x05'),
                ('category BETWEEN 0 AND 1', range_program('category', 0, 1)),
                ('category BETWEEN 0 AND 19', range_program('category', 0, 19))]
    with psycopg.connect(args.dsn, autocommit=True, prepare_threshold=None) as db:
        db.execute("LOAD 'vector'")
        # A deployed database may already enable the profile being replaced.
        db.execute("SET ann.frozen_cost_profile = ''")
        db.execute('SET ann.enable_frozen_search = off')
        db.execute("SET ann.worker_socket = ''")
        db.execute('SET max_parallel_workers_per_gather = 0')
        db.execute('SET jit = off')
        db.execute("SET statement_timeout = '120s'")
        vectors = [json.loads(r[0]) for r in db.execute(sql.SQL('SELECT v FROM {} ORDER BY q LIMIT 80').format(queries_table))]
        generation, total, dims = db.execute('SELECT generation,rows,dimensions FROM ann_frozen_info(%s::regclass)', (args.frozen_index,)).fetchone()
        # A vector operator is deliberately evaluated by PostgreSQL for local controls.
        oracles = {}
        if args.continue_run:
            if not truth_training or not truth_holdout:
                raise RuntimeError('continuation needs exact training and holdout references')
            for ci,(predicate,_) in enumerate(cases+holdouts):
                reference=truth_training if ci<len(cases) else truth_holdout
                for k in ([50,500,1000] if ci<len(cases) else [250,750]):
                    oracles[ci,k]=reference[''.join(predicate.split())][0][:k]
        for ci, (predicate, _) in enumerate([] if args.continue_run else cases + holdouts):
            validation_vector = vectors[0 if ci < len(cases) else 64]
            statement = sql.SQL('SELECT id FROM {} WHERE {} ORDER BY emb <-> %s::vector LIMIT %s').format(table, sql.SQL(predicate))
            db.execute(statement, (json.dumps(validation_vector), 1000)).fetchall()
            for k in ([50, 500, 1000] if ci < len(cases) else [250, 750]):
                samples = []
                for _ in range(3):
                    start = time.perf_counter_ns()
                    keys = [x[0] for x in db.execute(statement, (json.dumps(validation_vector), k))]
                    samples.append((time.perf_counter_ns() - start) / 1000)
                oracles[ci, k] = keys
                plan = db.execute(sql.SQL('EXPLAIN (FORMAT JSON) {}').format(statement), (json.dumps(validation_vector), k)).fetchone()[0][0]['Plan']
                entry = dict(case=ci, predicate=predicate, k=k, us=statistics.median(samples), samples_us=samples,
                             estimated_rows=plan['Plan Rows'], cost=plan['Total Cost'], returned=len(keys))
                if args.service_socket:
                    db.execute("SELECT set_config('ann.worker_socket',%s,false)", (args.service_socket,))
                    db.execute('SET ann.enable_frozen_search = on')
                    db.execute('SET ann.frozen_setup_cost = 0')
                    routed = []
                    for _ in range(4):
                        start = time.perf_counter_ns()
                        db.execute(statement, (json.dumps(validation_vector), k)).fetchall()
                        routed.append((time.perf_counter_ns() - start) / 1000)
                    entry['sql_us'] = statistics.median(routed[1:])
                    entry['sql_samples_us'] = routed
                    db.execute('SET ann.enable_frozen_search = off')
                local.append(entry)
        (args.output / 'local.json').write_text(json.dumps(local, indent=2)+'\n')
    worker = str(ROOT / 'ann/server/ann-worker-cuvs')
    for mode in ['scan_host', 'scan_gpu', 'graph_host', 'graph_gpu']:
        scan = mode.startswith('scan')
        gpu = mode.endswith('gpu')
        path = args.output / f'{mode}.sock'
        if path.exists():
            if not path.is_socket():raise RuntimeError('refusing to replace a non-socket file')
            path.unlink()
        command = [worker, '--backend', 'cuvs', '--listen', str(path), '--lanes', '1', '--max-batch', '32',
                   '--replicate', 'off', '--exact-limit', '0', '--itopk', str(512 if scan else args.graph_itopk), '--filtered-itopk', str(1024 if scan else args.graph_itopk),
                   '--filtered-algo', 'multi', '--filtered-width', '4', '--filtered-slack', str(0 if scan else args.graph_slack), '--matrix-gather', 'on',
                   '--gpu-rerank', 'on' if gpu else 'off', '--gather-limit', str(total if scan else 0),
                   '--bitset-gather-limit', str(total if scan else 0), '--group-gather-limit', str(total if scan else 0),
                   '--predicate-cache-mb', '2048', '--predicate-device-mb', '8192', '--memory-pool', '20']
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        with (args.output / f'{mode}.log').open('w') as log:
            process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 120
            while not path.exists():
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(f'{mode} failed to start')
                time.sleep(.05)
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(120)
                connection.connect(str(path))
                request(connection, load_frame(1, 'cost_calibration', args.index_dir))
                for ci, (predicate, program) in enumerate(cases + holdouts):
                    training = ci < len(cases)
                    for k in ([50, 500, 1000] if training else [250, 750]):
                        for batch in ([1, 16, 32] if training else [4, 8]):
                            if any(r['mode']==mode and r['case']==ci and r['k']==k and r['batch']==batch for r in rows):
                                continue
                            q = vectors[:batch] if training else vectors[64:64+batch]
                            packet = (search_predicate_frame(2, 'cost_calibration', q[0], k, program) if batch == 1 else
                                      search_batch_frame(2, 'cost_calibration', q, k, program))
                            samples = []
                            decoded = None
                            for rep in range(args.repeats + 2):
                                start = time.perf_counter_ns()
                                payload = request(connection, packet)
                                elapsed = (time.perf_counter_ns() - start) / 1000
                                decoded = decode(payload, batch)
                                if rep >= 2:
                                    samples.append(elapsed)
                            eligible = decoded[0][2]
                            cardinality = all(len(keys) == min(k, eligible) and len(set(keys)) == len(keys) for keys, _, _ in decoded)
                            expected = oracles[ci, k]
                            reference=truth_training if training else truth_holdout
                            query_recalls=[len(set(keys)&set(reference[''.join(predicate.split())][j][:k]))/max(min(k,eligible),1) for j,(keys,_,_) in enumerate(decoded)] if reference else [len(set(decoded[0][0]) & set(expected)) / max(len(expected),1)]
                            recall=statistics.mean(query_recalls)
                            row = dict(mode=mode, case=ci, predicate=predicate, matching=eligible, k=k, batch=batch,
                                       us=statistics.median(samples), samples_us=samples, route=decoded[0][1],
                                       recall=recall, min_recall=min(query_recalls), query_recalls=query_recalls, cardinality=cardinality, qualified=cardinality and recall >= args.min_recall,
                                       training=training)
                            rows.append(row)
                            (args.output / 'samples.json').write_text(json.dumps(rows, indent=2)+'\n')
                            print(json.dumps({k:row[k] for k in ['mode','matching','k','batch','us','recall','qualified','training']}), flush=True)
        finally:
            process.terminate()
            try:
                process.wait(30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    finalize(args, rows, local, total, dims, generation, cases)


if __name__ == '__main__':
    main()
