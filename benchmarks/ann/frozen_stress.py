#!/usr/bin/env python3
"""Measure frozen SQL throughput, latency, process CPU, waits and GPU use.

Needs a loaded frozen index, a query table (q integer, v text), psycopg,
NumPy and pgbench. The default workload returns all 500 keys to the client.
Read-only monitoring uses pg_stat_activity and /proc; no perf permission is
required. Run against a dedicated test database, not a production workload.
"""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import socket
import struct
import subprocess
import tempfile
import threading
import time

import numpy as np
import psycopg
from psycopg import sql


WORKLOADS = {
    '0.1pct': 'tenant = :t',
    '1pct': 'category = :c',
    '10pct': 'category BETWEEN :d * 10 AND :d * 10 + 9',
}


def identifier(name):
    return sql.Identifier(*name.split('.'))


def process(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        ticks = int(fields[11]) + int(fields[12])
        channel = Path(f'/proc/{pid}/wchan').read_text().strip()
        io = dict(line.split(':', 1) for line in Path(f'/proc/{pid}/io').read_text().splitlines())
        return ticks, fields[0], channel, int(io['read_bytes']), int(io['write_bytes'])
    except (OSError, ValueError, IndexError):
        return None


class Monitor:
    def __init__(self, dsn, application, worker_pid):
        self.dsn, self.application, self.worker_pid = dsn, application, worker_pid
        self.stop = threading.Event()
        self.waits, self.channels, self.states = Counter(), Counter(), Counter()
        self.first, self.last = {}, {}
        self.gpus = []
        self.errors = []

    def record(self, role, pid):
        value = process(pid)
        if value:
            key = (role, pid)
            self.first.setdefault(key, value)
            self.last[key] = value
            self.channels[f'{role}:{value[2]}'] += 1
            self.states[f'{role}:{value[1]}'] += 1

    def sample(self):
        try:
            with psycopg.connect(self.dsn, autocommit=True, application_name='frozen_observer',
                                 connect_timeout=5, options='-c statement_timeout=5000') as db:
                while not self.stop.is_set():
                    rows = db.execute('''SELECT pid,state,wait_event_type,wait_event
                                         FROM pg_stat_activity WHERE application_name=%s''',
                                      (self.application,)).fetchall()
                    for pid, state, kind, event in rows:
                        self.waits[f'{state}:{kind or "none"}:{event or "none"}'] += 1
                        self.record('postgres', pid)
                    if self.worker_pid:
                        self.record('worker', self.worker_pid)
                    if getattr(self, 'client_pid', None):
                        self.record('client', self.client_pid)
                    for pid in getattr(self, 'client_pids', []):
                        self.record('client', pid)
                    self.stop.wait(0.2)
        except Exception as error:
            self.errors.append(str(error))

    def sample_gpu(self):
        while not self.stop.is_set():
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,power.draw',
                                         '--format=csv,noheader,nounits'], text=True,
                                        capture_output=True, timeout=5)
                if result.returncode == 0:
                    for line in result.stdout.splitlines():
                        index, utilization, memory, power = [float(v.strip()) for v in line.split(',')]
                        self.gpus.append(dict(gpu=int(index), utilization=utilization, memory_mb=memory, watts=power))
            except (OSError, ValueError, subprocess.TimeoutExpired) as error:
                self.errors.append(str(error))
            self.stop.wait(1)

    def start(self):
        self.threads = [threading.Thread(target=self.sample), threading.Thread(target=self.sample_gpu)]
        for thread in self.threads:
            thread.start()

    def finish(self, seconds):
        self.stop.set()
        for thread in self.threads:
            thread.join(10)
        cpu, reads, writes = Counter(), Counter(), Counter()
        clock = os.sysconf('SC_CLK_TCK')
        for key, last in self.last.items():
            first = self.first[key]
            cpu[key[0]] += max(0, last[0] - first[0]) / clock
            reads[key[0]] += max(0, last[3] - first[3])
            writes[key[0]] += max(0, last[4] - first[4])
        gpus = {}
        for index in sorted({row['gpu'] for row in self.gpus}):
            rows = [row for row in self.gpus if row['gpu'] == index]
            gpus[index] = dict(utilization_mean=float(np.mean([r['utilization'] for r in rows])),
                               utilization_max=max(r['utilization'] for r in rows),
                               memory_mb_max=max(r['memory_mb'] for r in rows))
        return dict(cpu_seconds=dict(cpu), average_cpu_cores={k:v / seconds for k,v in cpu.items()},
                    disk_read_bytes=dict(reads), disk_write_bytes=dict(writes),
                    postgres_wait_samples=dict(self.waits), os_wait_channels=dict(self.channels),
                    os_states=dict(self.states), gpu_samples=gpus, monitor_errors=self.errors)


def routing_stats(path, index):
    """A control request outside the timed workload; old workers may lack it."""
    if not index:
        return None
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(2)
            connection.connect(path)
            payload = index.encode()
            connection.sendall(b'PVAN' + struct.pack('<HHQQII', 1, 31, 1, 0, len(payload), 0) + payload)

            def receive(size):
                data = b''
                while len(data) < size:
                    chunk = connection.recv(size - len(data))
                    if not chunk:
                        raise EOFError('routing stats connection closed')
                    data += chunk
                return data

            header = receive(32)
            version, operation, request, generation, size, status = struct.unpack('<HHQQII', header[4:])
            if header[:4] != b'PVAN' or version != 1 or operation != 31 or request != 1 or size > 65536:
                raise ValueError('malformed routing stats header')
            data = receive(size)
            return json.loads(data) if status == 0 else {'status': status}
    except (OSError, EOFError, ValueError) as error:
        return {'error': str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dsn', default=os.environ.get('ANN_PG_DSN'), required=not os.environ.get('ANN_PG_DSN'))
    parser.add_argument('--table', default='public.frozen_stress')
    parser.add_argument('--queries', default='public.frozen_stress_queries')
    parser.add_argument('--socket', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--log-dir', default='/dev/shm')
    parser.add_argument('--k', type=int, default=500)
    parser.add_argument('--batch-queries', type=int, default=0, metavar='N',
                        help='use the SQL matrix API with 1..64 vectors per request; 0 uses the scan route')
    parser.add_argument('--index-name', help='loaded frozen index name, required with --batch-queries')
    parser.add_argument('--clients', default='1,8,32,64,128')
    parser.add_argument('--threads', type=int, default=32)
    parser.add_argument('--seconds', type=int, default=15)
    parser.add_argument('--workloads', default='0.1pct,1pct,10pct')
    parser.add_argument('--mmap', choices=['on', 'off'], default='on')
    parser.add_argument('--local-cache-mb', type=int, default=16)
    parser.add_argument('--worker-pid', type=int)
    parser.add_argument('--label', default='baseline')
    parser.add_argument('--plan-cache-mode', choices=['auto', 'force_generic_plan', 'force_custom_plan'],
                        default='force_generic_plan')
    parser.add_argument('--setup-cost', type=float, default=1000.0)
    parser.add_argument('--prefetch-distance', type=int, choices=range(65))
    parser.add_argument('--materialize-batch', type=int, default=0, metavar='ROWS',
                        help='mapped TID visibility window for covering scans, 0 disables, maximum 1024')
    parser.add_argument('--profile-library', help='optional session preload library for isolated debugger runs')
    parser.add_argument('--cost-profile', help='optional calibrated planner profile on the PostgreSQL host')
    parser.add_argument('--count-only', action='store_true', help='diagnostic: aggregate the 500 rows on the server')
    args = parser.parse_args()
    if not 0 <= args.materialize_batch <= 1024:
        parser.error('materialization batch must be between 0 and 1024')
    if not 1 <= args.k <= 10000 or args.seconds < 1 or not np.isfinite(args.setup_cost) or args.setup_cost < 0:
        parser.error('k must be 1..10000, seconds must be positive and setup cost must be finite and nonnegative')
    if not 0 <= args.batch_queries <= 64 or (args.batch_queries and not args.index_name):
        parser.error('batch queries must be 0..64 and require --index-name when enabled')
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    with psycopg.connect(args.dsn) as db:
        table = identifier(args.table).as_string(db)
        queries = identifier(args.queries).as_string(db)
        index_name = sql.Literal(args.index_name).as_string(db) if args.index_name else None
    for workload in args.workloads.split(','):
        if workload not in WORKLOADS:
            parser.error(f'unknown workload: {workload}')
        for clients in map(int, args.clients.split(',')):
            if clients < 1:
                parser.error('clients must be positive')
            tag = f'{args.label}-{workload}-{clients}'
            application = 'frozen_stress_' + tag
            with tempfile.TemporaryDirectory(prefix='frozen-pgbench-', dir=args.log_dir) as directory:
                script = Path(directory) / 'query.sql'
                query = f'SELECT id FROM {table} WHERE {WORKLOADS[workload]} ORDER BY emb <-> :v::vector LIMIT {args.k}'
                setup = (f'SELECT v FROM {queries} WHERE q=:q \\gset\n')
                if args.batch_queries:
                    predicate = {
                        '0.1pct': "format('tenant = %s', :t::integer)",
                        '1pct': "format('category = %s', :c::integer)",
                        '10pct': "format('category BETWEEN %s AND %s', :d * 10, :d * 10 + 9)",
                    }[workload]
                    query = (f'SELECT id FROM ann_frozen_search_batch({index_name}, :batch::vector[], '
                             f'{args.k}, {predicate})')
                    setup = (f'SELECT array_agg(v ORDER BY q)::text AS batch FROM {queries} '
                             f'WHERE q >= :q AND q < :q + {args.batch_queries} \\gset\n')
                if args.count_only:
                    query = f'SELECT count(*) FROM ({query}) hits'
                script.write_text(f'\\set q random(1,{1001 - max(args.batch_queries, 1)})\n'
                                  '\\set t random(0,99)\n\\set c random(0,99)\n\\set d random(0,9)\n'
                                  + setup + query + ';\n')
                env = dict(os.environ, PGAPPNAME=application,
                           PGOPTIONS=(f'-c ann.worker_socket={args.socket} -c ann.enable_frozen_search=on '
                                      f'-c ann.frozen_mmap={args.mmap} -c ann.frozen_filter_local_cache_mb={args.local_cache_mb} '
                                      f'-c ann.frozen_setup_cost={args.setup_cost} -c jit=off '
                                      f'-c statement_timeout=30000 -c plan_cache_mode={args.plan_cache_mode}'))
                if args.profile_library:
                    env['PGOPTIONS'] += f' -c session_preload_libraries={args.profile_library}'
                if args.prefetch_distance is not None:
                    env['PGOPTIONS'] += f' -c ann.frozen_prefetch_distance={args.prefetch_distance}'
                env['PGOPTIONS'] += f' -c ann.frozen_materialize_batch={args.materialize_batch}'
                if args.cost_profile:
                    escaped = ''.join(('\\' + c) if c.isspace() or c == '\\' else c for c in args.cost_profile)
                    env['PGOPTIONS'] += f' -c ann.frozen_cost_profile={escaped}'
                command = ['pgbench', args.dsn, '-n', '-M', 'prepared', '-c', str(clients),
                           '-j', str(min(clients, args.threads)), '-T', str(args.seconds), '-r',
                           '-f', str(script), '-l', '--sampling-rate=0.05', '--log-prefix', str(Path(directory) / 'latency')]
                routing_before = routing_stats(args.socket, args.index_name)
                monitor = Monitor(args.dsn, application, args.worker_pid)
                monitor.start()
                started = time.monotonic()
                client = None
                try:
                    with (args.output / f'{tag}.pgbench.txt').open('w') as output:
                        client = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, env=env)
                        monitor.client_pid = client.pid
                        returncode = client.wait(timeout=args.seconds+60)
                finally:
                    if client and client.poll() is None:
                        client.kill()
                        client.wait()
                    elapsed = time.monotonic() - started
                    metrics = monitor.finish(elapsed)
                output = (args.output / f'{tag}.pgbench.txt').read_text()
                tps = re.search(r'tps = ([\d.]+)', output)
                failed = re.search(r'number of failed transactions: (\d+)', output)
                transactions = re.search(r'number of transactions actually processed: (\d+)', output)
                latencies = []
                for path in Path(directory).glob('latency.*'):
                    for line in path.read_text().splitlines():
                        fields = line.split()
                        if len(fields) >= 3:
                            try:
                                latencies.append(int(fields[2]) / 1000)
                            except ValueError:
                                pass
                row = dict(label=args.label, workload=workload, clients=clients, k=args.k,
                           plan_cache_mode=args.plan_cache_mode, setup_cost=args.setup_cost,
                           prefetch_distance=args.prefetch_distance,
                           materialize_batch=args.materialize_batch, cost_profile=args.cost_profile,
                           batch_queries=args.batch_queries, index_name=args.index_name,
                           routing_stats_before=routing_before,
                           routing_stats_after=routing_stats(args.socket, args.index_name),
                           vector_qps=float(tps.group(1)) * max(args.batch_queries, 1) if tps else None,
                           mmap=args.mmap, local_cache_mb=args.local_cache_mb,
                           client_threads=min(clients,args.threads), table=args.table, query_table=args.queries,
                           count_only=args.count_only, seconds=elapsed,
                           qps=float(tps.group(1)) if tps else None, returncode=returncode,
                           failed_transactions=int(failed.group(1)) if failed else None,
                           transactions=int(transactions.group(1)) if transactions else None,
                           latency_samples=len(latencies), sampling_rate=0.05,
                           **{f'p{p}_ms':float(np.percentile(latencies, p)) if latencies else None for p in [50,95,99]},
                           **metrics)
                results.append(row)
                (args.output / f'{tag}.json').write_text(json.dumps(row, indent=2) + '\n')
                print(json.dumps({key:row[key] for key in ['label','workload','clients','qps','vector_qps','p50_ms','p95_ms','p99_ms','failed_transactions']}), flush=True)
                if returncode != 0 or not tps or row['failed_transactions']:
                    raise RuntimeError(f'pgbench failed; inspect {args.output / (tag + ".pgbench.txt")}')
    (args.output / f'{args.label}-results.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
