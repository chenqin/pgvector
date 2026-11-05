#!/usr/bin/env python3
"""Run the native operation-22 client, bypassing PostgreSQL query execution.

Compile frozen_direct.c with cc -O2 -std=c11 -pthread, then pass --binary.
The database supplies only the initial query pool. The timed request path
is C, with one connection and one outstanding request per client.
"""

import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import time

import psycopg
from psycopg import sql

from frozen_stress import Monitor, identifier


def packets(args, workload):
    name = args.index.encode()
    output = []
    with psycopg.connect(args.dsn) as db:
        rows = db.execute(sql.SQL('SELECT q,v FROM {} ORDER BY q').format(identifier(args.queries))).fetchall()
    for q, vector in rows:
        values = json.loads(vector)
        if workload == '0.1pct':
            program = b'\x01\x06tenant\x00' + struct.pack('<Ii', 1, q % 100)
        elif workload == '1pct':
            program = b'\x01\x08category\x00' + struct.pack('<Ii', 1, q % 100)
        else:
            low = q % 10 * 10
            program = b'\x02\x08category\x00' + struct.pack('<ii', low, low + 9)
        body = struct.pack('<I', len(name)) + name
        body += struct.pack('<IIII', len(values), args.k, 1, len(program)) + program
        body += struct.pack(f'<{len(values)}f', *values)
        output.append(b'PVAN' + struct.pack('<HHQQII', 1, 22, 1, 0, len(body), 0) + body)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dsn', default=os.environ.get('ANN_PG_DSN'), required=not os.environ.get('ANN_PG_DSN'))
    parser.add_argument('--queries', default='public.frozen_stress_queries')
    parser.add_argument('--index', default='frozen_stress')
    parser.add_argument('--socket', required=True)
    parser.add_argument('--binary', required=True)
    parser.add_argument('--k', type=int, default=500)
    parser.add_argument('--clients', type=int, default=256)
    parser.add_argument('--seconds', type=int, default=20)
    parser.add_argument('--workloads', default='0.1pct,1pct,10pct')
    parser.add_argument('--worker-pid', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--log-dir', default='/dev/shm')
    parser.add_argument('--label', default='direct')
    args = parser.parse_args()
    if not 1 <= args.k <= 10000 or min(args.clients,args.seconds) < 1:
        parser.error('k must be 1..10000 and clients/seconds must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    output = []
    for workload in args.workloads.split(','):
        if workload not in ['0.1pct','1pct','10pct']:
            parser.error(f'unknown workload: {workload}')
        requests = packets(args, workload)
        if not requests or any(len(r) != len(requests[0]) for r in requests):
            raise RuntimeError('Query pool must contain vectors of one dimension')
        with tempfile.TemporaryDirectory(prefix='frozen-direct-', dir=args.log_dir) as directory:
            path = Path(directory) / 'requests.bin'
            with path.open('wb') as data:
                data.write(struct.pack('<II',len(requests),len(requests[0])))
                data.writelines(requests)
            monitor = Monitor(args.dsn, 'frozen_direct_observer_only', args.worker_pid)
            monitor.start()
            started = time.monotonic()
            client = None
            try:
                client = subprocess.Popen([args.binary,args.socket,str(path),str(args.seconds),str(args.clients),str(args.k)],
                                          stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
                monitor.client_pid = client.pid
                stdout, stderr = client.communicate(timeout=args.seconds+60)
            finally:
                if client and client.poll() is None:
                    client.kill()
                    client.wait()
                metrics = monitor.finish(time.monotonic()-started)
            if client.returncode != 0:
                raise RuntimeError(f'Native client failed: {stdout}\n{stderr}')
            row = dict(mode='direct-worker',label=args.label,workload=workload,k=args.k,
                       sampling_rate=.05,latency_bin_ms=.01,**json.loads(stdout),**metrics)
            output.append(row)
            (args.output / f'{args.label}-{workload}.json').write_text(json.dumps(row,indent=2)+'\n')
            print(json.dumps({key:row[key] for key in ['mode','workload','clients','qps','p50_ms','p95_ms','p99_ms','errors']}),flush=True)
    (args.output / f'{args.label}-results.json').write_text(json.dumps(output,indent=2)+'\n')


if __name__ == '__main__':
    main()
