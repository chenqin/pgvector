#!/usr/bin/env python3
"""Compare fixed and calibrated routing/planning using held-out query vectors."""
import argparse
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time

import psycopg
from psycopg import sql
from calibrate_cost import ROOT, decode, range_program, request
from test_index import load_frame
from test_predicate_cache import search_batch_frame, search_predicate_frame, values_program
from frozen_stress import routing_stats


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dsn',default=os.environ.get('ANN_PG_DSN'),required=not os.environ.get('ANN_PG_DSN'))
    parser.add_argument('--index-dir',required=True)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',default='7')
    parser.add_argument('--query-start',type=int,default=65)
    parser.add_argument('--truth',type=Path,required=True,help='exact held-out per-vector reference from cost_truth.py')
    parser.add_argument('--repeats',type=int,default=7)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'controller.pid').write_text(str(os.getpid()))
    cases=[('tenant=7 AND category BETWEEN 0 AND 1',values_program('tenant',[7])+range_program('category',0,1)+b'\x05')]
    cases += [(f'category BETWEEN 0 AND {hi}',range_program('category',0,hi)) for hi in [1,9,19,49,99]]
    rows=[]; plans=[]
    truth=json.loads(args.truth.read_text()) if args.truth else None
    with psycopg.connect(args.dsn,autocommit=True,prepare_threshold=None) as db:
        db.execute("LOAD 'vector'")
        db.execute('SET jit=off')
        db.execute('SET max_parallel_workers_per_gather=0')
        vectors=[json.loads(x[0]) for x in db.execute('SELECT v FROM frozen_stress_queries WHERE q BETWEEN %s AND %s ORDER BY q',(args.query_start,args.query_start+31))]
        if len(vectors) != 32:raise RuntimeError('validation requires 32 query vectors')
        db.execute('SET ann.enable_frozen_search=off')
        expected={}
        for ci,(predicate,_) in enumerate(cases):
            for k in [250,500,750]:
                expected[ci,k]=set(x[0] for x in db.execute(f'SELECT id FROM frozen_stress WHERE {predicate} ORDER BY emb <-> %s::vector LIMIT %s',(json.dumps(vectors[0]),k)))
    for calibrated in [False,True]:
        label='calibrated' if calibrated else 'fixed'
        path=args.output/f'{label}.sock'
        command=[str(ROOT/'ann/ann-worker-cuvs'),'--backend','cuvs','--listen',str(path),'--lanes','1','--max-batch','32',
                 '--replicate','off','--itopk','512','--filtered-itopk','1024','--filtered-algo','multi','--filtered-width','4',
                 '--exact-limit','0','--gather-limit','4000000','--bitset-gather-limit','500000','--group-gather-limit','500000',
                 '--matrix-gather','on','--gpu-rerank','on','--memory-pool','20','--predicate-cache-mb','2048','--predicate-device-mb','8192']
        if calibrated:command += ['--cost-profile',str(args.profile)]
        with (args.output/f'{label}.log').open('w') as log:
            process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                     env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
        try:
            deadline=time.monotonic()+120
            while not path.exists():
                if process.poll() is not None or time.monotonic()>deadline:raise RuntimeError('worker startup failed')
                time.sleep(.05)
            with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as conn:
                conn.settimeout(120);conn.connect(str(path))
                request(conn,load_frame(1,'frozen_stress',args.index_dir))
                for ci,(predicate,program) in enumerate(cases):
                    for k in [250,500,750]:
                        for batch in [1,4,8,16,32]:
                            packet=search_predicate_frame(2,'frozen_stress',vectors[0],k,program) if batch==1 else search_batch_frame(2,'frozen_stress',vectors[:batch],k,program)
                            samples=[]
                            for rep in range(args.repeats+2):
                                start=time.perf_counter_ns(); result=decode(request(conn,packet),batch)
                                elapsed=(time.perf_counter_ns()-start)/1000
                                if rep>=2:samples.append(elapsed)
                            eligible=result[0][2]
                            recalls=[len(set(hits)&set(truth[predicate][j][:k]))/max(min(k,eligible),1) for j,(hits,_,_) in enumerate(result)] if truth else [len(set(result[0][0])&expected[ci,k])/max(len(expected[ci,k]),1)]
                            recall=statistics.mean(recalls)
                            cardinality=all(len(hits)==min(k,eligible) and len(set(hits))==len(hits) for hits,_,_ in result)
                            if calibrated and (not cardinality or recall<.95):raise RuntimeError(f'calibrated quality failed: {ci} {k} {batch} {recall}')
                            row=dict(mode=label,predicate=predicate,k=k,batch=batch,matching=eligible,us=statistics.median(samples),samples_us=samples,recall=recall,min_recall=min(recalls),query_recalls=recalls,cardinality=cardinality,route=result[0][1])
                            rows.append(row)
                            print(json.dumps({x:row[x] for x in ['mode','matching','k','batch','us','recall','route']}),flush=True)
            stats=routing_stats(str(path),'frozen_stress')
            if calibrated and (stats is None or sum(stats.get('cost_queries',[])) == 0):
                raise RuntimeError('calibrated worker did not use the cost profile')
            (args.output/f'{label}-stats.json').write_text(json.dumps(stats,indent=2)+'\n')
            with psycopg.connect(args.dsn,autocommit=True,prepare_threshold=None) as db:
                db.execute("LOAD 'vector'")
                db.execute("SELECT set_config('ann.worker_socket',%s,false)",(str(path),))
                db.execute('SET max_parallel_workers_per_gather=0');db.execute('SET jit=off')
                if calibrated:db.execute("SELECT set_config('ann.frozen_cost_profile',%s,false)",(str(args.profile),))
                # Include empty/small correlated predicates and large matching sets.
                for predicate in ['tenant=7 AND category=0']+[p for p,_ in cases]:
                    for k in [50,500]:
                        statement=f'SELECT id FROM frozen_stress WHERE {predicate} ORDER BY emb <-> %s::vector LIMIT {k}'
                        db.execute(statement,(json.dumps(vectors[0]),)).fetchall()
                        measurements=[]
                        for _ in range(5):
                            plan=db.execute('EXPLAIN (ANALYZE,FORMAT JSON,TIMING OFF) '+statement,(json.dumps(vectors[0]),)).fetchone()[0][0]
                            measurements.append(plan['Execution Time'])
                        plans.append(dict(mode=label,predicate=predicate,k=k,execution_ms=statistics.median(measurements),plan=plan))
        finally:
            process.terminate()
            try:process.wait(30)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        (args.output/'native.json').write_text(json.dumps(rows,indent=2)+'\n')
        (args.output/'plans.json').write_text(json.dumps(plans,indent=2)+'\n')
    (args.output/'complete.json').write_text(json.dumps({'completed':time.time(),'query_start':args.query_start,'native_points':len(rows),'plan_points':len(plans)})+'\n')


if __name__=='__main__':main()
