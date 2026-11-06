#!/usr/bin/env python3
"""Exact FP64 GPU reference for held-out vectors; cross-check against SQL.

Uses original FP32 exported vectors, never NVFP4 scores or CAGRA candidates.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import psycopg


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dsn',default=os.environ.get('ANN_PG_DSN'),required=not os.environ.get('ANN_PG_DSN'))
    p.add_argument('--index-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dimensions',type=int,default=128)
    p.add_argument('--query-start',type=int,default=65)
    p.add_argument('--calibration-predicates',action='store_true')
    args=p.parse_args()
    import cupy as cp
    vectors=np.memmap(args.index_dir/'vectors.f32',mode='r',dtype='<f4').reshape(-1,args.dimensions)
    keys=np.memmap(args.index_dir/'keys.i64',mode='r',dtype='<i8')
    if len(keys) != len(vectors):raise RuntimeError('vector dimensions/keys do not agree')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with psycopg.connect(args.dsn,autocommit=True,prepare_threshold=None) as db:
        queries=[json.loads(r[0]) for r in db.execute('SELECT v FROM frozen_stress_queries WHERE q BETWEEN %s AND %s ORDER BY q',(args.query_start,args.query_start+31))]
        gpu_vectors=cp.asarray(vectors,dtype=cp.float64)
        gpu_queries=cp.asarray(queries,dtype=cp.float64)
        query_norm=cp.sum(gpu_queries*gpu_queries,axis=1)
        cases=['tenant=7 AND category BETWEEN 0 AND 1']+[f'category BETWEEN 0 AND {hi}' for hi in [1,9,19,49,99]]
        if args.calibration_predicates:
            from calibrate_cost import predicates
            cases=list(dict.fromkeys([p for p,_ in predicates()]+cases))
        result={}
        for predicate in cases:
            count,bitmap=db.execute("SELECT eligible,bitmap FROM ann_frozen_filter('frozen_stress_ann',%s)",(predicate,)).fetchone()
            ordinals=np.flatnonzero(np.unpackbits(np.frombuffer(bitmap,dtype=np.uint8),bitorder='little')[:len(keys)])
            assert len(ordinals)==count
            if count == 0:
                result[predicate]=[[] for _ in queries]
                args.output.write_text(json.dumps(result)+'\n')
                continue
            indices=cp.asarray(ordinals)
            selected=gpu_vectors[indices]
            scores=-2*(gpu_queries@selected.T)
            scores+=cp.sum(selected*selected,axis=1)[None,:]
            scores+=query_norm[:,None]
            top=min(1000,count)
            positions=cp.argpartition(scores,top-1,axis=1)[:,:top]
            order=cp.argsort(cp.take_along_axis(scores,positions,axis=1),axis=1)
            positions=cp.take_along_axis(positions,order,axis=1)
            truth=keys[cp.asnumpy(indices[positions])].tolist()
            db.execute('SET ann.enable_frozen_search=off')
            sql_keys=[r[0] for r in db.execute(f'SELECT id FROM frozen_stress WHERE {predicate} ORDER BY emb <-> %s::vector LIMIT 1000',(json.dumps(queries[0]),))]
            if set(sql_keys)!=set(truth[0]):raise RuntimeError('FP64 reference disagrees with SQL')
            result[predicate]=truth
            args.output.write_text(json.dumps(result)+'\n')
            print('reference',predicate,'rows',count,'queries',len(truth),flush=True)
            del selected,scores,positions,order,indices
            cp.get_default_memory_pool().free_all_blocks()
    print('FP64 exact references agree with SQL on every first vector',flush=True)


if __name__=='__main__':main()
