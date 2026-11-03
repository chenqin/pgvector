#!/usr/bin/env python3
"""Generate, export and benchmark a sharded ann collection (docs/ann-collections.md).

    # on each shard host: range partitions of tenant-major keys, frozen, indexed and exported
    python3 benchmarks/ann/collection_bench.py generate --dsn 'host=/tmp port=54329 dbname=shard' \\
        --partitions 0-3 --total-partitions 8 --tenants 1024 --rows 16000000 --dimensions 1024 \\
        --export /data/ann/exports

    # on each coordinator: the foreign-partitioned parent, query vectors
    python3 benchmarks/ann/collection_bench.py coordinator --dsn 'port=54341 dbname=postgres' \\
        --shards 's1=host=... port=54329 dbname=shard;s2=...' --total-partitions 8 --dimensions 1024

    # sweep client counts through every coordinator at once; report the best QPS within the SLO
    python3 benchmarks/ann/collection_bench.py bench --dsn 'port=54341 dbname=postgres' \\
        --dsn 'port=54342 dbname=postgres' --collection docs1024 --workloads ann,tenant \\
        --clients 32,64,128,256 --p99-ms 30 --timeout-ms 50

Row i of partition p belongs to tenant t (partitions hold consecutive tenant
ranges) and has key t * 2^20 + i, so key ranges are tenant ranges and a
collection can be placed by tenant. category is uniform over 0..99. Vectors are
clustered in a 16-dimensional latent space and projected, like the other
benchmarks' (isotropic vectors defeat graph indexes). Offline tooling only:
nothing here is on the query path.
"""

import argparse
import concurrent.futures
import os
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np

CLUSTERS = 1024
LATENT = 16
TENANT_SHIFT = 20

WORKLOADS = {
    # plain ANN over the whole collection
    'ann': "SELECT key FROM ann_collection_search(:'coll', :v::vector, 10);",
    # one tenant (0.1% of rows at 1,024 tenants): placement sends it to one part
    'tenant': "SELECT key FROM ann_collection_search(:'coll', :v::vector, 10, 'tenant = ' || :t);",
    # one tenant and half the categories
    'tenant-and': ("SELECT key FROM ann_collection_search(:'coll', :v::vector, 10, "
                   "'tenant = ' || :t || ' AND category < 50');"),
    # 1% of rows in every part
    'category': "SELECT key FROM ann_collection_search(:'coll', :v::vector, 10, 'category = ' || :c);",
    # 10% of rows in every part
    'category10': ("SELECT key FROM ann_collection_search(:'coll', :v::vector, 10, "
                   "'category BETWEEN ' || :d * 10 || ' AND ' || :d * 10 + 9);"),
    # non-covering: rows fetched from the shard hosts through postgres_fdw
    'fetch': ("SELECT id, category FROM docs WHERE id = ANY(ARRAY("
              "SELECT key FROM ann_collection_search(:'coll', :v::vector, 10, 'tenant = ' || :t)));"),
    # the same answer with the attributes from the GPU nodes: no shard fetch
    'rows': ("SELECT * FROM ann_collection_search_rows(:'coll', :v::vector, 10, 'tenant = ' || :t) "
             "AS r(key bigint, distance real, category integer);"),
    'category-rows': ("SELECT * FROM ann_collection_search_rows(:'coll', :v::vector, 10, 'category = ' || :c) "
                      "AS r(key bigint, distance real, tenant integer, category integer);"),
}


def log(message):
    print(time.strftime('%H:%M:%S'), message, file=sys.stderr, flush=True)


def sample(rng, n, dimensions):
    shared = np.random.default_rng(12345)
    centroids = (shared.normal(size=(CLUSTERS, LATENT)) * 2).astype(np.float32)
    projection = (shared.normal(size=(LATENT, dimensions)) / np.sqrt(LATENT)).astype(np.float32)
    latent = centroids[rng.integers(0, CLUSTERS, n)] + rng.normal(size=(n, LATENT)).astype(np.float32)
    return (latent @ projection + rng.normal(scale=0.05, size=(n, dimensions))).astype(np.float32)


def partition_range(args, p):
    per = args.tenants // args.total_partitions
    return p * per, (p + 1) * per


def build_partition(args, p):
    import psycopg

    table = f'docs_{p}'
    first_tenant, end_tenant = partition_range(args, p)
    per_tenant = args.rows // args.tenants
    rng = np.random.default_rng(1000 + p)
    layout = np.dtype([('fields', '>i2'), ('l1', '>i4'), ('id', '>i8'), ('l2', '>i4'), ('tenant', '>i4'),
                       ('l3', '>i4'), ('category', '>i4'), ('l4', '>i4'), ('dim', '>u2'), ('unused', '>u2'),
                       ('vector', '>f4', (args.dimensions,))])
    started = time.time()
    with psycopg.connect(args.dsn, autocommit=True) as db:
        if args.resume and db.execute('SELECT to_regclass(%s) IS NOT NULL', (table,)).fetchone()[0]:
            return finish_partition(args, db, table, p)
        db.execute(f'DROP TABLE IF EXISTS {table}')
        db.execute(f'CREATE TABLE {table} (id bigint, tenant int, category int, emb vector({args.dimensions}))')
        with db.cursor().copy(f'COPY {table} (id, tenant, category, emb) FROM STDIN (FORMAT binary)') as copy:
            copy.write(b'PGCOPY\n\xff\r\n\x00' + b'\x00' * 8)
            for tenant in range(first_tenant, end_tenant):
                for start in range(0, per_tenant, 50000):
                    n = min(50000, per_tenant - start)
                    batch = np.zeros(n, dtype=layout)
                    batch['fields'] = 4
                    batch['l1'] = 8
                    batch['id'] = (tenant << TENANT_SHIFT) + np.arange(start, start + n)
                    batch['l2'] = batch['l3'] = 4
                    batch['tenant'] = tenant
                    batch['category'] = rng.integers(0, 100, n)
                    batch['l4'] = 4 + 4 * args.dimensions
                    batch['dim'] = args.dimensions
                    batch['vector'] = sample(rng, n, args.dimensions)
                    copy.write(batch.tobytes())
            copy.write(b'\xff\xff')
        log(f'{table}: loaded {per_tenant * (end_tenant - first_tenant)} rows in {time.time() - started:.0f}s')
        return finish_partition(args, db, table, p)


def finish_partition(args, db, table, p):
    """Freeze, index and export a loaded partition; steps already done are skipped."""
    exists = lambda name: db.execute('SELECT to_regclass(%s) IS NOT NULL', (name,)).fetchone()[0]
    if not exists(f'{table}_ann'):
        db.execute(f"SELECT ann_freeze('{table}')")
        # Pages stay unfrozen while transactions older than the load run (other loads, here or
        # in other databases of the cluster): retry until they have finished.
        for attempt in range(120):
            db.execute(f'VACUUM (FREEZE) {table}')
            if db.execute(f"SELECT relallvisible = relpages FROM pg_class WHERE oid = '{table}'::regclass"
                          ).fetchone()[0]:
                break
            time.sleep(5)
        built = time.time()
        db.execute(f'CREATE INDEX {table}_ann ON {table} USING ann (emb vector_l2_ops) '
                   'INCLUDE (id, tenant, category) WITH (frozen = true)')
        log(f'{table}: frozen index in {time.time() - built:.0f}s')
    path = os.path.join(args.export, f'export-{p}')
    if not os.path.exists(os.path.join(path, 'manifest')):
        exported = time.time()
        db.execute('SELECT ann_index_export(%s, %s)', (f'{table}_ann', path))
        log(f'{table}: exported to {path} in {time.time() - exported:.0f}s')
    if not exists(f'{table}_id'):
        # Rows of non-covering queries are fetched by key.
        db.execute(f'CREATE INDEX {table}_id ON {table} (id)')
    return p


def parse_partitions(text):
    first, _, last = text.partition('-')
    return list(range(int(first), int(last or first) + 1))


def generate(args):
    partitions = parse_partitions(args.partitions)
    os.makedirs(args.export, exist_ok=True)
    with concurrent.futures.ProcessPoolExecutor(len(partitions)) as pool:
        for p in pool.map(build_partition, [args] * len(partitions), partitions):
            log(f'partition {p} done')


def coordinator(args):
    """The parent table with foreign partitions, and the query vectors, on one coordinator."""
    import psycopg

    shards = dict(entry.split('=', 1) for entry in args.shards.split(';'))
    owner = {}
    for name, value in shards.items():
        conninfo, _, parts = value.rpartition('@')
        for p in parse_partitions(parts):
            owner[p] = (name, conninfo)
    per = args.tenants // args.total_partitions
    with psycopg.connect(args.dsn, autocommit=True) as db:
        db.execute('CREATE EXTENSION IF NOT EXISTS postgres_fdw')
        db.execute('DROP TABLE IF EXISTS docs CASCADE')
        db.execute(f'CREATE TABLE docs (id bigint, tenant int, category int, emb vector({args.dimensions})) '
                   'PARTITION BY RANGE (id)')
        for name, conninfo in set(owner.values()):
            options = dict(item.split('=', 1) for item in conninfo.split())
            db.execute(f'DROP SERVER IF EXISTS {name} CASCADE')
            db.execute(f'CREATE SERVER {name} FOREIGN DATA WRAPPER postgres_fdw OPTIONS ('
                       + ', '.join(f"{key} '{value}'" for key, value in options.items() if key != 'user') + ')')
            db.execute(f'CREATE USER MAPPING FOR CURRENT_USER SERVER {name}'
                       + (f" OPTIONS (user '{options['user']}')" if 'user' in options else ''))
        for p, (name, _) in sorted(owner.items()):
            low, high = p * per << TENANT_SHIFT, (p + 1) * per << TENANT_SHIFT
            db.execute(f'CREATE FOREIGN TABLE docs_{p} PARTITION OF docs FOR VALUES FROM ({low}) TO ({high}) '
                       f"SERVER {name} OPTIONS (table_name 'docs_{p}')")
        db.execute('DROP TABLE IF EXISTS docs_queries')
        db.execute('CREATE TABLE docs_queries (q int PRIMARY KEY, v text)')
        queries = sample(np.random.default_rng(999), 1000, args.dimensions)
        with db.cursor().copy('COPY docs_queries FROM STDIN') as copy:
            for i, row in enumerate(queries, 1):
                copy.write_row((i, '[' + ','.join(f'{x:.6g}' for x in row) + ']'))
    log(f'coordinator {args.dsn}: docs with {len(owner)} foreign partitions')


def run_pgbench(dsn, cores, script, clients, threads, seconds, options, prefix):
    command = ['pgbench', dsn, '-n', '-M', 'prepared', '-c', str(clients), '-j', str(min(threads, clients)),
               '-T', str(seconds), '-f', script, '--log', '--log-prefix', prefix]
    if cores:
        command = ['taskset', '-c', cores] + command
    return subprocess.run(command, capture_output=True, text=True, env={**os.environ, 'PGOPTIONS': options})


def latencies(prefix):
    values = []
    directory = os.path.dirname(prefix)
    for name in os.listdir(directory):
        if name.startswith(os.path.basename(prefix) + '.'):
            with open(os.path.join(directory, name)) as data:
                for line in data:
                    fields = line.split()
                    if len(fields) >= 3 and fields[2].isdigit():
                        values.append(int(fields[2]) / 1000.0)
    return values


def bench(args):
    # The search deadline is ann.worker_timeout_ms; statement_timeout only guards a new
    # connection's first search, which also connects to every node.
    options = (f'-c ann.worker_timeout_ms={args.timeout_ms} -c statement_timeout={args.statement_timeout_ms} '
               + args.options)
    cores = args.cores.split(';') if args.cores else [None] * len(args.dsn)
    print(f'{"workload":<11} {"clients":>8} {"QPS":>9} {"p50":>7} {"p95":>7} {"p99":>7} {"over":>6} '
          f'{"aborted":>8}  SLO')
    for workload in args.workloads.split(','):
        best = None
        with tempfile.TemporaryDirectory() as directory:
            script = os.path.join(directory, f'{workload}.sql')
            with open(script, 'w') as out:
                out.write(f'\\set q random(1, 1000)\n\\set t random(0, {args.tenants - 1})\n'
                          '\\set c random(0, 99)\n\\set d random(0, 9)\n')
                out.write('SELECT v FROM docs_queries WHERE q = :q \\gset\n')
                out.write(WORKLOADS[workload].replace(":'coll'", f"'{args.collection}'") + '\n')
            # warm up: caches, connections, program translation
            with concurrent.futures.ThreadPoolExecutor(len(args.dsn)) as pool:
                list(pool.map(lambda i: run_pgbench(args.dsn[i], cores[i], script, 64, args.threads, 3, options,
                                                    os.path.join(directory, f'warm{i}')), range(len(args.dsn))))
            for clients in [int(c) for c in args.clients.split(',')]:
                prefixes = [os.path.join(directory, f'{workload}-{clients}-{i}') for i in range(len(args.dsn))]
                with concurrent.futures.ThreadPoolExecutor(len(args.dsn)) as pool:
                    results = list(pool.map(
                        lambda i: run_pgbench(args.dsn[i], cores[i], script, clients, args.threads, args.seconds,
                                              options, prefixes[i]), range(len(args.dsn))))
                values = sorted(v for prefix in prefixes for v in latencies(prefix))
                aborted = sum(r.stderr.count('aborted') + r.stdout.count('aborted in command') for r in results)
                if not values:
                    print(f'{workload:<11} {clients:>8} no transactions: {results[0].stderr.strip()[:200]}')
                    continue
                qps = len(values) / args.seconds
                p = [values[min(len(values) - 1, int(len(values) * q))] for q in (0.5, 0.95, 0.99)]
                over = sum(1 for v in values if v > args.timeout_ms)
                ok = p[2] <= args.p99_ms and aborted == 0
                print(f'{workload:<11} {clients * len(args.dsn):>8} {qps:>9.0f} {p[0]:>7.2f} {p[1]:>7.2f} '
                      f'{p[2]:>7.2f} {over:>6} {aborted:>8}  {"ok" if ok else "-"}', flush=True)
                if ok and (best is None or qps > best[1]):
                    best = (clients * len(args.dsn), qps, p[2])
        if best:
            print(f'{workload:<11} best within p99 <= {args.p99_ms} ms: {best[1]:.0f} QPS at {best[0]} clients '
                  f'(p99 {best[2]:.1f} ms)', flush=True)
        else:
            print(f'{workload:<11} no client count met the SLO', flush=True)


def recall(args):
    """Recall@k of collection searches against exact top k merged from the shard hosts' partitions."""
    import psycopg

    shards = []
    for entry in args.shards.split(';'):
        conninfo, _, parts = entry.rpartition('@')
        shards.append((conninfo, parse_partitions(parts)))
    filters = [None if f == 'none' else f for f in args.filters.split(';')]
    with psycopg.connect(args.dsn, autocommit=True) as db:
        db.execute(f"SET ann.worker_auth_token = '{args.token}'") if args.token else None
        queries = [row[0] for row in db.execute('SELECT v FROM docs_queries ORDER BY q LIMIT %s',
                                                (args.queries,)).fetchall()]
        connections = [psycopg.connect(conninfo, autocommit=True) for conninfo, _ in shards]
        try:
            for c in connections:
                c.execute("SET ann.worker_address = ''")
                c.execute("SET ann.worker_socket = ''")
            print(f'{"collection":<12} {"filter":<34} recall@{args.k}')
            for condition in filters:
                for collection in args.collection.split(','):
                    found = total = 0
                    for query in queries:
                        exact = []
                        for (conninfo, parts), c in zip(shards, connections):
                            for p in parts:
                                where = f'WHERE {condition}' if condition else ''
                                exact += c.execute(f'SELECT id, (emb <-> %s::vector)::real FROM docs_{p} {where} '
                                                   f'ORDER BY emb <-> %s::vector LIMIT %s',
                                                   (query, query, args.k)).fetchall()
                        exact.sort(key=lambda row: (row[1], row[0]))
                        truth = {key for key, _ in exact[:args.k]}
                        got = {row[0] for row in db.execute(
                            'SELECT key FROM ann_collection_search(%s, %s::vector, %s, %s)',
                            (collection, query, args.k, condition)).fetchall()}
                        found += len(truth & got)
                        total += len(truth)
                    print(f'{collection:<12} {condition or "none":<34} {found / max(total, 1):.3f}', flush=True)
        finally:
            for c in connections:
                c.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    g = commands.add_parser('generate')
    g.add_argument('--dsn', required=True)
    g.add_argument('--partitions', required=True, help='this host\'s partitions, e.g. 0-3')
    g.add_argument('--total-partitions', type=int, required=True)
    g.add_argument('--tenants', type=int, default=1024)
    g.add_argument('--rows', type=int, required=True, help='rows of the whole collection')
    g.add_argument('--dimensions', type=int, default=1024)
    g.add_argument('--export', required=True, help='directory for ann_index_export output')
    g.add_argument('--resume', action='store_true', help='keep loaded tables; finish freezing, indexing, exporting')
    c = commands.add_parser('coordinator')
    c.add_argument('--dsn', required=True)
    c.add_argument('--shards', required=True, help="'name=conninfo@0-3;name2=conninfo@4-7'")
    c.add_argument('--total-partitions', type=int, required=True)
    c.add_argument('--tenants', type=int, default=1024)
    c.add_argument('--dimensions', type=int, default=1024)
    b = commands.add_parser('bench')
    b.add_argument('--dsn', action='append', required=True, help='a coordinator (repeatable)')
    b.add_argument('--cores', help="taskset core lists per coordinator, ';'-separated")
    b.add_argument('--collection', required=True)
    b.add_argument('--workloads', default='ann,tenant,category')
    b.add_argument('--tenants', type=int, default=1024)
    b.add_argument('--clients', default='16,32,64,128,256', help='clients per coordinator')
    b.add_argument('--threads', type=int, default=32)
    b.add_argument('--seconds', type=int, default=15)
    b.add_argument('--p99-ms', type=float, default=30)
    b.add_argument('--timeout-ms', type=int, default=50, help='search deadline (ann.worker_timeout_ms)')
    b.add_argument('--statement-timeout-ms', type=int, default=1000)
    b.add_argument('--options', default='', help='extra PGOPTIONS')
    r = commands.add_parser('recall')
    r.add_argument('--dsn', required=True, help='a coordinator')
    r.add_argument('--shards', required=True, help="'conninfo@0-3;conninfo@4-7'")
    r.add_argument('--collection', required=True, help='comma-separated collections')
    r.add_argument('--filters', default='none;tenant = 7;category = 3;category BETWEEN 30 AND 39')
    r.add_argument('--queries', type=int, default=20)
    r.add_argument('--k', type=int, default=10)
    r.add_argument('--token', help='ann.worker_auth_token for the coordinator session')
    args = parser.parse_args()
    {'generate': generate, 'coordinator': coordinator, 'bench': bench, 'recall': recall}[args.command](args)


if __name__ == '__main__':
    main()
