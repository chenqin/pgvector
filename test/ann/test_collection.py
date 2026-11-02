"""Collections: a table's partitions built into parts served by several worker nodes, searched together.

Needs ANN_PG_DSN for a local superuser (exports write server files, and the
workers read them). Builds the CPU worker and runs scripts/ann/build_cuvs_index.py
--no-cagra, which needs numpy and psycopg. Each node is a CPU worker on its
own TCP port, so the coordinator's fan-out, replica choice and failover run
as they would across hosts.
"""

import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_server import ROOT, STATS, Server, frame, receive  # noqa: E402

DSN = os.environ.get('ANN_PG_DSN')
BUILD = os.path.join(ROOT, 'scripts', 'ann', 'build_cuvs_index.py')
TENANTS = 16
PER_TENANT = 150


def free_port():
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]


@unittest.skipUnless(DSN, 'set ANN_PG_DSN for live PostgreSQL tests')
class CollectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.run(['make', '-C', os.path.join(ROOT, 'ann'), 'ann-worker'], check=True,
                       stdout=subprocess.DEVNULL)

    def setUp(self):
        import psycopg
        from psycopg import sql

        self.psycopg = psycopg
        self.sql = sql
        self.directory = tempfile.TemporaryDirectory()
        os.chmod(self.directory.name, 0o777)
        self.token = os.path.join(self.directory.name, 'token')
        with open(self.token, 'w') as out:
            out.write('collection-test-token\n')
        os.chmod(self.token, 0o600)
        self.nodes = {}
        self.db = psycopg.connect(DSN, autocommit=True)
        self.db.execute('SET synchronous_commit = on')
        self.schema = 'ann_coll_' + uuid.uuid4().hex[:12]
        self.db.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(self.schema)))
        self.db.execute(sql.SQL('SET search_path = {}, public').format(sql.Identifier(self.schema)))
        self.db.execute("SET ann.worker_auth_token = 'collection-test-token'")
        self.db.execute('SET ann.worker_heartbeat_ms = 0')
        self.db.execute('SET ann.worker_retry_ms = 60000')

    def tearDown(self):
        for node in self.nodes.values():
            if node.process.poll() is None:
                node.close()
        self.db.execute(self.sql.SQL('DROP SCHEMA {} CASCADE').format(self.sql.Identifier(self.schema)))
        self.db.execute('DELETE FROM ann_collection WHERE name LIKE %s', (self.schema + '%',))
        self.db.close()
        self.directory.cleanup()

    def node(self, name):
        port = free_port()
        self.nodes[name] = Server('--backend', 'cpu-exact', '--listen-tcp', f'127.0.0.1:{port}',
                                  '--auth-token-file', self.token)
        self.nodes[name].address = f'127.0.0.1:{port}'
        return self.nodes[name].address

    def stats(self, name):
        with self.nodes[name].connect() as connection:
            connection.sendall(frame(STATS, 1))
            return json.loads(receive(connection)[3])

    def value(self, statement, params=None):
        return self.db.execute(statement, params).fetchone()[0]

    def error(self, statement, params=None):
        with self.assertRaises(self.psycopg.Error) as caught:
            self.db.execute(statement, params)
        return str(caught.exception)

    def vacuum(self, table):
        for _ in range(5):
            self.db.execute(f'VACUUM (FREEZE) {table}')
            if self.value(f"SELECT relallvisible = relpages FROM pg_class WHERE oid = '{table}'::regclass"):
                return

    def create(self, partitions=4):
        """docs, range-partitioned by id; tenant t owns ids t * 1000 .. t * 1000 + 149, so the key
        ranges of the partitions are tenant ranges too."""
        self.db.execute('CREATE TABLE docs (id bigint, tenant int, category int, body text, emb vector(4)) '
                        'PARTITION BY RANGE (id)')
        per = TENANTS // partitions
        for p in range(partitions):
            self.db.execute(f'CREATE TABLE docs_{p} PARTITION OF docs FOR VALUES FROM ({p * per * 1000}) '
                            f'TO ({(p + 1) * per * 1000})')
        self.db.execute(f"""INSERT INTO docs SELECT t * 1000 + j, t, (t * 7 + j) % 5, 'row ' || j,
                                ARRAY[sin(t * 1000 + j) * 50, cos((t * 1000 + j) * 1.3) * 40,
                                      sin((t * 1000 + j) * 0.7) * 30, j % 9]::vector
                            FROM generate_series(0, {TENANTS - 1}) t, generate_series(0, {PER_TENANT - 1}) j""")
        exports = []
        for p in range(partitions):
            self.assertTrue(self.value(f"SELECT ann_freeze('docs_{p}')"))
            self.vacuum(f'docs_{p}')
            self.db.execute(f'CREATE INDEX docs_{p}_ann ON docs_{p} USING ann (emb vector_l2_ops) '
                            'INCLUDE (id, tenant, category) WITH (frozen = true)')
            export = os.path.join(self.directory.name, f'export-{p}')
            self.db.execute('SELECT ann_index_export(%s, %s)', (f'docs_{p}_ann', export))
            exports.append(export)
        return exports

    def build(self, name, exports, extra=()):
        path = os.path.join(self.directory.name, name)
        command = [sys.executable, BUILD, '--out', path, '--no-cagra', *extra]
        for export in exports:
            command += ['--from-export', export]
        subprocess.run(command, check=True, capture_output=True)
        return path

    def setup_collection(self):
        """Two parts: partitions 0-1 on replicas a and b, partitions 2-3 on node c."""
        exports = self.create()
        parts = [self.build('part-0', exports[:2]), self.build('part-1', exports[2:])]
        a, b, c = self.node('a'), self.node('b'), self.node('c')
        self.collection = self.schema + '_docs'
        self.db.execute('SELECT ann_collection_create(%s, %s, %s, %s)',
                        (self.collection, f'{self.schema}.docs', ['tenant', 'category'], 'tenant'))
        index0, index1 = self.collection + '_p0', self.collection + '_p1'
        self.assertEqual(self.value('SELECT ann_collection_load(%s, 0, %s, %s, %s, 0, 7)',
                                    (self.collection, index0, parts[0], f'{a},{b}')), 8 * PER_TENANT)
        self.assertEqual(self.value('SELECT ann_collection_load(%s, 1, %s, %s, %s, 8, 15)',
                                    (self.collection, index1, parts[1], c)), 8 * PER_TENANT)
        return parts

    def search(self, query, k, condition=None):
        return [(key, round(distance, 3)) for key, distance in self.db.execute(
            'SELECT key, distance FROM ann_collection_search(%s, %s, %s, %s) ORDER BY distance, key',
            (self.collection, query, k, condition)).fetchall()]

    def exact(self, query, k, condition=None):
        where = f'WHERE {condition}' if condition else ''
        return [(key, round(distance, 3)) for key, distance in self.db.execute(
            f'SELECT id, (emb <-> %s::vector)::real AS d FROM docs {where} ORDER BY d, id LIMIT %s',
            (query, k)).fetchall()]

    def test_builder_combines_exports(self):
        exports = self.create()
        path = self.build('combined', exports[:2])
        with open(os.path.join(path, 'manifest')) as manifest:
            lines = manifest.read().splitlines()
        members = [line for line in lines if line.startswith('# member ')]
        self.assertEqual(len(members), 2)
        self.assertIn(f'rows {8 * PER_TENANT}', lines)
        import numpy as np
        keys = np.fromfile(os.path.join(path, 'keys.i64'), dtype='<i8')
        self.assertTrue((np.diff(keys) > 0).all())
        tenants = np.fromfile(os.path.join(path, 'attr-tenant.i32'), dtype='<i4')
        self.assertTrue((tenants == keys // 1000).all())
        # Overlapping members are refused.
        result = subprocess.run([sys.executable, BUILD, '--out', os.path.join(self.directory.name, 'bad'),
                                 '--no-cagra', '--from-export', exports[0], '--from-export', exports[0]],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('overlap', result.stderr)

    def test_search_matches_sql(self):
        self.setup_collection()
        for query in ['[1,2,3,4]', '[-30,10,5,0]', '[40,-20,0,8]']:
            for condition in [None, 'category = 3', 'tenant IN (2, 9, 14) AND category <> 1',
                              'tenant BETWEEN 6 AND 9', 'NOT (tenant < 12) OR category = 0',
                              'tenant = 99']:
                for k in [1, 10, 37]:
                    self.assertEqual(self.search(query, k, condition), self.exact(query, k, condition),
                                     (query, condition, k))

    def test_placement_prunes_parts(self):
        self.setup_collection()
        before = {name: self.stats(name)['jobs'] for name in 'abc'}
        for _ in range(6):
            self.search('[1,2,3,4]', 5, 'tenant = 11 AND category = 2')
        middle = {name: self.stats(name)['jobs'] for name in 'abc'}
        self.assertEqual(middle['c'] - before['c'], 6)
        self.assertEqual(middle['a'] + middle['b'], before['a'] + before['b'])
        # Unpinned filters reach every part; the replicas of part 0 share its requests.
        for _ in range(8):
            self.search('[1,2,3,4]', 5, 'category = 2')
        after = {name: self.stats(name)['jobs'] for name in 'abc'}
        self.assertEqual(after['c'] - middle['c'], 8)
        self.assertEqual((after['a'] - middle['a']) + (after['b'] - middle['b']), 8)
        # An OR spanning both parts' ranges reaches both.
        self.assertEqual(self.search('[1,2,3,4]', 10, 'tenant = 1 OR tenant = 12'),
                         self.exact('[1,2,3,4]', 10, 'tenant = 1 OR tenant = 12'))

    def test_failover_and_errors(self):
        self.setup_collection()
        expected = self.exact('[1,2,3,4]', 10)
        self.nodes['a'].close()
        for _ in range(4):
            self.assertEqual(self.search('[1,2,3,4]', 10), expected)
        self.nodes['c'].close()
        self.assertIn('no node of ann collection', self.error(
            'SELECT * FROM ann_collection_search(%s, %s, 10)', (self.collection, '[1,2,3,4]')))
        # A filter pinned to part 0 still works without part 1's node.
        self.assertEqual(self.search('[1,2,3,4]', 10, 'tenant = 3'), self.exact('[1,2,3,4]', 10, 'tenant = 3'))
        self.assertIn('is not a predicate', self.error(
            "SELECT * FROM ann_collection_search(%s, %s, 10, 'body = ''x''')", (self.collection, '[1,2,3,4]')))
        self.assertIn('does not exist', self.error(
            "SELECT * FROM ann_collection_search('nope', '[1,2,3,4]', 10)"))

    def test_catalog_checks(self):
        self.create(partitions=2)
        self.assertIn('must be smallint, integer', self.error(
            'SELECT ann_collection_create(%s, %s, %s)', (self.schema + '_x', f'{self.schema}.docs', ['body'])))
        self.assertIn('is not one of', self.error(
            'SELECT ann_collection_create(%s, %s, %s, %s)',
            (self.schema + '_x', f'{self.schema}.docs', ['tenant'], 'category')))
        self.assertIn('is not a column', self.error(
            'SELECT ann_collection_create(%s, %s, %s)', (self.schema + '_x', f'{self.schema}.docs', ['missing'])))


if __name__ == '__main__':
    unittest.main()
