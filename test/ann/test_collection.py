"""Collections: a table's partitions built into parts served by several worker nodes, searched together.

Needs ANN_PG_DSN for a local superuser (exports write server files, and the
workers read them). Builds the CPU worker and runs ann/builder/build_cuvs_index.py
--no-cagra, which needs numpy and psycopg. Each node is a CPU worker on its
own TCP port, so the coordinator's fan-out, replica choice and failover run
as they would across hosts.
"""

import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import unittest
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_server import ROOT, STATS, Server, frame, receive  # noqa: E402

DSN = os.environ.get('ANN_PG_DSN')
BUILD = os.path.join(ROOT, 'ann', 'builder', 'build_cuvs_index.py')
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
        subprocess.run(['make', '-C', os.path.join(ROOT, 'ann', 'server'), 'ann-worker'], check=True,
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

    def rows(self, query, k, condition=None, columns='tenant integer, category integer'):
        return [(key, round(distance, 3), *rest) for key, distance, *rest in self.db.execute(
            f'SELECT * FROM ann_collection_search_rows(%s, %s, %s, %s) AS r(key bigint, distance real, {columns}) '
            'ORDER BY distance, key', (self.collection, query, k, condition)).fetchall()]

    def exact_rows(self, query, k, condition=None, table='docs', columns='tenant, category'):
        where = f'WHERE {condition}' if condition else ''
        return [(key, round(distance, 3), *rest) for key, distance, *rest in self.db.execute(
            f'SELECT id, (emb <-> %s::vector)::real AS d, {columns} FROM {table} {where} ORDER BY d, id LIMIT %s',
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

    def test_one_request_per_replica_set(self):
        """Parts on the same replicas go in one request (operation 25), merged by the node."""
        exports = self.create()
        parts = [self.build('p0', exports[:1]), self.build('p1', exports[1:2]), self.build('p2', exports[2:])]
        a, b, c = self.node('a'), self.node('b'), self.node('c')
        self.collection = self.schema + '_docs'
        self.db.execute('SELECT ann_collection_create(%s, %s, %s, %s)',
                        (self.collection, f'{self.schema}.docs', ['tenant', 'category'], 'tenant'))
        names = [self.collection + f'_p{i}' for i in range(3)]
        for i, (low, high, nodes) in enumerate([(0, 3, f'{a},{b}'), (4, 7, f'{a},{b}'), (8, 15, c)]):
            self.db.execute('SELECT ann_collection_load(%s, %s, %s, %s, %s, %s, %s)',
                            (self.collection, i, names[i], parts[i], nodes, low, high))
        for query in ['[1,2,3,4]', '[-30,10,5,0]']:
            for condition in [None, 'category = 3', 'tenant BETWEEN 2 AND 6', 'tenant IN (1, 12)']:
                self.assertEqual(self.search(query, 9, condition), self.exact(query, 9, condition),
                                 (query, condition))
                self.assertEqual(self.rows(query, 9, condition), self.exact_rows(query, 9, condition),
                                 (query, condition))
        # Straight to a node: both parts in one request.
        generations = {name: int(g, 16) for name, g in self.db.execute(
            'SELECT index_name, generation FROM ann_collection_part WHERE collection = %s',
            (self.collection,)).fetchall()}

        def request(parts, k=5, flags=0):
            payload = struct.pack('<I', len(parts))
            for name, generation in parts:
                payload += struct.pack('<I', len(name)) + name.encode() + struct.pack('<Q', generation)
            query = [1.0, 2.0, 3.0, 4.0]
            payload += struct.pack('<IIII', 4, k, flags, 1) + bytes([8]) + struct.pack('<4f', *query)
            with self.nodes['a'].connect() as connection:
                connection.sendall(frame(25, 7, payload))
                return receive(connection)

        _, _, status, payload = request([(names[0], generations[names[0]]), (names[1], generations[names[1]])])
        self.assertEqual(status, 0)
        count, _, eligible = struct.unpack_from('<IIQ', payload)
        self.assertEqual(eligible, 8 * PER_TENANT)
        keys = [struct.unpack_from('<q', payload, 16 + i * 12)[0] for i in range(count)]
        self.assertEqual(keys, [key for key, _ in self.exact('[1,2,3,4]', 5, 'tenant < 8')])
        self.assertEqual(request([(names[0], generations[names[0]] ^ 1), (names[1], 0)])[2], 12)
        self.assertEqual(request([(names[0], 0), ('missing', 0)])[2], 3)
        self.assertEqual(request([(names[0], 0)], flags=1)[2], 1)
        # With ANN_PREDICATE_ATTRIBUTES: u32 attributes, then each hit's values.
        _, _, status, payload = request([(names[0], 0), (names[1], 0)], flags=2)
        self.assertEqual(status, 0)
        count = struct.unpack_from('<I', payload)[0]
        (nattributes,) = struct.unpack_from('<I', payload, 16 + count * 12)
        self.assertEqual(nattributes, 2)
        values = struct.unpack_from(f'<{count * 2}i', payload, 20 + count * 12)
        self.assertEqual([(values[2 * i], values[2 * i + 1]) for i in range(count)],
                         [tuple(r[2:]) for r in self.exact_rows('[1,2,3,4]', 5, 'tenant < 8')])

    def test_delta_rows_are_searched(self):
        """Rows not yet in a part sit in a delta table, scored exactly with the same filter and merged in."""
        self.setup_collection()
        self.db.execute('CREATE TABLE docs_delta (id bigint, tenant int, category int, emb vector(4))')
        self.db.execute("""INSERT INTO docs_delta SELECT 900000 + i, i % 16, i % 5,
                               ARRAY[1 + i * 0.01, 2, 3, 4 + i % 3]::vector FROM generate_series(0, 39) i""")
        nearest = self.exact('[1,2,3,4]', 1)[0][0]
        # A row moving into a part is briefly in both: returned once.
        self.db.execute('INSERT INTO docs_delta SELECT id, tenant, category, emb FROM docs WHERE id = %s', (nearest,))
        self.db.execute("SELECT ann_collection_delta(%s, 'docs_delta', 'id', 'emb')", (self.collection,))

        def exact(query, k, condition=None):
            where = f'WHERE {condition}' if condition else ''
            return [(key, round(distance, 3)) for key, distance in self.db.execute(
                f"""SELECT id, min(d) FROM (SELECT id, (emb <-> %s::vector)::real AS d FROM docs {where}
                    UNION ALL SELECT id, (emb <-> %s::vector)::real FROM docs_delta {where}) s
                    GROUP BY id ORDER BY 2, 1 LIMIT %s""", (query, query, k)).fetchall()]

        for query in ['[1,2,3,4]', '[-30,10,5,0]']:
            for condition in [None, 'tenant = 3', 'category = 2 AND tenant > 9', 'tenant IN (1, 12)']:
                for k in [1, 10, 60]:
                    self.assertEqual(self.search(query, k, condition), exact(query, k, condition),
                                     (query, condition, k))
        self.assertEqual(len([key for key, _ in self.search('[1,2,3,4]', 60) if key == nearest]), 1)
        # A filter that prunes every part still scores the delta.
        self.db.execute('INSERT INTO docs_delta VALUES (990000, 40, 1, %s)', ('[1,2,3,4]',))
        self.assertEqual(self.search('[1,2,3,4]', 5, 'tenant = 40'), [(990000, 0.0)])
        # Another session sees the delta too.
        with self.psycopg.connect(DSN, autocommit=True) as other:
            other.execute(self.sql.SQL('SET search_path = {}, public').format(self.sql.Identifier(self.schema)))
            other.execute("SET ann.worker_auth_token = 'collection-test-token'")
            self.assertEqual(other.execute('SELECT key FROM ann_collection_search(%s, %s, 1, %s)',
                                           (self.collection, '[1,2,3,4]', 'tenant = 40')).fetchall(),
                             [(990000,)])
        self.assertIn('does not exist', self.error("SELECT ann_collection_delta('missing', 'docs_delta')"))
        self.db.execute('SELECT ann_collection_delta(%s, NULL)', (self.collection,))
        self.assertEqual(self.search('[1,2,3,4]', 10), self.exact('[1,2,3,4]', 10))

    def test_rolling_rebuild(self):
        """Load a rebuilt part under a new name, switch the catalog, unload the old one: searches stay exact."""
        parts = self.setup_collection()
        old = self.collection + '_p0'
        expected = self.exact('[1,2,3,4]', 10, 'tenant < 8')
        # A session that has the old catalog cached.
        other = self.psycopg.connect(DSN, autocommit=True)
        self.addCleanup(other.close)
        other.execute(self.sql.SQL('SET search_path = {}, public').format(self.sql.Identifier(self.schema)))
        other.execute("SET ann.worker_auth_token = 'collection-test-token'")

        def search_other():
            return [(key, round(distance, 3)) for key, distance in other.execute(
                'SELECT key, distance FROM ann_collection_search(%s, %s, 10, %s) ORDER BY distance, key',
                (self.collection, '[1,2,3,4]', 'tenant < 8')).fetchall()]

        self.assertEqual(search_other(), expected)
        rebuilt = self.build('part-0b', [os.path.join(self.directory.name, f'export-{p}') for p in (0, 1)])
        nodes = f"{self.nodes['a'].address},{self.nodes['b'].address}"
        self.db.execute('SELECT ann_collection_load(%s, 0, %s, %s, %s, 0, 7)',
                        (self.collection, old + 'b', rebuilt, nodes))
        self.assertEqual(self.search('[1,2,3,4]', 10, 'tenant < 8'), expected)
        self.assertEqual(self.value('SELECT ann_index_unload(%s, %s)', (old, nodes)), 2)
        self.assertEqual(self.value('SELECT ann_index_unload(%s, %s)', (old, nodes)), 0)
        self.assertEqual(self.search('[1,2,3,4]', 10, 'tenant < 8'), expected)
        # The cached catalog names the unloaded index: the node says stale and the session rereads it.
        self.assertEqual(search_other(), expected)
        self.assertEqual(self.search('[1,2,3,4]', 10), self.exact('[1,2,3,4]', 10))
        del parts

    def test_rows_carry_attributes(self):
        """Hits bring their attribute values from the nodes: no fetch from the shard hosts."""
        self.setup_collection()
        for query in ['[1,2,3,4]', '[-30,10,5,0]']:
            for condition in [None, 'tenant = 3', 'category = 2 AND tenant > 9', 'tenant IN (1, 12)']:
                self.assertEqual(self.rows(query, 10, condition), self.exact_rows(query, 10, condition),
                                 (query, condition))
        # Any subset, in any order.
        self.assertEqual(self.rows('[1,2,3,4]', 5, 'tenant < 8', 'category integer'),
                         self.exact_rows('[1,2,3,4]', 5, 'tenant < 8', columns='category'))
        self.assertEqual(self.rows('[1,2,3,4]', 5, None, 'category integer, tenant integer'),
                         self.exact_rows('[1,2,3,4]', 5, None, columns='category, tenant'))
        self.assertEqual(self.rows('[1,2,3,4]', 5, None, 'tenant integer')[0][:2],
                         self.search('[1,2,3,4]', 5)[0])
        # The delta's rows carry theirs too.
        self.db.execute('CREATE TABLE docs_delta (id bigint, tenant int, category int, emb vector(4))')
        self.db.execute("INSERT INTO docs_delta VALUES (990000, 3, 4, '[1,2,3,4]'), (990001, 12, 1, '[1,2,3,5]')")
        self.db.execute("SELECT ann_collection_delta(%s, 'docs_delta', 'id', 'emb')", (self.collection,))
        both = '(SELECT id, tenant, category, emb FROM docs UNION ALL SELECT * FROM docs_delta) b'
        for condition in [None, 'tenant = 3', 'tenant IN (1, 12)']:
            self.assertEqual(self.rows('[1,2,3,4]', 10, condition),
                             self.exact_rows('[1,2,3,4]', 10, condition, table=both), condition)
        # The column definition list is checked.
        self.assertIn('is not an attribute', self.error(
            'SELECT * FROM ann_collection_search_rows(%s, %s, 5) AS r(key bigint, distance real, body text)',
            (self.collection, '[1,2,3,4]')))
        self.assertIn('is integer, not bigint', self.error(
            'SELECT * FROM ann_collection_search_rows(%s, %s, 5) AS r(key bigint, distance real, tenant bigint)',
            (self.collection, '[1,2,3,4]')))
        self.assertIn('returns key bigint, distance real', self.error(
            'SELECT * FROM ann_collection_search_rows(%s, %s, 5) AS r(key integer, distance real)',
            (self.collection, '[1,2,3,4]')))

    def test_text_attributes(self):
        """Text attributes: each part has its own dictionary; the collection keeps one, maps each part's
        codes to it at load, and translates text filters (including ranges) through it."""
        self.db.execute('CREATE TABLE docs (id bigint, tenant int, category int, lang text, emb vector(4)) '
                        'PARTITION BY RANGE (id)')
        languages = [["'en'", "'fr'"], ["'en'", "'de'"], ["'zh'", "'fr'", "'ja'"], ['NULL', "'es'", "'en'"]]
        exports = []
        for p in range(4):
            self.db.execute(f'CREATE TABLE docs_{p} PARTITION OF docs FOR VALUES FROM ({p * 4000}) TO ({(p + 1) * 4000})')
            options = ', '.join(languages[p])
            self.db.execute(f"""INSERT INTO docs_{p} SELECT t * 1000 + j, t, (t * 7 + j) % 5,
                                    (ARRAY[{options}]::text[])[1 + j % {len(languages[p])}],
                                    ARRAY[sin(t * 1000 + j) * 50, cos((t * 1000 + j) * 1.3) * 40,
                                          sin((t * 1000 + j) * 0.7) * 30, j % 9]::vector
                                FROM generate_series({p * 4}, {p * 4 + 3}) t, generate_series(0, {PER_TENANT - 1}) j""")
            self.assertTrue(self.value(f"SELECT ann_freeze('docs_{p}')"))
            self.vacuum(f'docs_{p}')
            self.db.execute(f'CREATE INDEX docs_{p}_ann ON docs_{p} USING ann (emb vector_l2_ops) '
                            'INCLUDE (id, tenant, category, lang) WITH (frozen = true)')
            export = os.path.join(self.directory.name, f'export-{p}')
            self.db.execute('SELECT ann_index_export(%s, %s)', (f'docs_{p}_ann', export))
            exports.append(export)
        parts = [self.build('text-0', exports[:2]), self.build('text-1', exports[2:])]
        a, b, c = self.node('a'), self.node('b'), self.node('c')
        self.collection = self.schema + '_text'
        self.assertIn('must not be text', self.error(
            'SELECT ann_collection_create(%s, %s, %s, %s)',
            (self.collection, f'{self.schema}.docs', ['tenant', 'lang'], 'lang')))
        self.db.execute('SELECT ann_collection_create(%s, %s, %s, %s)',
                        (self.collection, f'{self.schema}.docs', ['tenant', 'category', 'lang'], 'tenant'))
        self.db.execute('SELECT ann_collection_load(%s, 0, %s, %s, %s, 0, 7)',
                        (self.collection, self.collection + '_p0', parts[0], f'{a},{b}'))
        self.db.execute('SELECT ann_collection_load(%s, 1, %s, %s, %s, 8, 15)',
                        (self.collection, self.collection + '_p1', parts[1], c))
        codes = dict(self.db.execute('SELECT value, code FROM ann_collection_dictionary WHERE collection = %s',
                                     (self.collection,)).fetchall())
        self.assertEqual(set(codes), {'en', 'fr', 'de', 'zh', 'ja', 'es'})
        self.assertEqual(sorted(codes.values()), list(range(6)))
        for query in ['[1,2,3,4]', '[-30,10,5,0]']:
            for condition in ["lang = 'fr'", "lang = 'xx'", "lang IN ('de', 'zh')", "lang < 'f'", "lang >= 'fr'",
                              "lang BETWEEN 'de' AND 'fr'", "lang <> 'en' AND tenant < 8", "lang IS NULL",
                              "NOT (lang > 'e') OR category = 3", "lang NOT IN ('en', 'fr')"]:
                self.assertEqual(self.search(query, 12, condition), self.exact(query, 12, condition),
                                 (query, condition))
                self.assertEqual(self.rows(query, 12, condition, 'lang text, tenant integer'),
                                 self.exact_rows(query, 12, condition, columns='lang, tenant'), (query, condition))
        # ann.collection_text_range_limit bounds how many strings a range may match.
        self.db.execute('SET ann.collection_text_range_limit = 2')
        self.assertIn('matches 3 strings, more than 2', self.error(
            'SELECT * FROM ann_collection_search(%s, %s, 5, %s)', (self.collection, '[1,2,3,4]', "lang < 'f'")))
        self.db.execute('SET ann.collection_text_range_limit = 3')
        self.assertEqual(self.search('[1,2,3,4]', 12, "lang < 'f'"), self.exact('[1,2,3,4]', 12, "lang < 'f'"))
        self.db.execute('RESET ann.collection_text_range_limit')
        # Reloading a part keeps its codes; a second coordinator attaching adopts the nodes'.
        self.db.execute('SELECT ann_collection_load(%s, 1, %s, NULL, %s, 8, 15)',
                        (self.collection, self.collection + '_p1', c))
        self.assertEqual(dict(self.db.execute('SELECT value, code FROM ann_collection_dictionary '
                                              'WHERE collection = %s', (self.collection,)).fetchall()), codes)
        # A node serving the part without the collection's map answers stale (3) to collection codes.
        name = 'unmapped_' + self.schema
        with self.nodes['c'].connect() as connection:
            connection.sendall(frame(11, 1, struct.pack('<I', len(name)) + name.encode() + parts[1].encode()))
            self.assertEqual(receive(connection)[2], 0)
            payload = (struct.pack('<I', len(name)) + name.encode() + struct.pack('<IIII', 4, 5, 4, 1) +
                       bytes([8]) + struct.pack('<4f', 1, 2, 3, 4))
            connection.sendall(frame(22, 2, payload))
            self.assertEqual(receive(connection)[2], 3)

    def test_unload_previous(self):
        """ann_collection_load(..., unload_previous => true) frees the replaced index on its nodes."""
        parts = self.setup_collection()
        old = self.collection + '_p0'
        nodes = f"{self.nodes['a'].address},{self.nodes['b'].address}"
        expected = self.exact('[1,2,3,4]', 10, 'tenant < 8')
        self.db.execute('SELECT ann_collection_load(%s, 0, %s, %s, %s, 0, 7, true)',
                        (self.collection, old + 'b', parts[0], nodes))
        self.assertEqual(self.search('[1,2,3,4]', 10, 'tenant < 8'), expected)
        self.assertEqual(self.value('SELECT ann_index_unload(%s, %s)', (old, nodes)), 0)  # already gone
        # Reloading under the same name keeps it.
        self.db.execute('SELECT ann_collection_load(%s, 0, %s, NULL, %s, 0, 7, true)', (self.collection, old + 'b', nodes))
        self.assertEqual(self.search('[1,2,3,4]', 10, 'tenant < 8'), expected)

    def test_tombstones(self):
        """ann_collection_delete: deleted keys leave every part's searches, and parts loaded later get them."""
        parts = self.setup_collection()
        deleted = [key for key, _ in self.exact('[1,2,3,4]', 4)] + [key for key, _ in self.exact('[1,2,3,4]', 3, 'tenant = 12')]
        # Keys the collection doesn't hold are recorded too (the catalog can't tell); nodes ignore them.
        self.assertEqual(self.value('SELECT ann_collection_delete(%s, %s)', (self.collection, deleted + [999999])), 8)
        self.assertEqual(self.value('SELECT ann_collection_delete(%s, %s)', (self.collection, deleted)), 0)

        def exact(query, k, condition=None):
            where = f'AND ({condition})' if condition else ''
            return [(key, round(distance, 3)) for key, distance in self.db.execute(
                f'SELECT id, (emb <-> %s::vector)::real AS d FROM docs WHERE id <> ALL(%s) {where} '
                'ORDER BY d, id LIMIT %s', (query, deleted, k)).fetchall()]

        for query in ['[1,2,3,4]', '[-30,10,5,0]']:
            for condition in [None, 'tenant = 12', 'category = 2', 'tenant IN (1, 12)']:
                self.assertEqual(self.search(query, 10, condition), exact(query, 10, condition), (query, condition))
        # A rebuilt part loaded later (here from the same exports, so it still holds the rows) gets them.
        nodes = f"{self.nodes['a'].address},{self.nodes['b'].address}"
        rebuilt = self.build('part-0-again', [os.path.join(self.directory.name, f'export-{p}') for p in (0, 1)])
        self.db.execute('SELECT ann_collection_load(%s, 0, %s, %s, %s, 0, 7, true)',
                        (self.collection, self.collection + '_p0v2', rebuilt, nodes))
        self.assertEqual(self.search('[1,2,3,4]', 10), exact('[1,2,3,4]', 10))
        del parts

    def test_wide_text_ranges(self):
        """A text range matching more than 65,536 strings becomes several OR'd value lists."""
        self.db.execute('CREATE TABLE words (id bigint, word text, emb vector(4))')
        self.db.execute("""INSERT INTO words SELECT i, 'w' || lpad(i::text, 6, '0'),
                               ARRAY[sin(i) * 50, cos(i * 1.3) * 40, sin(i * 0.7) * 30, i % 9]::vector
                           FROM generate_series(0, 79999) i""")
        self.assertTrue(self.value("SELECT ann_freeze('words')"))
        self.vacuum('words')
        self.db.execute('CREATE INDEX words_ann ON words USING ann (emb vector_l2_ops) '
                        'INCLUDE (id, word) WITH (frozen = true)')
        export = os.path.join(self.directory.name, 'words-export')
        self.db.execute('SELECT ann_index_export(%s, %s)', ('words_ann', export))
        part = self.build('words-part', [export])
        node = self.node('a')
        self.collection = self.schema + '_words'
        self.db.execute('SELECT ann_collection_create(%s, %s, %s)', (self.collection, f'{self.schema}.words', ['word']))
        self.db.execute('SELECT ann_collection_load(%s, 0, %s, %s, %s)', (self.collection, self.collection, part, node))
        self.db.execute('SET ann.collection_text_range_limit = 250000')
        for condition in ["word < 'w070000'", "word >= 'w000100'", "word BETWEEN 'w010000' AND 'w079000'",
                          "NOT (word < 'w075000')"]:
            expected = [(key, round(distance, 3)) for key, distance in self.db.execute(
                f"SELECT id, (emb <-> '[1,2,3,4]')::real AS d FROM words WHERE {condition} ORDER BY d, id LIMIT 10"
            ).fetchall()]
            self.assertEqual(self.search('[1,2,3,4]', 10, condition), expected, condition)
        self.db.execute('RESET ann.collection_text_range_limit')
        self.assertIn('more than 65536', self.error(
            'SELECT * FROM ann_collection_search(%s, %s, 5, %s)', (self.collection, '[1,2,3,4]', "word < 'w070000'")))

    def test_catalog_checks(self):
        self.create(partitions=2)
        self.assertIn('must be smallint, integer', self.error(
            'SELECT ann_collection_create(%s, %s, %s)', (self.schema + '_x', f'{self.schema}.docs', ['emb'])))
        self.assertIn('is not one of', self.error(
            'SELECT ann_collection_create(%s, %s, %s, %s)',
            (self.schema + '_x', f'{self.schema}.docs', ['tenant'], 'category')))
        self.assertIn('is not a column', self.error(
            'SELECT ann_collection_create(%s, %s, %s)', (self.schema + '_x', f'{self.schema}.docs', ['missing'])))


if __name__ == '__main__':
    unittest.main()
