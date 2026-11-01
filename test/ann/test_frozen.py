"""Frozen ann indexes: the freeze contract, forward and inverted pages, export to the worker.

Needs ANN_PG_DSN for a local superuser (exports write server files, and the
worker reads them). Export tests also build the CPU worker and run
scripts/ann/build_cuvs_index.py --no-cagra, which needs numpy and psycopg.
"""

import filecmp
import os
import random
import subprocess
import sys
import tempfile
import unittest
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_server import ROOT, Server  # noqa: E402

DSN = os.environ.get('ANN_PG_DSN')
BUILD = os.path.join(ROOT, 'scripts', 'ann', 'build_cuvs_index.py')
ROWS = 3000


@unittest.skipUnless(DSN, 'set ANN_PG_DSN for live PostgreSQL tests')
class FrozenTests(unittest.TestCase):
    def setUp(self):
        import psycopg
        from psycopg import sql

        self.psycopg = psycopg
        self.sql = sql
        self.db = psycopg.connect(DSN, autocommit=True)
        # VACUUM can mark pages all-visible only once their commits are flushed.
        self.db.execute('SET synchronous_commit = on')
        self.schema = 'ann_frozen_' + uuid.uuid4().hex[:12]
        self.db.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(self.schema)))
        self.db.execute(sql.SQL('SET search_path = {}, public').format(sql.Identifier(self.schema)))
        self.directory = tempfile.TemporaryDirectory()
        os.chmod(self.directory.name, 0o777)

    def tearDown(self):
        self.db.execute(self.sql.SQL('DROP SCHEMA {} CASCADE').format(self.sql.Identifier(self.schema)))
        self.db.close()
        self.directory.cleanup()

    def create(self, table='docs', rows=ROWS, freeze=True, index=True, extra=''):
        """Rows with keys 5, 10, ...; tenant 0..12; category 0..6 or NULL; a text body."""
        self.db.execute(f'CREATE TABLE {table} (id bigint, tenant int, category smallint, body text, '
                        f'emb vector(4){extra})')
        self.db.execute(f"""INSERT INTO {table} SELECT i * 5, i % 13,
                                CASE WHEN i % 11 = 0 THEN NULL ELSE (i % 7)::smallint END, 'row ' || i,
                                ARRAY[sin(i) * 50, cos(i * 1.3) * 40, sin(i * 0.7) * 30, i % 5]::vector
                            FROM generate_series({rows}, 1, -1) i""")
        if freeze:
            self.assertTrue(self.value(f"SELECT ann_freeze('{table}')"))
            self.db.execute(f'VACUUM (FREEZE) {table}')
        if index:
            self.db.execute(f'CREATE INDEX {table}_ann ON {table} USING ann (emb vector_l2_ops) '
                            'INCLUDE (id, tenant, category) WITH (frozen = true)')

    def value(self, statement, params=None):
        return self.db.execute(statement, params).fetchone()[0]

    def error(self, statement, params=None):
        with self.assertRaises(self.psycopg.Error) as caught:
            self.db.execute(statement, params)
        return str(caught.exception)

    def plan(self, statement, params=None):
        return '\n'.join(row[0] for row in self.db.execute(
            'EXPLAIN (ANALYZE, COSTS OFF, TIMING OFF, SUMMARY OFF) ' + statement, params))

    def ordered(self, columns, predicate, query, limit, index):
        statement = (f'SELECT {columns}, emb <-> %s::vector AS d FROM docs WHERE {predicate} '
                     f'ORDER BY emb <-> %s::vector, id LIMIT {limit}')
        self.db.execute('SET enable_seqscan = ' + ('off' if index else 'on'))
        for setting in ['enable_indexscan', 'enable_indexonlyscan', 'enable_bitmapscan']:
            self.db.execute(f'SET {setting} = ' + ('on' if index else 'off'))
        try:
            return self.db.execute(statement, (query, query)).fetchall(), self.plan(statement, (query, query))
        finally:
            for setting in ['enable_seqscan', 'enable_indexscan', 'enable_indexonlyscan', 'enable_bitmapscan']:
                self.db.execute(f'RESET {setting}')

    def test_contract(self):
        self.create(freeze=False, index=False)
        ann = 'CREATE INDEX ON docs USING ann (emb vector_l2_ops)'
        self.assertIn('require frozen = true', self.error(f'{ann} INCLUDE (id)'))
        self.assertIn('first INCLUDE column', self.error(f'{ann} WITH (frozen = true)'))
        self.assertIn('is not frozen', self.error(f'{ann} INCLUDE (id) WITH (frozen = true)'))
        self.assertTrue(self.value("SELECT ann_freeze('docs')"))
        self.assertFalse(self.value("SELECT ann_freeze('docs')"))
        self.assertIn('pages of table "docs" are not frozen', self.error(f'{ann} INCLUDE (id) WITH (frozen = true)'))

        # Every kind of write fails, even with triggers set to replica mode.
        for statement in ["INSERT INTO docs (id) VALUES (1)", 'UPDATE docs SET tenant = 1 WHERE false',
                          'DELETE FROM docs WHERE id = 5', 'TRUNCATE docs',
                          "COPY docs (id) FROM PROGRAM 'echo 7'"]:
            self.assertIn('is frozen for ann indexes', self.error(statement))
        self.db.execute('SET session_replication_role = replica')
        self.assertIn('is frozen for ann indexes', self.error('DELETE FROM docs'))
        self.db.execute('RESET session_replication_role')

        self.db.execute('VACUUM (FREEZE) docs')
        self.assertIn('smallint or integer', self.error(f'{ann} INCLUDE (id, body) WITH (frozen = true)'))
        self.assertIn('smallint, integer or bigint', self.error(f'{ann} INCLUDE (body) WITH (frozen = true)'))
        self.assertIn('partial', self.error(f'{ann} INCLUDE (id) WITH (frozen = true) WHERE tenant = 1'))
        self.assertIn('appears more than once', self.error(f'{ann} INCLUDE (tenant) WITH (frozen = true)'))
        self.db.execute(f'{ann} INCLUDE (id, tenant, category) WITH (frozen = true)')

        # NULL keys or vectors, and the value reserved for NULL attributes.
        for table, row in [('nullkey', "(NULL, 1, 1, '', '[1,2,3,4]')"), ('nullvec', '(1, 1, 1, NULL, NULL)'),
                           ('reserved', "(1, -2147483648, 1, '', '[1,2,3,4]')")]:
            self.db.execute(f'CREATE TABLE {table} (LIKE docs)')
            self.db.execute(f'INSERT INTO {table} VALUES {row}')
            self.db.execute(f"SELECT ann_freeze('{table}')")
            self.db.execute(f'VACUUM (FREEZE) {table}')
            message = self.error(f'CREATE INDEX ON {table} USING ann (emb vector_l2_ops) '
                                 'INCLUDE (id, tenant) WITH (frozen = true)')
            self.assertIn({'nullkey': 'NULL keys', 'nullvec': 'NULL vectors', 'reserved': 'reserve'}[table], message)
        self.db.execute('CREATE UNLOGGED TABLE scratch (LIKE docs)')
        self.db.execute("SELECT ann_freeze('scratch')")
        self.assertIn('permanent table', self.error('CREATE INDEX ON scratch USING ann (emb vector_l2_ops) '
                                                   'INCLUDE (id) WITH (frozen = true)'))

    def test_scans_match_sql(self):
        self.create()
        rng = random.Random(3)
        for _ in range(8):
            query = str([rng.uniform(-50, 50), rng.uniform(-40, 40), rng.uniform(-30, 30), rng.uniform(0, 5)])
            for columns, predicate in [('id, tenant, category', 'true'), ('id, category', 'tenant = 4'),
                                       ('id, body', 'category IS NULL')]:
                actual, plan = self.ordered(columns, predicate, query, 25, True)
                expected, _ = self.ordered(columns, predicate, query, 25, False)
                self.assertEqual(actual, expected)
                self.assertIn('docs_ann', plan)
                if 'body' in columns:
                    self.assertIn('Index Scan using docs_ann', plan)
                else:
                    self.assertIn('Index Only Scan using docs_ann', plan)
                    self.assertIn('Heap Fetches: 0', plan)
        # Every row comes back once, nearest first.
        rows, _ = self.ordered('id', 'true', '[1,2,3,4]', ROWS + 10, True)
        self.assertEqual(sorted(row[0] for row in rows), list(range(5, 5 * ROWS + 1, 5)))

    def test_postings_match_sql(self):
        self.create()
        for column, values in [('tenant', [4]), ('tenant', [0, 12, 4, 4, None]), ('category', [6, 0]),
                               ('category', [99]), ('tenant', []), ('category', [None])]:
            actual = [row[0] for row in self.db.execute('SELECT ann_frozen_keys(%s, %s, %s)',
                                                        ('docs_ann', column, values))]
            expected = [row[0] for row in self.db.execute(
                f'SELECT id FROM docs WHERE {column} = ANY(%s) ORDER BY id', (values,))]
            self.assertEqual(actual, expected, (column, values))
        self.assertIn('no INCLUDE column', self.error("SELECT ann_frozen_keys('docs_ann', 'id', ARRAY[5])"))
        self.db.execute('CREATE TABLE mutable (emb vector(4))')
        self.db.execute('CREATE INDEX mutable_ann ON mutable USING ann (emb vector_l2_ops)')
        self.assertIn('not a frozen ann index', self.error("SELECT ann_frozen_info('mutable_ann')"))
        self.assertIn('is not an index', self.error("SELECT ann_frozen_info('docs')"))
        self.db.execute('CREATE INDEX docs_id ON docs (id)')
        self.assertIn('not an ann index', self.error("SELECT ann_frozen_info('docs_id')"))

    def test_generation(self):
        self.create()
        info = self.db.execute('SELECT * FROM ann_frozen_info(%s)', ('docs_ann',)).fetchone()
        self.assertEqual(info[1:6], (ROWS, 4, 'id', ['tenant', 'category'], [13, 8]))
        generation = info[0]
        self.db.execute('REINDEX INDEX docs_ann')
        self.assertEqual(self.value("SELECT generation FROM ann_frozen_info('docs_ann')"), generation)
        # The same rows in another table, loaded in another order, hash the same.
        self.db.execute('CREATE TABLE copy AS SELECT * FROM docs ORDER BY random()')
        self.db.execute("SELECT ann_freeze('copy')")
        self.db.execute('VACUUM (FREEZE) copy')
        self.db.execute('CREATE INDEX copy_ann ON copy USING ann (emb vector_l2_ops) '
                        'INCLUDE (id, tenant, category) WITH (frozen = true)')
        self.assertEqual(self.value("SELECT generation FROM ann_frozen_info('copy_ann')"), generation)
        # Changing one value changes the generation.
        self.db.execute('DROP INDEX copy_ann')
        self.db.execute('DROP TRIGGER ann_frozen_guard ON copy')
        self.db.execute('UPDATE copy SET category = 6 WHERE id = 5')
        self.db.execute("SELECT ann_freeze('copy')")
        self.db.execute('VACUUM (FREEZE) copy')
        self.db.execute('CREATE INDEX copy_ann ON copy USING ann (emb vector_l2_ops) '
                        'INCLUDE (id, tenant, category) WITH (frozen = true)')
        self.assertNotEqual(self.value("SELECT generation FROM ann_frozen_info('copy_ann')"), generation)

    def test_vacuum_without_guard(self):
        self.create()
        self.db.execute('DROP TRIGGER ann_frozen_guard ON docs')
        self.assertIn('cannot insert into frozen ann index', self.error(
            "INSERT INTO docs VALUES (1, 1, 1, '', '[1,2,3,4]')"))
        self.db.execute('DELETE FROM docs WHERE id % 3 = 0')
        self.db.execute('VACUUM docs')
        for query in ['[1,2,3,4]', '[50,40,30,2]']:
            actual, _ = self.ordered('id, tenant', 'true', query, 40, True)
            expected, _ = self.ordered('id, tenant', 'true', query, 40, False)
            self.assertEqual(actual, expected)
        self.assertEqual(self.value("SELECT rows FROM ann_frozen_info('docs_ann')"), ROWS)

    def export(self, name):
        path = os.path.join(self.directory.name, name)
        self.assertEqual(self.value('SELECT ann_index_export(%s, %s)', ('docs_ann', path)), ROWS)
        return path

    def test_export_matches_builder(self):
        self.create()
        path = self.export('export')
        built = os.path.join(self.directory.name, 'built')
        subprocess.run([sys.executable, BUILD, '--dsn', DSN, '--table', f'{self.schema}.docs', '--key', 'id',
                        '--vector', 'emb', '--out', built, '--no-cagra', '--attribute', 'tenant',
                        '--attribute', 'category'], check=True, capture_output=True)
        for name in ['keys.i64', 'vectors.f32', 'attr-tenant.i32', 'attr-category.i32']:
            self.assertTrue(filecmp.cmp(os.path.join(path, name), os.path.join(built, name), shallow=False), name)
        with open(os.path.join(path, 'manifest')) as manifest:
            lines = manifest.read().splitlines()
        generation = self.value("SELECT generation FROM ann_frozen_info('docs_ann')")
        self.assertIn(f'generation {generation}', lines)
        self.assertIn(f'source {self.schema}.docs', lines)
        self.assertIn('attribute category attr-category.i32', lines)
        self.assertIn('is not empty', self.error('SELECT ann_index_export(%s, %s)', ('docs_ann', path)))
        self.assertIn('absolute path', self.error("SELECT ann_index_export('docs_ann', 'relative')"))

    def test_export_build_and_load(self):
        subprocess.run(['make', '-C', os.path.join(ROOT, 'ann'), 'ann-worker'], check=True,
                       stdout=subprocess.DEVNULL)
        server = Server('--backend', 'cpu-exact')
        try:
            self.create()
            export = self.export('export')
            path = os.path.join(self.directory.name, 'index')
            subprocess.run([sys.executable, BUILD, '--from-export', export, '--out', path, '--no-cagra'],
                           check=True, capture_output=True)
            with open(os.path.join(path, 'manifest')) as manifest:
                text = manifest.read()
            generation = self.value("SELECT generation FROM ann_frozen_info('docs_ann')")
            self.assertIn(f'generation {generation}\n', text)
            self.db.execute('SET ann.worker_socket = %s' % self.sql.Literal(server.path).as_string(self.db))
            name = 'frozen_' + uuid.uuid4().hex[:8]
            self.assertEqual(self.value('SELECT ann_index_load(%s, %s)', (name, path)), ROWS)
            mismatches = self.value(f"""
                SELECT count(*) FILTER (WHERE a IS DISTINCT FROM b) FROM (
                    SELECT ARRAY(SELECT s.id FROM ann_attribute_search('{name}', q.emb, 7, 'tenant',
                                 ARRAY[q.tenant]) s ORDER BY s.distance, s.id) a,
                           ARRAY(SELECT t.id FROM ann_frozen_keys('docs_ann', 'tenant', ARRAY[q.tenant]) k
                                 JOIN docs t ON t.id = k ORDER BY t.emb <-> q.emb, t.id LIMIT 7) b
                    FROM docs q WHERE q.id % 170 = 0) x""")
            self.assertEqual(mismatches, 0)

            # After the table changes and is frozen again, the old files no longer load.
            self.db.execute('DROP INDEX docs_ann')
            self.db.execute('DROP TRIGGER ann_frozen_guard ON docs')
            self.db.execute("UPDATE docs SET emb = '[0,0,0,1]' WHERE id = 10")
            self.db.execute("SELECT ann_freeze('docs')")
            self.db.execute('VACUUM (FREEZE) docs')
            self.db.execute('CREATE INDEX docs_ann ON docs USING ann (emb vector_l2_ops) '
                            'INCLUDE (id, tenant, category) WITH (frozen = true)')
            message = self.error('SELECT ann_index_load(%s, %s)', ('stale_' + name[7:], path))
            self.assertIn(f'generation {generation} matches no frozen ann index', message)
        finally:
            server.close()



@unittest.skipUnless(os.environ.get('ANN_CUVS_TEST') and DSN, 'set ANN_CUVS_TEST=1 and ANN_PG_DSN on a cuVS host')
class FrozenCuvsTests(unittest.TestCase):
    """A frozen index exported, built into CAGRA shards and an NVFP4 copy, and served by ann-worker-cuvs."""

    def test_export_cagra_and_search(self):
        import psycopg
        from psycopg import sql

        directory = tempfile.TemporaryDirectory()
        os.chmod(directory.name, 0o777)
        server = Server('--backend', 'cuvs', '--lanes', '4', '--exact-limit', '200',
                        executable=os.path.join(ROOT, 'ann', 'ann-worker-cuvs'))
        db = psycopg.connect(DSN, autocommit=True)
        table = 'ann_frozen_cuvs_' + uuid.uuid4().hex[:8]
        try:
            db.execute('SET synchronous_commit = on')
            db.execute(f'CREATE TABLE {table} (id bigint, tenant int, emb vector(32))')
            db.execute('SELECT setseed(0.3)')
            db.execute(f"""INSERT INTO {table} SELECT i * 3, i % 40,
                               (SELECT array_agg(random()::real) FROM generate_series(1, 32) WHERE i > 0)::vector
                           FROM generate_series(1, 30000) i""")
            db.execute(f"SELECT ann_freeze('{table}')")
            db.execute(f'VACUUM (FREEZE) {table}')
            db.execute(f'CREATE INDEX {table}_ann ON {table} USING ann (emb vector_l2_ops) '
                       'INCLUDE (id, tenant) WITH (frozen = true)')
            export = os.path.join(directory.name, 'export')
            db.execute('SELECT ann_index_export(%s, %s)', (f'{table}_ann', export))
            path = os.path.join(directory.name, 'index')
            subprocess.run([sys.executable, BUILD, '--from-export', export, '--out', path, '--shards', '2',
                            '--quantize', 'nvfp4'], check=True, capture_output=True)
            db.execute('SET ann.worker_socket = %s' % sql.Literal(server.path).as_string(db))
            db.execute('SET ann.gpu_min_eligible = 0')
            self.assertEqual(db.execute('SELECT ann_index_load(%s, %s)', (table, path)).fetchone()[0], 30000)
            mismatches, recall = db.execute(f"""
                SELECT count(*) FILTER (WHERE f IS DISTINCT FROM e),
                       avg(cardinality(ARRAY(SELECT unnest(a) INTERSECT SELECT unnest(b)))) / 10 FROM (
                    SELECT ARRAY(SELECT s.id FROM ann_attribute_search('{table}', q.emb, 10, 'tenant',
                                 ARRAY[q.tenant]) s ORDER BY s.distance, s.id) f,
                           ARRAY(SELECT t.id FROM ann_frozen_keys('{table}_ann', 'tenant', ARRAY[q.tenant]) k
                                 JOIN {table} t ON t.id = k ORDER BY t.emb <-> q.emb, t.id LIMIT 10) e,
                           ARRAY(SELECT s.id FROM ann_index_search('{table}', q.emb, 10) s) a,
                           ARRAY(SELECT t.id FROM {table} t ORDER BY t.emb <-> q.emb LIMIT 10) b
                    FROM {table} q WHERE q.id % 900 = 0) x""").fetchone()
            self.assertEqual(mismatches, 0)
            self.assertGreaterEqual(recall, 0.9)
        finally:
            db.execute(f'DROP TABLE IF EXISTS {table}')
            db.close()
            server.close()
            directory.cleanup()


if __name__ == '__main__':
    unittest.main()
