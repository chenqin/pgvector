"""Frozen ann indexes: the freeze contract, forward and inverted pages, export to the worker.

Needs ANN_PG_DSN for a local superuser (exports write server files, and the
worker reads them). Export tests also build the CPU worker and run
scripts/ann/build_cuvs_index.py --no-cagra, which needs numpy and psycopg.
"""

import filecmp
import json
import os
import random
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_server import ROOT, STATS, Server, frame, receive  # noqa: E402

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
        # The tables are tiny: without a free call, a sort over them is cheaper than the route.
        self.db.execute('SET ann.frozen_setup_cost = 0')
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
            self.vacuum(table)
        if index:
            self.db.execute(f'CREATE INDEX {table}_ann ON {table} USING ann (emb vector_l2_ops) '
                            'INCLUDE (id, tenant, category) WITH (frozen = true)')

    def vacuum(self, table):
        """VACUUM (FREEZE) until every page is all-visible: pages other sessions hold stay unfrozen."""
        for _ in range(5):
            self.db.execute(f'VACUUM (FREEZE) {table}')
            if self.value(f"SELECT relallvisible = relpages FROM pg_class WHERE oid = '{table}'::regclass"):
                return

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
        # The access method's own scans; the frozen route is tested separately.
        self.db.execute('SET ann.enable_frozen_search = off')
        self.db.execute('SET enable_seqscan = ' + ('off' if index else 'on'))
        for setting in ['enable_indexscan', 'enable_indexonlyscan', 'enable_bitmapscan']:
            self.db.execute(f'SET {setting} = ' + ('on' if index else 'off'))
        try:
            return self.db.execute(statement, (query, query)).fetchall(), self.plan(statement, (query, query))
        finally:
            for setting in ['enable_seqscan', 'enable_indexscan', 'enable_indexonlyscan', 'enable_bitmapscan',
                            'ann.enable_frozen_search']:
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
        self.assertIn('date, text or varchar', self.error(f'{ann} INCLUDE (id, emb) WITH (frozen = true)'))
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
            self.assertIn({'nullkey': 'NULL keys', 'nullvec': 'NULL vectors', 'reserved': 'out of range'}[table], message)
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

    FILTERS = ['tenant = 3', 'tenant <> 3', 'category IS NULL', 'category IS NOT NULL', 'tenant IN (1, 2, 99)',
               'tenant NOT IN (1, 2)', 'tenant NOT IN (1, NULL)', 'tenant = ANY(ARRAY[]::int[])',
               'tenant <> ALL(ARRAY[]::int[])', 'tenant BETWEEN 3 AND 7 AND category > 2',
               'NOT (tenant < 5 OR category = 1)', 'tenant > 100000000000', 'category < 3 OR tenant = 12',
               'NOT (category = 2)', 'NOT (category IN (1, 2) AND tenant >= 6)', '6 > category',
               'tenant >= -5 AND tenant < 2::bigint', 'category = 2::smallint OR category IS NULL', 'true', 'false']

    def bitmap_rows(self, bitmap):
        """Keys of the rows a bitmap over ordinals selects (keys are 5, 10, ...)."""
        return [(i + 1) * 5 for i in range(len(bitmap) * 8) if bitmap[i // 8] >> (i % 8) & 1]

    def assert_filter(self, index, predicate):
        eligible, bitmap = self.db.execute('SELECT eligible, bitmap FROM ann_frozen_filter(%s, %s)',
                                           (index, predicate)).fetchone()
        expected = [row[0] for row in self.db.execute(f'SELECT id FROM docs WHERE {predicate} ORDER BY id')]
        self.assertEqual(self.bitmap_rows(bytes(bitmap)), expected, predicate)
        self.assertEqual(eligible, len(expected), predicate)
        return bytes(bitmap)

    def test_predicates_match_sql(self):
        self.create()
        self.db.execute('CREATE INDEX docs_bits ON docs USING ann (emb vector_l2_ops) '
                        'INCLUDE (id, tenant, category) WITH (frozen = true, bitset_min = 0)')
        for predicate in self.FILTERS:
            self.assertEqual(self.assert_filter('docs_ann', predicate), self.assert_filter('docs_bits', predicate))
        # Pages read through shared buffers instead of the file mappings agree.
        self.db.execute('SET ann.frozen_mmap = off')
        for predicate in self.FILTERS[:8]:
            self.assert_filter('docs_ann', predicate)
        self.db.execute('RESET ann.frozen_mmap')
        for predicate in ['body = $$x$$', 'tenant = random()::int', 'tenant + 1 = 3']:
            self.assertIn('not a predicate over the INCLUDE attributes',
                          self.error('SELECT * FROM ann_frozen_filter(%s, %s)', ('docs_ann', predicate)))

    def route(self, predicate, query, limit, route=True):
        statement = (f'SELECT id, tenant, emb <-> %s::vector AS d FROM docs WHERE {predicate} '
                     f'ORDER BY emb <-> %s::vector LIMIT {limit}')
        self.db.execute('SET ann.enable_frozen_search = ' + ('on' if route else 'off'))
        self.db.execute('SET enable_indexscan = ' + ('on' if route else 'off'))
        self.db.execute('SET enable_indexonlyscan = ' + ('on' if route else 'off'))
        try:
            rows = self.db.execute(statement, (query, query)).fetchall()
            plan = self.plan(statement, (query, query))
        finally:
            for setting in ['ann.enable_frozen_search', 'enable_indexscan', 'enable_indexonlyscan']:
                self.db.execute(f'RESET {setting}')
        return rows, plan

    def test_frozen_route_matches_sql(self):
        self.create()
        rng = random.Random(5)
        for predicate in self.FILTERS:
            query = str([rng.uniform(-50, 50), rng.uniform(-40, 40), rng.uniform(-30, 30), rng.uniform(0, 5)])
            actual, plan = self.route(predicate, query, 25)
            expected, _ = self.route(predicate, query, 25, route=False)
            self.assertEqual(actual, expected, predicate)
            if predicate not in ('true', 'false'):
                self.assertIn('AnnFrozenSearch', plan, predicate)
                self.assertIn('Heap Fetches: 0', plan, predicate)
        # Columns outside the index come from the heap.
        rows = self.db.execute("SELECT id, body FROM docs WHERE tenant = 4 ORDER BY emb <-> '[1,2,3,4]' "
                               'LIMIT 5').fetchall()
        self.assertEqual([row[1] for row in rows], [f'row {row[0] // 5}' for row in rows])
        # A parameterized LIMIT starts from 100 rows and refills.
        results = []
        for setting in ['on', 'off']:
            self.db.execute(f'SET ann.enable_frozen_search = {setting}')
            self.db.execute('SET plan_cache_mode = force_generic_plan')
            self.db.execute('PREPARE nearest(int, vector, int) AS SELECT id FROM docs WHERE tenant <> $1 '
                            'ORDER BY emb <-> $2 LIMIT $3')
            results.append([row[0] for row in self.db.execute("EXECUTE nearest(3, '[5,5,5,1]', 700)")])
            self.db.execute('DEALLOCATE nearest')
        self.assertEqual(len(results[0]), 700)
        self.assertEqual(results[0], results[1])
        self.db.execute('RESET ann.enable_frozen_search')
        # Unsupported predicates keep ordinary plans.
        plan = self.plan("SELECT id FROM docs WHERE body = 'row 7' ORDER BY emb <-> '[1,2,3,4]' LIMIT 3")
        self.assertNotIn('AnnFrozenSearch', plan)

    def test_partition_base_and_delta(self):
        self.create(table='base', index=False)
        self.db.execute('CREATE TABLE parts (id bigint, tenant int, category smallint, body text, emb vector(4)) '
                        'PARTITION BY RANGE (id)')
        self.db.execute('CREATE TABLE delta PARTITION OF parts DEFAULT')
        self.db.execute(f'ALTER TABLE parts ATTACH PARTITION base FOR VALUES FROM (0) TO ({5 * ROWS + 1})')
        self.db.execute('CREATE INDEX base_ann ON base USING ann (emb vector_l2_ops) '
                        'INCLUDE (id, tenant, category) WITH (frozen = true)')
        # New rows land in the delta; the frozen base refuses them.
        self.db.execute(f"""INSERT INTO parts SELECT {5 * ROWS} + i, i % 13, (i % 7)::smallint, 'new ' || i,
                               ARRAY[cos(i) * 50, sin(i * 1.1) * 40, cos(i * 0.3) * 30, i % 5]::vector
                           FROM generate_series(1, 400) i""")
        self.assertIn('is frozen for ann indexes',
                      self.error("INSERT INTO parts VALUES (1, 1, 1, '', '[1,2,3,4]')"))
        for statement in ["DELETE FROM parts WHERE id = 5",
                          "UPDATE parts SET body = 'changed' WHERE id = 5"]:
            self.assertIn('is frozen for ann indexes', self.error(statement))
        self.db.execute('SET session_replication_role = replica')
        try:
            self.assertIn('is frozen for ann indexes', self.error('DELETE FROM parts WHERE id = 5'))
        finally:
            self.db.execute('RESET session_replication_role')
        for predicate in ['tenant = 3', 'category IN (1, 2) AND tenant < 9', 'true']:
            statement = f"SELECT id FROM parts WHERE {predicate} ORDER BY emb <-> '[3,1,4,1]' LIMIT 30"
            plan = self.plan(statement)
            self.assertIn('AnnFrozenSearch', plan)
            self.assertIn('Merge Append', plan)
            actual = [row[0] for row in self.db.execute(statement)]
            self.db.execute('SET ann.enable_frozen_search = off')
            expected = [row[0] for row in self.db.execute(statement)]
            self.db.execute('RESET ann.enable_frozen_search')
            self.assertEqual(actual, expected, predicate)

        # Publish the next generation: a new base holding base and delta, swapped in one transaction.
        self.db.execute('CREATE TABLE base2 AS SELECT * FROM parts')
        self.db.execute("SELECT ann_freeze('base2')")
        self.vacuum('base2')
        self.db.execute('CREATE INDEX base2_ann ON base2 USING ann (emb vector_l2_ops) '
                        'INCLUDE (id, tenant, category) WITH (frozen = true)')
        statement = "SELECT id FROM parts WHERE tenant = 3 ORDER BY emb <-> '[3,1,4,1]' LIMIT 30"
        before = [row[0] for row in self.db.execute(statement)]
        with self.db.transaction():
            self.db.execute('ALTER TABLE parts DETACH PARTITION base')
            self.db.execute('ALTER TABLE parts DETACH PARTITION delta')
            self.db.execute('ALTER TABLE parts ATTACH PARTITION base2 FOR VALUES FROM (0) TO (100000000)')
            self.db.execute('CREATE TABLE delta2 PARTITION OF parts DEFAULT')
        self.assertIn('AnnFrozenSearch) on base2', self.plan(statement))
        self.assertEqual([row[0] for row in self.db.execute(statement)], before)
        self.db.execute('DROP TABLE base, delta')
        self.assertEqual(self.value("SELECT count(*) FROM pg_trigger t JOIN pg_class c ON c.oid = t.tgrelid "
                                    "WHERE c.relname = 'base' AND c.relnamespace = %s::regnamespace",
                                    (self.schema,)), 0)

    def test_bool_date_bigint_attributes(self):
        self.db.execute('CREATE TABLE typed (id bigint, flag boolean, day date, big bigint, emb vector(4))')
        self.db.execute("""INSERT INTO typed SELECT i * 5, CASE WHEN i % 9 = 0 THEN NULL ELSE i % 3 = 0 END,
                               CASE WHEN i % 13 = 0 THEN NULL ELSE date '2024-01-01' + (i % 90) END,
                               CASE WHEN i % 17 = 0 THEN NULL ELSE i % 11 END,
                               ARRAY[sin(i) * 50, cos(i * 1.3) * 40, sin(i * 0.7) * 30, i % 5]::vector
                           FROM generate_series(1, 2000) i""")
        self.db.execute("SELECT ann_freeze('typed')")
        self.vacuum('typed')
        self.db.execute('CREATE INDEX typed_ann ON typed USING ann (emb vector_l2_ops) '
                        'INCLUDE (id, flag, day, big) WITH (frozen = true)')
        for predicate in ['flag', 'NOT flag', 'flag IS NOT TRUE', 'flag IS UNKNOWN', 'flag IS NOT FALSE',
                          'flag = false', "day >= '2024-03-01'", "day <> '2024-01-05'",
                          "day BETWEEN '2024-01-10' AND '2024-02-01' AND flag", 'big IN (5, 7)',
                          'big > 1000', 'NOT (big < 4 OR flag)', 'day IS NULL OR big = 0']:
            eligible, bitmap = self.db.execute('SELECT eligible, bitmap FROM ann_frozen_filter(%s, %s)',
                                               ('typed_ann', predicate)).fetchone()
            expected = [row[0] for row in self.db.execute(f'SELECT id FROM typed WHERE {predicate} ORDER BY id')]
            self.assertEqual(self.bitmap_rows(bytes(bitmap)), expected, predicate)
            self.assertEqual(eligible, len(expected), predicate)
            statement = f"SELECT id, flag, day, big FROM typed WHERE {predicate} ORDER BY emb <-> '[1,2,3,4]' LIMIT 15"
            plan = self.plan(statement)
            self.assertIn('AnnFrozenSearch', plan, predicate)
            actual = self.db.execute(statement).fetchall()
            self.db.execute('SET ann.enable_frozen_search = off')
            self.assertEqual(actual, self.db.execute(statement).fetchall(), predicate)
            self.db.execute('RESET ann.enable_frozen_search')
        self.db.execute('CREATE TABLE wide (id bigint, big bigint, emb vector(4))')
        self.db.execute("INSERT INTO wide VALUES (1, 5000000000, '[1,2,3,4]')")
        self.db.execute("SELECT ann_freeze('wide')")
        self.vacuum('wide')
        self.assertIn('out of range', self.error('CREATE INDEX ON wide USING ann (emb vector_l2_ops) '
                                                 'INCLUDE (id, big) WITH (frozen = true)'))

    def test_text_attributes(self):
        self.db.execute('CREATE TABLE named (id bigint, label text, code varchar(8) COLLATE "C", emb vector(4))')
        self.db.execute("""INSERT INTO named SELECT i * 5,
                               CASE WHEN i % 10 = 0 THEN NULL ELSE (ARRAY['apple', 'Banana', 'cherry', 'date',
                                   'Éclair', 'fig', 'grape'])[1 + i % 7] END,
                               CASE WHEN i % 15 = 0 THEN NULL ELSE 'c' || (i % 23) END,
                               ARRAY[sin(i) * 50, cos(i * 1.3) * 40, sin(i * 0.7) * 30, i % 5]::vector
                           FROM generate_series(1, 2000) i""")
        self.db.execute("SELECT ann_freeze('named')")
        self.vacuum('named')
        self.db.execute('CREATE INDEX named_ann ON named USING ann (emb vector_l2_ops) '
                        'INCLUDE (id, label, code) WITH (frozen = true)')
        self.assertEqual(self.value("SELECT distinct_values FROM ann_frozen_info('named_ann')"), [8, 24])
        for predicate in ["label = 'cherry'", "label <> 'fig'", "label = 'missing'", "label <> 'missing'",
                          "label IN ('apple', 'grape', 'zzz')", "label NOT IN ('apple', 'Banana')",
                          "label < 'cherry'", "label >= 'date'", "label BETWEEN 'b' AND 'e'", 'label IS NULL',
                          "code = 'c7'", "code > 'c19' AND label IS NOT NULL", "NOT (code < 'c3' OR label = 'fig')",
                          "code IN ('c1', 'c22', 'c99')"]:
            eligible, bitmap = self.db.execute('SELECT eligible, bitmap FROM ann_frozen_filter(%s, %s)',
                                               ('named_ann', predicate)).fetchone()
            expected = [row[0] for row in self.db.execute(f'SELECT id FROM named WHERE {predicate} ORDER BY id')]
            self.assertEqual(self.bitmap_rows(bytes(bitmap)), expected, predicate)
            self.assertEqual(eligible, len(expected), predicate)
            statement = f"SELECT id, label, code FROM named WHERE {predicate} ORDER BY emb <-> '[1,2,3,4]' LIMIT 12"
            self.assertIn('AnnFrozenSearch', self.plan(statement), predicate)
            actual = self.db.execute(statement).fetchall()
            self.db.execute('SET ann.enable_frozen_search = off')
            self.assertEqual(actual, self.db.execute(statement).fetchall(), predicate)
            self.db.execute('RESET ann.enable_frozen_search')
        # Another collation orders strings differently: not translated.
        plan = self.plan("SELECT id FROM named WHERE label < 'cherry' COLLATE \"C\" ORDER BY emb <-> '[1,2,3,4]' LIMIT 3")
        self.assertNotIn('AnnFrozenSearch', plan)

    def test_renames_rescans_and_limits(self):
        self.create()
        statement = "SELECT id FROM docs WHERE {} ORDER BY emb <-> '[3,1,4,1]' LIMIT 20"
        before = [row[0] for row in self.db.execute(statement.format('tenant = 4 AND category < 3'))]
        # Swapping two columns' names: programs use the names of the build.
        self.db.execute('ALTER TABLE docs RENAME COLUMN tenant TO swap')
        self.db.execute('ALTER TABLE docs RENAME COLUMN category TO tenant')
        self.db.execute('ALTER TABLE docs RENAME COLUMN swap TO category')
        renamed = statement.format('category = 4 AND tenant < 3')
        self.assertIn('AnnFrozenSearch', self.plan(renamed))
        self.assertEqual([row[0] for row in self.db.execute(renamed)], before)
        self.db.execute('SET ann.enable_frozen_search = off')
        self.assertEqual([row[0] for row in self.db.execute(renamed)], before)
        self.db.execute('RESET ann.enable_frozen_search')
        # A lateral join rescans the route once per outer row.
        lateral = ("SELECT q.t, x.id FROM generate_series(0, 12) q(t) CROSS JOIN LATERAL "
                   "(SELECT id FROM docs WHERE category = q.t ORDER BY emb <-> '[1,1,1,1]' LIMIT 3) x ORDER BY 1, 2")
        actual = self.db.execute(lateral).fetchall()
        self.db.execute('SET ann.enable_frozen_search = off')
        self.assertEqual(actual, self.db.execute(lateral).fetchall())
        self.db.execute('RESET ann.enable_frozen_search')
        # A predicate too large for a program keeps ordinary plans.
        huge = ' OR '.join(f'category = {i}' for i in range(600))
        self.assertNotIn('AnnFrozenSearch', self.plan(statement.format(huge)))
        self.assertEqual(len(self.db.execute(statement.format(huge)).fetchall()), 20)

    def cache_stats(self, db=None):
        return (db or self.db).execute('SELECT entries, bytes, hits, misses, evictions FROM ann_frozen_cache()').fetchone()

    def test_shared_filter_cache(self):
        self.create()
        predicate = 'tenant IN (2, 7) AND category <> 4'
        before = self.cache_stats()
        first = self.db.execute('SELECT eligible, bitmap FROM ann_frozen_filter(%s, %s)', ('docs_ann', predicate)).fetchone()
        middle = self.cache_stats()
        self.assertGreaterEqual(middle[3], before[3] + 1)
        self.assertGreaterEqual(middle[0] + middle[4], before[0] + before[4] + 1)
        # Another backend finds the rows cached.
        other = self.psycopg.connect(DSN, autocommit=True)
        try:
            other.execute(f'SET search_path = {self.schema}, public')
            second = other.execute('SELECT eligible, bitmap FROM ann_frozen_filter(%s, %s)',
                                   ('docs_ann', predicate)).fetchone()
            self.assertGreaterEqual(self.cache_stats(other)[2], middle[2] + 1)
        finally:
            other.close()
        self.assertEqual((second[0], bytes(second[1])), (first[0], bytes(first[1])))
        self.assert_filter('docs_ann', predicate)
        # A conjunction is one program however its clauses are ordered.
        hits = self.cache_stats()[2]
        swapped = self.db.execute('SELECT eligible, bitmap FROM ann_frozen_filter(%s, %s)',
                                  ('docs_ann', 'category <> 4 AND tenant IN (2, 7)')).fetchone()
        self.assertEqual((swapped[0], bytes(swapped[1])), (first[0], bytes(first[1])))
        self.assertEqual(self.cache_stats()[2], hits + 1)
        # The backend-local copy is independent of shared-cache reclamation.
        self.db.execute('SET ann.frozen_filter_local_cache_mb = 0')
        self.assert_filter('docs_ann', predicate)
        self.db.execute('SET ann.frozen_filter_local_cache_mb = 1')
        for _ in range(3):
            self.assert_filter('docs_ann', predicate)
        self.db.execute('SET ann.frozen_filter_cache_mb = 0')
        self.assert_filter('docs_ann', predicate)
        self.db.execute('RESET ann.frozen_filter_cache_mb')
        # A sparse result (under 1/64 of the rows) is kept as ordinals and rebuilt on a hit.
        sparse = 'tenant = 7 AND category = 3'
        hits = self.cache_stats()[2]
        for _ in range(2):
            self.assert_filter('docs_ann', sparse)
        self.assertGreaterEqual(self.cache_stats()[2], hits + 1)
        # A tiny budget evicts, and results stay exact.
        self.db.execute("SET ann.frozen_filter_cache_mb = 1")
        self.db.execute("""SELECT count(*) FROM generate_series(1, 2500) g,
                               ann_frozen_filter('docs_ann', 'tenant >= ' || (g % 13) || ' AND category <> ' || g) f""")
        stats = self.cache_stats()
        self.assertGreater(stats[4], middle[4])
        self.assertLessEqual(stats[1], 1024 * 1024)
        # Concurrent backends stay within the budget while one of them evicts.
        errors = []

        def fill(offset):
            try:
                db = self.psycopg.connect(DSN, autocommit=True)
                try:
                    db.execute(f'SET search_path = {self.schema}, public')
                    db.execute('SET ann.frozen_filter_cache_mb = 1')
                    db.execute("""SELECT count(*) FROM generate_series(1, 1500) g,
                                      ann_frozen_filter('docs_ann', 'tenant >= ' || (g %% 11) ||
                                                        ' AND category <> ' || (g + %s)) f""", (offset,))
                finally:
                    db.close()
            except Exception as e:  # reported below
                errors.append(e)
        threads = [threading.Thread(target=fill, args=(100000 * (i + 1),)) for i in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        stats = self.cache_stats()
        self.assertLessEqual(stats[1], 1024 * 1024)
        for predicate in self.FILTERS[:6]:
            self.assert_filter('docs_ann', predicate)
        # The route's page evaluation goes through the cache too.
        actual, plan = self.route('tenant = 9', '[1,2,3,4]', 10)
        expected, _ = self.route('tenant = 9', '[1,2,3,4]', 10, route=False)
        self.assertEqual(actual, expected)
        self.db.execute('RESET ann.frozen_filter_cache_mb')

    def test_prewarm_and_precompute(self):
        self.create()
        size = self.value("SELECT pg_relation_size('docs_ann')")
        self.assertEqual(self.value("SELECT ann_frozen_prewarm('docs_ann')"), size - 8192)
        self.db.execute('SET ann.frozen_mmap = off')
        self.assertEqual(self.value("SELECT ann_frozen_prewarm('docs_ann')"), size - 8192)
        self.db.execute('RESET ann.frozen_mmap')
        filters = ['tenant = 11 AND category = 3', 'category BETWEEN 2 AND 4', 'NOT (tenant < 6)']
        self.assertEqual(self.value('SELECT ann_frozen_precompute(%s, %s)', ('docs_ann', filters)), 3)
        before = self.cache_stats()
        for predicate in filters:
            self.assert_filter('docs_ann', predicate)
        self.assertGreaterEqual(self.cache_stats()[2], before[2] + 3)
        self.assertIn('is not a predicate over the INCLUDE attributes',
                      self.error('SELECT ann_frozen_precompute(%s, %s)', ('docs_ann', ['body = $$x$$'])))
        role = 'ann_frozen_nobody_' + uuid.uuid4().hex[:6]
        self.db.execute(f'CREATE ROLE {role}')
        self.db.execute(f'GRANT USAGE ON SCHEMA {self.schema} TO {role}')
        try:
            self.db.execute(f'SET ROLE {role}')
            for statement in ["SELECT ann_frozen_prewarm('docs_ann')",
                              "SELECT ann_frozen_precompute('docs_ann', ARRAY['tenant = 1'])"]:
                self.assertIn('permission denied', self.error(statement))
        finally:
            self.db.execute('RESET ROLE')
            self.db.execute(f'DROP OWNED BY {role}')
            self.db.execute(f'DROP ROLE {role}')

    def test_precompute_on_the_worker(self):
        subprocess.run(['make', '-C', os.path.join(ROOT, 'ann'), 'ann-worker'], check=True,
                       stdout=subprocess.DEVNULL)
        server = Server('--backend', 'cpu-exact')
        try:
            self.create()
            name = 'pre_' + uuid.uuid4().hex[:8]
            self.load(server, name)

            def worker_stats():
                with server.connect() as connection:
                    connection.sendall(frame(STATS, 1))
                    return json.loads(receive(connection)[3])

            before = worker_stats()
            self.assertEqual(self.value('SELECT ann_frozen_precompute(%s, %s)',
                                        ('docs_ann', ['tenant = 4 AND category IN (1, 2)'])), 1)
            middle = worker_stats()
            self.assertEqual(middle['predicate_evaluations'], before['predicate_evaluations'] + 1)
            actual, plan = self.route('tenant = 4 AND category IN (1, 2)', '[1,2,3,4]', 10)
            self.assertIn(f'GPU worker "{name}"', plan)
            after = worker_stats()
            self.assertEqual(after['predicate_evaluations'], middle['predicate_evaluations'])
            self.assertGreater(after['predicate_hits'], middle['predicate_hits'])
            expected, _ = self.route('tenant = 4 AND category IN (1, 2)', '[1,2,3,4]', 10, route=False)
            self.assertEqual([row[:2] for row in actual], [row[:2] for row in expected])
        finally:
            server.close()

    def load(self, server, name, table='docs'):
        export = os.path.join(self.directory.name, name + '-export')
        path = os.path.join(self.directory.name, name)
        self.db.execute('SELECT ann_index_export(%s, %s)', (f'{table}_ann', export))
        subprocess.run([sys.executable, BUILD, '--from-export', export, '--out', path, '--no-cagra'],
                       check=True, capture_output=True)
        self.db.execute('SET ann.worker_socket = %s' % self.sql.Literal(server.path).as_string(self.db))
        self.db.execute('SELECT ann_index_load(%s, %s)', (name, path))

    def test_worker_route_and_bitmap_identity(self):
        subprocess.run(['make', '-C', os.path.join(ROOT, 'ann'), 'ann-worker'], check=True,
                       stdout=subprocess.DEVNULL)
        server = Server('--backend', 'cpu-exact')
        try:
            self.create()
            name = 'frozen_' + uuid.uuid4().hex[:8]
            self.load(server, name)
            # The worker's evaluation from its postings equals the pages', word for word.
            for predicate in self.FILTERS:
                worker = self.db.execute('SELECT eligible, bitmap FROM ann_index_filter(%s, %s)',
                                         (name, predicate)).fetchone()
                pages = self.db.execute('SELECT eligible, bitmap FROM ann_frozen_filter(%s, %s)',
                                        ('docs_ann', predicate)).fetchone()
                self.assertEqual((worker[0], bytes(worker[1])), (pages[0], bytes(pages[1])), predicate)
            rng = random.Random(9)
            for predicate in self.FILTERS[:12]:
                query = str([rng.uniform(-50, 50), rng.uniform(-40, 40), rng.uniform(-30, 30), 2])
                actual, plan = self.route(predicate, query, 20)
                expected, _ = self.route(predicate, query, 20, route=False)
                self.assertEqual([row[:2] for row in actual], [row[:2] for row in expected], predicate)
                self.assertIn(f'GPU worker "{name}"', plan, predicate)
            # Stored filters over INCLUDE attributes build from the pages, with generation ids.
            self.assertEqual(self.value('SELECT ann_filter_bitmap_build(%s, %s)', (name, 'tenant IN (2, 5)')),
                             self.value('SELECT count(*) FROM docs WHERE tenant IN (2, 5)'))
            # ann_index_filter reads rows: it needs SELECT on the table.
            role = 'ann_frozen_reader_' + uuid.uuid4().hex[:6]
            self.db.execute(f'CREATE ROLE {role}')
            self.db.execute(f'GRANT USAGE ON SCHEMA {self.schema} TO {role}')
            try:
                self.db.execute(f'SET ROLE {role}')
                self.assertIn('permission denied', self.error('SELECT * FROM ann_index_filter(%s, %s)',
                                                              (name, 'tenant = 1')))
            finally:
                self.db.execute('RESET ROLE')
                self.db.execute(f'DROP OWNED BY {role}')
                self.db.execute(f'DROP ROLE {role}')
            hits = self.db.execute('SELECT id FROM ann_bitmap_search(%s, %s, 5, %s) ORDER BY distance, id',
                                   (name, '[1,2,3,4]', 'tenant IN (2, 5)')).fetchall()
            expected = self.db.execute("SELECT id FROM docs WHERE tenant IN (2, 5) "
                                       "ORDER BY emb <-> '[1,2,3,4]', id LIMIT 5").fetchall()
            self.assertEqual(hits, expected)
        finally:
            server.close()

    def test_tombstones(self):
        """ann_frozen_delete: deleted keys leave the frozen route, on the pages and on the worker, and reach
        exports loaded later; the table itself is untouched."""
        subprocess.run(['make', '-C', os.path.join(ROOT, 'ann'), 'ann-worker'], check=True,
                       stdout=subprocess.DEVNULL)
        server = Server('--backend', 'cpu-exact')
        try:
            self.create()
            query = '[1,2,3,4]'
            nearest = [row[0] for row in self.db.execute(
                "SELECT id FROM docs ORDER BY emb <-> %s::vector, id LIMIT 6", (query,)).fetchall()]
            deleted = nearest[:3] + [999999]  # an unknown key is ignored
            self.assertEqual(self.value('SELECT ann_frozen_delete(%s, %s)', ('docs', deleted)), 4)
            self.assertEqual(self.value('SELECT ann_frozen_delete(%s, %s)', ('docs', deleted)), 0)
            self.assertEqual(self.value('SELECT count(*) FROM docs WHERE id = ANY(%s)', (deleted,)), 3)

            def expected(predicate, limit):
                return [row[0] for row in self.db.execute(
                    f'SELECT id FROM docs WHERE ({predicate}) AND id <> ALL(%s) '
                    f'ORDER BY emb <-> %s::vector, id LIMIT {limit}', (deleted, query)).fetchall()]

            for predicate in ['true', 'tenant < 7', 'category IS NOT NULL']:
                actual, plan = self.route(predicate, query, 10)          # pages: no worker yet
                self.assertIn('AnnFrozenSearch', plan)
                self.assertEqual([row[0] for row in actual], expected(predicate, 10), predicate)
            name = 'frozen_' + uuid.uuid4().hex[:8]
            self.load(server, name)                                       # tombstones go with the load
            # Earlier page scans cached that no worker was loaded. Discovery
            # refreshes that negative entry after a second, not at load time.
            deadline = time.monotonic() + 5
            while True:
                _, plan = self.route('true', query, 10)
                if f'GPU worker "{name}"' in plan or time.monotonic() >= deadline:
                    break
                time.sleep(0.05)
            self.assertIn(f'GPU worker "{name}"', plan)
            for predicate in ['true', 'tenant < 7', 'category IS NOT NULL']:
                actual, plan = self.route(predicate, query, 10)
                self.assertIn(f'GPU worker "{name}"', plan, predicate)
                self.assertEqual([row[0] for row in actual], expected(predicate, 10), predicate)
            # Deleting while loaded reaches the worker at once, and drops its cached predicate results.
            more = expected('tenant < 7', 2)
            self.db.execute('SELECT ann_frozen_delete(%s, %s)', ('docs', more))
            deleted += more
            actual, _ = self.route('tenant < 7', query, 10)
            self.assertEqual([row[0] for row in actual], expected('tenant < 7', 10))
            self.assertIn('is not a frozen table', self.error('SELECT ann_frozen_delete(%s, %s)',
                                                              ('pg_class', [1])))
        finally:
            server.close()

    def test_sampled_relation(self):
        self.create()
        self.db.execute('SET enable_sort = off')
        try:
            statement = "SELECT id FROM docs TABLESAMPLE SYSTEM(0) ORDER BY emb <-> '[1,2,3,4]' LIMIT 3"
            self.assertNotIn('AnnFrozenSearch', self.plan(statement))
            self.assertEqual(self.db.execute(statement).fetchall(), [])
        finally:
            self.db.execute('RESET enable_sort')

    def test_route_is_costed(self):
        self.create()
        self.db.execute('CREATE INDEX docs_tenant ON docs (tenant)')
        self.db.execute('ANALYZE docs')
        statement = "SELECT id FROM docs WHERE tenant = 3 ORDER BY emb <-> '[1,2,3,4]', id LIMIT 10"
        routed = self.plan(statement)
        self.assertIn('AnnFrozenSearch', routed)
        expected = self.db.execute(statement).fetchall()
        # At the default call cost, a sort of the 230 rows the btree finds is cheaper.
        self.db.execute('RESET ann.frozen_setup_cost')
        try:
            plan = self.plan(statement)
            self.assertNotIn('AnnFrozenSearch', plan)
            self.assertIn('docs_tenant', plan)
            self.assertEqual(self.db.execute(statement).fetchall(), expected)
        finally:
            self.db.execute('SET ann.frozen_setup_cost = 0')

    def test_upgrade_guard(self):
        self.create(index=False)
        self.db.execute('DROP TRIGGER ann_frozen_row_guard ON docs')
        self.assertTrue(self.value("SELECT ann_freeze('docs')"))
        self.assertFalse(self.value("SELECT ann_freeze('docs')"))
        self.db.execute('CREATE INDEX docs_ann ON docs USING ann (emb vector_l2_ops) '
                        'INCLUDE (id, tenant, category) WITH (frozen = true)')
        # The row guard follows the statement guard's lifecycle.
        self.db.execute('DROP INDEX docs_ann')
        self.db.execute('DROP TRIGGER ann_frozen_guard ON docs')
        self.db.execute("UPDATE docs SET body = 'changed' WHERE id = 5")

    def test_tombstone_delivery_failure(self):
        subprocess.run(['make', '-C', os.path.join(ROOT, 'ann'), 'ann-worker'], check=True,
                       stdout=subprocess.DEVNULL)
        self.create()
        server = Server('--backend', 'cpu-exact')
        name = 'delivery_' + uuid.uuid4().hex[:8]
        notices = []
        self.db.add_notice_handler(lambda notice: notices.append(notice.message_primary))
        try:
            self.load(server, name)
            self.db.execute('SET ann.worker_timeout_ms = 100')
            with self.db.transaction():
                self.db.execute("SELECT ann_frozen_delete('docs', ARRAY[5]::bigint[])")
                server.close()
                server = None
            self.assertEqual(self.value("SELECT count(*) FROM ann_frozen_tombstone WHERE source='docs'::regclass"), 1)
            self.assertTrue(any('committed tombstones could not be sent' in notice for notice in notices))
            server = Server('--backend', 'cpu-exact')
            self.db.execute("SELECT set_config('ann.worker_socket', %s, false)", (server.path,))
            self.db.execute('SELECT ann_index_load(%s, %s)', (name, os.path.join(self.directory.name, name)))
            self.assertEqual(self.value('SELECT eligible FROM ann_index_filter(%s, %s)', (name, 'true')), ROWS - 1)
        finally:
            if server:
                server.close()

    def test_tombstone_cursor_lifetime(self):
        self.create()
        query = '[1,2,3,4]'
        deleted = [r[0] for r in self.db.execute(
            'SELECT id FROM docs ORDER BY emb <-> %s::vector OFFSET 10 LIMIT 10', (query,))]
        self.db.execute("SELECT ann_frozen_delete('docs', %s)", (deleted,))
        with self.db.transaction():
            self.db.execute("DECLARE frozen_cursor NO SCROLL CURSOR FOR SELECT id FROM docs "
                            "ORDER BY emb <-> '[1,2,3,4]'")
            first = self.db.execute('FETCH 1 FROM frozen_cursor').fetchall()
            # Grow beyond the old allocation's size class, then refresh in another scan.
            self.db.execute("SELECT ann_frozen_delete('docs', ARRAY(SELECT generate_series(100000,100200))::bigint[])")
            self.db.execute("SELECT id FROM docs ORDER BY emb <-> '[1,2,3,4]' LIMIT 1")
            rest = self.db.execute('FETCH ALL FROM frozen_cursor').fetchall()
            self.assertEqual(len(first) + len(rest), ROWS - len(deleted))
            self.assertFalse(set(deleted) & {r[0] for r in first + rest})

    def test_tombstone_rollback(self):
        subprocess.run(['make', '-C', os.path.join(ROOT, 'ann'), 'ann-worker'], check=True,
                       stdout=subprocess.DEVNULL)
        self.create()
        query = '[1,2,3,4]'
        nearest, _ = self.route('true', query, 10)
        deleted = [r[0] for r in nearest[:3]]
        server = Server('--backend', 'cpu-exact')
        name = 'rollback_' + uuid.uuid4().hex[:8]
        try:
            self.load(server, name)
            self.db.execute('BEGIN')
            self.db.execute("SELECT ann_frozen_delete('docs', %s)", (deleted,))
            self.assertFalse(set(deleted) & {r[0] for r in self.route('true', query, 10)[0]})
            self.db.execute('ROLLBACK')
            self.assertEqual(self.value('SELECT count(*) FROM ann_frozen_tombstone WHERE source = %s::regclass',
                                        ('docs',)), 0)
            actual, _ = self.route('true', query, 10)
            self.assertEqual(actual, nearest)
            # A savepoint rollback must discard its worker notification too.
            with self.db.transaction():
                self.db.execute('SAVEPOINT deleted')
                self.db.execute("SELECT ann_frozen_delete('docs', %s)", (deleted,))
                self.route('true', query, 10)  # populate this transaction's local cache
                self.db.execute('ROLLBACK TO deleted')
            self.assertEqual(self.route('true', query, 10)[0], nearest)
            # Committing a surviving child transaction publishes the delete.
            with self.db.transaction():
                self.db.execute('SAVEPOINT deleted')
                self.db.execute("SELECT ann_frozen_delete('docs', %s)", (deleted,))
                self.db.execute('RELEASE deleted')
                # Delivery uses the destination captured by the delete.
                self.db.execute("SET LOCAL ann.worker_socket = '/tmp/ann-no-such-worker.sock'")
            eligible = self.value('SELECT eligible FROM ann_index_filter(%s, %s)', (name, 'true'))
            self.assertEqual(eligible, ROWS - len(deleted))
            hits = self.db.execute('SELECT id FROM ann_index_search(%s, %s::vector, 10)', (name, query)).fetchall()
            self.assertEqual(len(hits), 10)
            self.assertFalse(set(deleted) & {r[0] for r in hits})
        finally:
            self.db.execute('ROLLBACK')
            server.close()

    def test_generation_mismatch_and_switch(self):
        subprocess.run(['make', '-C', os.path.join(ROOT, 'ann'), 'ann-worker'], check=True,
                       stdout=subprocess.DEVNULL)
        server = Server('--backend', 'cpu-exact')
        try:
            self.create()
            first = 'gen1_' + uuid.uuid4().hex[:8]
            self.load(server, first)
            # A catalog naming another generation: tagged requests are refused.
            self.db.execute("UPDATE ann_index_catalog SET generation = '0123456789abcdef' WHERE name = %s",
                            (first,))
            other = self.psycopg.connect(DSN, autocommit=True)
            try:
                other.execute('SET ann.frozen_setup_cost = 0')
                other.execute('SET ann.worker_socket = %s' % self.sql.Literal(server.path).as_string(other))
                other.execute(f'SET search_path = {self.schema}, public')
                with self.assertRaises(self.psycopg.Error) as caught:
                    other.execute("SELECT * FROM ann_attribute_search(%s, '[1,2,3,4]', 3, 'tenant', ARRAY[1])",
                                  (first,))
                self.assertIn('another generation', str(caught.exception))
                # The frozen route finds no worker of its generation: the pages answer.
                plan = '\n'.join(row[0] for row in other.execute(
                    "EXPLAIN (ANALYZE, COSTS OFF, TIMING OFF, SUMMARY OFF) SELECT id FROM docs WHERE tenant = 1 "
                    "ORDER BY emb <-> '[1,2,3,4]' LIMIT 3"))
                self.assertIn('from index pages', plan)
            finally:
                other.close()
            self.db.execute('UPDATE ann_index_catalog SET generation = (SELECT generation FROM '
                            "ann_frozen_info('docs_ann')) WHERE name = %s", (first,))

            # A new generation: the base absorbs new rows, is frozen, exported and loaded under a
            # new name; the route follows the generation of the table it scans.
            self.db.execute('CREATE TABLE docs2 AS SELECT * FROM docs')
            self.db.execute("INSERT INTO docs2 VALUES (99995, 1, 1, 'new', '[1,2,3,4]')")
            self.db.execute("SELECT ann_freeze('docs2')")
            self.vacuum('docs2')
            self.db.execute('CREATE INDEX docs2_ann ON docs2 USING ann (emb vector_l2_ops) '
                            'INCLUDE (id, tenant, category) WITH (frozen = true)')
            second = 'gen2_' + uuid.uuid4().hex[:8]
            self.load(server, second, table='docs2')
            plan = self.plan("SELECT id FROM docs2 WHERE tenant = 1 ORDER BY emb <-> '[1,2,3,4]' LIMIT 3")
            self.assertIn(f'GPU worker "{second}"', plan)
            self.assertEqual(self.value("SELECT id FROM docs2 WHERE tenant = 1 ORDER BY emb <-> '[1,2,3,4]' "
                                        'LIMIT 1'), 99995)
            # Retiring the old generation's table drops its index and guard.
            self.db.execute('DROP TABLE docs')
            self.assertEqual(self.value(f"SELECT count(*) FROM pg_trigger t JOIN pg_class c ON c.oid = t.tgrelid "
                                        f"JOIN pg_namespace n ON n.oid = c.relnamespace "
                                        f"WHERE n.nspname = '{self.schema}' AND c.relname = 'docs'"), 0)
        finally:
            server.close()

    def export(self, name):
        path = os.path.join(self.directory.name, name)
        self.assertEqual(self.value('SELECT ann_index_export(%s, %s)', ('docs_ann', path)), ROWS)
        return path

    def test_vector_storage(self):
        """vectors = half keeps FP16 pages for the CPU routes, vectors = heap none; exports stay exact."""
        self.create(index=False)
        build = ('CREATE INDEX docs_ann ON docs USING ann (emb vector_l2_ops) INCLUDE (id, tenant, category) '
                 'WITH (frozen = true{})')
        self.db.execute(build.format(''))
        generation = self.value("SELECT generation FROM ann_frozen_info('docs_ann')")
        base = self.export('float')
        float_pages = self.value("SELECT pages FROM ann_frozen_sizes('docs_ann') WHERE structure = 'vectors (float)'")
        self.assertGreater(float_pages, 0)
        query = '[11,-7,3,2]'
        for mode in ('half', 'heap'):
            self.db.execute('DROP INDEX docs_ann')
            self.db.execute(build.format(f', vectors = {mode}'))
            self.assertEqual(self.value("SELECT generation FROM ann_frozen_info('docs_ann')"), generation)
            pages = self.value(f"SELECT pages FROM ann_frozen_sizes('docs_ann') WHERE structure = 'vectors ({mode})'")
            if mode == 'heap':
                self.assertEqual(pages, 0)
            else:
                self.assertTrue(0 < pages < float_pages, (pages, float_pages))
            # Exports read the heap's exact vectors: the same files as from a float index.
            path = self.export(mode)
            for name in ['keys.i64', 'vectors.f32', 'attr-tenant.i32', 'attr-category.i32']:
                self.assertTrue(filecmp.cmp(os.path.join(base, name), os.path.join(path, name), shallow=False),
                                (mode, name))
            # The frozen route without a worker scores from the pages (half) or the heap.
            for predicate in ['tenant = 3', 'category IN (1, 4) AND tenant > 6', 'TRUE']:
                # The vector is not covered now, so on 3,000 rows other plans cost less: switch them off.
                statement = (f'SELECT id, tenant, emb <-> %s::vector AS d FROM docs WHERE {predicate} '
                             f'ORDER BY emb <-> %s::vector LIMIT 10')
                for setting in ['enable_seqscan', 'enable_indexscan', 'enable_indexonlyscan', 'enable_bitmapscan']:
                    self.db.execute(f'SET {setting} = off')
                try:
                    actual = self.db.execute(statement, (query, query)).fetchall()
                    plan = self.plan(statement, (query, query))
                finally:
                    for setting in ['enable_seqscan', 'enable_indexscan', 'enable_indexonlyscan', 'enable_bitmapscan']:
                        self.db.execute(f'RESET {setting}')
                expected, _ = self.route(predicate, query, 10, route=False)
                self.assertIn('AnnFrozenSearch', plan)
                if mode == 'heap':
                    self.assertEqual([r[:2] for r in actual], [r[:2] for r in expected], predicate)
                else:
                    overlap = len({r[0] for r in actual} & {r[0] for r in expected})
                    self.assertGreaterEqual(overlap, 8, predicate)
                    for got, want in zip(actual, expected):
                        self.assertAlmostEqual(got[2], want[2], delta=max(0.02 * want[2], 0.05))
            # The vector column is never served from half or missing pages.
            statement = 'SELECT id, emb FROM docs WHERE tenant = 3 ORDER BY emb <-> %s::vector LIMIT 5'
            rows = self.db.execute(statement, (query,)).fetchall()
            exact = dict(self.db.execute('SELECT id, emb FROM docs WHERE tenant = 3').fetchall())
            for key, vector in rows:
                self.assertEqual(vector, exact[key])
            self.assertNotIn('index pages when all-visible', self.plan(statement, (query,)))
            # The access method's ordered scan.
            got, plan = self.ordered('id', 'TRUE', query, 7, True)
            want, _ = self.ordered('id', 'TRUE', query, 7, False)
            if mode == 'heap':
                self.assertEqual([r[0] for r in got], [r[0] for r in want])
            else:
                self.assertGreaterEqual(len({r[0] for r in got} & {r[0] for r in want}), 6)
        self.assertIn('invalid value', self.error(build.replace('docs_ann', 'docs_bad').format(', vectors = int8')))

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
            db.execute('SET ann.frozen_setup_cost = 0')
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
            # k=500, with 750 eligible rows: above this worker's exact CPU
            # limit, so rollback and committed deletes exercise the GPU route.
            query = db.execute(f'SELECT emb::text FROM {table} WHERE id=3').fetchone()[0]
            statement = f'SELECT id FROM {table} WHERE tenant=1 ORDER BY emb <-> %s::vector LIMIT 500'
            before = db.execute(statement, (query,)).fetchall()
            self.assertEqual(len(before), 500)
            deleted = [r[0] for r in before[:3]]
            db.execute('BEGIN')
            db.execute('SELECT ann_frozen_delete(%s,%s)', (table, deleted))
            own = db.execute(statement, (query,)).fetchall()
            self.assertFalse(set(deleted) & {r[0] for r in own})
            db.execute('ROLLBACK')
            self.assertEqual(db.execute(statement, (query,)).fetchall(), before)
            self.assertEqual(db.execute('SELECT eligible FROM ann_index_filter(%s,%s)',
                                       (table, 'true')).fetchone()[0], 30000)
            db.execute('SELECT ann_frozen_delete(%s,%s)', (table, deleted))
            hits = db.execute('SELECT id FROM ann_index_search(%s,%s::vector,500)', (table, query)).fetchall()
            self.assertEqual(len(hits), 500)
            self.assertFalse(set(deleted) & {r[0] for r in hits})
            db.execute('SELECT ann_index_unload(%s)', (table,))
        finally:
            db.execute('ROLLBACK')
            db.execute('DELETE FROM ann_frozen_tombstone WHERE source=to_regclass(%s)', (table,))
            db.execute(f'DROP TABLE IF EXISTS {table}')
            db.close()
            server.close()
            directory.cleanup()


if __name__ == '__main__':
    unittest.main()
