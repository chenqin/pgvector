"""The worker's predicate cache: a memory budget with least-recently-used
eviction, pushed bitmaps kept, and index prewarming and memory locking.

Run with the CPU worker (make -C ann check). The fixture has no CAGRA
shards, so GPU backends cannot load it.
"""

import json
import os
import random
import struct
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_server import ROOT, STATS, Server, frame, receive  # noqa: E402
from test_index import (MISSING, decode_hits, exact, load_frame, put_bitmap_frame,  # noqa: E402
                        search_cached_frame, write_index)

SEARCH_PREDICATE = 22
PREDICATE = 23
FILTER_VALUES = 1

ROWS = 40000
DIMENSIONS = 4
GROUPS = 1000      # attribute a = row % GROUPS
WINDOW = 100       # each program selects WINDOW groups: 4000 rows
BUDGET_MB = 1


def values_program(column, values, negate=False):
    return (struct.pack('<BB', FILTER_VALUES, len(column)) + column.encode() +
            struct.pack('<BI', 1 if negate else 0, len(values)) + struct.pack('<%di' % len(values), *values))


def search_predicate_frame(request, name, query, k, program, flags=0):
    payload = struct.pack('<I', len(name)) + name.encode()
    payload += struct.pack('<IIII', len(query), k, flags, len(program)) + program
    payload += struct.pack('<%df' % len(query), *query)
    return frame(SEARCH_PREDICATE, request, payload)


def search_batch_frame(request, name, queries, k, program, flags=0):
    payload = struct.pack('<I', len(name)) + name.encode()
    payload += struct.pack('<IIIII', len(queries[0]), k, flags, len(program), len(queries))
    payload += program
    payload += b''.join(struct.pack('<%df' % len(q), *q) for q in queries)
    return frame(30, request, payload)


def predicate_frame(request, name, program):
    return frame(PREDICATE, request, struct.pack('<I', len(name)) + name.encode() + program)


def words_of(rows, total):
    words = [0] * ((total + 31) // 32)
    for row in rows:
        words[row // 32] |= 1 << (row % 32)
    return words


class PredicateCacheTests(unittest.TestCase):
    worker = os.path.join(ROOT, 'ann', os.environ.get('ANN_INDEX_WORKER', 'ann-worker'))
    backend = os.environ.get('ANN_INDEX_BACKEND', 'cpu-exact')

    @classmethod
    def setUpClass(cls):
        if not os.environ.get('ANN_INDEX_WORKER'):
            subprocess.run(['make', '-C', os.path.join(ROOT, 'ann'), 'ann-worker'], check=True,
                           stdout=subprocess.DEVNULL)
        cls.directory = tempfile.TemporaryDirectory()
        rng = random.Random(5)
        # Keys 0 .. ROWS - 1, so a row's ordinal is its key.
        cls.keys = list(range(ROWS))
        cls.vectors = [[round(rng.uniform(-3, 3), 3) for _ in range(DIMENSIONS)] for _ in range(ROWS)]
        cls.groups = [row % GROUPS for row in range(ROWS)]
        cls.path = os.path.join(cls.directory.name, 'docs')
        write_index(cls.path, cls.keys, cls.vectors, shards=3, attributes={'a': cls.groups})

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def start(self, *options):
        self.server = Server('--backend', self.backend, *options, executable=self.worker)
        self.connection = self.server.connect()
        _, _, status, info = self.call(load_frame(1, 'docs', self.path))
        self.assertEqual(status, 0, info)

    def stop(self):
        """Stop the worker and return what it wrote to stderr."""
        self.connection.close()
        self.server.process.terminate()
        self.server.process.wait(10)
        log = self.server.process.stderr.read().decode(errors='replace')
        self.server.close()
        return log

    def call(self, data):
        self.connection.sendall(data)
        return receive(self.connection)

    def stats(self):
        _, _, status, payload = self.call(frame(STATS, 99))
        self.assertEqual(status, 0)
        return json.loads(payload)

    def window(self, start):
        return list(range(start, start + WINDOW))

    def rows(self, groups):
        groups = set(groups)
        return [row for row in range(ROWS) if self.groups[row] in groups]

    def evaluate(self, request, groups):
        """Operation 23 over a VALUES program, checked against the attribute."""
        _, _, status, payload = self.call(predicate_frame(request, 'docs', values_program('a', groups)))
        self.assertEqual(status, 0)
        rows = self.rows(groups)
        eligible, = struct.unpack_from('<Q', payload)
        self.assertEqual(eligible, len(rows))
        words = list(struct.unpack_from('<%dI' % ((ROWS + 31) // 32), payload, 8))
        self.assertEqual(words, words_of(rows, ROWS))

    def search(self, request, groups, query, k=10):
        """Operation 22 over a VALUES program, checked against an exact scan."""
        _, _, status, payload = self.call(
            search_predicate_frame(request, 'docs', query, k, values_program('a', groups)))
        self.assertEqual(status, 0)
        count, route, eligible = struct.unpack_from('<IIQ', payload)
        allowed = set(self.rows(groups))
        self.assertEqual(eligible, len(allowed))
        hits = [struct.unpack_from('<qf', payload, 16 + i * 12) for i in range(count)]
        want = exact(self.keys, self.vectors, query, k, allowed)
        self.assertEqual([key for key, _ in hits], [key for _, key in want])
        for (_, distance), (expected, _) in zip(hits, want):
            self.assertAlmostEqual(distance, expected, places=4)

    def test_query_matrix(self):
        """One IPC preserves each query's top-k, ordering and shared filter."""
        self.start('--lanes', '4')
        try:
            rng = random.Random(109)
            for count in (1, 7, 64):
                queries = [[round(rng.uniform(-3, 3), 2) for _ in range(DIMENSIONS)]
                           for _ in range(count)]
                for program in (b'\x08', b'\x09', values_program('a', [3, 7, 11])):
                    expected = []
                    for q in queries:
                        _, _, status, data = self.call(
                            search_predicate_frame(1, 'docs', q, 17, program))
                        self.assertEqual(status, 0)
                        expected.append(data)
                    op, request, status, data = self.call(
                        search_batch_frame(2, 'docs', queries, 17, program))
                    self.assertEqual((op, request, status), (30, 2, 0))
                    self.assertEqual(struct.unpack_from('<I', data)[0], count)
                    offset = 4
                    for want in expected:
                        n = struct.unpack_from('<I', data, offset)[0]
                        size = 16 + n * 12
                        self.assertEqual(data[offset:offset + size], want)
                        offset += size
                    self.assertEqual(offset, len(data))
        finally:
            self.stop()

    def test_query_matrix_snapshot_candidates(self):
        self.start('--predicate-cache-mb', str(BUDGET_MB))
        try:
            query = [1., 2., 3., 4.]
            program = values_program('a', [3, 7, 11])
            want = [key for _, key in exact(self.keys, self.vectors, query, 17, set(self.rows([3, 7, 11])))]
            payload = struct.pack('<I', 4) + b'docs' + struct.pack('<I3q', 3, *want[:3])
            self.assertEqual(self.call(frame(29, 1, payload))[2], 0)
            for flags in (8, 0, 8):
                _, _, status, data = self.call(search_batch_frame(2, 'docs', [query] * 7, 17, program, flags))
                self.assertEqual(status, 0)
                offset = 4
                for _ in range(7):
                    count, _, eligible = struct.unpack_from('<IIQ', data, offset)
                    got = [struct.unpack_from('<qf', data, offset + 16 + i * 12)[0] for i in range(count)]
                    if flags:
                        self.assertEqual(got, want)
                        self.assertEqual(eligible, 120)
                    else:
                        self.assertFalse(set(got) & set(want[:3]))
                        self.assertEqual(eligible, 117)
                    offset += 16 + count * 12
                self.assertEqual(offset, len(data))
            stats = self.stats()
            self.assertEqual(stats['predicate_entries'], 2)
            self.assertLessEqual(stats['predicate_bytes'], BUDGET_MB << 20)
        finally:
            self.stop()

    def test_query_matrix_validation(self):
        self.start()
        try:
            queries = [[0.] * DIMENSIONS] * 3
            valid = search_batch_frame(1, 'docs', queries, 5, b'\x08')
            cases = [valid[:-1], valid + b'\0',
                     search_batch_frame(1, 'docs', queries, 0, b'\x08'),
                     search_batch_frame(1, 'docs', queries * 22, 5, b'\x08'),
                     search_batch_frame(1, 'docs', queries, 5, b'\xff'),
                     search_batch_frame(1, 'docs', queries, 5, b'\x08', flags=2)]
            # Reframe truncated/extended payloads so framing itself remains valid.
            for data in cases:
                data = frame(30, 1, data[32:])
                self.assertEqual(self.call(data)[2], 1)
            stale = bytearray(valid)
            struct.pack_into('<Q', stale, 16, 1)  # fixture generation is zero
            self.assertEqual(self.call(bytes(stale))[2], 12)
            # Errors after creating the composite must reply exactly once, too.
            for name, program in [('missing', b'\x08'),
                                  ('docs', values_program('missing', [1]))]:
                self.assertNotEqual(self.call(
                    search_batch_frame(2, name, queries, 5, program))[2], 0)
            self.assertEqual(self.call(search_predicate_frame(3, 'docs', queries[0], 5, b'\x08', flags=8))[2], 1)
            self.assertEqual(self.call(valid)[2], 0)
            self.assertEqual(self.stats()['failed'], 0)
        finally:
            self.stop()

    def test_sparse_results_keep_rows_only(self):
        """Results under 1/64 of the rows keep their rows, not a bitmap; replies and searches are unchanged."""
        self.start()
        try:
            bitmap = (ROWS + 31) // 32 * 4
            before = self.stats()['predicate_bytes']
            sparse = [3, 7, 11]   # 120 rows, under ROWS / 64
            self.evaluate(1, sparse)
            added = self.stats()['predicate_bytes'] - before
            # Ordinals only (8 bytes a row): predicate results keep no host keys.
            self.assertEqual(added, len(self.rows(sparse)) * 8)
            self.assertLess(added, bitmap)
            rng = random.Random(9)
            for i in range(5):
                self.search(10 + i, sparse, [round(rng.uniform(-3, 3), 2) for _ in range(DIMENSIONS)], k=7)
            self.evaluate(20, sparse)   # words rebuilt from the cached rows
            before = self.stats()['predicate_bytes']
            self.evaluate(30, self.window(0))   # 4,000 rows: a bitmap is kept
            self.assertGreaterEqual(self.stats()['predicate_bytes'] - before, bitmap)
        finally:
            self.stop()

    def test_budget_evicts_least_recently_used(self):
        self.start('--predicate-cache-mb', str(BUDGET_MB))
        try:
            rng = random.Random(11)
            budget = BUDGET_MB << 20

            # A pushed bitmap (operation 20) is outside the predicate budget.
            pushed = [row for row in range(ROWS) if self.groups[row] < 5]
            bitmap = struct.pack('<%dI' % ((ROWS + 31) // 32), *words_of(pushed, ROWS))
            self.assertEqual(self.call(put_bitmap_frame(2, 'docs', 7, bitmap))[2], 0)

            hot = self.window(0)
            cold = self.window(GROUPS - WINDOW)
            self.evaluate(3, hot)
            self.evaluate(4, cold)
            request = 5
            programs = 2
            for start in range(WINDOW, GROUPS - 2 * WINDOW, 10):
                self.evaluate(request, self.window(start))
                request += 1
                programs += 1
                if programs % 4 == 0:
                    # Keep the hot predicate recently used.
                    self.search(request, hot, [rng.uniform(-3, 3) for _ in range(DIMENSIONS)])
                    request += 1
                stats = self.stats()
                self.assertLessEqual(stats['predicate_bytes'], budget)
            self.assertGreater(stats['predicate_evictions'], 0)
            self.assertLess(stats['predicate_entries'], programs)
            self.assertGreater(stats['predicate_entries'], 0)
            self.assertEqual(stats['predicate_device_bytes'], 0)

            # The recently used predicate is still cached: a hit, no evaluation.
            before = self.stats()
            self.evaluate(request, hot)
            after = self.stats()
            self.assertEqual(after['predicate_hits'], before['predicate_hits'] + 1)
            self.assertEqual(after['predicate_evaluations'], before['predicate_evaluations'])

            # The old one was evicted: evaluated again, and still correct.
            self.evaluate(request + 1, cold)
            again = self.stats()
            self.assertEqual(again['predicate_evaluations'], after['predicate_evaluations'] + 1)
            self.assertLessEqual(again['predicate_bytes'], budget)

            # The pushed bitmap survived every eviction.
            query = [rng.uniform(-3, 3) for _ in range(DIMENSIONS)]
            _, _, status, payload = self.call(search_cached_frame(request + 2, 'docs', query, 10, 7))
            self.assertNotEqual(status, MISSING)
            self.assertEqual(status, 0)
            _, hits = decode_hits(payload)
            want = exact(self.keys, self.vectors, query, 10, set(pushed))
            self.assertEqual([key for key, _ in hits], [key for _, key in want])
        finally:
            self.stop()

    def search_program(self, request, program, allowed, query, k=10):
        """Operation 22 over any program whose rows are allowed, checked against an exact scan."""
        _, _, status, payload = self.call(search_predicate_frame(request, 'docs', query, k, program))
        self.assertEqual(status, 0)
        count, _, eligible = struct.unpack_from('<IIQ', payload)
        self.assertEqual(eligible, len(allowed))
        hits = [struct.unpack_from('<qf', payload, 16 + i * 12)[0] for i in range(count)]
        self.assertEqual(hits, [key for _, key in exact(self.keys, self.vectors, query, k, set(allowed))])

    def test_canonical_predicates_share_entries(self):
        def range_program(low, high, negate=False):
            return struct.pack('<BB', 2, 1) + b'a' + struct.pack('<Bii', 1 if negate else 0, low, high)

        rng = random.Random(3)
        query = [rng.uniform(-3, 3) for _ in range(DIMENSIONS)]
        rows = self.rows([3, 4, 5])
        # One row set, five spellings: one evaluation, four hits.
        spellings = [values_program('a', [5, 3, 4]), range_program(3, 5), values_program('a', [3, 4, 5, 5, 4]),
                     range_program(3, 5) + values_program('a', [GROUPS + 7], negate=True) + b'\x05',
                     values_program('a', [3]) + range_program(4, 5) + b'\x06']
        for mode, evaluations in [('off', len(spellings)), ('on', 1)]:
            self.start('--canonical-predicates', mode)
            try:
                before = self.stats()
                for request, program in enumerate(spellings, 2):
                    self.search_program(request, program, rows, query)
                after = self.stats()
                self.assertEqual(after['predicate_evaluations'] - before['predicate_evaluations'], evaluations)
                self.assertEqual(after['predicate_canonical'] - before['predicate_canonical'],
                                 len(spellings) if mode == 'on' else 0)
                # Values the column lacks match nothing; every value is every row.
                self.search_program(20, values_program('a', [GROUPS + 1, -5]), [], query)
                self.search_program(21, range_program(0, GROUPS - 1), list(range(ROWS)), query)
            finally:
                self.stop()

    def test_second_sighting_admission(self):
        rng = random.Random(4)
        query = [rng.uniform(-3, 3) for _ in range(DIMENSIONS)]
        groups = self.window(0)
        self.start('--predicate-admission', 'second')
        try:
            # First sighting: served, not cached.
            self.search(2, groups, query)
            stats = self.stats()
            self.assertEqual((stats['predicate_transient'], stats['predicate_entries']), (1, 0))
            # Second: evaluated again and cached; third: a hit.
            self.search(3, groups, query)
            stats = self.stats()
            self.assertEqual((stats['predicate_evaluations'], stats['predicate_entries']), (2, 1))
            self.search(4, groups, query)
            self.assertEqual(self.stats()['predicate_hits'], stats['predicate_hits'] + 1)
            # Precomputed predicates (operation 23) are cached at once.
            self.evaluate(5, self.window(WINDOW))
            self.assertEqual(self.stats()['predicate_entries'], 2)
        finally:
            self.stop()

    def test_entry_cap(self):
        self.start('--predicate-cache', '3')
        try:
            for request, start in enumerate(range(0, 10 * WINDOW, WINDOW), 2):
                self.evaluate(request, self.window(start))
                self.assertLessEqual(self.stats()['predicate_entries'], 3)
            stats = self.stats()
            self.assertGreater(stats['predicate_evictions'], 0)
            # The last result is cached.
            self.evaluate(50, self.window(9 * WINDOW))
            self.assertEqual(self.stats()['predicate_hits'], stats['predicate_hits'] + 1)
        finally:
            self.stop()

    def test_prewarm_and_lock_memory(self):
        rng = random.Random(13)
        self.start('--prewarm', 'on', '--lock-memory', 'on')
        try:
            self.search(2, self.window(300), [rng.uniform(-3, 3) for _ in range(DIMENSIONS)])
            self.evaluate(3, self.window(500))
        finally:
            log = self.stop()
        self.assertIn('prewarmed', log)
        # Locking may be refused by RLIMIT_MEMLOCK; the worker warns and serves.
        self.assertTrue('locked' in log or 'could not lock' in log, log)

        self.start('--prewarm', 'off')
        try:
            self.search(2, self.window(300), [rng.uniform(-3, 3) for _ in range(DIMENSIONS)])
        finally:
            log = self.stop()
        self.assertNotIn('prewarmed', log)


if __name__ == '__main__':
    unittest.main()
