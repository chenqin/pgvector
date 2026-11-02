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

    def test_sparse_results_keep_rows_only(self):
        """Results under 1/64 of the rows keep their rows, not a bitmap; replies and searches are unchanged."""
        self.start()
        try:
            bitmap = (ROWS + 31) // 32 * 4
            before = self.stats()['predicate_bytes']
            sparse = [3, 7, 11]   # 120 rows, under ROWS / 64
            self.evaluate(1, sparse)
            added = self.stats()['predicate_bytes'] - before
            self.assertEqual(added, len(self.rows(sparse)) * 16)
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
