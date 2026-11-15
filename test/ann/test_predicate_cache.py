"""The worker's predicate cache: a memory budget with least-recently-used
eviction, pushed bitmaps kept, and index prewarming and memory locking.

Run with the CPU worker (make -C ann/server check). The fixture has no CAGRA
shards, so GPU backends cannot load it.
"""

import json
import os
import random
import select
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_server import ROOT, STATS, Server, frame, receive  # noqa: E402
from test_index import (MISSING, decode_hits, exact, load_frame, put_bitmap_frame,  # noqa: E402
                        search_cached_frame, write_index)

SEARCH_PREDICATE = 22
PREDICATE = 23
COLLECTION = 25
STALE = 3
INVALID = 1
OVERLOADED = 7
COLLECTION_CODES = 4    # ANN_PREDICATE_COLLECTION_CODES
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


def remap_frame(request, name, attribute, mapping):
    """Operation 28: rewrite a text attribute's codes into a collection's dictionary."""
    payload = struct.pack('<I', len(name)) + name.encode() + struct.pack('<I', len(attribute)) + attribute.encode()
    return frame(28, request, payload + struct.pack('<%dI' % (len(mapping) + 1), len(mapping), *mapping))


def predicate_frame(request, name, program):
    return frame(PREDICATE, request, struct.pack('<I', len(name)) + name.encode() + program)


def words_of(rows, total):
    words = [0] * ((total + 31) // 32)
    for row in rows:
        words[row // 32] |= 1 << (row % 32)
    return words


class PredicateFixture:
    """The fixture index and the helpers both test classes use."""
    worker = os.path.join(ROOT, 'ann', 'server', os.environ.get('ANN_INDEX_WORKER', 'ann-worker'))
    backend = os.environ.get('ANN_INDEX_BACKEND', 'cpu-exact')
    options = ()        # a variant's flags, before each test's own

    @classmethod
    def setUpClass(cls):
        if not os.environ.get('ANN_INDEX_WORKER'):
            subprocess.run(['make', '-C', os.path.join(ROOT, 'ann', 'server'), 'ann-worker'], check=True,
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

    @classmethod
    def write_text(cls):
        """A copy of the fixture plus a text attribute t (code row % 4) a collection can remap."""
        cls.text = os.path.join(cls.directory.name, 'text')
        shutil.copytree(cls.path, cls.text)
        with open(os.path.join(cls.text, 'attr-t.i32'), 'wb') as out:
            out.write(struct.pack('<%di' % ROWS, *[row % 4 for row in range(ROWS)]))
        with open(os.path.join(cls.text, 'strings.bin'), 'wb') as out:
            out.write(b'wxyz')
        with open(os.path.join(cls.text, 'starts.u32'), 'wb') as out:
            out.write(struct.pack('<5I', 0, 1, 2, 3, 4))
        with open(os.path.join(cls.text, 'manifest'), 'a') as out:
            out.write('attribute t attr-t.i32\ndictionary t strings.bin starts.u32\n')

    def start(self, *options, env=None):
        options = self.options + options
        # Whether deletes patch cached results: the last --delete-patch the worker gets wins.
        flags = shlex.split(os.environ.get('ANN_WORKER_EXTRA_FLAGS', '')) + list(options)
        values = [flags[i + 1] for i in range(len(flags) - 1) if flags[i] == '--delete-patch']
        self.patching = not values or values[-1] != 'off'
        self.server = Server('--backend', self.backend, *options, env=env, executable=self.worker)
        self.connection = self.server.connect()
        self.log = ''
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

    def wait_log(self, line, timeout=10):
        """Read the worker's stderr until it holds line."""
        deadline = time.monotonic() + timeout
        descriptor = self.server.process.stderr.fileno()
        while line not in self.log:
            remaining = deadline - time.monotonic()
            self.assertGreater(remaining, 0, 'no %r in %r' % (line, self.log))
            ready, _, _ = select.select([descriptor], [], [], remaining)
            if ready:
                chunk = os.read(descriptor, 65536)
                self.assertTrue(chunk, 'worker exited: %r' % self.log)
                self.log += chunk.decode(errors='replace')

    def stats(self):
        _, _, status, payload = self.call(frame(STATS, 99))
        self.assertEqual(status, 0)
        return json.loads(payload)

    def window(self, start):
        return list(range(start, start + WINDOW))

    def rows(self, groups):
        groups = set(groups)
        return [row for row in range(ROWS) if self.groups[row] in groups]

    def evaluate(self, request, groups, name='docs'):
        """Operation 23 over a VALUES program, checked against the attribute."""
        _, _, status, payload = self.call(predicate_frame(request, name, values_program('a', groups)))
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


class PredicateCacheTests(PredicateFixture, unittest.TestCase):
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

    def delete(self, request, rows):
        """Operation 29 (keys are ordinals here); returns (newly deleted, deleted in all)."""
        payload = struct.pack('<I', 4) + b'docs' + struct.pack('<I%dq' % len(rows), len(rows), *rows)
        _, _, status, reply = self.call(frame(29, request, payload))
        self.assertEqual(status, 0)
        return struct.unpack('<QQ', reply)

    def batch(self, request, query, program, flags, count=7, k=17):
        """Operation 30: each query's (eligible, hit keys)."""
        _, _, status, data = self.call(search_batch_frame(request, 'docs', [query] * count, k, program, flags))
        self.assertEqual(status, 0)
        offset = 4
        answers = []
        for _ in range(count):
            n, _, eligible = struct.unpack_from('<IIQ', data, offset)
            answers.append((eligible, [struct.unpack_from('<qf', data, offset + 16 + i * 12)[0] for i in range(n)]))
            offset += 16 + n * 12
        self.assertEqual(offset, len(data))
        return answers

    def test_delete_keeps_snapshot_entries(self):
        """A snapshot entry (operation 30 flag 8) stays cached across deletes, with its dead rows."""
        self.start()
        try:
            query = [1., 2., 3., 4.]
            groups = [3, 7, 11]
            program = values_program('a', groups)
            rows = self.rows(groups)
            want = [key for _, key in exact(self.keys, self.vectors, query, 17, set(rows))]
            self.assertEqual(self.batch(2, query, program, 8), [(len(rows), want)] * 7)
            self.assertEqual(self.batch(3, query, program, 0), [(len(rows), want)] * 7)
            dead = []
            for request, gone in enumerate([want[:2], want[2:3]], 3):
                dead += gone
                before = self.stats()
                self.assertEqual(self.delete(request, gone), (len(gone), len(dead)))
                self.assertEqual(self.batch(10 + request, query, program, 8), [(len(rows), want)] * 7)
                alive = [row for row in rows if row not in dead]
                expected = [key for _, key in exact(self.keys, self.vectors, query, 17, set(alive))]
                self.assertEqual(self.batch(20 + request, query, program, 0), [(len(alive), expected)] * 7)
                after = self.stats()
                self.assertGreater(after['predicate_hits'], before['predicate_hits'])
                self.assertEqual(after['predicate_entries'], 2)
                if self.patching:
                    # The snapshot entry is kept, the other one patched: no evaluation.
                    self.assertEqual(after['predicate_evaluations'], before['predicate_evaluations'])
                    self.assertEqual(after['predicate_patched'], before['predicate_patched'] + 1)
                    self.assertEqual(after['predicate_dropped'], 0)
                else:
                    # The snapshot entry is kept, the other one dropped and evaluated again.
                    self.assertEqual(after['predicate_evaluations'], before['predicate_evaluations'] + 1)
                    self.assertEqual(after['predicate_patched'], 0)
                    self.assertEqual(after['predicate_dropped'], before['predicate_dropped'] + 1)
        finally:
            self.stop()

    def check_program(self, request, program, allowed, query, k=10):
        """Operation 23's words and count, and operation 22's eligible count and hits, over allowed rows."""
        _, _, status, payload = self.call(predicate_frame(request, 'docs', program))
        self.assertEqual(status, 0)
        eligible, = struct.unpack_from('<Q', payload)
        self.assertEqual(eligible, len(allowed))
        words = list(struct.unpack_from('<%dI' % ((ROWS + 31) // 32), payload, 8))
        self.assertEqual(words, words_of(allowed, ROWS))
        self.search_program(request + 1, program, allowed, query, k)

    def delete_programs(self):
        """Dense, sparse, negated and untouched programs, and rows deleted from the first three."""
        programs = [values_program('a', list(range(800))),          # 32,000 rows: a bitmap
                    values_program('a', list(range(900, 905))),     # 200 rows: rows only
                    values_program('a', list(range(850, 860)), negate=True),
                    values_program('a', list(range(950, 955)))]     # no deleted row
        rows = [self.rows(range(800)), self.rows(range(900, 905)),
                [row for row in range(ROWS) if not 850 <= self.groups[row] < 860], self.rows(range(950, 955))]
        dead = [5, 1005, 901, 1902, 855]
        return programs, rows, dead

    def test_delete_patches_entries(self):
        """A delete patches the cached results holding its rows: no evaluation, exact replies."""
        self.start()
        try:
            rng = random.Random(21)
            query = [rng.uniform(-3, 3) for _ in range(DIMENSIONS)]
            programs, rows, dead = self.delete_programs()
            for request, (program, allowed) in enumerate(zip(programs, rows)):
                self.check_program(10 + 2 * request, program, allowed, query)
            before = self.stats()
            self.assertEqual(self.delete(20, dead), (len(dead), len(dead)))
            middle = self.stats()
            if self.patching:
                self.assertEqual(middle['predicate_patched'] - before['predicate_patched'], 3)
                self.assertEqual(middle['predicate_dropped'], 0)
                self.assertEqual(middle['predicate_entries'], len(programs))
            for request, (program, allowed) in enumerate(zip(programs, rows)):
                alive = [row for row in allowed if row not in dead]
                # Queries at the deleted rows themselves: they must not come back.
                for i, row in enumerate([None] + dead):
                    self.check_program(30 + 20 * request + 2 * i, program, alive,
                                       query if row is None else self.vectors[row])
            after = self.stats()
            if self.patching:
                self.assertEqual(after['predicate_evaluations'], before['predicate_evaluations'])
            self.assertGreater(after['predicate_hits'], middle['predicate_hits'])
            self.assertEqual(after['predicate_raced'], 0)
        finally:
            self.stop()

    def test_delete_patch_off_flushes(self):
        """--delete-patch off, or a delete of more than --delete-patch-rows rows, drops cached results."""
        rng = random.Random(22)
        query = [rng.uniform(-3, 3) for _ in range(DIMENSIONS)]
        programs, rows, dead = self.delete_programs()
        for options in [('--delete-patch', 'off'), ('--delete-patch-rows', str(len(dead) - 1))]:
            self.start(*options)
            try:
                for request, (program, allowed) in enumerate(zip(programs, rows)):
                    self.check_program(10 + 2 * request, program, allowed, query)
                before = self.stats()
                self.assertEqual(self.delete(20, dead), (len(dead), len(dead)))
                for request, (program, allowed) in enumerate(zip(programs, rows)):
                    self.check_program(30 + 2 * request, program, [row for row in allowed if row not in dead], query)
                after = self.stats()
                self.assertEqual(after['predicate_evaluations'] - before['predicate_evaluations'], len(programs))
                self.assertEqual(after['predicate_dropped'], len(programs))
                self.assertEqual(after['predicate_patched'], 0)
            finally:
                self.stop()

    def test_delete_threshold_crossing(self):
        """A patched result that falls under rows / 64 keeps only its rows, as a fresh evaluation would."""
        groups = list(range(16))                # 640 rows, over ROWS / 64 = 625: a bitmap
        program = values_program('a', groups)
        rows = self.rows(groups)
        dead = rows[::32][:20]
        alive = [row for row in rows if row not in dead]
        query = [0.5, -1., 2., 0.]
        self.start()
        try:
            self.check_program(2, program, rows, query)
            self.assertGreater(self.stats()['predicate_bytes'], len(rows) * 8 + (ROWS + 31) // 32 * 4)
            self.assertEqual(self.delete(4, dead), (20, 20))
            stats = self.stats()
            if self.patching:
                self.assertEqual((stats['predicate_patched'], stats['predicate_entries']), (1, 1))
                self.assertEqual(stats['predicate_bytes'], len(alive) * 8)
            self.check_program(5, program, alive, query)
            self.assertEqual(self.stats()['predicate_evaluations'], 1 if self.patching else 2)
            self.assertEqual(self.stats()['predicate_bytes'], len(alive) * 8)
        finally:
            self.stop()
        # A worker that evaluates it after the delete holds the same entry.
        self.start()
        try:
            self.assertEqual(self.delete(2, dead), (20, 20))
            self.check_program(3, program, alive, query)
            self.assertEqual(self.stats()['predicate_bytes'], len(alive) * 8)
        finally:
            self.stop()

    def test_deleted_rows_never_return(self):
        """After deletes, patches, eviction and re-evaluation, a deleted row is in no reply."""
        groups = list(range(40, 60))
        program = values_program('a', groups)
        rows = self.rows(groups)
        self.start('--predicate-cache', '1')
        try:
            dead = []
            for request, row in enumerate(rows[3:300:37], 1):
                dead.append(row)
                self.assertEqual(self.delete(100 + request, [row]), (1, len(dead)))
                alive = [r for r in rows if r not in dead]
                for i, at in enumerate(dead):
                    self.check_program(200 + 20 * request + 2 * i, program, alive, self.vectors[at], k=5)
                if request % 3 == 0:
                    # Evict it with another program, then evaluate it again.
                    self.check_program(150 + 2 * request, values_program('a', [999]), self.rows([999]), [0.] * 4)
            stats = self.stats()
            if self.patching:
                self.assertGreater(stats['predicate_patched'], 0)
            self.assertGreater(stats['predicate_evictions'], 0)
        finally:
            self.stop()

    def test_delete_drops_bitmap_programs(self):
        """A result that reads a pushed bitmap is dropped by a delete; one over the columns alone is patched."""
        pushed = [row for row in range(ROWS) if self.groups[row] < 500]
        bitmap = struct.pack('<%dI' % ((ROWS + 31) // 32), *words_of(pushed, ROWS))
        both = struct.pack('<BQ', 4, 7) + values_program('a', [1, 2, 3, 600]) + b'\x05'
        columns = values_program('a', [1, 2, 3, 600])
        query = [1., 0., -1., 2.]
        self.start()
        try:
            self.assertEqual(self.call(put_bitmap_frame(2, 'docs', 7, bitmap))[2], 0)
            self.check_program(3, both, self.rows([1, 2, 3]), query)
            self.check_program(5, columns, self.rows([1, 2, 3, 600]), query)
            before = self.stats()
            self.assertEqual(self.delete(7, [1001, 1600]), (2, 2))
            self.check_program(8, both, [row for row in self.rows([1, 2, 3]) if row != 1001], query)
            self.check_program(10, columns, [row for row in self.rows([1, 2, 3, 600]) if row not in (1001, 1600)],
                               query)
            after = self.stats()
            if self.patching:
                self.assertEqual(after['predicate_evaluations'] - before['predicate_evaluations'], 1)
                self.assertEqual((after['predicate_patched'], after['predicate_dropped']), (1, 1))
            else:
                self.assertEqual(after['predicate_evaluations'] - before['predicate_evaluations'], 2)
                self.assertEqual((after['predicate_patched'], after['predicate_dropped']), (0, 2))
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


class PredicateCachePatchOffTests(PredicateCacheTests):
    """The same with --delete-patch off: deletes drop cached results; every reply stays exact."""
    options = ('--delete-patch', 'off')


def collection_frame(request, names, query, k, program):
    """Operation 25: one operation 22 search per named index, answered as one."""
    payload = struct.pack('<I', len(names))
    for name in names:
        payload += struct.pack('<I', len(name)) + name.encode() + struct.pack('<Q', 0)
    payload += struct.pack('<IIII', len(query), k, 0, len(program)) + program
    payload += struct.pack('<%df' % len(query), *query)
    return frame(COLLECTION, request, payload)


class ResolverTests(PredicateFixture, unittest.TestCase):
    """--predicate-resolvers: a search whose predicate misses is evaluated off its reader."""
    backend = 'mock-gpu'

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.write_text()
        # A copy with a second attribute b (row % 997): "a in a window AND b <> v" leaves out a few rows.
        cls.second = [row % 997 for row in range(ROWS)]
        cls.residual = os.path.join(cls.directory.name, 'residual')
        shutil.copytree(cls.path, cls.residual)
        with open(os.path.join(cls.residual, 'attr-b.i32'), 'wb') as out:
            out.write(struct.pack('<%di' % ROWS, *cls.second))
        with open(os.path.join(cls.residual, 'manifest'), 'a') as out:
            out.write('attribute b attr-b.i32\n')
    # The mock's evaluation delay needs --gpu-predicates on; it then falls back to the host evaluator.
    DELAY = {'ANN_MOCK_GPU_EVALUATE_US': '300000'}

    def start(self, *options, env=None):
        super().start('--gpu-predicates', 'on', *options, env=env)
        self.connection.settimeout(10)

    def check_hits(self, payload, allowed, query, k, copies=1):
        """A search reply against an exact scan over allowed rows, in each of copies indexes."""
        count, _, eligible = struct.unpack_from('<IIQ', payload)
        self.assertEqual(eligible, copies * len(allowed))
        hits = [struct.unpack_from('<qf', payload, 16 + i * 12) for i in range(count)]
        want = [(distance, key) for distance, key in exact(self.keys, self.vectors, query, k, set(allowed))
                for _ in range(copies)][:k]
        self.assertEqual([key for key, _ in hits], [key for _, key in want])
        for (_, distance), (expected, _) in zip(hits, want):
            self.assertAlmostEqual(distance, expected, places=4)

    def only_reply(self, data):
        """Send a frame; return its one reply, after checking nothing else follows it."""
        self.connection.sendall(data)
        reply = receive(self.connection)
        self.connection.sendall(frame(STATS, 99))
        operation, request, status, payload = receive(self.connection)
        self.assertEqual((operation, request, status), (STATS, 99, 0))
        return reply, json.loads(payload)

    def test_hit_not_behind_miss(self):
        """A miss's evaluation leaves its reader free: a hit sent after it on the connection answers first."""
        self.start('--predicate-resolvers', '4', env=self.DELAY)
        try:
            query = [0.5, -1., 2., 0.]
            hot, cold = self.window(100), self.window(700)
            self.search(1, hot, query)
            before = self.stats()
            started = time.monotonic()
            self.connection.sendall(search_predicate_frame(2, 'docs', query, 10, values_program('a', cold)) +
                                    search_predicate_frame(3, 'docs', query, 10, values_program('a', hot)))
            first = receive(self.connection)
            elapsed = time.monotonic() - started
            second = receive(self.connection)
            self.assertEqual((first[1], second[1]), (3, 2))
            self.assertLess(elapsed, 0.25)
            for (_, request, status, payload), groups in ((first, hot), (second, cold)):
                self.assertEqual(status, 0)
                self.check_hits(payload, self.rows(groups), query, 10)
            after = self.stats()
            self.assertEqual(after['predicate_resolved'] - before['predicate_resolved'], 1)
            self.assertEqual(after['predicate_evaluations'] - before['predicate_evaluations'], 1)
            self.assertEqual(after['predicate_inline'], 0)
        finally:
            self.stop()

    def test_concurrent_misses_evaluate_once(self):
        """Resolvers given the same new program at once: one evaluation, the others wait for it, all exact."""
        self.start('--predicate-resolvers', '8', env=self.DELAY)
        connections = [self.server.connect() for _ in range(6)]
        try:
            query = [2., 1., 0., -1.]
            groups = self.window(400)
            before = self.stats()
            for request, connection in enumerate(connections, 10):
                connection.settimeout(10)
                connection.sendall(search_predicate_frame(request, 'docs', query, 10, values_program('a', groups)))
            for request, connection in enumerate(connections, 10):
                _, got, status, payload = receive(connection)
                self.assertEqual((got, status), (request, 0))
                self.check_hits(payload, self.rows(groups), query, 10)
            after = self.stats()
            self.assertEqual(after['predicate_evaluations'] - before['predicate_evaluations'], 1)
            self.assertEqual(after['predicate_resolved'] - before['predicate_resolved'], len(connections))
        finally:
            for connection in connections:
                connection.close()
            self.stop()

    def test_one_reply_per_frame(self):
        """Operations 22, 25 and 30 resolved off the reader answer once each, errors included."""
        self.start('--predicate-resolvers', '4')
        try:
            _, _, status, info = self.call(load_frame(2, 'docs2', self.path))
            self.assertEqual(status, 0, info)
            query = [1., -2., 0.5, 3.]
            before = self.stats()
            for request, start in enumerate(range(0, 300, WINDOW), 10):
                groups = self.window(start)
                program = values_program('a', groups)
                (operation, got, status, payload), _ = self.only_reply(
                    search_predicate_frame(request, 'docs', query, 10, program))
                self.assertEqual((operation, got, status), (SEARCH_PREDICATE, request, 0))
                self.check_hits(payload, self.rows(groups), query, 10)
                program = values_program('a', [group + 1 for group in groups])
                (operation, got, status, payload), _ = self.only_reply(
                    collection_frame(request, ['docs', 'docs2'], query, 10, program))
                self.assertEqual((operation, got, status), (COLLECTION, request, 0))
                self.check_hits(payload, self.rows([group + 1 for group in groups]), query, 10, copies=2)
                program = values_program('a', [group + 2 for group in groups])
                (operation, got, status, payload), _ = self.only_reply(
                    search_batch_frame(request, 'docs', [query] * 3, 10, program))
                self.assertEqual((operation, got, status), (30, request, 0))
                self.assertEqual(struct.unpack_from('<I', payload)[0], 3)
                offset = 4
                for _ in range(3):
                    count = struct.unpack_from('<I', payload, offset)[0]
                    self.check_hits(payload[offset:offset + 16 + count * 12],
                                    self.rows([group + 2 for group in groups]), query, 10)
                    offset += 16 + count * 12
            # An unknown column fails in the evaluation, on the resolver.
            missing = values_program('missing', [1])
            for data, operation in ((search_predicate_frame(20, 'docs', query, 10, missing), SEARCH_PREDICATE),
                                    (collection_frame(20, ['docs', 'docs2'], query, 10, missing), COLLECTION),
                                    (search_batch_frame(20, 'docs', [query] * 2, 10, missing), 30)):
                (got, request, status, _), stats = self.only_reply(data)
                self.assertEqual((got, request), (operation, 20))
                self.assertNotIn(status, (0, 6))
            self.assertEqual(stats['predicate_resolved'] - before['predicate_resolved'], 3 * 4 + 4)
            self.assertEqual(stats['failed'], 0)
        finally:
            self.stop()

    def test_refused_collection_part_answers_once(self):
        """A resolved operation 25 part over the pending budget answers through its composite."""
        self.start('--predicate-resolvers', '2', '--max-pending-mb', '0')
        try:
            _, _, status, info = self.call(load_frame(2, 'docs2', self.path))
            self.assertEqual(status, 0, info)
            (operation, request, status, _), stats = self.only_reply(
                collection_frame(5, ['docs', 'docs2'], [0.] * DIMENSIONS, 10, values_program('a', [4, 5])))
            self.assertEqual((operation, request, status), (COLLECTION, 5, OVERLOADED))
            self.assertEqual(stats['predicate_resolved'], 2)
        finally:
            self.stop()

    def test_unremapped_text_index_handed_off(self):
        """A search on an index with a text attribute not yet remapped is handed off like any other.

        Remaps wait for every read of the arrays (attributeLock), so hand-off needs no exception
        for indexes a collection has not remapped yet.
        """
        query = [1., -1., 0.5, 2.]
        self.start('--predicate-resolvers', '4', env=self.DELAY)
        try:
            _, _, status, info = self.call(load_frame(2, 'text', self.text))
            self.assertEqual(status, 0, info)
            before = self.stats()
            (operation, request, status, payload), stats = self.only_reply(
                search_predicate_frame(3, 'text', query, 10, values_program('t', [3])))
            self.assertEqual((operation, request, status), (SEARCH_PREDICATE, 3, 0))
            # Local code 3: never remapped.
            self.check_hits(payload, [row for row in range(ROWS) if row % 4 == 3], query, 10)
            self.assertEqual(stats['predicate_resolved'] - before['predicate_resolved'], 1)
            self.assertEqual(stats['predicate_inline'], before['predicate_inline'])
            # Its result is cached, so a remap now is refused, as without resolvers.
            self.assertEqual(self.call(remap_frame(4, 'text', 't', [1, 2, 3, 0]))[2], INVALID)
            # Remapped before any search: also handed off. Collection code 3 is local code 2.
            _, _, status, info = self.call(load_frame(5, 'text2', self.text))
            self.assertEqual(status, 0, info)
            self.assertEqual(self.call(remap_frame(6, 'text2', 't', [1, 2, 3, 0]))[2], 0)
            (operation, request, status, payload), after = self.only_reply(
                search_predicate_frame(7, 'text2', query, 10, values_program('t', [3]), COLLECTION_CODES))
            self.assertEqual((operation, request, status), (SEARCH_PREDICATE, 7, 0))
            self.check_hits(payload, [row for row in range(ROWS) if row % 4 == 2], query, 10)
            self.assertEqual(after['predicate_resolved'] - stats['predicate_resolved'], 1)
            self.assertEqual(after['predicate_inline'], stats['predicate_inline'])
        finally:
            self.stop()

    def test_remap_after_hand_off_is_stale(self):
        """A frame handed off before a remap pipelined after it on the connection answers stale.

        Its program holds local codes; the resolver reaches it only after the remap (the one
        resolver is busy with another miss), and nothing it read is cached.
        """
        query = [1., -1., 0.5, 2.]
        self.start('--predicate-resolvers', '1', env=self.DELAY)
        try:
            _, _, status, info = self.call(load_frame(2, 'text', self.text))
            self.assertEqual(status, 0, info)
            before = self.stats()
            busy = self.window(300)
            program = values_program('t', [3])
            self.connection.sendall(search_predicate_frame(3, 'docs', query, 10, values_program('a', busy)) +
                                    search_predicate_frame(4, 'text', query, 10, program) +
                                    remap_frame(5, 'text', 't', [1, 2, 3, 0]))
            replies = {request: (operation, status, payload) for operation, request, status, payload in
                       (receive(self.connection) for _ in range(3))}
            self.assertEqual(replies[5][:2], (28, 0))
            self.assertEqual(replies[3][:2], (SEARCH_PREDICATE, 0))
            self.check_hits(replies[3][2], self.rows(busy), query, 10)
            self.assertEqual(replies[4][:2], (SEARCH_PREDICATE, STALE))
            # Nothing of the local-code read was kept: collection code 3 is local code 2.
            (operation, request, status, payload), stats = self.only_reply(
                search_predicate_frame(6, 'text', query, 10, program, COLLECTION_CODES))
            self.assertEqual((operation, request, status), (SEARCH_PREDICATE, 6, 0))
            self.check_hits(payload, [row for row in range(ROWS) if row % 4 == 2], query, 10)
            self.assertEqual(stats['predicate_resolved'] - before['predicate_resolved'], 3)
            self.assertEqual(stats['predicate_hits'], before['predicate_hits'])
        finally:
            self.stop()

    def test_hit_after_remap_is_stale(self):
        """A frame handed off before a remap does not take a result cached after it.

        The resolver reaches the local-code frame only after the remap and after operation 23
        cached the same program bytes over the collection codes: the lookup hits, and the
        frame still answers stale.
        """
        query = [1., -1., 0.5, 2.]
        self.start('--predicate-resolvers', '1', env=self.DELAY)
        try:
            _, _, status, info = self.call(load_frame(2, 'text', self.text))
            self.assertEqual(status, 0, info)
            before = self.stats()
            program = values_program('t', [3])
            # Two misses ahead of it keep the one resolver busy for twice the evaluation operation 23 takes.
            busy = [values_program('a', self.window(start)) for start in (300, 500)]
            self.connection.sendall(search_predicate_frame(3, 'docs', query, 10, busy[0]) +
                                    search_predicate_frame(7, 'docs', query, 10, busy[1]) +
                                    search_predicate_frame(4, 'text', query, 10, program) +
                                    remap_frame(5, 'text', 't', [1, 2, 3, 0]) +
                                    predicate_frame(6, 'text', program))
            replies = {request: (operation, status, payload) for operation, request, status, payload in
                       (receive(self.connection) for _ in range(5))}
            self.assertEqual(replies[5][:2], (28, 0))
            # Collection code 3 is local code 2.
            self.assertEqual(replies[6][:2], (PREDICATE, 0))
            self.assertEqual(struct.unpack_from('<Q', replies[6][2])[0], ROWS // 4)
            self.assertEqual((replies[3][:2], replies[7][:2]), ((SEARCH_PREDICATE, 0), (SEARCH_PREDICATE, 0)))
            self.assertEqual(replies[4][:2], (SEARCH_PREDICATE, STALE))
            stats = self.stats()
            # The frame's lookup found operation 23's entry; it is not counted, and nothing evaluated it.
            self.assertEqual(stats['predicate_hits'], before['predicate_hits'])
            self.assertEqual(stats['predicate_evaluations'] - before['predicate_evaluations'], 3)
        finally:
            self.stop()

    REMAP_VARIANTS = ((), ('--predicate-resolvers', '4'), ('--predicate-admission', 'second'),
                      ('--predicate-resolvers', '4', '--predicate-admission', 'second'))

    def remap_inside(self, options, env, log):
        """A search on text whose worker logs line log and waits; another connection remaps then.

        The search answers stale, nothing is cached, and the evaluation counts.
        """
        query = [1., -1., 0.5, 2.]
        self.start(*options, env=env)
        other = self.server.connect()
        other.settimeout(10)
        try:
            _, _, status, info = self.call(load_frame(2, 'text', self.text))
            self.assertEqual(status, 0, info)
            before = self.stats()
            program = values_program('t', [3])
            self.connection.sendall(search_predicate_frame(3, 'text', query, 10, program))
            self.wait_log(log)
            other.sendall(remap_frame(4, 'text', 't', [1, 2, 3, 0]))
            self.assertEqual(receive(other)[:3], (28, 4, 0))
            self.assertEqual(receive(self.connection)[:3], (SEARCH_PREDICATE, 3, STALE))
            # Collection code 3 is local code 2.
            (operation, request, status, payload), stats = self.only_reply(
                search_predicate_frame(5, 'text', query, 10, program, COLLECTION_CODES))
            self.assertEqual((operation, request, status), (SEARCH_PREDICATE, 5, 0))
            self.check_hits(payload, [row for row in range(ROWS) if row % 4 == 2], query, 10)
            self.assertEqual(stats['predicate_evaluations'] - before['predicate_evaluations'], 2)
            self.assertGreater(stats['predicate_eval_us'], before['predicate_eval_us'])
            self.assertEqual(stats['predicate_hits'], before['predicate_hits'])
        finally:
            other.close()
            self.stop()

    def test_remap_during_evaluation_is_not_cached(self):
        """A remap from another connection inside a search's evaluation: the search answers stale, nothing is cached.

        A later search with the same program bytes in collection codes must not find rows the
        evaluation read with the local codes. With and without resolvers, cached and transient.
        """
        for options in self.REMAP_VARIANTS:
            with self.subTest(options=options):
                self.remap_inside(options, {'ANN_MOCK_GPU_EVALUATED_US': '1000000'}, 'mock-gpu: evaluated text')

    def test_remap_after_canonical_form_is_stale(self):
        """A remap between a search's canonical form and its evaluation: the search answers stale.

        The canonical form was read from the local-code arrays; the evaluation reads the
        collection-code ones, so the generation must be the one read before the canonical form.
        """
        for options in self.REMAP_VARIANTS:
            with self.subTest(options=options):
                self.remap_inside(('--canonical-predicates', 'on') + options,
                                  {'ANN_MOCK_GPU_CANONICAL_US': '1000000'}, 'mock-gpu: canonical text')

    def test_reloaded_index_is_stale(self):
        """A queued frame whose index is unloaded and reloaded under its name answers stale, once."""
        self.start('--predicate-resolvers', '4', env=self.DELAY)
        other = self.server.connect()
        other.settimeout(10)
        try:
            _, _, status, info = self.call(load_frame(2, 'docs2', self.path))
            self.assertEqual(status, 0, info)
            query = [0.5, 2., -1., 1.]

            def resolved():
                other.sendall(frame(STATS, 98))
                _, _, status, payload = receive(other)
                self.assertEqual(status, 0)
                return json.loads(payload)['predicate_resolved']

            for request, operation, parts in ((10, SEARCH_PREDICATE, 1), (11, COLLECTION, 2)):
                program = values_program('a', self.window(100 * request))
                before = resolved()
                self.connection.sendall(search_predicate_frame(request, 'docs', query, 10, program)
                                        if operation == SEARCH_PREDICATE else
                                        collection_frame(request, ['docs', 'docs2'], query, 10, program))
                # Handed off (so the reader's own lookup found the index), then replaced while evaluated.
                deadline = time.monotonic() + 5
                while resolved() < before + parts:
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(0.01)
                other.sendall(frame(26, request, struct.pack('<I', 4) + b'docs'))
                self.assertEqual(receive(other)[2], 0)
                other.sendall(load_frame(request, 'docs', self.path))
                self.assertEqual(receive(other)[2], 0)
                # One reply each: the stale search's, and the statistics' sent after it.
                self.connection.sendall(frame(STATS, 99))
                replies = sorted(receive(self.connection)[:3] for _ in range(2))
                self.assertEqual(replies, sorted([(operation, request, STALE), (STATS, 99, 0)]))
                self.connection.sendall(frame(STATS, 99))
                self.assertEqual(receive(self.connection)[:3], (STATS, 99, 0))
            # The reloaded index answers.
            (_, request, status, payload), _ = self.only_reply(
                search_predicate_frame(12, 'docs', query, 10, values_program('a', self.window(0))))
            self.assertEqual((request, status), (12, 0))
            self.check_hits(payload, self.rows(self.window(0)), query, 10)
        finally:
            other.close()
            self.stop()

    def test_deferred_upload_once(self):
        """Concurrent hits on an entry not on the device yet: one upload, on a resolver; all search with it."""
        self.start('--predicate-resolvers', '8', env={'ANN_MOCK_GPU_UPLOAD_US': '300000'})
        connections = [self.server.connect() for _ in range(4)]
        try:
            groups = self.window(500)
            program = values_program('a', groups)
            # The miss: evaluated on a resolver and cached for the device's next lookup.
            (_, request, status, payload), stats = self.only_reply(
                search_predicate_frame(1, 'docs', [1., 1., 1., 1.], 10, program))
            self.assertEqual((request, status), (1, 0))
            self.check_hits(payload, self.rows(groups), [1., 1., 1., 1.], 10)
            self.assertEqual((stats['predicate_resolved'], stats['predicate_hits']), (1, 0))
            queries = [[float(i), -1., 0.5, 2.] for i in range(len(connections))]
            for request, (connection, query) in enumerate(zip(connections, queries), 10):
                connection.settimeout(10)
                connection.sendall(search_predicate_frame(request, 'docs', query, 10, program))
            for request, (connection, query) in enumerate(zip(connections, queries), 10):
                _, got, status, payload = receive(connection)
                self.assertEqual((got, status), (request, 0))
                self.check_hits(payload, self.rows(groups), query, 10)
            stats = self.stats()
            # Each hit was handed off (the reader neither uploads nor waits for one); nothing evaluated again.
            self.assertEqual(stats['predicate_resolved'], 1 + len(connections))
            self.assertEqual(stats['predicate_evaluations'], 1)
            # Each hit counted once, by the resolver that answered it; none by the readers.
            self.assertEqual(stats['predicate_hits'], len(connections))
        finally:
            for connection in connections:
                connection.close()
            log = self.stop()
        uploads = [line.split()[-1] for line in log.splitlines() if line.startswith('mock-gpu: uploaded docs ')]
        searched = [line.split()[-1] for line in log.splitlines() if line.startswith('mock-gpu: searched docs ')]
        self.assertEqual(len(uploads), 1, log)
        self.assertEqual(len(searched), 1 + len(connections), log)
        self.assertNotEqual(searched[0], uploads[0])
        self.assertEqual(searched[1:], uploads * len(connections))

    def test_residual_hits_counted_once(self):
        """--residual-masks frames handed off after finding a base: hits counted as without resolvers."""
        query = [0.5, 1., -2., 1.]
        groups = self.window(0)
        program = values_program('a', groups) + values_program('b', [5], negate=True) + b'\x05'
        allowed = [row for row in self.rows(groups) if self.second[row] != 5]
        self.assertLess(len(allowed), len(self.rows(groups)))
        # k 10: served from the base's entry with b = 5 masked. k 4000: the base can't keep k
        # rows once they go, so the whole program is looked up (and cached) too.
        frames = [(10, 0, 1), (10, 1, 1), (10, 1, 0), (4000, 1, 1), (4000, 2, 1), (4000, 2, 0)]
        counts = {}
        for resolvers in ('0', '4'):
            # The upload delay gives indexes device state (residual masks need it); entries upload deferred.
            self.start('--predicate-resolvers', resolvers, '--canonical-predicates', 'on', '--residual-masks', 'on',
                       '--exact-limit', '0', '--group-gather-limit', '4000000',
                       env={'ANN_MOCK_GPU_UPLOAD_US': '1000'})
            try:
                _, _, status, info = self.call(load_frame(2, 'residual', self.residual))
                self.assertEqual(status, 0, info)
                before = self.stats()
                counts[resolvers] = []
                for request, (k, hits, resolved) in enumerate(frames, 10):
                    (operation, got, status, payload), stats = self.only_reply(
                        search_predicate_frame(request, 'residual', query, k, program))
                    self.assertEqual((operation, got, status), (SEARCH_PREDICATE, request, 0))
                    count, _, eligible = struct.unpack_from('<IIQ', payload)
                    hits = [struct.unpack_from('<qf', payload, 16 + i * 12) for i in range(count)]
                    if k < len(allowed):
                        # Served from the base: eligible counts the base's rows, masked ones included.
                        self.assertEqual(eligible, len(self.rows(groups)))
                        want = exact(self.keys, self.vectors, query, k, set(allowed))
                        self.assertEqual([key for key, _ in hits], [key for _, key in want])
                    else:
                        # Every allowed row (distances tie, so not in a fixed order).
                        self.assertEqual((eligible, sorted(key for key, _ in hits)), (len(allowed), allowed))
                    counts[resolvers].append((stats['predicate_hits'] - before['predicate_hits'],
                                              stats['predicate_resolved'] - before['predicate_resolved']))
                    before = stats
                self.assertEqual(stats['predicate_residual'], 3)
                self.assertEqual(stats['predicate_evaluations'], 2)
            finally:
                self.stop()
        self.assertEqual(counts['0'], [(hits, 0) for _, hits, _ in frames])
        # A frame found deferred or missing on its reader is handed off; its hits are the resolver's.
        self.assertEqual(counts['4'], [(hits, resolved) for _, hits, resolved in frames])

    def test_batch_after_deletes(self):
        """Operation 30 resolved off the reader after deletes, with and without the snapshot flag."""
        self.start('--predicate-resolvers', '4')
        try:
            query = [1., 2., 3., 4.]
            groups = [3, 7, 11]
            rows = self.rows(groups)
            want = [key for _, key in exact(self.keys, self.vectors, query, 17, set(rows))]
            dead = want[:3]
            payload = struct.pack('<I', 4) + b'docs' + struct.pack('<I3q', 3, *dead)
            self.assertEqual(self.call(frame(29, 1, payload))[2], 0)
            alive = [row for row in rows if row not in dead]
            expected = [key for _, key in exact(self.keys, self.vectors, query, 17, set(alive))]
            # A snapshot keeps the dead rows (the coordinator applies visibility); the others leave them out.
            for request, flags, resolved in ((2, 8, 1), (3, 0, 2), (4, 8, 2), (5, 0, 2)):
                (operation, got, status, data), stats = self.only_reply(
                    search_batch_frame(request, 'docs', [query] * 5, 17, values_program('a', groups), flags))
                self.assertEqual((operation, got, status), (30, request, 0))
                self.assertEqual(struct.unpack_from('<I', data)[0], 5)
                offset = 4
                for _ in range(5):
                    count, _, eligible = struct.unpack_from('<IIQ', data, offset)
                    hits = [struct.unpack_from('<q', data, offset + 16 + i * 12)[0] for i in range(count)]
                    self.assertEqual((eligible, hits), (len(rows), want) if flags else (len(alive), expected))
                    offset += 16 + count * 12
                self.assertEqual(offset, len(data))
                # The first of each kind missed and was resolved off the reader; the second hit.
                self.assertEqual(stats['predicate_resolved'], resolved)
            self.assertEqual((stats['predicate_evaluations'], stats['predicate_inline']), (2, 0))
        finally:
            self.stop()

    def test_bitmap_program_keeps_serial_order(self):
        """A program over a pushed bitmap is evaluated before the next frame can replace that bitmap."""
        old = [row for row in range(ROWS) if self.groups[row] < 500]
        new = [row for row in range(ROWS) if self.groups[row] >= 500]
        program = struct.pack('<BQ', 4, 7) + values_program('a', [1, 2, 600]) + b'\x05'
        query = [1., 0., -1., 2.]
        self.start('--predicate-resolvers', '4', env=self.DELAY)
        try:
            words = '<%dI' % ((ROWS + 31) // 32)
            self.connection.sendall(put_bitmap_frame(2, 'docs', 7, struct.pack(words, *words_of(old, ROWS))) +
                                    search_predicate_frame(3, 'docs', query, 10, program) +
                                    put_bitmap_frame(4, 'docs', 7, struct.pack(words, *words_of(new, ROWS))))
            replies = {request: (status, payload) for _, request, status, payload in
                       (receive(self.connection) for _ in range(3))}
            self.assertEqual((replies[2][0], replies[3][0], replies[4][0]), (0, 0, 0))
            self.check_hits(replies[3][1], self.rows([1, 2]), query, 10)
            stats = self.stats()
            self.assertEqual((stats['predicate_resolved'], stats['predicate_inline']), (0, 1))
        finally:
            self.stop()



class RemapRaceTests(ResolverTests):
    """Searches by attribute match and by predicate on several connections
    while the index they name is remapped and same-shape copies load: every
    reply is an answer or STALE. Under a ThreadSanitizer build of the worker
    this also exercises the attribute lock in every reader. Only this test
    runs here; the inherited ones run in ResolverTests."""

    test_batch_after_deletes = None
    test_bitmap_program_keeps_serial_order = None
    test_concurrent_misses_evaluate_once = None
    test_deferred_upload_once = None
    test_hit_after_remap_is_stale = None
    test_hit_not_behind_miss = None
    test_one_reply_per_frame = None
    test_refused_collection_part_answers_once = None
    test_reloaded_index_is_stale = None
    test_remap_after_canonical_form_is_stale = None
    test_remap_after_hand_off_is_stale = None
    test_remap_during_evaluation_is_not_cached = None
    test_residual_hits_counted_once = None
    test_unremapped_text_index_handed_off = None

    def test_searches_during_remaps(self):
        self.start('--predicate-resolvers', '16', '--shared-predicate-mb', '64',
                   '--predicate-admission', 'second', '--canonical-predicates', 'off')
        query = [1., -1., .5, 2.]
        stop = threading.Event()
        current = {'name': 'text0'}
        bad = []
        unique = [0]
        lock = threading.Lock()

        def match(request, name):
            p = struct.pack('<I', len(name)) + name.encode() + struct.pack('<III', len(query), 10, 1)
            p += b't' + struct.pack('<I', 2) + struct.pack('<2i', 1, 3) + struct.pack('<4f', *query)
            return frame(17, request, p)

        def spam(kind):
            connection = self.server.connect()
            connection.settimeout(30)
            i = 0
            while not stop.is_set():
                name = current['name']
                i += 1
                if kind == 'match':
                    connection.sendall(match(i, name))
                else:
                    with lock:
                        unique[0] += 1
                        value = 100000 + unique[0]
                    connection.sendall(search_predicate_frame(i, name, query, 10,
                                                              values_program('t', [i % 4, value])))
                status = receive(connection)[2]
                if status not in (0, 2, STALE):  # 2: an index not loaded yet
                    bad.append((kind, status))
            connection.close()

        other = self.server.connect()
        other.settimeout(30)
        _, _, status, info = self.call(load_frame(2, 'text0', self.text))
        self.assertEqual(status, 0, info)
        threads = [threading.Thread(target=spam, args=(kind,)) for kind in ('match', 'match', 'pred', 'pred')]
        for thread in threads:
            thread.start()
        remapped = 0
        for r in range(8):
            other.sendall(load_frame(3, 'text%d' % (r + 1), self.text))
            self.assertEqual(receive(other)[2], 0)
            time.sleep(0.02)
            other.sendall(remap_frame(4, 'text%d' % r, 't', [1, 2, 3, 0]))
            remapped += receive(other)[2] == 0
            current['name'] = 'text%d' % (r + 1)
            time.sleep(0.02)
        stop.set()
        for thread in threads:
            thread.join()
        other.close()
        self.stop()
        self.assertFalse(bad)
        self.assertGreater(remapped, 0)

class RowSetTests(PredicateFixture, unittest.TestCase):
    """--shared-predicate-mb: indexes holding the same filter inputs form a row set."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # The same documents with the last key changed, and with two rows' groups swapped (equal counts).
        cls.keyed = os.path.join(cls.directory.name, 'keyed')
        shutil.copytree(cls.path, cls.keyed)
        with open(os.path.join(cls.keyed, 'keys.i64'), 'r+b') as out:
            out.seek((ROWS - 1) * 8)
            out.write(struct.pack('<q', ROWS + 5))
        cls.swapped = os.path.join(cls.directory.name, 'swapped')
        shutil.copytree(cls.path, cls.swapped)
        groups = list(cls.groups)
        groups[0], groups[1] = groups[1], groups[0]
        with open(os.path.join(cls.swapped, 'attr-a.i32'), 'wb') as out:
            out.write(struct.pack('<%di' % ROWS, *groups))
        cls.swapped_groups = groups
        cls.write_text()
        # A multi-valued attribute tags: NULL, empty and repeated elements included.
        cls.tags = [None if row % 17 == 0 else [] if row % 13 == 0 else [row % 7, row % 11, row % 7]
                    for row in range(ROWS)]
        cls.multi = os.path.join(cls.directory.name, 'multi')
        shutil.copytree(cls.path, cls.multi)
        elements, offsets = [], [0]
        for row in cls.tags:
            elements += [-2 ** 31] if row is None else row
            offsets.append(len(elements))
        with open(os.path.join(cls.multi, 'attr-tags.i32'), 'wb') as out:
            out.write(struct.pack('<%di' % len(elements), *elements))
        with open(os.path.join(cls.multi, 'attr-tags.offsets'), 'wb') as out:
            out.write(struct.pack('<%dI' % len(offsets), *offsets))
        with open(os.path.join(cls.multi, 'manifest'), 'a') as out:
            out.write('attribute tags attr-tags.i32\nmultivalued tags attr-tags.offsets\n')

    def start(self, *options, names=('g1', 'g2', 'g3'), paths=None, env=None):
        self.server = Server('--backend', self.backend, *options, env=env, executable=self.worker)
        self.connection = self.server.connect()
        self.log = ''
        for request, name in enumerate(names, 1):
            path = paths[name] if paths and name in paths else self.path
            _, _, status, info = self.call(load_frame(request, name, path))
            self.assertEqual(status, 0, info)

    def words(self, request, name, program, rows):
        """Operation 23 on name: its count and words are exactly rows."""
        _, _, status, payload = self.call(predicate_frame(request, name, program))
        self.assertEqual(status, 0)
        eligible, = struct.unpack_from('<Q', payload)
        self.assertEqual(eligible, len(rows))
        self.assertEqual(list(struct.unpack_from('<%dI' % ((ROWS + 31) // 32), payload, 8)), words_of(rows, ROWS))

    def search_on(self, request, name, program, query, k=10):
        _, _, status, payload = self.call(search_predicate_frame(request, name, query, k, program))
        self.assertEqual(status, 0)
        return payload

    def delete_on(self, request, name, rows):
        payload = struct.pack('<I', len(name)) + name.encode() + struct.pack('<I%dq' % len(rows), len(rows), *rows)
        _, _, status, _ = self.call(frame(29, request, payload))
        self.assertEqual(status, 0)

    def shared(self, *names):
        for name in names:
            self.wait_log('ann index %s: filter results shared with' % name)

    def test_identical_indexes_form_a_row_set(self):
        self.start('--shared-predicate-mb', '64')
        try:
            self.shared('g2', 'g3')
            self.assertNotIn('not shared', self.log)
        finally:
            self.stop()

    def test_row_set_evaluates_once(self):
        """One evaluation for the three indexes; the others copy it; identical replies."""
        self.start('--shared-predicate-mb', '64')
        try:
            self.shared('g2', 'g3')
            before = self.stats()
            groups = self.window(300)
            for request, name in enumerate(('g1', 'g2', 'g3'), 10):
                self.words(request, name, values_program('a', groups), self.rows(groups))
            query = [0.5, 1., -2., 3.]
            program = values_program('a', self.window(600))
            replies = [self.search_on(request, name, program, query) for request, name in
                       enumerate(('g2', 'g1', 'g3'), 20)]
            self.assertEqual(replies, replies[:1] * len(replies))
            count, _, eligible = struct.unpack_from('<IIQ', replies[0])
            allowed = set(self.rows(self.window(600)))
            self.assertEqual(eligible, len(allowed))
            self.assertEqual([struct.unpack_from('<q', replies[0], 16 + i * 12)[0] for i in range(count)],
                             [key for _, key in exact(self.keys, self.vectors, query, 10, allowed)])
            after = self.stats()
            # predicate_evaluations counts entries built; the shared ones were copied, not evaluated.
            self.assertEqual(after['predicate_evaluations'] - before['predicate_evaluations'], 6)
            self.assertEqual(after['predicate_shared'] - before['predicate_shared'], 4)
            self.assertEqual(after['predicate_shared_entries'], 2)
            self.assertEqual(after['predicate_shared_bytes'], 2 * 4 * ((ROWS + 31) // 32))
        finally:
            self.stop()

    def test_concurrent_misses_across_a_row_set(self):
        """Searches of one new program on every member at once: evaluated once, the rest wait and copy."""
        self.backend = 'mock-gpu'
        self.start('--shared-predicate-mb', '64', '--gpu-predicates', 'on', '--predicate-resolvers', '8',
                   env={'ANN_MOCK_GPU_EVALUATE_US': '300000'})
        connections = [self.server.connect() for _ in range(6)]
        try:
            self.shared('g2', 'g3')
            query = [2., 1., 0., -1.]
            groups = self.window(400)
            before = self.stats()
            for request, connection in enumerate(connections, 10):
                connection.settimeout(10)
                connection.sendall(search_predicate_frame(request, 'g%d' % (request % 3 + 1), query, 10,
                                                          values_program('a', groups)))
            replies = []
            for request, connection in enumerate(connections, 10):
                _, got, status, payload = receive(connection)
                self.assertEqual((got, status), (request, 0))
                replies.append(payload)
            self.assertEqual(replies, replies[:1] * len(replies))
            after = self.stats()
            self.assertEqual(after['predicate_evaluations'] - before['predicate_evaluations'], 3)
            self.assertEqual(after['predicate_shared'] - before['predicate_shared'], 2)
        finally:
            for connection in connections:
                connection.close()
            self.stop()

    def test_tombstones_stay_per_index(self):
        """A delete on one member changes only its own results, whichever member evaluated first."""
        self.start('--shared-predicate-mb', '64')
        try:
            self.shared('g2', 'g3')
            query = [1., -1., 2., 0.]
            hot = values_program('a', self.window(0))
            reply = self.search_on(5, 'g2', hot, query)
            dead = self.rows([3, 50, 750])[:6]
            self.delete_on(6, 'g1', dead)
            self.assertEqual(self.search_on(7, 'g2', hot, query), reply)
            for request, (first, second) in enumerate((('g1', 'g2'), ('g2', 'g1')), 10):
                groups = [3, 50, 750] + self.window(100 * (request - 9))
                rows = self.rows(groups)
                for name in (first, second):
                    alive = [row for row in rows if name != 'g1' or row not in dead]
                    self.words(request, name, values_program('a', groups), alive)
            # A snapshot search on g1 keeps the dead rows of the result g3 kept.
            groups = [3, 50, 750]
            program = values_program('a', groups)
            self.words(30, 'g3', program, self.rows(groups))
            before = self.stats()
            _, _, status, data = self.call(search_batch_frame(31, 'g1', [query], 10, program, 8))
            self.assertEqual(status, 0)
            self.assertEqual(struct.unpack_from('<Q', data, 12)[0], len(self.rows(groups)))
            self.assertEqual(self.stats()['predicate_shared'] - before['predicate_shared'], 1)
        finally:
            self.stop()

    def test_different_inputs_do_not_share(self):
        self.start('--shared-predicate-mb', '64', names=('g1', 'keyed', 'swapped'),
                   paths={'keyed': self.keyed, 'swapped': self.swapped})
        try:
            for name in ('keyed', 'swapped'):
                self.wait_log('ann index %s: filter results not shared' % name)
            groups = [0, 1, 2] + self.window(500)
            self.words(10, 'g1', values_program('a', groups), self.rows(groups))
            self.words(11, 'keyed', values_program('a', groups), self.rows(groups))
            self.words(12, 'swapped', values_program('a', [0]), [1] + self.rows([0])[1:])
            self.assertEqual(self.stats()['predicate_shared'], 0)
        finally:
            self.stop()

    def test_remapped_index_stops_sharing(self):
        """A remap leaves the row set: the same program bytes then select other rows there."""
        self.start('--shared-predicate-mb', '64', names=('t1', 't2'), paths={'t1': self.text, 't2': self.text})
        try:
            self.shared('t2')
            name, attribute = b't2', b't'
            payload = (struct.pack('<I', len(name)) + name + struct.pack('<I', len(attribute)) + attribute +
                       struct.pack('<5I', 4, 1, 2, 3, 0))
            _, _, status, _ = self.call(frame(28, 2, payload))
            self.assertEqual(status, 0)
            for request, (code, order) in enumerate(((1, ('t1', 't2')), (2, ('t2', 't1'))), 10):
                program = values_program('t', [code]) + values_program('a', self.window(0)) + b'\x05'
                for name in order:
                    old = code if name == 't1' else (code - 1) % 4
                    rows = [row for row in self.rows(self.window(0)) if row % 4 == old]
                    self.words(request, name, program, rows)
            stats = self.stats()
            self.assertEqual(stats['predicate_shared'], 0)
            # t1 is the only member left attached: nobody could copy what it kept, so it keeps nothing.
            self.assertEqual(stats['predicate_shared_entries'], 0)
        finally:
            self.stop()

    def test_different_inputs_do_not_join(self):
        """One key or one posting apart: compared, not joined."""
        self.start('--shared-predicate-mb', '64', names=('g1', 'keyed', 'swapped'),
                   paths={'keyed': self.keyed, 'swapped': self.swapped})
        try:
            for name in ('keyed', 'swapped'):
                self.wait_log('ann index %s: filter results not shared' % name)
            self.assertNotIn('shared with', self.log)
        finally:
            self.stop()

    def test_bitmap_programs_not_shared(self):
        """A program reading a pushed bitmap is each index's own: same id, different bitmaps."""
        self.start('--shared-predicate-mb', '64', names=('g1', 'g2'))
        try:
            self.shared('g2')
            words = '<%dI' % ((ROWS + 31) // 32)
            for request, (name, groups) in enumerate((('g1', [1, 2]), ('g2', [600])), 2):
                _, _, status, _ = self.call(put_bitmap_frame(request, name, 7, struct.pack(
                    words, *words_of(self.rows(groups), ROWS))))
                self.assertEqual(status, 0)
            program = struct.pack('<BQ', 4, 7) + values_program('a', [1, 2, 600]) + b'\x05'
            self.words(10, 'g1', program, self.rows([1, 2]))
            self.words(11, 'g2', program, self.rows([600]))
            stats = self.stats()
            self.assertEqual((stats['predicate_shared'], stats['predicate_shared_entries']), (0, 0))
        finally:
            self.stop()

    def test_budget_keeps_the_most_recent(self):
        """--shared-predicate-mb 1 keeps 209 results of 5,000 bytes, the least recently used dropped."""
        self.start('--shared-predicate-mb', '1', names=('g1', 'g2'))
        try:
            self.shared('g2')
            for group in range(215):
                self.words(10 + group, 'g1', values_program('a', [group]), self.rows([group]))
            stats = self.stats()
            self.assertEqual((stats['predicate_shared_entries'], stats['predicate_shared_bytes']), (209, 1045000))
            self.words(300, 'g2', values_program('a', [214]), self.rows([214]))
            self.assertEqual(self.stats()['predicate_shared'], 1)
            self.words(301, 'g2', values_program('a', [0]), self.rows([0]))
            stats = self.stats()
            self.assertEqual(stats['predicate_shared'], 1)
            self.assertEqual(stats['predicate_shared_entries'], 209)
        finally:
            self.stop()

    def test_unload_keeps_sharing(self):
        """Members unloaded one by one: the others keep sharing; the last one frees the results."""
        self.start('--shared-predicate-mb', '64')
        try:
            self.shared('g2', 'g3')
            groups = self.window(200)
            self.words(10, 'g1', values_program('a', groups), self.rows(groups))
            for request, (gone, name) in enumerate((('g1', 'g2'), ('g2', 'g3')), 20):
                _, _, status, _ = self.call(frame(26, request, struct.pack('<I', 2) + gone.encode()))
                self.assertEqual(status, 0)
                before = self.stats()['predicate_shared']
                self.words(30 + request, name, values_program('a', groups), self.rows(groups))
                self.assertEqual(self.stats()['predicate_shared'], before + 1)
            _, _, status, _ = self.call(frame(26, 40, struct.pack('<I', 2) + b'g3'))
            self.assertEqual(status, 0)
            stats = self.stats()
            self.assertEqual((stats['predicate_shared_entries'], stats['predicate_shared_bytes']), (0, 0))
        finally:
            self.stop()

    def test_alone_keeps_nothing(self):
        """A member whose partner is unloaded still copies what was kept, but stops keeping new results."""
        self.start('--shared-predicate-mb', '64', names=('g1', 'g2'))
        try:
            self.shared('g2')
            kept = self.window(200)
            self.words(10, 'g1', values_program('a', kept), self.rows(kept))
            self.assertEqual(self.stats()['predicate_shared_entries'], 1)
            _, _, status, _ = self.call(frame(26, 11, struct.pack('<I', 2) + b'g1'))
            self.assertEqual(status, 0)
            for request, start in enumerate(range(300, 800, WINDOW), 20):
                self.words(request, 'g2', values_program('a', self.window(start)), self.rows(self.window(start)))
            self.words(30, 'g2', values_program('a', kept), self.rows(kept))
            stats = self.stats()
            self.assertEqual((stats['predicate_shared_entries'], stats['predicate_shared']), (1, 1))
        finally:
            self.stop()

    def test_multivalued_row_sets(self):
        """OVERLAP over a multi-valued column (NULL and empty arrays included) is shared exactly."""
        self.start('--shared-predicate-mb', '64', names=('m1', 'm2'), paths={'m1': self.multi, 'm2': self.multi})
        try:
            self.shared('m2')
            for request, negate in enumerate((False, True), 10):
                program = (struct.pack('<BB', 10, 4) + b'tags' + struct.pack('<BI2i', 1 if negate else 0, 2, 3, 9))
                rows = [row for row, tags in enumerate(self.tags) if tags is not None and
                        bool({3, 9} & set(tags)) != negate]
                self.words(request, 'm1', program, rows)
                self.words(request + 10, 'm2', program, rows)
            self.assertEqual(self.stats()['predicate_shared'], 2)
        finally:
            self.stop()

    def test_off_compares_nothing(self):
        self.start()
        try:
            for request, name in enumerate(('g1', 'g2', 'g3'), 10):
                self.evaluate(request, self.window(0), name)
            stats = self.stats()
            self.assertEqual(stats['predicate_evaluations'], 3)
            self.assertEqual((stats['predicate_shared'], stats['predicate_shared_entries'],
                              stats['predicate_shared_bytes']), (0, 0, 0))
        finally:
            log = self.stop()
        self.assertNotIn('filter results', log)


if __name__ == '__main__':
    unittest.main()
