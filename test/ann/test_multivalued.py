"""Multi-valued attributes on the worker: the "multivalued" manifest line, its
row offsets file, and OVERLAP (op 10) predicate leaves.

Run with the CPU worker (make -C ann/server ann-worker); ANN_INDEX_WORKER selects
another executable. Results are checked against the same predicates
evaluated here over the rows' arrays, so they match PostgreSQL's &&.
"""

import os
import random
import struct
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_server import ROOT, Server, receive  # noqa: E402
from test_index import decode_hits, exact, load_frame, match_frame, write_index  # noqa: E402
from test_predicate_cache import predicate_frame, search_predicate_frame, words_of  # noqa: E402

VALUES = 1
RANGE = 2
NULL = 3
AND = 5
OR = 6
NOT = 7
OVERLAP = 10
INT32_MIN = -2 ** 31
INVALID = 1

ROWS = 3000
DIMENSIONS = 4


def leaf(op, column, negate=False, values=None, low=None, high=None):
    out = struct.pack('<BB', op, len(column)) + column.encode() + struct.pack('<B', 1 if negate else 0)
    if op in (VALUES, OVERLAP):
        out += struct.pack('<I%di' % len(values), len(values), *values)
    elif op == RANGE:
        out += struct.pack('<ii', low, high)
    return out


def write_multi(directory, keys, columns, sort_keys=True):
    """Add multi-valued columns (one list or None per row, in the order of keys) to an index directory."""
    order = sorted(range(len(keys)), key=lambda i: keys[i]) if sort_keys else range(len(keys))
    lines = []
    for name, rows in columns.items():
        elements, offsets = [], [0]
        for i in order:
            elements += [INT32_MIN] if rows[i] is None else rows[i]
            offsets.append(len(elements))
        with open(os.path.join(directory, f'attr-{name}.i32'), 'wb') as out:
            out.write(struct.pack('<%di' % len(elements), *elements))
        with open(os.path.join(directory, f'attr-{name}.offsets'), 'wb') as out:
            out.write(struct.pack('<%dI' % len(offsets), *offsets))
        lines += [f'attribute {name} attr-{name}.i32', f'multivalued {name} attr-{name}.offsets']
    with open(os.path.join(directory, 'manifest'), 'a') as out:
        out.write('\n'.join(lines) + '\n')


def tags_of(rng, rows):
    """NULL, '{}', '{3,3}', '{5,1}', rare codes 100 and 101 (sharing rows), then about 5 of 0..40."""
    tags = [None, [], [3, 3], [5, 1]]
    tags += [[100, 101, 100] for _ in range(15)] + [[101] for _ in range(5)]
    while len(tags) < rows:
        roll = rng.random()
        tags.append(None if roll < 0.03 else [] if roll < 0.06 else
                    [rng.randrange(41) for _ in range(rng.randint(1, 9))])
    rng.shuffle(tags)
    return tags


class MultivaluedTests(unittest.TestCase):
    worker = os.path.join(ROOT, 'ann', 'server', os.environ.get('ANN_INDEX_WORKER', 'ann-worker'))
    backend = os.environ.get('ANN_INDEX_BACKEND', 'cpu-exact')

    @classmethod
    def setUpClass(cls):
        if not os.environ.get('ANN_INDEX_WORKER'):
            subprocess.run(['make', '-C', os.path.join(ROOT, 'ann', 'server'), 'ann-worker'], check=True,
                           stdout=subprocess.DEVNULL)
        rng = random.Random(11)
        # Keys 0 .. ROWS - 1, so a row's ordinal is its key.
        cls.keys = list(range(ROWS))
        cls.vectors = [[round(rng.uniform(-3, 3), 3) for _ in range(DIMENSIONS)] for _ in range(ROWS)]
        cls.tenant = [rng.randrange(6) for _ in range(ROWS)]
        cls.tags = tags_of(rng, ROWS)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        # Prebuilt bitsets for codes with more than 300 rows, as on a real segment.
        self.server = Server('--backend', self.backend, '--bitset-min', '300', executable=self.worker)
        self.connection = self.server.connect()

    def tearDown(self):
        self.connection.close()
        self.server.close()
        self.directory.cleanup()

    def call(self, data):
        self.connection.sendall(data)
        return receive(self.connection)

    def index(self, name='docs', shards=3, tags=None):
        path = os.path.join(self.directory.name, name)
        write_index(path, self.keys, self.vectors, shards=shards, attributes={'tenant': self.tenant})
        write_multi(path, self.keys, {'tags': self.tags if tags is None else tags})
        return path

    def load(self, request, name, path):
        return self.call(load_frame(request, name, path))[2]

    def overlap(self, values, negate=False):
        wanted = set(values) - {INT32_MIN}
        return {row for row, tags in enumerate(self.tags)
                if tags is not None and bool(wanted & set(tags)) != negate}

    def evaluate(self, request, program):
        status, payload = self.call(predicate_frame(request, 'docs', program))[2:]
        self.assertEqual(status, 0)
        eligible, = struct.unpack_from('<Q', payload)
        words = list(struct.unpack_from('<%dI' % ((ROWS + 31) // 32), payload, 8))
        rows = {row for row in range(ROWS) if words[row // 32] >> (row % 32) & 1}
        self.assertEqual(eligible, len(rows))
        return rows

    def test_overlap_matches_arrays(self):
        self.assertEqual(self.load(1, 'docs', self.index()), 0)
        null = {row for row, tags in enumerate(self.tags) if tags is None}
        cases = [
            (leaf(OVERLAP, 'tags', values=[1, 5]), self.overlap([1, 5])),
            (leaf(OVERLAP, 'tags', values=[7]), self.overlap([7])),
            (leaf(OVERLAP, 'tags', negate=True, values=[1]), self.overlap([1], negate=True)),
            (leaf(OVERLAP, 'tags', values=[]), set()),
            (leaf(OVERLAP, 'tags', negate=True, values=[]), self.overlap([], negate=True)),
            (leaf(OVERLAP, 'tags', values=[INT32_MIN, 1]), self.overlap([1])),
            (leaf(OVERLAP, 'tags', values=[3]), self.overlap([3])),
            # Rare codes sharing rows: the list form, where one row's two postings must count once.
            (leaf(OVERLAP, 'tags', values=[100, 101]), self.overlap([100, 101])),
            (leaf(OVERLAP, 'tags', values=[101, 999]), self.overlap([101])),
            (leaf(OVERLAP, 'tags', values=list(range(0, 41, 3))), self.overlap(range(0, 41, 3))),
            (leaf(OVERLAP, 'tags', negate=True, values=list(range(41))), self.overlap(range(41), negate=True)),
            (leaf(NULL, 'tags'), null),
            (leaf(NULL, 'tags', negate=True), set(range(ROWS)) - null),
            (leaf(VALUES, 'tenant', values=[3]) + leaf(OVERLAP, 'tags', values=[7]) + bytes([OR]),
             {row for row in range(ROWS) if self.tenant[row] == 3} | self.overlap([7])),
            (leaf(VALUES, 'tenant', values=[2]) + leaf(OVERLAP, 'tags', values=[100, 101]) + bytes([AND]),
             {row for row in range(ROWS) if self.tenant[row] == 2} & self.overlap([100, 101])),
            (leaf(OVERLAP, 'tags', values=[1]) + bytes([NOT]), set(range(ROWS)) - self.overlap([1])),
        ]
        self.assertEqual(len(self.overlap([100, 101])), 20)
        for request, (program, want) in enumerate(cases, 2):
            self.assertEqual(self.evaluate(request, program), want, request)

    def test_search_and_match_return_each_row_once(self):
        self.assertEqual(self.load(1, 'docs', self.index()), 0)
        rng = random.Random(12)
        for request, values in enumerate([[100, 101], [1, 5], [7], list(range(20))], 2):
            query = [rng.uniform(-3, 3) for _ in range(DIMENSIONS)]
            allowed = self.overlap(values)
            status, payload = self.call(search_predicate_frame(request * 2, 'docs', query, 50,
                                                               leaf(OVERLAP, 'tags', values=values)))[2:]
            self.assertEqual(status, 0)
            count, _, eligible = struct.unpack_from('<IIQ', payload)
            hits = [struct.unpack_from('<q', payload, 16 + i * 12)[0] for i in range(count)]
            self.assertEqual(eligible, len(allowed))
            want = [key for _, key in exact(self.keys, self.vectors, query, 50, allowed)]
            self.assertEqual(sorted(hits), sorted(want), values)
            # Operation 17 resolves "tags IN (values)" from the same postings.
            status, payload = self.call(match_frame(request * 2 + 1, 'docs', query, 50, 'tags', values))[2:]
            self.assertEqual(status, 0)
            hits = [key for key, _ in decode_hits(payload)[1]]
            self.assertEqual(len(hits), len(set(hits)))
            self.assertEqual(sorted(hits), sorted(want), values)

    def test_leaf_kind_must_match_the_column(self):
        self.assertEqual(self.load(1, 'docs', self.index()), 0)
        for request, program in enumerate([leaf(VALUES, 'tags', values=[1]),
                                           leaf(RANGE, 'tags', low=0, high=10),
                                           leaf(OVERLAP, 'tenant', values=[1])], 2):
            self.assertEqual(self.call(predicate_frame(request, 'docs', program))[2], INVALID)
        self.assertEqual(self.evaluate(10, leaf(VALUES, 'tenant', values=[1])),
                         {row for row in range(ROWS) if self.tenant[row] == 1})

    def test_one_shard_and_empty_arrays(self):
        self.assertEqual(self.load(1, 'one', self.index('one', shards=1)), 0)
        self.assertEqual(self.load(2, 'empty', self.index('empty', tags=[[] for _ in range(ROWS)])), 0)
        program = leaf(OVERLAP, 'tags', values=[1, 5])
        self.assertEqual(self.call(predicate_frame(3, 'one', program))[3][8:],
                         struct.pack('<%dI' % ((ROWS + 31) // 32), *words_of(self.overlap([1, 5]), ROWS)))
        status, payload = self.call(predicate_frame(4, 'empty', program))[2:]
        self.assertEqual((status, struct.unpack_from('<Q', payload)[0]), (0, 0))
        status, payload = self.call(predicate_frame(5, 'empty', leaf(OVERLAP, 'tags', negate=True,
                                                                     values=[1])))[2:]
        self.assertEqual((status, struct.unpack_from('<Q', payload)[0]), (0, ROWS))

    def test_invalid_files_are_refused(self):
        def offsets_of(path):
            with open(os.path.join(path, 'attr-tags.offsets'), 'rb') as data:
                return list(struct.unpack('<%dI' % (ROWS + 1), data.read()))

        def rewrite(path, name, data):
            with open(os.path.join(path, name), 'wb') as out:
                out.write(data)

        cases = []
        path = self.index('short')
        with open(os.path.join(path, 'attr-tags.offsets'), 'r+b') as out:
            out.truncate(ROWS * 4)
        cases.append(('short', path))
        path = self.index('decreasing')
        offsets = offsets_of(path)
        row = next(i for i in range(1, ROWS) if offsets[i + 1] > offsets[i])
        offsets[row] = offsets[row + 1] + 1
        rewrite(path, 'attr-tags.offsets', struct.pack('<%dI' % len(offsets), *offsets))
        cases.append(('decreasing', path))
        path = self.index('start')
        offsets = offsets_of(path)
        offsets[0] = 1
        rewrite(path, 'attr-tags.offsets', struct.pack('<%dI' % len(offsets), *offsets))
        cases.append(('start', path))
        path = self.index('end')
        with open(os.path.join(path, 'attr-tags.i32'), 'ab') as out:
            out.write(struct.pack('<i', 4))
        cases.append(('end', path))
        tags = [list(row) if row else row for row in self.tags]
        tags[next(i for i, row in enumerate(tags) if row and len(row) > 1)][0] = INT32_MIN
        cases.append(('element', self.index('element', tags=tags)))
        for name, line in [('orphan', 'multivalued other attr-tags.offsets\n'),
                           ('repeated', 'multivalued tags attr-tags.offsets\n'),
                           ('slash', 'multivalued tenant ../attr-tags.offsets\n'),
                           ('text', 'dictionary tags strings.bin starts.u32\n')]:
            path = self.index(name)
            with open(os.path.join(path, 'manifest'), 'a') as out:
                out.write(line)
            cases.append((name, path))
        # A text attribute cannot be multi-valued either.
        path = os.path.join(self.directory.name, 'dictionary')
        write_index(path, self.keys, self.vectors, shards=1, attributes={'tenant': self.tenant})
        with open(os.path.join(path, 'manifest'), 'a') as out:
            out.write('dictionary tenant strings.bin starts.u32\nmultivalued tenant attr-tenant.offsets\n')
        cases.append(('dictionary', path))
        for request, (name, path) in enumerate(cases, 1):
            self.assertEqual(self.load(request, name, path), INVALID, name)
        self.assertEqual(self.load(50, 'docs', self.index()), 0)

    def test_old_manifests_still_load(self):
        path = os.path.join(self.directory.name, 'single')
        write_index(path, self.keys, self.vectors, shards=3, attributes={'tenant': self.tenant})
        self.assertEqual(self.load(1, 'docs', path), 0)
        self.assertEqual(self.evaluate(2, leaf(VALUES, 'tenant', values=[0, 4])),
                         {row for row in range(ROWS) if self.tenant[row] in (0, 4)})
        self.assertEqual(self.evaluate(3, leaf(RANGE, 'tenant', low=1, high=2)),
                         {row for row in range(ROWS) if 1 <= self.tenant[row] <= 2})
        self.assertEqual(self.evaluate(4, leaf(VALUES, 'tenant', negate=True, values=[5])),
                         {row for row in range(ROWS) if self.tenant[row] != 5})


if __name__ == '__main__':
    unittest.main()
