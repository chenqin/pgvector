"""benchmarks/ann/muse_frames.py on tiny index directories: filters, their evaluator, emit, describe, verify.

Needs numpy only: no worker (a thread stands in for one), no PostgreSQL. python3 test/ann/test_muse_frames.py
"""

import contextlib
import gc
import io
import json
import os
import random
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import tracemalloc
import types
import unittest
import warnings
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOL = os.path.join(ROOT, 'benchmarks', 'ann', 'muse_frames.py')
sys.path.insert(0, os.path.dirname(TOOL))
import muse_frames as mf  # noqa: E402

NULL = -2 ** 31
# The tool as it was before filter specs, whose emit the default mode must still write byte for byte: by
# its commit, so the comparison keeps its meaning once the new file is HEAD, else HEAD.
BASELINES = ('8b3fbdf0a974ba49f01084032c42f2635b854510', 'HEAD')
TOKEN = 'a-token-of-sixteen-bytes'
# describe's pass-rate bins as the spec lists them, not as the tool has them.
EDGES = [0, 1e-6, 1e-4, 1e-3, 3e-3, 1e-2, 2e-2, 3e-2, 5e-2, 1e-1, 0.25, 0.5, 1]
LABELS = ['0', '(0, 1e-06]', '(1e-06, 0.0001]', '(0.0001, 0.001]', '(0.001, 0.003]', '(0.003, 0.01]', '(0.01, 0.02]',
          '(0.02, 0.03]', '(0.03, 0.05]', '(0.05, 0.1]', '(0.1, 0.25]', '(0.25, 0.5]', '(0.5, 1]']


def write_single(directory, column, values):
    """A single-valued column's file; returns its manifest lines."""
    np.asarray(values, dtype='<i4').tofile(os.path.join(directory, f'attr-{column}.i32'))
    return [f'attribute {column} attr-{column}.i32']


def write_multi(directory, column, arrays):
    """A multi-valued column's files (None is a NULL array); returns its manifest lines."""
    elements, offsets = [], [0]
    for array in arrays:
        elements += [NULL] if array is None else array
        offsets.append(len(elements))
    np.asarray(elements, dtype='<i4').tofile(os.path.join(directory, f'attr-{column}.i32'))
    np.asarray(offsets, dtype='<u4').tofile(os.path.join(directory, f'attr-{column}.offsets'))
    return [f'attribute {column} attr-{column}.i32', f'multivalued {column} attr-{column}.offsets']


class Plain:
    """An index directory of the given attribute columns and nothing else: enough for Columns and reference()."""

    def __init__(self, root, name, single=None, multi=None):
        self.name, self.directory, self.single, self.multi = name, os.path.join(root, name), single or {}, multi or {}
        self.rows = len(next(iter({**self.single, **self.multi}.values())))
        os.makedirs(self.directory)
        lines = [f'rows {self.rows}', 'dimensions 1']
        for column, values in self.single.items():
            lines += write_single(self.directory, column, values)
        for column, arrays in self.multi.items():
            lines += write_multi(self.directory, column, arrays)
        with open(os.path.join(self.directory, 'manifest'), 'w') as out:
            out.write('\n'.join(lines) + '\n')


class Table:
    """An index directory and its columns as Python lists (a multi-valued row is None for a NULL array)."""

    def __init__(self, root, name, rows, dimensions, seed, distinct=30, wide=False):
        rng = random.Random(seed)
        self.name, self.rows, self.directory = name, rows, os.path.join(root, name)
        self.single = {
            's0': [NULL if rng.random() < 0.1 else rng.randrange(8) for _ in range(rows)],
            's1': [NULL if rng.random() < 0.03 else rng.choice([NULL + 1, -5, 0, 3, 7, 8, 9, 1000, 2 ** 31 - 1])
                   for _ in range(rows)],
            'price': [max(1, round(1000 * 2.718281828 ** rng.gauss(0, 1))) for _ in range(rows)],     # no NULL
            'allnull': [NULL] * rows,
        }
        self.multi = {'m0': [], 'm1': []}
        for _ in range(rows):
            r = rng.random()
            # Duplicates within a row happen; a wide table has rows of many values.
            self.multi['m0'].append(None if r < 0.08 else [] if r < 0.2 else
                                    [rng.randrange(distinct) for _ in range(rng.randrange(1, 12 if wide else 6))])
            self.multi['m1'].append([] if rng.random() < 0.5 else [rng.choice([-7, 0, 5, 2 ** 31 - 1])])
        self.keys = np.arange(rows, dtype=np.int64) * 3 + 1
        self.vectors = np.random.default_rng(seed).standard_normal((rows, dimensions)).astype('<f4')
        os.makedirs(self.directory)
        self.keys.astype('<i8').tofile(os.path.join(self.directory, 'keys.i64'))
        self.vectors.tofile(os.path.join(self.directory, 'vectors.f32'))
        lines = ['pgvector-ann-index 1', f'dimensions {dimensions}', f'rows {rows}']
        for column, values in self.single.items():
            lines += write_single(self.directory, column, values)
        for column, arrays in self.multi.items():
            lines += write_multi(self.directory, column, arrays)
        with open(os.path.join(self.directory, 'manifest'), 'w') as out:
            out.write('\n'.join(lines) + '\n')

    def replace(self, column, change):
        """Rewrite a column with change(value) for each of its values; NULLs and empty arrays stay."""
        if column in self.single:
            self.single[column] = [value if value == NULL else change(value) for value in self.single[column]]
            write_single(self.directory, column, self.single[column])
        else:
            self.multi[column] = [array and [change(value) for value in array] for array in self.multi[column]]
            write_multi(self.directory, column, self.multi[column])

    def held(self, column):
        """The non-NULL values some row of a column holds."""
        if column in self.single:
            return set(self.single[column]) - {NULL}
        return {value for array in self.multi[column] if array for value in array}


def reference(expr, table):
    """The rows an expression selects, row by row from filter.h's comment: a list of bools.

    Written apart from the tool: no program, no postings, no numpy.
    """
    op = expr['op']
    if op in ('all', 'none'):
        return [op == 'all'] * table.rows
    if op == 'not':
        return [not hit for hit in reference(expr['arg'], table)]
    if op in ('and', 'or'):
        parts = [reference(arg, table) for arg in expr['args']]
        return [(all if op == 'and' else any)(hits) for hits in zip(*parts)]
    negate = expr.get('negate', False)
    if expr['column'] in table.multi:
        arrays = table.multi[expr['column']]
        if op == 'null':
            return [(array is None) != negate for array in arrays]
        assert op == 'overlap', expr
        wanted = set(expr['values']) - {NULL}
        return [array is not None and any(value in wanted for value in array) != negate for array in arrays]
    values = table.single[expr['column']]
    if op == 'null':
        return [(value == NULL) != negate for value in values]
    assert op in ('values', 'range'), expr
    wanted = set(expr['values']) if op == 'values' else None
    return [value != NULL and ((value in wanted) if wanted is not None else expr['low'] <= value <= expr['high'])
            != negate for value in values]


INTERESTING = [NULL, NULL + 1, -7, -5, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 29, 30, 1000, 2 ** 31 - 1, 12345]


def random_leaf(rng):
    kind = rng.choice(['values', 'values', 'range', 'null', 'overlap', 'overlap', 'null-multi'])
    negate = rng.random() < 0.35
    single = rng.choice(['s0', 's1', 'price', 'allnull'])
    if kind == 'values':
        values = [rng.choice(INTERESTING) if rng.random() < 0.4 else rng.randrange(-50, 50)
                  for _ in range(rng.choice([0, 1, 1, 2, 3, 5, 12]))]
        return {'op': 'values', 'column': single, 'values': values, 'negate': negate}
    if kind == 'range':
        low, high = rng.choice(INTERESTING + [500, 3000]), rng.choice(INTERESTING + [500, 3000])
        if low > high and rng.random() < 0.8:
            low, high = high, low
        return {'op': 'range', 'column': single, 'low': low, 'high': high, 'negate': negate}
    if kind == 'null':
        return {'op': 'null', 'column': single, 'negate': negate}
    if kind == 'null-multi':
        return {'op': 'null', 'column': rng.choice(['m0', 'm1']), 'negate': negate}
    values = [rng.choice(INTERESTING) if rng.random() < 0.3 else rng.randrange(32)
              for _ in range(rng.choice([0, 1, 1, 2, 3, 8, 20]))]
    return {'op': 'overlap', 'column': rng.choice(['m0', 'm1']), 'values': values, 'negate': negate}


def random_expr(rng, depth):
    r = rng.random()
    if depth <= 0 or r < 0.35:
        return random_leaf(rng)
    if r < 0.42:
        return {'op': rng.choice(['all', 'none'])}
    if r < 0.6:
        return {'op': 'not', 'arg': random_expr(rng, depth - 1)}
    return {'op': rng.choice(['and', 'or']),
            'args': [random_expr(rng, depth - 1) for _ in range(rng.choice([2, 2, 3, 4]))]}


def canonical(expr):
    """An expression as decode gives it back: an AND that is the last argument of an AND merged into it."""
    if expr['op'] == 'not':
        return {'op': 'not', 'arg': canonical(expr['arg'])}
    if expr['op'] not in ('and', 'or'):
        return dict(expr)
    args = [canonical(arg) for arg in expr['args']]
    return {'op': expr['op'], 'args': args[:-1] + (args[-1]['args'] if args[-1]['op'] == expr['op'] else args[-1:])}


def words(hits):
    """A bitmap as operation 23 returns it: u32 words, bit i of word i / 32."""
    out = [0] * ((len(hits) + 31) // 32)
    for row, hit in enumerate(hits):
        if hit:
            out[row // 32] |= 1 << (row % 32)
    return struct.pack(f'<{len(out)}I', *out)


def unique_variant(program):
    """A program as benchmarks/ann/muse_direct.c --unique-share sends it: the overlap leaf it starts with has one
    more value (BuildUnique)."""
    at = 3 + program[1]
    count = struct.unpack_from('<I', program, at)[0]
    end = at + 4 + 4 * count
    return program[:at] + struct.pack('<I', count + 1) + program[at + 4:end] + struct.pack('<i', -2) + program[end:]


def baseline():
    """muse_frames.py as it was before filter specs, as a module; None without git or that history."""
    for revision in BASELINES:
        try:
            source = subprocess.run(['git', '-C', ROOT, 'show', f'{revision}:benchmarks/ann/muse_frames.py'],
                                    capture_output=True, check=True, timeout=60).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        module = types.ModuleType('muse_frames_baseline')
        exec(compile(source, f'{revision}:muse_frames.py', 'exec'), module.__dict__)
        return module
    return None


class FakeWorker(threading.Thread):
    """One connection's worth of a worker: operations 13, 22 (exact) and 23, answered from reference().

    wrong(table, count, hits) may change a predicate's eligible count and rows, each on its own, before
    operation 23 replies, to stand for a worker that disagrees. asked is the (table, program) of every
    operation 23, in order.
    """

    def __init__(self, tables, wrong=None):
        super().__init__(daemon=True)
        self.tables, self.wrong, self.seen, self.asked = tables, wrong, [], []
        self.server = socket.create_server(('127.0.0.1', 0))
        self.port = self.server.getsockname()[1]

    def read(self, connection, size):
        data = b''
        while len(data) < size:
            chunk = connection.recv(size - len(data))
            if not chunk:
                return None
            data += chunk
        return data

    def answer(self, operation, payload):
        if operation == 13:
            return (0 if payload == TOKEN.encode() else 1), b''
        at = 4 + struct.unpack_from('<I', payload)[0]
        table = self.tables.get(payload[4:at].decode())
        if table is None:
            return 3, b''
        if operation == 22:
            dimensions, k, _, size = struct.unpack_from('<IIII', payload, at)
            encoded, query = payload[at + 16:at + 16 + size], payload[at + 16 + size:]
        else:
            encoded = payload[at:]
            self.asked.append((table.name, encoded))
        try:
            hits = reference(mf.decode(encoded), table)
        except (ValueError, KeyError, AssertionError):
            return 1, b''
        if operation == 23:
            count, hits = self.wrong(table.name, sum(hits), hits) if self.wrong else (sum(hits), hits)
            return 0, struct.pack('<Q', count) + words(hits)
        rows = np.flatnonzero(np.asarray(hits, dtype=bool))
        distances = ((table.vectors[rows] - np.frombuffer(query, dtype='<f4')) ** 2).sum(1)
        order = np.argsort(distances, kind='stable')[:k]
        body = struct.pack('<IIQ', len(order), 2, len(rows))
        return 0, body + b''.join(struct.pack('<qf', int(table.keys[rows[i]]), float(distances[i])) for i in order)

    def run(self):
        connection, _ = self.server.accept()
        with connection, self.server:
            while True:
                header = self.read(connection, 32)
                if header is None:
                    return
                _, _, operation, request, _, size, _ = struct.unpack('<4sHHQQII', header)
                payload = self.read(connection, size)
                self.seen.append(operation)
                status, body = self.answer(operation, payload)
                if status:
                    body = b''
                connection.sendall(b'PVAN' + struct.pack('<HHQQII', 1, operation, request, 0, len(body), status) + body)


class MuseFramesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = os.path.join(cls.temp.name, 'index')
        cls.tables = {table.name: table for table in (
            Table(cls.root, 't0', 300, 4, 1), Table(cls.root, 't1', 257, 4, 2), Table(cls.root, 't2', 33, 6, 3))}
        cls.big = Table(os.path.join(cls.temp.name, 'big'), 'big', 6000, 2, 4, distinct=400, wide=True)
        cls.groups = os.path.join(cls.temp.name, 'groups.json')
        cls.terms = os.path.join(cls.temp.name, 'terms.json')
        with open(cls.groups, 'w') as out:
            json.dump([{'stem': 'a', 'table': 't0', 'dimension': 4}, {'stem': 'b', 'table': 't1', 'dimension': 4},
                       {'stem': 'c', 'table': 't2', 'dimension': 6}], out)
        with open(cls.terms, 'w') as out:
            json.dump({'101': 3, '102': 4, '103': 11}, out)
        # Two clauses (terms by code and by fid, one fid the dictionary lacks), and two single-clause filters.
        wide = {'clauses': [{'column': 'm0', 'op': 'overlap', 'terms': [{'code': c} for c in range(12, 24)] +
                             [{'fid': 101}, {'code': None, 'fid': 103}, {'fid': 999}]},
                            {'column': 's0', 'op': 'eq', 'terms': [{'code': 2}]}]}
        narrow = {'clauses': [{'column': 'm0', 'op': 'overlap', 'terms': [{'code': 5}, {'code': 6}, {'fid': 102}]}]}
        single = {'clauses': [{'column': 's0', 'op': 'in', 'terms': [{'code': 1}, {'code': 3}]}]}
        v4, w4, v6 = [0.1, -0.2, 0.3, 0.4], [1.0, 0.0, -1.0, 0.5], [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        cls.queries = [
            {'repeat': 0, 'tag': 'r1', 'k': 10, 'group': 'a', 'vector': v4, 'filter': wide},
            {'repeat': 0, 'tag': 'r2', 'k': 50, 'group': 'a', 'vector': v4, 'filter': wide},
            {'repeat': 0, 'tag': 'r2', 'k': 50, 'group': 'a', 'vector': v4, 'filter': wide},     # r2 again, exactly
            {'repeat': 1, 'tag': 'r1', 'k': 10, 'group': 'a', 'vector': v4, 'filter': wide},     # skipped
            {'repeat': 0, 'tag': 'r3', 'k': 5, 'group': 'b', 'vector': w4, 'filter': narrow},
            {'repeat': 0, 'tag': 'r4', 'k': 20, 'group': 'c', 'vector': v6, 'filter': single},
            {'repeat': 0, 'tag': 'r5', 'k': 20, 'group': 'c', 'vector': v6, 'filter': wide},
        ]
        cls.capture = cls.write('capture.json', {'queries': cls.queries})
        cls.token = os.path.join(cls.temp.name, 'token')
        with open(cls.token, 'w') as out:
            out.write(TOKEN + '\n')

    @classmethod
    def write(cls, name, value):
        path = os.path.join(cls.temp.name, name)
        with open(path, 'w') as out:
            json.dump(value, out)
        return path

    def run_tool(self, argv, module=mf):
        """Run a subcommand in this process; returns its stdout and stderr."""
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            if module is mf:
                mf.main(argv)
            else:
                with mock.patch.object(sys, 'argv', ['muse_frames.py'] + argv):
                    module.main()
        return stdout.getvalue(), stderr.getvalue()

    def emit(self, name, *extra, module=mf, capture=None, groups=None, root=None):
        """emit to a file of the temp directory; returns its path, the summary line and stderr."""
        out = os.path.join(self.temp.name, name)
        stdout, stderr = self.run_tool(['emit', '--capture', capture or self.capture, '--groups', groups or self.groups,
                                        '--terms', self.terms, '--index-root', root or self.root, '--out', out,
                                        '--requests', '25', *extra], module)
        return out, json.loads(stdout), stderr

    def entries(self, frames):
        """A frames file's check entries, by request."""
        with open(frames + '.check.jsonl') as lines:
            return [json.loads(line) for line in lines]

    def test_filter_c_facts(self):
        """Rows ann/common/filter.c gives on a hand-made index (recorded from the C code, not from this tool)."""
        table = Table(os.path.join(self.temp.name, 'facts'), 'f', 8, 2, 0)
        s = [NULL, 1, 2, 3, 5, NULL, 2, 9]
        m = [None, [], [1], [1, 2], [2, 2], [], None, [7]]
        np.asarray(s, dtype='<i4').tofile(os.path.join(table.directory, 'attr-s0.i32'))
        np.asarray([NULL, 1, 1, 2, 2, 2, NULL, 7], dtype='<i4').tofile(os.path.join(table.directory, 'attr-m0.i32'))
        np.asarray([0, 1, 1, 2, 4, 6, 6, 7, 8], dtype='<u4').tofile(os.path.join(table.directory, 'attr-m0.offsets'))
        table.single['s0'], table.multi['m0'] = s, m
        columns = mf.Columns(table.directory)

        def leaf(op, column, negate=False, **operands):
            return dict(op=op, column=column, negate=negate, **operands)

        facts = [
            (leaf('values', 's0', values=[2, 3]), [2, 3, 6]),
            (leaf('values', 's0', True, values=[2, 3]), [1, 4, 7]),
            ({'op': 'not', 'arg': leaf('values', 's0', values=[2, 3])}, [0, 1, 4, 5, 7]),
            (leaf('values', 's0', values=[NULL]), []),
            (leaf('values', 's0', True, values=[NULL]), [1, 2, 3, 4, 6, 7]),
            (leaf('values', 's0', values=[]), []),
            (leaf('values', 's0', True, values=[]), [1, 2, 3, 4, 6, 7]),
            (leaf('values', 's0', values=[2, 2, 2]), [2, 6]),
            (leaf('range', 's0', low=2, high=5), [2, 3, 4, 6]),
            (leaf('range', 's0', True, low=2, high=5), [1, 7]),
            (leaf('range', 's0', low=NULL, high=2), [1, 2, 6]),
            (leaf('range', 's0', low=NULL, high=NULL), []),
            (leaf('range', 's0', low=5, high=2), []),
            (leaf('range', 's0', True, low=5, high=2), [1, 2, 3, 4, 6, 7]),
            (leaf('range', 's0', low=9, high=2 ** 31 - 1), [7]),
            (leaf('null', 's0'), [0, 5]),
            (leaf('null', 's0', True), [1, 2, 3, 4, 6, 7]),
            (leaf('overlap', 'm0', values=[1]), [2, 3]),
            (leaf('overlap', 'm0', values=[1, 2]), [2, 3, 4]),
            (leaf('overlap', 'm0', True, values=[1, 2]), [1, 5, 7]),          # empty arrays pass, NULL ones do not
            ({'op': 'not', 'arg': leaf('overlap', 'm0', values=[1, 2])}, [0, 1, 5, 6, 7]),
            (leaf('overlap', 'm0', values=[NULL]), []),
            (leaf('overlap', 'm0', True, values=[NULL]), [1, 2, 3, 4, 5, 7]),
            (leaf('overlap', 'm0', True, values=[]), [1, 2, 3, 4, 5, 7]),
            (leaf('null', 'm0'), [0, 6]),
            (leaf('null', 'm0', True), [1, 2, 3, 4, 5, 7]),
            ({'op': 'all'}, list(range(8))),
            ({'op': 'none'}, []),
        ]
        for expr, rows in facts:
            self.assertEqual(np.flatnonzero(columns.mask(expr)).tolist(), rows, expr)
            self.assertEqual(columns.count(expr), len(rows), expr)
            self.assertEqual(np.flatnonzero(reference(expr, table)).tolist(), rows, expr)    # the reference too
        # A leaf of the wrong kind for its column, and an unknown column, are errors (filter.c: -6 and -2).
        for expr in (leaf('values', 'm0', values=[1]), leaf('range', 'm0', low=0, high=9),
                     leaf('overlap', 's0', values=[1]), leaf('values', 'nosuch', values=[1])):
            with self.assertRaises(ValueError, msg=expr):
                columns.mask(expr)

    def test_evaluator_matches_reference(self):
        """Every leaf kind, negation, AND/OR/NOT/ALL/NONE and nesting, against the row-by-row reference."""
        rng = random.Random(11)
        sizes = [(1, 5), (31, 6), (32, 7), (33, 8), (64, 9), (1000, 10)]
        tables = list(self.tables.values()) + [Table(os.path.join(self.temp.name, 'sizes'), f's{rows}', rows, 2, seed)
                                               for rows, seed in sizes]
        kinds = set()
        for table in tables:
            columns = mf.Columns(table.directory)
            for _ in range(250):
                expr = random_expr(rng, rng.randrange(5))
                want = reference(expr, table)
                mask = columns.mask(expr)
                self.assertEqual((mask.dtype, mask.shape), (np.dtype(bool), (table.rows,)))
                self.assertEqual(mask.tolist(), want, (table.name, expr))
                self.assertEqual(columns.count(expr), sum(want), (table.name, expr))
                self.assertEqual(columns.mask(mf.encode(expr)).tolist(), want)      # a program's bytes do as well
                kinds.add(bool(columns.select(expr).dtype == bool))
                kinds.add((expr['op'], expr.get('negate')))
        # Both of the evaluator's forms were used, and every kind of node.
        self.assertTrue({True, False} <= kinds)
        self.assertTrue({(op, negate) for op in ('values', 'overlap', 'range', 'null') for negate in (True, False)} |
                        {(op, None) for op in ('and', 'or', 'not', 'all', 'none')} <= kinds)

    def test_columns_refuses_what_the_worker_refuses(self):
        directory = os.path.join(self.temp.name, 'bad')
        os.makedirs(directory)
        with open(os.path.join(directory, 'manifest'), 'w') as out:
            out.write('rows 2\ndimensions 1\nattribute m attr-m.i32\nmultivalued m attr-m.offsets\n'
                      'attribute s attr-s.i32\n')
        np.asarray([4, NULL, 5], dtype='<i4').tofile(os.path.join(directory, 'attr-m.i32'))      # NULL in a longer row
        np.asarray([0, 2, 3], dtype='<u4').tofile(os.path.join(directory, 'attr-m.offsets'))
        np.asarray([1, 2, 3], dtype='<i4').tofile(os.path.join(directory, 'attr-s.i32'))         # 3 values, 2 rows
        for column, op in (('m', 'overlap'), ('s', 'values')):
            with self.assertRaises(ValueError):
                mf.Columns(directory).mask({'op': op, 'column': column, 'values': [4]})

    def test_encode_decode_round_trip(self):
        rng = random.Random(5)
        for _ in range(600):
            expr = random_expr(rng, rng.randrange(6))
            encoded = mf.encode(expr)
            self.assertEqual(mf.decode(encoded), canonical(expr))
            self.assertEqual(mf.encode(mf.decode(encoded)), encoded)
            self.assertEqual(json.loads(json.dumps(mf.decode(encoded))), mf.decode(encoded))     # JSON-able
        a, b, c = (mf.encode({'op': 'null', 'column': name}) for name in 'abc')
        AND, OR, NOT = (bytes([code]) for code in (5, 6, 7))
        # The same three arguments written both ways come back as they were written.
        for written in (a + b + c + AND + AND, a + b + AND + c + AND, a + b + OR + c + AND + NOT, a + b + c + OR + AND,
                        a + b + AND + c + OR + a + OR, b'\x08', b'\x09' + NOT):
            self.assertEqual(mf.encode(mf.decode(written)), written)
        self.assertEqual(len(mf.decode(a + b + c + AND + AND)['args']), 3)
        self.assertEqual(len(mf.decode(a + b + AND + c + AND)['args']), 2)
        self.assertEqual(mf.decode(mf.encode({'op': 'null', 'column': 'a'})),
                         {'op': 'null', 'column': 'a', 'negate': False})
        # Exactly filter.h's layout.
        self.assertEqual(mf.encode({'op': 'range', 'column': 'ab', 'low': -2, 'high': 258, 'negate': True}),
                         b'\x02\x02ab\x01\xfe\xff\xff\xff\x02\x01\x00\x00')
        self.assertEqual(mf.encode({'op': 'values', 'column': 'x', 'values': [3, NULL, 3]}),
                         b'\x01\x01x\x00\x03\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x80\x03\x00\x00\x00')
        self.assertEqual(mf.encode({'op': 'or', 'args': [{'op': 'overlap', 'column': 'm', 'values': []},
                                                         {'op': 'not', 'arg': {'op': 'none'}}]}),
                         b'\x0a\x01m\x00\x00\x00\x00\x00\x09\x07\x06')

    def test_encode_matches_legacy_program(self):
        """An AND of leaves is byte for byte what program() writes, and decode reads it back."""
        rng = random.Random(3)
        for clauses in range(1, 6):
            for _ in range(20):
                columns = [rng.choice(['m0', 'a6', 'pg_ids']) for _ in range(clauses)]
                ops = [rng.choice(['overlap', 'eq', 'in']) for _ in range(clauses)]
                codes = [sorted(rng.sample(range(-9, 5000), rng.choice([0, 1, 3, 168]))) for _ in range(clauses)]
                legacy = mf.program(columns, ops, codes)
                expr = mf.legacy_expr(columns, ops, codes)
                self.assertEqual(mf.encode(expr), legacy)
                self.assertEqual(mf.decode(legacy), expr)
                self.assertEqual(expr['op'], 'and' if clauses > 1 else 'overlap' if ops[0] == 'overlap' else 'values')
                entry = {'columns': columns, 'ops': ops, 'codes': codes, 'filtered': True}
                self.assertEqual(mf.entry_expr(entry), expr)
                self.assertEqual(mf.entry_expr(dict(entry, filtered=False)), {'op': 'all'})

    def test_no_clause_filter(self):
        """A captured filter without clauses: no bytes on the wire, as before; every row for check, as before."""
        self.assertEqual(mf.program([], [], []), b'')
        with self.assertRaises(ValueError):
            mf.decode(b'')                              # filter.c refuses the empty program
        self.assertEqual(mf.legacy_expr([], [], []), {'op': 'all'})
        columns = mf.Columns(self.tables['t0'].directory)
        self.assertTrue(mf.eligible(columns, {'columns': [], 'ops': [], 'codes': [], 'filtered': True}).all())
        queries = [dict(self.queries[0], filter={'clauses': []}), self.queries[4]]
        frames, _, stderr = self.emit('noclause.frames', '--unfiltered-share', '0',
                                      capture=self.write('noclause.json', {'queries': queries}))
        self.assertIn('1 captured sub-requests have a filter without clauses', stderr)
        first = next(mf.read_frames(frames))
        self.assertEqual(mf.parse_frame(first[0])[3], b'')
        self.assertNotEqual(mf.parse_frame(first[1])[3], b'')
        self.assertEqual(self.emit('clauses.frames')[2], '')        # no warning for the usual capture
        with self.assertRaises(SystemExit) as raised:
            self.describe(frames)
        self.assertIn('a program the worker refuses, sent 25 times', str(raised.exception))
        # verify agrees with a worker that refuses it, and reports one that does not.
        worker, address = self.worker()
        stdout, _ = self.run_tool(['verify', '--frames', frames, '--index-root', self.root, *address])
        self.assertEqual(json.loads(stdout)['mismatched'], 0)
        self.assertEqual(worker.seen.count(23), json.loads(stdout)['compared'])

    def test_limits(self):
        """encode and decode refuse what AnnFilterCheck refuses, and nothing it accepts."""
        def leaf(count=1, name='s'):
            return {'op': 'values', 'column': name, 'values': [1] * count}

        def left(n):
            expr = leaf()
            for _ in range(n - 1):
                expr = {'op': 'and', 'args': [expr, leaf()]}
            return expr

        accepted = [leaf(65536), {'op': 'and', 'args': [leaf(65536)] * 3}, {'op': 'and', 'args': [leaf()] * 64},
                    left(65), {'op': 'not', 'arg': left(512)}, leaf(name='x' * 63)]
        refused = [leaf(65537), {'op': 'and', 'args': [leaf(65536)] * 4}, {'op': 'and', 'args': [leaf()] * 65},
                   left(513), leaf(name='x' * 64), leaf(name=''), leaf(name='a\0b'), {'op': 'and', 'args': [leaf()]},
                   {'op': 'values', 'column': 's', 'values': [2 ** 31]}, {'op': 'range', 'column': 's', 'low': 0},
                   {'op': 'nand', 'args': [leaf(), leaf()]}, {'op': 'null', 'column': 's', 'negate': 2}, [leaf()], None]
        for expr in accepted:
            self.assertEqual(mf.encode(mf.decode(mf.encode(expr))), mf.encode(expr))
        self.assertEqual(len(mf.steps(mf.encode({'op': 'not', 'arg': left(512)}))), 1024)
        for at, expr in enumerate(refused):
            with self.assertRaises(ValueError, msg=at):
                mf.encode(expr)
        one = mf.encode(leaf())
        AND, NOT, ALL = b'\x05', b'\x07', b'\x08'
        for malformed in (b'', one + one, ALL + AND, NOT, one + b'\x00', one[:-1], b'\x04' + bytes(8), b'\x0b',
                          one[:3] + b'\x02' + one[4:], one * 65 + AND * 64, one + (one + AND) * 512):
            with self.assertRaises(ValueError, msg=malformed[:40]):
                mf.decode(malformed)
        self.assertEqual(mf.decode(one + (one + AND) * 511 + NOT)['op'], 'not')

    def test_default_emit_is_byte_identical(self):
        """Without --filter-spec, emit writes what the tool wrote before it had one."""
        old = baseline()
        if old is None:
            self.skipTest('git, or the history of benchmarks/ann/muse_frames.py, is not available')
        for name, extra in (('mixed', ()), ('replay', ('--mode', 'replay')), ('independent', ('--independent',)),
                            ('options', ('--seed', '3', '--unfiltered-share', '0.5', '--distinct-filters', '5',
                                         '--noise', '0.2', '--check-requests', '4'))):
            written = []
            for module in (old, mf):
                frames, summary, _ = self.emit(f'identical-{name}.frames', *extra, module=module)
                with open(frames, 'rb') as file, open(frames + '.check.jsonl', 'rb') as check:
                    written.append((file.read(), check.read(), summary))
            self.assertEqual(written[0][0], written[1][0], name)
            self.assertEqual(written[0][1], written[1][1], name)
            self.assertEqual(written[0][2], written[1][2], name)
            self.assertGreater(len(written[1][0]), 25 * 6 * 60)
            self.assertEqual(written[1][1].count(b'\n'), 4 if name == 'options' else 25)

    def test_old_format_check_entries(self):
        """columns/ops/codes/filtered entries select the rows check always selected for them."""
        def before(table, spec):
            # check's own loop as it was: np.isin over the column, a multi-valued row by any element.
            mask = np.ones(table.rows, dtype=bool)
            if not spec['filtered']:
                return mask
            for column, codes in zip(spec['columns'], spec['codes']):
                if column in table.multi:
                    mask &= np.array([bool(array) and any(value in codes for value in array)
                                      for array in table.multi[column]])
                else:
                    mask &= np.isin(np.asarray(table.single[column]), np.asarray(codes, dtype=np.int32))
            return mask

        frames, _, _ = self.emit('old.frames', '--unfiltered-share', '0.3')
        seen = set()
        for request in self.entries(frames):
            for spec in request:
                self.assertEqual(set(spec), {'table', 'k', 'vector', 'filtered', 'columns', 'ops', 'codes'})
                table = self.tables[spec['table']]
                got = mf.eligible(mf.Columns(table.directory), spec)
                self.assertEqual(got.tolist(), before(table, spec).tolist())
                self.assertEqual(got.tolist(), reference(mf.entry_expr(spec), table))
                seen.add((spec['filtered'], len(spec['columns']), bool(got.any()), bool(got.all())))
        self.assertTrue({(True, 1, True, False), (True, 2, True, False), (False, 2, True, True)} <= seen, seen)
        # INT32_MIN among the codes is no value: filter.c skips it, where the old loop matched NULL rows with it.
        for column, op in (('s0', 'eq'), ('m0', 'overlap')):
            table = self.tables['t0']
            spec = {'filtered': True, 'columns': [column], 'ops': [op], 'codes': [[NULL]]}
            self.assertGreater(before(table, spec).sum() if column == 's0' else table.multi['m0'].count(None), 0)
            self.assertEqual(mf.eligible(mf.Columns(table.directory), spec).sum(), 0)
            spec['codes'] = [[NULL, 3]]
            self.assertEqual(mf.eligible(mf.Columns(table.directory), spec).tolist(),
                             reference({'op': 'values' if op == 'eq' else 'overlap', 'column': column, 'values': [3]},
                                       table))

    SPEC = {'distinct': 12, 'popularity': 'zipf', 'zipf_s': 1.2, 'templates': [
        {'name': 'wide', 'weight': 2, 'expr': {'op': 'overlap', 'column': 'm0', 'pass': [0.05, 0.3]}},
        {'name': 'combo', 'pass': [0, 1], 'expr': {'op': 'and', 'args': [
            {'op': 'overlap', 'column': 'm0', 'count': 3, 'pad': 40},
            {'op': 'or', 'args': [{'op': 'range', 'column': 'price', 'pass': [0.2, 0.6]},
                                  {'op': 'null', 'column': 's0'}]}]}},
        {'name': 'explicit',
         'expr': {'op': 'not', 'arg': {'op': 'values', 'column': 's0', 'values': [1, 2], 'negate': True}}},
    ]}

    def test_filter_spec_emit(self):
        spec = self.write('spec.json', self.SPEC)
        frames, summary, _ = self.emit('spec.frames', '--filter-spec', spec, '--unfiltered-share', '0.2')
        again, _, _ = self.emit('spec-again.frames', '--filter-spec', spec, '--unfiltered-share', '0.2')
        other, _, _ = self.emit('spec-other.frames', '--filter-spec', spec, '--unfiltered-share', '0.2', '--seed', '8')
        plain, plain_summary, _ = self.emit('spec-plain.frames', '--unfiltered-share', '0.2')
        files = []
        for path in (frames, again, other):
            with open(path, 'rb') as file, open(path + '.check.jsonl', 'rb') as check:
                files.append((file.read(), check.read()))
        self.assertEqual(files[0], files[1])                    # the same seed writes the same bytes
        self.assertNotEqual(files[0][0], files[2][0])

        # Only the filters differ from an emit without the spec: tables, vectors, k and who runs unfiltered.
        entries, plain_entries = self.entries(frames), self.entries(plain)
        shape = [[(e['table'], e['k'], e['vector'], e['filtered']) for e in request] for request in entries]
        self.assertEqual(shape, [[(e['table'], e['k'], e['vector'], e['filtered']) for e in request]
                                 for request in plain_entries])
        self.assertEqual({key: summary[key] for key in plain_summary if key != 'frames'},
                         {key: plain_summary[key] for key in plain_summary if key != 'frames'})

        pool = summary['filter_pool']
        self.assertEqual((pool['filters'], pool['reference_table'], pool['popularity']), (12, 't0', 'zipf'))
        self.assertEqual(sum(pool['templates'].values()), 12)
        self.assertEqual(list(pool['templates']), ['wide', 'combo', 'explicit'])
        self.assertTrue(all(pool['templates'].values()))
        rates = [pool['pass_rate'][key] for key in ('min', 'p10', 'p50', 'p90', 'max')]
        self.assertEqual(rates, sorted(rates))
        self.assertLessEqual(pool['distinct_programs'], 12)
        self.assertGreater(pool['distinct_programs'], 6)

        programs, shares = set(), []
        stream = mf.read_frames(frames)
        for request in entries:
            sent = next(stream)
            self.assertEqual(len(sent), 6)
            for at, (spec_entry, frame) in enumerate(zip(request, sent)):
                self.assertEqual(set(spec_entry), {'table', 'k', 'vector', 'filtered', 'expr'})
                table, dimensions, k, encoded = mf.parse_frame(frame)
                self.assertEqual((table, dimensions, k),
                                 (spec_entry['table'], len(spec_entry['vector']), spec_entry['k']))
                self.assertEqual(encoded, mf.encode(spec_entry['expr']))
                self.assertEqual(spec_entry['expr'] == {'op': 'all'}, not spec_entry['filtered'])
                if spec_entry['filtered']:
                    programs.add(encoded)
                    shares.append(sum(reference(spec_entry['expr'], self.tables['t0'])) / 300)
                    values = [leaf['values'] for leaf in self.leaves(spec_entry['expr']) if 'values' in leaf]
                    self.assertTrue(all(v == sorted(set(v)) and NULL not in v for v in values))
            # Sub-requests that shared a captured filter (the wide one: 0, 1, 2 and 5) share the drawn one.
            self.assertEqual(len({json.dumps(request[at]['expr']) for at in (0, 1, 2, 5)}), 1)
        self.assertLessEqual(len(programs), pool['distinct_programs'])
        self.assertGreater(len(programs), 4)
        self.assertGreaterEqual(min(shares), pool['pass_rate']['min'] - 1e-12)
        self.assertLessEqual(max(shares), pool['pass_rate']['max'] + 1e-12)
        # --independent draws a filter per sub-request.
        independent, _, _ = self.emit('spec-independent.frames', '--filter-spec', spec, '--independent')
        self.assertTrue(any(len({json.dumps(request[at]['expr']) for at in (0, 1, 2, 5)}) > 1
                            for request in self.entries(independent)))

    def leaves(self, expr):
        if expr['op'] in ('and', 'or'):
            return [leaf for arg in expr['args'] for leaf in self.leaves(arg)]
        if expr['op'] == 'not':
            return self.leaves(expr['arg'])
        return [expr] if 'column' in expr else []

    def pool(self, templates, distinct=40, seed=7, table=None, held=None):
        """A pool drawn straight from templates on the big table."""
        path = self.write('pool.json', {'distinct': distinct, 'templates': templates})
        return mf.filter_pool(mf.load_filter_spec(path), mf.Columns((table or self.big).directory), seed,
                              *([] if held is None else [held]))

    def leaf_shares(self, pool, table):
        """The share of a table's rows that each entry's leaf (the whole expression) holds, before negation."""
        return [sum(reference(dict(entry['expr'], negate=False), table)) / table.rows for entry in pool]

    @classmethod
    def shapes(cls):
        """A table of columns with heavy values, which a share has to be built around; made once."""
        if not hasattr(cls, 'shaped'):
            rng = random.Random(9)

            def shuffled(values):
                rng.shuffle(values)
                return values

            light = [value for value in range(401) if value != 200]
            cls.shaped = Plain(os.path.join(cls.temp.name, 'plain'), 'shapes', {
                'flag': shuffled([1] * 1940 + [0] * 60),                            # the W2 spec's flag: 97% and 3%
                'heavy': shuffled([1] * 1000 + [2] * 600 + [3] * 400),              # 50%, 30% and 20%
                'mixed': shuffled([200] * 1200 + light * 2),                        # 60% in the middle of 0.1% values
                # A quarter of the values hold 3 rows and the others 1; a quarter of the rows are NULL.
                'fine': shuffled([value for value in range(1000) for _ in range(1 if value % 4 else 3)] + [NULL] * 500),
                'rare': shuffled(list(range(980)) * 2 + list(range(980, 1020))),    # 2 rows a value but for the last 40
            }, {'mflag': [[value] for value in shuffled([1] * 1940 + [0] * 60)],
                'mheavy': [[value, value] for value in shuffled([1] * 1000 + [2] * 600 + [3] * 400)]})
        return cls.shaped

    def test_pass_around_heavy_values(self):
        """A values or overlap generator lands inside where a heavy value must be taken before the light ones."""
        shapes = self.shapes()
        for op, column, low, high, values in (
                ('values', 'flag', 0.95, 0.98, [1]), ('values', 'heavy', 0.25, 0.35, [2]),
                ('values', 'heavy', 0.75, 0.85, [1, 2]), ('values', 'heavy', 0.45, 0.55, None),
                ('values', 'mixed', 0.55, 0.65, None), ('values', 'mixed', 0.6, 0.9, None),
                ('overlap', 'mflag', 0.95, 0.98, [1]), ('overlap', 'mheavy', 0.75, 0.85, [1, 2])):
            pool = self.pool([{'expr': {'op': op, 'column': column, 'pass': [low, high]}}], table=shapes)
            shares = self.leaf_shares(pool, shapes)
            self.assertTrue(all(low <= share <= high for share in shares), (column, low, high, sorted(shares)))
            if values:
                self.assertTrue(all(entry['expr']['values'] == values for entry in pool), (column, low, high))
            if column == 'mixed':
                self.assertTrue(all(200 in entry['expr']['values'] for entry in pool))
                self.assertGreater(len({entry['program'] for entry in pool}), 20)       # still random lists
        # Both ways to half of the rows are drawn: the heaviest-first list is not the only one.
        halves = {tuple(entry['expr']['values']) for entry in self.pool(
            [{'expr': {'op': 'values', 'column': 'heavy', 'pass': [0.45, 0.55]}}], table=shapes)}
        self.assertEqual(halves, {(1,), (2, 3)})

    def test_range_lands_inside(self):
        """A range generator lands inside wherever a run of the column's values does, else on the nearest run."""
        shapes = self.shapes()

        def ranges(column, low, high, distinct=60):
            pool = self.pool([{'expr': {'op': 'range', 'column': column, 'pass': [low, high]}}], distinct, table=shapes)
            held = set(shapes.single[column]) - {NULL}
            self.assertTrue(all({entry['expr']['low'], entry['expr']['high']} <= held for entry in pool))
            return pool, self.leaf_shares(pool, shapes)

        # A heavy value among light ones: a low a little under it has no run inside. Values heavier than the
        # range among lighter ones. One value of three that fits alone.
        for column, low, high in (('mixed', 0.05, 0.1), ('fine', 0.0005, 0.001), ('heavy', 0.25, 0.35),
                                  ('rare', 0.0005, 0.0005)):
            pool, shares = ranges(column, low, high)
            self.assertTrue(all(low <= share <= high for share in shares), (column, low, high, sorted(shares)))
            if column == 'heavy':
                self.assertEqual({(entry['expr']['low'], entry['expr']['high']) for entry in pool}, {(2, 2)})
            else:
                self.assertGreater(len({entry['expr']['low'] for entry in pool}), 15, column)
        self.assertGreater(len(set(ranges('mixed', 0.05, 0.1)[1])), 15)                  # shares over the range
        # The same when the random lows give out and every low is looked at.
        with mock.patch.object(mf, 'RANGE_LOWS', 1):
            for column, low, high in (('rare', 0.0005, 0.0005), ('mixed', 0.05, 0.1)):
                pool, shares = ranges(column, low, high)
                self.assertTrue(all(low <= share <= high for share in shares), (column, sorted(shares)))
                self.assertGreater(len({entry['expr']['low'] for entry in pool}), 15, column)
        # No run inside: the nearest of all, which here is the lightest value alone (20%), whatever the low.
        pool, _ = ranges('heavy', 0.01, 0.05, 12)
        self.assertEqual({(entry['expr']['low'], entry['expr']['high']) for entry in pool}, {(3, 3)})
        # No count of rows is inside (between 1 and 2 rows of 2,000): a run of 1 or of 2.
        _, shares = ranges('fine', 0.00051, 0.00099, 12)
        self.assertTrue(set(shares) <= {1 / 2000, 2 / 2000}, shares)

    def test_range_leaf_costs_its_rows(self):
        """A leaf on a column of many values allocates for the rows it matches, not for the column's values."""
        rows = 400000
        values = np.random.default_rng(0).permutation(rows)
        table = Plain(os.path.join(self.temp.name, 'plain'), 'cost', {'n': values})
        columns = mf.Columns(table.directory)
        columns.column('n')
        want = np.flatnonzero((values >= 1000) & (values < 1020)).tolist()
        for expr in ({'op': 'range', 'column': 'n', 'low': 1000, 'high': 1019},
                     {'op': 'range', 'column': 'n', 'low': NULL, 'high': 19, 'negate': False},
                     {'op': 'values', 'column': 'n', 'values': list(range(1000, 1020))}):
            encoded = mf.encode(expr)
            tracemalloc.start()
            try:
                selected = columns.select(encoded)
                peak = tracemalloc.get_traced_memory()[1]
            finally:
                tracemalloc.stop()
            self.assertEqual(len(selected), 20)
            if expr.get('low') != NULL:
                self.assertEqual(selected.tolist(), want)
            # The column's distinct values are 1.6 MB as stored, and twice that as 64-bit integers.
            self.assertLess(peak, 100000, expr)

    def test_pool_draws_are_not_repeated(self):
        """A program is counted once however often it is drawn, and a template with nothing new is drawn once."""
        t0 = self.tables['t0']

        def drawn(templates, distinct=50, table=t0):
            with mock.patch.object(mf.Columns, 'count', autospec=True, side_effect=mf.Columns.count) as counts, \
                    mock.patch.object(mf, 'draw_filter', side_effect=mf.draw_filter) as draws:
                pool = self.pool(templates, distinct, table=table)
            self.assertEqual(len(pool), distinct)
            return pool, counts.call_count, draws.call_count

        # Without generators: one program, counted once; redrawn for the second entry only.
        explicit = {'op': 'values', 'column': 's0', 'values': [1]}
        pool, counts, draws = drawn([{'expr': explicit}])
        self.assertEqual({entry['program'] for entry in pool}, {mf.encode(explicit)})
        self.assertEqual({entry['share'] for entry in pool}, {t0.single['s0'].count(1) / t0.rows})
        self.assertEqual((counts, draws), (1, 1 + mf.DUPLICATE_REDRAWS + 48))
        # Eight outcomes: each counted once, and all found. A round of redraws that finds nothing new comes
        # at most once per outcome found after it, and once at the end.
        pool, counts, draws = drawn([{'expr': {'op': 'values', 'column': 's0', 'count': 1}}])
        self.assertEqual(len({entry['program'] for entry in pool}), 8)
        self.assertEqual(counts, 8)
        self.assertLessEqual(draws, 50 + 2 * 8 * (mf.DUPLICATE_REDRAWS - 1))
        # A template's "pass" that refuses some of the outcomes counts those once too.
        shares = sorted(t0.single['s0'].count(value) / t0.rows for value in range(8))
        pool, counts, draws = drawn([{'pass': [shares[3], 1], 'expr': {'op': 'values', 'column': 's0', 'count': 1}}])
        self.assertTrue(all(entry['share'] >= shares[3] for entry in pool))
        self.assertGreaterEqual(len({entry['program'] for entry in pool}), 4)
        self.assertLessEqual(counts, 8)
        # Templates are spent one by one: beside a spent one, another still gives a new program per entry.
        pool, counts, draws = drawn([{'name': 'one', 'expr': explicit},
                                     {'name': 'many', 'expr': {'op': 'values', 'column': 'price', 'count': 3}}], 60,
                                    self.big)
        many = [entry['program'] for entry in pool if entry['template'] == 'many']
        self.assertEqual(len(set(many)), len(many))
        self.assertEqual(counts, len(many) + 1)
        # As many entries as outcomes, and twice as many: nearly every outcome, then every one, is drawn.
        for distinct, least in ((30, 25), (60, 30)):
            pool, counts, draws = drawn([{'expr': {'op': 'overlap', 'column': 'm0', 'count': 1}}], distinct)
            self.assertGreaterEqual(len({entry['program'] for entry in pool}), least)
            self.assertLessEqual(counts, 30)

    def test_pass_targeting(self):
        big = self.big
        held = big.held('m0')
        self.assertEqual(len(held), 400)

        # A leaf's share of the rows is drawn inside its range, before negation.
        for negate in (False, True):
            pool = self.pool([{'expr': {'op': 'overlap', 'column': 'm0', 'pass': [0.03, 0.06], 'negate': negate}}])
            shares = []
            for entry in pool:
                expr = entry['expr']
                self.assertEqual((expr['op'], expr['column'], expr['negate']), ('overlap', 'm0', negate))
                self.assertTrue(set(expr['values']) <= held and expr['values'] == sorted(set(expr['values'])))
                shares.append(sum(reference(dict(expr, negate=False), big)) / big.rows)
                self.assertAlmostEqual(entry['share'], sum(reference(expr, big)) / big.rows, places=12)
                self.assertEqual(entry['program'], mf.encode(expr))
            self.assertTrue(all(0.03 <= share <= 0.06 for share in shares), shares)
            self.assertGreater(len(set(shares)), 10)                        # drawn over the range, not one share
            self.assertEqual(len({entry['program'] for entry in pool}), 40)

        # Values of a single-valued column, and a range of one.
        for entry in self.pool([{'expr': {'op': 'values', 'column': 'price', 'pass': [0.01, 0.02]}}]):
            self.assertTrue(0.01 <= sum(reference(entry['expr'], big)) / big.rows <= 0.02)
        lows = set()
        for entry in self.pool([{'expr': {'op': 'range', 'column': 'price', 'pass': [0.05, 0.1], 'negate': True}}]):
            expr = entry['expr']
            self.assertTrue({expr['low'], expr['high']} <= big.held('price') and expr['negate'])
            self.assertTrue(0.05 <= sum(reference(dict(expr, negate=False), big)) / big.rows <= 0.1, expr)
            lows.add(expr['low'])
        self.assertGreater(len(lows), 20)

        # A template's own range holds for the whole expression, by redrawing.
        template = {'name': 'both', 'pass': [0.002, 0.004], 'expr': {'op': 'and', 'args': [
            {'op': 'overlap', 'column': 'm0', 'pass': [0.02, 0.2]},
            {'op': 'range', 'column': 'price', 'pass': [0.02, 0.2]}]}}
        for entry in self.pool([template], distinct=25):
            self.assertTrue(0.002 <= sum(reference(entry['expr'], big)) / big.rows <= 0.004)
            self.assertEqual(entry['template'], 'both')
        # Zero is a range a template can ask for.
        for entry in self.pool([{'pass': [0, 0], 'expr': {'op': 'and', 'args': [
                {'op': 'overlap', 'column': 'm0', 'count': 1}, {'op': 'values', 'column': 'price', 'count': 1}]}}], 5):
            self.assertEqual(sum(reference(entry['expr'], big)), 0)

    def test_pass_out_of_reach(self):
        """What a generator does when its column cannot give the share."""
        t0 = self.tables['t0']
        # allnull has no value: nothing to pick. s0's 8 values are each over 5% of the rows: the nearest one.
        self.assertEqual(self.pool([{'expr': {'op': 'values', 'column': 'allnull', 'pass': [0.1, 0.2]}}], 3,
                                   table=t0)[0]['expr']['values'], [])
        for entry in self.pool([{'expr': {'op': 'values', 'column': 's0', 'pass': [0.01, 0.05]}}], 5, table=t0):
            self.assertEqual(len(entry['expr']['values']), 1)
        counts = {value: t0.single['s0'].count(value) for value in t0.held('s0')}
        rarest = min(counts.values())
        self.assertTrue(all(counts[entry['expr']['values'][0]] == rarest for entry in self.pool(
            [{'expr': {'op': 'values', 'column': 's0', 'pass': [0.01, 0.05]}}], 5, table=t0)))
        # More than the column holds: every value, and for a range the whole column.
        every = self.pool([{'expr': {'op': 'overlap', 'column': 'm1', 'pass': [0.9, 1]}}], 2, table=t0)[0]['expr']
        self.assertEqual(set(every['values']), t0.held('m1'))
        whole = self.pool([{'expr': {'op': 'range', 'column': 's0', 'pass': [0.99, 1]}}], 2, table=t0)[0]['expr']
        self.assertEqual((whole['low'], whole['high']), (min(t0.held('s0')), max(t0.held('s0'))))
        # A template whose range no draw reaches fails, naming itself and the shares it saw.
        with self.assertRaises(SystemExit) as raised:
            self.pool([{'name': 'unreachable', 'pass': [0.9, 0.95],
                        'expr': {'op': 'overlap', 'column': 'm0', 'pass': [0.03, 0.06]}}])
        message = str(raised.exception)
        self.assertIn('template unreachable', message)
        self.assertIn('none of 200 draws', message)
        self.assertIn('[0.9, 0.95]', message)
        low, high = (float(word) for word in message.split('run from ')[1].replace(',', '').split()[:3:2])
        self.assertTrue(0.03 <= low <= high <= 0.06, message)

    def test_pad_and_count(self):
        big = self.big
        held = big.held('m0')
        for entry in self.pool([{'expr': {'op': 'overlap', 'column': 'm0', 'pass': [0.03, 0.06], 'pad': 2582}}], 10):
            values = entry['expr']['values']
            self.assertEqual(len(values), 2582)
            self.assertEqual(values, sorted(set(values)))
            self.assertNotIn(NULL, values)
            kept = [value for value in values if value in held]
            self.assertTrue(0 < len(kept) < 60)
            # The rows that pass are those of the values some row holds.
            self.assertEqual(reference(entry['expr'], big), reference(dict(entry['expr'], values=kept), big))
            self.assertTrue(0.03 <= entry['share'] <= 0.06)
        for entry in self.pool([{'expr': {'op': 'values', 'column': 'price', 'count': 7, 'pad': 7}},
                                {'expr': {'op': 'overlap', 'column': 'm0', 'count': 7, 'pad': 3}}], 10):
            values = entry['expr']['values']
            self.assertEqual(len(set(values)), 7)
            self.assertTrue(set(values) <= big.held(entry['expr']['column']))
        # A pad at the limit (see test_generated_overlap_leaves_room) still encodes; a column of every value in
        # a span is padded from outside it.
        full = self.pool([{'expr': {'op': 'values', 'column': 'price', 'count': 2, 'pad': 65536}}], 1)[0]
        self.assertEqual(len(full['expr']['values']), 65536)
        dense = self.pool([{'expr': {'op': 'values', 'column': 's0', 'count': 8, 'pad': 30}}], 1,
                          table=self.tables['t0'])[0]['expr']['values']
        self.assertEqual((len(dense), len(set(dense) & set(range(8)))), (30, 8))
        with self.assertRaises(SystemExit) as raised:
            self.pool([{'expr': {'op': 'values', 'column': 's0', 'count': 9}}], 1, table=self.tables['t0'])
        self.assertIn('has 8 values', str(raised.exception))

    def test_generated_overlap_leaves_room(self):
        """A generated overlap list is at most 65,535 values: muse_direct --unique-share adds one to the leaf a
        program starts with, and the worker refuses 65,537."""
        full = self.pool([{'expr': {'op': 'overlap', 'column': 'm0', 'count': 2, 'pad': 65535}}], 1)[0]
        self.assertEqual(len(full['expr']['values']), 65535)
        self.assertEqual(len(mf.steps(unique_variant(full['program']))[0][3]), 65536)
        with self.assertRaises(ValueError):
            mf.steps(unique_variant(mf.encode({'op': 'overlap', 'column': 'm0', 'values': list(range(65536))})))
        for key in ('pad', 'count'):
            with self.assertRaises(SystemExit) as raised:
                self.pool([{'expr': {'op': 'overlap', 'column': 'm0', 'count': 2, **{key: 65536}}}], 1)
            self.assertIn(f'"{key}" of a generated overlap leaf is an integer from {key == "count":d} to 65535',
                          str(raised.exception))
        # A share that needs more values than that stops there, where a values leaf takes one more.
        rows = 70000
        table = Plain(os.path.join(self.temp.name, 'plain'), 'wide', {'u': np.arange(rows)},
                      {'w': [[value] for value in range(rows)]})
        for op, column, most in (('overlap', 'w', 65535), ('values', 'u', 65536)):
            entry = self.pool([{'expr': {'op': op, 'column': column, 'pass': [0.99, 1]}}], 1, table=table)[0]
            self.assertEqual(len(entry['expr']['values']), most)
            self.assertEqual(entry['share'], most / rows)
            if op == 'overlap':
                mf.steps(unique_variant(entry['program']))

    def test_pad_on_every_table(self):
        """A pad's values are ones no row of any table of --groups holds, so the same rows pass on each."""
        root = os.path.join(self.temp.name, 'pads')
        tables = [Table(root, 'even', 200, 4, 21), Table(root, 'odd', 180, 4, 22), Table(root, 'six', 90, 6, 23)]
        # No two tables share a value of m0 (0..29 in each as made) or of s0 (0..7).
        for at, table in enumerate(tables[:2]):
            table.replace('m0', lambda value: 2 * value + at)
            table.replace('s0', lambda value: 2 * value + at)
        tables[2].replace('m0', lambda value: 100 + value)
        tables[2].replace('s0', lambda value: -3 - value)
        groups = self.write('pads-groups.json', [{'stem': 'a', 'table': 'even', 'dimension': 4},
                                                 {'stem': 'b', 'table': 'odd', 'dimension': 4},
                                                 {'stem': 'c', 'table': 'six', 'dimension': 6}])
        spec = self.write('pads.json', {'distinct': 12, 'templates': [
            {'name': 'm', 'expr': {'op': 'overlap', 'column': 'm0', 'pass': [0.05, 0.3], 'pad': 150}},
            {'name': 's', 'expr': {'op': 'values', 'column': 's0', 'count': 2, 'pad': 40, 'negate': True}}]})
        frames, summary, _ = self.emit('pads.frames', '--filter-spec', spec, '--unfiltered-share', '0', groups=groups,
                                       root=root)
        self.assertEqual(summary['filter_pool']['reference_table'], 'even')
        exprs = {json.dumps(entry['expr']): entry['expr'] for request in self.entries(frames) for entry in request}
        self.assertGreater(len(exprs), 6)
        for expr in exprs.values():
            column, values = expr['column'], expr['values']
            self.assertEqual(len(values), 150 if column == 'm0' else 40)
            kept = [value for value in values if value in tables[0].held(column)]
            self.assertTrue(len(kept) == 2 if column == 's0' else kept)
            self.assertFalse(set(values) & (tables[1].held(column) | tables[2].held(column)))
            for table in tables:
                self.assertEqual(reference(expr, table), reference(dict(expr, values=kept), table), table.name)
            # The other tables hold none of the reference table's values: nothing of theirs overlaps the list.
            if column == 'm0':
                self.assertEqual(sum(reference(expr, tables[1])) + sum(reference(expr, tables[2])), 0)
        # Straight from filter_pool, the values to stay clear of are given by column.
        directories = [table.directory for table in tables]
        for column in ('m0', 's0'):
            self.assertEqual(mf.held_values(directories, column).tolist(),
                             sorted(set().union(*(table.held(column) for table in tables))))
        self.assertEqual(mf.held_values(directories[:1], 'allnull').tolist(), [])
        held = {'m0': mf.held_values(directories, 'm0')}
        for entry in self.pool([{'expr': {'op': 'overlap', 'column': 'm0', 'count': 3, 'pad': 500}}], 5,
                               table=tables[0], held=held):
            self.assertEqual(len(set(entry['expr']['values']) - set(held['m0'].tolist())), 497)

    def test_filter_spec_refusals(self):
        def refused(spec, message, *extra):
            with self.assertRaises(SystemExit) as raised:
                self.emit('refused.frames', '--filter-spec', self.write('refused.json', spec), *extra)
            self.assertIn(message, str(raised.exception))

        leaf = {'op': 'overlap', 'column': 'm0', 'pass': [0.05, 0.3]}
        refused({'templates': []}, 'templates')
        refused({'templates': [{'expr': leaf}], 'distinct': 0}, 'distinct')
        refused({'templates': [{'expr': leaf}], 'popularity': 'pareto'}, 'popularity')
        refused({'templates': [{'expr': leaf, 'weight': 0}]}, 'weight')
        refused({'templates': [{'expr': leaf, 'pass': [0.5, 0.1]}]}, '"pass" is [lo, hi]')
        refused({'templates': [{'expr': dict(leaf, **{'pass': [0, 0.1]})}]}, '0 < lo')
        refused({'templates': [{'expr': dict(leaf, count=2)}]}, 'takes "column"')
        refused({'templates': [{'expr': {'op': 'overlap', 'column': 'm0'}}]}, 'needs "pass" or "count"')
        refused({'templates': [{'expr': dict(leaf, pad=70000)}]}, '"pad"')
        refused({'templates': [{'expr': {'op': 'and', 'args': [leaf]}}]}, 'not a filter expression')
        refused({'templates': [{'expr': {'op': 'range', 'column': 'price', 'pass': [0.1, 0.2], 'pad': 3}}]}, 'takes')
        refused({'templates': [{'expr': dict(leaf, column='nosuch')}]}, 'no column nosuch')
        refused({'templates': [{'expr': dict(leaf, column='s0')}]}, 'no column s0 that takes a overlap leaf')
        refused({'templates': [{'expr': leaf}], 'reference_table': 't9'}, 'reference_table t9')
        refused({'templates': [{'expr': leaf}]}, '--mode replay', '--mode', 'replay')
        refused({'templates': [{'name': 'x', 'expr': leaf}, {'name': 'x', 'expr': leaf}]}, 'same name')
        wide = {'op': 'overlap', 'column': 'm0', 'count': 1}
        refused({'templates': [{'name': 'deep', 'expr': {'op': 'and', 'args': [wide] * 65}}]}, 'template deep')
        self.assertFalse(os.path.exists(os.path.join(self.temp.name, 'refused.frames')))

    def describe(self, frames, *extra):
        stdout, _ = self.run_tool(['describe', '--frames', frames, '--index-root', self.root, *extra])
        return json.loads(stdout)

    def expected_description(self, entries):
        """describe's numbers from check entries (one line per request), by the reference."""
        def operators(expr, found):
            if expr['op'] in ('and', 'or'):
                found[expr['op']] += len(expr['args']) - 1
            if expr['op'] == 'not':
                found['not'] += 1
            for arg in expr.get('args', []) + ([expr['arg']] if 'arg' in expr else []):
                operators(arg, found)
            return found

        subs = [dict(spec, expr=mf.entry_expr(spec)) for request in entries for spec in request]
        tables, ks, leaves, lengths = {}, {}, {}, {}
        found = {'and': 0, 'or': 0, 'not': 0}
        bins = [[0, 0] for _ in EDGES]
        for spec in subs:
            tables[spec['table']] = tables.get(spec['table'], 0) + 1
            ks[str(spec['k'])] = ks.get(str(spec['k']), 0) + 1
            operators(spec['expr'], found)
            for leaf in self.leaves(spec['expr']):
                key = (leaf['column'], leaf['op'], leaf.get('negate', False))
                leaves[key] = leaves.get(key, 0) + 1
                if 'values' in leaf:
                    lengths.setdefault(leaf['column'], []).append(len(leaf['values']))
            count = sum(reference(spec['expr'], self.tables[spec['table']]))
            rate = count / self.tables[spec['table']].rows
            at = 0 if count == 0 else next(i for i, edge in enumerate(EDGES) if rate <= edge)
            bins[at][0] += 1
            bins[at][1] += count
        exprs = {json.dumps(spec['expr'], sort_keys=True) for spec in subs}
        histogram = {}
        for request in entries:
            distinct = {json.dumps(mf.entry_expr(spec), sort_keys=True) for spec in request} - {'{"op": "all"}'}
            histogram[str(len(distinct))] = histogram.get(str(len(distinct)), 0) + 1
        duplicates = {}
        for request in entries:
            same = len(request) - len({json.dumps([s['table'], s['k'], s['vector'], mf.entry_expr(s)])
                                       for s in request})
            duplicates[str(same)] = duplicates.get(str(same), 0) + 1
        return {
            'requests': len(entries), 'subrequests_per_request': len(entries[0]),
            'tables': {'count': len(tables), 'subrequests_min': min(tables.values()),
                       'subrequests_max': max(tables.values())},
            'k': ks, 'dimensions': [{str(len(spec['vector'])): len(entries)} for spec in entries[0]],
            'distinct_programs': len(exprs),
            'distinct_table_programs': len({(spec['table'], json.dumps(spec['expr'], sort_keys=True))
                                            for spec in subs}),
            'leaves': [{'column': c, 'op': o, 'negate': n, 'count': leaves[c, o, n]} for c, o, n in sorted(leaves)],
            'operators': found,
            'values_per_leaf': {column: {'min': min(sizes), 'median': sorted(sizes)[(len(sizes) - 1) // 2],
                                         'max': max(sizes)} for column, sizes in sorted(lengths.items())},
            'duplicate_subrequests_per_request': duplicates, 'distinct_filters_per_request': histogram,
            'unfiltered_share': sum(spec['expr'] == {'op': 'all'} for spec in subs) / len(subs),
            'pass_rate': [{'subrequests': sends, 'share': sends / len(subs),
                           'eligible_share': rows / sum(rows for _, rows in bins)} for sends, rows in bins],
        }

    def test_describe(self):
        spec = self.write('describe.json', dict(self.SPEC, templates=self.SPEC['templates'] + [
            {'name': 'nothing', 'expr': {'op': 'values', 'column': 'allnull', 'values': [1]}},
            {'name': 'rare', 'expr': {'op': 'values', 'column': 's1', 'values': [1000], 'negate': True}}]))
        for name, extra in (('captured', ()), ('spec', ('--filter-spec', spec))):
            frames, _, _ = self.emit(f'describe-{name}.frames', *extra, '--unfiltered-share', '0.2',
                                     '--check-requests', '25')
            got = self.describe(frames, '--processes', '1')
            want = self.expected_description(self.entries(frames))
            self.assertEqual(got.pop('frames'), frames)
            self.assertEqual([row.pop('pass_rate') for row in got['pass_rate']], LABELS)
            self.assertEqual(got, want, name)
            self.assertAlmostEqual(sum(row['share'] for row in got['pass_rate']), 1)
            self.assertAlmostEqual(sum(row['eligible_share'] for row in got['pass_rate']), 1)
            self.assertGreater(sum(row['subrequests'] > 0 for row in got['pass_rate']), 2)
            self.assertGreater(got['unfiltered_share'], 0.05)
            # The captured request has one exact duplicate (r2 twice) in every request.
            self.assertEqual(sum(int(n) * count for n, count in got['duplicate_subrequests_per_request'].items()) >= 25,
                             True)
            # The first requests alone are described as a file of them would be; more than the file has is all.
            for requests in (1, 7):
                first = self.describe(frames, '--requests', str(requests), '--processes', '1')
                first.pop('frames')
                self.assertEqual([row.pop('pass_rate') for row in first['pass_rate']], LABELS)
                self.assertEqual(first, self.expected_description(self.entries(frames)[:requests]), (name, requests))
                self.assertNotEqual(first['pass_rate'], got['pass_rate'])
            whole = self.describe(frames, '--processes', '1')
            self.assertEqual(self.describe(frames, '--requests', '1000', '--processes', '1'), whole)
            # Tables counted in processes say the same.
            self.assertEqual(self.describe(frames, '--processes', '3'), whole)

    def test_describe_pass_rate_edges(self):
        """A pass rate that is exactly an edge is in the bin the edge closes, and one row more is in the next."""
        rows = 1000000
        root = os.path.join(self.temp.name, 'edges')
        Plain(root, 'million', {'n': np.arange(rows)})
        counts = [0, 1, 2, 100, 101, 1000, 1001, 3000, 3001, 10000, 10001, 20000, 20001, 30000, 30001, 50000, 50001,
                  100000, 100001, 250000, 250001, 500000, 500001, rows]
        self.assertEqual(sorted({round(edge * rows) + more for edge in EDGES for more in (0, 1)} - {rows + 1}), counts)
        frames = os.path.join(self.temp.name, 'edges.frames')
        with open(frames, 'wb') as out:
            out.write(struct.pack('<II', 1, len(counts)))
            for count in counts:
                sent = mf.frame('million', [0.5], 3, mf.encode({'op': 'range', 'column': 'n', 'low': 0,
                                                                'high': count - 1}))
                out.write(struct.pack('<I', len(sent)) + sent)
        stdout, _ = self.run_tool(['describe', '--frames', frames, '--index-root', root, '--processes', '1'])
        got = json.loads(stdout)
        # By bin: no row; one row, which is the first edge; then the row past the edge before, and the edge.
        binned = [[0], [1]] + [[low + 1, high] for low, high in zip(counts[1::2], counts[3::2])]
        self.assertEqual(got['pass_rate'], [
            {'pass_rate': label, 'subrequests': len(inside), 'share': len(inside) / len(counts),
             'eligible_share': sum(inside) / sum(counts)} for label, inside in zip(LABELS, binned)])
        self.assertEqual((got['requests'], got['subrequests_per_request'], got['distinct_programs']), (1, 24, 24))

    def worker(self, wrong=None):
        worker = FakeWorker(self.tables, wrong)
        worker.start()
        self.addCleanup(worker.join, 10)
        return worker, ['--host', '127.0.0.1', '--port', str(worker.port), '--token-file', self.token]

    def test_verify(self):
        spec = self.write('verify.json', self.SPEC)
        frames, summary, _ = self.emit('verify.frames', '--filter-spec', spec)
        pairs = {(e['table'], json.dumps(e['expr'])) for request in self.entries(frames) for e in request}
        # The distinct (table, program) pairs in the file's order, and how many there are by the end of each
        # request.
        ordered, by_request = [], []
        for request in mf.read_frames(frames):
            for sent in request:
                table, _, _, encoded = mf.parse_frame(sent)
                if (table, encoded) not in ordered:
                    ordered.append((table, encoded))
            by_request.append(len(ordered))
        self.assertEqual(len(ordered), len(pairs))
        worker, address = self.worker()
        stdout, _ = self.run_tool(['verify', '--frames', frames, '--index-root', self.root, *address])
        self.assertEqual(json.loads(stdout), {'compared': len(pairs), 'mismatched': 0, 'mismatches': []})
        self.assertEqual(worker.seen, [13] + [23] * len(pairs))            # each pair once, after the token
        self.assertEqual(worker.asked, ordered)
        self.assertGreater(len(pairs), 12)

        # --programs bounds the pairs, also where the bound falls inside a request; a worker that disagrees on
        # one table is reported and fails the run. Three ways to disagree: another row with its count, another
        # row under the right count, and the right rows under another count.
        bound = next(before + 1 for before, after in zip(by_request, by_request[1:]) if after - before > 1)
        self.assertNotIn(bound, by_request)

        def flipped(hits):
            return hits[:5] + [not hits[5]] + hits[6:]

        for wrong, count_off, rows_off in ((lambda table, count, hits: (sum(flipped(hits)), flipped(hits)), 1, 1),
                                           (lambda table, count, hits: (count, flipped(hits)), 0, 1),
                                           (lambda table, count, hits: (count + 1, hits), 1, 0)):
            worker, address = self.worker(lambda table, count, hits, wrong=wrong:
                                          wrong(table, count, hits) if table == 't1' else (count, hits))
            stdout = io.StringIO()
            with self.assertRaises(SystemExit) as raised, contextlib.redirect_stdout(stdout):
                mf.main(['verify', '--frames', frames, '--index-root', self.root, '--programs', str(bound), *address])
            self.assertEqual(raised.exception.code, 1)
            line = json.loads(stdout.getvalue())
            self.assertEqual(line['compared'], bound)
            self.assertEqual(worker.seen, [13] + [23] * bound)
            self.assertEqual(worker.asked, ordered[:bound])
            self.assertEqual(line['mismatched'], len(line['mismatches']))
            self.assertEqual(line['mismatched'], sum(table == 't1' for table, _ in ordered[:bound]))
            self.assertTrue(0 < line['mismatched'] < bound)
            for entry in line['mismatches']:
                self.assertEqual((entry['table'], entry['status'], entry['rows_differing']), ('t1', 0, rows_off))
                self.assertEqual(abs(entry['worker'] - entry['expected']), count_off)
                self.assertEqual(entry['expected'], sum(reference(entry['expr'], self.tables['t1'])))

        # A worker that refuses a program this evaluator accepts is a mismatch too (here: an index it lacks).
        worker = FakeWorker({'t0': self.tables['t0']})
        worker.start()
        self.addCleanup(worker.join, 10)
        with self.assertRaises(SystemExit), contextlib.redirect_stdout(io.StringIO()) as stdout:
            mf.main(['verify', '--frames', frames, '--index-root', self.root, '--host', '127.0.0.1',
                     '--port', str(worker.port), '--token-file', self.token])
        self.assertTrue(all(entry['status'] == 3 and entry['table'] != 't0'
                            for entry in json.loads(stdout.getvalue())['mismatches']))

    def test_check_against_exact_worker(self):
        """check's truth, for old-format entries and for expressions, is what an exact worker returns."""
        spec = self.write('check.json', self.SPEC)
        for name, extra in (('old', ()), ('expr', ('--filter-spec', spec))):
            frames, _, _ = self.emit(f'check-{name}.frames', *extra, '--check-requests', '6')
            _, address = self.worker()
            stdout, _ = self.run_tool(['check', '--frames', frames, '--index-root', self.root, '--verbose', *address])
            *lines, total = [json.loads(line) for line in stdout.splitlines()]
            self.assertEqual((total['checked'], total['recall_min'], total['below_0.99']), (36, 1.0, 0), name)
            self.assertEqual(total['routes'], {'2': 36})
            entries = [spec for request in self.entries(frames) for spec in request]
            for line, entry in zip(lines, entries):
                rows = sum(reference(mf.entry_expr(entry), self.tables[entry['table']]))
                self.assertEqual((line['table'], line['eligible'], line['filtered']),
                                 (entry['table'], rows, entry['filtered']))
                self.assertEqual(line['returned'], min(rows, entry['k']))
        with open(self.token, 'w') as out:
            out.write('another-token-of-some-length\n')
        self.addCleanup(self.write_token)
        _, address = self.worker()
        with self.assertRaises(SystemExit) as raised:
            self.run_tool(['check', '--frames', frames, '--index-root', self.root, *address])
        self.assertIn('authentication failed', str(raised.exception))

    def write_token(self):
        with open(self.token, 'w') as out:
            out.write(TOKEN + '\n')

    def test_no_file_left_open(self):
        spec = self.write('open.json', self.SPEC)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            frames, _, _ = self.emit('open.frames', '--filter-spec', spec)
            self.emit('open-plain.frames')
            self.describe(frames, '--processes', '1')
            _, address = self.worker()
            self.run_tool(['verify', '--frames', frames, '--index-root', self.root, '--programs', '3', *address])
            gc.collect()
        self.assertEqual([str(warning.message) for warning in caught
                          if issubclass(warning.category, ResourceWarning)], [])


if __name__ == '__main__':
    unittest.main()
