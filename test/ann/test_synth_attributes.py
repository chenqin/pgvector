"""benchmarks/ann/synth_attributes.py on tiny index trees: links, manifests, determinism and the shares.

Needs numpy only: no worker, no PostgreSQL. python3 test/ann/test_synth_attributes.py
"""

import contextlib
import gc
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import warnings
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOL = os.path.join(ROOT, 'benchmarks', 'ann', 'synth_attributes.py')
sys.path.insert(0, os.path.dirname(TOOL))
import synth_attributes as synth  # noqa: E402

NULL = -2 ** 31
STRIDE = 1000
SPEC = {'columns': {
    'a0': {'kind': 'flag', 'true_share': 0.7},
    'cat': {'kind': 'categorical', 'values': 6, 'zipf_s': 1.0, 'base': 10, 'null_share': 0.1},
    'price': {'kind': 'lognormal', 'median': 50, 'sigma': 1.0, 'max': 500},
}}
UNTOUCHED = ('keys.i64', 'vectors.f32', 'attr-a1.i32', 'attr-m0.i32', 'attr-m0.offsets', 'shard-0.cagra')
GENERATED = ('attr-a0.i32', 'attr-cat.i32', 'attr-price.i32')


def scaled_keys(documents, copies):
    """Keys as scale_export.py makes them: copy j of document d is j * STRIDE + d."""
    return np.concatenate([np.asarray(documents, dtype=np.int64) + j * STRIDE for j in range(copies)])


def write_group(root, group, keys):
    """A source index directory: constant a0, a1, a multi-valued m0 and a stand-in shard file."""
    directory = os.path.join(root, group)
    os.makedirs(directory)
    rows = len(keys)
    rng = np.random.default_rng(rows)
    keys.astype('<i8').tofile(os.path.join(directory, 'keys.i64'))
    rng.standard_normal((rows, 4)).astype('<f4').tofile(os.path.join(directory, 'vectors.f32'))
    np.full(rows, 3, dtype='<i4').tofile(os.path.join(directory, 'attr-a0.i32'))
    np.full(rows, 9, dtype='<i4').tofile(os.path.join(directory, 'attr-a1.i32'))
    np.arange(2 * rows, dtype='<i4').tofile(os.path.join(directory, 'attr-m0.i32'))
    np.arange(0, 2 * rows + 1, 2, dtype='<u4').tofile(os.path.join(directory, 'attr-m0.offsets'))
    with open(os.path.join(directory, 'shard-0.cagra'), 'wb') as out:
        out.write(b'graph of ' + group.encode())
    lines = ['pgvector-ann-index 1', 'dimensions 4', f'rows {rows}', 'metric l2', 'dtype float32',
             f'source seg.{group}', 'key id', 'vector emb', 'shards 1', f'shard 0 0 {rows} shard-0.cagra',
             'attribute a0 attr-a0.i32', 'attribute a1 attr-a1.i32', 'attribute m0 attr-m0.i32',
             'multivalued m0 attr-m0.offsets', 'generation 00000000000000ab']
    with open(os.path.join(directory, 'manifest'), 'w') as out:
        out.write('\n'.join(lines) + '\n')
    return lines


def read(root, group, file, dtype='<i4'):
    return np.fromfile(os.path.join(root, group, file), dtype=dtype)


def inode(root, group, file):
    return os.stat(os.path.join(root, group, file)).st_ino


def tree_bytes(root):
    """Every file of a tree by relative path."""
    found = {}
    for directory, _, files in os.walk(root):
        for file in files:
            path = os.path.join(directory, file)
            with open(path, 'rb') as handle:
                found[os.path.relpath(path, root)] = handle.read()
    return found


class SynthAttributesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.src = os.path.join(self.temp.name, 'src')
        documents = np.arange(60)
        self.keys = {
            'g0': scaled_keys(documents, 4),
            'g1': scaled_keys(documents, 4),                 # g0's keys: shares g0's generated files
            'g2': scaled_keys(documents[10:45], 3),          # another size
            'g10': scaled_keys(documents + 5, 4),            # g0's size, other keys: told apart by digest
        }
        self.lines = {group: write_group(self.src, group, keys) for group, keys in self.keys.items()}
        self.before = tree_bytes(self.src)

    def run_tool(self, out, spec=SPEC, extra=()):
        """Run main in this process; returns the JSON lines by group and what went to stderr."""
        spec_path = os.path.join(self.temp.name, f'spec-{os.path.basename(out)}.json')
        with open(spec_path, 'w') as file:
            json.dump(spec, file)
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            synth.main(['--index-root', self.src, '--out', out, '--spec', spec_path, '--key-stride', str(STRIDE),
                        *extra])
        lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
        return {line['group']: line for line in lines}, stderr.getvalue()

    def out(self, name='dst'):
        return os.path.join(self.temp.name, name)

    def test_links(self):
        dst = self.out()
        lines, stderr = self.run_tool(dst)
        self.assertEqual(list(lines), ['g0', 'g1', 'g2', 'g10'])     # one line a group, g2 before g10
        self.assertEqual(stderr, '')
        for group in self.keys:
            for file in UNTOUCHED:
                self.assertEqual(inode(dst, group, file), inode(self.src, group, file), (group, file))
            # A replaced column and the manifest are new files, not writes through a link.
            self.assertNotEqual(inode(dst, group, 'attr-a0.i32'), inode(self.src, group, 'attr-a0.i32'))
            self.assertNotEqual(inode(dst, group, 'manifest'), inode(self.src, group, 'manifest'))
            self.assertEqual(sorted(os.listdir(os.path.join(dst, group))), sorted(UNTOUCHED + GENERATED + ('manifest',)))
            self.assertEqual(lines[group]['source_files_linked'], len(UNTOUCHED))
            self.assertEqual(lines[group]['files_copied'], 0)
        for file in GENERATED:
            self.assertEqual(inode(dst, 'g1', file), inode(dst, 'g0', file), file)
            self.assertEqual(len({inode(dst, group, file) for group in ('g0', 'g2', 'g10')}), 3, file)
        self.assertEqual(lines['g1']['same_keys_as'], 'g0')
        self.assertEqual([lines[group]['same_keys_as'] for group in ('g0', 'g2', 'g10')], [None] * 3)
        self.assertEqual(lines['g1']['columns'], lines['g0']['columns'])
        self.assertEqual(tree_bytes(self.src), self.before)          # the source is untouched

    def test_manifest(self):
        dst = self.out()
        self.run_tool(dst)
        for group, source in self.lines.items():
            at = source.index('attribute m0 attr-m0.i32') + 1
            expected = source[:at] + ['attribute cat attr-cat.i32', 'attribute price attr-price.i32'] + source[at:]
            with open(os.path.join(dst, group, 'manifest')) as file:
                self.assertEqual(file.read(), '\n'.join(expected) + '\n')
            for file in GENERATED:
                self.assertEqual(os.path.getsize(os.path.join(dst, group, file)), 4 * len(self.keys[group]))

    def test_manifest_without_attributes(self):
        """New columns go at the end of a manifest that has no attribute line."""
        directory = os.path.join(self.src, 'g0')
        kept = [line for line in self.lines['g0'] if not line.startswith(('attribute', 'multivalued'))]
        with open(os.path.join(directory, 'manifest'), 'w') as out:
            out.write('\n'.join(kept) + '\n')
        # The manifest no longer names a0, and a file left where a new column's goes is refused (test_refusals).
        os.remove(os.path.join(directory, 'attr-a0.i32'))
        dst = self.out()
        lines, _ = self.run_tool(dst, extra=('--groups', 'g0'))
        self.assertEqual(list(lines), ['g0'])
        self.assertEqual(os.listdir(dst), ['g0'])
        with open(os.path.join(dst, 'g0', 'manifest')) as file:
            self.assertEqual(file.read().splitlines(), kept + [
                'attribute a0 attr-a0.i32', 'attribute cat attr-cat.i32', 'attribute price attr-price.i32'])
        self.assertTrue(all(column['new'] for column in lines['g0']['columns'].values()))
        # Files the manifest does not name are linked like any other.
        self.assertEqual(inode(dst, 'g0', 'attr-a1.i32'), inode(self.src, 'g0', 'attr-a1.i32'))
        self.assertEqual(os.path.getsize(os.path.join(dst, 'g0', 'attr-a0.i32')), 4 * len(self.keys['g0']))

    def test_deterministic(self):
        first, again, chunked, reseeded = (self.out(name) for name in ('a', 'b', 'c', 'd'))
        self.run_tool(first)
        self.run_tool(again)
        self.run_tool(chunked, extra=('--chunk-rows', '7'))
        self.run_tool(reseeded, extra=('--seed', '8'))
        self.assertEqual(tree_bytes(first), tree_bytes(again))
        self.assertEqual(tree_bytes(first), tree_bytes(chunked))
        for file in GENERATED:
            self.assertFalse(np.array_equal(read(first, 'g0', file), read(reseeded, 'g0', file)), file)
        # by row is chunk-independent too.
        row = dict(SPEC, by='row')
        self.run_tool(self.out('e'), spec=row)
        self.run_tool(self.out('f'), spec=row, extra=('--chunk-rows', '7'))
        self.assertEqual(tree_bytes(self.out('e')), tree_bytes(self.out('f')))

    def values_by(self, dst, file, of):
        """{of(key): the set of values its rows hold} over every group."""
        found = {}
        for group, keys in self.keys.items():
            for key, value in zip(keys.tolist(), read(dst, group, file).tolist()):
                found.setdefault(of(key), set()).add(value)
        return found

    def test_same_document_same_value(self):
        dst = self.out()
        self.run_tool(dst)
        for file in GENERATED:
            by_document = self.values_by(dst, file, lambda key: key % STRIDE)
            self.assertEqual(len(by_document), 65)
            self.assertTrue(all(len(values) == 1 for values in by_document.values()), file)
            # Documents differ from each other: the column selects something.
            self.assertGreater(len(set.union(*by_document.values())), 1, file)

    def test_by_row_same_key_same_value(self):
        dst = self.out()
        lines, _ = self.run_tool(dst, spec=dict(SPEC, by='row'))
        self.assertEqual(lines['g0']['by'], 'row')
        self.assertEqual(lines['g1']['same_keys_as'], 'g0')
        shared = set(self.keys['g0'].tolist()) & set(self.keys['g10'].tolist())
        self.assertTrue(0 < len(shared) < len(self.keys['g0']))       # overlapping, not identical, key sets
        for file in GENERATED:
            by_key = self.values_by(dst, file, lambda key: key)
            self.assertTrue(all(len(values) == 1 for values in by_key.values()), file)
            # Copies of a document are independent rows here.
            by_document = self.values_by(dst, file, lambda key: key % STRIDE)
            self.assertTrue(any(len(values) > 1 for values in by_document.values()), file)

    def test_summary_matches_files(self):
        """The printed distinct count, NULL share and top values are those of the files written."""
        for name, spec in (('document', SPEC), ('row', dict(SPEC, by='row'))):
            dst = self.out(name)
            lines, _ = self.run_tool(dst)if spec is SPEC else self.run_tool(dst, spec=spec)
            for group, keys in self.keys.items():
                self.assertEqual(lines[group]['rows'], len(keys))
                for column, reported in lines[group]['columns'].items():
                    self.assertEqual(reported['file'], f'attr-{column}.i32')
                    self.assertEqual(reported['new'], column != 'a0')
                    values = read(dst, group, reported['file'])
                    distinct, counts = np.unique(values[values != NULL], return_counts=True)
                    order = np.lexsort((distinct, -counts))[:5]
                    self.assertEqual(reported['distinct'], len(distinct), (name, group, column))
                    self.assertEqual(reported['null_share'], round(float((values == NULL).mean()), 6))
                    self.assertEqual(reported['top'], [[int(distinct[i]), round(int(counts[i]) / len(keys), 6)]
                                                       for i in order])

    def test_categorical_up_to_int32_max(self):
        """The last value may be INT32_MAX: base 2**31 - 2 with two values is a valid column."""
        dst = self.out()
        spec = {'columns': {'x': {'kind': 'categorical', 'values': 2, 'zipf_s': 0, 'base': 2 ** 31 - 2}}}
        lines, _ = self.run_tool(dst, spec=spec, extra=('--groups', 'g0'))
        self.assertEqual(np.unique(read(dst, 'g0', 'attr-x.i32')).tolist(), [2 ** 31 - 2, 2 ** 31 - 1])
        self.assertEqual(lines['g0']['columns']['x']['distinct'], 2)

    def test_by_row_summary_is_bounded(self):
        """By row the summary counts a group's first SAMPLE_ROWS rows, so it never holds more values than that."""
        spec = {'by': 'row', 'columns': {'wide': {'kind': 'lognormal', 'median': 1e6, 'sigma': 2.0, 'null_share': 0.1},
                                         'flag': {'kind': 'flag', 'true_share': 0.3}}}
        exact, _ = self.run_tool(self.out('exact'), spec=spec)
        self.assertEqual([line['sampled_rows'] for line in exact.values()], [None] * 4)
        self.assertGreater(exact['g0']['columns']['wide']['distinct'], 200)
        held = []
        tally = synth.tally

        def spy(total, values, weights):
            held.append(len(total[0]) + len(values))
            return tally(total, values, weights)

        for name, chunk in (('whole', ()), ('chunked', ('--chunk-rows', '7'))):
            dst = self.out(name)
            with mock.patch.object(synth, 'SAMPLE_ROWS', 100), mock.patch.object(synth, 'tally', spy):
                lines, _ = self.run_tool(dst, spec=spec, extra=chunk)
            self.assertEqual(tree_bytes(dst), tree_bytes(self.out('exact')))      # the files do not depend on it
            for group, keys in self.keys.items():
                self.assertEqual(lines[group]['sampled_rows'], 100, group)
                self.assertEqual(lines[group]['rows'], len(keys))
                for column, reported in lines[group]['columns'].items():
                    values = read(dst, group, reported['file'])[:100]
                    distinct, counts = np.unique(values[values != NULL], return_counts=True)
                    order = np.lexsort((distinct, -counts))[:5]
                    self.assertEqual(reported['distinct'], len(distinct), (name, group, column))
                    self.assertEqual(reported['null_share'], round(float((values == NULL).mean()), 6))
                    self.assertEqual(reported['top'], [[int(distinct[i]), round(int(counts[i]) / 100, 6)]
                                                       for i in order])
        self.assertLessEqual(max(held), 100)
        # By document the table is exact whatever the bound.
        with mock.patch.object(synth, 'SAMPLE_ROWS', 100):
            lines, _ = self.run_tool(self.out('document'), spec=dict(spec, by='document'))
        self.assertEqual([line['sampled_rows'] for line in lines.values()], [None] * 4)

    def test_no_file_left_open(self):
        """Every file the tool opens is closed: no ResourceWarning for the manifests, keys or columns."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            self.run_tool(self.out())
            self.run_tool(self.out('row'), spec=dict(SPEC, by='row'))
            gc.collect()
        self.assertEqual([str(warning.message) for warning in caught
                          if issubclass(warning.category, ResourceWarning)], [])

    def test_shares(self):
        """Each kind's distribution on 400,000 ids, by document table and by key."""
        ids = np.arange(400000, dtype=np.int64) * 7 + 3
        columns, _ = synth.load_spec({'columns': {
            'flag': {'kind': 'flag', 'true_share': 0.97},
            'zipf': {'kind': 'categorical', 'values': 50, 'zipf_s': 1.0, 'base': 100},
            'flat': {'kind': 'categorical', 'values': 8, 'zipf_s': 0},
            'steep': {'kind': 'categorical', 'values': 1000, 'zipf_s': 2.0, 'base': -5},
            'price': {'kind': 'lognormal', 'median': 1000, 'sigma': 1.0, 'max': 1000000},
            'capped': {'kind': 'lognormal', 'median': 1000, 'sigma': 2.0, 'max': 3000},
            'holes': {'kind': 'categorical', 'values': 4, 'zipf_s': 0, 'null_share': 0.01},
            'empty': {'kind': 'flag', 'true_share': 0.5, 'null_share': 1},
        }})
        got = {name: synth.generate(column, 7, name, ids) for name, column in columns.items()}
        for name, values in got.items():
            self.assertEqual(values.dtype, np.dtype('<i4'), name)
            self.assertEqual(len(values), len(ids), name)

        self.assertAlmostEqual(got['flag'].mean(), 0.97, delta=0.002)
        self.assertEqual(set(np.unique(got['flag']).tolist()), {0, 1})

        weights = 1.0 / np.arange(1, 51)
        shares = np.bincount(got['zipf'] - 100, minlength=50) / len(ids)
        self.assertEqual((got['zipf'].min(), got['zipf'].max()), (100, 149))
        np.testing.assert_allclose(shares, weights / weights.sum(), atol=0.003)

        np.testing.assert_allclose(np.bincount(got['flat'], minlength=8) / len(ids), 1 / 8, atol=0.003)
        self.assertEqual(len(np.unique(got['flat'])), 8)

        weights = np.arange(1, 1001) ** -2.0
        self.assertAlmostEqual((got['steep'] == -5).mean(), weights[0] / weights.sum(), delta=0.003)
        self.assertTrue(-5 <= got['steep'].min() and got['steep'].max() <= 994)

        price = got['price']
        self.assertTrue(1 <= price.min() and price.max() <= 1000000)
        self.assertAlmostEqual(float(np.median(price)), 1000, delta=30)
        self.assertAlmostEqual(float(np.log(price).mean()), np.log(1000), delta=0.02)
        self.assertAlmostEqual(float(np.log(price).std()), 1.0, delta=0.02)
        self.assertGreater(len(np.unique(price)), 5000)

        # Clipped, not redrawn: P(value > 3000) = P(normal > ln(3) / 2) = 0.2914 of rows sit on max.
        self.assertEqual(got['capped'].max(), 3000)
        self.assertAlmostEqual((got['capped'] == 3000).mean(), 0.2914, delta=0.004)
        self.assertGreaterEqual(got['capped'].min(), 1)

        holes = got['holes']
        self.assertAlmostEqual((holes == NULL).mean(), 0.01, delta=0.001)
        # NULLs fall on every value alike: the rest keep the value distribution.
        np.testing.assert_allclose(np.bincount(holes[holes != NULL], minlength=4) / (holes != NULL).sum(), 0.25,
                                   atol=0.003)
        self.assertTrue((got['empty'] == NULL).all())

        # Columns are independent of each other, and a value depends on its id alone.
        self.assertAlmostEqual(float(np.corrcoef(got['flat'], got['zipf'])[0, 1]), 0, delta=0.01)
        self.assertAlmostEqual(got['flag'][holes == NULL].mean(), 0.97, delta=0.02)
        picked = np.random.default_rng(1).permutation(len(ids))[:5000]
        for name, column in columns.items():
            np.testing.assert_array_equal(synth.generate(column, 7, name, ids[picked]), got[name][picked])
        self.assertFalse(np.array_equal(synth.generate(columns['zipf'], 8, 'zipf', ids), got['zipf']))
        self.assertFalse(np.array_equal(synth.generate(columns['zipf'], 7, 'other', ids), got['zipf']))

    def test_shares_in_written_group(self):
        """A larger group end to end: 60,000 documents, 4 copies each, the file's shares and the report."""
        src = os.path.join(self.temp.name, 'big')
        keys = scaled_keys(np.arange(60000), 4).astype(np.int64)
        keys = np.concatenate([np.arange(60000, dtype=np.int64) + j * 100000 for j in range(4)])
        write_group(src, 'g0', keys)
        spec_path = os.path.join(self.temp.name, 'big.json')
        with open(spec_path, 'w') as file:
            json.dump({'columns': {'active': {'kind': 'flag', 'true_share': 0.97, 'null_share': 0.02},
                                   'country': {'kind': 'categorical', 'values': 50}}}, file)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            synth.main(['--index-root', src, '--out', self.out('big-dst'), '--spec', spec_path, '--chunk-rows', '50000'])
        line = json.loads(stdout.getvalue())
        active, country = (read(self.out('big-dst'), 'g0', f'attr-{name}.i32') for name in ('active', 'country'))
        self.assertEqual(len(active), 240000)
        self.assertAlmostEqual((active == NULL).mean(), 0.02, delta=0.003)
        self.assertAlmostEqual(active[active != NULL].mean(), 0.97, delta=0.004)
        self.assertEqual(line['columns']['active']['null_share'], round(float((active == NULL).mean()), 6))
        self.assertEqual(line['columns']['active']['top'][0][0], 1)
        weights = 1.0 / np.arange(1, 51)
        np.testing.assert_allclose(np.bincount(country, minlength=50) / len(country), weights / weights.sum(),
                                   atol=0.006)
        self.assertEqual(line['columns']['country']['distinct'], 50)
        self.assertEqual([value for value, _ in line['columns']['country']['top']], [0, 1, 2, 3, 4])
        # Every copy of a document has its document's value.
        np.testing.assert_array_equal(country.reshape(4, 60000), np.tile(country[:60000], (4, 1)))

    def test_existing_output_refused(self):
        dst = self.out()
        os.makedirs(os.path.join(dst, 'g2'))
        spec_path = os.path.join(self.temp.name, 'spec.json')
        with open(spec_path, 'w') as file:
            json.dump(SPEC, file)
        command = [sys.executable, TOOL, '--index-root', self.src, '--out', dst, '--spec', spec_path,
                   '--key-stride', str(STRIDE)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn('g2', result.stderr)
        self.assertIn('exists', result.stderr)
        self.assertEqual(result.stdout, '')
        self.assertEqual(os.listdir(dst), ['g2'])                     # nothing was written for the others
        self.assertEqual(os.listdir(os.path.join(dst, 'g2')), [])
        # The other groups alone are fine, and a second run over them is refused in turn.
        for expected in (0, 1):
            result = subprocess.run(command + ['--groups', 'g0,g1'], capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, expected, result.stderr)
        self.assertEqual(sorted(os.listdir(dst)), ['g0', 'g1', 'g2'])
        self.assertEqual(tree_bytes(self.src), self.before)

    def test_copy_when_links_fail(self):
        dst, linked = self.out(), self.out('linked')
        self.run_tool(linked)
        with mock.patch.object(synth.os, 'link', side_effect=OSError(18, 'Cross-device link')):
            lines, stderr = self.run_tool(dst)
        self.assertEqual(tree_bytes(dst), tree_bytes(linked))
        for group in self.keys:
            self.assertIn(f'{group}: warning: hard links failed (Cross-device link)', stderr)
            for file in UNTOUCHED:
                self.assertNotEqual(inode(dst, group, file), inode(self.src, group, file))
        self.assertEqual(stderr.count('\n'), len(self.keys))          # one warning a group
        self.assertEqual(lines['g0']['files_copied'], len(UNTOUCHED))
        self.assertEqual(lines['g1']['files_copied'], len(UNTOUCHED) + len(GENERATED))
        self.assertNotEqual(inode(dst, 'g1', 'attr-cat.i32'), inode(dst, 'g0', 'attr-cat.i32'))

    def refused(self, spec, message, extra=()):
        dst = self.out('refused')
        with self.assertRaises(SystemExit) as raised:
            self.run_tool(dst, spec=spec, extra=extra)
        self.assertIn(message, str(raised.exception))
        self.assertFalse(os.path.lexists(dst), 'a refused run wrote something')

    def test_refusals(self):
        flag = {'kind': 'flag', 'true_share': 0.5}
        self.refused({'columns': {'m0': flag}}, 'multi-valued')
        self.refused({'columns': {'a0': flag}, 'by': 'group'}, 'by must be')
        self.refused({'columns': {}}, 'at least one column')
        self.refused({'columns': {'x': {'kind': 'gaussian'}}}, 'kind must be one of')
        self.refused({'columns': {'x': {'kind': 'flag'}}}, 'needs true_share')
        self.refused({'columns': {'x': dict(flag, values=3)}}, 'unknown field values')
        self.refused({'columns': {'x': dict(flag, null_share=1.5)}}, 'null_share')
        self.refused({'columns': {'x y': flag}}, 'name')
        self.refused({'columns': {'x' * 64: flag}}, 'name')
        self.refused({'columns': {'x': {'kind': 'categorical', 'values': 0}}}, 'values')
        self.refused({'columns': {'x': {'kind': 'categorical', 'values': 2, 'base': NULL}}}, 'base')
        self.refused({'columns': {'x': {'kind': 'categorical', 'values': 2, 'base': 2 ** 31 - 1}}}, 'base')
        self.refused({'columns': {'x': {'kind': 'lognormal', 'median': 10, 'max': 2 ** 31}}}, 'max')
        self.refused({'columns': {f'c{i}': flag for i in range(30)}}, 'more than')
        self.refused(SPEC, 'no index directory', extra=('--groups', 'g0,g7'))
        # A stray file where a new column's would go is not overwritten or linked over.
        with open(os.path.join(self.src, 'g2', 'attr-cat.i32'), 'wb') as out:
            out.write(b'stray')
        self.refused(SPEC, 'already has a file attr-cat.i32')


if __name__ == '__main__':
    unittest.main()
