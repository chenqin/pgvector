#!/usr/bin/env python3
"""Give ANN index directories attribute columns that select something, without copying the vectors.

A scaled test tree (scale_export.py) inherits its seed's attributes, and where every single-valued
column of the seed holds one constant no filter on them selects anything. This writes a second tree
whose groups share every file with the source's except the attribute columns a spec generates:

    python3 benchmarks/ann/synth_attributes.py --index-root /data/scale4715/c1 \\
        --out /data/scale4715/c1-attrs --spec spec.json

For every group directory under --index-root (a directory with a manifest, as ann/common/INDEX_FORMAT.md
describes; --groups picks some) DST/GROUP gets a hard link to every file of the source except the
manifest and the files of the columns being replaced, the generated attr files, and last a manifest equal
to the source's with one `attribute NAME attr-NAME.i32` line added per new column, after the source's
last attribute line. A column the source already has keeps its line and its file name. Files are copied,
with a warning on stderr, where a hard link is refused (another filesystem). An existing DST/GROUP is
refused before anything is written.

SPEC.json is {"columns": {NAME: COLUMN, ...}, "by": "document" | "row"}. Every column is a single-valued
i32 one, of kind:

    {"kind": "flag", "true_share": 0.97}                              1 with that probability, else 0
    {"kind": "categorical", "values": 50, "zipf_s": 1.0, "base": 0}   base .. base + values - 1; the r-th
                                                                      (from 1) has weight r ** -zipf_s, so
                                                                      zipf_s 0 is uniform
    {"kind": "lognormal", "median": 1000, "sigma": 1.0, "max": 1000000}
                                                                      exp(normal(log median, sigma)) rounded
                                                                      and clipped to 1 .. max

zipf_s (1.0), base (0), sigma (1.0) and max (2147483647) may be left out. Any kind takes "null_share": that
share of values is NULL (INT32_MIN), independently of the value it replaces.

"by": "document" (the default) gives a value to the seed document, key % --key-stride, so every copy of a
document in every group has the same attributes, as scale_export.py's inherited ones do. "by": "row" gives
one to each key. Either way a value is a function of (--seed, column name, document or key) alone: it does
not depend on the group, on the other rows, or on --chunk-rows, so the same key has the same value in every
group and a rerun writes the same bytes.

Groups whose keys.i64 are identical (same size, then same BLAKE2 digest) get the same columns: the later
group's generated files are hard links to the first's. Keys are read --chunk-rows at a time, so memory does
not grow with the group; by document it also holds --key-stride values per column.

One JSON line per group goes to stdout: rows, the file of each column and whether it is new, and per
column the distinct non-NULL values, the NULL share and the most common values with their share of rows.
By document these are exact, from the document table weighted by each document's rows. By row they are
counted as written over the group's first 1,048,576 rows, so they are exact for a group that small and a
sample of a larger one, whose line says so in "sampled_rows" (a value does not depend on its row's place,
so the first rows are a fair sample; "distinct" is then only what the sample holds). Counting every row
would keep every distinct value of a wide column, as many as the group has rows.

Refused: a spec column that is multi-valued or a text (dictionary) attribute in the source, since only
single-valued integer columns are generated; more than 32 attributes; a new column whose attr-NAME.i32
already exists in the source directory.
"""

import argparse
import functools
import hashlib
import json
import os
import re
import shutil
import sys
import time

import numpy as np

BUFFER = 16 << 20           # bytes of output buffered per file
NULL = -2 ** 31             # ANN_FILTER_NULL_VALUE
INT32_MAX = 2 ** 31 - 1
MAX_ATTRIBUTES = 32         # ANN_INDEX_MAX_ATTRIBUTES
TOP = 5                     # most common values reported per column
SAMPLE_ROWS = 1 << 20       # by row, the rows of a group the summary counts: bounds the distinct values held
VALUE, SECOND, NULLS = 0, 1, 2      # a column's independent streams of uniforms

# Fields of each kind and their defaults; None marks a required one.
KINDS = {
    'flag': {'true_share': None},
    'categorical': {'values': None, 'zipf_s': 1.0, 'base': 0},
    'lognormal': {'median': None, 'sigma': 1.0, 'max': INT32_MAX},
}


def mix(x):
    """SplitMix64's finalizer over a uint64 array."""
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def salt(seed, name):
    """A column's 64-bit seed. hashlib, because hash() of a string differs from process to process."""
    return int.from_bytes(hashlib.blake2b(f'{seed}\0{name}'.encode(), digest_size=8).digest(), 'little')


def uniform(ids, column_salt, stream):
    """A float64 in [0, 1) per id: a hash of (column, stream, id), so it depends on nothing else."""
    step = np.uint64((stream + 1) * 0x9E3779B97F4A7C15 % 2 ** 64)
    return (mix(mix(ids ^ np.uint64(column_salt)) + step) >> np.uint64(11)).astype(np.float64) * 2.0 ** -53


@functools.lru_cache(maxsize=None)
def zipf_cdf(values, zipf_s):
    """Cumulative popularity of ranks 1..values with weight rank ** -zipf_s."""
    weights = np.arange(1, values + 1, dtype=np.float64) ** -zipf_s
    return np.cumsum(weights / weights.sum())


def generate(column, seed, name, ids):
    """A column's '<i4' values for these documents or keys (int64), each from (seed, name, id) alone."""
    ids = np.ascontiguousarray(ids, dtype=np.int64).view(np.uint64)
    column_salt = salt(seed, name)
    u = uniform(ids, column_salt, VALUE)
    if column['kind'] == 'flag':
        out = (u < column['true_share']).astype('<i4')
    elif column['kind'] == 'categorical':
        # The float sum can end a hair under 1: an id above it takes the last rank.
        rank = np.searchsorted(zipf_cdf(column['values'], float(column['zipf_s'])), u, side='right')
        out = (np.minimum(rank, column['values'] - 1) + column['base']).astype('<i4')
    else:
        # Box-Muller; 1 - u is in (0, 1], so the logarithm is finite.
        normal = np.sqrt(-2.0 * np.log(1.0 - u)) * np.cos(2.0 * np.pi * uniform(ids, column_salt, SECOND))
        with np.errstate(over='ignore'):
            out = np.rint(np.clip(column['median'] * np.exp(column['sigma'] * normal), 1, column['max'])).astype('<i4')
    if column['null_share'] > 0:
        out[uniform(ids, column_salt, NULLS) < column['null_share']] = NULL
    return out


def number(where, column, field, low, high, integer=False):
    """Check one numeric spec field's type and range."""
    value = column[field]
    if isinstance(value, bool) or not isinstance(value, int if integer else (int, float)) or not low <= value <= high:
        raise SystemExit(f'{where}: {field} must be {"an integer" if integer else "a number"} in '
                         f'{low}..{high}, not {value!r}')


def load_spec(spec):
    """Validate a parsed SPEC.json; returns its columns with the defaults filled in, and "by"."""
    if not isinstance(spec, dict) or set(spec) - {'columns', 'by'}:
        raise SystemExit('spec: expected {"columns": {NAME: {...}}, "by": "document" | "row"}')
    by = spec.get('by', 'document')
    if by not in ('document', 'row'):
        raise SystemExit(f'spec: by must be "document" or "row", not {by!r}')
    if not isinstance(spec.get('columns'), dict) or not spec['columns']:
        raise SystemExit('spec: columns must name at least one column')
    columns = {}
    for name, given in spec['columns'].items():
        where = f'spec column {name!r}'
        # The server reads a manifest name with %63s, and the name is part of a file name.
        if not re.fullmatch(r'[^\s/\x00]+', name) or len(name.encode()) > 63:
            raise SystemExit(f'{where}: a name is 1 to 63 bytes without whitespace or /')
        if not isinstance(given, dict) or given.get('kind') not in KINDS:
            raise SystemExit(f'{where}: kind must be one of {", ".join(KINDS)}')
        fields = dict(KINDS[given['kind']], null_share=0.0)
        unknown = sorted(set(given) - set(fields) - {'kind'})
        missing = sorted(field for field, default in fields.items() if default is None and field not in given)
        if unknown or missing:
            raise SystemExit(f'{where}: ' + '; '.join(
                text for text in (unknown and f'unknown field {", ".join(unknown)}',
                                  missing and f'{given["kind"]} needs {", ".join(missing)}') if text))
        column = dict(fields, **given)
        number(where, column, 'null_share', 0, 1)
        if column['kind'] == 'flag':
            number(where, column, 'true_share', 0, 1)
        elif column['kind'] == 'categorical':
            number(where, column, 'values', 1, 2 ** 31, integer=True)
            number(where, column, 'zipf_s', 0, 100)
            # No value may be NULL's, and the last must fit an i32.
            number(where, column, 'base', NULL + 1, INT32_MAX - (column['values'] - 1), integer=True)
        else:
            number(where, column, 'median', 1e-9, INT32_MAX)
            number(where, column, 'sigma', 0, 100)
            number(where, column, 'max', 1, INT32_MAX, integer=True)
        columns[name] = column
    return columns, by


def natural(name):
    """Sort key that puts g2 before g10."""
    return [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', name)]


def plan(root, group, columns):
    """Read a group's manifest and decide the output: rows, each spec column's file, the new manifest.

    Raises SystemExit for a group this tool cannot write, so main can refuse before writing anything.
    """
    directory = os.path.join(root, group)
    with open(os.path.join(directory, 'manifest')) as file:
        lines = file.read().splitlines()
    rows, attributes, other, last = None, {}, {}, None
    for at, line in enumerate(lines):
        parts = line.split()
        if not parts or line.startswith('#'):
            continue
        if parts[0] == 'rows' and len(parts) == 2:
            rows = int(parts[1])
        elif parts[0] == 'attribute' and len(parts) == 3:
            attributes[parts[1]] = parts[2]
            last = at
        elif parts[0] in ('multivalued', 'dictionary') and len(parts) > 1:
            other[parts[1]] = 'a multi-valued' if parts[0] == 'multivalued' else 'a text (dictionary)'
    if rows is None:
        raise SystemExit(f'{group}: no rows line in {directory}/manifest')
    size = os.path.getsize(os.path.join(directory, 'keys.i64'))
    if size != rows * 8:
        raise SystemExit(f'{group}: keys.i64 is {size} bytes, not the manifest\'s {rows} rows')
    for name in columns:
        if name in other:
            raise SystemExit(f'{group}: column {name} is {other[name]} attribute in the source; '
                             'only single-valued integer columns can be replaced')
    new = [name for name in columns if name not in attributes]
    if len(attributes) + len(new) > MAX_ATTRIBUTES:
        raise SystemExit(f'{group}: {len(attributes)} attributes plus {len(new)} new ones is more than '
                         f'the server\'s {MAX_ATTRIBUTES}')
    files = {name: attributes.get(name, f'attr-{name}.i32') for name in columns}
    for name in new:
        if files[name] in attributes.values() or os.path.lexists(os.path.join(directory, files[name])):
            raise SystemExit(f'{group}: the source already has a file {files[name]}, which new column {name} needs')
    if len(set(files.values())) != len(files):
        raise SystemExit(f'{group}: two spec columns share a file in the source manifest')
    at = len(lines) if last is None else last + 1
    added = [f'attribute {name} {files[name]}' for name in new]
    return {'rows': rows, 'files': files, 'new': new, 'manifest': lines[:at] + added + lines[at:]}


def digest(path):
    """BLAKE2 digest of a file, read in pieces."""
    state = hashlib.blake2b(digest_size=16)
    with open(path, 'rb') as file:
        for piece in iter(lambda: file.read(BUFFER), b''):
            state.update(piece)
    return state.digest()


def same_keys(root, group, seen):
    """The earlier group whose keys.i64 holds this group's bytes, or None.

    seen maps a size to {group: digest or None} for the groups written so far: sizes are compared first,
    so a keys file is hashed only once another group's is as large.
    """
    path = os.path.join(root, group, 'keys.i64')
    peers = seen.setdefault(os.path.getsize(path), {})
    if not peers:
        peers[group] = None
        return None
    for peer in peers:
        if peers[peer] is None:
            peers[peer] = digest(os.path.join(root, peer, 'keys.i64'))
    mine = digest(path)
    found = next((peer for peer in peers if peers[peer] == mine), None)
    if found is None:
        peers[group] = mine
    return found


def link(source, target, failures):
    """Hard-link source as target (a new name); copy instead, noting why in failures, where that is refused."""
    try:
        os.link(os.path.realpath(source), target)
    except OSError as error:
        shutil.copyfile(source, target)
        failures.append(error.strerror or str(error))


def tally(total, values, weights):
    """Add each value's weight to total, a pair (sorted distinct values, their counts)."""
    distinct, inverse = np.unique(np.concatenate([total[0], values]), return_inverse=True)
    counts = np.bincount(inverse.ravel(), weights=np.concatenate([total[1], weights]), minlength=len(distinct))
    return distinct, counts


def summary(total, rows):
    """A column's distinct non-NULL values, NULL share and most common values with their share of rows."""
    present = total[1] > 0
    distinct, counts = total[0][present], total[1][present]
    nulls = float(counts[distinct == NULL].sum())
    distinct, counts = distinct[distinct != NULL], counts[distinct != NULL]
    order = np.lexsort((distinct, -counts))[:TOP]
    return {'distinct': len(distinct), 'null_share': round(nulls / rows, 6),
            'top': [[int(distinct[i]), round(float(counts[i]) / rows, 6)] for i in order]}


def write_columns(args, source, target, group_plan, columns, tables):
    """Generate the spec's columns for one group's keys into target, a chunk of keys at a time.

    tables holds each column's value per document, or is None when values go by key. Returns each
    column's summary and, when it covers only the group's first rows, how many (else None).
    """
    rows, files = group_plan['rows'], group_plan['files']
    empty = np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    totals = {name: empty for name in columns}
    per_document = np.zeros(args.key_stride if tables else 0, dtype=np.int64)
    counted = rows if tables else min(rows, SAMPLE_ROWS)
    # 'xb': a generated file is always a new file, never a write through a hard link into another tree.
    outs = {name: open(os.path.join(target, files[name]), 'xb', BUFFER) for name in columns}
    try:
        with open(os.path.join(source, 'keys.i64'), 'rb') as keys_file:
            done = 0
            while done < rows:
                keys = np.fromfile(keys_file, dtype='<i8', count=min(args.chunk_rows, rows - done))
                if not len(keys):
                    raise SystemExit(f'{source}/keys.i64 ended after {done} of {rows} rows')
                sample = max(0, min(len(keys), counted - done))
                done += len(keys)
                if tables:
                    documents = keys % args.key_stride
                    per_document += np.bincount(documents, minlength=args.key_stride)
                for name, column in columns.items():
                    values = tables[name][documents] if tables else generate(column, args.seed, name, keys)
                    outs[name].write(values.data)
                    if not tables and sample:
                        totals[name] = tally(totals[name], values[:sample], np.ones(sample))
    finally:
        for out in outs.values():
            out.close()
    if tables:
        totals = {name: tally(empty, tables[name], per_document.astype(np.float64)) for name in columns}
    return {name: summary(totals[name], counted) for name in columns}, counted if counted < rows else None


def synthesize(args, group, group_plan, columns, tables, origin):
    """Write DST/GROUP and return its JSON summary and what write_columns returned for its columns.

    origin is (group, its plan, what write_columns returned) for an earlier group with the same keys,
    whose generated files this group links, or None.
    """
    started = time.time()
    source, target = os.path.join(args.index_root, group), os.path.join(args.out, group)
    files = group_plan['files']
    try:
        os.makedirs(target, exist_ok=False)
    except FileExistsError:
        raise SystemExit(f'{group}: output directory {target} already exists; remove it or choose another --out')
    failures, linked = [], 0
    replaced = set(files.values()) | {'manifest', 'manifest.tmp'}
    for entry in sorted(os.scandir(source), key=lambda entry: entry.name):
        if entry.name in replaced:
            continue
        if not entry.is_file():
            print(f'{group}: {entry.name} is not a regular file; not linked', file=sys.stderr)
            continue
        link(entry.path, os.path.join(target, entry.name), failures)
        linked += 1
    if origin is None:
        stats, sampled = write_columns(args, source, target, group_plan, columns, tables)
    else:
        stats, sampled = origin[2]
        for name in columns:
            link(os.path.join(args.out, origin[0], origin[1]['files'][name]), os.path.join(target, files[name]),
                 failures)
    if failures:
        print(f'{group}: warning: hard links failed ({failures[0]}); {len(failures)} files were copied',
              file=sys.stderr)
    # Manifest last, by rename, so a partial directory never loads.
    with open(os.path.join(target, 'manifest.tmp'), 'x') as out:
        out.write('\n'.join(group_plan['manifest']) + '\n')
    os.replace(os.path.join(target, 'manifest.tmp'), os.path.join(target, 'manifest'))
    line = {'group': group, 'rows': group_plan['rows'], 'by': 'document' if tables else 'row',
            'columns': {name: dict(stats[name], file=files[name], new=name in group_plan['new']) for name in columns},
            'sampled_rows': sampled, 'source_files_linked': linked, 'files_copied': len(failures),
            'same_keys_as': origin[0] if origin else None, 'seconds': round(time.time() - started, 1)}
    return line, (stats, sampled)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--index-root', required=True, help='directory of GROUP index directories')
    parser.add_argument('--out', required=True, help='directory the GROUP directories are written under')
    parser.add_argument('--spec', required=True, help='JSON: the columns to generate (see above)')
    parser.add_argument('--groups', metavar='GROUP[,GROUP...]',
                        help='write only these groups (default: every directory of --index-root with a manifest)')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--key-stride', type=int, default=100000,
                        help='scale_export.py\'s: a key\'s seed document is key %% KEY_STRIDE')
    parser.add_argument('--chunk-rows', type=int, default=1 << 22, help='keys read at a time')
    args = parser.parse_args(argv)
    if args.key_stride < 1 or args.chunk_rows < 1:
        raise SystemExit('--key-stride and --chunk-rows must be positive')
    with open(args.spec) as file:
        columns, by = load_spec(json.load(file))
    if args.groups is not None:
        groups = [group for group in args.groups.split(',') if group]
        missing = [group for group in groups
                   if not os.path.isfile(os.path.join(args.index_root, group, 'manifest'))]
        if missing or not groups or len(set(groups)) != len(groups):
            raise SystemExit(f'--groups: no index directory (or a repeated one) for {", ".join(missing) or "(empty)"} '
                             f'under {args.index_root}')
    else:
        groups = sorted((entry.name for entry in os.scandir(args.index_root)
                         if os.path.isfile(os.path.join(entry.path, 'manifest'))), key=natural)
        if not groups:
            raise SystemExit(f'no index directory (a directory with a manifest) under {args.index_root}')
    plans = {group: plan(args.index_root, group, columns) for group in groups}
    existing = [group for group in groups if os.path.lexists(os.path.join(args.out, group))]
    if existing:
        raise SystemExit(f'output directory exists for {", ".join(existing)} under {args.out}; '
                         'remove them or choose another --out (nothing was written)')
    os.makedirs(args.out, exist_ok=True)
    tables = None
    if by == 'document':
        documents = np.arange(args.key_stride, dtype=np.int64)
        tables = {name: generate(column, args.seed, name, documents) for name, column in columns.items()}
    seen, written = {}, {}
    for group in groups:
        first = same_keys(args.index_root, group, seen)
        origin = (first, plans[first], written[first]) if first is not None else None
        line, written[group] = synthesize(args, group, plans[group], columns, tables, origin)
        print(json.dumps(line), flush=True)


if __name__ == '__main__':
    main()
