#!/usr/bin/env python3
"""Production-shaped ANN requests for a GPU node, without PostgreSQL.

Mocks what PostgreSQL would send for a production request's ANN sub-requests:
one worker operation 22 (search by predicate program) per sub-request, built
from an ANN capture (each sub-request's tag, k, group, vector and pg_ids
filter), the group catalog and term codes (JSON, dumped once from the
segment), and the index directories themselves (dimensions, row vectors to
draw queries from).

    # requests for benchmarks/ann/request_direct.c
    request_frames.py emit --capture C.json --groups groups.json --terms terms.json \\
        --index-root /data/scale100/i8 --out requests.frames --requests 20000
    # exact recall of the first requests' answers, from the index files
    request_frames.py check --frames requests.frames --index-root /data/scale100/i8 \\
        --host 127.0.0.1 --port 7700 --token-file token
    # what a frames file asks for: request shape, filters, pass rates
    request_frames.py describe --frames requests.frames --index-root /data/scale100/i8
    # the worker's rows for the file's filters against this file's evaluator
    request_frames.py verify --frames requests.frames --index-root /data/scale100/i8 \\
        --host 127.0.0.1 --port 7700 --token-file token

Mixed requests (the default) keep each sub-request's tag, k and filter shape:
the group is drawn from every group of its dimension, the vector from that
group's rows plus noise, the pg_ids terms from a pool of --distinct-filters
random subsets of the captured terms, and --unfiltered-share run without a
filter. Sub-requests that share a group and vector in the capture (rankers
asking the same group with the same user embedding) keep sharing the drawn
group and vector, and those sharing a filter keep sharing its drawn terms, so
a request's duplicate sub-requests stay duplicates; --independent draws every
sub-request on its own. Worker index names are the group tables (load them
under those names).

emit --filter-spec FILE replaces the captured filters with ones drawn from a JSON spec, leaving the rest of
the request as it is: for the same --seed the tables, vectors, k and unfiltered sub-requests are those of an
emit without it, and sub-requests that shared a filter share the drawn one.

    {"distinct": 5000, "popularity": "uniform" | "zipf", "zipf_s": 1.1, "reference_table": "g0",
     "templates": [{"weight": 1.0, "pass": [lo, hi], "expr": TEMPLATE, "name": "wide"}, ...]}

emit draws "distinct" filters once, each from a template chosen by weight (a draw equal to an earlier one is
redrawn up to 20 times, and not at all while its template's last round of redraws gave nothing new, so a
template without generators repeats), and every shared filter of every request takes one of them, uniformly
or with the r-th drawn (from 1) weighing r ** -zipf_s. A TEMPLATE is a filter expression (see encode) whose
leaves may be generators instead of explicit operands:

    {"op": "overlap" | "values", "column": C, "pass": [lo, hi], "pad": N, "negate": bool}
        random distinct values of the column, added until the rows holding one of them are a share of the
        table drawn log-uniformly from [lo, hi] (the share before negation). No value that would take the
        share over hi is added. A list that ends below lo that way (a light value taken first can leave a
        heavy one no room) is drawn again heaviest value first, and the one with more rows is used. That
        is a rule of thumb, not a search of every list: the list is below lo where the rule does not land
        inside, as when the column's values are too heavy, or run out, or the most a leaf may hold are not
        enough. When every value alone is over hi the one nearest the drawn share is used.
    {"op": "overlap" | "values", "column": C, "count": N, "pad": N, "negate": bool}
        exactly N random distinct values of the column.
    {"op": "range", "column": C, "pass": [lo, hi], "negate": bool}
        low and high among the column's values: a run of consecutive values whose rows are a share inside
        [lo, hi], from a random low among those with such a run, and of that low's runs inside the first
        to reach a share drawn the same way, else the longest. Where no low has one, as in a column of a
        few heavy values, the run nearest [lo, hi] of all is used: the whole column when its non-NULL rows
        are fewer than lo.

"pad": N extends a generated list to N values with values no row of any table of --groups holds (never
INT32_MIN), so the same rows pass, on every table. A generated overlap list has at most 65,535 values
("pad" and "count" too), one under the worker's limit: request_direct --unique-share adds a value to the
overlap leaf a program starts with (programs that start otherwise it sends as they are). A template's own
"pass": [lo, hi] redraws the whole expression until its share of rows is inside, and emit fails after 200
draws, with the shares it saw. Shares and the values to choose from are those of the reference table
(default: the first group of --groups); the pool serves every table, as the captured variants do, so every
table needs the templates' columns, and check and describe count each table's own rows. Everything is a
function of --seed.
"""

import argparse
import bisect
import collections
import concurrent.futures
import hashlib
import json
import math
import os
import random
import socket
import struct
import sys

import numpy as np

# ann/common/filter.h
OPS = {'values': 1, 'range': 2, 'null': 3, 'and': 5, 'or': 6, 'not': 7, 'all': 8, 'none': 9, 'overlap': 10}
NAMES = {code: name for name, code in OPS.items()}
VALUES, AND, ALL, OVERLAP = OPS['values'], OPS['and'], OPS['all'], OPS['overlap']
LEAVES = ('values', 'range', 'null', 'overlap')
NULL = -2 ** 31             # ANN_FILTER_NULL_VALUE
INT32_MAX = 2 ** 31 - 1
MAX_VALUES, MAX_STEPS, MAX_DEPTH, MAX_PROGRAM = 65536, 1024, 64, 1 << 20

MAX_DRAWN_OVERLAP = MAX_VALUES - 1  # a generated overlap list: request_direct --unique-share adds a value to it
REDRAWS = 200               # draws of a template for its own "pass" before emit gives up
DUPLICATE_REDRAWS = 20      # draws for a pool entry that is not an earlier one's program
RANGE_LOWS = 32             # random lows tried for a range leaf before every low is looked at
PASS_EDGES = (0, 1e-6, 1e-4, 1e-3, 3e-3, 1e-2, 2e-2, 3e-2, 5e-2, 1e-1, 0.25, 0.5, 1)    # describe's bins


def manifest(directory):
    fields = {}
    with open(os.path.join(directory, 'manifest')) as lines:
        for line in lines:
            parts = line.split()
            if parts and not parts[0].startswith('#'):
                fields.setdefault(parts[0], []).append(parts[1:])
    return fields


def load_json(path):
    with open(path) as file:
        return json.load(file)


def program(columns, ops, codes):
    """A captured filter's program: one leaf per clause, then the ANDs. No clause gives no bytes at all,
    which the worker refuses as malformed."""
    out = b''
    for column, op, values in zip(columns, ops, codes):
        name = column.encode()
        out += struct.pack('<BB', OVERLAP if op == 'overlap' else VALUES, len(name)) + name
        out += struct.pack('<BI', 0, len(values)) + struct.pack(f'<{len(values)}i', *values)
    return out + bytes([AND]) * (len(columns) - 1)


def steps(program):
    """A program's instructions in order, each (op name, column, negate, operand).

    The operand is a values or overlap leaf's values as an '<i4' array, a range leaf's (low, high), else
    None; column and negate are None for what is not a leaf. Raises ValueError for a program that
    AnnFilterCheck in ann/common/filter.c refuses, and for a stored-bitmap leaf, whose rows no index
    directory holds.
    """
    size = len(program)
    if size == 0 or size > MAX_PROGRAM:
        raise ValueError(f'a program is 1 to {MAX_PROGRAM} bytes, not {size}')
    out, depth, at = [], 0, 0
    while at < size:
        op = NAMES.get(program[at])
        column = negate = operand = None
        if op is None:
            raise ValueError(f'instruction {program[at]} at byte {at} is not one this tool reads')
        if op in LEAVES:
            length = program[at + 1] if size - at >= 3 else 0
            end = at + 2 + length
            if not 1 <= length <= 63 or size - at < 3 + length or 0 in program[at + 2:end] or program[end] > 1:
                raise ValueError(f'malformed {op} leaf at byte {at}')
            column, negate, start = program[at + 2:end].decode(), program[end] == 1, at
            at = end + 1
            if op == 'range':
                if size - at < 8:
                    raise ValueError(f'malformed range leaf at byte {start}')
                operand = struct.unpack_from('<ii', program, at)
                at += 8
            elif op != 'null':
                count = struct.unpack_from('<I', program, at)[0] if size - at >= 4 else MAX_VALUES + 1
                if count > MAX_VALUES or size - at - 4 < 4 * count:
                    raise ValueError(f'malformed {op} leaf at byte {start}: over {MAX_VALUES} values, or cut short')
                operand = np.frombuffer(program, dtype='<i4', count=count, offset=at + 4)
                at += 4 + 4 * count
            depth += 1
        else:
            if depth < {'and': 2, 'or': 2, 'not': 1}.get(op, 0):
                raise ValueError(f'{op} at byte {at} has too few operands')
            depth += {'and': -1, 'or': -1, 'not': 0}.get(op, 1)
            at += 1
        if depth > MAX_DEPTH:
            raise ValueError(f'the program needs a stack deeper than {MAX_DEPTH}')
        if len(out) == MAX_STEPS:
            raise ValueError(f'the program has more than {MAX_STEPS} instructions')
        out.append((op, column, negate, operand))
    if depth != 1:
        raise ValueError(f'the program leaves {depth} results, not 1')
    return out


def brief(value):
    """A value's repr for an error message, cut short."""
    try:
        return repr(value)[:200]
    except RecursionError:
        return 'a deeply nested value'


def encode(expr):
    """The postfix program of a filter expression, in ann/common/filter.h's layout.

    An expression is a JSON-able dict:

        {"op": "values" | "overlap", "column": C, "values": [ints], "negate": bool}
        {"op": "range", "column": C, "low": int, "high": int, "negate": bool}
        {"op": "null", "column": C, "negate": bool}
        {"op": "and" | "or", "args": [expr, expr, ...]}       {"op": "not", "arg": expr}
        {"op": "all"}                                          {"op": "none"}

    "negate" may be left out. An AND or OR writes its arguments and then its n - 1 operators, so an AND of
    leaves gives program()'s bytes; values are written as given, neither sorted nor deduplicated. Raises
    ValueError for a malformed expression and for what the worker would refuse: a leaf of more than 65,536
    values, more than 1,024 instructions, a program over 1 MiB, or a stack deeper than 64, which an AND of
    more than 64 arguments needs (nested as [[a AND b] AND c] it needs 2).
    """
    out, todo = [], [expr]
    try:
        while todo:
            item = todo.pop()
            if isinstance(item, bytes):
                out.append(item)
                continue
            op = item['op']
            if op in LEAVES:
                name, negate = item['column'].encode(), item.get('negate', False)
                if not 1 <= len(name) <= 63 or b'\0' in name or not isinstance(negate, bool):
                    raise ValueError(f'a column name is 1 to 63 bytes and negate a bool: {brief(item)}')
                step = struct.pack('<BB', OPS[op], len(name)) + name + bytes([negate])
                if op == 'range':
                    step += struct.pack('<ii', item['low'], item['high'])
                elif op != 'null':
                    values = item['values']
                    if len(values) > MAX_VALUES:
                        raise ValueError(f'{len(values)} values in a leaf on {item["column"]}: at most {MAX_VALUES}')
                    step += struct.pack('<I', len(values)) + struct.pack(f'<{len(values)}i', *values)
                out.append(step)
            elif op in ('and', 'or'):
                if len(item['args']) < 2:
                    raise ValueError(f'{op} takes 2 or more args')
                todo.append(bytes([OPS[op]]) * (len(item['args']) - 1))
                todo.extend(reversed(item['args']))
            elif op == 'not':
                todo.append(bytes([OPS[op]]))
                todo.append(item['arg'])
            elif op in ('all', 'none'):
                out.append(bytes([OPS[op]]))
            else:
                raise ValueError(f'unknown op {op!r}')
    except (KeyError, TypeError, AttributeError, struct.error) as error:
        raise ValueError(f'not a filter expression ({error!r}): {brief(item)}') from None
    encoded = b''.join(out)
    steps(encoded)
    return encoded


def decode(program):
    """The expression of a program: encode(decode(p)) is p for every program without a stored-bitmap leaf.

    An AND (or OR) whose last operand is an AND comes back as one AND of all the arguments, which is how
    encode writes that: decode(encode(e)) is e unless e nests an AND as the last argument of an AND, and
    then it is e with the two merged. Raises ValueError as steps does; the empty program is one it refuses.
    """
    stack = []
    for op, column, negate, operand in steps(program):
        if op in LEAVES:
            expr = {'op': op, 'column': column}
            if op == 'range':
                expr['low'], expr['high'] = operand
            elif op != 'null':
                expr['values'] = operand.tolist()
            expr['negate'] = negate
        elif op in ('and', 'or'):
            right, left = stack.pop(), stack.pop()
            expr = {'op': op, 'args': [left] + (right['args'] if right['op'] == op else [right])}
        elif op == 'not':
            expr = {'op': op, 'arg': stack.pop()}
        else:
            expr = {'op': op}
        stack.append(expr)
    return stack[0]


def legacy_expr(columns, ops, codes, filtered=True):
    """The expression of a capture-shaped filter: what program(columns, ops, codes) encodes, or all rows.

    One clause is its leaf and several are their AND. No clause is every row: that is what check has always
    counted for it, though program() writes it as no bytes, which the worker refuses.
    """
    leaves = [{'op': 'overlap' if op == 'overlap' else 'values', 'column': column, 'values': list(values),
               'negate': False} for column, op, values in zip(columns, ops, codes)]
    if not filtered or not leaves:
        return {'op': 'all'}
    return leaves[0] if len(leaves) == 1 else {'op': 'and', 'args': leaves}


def entry_expr(spec):
    """A .check.jsonl entry's filter: its "expr", or the older columns/ops/codes/filtered form's."""
    if 'expr' in spec:
        return spec['expr']
    return legacy_expr(spec['columns'], spec['ops'], spec['codes'], spec['filtered'])


class Columns:
    """A table's attribute columns, read from its index directory as filters name them, and the rows a
    filter selects.

    mask() and count() give the rows ann/common/filter.c gives for the same program: NULL is INT32_MIN; a
    values, overlap or range leaf never matches a NULL row, and negated it matches the non-NULL rows it
    does not; INT32_MIN in a value list is skipped and a range's low starts above it; a null leaf matches
    the NULL rows and negated every other; NOT complements over all rows. values and range are for
    single-valued columns and overlap for multi-valued ones (ValueError otherwise, as for an unknown
    column). A multi-valued row is NULL when its array is, stored as the sole element INT32_MIN; an empty
    array has no element, so it matches no overlap and every negated one.

    A column is inverted once, the way ann/server/indexfile.c does it: its sorted distinct values and,
    value by value, the rows holding it in ascending order (a multi-valued row once under each of its
    distinct elements), so a leaf costs time in the rows it matches, not in the table. Rows travel as a
    sorted array of row numbers while they are few (rows / 64, filter.c's bound) and as a bool mask beyond.
    """

    def __init__(self, directory):
        self.directory = directory
        self.fields = manifest(directory)
        self.rows = int(self.fields['rows'][0][0])
        self.files = {a[0]: a[1] for a in self.fields.get('attribute', [])}
        self.offsets = {m[0]: m[1] for m in self.fields.get('multivalued', [])}
        self.few = max(self.rows // 64, 1)
        self.loaded = {}

    def column(self, name):
        """A column's sorted distinct values, where each one's rows start in the postings (and, last, where
        they end), the postings, and whether the column is multi-valued."""
        if name in self.loaded:
            return self.loaded[name]
        if name not in self.files:
            raise ValueError(f'{self.directory} has no attribute column {name}')
        values = np.fromfile(os.path.join(self.directory, self.files[name]), dtype='<i4')
        holders = None
        if name in self.offsets:
            offsets = np.fromfile(os.path.join(self.directory, self.offsets[name]), dtype='<u4').astype(np.int64)
            lengths = np.diff(offsets)
            if len(offsets) != self.rows + 1 or offsets[0] != 0 or (lengths < 0).any() or offsets[-1] != len(values):
                raise ValueError(f'{self.directory}: the offsets of {name} do not fit its {len(values)} elements')
            holders = np.repeat(np.arange(self.rows, dtype=np.uint32), lengths)
            # The worker refuses to load such an index, so there are no rows of it to mirror.
            if (lengths[holders[values == NULL]] > 1).any():
                raise ValueError(f'{self.directory}: a row of {name} has a NULL element')
        elif len(values) != self.rows:
            raise ValueError(f'{self.directory}: {name} has {len(values)} values for {self.rows} rows')
        order = np.argsort(values, kind='stable')
        values = values[order]
        postings = order.astype(np.uint32) if holders is None else holders[order]
        del order
        if holders is not None and len(values) > 1:
            keep = np.ones(len(values), dtype=bool)
            keep[1:] = (values[1:] != values[:-1]) | (postings[1:] != postings[:-1])
            values, postings = values[keep], postings[keep]
        starts = np.concatenate([np.zeros(min(len(values), 1), dtype=np.int64),
                                 np.flatnonzero(values[1:] != values[:-1]) + 1])
        self.loaded[name] = (values[starts], np.append(starts, len(values)), postings, holders is not None)
        return self.loaded[name]

    def leaf(self, op, name, negate, operand):
        """The rows of one leaf."""
        distinct, starts, postings, multi = self.column(name)
        if (op != 'null') and (multi != (op == 'overlap')):
            raise ValueError(f'{op} leaf on the {"multi" if multi else "single"}-valued column {name}')
        nulls = [postings[:starts[1]]] if len(distinct) and distinct[0] == NULL else []
        if op == 'null':
            runs = nulls
        elif op == 'range':
            # Keys of the column's own type: for a Python int numpy converts the whole array, every time.
            low, high = np.int32(max(operand[0], NULL + 1)), np.int32(operand[1])
            first = int(np.searchsorted(distinct, low, side='left'))
            end = int(np.searchsorted(distinct, high, side='right'))
            runs = [postings[starts[first]:starts[end]]] if low <= high and first < end else []
        else:
            wanted = np.unique(operand)
            wanted = wanted[wanted != NULL]
            at = np.searchsorted(distinct, wanted)
            inside = at < len(distinct)
            at = at[inside][distinct[at[inside]] == wanted[inside]]
            runs = [postings[starts[i]:starts[i + 1]] for i in at.tolist()]
        if negate and op != 'null':
            runs = runs + nulls
        if sum(len(run) for run in runs) <= self.few:
            found = np.concatenate(runs) if runs else np.empty(0, dtype=np.uint32)
            # A multi-valued row can be under several of the values; a range's run is in value order.
            if len(runs) > 1 or op == 'range':
                found = np.unique(found) if multi else np.sort(found)
        else:
            found = np.zeros(self.rows, dtype=bool)
            for run in runs:
                found[run] = True
        return ~self.bits(found) if negate else found

    def bits(self, selected):
        """Selected rows as a bool mask."""
        if selected.dtype == bool:
            return selected
        mask = np.zeros(self.rows, dtype=bool)
        mask[selected] = True
        return mask

    def select(self, predicate):
        """The rows a filter (an expression, or a program's bytes) selects: sorted row numbers, or a mask."""
        stack = []
        for op, column, negate, operand in steps(predicate if isinstance(predicate, bytes) else encode(predicate)):
            if op in LEAVES:
                stack.append(self.leaf(op, column, negate, operand))
            elif op == 'all':
                stack.append(np.ones(self.rows, dtype=bool))
            elif op == 'none':
                stack.append(np.empty(0, dtype=np.uint32))
            elif op == 'not':
                stack.append(~self.bits(stack.pop()))
            else:
                b, a = stack.pop(), stack.pop()
                if a.dtype == bool and b.dtype != bool:
                    a, b = b, a
                if op == 'or':
                    few = b.dtype != bool and len(a) + len(b) <= self.few
                    stack.append(np.union1d(a, b) if few else self.bits(a) | self.bits(b))
                elif a.dtype == bool:
                    stack.append(a & b)
                else:
                    stack.append(a[b[a]] if b.dtype == bool else np.intersect1d(a, b, assume_unique=True))
        return stack[0]

    def mask(self, predicate):
        """The rows a filter (an expression, or a program's bytes) selects, as a bool per row."""
        return self.bits(self.select(predicate))

    def count(self, predicate):
        """How many rows a filter selects."""
        selected = self.select(predicate)
        return int(np.count_nonzero(selected)) if selected.dtype == bool else len(selected)


def eligible(columns, spec):
    """The rows of a table (its Columns) that a .check.jsonl entry's filter selects, as a bool per row."""
    return columns.mask(entry_expr(spec))


def frame(index, vector, k, prog):
    name = index.encode()
    payload = struct.pack('<I', len(name)) + name + struct.pack('<IIII', len(vector), k, 0, len(prog))
    payload += prog + np.asarray(vector, dtype='<f4').tobytes()
    return b'PVAN' + struct.pack('<HHQQII', 1, 22, 1, 0, len(payload), 0) + payload


def drawn_share(rng, low, high):
    """A share of the rows drawn log-uniformly from [low, high]."""
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def rows_inside(rows, low, high):
    """The fewest and the most rows that are a share of a table's rows inside [low, high], the share being
    count / rows as for a pool entry. The fewest is over the most when no count is inside."""
    least, most = math.ceil(low * rows), math.floor(high * rows)
    while least > 0 and (least - 1) / rows >= low:
        least -= 1
    while least / rows < low:
        least += 1
    while most < rows and (most + 1) / rows <= high:
        most += 1
    while most / rows > high:
        most -= 1
    return least, most


def pick_values(columns, name, low, high, rng):
    """Random distinct values of a column whose rows together are a drawn share of the table (see the
    module's notes on generator leaves for what happens when the column cannot give it)."""
    distinct, starts, _, multi = columns.column(name)
    counts = np.diff(starts)
    order = rng.permutation(np.flatnonzero(distinct != NULL))
    if not len(order):
        return []
    need, limit = drawn_share(rng, low, high) * columns.rows, MAX_DRAWN_OVERLAP if multi else MAX_VALUES
    least, top = rows_inside(columns.rows, low, high)

    def fill(rest):
        """Values taken in the order given, and the rows holding one of them."""
        chosen, reached = rest[:0], 0
        while reached < need and len(chosen) < limit:
            rest = rest[reached + counts[rest] <= top]
            if not len(rest):
                break
            total = reached + np.cumsum(counts[rest])
            take = int(np.searchsorted(total, need))
            # The value that reaches the share is taken unless it goes over the top: then it is passed over
            # and the values after it are tried.
            over = bool(take < len(rest) and total[take] > top)
            take = min(take + (take < len(rest) and not over), limit - len(chosen))
            chosen, rest, reached = np.concatenate([chosen, rest[:take]]), rest[take + over:], int(total[take - 1])
            if multi:
                # Rows holding several of the values count once, so the sum above is only a bound.
                reached = columns.count({'op': 'overlap', 'column': name, 'values': distinct[chosen].tolist()})
        return chosen, reached

    chosen, reached = fill(order)
    if reached < least:
        # Light values taken first can leave no room for a heavy one the share needs: heaviest first, then.
        heaviest = fill(order[np.argsort(-counts[order], kind='stable')])
        if heaviest[1] > reached:
            chosen = heaviest[0]
    if not len(chosen):
        chosen = order[[np.argmin(np.abs(counts[order] - need))]]
    return sorted(distinct[chosen].tolist())


def held_values(directories, name):
    """The non-NULL values of a column that some row of any of the tables holds, sorted."""
    held = np.empty(0, dtype='<i4')
    for directory in directories:
        files = {a[0]: a[1] for a in manifest(directory).get('attribute', [])}
        held = np.union1d(held, np.fromfile(os.path.join(directory, files[name]), dtype='<i4'))
    return held[held != NULL].astype(np.int64)


def pad_values(held, values, size, rng):
    """values extended to size values with ones outside held (a column's values, sorted, without NULL) and
    never NULL; sorted."""
    low, high = (int(held[0]), int(held[-1])) if len(held) else (0, 0)
    # Draw from a span of which at least half is free, so most draws are values no row holds.
    spare = max(0, 2 * (len(held) + size) - (high - low + 1))
    low, high = max(low - spare // 2, NULL + 1), min(high + spare - spare // 2, INT32_MAX)
    padded = set(values)
    while len(padded) < size:
        draws = rng.integers(low, high + 1, size=2 * (size - len(padded)))
        at = np.minimum(np.searchsorted(held, draws), max(len(held) - 1, 0))
        for value in (draws[held[at] != draws] if len(held) else draws).tolist():
            if len(padded) < size:
                padded.add(value)
    return sorted(padded)


def pick_range(columns, name, low, high, rng):
    """low and high among a single-valued column's values: a run of them whose rows are a share of the table
    inside [low, high] when any run's are, else the run nearest (see the module's notes on generator leaves)."""
    distinct, starts, _, multi = columns.column(name)
    first = 1 if len(distinct) and distinct[0] == NULL else 0
    edges = starts[first:]                  # the rows before each non-NULL value in the postings, then their end
    if multi or len(edges) < 2:
        raise SystemExit(f'filter spec: a range needs a single-valued column with values, which {name} is not')
    least, most = rows_inside(columns.rows, low, high)
    least = max(least, 1)
    need = round(drawn_share(rng, low, high) * columns.rows)
    lows = int(np.searchsorted(edges, edges[-1] - least, side='right'))     # the lows with that many rows from them on
    if lows == 0:
        return int(distinct[first]), int(distinct[-1])

    def ends(start):
        """Where the first run from start (one low, or an array of them) with the fewest rows ends, and the
        last with at most the most: a run is inside when the first is not past the last."""
        return (np.searchsorted(edges, edges[start] + least, side='left'),
                np.searchsorted(edges, edges[start] + most, side='right') - 1)

    for _ in range(RANGE_LOWS):
        start = int(rng.integers(lows))
        shortest, longest = ends(start)
        if shortest <= longest:
            break
    else:
        # Lows with a run inside are rare, or there is none: every low is looked at.
        every = np.arange(len(edges) - 1)
        shortest, longest = ends(every)
        fits = np.flatnonzero(shortest <= longest)
        if not len(fits):
            # Of every low's last run under the range and first run over it, the nearest.
            runs = np.stack([shortest - 1, np.minimum(shortest, len(edges) - 1)])
            inside = edges[runs] - edges[every]
            miss = np.where(runs > every, np.maximum(least - inside, inside - most), np.iinfo(np.int64).max)
            which, start = np.unravel_index(rng.choice(np.flatnonzero(miss.ravel() == miss.min())), miss.shape)
            return int(distinct[first + start]), int(distinct[first + runs[which, start] - 1])
        start = int(rng.choice(fits))
        shortest, longest = shortest[start], longest[start]
    end = min(max(int(np.searchsorted(edges, edges[start] + need, side='left')), int(shortest)), int(longest))
    return int(distinct[first + start]), int(distinct[first + end - 1])


def generator(leaf):
    """Whether a template's leaf is a generator: one without its operands."""
    if leaf['op'] == 'range':
        return 'low' not in leaf
    return leaf['op'] in ('values', 'overlap') and 'values' not in leaf


def template_leaves(template, where):
    """A template's leaves, after checking its shape; where names it in errors."""
    def fail(text, at):
        raise SystemExit(f'filter spec {where}: {text}: {json.dumps(at)[:200]}')

    def share(at, positive):
        value = at['pass']
        numbers = isinstance(value, list) and len(value) == 2 and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
        if not numbers or not 0 <= value[0] <= value[1] <= 1 or (positive and value[0] == 0):
            fail(f'"pass" is [lo, hi] with {"0 < lo" if positive else "0 <= lo"} <= hi <= 1', at)

    def whole(at, key, least):
        most = MAX_DRAWN_OVERLAP if at['op'] == 'overlap' else MAX_VALUES
        if isinstance(at[key], bool) or not isinstance(at[key], int) or not least <= at[key] <= most:
            fail(f'"{key}" of a generated {at["op"]} leaf is an integer from {least} to {most}', at)

    if 'pass' in template:
        share(template, False)
    leaves, todo = [], [template.get('expr')]
    while todo:
        at = todo.pop()
        op = at.get('op') if isinstance(at, dict) else None
        if op in ('and', 'or') and set(at) == {'op', 'args'} and isinstance(at['args'], list) and len(at['args']) >= 2:
            todo.extend(at['args'])
        elif op == 'not' and set(at) == {'op', 'arg'}:
            todo.append(at['arg'])
        elif op in LEAVES:
            allowed = {'op', 'column', 'negate'}
            if generator(at):
                allowed |= {'pass'} if op == 'range' else {'pad'} | ({'count'} if 'count' in at else {'pass'})
                if 'pass' in allowed and 'pass' not in at:
                    fail('a leaf without its operands needs "pass" or "count"', at)
                if 'pass' in allowed:
                    share(at, True)
                for key in allowed & set(at) & {'count', 'pad'}:
                    whole(at, key, 0 if key == 'pad' else 1)
            else:
                allowed |= {'range': {'low', 'high'}, 'null': set()}.get(op, {'values'})
            typed = isinstance(at.get('column'), str) and isinstance(at.get('negate', False), bool)
            if set(at) - allowed or not typed:
                fail(f'a {op} leaf here takes "column" and {sorted(allowed - {"op", "column"})}', at)
            leaves.append(at)
        elif op not in ('all', 'none') or set(at) != {'op'}:
            fail('not a filter expression or generator', at)
    return leaves


def load_filter_spec(path):
    """A --filter-spec file, checked, with the defaults filled in and each template's leaves."""
    spec = load_json(path)
    known = {'distinct', 'popularity', 'zipf_s', 'reference_table', 'templates'}
    if not isinstance(spec, dict) or set(spec) - known or not (isinstance(spec.get('templates'), list) and
                                                               spec['templates']):
        raise SystemExit(f'filter spec: expected an object with "templates" and any of {sorted(known)}')
    spec = dict({'distinct': 64, 'popularity': 'uniform', 'zipf_s': 1.0, 'reference_table': None}, **spec)
    if isinstance(spec['distinct'], bool) or not isinstance(spec['distinct'], int) or spec['distinct'] < 1:
        raise SystemExit('filter spec: "distinct" is a positive integer')
    if spec['popularity'] not in ('uniform', 'zipf') or not (isinstance(spec['zipf_s'], (int, float)) and
                                                             spec['zipf_s'] >= 0):
        raise SystemExit('filter spec: "popularity" is "uniform" or "zipf", and "zipf_s" is not negative')
    templates = []
    for at, template in enumerate(spec['templates']):
        if not isinstance(template, dict) or set(template) - {'weight', 'pass', 'expr', 'name'}:
            raise SystemExit(f'filter spec template {at}: expected {{"weight", "pass", "expr", "name"}}')
        template = dict({'weight': 1.0, 'name': f't{at}'}, **template)
        weight = template['weight']
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight <= 0:
            raise SystemExit(f'filter spec template {template["name"]}: "weight" is a positive number')
        template['leaves'] = template_leaves(template, f'template {template["name"]}')
        templates.append(template)
    if len({template['name'] for template in templates}) != len(templates):
        raise SystemExit('filter spec: two templates have the same name')
    return dict(spec, templates=templates)


def instantiate(template, columns, rng, held=None):
    """A template's expression with its generator leaves drawn on a table's columns.

    held gives, by column, the values a pad must not be (held_values of every table the filter is for);
    for a column it lacks they are the values of this table's rows.
    """
    if template['op'] in ('and', 'or'):
        return {'op': template['op'], 'args': [instantiate(arg, columns, rng, held) for arg in template['args']]}
    if template['op'] == 'not':
        return {'op': 'not', 'arg': instantiate(template['arg'], columns, rng, held)}
    if template['op'] not in LEAVES or not generator(template):
        return dict(template)
    name = template['column']
    drawn = {'op': template['op'], 'column': name}
    if template['op'] == 'range':
        drawn['low'], drawn['high'] = pick_range(columns, name, *template['pass'], rng)
    else:
        distinct, _, _, multi = columns.column(name)
        if multi != (template['op'] == 'overlap'):
            raise SystemExit(f'filter spec: a {template["op"]} leaf on the {"multi" if multi else "single"}-valued '
                             f'column {name}')
        own = distinct[distinct != NULL]
        if 'count' in template:
            if template['count'] > len(own):
                raise SystemExit(f'filter spec: "count": {template["count"]} on column {name}, which has '
                                 f'{len(own)} values in {columns.directory}')
            values = sorted(rng.choice(own, size=template['count'], replace=False).tolist())
        else:
            values = pick_values(columns, name, *template['pass'], rng)
        taken = (held or {}).get(name)
        drawn['values'] = pad_values(own.astype(np.int64) if taken is None else taken, values,
                                     template.get('pad', 0), rng)
    drawn['negate'] = template.get('negate', False)
    return drawn


def draw_filter(template, columns, rng, held=None, counted=None):
    """One filter of a template: its expression, program and share of the table's rows.

    held is instantiate's. counted keeps the share of every program drawn, by the program's digest (a
    padded program is kilobytes, and a template's "pass" can draw hundreds for one filter), so that a
    program drawn again is not counted again.
    """
    shares, counted = [], {} if counted is None else counted
    for _ in range(REDRAWS if 'pass' in template else 1):
        try:
            expr = instantiate(template['expr'], columns, rng, held)
            encoded = encode(expr)
            digest = hashlib.blake2b(encoded, digest_size=16).digest()
            if digest not in counted:
                counted[digest] = columns.count(encoded) / max(columns.rows, 1)
            share = counted[digest]
        except ValueError as error:
            raise SystemExit(f'filter spec template {template["name"]}: {error}')
        if 'pass' not in template or template['pass'][0] <= share <= template['pass'][1]:
            return {'expr': expr, 'program': encoded, 'share': share, 'template': template['name']}
        shares.append(share)
    low, high = template['pass']
    raise SystemExit(f'filter spec template {template["name"]}: none of {REDRAWS} draws selects a share of the rows of '
                     f'{columns.directory} in [{low:g}, {high:g}]; the {len(set(shares))} distinct shares drawn run '
                     f'from {min(shares):.3g} to {max(shares):.3g}, median {float(np.median(shares)):.3g}')


def filter_pool(spec, columns, seed, held=None):
    """A filter spec's pool, drawn on the reference table's columns: draw_filter's entries, in drawn order.

    held is instantiate's: without it a pad's values are ones no row of the reference table holds.
    """
    # Its own stream: the request's other draws stay those of an emit without a spec.
    rng = np.random.default_rng([seed, 0x66696C74])
    weights = np.cumsum([template['weight'] for template in spec['templates']]).tolist()
    pool, programs, counted, spent = [], set(), {}, set()
    for _ in range(spec['distinct']):
        template = spec['templates'][min(bisect.bisect_right(weights, rng.random() * weights[-1]), len(weights) - 1)]
        # A template whose every redraw was a program already in the pool has few others to give, or none:
        # its entries are drawn once until one of them is new.
        for _ in range(1 if template['name'] in spent else DUPLICATE_REDRAWS):
            entry = draw_filter(template, columns, rng, held, counted)
            if entry['program'] not in programs:
                spent.discard(template['name'])
                break
        else:
            spent.add(template['name'])
        programs.add(entry['program'])
        pool.append(entry)
    return pool


def emit(args):
    rng = random.Random(args.seed)
    nrng = np.random.default_rng(args.seed)
    capture = load_json(args.capture)
    catalog = load_json(args.groups)
    groups = {g['stem']: g for g in catalog}
    terms = {int(k): v for k, v in load_json(args.terms).items()}
    by_dim = {}
    for g in groups.values():
        by_dim.setdefault(g['dimension'], []).append(g['table'])
    vectors, attributes = {}, {}
    for g in groups.values():
        d = os.path.join(args.index_root, g['table'])
        f = manifest(d)
        rows, dims = int(f['rows'][0][0]), int(f['dimensions'][0][0])
        vectors[g['table']] = np.memmap(os.path.join(d, 'vectors.f32'), dtype='<f4', mode='r', shape=(rows, dims))
        multi = {m[0] for m in f.get('multivalued', [])}
        attributes[g['table']] = {a[0]: a[0] in multi for a in f.get('attribute', [])}
    pool = popularity = summary = None
    if args.filter_spec:
        if args.mode != 'mixed':
            raise SystemExit('--filter-spec draws filters for mixed requests: it does not go with --mode replay')
        spec = load_filter_spec(args.filter_spec)
        reference = spec['reference_table'] or catalog[0]['table']
        if reference not in attributes:
            raise SystemExit(f'filter spec: reference_table {reference} is not a table of {args.groups}')
        for template in spec['templates']:
            for leaf in template['leaves']:
                wrong = [table for table, kinds in attributes.items() if leaf['column'] not in kinds or
                         (leaf['op'] != 'null' and kinds[leaf['column']] != (leaf['op'] == 'overlap'))]
                if wrong:
                    raise SystemExit(f'filter spec template {template["name"]}: {", ".join(wrong[:8])} '
                                     f'{"have" if len(wrong) > 1 else "has"} no column {leaf["column"]} that takes '
                                     f'a {leaf["op"]} leaf')
        padded = sorted({leaf['column'] for template in spec['templates'] for leaf in template['leaves']
                         if leaf.get('pad')})
        directories = [os.path.join(args.index_root, table) for table in attributes]
        pool = filter_pool(spec, Columns(os.path.join(args.index_root, reference)), args.seed,
                           {name: held_values(directories, name) for name in padded})
        if spec['popularity'] == 'zipf':
            popularity = np.cumsum(np.arange(1, len(pool) + 1, dtype=np.float64) ** -spec['zipf_s']).tolist()
        frng = random.Random(f'request_frames filters {args.seed}')
        quantiles = np.quantile([entry['share'] for entry in pool], [0, 0.1, 0.5, 0.9, 1]).tolist()
        counts = collections.Counter(entry['template'] for entry in pool)
        summary = {'filters': len(pool), 'distinct_programs': len({entry['program'] for entry in pool}),
                   'popularity': spec['popularity'], 'reference_table': reference,
                   'pass_rate': dict(zip(('min', 'p10', 'p50', 'p90', 'max'), quantiles)),
                   'templates': {template['name']: counts[template['name']] for template in spec['templates']}}
    subs, variants_by_filter, empty = [], {}, 0
    for q in capture['queries']:
        if q['repeat'] != 0:
            continue
        clauses = q['filter']['clauses']
        empty += not clauses
        codes = [sorted({t['code'] if t.get('code') is not None else terms.get(t.get('fid')) for t in c['terms']}
                        - {None}) for c in clauses]
        fkey = json.dumps(q['filter'], sort_keys=True)
        if fkey not in variants_by_filter:
            variants_by_filter[fkey] = [
                [sorted(rng.sample(v, max(1, int(len(v) * rng.uniform(0.25, 1.0)))))
                 if c['op'] == 'overlap' and len(v) > 1 else v for c, v in zip(clauses, codes)]
                for _ in range(args.distinct_filters)]
        subs.append({'tag': q['tag'], 'k': q['k'], 'group': groups[q['group']], 'vector': q['vector'],
                     'columns': [c['column'] for c in clauses], 'ops': [c['op'] for c in clauses],
                     'codes': codes, 'variants': variants_by_filter[fkey], 'fkey': fkey,
                     'vkey': (q['group'], json.dumps(q['vector']))})
    if empty and pool is None:
        print(f'warning: {empty} captured sub-requests have a filter without clauses; unless drawn unfiltered they '
              'are sent with an empty program, which the worker refuses', file=sys.stderr)
    with open(args.out, 'wb') as out, open(args.out + '.check.jsonl', 'w') as check:
        out.write(struct.pack('<II', args.requests, len(subs)))
        for r in range(args.requests):
            specs, drawn_vectors, drawn_filters = [], {}, {}
            for i, sub in enumerate(subs):
                table, vector, codes, filtered, pooled = sub['group']['table'], sub['vector'], sub['codes'], True, None
                if args.mode == 'mixed':
                    vkey = i if args.independent else sub['vkey']
                    fkey = i if args.independent else sub['fkey']
                    if vkey not in drawn_vectors:
                        t = rng.choice(by_dim[sub['group']['dimension']])
                        rows = vectors[t]
                        drawn_vectors[vkey] = (t, (rows[nrng.integers(len(rows))] +
                                                   nrng.normal(0, args.noise, rows.shape[1])).tolist())
                    if fkey not in drawn_filters:
                        # The captured variant is drawn even when a pooled filter replaces it, so that a spec
                        # changes no other draw of the request.
                        drawn_filters[fkey] = [rng.choice(sub['variants']), rng.random() >= args.unfiltered_share, None]
                        if pool is not None:
                            at = (frng.randrange(len(pool)) if popularity is None else
                                  min(bisect.bisect_right(popularity, frng.random() * popularity[-1]), len(pool) - 1))
                            drawn_filters[fkey][2] = pool[at]
                    table, vector = drawn_vectors[vkey]
                    codes, filtered, pooled = drawn_filters[fkey]
                if pooled is not None:
                    prog = pooled['program'] if filtered else bytes([ALL])
                    specs.append({'table': table, 'k': sub['k'], 'vector': vector, 'filtered': filtered,
                                  'expr': pooled['expr'] if filtered else {'op': 'all'}})
                else:
                    prog = program(sub['columns'], sub['ops'], codes) if filtered else bytes([ALL])
                    specs.append({'table': table, 'k': sub['k'], 'vector': vector, 'filtered': filtered,
                                  'columns': sub['columns'], 'ops': sub['ops'], 'codes': codes})
                f = frame(table, vector, sub['k'], prog)
                out.write(struct.pack('<I', len(f)) + f)
            if r < args.check_requests:
                check.write(json.dumps(specs) + '\n')
    line = {'requests': args.requests, 'subrequests': len(subs), 'frames': args.out,
            'tags': [(s['tag'], s['k']) for s in subs]}
    if summary:
        line['filter_pool'] = summary
    print(json.dumps(line))


def recv(sock, n):
    data = b''
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise EOFError('connection closed')
        data += chunk
    return data


def call(sock, frame_bytes):
    sock.sendall(frame_bytes)
    header = recv(sock, 32)
    size, status = struct.unpack('<II', header[24:32])
    return status, recv(sock, size)


def connect(args):
    """A connection to the worker, authenticated with the token file (operation 13)."""
    with open(args.token_file) as file:
        token = file.read().strip().encode()
    sock = socket.create_connection((args.host, args.port))
    status, _ = call(sock, b'PVAN' + struct.pack('<HHQQII', 1, 13, 1, 0, len(token), 0) + token)
    if status:
        sock.close()
        raise SystemExit('authentication failed')
    return sock


def check(args):
    """Exact answers from the index files (keys, vectors, attributes) for the first requests."""
    files = {}

    def table_files(table):
        if table not in files:
            d = os.path.join(args.index_root, table)
            columns = Columns(d)
            dims = int(columns.fields['dimensions'][0][0])
            files[table] = (np.fromfile(os.path.join(d, 'keys.i64'), dtype='<i8'),
                            np.memmap(os.path.join(d, 'vectors.f32'), dtype='<f4', mode='r',
                                      shape=(columns.rows, dims)), columns)
        return files[table]

    recalls, by_distance, routes = [], [], {}
    failing = 0  # returned keys the filter does not pass
    with connect(args) as sock, open(args.frames, 'rb') as stream, open(args.frames + '.check.jsonl') as lines:
        stream.read(8)
        for line in lines:
            for spec in json.loads(line):
                size = struct.unpack('<I', stream.read(4))[0]
                frame_bytes = stream.read(size)
                if args.only_table and spec['table'] not in args.only_table.split(','):
                    continue
                status, body = call(sock, frame_bytes)
                if status:
                    raise SystemExit(f'status {status} for {spec["table"]}')
                count, route, _ = struct.unpack('<IIQ', body[:16])
                routes[route] = routes.get(route, 0) + 1
                hits = np.array([struct.unpack_from('<q', body, 16 + 12 * i)[0] for i in range(count)], dtype=np.int64)
                got = set(hits.tolist())
                keys, vectors, columns = table_files(spec['table'])
                rows = np.flatnonzero(eligible(columns, spec))
                q = np.asarray(spec['vector'], dtype=np.float32)
                dist = ((vectors[rows] - q) ** 2).sum(1)
                order = np.argsort(dist, kind='stable')[:spec['k']]
                truth = set(keys[rows[order]].tolist())
                failing += len(got - set(keys[rows].tolist()))
                recalls.append(len(got & truth) / len(truth) if truth else float(not got))
                # By distance: a hit counts when its exact distance is within the true k-th distance (ties
                # between near-duplicate rows make a hit as good as the true one it displaced).
                if count and truth:
                    hit_rows = np.minimum(np.searchsorted(keys, hits), len(keys) - 1)
                    hit_dist = np.where(keys[hit_rows] == hits, ((vectors[hit_rows] - q) ** 2).sum(1), np.inf)
                    by_distance.append(float(np.mean(hit_dist <= dist[order[-1]] * (1 + 1e-4))))
                else:
                    by_distance.append(recalls[-1])
                if args.verbose:
                    print(json.dumps({'table': spec['table'], 'k': spec['k'], 'filtered': spec['filtered'],
                                      'eligible': int(len(rows)), 'returned': count, 'route': route,
                                      'recall': round(recalls[-1], 4),
                                      'recall_by_distance': round(by_distance[-1], 4)}))
    print(json.dumps({'checked': len(recalls), 'routes': routes, 'recall_mean': round(float(np.mean(recalls)), 4),
                      'recall_min': round(float(np.min(recalls)), 4),
                      'below_0.99': int(sum(r < 0.99 for r in recalls)),
                      'recall_by_distance_mean': round(float(np.mean(by_distance)), 4),
                      'not_passing': failing}))


def read_frames(path, requests=None):
    """A frames file's requests, each the list of its sub-requests' frames (the first `requests`, or all)."""
    with open(path, 'rb') as stream:
        count, subrequests = struct.unpack('<II', stream.read(8))
        for _ in range(count if requests is None else min(count, requests)):
            frames = []
            for _ in range(subrequests):
                size = stream.read(4)
                frames.append(stream.read(struct.unpack('<I', size)[0]) if len(size) == 4 else None)
                if frames[-1] is None or len(frames[-1]) != struct.unpack('<I', size)[0]:
                    raise SystemExit(f'{path} ends before its last request does')
            yield frames


def parse_frame(frame_bytes):
    """A search-by-predicate frame's table, dimensions, k and program."""
    if frame_bytes[:4] != b'PVAN' or struct.unpack_from('<H', frame_bytes, 6)[0] != 22:
        raise SystemExit('not a frame of operation 22')
    at = 36 + struct.unpack_from('<I', frame_bytes, 32)[0]
    dimensions, k, _, size = struct.unpack_from('<IIII', frame_bytes, at)
    return frame_bytes[36:at].decode(), dimensions, k, frame_bytes[at + 16:at + 16 + size]


def eligible_counts(job):
    """A table's rows and how many of them each program selects; one job per table for describe's processes."""
    directory, programs = job
    columns = Columns(directory)
    return columns.rows, [columns.count(encoded) for encoded in programs]


def describe(args):
    """One JSON object saying what a frames file asks for.

    Counts are over sub-requests unless named otherwise: "leaves", "operators" (AND, OR and NOT
    instructions) and "values_per_leaf" count a program once for every sub-request sending it, and the
    median is the lower one. "distinct_programs" includes the unfiltered program, and a request's distinct
    filters do not. "pass_rate" bins each sub-request by eligible rows / its table's rows, one bin for none
    and then (edge, next edge], unfiltered sub-requests among them at 1, with the bin's share of the
    sub-requests and of all their eligible rows. Eligible rows are counted once per distinct (table,
    program), a table per process.
    """
    counter = collections.Counter
    tables, ks, programs, pairs, duplicates, filters = (counter() for _ in range(6))
    dimensions, unfiltered, requests, subrequests = [], bytes([ALL]), 0, 0
    for frames in read_frames(args.frames, args.requests):
        requests += 1
        subrequests = len(frames)
        sent = set()
        for at, frame_bytes in enumerate(frames):
            table, dimension, k, encoded = parse_frame(frame_bytes)
            tables[table] += 1
            ks[k] += 1
            programs[encoded] += 1
            pairs[table, encoded] += 1
            sent.add(encoded)
            if at == len(dimensions):
                dimensions.append(counter())
            dimensions[at][dimension] += 1
        duplicates[len(frames) - len(set(frames))] += 1
        filters[len(sent - {unfiltered})] += 1
    total = sum(programs.values())

    leaves, operators, lengths = counter(), counter(), {}
    for encoded, sends in programs.items():
        try:
            instructions = steps(encoded)
        except ValueError as error:
            raise SystemExit(f'{args.frames} has a program the worker refuses, sent {sends} times: {error}')
        for op, column, negate, operand in instructions:
            if op in LEAVES:
                leaves[column, op, negate] += sends
                if op in ('values', 'overlap'):
                    lengths.setdefault(column, counter())[len(operand)] += sends
            elif op in ('and', 'or', 'not'):
                operators[op] += sends

    def spread(sizes):
        ordered = sorted(sizes)
        reached = np.cumsum([sizes[size] for size in ordered])
        median = ordered[int(np.searchsorted(reached, reached[-1] / 2))]
        return {'min': ordered[0], 'median': median, 'max': ordered[-1]}

    by_table = {}
    for table, encoded in pairs:
        by_table.setdefault(table, []).append(encoded)
    jobs = [(os.path.join(args.index_root, table), by_table[table]) for table in sorted(by_table)]
    if args.processes > 1 and len(jobs) > 1:
        with concurrent.futures.ProcessPoolExecutor(min(args.processes, len(jobs))) as workers:
            results = list(workers.map(eligible_counts, jobs))
    else:
        results = [eligible_counts(job) for job in jobs]
    bins = [[0, 0] for _ in PASS_EDGES]
    for table, (rows, counts) in zip(sorted(by_table), results):
        for encoded, count in zip(by_table[table], counts):
            at = bisect.bisect_left(PASS_EDGES, count / rows) if count else 0
            bins[at][0] += pairs[table, encoded]
            bins[at][1] += pairs[table, encoded] * count
    selected = sum(rows for _, rows in bins)
    labels = ['0'] + [f'({low:g}, {high:g}]' for low, high in zip(PASS_EDGES, PASS_EDGES[1:])]
    print(json.dumps({
        'frames': args.frames, 'requests': requests, 'subrequests_per_request': subrequests,
        'tables': {'count': len(tables), 'subrequests_min': min(tables.values(), default=0),
                   'subrequests_max': max(tables.values(), default=0)},
        'k': {str(k): ks[k] for k in sorted(ks)},
        'dimensions': [{str(d): seen[d] for d in sorted(seen)} for seen in dimensions],
        'distinct_programs': len(programs), 'distinct_table_programs': len(pairs),
        'leaves': [{'column': column, 'op': op, 'negate': negate, 'count': leaves[column, op, negate]}
                   for column, op, negate in sorted(leaves)],
        'operators': {op: operators[op] for op in ('and', 'or', 'not')},
        'values_per_leaf': {column: spread(lengths[column]) for column in sorted(lengths)},
        'duplicate_subrequests_per_request': {str(n): duplicates[n] for n in sorted(duplicates)},
        'distinct_filters_per_request': {str(n): filters[n] for n in sorted(filters)},
        'unfiltered_share': programs[unfiltered] / total if total else 0.0,
        'pass_rate': [{'pass_rate': label, 'subrequests': sends, 'share': sends / total if total else 0.0,
                       'eligible_share': rows / selected if selected else 0.0}
                      for label, (sends, rows) in zip(labels, bins)]}))


def verify(args):
    """The worker's rows for the file's first distinct (table, program) pairs against Columns', bit for bit.

    Sends operation 23 (evaluate predicate) and compares the eligible count and the bitmap. Prints one JSON
    line and exits with status 1 when any pair differs; a pair the worker refuses differs unless this
    evaluator refuses it too (its "expr" is then the reason).
    """
    pairs = {}
    for frames in read_frames(args.frames):
        for frame_bytes in frames:
            table, _, _, encoded = parse_frame(frame_bytes)
            if len(pairs) < args.programs:
                pairs.setdefault((table, encoded))
        if len(pairs) >= args.programs:
            break
    tables, mismatches = {}, []
    with connect(args) as sock:
        for table, encoded in pairs:
            name = table.encode()
            payload = struct.pack('<I', len(name)) + name + encoded
            status, body = call(sock, b'PVAN' + struct.pack('<HHQQII', 1, 23, 1, 0, len(payload), 0) + payload)
            if table not in tables:
                tables[table] = Columns(os.path.join(args.index_root, table))
            try:
                expr = decode(encoded)
                mask = tables[table].mask(encoded)
                # Bit i of word i / 32, little-endian words: bit i % 8 of byte i / 8.
                words = np.packbits(mask, bitorder='little').tobytes().ljust((len(mask) + 31) // 32 * 4, b'\0')
                expected = struct.pack('<Q', int(mask.sum())) + words
            except ValueError as error:
                expr, mask, expected = str(error), None, None
            if (status != 0) == (expected is None) and (status != 0 or body == expected):
                continue
            entry = {'table': table, 'expr': expr, 'status': status,
                     'expected': None if mask is None else int(mask.sum()),
                     'worker': struct.unpack_from('<Q', body)[0] if status == 0 and len(body) >= 8 else None}
            if status == 0 and mask is not None and len(body) == len(expected):
                theirs = np.unpackbits(np.frombuffer(body, dtype=np.uint8, offset=8), bitorder='little')[:len(mask)]
                entry['rows_differing'] = int(np.count_nonzero(theirs != mask))
            mismatches.append(entry)
    print(json.dumps({'compared': len(pairs), 'mismatched': len(mismatches), 'mismatches': mismatches[:10]}))
    if mismatches:
        raise SystemExit(1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    e = commands.add_parser('emit')
    e.add_argument('--capture', required=True)
    e.add_argument('--groups', required=True)
    e.add_argument('--terms', required=True)
    e.add_argument('--index-root', required=True)
    e.add_argument('--out', required=True)
    e.add_argument('--requests', type=int, default=20000)
    e.add_argument('--mode', choices=['mixed', 'replay'], default='mixed')
    e.add_argument('--distinct-filters', type=int, default=64)
    e.add_argument('--unfiltered-share', type=float, default=0.1)
    e.add_argument('--noise', type=float, default=0.01)
    e.add_argument('--check-requests', type=int, default=30)
    e.add_argument('--independent', action='store_true',
                   help='draw every sub-request on its own (no duplicates within a request)')
    e.add_argument('--filter-spec',
                   help='JSON: draw the filters from templates instead of the capture\'s (request_frames.py --help)')
    e.add_argument('--seed', type=int, default=7)
    c = commands.add_parser('check')
    c.add_argument('--frames', required=True)
    c.add_argument('--index-root', required=True)
    c.add_argument('--host', default='127.0.0.1')
    c.add_argument('--port', type=int, default=7700)
    c.add_argument('--token-file', required=True)
    c.add_argument('--verbose', action='store_true', help='one line per checked sub-request')
    c.add_argument('--only-table', help='comma-separated tables: check only their sub-requests (a worker with a subset loaded)')
    d = commands.add_parser('describe')
    d.add_argument('--frames', required=True)
    d.add_argument('--index-root', required=True)
    d.add_argument('--requests', type=int, help='read only the first N requests (default: all)')
    d.add_argument('--processes', type=int, default=min(8, os.cpu_count() or 1),
                   help='tables whose eligible rows are counted at once; each holds its table\'s filtered columns')
    v = commands.add_parser('verify')
    v.add_argument('--frames', required=True)
    v.add_argument('--index-root', required=True)
    v.add_argument('--host', default='127.0.0.1')
    v.add_argument('--port', type=int, default=7700)
    v.add_argument('--token-file', required=True)
    v.add_argument('--programs', type=int, default=200, help='distinct (table, program) pairs to compare')
    args = parser.parse_args(argv)
    {'emit': emit, 'check': check, 'describe': describe, 'verify': verify}[args.command](args)


if __name__ == '__main__':
    main()
