#!/usr/bin/env python3
"""twohop_udf_cases.py FRAMES TABLE OUT [--count N] [--distinct]: cases for cagra_twohop_udf_probe.

Takes the sub-requests of a request_direct frames file (benchmarks/ann/request_frames.py emit) that search
TABLE, in file order, and writes their predicate programs and query vectors: u32 count, u32 dims,
then per case u32 k, u32 program length, the program, dims FP32. --distinct keeps the first
sub-request of each program only. --first-leaf keeps only the first operand of a top-level AND, as the
worker's canonical form does when the other operands pass every row (the benchmark's item_bid clause).
"""

import argparse
import os
import struct
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('frames')
    parser.add_argument('table')
    parser.add_argument('out')
    parser.add_argument('--count', type=int, default=256)
    parser.add_argument('--distinct', action='store_true')
    parser.add_argument('--first-leaf', action='store_true')
    args = parser.parse_args()
    if args.first_leaf:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import request_frames
    cases, seen, dims = [], set(), None
    with open(args.frames, 'rb') as stream:
        nreq, nsub = struct.unpack('<II', stream.read(8))
        for _ in range(nreq * nsub):
            size = struct.unpack('<I', stream.read(4))[0]
            payload = stream.read(size)[32:]
            namelen = struct.unpack_from('<I', payload, 0)[0]
            name = payload[4:4 + namelen].decode()
            if name != args.table:
                continue
            d, k, _, proglen = struct.unpack_from('<IIII', payload, 4 + namelen)
            prog = payload[4 + namelen + 16:4 + namelen + 16 + proglen]
            vector = payload[4 + namelen + 16 + proglen:4 + namelen + 16 + proglen + 4 * d]
            if args.first_leaf:
                expr = request_frames.decode(prog)
                if expr.get('op') == 'and':
                    prog = request_frames.encode(expr['args'][0])
            if args.distinct and prog in seen:
                continue
            seen.add(prog)
            dims = d
            cases.append((k, prog, vector))
            if len(cases) == args.count:
                break
    with open(args.out, 'wb') as out:
        out.write(struct.pack('<II', len(cases), dims or 0))
        for k, prog, vector in cases:
            out.write(struct.pack('<II', k, len(prog)) + prog + vector)
    print(f'{len(cases)} cases of {args.table}, {dims} dims, {len(seen)} distinct programs')


if __name__ == '__main__':
    main()
