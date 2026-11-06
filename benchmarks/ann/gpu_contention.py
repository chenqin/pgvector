#!/usr/bin/env python3
"""Bounded GPU contention for replica-routing benchmarks on a dedicated node.

Run alongside the ANN worker, using the same CUDA_VISIBLE_DEVICES mapping.
The ready file appears after allocations and a warm-up GEMM complete.
"""

import argparse
import json
import math
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--size', type=int, default=4096)
    parser.add_argument('--ready-file', type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 3600 or not 128 <= args.size <= 16384:
        parser.error('seconds must be finite, positive and at most 3600; size must be 128..16384')
    import cupy as cp

    with cp.cuda.Device(args.device):
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            a = cp.ones((args.size, args.size), dtype=cp.float16)
            b = cp.full_like(a, 0.125)
            out = cp.empty_like(a)
            cp.matmul(a, b, out=out)
            stream.synchronize()
            started = time.monotonic()
            if args.ready_file:
                args.ready_file.write_text('ready\n')
            iterations = 0
            while time.monotonic() - started < args.seconds:
                # Bound the queued work, as well as the wall-clock duration.
                for _ in range(16):
                    cp.matmul(a, b, out=out)
                stream.synchronize()
                iterations += 16
            print(json.dumps(dict(device=args.device, size=args.size, iterations=iterations,
                                  seconds=time.monotonic() - started)), flush=True)


if __name__ == '__main__':
    main()
