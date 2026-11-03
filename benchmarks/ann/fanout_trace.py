"""Join a coordinator's and the workers' per-request traces and break each request's time down.

Coordinators: SET ann.worker_trace_sample = N (lines in /tmp/ann-fanout-PID.trace).
Workers: --trace FILE --trace-sample N (the same N). Both ends run on one host, so their
realtime clocks agree; request ids carry the backend's pid and match exactly.

    python3 benchmarks/ann/fanout_trace.py /tmp/ann-fanout-*.trace /tmp/ann-node*.trace
"""

import sys
from collections import defaultdict

STAGES = [
    ('send', 'coordinator: fan-out start to this request sent'),
    ('wire_in', 'sent to frame read by the worker'),
    ('decode', 'frame read to queued (decode, program, filter lookup)'),
    ('queue', 'queued to taken by a lane'),
    ('search', 'taken to search finished (batch and GPU)'),
    ('reply', 'finished to reply posted'),
    ('wire_out', 'reply posted to seen by the coordinator'),
    ('total', 'sent to seen'),
]


def percentile(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * p))] if values else 0


def main(paths):
    coordinator, worker = {}, {}
    for path in paths:
        with open(path) as lines:
            for line in lines:
                f = line.split()
                if f and f[0] == 'C' and len(f) == 8:
                    coordinator[int(f[2])] = (f[3], int(f[4]), *map(int, f[5:8]))
                elif f and f[0] == 'W' and len(f) == 8:
                    worker[int(f[1])] = tuple(map(int, f[2:8]))
    by = defaultdict(lambda: defaultdict(list))
    for request, (node, rdma, fanout, sent, seen) in coordinator.items():
        w = worker.get(request)
        if w is None:
            continue
        _, arrived, queued, started, done, replied = w
        key = 'rdma' if rdma else 'tcp'
        for name, value in [('send', sent - fanout), ('wire_in', arrived - sent), ('decode', queued - arrived),
                            ('queue', started - queued), ('search', done - started), ('reply', replied - done),
                            ('wire_out', seen - replied), ('total', seen - sent)]:
            by[key][name].append(value / 1000.0)
    print(f'{len(coordinator)} coordinator and {len(worker)} worker lines')
    for key, stages in sorted(by.items()):
        print(f'== {key}: {len(stages["total"])} requests, microseconds')
        print(f'{"stage":10} {"mean":>9} {"p50":>9} {"p99":>9}  ')
        for name, label in STAGES:
            values = stages[name]
            print(f'{name:10} {sum(values) / len(values):9.1f} {percentile(values, 0.5):9.1f} '
                  f'{percentile(values, 0.99):9.1f}  {label}')


if __name__ == '__main__':
    main(sys.argv[1:])
