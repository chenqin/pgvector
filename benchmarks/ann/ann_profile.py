#!/usr/bin/env python3
"""Helpers for ann_profile.sh: collectors and the report writer.

Standard library only, so it runs on a plain python3.

  ann_profile.py counters OUT.json --host H --port P --token-file F [--index NAME ...]
      one snapshot of op 8 (statistics) and op 31 (routing statistics) per index
  ann_profile.py threads PID OUT.json       one snapshot of /proc/PID/task/*/stat
  ann_profile.py stall PID OUT.txt           poll the main thread's state until SIGTERM; one
                                             line per stop (gdb attach) in ms
  ann_profile.py gdb-roles SNAPSHOT.txt      print "TID role" from one gdb snapshot
  ann_profile.py report DIR                  write DIR/report.txt and DIR/report_full.txt
"""
import csv
import glob
import json
import os
import re
import signal
import socket
import sqlite3
import struct
import sys
import time
from collections import Counter, defaultdict


# ---------------------------------------------------------------- wire protocol

def call(sock, op, payload=b""):
    sock.sendall(b"PVAN" + struct.pack("<HHQQII", 1, op, 1, 0, len(payload), 0) + payload)
    head = b""
    while len(head) < 32:
        chunk = sock.recv(32 - len(head))
        if not chunk:
            raise ConnectionError("server closed the connection")
        head += chunk
    size, status = struct.unpack("<II", head[24:32])
    body = b""
    while len(body) < size:
        chunk = sock.recv(size - len(body))
        if not chunk:
            raise ConnectionError("short reply")
        body += chunk
    return status, body


def cmd_counters(args):
    out = args[0]
    opts = {"--host": "127.0.0.1", "--port": "7700", "--token-file": None}
    names = []
    i = 1
    while i < len(args):
        if args[i] == "--index":
            names.append(args[i + 1])
            i += 2
        elif args[i] in opts:
            opts[args[i]] = args[i + 1]
            i += 2
        else:
            raise SystemExit("counters: unknown argument " + args[i])
    sock = socket.create_connection((opts["--host"], int(opts["--port"])), timeout=20)
    if opts["--token-file"]:
        token = open(opts["--token-file"]).read().strip().encode()
        status, _ = call(sock, 13, token)
        if status:
            raise SystemExit("authentication failed, status %d" % status)
    snap = {"time": time.time(), "indexes": {}}
    status, body = call(sock, 8)
    if status:
        raise SystemExit("op 8 failed, status %d" % status)
    snap["stats"] = json.loads(body.decode().strip("\x00"))
    for name in names:
        status, body = call(sock, 31, name.encode())
        if status == 0 and body:
            try:
                snap["indexes"][name] = json.loads(body.decode().strip("\x00"))
            except ValueError:
                pass
    sock.close()
    with open(out, "w") as f:
        json.dump(snap, f)


# ---------------------------------------------------------------- /proc

def read_tasks(pid):
    tasks = {}
    base = "/proc/%s/task" % pid
    for tid in os.listdir(base):
        try:
            stat = open("%s/%s/stat" % (base, tid)).read()
        except OSError:
            continue
        comm = stat[stat.index("(") + 1:stat.rindex(")")]
        fields = stat[stat.rindex(")") + 2:].split()
        tasks[tid] = {"comm": comm, "state": fields[0], "utime": int(fields[11]), "stime": int(fields[12])}
    return tasks


def cmd_threads(args):
    pid, out = args
    snap = {"time": time.time(), "hz": os.sysconf("SC_CLK_TCK"), "tasks": read_tasks(pid)}
    with open(out, "w") as f:
        json.dump(snap, f)


def cmd_stall(args):
    pid, out = args
    stalls = []
    stop = [False]

    def finish(*_):
        stop[0] = True

    signal.signal(signal.SIGTERM, finish)
    signal.signal(signal.SIGINT, finish)
    path = "/proc/%s/stat" % pid
    started = None
    while not stop[0]:
        try:
            state = open(path).read().rsplit(")", 1)[1].split()[0]
        except OSError:
            break
        now = time.monotonic()
        if state in ("t", "T") and started is None:
            started = now
        elif state not in ("t", "T") and started is not None:
            stalls.append(now - started)
            started = None
        time.sleep(0.001)
    with open(out, "w") as f:
        for s in stalls:
            f.write("%.1f\n" % (s * 1000))


# ---------------------------------------------------------------- gdb snapshots

THREAD_RE = re.compile(r'^Thread (\d+) \(Thread 0x[0-9a-f]+ \(LWP (\d+)\) "([^"]*)"\):')
FRAME_RE = re.compile(r'^#(\d+)\s+(?:0x[0-9a-f]+ in )?(.*)$')
VALUE_RE = re.compile(r'^\$\d+ = (-?\d+)')

WAIT_FUNCS = {"pthread_mutex_lock", "__pthread_mutex_lock", "pthread_cond_wait", "pthread_cond_timedwait",
              "sem_wait", "sem_timedwait", "pthread_rwlock_rdlock", "pthread_rwlock_wrlock",
              "std::condition_variable::wait", "std::mutex::lock", "__gthread_mutex_lock",
              "epoll_wait", "poll", "ppoll", "select", "nanosleep", "clock_nanosleep", "usleep", "sleep",
              "read", "recv", "recvfrom", "recvmsg", "write", "send", "sendto", "sendmsg", "writev", "accept", "accept4",
              "ioctl", "sched_yield", "pthread_join", "futex"}
SYNC_FUNCS = {"cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize", "cuStreamSynchronize",
              "cuCtxSynchronize", "cuEventSynchronize", "LaneSync", "cudaMemcpy", "cudaMemcpy2D",
              "cudaStreamQuery", "cudaEventQuery", "cudaMemcpyAsync", "cuMemcpyDtoH_v2", "cuMemcpyHtoD_v2"}
SYSCALLS = {0: "read", 1: "write", 7: "poll", 16: "ioctl", 19: "readv", 20: "writev", 24: "sched_yield",
            35: "nanosleep", 44: "sendto", 45: "recvfrom", 46: "sendmsg", 47: "recvmsg", 202: "futex",
            230: "clock_nanosleep", 232: "epoll_wait", 271: "ppoll", 281: "epoll_pwait", 288: "accept4",
            43: "accept", 23: "select", 270: "pselect6", 61: "wait4", 273: "set_robust_list", 128: "rt_sigtimedwait",
            34: "pause", 219: "restart_syscall", 213: "epoll_create", 233: "epoll_ctl", 318: "getrandom", 9: "mmap", 11: "munmap", 12: "brk",
            28: "madvise", 10: "mprotect", 4: "stat", 5: "fstat", 2: "open", 257: "openat", 3: "close",
            17: "pread64", 18: "pwrite64", 72: "fcntl", 231: "exit_group", 60: "exit"}
WAIT_SYSCALLS = {"read", "poll", "nanosleep", "recvfrom", "recvmsg", "futex", "clock_nanosleep", "epoll_wait", "ppoll",
                 "epoll_pwait", "accept4", "accept", "select", "pselect6", "pause", "restart_syscall", "wait4", "rt_sigtimedwait"}
LIB_HINTS = [("libcufile", "cufile"), ("libkvikio", "kvikio"), ("libcuda", "cuda"), ("libnvidia", "cuda"),
             ("libcudart", "cuda"), ("libcuvs", "cuvs"), ("libraft", "cuvs"), ("librmm", "cuvs"), ("libgomp", "cuvs"),
             ("libcublas", "cuvs"), ("libnccl", "cuda")]


def short_name(text):
    """Function name from a gdb frame text: 'Lane (argument=...) at server.c:1137' -> Lane."""
    cut = text.find(" (")
    name = text[:cut] if cut >= 0 else text
    cut = name.find("(")
    if cut > 0:
        name = name[:cut]
    while True:
        new = re.sub(r"<[^<>]*>", "", name)
        if new == name:
            break
        name = new
    return name.strip()


def parse_frame(text):
    """Return (function or None, lib basename or None, source file:line or None)."""
    lib = None
    src = None
    m = re.search(r" from (\S+)$", text)
    if m:
        lib = os.path.basename(m.group(1))
        text = text[:m.start()]
    m = re.search(r" at (\S+):(\d+)$", text)
    if m:
        src = "%s:%s" % (os.path.basename(m.group(1)), m.group(2))
        text = text[:m.start()]
    name = short_name(text)
    if name in ("??", ""):
        return None, lib, src
    return name, lib, src


def parse_snapshot(path):
    """One gdb snapshot -> list of threads: {tid, name, frames:[(fn, lib, src)], syscall}."""
    threads = {}
    order = []
    current = None
    section_values = False
    for line in open(path, errors="replace"):
        line = line.rstrip("\n")
        m = THREAD_RE.match(line)
        if m:
            tid = m.group(2)
            if tid not in threads:
                threads[tid] = {"tid": tid, "name": m.group(3), "frames": [], "syscall": None}
                order.append(tid)
            current = threads[tid]
            continue
        if current is None:
            continue
        m = FRAME_RE.match(line)
        if m:
            current["frames"].append(parse_frame(m.group(2)))
            continue
        m = VALUE_RE.match(line)
        if m:
            current["syscall"] = int(m.group(1))
            section_values = True
    return [threads[t] for t in order]


def thread_role(thread):
    fns = [f[0] for f in thread["frames"] if f[0]]
    libs = [f[1] for f in thread["frames"] if f[1]]
    if "Lane" in fns:
        return "lane"
    if "PoolReader" in fns or "Reader" in fns:
        return "reader"
    if "Dispatch" in fns or "Watch" in fns:
        return "batcher"
    if "Progress" in fns:
        return "rdma"
    if "main" in fns or "AnnServe" in fns:
        return "main"
    if "LoadShard" in fns:
        return "loader"
    name = thread["name"]
    if name.startswith("cuda"):
        return "cuda"
    if name.startswith("kvikio"):
        return "kvikio"
    for lib in libs:
        for prefix, role in LIB_HINTS:
            if lib.startswith(prefix):
                return role
    return "other"


def classify(thread):
    """-> (kind, state, caller) where kind is 'cpu' (user), 'kernel' (in a non-blocking syscall),
    or 'blocked'; state names the wait; caller the first named frame above the wait."""
    frames = thread["frames"]
    fns = [f[0] for f in frames]
    named = [f for f in frames if f[0]]
    first_named = named[0][0] if named else None

    def caller_after(index):
        for f in frames[index + 1:]:
            if f[0] and f[0] not in WAIT_FUNCS and not f[0].startswith("__"):
                return f[0] + ((":" + f[2].split(":")[1]) if f[2] else "")
        return "?"

    wait_index = None
    for i, f in enumerate(frames):
        if f[0] in WAIT_FUNCS:
            wait_index = i
            break
    sysno = thread["syscall"]
    if sysno is not None and sysno >= 0:
        sysname = SYSCALLS.get(sysno, "syscall%d" % sysno)
        if sysname == "futex":
            waker = next((f[0] for f in frames[:3] if f[0] in ("pthread_mutex_unlock", "pthread_cond_signal",
                                                               "pthread_cond_broadcast", "sem_post")), None)
            if waker and (wait_index is None or wait_index > frames.index(next(f for f in frames if f[0] == waker))):
                # the futex wake inside an unlock/signal: kernel time, not a wait
                return "kernel", "futex_wake", waker + "<-" + caller_after(frames.index(next(f for f in frames if f[0] == waker)))
            if wait_index is not None:
                w = frames[wait_index][0]
                if "mutex" in w:
                    state = "mutex"
                elif "cond" in w:
                    state = "cond_wait"
                elif "sem" in w:
                    state = "sem_wait"
                elif "rwlock" in w:
                    state = "rwlock"
                else:
                    state = "futex"
                return "blocked", state, caller_after(wait_index)
            return "blocked", "futex", first_named or "?"
        if sysname in WAIT_SYSCALLS:
            idx = wait_index if wait_index is not None else -1
            if sysname == "restart_syscall":	# poll/nanosleep restarted after the stop signal
                sysname = frames[wait_index][0] if wait_index is not None else "poll/sleep"
            return "blocked", sysname, caller_after(idx)
        idx = wait_index if wait_index is not None else -1
        return "kernel", sysname, caller_after(idx)
    if sysno is None and wait_index is not None:
        # no register information: fall back to the frame names
        w = frames[wait_index][0]
        state = {"pthread_mutex_lock": "mutex", "pthread_cond_wait": "cond_wait", "pthread_cond_timedwait": "cond_wait",
                 "std::condition_variable::wait": "cond_wait", "sem_wait": "sem_wait"}.get(w, w)
        if w in ("write", "send", "sendto", "sendmsg", "writev", "ioctl", "sched_yield"):
            return "kernel", w, caller_after(wait_index)
        return "blocked", state, caller_after(wait_index)
    return "cpu", "on-CPU", first_named or "?"


def in_sync(thread, role=None):
    if any(f[0] in SYNC_FUNCS for f in thread["frames"]):
        return True
    # a lane blocked inside libcuda whose stack gdb could not unwind: the blocking stream sync
    # (cudaDeviceScheduleBlockingSync) is the only place a lane sleeps inside the driver
    frames = thread["frames"]
    return role == "lane" and frames and thread["syscall"] not in (None, -1) and not any(f[0] for f in frames) \
        and any(f[1] and f[1].startswith("libcuda") for f in frames)


def exclusive_label(thread):
    frames = thread["frames"]
    if not frames:
        return "?"
    fn, lib, _ = frames[0]
    if fn:
        return fn
    caller = next((f[0] for f in frames[1:] if f[0]), None)
    return "[%s]" % (lib or "?") + ("<-" + caller if caller else "")


def cmd_gdb_roles(args):
    for t in parse_snapshot(args[0]):
        print(t["tid"], thread_role(t))


# ---------------------------------------------------------------- report helpers

def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def read_text(path):
    try:
        return open(path, errors="replace").read()
    except OSError:
        return ""


def fmt_num(x):
    if x is None:
        return "-"
    if abs(x) >= 1e6:
        return "%.2fM" % (x / 1e6)
    if abs(x) >= 1e4:
        return "%.0fk" % (x / 1e3)
    if isinstance(x, float):
        return ("%.0f" if abs(x) >= 100 else "%.1f") % x
    return str(x)


def pct(part, whole):
    return 100.0 * part / whole if whole else 0.0


class Report:
    def __init__(self):
        self.short = []
        self.full = []

    def line(self, text="", full_only=False):
        if not full_only:
            self.short.append(text)
        self.full.append(text)


def phase_windows(meta):
    """[(name, counters-before, counters-after)] from the phase list in meta."""
    return meta.get("phases", [])


# ---------------------------------------------------------------- report: GPU coarse

def gpu_coarse(d, rep, meta, verdict):
    rows = []
    for row in csv.reader(read_text(os.path.join(d, "gpu.csv")).splitlines()):
        if len(row) < 6:
            continue
        try:
            rows.append({"mono": float(row[0]), "gpu": int(row[1]), "util": float(row[2]), "mem_act": float(row[3]),
                         "mem": float(row[4]), "power": float(row[5])})
        except ValueError:
            continue
    pmon = defaultdict(list)
    pmon_fb = defaultdict(list)
    for row in csv.reader(read_text(os.path.join(d, "pmon.csv")).splitlines()):
        if len(row) >= 4:
            try:
                pmon[(float(row[0]), int(row[1]))].append(float(row[3]))
                if len(row) >= 5:
                    pmon_fb[(float(row[0]), int(row[1]))].append(float(row[4]))
            except ValueError:
                pass
    phases = phase_windows(meta)
    quiet = next((p for p in phases if p["name"] == "quiet"), None)
    rep.line("1. GPU (nvidia-smi, device-wide, %d GPUs; shared with other processes): mean/max over the quiet phase%s"
             % (len({r["gpu"] for r in rows}), "" if quiet else " (no quiet phase: whole window)"))
    rep.line("   gpu  util%% mean/max  mem-act%%  worker-sm%%  mem-used-MiB (worker)  power-W   phase util means: %s"
             % " ".join("%s %.0f" % (p["name"], mean_in(rows, p)) for p in phases))
    per_gpu = defaultdict(list)
    for r in rows:
        if quiet is None or quiet["t0"] <= r["mono"] <= quiet["t1"]:
            per_gpu[r["gpu"]].append(r)
    utils = {}
    shares = {}
    for g in sorted(per_gpu):
        rs = per_gpu[g]
        u = [r["util"] for r in rs]
        utils[g] = sum(u) / len(u)
        sm = [v for (t, gg), vs in pmon.items() if gg == g and (quiet is None or quiet["t0"] <= t <= quiet["t1"]) for v in vs]
        fb = [v for (t, gg), vs in pmon_fb.items() if gg == g and (quiet is None or quiet["t0"] <= t <= quiet["t1"]) for v in vs]
        if sm:
            shares[g] = sum(sm) / len(sm)
        rep.line("   %3d  %5.0f / %3.0f     %5.0f       %5s       %7.0f (%6s)      %5.0f" %
                 (g, utils[g], max(u), sum(r["mem_act"] for r in rs) / len(rs),
                  ("%.0f" % shares[g]) if sm else "-", sum(r["mem"] for r in rs) / len(rs),
                  ("%.0f" % (sum(fb) / len(fb))) if fb else "-", sum(r["power"] for r in rs) / len(rs)))
    if utils:
        top = sorted(utils, key=lambda g: utils[g], reverse=True)[:2]
        verdict["gpu_busiest"] = sum(utils[g] for g in top) / len(top)
        verdict["gpu_mean"] = sum(utils.values()) / len(utils)
        verdict["gpu_min"] = min(utils.values())
        if shares:
            verdict["gpu_worker_share"] = sum(shares.values()) / len(shares)
    else:
        rep.line("   (no nvidia-smi samples)")


def mean_in(rows, phase):
    u = [r["util"] for r in rows if phase["t0"] <= r["mono"] <= phase["t1"]]
    return sum(u) / len(u) if u else 0.0


# ---------------------------------------------------------------- report: GPU fine

def shorten_kernel(name, width):
    name = re.sub(r"^void ", "", name)
    depth = 0
    out = []
    for ch in name:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif depth == 0:
            out.append(ch)
    name = "".join(out)
    name = name.split("(")[0]
    name = name.replace("cuvs::neighbors::", "").replace("raft::", "").replace("detail::", "")
    return name[:width]


def read_stats_csv(path):
    text = read_text(path)
    rows = list(csv.DictReader([l for l in text.splitlines() if l.strip()]))
    return rows


def gpu_fine(d, rep, meta, verdict):
    nsys_dir = os.path.join(d, "nsys")
    info = load_json(os.path.join(nsys_dir, "info.json")) or {}
    rep.line("2. GPU fine")
    if info.get("ok"):
        trace_phase = next((p for p in phase_windows(meta) if p["name"] == "nsys"), None)
        kern = read_stats_csv(glob.glob(os.path.join(nsys_dir, "*cuda_gpu_kern_sum.csv"))[0]) \
            if glob.glob(os.path.join(nsys_dir, "*cuda_gpu_kern_sum.csv")) else []
        api = read_stats_csv(glob.glob(os.path.join(nsys_dir, "*cuda_api_sum.csv"))[0]) \
            if glob.glob(os.path.join(nsys_dir, "*cuda_api_sum.csv")) else []
        mem = read_stats_csv(glob.glob(os.path.join(nsys_dir, "*cuda_gpu_mem_time_sum.csv"))[0]) \
            if glob.glob(os.path.join(nsys_dir, "*cuda_gpu_mem_time_sum.csv")) else []
        rep.line("   nsys (launch + start/stop, --trace=cuda): collected %.1f s, rep %s MB, %s kernels, %s API calls; %s" %
                 (info.get("seconds", 0), fmt_num(info.get("rep_mb")), fmt_num(info.get("kernels")), fmt_num(info.get("api_calls")),
                  info.get("diagnostics", "no diagnostics read")))
        if trace_phase and trace_phase.get("jobs_per_s") is not None:
            change = pct(trace_phase["jobs_per_s"] - (meta.get("quiet_jobs_per_s") or 0), meta.get("quiet_jobs_per_s"))
            trust = "trustworthy (<1% events dropped, throughput within 25% of quiet)" \
                if info.get("dropped_pct", 0) < 1 and abs(change) < 25 else "QUESTIONABLE (drops or a large throughput change)"
            rep.line("   throughput during the trace: %s jobs/s vs %s jobs/s quiet (%+.0f%%); trace %s"
                     % (fmt_num(trace_phase["jobs_per_s"]), fmt_num(meta.get("quiet_jobs_per_s")), change, trust))
        per_dev = info.get("per_device") or []
        if per_dev:
            rep.line("   kernel time per GPU, ms per s of trace: " + " ".join("%d:%.0f" % (x["device"], x["ms_per_s"]) for x in per_dev))
        total_k = sum(float(r.get("Total Time (ns)", 0)) for r in kern)
        rep.line("   top kernels (% of GPU kernel time, count, avg us)          | top CUDA API calls (% of API time, count, avg us)")
        total_a = sum(float(r.get("Total Time (ns)", 0)) for r in api)
        n = 7
        kl = ["%5.1f%% %7s %7.1f %s" % (pct(float(r["Total Time (ns)"]), total_k), fmt_num(int(r["Instances"])),
                                       float(r["Avg (ns)"]) / 1000, shorten_kernel(r["Name"], 34)) for r in kern[:n]]
        al = ["%5.1f%% %7s %7.1f %s" % (pct(float(r["Total Time (ns)"]), total_a), fmt_num(int(r["Num Calls"])),
                                       float(r["Avg (ns)"]) / 1000, r["Name"][:28]) for r in api[:n]]
        for i in range(max(len(kl), len(al))):
            rep.line("   %-58s | %s" % (kl[i] if i < len(kl) else "", al[i] if i < len(al) else ""))
        for r in kern[n:25]:
            rep.line("   kernel %5.1f%% %7s %7.1f %s" % (pct(float(r["Total Time (ns)"]), total_k), fmt_num(int(r["Instances"])),
                                                        float(r["Avg (ns)"]) / 1000, shorten_kernel(r["Name"], 90)), full_only=True)
        for r in api[n:25]:
            rep.line("   api    %5.1f%% %7s %7.1f %s" % (pct(float(r["Total Time (ns)"]), total_a), fmt_num(int(r["Num Calls"])),
                                                        float(r["Avg (ns)"]) / 1000, r["Name"]), full_only=True)
        if mem:
            rep.line("   memory ops: " + "; ".join("%s %s x %.1f us" % (r["Operation"], fmt_num(int(r["Count"])), float(r["Avg (ns)"]) / 1000)
                                                   for r in mem[:4]))
        sync = sum(float(r["Total Time (ns)"]) for r in api if "Synchronize" in r["Name"])
        launch = sum(float(r["Total Time (ns)"]) for r in api if "Launch" in r["Name"])
        verdict["api_sync_share"] = pct(sync, total_a)
        verdict["api_launch_share"] = pct(launch, total_a)
    else:
        rep.line("   nsys: %s" % info.get("reason", "worker not started under nsys launch (ann_profile.sh nsys-worker ROOT); skipped"))
    kt = meta.get("kernel_timing") or {}
    if kt.get("lines"):
        rep.line("   ANN_KERNEL_TIMING (worker log, last window%s):" % (" before this run" if kt.get("stale") else ""))
        for l in kt["lines"]:
            rep.line("     " + l.strip())
    else:
        rep.line("   ANN_KERNEL_TIMING: no 'matrix gather stages' lines in the worker log (export ANN_KERNEL_TIMING=1 before worker.sh)")


# ---------------------------------------------------------------- report: CPU

def cpu_section(d, rep, meta, verdict):
    snaps = sorted(glob.glob(os.path.join(d, "gdb", "*.txt")))
    parsed = [parse_snapshot(p) for p in snaps]
    parsed = [p for p in parsed if p]
    stalls = [float(x) for x in read_text(os.path.join(d, "stalls.txt")).split()]
    walls = [float(x) for x in read_text(os.path.join(d, "gdb", "walls.txt")).split()]
    gdb_phase = next((p for p in phase_windows(meta) if p["name"] == "gdb"), None)
    rep.line("3. CPU (gdb stack snapshots; worker has no thread names, roles from stacks)")
    if not parsed:
        rep.line("   no usable gdb snapshots (is the worker started with LD_PRELOAD=ptrace_any.so?)")
        return
    # role per TID: majority over snapshots
    role_votes = defaultdict(Counter)
    for threads in parsed:
        for t in threads:
            role_votes[t["tid"]][thread_role(t)] += 1
    roles = {}
    for tid, votes in role_votes.items():
        # a thread whose stack ever reaches Lane/PoolReader is that thread for good (unwinding through libcuda
        # fails now and then, which would otherwise vote it into "cuda")
        fixed = next((r for r in ("lane", "reader", "batcher", "main", "rdma", "loader") if votes.get(r)), None)
        if fixed:
            roles[tid] = fixed
            continue
        best = votes.most_common()
        named = [r for r in best if r[0] != "other"]
        roles[tid] = (named[0] if named else best[0])[0]
    # per-thread CPU from /proc over the quiet phase
    t0 = load_json(os.path.join(d, "threads0.json"))
    t1 = load_json(os.path.join(d, "threads1.json"))
    cpu_role = defaultdict(lambda: [0.0, 0.0])
    cpu_tid = {}
    if t0 and t1:
        hz = t1["hz"]
        span = t1["time"] - t0["time"]
        for tid, b in t1["tasks"].items():
            a = t0["tasks"].get(tid)
            if a is None:
                continue
            u = (b["utime"] - a["utime"]) / hz / span
            s = (b["stime"] - a["stime"]) / hz / span
            cpu_tid[tid] = (u, s)
            r = roles.get(tid, "unknown")
            cpu_role[r][0] += u
            cpu_role[r][1] += s
    n = len(parsed)
    stall_txt = "stall per attach %.0f ms mean / %.0f max" % (sum(stalls) / len(stalls), max(stalls)) if stalls else "stall not measured"
    gdb_tp = ""
    if gdb_phase and gdb_phase.get("jobs_per_s") is not None and meta.get("quiet_jobs_per_s"):
        gdb_tp = "; throughput during sampling %s jobs/s (%+.0f%% vs quiet)" % (
            fmt_num(gdb_phase["jobs_per_s"]), pct(gdb_phase["jobs_per_s"] - meta["quiet_jobs_per_s"], meta["quiet_jobs_per_s"]))
    rep.line("   %d snapshots, gdb wall %.2f s each, %s%s" % (n, (sum(walls) / len(walls)) if walls else 0, stall_txt, gdb_tp))
    cmd = meta.get("cmdline", [])
    expect = {}
    for flag, role in (("--lanes", "lane"), ("--readers", "reader")):
        if flag in cmd:
            expect[role] = int(cmd[len(cmd) - 1 - cmd[::-1].index(flag) + 1])
    counts = Counter(roles.values())
    check = ", ".join("%s %d%s" % (r, counts.get(r, 0), "" if expect.get(r) in (None, counts.get(r, 0)) else " (expected %d!)" % expect[r])
                      for r in ("lane", "reader"))
    total_cpu = sum(u + s for u, s in cpu_tid.values())
    on_cpu_avg = sum(1 for threads in parsed for t in threads if classify(t)[0] != "blocked") / n
    rep.line("   threads by role: %s, others %d; /proc CPU over the quiet phase: %.1f cores (gdb phase, at lower throughput: %.1f threads on CPU per snapshot)"
             % (check, sum(c for r, c in counts.items() if r not in ("lane", "reader")), total_cpu, on_cpu_avg))
    # per-role state table
    role_samples = defaultdict(list)
    for threads in parsed:
        for t in threads:
            role_samples[roles.get(t["tid"], "other")].append(t)
    role_stats = {}
    for role in ("lane", "reader", "cuda", "cufile", "kvikio", "cuvs", "batcher", "main", "rdma", "other", "unknown"):
        samples = role_samples.get(role)
        if not samples:
            continue
        states = Counter()
        sync = 0
        sync_cpu = 0
        kinds = Counter()
        for t in samples:
            kind, state, caller = classify(t)
            kinds[kind] += 1
            if in_sync(t, role):
                sync += 1
                if kind != "blocked":
                    sync_cpu += 1
                states["cudaStreamSynchronize" + (" (spin)" if kind != "blocked" else "")] += 1
            elif kind == "cpu":
                states["on-CPU"] += 1
            else:
                states["%s<-%s" % (state, caller)] += 1
        tot = len(samples)
        u, s = cpu_role.get(role, (0.0, 0.0))
        role_stats[role] = {"n": tot, "sync": sync, "sync_cpu": sync_cpu, "kinds": kinds, "states": states, "cpu": (u, s)}
        top = ", ".join("%s %.0f%%" % (k, pct(v, tot)) for k, v in states.most_common(4))
        rep.line("   %-7s %3d thr %5.1f cores (usr %.1f sys %.1f) | on-CPU %2.0f%% kernel-io %2.0f%% blocked %2.0f%% | %s" %
                 (role, sum(1 for r in roles.values() if r == role), u + s, u, s, pct(kinds["cpu"], tot), pct(kinds["kernel"], tot),
                  pct(kinds["blocked"], tot), top), full_only=role not in ("lane", "reader", "cuda", "cuvs", "batcher"))
        for k, v in states.most_common(12)[4:]:
            rep.line("           %s %.1f%%" % (k, pct(v, tot)), full_only=True)
    # top frames over on-CPU samples (user + kernel-io)
    excl = Counter()
    incl = Counter()
    oncpu = 0
    for threads in parsed:
        for t in threads:
            kind, state, caller = classify(t)
            if kind == "blocked":
                continue
            oncpu += 1
            label = exclusive_label(t) if kind == "cpu" else "%s<-%s" % (state, caller)
            excl[label] += 1
            seen = set()
            for f in t["frames"]:
                if f[0] and f[0] not in seen and f[0] not in ("clone", "start_thread", "clone3"):
                    seen.add(f[0])
                    incl[f[0]] += 1
    rep.line("   on-CPU samples %d (%.1f threads/snapshot): exclusive (leaf, syscall<-caller)       | inclusive" % (oncpu, oncpu / n))
    ex = excl.most_common(25)
    inc = incl.most_common(25)
    for i in range(max(min(len(ex), 7), min(len(inc), 7))):
        a = "%5.1f%% %s" % (pct(ex[i][1], oncpu), ex[i][0][:50]) if i < len(ex) else ""
        b = "%5.1f%% %s" % (pct(inc[i][1], oncpu), inc[i][0][:40]) if i < len(inc) else ""
        rep.line("   %-58s | %s" % (a, b), full_only=i >= 7)
    # lock contention
    locks = Counter()
    lock_role = Counter()
    total_samples = sum(len(t) for t in parsed)
    for threads in parsed:
        for t in threads:
            kind, state, caller = classify(t)
            if state == "mutex":
                locks[(caller, roles.get(t["tid"], "?"))] += 1
                lock_role[roles.get(t["tid"], "?")] += 1
    lr = role_stats.get("lane", {}).get("n", 0)
    rr = role_stats.get("reader", {}).get("n", 0)
    rep.line("   mutex waits: %d of %d samples (lanes %.0f%%, readers %.0f%%); by caller: %s" %
             (sum(locks.values()), total_samples, pct(lock_role["lane"], lr), pct(lock_role["reader"], rr),
              ", ".join("%s[%s] %d" % (c, r, v) for (c, r), v in locks.most_common(5)) or "none"))
    for (c, r), v in locks.most_common(20)[5:]:
        rep.line("      %s[%s] %d" % (c, r, v), full_only=True)
    # verdict inputs
    ls = role_stats.get("lane")
    rs = role_stats.get("reader")
    if ls:
        verdict["lane_sync"] = pct(ls["sync"], ls["n"])
        verdict["lane_idle"] = pct(sum(v for k, v in ls["states"].items() if k.startswith("cond_wait<-Lane")), ls["n"])
        verdict["lane_mutex"] = pct(lock_role["lane"], ls["n"])
        verdict["lane_cpu"] = pct(ls["kinds"]["cpu"] + ls["kinds"]["kernel"], ls["n"])
        verdict["lane_cores"] = sum(ls["cpu"])
    if rs:
        verdict["reader_epoll"] = pct(sum(v for k, v in rs["states"].items() if k.startswith("epoll_wait")), rs["n"])
        verdict["reader_cpu"] = pct(rs["kinds"]["cpu"] + rs["kinds"]["kernel"], rs["n"])
        verdict["reader_mutex"] = pct(lock_role["reader"], rs["n"])
        verdict["reader_cores"] = sum(rs["cpu"])
    verdict["total_cores"] = total_cpu


# ---------------------------------------------------------------- report: counters

def delta(a, b, key):
    try:
        return b[key] - a[key]
    except (KeyError, TypeError):
        return None


def counters_section(d, rep, meta, verdict):
    phases = phase_windows(meta)
    c0 = load_json(os.path.join(d, "counters_0.json"))
    files = sorted(glob.glob(os.path.join(d, "counters_*.json")))
    c_last = load_json(files[-1]) if files else None
    rep.line("4. Server counters (op 8 / op 31)")
    if not c0 or not c_last:
        rep.line("   no counter snapshots")
        return
    rep.line("   jobs/s by phase: " + ", ".join("%s %s (%.0f s)" % (p["name"], fmt_num(p.get("jobs_per_s")), p["t1"] - p["t0"]) for p in phases))
    quiet = next((p for p in phases if p["name"] == "quiet"), phases[0] if phases else None)
    a = load_json(os.path.join(d, "counters_%d.json" % quiet["c0"])) if quiet else c0
    b = load_json(os.path.join(d, "counters_%d.json" % quiet["c1"])) if quiet else c_last
    span = b["time"] - a["time"]
    sa, sb = a["stats"], b["stats"]
    jobs = delta(sa, sb, "jobs") or 0
    batches = delta(sa, sb, "batches") or 0
    rep.line("   %s phase %.1f s: jobs/s %s, batches/s %s, jobs/batch %.2f, merged duplicates %s (%.1f%% of jobs), rejected %s, failed %s, groups/s %s, held %s"
             % (quiet["name"] if quiet else "window", span, fmt_num(jobs / span), fmt_num(batches / span), jobs / max(batches, 1),
                fmt_num(delta(sa, sb, "merged")), pct(delta(sa, sb, "merged") or 0, jobs), fmt_num(delta(sa, sb, "rejected")),
                fmt_num(delta(sa, sb, "failed")), fmt_num((delta(sa, sb, "groups") or 0) / span), fmt_num(delta(sa, sb, "held"))))
    ev = delta(sa, sb, "predicate_evaluations") or 0
    rep.line("   predicate cache: hits %s, evaluations %s (%.1f ms each), evictions %s, entries %s, host %s MB, device %s MB, transient %s, residual %s"
             % (fmt_num(delta(sa, sb, "predicate_hits")), fmt_num(ev),
                (delta(sa, sb, "predicate_eval_us") or 0) / 1000 / max(ev, 1), fmt_num(delta(sa, sb, "predicate_evictions")),
                fmt_num(sb.get("predicate_entries")), fmt_num((sb.get("predicate_bytes") or 0) / 2 ** 20),
                fmt_num((sb.get("predicate_device_bytes") or 0) / 2 ** 20), fmt_num(delta(sa, sb, "predicate_transient")),
                fmt_num(delta(sa, sb, "predicate_residual"))))
    tot = Counter()
    per_device = defaultdict(Counter)
    copies = Counter()
    masks = Counter()
    for name, ib in b["indexes"].items():
        ia = a["indexes"].get(name)
        if not ia:
            continue
        for key in ("matrix_queries", "matrix_pruned_queries", "matrix_prune_fallbacks", "mixed_queries", "gpu_predicates",
                    "cagra_fp32_queries", "cagra_fp16_queries", "gpu_rerank_candidates"):
            tot[key] += delta(ia, ib, key) or 0
        copies[ib.get("copies")] += 1
        ra = {r["copy"]: r for r in ia.get("replicas", [])}
        for r in ib.get("replicas", []):
            prev = ra.get(r["copy"])
            if not prev:
                continue
            mask = r.get("device_mask", 1 << r.get("device", 0))
            masks[mask] += 1
            devs = [i for i in range(32) if mask & (1 << i)]
            for dev in devs:
                for key in ("queries", "started", "finished", "migrations", "failures"):
                    per_device[dev][key] += delta(prev, r, key) or 0
                per_device[dev]["reserved_ns"] += delta(prev, r, "reserved_ns") or 0
                per_device[dev]["copies"] += 1
    mq = tot["matrix_queries"]
    rep.line("   routing (%d indexes, copies per index %s): matrix queries %s (%s/s), pruned %s (%.0f%%), prune fallbacks %s, cagra fp16/fp32 %s/%s, mixed %s, gpu predicates %s"
             % (len(b["indexes"]), dict(copies), fmt_num(mq), fmt_num(mq / span), fmt_num(tot["matrix_pruned_queries"]),
                pct(tot["matrix_pruned_queries"], mq), fmt_num(tot["matrix_prune_fallbacks"]), fmt_num(tot["cagra_fp16_queries"]),
                fmt_num(tot["cagra_fp32_queries"]), fmt_num(tot["mixed_queries"]), fmt_num(tot["gpu_predicates"])))
    multi = [m for m in masks if bin(m).count("1") > 1]
    note = " (copies span several GPUs: per-GPU numbers count every GPU a copy touches)" if multi else ""
    rep.line("   per GPU: queries/s " + " ".join("%d:%s" % (dev, fmt_num(per_device[dev]["queries"] / span)) for dev in sorted(per_device))
             + " | batches/s " + " ".join("%d:%s" % (dev, fmt_num(per_device[dev]["finished"] / span)) for dev in sorted(per_device))
             + " | migrations %s" % fmt_num(sum(c["migrations"] for c in per_device.values())) + note)
    rep.line("   per GPU busy (sum of reserved batch time / wall): " + " ".join("%d:%.2f" % (dev, per_device[dev]["reserved_ns"] / 1e9 / span)
                                                                               for dev in sorted(per_device)), full_only=True)
    if per_device:
        q = [per_device[dev]["queries"] for dev in sorted(per_device)]
        verdict["gpu_query_imbalance"] = pct(max(q) - min(q), max(q)) if max(q) else 0


# ---------------------------------------------------------------- report: verdict

def verdict_section(rep, v):
    rep.line("5. Bottleneck summary (rules: GPU-bound util>85 on the busiest GPUs; lane-bound lanes>=50% in sync and GPUs<70;"
             " reader/CPU-bound readers>=50% on-CPU or >=30% on a mutex; client-bound readers>=50% in epoll_wait and lanes>=50% waiting for jobs)")
    g = v.get("gpu_busiest")
    rules = []
    if g is not None:
        share = v.get("gpu_worker_share")
        note = ""
        if share is not None:
            note = "; this worker's SM share %.0f%%" % share
            if g > 85 and share < 0.6 * g:
                note += " -> GPUs saturated mostly by OTHER processes (contention)"
        rules.append(("GPU-bound", g > 85, "busiest-2 GPUs util %.0f%%, mean %.0f%%, min %.0f%%%s" % (g, v["gpu_mean"], v["gpu_min"], note)))
    if "lane_sync" in v:
        rules.append(("lane-bound (lanes wait for the GPU)", v["lane_sync"] >= 50 and (g is None or g < 70),
                      "lanes in cudaStreamSynchronize %.0f%%, on-CPU %.0f%%, on mutex %.0f%%, waiting for jobs %.0f%%, %.1f cores"
                      % (v["lane_sync"], v["lane_cpu"], v["lane_mutex"], v["lane_idle"], v["lane_cores"])))
    if "reader_cpu" in v:
        rules.append(("reader/CPU-bound", v["reader_cpu"] >= 50 or v["reader_mutex"] >= 30,
                      "readers on-CPU %.0f%%, on mutex %.0f%%, in epoll_wait %.0f%%, %.1f cores; worker total %.1f cores"
                      % (v["reader_cpu"], v["reader_mutex"], v["reader_epoll"], v["reader_cores"], v.get("total_cores", 0))))
    if "reader_epoll" in v and "lane_idle" in v:
        rules.append(("client-bound (load generator too slow)", v["reader_epoll"] >= 50 and v["lane_idle"] >= 50,
                      "readers in epoll_wait %.0f%%, lanes waiting for jobs %.0f%%" % (v["reader_epoll"], v["lane_idle"])))
    for name, hit, evidence in rules:
        rep.line("   [%s] %-42s %s" % ("YES" if hit else " no", name, evidence))
    extras = []
    if v.get("api_sync_share") is not None:
        extras.append("CUDA API time: %.0f%% in *Synchronize, %.0f%% in launches" % (v["api_sync_share"], v["api_launch_share"]))
    if v.get("gpu_query_imbalance") is not None:
        extras.append("per-GPU query imbalance (max-min)/max %.0f%%" % v["gpu_query_imbalance"])
    if v.get("lane_mutex", 0) >= 30 or v.get("reader_mutex", 0) >= 30:
        extras.append("lock contention is significant (see section 3)")
    if extras:
        rep.line("   also: " + "; ".join(extras))
    hits = [name for name, hit, _ in rules if hit]
    rep.line("   VERDICT: " + (", ".join(hits) if hits else "no single rule fires: mixed; see the evidence above"))


# ---------------------------------------------------------------- report: driver

def parse_kernel_timing(d, meta):
    """Last complete ANN_KERNEL_TIMING block from DIR/kernel_timing.txt (copied from the worker log by the shell)."""
    data = read_text(os.path.join(d, "kernel_timing.txt"))
    stale = data.startswith("STALE")
    blocks = re.findall(r"(matrix gather stages, us per batch.*\n(?:  .*\n)*)", data)
    if not blocks:
        return {}
    return {"lines": blocks[-1].splitlines(), "stale": stale}


def cmd_report(args):
    d = args[0]
    meta = load_json(os.path.join(d, "meta.json")) or {}
    files = sorted(glob.glob(os.path.join(d, "counters_*.json")), key=lambda p: int(re.search(r"_(\d+)\.json$", p).group(1)))
    snaps = {int(re.search(r"_(\d+)\.json$", p).group(1)): load_json(p) for p in files}
    for p in meta.get("phases", []):
        a, b = snaps.get(p["c0"]), snaps.get(p["c1"])
        if a and b and b["time"] > a["time"]:
            p["jobs_per_s"] = (b["stats"]["jobs"] - a["stats"]["jobs"]) / (b["time"] - a["time"])
            p["t0"], p["t1"] = a["time"], b["time"]
        else:
            p["jobs_per_s"] = None
            p.setdefault("t0", 0)
            p.setdefault("t1", 0)
    quiet = next((p for p in meta.get("phases", []) if p["name"] == "quiet"), None)
    meta["quiet_jobs_per_s"] = quiet["jobs_per_s"] if quiet else None
    meta["kernel_timing"] = parse_kernel_timing(d, meta)
    rep = Report()
    verdict = {}
    rep.line("ann_profile %s: pid %s, %s, window %s s (%s), host load avg %s" %
             (meta.get("label"), meta.get("pid"), meta.get("started"), meta.get("seconds"),
              " -> ".join("%s %.0fs" % (p["name"], p["t1"] - p["t0"]) for p in meta.get("phases", [])), meta.get("loadavg", "?")))
    cmd = meta.get("cmdline", [])
    flags = [cmd[i] + " " + cmd[i + 1] for i in range(1, len(cmd) - 1) if cmd[i] in ("--lanes", "--readers", "--max-batch", "--replicate", "--matrix-gather", "--replica-routing")]
    rep.line("   worker: %s; %s" % (os.path.basename(cmd[0]) if cmd else "?", ", ".join(flags)))
    gpu_coarse(d, rep, meta, verdict)
    gpu_fine(d, rep, meta, verdict)
    cpu_section(d, rep, meta, verdict)
    counters_section(d, rep, meta, verdict)
    verdict_section(rep, verdict)
    with open(os.path.join(d, "report.txt"), "w") as f:
        f.write("\n".join(rep.short) + "\n")
    with open(os.path.join(d, "report_full.txt"), "w") as f:
        f.write("\n".join(rep.full) + "\n")
    print("\n".join(rep.short))


def cmd_nsys_info(args):
    """nsys-info DIR SECONDS: summarize DIR/trace.sqlite (events, drops, per-device kernel time) into DIR/info.json."""
    d, seconds = args[0], float(args[1])
    info = load_json(os.path.join(d, "info.json")) or {}
    info["seconds"] = seconds
    rep = glob.glob(os.path.join(d, "*.nsys-rep"))
    if rep:
        info["rep_mb"] = os.path.getsize(rep[0]) / 2 ** 20
    db = glob.glob(os.path.join(d, "*.sqlite"))
    if db:
        try:
            con = sqlite3.connect(db[0])
            cur = con.cursor()
            kernels = cur.execute("SELECT COUNT(*), MIN(start), MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()
            info["kernels"] = kernels[0]
            span = (kernels[2] - kernels[1]) / 1e9 if kernels[0] else 0
            info["trace_span_s"] = span
            info["per_device"] = [{"device": r[0], "kernels": r[1], "ms_per_s": r[2] / 1e6 / span if span else 0}
                                  for r in cur.execute("SELECT deviceId, COUNT(*), SUM(end-start) FROM CUPTI_ACTIVITY_KIND_KERNEL GROUP BY deviceId")]
            info["api_calls"] = cur.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME").fetchone()[0]
            if kernels[0] == 0:
                info["ok"] = False
                info["reason"] = "the start/stop cycle collected no CUDA events (seen once in three cycles, cause unknown): run again"
                with open(os.path.join(d, "info.json"), "w") as f:
                    json.dump(info, f)
                return
            diag = []
            produced = dropped = buffers = None
            try:
                for r in cur.execute("SELECT severity, text FROM DIAGNOSTIC_EVENT"):
                    text = " ".join(str(r[1]).split())
                    m = re.search(r"CUPTI events produced:\s*(\d+)", text)
                    if m:
                        produced = int(m.group(1))
                        m2 = re.search(r"CUPTI buffers: (\d+) of (\d+)", text)
                        buffers = (int(m2.group(1)), int(m2.group(2))) if m2 else None
                        continue
                    m = re.search(r"incomplete CUPTI events dropped:\s*(\d+)", text)
                    if m:
                        dropped = int(m.group(1))
                        continue
                    if re.search(r"drop|lost|overflow|fail", text, re.I):
                        diag.append(text[:100])
            except sqlite3.Error:
                diag.append("no DIAGNOSTIC_EVENT table")
            parts = []
            if produced is not None:
                parts.append("CUPTI events %s" % fmt_num(produced))
            if buffers:
                parts.append("buffers %d/%d used" % buffers)
            if dropped is not None:
                parts.append("incomplete events dropped %d (%.2f%%)" % (dropped, pct(dropped, produced or dropped)))
            if diag:
                parts.append("WARNINGS: " + " | ".join(diag[:2]))
            info["diagnostics"] = "; ".join(parts) if parts else "no drop/loss diagnostics in the trace"
            info["dropped_pct"] = pct(dropped or 0, produced or 1)
            info["ok"] = True
        except sqlite3.Error as e:
            info["ok"] = False
            info["reason"] = "trace has no CUDA tables (%s): the start/stop cycle collected nothing, run again" % e
    else:
        info.setdefault("ok", False)
        info.setdefault("reason", "no sqlite export")
    with open(os.path.join(d, "info.json"), "w") as f:
        json.dump(info, f)


def cmd_meta(args):
    """meta OUT.json key=value...: record the run's parameters, the worker's command line and the load average."""
    out = args[0]
    meta = {}
    for kv in args[1:]:
        k, v = kv.split("=", 1)
        meta[k] = v
    for k in ("pid", "seconds", "log_offset"):
        if k in meta:
            meta[k] = int(meta[k])
    phases = []
    for item in meta.get("phases", "").split(","):
        if item:
            name, c0, c1 = item.split(":")
            phases.append({"name": name, "c0": int(c0), "c1": int(c1)})
    meta["phases"] = phases
    home = os.path.expanduser("~")
    try:
        meta["cmdline"] = [a.replace(home, "~") for a in
                           open("/proc/%d/cmdline" % meta["pid"], "rb").read().decode(errors="replace").split("\0")[:-1]]
    except (OSError, KeyError):
        meta["cmdline"] = []
    if "log" in meta:
        meta["log"] = meta["log"].replace(home, "~")
    try:
        meta["loadavg"] = open("/proc/loadavg").read().split()[0]
    except OSError:
        pass
    with open(out, "w") as f:
        json.dump(meta, f)


COMMANDS = {"counters": cmd_counters, "meta": cmd_meta, "threads": cmd_threads, "stall": cmd_stall, "gdb-roles": cmd_gdb_roles,
            "report": cmd_report, "nsys-info": cmd_nsys_info}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        raise SystemExit(__doc__)
    COMMANDS[sys.argv[1]](sys.argv[2:])
