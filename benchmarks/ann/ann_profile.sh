#!/bin/bash
# ann_profile.sh LABEL SECONDS            profile the running ANN worker on both paths while a load generator runs
# ann_profile.sh nsys-worker ROOT [flags]  restart the running worker under "nsys launch" (same argv and environment,
#                                          same port), load every index under ROOT; then LABEL runs collect a CUDA trace
#
# Output: $ANN_PROF_DIR/LABEL/report.txt (60 lines), report_full.txt, and the raw data (gpu.csv, pmon.csv, gdb/*.txt,
# counters_*.json, threads*.json, nsys/).
#
# Environment (defaults in brackets): ANN_B [~/b200dev] the test tree; ANN_PROF_DIR [$ANN_B/prof]; ANN_PID_FILE
# [$ANN_B/worker.pid]; ANN_HOST [127.0.0.1]; ANN_PORT and ANN_TOKEN_FILE [from the worker's command line];
# ANN_INDEX_ROOT [index names from the worker log]; ANN_GDB_SNAPSHOTS [20]; ANN_GDB_DEPTH [16];
# ANN_NSYS_SECONDS [5] seconds of CUDA trace inside the window; ANN_NSYS_SESSION [annprof]; ANN_PY [python3];
# ANN_GDB_FAST [0] 1 = cheaper gdb snapshots without libcuvs symbols.
#
# The window is split into phases: quiet (counters, per-thread CPU and nvidia-smi only), gdb (stack snapshots: each
# attach stops the worker for ~0.5 s), and, when the worker runs under nsys launch, nsys (CUDA trace). Throughput is
# measured per phase from op 8 so the report states what each profiler cost.
set -u
B=${ANN_B:-$HOME/b200dev}
PROF=${ANN_PROF_DIR:-$B/prof}
PIDFILE=${ANN_PID_FILE:-$B/worker.pid}
HOST=${ANN_HOST:-127.0.0.1}
PY=${ANN_PY:-python3}
HELPER=$(dirname "$(readlink -f "$0")")/ann_profile.py
SESSION=${ANN_NSYS_SESSION:-annprof}
export TMPDIR=$PROF/tmp
mkdir -p "$TMPDIR"

die() { echo "ann_profile: $*" >&2; exit 1; }
now() { date +%s.%N; }

worker_pid() {
	local p
	p=$(cat "$PIDFILE" 2>/dev/null) || die "no pid file $PIDFILE"
	[ -d /proc/$p ] || die "worker $p (from $PIDFILE) is not running"
	echo $p
}

# argument following FLAG in the worker's command line
worker_arg() { tr '\0' '\n' < /proc/$1/cmdline | grep -A1 -x -- "$2" | tail -1; }

# ------------------------------------------------------------------ nsys-worker
nsys_worker() {
	local root=$1; shift
	[ -d "$root" ] || die "no index root $root"
	local old; old=$(worker_pid)
	local sock port
	sock=$(worker_arg $old --listen); port=$(worker_arg $old --listen-tcp)
	cp /proc/$old/cmdline "$PROF/worker.argv"; cp /proc/$old/environ "$PROF/worker.environ"
	local -a ARGV ENVV
	mapfile -d '' -t ARGV < "$PROF/worker.argv"; mapfile -d '' -t ENVV < "$PROF/worker.environ"
	local log; log=$(readlink /proc/$old/fd/2); [ -n "$log" ] || log=$B/worker-run.log
	local cwd; cwd=$(readlink /proc/$old/cwd)
	echo "stopping worker $old, restarting under nsys launch (session $SESSION): ${ARGV[*]:0:1} ... $*"
	kill $old; for i in $(seq 150); do [ -d /proc/$old ] || break; sleep 0.2; done
	[ -d /proc/$old ] && die "worker $old did not exit"
	cd "$cwd" || cd "$B"
	setsid nohup env -i "${ENVV[@]}" TMPDIR="$TMPDIR" nsys launch --session-new="$SESSION" --trace=cuda \
		"${ARGV[@]}" "$@" > "$log" 2>&1 < /dev/null &
	local launcher=$!
	local w=""
	for i in $(seq 240); do
		w=$(ss -ltnp 2>/dev/null | grep "[: ]${port##*:} " | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | head -1)
		[ -n "$w" ] && [ -S "$sock" ] && break; sleep 0.5
	done
	[ -n "$w" ] || die "worker did not come up under nsys (launcher $launcher, log $log)"
	echo $w > "$PIDFILE"
	echo "worker $w (nsys launcher $launcher) listening on $port; LD_PRELOAD=$(tr '\0' '\n' < /proc/$w/environ | sed -n 's/^LD_PRELOAD=//p' | cut -c1-80)..."
	# the environment that load.py needs (a python with the right libraries is not required: stdlib only)
	$PY -I "$B/load.py" "$sock" "$root"
	nsys sessions list
}

# ------------------------------------------------------------------ collectors
counters() {	# counters INDEX -> $OUT/counters_INDEX.json
	$PY -I "$HELPER" counters "$OUT/counters_$1.json" --host "$HOST" --port "$PORT" --token-file "$TOKEN" "${INDEX_ARGS[@]}" \
		|| echo "counters $1 failed" >&2
}

gpu_sampler() {
	while :; do
		local t; t=$(now)
		nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,power.draw --format=csv,noheader,nounits 2>/dev/null \
			| sed "s/^/$t, /" >> "$OUT/gpu.csv"
		nvidia-smi pmon -c 1 -s um 2>/dev/null | awk -v t="$t" -v p="$PID" '$2==p {print t","$1","$2","$4","$10}' >> "$OUT/pmon.csv"
		sleep 1
	done
}

# One stack snapshot. SIGSTOP first so every thread stops at the same instant (gdb attaches thread by thread over
# tens of ms, during which the lanes it stopped first starve the readers of work and the sample is biased), then gdb
# prints every thread's stack and orig_rax (the syscall the thread is in, -1 when running user code), then SIGCONT.
# ANN_GDB_FAST=1 loads symbols for libc, libcuda, libcudart and libstdc++ only: about a third of the stall, but frames
# inside libcuvs stay unresolved.
gdb_snapshot() {	# gdb_snapshot N
	local s e
	local -a libs=()
	if [ "${ANN_GDB_FAST:-0}" = 1 ]; then
		libs=(-iex "set auto-solib-add off" -ex "sharedlibrary libc.so" -ex "sharedlibrary libcuda.so" -ex "sharedlibrary libcudart" -ex "sharedlibrary libstdc++")
	fi
	s=$(now)
	kill -STOP $PID
	gdb -nx -batch -iex "set debuginfod enabled off" -iex "set auto-load off" -iex "set print thread-events off" \
		-iex "set pagination off" "${libs[@]}" -p $PID \
		-ex "thread apply all bt ${ANN_GDB_DEPTH:-16}" -ex 'thread apply all p (long)$orig_rax' > "$OUT/gdb/$(printf %02d $1).txt" 2>&1
	kill -CONT $PID
	e=$(now)
	echo "$e - $s" | bc >> "$OUT/gdb/walls.txt"
}

# ------------------------------------------------------------------ profile
profile() {
	local label=$1 seconds=$2
	PID=$(worker_pid)
	trap 'kill -CONT $PID 2>/dev/null' EXIT INT TERM	# never leave the worker SIGSTOP'd
	OUT=$PROF/$label
	rm -rf "$OUT"; mkdir -p "$OUT/gdb" "$OUT/nsys"
	PORT=${ANN_PORT:-$(worker_arg $PID --listen-tcp | sed 's/.*://')}; PORT=${PORT:-7700}
	TOKEN=${ANN_TOKEN_FILE:-$(worker_arg $PID --auth-token-file)}; TOKEN=${TOKEN:-$B/token}
	LOG=$(readlink /proc/$PID/fd/2); [ -f "$LOG" ] || LOG=$B/worker-run.log
	LOG_OFFSET=$(stat -c %s "$LOG" 2>/dev/null || echo 0)
	INDEX_ARGS=()
	if [ -n "${ANN_INDEX_ROOT:-}" ]; then
		for n in $(ls "$ANN_INDEX_ROOT"); do INDEX_ARGS+=(--index "$n"); done
	else
		for n in $(sed -n 's/^ann index \([^:]*\):.*/\1/p' "$LOG" | sort -u); do INDEX_ARGS+=(--index "$n"); done
	fi
	local nsys_on=0 nsys_s=0
	if tr '\0' '\n' < /proc/$PID/environ | grep -q '^NSYS_PROFILING_SESSION_ID='; then
		if nsys sessions list 2>/dev/null | grep -q " $SESSION\$\| $SESSION "; then
			nsys_on=1
			nsys_s=${ANN_NSYS_SECONDS:-5}
			[ $nsys_s -gt $((seconds / 4)) ] && nsys_s=$((seconds / 4))
			[ $nsys_s -lt 2 ] && nsys_s=2
		else
			echo '{"ok": false, "reason": "worker runs under nsys but session '"$SESSION"' is not listed (TMPDIR mismatch?)"}' > "$OUT/nsys/info.json"
		fi
	fi
	local rest=$((seconds - nsys_s))
	local quiet_s=$((rest * 40 / 100)) gdb_s
	[ $quiet_s -lt 3 ] && quiet_s=3
	gdb_s=$((rest - quiet_s))
	local snapshots=${ANN_GDB_SNAPSHOTS:-20}
	echo "ann_profile $label: worker $PID port $PORT, $((${#INDEX_ARGS[@]} / 2)) indexes, phases quiet ${quiet_s}s gdb ${gdb_s}s nsys ${nsys_s}s (nsys $([ $nsys_on = 1 ] && echo on || echo off)), $snapshots snapshots -> $OUT"
	local started; started=$(date +%FT%T)
	gpu_sampler & local sampler=$!
	local c=0 phases=""
	# quiet phase
	counters $c; $PY -I "$HELPER" threads $PID "$OUT/threads0.json"
	sleep $quiet_s
	counters $((c + 1)); $PY -I "$HELPER" threads $PID "$OUT/threads1.json"
	phases="quiet:$c:$((c + 1))"; c=$((c + 1))
	# gdb phase
	$PY -I "$HELPER" stall $PID "$OUT/stalls.txt" & local watcher=$!
	local t_end=$(echo "$(now) + $gdb_s" | bc) interval
	interval=$(echo "scale=3; $gdb_s / $snapshots" | bc)
	local n=0
	while [ $n -lt $snapshots ]; do
		local t0; t0=$(now)
		[ "$(echo "$t0 + 0.5 > $t_end" | bc)" = 1 ] && break
		n=$((n + 1))
		gdb_snapshot $n
		local gap; gap=$(echo "$interval - ($(now) - $t0)" | bc)
		[ "$(echo "$gap < 0.25" | bc)" = 1 ] && gap=0.25
		sleep $gap
	done
	kill -TERM $watcher 2>/dev/null; wait $watcher 2>/dev/null
	counters $((c + 1)); phases="$phases,gdb:$c:$((c + 1))"; c=$((c + 1))
	# nsys phase
	if [ $nsys_on = 1 ]; then
		nsys start --session="$SESSION" --sample=none --cpuctxsw=none -o "$OUT/nsys/trace" -f true > "$OUT/nsys/start.log" 2>&1 \
			|| { echo "nsys start failed:"; cat "$OUT/nsys/start.log"; }
		sleep 1	# CUPTI initialisation; keep it out of the trace phase's throughput
		counters $((c + 1)); c=$((c + 1))
		sleep $nsys_s
		counters $((c + 1)); phases="$phases,nsys:$c:$((c + 1))"; c=$((c + 1))
		nsys stop --session="$SESSION" > "$OUT/nsys/stop.log" 2>&1 || { echo "nsys stop failed:"; cat "$OUT/nsys/stop.log"; }
	fi
	kill $sampler 2>/dev/null; wait $sampler 2>/dev/null
	# ANN_KERNEL_TIMING stage lines the worker printed during the window (else the last ones in the log, marked stale)
	tail -c +$((LOG_OFFSET + 1)) "$LOG" 2>/dev/null | grep -A2 "^matrix gather stages" > "$OUT/kernel_timing.txt"
	[ -s "$OUT/kernel_timing.txt" ] || { grep -A2 "^matrix gather stages" "$LOG" 2>/dev/null | tail -3 | sed '1i STALE: from before this run' > "$OUT/kernel_timing.txt"; }
	$PY -I "$HELPER" meta "$OUT/meta.json" label="$label" pid=$PID seconds=$seconds started="$started" log="$LOG" \
		log_offset=$LOG_OFFSET phases="$phases"
	if [ $nsys_on = 1 ] && [ -f "$OUT/nsys/trace.nsys-rep" ]; then
		nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum --format csv \
			--force-export=true -o "$OUT/nsys/stats" "$OUT/nsys/trace.nsys-rep" > "$OUT/nsys/stats.log" 2>&1 \
			|| { echo "nsys stats failed:"; tail -5 "$OUT/nsys/stats.log"; }
		$PY -I "$HELPER" nsys-info "$OUT/nsys" $nsys_s
	elif [ $nsys_on = 1 ]; then
		echo '{"ok": false, "reason": "nsys stop produced no trace.nsys-rep (see nsys/stop.log)"}' > "$OUT/nsys/info.json"
	fi
	$PY -I "$HELPER" report "$OUT"
	echo "report: $OUT/report.txt (full: report_full.txt)"
}

case ${1:-} in
	nsys-worker) shift; nsys_worker "$@" ;;
	"" | -h | --help) sed -n '2,20p' "$0"; exit 0 ;;
	*) [ $# -ge 2 ] || die "usage: ann_profile.sh LABEL SECONDS | nsys-worker ROOT [flags]"; profile "$1" "$2" ;;
esac
