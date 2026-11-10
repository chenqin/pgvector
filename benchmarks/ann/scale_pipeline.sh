#!/bin/bash
# scale_pipeline.sh OUT_ROOT FACTOR SHARDS QUANT [N]
#
# Export and build the synthetic group indexes one group at a time, so the export of
# every group never has to exist at once: for each group of the catalog, in order,
#   scale_export.py --only GROUP --factor FACTOR   -> EXPORT_ROOT/GROUP
#   build_cuvs_index.py --from-export ... --shards SHARDS --dtype float16 --quantize QUANT
#                                                  -> OUT_ROOT/GROUP
#   rm -r EXPORT_ROOT/GROUP                           (after a successful build only)
# and one line per group in LOG_DIR/pipeline.log with the seconds of each step.
#
#   OUT_ROOT  index root: one directory per group, nothing else (a worker loads all of it)
#   FACTOR    copies of each seed row (590: 5.82M rows a group; 4715: 46.5M)
#   SHARDS    1: N groups in flight, each on its own GPU (CUDA_VISIBLE_DEVICES rotates
#             over GPUS). More than 1: one group at a time; the builder puts shard I on
#             visible GPU I modulo their number, all at once
#   QUANT     nvfp4 or fp8
#   N         groups in flight with SHARDS=1 (default: the number of GPUS)
#
# A group whose OUT_ROOT/GROUP/manifest exists is skipped (the builder writes the
# manifest last). A group directory without a manifest, or an export directory, left by
# an earlier failed run is removed and redone. The first failure stops new groups from
# starting, lets the groups in flight finish, keeps the failed group's export, partial
# index and logs, and exits 1. Keep EXPORT_ROOT on OUT_ROOT's file system: the builder
# then hard-links the export's files into the index instead of copying them, so an
# export costs no disk beyond the index it becomes.
#
# Environment (defaults suit the GPU test host):
#   ONLY         comma-separated groups to do instead of the whole catalog
#   EXPORT_ROOT  default OUT_ROOT.export        LOG_DIR  default OUT_ROOT.logs
#   SEED_ROOT    directory of <group>/index seed directories
#   GROUPS_JSON  group catalog, a JSON list of {stem, table, dimension, augment}
#   EXPORTER, BUILDER  scale_export.py and build_cuvs_index.py
#   ENV_SH       sourced first, for a python3 with numpy, cupy and cuvs
#   GPUS         GPU indexes to use, e.g. "2 3 4 5" (default: every GPU of nvidia-smi)
#   MIN_FREE_GB  refuse to start a group with less free disk than this (default 50)

usage() { sed -n '2,/^$/s/^# \{0,1\}//p' "$0" >&2; exit 2; }
[ $# -ge 4 ] && [ $# -le 5 ] || usage
OUT_ROOT=$1; FACTOR=$2; SHARDS=$3; QUANT=$4
case "$FACTOR$SHARDS" in *[!0-9]*|'') usage;; esac
case "$QUANT" in nvfp4|fp8) ;; *) usage;; esac
[ "$FACTOR" -ge 1 ] && [ "$SHARDS" -ge 1 ] || usage

HERE=$(cd "$(dirname "$0")" && pwd)
ENV_SH=${ENV_SH-$HOME/pgvector-test.tKp5o0/env.sh}
if [ -n "$ENV_SH" ] && [ -f "$ENV_SH" ]; then
  source "$ENV_SH"; unset PGDATABASE ANN_PG_DSN
fi
first_file() { for f in "$@"; do [ -e "$f" ] && { echo "$f"; return; }; done; echo "$1"; }
EXPORTER=${EXPORTER:-$(first_file "$HERE/scale_export.py" "$HOME/b200dev/src/benchmarks/ann/scale_export.py")}
BUILDER=${BUILDER:-$(first_file "$HERE/../../ann/builder/build_cuvs_index.py" "$HOME/b200dev/src/ann/builder/build_cuvs_index.py")}
SEED_ROOT=${SEED_ROOT:-$(first_file "$HOME"/b200dev/bundle/pgvector_*/build/seg_ads)}
GROUPS_JSON=${GROUPS_JSON:-$HOME/b200dev/mock/groups.json}
EXPORT_ROOT=${EXPORT_ROOT:-${OUT_ROOT%/}.export}
LOG_DIR=${LOG_DIR:-${OUT_ROOT%/}.logs}
MIN_FREE_GB=${MIN_FREE_GB:-50}
[ -n "${GPUS-}" ] && gpus_given=1 || gpus_given=0
GPUS=${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null)}
read -r -a gpus <<< "$(echo $GPUS | tr ',' ' ')"

die() { echo "scale_pipeline: $*" >&2; exit 1; }
for f in "$EXPORTER" "$BUILDER" "$GROUPS_JSON"; do [ -f "$f" ] || die "no file $f"; done
[ -d "$SEED_ROOT" ] || die "no seed directory $SEED_ROOT (set SEED_ROOT)"
[ ${#gpus[@]} -ge 1 ] || die "no GPUs (set GPUS)"
if [ "$SHARDS" -eq 1 ]; then N=${5:-${#gpus[@]}}; else N=1; fi
case "$N" in *[!0-9]*|''|0) usage;; esac
[ "$SHARDS" -eq 1 ] || [ "${5:-1}" = 1 ] || echo "scale_pipeline: SHARDS=$SHARDS builds one group at a time; N=$5 ignored" >&2

mkdir -p "$OUT_ROOT" "$EXPORT_ROOT" "$LOG_DIR" || die "cannot create the output directories"
exec 9> "$LOG_DIR/lock"
flock -n 9 || die "another scale_pipeline.sh is running on $OUT_ROOT"
[ "$(stat -c %d "$OUT_ROOT")" = "$(stat -c %d "$EXPORT_ROOT")" ] ||
  echo "scale_pipeline: $EXPORT_ROOT is on another file system than $OUT_ROOT: the builder will copy, not link" >&2

groups=$(python3 -I - "$GROUPS_JSON" "${ONLY-}" <<'PY'
import json, re, sys
tables = [group['table'] for group in json.load(open(sys.argv[1]))]
bad = [table for table in tables if not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*', table)]
if bad:
    sys.exit('group names must be plain directory names: ' + ', '.join(bad))
only = [table for table in sys.argv[2].split(',') if table]
unknown = [table for table in only if table not in tables]
if unknown:
    sys.exit('ONLY: no group named ' + ', '.join(unknown))
print(' '.join(table for table in tables if not only or table in only))
PY
) || die "cannot read the groups of $GROUPS_JSON"
[ -n "$groups" ] || die "no groups in $GROUPS_JSON"

now_ms() { date +%s%3N; }
seconds() { echo "$(( $1 / 1000 )).$(( $1 % 1000 / 100 ))"; }
note() { echo "$(date +%Y-%m-%dT%H:%M:%S) $*" | tee -a "$LOG_DIR/pipeline.log"; }

# one_group GROUP GPU: export, build, delete the export. GPU is "all" when the builder spreads the shards.
one_group() {
  local t=$1 gpu=$2 child= start export_ms build_ms rows shards
  trap '[ -n "$child" ] && kill "$child" 2>/dev/null; exit 143' TERM INT
  [ -n "$t" ] || return 1
  [ ! -e "$EXPORT_ROOT/$t" ] || { note "$t: removing the export an earlier run left"; rm -rf "$EXPORT_ROOT/$t"; }
  [ ! -e "$OUT_ROOT/$t" ] || { note "$t: removing the incomplete index an earlier run left"; rm -rf "$OUT_ROOT/$t"; }
  start=$(now_ms)
  python3 -I "$EXPORTER" --seed-root "$SEED_ROOT" --groups "$GROUPS_JSON" --factor "$FACTOR" \
    --out "$EXPORT_ROOT" --only "$t" --jobs 1 > "$LOG_DIR/$t.export.log" 2>&1 & child=$!
  wait $child || { note "$t FAILED in export after $(seconds $(( $(now_ms) - start )))s: see $LOG_DIR/$t.export.log"; return 1; }
  export_ms=$(( $(now_ms) - start )); start=$(now_ms)
  if [ "$gpu" != all ]; then
    export CUDA_VISIBLE_DEVICES=$gpu
  elif [ $gpus_given -eq 1 ]; then
    export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${gpus[*]}")
  fi
  python3 -I "$BUILDER" --from-export "$EXPORT_ROOT/$t" --out "$OUT_ROOT/$t" --dtype float16 \
    --quantize "$QUANT" --shards "$SHARDS" > "$LOG_DIR/$t.build.log" 2>&1 & child=$!
  wait $child && [ -f "$OUT_ROOT/$t/manifest" ] ||
    { note "$t FAILED in build after $(seconds $(( $(now_ms) - start )))s (export $(seconds $export_ms)s): see $LOG_DIR/$t.build.log"; return 1; }
  build_ms=$(( $(now_ms) - start ))
  rm -rf "$EXPORT_ROOT/$t"
  rows=$(awk '$1 == "rows" {print $2}' "$OUT_ROOT/$t/manifest")
  shards=$(awk '$1 == "shards" {print $2}' "$OUT_ROOT/$t/manifest")
  note "$t ok rows=$rows shards=$shards quant=$QUANT gpu=$gpu export_s=$(seconds $export_ms) build_s=$(seconds $build_ms)" \
    "index_mb=$(( $(du -sk "$OUT_ROOT/$t" | cut -f1) / 1024 ))"
}

slot_pid=(); slot_group=(); in_flight=0; failed=0; built=0; skipped=0
started=$(now_ms)
trap 'kill "${slot_pid[@]}" 2>/dev/null; wait; note "interrupted"; exit 130' INT TERM
reap() {   # wait for a group in flight to end; free its slot; a group without a manifest has failed
  wait -n
  local s
  for s in "${!slot_pid[@]}"; do
    kill -0 "${slot_pid[$s]}" 2>/dev/null && continue
    if [ -f "$OUT_ROOT/${slot_group[$s]}/manifest" ]; then built=$((built + 1)); else failed=1; fi
    unset "slot_pid[$s]" "slot_group[$s]"; in_flight=$((in_flight - 1))
  done
}
note "start out=$OUT_ROOT factor=$FACTOR shards=$SHARDS quant=$QUANT in_flight=$N gpus=${gpus[*]} groups=$(echo $groups | wc -w)"
for t in $groups; do
  if [ -f "$OUT_ROOT/$t/manifest" ]; then
    note "$t skipped: $OUT_ROOT/$t is complete"; skipped=$((skipped + 1)); continue
  fi
  while [ $in_flight -ge "$N" ] && [ $failed -eq 0 ]; do reap; done
  [ $failed -eq 0 ] || break
  free_gb=$(( $(df -Pk "$OUT_ROOT" | awk 'NR == 2 {print $4}') / 1048576 ))
  if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
    note "$t FAILED: $free_gb GB free under $OUT_ROOT, less than MIN_FREE_GB=$MIN_FREE_GB"; failed=1; break
  fi
  s=0; while [ -n "${slot_pid[$s]-}" ]; do s=$((s + 1)); done    # the first free slot keeps its GPU
  if [ "$SHARDS" -eq 1 ]; then gpu=${gpus[$((s % ${#gpus[@]}))]}; else gpu=all; fi
  one_group "$t" "$gpu" &
  slot_pid[$s]=$!; slot_group[$s]=$t; in_flight=$((in_flight + 1))
done
while [ $in_flight -gt 0 ]; do reap; done
total=$(seconds $(( $(now_ms) - started )))
if [ $failed -ne 0 ]; then
  note "stopped on a failure after ${total}s: $built built; logs in $LOG_DIR, the failed group's export under $EXPORT_ROOT"
  exit 1
fi
rmdir "$EXPORT_ROOT" 2>/dev/null
note "done in ${total}s: $built built, $skipped skipped, $(du -sh "$OUT_ROOT" | cut -f1) under $OUT_ROOT"
