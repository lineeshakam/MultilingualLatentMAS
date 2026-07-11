#!/usr/bin/env bash
# het_belebele_sg and hom_belebele_sg's driver gave up after 3 attempts each
# once token_based_mas finished, because of a stale-bytecode AttributeError
# ('CheckpointManager' object has no attribute 'delete_result' -- the method
# exists in the current source, but a __pycache__ .pyc compiled before it
# was added got treated as still-valid; all __pycache__ dirs repo-wide have
# now been cleared). The driver's own retry loop won't come back to
# belebele_sg configs -- it already moved on to het_mgsm/hom_mgsm
# permanently -- so this relaunches the two missing modes (oneflow,
# latent_based_mas_ours) directly via --comm-modes, --resume picking up the
# cached single_agent_baseline + token_based_mas results.
#
# $1: instance name (het|hom), used only for logging/config selection.
# 2026-07-11 update: the driver crashes were chimera imports, not stale
# bytecode (see memory chimera-shared-package-root-cause), and the driver-made
# caches were quarantined. The mode list is therefore instance-specific now:
#   hom: all four modes (single_agent + token were chimera-tainted and
#        quarantined; oneflow/latent never ran)
#   het: token_based_mas only (single_agent/oneflow/latent covered by the
#        clean live safety_rerun process; its chimera token was quarantined)
set -u
# Prevents a class of bug hit 2026-07-08: a __pycache__ .pyc compiled before
# a source edit can be treated as still-valid if the .py mtime ever moves
# backward (e.g. a git checkout/reset after the edit), silently running old
# bytecode in a fresh process despite the source file on disk being current.
# Cost is a full recompile per launch, negligible next to these jobs' runtime.
export PYTHONDONTWRITEBYTECODE=1
INSTANCE=$1
CONFIG="configs/bench_suite/${INSTANCE}_belebele_sg.yaml"
if [ "$INSTANCE" = "hom" ]; then
  MODES=single_agent_baseline,token_based_mas,oneflow,latent_based_mas_ours
else
  MODES=token_based_mas
fi
cd "$(dirname "$0")/.."
LOG="logs/bench_suite/${INSTANCE}_remaining_modes_watcher.log"
LOCK=/tmp/multilinguallatentmas_gpu_claim.lock
CLAIMS=/tmp/multilinguallatentmas_gpu_claims
mkdir -p logs/bench_suite
touch "$CLAIMS"
export LOG CONFIG INSTANCE MODES
log() { echo "[$(date -u +%FT%TZ)] [${INSTANCE}_remaining] $*" >> "$LOG"; }

try_claim_and_launch() (
  set -u
  N_NEEDED=3
  FREE_GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F',' '{gsub(/ /,"",$2); if ($2+0 < 500) print $1}')
  # Exclude GPUs claimed by a still-alive pid (claims file rows:
  # "<epoch> <pid> <gpu,gpu,...> <label>"). The <500MiB test alone
  # double-books a slice mid-load (a fresh multi-agent job occupies its
  # 2nd/3rd GPU only minutes after launch); it also can't see reservations
  # like the het router-fix requeue, which relaunches pinned to 4,5,6 the
  # moment the het driver exits, without any lock of its own.
  LIVE_CLAIMED=$(while read -r _ts _pid _gpus _label; do
      [ -n "${_pid:-}" ] || continue
      kill -0 "$_pid" 2>/dev/null && echo "$_gpus" | tr ',' '\n'
    done < "$CLAIMS" | sort -u)
  if [ -n "$LIVE_CLAIMED" ]; then
    FREE_GPUS=$(echo "$FREE_GPUS" | grep -vxF "$LIVE_CLAIMED" || true)
  fi
  N_FREE=$(echo "$FREE_GPUS" | grep -c '[0-9]')
  [ "$N_FREE" -lt "$N_NEEDED" ] && return 1

  CLAIMED=$(echo "$FREE_GPUS" | head -"$N_NEEDED" | paste -sd, -)
  log "claimed gpus=$CLAIMED"

  export CUDA_VISIBLE_DEVICES=$CLAIMED
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export PYTHONPATH=/home/hthakur/MultilingualLatentMAS/src

  JOBLOG="logs/bench_suite/${INSTANCE}_belebele_sg.remaining_modes.log"
  log "launching $CONFIG --comm-modes $MODES on gpus=$CLAIMED"
  setsid nohup python scripts/run_coordination_pipeline.py \
    --config "$CONFIG" --resume \
    --comm-modes "$MODES" \
    >> "$JOBLOG" 2>&1 < /dev/null {LOCK_FD}<&- &
  JOB_PID=$!
  disown -a
  echo "$(date +%s) $JOB_PID $CLAIMED ${INSTANCE}_remaining" >> "$CLAIMS"
  log "launched, pid=$JOB_PID -- verifying it survives startup (45s)"
  sleep 45
  if kill -0 "$JOB_PID" 2>/dev/null; then
    log "pid=$JOB_PID alive after 45s, launch looks healthy"
    return 0
  else
    log "pid=$JOB_PID died within 45s -- likely OOM/crash; will retry. Last lines:"
    tail -15 "$JOBLOG" >> "$LOG"
    return 1
  fi
)

log "watching for >=3 idle GPUs (<500MiB used)"
# Bug fixed 2026-07-08: see watch_and_launch_staircase.sh -- backgrounded
# jobs launched inside a 'flock "$LOCK" bash -c ...' inherited the lock FD
# and held it for their entire lifetime, starving every other watcher.
exec {LOCK_FD}<>"$LOCK"
while true; do
  # Never launch while another pipeline process is using this same config --
  # two runs would share the stage checkpoints and _results cache dir.
  if pgrep -f "run_coordination_pipeline.*${INSTANCE}_belebele_sg" >/dev/null 2>&1; then
    sleep 600
    continue
  fi
  flock -x "$LOCK_FD"
  RC=0
  try_claim_and_launch || RC=$?
  flock -u "$LOCK_FD"
  [ "$RC" -eq 0 ] && break
  sleep 300
done
