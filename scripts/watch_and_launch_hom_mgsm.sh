#!/usr/bin/env bash
# hom's driver (run_instance.sh) gave up on hom_mgsm after 3 attempts and
# exited entirely (instance=hom DONE rc=1) -- the 3rd attempt OOM'd because
# it was pinned to GPUs 0,1,2 (hom's original slice from launch) at the exact
# moment watch_and_launch_safety_rerun.sh's nvidia-smi check saw those same
# GPUs as briefly free (right after hom_belebele_sg's crash, before hom's own
# driver reclaimed them) and launched het_belebele_sg's rerun there too --
# oversubscribing the slice once hom_mgsm needed to load another agent.
# hom's driver won't retry on its own (it already exited), so this relaunches
# hom_mgsm directly once 3 GPUs are free elsewhere -- not hardcoded to 0,1,2,
# to avoid repeating the exact collision.
set -u
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
# Absolute PYTHONPATH: without it the LRL-MRRE-MAS editable .pth hijacks
# `import shared` (chimera imports; tainted the 2026-07-11 hom_mgsm launch).
export PYTHONPATH=/home/hthakur/MultilingualLatentMAS/src
LOG=logs/bench_suite/hom_mgsm_relaunch_watcher.log
LOCK=/tmp/multilinguallatentmas_gpu_claim.lock
CLAIMS=/tmp/multilinguallatentmas_gpu_claims
mkdir -p logs/bench_suite
touch "$CLAIMS"
export LOG
log() { echo "[$(date -u +%FT%TZ)] [hom_mgsm_relaunch] $*" >> "$LOG"; }

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

  JOBLOG=logs/bench_suite/hom_mgsm.log
  log "launching hom_mgsm --resume on gpus=$CLAIMED"
  setsid nohup python scripts/run_coordination_pipeline.py \
    --config configs/bench_suite/hom_mgsm.yaml --resume \
    >> "$JOBLOG" 2>&1 < /dev/null {LOCK_FD}<&- &
  JOB_PID=$!
  disown -a
  echo "$(date +%s) $JOB_PID $CLAIMED hom_mgsm" >> "$CLAIMS"
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
# See watch_and_launch_staircase.sh: backgrounded jobs launched inside a
# 'flock "$LOCK" bash -c ...' inherit the lock FD and hold it for their
# entire lifetime, starving every other watcher. Use an explicit named FD
# instead, closed in the backgrounded child via {LOCK_FD}<&-.
exec {LOCK_FD}<>"$LOCK"
while true; do
  flock -x "$LOCK_FD"
  RC=0
  try_claim_and_launch || RC=$?
  flock -u "$LOCK_FD"
  [ "$RC" -eq 0 ] && break
  sleep 300
done
