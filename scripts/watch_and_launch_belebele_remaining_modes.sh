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
set -u
# Prevents a class of bug hit 2026-07-08: a __pycache__ .pyc compiled before
# a source edit can be treated as still-valid if the .py mtime ever moves
# backward (e.g. a git checkout/reset after the edit), silently running old
# bytecode in a fresh process despite the source file on disk being current.
# Cost is a full recompile per launch, negligible next to these jobs' runtime.
export PYTHONDONTWRITEBYTECODE=1
INSTANCE=$1
CONFIG="configs/bench_suite/${INSTANCE}_belebele_sg.yaml"
cd "$(dirname "$0")/.."
LOG="logs/bench_suite/${INSTANCE}_remaining_modes_watcher.log"
LOCK=/tmp/multilinguallatentmas_gpu_claim.lock
mkdir -p logs/bench_suite
export LOG CONFIG INSTANCE
log() { echo "[$(date -u +%FT%TZ)] [${INSTANCE}_remaining] $*" >> "$LOG"; }

try_claim_and_launch() (
  set -u
  N_NEEDED=3
  FREE_GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F',' '{gsub(/ /,"",$2); if ($2+0 < 500) print $1}')
  N_FREE=$(echo "$FREE_GPUS" | grep -c '[0-9]')
  [ "$N_FREE" -lt "$N_NEEDED" ] && return 1

  CLAIMED=$(echo "$FREE_GPUS" | head -"$N_NEEDED" | paste -sd, -)
  log "claimed gpus=$CLAIMED"

  export CUDA_VISIBLE_DEVICES=$CLAIMED
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export PYTHONPATH=src

  JOBLOG="logs/bench_suite/${INSTANCE}_belebele_sg.remaining_modes.log"
  log "launching $CONFIG --comm-modes oneflow,latent_based_mas_ours on gpus=$CLAIMED"
  setsid nohup python scripts/run_coordination_pipeline.py \
    --config "$CONFIG" --resume \
    --comm-modes oneflow,latent_based_mas_ours \
    >> "$JOBLOG" 2>&1 < /dev/null &
  JOB_PID=$!
  disown -a
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
while true; do
  if flock "$LOCK" bash -c "$(declare -f try_claim_and_launch log); try_claim_and_launch"; then
    break
  fi
  sleep 300
done
