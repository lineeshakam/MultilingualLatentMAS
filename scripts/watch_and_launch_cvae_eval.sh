#!/usr/bin/env bash
# Launches a CVAE-topology production eval config (het_mgsm_cvae,
# hom_mgsm_cvae, het_belebele_sg_cvae, hom_belebele_sg_cvae) once
# results/mechanistic/geo_profiles.json exists (required by
# cvae.condition_on_geometry=true, see each config's header comment) and
# enough GPUs are simultaneously idle. Not started automatically -- these
# configs deliberately queue behind the ablation staircase's rows 3-6, which
# exercise the same cvae_topology + condition_on_geometry module combination
# more cheaply (no separate Stage-B retrain per production config); revisit
# launching this once staircase results land. Launch manually with e.g.:
#   setsid nohup bash scripts/watch_and_launch_cvae_eval.sh het_mgsm_cvae \
#     >> logs/bench_suite/het_mgsm_cvae_watcher.log 2>&1 &
#
# $1: config basename under configs/bench_suite/ (without .yaml), e.g.
#     het_mgsm_cvae | hom_mgsm_cvae | het_belebele_sg_cvae | hom_belebele_sg_cvae
set -u
# Prevents a class of bug hit 2026-07-08: a __pycache__ .pyc compiled before
# a source edit can be treated as still-valid if the .py mtime ever moves
# backward (e.g. a git checkout/reset after the edit), silently running old
# bytecode in a fresh process despite the source file on disk being current.
# Cost is a full recompile per launch, negligible next to these jobs' runtime.
export PYTHONDONTWRITEBYTECODE=1
CONFIG_NAME=$1
CONFIG="configs/bench_suite/${CONFIG_NAME}.yaml"
cd "$(dirname "$0")/.."
if [ ! -f "$CONFIG" ]; then
  echo "no such config: $CONFIG" >&2
  exit 1
fi
LOG="logs/bench_suite/${CONFIG_NAME}_watcher.log"
LOCK=/tmp/multilinguallatentmas_gpu_claim.lock
GEO_PROFILES=results/mechanistic/geo_profiles.json
mkdir -p logs/bench_suite
export LOG CONFIG CONFIG_NAME
log() { echo "[$(date -u +%FT%TZ)] [${CONFIG_NAME}] $*" >> "$LOG"; }

try_claim_and_launch() (
  set -u
  # All four *_cvae.yaml configs use the same 3-GPU footprint as their
  # production counterparts (orchestrator+translation on cuda:0, reasoning
  # on cuda:1, safety on cuda:2).
  N_NEEDED=3
  if [ ! -f "$GEO_PROFILES" ]; then
    log "waiting: $GEO_PROFILES does not exist yet (required by condition_on_geometry=true)"
    return 1
  fi

  FREE_GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F',' '{gsub(/ /,"",$2); if ($2+0 < 500) print $1}')
  N_FREE=$(echo "$FREE_GPUS" | grep -c '[0-9]')
  [ "$N_FREE" -lt "$N_NEEDED" ] && return 1

  CLAIMED=$(echo "$FREE_GPUS" | head -"$N_NEEDED" | paste -sd, -)
  log "claimed gpus=$CLAIMED"

  export CUDA_VISIBLE_DEVICES=$CLAIMED
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export PYTHONPATH=src

  JOBLOG="logs/bench_suite/${CONFIG_NAME}.log"
  # First launch: no checkpoint exists yet at this config's own (new)
  # checkpoint_dir, so default --stages (all) runs Stage B fresh under
  # condition_on_geometry=true. --resume is safe to keep for idempotent
  # re-launches after a crash (see each *_cvae.yaml's header comment for why
  # it must NOT point at the attention-router checkpoint dir).
  log "launching $CONFIG on gpus=$CLAIMED"
  setsid nohup python scripts/run_coordination_pipeline.py \
    --config "$CONFIG" --resume \
    >> "$JOBLOG" 2>&1 < /dev/null {LOCK_FD}<&- &
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

log "watching for >=3 idle GPUs (<500MiB used) + $GEO_PROFILES"
# Bug fixed 2026-07-08: see watch_and_launch_staircase.sh -- backgrounded
# jobs launched inside a 'flock "$LOCK" bash -c ...' inherited the lock FD
# and held it for their entire lifetime, starving every other watcher.
exec {LOCK_FD}<>"$LOCK"
while true; do
  flock -x "$LOCK_FD"
  RC=0
  try_claim_and_launch || RC=$?
  flock -u "$LOCK_FD"
  [ "$RC" -eq 0 ] && break
  sleep 300
done
