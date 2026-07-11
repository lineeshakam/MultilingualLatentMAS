#!/usr/bin/env bash
# Polls nvidia-smi until >=3 GPUs are simultaneously idle (<500MiB used), then
# reruns het_belebele_sg's single_agent_baseline mode with the fixed
# SafetyAgent (specialized_agents.py: anti-disclaimer prompt instruction +
# max_new_tokens 256->384, landed 20260707). That mode's cached result
# (.cache/checkpoints/bench_suite/het_belebele_sg/coordination/_results/
# ..._mode__single_agent_baseline.pt) was computed pre-fix and carries 82/200
# unparseable safety verdicts (mostly disclaimer-only responses and verdicts
# truncated by the old 256-token budget) -- see
# results/bench_suite/het_belebele_sg/safety_reparse_summary.json.
#
# 3 GPUs requested to match het_belebele_sg.yaml's per-role device assignment
# (translation/reasoning/safety each get their own cuda:N) -- single_agent_
# baseline's per-task executor is picked by the router (_pick_single_agent),
# not fixed to one role, so any of the 3 loaded agents' devices may be used.
#
# The stale cache is moved aside (not deleted) as *.stale-safety-parser-v2,
# following the existing *.stale-safety-bug convention already in that
# directory from the previous safety-parser fix.
#
# GPU claims are serialized via flock across ALL gpu-watching scripts so two
# watchers can never double-book the same idle GPU.
set -u
# Prevents a class of bug hit 2026-07-08: a __pycache__ .pyc compiled before
# a source edit can be treated as still-valid if the .py mtime ever moves
# backward (e.g. a git checkout/reset after the edit), silently running old
# bytecode in a fresh process despite the source file on disk being current.
# Cost is a full recompile per launch, negligible next to these jobs' runtime.
export PYTHONDONTWRITEBYTECODE=1
cd "$(dirname "$0")/.."
LOG=logs/bench_suite/safety_rerun_watcher.log
LOCK=/tmp/multilinguallatentmas_gpu_claim.lock
CACHE_DIR=.cache/checkpoints/bench_suite/het_belebele_sg/coordination/_results
CACHE_FILE="$CACHE_DIR/coord__aisingapore_Llama-SEA-LION-v3-8B-IT_sail_Sailor2-8B-Chat_meta-llama_Llama-3.1-8B-Instruct_CohereLabs_aya-expanse-8b__06c47cdd__mode__single_agent_baseline.pt"
mkdir -p logs/bench_suite
export LOG CACHE_DIR CACHE_FILE
log() { echo "[$(date -u +%FT%TZ)] [safety_rerun] $*" >> "$LOG"; }

try_claim_and_launch() (
  set -u
  N_NEEDED=3
  FREE_GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F',' '{gsub(/ /,"",$2); if ($2+0 < 500) print $1}')
  N_FREE=$(echo "$FREE_GPUS" | grep -c '[0-9]')
  [ "$N_FREE" -lt "$N_NEEDED" ] && return 1

  CLAIMED=$(echo "$FREE_GPUS" | head -"$N_NEEDED" | paste -sd, -)
  log "claimed gpus=$CLAIMED"

  if [ -f "$CACHE_FILE" ]; then
    mv "$CACHE_FILE" "$CACHE_FILE.stale-safety-parser-v2"
    log "moved stale cache -> $(basename "$CACHE_FILE").stale-safety-parser-v2"
  else
    log "no cached single_agent_baseline result found at $CACHE_FILE; proceeding anyway"
  fi
  [ -f "$CACHE_FILE.reparsed.json" ] && mv "$CACHE_FILE.reparsed.json" "$CACHE_FILE.reparsed.json.stale-safety-parser-v2"

  export CUDA_VISIBLE_DEVICES=$CLAIMED
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export PYTHONPATH=/home/hthakur/MultilingualLatentMAS/src

  # Consolidated with the previously-separate oneflow/latent_based_mas_ours
  # rerun (het_belebele_sg's driver gave up on both after a stale-bytecode
  # AttributeError crash -- see watch_and_launch_belebele_remaining_modes.sh)
  # into one process so the 4 agent models load once, not twice across two
  # separate 3-GPU jobs.
  log "launching het_belebele_sg --comm-modes single_agent_baseline,oneflow,latent_based_mas_ours on gpus=$CLAIMED"
  setsid nohup python scripts/run_coordination_pipeline.py \
    --config configs/bench_suite/het_belebele_sg.yaml --resume \
    --comm-modes single_agent_baseline,oneflow,latent_based_mas_ours \
    >> logs/bench_suite/het_belebele_sg.safety_rerun.log 2>&1 < /dev/null {LOCK_FD}<&- &
  DRIVER_PID=$!
  disown -a
  log "launched, driver pid=$DRIVER_PID -- verifying it survives startup (45s)"
  sleep 45
  if ! kill -0 "$DRIVER_PID" 2>/dev/null; then
    log "pid=$DRIVER_PID died within 45s -- likely OOM/crash; will retry. Last lines:"
    tail -15 logs/bench_suite/het_belebele_sg.safety_rerun.log >> "$LOG"
    return 1
  fi
  log "pid=$DRIVER_PID alive after 45s, launch looks healthy"

  ( while kill -0 "$DRIVER_PID" 2>/dev/null; do sleep 60; done
    log "driver pid=$DRIVER_PID done -- refreshing safety_reparse_summary.json"
    PYTHONPATH=/home/hthakur/MultilingualLatentMAS/src python scripts/recompute_safety_rate.py --config het_belebele_sg --force \
      >> logs/bench_suite/het_belebele_sg.safety_rerun.log 2>&1
    log "safety_reparse_summary.json refreshed" ) \
    >> "$LOG" 2>&1 < /dev/null {LOCK_FD}<&- &
  disown -a

  return 0
)

log "watching for >=3 idle GPUs (<500MiB used)"
# Bug fixed 2026-07-08: 'flock "$LOCK" bash -c "...try_claim_and_launch"' let
# both backgrounded children above inherit the lock FD across the exec
# chain; since neither closed it, the advisory lock stayed held for their
# entire lifetime (the multi-day eval run, then the summary-refresh waiter),
# starving geo_profiles/staircase/hom_remaining_modes regardless of real GPU
# availability -- confirmed live 2026-07-08 (see fuser output showing this
# script's launched driver PID still holding the lock FD 3+ hours in). Fixed
# by taking the lock on an explicit named FD in this shell instead of
# letting 'flock' exec a throwaway bash whose descriptors leak into whatever
# it launches; {LOCK_FD}<&- above closes it in each child specifically.
exec {LOCK_FD}<>"$LOCK"
while true; do
  flock -x "$LOCK_FD"
  RC=0
  try_claim_and_launch || RC=$?
  flock -u "$LOCK_FD"
  [ "$RC" -eq 0 ] && break
  sleep 300
done
