#!/usr/bin/env bash
# Rerun het_mgsm's single_agent_baseline mode with the fixed lead-agent
# selection (_pick_single_agent, 2026-07-11): the cached mode result from
# 2026-07-09 routed agent_trans on 2076/2200 mgsm tasks and scored 0.0436 —
# a routing artifact, not a baseline (see README_single_agent_INVALID.md next
# to the cache). The invalid .pt cannot be moved while the token/latent
# requeue (PID file row "het-router-fix-reservation") still runs, so this
# watcher:
#   1. waits until no run_coordination_pipeline process is using
#      configs/bench_suite/het_mgsm.yaml,
#   2. quarantines the single_agent cache (full + partial) so --resume redoes
#      the mode from scratch on fixed code,
#   3. claims 3 free GPUs (claims-file aware) and relaunches
#      --comm-modes single_agent_baseline.
# Priority courtesy: while the hom relaunch watchers are still pending, only
# claim when >=6 GPUs are free so a 3-set is always left for them
# (hom > this rerun, same convention as watch_and_launch_mrre_crossbb_batch_v2.sh).
# 2026-07-11 (later session) SCOPE WIDENED to all three modes: the Jul 4-11
# driver runs resolved `shared` to LRL-MRRE-MAS's older package (editable
# .pth sys.path injection — see memory chimera-shared-package-root-cause), so
# the token_based_mas cache is provenance-tainted too and latent never ran at
# all (the requeue probe's empty output was misread as "already fixed"). This
# watcher now quarantines single_agent AND token caches and reruns
# single_agent_baseline,token_based_mas,latent_based_mas_ours.
set -u
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
# Absolute PYTHONPATH so `import shared` can never bind to the LRL repo copy.
export PYTHONPATH=/home/hthakur/MultilingualLatentMAS/src
LOG=logs/bench_suite/het_mgsm_single_agent_rerun_watcher.log
LOCK=/tmp/multilinguallatentmas_gpu_claim.lock
CLAIMS=/tmp/multilinguallatentmas_gpu_claims
CACHE_DIR=.cache/checkpoints/bench_suite/het_mgsm/coordination/_results
mkdir -p logs/bench_suite
touch "$CLAIMS"
export LOG
log() { echo "[$(date -u +%FT%TZ)] [het_mgsm_sa_rerun] $*" >> "$LOG"; }

# Gate on BOTH the het_mgsm pipeline process and the router-fix requeue
# watcher: the requeue retries `--resume` (all modes) on crash, so if we
# quarantined the single_agent cache while it can still relaunch, two
# processes would redo the mode concurrently and race on the same cache key.
het_mgsm_running() {
  pgrep -f "run_coordination_pipeline.*het_mgsm" >/dev/null 2>&1 \
    || pgrep -f "requeue_router_fix.sh het" >/dev/null 2>&1
}
hom_pending() {
  pgrep -f watch_and_launch_hom_mgsm.sh >/dev/null 2>&1 \
    || pgrep -f "watch_and_launch_belebele_remaining_modes.sh hom" >/dev/null 2>&1
}

quarantine_cache() {
  local qdir="$CACHE_DIR/quarantine_routing_artifact"
  mkdir -p "$qdir"
  local moved=0
  local f
  # single_agent: routing artifact (_pick_single_agent); token: chimera-shared
  # provenance (old LRL model_loader in-process for the whole 2-day run).
  for f in "$CACHE_DIR"/*__mode__single_agent_baseline.pt \
           "$CACHE_DIR"/*__mode__single_agent_baseline__partial.pt \
           "$CACHE_DIR"/*__mode__token_based_mas.pt \
           "$CACHE_DIR"/*__mode__token_based_mas__partial.pt; do
    [ -e "$f" ] || continue
    mv "$f" "$qdir/" && moved=1 && log "quarantined $(basename "$f")"
  done
  [ "$moved" -eq 1 ] || log "no tainted cache left to quarantine (already moved?)"
}

try_claim_and_launch() (
  set -u
  N_NEEDED=3
  FREE_GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F',' '{gsub(/ /,"",$2); if ($2+0 < 500) print $1}')
  LIVE_CLAIMED=$(while read -r _ts _pid _gpus _label; do
      [ -n "${_pid:-}" ] || continue
      kill -0 "$_pid" 2>/dev/null && echo "$_gpus" | tr ',' '\n'
    done < "$CLAIMS" | sort -u)
  if [ -n "$LIVE_CLAIMED" ]; then
    FREE_GPUS=$(echo "$FREE_GPUS" | grep -vxF "$LIVE_CLAIMED" || true)
  fi
  N_FREE=$(echo "$FREE_GPUS" | grep -c '[0-9]')
  [ "$N_FREE" -lt "$N_NEEDED" ] && return 1
  if hom_pending && [ "$N_FREE" -lt 6 ]; then
    return 1   # leave a whole 3-slice to the paper-blocking hom relaunches
  fi

  quarantine_cache

  CLAIMED=$(echo "$FREE_GPUS" | head -"$N_NEEDED" | paste -sd, -)
  log "claimed gpus=$CLAIMED"
  export CUDA_VISIBLE_DEVICES=$CLAIMED
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

  JOBLOG=logs/bench_suite/het_mgsm.single_agent_rerun.log
  log "launching het_mgsm --resume --comm-modes single_agent_baseline,token_based_mas,latent_based_mas_ours on gpus=$CLAIMED"
  setsid nohup python scripts/run_coordination_pipeline.py \
    --config configs/bench_suite/het_mgsm.yaml --resume \
    --comm-modes single_agent_baseline,token_based_mas,latent_based_mas_ours \
    >> "$JOBLOG" 2>&1 < /dev/null {LOCK_FD}<&- &
  JOB_PID=$!
  disown -a
  echo "$(date +%s) $JOB_PID $CLAIMED het_mgsm_single_agent_rerun" >> "$CLAIMS"
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

log "waiting for the het_mgsm token/latent requeue to finish, then >=3 idle GPUs"
exec {LOCK_FD}<>"$LOCK"
while true; do
  if het_mgsm_running; then
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
log "rerun launched successfully; watcher exiting"
