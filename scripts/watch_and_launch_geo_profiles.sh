#!/usr/bin/env bash
# The first attempt at export_geo_profiles.py (from the GPU7 chain watcher)
# crashed in 3s: it passed 'en' as a target language, but 'en' is the
# anchor/pivot language, not a valid target (fixed in
# watch_and_launch_gpu7_chain.sh for future chains). This retries the
# corrected command once a full GPU is free (needs ~8-9GB for the 8-bit
# model; current per-GPU headroom is ~6-7GB, not enough to share).
set -u
# Prevents a class of bug hit 2026-07-08: a __pycache__ .pyc compiled before
# a source edit can be treated as still-valid if the .py mtime ever moves
# backward (e.g. a git checkout/reset after the edit), silently running old
# bytecode in a fresh process despite the source file on disk being current.
# Cost is a full recompile per launch, negligible next to these jobs' runtime.
export PYTHONDONTWRITEBYTECODE=1
cd "$(dirname "$0")/.."
LOG=logs/baselines/geo_profiles_watcher.log
LOCK=/tmp/multilinguallatentmas_gpu_claim.lock
mkdir -p logs/baselines
export LOG
log() { echo "[$(date -u +%FT%TZ)] [geo_profiles] $*" >> "$LOG"; }

try_claim_and_launch() (
  set -u
  FREE_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F',' '{gsub(/ /,"",$2); if ($2+0 < 500) {print $1; exit}}')
  [ -z "$FREE_GPU" ] && return 1

  log "claimed gpu=$FREE_GPU"
  # --load-in-8bit was missing on both prior attempts: SEA-LION-8B in fp16
  # needs ~16GB, which doesn't reliably fit on a 16GB GPU even fully idle
  # (zero headroom for activations) -- that's the real reason both attempts
  # OOM'd, not just GPU contention.
  CUDA_VISIBLE_DEVICES=$FREE_GPU PYTHONPATH=src python scripts/export_geo_profiles.py \
    --model aisingapore/Llama-SEA-LION-v3-8B-IT \
    --languages th,my,km,lo,am,sw,bn,te \
    --n-samples 64 \
    --load-in-8bit \
    --output results/mechanistic/geo_profiles.json \
    >> logs/baselines/export_geo_profiles.log 2>&1
  RC=$?
  log "export_geo_profiles.py exit=$RC"
  # This runs synchronously (not backgrounded), so the exit code is known
  # immediately -- but it was previously discarded, always returning 0. That
  # made the outer while-loop break (stop retrying) even on a crash, which is
  # exactly what silently ate both prior OOM failures. Propagate it instead
  # so a real failure triggers a retry with the standard 300s backoff.
  return $RC
)

log "watching for 1 idle GPU (<500MiB used)"
# Bug fixed 2026-07-08: 'flock "$LOCK" bash -c "...try_claim_and_launch"' let
# this script's lock FD leak into whatever it exec'd; harmless here (the
# export runs synchronously, no backgrounded child to inherit it), but
# switched to the same explicit-FD pattern as the other watchers for
# consistency and so a future edit adding a backgrounded step doesn't
# silently reintroduce the cross-script starvation bug (see
# watch_and_launch_staircase.sh for the full story).
exec {LOCK_FD}<>"$LOCK"
while true; do
  flock -x "$LOCK_FD"
  RC=0
  try_claim_and_launch || RC=$?
  flock -u "$LOCK_FD"
  [ "$RC" -eq 0 ] && break
  sleep 300
done
