#!/usr/bin/env bash
set -euo pipefail

# Run TextMAS on MGSM for all languages (one sample per language).
# Usage: ./scripts/run_mgsm_text_mas.sh [MODEL] [METHOD] [PROMPT] [DEVICE]
# Example: ./scripts/run_mgsm_text_mas.sh Qwen/Qwen3-4B text_mas sequential mps

langs=(bn de en es fr ja ru sw te th zh)
MODEL=${1:-Qwen/Qwen3-4B}
METHOD=${2:-text_mas}
PROMPT=${3:-sequential}
DEVICE=${4:-mps}

for L in "${langs[@]}"; do
  echo "=== $L ==="
  python run.py --method "$METHOD" --model_name "$MODEL" \
    --task mgsm --mgsm_lang "$L" --max_samples 1 --generate_bs 1 \
    --device "$DEVICE" --max_new_tokens 512 --prompt "$PROMPT"
done
