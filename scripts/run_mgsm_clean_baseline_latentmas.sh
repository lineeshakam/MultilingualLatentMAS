#!/usr/bin/env bash
set -euo pipefail

# Clean MGSM comparison after padding/decode fixes:
#   1. single-agent baseline
#   2. LatentMAS with latent_space_realign
#
# Set PROMPT_LANGUAGE_MODE=neutral to avoid forcing any reasoning language.
# Other options:
#   target  = current multilingual prompts/directives
#   english = English-control prompts/directives

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B}"
LANGUAGES="${LANGUAGES:-bn,de,en,es,fr,ja,ru,sw,te,th,zh}"
PROMPT_LANGUAGE_MODE="${PROMPT_LANGUAGE_MODE:-neutral}"
MAX_EXAMPLES="${MAX_EXAMPLES:-50}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
DEVICE="${DEVICE:-auto}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-1.0}"
SEED="${SEED:-42}"
LATENT_STEPS="${LATENT_STEPS:-3}"
GENERATE_BS="${GENERATE_BS:-1}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_LATENT_MAS="${RUN_LATENT_MAS:-1}"

export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "================ Clean MGSM comparison ================"
echo "  model       : ${MODEL_NAME}"
echo "  languages   : ${LANGUAGES}"
echo "  prompt lang : ${PROMPT_LANGUAGE_MODE}"
echo "  max examples: ${MAX_EXAMPLES}"
echo "  max tokens  : ${MAX_NEW_TOKENS}"
echo "  temperature : ${TEMPERATURE}"
echo "  device      : ${DEVICE}"
echo "  latent steps: ${LATENT_STEPS}"
echo "  baseline bs : ${GENERATE_BS}"
echo "======================================================="

if [[ "${RUN_BASELINE}" == "1" || "${RUN_BASELINE}" == "true" ]]; then
  MODEL="${MODEL_NAME}" \
  LANGUAGES="${LANGUAGES}" \
  DEVICE="${DEVICE}" \
  PROMPT_LANGUAGE_MODE="${PROMPT_LANGUAGE_MODE}" \
  MAX_SAMPLES="${MAX_EXAMPLES}" \
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
  GENERATE_BS="${GENERATE_BS}" \
  TEMPERATURE="${TEMPERATURE}" \
  TOP_P="${TOP_P}" \
  SEED="${SEED}" \
  RUN_NAME="mgsm_baseline_${PROMPT_LANGUAGE_MODE}_clean_${MAX_EXAMPLES}" \
  bash scripts/run_mgsm_baseline_all.sh
fi

if [[ "${RUN_LATENT_MAS}" == "1" || "${RUN_LATENT_MAS}" == "true" ]]; then
  MODEL_NAME="${MODEL_NAME}" \
  LANGUAGES="${LANGUAGES}" \
  DEVICE="${DEVICE}" \
  PROMPT="sequential" \
  PROMPT_LANGUAGE_MODE="${PROMPT_LANGUAGE_MODE}" \
  LATENT_STEPS="${LATENT_STEPS}" \
  LATENT_SPACE_REALIGN=1 \
  MAX_EXAMPLES="${MAX_EXAMPLES}" \
  MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
  TEMPERATURE="${TEMPERATURE}" \
  TOP_P="${TOP_P}" \
  SEED="${SEED}" \
  CHECKPOINT_EVERY="${CHECKPOINT_EVERY}" \
  RUN_NAME="mgsm_latent_mas_${PROMPT_LANGUAGE_MODE}_realign_clean_${MAX_EXAMPLES}" \
  bash scripts/run_mgsm_latent_mas_full_csv.sh
fi
