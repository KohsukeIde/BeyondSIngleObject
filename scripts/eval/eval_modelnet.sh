#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/modelnet_classification/$(date +%Y%m%d_%H%M%S)}"
OUTPUT_JSON="${OUTPUT_JSON:-${OUTPUT_DIR}/modelnet_inference.json}"
CLIP_OUTPUT_JSON="${CLIP_OUTPUT_JSON:-${OUTPUT_DIR}/clip_metrics.json}"
DATA_PATH="${DATA_PATH:-data/modelnet40_data/modelnet40_test_8192pts_fps.dat}"
CATEGORIES="${CATEGORIES:-configs/eval/modelnet40_shape_names_modified.txt}"
LIMIT="${LIMIT:-0}"
SAMPLING="${SAMPLING:-balanced}"
SEED="${SEED:-42}"
PROMPT_MODE="${PROMPT_MODE:-paper}"
NUM_OBJECTS="${NUM_OBJECTS:-1}"
TARGET_POSITION="${TARGET_POSITION:-1}"
RUN_CLIP="${RUN_CLIP:-1}"
CLIP_MODEL="${CLIP_MODEL:-openai/clip-vit-large-patch14}"

INFER_CMD=(
  python -m pointllm.eval.cvpr.eval_modelnet_classification
  --model_path "${MODEL_PATH}"
  --data_path "${DATA_PATH}"
  --categories "${CATEGORIES}"
  --output "${OUTPUT_JSON}"
  --limit "${LIMIT}"
  --sampling "${SAMPLING}"
  --seed "${SEED}"
  --prompt_mode "${PROMPT_MODE}"
  --num_objects "${NUM_OBJECTS}"
  --target_position "${TARGET_POSITION}"
)

if [[ "${RUN_CLIP}" == "1" ]]; then
  CLIP_CMD=(
    python tools/evaluate_clip_classification.py
    "${OUTPUT_JSON}"
    --categories "${CATEGORIES}"
    --output "${CLIP_OUTPUT_JSON}"
    --clip_model "${CLIP_MODEL}"
  )
else
  CLIP_CMD=()
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${INFER_CMD[@]}"
  printf '\n'
  if [[ "${#CLIP_CMD[@]}" -gt 0 ]]; then
    printf '%q ' "${CLIP_CMD[@]}"
    printf '\n'
  fi
  exit 0
fi

mkdir -p "${OUTPUT_DIR}"
"${INFER_CMD[@]}"
if [[ "${#CLIP_CMD[@]}" -gt 0 ]]; then
  "${CLIP_CMD[@]}"
fi
