#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-checkpoints/multi-3dllm}"
ANNO_PATH="${ANNO_PATH:-data/mo3d/test.json}"
DATA_PATH="${DATA_PATH:-data/point_clouds}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/eval}"
RELATION_MODE="${RELATION_MODE:-patch}"
OUTPUT_NAME="${OUTPUT_NAME:-inference.json}"
LIMIT="${LIMIT:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-0.9}"
TOP_K="${TOP_K:-40}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
NO_REPEAT_NGRAM_SIZE="${NO_REPEAT_NGRAM_SIZE:-0}"
PYTHONPATH=".:${PYTHONPATH:-}"

CMD=(
  python -m pointllm.eval.cvpr.eval_cvpr_patch
  --model_path "${MODEL_PATH}"
  --anno_path "${ANNO_PATH}"
  --data_path "${DATA_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --output_name "${OUTPUT_NAME}"
  --relation_mode "${RELATION_MODE}"
  --limit "${LIMIT}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --temperature "${TEMPERATURE}"
  --top_p "${TOP_P}"
  --top_k "${TOP_K}"
  --repetition_penalty "${REPETITION_PENALTY}"
  --no_repeat_ngram_size "${NO_REPEAT_NGRAM_SIZE}"
)

if [[ "${SELECT_ONE_MODE:-0}" == "1" ]]; then
  CMD+=(--select_one_mode)
fi

if [[ "${MULTI_TURN:-0}" == "1" ]]; then
  CMD+=(--multi_turn)
fi

if [[ "${SCORE_VERIFY_OPTIONS:-0}" == "1" ]]; then
  CMD+=(--score_verify_options)
fi

if [[ "${DEDUPE_DELTA_OUTPUT:-0}" == "1" ]]; then
  CMD+=(--dedupe_delta_output)
fi

if [[ -n "${MAX_DELTA_OUTPUT_CLAUSES:-}" ]]; then
  CMD+=(--max_delta_output_clauses "${MAX_DELTA_OUTPUT_CLAUSES}")
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'PYTHONPATH=%q ' "${PYTHONPATH}"
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

export PYTHONPATH
"${CMD[@]}"
