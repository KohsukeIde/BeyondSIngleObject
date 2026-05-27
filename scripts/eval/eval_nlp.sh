#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: TASK=mo3d|shape_mating|change_captioning $0 /path/to/inference.json" >&2
  exit 2
fi

INFERENCE_JSON="$1"
TASK="${TASK:-auto}"
OUTPUT_FILE="${OUTPUT_FILE:-${INFERENCE_JSON%.json}_text_metrics.json}"
ANNO_PATH="${ANNO_PATH:-}"
INCLUDE_VERIFY="${INCLUDE_VERIFY:-0}"
INCLUDE_ANSWER_TURN="${INCLUDE_ANSWER_TURN:-0}"

CMD=(
  python tools/evaluate_text_metrics.py
  "${INFERENCE_JSON}"
  --task "${TASK}"
  --output "${OUTPUT_FILE}"
)

if [[ -n "${ANNO_PATH}" ]]; then
  CMD+=(--annotation_json "${ANNO_PATH}")
fi

if [[ "${INCLUDE_VERIFY}" == "1" ]]; then
  CMD+=(--include_verify)
fi

if [[ "${INCLUDE_ANSWER_TURN}" == "1" ]]; then
  CMD+=(--include_answer_turn)
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

"${CMD[@]}"
