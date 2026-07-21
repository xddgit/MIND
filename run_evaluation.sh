#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLE_DIR="${SAMPLE_DIR:?Set SAMPLE_DIR to the generated image directory.}"
MAX_IMAGES="${MAX_IMAGES:-50000}"
EVAL_BACKEND="${EVAL_BACKEND:-torch}"
EVAL_DEVICE="${EVAL_DEVICE:-auto}"
EVAL_GPU="${EVAL_GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export PYTHONPATH="$ROOT:$ROOT/model:${PYTHONPATH:-}"
"$PYTHON_BIN" "$ROOT/I_evaluator_grid.py" \
  --root_dir "$ROOT/eval_outputs" \
  --folders "$SAMPLE_DIR" \
  --ref_npz "$ROOT/VIRTUAL_imagenet256_labeled.npz" \
  --max_images "$MAX_IMAGES" \
  --backend "$EVAL_BACKEND" \
  --device "$EVAL_DEVICE" \
  --gpu "$EVAL_GPU"
