#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/python3.11.0/bin/python3}"
DEVICES="${DEVICES:-0}"
MASTER_PORT="${MASTER_PORT:-39177}"
NUM_SAMPLES="${NUM_SAMPLES:-50000}"
TARGET_TAG="${TARGET_TAG:-global_step1400000}"
OUT_DIR="${OUT_DIR:-$ROOT/eval_outputs/step1400000}"

export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
export ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-$ASCEND_HOME_PATH}"
export ASCEND_OPP_PATH="${ASCEND_OPP_PATH:-$ASCEND_HOME_PATH/opp}"
export ASCEND_AICPU_PATH="${ASCEND_AICPU_PATH:-$ASCEND_HOME_PATH}"
export PATH="$ASCEND_HOME_PATH/bin:/usr/local/python3.11.0/bin:$PATH"
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/lib64:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/python3.11.0/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ASCEND_HOME_PATH/python/site-packages:$ASCEND_HOME_PATH/opp/built-in/op_impl/ai_core/tbe:$ROOT:$ROOT/model:${PYTHONPATH:-}"
export HCCL_WHITELIST_DISABLE=1
export HCCL_CONNECT_TIMEOUT=3600

mkdir -p "$ROOT/logs" "$OUT_DIR"
"$PYTHON_BIN" -m deepspeed.launcher.runner \
  --include "localhost:${DEVICES}" \
  --master_port "$MASTER_PORT" \
  "$ROOT/train_for_eval.py" \
  --eval_only \
  --output_dir "$ROOT/checkpoints/imagenet64_checkpoints_v90_21" \
  --target_tag "$TARGET_TAG" \
  --num_samples "$NUM_SAMPLES" \
  --eval_output_dir "$OUT_DIR" \
  > "$ROOT/logs/generate_${TARGET_TAG}_${NUM_SAMPLES}.log" 2>&1
