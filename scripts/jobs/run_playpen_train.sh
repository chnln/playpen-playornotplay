#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 ROOT_DIR TRAINER_FILE LEARNER_NAME OUTPUT_DIR" >&2
  exit 1
fi

ROOT_DIR="$1"
TRAINER_FILE="$2"
LEARNER_NAME="$3"
OUTPUT_DIR="$4"

PLAYPEN_DIR="$ROOT_DIR/environment/playpen"
PLAYPEN_VENV_DIR="${PLAYPEN_VENV_DIR:-$PLAYPEN_DIR/venv_repro}"
PYTHON_BIN="$PLAYPEN_VENV_DIR/bin/python"
PLAYPEN_BIN="$PLAYPEN_VENV_DIR/bin/playpen"
JOB_CONTEXT_HELPER="$ROOT_DIR/scripts/resolve_playpen_job_context.py"
RUN_METADATA_HELPER="$ROOT_DIR/scripts/write_playpen_run_metadata.py"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python interpreter at $PYTHON_BIN" >&2
  exit 1
fi

eval "$("$PYTHON_BIN" "$JOB_CONTEXT_HELPER" \
  --root-dir "$ROOT_DIR" \
  --playpen-dir "$PLAYPEN_DIR" \
  --job-id "${SLURM_JOB_ID:-}" \
  --trainer-file "$TRAINER_FILE" \
  --output-dir "$OUTPUT_DIR")"

mkdir -p "$JOB_DIR" "$OUTPUT_DIR"
cd "$JOB_DIR"

export CLEMBENCH_HOME="$PLAYPEN_DIR"
export PLAYPEN_OUTPUT_DIR="$OUTPUT_DIR"

"$PYTHON_BIN" "$RUN_METADATA_HELPER" \
  --root-dir "$ROOT_DIR" \
  --target-dir "$OUTPUT_DIR" \
  --mode train \
  --trainer-file "$TRAINER_FILE" \
  --learner-name "$LEARNER_NAME" \
  --output-dir "$OUTPUT_DIR"

REGISTRY_ARGS=(--output "$MODEL_REGISTRY_PATH")
if [[ -n "${PLAYPEN_QWEN_PEFT_MODEL_NAME:-}" || -n "${PLAYPEN_QWEN_PEFT_ADAPTER_PATH:-}" ]]; then
  REGISTRY_ARGS+=(--qwen-peft-model-name "${PLAYPEN_QWEN_PEFT_MODEL_NAME:-}" --qwen-peft-adapter-path "${PLAYPEN_QWEN_PEFT_ADAPTER_PATH:-}")
fi
if [[ -n "${PLAYPEN_QWEN_BASE_MODEL:-}" ]]; then
  REGISTRY_ARGS+=(--qwen-base-model "${PLAYPEN_QWEN_BASE_MODEL}")
fi

"$PYTHON_BIN" "$ROOT_DIR/scripts/setup_playpen_registry.py" "${REGISTRY_ARGS[@]}"
"$PYTHON_BIN" "$ROOT_DIR/scripts/check_torch_cuda.py" --require-gpu

CMD=("$PLAYPEN_BIN" run "$TRAINER_FILE" -l "$LEARNER_NAME")

printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '\n'
env | egrep '^PLAYPEN_' | sort || true
"${CMD[@]}"
