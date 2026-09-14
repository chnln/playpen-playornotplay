#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 ROOT_DIR MODEL SUITE GAME RESULTS_DIR" >&2
  exit 1
fi

ROOT_DIR="$1"
MODEL_NAME="$2"
SUITE_NAME="$3"
GAME_NAME="$4"
RESULTS_DIR="$5"
PLAYPEN_DIR="$ROOT_DIR/environment/playpen"
PLAYPEN_VENV_DIR="${PLAYPEN_VENV_DIR:-$PLAYPEN_DIR/venv_repro}"
PYTHON_BIN="$PLAYPEN_VENV_DIR/bin/python"
PLAYPEN_BIN="$PLAYPEN_VENV_DIR/bin/playpen"
NLTK_DATA_DIR="$PLAYPEN_DIR/nltk_data"
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
  --results-dir "$RESULTS_DIR")"

mkdir -p "$JOB_DIR" "$RESULTS_DIR"
cd "$JOB_DIR"

"$PYTHON_BIN" "$RUN_METADATA_HELPER" \
  --root-dir "$ROOT_DIR" \
  --target-dir "$RESULTS_DIR" \
  --mode eval \
  --model-name "$MODEL_NAME" \
  --suite-name "$SUITE_NAME" \
  --game-name "$GAME_NAME" \
  --results-dir "$RESULTS_DIR"

REGISTRY_ARGS=(--output "$MODEL_REGISTRY_PATH")
if [[ -n "${PLAYPEN_QWEN_PEFT_MODEL_NAME:-}" || -n "${PLAYPEN_QWEN_PEFT_ADAPTER_PATH:-}" ]]; then
  REGISTRY_ARGS+=(--qwen-peft-model-name "${PLAYPEN_QWEN_PEFT_MODEL_NAME:-}" --qwen-peft-adapter-path "${PLAYPEN_QWEN_PEFT_ADAPTER_PATH:-}")
fi
if [[ -n "${PLAYPEN_QWEN_BASE_MODEL:-}" ]]; then
  REGISTRY_ARGS+=(--qwen-base-model "${PLAYPEN_QWEN_BASE_MODEL}")
fi

if [[ -n "${PLAYPEN_QWEN_VANILLA_MODEL_NAME:-}" || -n "${PLAYPEN_QWEN_VANILLA_HF_ID:-}" ]]; then
  REGISTRY_ARGS+=(--qwen-vanilla-model-name "${PLAYPEN_QWEN_VANILLA_MODEL_NAME:-}" --qwen-vanilla-hf-id "${PLAYPEN_QWEN_VANILLA_HF_ID:-}")
fi

"$PYTHON_BIN" "$ROOT_DIR/scripts/setup_playpen_registry.py" "${REGISTRY_ARGS[@]}"
"$PYTHON_BIN" "$ROOT_DIR/scripts/check_torch_cuda.py" --require-gpu
"$PYTHON_BIN" "$ROOT_DIR/scripts/ensure_nltk_data.py" --download-dir "$NLTK_DATA_DIR" stopwords wordnet omw-1.4

export CLEMBENCH_HOME="$PLAYPEN_DIR"
export NLTK_DATA="$NLTK_DATA_DIR"

CMD=("$PLAYPEN_BIN" eval "$MODEL_NAME" --suite "$SUITE_NAME" --results_dir "$RESULTS_DIR" -T "${PLAYPEN_EVAL_TEMPERATURE:-0}" -L "${PLAYPEN_EVAL_MAX_TOKENS:-300}")
if [[ -n "$GAME_NAME" ]]; then
  CMD+=(-g "$GAME_NAME")
fi

printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
