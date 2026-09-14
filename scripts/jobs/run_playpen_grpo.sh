#!/usr/bin/env bash
# Historical exploration entrypoint; not the submitted effective recipe.
# Use scripts/submit_recipe_phase.py for phases A/B1/B2/C.
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 ROOT_DIR TRAINER_FILE LEARNER_NAME OUTPUT_DIR GAME_NAME" >&2
  exit 1
fi

ROOT_DIR="$1"
TRAINER_FILE="$2"
LEARNER_NAME="$3"
OUTPUT_DIR="$4"
GAME_NAME="$5"

PLAYPEN_DIR="$ROOT_DIR/environment/playpen"
PLAYPEN_VENV_DIR="${PLAYPEN_VENV_DIR:-$PLAYPEN_DIR/venv}"
PYTHON_BIN="$PLAYPEN_VENV_DIR/bin/python"
PLAYPEN_BIN="$PLAYPEN_VENV_DIR/bin/playpen"
CLEM_BIN="$PLAYPEN_VENV_DIR/bin/clem"
JOB_CONTEXT_HELPER="$ROOT_DIR/scripts/resolve_playpen_job_context.py"
RUN_METADATA_HELPER="$ROOT_DIR/scripts/write_playpen_run_metadata.py"
PORT="${PLAYPEN_OPENENV_PORT:-9101}"
HOST="${PLAYPEN_OPENENV_HOST:-127.0.0.1}"
RUN_ID="${PLAYPEN_OPENENV_RUN_ID:-$(basename "$OUTPUT_DIR")}"
SERVER_RESULTS_DIR="${PLAYPEN_OPENENV_RESULTS_DIR:-openenv-records/$RUN_ID}"
SERVER_LOG_DIR="$PLAYPEN_DIR/results/slurm/openenv"
SERVER_LOG_PATH="$SERVER_LOG_DIR/${RUN_ID}.log"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python interpreter at $PYTHON_BIN" >&2
  exit 1
fi

eval "$("$PYTHON_BIN" "$JOB_CONTEXT_HELPER" \
  --root-dir "$ROOT_DIR" \
  --playpen-dir "$PLAYPEN_DIR" \
  --job-id "${SLURM_JOB_ID:-}" \
  --trainer-file "$TRAINER_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --results-dir "$SERVER_RESULTS_DIR")"

mkdir -p "$SERVER_LOG_DIR" "$(dirname "$SERVER_RESULTS_DIR")" "$JOB_DIR" "$OUTPUT_DIR"

wait_for_port() {
  local host="$1"
  local port="$2"
  local attempts="${3:-60}"
  local sleep_seconds="${4:-2}"
  local attempt
  for attempt in $(seq 1 "$attempts"); do
    if (echo >"/dev/tcp/${host}/${port}") >/dev/null 2>&1; then
      return 0
    fi
    sleep "$sleep_seconds"
  done
  return 1
}

SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" || true
  fi
}
trap cleanup EXIT

cd "$JOB_DIR"

REGISTRY_ARGS=(--output "$MODEL_REGISTRY_PATH")
if [[ -n "${PLAYPEN_QWEN_PEFT_MODEL_NAME:-}" || -n "${PLAYPEN_QWEN_PEFT_ADAPTER_PATH:-}" ]]; then
  REGISTRY_ARGS+=(--qwen-peft-model-name "${PLAYPEN_QWEN_PEFT_MODEL_NAME:-}" --qwen-peft-adapter-path "${PLAYPEN_QWEN_PEFT_ADAPTER_PATH:-}")
fi
if [[ -n "${PLAYPEN_QWEN_BASE_MODEL:-}" ]]; then
  REGISTRY_ARGS+=(--qwen-base-model "${PLAYPEN_QWEN_BASE_MODEL}")
fi

"$PYTHON_BIN" "$ROOT_DIR/scripts/setup_playpen_registry.py" "${REGISTRY_ARGS[@]}"
"$PYTHON_BIN" "$ROOT_DIR/scripts/check_torch_cuda.py" --require-gpu

export CLEMBENCH_HOME="$PLAYPEN_DIR"
export PLAYPEN_OUTPUT_DIR="$OUTPUT_DIR"
export PLAYPEN_OPENENV_BASE_URL="http://${HOST}:${PORT}"
export TRL_EXPERIMENTAL_SILENCE="${TRL_EXPERIMENTAL_SILENCE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

"$PYTHON_BIN" "$RUN_METADATA_HELPER" \
  --root-dir "$ROOT_DIR" \
  --target-dir "$OUTPUT_DIR" \
  --mode grpo \
  --trainer-file "$TRAINER_FILE" \
  --learner-name "$LEARNER_NAME" \
  --output-dir "$OUTPUT_DIR" \
  --results-dir "$RESULTS_DIR"

SERVER_CMD=(
  "$CLEM_BIN"
  serve
  -g "$GAME_NAME"
  --split train
  --single-pass
  --host "$HOST"
  --port "$PORT"
  -r "$SERVER_RESULTS_DIR"
  --run-id "$RUN_ID"
)

printf 'Starting OpenEnv server:'
printf ' %q' "${SERVER_CMD[@]}"
printf '\n'
printf 'OpenEnv log: %s\n' "$SERVER_LOG_PATH"
printf 'OpenEnv base URL: %s\n' "$PLAYPEN_OPENENV_BASE_URL"

"${SERVER_CMD[@]}" >"$SERVER_LOG_PATH" 2>&1 &
SERVER_PID="$!"

if ! wait_for_port "$HOST" "$PORT" 90 2; then
  echo "OpenEnv server failed to become ready on ${HOST}:${PORT}" >&2
  tail -n 200 "$SERVER_LOG_PATH" >&2 || true
  exit 1
fi

CMD=("$PLAYPEN_BIN" run "$TRAINER_FILE" -l "$LEARNER_NAME")

printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '\n'
env | egrep '^(PLAYPEN_|TRL_|TOKENIZERS_)' | sort || true
"${CMD[@]}"
