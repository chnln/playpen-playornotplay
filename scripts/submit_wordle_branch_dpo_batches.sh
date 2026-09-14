#!/usr/bin/env bash
# Exploratory sweep launcher; not part of the submitted recipe.
# Use scripts/submit_recipe_phase.py for phases A/B1/B2/C.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 PARENT_ADAPTER_PATH [START_DEPENDENCY_JOB] [STAMP]" >&2
  exit 1
fi

PARENT_ADAPTER_PATH="$1"
START_DEPENDENCY_JOB="${2:-}"
STAMP="${3:-$(date +%Y%m%dT%H%M%S)}"

if [[ -n "${ROOT_DIR:-}" ]]; then
  ROOT_DIR="$(cd "$ROOT_DIR" && pwd)"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
PLAYPEN_DIR="$ROOT_DIR/environment/playpen"
if [[ -z "${PLAYPEN_VENV_DIR:-}" ]]; then
  if [[ -d "$PLAYPEN_DIR/venv_repro" ]]; then
    PLAYPEN_VENV_DIR="$PLAYPEN_DIR/venv_repro"
  else
    PLAYPEN_VENV_DIR="$PLAYPEN_DIR/venv"
  fi
fi
SBATCH_ENV=(PLAYPEN_VENV_DIR="$PLAYPEN_VENV_DIR")

mkdir -p "$PLAYPEN_DIR/results/slurm" "$ROOT_DIR/artifacts"

submit_train() {
  local dependency_job="$1"
  local dependency_mode="$2"
  local model_name="$3"
  local output_dir="$4"
  local time_limit="$5"
  shift 5

  local safe_model="${model_name//[^A-Za-z0-9]/-}"
  local dependency_args=()
  if [[ -n "$dependency_job" ]]; then
    dependency_args+=(--dependency="${dependency_mode}:${dependency_job}")
  fi

  (
    cd "$ROOT_DIR"
    env "${SBATCH_ENV[@]}" "$@" \
      sbatch \
        --parsable \
        "${dependency_args[@]}" \
        -p "${PLAYPEN_SLURM_PARTITION:-gpua100}" \
        --gres=gpu:1 \
        --time "$time_limit" \
        --job-name "train-${safe_model}" \
        --output "$PLAYPEN_DIR/results/slurm/%x-%j.out" \
        "$ROOT_DIR/scripts/jobs/run_playpen_train.sh" \
        "$ROOT_DIR" \
        "trainers/qwen35_2b_wordle_branch_dpo_lora.py" \
        "$model_name" \
        "$output_dir"
  )
}

submit_eval() {
  local dependency_job="$1"
  local model_name="$2"
  local game_name="$3"
  local results_dir="$4"
  local adapter_path="$5"
  local time_limit="${6:-02:30:00}"

  local safe_model="${model_name//[^A-Za-z0-9]/-}"
  local safe_game="${game_name//[^A-Za-z0-9]/-}"
  local job_name="eval-${safe_model}"
  if [[ -n "$safe_game" ]]; then
    job_name="${job_name}-${safe_game}"
  fi

  (
    cd "$ROOT_DIR"
    env "${SBATCH_ENV[@]}" \
      PLAYPEN_QWEN_PEFT_MODEL_NAME="$model_name" \
      PLAYPEN_QWEN_PEFT_ADAPTER_PATH="$adapter_path" \
      sbatch \
        --parsable \
        --dependency="afterok:${dependency_job}" \
        -p "${PLAYPEN_SLURM_PARTITION:-gpua100}" \
        --gres=gpu:1 \
        --time "$time_limit" \
        --job-name "$job_name" \
        --output "$PLAYPEN_DIR/results/slurm/%x-%j.out" \
        "$ROOT_DIR/scripts/jobs/run_playpen_eval.sh" \
        "$ROOT_DIR" \
        "$model_name" \
        "clem" \
        "$game_name" \
        "$results_dir"
  )
}

COMMON_ENV=(
  PLAYPEN_MAX_TOKENS=256
  PLAYPEN_PLAYER_NAME="Player 1"
  PLAYPEN_EVAL_RATIO=0.05
  PLAYPEN_BATCH_SIZE=1
  PLAYPEN_GRAD_ACCUM=16
  PLAYPEN_MAX_LENGTH=1024
  PLAYPEN_NUM_TRAIN_EPOCHS=1
  PLAYPEN_LR=5e-6
  PLAYPEN_DPO_BETA=0.1
  PLAYPEN_GRADIENT_CHECKPOINTING=0
  PLAYPEN_USE_SEPARATE_REF_MODEL=1
  PLAYPEN_SAVE_STEPS=50
  PLAYPEN_EVAL_STEPS=50
  PLAYPEN_LOGGING_STEPS=10
  PLAYPEN_SAVE_TOTAL_LIMIT=2
)

PREV_JOB="$START_DEPENDENCY_JOB"
PREV_MODE="afterany"

PILOT_MODEL="Qwen3.5-2B-wordle-branch-dpo-pilot-r1"
PILOT_SAFE="${PILOT_MODEL//[^A-Za-z0-9]/-}"
PILOT_OUTPUT_DIR="models/runs/${STAMP}-${PILOT_SAFE}"
PILOT_JOB="$(
  submit_train \
    "$PREV_JOB" \
    "$PREV_MODE" \
    "$PILOT_MODEL" \
    "$PILOT_OUTPUT_DIR" \
    "03:00:00" \
    PLAYPEN_QWEN_PEFT_MODEL_NAME="$PILOT_MODEL" \
    PLAYPEN_QWEN_PEFT_ADAPTER_PATH="$PARENT_ADAPTER_PATH" \
    PLAYPEN_TEMPERATURE=0.7 \
    PLAYPEN_BRANCHING_FACTOR=2 \
    PLAYPEN_MIN_ROUND=1 \
    PLAYPEN_MAX_ROUND=2 \
    PLAYPEN_MAX_INSTANCES=8 \
    PLAYPEN_BRANCH_RECORDS_DIR="playpen-records-branching-wordle/${STAMP}-${PILOT_SAFE}" \
    "${COMMON_ENV[@]}"
)"
PILOT_FINAL="$PLAYPEN_DIR/${PILOT_OUTPUT_DIR}/final"
PILOT_EVAL_MODEL="Qwen3.5-2B-wordle-branch-dpo-pilot-lora-r1"
PILOT_EVAL_SAFE="${PILOT_EVAL_MODEL//[^A-Za-z0-9]/-}"
PILOT_WORDLE_JOB="$(
  submit_eval \
    "$PILOT_JOB" \
    "$PILOT_EVAL_MODEL" \
    "wordle" \
    "playpen-eval/${STAMP}-eval-${PILOT_EVAL_SAFE}-wordle" \
    "$PILOT_FINAL" \
    "02:00:00"
)"
PILOT_FULL_JOB="$(
  submit_eval \
    "$PILOT_JOB" \
    "$PILOT_EVAL_MODEL" \
    "" \
    "playpen-eval/${STAMP}-eval-${PILOT_EVAL_SAFE}" \
    "$PILOT_FINAL" \
    "02:30:00"
)"

MAIN_MODEL="Qwen3.5-2B-wordle-branch-dpo-main-r1"
MAIN_SAFE="${MAIN_MODEL//[^A-Za-z0-9]/-}"
MAIN_OUTPUT_DIR="models/runs/${STAMP}-${MAIN_SAFE}"
MAIN_JOB="$(
  submit_train \
    "$PILOT_JOB" \
    "afterok" \
    "$MAIN_MODEL" \
    "$MAIN_OUTPUT_DIR" \
    "05:30:00" \
    PLAYPEN_QWEN_PEFT_MODEL_NAME="$MAIN_MODEL" \
    PLAYPEN_QWEN_PEFT_ADAPTER_PATH="$PARENT_ADAPTER_PATH" \
    PLAYPEN_TEMPERATURE=0.7 \
    PLAYPEN_BRANCHING_FACTOR=2 \
    PLAYPEN_MIN_ROUND=1 \
    PLAYPEN_MAX_ROUND=3 \
    PLAYPEN_MAX_INSTANCES=27 \
    PLAYPEN_BRANCH_RECORDS_DIR="playpen-records-branching-wordle/${STAMP}-${MAIN_SAFE}" \
    "${COMMON_ENV[@]}"
)"
MAIN_FINAL="$PLAYPEN_DIR/${MAIN_OUTPUT_DIR}/final"
MAIN_EVAL_MODEL="Qwen3.5-2B-wordle-branch-dpo-main-lora-r1"
MAIN_EVAL_SAFE="${MAIN_EVAL_MODEL//[^A-Za-z0-9]/-}"
MAIN_WORDLE_JOB="$(
  submit_eval \
    "$MAIN_JOB" \
    "$MAIN_EVAL_MODEL" \
    "wordle" \
    "playpen-eval/${STAMP}-eval-${MAIN_EVAL_SAFE}-wordle" \
    "$MAIN_FINAL" \
    "02:00:00"
)"
MAIN_FULL_JOB="$(
  submit_eval \
    "$MAIN_JOB" \
    "$MAIN_EVAL_MODEL" \
    "" \
    "playpen-eval/${STAMP}-eval-${MAIN_EVAL_SAFE}" \
    "$MAIN_FINAL" \
    "02:30:00"
)"

DENSE_MODEL="Qwen3.5-2B-wordle-branch-dpo-dense-r1"
DENSE_SAFE="${DENSE_MODEL//[^A-Za-z0-9]/-}"
DENSE_OUTPUT_DIR="models/runs/${STAMP}-${DENSE_SAFE}"
DENSE_JOB="$(
  submit_train \
    "$PILOT_JOB" \
    "afterok" \
    "$DENSE_MODEL" \
    "$DENSE_OUTPUT_DIR" \
    "06:00:00" \
    PLAYPEN_QWEN_PEFT_MODEL_NAME="$DENSE_MODEL" \
    PLAYPEN_QWEN_PEFT_ADAPTER_PATH="$PARENT_ADAPTER_PATH" \
    PLAYPEN_TEMPERATURE=0.8 \
    PLAYPEN_BRANCHING_FACTOR=3 \
    PLAYPEN_MIN_ROUND=1 \
    PLAYPEN_MAX_ROUND=2 \
    PLAYPEN_MAX_INSTANCES=27 \
    PLAYPEN_BRANCH_RECORDS_DIR="playpen-records-branching-wordle/${STAMP}-${DENSE_SAFE}" \
    "${COMMON_ENV[@]}"
)"
DENSE_FINAL="$PLAYPEN_DIR/${DENSE_OUTPUT_DIR}/final"
DENSE_EVAL_MODEL="Qwen3.5-2B-wordle-branch-dpo-dense-lora-r1"
DENSE_EVAL_SAFE="${DENSE_EVAL_MODEL//[^A-Za-z0-9]/-}"
DENSE_WORDLE_JOB="$(
  submit_eval \
    "$DENSE_JOB" \
    "$DENSE_EVAL_MODEL" \
    "wordle" \
    "playpen-eval/${STAMP}-eval-${DENSE_EVAL_SAFE}-wordle" \
    "$DENSE_FINAL" \
    "02:00:00"
)"
DENSE_FULL_JOB="$(
  submit_eval \
    "$DENSE_JOB" \
    "$DENSE_EVAL_MODEL" \
    "" \
    "playpen-eval/${STAMP}-eval-${DENSE_EVAL_SAFE}" \
    "$DENSE_FINAL" \
    "02:30:00"
)"

WATCH_LOG="$ROOT_DIR/artifacts/job_watch_${STAMP}_wordle_branch_dpo.log"
nohup "$ROOT_DIR/scripts/watch_slurm_jobs.sh" 300 "$WATCH_LOG" \
  "$PILOT_JOB" "$PILOT_WORDLE_JOB" "$PILOT_FULL_JOB" \
  "$MAIN_JOB" "$MAIN_WORDLE_JOB" "$MAIN_FULL_JOB" \
  "$DENSE_JOB" "$DENSE_WORDLE_JOB" "$DENSE_FULL_JOB" >/dev/null 2>&1 &

echo "STAMP=$STAMP"
echo "PILOT_JOB=$PILOT_JOB"
echo "PILOT_WORDLE_JOB=$PILOT_WORDLE_JOB"
echo "PILOT_FULL_JOB=$PILOT_FULL_JOB"
echo "MAIN_JOB=$MAIN_JOB"
echo "MAIN_WORDLE_JOB=$MAIN_WORDLE_JOB"
echo "MAIN_FULL_JOB=$MAIN_FULL_JOB"
echo "DENSE_JOB=$DENSE_JOB"
echo "DENSE_WORDLE_JOB=$DENSE_WORDLE_JOB"
echo "DENSE_FULL_JOB=$DENSE_FULL_JOB"
echo "WATCH_LOG=$WATCH_LOG"
