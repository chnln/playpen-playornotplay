#!/usr/bin/env bash
# Historical exploration entrypoint; not the submitted effective recipe.
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

submit_grpo_train() {
  local dependency_job="$1"
  local dependency_mode="$2"
  local model_name="$3"
  local output_dir="$4"
  local game_name="$5"
  local port="$6"
  local time_limit="$7"
  shift 7

  local safe_model="${model_name//[^A-Za-z0-9]/-}"
  local dependency_args=()
  if [[ -n "$dependency_job" ]]; then
    dependency_args+=(--dependency="${dependency_mode}:${dependency_job}")
  fi

  (
    cd "$ROOT_DIR"
    env "${SBATCH_ENV[@]}" "$@" \
      PLAYPEN_OPENENV_PORT="$port" \
      PLAYPEN_OPENENV_RUN_ID="${STAMP}-${safe_model}-${game_name}" \
      sbatch \
        --parsable \
        "${dependency_args[@]}" \
        -p "${PLAYPEN_SLURM_PARTITION:-gpua100}" \
        --gres=gpu:1 \
        --time "$time_limit" \
        --job-name "grpo-${safe_model}" \
        --output "$PLAYPEN_DIR/results/slurm/%x-%j.out" \
        "$ROOT_DIR/scripts/jobs/run_playpen_grpo.sh" \
        "$ROOT_DIR" \
        "trainers/qwen35_2b_grpo_lora.py" \
        "$model_name" \
        "$output_dir" \
        "$game_name"
  )
}

submit_eval() {
  local dependency_job="$1"
  local model_name="$2"
  local game_name="$3"
  local results_dir="$4"
  local adapter_path="$5"
  local time_limit="${6:-02:00:00}"

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

COMMON_REWARD_ENV=(
  PLAYPEN_WORDLE_SCHEMA_REWARD=0.10
  PLAYPEN_WORDLE_MISSING_EXPLANATION_PENALTY=0.50
  PLAYPEN_WORDLE_MISSING_GUESS_PENALTY=1.00
  PLAYPEN_WORDLE_PREAMBLE_PENALTY=0.25
  PLAYPEN_WORDLE_BAD_LENGTH_PENALTY=1.50
  PLAYPEN_WORDLE_REPEAT_GUESS_PENALTY=1.50
  PLAYPEN_WORDLE_NOVEL_GUESS_REWARD=0.10
  PLAYPEN_WORDLE_CONSTRAINT_VALID_REWARD=0.30
  PLAYPEN_WORDLE_MISSING_REQUIRED_LETTER_PENALTY=0.50
  PLAYPEN_WORDLE_ABSENT_LETTER_PENALTY=0.30
  PLAYPEN_WORDLE_GREEN_POSITION_PENALTY=0.75
  PLAYPEN_WORDLE_BAD_POSITION_PENALTY=0.50
)

PILOT_MODEL="Qwen3.5-2B-wordle-grpo-constraint-pilot-r1"
PILOT_SAFE="${PILOT_MODEL//[^A-Za-z0-9]/-}"
PILOT_OUTPUT_DIR="models/runs/${STAMP}-${PILOT_SAFE}"
PILOT_JOB="$(
  submit_grpo_train \
    "$START_DEPENDENCY_JOB" \
    "afterany" \
    "$PILOT_MODEL" \
    "$PILOT_OUTPUT_DIR" \
    "wordle" \
    "9104" \
    "03:00:00" \
    PLAYPEN_QWEN_PEFT_MODEL_NAME="$PILOT_MODEL" \
    PLAYPEN_QWEN_PEFT_ADAPTER_PATH="$PARENT_ADAPTER_PATH" \
    PLAYPEN_SEED=42 \
    PLAYPEN_MAX_INSTANCES=8 \
    PLAYPEN_BATCH_SIZE=1 \
    PLAYPEN_GRAD_ACCUM=4 \
    PLAYPEN_NUM_TRAIN_EPOCHS=1 \
    PLAYPEN_LR=1e-6 \
    PLAYPEN_NUM_GENERATIONS=4 \
    PLAYPEN_TEMPERATURE=0.60 \
    PLAYPEN_TOP_P=0.90 \
    PLAYPEN_GRPO_BETA=0.03 \
    PLAYPEN_MAX_COMPLETION_LENGTH=256 \
    PLAYPEN_SAVE_STEPS=20 \
    PLAYPEN_LOGGING_STEPS=1 \
    PLAYPEN_SAVE_TOTAL_LIMIT=2 \
    "${COMMON_REWARD_ENV[@]}"
)"
PILOT_FINAL="$PLAYPEN_DIR/${PILOT_OUTPUT_DIR}/final"
PILOT_EVAL_MODEL="Qwen3.5-2B-wordle-grpo-constraint-pilot-lora-r1"
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

DENSE_MODEL="Qwen3.5-2B-wordle-grpo-constraint-dense-r1"
DENSE_SAFE="${DENSE_MODEL//[^A-Za-z0-9]/-}"
DENSE_OUTPUT_DIR="models/runs/${STAMP}-${DENSE_SAFE}"
DENSE_JOB="$(
  submit_grpo_train \
    "$PILOT_JOB" \
    "afterok" \
    "$DENSE_MODEL" \
    "$DENSE_OUTPUT_DIR" \
    "wordle" \
    "9105" \
    "05:00:00" \
    PLAYPEN_QWEN_PEFT_MODEL_NAME="$DENSE_MODEL" \
    PLAYPEN_QWEN_PEFT_ADAPTER_PATH="$PARENT_ADAPTER_PATH" \
    PLAYPEN_SEED=42 \
    PLAYPEN_BATCH_SIZE=1 \
    PLAYPEN_GRAD_ACCUM=4 \
    PLAYPEN_NUM_TRAIN_EPOCHS=2 \
    PLAYPEN_LR=7.5e-7 \
    PLAYPEN_NUM_GENERATIONS=4 \
    PLAYPEN_TEMPERATURE=0.60 \
    PLAYPEN_TOP_P=0.90 \
    PLAYPEN_GRPO_BETA=0.03 \
    PLAYPEN_MAX_COMPLETION_LENGTH=256 \
    PLAYPEN_SAVE_STEPS=25 \
    PLAYPEN_LOGGING_STEPS=1 \
    PLAYPEN_SAVE_TOTAL_LIMIT=2 \
    "${COMMON_REWARD_ENV[@]}"
)"
DENSE_FINAL="$PLAYPEN_DIR/${DENSE_OUTPUT_DIR}/final"
DENSE_EVAL_MODEL="Qwen3.5-2B-wordle-grpo-constraint-dense-lora-r1"
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

CAUTIOUS_MODEL="Qwen3.5-2B-wordle-grpo-constraint-cautious-r1"
CAUTIOUS_SAFE="${CAUTIOUS_MODEL//[^A-Za-z0-9]/-}"
CAUTIOUS_OUTPUT_DIR="models/runs/${STAMP}-${CAUTIOUS_SAFE}"
CAUTIOUS_JOB="$(
  submit_grpo_train \
    "$PILOT_JOB" \
    "afterok" \
    "$CAUTIOUS_MODEL" \
    "$CAUTIOUS_OUTPUT_DIR" \
    "wordle" \
    "9106" \
    "05:00:00" \
    PLAYPEN_QWEN_PEFT_MODEL_NAME="$CAUTIOUS_MODEL" \
    PLAYPEN_QWEN_PEFT_ADAPTER_PATH="$PARENT_ADAPTER_PATH" \
    PLAYPEN_SEED=43 \
    PLAYPEN_BATCH_SIZE=1 \
    PLAYPEN_GRAD_ACCUM=4 \
    PLAYPEN_NUM_TRAIN_EPOCHS=2 \
    PLAYPEN_LR=5e-7 \
    PLAYPEN_NUM_GENERATIONS=4 \
    PLAYPEN_TEMPERATURE=0.50 \
    PLAYPEN_TOP_P=0.90 \
    PLAYPEN_GRPO_BETA=0.05 \
    PLAYPEN_MAX_COMPLETION_LENGTH=256 \
    PLAYPEN_SAVE_STEPS=25 \
    PLAYPEN_LOGGING_STEPS=1 \
    PLAYPEN_SAVE_TOTAL_LIMIT=2 \
    "${COMMON_REWARD_ENV[@]}"
)"
CAUTIOUS_FINAL="$PLAYPEN_DIR/${CAUTIOUS_OUTPUT_DIR}/final"
CAUTIOUS_EVAL_MODEL="Qwen3.5-2B-wordle-grpo-constraint-cautious-lora-r1"
CAUTIOUS_EVAL_SAFE="${CAUTIOUS_EVAL_MODEL//[^A-Za-z0-9]/-}"
CAUTIOUS_WORDLE_JOB="$(
  submit_eval \
    "$CAUTIOUS_JOB" \
    "$CAUTIOUS_EVAL_MODEL" \
    "wordle" \
    "playpen-eval/${STAMP}-eval-${CAUTIOUS_EVAL_SAFE}-wordle" \
    "$CAUTIOUS_FINAL" \
    "02:00:00"
)"
CAUTIOUS_FULL_JOB="$(
  submit_eval \
    "$CAUTIOUS_JOB" \
    "$CAUTIOUS_EVAL_MODEL" \
    "" \
    "playpen-eval/${STAMP}-eval-${CAUTIOUS_EVAL_SAFE}" \
    "$CAUTIOUS_FINAL" \
    "02:30:00"
)"

nohup "$ROOT_DIR/scripts/watch_slurm_jobs.sh" \
  300 \
  "$ROOT_DIR/artifacts/job_watch_${STAMP}_grpo_constraint.log" \
  "$PILOT_JOB" "$PILOT_WORDLE_JOB" \
  "$DENSE_JOB" "$DENSE_WORDLE_JOB" "$DENSE_FULL_JOB" \
  "$CAUTIOUS_JOB" "$CAUTIOUS_WORDLE_JOB" "$CAUTIOUS_FULL_JOB" \
  >/dev/null 2>&1 &

echo "STAMP=$STAMP"
echo "PILOT_JOB=$PILOT_JOB"
echo "PILOT_WORDLE_JOB=$PILOT_WORDLE_JOB"
echo "DENSE_JOB=$DENSE_JOB"
echo "DENSE_WORDLE_JOB=$DENSE_WORDLE_JOB"
echo "DENSE_FULL_JOB=$DENSE_FULL_JOB"
echo "CAUTIOUS_JOB=$CAUTIOUS_JOB"
echo "CAUTIOUS_WORDLE_JOB=$CAUTIOUS_WORDLE_JOB"
echo "CAUTIOUS_FULL_JOB=$CAUTIOUS_FULL_JOB"
echo "WATCH_LOG=$ROOT_DIR/artifacts/job_watch_${STAMP}_grpo_constraint.log"
