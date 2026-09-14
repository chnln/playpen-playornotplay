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

submit_train() {
  local dependency_job="$1"
  local dependency_mode="$2"
  local model_name="$3"
  local trainer_file="$4"
  local output_dir="$5"
  local time_limit="$6"
  shift 6

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
        "$trainer_file" \
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

PREV_JOB="$START_DEPENDENCY_JOB"
PREV_MODE="afterany"

REPEAT_MODEL="Qwen3.5-2B-wordle-turn-dpo-repeat-r1"
REPEAT_SAFE="${REPEAT_MODEL//[^A-Za-z0-9]/-}"
REPEAT_OUTPUT_DIR="models/runs/${STAMP}-${REPEAT_SAFE}"
REPEAT_JOB="$(
  submit_train \
    "$PREV_JOB" \
    "$PREV_MODE" \
    "$REPEAT_MODEL" \
    "trainers/qwen35_2b_wordle_turn_dpo_lora.py" \
    "$REPEAT_OUTPUT_DIR" \
    "04:00:00" \
    PLAYPEN_QWEN_PEFT_MODEL_NAME="$REPEAT_MODEL" \
    PLAYPEN_QWEN_PEFT_ADAPTER_PATH="$PARENT_ADAPTER_PATH" \
    PLAYPEN_WORDLE_CLEAN_ONLY=1 \
    PLAYPEN_MIN_ASSISTANT_TURN=2 \
    PLAYPEN_REQUIRE_PREV_USER_SUBSTRING="guess_feedback:" \
    PLAYPEN_NEGATIVE_MODES="repeat_last" \
    PLAYPEN_EVAL_RATIO=0.05 \
    PLAYPEN_BATCH_SIZE=1 \
    PLAYPEN_GRAD_ACCUM=16 \
    PLAYPEN_MAX_LENGTH=1024 \
    PLAYPEN_NUM_TRAIN_EPOCHS=1 \
    PLAYPEN_LR=5e-6 \
    PLAYPEN_DPO_BETA=0.1 \
    PLAYPEN_GRADIENT_CHECKPOINTING=0 \
    PLAYPEN_USE_SEPARATE_REF_MODEL=1 \
    PLAYPEN_SAVE_STEPS=50 \
    PLAYPEN_EVAL_STEPS=50 \
    PLAYPEN_LOGGING_STEPS=10 \
    PLAYPEN_SAVE_TOTAL_LIMIT=2
)"
REPEAT_FINAL="$PLAYPEN_DIR/${REPEAT_OUTPUT_DIR}/final"
REPEAT_EVAL_MODEL="Qwen3.5-2B-wordle-turn-dpo-repeat-lora-r1"
REPEAT_EVAL_SAFE="${REPEAT_EVAL_MODEL//[^A-Za-z0-9]/-}"
REPEAT_WORDLE_JOB="$(
  submit_eval \
    "$REPEAT_JOB" \
    "$REPEAT_EVAL_MODEL" \
    "wordle" \
    "playpen-eval/${STAMP}-eval-${REPEAT_EVAL_SAFE}-wordle" \
    "$REPEAT_FINAL" \
    "02:00:00"
)"
REPEAT_FULL_JOB="$(
  submit_eval \
    "$REPEAT_JOB" \
    "$REPEAT_EVAL_MODEL" \
    "" \
    "playpen-eval/${STAMP}-eval-${REPEAT_EVAL_SAFE}" \
    "$REPEAT_FINAL" \
    "02:30:00"
)"

PREV_JOB="$REPEAT_FULL_JOB"
PREV_MODE="afterany"

HYBRID_MODEL="Qwen3.5-2B-wordle-turn-dpo-repeat-badlen-r1"
HYBRID_SAFE="${HYBRID_MODEL//[^A-Za-z0-9]/-}"
HYBRID_OUTPUT_DIR="models/runs/${STAMP}-${HYBRID_SAFE}"
HYBRID_JOB="$(
  submit_train \
    "$PREV_JOB" \
    "$PREV_MODE" \
    "$HYBRID_MODEL" \
    "trainers/qwen35_2b_wordle_turn_dpo_lora.py" \
    "$HYBRID_OUTPUT_DIR" \
    "04:00:00" \
    PLAYPEN_QWEN_PEFT_MODEL_NAME="$HYBRID_MODEL" \
    PLAYPEN_QWEN_PEFT_ADAPTER_PATH="$PARENT_ADAPTER_PATH" \
    PLAYPEN_WORDLE_CLEAN_ONLY=1 \
    PLAYPEN_MIN_ASSISTANT_TURN=2 \
    PLAYPEN_REQUIRE_PREV_USER_SUBSTRING="guess_feedback:" \
    PLAYPEN_NEGATIVE_MODES="repeat_last,bad_length" \
    PLAYPEN_EVAL_RATIO=0.05 \
    PLAYPEN_BATCH_SIZE=1 \
    PLAYPEN_GRAD_ACCUM=16 \
    PLAYPEN_MAX_LENGTH=1024 \
    PLAYPEN_NUM_TRAIN_EPOCHS=1 \
    PLAYPEN_LR=5e-6 \
    PLAYPEN_DPO_BETA=0.1 \
    PLAYPEN_GRADIENT_CHECKPOINTING=0 \
    PLAYPEN_USE_SEPARATE_REF_MODEL=1 \
    PLAYPEN_SAVE_STEPS=50 \
    PLAYPEN_EVAL_STEPS=50 \
    PLAYPEN_LOGGING_STEPS=10 \
    PLAYPEN_SAVE_TOTAL_LIMIT=2
)"
HYBRID_FINAL="$PLAYPEN_DIR/${HYBRID_OUTPUT_DIR}/final"
HYBRID_EVAL_MODEL="Qwen3.5-2B-wordle-turn-dpo-repeat-badlen-lora-r1"
HYBRID_EVAL_SAFE="${HYBRID_EVAL_MODEL//[^A-Za-z0-9]/-}"
HYBRID_WORDLE_JOB="$(
  submit_eval \
    "$HYBRID_JOB" \
    "$HYBRID_EVAL_MODEL" \
    "wordle" \
    "playpen-eval/${STAMP}-eval-${HYBRID_EVAL_SAFE}-wordle" \
    "$HYBRID_FINAL" \
    "02:00:00"
)"
HYBRID_FULL_JOB="$(
  submit_eval \
    "$HYBRID_JOB" \
    "$HYBRID_EVAL_MODEL" \
    "" \
    "playpen-eval/${STAMP}-eval-${HYBRID_EVAL_SAFE}" \
    "$HYBRID_FINAL" \
    "02:30:00"
)"

WATCH_LOG="$ROOT_DIR/artifacts/job_watch_${STAMP}_wordle_turn_dpo.log"
nohup "$ROOT_DIR/scripts/watch_slurm_jobs.sh" 300 "$WATCH_LOG" \
  "$REPEAT_JOB" "$REPEAT_WORDLE_JOB" "$REPEAT_FULL_JOB" \
  "$HYBRID_JOB" "$HYBRID_WORDLE_JOB" "$HYBRID_FULL_JOB" >/dev/null 2>&1 &

echo "STAMP=$STAMP"
echo "REPEAT_JOB=$REPEAT_JOB"
echo "REPEAT_WORDLE_JOB=$REPEAT_WORDLE_JOB"
echo "REPEAT_FULL_JOB=$REPEAT_FULL_JOB"
echo "HYBRID_JOB=$HYBRID_JOB"
echo "HYBRID_WORDLE_JOB=$HYBRID_WORDLE_JOB"
echo "HYBRID_FULL_JOB=$HYBRID_FULL_JOB"
echo "WATCH_LOG=$WATCH_LOG"
