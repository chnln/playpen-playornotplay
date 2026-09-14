#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 TRAINER_FILE MODEL [TIME_LIMIT]" >&2
  exit 1
fi

TRAINER_FILE="$1"
MODEL_NAME="$2"
TIME_LIMIT="${3:-04:00:00}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLAYPEN_DIR="$ROOT_DIR/environment/playpen"
if [[ -z "${PLAYPEN_VENV_DIR:-}" ]]; then
  if [[ -d "$PLAYPEN_DIR/venv_repro" ]]; then
    PLAYPEN_VENV_DIR="$PLAYPEN_DIR/venv_repro"
  else
    PLAYPEN_VENV_DIR="$PLAYPEN_DIR/venv"
  fi
fi
SBATCH_ENV=(PLAYPEN_VENV_DIR="$PLAYPEN_VENV_DIR")
STAMP="$(date +%Y%m%dT%H%M%S)"
SAFE_MODEL="${MODEL_NAME//[^A-Za-z0-9]/-}"
JOB_NAME="train-${SAFE_MODEL}"
OUTPUT_DIR="models/runs/${STAMP}-${SAFE_MODEL}"

mkdir -p "$PLAYPEN_DIR/results/slurm"

JOB_ID="$(
  cd "$ROOT_DIR"
  env "${SBATCH_ENV[@]}" \
    sbatch \
    --parsable \
    -p "${PLAYPEN_SLURM_PARTITION:-gpua100}" \
    --gres=gpu:1 \
    --time "$TIME_LIMIT" \
    --job-name "$JOB_NAME" \
    --output "$PLAYPEN_DIR/results/slurm/%x-%j.out" \
    "$ROOT_DIR/scripts/jobs/run_playpen_train.sh" \
    "$ROOT_DIR" \
    "$TRAINER_FILE" \
    "$MODEL_NAME" \
    "$OUTPUT_DIR"
)"

echo "JOB_ID=$JOB_ID"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "LOG_PATH=$PLAYPEN_DIR/results/slurm/${JOB_NAME}-${JOB_ID}.out"
