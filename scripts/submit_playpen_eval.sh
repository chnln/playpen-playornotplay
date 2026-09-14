#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 MODEL [SUITE] [GAME] [TIME_LIMIT]" >&2
  exit 1
fi

MODEL_NAME="$1"
SUITE_NAME="${2:-clem}"
GAME_NAME="${3:-}"
TIME_LIMIT="${4:-02:00:00}"

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
SAFE_GAME="${GAME_NAME//[^A-Za-z0-9]/-}"
JOB_NAME="eval-${SAFE_MODEL}"
if [[ -n "$SAFE_GAME" ]]; then
  JOB_NAME="${JOB_NAME}-${SAFE_GAME}"
fi
RESULTS_DIR="playpen-eval/${STAMP}-${JOB_NAME}"

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
    "$ROOT_DIR/scripts/jobs/run_playpen_eval.sh" \
    "$ROOT_DIR" \
    "$MODEL_NAME" \
    "$SUITE_NAME" \
    "$GAME_NAME" \
    "$RESULTS_DIR"
)"

echo "JOB_ID=$JOB_ID"
echo "RESULTS_DIR=$RESULTS_DIR"
echo "LOG_PATH=$PLAYPEN_DIR/results/slurm/${JOB_NAME}-${JOB_ID}.out"
