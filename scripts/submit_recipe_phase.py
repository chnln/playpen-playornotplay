#!/usr/bin/env python3
"""Submit one training phase of the recipe; --dry-run never submits."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
KEEP_PLAYPEN_ENV = {"PLAYPEN_VENV_DIR", "PLAYPEN_SLURM_PARTITION"}


def prepare(phase_name: str, parent: Path | None, inherited: dict[str, str], time_limit: str | None = None):
    recipe = json.loads((ROOT / "configs/recipe.json").read_text())
    phase = recipe["phases"][phase_name]
    if phase["parent"] is None:
        if parent is not None:
            raise ValueError("Phase A starts from the base model and must not receive --parent")
    elif parent is None:
        raise ValueError(f"Phase {phase_name} requires --parent pointing to Phase {phase['parent']}'s completed adapter")
    if parent is not None:
        parent = parent.expanduser().resolve()
        if not (parent / "adapter_config.json").is_file() or not (parent / "adapter_model.safetensors").is_file():
            raise ValueError(f"Expected a completed safetensors LoRA adapter at {parent}")

    trainer = ROOT / phase["trainer"]
    if not trainer.is_file():
        raise ValueError(f"Trainer does not exist: {trainer}")

    # Avoid accidentally inheriting an experimental data filter, warm start,
    # beta, or pair file from a previous shell session. Keep HF/W&B credentials.
    env = {k: v for k, v in inherited.items() if not k.startswith("PLAYPEN_") or k in KEEP_PLAYPEN_ENV}
    env.update(recipe["common_env"])
    env.update(phase["env"])
    if phase_name != "A":
        word_list = ROOT / "environment/playpen/clembench/wordle/resources/target_words/en/official_recognized_words.txt"
        if not word_list.is_file() or not word_list.read_text().strip():
            raise ValueError(f"Required Wordle lexicon is missing or empty; see Installation in README.md: {word_list}")
        word_key = "PLAYPEN_VALID_WORDS_FILE" if phase_name == "C" else "PLAYPEN_WORDLE_VALID_WORDS_FILE"
        env[word_key] = str(word_list)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    env["WANDB_PROJECT"] = inherited.get("WANDB_PROJECT") or "playpen-wordle"
    env["WANDB_RUN_NAME"] = f"playornotplay-{phase_name}-{stamp}"
    learner = "Qwen3.5-2B" if parent is None else f"playornotplay-{phase_name}"
    if parent is not None:
        env["PLAYPEN_QWEN_PEFT_MODEL_NAME"] = learner
        env["PLAYPEN_QWEN_PEFT_ADAPTER_PATH"] = str(parent)
    if phase_name == "C":
        records = ROOT / "artifacts" / f"branch-{stamp}"
        env["PLAYPEN_BRANCH_RECORDS_DIR"] = str(records / "trajectories")
        # Save future realizations; this does not recreate the unsaved original corpus.
        env["PLAYPEN_SAVE_BRANCH_PAIRS"] = str(records / "pairs.jsonl.gz")

    command = ["bash", str(ROOT / "scripts/submit_playpen_train.sh"), str(trainer), learner,
               time_limit or phase["time_limit"]]
    return command, env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["A", "B1", "B2", "C"])
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--time", help="Explicit SLURM wall-time override")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        command, env = prepare(args.phase, args.parent, dict(os.environ), args.time)
    except ValueError as error:
        parser.error(str(error))
    if args.dry_run:
        # Only show recipe settings, never the complete inherited credential environment.
        visible = {k: v for k, v in env.items() if k.startswith("PLAYPEN_") or k in {"WANDB_PROJECT", "WANDB_RUN_NAME"}}
        print(json.dumps({"command": command, "recipe_env": visible}, indent=2))
        return 0
    return subprocess.run(command, cwd=ROOT, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
