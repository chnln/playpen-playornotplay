#!/usr/bin/env python3
"""Write reproducibility metadata for playpen train/eval job outputs."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _run_git(root_dir: Path, *args: str) -> str | None:
    proc = subprocess.run(
        ["git", "-C", str(root_dir), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def collect_git_metadata(root_dir: Path) -> dict[str, object]:
    commit = _run_git(root_dir, "rev-parse", "HEAD")
    branch = _run_git(root_dir, "branch", "--show-current")
    status_porcelain = _run_git(root_dir, "status", "--porcelain")
    status_lines = status_porcelain.splitlines() if status_porcelain else []
    return {
        "commit": commit,
        "branch": branch or None,
        "dirty": bool(status_lines),
        "status_porcelain": status_lines,
    }


def collect_playpen_env(env: dict[str, str]) -> dict[str, str]:
    return {key: env[key] for key in sorted(env) if key.startswith("PLAYPEN_")}


def collect_slurm_metadata(env: dict[str, str]) -> dict[str, str | None]:
    return {
        "job_id": env.get("SLURM_JOB_ID"),
        "job_name": env.get("SLURM_JOB_NAME"),
        "partition": env.get("SLURM_JOB_PARTITION"),
        "node_list": env.get("SLURM_JOB_NODELIST"),
        "node_name": env.get("SLURMD_NODENAME"),
    }


def build_invocation_metadata(
    trainer_file: str | None = None,
    learner_name: str | None = None,
    output_dir: str | None = None,
    model_name: str | None = None,
    suite_name: str | None = None,
    game_name: str | None = None,
    results_dir: str | None = None,
) -> dict[str, str]:
    payload: dict[str, str] = {}
    if trainer_file:
        payload["trainer_file"] = trainer_file
    if learner_name:
        payload["learner_name"] = learner_name
    if output_dir:
        payload["output_dir"] = output_dir
    if model_name:
        payload["model_name"] = model_name
    if suite_name:
        payload["suite_name"] = suite_name
    if game_name:
        payload["game_name"] = game_name
    if results_dir:
        payload["results_dir"] = results_dir
    return payload


def build_run_metadata(
    *,
    root_dir: Path,
    target_dir: Path,
    mode: str,
    trainer_file: str | None = None,
    learner_name: str | None = None,
    output_dir: str | None = None,
    model_name: str | None = None,
    suite_name: str | None = None,
    game_name: str | None = None,
    results_dir: str | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, object]:
    effective_env = dict(os.environ if env is None else env)
    root_dir = root_dir.resolve()
    target_dir = target_dir.resolve()
    return {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "mode": mode,
        "target_dir": str(target_dir),
        "root_dir": str(root_dir),
        "hostname": socket.gethostname(),
        "python_executable": sys.executable,
        "git": collect_git_metadata(root_dir),
        "slurm": collect_slurm_metadata(effective_env),
        "playpen_env": collect_playpen_env(effective_env),
        "invocation": build_invocation_metadata(
            trainer_file=trainer_file,
            learner_name=learner_name,
            output_dir=output_dir,
            model_name=model_name,
            suite_name=suite_name,
            game_name=game_name,
            results_dir=results_dir,
        ),
    }


def write_run_metadata(
    *,
    root_dir: Path,
    target_dir: Path,
    mode: str,
    trainer_file: str | None = None,
    learner_name: str | None = None,
    output_dir: str | None = None,
    model_name: str | None = None,
    suite_name: str | None = None,
    game_name: str | None = None,
    results_dir: str | None = None,
    env: dict[str, str] | None = None,
) -> Path:
    target_dir = target_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    payload = build_run_metadata(
        root_dir=root_dir,
        target_dir=target_dir,
        mode=mode,
        trainer_file=trainer_file,
        learner_name=learner_name,
        output_dir=output_dir,
        model_name=model_name,
        suite_name=suite_name,
        game_name=game_name,
        results_dir=results_dir,
        env=env,
    )
    output_path = target_dir / "run_metadata.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["train", "eval", "grpo"], required=True)
    parser.add_argument("--trainer-file", default=None)
    parser.add_argument("--learner-name", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--suite-name", default=None)
    parser.add_argument("--game-name", default=None)
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()

    output_path = write_run_metadata(
        root_dir=args.root_dir,
        target_dir=args.target_dir,
        mode=args.mode,
        trainer_file=args.trainer_file,
        learner_name=args.learner_name,
        output_dir=args.output_dir,
        model_name=args.model_name,
        suite_name=args.suite_name,
        game_name=args.game_name,
        results_dir=args.results_dir,
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
