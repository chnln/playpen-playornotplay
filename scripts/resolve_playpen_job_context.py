#!/usr/bin/env python3
"""Resolve isolated per-job paths for playpen SLURM wrappers."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path


def resolve_path(base_dir: Path, value: str | None) -> str | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def resolve_trainer_path(root_dir: Path, playpen_dir: Path, trainer_file: str | None) -> str | None:
    if not trainer_file:
        return None
    path = Path(trainer_file)
    if path.is_absolute():
        return str(path)
    playpen_candidate = (playpen_dir / path).resolve()
    if playpen_candidate.exists():
        return str(playpen_candidate)
    return str((root_dir / path).resolve())


def resolve_job_context(
    root_dir: Path,
    playpen_dir: Path,
    job_id: str | None,
    trainer_file: str | None = None,
    output_dir: str | None = None,
    results_dir: str | None = None,
) -> dict[str, str]:
    effective_job_id = job_id or "local"
    job_dir = (playpen_dir / ".job_env" / effective_job_id).resolve()
    context = {
        "job_dir": str(job_dir),
        "model_registry_path": str(job_dir / "model_registry.json"),
    }
    trainer_path = resolve_trainer_path(root_dir, playpen_dir, trainer_file)
    if trainer_path is not None:
        context["trainer_file"] = trainer_path
    output_path = resolve_path(playpen_dir, output_dir)
    if output_path is not None:
        context["output_dir"] = output_path
    results_path = resolve_path(playpen_dir, results_dir)
    if results_path is not None:
        context["results_dir"] = results_path
    return context


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, required=True)
    parser.add_argument("--playpen-dir", type=Path, required=True)
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--trainer-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()

    context = resolve_job_context(
        root_dir=args.root_dir.resolve(),
        playpen_dir=args.playpen_dir.resolve(),
        job_id=args.job_id,
        trainer_file=args.trainer_file,
        output_dir=args.output_dir,
        results_dir=args.results_dir,
    )

    for key, value in context.items():
        print(f"{key.upper()}={shlex.quote(value)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
