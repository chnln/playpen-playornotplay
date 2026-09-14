#!/usr/bin/env python3

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


NUMERIC_METRICS = {
    "Aborted",
    "Guess Repetitions",
    "Lose",
    "Main Score",
    "Parsed Request Count",
    "Played",
    "Request Count",
    "Request Success Ratio",
    "Success",
    "Violated Request Count",
}


def read_val_metrics(result_dir: Path) -> dict[str, float]:
    for path in sorted(result_dir.glob("*.val.json")):
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def read_raw_rows(result_dir: Path) -> list[dict[str, str]]:
    raw_path = result_dir / "clem" / "raw.csv"
    with raw_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def coerce_float(value: str) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return float(value)


def aggregate_rows(rows: list[dict[str, str]]) -> dict[str, dict[tuple[str, str], dict[str, float]]]:
    per_game: dict[str, dict[tuple[str, str], dict[str, float]]] = defaultdict(dict)
    for row in rows:
        game = row["game"]
        episode_key = (row["experiment"], row["episode"])
        episode_metrics = per_game[game].setdefault(episode_key, {})
        metric = row["metric"]
        if metric not in NUMERIC_METRICS:
            continue
        value = coerce_float(row["value"])
        if value is not None:
            episode_metrics[metric] = value
    return per_game


def summarize_game(game: str, episodes: dict[tuple[str, str], dict[str, float]]) -> dict[str, float]:
    total = len(episodes)
    summary: dict[str, float] = {"episodes": float(total)}
    if total == 0:
        return summary

    def metric_mean(metric: str) -> float:
        values = [metrics[metric] for metrics in episodes.values() if metric in metrics]
        return sum(values) / len(values) if values else 0.0

    def metric_sum(metric: str) -> float:
        return sum(metrics.get(metric, 0.0) for metrics in episodes.values())

    summary.update(
        {
            "played_rate": metric_mean("Played"),
            "success_rate": metric_mean("Success"),
            "abort_count": metric_sum("Aborted"),
            "lose_count": metric_sum("Lose"),
            "mean_violated_requests": metric_mean("Violated Request Count"),
            "mean_request_success_ratio": metric_mean("Request Success Ratio"),
            "mean_guess_repetitions": metric_mean("Guess Repetitions"),
            "mean_main_score": metric_mean("Main Score"),
        }
    )
    return summary


def format_summary(name: str, summary: dict[str, float]) -> str:
    ordered_keys = [
        "episodes",
        "played_rate",
        "success_rate",
        "abort_count",
        "lose_count",
        "mean_violated_requests",
        "mean_request_success_ratio",
        "mean_guess_repetitions",
        "mean_main_score",
    ]
    parts = [f"{key}={summary[key]:.4f}" for key in ordered_keys if key in summary]
    return f"{name}: " + ", ".join(parts)


def list_failures(episodes: dict[tuple[str, str], dict[str, float]]) -> list[str]:
    lines = []
    for (experiment, episode), metrics in sorted(episodes.items()):
        if metrics.get("Success", 0.0) >= 1.0:
            continue
        parts = []
        for key in [
            "Played",
            "Aborted",
            "Lose",
            "Violated Request Count",
            "Request Success Ratio",
            "Guess Repetitions",
            "Main Score",
        ]:
            if key in metrics:
                parts.append(f"{key}={metrics[key]:.4f}")
        lines.append(f"{experiment}/{episode}: " + ", ".join(parts))
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", help="Path to a playpen eval result directory.")
    parser.add_argument("--show-failures", action="store_true")
    args = parser.parse_args()

    result_dir = Path(args.result_dir).resolve()
    val_metrics = read_val_metrics(result_dir)
    raw_rows = read_raw_rows(result_dir)
    per_game = aggregate_rows(raw_rows)

    print(f"result_dir={result_dir}")
    if val_metrics:
        for key, value in sorted(val_metrics.items()):
            print(f"{key}={value}")

    for game in sorted(per_game):
        summary = summarize_game(game, per_game[game])
        print(format_summary(game, summary))
        if args.show_failures:
            for line in list_failures(per_game[game]):
                print(f"  {line}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
