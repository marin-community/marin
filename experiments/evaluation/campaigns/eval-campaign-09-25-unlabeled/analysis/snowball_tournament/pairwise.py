#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select a Snowball-lineage checkpoint by paired benchmark win rate.

Adapted from eval-policy-09-17/artifacts/generate_snowball_paired_win_rates.py.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

SCORE = re.compile(r"^([01](?:\.\d+)?)\s+\(s3://[^)]+\)$")


def is_candidate(model: str) -> bool:
    return model.rsplit("/", 1)[-1].lower().startswith(("snowball-", "grug-"))


def tracker_scores(path: Path) -> tuple[list[str], list[str], dict[str, dict[str, float]]]:
    """Read durable numeric scores for every Snowball or Grug tracker row."""
    lines = [line for line in path.read_text().splitlines() if line.startswith("|")]
    header = [cell.strip() for cell in lines[0].strip("|").split("|")]
    benchmarks = header[1:]
    candidates = []
    scores: dict[str, dict[str, float]] = {}
    for line in lines[3:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != len(header) or not is_candidate(cells[0]):
            continue
        if cells[0] in scores:
            raise ValueError(f"duplicate Snowball/Grug row in {path}: {cells[0]}")
        candidates.append(cells[0])
        scores[cells[0]] = {
            benchmark: float(match.group(1))
            for benchmark, value in zip(benchmarks, cells[1:], strict=True)
            if (match := SCORE.match(value)) is not None
        }
    if len(candidates) < 2:
        raise ValueError(f"fewer than two Snowball/Grug rows in {path}")
    return benchmarks, candidates, scores


def pairwise_rows(
    benchmarks: list[str], candidates: list[str], scores: dict[str, dict[str, float]]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Give each jointly scored benchmark one vote, with half a vote for a tie."""
    pairs = []
    for model in candidates:
        for opponent in candidates:
            if model == opponent:
                continue
            common = [name for name in benchmarks if name in scores[model] and name in scores[opponent]]
            if not common:
                raise ValueError(f"no paired scores for {model} and {opponent}")
            wins = sum(scores[model][name] > scores[opponent][name] for name in common)
            losses = sum(scores[model][name] < scores[opponent][name] for name in common)
            ties = len(common) - wins - losses
            pairs.append(
                {
                    "model": model,
                    "opponent": opponent,
                    "wins": wins,
                    "losses": losses,
                    "ties": ties,
                    "benchmarks": len(common),
                    "win_rate": (wins + 0.5 * ties) / len(common),
                }
            )
    summary = []
    for model in candidates:
        opponents = [row for row in pairs if row["model"] == model]
        wins = sum(int(row["wins"]) for row in opponents)
        losses = sum(int(row["losses"]) for row in opponents)
        ties = sum(int(row["ties"]) for row in opponents)
        summary.append(
            {
                "model": model,
                "mean_pairwise_win_rate": sum(float(row["win_rate"]) for row in opponents) / len(opponents),
                "pooled_win_rate": (wins + 0.5 * ties) / (wins + losses + ties),
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "comparisons": wins + losses + ties,
            }
        )
    summary.sort(
        key=lambda row: (-float(row["mean_pairwise_win_rate"]), -float(row["pooled_win_rate"]), str(row["model"]))
    )
    return pairs, summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    benchmarks, candidates, scores = tracker_scores(args.tracker)
    pairs, summary = pairwise_rows(benchmarks, candidates, scores)
    write_csv(args.output_dir / "snowball_pairwise_head_to_head.csv", pairs)
    write_csv(args.output_dir / "snowball_pairwise_summary.csv", summary)
    source_hash = hashlib.sha256(args.tracker.read_bytes()).hexdigest()
    winner = str(summary[0]["model"])
    top_rate = float(summary[0]["mean_pairwise_win_rate"])
    leaders = [
        str(row["model"])
        for row in summary
        if math.isclose(float(row["mean_pairwise_win_rate"]), top_rate, rel_tol=0, abs_tol=1e-12)
    ]
    selection = {
        "winner": winner,
        "joint_leaders": leaders,
        "tie_break": "pooled win rate, then model ID" if len(leaders) > 1 else None,
        "criterion": f"mean of {len(candidates) - 1} paired win rates, each over jointly scored benchmarks",
        "tracker_sha256": source_hash,
        "candidates": candidates,
        "benchmarks": benchmarks,
    }
    (args.output_dir / "snowball_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    ids = {model: f"S{index + 1}" for index, model in enumerate(candidates)}
    lines = [
        "# Snowball-lineage tournament",
        "",
        f"Top checkpoint: `{winner}`." if len(leaders) == 1 else f"Joint first: {', '.join(f'`{m}`' for m in leaders)}.",
        "",
        f"Source tracker SHA-256: `{source_hash}`. Numeric scores with durable result links are eligible; "
        "pending cells are excluded pairwise. Ties use the displayed tracker point estimates.",
        "",
        "| Rank | Model | Mean paired win rate | Pooled win rate | W-L-T | Paired comparisons |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        rank = 1 + sum(
            float(other["mean_pairwise_win_rate"]) > float(row["mean_pairwise_win_rate"]) + 1e-12 for other in summary
        )
        lines.append(
            f"| {rank} | {ids[str(row['model'])]} | {100 * float(row['mean_pairwise_win_rate']):.1f}% | "
            f"{100 * float(row['pooled_win_rate']):.1f}% | {row['wins']}-{row['losses']}-{row['ties']} | "
            f"{row['comparisons']} |"
        )
    lines += [
        "",
        "| Model | " + " | ".join(ids.values()) + " |",
        "| --- | " + " | ".join("---:" for _ in candidates) + " |",
    ]
    for model in candidates:
        values = []
        for opponent in candidates:
            if model == opponent:
                values.append("—")
                continue
            row = next(row for row in pairs if row["model"] == model and row["opponent"] == opponent)
            values.append(f"{100 * float(row['win_rate']):.1f}% ({row['benchmarks']})")
        lines.append(f"| {ids[model]} | " + " | ".join(values) + " |")
    lines += ["", "| ID | Model |", "| --- | --- |"]
    lines += [f"| {ids[model]} | {model} |" for model in candidates]
    lines += [
        "",
        "The matrix denominator in parentheses is the number of jointly scored benchmarks. Each benchmark "
        "receives equal weight within its pair. A win scores 1, a tie 0.5, and a loss 0. The selection "
        f"criterion averages each candidate's {len(candidates) - 1} pairwise rates, so an opponent with fewer completed "
        "benchmarks does not receive less weight solely for that reason.",
        "",
    ]
    (args.output_dir / "snowball_pairwise_win_rates.md").write_text("\n".join(lines))
    print(f"selected {winner}; tracker SHA-256 {source_hash}")


if __name__ == "__main__":
    main()
