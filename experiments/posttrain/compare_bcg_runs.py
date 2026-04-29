#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare multiple BCG paired-rubric summary files.

Example:
    uv run python experiments/posttrain/compare_bcg_runs.py \
      --run M1=experiments/posttrain/stage4_output/bcg_M1_seed_n10_gemini3flash/bcg_summary.json \
      --run M2=experiments/posttrain/stage4_output/bcg_M2_seed_n10_gemini3flash/bcg_summary.json \
      --run M3=experiments/posttrain/stage4_output/bcg_M3_seed_n10_gemini3flash/bcg_summary.json \
      --out-json experiments/posttrain/stage4_output/m1_m2_m3_gemini3flash_comparison.json \
      --out-md experiments/posttrain/stage4_output/m1_m2_m3_gemini3flash_report.md
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REGRESSION_THRESHOLD = 0.05


@dataclass(frozen=True)
class RunSpec:
    label: str
    path: Path


def parse_run(value: str) -> RunSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be LABEL=PATH")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("--run label cannot be empty")
    return RunSpec(label=label, path=Path(path))


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def per_point_by_key(summary: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    return {(row["pair_id"], int(row["tension_point_idx"])): row for row in summary["per_point"]}


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def round_float(value: float) -> float:
    return round(value, 4)


def build_comparison(run_specs: list[RunSpec]) -> dict[str, Any]:
    summaries = {run.label: load_summary(run.path) for run in run_specs}
    per_point = {label: per_point_by_key(summary) for label, summary in summaries.items()}
    shared_keys = sorted(set.intersection(*(set(rows) for rows in per_point.values())))
    baseline_label = run_specs[0].label

    pairwise_vs_baseline = {}
    for run in run_specs[1:]:
        jsr_deltas = []
        bjs_deltas = []
        weakest_deltas = []
        improved = 0
        regressed = 0
        for key in shared_keys:
            base = per_point[baseline_label][key]
            current = per_point[run.label][key]
            jsr_delta = current["joint_satisfaction_rate"] - base["joint_satisfaction_rate"]
            bjs_delta = current["balanced_joint_score"] - base["balanced_joint_score"]
            weakest_delta = current["weakest_marginal_score"] - base["weakest_marginal_score"]
            jsr_deltas.append(jsr_delta)
            bjs_deltas.append(bjs_delta)
            weakest_deltas.append(weakest_delta)
            if jsr_delta > REGRESSION_THRESHOLD:
                improved += 1
            elif jsr_delta < -REGRESSION_THRESHOLD:
                regressed += 1
        pairwise_vs_baseline[run.label] = {
            "baseline": baseline_label,
            "mean_jsr_delta": round_float(mean(jsr_deltas)),
            "mean_bjs_delta": round_float(mean(bjs_deltas)),
            "mean_weakest_delta": round_float(mean(weakest_deltas)),
            "n_improved": improved,
            "n_regressed": regressed,
            "n_unchanged": len(shared_keys) - improved - regressed,
        }

    rows = []
    for key in shared_keys:
        row: dict[str, Any] = {
            "pair_id": key[0],
            "tension_point_idx": key[1],
            "tension_name": per_point[baseline_label][key].get("tension_name"),
        }
        for run in run_specs:
            point = per_point[run.label][key]
            prefix = run.label
            row[f"{prefix}_jsr"] = point["joint_satisfaction_rate"]
            row[f"{prefix}_bjs"] = point["balanced_joint_score"]
            row[f"{prefix}_weakest"] = point["weakest_marginal_score"]
        rows.append(row)

    return {
        "runs": [{"label": run.label, "path": str(run.path)} for run in run_specs],
        "shared_points": len(shared_keys),
        "aggregate": {label: summary["aggregate"] for label, summary in summaries.items()},
        "pairwise_vs_baseline": pairwise_vs_baseline,
        "per_point": rows,
    }


def markdown_report(comparison: dict[str, Any]) -> str:
    lines = [
        "# BCG Gemini Flash Comparison",
        "",
        f"Shared points: {comparison['shared_points']}",
        "",
        "## Aggregate",
        "",
        "| run | mean JSR | mean BJS | mean weakest |",
        "|---|---:|---:|---:|",
    ]
    for label, aggregate in comparison["aggregate"].items():
        lines.append(
            "| {label} | {jsr:.4f} | {bjs:.4f} | {weakest:.4f} |".format(
                label=label,
                jsr=aggregate.get("mean_joint_satisfaction", 0.0),
                bjs=aggregate.get("mean_balanced_joint_score", 0.0),
                weakest=aggregate.get("mean_weakest_marginal", 0.0),
            )
        )

    lines.extend(
        [
            "",
            "## Pairwise vs Baseline",
            "",
            "| run | baseline | delta JSR | delta BJS | delta weakest | improved | regressed | unchanged |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label, row in comparison["pairwise_vs_baseline"].items():
        lines.append(
            "| {label} | {baseline} | {mean_jsr_delta:+.4f} | {mean_bjs_delta:+.4f} | "
            "{mean_weakest_delta:+.4f} | {n_improved} | {n_regressed} | {n_unchanged} |".format(
                label=label,
                **row,
            )
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=parse_run, required=True, help="LABEL=PATH; pass at least two")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.run) < 2:
        raise ValueError("pass at least two --run entries")
    comparison = build_comparison(args.run)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(markdown_report(comparison), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
