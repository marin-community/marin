# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Export a compact fit/heldout scorecard for the Delphi 3e18 Observatory swarm."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT = SCRIPT_DIR / "mixture_fit_debugger/src/generated/dashboard_data.json"
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/delphi_3e18_observatory_metrics_20260715"
FIELDS = (
    "target",
    "model",
    "model_label",
    "split",
    "n",
    "rmse",
    "mae",
    "spearman",
    "regret_at_1",
    "fold_mean_regret_at_1",
    "lower_tail_optimism",
    "low_tail_rmse",
    "lower_tail_count",
)
SPLITS = ("fitOof", "heldout", "heldoutSinglePhase", "heldoutTwoPhase")


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def value(metric: Mapping[str, object], key: str) -> object:
    result = metric.get(key)
    return "" if result is None else result


def fmt(value: object, digits: int = 5) -> str:
    return "-" if value in (None, "") else f"{float(value):.{digits}f}"


def markdown_table(rows: Sequence[Sequence[object]], headers: Sequence[str]) -> str:
    output = [f"| {' | '.join(headers)} |", f"| {' | '.join('---' for _ in headers)} |"]
    output.extend(f"| {' | '.join(str(value) for value in row)} |" for row in rows)
    return "\n".join(output)


def main() -> None:
    bundle = json.loads(INPUT.read_text())
    swarm = bundle["swarms"]["delphi_3e18"]
    models = bundle["models"]
    rows: list[dict[str, object]] = []
    for target, policies in swarm["diagnostics"].items():
        diagnostics = policies["two_phase"]
        for model, model_metrics in diagnostics.items():
            for split in SPLITS:
                metric = model_metrics[split]
                rows.append(
                    {
                        "target": target,
                        "model": model,
                        "model_label": models[model]["label"],
                        "split": split,
                        "n": metric["n"],
                        "rmse": value(metric, "rmse"),
                        "mae": value(metric, "mae"),
                        "spearman": value(metric, "spearman"),
                        "regret_at_1": value(metric, "regretAt1"),
                        "fold_mean_regret_at_1": value(metric, "foldMeanRegretAt1"),
                        "lower_tail_optimism": value(metric, "lowerTailOptimism"),
                        "low_tail_rmse": value(metric, "lowTailRmse"),
                        "lower_tail_count": metric["lowerTailCount"],
                    }
                )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "model_metrics.csv", rows)
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "fit_rows": swarm["dataset"]["fitDesignCount"],
        "coordinate_disjoint_heldouts": swarm["dataset"]["heldoutCount"],
        "exact_coordinate_aliases": swarm["dataset"]["sharedAliasCount"],
        "model_count": len(models),
        "targets": list(swarm["targets"]),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    report = [
        "# Delphi 3e18 Observatory scorecard",
        "",
        (
            f"Models are fit on {summary['fit_rows']} swarm designs and projected onto "
            f"{summary['coordinate_disjoint_heldouts']} coordinate-disjoint historical validations. "
            f"The {summary['exact_coordinate_aliases']} exact-coordinate repeats remain visible in Observatory but "
            "are excluded from heldout metrics."
        ),
        "",
        (
            "The heldout archive is deliberately append-only and heterogeneous: it contains optimizer proposals, "
            "regularization sweeps, baselines, and repeats rather than an IID random sample. Heldout RMSE and "
            "Spearman diagnose transfer across explored interventions; they do not estimate population risk."
        ),
    ]
    for target in swarm["targets"]:
        report.extend(["", f"## {swarm['targets'][target]['label']}", ""])
        target_rows = [row for row in rows if row["target"] == target]
        by_model = {
            model: {row["split"]: row for row in target_rows if row["model"] == model}
            for model in sorted({str(row["model"]) for row in target_rows})
        }
        ranked = sorted(
            by_model,
            key=lambda model: float(by_model[model]["heldout"]["rmse"]),
        )
        report.append(
            markdown_table(
                [
                    (
                        models[model]["label"],
                        fmt(by_model[model]["fitOof"]["rmse"]),
                        fmt(by_model[model]["fitOof"]["spearman"], 3),
                        fmt(by_model[model]["heldout"]["rmse"]),
                        fmt(by_model[model]["heldout"]["spearman"], 3),
                        fmt(by_model[model]["heldout"]["regret_at_1"]),
                        fmt(by_model[model]["heldout"]["lower_tail_optimism"]),
                    )
                    for model in ranked
                ],
                (
                    "Model",
                    "OOF RMSE",
                    "OOF rho",
                    "Heldout RMSE",
                    "Heldout rho",
                    "Heldout regret@1",
                    "Heldout tail optimism",
                ),
            )
        )
    (OUTPUT_DIR / "report.md").write_text("\n".join(report) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
