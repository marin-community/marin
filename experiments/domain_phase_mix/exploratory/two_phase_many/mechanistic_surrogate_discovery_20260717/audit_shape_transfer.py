# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pandas",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Audit transfer and grid-boundary selection of initial nonlinear shapes."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (
    freeze_baseline_gate as gate,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent
DEFAULT_INPUT = RESEARCH_DIR / ("reference_outputs/mechanistic_surrogate_discovery_20260717/initial_screen")
DEFAULT_OUTPUT = RESEARCH_DIR / ("reference_outputs/mechanistic_surrogate_discovery_20260717/shape_transfer_audit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selections: list[dict[str, object]] = []
    grids: dict[tuple[str, str], set[float]] = {}
    for panel_dir in sorted(path for path in args.input_dir.iterdir() if path.is_dir()):
        selection_path = panel_dir / "selection.json"
        screen_path = panel_dir / "hyperparameter_screen.csv"
        if not selection_path.exists() or not screen_path.exists():
            continue
        gate.assert_sealed_absent(selection_path)
        gate.assert_sealed_absent(screen_path)
        screen = pd.read_csv(screen_path)
        for row in screen[["family", "parameters"]].drop_duplicates().itertuples(index=False):
            for parameter, value in json.loads(row.parameters).items():
                grids.setdefault((row.family, parameter), set()).add(float(value))
        selection = json.loads(selection_path.read_text())
        for family, record in selection.items():
            parameters = {name: float(value) for name, value in record["parameters"].items()}
            selections.append(
                {
                    "panel": panel_dir.name,
                    "family": family,
                    "config": record["config"],
                    "l2": float(record["l2"]),
                    "parameters": json.dumps(parameters, sort_keys=True),
                    "nonlinear_parameter_count": len(parameters),
                }
            )
    records = pd.DataFrame(selections)
    boundary_counts: list[int] = []
    boundary_names: list[str] = []
    for row in records.itertuples(index=False):
        parameters = json.loads(row.parameters)
        boundary_parameters: list[str] = []
        for parameter, value in parameters.items():
            grid = sorted(grids.get((row.family, parameter), {value}))
            if len(grid) > 1 and (value == grid[0] or value == grid[-1]):
                boundary_parameters.append(parameter)
        boundary_counts.append(len(boundary_parameters))
        boundary_names.append(";".join(boundary_parameters))
    records["boundary_parameter_count"] = boundary_counts
    records["boundary_parameters"] = boundary_names
    records["any_boundary"] = records["boundary_parameter_count"].gt(0)
    summaries: list[dict[str, object]] = []
    for family, group in records.groupby("family", sort=False):
        configs = Counter(group["config"])
        l2s = Counter(group["l2"])
        nonlinear = group.loc[group["nonlinear_parameter_count"].gt(0)]
        summaries.append(
            {
                "family": family,
                "panels": len(group),
                "unique_selected_configs": group["config"].nunique(),
                "modal_config": configs.most_common(1)[0][0],
                "modal_config_frequency": configs.most_common(1)[0][1] / len(group),
                "unique_selected_l2": group["l2"].nunique(),
                "modal_l2": l2s.most_common(1)[0][0],
                "modal_l2_frequency": l2s.most_common(1)[0][1] / len(group),
                "nonlinear_panels": len(nonlinear),
                "boundary_selection_frequency": float(nonlinear["any_boundary"].mean()) if len(nonlinear) else 0.0,
            }
        )
    summary = pd.DataFrame(summaries)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records.to_csv(args.output_dir / "selection_records.csv", index=False)
    summary.to_csv(args.output_dir / "transfer_summary.csv", index=False)
    (args.output_dir / "report.md").write_text(
        "# Nonlinear-shape transfer audit\n\n"
        "A grid-edge selection is not automatically wrong, but repeated edge changes across related panels "
        "show that the nonlinear mechanism is not identified at a transferable value.\n\n"
        + summary.to_markdown(index=False, floatfmt=".4f")
        + "\n\n## Selected settings\n\n"
        + records.to_markdown(index=False)
        + "\n"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
