# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "plotly",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Training-series block bootstrap for frozen 3e18 heldout calibration."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (
    freeze_baseline_gate as gate,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent
ARTIFACT_ROOT = RESEARCH_DIR / "reference_outputs/mechanistic_surrogate_discovery_20260717"
DEFAULT_PREDICTIONS = ARTIFACT_ROOT / "failure_atlas/heldout_failure_atlas.csv"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "heldout_calibration_bootstrap"
N_BOOTSTRAPS = 5000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstraps", type=int, default=N_BOOTSTRAPS)
    return parser.parse_args()


def slope(observed: np.ndarray, predicted: np.ndarray) -> float:
    centered = predicted - predicted.mean()
    denominator = float(centered @ centered)
    if denominator <= 1e-15:
        return float("nan")
    return float(centered @ (observed - observed.mean()) / denominator)


def diagnostics(frame: pd.DataFrame) -> dict[str, float]:
    observed = frame["observed"].to_numpy(dtype=float)
    predicted = frame["predicted"].to_numpy(dtype=float)
    optimism = observed - predicted
    selected = int(np.argmin(predicted))
    return {
        "rmse": float(np.sqrt(np.mean(np.square(optimism)))),
        "mean_optimism": float(np.mean(optimism)),
        "calibration_slope": slope(observed, predicted),
        "optimism_gt_0p05_rate": float(np.mean(optimism > 0.05)),
        "worst_optimism": float(np.max(optimism)),
        "regret_at_1": float(observed[selected] - observed.min()),
        "selected_optimism": float(optimism[selected]),
    }


def main() -> None:
    args = parse_args()
    gate.assert_sealed_absent(args.predictions)
    frame = pd.read_csv(args.predictions)
    frame = frame.loc[frame["mechanism"].eq("baseline")].copy()
    rng = np.random.default_rng(20260717)
    records: list[dict[str, float | str | int]] = []
    summaries: list[dict[str, float | str]] = []
    for dataset, panel in frame.groupby("dataset"):
        groups = {name: local for name, local in panel.groupby("training_series")}
        names = np.asarray(list(groups), dtype=object)
        point = diagnostics(panel)
        bootstrap_values = {metric: [] for metric in point}
        for bootstrap in range(args.bootstraps):
            sampled = rng.choice(names, size=len(names), replace=True)
            resampled = pd.concat(
                [groups[str(name)].assign(_bootstrap_copy=index) for index, name in enumerate(sampled)],
                ignore_index=True,
            )
            values = diagnostics(resampled)
            for metric, value in values.items():
                bootstrap_values[metric].append(value)
                records.append(
                    {
                        "dataset": dataset,
                        "bootstrap": bootstrap,
                        "metric": metric,
                        "value": value,
                    }
                )
        for metric, values in bootstrap_values.items():
            array = np.asarray(values)
            reference = 1.0 if metric == "calibration_slope" else 0.0
            summaries.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "point": point[metric],
                    "bootstrap_mean": float(array.mean()),
                    "ci_2p5": float(np.quantile(array, 0.025)),
                    "ci_97p5": float(np.quantile(array, 0.975)),
                    "reference": reference,
                    "probability_above_reference": float(np.mean(array > reference)),
                }
            )
    records_frame = pd.DataFrame(records)
    summary = pd.DataFrame(summaries)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_frame.to_csv(args.output_dir / "bootstrap_records.csv", index=False)
    summary.to_csv(args.output_dir / "bootstrap_summary.csv", index=False)

    plot = records_frame.loc[
        records_frame["metric"].isin(("calibration_slope", "mean_optimism", "worst_optimism"))
    ].copy()
    figure = px.box(
        plot,
        x="dataset",
        y="value",
        color="dataset",
        facet_col="metric",
        facet_col_wrap=1,
        points=False,
        color_discrete_sequence=px.colors.diverging.RdYlGn_r,
        title="Training-series block-bootstrap uncertainty for frozen heldout calibration",
    )
    figure.update_yaxes(matches=None)
    figure.update_layout(height=1050, width=1200, showlegend=False)
    figure.write_html(
        args.output_dir / "heldout_calibration_bootstrap.html",
        include_plotlyjs="cdn",
        config={"toImageButtonOptions": {"scale": 4}},
    )

    (args.output_dir / "report.md").write_text(
        "# Frozen heldout calibration block bootstrap\n\n"
        "The 259 heldouts come from 28 training series, including dense hyperparameter paths. Resampling rows "
        "independently would overstate precision, so this audit samples entire training series with replacement. "
        "The model and hyperparameters remain fixed.\n\n" + summary.to_markdown(index=False, floatfmt=".6f") + "\n"
    )


if __name__ == "__main__":
    main()
