# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

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
"""Quantify clustering of heldout residuals by candidate-generation series."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (
    freeze_baseline_gate as gate,
)

SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260717"
FAILURE_ATLAS = ARTIFACT_ROOT / "failure_atlas/heldout_failure_atlas.csv"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "series_residual_structure"
BOOTSTRAP_REPEATS = 10_000


def eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
    center = float(np.mean(values))
    total = float(np.sum(np.square(values - center)))
    if total <= 0.0:
        return 0.0
    between = 0.0
    for label in np.unique(labels):
        group = values[labels == label]
        between += len(group) * float(np.square(np.mean(group) - center))
    return between / total


def main() -> None:
    gate.assert_sealed_absent(FAILURE_ATLAS)
    DEFAULT_OUTPUT.mkdir(parents=True, exist_ok=True)
    atlas = pd.read_csv(FAILURE_ATLAS)
    baseline = atlas.loc[atlas["mechanism"].eq("baseline")].copy()
    rng = np.random.default_rng(20260717)
    summary_rows: list[dict[str, float | int | str]] = []
    series_frames: list[pd.DataFrame] = []
    for dataset, frame in baseline.groupby("dataset", sort=True):
        values = frame["optimism"].to_numpy(dtype=float)
        labels = frame["training_series"].astype(str).to_numpy()
        observed_eta = eta_squared(values, labels)
        null = np.asarray([eta_squared(values, rng.permutation(labels)) for _ in range(BOOTSTRAP_REPEATS)])
        series = (
            frame.groupby("training_series", as_index=False)
            .agg(
                n=("row_id", "size"),
                mean_optimism=("optimism", "mean"),
                rmse=("optimism", lambda residual: float(np.sqrt(np.mean(np.square(residual))))),
                worst_optimism=("optimism", "max"),
                optimism_gt_0p05_count=("optimism", lambda residual: int(np.sum(np.asarray(residual) > 0.05))),
            )
            .sort_values("mean_optimism", ascending=False)
        )
        series.insert(0, "dataset", dataset)
        series_frames.append(series)
        extreme = frame.loc[frame["optimism"].gt(0.05)]
        summary_rows.append(
            {
                "dataset": dataset,
                "n_rows": len(frame),
                "n_series": frame["training_series"].nunique(),
                "eta_squared_series": observed_eta,
                "permutation_p_value": float((1 + np.sum(null >= observed_eta)) / (1 + len(null))),
                "extreme_optimism_rows": len(extreme),
                "series_with_extreme_optimism": extreme["training_series"].nunique(),
                "largest_series_fraction": float(frame["training_series"].value_counts(normalize=True).max()),
            }
        )

    summary = pd.DataFrame(summary_rows)
    series_metrics = pd.concat(series_frames, ignore_index=True)
    summary.to_csv(DEFAULT_OUTPUT / "series_structure_summary.csv", index=False)
    series_metrics.to_csv(DEFAULT_OUTPUT / "series_metrics.csv", index=False)

    figure = px.scatter(
        series_metrics,
        x="mean_optimism",
        y="rmse",
        size="n",
        color="worst_optimism",
        facet_col="dataset",
        color_continuous_scale="RdYlGn_r",
        hover_name="training_series",
        hover_data=["optimism_gt_0p05_count"],
        title="Frozen residuals cluster by candidate-generation series",
        labels={
            "mean_optimism": "Mean optimism (observed - predicted BPB)",
            "rmse": "Within-series residual RMSE",
            "worst_optimism": "Worst optimism",
        },
        width=1250,
        height=620,
    )
    figure.add_vline(x=0.0, line_dash="dash", line_color="#607682")
    figure.update_layout(template="plotly_white", margin={"l": 70, "r": 40, "t": 90, "b": 65})
    figure.write_html(
        DEFAULT_OUTPUT / "series_residual_structure.html",
        include_plotlyjs="cdn",
        config={"toImageButtonOptions": {"format": "png", "scale": 4}},
    )

    worst_series = (
        series_metrics.sort_values(["dataset", "worst_optimism"], ascending=[True, False])
        .groupby("dataset", as_index=False)
        .head(8)
    )
    report = f"""# Candidate-series residual structure

The series label is never an admissible model input. This audit asks whether frozen errors cluster by the experiment that generated a candidate path, after the surrogate has seen only its policy. Eta-squared is the fraction of residual sum of squares explained by in-sample series means; its permutation test preserves the unbalanced series sizes.

{summary.to_markdown(index=False, floatfmt=".6f")}

## Series with the largest worst-case optimism

{worst_series.to_markdown(index=False, floatfmt=".6f")}

## Interpretation

Significant clustering does not justify a series lookup term. It means the existing fit panel does not independently vary the policy properties carried together by those candidate generators. The remedy is a causal intervention that breaks those bundles, not more output calibration.
"""
    (DEFAULT_OUTPUT / "report.md").write_text(report)


if __name__ == "__main__":
    main()
