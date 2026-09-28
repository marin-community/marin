# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2.0",
#   "pandas>=2.2",
#   "plotly>=6.0",
#   "scipy>=1.15",
#   "tabulate>=0.9",
# ]
# ///

"""Describe how much frozen heldout failure is concentrated in the extreme tail."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717.freeze_baseline_gate import (  # noqa: E402
    assert_sealed_absent,
)

DEFAULT_INPUT = (
    SCRIPT_DIR.parent
    / "reference_outputs"
    / "mechanistic_surrogate_discovery_20260717"
    / "final_synthesis"
    / "all_3e18_heldout_residuals.csv"
)
DEFAULT_OUTPUT = (
    SCRIPT_DIR.parent / "reference_outputs" / "mechanistic_surrogate_discovery_20260717" / "trimmed_calibration_audit"
)
TRIM_FRACTIONS = (0.0, 0.01, 0.02, 0.05, 0.10)
TARGET_LABELS = {
    "delphi_3e18_uncheatable": "Uncheatable",
    "delphi_3e18_table9": "Table-9",
}
TARGET_COLORS = {
    "delphi_3e18_uncheatable": "#1a9850",
    "delphi_3e18_table9": "#d73027",
}


def observed_on_predicted_slope(observed: np.ndarray, predicted: np.ndarray) -> float:
    variance = float(np.var(predicted))
    if variance == 0.0:
        return float("nan")
    return float(np.cov(predicted, observed, ddof=0)[0, 1] / variance)


def trimmed_metrics(frame: pd.DataFrame, trim_fraction: float) -> dict[str, float | int]:
    count_to_remove = int(np.floor(trim_fraction * len(frame)))
    ordered = frame.assign(absolute_residual=frame["optimism"].abs()).sort_values(
        ["absolute_residual", "row_id"], kind="stable"
    )
    retained = ordered.iloc[: len(ordered) - count_to_remove] if count_to_remove else ordered
    observed = retained["observed"].to_numpy(dtype=float)
    predicted = retained["predicted"].to_numpy(dtype=float)
    optimism = observed - predicted
    return {
        "trim_fraction": trim_fraction,
        "retained_rows": len(retained),
        "removed_rows": count_to_remove,
        "rmse": float(np.sqrt(np.mean(np.square(optimism)))),
        "bias_predicted_minus_observed": float(np.mean(predicted - observed)),
        "observed_on_predicted_slope": observed_on_predicted_slope(observed, predicted),
        "spearman": float(spearmanr(observed, predicted).statistic),
        "optimism_gt_0p05_count": int(np.sum(optimism > 0.05)),
        "worst_optimism": float(np.max(optimism)),
    }


def build_figure(metrics: pd.DataFrame) -> go.Figure:
    figure = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Heldout RMSE", "Observed-on-predicted slope", "Bias (predicted - observed)"),
    )
    for dataset, group in metrics.groupby("dataset", sort=False):
        label = TARGET_LABELS[dataset]
        color = TARGET_COLORS[dataset]
        for column, col in (("rmse", 1), ("observed_on_predicted_slope", 2), ("bias_predicted_minus_observed", 3)):
            figure.add_trace(
                go.Scatter(
                    x=100.0 * group["trim_fraction"],
                    y=group[column],
                    mode="lines+markers",
                    name=label,
                    legendgroup=dataset,
                    showlegend=col == 1,
                    line={"color": color, "width": 3},
                    marker={"size": 9},
                    customdata=np.column_stack((group["retained_rows"], group["optimism_gt_0p05_count"])),
                    hovertemplate=(
                        "trim=%{x:.0f}%<br>value=%{y:.5f}<br>retained=%{customdata[0]}"
                        "<br>optimism > 0.05=%{customdata[1]}<extra></extra>"
                    ),
                ),
                row=1,
                col=col,
            )
    figure.add_hline(y=1.0, line_dash="dash", line_color="#334155", row=1, col=2)
    figure.add_hline(y=0.0, line_dash="dash", line_color="#334155", row=1, col=3)
    figure.update_xaxes(title_text="Worst absolute residuals removed (%)")
    figure.update_layout(
        title=(
            "Frozen-heldout tail concentration<br>"
            "<sup>Descriptive only: trimming uses heldout outcomes and is not a model or deployment rule</sup>"
        ),
        template="plotly_white",
        width=1500,
        height=540,
        margin={"l": 75, "r": 35, "t": 100, "b": 70},
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.22},
    )
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    assert_sealed_absent(args.input)

    residuals = pd.read_csv(args.input)
    residuals = residuals.loc[residuals["mechanism"].eq("baseline")].copy()
    rows: list[dict[str, object]] = []
    for dataset, group in residuals.groupby("dataset", sort=False):
        for trim_fraction in TRIM_FRACTIONS:
            rows.append({"dataset": dataset, **trimmed_metrics(group, trim_fraction)})
    metrics = pd.DataFrame(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "trimmed_calibration_metrics.csv", index=False)
    figure = build_figure(metrics)
    figure.write_html(
        args.output_dir / "trimmed_calibration.html",
        include_plotlyjs="cdn",
        config={"toImageButtonOptions": {"format": "png", "scale": 4}},
    )
    report = [
        "# Frozen-heldout tail-concentration audit",
        "",
        "This is descriptive only. Rows are trimmed using heldout outcomes, so the result cannot select a model, fit a "
        "calibrator, or define a deployment rule.",
        "",
        metrics.to_markdown(index=False, floatfmt=".6f"),
        "",
        "The bulk fit is materially better than the headline tail metrics: removing the worst 2% eliminates every "
        ">0.05-BPB optimism error and moves Uncheatable's slope close to one. Table-9 retains slope above one and "
        "positive bias. This concentration explains why ordinary ranking is strong while optimization is unsafe: the "
        "optimizer targets exactly the rare region where the structural law fails.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n")
    print(metrics.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
