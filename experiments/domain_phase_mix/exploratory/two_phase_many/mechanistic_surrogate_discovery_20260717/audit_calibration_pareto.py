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
"""Audit tradeoffs between global calibration and optimum-region failures."""

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
BASELINE_METRICS = ARTIFACT_ROOT / "frozen_gate/baseline_metrics.csv"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "calibration_pareto_audit"
OBJECTIVES = (
    "rmse",
    "calibration_slope_error",
    "optimism_gt_0p05_count",
    "worst_optimism",
    "regret_at_1",
)


def pareto_mask(values: np.ndarray) -> np.ndarray:
    """Return points not weakly dominated on every lower-is-better objective."""
    result = np.ones(len(values), dtype=bool)
    for row_index, row in enumerate(values):
        weakly_better = np.all(values <= row, axis=1)
        strictly_better = np.any(values < row, axis=1)
        if np.any(weakly_better & strictly_better):
            result[row_index] = False
    return result


def main() -> None:
    gate.assert_sealed_absent(BASELINE_METRICS)
    DEFAULT_OUTPUT.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(BASELINE_METRICS)
    heldout = metrics.loc[
        metrics["swarm"].eq("delphi_3e18")
        & metrics["policy"].eq("two_phase")
        & metrics["split"].eq("heldout_policy_matched")
    ].copy()
    heldout["calibration_slope_error"] = (heldout["calibration_slope_observed_on_predicted"] - 1.0).abs()
    heldout["pareto_optimal"] = False
    for _, indices in heldout.groupby("target").groups.items():
        heldout.loc[indices, "pareto_optimal"] = pareto_mask(heldout.loc[indices, OBJECTIVES].to_numpy(dtype=float))
    heldout["frontier"] = heldout["pareto_optimal"].map({True: "nondominated", False: "dominated"})
    heldout.to_csv(DEFAULT_OUTPUT / "calibration_pareto_metrics.csv", index=False)

    figure = px.scatter(
        heldout,
        x="calibration_slope_error",
        y="rmse",
        color="worst_optimism",
        symbol="frontier",
        size="optimism_gt_0p05_count",
        facet_col="target",
        color_continuous_scale="RdYlGn_r",
        hover_name="model",
        hover_data={
            "calibration_slope_observed_on_predicted": ":.3f",
            "calibration_slope_error": ":.3f",
            "rmse": ":.4f",
            "regret_at_1": ":.4f",
            "optimism_gt_0p05_count": True,
            "worst_optimism": ":.4f",
        },
        labels={
            "calibration_slope_error": "Absolute calibration-slope error",
            "rmse": "Policy-matched heldout RMSE",
            "worst_optimism": "Worst optimism (BPB)",
        },
        title="Global calibration does not remove optimum-region optimism",
        height=620,
        width=1260,
    )
    figure.update_traces(marker={"line": {"color": "#173042", "width": 1.2}})
    figure.update_layout(
        template="plotly_white",
        margin={"l": 70, "r": 40, "t": 90, "b": 70},
        legend={"orientation": "h", "y": -0.18},
    )
    figure.write_html(
        DEFAULT_OUTPUT / "calibration_pareto.html",
        include_plotlyjs="cdn",
        config={"toImageButtonOptions": {"format": "png", "scale": 4}},
    )

    frontier = heldout.loc[heldout["pareto_optimal"]].sort_values(["target", "rmse"])
    report = f"""# Calibration Pareto audit

This audit treats policy-matched heldout RMSE, absolute observed-on-predicted slope error, count of optimism errors above 0.05 BPB, worst optimism, and Regret@1 as simultaneous lower-is-better objectives. It does not fit a calibrator or choose a new model.

## Nondominated frozen baselines

{frontier[["target", "model", *OBJECTIVES, "calibration_slope_observed_on_predicted"]].to_markdown(index=False, floatfmt=".6f")}

## Interpretation

No frozen baseline jointly solves calibration and extreme optimism. A slope close to one is not sufficient: inverse-deficit/log-link has the smallest Table-9 slope error ({heldout.loc[(heldout["target"].eq("table9")) & heldout["model"].eq("inverse_deficit_log_link"), "calibration_slope_error"].iloc[0]:.3f}) while retaining four >0.05-BPB optimism errors and worst optimism {heldout.loc[(heldout["target"].eq("table9")) & heldout["model"].eq("inverse_deficit_log_link"), "worst_optimism"].iloc[0]:.3f}. Conversely, reducing threshold crossings can coexist with poor global calibration or regret. This blocks selecting an output link from a single aggregate calibration statistic.
"""
    (DEFAULT_OUTPUT / "report.md").write_text(report)


if __name__ == "__main__":
    main()
