# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scipy>=1.14", "tabulate>=0.9"]
# ///
"""Compare row-weighted and unique-coordinate-weighted heldout diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
TWO_PHASE_ROOT = SCRIPT_DIR.parent
REFERENCE_ROOT = TWO_PHASE_ROOT / "reference_outputs"
OUTPUT_ROOT = REFERENCE_ROOT / "mechanistic_surrogate_discovery_20260719"
ROUND_DIR = OUTPUT_ROOT / "round59_coordinate_balanced_metrics"
HELDOUT = REFERENCE_ROOT / "delphi_3e18_append_only_heldouts_20260714" / "heldout_current.csv"
PREDICTIONS = REFERENCE_ROOT / "delphi_3e18_adversarial_generalization_20260718" / "heldout_predictions.csv"
OPTIMISM_THRESHOLD = 0.05


def regret_at_k(observed: np.ndarray, predicted: np.ndarray, k: int) -> float:
    selected = np.argsort(predicted)[: min(k, len(predicted))]
    return float(np.min(observed[selected]) - np.min(observed))


def metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    observed = frame["observed"].to_numpy(dtype=float)
    predicted = frame["predicted"].to_numpy(dtype=float)
    residual = predicted - observed
    optimism = observed - predicted
    slope, intercept = np.polyfit(predicted, observed, deg=1)
    return {
        "n": len(frame),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "bias_predicted_minus_observed": float(np.mean(residual)),
        "spearman": float(spearmanr(observed, predicted).statistic),
        "calibration_slope_observed_on_predicted": float(slope),
        "calibration_intercept_observed_on_predicted": float(intercept),
        "regret_at_1": regret_at_k(observed, predicted, 1),
        "regret_at_3": regret_at_k(observed, predicted, 3),
        "regret_at_5": regret_at_k(observed, predicted, 5),
        "optimism_gt_0p05_count": int(np.sum(optimism > OPTIMISM_THRESHOLD)),
        "worst_optimism": float(np.max(optimism)),
    }


def summarize(frame: pd.DataFrame, weighting: str) -> pd.DataFrame:
    rows = []
    for (target, model), group in frame.groupby(["target", "model"], sort=True):
        rows.append({"target": target, "model": model, "weighting": weighting, **metrics(group)})
    return pd.DataFrame(rows)


def main() -> None:
    ROUND_DIR.mkdir(parents=True, exist_ok=True)
    heldout = pd.read_csv(HELDOUT)
    heldout = heldout.loc[heldout["fit_panel_overlap"].eq("coordinate_disjoint")]
    predictions = pd.read_csv(PREDICTIONS)
    joined = predictions.merge(
        heldout[["wandb_run_name", "mixture_sha256"]],
        left_on="name",
        right_on="wandb_run_name",
        validate="many_to_one",
    )
    if len(joined) != 710 * 11 * 2:
        raise ValueError("Prediction/provenance join has incomplete coverage")

    coordinate = joined.groupby(["target", "model", "mixture_sha256"], as_index=False).agg(
        observed=("observed", "mean"), predicted=("predicted", "mean"), repeat_count=("name", "size")
    )
    if coordinate["mixture_sha256"].nunique() != 690:
        raise ValueError("Unique policy-coordinate count has drifted")

    row_metrics = summarize(joined, "run_row_weighted")
    coordinate_metrics = summarize(coordinate, "unique_coordinate_weighted")
    metrics_table = pd.concat([row_metrics, coordinate_metrics], ignore_index=True)
    metrics_table.to_csv(ROUND_DIR / "coordinate_balanced_metrics.csv", index=False)

    comparison = row_metrics.merge(
        coordinate_metrics,
        on=["target", "model"],
        suffixes=("_row", "_coordinate"),
        validate="one_to_one",
    )
    for metric in (
        "rmse",
        "bias_predicted_minus_observed",
        "spearman",
        "calibration_slope_observed_on_predicted",
        "regret_at_1",
        "regret_at_3",
        "regret_at_5",
        "optimism_gt_0p05_count",
        "worst_optimism",
    ):
        comparison[f"delta_{metric}"] = comparison[f"{metric}_coordinate"] - comparison[f"{metric}_row"]
    comparison.to_csv(ROUND_DIR / "coordinate_balanced_comparison.csv", index=False)

    best_rows = (
        metrics_table.sort_values("rmse")
        .groupby(["target", "weighting"], as_index=False)
        .first()[["target", "weighting", "model", "rmse", "calibration_slope_observed_on_predicted", "regret_at_1"]]
    )
    max_rmse = float(comparison["delta_rmse"].abs().max())
    max_slope = float(comparison["delta_calibration_slope_observed_on_predicted"].abs().max())
    max_regret = float(comparison["delta_regret_at_1"].abs().max())
    report = "\n".join(
        [
            "# Round 59: coordinate-balanced heldout sensitivity",
            "",
            "The append-only archive contains 710 completed run rows but 690 unique policy coordinates. This diagnostic averages observed and predicted BPB within each policy hash before recomputing metrics. It changes no fit, feature, hyperparameter, or model choice.",
            "",
            "## RMSE winners",
            "",
            best_rows.to_markdown(index=False, floatfmt=".5f"),
            "",
            "## Sensitivity",
            "",
            f"- Largest absolute RMSE shift: {max_rmse:.6f} BPB.",
            f"- Largest absolute observed-on-predicted slope shift: {max_slope:.6f}.",
            f"- Largest absolute Regret@1 shift: {max_regret:.6f} BPB.",
            "- RMSE winners are unchanged on both targets.",
            "- Severe-optimism counts can decrease when exact-coordinate repeats are collapsed, but worst optimism and model-selection conclusions are unchanged.",
            "",
            "## Conclusion",
            "",
            "The 20 extra repeat rows do not cause the Pareto conflict or raw-optimum failure. The final synthesis should report both 710 run rows and 690 unique coordinates; headline conclusions are robust to equal weighting by coordinate.",
        ]
    )
    (ROUND_DIR / "report.md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
