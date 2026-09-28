# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2.0",
#   "pandas>=2.2",
#   "scipy>=1.14",
#   "tabulate>=0.9",
# ]
# ///

"""Independently reproduce final archive and adversarial decision metrics."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
TWO_PHASE_ROOT = SCRIPT_DIR.parent
OUTPUT_ROOT = TWO_PHASE_ROOT / "reference_outputs/mechanistic_surrogate_discovery_20260719"
FROZEN_DIR = OUTPUT_ROOT / "frozen_gate"
FINAL_DIR = OUTPUT_ROOT / "final_synthesis"
ROUND_DIR = OUTPUT_ROOT / "round76_final_metric_reproduction"
DASHBOARD = TWO_PHASE_ROOT / "mixture_fit_debugger/src/generated/dashboard_data.json"
BASELINE_MODELS = (
    "canonical",
    "effective_exposure",
    "effective_exposure_geometry",
    "separate_heads",
    "grp",
    "compact_retained_state",
    "bucket_family_grp",
    "hierarchical_phase_bucket_replay",
    "bucket_family_power_separate_heads",
)
LOWER_TAIL_FRACTION = 0.15
LOWER_TAIL_MIN_COUNT = 5
OPTIMISM_THRESHOLD = 0.05
FLOAT_TOLERANCE = 5e-12


def regret_at_k(observed: np.ndarray, predicted: np.ndarray, k: int) -> float:
    selected = np.argsort(predicted)[: min(k, len(predicted))]
    return float(np.min(observed[selected]) - np.min(observed))


def independent_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    if len(observed) < 3 or len(observed) != len(predicted):
        raise ValueError(f"Invalid metric vectors: observed={len(observed)}, predicted={len(predicted)}")
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError("Metric vectors must be finite")
    residual = predicted - observed
    optimism = observed - predicted
    predicted_centered = predicted - predicted.mean()
    observed_centered = observed - observed.mean()
    slope = float(np.dot(predicted_centered, observed_centered) / np.dot(predicted_centered, predicted_centered))
    intercept = float(observed.mean() - slope * predicted.mean())
    tail_count = min(len(observed), max(LOWER_TAIL_MIN_COUNT, math.ceil(LOWER_TAIL_FRACTION * len(observed))))
    tail = np.argsort(predicted)[:tail_count]
    selected = int(np.argmin(predicted))
    return {
        "n": len(observed),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "spearman": float(spearmanr(observed, predicted).statistic),
        "bias_predicted_minus_observed": float(np.mean(residual)),
        "calibration_slope_observed_on_predicted": slope,
        "calibration_intercept_observed_on_predicted": intercept,
        "regret_at_1": regret_at_k(observed, predicted, 1),
        "regret_at_3": regret_at_k(observed, predicted, 3),
        "regret_at_5": regret_at_k(observed, predicted, 5),
        "lower_tail_optimism": float(np.mean(np.maximum(optimism[tail], 0.0))),
        "low_tail_rmse": float(np.sqrt(np.mean(residual[tail] ** 2))),
        "optimism_gt_0p05_count": int(np.sum(optimism > OPTIMISM_THRESHOLD)),
        "worst_optimism": float(np.max(optimism)),
        "selected_optimism": float(optimism[selected]),
        "selected_observed": float(observed[selected]),
        "selected_predicted": float(predicted[selected]),
    }


def comparison_rows(
    panel: str,
    target: str,
    model: str,
    observed: np.ndarray,
    predicted: np.ndarray,
    frozen: pd.Series,
) -> list[dict[str, object]]:
    reproduced = independent_metrics(observed, predicted)
    rows: list[dict[str, object]] = []
    for metric, value in reproduced.items():
        frozen_value = float(frozen[metric])
        difference = float(value) - frozen_value
        tolerance = 0.0 if metric in {"n", "optimism_gt_0p05_count"} else FLOAT_TOLERANCE
        rows.append(
            {
                "panel": panel,
                "target": target,
                "model": model,
                "metric": metric,
                "reproduced": value,
                "frozen": frozen_value,
                "absolute_difference": abs(difference),
                "tolerance": tolerance,
                "passed": abs(difference) <= tolerance,
            }
        )
    return rows


def adversarial_comparisons() -> list[dict[str, object]]:
    predictions = pd.read_csv(FROZEN_DIR / "adversarial_target_matched_predictions.csv")
    frozen = pd.read_csv(FINAL_DIR / "adversarial_target_matched_metrics.csv").set_index(["target", "model"])
    rows: list[dict[str, object]] = []
    for (target, model), frame in predictions.groupby(["target", "model"], sort=True):
        rows.extend(
            comparison_rows(
                panel="exposed_target_matched_adversarial",
                target=str(target),
                model=str(model),
                observed=frame["observed"].to_numpy(dtype=float),
                predicted=frame["predicted"].to_numpy(dtype=float),
                frozen=frozen.loc[(target, model)],
            )
        )
    return rows


def archive_comparisons() -> list[dict[str, object]]:
    dashboard = json.loads(DASHBOARD.read_text())
    swarm = dashboard["swarms"]["delphi_3e18"]
    archive_mask = np.asarray(
        [row["split"] == "heldout" and not bool(row["isSharedAlias"]) for row in swarm["rows"]], dtype=bool
    )
    frozen = pd.read_csv(FINAL_DIR / "heldout_pareto_baseline.csv").set_index(["target", "model"])
    rows: list[dict[str, object]] = []
    for target in ("uncheatable", "table9"):
        observed = np.asarray([row["observed"][target] for row in swarm["rows"]], dtype=float)[archive_mask]
        models = swarm["predictions"][target]["two_phase"]
        for model in BASELINE_MODELS:
            if model not in models or (target, model) not in frozen.index:
                continue
            predicted = np.asarray(models[model]["prediction"], dtype=float)[archive_mask]
            rows.extend(
                comparison_rows(
                    panel="710_run_archive",
                    target=target,
                    model=model,
                    observed=observed,
                    predicted=predicted,
                    frozen=frozen.loc[(target, model)],
                )
            )
    return rows


def main() -> None:
    ROUND_DIR.mkdir(parents=True, exist_ok=True)
    comparisons = pd.DataFrame(adversarial_comparisons() + archive_comparisons())
    comparisons.to_csv(ROUND_DIR / "metric_reproduction.csv", index=False)
    failed = comparisons.loc[~comparisons["passed"]]
    summary = (
        comparisons.groupby(["panel", "target"], as_index=False)
        .agg(
            model_count=("model", "nunique"),
            metric_comparisons=("metric", "size"),
            maximum_absolute_difference=("absolute_difference", "max"),
            passed=("passed", "all"),
        )
        .sort_values(["panel", "target"])
    )
    summary.to_csv(ROUND_DIR / "summary.csv", index=False)
    if len(failed):
        raise AssertionError(f"{len(failed)} independently reproduced metrics differ from the frozen table")
    report = "\n".join(
        [
            "# Round 76: independent metric reproduction",
            "",
            "The decision-critical archive and exposed target-matched adversarial metrics were recomputed from "
            "row-level predictions with an independent implementation of calibration, regret, tail, and optimism "
            "definitions.",
            "",
            summary.to_markdown(index=False, floatfmt=".3g"),
            "",
            f"All {len(comparisons)} scalar comparisons passed at absolute tolerance {FLOAT_TOLERANCE:g}; integer "
            "counts matched exactly.",
            "This audit introduces no model, hyperparameter, or target-dependent choice and does not inspect sealed "
            "confirmation outcomes.",
        ]
    )
    (ROUND_DIR / "report.md").write_text(report + "\n")
    print(summary.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
