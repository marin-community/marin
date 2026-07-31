# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Paired heldout bootstrap for the closest nested collision candidate."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (
    freeze_baseline_gate as gate,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent
DEFAULT_PREDICTIONS = RESEARCH_DIR / (
    "reference_outputs/mechanistic_surrogate_discovery_20260717/round12_kish_collision/predictions.csv"
)
DEFAULT_OUTPUT = RESEARCH_DIR / ("reference_outputs/mechanistic_surrogate_discovery_20260717/collision_paired_bootstrap")
N_BOOTSTRAPS = 5000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstraps", type=int, default=N_BOOTSTRAPS)
    return parser.parse_args()


def slope(observed: np.ndarray, predicted: np.ndarray) -> float:
    centered = predicted - np.mean(predicted)
    denominator = float(np.dot(centered, centered))
    if denominator <= 1e-15:
        return float("nan")
    return float(np.dot(centered, observed - np.mean(observed)) / denominator)


def diagnostics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = observed - predicted
    selected = int(np.argmin(predicted))
    return {
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "mae": float(np.mean(np.abs(residual))),
        "slope_error": abs(slope(observed, predicted) - 1.0),
        "optimism_rate": float(np.mean(residual > 0.05)),
        "worst_optimism": float(np.max(residual)),
        "regret_at_1": float(observed[selected] - np.min(observed)),
    }


def summarize(values: np.ndarray, metric: str) -> dict[str, object]:
    return {
        "metric": metric,
        "candidate_minus_baseline_point": float(values[0]),
        "bootstrap_mean_delta": float(np.mean(values[1:])),
        "ci_2p5": float(np.quantile(values[1:], 0.025)),
        "ci_97p5": float(np.quantile(values[1:], 0.975)),
        "probability_candidate_lower": float(np.mean(values[1:] < 0.0)),
    }


def main() -> None:
    args = parse_args()
    gate.assert_sealed_absent(args.predictions)
    predictions = pd.read_csv(args.predictions)
    predictions = predictions.loc[predictions["split"].eq("heldout_policy_matched")]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    rng = np.random.default_rng(20260717)
    for dataset in ("delphi_3e18_uncheatable", "delphi_3e18_table9"):
        panel = predictions.loc[predictions["dataset"].eq(dataset)]
        baseline = panel.loc[panel["mechanism"].eq("baseline"), ["row_id", "observed", "predicted"]]
        candidate = panel.loc[panel["mechanism"].eq("within_phase_collision"), ["row_id", "observed", "predicted"]]
        paired = baseline.merge(candidate, on="row_id", suffixes=("_baseline", "_candidate"), validate="one_to_one")
        if not np.allclose(paired["observed_baseline"], paired["observed_candidate"]):
            raise ValueError(f"Observed values differ in paired predictions for {dataset}")
        observed = paired["observed_baseline"].to_numpy(dtype=float)
        base_predicted = paired["predicted_baseline"].to_numpy(dtype=float)
        candidate_predicted = paired["predicted_candidate"].to_numpy(dtype=float)
        base_point = diagnostics(observed, base_predicted)
        candidate_point = diagnostics(observed, candidate_predicted)
        deltas = {metric: [candidate_point[metric] - base_point[metric]] for metric in base_point}
        for bootstrap in range(args.bootstraps):
            indices = rng.integers(0, len(observed), size=len(observed))
            base = diagnostics(observed[indices], base_predicted[indices])
            cand = diagnostics(observed[indices], candidate_predicted[indices])
            for metric in deltas:
                delta = cand[metric] - base[metric]
                deltas[metric].append(delta)
                records.append({"dataset": dataset, "bootstrap": bootstrap, "metric": metric, "delta": delta})
        for metric, values in deltas.items():
            rows.append({"dataset": dataset, **summarize(np.asarray(values), metric)})
    summary = pd.DataFrame(rows)
    pd.DataFrame(records).to_csv(args.output_dir / "bootstrap_records.csv", index=False)
    summary.to_csv(args.output_dir / "bootstrap_summary.csv", index=False)
    (args.output_dir / "report.md").write_text(
        "# Paired heldout bootstrap: finite collision minus nested baseline\n\n"
        "Negative deltas favor the collision candidate. Resampling is paired by heldout row and never changes "
        "the model or hyperparameters.\n\n" + summary.to_markdown(index=False, floatfmt=".6f") + "\n"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
