# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2.0",
#   "pandas>=2.2",
# ]
# ///

"""Paired-bootstrap comparison of separate-target and shared-recency FPT."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = HERE.parent / "reference_outputs"
BASELINE_PATH = REFERENCE_OUTPUTS / "cross_session_phase_transport_20260723" / "heldout_predictions.csv"
CANDIDATE_PATH = REFERENCE_OUTPUTS / "cross_session_shared_recency_20260723" / "heldout_predictions.csv"
OUTPUT = REFERENCE_OUTPUTS / "cross_session_shared_recency_20260723" / "paired_bootstrap_vs_separate_fpt.csv"
TARGET_LABELS = {
    "uncheatable_bpb": "uncheatable",
    "table9_macro_bpb": "table9",
}
BOOTSTRAPS = 5000
SEED = 20260723


def rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((predicted - observed) ** 2)))


def slope(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.polyfit(predicted, observed, 1)[0])


def slope_error(observed: np.ndarray, predicted: np.ndarray) -> float:
    return abs(slope(observed, predicted) - 1.0)


def bootstrap_difference(
    observed: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    metric,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    point = metric(observed, candidate) - metric(observed, baseline)
    count = len(observed)
    values = np.empty(BOOTSTRAPS)
    for bootstrap in range(BOOTSTRAPS):
        indices = rng.integers(0, count, count)
        values[bootstrap] = metric(
            observed[indices],
            candidate[indices],
        ) - metric(
            observed[indices],
            baseline[indices],
        )
    return (
        point,
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
        float(np.mean(values < 0.0)),
    )


def main() -> None:
    baseline = pd.read_csv(BASELINE_PATH, low_memory=False)
    baseline = baseline.loc[baseline["model_id"] == "fpt_total_global"]
    candidate = pd.read_csv(CANDIDATE_PATH, low_memory=False)
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []
    for target, proposal_target in TARGET_LABELS.items():
        base = baseline.loc[
            (baseline["fit_target"] == target) & (baseline["proposal_target"].fillna("").astype(str) == proposal_target)
        ]
        for model_id, proposed in candidate.loc[
            (candidate["fit_target"] == target)
            & (candidate["proposal_target"].fillna("").astype(str) == proposal_target)
        ].groupby("model_id"):
            joined = base[["observation_fingerprint", "observed_target", "predicted_target"]].merge(
                proposed[["observation_fingerprint", "observed_target", "predicted_target"]],
                on="observation_fingerprint",
                suffixes=("_baseline", "_candidate"),
                validate="one_to_one",
            )
            if not np.allclose(
                joined["observed_target_baseline"],
                joined["observed_target_candidate"],
            ):
                raise ValueError("Observed target mismatch across paired predictions")
            observed = joined["observed_target_baseline"].to_numpy(float)
            baseline_prediction = joined["predicted_target_baseline"].to_numpy(float)
            candidate_prediction = joined["predicted_target_candidate"].to_numpy(float)
            for metric_name, metric in (
                ("rmse", rmse),
                ("slope_error", slope_error),
            ):
                point, low, high, probability = bootstrap_difference(
                    observed,
                    baseline_prediction,
                    candidate_prediction,
                    metric,
                    rng,
                )
                rows.append(
                    {
                        "target": target,
                        "candidate": model_id,
                        "metric": metric_name,
                        "n": len(joined),
                        "candidate_minus_separate_fpt": point,
                        "ci_2p5": low,
                        "ci_97p5": high,
                        "probability_candidate_better": probability,
                    }
                )
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
