# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Test whether one scalar exposure invariant can explain heldout optimism.

The isotonic correction is a diagnostic upper bound, not a candidate model: it
is fit directly to development-heldout residuals and is never used to select a
mixture.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (
    freeze_baseline_gate as gate,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent
DEFAULT_ATLAS = RESEARCH_DIR / (
    "reference_outputs/mechanistic_surrogate_discovery_20260717/failure_atlas/heldout_failure_atlas.csv"
)
DEFAULT_OUTPUT = RESEARCH_DIR / (
    "reference_outputs/mechanistic_surrogate_discovery_20260717/scalar_invariant_sufficiency"
)
INVARIANTS = (
    "support_distance",
    "max_epoch",
    "mean_literal_replay",
    "phase_tv",
    "aggregate_kl_to_proportional",
    "min_family_ratio",
    "mass_ratio_lt_0p25",
    "bucket_reverse_kl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", type=Path, default=DEFAULT_ATLAS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def isotonic_oof(values: np.ndarray, residual: np.ndarray, groups: np.ndarray, increasing: bool) -> np.ndarray:
    predictions = np.zeros(len(values), dtype=float)
    splits = min(5, len(np.unique(groups)))
    splitter = GroupKFold(n_splits=splits)
    for train, test in splitter.split(values, groups=groups):
        model = IsotonicRegression(increasing=increasing, out_of_bounds="clip")
        model.fit(values[train], residual[train])
        predictions[test] = model.predict(values[test])
    return predictions


def adjacent_contradictions(values: np.ndarray, residual: np.ndarray) -> tuple[int, float, float]:
    order = np.argsort(values)
    x = values[order]
    r = residual[order]
    width = np.diff(x)
    difference = np.abs(np.diff(r))
    if len(width) == 0:
        return 0, 0.0, 0.0
    near = width <= np.quantile(width, 0.25)
    opposite = np.signbit(r[:-1]) != np.signbit(r[1:])
    contradictory = near & opposite & (difference > 0.05)
    return int(np.sum(contradictory)), float(np.max(difference[near], initial=0.0)), float(np.median(difference[near]))


def main() -> None:
    args = parse_args()
    gate.assert_sealed_absent(args.atlas)
    atlas = pd.read_csv(args.atlas)
    atlas = atlas.loc[atlas["mechanism"].eq("baseline")].copy()
    rows: list[dict[str, object]] = []
    for dataset, panel in atlas.groupby("dataset", sort=False):
        residual = panel["optimism"].to_numpy(dtype=float)
        groups = panel["training_series"].fillna(panel["panel"]).to_numpy()
        raw_rmse = float(np.sqrt(np.mean(np.square(residual))))
        raw_count = int(np.sum(residual > 0.05))
        for invariant in INVARIANTS:
            values = panel[invariant].to_numpy(dtype=float)
            correlation = float(spearmanr(values, residual).statistic)
            increasing = correlation >= 0.0
            correction = isotonic_oof(values, residual, groups, increasing)
            corrected = residual - correction
            contradictions, max_adjacent_gap, median_adjacent_gap = adjacent_contradictions(values, residual)
            rows.append(
                {
                    "dataset": dataset,
                    "invariant": invariant,
                    "spearman_with_optimism": correlation,
                    "isotonic_direction": "increasing" if increasing else "decreasing",
                    "raw_residual_rmse": raw_rmse,
                    "oof_corrected_residual_rmse": float(np.sqrt(np.mean(np.square(corrected)))),
                    "relative_oof_corrected_rmse": float(np.sqrt(np.mean(np.square(corrected)))) / raw_rmse,
                    "raw_optimism_gt_0p05_count": raw_count,
                    "oof_corrected_optimism_gt_0p05_count": int(np.sum(corrected > 0.05)),
                    "near_adjacent_opposite_sign_gap_gt_0p05": contradictions,
                    "max_near_adjacent_residual_gap": max_adjacent_gap,
                    "median_near_adjacent_residual_gap": median_adjacent_gap,
                }
            )
    summary = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "scalar_invariant_diagnostics.csv", index=False)
    best = summary.sort_values(["dataset", "oof_corrected_residual_rmse"]).groupby("dataset", as_index=False).first()
    (args.output_dir / "report.md").write_text(
        "# Scalar-invariant sufficiency audit\n\n"
        "The isotonic residual correction is an intentionally inadmissible diagnostic upper bound: it uses "
        "development-heldout residuals and has no latent-state interpretation. Even that correction is evaluated "
        "out of fold by training series. A scalar with many near-value/opposite-sign residual pairs cannot be the "
        "missing mechanism.\n\n## Best scalar upper bounds\n\n"
        + best.to_markdown(index=False, floatfmt=".5f")
        + "\n\n## All invariants\n\n"
        + summary.to_markdown(index=False, floatfmt=".5f")
        + "\n"
    )
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
