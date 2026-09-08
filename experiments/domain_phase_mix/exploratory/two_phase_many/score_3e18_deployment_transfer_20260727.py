# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy"]
# ///
"""Rank surrogate families by how well they predict 3e18 outcomes, not by 300M fit.

Every model-selection decision in this project has been made on 300M out-of-fold error, and that
objective is exhausted: the fit ceiling is representation-limited, and a measured inversion runs the
wrong way. On the twelve candidates validated in July, the family with the *worst* 300M fit
(aggregate, 0.415) had the *best* 3e18 rank correlation (0.909), while the best-fitting family
(effective exposure, 0.897) transferred worst (0.797). Selecting on 300M error therefore actively
picks the weaker deployment predictor.

This pools every validated 3e18 candidate for which we recorded both a prediction and an outcome,
and scores each family by deployment rank correlation. Two properties matter more than the headline
number:

* Rank correlation is computed *within* an objective, because Uncheatable and Table-9 sit on
  different scales and pooling them would manufacture correlation from the between-target offset.
* Calibration is reported separately from ranking. A family can rank candidates correctly while
  predicting absolute BPB far off, which is exactly the fantasy-optimum failure mode. Ranking is
  what a proposer needs; calibration is what a stopping rule needs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "deployment_transfer_scores_20260727"

# Panels recording a prediction and the realized 3e18 outcome for the same candidate.
SOURCES = (
    ("bucket_family_power_heads_validation_results_20260715", "results.csv", "model"),
    ("decoupled_phase_information_low_epsilon_validation_results_20260712", "combined_uncheatable_paths.csv", "family"),
    ("bucket_family_power_heads_validation_results_20260715", "ranking.csv", "model"),
)
PREDICTED = "predicted_bpb"
OBSERVED = "observed_target_bpb"
MIN_CANDIDATES = 3
# Regularization strength parsed out of the candidate name. The validated panels vary a candidate's
# aggregate-KL coefficient (akl) and its phase-information budget (e), and those two knobs turn out
# to order the outcomes far better than any predicted BPB does.
# Order matters: "akl0" is a prefix of "akl0p05" and "akl0p1", so the longer tokens must be tried
# first or every regularized candidate is silently assigned the unregularized value.
TRUST_TOKENS = (("akl0p05", 0.05), ("akl0p1", 0.1), ("akl0", 0.0), ("raw", -1.0))


def load_pooled() -> pd.DataFrame:
    """Collect (family, objective, candidate, predicted, observed) rows across validation panels."""
    frames = []
    for directory, filename, family_column in SOURCES:
        path = REFERENCE_OUTPUTS / directory / filename
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if not {PREDICTED, OBSERVED, family_column, "candidate", "objective"} <= set(frame.columns):
            continue
        subset = frame[["candidate", "objective", family_column, PREDICTED, OBSERVED]].copy()
        subset = subset.rename(columns={family_column: "family"})
        subset["source"] = directory
        frames.append(subset.dropna(subset=[PREDICTED, OBSERVED]))
    if not frames:
        raise ValueError("No validation panel exposed both a prediction and an outcome")
    pooled = pd.concat(frames, ignore_index=True)
    # A candidate evaluated in two panels is one observation, not two.
    return pooled.drop_duplicates(["candidate", "objective", "family"]).reset_index(drop=True)


def score(pooled: pd.DataFrame) -> pd.DataFrame:
    """Per family and objective: deployment rank correlation, plus calibration reported apart."""
    rows = []
    for (family, objective), group in pooled.groupby(["family", "objective"]):
        if len(group) < MIN_CANDIDATES or group[OBSERVED].nunique() < 3:
            continue
        predicted = group[PREDICTED].to_numpy()
        observed = group[OBSERVED].to_numpy()
        spearman = stats.spearmanr(predicted, observed).statistic
        pearson = stats.pearsonr(predicted, observed).statistic
        # Rank correlation of a constant-shifted prediction is unchanged, so report the offset and
        # the slope separately rather than folding them into a single "accuracy" number.
        slope = float(np.polyfit(predicted, observed, 1)[0])
        rows.append(
            {
                "family": family,
                "objective": objective,
                "n": len(group),
                "deployment_spearman": spearman,
                "deployment_pearson": pearson,
                "calibration_slope": slope,
                "mean_offset_bpb": float(np.mean(observed - predicted)),
                "best_selected_observed": float(observed[np.argmin(predicted)]),
                "best_available_observed": float(observed.min()),
                "selection_regret_bpb": float(observed[np.argmin(predicted)] - observed.min()),
            }
        )
    return pd.DataFrame(rows).sort_values(["objective", "deployment_spearman"], ascending=[True, False])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pooled = load_pooled()
    scores = score(pooled)
    pooled.to_csv(args.output_dir / "pooled_predictions.csv", index=False)
    scores.to_csv(args.output_dir / "deployment_transfer_scores.csv", index=False)

    print(f"pooled {len(pooled)} validated candidates across {pooled['family'].nunique()} families\n")
    print(scores.to_string(index=False))

    print("\n" + "=" * 96)
    print("TRUST-REGION ORDERING: does regularization strength order outcomes better than the model does?")
    print("=" * 96)
    trust = pooled.copy()
    trust["trust"] = np.nan
    for token, value in TRUST_TOKENS:
        mask = trust["candidate"].str.contains(f"_{token}", regex=False) & trust["trust"].isna()
        trust.loc[mask, "trust"] = value
    trust = trust.dropna(subset=["trust"])
    for objective, group in trust.groupby("objective"):
        if len(group) < MIN_CANDIDATES:
            continue
        by_predicted = stats.spearmanr(group[PREDICTED], group[OBSERVED]).statistic
        by_trust = stats.spearmanr(group["trust"], group[OBSERVED]).statistic
        # The two rows carry opposite sign conventions: a lower predicted BPB means the model
        # prefers the candidate, whereas a higher trust value means more regularization. So a
        # negative first row is an inverted ranking, while a negative second row is a correct one.
        print(f"\n  {objective}  (n={len(group)})")
        print(
            f"    rank by the model's predicted BPB : spearman {by_predicted:+.3f}  "
            f"({'INVERTED' if by_predicted < 0 else 'correct'})"
        )
        print(
            f"    rank by trust-region strength     : spearman {by_trust:+.3f}  "
            f"({'more regularization is better' if by_trust < 0 else 'less is better'})"
        )
        best = group.loc[group[OBSERVED].idxmin()]
        worst_pick = group.loc[group[PREDICTED].idxmin()]
        print(f"    best observed        : {best['candidate']}  {best[OBSERVED]:.6f}")
        print(
            f"    model's own argmin   : {worst_pick['candidate']}  {worst_pick[OBSERVED]:.6f}"
            f"   (regret {worst_pick[OBSERVED] - best[OBSERVED]:+.4f} BPB)"
        )
        # Dropping the unregularized arm is the test that matters: does MORE regularization help
        # among candidates that are all regularized? It holds on Table-9 and not on Uncheatable.
        regularized = group[group["trust"] >= 0]
        if len(regularized) >= 4:
            result = stats.spearmanr(regularized["trust"], regularized[OBSERVED])
            verdict = "monotone benefit" if result.pvalue < 0.05 else "NOT established"
            print(
                f"    regularized only (raw dropped)    : spearman {result.statistic:+.3f}  "
                f"n={len(regularized)}  p={result.pvalue:.3f}  -> {verdict}"
            )
    print("\n  Predicted BPB ranks these ladders backwards: it is measuring how far the model has")
    print("  extrapolated, not how good the candidate is. Regularization strength does better, but a")
    print("  monotone benefit among already-regularized candidates is established only on Table-9.")
    print("  Neither result licenses a scalar rule for choosing radius before validation.")
    print(f"\nwrote {args.output_dir}")


if __name__ == "__main__":
    main()
