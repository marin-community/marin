# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score the band ensemble's selection protocol against three corrected versions of itself.

The audit raises three separable objections to how the band is built, and they need to be measured
apart because two are correctness bugs and one is a leakage question with a knowable size.

**The solver is not exact.** ``stack_weights`` runs unconstrained nonnegative least squares on the
differenced predictions and then, if the coefficients sum above one, divides through. That projection
lands on the boundary of the simplex rather than at the constrained optimum, so the documented
guarantee -- that the combination cannot fit worse than its best member, because all-weight-on-the-
winner is feasible -- does not hold whenever the renormalization fires. This measures how often it
fires and what it costs against an exact simplex solve.

**The band width is not a standard error.** Membership is everything within one run-to-run standard
deviation of the best out-of-fold RMSE. Run noise is the spread of retraining the same mixture; it is
not the standard error of a *difference in cross-validated risk* between two configurations, which is
what "cannot be separated" should mean. The corrected width is that paired standard error, computed
on per-policy squared-error differences against the best configuration, with the policy row as the
resampling unit.

**Membership and weights are fitted on all rows.** They are estimated from every row's out-of-fold
prediction and then held fixed while only member heads are refitted inside the displayed folds, so a
held-out row helped choose the band that predicts it. The nested arm redoes screening, band formation
and weighting inside each outer fold. The gap between the two is the leakage, and reporting it is
more useful than asserting it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import nnls

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_hierarchical_coverage_grp_20260715 as bench,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    hierarchical_band_model_20260726 as band,
)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "reference_outputs" / "band_protocol_audit_20260727"
VARIANT = bench.Variant.HIERARCHICAL_PHASE_BUCKET_REPLAY
DATASETS = (
    bench.DatasetId.THREE_HUNDRED_M_UNCHEATABLE,
    bench.DatasetId.THREE_HUNDRED_M_TABLE9,
    bench.DatasetId.DELPHI_3E18_UNCHEATABLE,
    bench.DatasetId.DELPHI_3E18_TABLE9,
)
TARGET_ID = {
    "300m_uncheatable": "uncheatable",
    "300m_table9": "table9",
    "delphi_3e18_uncheatable": "uncheatable",
    "delphi_3e18_table9": "table9",
}
NESTED_SEED = 20260727
# Significance level for "this configuration's cross-validated risk is not distinguishable from the
# best one". Two-sided on a paired t statistic over per-policy squared-error differences.
BAND_ALPHA = 0.05
EPSILON = 1e-12


def exact_simplex_weights(predictions: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Least-squares weights on the probability simplex, solved exactly.

    Equality-constrained nonnegative least squares admits an exact reduction: append a heavily
    weighted row enforcing ``sum(w) == 1`` to a nonnegative least-squares problem in ``w`` directly.
    The penalty weight is scaled to the design so the constraint is satisfied to solver tolerance
    without dominating the conditioning, and the result is verified before it is returned.
    """
    if predictions.shape[1] == 1:
        return np.ones(1)
    finite = np.isfinite(observed) & np.isfinite(predictions).all(axis=1)
    design, truth = predictions[finite], observed[finite]
    penalty = 1e6 * max(float(np.abs(design).max()), 1.0)
    augmented = np.vstack([design, np.full((1, design.shape[1]), penalty)])
    augmented_truth = np.concatenate([truth, [penalty]])
    weights, _residual = nnls(augmented, augmented_truth, maxiter=400 * design.shape[1])
    total = weights.sum()
    assert abs(total - 1.0) < 1e-6, f"simplex constraint violated: sum={total}"
    return weights / total


def paired_band_half_width(
    predictions: dict[int, np.ndarray],
    observed: np.ndarray,
    best_index: int,
) -> float:
    """Half-width from the paired standard error of cross-validated risk differences.

    For each candidate, the per-policy squared-error difference against the best configuration has a
    paired standard error; a candidate is indistinguishable when a two-sided paired t test on that
    difference fails to reject. Converting the largest such tolerated risk gap back to RMSE units
    gives a width in the same units the caller expects, but derived from the variability of the
    comparison rather than from the variability of retraining a single mixture.
    """
    best_error = (predictions[best_index] - observed) ** 2
    best_rmse = float(np.sqrt(np.nanmean(best_error)))
    tolerated = 0.0
    for index, prediction in predictions.items():
        if index == best_index:
            continue
        difference = (prediction - observed) ** 2 - best_error
        difference = difference[np.isfinite(difference)]
        if difference.size < 3 or np.allclose(difference, difference[0]):
            continue
        result = stats.ttest_1samp(difference, 0.0)
        if result.pvalue > BAND_ALPHA:
            tolerated = max(tolerated, float(np.mean(difference)))
    return float(np.sqrt(max(best_rmse**2 + tolerated, 0.0)) - best_rmse)


def partial_oof_prediction(
    dataset,
    config: bench.Config,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Out-of-fold prediction over only the rows the splits cover, NaN elsewhere.

    The nested arm builds a band from a single outer fold's training rows, so the inner splits touch a
    subset of the panel. The benchmark's own helper demands full coverage and would reject that.
    """
    prediction = np.full(dataset.n, np.nan, dtype=float)
    for train, test in splits:
        prediction[test] = bench.fit_model(dataset, config, train).predict(dataset.weights[test])
    return prediction


def build_band_variant(
    dataset,
    configs: list[bench.Config],
    splits: list[tuple[np.ndarray, np.ndarray]],
    dataset_name: str,
    indices: np.ndarray,
    exact_solver: bool,
    calibrated_width: bool,
) -> tuple[band.BandModel, dict[str, float]]:
    """Band construction with the solver and the width each switchable independently."""
    observed = np.asarray(dataset.target, dtype=float)
    predictions = {index: partial_oof_prediction(dataset, config, splits) for index, config in enumerate(configs)}
    scored = sorted((float(np.sqrt(np.nanmean((predictions[i] - observed) ** 2))), i) for i in predictions)
    best_rmse, best_index = scored[0]
    if calibrated_width:
        half_width = paired_band_half_width(predictions, observed, best_index)
    else:
        half_width = band.band_half_width(TARGET_ID[dataset_name], best_rmse)
    inside = [(rmse, index) for rmse, index in scored if rmse <= best_rmse + half_width][: band.MAX_MEMBERS]
    stacked = np.column_stack([predictions[index] for _, index in inside])

    legacy = band.stack_weights(stacked, observed)
    weights = exact_simplex_weights(stacked, observed) if exact_solver else legacy
    # The documented guarantee is that the stack cannot fit worse than its best member. Checking it
    # directly is cheaper than reasoning about when the renormalization fires.
    member_rmse = np.array([np.sqrt(np.nanmean((stacked[:, i] - observed) ** 2)) for i in range(stacked.shape[1])])
    combined = float(np.sqrt(np.nanmean((stacked @ weights - observed) ** 2)))
    members = tuple(
        band.BandMember(config=configs[index], oof_rmse=rmse, weight=float(weight))
        for (rmse, index), weight in zip(inside, weights, strict=True)
    )
    model = band.BandModel(
        members=members,
        fitted=tuple(bench.fit_model(dataset, member.config, indices) for member in members),
        best_oof_rmse=best_rmse,
        band_half_width=half_width,
        n_candidates=len(configs),
    )
    detail = {
        "band_size": len(members),
        "active_members": model.active_members,
        "half_width": half_width,
        "half_width_over_best_rmse": half_width / max(best_rmse, EPSILON),
        "stacked_oof_rmse": combined,
        "best_member_oof_rmse": float(member_rmse.min()),
        "guarantee_violation": combined - float(member_rmse.min()),
        "renormalization_fired": float(abs(legacy.sum() - 1.0) > 1e-9 or legacy[0] <= EPSILON),
        "legacy_vs_exact_weight_gap": float(np.abs(legacy - exact_simplex_weights(stacked, observed)).max()),
    }
    return model, detail


def nested_band_rmse(
    dataset,
    dataset_id,
    configs: list[bench.Config],
    dataset_name: str,
    exact_solver: bool,
    calibrated_width: bool,
) -> dict[str, float | int]:
    """Screening, band membership and stacking weights all refitted inside each outer fold."""
    outer = bench.split_indices(dataset, dataset_id, np.arange(dataset.n), NESTED_SEED)
    prediction = np.full(dataset.n, np.nan, dtype=float)
    for train, test in outer:
        inner = bench.split_indices(dataset, dataset_id, train, bench.SCREEN_SEED)
        model, _detail = build_band_variant(
            dataset,
            configs,
            inner,
            dataset_name,
            train,
            exact_solver=exact_solver,
            calibrated_width=calibrated_width,
        )
        prediction[test] = model.predict(dataset.weights[test])
    if not np.isfinite(prediction).all():
        raise RuntimeError("Incomplete nested band prediction")
    return bench.metric_summary(dataset.target, prediction)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-shapes", type=int, default=6)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    shapes = bench.family_grp.shape_candidates(bench.family_grp.Variant.BUCKET_RESOLVED, args.num_shapes)
    configs = [
        bench.Config(VARIANT, index, shape, l2, residual, 0.0, 0.0)
        for index, shape in enumerate(shapes)
        for l2 in bench.L2_GRID
        for residual in bench.RESIDUAL_SHRINK_GRID
    ]
    arms = (
        ("current", False, False),
        ("exact_solver", True, False),
        ("calibrated_width", False, True),
        ("exact+calibrated", True, True),
    )

    rows = []
    for dataset_id in DATASETS:
        dataset = bench.load_dataset(dataset_id)
        name = dataset_id.value
        splits = bench.split_indices(dataset, dataset_id, np.arange(dataset.n), bench.SCREEN_SEED)
        all_rows = np.arange(dataset.n)
        print(f"\n{name}")
        for arm, exact_solver, calibrated_width in arms:
            _model, detail = build_band_variant(dataset, configs, splits, name, all_rows, exact_solver, calibrated_width)
            nested = nested_band_rmse(dataset, dataset_id, configs, name, exact_solver, calibrated_width)
            rows.append({"dataset": name, "arm": arm, **detail, **{f"nested_{k}": v for k, v in nested.items()}})
            print(
                f"  {arm:<18} band {detail['band_size']:>2} ({detail['active_members']} active)  "
                f"half-width {detail['half_width']:.6f}  stacked {detail['stacked_oof_rmse']:.6f}  "
                f"violation {detail['guarantee_violation']:+.2e}  nested rmse {nested['rmse']:.6f}"
            )

    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / "band_protocol_scores.csv", index=False)

    print("\n" + "=" * 100)
    print("WHAT EACH CORRECTION CHANGES")
    print("=" * 100)
    fired = table[table["arm"] == "current"]["renormalization_fired"].sum()
    print(f"\n  Renormalization fired in {int(fired)}/{len(DATASETS)} cells under the current solver.")
    worst = table[table["arm"] == "current"]["guarantee_violation"].max()
    print(f"  Worst 'cannot fit worse than best member' violation under the current solver: {worst:+.3e}")
    for dataset_name, group in table.groupby("dataset"):
        current = group[group["arm"] == "current"].iloc[0]
        print(f"\n  {dataset_name}")
        print(
            f"    post-selection stacked rmse {current['stacked_oof_rmse']:.6f}  ->  "
            f"nested {current['nested_rmse']:.6f}  (leakage {current['stacked_oof_rmse'] - current['nested_rmse']:+.6f})"
        )
        for _, row in group.iterrows():
            print(
                f"    {row['arm']:<18} half-width {row['half_width']:.6f} "
                f"({row['half_width_over_best_rmse'] * 100:.1f}% of best)  band {int(row['band_size']):>2}  "
                f"nested rmse {row['nested_rmse']:.6f}  nested regret@1 {row['nested_regret_at_1']:.6f}"
            )
    print(f"\nwrote {args.output_dir}")


if __name__ == "__main__":
    main()
