# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare resampling units for the band's "configurations we cannot separate" threshold.

Replacing the band's run-sigma width with a paired standard error of cross-validated risk differences
tightened it by a factor of 2.6 to 6.8. That correction assumed the policy row is the resampling unit.
It is not obviously the right one: configurations inside a fold share a training set, so their errors
on a held-out row are correlated through that shared fit, and treating 280 rows as 280 independent
comparisons understates the standard error and therefore still leaves the band too wide.

The fold is the unit that respects that dependence, but there are only five of them, so a fold-level
test has almost no power and would widen the band by fiat rather than by evidence. Three estimators
are compared to see whether the choice actually matters: a row-level paired t test, a fold-level
paired t test on per-fold mean risk differences, and a cluster bootstrap that resamples whole folds
and so needs no distributional assumption at five units.

If the three agree, the row-level width stands and the question is closed. If the fold-level width is
much larger, the corrected band is still miscalibrated and the direction of the remaining error is
known.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_hierarchical_coverage_grp_20260715 as bench,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    evaluate_band_protocol_20260727 as protocol,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    hierarchical_band_model_20260726 as band,
)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "reference_outputs" / "band_width_unit_20260727"
BAND_ALPHA = 0.05
BOOTSTRAP_DRAWS = 4000
BOOTSTRAP_SEED = 20260727


def fold_of_row(dataset, dataset_id, indices: np.ndarray) -> np.ndarray:
    """Which held-out fold each policy row belongs to, so risk can be aggregated per fold."""
    splits = bench.split_indices(dataset, dataset_id, indices, bench.SCREEN_SEED)
    assignment = np.full(dataset.n, -1, dtype=int)
    for fold, (_train, test) in enumerate(splits):
        assignment[test] = fold
    return assignment


def widths(
    predictions: dict[int, np.ndarray],
    observed: np.ndarray,
    best_index: int,
    folds: np.ndarray,
) -> dict[str, float]:
    """Tolerated risk gap under each resampling unit, converted to an RMSE half-width."""
    best_error = (predictions[best_index] - observed) ** 2
    best_rmse = float(np.sqrt(np.nanmean(best_error)))
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    unique_folds = np.unique(folds[folds >= 0])
    tolerated = {"row_ttest": 0.0, "fold_ttest": 0.0, "fold_bootstrap": 0.0}
    for index, prediction in predictions.items():
        if index == best_index:
            continue
        difference = (prediction - observed) ** 2 - best_error
        valid = np.isfinite(difference)
        if valid.sum() < 3 or np.allclose(difference[valid], difference[valid][0]):
            continue
        mean_difference = float(np.mean(difference[valid]))

        if stats.ttest_1samp(difference[valid], 0.0).pvalue > BAND_ALPHA:
            tolerated["row_ttest"] = max(tolerated["row_ttest"], mean_difference)

        per_fold = np.asarray(
            [float(np.mean(difference[(folds == fold) & valid])) for fold in unique_folds],
            dtype=float,
        )
        per_fold = per_fold[np.isfinite(per_fold)]
        if per_fold.size >= 3 and not np.allclose(per_fold, per_fold[0]):
            if stats.ttest_1samp(per_fold, 0.0).pvalue > BAND_ALPHA:
                tolerated["fold_ttest"] = max(tolerated["fold_ttest"], mean_difference)
            draws = np.asarray(
                [np.mean(per_fold[generator.integers(0, per_fold.size, per_fold.size)]) for _ in range(BOOTSTRAP_DRAWS)]
            )
            low, high = np.quantile(draws, [0.025, 0.975])
            if low <= 0.0 <= high:
                tolerated["fold_bootstrap"] = max(tolerated["fold_bootstrap"], mean_difference)

    return {key: float(np.sqrt(max(best_rmse**2 + value, 0.0)) - best_rmse) for key, value in tolerated.items()} | {
        "best_rmse": best_rmse
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-shapes", type=int, default=6)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    shapes = bench.family_grp.shape_candidates(bench.family_grp.Variant.BUCKET_RESOLVED, args.num_shapes)
    configs = [
        bench.Config(protocol.VARIANT, index, shape, l2, residual, 0.0, 0.0)
        for index, shape in enumerate(shapes)
        for l2 in bench.L2_GRID
        for residual in bench.RESIDUAL_SHRINK_GRID
    ]

    rows = []
    for dataset_id in protocol.DATASETS:
        dataset = bench.load_dataset(dataset_id)
        name = dataset_id.value
        all_rows = np.arange(dataset.n)
        splits = bench.split_indices(dataset, dataset_id, all_rows, bench.SCREEN_SEED)
        observed = np.asarray(dataset.target, dtype=float)
        predictions = {index: bench.oof_prediction(dataset, config, splits) for index, config in enumerate(configs)}
        scored = sorted((float(np.sqrt(np.nanmean((predictions[i] - observed) ** 2))), i) for i in predictions)
        best_rmse, best_index = scored[0]
        folds = fold_of_row(dataset, dataset_id, all_rows)
        result = widths(predictions, observed, best_index, folds)
        run_sigma_width = band.band_half_width(protocol.TARGET_ID[name], best_rmse)
        sizes = {
            key: len([1 for rmse, _index in scored if rmse <= best_rmse + value])
            for key, value in result.items()
            if key != "best_rmse"
        }
        sizes["run_sigma"] = len([1 for rmse, _index in scored if rmse <= best_rmse + run_sigma_width])
        rows.append(
            {"dataset": name, "run_sigma_width": run_sigma_width, **result, **{f"size_{k}": v for k, v in sizes.items()}}
        )
        print(f"\n{name}   best OOF rmse {best_rmse:.6f}")
        print(
            f"  run sigma        width {run_sigma_width:.6f} "
            f"({run_sigma_width / best_rmse * 100:5.1f}% of best)  band {sizes['run_sigma']:>3}"
        )
        for key in ("row_ttest", "fold_ttest", "fold_bootstrap"):
            print(
                f"  {key:<16} width {result[key]:.6f} ({result[key] / best_rmse * 100:5.1f}% of best)  "
                f"band {sizes[key]:>3}   run-sigma is {run_sigma_width / max(result[key], 1e-12):.1f}x wider"
            )

    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / "band_width_units.csv", index=False)
    print("\n" + "=" * 96)
    print("Does the resampling unit change the conclusion that the run-sigma width is too wide?")
    print("=" * 96)
    for _, row in table.iterrows():
        ratios = [row["run_sigma_width"] / max(row[key], 1e-12) for key in ("row_ttest", "fold_ttest", "fold_bootstrap")]
        print(
            f"  {row['dataset']:<26} run-sigma is {min(ratios):.1f}x to {max(ratios):.1f}x wider "
            f"than a paired estimate, across all three units"
        )
    print(f"\nwrote {args.output_dir}")


if __name__ == "__main__":
    main()
