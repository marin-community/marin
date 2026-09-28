# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["joblib", "numpy", "pandas", "scikit-learn", "scipy", "tabulate"]
# ///

"""Benchmark the historical asymmetric-bowl response on complete OLMix swarms.

The two-phase separate-heads model reduces to one asymmetric bowl in a
single-phase swarm. Bucket centers are empirical training-fold medians plus one
shared shift; nonnegative under- and over-exposure curvatures are ridge fitted.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from scipy import optimize, stats

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_olmix_swarm_single_phase_dsp_20260901 as incumbent,
)

INCUMBENT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "olmix_swarm_single_phase_dsp_20260901"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "olmix_swarm_asymmetric_bowl_20260901"
SHARED_SHIFT_GRID = tuple(float(value) for value in np.linspace(-2.0, 4.0, 19))
RIDGE_GRID = incumbent.RIDGE_GRID
VARIANT = "asymmetric_bowl_shared_shift_macro"
COMPARATORS = (
    "linear_epoch_log_link",
    "dsp_benefit_log_link",
    "olmix_exact_macro",
)


@dataclasses.dataclass(frozen=True)
class BowlShape:
    shared_shift: float
    ridge: float


@dataclasses.dataclass(frozen=True)
class BowlHead:
    center: np.ndarray
    intercept: float
    coefficients: np.ndarray


def empirical_centers(exposure: np.ndarray, shared_shift: float) -> np.ndarray:
    positive = np.where(exposure > 1e-8, exposure, np.nan)
    log_exposure = np.log1p(positive)
    observed = np.any(np.isfinite(log_exposure), axis=0)
    base = np.full(exposure.shape[1], 2.0)
    base[observed] = np.nanmedian(log_exposure[:, observed], axis=0)
    return np.clip(base + shared_shift, -2.0, 8.0)


def asymmetric_bowl_design(exposure: np.ndarray, center: np.ndarray) -> np.ndarray:
    displacement = np.log1p(exposure) - center[None, :]
    return np.hstack(
        [
            np.minimum(displacement, 0.0) ** 2,
            np.maximum(displacement, 0.0) ** 2,
        ]
    )


def fit_nonnegative_ridge(design: np.ndarray, target: np.ndarray, ridge: float) -> tuple[float, np.ndarray]:
    design_mean = design.mean(axis=0)
    target_mean = float(target.mean())
    centered_design = design - design_mean[None, :]
    centered_target = target - target_mean
    if ridge > 0.0:
        width = design.shape[1]
        centered_design = np.vstack([centered_design, np.sqrt(ridge) * np.eye(width)])
        centered_target = np.concatenate([centered_target, np.zeros(width)])
    coefficients, _ = optimize.nnls(centered_design, centered_target, maxiter=300 * design.shape[1])
    intercept = target_mean - float(design_mean @ coefficients)
    return intercept, coefficients


def fit_bowl(
    exposure: np.ndarray,
    target: np.ndarray,
    shape: BowlShape,
) -> BowlHead:
    center = empirical_centers(exposure, shape.shared_shift)
    intercept, coefficients = fit_nonnegative_ridge(
        asymmetric_bowl_design(exposure, center),
        target,
        shape.ridge,
    )
    return BowlHead(center=center, intercept=intercept, coefficients=coefficients)


def predict_bowl(head: BowlHead, exposure: np.ndarray) -> np.ndarray:
    design = asymmetric_bowl_design(exposure, head.center)
    return head.intercept + design @ head.coefficients


def candidate_shapes() -> tuple[BowlShape, ...]:
    return tuple(BowlShape(shared_shift=shift, ridge=ridge) for shift in SHARED_SHIFT_GRID for ridge in RIDGE_GRID)


def select_shape(pool: incumbent.Pool, train: np.ndarray, seed: int) -> BowlShape:
    inner = incumbent.block_labels(pool.weights[train], incumbent.INNER_FOLDS, seed)
    target = pool.outcomes.mean(axis=1)
    scores: list[tuple[float, BowlShape]] = []
    for shape in candidate_shapes():
        squared_error = 0.0
        count = 0
        for fold in range(incumbent.INNER_FOLDS):
            fit_rows = train[inner != fold]
            test_rows = train[inner == fold]
            head = fit_bowl(pool.exposures[fit_rows], target[fit_rows], shape)
            prediction = predict_bowl(head, pool.exposures[test_rows])
            squared_error += float(np.sum((prediction - target[test_rows]) ** 2))
            count += len(test_rows)
        scores.append((squared_error / count, shape))
    return min(scores, key=lambda item: (item[0], item[1].ridge, abs(item[1].shared_shift), item[1].shared_shift))[1]


def benchmark_fold(
    pool: incumbent.Pool,
    repeat: int,
    fold: int,
    outer: np.ndarray,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = np.arange(len(pool.runs))
    train = rows[outer != fold]
    test = rows[outer == fold]
    target = pool.outcomes.mean(axis=1)
    shape = select_shape(pool, train, incumbent.FOLD_SEED + 1000 * repeat + fold)
    head = fit_bowl(pool.exposures[train], target[train], shape)
    prediction = predict_bowl(head, pool.exposures[test])
    fold_row: dict[str, object] = {
        "pool": pool.name,
        "variant": VARIANT,
        "repeat": repeat,
        "fold": fold,
        "test_rows": len(test),
        **dataclasses.asdict(shape),
        **incumbent.fold_scores(target[test], prediction),
    }
    prediction_rows = [
        {
            "pool": pool.name,
            "variant": VARIANT,
            "repeat": repeat,
            "fold": fold,
            "run": pool.runs[row],
            "index": int(row),
            "observed_macro_bpb": target[row],
            "predicted_macro_bpb": prediction[local],
        }
        for local, row in enumerate(test)
    ]
    return prediction_rows, fold_row


def benchmark_pool(pool: incumbent.Pool) -> tuple[pd.DataFrame, pd.DataFrame]:
    jobs = []
    for repeat in range(incumbent.OUTER_REPEATS):
        outer = incumbent.block_labels(pool.weights, incumbent.OUTER_FOLDS, incumbent.FOLD_SEED + 100 * repeat)
        for fold in range(incumbent.OUTER_FOLDS):
            jobs.append(delayed(benchmark_fold)(pool, repeat, fold, outer))
    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=incumbent.OUTER_WORKERS)(jobs)
    predictions = pd.DataFrame([row for result in results for row in result[0]])
    folds = pd.DataFrame([result[1] for result in results])
    return predictions, folds


def aggregate_metrics(predictions: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pool, group in predictions.groupby("pool", sort=False):
        repeat_scores = []
        for repeat, repeated in group.groupby("repeat"):
            score = incumbent.fold_scores(
                repeated.observed_macro_bpb.to_numpy(float),
                repeated.predicted_macro_bpb.to_numpy(float),
            )
            score["repeat"] = int(repeat)
            repeat_scores.append(score)
        frame = pd.DataFrame(repeat_scores)
        pool_folds = folds[folds.pool.eq(pool)]
        row: dict[str, object] = {"pool": pool, "variant": VARIANT}
        for metric in ("rmse", "mae", "spearman", "calibration_slope"):
            row[metric] = float(frame[metric].mean())
            row[f"{metric}_repeat_sd"] = float(frame[metric].std(ddof=1))
        row["mean_fold_selection_regret"] = float(pool_folds.selection_regret.mean())
        row["median_selected_shift"] = float(pool_folds.shared_shift.median())
        row["median_selected_ridge"] = float(pool_folds.ridge.median())
        rows.append(row)
    return pd.DataFrame(rows)


def corrected_contrasts(bowl_folds: pd.DataFrame, incumbent_folds: pd.DataFrame) -> pd.DataFrame:
    factor = 1.0 / (incumbent.OUTER_FOLDS * incumbent.OUTER_REPEATS) + 1.0 / (incumbent.OUTER_FOLDS - 1.0)
    rows = []
    for pool, group in bowl_folds.groupby("pool"):
        index = ["repeat", "fold"]
        bowl = group.set_index(index).rmse.sort_index()
        incumbent_pool = incumbent_folds[incumbent_folds.pool.eq(pool)]
        for comparator in COMPARATORS:
            baseline = incumbent_pool[incumbent_pool.variant.eq(comparator)].set_index(index).rmse.sort_index()
            if not bowl.index.equals(baseline.index):
                raise ValueError(f"{pool}/{comparator}: incumbent and bowl folds do not align")
            difference = bowl - baseline
            mean = float(difference.mean())
            se = float(np.sqrt(factor * difference.var(ddof=1)))
            critical = float(stats.t.ppf(0.975, len(difference) - 1))
            rows.append(
                {
                    "pool": pool,
                    "comparison": f"{VARIANT}_minus_{comparator}",
                    "mean_rmse_difference": mean,
                    "corrected_se": se,
                    "ci_low": mean - critical * se,
                    "ci_high": mean + critical * se,
                }
            )
    return pd.DataFrame(rows)


def write_report(
    output_dir: Path,
    aggregate: pd.DataFrame,
    incumbent_aggregate: pd.DataFrame,
    contrasts: pd.DataFrame,
    folds: pd.DataFrame,
) -> None:
    comparison = pd.concat(
        [
            aggregate,
            incumbent_aggregate[incumbent_aggregate.variant.isin(COMPARATORS)],
        ],
        ignore_index=True,
    ).sort_values(["pool", "rmse"])
    selected = comparison.set_index(["pool", "variant"])
    contrast_index = contrasts.set_index(["pool", "comparison"])
    dclm_comparison = f"{VARIANT}_minus_linear_epoch_log_link"
    high_quality_comparison = f"{VARIANT}_minus_linear_epoch_log_link"
    selected_above_historical_grid = int(np.sum(folds.shared_shift > 2.0))
    lines = [
        "# OLMix proxy-swarm asymmetric-bowl benchmark",
        "",
        "This tests the single-phase reduction of the historical separate-heads model: each bucket has "
        "nonnegative under- and over-exposure curvature around its empirical log-epoch center. The centers "
        "are recomputed inside every training fold and share one tuned shift. Ridge and shift selection use "
        "nested mixture-blocked folds; all reported predictions are outer-fold held out.",
        "",
        "The bowl is fitted directly to scalar macro BPB, matching its historical use. It therefore has "
        "roughly twice as many linear coefficients as buckets, while the incumbent exposure models fit 42 "
        "atomic-task heads and average their predictions. This is a model-package comparison, not an isolated "
        "test of response-link shape.",
        "",
        "## Verdict",
        "",
        "The asymmetric bowl is not a better surrogate on these swarms. On DCLM its RMSE is "
        f"{selected.loc[('dclm_10k', VARIANT), 'rmse']:.5f}, versus "
        f"{selected.loc[('dclm_10k', 'linear_epoch_log_link'), 'rmse']:.5f} for the linear exposure head; "
        "the point estimate favors the linear head, while the corrected difference remains inconclusive "
        f"([{contrast_index.loc[('dclm_10k', dclm_comparison), 'ci_low']:.5f}, "
        f"{contrast_index.loc[('dclm_10k', dclm_comparison), 'ci_high']:.5f}]). On High Quality its RMSE is "
        f"{selected.loc[('high_quality_10k', VARIANT), 'rmse']:.5f}, versus "
        f"{selected.loc[('high_quality_10k', 'linear_epoch_log_link'), 'rmse']:.5f}; the corrected interval "
        f"([{contrast_index.loc[('high_quality_10k', high_quality_comparison), 'ci_low']:.5f}, "
        f"{contrast_index.loc[('high_quality_10k', high_quality_comparison), 'ci_high']:.5f}]) again includes "
        "zero. High Quality selection regret nevertheless rises from "
        f"{selected.loc[('high_quality_10k', 'linear_epoch_log_link'), 'mean_fold_selection_regret']:.5f} to "
        f"{selected.loc[('high_quality_10k', VARIANT), 'mean_fold_selection_regret']:.5f}.",
        "",
        "The shared-shift search was widened from the historical upper limit of +2 to +4 as a sensitivity "
        f"check; {selected_above_historical_grid} of 50 outer fits selected a value above +2. The result is "
        "therefore not explained by the old center grid binding, and the extra flexibility does not rescue "
        "held-out performance.",
        "",
        "## Results",
        "",
        "| pool | model | RMSE | Spearman | mean fold regret | median ridge |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.pool} | {row.variant} | {row.rmse:.5f} | {row.spearman:.3f} | "
            f"{row.mean_fold_selection_regret:.5f} | {row.median_selected_ridge:.4g} |"
        )
    lines.extend(
        [
            "",
            "## Corrected RMSE contrasts",
            "",
            contrasts.to_markdown(index=False, floatfmt=".6f"),
            "",
            "Negative differences favor the asymmetric bowl. Intervals use the Nadeau-Bengio correction over "
            "the same 25 repeated blocked folds used by the incumbent benchmark.",
            "",
            "## Complexity",
            "",
            r"For \(m\) buckets, the model fits \(2m\) nonnegative curvature coefficients and one intercept; "
            "one shared center shift and ridge value are selected by inner validation. DCLM therefore has 237 "
            "linear coefficients and High Quality has 241, versus about 290 rows in each outer training split.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=INCUMBENT_OUTPUT_DIR / "input")
    parser.add_argument("--incumbent-output-dir", type=Path, default=INCUMBENT_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pools = tuple(incumbent.load_pool(args.input_dir, name) for name in incumbent.POOLS)
    results = [benchmark_pool(pool) for pool in pools]
    predictions = pd.concat([result[0] for result in results], ignore_index=True)
    folds = pd.concat([result[1] for result in results], ignore_index=True)
    aggregate = aggregate_metrics(predictions, folds)
    incumbent_aggregate = pd.read_csv(args.incumbent_output_dir / "aggregate_metrics.csv")
    incumbent_folds = pd.read_csv(args.incumbent_output_dir / "fold_metrics.csv")
    contrasts = corrected_contrasts(folds, incumbent_folds)

    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    folds.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    aggregate.to_csv(args.output_dir / "aggregate_metrics.csv", index=False)
    contrasts.to_csv(args.output_dir / "corrected_contrasts.csv", index=False)
    protocol = {
        "variant": VARIANT,
        "target": "mean BPB across the 42 OLMo Base-Easy tasks",
        "outer_folds": incumbent.OUTER_FOLDS,
        "outer_repeats": incumbent.OUTER_REPEATS,
        "inner_folds": incumbent.INNER_FOLDS,
        "fold_geometry": "KMeans on square-root mixture weights",
        "shared_shift_grid": list(SHARED_SHIFT_GRID),
        "ridge_grid": list(RIDGE_GRID),
        "inputs": {pool.name: pool.input_hashes for pool in pools},
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    write_report(args.output_dir, aggregate, incumbent_aggregate, contrasts, folds)


if __name__ == "__main__":
    main()
