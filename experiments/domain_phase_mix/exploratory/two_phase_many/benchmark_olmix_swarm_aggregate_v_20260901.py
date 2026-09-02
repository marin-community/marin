# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["joblib", "numpy", "pandas", "scikit-learn", "scipy", "tabulate"]
# ///

"""Benchmark the direct-macro aggregate-V response on complete OLMix swarms.

The model is the single-phase aggregate-linear-V challenger used for Delphi:
each bucket contributes nonnegative under- and over-exposure hinges around an
empirical log-epoch center. It is fitted directly to scalar macro BPB.
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
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_olmix_swarm_asymmetric_bowl_20260901 as bowl,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_olmix_swarm_single_phase_dsp_20260901 as incumbent,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    swarm39_harness_20260725 as swarm39,
)

INCUMBENT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "olmix_swarm_single_phase_dsp_20260901"
BOWL_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "olmix_swarm_asymmetric_bowl_20260901"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "olmix_swarm_aggregate_v_20260901"
CENTER_SHIFT_GRID = tuple(float(value) for value in np.arange(-2.0, 8.5, 0.5))
RIDGE_GRID = tuple(sorted({0.0, 0.01, 0.1, 1.0, *incumbent.RIDGE_GRID}))
VARIANT = "aggregate_linear_v_macro"
COMPARATORS = (
    "linear_epoch_log_link",
    "dsp_benefit_log_link",
    "dsp_shared_task_log_link",
    bowl.VARIANT,
    "olmix_exact_macro",
)


@dataclasses.dataclass(frozen=True)
class AggregateVShape:
    center_shift: float
    ridge: float


@dataclasses.dataclass(frozen=True)
class AggregateVHead:
    center: np.ndarray
    intercept: float
    coefficients: np.ndarray


def empirical_centers(exposure: np.ndarray, center_shift: float) -> np.ndarray:
    """Return per-bucket positive-exposure medians on the log-epoch scale."""
    log_exposure = np.log1p(exposure)
    positive = np.where(exposure > 1e-8, log_exposure, np.nan)
    observed = np.any(np.isfinite(positive), axis=0)
    center = np.zeros(exposure.shape[1])
    center[observed] = np.nanmedian(positive[:, observed], axis=0)
    return np.clip(center + center_shift, -2.0, 8.0)


def aggregate_v_design(exposure: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Build positive under- and over-exposure hinges around each center."""
    displacement = np.log1p(exposure) - center[None, :]
    return np.hstack(
        [
            np.maximum(-displacement, 0.0),
            np.maximum(displacement, 0.0),
        ]
    )


def fit_aggregate_v(exposure: np.ndarray, target: np.ndarray, shape: AggregateVShape) -> AggregateVHead:
    center = empirical_centers(exposure, shape.center_shift)
    intercept, coefficients = swarm39.fit_head(
        aggregate_v_design(exposure, center),
        target,
        shape.ridge,
    )
    return AggregateVHead(center=center, intercept=intercept, coefficients=coefficients)


def predict_aggregate_v(head: AggregateVHead, exposure: np.ndarray) -> np.ndarray:
    return head.intercept + aggregate_v_design(exposure, head.center) @ head.coefficients


def candidate_shapes() -> tuple[AggregateVShape, ...]:
    return tuple(
        AggregateVShape(center_shift=center_shift, ridge=ridge)
        for center_shift in CENTER_SHIFT_GRID
        for ridge in RIDGE_GRID
    )


def select_shape(pool: incumbent.Pool, train: np.ndarray, seed: int) -> AggregateVShape:
    """Select center shift and ridge using only inner training-fold outcomes."""
    inner = incumbent.block_labels(pool.weights[train], incumbent.INNER_FOLDS, seed)
    target = pool.outcomes.mean(axis=1)
    scores: list[tuple[float, AggregateVShape]] = []
    for shape in candidate_shapes():
        squared_error = 0.0
        count = 0
        for fold in range(incumbent.INNER_FOLDS):
            fit_rows = train[inner != fold]
            test_rows = train[inner == fold]
            head = fit_aggregate_v(pool.exposures[fit_rows], target[fit_rows], shape)
            prediction = predict_aggregate_v(head, pool.exposures[test_rows])
            squared_error += float(np.sum((prediction - target[test_rows]) ** 2))
            count += len(test_rows)
        scores.append((squared_error / count, shape))
    return min(
        scores,
        key=lambda item: (item[0], item[1].ridge, abs(item[1].center_shift), item[1].center_shift),
    )[1]


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
    head = fit_aggregate_v(pool.exposures[train], target[train], shape)
    prediction = predict_aggregate_v(head, pool.exposures[test])
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
        outer = incumbent.block_labels(
            pool.weights,
            incumbent.OUTER_FOLDS,
            incumbent.FOLD_SEED + 100 * repeat,
        )
        for fold in range(incumbent.OUTER_FOLDS):
            jobs.append(delayed(benchmark_fold)(pool, repeat, fold, outer))
    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=incumbent.OUTER_WORKERS)(jobs)
    predictions = pd.DataFrame([row for result in results for row in result[0]])
    folds = pd.DataFrame([result[1] for result in results])
    return predictions, folds


def aggregate_metrics(predictions: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
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
        row["median_selected_center_shift"] = float(pool_folds.center_shift.median())
        row["median_selected_ridge"] = float(pool_folds.ridge.median())
        rows.append(row)
    return pd.DataFrame(rows)


def corrected_contrasts(aggregate_v_folds: pd.DataFrame, comparator_folds: pd.DataFrame) -> pd.DataFrame:
    factor = 1.0 / (incumbent.OUTER_FOLDS * incumbent.OUTER_REPEATS) + 1.0 / (incumbent.OUTER_FOLDS - 1.0)
    rows = []
    for pool, group in aggregate_v_folds.groupby("pool"):
        index = ["repeat", "fold"]
        aggregate_v = group.set_index(index).rmse.sort_index()
        comparator_pool = comparator_folds[comparator_folds.pool.eq(pool)]
        for comparator in COMPARATORS:
            baseline = comparator_pool[comparator_pool.variant.eq(comparator)].set_index(index).rmse.sort_index()
            if not aggregate_v.index.equals(baseline.index):
                raise ValueError(f"{pool}/{comparator}: comparator folds do not align")
            difference = aggregate_v - baseline
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
    comparator_aggregate: pd.DataFrame,
    contrasts: pd.DataFrame,
    folds: pd.DataFrame,
) -> None:
    comparison = pd.concat([aggregate, comparator_aggregate], ignore_index=True).sort_values(["pool", "rmse"])
    selected = comparison.set_index(["pool", "variant"])
    contrast_index = contrasts.set_index(["pool", "comparison"])
    lines = [
        "# OLMix proxy-swarm aggregate-V benchmark",
        "",
        "This is the exact response basis of the Delphi aggregate-linear-V challenger, evaluated on Michael "
        "Ryan's two complete OLMix proxy swarms. For each bucket, log materialized epochs enter through "
        "nonnegative under- and over-exposure hinges around an empirical center. The model predicts scalar "
        "macro BPB directly.",
        "",
        "Centers are recomputed inside every training fold. One shared center shift and ridge value are selected "
        "by nested mixture-blocked validation; all reported predictions use the incumbent benchmark's same 25 "
        "outer held-out folds.",
        "",
        "## Results",
        "",
        "| pool | model | RMSE | Spearman | mean fold regret |",
        "|---|---|---:|---:|---:|",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.pool} | {row.variant} | {row.rmse:.5f} | {row.spearman:.3f} | "
            f"{row.mean_fold_selection_regret:.5f} |"
        )
    lines.extend(["", "## Corrected RMSE contrasts", "", contrasts.to_markdown(index=False, floatfmt=".6f"), ""])
    for pool in incumbent.POOLS:
        aggregate_row = selected.loc[(pool, VARIANT)]
        linear_row = selected.loc[(pool, "linear_epoch_log_link")]
        comparison_name = f"{VARIANT}_minus_linear_epoch_log_link"
        interval = contrast_index.loc[(pool, comparison_name)]
        direction = "better" if aggregate_row.rmse < linear_row.rmse else "worse"
        lines.append(
            f"On `{pool}`, aggregate-V is {direction} in point-estimate RMSE "
            f"({aggregate_row.rmse:.5f} versus {linear_row.rmse:.5f}); the corrected difference is "
            f"[{interval.ci_low:.5f}, {interval.ci_high:.5f}]."
        )
    lower_shift = int(folds.center_shift.eq(min(CENTER_SHIFT_GRID)).sum())
    upper_shift = int(folds.center_shift.eq(max(CENTER_SHIFT_GRID)).sum())
    above_original_shift_grid = int(folds.center_shift.gt(2.0).sum())
    lower_ridge = int(folds.ridge.eq(min(RIDGE_GRID)).sum())
    upper_ridge = int(folds.ridge.eq(max(RIDGE_GRID)).sum())
    lines.extend(
        [
            "",
            "## Diagnostics",
            "",
            f"Across 50 outer fits, the center shift selected the lower/upper grid edge {lower_shift}/{upper_shift} "
            f"times and ridge selected the lower/upper edge {lower_ridge}/{upper_ridge} times. "
            f"The selected center shift exceeded the original Delphi upper limit of +2 in "
            f"{above_original_shift_grid} fits; this widened grid is a prespecified sensitivity to the first "
            "run's bound, not a change to the V-shaped response basis.",
            "",
            r"For \(m\) buckets, aggregate-V fits \(2m\) nonnegative slopes and one intercept. DCLM therefore "
            "has 237 linear coefficients and High Quality has 241, against about 290 mixtures in each outer "
            "training fold. The nested ridge choice is consequently part of the estimator, not an optional "
            "post-processing detail.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=INCUMBENT_OUTPUT_DIR / "input")
    parser.add_argument("--incumbent-output-dir", type=Path, default=INCUMBENT_OUTPUT_DIR)
    parser.add_argument("--bowl-output-dir", type=Path, default=BOWL_OUTPUT_DIR)
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
    bowl_aggregate = pd.read_csv(args.bowl_output_dir / "aggregate_metrics.csv")
    bowl_folds = pd.read_csv(args.bowl_output_dir / "fold_metrics.csv")
    comparator_aggregate = pd.concat(
        [
            incumbent_aggregate[incumbent_aggregate.variant.isin(COMPARATORS)],
            bowl_aggregate[bowl_aggregate.variant.eq(bowl.VARIANT)],
        ],
        ignore_index=True,
    )
    comparator_folds = pd.concat(
        [
            incumbent_folds[incumbent_folds.variant.isin(COMPARATORS)],
            bowl_folds[bowl_folds.variant.eq(bowl.VARIANT)],
        ],
        ignore_index=True,
    )
    contrasts = corrected_contrasts(folds, comparator_folds)

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
        "center_shift_grid": list(CENTER_SHIFT_GRID),
        "ridge_grid": list(RIDGE_GRID),
        "inputs": {pool.name: pool.input_hashes for pool in pools},
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    write_report(args.output_dir, aggregate, comparator_aggregate, contrasts, folds)


if __name__ == "__main__":
    main()
