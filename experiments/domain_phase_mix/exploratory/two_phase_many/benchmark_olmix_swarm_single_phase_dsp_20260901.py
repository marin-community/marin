# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["joblib", "numpy", "pandas", "scikit-learn", "scipy", "tabulate"]
# ///

"""Benchmark label-blind single-phase exposure models on complete OLMix swarms.

Every endpoint prediction is out of fold. Mixture geometry defines the folds,
and nonlinear shape selection is repeated inside each outer training split.
The primary target is the row-wise mean of all 42 OLMo Base-Easy tasks.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from scipy import optimize, stats
from sklearn.cluster import KMeans

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.olmix_loglinear_fit import (  # noqa: E402
    OlmixLoglinearFit,
    fit_olmix_loglinear_model,
)

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "olmix_swarm_single_phase_dsp_20260901"
DEFAULT_INPUT_DIR = DEFAULT_OUTPUT_DIR / "input"
POOLS = ("dclm_10k", "high_quality_10k")
PROXY_TOKENS = 4_304_928_768
OUTER_FOLDS = 5
OUTER_REPEATS = 5
OUTER_WORKERS = min(8, os.cpu_count() or 1)
INNER_FOLDS = 3
FOLD_SEED = 20_260_901
RATE_GRID = (0.1, 0.3, 1.0, 3.0)
RIDGE_GRID = (0.0003, 0.003, 0.03, 0.3, 3.0, 30.0, 300.0, 3000.0)
FLOOR_MARGIN_GRID = (0.02, 0.08)
VARIANTS = (
    "constant",
    "olmix_exact_macro",
    "raw_weight_ridge_log_link",
    "linear_epoch_log_link",
    "linear_epoch_log_link_permuted_inventory",
    "dsp_benefit_log_link",
    "dsp_shared_task_log_link",
    "dsp_benefit_damage_log_link",
    "dsp_permuted_inventory",
    "dsp_outcome_permutation",
)


@dataclasses.dataclass(frozen=True)
class Pool:
    name: str
    runs: tuple[str, ...]
    buckets: tuple[str, ...]
    tasks: tuple[str, ...]
    weights: np.ndarray
    exposures: np.ndarray
    outcomes: np.ndarray
    input_hashes: dict[str, str]


@dataclasses.dataclass(frozen=True)
class Shape:
    rate: float
    ridge: float
    floor_margin: float


@dataclasses.dataclass(frozen=True)
class Head:
    floor: np.ndarray
    intercept: np.ndarray
    coefficients: np.ndarray


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pool(input_dir: Path, name: str) -> Pool:
    pool_dir = input_dir / name
    paths = {filename: pool_dir / filename for filename in ("metrics.csv", "ratios.csv", "swarm_s42_K363.json")}
    for path in paths.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing nonempty OLMix input: {path}")

    metrics = pd.read_csv(paths["metrics.csv"])
    ratios = pd.read_csv(paths["ratios.csv"])
    payload = json.loads(paths["swarm_s42_K363.json"].read_text())
    identities = ["run", "name", "index"]
    if len(metrics) != 363 or len(ratios) != 363:
        raise ValueError(f"{name}: expected complete 363-row tables")
    if not metrics[identities].equals(ratios[identities]):
        raise ValueError(f"{name}: metric and mixture identities differ")

    buckets = tuple(str(value) for value in payload["domains"])
    tasks = tuple(str(column) for column in metrics.columns if column not in identities)
    if tuple(ratios.columns[3:]) != buckets or len(tasks) != 42:
        raise ValueError(f"{name}: expected the canonical bucket order and 42 tasks")
    weights = ratios.loc[:, buckets].to_numpy(float)
    payload_weights = np.asarray(payload["weights"], dtype=float)
    if weights.shape != payload_weights.shape:
        raise ValueError(f"{name}: ratios.csv shape does not match the frozen swarm payload")
    pairwise_distance = np.max(np.abs(weights[:, None, :] - payload_weights[None, :, :]), axis=2)
    payload_match = np.argmin(pairwise_distance, axis=1)
    if (
        len(np.unique(payload_match)) != len(weights)
        or np.max(pairwise_distance[np.arange(len(weights)), payload_match]) > 1e-12
    ):
        raise ValueError(f"{name}: ratios.csv mixture multiset does not match the frozen swarm payload")
    if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-12):
        raise ValueError(f"{name}: mixtures do not sum to one")
    token_counts = np.asarray([float(payload["tokens"][bucket]) for bucket in buckets])
    if np.any(token_counts <= 0.0):
        raise ValueError(f"{name}: nonpositive bucket inventory")
    exposures = weights * PROXY_TOKENS / token_counts[None, :]
    outcomes = metrics.loc[:, tasks].to_numpy(float)
    if not np.isfinite(outcomes).all() or np.any(outcomes <= 0.0):
        raise ValueError(f"{name}: missing or invalid endpoint metrics")
    return Pool(
        name=name,
        runs=tuple(metrics.run.astype(str)),
        buckets=buckets,
        tasks=tasks,
        weights=weights,
        exposures=exposures,
        outcomes=outcomes,
        input_hashes={filename: file_sha256(path) for filename, path in paths.items()},
    )


def block_labels(weights: np.ndarray, folds: int, seed: int) -> np.ndarray:
    labels = KMeans(n_clusters=folds, n_init=50, random_state=seed).fit_predict(np.sqrt(weights))
    if len(np.unique(labels)) != folds:
        raise ValueError("Mixture-block construction produced an empty fold")
    return labels


def permuted_exposures(pool: Pool) -> np.ndarray:
    generator = np.random.default_rng(FOLD_SEED)
    permutation = generator.permutation(len(pool.buckets))
    rates = np.divide(
        pool.exposures,
        pool.weights,
        out=np.zeros_like(pool.exposures),
        where=pool.weights > 0.0,
    )
    bucket_rates = np.max(rates, axis=0)
    return pool.weights * bucket_rates[permutation][None, :]


def features(pool: Pool, rows: np.ndarray, variant: str, shape: Shape) -> tuple[np.ndarray, bool]:
    if variant == "raw_weight_ridge_log_link":
        return pool.weights[rows], False
    permuted_inventory = variant in ("linear_epoch_log_link_permuted_inventory", "dsp_permuted_inventory")
    exposure = permuted_exposures(pool)[rows] if permuted_inventory else pool.exposures[rows]
    if variant in ("linear_epoch_log_link", "linear_epoch_log_link_permuted_inventory"):
        return np.log1p(exposure), False
    benefit = 1.0 - np.exp(-shape.rate * exposure)
    if variant in (
        "dsp_benefit_log_link",
        "dsp_shared_task_log_link",
        "dsp_permuted_inventory",
        "dsp_outcome_permutation",
    ):
        return -benefit, True
    if variant == "dsp_benefit_damage_log_link":
        repeated = np.log1p(np.maximum(exposure - 1.0, 0.0))
        return np.column_stack([-benefit, repeated]), True
    raise ValueError(f"Unknown feature variant: {variant}")


def response_floor(outcomes: np.ndarray, margin: float) -> np.ndarray:
    low = outcomes.min(axis=0)
    span = outcomes.max(axis=0) - low
    scale = np.maximum(span, 0.05 * np.maximum(low, 1e-6))
    return low - margin * scale


def fit_head(design: np.ndarray, outcomes: np.ndarray, shape: Shape, positive: bool) -> Head:
    floor = response_floor(outcomes, shape.floor_margin)
    transformed = np.log(outcomes - floor[None, :])
    center = design.mean(axis=0)
    centered = design - center[None, :]
    scale = np.sqrt(np.mean(centered**2, axis=0))
    scale[scale < 1e-10] = 1.0
    normalized = centered / scale[None, :]
    width = normalized.shape[1]
    augmented = np.vstack([normalized, np.sqrt(shape.ridge) * np.eye(width)])
    coefficients = np.empty((width, transformed.shape[1]))
    for task in range(transformed.shape[1]):
        target_mean = float(transformed[:, task].mean())
        target = np.concatenate([transformed[:, task] - target_mean, np.zeros(width)])
        if positive:
            solution = optimize.lsq_linear(
                augmented,
                target,
                bounds=(np.zeros(width), np.full(width, np.inf)),
                lsmr_tol="auto",
                max_iter=300 * width,
            )
            if not solution.success:
                raise ValueError(solution.message)
            coefficients[:, task] = solution.x / scale
        else:
            gram = normalized.T @ normalized + shape.ridge * np.eye(width)
            coefficients[:, task] = np.linalg.solve(gram, normalized.T @ (transformed[:, task] - target_mean)) / scale
    intercept = transformed.mean(axis=0) - center @ coefficients
    return Head(floor=floor, intercept=intercept, coefficients=coefficients)


def fit_shared_task_head(design: np.ndarray, outcomes: np.ndarray, shape: Shape) -> Head:
    """Fit one nonnegative exposure response shared by all atomic tasks."""
    floor = response_floor(outcomes, shape.floor_margin)
    transformed = np.log(outcomes - floor[None, :])
    center = design.mean(axis=0)
    centered = design - center[None, :]
    scale = np.sqrt(np.mean(centered**2, axis=0))
    scale[scale < 1e-10] = 1.0
    normalized = centered / scale[None, :]
    width = normalized.shape[1]
    augmented = np.vstack([normalized, np.sqrt(shape.ridge) * np.eye(width)])
    shared_target = (transformed - transformed.mean(axis=0)[None, :]).mean(axis=1)
    target = np.concatenate([shared_target, np.zeros(width)])
    solution = optimize.lsq_linear(
        augmented,
        target,
        bounds=(np.zeros(width), np.full(width, np.inf)),
        lsmr_tol="auto",
        max_iter=300 * width,
    )
    if not solution.success:
        raise ValueError(solution.message)
    shared_coefficients = solution.x / scale
    coefficients = np.broadcast_to(shared_coefficients[:, None], (width, transformed.shape[1])).copy()
    intercept = transformed.mean(axis=0) - center @ coefficients
    return Head(floor=floor, intercept=intercept, coefficients=coefficients)


def predict(head: Head, design: np.ndarray) -> np.ndarray:
    linear = head.intercept[None, :] + design @ head.coefficients
    return head.floor[None, :] + np.exp(np.clip(linear, -30.0, 30.0))


def candidate_shapes(variant: str) -> tuple[Shape, ...]:
    if variant in ("constant", "olmix_exact_macro"):
        return (Shape(0.0, 0.0, 0.0),)
    rates = RATE_GRID if variant.startswith("dsp_") else (0.0,)
    return tuple(
        Shape(rate=rate, ridge=ridge, floor_margin=margin)
        for rate in rates
        for ridge in RIDGE_GRID
        for margin in FLOOR_MARGIN_GRID
    )


def fit_variant(
    pool: Pool,
    train: np.ndarray,
    variant: str,
    shape: Shape,
    outcomes: np.ndarray | None = None,
) -> Head | OlmixLoglinearFit:
    target = pool.outcomes[train] if outcomes is None else outcomes
    if variant == "constant":
        mean = target.mean(axis=0)
        return Head(floor=np.zeros_like(mean), intercept=np.log(mean), coefficients=np.zeros((0, len(mean))))
    if variant == "olmix_exact_macro":
        if target.ndim != 2 or target.shape[1] != 1:
            raise ValueError("Exact OLMix is fitted directly to the scalar macro target")
        return fit_olmix_loglinear_model(pool.weights[train], target[:, 0])
    if variant == "dsp_outcome_permutation":
        generator = np.random.default_rng(FOLD_SEED + int(np.sum(train)))
        target = target[generator.permutation(len(target))]
    design, positive = features(pool, train, variant, shape)
    if variant == "dsp_shared_task_log_link":
        return fit_shared_task_head(design, target, shape)
    return fit_head(design, target, shape, positive)


def predict_variant(
    pool: Pool,
    rows: np.ndarray,
    variant: str,
    shape: Shape,
    head: Head | OlmixLoglinearFit,
) -> np.ndarray:
    if variant == "olmix_exact_macro":
        if not isinstance(head, OlmixLoglinearFit):
            raise TypeError("Exact OLMix variant requires an OLMix fit")
        return head.predict(pool.weights[rows])[:, None]
    if not isinstance(head, Head):
        raise TypeError(f"{variant} requires a DSP head")
    if variant == "constant":
        return np.broadcast_to(np.exp(head.intercept)[None, :], (len(rows), len(head.intercept))).copy()
    design, _ = features(pool, rows, variant, shape)
    return predict(head, design)


def select_shape(pool: Pool, train: np.ndarray, variant: str, seed: int) -> Shape:
    if variant in ("constant", "olmix_exact_macro"):
        return candidate_shapes(variant)[0]
    inner = block_labels(pool.weights[train], INNER_FOLDS, seed)
    macro = pool.outcomes.mean(axis=1, keepdims=True)
    candidates = []
    for shape in candidate_shapes(variant):
        squared_error = 0.0
        count = 0
        for fold in range(INNER_FOLDS):
            fit_rows = train[inner != fold]
            test_rows = train[inner == fold]
            head = fit_variant(pool, fit_rows, variant, shape, outcomes=macro[fit_rows])
            prediction = predict_variant(pool, test_rows, variant, shape, head)[:, 0]
            squared_error += float(np.sum((prediction - macro[test_rows, 0]) ** 2))
            count += len(test_rows)
        candidates.append((squared_error / count, shape))
    return min(candidates, key=lambda item: (item[0], dataclasses.astuple(item[1])))[1]


def fold_scores(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = predicted - actual
    selected = int(np.argmin(predicted))
    constant_prediction = bool(np.ptp(predicted) <= 1e-12)
    spearman = 0.0 if constant_prediction else float(stats.spearmanr(actual, predicted).statistic)
    calibration_slope = 0.0 if constant_prediction else float(stats.linregress(predicted, actual).slope)
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "spearman": spearman,
        "calibration_slope": calibration_slope,
        "selection_regret": float(actual[selected] - actual.min()),
    }


def benchmark_fold(
    pool: Pool,
    repeat: int,
    fold: int,
    outer: np.ndarray,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    prediction_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    all_rows = np.arange(len(pool.runs))
    actual_macro = pool.outcomes.mean(axis=1)
    train = all_rows[outer != fold]
    test = all_rows[outer == fold]
    for variant in VARIANTS:
        shape = select_shape(pool, train, variant, FOLD_SEED + 1000 * repeat + fold)
        fit_outcomes = actual_macro[train, None] if variant == "olmix_exact_macro" else None
        head = fit_variant(pool, train, variant, shape, outcomes=fit_outcomes)
        task_predictions = predict_variant(pool, test, variant, shape, head)
        macro_prediction = task_predictions.mean(axis=1)
        scores = fold_scores(actual_macro[test], macro_prediction)
        fold_rows.append(
            {
                "pool": pool.name,
                "variant": variant,
                "repeat": repeat,
                "fold": fold,
                "test_rows": len(test),
                **dataclasses.asdict(shape),
                **scores,
            }
        )
        for local, row in enumerate(test):
            prediction_rows.append(
                {
                    "pool": pool.name,
                    "variant": variant,
                    "repeat": repeat,
                    "fold": fold,
                    "run": pool.runs[row],
                    "index": int(row),
                    "observed_macro_bpb": actual_macro[row],
                    "predicted_macro_bpb": macro_prediction[local],
                }
            )
    return prediction_rows, fold_rows


def benchmark_pool(pool: Pool) -> tuple[pd.DataFrame, pd.DataFrame]:
    jobs = []
    for repeat in range(OUTER_REPEATS):
        outer = block_labels(pool.weights, OUTER_FOLDS, FOLD_SEED + 100 * repeat)
        for fold in range(OUTER_FOLDS):
            jobs.append(delayed(benchmark_fold)(pool, repeat, fold, outer))
    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=OUTER_WORKERS)(jobs)
    prediction_rows = [row for result in results for row in result[0]]
    fold_rows = [row for result in results for row in result[1]]
    return pd.DataFrame(prediction_rows), pd.DataFrame(fold_rows)


def aggregate_metrics(predictions: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pool, variant), group in predictions.groupby(["pool", "variant"], sort=False):
        repeat_metrics = []
        for repeat, repeated in group.groupby("repeat"):
            score = fold_scores(
                repeated.observed_macro_bpb.to_numpy(float),
                repeated.predicted_macro_bpb.to_numpy(float),
            )
            score["repeat"] = int(repeat)
            repeat_metrics.append(score)
        frame = pd.DataFrame(repeat_metrics)
        fold_group = folds[(folds.pool == pool) & (folds.variant == variant)]
        row: dict[str, object] = {"pool": pool, "variant": variant}
        for metric in ("rmse", "mae", "spearman", "calibration_slope"):
            row[metric] = float(frame[metric].mean())
            row[f"{metric}_repeat_sd"] = float(frame[metric].std(ddof=1))
        row["mean_fold_selection_regret"] = float(fold_group.selection_regret.mean())
        row["median_selected_rate"] = float(fold_group.rate.median())
        row["median_selected_ridge"] = float(fold_group.ridge.median())
        row["median_selected_floor_margin"] = float(fold_group.floor_margin.median())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["pool", "rmse", "mean_fold_selection_regret"])


def corrected_contrasts(folds: pd.DataFrame) -> pd.DataFrame:
    comparisons = (
        ("linear_epoch_log_link", "olmix_exact_macro"),
        ("linear_epoch_log_link", "linear_epoch_log_link_permuted_inventory"),
        ("dsp_benefit_log_link", "olmix_exact_macro"),
        ("dsp_benefit_log_link", "dsp_permuted_inventory"),
        ("linear_epoch_log_link", "dsp_benefit_log_link"),
        ("linear_epoch_log_link", "raw_weight_ridge_log_link"),
    )
    factor = 1.0 / (OUTER_FOLDS * OUTER_REPEATS) + 1.0 / (OUTER_FOLDS - 1.0)
    rows = []
    for pool, group in folds.groupby("pool"):
        pivot = group.pivot(index=["repeat", "fold"], columns="variant", values="rmse")
        critical = float(stats.t.ppf(0.975, len(pivot) - 1))
        for challenger, comparator in comparisons:
            difference = pivot[challenger] - pivot[comparator]
            mean = float(difference.mean())
            se = float(np.sqrt(factor * difference.var(ddof=1)))
            rows.append(
                {
                    "pool": pool,
                    "comparison": f"{challenger}_minus_{comparator}",
                    "mean_rmse_difference": mean,
                    "corrected_se": se,
                    "ci_low": mean - critical * se,
                    "ci_high": mean + critical * se,
                }
            )
    return pd.DataFrame(rows)


def promotion_gate(aggregate: pd.DataFrame, contrasts: pd.DataFrame) -> dict[str, object]:
    conditions: dict[str, bool] = {}
    for pool in POOLS:
        selected = aggregate[aggregate.pool.eq(pool)].set_index("variant")
        pool_contrasts = contrasts[contrasts.pool.eq(pool)].set_index("comparison")
        conditions[f"{pool}.candidate_rmse_beats_exact_olmix"] = (
            float(pool_contrasts.loc["linear_epoch_log_link_minus_olmix_exact_macro", "ci_high"]) < 0.0
        )
        conditions[f"{pool}.candidate_selection_beats_exact_olmix"] = float(
            selected.loc["linear_epoch_log_link", "mean_fold_selection_regret"]
        ) < float(selected.loc["olmix_exact_macro", "mean_fold_selection_regret"])
        conditions[f"{pool}.saturating_exposure_rmse_beats_matched_scramble"] = (
            float(pool_contrasts.loc["dsp_benefit_log_link_minus_dsp_permuted_inventory", "ci_high"]) < 0.0
        )
        conditions[f"{pool}.saturating_exposure_selection_beats_matched_scramble"] = float(
            selected.loc["dsp_benefit_log_link", "mean_fold_selection_regret"]
        ) < float(selected.loc["dsp_permuted_inventory", "mean_fold_selection_regret"])
    passed = bool(all(conditions.values()))
    linear_inventory_diagnostic = {
        pool: {
            "rmse_corrected_ci_below_zero": (
                float(
                    contrasts[
                        contrasts.pool.eq(pool)
                        & contrasts.comparison.eq("linear_epoch_log_link_minus_linear_epoch_log_link_permuted_inventory")
                    ]
                    .iloc[0]
                    .ci_high
                )
                < 0.0
            ),
            "selection_regret_improves": (
                float(
                    aggregate[aggregate.pool.eq(pool) & aggregate.variant.eq("linear_epoch_log_link")]
                    .iloc[0]
                    .mean_fold_selection_regret
                )
                < float(
                    aggregate[aggregate.pool.eq(pool) & aggregate.variant.eq("linear_epoch_log_link_permuted_inventory")]
                    .iloc[0]
                    .mean_fold_selection_regret
                )
            ),
        }
        for pool in POOLS
    }
    return {
        "candidate": "linear_epoch_log_link",
        "mechanism_witness": "dsp_benefit_log_link",
        "conditions": conditions,
        "linear_matched_inventory_diagnostic": linear_inventory_diagnostic,
        "pass": passed,
        "verdict": (
            "Promote the simple linear epoch-exposure head for fresh single-phase validation."
            if passed
            else "Do not promote a single-phase challenger from this benchmark."
        ),
    }


def write_report(
    output_dir: Path,
    pools: tuple[Pool, ...],
    aggregate: pd.DataFrame,
    contrasts: pd.DataFrame,
    gate: dict[str, object],
) -> None:
    linear_inventory_summary = (
        "clears corrected intervals in both swarms"
        if all(
            bool(result["rmse_corrected_ci_below_zero"])
            for result in gate["linear_matched_inventory_diagnostic"].values()
        )
        else "does not clear corrected intervals in both swarms"
    )
    primary_ridge_rows = aggregate[
        aggregate.variant.isin(
            (
                "linear_epoch_log_link",
                "dsp_benefit_log_link",
                "dsp_permuted_inventory",
                "raw_weight_ridge_log_link",
            )
        )
    ]
    boundary_ridge_rows = primary_ridge_rows[
        primary_ridge_rows.median_selected_ridge.isin((min(RIDGE_GRID), max(RIDGE_GRID)))
    ]
    boundary_ridge_summary = ", ".join(
        f"`{row.pool}/{row.variant}`={row.median_selected_ridge:g}"
        for row in boundary_ridge_rows.itertuples(index=False)
    )
    primary_rate_rows = aggregate[aggregate.variant.isin(("dsp_benefit_log_link", "dsp_permuted_inventory"))]
    boundary_rate_rows = primary_rate_rows[primary_rate_rows.median_selected_rate.isin((min(RATE_GRID), max(RATE_GRID)))]
    boundary_rate_summary = ", ".join(
        f"`{row.pool}/{row.variant}`={row.median_selected_rate:g}" for row in boundary_rate_rows.itertuples(index=False)
    )
    lines = [
        "# OLMix proxy-swarm single-phase surrogate benchmark",
        "",
        "This exploratory benchmark uses two complete 363-mixture OLMix proxy swarms. All 42 endpoint tasks are "
        "predicted out of fold, then macro-averaged. Outer and inner folds are K-means blocks in square-root mixture "
        "coordinates. This reduces proximity leakage, but K-means does not guarantee a strict margin between folds.",
        "",
        "The `olmix_exact_macro` row faithfully reproduces Michael's summed-Huber 48-start law on the scalar macro "
        "target, without regularization or inner tuning. The challengers use nested ridge/floor selection and fit the "
        "42 atomic tasks separately, so their contrast against OLMix measures the complete estimator package rather "
        "than exposure coordinates alone. The matched `linear_epoch_log_link_permuted_inventory` row keeps the "
        "linear link, sign freedom, model size, and tuning grid fixed while permuting bucket inventories. The "
        "`raw_weight_ridge_log_link` row is a capacity ablation, not an OLMix reproduction. The saturating DSP rows "
        "provide a second matched inventory pair. The outcome permutation is a training-fold-only negative control.",
        "",
        "The widened hyperparameter grids still bind for gate-relevant rows: ridge "
        f"{boundary_ridge_summary}; rate {boundary_rate_summary}. Their model-package comparisons remain usable, but "
        "the benchmark does not identify their optimal regularization or establish that raw weights are intrinsically "
        "inadequate.",
        "",
        "## Results",
        "",
        "| pool | model | RMSE | Spearman | mean fold regret |",
        "|---|---|---:|---:|---:|",
    ]
    for row in aggregate.itertuples(index=False):
        lines.append(
            f"| {row.pool} | {row.variant} | {row.rmse:.5f} | {row.spearman:.3f} | "
            f"{row.mean_fold_selection_regret:.5f} |"
        )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            str(gate["verdict"]),
            "",
            "The promotion gate compares the linear epoch-exposure candidate with exact OLMix, and uses the "
            "saturating DSP pair as the matched mechanism witness. The linear head's own matched inventory control "
            f"{linear_inventory_summary}, "
            "so it is not used to claim a linear-link-specific mechanism. The linear head is selected for prospective "
            "validation because it removes the saturating rate hyperparameter while remaining statistically "
            "indistinguishable from the saturating head. The "
            "linear-versus-OLMix contrast supports testing the full regularized, taskwise model package prospectively; "
            "it does not by itself attribute the gain to epoch exposure. The inventory controls use one fixed "
            "permutation, so they are mechanism evidence rather than a population estimate over permutations.",
            "",
            "## Corrected RMSE contrasts",
            "",
            contrasts.to_markdown(index=False, floatfmt=".6f"),
            "",
            "## Interpretation rule",
            "",
            "An exposure mechanism is supported only by matched inventory-correct versus inventory-permuted heads; "
            "because each design column is normalized, this permutation specifically tests whether inventory-indexed "
            "curvature is assigned to the correct buckets rather than identifying epoch exposure uniquely. "
            "The OLMix contrast instead asks whether the full challenger is worth fresh prospective validation. A "
            "rank-only win is insufficient for optimum selection. These are model-development results, not a fresh "
            "endpoint validation.",
            "",
            "## Inputs",
            "",
        ]
    )
    for pool in pools:
        lines.append(f"- `{pool.name}`: 363 mixtures, {len(pool.buckets)} buckets, 42 tasks.")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--postprocess-existing", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pools = tuple(load_pool(args.input_dir, name) for name in POOLS)
    if args.postprocess_existing:
        predictions = pd.read_csv(args.output_dir / "predictions.csv")
        folds = pd.read_csv(args.output_dir / "fold_metrics.csv")
    else:
        results = [benchmark_pool(pool) for pool in pools]
        predictions = pd.concat([item[0] for item in results], ignore_index=True)
        folds = pd.concat([item[1] for item in results], ignore_index=True)
    aggregate = aggregate_metrics(predictions, folds)
    contrasts = corrected_contrasts(folds)
    gate = promotion_gate(aggregate, contrasts)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    folds.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    aggregate.to_csv(args.output_dir / "aggregate_metrics.csv", index=False)
    contrasts.to_csv(args.output_dir / "corrected_contrasts.csv", index=False)
    (args.output_dir / "promotion_gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    protocol = {
        "primary_target": "mean BPB across the 42 OLMo Base-Easy tasks",
        "outer_folds": OUTER_FOLDS,
        "outer_repeats": OUTER_REPEATS,
        "outer_workers": OUTER_WORKERS,
        "inner_folds": INNER_FOLDS,
        "fold_geometry": "KMeans on square-root mixture weights",
        "variants": list(VARIANTS),
        "rate_grid": list(RATE_GRID),
        "ridge_grid": list(RIDGE_GRID),
        "floor_margin_grid": list(FLOOR_MARGIN_GRID),
        "proxy_tokens": PROXY_TOKENS,
        "inputs": {pool.name: pool.input_hashes for pool in pools},
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    write_report(args.output_dir, pools, aggregate, contrasts, gate)


if __name__ == "__main__":
    main()
