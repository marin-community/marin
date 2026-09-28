# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scikit-learn", "scipy", "tabulate"]
# ///
"""Benchmark an aggregate-exposure model with a bounded recency residual.

The model separates phase-invariant data exposure from phase ordering:

    E_i = e_i^(0) + e_i^(1)
    q_i = tanh(log(1 + e_i^(1) / alpha_1)
               - log(1 + e_i^(0) / alpha_0))

    L(w) = b + B(E; mu) beta + R(E, q; mu) delta.

``B`` is a nonnegative two-sided preferred-exposure bowl. ``R`` is a signed,
ridge-regularized phase residual. It vanishes exactly when the two phase
mixtures are tied, because phase-normalized exposures are then equal. The
bounded ``q`` prevents the optimizer from obtaining unbounded credit by
making the phases arbitrarily different.

The ``slope`` residual uses ``q`` times the lower and upper linear slopes of
the aggregate bowl. It is the first-order phase-contrast correction to a
shared exposure curve, rather than a second independent exposure model.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.optimize import lsq_linear

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)

DEFAULT_OUTPUT_DIR = pooled.REFERENCE_OUTPUTS / "aggregate_recency_phase_model_20260709"
SCALE_FLOOR = 1e-8
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class Config:
    name: str
    residual: str
    aggregate_l2: float
    phase_l2: float


@dataclass(frozen=True)
class Model:
    config: Config
    mu: np.ndarray
    alpha0: float
    alpha1: float
    aggregate_scale: np.ndarray
    phase_scale: np.ndarray
    intercept: float
    aggregate_coef: np.ndarray
    phase_coef: np.ndarray


def phase_fractions(dataset: pooled.Dataset) -> tuple[float, float]:
    ratio = float(np.median(dataset.c0 / dataset.c1))
    alpha0 = ratio / (1.0 + ratio)
    return alpha0, 1.0 - alpha0


def aggregate_design(exposure: np.ndarray, mu: np.ndarray) -> np.ndarray:
    delta = np.log1p(exposure) - mu[None, :]
    return np.hstack([np.minimum(delta, 0.0) ** 2, np.maximum(delta, 0.0) ** 2])


def recency_coordinate(
    exposure0: np.ndarray,
    exposure1: np.ndarray,
    alpha0: float,
    alpha1: float,
) -> np.ndarray:
    contrast = np.log1p(exposure1 / alpha1) - np.log1p(exposure0 / alpha0)
    return np.tanh(contrast)


def phase_design(
    exposure0: np.ndarray,
    exposure1: np.ndarray,
    mu: np.ndarray,
    alpha0: float,
    alpha1: float,
    residual: str,
) -> np.ndarray:
    aggregate = exposure0 + exposure1
    delta = np.log1p(aggregate) - mu[None, :]
    recency = recency_coordinate(exposure0, exposure1, alpha0, alpha1)
    if residual == "none":
        return np.zeros((len(aggregate), 0), dtype=float)
    if residual == "main":
        return recency
    slope = np.hstack([recency * np.minimum(delta, 0.0), recency * np.maximum(delta, 0.0)])
    if residual == "slope":
        return slope
    if residual == "main_slope":
        return np.hstack([recency, slope])
    raise ValueError(f"Unknown residual {residual!r}")


def selected_mu(aggregate: np.ndarray, y: np.ndarray) -> np.ndarray:
    median = pooled.base_mu(aggregate)
    best_rmse = np.inf
    best = median
    for shift in pooled.MU_SHIFTS:
        mu = np.clip(median + shift, -2.0, 8.0)
        design = aggregate_design(aggregate, mu)
        intercept, coef = pooled.fit_raw_nnls(design, y, l2=0.1)
        prediction = intercept + design @ coef
        rmse = float(np.sqrt(np.mean((prediction - y) ** 2)))
        if rmse < best_rmse:
            best_rmse = rmse
            best = mu
    return best


def column_scale(design: np.ndarray) -> np.ndarray:
    if design.shape[1] == 0:
        return np.zeros(0, dtype=float)
    return np.maximum(np.sqrt(np.mean(design**2, axis=0)), SCALE_FLOOR)


def fit_model(
    dataset: pooled.Dataset,
    row_indices: np.ndarray,
    config: Config,
) -> Model:
    exposure0, exposure1 = pooled.phase_exposures(dataset, row_indices)
    y = dataset.y[row_indices]
    aggregate = exposure0 + exposure1
    mu = selected_mu(aggregate, y)
    alpha0, alpha1 = phase_fractions(dataset)
    aggregate_raw = aggregate_design(aggregate, mu)
    phase_raw = phase_design(exposure0, exposure1, mu, alpha0, alpha1, config.residual)
    aggregate_scale = column_scale(aggregate_raw)
    phase_scale = column_scale(phase_raw)
    aggregate_features = aggregate_raw / aggregate_scale[None, :]
    phase_features = phase_raw / phase_scale[None, :] if phase_raw.shape[1] else phase_raw
    design = np.hstack([aggregate_features, phase_features])
    center = design.mean(axis=0)
    y_mean = float(y.mean())
    augmented_design = design - center[None, :]
    augmented_target = y - y_mean
    aggregate_dim = aggregate_features.shape[1]
    phase_dim = phase_features.shape[1]
    if config.aggregate_l2 > 0.0:
        regularizer = np.zeros((aggregate_dim, aggregate_dim + phase_dim), dtype=float)
        regularizer[:, :aggregate_dim] = np.sqrt(config.aggregate_l2) * np.eye(aggregate_dim)
        augmented_design = np.vstack([augmented_design, regularizer])
        augmented_target = np.concatenate([augmented_target, np.zeros(aggregate_dim)])
    if config.phase_l2 > 0.0 and phase_dim:
        regularizer = np.zeros((phase_dim, aggregate_dim + phase_dim), dtype=float)
        regularizer[:, aggregate_dim:] = np.sqrt(config.phase_l2) * np.eye(phase_dim)
        augmented_design = np.vstack([augmented_design, regularizer])
        augmented_target = np.concatenate([augmented_target, np.zeros(phase_dim)])
    lower = np.concatenate([np.zeros(aggregate_dim), np.full(phase_dim, -np.inf)])
    upper = np.full(aggregate_dim + phase_dim, np.inf)
    result = lsq_linear(
        augmented_design,
        augmented_target,
        bounds=(lower, upper),
        method="trf",
        lsmr_tol="auto",
        max_iter=500,
    )
    if not result.success:
        raise RuntimeError(f"Head fit failed: {result.message}")
    coef = np.asarray(result.x, dtype=float)
    intercept = y_mean - float(center @ coef)
    return Model(
        config=config,
        mu=mu,
        alpha0=alpha0,
        alpha1=alpha1,
        aggregate_scale=aggregate_scale,
        phase_scale=phase_scale,
        intercept=intercept,
        aggregate_coef=coef[:aggregate_dim],
        phase_coef=coef[aggregate_dim:],
    )


def predict(
    model: Model,
    exposure0: np.ndarray,
    exposure1: np.ndarray,
) -> np.ndarray:
    aggregate = exposure0 + exposure1
    aggregate_features = aggregate_design(aggregate, model.mu) / model.aggregate_scale[None, :]
    phase_raw = phase_design(
        exposure0,
        exposure1,
        model.mu,
        model.alpha0,
        model.alpha1,
        model.config.residual,
    )
    prediction = model.intercept + aggregate_features @ model.aggregate_coef
    if phase_raw.shape[1]:
        prediction = prediction + (phase_raw / model.phase_scale[None, :]) @ model.phase_coef
    return prediction


def benchmark_dataset(
    dataset: pooled.Dataset,
    configs: list[Config],
    seeds: list[int],
    n_splits: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    all_indices = np.arange(dataset.n)
    for seed in seeds:
        folds = pooled.dataset_folds(dataset, seed, n_splits)
        predictions = {config.name: np.zeros(dataset.n, dtype=float) for config in configs}
        for fold_id, (train_idx, test_idx) in enumerate(folds):
            print(f"{dataset.name}: seed={seed} fold={fold_id + 1}/{n_splits}", flush=True)
            test_exposure0, test_exposure1 = pooled.phase_exposures(dataset, test_idx)
            for config in configs:
                model = fit_model(dataset, train_idx, config)
                predictions[config.name][test_idx] = predict(model, test_exposure0, test_exposure1)
        for config in configs:
            metric = pooled.metrics(dataset, config.name, seed, predictions[config.name], folds)
            row = asdict(metric)
            phase_multiplier = {"none": 0, "main": 1, "slope": 2, "main_slope": 3}[config.residual]
            row["nominal_param_count"] = (2 + phase_multiplier) * dataset.m + 2
            metric_rows.append(row)
            full_model = fit_model(dataset, all_indices, config)
            coefficient_rows.append(
                {
                    "dataset": dataset.name,
                    "model": config.name,
                    "seed": seed,
                    "aggregate_coef_norm": float(np.linalg.norm(full_model.aggregate_coef)),
                    "phase_coef_norm": float(np.linalg.norm(full_model.phase_coef)),
                    "phase_to_aggregate_norm_ratio": float(
                        np.linalg.norm(full_model.phase_coef) / max(np.linalg.norm(full_model.aggregate_coef), 1e-12)
                    ),
                    "phase_fraction_0": full_model.alpha0,
                    "phase_fraction_1": full_model.alpha1,
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(coefficient_rows)


def configs(aggregate_l2: float, phase_l2_values: list[float]) -> list[Config]:
    values = [Config("aggregate_only", "none", aggregate_l2, 0.0)]
    for residual in ("main", "slope", "main_slope"):
        for phase_l2 in phase_l2_values:
            values.append(
                Config(
                    name=f"aggregate_{residual}_phase_l2_{phase_l2:g}",
                    residual=residual,
                    aggregate_l2=aggregate_l2,
                    phase_l2=phase_l2,
                )
            )
    return values


def write_plots(summary: pd.DataFrame, output_dir: Path) -> None:
    for metric in ("oof_spearman_mean", "oof_rmse_mean", "fold_mean_regret_at_1_mean", "lower_tail_optimism_mean"):
        figure = px.bar(
            summary,
            x="model",
            y=metric,
            color="dataset",
            barmode="group",
            color_discrete_sequence=["#1a9850", "#fee08b", "#d73027"],
            title=f"Aggregate + bounded recency: {metric.removesuffix('_mean')}",
        )
        figure.update_layout(xaxis_tickangle=-30, legend_title="Dataset")
        figure.write_html(
            output_dir / f"{metric.removesuffix('_mean')}.html",
            include_plotlyjs="cdn",
            config=PLOT_CONFIG,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--datasets", default="300m_uncheatable,300m_table9,production_uncheatable")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--aggregate-l2", type=float, default=0.1)
    parser.add_argument("--phase-l2-values", default="0.01,0.1,1,10,100")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    loaders = {
        "300m_uncheatable": lambda: pooled.load_300m_dataset("uncheatable"),
        "300m_table9": lambda: pooled.load_300m_dataset("table9"),
        "production_uncheatable": pooled.load_production_dataset,
    }
    selected = [part.strip() for part in args.datasets.split(",") if part.strip()]
    unknown = sorted(set(selected).difference(loaders))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    model_configs = configs(args.aggregate_l2, pooled.parse_float_list(args.phase_l2_values))
    seeds = pooled.parse_int_list(args.seeds)

    raw_frames = []
    coefficient_frames = []
    for name in selected:
        raw, coefficients = benchmark_dataset(loaders[name](), model_configs, seeds, args.n_splits)
        raw_frames.append(raw)
        coefficient_frames.append(coefficients)
    raw = pd.concat(raw_frames, ignore_index=True)
    coefficients = pd.concat(coefficient_frames, ignore_index=True)
    summary = pooled.summarize(raw)
    raw.to_csv(args.output_dir / "cv_metrics_by_seed.csv", index=False)
    coefficients.to_csv(args.output_dir / "coefficient_diagnostics.csv", index=False)
    summary.to_csv(args.output_dir / "cv_summary.csv", index=False)
    write_plots(summary, args.output_dir)
    print(summary.to_string(index=False))
    print(f"Wrote benchmark artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
