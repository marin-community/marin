# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scikit-learn", "scipy", "tabulate"]
# ///
"""Screen bounded recency residuals on top of effective-exposure DSP.

This two-stage model preserves the robust effective-exposure DSP prediction and
adds only a correction that is zero for a tied schedule. The correction uses a
bounded phase contrast times the fitted DSP derivative channels:

    q_i = tanh(log(1 + e_i^(1) / alpha_1)
               - log(1 + e_i^(0) / alpha_0))

    g_benefit_i = q_i a_i rho_i exp(-rho_i z_i)
    g_penalty_i = q_i p_i 2 softplus(x_i) sigmoid(x_i) / (1 + z_i)

where ``z_i=e_i^(0)+gamma e_i^(1)`` and ``x_i=log(1+z_i)-tau_i``.
Signed ridge coefficients fit a bounded, first-order phase-order correction.
The baseline's nonlinear parameters are fixed from the full panel in this
screen, matching the historical fixed-feature OOF protocol. A winning residual
must subsequently pass fully nested refitting before candidate materialization.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.linear_model import Ridge

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_table9_phase_split_dsp_300m as phase_dsp,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import dsp_exact as dsp  # noqa: E402

DEFAULT_OUTPUT_DIR = pooled.REFERENCE_OUTPUTS / "effective_exposure_recency_residual_20260709"
THREE_HUNDRED_LINEAR_REG = 0.01
PRODUCTION_LINEAR_REG = 1e-6
SCALE_FLOOR = 1e-8
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class ResidualConfig:
    name: str
    feature_mode: str
    ridge_alpha: float


@dataclass(frozen=True)
class ResidualModel:
    config: ResidualConfig
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    intercept: float
    coef: np.ndarray


def packet(dataset: pooled.Dataset, indices: np.ndarray | None = None) -> dsp.PacketData:
    selected = np.arange(dataset.n) if indices is None else np.asarray(indices, dtype=int)
    name_col = "run_name" if "run_name" in dataset.frame.columns else "candidate_name"
    return dsp.PacketData(
        frame=dataset.frame.iloc[selected].reset_index(drop=True),
        name_col=name_col,
        y=dataset.y[selected],
        w=dataset.weights[selected],
        m=dataset.m,
        c0=dataset.c0,
        c1=dataset.c1,
        domain_names=list(dataset.domain_names),
    )


def fitted_production_model(dataset: pooled.Dataset) -> dsp.FittedDSPModel:
    raw = json.loads(pooled.PRODUCTION_MODEL.read_text())
    params = {
        key: np.asarray(value, dtype=float) if isinstance(value, list) else float(value)
        for key, value in raw["params"].items()
    }
    return dsp.FittedDSPModel(
        variant=dsp.VARIANTS["effective_exposure"],
        params=params,
        intercept=float(raw["intercept"]),
        benefit_coef=np.asarray(raw["benefit_coef"], dtype=float),
        penalty_coef=np.asarray(raw["penalty_coef"], dtype=float),
        domain_names=list(raw["domain_names"]),
        c0=np.asarray(raw["c0"], dtype=float),
        c1=np.asarray(raw["c1"], dtype=float),
    )


def fitted_300m_model(dataset: pooled.Dataset) -> dsp.FittedDSPModel:
    model, _tuning = phase_dsp.fit_variant_with_l2(
        packet(dataset),
        "effective_exposure",
        THREE_HUNDRED_LINEAR_REG,
        maxiter=40,
        coarse_top_k=3,
        basin_hopping_iters=0,
    )
    return model


def nonlinear_anchor(dataset: pooled.Dataset) -> dsp.FittedDSPModel:
    if dataset.name == "production_uncheatable":
        return fitted_production_model(dataset)
    return fitted_300m_model(dataset)


def refit_linear_head(
    dataset: pooled.Dataset,
    indices: np.ndarray,
    anchor: dsp.FittedDSPModel,
) -> dsp.FittedDSPModel:
    linear_reg = PRODUCTION_LINEAR_REG if dataset.name == "production_uncheatable" else THREE_HUNDRED_LINEAR_REG
    old_linear_reg = dsp.LINEAR_REG
    dsp.LINEAR_REG = linear_reg
    try:
        return dsp.fit_linear_head(
            dataset.weights[indices],
            dataset.y[indices],
            packet(dataset),
            anchor.variant,
            anchor.params,
        )
    finally:
        dsp.LINEAR_REG = old_linear_reg


def phase_fractions(model: dsp.FittedDSPModel) -> tuple[float, float]:
    ratio = float(np.median(model.c0 / model.c1))
    alpha0 = ratio / (1.0 + ratio)
    return alpha0, 1.0 - alpha0


def recency_features(model: dsp.FittedDSPModel, weights: np.ndarray, mode: str) -> np.ndarray:
    e0 = weights[:, 0, :] * model.c0[None, :]
    e1 = weights[:, 1, :] * model.c1[None, :]
    alpha0, alpha1 = phase_fractions(model)
    q = np.tanh(np.log1p(e1 / alpha1) - np.log1p(e0 / alpha0))
    gamma = float(model.params["gamma"])
    rho = np.asarray(model.params["rho"], dtype=float)
    tau = np.asarray(model.params["tau"], dtype=float)
    z = e0 + gamma * e1
    benefit_gradient = model.benefit_coef[None, :] * rho[None, :] * np.exp(-rho[None, :] * z)
    penalty_argument = np.log1p(z) - tau[None, :]
    penalty_gradient = (
        model.penalty_coef[None, :] * 2.0 * dsp.softplus(penalty_argument) * dsp.sigmoid(penalty_argument) / (1.0 + z)
    )
    domain_gradient = q * (-benefit_gradient + penalty_gradient)
    if mode == "global_gradient":
        return domain_gradient.sum(axis=1, keepdims=True)
    if mode == "domain_gradient":
        return domain_gradient
    if mode == "channels":
        return np.hstack([q * benefit_gradient, q * penalty_gradient])
    if mode == "contrast":
        return q
    if mode == "contrast_gradient":
        return np.hstack([q, domain_gradient])
    raise ValueError(f"Unknown feature mode {mode!r}")


def fit_residual(
    config: ResidualConfig,
    model: dsp.FittedDSPModel,
    weights: np.ndarray,
    target_residual: np.ndarray,
) -> ResidualModel:
    features = recency_features(model, weights, config.feature_mode)
    feature_mean = features.mean(axis=0)
    feature_scale = np.maximum(features.std(axis=0, ddof=0), SCALE_FLOOR)
    standardized = (features - feature_mean[None, :]) / feature_scale[None, :]
    ridge = Ridge(alpha=config.ridge_alpha, fit_intercept=True)
    ridge.fit(standardized, target_residual)
    return ResidualModel(
        config=config,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        intercept=float(ridge.intercept_),
        coef=np.asarray(ridge.coef_, dtype=float),
    )


def predict_residual(
    residual: ResidualModel,
    base: dsp.FittedDSPModel,
    weights: np.ndarray,
) -> np.ndarray:
    features = recency_features(base, weights, residual.config.feature_mode)
    standardized = (features - residual.feature_mean[None, :]) / residual.feature_scale[None, :]
    return residual.intercept + standardized @ residual.coef


def configs(ridge_alphas: list[float]) -> list[ResidualConfig]:
    values = [ResidualConfig("effective_exposure_baseline", "global_gradient", np.inf)]
    for mode in ("global_gradient", "domain_gradient", "channels", "contrast", "contrast_gradient"):
        for alpha in ridge_alphas:
            values.append(ResidualConfig(f"{mode}_ridge_{alpha:g}", mode, alpha))
    return values


def benchmark_dataset(
    dataset: pooled.Dataset,
    model_configs: list[ResidualConfig],
    seeds: list[int],
    n_splits: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    anchor = nonlinear_anchor(dataset)
    metric_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    for seed in seeds:
        folds = pooled.dataset_folds(dataset, seed, n_splits)
        predictions = {config.name: np.zeros(dataset.n, dtype=float) for config in model_configs}
        for fold_id, (train_idx, test_idx) in enumerate(folds):
            print(f"{dataset.name}: seed={seed} fold={fold_id + 1}/{n_splits}", flush=True)
            base = refit_linear_head(dataset, train_idx, anchor)
            train_base = dsp.predict(base, dataset.weights[train_idx])
            test_base = dsp.predict(base, dataset.weights[test_idx])
            predictions["effective_exposure_baseline"][test_idx] = test_base
            target_residual = dataset.y[train_idx] - train_base
            for config in model_configs:
                if config.name == "effective_exposure_baseline":
                    continue
                residual = fit_residual(config, base, dataset.weights[train_idx], target_residual)
                predictions[config.name][test_idx] = test_base + predict_residual(
                    residual, base, dataset.weights[test_idx]
                )
        full_base = refit_linear_head(dataset, np.arange(dataset.n), anchor)
        full_prediction = dsp.predict(full_base, dataset.weights)
        full_target_residual = dataset.y - full_prediction
        for config in model_configs:
            row = asdict(pooled.metrics(dataset, config.name, seed, predictions[config.name], folds))
            if config.name != "effective_exposure_baseline":
                residual = fit_residual(config, full_base, dataset.weights, full_target_residual)
                row["nominal_param_count"] = anchor.total_param_count + len(residual.coef) + 1
                coefficient_rows.append(
                    {
                        "dataset": dataset.name,
                        "model": config.name,
                        "seed": seed,
                        "residual_coef_norm": float(np.linalg.norm(residual.coef)),
                        "residual_prediction_std": float(
                            np.std(predict_residual(residual, full_base, dataset.weights), ddof=0)
                        ),
                    }
                )
            else:
                row["nominal_param_count"] = anchor.total_param_count
                coefficient_rows.append(
                    {
                        "dataset": dataset.name,
                        "model": config.name,
                        "seed": seed,
                        "residual_coef_norm": 0.0,
                        "residual_prediction_std": 0.0,
                    }
                )
            metric_rows.append(row)
    return pd.DataFrame(metric_rows), pd.DataFrame(coefficient_rows)


def write_plots(summary: pd.DataFrame, output_dir: Path) -> None:
    for metric in ("oof_spearman_mean", "oof_rmse_mean", "fold_mean_regret_at_1_mean", "lower_tail_optimism_mean"):
        figure = px.bar(
            summary,
            x="model",
            y=metric,
            color="dataset",
            barmode="group",
            color_discrete_sequence=["#1a9850", "#fee08b", "#d73027"],
            title=f"Effective-exposure + bounded recency residual: {metric.removesuffix('_mean')}",
        )
        figure.update_layout(xaxis_tickangle=-35, legend_title="Dataset")
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
    parser.add_argument("--ridge-alphas", default="0.01,0.1,1,10,100")
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
    model_configs = configs(pooled.parse_float_list(args.ridge_alphas))
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
