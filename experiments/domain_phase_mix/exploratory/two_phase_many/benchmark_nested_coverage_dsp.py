# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scikit-learn", "scipy", "tabulate"]
# ///
"""Nested-CV benchmark for coverage-augmented effective-exposure DSP.

The model adds three nonnegative mixture-level terms to effective-exposure DSP:

    theta_tv  * TV(w0, w1)
  + theta_hhi * HHI(alpha0 w0 + alpha1 w1)
  + theta_p1  * HHI(w1).

These terms represent phase divergence and concentration, which an additive
per-bucket DSP cannot express. Nonlinear DSP parameters and the complete NNLS
head are refit inside every fold. The production fit uses its historical
coarse-start-only protocol; the 300M fits refine the best coarse starts.
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
from scipy.optimize import minimize, nnls

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import dsp_exact as dsp  # noqa: E402

DEFAULT_OUTPUT_DIR = pooled.REFERENCE_OUTPUTS / "nested_coverage_dsp_20260709"
LOWER_TAIL_FRAC = 0.15
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class CoverageModel:
    base: dsp.FittedDSPModel
    coverage_coef: np.ndarray


@dataclass(frozen=True)
class FitConfig:
    name: str
    use_coverage: bool
    variant_name: str = "effective_exposure"
    coverage_indices: tuple[int, ...] = (0, 1, 2)


def packet(dataset: pooled.Dataset, indices: np.ndarray) -> dsp.PacketData:
    name_col = "run_name" if "run_name" in dataset.frame.columns else "candidate_name"
    return dsp.PacketData(
        frame=dataset.frame.iloc[indices].reset_index(drop=True),
        name_col=name_col,
        y=dataset.y[indices],
        w=dataset.weights[indices],
        m=dataset.m,
        c0=dataset.c0,
        c1=dataset.c1,
        domain_names=list(dataset.domain_names),
    )


def phase_fractions(dataset: pooled.Dataset) -> tuple[float, float]:
    ratio = float(np.median(dataset.c0 / dataset.c1))
    alpha0 = ratio / (1.0 + ratio)
    return alpha0, 1.0 - alpha0


def coverage_features(weights: np.ndarray, alpha0: float, alpha1: float) -> np.ndarray:
    w0 = weights[:, 0, :]
    w1 = weights[:, 1, :]
    aggregate = alpha0 * w0 + alpha1 * w1
    return np.column_stack(
        [
            0.5 * np.abs(w0 - w1).sum(axis=1),
            np.sum(aggregate**2, axis=1),
            np.sum(w1**2, axis=1),
        ]
    )


def assert_unique_mixtures(dataset: pooled.Dataset) -> None:
    flattened = np.ascontiguousarray(dataset.weights.reshape(dataset.n, -1))
    unique = np.unique(flattened, axis=0)
    if len(unique) != dataset.n:
        raise ValueError(f"{dataset.name} has {dataset.n - len(unique)} exact duplicate mixture rows")


def fit_head(
    fit_packet: dsp.PacketData,
    params: dict[str, float | np.ndarray],
    *,
    variant_name: str,
    linear_reg: float,
    use_coverage: bool,
    coverage_indices: tuple[int, ...],
    alpha0: float,
    alpha1: float,
) -> CoverageModel:
    variant = dsp.VARIANTS[variant_name]
    signal, penalty = dsp.features(fit_packet.w, fit_packet.c0, fit_packet.c1, variant, params)
    pieces = [-signal, penalty]
    if use_coverage:
        pieces.append(coverage_features(fit_packet.w, alpha0, alpha1)[:, coverage_indices])
    design = np.hstack(pieces)
    design_mean = design.mean(axis=0, keepdims=True)
    target_mean = float(fit_packet.y.mean())
    centered_design = design - design_mean
    centered_target = fit_packet.y - target_mean
    if linear_reg > 0.0:
        centered_design = np.vstack([centered_design, np.sqrt(linear_reg) * np.eye(design.shape[1])])
        centered_target = np.concatenate([centered_target, np.zeros(design.shape[1])])
    coef, _residual = nnls(centered_design, centered_target, maxiter=20 * design.shape[1])
    intercept = target_mean - float((design_mean @ coef).item())
    m = fit_packet.m
    base = dsp.FittedDSPModel(
        variant=variant,
        params=params,
        intercept=intercept,
        benefit_coef=np.asarray(coef[:m], dtype=float),
        penalty_coef=np.asarray(coef[m : 2 * m], dtype=float),
        domain_names=list(fit_packet.domain_names),
        c0=np.asarray(fit_packet.c0, dtype=float),
        c1=np.asarray(fit_packet.c1, dtype=float),
    )
    coverage_coef = np.zeros(3, dtype=float) if use_coverage else np.asarray([], dtype=float)
    if use_coverage:
        coverage_coef[np.asarray(coverage_indices, dtype=int)] = coef[2 * m :]
    return CoverageModel(base=base, coverage_coef=coverage_coef)


def predict(model: CoverageModel, weights: np.ndarray, alpha0: float, alpha1: float) -> np.ndarray:
    prediction = dsp.predict(model.base, weights)
    if len(model.coverage_coef):
        prediction = prediction + coverage_features(weights, alpha0, alpha1) @ model.coverage_coef
    return prediction


def profile_objective(
    theta: np.ndarray,
    fit_packet: dsp.PacketData,
    *,
    config: FitConfig,
    linear_reg: float,
    alpha0: float,
    alpha1: float,
) -> float:
    params = dsp.unpack_theta(theta, dsp.VARIANTS[config.variant_name], fit_packet.m)
    model = fit_head(
        fit_packet,
        params,
        variant_name=config.variant_name,
        linear_reg=linear_reg,
        use_coverage=config.use_coverage,
        coverage_indices=config.coverage_indices,
        alpha0=alpha0,
        alpha1=alpha1,
    )
    prediction = predict(model, fit_packet.w, alpha0, alpha1)
    rmse = float(np.sqrt(np.mean((prediction - fit_packet.y) ** 2)))
    tail_count = max(5, int(np.ceil(LOWER_TAIL_FRAC * len(fit_packet.y))))
    tail_idx = np.argsort(prediction)[:tail_count]
    optimism = float(np.mean(np.maximum(fit_packet.y[tail_idx] - prediction[tail_idx], 0.0)))
    return rmse + 0.5 * optimism


def fit_model(
    dataset: pooled.Dataset,
    indices: np.ndarray,
    config: FitConfig,
    *,
    linear_reg: float,
    maxiter: int,
    coarse_top_k: int,
) -> CoverageModel:
    fit_packet = packet(dataset, indices)
    variant = dsp.VARIANTS[config.variant_name]
    alpha0, alpha1 = phase_fractions(dataset)
    starts = dsp.start_bank(fit_packet, variant)
    scored = sorted(
        (
            profile_objective(
                start,
                fit_packet,
                config=config,
                linear_reg=linear_reg,
                alpha0=alpha0,
                alpha1=alpha1,
            ),
            start,
        )
        for start in starts
    )
    best_value, best_theta = scored[0]
    if maxiter > 0:
        coord_bounds = dsp.bounds(variant, fit_packet.m)
        for _coarse_value, start in scored[:coarse_top_k]:
            result = minimize(
                lambda theta: profile_objective(
                    np.asarray(theta, dtype=float),
                    fit_packet,
                    config=config,
                    linear_reg=linear_reg,
                    alpha0=alpha0,
                    alpha1=alpha1,
                ),
                start,
                method="L-BFGS-B",
                bounds=coord_bounds,
                options={"maxiter": maxiter, "ftol": 1e-7, "maxls": 20},
            )
            if float(result.fun) < best_value:
                best_value = float(result.fun)
                best_theta = np.asarray(result.x, dtype=float)
    params = dsp.unpack_theta(best_theta, variant, fit_packet.m)
    return fit_head(
        fit_packet,
        params,
        variant_name=config.variant_name,
        linear_reg=linear_reg,
        use_coverage=config.use_coverage,
        coverage_indices=config.coverage_indices,
        alpha0=alpha0,
        alpha1=alpha1,
    )


def dataset_linear_reg(dataset: pooled.Dataset) -> float:
    return 1e-6 if dataset.name == "production_uncheatable" else 0.01


def dataset_maxiter(dataset: pooled.Dataset, requested_300m_maxiter: int) -> int:
    return 0 if dataset.name == "production_uncheatable" else requested_300m_maxiter


def benchmark_dataset(
    dataset: pooled.Dataset,
    configs: list[FitConfig],
    seeds: list[int],
    n_splits: int,
    maxiter_300m: int,
    coarse_top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assert_unique_mixtures(dataset)
    alpha0, alpha1 = phase_fractions(dataset)
    metric_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    for seed in seeds:
        folds = pooled.dataset_folds(dataset, seed, n_splits)
        predictions = {config.name: np.zeros(dataset.n, dtype=float) for config in configs}
        for fold_id, (train_idx, test_idx) in enumerate(folds):
            print(f"{dataset.name}: seed={seed} fold={fold_id + 1}/{n_splits}", flush=True)
            for config in configs:
                model = fit_model(
                    dataset,
                    train_idx,
                    config,
                    linear_reg=dataset_linear_reg(dataset),
                    maxiter=dataset_maxiter(dataset, maxiter_300m),
                    coarse_top_k=coarse_top_k,
                )
                predictions[config.name][test_idx] = predict(model, dataset.weights[test_idx], alpha0, alpha1)
                parameter_rows.append(
                    {
                        "dataset": dataset.name,
                        "model": config.name,
                        "seed": seed,
                        "fold": fold_id,
                        "gamma": float(model.base.params["gamma"]),
                        "theta_tv": float(model.coverage_coef[0]) if len(model.coverage_coef) else 0.0,
                        "theta_hhi_aggregate": float(model.coverage_coef[1]) if len(model.coverage_coef) else 0.0,
                        "theta_hhi_phase1": float(model.coverage_coef[2]) if len(model.coverage_coef) else 0.0,
                    }
                )
        for config in configs:
            row = asdict(pooled.metrics(dataset, config.name, seed, predictions[config.name], folds))
            row["nominal_param_count"] = 4 * dataset.m + 2 + len(config.coverage_indices) * int(config.use_coverage)
            metric_rows.append(row)
    return pd.DataFrame(metric_rows), pd.DataFrame(parameter_rows)


def write_plots(summary: pd.DataFrame, output_dir: Path) -> None:
    for metric in ("oof_spearman_mean", "oof_rmse_mean", "fold_mean_regret_at_1_mean", "lower_tail_optimism_mean"):
        figure = px.bar(
            summary,
            x="model",
            y=metric,
            color="dataset",
            barmode="group",
            color_discrete_sequence=["#1a9850", "#fee08b", "#d73027"],
            title=f"Nested effective-exposure DSP: {metric.removesuffix('_mean')}",
        )
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
    parser.add_argument("--maxiter-300m", type=int, default=16)
    parser.add_argument("--coarse-top-k", type=int, default=2)
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
    configs = [
        FitConfig("effective_exposure", False),
        FitConfig("effective_exposure_coverage", True),
    ]
    seeds = pooled.parse_int_list(args.seeds)

    metric_frames = []
    parameter_frames = []
    for name in selected:
        metrics, parameters = benchmark_dataset(
            loaders[name](),
            configs,
            seeds,
            args.n_splits,
            args.maxiter_300m,
            args.coarse_top_k,
        )
        metric_frames.append(metrics)
        parameter_frames.append(parameters)
    raw = pd.concat(metric_frames, ignore_index=True)
    parameters = pd.concat(parameter_frames, ignore_index=True)
    summary = pooled.summarize(raw)
    raw.to_csv(args.output_dir / "cv_metrics_by_seed.csv", index=False)
    parameters.to_csv(args.output_dir / "fold_parameters.csv", index=False)
    summary.to_csv(args.output_dir / "cv_summary.csv", index=False)
    write_plots(summary, args.output_dir)
    print(summary.to_string(index=False))
    print(
        parameters.groupby(["dataset", "model"])[["gamma", "theta_tv", "theta_hhi_aggregate", "theta_hhi_phase1"]]
        .agg(["mean", "std"])
        .to_string()
    )
    print(f"Wrote benchmark artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
