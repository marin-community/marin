# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "fsspec",
#   "gcsfs",
#   "matplotlib",
#   "numpy",
#   "pandas",
#   "plotly",
#   "pyarrow",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Partially pool effective-exposure DSP saturation rates.

Hard sharing of ``rho`` improves both 300M objectives but loses some production
fit, while an unconstrained ``rho_i`` per bucket is statistically expensive.
This model interpolates between them with one learned spread:

    log rho_i = mu_rho + lambda_rho (log rho_i^prior - mean(log rho^prior)).

The prior is computed without benchmark labels from median realized bucket
exposure. ``lambda_rho=0`` is hard sharing and ``lambda_rho=1`` preserves the
unlabeled exposure-derived dispersion. Per-bucket overexposure thresholds and
linear benefit/penalty heads remain free.
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
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_centered_recency_residual as centered,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_nested_coverage_dsp as geometry,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import (  # noqa: E402
    dsp_exact as dsp,
)

DEFAULT_OUTPUT_DIR = pooled.REFERENCE_OUTPUTS / "hierarchical_rho_effective_exposure_20260710"
LOWER_TAIL_FRAC = 0.15
RHO_MIN = 1e-4
RHO_MAX = 2.0
SPREAD_MIN = 1e-3
SPREAD_MAX = 3.0
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class Config:
    """Hierarchical spread penalty."""

    spread_l2: float

    @property
    def name(self) -> str:
        return f"hierarchical_rho_spread_l2_{self.spread_l2:g}"


@dataclass(frozen=True)
class FittedModel:
    """Hierarchical saturation geometry plus the established linear head."""

    config: Config
    fitted: geometry.CoverageModel
    rho_prior: np.ndarray
    rho_spread: float

    @property
    def parameter_count(self) -> int:
        num_domains = len(self.rho_prior)
        return 3 * num_domains + 6


def empirical_rho_prior(packet: dsp.PacketData) -> np.ndarray:
    """Estimate bucket response scales from unlabeled realized exposure."""
    exposure = packet.w[:, 0, :] * packet.c0[None, :] + packet.w[:, 1, :] * packet.c1[None, :]
    positive = np.where(exposure > 1e-8, exposure, np.nan)
    median = np.nanmedian(positive, axis=0)
    fallback = float(np.nanmedian(median[np.isfinite(median)]))
    median = np.where(np.isfinite(median), median, fallback)
    return np.clip(1.0 / np.maximum(median, 1e-3), RHO_MIN, RHO_MAX)


def decode_params(
    theta: np.ndarray,
    rho_prior: np.ndarray,
) -> tuple[dict[str, float | np.ndarray], float]:
    """Decode one rho mean, one prior spread, tau_i, and gamma."""
    num_domains = len(rho_prior)
    rho_mean = float(np.clip(theta[0], np.log(RHO_MIN), np.log(RHO_MAX)))
    spread = float(np.exp(np.clip(theta[1], np.log(SPREAD_MIN), np.log(SPREAD_MAX))))
    centered_prior = np.log(rho_prior) - float(np.mean(np.log(rho_prior)))
    rho = np.clip(np.exp(rho_mean + spread * centered_prior), RHO_MIN, RHO_MAX)
    tau = np.clip(theta[2 : 2 + num_domains], -2.0, 8.0)
    gamma = float(np.exp(np.clip(theta[-1], np.log(1e-4), np.log(100.0))))
    return {"rho": rho, "tau": tau, "gamma": gamma}, spread


def start_bank(packet: dsp.PacketData) -> tuple[np.ndarray, ...]:
    """Project standard DSP starts into the hierarchical parameterization."""
    variant = dsp.VARIANTS["effective_exposure"]
    starts = []
    for raw in dsp.start_bank(packet, variant):
        params = dsp.unpack_theta(raw, variant, packet.m)
        rho_mean = float(np.mean(np.log(np.asarray(params["rho"], dtype=float))))
        tau = np.asarray(params["tau"], dtype=float)
        gamma = float(params["gamma"])
        for spread in (0.05, 0.25, 0.5, 1.0, 2.0):
            starts.append(np.concatenate([[rho_mean, np.log(spread)], tau, [np.log(gamma)]]))
    unique = []
    for start in starts:
        if not any(np.allclose(start, previous) for previous in unique):
            unique.append(start)
    return tuple(unique)


def fit_head(
    packet: dsp.PacketData,
    params: dict[str, float | np.ndarray],
    linear_reg: float,
    alpha0: float,
    alpha1: float,
) -> geometry.CoverageModel:
    """Fit the established nonnegative heads and geometry terms."""
    return geometry.fit_head(
        packet,
        params,
        variant_name="effective_exposure",
        linear_reg=linear_reg,
        use_coverage=True,
        coverage_indices=(0, 1),
        alpha0=alpha0,
        alpha1=alpha1,
    )


def profile_objective(
    theta: np.ndarray,
    packet: dsp.PacketData,
    config: Config,
    rho_prior: np.ndarray,
    linear_reg: float,
    alpha0: float,
    alpha1: float,
) -> float:
    """Profile prediction error plus shrinkage toward hard sharing."""
    params, spread = decode_params(theta, rho_prior)
    model = fit_head(packet, params, linear_reg, alpha0, alpha1)
    prediction = geometry.predict(model, packet.w, alpha0, alpha1)
    residual = prediction - packet.y
    tail_count = max(5, int(np.ceil(LOWER_TAIL_FRAC * len(packet.y))))
    tail = np.argsort(prediction)[:tail_count]
    optimism = float(np.mean(np.maximum(-residual[tail], 0.0)))
    spread_penalty = config.spread_l2 * spread**2 / len(packet.y)
    return float(np.sqrt(np.mean(residual**2)) + 0.5 * optimism + spread_penalty)


def fit_model(
    dataset: pooled.Dataset,
    indices: np.ndarray,
    config: Config,
    maxiter: int,
    coarse_top_k: int,
) -> FittedModel:
    """Fit the hierarchical nonlinear geometry inside one fold."""
    packet = geometry.packet(dataset, indices)
    rho_prior = empirical_rho_prior(packet)
    alpha0, alpha1 = geometry.phase_fractions(dataset)
    linear_reg = geometry.dataset_linear_reg(dataset)
    scored = [
        (
            profile_objective(start, packet, config, rho_prior, linear_reg, alpha0, alpha1),
            start,
        )
        for start in start_bank(packet)
    ]
    scored.sort(key=lambda item: item[0])
    best_value, best_theta = scored[0]
    bounds = [
        (np.log(RHO_MIN), np.log(RHO_MAX)),
        (np.log(SPREAD_MIN), np.log(SPREAD_MAX)),
        *[(-2.0, 8.0)] * packet.m,
        (np.log(1e-4), np.log(100.0)),
    ]
    for _value, start in scored[:coarse_top_k]:
        result = minimize(
            lambda theta: profile_objective(
                np.asarray(theta, dtype=float),
                packet,
                config,
                rho_prior,
                linear_reg,
                alpha0,
                alpha1,
            ),
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-7, "maxls": 20},
        )
        if float(result.fun) < best_value:
            best_value = float(result.fun)
            best_theta = np.asarray(result.x, dtype=float)
    params, spread = decode_params(best_theta, rho_prior)
    return FittedModel(
        config=config,
        fitted=fit_head(packet, params, linear_reg, alpha0, alpha1),
        rho_prior=rho_prior,
        rho_spread=spread,
    )


def predict(model: FittedModel, dataset: pooled.Dataset, indices: np.ndarray) -> np.ndarray:
    """Predict held-out rows."""
    alpha0, alpha1 = geometry.phase_fractions(dataset)
    return geometry.predict(model.fitted, dataset.weights[indices], alpha0, alpha1)


def benchmark_dataset(
    dataset: pooled.Dataset,
    model_configs: list[Config],
    seeds: list[int],
    n_splits: int,
    maxiter: int,
    coarse_top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run fully refit grouped CV for one dataset."""
    metric_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    for seed in seeds:
        folds = centered.folds_for(dataset, seed, n_splits)
        predictions = {config.name: np.zeros(dataset.n, dtype=float) for config in model_configs}
        for fold_id, (train_indices, test_indices) in enumerate(folds):
            print(f"{dataset.name}: seed={seed} fold={fold_id + 1}/{n_splits}", flush=True)
            for config in model_configs:
                model = fit_model(dataset, train_indices, config, maxiter, coarse_top_k)
                predictions[config.name][test_indices] = predict(model, dataset, test_indices)
                params = model.fitted.base.params
                parameter_rows.append(
                    {
                        "dataset": dataset.name,
                        "model": config.name,
                        "seed": seed,
                        "fold": fold_id,
                        "rho_spread": model.rho_spread,
                        "rho_log_sd": float(np.std(np.log(params["rho"]))),
                        "gamma": float(params["gamma"]),
                        "theta_tv": float(model.fitted.coverage_coef[0]),
                        "theta_hhi_aggregate": float(model.fitted.coverage_coef[1]),
                    }
                )
        for config in model_configs:
            row = asdict(pooled.metrics(dataset, config.name, seed, predictions[config.name], folds))
            row["nominal_param_count"] = 3 * dataset.m + 6
            metric_rows.append(row)
    return pd.DataFrame(metric_rows), pd.DataFrame(parameter_rows)


def write_plot(summary: pd.DataFrame, output_dir: Path) -> None:
    """Write a compact cross-swarm comparison."""
    long = summary.melt(
        id_vars=["dataset", "model"],
        value_vars=["oof_rmse_mean", "oof_spearman_mean", "fold_mean_regret_at_1_mean"],
        var_name="metric",
        value_name="value",
    )
    figure = px.bar(
        long,
        x="model",
        y="value",
        color="model",
        facet_row="dataset",
        facet_col="metric",
        color_discrete_sequence=px.colors.diverging.RdYlGn_r,
        title="Hierarchical saturation-rate pooling",
    )
    figure.update_layout(showlegend=False, height=1000)
    figure.write_html(output_dir / "crossswarm_cv_comparison.html", include_plotlyjs="cdn", config=PLOT_CONFIG)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--datasets",
        default=f"{centered.STARCODER_NAME},300m_uncheatable,300m_table9,production_uncheatable",
    )
    parser.add_argument("--spread-l2-values", default="0,0.001,0.01,0.1,1")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--maxiter-300m", type=int, default=16)
    parser.add_argument("--maxiter-production", type=int, default=8)
    parser.add_argument("--maxiter-starcoder", type=int, default=30)
    parser.add_argument("--coarse-top-k", type=int, default=2)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets, _external = centered.load_datasets()
    names = [part.strip() for part in args.datasets.split(",") if part.strip()]
    unknown = sorted(set(names).difference(datasets))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    model_configs = [Config(value) for value in pooled.parse_float_list(args.spread_l2_values)]
    metric_frames = []
    parameter_frames = []
    for name in names:
        maxiter = args.maxiter_300m
        if name == centered.STARCODER_NAME:
            maxiter = args.maxiter_starcoder
        elif name == "production_uncheatable":
            maxiter = args.maxiter_production
        metrics, parameters = benchmark_dataset(
            datasets[name],
            model_configs,
            pooled.parse_int_list(args.seeds),
            args.n_splits,
            maxiter,
            args.coarse_top_k,
        )
        metric_frames.append(metrics)
        parameter_frames.append(parameters)
    raw = pd.concat(metric_frames, ignore_index=True)
    summary = pooled.summarize(raw)
    parameters = pd.concat(parameter_frames, ignore_index=True)
    raw.to_csv(args.output_dir / "cv_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "cv_summary.csv", index=False)
    parameters.to_csv(args.output_dir / "cv_parameters.csv", index=False)
    write_plot(summary, args.output_dir)
    report = [
        "# Hierarchical saturation-rate pooling",
        "",
        "One learned spread interpolates between hard sharing and an unlabeled exposure-derived bucket prior.",
        "",
        summary.to_markdown(index=False),
        "",
        "## Parameter stability",
        "",
        parameters.groupby(["dataset", "model"])[["rho_spread", "rho_log_sd", "gamma"]]
        .agg(["mean", "std"])
        .reset_index()
        .to_markdown(index=False),
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report))
    print(summary.to_string(index=False))
    parameter_summary = parameters.groupby(["dataset", "model"])[["rho_spread", "rho_log_sd", "gamma"]].agg(
        ["mean", "std"]
    )
    print(parameter_summary.to_string())
    print(f"Wrote hierarchical-rho benchmark to {args.output_dir}")


if __name__ == "__main__":
    main()
