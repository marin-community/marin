# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scikit-learn", "scipy", "tabulate"]
# ///
"""Benchmark pooled nonlinear parameters in effective-exposure DSP.

The full DSP model fits one saturation rate ``rho_i`` and one overexposure
threshold ``tau_i`` per bucket. This script evaluates three lower-dimensional
limits: shared rho, shared tau, and both shared. Per-bucket linear benefit and
penalty coefficients remain free, and every model uses phase-TV plus aggregate
HHI geometry penalties.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_joint_phase_correspondence_dsp as joint,
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

DEFAULT_OUTPUT_DIR = pooled.REFERENCE_OUTPUTS / "pooled_nonlinear_phase_dsp_20260710"
LOWER_TAIL_FRAC = 0.15


@dataclass(frozen=True)
class PooledConfig:
    name: str
    shared_rho: bool
    shared_tau: bool
    variant_name: str = "effective_exposure"


@dataclass(frozen=True)
class PooledModel:
    config: PooledConfig
    fitted: geometry.CoverageModel


def configs() -> tuple[PooledConfig, ...]:
    return (
        PooledConfig("shared_rho", True, False),
        PooledConfig("shared_tau", False, True),
        PooledConfig("shared_rho_tau", True, True),
        PooledConfig("shared_rho_split", True, False, "split_saturation_penalty"),
    )


def encode_params(params: dict[str, float | np.ndarray], config: PooledConfig) -> np.ndarray:
    rho = np.asarray(params["rho"], dtype=float)
    tau = np.asarray(params["tau"], dtype=float)
    pieces = [
        np.asarray([np.mean(np.log(rho))], dtype=float) if config.shared_rho else np.log(rho),
        np.asarray([np.median(tau)], dtype=float) if config.shared_tau else tau,
    ]
    if config.variant_name == "split_saturation_penalty":
        pieces.append(
            np.log(
                np.asarray(
                    [params["gamma_saturation"], params["gamma_penalty"]],
                    dtype=float,
                )
            )
        )
    else:
        pieces.append(np.asarray([np.log(float(params["gamma"]))], dtype=float))
    return np.concatenate(pieces)


def decode_params(theta: np.ndarray, config: PooledConfig, num_domains: int) -> dict[str, float | np.ndarray]:
    cursor = 0
    rho_count = 1 if config.shared_rho else num_domains
    rho = np.exp(np.clip(theta[cursor : cursor + rho_count], np.log(1e-4), np.log(2.0)))
    cursor += rho_count
    if config.shared_rho:
        rho = np.full(num_domains, float(rho[0]), dtype=float)

    tau_count = 1 if config.shared_tau else num_domains
    tau = np.clip(theta[cursor : cursor + tau_count], -2.0, 8.0)
    cursor += tau_count
    if config.shared_tau:
        tau = np.full(num_domains, float(tau[0]), dtype=float)

    if config.variant_name == "split_saturation_penalty":
        gamma_saturation = float(np.exp(np.clip(theta[cursor], np.log(1e-4), np.log(100.0))))
        gamma_penalty = float(np.exp(np.clip(theta[cursor + 1], np.log(1e-4), np.log(100.0))))
        return {
            "rho": rho,
            "tau": tau,
            "gamma_saturation": gamma_saturation,
            "gamma_penalty": gamma_penalty,
        }
    gamma = float(np.exp(np.clip(theta[cursor], np.log(1e-4), np.log(100.0))))
    return {"rho": rho, "tau": tau, "gamma": gamma}


def parameter_bounds(config: PooledConfig, num_domains: int) -> list[tuple[float, float]]:
    rho_count = 1 if config.shared_rho else num_domains
    tau_count = 1 if config.shared_tau else num_domains
    phase_count = 2 if config.variant_name == "split_saturation_penalty" else 1
    return (
        [(np.log(1e-4), np.log(2.0))] * rho_count
        + [(-2.0, 8.0)] * tau_count
        + [(np.log(1e-4), np.log(100.0))] * phase_count
    )


def start_bank(packet: dsp.PacketData, config: PooledConfig) -> tuple[np.ndarray, ...]:
    variant = dsp.VARIANTS[config.variant_name]
    starts = [
        encode_params(dsp.unpack_theta(start, variant, packet.m), config) for start in dsp.start_bank(packet, variant)
    ]
    unique: list[np.ndarray] = []
    for start in starts:
        if not any(np.allclose(start, previous) for previous in unique):
            unique.append(start)
    return tuple(unique)


def fit_head(
    packet: dsp.PacketData,
    params: dict[str, float | np.ndarray],
    config: PooledConfig,
    linear_reg: float,
    alpha0: float,
    alpha1: float,
) -> geometry.CoverageModel:
    return geometry.fit_head(
        packet,
        params,
        variant_name=config.variant_name,
        linear_reg=linear_reg,
        use_coverage=True,
        coverage_indices=(0, 1),
        alpha0=alpha0,
        alpha1=alpha1,
    )


def profile_objective(
    theta: np.ndarray,
    packet: dsp.PacketData,
    config: PooledConfig,
    linear_reg: float,
    alpha0: float,
    alpha1: float,
) -> float:
    model = fit_head(
        packet,
        decode_params(theta, config, packet.m),
        config,
        linear_reg,
        alpha0,
        alpha1,
    )
    prediction = geometry.predict(model, packet.w, alpha0, alpha1)
    rmse = float(np.sqrt(np.mean((prediction - packet.y) ** 2)))
    tail_count = max(5, int(np.ceil(LOWER_TAIL_FRAC * len(packet.y))))
    tail = np.argsort(prediction)[:tail_count]
    optimism = float(np.mean(np.maximum(packet.y[tail] - prediction[tail], 0.0)))
    return rmse + 0.5 * optimism


def fit_model(
    dataset: pooled.Dataset,
    indices: np.ndarray,
    config: PooledConfig,
    maxiter: int,
    coarse_top_k: int,
) -> PooledModel:
    packet = geometry.packet(dataset, indices)
    alpha0, alpha1 = geometry.phase_fractions(dataset)
    linear_reg = geometry.dataset_linear_reg(dataset)
    scored = sorted(
        (
            profile_objective(start, packet, config, linear_reg, alpha0, alpha1),
            start,
        )
        for start in start_bank(packet, config)
    )
    best_value, best_theta = scored[0]
    if maxiter > 0:
        bounds = parameter_bounds(config, packet.m)
        for _coarse_value, start in scored[:coarse_top_k]:
            result = minimize(
                lambda theta: profile_objective(
                    np.asarray(theta, dtype=float),
                    packet,
                    config,
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
    params = decode_params(best_theta, config, packet.m)
    return PooledModel(
        config=config,
        fitted=fit_head(packet, params, config, linear_reg, alpha0, alpha1),
    )


def predict(model: PooledModel, dataset: pooled.Dataset, indices: np.ndarray) -> np.ndarray:
    alpha0, alpha1 = geometry.phase_fractions(dataset)
    return geometry.predict(model.fitted, dataset.weights[indices], alpha0, alpha1)


def folds_for(dataset: pooled.Dataset, seed: int, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if "phase_correspondence_key" in dataset.frame.columns:
        return joint.grouped_folds(dataset.frame, seed, n_splits)
    return pooled.dataset_folds(dataset, seed, n_splits)


def parameter_count(dataset: pooled.Dataset, config: PooledConfig) -> int:
    phase_parameters = 2 if config.variant_name == "split_saturation_penalty" else 1
    nonlinear = (1 if config.shared_rho else dataset.m) + (1 if config.shared_tau else dataset.m) + phase_parameters
    linear = 2 * dataset.m + 1 + 2
    return nonlinear + linear


def benchmark_dataset(
    dataset: pooled.Dataset,
    model_configs: tuple[PooledConfig, ...],
    seeds: list[int],
    n_splits: int,
    maxiter: int,
    coarse_top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    parameter_rows = []
    for seed in seeds:
        folds = folds_for(dataset, seed, n_splits)
        predictions = {config.name: np.zeros(dataset.n, dtype=float) for config in model_configs}
        for fold_id, (train_idx, test_idx) in enumerate(folds):
            print(f"{dataset.name}: seed={seed} fold={fold_id + 1}/{n_splits}", flush=True)
            for config in model_configs:
                model = fit_model(dataset, train_idx, config, maxiter, coarse_top_k)
                predictions[config.name][test_idx] = predict(model, dataset, test_idx)
                params = model.fitted.base.params
                parameter_rows.append(
                    {
                        "dataset": dataset.name,
                        "model": config.name,
                        "seed": seed,
                        "fold": fold_id,
                        "rho_geometric_mean": float(np.exp(np.mean(np.log(params["rho"])))),
                        "rho_log_sd": float(np.std(np.log(params["rho"]), ddof=0)),
                        "tau_mean": float(np.mean(params["tau"])),
                        "tau_sd": float(np.std(params["tau"], ddof=0)),
                        "gamma": float(params.get("gamma", np.nan)),
                        "gamma_saturation": float(params.get("gamma_saturation", np.nan)),
                        "gamma_penalty": float(params.get("gamma_penalty", np.nan)),
                        "theta_tv": float(model.fitted.coverage_coef[0]),
                        "theta_hhi_aggregate": float(model.fitted.coverage_coef[1]),
                    }
                )
        for config in model_configs:
            row = asdict(pooled.metrics(dataset, config.name, seed, predictions[config.name], folds))
            row["nominal_param_count"] = parameter_count(dataset, config)
            metric_rows.append(row)
    return pd.DataFrame(metric_rows), pd.DataFrame(parameter_rows)


def external_evaluation(
    fit_dataset: pooled.Dataset,
    external: pooled.Dataset,
    model_configs: tuple[PooledConfig, ...],
    maxiter: int,
    coarse_top_k: int,
) -> pd.DataFrame:
    rows = []
    for config in model_configs:
        model = fit_model(fit_dataset, np.arange(fit_dataset.n), config, maxiter, coarse_top_k)
        row = joint.external_metrics(config.name, external.y, predict(model, external, np.arange(external.n)))
        row["dataset"] = fit_dataset.name
        row["external_rows"] = external.n
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=joint.PACKET)
    parser.add_argument("--one-phase-source", type=Path, default=joint.ONE_PHASE_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--datasets", default="300m_uncheatable,300m_table9,production_uncheatable")
    parser.add_argument("--models", default=",".join(config.name for config in configs()))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--maxiter-300m", type=int, default=16)
    parser.add_argument("--maxiter-production", type=int, default=16)
    parser.add_argument("--coarse-top-k", type=int, default=2)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.packet)
    domains = pooled.load_300m_dataset("table9").domain_names
    frame = joint.attach_single_phase_weights(frame, args.one_phase_source, domains)
    dataset_by_name = {}
    external_by_name = {}
    for objective, target in joint.TARGET_COLUMNS.items():
        dataset = joint.dataset_from_frame(
            objective,
            frame.loc[frame["split"].eq("train") | frame["policy_family"].eq("single_phase")].copy(),
            target,
        )
        dataset_by_name[dataset.name] = dataset
        external_by_name[dataset.name] = joint.dataset_from_frame(
            objective,
            frame.loc[frame["split"].eq("heldout") & frame["policy_family"].eq("two_phase")].copy(),
            target,
        )
    production = pooled.load_production_dataset()
    dataset_by_name[production.name] = production

    selected_names = [part.strip() for part in args.datasets.split(",") if part.strip()]
    unknown = sorted(set(selected_names).difference(dataset_by_name))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}")
    config_by_name = {config.name: config for config in configs()}
    selected_models = [part.strip() for part in args.models.split(",") if part.strip()]
    unknown_models = sorted(set(selected_models).difference(config_by_name))
    if unknown_models:
        raise ValueError(f"Unknown models: {unknown_models}")
    model_configs = tuple(config_by_name[name] for name in selected_models)

    metric_frames = []
    parameter_frames = []
    external_frames = []
    for name in selected_names:
        dataset = dataset_by_name[name]
        maxiter = args.maxiter_production if name == "production_uncheatable" else args.maxiter_300m
        metrics, parameters = benchmark_dataset(
            dataset,
            model_configs,
            pooled.parse_int_list(args.seeds),
            args.n_splits,
            maxiter,
            args.coarse_top_k,
        )
        metric_frames.append(metrics)
        parameter_frames.append(parameters)
        if name in external_by_name:
            external_frames.append(
                external_evaluation(
                    dataset,
                    external_by_name[name],
                    model_configs,
                    maxiter,
                    args.coarse_top_k,
                )
            )

    metrics = pd.concat(metric_frames, ignore_index=True)
    parameters = pd.concat(parameter_frames, ignore_index=True)
    summary = pooled.summarize(metrics)
    external = pd.concat(external_frames, ignore_index=True) if external_frames else pd.DataFrame()
    metrics.to_csv(args.output_dir / "cv_metrics_by_seed.csv", index=False)
    parameters.to_csv(args.output_dir / "fold_parameters.csv", index=False)
    summary.to_csv(args.output_dir / "cv_summary.csv", index=False)
    external.to_csv(args.output_dir / "external_two_phase_heldout_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(external.to_string(index=False))
    print(f"Wrote pooled nonlinear benchmark to {args.output_dir}")


if __name__ == "__main__":
    main()
