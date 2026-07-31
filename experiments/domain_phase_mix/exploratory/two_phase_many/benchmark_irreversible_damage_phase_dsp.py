# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scikit-learn", "scipy", "tabulate"]
# ///
"""Benchmark irreversible repetition damage in two-phase DSP.

Effective-exposure DSP uses the same recency-weighted exposure for useful
learning and overexposure damage. This screen tests the constrained alternative

    u_benefit = e0 + gamma * e1
    u_damage  = e0 + gamma**q * e1,

with fixed ``q=0`` for irreversible physical repetition damage and ``q=1`` for
the effective-exposure control. The primary comparison is replicated with full
per-bucket saturation rates and with one shared saturation rate. Geometry is
held at phase TV plus aggregate HHI, except for one pre-screened Hellinger
overlap control.

This script performs local fitting and evaluation only. It does not materialize
mixture proposals or submit training jobs.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls
from scipy.stats import spearmanr

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
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    diagnose_matched_phase_ordering as ordering,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import (  # noqa: E402
    dsp_exact as dsp,
)

DEFAULT_OUTPUT_DIR = pooled.REFERENCE_OUTPUTS / "irreversible_damage_phase_dsp_20260710"
LOWER_TAIL_FRAC = 0.15
RHO_MIN = 1e-4
RHO_MAX = 2.0
TAU_MIN = -2.0
TAU_MAX = 8.0
GAMMA_MIN = 1e-4
GAMMA_MAX = 100.0
SCALE_FLOOR = 1e-12
PENALTY_ACTIVE_FRACTION = 0.10
PHASE_SHARE_SD_MIN = 0.05
GEOMETRY_REDUNDANT_RHO = 0.97
NOISE_SD = {
    "300m_uncheatable": 0.0011270969148995812,
    "300m_table9": 0.003325782675083218,
}


class GeometryKind(StrEnum):
    TV = "tv"
    HELLINGER = "hellinger"


@dataclass(frozen=True)
class ModelConfig:
    name: str
    shared_rho: bool
    damage_q: float
    geometry_kind: GeometryKind
    role: str


@dataclass(frozen=True)
class FittedModel:
    config: ModelConfig
    rho: np.ndarray
    tau: np.ndarray
    gamma: float
    intercept: float
    benefit_coef: np.ndarray
    penalty_coef: np.ndarray
    geometry_coef: float
    aggregate_hhi_coef: float


def configs(include_hellinger: bool) -> tuple[ModelConfig, ...]:
    base = [
        ModelConfig("full_effective_tv", False, 1.0, GeometryKind.TV, "deployment_control"),
        ModelConfig("shared_effective_tv", True, 1.0, GeometryKind.TV, "pooling_control"),
        ModelConfig("shared_irreversible_tv", True, 0.0, GeometryKind.TV, "damage_test"),
        ModelConfig("full_irreversible_tv", False, 0.0, GeometryKind.TV, "damage_confound_check"),
    ]
    if include_hellinger:
        base.append(
            ModelConfig(
                "shared_effective_hellinger",
                True,
                1.0,
                GeometryKind.HELLINGER,
                "geometry_control",
            )
        )
    return tuple(base)


def phase_fractions(dataset: pooled.Dataset) -> tuple[float, float]:
    return geometry.phase_fractions(dataset)


def geometry_features(weights: np.ndarray) -> pd.DataFrame:
    w0 = weights[:, 0, :]
    w1 = weights[:, 1, :]
    midpoint = 0.5 * (w0 + w1)
    eps = 1e-30
    js = 0.5 * np.sum(
        np.where(w0 > 0.0, w0 * np.log((w0 + eps) / (midpoint + eps)), 0.0),
        axis=1,
    )
    js += 0.5 * np.sum(
        np.where(w1 > 0.0, w1 * np.log((w1 + eps) / (midpoint + eps)), 0.0),
        axis=1,
    )
    return pd.DataFrame(
        {
            "tv": 0.5 * np.abs(w0 - w1).sum(axis=1),
            "hellinger": 1.0 - np.sqrt(w0 * w1).sum(axis=1),
            "js": js,
        }
    )


def selected_geometry(weights_3d: np.ndarray, kind: GeometryKind) -> np.ndarray:
    return geometry_features(weights_3d)[str(kind)].to_numpy(dtype=float)


def aggregate_hhi(weights: np.ndarray, alpha0: float, alpha1: float) -> np.ndarray:
    aggregate = alpha0 * weights[:, 0, :] + alpha1 * weights[:, 1, :]
    return np.sum(aggregate**2, axis=1)


def nonlinear_features(
    weights: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    rho: np.ndarray,
    tau: np.ndarray,
    gamma: float,
    damage_q: float,
) -> tuple[np.ndarray, np.ndarray]:
    e0 = weights[:, 0, :] * c0[None, :]
    e1 = weights[:, 1, :] * c1[None, :]
    benefit_exposure = e0 + gamma * e1
    damage_gamma = gamma**damage_q
    damage_exposure = e0 + damage_gamma * e1
    signal = 1.0 - np.exp(-rho[None, :] * benefit_exposure)
    penalty = dsp.softplus(np.log1p(damage_exposure) - tau[None, :]) ** 2
    return signal, penalty


def fit_nonnegative_head(
    design: np.ndarray,
    target: np.ndarray,
    linear_reg: float,
) -> tuple[float, np.ndarray]:
    design_mean = design.mean(axis=0, keepdims=True)
    target_mean = float(target.mean())
    centered_design = design - design_mean
    centered_target = target - target_mean
    if linear_reg > 0.0:
        centered_design = np.vstack([centered_design, np.sqrt(linear_reg) * np.eye(design.shape[1])])
        centered_target = np.concatenate([centered_target, np.zeros(design.shape[1])])
    coef, _residual = nnls(centered_design, centered_target, maxiter=20 * design.shape[1])
    intercept = target_mean - float((design_mean @ coef).item())
    return intercept, np.asarray(coef, dtype=float)


def fit_head(
    dataset: pooled.Dataset,
    indices: np.ndarray,
    config: ModelConfig,
    rho: np.ndarray,
    tau: np.ndarray,
    gamma: float,
) -> FittedModel:
    weights = dataset.weights[indices]
    signal, penalty = nonlinear_features(
        weights,
        dataset.c0,
        dataset.c1,
        rho,
        tau,
        gamma,
        config.damage_q,
    )
    alpha0, alpha1 = phase_fractions(dataset)
    design = np.column_stack(
        [
            -signal,
            penalty,
            selected_geometry(weights, config.geometry_kind),
            aggregate_hhi(weights, alpha0, alpha1),
        ]
    )
    intercept, coef = fit_nonnegative_head(
        design,
        dataset.y[indices],
        geometry.dataset_linear_reg(dataset),
    )
    m = dataset.m
    return FittedModel(
        config=config,
        rho=np.asarray(rho, dtype=float),
        tau=np.asarray(tau, dtype=float),
        gamma=float(gamma),
        intercept=intercept,
        benefit_coef=coef[:m],
        penalty_coef=coef[m : 2 * m],
        geometry_coef=float(coef[-2]),
        aggregate_hhi_coef=float(coef[-1]),
    )


def predict(model: FittedModel, dataset: pooled.Dataset, indices: np.ndarray) -> np.ndarray:
    weights = dataset.weights[indices]
    signal, penalty = nonlinear_features(
        weights,
        dataset.c0,
        dataset.c1,
        model.rho,
        model.tau,
        model.gamma,
        model.config.damage_q,
    )
    alpha0, alpha1 = phase_fractions(dataset)
    return np.asarray(
        model.intercept
        - signal @ model.benefit_coef
        + penalty @ model.penalty_coef
        + model.geometry_coef * selected_geometry(weights, model.config.geometry_kind)
        + model.aggregate_hhi_coef * aggregate_hhi(weights, alpha0, alpha1),
        dtype=float,
    )


def encode_start(params: dict[str, float | np.ndarray], config: ModelConfig) -> np.ndarray:
    rho = np.asarray(params["rho"], dtype=float)
    tau = np.asarray(params["tau"], dtype=float)
    encoded_rho = np.asarray([np.mean(np.log(rho))]) if config.shared_rho else np.log(rho)
    return np.concatenate(
        [
            encoded_rho,
            tau,
            np.asarray([np.log(float(params["gamma"]))]),
        ]
    )


def decode_params(
    theta: np.ndarray,
    config: ModelConfig,
    num_domains: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    rho_count = 1 if config.shared_rho else num_domains
    encoded_rho = theta[:rho_count]
    rho = np.exp(np.clip(encoded_rho, np.log(RHO_MIN), np.log(RHO_MAX)))
    if config.shared_rho:
        rho = np.full(num_domains, float(rho[0]))
    tau_start = rho_count
    tau = np.clip(theta[tau_start : tau_start + num_domains], TAU_MIN, TAU_MAX)
    gamma = float(
        np.exp(
            np.clip(
                theta[tau_start + num_domains],
                np.log(GAMMA_MIN),
                np.log(GAMMA_MAX),
            )
        )
    )
    return np.asarray(rho, dtype=float), np.asarray(tau, dtype=float), gamma


def parameter_bounds(config: ModelConfig, num_domains: int) -> list[tuple[float, float]]:
    rho_count = 1 if config.shared_rho else num_domains
    return (
        [(np.log(RHO_MIN), np.log(RHO_MAX))] * rho_count
        + [(TAU_MIN, TAU_MAX)] * num_domains
        + [(np.log(GAMMA_MIN), np.log(GAMMA_MAX))]
    )


def start_bank(dataset: pooled.Dataset, indices: np.ndarray, config: ModelConfig) -> tuple[np.ndarray, ...]:
    packet = geometry.packet(dataset, indices)
    variant = dsp.VARIANTS["effective_exposure"]
    starts = [
        encode_start(dsp.unpack_theta(start, variant, packet.m), config) for start in dsp.start_bank(packet, variant)
    ]
    unique: list[np.ndarray] = []
    for start in starts:
        if not any(np.allclose(start, previous) for previous in unique):
            unique.append(start)
    return tuple(unique)


def profile_objective(
    theta: np.ndarray,
    dataset: pooled.Dataset,
    indices: np.ndarray,
    config: ModelConfig,
) -> float:
    rho, tau, gamma = decode_params(theta, config, dataset.m)
    model = fit_head(dataset, indices, config, rho, tau, gamma)
    prediction = predict(model, dataset, indices)
    residual = prediction - dataset.y[indices]
    tail_count = max(5, int(np.ceil(LOWER_TAIL_FRAC * len(indices))))
    tail = np.argsort(prediction)[:tail_count]
    optimism = float(np.mean(np.maximum(-residual[tail], 0.0)))
    return float(np.sqrt(np.mean(residual**2))) + 0.5 * optimism


def fit_model(
    dataset: pooled.Dataset,
    indices: np.ndarray,
    config: ModelConfig,
    maxiter: int,
    coarse_top_k: int,
) -> FittedModel:
    starts = start_bank(dataset, indices, config)
    scored = sorted((profile_objective(start, dataset, indices, config), start) for start in starts)
    best_value, best_theta = scored[0]
    if maxiter > 0:
        bounds = parameter_bounds(config, dataset.m)
        for _coarse_value, start in scored[:coarse_top_k]:
            result = minimize(
                lambda theta: profile_objective(
                    np.asarray(theta, dtype=float),
                    dataset,
                    indices,
                    config,
                ),
                start,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": maxiter, "ftol": 1e-7, "maxls": 20},
            )
            if float(result.fun) < best_value:
                best_value = float(result.fun)
                best_theta = np.asarray(result.x, dtype=float)
    rho, tau, gamma = decode_params(best_theta, config, dataset.m)
    return fit_head(dataset, indices, config, rho, tau, gamma)


def dataset_folds(dataset: pooled.Dataset, seed: int, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if "phase_correspondence_key" in dataset.frame:
        return joint.grouped_folds(dataset.frame, seed, n_splits)
    return pooled.dataset_folds(dataset, seed, n_splits)


def parameter_count(dataset: pooled.Dataset, config: ModelConfig) -> int:
    nonlinear = (1 if config.shared_rho else dataset.m) + dataset.m + 1
    linear = 2 * dataset.m + 3
    return nonlinear + linear


def benchmark_dataset(
    dataset: pooled.Dataset,
    model_configs: tuple[ModelConfig, ...],
    seeds: list[int],
    n_splits: int,
    maxiter: int,
    coarse_top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, float | int | str]] = []
    parameter_rows: list[dict[str, float | int | str]] = []
    pair_metric_rows: list[dict[str, float | int | str]] = []
    prediction_frames: list[pd.DataFrame] = []
    for seed in seeds:
        folds = dataset_folds(dataset, seed, n_splits)
        predictions = {config.name: np.zeros(dataset.n, dtype=float) for config in model_configs}
        for fold_id, (train_indices, test_indices) in enumerate(folds):
            print(f"{dataset.name}: seed={seed} fold={fold_id + 1}/{n_splits}", flush=True)
            for config in model_configs:
                model = fit_model(dataset, train_indices, config, maxiter, coarse_top_k)
                predictions[config.name][test_indices] = predict(model, dataset, test_indices)
                parameter_rows.append(
                    {
                        "dataset": dataset.name,
                        "model": config.name,
                        "seed": seed,
                        "fold": fold_id,
                        "rho_geometric_mean": float(np.exp(np.mean(np.log(model.rho)))),
                        "rho_log_sd": float(np.std(np.log(model.rho))),
                        "tau_mean": float(np.mean(model.tau)),
                        "tau_sd": float(np.std(model.tau)),
                        "gamma": model.gamma,
                        "damage_gamma": model.gamma**config.damage_q,
                        "theta_geometry": model.geometry_coef,
                        "theta_hhi_aggregate": model.aggregate_hhi_coef,
                    }
                )
        for config in model_configs:
            prediction = predictions[config.name]
            row = asdict(pooled.metrics(dataset, config.name, seed, prediction, folds))
            row["nominal_param_count"] = parameter_count(dataset, config)
            row["role"] = config.role
            metric_rows.append(row)
            prediction_frame = dataset.frame[
                [
                    column
                    for column in (
                        "run_name",
                        "candidate_name",
                        "policy_family",
                        "split",
                        "packet_panel",
                        "phase_correspondence_key",
                    )
                    if column in dataset.frame
                ]
            ].copy()
            prediction_frame["dataset"] = dataset.name
            prediction_frame["model"] = config.name
            prediction_frame["seed"] = seed
            prediction_frame["observed"] = dataset.y
            prediction_frame["predicted"] = prediction
            prediction_frames.append(prediction_frame)
            if dataset.name in NOISE_SD:
                pairs = ordering.pair_frame(dataset, prediction, config.name)
                pair_row = ordering.pair_metrics(pairs, NOISE_SD[dataset.name])
                pair_row["seed"] = seed
                pair_metric_rows.append(pair_row)
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(parameter_rows),
        pd.DataFrame(pair_metric_rows),
        pd.concat(prediction_frames, ignore_index=True),
    )


def summarize(raw: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    numeric_columns = [
        column for column in raw.columns if column not in {*keys, "seed"} and pd.api.types.is_numeric_dtype(raw[column])
    ]
    rows = []
    for group_key, frame in raw.groupby(keys, sort=True):
        values = group_key if isinstance(group_key, tuple) else (group_key,)
        row = dict(zip(keys, values, strict=True))
        for column in numeric_columns:
            row[f"{column}_mean"] = float(frame[column].mean())
            row[f"{column}_sd"] = float(frame[column].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def external_evaluation(
    fit_dataset: pooled.Dataset,
    external: pooled.Dataset,
    model_configs: tuple[ModelConfig, ...],
    maxiter: int,
    coarse_top_k: int,
) -> pd.DataFrame:
    rows = []
    for config in model_configs:
        model = fit_model(fit_dataset, np.arange(fit_dataset.n), config, maxiter, coarse_top_k)
        row = joint.external_metrics(
            config.name,
            external.y,
            predict(model, external, np.arange(external.n)),
        )
        row["dataset"] = fit_dataset.name
        row["external_rows"] = external.n
        rows.append(row)
    return pd.DataFrame(rows)


def geometry_audit(datasets: list[pooled.Dataset]) -> pd.DataFrame:
    rows = []
    for dataset in datasets:
        if "policy_family" in dataset.frame:
            indices = np.flatnonzero(dataset.frame["policy_family"].eq("two_phase").to_numpy())
        else:
            indices = np.arange(dataset.n)
        weights = dataset.weights[indices]
        features = geometry_features(weights)
        for left, right in (("tv", "hellinger"), ("tv", "js"), ("hellinger", "js")):
            rows.append(
                {
                    "dataset": dataset.name,
                    "left": left,
                    "right": right,
                    "spearman": float(spearmanr(features[left], features[right]).statistic),
                }
            )
        w0 = weights[:, 0, :]
        w1 = weights[:, 1, :]
        changed = (w0 <= 1e-12) != (w1 <= 1e-12)
        rows.append(
            {
                "dataset": dataset.name,
                "left": "support_churn",
                "right": "row_fraction",
                "spearman": float(np.mean(np.any(changed, axis=1))),
            }
        )
    return pd.DataFrame(rows)


def should_include_hellinger(audit: pd.DataFrame) -> bool:
    correlations = audit.loc[audit["left"].eq("tv") & audit["right"].isin(["hellinger", "js"])]
    if correlations.empty:
        raise ValueError("Geometry audit lacks TV correlations")
    best_by_dataset = correlations.groupby("dataset")["spearman"].min()
    return bool(np.any(np.abs(best_by_dataset.to_numpy()) < GEOMETRY_REDUNDANT_RHO))


def identification_audit(
    dataset: pooled.Dataset,
    config: ModelConfig,
    maxiter: int,
    coarse_top_k: int,
) -> pd.DataFrame:
    model = fit_model(dataset, np.arange(dataset.n), config, maxiter, coarse_top_k)
    weights = dataset.weights
    e0 = weights[:, 0, :] * dataset.c0[None, :]
    e1 = weights[:, 1, :] * dataset.c1[None, :]
    exposure = e0 + model.gamma * e1
    argument = np.log1p(exposure) - model.tau[None, :]
    active_fraction = np.mean(argument > 0.0, axis=0)
    phase_share = e1 / np.maximum(e0 + e1, SCALE_FLOOR)
    phase_share_sd = np.std(phase_share, axis=0)
    active = active_fraction >= PENALTY_ACTIVE_FRACTION
    heterogeneous = phase_share_sd >= PHASE_SHARE_SD_MIN
    return pd.DataFrame(
        {
            "dataset": dataset.name,
            "domain": dataset.domain_names,
            "penalty_active_fraction": active_fraction,
            "phase_share_sd": phase_share_sd,
            "penalty_coef": model.penalty_coef,
            "identified_for_q": active & heterogeneous & (model.penalty_coef > 1e-10),
        }
    )


def load_datasets(
    packet_path: Path,
    one_phase_source: Path,
) -> tuple[dict[str, pooled.Dataset], dict[str, pooled.Dataset]]:
    frame = pd.read_csv(packet_path)
    domains = pooled.load_300m_dataset("table9").domain_names
    frame = joint.attach_single_phase_weights(frame, one_phase_source, domains)
    datasets: dict[str, pooled.Dataset] = {}
    external: dict[str, pooled.Dataset] = {}
    for objective, target in joint.TARGET_COLUMNS.items():
        dataset = joint.dataset_from_frame(
            objective,
            frame.loc[frame["split"].eq("train") | frame["policy_family"].eq("single_phase")].copy(),
            target,
        )
        datasets[dataset.name] = dataset
        external[dataset.name] = joint.dataset_from_frame(
            objective,
            frame.loc[frame["split"].eq("heldout") & frame["policy_family"].eq("two_phase")].copy(),
            target,
        )
    production = pooled.load_production_dataset()
    datasets[production.name] = production
    return datasets, external


def write_report(
    metrics: pd.DataFrame,
    pair_metrics: pd.DataFrame,
    parameters: pd.DataFrame,
    geometry_frame: pd.DataFrame,
    identification: pd.DataFrame,
    external: pd.DataFrame,
    include_hellinger: bool,
    output_dir: Path,
) -> None:
    metric_columns = [
        "dataset",
        "model",
        "role",
        "nominal_param_count_mean",
        "oof_rmse_mean",
        "oof_rmse_sd",
        "oof_spearman_mean",
        "oof_spearman_sd",
        "fold_mean_regret_at_1_mean",
    ]
    pair_columns = [
        "dataset",
        "model",
        "delta_rmse_mean",
        "delta_rmse_sd",
        "delta_spearman_mean",
        "delta_spearman_sd",
        "sign_accuracy_mean",
    ]
    parameter_columns = [
        "dataset",
        "model",
        "gamma_mean",
        "gamma_sd",
        "theta_geometry_mean",
        "theta_geometry_sd",
        "theta_hhi_aggregate_mean",
        "theta_hhi_aggregate_sd",
    ]
    identified = identification.groupby("dataset", as_index=False)["identified_for_q"].sum()
    lines = [
        "# Irreversible-damage two-phase DSP screen",
        "",
        "The primary test fixes the overexposure penalty to physical total exposure (`q=0`) while retaining "
        "effective exposure for useful learning. Every damage comparison is replicated under full and shared "
        "saturation rates.",
        "",
        f"The pre-fit geometry audit {'retained' if include_hellinger else 'rejected'} Hellinger as the single "
        "alternative to TV.",
        "",
        "## Grouped-CV metrics",
        "",
        metrics[metric_columns].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Matched ordering metrics",
        "",
        pair_metrics[pair_columns].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Parameter stability",
        "",
        parameters[parameter_columns].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Pre-fit geometry audit",
        "",
        geometry_frame.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Damage-identification support",
        "",
        identified.to_markdown(index=False),
        "",
        "## Untouched 300M intervention transfer",
        "",
        external.to_markdown(index=False, floatfmt=".6f"),
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=joint.PACKET)
    parser.add_argument("--one-phase-source", type=Path, default=joint.ONE_PHASE_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--datasets", default="300m_uncheatable,300m_table9,production_uncheatable")
    parser.add_argument("--models", default="")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--maxiter-300m", type=int, default=12)
    parser.add_argument("--maxiter-production", type=int, default=2)
    parser.add_argument("--coarse-top-k", type=int, default=1)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    datasets, external_by_name = load_datasets(args.packet, args.one_phase_source)
    selected_names = [part.strip() for part in args.datasets.split(",") if part.strip()]
    unknown_datasets = sorted(set(selected_names).difference(datasets))
    if unknown_datasets:
        raise ValueError(f"Unknown datasets: {unknown_datasets}")
    selected_datasets = [datasets[name] for name in selected_names]

    audit_datasets = [datasets["300m_uncheatable"], datasets["production_uncheatable"]]
    geometry_frame = geometry_audit(audit_datasets)
    include_hellinger = should_include_hellinger(geometry_frame)
    available_configs = configs(include_hellinger)
    config_by_name = {config.name: config for config in available_configs}
    selected_models = [part.strip() for part in args.models.split(",") if part.strip()]
    if selected_models:
        unknown_models = sorted(set(selected_models).difference(config_by_name))
        if unknown_models:
            raise ValueError(f"Unknown models: {unknown_models}")
        model_configs = tuple(config_by_name[name] for name in selected_models)
    else:
        model_configs = available_configs

    seeds = pooled.parse_int_list(args.seeds)
    metric_frames = []
    parameter_frames = []
    pair_metric_frames = []
    prediction_frames = []
    external_frames = []
    for dataset in selected_datasets:
        maxiter = args.maxiter_production if dataset.name == "production_uncheatable" else args.maxiter_300m
        metrics, parameters, pair_metrics, predictions = benchmark_dataset(
            dataset,
            model_configs,
            seeds,
            args.n_splits,
            maxiter,
            args.coarse_top_k,
        )
        metric_frames.append(metrics)
        parameter_frames.append(parameters)
        pair_metric_frames.append(pair_metrics)
        prediction_frames.append(predictions)
        if dataset.name in external_by_name:
            external_frames.append(
                external_evaluation(
                    dataset,
                    external_by_name[dataset.name],
                    model_configs,
                    maxiter,
                    args.coarse_top_k,
                )
            )

    raw_metrics = pd.concat(metric_frames, ignore_index=True)
    raw_parameters = pd.concat(parameter_frames, ignore_index=True)
    raw_pair_metrics = pd.concat(pair_metric_frames, ignore_index=True)
    raw_predictions = pd.concat(prediction_frames, ignore_index=True)
    external = pd.concat(external_frames, ignore_index=True)
    metric_summary = summarize(raw_metrics, ["dataset", "model", "role"])
    parameter_summary = summarize(raw_parameters, ["dataset", "model"])
    pair_summary = summarize(raw_pair_metrics, ["dataset", "model"])

    identification_frames = []
    for name in ("300m_uncheatable", "300m_table9"):
        dataset = datasets[name]
        control = next(config for config in model_configs if config.name == "full_effective_tv")
        identification_frames.append(
            identification_audit(
                dataset,
                control,
                args.maxiter_300m,
                args.coarse_top_k,
            )
        )
    identification = pd.concat(identification_frames, ignore_index=True)

    raw_metrics.to_csv(args.output_dir / "cv_metrics_by_seed.csv", index=False)
    metric_summary.to_csv(args.output_dir / "cv_summary.csv", index=False)
    raw_parameters.to_csv(args.output_dir / "fold_parameters.csv", index=False)
    parameter_summary.to_csv(args.output_dir / "parameter_summary.csv", index=False)
    raw_pair_metrics.to_csv(args.output_dir / "matched_pair_metrics_by_seed.csv", index=False)
    pair_summary.to_csv(args.output_dir / "matched_pair_summary.csv", index=False)
    raw_predictions.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    external.to_csv(args.output_dir / "external_two_phase_heldout_summary.csv", index=False)
    geometry_frame.to_csv(args.output_dir / "geometry_audit.csv", index=False)
    identification.to_csv(args.output_dir / "damage_identification_audit.csv", index=False)
    metadata = {
        "models": [asdict(config) for config in model_configs],
        "seeds": seeds,
        "n_splits": args.n_splits,
        "maxiter_300m": args.maxiter_300m,
        "maxiter_production": args.maxiter_production,
        "coarse_top_k": args.coarse_top_k,
        "geometry_redundant_rho": GEOMETRY_REDUNDANT_RHO,
        "include_hellinger": include_hellinger,
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    write_report(
        metric_summary,
        pair_summary,
        parameter_summary,
        geometry_frame,
        identification,
        external,
        include_hellinger,
        args.output_dir,
    )
    print(metric_summary.to_string(index=False))
    print(pair_summary.to_string(index=False))
    print(f"Wrote irreversible-damage DSP screen to {args.output_dir}")


if __name__ == "__main__":
    main()
