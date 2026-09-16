# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["cvxpy", "numpy", "pandas", "plotly", "scikit-learn", "scipy", "tabulate"]
# ///
"""Benchmark simple phase mechanisms with aggregate and ordering fit separately.

The aggregate DSP backbone is fit only on actual constant-mixture checkpoints.
Each phase mechanism is then fit only on matched two-phase-minus-constant
differences while the aggregate backbone remains frozen. This prevents phase
parameters from absorbing aggregate-response error.
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
from scipy.optimize import minimize, minimize_scalar, nnls
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

DEFAULT_OUTPUT_DIR = pooled.REFERENCE_OUTPUTS / "staged_mechanistic_phase_zoo_20260710"
LOWER_TAIL_FRAC = 0.15
RHO_MIN = 1e-4
RHO_MAX = 2.0
TAU_MIN = -2.0
TAU_MAX = 8.0
PHASE_PARAMETER_MAX = 100.0


class OrderKind(StrEnum):
    NULL = "null"
    TV_ONLY = "tv_only"
    EFFECTIVE_BENEFIT = "effective_benefit"
    CANONICAL_LATE_SHARE = "canonical_late_share"
    LATE_ROOM_BONUS = "late_room_bonus"
    DEMOTION_RETENTION = "demotion_retention"
    INTERFERENCE_RETENTION = "interference_retention"
    PHASE_SEPARABLE = "phase_separable"


@dataclass(frozen=True)
class OrderConfig:
    name: str
    kind: OrderKind
    use_tv: bool
    eligible: bool


@dataclass(frozen=True)
class AggregateModel:
    rho: float
    tau: np.ndarray
    intercept: float
    benefit_coef: np.ndarray
    penalty_coef: np.ndarray
    hhi_coef: float


@dataclass(frozen=True)
class OrderModel:
    config: OrderConfig
    phase_parameter: float
    tv_coef: float
    phase_coef: np.ndarray


@dataclass(frozen=True)
class PairIndices:
    keys: np.ndarray
    single: np.ndarray
    two: np.ndarray


def configs() -> tuple[OrderConfig, ...]:
    return (
        OrderConfig("aggregate_null", OrderKind.NULL, False, True),
        OrderConfig("direct_tv", OrderKind.TV_ONLY, True, True),
        OrderConfig("effective_benefit", OrderKind.EFFECTIVE_BENEFIT, False, True),
        OrderConfig("effective_benefit_tv", OrderKind.EFFECTIVE_BENEFIT, True, True),
        OrderConfig("canonical_late_share", OrderKind.CANONICAL_LATE_SHARE, False, True),
        OrderConfig("canonical_late_share_tv", OrderKind.CANONICAL_LATE_SHARE, True, True),
        OrderConfig("late_room_bonus", OrderKind.LATE_ROOM_BONUS, False, True),
        OrderConfig("late_room_bonus_tv", OrderKind.LATE_ROOM_BONUS, True, True),
        OrderConfig("demotion_retention", OrderKind.DEMOTION_RETENTION, False, True),
        OrderConfig("demotion_retention_tv", OrderKind.DEMOTION_RETENTION, True, True),
        OrderConfig("interference_retention", OrderKind.INTERFERENCE_RETENTION, False, False),
        OrderConfig("interference_retention_tv", OrderKind.INTERFERENCE_RETENTION, True, False),
        OrderConfig("phase_separable", OrderKind.PHASE_SEPARABLE, False, False),
        OrderConfig("phase_separable_tv", OrderKind.PHASE_SEPARABLE, True, False),
    )


def phase_fractions(dataset: pooled.Dataset) -> tuple[float, float]:
    return geometry.phase_fractions(dataset)


def aggregate_weights(dataset: pooled.Dataset, indices: np.ndarray) -> np.ndarray:
    alpha0, alpha1 = phase_fractions(dataset)
    weights = dataset.weights[indices]
    return alpha0 * weights[:, 0, :] + alpha1 * weights[:, 1, :]


def phase_exposures(dataset: pooled.Dataset, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weights = dataset.weights[indices]
    return weights[:, 0, :] * dataset.c0[None, :], weights[:, 1, :] * dataset.c1[None, :]


def aggregate_features(
    dataset: pooled.Dataset,
    indices: np.ndarray,
    rho: float,
    tau: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    e0, e1 = phase_exposures(dataset, indices)
    exposure = e0 + e1
    signal = 1.0 - np.exp(-rho * exposure)
    penalty = dsp.softplus(np.log1p(exposure) - tau[None, :]) ** 2
    hhi = np.sum(aggregate_weights(dataset, indices) ** 2, axis=1)
    return signal, penalty, hhi


def fit_nonnegative_head(
    design: np.ndarray,
    target: np.ndarray,
    linear_reg: float,
) -> tuple[float, np.ndarray]:
    mean = design.mean(axis=0, keepdims=True)
    target_mean = float(target.mean())
    centered_design = design - mean
    centered_target = target - target_mean
    if linear_reg > 0.0:
        centered_design = np.vstack([centered_design, np.sqrt(linear_reg) * np.eye(design.shape[1])])
        centered_target = np.concatenate([centered_target, np.zeros(design.shape[1])])
    coef, _residual = nnls(centered_design, centered_target, maxiter=20 * design.shape[1])
    intercept = target_mean - float((mean @ coef).item())
    return intercept, np.asarray(coef, dtype=float)


def fit_aggregate_head(
    dataset: pooled.Dataset,
    indices: np.ndarray,
    rho: float,
    tau: np.ndarray,
    linear_reg: float,
) -> AggregateModel:
    signal, penalty, hhi = aggregate_features(dataset, indices, rho, tau)
    design = np.column_stack([-signal, penalty, hhi])
    intercept, coef = fit_nonnegative_head(design, dataset.y[indices], linear_reg)
    m = dataset.m
    return AggregateModel(
        rho=rho,
        tau=np.asarray(tau, dtype=float),
        intercept=intercept,
        benefit_coef=coef[:m],
        penalty_coef=coef[m : 2 * m],
        hhi_coef=float(coef[-1]),
    )


def aggregate_prediction(model: AggregateModel, dataset: pooled.Dataset, indices: np.ndarray) -> np.ndarray:
    signal, penalty, hhi = aggregate_features(dataset, indices, model.rho, model.tau)
    return np.asarray(
        model.intercept - signal @ model.benefit_coef + penalty @ model.penalty_coef + model.hhi_coef * hhi,
        dtype=float,
    )


def aggregate_profile_objective(
    theta: np.ndarray,
    dataset: pooled.Dataset,
    indices: np.ndarray,
    linear_reg: float,
) -> float:
    rho = float(np.exp(np.clip(theta[0], np.log(RHO_MIN), np.log(RHO_MAX))))
    tau = np.clip(theta[1:], TAU_MIN, TAU_MAX)
    model = fit_aggregate_head(dataset, indices, rho, tau, linear_reg)
    prediction = aggregate_prediction(model, dataset, indices)
    residual = prediction - dataset.y[indices]
    tail_count = max(5, int(np.ceil(LOWER_TAIL_FRAC * len(indices))))
    tail = np.argsort(prediction)[:tail_count]
    optimism = float(np.mean(np.maximum(-residual[tail], 0.0)))
    return float(np.sqrt(np.mean(residual**2))) + 0.5 * optimism


def aggregate_starts(dataset: pooled.Dataset, indices: np.ndarray) -> tuple[np.ndarray, ...]:
    e0, e1 = phase_exposures(dataset, indices)
    exposure = e0 + e1
    positive = np.where(exposure > 1e-8, exposure, np.nan)
    median = np.nanmedian(positive, axis=0)
    base_rho = float(
        np.clip(
            np.exp(np.nanmean(np.log(np.clip(1.0 / np.maximum(median, 1e-3), RHO_MIN, RHO_MAX)))),
            RHO_MIN,
            RHO_MAX,
        )
    )
    base_tau = np.log1p(np.nanpercentile(positive, 85, axis=0))
    base_tau = np.clip(np.where(np.isfinite(base_tau), base_tau, 3.0), TAU_MIN, TAU_MAX)
    return tuple(
        np.concatenate(
            [
                np.asarray([np.log(np.clip(base_rho * rho_scale, RHO_MIN, RHO_MAX))]),
                np.clip(base_tau + tau_shift, TAU_MIN, TAU_MAX),
            ]
        )
        for rho_scale, tau_shift in (
            (0.5, -1.0),
            (0.5, 0.0),
            (1.0, -1.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (2.0, 0.0),
            (2.0, 1.0),
        )
    )


def fit_aggregate_model(
    dataset: pooled.Dataset,
    indices: np.ndarray,
    maxiter: int,
    coarse_top_k: int,
) -> AggregateModel:
    linear_reg = geometry.dataset_linear_reg(dataset)
    scored = sorted(
        (
            aggregate_profile_objective(start, dataset, indices, linear_reg),
            start,
        )
        for start in aggregate_starts(dataset, indices)
    )
    best_value, best_theta = scored[0]
    if maxiter > 0:
        bounds = [(np.log(RHO_MIN), np.log(RHO_MAX))] + [(TAU_MIN, TAU_MAX)] * dataset.m
        for _value, start in scored[:coarse_top_k]:
            result = minimize(
                lambda theta: aggregate_profile_objective(np.asarray(theta, dtype=float), dataset, indices, linear_reg),
                start,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": maxiter, "ftol": 1e-7, "maxls": 20},
            )
            if float(result.fun) < best_value:
                best_value = float(result.fun)
                best_theta = np.asarray(result.x, dtype=float)
    rho = float(np.exp(best_theta[0]))
    tau = np.asarray(best_theta[1:], dtype=float)
    return fit_aggregate_head(dataset, indices, rho, tau, linear_reg)


def pair_indices(dataset: pooled.Dataset) -> PairIndices:
    frame = dataset.frame.reset_index(drop=True)
    single = frame.loc[frame["policy_family"].eq("single_phase")].copy()
    two = frame.loc[frame["split"].eq("train") & frame["packet_panel"].eq("augmented_fit_panel")].copy()
    single["row_index"] = single.index
    two["row_index"] = two.index
    single = single.set_index("phase_correspondence_key")
    two = two.set_index("phase_correspondence_key")
    keys = single.index.intersection(two.index)
    return PairIndices(
        keys=keys.to_numpy(dtype=str),
        single=single.loc[keys, "row_index"].to_numpy(dtype=int),
        two=two.loc[keys, "row_index"].to_numpy(dtype=int),
    )


def selected_pairs(pairs: PairIndices, allowed: np.ndarray) -> PairIndices:
    allowed_set = set(int(value) for value in allowed)
    mask = np.fromiter(
        (
            int(single) in allowed_set and int(two) in allowed_set
            for single, two in zip(pairs.single, pairs.two, strict=True)
        ),
        dtype=bool,
    )
    return PairIndices(keys=pairs.keys[mask], single=pairs.single[mask], two=pairs.two[mask])


def tied_quantities(
    dataset: pooled.Dataset,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    aggregate = aggregate_weights(dataset, indices)
    tied_e0 = aggregate * dataset.c0[None, :]
    tied_e1 = aggregate * dataset.c1[None, :]
    return aggregate, tied_e0, tied_e1


def tv_feature(dataset: pooled.Dataset, indices: np.ndarray) -> np.ndarray:
    weights = dataset.weights[indices]
    return 0.5 * np.abs(weights[:, 0, :] - weights[:, 1, :]).sum(axis=1)


def mechanism_delta(
    config: OrderConfig,
    phase_parameter: float,
    aggregate_model: AggregateModel,
    dataset: pooled.Dataset,
    indices: np.ndarray,
) -> np.ndarray:
    if config.kind in (OrderKind.NULL, OrderKind.TV_ONLY, OrderKind.PHASE_SEPARABLE):
        return np.zeros(len(indices), dtype=float)
    weights = dataset.weights[indices]
    e0, e1 = phase_exposures(dataset, indices)
    aggregate, tied_e0, tied_e1 = tied_quantities(dataset, indices)
    rho = aggregate_model.rho
    if config.kind == OrderKind.EFFECTIVE_BENEFIT:
        gamma = phase_parameter
        signal = 1.0 - np.exp(-rho * (e0 + gamma * e1))
        tied_signal = 1.0 - np.exp(-rho * (tied_e0 + gamma * tied_e1))
    elif config.kind == OrderKind.CANONICAL_LATE_SHARE:
        exposure = e0 + e1
        tied_exposure = tied_e0 + tied_e1
        base = 1.0 - np.exp(-rho * exposure)
        tied_base = 1.0 - np.exp(-rho * tied_exposure)
        late_share = e1 / np.maximum(exposure, 1e-12)
        tied_late_share = tied_e1 / np.maximum(tied_exposure, 1e-12)
        signal = (1.0 + phase_parameter * late_share) * base
        tied_signal = (1.0 + phase_parameter * tied_late_share) * tied_base
    elif config.kind == OrderKind.LATE_ROOM_BONUS:
        exposure = e0 + e1
        tied_exposure = tied_e0 + tied_e1
        base = 1.0 - np.exp(-rho * exposure)
        tied_base = 1.0 - np.exp(-rho * tied_exposure)
        late_refresh = 1.0 - np.exp(-rho * e1)
        tied_late_refresh = 1.0 - np.exp(-rho * tied_e1)
        signal = base + phase_parameter * (1.0 - base) * late_refresh
        tied_signal = tied_base + phase_parameter * (1.0 - tied_base) * tied_late_refresh
    elif config.kind == OrderKind.DEMOTION_RETENTION:
        retention = np.exp(-phase_parameter * np.maximum(aggregate - weights[:, 1, :], 0.0))
        phase0_state = 1.0 - np.exp(-rho * e0)
        signal = 1.0 - (1.0 - retention * phase0_state) * np.exp(-rho * e1)
        tied_signal = 1.0 - np.exp(-rho * (tied_e0 + tied_e1))
    elif config.kind == OrderKind.INTERFERENCE_RETENTION:
        retention = np.exp(-phase_parameter * (1.0 - weights[:, 1, :]))
        tied_retention = np.exp(-phase_parameter * (1.0 - aggregate))
        phase0_state = 1.0 - np.exp(-rho * e0)
        tied_phase0_state = 1.0 - np.exp(-rho * tied_e0)
        signal = 1.0 - (1.0 - retention * phase0_state) * np.exp(-rho * e1)
        tied_signal = 1.0 - (1.0 - tied_retention * tied_phase0_state) * np.exp(-rho * tied_e1)
    else:
        raise ValueError(f"Unsupported mechanism {config.kind}")
    return np.asarray(-(signal - tied_signal) @ aggregate_model.benefit_coef, dtype=float)


def phase_separable_design(
    aggregate_model: AggregateModel,
    dataset: pooled.Dataset,
    indices: np.ndarray,
    use_tv: bool,
) -> np.ndarray:
    e0, e1 = phase_exposures(dataset, indices)
    _aggregate, tied_e0, tied_e1 = tied_quantities(dataset, indices)
    rho = aggregate_model.rho
    phase0 = 1.0 - np.exp(-rho * e0)
    phase1 = 1.0 - np.exp(-rho * e1)
    tied0 = 1.0 - np.exp(-rho * tied_e0)
    tied1 = 1.0 - np.exp(-rho * tied_e1)
    pieces = [-(phase0 - tied0), -(phase1 - tied1)]
    if use_tv:
        pieces.append(tv_feature(dataset, indices)[:, None])
    return np.hstack(pieces)


def fit_tv(residual: np.ndarray, tv: np.ndarray, linear_reg: float) -> float:
    denominator = float(tv @ tv + linear_reg)
    return max(0.0, float(tv @ residual) / max(denominator, 1e-12))


def fit_order_model(
    config: OrderConfig,
    aggregate_model: AggregateModel,
    dataset: pooled.Dataset,
    pairs: PairIndices,
) -> OrderModel:
    observed = dataset.y[pairs.two] - dataset.y[pairs.single]
    linear_reg = geometry.dataset_linear_reg(dataset)
    if config.kind == OrderKind.PHASE_SEPARABLE:
        design = phase_separable_design(aggregate_model, dataset, pairs.two, config.use_tv)
        augmented_design = np.vstack([design, np.sqrt(linear_reg) * np.eye(design.shape[1])])
        augmented_target = np.concatenate([observed, np.zeros(design.shape[1])])
        coef, _residual = nnls(augmented_design, augmented_target, maxiter=20 * design.shape[1])
        tv_coef = float(coef[-1]) if config.use_tv else 0.0
        return OrderModel(config, float("nan"), tv_coef, np.asarray(coef, dtype=float))

    tv = tv_feature(dataset, pairs.two)

    def score(transformed: float) -> tuple[float, float, float]:
        if config.kind in (OrderKind.NULL, OrderKind.TV_ONLY):
            phase_parameter = 0.0
        elif config.kind == OrderKind.EFFECTIVE_BENEFIT:
            phase_parameter = float(np.exp(transformed))
        else:
            phase_parameter = float(np.expm1(transformed))
        base = mechanism_delta(config, phase_parameter, aggregate_model, dataset, pairs.two)
        tv_coef = fit_tv(observed - base, tv, linear_reg) if config.use_tv else 0.0
        prediction = base + tv_coef * tv
        return float(np.sqrt(np.mean((prediction - observed) ** 2))), phase_parameter, tv_coef

    if config.kind in (OrderKind.NULL, OrderKind.TV_ONLY):
        _value, phase_parameter, tv_coef = score(0.0)
    else:
        if config.kind == OrderKind.EFFECTIVE_BENEFIT:
            bounds = (np.log(1e-3), np.log(PHASE_PARAMETER_MAX))
        elif config.kind == OrderKind.LATE_ROOM_BONUS:
            bounds = (0.0, np.log(2.0))
        else:
            bounds = (0.0, np.log1p(PHASE_PARAMETER_MAX))
        result = minimize_scalar(lambda value: score(float(value))[0], bounds=bounds, method="bounded")
        _value, phase_parameter, tv_coef = score(float(result.x))
    return OrderModel(config, phase_parameter, tv_coef, np.asarray([], dtype=float))


def order_prediction(
    model: OrderModel,
    aggregate_model: AggregateModel,
    dataset: pooled.Dataset,
    indices: np.ndarray,
) -> np.ndarray:
    if model.config.kind == OrderKind.PHASE_SEPARABLE:
        design = phase_separable_design(aggregate_model, dataset, indices, model.config.use_tv)
        prediction = design @ model.phase_coef
    else:
        prediction = mechanism_delta(
            model.config,
            model.phase_parameter,
            aggregate_model,
            dataset,
            indices,
        )
        if model.config.use_tv:
            prediction = prediction + model.tv_coef * tv_feature(dataset, indices)
    single = dataset.frame.iloc[indices]["policy_family"].eq("single_phase").to_numpy()
    prediction = np.asarray(prediction, dtype=float)
    prediction[single] = 0.0
    return prediction


def nominal_parameter_count(dataset: pooled.Dataset, config: OrderConfig) -> int:
    aggregate_count = 3 * dataset.m + 3
    if config.kind in (OrderKind.NULL,):
        order_count = 0
    elif config.kind == OrderKind.TV_ONLY:
        order_count = 1
    elif config.kind == OrderKind.PHASE_SEPARABLE:
        order_count = 2 * dataset.m + int(config.use_tv)
    else:
        order_count = 1 + int(config.use_tv)
    return aggregate_count + order_count


def fit_fold(
    dataset: pooled.Dataset,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    pairs: PairIndices,
    model_configs: tuple[OrderConfig, ...],
    maxiter: int,
    coarse_top_k: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, float | int | str]]]:
    train_single = train_indices[dataset.frame.iloc[train_indices]["policy_family"].eq("single_phase").to_numpy()]
    train_pairs = selected_pairs(pairs, train_indices)
    if len(train_single) == 0 or len(train_pairs.keys) == 0:
        raise ValueError("Fold lacks single-phase rows or matched training pairs")
    aggregate_model = fit_aggregate_model(dataset, train_single, maxiter, coarse_top_k)
    aggregate_test = aggregate_prediction(aggregate_model, dataset, test_indices)
    predictions = {}
    parameters = []
    for config in model_configs:
        order_model = fit_order_model(config, aggregate_model, dataset, train_pairs)
        predictions[config.name] = aggregate_test + order_prediction(order_model, aggregate_model, dataset, test_indices)
        parameters.append(
            {
                "model": config.name,
                "rho": aggregate_model.rho,
                "tau_mean": float(np.mean(aggregate_model.tau)),
                "phase_parameter": order_model.phase_parameter,
                "theta_tv": order_model.tv_coef,
                "active_phase_coef": int(np.sum(order_model.phase_coef > 1e-10)),
            }
        )
    return predictions, parameters


def benchmark_dataset(
    dataset: pooled.Dataset,
    model_configs: tuple[OrderConfig, ...],
    seeds: list[int],
    n_splits: int,
    maxiter: int,
    coarse_top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pairs = pair_indices(dataset)
    proportional_mask = (
        dataset.frame["phase_correspondence_key"].eq("baseline_proportional") & dataset.frame["split"].eq("train")
    ).to_numpy()
    proportional = dataset.y[proportional_mask]
    if len(proportional) < 2:
        raise ValueError(f"{dataset.name} lacks proportional repeats for the noise estimate")
    noise_sd = float(np.std(proportional, ddof=1))
    metric_rows = []
    pair_metric_rows = []
    pair_frames = []
    parameter_rows = []
    subset_rows = []
    for seed in seeds:
        folds = joint.grouped_folds(dataset.frame, seed, n_splits)
        oof = {config.name: np.zeros(dataset.n, dtype=float) for config in model_configs}
        for fold_id, (train_indices, test_indices) in enumerate(folds):
            print(f"{dataset.name}: seed={seed} fold={fold_id + 1}/{n_splits}", flush=True)
            fold_predictions, fold_parameters = fit_fold(
                dataset,
                train_indices,
                test_indices,
                pairs,
                model_configs,
                maxiter,
                coarse_top_k,
            )
            for config in model_configs:
                oof[config.name][test_indices] = fold_predictions[config.name]
            for row in fold_parameters:
                parameter_rows.append({"dataset": dataset.name, "seed": seed, "fold": fold_id, **row})
        for config in model_configs:
            row = asdict(pooled.metrics(dataset, config.name, seed, oof[config.name], folds))
            row["nominal_param_count"] = nominal_parameter_count(dataset, config)
            row["eligible"] = config.eligible
            metric_rows.append(row)
            for policy_family in ("single_phase", "two_phase"):
                if policy_family == "single_phase":
                    subset = np.flatnonzero(dataset.frame["policy_family"].eq("single_phase").to_numpy())
                else:
                    subset = np.flatnonzero(dataset.frame["policy_family"].eq("two_phase").to_numpy())
                residual = oof[config.name][subset] - dataset.y[subset]
                subset_rows.append(
                    {
                        "dataset": dataset.name,
                        "model": config.name,
                        "seed": seed,
                        "policy_family": policy_family,
                        "n_rows": len(subset),
                        "rmse": float(np.sqrt(np.mean(residual**2))),
                        "spearman": float(spearmanr(dataset.y[subset], oof[config.name][subset]).statistic),
                        "mean_residual": float(np.mean(residual)),
                    }
                )
            pair_frame = ordering.pair_frame(dataset, oof[config.name], config.name)
            pair_frame["seed"] = seed
            pair_frames.append(pair_frame)
            pair_metric = ordering.pair_metrics(pair_frame, noise_sd)
            pair_metric["seed"] = seed
            pair_metric["eligible"] = config.eligible
            pair_metric_rows.append(pair_metric)
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(pair_metric_rows),
        pd.concat(pair_frames, ignore_index=True),
        pd.DataFrame(parameter_rows),
        pd.DataFrame(subset_rows),
    )


def summarize(raw: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    numeric = [
        column for column in raw.columns if column not in {*keys, "seed"} and pd.api.types.is_numeric_dtype(raw[column])
    ]
    rows = []
    for group_key, frame in raw.groupby(keys, sort=True):
        values = group_key if isinstance(group_key, tuple) else (group_key,)
        row = dict(zip(keys, values, strict=True))
        for column in numeric:
            row[f"{column}_mean"] = float(frame[column].mean())
            row[f"{column}_sd"] = float(frame[column].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(
    metrics: pd.DataFrame,
    pair_metrics: pd.DataFrame,
    parameters: pd.DataFrame,
    subsets: pd.DataFrame,
    output_dir: Path,
) -> None:
    metric_columns = [
        "dataset",
        "model",
        "eligible",
        "nominal_param_count_mean",
        "oof_rmse_mean",
        "oof_spearman_mean",
        "fold_mean_regret_at_1_mean",
        "lower_tail_optimism_mean",
    ]
    pair_columns = [
        "dataset",
        "model",
        "delta_rmse_mean",
        "delta_spearman_mean",
        "sign_accuracy_mean",
        "reliable_sign_accuracy_mean",
    ]
    lines = [
        "# Staged mechanistic phase-model zoo",
        "",
        "The aggregate shared-rho DSP backbone is fit only on actual constant-mixture checkpoints. "
        "Phase mechanisms are fit only on matched two-phase-minus-constant differences.",
        "",
        "## Global grouped-CV metrics",
        "",
        metrics[metric_columns].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Matched ordering metrics",
        "",
        pair_metrics[pair_columns].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Policy-subset metrics",
        "",
        subsets.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Parameter stability",
        "",
        parameters.to_markdown(index=False, floatfmt=".6f"),
        "",
        "High-capacity diagnostics and robustness variants are marked ineligible before results are inspected.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=joint.PACKET)
    parser.add_argument("--one-phase-source", type=Path, default=joint.ONE_PHASE_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--datasets", default="uncheatable,table9")
    parser.add_argument("--models", default=",".join(config.name for config in configs()))
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--maxiter", type=int, default=12)
    parser.add_argument("--coarse-top-k", type=int, default=1)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config_by_name = {config.name: config for config in configs()}
    selected_models = [part.strip() for part in args.models.split(",") if part.strip()]
    unknown_models = sorted(set(selected_models).difference(config_by_name))
    if unknown_models:
        raise ValueError(f"Unknown models: {unknown_models}")
    model_configs = tuple(config_by_name[name] for name in selected_models)

    frame = pd.read_csv(args.packet)
    domains = pooled.load_300m_dataset("table9").domain_names
    frame = joint.attach_single_phase_weights(frame, args.one_phase_source, domains)
    selected_datasets = [part.strip() for part in args.datasets.split(",") if part.strip()]
    unknown_datasets = sorted(set(selected_datasets).difference(joint.TARGET_COLUMNS))
    if unknown_datasets:
        raise ValueError(f"Unknown datasets: {unknown_datasets}")

    metric_frames = []
    pair_metric_frames = []
    pair_frames = []
    parameter_frames = []
    subset_frames = []
    for objective in selected_datasets:
        dataset = joint.dataset_from_frame(
            objective,
            frame.loc[frame["split"].eq("train") | frame["policy_family"].eq("single_phase")].copy(),
            joint.TARGET_COLUMNS[objective],
        )
        metrics, pair_metrics, pairs, parameters, subsets = benchmark_dataset(
            dataset,
            model_configs,
            pooled.parse_int_list(args.seeds),
            args.n_splits,
            args.maxiter,
            args.coarse_top_k,
        )
        metric_frames.append(metrics)
        pair_metric_frames.append(pair_metrics)
        pair_frames.append(pairs)
        parameter_frames.append(parameters)
        subset_frames.append(subsets)

    raw_metrics = pd.concat(metric_frames, ignore_index=True)
    raw_pair_metrics = pd.concat(pair_metric_frames, ignore_index=True)
    raw_pairs = pd.concat(pair_frames, ignore_index=True)
    raw_parameters = pd.concat(parameter_frames, ignore_index=True)
    raw_subsets = pd.concat(subset_frames, ignore_index=True)
    metric_summary = summarize(raw_metrics, ["dataset", "model", "eligible"])
    pair_summary = summarize(raw_pair_metrics, ["dataset", "model", "eligible"])
    parameter_summary = summarize(raw_parameters, ["dataset", "model"])
    subset_summary = summarize(raw_subsets, ["dataset", "model", "policy_family"])

    raw_metrics.to_csv(args.output_dir / "cv_metrics_by_seed.csv", index=False)
    raw_pair_metrics.to_csv(args.output_dir / "matched_pair_metrics_by_seed.csv", index=False)
    raw_pairs.to_csv(args.output_dir / "matched_pair_predictions.csv", index=False)
    raw_parameters.to_csv(args.output_dir / "fold_parameters.csv", index=False)
    raw_subsets.to_csv(args.output_dir / "policy_subset_metrics_by_seed.csv", index=False)
    metric_summary.to_csv(args.output_dir / "cv_summary.csv", index=False)
    pair_summary.to_csv(args.output_dir / "matched_pair_summary.csv", index=False)
    parameter_summary.to_csv(args.output_dir / "parameter_summary.csv", index=False)
    subset_summary.to_csv(args.output_dir / "policy_subset_summary.csv", index=False)
    metadata = {
        "models": [asdict(config) for config in model_configs],
        "seeds": pooled.parse_int_list(args.seeds),
        "n_splits": args.n_splits,
        "maxiter": args.maxiter,
        "coarse_top_k": args.coarse_top_k,
    }
    (args.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    write_report(metric_summary, pair_summary, parameter_summary, subset_summary, args.output_dir)
    print(metric_summary.to_string(index=False))
    print(pair_summary.to_string(index=False))
    print(f"Wrote staged phase-model zoo to {args.output_dir}")


if __name__ == "__main__":
    main()
