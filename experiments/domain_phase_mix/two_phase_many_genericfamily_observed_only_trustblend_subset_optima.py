# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Observed-only trustblend GRP subset-fit deployments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.exploratory.two_phase_many.dataset_metadata import (
    load_two_phase_many_candidate_summary_spec,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_followup import (
    GenericFamilyPacket,
    GenericFamilySignalTransform,
    GenericFamilyRetainedTotalSurrogate,
    family_shares,
    load_generic_family_packet,
    optimize_generic_family_convex_hull,
    optimize_generic_family_model,
)
from experiments.domain_phase_mix.static_batch_selection import retrospective_generic_selection
from experiments.domain_phase_mix.two_phase_many_genericfamily_retuned_subset_optima import (
    CSV_PATH,
    GENERICFAMILY_RETUNED_SUBSET_OPTIMA_ALL_SUBSET_SIZES,
    OBJECTIVE_METRIC,
    _mean_phase_tv_distance,
    _phase_weights_from_array,
    _subset_packet,
    _top_domains,
)

GENERICFAMILY_OBSERVED_ONLY_TRUSTBLEND_SUBSET_OPTIMA_SOURCE_EXPERIMENT = (
    "pinlin_calvin_xu/data_mixture/"
    "ngd3dm2_genericfamily_observed_only_trustblend_top8actual_cap_subset_optima_rep_uncheatable_bpb"
)
GENERICFAMILY_OBSERVED_ONLY_TRUSTBLEND_SUBSET_OPTIMA_BASE_RUN_ID = 400
GENERICFAMILY_OBSERVED_ONLY_TRUSTBLEND_SUBSET_OPTIMA_REPRESENTATIVE_SUBSET_SIZES = (
    20,
    40,
    60,
    80,
    100,
    140,
    180,
    220,
)
GENERICFAMILY_OBSERVED_ONLY_TRUSTBLEND_SUBSET_OPTIMA_POLICY = (
    "feature_bayes_linear_observed_only_trustblend_top8actual_cap"
)
GENERICFAMILY_OBSERVED_ONLY_TRUSTBLEND_SUBSET_OPTIMA_OBJECTIVE = (
    "cv_rmse+0.05*cv_foldmean_regret+0.5*lower_tail_optimism"
)
DEFAULT_TUNING_METHOD = "Powell"
TRUSTBLEND_TOPK_ACTUAL = 8
TRUSTBLEND_LINE_GRID = 81
OBSERVED_ONLY_CV_WEIGHT = 1.0
OBSERVED_ONLY_FOLDMEAN_WEIGHT = 0.05
OBSERVED_ONLY_TAIL_WEIGHT = 0.5
OBSERVED_ONLY_LOWER_TAIL_FRAC = 0.15
OBSERVED_ONLY_START_PARAM_BANK: tuple[dict[str, float], ...] = (
    {"alpha": 8.0, "eta": 8.0, "lam": 0.05, "tau": 3.0, "reg": 1e-3, "beta": 0.70},
    {"alpha": 16.0, "eta": 8.0, "lam": 0.20, "tau": 2.7, "reg": 3e-4, "beta": 0.90},
    {"alpha": 4.0, "eta": 16.0, "lam": 0.02, "tau": 3.5, "reg": 3e-3, "beta": 0.50},
)
OBSERVED_ONLY_POWER_START_PARAM_BANK: tuple[dict[str, float], ...] = (
    {"alpha": 0.20, "eta": 8.0, "lam": 0.05, "tau": 3.0, "reg": 1e-3, "beta": 0.70},
    {"alpha": 0.35, "eta": 8.0, "lam": 0.20, "tau": 2.7, "reg": 3e-4, "beta": 0.90},
    {"alpha": 0.70, "eta": 16.0, "lam": 0.02, "tau": 3.5, "reg": 3e-3, "beta": 0.50},
)
OBSERVED_ONLY_LOG_SATIETY_ALPHA_LOG_BOUNDS = (-8.0, 8.0)
OBSERVED_ONLY_POWER_ALPHA_LOG_BOUNDS = (-4.0, 1.0)


@dataclass(frozen=True)
class GenericFamilyObservedOnlyTrustblendSubsetOptimumSummary:
    """Summary for one observed-only trustblend subset-fit deployment."""

    subset_size: int
    run_id: int
    run_name: str
    policy: str
    objective_metric: str
    tuning_method: str
    tuning_objective_name: str
    tuning_objective: float
    tuning_cv_rmse: float
    tuning_cv_r2: float
    tuning_cv_spearman: float
    tuning_cv_regret_at_1: float
    tuning_cv_foldmean_regret_at_1: float
    tuning_lower_tail_optimism: float
    tuned_params: dict[str, float]
    predicted_optimum_value: float
    deployment_delta: float
    deployment_realized_gain: float
    deployment_gain_budget: float
    deployment_raw_predicted_optimum_value: float
    deployment_hull_predicted_optimum_value: float
    fullswarm_chosen_run_name: str
    fullswarm_chosen_value: float
    fullswarm_regret_at_1: float
    observed_best_run_name: str
    observed_best_value: float
    gap_below_observed_best: float
    nearest_observed_run_name: str
    nearest_observed_value: float
    nearest_observed_tv_distance: float
    optimum_move_mean_phase_tv_vs_prev: float | None
    hull_anchor_count: int
    hull_anchor_summaries: list[dict[str, float | str]]
    phase0_max_weight: float
    phase1_max_weight: float
    phase0_support_below_1e4: int
    phase1_support_below_1e4: int
    phase0_top_domains: list[dict[str, float | str]]
    phase1_top_domains: list[dict[str, float | str]]
    optimizer_success: bool
    optimizer_message: str
    tuning_success: bool
    tuning_message: str
    family_shares: dict[str, float]
    phase_weights: dict[str, dict[str, float]]


def _sigmoid_scalar_clipped(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0))))


def _pack_params_observed_only(params: dict[str, float]) -> np.ndarray:
    beta = float(np.clip(params["beta"], 1e-8, 1.0 - 1e-8))
    return np.asarray(
        [
            np.log(float(params["alpha"])),
            np.log(float(params["eta"])),
            np.log(float(params["lam"])),
            float(params["tau"]),
            np.log(float(params["reg"])),
            np.log(beta / (1.0 - beta)),
        ],
        dtype=float,
    )


def _alpha_log_bounds(signal_transform: GenericFamilySignalTransform) -> tuple[float, float]:
    if signal_transform == GenericFamilySignalTransform.LOG_SATIETY:
        return OBSERVED_ONLY_LOG_SATIETY_ALPHA_LOG_BOUNDS
    if signal_transform == GenericFamilySignalTransform.POWER:
        return OBSERVED_ONLY_POWER_ALPHA_LOG_BOUNDS
    raise ValueError(f"Unsupported signal_transform: {signal_transform}")


def _unpack_params_observed_only(
    z: np.ndarray,
    *,
    signal_transform: GenericFamilySignalTransform = GenericFamilySignalTransform.LOG_SATIETY,
) -> dict[str, float]:
    alpha_lo, alpha_hi = _alpha_log_bounds(signal_transform)
    return {
        "alpha": float(np.exp(np.clip(z[0], alpha_lo, alpha_hi))),
        "eta": float(np.exp(np.clip(z[1], -8.0, 8.0))),
        "lam": float(np.exp(np.clip(z[2], -12.0, 4.0))),
        "tau": float(np.clip(z[3], -2.0, 8.0)),
        "reg": float(np.exp(np.clip(z[4], -18.0, 0.0))),
        "beta": float(np.clip(_sigmoid_scalar_clipped(float(z[5])), 1e-6, 1.0 - 1e-6)),
    }


def _evaluate_params_observed_only(
    z: np.ndarray,
    packet: GenericFamilyPacket,
    *,
    seed: int = 0,
    lower_tail_frac: float = OBSERVED_ONLY_LOWER_TAIL_FRAC,
    signal_transform: GenericFamilySignalTransform = GenericFamilySignalTransform.LOG_SATIETY,
) -> dict[str, float | bool]:
    params = _unpack_params_observed_only(z, signal_transform=signal_transform)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros_like(packet.base.y)
    fold_regrets: list[float] = []

    for tr, te in kf.split(packet.base.w):
        model = GenericFamilyRetainedTotalSurrogate(
            packet,
            params=params,
            signal_transform=signal_transform,
        ).fit(packet.base.w[tr], packet.base.y[tr])
        pred = model.predict(packet.base.w[te])
        oof[te] = pred
        fold_regrets.append(float(packet.base.y[te][int(np.argmin(pred))] - np.min(packet.base.y[te])))

    residuals = oof - packet.base.y
    sst = float(np.sum((packet.base.y - np.mean(packet.base.y)) ** 2))
    tail_count = max(5, int(np.ceil(float(lower_tail_frac) * float(len(packet.base.y)))))
    tail_idx = np.argsort(oof)[:tail_count]
    lower_tail_optimism = float(np.mean(np.maximum(packet.base.y[tail_idx] - oof[tail_idx], 0.0)))

    metrics = {
        **params,
        "cv_rmse": float(np.sqrt(np.mean(residuals**2))),
        "cv_r2": float(1.0 - float(np.sum(residuals**2)) / sst),
        "cv_spearman": float(spearmanr(packet.base.y, oof).statistic),
        "cv_regret_at_1": float(packet.base.y[int(np.argmin(oof))] - np.min(packet.base.y)),
        "cv_foldmean_regret_at_1": float(np.mean(fold_regrets)),
        "lower_tail_optimism": lower_tail_optimism,
    }
    metrics["objective"] = (
        OBSERVED_ONLY_CV_WEIGHT * float(metrics["cv_rmse"])
        + OBSERVED_ONLY_FOLDMEAN_WEIGHT * float(metrics["cv_foldmean_regret_at_1"])
        + OBSERVED_ONLY_TAIL_WEIGHT * float(metrics["lower_tail_optimism"])
    )
    return metrics


def tune_genericfamily_subset_params_observed_only(
    packet: GenericFamilyPacket,
    *,
    method: str = DEFAULT_TUNING_METHOD,
    start_bank: tuple[dict[str, float], ...] | list[dict[str, float]] | None = None,
    seed: int = 0,
    signal_transform: GenericFamilySignalTransform = GenericFamilySignalTransform.LOG_SATIETY,
) -> tuple[dict[str, float | bool], Any]:
    """Tune nonlinear GRP params using the observed-only tail-aware CV objective."""
    if start_bank is None:
        start_bank = (
            OBSERVED_ONLY_START_PARAM_BANK
            if signal_transform == GenericFamilySignalTransform.LOG_SATIETY
            else OBSERVED_ONLY_POWER_START_PARAM_BANK
        )
    best_metrics: dict[str, float | bool] | None = None
    best_result: Any | None = None
    best_objective = float("inf")
    options = {
        "L-BFGS-B": {"maxiter": 120, "ftol": 1e-6},
        "Nelder-Mead": {"maxiter": 600, "xatol": 1e-4, "fatol": 1e-6},
        "Powell": {"maxiter": 220, "xtol": 1e-4, "ftol": 1e-6},
    }.get(method, {"maxiter": 200})

    for start_id, start_params in enumerate(start_bank):
        start = _pack_params_observed_only(start_params)

        def objective(z: np.ndarray) -> float:
            return float(
                _evaluate_params_observed_only(
                    z,
                    packet,
                    seed=seed,
                    signal_transform=signal_transform,
                )["objective"]
            )

        result = minimize(objective, start, method=method, options=options)
        metrics = _evaluate_params_observed_only(
            np.asarray(result.x, dtype=float),
            packet,
            seed=seed,
            signal_transform=signal_transform,
        )
        metrics = {
            "success": bool(result.success),
            "message": str(result.message),
            "method": method,
            "objective_name": "observed_only_tail",
            "start_id": int(start_id),
            **metrics,
        }
        if float(metrics["objective"]) < best_objective:
            best_objective = float(metrics["objective"])
            best_metrics = metrics
            best_result = result

    if best_metrics is None or best_result is None:
        raise RuntimeError("Observed-only GRP tuning failed")
    return best_metrics, best_result


def deploy_genericfamily_trustblend_topkactual(
    packet: GenericFamilyPacket,
    model: GenericFamilyRetainedTotalSurrogate,
    tuning_metrics: dict[str, float | bool],
    *,
    top_k: int = TRUSTBLEND_TOPK_ACTUAL,
    line_grid: int = TRUSTBLEND_LINE_GRID,
) -> dict[str, Any]:
    """Deploy by blending top-k-actual hull optimum toward the raw optimum under a gain budget."""
    raw_result, phase0, phase1 = optimize_generic_family_model(packet, model, seed=0)
    raw_weights = np.stack([phase0, phase1], axis=0)
    top_indices = np.argsort(packet.base.y)[: min(int(top_k), len(packet.base.y))]
    hull_anchor_weights = packet.base.w[top_indices]
    hull_predicted_value, hull_coeffs, hull_weights = optimize_generic_family_convex_hull(
        model,
        hull_anchor_weights,
        start_indices=np.arange(min(len(top_indices), 8), dtype=int),
    )

    gain_budget = float(tuning_metrics["cv_rmse"]) + float(tuning_metrics["cv_foldmean_regret_at_1"])
    raw_predicted_value = float(raw_result.fun)
    target_gain = min(float(hull_predicted_value) - raw_predicted_value, gain_budget)
    best: tuple[tuple[int, float, float], float, float, np.ndarray, float] | None = None

    for delta in np.linspace(0.0, 1.0, int(line_grid)):
        weights = (1.0 - delta) * hull_weights + delta * raw_weights
        predicted_value = float(model.predict(weights[None, :, :])[0])
        realized_gain = float(hull_predicted_value) - predicted_value
        feasible = realized_gain <= gain_budget + 1e-12
        key = (0 if feasible else 1, predicted_value, abs(realized_gain - target_gain))
        if best is None or key < best[0]:
            best = (key, float(delta), predicted_value, weights, realized_gain)

    if best is None:
        raise RuntimeError("Observed-only trustblend deployment selection failed")

    _key, delta, predicted_value, weights, realized_gain = best
    return {
        "predicted_optimum_value": predicted_value,
        "weights": weights,
        "delta": delta,
        "realized_gain": realized_gain,
        "gain_budget": gain_budget,
        "raw_predicted_optimum_value": raw_predicted_value,
        "hull_predicted_optimum_value": float(hull_predicted_value),
        "hull_top_indices": top_indices.tolist(),
        "hull_top_run_names": [str(packet.base.frame.iloc[idx][packet.base.name_col]) for idx in top_indices.tolist()],
        "hull_coefficients": np.asarray(hull_coeffs, dtype=float),
        "optimizer_success": bool(raw_result.success),
        "optimizer_message": str(raw_result.message),
    }


def genericfamily_observed_only_trustblend_subset_optimum_run_id(subset_size: int) -> int:
    """Return the canonical run id for one observed-only trustblend subset optimum."""
    if subset_size not in GENERICFAMILY_RETUNED_SUBSET_OPTIMA_ALL_SUBSET_SIZES:
        raise ValueError(f"Unsupported subset size: {subset_size}")
    return GENERICFAMILY_OBSERVED_ONLY_TRUSTBLEND_SUBSET_OPTIMA_BASE_RUN_ID + (
        GENERICFAMILY_RETUNED_SUBSET_OPTIMA_ALL_SUBSET_SIZES.index(subset_size)
    )


def genericfamily_observed_only_trustblend_subset_optimum_run_name(subset_size: int) -> str:
    """Return the canonical run name for one observed-only trustblend subset optimum."""
    return f"baseline_genericfamily_trustblend_top8actual_cap_k{subset_size:03d}_uncheatable_bpb"


def _summary_to_dict(summary: GenericFamilyObservedOnlyTrustblendSubsetOptimumSummary) -> dict[str, Any]:
    return {
        "subset_size": summary.subset_size,
        "run_id": summary.run_id,
        "run_name": summary.run_name,
        "policy": summary.policy,
        "objective_metric": summary.objective_metric,
        "tuning_method": summary.tuning_method,
        "tuning_objective_name": summary.tuning_objective_name,
        "tuning_objective": summary.tuning_objective,
        "tuning_cv_rmse": summary.tuning_cv_rmse,
        "tuning_cv_r2": summary.tuning_cv_r2,
        "tuning_cv_spearman": summary.tuning_cv_spearman,
        "tuning_cv_regret_at_1": summary.tuning_cv_regret_at_1,
        "tuning_cv_foldmean_regret_at_1": summary.tuning_cv_foldmean_regret_at_1,
        "tuning_lower_tail_optimism": summary.tuning_lower_tail_optimism,
        "tuned_params": summary.tuned_params,
        "predicted_optimum_value": summary.predicted_optimum_value,
        "deployment_delta": summary.deployment_delta,
        "deployment_realized_gain": summary.deployment_realized_gain,
        "deployment_gain_budget": summary.deployment_gain_budget,
        "deployment_raw_predicted_optimum_value": summary.deployment_raw_predicted_optimum_value,
        "deployment_hull_predicted_optimum_value": summary.deployment_hull_predicted_optimum_value,
        "fullswarm_chosen_run_name": summary.fullswarm_chosen_run_name,
        "fullswarm_chosen_value": summary.fullswarm_chosen_value,
        "fullswarm_regret_at_1": summary.fullswarm_regret_at_1,
        "observed_best_run_name": summary.observed_best_run_name,
        "observed_best_value": summary.observed_best_value,
        "gap_below_observed_best": summary.gap_below_observed_best,
        "nearest_observed_run_name": summary.nearest_observed_run_name,
        "nearest_observed_value": summary.nearest_observed_value,
        "nearest_observed_tv_distance": summary.nearest_observed_tv_distance,
        "optimum_move_mean_phase_tv_vs_prev": summary.optimum_move_mean_phase_tv_vs_prev,
        "hull_anchor_count": summary.hull_anchor_count,
        "hull_anchor_summaries": summary.hull_anchor_summaries,
        "phase0_max_weight": summary.phase0_max_weight,
        "phase1_max_weight": summary.phase1_max_weight,
        "phase0_support_below_1e4": summary.phase0_support_below_1e4,
        "phase1_support_below_1e4": summary.phase1_support_below_1e4,
        "phase0_top_domains": summary.phase0_top_domains,
        "phase1_top_domains": summary.phase1_top_domains,
        "optimizer_success": summary.optimizer_success,
        "optimizer_message": summary.optimizer_message,
        "tuning_success": summary.tuning_success,
        "tuning_message": summary.tuning_message,
        "family_shares": summary.family_shares,
        "phase_weights": summary.phase_weights,
    }


@cache
def genericfamily_observed_only_trustblend_subset_optima_summaries(
    subset_sizes: tuple[int, ...] = GENERICFAMILY_OBSERVED_ONLY_TRUSTBLEND_SUBSET_OPTIMA_REPRESENTATIVE_SUBSET_SIZES,
    tuning_method: str = DEFAULT_TUNING_METHOD,
) -> tuple[GenericFamilyObservedOnlyTrustblendSubsetOptimumSummary, ...]:
    """Return observed-only trustblend subset-fit deployment summaries."""
    _, spec, _ = load_two_phase_many_candidate_summary_spec(
        CSV_PATH,
        objective_metric=OBJECTIVE_METRIC,
        name="two_phase_many_genericfamily_observed_only_trustblend_subset_optima",
    )
    packet = load_generic_family_packet(target=OBJECTIVE_METRIC)
    best_idx = int(np.argmin(packet.base.y))
    best_value = float(packet.base.y[best_idx])
    previous_deployment: np.ndarray | None = None
    summaries: list[GenericFamilyObservedOnlyTrustblendSubsetOptimumSummary] = []

    for subset_size in subset_sizes:
        selection = retrospective_generic_selection(spec, method="feature_bayes_linear", k=subset_size, seed=0)
        subset_indices = np.asarray(selection.selected_indices, dtype=int)
        train_packet = _subset_packet(packet, subset_indices)
        tuning_metrics, _ = tune_genericfamily_subset_params_observed_only(
            train_packet,
            method=tuning_method,
            seed=0,
        )
        tuned_params = {key: float(tuning_metrics[key]) for key in ("alpha", "eta", "lam", "tau", "reg", "beta")}
        model = GenericFamilyRetainedTotalSurrogate(train_packet, params=tuned_params).fit(
            train_packet.base.w,
            train_packet.base.y,
        )
        deployment = deploy_genericfamily_trustblend_topkactual(train_packet, model, tuning_metrics)
        weights = np.asarray(deployment["weights"], dtype=float)

        fullswarm_predictions = model.predict(packet.base.w)
        chosen_idx = int(np.argmin(fullswarm_predictions))
        distances = 0.5 * np.abs(packet.base.w - weights[None, :, :]).sum(axis=2).mean(axis=1)
        nearest_idx = int(np.argmin(distances))
        hull_coeffs = np.asarray(deployment["hull_coefficients"], dtype=float)
        coeff_order = np.argsort(hull_coeffs)[::-1]
        hull_anchor_summaries = [
            {
                "run_name": str(
                    train_packet.base.frame.iloc[deployment["hull_top_indices"][idx]][train_packet.base.name_col]
                ),
                "actual_value": float(train_packet.base.y[deployment["hull_top_indices"][idx]]),
                "coefficient": float(hull_coeffs[idx]),
            }
            for idx in coeff_order
        ]

        summaries.append(
            GenericFamilyObservedOnlyTrustblendSubsetOptimumSummary(
                subset_size=subset_size,
                run_id=genericfamily_observed_only_trustblend_subset_optimum_run_id(subset_size),
                run_name=genericfamily_observed_only_trustblend_subset_optimum_run_name(subset_size),
                policy=GENERICFAMILY_OBSERVED_ONLY_TRUSTBLEND_SUBSET_OPTIMA_POLICY,
                objective_metric=OBJECTIVE_METRIC,
                tuning_method=tuning_method,
                tuning_objective_name=GENERICFAMILY_OBSERVED_ONLY_TRUSTBLEND_SUBSET_OPTIMA_OBJECTIVE,
                tuning_objective=float(tuning_metrics["objective"]),
                tuning_cv_rmse=float(tuning_metrics["cv_rmse"]),
                tuning_cv_r2=float(tuning_metrics["cv_r2"]),
                tuning_cv_spearman=float(tuning_metrics["cv_spearman"]),
                tuning_cv_regret_at_1=float(tuning_metrics["cv_regret_at_1"]),
                tuning_cv_foldmean_regret_at_1=float(tuning_metrics["cv_foldmean_regret_at_1"]),
                tuning_lower_tail_optimism=float(tuning_metrics["lower_tail_optimism"]),
                tuned_params=tuned_params,
                predicted_optimum_value=float(deployment["predicted_optimum_value"]),
                deployment_delta=float(deployment["delta"]),
                deployment_realized_gain=float(deployment["realized_gain"]),
                deployment_gain_budget=float(deployment["gain_budget"]),
                deployment_raw_predicted_optimum_value=float(deployment["raw_predicted_optimum_value"]),
                deployment_hull_predicted_optimum_value=float(deployment["hull_predicted_optimum_value"]),
                fullswarm_chosen_run_name=str(packet.base.frame.iloc[chosen_idx][packet.base.name_col]),
                fullswarm_chosen_value=float(packet.base.y[chosen_idx]),
                fullswarm_regret_at_1=float(packet.base.y[chosen_idx] - best_value),
                observed_best_run_name=str(packet.base.frame.iloc[best_idx][packet.base.name_col]),
                observed_best_value=best_value,
                gap_below_observed_best=float(deployment["predicted_optimum_value"] - best_value),
                nearest_observed_run_name=str(packet.base.frame.iloc[nearest_idx][packet.base.name_col]),
                nearest_observed_value=float(packet.base.y[nearest_idx]),
                nearest_observed_tv_distance=float(distances[nearest_idx]),
                optimum_move_mean_phase_tv_vs_prev=(
                    None if previous_deployment is None else _mean_phase_tv_distance(weights, previous_deployment)
                ),
                hull_anchor_count=len(deployment["hull_top_indices"]),
                hull_anchor_summaries=hull_anchor_summaries,
                phase0_max_weight=float(weights[0].max()),
                phase1_max_weight=float(weights[1].max()),
                phase0_support_below_1e4=int(np.sum(weights[0] < 1e-4)),
                phase1_support_below_1e4=int(np.sum(weights[1] < 1e-4)),
                phase0_top_domains=_top_domains(packet.base.domain_names, weights[0], weights[0] * packet.base.c0),
                phase1_top_domains=_top_domains(packet.base.domain_names, weights[1], weights[1] * packet.base.c1),
                optimizer_success=bool(deployment["optimizer_success"]),
                optimizer_message=str(deployment["optimizer_message"]),
                tuning_success=bool(tuning_metrics["success"]),
                tuning_message=str(tuning_metrics["message"]),
                family_shares=family_shares(packet, weights),
                phase_weights=_phase_weights_from_array(packet.base.domain_names, weights),
            )
        )
        previous_deployment = weights

    return tuple(summaries)


def genericfamily_observed_only_trustblend_subset_optima_summaries_json(
    subset_sizes: tuple[int, ...] = GENERICFAMILY_OBSERVED_ONLY_TRUSTBLEND_SUBSET_OPTIMA_REPRESENTATIVE_SUBSET_SIZES,
    tuning_method: str = DEFAULT_TUNING_METHOD,
) -> str:
    """Return the observed-only trustblend subset summaries as JSON."""
    return json.dumps(
        [
            _summary_to_dict(summary)
            for summary in genericfamily_observed_only_trustblend_subset_optima_summaries(subset_sizes, tuning_method)
        ],
        indent=2,
    )


def genericfamily_observed_only_trustblend_subset_optima_summaries_frame(
    subset_sizes: tuple[int, ...] = GENERICFAMILY_OBSERVED_ONLY_TRUSTBLEND_SUBSET_OPTIMA_REPRESENTATIVE_SUBSET_SIZES,
    tuning_method: str = DEFAULT_TUNING_METHOD,
) -> pd.DataFrame:
    """Return a flat summary frame for the observed-only trustblend subset sweep."""
    return pd.DataFrame(
        [
            {
                "subset_size": summary.subset_size,
                "run_id": summary.run_id,
                "run_name": summary.run_name,
                "policy": summary.policy,
                "tuning_method": summary.tuning_method,
                "predicted_optimum_value": summary.predicted_optimum_value,
                "deployment_delta": summary.deployment_delta,
                "deployment_gain_budget": summary.deployment_gain_budget,
                "deployment_raw_predicted_optimum_value": summary.deployment_raw_predicted_optimum_value,
                "deployment_hull_predicted_optimum_value": summary.deployment_hull_predicted_optimum_value,
                "fullswarm_chosen_run_name": summary.fullswarm_chosen_run_name,
                "fullswarm_chosen_value": summary.fullswarm_chosen_value,
                "fullswarm_regret_at_1": summary.fullswarm_regret_at_1,
                "observed_best_value": summary.observed_best_value,
                "gap_below_observed_best": summary.gap_below_observed_best,
                "nearest_observed_run_name": summary.nearest_observed_run_name,
                "nearest_observed_value": summary.nearest_observed_value,
                "nearest_observed_tv_distance": summary.nearest_observed_tv_distance,
                "optimum_move_mean_phase_tv_vs_prev": summary.optimum_move_mean_phase_tv_vs_prev,
                "phase0_max_weight": summary.phase0_max_weight,
                "phase1_max_weight": summary.phase1_max_weight,
                "phase0_support_below_1e4": summary.phase0_support_below_1e4,
                "phase1_support_below_1e4": summary.phase1_support_below_1e4,
                "tuning_objective": summary.tuning_objective,
                "tuning_cv_rmse": summary.tuning_cv_rmse,
                "tuning_cv_r2": summary.tuning_cv_r2,
                "tuning_cv_spearman": summary.tuning_cv_spearman,
                "tuning_cv_regret_at_1": summary.tuning_cv_regret_at_1,
                "tuning_cv_foldmean_regret_at_1": summary.tuning_cv_foldmean_regret_at_1,
                "tuning_lower_tail_optimism": summary.tuning_lower_tail_optimism,
                "hull_anchor_count": summary.hull_anchor_count,
            }
            for summary in genericfamily_observed_only_trustblend_subset_optima_summaries(subset_sizes, tuning_method)
        ]
    )


def create_genericfamily_observed_only_trustblend_subset_optimum_weight_config(subset_size: int) -> WeightConfig:
    """Return the weight config for one observed-only trustblend subset-fit deployment."""
    summary = next(
        summary
        for summary in genericfamily_observed_only_trustblend_subset_optima_summaries((subset_size,))
        if summary.subset_size == subset_size
    )
    return WeightConfig(run_id=summary.run_id, phase_weights=summary.phase_weights)


def create_genericfamily_observed_only_trustblend_subset_optima_weight_configs(
    subset_sizes: tuple[int, ...] = GENERICFAMILY_OBSERVED_ONLY_TRUSTBLEND_SUBSET_OPTIMA_REPRESENTATIVE_SUBSET_SIZES,
) -> tuple[WeightConfig, ...]:
    """Return observed-only trustblend subset-fit deployment weight configs."""
    return tuple(
        WeightConfig(run_id=summary.run_id, phase_weights=summary.phase_weights)
        for summary in genericfamily_observed_only_trustblend_subset_optima_summaries(subset_sizes)
    )
