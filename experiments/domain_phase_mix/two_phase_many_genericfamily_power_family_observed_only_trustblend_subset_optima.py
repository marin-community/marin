# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Observed-only trustblend subset-fit deployments for power-family GRP."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.exploratory.two_phase_many.dataset_metadata import (
    load_two_phase_many_candidate_summary_spec,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_followup import (
    family_shares,
    load_generic_family_packet,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_flexible_signal import (
    GenericFamilyFlexibleSignalSurrogate,
    deploy_flexible_signal_gaincapped_topkactual,
    flexible_signal_params_from_metrics,
    tune_flexible_signal_params_observed_only,
)
from experiments.domain_phase_mix.static_batch_selection import retrospective_generic_selection
from experiments.domain_phase_mix.two_phase_many_genericfamily_observed_only_trustblend_subset_optima import (
    DEFAULT_TUNING_METHOD,
    GENERICFAMILY_OBSERVED_ONLY_TRUSTBLEND_SUBSET_OPTIMA_OBJECTIVE,
    GENERICFAMILY_OBSERVED_ONLY_TRUSTBLEND_SUBSET_OPTIMA_REPRESENTATIVE_SUBSET_SIZES,
    OBJECTIVE_METRIC,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_retuned_subset_optima import (
    CSV_PATH,
    GENERICFAMILY_RETUNED_SUBSET_OPTIMA_ALL_SUBSET_SIZES,
    _mean_phase_tv_distance,
    _phase_weights_from_array,
    _subset_packet,
    _top_domains,
)

GENERICFAMILY_POWER_FAMILY_OBSERVED_ONLY_TRUSTBLEND_SUBSET_OPTIMA_SOURCE_EXPERIMENT = (
    "pinlin_calvin_xu/data_mixture/ngd3dm2_genericfamily_power_family_observed_only_trustblend_"
    "top8actual_cap_subset_optima_rep_uncheatable_bpb"
)
GENERICFAMILY_POWER_FAMILY_OBSERVED_ONLY_TRUSTBLEND_SUBSET_OPTIMA_BASE_RUN_ID = 420
GENERICFAMILY_POWER_FAMILY_OBSERVED_ONLY_TRUSTBLEND_SUBSET_OPTIMA_POLICY = (
    "feature_bayes_linear_power_family_observed_only_trustblend_top8actual_cap"
)
GENERICFAMILY_POWER_FAMILY_OBSERVED_ONLY_TRUSTBLEND_VARIANT = "power_family"


@dataclass(frozen=True)
class GenericFamilyPowerFamilyObservedOnlyTrustblendSubsetOptimumSummary:
    """Summary for one power-family observed-only trustblend subset-fit deployment."""

    subset_size: int
    run_id: int
    run_name: str
    policy: str
    objective_metric: str
    tuning_method: str
    tuning_objective_name: str
    tuning_objective: float
    tuning_cv_rmse: float
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


def genericfamily_power_family_observed_only_trustblend_subset_optimum_run_id(subset_size: int) -> int:
    """Return the canonical run id for one power-family trustblend subset optimum."""
    if subset_size not in GENERICFAMILY_RETUNED_SUBSET_OPTIMA_ALL_SUBSET_SIZES:
        raise ValueError(f"Unsupported subset size: {subset_size}")
    return GENERICFAMILY_POWER_FAMILY_OBSERVED_ONLY_TRUSTBLEND_SUBSET_OPTIMA_BASE_RUN_ID + (
        GENERICFAMILY_RETUNED_SUBSET_OPTIMA_ALL_SUBSET_SIZES.index(subset_size)
    )


def genericfamily_power_family_observed_only_trustblend_subset_optimum_run_name(subset_size: int) -> str:
    """Return the canonical run name for one power-family trustblend subset optimum."""
    return f"baseline_genericfamily_power_family_trustblend_top8actual_cap_k{subset_size:03d}_uncheatable_bpb"


def _summary_to_dict(summary: GenericFamilyPowerFamilyObservedOnlyTrustblendSubsetOptimumSummary) -> dict[str, Any]:
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
def genericfamily_power_family_observed_only_trustblend_subset_optima_summaries(
    subset_sizes: tuple[int, ...] = GENERICFAMILY_OBSERVED_ONLY_TRUSTBLEND_SUBSET_OPTIMA_REPRESENTATIVE_SUBSET_SIZES,
    tuning_method: str = DEFAULT_TUNING_METHOD,
) -> tuple[GenericFamilyPowerFamilyObservedOnlyTrustblendSubsetOptimumSummary, ...]:
    """Return power-family observed-only trustblend subset-fit deployment summaries."""
    _, spec, _ = load_two_phase_many_candidate_summary_spec(
        CSV_PATH,
        objective_metric=OBJECTIVE_METRIC,
        name="two_phase_many_genericfamily_power_family_observed_only_trustblend_subset_optima",
    )
    packet = load_generic_family_packet(target=OBJECTIVE_METRIC)
    best_idx = int(np.argmin(packet.base.y))
    best_value = float(packet.base.y[best_idx])
    previous_deployment: np.ndarray | None = None
    summaries: list[GenericFamilyPowerFamilyObservedOnlyTrustblendSubsetOptimumSummary] = []

    for subset_size in subset_sizes:
        selection = retrospective_generic_selection(spec, method="feature_bayes_linear", k=subset_size, seed=0)
        subset_indices = np.asarray(selection.selected_indices, dtype=int)
        train_packet = _subset_packet(packet, subset_indices)
        _, _, tuning_metrics, _ = tune_flexible_signal_params_observed_only(
            train_packet,
            variant_name=GENERICFAMILY_POWER_FAMILY_OBSERVED_ONLY_TRUSTBLEND_VARIANT,
            method=tuning_method,
            coarse_top_k=4,
            seed=0,
        )
        tuned_params = flexible_signal_params_from_metrics(
            tuning_metrics,
            GENERICFAMILY_POWER_FAMILY_OBSERVED_ONLY_TRUSTBLEND_VARIANT,
        )
        model = GenericFamilyFlexibleSignalSurrogate(
            train_packet,
            params=tuned_params,
            signal_kind="power",
            family_curvature=True,
        ).fit(train_packet.base.w, train_packet.base.y)
        deployment = deploy_flexible_signal_gaincapped_topkactual(train_packet, model, tuning_metrics)
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
            GenericFamilyPowerFamilyObservedOnlyTrustblendSubsetOptimumSummary(
                subset_size=subset_size,
                run_id=genericfamily_power_family_observed_only_trustblend_subset_optimum_run_id(subset_size),
                run_name=genericfamily_power_family_observed_only_trustblend_subset_optimum_run_name(subset_size),
                policy=GENERICFAMILY_POWER_FAMILY_OBSERVED_ONLY_TRUSTBLEND_SUBSET_OPTIMA_POLICY,
                objective_metric=OBJECTIVE_METRIC,
                tuning_method=tuning_method,
                tuning_objective_name=GENERICFAMILY_OBSERVED_ONLY_TRUSTBLEND_SUBSET_OPTIMA_OBJECTIVE,
                tuning_objective=float(tuning_metrics["objective"]),
                tuning_cv_rmse=float(tuning_metrics["cv_rmse"]),
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


def genericfamily_power_family_observed_only_trustblend_subset_optima_summaries_json(
    subset_sizes: tuple[int, ...] = GENERICFAMILY_OBSERVED_ONLY_TRUSTBLEND_SUBSET_OPTIMA_REPRESENTATIVE_SUBSET_SIZES,
    tuning_method: str = DEFAULT_TUNING_METHOD,
) -> str:
    """Return the power-family observed-only trustblend subset summaries as JSON."""
    return json.dumps(
        [
            _summary_to_dict(summary)
            for summary in genericfamily_power_family_observed_only_trustblend_subset_optima_summaries(
                subset_sizes,
                tuning_method,
            )
        ],
        indent=2,
    )


def genericfamily_power_family_observed_only_trustblend_subset_optima_summaries_frame(
    subset_sizes: tuple[int, ...] = GENERICFAMILY_OBSERVED_ONLY_TRUSTBLEND_SUBSET_OPTIMA_REPRESENTATIVE_SUBSET_SIZES,
    tuning_method: str = DEFAULT_TUNING_METHOD,
) -> pd.DataFrame:
    """Return a flat summary frame for the power-family subset sweep."""
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
                "tuning_cv_regret_at_1": summary.tuning_cv_regret_at_1,
                "tuning_cv_foldmean_regret_at_1": summary.tuning_cv_foldmean_regret_at_1,
                "tuning_lower_tail_optimism": summary.tuning_lower_tail_optimism,
                "hull_anchor_count": summary.hull_anchor_count,
            }
            for summary in genericfamily_power_family_observed_only_trustblend_subset_optima_summaries(
                subset_sizes,
                tuning_method,
            )
        ]
    )


def create_genericfamily_power_family_observed_only_trustblend_subset_optimum_weight_config(
    subset_size: int,
) -> WeightConfig:
    """Return the weight config for one power-family trustblend subset-fit deployment."""
    summary = next(
        summary
        for summary in genericfamily_power_family_observed_only_trustblend_subset_optima_summaries((subset_size,))
        if summary.subset_size == subset_size
    )
    return WeightConfig(run_id=summary.run_id, phase_weights=summary.phase_weights)


def create_genericfamily_power_family_observed_only_trustblend_subset_optima_weight_configs(
    subset_sizes: tuple[int, ...] = GENERICFAMILY_OBSERVED_ONLY_TRUSTBLEND_SUBSET_OPTIMA_REPRESENTATIVE_SUBSET_SIZES,
) -> tuple[WeightConfig, ...]:
    """Return power-family trustblend subset-fit deployment weight configs."""
    return tuple(
        WeightConfig(run_id=summary.run_id, phase_weights=summary.phase_weights)
        for summary in genericfamily_power_family_observed_only_trustblend_subset_optima_summaries(subset_sizes)
    )
