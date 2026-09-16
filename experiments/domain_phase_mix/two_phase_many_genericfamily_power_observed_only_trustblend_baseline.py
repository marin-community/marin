# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Observed-only trustblend GRP baseline with power-law signal features."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np

from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_followup import (
    GenericFamilyRetainedTotalSurrogate,
    GenericFamilySignalTransform,
    family_shares,
    load_generic_family_packet,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_observed_only_trustblend_subset_optima import (
    DEFAULT_TUNING_METHOD,
    OBJECTIVE_METRIC,
    TRUSTBLEND_TOPK_ACTUAL,
    _phase_weights_from_array,
    deploy_genericfamily_trustblend_topkactual,
    tune_genericfamily_subset_params_observed_only,
)

GENERICFAMILY_POWER_OBSERVED_ONLY_TRUSTBLEND_RUN_NAME = (
    "baseline_genericfamily_power_observed_only_trustblend_top8actual_cap"
)
GENERICFAMILY_POWER_OBSERVED_ONLY_TRUSTBLEND_SOURCE_EXPERIMENT = (
    "pinlin_calvin_xu/data_mixture/ngd3dm2_genericfamily_power_observed_only_trustblend_top8actual_cap_uncheatable_bpb"
)
GENERICFAMILY_POWER_OBSERVED_ONLY_TRUSTBLEND_RUN_ID = 408


@dataclass(frozen=True)
class GenericFamilyPowerObservedOnlyTrustblendSummary:
    """Summary for the full-data observed-only trustblend power-law baseline."""

    run_name: str
    run_id: int
    objective_metric: str
    tuning_method: str
    tuning_objective: float
    tuning_cv_rmse: float
    tuning_cv_r2: float
    tuning_cv_spearman: float
    tuning_cv_regret_at_1: float
    tuning_cv_foldmean_regret_at_1: float
    tuning_lower_tail_optimism: float
    predicted_optimum_value: float
    deployment_delta: float
    deployment_realized_gain: float
    deployment_gain_budget: float
    deployment_raw_predicted_optimum_value: float
    deployment_hull_predicted_optimum_value: float
    fullswarm_chosen_run_name: str
    fullswarm_chosen_value: float
    fullswarm_regret_at_1: float
    nearest_observed_run_name: str
    nearest_observed_value: float
    nearest_observed_tv_distance: float
    family_shares: dict[str, float]
    phase_weights: dict[str, dict[str, float]]


@cache
def genericfamily_power_observed_only_trustblend_summary(
    tuning_method: str = DEFAULT_TUNING_METHOD,
) -> GenericFamilyPowerObservedOnlyTrustblendSummary:
    """Return the full-data observed-only trustblend power-law summary."""
    packet = load_generic_family_packet(target=OBJECTIVE_METRIC)
    tuning_metrics, _ = tune_genericfamily_subset_params_observed_only(
        packet,
        method=tuning_method,
        seed=0,
        signal_transform=GenericFamilySignalTransform.POWER,
    )
    tuned_params = {key: float(tuning_metrics[key]) for key in ("alpha", "eta", "lam", "tau", "reg", "beta")}
    model = GenericFamilyRetainedTotalSurrogate(
        packet,
        params=tuned_params,
        signal_transform=GenericFamilySignalTransform.POWER,
    ).fit(packet.base.w, packet.base.y)
    deployment = deploy_genericfamily_trustblend_topkactual(
        packet,
        model,
        tuning_metrics,
        top_k=TRUSTBLEND_TOPK_ACTUAL,
    )
    weights = np.asarray(deployment["weights"], dtype=float)
    best_idx = int(np.argmin(packet.base.y))
    chosen_idx = int(np.argmin(model.predict(packet.base.w)))
    distances = 0.5 * np.abs(packet.base.w - weights[None, :, :]).sum(axis=2).mean(axis=1)
    nearest_idx = int(np.argmin(distances))

    return GenericFamilyPowerObservedOnlyTrustblendSummary(
        run_name=GENERICFAMILY_POWER_OBSERVED_ONLY_TRUSTBLEND_RUN_NAME,
        run_id=GENERICFAMILY_POWER_OBSERVED_ONLY_TRUSTBLEND_RUN_ID,
        objective_metric=OBJECTIVE_METRIC,
        tuning_method=tuning_method,
        tuning_objective=float(tuning_metrics["objective"]),
        tuning_cv_rmse=float(tuning_metrics["cv_rmse"]),
        tuning_cv_r2=float(tuning_metrics["cv_r2"]),
        tuning_cv_spearman=float(tuning_metrics["cv_spearman"]),
        tuning_cv_regret_at_1=float(tuning_metrics["cv_regret_at_1"]),
        tuning_cv_foldmean_regret_at_1=float(tuning_metrics["cv_foldmean_regret_at_1"]),
        tuning_lower_tail_optimism=float(tuning_metrics["lower_tail_optimism"]),
        predicted_optimum_value=float(deployment["predicted_optimum_value"]),
        deployment_delta=float(deployment["delta"]),
        deployment_realized_gain=float(deployment["realized_gain"]),
        deployment_gain_budget=float(deployment["gain_budget"]),
        deployment_raw_predicted_optimum_value=float(deployment["raw_predicted_optimum_value"]),
        deployment_hull_predicted_optimum_value=float(deployment["hull_predicted_optimum_value"]),
        fullswarm_chosen_run_name=str(packet.base.frame.iloc[chosen_idx][packet.base.name_col]),
        fullswarm_chosen_value=float(packet.base.y[chosen_idx]),
        fullswarm_regret_at_1=float(packet.base.y[chosen_idx] - packet.base.y[best_idx]),
        nearest_observed_run_name=str(packet.base.frame.iloc[nearest_idx][packet.base.name_col]),
        nearest_observed_value=float(packet.base.y[nearest_idx]),
        nearest_observed_tv_distance=float(distances[nearest_idx]),
        family_shares=family_shares(packet, weights),
        phase_weights=_phase_weights_from_array(packet.base.domain_names, weights),
    )


def create_genericfamily_power_observed_only_trustblend_weight_config() -> WeightConfig:
    """Return the weight config for the full-data observed-only trustblend power-law deployment."""
    summary = genericfamily_power_observed_only_trustblend_summary()
    return WeightConfig(run_id=summary.run_id, phase_weights=summary.phase_weights)
