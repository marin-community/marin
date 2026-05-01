# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Observed-only trustblend GRP baselines for family-curvature signal variants."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import cache
from collections.abc import Iterable

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_flexible_signal import (
    build_flexible_signal_surrogate,
    compute_flexible_surrogate_metrics,
    deploy_flexible_signal_gaincapped_topkactual,
    flexible_signal_params_from_metrics,
    tune_flexible_signal_params_observed_only,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_followup import (
    family_shares,
    load_generic_family_packet,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_observed_only_trustblend_subset_optima import (
    DEFAULT_TUNING_METHOD,
    OBJECTIVE_METRIC,
    TRUSTBLEND_TOPK_ACTUAL,
    _phase_weights_from_array,
)

GENERICFAMILY_FAMILY_CURVATURE_OBSERVED_ONLY_TRUSTBLEND_SOURCE_EXPERIMENT = (
    "pinlin_calvin_xu/data_mixture/"
    "ngd3dm2_genericfamily_family_curvature_observed_only_trustblend_top8actual_cap_uncheatable_bpb"
)


@dataclass(frozen=True)
class GenericFamilyFlexibleSignalVariantSpec:
    """Identity for one flexible-signal deployment variant."""

    variant_name: str
    run_name: str
    run_id: int


GENERICFAMILY_FAMILY_CURVATURE_VARIANT_SPECS: tuple[GenericFamilyFlexibleSignalVariantSpec, ...] = (
    GenericFamilyFlexibleSignalVariantSpec(
        variant_name="power_family",
        run_name="baseline_genericfamily_power_family_observed_only_trustblend_top8actual_cap",
        run_id=409,
    ),
    GenericFamilyFlexibleSignalVariantSpec(
        variant_name="boxcox_family",
        run_name="baseline_genericfamily_boxcox_family_observed_only_trustblend_top8actual_cap",
        run_id=410,
    ),
    GenericFamilyFlexibleSignalVariantSpec(
        variant_name="power_boxcox_family",
        run_name="baseline_genericfamily_power_boxcox_family_observed_only_trustblend_top8actual_cap",
        run_id=411,
    ),
)
GENERICFAMILY_FAMILY_CURVATURE_DEFAULT_VARIANTS: tuple[str, ...] = tuple(
    spec.variant_name for spec in GENERICFAMILY_FAMILY_CURVATURE_VARIANT_SPECS
)
_GENERICFAMILY_FAMILY_CURVATURE_VARIANT_SPECS_BY_NAME = {
    spec.variant_name: spec for spec in GENERICFAMILY_FAMILY_CURVATURE_VARIANT_SPECS
}


@dataclass(frozen=True)
class GenericFamilyFlexibleSignalObservedOnlyTrustblendSummary:
    """Summary for a full-data observed-only trustblend flexible-signal baseline."""

    run_name: str
    run_id: int
    variant_name: str
    objective_metric: str
    tuning_method: str
    tuned_params: dict[str, float]
    tuning_objective: float
    tuning_cv_rmse: float
    tuning_cv_r2: float
    tuning_cv_regret_at_1: float
    tuning_cv_foldmean_regret_at_1: float
    tuning_lower_tail_optimism: float
    predicted_optimum_value: float
    deployment_delta: float
    deployment_realized_gain: float
    deployment_gain_budget: float
    deployment_raw_predicted_optimum_value: float
    deployment_hull_predicted_optimum_value: float
    optimizer_success: bool
    optimizer_message: str
    fullswarm_chosen_run_name: str
    fullswarm_chosen_value: float
    fullswarm_regret_at_1: float
    nearest_observed_run_name: str
    nearest_observed_value: float
    nearest_observed_tv_distance: float
    family_shares: dict[str, float]
    phase_weights: dict[str, dict[str, float]]


def _flexible_signal_variant_spec(variant_name: str) -> GenericFamilyFlexibleSignalVariantSpec:
    try:
        return _GENERICFAMILY_FAMILY_CURVATURE_VARIANT_SPECS_BY_NAME[variant_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown family-curvature variant {variant_name!r}; expected one of "
            f"{GENERICFAMILY_FAMILY_CURVATURE_DEFAULT_VARIANTS!r}"
        ) from exc


def parse_family_curvature_variants(raw_variants: str | Iterable[str] | None) -> tuple[str, ...]:
    """Parse a variant selector into a validated ordered tuple."""
    if raw_variants is None:
        return GENERICFAMILY_FAMILY_CURVATURE_DEFAULT_VARIANTS
    if isinstance(raw_variants, str):
        values = tuple(part.strip() for part in raw_variants.split(",") if part.strip())
    else:
        values = tuple(str(part).strip() for part in raw_variants if str(part).strip())
    if not values:
        raise ValueError("Expected at least one family-curvature variant")
    deduped: list[str] = []
    for value in values:
        _flexible_signal_variant_spec(value)
        if value not in deduped:
            deduped.append(value)
    return tuple(deduped)


@cache
def genericfamily_family_curvature_observed_only_trustblend_summary(
    variant_name: str,
    tuning_method: str = DEFAULT_TUNING_METHOD,
) -> GenericFamilyFlexibleSignalObservedOnlyTrustblendSummary:
    """Return the full-data observed-only trustblend summary for one flexible-signal variant."""
    spec = _flexible_signal_variant_spec(variant_name)
    packet = load_generic_family_packet(target=OBJECTIVE_METRIC)
    _, _, tuning_metrics, _ = tune_flexible_signal_params_observed_only(
        packet,
        variant_name=variant_name,
        method=tuning_method,
        coarse_top_k=4,
        seed=0,
    )
    tuned_params = flexible_signal_params_from_metrics(tuning_metrics, variant_name)
    model = build_flexible_signal_surrogate(
        packet,
        params=tuned_params,
        variant_name=variant_name,
    ).fit(packet.base.w, packet.base.y)
    fit_metrics = compute_flexible_surrogate_metrics(packet, model, seed=0)
    deployment = deploy_flexible_signal_gaincapped_topkactual(
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
    return GenericFamilyFlexibleSignalObservedOnlyTrustblendSummary(
        run_name=spec.run_name,
        run_id=spec.run_id,
        variant_name=spec.variant_name,
        objective_metric=OBJECTIVE_METRIC,
        tuning_method=tuning_method,
        tuned_params=tuned_params,
        tuning_objective=float(tuning_metrics["objective"]),
        tuning_cv_rmse=float(fit_metrics["cv_rmse"]),
        tuning_cv_r2=float(fit_metrics["cv_r2"]),
        tuning_cv_regret_at_1=float(fit_metrics["cv_regret_at_1"]),
        tuning_cv_foldmean_regret_at_1=float(fit_metrics["cv_foldmean_regret_at_1"]),
        tuning_lower_tail_optimism=float(fit_metrics["lower_tail_optimism"]),
        predicted_optimum_value=float(deployment["predicted_optimum_value"]),
        deployment_delta=float(deployment["delta"]),
        deployment_realized_gain=float(deployment["realized_gain"]),
        deployment_gain_budget=float(deployment["gain_budget"]),
        deployment_raw_predicted_optimum_value=float(deployment["raw_predicted_optimum_value"]),
        deployment_hull_predicted_optimum_value=float(deployment["hull_predicted_optimum_value"]),
        optimizer_success=bool(deployment["optimizer_success"]),
        optimizer_message=str(deployment["optimizer_message"]),
        fullswarm_chosen_run_name=str(packet.base.frame.iloc[chosen_idx][packet.base.name_col]),
        fullswarm_chosen_value=float(packet.base.y[chosen_idx]),
        fullswarm_regret_at_1=float(packet.base.y[chosen_idx] - packet.base.y[best_idx]),
        nearest_observed_run_name=str(packet.base.frame.iloc[nearest_idx][packet.base.name_col]),
        nearest_observed_value=float(packet.base.y[nearest_idx]),
        nearest_observed_tv_distance=float(distances[nearest_idx]),
        family_shares=family_shares(packet, weights),
        phase_weights=_phase_weights_from_array(packet.base.domain_names, weights),
    )


@cache
def _cached_family_curvature_summaries(
    variant_names: tuple[str, ...],
    tuning_method: str,
) -> tuple[GenericFamilyFlexibleSignalObservedOnlyTrustblendSummary, ...]:
    return tuple(
        genericfamily_family_curvature_observed_only_trustblend_summary(
            variant_name=variant_name,
            tuning_method=tuning_method,
        )
        for variant_name in variant_names
    )


def genericfamily_family_curvature_observed_only_trustblend_summaries(
    variant_names: Iterable[str] | None = None,
    *,
    tuning_method: str = DEFAULT_TUNING_METHOD,
) -> tuple[GenericFamilyFlexibleSignalObservedOnlyTrustblendSummary, ...]:
    """Return ordered summaries for the requested family-curvature variants."""
    parsed_variant_names = parse_family_curvature_variants(variant_names)
    return _cached_family_curvature_summaries(parsed_variant_names, tuning_method)


def genericfamily_family_curvature_observed_only_trustblend_summaries_frame(
    variant_names: Iterable[str] | None = None,
    *,
    tuning_method: str = DEFAULT_TUNING_METHOD,
) -> pd.DataFrame:
    """Return a tabular summary for the requested family-curvature variants."""
    return pd.DataFrame(
        asdict(summary)
        for summary in genericfamily_family_curvature_observed_only_trustblend_summaries(
            variant_names,
            tuning_method=tuning_method,
        )
    )


def genericfamily_family_curvature_observed_only_trustblend_summaries_json(
    variant_names: Iterable[str] | None = None,
    *,
    tuning_method: str = DEFAULT_TUNING_METHOD,
) -> str:
    """Return JSON summaries for the requested family-curvature variants."""
    return json.dumps(
        [
            asdict(summary)
            for summary in genericfamily_family_curvature_observed_only_trustblend_summaries(
                variant_names,
                tuning_method=tuning_method,
            )
        ],
        indent=2,
        sort_keys=True,
    )


def create_genericfamily_family_curvature_observed_only_trustblend_weight_config(
    variant_name: str,
    *,
    tuning_method: str = DEFAULT_TUNING_METHOD,
) -> WeightConfig:
    """Return the weight config for one family-curvature trustblend deployment."""
    summary = genericfamily_family_curvature_observed_only_trustblend_summary(
        variant_name=variant_name,
        tuning_method=tuning_method,
    )
    return WeightConfig(run_id=summary.run_id, phase_weights=summary.phase_weights)
