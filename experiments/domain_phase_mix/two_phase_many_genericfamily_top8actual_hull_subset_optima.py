# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Top-8 actual frontier-hull GRP deployments fit on increasing observed-run subsets."""

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
    GenericFamilyRetainedTotalSurrogate,
    family_shares,
    load_generic_family_packet,
    optimize_generic_family_convex_hull,
)
from experiments.domain_phase_mix.nextgen.utils import normalize_phase_weights
from experiments.domain_phase_mix.static_batch_selection import retrospective_generic_selection
from experiments.domain_phase_mix.two_phase_many_ccglobalpremium_baselines import (
    ccglobalpremium_retainedtotal_summary,
)
from experiments.domain_phase_mix.two_phase_many_ccpairtotal_baseline import (
    ccpairtotal_retainedtotal_summary,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_retuned_subset_optima import (
    CSV_PATH,
    GENERICFAMILY_RETUNED_SUBSET_OPTIMA_ALL_SUBSET_SIZES,
    GENERICFAMILY_RETUNED_SUBSET_OPTIMA_REPRESENTATIVE_SUBSET_SIZES,
    OBJECTIVE_METRIC,
    VALIDATED_GLOBAL_BPB,
    VALIDATED_PAIR_BPB,
    _mean_phase_tv_distance,
    _subset_packet,
    _summary_weights,
    _top_domains,
    tune_genericfamily_subset_params,
)

GENERICFAMILY_TOP8ACTUAL_HULL_SUBSET_OPTIMA_SOURCE_EXPERIMENT = (
    "pinlin_calvin_xu/data_mixture/ngd3dm2_genericfamily_top8actual_hull_subset_optima_rep_uncheatable_bpb"
)
GENERICFAMILY_TOP8ACTUAL_HULL_SUBSET_OPTIMA_BASE_RUN_ID = 380
GENERICFAMILY_TOP8ACTUAL_HULL_SUBSET_OPTIMA_POLICY = "feature_bayes_linear_observed_top8actual_hull"
TOP8_ACTUAL_HULL_COUNT = 8


@dataclass(frozen=True)
class GenericFamilyTop8ActualHullSubsetOptimumSummary:
    """Summary for one top-8-actual-hull subset-fit deployment."""

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
    tuning_anchor_mae: float
    tuning_anchor_rmse: float
    tuned_params: dict[str, float]
    predicted_optimum_value: float
    fullswarm_chosen_run_name: str
    fullswarm_chosen_value: float
    fullswarm_regret_at_1: float
    observed_best_run_name: str
    observed_best_value: float
    gap_below_observed_best: float
    subset_best_run_name: str
    subset_best_value: float
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
    tuning_success: bool
    tuning_message: str
    family_shares: dict[str, float]
    phase_weights: dict[str, dict[str, float]]


def genericfamily_top8actual_hull_subset_optimum_run_id(subset_size: int) -> int:
    """Return the canonical run id for one top-8-actual-hull subset optimum."""
    if subset_size not in GENERICFAMILY_RETUNED_SUBSET_OPTIMA_ALL_SUBSET_SIZES:
        raise ValueError(f"Unsupported subset size: {subset_size}")
    return (
        GENERICFAMILY_TOP8ACTUAL_HULL_SUBSET_OPTIMA_BASE_RUN_ID
        + GENERICFAMILY_RETUNED_SUBSET_OPTIMA_ALL_SUBSET_SIZES.index(subset_size)
    )


def genericfamily_top8actual_hull_subset_optimum_run_name(subset_size: int) -> str:
    """Return the canonical run name for one top-8-actual-hull subset optimum."""
    return f"baseline_genericfamily_top8actual_hull_k{subset_size:03d}_uncheatable_bpb"


def _phase_weights_from_array(domain_names: list[str], weights: np.ndarray) -> dict[str, dict[str, float]]:
    return normalize_phase_weights(
        {
            "phase_0": {
                domain_name: float(weight) for domain_name, weight in zip(domain_names, weights[0], strict=True)
            },
            "phase_1": {
                domain_name: float(weight) for domain_name, weight in zip(domain_names, weights[1], strict=True)
            },
        }
    )


def _summary_to_dict(summary: GenericFamilyTop8ActualHullSubsetOptimumSummary) -> dict[str, Any]:
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
        "tuning_anchor_mae": summary.tuning_anchor_mae,
        "tuning_anchor_rmse": summary.tuning_anchor_rmse,
        "tuned_params": summary.tuned_params,
        "predicted_optimum_value": summary.predicted_optimum_value,
        "fullswarm_chosen_run_name": summary.fullswarm_chosen_run_name,
        "fullswarm_chosen_value": summary.fullswarm_chosen_value,
        "fullswarm_regret_at_1": summary.fullswarm_regret_at_1,
        "observed_best_run_name": summary.observed_best_run_name,
        "observed_best_value": summary.observed_best_value,
        "gap_below_observed_best": summary.gap_below_observed_best,
        "subset_best_run_name": summary.subset_best_run_name,
        "subset_best_value": summary.subset_best_value,
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
        "tuning_success": summary.tuning_success,
        "tuning_message": summary.tuning_message,
        "family_shares": summary.family_shares,
        "phase_weights": summary.phase_weights,
    }


@cache
def genericfamily_top8actual_hull_subset_optima_summaries(
    subset_sizes: tuple[int, ...] = GENERICFAMILY_RETUNED_SUBSET_OPTIMA_REPRESENTATIVE_SUBSET_SIZES,
) -> tuple[GenericFamilyTop8ActualHullSubsetOptimumSummary, ...]:
    """Return top-8-actual-hull subset-fit deployment summaries."""
    _, spec, _ = load_two_phase_many_candidate_summary_spec(
        CSV_PATH,
        objective_metric=OBJECTIVE_METRIC,
        name="two_phase_many_genericfamily_top8actual_hull_subset_optima",
    )
    packet = load_generic_family_packet(target=OBJECTIVE_METRIC)
    valid_weights = np.stack(
        [
            _summary_weights(ccglobalpremium_retainedtotal_summary(), packet.base.domain_names),
            _summary_weights(ccpairtotal_retainedtotal_summary(), packet.base.domain_names),
        ],
        axis=0,
    )
    valid_y = np.asarray([VALIDATED_GLOBAL_BPB, VALIDATED_PAIR_BPB], dtype=float)
    best_idx = int(np.argmin(packet.base.y))
    best_value = float(packet.base.y[best_idx])
    previous_deployment: np.ndarray | None = None
    summaries: list[GenericFamilyTop8ActualHullSubsetOptimumSummary] = []

    for subset_size in subset_sizes:
        selection = retrospective_generic_selection(spec, method="feature_bayes_linear", k=subset_size, seed=0)
        subset_indices = np.asarray(selection.selected_indices, dtype=int)
        train_packet = _subset_packet(packet, subset_indices)
        tuning_metrics, _ = tune_genericfamily_subset_params(train_packet, valid_weights, valid_y, seed=0)
        tuned_params = {key: float(tuning_metrics[key]) for key in ("alpha", "eta", "lam", "tau", "reg", "beta")}
        deploy_model = GenericFamilyRetainedTotalSurrogate(train_packet, params=tuned_params).fit(
            train_packet.base.w,
            train_packet.base.y,
        )

        actual_order = np.argsort(train_packet.base.y)
        hull_anchor_indices = actual_order[: min(TOP8_ACTUAL_HULL_COUNT, len(actual_order))]
        hull_anchor_weights = train_packet.base.w[hull_anchor_indices]
        start_indices = np.arange(min(TOP8_ACTUAL_HULL_COUNT, len(hull_anchor_indices)), dtype=int)
        deploy_predicted_value, anchor_coeffs, deployment = optimize_generic_family_convex_hull(
            deploy_model,
            hull_anchor_weights,
            start_indices=start_indices,
        )

        fullswarm_predictions = deploy_model.predict(packet.base.w)
        chosen_idx = int(np.argmin(fullswarm_predictions))
        distances = 0.5 * np.abs(packet.base.w - deployment[None, :, :]).sum(axis=2).mean(axis=1)
        nearest_idx = int(np.argmin(distances))
        anchor_order = np.argsort(anchor_coeffs)[::-1]
        hull_anchor_summaries = [
            {
                "run_name": str(train_packet.base.frame.iloc[hull_anchor_indices[idx]][train_packet.base.name_col]),
                "actual_value": float(train_packet.base.y[hull_anchor_indices[idx]]),
                "coefficient": float(anchor_coeffs[idx]),
            }
            for idx in anchor_order
        ]
        subset_best_idx_local = int(actual_order[0])

        summaries.append(
            GenericFamilyTop8ActualHullSubsetOptimumSummary(
                subset_size=subset_size,
                run_id=genericfamily_top8actual_hull_subset_optimum_run_id(subset_size),
                run_name=genericfamily_top8actual_hull_subset_optimum_run_name(subset_size),
                policy=GENERICFAMILY_TOP8ACTUAL_HULL_SUBSET_OPTIMA_POLICY,
                objective_metric=OBJECTIVE_METRIC,
                tuning_method=str(tuning_metrics["method"]),
                tuning_objective_name=str(tuning_metrics["objective_name"]),
                tuning_objective=float(tuning_metrics["objective"]),
                tuning_cv_rmse=float(tuning_metrics["cv_rmse"]),
                tuning_cv_r2=float(tuning_metrics["cv_r2"]),
                tuning_cv_spearman=float(tuning_metrics["cv_spearman"]),
                tuning_cv_regret_at_1=float(tuning_metrics["cv_regret_at_1"]),
                tuning_cv_foldmean_regret_at_1=float(tuning_metrics["cv_foldmean_regret_at_1"]),
                tuning_anchor_mae=float(tuning_metrics["anchor_mae"]),
                tuning_anchor_rmse=float(tuning_metrics["anchor_rmse"]),
                tuned_params=tuned_params,
                predicted_optimum_value=float(deploy_predicted_value),
                fullswarm_chosen_run_name=str(packet.base.frame.iloc[chosen_idx][packet.base.name_col]),
                fullswarm_chosen_value=float(packet.base.y[chosen_idx]),
                fullswarm_regret_at_1=float(packet.base.y[chosen_idx] - best_value),
                observed_best_run_name=str(packet.base.frame.iloc[best_idx][packet.base.name_col]),
                observed_best_value=best_value,
                gap_below_observed_best=float(deploy_predicted_value - best_value),
                subset_best_run_name=str(
                    train_packet.base.frame.iloc[subset_best_idx_local][train_packet.base.name_col]
                ),
                subset_best_value=float(train_packet.base.y[subset_best_idx_local]),
                nearest_observed_run_name=str(packet.base.frame.iloc[nearest_idx][packet.base.name_col]),
                nearest_observed_value=float(packet.base.y[nearest_idx]),
                nearest_observed_tv_distance=float(distances[nearest_idx]),
                optimum_move_mean_phase_tv_vs_prev=(
                    None if previous_deployment is None else _mean_phase_tv_distance(deployment, previous_deployment)
                ),
                hull_anchor_count=len(hull_anchor_indices),
                hull_anchor_summaries=hull_anchor_summaries,
                phase0_max_weight=float(deployment[0].max()),
                phase1_max_weight=float(deployment[1].max()),
                phase0_support_below_1e4=int(np.sum(deployment[0] < 1e-4)),
                phase1_support_below_1e4=int(np.sum(deployment[1] < 1e-4)),
                phase0_top_domains=_top_domains(packet.base.domain_names, deployment[0], deployment[0] * packet.base.c0),
                phase1_top_domains=_top_domains(packet.base.domain_names, deployment[1], deployment[1] * packet.base.c1),
                tuning_success=bool(tuning_metrics["success"]),
                tuning_message=str(tuning_metrics["message"]),
                family_shares=family_shares(packet, deployment),
                phase_weights=_phase_weights_from_array(packet.base.domain_names, deployment),
            )
        )
        previous_deployment = deployment

    return tuple(summaries)


def genericfamily_top8actual_hull_subset_optima_summaries_json(
    subset_sizes: tuple[int, ...] = GENERICFAMILY_RETUNED_SUBSET_OPTIMA_REPRESENTATIVE_SUBSET_SIZES,
) -> str:
    """Return the top-8-actual-hull subset summaries as JSON."""
    return json.dumps(
        [_summary_to_dict(summary) for summary in genericfamily_top8actual_hull_subset_optima_summaries(subset_sizes)],
        indent=2,
    )


def genericfamily_top8actual_hull_subset_optima_summaries_frame(
    subset_sizes: tuple[int, ...] = GENERICFAMILY_RETUNED_SUBSET_OPTIMA_REPRESENTATIVE_SUBSET_SIZES,
) -> pd.DataFrame:
    """Return a flat summary frame for the top-8-actual-hull subset sweep."""
    return pd.DataFrame(
        [
            {
                "subset_size": summary.subset_size,
                "run_id": summary.run_id,
                "run_name": summary.run_name,
                "policy": summary.policy,
                "predicted_optimum_value": summary.predicted_optimum_value,
                "observed_best_value": summary.observed_best_value,
                "gap_below_observed_best": summary.gap_below_observed_best,
                "fullswarm_chosen_run_name": summary.fullswarm_chosen_run_name,
                "fullswarm_chosen_value": summary.fullswarm_chosen_value,
                "fullswarm_regret_at_1": summary.fullswarm_regret_at_1,
                "nearest_observed_run_name": summary.nearest_observed_run_name,
                "nearest_observed_value": summary.nearest_observed_value,
                "nearest_observed_tv_distance": summary.nearest_observed_tv_distance,
                "optimum_move_mean_phase_tv_vs_prev": summary.optimum_move_mean_phase_tv_vs_prev,
                "phase0_max_weight": summary.phase0_max_weight,
                "phase1_max_weight": summary.phase1_max_weight,
                "phase0_support_below_1e4": summary.phase0_support_below_1e4,
                "phase1_support_below_1e4": summary.phase1_support_below_1e4,
                "hull_anchor_count": summary.hull_anchor_count,
            }
            for summary in genericfamily_top8actual_hull_subset_optima_summaries(subset_sizes)
        ]
    )


def create_genericfamily_top8actual_hull_subset_optimum_weight_config(subset_size: int) -> WeightConfig:
    """Return the weight config for one top-8-actual-hull subset-fit deployment."""
    summary = next(
        summary
        for summary in genericfamily_top8actual_hull_subset_optima_summaries((subset_size,))
        if summary.subset_size == subset_size
    )
    return WeightConfig(run_id=summary.run_id, phase_weights=summary.phase_weights)


def create_genericfamily_top8actual_hull_subset_optima_weight_configs(
    subset_sizes: tuple[int, ...] = GENERICFAMILY_RETUNED_SUBSET_OPTIMA_REPRESENTATIVE_SUBSET_SIZES,
) -> tuple[WeightConfig, ...]:
    """Return top-8-actual-hull subset-fit deployment weight configs."""
    return tuple(
        WeightConfig(run_id=summary.run_id, phase_weights=summary.phase_weights)
        for summary in genericfamily_top8actual_hull_subset_optima_summaries(subset_sizes)
    )
