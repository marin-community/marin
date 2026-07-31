# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Predicted GRP optima fitted on increasing observed-run subsets."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.exploratory.two_phase_many.dataset_metadata import (
    load_two_phase_many_candidate_summary_spec,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_followup import (
    GenericFamilyPacket,
    GenericFamilyRetainedTotalSurrogate,
    family_shares,
    load_generic_family_packet,
    optimize_generic_family_model,
)
from experiments.domain_phase_mix.nextgen.utils import normalize_phase_weights
from experiments.domain_phase_mix.static_batch_selection import retrospective_generic_selection

GENERICFAMILY_SUBSET_OPTIMA_SOURCE_EXPERIMENT = (
    "pinlin_calvin_xu/data_mixture/ngd3dm2_genericfamily_subset_optima_uncheatable_bpb"
)
GENERICFAMILY_SUBSET_OPTIMA_BASE_RUN_ID = 320
GENERICFAMILY_SUBSET_OPTIMA_SUBSET_SIZES = tuple(range(20, 240, 20))
GENERICFAMILY_SUBSET_OPTIMA_POLICY = "feature_bayes_linear_observed"
CSV_PATH = Path(__file__).resolve().parent / "exploratory" / "two_phase_many" / "two_phase_many.csv"
OBJECTIVE_METRIC = "eval/uncheatable_eval/bpb"


@dataclass(frozen=True)
class GenericFamilySubsetOptimumSummary:
    """Summary for one GRP optimum fitted on a subset of observed runs."""

    subset_size: int
    run_id: int
    run_name: str
    policy: str
    objective_metric: str
    predicted_optimum_value: float
    observed_best_run_name: str
    observed_best_value: float
    gap_below_observed_best: float
    nearest_observed_run_name: str
    nearest_observed_value: float
    nearest_observed_tv_distance: float
    optimum_move_mean_phase_tv_vs_prev: float | None
    phase0_max_weight: float
    phase1_max_weight: float
    phase0_support_below_1e4: int
    phase1_support_below_1e4: int
    phase0_top_domains: list[dict[str, float | str]]
    phase1_top_domains: list[dict[str, float | str]]
    optimizer_success: bool
    optimizer_message: str
    family_shares: dict[str, float]
    phase_weights: dict[str, dict[str, float]]


def _subset_packet(packet: GenericFamilyPacket, indices: np.ndarray) -> GenericFamilyPacket:
    indices = np.asarray(indices, dtype=int)
    return GenericFamilyPacket(
        base=replace(
            packet.base,
            frame=packet.base.frame.iloc[indices].reset_index(drop=True),
            y=packet.base.y[indices],
            w=packet.base.w[indices],
        ),
        pairs=packet.pairs,
        pair_topics=packet.pair_topics,
        singletons=packet.singletons,
        family_map=packet.family_map,
    )


def _phase_entropy(weights: np.ndarray) -> float:
    clipped = np.clip(np.asarray(weights, dtype=float), 1e-12, 1.0)
    return float(-np.sum(clipped * np.log(clipped)))


def _top_domains(
    domain_names: list[str],
    weights: np.ndarray,
    epochs: np.ndarray,
    *,
    top_k: int = 10,
) -> list[dict[str, float | str]]:
    frame = pd.DataFrame({"domain": domain_names, "weight": weights, "epochs": epochs})
    return frame.sort_values(["weight", "epochs"], ascending=False).head(top_k).to_dict(orient="records")


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


def _mean_phase_tv_distance(lhs: np.ndarray, rhs: np.ndarray) -> float:
    return 0.5 * float(np.mean(np.sum(np.abs(lhs - rhs), axis=1)))


def genericfamily_subset_optimum_run_name(subset_size: int) -> str:
    """Return the canonical run name for one subset-fit GRP optimum."""
    return f"baseline_genericfamily_k{subset_size:03d}_uncheatable_bpb"


def _summary_to_dict(summary: GenericFamilySubsetOptimumSummary) -> dict[str, Any]:
    return {
        "subset_size": summary.subset_size,
        "run_id": summary.run_id,
        "run_name": summary.run_name,
        "policy": summary.policy,
        "objective_metric": summary.objective_metric,
        "predicted_optimum_value": summary.predicted_optimum_value,
        "observed_best_run_name": summary.observed_best_run_name,
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
        "phase0_top_domains": summary.phase0_top_domains,
        "phase1_top_domains": summary.phase1_top_domains,
        "optimizer_success": summary.optimizer_success,
        "optimizer_message": summary.optimizer_message,
        "family_shares": summary.family_shares,
        "phase_weights": summary.phase_weights,
    }


@cache
def genericfamily_subset_optima_summaries() -> tuple[GenericFamilySubsetOptimumSummary, ...]:
    """Return predicted GRP optima for increasing subset sizes."""
    _, spec, _ = load_two_phase_many_candidate_summary_spec(
        CSV_PATH,
        objective_metric=OBJECTIVE_METRIC,
        name="two_phase_many_genericfamily_subset_optima",
    )
    packet = load_generic_family_packet(target=OBJECTIVE_METRIC)
    best_idx = int(np.argmin(packet.base.y))
    best_value = float(packet.base.y[best_idx])
    previous_optimum: np.ndarray | None = None
    summaries: list[GenericFamilySubsetOptimumSummary] = []

    for offset, subset_size in enumerate(GENERICFAMILY_SUBSET_OPTIMA_SUBSET_SIZES):
        selection = retrospective_generic_selection(spec, method="feature_bayes_linear", k=subset_size, seed=0)
        subset_indices = np.asarray(selection.selected_indices, dtype=int)
        train_packet = _subset_packet(packet, subset_indices)
        model = GenericFamilyRetainedTotalSurrogate(train_packet).fit(train_packet.base.w, train_packet.base.y)
        result, phase0, phase1 = optimize_generic_family_model(train_packet, model, seed=0)
        optimum = np.stack([phase0, phase1], axis=0)
        distances = 0.5 * np.abs(packet.base.w - optimum[None, :, :]).sum(axis=2).mean(axis=1)
        nearest_idx = int(np.argmin(distances))

        summaries.append(
            GenericFamilySubsetOptimumSummary(
                subset_size=subset_size,
                run_id=GENERICFAMILY_SUBSET_OPTIMA_BASE_RUN_ID + offset,
                run_name=genericfamily_subset_optimum_run_name(subset_size),
                policy=GENERICFAMILY_SUBSET_OPTIMA_POLICY,
                objective_metric=OBJECTIVE_METRIC,
                predicted_optimum_value=float(result.fun),
                observed_best_run_name=str(packet.base.frame.iloc[best_idx][packet.base.name_col]),
                observed_best_value=best_value,
                gap_below_observed_best=float(result.fun - best_value),
                nearest_observed_run_name=str(packet.base.frame.iloc[nearest_idx][packet.base.name_col]),
                nearest_observed_value=float(packet.base.y[nearest_idx]),
                nearest_observed_tv_distance=float(distances[nearest_idx]),
                optimum_move_mean_phase_tv_vs_prev=(
                    None if previous_optimum is None else _mean_phase_tv_distance(optimum, previous_optimum)
                ),
                phase0_max_weight=float(phase0.max()),
                phase1_max_weight=float(phase1.max()),
                phase0_support_below_1e4=int(np.sum(phase0 < 1e-4)),
                phase1_support_below_1e4=int(np.sum(phase1 < 1e-4)),
                phase0_top_domains=_top_domains(packet.base.domain_names, phase0, phase0 * packet.base.c0),
                phase1_top_domains=_top_domains(packet.base.domain_names, phase1, phase1 * packet.base.c1),
                optimizer_success=bool(result.success),
                optimizer_message=str(result.message),
                family_shares=family_shares(packet, optimum),
                phase_weights=_phase_weights_from_array(packet.base.domain_names, optimum),
            )
        )
        previous_optimum = optimum

    return tuple(summaries)


def genericfamily_subset_optima_summaries_json() -> str:
    """Return the subset-optima summaries as JSON."""
    return json.dumps([_summary_to_dict(summary) for summary in genericfamily_subset_optima_summaries()], indent=2)


def genericfamily_subset_optima_summaries_frame() -> pd.DataFrame:
    """Return a flat summary frame for the subset-optimum sweep."""
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
                "nearest_observed_run_name": summary.nearest_observed_run_name,
                "nearest_observed_value": summary.nearest_observed_value,
                "nearest_observed_tv_distance": summary.nearest_observed_tv_distance,
                "optimum_move_mean_phase_tv_vs_prev": summary.optimum_move_mean_phase_tv_vs_prev,
                "phase0_max_weight": summary.phase0_max_weight,
                "phase1_max_weight": summary.phase1_max_weight,
                "phase0_support_below_1e4": summary.phase0_support_below_1e4,
                "phase1_support_below_1e4": summary.phase1_support_below_1e4,
            }
            for summary in genericfamily_subset_optima_summaries()
        ]
    )


def create_genericfamily_subset_optimum_weight_config(subset_size: int) -> WeightConfig:
    """Return the weight config for one subset-fit predicted optimum."""
    summary = next(summary for summary in genericfamily_subset_optima_summaries() if summary.subset_size == subset_size)
    return WeightConfig(run_id=summary.run_id, phase_weights=summary.phase_weights)


def create_genericfamily_subset_optima_weight_configs() -> tuple[WeightConfig, ...]:
    """Return all subset-fit GRP optimum weight configs."""
    return tuple(
        WeightConfig(run_id=summary.run_id, phase_weights=summary.phase_weights)
        for summary in genericfamily_subset_optima_summaries()
    )
