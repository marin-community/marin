# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact subset-fit raw-optimum deployments for the no-L2 power-family-penalty GRP variant."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from functools import cache
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_followup import (
    family_shares,
    load_generic_family_packet,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_observed_only_trustblend_subset_optima import (
    OBJECTIVE_METRIC,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_retuned_subset_optima import (
    _phase_weights_from_array,
    _top_domains,
)

GENERICFAMILY_POWER_FAMILY_PENALTY_NO_L2_RAW_SUBSET_OPTIMA_SOURCE_EXPERIMENT = (
    "pinlin_calvin_xu/data_mixture/"
    "ngd3dm2_genericfamily_power_family_penalty_no_l2_raw_subset_optima_rep_uncheatable_bpb"
)
GENERICFAMILY_POWER_FAMILY_PENALTY_NO_L2_RAW_SUBSET_OPTIMA_BASE_RUN_ID = 470
GENERICFAMILY_POWER_FAMILY_PENALTY_NO_L2_RAW_SUBSET_OPTIMA_SUBSET_SIZES = (20, 40, 60, 80, 100, 140, 180, 220, 242)
GENERICFAMILY_POWER_FAMILY_PENALTY_NO_L2_RAW_SUBSET_OPTIMA_POLICY = (
    "feature_bayes_linear_power_family_penalty_no_l2_raw_optimum"
)
GENERICFAMILY_POWER_FAMILY_PENALTY_NO_L2_RAW_SUBSET_OPTIMA_VARIANT = "power_family_penalty_no_l2"
GENERICFAMILY_POWER_FAMILY_PENALTY_NO_L2_RAW_SUBSET_OPTIMA_TUNING_METHOD = "Powell"
_SUMMARY_DIR = Path(__file__).resolve().parent / "exploratory" / "two_phase_many"
SUMMARY_JSON_PATH = _SUMMARY_DIR / "two_phase_many_grp_power_family_penalty_no_l2_raw_summary.json"
_EMBEDDED_ROWS_GZIP_BASE64 = (
    "H4sIAD3g3WkC/92d65IcR3KlX0XG37OluF/0ELL9L5O1NYAm0RoADQENcriyefc9XxQuXRF5iSw0SJp2pZnRsKozKzPC47gf"
    "P8f/439++vn27f2b328+vr79cPfxp3/7l//56f3r24935ubFh4fbVzePd/941H9rTsXkVKzxJYTga4l/+5cvH/xwd/vx4d39"
    "u1/a54wJNRTvTHClWpdN+vbJx7uXr29ePry6a5+0JXv942CtS6mU9PWDtr92trnY6mpKJiT98W8f7K/ti/VO187OR5+rjd8+"
    "enlxl4tPORRXnas1x3/qgz9/evPm42+3H97evHz98PHu3c2HT+9u3t2+5Rs/vdAfeXP/7u7m07v7t7f/+Gnp87/evvnEh+3J"
    "FB9cjTnonoM39eLTH+5++XD3eHP7eGPPt+2S86ZYZ7LLMVh+37s73sfjzcOLj3cffr17dXEv/GdjbHQ/LX3y8debV/cfH2/f"
    "vTz/1OhMytFFY7IedS5L33l65zmEUmOIxTkbgj798OK/7l4+3v96d/P27vHD/Utu4U7f+NdP716+vrt9vH3x5u6m/Rcv3r/g"
    "lh7eP96//fT25u1D+87tu5v2DrizXz/evP9w96v+xL/f/vu3laEnevPb3f0vr88v3Hlbs7e2VOOi1ZL69smPn96/f/jwePPi"
    "7s3Dbzf2LugLzj5ZYw/vb149vL29f8dq/o//+en8f3DPrx7evL31Ny9f/qvu+s3j65vXuiD3e/f+4eVrPu5Oqcboak42Z2Mz"
    "b2LrtrRqli/w88PDq5vbd6/0JlidS1cqJqUastVSccGaeHkpo38UtYGyPuO1o+zqpe7fvfr0Ua/l9s14lXCK1moZtu1TfE3d"
    "RbIrLuhVp+Jirs4vX0Rr6eXfb+5efep+QfbRplhLzVbPK4x/PPA0fTRFy7qsP6yXH+7f3rWn9eb2t6VH5UzUqtR1Si0l2P5C"
    "eo7a6ckkbaJqYly40P27B+39t28f3t28/HD7mx7Vf19cQ+/WFW1UF/SCQ63e5+4iUS+k6p3pN9dkTFr9Nb9ok34cf4U9eWMJ"
    "Yvpfrwdj+7+v4GpMNFn/avSylv/+z4pAb2+1dP37N58+XlygnIKxXrG0RC0bxdX+AiloZYfqtaBKLCmvPCb928PLDzfvX/38"
    "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"
)


@dataclass(frozen=True)
class GenericFamilyPowerFamilyPenaltyNoL2RawSubsetOptimumSummary:
    """Summary for one exact subset-fit no-L2 raw optimum deployment."""

    subset_size: int
    run_id: int
    run_name: str
    policy: str
    objective_metric: str
    variant_name: str
    tuning_method: str
    predicted_optimum_value: float
    subset_best_observed_run_name: str
    subset_best_observed_bpb: float
    fullswarm_chosen_run_name: str
    fullswarm_chosen_value: float
    fullswarm_regret_at_1: float
    nearest_observed_run_name: str
    nearest_observed_value: float
    nearest_observed_tv_distance: float
    optimum_move_mean_phase_tv_vs_prev: float | None
    tuning_objective: float
    tuning_cv_rmse: float
    tuning_cv_regret_at_1: float
    tuning_cv_foldmean_regret_at_1: float
    tuning_lower_tail_optimism: float
    tuning_cv_depopt_best8: float
    tuning_cv_rawopt_nearest_tv: float
    phase0_max_weight: float
    phase1_max_weight: float
    phase0_support_below_1e4: int
    phase1_support_below_1e4: int
    phase0_top_domains: list[dict[str, float | str]]
    phase1_top_domains: list[dict[str, float | str]]
    family_shares: dict[str, float]
    phase_weights: dict[str, dict[str, float]]


def genericfamily_power_family_penalty_no_l2_raw_subset_optimum_run_id(subset_size: int) -> int:
    """Return the canonical run id for one exact subset-fit no-L2 raw optimum."""
    if subset_size not in GENERICFAMILY_POWER_FAMILY_PENALTY_NO_L2_RAW_SUBSET_OPTIMA_SUBSET_SIZES:
        raise ValueError(f"Unsupported subset size: {subset_size}")
    return (
        GENERICFAMILY_POWER_FAMILY_PENALTY_NO_L2_RAW_SUBSET_OPTIMA_BASE_RUN_ID
        + GENERICFAMILY_POWER_FAMILY_PENALTY_NO_L2_RAW_SUBSET_OPTIMA_SUBSET_SIZES.index(subset_size)
    )


def genericfamily_power_family_penalty_no_l2_raw_subset_optimum_run_name(subset_size: int) -> str:
    """Return the canonical run name for one exact subset-fit no-L2 raw optimum."""
    return f"baseline_genericfamily_power_family_penalty_no_l2_raw_optimum_k{subset_size:03d}_uncheatable_bpb"


def _weights_from_phase_weights(
    phase_weights: dict[str, dict[str, float]],
    domain_names: list[str],
) -> np.ndarray:
    return np.asarray(
        [
            [float(phase_weights["phase_0"][domain_name]) for domain_name in domain_names],
            [float(phase_weights["phase_1"][domain_name]) for domain_name in domain_names],
        ],
        dtype=float,
    )


def _embedded_rows() -> list[dict[str, Any]]:
    if not _EMBEDDED_ROWS_GZIP_BASE64:
        raise ValueError("Missing embedded no-L2 subset summaries")
    payload = gzip.decompress(base64.b64decode(_EMBEDDED_ROWS_GZIP_BASE64))
    return json.loads(payload.decode("utf-8"))


def _load_rows() -> list[dict[str, Any]]:
    if SUMMARY_JSON_PATH.exists():
        payload = json.loads(SUMMARY_JSON_PATH.read_text())
        rows = payload.get("rows")
        if isinstance(rows, list):
            return rows
    return _embedded_rows()


@cache
def _raw_subset_rows_by_size() -> dict[int, dict[str, Any]]:
    rows = _load_rows()
    rows_by_size = {int(row["subset_size"]): row for row in rows}
    missing = set(GENERICFAMILY_POWER_FAMILY_PENALTY_NO_L2_RAW_SUBSET_OPTIMA_SUBSET_SIZES).difference(rows_by_size)
    if missing:
        raise ValueError(f"Missing embedded/disk no-L2 subset summaries for sizes: {sorted(missing)}")
    return rows_by_size


def _summary_to_dict(summary: GenericFamilyPowerFamilyPenaltyNoL2RawSubsetOptimumSummary) -> dict[str, Any]:
    return {
        "subset_size": summary.subset_size,
        "run_id": summary.run_id,
        "run_name": summary.run_name,
        "policy": summary.policy,
        "objective_metric": summary.objective_metric,
        "variant_name": summary.variant_name,
        "tuning_method": summary.tuning_method,
        "predicted_optimum_value": summary.predicted_optimum_value,
        "subset_best_observed_run_name": summary.subset_best_observed_run_name,
        "subset_best_observed_bpb": summary.subset_best_observed_bpb,
        "fullswarm_chosen_run_name": summary.fullswarm_chosen_run_name,
        "fullswarm_chosen_value": summary.fullswarm_chosen_value,
        "fullswarm_regret_at_1": summary.fullswarm_regret_at_1,
        "nearest_observed_run_name": summary.nearest_observed_run_name,
        "nearest_observed_value": summary.nearest_observed_value,
        "nearest_observed_tv_distance": summary.nearest_observed_tv_distance,
        "optimum_move_mean_phase_tv_vs_prev": summary.optimum_move_mean_phase_tv_vs_prev,
        "tuning_objective": summary.tuning_objective,
        "tuning_cv_rmse": summary.tuning_cv_rmse,
        "tuning_cv_regret_at_1": summary.tuning_cv_regret_at_1,
        "tuning_cv_foldmean_regret_at_1": summary.tuning_cv_foldmean_regret_at_1,
        "tuning_lower_tail_optimism": summary.tuning_lower_tail_optimism,
        "tuning_cv_depopt_best8": summary.tuning_cv_depopt_best8,
        "tuning_cv_rawopt_nearest_tv": summary.tuning_cv_rawopt_nearest_tv,
        "phase0_max_weight": summary.phase0_max_weight,
        "phase1_max_weight": summary.phase1_max_weight,
        "phase0_support_below_1e4": summary.phase0_support_below_1e4,
        "phase1_support_below_1e4": summary.phase1_support_below_1e4,
        "phase0_top_domains": summary.phase0_top_domains,
        "phase1_top_domains": summary.phase1_top_domains,
        "family_shares": summary.family_shares,
        "phase_weights": summary.phase_weights,
    }


@cache
def genericfamily_power_family_penalty_no_l2_raw_subset_optima_summaries(
    subset_sizes: tuple[int, ...] = GENERICFAMILY_POWER_FAMILY_PENALTY_NO_L2_RAW_SUBSET_OPTIMA_SUBSET_SIZES,
) -> tuple[GenericFamilyPowerFamilyPenaltyNoL2RawSubsetOptimumSummary, ...]:
    """Return exact subset-fit raw-optimum summaries for the no-L2 GRP."""
    packet = load_generic_family_packet(target=OBJECTIVE_METRIC)
    rows_by_size = _raw_subset_rows_by_size()
    summaries: list[GenericFamilyPowerFamilyPenaltyNoL2RawSubsetOptimumSummary] = []
    for subset_size in subset_sizes:
        row = rows_by_size[subset_size]
        phase_weights = row["phase_weights"]
        weights = _weights_from_phase_weights(phase_weights, packet.base.domain_names)
        summaries.append(
            GenericFamilyPowerFamilyPenaltyNoL2RawSubsetOptimumSummary(
                subset_size=subset_size,
                run_id=genericfamily_power_family_penalty_no_l2_raw_subset_optimum_run_id(subset_size),
                run_name=genericfamily_power_family_penalty_no_l2_raw_subset_optimum_run_name(subset_size),
                policy=GENERICFAMILY_POWER_FAMILY_PENALTY_NO_L2_RAW_SUBSET_OPTIMA_POLICY,
                objective_metric=OBJECTIVE_METRIC,
                variant_name=GENERICFAMILY_POWER_FAMILY_PENALTY_NO_L2_RAW_SUBSET_OPTIMA_VARIANT,
                tuning_method=GENERICFAMILY_POWER_FAMILY_PENALTY_NO_L2_RAW_SUBSET_OPTIMA_TUNING_METHOD,
                predicted_optimum_value=float(row["predicted_optimum_value"]),
                subset_best_observed_run_name=str(row["subset_best_observed_run_name"]),
                subset_best_observed_bpb=float(row["subset_best_observed_bpb"]),
                fullswarm_chosen_run_name=str(row["fullswarm_chosen_run_name"]),
                fullswarm_chosen_value=float(row["fullswarm_chosen_value"]),
                fullswarm_regret_at_1=float(row["fullswarm_regret_at_1"]),
                nearest_observed_run_name=str(row["nearest_observed_run_name"]),
                nearest_observed_value=float(row["nearest_observed_value"]),
                nearest_observed_tv_distance=float(row["nearest_observed_tv_distance"]),
                optimum_move_mean_phase_tv_vs_prev=(
                    None
                    if row["optimum_move_mean_phase_tv_vs_prev"] is None
                    else float(row["optimum_move_mean_phase_tv_vs_prev"])
                ),
                tuning_objective=float(row["tuning_objective"]),
                tuning_cv_rmse=float(row["tuning_cv_rmse"]),
                tuning_cv_regret_at_1=float(row["tuning_cv_regret_at_1"]),
                tuning_cv_foldmean_regret_at_1=float(row["tuning_cv_foldmean_regret_at_1"]),
                tuning_lower_tail_optimism=float(row["tuning_lower_tail_optimism"]),
                tuning_cv_depopt_best8=float(row["tuning_cv_depopt_best8"]),
                tuning_cv_rawopt_nearest_tv=float(row["tuning_cv_rawopt_nearest_tv"]),
                phase0_max_weight=float(np.max(weights[0])),
                phase1_max_weight=float(np.max(weights[1])),
                phase0_support_below_1e4=int(row["phase0_support_below_1e4"]),
                phase1_support_below_1e4=int(row["phase1_support_below_1e4"]),
                phase0_top_domains=_top_domains(packet.base.domain_names, weights[0], weights[0] * packet.base.c0),
                phase1_top_domains=_top_domains(packet.base.domain_names, weights[1], weights[1] * packet.base.c1),
                family_shares=family_shares(packet, weights),
                phase_weights=_phase_weights_from_array(packet.base.domain_names, weights),
            )
        )
    return tuple(summaries)


def genericfamily_power_family_penalty_no_l2_raw_subset_optima_summaries_json(
    subset_sizes: tuple[int, ...] = GENERICFAMILY_POWER_FAMILY_PENALTY_NO_L2_RAW_SUBSET_OPTIMA_SUBSET_SIZES,
) -> str:
    """Return the exact subset-fit no-L2 raw-optimum summaries as JSON."""
    return json.dumps(
        [
            _summary_to_dict(summary)
            for summary in genericfamily_power_family_penalty_no_l2_raw_subset_optima_summaries(subset_sizes)
        ],
        indent=2,
        sort_keys=True,
    )


def genericfamily_power_family_penalty_no_l2_raw_subset_optima_summaries_frame(
    subset_sizes: tuple[int, ...] = GENERICFAMILY_POWER_FAMILY_PENALTY_NO_L2_RAW_SUBSET_OPTIMA_SUBSET_SIZES,
) -> pd.DataFrame:
    """Return a flat summary frame for the exact subset-fit no-L2 raw-optimum sweep."""
    return pd.DataFrame(
        [
            {
                "subset_size": summary.subset_size,
                "run_id": summary.run_id,
                "run_name": summary.run_name,
                "policy": summary.policy,
                "variant_name": summary.variant_name,
                "tuning_method": summary.tuning_method,
                "predicted_optimum_value": summary.predicted_optimum_value,
                "subset_best_observed_run_name": summary.subset_best_observed_run_name,
                "subset_best_observed_bpb": summary.subset_best_observed_bpb,
                "fullswarm_chosen_run_name": summary.fullswarm_chosen_run_name,
                "fullswarm_chosen_value": summary.fullswarm_chosen_value,
                "fullswarm_regret_at_1": summary.fullswarm_regret_at_1,
                "nearest_observed_run_name": summary.nearest_observed_run_name,
                "nearest_observed_value": summary.nearest_observed_value,
                "nearest_observed_tv_distance": summary.nearest_observed_tv_distance,
                "optimum_move_mean_phase_tv_vs_prev": summary.optimum_move_mean_phase_tv_vs_prev,
                "tuning_objective": summary.tuning_objective,
                "tuning_cv_rmse": summary.tuning_cv_rmse,
                "tuning_cv_regret_at_1": summary.tuning_cv_regret_at_1,
                "tuning_cv_foldmean_regret_at_1": summary.tuning_cv_foldmean_regret_at_1,
                "tuning_lower_tail_optimism": summary.tuning_lower_tail_optimism,
                "tuning_cv_depopt_best8": summary.tuning_cv_depopt_best8,
                "tuning_cv_rawopt_nearest_tv": summary.tuning_cv_rawopt_nearest_tv,
                "phase0_max_weight": summary.phase0_max_weight,
                "phase1_max_weight": summary.phase1_max_weight,
                "phase0_support_below_1e4": summary.phase0_support_below_1e4,
                "phase1_support_below_1e4": summary.phase1_support_below_1e4,
            }
            for summary in genericfamily_power_family_penalty_no_l2_raw_subset_optima_summaries(subset_sizes)
        ]
    )


def create_genericfamily_power_family_penalty_no_l2_raw_subset_optimum_weight_config(
    subset_size: int,
) -> WeightConfig:
    """Return the weight config for one exact subset-fit no-L2 raw optimum."""
    summary = next(
        summary
        for summary in genericfamily_power_family_penalty_no_l2_raw_subset_optima_summaries((subset_size,))
        if summary.subset_size == subset_size
    )
    return WeightConfig(run_id=summary.run_id, phase_weights=summary.phase_weights)


def create_genericfamily_power_family_penalty_no_l2_raw_subset_optima_weight_configs(
    subset_sizes: tuple[int, ...] = GENERICFAMILY_POWER_FAMILY_PENALTY_NO_L2_RAW_SUBSET_OPTIMA_SUBSET_SIZES,
) -> tuple[WeightConfig, ...]:
    """Return weight configs for the exact subset-fit no-L2 raw-optimum sweep."""
    return tuple(
        WeightConfig(run_id=summary.run_id, phase_weights=summary.phase_weights)
        for summary in genericfamily_power_family_penalty_no_l2_raw_subset_optima_summaries(subset_sizes)
    )
