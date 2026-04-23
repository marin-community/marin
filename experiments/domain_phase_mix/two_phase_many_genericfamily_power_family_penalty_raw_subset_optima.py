# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact subset-fit raw-optimum deployments for the power-family-penalty GRP variant."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import gzip
import json
from functools import cache
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
    GENERICFAMILY_RETUNED_SUBSET_OPTIMA_ALL_SUBSET_SIZES,
    _top_domains,
)

GENERICFAMILY_POWER_FAMILY_PENALTY_RAW_SUBSET_OPTIMA_SOURCE_EXPERIMENT = (
    "pinlin_calvin_xu/data_mixture/" "ngd3dm2_genericfamily_power_family_penalty_raw_subset_optima_rep_uncheatable_bpb"
)
GENERICFAMILY_POWER_FAMILY_PENALTY_RAW_SUBSET_OPTIMA_BASE_RUN_ID = 440
GENERICFAMILY_POWER_FAMILY_PENALTY_RAW_SUBSET_OPTIMA_REPRESENTATIVE_SUBSET_SIZES = (
    20,
    40,
    60,
    80,
    100,
    140,
    180,
    220,
)
GENERICFAMILY_POWER_FAMILY_PENALTY_RAW_SUBSET_OPTIMA_POLICY = "feature_bayes_linear_power_family_penalty_raw_optimum"
GENERICFAMILY_POWER_FAMILY_PENALTY_RAW_SUBSET_OPTIMA_VARIANT = "power_family_penalty"
GENERICFAMILY_POWER_FAMILY_PENALTY_RAW_SUBSET_OPTIMA_TUNING_METHOD = "Powell"
SUMMARY_JSON_PATH = (
    Path(__file__).resolve().parent
    / "exploratory"
    / "two_phase_many"
    / "two_phase_many_grp_power_family_penalty_raw_summary.json"
)
_EMBEDDED_ROWS_GZIP_BASE64 = (
    "H4sIAKdH2mkC/+Wd6XIcx5WF34W/NT25L3qViQkECLZJ2ARAAyBljUPvPufLxtLd1ZWZpbEVYY8jLCrEqq6qXO4599wl/+vvH65v"
    "nr9ff736cf319tP18/7T1cdvHz/8fP/969efPvxJ/3z65frx7urmy8PT/v7q8fv91f313f7Dzx/4V2OcMR8uXKdf+66L7M6kkpOJ"
    "NrhYvS/h+NrH/efH/fPV9fOV/fCz2RlrY7I5Wp9iSjY5+9OH+/314/7p+erh49P+8YdebuUFFtc9/7j6dPv0fH1/s+e3s7MpOP26"
    "158pXbih88YP355v777fXd09/Nhf3e2v76++fbl+2vOMH09X3x73P17Hq/13c/X0/du3h8fnq4/7rw+/XNl9+PCziy9/ay/9rc8v"
    "f3v1y/7285fnpw8///3lPxj+9dPD17trf3X9+LdbPSvsiok151KLcTnFvP8Po096uejm5j+v9evX95+uPu2fbj/fX33RT3KXNTb4"
    "Yky1tVhrB3fp5fRiO69BM1aDEmKuMZ3fdPN4e7dvt329/uXlSUxl1Lxkb3xwvrhgy/o97TlhF60utql6U50P54/Zf/p+c/18+3Df"
    "bvuzZu3lWX5Xoys2mJiK99bFiTtfvsxlm100PsZkUvCLUdx/3d88Pz7c3948tXu/XD9+0rrdvz056ru8r/odk+uFd167/+WLbag+"
    "lZJzSrWW5YvfP+8fn69v7+/0b+8jm13xpeq9cyp667p+S3tM3Jmq72TSc/DGLQfoT7f37JL2ih+/P93e75+e3h+XStSoamN6n5xm"
    "cnhne2rS4NbqjO5xJeiNl099ePh0WG2397f3n98faHzJmpBYStTjks+hd9vrN6YQa5DpKNoWaTGTn2Uwjr7JmxgSY1hd0m1+eenL"
    "DKUUaqrJRq+vMPX8Z7/sr78+f3nbXkE/qVevNhRf89rVL4uvJCcDVyrvoOWzuFi26+Hx1/apn/cPnx+vv3359e0LrPU2ajPm5EzK"
    "OY7vbE91ujG5GE2x+p4cw+Kxt/efvj89P94KD14XudPesjIzWnXBaoA7t7x8mtazNc2AFlnQupj6r7dapdfP3992EstMl9eg67Xe"
    "rCzayvUvT9D68NEx45pGbxc77+nmds/CvLvWgDMQz/ubL/cPXx8+//puDLUqQ9ACrV6G1MQtv/GyPLTmUrBVnylIOZ1vbY19u9N/"
    "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"
    "L7QFzTGmus6Z24u2YAtdF/KmKgkkFXwSdF9ZrOjm5LO3hrLoTaTxcU4l2fjD3l6ZLpNk//kXROkd6kirPXpfAbcn2eodCYxNWGhp"
    "RVFNrmXAmoun+k6ri0yAZNzmfiCRjde6OxUW6VbG3M6roqLBkLPguzUSrcaQs54TyYbEkY8+7zTzK7e6MqK8HBMd1ook2M25OYqh"
    "0KWt9uly5MA4z9FPNJUvx5n+l0okMjV0rCrRcW96LXE5fZWINNFXgioj2YvkQZqFtWTLIFqzXuFYq6MNfyV+bIsbkGXfztOztgm0"
    "x4lHl3Sv0hp7aBlTplTDWjfcElurajqYBBN9J/ELl4jTxEzri+IO7WFGNY50i88OO9dU1Cnc1h4MtYleIdJKKPhxT/vEcSaHliac"
    "Vx8mZS/K5wN9gziaN1UTh6296IssXkBappZb2oTdssJsJEvNTTzbjZ0Sx4K6UlKT9MvysJ2J5C+EWPnf1KqIup2lxYwOpCnUztDa"
    "C9af/GTittwJGQF6xcvs+zIqcYxo7ZxRSmdAWUm/LW+bxtp0W/Otls6U2Yb2JNYg6nhiRTn5aezGocvIUp5+7TnkieQviIUnCa8d"
    "aDKCbjomUGtGM6YSy6riRfsHA/3QDHM+zgi6LUWdoVV7hQQ4rAevqKn11JJkziHe1KDAcRYCoZDWupCjJrZht3am5qS2Huex0blx"
    "6lehnjZzgjEBvdpXvNC69GGuGb5oZ9C7opDSCxth1do0Oo5GRoxQEAWUMbu0vaG95j0UTpgm57jYbZ05647CnkT6eKB9YRomf+Fg"
    "ISJpvbEn4gp8wyOR0ai3LsdHX5+XODZnlCRw71tp0kDtskJATp53nPsaQu4f3EjNnW+HNWk7JdftaF+hwIR27KFCbVTgmNphJ5yW"
    "6Ohfu97RXoaZSlnO6eKMRTvsUZDItaJIlnhbCF0EpzRdUM9x82JTqawAuOY402VWa94SaO5pXZVWQaxg4u+HU2tHUhdeRnP7cmo5"
    "Sl2pa6ErzEldB4XjROpyri910ZmNgnaqmYlZT0pd+Hh0jszon3JI6zjvywfOLCFUm2LM6ff08qIjkYwtqbzFzLSvBwZzIN2roF+u"
    "KVstxB8I+DmA9rf//l9GgDoA15cAAA=="
)


@dataclass(frozen=True)
class GenericFamilyPowerFamilyPenaltyRawSubsetOptimumSummary:
    """Summary for one exact subset-fit raw-optimum deployment."""

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


def genericfamily_power_family_penalty_raw_subset_optimum_run_id(subset_size: int) -> int:
    """Return the canonical run id for one exact subset-fit raw optimum."""
    if subset_size not in GENERICFAMILY_RETUNED_SUBSET_OPTIMA_ALL_SUBSET_SIZES:
        raise ValueError(f"Unsupported subset size: {subset_size}")
    return (
        GENERICFAMILY_POWER_FAMILY_PENALTY_RAW_SUBSET_OPTIMA_BASE_RUN_ID
        + GENERICFAMILY_RETUNED_SUBSET_OPTIMA_ALL_SUBSET_SIZES.index(subset_size)
    )


def genericfamily_power_family_penalty_raw_subset_optimum_run_name(subset_size: int) -> str:
    """Return the canonical run name for one exact subset-fit raw optimum."""
    return f"baseline_genericfamily_power_family_penalty_raw_optimum_k{subset_size:03d}_uncheatable_bpb"


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
    representative_sizes = set(GENERICFAMILY_POWER_FAMILY_PENALTY_RAW_SUBSET_OPTIMA_REPRESENTATIVE_SUBSET_SIZES)
    filtered = [row for row in rows if int(row["subset_size"]) in representative_sizes]
    rows_by_size = {int(row["subset_size"]): row for row in filtered}
    missing = representative_sizes.difference(rows_by_size)
    if missing:
        raise ValueError(f"Missing embedded/disk subset summaries for sizes: {sorted(missing)}")
    return rows_by_size


def _summary_to_dict(summary: GenericFamilyPowerFamilyPenaltyRawSubsetOptimumSummary) -> dict[str, Any]:
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
def genericfamily_power_family_penalty_raw_subset_optima_summaries(
    subset_sizes: tuple[int, ...] = GENERICFAMILY_POWER_FAMILY_PENALTY_RAW_SUBSET_OPTIMA_REPRESENTATIVE_SUBSET_SIZES,
) -> tuple[GenericFamilyPowerFamilyPenaltyRawSubsetOptimumSummary, ...]:
    """Return exact subset-fit raw-optimum summaries for the power-family-penalty GRP."""
    packet = load_generic_family_packet(target=OBJECTIVE_METRIC)
    rows_by_size = _raw_subset_rows_by_size()
    summaries: list[GenericFamilyPowerFamilyPenaltyRawSubsetOptimumSummary] = []
    for subset_size in subset_sizes:
        row = rows_by_size[subset_size]
        phase_weights = row["phase_weights"]
        weights = _weights_from_phase_weights(phase_weights, packet.base.domain_names)
        summaries.append(
            GenericFamilyPowerFamilyPenaltyRawSubsetOptimumSummary(
                subset_size=subset_size,
                run_id=genericfamily_power_family_penalty_raw_subset_optimum_run_id(subset_size),
                run_name=genericfamily_power_family_penalty_raw_subset_optimum_run_name(subset_size),
                policy=GENERICFAMILY_POWER_FAMILY_PENALTY_RAW_SUBSET_OPTIMA_POLICY,
                objective_metric=OBJECTIVE_METRIC,
                variant_name=GENERICFAMILY_POWER_FAMILY_PENALTY_RAW_SUBSET_OPTIMA_VARIANT,
                tuning_method=GENERICFAMILY_POWER_FAMILY_PENALTY_RAW_SUBSET_OPTIMA_TUNING_METHOD,
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
                phase_weights=phase_weights,
            )
        )
    return tuple(summaries)


def genericfamily_power_family_penalty_raw_subset_optima_summaries_json(
    subset_sizes: tuple[int, ...] = GENERICFAMILY_POWER_FAMILY_PENALTY_RAW_SUBSET_OPTIMA_REPRESENTATIVE_SUBSET_SIZES,
) -> str:
    """Return the exact subset-fit raw-optimum summaries as JSON."""
    return json.dumps(
        [
            _summary_to_dict(summary)
            for summary in genericfamily_power_family_penalty_raw_subset_optima_summaries(subset_sizes)
        ],
        indent=2,
        sort_keys=True,
    )


def genericfamily_power_family_penalty_raw_subset_optima_summaries_frame(
    subset_sizes: tuple[int, ...] = GENERICFAMILY_POWER_FAMILY_PENALTY_RAW_SUBSET_OPTIMA_REPRESENTATIVE_SUBSET_SIZES,
) -> pd.DataFrame:
    """Return a flat summary frame for the exact subset-fit raw-optimum sweep."""
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
            for summary in genericfamily_power_family_penalty_raw_subset_optima_summaries(subset_sizes)
        ]
    )


def create_genericfamily_power_family_penalty_raw_subset_optimum_weight_config(
    subset_size: int,
) -> WeightConfig:
    """Return the weight config for one exact subset-fit raw optimum."""
    summary = next(
        summary
        for summary in genericfamily_power_family_penalty_raw_subset_optima_summaries((subset_size,))
        if summary.subset_size == subset_size
    )
    return WeightConfig(run_id=summary.run_id, phase_weights=summary.phase_weights)


def create_genericfamily_power_family_penalty_raw_subset_optima_weight_configs(
    subset_sizes: tuple[int, ...] = GENERICFAMILY_POWER_FAMILY_PENALTY_RAW_SUBSET_OPTIMA_REPRESENTATIVE_SUBSET_SIZES,
) -> tuple[WeightConfig, ...]:
    """Return weight configs for the exact subset-fit raw-optimum sweep."""
    return tuple(
        WeightConfig(run_id=summary.run_id, phase_weights=summary.phase_weights)
        for summary in genericfamily_power_family_penalty_raw_subset_optima_summaries(subset_sizes)
    )
