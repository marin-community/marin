# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recovered convex-hull GRP deployment from broad-beta Powell tuning."""

from __future__ import annotations

import json
from functools import cache
from typing import Any

import numpy as np

from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_followup import (
    GenericFamilyRetainedTotalSurrogate,
    family_shares,
    load_generic_family_packet,
    optimize_generic_family_convex_hull,
)
from experiments.domain_phase_mix.static_batch_selection import average_phase_tv_distance
from experiments.domain_phase_mix.two_phase_many_ccglobalpremium_baselines import (
    ccglobalpremium_retainedtotal_summary,
)
from experiments.domain_phase_mix.two_phase_many_ccpairtotal_baseline import (
    ccpairtotal_retainedtotal_summary,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_retuned_subset_optima import (
    BROAD_BETA_GENERIC_FAMILY_PARAMS,
    VALIDATED_GLOBAL_BPB,
    VALIDATED_PAIR_BPB,
    tune_genericfamily_subset_params,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_tuned_baseline import (
    _phase_entropy,
    _phase_weights_from_array,
    _row_weights,
    _summary_weights,
)

GENERICFAMILY_RECOVERED_HULL_SOURCE_EXPERIMENT = (
    "pinlin_calvin_xu/data_mixture/ngd3dm2_genericfamily_recovered_hull_uncheatable_bpb"
)
GENERICFAMILY_RECOVERED_HULL_RUN_ID = 353
GENERICFAMILY_RECOVERED_HULL_RUN_NAME = "baseline_genericfamily_recovered_hull_uncheatable_bpb"
GENERICFAMILY_RECOVERED_HULL_TUNING_METHOD = "Powell"
GENERICFAMILY_RECOVERED_HULL_OBJECTIVE_NAME = "single_both"


@cache
def genericfamily_recovered_hull_summary() -> dict[str, Any]:
    """Return the recovered convex-hull GRP deployment summary."""
    packet = load_generic_family_packet()
    frame = packet.base.frame
    valid_global_weights = _summary_weights(ccglobalpremium_retainedtotal_summary(), packet.base.domain_names)
    valid_pair_weights = _summary_weights(ccpairtotal_retainedtotal_summary(), packet.base.domain_names)
    valid_weights = np.stack([valid_global_weights, valid_pair_weights], axis=0)
    valid_y = np.asarray([VALIDATED_GLOBAL_BPB, VALIDATED_PAIR_BPB], dtype=float)

    tuning_metrics, _ = tune_genericfamily_subset_params(
        packet,
        valid_weights,
        valid_y,
        method=GENERICFAMILY_RECOVERED_HULL_TUNING_METHOD,
        objective_name=GENERICFAMILY_RECOVERED_HULL_OBJECTIVE_NAME,
        start_params=BROAD_BETA_GENERIC_FAMILY_PARAMS,
        seed=0,
    )
    tuned_params = {key: float(tuning_metrics[key]) for key in ("alpha", "eta", "lam", "tau", "reg", "beta")}

    aug_w = np.concatenate([packet.base.w, valid_weights], axis=0)
    aug_y = np.concatenate([packet.base.y, valid_y], axis=0)
    deploy_model = GenericFamilyRetainedTotalSurrogate(packet, params=tuned_params).fit(aug_w, aug_y)

    best_idx = int(np.argmin(packet.base.y))
    best_weights = _row_weights(frame, packet.base.domain_names, best_idx)
    proportional_idx = int(frame.index[frame["run_name"] == "baseline_proportional"][0])
    proportional_weights = _row_weights(frame, packet.base.domain_names, proportional_idx)
    anchors = np.stack([best_weights, valid_global_weights, valid_pair_weights, proportional_weights], axis=0)
    predicted_value, anchor_coeffs, combined = optimize_generic_family_convex_hull(deploy_model, anchors)

    distances = average_phase_tv_distance(packet.base.w, combined[None, :, :])
    nearest_idx = int(np.argmin(distances))
    anchor_coefficients = {
        "best_observed": float(anchor_coeffs[0]),
        "validated_global": float(anchor_coeffs[1]),
        "validated_pair": float(anchor_coeffs[2]),
        "baseline_proportional": float(anchor_coeffs[3]),
    }

    return {
        "run_name": GENERICFAMILY_RECOVERED_HULL_RUN_NAME,
        "model": "GenericFamily-RecoveredHull",
        "objective_metric": packet.base.frame.columns[packet.base.frame.columns.get_loc("eval/uncheatable_eval/bpb")],
        "predicted_optimum_value": float(predicted_value),
        "anchor_coefficients": anchor_coefficients,
        "tuned_params": tuned_params,
        "tuning_method": GENERICFAMILY_RECOVERED_HULL_TUNING_METHOD,
        "tuning_objective_name": GENERICFAMILY_RECOVERED_HULL_OBJECTIVE_NAME,
        "tuning_objective": float(tuning_metrics["objective"]),
        "observed_best_run_name": str(frame.iloc[best_idx][packet.base.name_col]),
        "observed_best_value": float(packet.base.y[best_idx]),
        "nearest_observed_run_name": str(frame.iloc[nearest_idx][packet.base.name_col]),
        "nearest_observed_value": float(packet.base.y[nearest_idx]),
        "nearest_observed_tv_distance": float(distances[nearest_idx]),
        "phase0_max_weight": float(combined[0].max()),
        "phase1_max_weight": float(combined[1].max()),
        "phase0_entropy": _phase_entropy(combined[0]),
        "phase1_entropy": _phase_entropy(combined[1]),
        "phase0_support_below_1e4": int(np.sum(combined[0] < 1e-4)),
        "phase1_support_below_1e4": int(np.sum(combined[1] < 1e-4)),
        "phase0_support_below_1e6": int(np.sum(combined[0] < 1e-6)),
        "phase1_support_below_1e6": int(np.sum(combined[1] < 1e-6)),
        "family_shares": family_shares(packet, combined),
        "phase_weights": _phase_weights_from_array(packet.base.domain_names, combined),
    }


def genericfamily_recovered_hull_summary_json() -> str:
    """Return the recovered convex-hull deployment summary as JSON."""
    return json.dumps(genericfamily_recovered_hull_summary(), indent=2, sort_keys=True)


def create_genericfamily_recovered_hull_weight_config(
    run_id: int = GENERICFAMILY_RECOVERED_HULL_RUN_ID,
) -> WeightConfig:
    """Return the recovered convex-hull deployment baseline weight config."""
    summary = genericfamily_recovered_hull_summary()
    return WeightConfig(run_id=run_id, phase_weights=summary["phase_weights"])
