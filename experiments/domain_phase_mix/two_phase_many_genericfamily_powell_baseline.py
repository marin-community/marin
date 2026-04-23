# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""All-data GRP optimum after retuning nonlinear parameters with Powell."""

from __future__ import annotations

import json
from functools import cache
from typing import Any

import numpy as np

from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_followup import (
    family_shares,
    load_generic_family_packet,
    optimize_generic_family_model,
)
from experiments.domain_phase_mix.nextgen.utils import normalize_phase_weights
from experiments.domain_phase_mix.static_batch_selection import average_phase_tv_distance
from experiments.domain_phase_mix.two_phase_many_ccglobalpremium_baselines import (
    ccglobalpremium_retainedtotal_summary,
)
from experiments.domain_phase_mix.two_phase_many_ccpairtotal_baseline import (
    ccpairtotal_retainedtotal_summary,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_retuned_subset_optima import (
    VALIDATED_GLOBAL_BPB,
    VALIDATED_PAIR_BPB,
    _summary_weights,
    tune_genericfamily_subset_params,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_followup import (
    GenericFamilyRetainedTotalSurrogate,
)

GENERICFAMILY_POWELL_SOURCE_EXPERIMENT = "pinlin_calvin_xu/data_mixture/ngd3dm2_genericfamily_powell_uncheatable_bpb"
GENERICFAMILY_POWELL_RUN_ID = 351
GENERICFAMILY_POWELL_RUN_NAME = "baseline_genericfamily_powell_uncheatable_bpb"
GENERICFAMILY_POWELL_TUNING_METHOD = "Powell"


def _phase_entropy(weights: np.ndarray) -> float:
    clipped = np.clip(np.asarray(weights, dtype=float), 1e-12, 1.0)
    return float(-np.sum(clipped * np.log(clipped)))


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


@cache
def genericfamily_powell_summary() -> dict[str, Any]:
    """Return the all-data Powell-retuned GRP predicted optimum summary."""
    packet = load_generic_family_packet()
    valid_weights = np.stack(
        [
            _summary_weights(ccglobalpremium_retainedtotal_summary(), packet.base.domain_names),
            _summary_weights(ccpairtotal_retainedtotal_summary(), packet.base.domain_names),
        ],
        axis=0,
    )
    valid_y = np.asarray([VALIDATED_GLOBAL_BPB, VALIDATED_PAIR_BPB], dtype=float)

    tuning_metrics, _ = tune_genericfamily_subset_params(
        packet,
        valid_weights,
        valid_y,
        method=GENERICFAMILY_POWELL_TUNING_METHOD,
        seed=0,
    )
    tuned_params = {key: float(tuning_metrics[key]) for key in ("alpha", "eta", "lam", "tau", "reg", "beta")}
    model = GenericFamilyRetainedTotalSurrogate(packet, params=tuned_params).fit(packet.base.w, packet.base.y)
    full_predictions = model.predict(packet.base.w)
    chosen_idx = int(np.argmin(full_predictions))
    best_idx = int(np.argmin(packet.base.y))

    result, phase0, phase1 = optimize_generic_family_model(packet, model, seed=0)
    optimum = np.stack([phase0, phase1], axis=0)
    distances = average_phase_tv_distance(packet.base.w, optimum[None, :, :])
    nearest_idx = int(np.argmin(distances))

    return {
        "run_name": GENERICFAMILY_POWELL_RUN_NAME,
        "model": "GenericFamily-Powell-AllData",
        "objective_metric": packet.base.frame.columns[packet.base.frame.columns.get_loc("eval/uncheatable_eval/bpb")],
        "tuning_method": GENERICFAMILY_POWELL_TUNING_METHOD,
        "tuned_params": tuned_params,
        "tuning_objective": float(tuning_metrics["objective"]),
        "predicted_optimum_value": float(result.fun),
        "fullswarm_chosen_run_name": str(packet.base.frame.iloc[chosen_idx][packet.base.name_col]),
        "fullswarm_chosen_value": float(packet.base.y[chosen_idx]),
        "fullswarm_regret_at_1": float(packet.base.y[chosen_idx] - packet.base.y[best_idx]),
        "observed_best_run_name": str(packet.base.frame.iloc[best_idx][packet.base.name_col]),
        "observed_best_value": float(packet.base.y[best_idx]),
        "nearest_observed_run_name": str(packet.base.frame.iloc[nearest_idx][packet.base.name_col]),
        "nearest_observed_value": float(packet.base.y[nearest_idx]),
        "nearest_observed_tv_distance": float(distances[nearest_idx]),
        "phase0_max_weight": float(optimum[0].max()),
        "phase1_max_weight": float(optimum[1].max()),
        "phase0_entropy": _phase_entropy(optimum[0]),
        "phase1_entropy": _phase_entropy(optimum[1]),
        "phase0_support_below_1e4": int(np.sum(optimum[0] < 1e-4)),
        "phase1_support_below_1e4": int(np.sum(optimum[1] < 1e-4)),
        "family_shares": family_shares(packet, optimum),
        "phase_weights": _phase_weights_from_array(packet.base.domain_names, optimum),
    }


def genericfamily_powell_summary_json() -> str:
    """Return the all-data Powell-retuned summary as JSON."""
    return json.dumps(genericfamily_powell_summary(), indent=2, sort_keys=True)


def create_genericfamily_powell_weight_config(run_id: int = GENERICFAMILY_POWELL_RUN_ID) -> WeightConfig:
    """Return the weight config for the all-data Powell-retuned predicted optimum."""
    summary = genericfamily_powell_summary()
    return WeightConfig(run_id=run_id, phase_weights=summary["phase_weights"])
