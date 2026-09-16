# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Predicted baseline from the GRP no-overexposure-penalty ablation."""

from __future__ import annotations

import json
from functools import cache
from typing import Any

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_followup import (
    GENERIC_FAMILY_NAMES,
    MANY_DOMAIN_TARGET,
    TUNED_GENERIC_FAMILY_PARAMS,
    GenericFamilyRetainedTotalSurrogate,
    family_shares,
    load_generic_family_packet,
    optimize_generic_family_model,
)
from experiments.domain_phase_mix.nextgen.utils import normalize_phase_weights

GENERICFAMILY_NO_PENALTY_SOURCE_EXPERIMENT = (
    "pinlin_calvin_xu/data_mixture/ngd3dm2_genericfamily_no_penalty_uncheatable_bpb"
)
GENERICFAMILY_NO_PENALTY_RUN_ID = 258
GENERICFAMILY_NO_PENALTY_RUN_NAME = "baseline_genericfamily_no_penalty_uncheatable_bpb"


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


@cache
def genericfamily_no_penalty_summary() -> dict[str, Any]:
    """Return the optimum summary for the GRP no-overexposure-penalty ablation."""
    packet = load_generic_family_packet()
    model = GenericFamilyRetainedTotalSurrogate(
        packet,
        params=dict(TUNED_GENERIC_FAMILY_PARAMS),
        family_totals=GENERIC_FAMILY_NAMES,
        quality_discount=True,
        include_penalty=False,
    ).fit(packet.base.w, packet.base.y)
    result, phase0, phase1 = optimize_generic_family_model(packet, model, seed=0)
    optimum = np.stack([phase0, phase1], axis=0)
    deltas = packet.base.w - optimum[None, :, :]
    distances = 0.5 * np.abs(deltas).sum(axis=2).mean(axis=1)
    nearest_idx = int(np.argmin(distances))
    best_idx = int(np.argmin(packet.base.y))

    return {
        "run_name": GENERICFAMILY_NO_PENALTY_RUN_NAME,
        "model": "GRP w/o overexposure penalty",
        "objective_metric": MANY_DOMAIN_TARGET,
        "predicted_optimum_value": float(result.fun),
        "observed_best_run_name": str(packet.base.frame.iloc[best_idx][packet.base.name_col]),
        "observed_best_value": float(packet.base.y[best_idx]),
        "gap_below_observed_best": float(result.fun - float(packet.base.y[best_idx])),
        "nearest_observed_run_name": str(packet.base.frame.iloc[nearest_idx][packet.base.name_col]),
        "nearest_observed_value": float(packet.base.y[nearest_idx]),
        "nearest_observed_tv_distance": float(distances[nearest_idx]),
        "phase0_max_weight": float(phase0.max()),
        "phase1_max_weight": float(phase1.max()),
        "phase0_entropy": _phase_entropy(phase0),
        "phase1_entropy": _phase_entropy(phase1),
        "phase0_support_below_1e4": int(np.sum(phase0 < 1e-4)),
        "phase1_support_below_1e4": int(np.sum(phase1 < 1e-4)),
        "phase0_support_below_1e6": int(np.sum(phase0 < 1e-6)),
        "phase1_support_below_1e6": int(np.sum(phase1 < 1e-6)),
        "phase0_top_domains": _top_domains(packet.base.domain_names, phase0, phase0 * packet.base.c0),
        "phase1_top_domains": _top_domains(packet.base.domain_names, phase1, phase1 * packet.base.c1),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "family_shares": family_shares(packet, optimum),
        "params": dict(TUNED_GENERIC_FAMILY_PARAMS),
        "phase_weights": _phase_weights_from_array(packet.base.domain_names, optimum),
    }


def genericfamily_no_penalty_summary_json() -> str:
    """Return the no-overexposure-penalty baseline summary as JSON."""
    return json.dumps(genericfamily_no_penalty_summary(), indent=2, sort_keys=True)


def create_genericfamily_no_penalty_weight_config(
    run_id: int = GENERICFAMILY_NO_PENALTY_RUN_ID,
) -> WeightConfig:
    """Return the no-overexposure-penalty optimum baseline weight config."""
    summary = genericfamily_no_penalty_summary()
    return WeightConfig(run_id=run_id, phase_weights=summary["phase_weights"])
