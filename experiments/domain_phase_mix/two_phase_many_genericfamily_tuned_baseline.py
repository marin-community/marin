# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convex-hull deployment baseline from GenericFamily-RetainedTotal-Tuned."""

from __future__ import annotations

import json
from functools import cache
from typing import Any

import numpy as np

from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_followup import (
    family_shares,
    load_generic_family_packet,
)
from experiments.domain_phase_mix.nextgen.utils import normalize_phase_weights
from experiments.domain_phase_mix.static_batch_selection import average_phase_tv_distance
from experiments.domain_phase_mix.two_phase_many_ccglobalpremium_baselines import (
    ccglobalpremium_retainedtotal_summary,
)
from experiments.domain_phase_mix.two_phase_many_ccpairtotal_baseline import (
    ccpairtotal_retainedtotal_summary,
)

GENERICFAMILY_TUNED_SOURCE_EXPERIMENT = "pinlin_calvin_xu/data_mixture/ngd3dm2_genericfamily_tuned_uncheatable_bpb"
GENERICFAMILY_TUNED_RUN_ID = 254
GENERICFAMILY_TUNED_RUN_NAME = "baseline_genericfamily_retainedtotal_tuned_uncheatable_bpb"
GENERICFAMILY_TUNED_PREDICTED_VALUE = 1.0436372226731572
GENERICFAMILY_TUNED_ANCHOR_COEFFICIENTS = {
    "best_observed": 0.02587662289245359,
    "validated_global": 0.9175270003197237,
    "validated_pair": 0.0015381110202398686,
    "baseline_proportional": 0.05505826576758283,
}


def _row_weights(frame, domain_names: list[str], row_idx: int) -> np.ndarray:
    row = frame.iloc[row_idx]
    return np.asarray(
        [
            [float(row[f"phase_0_{domain_name}"]) for domain_name in domain_names],
            [float(row[f"phase_1_{domain_name}"]) for domain_name in domain_names],
        ],
        dtype=float,
    )


def _summary_weights(summary: dict[str, object], domain_names: list[str]) -> np.ndarray:
    phase_weights = summary["phase_weights"]
    phase0 = np.asarray([float(phase_weights["phase_0"][domain_name]) for domain_name in domain_names], dtype=float)
    phase1 = np.asarray([float(phase_weights["phase_1"][domain_name]) for domain_name in domain_names], dtype=float)
    return np.stack([phase0, phase1], axis=0)


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
def genericfamily_tuned_summary() -> dict[str, Any]:
    """Return the convex-hull deployment summary for GenericFamily-RetainedTotal-Tuned."""
    packet = load_generic_family_packet()
    frame = packet.base.frame
    best_idx = int(np.argmin(packet.base.y))
    best_weights = _row_weights(frame, packet.base.domain_names, best_idx)
    proportional_idx = int(frame.index[frame["run_name"] == "baseline_proportional"][0])
    proportional_weights = _row_weights(frame, packet.base.domain_names, proportional_idx)
    validated_global_weights = _summary_weights(ccglobalpremium_retainedtotal_summary(), packet.base.domain_names)
    validated_pair_weights = _summary_weights(ccpairtotal_retainedtotal_summary(), packet.base.domain_names)

    combined = (
        GENERICFAMILY_TUNED_ANCHOR_COEFFICIENTS["best_observed"] * best_weights
        + GENERICFAMILY_TUNED_ANCHOR_COEFFICIENTS["validated_global"] * validated_global_weights
        + GENERICFAMILY_TUNED_ANCHOR_COEFFICIENTS["validated_pair"] * validated_pair_weights
        + GENERICFAMILY_TUNED_ANCHOR_COEFFICIENTS["baseline_proportional"] * proportional_weights
    )
    distances = average_phase_tv_distance(packet.base.w, combined[None, :, :])
    nearest_idx = int(np.argmin(distances))

    return {
        "run_name": GENERICFAMILY_TUNED_RUN_NAME,
        "model": "GenericFamily-RetainedTotal-Tuned",
        "objective_metric": packet.base.frame.columns[packet.base.frame.columns.get_loc("eval/uncheatable_eval/bpb")],
        "predicted_optimum_value": GENERICFAMILY_TUNED_PREDICTED_VALUE,
        "anchor_coefficients": dict(GENERICFAMILY_TUNED_ANCHOR_COEFFICIENTS),
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


def genericfamily_tuned_summary_json() -> str:
    """Return the GenericFamily deployment summary as JSON."""
    return json.dumps(genericfamily_tuned_summary(), indent=2, sort_keys=True)


def create_genericfamily_tuned_weight_config(
    run_id: int = GENERICFAMILY_TUNED_RUN_ID,
) -> WeightConfig:
    """Return the GenericFamily tuned deployment baseline weight config."""
    summary = genericfamily_tuned_summary()
    return WeightConfig(run_id=run_id, phase_weights=summary["phase_weights"])
