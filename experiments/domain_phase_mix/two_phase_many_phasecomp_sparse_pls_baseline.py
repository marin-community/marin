# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Predicted baseline from the phase-composition sparse PLS surrogate."""

from __future__ import annotations

import json
from functools import cache
from typing import Any

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.phase_composition_sparse_pls import (
    PhaseCompositionSparsePLSSurrogate,
    load_phase_composition_packet,
    optimize_phase_composition_sparse_pls_model,
    reproduction_cv_summary,
)
from experiments.domain_phase_mix.nextgen.utils import normalize_phase_weights

PHASECOMP_SPARSE_PLS_SOURCE_EXPERIMENT = "pinlin_calvin_xu/data_mixture/ngd3dm2_phasecomp_sparse_pls_uncheatable_bpb"
PHASECOMP_SPARSE_PLS_RUN_ID = 257
PHASECOMP_SPARSE_PLS_RUN_NAME = "baseline_phasecomp_sparse_pls_uncheatable_bpb"


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
def phasecomp_sparse_pls_summary() -> dict[str, Any]:
    """Return the optimum summary for the phase-composition sparse PLS model."""
    data = load_phase_composition_packet()
    _, model = reproduction_cv_summary(data)
    if not isinstance(model, PhaseCompositionSparsePLSSurrogate):
        raise TypeError(f"Expected PhaseCompositionSparsePLSSurrogate, got {type(model)!r}")
    result, phase0, phase1 = optimize_phase_composition_sparse_pls_model(data, model, seed=0)
    optimum = np.stack([phase0, phase1], axis=0)
    deltas = data.w - optimum[None, :, :]
    distances = 0.5 * np.abs(deltas).sum(axis=2).mean(axis=1)
    nearest_idx = int(np.argmin(distances))
    best_idx = int(np.argmin(data.y))

    return {
        "run_name": PHASECOMP_SPARSE_PLS_RUN_NAME,
        "model": "Phase Composition Sparse PLS",
        "objective_metric": "eval/uncheatable_eval/bpb",
        "predicted_optimum_value": float(result.fun),
        "observed_best_run_name": str(data.frame.iloc[best_idx][data.name_col]),
        "observed_best_value": float(data.y[best_idx]),
        "gap_below_observed_best": float(result.fun - float(data.y[best_idx])),
        "nearest_observed_run_name": str(data.frame.iloc[nearest_idx][data.name_col]),
        "nearest_observed_value": float(data.y[nearest_idx]),
        "nearest_observed_tv_distance": float(distances[nearest_idx]),
        "phase0_max_weight": float(phase0.max()),
        "phase1_max_weight": float(phase1.max()),
        "phase0_entropy": _phase_entropy(phase0),
        "phase1_entropy": _phase_entropy(phase1),
        "phase0_support_below_1e4": int(np.sum(phase0 < 1e-4)),
        "phase1_support_below_1e4": int(np.sum(phase1 < 1e-4)),
        "phase0_support_below_1e6": int(np.sum(phase0 < 1e-6)),
        "phase1_support_below_1e6": int(np.sum(phase1 < 1e-6)),
        "phase0_top_domains": _top_domains(data.domain_names, phase0, phase0 * data.c0),
        "phase1_top_domains": _top_domains(data.domain_names, phase1, phase1 * data.c1),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "phase_weights": _phase_weights_from_array(data.domain_names, optimum),
    }


def phasecomp_sparse_pls_summary_json() -> str:
    """Return the phase-composition sparse PLS summary as JSON."""
    return json.dumps(phasecomp_sparse_pls_summary(), indent=2, sort_keys=True)


def create_phasecomp_sparse_pls_weight_config(
    run_id: int = PHASECOMP_SPARSE_PLS_RUN_ID,
) -> WeightConfig:
    """Return the phase-composition sparse PLS optimum baseline weight config."""
    summary = phasecomp_sparse_pls_summary()
    return WeightConfig(run_id=run_id, phase_weights=summary["phase_weights"])
