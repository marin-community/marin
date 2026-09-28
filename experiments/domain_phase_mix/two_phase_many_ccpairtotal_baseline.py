# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CCPairTotal-RetainedTotal predicted baseline for many-domain Uncheatable BPB."""

from __future__ import annotations

import json
from functools import cache
from typing import Any

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.structured_epoch_family import (
    MANY_DOMAIN_TARGET,
    PENALTY_KIND_GROUP_LOG_THRESHOLD,
    PREMIUM_MODE_GLOBAL,
    SIGNAL_KIND_RETAINED_TOTAL,
    evaluate_cc_pairtotal_model,
    load_two_phase_many_packet,
    optimize_cc_pairtotal_model,
)
from experiments.domain_phase_mix.nextgen.utils import normalize_phase_weights

CCPAIRTOTAL_RETAINEDTOTAL_SOURCE_EXPERIMENT = (
    "pinlin_calvin_xu/data_mixture/ngd3dm2_ccpairtotal_retainedtotal_uncheatable_bpb"
)
CCPAIRTOTAL_RETAINEDTOTAL_RUN_ID = 253
CCPAIRTOTAL_RETAINEDTOTAL_RUN_NAME = "baseline_ccpairtotal_retainedtotal_uncheatable_bpb"
CCPAIRTOTAL_RETAINEDTOTAL_MODEL_NAME = "CCPairTotal-RetainedTotal"


def _phase_entropy(weights) -> float:
    clipped = np.clip(np.asarray(weights, dtype=float), 1e-12, 1.0)
    return float(-np.sum(clipped * np.log(clipped)))


def _top_domains(domain_names: list[str], weights, epochs, *, top_k: int = 10) -> list[dict[str, float | str]]:
    frame = pd.DataFrame({"domain": domain_names, "weight": weights, "epochs": epochs})
    return frame.sort_values(["weight", "epochs"], ascending=False).head(top_k).to_dict(orient="records")


def _phase_weights_from_optimum(domain_names: list[str], phase0, phase1) -> dict[str, dict[str, float]]:
    return normalize_phase_weights(
        {
            "phase_0": {domain_name: float(weight) for domain_name, weight in zip(domain_names, phase0, strict=True)},
            "phase_1": {domain_name: float(weight) for domain_name, weight in zip(domain_names, phase1, strict=True)},
        }
    )


def _ccpairtotal_retainedtotal_params() -> dict[str, float | str]:
    return {
        "signal_kind": SIGNAL_KIND_RETAINED_TOTAL,
        "group_signal_kind": "log_after_sum",
        "premium_mode": PREMIUM_MODE_GLOBAL,
        "diversity_mode": "none",
        "alpha": 5.693767311270728,
        "eta": 6.323564464532408,
        "lam": 0.004606280004722357,
        "tau": 1.3976070420563144,
        "reg": 2.1562923313580245e-06,
        "pen_kind": PENALTY_KIND_GROUP_LOG_THRESHOLD,
    }


@cache
def ccpairtotal_retainedtotal_summary() -> dict[str, Any]:
    """Return the CCPairTotal-RetainedTotal optimum summary."""
    data = load_two_phase_many_packet(target=MANY_DOMAIN_TARGET)
    row, model = evaluate_cc_pairtotal_model(
        data,
        CCPAIRTOTAL_RETAINEDTOTAL_MODEL_NAME,
        _ccpairtotal_retainedtotal_params(),
    )
    result, phase0, phase1 = optimize_cc_pairtotal_model(model, data, seed=0)
    optimum = np.stack([phase0, phase1], axis=0)
    observed_weights = data.w
    deltas = observed_weights - optimum[None, :, :]
    distances = 0.5 * np.abs(deltas).sum(axis=2).mean(axis=1)
    nearest_idx = int(distances.argmin())
    best_idx = int(data.y.argmin())
    support_phase0 = int(np.sum(phase0 < 1e-4))
    support_phase1 = int(np.sum(phase1 < 1e-4))
    support_phase0_strict = int(np.sum(phase0 < 1e-6))
    support_phase1_strict = int(np.sum(phase1 < 1e-6))

    return {
        "run_name": CCPAIRTOTAL_RETAINEDTOTAL_RUN_NAME,
        "model": CCPAIRTOTAL_RETAINEDTOTAL_MODEL_NAME,
        "objective_metric": MANY_DOMAIN_TARGET,
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
        "phase0_support_below_1e4": support_phase0,
        "phase1_support_below_1e4": support_phase1,
        "phase0_support_below_1e6": support_phase0_strict,
        "phase1_support_below_1e6": support_phase1_strict,
        "phase0_top_domains": _top_domains(data.domain_names, phase0, phase0 * data.c0),
        "phase1_top_domains": _top_domains(data.domain_names, phase1, phase1 * data.c1),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "cv_rmse": float(row["cv_rmse"]),
        "cv_regret_at_1": float(row["cv_regret_at_1"]),
        "cv_foldmean_regret_at_1": float(row["cv_foldmean_regret_at_1"]),
        "phase_weights": _phase_weights_from_optimum(data.domain_names, phase0, phase1),
    }


def ccpairtotal_retainedtotal_summary_json() -> str:
    """Return the CCPairTotal-RetainedTotal optimum summary as JSON."""
    return json.dumps(ccpairtotal_retainedtotal_summary(), indent=2, sort_keys=True)


def create_ccpairtotal_retainedtotal_weight_config(
    run_id: int = CCPAIRTOTAL_RETAINEDTOTAL_RUN_ID,
) -> WeightConfig:
    """Return the CCPairTotal-RetainedTotal predicted baseline weight config."""
    summary = ccpairtotal_retainedtotal_summary()
    return WeightConfig(run_id=run_id, phase_weights=summary["phase_weights"])
