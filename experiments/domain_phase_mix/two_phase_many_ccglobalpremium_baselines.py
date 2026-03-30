# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CCGlobalPremium predicted baselines for many-domain Uncheatable BPB."""

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
    SIGNAL_KIND_THRESHOLD_TOTAL,
    evaluate_cc_model,
    load_two_phase_many_packet,
    optimize_cc_globalpremium_model,
)
from experiments.domain_phase_mix.nextgen.utils import normalize_phase_weights

CCGLOBALPREMIUM_SOURCE_EXPERIMENT = "pinlin_calvin_xu/data_mixture/ngd3dm2_ccglobalpremium_uncheatable_bpb"

CCGLOBALPREMIUM_THRESHOLD_RUN_ID = 251
CCGLOBALPREMIUM_THRESHOLD_RUN_NAME = "baseline_ccglobalpremium_threshold_uncheatable_bpb"
CCGLOBALPREMIUM_RETAINEDTOTAL_RUN_ID = 252
CCGLOBALPREMIUM_RETAINEDTOTAL_RUN_NAME = "baseline_ccglobalpremium_retainedtotal_uncheatable_bpb"


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


def _threshold_params() -> dict[str, float | str]:
    return {
        "signal_kind": SIGNAL_KIND_THRESHOLD_TOTAL,
        "alpha": 8.0,
        "eta": 5.0,
        "lam": 0.0,
        "sig_tau": 0.25,
        "premium_mode": PREMIUM_MODE_GLOBAL,
        "pen_kind": PENALTY_KIND_GROUP_LOG_THRESHOLD,
        "tau": 2.0,
        "reg": 0.01,
    }


def _retained_params() -> dict[str, float | str]:
    return {
        "signal_kind": SIGNAL_KIND_RETAINED_TOTAL,
        "alpha": 8.0,
        "eta": 3.0,
        "lam": 1.0,
        "sig_tau": 0.0,
        "premium_mode": PREMIUM_MODE_GLOBAL,
        "pen_kind": PENALTY_KIND_GROUP_LOG_THRESHOLD,
        "tau": 1.0,
        "reg": 0.01,
    }


def _summarize_optimum(
    *,
    data,
    model_name: str,
    run_name: str,
    params: dict[str, float | str],
) -> dict[str, Any]:
    row, model = evaluate_cc_model(data, model_name, params)
    result, phase0, phase1 = optimize_cc_globalpremium_model(model, data, seed=0)
    optimum = np.stack([phase0, phase1], axis=0)
    observed_weights = data.w
    deltas = observed_weights - optimum[None, :, :]
    distances = 0.5 * np.abs(deltas).sum(axis=2).mean(axis=1)
    nearest_idx = int(distances.argmin())
    best_idx = int(data.y.argmin())

    return {
        "run_name": run_name,
        "model": model_name,
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
        "phase0_top_domains": _top_domains(data.domain_names, phase0, phase0 * data.c0),
        "phase1_top_domains": _top_domains(data.domain_names, phase1, phase1 * data.c1),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "cv_rmse": float(row["cv_rmse"]),
        "cv_regret_at_1": float(row["cv_regret_at_1"]),
        "phase_weights": _phase_weights_from_optimum(data.domain_names, phase0, phase1),
    }


@cache
def _all_summaries() -> dict[str, dict[str, Any]]:
    data = load_two_phase_many_packet(target=MANY_DOMAIN_TARGET)
    return {
        CCGLOBALPREMIUM_THRESHOLD_RUN_NAME: _summarize_optimum(
            data=data,
            model_name="CCGlobalPremium-Threshold",
            run_name=CCGLOBALPREMIUM_THRESHOLD_RUN_NAME,
            params=_threshold_params(),
        ),
        CCGLOBALPREMIUM_RETAINEDTOTAL_RUN_NAME: _summarize_optimum(
            data=data,
            model_name="CCGlobalPremium-RetainedTotal",
            run_name=CCGLOBALPREMIUM_RETAINEDTOTAL_RUN_NAME,
            params=_retained_params(),
        ),
    }


def ccglobalpremium_threshold_summary() -> dict[str, Any]:
    """Return the Threshold optimum summary."""
    return dict(_all_summaries()[CCGLOBALPREMIUM_THRESHOLD_RUN_NAME])


def ccglobalpremium_retainedtotal_summary() -> dict[str, Any]:
    """Return the RetainedTotal optimum summary."""
    return dict(_all_summaries()[CCGLOBALPREMIUM_RETAINEDTOTAL_RUN_NAME])


def ccglobalpremium_summary_json() -> str:
    """Return both summaries as JSON."""
    return json.dumps(
        {
            "threshold": ccglobalpremium_threshold_summary(),
            "retainedtotal": ccglobalpremium_retainedtotal_summary(),
        },
        indent=2,
        sort_keys=True,
    )


def create_ccglobalpremium_threshold_weight_config(
    run_id: int = CCGLOBALPREMIUM_THRESHOLD_RUN_ID,
) -> WeightConfig:
    """Return the Threshold predicted baseline weight config."""
    summary = ccglobalpremium_threshold_summary()
    return WeightConfig(run_id=run_id, phase_weights=summary["phase_weights"])


def create_ccglobalpremium_retainedtotal_weight_config(
    run_id: int = CCGLOBALPREMIUM_RETAINEDTOTAL_RUN_ID,
) -> WeightConfig:
    """Return the RetainedTotal predicted baseline weight config."""
    summary = ccglobalpremium_retainedtotal_summary()
    return WeightConfig(run_id=run_id, phase_weights=summary["phase_weights"])
