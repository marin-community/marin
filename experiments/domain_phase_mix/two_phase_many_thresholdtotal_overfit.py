# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Frozen ThresholdTotal-Overfit optimum for Uncheatable BPB."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.nextgen.utils import normalize_phase_weights
from experiments.domain_phase_mix.static_batch_selection import average_phase_tv_distance
from experiments.domain_phase_mix.two_phase_many_observed_runs import TWO_PHASE_MANY_CSV_PATH

THRESHOLDTOTAL_OVERFIT_RUN_ID = 249
THRESHOLDTOTAL_OVERFIT_RUN_NAME = "baseline_thresholdtotal_overfit_uncheatable_bpb"
THRESHOLDTOTAL_OVERFIT_SOURCE_EXPERIMENT = "pinlin_calvin_xu/data_mixture/ngd3dm2_thresholdtotal_overfit_uncheatable_bpb"
THRESHOLDTOTAL_OVERFIT_OBJECTIVE_METRIC = "eval/uncheatable_eval/bpb"
THRESHOLDTOTAL_OVERFIT_MODEL_NAME = "ThresholdTotal-Overfit"
THRESHOLDTOTAL_OVERFIT_PREDICTED_OPTIMUM_VALUE = 1.0137235104068183
THRESHOLDTOTAL_OVERFIT_SUCCESS = True
THRESHOLDTOTAL_OVERFIT_MESSAGE = "CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL"
THRESHOLDTOTAL_OVERFIT_SECONDS = 1.8068296909332275

_THIS_DIR = Path(__file__).resolve().parent
_SURROGATE_SEARCH_DIR = _THIS_DIR / "exploratory" / "two_phase_many" / "surrogate_search"
_WEIGHTS_PATH = _SURROGATE_SEARCH_DIR / "thresholdtotal_overfit_optimum_full.csv"


def _renormalize_phase(domain_weights: dict[str, float]) -> dict[str, float]:
    total = float(sum(domain_weights.values()))
    if total <= 0.0:
        raise ValueError("Phase weights must sum to a positive value")
    return {domain_name: weight / total for domain_name, weight in domain_weights.items()}


def _phase_entropy(weights: np.ndarray) -> float:
    clipped = np.clip(np.asarray(weights, dtype=float), 1e-12, 1.0)
    return float(-np.sum(clipped * np.log(clipped)))


def _load_optimum_frame() -> pd.DataFrame:
    frame = pd.read_csv(_WEIGHTS_PATH)
    if set(frame.columns) != {"domain", "phase0_weight", "phase0_epochs", "phase1_weight", "phase1_epochs"}:
        raise ValueError(f"Unexpected columns in {_WEIGHTS_PATH}")
    return frame


def thresholdtotal_overfit_phase_weights() -> dict[str, dict[str, float]]:
    """Return the saved ThresholdTotal-Overfit phase weights."""
    frame = _load_optimum_frame()
    phase_0 = {str(row.domain): float(row.phase0_weight) for row in frame.itertuples(index=False)}
    phase_1 = {str(row.domain): float(row.phase1_weight) for row in frame.itertuples(index=False)}
    return normalize_phase_weights(
        {
            "phase_0": _renormalize_phase(phase_0),
            "phase_1": _renormalize_phase(phase_1),
        }
    )


def thresholdtotal_overfit_summary() -> dict[str, object]:
    """Return the saved ThresholdTotal-Overfit optimum summary with local geometry diagnostics."""
    optimum_frame = _load_optimum_frame()
    swarm = pd.read_csv(TWO_PHASE_MANY_CSV_PATH)
    if "status" in swarm.columns:
        swarm = swarm[swarm["status"] == "completed"].reset_index(drop=True)

    domain_names = optimum_frame["domain"].astype(str).tolist()
    optimum_weights = np.asarray(
        [
            optimum_frame["phase0_weight"].to_numpy(dtype=float),
            optimum_frame["phase1_weight"].to_numpy(dtype=float),
        ],
        dtype=float,
    )
    observed_weights = np.asarray(
        [
            [
                [float(row[f"phase_0_{domain_name}"]) for domain_name in domain_names],
                [float(row[f"phase_1_{domain_name}"]) for domain_name in domain_names],
            ]
            for _, row in swarm.iterrows()
        ],
        dtype=float,
    )
    distances = average_phase_tv_distance(observed_weights, optimum_weights[None, :, :])
    nearest_idx = int(np.argmin(distances))
    best_idx = int(np.argmin(swarm[THRESHOLDTOTAL_OVERFIT_OBJECTIVE_METRIC].to_numpy(dtype=float)))

    top_phase0 = optimum_frame.nlargest(10, "phase0_weight")[["domain", "phase0_weight", "phase0_epochs"]].rename(
        columns={"phase0_weight": "weight", "phase0_epochs": "epochs"}
    )
    top_phase1 = optimum_frame.nlargest(10, "phase1_weight")[["domain", "phase1_weight", "phase1_epochs"]].rename(
        columns={"phase1_weight": "weight", "phase1_epochs": "epochs"}
    )
    best_value = float(swarm.iloc[best_idx][THRESHOLDTOTAL_OVERFIT_OBJECTIVE_METRIC])

    return {
        "model": THRESHOLDTOTAL_OVERFIT_MODEL_NAME,
        "run_name": THRESHOLDTOTAL_OVERFIT_RUN_NAME,
        "objective_metric": THRESHOLDTOTAL_OVERFIT_OBJECTIVE_METRIC,
        "predicted_optimum_value": THRESHOLDTOTAL_OVERFIT_PREDICTED_OPTIMUM_VALUE,
        "observed_best_run_name": str(swarm.iloc[best_idx]["run_name"]),
        "observed_best_value": best_value,
        "gap_below_observed_best": THRESHOLDTOTAL_OVERFIT_PREDICTED_OPTIMUM_VALUE - best_value,
        "phase0_max_weight": float(optimum_frame["phase0_weight"].max()),
        "phase1_max_weight": float(optimum_frame["phase1_weight"].max()),
        "phase0_entropy": _phase_entropy(optimum_frame["phase0_weight"].to_numpy(dtype=float)),
        "phase1_entropy": _phase_entropy(optimum_frame["phase1_weight"].to_numpy(dtype=float)),
        "phase0_top_domains": top_phase0.to_dict(orient="records"),
        "phase1_top_domains": top_phase1.to_dict(orient="records"),
        "nearest_observed_run_name": str(swarm.iloc[nearest_idx]["run_name"]),
        "nearest_observed_value": float(swarm.iloc[nearest_idx][THRESHOLDTOTAL_OVERFIT_OBJECTIVE_METRIC]),
        "nearest_observed_tv_distance": float(distances[nearest_idx]),
        "success": THRESHOLDTOTAL_OVERFIT_SUCCESS,
        "message": THRESHOLDTOTAL_OVERFIT_MESSAGE,
        "seconds": THRESHOLDTOTAL_OVERFIT_SECONDS,
    }


def thresholdtotal_overfit_summary_json() -> str:
    """Return the ThresholdTotal-Overfit summary JSON."""
    return json.dumps(thresholdtotal_overfit_summary(), indent=2, sort_keys=True)


def create_thresholdtotal_overfit_weight_config(
    run_id: int = THRESHOLDTOTAL_OVERFIT_RUN_ID,
) -> WeightConfig:
    """Return the frozen ThresholdTotal-Overfit baseline."""
    return WeightConfig(
        run_id=run_id,
        phase_weights=thresholdtotal_overfit_phase_weights(),
    )
