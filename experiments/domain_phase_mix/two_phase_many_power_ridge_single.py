# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Frozen constant-mix realization of the Power-Ridge single-equation optimum."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.nextgen.utils import normalize_phase_weights

POWER_RIDGE_SINGLE_RUN_ID = 245
POWER_RIDGE_SINGLE_RUN_NAME = "baseline_power_ridge_single_constant_mix"
POWER_RIDGE_SINGLE_SOURCE_EXPERIMENT = "pinlin_calvin_xu/data_mixture/ngd3dm2_power_ridge_uncheatable_bpb"
POWER_RIDGE_SINGLE_OBJECTIVE_METRIC = "eval/uncheatable_eval/bpb"

_THIS_DIR = Path(__file__).resolve().parent
_SURROGATE_SEARCH_DIR = _THIS_DIR / "exploratory" / "two_phase_many" / "surrogate_search"
_SUMMARY_PATH = _SURROGATE_SEARCH_DIR / "power_ridge_single_optimum_summary.json"
_WEIGHTS_PATH = _SURROGATE_SEARCH_DIR / "power_ridge_single_optimum_weights.csv"


def power_ridge_single_summary() -> dict[str, object]:
    """Load the saved Power-Ridge optimum analysis summary."""
    return json.loads(_SUMMARY_PATH.read_text())


def power_ridge_single_summary_json() -> str:
    """Return the saved Power-Ridge optimum analysis summary JSON."""
    return _SUMMARY_PATH.read_text()


def power_ridge_single_phase_weights() -> dict[str, dict[str, float]]:
    """Return the constant-mix phase realization of the Power-Ridge optimum."""
    frame = pd.read_csv(_WEIGHTS_PATH)
    optimum_weights = {str(row.domain_name): float(row.optimum_weight) for row in frame.itertuples(index=False)}
    return normalize_phase_weights(
        {
            "phase_0": dict(optimum_weights),
            "phase_1": dict(optimum_weights),
        }
    )


def create_power_ridge_single_weight_config(
    run_id: int = POWER_RIDGE_SINGLE_RUN_ID,
) -> WeightConfig:
    """Return the frozen Power-Ridge constant-mix baseline."""
    phase_weights = power_ridge_single_phase_weights()
    return WeightConfig(run_id=run_id, phase_weights=phase_weights)
