# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Frozen DS-RE-derived baselines for the two-phase many-domain sweep."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.nextgen.import_sources import NamedWandbRunImportSource
from experiments.domain_phase_mix.nextgen.utils import normalize_phase_weights
from experiments.domain_phase_mix.two_phase_many_observed_runs import load_two_phase_many_phase_weights

DSRE_BASELINES_SOURCE_EXPERIMENT = "pinlin_calvin_xu/data_mixture/ngd3dm2_dsre_bpb"
DSRE_OBJECTIVE_METRIC = "lm_eval/mmlu_5shot/bpb"

DSRE_ENSEMBLE_RUN_ID = 241
DSRE_ENSEMBLE_RUN_NAME = "baseline_dsre_ensemble"
DSRE_ENSEMBLE_PREDICTED_BPB_MEAN = 2.152294733659482
DSRE_ENSEMBLE_PREDICTED_BPB_STD = 0.016769688906970757

DSRE_OBSERVED_CONSENSUS_RUN_ID = 242
DSRE_OBSERVED_CONSENSUS_RUN_NAME = "baseline_dsre_observed_consensus"
DSRE_OBSERVED_CONSENSUS_SOURCE_RUN_NAME = "run_00097"
DSRE_OBSERVED_CONSENSUS_ACTUAL_BPB = 2.16427828392407

_THIS_DIR = Path(__file__).resolve().parent
_EXPLORATORY_DIR = _THIS_DIR / "exploratory" / "two_phase_many"
_ENSEMBLE_CANDIDATES_PATH = _EXPLORATORY_DIR / "dsre_ceq_debug" / "dsre_ceq_ensemble_candidates.json"


def _load_ensemble_phase_weights() -> dict[str, dict[str, float]]:
    payload = json.loads(_ENSEMBLE_CANDIDATES_PATH.read_text())
    mean_phase_weights = payload["mean_candidate"]["phase_weights"]
    robust_phase_weights = payload["robust_candidate"]["phase_weights"]
    if mean_phase_weights != robust_phase_weights:
        raise ValueError("Expected DS-RE mean and robust selectors to collapse to the same schedule")
    return normalize_phase_weights(mean_phase_weights)


DSRE_ENSEMBLE_PHASE_WEIGHTS = _load_ensemble_phase_weights()
DSRE_OBSERVED_CONSENSUS_PHASE_WEIGHTS = load_two_phase_many_phase_weights(DSRE_OBSERVED_CONSENSUS_SOURCE_RUN_NAME)


def create_dsre_ensemble_weight_config(run_id: int = DSRE_ENSEMBLE_RUN_ID) -> WeightConfig:
    """Return the frozen DS-RE ensemble baseline."""
    phase_items = DSRE_ENSEMBLE_PHASE_WEIGHTS.items()
    copied_phase_weights = {phase_name: dict(phase_weights) for phase_name, phase_weights in phase_items}
    return WeightConfig(run_id=run_id, phase_weights=copied_phase_weights)


def create_dsre_observed_consensus_weight_config(
    run_id: int = DSRE_OBSERVED_CONSENSUS_RUN_ID,
) -> WeightConfig:
    """Return the frozen observed-consensus DS-RE baseline."""
    phase_items = DSRE_OBSERVED_CONSENSUS_PHASE_WEIGHTS.items()
    copied_phase_weights = {phase_name: dict(phase_weights) for phase_name, phase_weights in phase_items}
    return WeightConfig(run_id=run_id, phase_weights=copied_phase_weights)


def create_dsre_ensemble_import_source(
    *,
    local_run_id: int = DSRE_ENSEMBLE_RUN_ID,
    source_experiment: str = DSRE_BASELINES_SOURCE_EXPERIMENT,
) -> NamedWandbRunImportSource:
    """Return a named-run import source for the standalone DS-RE ensemble baseline."""
    return NamedWandbRunImportSource(
        source_experiment=source_experiment,
        local_run_id=local_run_id,
        run_name=DSRE_ENSEMBLE_RUN_NAME,
        phase_weights=DSRE_ENSEMBLE_PHASE_WEIGHTS,
    )


def create_dsre_observed_consensus_import_source(
    *,
    local_run_id: int = DSRE_OBSERVED_CONSENSUS_RUN_ID,
    source_experiment: str = DSRE_BASELINES_SOURCE_EXPERIMENT,
) -> NamedWandbRunImportSource:
    """Return a named-run import source for the observed-consensus DS-RE baseline."""
    return NamedWandbRunImportSource(
        source_experiment=source_experiment,
        local_run_id=local_run_id,
        run_name=DSRE_OBSERVED_CONSENSUS_RUN_NAME,
        phase_weights=DSRE_OBSERVED_CONSENSUS_PHASE_WEIGHTS,
    )
