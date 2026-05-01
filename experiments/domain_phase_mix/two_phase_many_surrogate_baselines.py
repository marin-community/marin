# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Frozen surrogate-derived baselines for the two-phase many-domain sweep."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.nextgen.import_sources import NamedWandbRunImportSource
from experiments.domain_phase_mix.nextgen.utils import normalize_phase_weights

SURROGATE_BASELINES_SOURCE_EXPERIMENT = "pinlin_calvin_xu/data_mixture/ngd3dm2_surrogate_bpb"
SURROGATE_OBJECTIVE_METRIC = "lm_eval/mmlu_5shot/bpb"

CLR_RIDGE_RUN_ID = 243
CLR_RIDGE_RUN_NAME = "baseline_clr_ridge_balanced"
CLR_RIDGE_ALPHA = 0.10
CLR_RIDGE_PREDICTED_BPB = 2.159002632830922

DSRE_CEQ_ST_LITE_RUN_ID = 244
DSRE_CEQ_ST_LITE_RUN_NAME = "baseline_dsre_ceq_st_lite"
DSRE_CEQ_ST_LITE_PREDICTED_BPB = 2.1292702211468337

_THIS_DIR = Path(__file__).resolve().parent
_OPTIMA_PATH = _THIS_DIR / "exploratory" / "two_phase_many" / "dsre_ceq_debug" / "clr_ridge_and_stlite_optima.json"


def _load_phase_weights() -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    payload = json.loads(_OPTIMA_PATH.read_text())
    clr_phase_weights = normalize_phase_weights(payload["clr_ridge_balanced"]["phase_weights"])
    st_lite_phase_weights = normalize_phase_weights(payload["dsre_ceq_st_lite"]["phase_weights"])
    return clr_phase_weights, st_lite_phase_weights


CLR_RIDGE_PHASE_WEIGHTS, DSRE_CEQ_ST_LITE_PHASE_WEIGHTS = _load_phase_weights()


def create_clr_ridge_weight_config(run_id: int = CLR_RIDGE_RUN_ID) -> WeightConfig:
    """Return the frozen CLR-Ridge baseline."""
    phase_items = CLR_RIDGE_PHASE_WEIGHTS.items()
    copied_phase_weights = {phase_name: dict(phase_weights) for phase_name, phase_weights in phase_items}
    return WeightConfig(run_id=run_id, phase_weights=copied_phase_weights)


def create_dsre_ceq_st_lite_weight_config(run_id: int = DSRE_CEQ_ST_LITE_RUN_ID) -> WeightConfig:
    """Return the frozen DS-RE-CEQ-ST(lite) baseline."""
    phase_items = DSRE_CEQ_ST_LITE_PHASE_WEIGHTS.items()
    copied_phase_weights = {phase_name: dict(phase_weights) for phase_name, phase_weights in phase_items}
    return WeightConfig(run_id=run_id, phase_weights=copied_phase_weights)


def create_clr_ridge_import_source(
    *,
    local_run_id: int = CLR_RIDGE_RUN_ID,
    source_experiment: str = SURROGATE_BASELINES_SOURCE_EXPERIMENT,
) -> NamedWandbRunImportSource:
    """Return a named-run import source for the standalone CLR-Ridge baseline."""
    return NamedWandbRunImportSource(
        source_experiment=source_experiment,
        local_run_id=local_run_id,
        run_name=CLR_RIDGE_RUN_NAME,
        phase_weights=CLR_RIDGE_PHASE_WEIGHTS,
    )


def create_dsre_ceq_st_lite_import_source(
    *,
    local_run_id: int = DSRE_CEQ_ST_LITE_RUN_ID,
    source_experiment: str = SURROGATE_BASELINES_SOURCE_EXPERIMENT,
) -> NamedWandbRunImportSource:
    """Return a named-run import source for the standalone DS-RE-CEQ-ST(lite) baseline."""
    return NamedWandbRunImportSource(
        source_experiment=source_experiment,
        local_run_id=local_run_id,
        run_name=DSRE_CEQ_ST_LITE_RUN_NAME,
        phase_weights=DSRE_CEQ_ST_LITE_PHASE_WEIGHTS,
    )
