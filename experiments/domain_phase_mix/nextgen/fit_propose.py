# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fit and propose steps for next-gen loops."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from dataclasses import dataclass

import fsspec
import pandas as pd

from marin.execution.executor import ExecutorStep, InputName, output_path_of, this_output_path

from experiments.domain_phase_mix.nextgen.contracts import LoopConfig
from experiments.domain_phase_mix.nextgen.merge_export import RUNS_PARQUET
from experiments.domain_phase_mix.nextgen.model_registry import (
    FitAndProposeResult,
    fit_and_propose,
)
from experiments.domain_phase_mix.nextgen.utils import write_json

logger = logging.getLogger(__name__)

FIT_RESULTS_JSON = "fit_results.json"
CANDIDATES_JSON = "candidates.json"
CANDIDATE_ASSIGNMENTS_JSON = "candidate_assignments.json"


@dataclass(frozen=True)
class FitProposeConfig:
    """Executor config for fit/propose stage."""

    output_path: str
    runs_parquet_path: InputName | str
    loop_json: str


def _decode_loop_config(payload: str) -> LoopConfig:
    data = json.loads(payload)
    return LoopConfig(
        name=data["name"],
        objective_metric=data["objective_metric"],
        model_names=tuple(data["model_names"]),
        n_new_runs=int(data.get("n_new_runs", 0)),
        import_sources=(),
        validation_policy=data.get("validation_policy", "top1_per_model_dedup"),
        trajectory_granularity=data.get("trajectory_granularity", "eval_checkpoints_only"),
        state_root=data.get("state_root", "domain_phase_mix/nextgen"),
        candidate_search_points=int(data.get("candidate_search_points", 8192)),
        candidate_search_seed=int(data.get("candidate_search_seed", 42)),
        execute_validation_slots=bool(data.get("execute_validation_slots", False)),
    )


def _encode_loop_config(loop: LoopConfig) -> str:
    return json.dumps(
        {
            "name": loop.name,
            "objective_metric": loop.objective_metric,
            "model_names": list(loop.model_names),
            "n_new_runs": loop.n_new_runs,
            "validation_policy": loop.validation_policy,
            "trajectory_granularity": loop.trajectory_granularity,
            "state_root": loop.state_root,
            "candidate_search_points": loop.candidate_search_points,
            "candidate_search_seed": loop.candidate_search_seed,
            "execute_validation_slots": loop.execute_validation_slots,
        },
        sort_keys=True,
    )


def run_fit_propose(config: FitProposeConfig) -> None:
    """Fit all configured models and write candidate proposals."""
    loop = _decode_loop_config(config.loop_json)

    with fsspec.open(str(config.runs_parquet_path), "rb") as f:
        run_df = pd.read_parquet(f)

    training_setup = {
        "loop_name": loop.name,
        "objective_metric": loop.objective_metric,
        "n_rows": len(run_df),
        "phase_domain_columns": sorted(col for col in run_df.columns if col.startswith("phase_")),
    }

    result: FitAndProposeResult = fit_and_propose(run_df, loop=loop, training_setup=training_setup)

    write_json(
        os.path.join(config.output_path, FIT_RESULTS_JSON),
        {
            "model_fits": [dataclasses.asdict(item) for item in result.model_fits],
            "training_setup": training_setup,
        },
    )

    write_json(
        os.path.join(config.output_path, CANDIDATES_JSON),
        [dataclasses.asdict(candidate) for candidate in result.candidates],
    )

    assignment = {
        item.model_name: item.candidate_id
        for item in result.model_fits
        if item.candidate_id is not None and item.error is None
    }
    write_json(os.path.join(config.output_path, CANDIDATE_ASSIGNMENTS_JSON), assignment)



def create_fit_propose_step(*, loop: LoopConfig, export_step: ExecutorStep, output_override_path: str) -> ExecutorStep:
    """Create fit/propose executor step."""
    return ExecutorStep(
        name=f"{loop.name}/fit_propose/models",
        description="Fit configured models and propose top-1 candidates",
        fn=run_fit_propose,
        config=FitProposeConfig(
            output_path=this_output_path(),
            runs_parquet_path=output_path_of(export_step, RUNS_PARQUET),
            loop_json=_encode_loop_config(loop),
        ),
        override_output_path=output_override_path,
    )
