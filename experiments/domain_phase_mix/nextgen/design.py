# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Planning logic for new runs in next-gen loops."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

import fsspec

from marin.execution.executor import ExecutorStep, this_output_path

from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.experiment import MixtureExperiment
from experiments.domain_phase_mix.nextgen.contracts import LoopConfig, LoopState, PlannedRun
from experiments.domain_phase_mix.static_batch_selection import (
    build_dataset_spec_from_frame,
    prospective_d_optimal_selection,
    run_records_to_dataframe,
)

logger = logging.getLogger(__name__)

PLANNED_RUNS_FILE = "planned_runs.json"


@dataclass(frozen=True)
class SavePlannedRunsConfig:
    """Executor config for writing planned run artifacts."""

    output_path: str
    loop_name: str
    payload_json: str


def save_planned_runs(config: SavePlannedRunsConfig) -> None:
    """Write planned run metadata for reproducibility/debugging."""
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    target = os.path.join(config.output_path, PLANNED_RUNS_FILE)
    with fsspec.open(target, "w") as f:
        f.write(config.payload_json)


def _existing_weight_configs(state: LoopState) -> list[WeightConfig]:
    rows = [r for r in state.runs if r.local_run_id is not None and r.phase_weights]
    return [WeightConfig(run_id=int(r.local_run_id), phase_weights=r.phase_weights) for r in rows]


def _initial_fixed_runs(
    experiment: MixtureExperiment,
    state: LoopState,
) -> tuple[list[PlannedRun], list[WeightConfig]]:
    if state.runs:
        return [], []

    planned: list[PlannedRun] = []
    remapped_configs: list[WeightConfig] = []
    for offset, fixed in enumerate(experiment.initial_fixed_weight_configs):
        local_run_id = state.next_local_run_id + offset
        phase_weights = {
            phase_name: dict(domain_weights) for phase_name, domain_weights in fixed.weight_config.phase_weights.items()
        }
        planned.append(
            PlannedRun(
                local_run_id=local_run_id,
                run_name=fixed.run_name,
                phase_weights=phase_weights,
            )
        )
        remapped_configs.append(WeightConfig(run_id=local_run_id, phase_weights=phase_weights))

    return planned, remapped_configs


def _sample_with_current_sampler(
    loop: LoopConfig,
    experiment: MixtureExperiment,
    existing: list[WeightConfig],
) -> list[WeightConfig]:
    sampler = experiment.create_weight_sampler(seed=loop.candidate_search_seed)
    return sampler.sample_n_configs(
        loop.n_new_runs,
        deduplicate=True,
        existing_configs=existing,
    )


def _sample_with_static_selector(
    loop: LoopConfig,
    experiment: MixtureExperiment,
    state: LoopState,
    existing: list[WeightConfig],
) -> list[WeightConfig]:
    run_df = run_records_to_dataframe(state.runs)
    spec = build_dataset_spec_from_frame(
        run_df,
        objective_metric=loop.objective_metric,
        name=f"{loop.name}_state",
        loop=loop,
    )
    selected, selection = prospective_d_optimal_selection(
        spec,
        experiment,
        n_select=loop.n_new_runs,
        seed=loop.candidate_search_seed,
        pool_size=max(loop.design_candidate_pool_size, loop.n_new_runs),
        existing_configs=existing,
    )
    logger.info(
        "Static D-optimal design selected %d configs for loop %s (logdet=%.4f)",
        len(selected),
        loop.name,
        selection.info_logdet,
    )
    return selected


def plan_new_runs(loop: LoopConfig, experiment: MixtureExperiment, state: LoopState) -> list[PlannedRun]:
    """Sample new runs incrementally against prior loop state."""
    fixed_planned_runs, fixed_configs = _initial_fixed_runs(experiment, state)
    if loop.n_new_runs <= 0 and not fixed_planned_runs:
        return []

    existing = _existing_weight_configs(state)
    sampler_existing = [*existing, *fixed_configs]
    sampled: list[WeightConfig]
    if loop.n_new_runs <= 0:
        sampled = []
    elif loop.design_policy == "static_d_optimal":
        try:
            sampled = _sample_with_static_selector(loop, experiment, state, sampler_existing)
        except Exception as exc:
            logger.warning(
                "Static D-optimal design unavailable for loop %s; falling back to sampler: %s",
                loop.name,
                exc,
            )
            sampled = _sample_with_current_sampler(loop, experiment, sampler_existing)
    else:
        sampled = _sample_with_current_sampler(loop, experiment, sampler_existing)

    next_id = state.next_local_run_id + len(fixed_planned_runs)
    planned: list[PlannedRun] = list(fixed_planned_runs)
    for i, weight_config in enumerate(sampled):
        local_run_id = next_id + i
        phase_weights = {phase: dict(weights) for phase, weights in weight_config.phase_weights.items()}
        planned.append(
            PlannedRun(
                local_run_id=local_run_id,
                run_name=f"run_{local_run_id:05d}",
                phase_weights=phase_weights,
            )
        )

    logger.info("Planned %d new runs for loop %s", len(planned), loop.name)
    return planned


def create_planned_runs_step(
    loop: LoopConfig,
    planned_runs: list[PlannedRun],
    output_override_path: str,
) -> ExecutorStep:
    """Create executor step writing planned runs artifact."""
    payload = {
        "loop_name": loop.name,
        "n_planned_runs": len(planned_runs),
        "planned_runs": [
            {
                "local_run_id": run.local_run_id,
                "run_name": run.run_name,
                "phase_weights": run.phase_weights,
            }
            for run in planned_runs
        ],
    }

    return ExecutorStep(
        name=f"{loop.name}/design/plan_new_runs",
        description="Persist planned run configurations",
        fn=save_planned_runs,
        config=SavePlannedRunsConfig(
            output_path=this_output_path(),
            loop_name=loop.name,
            payload_json=json.dumps(payload, indent=2, sort_keys=True),
        ),
        override_output_path=output_override_path,
    )
