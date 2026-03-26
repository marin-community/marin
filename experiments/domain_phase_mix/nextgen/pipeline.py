# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Executor DAG constructor for the next-gen mixture loop."""

from __future__ import annotations

import os
from collections.abc import Sequence

from marin.execution.executor import ExecutorStep

from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.experiment import MixtureExperiment
from experiments.domain_phase_mix.nextgen.collect import (
    create_collect_imported_step,
    create_collect_new_step,
)
from experiments.domain_phase_mix.nextgen.contracts import LoopConfig, PlannedRun
from experiments.domain_phase_mix.nextgen.design import (
    create_planned_runs_step,
    plan_new_runs,
)
from experiments.domain_phase_mix.nextgen.fit_propose import create_fit_propose_step
from experiments.domain_phase_mix.nextgen.merge_export import (
    create_export_step,
    create_merge_step,
)
from experiments.domain_phase_mix.nextgen.state_store import (
    STATE_FILE,
    create_bootstrap_step,
    load_loop_state,
)
from experiments.domain_phase_mix.nextgen.utils import loop_root_path
from experiments.domain_phase_mix.nextgen.validation import (
    create_collect_validation_step,
    create_finalize_state_step,
    create_plan_validation_step,
    create_validation_slot_step,
)


def _create_training_steps(
    loop: LoopConfig,
    experiment: MixtureExperiment,
    planned_runs: list[PlannedRun],
) -> list[ExecutorStep]:
    train_steps: list[ExecutorStep] = []
    for planned in planned_runs:
        weight_config = WeightConfig(run_id=planned.local_run_id, phase_weights=planned.phase_weights)
        step = experiment.create_training_step(
            weight_config,
            name_prefix=loop.name,
            run_name=planned.run_name,
        )
        train_steps.append(step)
    return train_steps


def create_nextgen_steps(
    *,
    loop: LoopConfig,
    experiment: MixtureExperiment,
    wandb_entity: str = "marin-community",
    wandb_project: str = "marin",
) -> list[ExecutorStep]:
    """Create a full next-gen loop DAG with incremental state handling."""
    root = loop_root_path(loop.state_root, loop.name)

    state_path = os.path.join(root, "state", STATE_FILE)
    state = load_loop_state(state_path, loop_name=loop.name, objective_metric=loop.objective_metric)
    planned_runs = plan_new_runs(loop, experiment, state)

    bootstrap_step = create_bootstrap_step(
        loop=loop,
        output_override_path=os.path.join(root, "state"),
    )

    planned_runs_step = create_planned_runs_step(
        loop=loop,
        planned_runs=planned_runs,
        output_override_path=os.path.join(root, "design"),
    )

    training_steps = _create_training_steps(loop, experiment, planned_runs)

    import_step = create_collect_imported_step(
        loop_name=loop.name,
        objective_metric=loop.objective_metric,
        sources=tuple(loop.import_sources),
        output_override_path=os.path.join(root, "collect_import"),
    )

    collect_new_step = create_collect_new_step(
        loop_name=loop.name,
        objective_metric=loop.objective_metric,
        planned_runs=planned_runs,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        depends_on=training_steps,
        output_override_path=os.path.join(root, "collect_new"),
    )

    merge_step = create_merge_step(
        loop_name=loop.name,
        objective_metric=loop.objective_metric,
        state_step=bootstrap_step,
        imported_step=import_step,
        new_step=collect_new_step,
        output_override_path=os.path.join(root, "merge"),
    )

    export_step = create_export_step(
        loop_name=loop.name,
        merge_step=merge_step,
        output_override_path=os.path.join(root, "export"),
    )

    fit_step = create_fit_propose_step(
        loop=loop,
        export_step=export_step,
        output_override_path=os.path.join(root, "fit"),
    )

    plan_validation_step = create_plan_validation_step(
        loop=loop,
        fit_step=fit_step,
        state_step=bootstrap_step,
        output_override_path=os.path.join(root, "validation_plan"),
    )

    slot_steps = [
        create_validation_slot_step(
            loop=loop,
            model_name=model_name,
            fit_step=fit_step,
            plan_step=plan_validation_step,
            output_override_path=os.path.join(root, "validation_slots", model_name),
        )
        for model_name in loop.model_names
    ]

    collect_validation_step = create_collect_validation_step(
        loop_name=loop.name,
        slot_steps=slot_steps,
        output_override_path=os.path.join(root, "validation_collect"),
    )

    finalize_state_step = create_finalize_state_step(
        loop_name=loop.name,
        state_step=bootstrap_step,
        validation_collect_step=collect_validation_step,
        merge_step=merge_step,
        output_override_path=os.path.join(root, "report"),
    )

    steps: list[ExecutorStep] = [
        bootstrap_step,
        planned_runs_step,
        import_step,
        *training_steps,
        collect_new_step,
        merge_step,
        export_step,
        fit_step,
        plan_validation_step,
        *slot_steps,
        collect_validation_step,
        finalize_state_step,
    ]
    return steps



def summarize_step_names(steps: Sequence[ExecutorStep]) -> list[str]:
    """Return step names for debug output."""
    return [step.name for step in steps]
