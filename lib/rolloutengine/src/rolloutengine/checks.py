# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reject a spec that no harness can run, before an attempt acquires resources."""

from collections.abc import Mapping

from shellbox.machine import MachineFactory
from taskcompendium.models import (
    AnswerType,
    CommandSemantics,
    EnvironmentRequirements,
    NoGrader,
    ScriptGrader,
    SessionGrader,
    VerifyitGrader,
    require_resolved_environment,
)
from taskcompendium.runtime.environment import validate_machine_spec

from rolloutengine.machines import _machine_spec
from rolloutengine.spec import RolloutSpec

WORKSPACE_ANSWERS = frozenset({AnswerType.FILE, AnswerType.STATE, AnswerType.WORKSPACE_STATE})


def check_spec(spec: RolloutSpec, machine_factories: Mapping[str, MachineFactory]) -> None:
    """Check the machine selections against the task's environments, resources, and grader.

    A workspace answer needs a task machine only when the engine grades it: a harness that grades a
    ``SessionGrader`` task can keep its state elsewhere, and nothing reads a ``NoGrader`` task's workspace.
    """
    task = spec.task
    grader = task.grader
    grader_environment = grader.environment if isinstance(grader, VerifyitGrader | ScriptGrader) else None
    require_resolved_environment(task.environment_requirements)
    if grader_environment is not None:
        require_resolved_environment(grader_environment)
    if grader_environment is not None and spec.verifier_machine is None:
        raise ValueError("A grader with an environment requires a verifier machine")
    if grader_environment is None and spec.verifier_machine is not None:
        raise ValueError("Only a grader with an environment uses a verifier machine")
    task_resources = task.resources.all + task.resources.worker
    selections = [(spec.task_machine, task.environment_requirements, task_resources)]
    if grader_environment is not None:
        selections.append((spec.verifier_machine, grader_environment, task_resources + task.resources.verifier))
    for selection, requirements, resources in selections:
        if selection is None:
            # Tool providers run on the host, so they do not need a machine.
            if requirements.model_copy(update={"tool_providers": {}}) != EnvironmentRequirements():
                raise ValueError("Environment requirements need a selected machine")
            continue
        if selection.backend not in machine_factories:
            raise ValueError(f"Unknown Shellbox factory: {selection.backend!r}")
        validate_machine_spec(requirements, machine_factories[selection.backend], _machine_spec(requirements, selection))
        if requirements.command_semantics == CommandSemantics.SHELL_SIMULATOR and any(
            resource.mtime_ns is not None for resource in resources
        ):
            raise NotImplementedError("The built-in filesystem cannot preserve resource timestamps")
    if spec.task_machine is None and task_resources:
        raise ValueError("Task resources require a task machine")
    engine_grades_workspace = not isinstance(grader, SessionGrader | NoGrader)
    if spec.task_machine is None and engine_grades_workspace and task.answer_type in WORKSPACE_ANSWERS:
        raise ValueError("Workspace tasks require a task machine")
    if isinstance(grader, ScriptGrader) and (grader.collect or grader.artifacts) and spec.task_machine is None:
        raise ValueError("Artifact grading requires a task machine")
