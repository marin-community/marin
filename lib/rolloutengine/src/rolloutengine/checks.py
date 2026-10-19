# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reject a spec that no harness can run, before an attempt acquires resources."""

from collections.abc import Mapping

from shellbox.machine import MachineFactory
from taskcompendium.models import (
    CONVERSATION_ANSWERS,
    AnswerType,
    CommandSemantics,
    EnvironmentRequirements,
    NoGrader,
    ScriptGrader,
    VerifyitGrader,
    grades_with_judge,
    require_resolved_environment,
)
from taskcompendium.runtime.environment import validate_machine_spec

from rolloutengine.grading import judge_connection
from rolloutengine.machines import _machine_spec
from rolloutengine.models import TaskRolloutSpec

WORKSPACE_ANSWERS = frozenset({AnswerType.FILE, AnswerType.STATE, AnswerType.WORKSPACE_STATE})


def check_spec(spec: TaskRolloutSpec, machine_factories: Mapping[str, MachineFactory]) -> None:
    """Check the machine selections against the task's environments, resources, and grader.

    A workspace answer needs a task machine only when the engine grades it: nothing reads a ``NoGrader``
    task's workspace.
    """
    task = spec.task
    grader = task.grader
    judge = grades_with_judge(grader)
    # The engine grades a judge on the host, so it uses no machine for the judge's environment.
    grader_environment = grader.environment if isinstance(grader, VerifyitGrader | ScriptGrader) and not judge else None
    if judge:
        judge_connection(spec)
        if task.answer_type not in CONVERSATION_ANSWERS:
            raise NotImplementedError(f"The engine grades a judge only on a conversation answer, not {task.answer_type}")
    if task.answer_type == AnswerType.STATE and not isinstance(grader, NoGrader):
        raise NotImplementedError("The engine does not capture a state answer, so it cannot grade one")
    require_resolved_environment(task.environment_requirements)
    if grader_environment is not None:
        require_resolved_environment(grader_environment)
    if grader_environment is not None and spec.verifier_machine is None:
        raise ValueError("A grader with an environment requires a verifier machine")
    if grader_environment is None and spec.verifier_machine is not None:
        raise ValueError("The task's grader does not use a verifier machine")
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
    if spec.task_machine is None and task.tool_bindings:
        raise ValueError("A shell binding requires a task machine")
    if spec.task_machine is None and not isinstance(grader, NoGrader) and task.answer_type in WORKSPACE_ANSWERS:
        raise ValueError("Workspace tasks require a task machine")
    if isinstance(grader, ScriptGrader) and (grader.collect or grader.artifacts) and spec.task_machine is None:
        raise ValueError("Artifact grading requires a task machine")
