# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate runtime selections before an attempt acquires resources."""

from collections.abc import Callable, Mapping

from shellbox.machine import Machine, MachineFactory
from taskcompendium.models import (
    AnswerType,
    EnvironmentRequirements,
    NoGrader,
    ScriptGrader,
    SessionGrader,
    TaskSpec,
    VerifyitGrader,
)

from rolloutengine.contracts import TaskSession
from rolloutengine.spec import LoweredTaskSpec, TaskRuntimeSpec, TaskSessionSpec

SHELLBOX_SESSION = "shellbox"


def lower_task(
    task: TaskSpec,
    runtime: TaskRuntimeSpec,
    session: TaskSessionSpec,
    *,
    factories: Mapping[str, MachineFactory],
    sessions: Mapping[str, Callable[[LoweredTaskSpec, Machine | None], TaskSession]],
) -> LoweredTaskSpec:
    """Preserve the task definition and validate its selected execution providers."""
    lowered = LoweredTaskSpec(task=task, runtime=runtime, session=session)
    validate_lowered_task(lowered, factories=factories, sessions=sessions)
    return lowered


def validate_lowered_task(
    lowered: LoweredTaskSpec,
    *,
    factories: Mapping[str, MachineFactory],
    sessions: Mapping[str, Callable[[LoweredTaskSpec, Machine | None], TaskSession]],
) -> None:
    """Reject unknown providers and unsupported task requirements before startup."""
    task = lowered.task
    grader = task.grader
    verifier_machine = lowered.runtime.verifier_machine
    grader_environment = grader.environment if isinstance(grader, VerifyitGrader | ScriptGrader) else None
    if grader_environment is not None and verifier_machine is None:
        raise ValueError("A grader with an environment requires a verifier machine")
    if grader_environment is None and verifier_machine is not None:
        raise ValueError("Only a grader with an environment uses a verifier machine")
    task_resources = task.resources.all + task.resources.worker
    selections = [(lowered.runtime.task_machine, task.environment_requirements, task_resources)]
    if grader_environment is not None:
        selections.append((verifier_machine, grader_environment, task_resources + task.resources.verifier))
    for selection, requirements, resources in selections:
        if selection is None:
            if requirements != EnvironmentRequirements():
                raise ValueError("Environment requirements need a selected machine")
            continue
        if selection.backend not in factories:
            raise ValueError(f"Unknown Shellbox factory: {selection.backend!r}")
        if requirements.docker_image is None and any(resource.mtime_ns is not None for resource in resources):
            raise NotImplementedError("The built-in filesystem cannot preserve resource timestamps")
    if lowered.session.task_session != SHELLBOX_SESSION:
        if lowered.session.task_session not in sessions:
            raise ValueError(f"Unknown task session: {lowered.session.task_session!r}")
        return
    limits = lowered.session
    if (
        limits.command_timeout is not None
        and limits.tool_turn_timeout is not None
        and limits.command_timeout >= limits.tool_turn_timeout
    ):
        raise ValueError("The command timeout must be less than the tool-turn timeout")
    if isinstance(grader, SessionGrader):
        raise ValueError("Session grading requires a registered task session")
    if task.answer_type == AnswerType.STATE:
        raise NotImplementedError("The Shellbox session does not capture state answers")
    if task.interaction_tools or task.environment_requirements.tool_providers:
        raise NotImplementedError("Native tool providers require a registered task session")
    if set(task.environment_requirements.capabilities) - {"shell", "filesystem"}:
        raise NotImplementedError("The Shellbox session supports only shell and filesystem capabilities")
    if lowered.runtime.task_machine is None and (
        task.resources.all
        or task.resources.worker
        or task.answer_type in {AnswerType.FILE, AnswerType.STATE, AnswerType.WORKSPACE_STATE}
    ):
        raise ValueError("Workspace tasks require a task machine")
    if isinstance(grader, NoGrader):
        return
    if (
        isinstance(grader, ScriptGrader)
        and (grader.collect or grader.artifacts)
        and lowered.runtime.task_machine is None
    ):
        raise ValueError("Artifact grading requires a task machine")
