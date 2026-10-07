# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate runtime selections before an attempt acquires resources."""

import json
from collections.abc import Callable, Mapping
from pathlib import PurePosixPath

from shellbox.machine import Machine, MachineFactory
from taskcompendium.grading_contract import resolve_verifier
from taskcompendium.models import AnswerType, EnvironmentRequirements, TaskSpec
from taskcompendium.shell_verifier import ShellVerifierSpec
from verifyit.spec import (
    DEFAULT_OUTPUT,
    GotestSpec,
    JunitSpec,
    PredictedActionSpec,
    PytestSpec,
    ScriptSpec,
    StdioSpec,
    StructuredExactSpec,
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
    if task.verifier.kind == "shell" and lowered.runtime.verifier_machine is None:
        raise ValueError("Shell grading requires a separate verifier machine")
    task_resources = task.resources.all + task.resources.worker
    for selection, requirements, resources in (
        (lowered.runtime.task_machine, task.environment_requirements, task_resources),
        (
            lowered.runtime.verifier_machine,
            task.verifier.environment_requirements,
            task.resources.all + task.resources.verifier,
        ),
    ):
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
    if task.verifier.kind == "external":
        raise ValueError("External grading requires a registered task session")
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
    if task.verifier.kind == "skipped":
        if not isinstance(json.loads(task.verifier.parameters_json).get("reason"), str):
            raise ValueError("Skipped grading requires a reason")
        return
    if task.verifier.kind == "shell":
        verifier = ShellVerifierSpec.model_validate_json(task.verifier.parameters_json)
        if (verifier.collect or verifier.artifacts) and lowered.runtime.task_machine is None:
            raise ValueError("Artifact grading requires a task machine")
        return
    verifier = resolve_verifier(task.verifier)
    executable = isinstance(verifier, StdioSpec | PytestSpec | JunitSpec | GotestSpec)
    if executable and lowered.runtime.verifier_machine is None:
        raise ValueError("Executable grading requires a separate verifier machine")
    if lowered.runtime.verifier_machine is None:
        return
    if isinstance(verifier, PredictedActionSpec):
        raise NotImplementedError("Predicted-action grading does not support a separate verifier machine")
    if isinstance(verifier, StructuredExactSpec) or task.answer_type == AnswerType.JSON:
        raise NotImplementedError("Structured candidate grading does not support a separate verifier machine")
    paths = task.output_paths
    if task.answer_type in {AnswerType.TEXT, AnswerType.NUMBER}:
        if isinstance(verifier, StdioSpec | PytestSpec | JunitSpec | GotestSpec):
            raise ValueError("Executable graders require workspace submissions")
        paths += (DEFAULT_OUTPUT if isinstance(verifier, ScriptSpec) else verifier.output,)
    for path in paths:
        candidate = PurePosixPath(path)
        if (
            not candidate.is_absolute()
            or ".." in candidate.parts
            or candidate.is_relative_to("/tests")
            or candidate.is_relative_to("/logs/verifier")
        ):
            raise ValueError(f"Submission overlaps private grading files: {path}")
