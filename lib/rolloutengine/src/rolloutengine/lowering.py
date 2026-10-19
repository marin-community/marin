# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate runtime selections before an attempt acquires resources."""

from collections.abc import Mapping
from dataclasses import dataclass

from shellbox.machine import MachineFactory
from taskcompendium.models import (
    ANSWER_CALL_NAME,
    AnswerCall,
    AnswerType,
    CommandSemantics,
    EnvironmentRequirements,
    FunctionDefinition,
    NoGrader,
    ProviderRequirement,
    ScriptGrader,
    SessionGrader,
    TaskSpec,
    VerifyitGrader,
    require_resolved_environment,
)
from taskcompendium.runtime.environment import validate_machine_spec
from taskcompendium.runtime.shell import BoundShellTool, resolve_shell_tool

from rolloutengine.contracts import TaskSessionFactory, ToolProvider
from rolloutengine.machines import _machine_spec
from rolloutengine.spec import LoweredTaskSpec, TaskRuntimeSpec, TaskSessionSpec

SHELLBOX_SESSION = "shellbox"


@dataclass(frozen=True)
class BoundProviderTool:
    """One advertised function and the provider that serves it for a named task requirement."""

    definition: FunctionDefinition
    provider: ToolProvider
    requirement_name: str
    requirement: ProviderRequirement


def bind_provider_tools(
    task: TaskSpec, shell_tool: BoundShellTool | None, providers: Mapping[str, ToolProvider]
) -> dict[str, BoundProviderTool]:
    """Bind each task provider requirement to a registered provider and reject tool name collisions."""
    reserved = {function.name for function in task.final_tools}
    if isinstance(task.answer_format, AnswerCall):
        reserved.add(ANSWER_CALL_NAME)
    if shell_tool is not None:
        reserved.add(shell_tool.definition.name)
    bound: dict[str, BoundProviderTool] = {}
    for name, requirement in task.environment_requirements.tool_providers.items():
        if requirement.action_interface not in providers:
            raise ValueError(f"No tool provider is registered for action interface {requirement.action_interface!r}")
        provider = providers[requirement.action_interface]
        for definition in provider.tools(requirement):
            if definition.name in reserved or definition.name in bound:
                raise ValueError(f"Provider tool collides with another advertised tool: {definition.name}")
            bound[definition.name] = BoundProviderTool(definition, provider, name, requirement)
    return bound


def lower_task(
    task: TaskSpec,
    runtime: TaskRuntimeSpec,
    session: TaskSessionSpec,
    *,
    factories: Mapping[str, MachineFactory],
    sessions: Mapping[str, TaskSessionFactory],
    providers: Mapping[str, ToolProvider],
) -> LoweredTaskSpec:
    """Preserve the task definition and validate its selected execution providers."""
    lowered = LoweredTaskSpec(task=task, runtime=runtime, session=session)
    validate_lowered_task(lowered, factories=factories, sessions=sessions, providers=providers)
    return lowered


def validate_lowered_task(
    lowered: LoweredTaskSpec,
    *,
    factories: Mapping[str, MachineFactory],
    sessions: Mapping[str, TaskSessionFactory],
    providers: Mapping[str, ToolProvider],
) -> None:
    """Reject unknown providers and unsupported task requirements before startup."""
    task = lowered.task
    grader = task.grader
    verifier_machine = lowered.runtime.verifier_machine
    grader_environment = grader.environment if isinstance(grader, VerifyitGrader | ScriptGrader) else None
    require_resolved_environment(task.environment_requirements)
    if grader_environment is not None:
        require_resolved_environment(grader_environment)
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
            # Tool providers run on the host, so they do not need a machine.
            if requirements.model_copy(update={"tool_providers": {}}) != EnvironmentRequirements():
                raise ValueError("Environment requirements need a selected machine")
            continue
        if selection.backend not in factories:
            raise ValueError(f"Unknown Shellbox factory: {selection.backend!r}")
        validate_machine_spec(requirements, factories[selection.backend], _machine_spec(requirements, selection))
        if requirements.command_semantics == CommandSemantics.SHELL_SIMULATOR and any(
            resource.mtime_ns is not None for resource in resources
        ):
            raise NotImplementedError("The built-in filesystem cannot preserve resource timestamps")
    if lowered.session.task_session != SHELLBOX_SESSION:
        if lowered.session.task_session not in sessions:
            raise ValueError(f"Unknown task session: {lowered.session.task_session!r}")
        return
    shell_tool = resolve_shell_tool(task, lowered.session.shell_tool)
    if shell_tool is not None and lowered.runtime.task_machine is None:
        raise ValueError("Shell tools require a task machine")
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
    provider_tools = bind_provider_tools(task, shell_tool, providers)
    for tool in task.interaction_tools:
        if shell_tool is not None and tool.name == shell_tool.definition.name:
            continue
        if tool.name not in provider_tools:
            raise NotImplementedError(f"No shell binding or registered tool provider serves {tool.name!r}")
        if tool != provider_tools[tool.name].definition:
            raise ValueError(f"Interaction tool {tool.name!r} differs from its provider's definition")
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
