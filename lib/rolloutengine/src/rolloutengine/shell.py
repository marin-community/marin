# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The shell harness: a shell tool on the task machine and host tools from registered providers."""

import json
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

from pydantic import JsonValue
from shellbox.machine import Machine
from taskcompendium.chat import assistant_message
from taskcompendium.models import (
    ANSWER_CALL_NAME,
    CONVERSATION_ANSWERS,
    TOOL_ARGUMENTS_OBJECT_ERROR,
    AnswerCall,
    AnswerType,
    AssistantToolCalls,
    FinalAction,
    FunctionCall,
    FunctionDefinition,
    ProviderRequirement,
    TaskSpec,
)
from taskcompendium.runtime.shell import BoundShellTool, ShellToolConfig, resolve_shell_tool, run_shell_call
from taskcompendium.submission import (
    answer_call_tool,
    conversation_messages,
    require_submission_compatibility,
    submission_instruction,
)

from rolloutengine.contracts import LENGTH_STOP_REASON, EventSink, ModelTurn, Prompt, ToolCallCompleted, Transition
from rolloutengine.harness import Harness, HarnessContext, RolloutRecorder
from rolloutengine.spec import RolloutLimits

UNKNOWN_TOOL_ERROR = "Unknown tool"
DEFAULT_SHELL_TOOL = ShellToolConfig()
WORKSPACE_INSTRUCTION = (
    "Use the {tool_name} tool to inspect and change the workspace. Send a final response when the task is completed."
)


@dataclass(frozen=True)
class ToolContext:
    """The attempt's task machine and the task requirement a provider session serves."""

    machine: Machine | None
    requirement: ProviderRequirement


@dataclass(frozen=True)
class ToolCallResult:
    """A tool observation; ``done`` ends the rollout after this call, and later calls of the turn do not run."""

    content: str
    done: bool = False


class ToolProviderSession(Protocol):
    """One attempt's provider state for one task requirement."""

    async def call(self, name: str, arguments: dict[str, JsonValue]) -> ToolCallResult: ...

    async def close(self) -> None: ...


class ToolProvider(Protocol):
    """Host tools for one action interface, served beside the shell tool."""

    def tools(self, requirement: ProviderRequirement) -> tuple[FunctionDefinition, ...]:
        """The functions advertised to the model for one task requirement."""
        ...

    async def open(self, context: ToolContext) -> ToolProviderSession:
        """Start the state of one attempt, typically from ``context.requirement.initial_state``."""
        ...


NO_TOOLS: Mapping[str, ToolProvider] = MappingProxyType({})


def task_prompt(
    task: TaskSpec, shell_tool: BoundShellTool | None, provider_tools: tuple[FunctionDefinition, ...]
) -> Prompt:
    """Render only public task fields, with the answer format's instruction and tools.

    Provider tools that the task already declares as interaction tools are advertised once.
    """
    answer_format = task.answer_format
    messages = conversation_messages(task.context.events)
    options: dict[str, Any] = {}
    tools: list[dict[str, Any]] = [
        {"type": "function", "function": function.model_dump(exclude_none=True)}
        for function in (*task.final_tools, *task.interaction_tools)
    ]
    if task.answer_type in CONVERSATION_ANSWERS:
        require_submission_compatibility(task)
        instruction = submission_instruction(answer_format)
        if instruction:
            messages.append({"role": "user", "content": instruction})
        if isinstance(answer_format, AnswerCall):
            tools.append(answer_call_tool())
            if not task.final_tools:
                options.update(tool_choice="required", parallel_tool_calls=False)
        if isinstance(answer_format, FinalAction):
            if answer_format.require_call:
                options["tool_choice"] = "required"
            if answer_format.max_calls == 1:
                options["parallel_tool_calls"] = False
    if shell_tool is not None:
        if not task.tool_bindings:
            tools.append({"type": "function", "function": shell_tool.definition.model_dump(exclude_none=True)})
        if task.answer_type == AnswerType.WORKSPACE_STATE:
            messages.append(
                {"role": "user", "content": WORKSPACE_INSTRUCTION.format(tool_name=shell_tool.definition.name)}
            )
    tools.extend(
        {"type": "function", "function": function.model_dump(exclude_none=True)}
        for function in provider_tools
        if function not in task.interaction_tools
    )
    if tools:
        options["tools"] = tools
    return Prompt(tuple(messages), options)


@dataclass(frozen=True)
class _ProviderTool:
    """One advertised function and the provider that serves it for a named task requirement."""

    definition: FunctionDefinition
    provider: ToolProvider
    requirement_name: str
    requirement: ProviderRequirement


def _provider_tools(
    task: TaskSpec, shell_tool: BoundShellTool | None, providers: Mapping[str, ToolProvider]
) -> dict[str, _ProviderTool]:
    """Bind each task provider requirement to a registered provider and reject tool name collisions."""
    reserved = {function.name for function in task.final_tools}
    if isinstance(task.answer_format, AnswerCall):
        reserved.add(ANSWER_CALL_NAME)
    if shell_tool is not None:
        reserved.add(shell_tool.definition.name)
    bound: dict[str, _ProviderTool] = {}
    for name, requirement in task.environment_requirements.tool_providers.items():
        if requirement.action_interface not in providers:
            raise ValueError(f"No tool provider is registered for action interface {requirement.action_interface!r}")
        provider = providers[requirement.action_interface]
        for definition in provider.tools(requirement):
            if definition.name in reserved or definition.name in bound:
                raise ValueError(f"Provider tool collides with another advertised tool: {definition.name}")
            bound[definition.name] = _ProviderTool(definition, provider, name, requirement)
    return bound


@dataclass(frozen=True)
class _ShellHarness:
    command_timeout: float | None
    shell_tool: ShellToolConfig
    tools: Mapping[str, ToolProvider]

    def __post_init__(self) -> None:
        if self.command_timeout is not None and not (math.isfinite(self.command_timeout) and self.command_timeout > 0):
            raise ValueError(f"The command timeout must be a positive, finite number: {self.command_timeout!r}")

    def check(self, task: TaskSpec, limits: RolloutLimits) -> None:
        shell_tool = resolve_shell_tool(task, self.shell_tool)
        turn_limit = limits.tool_turn_timeout
        if self.command_timeout is not None and turn_limit is not None and self.command_timeout >= turn_limit:
            raise ValueError("The command timeout must be less than the tool-turn timeout")
        provider_tools = _provider_tools(task, shell_tool, self.tools)
        for tool in task.interaction_tools:
            if shell_tool is not None and tool.name == shell_tool.definition.name:
                continue
            if tool.name not in provider_tools:
                raise NotImplementedError(f"No shell binding or registered tool provider serves {tool.name!r}")
            if tool != provider_tools[tool.name].definition:
                raise ValueError(f"Interaction tool {tool.name!r} differs from its provider's definition")

    async def run(self, context: HarnessContext, recorder: RolloutRecorder) -> None:
        task, machine = context.task, context.machine
        shell_tool = resolve_shell_tool(task, self.shell_tool)
        provider_tools = _provider_tools(task, shell_tool, self.tools)
        if shell_tool is not None and machine is None:
            raise ValueError("Shell tools require a task machine")
        sessions: dict[str, ToolProviderSession] = {}
        tools: dict[str, ToolProviderSession] = {}
        if machine is not None and shell_tool is not None:
            tools[shell_tool.definition.name] = _ShellToolSession(machine, shell_tool, self.command_timeout)
        for name, bound in provider_tools.items():
            if bound.requirement_name not in sessions:
                session = await bound.provider.open(ToolContext(machine, bound.requirement))
                context.cleanup.push("tool_provider_close", session.close)
                sessions[bound.requirement_name] = session
            tools[name] = sessions[bound.requirement_name]
        definitions = tuple(tool.definition for tool in provider_tools.values())
        turn = await recorder.start(task_prompt(task, shell_tool, definitions))
        while True:
            transition = await _call_tools(task, machine, tools, context.events, turn)
            if transition.done:
                recorder.record(observations=transition.observations, metrics=transition.metrics)
                return
            turn = await recorder.reply(transition.observations)


def shell_harness(
    *,
    command_timeout: float | None,
    shell_tool: ShellToolConfig = DEFAULT_SHELL_TOOL,
    tools: Mapping[str, ToolProvider] = NO_TOOLS,
) -> Harness:
    """Serve a shell tool on the task machine and host tools from ``tools``, and end on a final response.

    ``shell_tool`` names the shell interface for tasks that declare the shell capability without
    binding a tool of their own. ``command_timeout`` limits each shell command; a timed-out command
    returns an observation and the model can continue. ``tools`` maps
    ``ProviderRequirement.action_interface`` to the provider that serves it.
    """
    return _ShellHarness(command_timeout, shell_tool, tools)


class _ShellToolSession:
    """Serve the shell tool as a ``ToolProviderSession`` that runs commands on the task machine."""

    def __init__(self, machine: Machine, tool: BoundShellTool, command_timeout: float | None):
        self.machine = machine
        self.tool = tool
        self.command_timeout = command_timeout

    async def call(self, name: str, arguments: dict[str, JsonValue]) -> ToolCallResult:
        call = FunctionCall(name=name, arguments=arguments)
        return ToolCallResult(await run_shell_call(self.machine, call, self.tool, timeout=self.command_timeout))

    async def close(self) -> None:
        pass


async def _call_tools(
    task: TaskSpec,
    machine: Machine | None,
    tools: Mapping[str, ToolProviderSession],
    events: EventSink,
    turn: ModelTurn,
) -> Transition:
    """Dispatch each tool call of ``turn`` to the shell or a provider session by name."""
    try:
        message = assistant_message(turn.message)
    except (TypeError, ValueError):
        return Transition(done=True, metrics={"invalid_assistant_message": 1.0})
    if (
        (machine is None and not tools)
        or not isinstance(message, AssistantToolCalls)
        or turn.stop_reason == LENGTH_STOP_REASON
    ):
        return Transition(done=True)
    observations = []
    final_tools = {function.name for function in task.final_tools}
    for call in message.calls:
        answer_call = isinstance(task.answer_format, AnswerCall) and call.name == ANSWER_CALL_NAME
        if call.name in final_tools or answer_call:
            return Transition(done=True, observations=tuple(observations))
        started = time.monotonic()
        if not isinstance(call.arguments, dict):
            result = ToolCallResult(json.dumps({"error": TOOL_ARGUMENTS_OBJECT_ERROR}))
        elif call.name not in tools:
            result = ToolCallResult(json.dumps({"error": f"{UNKNOWN_TOOL_ERROR}: {call.name}"}))
        else:
            result = await tools[call.name].call(call.name, call.arguments)
        events.emit(
            ToolCallCompleted(
                call.call_id, call.name, len(result.content.encode()), result.done, time.monotonic() - started
            ),
        )
        observations.append(
            {
                "role": "tool",
                "tool_call_id": call.call_id,
                "content": result.content,
            }
        )
        if result.done:
            return Transition(done=True, observations=tuple(observations))
    return Transition(done=False, observations=tuple(observations))
