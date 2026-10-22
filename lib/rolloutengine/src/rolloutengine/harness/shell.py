# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The built-in shell harness.

This supplies a single shell tool on the task machine as well as specified host tools.
"""

import json
import logging
import math
import re
import shlex
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Protocol

from pydantic import JsonValue
from shellbox.machine import DEFAULT_MACHINE_OUTPUT_LIMIT_BYTES, Command, Machine
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

from rolloutengine.harness.core import HarnessContext, ModelSession
from rolloutengine.models import (
    LENGTH_STOP_REASON,
    EventSink,
    ModelTurn,
    Prompt,
    RolloutEnded,
    RolloutLimits,
    ToolCallCompleted,
    Transition,
)

logger = logging.getLogger(__name__)

UNKNOWN_TOOL_ERROR = "Unknown tool"
DEFAULT_SHELL_TOOL = ShellToolConfig()
WORKSPACE_INSTRUCTION = (
    "Use the {tool_name} tool to inspect and change the workspace. Send a final response when the task is completed."
)
FILES_INSTRUCTION = "Read the files this task names with the {tool_name} tool; do not assume their contents."
TOOL_INSTRUCTION = "You can run commands with the {tool_name} tool."
OUTPUTS_INSTRUCTION = "Write each output file at the path the task states with the {tool_name} tool."
FINISH_INSTRUCTION = "When you have finished, {instruction}"
DEFAULT_MAX_PARSE_RETRIES = 3
TOOL_CALL_MARKER = "<tool_call>"
REASONING_END = "</think>"
PARSE_ERROR = (
    "Error: your tool call could not be parsed ({error}). Write exactly one JSON object "
    '{{"name": "{tool_name}", "arguments": {{"{parameter}": "..."}}}} inside <tool_call></tool_call>, with every '
    "string JSON-escaped, and try again."
)
SETUP_COMMAND_TIMEOUT = 60.0
"""The limit of the harness's own commands on the task machine: the working-directory link and the output check."""
TOOL_USAGE_METRICS = ("tool_calls", "parse_retry", "opened_input_file", "wrote_output_file")
"""The ``Transition.metrics`` keys that ``ShellHarness`` records on the last step of a task with a shell tool.

``opened_input_file`` is present only for a task whose conversation names one of its files, and
``wrote_output_file`` only for a task with ``output_paths``.
"""


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
    A task with a shell tool gets its instructions folded into its last user message: a separate
    trailing user turn reads as a request to answer at once, and models then skip the tools. A task
    without a shell tool keeps the answer instruction as its own final user turn.
    """
    answer_format = task.answer_format
    messages = conversation_messages(task.context.events)
    options: dict[str, Any] = {}
    tools: list[dict[str, Any]] = [
        {"type": "function", "function": function.model_dump(exclude_none=True)}
        for function in (*task.final_tools, *task.interaction_tools)
    ]
    answer_instruction = ""
    if task.answer_type in CONVERSATION_ANSWERS:
        require_submission_compatibility(task)
        answer_instruction = submission_instruction(answer_format)
        if answer_instruction and shell_tool is None:
            messages.append({"role": "user", "content": answer_instruction})
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
        messages = _with_guidance(messages, _shell_guidance(task, shell_tool, answer_instruction))
    tools.extend(
        {"type": "function", "function": function.model_dump(exclude_none=True)}
        for function in provider_tools
        if function not in task.interaction_tools
    )
    if tools:
        options["tools"] = tools
    return Prompt(tuple(messages), options)


def _conversation_text(task: TaskSpec) -> str:
    return "\n".join(
        message["content"]
        for message in conversation_messages(task.context.events)
        if isinstance(message["content"], str)
    )


def _named_task_files(task: TaskSpec) -> tuple[str, ...]:
    """The paths of the worker's task files whose names appear in the task's conversation."""
    text = _conversation_text(task)
    paths = {resource.path for resource in (*task.resources.all, *task.resources.worker)}
    return tuple(sorted(path for path in paths if PurePosixPath(path).name in text))


def _shell_guidance(task: TaskSpec, shell_tool: BoundShellTool, answer_instruction: str) -> str:
    """How to use the shell tool on this task, then how to finish.

    A task that binds its own shell tool already presents it, so it gets only the answer instruction.
    """
    sentences = []
    if task.answer_type == AnswerType.WORKSPACE_STATE:
        sentences.append(WORKSPACE_INSTRUCTION)
    elif not task.tool_bindings:
        sentences.append(FILES_INSTRUCTION if _named_task_files(task) else TOOL_INSTRUCTION)
        if task.output_paths:
            sentences.append(OUTPUTS_INSTRUCTION)
    guidance = [sentence.format(tool_name=shell_tool.definition.name) for sentence in sentences]
    if answer_instruction:
        guidance.append(FINISH_INSTRUCTION.format(instruction=answer_instruction[0].lower() + answer_instruction[1:]))
    return " ".join(guidance)


def _with_guidance(messages: list[dict[str, Any]], guidance: str) -> list[dict[str, Any]]:
    """Append ``guidance`` to the last user message, or add it as a user message after other roles."""
    if not guidance:
        return messages
    if messages and messages[-1]["role"] == "user" and isinstance(messages[-1]["content"], str):
        last = messages[-1]
        return [*messages[:-1], {**last, "content": f"{last['content'].rstrip()}\n\n{guidance}"}]
    return [*messages, {"role": "user", "content": guidance}]


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


@dataclass(frozen=True, kw_only=True)
class ShellHarness:
    """Serve a shell tool on the task machine and host tools from registered providers, and end on a final response."""

    command_timeout: float | None
    """The limit of each shell command. A timed-out command returns an observation, and the model can continue."""
    output_limit_bytes: int = DEFAULT_MACHINE_OUTPUT_LIMIT_BYTES
    """The most bytes of stdout, and of stderr, that a shell command returns to the model."""
    shell_tool: ShellToolConfig = DEFAULT_SHELL_TOOL
    """The shell interface for tasks that declare the shell capability without binding a tool of their own."""
    tools: Mapping[str, ToolProvider] = NO_TOOLS
    """The provider that serves each ``ProviderRequirement.action_interface``."""
    max_parse_retries: int = DEFAULT_MAX_PARSE_RETRIES
    """How many replies with a ``<tool_call>`` that the server did not parse get an error and another turn."""

    def __post_init__(self) -> None:
        if self.max_parse_retries < 0:
            raise ValueError(f"The parse retry limit must not be negative: {self.max_parse_retries!r}")
        if self.command_timeout is not None and not (math.isfinite(self.command_timeout) and self.command_timeout > 0):
            raise ValueError(f"The command timeout must be a positive, finite number: {self.command_timeout!r}")
        # Machines treat a negative limit differently, for example ShellSim slices with it and returns most of a stream.
        if self.output_limit_bytes <= 0:
            raise ValueError(f"The output limit must be a positive number of bytes: {self.output_limit_bytes!r}")

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

    async def run(self, context: HarnessContext, session: ModelSession) -> None:
        task, machine = context.task, context.machine
        shell_tool = resolve_shell_tool(task, self.shell_tool)
        provider_tools = _provider_tools(task, shell_tool, self.tools)
        if shell_tool is not None and machine is None:
            raise ValueError("Shell tools require a task machine")
        sessions: dict[str, ToolProviderSession] = {}
        tools: dict[str, ToolProviderSession] = {}
        if machine is not None and shell_tool is not None:
            tools[shell_tool.definition.name] = _ShellToolSession(
                machine, shell_tool, self.command_timeout, self.output_limit_bytes
            )
        for name, bound in provider_tools.items():
            if bound.requirement_name not in sessions:
                provider_session = await bound.provider.open(ToolContext(machine, bound.requirement))
                context.cleanup.push("tool_provider_close", provider_session.close)
                sessions[bound.requirement_name] = provider_session
            tools[name] = sessions[bound.requirement_name]
        definitions = tuple(tool.definition for tool in provider_tools.values())
        usage = None
        if machine is not None and shell_tool is not None:
            await _link_working_directory(machine, task)
            usage = _ToolUsage(task, machine, shell_tool)
        turn = await session.start(task_prompt(task, shell_tool, definitions))
        try:
            await self._tool_loop(context, session, tools, usage, turn)
        except RolloutEnded:
            # A limit or a failed model request ends the attempt; its tool usage still belongs to the last step.
            if usage is not None:
                session.record(metrics=await usage.metrics())
            raise

    async def _tool_loop(
        self,
        context: HarnessContext,
        session: ModelSession,
        tools: Mapping[str, ToolProviderSession],
        usage: "_ToolUsage | None",
        turn: ModelTurn,
    ) -> None:
        """Run each turn's tool calls and reply until a turn ends the attempt; retry unparsed shell-tool calls."""
        while True:
            if usage is not None:
                usage.observe(turn)
                error = _unparsed_tool_call(turn)
                if error is not None and usage.parse_retry < self.max_parse_retries:
                    usage.parse_retry += 1
                    turn = await session.reply((_parse_error_message(error, usage.shell_tool),))
                    continue
            transition = await _call_tools(context.task, context.machine, tools, context.events, turn)
            if transition.done:
                metrics = transition.metrics if usage is None else {**transition.metrics, **await usage.metrics()}
                session.record(observations=transition.observations, metrics=metrics)
                return
            turn = await session.reply(transition.observations)


async def _link_working_directory(machine: Machine, task: TaskSpec) -> None:
    """Make ``<name>/X`` resolve in a working directory ``/<name>`` when the conversation names such paths.

    Resource paths are relative to the root, so prompts often name ``/workspace/data.txt`` as
    ``workspace/data.txt`` while commands start in ``/workspace``. A ``workspace -> .`` link there makes both
    spellings work. The harness skips the link when the name already exists, and continues without it when the
    machine refuses it.
    """
    workdir = task.environment_requirements.working_directory
    if workdir is None:
        return
    path = PurePosixPath(workdir)
    if not path.name or path.parent != PurePosixPath("/"):
        return
    if not re.search(rf"(?<![\w./-]){re.escape(path.name)}/", _conversation_text(task)):
        return
    name = shlex.quote(path.name)
    command = Command(("sh", "-c", f"[ -e {name} ] || ln -s . {name}"), cwd=workdir, timeout=SETUP_COMMAND_TIMEOUT)
    result = await machine.run(command)
    if result.exit_code != 0:
        logger.warning(
            "Linking %s -> . in %s failed for task %s: %s",
            path.name,
            workdir,
            task.id,
            result.stderr.decode(errors="replace")[-500:],
        )


def _unparsed_tool_call(turn: ModelTurn) -> str | None:
    """Why the server did not return the ``<tool_call>`` that a complete reply wrote, or None.

    This reads the Hermes tool-call format, which Qwen chat templates use. Reasoning before ``</think>`` is ignored.
    """
    message = turn.message
    content = message.get("content")
    if turn.stop_reason == LENGTH_STOP_REASON or message.get("tool_calls") or not isinstance(content, str):
        return None
    answer = content.rsplit(REASONING_END, 1)[-1]
    if TOOL_CALL_MARKER not in answer:
        return None
    body = answer.split(TOOL_CALL_MARKER, 1)[1].split("</tool_call>", 1)[0].strip()
    try:
        json.loads(body)
    except json.JSONDecodeError as error:
        return f"invalid JSON: {error.msg} at character {error.pos}"
    return "the call was not recognized as a tool call"


def _parse_error_message(error: str, shell_tool: BoundShellTool) -> dict[str, str]:
    """A user turn that Hermes chat templates render like a tool response.

    A tool message needs the ID of a parsed call. A plain user turn would count as a new query, and Qwen thinking
    templates drop the reasoning of earlier turns before the last query, so the next prompt would not extend the
    served tokens.
    """
    feedback = PARSE_ERROR.format(
        error=error, tool_name=shell_tool.definition.name, parameter=shell_tool.command_parameter
    )
    return {"role": "user", "content": f"<tool_response>\n{feedback}\n</tool_response>"}


class _ToolUsage:
    """Count the tool calls and parse retries of one attempt, and whether it touched the task's named files."""

    def __init__(self, task: TaskSpec, machine: Machine, shell_tool: BoundShellTool):
        self.machine = machine
        self.shell_tool = shell_tool
        self.input_names = tuple(PurePosixPath(path).name for path in _named_task_files(task))
        self.output_paths = task.output_paths
        self.tool_calls = 0
        self.parse_retry = 0
        self.opened_input_file = False

    def observe(self, turn: ModelTurn) -> None:
        try:
            message = assistant_message(turn.message)
        except (TypeError, ValueError):
            # _call_tools ends the attempt on this turn and records it as invalid_assistant_message.
            return
        if not isinstance(message, AssistantToolCalls):
            return
        self.tool_calls += len(message.calls)
        for call in message.calls:
            if call.name != self.shell_tool.definition.name or not isinstance(call.arguments, dict):
                continue
            command = call.arguments.get(self.shell_tool.command_parameter)
            if isinstance(command, str) and any(name in command for name in self.input_names):
                self.opened_input_file = True

    async def metrics(self) -> dict[str, float]:
        metrics = {"tool_calls": float(self.tool_calls), "parse_retry": float(self.parse_retry)}
        if self.input_names:
            metrics["opened_input_file"] = float(self.opened_input_file)
        if self.output_paths:
            test = " && ".join(f"test -e {shlex.quote(path)}" for path in self.output_paths)
            result = await self.machine.run(Command(("sh", "-c", test), timeout=SETUP_COMMAND_TIMEOUT))
            metrics["wrote_output_file"] = float(result.exit_code == 0)
        return metrics


class _ShellToolSession:
    """Serve the shell tool as a ``ToolProviderSession`` that runs commands on the task machine."""

    def __init__(self, machine: Machine, tool: BoundShellTool, command_timeout: float | None, output_limit_bytes: int):
        self.machine = machine
        self.tool = tool
        self.command_timeout = command_timeout
        self.output_limit_bytes = output_limit_bytes

    async def call(self, name: str, arguments: dict[str, JsonValue]) -> ToolCallResult:
        call = FunctionCall(name=name, arguments=arguments)
        observation = await run_shell_call(
            self.machine, call, self.tool, timeout=self.command_timeout, output_limit_bytes=self.output_limit_bytes
        )
        return ToolCallResult(observation)

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
