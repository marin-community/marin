# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shellbox task operations and public model request preparation."""

import json
import time
from collections.abc import Mapping
from typing import Any

from pydantic import JsonValue
from shellbox.machine import Machine
from taskcompendium.chat import assistant_message
from taskcompendium.grading_result import GradeResult
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
    SessionGrader,
    TaskSpec,
)
from taskcompendium.runtime.shell import BoundShellTool, resolve_shell_tool, run_shell_call
from taskcompendium.submission import (
    answer_call_tool,
    conversation_messages,
    require_submission_compatibility,
    submission_instruction,
)

from rolloutengine.contracts import (
    LENGTH_STOP_REASON,
    ModelTurn,
    SessionStart,
    TaskSessionContext,
    ToolCallCompleted,
    ToolCallResult,
    ToolContext,
    ToolProvider,
    ToolProviderSession,
    Transition,
)
from rolloutengine.lowering import bind_provider_tools

UNKNOWN_TOOL_ERROR = "Unknown tool"
WORKSPACE_INSTRUCTION = (
    "Use the {tool_name} tool to inspect and change the workspace. Send a final response when the task is completed."
)


def session_start(
    task: TaskSpec, shell_tool: BoundShellTool | None, provider_tools: tuple[FunctionDefinition, ...]
) -> SessionStart:
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
    if not isinstance(task.grader, SessionGrader) and task.answer_type in CONVERSATION_ANSWERS:
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
    return SessionStart(tuple(messages), options)


class _ShellToolSession:
    """Serve the built-in shell tool as a ``ToolProviderSession`` that runs commands on the task machine."""

    def __init__(self, machine: Machine, tool: BoundShellTool, command_timeout: float | None):
        self.machine = machine
        self.tool = tool
        self.command_timeout = command_timeout

    async def call(self, name: str, arguments: dict[str, JsonValue]) -> ToolCallResult:
        call = FunctionCall(name=name, arguments=arguments)
        return ToolCallResult(await run_shell_call(self.machine, call, self.tool, timeout=self.command_timeout))

    async def close(self) -> None:
        pass


class _ShellboxTaskSession:
    """Dispatch each tool call to the shell or a provider session by name, and grade final task evidence."""

    def __init__(self, context: TaskSessionContext, providers: Mapping[str, ToolProvider]):
        self.lowered = context.lowered
        self.machine = context.machine
        self.grader = context.require_task_grader()
        self.events = context.events
        self.shell_tool = resolve_shell_tool(context.lowered.task, context.lowered.session.shell_tool)
        self.provider_tools = bind_provider_tools(context.lowered.task, self.shell_tool, providers)
        self.provider_sessions: dict[str, ToolProviderSession] = {}
        self.tools: dict[str, ToolProviderSession] = {}

    async def prepare(self) -> SessionStart:
        if self.machine is not None and self.shell_tool is not None:
            shell = _ShellToolSession(self.machine, self.shell_tool, self.lowered.session.command_timeout)
            self.tools[self.shell_tool.definition.name] = shell
        for name, bound in self.provider_tools.items():
            if bound.requirement_name not in self.provider_sessions:
                self.provider_sessions[bound.requirement_name] = await bound.provider.open(
                    ToolContext(self.machine, bound.requirement)
                )
            self.tools[name] = self.provider_sessions[bound.requirement_name]
        definitions = tuple(tool.definition for tool in self.provider_tools.values())
        return session_start(self.lowered.task, self.shell_tool, definitions)

    async def advance(self, turn: ModelTurn) -> Transition:
        try:
            message = assistant_message(turn.message)
        except (TypeError, ValueError):
            return Transition(done=True, metrics={"invalid_assistant_message": 1.0})
        if (
            (self.machine is None and not self.tools)
            or not isinstance(message, AssistantToolCalls)
            or turn.stop_reason == LENGTH_STOP_REASON
        ):
            return Transition(done=True)
        observations = []
        final_tools = {function.name for function in self.lowered.task.final_tools}
        for call in message.calls:
            answer_call = isinstance(self.lowered.task.answer_format, AnswerCall) and call.name == ANSWER_CALL_NAME
            if call.name in final_tools or answer_call:
                return Transition(done=True)
            started = time.monotonic()
            if not isinstance(call.arguments, dict):
                result = ToolCallResult(json.dumps({"error": TOOL_ARGUMENTS_OBJECT_ERROR}))
            elif call.name not in self.tools:
                result = ToolCallResult(json.dumps({"error": f"{UNKNOWN_TOOL_ERROR}: {call.name}"}))
            else:
                result = await self.tools[call.name].call(call.name, call.arguments)
            self.events.emit(
                ToolCallCompleted(call.name, len(result.content.encode()), time.monotonic() - started),
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

    async def grade(
        self, messages: tuple[dict[str, Any], ...], *, stop_reason: str, outputs: dict[str, bytes]
    ) -> GradeResult:
        return await self.grader(messages, stop_reason=stop_reason, outputs=outputs)

    async def close(self) -> None:
        for provider in self.provider_sessions.values():
            await provider.close()
