# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shellbox task operations and public model request preparation."""

import json
import time
from typing import Any

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
    Transition,
    emit_event,
)

WORKSPACE_INSTRUCTION = (
    "Use the {tool_name} tool to inspect and change the workspace. Send a final response when the task is completed."
)


def session_start(task: TaskSpec, shell_tool: BoundShellTool | None) -> SessionStart:
    """Render only public task fields, with the answer format's instruction and tools."""
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
    if tools:
        options["tools"] = tools
    return SessionStart(tuple(messages), options)


class _ShellboxTaskSession:
    """Execute shell calls and grade final task evidence."""

    def __init__(self, context: TaskSessionContext):
        self.lowered = context.lowered
        self.machine = context.machine
        self.grader = context.require_task_grader()
        self.observer = context.observer
        self.shell_tool = resolve_shell_tool(context.lowered.task, context.lowered.session.shell_tool)

    async def prepare(self) -> SessionStart:
        return session_start(self.lowered.task, self.shell_tool)

    async def advance(self, turn: ModelTurn) -> Transition:
        try:
            message = assistant_message(turn.message)
        except (TypeError, ValueError):
            return Transition(done=True, metrics={"invalid_assistant_message": 1.0})
        if self.machine is None or not isinstance(message, AssistantToolCalls) or turn.stop_reason == LENGTH_STOP_REASON:
            return Transition(done=True)
        observations = []
        final_tools = {function.name for function in self.lowered.task.final_tools}
        for call in message.calls:
            answer_call = isinstance(self.lowered.task.answer_format, AnswerCall) and call.name == ANSWER_CALL_NAME
            if call.name in final_tools or answer_call:
                return Transition(done=True)
            started = time.monotonic()
            if not isinstance(call.arguments, dict):
                observation = json.dumps({"error": TOOL_ARGUMENTS_OBJECT_ERROR})
            elif self.shell_tool is None:
                observation = json.dumps({"error": "No shell tool is available"})
            else:
                observation = await run_shell_call(
                    self.machine,
                    FunctionCall(name=call.name, arguments=call.arguments),
                    self.shell_tool,
                    timeout=self.lowered.session.command_timeout,
                )
            await emit_event(
                self.observer,
                ToolCallCompleted(
                    self.lowered.task.id, call.name, len(observation.encode()), time.monotonic() - started
                ),
            )
            observations.append(
                {
                    "role": "tool",
                    "tool_call_id": call.call_id,
                    "content": observation,
                }
            )
        return Transition(done=False, observations=tuple(observations))

    async def grade(self, messages: tuple[dict[str, Any], ...], *, stop_reason: str) -> GradeResult:
        return await self.grader(messages, stop_reason=stop_reason)

    async def close(self) -> None:
        pass
