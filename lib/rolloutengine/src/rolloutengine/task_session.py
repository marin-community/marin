# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shellbox task operations and public model request preparation."""

import json
from collections.abc import Mapping
from contextlib import AsyncExitStack
from typing import Any

from shellbox.machine import Command, Machine, MachineFactory
from taskcompendium.chat import assistant_message
from taskcompendium.grading_result import GradeResult
from taskcompendium.models import (
    ANSWER_CALL_NAME,
    CONVERSATION_ANSWERS,
    AnswerCall,
    AnswerType,
    AssistantToolCalls,
    FinalAction,
    SessionGrader,
    TaskSpec,
)
from taskcompendium.runtime.shell import SHELL
from taskcompendium.submission import (
    answer_call_tool,
    conversation_messages,
    require_submission_compatibility,
    submission_instruction,
)

from rolloutengine.cleanup import _Cleanup
from rolloutengine.contracts import LENGTH_STOP_REASON, ModelTurn, SessionStart, Transition
from rolloutengine.grading import _grade_rollout
from rolloutengine.spec import LoweredTaskSpec

SHELL_TOOL_NAME = SHELL.name
WORKSPACE_INSTRUCTION = (
    "Use the shell tool to inspect and change the workspace. Send a final response when the task is completed."
)
SHELL_TOOL = {"type": "function", "function": SHELL.model_dump(exclude_none=True)}


def session_start(task: TaskSpec) -> SessionStart:
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
    if "shell" in task.environment_requirements.capabilities:
        if any(function.name == SHELL_TOOL_NAME for function in (*task.final_tools, *task.interaction_tools)):
            raise ValueError("The shell tool name is reserved for the Shellbox session")
        tools.append(SHELL_TOOL)
        if task.answer_type == AnswerType.WORKSPACE_STATE:
            messages.append({"role": "user", "content": WORKSPACE_INSTRUCTION})
    if tools:
        options["tools"] = tools
    return SessionStart(tuple(messages), options)


class _ShellboxTaskSession:
    """Execute shell calls and grade final task evidence."""

    def __init__(
        self,
        lowered: LoweredTaskSpec,
        machine: Machine | None,
        factories: Mapping[str, MachineFactory],
        cleanup: _Cleanup,
        resources: AsyncExitStack,
    ):
        self.lowered = lowered
        self.machine = machine
        self.factories = factories
        self.cleanup = cleanup
        self.resources = resources

    async def prepare(self) -> SessionStart:
        return session_start(self.lowered.task)

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
            if call.name != SHELL_TOOL_NAME or set(call.arguments) != {"command"}:
                observations.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.call_id,
                        "content": json.dumps({"error": "This session requires shell(command: string) calls"}),
                    }
                )
                continue
            command = call.arguments["command"]
            if not isinstance(command, str):
                observations.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.call_id,
                        "content": json.dumps({"error": "Shell command must be a string"}),
                    }
                )
                continue
            result = await self.machine.run(
                Command(
                    argv=("sh", "-c", command),
                    timeout=self.lowered.session.command_timeout,
                )
            )
            observations.append(
                {
                    "role": "tool",
                    "tool_call_id": call.call_id,
                    "content": json.dumps(
                        {
                            "stdout": result.stdout.decode(errors="replace"),
                            "stderr": result.stderr.decode(errors="replace"),
                            "exit_code": result.exit_code,
                            "reason": result.reason.value,
                            "truncated": result.stdout_truncated or result.stderr_truncated,
                        }
                    ),
                }
            )
        return Transition(done=False, observations=tuple(observations))

    async def grade(self, messages: tuple[dict[str, Any], ...]) -> GradeResult:
        return await _grade_rollout(self.lowered, messages, self.machine, self.factories, self.cleanup, self.resources)

    async def close(self) -> None:
        pass
