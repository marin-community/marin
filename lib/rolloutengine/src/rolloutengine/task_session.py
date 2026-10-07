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
from taskcompendium.models import AnswerType, AssistantToolCalls, TaskSpec
from taskcompendium.runtime.task_grading import resolve_verifier
from taskcompendium.submission import (
    ANSWER_CALL_NAME,
    AnswerCall,
    FinalAction,
    JsonValueAnswer,
    SubmissionConvention,
    answer_call_tool,
    conversation_messages,
    submission_compatibility,
    submission_instruction,
)

from rolloutengine.cleanup import _Cleanup
from rolloutengine.contracts import LENGTH_STOP_REASON, ModelTurn, SessionStart, Transition
from rolloutengine.grading import _grade_rollout
from rolloutengine.spec import LoweredTaskSpec

SHELL_TOOL_NAME = "shell"
SHELL_TOOL = {
    "type": "function",
    "function": {
        "name": SHELL_TOOL_NAME,
        "description": "Run a shell command in the task workspace. Files persist between commands.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
            "additionalProperties": False,
        },
    },
}


def _task_submission(task: TaskSpec, convention: SubmissionConvention) -> SubmissionConvention:
    if task.answer_type == AnswerType.NATIVE_ACTION and not isinstance(convention, FinalAction):
        return FinalAction(id="final-action")
    if task.answer_type == AnswerType.JSON and not isinstance(convention, JsonValueAnswer):
        return JsonValueAnswer(id="json-value")
    return convention


def session_start(task: TaskSpec, convention: SubmissionConvention) -> SessionStart:
    """Render only public task fields, with the selected final-action limits."""
    convention = _task_submission(task, convention)
    messages = conversation_messages(task.context)
    options: dict[str, Any] = {}
    tools: list[dict[str, Any]] = [
        {"type": "function", "function": function.model_dump(exclude_none=True)}
        for function in (*task.final_tools, *task.interaction_tools)
    ]
    if task.verifier.kind != "external" and task.answer_type not in {
        AnswerType.FILE,
        AnswerType.STATE,
        AnswerType.WORKSPACE_STATE,
    }:
        if task.verifier.kind not in {"shell", "skipped"}:
            compatibility = submission_compatibility(task, convention, resolved_verifier=resolve_verifier(task.verifier))
            if not compatibility.compatible:
                raise ValueError(f"Submission convention is incompatible: {compatibility.reasons}")
        elif not convention.supports(task.answer_type):
            raise ValueError("Submission convention is incompatible with the task")
        instruction = submission_instruction(convention)
        if instruction:
            messages.append({"role": "user", "content": instruction})
        if isinstance(convention, AnswerCall):
            tools.append(answer_call_tool())
            if not task.final_tools:
                options.update(tool_choice="required", parallel_tool_calls=False)
        if isinstance(convention, FinalAction):
            if convention.require_call:
                options["tool_choice"] = "required"
            if convention.max_calls == 1:
                options["parallel_tool_calls"] = False
    if "shell" in task.environment_requirements.capabilities:
        if any(function.name == SHELL_TOOL_NAME for function in (*task.final_tools, *task.interaction_tools)):
            raise ValueError("The shell tool name is reserved for the Shellbox session")
        tools.append(SHELL_TOOL)
    if tools:
        options["tools"] = tools
    return SessionStart(tuple(messages), options)


class _ShellboxTaskSession:
    """Execute shell calls and grade final task evidence."""

    def __init__(
        self,
        lowered: LoweredTaskSpec,
        machine: Machine | None,
        convention: SubmissionConvention,
        factories: Mapping[str, MachineFactory],
        cleanup: _Cleanup,
        resources: AsyncExitStack,
    ):
        self.lowered = lowered
        self.machine = machine
        self.convention = _task_submission(lowered.task, convention)
        self.factories = factories
        self.cleanup = cleanup
        self.resources = resources

    async def prepare(self) -> SessionStart:
        return session_start(self.lowered.task, self.convention)

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
            if call.name in final_tools or (isinstance(self.convention, AnswerCall) and call.name == ANSWER_CALL_NAME):
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
                    timeout=self.lowered.session.tool_turn_timeout,
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
        return await _grade_rollout(
            self.lowered, self.convention, messages, self.machine, self.factories, self.cleanup, self.resources
        )

    async def close(self) -> None:
        pass
