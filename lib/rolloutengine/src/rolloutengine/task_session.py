# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shellbox task operations and model request preparation."""

import json
from collections.abc import Mapping
from typing import Any

from shellbox.machine import Command, Machine, MachineFactory
from taskcompendium.chat import assistant_message
from taskcompendium.environment import EnvironmentKind
from taskcompendium.execution import StageExecution
from taskcompendium.grading_result import GradeResult
from taskcompendium.models import (
    FILESYSTEM_CAPABILITY,
    SHELL_CAPABILITY,
    AnswerType,
    AssistantToolCalls,
    TaskSpec,
)
from taskcompendium.submission import (
    ANSWER_CALL_NAME,
    AnswerFormat,
    FinalAction,
    Submission,
    conversation_messages,
    submission_request,
)

from rolloutengine.cleanup import _Cleanup
from rolloutengine.contracts import LENGTH_STOP_REASON, ModelTurn, SessionStart, Transition
from rolloutengine.grading import _grade_rollout
from rolloutengine.machines import _install_files, _run_setup_commands, _wait_for_healthcheck

SHELL_TOOL_NAME = "shell"


def _task_submission(task: TaskSpec, convention: Submission) -> Submission:
    if task.answer_type == AnswerType.NATIVE_ACTION:
        return FinalAction(id="final-action")
    return convention


def session_start(task: TaskSpec, convention: Submission) -> SessionStart:
    """Prepare only the public task fields for inference."""
    convention = _task_submission(task, convention)
    if task.environment.interaction is not None or task.answer_type in (AnswerType.FILE, AnswerType.STATE):
        messages = conversation_messages(task.context)
        options = {}
    else:
        request = submission_request(task, convention)
        messages = request.pop("messages")
        options = request
    if task.environment.kind != EnvironmentKind.NULL:
        if any(function.name == SHELL_TOOL_NAME for function in task.final_tools):
            raise ValueError("The shell tool name is reserved for executable tasks")
        options.setdefault("tools", []).append(
            {
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
        )
    return SessionStart(tuple(messages), options)


class _ShellboxTaskSession:
    """Execute shell calls and grade the final task state."""

    def __init__(
        self,
        task: TaskSpec,
        machine: Machine | None,
        convention: Submission,
        command_timeout: float,
        factories: Mapping[EnvironmentKind, MachineFactory],
        cleanup: _Cleanup,
        execution: StageExecution,
    ):
        self.task = task
        self.machine = machine
        self.convention = _task_submission(task, convention)
        self.command_timeout = command_timeout
        self.factories = factories
        self.cleanup = cleanup
        self.execution = execution

    async def prepare(self) -> SessionStart:
        available = set() if self.machine is None else {SHELL_CAPABILITY, FILESYSTEM_CAPABILITY}
        if not set(self.task.environment_requirements.capabilities) <= available:
            raise ValueError("The task environment does not supply its required capabilities")
        if self.execution.workdir_files or self.execution.setup or self.execution.healthcheck is not None:
            assert self.machine is not None
            if self.execution.workdir_files:
                result = await self.machine.run(
                    Command(("pwd",), user=self.execution.agent_user, timeout=self.command_timeout)
                )
                if result.exit_code != 0:
                    raise RuntimeError("Cannot find the stage working directory")
                workdir = result.stdout.decode().strip()
                await _install_files(
                    self.machine,
                    tuple(
                        file.model_copy(update={"path": f"{workdir.rstrip('/')}{file.path}"})
                        for file in self.execution.workdir_files
                    ),
                )
            await _run_setup_commands(self.machine, self.execution.setup, "Task stage setup")
            if self.execution.healthcheck is not None:
                await _wait_for_healthcheck(self.machine, self.execution.healthcheck)
        return session_start(self.task, self.convention)

    async def advance(self, turn: ModelTurn) -> Transition:
        try:
            message = assistant_message(turn.message)
        except (TypeError, ValueError):
            return Transition(done=True, metrics={"invalid_assistant_message": 1.0})
        if self.machine is None or not isinstance(message, AssistantToolCalls) or turn.stop_reason == LENGTH_STOP_REASON:
            return Transition(done=True)
        observations = []
        final_tools = {function.name for function in self.task.final_tools}
        for call in message.calls:
            if call.name in final_tools or (
                self.convention.answer_format == AnswerFormat.ANSWER_CALL and call.name == ANSWER_CALL_NAME
            ):
                return Transition(done=True)
            if call.name != SHELL_TOOL_NAME or set(call.arguments) != {"command"}:
                observations.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.call_id,
                        "content": json.dumps({"error": "Executable tasks require shell(command: string) calls"}),
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
                Command(argv=("sh", "-c", command), timeout=self.command_timeout, user=self.execution.agent_user)
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
        return await _grade_rollout(self.task, self.convention, messages, self.machine, self.factories, self.cleanup)

    async def close(self) -> None:
        pass
