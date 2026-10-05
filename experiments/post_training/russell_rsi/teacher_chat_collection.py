# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run chat-only teacher sessions with separate native tokens for each request."""

import asyncio
import traceback
from collections.abc import Callable, Mapping
from contextlib import AsyncExitStack
from dataclasses import asdict, dataclass

import httpx
from rigging.filesystem.storage_path import StoragePath
from rolloutengine.contracts import LENGTH_STOP_REASON, MAX_TURNS_STOP_REASON, ModelTurn, RolloutContractError
from rolloutengine.grading import _validate_task
from rolloutengine.machines import _task_machine
from rolloutengine.task_session import _ShellboxTaskSession
from shellbox.machine import MachineFactory, MachineStartupError
from taskcompendium.chat import assistant_message
from taskcompendium.environment import EnvironmentKind
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import TaskSpec
from taskcompendium.submission import AnswerFormat, SubmissionConvention

from experiments.post_training.russell_rsi.bootstrap_loop import write_once
from experiments.post_training.russell_rsi.sources import compact_json_sha256
from experiments.post_training.russell_rsi.teacher_collection import (
    TEACHER_COMMAND_TIMEOUT,
    TEACHER_MAX_TURNS,
    TEACHER_STARTUP_ATTEMPTS,
    TeacherModelConfig,
    native_teacher_turn,
    teacher_request,
)
from experiments.post_training.russell_rsi.teacher_http import journal_teacher_response

CHAT_TEACHER_KIND = "teacher-chat-only-v1"


@dataclass
class TeacherChatProvider:
    """Keep per-request tokens without a cross-request token-prefix contract."""

    client: httpx.AsyncClient
    resolve_base_url: Callable[[], str]
    config: TeacherModelConfig
    directory: StoragePath
    turn_index: int = 0

    async def __call__(self, messages: tuple[dict, ...], options: dict) -> ModelTurn:
        directory = self.directory / "turns" / f"{self.turn_index:03d}"
        self.turn_index += 1
        directory.mkdirs()
        body = teacher_request(messages, options, self.config)
        identity = {
            "kind": CHAT_TEACHER_KIND,
            "session_identity": self.config.session_identity,
            "request_sha256": compact_json_sha256(body),
        }
        raw = await journal_teacher_response(self.client, self.resolve_base_url, directory, body, identity)
        result = native_teacher_turn(raw)
        thinking_exhausted = (
            result.stop_reason == LENGTH_STOP_REASON
            and not result.message.get("content")
            and not result.message.get("tool_calls")
        )
        if not thinking_exhausted:
            assistant_message(result.message)
        write_once(directory / "model-turn.json", asdict(result))
        return result


@dataclass(frozen=True)
class ChatTeacherResult:
    """Chat-only evidence, without an RL token stream or loss mask."""

    task_id: str
    messages: tuple[dict, ...]
    turns: tuple[ModelTurn, ...]
    grade: GradeResult
    stop_reason: str
    execution_error: dict | None
    interrupted_operation: str | None
    startup_attempt: int


def chat_teacher_evidence(result: ChatTeacherResult) -> dict:
    return {"kind": CHAT_TEACHER_KIND, **asdict(result)}


async def run_teacher_chat(
    task: TaskSpec,
    directory: StoragePath,
    model: TeacherModelConfig,
    client: httpx.AsyncClient,
    resolve_base_url: Callable[[], str],
    factories: Mapping[EnvironmentKind, MachineFactory],
) -> ChatTeacherResult:
    """Run one fresh, bounded task session and preserve its request journal."""
    _validate_task(task)
    if task.stages or task.environment.interaction is not None:
        raise ValueError("Chat teacher supports only nonstaged built-in task sessions")
    reservation = directory / "chat-reservation.json"
    if reservation.exists():
        raise RuntimeError("A reserved chat session cannot replay an interrupted task workspace")
    directory.mkdirs()
    write_once(reservation, {"task_sha256": compact_json_sha256(task.model_dump(mode="json")), "model": asdict(model)})
    provider = TeacherChatProvider(client, resolve_base_url, model, directory)
    messages: list[dict] = []
    turns: list[ModelTurn] = []
    grade = GradeResult(Outcome.UNAVAILABLE, None, "Chat task did not finish grading")
    stop_reason = MAX_TURNS_STOP_REASON
    operation = "start"
    startup_attempt = 1
    try:
        for startup_attempt in range(1, TEACHER_STARTUP_ATTEMPTS + 1):
            async with asyncio.timeout(task.attempt_timeout) as attempt_deadline:
                async with AsyncExitStack() as resources:
                    try:
                        machine = await resources.enter_async_context(_task_machine(task.environment, factories))
                    except MachineStartupError as error:
                        write_once(
                            directory / f"startup-failure-{startup_attempt}.json",
                            {"type": type(error).__name__, "message": str(error)},
                        )
                        if startup_attempt == TEACHER_STARTUP_ATTEMPTS:
                            raise
                        continue
                    session = _ShellboxTaskSession(
                        task,
                        machine,
                        SubmissionConvention(id="russell-teacher", answer_format=AnswerFormat.PLAIN),
                        TEACHER_COMMAND_TIMEOUT,
                        factories,
                    )
                    resources.push_async_callback(session.close)
                    operation = "prepare"
                    start = await session.prepare()
                    messages.extend(start.messages)
                    deadline = asyncio.get_running_loop().time() + task.agent_timeout if task.agent_timeout else None
                    for index in range(TEACHER_MAX_TURNS):
                        operation = "model"
                        async with asyncio.timeout_at(deadline):
                            turn = await provider(tuple(messages), start.options)
                        turns.append(turn)
                        messages.append(turn.message)
                        if (
                            turn.stop_reason == LENGTH_STOP_REASON
                            and not turn.message.get("content")
                            and not turn.message.get("tool_calls")
                        ):
                            stop_reason = LENGTH_STOP_REASON
                            break
                        operation = "advance"
                        async with asyncio.timeout_at(deadline):
                            transition = await session.advance(turn)
                        if transition.reset_conversation is not None:
                            raise ValueError("Chat teacher does not support conversation resets")
                        stop_reason = turn.stop_reason
                        if transition.done or stop_reason == LENGTH_STOP_REASON:
                            break
                        if index + 1 == TEACHER_MAX_TURNS:
                            stop_reason = MAX_TURNS_STOP_REASON
                            break
                        messages.extend(transition.observations)
                    operation = "grade"
                    if (
                        stop_reason == LENGTH_STOP_REASON
                        and not turn.message.get("content")
                        and not turn.message.get("tool_calls")
                    ):
                        grade = GradeResult(Outcome.EXTRACTION_ERROR, None, "Teacher exhausted its thinking budget")
                    else:
                        grade = await session.grade(tuple(messages))
                    break
    except RolloutContractError:
        raise
    except Exception as error:
        if isinstance(error, TimeoutError) and attempt_deadline.expired():
            operation = "attempt"
        result = ChatTeacherResult(
            task.id,
            tuple(messages),
            tuple(turns),
            grade,
            "error",
            {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": "".join(traceback.format_exception(error)),
            },
            operation,
            startup_attempt,
        )
    else:
        result = ChatTeacherResult(
            task.id, tuple(messages), tuple(turns), grade, stop_reason, None, None, startup_attempt
        )
    write_once(directory / "chat-result.json", chat_teacher_evidence(result))
    return result
