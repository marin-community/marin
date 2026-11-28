# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade a finished rollout with the task's grader, in process, in a fresh machine, or in the task machine."""

import asyncio
import os
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AsyncExitStack
from dataclasses import replace
from typing import Any

from pydantic import ValidationError
from shellbox.machine import Machine, MachineFactory, MachineSpec
from taskcompendium.chat import assistant_message, conversation_event
from taskcompendium.grading import grade_answer, grade_remote_judge_answer
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import (
    AgentMachine,
    AssistantToolCalls,
    ConversationEvent,
    ConversationTrace,
    GraderMachine,
    GradingAttempt,
    NoGrader,
    VerifyitGrader,
    grades_with_judge,
    judge_spec,
)
from taskcompendium.runtime.grading import grade_in_agent_machine, grade_in_sandbox
from verifyit.modes.grade_judge import API_KEY_ENV, BASE_URL_ENV, MODEL_ENV, JudgeConnection

from rolloutengine.cleanup import _Cleanup
from rolloutengine.machines import _AttemptMachineFactory, _verifier_view, machine_spec
from rolloutengine.models import TaskRolloutSpec


def judge_connection(spec: TaskRolloutSpec) -> JudgeConnection:
    """The spec's judge connection, or else one from ``VERIFYIT_JUDGE_BASE_URL`` and ``VERIFYIT_JUDGE_API_KEY``.

    TaskCompendium's host judge path calls verifyit with ``runtime=None``, which needs an explicit connection, so
    the engine reads the environment here. The judge model comes from ``JudgeSpec.model``, the explicit
    connection's default model, or ``VERIFYIT_JUDGE_MODEL`` for environment-configured callers.
    """
    grader = spec.task.grader
    judge = judge_spec(grader)
    assert judge is not None
    connection = spec.judge_connection
    fallback = (
        connection.default_model
        if connection is not None and connection.default_model is not None
        else os.environ.get(MODEL_ENV, "")
    )
    if not (judge.model.strip() or fallback.strip()):
        raise ValueError(f"A judge grader needs JudgeSpec.model or {MODEL_ENV}")
    if connection is not None:
        return replace(connection, default_model=fallback)
    base_url = os.environ.get(BASE_URL_ENV, "").strip()
    if not base_url:
        raise ValueError(f"A judge grader needs TaskRolloutSpec.judge_connection or {BASE_URL_ENV}")
    # Local OpenAI-compatible servers ignore the key, but the client needs one.
    return JudgeConnection(base_url, os.environ.get(API_KEY_ENV) or "unused", default_model=fallback)


def _grading_attempt(
    messages: tuple[dict[str, Any], ...],
    model_message_indices: tuple[int, ...],
    outputs: dict[str, bytes],
    *,
    stop_reason: str,
) -> GradingAttempt | GradeResult:
    """Validate transcript evidence and attribute malformed model turns to failed submissions."""
    events: list[ConversationEvent | None] = [None] * len(messages)
    model_indices = set(model_message_indices)
    for index, message in enumerate(messages):
        if index not in model_indices:
            events[index] = conversation_event(message)
    invalid_model_message: str | None = None
    for index in model_message_indices:
        try:
            events[index] = assistant_message(messages[index])
        except (ValidationError, ValueError) as error:
            invalid_model_message = str(error)
    seen_call_ids: set[str] = set()
    # ConversationTrace checks uniqueness too, but this pass identifies which side introduced a duplicate.
    for index, event in enumerate(events):
        if isinstance(event, AssistantToolCalls):
            for call in event.calls:
                if call.call_id in seen_call_ids:
                    if index in model_indices:
                        return GradeResult(
                            Outcome.SUBMISSION_FAILURE, 0.0, f"Duplicate model tool-call ID: {call.call_id}"
                        )
                    raise ValueError(f"Duplicate tool-call ID in harness prompt: {call.call_id}")
                seen_call_ids.add(call.call_id)
    if invalid_model_message is not None:
        return GradeResult(Outcome.SUBMISSION_FAILURE, 0.0, f"Invalid model message: {invalid_model_message}")
    parsed_events = tuple(event for event in events if event is not None)
    assert len(parsed_events) == len(messages)
    return GradingAttempt(ConversationTrace(events=parsed_events), outputs, stop_reason=stop_reason)


async def _grade_rollout(
    spec: TaskRolloutSpec,
    messages: tuple[dict[str, Any], ...],
    model_message_indices: tuple[int, ...],
    outputs: dict[str, bytes],
    machine: Machine | None,
    factories: Mapping[str, MachineFactory],
    cleanup: _Cleanup,
    resources: AsyncExitStack,
    *,
    stop_reason: str,
    verifier: Callable[[MachineSpec], Awaitable[Machine]] | None = None,
) -> GradeResult:
    """Grade locally, through the judge service, in the task machine, or in a verifier machine.

    ``outputs`` holds the task's ``output_paths`` that the engine captured after the harness ended. The
    caller can supply a verifier machine or let the attempt provision one. Grading in the task machine runs
    after the harness has returned.
    """
    task = spec.task
    grader = task.grader
    if isinstance(grader, NoGrader):
        return GradeResult(Outcome.UNAVAILABLE, None, grader.reason)
    attempt = _grading_attempt(messages, model_message_indices, outputs, stop_reason=stop_reason)
    if isinstance(attempt, GradeResult):
        return attempt
    if isinstance(grader, VerifyitGrader) and grades_with_judge(grader):
        return await asyncio.to_thread(grade_remote_judge_answer, task, attempt, judge_connection(spec))
    placement = grader.placement
    if placement is None:
        return await asyncio.to_thread(grade_answer, task, attempt)
    judge = judge_connection(spec) if grades_with_judge(grader) else None
    if isinstance(placement, AgentMachine):
        assert machine is not None
        return await grade_in_agent_machine(
            task, attempt, _verifier_view(machine), timeout=spec.limits.verifier_timeout, judge_connection=judge
        )
    assert isinstance(placement, GraderMachine)
    selection = spec.verifier_machine
    assert selection is not None
    with machine_spec(placement.environment, selection) as bound_spec:
        return await grade_in_sandbox(
            task,
            attempt,
            _AttemptMachineFactory(selection, factories, cleanup, resources, verifier),
            bound_spec,
            task_machine=machine,
            timeout=spec.limits.verifier_timeout,
            judge_connection=judge,
        )
