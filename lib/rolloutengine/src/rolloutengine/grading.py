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
from taskcompendium.chat import chat_conversation
from taskcompendium.grading import grade_answer, grade_remote_judge_answer
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import (
    AgentMachine,
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
from rolloutengine.machines import _AttemptMachineFactory, _machine_spec, _verifier_view
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


async def _grade_rollout(
    spec: TaskRolloutSpec,
    messages: tuple[dict[str, Any], ...],
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
    try:
        conversation = chat_conversation(list(messages))
    except ValidationError as error:
        return GradeResult(Outcome.SUBMISSION_FAILURE, 0.0, f"Invalid model conversation: {error}")
    attempt = GradingAttempt(conversation, outputs, stop_reason=stop_reason)
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
    with _machine_spec(placement.environment, selection) as machine_spec:
        return await grade_in_sandbox(
            task,
            attempt,
            _AttemptMachineFactory(selection, factories, cleanup, resources, verifier),
            machine_spec,
            task_machine=machine,
            timeout=spec.limits.verifier_timeout,
            judge_connection=judge,
        )
