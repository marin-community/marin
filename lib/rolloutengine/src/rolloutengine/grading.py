# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade a finished rollout with the task's grader, outside the model's machine."""

import asyncio
import os
from collections.abc import Mapping
from contextlib import AsyncExitStack
from dataclasses import replace
from typing import Any

from shellbox.machine import Machine, MachineFactory
from taskcompendium.chat import chat_conversation
from taskcompendium.grading import grade_answer, grade_remote_judge_answer
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import (
    GradingAttempt,
    NoGrader,
    ScriptGrader,
    TurnEvidence,
    VerifyitGrader,
    grades_with_judge,
    verifyit_spec,
)
from taskcompendium.protocol import with_protocol_assessment
from taskcompendium.runtime.grading import grade_in_sandbox
from verifyit.modes.grade_judge import API_KEY_ENV, BASE_URL_ENV, MODEL_ENV, JudgeConnection
from verifyit.spec import JudgeSpec

from rolloutengine.cleanup import _Cleanup
from rolloutengine.machines import _AttemptMachineFactory, _machine_spec
from rolloutengine.models import TaskRolloutSpec


def judge_connection(spec: TaskRolloutSpec) -> JudgeConnection:
    """The spec's judge connection, or else one from ``VERIFYIT_JUDGE_BASE_URL`` and ``VERIFYIT_JUDGE_API_KEY``.

    TaskCompendium's host judge path calls verifyit with ``runtime=None``, which needs an explicit connection, so
    the engine reads the environment here. The judge model comes from ``JudgeSpec.model`` or ``VERIFYIT_JUDGE_MODEL``.
    """
    grader = spec.task.grader
    assert isinstance(grader, VerifyitGrader)
    judge = verifyit_spec(grader)
    assert isinstance(judge, JudgeSpec)
    if not (judge.model.strip() or os.environ.get(MODEL_ENV, "").strip()):
        raise ValueError(f"A judge grader needs JudgeSpec.model or {MODEL_ENV}")
    if spec.judge_connection is not None:
        return spec.judge_connection
    base_url = os.environ.get(BASE_URL_ENV, "").strip()
    if not base_url:
        raise ValueError(f"A judge grader needs TaskRolloutSpec.judge_connection or {BASE_URL_ENV}")
    # Local OpenAI-compatible servers ignore the key, but the client needs one.
    return JudgeConnection(base_url, os.environ.get(API_KEY_ENV) or "unused")


async def _grade_rollout(
    spec: TaskRolloutSpec,
    messages: tuple[dict[str, Any], ...],
    outputs: dict[str, bytes],
    turns: tuple[TurnEvidence, ...],
    machine: Machine | None,
    factories: Mapping[str, MachineFactory],
    cleanup: _Cleanup,
    resources: AsyncExitStack,
    *,
    stop_reason: str,
) -> GradeResult:
    """Grade the attempt, then add the task's protocol assessment when it declares a protocol check.

    ``outputs`` holds the task's ``output_paths`` that the engine captured after the harness ended, and
    ``turns`` the serving evidence of each model turn.
    """
    task = spec.task
    if isinstance(task.grader, NoGrader):
        return GradeResult(Outcome.UNAVAILABLE, None, task.grader.reason)
    attempt = GradingAttempt(chat_conversation(list(messages)), outputs, stop_reason=stop_reason, turns=turns)
    grade = await _answer_grade(spec, attempt, machine, factories, cleanup, resources)
    return with_protocol_assessment(task, attempt, grade)


async def _answer_grade(
    spec: TaskRolloutSpec,
    attempt: GradingAttempt,
    machine: Machine | None,
    factories: Mapping[str, MachineFactory],
    cleanup: _Cleanup,
    resources: AsyncExitStack,
) -> GradeResult:
    """Grade in process, through the judge service, or in a fresh verifier machine that the attempt owns."""
    task = spec.task
    grader = task.grader
    if grades_with_judge(grader):
        return await asyncio.to_thread(grade_remote_judge_answer, task, attempt, judge_connection(spec))
    if isinstance(grader, VerifyitGrader) and grader.environment is None:
        # In-process verifiers grade the conversation answer alone.
        return await asyncio.to_thread(grade_answer, task, replace(attempt, files={}))
    assert isinstance(grader, VerifyitGrader | ScriptGrader) and grader.environment is not None
    selection = spec.verifier_machine
    assert selection is not None
    return await grade_in_sandbox(
        task,
        attempt,
        _AttemptMachineFactory(selection, factories, cleanup, resources),
        _machine_spec(grader.environment, selection),
        task_machine=machine,
        timeout=spec.limits.verifier_timeout,
    )
