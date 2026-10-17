# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade a finished rollout with the task's grader, outside the model's machine."""

import asyncio
import os
from collections.abc import Mapping
from contextlib import AsyncExitStack
from typing import Any

from shellbox.machine import Machine, MachineFactory
from taskcompendium.chat import chat_conversation
from taskcompendium.grading import grade_answer, grade_remote_judge_answer
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import Grader, GradingAttempt, NoGrader, ScriptGrader, VerifyitGrader, verifyit_spec
from taskcompendium.runtime.grading import grade_in_sandbox
from verifyit.modes.grade_judge import API_KEY_ENV, BASE_URL_ENV, JudgeConnection
from verifyit.spec import JudgeSpec

from rolloutengine.cleanup import _Cleanup
from rolloutengine.machines import _AttemptMachineFactory, _machine_spec
from rolloutengine.spec import TaskRolloutSpec


def is_judge(grader: Grader) -> bool:
    """Whether ``grader`` is a judge, which the engine grades on the host, without a verifier machine."""
    return isinstance(grader, VerifyitGrader) and isinstance(verifyit_spec(grader), JudgeSpec)


def judge_connection(spec: TaskRolloutSpec) -> JudgeConnection:
    """The spec's judge connection, or else one from ``VERIFYIT_JUDGE_BASE_URL`` and ``VERIFYIT_JUDGE_API_KEY``."""
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
    machine: Machine | None,
    factories: Mapping[str, MachineFactory],
    cleanup: _Cleanup,
    resources: AsyncExitStack,
    *,
    stop_reason: str,
) -> GradeResult:
    """Grade in process, through the judge service, or in a fresh verifier machine that the attempt owns.

    ``outputs`` holds the task's ``output_paths`` that the engine captured after the harness ended.
    """
    task = spec.task
    grader = task.grader
    if isinstance(grader, NoGrader):
        return GradeResult(Outcome.UNAVAILABLE, None, grader.reason)
    conversation = chat_conversation(list(messages))
    if is_judge(grader):
        attempt = GradingAttempt(conversation, outputs, stop_reason=stop_reason)
        return await asyncio.to_thread(grade_remote_judge_answer, task, attempt, judge_connection(spec))
    if isinstance(grader, VerifyitGrader) and grader.environment is None:
        return await asyncio.to_thread(grade_answer, task, GradingAttempt(conversation, stop_reason=stop_reason))
    attempt = GradingAttempt(conversation, outputs, stop_reason=stop_reason)
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
