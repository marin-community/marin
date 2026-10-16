# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade a finished rollout with the task's grader, outside the model's machine."""

import asyncio
from collections.abc import Mapping
from contextlib import AsyncExitStack
from typing import Any

from shellbox.machine import Machine, MachineFactory
from taskcompendium.chat import chat_conversation
from taskcompendium.grading import grade_answer
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import GradingAttempt, NoGrader, ScriptGrader, VerifyitGrader
from taskcompendium.runtime.grading import grade_in_sandbox

from rolloutengine.cleanup import _Cleanup
from rolloutengine.machines import _AttemptMachineFactory, _machine_spec
from rolloutengine.spec import TaskRolloutSpec


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
    """Grade in process, or in a fresh verifier machine that the attempt owns.

    ``outputs`` holds the task's ``output_paths`` that the engine captured after the turn loop.
    """
    task = spec.task
    grader = task.grader
    if isinstance(grader, NoGrader):
        return GradeResult(Outcome.UNAVAILABLE, None, grader.reason)
    conversation = chat_conversation(list(messages))
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
