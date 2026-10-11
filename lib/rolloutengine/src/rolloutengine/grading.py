# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade a finished rollout with the task's grader, outside the model's machine."""

import asyncio
from collections.abc import Mapping
from contextlib import AsyncExitStack
from typing import Any

from shellbox.machine import DEFAULT_MACHINE_OUTPUT_LIMIT_BYTES, Machine, MachineFactory
from taskcompendium.chat import chat_conversation
from taskcompendium.grading import grade_answer
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import GradingAttempt, NoGrader, ScriptGrader, VerifyitGrader
from taskcompendium.runtime.grading import grade_in_sandbox
from taskcompendium.runtime.output_capture import captured_output_files

from rolloutengine.cleanup import _Cleanup
from rolloutengine.machines import _AttemptMachineFactory, _machine_spec
from rolloutengine.spec import LoweredTaskSpec


async def _grade_rollout(
    lowered: LoweredTaskSpec,
    messages: tuple[dict[str, Any], ...],
    machine: Machine | None,
    factories: Mapping[str, MachineFactory],
    cleanup: _Cleanup,
    resources: AsyncExitStack,
) -> GradeResult:
    """Grade in process, or in a fresh verifier machine that the attempt owns."""
    task = lowered.task
    grader = task.grader
    if isinstance(grader, NoGrader):
        return GradeResult(Outcome.UNAVAILABLE, None, grader.reason)
    grading_messages = list(messages)
    final = grading_messages[-1]
    if final.get("role") == "assistant" and final.get("content") is None and final.get("tool_calls") is None:
        # A truncated reasoning-only turn has no answer, but earlier files can still be graded.
        grading_messages[-1] = {**final, "content": ""}
    conversation = chat_conversation(grading_messages)
    if isinstance(grader, VerifyitGrader) and grader.environment is None:
        return await asyncio.to_thread(grade_answer, task, GradingAttempt(conversation))
    timeout = lowered.session.verifier_timeout
    files = (
        {}
        if machine is None
        else await captured_output_files(
            machine, task.output_paths, timeout=timeout, limit_bytes=DEFAULT_MACHINE_OUTPUT_LIMIT_BYTES
        )
    )
    attempt = GradingAttempt(conversation, files)
    assert isinstance(grader, VerifyitGrader | ScriptGrader) and grader.environment is not None
    selection = lowered.runtime.verifier_machine
    assert selection is not None
    return await grade_in_sandbox(
        task,
        attempt,
        _AttemptMachineFactory(selection, factories, cleanup, resources),
        _machine_spec(grader.environment, selection),
        task_machine=machine,
        timeout=timeout,
    )
