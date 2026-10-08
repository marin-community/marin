# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade a finished rollout with the task's grader, outside the model's machine."""

import asyncio
from collections.abc import Mapping
from contextlib import AsyncExitStack
from typing import Any

from shellbox.machine import DEFAULT_MACHINE_OUTPUT_LIMIT_BYTES, Command, Machine, MachineFactory
from taskcompendium.chat import chat_conversation
from taskcompendium.grading import grade_answer
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import GradingAttempt, NoGrader, ScriptGrader, VerifyitGrader
from taskcompendium.runtime.grading import grade_in_sandbox

from rolloutengine.cleanup import _Cleanup
from rolloutengine.machines import _AttemptMachineFactory, _machine_spec
from rolloutengine.spec import LoweredTaskSpec

MISSING_FILE_EXIT = 44


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
    conversation = chat_conversation(list(messages))
    if isinstance(grader, VerifyitGrader) and grader.environment is None:
        return await asyncio.to_thread(grade_answer, task, GradingAttempt(conversation))
    timeout = lowered.session.verifier_timeout
    attempt = GradingAttempt(conversation, await _capture_outputs(machine, task.output_paths, timeout))
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


async def _capture_outputs(machine: Machine | None, paths: tuple[str, ...], timeout: float | None) -> dict[str, bytes]:
    if machine is None:
        return {}
    files = {}
    for path in paths:
        result = await machine.run(
            Command(
                (
                    "sh",
                    "-c",
                    f'if [ -f "$1" ]; then head -c "$2" -- "$1"; else exit {MISSING_FILE_EXIT}; fi',
                    "capture-output",
                    path,
                    str(DEFAULT_MACHINE_OUTPUT_LIMIT_BYTES + 1),
                ),
                timeout=timeout,
                user="0",
                output_limit_bytes=DEFAULT_MACHINE_OUTPUT_LIMIT_BYTES + 1,
            )
        )
        if result.exit_code == MISSING_FILE_EXIT:
            continue
        if result.exit_code != 0 or result.stdout_truncated or len(result.stdout) > DEFAULT_MACHINE_OUTPUT_LIMIT_BYTES:
            raise RuntimeError(f"Cannot capture task output within the size limit: {path}")
        files[path] = result.stdout
    return files
