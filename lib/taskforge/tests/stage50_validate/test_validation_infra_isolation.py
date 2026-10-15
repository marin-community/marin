# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Infrastructure failures during a validation trial are retried, never charged to the task.

Each case is the exception RolloutEngine raises for one failure pattern of a 2,049-item run's infrastructure
revisions during a trial (the chain ``engine.py`` builds: ``RolloutInterrupted(operation) from <original>``),
classified by ``classify``, validation's one classifier. A pattern passes when the trial is ``Ungraded`` with a
retryable cause. Validation never revises a program (an ungraded validation ends the item as retryable), so
retryable-or-ungraded is the whole rule here. The preflight's host failures are pinned in ``test_preflight.py``.
"""

import asyncio
from dataclasses import replace

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from rolloutengine.contracts import RolloutData, RolloutInterrupted, RolloutOperation
from rolloutengine.engine import _empty_rollout
from shellbox.machine import MachineTerminated
from taskcompendium.grading_result import GradeResult, GradingFailure
from taskcompendium.grading_result import Outcome as GradeStatus
from taskcompendium.models import AnswerType, PlainText, Source, TaskSpec
from verifyit.spec import ExactSpec

from taskforge.llm.client import ModelUnavailable
from taskforge.spec import draft
from taskforge.stage50_validate.classify import trial_outcome
from taskforge.stage50_validate.outcome import Cause, Ungraded


def task() -> TaskSpec:
    return draft.assemble(
        "infra",
        "Reply with 42.",
        AnswerType.TEXT,
        PlainText(),
        draft.answer_grader(ExactSpec(expected=("42",))),
        Source(dataset="ws5", revision="1", row="infra", importer_revision="1"),
        environment=None,
    )


def interrupted(operation: RolloutOperation, original: BaseException) -> RolloutInterrupted:
    try:
        try:
            raise original
        except BaseException as cause:
            raise RolloutInterrupted(_empty_rollout(task()), operation) from cause
    except RolloutInterrupted as error:
        return error


def deadline() -> TimeoutError:
    """``asyncio.timeout``'s ``TimeoutError`` raised from the ``CancelledError`` of the operation it cut."""
    try:
        try:
            raise asyncio.CancelledError()
        except asyncio.CancelledError as cancelled:
            raise TimeoutError() from cancelled
    except TimeoutError as error:
        return error


PARENT_TERMINATED = "Iris sandbox task failed before running: Parent job terminated"
CANNOT_SUBMIT = "Cannot submit job: parent job /muchanem/gen1000-s1p has terminated (state=JOB_STATE_KILLED)"

CASES: dict[str, tuple[RolloutInterrupted | RolloutData, Cause]] = {
    "grade-deadline (slow exec or transfer)": (interrupted(RolloutOperation.GRADE, deadline()), Cause.GRADER_TIMEOUT),
    "start-deadline": (interrupted(RolloutOperation.START, deadline()), Cause.MACHINE_START_TIMEOUT),
    "parent-terminated-at-start": (
        interrupted(RolloutOperation.START, RuntimeError(PARENT_TERMINATED)),
        Cause.MACHINE_START,
    ),
    "parent-terminated-verifier": (
        interrupted(RolloutOperation.GRADE, RuntimeError(PARENT_TERMINATED)),
        Cause.GRADER_RAISED,
    ),
    "cannot-submit-at-start": (
        interrupted(RolloutOperation.START, ConnectError(Code.FAILED_PRECONDITION, CANNOT_SUBMIT)),
        Cause.MACHINE_START,
    ),
    "cannot-submit-verifier": (
        interrupted(RolloutOperation.GRADE, ConnectError(Code.FAILED_PRECONDITION, CANNOT_SUBMIT)),
        Cause.GRADER_RAISED,
    ),
    "machine-terminated-in-tool": (
        interrupted(RolloutOperation.ADVANCE, MachineTerminated("Iris sandbox task t/0 is TASK_STATE_KILLED")),
        Cause.MACHINE_TERMINATED,
    ),
    "machine-terminated-in-grade": (
        interrupted(RolloutOperation.GRADE, MachineTerminated("Iris sandbox task t/0 is TASK_STATE_KILLED")),
        Cause.MACHINE_TERMINATED,
    ),
    "shellsim-machine-closed (#10109)": (
        interrupted(RolloutOperation.ADVANCE, RuntimeError("Machine is closed")),
        Cause.MACHINE_TERMINATED,
    ),
    "controller-unavailable": (
        interrupted(RolloutOperation.GRADE, ConnectError(Code.UNAVAILABLE, "controller connection refused")),
        Cause.MACHINE_TERMINATED,
    ),
    "package-install-network": (
        interrupted(RolloutOperation.START, OSError("uv: failed to fetch https://pypi.org/simple/numpy/")),
        Cause.MACHINE_START,
    ),
    "model-unavailable": (
        interrupted(RolloutOperation.MODEL, ModelUnavailable("hold ran out", ())),
        Cause.MODEL_UNAVAILABLE,
    ),
}


def timed_out_grade() -> RolloutData:
    """A rollout the engine returned whose grader command hit its own timeout (``GradingFailure.TIMEOUT``)."""
    grade = GradeResult(GradeStatus.INFRA_ERROR, None, "Grader command timed out", failure=GradingFailure.TIMEOUT)
    return replace(_empty_rollout(task()), grade=grade)


CASES["grader-command-timed-out"] = (timed_out_grade(), Cause.GRADER_TIMEOUT)


@pytest.mark.parametrize(("failure", "cause"), CASES.values(), ids=CASES.keys())
def test_a_host_failure_in_a_trial_is_retried(failure, cause):
    outcome = trial_outcome(failure, task())

    assert isinstance(outcome, Ungraded)
    assert outcome.cause is cause
    assert outcome.retryable, f"{cause} is not retryable"
