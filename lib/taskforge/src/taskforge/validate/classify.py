# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The one classifier from trial failures to ``Cause``, and the trial outcome built on it.

A failure is either an exception (a ``RolloutInterrupted`` from the engine with the original error
as its ``__cause__``, a ``RolloutContractError``, a ``GlmClient`` error, a shellbox error) or a
rollout the engine returned without a usable grade. A recognized original error wins over the
interrupted operation, so a ``GlmUnavailable`` during ``MODEL`` is ``MODEL_UNAVAILABLE``. Anything
no rule matches is ``UNCLASSIFIED``, which ``Evidence`` counts like any other cause.

Task setup failures are told apart from machine failures only by their message text
(``_matches_rolloutengine_setup_message``), because the engine raises a bare ``RuntimeError`` or
``TimeoutError`` for both.
"""

import traceback

from rolloutengine.contracts import (
    AGENT_TIMEOUT_STOP_REASON,
    LENGTH_STOP_REASON,
    GenerationLimitReached,
    RolloutContractError,
    RolloutData,
    RolloutInterrupted,
    RolloutOperation,
)
from shellbox.machine import UnsupportedMachineSpec
from taskcompendium.grading_result import GradingFailure
from taskcompendium.grading_result import Outcome as GradeStatus

from taskforge.llm.client import GlmContextExhausted, GlmRequestRejected, GlmUnavailable
from taskforge.validate.outcome import GRADED_STATUSES, Cause, Graded, Outcome, Ungraded

GRADING_FAILURES = {
    GradingFailure.TIMEOUT: Cause.GRADER_TIMEOUT,
    GradingFailure.MISSING_REWARD: Cause.GRADER_MISSING_REWARD,
    GradingFailure.EMPTY_REWARD: Cause.GRADER_EMPTY_REWARD,
    GradingFailure.INVALID_REWARD: Cause.GRADER_INVALID_REWARD,
    GradingFailure.EXECUTION: Cause.GRADER_EXECUTION,
}
ROLLOUTENGINE_SETUP_MESSAGE_PREFIXES = ("Environment setup command ", "Environment healthcheck failed", "Task stage ")
"""Message prefixes of the setup, healthcheck and stage-setup errors RolloutEngine raises."""


def classify(failure: BaseException | RolloutData) -> Cause:
    """Map an exception, or a rollout returned without a usable grade, to its ``Cause``.

    Raises:
        ValueError: ``failure`` is a rollout with a usable grade.
    """
    if isinstance(failure, RolloutData):
        return _grade_cause(failure)
    return _exception_cause(failure)


def trial_outcome(result: RolloutData | Exception) -> Outcome:
    """The outcome of one engine run: what ``ShellboxRolloutEngine.run`` returned or raised.

    An agent deadline is a normal ending: RolloutEngine grades the state the agent left, and the
    trial is ``Graded`` with that grade and ``timed_out`` set, so timed-out trials stay in the
    denominator. A deadline that expired before the first response leaves nothing to grade
    (``UNAVAILABLE``), which is ``Ungraded(AGENT_TIMEOUT)``.
    """
    if isinstance(result, RolloutData):
        if result.grade.status in GRADED_STATUSES:
            return Graded(result)
        return Ungraded(classify(result), result.grade.error or str(result.grade.status), result)
    cause = classify(result)
    detail = "".join(traceback.format_exception(result))
    if not isinstance(result, RolloutInterrupted):
        return Ungraded(cause, detail, None)
    return Ungraded(cause, detail, result.rollout)


def _exception_cause(error: BaseException) -> Cause:
    if isinstance(error, RolloutContractError):
        return Cause.TOKEN_CONTRACT
    if isinstance(error, GenerationLimitReached | GlmContextExhausted):
        return Cause.GENERATION_LIMIT
    if isinstance(error, GlmUnavailable):
        return Cause.MODEL_UNAVAILABLE
    if isinstance(error, GlmRequestRejected):
        return Cause.MODEL_REJECTED
    if isinstance(error, UnsupportedMachineSpec):
        return Cause.MACHINE_UNSUPPORTED
    if isinstance(error, RolloutInterrupted):
        return _interruption_cause(error)
    return Cause.UNCLASSIFIED


def _matches_rolloutengine_setup_message(error: BaseException | None) -> bool:
    """Whether ``error`` is a task setup, healthcheck or stage-setup failure from RolloutEngine.

    ``rolloutengine.machines`` and ``rolloutengine.task_session`` raise these as a bare
    ``RuntimeError`` or ``TimeoutError``, the same types as machine failures, so this matches
    message prefixes. Replace it with the typed errors requested in #9782 (rolloutengine typed
    setup errors) once they land.
    """
    return isinstance(error, RuntimeError | TimeoutError) and str(error).startswith(ROLLOUTENGINE_SETUP_MESSAGE_PREFIXES)


def _is_setup_failure(error: BaseException | None, operation: RolloutOperation) -> bool:
    if operation is RolloutOperation.PREPARE and isinstance(error, ValueError):
        return True  # _ShellboxTaskSession.prepare: the environment lacks the task's required capabilities
    return _matches_rolloutengine_setup_message(error)


def _interruption_cause(error: RolloutInterrupted) -> Cause:
    operation = error.operation
    if operation is RolloutOperation.ATTEMPT:
        return Cause.ATTEMPT_TIMEOUT
    if operation is RolloutOperation.CLEANUP:
        return Cause.CLEANUP
    original = error.__cause__
    if original is not None:
        cause = _exception_cause(original)
        if cause is not Cause.UNCLASSIFIED:
            return cause
    timed_out = isinstance(original, TimeoutError)
    if operation in (RolloutOperation.START, RolloutOperation.PREPARE) and _is_setup_failure(original, operation):
        return Cause.TASK_SETUP
    if operation is RolloutOperation.START:
        return Cause.MACHINE_START_TIMEOUT if timed_out else Cause.MACHINE_START
    if operation is RolloutOperation.PREPARE:
        return Cause.SESSION_PREPARE
    if operation is RolloutOperation.ADVANCE:
        return Cause.TOOL_EXECUTION
    if operation is RolloutOperation.GRADE:
        return Cause.GRADER_RAISED
    return Cause.UNCLASSIFIED


def _grade_cause(rollout: RolloutData) -> Cause:
    grade = rollout.grade
    if grade.status in GRADED_STATUSES:
        raise ValueError(f"A {grade.status} rollout has no failure cause")
    if grade.status is GradeStatus.INFRA_ERROR:
        return Cause.GRADER_INFRA if grade.failure is None else GRADING_FAILURES[grade.failure]
    if grade.status is GradeStatus.SKIPPED:
        return Cause.VERIFIER_SKIPPED
    if grade.status is GradeStatus.INVALID_TASK:
        return Cause.INVALID_TASK
    if rollout.stop_reason == AGENT_TIMEOUT_STOP_REASON:
        return Cause.AGENT_TIMEOUT
    if rollout.stop_reason == LENGTH_STOP_REASON and not rollout.steps:
        return Cause.GENERATION_LIMIT
    return Cause.NO_GRADE
