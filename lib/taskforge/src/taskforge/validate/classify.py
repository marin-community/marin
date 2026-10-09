# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The one classifier from trial failures to ``Cause``, and the trial outcome built on it.

A failure is either an exception (a ``RolloutInterrupted`` from the engine with the original error
as its ``__cause__``, a ``RolloutContractError``, a ``GlmClient`` error, a shellbox error such as
``MachineTerminated``) or a
rollout the engine returned without a usable grade. A recognized original error wins over the
interrupted operation, so a ``GlmUnavailable`` during ``MODEL`` is ``MODEL_UNAVAILABLE``. Anything
no rule matches is ``UNCLASSIFIED``, which ``Evidence`` counts like any other cause.

A rollout with a grade can still have a cause. A task without a grader (``NoGrader``) grades
``UNAVAILABLE``, which is ``VERIFIER_SKIPPED``. A verifyit ``pytest`` grader scores a candidate whose
own code fails to import or be collected as reward 0 (verifyit #9923); that is
``CANDIDATE_CODE_ERROR``, not a wrong answer, so the classifier needs the task.

Task setup failures are told apart from machine failures only by their message text
(``_matches_rolloutengine_setup_message``), because the engine raises a bare ``RuntimeError`` or
``TimeoutError`` for both. A machine that ended under the engine is typed (``MachineTerminated``)
and is ``MACHINE_TERMINATED`` whichever operation it interrupted.
"""

import traceback

from rolloutengine.contracts import (
    LENGTH_STOP_REASON,
    TOTAL_TURN_TIMEOUT_STOP_REASON,
    GenerationLimitReached,
    RolloutContractError,
    RolloutData,
    RolloutInterrupted,
    RolloutOperation,
)
from shellbox.machine import MachineTerminated, UnsupportedMachineSpec
from taskcompendium.grading_result import GradeResult, GradingFailure
from taskcompendium.grading_result import Outcome as GradeStatus
from taskcompendium.models import NoGrader, TaskSpec, VerifyitGrader
from verifyit.spec import Mode

from taskforge.llm.client import GlmContextExhausted, GlmRequestRejected, GlmUnavailable
from taskforge.validate.outcome import GRADED_STATUSES, Cause, Graded, Outcome, Ungraded

GRADING_FAILURES = {
    GradingFailure.TIMEOUT: Cause.GRADER_TIMEOUT,
    GradingFailure.MISSING_REWARD: Cause.GRADER_MISSING_REWARD,
    GradingFailure.EMPTY_REWARD: Cause.GRADER_EMPTY_REWARD,
    GradingFailure.INVALID_REWARD: Cause.GRADER_INVALID_REWARD,
    GradingFailure.EXECUTION: Cause.GRADER_EXECUTION,
}
ROLLOUTENGINE_SETUP_MESSAGE_PREFIXES = ("Environment setup command ",)
"""Message prefixes of the errors RolloutEngine raises when a task's setup command fails or times out."""
CANDIDATE_CODE_REASONS = frozenset({"startup_error", "collection_error"})
"""The ``reason`` values in a verifyit pytest verdict's detail when the candidate's code did not load."""


def classify(failure: BaseException | RolloutData, task: TaskSpec) -> Cause:
    """Map an exception, or a rollout of ``task`` returned without a usable grade, to its ``Cause``.

    Raises:
        ValueError: ``failure`` is a rollout with a usable grade.
    """
    if isinstance(failure, RolloutData):
        return _grade_cause(failure, task)
    return _exception_cause(failure)


def candidate_code_error(grade: GradeResult, task: TaskSpec) -> bool:
    """Whether a verifyit pytest grader gave reward 0 because the candidate's own code did not load.

    Since #9923 verifyit scores a candidate's startup or collection error as a failed attempt with
    reward 0. The attribution arguably belongs upstream in verifyit: it should surface it as a
    documented field of ``GradeResult.detail`` (today it is an undocumented ``reason`` key, with
    ``category`` "agent", that TaskCompendium copies through), so that every consumer can tell a
    candidate that did not load from a wrong one. Taskforge reads the key locally for now because a
    curation run must not count a broken scaffold or a solver's missing dependency as a wrong answer,
    while RL training upstream wants exactly the reward 0 it gets.
    """
    grader = task.grader
    return (
        isinstance(grader, VerifyitGrader)
        and grader.mode == Mode.PYTEST.value
        and grade.status is GradeStatus.GRADED
        and grade.reward == 0.0
        and grade.detail is not None
        and grade.detail.get("reason") in CANDIDATE_CODE_REASONS
    )


def trial_outcome(result: RolloutData | Exception, task: TaskSpec) -> Outcome:
    """The outcome of one engine run: what ``ShellboxRolloutEngine.run`` returned or raised.

    A total-turn deadline is a normal ending: RolloutEngine grades the state the agent left, and the
    trial is ``Graded`` with that grade and ``timed_out`` set, so timed-out trials stay in the
    denominator. A deadline that expired before the first response leaves nothing to grade
    (``UNAVAILABLE``), which is ``Ungraded(AGENT_TIMEOUT)``.
    """
    if isinstance(result, RolloutData):
        if result.grade.status in GRADED_STATUSES and not candidate_code_error(result.grade, task):
            return Graded(result)
        return Ungraded(classify(result, task), result.grade.error or str(result.grade.status), result)
    cause = classify(result, task)
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
    if isinstance(error, MachineTerminated):
        return Cause.MACHINE_TERMINATED
    if isinstance(error, RolloutInterrupted):
        return _interruption_cause(error)
    return Cause.UNCLASSIFIED


def _matches_rolloutengine_setup_message(error: BaseException | None) -> bool:
    """Whether ``error`` is a task setup failure from RolloutEngine.

    RolloutEngine's environment setup raises these as a bare
    ``RuntimeError`` or ``TimeoutError``, the same types as machine failures, so this matches
    message prefixes. Replace it with the typed errors requested in #9782 once #9799 (rolloutengine
    typed setup failures) lands.
    """
    return isinstance(error, RuntimeError | TimeoutError) and str(error).startswith(ROLLOUTENGINE_SETUP_MESSAGE_PREFIXES)


def _is_setup_failure(error: BaseException | None, operation: RolloutOperation) -> bool:
    if operation is RolloutOperation.PREPARE and isinstance(error, ValueError):
        return True  # the default task session's prepare: the environment lacks the task's required capabilities
    return _matches_rolloutengine_setup_message(error)


def _interruption_cause(error: RolloutInterrupted) -> Cause:
    operation = error.operation
    if operation is RolloutOperation.ATTEMPT:
        return Cause.ATTEMPT_TIMEOUT
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
    if operation is RolloutOperation.MODEL and timed_out:
        return Cause.MODEL_TIMEOUT
    if operation is RolloutOperation.ADVANCE:
        return Cause.TOOL_EXECUTION
    if operation is RolloutOperation.GRADE:
        return Cause.GRADER_TIMEOUT if timed_out else Cause.GRADER_RAISED
    return Cause.UNCLASSIFIED


def _grade_cause(rollout: RolloutData, task: TaskSpec) -> Cause:
    grade = rollout.grade
    if candidate_code_error(grade, task):
        return Cause.CANDIDATE_CODE_ERROR
    if grade.status in GRADED_STATUSES:
        raise ValueError(f"A {grade.status} rollout has no failure cause")
    if grade.status is GradeStatus.INFRA_ERROR:
        return Cause.GRADER_INFRA if grade.failure is None else GRADING_FAILURES[grade.failure]
    if grade.status is GradeStatus.UNAVAILABLE and isinstance(task.grader, NoGrader):
        return Cause.VERIFIER_SKIPPED
    if grade.status is GradeStatus.INVALID_TASK:
        return Cause.INVALID_TASK
    if rollout.stop_reason == TOTAL_TURN_TIMEOUT_STOP_REASON:
        return Cause.AGENT_TIMEOUT
    if rollout.stop_reason == LENGTH_STOP_REASON and not rollout.steps:
        return Cause.GENERATION_LIMIT
    return Cause.NO_GRADE
