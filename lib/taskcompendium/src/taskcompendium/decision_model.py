# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade a ``decision_model`` grader by the level that one caller-supplied read returns for the final answer."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from taskcompendium.grading import answer_submission
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import (
    DECISION_LEVELS,
    DECISION_QUESTION,
    ActionSubmission,
    DecisionModelGrader,
    GradingAttempt,
    SubmissionFailure,
    TaskSpec,
    TextMessage,
    TextSubmission,
)
from taskcompendium.submission import is_ejection, native_action_text, require_submission_compatibility


@dataclass(frozen=True)
class DecisionModelRequest:
    """Everything that determines one read's level, so a caller may key a recorded read by it."""

    model: str
    question: str
    levels: tuple[str, ...]
    deliverable: str
    rubric: str
    reference: str
    candidate: str


@dataclass(frozen=True)
class DecisionModelConnection:
    """The caller's decision model, which owns its endpoint, health, retries, concurrency and timeouts.

    ``model`` must identify the deployed revision and any read setting that changes its levels. ``read`` returns an
    index into ``request.levels`` or an expected level between two indices; what it raises reaches the caller.
    """

    model: str
    read: Callable[[DecisionModelRequest], Awaitable[float]]


async def grade_decision_model_answer(
    task: TaskSpec, attempt: GradingAttempt, connection: DecisionModelConnection
) -> GradeResult:
    """Score the final answer 1 when its level reaches the grader's cut, else 0; an off-scale level is infra_error."""
    grader = task.grader
    if not isinstance(grader, DecisionModelGrader):
        raise TypeError(f"Decision-model grading requires a decision_model grader, not {grader.kind}")
    require_submission_compatibility(task)
    try:
        submission = answer_submission(task, attempt)
    except SubmissionFailure as error:
        return GradeResult(Outcome.SUBMISSION_FAILURE, 0.0, str(error))
    if is_ejection(submission):
        return GradeResult(Outcome.GRADED, 0.0, detail={})
    assert isinstance(submission, TextSubmission | ActionSubmission)
    if isinstance(submission, TextSubmission):
        text, candidate = submission.value, submission.value
    else:
        text = submission.message.content if isinstance(submission.message, TextMessage) else None
        candidate = native_action_text(submission.message)
    if text is not None and not text.strip():
        return GradeResult(Outcome.GRADED, 0.0, detail={"reason": "empty_candidate"})
    request = DecisionModelRequest(
        connection.model,
        DECISION_QUESTION,
        DECISION_LEVELS,
        grader.deliverable,
        grader.rubric,
        grader.reference,
        candidate,
    )
    level = await connection.read(request)
    top = len(DECISION_LEVELS) - 1
    if isinstance(level, bool) or not isinstance(level, int | float) or not 0 <= level <= top:
        small = isinstance(level, float) or (isinstance(level, int) and abs(level) < 10**6)
        shown = repr(level) if small else f"a value of type {type(level).__name__}"
        return GradeResult(
            Outcome.INFRA_ERROR, None, f"Decision model {connection.model!r} returned {shown}, not a level in 0..{top}"
        )
    return GradeResult(
        Outcome.GRADED,
        1.0 if level >= grader.cut else 0.0,
        detail={"decision_model": {"model": connection.model, "level": level, "cut": grader.cut}},
    )
