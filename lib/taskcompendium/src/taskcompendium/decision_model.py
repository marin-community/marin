# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade a ``decision_model`` grader through a caller-supplied decision-model read.

The caller owns the decision model: its endpoint, health, retries and concurrency. This module builds the read
request, validates the returned level and applies the author's cut. It shares nothing with the LLM judge path.
"""

import math
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
    TextSubmission,
)
from taskcompendium.submission import native_action_text, require_submission_compatibility


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
    """The caller's decision model, kept outside serialized task specifications.

    ``model`` names the model and version that ``read`` reaches. ``read`` returns a level on ``request.levels``:
    an index, or an expected level between two indices.
    """

    model: str
    read: Callable[[DecisionModelRequest], Awaitable[float]]


async def grade_decision_model_answer(
    task: TaskSpec, attempt: GradingAttempt, connection: DecisionModelConnection
) -> GradeResult:
    """Score the final answer 1 when its level reaches the grader's cut, else 0.

    A read that raises or returns no level on ``DECISION_LEVELS`` is an infrastructure error, never a score.
    """
    grader = task.grader
    if not isinstance(grader, DecisionModelGrader):
        raise TypeError(f"Decision-model grading requires a decision_model grader, not {grader.kind}")
    require_submission_compatibility(task)
    try:
        submission = answer_submission(task, attempt)
    except SubmissionFailure as error:
        return GradeResult(Outcome.SUBMISSION_FAILURE, 0.0, str(error))
    if isinstance(submission, TextSubmission):
        candidate = submission.value
    elif isinstance(submission, ActionSubmission):
        candidate = native_action_text(submission.message)
    else:
        return GradeResult(Outcome.INVALID_TASK, None, f"A decision model reads text, not a {type(submission).__name__}")
    if not candidate.strip():
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
    try:
        level = await connection.read(request)
    except Exception as error:
        # The caller's read owns retries and outage policy; whatever it raises leaves the answer ungraded.
        return GradeResult(
            Outcome.INFRA_ERROR,
            None,
            f"Decision model {connection.model!r} read failed: {type(error).__name__}: {error}",
        )
    top = len(DECISION_LEVELS) - 1
    if isinstance(level, bool) or not isinstance(level, int | float) or not (math.isfinite(level) and 0 <= level <= top):
        return GradeResult(
            Outcome.INFRA_ERROR, None, f"Decision model {connection.model!r} returned {level!r}, not a level in 0..{top}"
        )
    return GradeResult(
        Outcome.GRADED,
        1.0 if level >= grader.cut else 0.0,
        detail={"decision_model": {"model": connection.model, "level": level, "cut": grader.cut}},
    )
