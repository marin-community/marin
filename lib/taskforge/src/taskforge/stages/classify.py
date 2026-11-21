# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0


"""Classify completed grades and interrupted trials without discarding evidence."""

import math
from types import MappingProxyType

from rolloutengine.models import (
    LENGTH_STOP_REASON,
    MAX_TURNS_STOP_REASON,
    TOTAL_TURN_TIMEOUT_STOP_REASON,
    GradedRollout,
)
from taskcompendium.grading_result import GradeResult, GradingFailure, Outcome
from taskcompendium.models import NoGrader, TaskSpec

from taskforge.effects.events import Resolved
from taskforge.effects.issues import Hold
from taskforge.model.review import Cause, Incomplete, Modify, Reject, Retry
from taskforge.model.verdict import grader_checks

# One review disposition per cause; solver generation limits are handled as failed trials.
CAUSE_DECISIONS = MappingProxyType(
    {
        None: None,
        Cause.AGENT_TIMEOUT: Retry,
        Cause.GRADER_INFRA: Retry,
        Cause.NO_GRADE: Retry,
        Cause.BUDGET_EXHAUSTED: Reject,
        Cause.GENERATION_LIMIT: Incomplete,
        Cause.GRADER_MISSING_REWARD: Modify,
        Cause.GRADER_EMPTY_REWARD: Modify,
        Cause.GRADER_INVALID_REWARD: Modify,
        Cause.GRADER_INVALID_CHECKS: Modify,
        Cause.GRADER_TIMEOUT: Modify,
        Cause.GRADER_EXECUTION: Modify,
        Cause.GRADER_REWARD_RANGE: Modify,
        Cause.INVALID_TASK: Modify,
        Cause.VERIFIER_SKIPPED: Modify,
    }
)


def grade_cause(grade: GradeResult, task: TaskSpec | None = None) -> Cause | None:
    try:
        grader_checks(grade)
    except ValueError:
        return Cause.GRADER_INVALID_CHECKS
    if grade.status is Outcome.SUBMISSION_FAILURE:
        return None
    if grade.status is Outcome.GRADED:
        if grade.reward is None:
            return Cause.GRADER_MISSING_REWARD
        if not math.isfinite(grade.reward):
            return Cause.GRADER_INVALID_REWARD
        return None if grade.score_min <= grade.reward <= grade.score_max else Cause.GRADER_REWARD_RANGE
    if grade.status is Outcome.INVALID_TASK:
        return Cause.INVALID_TASK
    if grade.status is Outcome.UNAVAILABLE:
        return Cause.VERIFIER_SKIPPED if task is not None and isinstance(task.grader, NoGrader) else Cause.NO_GRADE
    return {
        GradingFailure.GRADER_TIMEOUT: Cause.GRADER_TIMEOUT,
        GradingFailure.TIMEOUT: Cause.GRADER_INFRA,
        GradingFailure.MISSING_REWARD: Cause.GRADER_MISSING_REWARD,
        GradingFailure.EMPTY_REWARD: Cause.GRADER_EMPTY_REWARD,
        GradingFailure.INVALID_REWARD: Cause.GRADER_INVALID_REWARD,
        GradingFailure.GRADER_EXIT: Cause.GRADER_EXECUTION,
        GradingFailure.EXECUTION: Cause.GRADER_INFRA,
        None: Cause.GRADER_INFRA,
    }[grade.failure]


def trial_cause(result: GradedRollout, task: TaskSpec) -> Cause | None:
    stop = result.rollout.stop_reason
    if stop == TOTAL_TURN_TIMEOUT_STOP_REASON:
        return Cause.AGENT_TIMEOUT
    if stop == MAX_TURNS_STOP_REASON:
        return Cause.BUDGET_EXHAUSTED
    if stop in (LENGTH_STOP_REASON, "context"):
        return Cause.GENERATION_LIMIT
    return grade_cause(result.grade, task)


async def grader_infra_unavailable(event: Resolved[object]) -> None:
    value = event.value
    grade = value.grade if isinstance(value, GradedRollout) else value
    if isinstance(grade, GradeResult) and grade_cause(grade) is Cause.GRADER_INFRA:
        raise Hold(grade.error or "Grader infrastructure unavailable")


def replayable_grade(event: Resolved[object]) -> bool:
    """Resume successful evidence, but reevaluate unavailable infrastructure."""
    value = event.value
    grade = value.grade if isinstance(value, GradedRollout) else value
    return not isinstance(grade, GradeResult) or grade_cause(grade) not in (Cause.GRADER_INFRA, Cause.NO_GRADE)
