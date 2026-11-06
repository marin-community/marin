# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0


"""Classify completed grades and interrupted trials without discarding evidence."""

import math

from rolloutengine.models import (
    LENGTH_STOP_REASON,
    MAX_TURNS_STOP_REASON,
    TOTAL_TURN_TIMEOUT_STOP_REASON,
    GradedRollout,
)
from taskcompendium.grading_result import GradeResult, GradingFailure, Outcome
from taskcompendium.models import NoGrader, TaskSpec

from forge.effects.events import Resolved
from forge.effects.issues import Hold
from forge.model.review import Cause
from forge.model.verdict import grader_checks
from forge.resources.grade_health import check_timeout

TASK_DEFECTS = frozenset(
    {
        Cause.TASK_SETUP,
        Cause.SUBMISSION_UNSUPPORTED,
        Cause.GRADER_MISSING_REWARD,
        Cause.GRADER_EMPTY_REWARD,
        Cause.GRADER_INVALID_REWARD,
        Cause.GRADER_INVALID_CHECKS,
        Cause.GRADER_TIMEOUT,
        Cause.GRADER_REWARD_RANGE,
        Cause.INVALID_TASK,
        Cause.VERIFIER_SKIPPED,
    }
)


def grade_cause(grade: GradeResult, task: TaskSpec | None = None) -> Cause | None:
    check_timeout(grade, task)
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
        GradingFailure.TIMEOUT: Cause.GRADER_TIMEOUT,
        GradingFailure.MISSING_REWARD: Cause.GRADER_MISSING_REWARD,
        GradingFailure.EMPTY_REWARD: Cause.GRADER_EMPTY_REWARD,
        GradingFailure.INVALID_REWARD: Cause.GRADER_INVALID_REWARD,
        GradingFailure.EXECUTION: Cause.GRADER_EXECUTION,
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


async def host_failure(event: Resolved[object]) -> None:
    value = event.value
    grade = value.grade if isinstance(value, GradedRollout) else value
    if isinstance(grade, GradeResult) and grade_cause(grade) is Cause.GRADER_INFRA:
        raise Hold(grade.error or "Grader infrastructure unavailable")
