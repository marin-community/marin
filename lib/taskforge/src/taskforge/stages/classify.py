# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0


"""Classify completed grades and interrupted trials without discarding evidence."""

import math
from collections.abc import Callable, Sequence
from types import MappingProxyType

from rolloutengine.models import MAX_TURNS_STOP_REASON, TOTAL_TURN_TIMEOUT_STOP_REASON, GradedRollout
from taskcompendium.grading_result import GradeResult, GradingFailure, Outcome
from taskcompendium.models import NoGrader, TaskSpec

from taskforge.effects.events import Resolved, StageBoundary
from taskforge.effects.issues import Hold, LogFailure
from taskforge.effects.log import Record, physical
from taskforge.model.review import FAULTS, Cause, Incomplete, Modify, Reject, Retry
from taskforge.model.verdict import grader_checks
from taskforge.resources.rollouts import LIMIT_STOP_REASONS, TIME_LIMIT_STOP_REASON, SolverRollout, refused

# One review disposition per cause. GENERATION_LIMIT and TIME_LIMIT have none: such a trial is unsolved
# (``TrialResult.outcome``). GRADER_INFRA has none: the grader_infra_unavailable observer holds validation before
# review sees such a grade.
CAUSE_DECISIONS = MappingProxyType(
    {
        None: None,
        Cause.AGENT_TIMEOUT: Retry,
        Cause.NO_GRADE: Retry,
        Cause.BUDGET_EXHAUSTED: Reject,
        Cause.HELD: Incomplete,
        **dict.fromkeys(FAULTS, Modify),  # the stock policy returns every fault to the author
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
    if refused(result):
        return Cause.HELD
    if stop == TOTAL_TURN_TIMEOUT_STOP_REASON:
        return Cause.AGENT_TIMEOUT
    if stop == MAX_TURNS_STOP_REASON:
        return Cause.BUDGET_EXHAUSTED
    if stop == TIME_LIMIT_STOP_REASON:
        return Cause.TIME_LIMIT
    return Cause.GENERATION_LIMIT if stop in LIMIT_STOP_REASONS else grade_cause(result.grade, task)


def unsettled_grade(value: object) -> GradeResult | None:
    """A grade that can still hold validation or reopen its slot. A generation limit settles a loss on its own."""
    if isinstance(value, GradedRollout):
        value = None if value.rollout.stop_reason in LIMIT_STOP_REASONS else value.grade
    return value if isinstance(value, GradeResult) else None


async def grader_infra_unavailable(event: Resolved[object]) -> None:
    grade = unsettled_grade(event.value)
    if grade is not None and grade_cause(grade) is Cause.GRADER_INFRA:
        raise Hold(grade.error or "Grader infrastructure unavailable")


def replayable_grade(event: Resolved[object]) -> bool:
    """Resume successful evidence, settled losses and slot holds, but reevaluate unavailable grading."""
    grade = unsettled_grade(event.value)
    return refused(event.value) or grade is None or grade_cause(grade) not in (Cause.GRADER_INFRA, Cause.NO_GRADE)


def replaced_refusals(previous: Sequence[Record], resume: bool) -> Callable[[Resolved[object]], bool]:
    """The slot holds an earlier launch recorded, which an explicit resume runs again: each replacement invalidates the
    refusal it replaces as it starts, so a later launch resumes the replacement and still replays any other refusal."""
    stale = {id(record.event) for record in previous if resume and isinstance(record.event, Resolved)}
    return lambda event: id(event) in stale and refused(event.value)


def boundary(records: Sequence[Record]) -> StageBoundary | None:
    """The journal's latest stage boundary, after a best-effort check for accidental corruption: each trial receipt in a
    boundary names an earlier physical solver resolution of that trial's slot. A deliberately rewritten journal is out
    of scope; reading a transcript at its receipt checks the full request again, and a mismatch there holds."""
    found = None
    for record in records:
        if not isinstance(event := record.event, StageBoundary):
            continue
        for trial in (trial for result in event.workpiece.evidence for trial in result.solver):
            row = None if trial.receipt is None else physical(records, trial.receipt, record.sequence)
            request = None if row is None else row.request
            if trial.receipt is not None and not (isinstance(request, SolverRollout) and request.trial == trial.trial):
                raise LogFailure(f"Row {record.sequence}: trial {trial.trial}'s receipt names no resolution of its slot")
        found = event
    return found
