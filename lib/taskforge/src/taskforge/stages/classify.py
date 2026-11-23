# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0


"""Classify completed grades and interrupted trials without discarding evidence."""

import math
from collections.abc import Callable, Sequence
from types import MappingProxyType

from rolloutengine.models import (
    LENGTH_STOP_REASON,
    MAX_TURNS_STOP_REASON,
    TOTAL_TURN_TIMEOUT_STOP_REASON,
    GradedRollout,
)
from taskcompendium.grading_result import GradeResult, GradingFailure, Outcome
from taskcompendium.models import NoGrader, TaskSpec

from taskforge.effects.events import Committed, Resolved, StageCompleted
from taskforge.effects.issues import Hold, LogFailure
from taskforge.effects.log import Record, physical
from taskforge.model.review import Cause, Incomplete, Modify, Reject, Retry
from taskforge.model.verdict import grader_checks
from taskforge.resources.rollouts import SolverRollout, refused

# One review disposition per cause; solver generation limits are handled as failed trials. GRADER_INFRA
# has none: the grader_infra_unavailable observer holds validation before review sees such a grade.
CAUSE_DECISIONS = MappingProxyType(
    {
        None: None,
        Cause.AGENT_TIMEOUT: Retry,
        Cause.NO_GRADE: Retry,
        Cause.BUDGET_EXHAUSTED: Reject,
        Cause.GENERATION_LIMIT: Incomplete,
        Cause.HELD: Incomplete,
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
    if refused(result):
        return Cause.HELD
    if stop == TOTAL_TURN_TIMEOUT_STOP_REASON:
        return Cause.AGENT_TIMEOUT
    if stop == MAX_TURNS_STOP_REASON:
        return Cause.BUDGET_EXHAUSTED
    if generation_limited(result):
        return Cause.GENERATION_LIMIT
    return grade_cause(result.grade, task)


def generation_limited(result: GradedRollout) -> bool:
    """The solver's generation reached its limit: a definitive failed trial whatever the grade says."""
    return result.rollout.stop_reason in (LENGTH_STOP_REASON, "context")


async def grader_infra_unavailable(event: Resolved[object]) -> None:
    value = event.value
    grade = value.grade if isinstance(value, GradedRollout) else value
    if isinstance(grade, GradeResult) and grade_cause(grade) is Cause.GRADER_INFRA:
        raise Hold(grade.error or "Grader infrastructure unavailable")


def replayable_grade(event: Resolved[object]) -> bool:
    """Resume successful evidence, generation cutoffs and slot holds, but reevaluate unavailable grading. A cutoff is
    exempt only without a grade (no answer reached the grader); a graded answer cut at its limit whose grader failed
    is retried like any grader infrastructure failure."""
    value = event.value
    if isinstance(value, GradedRollout) and generation_limited(value) and grade_cause(value.grade) is Cause.NO_GRADE:
        return True
    grade = value.grade if isinstance(value, GradedRollout) else value
    unavailable = isinstance(grade, GradeResult) and grade_cause(grade) in (Cause.GRADER_INFRA, Cause.NO_GRADE)
    return refused(event.value) or not unavailable


def replaced_refusals(previous: Sequence[Record], resume: bool) -> Callable[[Resolved[object]], bool]:
    """The slot holds an earlier launch recorded, which an explicit resume runs again: each replacement invalidates the
    refusal it replaces as it starts, so a later launch resumes the replacement and still replays any other refusal."""
    stale = {id(record.event) for record in previous if resume and isinstance(record.event, Resolved)}
    return lambda event: id(event) in stale and refused(event.value)


def commitments(records: Sequence[Record]) -> dict[str, Committed]:
    """The journal's committed visits by identity, after a best-effort check for accidental corruption: each committed
    trial's receipt names an earlier physical solver resolution of that trial's slot. A deliberately rewritten journal is
    out of scope; reading a transcript at its receipt checks the full request again, and a mismatch there holds."""
    found: dict[str, Committed] = {}
    for record in records:
        if not isinstance(marker := record.event, StageCompleted) or (commit := marker.committed) is None:
            continue
        for trial in commit.result.solver:
            row = None if trial.receipt is None else physical(records, trial.receipt, record.sequence)
            request = None if row is None else row.request
            if trial.receipt is not None and not (isinstance(request, SolverRollout) and request.trial == trial.trial):
                raise LogFailure(f"Row {record.sequence}: trial {trial.trial}'s receipt names no resolution of its slot")
        found[commit.visit] = commit
    return found
