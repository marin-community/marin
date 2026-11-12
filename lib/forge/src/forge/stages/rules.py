# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0


"""Review complete evidence before accepting or requesting author repair."""

import math
from collections import Counter
from dataclasses import replace

from taskcompendium.grading_result import GradeResult, Outcome

from forge.model.review import (
    Accept,
    BandOutcome,
    BandRules,
    Budget,
    Cause,
    Decision,
    Incomplete,
    Modify,
    Reject,
    RejectKind,
    Retry,
    Revise,
    TrialResult,
    ValidationResult,
)
from forge.model.workpiece import Workpiece
from forge.stages.classify import CAUSE_DECISIONS


def passed(grade: GradeResult) -> bool:
    if grade.status is Outcome.SUBMISSION_FAILURE:
        return False
    if grade.passed is not None:
        return grade.passed
    reward, score_max = grade.reward or 0.0, grade.score_max
    return reward >= score_max or math.isclose(reward, score_max, rel_tol=1e-9, abs_tol=1e-9)


def modify_rounds(workpiece: Workpiece) -> int:
    return sum(isinstance(decision, Modify) for decision in workpiece.decisions)


def tagged(band: BandOutcome) -> tuple[str, ...]:
    return () if band is BandOutcome.IN_BAND else (f"accepted_with: {band}",)


def default_review(workpiece: Workpiece, band: BandRules, budget: Budget) -> Decision:
    """Mark's policy: band placement tags a task and never abandons it.

    A too-easy task gets one hardening round and a too-hard task easing rounds within the shared modification budget;
    a task still out of band afterwards is accepted with its placement. A revision that regresses (a defect, or a
    placement other than the fallback's) reverts to the last clean submission and accepts it with that placement.
    An acceptance with rollout-reviewer suggestions becomes ``Revise``; after that revision placement is final.
    """
    result = workpiece.evidence[-1]
    decision = assess(result, band)
    if isinstance(decision, Incomplete):
        retryable = [
            t.cause or (Cause.NO_GRADE if t.grade is None else None) for t in (*result.solver, *result.controls)
        ]
        retries = sum(isinstance(item, Retry) for item in workpiece.decisions)
        cause = next((cause for cause in retryable if CAUSE_DECISIONS.get(cause) is Retry), None)
        if cause is not None and retries < budget.max_validation_retries:
            return Retry(cause, retries + 1)
    if isinstance(decision, Accept):
        return reviewed(decision, result)
    if not isinstance(decision, Modify):
        return decision
    placement, fallback = decision.placement, workpiece.fallback
    if fallback is not None and placement is not fallback.band:
        reasons = ("Revision regressed; reverted to the previously validated submission", *decision.findings)
        return Accept(fallback.band, (*tagged(fallback.band), *reasons), revert=True)
    if placement is None:
        return decision
    if any(isinstance(item, Revise) for item in workpiece.decisions):
        reason = "Placement is final after a reviewer revision"
    elif placement is BandOutcome.TOO_EASY and fallback is not None:
        reason = "One hardening round used"
    elif modify_rounds(workpiece) >= budget.max_modify_rounds:
        reason = "Shared modification budget exhausted"
    else:
        return decision
    return reviewed(Accept(placement, (*tagged(placement), reason)), result)


def reviewed(decision: Accept, result: ValidationResult) -> Accept | Revise:
    """An acceptance the rollout reviewer suggested revisions for goes back to the author once, as suggestions."""
    review = result.reviewer
    if review is None or decision.revert:
        return decision
    if review.suggestions:
        return Revise(decision.band, review.suggestions, decision.reasons)
    return replace(decision, reasons=(*decision.reasons, *review.reasons))


def assess(result: ValidationResult, band: BandRules) -> Decision:
    if result.lowering_error is not None:
        decision = Modify if result.lowering_cause is Cause.INVALID_TASK else Incomplete
        return decision((f"Lowering failed: {result.lowering_error}",))
    if not result.solver or not result.controls:
        return Incomplete(("Required solver or control evidence is absent",))
    trials = (*result.solver, *result.controls, *(f for a in result.adversary for f in a.findings))
    causes = Counter(
        t.cause or (Cause.NO_GRADE if t.grade is None else None)
        for t in trials
        if not (isinstance(t, TrialResult) and t.cause is Cause.GENERATION_LIMIT)
    )
    unavailable = tuple(
        f"{count} ungraded trial(s): {cause}"
        for cause, count in sorted((cause, count) for cause, count in causes.items() if cause is not None)
        if CAUSE_DECISIONS[cause] not in (Modify, Reject)
    )
    unavailable += tuple(
        f"{trial.name}: {reason}"
        for trial in result.adversary
        if not trial.completed
        for reason in trial.reasons or ("Adversary search is incomplete (reason unavailable)",)
    )
    if unavailable:
        return Incomplete(unavailable)
    if causes[Cause.BUDGET_EXHAUSTED]:
        return Reject(RejectKind.BUDGET, ("An explicitly configured trial budget was exhausted",))
    findings = tuple(
        f"Control {control.index} violates its expected reward" for control in result.controls if not control.met
    )
    findings += tuple(
        f"{count} ungraded trial(s): {cause}" for cause, count in causes.items() if CAUSE_DECISIONS.get(cause) is Modify
    )
    findings += tuple(
        f"{trial.grade.error or trial.cause}\n{trial.grade.diagnostics.get('stderr', '')}"
        for trial in trials
        if CAUSE_DECISIONS.get(trial.cause) is Modify and trial.grade is not None
    )
    findings += tuple(
        f"{trial.name}: {finding.claim}"
        for trial in result.adversary
        for finding in trial.findings
        if finding.met or CAUSE_DECISIONS.get(finding.cause) is Modify
    )
    if findings:
        return Modify(findings)
    solved = sum(
        passed(trial.grade)
        for trial in result.solver
        if trial.grade is not None and trial.cause is not Cause.GENERATION_LIMIT
    )
    place = band.place(solved / len(result.solver))
    if place is BandOutcome.IN_BAND:
        return Accept(place)
    finding = f"{solved} of {len(result.solver)} trials solved: {place}. " + (
        "Make the task harder while preserving correct controls and grading. This is the one hardening attempt."
        if place is BandOutcome.TOO_EASY
        else "Make the task more tractable while preserving its objective, correct controls and grading."
    )
    return Modify((finding,), placement=place)
