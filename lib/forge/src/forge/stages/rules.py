# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0


"""Review complete evidence before accepting or requesting author repair."""

import math
from collections import Counter

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
    ValidationResult,
)
from forge.model.workpiece import Workpiece
from forge.stages.classify import TASK_DEFECTS


def passed(grade: GradeResult) -> bool:
    if grade.status is Outcome.SUBMISSION_FAILURE:
        return False
    if grade.passed is not None:
        return grade.passed
    reward, score_max = grade.reward or 0.0, grade.score_max
    return reward >= score_max or math.isclose(reward, score_max, rel_tol=1e-9, abs_tol=1e-9)


TRANSIENT_CAUSES = frozenset({Cause.AGENT_TIMEOUT, Cause.GRADER_INFRA, Cause.NO_GRADE})


def default_review(workpiece: Workpiece, band: BandRules, budget: Budget) -> Decision:
    result = workpiece.evidence[-1]
    decision = assess(result, band)
    if isinstance(decision, Incomplete):
        retryable = [
            t.cause or (Cause.NO_GRADE if t.grade is None else None) for t in (*result.solver, *result.controls)
        ]
        retries = sum(isinstance(item, Retry) for item in workpiece.decisions)
        cause = next((cause for cause in retryable if cause in TRANSIENT_CAUSES), None)
        if cause is not None and retries < budget.max_validation_retries:
            return Retry(cause, retries + 1)
    # Mark's policy: one hardening attempt, then tagged acceptance; restore baseline on regression.
    if isinstance(decision, Modify) and BandOutcome.TOO_EASY in band.accepted:
        if workpiece.too_easy_baseline is not None:
            revert = decision.placement is not BandOutcome.TOO_EASY
            reasons = ("accepted_with: too_easy",)
            if revert:
                reasons += ("Hardening regressed; reverted to the previously validated submission", *decision.findings)
            return Accept(BandOutcome.TOO_EASY, reasons, revert=revert)
        if decision.placement is BandOutcome.TOO_EASY and workpiece.round >= budget.max_modify_rounds:
            return Accept(BandOutcome.TOO_EASY, ("accepted_with: too_easy", "Shared modification budget exhausted"))
    return decision


def assess(result: ValidationResult, band: BandRules) -> Decision:
    if result.lowering_error is not None:
        return Incomplete((f"Lowering failed: {result.lowering_error}",))
    if not result.solver or not result.controls:
        return Incomplete(("Required solver or control evidence is absent",))
    trials = (*result.solver, *result.controls, *(f for a in result.adversary for f in a.findings))
    causes = Counter(t.cause or (Cause.NO_GRADE if t.grade is None else None) for t in trials)
    unavailable = tuple(
        f"{count} ungraded trial(s): {cause}"
        for cause, count in sorted((cause, count) for cause, count in causes.items() if cause is not None)
        if cause not in TASK_DEFECTS and cause is not Cause.BUDGET_EXHAUSTED
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
    findings += tuple(f"{count} ungraded trial(s): {cause}" for cause, count in causes.items() if cause in TASK_DEFECTS)
    findings += tuple(
        f"{trial.grade.error or trial.cause}\n{trial.grade.diagnostics.get('stderr', '')}"
        for trial in trials
        if trial.cause in TASK_DEFECTS and trial.grade is not None
    )
    findings += tuple(
        f"{trial.name}: {finding.claim}"
        for trial in result.adversary
        for finding in trial.findings
        if finding.met or finding.cause in TASK_DEFECTS
    )
    if findings:
        return Modify(findings)
    solved = sum(passed(trial.grade) for trial in result.solver if trial.grade is not None)
    place = band.place(solved / len(result.solver))
    if place is BandOutcome.IN_BAND and place in band.accepted:
        return Accept(place)
    finding = f"{solved} of {len(result.solver)} trials solved: {place}"
    if place is BandOutcome.TOO_EASY:
        finding += (
            ". Make the task harder while preserving correct controls and grading. This is the one hardening attempt."
        )
    return Modify((finding,), placement=place)
