# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0


"""Review complete evidence before accepting or requesting author repair."""

import math
from collections import Counter
from dataclasses import replace

from taskcompendium.grading_result import GradeResult, Outcome

from taskforge.model.review import (
    Accept,
    AdversaryFinding,
    AdversaryStatus,
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
    TrialResult,
    ValidationResult,
)
from taskforge.model.workpiece import Workpiece
from taskforge.stages.classify import CAUSE_DECISIONS


def passed(grade: GradeResult) -> bool:
    if grade.status is Outcome.SUBMISSION_FAILURE:
        return False
    if grade.passed is not None:
        return grade.passed
    reward, score_max = grade.reward or 0.0, grade.score_max
    return reward >= score_max or math.isclose(reward, score_max, rel_tol=1e-9, abs_tol=1e-9)


def actionable(finding: AdversaryFinding) -> bool:
    """Whether review acts on a finding: its defect reproduced, or its replay found a grader fault to repair."""
    return finding.met or CAUSE_DECISIONS.get(finding.cause) is Modify


def exhausted_notes(result: ValidationResult) -> tuple[str, ...]:
    """Why acceptance stands without an adversary finding: each search ran out of budget; never author-facing."""
    return tuple(
        f"{trial.name}: {reason}"
        for trial in result.adversary
        if trial.status is not AdversaryStatus.FAILED
        for reason in trial.reasons
    )


def missing(trial: TrialResult) -> bool:
    """A solver slot without a score (held, timed out or ungraded): never a failed trial, never a zero."""
    cause = trial.cause or (Cause.NO_GRADE if trial.grade is None else None)
    return cause not in (None, Cause.GENERATION_LIMIT) and CAUSE_DECISIONS.get(cause) not in (Modify, Reject)


def tagged(band: BandOutcome) -> tuple[str, ...]:
    return () if band is BandOutcome.IN_BAND else (f"accepted_with: {band}",)


RANK = {BandOutcome.IN_BAND: 0, BandOutcome.TOO_EASY: 1}
"""Which acceptable placement makes the better fallback; between equals, the later submission."""


def fallback_after(workpiece: Workpiece, placement: BandOutcome | None, band: BandRules) -> str | None:
    """The digest of the submission a revision of ``workpiece`` falls back to: the current one when it placed
    acceptably and no worse than the stored fallback, otherwise the stored one."""
    kept, current = workpiece.fallback, workpiece.submission
    stored = None if kept is None else kept.submission.digest
    if current is None or placement is None or placement not in band.accepted:
        return stored
    return current.digest if kept is None or RANK[placement] <= RANK[kept.band] else stored


def default_review(workpiece: Workpiece, band: BandRules, budget: Budget) -> Decision:
    result = workpiece.evidence[-1]
    decision = assess(result, band, budget.max_missing_trials)
    if isinstance(decision, Incomplete):
        retryable = [
            t.cause or (Cause.NO_GRADE if t.grade is None else None) for t in (*result.solver, *result.controls)
        ]
        retries = sum(isinstance(item, Retry) for item in workpiece.decisions)
        cause = next((cause for cause in retryable if CAUSE_DECISIONS.get(cause) is Retry), None)
        if cause is not None and retries < budget.max_validation_retries:
            return Retry(cause, retries + 1)
    if not isinstance(decision, Modify):
        return decision
    notes = exhausted_notes(result)
    # One band-only round: after it a revision that still places acceptably is accepted with its placement, and one
    # that leaves the band or has a fault restores the fallback, named by its digest.
    if (kept := workpiece.fallback) is not None:
        if (placement := decision.placement) is not None and placement in band.accepted:
            return Accept(placement, (*tagged(placement), *notes))
        restored = f"The revision left the acceptance band; restored submission {kept.submission.digest[:12]}"
        return Accept(kept.band, (*tagged(kept.band), restored, *decision.findings, *notes), kept.submission.digest)
    exhausted = workpiece.round >= budget.max_modify_rounds
    if decision.placement is BandOutcome.TOO_EASY and decision.placement in band.accepted and exhausted:
        return Accept(decision.placement, (*tagged(decision.placement), "Shared modification budget exhausted", *notes))
    return replace(decision, fallback=fallback_after(workpiece, decision.placement, band))


def assess(result: ValidationResult, band: BandRules, max_missing: int) -> Decision:
    """Decide on ``result``: a round lacking more than ``max_missing`` solver scores is incomplete, and one lacking
    fewer is placed over the slots that have one."""
    if result.lowering_error is not None:
        decision = Modify if result.lowering_cause is Cause.INVALID_TASK else Incomplete
        return decision((f"Lowering failed: {result.lowering_error}",))
    if not result.solver or not result.controls:
        return Incomplete(("Required solver or control evidence is absent",))
    scored = tuple(trial for trial in result.solver if not missing(trial))
    absent = len(result.solver) - len(scored)
    solver = scored if absent <= max_missing else result.solver
    trials = (*solver, *result.controls, *(f for a in result.adversary for f in a.findings))
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
        if trial.status is AdversaryStatus.FAILED
        for reason in trial.reasons or ("Adversary search is incomplete (reason unavailable)",)
    )
    unavailable += tuple(f"solver trial {t.trial}: {t.detail}" for t in solver if t.detail)
    unavailable += ("No solver trial has a score",) * (not scored)
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
        f"{trial.name}: {finding.claim if finding.met else f'its replay showed {finding.cause}, not the claimed defect'}"
        for trial in result.adversary
        for finding in trial.findings
        if actionable(finding)
    )
    if findings:
        return Modify(findings)
    solved = sum(passed(trial.grade) for trial in scored if trial.grade is not None and trial.cause is None)
    place = band.place(solved / len(scored))
    notes = (f"{absent} of {len(result.solver)} solver trials have no score; placed over {len(scored)}",) * bool(absent)
    if place is BandOutcome.IN_BAND and place in band.accepted:
        return Accept(place, (*notes, *exhausted_notes(result)))
    finding = f"{solved} of {len(scored)} trials solved: {place}"
    if place is BandOutcome.TOO_EASY:
        finding += (
            ". Make the task harder while preserving correct controls and grading. This is the one hardening attempt."
        )
    return Modify((finding, *notes), placement=place)
