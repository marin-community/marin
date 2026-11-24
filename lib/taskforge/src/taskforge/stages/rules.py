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
from taskforge.model.workpiece import Fallback, Workpiece
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


def better(candidate: Fallback, stored: Fallback) -> Fallback:
    """In band beats out of band; between equals, the later submission."""
    return stored if stored.band is BandOutcome.IN_BAND and candidate.band is not BandOutcome.IN_BAND else candidate


def fallback_for(workpiece: Workpiece, placement: BandOutcome | None, band: BandRules) -> Fallback | None:
    """The one choice of the submission to fall back to, or accept: the stored fallback while the policy accepts its
    band, against the current submission at ``placement`` when the policy accepts that. Returning the stored object
    itself is what names it: a digest covers the task and controls, not the author's notes, so it cannot tell rounds
    apart."""
    kept = workpiece.fallback if workpiece.fallback and workpiece.fallback.band in band.accepted else None
    if workpiece.submission is None or placement is None or placement not in band.accepted:
        return kept
    current = Fallback(workpiece.submission, placement)
    return current if kept is None else better(current, kept)


def default_review(workpiece: Workpiece, band: BandRules, budget: Budget) -> Decision:
    """Owner's ruling (2026-11-12): placement tags a task and never filters it. An out-of-band task gets one band-only
    round within the shared budget and is then accepted tagged; a clean revision is placed on its own evidence and a
    faulted one restores the fallback, named by digest and band."""
    result = workpiece.evidence[-1]
    decision = assess(result, band, budget.max_missing_trials)
    if isinstance(decision, Incomplete):
        retryable = [
            t.cause or (Cause.NO_GRADE if t.grade is None else None) for t in (*result.solver, *result.controls)
        ]
        retries = sum(isinstance(item, Retry) for item in workpiece.decisions)
        cause = next((cause for cause in retryable if CAUSE_DECISIONS.get(cause) is Retry), None)
        held = sum(trial.cause is Cause.HELD for trial in result.solver) > budget.max_missing_trials  # never re-bought
        if cause is not None and retries < budget.max_validation_retries and not held:
            return Retry(cause, retries + 1)
    if not isinstance(decision, Modify):
        return decision
    placement = decision.placement if decision.placement in band.accepted else None  # None: a fault or ineligible
    final = workpiece.fallback is not None or workpiece.round >= budget.max_modify_rounds
    chosen = fallback_for(workpiece, placement, band)
    if (placement is not None and not final) or chosen is None:
        return replace(decision, fallback=None if chosen is None else chosen.submission.digest)
    notes = exhausted_notes(result)
    if chosen is not workpiece.fallback:  # the selector's own choice; a digest cannot tell two rounds apart
        why = (
            "Clean revision placed on its own evidence" if workpiece.fallback else "Shared modification budget exhausted"
        )
        return Accept(chosen.band, (*tagged(chosen.band), why, *decision.findings[1:], *notes))
    fault = "has a fault" if decision.placement is None else f"placed {decision.placement}"
    why = f"The revision {fault}; restored submission {chosen.submission.digest[:12]}"
    return Accept(chosen.band, (*tagged(chosen.band), why, *decision.findings, *notes), chosen.submission.digest)


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
