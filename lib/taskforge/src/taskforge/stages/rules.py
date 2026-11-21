# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0


"""Review complete evidence before accepting or requesting author repair."""

import math
from collections import Counter
from dataclasses import replace

from taskcompendium.grading_result import GradeResult, Outcome

from taskforge.model.review import (
    Abandon,
    Accept,
    AdversaryFinding,
    AdversaryStatus,
    BandOutcome,
    BandRules,
    Budget,
    Cause,
    Decision,
    EvidenceKind,
    Incomplete,
    MissingEvidence,
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


def unscored(result: ValidationResult) -> tuple[str, ...]:
    """How many solver slots of ``result`` had no score, when any did, and how many its placement went over."""
    scored = sum(not missing(trial) for trial in result.solver)
    absent = len(result.solver) - scored
    return (f"{absent} of {len(result.solver)} solver trials have no score; placed over {scored}",) * bool(absent)


def missing(trial: TrialResult) -> bool:
    """A solver slot without a score (held, timed out or ungraded): never a failed trial, never a zero."""
    cause = trial.cause or (Cause.NO_GRADE if trial.grade is None else None)
    return cause not in (None, Cause.GENERATION_LIMIT) and CAUSE_DECISIONS.get(cause) not in (Modify, Reject)


def tagged(band: BandOutcome) -> tuple[str, ...]:
    return () if band is BandOutcome.IN_BAND else (f"accepted_with: {band}",)


def fallback_for(workpiece: Workpiece, placement: BandOutcome | None, band: BandRules) -> Fallback | None:
    """The one choice of the submission to fall back to, or accept: the current submission at ``placement`` when the
    policy accepts that, else the stored fallback while the policy accepts its band. Owner's ruling (2026-11-13): "A
    band change is never a regression. Only real faults or an author stopping revert to the fallback." Returning the
    stored object itself is what names it: a digest covers the task and controls, not the author's notes, so it cannot
    tell rounds apart."""
    if workpiece.submission is not None and placement is not None and placement in band.accepted:
        return Fallback(workpiece.submission, placement, len(workpiece.evidence) - 1)
    return workpiece.fallback if workpiece.fallback and workpiece.fallback.band in band.accepted else None


def accepted(decision: Decision, workpiece: Workpiece) -> Decision:
    """An acceptance with what the round of its accepted submission went without, and that round's review findings as
    notes: the stored fallback's round for a restore (a digest cannot tell two rounds apart), else the latest. A
    restore also carries that round's placement denominator and adversary exhaustion, which its decision did not."""
    if not isinstance(decision, Accept):
        return decision
    shipped = workpiece.fallback.evidence if decision.restore and workpiece.fallback else -1
    result = workpiece.evidence[shipped] if workpiece.evidence else None
    review = result.review if result is not None else None
    if decision.restore and result is not None:
        decision = replace(decision, reasons=(*decision.reasons, *unscored(result), *exhausted_notes(result)))
    missing = (MissingEvidence(EvidenceKind.SOLVE_REVIEW, "no solve-review ran"),) if review is None else review.missing
    notes = () if review is None else review.findings
    why = "of the restored round recorded as notes" if decision.restore else "recorded as notes: no revision round left"
    why = (f"Solve-review findings {why}",) * bool(notes)
    return replace(decision, reasons=(*decision.reasons, *why), missing=missing, notes=notes)


def author_stopped(workpiece: Workpiece, summary: str) -> Decision:
    """Owner's ruling (2026-11-13): an author stopping reverts to the stored fallback, or abandons without one."""
    why = f"author stopped: {summary}"
    if workpiece.fallback is None:
        return Abandon((why,))
    kept = workpiece.fallback
    restored = f"{why}; restored submission {kept.submission.digest[:12]}"
    return accepted(Accept(kept.band, (*tagged(kept.band), restored), kept.submission.digest), workpiece)


def band_only(decision: Decision) -> bool:
    """A modification round for difficulty alone: placed, with no review finding to revise (a defect round carries its
    placement too, and is not one)."""
    return isinstance(decision, Modify) and decision.placement is not None and not decision.review


def default_review(workpiece: Workpiece, band: BandRules, budget: Budget) -> Decision:
    """Owner's ruling (2026-11-12): placement tags a task and never filters it; live Forge's three decisions, in order.

    A clean revision placed out of its fallback's band is accepted at its own placement, tagged; otherwise an out-of-band
    task gets one band-only round (``band_only``) within the shared budget and is then accepted tagged. Review findings
    join a round's modification, or turn its acceptance into one at its placement, while the shared budget has a round
    left, and are the acceptance's notes after. A faulted revision restores the fallback, named by digest and band.
    """
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
    left, stored, notes = workpiece.round < budget.max_modify_rounds, workpiece.fallback, exhausted_notes(result)
    review = result.review.findings if result.review is not None else ()
    placement = decision.placement if isinstance(decision, Modify) and decision.placement in band.accepted else None
    if isinstance(decision, Modify) and placement is not None:  # a clean round out of band
        if stored is not None and placement is not stored.band:
            why = f"Clean revision placed {placement}, not {stored.band}: a placement change is tagged, not reverted"
        elif any(map(band_only, workpiece.decisions)):
            why = "The one band-only round is used"
        elif not left:
            why = "Shared modification budget exhausted"
        else:
            why = ""
        if why:
            decision = Accept(placement, (*tagged(placement), why, *decision.findings[1:], *notes))
    if isinstance(decision, Accept) and review and left:
        decision = Modify((f"The solve-review proposed {len(review)} edit(s)",), decision.band)
        placement = decision.placement
    if not isinstance(decision, Modify):
        return accepted(decision, workpiece)
    chosen = fallback_for(workpiece, placement, band)
    if placement is not None or chosen is None:  # a round to revise, or a fault with nothing to restore
        return replace(decision, fallback=None if chosen is None else chosen.submission.digest, review=review)
    fault = "has a fault" if decision.placement is None else f"placed {decision.placement}"
    why = f"The revision {fault}; restored submission {chosen.submission.digest[:12]}"
    faults = (f"Revision fault: {finding}" for finding in decision.findings)
    restored = Accept(chosen.band, (*tagged(chosen.band), why, *faults), chosen.submission.digest)
    return accepted(restored, workpiece)


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
        if CAUSE_DECISIONS.get(cause) not in (Modify, Reject)
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
    notes = unscored(result)
    if place is BandOutcome.IN_BAND and place in band.accepted:
        return Accept(place, (*notes, *exhausted_notes(result)))
    finding = f"{solved} of {len(scored)} trials solved: {place}"
    if place is BandOutcome.TOO_EASY:
        finding += (
            ". Make the task harder while preserving correct controls and grading. This is the one hardening attempt."
        )
    return Modify((finding, *notes), placement=place)
