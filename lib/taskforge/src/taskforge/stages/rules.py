# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0


"""Review complete evidence before accepting or requesting author repair."""

import math
from collections import Counter
from collections.abc import Sequence
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
    Response,
    Retry,
    ReviewFinding,
    Revise,
    SolveReview,
    TrialResult,
    ValidationResult,
    same,
    unique,
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


INSTRUCTIONS = {
    BandOutcome.IN_BAND: "Keep the difficulty.",
    BandOutcome.TOO_EASY: "Make the task harder while preserving correct controls and grading.",
    BandOutcome.TOO_HARD: "Make the task more tractable while preserving its objective, correct controls and grading.",
}
"""What each placement asks of the author."""


def band_line(result: ValidationResult, band: BandRules) -> tuple[BandOutcome, str]:
    """The placement of ``result``'s scored solver trials, and live Forge's line telling the author what it asks for."""
    scored = [trial for trial in result.solver if not missing(trial)]
    solved = sum(passed(trial.grade) for trial in scored if trial.grade is not None and trial.cause is None)
    place = band.place(solved / len(scored))
    return place, f"{solved} of {len(scored)} trials solved: {place}. {INSTRUCTIONS[place]}"


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


def shipped(decision: Accept, workpiece: Workpiece) -> int:
    """The evidence index of the round whose submission ``decision`` ships: the stored fallback's round for a restore
    (a digest cannot tell two rounds apart), else the latest."""
    return workpiece.fallback.evidence if decision.restore and workpiece.fallback else len(workpiece.evidence) - 1


def accepted(decision: Decision, workpiece: Workpiece) -> Decision:
    """An acceptance with what the round it ships (``shipped``) went without, and that round's review findings as
    notes. A restore also carries that round's placement denominator and adversary exhaustion, which its decision did
    not."""
    if not isinstance(decision, Accept):
        return decision
    result = workpiece.evidence[shipped(decision, workpiece)] if workpiece.evidence else None
    review = result.review if result is not None else None
    if decision.restore and result is not None:
        decision = replace(decision, reasons=(*decision.reasons, *unscored(result), *exhausted_notes(result)))
    missing = (MissingEvidence(EvidenceKind.SOLVE_REVIEW, "no solve-review ran"),) if review is None else review.missing
    notes = () if review is None else review.findings
    why = "of the restored round recorded as notes" if decision.restore else "recorded as notes: no revision round left"
    why = (f"Solve-review findings {why}",) * bool(notes)
    return replace(decision, reasons=(*decision.reasons, *why), missing=missing, notes=notes)


def final_revision(workpiece: Workpiece) -> Revise | None:
    """The latest revision the final review asked for, once one was: the workpiece then keeps no fallback."""
    return next((item for item in reversed(workpiece.decisions) if isinstance(item, Revise)), None)


def revision_left(workpiece: Workpiece, budget: Budget) -> bool:
    return sum(isinstance(item, Revise) for item in workpiece.decisions) < budget.final_revisions


def unrevised(revise: Revise | None, why: str, details: Sequence[str] = ()) -> Abandon:
    """A fault in a final revision, a fault or a refused placement in a confirmation with no revision left, or the
    author stopping a final revision, abandons the task: nothing is kept to restore, and nothing ships against a revise
    finding. The abandonment keeps why and the finding."""
    if revise is None:
        return Abandon((f"{CONFIRMATION_FAILED}: {why}", *details))
    unresolved = (f"Unresolved final-review finding: {finding.description}" for finding in revise.review)
    return Abandon((f"{CONFIRMATION_REVISION_FAILED}: {why}", *details, *revise.findings, *unresolved))


def author_stopped(workpiece: Workpiece, summary: str) -> Decision:
    """Owner's ruling (2026-11-13): an author stopping reverts to the stored fallback, or abandons without one."""
    why = f"author stopped: {summary}"
    if revise := final_revision(workpiece):
        return unrevised(revise, why)
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
    """The decision on a round. Placement in a band the policy accepts (``band.accepted``) tags a task and never
    filters it, and a band change is never a regression; a refused band is never accepted: it asks for revision
    rounds within the shared budget, restores an accepted fallback, or is abandoned once the budget is spent.

    In order: a clean revision placed in an accepted band other than its fallback's is accepted at its own placement,
    tagged; otherwise an out-of-band task gets one band-only round (``band_only``) within the shared budget and is then
    accepted tagged. Review findings join a round's modification, or turn its acceptance into one at its placement,
    while the shared budget has a round left, and are the acceptance's notes after. A faulted revision restores the
    fallback, named by digest and band.

    After a final revision (``Revise``) a clean round is accepted at its placement whatever the band, pending the
    confirmation that places it again; review findings are notes, and a fault abandons the task: a final revision
    keeps no fallback. A confirmation record is ``confirmation_review``'s.
    """
    result = workpiece.evidence[-1]
    if result.confirms is not None:
        return confirmation_review(workpiece, band, budget)
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
    revise = final_revision(workpiece)
    review = result.review.findings if result.review is not None else ()
    if revise and isinstance(decision, Modify) and decision.placement is not None:  # clean: its confirmation places it
        final = "Placement is final after the final review's revision"
        decision = Accept(decision.placement, (*tagged(decision.placement), final, *decision.findings[1:], *notes))
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
    if isinstance(decision, Accept) and review and left and not revise:
        decision = Modify((band_line(result, band)[1],), decision.band)
        placement = decision.placement
    if not isinstance(decision, Modify):
        return accepted(decision, workpiece)
    chosen = fallback_for(workpiece, placement, band)
    fault = "has a fault" if decision.placement is None else f"placed {decision.placement}"
    if revise:  # only a fault is left here: a clean revision was accepted above
        return unrevised(revise, f"the revision {fault}", decision.findings)
    if placement is not None or chosen is None:  # a round to revise, or a fault with nothing to restore
        return replace(decision, fallback=None if chosen is None else chosen.submission.digest, review=review)
    why = f"The revision {fault}; restored submission {chosen.submission.digest[:12]}"
    label = "Revision fault" if decision.placement is None else "Revision placement"  # a band the policy refuses
    faults = (f"{label}: {finding}" for finding in decision.findings)
    restored = Accept(chosen.band, (*tagged(chosen.band), why, *faults), chosen.submission.digest)
    return accepted(restored, workpiece)


CONFIRMATION_REVISION_FAILED = "Confirmation revision failed"
"""The first reason of the ``Abandon`` when a final revision has a fault and nothing can be kept."""
CONFIRMATION_FAILED = "Confirmation failed"
"""The first reason of the ``Abandon`` when a confirmation fails and no final revision is left to repair it."""


def confirmation_review(workpiece: Workpiece, band: BandRules, budget: Budget) -> Decision:
    """The decision on a confirmation record, after the round's own assessment: a fault in the new trials (an invalid
    task, a grader fault) or an exhausted trial budget is decided as validation decides it, so only valid scores and
    generation limits are placed; unscored slots are named and excluded. The placement of the shipped submission over
    every scored trial re-tags it.

    The final review's NEW revise findings, a fault or a placement the policy refuses (``band.accepted``) buy a
    revision outside the shared budget (``Revise``) while ``budget.final_revisions`` has one left; a finding an earlier
    review gave counts again only as a failed fix, after the author revised for it. With none left, revise findings are
    notes, and a fault or a refused placement abandons the task. The acceptance ships the record's notes (every earlier
    and new note its trials still support, each once) and its missing evidence, each item once: never a hold."""
    result, pending = workpiece.evidence[-1], workpiece.decisions[-1]
    assert (
        isinstance(pending, Accept) and result.confirms is not None
    ), "a confirmation follows the acceptance it confirms"
    review, revise, left = result.review, final_revision(workpiece), revision_left(workpiece, budget)
    decision = assess(result, band, len(result.solver))  # every unscored slot excluded; held ones held in the batch
    if isinstance(decision, Reject | Incomplete):
        return decision
    fixed = failed_fixes(workpiece, result.confirms)
    known = {same(f) for earlier in workpiece.evidence[:-1] if earlier.review for f in earlier_findings(earlier.review)}
    asked = () if review is None else review.findings
    fresh = tuple(f for f in asked if same(f) not in known or same(f) in fixed)
    faults = decision.findings if isinstance(decision, Modify) and decision.placement is None else ()
    notes = unique((*(f for f in asked if f not in fresh), *(() if review is None else review.notes)))
    if faults:
        return Revise(faults, fresh, notes=notes) if left else unrevised(revise, "its confirmation has a fault", faults)
    place, line = band_line(result, band)
    if (fresh or place not in band.accepted) and left:
        return Revise((line,), fresh, notes=notes, placement=place)
    if place not in band.accepted:
        return refused_band(workpiece, budget, line, revise, asked)
    reasons = (*tagged(place), f"Confirmation: {line.split('. ')[0]}", *unscored(result))
    reasons += ("No final revision is left: the final review's revise findings are recorded as notes",) * bool(
        asked and not left
    )
    absent = [trial.trial for trial in result.solver if missing(trial)]
    found = (MissingEvidence(EvidenceKind.SOLVE_REVIEW, "no final review ran"),) if review is None else review.missing
    found += (MissingEvidence(EvidenceKind.CONFIRMATION_TRIALS, f"solver trials {absent} have no score"),) * bool(absent)
    kept = tuple(reason for reason in pending.reasons if reason not in tagged(pending.band))
    return Accept(place, (*reasons, *kept), missing=tuple(dict.fromkeys((*pending.missing, *found))), notes=result.notes)


def refused_band(
    workpiece: Workpiece, budget: Budget, line: str, revise: Revise | None, asked: Sequence[ReviewFinding]
) -> Abandon:
    """A confirmation placed in a band the policy refuses, with no final revision left: the abandonment names the
    placement, the setting and the spent budget. A final revision's finding is unresolved only when this final review
    reports it again or the author's latest answer to it, on the history a restore kept, declined it."""
    count, place = line.split(". ")[0].rsplit(": ", 1)
    setting = "band.accept_too_easy is false" if place == BandOutcome.TOO_EASY else f"the policy refuses {place}"
    first = CONFIRMATION_FAILED if revise is None else CONFIRMATION_REVISION_FAILED
    spent = f"budget.final_revisions ({budget.final_revisions}) is spent"
    why = f"{first}: confirmation placed {place} ({count}); {setting} and {spent}"
    reported = {same(f) for f in asked}
    latest = {same(r.finding): r.response for r in workpiece.responses if not r.discarded}  # later answers override
    declined = {key for key, response in latest.items() if response is Response.DECLINED}
    unresolved = [f for f in (revise.review if revise else ()) if same(f) in reported | declined]
    return Abandon((why, *(f"Unresolved final-review finding: {finding.description}" for finding in unresolved)))


def failed_fixes(workpiece: Workpiece, shipped: int | None) -> set[tuple[str, str, str]]:
    """The findings (``same``) whose latest response on the history leading to evidence record ``shipped`` says the
    author revised for them: a repeat of one is evidence the fix failed. A response to that record or later, or one a
    restore discarded, belongs to another history; a later decline overrides an earlier revision."""
    latest = {
        same(r.finding): r.response
        for r in workpiece.responses
        if shipped is not None and r.review < shipped and not r.discarded
    }
    return {key for key, response in latest.items() if response is Response.REVISED}


def earlier_findings(review: SolveReview) -> tuple[ReviewFinding, ...]:
    return (*review.findings, *review.notes)


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
    place, line = band_line(result, band)
    notes = unscored(result)
    if place is BandOutcome.IN_BAND and place in band.accepted:
        return Accept(place, (*notes, *exhausted_notes(result)))
    return Modify((line, *notes), placement=place)
