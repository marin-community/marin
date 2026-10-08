# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Review: a pure rule table from a calibration summary to a ``Decision``. No model is called.

Rules, in order; the first that fires decides:

1. Decisive findings (a violated control, an adversary shortcut the calibration tiered as a repair, a task
   defect) give a ``Repair``, even when the evidence is incomplete: retrying cannot improve them. An item
   with no repairs left gets ``Reject(BUDGET)`` instead.
2. Incomplete evidence with ``MACHINE_UNSUPPORTED`` or ``SUBMISSION_UNSUPPORTED`` gives ``Reject(HOST)``:
   this host's factories or conventions cannot run the task, and no rebuild here changes that.
3. Other incomplete evidence gives ``Retry`` for its most frequent cause. The loop bounds retries and
   ends an item that exhausts them as abandoned, never rejected.
4. A band finding (``TOO_HARD``, ``TOO_EASY``) is decided by the consumer's ``BandRule`` for its kind:

   a. while the item has had fewer ``Repair`` decisions for that kind than ``BandRule.repairs`` and has
      repairs left, it gives a ``Repair``;
   b. otherwise ``BandRule.then`` decides. ``ACCEPT`` gives ``Accept`` with ``band`` naming the kind, also
      when the item's repair budget is what is spent. ``REJECT`` gives ``Reject(TASK)`` when the kind's
      repairs are spent, else ``Reject(BUDGET)``.

5. Complete evidence with no findings gives ``Accept`` with ``band`` ``IN_BAND``.

Noted adversary results ride along in the brief and in an accepted summary; they never change the decision.

Staged tasks are not validated. ``staged_repair`` gives the fixed repair the loop applies to a staged
draft before validation: build the sequence as separate tasks.
"""

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from taskforge.build.run import TaskDraft
from taskforge.build.step import StepRecord, StepRole
from taskforge.review.decision import (
    BAND_OUTCOMES,
    Accept,
    BandOutcome,
    Decision,
    Reject,
    RejectKind,
    Repair,
    RepairBrief,
    Retry,
)
from taskforge.spec.controls import controls_json
from taskforge.validate.calibration import DECISIVE, CalibrationSummary, Finding, FindingKind
from taskforge.validate.evidence import Incomplete
from taskforge.validate.outcome import Cause

BAND_FINDINGS = frozenset(FindingKind) - DECISIVE
"""``TOO_HARD`` and ``TOO_EASY``: findings about the solve rate, decided by the consumer's ``BandRules``."""

HOST_CAUSES = frozenset({Cause.MACHINE_UNSUPPORTED, Cause.SUBMISSION_UNSUPPORTED})
"""Causes that say the host cannot run the task, not that the task or the transport failed."""

BRIEF_HEADER = """\
Validating the task this program built found the defects below. Revise the program so the next build \
fixes every one of them. Each finding names the step roles responsible; rewrite those steps and keep \
the others unchanged so their memoized results are reused."""

NEW_CONTROLS_HEADER = """\
Validation turned these adversarial submissions into controls. The CONTROLS step of the revised program \
must return each of them verbatim, in addition to its own controls, so the next validation round replays \
them against the revised grader and shows that the grader now rejects them. Do not grade these exact \
submissions with `b.try_grader` during the build: a build whose controls include a candidate it graded \
fails. Prototype the grader on a different submission of the same kind:"""

NOTES_HEADER = """\
Validation also observed the adversary results below. They are recorded, not defects to fix: in each the \
adversary reached an accepted submission only after more verifier calls than the repair threshold, gave no \
verdict, or passed a many-answer grader with text no honest run produced while reporting no shortcut. Do not \
add controls for them. Mention them only if the finding you are fixing is related."""

STAGED_BRIEF = """\
Staged tasks are not validated. Build the sequence as separate single-stage tasks whose environment \
files and setup reconstruct the state the earlier stages leave, and grade each one on its own."""


class BandChoice(StrEnum):
    """What a consumer wants for a task still outside the band after that kind's repairs are spent."""

    ACCEPT = "accept"
    """``Accept`` with ``band`` naming the kind; the summary's solve rate labels the accepted task."""
    REJECT = "reject"
    """``Reject(TASK)``: Taskforge's own policies choose this."""


@dataclass(frozen=True)
class BandRule:
    """How a band kind is decided: up to ``repairs`` revisions, then ``then``."""

    repairs: int
    """``Repair`` decisions this kind may trigger per item before ``then`` applies."""
    then: BandChoice

    def __post_init__(self) -> None:
        if self.repairs < 0:
            raise ValueError(f"band repairs must be non-negative, got {self.repairs}")


@dataclass(frozen=True)
class BandRules:
    """The consumer's ``BandRule`` for each band kind."""

    too_easy: BandRule
    too_hard: BandRule

    def rule(self, kind: FindingKind) -> BandRule:
        if kind is FindingKind.TOO_EASY:
            return self.too_easy
        if kind is FindingKind.TOO_HARD:
            return self.too_hard
        raise ValueError(f"{kind.value} is not a band kind; expected one of {sorted(BAND_FINDINGS)}")


@dataclass(frozen=True)
class ItemHistory:
    """What the loop derives from the item's events for one decision, so review stays pure."""

    repairs_used: int
    max_repairs: int
    band_repairs: Mapping[FindingKind, int]
    """``Repair`` decisions already issued per band kind; a kind that is absent has had none."""

    def __post_init__(self) -> None:
        if self.repairs_used < 0 or self.max_repairs < 0:
            raise ValueError(f"repair counts must be non-negative: {self.repairs_used}, {self.max_repairs}")
        if not self.band_repairs.keys() <= BAND_FINDINGS:
            raise ValueError(f"band repairs must be keyed by band kinds, got {sorted(self.band_repairs)}")
        if any(count < 0 for count in self.band_repairs.values()):
            raise ValueError(f"band repair counts must be non-negative, got {dict(self.band_repairs)}")

    @property
    def repairs_left(self) -> bool:
        return self.repairs_used < self.max_repairs


def steps_for(roles: Iterable[StepRole], steps: Sequence[StepRecord]) -> tuple[str, ...]:
    """The names of the steps in ``steps`` with one of ``roles``, in build order, each once."""
    wanted = frozenset(roles)
    return tuple(dict.fromkeys(record.name for record in steps if record.role in wanted))


def render_brief(findings: Sequence[Finding], notes: Sequence[Finding]) -> RepairBrief:
    """The revision brief for ``findings``: each finding's kind, responsible roles and detail, then
    ``notes`` under ``NOTES_HEADER`` when there are any, then every new control of ``findings`` as JSON
    the revised CONTROLS step must include verbatim. Notes contribute no controls."""
    if not findings:
        raise ValueError("a repair brief needs at least one finding")
    sections = [BRIEF_HEADER]
    for number, finding in enumerate(findings, start=1):
        roles = ", ".join(role.value for role in finding.roles)
        sections.append(f"Finding {number}: {finding.kind.value} (revise the {roles} steps)\n{finding.detail}")
    if notes:
        sections.append(NOTES_HEADER)
        for number, note in enumerate(notes, start=1):
            roles = ", ".join(role.value for role in note.roles)
            sections.append(f"Note {number}: {note.kind.value} ({roles})\n{note.detail}")
    new_controls = tuple(control for finding in findings for control in finding.new_controls)
    if new_controls:
        sections.append(f"{NEW_CONTROLS_HEADER}\n\n```json\n{controls_json(new_controls).decode()}\n```")
    return RepairBrief(findings=tuple(findings), notes=tuple(notes), failure="\n\n".join(sections))


def _reason(finding: Finding) -> str:
    detail = finding.detail.strip().splitlines()
    return f"{finding.kind.value}: {detail[0]}" if detail else finding.kind.value


def _out_of_budget(summary: CalibrationSummary, findings: Sequence[Finding], history: ItemHistory) -> Reject:
    reasons = (
        *(_reason(f) for f in findings),
        f"repairs exhausted: {history.repairs_used} of {history.max_repairs}",
    )
    return Reject(kind=RejectKind.BUDGET, reasons=reasons, summary=summary)


def _repair(
    draft: TaskDraft, summary: CalibrationSummary, findings: Sequence[Finding], history: ItemHistory
) -> Repair | Reject:
    if not history.repairs_left:
        return _out_of_budget(summary, findings, history)
    roles = {role for finding in findings for role in finding.roles}
    return Repair(
        program_digest=draft.provenance.program_digest,
        brief=render_brief(findings, summary.notes),
        invalidate=steps_for(roles, draft.provenance.steps),
    )


def _band(
    draft: TaskDraft, summary: CalibrationSummary, finding: Finding, history: ItemHistory, rules: BandRules
) -> Decision:
    """Rule 4 for the round's band finding."""
    rule = rules.rule(finding.kind)
    used = history.band_repairs.get(finding.kind, 0)
    kind_spent = used >= rule.repairs
    if not kind_spent and history.repairs_left:
        return _repair(draft, summary, (finding,), history)
    if rule.then is BandChoice.ACCEPT:
        return Accept(summary=summary, band=BAND_OUTCOMES[finding.kind])
    if not kind_spent:
        return _out_of_budget(summary, (finding,), history)
    reason = f"{_reason(finding)} (repairs spent: {used} of {rule.repairs} for {finding.kind.value})"
    return Reject(kind=RejectKind.TASK, reasons=(reason,), summary=summary)


def _most_common(causes: Counter[Cause]) -> tuple[Cause, int]:
    """The most frequent cause; ties go to the cause named first alphabetically, so a decision is stable."""
    return min(causes.items(), key=lambda item: (-item[1], item[0].value))


def decide(draft: TaskDraft, summary: CalibrationSummary, history: ItemHistory, rules: BandRules) -> Decision:
    """The decision for one validation round of ``draft``; see the module docstring for the rules."""
    if draft.task.stages:
        raise ValueError(f"{draft.task.id} is staged; staged drafts take staged_repair, not validation")
    decisive = summary.decisive
    if decisive:
        return _repair(draft, summary, decisive, history)
    if isinstance(summary.status, Incomplete):
        causes = summary.status.causes
        host = sorted(cause for cause in causes if cause in HOST_CAUSES)
        if host:
            reasons = tuple(f"{cause.value}: {causes[cause]} trials ungraded" for cause in host)
            return Reject(kind=RejectKind.HOST, reasons=reasons, summary=summary)
        cause, count = _most_common(causes)
        return Retry(cause=cause, count=count)
    band = [finding for finding in summary.findings if finding.kind in BAND_FINDINGS]
    if len(band) > 1:
        raise ValueError(f"one solve rate gives at most one band finding, got {[f.kind.value for f in band]}")
    if band:
        return _band(draft, summary, band[0], history, rules)
    return Accept(summary=summary, band=BandOutcome.IN_BAND)


def staged_repair(draft: TaskDraft, history: ItemHistory) -> Repair | Reject:
    """The fixed decision for a staged draft at BUILT, counted against the repair budget."""
    if not draft.task.stages:
        raise ValueError(f"{draft.task.id} is not staged")
    if not history.repairs_left:
        reasons = (
            "staged tasks are not validated",
            f"repairs exhausted: {history.repairs_used} of {history.max_repairs}",
        )
        return Reject(kind=RejectKind.BUDGET, reasons=reasons, summary=None)
    brief = RepairBrief(findings=(), notes=(), failure=STAGED_BRIEF)
    return Repair(program_digest=draft.provenance.program_digest, brief=brief, invalidate=())
