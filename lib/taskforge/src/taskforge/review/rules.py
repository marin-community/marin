# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Review: a pure rule table from a calibration summary to a ``Decision``. No model is called.

Rules, in order; the first that fires decides:

1. Decisive findings (a violated control, a shortcut or leak pass, an ambiguity, a task defect) give a
   ``Repair``, even when the evidence is incomplete: retrying cannot improve them.
2. Incomplete evidence with ``MACHINE_UNSUPPORTED`` or ``SUBMISSION_UNSUPPORTED`` gives ``Reject(HOST)``:
   this host's factories or conventions cannot run the task, and no rebuild here changes that.
3. Other incomplete evidence gives ``Retry`` for its most frequent cause. The loop bounds retries and
   ends an item that exhausts them as abandoned, never rejected.
4. A band finding (``TOO_HARD``, ``TOO_EASY``) gives one ``Repair`` per kind; the same kind again after
   that repair gives ``Reject(TASK)``.
5. Complete evidence with no findings gives ``Accept``.

A ``Repair`` the item cannot afford (``repairs_used >= max_repairs``) becomes ``Reject(BUDGET)``.

Staged tasks are not validated. ``staged_repair`` gives the fixed repair the loop applies to a staged
draft before validation: build the sequence as separate tasks.
"""

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from taskforge.build.run import TaskDraft
from taskforge.build.step import StepRecord, StepRole
from taskforge.review.decision import Accept, Decision, Reject, RejectKind, Repair, RepairBrief, Retry
from taskforge.spec.controls import controls_json
from taskforge.validate.calibration import DECISIVE, CalibrationSummary, Finding, FindingKind
from taskforge.validate.evidence import Incomplete
from taskforge.validate.outcome import Cause

BAND_FINDINGS = frozenset(FindingKind) - DECISIVE
"""``TOO_HARD`` and ``TOO_EASY``: findings about the solve rate, repaired once per kind."""

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

STAGED_BRIEF = """\
Staged tasks are not validated. Build the sequence as separate single-stage tasks whose environment \
files and setup reconstruct the state the earlier stages leave, and grade each one on its own."""


@dataclass(frozen=True)
class ItemHistory:
    """What the loop derives from the item's events for one decision, so review stays pure."""

    repairs_used: int
    max_repairs: int
    prior_band_findings: frozenset[FindingKind]
    """Band findings (``TOO_HARD``, ``TOO_EASY``) this item has already been repaired for."""

    def __post_init__(self) -> None:
        if self.repairs_used < 0 or self.max_repairs < 0:
            raise ValueError(f"repair counts must be non-negative: {self.repairs_used}, {self.max_repairs}")
        if not self.prior_band_findings <= BAND_FINDINGS:
            raise ValueError(f"prior band findings must be band kinds, got {sorted(self.prior_band_findings)}")

    @property
    def repairs_left(self) -> bool:
        return self.repairs_used < self.max_repairs


def steps_for(roles: Iterable[StepRole], steps: Sequence[StepRecord]) -> tuple[str, ...]:
    """The names of the steps in ``steps`` with one of ``roles``, in build order, each once."""
    wanted = frozenset(roles)
    return tuple(dict.fromkeys(record.name for record in steps if record.role in wanted))


def render_brief(findings: Sequence[Finding]) -> RepairBrief:
    """The revision brief for ``findings``: each finding's kind, responsible roles and detail, then
    every new control as JSON the revised CONTROLS step must include verbatim."""
    if not findings:
        raise ValueError("a repair brief needs at least one finding")
    sections = [BRIEF_HEADER]
    for number, finding in enumerate(findings, start=1):
        roles = ", ".join(role.value for role in finding.roles)
        sections.append(f"Finding {number}: {finding.kind.value} (revise the {roles} steps)\n{finding.detail}")
    new_controls = tuple(control for finding in findings for control in finding.new_controls)
    if new_controls:
        sections.append(f"{NEW_CONTROLS_HEADER}\n\n```json\n{controls_json(new_controls).decode()}\n```")
    return RepairBrief(findings=tuple(findings), failure="\n\n".join(sections))


def _reason(finding: Finding) -> str:
    detail = finding.detail.strip().splitlines()
    return f"{finding.kind.value}: {detail[0]}" if detail else finding.kind.value


def _repair(
    draft: TaskDraft, summary: CalibrationSummary, findings: Sequence[Finding], history: ItemHistory
) -> Repair | Reject:
    if not history.repairs_left:
        reasons = (
            *(_reason(f) for f in findings),
            f"repairs exhausted: {history.repairs_used} of {history.max_repairs}",
        )
        return Reject(kind=RejectKind.BUDGET, reasons=reasons, summary=summary)
    roles = {role for finding in findings for role in finding.roles}
    return Repair(
        program_digest=draft.provenance.program_digest,
        brief=render_brief(findings),
        invalidate=steps_for(roles, draft.provenance.steps),
    )


def _most_common(causes: Counter[Cause]) -> tuple[Cause, int]:
    """The most frequent cause; ties go to the cause named first alphabetically, so a decision is stable."""
    return min(causes.items(), key=lambda item: (-item[1], item[0].value))


def decide(draft: TaskDraft, summary: CalibrationSummary, history: ItemHistory) -> Decision:
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
    repeated = [finding for finding in band if finding.kind in history.prior_band_findings]
    if repeated:
        reasons = tuple(f"{_reason(finding)} (again after a repair)" for finding in repeated)
        return Reject(kind=RejectKind.TASK, reasons=reasons, summary=summary)
    if band:
        return _repair(draft, summary, band, history)
    return Accept(summary=summary)


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
    brief = RepairBrief(findings=(), failure=STAGED_BRIEF)
    return Repair(program_digest=draft.provenance.program_digest, brief=brief, invalidate=())
