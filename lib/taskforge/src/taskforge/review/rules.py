# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Review: a pure rule table from a calibration summary to a ``Decision``. No model is called.

Rules, in order; the first that fires decides:

1. Decisive findings (a task defect) give ``Reject(TASK)``, even when the evidence is incomplete:
   retrying cannot improve them.
2. Incomplete evidence with ``MACHINE_UNSUPPORTED`` or ``SUBMISSION_UNSUPPORTED`` gives ``Reject(HOST)``:
   this host's factories cannot run the task, or its answer format cannot carry its answer.
3. Other incomplete evidence gives ``Retry`` for its most frequent cause a re-run can change
   (``validate.attempts.RERUNNABLE``). The loop bounds retries and ends an item that exhausts them as
   abandoned, never rejected. Incomplete evidence with no such cause gives ``Reject(TASK)`` naming its
   settled causes: re-running would load the same outcomes.
4. A band finding (``TOO_HARD``, ``TOO_EASY``) is decided by the consumer's ``BandRules`` for its kind:
   ``ACCEPT`` gives ``Accept`` with ``band`` naming the kind, ``REJECT`` gives ``Reject(TASK)``.
5. Complete evidence with no findings gives ``Accept`` with ``band`` ``IN_BAND``.

No rule returns ``Modify``: this review cannot revise a program.
"""

from collections import Counter
from dataclasses import dataclass
from enum import StrEnum

from taskforge.review.decision import BAND_OUTCOMES, Accept, BandOutcome, Decision, Reject, RejectKind, Retry
from taskforge.validate.attempts import RERUNNABLE
from taskforge.validate.calibration import DECISIVE, CalibrationSummary, Finding, FindingKind
from taskforge.validate.evidence import Incomplete
from taskforge.validate.outcome import Cause

BAND_FINDINGS = frozenset(FindingKind) - DECISIVE
"""``TOO_HARD`` and ``TOO_EASY``: findings about the solve rate, decided by the consumer's ``BandRules``."""

HOST_CAUSES = frozenset({Cause.MACHINE_UNSUPPORTED, Cause.SUBMISSION_UNSUPPORTED})
"""Causes that say the host cannot run the task, not that the task or the transport failed."""


class BandChoice(StrEnum):
    """What a consumer wants for a task outside the band."""

    ACCEPT = "accept"
    """``Accept`` with ``band`` naming the kind; the summary's solve rate labels the accepted task."""
    REJECT = "reject"
    """``Reject(TASK)``."""


@dataclass(frozen=True)
class BandRules:
    """The consumer's ``BandChoice`` for each band kind."""

    too_easy: BandChoice
    too_hard: BandChoice

    def choice(self, kind: FindingKind) -> BandChoice:
        if kind is FindingKind.TOO_EASY:
            return self.too_easy
        if kind is FindingKind.TOO_HARD:
            return self.too_hard
        raise ValueError(f"{kind.value} is not a band kind; expected one of {sorted(BAND_FINDINGS)}")


def _reason(finding: Finding) -> str:
    return f"{finding.kind.value}: {finding.detail}"


def _most_common(causes: Counter[Cause]) -> tuple[Cause, int]:
    """The most frequent cause; ties go to the cause named first alphabetically, so a decision is stable."""
    return min(causes.items(), key=lambda item: (-item[1], item[0].value))


def _ungraded(causes: Counter[Cause], named: list[Cause]) -> tuple[str, ...]:
    return tuple(f"{cause.value}: {causes[cause]} trials ungraded" for cause in named)


def decide(summary: CalibrationSummary, rules: BandRules) -> Decision:
    """The decision for one validation round; see the module docstring for the rules."""
    decisive = summary.decisive_findings
    if decisive:
        return Reject(kind=RejectKind.TASK, reasons=tuple(_reason(finding) for finding in decisive))
    if isinstance(summary.status, Incomplete):
        causes = summary.status.causes
        host = sorted(cause for cause in causes if cause in HOST_CAUSES)
        if host:
            return Reject(kind=RejectKind.HOST, reasons=_ungraded(causes, host))
        rerunnable = Counter({cause: count for cause, count in causes.items() if cause in RERUNNABLE})
        if not rerunnable:
            return Reject(kind=RejectKind.TASK, reasons=_ungraded(causes, sorted(causes)))
        cause, count = _most_common(rerunnable)
        return Retry(cause=cause, count=count)
    band = [finding for finding in summary.findings if finding.kind in BAND_FINDINGS]
    if len(band) > 1:
        raise ValueError(f"one solve rate gives at most one band finding, got {[f.kind.value for f in band]}")
    if not band:
        return Accept(band=BandOutcome.IN_BAND)
    finding = band[0]
    if rules.choice(finding.kind) is BandChoice.ACCEPT:
        return Accept(band=BAND_OUTCOMES[finding.kind])
    return Reject(kind=RejectKind.TASK, reasons=(_reason(finding),))
