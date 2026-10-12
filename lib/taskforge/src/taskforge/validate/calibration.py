# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The calibration summary of one validation round: typed findings a rule table consumes.

``summarize`` is pure: it reads a round's evidence and its policy and returns a
``CalibrationSummary`` holding the solver's reward statistics and a closed set of ``Finding``s.
Findings come from graded results even when the evidence is incomplete: a task defect is reported
whatever else is ungraded. Band findings (``TOO_HARD``, ``TOO_EASY``) need complete evidence and at
least ``k`` graded solver trials.

The summary records the policy digest and band, so a decision is reproducible from
``calibration.json`` alone and a policy change shows as a different summary.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from pydantic import TypeAdapter

from taskforge.atomic_file import write_atomic
from taskforge.validate.evidence import Complete, Evidence, Incomplete, RewardStats
from taskforge.validate.outcome import Cause, Outcome, TrialKind, Ungraded


@dataclass(frozen=True)
class CalibrationBand:
    """Solve rates (``solved / graded``) that count as calibrated, inclusive."""

    min_solve_rate: float
    max_solve_rate: float

    def __post_init__(self) -> None:
        if not 0 <= self.min_solve_rate <= self.max_solve_rate <= 1:
            raise ValueError(f"A calibration band needs 0 <= min <= max <= 1, got {self}")


class FindingKind(StrEnum):
    TASK_DEFECT = "task_defect"
    """A trial is ungraded for a cause re-running cannot change and that the task's build owns."""
    TOO_EASY = "too_easy"
    """The solver's solve rate is above the band."""
    TOO_HARD = "too_hard"
    """The solver's solve rate is below the band."""


DECISIVE: frozenset[FindingKind] = frozenset({FindingKind.TASK_DEFECT})
"""Findings retrying cannot improve."""

DEFECT_CAUSES: frozenset[Cause] = frozenset(
    {
        Cause.TASK_SETUP,
        Cause.GRADER_MISSING_REWARD,
        Cause.GRADER_EMPTY_REWARD,
        Cause.GRADER_INVALID_REWARD,
        Cause.GRADER_EXECUTION,
        Cause.GRADER_INFRA,
        Cause.VERIFIER_SKIPPED,
        Cause.NO_GRADE,
        Cause.INVALID_TASK,
        Cause.GENERATION_LIMIT,
        Cause.AGENT_TIMEOUT,
    }
)
"""The ungraded causes that are task defects. The other causes outside ``RETRYABLE`` are not the
task's: the host cannot run it (``MACHINE_UNSUPPORTED``, ``SUBMISSION_UNSUPPORTED``), or the model
or classifier failed (``MODEL_REJECTED``, ``UNCLASSIFIED``)."""


@dataclass(frozen=True)
class Finding:
    """One defect the evidence shows; ``detail`` is one sentence naming its cause or solve rate and counts."""

    kind: FindingKind
    detail: str


@dataclass(frozen=True)
class CalibrationSummary:
    """Everything review decides from, for one round of one draft. ``status`` covers every trial."""

    task_digest: str
    policy_digest: str
    band: CalibrationBand
    status: Complete | Incomplete
    k: int
    solver: RewardStats
    solve_rate: float | None
    findings: tuple[Finding, ...]

    @property
    def decisive_findings(self) -> tuple[Finding, ...]:
        return tuple(finding for finding in self.findings if finding.kind in DECISIVE)


class SummaryPolicy(Protocol):
    """What ``summarize`` reads of ``validate.run.ValidationPolicy``."""

    @property
    def k(self) -> int: ...

    @property
    def band(self) -> CalibrationBand: ...

    @property
    def digest(self) -> str: ...


class RoundEvidence(Protocol):
    """What ``summarize`` reads of ``validate.run.ValidationEvidence``."""

    @property
    def task_digest(self) -> str: ...

    @property
    def solver(self) -> tuple[Outcome, ...]: ...


SUMMARY = TypeAdapter(CalibrationSummary)


def summarize(evidence: RoundEvidence, policy: SummaryPolicy) -> CalibrationSummary:
    """The calibration summary of ``evidence`` under ``policy``."""
    trials = Evidence({TrialKind.SOLVER: evidence.solver})
    status = trials.status
    solver = trials.reward_stats(TrialKind.SOLVER)
    solve_rate = None if solver.graded == 0 else solver.solved / solver.graded
    findings = (*_defect_findings(evidence.solver), *_band_findings(solver, status, policy))
    return CalibrationSummary(
        task_digest=evidence.task_digest,
        policy_digest=policy.digest,
        band=policy.band,
        status=status,
        k=policy.k,
        solver=solver,
        solve_rate=solve_rate,
        findings=findings,
    )


def write_summary(path: Path, summary: CalibrationSummary) -> None:
    """Write ``summary`` as ``calibration.json``."""
    write_atomic(path, SUMMARY.dump_json(summary, indent=1))


def load_summary(path: Path) -> CalibrationSummary:
    return SUMMARY.validate_json(path.read_bytes())


def _defect_findings(outcomes: Sequence[Outcome]) -> list[Finding]:
    """One ``TASK_DEFECT`` per defect cause, naming how many trials that cause left ungraded."""
    causes = Counter(outcome.cause for outcome in outcomes if isinstance(outcome, Ungraded))
    return [
        Finding(FindingKind.TASK_DEFECT, f"{count} trial(s) ungraded for {cause}.")
        for cause, count in causes.items()
        if cause in DEFECT_CAUSES
    ]


def _band_findings(stats: RewardStats, status: Complete | Incomplete, policy: SummaryPolicy) -> list[Finding]:
    """The band finding of a complete round with ``k`` graded trials."""
    if not isinstance(status, Complete) or stats.graded < policy.k:
        return []
    rate = stats.solved / stats.graded
    band = policy.band
    if band.min_solve_rate <= rate <= band.max_solve_rate:
        return []
    kind = FindingKind.TOO_HARD if rate < band.min_solve_rate else FindingKind.TOO_EASY
    detail = (
        f"The solver solved {stats.solved} of {stats.graded} graded trials ({rate:.3f}); the calibrated band is "
        f"[{band.min_solve_rate}, {band.max_solve_rate}]. {stats.timed_out} trial(s) hit the total-turn deadline."
    )
    return [Finding(kind, detail)]
