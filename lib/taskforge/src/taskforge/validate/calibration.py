# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The calibration summary of one validation round: typed findings a rule table consumes.

``summarize`` is pure: it reads a round's evidence and its policy and returns a
``CalibrationSummary`` holding the solver's reward statistics and a closed set of ``Finding``s.
Findings come from graded results even when the evidence is incomplete: a task defect is reported
whatever else is ungraded. Band findings (``TOO_HARD``, ``TOO_EASY``) need complete evidence and at
least ``k`` graded solver trials.

The summary keeps the fields of control replay and adversary trials (``controls_*``, ``roles``,
``assessments``, ``notes``); this round runs neither, so they are empty.

The summary records the policy digest and band, so a decision is reproducible from
``calibration.json`` alone and a policy change shows as a different summary.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from pydantic import TypeAdapter
from rolloutengine.contracts import RolloutData

from taskforge.atomic_file import write_atomic
from taskforge.builder.step import StepRole
from taskforge.spec.controls import Control
from taskforge.validate.adversary import AdversaryRole
from taskforge.validate.evidence import Complete, Evidence, Incomplete, RewardStats
from taskforge.validate.outcome import Cause, Graded, Outcome, TrialKind, Ungraded
from taskforge.validate.submissions import ClaimKind, final_reply, passing

DETAIL_CHARS = 2000
"""Longest command list or reply quoted per trial in a finding's detail."""


@dataclass(frozen=True)
class CalibrationBand:
    """Solve rates (``solved / graded``) that count as calibrated, inclusive."""

    min_solve_rate: float
    max_solve_rate: float

    def __post_init__(self) -> None:
        if not 0 <= self.min_solve_rate <= self.max_solve_rate <= 1:
            raise ValueError(f"A calibration band needs 0 <= min <= max <= 1, got {self}")


class FindingKind(StrEnum):
    CONTROL_VIOLATED = "control_violated"
    """A replayed control's grade missed its expectation: the grader or the controls are wrong."""
    SHORTCUT_PASSED = "shortcut_passed"
    """An adversary submission the grader accepted that is not the task's work (``tier_of``); the detail says
    where the hole is."""
    TASK_DEFECT = "task_defect"
    """A trial is ungraded for a cause re-running cannot change and that the task's build owns."""
    TOO_EASY = "too_easy"
    """The solver's solve rate is above the band."""
    TOO_HARD = "too_hard"
    """The solver's solve rate is below the band."""


DECISIVE: frozenset[FindingKind] = frozenset(
    {
        FindingKind.CONTROL_VIOLATED,
        FindingKind.SHORTCUT_PASSED,
        FindingKind.TASK_DEFECT,
    }
)
"""Findings retrying cannot improve."""

FINDING_ROLES: Mapping[FindingKind, tuple[StepRole, ...]] = {
    FindingKind.CONTROL_VIOLATED: (StepRole.GRADER, StepRole.CONTROLS),
    FindingKind.SHORTCUT_PASSED: (StepRole.GRADER, StepRole.CONTROLS),
    FindingKind.TOO_EASY: (StepRole.GRADER, StepRole.CONTROLS, StepRole.INSTRUCTIONS),
    FindingKind.TOO_HARD: (StepRole.INSTRUCTIONS, StepRole.FIXTURES),
}
"""The builder step roles each finding condemns; ``TASK_DEFECT`` takes its roles from ``DEFECT_ROLES``.

An adversary repair condemns the grader and the controls: whatever the hole (a lenient match, a trusted file, a
leaked answer), the grader accepted the candidate, and its negative control goes into the CONTROLS step."""

_GRADER = (StepRole.GRADER,)
DEFECT_ROLES: Mapping[Cause, tuple[StepRole, ...]] = {
    Cause.TASK_SETUP: (StepRole.ENVIRONMENT, StepRole.FIXTURES),
    Cause.GRADER_MISSING_REWARD: _GRADER,
    Cause.GRADER_EMPTY_REWARD: _GRADER,
    Cause.GRADER_INVALID_REWARD: _GRADER,
    Cause.GRADER_EXECUTION: _GRADER,
    Cause.GRADER_INFRA: _GRADER,
    Cause.VERIFIER_SKIPPED: _GRADER,
    Cause.NO_GRADE: _GRADER,
    Cause.INVALID_TASK: (StepRole.GRADER, StepRole.INSTRUCTIONS),
    Cause.GENERATION_LIMIT: (StepRole.INSTRUCTIONS, StepRole.FIXTURES),
    Cause.AGENT_TIMEOUT: (StepRole.INSTRUCTIONS, StepRole.FIXTURES),
}
"""The ungraded causes that are task defects, and the step roles that own each. The other causes
outside ``RERUNNABLE`` are not the task's: the host cannot run it (``MACHINE_UNSUPPORTED``,
``SUBMISSION_UNSUPPORTED``), the candidate's own code failed to import or be collected
(``CANDIDATE_CODE_ERROR``), or the model or classifier failed (``MODEL_REJECTED``, ``UNCLASSIFIED``)."""


@dataclass(frozen=True)
class Finding:
    """One defect the evidence shows, rendered for the author, and the step roles it condemns.

    ``new_controls`` holds adversary passes rendered as negative controls the revised program must ship.
    """

    kind: FindingKind
    detail: str
    roles: tuple[StepRole, ...]
    new_controls: tuple[Control, ...] = ()


class DefectTier(StrEnum):
    REPAIR = "repair"
    """A defect the program must fix: a decisive ``Finding``, with a negative control when the rule ships one."""
    NOTED = "noted"
    """Recorded for the author and the record as a note; never blocks an accept."""
    NONE = "none"
    """No defect."""


class Comparison(StrEnum):
    MATCH = "match"
    """The submission is one of the honest submissions (``compare``)."""
    MISMATCH = "mismatch"
    """It is not."""
    UNAVAILABLE = "unavailable"
    """A machine-state answer, no accepted submission, or no honest submission to compare with."""


@dataclass(frozen=True)
class AdversarySignals:
    """Every deterministic fact of one graded adversary trial that ``tier_of`` reads.

    Attributes:
        submissions: Verifier calls recorded.
        passes: Submissions that passed.
        budget_spent: ``submissions`` reached ``adversary_submissions``.
        first_pass: Ordinal of the first passing submission.
        exploit: Ordinal of the last passing submission: what the claim is about.
        claim: The verdict line's kind (``submissions.trial_claim``).
        why: The text after ``SHORTCUT:``.
        stop_reason: The rollout's stop reason.
        exhausted: ``stop_reason`` is in ``BUDGET_STOPS``.
        output_tokens: Served response tokens, reasoning included (``output_tokens``).
        turns: Model turns.
        shell_calls: Well-formed shell calls.
        inputs_consumed: ``TaskFacts.input_paths`` some shell command read for content (``consumed_inputs``).
        protected_supplied: Ordinals of passing submissions whose files include a ``TaskFacts.protected_paths`` path.
        fileless_passes: Ordinals of passing submissions with no files.
        mismatches: Ordinals of passing submissions whose reply compares ``MISMATCH`` (text tasks only).
        comparison: ``compare(exploit reply, honest submissions)``; ``UNAVAILABLE`` without an exploit or on a
            machine answer.
    """

    submissions: int
    passes: int
    budget_spent: bool
    first_pass: int | None
    exploit: int | None
    claim: ClaimKind
    why: str
    stop_reason: str
    exhausted: bool
    output_tokens: int
    turns: int
    shell_calls: int
    inputs_consumed: tuple[str, ...]
    protected_supplied: tuple[int, ...]
    fileless_passes: tuple[int, ...]
    mismatches: tuple[int, ...]
    comparison: Comparison


@dataclass(frozen=True)
class AdversaryAssessment:
    """One graded adversary trial's signals and the ruling of the first firing row of ``tier_of``."""

    role: AdversaryRole
    index: int
    signals: AdversarySignals
    tier: DefectTier
    rule: str
    reason: str
    subject: int | None


@dataclass(frozen=True)
class RoleStats:
    """One adversary role's trials.

    Attributes:
        required: Trials the policy asks for.
        graded: Graded trials.
        passes: Graded trials with a passing submission (``solved``).
        submissions: Verifier calls over graded trials.
        budget_spent: Graded trials that used the whole submission budget.
        claims: Graded trials per ``ClaimKind``; every kind is present.
        failed_audits: ``SHORTCUT`` claims whose accepted submission is an honest answer (``tier_of`` row 4).
        exhausted: Graded trials that stopped on output, turns, the total-turn deadline or context (``BUDGET_STOPS``).
        output_tokens: Served response tokens over every trial with a rollout.
        tiers: Graded trials per ``DefectTier``; every tier is present.
    """

    required: int
    graded: int
    passes: int
    submissions: int
    budget_spent: int
    claims: Mapping[ClaimKind, int]
    failed_audits: int
    exhausted: int
    output_tokens: int
    tiers: Mapping[DefectTier, int]


@dataclass(frozen=True)
class CalibrationSummary:
    """Everything review decides from, for one round of one draft. ``status`` covers every trial of every kind.

    ``findings`` holds the defects to repair and the band findings; ``notes`` holds the ``NOTED``
    adversary trials as ``SHORTCUT_PASSED`` findings without controls, never in ``findings``;
    ``assessments`` holds every graded adversary trial in ``AdversaryRole`` then index order.
    """

    task_digest: str
    policy_digest: str
    band: CalibrationBand
    status: Complete | Incomplete
    k: int
    solver: RewardStats
    solve_rate: float | None
    controls_met: tuple[str, ...]
    controls_violated: tuple[str, ...]
    controls_ungraded: tuple[tuple[str, Cause], ...]
    roles: Mapping[AdversaryRole, RoleStats]
    findings: tuple[Finding, ...]
    assessments: tuple[AdversaryAssessment, ...]
    notes: tuple[Finding, ...]

    @property
    def decisive(self) -> tuple[Finding, ...]:
        return tuple(finding for finding in self.findings if finding.kind in DECISIVE)

    @property
    def calibrated(self) -> bool:
        return isinstance(self.status, Complete) and not self.findings


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

    def trial_evidence(self) -> Evidence: ...


SUMMARY = TypeAdapter(CalibrationSummary)


def summarize(evidence: RoundEvidence, policy: SummaryPolicy) -> CalibrationSummary:
    """The calibration summary of ``evidence`` under ``policy``."""
    trials = evidence.trial_evidence()
    status = trials.status
    solver = trials.reward_stats(TrialKind.SOLVER)
    solve_rate = None if solver.graded == 0 else solver.solved / solver.graded
    findings = [
        *_defect_findings(evidence),
        *_band_findings(evidence.solver, solver, status, policy),
    ]
    return CalibrationSummary(
        task_digest=evidence.task_digest,
        policy_digest=policy.digest,
        band=policy.band,
        status=status,
        k=policy.k,
        solver=solver,
        solve_rate=solve_rate,
        controls_met=(),
        controls_violated=(),
        controls_ungraded=(),
        roles={},
        findings=tuple(findings),
        assessments=(),
        notes=(),
    )


def write_summary(path: Path, summary: CalibrationSummary) -> None:
    """Write ``summary`` as ``calibration.json``."""
    write_atomic(path, SUMMARY.dump_json(summary, indent=1))


def load_summary(path: Path) -> CalibrationSummary:
    return SUMMARY.validate_json(path.read_bytes())


def solved(outcome: Graded) -> bool:
    """Whether ``outcome`` passes, as ``Evidence.reward_stats`` counts it."""
    return passing(outcome.grade)


def tool_call_arguments(rollout: RolloutData) -> list[str]:
    """The arguments of every tool call the model made, in order."""
    return [
        call["function"]["arguments"] for step in rollout.steps for call in step.turn.message.get("tool_calls") or ()
    ]


def _clip(text: str) -> str:
    return text if len(text) <= DETAIL_CHARS else f"{text[:DETAIL_CHARS]} [... {len(text) - DETAIL_CHARS} chars cut]"


def _trial_lines(name: str, outcome: Outcome) -> list[str]:
    rollout = outcome.rollout
    if isinstance(outcome, Graded):
        head = f"{name}: status {outcome.grade.status}, reward {outcome.reward}, timed out {outcome.timed_out}"
    else:
        head = f"{name}: ungraded ({outcome.cause}): {_clip(outcome.detail)}"
    if rollout is None:
        return [head]
    return [
        f"{head}, stop reason {rollout.stop_reason}, {len(rollout.steps)} model turns",
        f"  commands: {_clip(json.dumps(tool_call_arguments(rollout), ensure_ascii=False))}",
        f"  final reply: {_clip(json.dumps(final_reply(rollout), ensure_ascii=False))}",
    ]


def _named_outcomes(evidence: RoundEvidence) -> list[tuple[str, Outcome]]:
    return [(f"solver/{index}", outcome) for index, outcome in enumerate(evidence.solver)]


def _defect_findings(evidence: RoundEvidence) -> list[Finding]:
    by_cause: dict[Cause, list[tuple[str, Ungraded]]] = {}
    for name, outcome in _named_outcomes(evidence):
        if not isinstance(outcome, Ungraded) or outcome.cause not in DEFECT_ROLES:
            continue
        by_cause.setdefault(outcome.cause, []).append((name, outcome))
    return [
        Finding(
            FindingKind.TASK_DEFECT,
            "\n".join(
                [
                    f"{len(trials)} trial(s) ungraded for {cause}.",
                    *(line for name, outcome in trials for line in _trial_lines(name, outcome)),
                ]
            ),
            DEFECT_ROLES[cause],
        )
        for cause, trials in by_cause.items()
    ]


def _band_findings(
    outcomes: Sequence[Outcome], stats: RewardStats, status: Complete | Incomplete, policy: SummaryPolicy
) -> list[Finding]:
    if not isinstance(status, Complete) or stats.graded < policy.k:
        return []
    rate = stats.solved / stats.graded
    band = policy.band
    if band.min_solve_rate <= rate <= band.max_solve_rate:
        return []
    kind = FindingKind.TOO_HARD if rate < band.min_solve_rate else FindingKind.TOO_EASY
    head = (
        f"The solver solved {stats.solved} of {stats.graded} graded trials ({rate:.3f}); the calibrated band is "
        f"[{band.min_solve_rate}, {band.max_solve_rate}]. {stats.timed_out} trial(s) hit the total-turn deadline."
    )
    shown = [
        (index, outcome)
        for index, outcome in enumerate(outcomes)
        if kind is FindingKind.TOO_HARD or (isinstance(outcome, Graded) and solved(outcome))
    ]
    lines = [head, *(line for index, outcome in shown for line in _trial_lines(f"solver/{index}", outcome))]
    return [Finding(kind, "\n".join(lines), FINDING_ROLES[kind])]
