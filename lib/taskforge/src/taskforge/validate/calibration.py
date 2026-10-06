# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The calibration summary of one validation round: typed findings a rule table consumes.

``summarize`` is pure: it reads a round's evidence and its policy and returns a
``CalibrationSummary`` holding the solver's reward statistics, the control verdicts, per-role
adversary statistics and a closed set of ``Finding``s. Findings come from graded results even when
the evidence is incomplete: a violated control, an adversary pass or a task defect is reported
whatever else is ungraded. Band findings (``TOO_HARD``, ``TOO_EASY``) need complete evidence and at
least ``k`` graded solver trials. An adversary pass that ends on its role's sentinel reply is no
finding: the role concluded there was nothing to exploit, and a machine-state task graded the
honest work it did first. A ``SHORTCUT`` or ``LEAK`` pass is also rendered as a negative control the
revised program must ship, so the next round's control replay proves the fix.

The summary records the policy digest and band, so a decision is reproducible from
``calibration.json`` alone and a policy change shows as a different summary.
"""

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from pydantic import TypeAdapter
from rolloutengine.contracts import RolloutData
from taskcompendium.grading_result import Outcome as GradeStatus
from taskcompendium.models import AssistantToolCalls, TextMessage

from taskforge.build.step import StepRole
from taskforge.canonical import write_atomic
from taskforge.spec.controls import (
    REJECTION_CEILING,
    Control,
    ControlCategory,
    ControlKind,
    Expectation,
    Transcript,
    reply,
)
from taskforge.validate.adversary import SENTINEL_REPLIES, AdversaryRole
from taskforge.validate.controls import WORKSPACE_REPLY, ControlOutcome, ControlVerdict
from taskforge.validate.evidence import Complete, Evidence, Incomplete, RewardStats
from taskforge.validate.outcome import Cause, Graded, Outcome, TrialKind, Ungraded

DETAIL_CHARS = 2000
"""Longest command list or reply quoted per trial in a finding's detail."""
SHELL = "shell"


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
    """A shortcut adversary was graded correct without doing the task."""
    LEAK_PASSED = "leak_passed"
    """A leak adversary was graded correct from what the machine or instruction revealed."""
    AMBIGUOUS = "ambiguous"
    """An answer under the reading least likely intended was graded correct."""
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
        FindingKind.LEAK_PASSED,
        FindingKind.AMBIGUOUS,
        FindingKind.TASK_DEFECT,
    }
)
"""Findings retrying cannot improve."""

FINDING_ROLES: Mapping[FindingKind, tuple[StepRole, ...]] = {
    FindingKind.CONTROL_VIOLATED: (StepRole.GRADER, StepRole.CONTROLS),
    FindingKind.SHORTCUT_PASSED: (StepRole.GRADER, StepRole.CONTROLS),
    FindingKind.LEAK_PASSED: (StepRole.FIXTURES, StepRole.ENVIRONMENT, StepRole.INSTRUCTIONS),
    FindingKind.AMBIGUOUS: (StepRole.INSTRUCTIONS,),
    FindingKind.TOO_EASY: (StepRole.GRADER, StepRole.CONTROLS, StepRole.INSTRUCTIONS),
    FindingKind.TOO_HARD: (StepRole.INSTRUCTIONS, StepRole.FIXTURES),
}
"""The builder step roles each finding condemns; ``TASK_DEFECT`` takes its roles from ``DEFECT_ROLES``."""

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
``SUBMISSION_UNSUPPORTED``) or the model or classifier failed (``MODEL_REJECTED``, ``UNCLASSIFIED``)."""


@dataclass(frozen=True)
class Finding:
    """One defect the evidence shows, rendered for the author, and the step roles it condemns.

    ``new_controls`` holds adversary passes rendered as negative controls the revised program must ship.
    """

    kind: FindingKind
    detail: str
    roles: tuple[StepRole, ...]
    new_controls: tuple[Control, ...] = ()


@dataclass(frozen=True)
class RoleStats:
    """One adversary role's trials: ``sentinel_replies`` counts final replies equal to its
    ``SENTINEL_REPLIES`` entry, which measures how often the role found nothing."""

    required: int
    graded: int
    passes: int
    sentinel_replies: int


@dataclass(frozen=True)
class CalibrationSummary:
    """Everything review decides from, for one round of one draft. ``status`` covers every trial of every kind."""

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
    def adversary_k(self) -> int: ...

    @property
    def band(self) -> CalibrationBand: ...

    @property
    def digest(self) -> str: ...


class RoundEvidence(Protocol):
    """What ``summarize`` reads of ``validate.run.ValidationEvidence``."""

    @property
    def task_digest(self) -> str: ...

    @property
    def controls(self) -> tuple[ControlOutcome, ...]: ...

    @property
    def solver(self) -> tuple[Outcome, ...]: ...

    @property
    def adversaries(self) -> Mapping[AdversaryRole, tuple[Outcome, ...]]: ...

    def trial_evidence(self) -> Evidence: ...


SUMMARY = TypeAdapter(CalibrationSummary)


def summarize(evidence: RoundEvidence, policy: SummaryPolicy) -> CalibrationSummary:
    """The calibration summary of ``evidence`` under ``policy``."""
    trials = evidence.trial_evidence()
    status = trials.status
    solver = trials.reward_stats(TrialKind.SOLVER)
    solve_rate = None if solver.graded == 0 else solver.solved / solver.graded
    findings = [
        *_control_findings(evidence.controls),
        *_adversary_findings(evidence.adversaries, evidence.task_digest),
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
        controls_met=_control_ids(evidence.controls, ControlVerdict.MET),
        controls_violated=_control_ids(evidence.controls, ControlVerdict.VIOLATED),
        controls_ungraded=tuple(
            (c.control.id, c.outcome.cause) for c in evidence.controls if isinstance(c.outcome, Ungraded)
        ),
        roles={role: _role_stats(role, outcomes, policy.adversary_k) for role, outcomes in evidence.adversaries.items()},
        findings=tuple(findings),
    )


def write_summary(path: Path, summary: CalibrationSummary) -> None:
    """Write ``summary`` as ``calibration.json``."""
    write_atomic(path, SUMMARY.dump_json(summary, indent=1))


def load_summary(path: Path) -> CalibrationSummary:
    return SUMMARY.validate_json(path.read_bytes())


def solved(outcome: Graded) -> bool:
    """Whether ``outcome`` passes, as ``Evidence.reward_stats`` counts it."""
    return Evidence({TrialKind.SOLVER: (outcome,)}).reward_stats(TrialKind.SOLVER).solved == 1


def final_reply(rollout: RolloutData) -> str | None:
    """The content of the rollout's last model turn when it is a text reply without tool calls."""
    if not rollout.steps:
        return None
    message = rollout.steps[-1].turn.message
    return None if message.get("tool_calls") else str(message.get("content") or "")


def commands(rollout: RolloutData) -> list[str]:
    """The arguments of every tool call the model made, in order."""
    return [
        call["function"]["arguments"] for step in rollout.steps for call in step.turn.message.get("tool_calls") or ()
    ]


def transcript_control(role: AdversaryRole, index: int, task_digest: str, rollout: RolloutData) -> Control:
    """A passing adversary rollout as a negative control: its model turns replayed, demanding no credit.

    Call ids are renumbered so they are unique across the transcript. A rollout whose last turn is
    shell calls (a task graded on machine state) gets ``WORKSPACE_REPLY`` as its final reply.

    Raises:
        ValueError: the rollout has no model turn, or a tool call's arguments are not a JSON object.
    """
    turns: list[TextMessage | AssistantToolCalls] = []
    serial = 0
    for step in rollout.steps:
        message = step.turn.message
        calls = message.get("tool_calls") or ()
        if not calls:
            turns.append(reply(str(message.get("content") or "")))
            continue
        decoded = []
        for call in calls:
            serial += 1
            decoded.append(
                {
                    "call_id": f"c{serial}",
                    "name": call["function"]["name"],
                    "arguments": json.loads(call["function"]["arguments"]),
                }
            )
        turns.append(AssistantToolCalls.model_validate({"calls": decoded, "content": message.get("content") or None}))
    if not turns:
        raise ValueError("The rollout has no model turn")
    last = turns[-1]
    if isinstance(last, AssistantToolCalls) and all(call.name == SHELL for call in last.calls):
        turns.append(reply(WORKSPACE_REPLY))
    return Control(
        id=f"adv-{role}-{index}-{task_digest[:8]}",
        kind=ControlKind.NEGATIVE,
        category=ControlCategory.REWARD_HACK,
        author=f"adversary/{role}/{index}",
        payload=Transcript(tuple(turns)),
        expect=Expectation(status=GradeStatus.GRADED, reward_max=REJECTION_CEILING),
    )


def _clip(text: str) -> str:
    return text if len(text) <= DETAIL_CHARS else f"{text[:DETAIL_CHARS]} [... {len(text) - DETAIL_CHARS} chars cut]"


def _trial_lines(name: str, outcome: Outcome) -> list[str]:
    rollout = outcome.rollout
    if isinstance(outcome, Graded):
        head = f"{name}: graded {outcome.grade.status}, reward {outcome.reward}, timed out {outcome.timed_out}"
    else:
        head = f"{name}: ungraded ({outcome.cause}): {_clip(outcome.detail)}"
    if rollout is None:
        return [head]
    return [
        f"{head}, stop reason {rollout.stop_reason}, {len(rollout.steps)} model turns",
        f"  commands: {_clip(json.dumps(commands(rollout), ensure_ascii=False))}",
        f"  final reply: {_clip(json.dumps(final_reply(rollout), ensure_ascii=False))}",
    ]


def _control_ids(controls: Sequence[ControlOutcome], verdict: ControlVerdict) -> tuple[str, ...]:
    return tuple(c.control.id for c in controls if c.verdict is verdict)


def _control_findings(controls: Sequence[ControlOutcome]) -> list[Finding]:
    findings = []
    for c in controls:
        if c.verdict is not ControlVerdict.VIOLATED:
            continue
        control = c.control
        lines = [
            f"Control {control.id} ({control.kind}, {control.category}, by {control.author}) expects {control.expect}.",
            *_trial_lines(f"control/{control.id}", c.outcome),
        ]
        findings.append(
            Finding(FindingKind.CONTROL_VIOLATED, "\n".join(lines), FINDING_ROLES[FindingKind.CONTROL_VIOLATED])
        )
    return findings


PASS_KINDS: Mapping[AdversaryRole, FindingKind] = {
    AdversaryRole.SHORTCUT: FindingKind.SHORTCUT_PASSED,
    AdversaryRole.LEAK: FindingKind.LEAK_PASSED,
    AdversaryRole.AMBIGUITY: FindingKind.AMBIGUOUS,
}
CONTROL_ROLES = frozenset({AdversaryRole.SHORTCUT, AdversaryRole.LEAK})
"""Roles whose passes become negative controls; an ambiguity pass solved the task under some reading."""


def _adversary_findings(adversaries: Mapping[AdversaryRole, Sequence[Outcome]], task_digest: str) -> list[Finding]:
    findings = []
    for role, outcomes in _by_role(adversaries):
        for index, outcome in enumerate(outcomes):
            if not (isinstance(outcome, Graded) and solved(outcome)) or _declared_nothing(role, outcome):
                continue
            kind = PASS_KINDS[role]
            lines = [f"The {role} adversary was graded as passing.", *_trial_lines(f"adversary/{role}/{index}", outcome)]
            new_controls: tuple[Control, ...] = ()
            if role in CONTROL_ROLES:
                try:
                    new_controls = (transcript_control(role, index, task_digest, outcome.rollout),)
                except ValueError as error:
                    lines.append(f"  (not rendered as a control: {error})")
            findings.append(Finding(kind, "\n".join(lines), FINDING_ROLES[kind], new_controls))
    return findings


def _by_role(adversaries: Mapping[AdversaryRole, Sequence[Outcome]]) -> list[tuple[AdversaryRole, Sequence[Outcome]]]:
    """The roles' outcomes in ``AdversaryRole`` order, so findings do not depend on the mapping's order."""
    return [(role, adversaries[role]) for role in AdversaryRole if role in adversaries]


def _declared_nothing(role: AdversaryRole, outcome: Graded) -> bool:
    """Whether the trial ended on its role's sentinel reply: a machine-state task can pass on the honest work
    an adversary did before it concluded there was nothing to exploit, and that pass is no finding."""
    return (final_reply(outcome.rollout) or "").strip() == SENTINEL_REPLIES[role]


def _named_outcomes(evidence: RoundEvidence) -> list[tuple[str, Outcome]]:
    return [
        *((f"control/{c.control.id}", c.outcome) for c in evidence.controls),
        *((f"solver/{index}", outcome) for index, outcome in enumerate(evidence.solver)),
        *(
            (f"adversary/{role}/{index}", outcome)
            for role, outcomes in _by_role(evidence.adversaries)
            for index, outcome in enumerate(outcomes)
        ),
    ]


def _defect_findings(evidence: RoundEvidence) -> list[Finding]:
    by_cause: dict[Cause, list[tuple[str, Ungraded]]] = {}
    for name, outcome in _named_outcomes(evidence):
        if isinstance(outcome, Ungraded) and outcome.cause in DEFECT_ROLES:
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
        f"[{band.min_solve_rate}, {band.max_solve_rate}]. {stats.timed_out} trial(s) hit the agent deadline."
    )
    shown = [
        (index, outcome)
        for index, outcome in enumerate(outcomes)
        if kind is FindingKind.TOO_HARD or (isinstance(outcome, Graded) and solved(outcome))
    ]
    lines = [head, *(line for index, outcome in shown for line in _trial_lines(f"solver/{index}", outcome))]
    return [Finding(kind, "\n".join(lines), FINDING_ROLES[kind])]


def _role_stats(role: AdversaryRole, outcomes: Sequence[Outcome], required: int) -> RoleStats:
    graded = [outcome for outcome in outcomes if isinstance(outcome, Graded)]
    replies = Counter(
        (final_reply(outcome.rollout) or "").strip() for outcome in outcomes if outcome.rollout is not None
    )
    return RoleStats(
        required=required,
        graded=len(graded),
        passes=sum(solved(outcome) for outcome in graded),
        sentinel_replies=replies[SENTINEL_REPLIES[role]],
    )
