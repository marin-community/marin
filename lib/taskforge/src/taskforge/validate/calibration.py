# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The calibration summary of one validation round: typed findings a rule table consumes.

``summarize`` is pure: it reads a round's evidence and its policy and returns a
``CalibrationSummary`` holding the solver's reward statistics, the control verdicts, per-role
adversary statistics and a closed set of ``Finding``s. Findings come from graded results even when
the evidence is incomplete: a violated control, a repair-tier adversary pass or a task defect is
reported whatever else is ungraded. Band findings (``TOO_HARD``, ``TOO_EASY``) need complete
evidence and at least ``k`` graded solver trials.

Every graded adversary trial is assessed into a ``DefectTier`` by a coded rule table (``tier_of``)
over deterministic transcript signals (``AdversarySignals``): which input files a command read or
wrote, whether any shell command ran, whether the final reply ends on the role's sentinel line, and
on text tasks whether the submission matches an honest one. The adversary's own account is never
evidence and no model judges. A ``REPAIR`` pass is a decisive finding; a shortcut or leak repair,
and every pass whose answer did not come from the task's work, is also rendered as a negative
control the revised program must ship, so the next round's control replay proves the fix. A
``NOTED`` pass (an adversary that read the inputs and submitted the real answer, or an ambiguity
pass that cannot be compared) is a note: recorded beside the findings, never blocking an accept.

The summary records the policy digest and band, so a decision is reproducible from
``calibration.json`` alone and a policy change shows as a different summary.
"""

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from pydantic import TypeAdapter
from rolloutengine.contracts import LENGTH_STOP_REASON, MAX_TURNS_STOP_REASON, RolloutData
from rolloutengine.shell_tool import SHELL_TOOL_NAME
from taskcompendium.grading_result import Outcome as GradeStatus
from taskcompendium.models import AssistantToolCalls, TaskSpec, TextMessage

from taskforge.build.step import StepRole
from taskforge.canonical import write_atomic
from taskforge.spec.controls import (
    REJECTION_CEILING,
    Control,
    ControlCategory,
    ControlConcern,
    ControlKind,
    Expectation,
    Transcript,
    reply,
)
from taskforge.spec.draft import MACHINE_ANSWER_TYPES
from taskforge.validate.adversary import SENTINEL_REPLIES, AdversaryRole
from taskforge.validate.controls import WORKSPACE_REPLY, ControlOutcome, ControlVerdict
from taskforge.validate.evidence import Complete, Evidence, Incomplete, RewardStats
from taskforge.validate.outcome import Cause, Graded, Outcome, TrialKind, Ungraded

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
    """An adversary pass whose answer did not come from the task's work (any role), or a shortcut pass on text
    no honest run produced."""
    LEAK_PASSED = "leak_passed"
    """A leak pass on text no honest run produced: what the machine or instruction revealed was accepted."""
    AMBIGUOUS = "ambiguous"
    """An ambiguity pass whose answer under its stated reading differs from the honest one."""
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
    """A machine-state answer, no text submission, or no honest submission to compare with."""


BUDGET_STOPS = frozenset({LENGTH_STOP_REASON, MAX_TURNS_STOP_REASON})
"""Stop reasons that mean the adversary ran out of output tokens or turns."""


@dataclass(frozen=True)
class TaskFacts:
    """What the adversary signals read of the task.

    Attributes:
        machine_answer: The answer is the machine state (``MACHINE_ANSWER_TYPES``), not a text submission.
        input_paths: The environment's files, in order.
        protected_paths: ``input_paths`` then the verifier's files, each once, in order.
    """

    machine_answer: bool
    input_paths: tuple[str, ...]
    protected_paths: tuple[str, ...]


def task_facts(task: TaskSpec) -> TaskFacts:
    inputs = tuple(f.path for f in task.environment.files)
    protected = tuple(dict.fromkeys((*inputs, *(f.path for f in task.verifier.files))))
    return TaskFacts(task.answer_type in MACHINE_ANSWER_TYPES, inputs, protected)


@dataclass(frozen=True)
class AdversarySignals:
    """Every deterministic fact of one graded adversary outcome that ``tier_of`` reads.

    Attributes:
        passed: ``solved(outcome)``.
        gave_up: The final reply ends on the role's sentinel line (``gave_up``).
        stop_reason: The rollout's stop reason.
        budget_exhausted: ``stop_reason`` is in ``BUDGET_STOPS``.
        output_tokens: Served response tokens, reasoning included (``loss_mask.count(1)``).
        turns: Model turns.
        shell_calls: Tool calls named ``SHELL_TOOL_NAME``.
        inputs_consumed: ``TaskFacts.input_paths`` some command read for content (``consumed_inputs``).
        inputs_written: ``TaskFacts.protected_paths`` some command wrote (``written_inputs``).
        submission: On a text task, the final reply; ``None`` on a machine task or when the last turn is tool calls.
        comparison: ``compare(submission, honest submissions)`` on a text task, else ``UNAVAILABLE``.
    """

    passed: bool
    gave_up: bool
    stop_reason: str
    budget_exhausted: bool
    output_tokens: int
    turns: int
    shell_calls: int
    inputs_consumed: tuple[str, ...]
    inputs_written: tuple[str, ...]
    submission: str | None
    comparison: Comparison


@dataclass(frozen=True)
class AdversaryAssessment:
    """One graded adversary trial's signals and the tier the first firing row of ``tier_of`` gave it."""

    role: AdversaryRole
    index: int
    signals: AdversarySignals
    tier: DefectTier
    rule: str
    reason: str


@dataclass(frozen=True)
class RoleStats:
    """One adversary role's trials.

    Attributes:
        required: Trials the policy asks for.
        graded: Graded trials.
        passes: Graded trials that passed.
        gave_up: Trials with a rollout whose final reply ends on the role's sentinel line.
        exhausted: Graded trials that stopped on their output budget or ``max_turns`` (``BUDGET_STOPS``).
        output_tokens: Served response tokens over every trial with a rollout.
        tiers: Graded trials per ``DefectTier``; every tier is present.
    """

    required: int
    graded: int
    passes: int
    gave_up: int
    exhausted: int
    output_tokens: int
    tiers: Mapping[DefectTier, int]


@dataclass(frozen=True)
class CalibrationSummary:
    """Everything review decides from, for one round of one draft. ``status`` covers every trial of every kind.

    ``findings`` holds the defects to repair and the band findings; ``notes`` holds the ``NOTED``
    adversary passes as findings of the role's pass kind without controls, never in ``findings``;
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

    @property
    def facts(self) -> TaskFacts: ...

    def trial_evidence(self) -> Evidence: ...


SUMMARY = TypeAdapter(CalibrationSummary)


def summarize(evidence: RoundEvidence, policy: SummaryPolicy) -> CalibrationSummary:
    """The calibration summary of ``evidence`` under ``policy``."""
    trials = evidence.trial_evidence()
    status = trials.status
    solver = trials.reward_stats(TrialKind.SOLVER)
    solve_rate = None if solver.graded == 0 else solver.solved / solver.graded
    adversary_findings, notes, assessments = _adversary_findings(evidence)
    findings = [
        *_control_findings(evidence.controls),
        *adversary_findings,
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
        roles={
            role: _role_stats(role, outcomes, policy.adversary_k, assessments)
            for role, outcomes in evidence.adversaries.items()
        },
        findings=tuple(findings),
        assessments=tuple(assessments),
        notes=tuple(notes),
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
    if isinstance(last, AssistantToolCalls) and all(call.name == SHELL_TOOL_NAME for call in last.calls):
        turns.append(reply(WORKSPACE_REPLY))
    return Control(
        id=f"adv-{role}-{index}-{task_digest[:8]}",
        kind=ControlKind.NEGATIVE,
        category=ControlCategory.REWARD_HACK,
        concern=ControlConcern.SHORTCUT,
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


PASS_KINDS: Mapping[AdversaryRole, FindingKind] = {
    AdversaryRole.SHORTCUT: FindingKind.SHORTCUT_PASSED,
    AdversaryRole.LEAK: FindingKind.LEAK_PASSED,
    AdversaryRole.AMBIGUITY: FindingKind.AMBIGUOUS,
}
CONTROL_ROLES = frozenset({AdversaryRole.SHORTCUT, AdversaryRole.LEAK})
"""Roles whose row-5 repairs become negative controls; an ambiguity pass solved the task under some reading."""

CONTROL_RULES = frozenset({"1", "2", "4"})
"""Rows whose repairs always ship the transcript as a negative control: the answer did not come from the task's work."""

SEGMENT_SPLIT = re.compile(r"\|\||&&|;|\||\n|\$\(|`")
"""Boundaries between shell segments, including command substitutions, so the inner command's first word is read."""

METADATA_COMMANDS = frozenset(
    {
        "ls", "stat", "wc", "du", "file", "find", "test", "[", "[[", "type", "which", "realpath", "readlink",
        "basename", "dirname", "cp", "mv", "rm", "chmod", "chown", "touch", "ln", "mkdir", "md5sum", "sha256sum",
        "cksum", "echo", "printf", "true", ":", "cd", "export", "env", "tee",
    }
)  # fmt: skip
"""First words of a segment that do not deliver a file's content to the model."""

REDIRECT_TARGET = re.compile(r">>?\s*\S+")
ASSIGNMENTS = re.compile(r"^(?:[A-Za-z_][A-Za-z0-9_]*=\S*\s+)*")
NUMERIC_TOKEN = re.compile(r"-?\d+(?:\.\d+)?")
PATH_CHARS = r"[\w./-]"


def last_line(text: str) -> str:
    """The last non-empty line of ``text``, stripped; ``""`` when there is none."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def gave_up(role: AdversaryRole, rollout: RolloutData) -> bool:
    """The final reply is a text reply whose last non-empty line is ``SENTINEL_REPLIES[role]``."""
    text = final_reply(rollout)
    return text is not None and last_line(text) == SENTINEL_REPLIES[role]


def _shell_command(arguments: str) -> str | None:
    """The ``command`` in one shell call's raw arguments, or None when they are not a JSON object with a string
    ``command``: a call cut at the output budget, or one the engine answered with an error and never ran."""
    try:
        decoded = json.loads(arguments)
    except ValueError:
        return None
    command = decoded.get("command") if isinstance(decoded, dict) else None
    return command if isinstance(command, str) else None


def shell_commands(rollout: RolloutData) -> list[str]:
    """The ``command`` of every well-formed shell call, in order. Malformed calls ran nothing and are skipped."""
    found = (
        _shell_command(call["function"]["arguments"])
        for step in rollout.steps
        for call in step.turn.message.get("tool_calls") or ()
        if call["function"]["name"] == SHELL_TOOL_NAME
    )
    return [command for command in found if command is not None]


def _path(path: str) -> str:
    return rf"(?<!{PATH_CHARS}){re.escape(path)}(?!{PATH_CHARS})"


def _consumes(segment: str, path: re.Pattern[str]) -> bool:
    body = REDIRECT_TARGET.sub("", segment)
    if not path.search(body):
        return False
    words = ASSIGNMENTS.sub("", body.strip()).split()
    return bool(words) and words[0] not in METADATA_COMMANDS


def consumed_inputs(commands: Sequence[str], input_paths: Sequence[str]) -> tuple[str, ...]:
    """Input paths that appear, outside a ``>``/``>>`` target, in a segment whose first word after leading
    ``NAME=value`` assignments is not in ``METADATA_COMMANDS``.

    ``cat P``, ``head -c 200 P``, ``od -c P``, ``awk ... P > out``, ``tr '\\n' '+' < P``,
    ``while read ...; done < P`` and ``printf '%s' "$(awk ... P)" > out`` consume; ``wc -l P``,
    ``ls -la P``, ``stat P``, ``cp P P.bak`` and ``echo 60 > P`` do not.
    """
    segments = [segment for command in commands for segment in SEGMENT_SPLIT.split(command)]
    consumed = []
    for path in input_paths:
        pattern = re.compile(_path(path))
        if any(_consumes(segment, pattern) for segment in segments):
            consumed.append(path)
    return tuple(consumed)


def _write_pattern(path: str) -> re.Pattern[str]:
    p = _path(path)
    return re.compile(
        "|".join(
            (
                rf">>?\s*{p}",
                rf"\btee\s+(?:-a\s+)?{p}",
                rf"\b(?:cp|mv)\s+(?:-\S+\s+)*\S+\s+{p}",
                rf"\bsed\s+-i\S*\s.*\s{p}",
                rf"\brm\s+(?:-\S+\s+)*{p}",
                rf"\btruncate\s.*\s{p}",
                rf"\bdd\s.*\bof={p}",
                rf"\binstall\s.*\s{p}",
            )
        )
    )


def written_inputs(commands: Sequence[str], protected_paths: Sequence[str]) -> tuple[str, ...]:
    """Protected paths matched as a write target.

    A target is ``>`` or ``>>`` then the path; ``tee [-a] path``; ``cp|mv [-flags] <src> path``;
    ``sed -i... path``; ``rm [-flags] path``; ``truncate ... path``; ``dd ... of=path``;
    ``install ... path``. The path must end at a non-path character, so ``numbers.txt.bak`` does not
    match ``numbers.txt``, and a path as a ``cp``/``mv`` source does not match.
    """
    return tuple(path for path in protected_paths if any(_write_pattern(path).search(command) for command in commands))


def normalised(text: str) -> str:
    """``text`` with whitespace collapsed, stripped and casefolded."""
    return " ".join(text.split()).casefold()


def numeric_tokens(text: str) -> frozenset[str]:
    return frozenset(NUMERIC_TOKEN.findall(text))


def honest_submissions(evidence: RoundEvidence) -> tuple[str, ...]:
    """Final text replies of solved solver trials, then the last turn of every positive transcript control when
    it is a text reply; deduplicated, in that order."""
    solver = (
        final_reply(outcome.rollout) for outcome in evidence.solver if isinstance(outcome, Graded) and solved(outcome)
    )
    controls = (
        c.control.payload.turns[-1].content
        for c in evidence.controls
        if c.control.kind is ControlKind.POSITIVE
        and isinstance(c.control.payload, Transcript)
        and isinstance(c.control.payload.turns[-1], TextMessage)
    )
    return tuple(dict.fromkeys(text for text in (*solver, *controls) if text is not None))


def compare(submission: str | None, references: Sequence[str]) -> Comparison:
    """Whether ``submission`` is one of the honest ``references``.

    ``UNAVAILABLE`` without a submission or references. ``MATCH`` when ``normalised(submission)``
    equals a normalised reference, or when some reference has numeric tokens and the submission's
    numeric tokens are exactly that set. ``MISMATCH`` otherwise.
    """
    if submission is None or not references:
        return Comparison.UNAVAILABLE
    text, tokens = normalised(submission), numeric_tokens(submission)
    for reference in references:
        expected = numeric_tokens(reference)
        if text == normalised(reference) or (expected and tokens == expected):
            return Comparison.MATCH
    return Comparison.MISMATCH


def adversary_signals(
    role: AdversaryRole, outcome: Graded, facts: TaskFacts, references: Sequence[str]
) -> AdversarySignals:
    rollout = outcome.rollout
    commands = shell_commands(rollout)
    submission = None if facts.machine_answer else final_reply(rollout)
    return AdversarySignals(
        passed=solved(outcome),
        gave_up=gave_up(role, rollout),
        stop_reason=rollout.stop_reason,
        budget_exhausted=rollout.stop_reason in BUDGET_STOPS,
        output_tokens=rollout.loss_mask.count(1),
        turns=len(rollout.steps),
        shell_calls=len(commands),
        inputs_consumed=consumed_inputs(commands, facts.input_paths),
        inputs_written=written_inputs(commands, facts.protected_paths),
        submission=submission,
        comparison=Comparison.UNAVAILABLE if facts.machine_answer else compare(submission, references),
    )


def tier_of(
    role: AdversaryRole, s: AdversarySignals, facts: TaskFacts
) -> tuple[DefectTier, FindingKind | None, str, str]:
    """(tier, finding kind, rule id, reason) of one graded adversary outcome; the first row that fires decides.

    Rows 1, 2 and 4 say the answer did not come from the task's work, whatever the role, so they
    condemn the grader (``SHORTCUT_PASSED``). Row 5 keeps the role's kind: lenient parsing or a hedge
    for shortcut and leak, an instruction and grader that disagree for ambiguity. Row 7 is an honest
    solve the protocol forbids: recorded, never repaired.
    """
    if not s.passed:
        reason = "gave up" if s.gave_up else "budget exhausted" if s.budget_exhausted else "rejected"
        return DefectTier.NONE, None, "0", reason
    if s.inputs_written:
        written = json.dumps(list(s.inputs_written))
        return (
            DefectTier.REPAIR,
            FindingKind.SHORTCUT_PASSED,
            "1",
            f"passed after writing {written}: the grader trusts files the agent can change",
        )
    if facts.machine_answer and s.shell_calls == 0:
        return (
            DefectTier.REPAIR,
            FindingKind.SHORTCUT_PASSED,
            "2",
            "passed without a shell command: the grader credited an untouched workspace",
        )
    if role is AdversaryRole.AMBIGUITY and s.gave_up:
        return (
            DefectTier.NONE,
            None,
            "3",
            "solved under the intended reading and reported that no reading changes the answer",
        )
    if facts.input_paths and not s.inputs_consumed:
        return (
            DefectTier.REPAIR,
            FindingKind.SHORTCUT_PASSED,
            "4",
            "passed without reading any input file: the answer did not come from the task's inputs",
        )
    if s.comparison is Comparison.MISMATCH:
        accepted = (
            "the grader accepted a different answer under the stated reading"
            if role is AdversaryRole.AMBIGUITY
            else "the grader accepted text no honest run produced"
        )
        return DefectTier.REPAIR, PASS_KINDS[role], "5", f"{accepted}: {_clip(json.dumps(s.submission))}"
    if role is AdversaryRole.AMBIGUITY and s.comparison is Comparison.MATCH:
        return DefectTier.NONE, None, "6", "the stated reading produced the honest answer"
    if role is AdversaryRole.AMBIGUITY:
        compared = (
            "the submission cannot be compared for a machine-state answer"
            if facts.machine_answer
            else "no honest submission to compare it with"
        )
        return DefectTier.NOTED, PASS_KINDS[role], "7", f"passed under a stated reading; {compared}"
    read = f"read {json.dumps(list(s.inputs_consumed))} and " if s.inputs_consumed else ""
    return (
        DefectTier.NOTED,
        PASS_KINDS[role],
        "7",
        f"solved against orders: {read}submitted the answer; a role violation, not a task defect",
    )


def assess(
    role: AdversaryRole, index: int, outcome: Graded, facts: TaskFacts, references: Sequence[str]
) -> AdversaryAssessment:
    signals = adversary_signals(role, outcome, facts, references)
    tier, _, rule, reason = tier_of(role, signals, facts)
    return AdversaryAssessment(role, index, signals, tier, rule, reason)


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


def _adversary_findings(
    evidence: RoundEvidence,
) -> tuple[list[Finding], list[Finding], list[AdversaryAssessment]]:
    """(findings, notes, assessments) of every graded adversary trial, in ``AdversaryRole`` then index order."""
    references = honest_submissions(evidence)
    findings: list[Finding] = []
    notes: list[Finding] = []
    assessments: list[AdversaryAssessment] = []
    for role, outcomes in _by_role(evidence.adversaries):
        for index, outcome in enumerate(outcomes):
            if not isinstance(outcome, Graded):
                continue
            signals = adversary_signals(role, outcome, evidence.facts, references)
            tier, kind, rule, reason = tier_of(role, signals, evidence.facts)
            assessments.append(AdversaryAssessment(role, index, signals, tier, rule, reason))
            if kind is None:
                continue
            lines = [
                f"The {role} adversary {index} was graded as passing. Tier {tier} (row {rule}): {reason}",
                f"signals: stop={signals.stop_reason} tokens={signals.output_tokens} turns={signals.turns} "
                f"shell_calls={signals.shell_calls} gave_up={signals.gave_up}",
                f"  inputs_consumed={json.dumps(list(signals.inputs_consumed))} "
                f"inputs_written={json.dumps(list(signals.inputs_written))} comparison={signals.comparison}",
                *_trial_lines(f"adversary/{role}/{index}", outcome),
            ]
            if tier is DefectTier.NOTED:
                notes.append(Finding(kind, "\n".join(lines), FINDING_ROLES[kind]))
                continue
            new_controls: tuple[Control, ...] = ()
            if rule in CONTROL_RULES or role in CONTROL_ROLES:
                try:
                    new_controls = (transcript_control(role, index, evidence.task_digest, outcome.rollout),)
                except ValueError as error:
                    lines.append(f"  (not rendered as a control: {error})")
            findings.append(Finding(kind, "\n".join(lines), FINDING_ROLES[kind], new_controls))
    return findings, notes, assessments


def _by_role(adversaries: Mapping[AdversaryRole, Sequence[Outcome]]) -> list[tuple[AdversaryRole, Sequence[Outcome]]]:
    """The roles' outcomes in ``AdversaryRole`` order, so findings do not depend on the mapping's order."""
    return [(role, adversaries[role]) for role in AdversaryRole if role in adversaries]


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
        f"[{band.min_solve_rate}, {band.max_solve_rate}]. {stats.timed_out} trial(s) hit the agent deadline."
    )
    shown = [
        (index, outcome)
        for index, outcome in enumerate(outcomes)
        if kind is FindingKind.TOO_HARD or (isinstance(outcome, Graded) and solved(outcome))
    ]
    lines = [head, *(line for index, outcome in shown for line in _trial_lines(f"solver/{index}", outcome))]
    return [Finding(kind, "\n".join(lines), FINDING_ROLES[kind])]


def _role_stats(
    role: AdversaryRole, outcomes: Sequence[Outcome], required: int, assessments: Sequence[AdversaryAssessment]
) -> RoleStats:
    graded = [outcome for outcome in outcomes if isinstance(outcome, Graded)]
    rollouts = [outcome.rollout for outcome in outcomes if outcome.rollout is not None]
    tiers = [a.tier for a in assessments if a.role is role]
    return RoleStats(
        required=required,
        graded=len(graded),
        passes=sum(solved(outcome) for outcome in graded),
        gave_up=sum(gave_up(role, rollout) for rollout in rollouts),
        exhausted=sum(outcome.rollout.stop_reason in BUDGET_STOPS for outcome in graded),
        output_tokens=sum(rollout.loss_mask.count(1) for rollout in rollouts),
        tiers={tier: tiers.count(tier) for tier in DefectTier},
    )
