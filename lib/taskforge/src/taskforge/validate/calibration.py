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
over deterministic signals of its verifier submissions (``AdversarySignals``): which files a
passing submission supplied, whether it supplied any on a machine-graded task, whether the shell
session read an input, whether a passing reply matches an honest one, and the ordinal of the
accepted submission the adversary's verdict line refers to against the consumer's
``adversary_repair_submissions``. Rows that describe how a submission was produced fire whatever
the adversary says; the claim decides only among the rest, and no model judges. A ``REPAIR`` trial
is a decisive finding and ships the accepted candidate as a negative control the revised program
must ship (``candidate_control``), so the next round's control replay proves the fix; a candidate
that is an honest answer ships none, because no grader may refuse it. A ``NOTED`` trial (a shortcut
found only after more verifier calls than the threshold, a pass with no verdict, or a many-answer
grader accepting text no honest run produced) is a note: recorded beside the findings, never
blocking an accept.

The summary records the policy digest and band, so a decision is reproducible from
``calibration.json`` alone and a policy change shows as a different summary.
"""

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Protocol

from pydantic import TypeAdapter
from rolloutengine.contracts import (
    LENGTH_STOP_REASON,
    MAX_TURNS_STOP_REASON,
    TOTAL_TURN_TIMEOUT_STOP_REASON,
    RolloutData,
)
from rolloutengine.shell_tool import SHELL_TOOL_NAME
from taskcompendium.grading_result import Outcome as GradeStatus
from taskcompendium.models import TaskSpec
from verifyit.spec import Mode

from taskforge.atomic_file import write_atomic
from taskforge.builder.step import StepRole
from taskforge.spec.controls import (
    REJECTION_CEILING,
    Control,
    ControlCategory,
    ControlConcern,
    ControlKind,
    Expectation,
    Transcript,
)
from taskforge.spec.draft import MACHINE_ANSWER_TYPES
from taskforge.validate.adversary import CONTEXT_STOP_REASON, AdversaryRole
from taskforge.validate.controls import ControlOutcome, ControlVerdict, wire_message, workspace_turn
from taskforge.validate.evidence import Complete, Evidence, Incomplete, RewardStats
from taskforge.validate.outcome import Cause, Graded, Outcome, TrialKind, Ungraded
from taskforge.validate.submissions import (
    AdversaryTrial,
    Candidate,
    ClaimKind,
    Submission,
    final_answer,
    final_reply,
    passing,
    submitted_answer,
    trial_claim,
)

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
    """A machine-state answer, no accepted submission, or no honest submission to compare with."""


BUDGET_STOPS = frozenset(
    {LENGTH_STOP_REASON, MAX_TURNS_STOP_REASON, TOTAL_TURN_TIMEOUT_STOP_REASON, CONTEXT_STOP_REASON}
)
"""Stop reasons that mean the adversary ran out of output, turns, time or context before a verdict."""

UNIQUE_ANSWER_KINDS: frozenset[str] = frozenset(
    {Mode.EXACT, Mode.NUMERIC, Mode.MATH, Mode.MCQ, Mode.PREDICTED_ACTION, Mode.STRUCTURED_EXACT}
)
"""Grader kinds (``VerifierSpec.kind``, verifyit modes) with one expected answer, where a passing submission that
mismatches every honest answer is a grader defect. ``shell``, ``script``, ``external`` and every other kind accept
what their script accepts, so a mismatch there is at most a note (``tier_of`` row 9)."""


@dataclass(frozen=True)
class TaskFacts:
    """What the adversary signals read of the task.

    Attributes:
        machine_answer: The answer is the machine state (``MACHINE_ANSWER_TYPES``), not a text submission.
        unique_answer: The grader has one expected answer (``UNIQUE_ANSWER_KINDS``).
        input_paths: The task machine's files (``resources.all`` then ``resources.worker``) as absolute paths, in
            order.
        protected_paths: ``input_paths`` then the verifier's private files as installed under ``VERIFIER_ROOT``,
            each once, in order.
    """

    machine_answer: bool
    unique_answer: bool
    input_paths: tuple[str, ...]
    protected_paths: tuple[str, ...]


VERIFIER_ROOT = PurePosixPath("/tests")
"""Where RolloutEngine installs ``resources.verifier`` for the grader."""


def task_facts(task: TaskSpec) -> TaskFacts:
    resources = task.resources
    inputs = tuple(str(PurePosixPath("/", r.path)) for r in (*resources.all, *resources.worker))
    private = tuple(str(VERIFIER_ROOT / r.path) for r in resources.verifier)
    return TaskFacts(
        task.answer_type in MACHINE_ANSWER_TYPES,
        task.verifier.kind in UNIQUE_ANSWER_KINDS,
        inputs,
        tuple(dict.fromkeys((*inputs, *private))),
    )


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
class TierRuling:
    """The verdict of the first ``tier_of`` row that fires: the tier, the finding kind it reports, the row, why, and
    the submission ordinal it is about (the control's candidate)."""

    tier: DefectTier
    kind: FindingKind | None
    rule: str
    reason: str
    subject: int | None


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
    def adversary_k(self) -> int: ...

    @property
    def adversary_submissions(self) -> int: ...

    @property
    def adversary_repair_submissions(self) -> int: ...

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
    def adversaries(self) -> Mapping[AdversaryRole, tuple[AdversaryTrial, ...]]: ...

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
    adversary = _adversary_findings(evidence, policy)
    findings = [
        *_control_findings(evidence.controls),
        *adversary.findings,
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
            role: _role_stats(role, trials, policy.adversary_k, adversary.assessments)
            for role, trials in evidence.adversaries.items()
        },
        findings=tuple(findings),
        assessments=tuple(adversary.assessments),
        notes=tuple(adversary.notes),
    )


def write_summary(path: Path, summary: CalibrationSummary) -> None:
    """Write ``summary`` as ``calibration.json``."""
    write_atomic(path, SUMMARY.dump_json(summary, indent=1))


def load_summary(path: Path) -> CalibrationSummary:
    return SUMMARY.validate_json(path.read_bytes())


def solved(outcome: Graded) -> bool:
    """Whether ``outcome`` passes, as ``Evidence.reward_stats`` counts it."""
    return passing(outcome.grade)


def output_tokens(rollout: RolloutData) -> int:
    """The served response tokens of a rollout whose turns carry usage (every GLM rollout and every agent rollout)."""
    return sum(int(step.turn.metadata["usage"]["completion_tokens"]) for step in rollout.steps)


def commands(rollout: RolloutData) -> list[str]:
    """The arguments of every tool call the model made, in order."""
    return [
        call["function"]["arguments"] for step in rollout.steps for call in step.turn.message.get("tool_calls") or ()
    ]


def candidate_control(role: AdversaryRole, index: int, ordinal: int, task_digest: str, candidate: Candidate) -> Control:
    """An accepted adversary candidate as a negative control that demands no credit.

    One ``validate.controls.workspace_turn`` per file (the shell writes the agent would make), then the
    candidate's final turn; a candidate without files is the final turn alone.
    """
    writes = tuple(workspace_turn(serial, file) for serial, file in enumerate(candidate.files))
    return Control(
        id=f"adv-{role}-{index}-{ordinal}-{task_digest[:8]}",
        kind=ControlKind.NEGATIVE,
        category=ControlCategory.REWARD_HACK,
        concern=ControlConcern.SHORTCUT,
        author=f"adversary/{role}/{index}#{ordinal}",
        payload=Transcript((*writes, candidate.turn)),
        expect=Expectation(status=GradeStatus.GRADED, reward_max=REJECTION_CEILING),
    )


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
        f"  commands: {_clip(json.dumps(commands(rollout), ensure_ascii=False))}",
        f"  final reply: {_clip(json.dumps(final_reply(rollout), ensure_ascii=False))}",
    ]


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


def normalised(text: str) -> str:
    """``text`` with whitespace collapsed, stripped and casefolded."""
    return " ".join(text.split()).casefold()


def numeric_tokens(text: str) -> frozenset[str]:
    return frozenset(NUMERIC_TOKEN.findall(text))


def honest_submissions(evidence: RoundEvidence) -> tuple[str, ...]:
    """The answers solved solver trials submitted, then the answer the last turn of every positive transcript
    control submits (``submissions.submitted_answer``: a text reply or an answer call's answer); deduplicated, in
    that order."""
    solver = (
        final_answer(outcome.rollout) for outcome in evidence.solver if isinstance(outcome, Graded) and solved(outcome)
    )
    controls = (
        submitted_answer(wire_message(c.control.payload.turns[-1]))
        for c in evidence.controls
        if c.control.kind is ControlKind.POSITIVE and isinstance(c.control.payload, Transcript)
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
    trial: AdversaryTrial, facts: TaskFacts, references: Sequence[str], policy: SummaryPolicy
) -> AdversarySignals:
    """The signals of one graded adversary trial against the round's honest ``references``."""
    outcome = trial.outcome
    rollout = outcome.rollout
    assert isinstance(outcome, Graded) and rollout is not None
    commands = shell_commands(rollout)
    passed = [s for s in trial.submissions if s.passed]
    exploit = passed[-1] if passed else None
    protected = set(facts.protected_paths)
    claim = trial_claim(outcome)
    return AdversarySignals(
        submissions=len(trial.submissions),
        passes=len(passed),
        budget_spent=len(trial.submissions) >= policy.adversary_submissions,
        first_pass=passed[0].ordinal if passed else None,
        exploit=None if exploit is None else exploit.ordinal,
        claim=claim.kind,
        why=claim.why,
        stop_reason=rollout.stop_reason,
        exhausted=rollout.stop_reason in BUDGET_STOPS,
        output_tokens=output_tokens(rollout),
        turns=len(rollout.steps),
        shell_calls=len(commands),
        inputs_consumed=consumed_inputs(commands, facts.input_paths),
        protected_supplied=tuple(s.ordinal for s in passed if protected & set(s.candidate.paths)),
        fileless_passes=tuple(s.ordinal for s in passed if not s.candidate.files),
        mismatches=(
            ()
            if facts.machine_answer
            else tuple(s.ordinal for s in passed if compare(s.candidate.reply, references) is Comparison.MISMATCH)
        ),
        comparison=(
            Comparison.UNAVAILABLE
            if facts.machine_answer or exploit is None
            else compare(exploit.candidate.reply, references)
        ),
    )


def _repair(rule: str, reason: str, subject: int | None) -> TierRuling:
    return TierRuling(DefectTier.REPAIR, FindingKind.SHORTCUT_PASSED, rule, reason, subject)


def _noted(rule: str, reason: str, subject: int | None) -> TierRuling:
    return TierRuling(DefectTier.NOTED, FindingKind.SHORTCUT_PASSED, rule, reason, subject)


def _rejected(s: AdversarySignals) -> str:
    if s.submissions == 0:
        reason = "no verifier submission"
    elif s.budget_spent:
        reason = f"{s.submissions} submissions, budget spent, none accepted"
    else:
        reason = f"all {s.submissions} submissions rejected"
    if s.claim is ClaimKind.SHORTCUT:
        reason += "; claimed a shortcut without an accepted submission"
    return reason


def tier_of(s: AdversarySignals, facts: TaskFacts, policy: SummaryPolicy) -> TierRuling:
    """The ruling on one graded adversary trial; the first row that fires decides.

    Rows 1, 2, 3 and 5 describe how a passing submission was produced (a supplied protected file, no files on a
    machine-graded task, no input read, a non-honest value on a single-answer grader), so they are grader defects
    whatever the adversary says; rows 1, 2 and 5 range over every passing submission and name the lowest. Row 4
    keeps an honest answer called a shortcut from becoming a defect. Row 6 is the primary signal: a claimed
    shortcut accepted within ``adversary_repair_submissions`` verifier calls. Honest probes the adversary does not
    claim are no defect (row 10).
    """
    t = policy.adversary_repair_submissions
    budget = policy.adversary_submissions
    if s.passes == 0:
        return TierRuling(DefectTier.NONE, None, "0", _rejected(s), None)
    if s.protected_supplied:
        first = s.protected_supplied[0]
        return _repair(
            "1",
            f"submission {first} passed after supplying a task input or grader file: the grader trusts files the "
            "agent can change",
            first,
        )
    if facts.machine_answer and s.fileless_passes:
        first = s.fileless_passes[0]
        return _repair(
            "2", f"submission {first} passed with no workspace files: the grader credited an untouched workspace", first
        )
    if facts.input_paths and not s.inputs_consumed:
        return _repair(
            "3",
            f"submission {s.exploit} passed without any input file being read: the answer did not come from the "
            "task's inputs",
            s.exploit,
        )
    if s.claim is ClaimKind.SHORTCUT and s.comparison is Comparison.MATCH:
        return TierRuling(
            DefectTier.NONE,
            None,
            "4",
            "the claimed shortcut is the honest answer: a failed audit, not a defect",
            s.exploit,
        )
    if facts.unique_answer and s.mismatches:
        first = s.mismatches[0]
        return _repair(
            "5", f"submission {first} passed with a value that is not the honest answer on a single-answer grader", first
        )
    if s.claim is ClaimKind.SHORTCUT and s.exploit is not None and s.exploit <= t:
        return _repair(
            "6",
            f"an accepted shortcut in {s.exploit} of {budget} verifier calls (repair threshold {t}); the adversary "
            f"reports: {s.why}",
            s.exploit,
        )
    if s.claim is ClaimKind.SHORTCUT:
        return _noted(
            "7",
            f"an accepted shortcut after {s.exploit} verifier calls, above the repair threshold {t}; the adversary "
            f"reports: {s.why}",
            s.exploit,
        )
    if s.claim is ClaimKind.NONE:
        return _noted(
            "8",
            f"{s.passes} of {s.submissions} submissions accepted and no verdict given (stop {s.stop_reason})",
            s.exploit,
        )
    if s.mismatches:
        first = s.mismatches[0]
        return _noted(
            "9",
            f"submission {first} passed with text no honest run produced, on a many-answer grader; the adversary "
            "reports no shortcut",
            first,
        )
    return TierRuling(DefectTier.NONE, None, "10", "its accepted submissions were probes; it reports no shortcut", None)


CONTROL_RULES = frozenset({"1", "2", "3", "5", "6"})
"""Rows whose repairs ship the subject candidate as a negative control; row 3 not when it is an honest answer."""


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


def _submission_line(submission: Submission) -> str:
    grade = submission.grade
    paths = json.dumps(list(submission.candidate.paths))
    head = f"submission {submission.ordinal}: {grade.status}, reward {grade.reward}, passed {submission.passed}"
    return f"{head}, files {paths}; reply: {_clip(json.dumps(submission.candidate.reply, ensure_ascii=False))}"


def _signal_lines(s: AdversarySignals) -> list[str]:
    return [
        f"claim: {s.claim} {s.why}".rstrip(),
        f"signals: submissions={s.submissions} passes={s.passes} first_pass={s.first_pass} exploit={s.exploit} "
        f"budget_spent={s.budget_spent} stop={s.stop_reason}",
        f"  tokens={s.output_tokens} turns={s.turns} shell_calls={s.shell_calls} "
        f"inputs_consumed={json.dumps(list(s.inputs_consumed))} "
        f"protected_supplied={json.dumps(list(s.protected_supplied))} "
        f"fileless_passes={json.dumps(list(s.fileless_passes))}",
        f"  mismatches={json.dumps(list(s.mismatches))} comparison={s.comparison}",
    ]


@dataclass(frozen=True)
class _AdversaryReview:
    """What the graded adversary trials of a round add to its summary, in ``AdversaryRole`` then index order."""

    findings: list[Finding]
    notes: list[Finding]
    assessments: list[AdversaryAssessment]


def _adversary_findings(evidence: RoundEvidence, policy: SummaryPolicy) -> _AdversaryReview:
    """The findings, notes and assessments of every graded adversary trial."""
    references = honest_submissions(evidence)
    findings: list[Finding] = []
    notes: list[Finding] = []
    assessments: list[AdversaryAssessment] = []
    for role, trials in _by_role(evidence.adversaries):
        for index, trial in enumerate(trials):
            if not isinstance(trial.outcome, Graded):
                continue
            signals = adversary_signals(trial, evidence.facts, references, policy)
            ruling = tier_of(signals, evidence.facts, policy)
            assessments.append(
                AdversaryAssessment(role, index, signals, ruling.tier, ruling.rule, ruling.reason, ruling.subject)
            )
            if ruling.kind is None:
                continue
            lines = [
                f"The {role} adversary {index + 1} of {len(trials)} found an accepted submission. "
                f"Tier {ruling.tier} (row {ruling.rule}): {ruling.reason}",
                *_signal_lines(signals),
                *(_submission_line(s) for s in trial.submissions),
                *_trial_lines(f"adversary/{role}/{index}", trial.outcome),
            ]
            if ruling.tier is DefectTier.NOTED:
                notes.append(Finding(ruling.kind, "\n".join(lines), FINDING_ROLES[ruling.kind]))
                continue
            new_controls: tuple[Control, ...] = ()
            subject = next((s for s in trial.submissions if s.ordinal == ruling.subject), None)
            if ruling.rule == "3" and signals.comparison is Comparison.MATCH:
                lines.append("  (not rendered as a control: it submitted an honest answer, which no grader may refuse)")
            elif ruling.rule in CONTROL_RULES and subject is not None:
                control = candidate_control(role, index, subject.ordinal, evidence.task_digest, subject.candidate)
                new_controls = (control,)
            findings.append(Finding(ruling.kind, "\n".join(lines), FINDING_ROLES[ruling.kind], new_controls))
    return _AdversaryReview(findings, notes, assessments)


def _by_role(
    adversaries: Mapping[AdversaryRole, Sequence[AdversaryTrial]],
) -> list[tuple[AdversaryRole, Sequence[AdversaryTrial]]]:
    """The roles' trials in ``AdversaryRole`` order, so findings do not depend on the mapping's order."""
    return [(role, adversaries[role]) for role in AdversaryRole if role in adversaries]


def _named_outcomes(evidence: RoundEvidence) -> list[tuple[str, Outcome]]:
    return [
        *((f"control/{c.control.id}", c.outcome) for c in evidence.controls),
        *((f"solver/{index}", outcome) for index, outcome in enumerate(evidence.solver)),
        *(
            (f"adversary/{role}/{index}", trial.outcome)
            for role, trials in _by_role(evidence.adversaries)
            for index, trial in enumerate(trials)
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
        f"[{band.min_solve_rate}, {band.max_solve_rate}]. {stats.timed_out} trial(s) hit the total-turn deadline."
    )
    shown = [
        (index, outcome)
        for index, outcome in enumerate(outcomes)
        if kind is FindingKind.TOO_HARD or (isinstance(outcome, Graded) and solved(outcome))
    ]
    lines = [head, *(line for index, outcome in shown for line in _trial_lines(f"solver/{index}", outcome))]
    return [Finding(kind, "\n".join(lines), FINDING_ROLES[kind])]


def _role_stats(
    role: AdversaryRole, trials: Sequence[AdversaryTrial], required: int, assessments: Sequence[AdversaryAssessment]
) -> RoleStats:
    graded = [trial for trial in trials if isinstance(trial.outcome, Graded)]
    rollouts = [trial.outcome.rollout for trial in trials if trial.outcome.rollout is not None]
    mine = [a for a in assessments if a.role is role]
    claims = [a.signals.claim for a in mine]
    tiers = [a.tier for a in mine]
    return RoleStats(
        required=required,
        graded=len(graded),
        passes=sum(a.signals.passes > 0 for a in mine),
        submissions=sum(a.signals.submissions for a in mine),
        budget_spent=sum(a.signals.budget_spent for a in mine),
        claims={kind: claims.count(kind) for kind in ClaimKind},
        failed_audits=sum(a.rule == "4" for a in mine),
        exhausted=sum(a.signals.exhausted for a in mine),
        output_tokens=sum(output_tokens(rollout) for rollout in rollouts),
        tiers={tier: tiers.count(tier) for tier in DefectTier},
    )
