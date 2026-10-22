# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One validation round of a draft: lowering, control writing, the k solver trials, control replay and the
adversary trials as units of one graph.

Lowering (``preflight``) and control writing (``write_controls``) start together. Each solver trial starts
as soon as the lowering is done and does not wait for the controls. Control replay and each adversary trial
(``adversary.run_adversary_trial``) start once both the lowered spec and the controls exist, and wait for no
solver trial. The result waits for every unit. A unit that raises cancels the others and the round raises its
error; the files the finished units wrote stay for the relaunch.

Each unit is idempotent on its own file, so a relaunch redoes only the units whose files are missing:

    <round_dir>/validation.json                                the ValidationResult
    <round_dir>/validation/<draft>/lowered.json                the lowering
    <round_dir>/validation/<draft>/controls.json               the controls
    <round_dir>/validation/<draft>/<task>/solver/<i>/attempt-<n>.json
    <round_dir>/validation/<draft>/<task>/controls-<controls>/control/<id>/attempt-<n>.json
    <round_dir>/validation/<draft>/<task>/adversary/<i>/attempt-<n>.json
    <round_dir>/validation/<draft>/<task>/adversary/<i>/submissions/<n>/control/s<m>/attempt-<k>.json

``<draft>`` is ``content_hash.digest`` of the draft's task, ``<task>`` the ``trials.task_digest`` of the
lowered spec as its trials run, and ``<controls>`` ``content_hash.digest`` of the control set. A file covers
what its path names, so the lowering or controls of another task, the trials of another lowered spec and the
replay of another control set are never read as done. ``lowered.json`` also records the host and the
factory backends it lowered for, and a lowering for another host is done again. An adversary attempt records its
brief, which holds the mode, the budget and, white-box, the controls: an attempt under another brief is kept, and
the trial runs again after it with its full retries. An adversary attempt is numbered after any submissions an
attempt cancelled in flight graded, so its submissions directory is its own.

A trial is done when its last attempt file holds a grade or a cause that is not retryable, or when it holds
``max_retries + 1`` attempts: where ``run_trial`` stops. Otherwise the trial resumes at the next attempt
number with the retries it has left. An attempt cancelled before its outcome writes no file
(``trials.write_evidence`` writes whole files only), so a cancelled trial is run again rather than counted as
ungraded.

The result is read back from the unit files, each outcome through ``trials.attempt_outcome``, so a resumed
round returns what an uninterrupted one would. ``validation.json`` holds every decision input and no rollout,
which stays in the attempt files it names: the lowering's status, refusals and backend fallback; per solver
trial its status, reward and cause, with the solver ``Evidence``'s reward totals and ungraded causes; per
control its verdict and cause; and per adversary trial its status, cause, claim and every submission with its
grade status, reward and whether it passed. Which passing submissions are shortcuts is review's judgement, not
the round's. ``load_validation`` reads it without the evidence files.

A spec RolloutEngine rejects is ``REJECTED`` and no trial runs. A spec a factory refuses is ``REFUSED`` and its
trials run, each recording the one ``MACHINE_UNSUPPORTED`` attempt ``run_trial`` records for it.
"""

import asyncio
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pydantic import TypeAdapter
from rolloutengine.models import TaskRolloutSpec
from taskcompendium.grading_result import Outcome as GradeStatus

from taskforge.content_hash import digest, pretty_json
from taskforge.ledger.records import Ledger
from taskforge.llm.client import ModelClient
from taskforge.llm.policy import LLMPolicy
from taskforge.llm.recording import CallLedger
from taskforge.sandbox.factories import MachineHost, Refusal, task_refusals
from taskforge.spec.controls import Control, ControlKind
from taskforge.stage10_proposal.model import TaskProposal
from taskforge.stage30_author.loader import GRADER_STEP
from taskforge.stage40_build_with_program.outputs import Grader
from taskforge.stage40_build_with_program.run import TaskDraft, step_output
from taskforge.stage50_validate.adversary import (
    SUBMISSIONS_DIR,
    AdversaryMode,
    AdversaryPolicy,
    Claim,
    adversary_brief,
    adversary_record,
    run_adversary_trial,
)
from taskforge.stage50_validate.control_drafts import CONTROLS_FILE, CONTROLS_STEP, load_controls, write_controls
from taskforge.stage50_validate.controls import ControlVerdict, check_replayable, control_outcome, replay_control
from taskforge.stage50_validate.evidence import Evidence, Incomplete, RewardStats
from taskforge.stage50_validate.outcome import Cause, Graded, Outcome, TrialKind
from taskforge.stage50_validate.preflight import BackendFallback, preflight
from taskforge.stage50_validate.trials import (
    ATTEMPT_PREFIX,
    Deadlines,
    EngineSettings,
    RetryBackoff,
    TrialModel,
    TrialPlan,
    attempt_outcome,
    run_trial,
    task_digest,
    write_evidence,
)

VALIDATION_FILE = "validation.json"
VALIDATION_DIR = "validation"
LOWERED_FILE = "lowered.json"
REPLAY_PREFIX = "controls-"
_ATTEMPT_FILE = re.compile(rf"{re.escape(ATTEMPT_PREFIX)}(\d+)\.json")


@dataclass(frozen=True)
class ValidationContext:
    """What one validation round of a workpiece needs besides its draft and its models.

    Attributes:
        workpiece_id: The workpiece the draft belongs to.
        round: The author, build and validation round.
        round_dir: The round's directory; ``validation.json`` and ``validation/`` are written under it.
        proposal: The proposal the round built, which the controls follow.
        store_root: The build's step store, holding the draft's grader step output.
        host: Where the machine factories run.
        engine: The engine settings every trial and the preflight probe run under.
        deadlines: The deadlines of every trial and of the probe.
        k: The number of solver trials.
        max_retries: The retries of a trial that fails for a retryable cause.
        retry_backoff: The wait between a trial's attempts.
        sampling: The sampling policy of the model writing the controls.
        adversary: The adversary trials' policy.
        ledger: Where the trials and the control writing are recorded.
    """

    workpiece_id: str
    round: int
    round_dir: Path
    proposal: TaskProposal
    store_root: str
    host: MachineHost
    engine: EngineSettings
    deadlines: Deadlines
    k: int
    max_retries: int
    retry_backoff: RetryBackoff
    sampling: LLMPolicy
    adversary: AdversaryPolicy
    ledger: Ledger


class LoweringStatus(StrEnum):
    LOWERED = "lowered"
    REFUSED = "refused"
    """A factory refuses a machine of the lowered spec; the trials record the refusal."""
    REJECTED = "rejected"
    """RolloutEngine rejects the lowered spec, so no trial runs."""


@dataclass(frozen=True)
class Lowering:
    """The lowering of the draft's task: ``detail`` is RolloutEngine's rejection, empty otherwise, and
    ``task_digest`` the ``trials.task_digest`` of the spec as this round's trials run it, ``None`` for a
    rejected spec."""

    status: LoweringStatus
    detail: str
    refusals: tuple[Refusal, ...]
    fallback: BackendFallback | None
    task_digest: str | None


class TrialStatus(StrEnum):
    GRADED = "graded"
    UNGRADED = "ungraded"


@dataclass(frozen=True)
class TrialResult:
    """A solver trial's final attempt: ``reward`` is set when it is graded, ``cause`` when it is not.
    ``attempt_file`` is relative to the round directory."""

    trial: str
    status: TrialStatus
    reward: float | None
    cause: Cause | None
    attempt_file: str


@dataclass(frozen=True)
class SolverEvidence:
    """The solver trials and the totals of their ``Evidence``: ``ungraded`` counts the trials without a grade
    by cause, and is empty when every trial is graded."""

    trials: tuple[TrialResult, ...]
    stats: RewardStats
    ungraded: dict[Cause, int]


@dataclass(frozen=True)
class ControlResult:
    """A control's replay: ``cause`` is set when the replay produced no grade. ``attempt_file`` is relative to
    the round directory."""

    id: str
    kind: ControlKind
    verdict: ControlVerdict
    cause: Cause | None
    attempt_file: str


@dataclass(frozen=True)
class SubmissionResult:
    """An adversary submission as the verifier graded it. ``attempt_file``, relative to the round directory,
    holds the full grade."""

    ordinal: int
    status: GradeStatus
    reward: float | None
    passed: bool
    attempt_file: str


@dataclass(frozen=True)
class AdversaryResult:
    """An adversary trial's final attempt: ``cause`` is set when it is ungraded, and ``claim`` is the adversary's
    own verdict. ``attempt_file`` is relative to the round directory."""

    trial: str
    mode: AdversaryMode
    status: TrialStatus
    cause: Cause | None
    claim: Claim
    submissions: tuple[SubmissionResult, ...]
    attempt_file: str


@dataclass(frozen=True)
class ValidationResult:
    """A validation round, written as ``validation.json``: ``solver`` is ``None`` and ``controls`` and
    ``adversary`` empty when the lowering was rejected."""

    workpiece_id: str
    round: int
    lowering: Lowering
    solver: SolverEvidence | None
    controls: tuple[ControlResult, ...]
    adversary: tuple[AdversaryResult, ...]


@dataclass(frozen=True)
class _Lowered:
    """``lowered.json``: the host and backends the task was lowered for, the spec it lowered to (``None`` when
    RolloutEngine rejected it, with the reason in ``detail``), and the preflight's refusals and fallback.

    It holds no task digest: the digest also covers the trials' deadlines and engine settings, which a relaunch
    may change, so each round derives it from ``spec``.
    """

    host: MachineHost
    backends: tuple[str, ...]
    spec: TaskRolloutSpec | None
    detail: str
    refusals: tuple[Refusal, ...]
    fallback: BackendFallback | None


async def validate_round(
    draft: TaskDraft,
    context: ValidationContext,
    solver_model: TrialModel,
    controls_model: ModelClient,
    adversary_model: ModelClient,
) -> ValidationResult:
    """Validate ``draft``, running only the units without a file, and write ``validation.json``.

    Args:
        draft: The round's draft.
        context: The round's settings.
        solver_model: The model of the solver trials.
        controls_model: The model that writes the controls.
        adversary_model: The model of the adversary trials.

    Raises:
        BuildInfrastructureFailure: the preflight probe's machine did not start twice.
        BuildFailure: the controls never passed their checks (``write_controls``).
        ValueError: ``check_replayable`` refuses the written controls.
    """
    directory = context.round_dir / VALIDATION_DIR / digest(draft.task)
    try:
        async with asyncio.TaskGroup() as group:
            lowering = group.create_task(_lowering_unit(draft, context, directory))
            controls = group.create_task(_controls_unit(draft, context, controls_model, directory))
            group.create_task(_solver_units(lowering, context, solver_model, directory))
            group.create_task(_replay_units(lowering, controls, context, directory))
            group.create_task(_adversary_units(lowering, controls, context, adversary_model, directory))
    except BaseExceptionGroup as failure:
        if len(failure.exceptions) == 1:
            raise failure.exceptions[0] from None
        raise
    result = _result(context, lowering.result(), controls.result(), directory)
    write_evidence(context.round_dir / VALIDATION_FILE, pretty_json(result).encode())
    return result


def load_validation(round_dir: Path) -> ValidationResult:
    """Read the ``validation.json`` ``validate_round`` wrote to ``round_dir``."""
    return _RESULT.validate_json((round_dir / VALIDATION_FILE).read_bytes())


async def _inputs_ready(*inputs: asyncio.Task) -> bool:
    """Wait for ``inputs``; whether each returned. A failed input is the group's error, not its dependents'."""
    await asyncio.wait(inputs)
    return all(not task.cancelled() and task.exception() is None for task in inputs)


async def _lowering_unit(draft: TaskDraft, context: ValidationContext, directory: Path) -> _Lowered:
    path = directory / LOWERED_FILE
    backends = tuple(sorted(context.engine.factories))
    if path.exists():
        done = _LOWERED.validate_json(path.read_bytes())
        if done.host == context.host and done.backends == backends:
            return done
    write_evidence(path, pretty_json(await _lower(draft, context, backends)).encode())
    return _LOWERED.validate_json(path.read_bytes())


async def _lower(draft: TaskDraft, context: ValidationContext, backends: tuple[str, ...]) -> _Lowered:
    engine = context.engine
    try:
        lowered = await preflight(draft.task, context.host, engine.factories, engine, context.deadlines)
    except (ValueError, NotImplementedError) as error:
        return _Lowered(context.host, backends, None, str(error), (), None)
    refusals = tuple(task_refusals(lowered.spec, engine.factories))
    return _Lowered(context.host, backends, lowered.spec, "", refusals, lowered.fallback)


def _trial_digest(context: ValidationContext, spec: TaskRolloutSpec) -> str:
    """The ``task_digest`` of ``spec`` as the round's trials run it."""
    engine = context.engine
    return task_digest(engine.apply(context.deadlines.apply(spec)), engine.command_timeout)


def _trials_dir(directory: Path, context: ValidationContext, spec: TaskRolloutSpec) -> Path:
    return directory / _trial_digest(context, spec)


async def _controls_unit(
    draft: TaskDraft, context: ValidationContext, model: ModelClient, directory: Path
) -> tuple[Control, ...]:
    if not (directory / CONTROLS_FILE).exists():
        grader = step_output(draft, context.store_root, GRADER_STEP, Grader)
        record = CallLedger(context.ledger, context.workpiece_id, context.round, CONTROLS_STEP)
        await write_controls(draft.task, grader, context.proposal, model, context.sampling, record, directory)
    return load_controls(directory)


async def _solver_units(
    lowering: asyncio.Task[_Lowered], context: ValidationContext, model: TrialModel, directory: Path
) -> None:
    if not await _inputs_ready(lowering):
        return
    lowered = lowering.result()
    if lowered.spec is None:
        return
    spec, evidence_dir = lowered.spec, _trials_dir(directory, context, lowered.spec)
    async with asyncio.TaskGroup() as group:
        for index in range(context.k):
            plan = _resumed_plan(context, TrialKind.SOLVER, evidence_dir, str(index), _any_attempt, 0)
            if plan is not None:
                group.create_task(run_trial(spec, plan, context.engine, model, str(index)))


async def _replay_units(
    lowering: asyncio.Task[_Lowered],
    controls: asyncio.Task[tuple[Control, ...]],
    context: ValidationContext,
    directory: Path,
) -> None:
    if not await _inputs_ready(lowering, controls):
        return
    lowered, control_set = lowering.result(), controls.result()
    if lowered.spec is None:
        return
    spec, evidence_dir = lowered.spec, _replay_dir(_trials_dir(directory, context, lowered.spec), control_set)
    check_replayable(spec.task, control_set, context.engine.max_turns)
    async with asyncio.TaskGroup() as group:
        for control in control_set:
            plan = _resumed_plan(context, TrialKind.CONTROL, evidence_dir, control.id, _any_attempt, 0)
            if plan is not None:
                group.create_task(replay_control(spec, control, plan, context.engine))


async def _adversary_units(
    lowering: asyncio.Task[_Lowered],
    controls: asyncio.Task[tuple[Control, ...]],
    context: ValidationContext,
    model: ModelClient,
    directory: Path,
) -> None:
    if not await _inputs_ready(lowering, controls):
        return
    lowered, control_set = lowering.result(), controls.result()
    if lowered.spec is None:
        return
    spec, evidence_dir = lowered.spec, _trials_dir(directory, context, lowered.spec)
    policy = context.adversary
    brief = adversary_brief(spec, control_set, policy)

    def under_brief(payload: bytes) -> bool:
        return adversary_record(payload).brief == brief

    async with asyncio.TaskGroup() as group:
        for index in range(policy.k):
            trial = str(index)
            graded_submissions = _numbered_dirs(evidence_dir / str(TrialKind.ADVERSARY) / trial / SUBMISSIONS_DIR)
            floor = graded_submissions[-1] + 1 if graded_submissions else 0
            plan = _resumed_plan(context, TrialKind.ADVERSARY, evidence_dir, trial, under_brief, floor)
            if plan is not None:
                run = run_adversary_trial(spec, brief, policy, plan, context.engine, model, trial, context.max_retries)
                group.create_task(run)


def _any_attempt(payload: bytes) -> bool:
    return True


def _resumed_plan(
    context: ValidationContext,
    kind: TrialKind,
    evidence_dir: Path,
    trial: str,
    covers: Callable[[bytes], bool],
    floor: int,
) -> TrialPlan | None:
    """The plan that runs the rest of ``trial`` after the attempts on disk, or ``None`` when it is done.

    Only the attempts whose file ``covers`` accepts belong to this run of the trial and spend its retries; the
    trial is done when the last attempt is one of them and ends it. The next attempt is numbered after every
    attempt on disk and at least ``floor``.
    """
    attempts = _attempt_numbers(evidence_dir / str(kind) / trial)
    payloads = {attempt: _attempt_path(evidence_dir, kind, trial, attempt).read_bytes() for attempt in attempts}
    covered = [attempt for attempt in attempts if covers(payloads[attempt])]
    spent = len(covered) if covered and covered[-1] == attempts[-1] else 0
    if spent:
        last = attempt_outcome(payloads[attempts[-1]])
        if isinstance(last, Graded) or not last.retryable or spent > context.max_retries:
            return None
    return TrialPlan(
        workpiece_id=context.workpiece_id,
        round=context.round,
        kind=kind,
        k=1,
        deadlines=context.deadlines,
        max_retries=context.max_retries - spent,
        retry_backoff=context.retry_backoff.schedule(),
        evidence_dir=evidence_dir,
        ledger=context.ledger,
        first_attempt=max(floor, attempts[-1] + 1 if attempts else 0),
    )


def _numbered_dirs(directory: Path) -> list[int]:
    if not directory.is_dir():
        return []
    return sorted(int(path.name) for path in directory.iterdir() if path.is_dir() and path.name.isdigit())


def _attempt_numbers(trial_dir: Path) -> list[int]:
    if not trial_dir.is_dir():
        return []
    matches = (_ATTEMPT_FILE.fullmatch(path.name) for path in trial_dir.iterdir())
    return sorted(int(match.group(1)) for match in matches if match is not None)


def _attempt_path(evidence_dir: Path, kind: TrialKind, trial: str, attempt: int) -> Path:
    return evidence_dir / str(kind) / trial / f"{ATTEMPT_PREFIX}{attempt}.json"


def _final_attempt(evidence_dir: Path, kind: TrialKind, trial: str) -> tuple[Path, Outcome]:
    """The last attempt file of a finished trial and the outcome it holds."""
    attempts = _attempt_numbers(evidence_dir / str(kind) / trial)
    assert attempts, f"the {kind} trial {trial} has no attempt under {evidence_dir}"
    path = _attempt_path(evidence_dir, kind, trial, attempts[-1])
    return path, attempt_outcome(path.read_bytes())


def _replay_dir(trials_dir: Path, controls: tuple[Control, ...]) -> Path:
    return trials_dir / f"{REPLAY_PREFIX}{digest(controls)}"


def _result(
    context: ValidationContext, lowered: _Lowered, controls: tuple[Control, ...], directory: Path
) -> ValidationResult:
    """The round's result, read back from its unit files."""
    if lowered.spec is None:
        rejected = Lowering(LoweringStatus.REJECTED, lowered.detail, (), None, None)
        return ValidationResult(context.workpiece_id, context.round, rejected, None, (), ())
    trial_digest = _trial_digest(context, lowered.spec)
    status = LoweringStatus.REFUSED if lowered.refusals else LoweringStatus.LOWERED
    lowering = Lowering(status, "", lowered.refusals, lowered.fallback, trial_digest)
    trials_dir = _trials_dir(directory, context, lowered.spec)
    return ValidationResult(
        context.workpiece_id,
        context.round,
        lowering,
        _solver_evidence(context, trials_dir),
        _control_results(context, _replay_dir(trials_dir, controls), controls),
        tuple(_adversary_result(context, trials_dir, str(index)) for index in range(context.adversary.k)),
    )


def _solver_evidence(context: ValidationContext, evidence_dir: Path) -> SolverEvidence:
    finals = [_final_attempt(evidence_dir, TrialKind.SOLVER, str(index)) for index in range(context.k)]
    evidence = Evidence({TrialKind.SOLVER: tuple(outcome for _, outcome in finals)})
    status = evidence.status
    return SolverEvidence(
        trials=tuple(
            _trial_result(str(index), outcome, _relative(context, path)) for index, (path, outcome) in enumerate(finals)
        ),
        stats=evidence.reward_stats(TrialKind.SOLVER),
        ungraded=dict(status.causes) if isinstance(status, Incomplete) else {},
    )


def _trial_result(trial: str, outcome: Outcome, attempt_file: str) -> TrialResult:
    if isinstance(outcome, Graded):
        return TrialResult(trial, TrialStatus.GRADED, outcome.reward, None, attempt_file)
    return TrialResult(trial, TrialStatus.UNGRADED, None, outcome.cause, attempt_file)


def _control_results(
    context: ValidationContext, evidence_dir: Path, controls: tuple[Control, ...]
) -> tuple[ControlResult, ...]:
    results = []
    for control in controls:
        path, outcome = _final_attempt(evidence_dir, TrialKind.CONTROL, control.id)
        verdict = control_outcome(control, outcome).verdict
        cause = None if isinstance(outcome, Graded) else outcome.cause
        results.append(ControlResult(control.id, control.kind, verdict, cause, _relative(context, path)))
    return tuple(results)


def _adversary_result(context: ValidationContext, evidence_dir: Path, trial: str) -> AdversaryResult:
    attempt = _attempt_numbers(evidence_dir / str(TrialKind.ADVERSARY) / trial)[-1]
    path = _attempt_path(evidence_dir, TrialKind.ADVERSARY, trial, attempt)
    payload = path.read_bytes()
    record = adversary_record(payload)
    graded = _trial_result(trial, attempt_outcome(payload), "")
    submissions_dir = evidence_dir / str(TrialKind.ADVERSARY) / trial / SUBMISSIONS_DIR / str(attempt)
    submissions = []
    for submission in record.submissions:
        grading = _attempt_numbers(submissions_dir / str(TrialKind.CONTROL) / submission.trial)
        grading_file = _attempt_path(submissions_dir, TrialKind.CONTROL, submission.trial, grading[-1])
        grade = submission.grade
        submissions.append(
            SubmissionResult(
                submission.ordinal, grade.status, grade.reward, submission.passed, _relative(context, grading_file)
            )
        )
    return AdversaryResult(
        trial, record.mode, graded.status, graded.cause, record.claim, tuple(submissions), _relative(context, path)
    )


def _relative(context: ValidationContext, path: Path) -> str:
    return str(path.relative_to(context.round_dir))


_LOWERED: TypeAdapter[_Lowered] = TypeAdapter(_Lowered)
_RESULT: TypeAdapter[ValidationResult] = TypeAdapter(ValidationResult)
