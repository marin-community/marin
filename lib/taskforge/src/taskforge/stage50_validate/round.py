# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One validation round of a draft: lowering, control writing, the k solver trials, control replay and the
adversary trials as units of one graph.

Lowering (``preflight``) and control writing (``write_controls``) start together. Each solver trial starts once
the lowering is done; control replay and each adversary trial start once the lowering and the controls exist. A
unit that raises cancels the others and the round raises its error. While the workpiece has a MODIFY round left
(``ValidationContext.modify_rounds_left``), the first violated control cancels every solver and adversary trial
still running, and the result's ``EarlyDecision`` names them; without one the round waits for complete evidence.

A relaunch resumes by one rule. Each unit writes under a directory named by the digest of what it covers: the
lowering by the draft's task, the engine limits, the host and the backends; the trials by the lowered spec; the
replays by the control set; an adversary's trials by its brief, policy and model. A file found there is finished
work for exactly that input. A trial is done when its last attempt is graded or a task defect. Otherwise each
validation of the round, a review ``Retry`` or a relaunch, continues it with one ``max_retries`` budget, numbered
after every attempt on disk; the review's retry budget, which the event log counts, bounds how often that happens.
An attempt cancelled in flight writes no file, so it runs again. The result is read from the final
attempt file of each trial, so a resumed round returns what an uninterrupted one would. The README gives the layout.
"""

import asyncio
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pydantic import TypeAdapter
from rigging.filesystem.atomic import write_bytes_atomic
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
    SearchReport,
    adversary_brief,
    adversary_record,
    run_adversary_trial,
)
from taskforge.stage50_validate.control_drafts import CONTROLS_FILE, CONTROLS_STEP, load_controls, write_controls
from taskforge.stage50_validate.controls import ControlVerdict, check_replayable, control_outcome, replay_control
from taskforge.stage50_validate.evidence import Evidence, RewardStats
from taskforge.stage50_validate.outcome import TASK_DEFECTS, Cause, Graded, Outcome, TrialKind
from taskforge.stage50_validate.preflight import BackendFallback, preflight
from taskforge.stage50_validate.trials import (
    ATTEMPT_PREFIX,
    EngineSettings,
    RetryBackoff,
    TrialModel,
    TrialPlan,
    attempt_outcome,
    run_trial,
    task_digest,
)

VALIDATION_FILE = "validation.json"
VALIDATION_DIR = "validation"
LOWERED_FILE = "lowered.json"
REPLAY_PREFIX = "controls-"
SEARCH_PREFIX = "search-"
_ATTEMPT_FILE = re.compile(rf"{re.escape(ATTEMPT_PREFIX)}(\d+)\.json")
_NUMBER = re.compile(r"(\d+)")


@dataclass(frozen=True)
class ValidationContext:
    """What the stage coordinator knows of the round it validates: ``validation/`` is written under ``round_dir``,
    the controls follow ``proposal``, ``store_root`` holds the draft's grader step output, and ``modify_rounds_left``
    counts the review ``Modify`` decisions the workpiece may still follow into a new round."""

    workpiece_id: str
    round: int
    round_dir: Path
    proposal: TaskProposal
    store_root: str
    modify_rounds_left: int


@dataclass(frozen=True)
class ValidationSettings:
    """The validating host's settings, the same for every round it validates.

    Attributes:
        host: Where the machine factories run.
        engine: The engine settings every trial and the preflight probe run under.
        k: The number of solver trials.
        max_retries: The retries of a trial that fails for a retryable cause, in each validation of the round.
        retry_backoff: The wait between a trial's attempts.
        sampling: The sampling policy of the model writing the controls.
        adversary: The adversary roster, one policy per adversary with unique names; empty runs no adversary.
        ledger: Where the trials and the control writing are recorded.
    """

    host: MachineHost
    engine: EngineSettings
    k: int
    max_retries: int
    retry_backoff: RetryBackoff
    sampling: LLMPolicy
    adversary: tuple[AdversaryPolicy, ...]
    ledger: Ledger

    def __post_init__(self) -> None:
        names = [policy.spec.name for policy in self.adversary]
        if len(set(names)) != len(names):
            raise ValueError(f"adversary names must be unique in the roster: {names}")


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
    """A control's replay: ``reward`` is set when the replay was graded, ``cause`` when it produced no grade.
    ``attempt_file`` is relative to the round directory."""

    id: str
    kind: ControlKind
    verdict: ControlVerdict
    reward: float | None
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
    """An adversary trial's final attempt: ``name`` is its ``AdversarySpec``'s, ``cause`` is set when it is ungraded,
    and ``report`` is the adversary's own ``finish_search`` report. ``attempt_file`` is relative to the round
    directory."""

    trial: str
    name: str
    mode: AdversaryMode
    status: TrialStatus
    cause: Cause | None
    report: SearchReport | None
    submissions: tuple[SubmissionResult, ...]
    attempt_file: str


@dataclass(frozen=True)
class EarlyDecision:
    """A round its first violated control decided before every trial ended.

    ``control`` is the id of that control, and ``cancelled`` names each solver and adversary trial the round
    stopped, as ``solver/<i>`` or ``adversary/<name>-<i>``; none of them is in the round's result.
    """

    control: str
    cancelled: tuple[str, ...]


@dataclass(frozen=True)
class ValidationResult:
    """A validation round, written as ``validation.json``: ``solver`` is ``None`` and ``controls`` and
    ``adversary`` empty when the lowering was rejected. ``early`` is set when a violated control decided the
    round before every trial ended, and ``solver`` and ``adversary`` then hold only the trials that ended."""

    workpiece_id: str
    round: int
    lowering: Lowering
    solver: SolverEvidence | None
    controls: tuple[ControlResult, ...]
    adversary: tuple[AdversaryResult, ...]
    early: EarlyDecision | None


@dataclass(frozen=True)
class _Lowered:
    """The spec the task lowered to (``None`` when RolloutEngine rejected it, with the reason in ``detail``), and the
    preflight's refusals and fallback."""

    spec: TaskRolloutSpec | None
    detail: str
    refusals: tuple[Refusal, ...]
    fallback: BackendFallback | None


@dataclass(frozen=True)
class _Final:
    """A finished trial's last attempt: its number, its file and the file's bytes."""

    attempt: int
    path: Path
    payload: bytes

    @property
    def outcome(self) -> Outcome:
        return attempt_outcome(self.payload)


async def validate_round(
    draft: TaskDraft,
    context: ValidationContext,
    settings: ValidationSettings,
    solver_model: TrialModel,
    controls_model: ModelClient,
    adversary_model: ModelClient,
) -> ValidationResult:
    """Validate ``draft``, resuming missing units and trials eligible for another attempt.

    A stage coordinator passes ``context``; a caller's validate function closes over ``settings`` and the models.
    ``write_validation`` writes the result as ``validation.json``.

    Raises:
        BuildInfrastructureFailure: the preflight probe's machine did not start twice.
        BuildFailure: the controls never passed their checks (``write_controls``).
        ValueError: ``check_replayable`` refuses the written controls.
    """
    directory = context.round_dir / VALIDATION_DIR / digest(draft.task)
    try:
        async with asyncio.TaskGroup() as group:
            lowering = group.create_task(_lowering_unit(draft, settings, directory))
            controls = group.create_task(_controls_unit(draft, context, settings, controls_model, directory))
            solver = {
                str(index): group.create_task(
                    _solver_unit(lowering, context, settings, solver_model, directory, str(index))
                )
                for index in range(settings.k)
            }
            adversary = {
                trial: group.create_task(
                    _adversary_unit(lowering, controls, context, settings, adversary_model, policy, directory, trial)
                )
                for policy, trial in _adversary_trials(settings)
            }
            trials = {f"{TrialKind.SOLVER}/{name}": task for name, task in solver.items()} | {
                f"{TrialKind.ADVERSARY}/{name}": task for name, task in adversary.items()
            }
            closing = _EarlyClose(armed=context.modify_rounds_left > 0, trials=trials)
            replays = group.create_task(_replay_units(lowering, controls, context, settings, directory, closing))
    except BaseExceptionGroup as failure:
        if len(failure.exceptions) == 1:
            raise failure.exceptions[0] from None
        raise
    lowered = lowering.result()
    if lowered.spec is None:
        rejected = Lowering(LoweringStatus.REJECTED, lowered.detail, (), None, None)
        return ValidationResult(context.workpiece_id, context.round, rejected, None, (), (), None)
    cancelled = tuple(name for name, task in trials.items() if task.cancelled())
    early = None
    if cancelled:
        assert closing.control is not None, f"{context.workpiece_id}: trials {cancelled} were cancelled undecided"
        early = EarlyDecision(closing.control, cancelled)
    status = LoweringStatus.REFUSED if lowered.refusals else LoweringStatus.LOWERED
    trial_digest = task_digest(lowered.spec, settings.engine.command_timeout)
    solver_finals = {name: _done(task) for name, task in solver.items() if not task.cancelled()}
    return ValidationResult(
        context.workpiece_id,
        context.round,
        Lowering(status, "", lowered.refusals, lowered.fallback, trial_digest),
        _solver_evidence(context, solver_finals),
        tuple(_control_result(context, control, final) for control, final in _done(replays)),
        tuple(_adversary_result(context, name, _done(task)) for name, task in adversary.items() if not task.cancelled()),
        early,
    )


def write_validation(round_dir: Path, result: ValidationResult) -> None:
    """Write ``result`` to ``round_dir`` as ``validation.json``, whole or not at all."""
    write_bytes_atomic(str(round_dir / VALIDATION_FILE), pretty_json(result).encode())


def load_validation(round_dir: Path) -> ValidationResult:
    """Read the ``validation.json`` ``write_validation`` wrote to ``round_dir``."""
    return _RESULT.validate_json((round_dir / VALIDATION_FILE).read_bytes())


def _done[T](task: asyncio.Task[T | None]) -> T:
    """The result of a unit that ran past a lowered spec."""
    result = task.result()
    assert result is not None, "a unit after a lowered spec returns its evidence"
    return result


async def _inputs_ready(*inputs: asyncio.Task) -> bool:
    """Wait for ``inputs``; whether each returned. A failed input is the group's error, not its dependents'."""
    await asyncio.wait(inputs)
    return all(not task.cancelled() and task.exception() is None for task in inputs)


async def _lowering_unit(draft: TaskDraft, settings: ValidationSettings, directory: Path) -> _Lowered:
    engine = settings.engine
    covered = {"limits": engine.limits, "host": settings.host, "backends": sorted(engine.factories)}
    path = directory / f"{digest(covered)}-{LOWERED_FILE}"
    if not path.exists():
        write_bytes_atomic(str(path), pretty_json(await _lower(draft, settings)).encode())
    return _LOWERED.validate_json(path.read_bytes())


async def _lower(draft: TaskDraft, settings: ValidationSettings) -> _Lowered:
    try:
        lowered = await preflight(draft.task, settings.host, settings.engine)
    except (ValueError, NotImplementedError) as error:
        return _Lowered(None, str(error), (), None)
    return _Lowered(lowered.spec, "", tuple(task_refusals(lowered.spec, settings.engine.factories)), lowered.fallback)


def _trials_dir(directory: Path, settings: ValidationSettings, spec: TaskRolloutSpec) -> Path:
    return directory / task_digest(spec, settings.engine.command_timeout)


async def _controls_unit(
    draft: TaskDraft, context: ValidationContext, settings: ValidationSettings, model: ModelClient, directory: Path
) -> tuple[Control, ...]:
    if not (directory / CONTROLS_FILE).exists():
        grader = step_output(draft, context.store_root, GRADER_STEP, Grader)
        record = CallLedger(settings.ledger, context.workpiece_id, context.round, CONTROLS_STEP)
        await write_controls(draft.task, grader, context.proposal, model, settings.sampling, record, directory)
    return load_controls(directory)


@dataclass
class _EarlyClose:
    """Cancels the round's ``trials`` that have not ended on its first violated control, when ``armed``.

    A round is armed while the workpiece has a MODIFY round left; without one it waits for every trial.
    ``control`` records the control that closed the round.
    """

    armed: bool
    trials: Mapping[str, asyncio.Task]
    control: str | None = None

    def violated(self, control_id: str) -> None:
        if not self.armed or self.control is not None:
            return
        self.control = control_id
        for task in self.trials.values():
            task.cancel()


async def _trial(
    context: ValidationContext,
    settings: ValidationSettings,
    kind: TrialKind,
    evidence_dir: Path,
    trial: str,
    run: Callable[[TrialPlan], Awaitable[object]],
) -> _Final:
    """Run the rest of ``trial`` under ``evidence_dir`` with ``run``, and return its last attempt.

    A trial whose last attempt is graded or a task defect is done. Otherwise it runs with one ``max_retries`` budget,
    its first attempt numbered after every attempt file and every submissions directory a cancelled attempt left.
    """
    trial_dir = evidence_dir / str(kind) / trial
    attempts = _numbered(trial_dir, _ATTEMPT_FILE)
    last = _final(trial_dir, attempts) if attempts else None
    if last is None or not _finished(last.outcome):
        taken = attempts + _numbered(trial_dir / SUBMISSIONS_DIR, _NUMBER)
        plan = TrialPlan(
            workpiece_id=context.workpiece_id,
            round=context.round,
            kind=kind,
            k=1,
            max_retries=settings.max_retries,
            retry_backoff=settings.retry_backoff,
            evidence_dir=evidence_dir,
            ledger=settings.ledger,
            first_attempt=max(taken) + 1 if taken else 0,
        )
        await run(plan)
        last = _final(trial_dir, _numbered(trial_dir, _ATTEMPT_FILE))
    return last


def _finished(outcome: Outcome) -> bool:
    return isinstance(outcome, Graded) or outcome.cause in TASK_DEFECTS


async def _solver_unit(
    lowering: asyncio.Task[_Lowered],
    context: ValidationContext,
    settings: ValidationSettings,
    model: TrialModel,
    directory: Path,
    trial: str,
) -> _Final | None:
    if not await _inputs_ready(lowering) or (spec := lowering.result().spec) is None:
        return None

    def run(plan: TrialPlan) -> Awaitable[Outcome]:
        return run_trial(spec, plan, settings.engine, model, trial)

    return await _trial(context, settings, TrialKind.SOLVER, _trials_dir(directory, settings, spec), trial, run)


async def _replay_units(
    lowering: asyncio.Task[_Lowered],
    controls: asyncio.Task[tuple[Control, ...]],
    context: ValidationContext,
    settings: ValidationSettings,
    directory: Path,
    closing: _EarlyClose,
) -> tuple[tuple[Control, _Final], ...] | None:
    if not await _inputs_ready(lowering, controls) or (spec := lowering.result().spec) is None:
        return None
    control_set = controls.result()
    evidence_dir = _trials_dir(directory, settings, spec) / f"{REPLAY_PREFIX}{digest(control_set)}"
    check_replayable(spec.task, control_set, settings.engine.limits.max_turns)

    async def replayed(control: Control) -> tuple[Control, _Final]:
        def run(plan: TrialPlan) -> Awaitable[object]:
            return replay_control(spec, control, plan, settings.engine)

        final = await _trial(context, settings, TrialKind.CONTROL, evidence_dir, control.id, run)
        if control_outcome(control, final.outcome).verdict is ControlVerdict.VIOLATED:
            closing.violated(control.id)
        return control, final

    async with asyncio.TaskGroup() as group:
        units = [group.create_task(replayed(control)) for control in control_set]
    return tuple(unit.result() for unit in units)


async def _adversary_unit(
    lowering: asyncio.Task[_Lowered],
    controls: asyncio.Task[tuple[Control, ...]],
    context: ValidationContext,
    settings: ValidationSettings,
    model: ModelClient,
    policy: AdversaryPolicy,
    directory: Path,
    trial: str,
) -> _Final | None:
    """Run the trial under a directory bound to its brief, policy and model: a search under another one stays, and
    this one runs with its full retries."""
    if not await _inputs_ready(lowering, controls) or (spec := lowering.result().spec) is None:
        return None
    brief = adversary_brief(spec, controls.result(), policy)
    bound = digest({"brief": brief, "policy": policy, "model": model.model})
    evidence_dir = _trials_dir(directory, settings, spec) / f"{SEARCH_PREFIX}{bound}"

    def run(plan: TrialPlan) -> Awaitable[Outcome]:
        return run_adversary_trial(spec, brief, policy, plan, settings.engine, model, trial, settings.max_retries)

    return await _trial(context, settings, TrialKind.ADVERSARY, evidence_dir, trial, run)


def _adversary_trials(settings: ValidationSettings) -> list[tuple[AdversaryPolicy, str]]:
    """Each adversary trial of the roster with its policy, named ``<name>-<i>``."""
    return [(policy, f"{policy.spec.name}-{index}") for policy in settings.adversary for index in range(policy.k)]


def _numbered(directory: Path, pattern: re.Pattern[str]) -> list[int]:
    if not directory.is_dir():
        return []
    matches = (pattern.fullmatch(path.name) for path in directory.iterdir())
    return sorted(int(match.group(1)) for match in matches if match is not None)


def _final(trial_dir: Path, attempts: list[int]) -> _Final:
    assert attempts, f"the trial {trial_dir} has no attempt"
    path = trial_dir / f"{ATTEMPT_PREFIX}{attempts[-1]}.json"
    return _Final(attempts[-1], path, path.read_bytes())


def _relative(context: ValidationContext, path: Path) -> str:
    return str(path.relative_to(context.round_dir))


def _solver_evidence(context: ValidationContext, finals: Mapping[str, _Final]) -> SolverEvidence:
    outcomes = {trial: final.outcome for trial, final in finals.items()}
    evidence = Evidence({TrialKind.SOLVER: tuple(outcomes.values())})
    return SolverEvidence(
        trials=tuple(
            _trial_result(trial, outcomes[trial], _relative(context, final.path)) for trial, final in finals.items()
        ),
        stats=evidence.reward_stats(TrialKind.SOLVER),
        ungraded=dict(evidence.ungraded),
    )


def _trial_result(trial: str, outcome: Outcome, attempt_file: str) -> TrialResult:
    if isinstance(outcome, Graded):
        return TrialResult(trial, TrialStatus.GRADED, outcome.reward, None, attempt_file)
    return TrialResult(trial, TrialStatus.UNGRADED, None, outcome.cause, attempt_file)


def _control_result(context: ValidationContext, control: Control, final: _Final) -> ControlResult:
    outcome = final.outcome
    verdict = control_outcome(control, outcome).verdict
    reward, cause = (outcome.reward, None) if isinstance(outcome, Graded) else (None, outcome.cause)
    return ControlResult(control.id, control.kind, verdict, reward, cause, _relative(context, final.path))


def _adversary_result(context: ValidationContext, trial: str, final: _Final) -> AdversaryResult:
    record = adversary_record(final.payload)
    graded = _trial_result(trial, final.outcome, "")
    submissions_dir = final.path.parent / SUBMISSIONS_DIR / str(final.attempt)
    submissions = []
    for submission in record.submissions:
        grading_dir = submissions_dir / str(TrialKind.CONTROL) / submission.trial
        grading = _final(grading_dir, _numbered(grading_dir, _ATTEMPT_FILE))
        grade = submission.grade
        submissions.append(
            SubmissionResult(
                submission.ordinal, grade.status, grade.reward, submission.passed, _relative(context, grading.path)
            )
        )
    return AdversaryResult(
        trial,
        record.name,
        record.mode,
        graded.status,
        graded.cause,
        record.report,
        tuple(submissions),
        _relative(context, final.path),
    )


_LOWERED: TypeAdapter[_Lowered] = TypeAdapter(_Lowered)
_RESULT: TypeAdapter[ValidationResult] = TypeAdapter(ValidationResult)
