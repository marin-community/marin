# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lower a task for RolloutEngine and run k independent trials of it, classifying and retrying failures.

Every attempt of every trial is one ``TRIAL`` ledger span (``step`` is ``<kind>/<trial>/<attempt>``,
``input_hash`` the ``task_digest``, ``cause`` the ``Cause`` of an ungraded attempt) and one file
``<evidence_dir>/<kind>/<trial>/attempt-<n>.json`` holding the outcome and the whole ``RolloutData``
(or the traceback when the engine kept none), so a round is reviewed from its files alone. Both
carry ``cleanup_errors`` (RolloutEngine's ``cleanup_error_count``), so a leaked machine shows.

A trial is attempted again, after ``plan.retry_backoff``, while it is ungraded for a ``RETRYABLE``
cause (up to ``max_retries``) or a broken exact-token contract, which is sampling noise (up to
``token_contract_retries``). Attempts are numbered from ``TrialPlan.first_attempt``, so a resumed
trial passes the count already on disk and overwrites no file.

``lower`` is validation's: a task with environment requirements or files gets a ShellSim task machine
with no network and the factory's limits, no task gets a verifier machine, and the session's limits
come from ``TrialPlan.limits`` and ``VERIFIER_TIMEOUT``. A task whose answer
format cannot carry its answer (``submission_compatibility``) or that the factories cannot run
(``sandbox.factories.task_refusals``) is not started: each of its trials is one
``SUBMISSION_UNSUPPORTED`` or ``MACHINE_UNSUPPORTED`` attempt naming why.
"""

import asyncio
import dataclasses
import json
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rigging.timing import ExponentialBackoff
from rolloutengine.contracts import ModelRequest, ModelTurn, RolloutData
from rolloutengine.engine import ShellboxRolloutEngine
from rolloutengine.lowering import SHELLBOX_SESSION
from rolloutengine.spec import LoweredTaskSpec, MachineRuntimeSpec, TaskRuntimeSpec, TaskSessionSpec
from shellbox.machine import Backend, MachineFactory, NetworkPolicy
from taskcompendium.models import CONVERSATION_ANSWERS, EnvironmentRequirements, TaskSpec
from taskcompendium.submission import submission_compatibility

from taskforge.atomic_file import write_atomic
from taskforge.content_hash import sha256_hex
from taskforge.ledger.records import EntryKind, Ledger, SpanFields, span
from taskforge.sandbox.factories import FactoryCapabilities, Refusal, task_refusals
from taskforge.validate.classify import trial_outcome
from taskforge.validate.outcome import Cause, Graded, Outcome, TrialKind, Ungraded

type RolloutModel = Callable[[ModelRequest], Awaitable[ModelTurn]]

CLEANUP_ERROR_COUNT = "cleanup_error_count"
"""The ``RolloutData.metrics`` key where RolloutEngine counts failed cleanup actions."""

STARTUP_TIMEOUT = 600.0
"""Seconds the task machine may take to start."""

VERIFIER_TIMEOUT = 420.0
"""Seconds the grader may take on one attempt."""


@dataclass(frozen=True)
class EngineSettings:
    """What this host can run: the machine factories ``ShellboxRolloutEngine`` runs on, and what they
    can provide.

    Both are keyed by shellbox ``Backend`` value, as ``machine_factories`` and ``factory_capabilities``
    return them. They are facts of the host, not bounds of the run; the limits every trial runs under
    are the validation policy's ``SessionLimits``.
    """

    factories: Mapping[str, MachineFactory]
    capabilities: Mapping[str, FactoryCapabilities]

    def engine(self, model: RolloutModel) -> ShellboxRolloutEngine:
        return ShellboxRolloutEngine(model, self.factories)


def submission_refusal(task: TaskSpec) -> str | None:
    """Why the task's answer format cannot carry its answer, or ``None`` when it can.

    Only an answer read from the conversation has a format to check, as in RolloutEngine's session.
    """
    if task.answer_type not in CONVERSATION_ANSWERS:
        return None
    compatibility = submission_compatibility(task)
    return None if compatibility.compatible else "; ".join(compatibility.reasons)


@dataclass(frozen=True)
class SessionLimits:
    """The turn and time limits, in seconds, that the validation policy imposes on every trial.

    They are part of ``ValidationPolicy``, so the run root's ``policy.json`` pins them and a relaunch
    under other limits is refused rather than reusing attempts settled under these. They are a
    property of the validation run, not of the task, so ``lower`` puts them in the session rather
    than the ``TaskSpec``.

    ``max_turns`` bounds the model's turns and ``model_turn_timeout`` one model call.
    ``command_timeout`` bounds one shell command and ``tool_turn_timeout`` one tool turn, so the first
    must be shorter. ``total_turn_timeout`` bounds the model's turns together: when it expires the
    engine grades the state the agent left (stop reason ``total_turn_timeout``), or reports no grade
    when no turn had completed (``Cause.AGENT_TIMEOUT``). ``attempt_timeout`` bounds the whole
    attempt, machine start and grading included. ``cleanup_timeout`` bounds each cleanup action
    (closing a session or machine); the engine records a cleanup that fails or overruns in
    ``grade.diagnostics``.
    """

    max_turns: int
    model_turn_timeout: float
    command_timeout: float
    tool_turn_timeout: float
    total_turn_timeout: float
    attempt_timeout: float
    cleanup_timeout: float

    def __post_init__(self) -> None:
        if self.max_turns < 1:
            raise ValueError("SessionLimits need max_turns >= 1")
        if not 0 < self.command_timeout < self.tool_turn_timeout:
            raise ValueError("SessionLimits need 0 < command_timeout < tool_turn_timeout")
        if not 0 < self.total_turn_timeout < self.attempt_timeout:
            raise ValueError("SessionLimits need 0 < total_turn_timeout < attempt_timeout")
        if self.model_turn_timeout <= 0 or self.cleanup_timeout <= 0:
            raise ValueError("SessionLimits need positive model_turn_timeout and cleanup_timeout")


def lower(task: TaskSpec, limits: SessionLimits) -> LoweredTaskSpec:
    """The ``LoweredTaskSpec`` a trial of ``task`` runs: a ShellSim task machine when the task has
    environment requirements or files to install on one (RolloutEngine refuses such a task without it),
    no verifier machine, and the shellbox session under ``limits``."""
    task_machine = None
    resources = task.resources
    if task.environment_requirements != EnvironmentRequirements() or resources.all or resources.worker:
        task_machine = MachineRuntimeSpec(
            backend=Backend.SHELLSIM.value,
            network=NetworkPolicy.DENY,
            cpus=None,
            memory_mb=None,
            storage_mb=None,
            gpus=0,
            user=None,
            startup_timeout=STARTUP_TIMEOUT,
            cleanup_timeout=None,
        )
    session = TaskSessionSpec(
        task_session=SHELLBOX_SESSION,
        max_turns=limits.max_turns,
        model_turn_timeout=limits.model_turn_timeout,
        command_timeout=limits.command_timeout,
        tool_turn_timeout=limits.tool_turn_timeout,
        total_turn_timeout=limits.total_turn_timeout,
        attempt_timeout=limits.attempt_timeout,
        verifier_timeout=VERIFIER_TIMEOUT,
        cleanup_timeout=limits.cleanup_timeout,
    )
    return LoweredTaskSpec(
        task=task, runtime=TaskRuntimeSpec(task_machine=task_machine, verifier_machine=None), session=session
    )


@dataclass(frozen=True)
class RetryBackoff:
    """The wait between a trial's attempts, as data: ``rigging.timing.ExponentialBackoff``'s constructor
    arguments, so a policy holding it digests and serializes them."""

    initial: float
    maximum: float
    factor: float
    jitter: float

    def __post_init__(self) -> None:
        self.schedule()  # ExponentialBackoff's own checks, at construction rather than when trials start

    def schedule(self) -> ExponentialBackoff:
        """A fresh ``ExponentialBackoff`` with these arguments."""
        return ExponentialBackoff(initial=self.initial, maximum=self.maximum, factor=self.factor, jitter=self.jitter)


@dataclass(frozen=True)
class TrialPlan:
    """Which item the trials belong to, how many run, their session limits, how they retry, and where
    they are recorded.

    ``max_retries`` bounds retries of ``RETRYABLE`` causes and ``token_contract_retries`` those of
    ``TOKEN_CONTRACT``. Each trial waits on its own ``retry_backoff.schedule()``.
    ``first_attempt`` numbers each trial's first attempt file and ledger step; later attempts follow it.
    """

    item_id: str
    round: int
    kind: TrialKind
    k: int
    limits: SessionLimits
    max_retries: int
    token_contract_retries: int
    retry_backoff: RetryBackoff
    evidence_dir: Path
    ledger: Ledger
    first_attempt: int

    def __post_init__(self) -> None:
        if self.k < 1 or self.max_retries < 0 or self.token_contract_retries < 0 or self.first_attempt < 0:
            raise ValueError("A trial plan needs k >= 1 and non-negative retry counts and first attempt")


async def run_trials(task: TaskSpec, plan: TrialPlan, settings: EngineSettings, model: RolloutModel) -> list[Outcome]:
    """Run ``plan.k`` trials of ``task`` under ``plan.limits`` on ``settings`` concurrently;
    return one final outcome per trial, in order."""
    async with asyncio.TaskGroup() as group:
        trials = [group.create_task(run_trial(task, plan, settings, model, str(index))) for index in range(plan.k)]
    return [trial.result() for trial in trials]


async def run_trial(
    task: TaskSpec, plan: TrialPlan, settings: EngineSettings, model: RolloutModel, trial: str
) -> Outcome:
    """Run one trial of ``task``, lowered under ``plan.limits`` and run on ``settings``, attempting it again
    after a backoff while it fails for a retryable cause or, within ``plan.token_contract_retries``,
    a broken token contract."""
    lowered = lower(task, plan.limits)
    unsupported = submission_refusal(lowered.task)
    if unsupported is not None:
        return _refuse(lowered, plan, trial, Ungraded(Cause.SUBMISSION_UNSUPPORTED, unsupported, None))
    refusals = task_refusals(lowered, settings.capabilities)
    if refusals:
        return _refuse(lowered, plan, trial, Ungraded(Cause.MACHINE_UNSUPPORTED, refusal_detail(refusals), None))
    engine = settings.engine(model)
    backoff = plan.retry_backoff.schedule()
    retries = contract_retries = 0
    attempt = plan.first_attempt
    while True:
        outcome = await _attempt(engine, lowered, plan, trial, attempt)
        if isinstance(outcome, Graded):
            return outcome
        if outcome.retryable and retries < plan.max_retries:
            retries += 1
        elif outcome.cause is Cause.TOKEN_CONTRACT and contract_retries < plan.token_contract_retries:
            contract_retries += 1
        else:
            return outcome
        await asyncio.sleep(backoff.next_interval())
        attempt += 1


def refusal_detail(refusals: Sequence[Refusal]) -> str:
    return "; ".join(f"{refusal.reason}: {refusal.detail}" for refusal in refusals)


def task_digest(task: TaskSpec) -> str:
    """The sha256 of the task's JSON, its answer format included."""
    return sha256_hex(task.model_dump_json().encode())


def _refuse(lowered: LoweredTaskSpec, plan: TrialPlan, trial: str, outcome: Ungraded) -> Ungraded:
    """Record ``outcome`` as the only attempt of a trial that is not started."""
    with attempt_span(lowered, plan, trial, plan.first_attempt) as fields:
        record_attempt(fields, outcome, plan, trial, plan.first_attempt)
    return outcome


@contextmanager
def attempt_span(lowered: LoweredTaskSpec, plan: TrialPlan, trial: str, attempt: int) -> Iterator[SpanFields]:
    """The ``TRIAL`` ledger span of one attempt: step ``<kind>/<trial>/<attempt>``, ``input_hash`` the task digest."""
    with span(
        plan.ledger,
        EntryKind.TRIAL,
        item_id=plan.item_id,
        round=plan.round,
        step=f"{plan.kind}/{trial}/{attempt}",
        input_hash=task_digest(lowered.task),
    ) as fields:
        yield fields


async def _attempt(
    engine: ShellboxRolloutEngine, lowered: LoweredTaskSpec, plan: TrialPlan, trial: str, attempt: int
) -> Outcome:
    with attempt_span(lowered, plan, trial, attempt) as fields:
        try:
            result: RolloutData | Exception = await engine.run(lowered)
        except Exception as error:
            result = error
        outcome = trial_outcome(result, lowered.task)
        record_attempt(fields, outcome, plan, trial, attempt)
    return outcome


def record_attempt(fields: SpanFields, outcome: Outcome, plan: TrialPlan, trial: str, attempt: int) -> None:
    """Fill the attempt's span from ``outcome`` and write ``outcome_json(outcome)`` as its attempt file."""
    payload = outcome_json(outcome)
    fields.attrs = _attributes(outcome)
    rollout = outcome.rollout
    if rollout is not None and rollout.steps:
        fields.tokens_in = len(rollout.prompt_token_ids) + rollout.loss_mask.count(0)
        fields.tokens_out = rollout.loss_mask.count(1)
        fields.finish_reason = rollout.stop_reason
    if isinstance(outcome, Ungraded):
        fields.cause = str(outcome.cause)
    fields.output_hash = sha256_hex(payload)
    write_evidence(plan.evidence_dir / str(plan.kind) / trial / f"attempt-{attempt}.json", payload)


def cleanup_errors(outcome: Outcome) -> int | None:
    """How many cleanup actions of the attempt failed or overran; ``None`` without a rollout."""
    if outcome.rollout is None:
        return None
    return int(outcome.rollout.metrics.get(CLEANUP_ERROR_COUNT, 0))


def _attributes(outcome: Outcome) -> dict[str, str]:
    cleanup = {"cleanup_errors": str(cleanup_errors(outcome))}
    if isinstance(outcome, Graded):
        return {
            "outcome": "graded",
            "status": str(outcome.grade.status),
            "reward": str(outcome.reward),
            "timed_out": str(outcome.timed_out),
            **cleanup,
        }
    return {"outcome": "ungraded", "retryable": str(outcome.retryable), **cleanup}


def outcome_json(outcome: Outcome) -> bytes:
    """The evidence record of one attempt: the outcome and its whole ``RolloutData``."""
    record: dict[str, Any] = (
        {"outcome": "graded", "reward": outcome.reward, "timed_out": outcome.timed_out}
        if isinstance(outcome, Graded)
        else {
            "outcome": "ungraded",
            "cause": outcome.cause,
            "retryable": outcome.retryable,
            "detail": outcome.detail,
        }
    )
    record["cleanup_errors"] = cleanup_errors(outcome)
    record["rollout"] = None if outcome.rollout is None else dataclasses.asdict(outcome.rollout)
    return json.dumps(record, indent=1).encode()


def write_evidence(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(path, payload)
