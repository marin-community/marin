# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run k independent trials of a task through RolloutEngine, classifying and retrying failures.

Every attempt of every trial is one ``TRIAL`` ledger span (``step`` is ``<kind>/<trial>/<attempt>``,
``cause`` is the ``Cause`` value of an ungraded attempt) and one JSON file under
``<evidence_dir>/<kind>/<trial>/attempt-<n>.json`` holding the outcome and the full ``RolloutData``
(or the traceback when the engine kept no rollout). Both carry ``cleanup_errors``, the number of
cleanup actions that failed or overran (RolloutEngine's ``cleanup_error_count``), so a leaked
machine shows in summaries; it is ``None`` when the engine kept no rollout.

A trial is attempted again while its outcome is ``Ungraded`` with a retryable cause, up to
``max_retries`` times, waiting ``plan.retry_backoff`` between attempts. Attempts are numbered from
``TrialPlan.first_attempt``: a trial re-entered after earlier attempts passes the count already on
disk, so no attempt file is overwritten and each ledger step stays unique.

A sampled rollout breaks the exact-token contract when the model samples a token sequence the
server does not reproduce when it re-renders the conversation (``llm.rollout_model``). That is
sampling noise, so a ``TOKEN_CONTRACT`` attempt is retried on its own budget,
``token_contract_retries``; each attempt's record keeps the traceback naming the divergent ids. A
control replays fixed turns, so its plan has no such budget.

Validation owns its trials' deadlines: ``TrialPlan.deadlines`` replaces the agent and attempt
deadlines of the builder's ``TaskExecution`` (``Deadlines.apply``), so every trial is bounded
whatever the builder set, and the ledger's ``input_hash`` covers the effective execution.

The submission convention is chosen per task (``task_convention``) from
``EngineSettings.conventions``: it is presentation, not part of the task, and it changes the
instruction the solver sees, so ``input_hash`` covers it too. A task no convention can carry is not
started: each of its trials is one ``SUBMISSION_UNSUPPORTED`` attempt naming the reasons.

``EngineSettings.factories`` and ``EngineSettings.capabilities`` come from
``taskforge.sandbox.factories.machine_factories`` and ``factory_capabilities`` for the same host. A
task those factories cannot run (``task_refusals``) is not started: each of its trials is one
``MACHINE_UNSUPPORTED`` attempt naming the refusals.
"""

import asyncio
import copy
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
from shellbox.machine import MachineFactory
from taskcompendium.environment import EnvironmentKind
from taskcompendium.execution import TaskExecution
from taskcompendium.models import TaskSpec
from taskcompendium.submission import SubmissionConvention, submission_compatibility

from taskforge.content_hash import sha256_hex
from taskforge.ledger.records import EntryKind, Ledger, SpanFields, span
from taskforge.sandbox.factories import FactoryCapabilities, Refusal, task_refusals
from taskforge.spec.draft import MACHINE_ANSWER_TYPES
from taskforge.validate.classify import trial_outcome
from taskforge.validate.outcome import Cause, Graded, Outcome, TrialKind, Ungraded

type RolloutModel = Callable[[ModelRequest], Awaitable[ModelTurn]]

CLEANUP_ERROR_COUNT = "cleanup_error_count"
"""The ``RolloutData.metrics`` key where RolloutEngine counts failed cleanup actions."""


@dataclass(frozen=True)
class EngineSettings:
    """Everything ``ShellboxRolloutEngine`` takes except the model, what the factories can run, and the
    submission conventions tasks may be presented with.

    ``cleanup_timeout`` bounds each cleanup action (closing a session or machine, removing a stage
    grader); the engine records a cleanup that fails or overruns in ``grade.diagnostics``.
    ``conventions`` is in preference order; each task runs under the first that can carry its answer
    (``task_convention``).
    """

    factories: Mapping[EnvironmentKind, MachineFactory]
    capabilities: Mapping[EnvironmentKind, FactoryCapabilities]
    max_turns: int
    command_timeout: float
    cleanup_timeout: float
    conventions: tuple[SubmissionConvention, ...]

    def __post_init__(self) -> None:
        ids = [convention.id for convention in self.conventions]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError(f"EngineSettings needs at least one convention and unique convention ids, got {ids}")

    def engine(self, model: RolloutModel, convention: SubmissionConvention) -> ShellboxRolloutEngine:
        return ShellboxRolloutEngine(
            model,
            self.factories,
            max_turns=self.max_turns,
            command_timeout=self.command_timeout,
            cleanup_timeout=self.cleanup_timeout,
            convention=convention,
        )


class ConventionUnavailable(ValueError):
    """No convention can carry a task's answer; the message gives each convention's reasons."""


def task_convention(task: TaskSpec, conventions: Sequence[SubmissionConvention]) -> SubmissionConvention:
    """The first of ``conventions`` that ``submission_compatibility`` accepts for ``task``.

    A task whose answer is the machine state (``MACHINE_ANSWER_TYPES``) submits nothing through a
    convention, so it takes the first.

    Raises:
        ConventionUnavailable: no convention is compatible with ``task``.
    """
    if task.answer_type in MACHINE_ANSWER_TYPES:
        return conventions[0]
    reasons = []
    for convention in conventions:
        compatibility = submission_compatibility(task, convention)
        if compatibility.compatible:
            return convention
        reasons.append(f"{convention.id}: {'; '.join(compatibility.reasons)}")
    raise ConventionUnavailable("; ".join(reasons))


@dataclass(frozen=True)
class Deadlines:
    """The agent and attempt deadlines, in seconds, that validation imposes on every trial.

    They are a property of the validation run, not of the task, so they live beside the
    ``TaskExecution`` a builder returns rather than in it or in the ``TaskSpec``.
    """

    agent_timeout: float
    attempt_timeout: float

    def __post_init__(self) -> None:
        if not 0 < self.agent_timeout < self.attempt_timeout:
            raise ValueError("Deadlines need 0 < agent_timeout < attempt_timeout")

    def apply(self, execution: TaskExecution) -> TaskExecution:
        """``execution`` with these deadlines in place of its own, including every stage's agent deadline."""
        stages = {
            name: stage.model_copy(update={"agent_timeout": self.agent_timeout})
            for name, stage in execution.stages.items()
        }
        return execution.model_copy(
            update={"agent_timeout": self.agent_timeout, "attempt_timeout": self.attempt_timeout, "stages": stages}
        )


@dataclass(frozen=True)
class RetryBackoff:
    """The wait between a trial's attempts, as data: ``rigging.timing.ExponentialBackoff``'s constructor
    arguments, so a policy holding it digests and serializes them."""

    initial: float
    maximum: float
    factor: float
    jitter: float

    def schedule(self) -> ExponentialBackoff:
        """A fresh ``ExponentialBackoff`` with these arguments."""
        return ExponentialBackoff(initial=self.initial, maximum=self.maximum, factor=self.factor, jitter=self.jitter)


@dataclass(frozen=True)
class TrialPlan:
    """Which item the trials belong to, how many run, their deadlines, how they retry, and where they
    are recorded.

    ``max_retries`` bounds retries of ``RETRYABLE`` causes and ``token_contract_retries`` those of
    ``TOKEN_CONTRACT``. ``retry_backoff`` is a template: each trial waits on its own copy.
    ``first_attempt`` numbers each trial's first attempt file and ledger step; later attempts follow it.
    """

    item_id: str
    round: int
    kind: TrialKind
    k: int
    deadlines: Deadlines
    max_retries: int
    token_contract_retries: int
    retry_backoff: ExponentialBackoff
    evidence_dir: Path
    ledger: Ledger
    first_attempt: int

    def __post_init__(self) -> None:
        if self.k < 1 or self.max_retries < 0 or self.token_contract_retries < 0 or self.first_attempt < 0:
            raise ValueError("A trial plan needs k >= 1 and non-negative retry counts and first attempt")
        if self.kind is TrialKind.CONTROL and self.token_contract_retries:
            raise ValueError("A control replays fixed turns, so a token contract break is not retried")


async def run_trials(
    task: TaskSpec, execution: TaskExecution, plan: TrialPlan, settings: EngineSettings, model: RolloutModel
) -> list[Outcome]:
    """Run ``plan.k`` trials of ``task`` with ``execution`` under ``plan.deadlines`` concurrently; return
    one final outcome per trial, in order."""
    async with asyncio.TaskGroup() as group:
        trials = [
            group.create_task(run_trial(task, execution, plan, settings, model, str(index))) for index in range(plan.k)
        ]
    return [trial.result() for trial in trials]


async def run_trial(
    task: TaskSpec,
    execution: TaskExecution,
    plan: TrialPlan,
    settings: EngineSettings,
    model: RolloutModel,
    trial: str,
) -> Outcome:
    """Run one trial of ``task`` with ``execution`` under ``plan.deadlines`` and its ``task_convention``,
    attempting it again after a backoff while it fails for a retryable cause or, within
    ``plan.token_contract_retries``, a broken token contract."""
    execution = plan.deadlines.apply(execution)
    try:
        convention = task_convention(task, settings.conventions)
    except ConventionUnavailable as error:
        return _refuse(task, execution, None, plan, trial, Ungraded(Cause.SUBMISSION_UNSUPPORTED, str(error), None))
    refusals = task_refusals(task, execution, settings.capabilities)
    if refusals:
        outcome = Ungraded(Cause.MACHINE_UNSUPPORTED, refusal_detail(refusals), None)
        return _refuse(task, execution, convention, plan, trial, outcome)
    engine = settings.engine(model, convention)
    backoff = copy.copy(plan.retry_backoff)
    retries = contract_retries = 0
    attempt = plan.first_attempt
    while True:
        outcome = await _attempt(engine, task, execution, convention, plan, trial, attempt)
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
    return "; ".join(f"{refusal.where}: {refusal.reason}: {refusal.detail}" for refusal in refusals)


def task_digest(task: TaskSpec, execution: TaskExecution, convention: SubmissionConvention | None) -> str:
    """The sha256 of the task, the execution settings it runs with, and the convention it is presented
    with (``None`` when no convention can carry it)."""
    presented = "null" if convention is None else f"{type(convention).__name__} {convention.model_dump_json()}"
    payload = f"{task.model_dump_json()}\n{execution.model_dump_json()}\n{presented}"
    return sha256_hex(payload.encode())


def _refuse(
    task: TaskSpec,
    execution: TaskExecution,
    convention: SubmissionConvention | None,
    plan: TrialPlan,
    trial: str,
    outcome: Ungraded,
) -> Ungraded:
    """Record ``outcome`` as the only attempt of a trial that is not started."""
    with _attempt_span(task, execution, convention, plan, trial, plan.first_attempt) as fields:
        _record(fields, outcome, plan, trial, plan.first_attempt)
    return outcome


@contextmanager
def _attempt_span(
    task: TaskSpec,
    execution: TaskExecution,
    convention: SubmissionConvention | None,
    plan: TrialPlan,
    trial: str,
    attempt: int,
) -> Iterator[SpanFields]:
    with span(
        plan.ledger,
        EntryKind.TRIAL,
        item_id=plan.item_id,
        round=plan.round,
        step=f"{plan.kind}/{trial}/{attempt}",
        input_hash=task_digest(task, execution, convention),
    ) as fields:
        yield fields


async def _attempt(
    engine: ShellboxRolloutEngine,
    task: TaskSpec,
    execution: TaskExecution,
    convention: SubmissionConvention,
    plan: TrialPlan,
    trial: str,
    attempt: int,
) -> Outcome:
    with _attempt_span(task, execution, convention, plan, trial, attempt) as fields:
        try:
            result: RolloutData | Exception = await engine.run(task, execution=execution)
        except Exception as error:
            result = error
        outcome = trial_outcome(result)
        _record(fields, outcome, plan, trial, attempt)
    return outcome


def _record(fields: SpanFields, outcome: Outcome, plan: TrialPlan, trial: str, attempt: int) -> None:
    fields.attrs = _attributes(outcome)
    rollout = outcome.rollout
    if rollout is not None and rollout.steps:
        fields.tokens_in = len(rollout.prompt_token_ids) + rollout.loss_mask.count(0)
        fields.tokens_out = rollout.loss_mask.count(1)
        fields.finish_reason = rollout.stop_reason
    if isinstance(outcome, Ungraded):
        fields.cause = str(outcome.cause)
    payload = outcome_json(outcome)
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
    """The evidence record of one attempt."""
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
    path.write_bytes(payload)
