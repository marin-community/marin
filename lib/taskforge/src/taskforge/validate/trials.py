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

Validation owns its trials' deadlines and engine limits: ``TrialPlan.deadlines`` replaces the
total-turn and attempt deadlines of the builder's ``TaskSessionSpec`` (``Deadlines.apply``) and
``EngineSettings.apply`` its turn, command, tool-turn, model-turn and cleanup limits, so every trial
is bounded whatever the builder set, and the ledger's ``input_hash`` covers the effective
``LoweredTaskSpec`` (``task_digest``). The builder's verifier deadline and machine settings stand.

The attempt file holds the whole ``RolloutData``, steps included: a validation round is reviewed
from its files alone, and the final reply, the tool calls and the per-turn usage are read from the
steps.

The answer format is part of the task (``TaskSpec.answer_format``), so ``input_hash`` covers it
with the rest of the lowered spec. A task whose format cannot carry its answer for its grader
(TaskCompendium ``submission_compatibility``) is not started: each of its trials is one
``SUBMISSION_UNSUPPORTED`` attempt naming the reasons.

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
from rolloutengine.spec import LoweredTaskSpec
from shellbox.machine import MachineFactory
from taskcompendium.models import CONVERSATION_ANSWERS, TaskSpec
from taskcompendium.submission import submission_compatibility

from taskforge.content_hash import sha256_hex
from taskforge.ledger.records import EntryKind, Ledger, SpanFields, span
from taskforge.sandbox.factories import FactoryCapabilities, Refusal, task_refusals
from taskforge.validate.classify import trial_outcome
from taskforge.validate.outcome import Cause, Graded, Outcome, TrialKind, Ungraded

type RolloutModel = Callable[[ModelRequest], Awaitable[ModelTurn]]

CLEANUP_ERROR_COUNT = "cleanup_error_count"
"""The ``RolloutData.metrics`` key where RolloutEngine counts failed cleanup actions."""


@dataclass(frozen=True)
class EngineSettings:
    """The machine factories ``ShellboxRolloutEngine`` runs on, what they can run, and the session
    limits validation imposes on every trial.

    ``factories`` and ``capabilities`` are keyed by shellbox ``Backend`` value, as
    ``machine_factories`` and ``factory_capabilities`` return them. ``command_timeout`` bounds one
    shell command and ``tool_turn_timeout`` one tool turn, so the first must be shorter.
    ``model_turn_timeout`` bounds one model call. ``cleanup_timeout`` bounds each cleanup action
    (closing a session or machine); the engine records a cleanup that fails or overruns in
    ``grade.diagnostics``.
    """

    factories: Mapping[str, MachineFactory]
    capabilities: Mapping[str, FactoryCapabilities]
    max_turns: int
    command_timeout: float
    tool_turn_timeout: float
    model_turn_timeout: float
    cleanup_timeout: float

    def __post_init__(self) -> None:
        if self.max_turns < 1:
            raise ValueError("EngineSettings needs max_turns >= 1")
        if not 0 < self.command_timeout < self.tool_turn_timeout:
            raise ValueError("EngineSettings needs 0 < command_timeout < tool_turn_timeout")
        if self.model_turn_timeout <= 0 or self.cleanup_timeout <= 0:
            raise ValueError("EngineSettings needs positive model_turn_timeout and cleanup_timeout")

    def apply(self, lowered: LoweredTaskSpec) -> LoweredTaskSpec:
        """``lowered`` with these session limits in place of the builder's."""
        session = lowered.session.model_copy(
            update={
                "max_turns": self.max_turns,
                "command_timeout": self.command_timeout,
                "tool_turn_timeout": self.tool_turn_timeout,
                "model_turn_timeout": self.model_turn_timeout,
                "cleanup_timeout": self.cleanup_timeout,
            }
        )
        return lowered.model_copy(update={"session": session})

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
class Deadlines:
    """The total-turn and attempt deadlines, in seconds, that validation imposes on every trial.

    ``total_turn_timeout`` bounds the model's turns together: when it expires the engine grades the
    state the agent left (stop reason ``total_turn_timeout``), or reports no grade when no turn had
    completed (``Cause.AGENT_TIMEOUT``). ``attempt_timeout`` bounds the whole attempt, machine start
    and grading included. They are a property of the validation run, not of the task, so they
    replace the builder's ``TaskSessionSpec`` values rather than living in the ``TaskSpec``.
    """

    total_turn_timeout: float
    attempt_timeout: float

    def __post_init__(self) -> None:
        if not 0 < self.total_turn_timeout < self.attempt_timeout:
            raise ValueError("Deadlines need 0 < total_turn_timeout < attempt_timeout")

    def apply(self, lowered: LoweredTaskSpec) -> LoweredTaskSpec:
        """``lowered`` with these deadlines in place of its session's own."""
        session = lowered.session.model_copy(
            update={"total_turn_timeout": self.total_turn_timeout, "attempt_timeout": self.attempt_timeout}
        )
        return lowered.model_copy(update={"session": session})


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
    lowered: LoweredTaskSpec, plan: TrialPlan, settings: EngineSettings, model: RolloutModel
) -> list[Outcome]:
    """Run ``plan.k`` trials of ``lowered`` under ``plan.deadlines`` and ``settings`` concurrently;
    return one final outcome per trial, in order."""
    async with asyncio.TaskGroup() as group:
        trials = [group.create_task(run_trial(lowered, plan, settings, model, str(index))) for index in range(plan.k)]
    return [trial.result() for trial in trials]


async def run_trial(
    lowered: LoweredTaskSpec, plan: TrialPlan, settings: EngineSettings, model: RolloutModel, trial: str
) -> Outcome:
    """Run one trial of ``lowered`` under ``plan.deadlines`` and ``settings``, attempting it again
    after a backoff while it fails for a retryable cause or, within ``plan.token_contract_retries``,
    a broken token contract."""
    lowered = settings.apply(plan.deadlines.apply(lowered))
    unsupported = submission_refusal(lowered.task)
    if unsupported is not None:
        return _refuse(lowered, plan, trial, Ungraded(Cause.SUBMISSION_UNSUPPORTED, unsupported, None))
    refusals = task_refusals(lowered, settings.capabilities)
    if refusals:
        return _refuse(lowered, plan, trial, Ungraded(Cause.MACHINE_UNSUPPORTED, refusal_detail(refusals), None))
    engine = settings.engine(model)
    backoff = copy.copy(plan.retry_backoff)
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
    return "; ".join(f"{refusal.where}: {refusal.reason}: {refusal.detail}" for refusal in refusals)


def task_digest(lowered: LoweredTaskSpec) -> str:
    """The sha256 of the lowered task as it runs (after ``Deadlines.apply`` and ``EngineSettings.apply``),
    its answer format included."""
    return sha256_hex(lowered.model_dump_json().encode())


def _refuse(lowered: LoweredTaskSpec, plan: TrialPlan, trial: str, outcome: Ungraded) -> Ungraded:
    """Record ``outcome`` as the only attempt of a trial that is not started."""
    with _attempt_span(lowered, plan, trial, plan.first_attempt) as fields:
        _record(fields, outcome, plan, trial, plan.first_attempt)
    return outcome


@contextmanager
def _attempt_span(lowered: LoweredTaskSpec, plan: TrialPlan, trial: str, attempt: int) -> Iterator[SpanFields]:
    with span(
        plan.ledger,
        EntryKind.TRIAL,
        item_id=plan.item_id,
        round=plan.round,
        step=f"{plan.kind}/{trial}/{attempt}",
        input_hash=task_digest(lowered),
    ) as fields:
        yield fields


async def _attempt(
    engine: ShellboxRolloutEngine, lowered: LoweredTaskSpec, plan: TrialPlan, trial: str, attempt: int
) -> Outcome:
    with _attempt_span(lowered, plan, trial, attempt) as fields:
        try:
            result: RolloutData | Exception = await engine.run(lowered)
        except Exception as error:
            result = error
        outcome = trial_outcome(result, lowered.task)
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
    path.write_bytes(payload)
