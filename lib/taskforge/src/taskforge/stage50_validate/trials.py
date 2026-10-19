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
disk, so no attempt file is overwritten and each ledger step stays unique. Each attempt runs on
its own model, ``model.for_step(<kind>/<trial>/<attempt>)``: a ``GlmRolloutModel`` records its
``LLM_CALL`` spans under the attempt's step and routes on it, so the k trials of a workpiece do not
share one prefix-cache key.

Every trial runs under the engine's messages trace (``TaskSessionSpec.trace = MESSAGES``, set by
``EngineSettings.apply``): a trial is graded and never trained on, so the engine records the token
ids a model reports without checking them, with an all-zero loss mask, and a model's next prompt
need not extend the ids of its last turn. The attempt span's ``tokens_in`` and
``tokens_out`` sum the prompt and response ids of the attempt's turns.

A graded attempt's record says whether the candidate's own code did not load
(``candidate_code_error``); the span carries ``candidate_code_error="true"`` when it did.

Validation owns its trials' deadlines and engine limits: ``TrialPlan.deadlines`` replaces the
total-turn and attempt deadlines of the builder's ``TaskSessionSpec`` (``Deadlines.apply``) and
``EngineSettings.apply`` its trace and its turn, command, tool-turn, model-turn and cleanup limits,
so every trial is bounded whatever the builder set, and the ledger's ``input_hash`` covers the
effective ``LoweredTaskSpec`` (``task_digest``). The builder's verifier deadline and machine settings stand.

The attempt file holds the whole ``RolloutData``, steps included: a validation round is reviewed
from its files alone, and the final reply, the tool calls and the per-turn usage are read from the
steps.

The answer format is part of the task (``TaskSpec.answer_format``), so ``input_hash`` covers it
with the rest of the lowered spec. A task whose format cannot carry its answer for its grader
(TaskCompendium ``submission_compatibility``) is not started: each of its trials is one
``SUBMISSION_UNSUPPORTED`` attempt naming the reasons.

``EngineSettings.factories`` come from ``taskforge.sandbox.factories.machine_factories``. Before a
trial starts, ``task_refusals`` passes each of the task's machines to its factory's
``check_environment``. When a factory is missing or rejects a machine, the task is not started: each
of its trials is one ``MACHINE_UNSUPPORTED`` attempt carrying the refusals.
"""

import asyncio
import base64
import copy
import dataclasses
import json
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from rigging.timing import ExponentialBackoff
from rolloutengine.contracts import ModelRequest, ModelTurn, RolloutData
from rolloutengine.engine import RolloutEngine
from rolloutengine.spec import LoweredTaskSpec, TraceMode
from shellbox.machine import MachineFactory
from taskcompendium.models import CONVERSATION_ANSWERS, TaskSpec
from taskcompendium.submission import submission_compatibility

from taskforge.content_hash import sha256_hex
from taskforge.ledger.records import EntryKind, Ledger, SpanFields, span
from taskforge.sandbox.factories import Refusal, task_refusals
from taskforge.stage50_validate.classify import trial_outcome
from taskforge.stage50_validate.outcome import Cause, Graded, Outcome, TrialKind, Ungraded

type RolloutModel = Callable[[ModelRequest], Awaitable[ModelTurn]]


class TrialModel(Protocol):
    """A RolloutEngine model that each attempt binds to its ledger step, as ``GlmRolloutModel`` is."""

    def for_step(self, step: str) -> RolloutModel: ...


ATTEMPT_PREFIX = "attempt-"
"""Each attempt file is ``<evidence_dir>/<kind>/<trial>/attempt-<n>.json``."""
CLEANUP_ERROR_COUNT = "cleanup_error_count"
"""The ``RolloutData.metrics`` key where RolloutEngine counts failed cleanup actions."""


@dataclass(frozen=True)
class EngineSettings:
    """The machine factories ``RolloutEngine`` runs on, and the session trace and limits validation
    imposes on every trial.

    ``factories`` is keyed by shellbox ``Backend`` value, as ``machine_factories`` returns it.
    ``command_timeout`` bounds one shell command and ``tool_turn_timeout`` one tool turn, so the first
    must be shorter. ``model_turn_timeout`` bounds one model call. ``cleanup_timeout`` bounds each
    cleanup action (closing a session or machine); the engine records a cleanup that fails or overruns
    in ``grade.diagnostics``.
    """

    factories: Mapping[str, MachineFactory]
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
        """``lowered`` under the messages trace, with these session limits in place of the builder's."""
        session = lowered.session.model_copy(
            update={
                "trace": TraceMode.MESSAGES,
                "max_turns": self.max_turns,
                "command_timeout": self.command_timeout,
                "tool_turn_timeout": self.tool_turn_timeout,
                "model_turn_timeout": self.model_turn_timeout,
                "cleanup_timeout": self.cleanup_timeout,
            }
        )
        return lowered.model_copy(update={"session": session})

    def engine(self, model: RolloutModel) -> RolloutEngine:
        return RolloutEngine(model, self.factories)


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
    """Which workpiece the trials belong to, how many run, their deadlines, how they retry, and where they
    are recorded.

    ``max_retries`` bounds retries of ``RETRYABLE`` causes. ``retry_backoff`` is a template: each
    trial waits on its own copy.
    ``first_attempt`` numbers each trial's first attempt file and ledger step; later attempts follow it.
    """

    workpiece_id: str
    round: int
    kind: TrialKind
    k: int
    deadlines: Deadlines
    max_retries: int
    retry_backoff: ExponentialBackoff
    evidence_dir: Path
    ledger: Ledger
    first_attempt: int

    def __post_init__(self) -> None:
        if self.k < 1 or self.max_retries < 0 or self.first_attempt < 0:
            raise ValueError("A trial plan needs k >= 1 and non-negative max retries and first attempt")


async def run_trials(
    lowered: LoweredTaskSpec, plan: TrialPlan, settings: EngineSettings, model: TrialModel
) -> list[Outcome]:
    """Run ``plan.k`` trials of ``lowered`` under ``plan.deadlines`` and ``settings`` concurrently;
    return one final outcome per trial, in order."""
    async with asyncio.TaskGroup() as group:
        trials = [group.create_task(run_trial(lowered, plan, settings, model, str(index))) for index in range(plan.k)]
    return [trial.result() for trial in trials]


async def run_trial(
    lowered: LoweredTaskSpec, plan: TrialPlan, settings: EngineSettings, model: TrialModel, trial: str
) -> Outcome:
    """Run one trial of ``lowered`` under ``plan.deadlines`` and ``settings``, attempting it again
    after a backoff while it fails for a retryable cause, up to ``plan.max_retries`` times."""
    lowered = settings.apply(plan.deadlines.apply(lowered))
    unsupported = submission_refusal(lowered.task)
    if unsupported is not None:
        return _refuse(lowered, plan, trial, Ungraded(Cause.SUBMISSION_UNSUPPORTED, unsupported, None))
    refusals = task_refusals(lowered, settings.factories)
    if refusals:
        return _refuse(lowered, plan, trial, Ungraded(Cause.MACHINE_UNSUPPORTED, refusal_detail(refusals), None))
    backoff = copy.copy(plan.retry_backoff)
    retries = 0
    attempt = plan.first_attempt
    while True:
        engine = settings.engine(model.for_step(attempt_step(plan, trial, attempt)))
        outcome = await _attempt(engine, lowered, plan, trial, attempt)
        if isinstance(outcome, Graded):
            return outcome
        if not outcome.retryable or retries >= plan.max_retries:
            return outcome
        retries += 1
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
    with attempt_span(lowered, plan, trial, plan.first_attempt) as fields:
        record_attempt(fields, outcome, plan, trial, plan.first_attempt, outcome_json(outcome))
    return outcome


def attempt_step(plan: TrialPlan, trial: str, attempt: int) -> str:
    """The ledger step of one attempt, ``<kind>/<trial>/<attempt>``."""
    return f"{plan.kind}/{trial}/{attempt}"


@contextmanager
def attempt_span(lowered: LoweredTaskSpec, plan: TrialPlan, trial: str, attempt: int) -> Iterator[SpanFields]:
    """The ``TRIAL`` ledger span of one attempt: step ``<kind>/<trial>/<attempt>``, ``input_hash`` the task digest."""
    with span(
        plan.ledger,
        EntryKind.TRIAL,
        workpiece_id=plan.workpiece_id,
        round=plan.round,
        step=attempt_step(plan, trial, attempt),
        input_hash=task_digest(lowered),
    ) as fields:
        yield fields


async def _attempt(
    engine: RolloutEngine, lowered: LoweredTaskSpec, plan: TrialPlan, trial: str, attempt: int
) -> Outcome:
    with attempt_span(lowered, plan, trial, attempt) as fields:
        try:
            result: RolloutData | Exception = await engine.run(lowered)
        except Exception as error:
            result = error
        outcome = trial_outcome(result, lowered.task)
        record_attempt(fields, outcome, plan, trial, attempt, outcome_json(outcome))
    return outcome


def record_attempt(
    fields: SpanFields, outcome: Outcome, plan: TrialPlan, trial: str, attempt: int, payload: bytes
) -> None:
    """Fill the attempt's span from ``outcome`` and write ``payload`` as its attempt file.

    ``payload`` is ``outcome_json(outcome)``, or a record that extends it (an adversary attempt's).
    """
    fields.attrs = _attributes(outcome)
    rollout = outcome.rollout
    if rollout is not None and rollout.steps:
        fields.tokens_in = sum(len(step.turn.prompt_token_ids) for step in rollout.steps)
        fields.tokens_out = sum(len(step.turn.response_token_ids) for step in rollout.steps)
        fields.finish_reason = rollout.stop_reason
    if isinstance(outcome, Ungraded):
        fields.cause = str(outcome.cause)
    fields.output_hash = sha256_hex(payload)
    write_evidence(plan.evidence_dir / str(plan.kind) / trial / f"{ATTEMPT_PREFIX}{attempt}.json", payload)


def cleanup_errors(outcome: Outcome) -> int | None:
    """How many cleanup actions of the attempt failed or overran; ``None`` without a rollout."""
    if outcome.rollout is None:
        return None
    return int(outcome.rollout.metrics.get(CLEANUP_ERROR_COUNT, 0))


def _attributes(outcome: Outcome) -> dict[str, str]:
    cleanup = {"cleanup_errors": str(cleanup_errors(outcome))}
    if isinstance(outcome, Graded):
        code_error = {"candidate_code_error": "true"} if outcome.candidate_code_error else {}
        return {
            "outcome": "graded",
            "status": str(outcome.grade.status),
            "reward": str(outcome.reward),
            "timed_out": str(outcome.timed_out),
            **code_error,
            **cleanup,
        }
    return {"outcome": "ungraded", "retryable": str(outcome.retryable), **cleanup}


def outcome_json(outcome: Outcome) -> bytes:
    """The evidence record of one attempt: the outcome and its whole ``RolloutData``."""
    record: dict[str, Any] = (
        {
            "outcome": "graded",
            "reward": outcome.reward,
            "timed_out": outcome.timed_out,
            "candidate_code_error": outcome.candidate_code_error,
        }
        if isinstance(outcome, Graded)
        else {
            "outcome": "ungraded",
            "cause": outcome.cause,
            "retryable": outcome.retryable,
            "detail": outcome.detail,
        }
    )
    record["cleanup_errors"] = cleanup_errors(outcome)
    record["rollout"] = None if outcome.rollout is None else _rollout_record(outcome.rollout)
    return json.dumps(record, indent=1).encode()


def _rollout_record(rollout: RolloutData) -> dict[str, Any]:
    """``rollout`` as JSON: the captured output files, bytes keyed by path, are base64 strings."""
    record = dataclasses.asdict(rollout)
    record["outputs"] = {path: base64.b64encode(content).decode() for path, content in rollout.outputs.items()}
    return record


def write_evidence(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
