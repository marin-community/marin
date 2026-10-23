# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run k independent trials of a task through RolloutEngine, classifying and retrying failures.

Every attempt of every trial is one ``TRIAL`` ledger span (``step`` is ``<kind>/<trial>/<attempt>``,
``cause`` is the ``Cause`` value of an ungraded attempt) and one JSON file under
``<evidence_dir>/<kind>/<trial>/attempt-<n>.json`` holding the outcome, its grade and the full
``Rollout`` (or the traceback when the engine kept no rollout). Both carry ``cleanup_errors``, the number of
cleanup actions that failed or overran (RolloutEngine's ``cleanup_error_count``), so a leaked
machine shows in summaries; it is ``None`` when the engine kept no rollout.

A trial is attempted again while its outcome is ``Ungraded`` with a retryable cause, up to
``max_retries`` times, waiting ``plan.retry_backoff`` between attempts. Attempts are numbered from
``TrialPlan.first_attempt``: a trial re-entered after earlier attempts passes the count already on
disk, so no attempt file is overwritten and each ledger step stays unique. Each attempt runs on
its own model, ``model.for_step(<kind>/<trial>/<attempt>)``: a ``GlmRolloutModel`` records its
``LLM_CALL`` spans under the attempt's step and routes on it, so the k trials of a workpiece do not
share one prefix-cache key.

Every trial runs under the engine's messages trace (``TaskRolloutSpec.trace = MESSAGES``, set by
``lowering.lower``): a trial is graded and never trained on, so the engine records the token
ids a model reports without checking them, with an all-zero loss mask, and a model's next prompt
need not extend the ids of its last turn. No trial has a token budget (``RolloutLimits.max_total_tokens``
is ``None``). The attempt span's ``tokens_in`` and ``tokens_out`` sum the ``usage`` of the turns of
every conversation of the attempt; a scripted turn reports none and counts nothing.

A graded attempt's record says whether the candidate's own code did not load
(``candidate_code_error``); the span carries ``candidate_code_error="true"`` when it did.

Validation supplies ``EngineSettings.limits`` when lowering the task. The lowered spec is the
single source of effective limits for probes, solver trials and controls. The engine's shell harness
uses ``command_timeout``. ``task_digest`` covers the task, machine selections, effective limits and
command timeout so changed validation settings do not reuse earlier evidence.

The attempt file holds the whole ``Rollout``, every conversation and its steps included: a validation
round is reviewed from its files alone, and the final reply, the tool calls and the per-turn usage are
read from the steps. ``attempt_outcome`` reads the outcome back from the file. The file is written whole
or not at all (``write_evidence``), and an attempt cancelled before its outcome writes none.

The answer format is part of the task (``TaskSpec.answer_format``), so ``input_hash`` covers it
with the rest of the task. A task whose format cannot carry its answer for its grader
(TaskCompendium ``submission_compatibility``) is not started: each of its trials is one
``SUBMISSION_UNSUPPORTED`` attempt naming the reasons.

``EngineSettings.factories`` come from ``taskforge.sandbox.factories.machine_factories``. Before a
trial starts, ``task_refusals`` passes each of the task's machines to its factory's
``check_environment``. When a factory is missing or rejects a machine, the task is not started: each
of its trials is one ``MACHINE_UNSUPPORTED`` attempt carrying the refusals.
"""

import asyncio
import base64
import dataclasses
import json
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import JsonValue, TypeAdapter
from rigging.filesystem.atomic import write_bytes_atomic
from rigging.timing import ExponentialBackoff
from rolloutengine.engine import RolloutEngine
from rolloutengine.models import GradedRollout, Model, Rollout, RolloutLimits, RolloutStep, TaskRolloutSpec
from shellbox.machine import MachineFactory
from taskcompendium.grading_result import GradeResult
from taskcompendium.models import CONVERSATION_ANSWERS, TaskSpec
from taskcompendium.submission import submission_compatibility

from taskforge.content_hash import sha256_hex
from taskforge.ledger.records import EntryKind, Ledger, SpanFields, span
from taskforge.sandbox.factories import Refusal, task_refusals
from taskforge.spec import draft
from taskforge.stage50_validate.classify import trial_outcome
from taskforge.stage50_validate.outcome import Cause, Graded, Outcome, TrialKind, Ungraded


class TrialModel(Protocol):
    """A RolloutEngine model that each attempt binds to its ledger step, as ``GlmRolloutModel`` is."""

    def for_step(self, step: str) -> Model: ...


ATTEMPT_PREFIX = "attempt-"
"""Each attempt file is ``<evidence_dir>/<kind>/<trial>/attempt-<n>.json``."""
CLEANUP_ERROR_COUNT = "cleanup_error_count"
"""The ``Rollout.metrics`` key where RolloutEngine counts failed cleanup actions."""


@dataclass(frozen=True)
class EngineSettings:
    """Factories, shell-command timeout and the limits supplied when lowering validation tasks."""

    factories: Mapping[str, MachineFactory]
    command_timeout: float
    limits: RolloutLimits

    def __post_init__(self) -> None:
        tool_timeout = self.limits.tool_turn_timeout
        if self.command_timeout <= 0 or (tool_timeout is not None and self.command_timeout >= tool_timeout):
            raise ValueError("The command timeout must be positive and shorter than the tool-turn timeout")

    def engine(self) -> RolloutEngine:
        """The engine every trial runs on, with the configured shell-command timeout."""
        return draft.rollout_engine(self.factories, command_timeout=self.command_timeout)


def submission_refusal(task: TaskSpec) -> str | None:
    """Why the task's answer format cannot carry its answer, or ``None`` when it can.

    Only an answer read from the conversation has a format to check, as in RolloutEngine's shell harness.
    """
    if task.answer_type not in CONVERSATION_ANSWERS:
        return None
    compatibility = submission_compatibility(task)
    return None if compatibility.compatible else "; ".join(compatibility.reasons)


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
    """Which workpiece the trials belong to, how many run, how they retry, and where they
    are recorded.

    ``max_retries`` bounds retries of ``RETRYABLE`` causes. Each trial starts a fresh schedule from ``retry_backoff``.
    ``first_attempt`` numbers each trial's first attempt file and ledger step; later attempts follow it.
    """

    workpiece_id: str
    round: int
    kind: TrialKind
    k: int
    max_retries: int
    retry_backoff: RetryBackoff
    evidence_dir: Path
    ledger: Ledger
    first_attempt: int

    def __post_init__(self) -> None:
        if self.k < 1 or self.max_retries < 0 or self.first_attempt < 0:
            raise ValueError("A trial plan needs k >= 1 and non-negative max retries and first attempt")


async def run_trials(
    spec: TaskRolloutSpec, plan: TrialPlan, settings: EngineSettings, model: TrialModel
) -> list[Outcome]:
    """Run ``plan.k`` trials of ``spec`` under the lowered spec's limits and ``settings`` concurrently;
    return one final outcome per trial, in order."""
    async with asyncio.TaskGroup() as group:
        trials = [group.create_task(run_trial(spec, plan, settings, model, str(index))) for index in range(plan.k)]
    return [trial.result() for trial in trials]


async def run_trial(
    spec: TaskRolloutSpec, plan: TrialPlan, settings: EngineSettings, model: TrialModel, trial: str
) -> Outcome:
    """Run one trial of ``spec`` under the lowered spec's limits and ``settings``, attempting it again
    after a backoff while it fails for a retryable cause, up to ``plan.max_retries`` times."""
    digest = task_digest(spec, settings.command_timeout)
    refused = refusal(spec, settings.factories)
    if refused is not None:
        return _refuse(digest, plan, trial, refused)
    engine = settings.engine()

    async def attempt(number: int, step: str) -> Outcome:
        return await _attempt(engine, spec, model.for_step(step), digest, plan, trial, number)

    return await run_attempts(plan, trial, attempt)


async def run_attempts(plan: TrialPlan, trial: str, attempt: Callable[[int, str], Awaitable[Outcome]]) -> Outcome:
    """Run and record attempts until graded, permanently failed, or out of retries."""
    backoff = plan.retry_backoff.schedule()
    for retry in range(plan.max_retries + 1):
        number = plan.first_attempt + retry
        outcome = await attempt(number, attempt_step(plan, trial, number))
        if isinstance(outcome, Graded) or not outcome.retryable or retry == plan.max_retries:
            return outcome
        await asyncio.sleep(backoff.next_interval())
    raise AssertionError("a nonnegative retry budget always runs an attempt")


def refusal(spec: TaskRolloutSpec, factories: Mapping[str, MachineFactory]) -> Ungraded | None:
    """Why a trial cannot start on these factories, or ``None`` when it can."""
    unsupported = submission_refusal(spec.task)
    if unsupported is not None:
        return Ungraded(Cause.SUBMISSION_UNSUPPORTED, unsupported, None, None)
    refusals = task_refusals(spec, factories)
    if refusals:
        return Ungraded(Cause.MACHINE_UNSUPPORTED, refusal_detail(refusals), None, None)
    return None


def refusal_detail(refusals: Sequence[Refusal]) -> str:
    return "; ".join(f"{refusal.where}: {refusal.reason}: {refusal.detail}" for refusal in refusals)


def trial_record(spec: TaskRolloutSpec, command_timeout: float) -> dict[str, JsonValue]:
    """What a trial of ``spec`` runs, as Taskforge records it: the input of ``task_digest``.

    ``spec`` is the spec as it runs (after lowering), and
    ``command_timeout`` the shell harness's. The record is a JSON object of exactly these keys:

    - ``task``: the ``TaskSpec``, its answer format included;
    - ``task_machine`` and ``verifier_machine``: each machine selection's ``MachineRuntimeSpec``, or ``null``;
    - ``limits``: the effective ``RolloutLimits``;
    - ``command_timeout``: the timeout of one shell command.

    Each value is its model's JSON dump. The harness name, which is always ``draft.TOOL_HARNESS``, and the
    trace, which is always ``MESSAGES``, are not in it, so the digest does not follow the shape of the
    engine's ``TaskRolloutSpec``.
    """
    task_machine, verifier_machine = spec.task_machine, spec.verifier_machine
    return {
        "task": spec.task.model_dump(mode="json"),
        "task_machine": None if task_machine is None else task_machine.model_dump(mode="json"),
        "verifier_machine": None if verifier_machine is None else verifier_machine.model_dump(mode="json"),
        "limits": spec.limits.model_dump(mode="json"),
        "command_timeout": command_timeout,
    }


def task_digest(spec: TaskRolloutSpec, command_timeout: float) -> str:
    """The sha256 of ``trial_record(spec, command_timeout)`` as JSON with sorted keys."""
    return sha256_hex(json.dumps(trial_record(spec, command_timeout), sort_keys=True).encode())


def _refuse(digest: str, plan: TrialPlan, trial: str, outcome: Ungraded) -> Ungraded:
    """Record ``outcome`` as the only attempt of a trial that is not started."""
    with attempt_span(digest, plan, trial, plan.first_attempt) as fields:
        record_attempt(fields, outcome, plan, trial, plan.first_attempt, outcome_json(outcome))
    return outcome


def attempt_step(plan: TrialPlan, trial: str, attempt: int) -> str:
    """The ledger step of one attempt, ``<kind>/<trial>/<attempt>``."""
    return f"{plan.kind}/{trial}/{attempt}"


@contextmanager
def attempt_span(digest: str, plan: TrialPlan, trial: str, attempt: int) -> Iterator[SpanFields]:
    """The ``TRIAL`` ledger span of one attempt: step ``<kind>/<trial>/<attempt>``, ``input_hash`` the
    trial's ``task_digest``."""
    with span(
        plan.ledger,
        EntryKind.TRIAL,
        workpiece_id=plan.workpiece_id,
        round=plan.round,
        step=attempt_step(plan, trial, attempt),
        input_hash=digest,
    ) as fields:
        yield fields


async def _attempt(
    engine: RolloutEngine, spec: TaskRolloutSpec, model: Model, digest: str, plan: TrialPlan, trial: str, attempt: int
) -> Outcome:
    with attempt_span(digest, plan, trial, attempt) as fields:
        try:
            result: GradedRollout | Exception = await engine.run_task(spec, model)
        except Exception as error:
            result = error
        outcome = trial_outcome(result, spec.task)
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
    steps = () if rollout is None else rollout_steps(rollout)
    if rollout is not None and steps:
        usages = [step.turn.usage for step in steps if step.turn.usage is not None]
        fields.tokens_in = sum(usage.prompt_tokens for usage in usages)
        fields.tokens_out = sum(usage.completion_tokens for usage in usages)
        fields.finish_reason = rollout.stop_reason
    if isinstance(outcome, Ungraded):
        fields.cause = str(outcome.cause)
    fields.output_hash = sha256_hex(payload)
    write_evidence(plan.evidence_dir / str(plan.kind) / trial / f"{ATTEMPT_PREFIX}{attempt}.json", payload)


def rollout_steps(rollout: Rollout) -> tuple[RolloutStep, ...]:
    """The steps of every conversation of ``rollout``, in order."""
    return tuple(step for conversation in rollout.conversations for step in conversation.steps)


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
    """The evidence record of one attempt: the outcome, its grade and its whole ``Rollout``."""
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
    record["grade"] = None if outcome.grade is None else dataclasses.asdict(outcome.grade)
    record["rollout"] = None if outcome.rollout is None else _rollout_record(outcome.rollout)
    return json.dumps(record, indent=1).encode()


def _rollout_record(rollout: Rollout) -> dict[str, Any]:
    """``rollout`` as JSON: the captured output files, bytes keyed by path, are base64 strings."""
    record = dataclasses.asdict(rollout)
    record["outputs"] = {path: base64.b64encode(content).decode() for path, content in rollout.outputs.items()}
    return record


def attempt_outcome(payload: bytes) -> Outcome:
    """The outcome an attempt file holds: the inverse of ``outcome_json``."""
    record = json.loads(payload)
    grade = None if record["grade"] is None else _GRADE.validate_python(record["grade"])
    rollout = None if record["rollout"] is None else _rollout_from_record(record["rollout"])
    if record["outcome"] == "graded":
        assert rollout is not None and grade is not None, "a graded attempt records its rollout and grade"
        return Graded(rollout, grade, record["candidate_code_error"])
    return Ungraded(Cause(record["cause"]), record["detail"], rollout, grade)


def _rollout_from_record(record: dict[str, Any]) -> Rollout:
    outputs = {path: base64.b64decode(content) for path, content in record["outputs"].items()}
    return _ROLLOUT.validate_python({**record, "outputs": outputs})


_GRADE: TypeAdapter[GradeResult] = TypeAdapter(GradeResult)
_ROLLOUT: TypeAdapter[Rollout] = TypeAdapter(Rollout)


def write_evidence(path: Path, payload: bytes) -> None:
    """Write ``payload`` to ``path`` through a temporary sibling, so the file is whole or absent."""
    write_bytes_atomic(str(path), payload)
