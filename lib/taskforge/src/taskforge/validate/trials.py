# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run k independent trials of a task through RolloutEngine, classifying and retrying failures.

Every attempt of every trial is one ``TRIAL`` ledger span (``step`` is ``<kind>/<trial>/<attempt>``,
``cause`` is the ``Cause`` value of an ungraded attempt) and one JSON file under
``<evidence_dir>/<kind>/<trial>/attempt-<n>.json`` holding the outcome and the full ``RolloutData``
(or the traceback when the engine kept no rollout). A trial is attempted again while its outcome is
``Ungraded`` with a retryable cause, up to ``max_retries`` times, waiting ``plan.retry_backoff``
between attempts.

``EngineSettings.factories`` and ``EngineSettings.capabilities`` come from
``taskforge.sandbox.factories.machine_factories`` and ``factory_capabilities`` for the same host. A
task those factories cannot run (``task_refusals``) is not started: each of its trials is one
``MACHINE_UNSUPPORTED`` attempt naming the refusals.
"""

import asyncio
import copy
import dataclasses
import hashlib
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
from taskcompendium.submission import SubmissionConvention

from taskforge.ledger.records import EntryKind, Ledger, SpanFields, span
from taskforge.sandbox.factories import FactoryCapabilities, Refusal, task_refusals
from taskforge.validate.classify import trial_outcome
from taskforge.validate.outcome import Cause, Graded, Outcome, TrialKind, Ungraded

type RolloutModel = Callable[[ModelRequest], Awaitable[ModelTurn]]


@dataclass(frozen=True)
class EngineSettings:
    """Everything ``ShellboxRolloutEngine`` takes except the model, and what the factories can run.

    ``cleanup_timeout`` bounds each cleanup action (closing a session or machine, removing a stage
    grader); the engine records a cleanup that fails or overruns in ``grade.diagnostics``.
    """

    factories: Mapping[EnvironmentKind, MachineFactory]
    capabilities: Mapping[EnvironmentKind, FactoryCapabilities]
    max_turns: int
    command_timeout: float
    cleanup_timeout: float
    convention: SubmissionConvention

    def engine(self, model: RolloutModel) -> ShellboxRolloutEngine:
        return ShellboxRolloutEngine(
            model,
            self.factories,
            max_turns=self.max_turns,
            command_timeout=self.command_timeout,
            cleanup_timeout=self.cleanup_timeout,
            convention=self.convention,
        )


@dataclass(frozen=True)
class TrialPlan:
    """Which item the trials belong to, how many run, how they retry, and where they are recorded.

    ``retry_backoff`` is a template: each trial waits on its own copy.
    """

    item_id: str
    round: int
    kind: TrialKind
    k: int
    max_retries: int
    retry_backoff: ExponentialBackoff
    evidence_dir: Path
    ledger: Ledger

    def __post_init__(self) -> None:
        if self.k < 1 or self.max_retries < 0:
            raise ValueError("A trial plan needs k >= 1 and max_retries >= 0")


async def run_trials(
    task: TaskSpec, execution: TaskExecution, plan: TrialPlan, settings: EngineSettings, model: RolloutModel
) -> list[Outcome]:
    """Run ``plan.k`` trials of ``task`` with ``execution`` concurrently; return one final outcome per
    trial, in order."""
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
    """Run one trial, attempting it again after a backoff while it fails for a retryable cause."""
    refusals = task_refusals(task, execution, settings.capabilities)
    if refusals:
        with _attempt_span(task, execution, plan, trial, 0) as fields:
            outcome = Ungraded(Cause.MACHINE_UNSUPPORTED, refusal_detail(refusals), None)
            _record(fields, outcome, plan, trial, 0)
        return outcome
    engine = settings.engine(model)
    backoff = copy.copy(plan.retry_backoff)
    attempt = 0
    while True:
        outcome = await _attempt(engine, task, execution, plan, trial, attempt)
        if isinstance(outcome, Graded) or not outcome.retryable or attempt == plan.max_retries:
            return outcome
        await asyncio.sleep(backoff.next_interval())
        attempt += 1


def refusal_detail(refusals: Sequence[Refusal]) -> str:
    return "; ".join(f"{refusal.where}: {refusal.reason}: {refusal.detail}" for refusal in refusals)


def task_digest(task: TaskSpec, execution: TaskExecution) -> str:
    """The sha256 of the task and the execution settings it runs with."""
    payload = f"{task.model_dump_json()}\n{execution.model_dump_json()}"
    return hashlib.sha256(payload.encode()).hexdigest()


@contextmanager
def _attempt_span(
    task: TaskSpec, execution: TaskExecution, plan: TrialPlan, trial: str, attempt: int
) -> Iterator[SpanFields]:
    with span(
        plan.ledger,
        EntryKind.TRIAL,
        item_id=plan.item_id,
        round=plan.round,
        step=f"{plan.kind}/{trial}/{attempt}",
        input_hash=task_digest(task, execution),
    ) as fields:
        yield fields


async def _attempt(
    engine: ShellboxRolloutEngine,
    task: TaskSpec,
    execution: TaskExecution,
    plan: TrialPlan,
    trial: str,
    attempt: int,
) -> Outcome:
    with _attempt_span(task, execution, plan, trial, attempt) as fields:
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
    fields.output_hash = hashlib.sha256(payload).hexdigest()
    write_evidence(plan.evidence_dir / str(plan.kind) / trial / f"attempt-{attempt}.json", payload)


def _attributes(outcome: Outcome) -> dict[str, str]:
    if isinstance(outcome, Graded):
        return {
            "outcome": "graded",
            "status": str(outcome.grade.status),
            "reward": str(outcome.reward),
            "timed_out": str(outcome.timed_out),
        }
    return {"outcome": "ungraded", "retryable": str(outcome.retryable)}


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
    record["rollout"] = None if outcome.rollout is None else dataclasses.asdict(outcome.rollout)
    return json.dumps(record, indent=1).encode()


def write_evidence(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
