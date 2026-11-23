# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Solver attempts through RolloutEngine, with explicit restart decisions."""

from dataclasses import dataclass, replace
from typing import Protocol

from rolloutengine.models import (
    GradedRollout,
    Model,
    RolloutFailure,
    RolloutInterrupted,
    RolloutOperation,
    TaskRolloutSpec,
)
from shellbox.machine import (
    ExecSetupFailed,
    MachineCancelled,
    MachineOutputLimitExceeded,
    MachineTerminated,
    MachineUnavailable,
)
from taskcompendium.grading_result import GradeResult, Outcome

from taskforge.effects.events import Event, Fingerprint, ReplayPolicy
from taskforge.effects.issues import (
    DecisionModelUnavailable,
    Hold,
    OutputBudgetExhausted,
    RequestRejected,
    Restart,
    Transient,
    describe_exception,
)
from taskforge.llm.client import ModelRequestRejected
from taskforge.resources.execution import Notify, retry_effect


@dataclass(frozen=True, slots=True)
class SolverRollout(Event[GradedRollout]):
    kind = "rollouts.solver_rollout"
    spec: TaskRolloutSpec
    trial: int
    attempt: int


SLOT_LOCAL: tuple[type[Hold], ...] = (RequestRejected,)
"""The holds that mean only "this slot produced no evidence": the batch records them as the slot's result. Every
other hold (``Unreproducible``, host refusals, anything unknown) holds the workpiece. ``DecisionModelUnavailable`` is
deliberately absent: a recorded result would be missing evidence, while an outage defers the slot unrecorded."""


def refused(value: object) -> bool:
    """A recorded slot-local hold: the engine sets ``failure`` only on an interrupted attempt, which a returned
    result never is, so no model output can produce it."""
    return isinstance(value, GradedRollout) and value.rollout.failure is not None


class Engine(Protocol):
    async def run_task(self, spec: TaskRolloutSpec, model: Model, *, run_id: str | None = None) -> GradedRollout: ...


def interruption(error: RolloutInterrupted) -> Transient | Restart | Hold:
    """Retry startup and confirmed pre-command websocket setup failures; hold on a request the model server rejects
    and on a decision-model outage."""
    chain: list[BaseException] = [error]
    while (cause := chain[-1].__cause__) is not None and all(cause is not seen for seen in chain):
        chain.append(cause)
    detail = ": " + " <- ".join(map(describe_exception, chain[1:]))[:300] if chain[1:] else ""
    message = f"Rollout interrupted during {error.operation}{detail}"

    def caused(*kinds: type[BaseException]) -> bool:
        return any(isinstance(cause, kinds) for cause in chain)

    if outage := next((cause for cause in chain if isinstance(cause, DecisionModelUnavailable)), None):
        return DecisionModelUnavailable(str(outage))  # a fresh hold: chaining the original would make a cycle
    if rejected := next((cause for cause in chain if isinstance(cause, ModelRequestRejected)), None):
        return RequestRejected(str(rejected))  # no evidence about the model
    if error.operation is RolloutOperation.ADVANCE and caused(MachineOutputLimitExceeded):
        return OutputBudgetExhausted("output budget exhausted")
    if caused(MachineCancelled, MachineUnavailable):
        return Transient(message)
    if caused(MachineTerminated):
        return Restart(message)
    retryable = error.operation == RolloutOperation.START or caused(ExecSetupFailed)
    return Transient(message) if retryable else Restart(message)


@dataclass(frozen=True, slots=True)
class RolloutResource:
    engine: Engine
    model: Model
    config_digest: str

    def fingerprint(self) -> Fingerprint:
        return Fingerprint("rollouts", self.config_digest, ReplayPolicy.CACHE)

    async def run(self, event: SolverRollout) -> GradedRollout:
        run_id = f"{event.spec.task.id}:{event.trial}:{event.attempt}"
        try:
            return await self.engine.run_task(event.spec, self.model, run_id=run_id)
        except RolloutInterrupted as error:
            issue = interruption(error)
            if isinstance(issue, SLOT_LOCAL):
                grade = GradeResult(Outcome.UNAVAILABLE, None, describe_exception(issue))
                failure = error.rollout.failure or RolloutFailure(type(issue).__name__)
                return GradedRollout(replace(error.rollout, failure=failure), grade)  # keeps the paid prefix
            raise issue from error


async def run_rollout(event: SolverRollout, resource: RolloutResource, *, notify: Notify) -> GradedRollout:
    return await retry_effect(event, lambda: resource.run(event), notify)
