# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Solver attempts through RolloutEngine, with explicit restart decisions."""

from dataclasses import dataclass
from typing import Protocol

from rolloutengine.models import GradedRollout, Model, RolloutInterrupted, RolloutOperation, TaskRolloutSpec
from shellbox.machine import (
    ExecSetupFailed,
    MachineCancelled,
    MachineOutputLimitExceeded,
    MachineTerminated,
    MachineUnavailable,
)

from taskforge.effects.events import Event, Fingerprint, ReplayPolicy
from taskforge.effects.issues import Hold, OutputBudgetExhausted, Restart, Transient, describe_exception
from taskforge.llm.client import ModelRequestRejected
from taskforge.resources.execution import Notify, retry_effect


@dataclass(frozen=True, slots=True)
class SolverRollout(Event[GradedRollout]):
    kind = "rollouts.solver_rollout"
    spec: TaskRolloutSpec
    trial: int
    attempt: int


class Engine(Protocol):
    async def run_task(self, spec: TaskRolloutSpec, model: Model, *, run_id: str | None = None) -> GradedRollout: ...


def interruption(error: RolloutInterrupted) -> Transient | Restart | Hold:
    """Retry startup and confirmed pre-command websocket setup failures; hold on a request the model server rejects."""
    chain: list[BaseException] = [error]
    while (cause := chain[-1].__cause__) is not None and all(cause is not seen for seen in chain):
        chain.append(cause)
    detail = ": " + " <- ".join(map(describe_exception, chain[1:]))[:300] if chain[1:] else ""
    message = f"Rollout interrupted during {error.operation}{detail}"

    def caused(*kinds: type[BaseException]) -> bool:
        return any(isinstance(cause, kinds) for cause in chain)

    if rejected := next((cause for cause in chain if isinstance(cause, ModelRequestRejected)), None):
        return Hold(str(rejected))  # the same request would be rejected again, and is no evidence about the model
    if error.operation is RolloutOperation.ADVANCE and caused(MachineOutputLimitExceeded):
        return OutputBudgetExhausted("output budget exhausted")
    if caused(MachineCancelled, MachineUnavailable):
        return Transient(message)
    if caused(MachineTerminated):
        return Restart(message)
    retryable = error.operation == RolloutOperation.START or caused(ExecSetupFailed)
    return Transient(message) if retryable else Restart(message)


STARTUP = frozenset({RolloutOperation.START, RolloutOperation.PREPARE})
"""Where a rollout fails before its first model turn; only there does its executor retry it."""


@dataclass(frozen=True, slots=True)
class RolloutResource:
    """``startup_attempts`` bounds the attempts of a rollout that keeps failing before its first model turn; it is
    scheduling, so it stays out of the fingerprint."""

    engine: Engine
    model: Model
    config_digest: str
    startup_attempts: int

    def fingerprint(self) -> Fingerprint:
        return Fingerprint("rollouts", self.config_digest, ReplayPolicy.CACHE)

    async def run(self, event: SolverRollout) -> GradedRollout:
        try:
            return await self.engine.run_task(
                event.spec, self.model, run_id=f"{event.spec.task.id}:{event.trial}:{event.attempt}"
            )
        except RolloutInterrupted as error:
            issue = interruption(error)
            if isinstance(issue, Transient) and error.operation not in STARTUP:
                issue = Restart(str(issue))  # the model has acted: running it again is a new trial, not a retry
            raise issue from error


async def run_rollout(event: SolverRollout, resource: RolloutResource, *, notify: Notify) -> GradedRollout:
    return await retry_effect(event, lambda: resource.run(event), notify, resource.startup_attempts)
