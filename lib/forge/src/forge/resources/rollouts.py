# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Solver attempts through RolloutEngine, with explicit restart decisions."""

from dataclasses import dataclass
from typing import ClassVar, Protocol

from rolloutengine.models import GradedRollout, Model, RolloutInterrupted, RolloutOperation, TaskRolloutSpec
from shellbox.machine import ExecSetupFailed, MachineTerminated

from forge.effects.events import Event, Fingerprint, ReplayPolicy
from forge.effects.issues import Restart, Transient
from forge.resources.execution import Notify, has_cause, retry_effect


@dataclass(frozen=True, slots=True)
class SolverRollout(Event[GradedRollout]):
    result_type: ClassVar[object] = GradedRollout
    spec: TaskRolloutSpec
    trial: int
    attempt: int


class Engine(Protocol):
    async def run_task(self, spec: TaskRolloutSpec, model: Model, *, run_id: str | None = None) -> GradedRollout: ...


def interruption(error: RolloutInterrupted) -> Transient | Restart:
    """Retry startup and confirmed pre-command websocket setup failures."""
    causes: list[str] = []
    seen = {id(error)}
    cause = error.__cause__
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        causes.append(str(cause))
        cause = cause.__cause__
    detail = ": " + " <- ".join(causes)[:300] if causes else ""
    message = f"Rollout interrupted during {error.operation}{detail}"
    if has_cause(error, MachineTerminated):
        return Restart(message)
    retryable = error.operation == RolloutOperation.START or has_cause(error, ExecSetupFailed)
    return Transient(message) if retryable else Restart(message)


@dataclass(frozen=True, slots=True)
class RolloutResource:
    engine: Engine
    model: Model
    config_digest: str

    def fingerprint(self) -> Fingerprint:
        return Fingerprint("rollouts", self.config_digest, ReplayPolicy.CACHE)

    async def run(self, event: SolverRollout) -> GradedRollout:
        try:
            result = await self.engine.run_task(
                event.spec, self.model, run_id=f"{event.spec.task.id}:{event.trial}:{event.attempt}"
            )
            return result
        except RolloutInterrupted as error:
            raise interruption(error) from error


async def run_rollout(event: SolverRollout, resource: RolloutResource, *, notify: Notify) -> GradedRollout:
    return await retry_effect(event, lambda: resource.run(event), notify)
