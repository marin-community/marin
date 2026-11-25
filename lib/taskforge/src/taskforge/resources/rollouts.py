# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Solver attempts through RolloutEngine, with explicit restart decisions."""

import json
from dataclasses import dataclass
from typing import Any, Protocol

from rolloutengine.machines import MachineSetupError, MachineSetupTimeout
from rolloutengine.models import GradedRollout, Model, Rollout, RolloutInterrupted, RolloutOperation, TaskRolloutSpec
from shellbox.machine import (
    ExecSetupFailed,
    MachineCancelled,
    MachineOutputLimitExceeded,
    MachineTerminated,
    MachineUnavailable,
)

from taskforge.effects.events import Event, Fingerprint, ReplayPolicy
from taskforge.effects.issues import (
    EnvironmentSetupFailed,
    OutputBudgetExhausted,
    Restart,
    Transient,
    describe_exception,
)
from taskforge.resources.execution import Notify, has_cause, retry_effect


@dataclass(frozen=True, slots=True)
class SolverRollout(Event[GradedRollout]):
    spec: TaskRolloutSpec
    trial: int
    attempt: int


INTERRUPTED_CALLS = 20
"""The most recent tool calls an ``InterruptedRollout`` keeps; earlier ones are only counted."""
CALL_CHARS = 500
"""The most characters of a call's arguments, and of its result, that an ``InterruptedRollout`` keeps."""


@dataclass(frozen=True, slots=True)
class ToolCallOutcome:
    """One tool call of an interrupted attempt: its arguments and the result the model saw, truncated."""

    name: str
    arguments: str
    result: str | None
    """``None`` when the attempt ended before the call returned, as for the call that met a closed machine."""


@dataclass(frozen=True, slots=True)
class InterruptedRollout(Event[None]):
    """The tool calls and failure of a solver attempt that was interrupted, recorded before its decision.

    An interrupted attempt has no ``Resolved`` value, so this is the log's only record of what the solver ran.
    """

    run_id: str
    operation: RolloutOperation
    failure: str
    calls: tuple[ToolCallOutcome, ...]
    omitted_calls: int


def _truncated(text: str) -> str:
    return text if len(text) <= CALL_CHARS else f"{text[:CALL_CHARS]}... [{len(text) - CALL_CHARS} more characters]"


def _tool_calls(rollout: Rollout) -> list[ToolCallOutcome]:
    """Pair every assistant tool call of ``rollout`` with its tool result, in order."""
    calls: list[ToolCallOutcome] = []
    for conversation in rollout.conversations:
        messages: tuple[dict[str, Any], ...] = conversation.messages
        results = {message["tool_call_id"]: message["content"] for message in messages if message["role"] == "tool"}
        for message in messages:
            for call in message.get("tool_calls") or ():
                function = call["function"]
                arguments = function["arguments"]
                result = results.get(call["id"])
                calls.append(
                    ToolCallOutcome(
                        function["name"],
                        _truncated(arguments if isinstance(arguments, str) else json.dumps(arguments)),
                        None if result is None else _truncated(str(result)),
                    )
                )
    return calls


def interrupted_rollout(error: RolloutInterrupted, decision: Exception) -> InterruptedRollout:
    """The bounded record of ``error``'s attempt: its last ``INTERRUPTED_CALLS`` tool calls and ``decision``."""
    calls = _tool_calls(error.rollout)
    kept = calls[-INTERRUPTED_CALLS:]
    return InterruptedRollout(
        error.rollout.run_id, error.operation, describe_exception(decision), tuple(kept), len(calls) - len(kept)
    )


class Engine(Protocol):
    async def run_task(self, spec: TaskRolloutSpec, model: Model, *, run_id: str | None = None) -> GradedRollout: ...


def interruption(error: RolloutInterrupted) -> Transient | Restart | EnvironmentSetupFailed:
    """Retry startup and confirmed pre-command websocket setup failures; a failing setup command is the task's."""
    causes: list[str] = []
    seen = {id(error)}
    cause = error.__cause__
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        causes.append(describe_exception(cause))
        cause = cause.__cause__
    detail = ": " + " <- ".join(causes)[:300] if causes else ""
    message = f"Rollout interrupted during {error.operation}{detail}"
    if has_cause(error, MachineSetupError) and not has_cause(error, MachineSetupTimeout):
        return EnvironmentSetupFailed(message)
    if error.operation is RolloutOperation.ADVANCE and has_cause(error, MachineOutputLimitExceeded):
        return OutputBudgetExhausted("output budget exhausted")
    if has_cause(error, MachineCancelled) or has_cause(error, MachineUnavailable):
        return Transient(message)
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

    async def run(self, event: SolverRollout, notify: Notify) -> GradedRollout:
        """Run one attempt; an interrupted one is recorded through ``notify`` before its decision is raised."""
        try:
            return await self.engine.run_task(
                event.spec, self.model, run_id=f"{event.spec.task.id}:{event.trial}:{event.attempt}"
            )
        except RolloutInterrupted as error:
            decision = interruption(error)
            notify(interrupted_rollout(error, decision))
            raise decision from error


async def run_rollout(event: SolverRollout, resource: RolloutResource, *, notify: Notify) -> GradedRollout:
    return await retry_effect(event, lambda: resource.run(event, notify), notify)
