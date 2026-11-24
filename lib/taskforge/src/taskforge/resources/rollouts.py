# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Solver attempts through RolloutEngine, with explicit restart decisions."""

from dataclasses import asdict, dataclass, field, replace
from typing import Protocol

from rolloutengine.models import (
    LENGTH_STOP_REASON,
    GenerationLimitReached,
    GradedRollout,
    GradedRolloutInterrupted,
    Model,
    ModelRequest,
    ModelTurn,
    Rollout,
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
from taskforge.llm.client import GenerationTimedOut, ModelRequestRejected
from taskforge.resources.execution import Notify, retry_effect


@dataclass(frozen=True, slots=True)
class SolverRollout(Event[GradedRollout]):
    kind = "rollouts.solver_rollout"
    spec: TaskRolloutSpec
    trial: int
    attempt: int


@dataclass(frozen=True, slots=True)
class SolverGeneration(Event[Rollout]):
    """A slot's finished generation that its decision model has not yet scored: an outage or an off-scale level ended
    its grade. Keyed by the slot's ``SolverRollout`` fields under the same fingerprint; ``generated``, outside the
    identity, is the rollout it records, so that a later execution of the slot grades it without generating again."""

    kind = "rollouts.solver_generation"
    spec: TaskRolloutSpec
    trial: int
    attempt: int
    generated: Rollout | None = field(default=None, compare=False, repr=False, metadata={"serialize": False})

    @property
    def slot(self) -> SolverRollout:
        return SolverRollout(self.spec, self.trial, self.attempt)


TIME_LIMIT_STOP_REASON = "time_limit"
"""The stop reason of an attempt whose generation was still running at ``solver.time_limit``: the engine stopped it as
at its generation limit, and ``RolloutResource`` records which limit it was, in a returned result and in an interrupted
one alike. Graders are shown the engine's ``length`` for it, as RolloutEngine's own grade
was."""


LIMIT_STOP_REASONS = frozenset({LENGTH_STOP_REASON, TIME_LIMIT_STOP_REASON, "context"})
"""The stop reasons of a generation that reached a limit: an unsolved trial whatever its grade says."""


SLOT_LOCAL: tuple[type[Hold], ...] = (RequestRejected,)
"""The holds that mean only "this slot produced no evidence": the batch records them as the slot's result. Every
other hold (host refusals, anything unknown) holds the workpiece. ``DecisionModelUnavailable`` is
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

    if error.operation is RolloutOperation.ATTEMPT:  # only the expired attempt deadline reports it
        return Restart(message)
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
        timed_out = False

        async def model(request: ModelRequest) -> ModelTurn:
            nonlocal timed_out
            try:
                return await self.model(request)
            except GenerationLimitReached as limit:
                timed_out = isinstance(limit.__cause__, GenerationTimedOut)
                raise

        def recorded(rollout: Rollout) -> Rollout:
            if timed_out and rollout.stop_reason == LENGTH_STOP_REASON:
                return replace(rollout, stop_reason=TIME_LIMIT_STOP_REASON)
            return rollout

        try:
            result = await self.engine.run_task(event.spec, model, run_id=run_id)
        except RolloutInterrupted as error:
            error.rollout = recorded(error.rollout)  # a grading interruption's rollout is checkpointed as recorded
            issue = interruption(error)
            if (
                isinstance(error, GradedRolloutInterrupted)
                and error.operation in (RolloutOperation.GRADE, RolloutOperation.ATTEMPT)
                and error.rollout.stop_reason in LIMIT_STOP_REASONS
                and not isinstance(issue, DecisionModelUnavailable)
            ):
                # A proven limit is a settled loss, whether its verifier or then the attempt cap stopped the grade: the
                # row carries no failure, so classification, replay and explicit resume all read the loss, and the
                # grade keeps what interrupted it for inspection.
                interrupted = {"interrupted": asdict(error.rollout.failure)} if error.rollout.failure else {}
                grade = replace(error.grade, diagnostics={**error.grade.diagnostics, **interrupted})
                return GradedRollout(replace(error.rollout, failure=None), grade)
            if isinstance(issue, SLOT_LOCAL):
                grade = GradeResult(Outcome.UNAVAILABLE, None, describe_exception(issue))
                failure = error.rollout.failure or RolloutFailure(type(issue).__name__)
                return GradedRollout(replace(error.rollout, failure=failure), grade)  # keeps the paid prefix
            raise issue from error
        return replace(result, rollout=recorded(result.rollout))


async def run_rollout(event: SolverRollout, resource: RolloutResource, *, notify: Notify) -> GradedRollout:
    return await retry_effect(event, lambda: resource.run(event), notify)


async def checkpoint(event: SolverGeneration, _resource: RolloutResource) -> Rollout:
    """Record the generation the event carries; nothing is generated here."""
    assert event.generated is not None, "a generation is checkpointed only once generated"
    return event.generated
