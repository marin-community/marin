# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run one task attempt: machines, the model turn loop under its limits, grading, and cleanup."""

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AsyncExitStack
from dataclasses import asdict, dataclass, replace
from typing import Any

from shellbox.machine import DEFAULT_MACHINE_OUTPUT_LIMIT_BYTES, Machine, MachineFactory
from taskcompendium.grading_result import GradeResult, GradingFailure, Outcome
from taskcompendium.models import SessionGrader
from taskcompendium.runtime.output_capture import captured_output_files
from taskcompendium.submission import conversation_messages

from rolloutengine.checks import check_spec
from rolloutengine.cleanup import _Cleanup
from rolloutengine.contracts import (
    BUDGET_STOP_REASON,
    LENGTH_STOP_REASON,
    MAX_TURNS_STOP_REASON,
    TOTAL_TURN_TIMEOUT_STOP_REASON,
    AttemptEnded,
    AttemptStarted,
    Continuation,
    GenerationLimitReached,
    Graded,
    GradedRollout,
    HarnessPrepared,
    MachineReady,
    ModelRequest,
    ModelTurn,
    ModelTurnCompleted,
    Prompt,
    Rollout,
    RolloutContractError,
    RolloutFailure,
    RolloutInterrupted,
    RolloutObserver,
    RolloutOperation,
    RolloutStep,
    Transition,
    TransitionCompleted,
)
from rolloutengine.event_delivery import _EventDelivery
from rolloutengine.grading import _grade_rollout
from rolloutengine.harness import Episode, Harness, HarnessContext
from rolloutengine.machines import _prepare_machine
from rolloutengine.spec import TRACE_METRIC, TaskRolloutSpec, TraceMode
from rolloutengine.trace import _token_trace

Model = Callable[[ModelRequest], Awaitable[ModelTurn]]
"""The policy under rollout: it serves one model request and returns the exact turn."""

ADVANCE_INCOMPLETE = "advance_incomplete"
"""``Transition.metrics`` key of a turn whose ``advance`` did not return; training must exclude its step."""


class _Interrupted(Exception):
    """Carry the failed operation to the attempt boundary; the original error is the cause."""

    def __init__(self, operation: RolloutOperation):
        super().__init__(operation)
        self.operation = operation


class _Record:
    """The attempt's record as it grows: conversation, token trace, steps, stop reason, and verdict.

    It lives outside the attempt deadline, so an interrupted attempt keeps all that it recorded.
    ``messages`` is the conversation through the last response; ``observations`` follow it in the
    next request only.
    """

    def __init__(self, spec: TaskRolloutSpec, run_id: str):
        self.spec = spec
        self.run_id = run_id
        self.messages: list[dict[str, Any]] = conversation_messages(spec.task.context.events)
        self.trace = _token_trace(spec.trace)
        self.steps: list[RolloutStep] = []
        self.pending: RolloutStep | None = None
        self.observations: tuple[dict[str, Any], ...] = ()
        self.options: dict[str, Any] = {}
        self.token_ids: tuple[int, ...] | None = None
        self.assistant_index: int | None = None
        self.unserved_prompt: tuple[int, ...] | None = None
        self.stop_reason = "error"
        self.grade = GradeResult(Outcome.UNAVAILABLE, None, "Execution has no final grade")
        self.outputs: dict[str, bytes] = {}

    @property
    def has_response(self) -> bool:
        return bool(self.steps) or self.pending is not None

    def start(self, prompt: Prompt) -> None:
        """Start the conversation, or replace it after a reset with a new training record."""
        self.messages = list(prompt.messages)
        self.options = prompt.options
        self.token_ids = prompt.token_ids
        self.trace = _token_trace(self.spec.trace)
        self.steps.clear()
        self.pending = None
        self.observations = ()
        self.assistant_index = None

    def request(self) -> ModelRequest:
        messages = (*self.messages, *self.observations)
        if self.token_ids is not None:
            return ModelRequest(messages, self.options, prompt_token_ids=self.token_ids)
        # A chat endpoint re-tokenizes every request, so a messages trace never continues served tokens.
        if self.assistant_index is None or self.spec.trace == TraceMode.MESSAGES:
            return ModelRequest(messages, self.options)
        return ModelRequest(messages, self.options, Continuation(self.trace.prefix, self.assistant_index))

    def add_turn(self, turn: ModelTurn) -> None:
        if self.token_ids is not None and turn.prompt_token_ids != self.token_ids:
            raise RolloutContractError("Model transport changed the replay token prefix")
        self.trace.append(turn)
        self.messages.extend(self.observations)
        self.observations = ()
        self.assistant_index = len(self.messages)
        self.messages.append(turn.message)
        self.token_ids = None
        self.stop_reason = turn.stop_reason
        # Token evidence belongs to the attempt even if the harness fails to handle the turn.
        incomplete = Transition(done=True, metrics={ADVANCE_INCOMPLETE: 1.0})
        self.pending = RolloutStep(turn, incomplete, self.trace.response_end, tuple(self.messages))

    def add_transition(self, transition: Transition) -> None:
        assert self.pending is not None
        self.steps.append(replace(self.pending, transition=transition))
        self.pending = None
        self.observations = transition.observations
        self.token_ids = transition.token_ids

    def no_response(self, reason: str, prompt_token_ids: tuple[int, ...] | None = None) -> None:
        self.grade = GradeResult(Outcome.UNAVAILABLE, None, reason)
        self.unserved_prompt = prompt_token_ids

    def snapshot(self) -> Rollout:
        steps = (*self.steps, self.pending) if self.pending is not None else tuple(self.steps)
        return Rollout(
            self.run_id,
            tuple(self.messages),
            self.trace.prompt if self.unserved_prompt is None else self.unserved_prompt,
            self.trace.response,
            self.trace.loss_mask,
            self.trace.logprobs,
            self.stop_reason,
            steps,
            dict(steps[-1].transition.metrics) if steps else {},
            self.outputs,
        )


async def _captured_outputs(spec: TaskRolloutSpec, machine: Machine | None) -> dict[str, bytes]:
    task = spec.task
    if machine is None or not task.output_paths:
        return {}
    return await captured_output_files(
        machine,
        task.output_paths,
        timeout=spec.limits.verifier_timeout,
        limit_bytes=DEFAULT_MACHINE_OUTPUT_LIMIT_BYTES,
    )


@dataclass(frozen=True)
class _Attempt:
    """One attempt's inputs, and the record and cleanup that outlive its deadline."""

    spec: TaskRolloutSpec
    harness: Harness[Any]
    model: Model
    run_id: str
    record: _Record
    cleanup: _Cleanup
    events: _EventDelivery


class RolloutEngine:
    """Run task attempts with registered machine factories and harnesses.

    ``machine_factories`` is keyed by ``MachineRuntimeSpec.backend`` and ``harnesses`` by
    ``TaskRolloutSpec.harness``; each run supplies the model under rollout. The engine grades each attempt
    with the task's grader, except that a ``SessionGrader`` task is graded by its harness. An optional
    observer receives progress events while each attempt runs.
    """

    def __init__(
        self,
        machine_factories: Mapping[str, MachineFactory],
        harnesses: Mapping[str, Harness[Any]],
        *,
        observer: RolloutObserver | None = None,
    ):
        self.machine_factories = machine_factories
        self.harnesses = harnesses
        self.observer = observer

    def check(self, spec: TaskRolloutSpec) -> None:
        """Reject a spec before any machine starts: unknown factories or harness, or unsupported requirements."""
        check_spec(spec, self.machine_factories)
        if spec.harness not in self.harnesses:
            raise ValueError(f"Unknown harness: {spec.harness!r}")
        harness = self.harnesses[spec.harness]
        if isinstance(spec.task.grader, SessionGrader) and harness.grade is None:
            raise ValueError(f"Harness {spec.harness!r} does not grade, so it cannot run a session-graded task")
        harness.check(spec)

    async def run_task(self, spec: TaskRolloutSpec, model: Model, *, run_id: str | None = None) -> GradedRollout:
        """Run and grade one attempt of ``spec`` with ``model``. Release its resources outside the attempt deadline.

        The events of the attempt and ``Rollout.run_id`` carry ``run_id``, a new random ID unless the caller gives
        one. A spec runs many attempts, so the ID is an argument of the attempt, not a field of the spec. Queued
        events go to the observer within the cleanup deadline after the attempt releases its resources, and
        ``Rollout.metrics`` counts the events that the observer did not receive. An attempt that fails raises
        ``RolloutInterrupted`` with the rollout so far and its grade.
        """
        self.check(spec)
        run_id = uuid.uuid4().hex if run_id is None else run_id
        attempt = _Attempt(
            spec,
            self.harnesses[spec.harness],
            model,
            run_id,
            _Record(spec, run_id),
            _Cleanup(spec.limits.cleanup_timeout),
            _EventDelivery(self.observer, run_id, spec.task.id),
        )
        started = time.monotonic()
        try:
            attempt.events.emit(AttemptStarted())
            failure = await self._attempt(attempt)
            duration = time.monotonic() - started
            if failure is None:
                attempt.events.emit(AttemptEnded(attempt.record.stop_reason, None, None, duration))
            else:
                attempt.events.emit(AttemptEnded(None, failure[0], repr(failure[1]), duration))
            delivery = await attempt.events.close(spec.limits.cleanup_timeout)
        finally:
            attempt.events.cancel()
        rollout = attempt.record.snapshot()
        metrics = {**rollout.metrics, **delivery}
        grade = attempt.record.grade
        if attempt.cleanup.errors:
            cleanup_errors = [asdict(error) for error in attempt.cleanup.errors]
            grade = replace(grade, diagnostics={**grade.diagnostics, "cleanup_errors": cleanup_errors})
            metrics["cleanup_error_count"] = float(len(cleanup_errors))
        if spec.trace == TraceMode.MESSAGES:
            metrics[TRACE_METRIC] = TraceMode.MESSAGES.value
        rollout = replace(rollout, metrics=metrics)
        if failure is None:
            return GradedRollout(rollout, grade)
        operation, cause = failure
        rollout = replace(rollout, failure=RolloutFailure(type(cause).__name__, {"operation": operation.value}))
        raise RolloutInterrupted(rollout, operation, grade) from cause

    async def _attempt(self, attempt: _Attempt) -> tuple[RolloutOperation, BaseException] | None:
        """Run the attempt under its deadline and release its resources; return the failure, if any."""
        deadline = asyncio.timeout(attempt.spec.limits.attempt_timeout)
        async with AsyncExitStack() as resources:
            try:
                async with deadline:
                    await self._run(attempt, resources)
            except TimeoutError as error:
                if not deadline.expired():
                    raise
                return RolloutOperation.ATTEMPT, error
            except _Interrupted as interrupted:
                assert interrupted.__cause__ is not None
                return interrupted.operation, interrupted.__cause__
        return None

    async def _run(self, attempt: _Attempt, resources: AsyncExitStack) -> None:
        spec, record, events = attempt.spec, attempt.record, attempt.events
        task = spec.task
        started = time.monotonic()
        try:
            machine = await _prepare_machine(
                task.environment_requirements,
                spec.task_machine,
                task.resources.all + task.resources.worker,
                self.machine_factories,
                attempt.cleanup,
                resources,
            )
        except Exception as error:
            raise _Interrupted(RolloutOperation.START) from error
        if machine is not None:
            events.emit(MachineReady(time.monotonic() - started))
        started = time.monotonic()
        try:
            episode = attempt.harness.open(HarnessContext(spec, machine, events, attempt.run_id))
            resources.push_async_callback(attempt.cleanup.run, "episode_close", episode.close)
            prompt = await episode.prepare()
        except Exception as error:
            raise _Interrupted(RolloutOperation.PREPARE) from error
        events.emit(HarnessPrepared(prompt.messages, prompt.options, time.monotonic() - started))
        record.start(prompt)
        model_error = await self._turns(attempt, episode)
        if record.has_response:
            await self._grade(attempt, episode, machine, resources)
        if model_error is not None:
            raise _Interrupted(RolloutOperation.MODEL) from model_error

    async def _turns(self, attempt: _Attempt, episode: Episode) -> Exception | None:
        """Sample turns until the episode is done or a limit ends the loop; return a model failure."""
        limits, record, events = attempt.spec.limits, attempt.record, attempt.events
        spent_tokens = 0
        deadline = asyncio.timeout(limits.total_turn_timeout)
        try:
            async with deadline:
                for index in range(limits.max_turns):
                    started = time.monotonic()
                    try:
                        async with asyncio.timeout(limits.model_turn_timeout):
                            turn = await attempt.model(record.request())
                    except GenerationLimitReached as limit:
                        record.stop_reason = LENGTH_STOP_REASON
                        if not record.has_response:
                            record.no_response(
                                "Generation limit reached before the first response", limit.prompt_token_ids
                            )
                        return None
                    except RolloutContractError:
                        raise
                    except Exception as error:
                        return error
                    if limits.max_total_tokens is not None and turn.usage is None and not turn.prompt_token_ids:
                        raise RolloutContractError(
                            "A token budget needs ModelTurn.usage or prompt token IDs from the model"
                        )
                    spent_tokens += turn.request_tokens
                    # Without a reset, the next prompt holds at least this request's prompt and response,
                    # so stop when that prompt alone would reach the budget.
                    budget_reached = (
                        limits.max_total_tokens is not None
                        and spent_tokens + turn.request_tokens >= limits.max_total_tokens
                    )
                    record.add_turn(turn)
                    events.emit(
                        ModelTurnCompleted(
                            index,
                            turn.message,
                            turn.stop_reason,
                            len(turn.prompt_token_ids),
                            len(turn.response_token_ids),
                            turn.usage,
                            turn.metadata,
                            time.monotonic() - started,
                        ),
                    )
                    started = time.monotonic()
                    try:
                        async with asyncio.timeout(limits.tool_turn_timeout):
                            transition = await episode.advance(turn)
                    except RolloutContractError:
                        raise
                    except Exception as error:
                        raise _Interrupted(RolloutOperation.ADVANCE) from error
                    events.emit(
                        TransitionCompleted(
                            index, transition.done, transition.observations, transition.reset, time.monotonic() - started
                        ),
                    )
                    last_turn = index + 1 == limits.max_turns
                    if transition.reset is not None and not last_turn and not budget_reached:
                        record.start(transition.reset)
                        continue
                    record.add_transition(transition)
                    if transition.done or turn.stop_reason == LENGTH_STOP_REASON:
                        return None
                    if budget_reached:
                        record.stop_reason = BUDGET_STOP_REASON
                        return None
                    if last_turn:
                        record.stop_reason = MAX_TURNS_STOP_REASON
                        return None
        except TimeoutError:
            if not deadline.expired():
                raise
            record.stop_reason = TOTAL_TURN_TIMEOUT_STOP_REASON
            if not record.has_response:
                record.no_response("Turn deadline expired before the first response")
        return None

    async def _grade(
        self, attempt: _Attempt, episode: Episode, machine: Machine | None, resources: AsyncExitStack
    ) -> None:
        """Capture the run outputs and grade them with the task's grader, or with the harness."""
        spec, record = attempt.spec, attempt.record
        started = time.monotonic()
        try:
            async with asyncio.timeout(spec.limits.verifier_timeout):
                record.outputs = await _captured_outputs(spec, machine)
                if isinstance(spec.task.grader, SessionGrader):
                    assert attempt.harness.grade is not None
                    grade = await attempt.harness.grade(episode, record.snapshot())
                else:
                    grade = await _grade_rollout(
                        spec,
                        tuple(record.messages),
                        record.outputs,
                        machine,
                        self.machine_factories,
                        attempt.cleanup,
                        resources,
                        stop_reason=record.stop_reason,
                    )
        except TimeoutError as error:
            record.grade = GradeResult(
                Outcome.INFRA_ERROR, None, "Verifier deadline expired", failure=GradingFailure.TIMEOUT
            )
            raise _Interrupted(RolloutOperation.GRADE) from error
        except Exception as error:
            raise _Interrupted(RolloutOperation.GRADE) from error
        record.grade = grade
        attempt.events.emit(Graded(grade.status, grade.reward, grade.error, grade.failure, time.monotonic() - started))
