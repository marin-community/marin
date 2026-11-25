# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run one task attempt: the machines, the harness under the attempt deadline, grading, and cleanup."""

import asyncio
import time
import uuid
import warnings
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AsyncExitStack
from dataclasses import asdict, dataclass, replace

from shellbox.machine import DEFAULT_MACHINE_OUTPUT_LIMIT_BYTES, Machine, MachineFactory, MachineSpec
from taskcompendium.grading_result import GRADING_DURATION_SECONDS, GRADING_FAILURE, GradeResult, GradingFailure, Outcome
from taskcompendium.models import CONVERSATION_ANSWERS, NoGrader
from taskcompendium.runtime.output_capture import captured_output_files
from taskcompendium.submission import conversation_messages

from rolloutengine.cleanup import _Cleanup
from rolloutengine.event_delivery import _EventDelivery
from rolloutengine.grading import _grade_rollout
from rolloutengine.harness.core import Harness, HarnessContext
from rolloutengine.machines import _prepare_machine
from rolloutengine.models import (
    GRADING_SECONDS_METRIC,
    SETUP_COMPLETED_METRIC,
    SETUP_SECONDS_METRIC,
    AttemptEnded,
    AttemptStarted,
    Graded,
    GradedRollout,
    GradedRolloutInterrupted,
    MachineReady,
    Model,
    RolloutObserver,
    RolloutOperation,
    TaskRolloutSpec,
)
from rolloutengine.recorder import _CleanupScope, _failure_record, _Recorder, _run_harness
from rolloutengine.validation import validate_spec


class _Interrupted(Exception):
    """Carry the failed operation to the attempt boundary; the original error is the cause."""

    def __init__(self, operation: RolloutOperation):
        super().__init__(operation)
        self.operation = operation


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


@dataclass
class _Attempt:
    """One attempt's inputs, and the record, grade, and cleanup that outlive its deadline."""

    spec: TaskRolloutSpec
    harness: Harness
    rollout_id: str
    recorder: _Recorder
    cleanup: _Cleanup
    events: _EventDelivery
    grade: GradeResult
    outputs: dict[str, bytes]
    verifier: Callable[[MachineSpec], Awaitable[Machine]] | None = None
    setup_seconds: float | None = None
    grading_seconds: float | None = None


class RolloutEngine:
    """Run task attempts with registered machine factories and harnesses.

    ``machine_factories`` is keyed by ``MachineRuntimeSpec.backend`` and ``harnesses`` by
    ``TaskRolloutSpec.harness``; each run supplies the model under rollout. The harness runs the turns, and
    the engine grades the rollout with the task's grader. An optional observer receives progress events
    while each attempt runs.
    """

    def __init__(
        self,
        machine_factories: Mapping[str, MachineFactory],
        harnesses: Mapping[str, Harness],
        *,
        observer: RolloutObserver | None = None,
    ):
        self.machine_factories = machine_factories
        self.harnesses = harnesses
        self.observer = observer

    def check(self, spec: TaskRolloutSpec) -> None:
        """Reject a spec before any machine starts: unknown factories or harness, or unsupported requirements."""
        validate_spec(spec, self.machine_factories)
        if spec.harness not in self.harnesses:
            raise ValueError(f"Unknown harness: {spec.harness!r}")
        self.harnesses[spec.harness].check(spec.task, spec.limits)

    async def run_task(
        self,
        spec: TaskRolloutSpec,
        model: Model,
        *,
        rollout_id: str | None = None,
        verifier: Callable[[MachineSpec], Awaitable[Machine]] | None = None,
        run_id: str | None = None,
    ) -> GradedRollout:
        """Run and grade one attempt of ``spec`` with ``model``. Release its resources outside the attempt deadline.

        The events of the attempt and ``Rollout.rollout_id`` carry ``rollout_id``, a new random ID unless the caller
        gives one. A spec runs many attempts, so the ID is an argument of the attempt, not a field of the spec. A
        given ID must be new for each call, a retry included, because ``(rollout_id, index)`` identifies an event. Queued
        events go to the observer within the cleanup deadline after the attempt releases its resources, and
        ``Rollout.metrics`` counts the events that the observer did not receive. An attempt that fails raises
        ``GradedRolloutInterrupted`` with the rollout so far and its grade.
        ``verifier``, if supplied, acquires the verifier machine from the prepared spec; the caller owns its cleanup.
        ``run_id`` is a deprecated alias of ``rollout_id`` for callers outside this repository; remove after 2026-12-31.
        """
        if run_id is not None:
            if rollout_id is not None:
                raise TypeError("Pass rollout_id, not both rollout_id and its deprecated alias run_id")
            warnings.warn("run_task(run_id=...) is deprecated; pass rollout_id", DeprecationWarning, stacklevel=2)
            rollout_id = run_id
        self.check(spec)
        harness = self.harnesses[spec.harness]
        rollout_id = uuid.uuid4().hex if rollout_id is None else rollout_id
        events = _EventDelivery(self.observer, rollout_id, spec.task.id)
        attempt = _Attempt(
            spec,
            harness,
            rollout_id,
            _Recorder(model, spec.limits, spec.trace, events, context_recovery=harness.context_recovery),
            _Cleanup(spec.limits.cleanup_timeout),
            events,
            GradeResult(Outcome.UNAVAILABLE, None, "Execution has no final grade"),
            {},
            verifier,
        )
        started = time.monotonic()
        try:
            events.emit(AttemptStarted(spec.harness, spec.trace, spec.task.source.dataset))
            failure = await self._attempt(attempt)
            duration = time.monotonic() - started
            turns, total_tokens = attempt.recorder.turns, attempt.recorder.spent_tokens
            if failure is None:
                events.emit(AttemptEnded(attempt.recorder.stop_reason, None, None, duration, turns, total_tokens))
            else:
                events.emit(AttemptEnded(None, failure[0], repr(failure[1]), duration, turns, total_tokens))
            delivery = await events.close(spec.limits.cleanup_timeout)
        finally:
            events.cancel()
        rollout = attempt.recorder.rollout(rollout_id, attempt.outputs)
        metrics = {**rollout.metrics, **delivery}
        if spec.task_machine is not None:
            metrics[SETUP_COMPLETED_METRIC] = float(attempt.setup_seconds is not None)
        grade = attempt.grade
        if attempt.cleanup.errors:
            cleanup_errors = [asdict(error) for error in attempt.cleanup.errors]
            grade = replace(grade, diagnostics={**grade.diagnostics, "cleanup_errors": cleanup_errors})
            metrics["cleanup_error_count"] = float(len(cleanup_errors))
        if (
            failure is None
            and attempt.grading_seconds is not None
            and grade.status in {Outcome.GRADED, Outcome.SUBMISSION_FAILURE}
        ):
            metrics[GRADING_SECONDS_METRIC] = attempt.grading_seconds - grade.verifier_setup_seconds
            metrics[SETUP_SECONDS_METRIC] = (attempt.setup_seconds or 0.0) + grade.verifier_setup_seconds
        rollout = replace(rollout, metrics=metrics)
        if failure is None:
            return GradedRollout(rollout, grade)
        operation, cause = failure
        rollout = replace(rollout, failure=_failure_record(operation, cause))
        raise GradedRolloutInterrupted(rollout, operation, grade) from cause

    async def _attempt(self, attempt: _Attempt) -> tuple[RolloutOperation, BaseException] | None:
        """Run the attempt under its deadline and release its resources; return the failure, if any."""
        deadline = asyncio.timeout(attempt.spec.limits.attempt_timeout)
        async with AsyncExitStack() as resources:
            try:
                async with deadline:
                    await self._run(attempt, resources, deadline)
            except TimeoutError as error:
                if not deadline.expired():
                    raise
                return RolloutOperation.ATTEMPT, error
            except _Interrupted as interrupted:
                assert interrupted.__cause__ is not None
                return interrupted.operation, interrupted.__cause__
        return None

    async def _run(self, attempt: _Attempt, resources: AsyncExitStack, deadline: asyncio.Timeout) -> None:
        spec, events = attempt.spec, attempt.events
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
            attempt.setup_seconds = time.monotonic() - started
            events.emit(MachineReady(attempt.setup_seconds))
        released = AsyncExitStack()
        await resources.enter_async_context(released)
        stopped = AsyncExitStack()
        await resources.enter_async_context(stopped)
        cleanup = _CleanupScope(stopped, released, attempt.cleanup)
        recorder = attempt.recorder
        context = HarnessContext(
            task, spec.limits, machine, events, cleanup, attempt.rollout_id, recorder.begin_turn_budget
        )
        failure = await _run_harness(attempt.harness, context, recorder, attempt_deadline=deadline)
        # Stop the agent before it can race output capture or grading. Services it left on the task machine
        # stay available to the grader until the machine is released.
        await stopped.aclose()
        if failure is None and not recorder.has_response and task.answer_type in CONVERSATION_ANSWERS:
            attempt.grade = GradeResult(Outcome.UNAVAILABLE, None, recorder.no_response_reason)
        # A model failure leaves the completed turns intact, so the engine grades them. An attempt that ended
        # before the first response, at a limit or with an agent that gave up, leaves its answer on the machine.
        elif failure is None or (failure[0] == RolloutOperation.MODEL and recorder.has_response):
            try:
                await self._grade(attempt, machine, resources)
            except _Interrupted as interrupted:
                if failure is None:
                    raise
                # The model failure stays the attempt's failure, and its cause names the grading failure.
                failure[1].add_note(f"Grading the completed turns failed: {interrupted.__cause__!r}")
        if failure is not None:
            raise _Interrupted(failure[0]) from failure[1]

    async def _grade(self, attempt: _Attempt, machine: Machine | None, resources: AsyncExitStack) -> None:
        spec, recorder = attempt.spec, attempt.recorder
        messages = recorder.graded_messages()
        if messages is None:
            # The attempt ended before the first response, so the grader sees an empty reply to the task.
            messages = (*conversation_messages(spec.task.context.events), {"role": "assistant", "content": ""})
        started = time.monotonic()
        failure = None
        try:
            async with asyncio.timeout(spec.limits.verifier_timeout):
                attempt.outputs = await _captured_outputs(spec, machine)
                grade = await _grade_rollout(
                    spec,
                    messages,
                    attempt.outputs,
                    machine,
                    self.machine_factories,
                    attempt.cleanup,
                    resources,
                    stop_reason=recorder.stop_reason,
                    verifier=attempt.verifier,
                )
        except TimeoutError as error:
            attempt.grade = GradeResult(
                Outcome.INFRA_ERROR, None, "Verifier deadline expired", failure=GradingFailure.TIMEOUT
            )
            raise _Interrupted(RolloutOperation.GRADE) from error
        except asyncio.CancelledError:
            # If the attempt deadline fired during grading, the caller receives this observation.
            failure = GradingFailure.TIMEOUT
            raise
        except Exception as error:
            failure = GradingFailure.EXECUTION
            raise _Interrupted(RolloutOperation.GRADE) from error
        else:
            attempt.grade = grade
        finally:
            duration = time.monotonic() - started
            attempt.grading_seconds = duration
            if not isinstance(spec.task.grader, NoGrader):
                grade = attempt.grade
                diagnostics = {**grade.diagnostics, GRADING_DURATION_SECONDS: duration}
                failure = failure or grade.failure
                if failure is not None:
                    diagnostics[GRADING_FAILURE] = failure.value
                elif grade.status in {Outcome.INFRA_ERROR, Outcome.INVALID_TASK}:
                    diagnostics[GRADING_FAILURE] = grade.status.value
                attempt.grade = replace(grade, diagnostics=diagnostics)
        grade = attempt.grade
        attempt.events.emit(Graded(grade.status, grade.reward, grade.error, grade.failure, duration))
