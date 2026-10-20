# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run one task attempt: the machines, the harness under the attempt deadline, grading, and cleanup."""

import asyncio
import time
import uuid
from collections.abc import Mapping
from contextlib import AsyncExitStack
from dataclasses import asdict, dataclass, replace

from shellbox.machine import DEFAULT_MACHINE_OUTPUT_LIMIT_BYTES, Machine, MachineFactory
from taskcompendium.grading_result import GradeResult, GradingFailure, GradingPhase, GradingProgress, Outcome
from taskcompendium.runtime.output_capture import captured_output_files

from rolloutengine.cleanup import _Cleanup
from rolloutengine.event_delivery import _EventDelivery
from rolloutengine.grading import _grade_rollout
from rolloutengine.harness.core import Harness, HarnessContext
from rolloutengine.machines import _prepare_machine
from rolloutengine.models import (
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


def _expired_verifier_grade(progress: GradingProgress) -> GradeResult:
    """The grade of an attempt whose verifier deadline expired.

    The deadline covers the verifier machine's preparation as well as the grader's command; which
    of the two ran out of time decides whether the host or the grader is reported.
    """
    if progress.phase is GradingPhase.STAGING:
        return GradeResult(
            Outcome.INFRA_ERROR,
            None,
            "Verifier deadline expired while preparing the verifier machine",
            failure=GradingFailure.STAGING,
        )
    return GradeResult(Outcome.INFRA_ERROR, None, "Verifier deadline expired", failure=GradingFailure.TIMEOUT)


@dataclass
class _Attempt:
    """One attempt's inputs, and the record, grade, and cleanup that outlive its deadline."""

    spec: TaskRolloutSpec
    harness: Harness
    run_id: str
    recorder: _Recorder
    cleanup: _Cleanup
    events: _EventDelivery
    grade: GradeResult
    outputs: dict[str, bytes]


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

    async def run_task(self, spec: TaskRolloutSpec, model: Model, *, run_id: str | None = None) -> GradedRollout:
        """Run and grade one attempt of ``spec`` with ``model``. Release its resources outside the attempt deadline.

        The events of the attempt and ``Rollout.run_id`` carry ``run_id``, a new random ID unless the caller gives
        one. A spec runs many attempts, so the ID is an argument of the attempt, not a field of the spec. Queued
        events go to the observer within the cleanup deadline after the attempt releases its resources, and
        ``Rollout.metrics`` counts the events that the observer did not receive. An attempt that fails raises
        ``GradedRolloutInterrupted`` with the rollout so far and its grade.
        """
        self.check(spec)
        run_id = uuid.uuid4().hex if run_id is None else run_id
        events = _EventDelivery(self.observer, run_id, spec.task.id)
        attempt = _Attempt(
            spec,
            self.harnesses[spec.harness],
            run_id,
            _Recorder(model, spec.limits, spec.trace, events),
            _Cleanup(spec.limits.cleanup_timeout),
            events,
            GradeResult(Outcome.UNAVAILABLE, None, "Execution has no final grade"),
            {},
        )
        started = time.monotonic()
        try:
            events.emit(AttemptStarted())
            failure = await self._attempt(attempt)
            duration = time.monotonic() - started
            if failure is None:
                events.emit(AttemptEnded(attempt.recorder.stop_reason, None, None, duration))
            else:
                events.emit(AttemptEnded(None, failure[0], repr(failure[1]), duration))
            delivery = await events.close(spec.limits.cleanup_timeout)
        finally:
            events.cancel()
        rollout = attempt.recorder.rollout(run_id, attempt.outputs)
        metrics = {**rollout.metrics, **delivery}
        grade = attempt.grade
        if attempt.recorder.no_response_reason is not None:
            grade = GradeResult(Outcome.UNAVAILABLE, None, attempt.recorder.no_response_reason)
        if attempt.cleanup.errors:
            cleanup_errors = [asdict(error) for error in attempt.cleanup.errors]
            grade = replace(grade, diagnostics={**grade.diagnostics, "cleanup_errors": cleanup_errors})
            metrics["cleanup_error_count"] = float(len(cleanup_errors))
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
            events.emit(MachineReady(time.monotonic() - started))
        # The harness's resources close after grading, before the machines.
        cleanup = _CleanupScope(resources, attempt.cleanup)
        context = HarnessContext(task, spec.limits, machine, events, cleanup, attempt.run_id)
        failure = await _run_harness(attempt.harness, context, attempt.recorder, attempt_deadline=deadline)
        # A model failure leaves the completed turns intact, so the engine grades them.
        if attempt.recorder.has_response and (failure is None or failure[0] == RolloutOperation.MODEL):
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
        assert messages is not None
        # Capturing the agent's output files is the runtime's work, like preparing the verifier machine;
        # the grading that follows reports its own phases.
        progress = GradingProgress(GradingPhase.STAGING)
        started = time.monotonic()
        try:
            async with asyncio.timeout(spec.limits.verifier_timeout):
                attempt.outputs = await _captured_outputs(spec, machine)
                progress.phase = GradingPhase.GRADER
                grade = await _grade_rollout(
                    spec,
                    messages,
                    attempt.outputs,
                    machine,
                    self.machine_factories,
                    attempt.cleanup,
                    resources,
                    stop_reason=recorder.stop_reason,
                    progress=progress,
                )
        except TimeoutError as error:
            attempt.grade = _expired_verifier_grade(progress)
            raise _Interrupted(RolloutOperation.GRADE) from error
        except Exception as error:
            raise _Interrupted(RolloutOperation.GRADE) from error
        attempt.grade = grade
        attempt.events.emit(Graded(grade.status, grade.reward, grade.error, grade.failure, time.monotonic() - started))
