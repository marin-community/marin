# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run one task attempt: the machines, the harness under the attempt deadline, grading, and cleanup."""

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AsyncExitStack
from dataclasses import asdict, dataclass, replace

from shellbox.machine import DEFAULT_MACHINE_OUTPUT_LIMIT_BYTES, Machine, MachineFactory, MachineSpec
from taskcompendium.grading_result import GradeResult, GradingFailure, Outcome
from taskcompendium.models import CONVERSATION_ANSWERS, ScriptGrader
from taskcompendium.runtime.output_capture import captured_output_files
from taskcompendium.submission import conversation_messages

from rolloutengine.cleanup import _Cleanup
from rolloutengine.event_delivery import _EventDelivery
from rolloutengine.grading import _grade_rollout
from rolloutengine.harness.core import Harness, HarnessContext
from rolloutengine.machines import _prepare_machine
from rolloutengine.models import (
    GRADING_SECONDS_METRIC,
    SETUP_SECONDS_METRIC,
    AttemptEnded,
    AttemptStarted,
    Graded,
    GradedRollout,
    GradedRolloutInterrupted,
    MachineReady,
    Model,
    Rollout,
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
    run_id: str
    recorder: _Recorder
    cleanup: _Cleanup
    events: _EventDelivery
    grade: GradeResult
    outputs: dict[str, bytes]
    verifier: Callable[[MachineSpec], Awaitable[Machine]] | None = None
    setup_seconds: float = 0.0
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
        save_submission: Callable[[Rollout], Awaitable[None]] | None = None,
        before_judge: Callable[[], Awaitable[None]] | None = None,
    ):
        self.machine_factories = machine_factories
        self.harnesses = harnesses
        self.before_judge = before_judge
        self.observer = observer
        self.save_submission = save_submission

    def check(self, spec: TaskRolloutSpec) -> None:
        """Reject a spec before any machine starts: unknown factories or harness, or unsupported requirements."""
        validate_spec(spec, self.machine_factories)
        if spec.harness not in self.harnesses:
            raise ValueError(f"Unknown harness: {spec.harness!r}")
        self.harnesses[spec.harness].check(spec.task, spec.limits)

    async def grade_saved_outputs(self, spec: TaskRolloutSpec, rollout: Rollout) -> GradeResult:
        """Replay an archived file submission in fresh machines after a grading infrastructure failure."""
        grader = spec.task.grader
        if not isinstance(grader, ScriptGrader) or grader.replay is None:
            raise ValueError("File recovery requires a clean replay and a separate grader")
        messages = next((r.messages for r in reversed(rollout.conversations) if r.steps), None)
        if messages is None:
            messages = (*conversation_messages(spec.task.context.events), {"role": "assistant", "content": ""})
        cleanup = _Cleanup(spec.limits.cleanup_timeout)
        async with AsyncExitStack() as resources:
            async with asyncio.timeout(spec.limits.verifier_timeout):
                grade = await _grade_rollout(
                    spec,
                    messages,
                    rollout.outputs,
                    None,
                    self.machine_factories,
                    cleanup,
                    resources,
                    before_judge=self.before_judge,
                    stop_reason=rollout.stop_reason,
                )
        if cleanup.errors:
            grade = replace(
                grade, diagnostics={**grade.diagnostics, "cleanup_errors": [asdict(e) for e in cleanup.errors]}
            )
        return grade

    async def run_task(
        self,
        spec: TaskRolloutSpec,
        model: Model,
        *,
        run_id: str | None = None,
        verifier: Callable[[MachineSpec], Awaitable[Machine]] | None = None,
    ) -> GradedRollout:
        """Run and grade one attempt of ``spec`` with ``model``. Release its resources outside the attempt deadline.

        The events of the attempt and ``Rollout.run_id`` carry ``run_id``, a new random ID unless the caller gives
        one. A spec runs many attempts, so the ID is an argument of the attempt, not a field of the spec. A given ID
        must be new for each call, a retry included, because ``(run_id, index)`` identifies an event. Queued
        events go to the observer within the cleanup deadline after the attempt releases its resources, and
        ``Rollout.metrics`` counts the events that the observer did not receive. An attempt that fails raises
        ``GradedRolloutInterrupted`` with the rollout so far and its grade.
        ``verifier``, if supplied, acquires the verifier machine from the prepared spec; the caller owns its cleanup.
        """
        self.check(spec)
        harness = self.harnesses[spec.harness]
        run_id = uuid.uuid4().hex if run_id is None else run_id
        events = _EventDelivery(self.observer, run_id, spec.task.id)
        attempt = _Attempt(
            spec,
            harness,
            run_id,
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
        rollout = attempt.recorder.rollout(run_id, attempt.outputs)
        metrics = {**rollout.metrics, **delivery}
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
            metrics[SETUP_SECONDS_METRIC] = attempt.setup_seconds + grade.verifier_setup_seconds
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
        context = HarnessContext(task, spec.limits, machine, events, cleanup, attempt.run_id)
        recorder = attempt.recorder
        failure = await _run_harness(attempt.harness, context, recorder, attempt_deadline=deadline)
        # As Harbor cancels its agent before verifying, the agent stops before output capture and grading, so that
        # it cannot change the task machine while it is graded. What the agent left running, such as a service in
        # its terminal, stays for the grader; the released closes run with the machines, after grading.
        await stopped.aclose()
        if failure is None and not recorder.has_response and task.answer_type in CONVERSATION_ANSWERS:
            attempt.grade = GradeResult(Outcome.UNAVAILABLE, None, recorder.no_response_reason)
        # A model failure leaves the completed turns intact, so the engine grades them. An attempt that ended
        # before the first response, at a limit or with an agent that gave up, leaves its answer on the machine.
        elif failure is None or (failure[0] == RolloutOperation.MODEL and recorder.has_response):
            try:
                await self._grade(attempt, machine, resources, self.save_submission if failure is None else None)
            except _Interrupted as interrupted:
                if failure is None:
                    raise
                # The model failure stays the attempt's failure, and its cause names the grading failure.
                failure[1].add_note(f"Grading the completed turns failed: {interrupted.__cause__!r}")
        if failure is not None:
            raise _Interrupted(failure[0]) from failure[1]

    async def _grade(
        self,
        attempt: _Attempt,
        machine: Machine | None,
        resources: AsyncExitStack,
        save_submission: Callable[[Rollout], Awaitable[None]] | None,
    ) -> None:
        spec, recorder = attempt.spec, attempt.recorder
        messages = recorder.graded_messages()
        if messages is None:
            # The attempt ended before the first response, so the grader sees an empty reply to the task.
            messages = (*conversation_messages(spec.task.context.events), {"role": "assistant", "content": ""})
        started = time.monotonic()
        try:
            async with asyncio.timeout(spec.limits.verifier_timeout):
                attempt.outputs = await _captured_outputs(spec, machine)
                if save_submission is not None and attempt.outputs:
                    await save_submission(recorder.rollout(attempt.run_id, attempt.outputs))
                grade = await _grade_rollout(
                    spec,
                    messages,
                    attempt.outputs,
                    machine,
                    self.machine_factories,
                    attempt.cleanup,
                    resources,
                    before_judge=self.before_judge,
                    stop_reason=recorder.stop_reason,
                    verifier=attempt.verifier,
                )
        except TimeoutError as error:
            attempt.grade = GradeResult(
                Outcome.INFRA_ERROR, None, "Verifier deadline expired", failure=GradingFailure.TIMEOUT
            )
            raise _Interrupted(RolloutOperation.GRADE) from error
        except Exception as error:
            raise _Interrupted(RolloutOperation.GRADE) from error
        attempt.grade = grade
        attempt.grading_seconds = time.monotonic() - started
        attempt.events.emit(Graded(grade.status, grade.reward, grade.error, grade.failure, attempt.grading_seconds))
