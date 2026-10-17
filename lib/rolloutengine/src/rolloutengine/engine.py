# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rollout iteration, cancellation, and model execution."""

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AsyncExitStack
from dataclasses import asdict, replace
from functools import partial
from typing import Any

from shellbox.machine import DEFAULT_MACHINE_OUTPUT_LIMIT_BYTES, Machine, MachineFactory
from taskcompendium.grading_result import GradeResult, GradingFailure, Outcome
from taskcompendium.models import SessionGrader
from taskcompendium.runtime.output_capture import captured_output_files
from taskcompendium.submission import conversation_messages
from verifyit.modes.grade_judge import JudgeConnection

from rolloutengine.cleanup import _Cleanup
from rolloutengine.contracts import (
    BUDGET_STOP_REASON,
    LENGTH_STOP_REASON,
    MAX_TURNS_STOP_REASON,
    TOTAL_TURN_TIMEOUT_STOP_REASON,
    AttemptEnded,
    AttemptStarted,
    GenerationLimitReached,
    Graded,
    MachineReady,
    ModelRequest,
    ModelTurn,
    ModelTurnCompleted,
    RolloutContractError,
    RolloutData,
    RolloutInterrupted,
    RolloutObserver,
    RolloutOperation,
    RolloutStep,
    SessionPrepared,
    TaskSession,
    TaskSessionContext,
    TaskSessionFactory,
    ToolProvider,
    Transition,
    TransitionCompleted,
)
from rolloutengine.event_delivery import _EventDelivery
from rolloutengine.grading import _grade_rollout
from rolloutengine.lowering import SHELLBOX_SESSION, validate_lowered_task
from rolloutengine.machines import _prepare_machine
from rolloutengine.spec import TRACE_METRIC, LoweredTaskSpec, TraceMode
from rolloutengine.task_session import _ShellboxTaskSession
from rolloutengine.trace import _token_trace


def _empty_rollout(lowered: LoweredTaskSpec) -> RolloutData:
    return RolloutData(
        lowered.task.id,
        tuple(conversation_messages(lowered.task.context.events)),
        (),
        (),
        (),
        _token_trace(lowered.session.trace).logprobs,
        GradeResult(Outcome.UNAVAILABLE, None, "Execution has no final grade"),
        "error",
    )


async def _captured_outputs(lowered: LoweredTaskSpec, machine: Machine | None) -> dict[str, bytes]:
    task = lowered.task
    if machine is None or not task.output_paths:
        return {}
    return await captured_output_files(
        machine,
        task.output_paths,
        timeout=lowered.session.verifier_timeout,
        limit_bytes=DEFAULT_MACHINE_OUTPUT_LIMIT_BYTES,
    )


class RolloutEngine:
    """Generate rollouts from lowered single-stage tasks under their session's trace mode.

    An optional observer receives progress events while each attempt runs.
    ``providers`` serve the built-in session's host tools, keyed by ``ProviderRequirement.action_interface``.
    """

    def __init__(
        self,
        model: Callable[[ModelRequest], Awaitable[ModelTurn]],
        factories: Mapping[str, MachineFactory],
        *,
        sessions: Mapping[str, TaskSessionFactory] | None = None,
        judge_connection: JudgeConnection | None = None,
        observer: RolloutObserver | None = None,
        providers: Mapping[str, ToolProvider] | None = None,
    ):
        self.model = model
        self.factories = factories
        self.sessions = {} if sessions is None else sessions
        self.judge_connection = judge_connection
        self.observer = observer
        self.providers = {} if providers is None else providers

    async def run(self, lowered: LoweredTaskSpec, run_key: str | None = None) -> RolloutData:
        """Run one attempt; release its resources outside the attempt deadline.

        Every event of the attempt carries ``run_key``, which defaults to the task id; callers that run
        a task more than once pass a key per attempt to tell the attempts' events apart.
        Queued events are delivered within the cleanup deadline after the attempt's resources are
        released; events the observer did not receive are counted in ``RolloutData.metrics``.
        """
        validate_lowered_task(lowered, factories=self.factories, sessions=self.sessions, providers=self.providers)
        events = _EventDelivery(self.observer, lowered.task.id if run_key is None else run_key, lowered.task.id)
        started = time.monotonic()
        try:
            events.emit(AttemptStarted())
            record, operation, cause = await self._run_attempt(lowered, events)
            duration = time.monotonic() - started
            if operation is None:
                events.emit(AttemptEnded(record.stop_reason, None, None, duration))
            else:
                events.emit(AttemptEnded(None, operation, None if cause is None else repr(cause), duration))
            delivery = await events.close(lowered.session.cleanup_timeout)
        finally:
            events.cancel()
        if delivery:
            record = replace(record, metrics={**record.metrics, **delivery})
        if operation is not None:
            raise RolloutInterrupted(record, operation) from cause
        return record

    async def _run_attempt(
        self, lowered: LoweredTaskSpec, events: _EventDelivery
    ) -> tuple[RolloutData, RolloutOperation | None, BaseException | None]:
        deadline = asyncio.timeout(lowered.session.attempt_timeout)
        cleanup = _Cleanup(lowered.session.cleanup_timeout)
        operation = None
        cause = None
        async with AsyncExitStack() as resources:
            try:
                async with deadline:
                    record = await self._run_task(lowered, resources, cleanup, deadline, events)
            except TimeoutError as error:
                if not deadline.expired():
                    raise
                operation, cause = RolloutOperation.ATTEMPT, error
                record = _empty_rollout(lowered)
            except RolloutInterrupted as error:
                operation, cause = error.operation, error.__cause__
                record = error.rollout
        if cleanup.errors:
            record = replace(
                record,
                grade=replace(
                    record.grade,
                    diagnostics={
                        **record.grade.diagnostics,
                        "cleanup_errors": [asdict(error) for error in cleanup.errors],
                    },
                ),
                metrics={**record.metrics, "cleanup_error_count": float(len(cleanup.errors))},
            )
        if lowered.session.trace == TraceMode.MESSAGES:
            record = replace(record, metrics={**record.metrics, TRACE_METRIC: TraceMode.MESSAGES.value})
        return record, operation, cause

    async def _run_task(
        self,
        lowered: LoweredTaskSpec,
        resources: AsyncExitStack,
        cleanup: _Cleanup,
        attempt: asyncio.Timeout,
        events: _EventDelivery,
    ) -> RolloutData:
        task = lowered.task
        started = time.monotonic()
        try:
            machine = await _prepare_machine(
                task.environment_requirements,
                lowered.runtime.task_machine,
                task.resources.all + task.resources.worker,
                self.factories,
                cleanup,
                resources,
            )
        except Exception as error:
            raise RolloutInterrupted(_empty_rollout(lowered), RolloutOperation.START) from error
        if machine is not None:
            events.emit(MachineReady(time.monotonic() - started))

        async def grade(
            messages: tuple[dict[str, Any], ...], *, stop_reason: str, outputs: dict[str, bytes]
        ) -> GradeResult:
            return await _grade_rollout(
                lowered,
                messages,
                outputs,
                machine,
                self.factories,
                cleanup,
                resources,
                stop_reason=stop_reason,
                judge_connection=self.judge_connection,
            )

        session_factories: Mapping[str, TaskSessionFactory] = {
            **self.sessions,
            SHELLBOX_SESSION: partial(_ShellboxTaskSession, providers=self.providers),
        }
        try:
            context = TaskSessionContext(
                lowered, machine, None if isinstance(task.grader, SessionGrader) else grade, events
            )
            session = session_factories[lowered.session.task_session](context)
        except Exception as error:
            raise RolloutInterrupted(_empty_rollout(lowered), RolloutOperation.PREPARE) from error
        resources.push_async_callback(cleanup.run, "session_close", session.close)
        return await self._run_session(lowered, session, machine, attempt, events)

    async def _run_session(
        self,
        lowered: LoweredTaskSpec,
        session: TaskSession,
        machine: Machine | None,
        attempt: asyncio.Timeout,
        events: _EventDelivery,
    ) -> RolloutData:
        task = lowered.task
        limits = lowered.session
        completed = _empty_rollout(lowered)
        owner = asyncio.current_task()
        assert owner is not None
        cancellation_count = owner.cancelling()
        started = time.monotonic()
        try:
            start = await session.prepare()
        except Exception as error:
            raise RolloutInterrupted(completed, RolloutOperation.PREPARE) from error
        events.emit(SessionPrepared(start.messages, start.options, time.monotonic() - started))
        messages = list(start.messages)
        options = start.options
        replay_prompt_token_ids = start.prompt_token_ids
        completed = replace(completed, messages=tuple(messages))
        trace = _token_trace(limits.trace)
        steps: list[RolloutStep] = []
        assistant_index = None
        stop_reason = MAX_TURNS_STOP_REASON
        spent_tokens = 0
        model_error = None
        deadline = asyncio.timeout(limits.total_turn_timeout)
        try:
            async with deadline:
                for index in range(limits.max_turns):
                    started = time.monotonic()
                    try:
                        async with asyncio.timeout(limits.model_turn_timeout):
                            turn = await self.model(
                                ModelRequest(
                                    tuple(messages),
                                    options,
                                    trace.prefix,
                                    assistant_index,
                                    replay_prompt_token_ids,
                                )
                            )
                    except GenerationLimitReached as limit:
                        stop_reason = LENGTH_STOP_REASON
                        # History resets clear segment evidence, not prior agent actions.
                        if index == 0:
                            return replace(
                                completed,
                                prompt_token_ids=trace.generation_limit_prompt(limit.prompt_token_ids),
                                grade=GradeResult(
                                    Outcome.UNAVAILABLE, None, "Generation limit reached before the first response"
                                ),
                                stop_reason=stop_reason,
                            )
                        messages = list(completed.messages)
                        break
                    except RolloutContractError:
                        raise
                    except Exception as error:
                        model_error = error
                        break
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
                    if replay_prompt_token_ids is not None and turn.prompt_token_ids != replay_prompt_token_ids:
                        raise RolloutContractError("Model transport changed the replay token prefix")
                    replay_prompt_token_ids = None
                    trace.append(turn)
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
                    assistant_index = len(messages)
                    messages.append(turn.message)
                    pending = RolloutStep(
                        turn,
                        Transition(done=True, metrics={"advance_incomplete": 1.0}),
                        trace.response_end,
                        tuple(messages),
                    )
                    # Token evidence belongs to the attempt even if a tool operation fails.
                    completed = RolloutData(
                        task.id,
                        tuple(messages),
                        trace.prompt,
                        trace.response,
                        trace.loss_mask,
                        trace.logprobs,
                        completed.grade,
                        turn.stop_reason,
                        (*steps, pending),
                        pending.transition.metrics,
                    )
                    started = time.monotonic()
                    try:
                        async with asyncio.timeout(limits.tool_turn_timeout):
                            transition = await session.advance(turn)
                    except RolloutContractError:
                        raise
                    except Exception as error:
                        raise RolloutInterrupted(completed, RolloutOperation.ADVANCE) from error
                    trace.check_transition(turn, transition)
                    events.emit(
                        TransitionCompleted(
                            index,
                            transition.done,
                            transition.observations,
                            transition.reset_conversation,
                            time.monotonic() - started,
                        ),
                    )
                    stop_reason = turn.stop_reason
                    if transition.next_options is not None:
                        options = transition.next_options
                    replay_prompt_token_ids = transition.next_prompt_token_ids
                    if transition.reset_conversation is not None and index + 1 < limits.max_turns and not budget_reached:
                        messages = list(transition.reset_conversation)
                        trace.reset()
                        steps.clear()
                        completed = replace(
                            completed,
                            messages=tuple(messages),
                            prompt_token_ids=trace.prompt,
                            response_token_ids=trace.response,
                            loss_mask=trace.loss_mask,
                            logprobs=trace.logprobs,
                            steps=(),
                            metrics={},
                        )
                        assistant_index = None
                        continue
                    steps.append(replace(pending, transition=transition))
                    completed = replace(
                        completed, steps=tuple(steps), stop_reason=stop_reason, metrics=transition.metrics
                    )
                    if transition.done or stop_reason == LENGTH_STOP_REASON:
                        break
                    if budget_reached:
                        stop_reason = BUDGET_STOP_REASON
                        break
                    if index + 1 == limits.max_turns:
                        stop_reason = MAX_TURNS_STOP_REASON
                        break
                    messages.extend(transition.observations)
        except TimeoutError:
            if not deadline.expired():
                raise
            stop_reason = TOTAL_TURN_TIMEOUT_STOP_REASON
            if not completed.steps:
                return replace(
                    completed,
                    grade=GradeResult(Outcome.UNAVAILABLE, None, "Turn deadline expired before the first response"),
                    stop_reason=stop_reason,
                )
            messages = list(completed.messages)
        except asyncio.CancelledError:
            if not attempt.expired() or owner.cancelling() > cancellation_count + 1:
                raise
            raise RolloutInterrupted(completed, RolloutOperation.ATTEMPT) from TimeoutError("Attempt deadline expired")
        if model_error is not None:
            if not completed.steps:
                raise RolloutInterrupted(completed, RolloutOperation.MODEL) from model_error
            messages = list(completed.messages)
        started = time.monotonic()
        try:
            async with asyncio.timeout(limits.verifier_timeout):
                outputs = await _captured_outputs(lowered, machine)
                completed = replace(completed, outputs=outputs)
                grade = await session.grade(tuple(messages), stop_reason=stop_reason, outputs=outputs)
        except asyncio.CancelledError:
            if not attempt.expired() or owner.cancelling() > cancellation_count + 1:
                raise
            raise RolloutInterrupted(completed, RolloutOperation.ATTEMPT) from TimeoutError("Attempt deadline expired")
        except TimeoutError as error:
            completed = replace(
                completed,
                grade=GradeResult(
                    Outcome.INFRA_ERROR, None, "Verifier deadline expired", failure=GradingFailure.TIMEOUT
                ),
            )
            raise RolloutInterrupted(completed, RolloutOperation.GRADE) from error
        except Exception as error:
            raise RolloutInterrupted(completed, RolloutOperation.GRADE) from error
        events.emit(Graded(grade.status, grade.reward, grade.error, grade.failure, time.monotonic() - started))
        completed = replace(completed, grade=grade, stop_reason=stop_reason)
        if model_error is not None:
            raise RolloutInterrupted(completed, RolloutOperation.MODEL) from model_error
        return completed
