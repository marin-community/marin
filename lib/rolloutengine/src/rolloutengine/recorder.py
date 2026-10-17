# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Record one attempt's model turns under its limits, and run a harness without the engine."""

import asyncio
import time
import uuid
from collections.abc import Callable, Coroutine, Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass, replace
from typing import Any, NoReturn

from shellbox.machine import Machine
from taskcompendium.models import TaskSpec

from rolloutengine.cleanup import _Cleanup
from rolloutengine.contracts import (
    BUDGET_STOP_REASON,
    LENGTH_STOP_REASON,
    MAX_TURNS_STOP_REASON,
    TOTAL_TURN_TIMEOUT_STOP_REASON,
    Continuation,
    ConversationRecord,
    ConversationStarted,
    EventSink,
    GenerationLimitReached,
    LimitReached,
    Model,
    ModelRequest,
    ModelTurn,
    ModelTurnCompleted,
    Prompt,
    Rollout,
    RolloutContractError,
    RolloutEnded,
    RolloutEventPayload,
    RolloutFailure,
    RolloutInterrupted,
    RolloutOperation,
    RolloutStep,
    Transition,
    TransitionCompleted,
)
from rolloutengine.harness import Harness, HarnessContext
from rolloutengine.spec import TRACE_METRIC, RolloutLimits, TraceMode
from rolloutengine.trace import _token_trace

ADVANCE_INCOMPLETE = "advance_incomplete"
"""``Transition.metrics`` key of a turn that the harness did not finish handling; training must exclude its step."""


class _NoEvents:
    def emit(self, payload: RolloutEventPayload) -> None:
        pass


NO_EVENTS: EventSink = _NoEvents()


class _Conversation:
    """One conversation as it grows: its chat through the last response, its token trace, and its steps."""

    def __init__(self, prompt: Prompt, mode: TraceMode):
        self.mode = mode
        self.messages: list[dict[str, Any]] = list(prompt.messages)
        self.options = prompt.options
        self.prompt_token_ids = prompt.token_ids
        self.trace = _token_trace(mode)
        self.steps: list[RolloutStep] = []
        self.pending: RolloutStep | None = None
        self.assistant_index: int | None = None
        self.unserved_prompt: tuple[int, ...] | None = None

    @property
    def has_turn(self) -> bool:
        return bool(self.steps) or self.pending is not None

    def request(self, observations: tuple[dict[str, Any], ...], token_ids: tuple[int, ...] | None) -> ModelRequest:
        messages = (*self.messages, *observations)
        if token_ids is not None:
            return ModelRequest(messages, self.options, prompt_token_ids=token_ids)
        # A chat endpoint re-tokenizes every request, so a messages trace never continues served tokens.
        if self.assistant_index is None or self.mode == TraceMode.MESSAGES:
            return ModelRequest(messages, self.options)
        return ModelRequest(messages, self.options, Continuation(self.trace.prefix, self.assistant_index))

    def add_turn(self, turn: ModelTurn) -> None:
        # The last step's transition holds the observations and the pinned tokens of the request that this turn answers.
        last = self.steps[-1].transition if self.steps else None
        pinned = self.prompt_token_ids if last is None else last.token_ids
        if pinned is not None and turn.prompt_token_ids != pinned:
            raise RolloutContractError("Model transport changed the replay token prefix")
        self.trace.append(turn)
        if last is not None:
            self.messages.extend(last.observations)
        self.assistant_index = len(self.messages)
        self.messages.append(turn.message)
        self.pending = RolloutStep(turn, Transition(done=True), self.trace.response_end, tuple(self.messages))

    def complete(self, transition: Transition) -> None:
        assert self.pending is not None
        self.steps.append(replace(self.pending, transition=transition))
        self.pending = None

    def record(self, incomplete: Transition) -> ConversationRecord:
        steps = tuple(self.steps)
        if self.pending is not None:
            # Token evidence belongs to the attempt even if the harness did not finish handling the turn.
            steps = (*steps, replace(self.pending, transition=incomplete))
        prompt = self.trace.prompt if self.unserved_prompt is None else self.unserved_prompt
        return ConversationRecord(
            tuple(self.messages), prompt, self.trace.response, self.trace.loss_mask, self.trace.logprobs, steps
        )


@dataclass(frozen=True)
class _CleanupScope:
    """``HarnessCleanup`` on an exit stack: each close runs within the cleanup deadline, and failures are recorded."""

    stack: AsyncExitStack
    cleanup: _Cleanup

    def push(self, name: str, close: Callable[[], Coroutine[Any, Any, None]]) -> None:
        self.stack.push_async_callback(self.cleanup.run, name, close)


@dataclass(frozen=True)
class _Deadlines:
    total: asyncio.Timeout
    tool: asyncio.Timeout


@dataclass(frozen=True)
class _Request:
    """A request that the harness sent and that the recorder did not record yet.

    ``conversation`` is the new conversation of a ``start``. ``waited`` is the time from the last turn to the
    request, that is, the time that the harness used on that turn.
    """

    observations: tuple[dict[str, Any], ...]
    token_ids: tuple[int, ...] | None
    conversation: _Conversation | None
    waited: float


class _Recorder:
    """The ``RolloutRecorder`` of one attempt: send the harness's requests, record them, and apply the limits.

    It is outside the attempt deadline, so an interrupted attempt keeps all that it recorded. A request
    changes the record only when the model answers or fails. Thus a request that is cancelled while the
    model generates changes nothing, and the harness can send it again. When the harness ends or a turn
    deadline expires, the record is final: the recorder stops an open model call, ignores its result, and
    records a sent request that has no answer. Final observations replace such a request.
    """

    def __init__(self, model: Model, limits: RolloutLimits, mode: TraceMode, events: EventSink):
        self.model = model
        self.limits = limits
        self.mode = mode
        self.events = events
        self.earlier: list[ConversationRecord] = []
        self.conversation: _Conversation | None = None
        self.request: _Request | None = None
        self.turns = 0
        self.spent_tokens = 0
        self.budget_reached = False
        self.stop_reason = "error"
        self.stopped = False
        self.model_error: Exception | None = None
        self.no_response_reason: str | None = None
        self.model_call: asyncio.Timeout | None = None
        self.step_observations: tuple[dict[str, Any], ...] = ()
        self.step_data: object = None
        self.step_metrics: dict[str, Any] = {}
        self.turn_ended = 0.0
        self.deadlines: _Deadlines | None = None

    @property
    def has_response(self) -> bool:
        """Whether a conversation has a model turn. The engine grades only then."""
        conversation = self.conversation
        return (conversation is not None and conversation.has_turn) or any(record.steps for record in self.earlier)

    def graded_messages(self) -> tuple[dict[str, Any], ...] | None:
        """The chat of the last conversation that has a model turn, or ``None``."""
        conversation = self.conversation
        if conversation is not None and conversation.has_turn:
            return tuple(conversation.messages)
        return next((record.messages for record in reversed(self.earlier) if record.steps), None)

    async def start(self, prompt: Prompt) -> ModelTurn:
        await self._enter()
        current = self.conversation
        if current is not None and (self.budget_reached or self.turns >= self.limits.max_turns):
            # The turn and token limits end the attempt before a new conversation. A length stop does not.
            self._stop_at_limit(self.step_observations, None)
        deadlines = self.deadlines
        assert deadlines is not None
        if current is None and deadlines.total.when() is None and self.limits.total_turn_timeout is not None:
            deadlines.total.reschedule(asyncio.get_running_loop().time() + self.limits.total_turn_timeout)
        conversation = _Conversation(prompt, self.mode)
        request = _Request(self.step_observations, None, conversation, self._waited())
        return await self._send(request, conversation.request((), prompt.token_ids))

    async def reply(
        self, observations: tuple[dict[str, Any], ...], *, token_ids: tuple[int, ...] | None = None
    ) -> ModelTurn:
        await self._enter()
        conversation = self.conversation
        if conversation is None or conversation.pending is None:
            raise RuntimeError("A harness replies only after a model turn")
        if self.step_observations:
            raise RuntimeError("A turn has recorded final observations, so the harness cannot reply to it")
        if self.stop_reason == LENGTH_STOP_REASON or self.budget_reached or self.turns >= self.limits.max_turns:
            self._stop_at_limit(observations, token_ids)
        request = _Request(observations, token_ids, None, self._waited())
        return await self._send(request, conversation.request(observations, token_ids))

    def record(
        self,
        *,
        observations: tuple[dict[str, Any], ...] = (),
        data: object = None,
        metrics: Mapping[str, Any] | None = None,
    ) -> None:
        if self._ended():
            raise RolloutEnded("The harness ended, or a turn deadline expired")
        if self.model_call is not None:
            raise RuntimeError("A harness records step data only between model requests")
        if self.conversation is None or self.conversation.pending is None:
            raise RuntimeError("A harness records step data only after a model turn")
        if observations:
            self.request = None
            self.step_observations = observations
        if data is not None:
            self.step_data = data
        if metrics:
            self.step_metrics.update(metrics)

    async def _enter(self) -> None:
        refusal = self._refusal()
        if refusal is not None:
            # Yield first, so that a harness that catches the refusal and asks again cannot block a deadline.
            await asyncio.sleep(0)
            raise refusal
        if self.model_call is not None:
            raise RuntimeError("A harness sends only one model request at a time")

    def _refusal(self) -> RolloutEnded | None:
        """Why the attempt takes no more requests, or ``None`` if it takes them."""
        if self.stopped:
            return LimitReached(self.stop_reason)
        if self.model_error is not None:
            return RolloutEnded("The rollout ended after a failed model request")
        if self._ended():
            return RolloutEnded("The harness ended, or a turn deadline expired")
        return None

    def _ended(self) -> bool:
        deadlines = self.deadlines
        return deadlines is None or deadlines.tool.expired() or deadlines.total.expired()

    def _stop_at_limit(self, observations: tuple[dict[str, Any], ...], token_ids: tuple[int, ...] | None) -> NoReturn:
        """Refuse the request at a limit, and record the observations as the end of the last turn's step."""
        self._end_step(done=False, observations=observations, token_ids=token_ids, waited=self._waited())
        self._set_tool_deadline(None)
        if self.stop_reason != LENGTH_STOP_REASON:
            self.stop_reason = BUDGET_STOP_REASON if self.budget_reached else MAX_TURNS_STOP_REASON
        self.stopped = True
        raise LimitReached(self.stop_reason)

    async def _send(self, request: _Request, model_request: ModelRequest) -> ModelTurn:
        self.request = request
        self._set_tool_deadline(None)
        started = time.monotonic()
        task = asyncio.current_task()
        assert task is not None
        cancelling = task.cancelling()
        call = self.model_call = asyncio.timeout(self.limits.model_turn_timeout)
        try:
            try:
                async with call:
                    outcome: ModelTurn | Exception = await self.model(model_request)
            except Exception as error:
                outcome = error
            if task.cancelling() > cancelling:
                # The transport caught the cancellation of the caller. The request is cancelled all the same.
                raise asyncio.CancelledError
        except asyncio.CancelledError:
            # The record does not change. After a turn, the harness gets a new tool deadline to send a request again.
            if self.turns:
                self._set_tool_deadline(self.limits.tool_turn_timeout)
            raise
        finally:
            self.model_call = None
        if call.expired():
            # Also when the transport caught the deadline's cancellation and answered or failed in another way.
            outcome = TimeoutError("The model turn deadline expired")
        if self._ended():
            raise RolloutEnded("The harness ended, or a turn deadline expired, while the model generated")
        self._record_request(request)
        conversation = self.conversation
        assert conversation is not None
        if isinstance(outcome, GenerationLimitReached):
            if not conversation.has_turn:
                conversation.unserved_prompt = outcome.prompt_token_ids
            if not self.has_response:
                self.no_response_reason = "Generation limit reached before the first response"
            self.stop_reason = LENGTH_STOP_REASON
            self.stopped = True
            raise LimitReached(LENGTH_STOP_REASON) from outcome
        if isinstance(outcome, Exception):
            self.model_error = outcome
            raise RolloutEnded("The model request failed") from outcome
        turn = outcome
        try:
            if self.limits.max_total_tokens is not None and turn.usage is None and not turn.prompt_token_ids:
                raise RolloutContractError("A token budget needs ModelTurn.usage or prompt token IDs from the model")
            conversation.add_turn(turn)
        except RolloutContractError as error:
            self.model_error = error
            raise RolloutEnded("The model turn broke the token contract") from error
        self.spent_tokens += turn.request_tokens
        # Without a new conversation, the next prompt holds at least this request's prompt and response,
        # so stop when that prompt alone would reach the budget.
        max_total = self.limits.max_total_tokens
        self.budget_reached = max_total is not None and self.spent_tokens + turn.request_tokens >= max_total
        self.stop_reason = turn.stop_reason
        self.step_observations, self.step_data, self.step_metrics = (), None, {}
        self.events.emit(
            ModelTurnCompleted(
                self.turns,
                turn.message,
                turn.stop_reason,
                len(turn.prompt_token_ids),
                len(turn.response_token_ids),
                turn.usage,
                turn.metadata,
                time.monotonic() - started,
            )
        )
        self.turns += 1
        self.turn_ended = time.monotonic()
        self._set_tool_deadline(self.limits.tool_turn_timeout)
        return turn

    def _record_request(self, request: _Request) -> None:
        """Record a sent request: it ends the previous step, and a ``start`` begins its conversation."""
        self.request = None
        new = request.conversation
        if new is None:
            self._end_step(
                done=False, observations=request.observations, token_ids=request.token_ids, waited=request.waited
            )
            return
        if self.conversation is not None:
            self._end_step(done=False, observations=request.observations, token_ids=None, waited=request.waited)
            self.earlier.append(self.conversation.record(self._incomplete()))
        self.conversation = new
        self.events.emit(ConversationStarted(tuple(new.messages), new.options, request.waited))

    def _end_step(
        self, *, done: bool, observations: tuple[dict[str, Any], ...], token_ids: tuple[int, ...] | None, waited: float
    ) -> None:
        conversation = self.conversation
        if conversation is None or conversation.pending is None:
            return
        transition = Transition(done, observations, token_ids, dict(self.step_metrics), self.step_data)
        conversation.complete(transition)
        self.events.emit(TransitionCompleted(self.turns - 1, done, observations, waited))

    def _waited(self) -> float:
        return time.monotonic() - self.turn_ended

    def _incomplete(self) -> Transition:
        metrics = {**self.step_metrics, ADVANCE_INCOMPLETE: 1.0}
        return Transition(True, self.step_observations, None, metrics, self.step_data)

    def _set_tool_deadline(self, delay: float | None) -> None:
        if self.deadlines is None or self.deadlines.tool.expired():
            return
        when = None if delay is None else asyncio.get_running_loop().time() + delay
        self.deadlines.tool.reschedule(when)

    def close(self) -> None:
        """The harness ended: stop an open model call, and record a sent request that has no answer."""
        self.deadlines = None
        call = self.model_call
        if call is not None and not call.expired():
            # Another task, such as an agent's HTTP handler, waits for the model. The record does not take its answer.
            call.reschedule(asyncio.get_running_loop().time())
        if self.request is not None:
            self._record_request(self.request)

    def finish(self) -> None:
        self._end_step(done=True, observations=self.step_observations, token_ids=None, waited=self._waited())

    def stop_at_turn_deadline(self) -> None:
        if not self.stopped:
            self.stop_reason = TOTAL_TURN_TIMEOUT_STOP_REASON
        self.stopped = True
        if not self.has_response:
            self.no_response_reason = "Turn deadline expired before the first response"

    def rollout(self, run_id: str, outputs: dict[str, bytes]) -> Rollout:
        conversations = list(self.earlier)
        if self.conversation is not None:
            conversations.append(self.conversation.record(self._incomplete()))
        steps = conversations[-1].steps if conversations else ()
        metrics = dict(steps[-1].transition.metrics) if steps else {}
        if self.mode == TraceMode.MESSAGES:
            metrics[TRACE_METRIC] = TraceMode.MESSAGES.value
        return Rollout(run_id, tuple(conversations), self.stop_reason, metrics, outputs)


async def _run_harness(
    harness: Harness, context: HarnessContext, recorder: _Recorder
) -> tuple[RolloutOperation, BaseException] | None:
    """Run the harness under the turn deadlines, and return its failure. A token-contract violation propagates."""
    failure: Exception | None = None
    recorder.turn_ended = time.monotonic()
    # The recorder sets both deadlines: the total one at the first request, the tool one after each turn.
    total, tool = asyncio.timeout(None), asyncio.timeout(None)
    try:
        async with total, tool:
            recorder.deadlines = _Deadlines(total, tool)
            try:
                await harness.run(context, recorder)
            except Exception as error:
                # The harness's own error, also when a deadline expired before it.
                failure = error
            finally:
                # The recorder refuses later requests, and no rescheduled deadline fires.
                recorder.close()
    except TimeoutError:
        pass
    refused = failure is None or isinstance(failure, RolloutEnded)
    if tool.expired() and refused:
        failure, refused = TimeoutError("Tool-turn deadline expired"), False
    if isinstance(recorder.model_error, RolloutContractError):
        raise recorder.model_error
    if isinstance(failure, RolloutContractError):
        raise failure
    if recorder.model_error is not None and refused:
        return RolloutOperation.MODEL, recorder.model_error
    if refused and total.expired():
        recorder.stop_at_turn_deadline()
        return None
    if failure is None or (refused and recorder.stopped):
        if recorder.conversation is None:
            return RolloutOperation.PREPARE, RuntimeError("The harness returned before its first model request")
        recorder.finish()
        return None
    return (RolloutOperation.ADVANCE if recorder.turns else RolloutOperation.PREPARE), failure


async def harness_rollout(
    harness: Harness,
    task: TaskSpec,
    model: Model,
    *,
    limits: RolloutLimits,
    machine: Machine | None = None,
    trace: TraceMode = TraceMode.EXACT_TOKENS,
    events: EventSink = NO_EVENTS,
    run_id: str | None = None,
) -> Rollout:
    """Run one harness on one task with one model, without the engine: no machine startup and no grading.

    ``machine`` is a task machine that is ready, for example from ``task_machine``. The turn limits and
    ``total_turn_timeout`` apply; the attempt, verifier, and machine deadlines belong to the engine. The
    harness's resources close at the end, and ``Rollout.metrics`` counts the closes that failed. An
    attempt that fails raises ``RolloutInterrupted`` with the rollout so far.
    """
    harness.check(task, limits)
    run_id = uuid.uuid4().hex if run_id is None else run_id
    recorder = _Recorder(model, limits, trace, events)
    cleanup = _Cleanup(limits.cleanup_timeout)
    async with AsyncExitStack() as resources:
        context = HarnessContext(task, machine, events, _CleanupScope(resources, cleanup), run_id)
        failure = await _run_harness(harness, context, recorder)
    rollout = recorder.rollout(run_id, {})
    if cleanup.errors:
        rollout = replace(rollout, metrics={**rollout.metrics, "cleanup_error_count": float(len(cleanup.errors))})
    if failure is None:
        return rollout
    operation, cause = failure
    rollout = replace(rollout, failure=RolloutFailure(type(cause).__name__, {"operation": operation.value}))
    raise RolloutInterrupted(rollout, operation) from cause
