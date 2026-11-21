# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The model client: the one owner of retries, holds and backoff, in front of any ``Provider``.

A ``Provider`` streams one request and reports how it failed (``AttemptFailed``). ``ModelClient`` raises
``ContextExhausted`` for a context overflow and ``ModelRequestRejected`` for another rejection a retry cannot fix.
Every other failure (408, 429, 5xx, a 404 ``no route for model``, no endpoint or a refused connection, a stall, a cut,
malformed or aborted stream) is waited out with backoff, polling while the provider reports no workers, for
``unavailable_wait``; then ``ModelUnavailable``. A failure after the provider streamed, other than a stall, restarts
that window. ``max_stalls`` consecutive stalls inside an open tool call (the server buffering degenerate arguments)
return an empty ``finish_reason=length`` completion. ``generation_timeout`` bounds a call until its first failure, so
its expiry (``GenerationTimedOut``) is a long healthy generation, never an outage.
"""

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Protocol

from rigging.timing import ExponentialBackoff

from taskforge.llm.policy import (
    CONTINUE_FINAL_MESSAGE_FIELDS,
    LLMPolicy,
    Message,
    ModelProfile,
    RetryPolicy,
    continuation_messages,
    reasoning_continuation_messages,
)
from taskforge.model.llm import FinishReason, ToolCall, Usage

logger = logging.getLogger(__name__)

RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
"""HTTP statuses a retry can cure: a timeout, rate limiting or a server-side failure."""
REASON_LIMIT = 300


class AttemptOutcome(StrEnum):
    COMPLETED = "completed"
    RETRYABLE_STATUS = "retryable_status"
    ROUTE_MISSING = "route_missing"
    CONTEXT_OVERFLOW = "context_overflow"
    TRANSPORT_ERROR = "transport_error"
    STALLED = "stalled"
    INCOMPLETE_STREAM = "incomplete_stream"
    STREAM_ERROR = "stream_error"
    ABORTED = "aborted"
    REJECTED = "rejected"


@dataclass(frozen=True)
class Attempt:
    """An HTTP request, successful or failed, with the raw SSE ``data:`` payloads it received.

    ``segment`` is the continuation index.
    """

    segment: int
    outcome: AttemptOutcome
    max_tokens: int
    started: float
    duration: float
    http_status: int | None
    detail: str
    events: tuple[str, ...]


@dataclass(frozen=True)
class Segment:
    """One streamed response, with the provider's monotonic timings."""

    content: str
    reasoning: str
    tool_calls: tuple[ToolCall, ...]
    finish_reason: FinishReason
    usage: Usage
    ttft: float | None
    started: float
    finished: float


@dataclass(frozen=True)
class Completion:
    """A logical call, joined from its continuation segments.

    ``ttft`` comes from the successful attempt of the first segment. ``decode_time`` adds up the time
    after the first token in each segment's successful attempt. ``wall_time`` includes backoff.
    """

    content: str
    reasoning: str
    tool_calls: tuple[ToolCall, ...]
    finish_reason: FinishReason
    usage: Usage
    wall_time: float
    ttft: float
    decode_time: float
    continuations: int
    attempts: tuple[Attempt, ...]


class AttemptFailed(Exception):
    """One request failed; ``outcome`` says how, which decides whether the model client waits or raises."""

    def __init__(
        self, outcome: AttemptOutcome, detail: str, http_status: int | None = None, retry_after: float = 0.0
    ) -> None:
        super().__init__(f"{outcome}: {detail}")
        self.outcome = outcome
        self.detail = detail
        self.http_status = http_status
        self.retry_after = retry_after


class ModelRequestRejected(Exception):
    """The server rejected a request for a reason a retry cannot fix; ``attempts`` ends with that request."""

    def __init__(self, status: int, body: str, attempts: Sequence[Attempt]) -> None:
        super().__init__(f"model server rejected request with HTTP {status}: {body}")
        self.status = status
        self.body = body
        self.attempts = tuple(attempts)


class ContextExhausted(ModelRequestRejected):
    """The prompt plus the requested output exceeds the model's context window."""

    def __init__(self, body: str, attempts: Sequence[Attempt]) -> None:
        super().__init__(400, body, attempts)


class ModelUnavailable(Exception):
    """The provider stayed unavailable for the retry policy's window; ``attempts`` holds every attempt made."""

    def __init__(self, message: str, attempts: Sequence[Attempt]) -> None:
        super().__init__(message)
        self.attempts = tuple(attempts)


class GenerationTimedOut(Exception):
    """A call generated for ``RetryPolicy.generation_timeout`` seconds without failing or finishing.

    Evidence about the request, not the provider (production 2026-11-19: three solver turns of 70 minutes each).
    """


class Provider(Protocol):
    """Where and how a model is reached; ``GlmProvider`` is the library default.

    ``stream`` sends one chat-completions ``body`` (every field but the streaming options), appends each raw ``data:``
    payload to ``events`` and raises ``AttemptFailed`` on failure. ``no_workers`` is True only on positive evidence that
    the service has no workers for ``model``.
    """

    @property
    def model(self) -> str: ...
    async def stream(self, body: Mapping[str, Any], events: list[str]) -> Segment: ...
    async def no_workers(self) -> bool: ...


class _Degenerate(Exception):
    pass


def degenerate_stall(failure: AttemptFailed, events: Sequence[str]) -> bool:
    """Whether a stall came inside an open tool call: the last chunk carrying generated text is a tool-call delta."""
    if failure.outcome is not AttemptOutcome.STALLED:
        return False
    for event in reversed(events):
        deltas = [choice.get("delta") or {} for choice in json.loads(event).get("choices") or ()]
        if any(delta.get("tool_calls") for delta in deltas):
            return True
        if any(delta.get("content") or delta.get("reasoning") for delta in deltas):
            return False
    return False


class _Call:
    """A logical call's attempts, availability window, consecutive degenerate stalls and generation bound."""

    def __init__(self, retry: RetryPolicy, clock: Callable[[], float]) -> None:
        self.retry, self.clock, self.stalls, self.started = retry, clock, 0, clock()
        self.attempts: list[Attempt] = []
        self.bound = asyncio.timeout(retry.generation_timeout)
        self.restart()

    def restart(self) -> None:
        retry = self.retry
        self.deadline = self.clock() + retry.unavailable_wait
        self.backoff = ExponentialBackoff(retry.backoff_initial, retry.backoff_maximum, retry.backoff_factor, 0.0)

    def remaining(self) -> float:
        return self.deadline - self.clock()


class ModelClient:
    """Logical calls to ``provider`` for ``profile``. ``clock`` and ``sleep`` time the availability window."""

    def __init__(
        self,
        provider: Provider,
        profile: ModelProfile,
        retry: RetryPolicy,
        *,
        role: str,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.provider, self.profile, self.retry, self.role = provider, profile, retry, role
        self.clock, self.sleep = clock, sleep

    @property
    def model(self) -> str:
        return self.provider.model

    def _name(self) -> str:
        return f"MODEL role={self.role} model={self.model}"

    async def complete(
        self,
        messages: Sequence[Message],
        policy: LLMPolicy,
        request_fields: Mapping[str, object] | None = None,
    ) -> Completion:
        """Run a logical call, continuing after ``finish_reason == "length"`` as ``policy`` allows.

        ``request_fields`` adds body fields such as ``tools``. When a continuation overflows the context window, the
        call ends with the segments gathered so far and ``finish_reason == "length"``.
        """
        call = _Call(self.retry, self.clock)
        try:
            async with call.bound:
                return await self._complete(messages, policy, request_fields or {}, call)
        except _Degenerate:
            logger.warning("%s: %d stalls inside a tool call; degenerate output", self._name(), call.stalls)
            wall_time = self.clock() - call.started
            return Completion("", "", (), FinishReason.LENGTH, Usage(), wall_time, 0.0, 0.0, 0, tuple(call.attempts))
        except TimeoutError as error:
            if not call.bound.expired():
                raise
            raise GenerationTimedOut(
                f"{self._name()} generated for {self.retry.generation_timeout}s without finishing"
            ) from error

    async def _complete(
        self,
        messages: Sequence[Message],
        policy: LLMPolicy,
        request_fields: Mapping[str, object],
        call: _Call,
    ) -> Completion:
        segments: list[Segment] = []
        conversation, fields = list(messages), request_fields
        while True:
            try:
                result = await self._segment(conversation, policy, fields, len(segments), call)
            except ContextExhausted:
                if not segments:
                    raise
                logger.warning("%s: context is full after %d segments; partial output", self._name(), len(segments))
                break
            segments.append(result)
            if (
                result.finish_reason is not FinishReason.LENGTH
                or result.tool_calls
                or len(segments) > policy.max_continuations
            ):
                break
            if content := "".join(s.content for s in segments):
                conversation, fields = continuation_messages(messages, content), request_fields
            else:
                reasoning = "".join(s.reasoning for s in segments)
                conversation = reasoning_continuation_messages(messages, reasoning)
                fields = {**request_fields, **CONTINUE_FINAL_MESSAGE_FIELDS}
        return Completion(
            content="".join(s.content for s in segments),
            reasoning="".join(s.reasoning for s in segments),
            tool_calls=segments[-1].tool_calls,
            finish_reason=segments[-1].finish_reason,
            usage=sum((s.usage for s in segments[1:]), segments[0].usage),
            wall_time=self.clock() - call.started,
            ttft=segments[0].ttft or 0.0,
            decode_time=sum(s.finished - s.started - (s.ttft or 0.0) for s in segments),
            continuations=len(segments) - 1,
            attempts=tuple(call.attempts),
        )

    async def _segment(
        self,
        messages: Sequence[Message],
        policy: LLMPolicy,
        request_fields: Mapping[str, object],
        segment: int,
        call: _Call,
    ) -> Segment:
        """Send one segment's request until an attempt succeeds, recording every attempt in ``call``."""
        output = self.profile.max_output_tokens
        limit = None if output is None else min(policy.max_tokens, output)
        body = request_body(self.model, messages, limit, policy, request_fields)
        while True:
            events: list[str] = []
            started_wall, started = time.time(), time.monotonic()
            try:
                result = await self.provider.stream(body, events)
            except AttemptFailed as failure:
                status, detail = failure.http_status, failure.detail
                duration = time.monotonic() - started
                call.attempts.append(
                    Attempt(segment, failure.outcome, limit or 0, started_wall, duration, status, detail, tuple(events))
                )
                call.bound.reschedule(None)  # only a call that has not failed is a generation
                if failure.outcome is AttemptOutcome.CONTEXT_OVERFLOW:
                    raise ContextExhausted(detail, call.attempts) from failure
                if failure.outcome is AttemptOutcome.REJECTED:
                    assert status is not None
                    raise ModelRequestRejected(status, detail, call.attempts) from failure
                await self._wait(failure, events, call)
                continue
            duration = time.monotonic() - started
            call.attempts.append(
                Attempt(segment, AttemptOutcome.COMPLETED, limit or 0, started_wall, duration, 200, "", tuple(events))
            )
            call.stalls = 0
            # vLLM's GLM tool parser reports a reply cut off inside a tool call as ``tool_calls`` (measured live).
            if (
                limit is not None
                and result.finish_reason is FinishReason.TOOL_CALLS
                and result.usage.completion_tokens >= limit
            ):
                return replace(result, finish_reason=FinishReason.LENGTH)
            return result

    async def _wait(self, failure: AttemptFailed, events: Sequence[str], call: _Call) -> None:
        """Back off after an unavailable attempt, then poll while the provider reports no workers.

        Raises:
            ModelUnavailable: The window closed first.
            _Degenerate: The call stalled inside a tool call ``max_stalls`` times in a row.
        """
        if degenerate_stall(failure, events):
            call.stalls += 1
            if call.stalls >= self.retry.max_stalls:
                raise _Degenerate
        else:
            call.stalls = 0
            if events and failure.outcome is not AttemptOutcome.STALLED:
                call.restart()  # the provider streamed before the stream was cut, so it was available
        backoff = call.backoff
        delay = max(backoff.next_interval(), failure.retry_after)
        waited = False
        while (remaining := call.remaining()) > 0:
            if waited:
                if not await self.provider.no_workers():
                    return
                delay = backoff.next_interval()
            wait = min(delay, remaining)
            logger.warning(
                "%s unavailable; %.1fs left in availability window, retrying in %.1fs: %s",
                self._name(),
                remaining,
                wait,
                str(failure)[:REASON_LIMIT],
            )
            await self.sleep(wait)
            waited = True
        raise ModelUnavailable(f"{self._name()} unavailable for {self.retry.unavailable_wait}s", call.attempts)


def request_body(
    model: str,
    messages: Sequence[Message],
    max_tokens: int | None,
    policy: LLMPolicy,
    request_fields: Mapping[str, object],
) -> dict[str, Any]:
    """The chat-completions fields of one request, apart from streaming; ``max_tokens`` None sends no limit."""
    if max_tokens is None:
        request_fields = {
            key: value for key, value in request_fields.items() if key not in ("max_tokens", "max_completion_tokens")
        }
    return {
        "model": model,
        "messages": list(messages),
        **({"max_tokens": max_tokens} if max_tokens is not None else {}),
        **policy.sampling_params(),
        "chat_template_kwargs": policy.template_kwargs(),
        **request_fields,
    }
