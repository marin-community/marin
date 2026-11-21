# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The model client: the one owner of retries, holds and backoff, in front of any ``Provider``.

A solver trial is charged as failed only on positive evidence that the model itself ran out: generation sustained past
the bound, or a recognised context-limit rejection. Anything the fleet, the relay, the transport or local admission may
have caused is unavailable evidence: no failed trial is recorded, and the stage waits, restarts or holds as for an
outage. When in doubt it is not the model's fault, because a fleet fault scored against the model corrupts solve rates
and can accept a task at the wrong band.

A ``Provider`` streams one request, reporting admission and each payload to the attempt's ``Progress``, and raises
``AttemptFailed`` saying how it failed. ``ModelClient`` raises ``ContextExhausted`` for a context overflow and
``ModelRequestRejected`` for another rejection a retry cannot fix. Every other failure (408, 429, any 5xx, a 404 ``no
route for model``, no endpoint, a refused connection or a full local pool, a stall, a cut, malformed or aborted
stream) is waited out with backoff, polling while the provider reports no workers, for ``unavailable_wait``; then
``ModelUnavailable``. The first failure of a call after its attempt generated text, other than a stall, restarts that
window; later ones count against it, so repeated partial answers cannot hold a call open. A health read is bounded by
the window and its expiry is unknown availability. Silence, including a stall inside an open tool call, is never
evidence about the model.

Progress is a payload carrying nonempty generated text (``progressed``); the bound, stall detection and the window
renewal all use it. ``generation_timeout`` bounds the seconds a call spends generating: from the first generated token
of each attempt, adding the call's completed segments, so neither admission nor an earlier fault counts. Crossing the
bound is not itself evidence: the call ends as ``GenerationTimedOut`` only when the same stream delivers progress again
``CONFIRMATION_INTERVAL`` after its first progress past the bound, with no finish, abort, cut or stall in between. The
cutoff consults no pool state.
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

RETRYABLE_STATUSES = frozenset({408, 429})
"""HTTP statuses below 500 that a retry can cure: a timeout or rate limiting. Every 5xx is an outage too."""
TOKEN_LIMIT_FIELDS = ("max_tokens", "max_completion_tokens")
REASON_LIMIT = 300
CONFIRMATION_INTERVAL = 1.0
"""Seconds after its first progress past the generation bound that a stream must progress again to end the call there.
A live stream progresses every decode step (tens of milliseconds), so confirming costs about a second of the 60-second
turn margin. Output queued on the path when its worker exited can still arrive for longer than this; that tail is
charged (a known limit: README, Models and providers)."""


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
    """A call generated for ``RetryPolicy.generation_timeout`` seconds without finishing.

    Evidence about the request, not the provider (production 2026-11-19: three solver turns of 70 minutes each).
    """


class _BoundReached(Exception):
    pass


def progressed(chunk: Mapping[str, Any]) -> bool:
    """Whether a chat-completions chunk carries nonempty generated text: content, reasoning (vLLM's ``reasoning``), a
    function name or an argument fragment. A role, an index, an id or an empty string is an envelope."""
    deltas = [choice.get("delta") or {} for choice in chunk.get("choices") or ()]
    calls = [call.get("function") or {} for delta in deltas for call in delta.get("tool_calls") or ()]
    return any(delta.get("content") or delta.get("reasoning") for delta in deltas) or any(
        function.get("name") or function.get("arguments") for function in calls
    )


class Progress:
    """One attempt as its provider reports it: ``admit`` when the server answers, ``event`` for each ``data:`` payload.

    ``event`` raises once generation past the call's bound is confirmed; a provider lets that propagate and abandons the
    stream.
    """

    def __init__(self, clock: Callable[[], float], remaining: float | None) -> None:
        self.clock, self.remaining = clock, remaining
        self.events: list[str] = []
        self.admitted: float | None = None
        self.first_token: float | None = None
        self.last_token: float | None = None
        self.past_bound: float | None = None

    def admit(self) -> None:
        if self.admitted is None:
            self.admitted = self.clock()

    def event(self, payload: str) -> bool:
        """Record a JSON ``payload`` and return whether it is progress, the only evidence a stream is generating."""
        self.admit()
        self.events.append(payload)
        chunk = json.loads(payload)
        if not progressed(chunk):
            return False
        self.last_token = now = self.clock()
        if self.first_token is None:
            self.first_token = now
        if self.remaining is None or now - self.first_token < self.remaining:
            return True
        # A finishing chunk ends the stream: its finish reason, an abort included, decides the attempt, not the bound.
        if any(choice.get("finish_reason") for choice in chunk.get("choices") or ()):
            return True
        if self.past_bound is None:
            self.past_bound = now
        elif now - self.past_bound >= CONFIRMATION_INTERVAL:
            raise _BoundReached
        return True

    def generated(self) -> float:
        """Seconds from the attempt's first generated token to its last."""
        if self.first_token is None or self.last_token is None:
            return 0.0
        return self.last_token - self.first_token


class Provider(Protocol):
    """Where and how a model is reached; ``GlmProvider`` is the library default. Subclass it.

    ``stream`` sends one chat-completions ``body`` (every field but the streaming options), reports each payload to
    ``progress``, times stalls by the payloads it says progressed, and raises ``AttemptFailed`` on failure.
    ``no_workers`` is True only on positive evidence that the service has no workers for ``model``; the default has none.
    """

    @property
    def model(self) -> str: ...
    async def stream(self, body: Mapping[str, Any], progress: Progress) -> Segment: ...

    async def no_workers(self) -> bool:
        return False


class _Call:
    """A logical call's attempts, availability window (renewed once) and seconds its completed segments generated."""

    def __init__(self, retry: RetryPolicy, clock: Callable[[], float]) -> None:
        self.retry, self.clock, self.started, self.generated = retry, clock, clock(), 0.0
        self.attempts: list[Attempt] = []
        self.renewable = True
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
        return await self._complete(messages, policy, request_fields or {}, _Call(self.retry, self.clock))

    async def _complete(
        self,
        messages: Sequence[Message],
        policy: LLMPolicy,
        request_fields: Mapping[str, object],
        call: _Call,
    ) -> Completion:
        # A cut segment resends only itself, continuing from the segments already complete; n4 restarted the call.
        # With a deterministic prefix both record the same answer, while a resampled prefix now keeps the first one.
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
        body = request_body(self.model, messages, policy, request_fields, self.profile.max_output_tokens)
        limit = min((body[key] for key in TOKEN_LIMIT_FIELDS if key in body), default=None)
        bound = self.retry.generation_timeout
        while True:
            progress = Progress(self.clock, None if bound is None else bound - call.generated)
            started_wall, started = time.time(), time.monotonic()
            result, failure = None, None
            try:
                result = await self.provider.stream(body, progress)
            except _BoundReached as reached:
                raise GenerationTimedOut(f"{self._name()} generated for {bound}s without finishing") from reached
            except AttemptFailed as error:
                failure = error
            outcome, status, detail = (
                (failure.outcome, failure.http_status, failure.detail)
                if failure
                else (AttemptOutcome.COMPLETED, 200, "")
            )
            duration, events = time.monotonic() - started, tuple(progress.events)
            call.attempts.append(Attempt(segment, outcome, limit or 0, started_wall, duration, status, detail, events))
            if result is not None:
                call.generated += progress.generated()
                # vLLM's GLM tool parser reports a reply cut off inside a tool call as ``tool_calls`` (measured live).
                if limit is not None and result.finish_reason is FinishReason.TOOL_CALLS:
                    if result.usage.completion_tokens >= limit:
                        return replace(result, finish_reason=FinishReason.LENGTH)
                return result
            assert failure is not None
            if failure.outcome is AttemptOutcome.CONTEXT_OVERFLOW:
                raise ContextExhausted(detail, call.attempts) from failure
            if failure.outcome is AttemptOutcome.REJECTED:
                assert status is not None
                raise ModelRequestRejected(status, detail, call.attempts) from failure
            await self._wait(failure, progress, call)

    async def _wait(self, failure: AttemptFailed, progress: Progress, call: _Call) -> None:
        """Back off after an unavailable attempt, then poll while the provider reports no workers.

        Raises:
            ModelUnavailable: The window closed first.
        """
        if call.renewable and progress.first_token is not None and failure.outcome is not AttemptOutcome.STALLED:
            call.renewable = False
            call.restart()  # the attempt generated text before it was cut, so the provider was serving
        backoff = call.backoff
        delay = max(backoff.next_interval(), failure.retry_after)
        waited = False
        while (remaining := call.remaining()) > 0:
            if waited:
                try:
                    empty = await asyncio.wait_for(self.provider.no_workers(), remaining)
                except TimeoutError:
                    empty = False  # a health read that outlasts the window is unknown availability
                if call.remaining() <= 0:
                    break
                if not empty:
                    return
                delay = backoff.next_interval()
            wait = min(delay, remaining)
            logger.warning(
                "%s unavailable (admitted=%s); %.1fs left in availability window, retrying in %.1fs: %s",
                self._name(),
                progress.admitted is not None,
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
    policy: LLMPolicy,
    request_fields: Mapping[str, object],
    max_output_tokens: int | None,
) -> dict[str, Any]:
    """The chat-completions fields of one request, apart from streaming.

    ``max_output_tokens`` caps every token-limit field, whether ``policy`` or ``request_fields`` set it; None sends none.
    """
    body: dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "max_tokens": policy.max_tokens,
        **policy.sampling_params(),
        "chat_template_kwargs": policy.template_kwargs(),
        **request_fields,
    }
    for key in (key for key in TOKEN_LIMIT_FIELDS if key in body):
        if max_output_tokens is None:
            del body[key]
        else:
            body[key] = min(body[key], max_output_tokens)
    return body
