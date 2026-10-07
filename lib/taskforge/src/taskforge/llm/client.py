# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The single GLM-5.3 transport: streaming OpenAI-compatible chat completions over httpx.

Every request streams, so a stall (no token for ``LLMPolicy.stall_timeout`` seconds) is detected
without a wall-clock cap. Retryable failures (408, 429, 5xx, transport errors, stalls, truncated,
malformed, or aborted streams) spend one of ``max_attempts`` with exponential backoff. A missing
relay route (404 whose body names the route), or a retryable failure while ``/health`` answers 200
and reports zero workers for the endpoint's pool, is an infrastructure hold: the client waits for
the pool to come back, up to ``hold_timeout``, without spending an attempt. A ``/health`` that is
unreachable or does not report the pool never causes a hold. A 400 saying the prompt plus
``max_tokens`` exceeds the context window lowers ``max_tokens`` to the remaining context, measured
by a one-token probe of the same prompt. When the prompt alone fills the window, or the measured
remaining context does not lower ``max_tokens`` (the router and the server count the prompt
differently), ``GlmContextExhausted`` is raised, and a continuation that hits it ends the call with
the output gathered so far.

vLLM's GLM tool parser reports a reply cut off by ``max_tokens`` inside a tool call as
``finish_reason == "tool_calls"`` with the partial arguments (measured live). A ``tool_calls``
segment that spent its whole ``max_tokens`` is therefore reported as ``LENGTH``; the raw events in
``Attempt.events`` keep what the server sent. A call that ends exactly on the budget is reported
as cut too.
"""

import asyncio
import json
import logging
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from types import TracebackType
from typing import Self

import httpx
from rigging.timing import ExponentialBackoff

from taskforge.llm.endpoint import GLM_MODEL, resolve_glm_base_url
from taskforge.llm.policy import (
    CONTINUE_FINAL_MESSAGE_FIELDS,
    PREFILL_TEMPLATE_KWARGS,
    LLMPolicy,
    Message,
    continuation_messages,
    prefilled_messages,
    reasoning_continuation_messages,
)

logger = logging.getLogger(__name__)

RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
CONTEXT_LIMIT_PATTERN = re.compile(r"maximum context length is (\d+) tokens")
ERROR_BODY_LIMIT = 2000
CONNECT_TIMEOUT = 30.0
WRITE_TIMEOUT = 120.0
HEALTH_TIMEOUT = 20.0
PROBE_SEGMENT = -1


class Pool(StrEnum):
    """Router worker pool; the bearer token binds which one serves a request."""

    HIGH = "high"
    BULK = "bulk"


@dataclass(frozen=True)
class GlmEndpoint:
    """An OpenAI-compatible base URL ending in ``/v1``, its bearer token, and the pool it serves."""

    base_url: str
    token: str = field(repr=False)
    pool: Pool
    model: str = GLM_MODEL

    def __post_init__(self) -> None:
        if not self.base_url.endswith("/v1"):
            raise ValueError(f"base_url must end in /v1: {self.base_url}")


def endpoint_in_task(relay_job: str, token: str, pool: Pool) -> GlmEndpoint:
    """Resolve the endpoint registered by an Iris GLM relay job; call only inside an Iris task."""
    return GlmEndpoint(base_url=resolve_glm_base_url(relay_job), token=token, pool=pool)


class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    ABORT = "abort"
    """vLLM aborted the request (for example on worker drain); the attempt is retried."""


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


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    cached_tokens: int

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
        )


@dataclass(frozen=True)
class Attempt:
    """One HTTP request, successful or not, with the raw SSE ``data:`` payloads it received.

    ``segment`` is the continuation index; -1 marks the one-token probe that measures a prompt
    after a context-overflow rejection.
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
class Completion:
    """One logical call: the concatenation of its continuation segments.

    ``ttft`` is measured on the first segment's successful attempt; ``decode_time`` sums, over each
    segment's successful attempt, the time after its first token. ``wall_time`` includes backoff.
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

    @property
    def decode_tokens_per_second(self) -> float:
        return self.usage.completion_tokens / self.decode_time


class GlmRequestRejected(Exception):
    """The server rejected the request in a way a retry cannot fix."""

    def __init__(self, status: int, body: str):
        super().__init__(f"GLM rejected request with HTTP {status}: {body}")
        self.status = status
        self.body = body


class GlmContextExhausted(GlmRequestRejected):
    """The prompt alone fills the context window, so no output token fits."""

    def __init__(self, body: str):
        super().__init__(400, body)


class GlmUnavailable(Exception):
    """Retries or the infrastructure hold ran out; ``attempts`` holds the record."""

    def __init__(self, message: str, attempts: Sequence[Attempt]):
        super().__init__(message)
        self.attempts = tuple(attempts)


class _AttemptFailed(Exception):
    def __init__(self, outcome: AttemptOutcome, detail: str, http_status: int | None = None, retry_after: float = 0.0):
        super().__init__(f"{outcome}: {detail}")
        self.outcome = outcome
        self.detail = detail
        self.http_status = http_status
        self.retry_after = retry_after


@dataclass
class _Stream:
    """Accumulates one streamed response."""

    started: float
    content: list[str] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)
    tool_ids: dict[int, str] = field(default_factory=dict)
    tool_names: dict[int, str] = field(default_factory=dict)
    tool_arguments: dict[int, list[str]] = field(default_factory=dict)
    finish_reason: FinishReason | None = None
    usage: Usage | None = None
    ttft: float | None = None
    finished: float | None = None

    def add(self, event: dict) -> bool:
        """Fold one SSE event in; return whether it carried a token."""
        if event.get("usage"):
            self.usage = _usage(event["usage"])
        carried = False
        for choice in event.get("choices", []):
            delta = choice.get("delta", {})
            if delta.get("content"):
                self.content.append(delta["content"])
                carried = True
            if delta.get("reasoning"):
                self.reasoning.append(delta["reasoning"])
                carried = True
            for call in delta.get("tool_calls") or []:
                index = call["index"]
                if call.get("id"):
                    self.tool_ids[index] = call["id"]
                function = call.get("function", {})
                if function.get("name"):
                    self.tool_names[index] = function["name"]
                self.tool_arguments.setdefault(index, []).append(function.get("arguments") or "")
                carried = True
            if choice.get("finish_reason"):
                self.finish_reason = FinishReason(choice["finish_reason"])
        if carried and self.ttft is None:
            self.ttft = time.monotonic() - self.started
        return carried

    def tool_calls(self) -> tuple[ToolCall, ...]:
        return tuple(
            ToolCall(id=self.tool_ids[i], name=self.tool_names[i], arguments="".join(self.tool_arguments[i]))
            for i in sorted(self.tool_names)
        )


def _usage(raw: Mapping) -> Usage:
    completion_details = raw.get("completion_tokens_details") or {}
    prompt_details = raw.get("prompt_tokens_details") or {}
    return Usage(
        prompt_tokens=raw["prompt_tokens"],
        completion_tokens=raw["completion_tokens"],
        reasoning_tokens=completion_details.get("reasoning_tokens") or 0,
        cached_tokens=prompt_details.get("cached_tokens") or 0,
    )


def _retry_after(response: httpx.Response) -> float:
    value = response.headers.get("retry-after", "")
    return float(value) if value.replace(".", "", 1).isdigit() else 0.0


def _status_failure(response: httpx.Response, body: str) -> _AttemptFailed:
    status = response.status_code
    if status == 404 and "route" in body.lower():
        return _AttemptFailed(AttemptOutcome.ROUTE_MISSING, body, status)
    if status == 400 and CONTEXT_LIMIT_PATTERN.search(body):
        return _AttemptFailed(AttemptOutcome.CONTEXT_OVERFLOW, body, status)
    if status in RETRYABLE_STATUSES:
        return _AttemptFailed(AttemptOutcome.RETRYABLE_STATUS, body, status, _retry_after(response))
    raise GlmRequestRejected(status, body)


def request_body(
    model: str,
    messages: Sequence[Message],
    max_tokens: int,
    policy: LLMPolicy,
    request_fields: Mapping[str, object],
) -> dict[str, object]:
    """The exact chat-completions body the client sends, without the bearer token."""
    return {
        "model": model,
        "messages": list(messages),
        "max_tokens": max_tokens,
        **policy.sampling_fields(),
        **request_fields,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


class GlmClient:
    """Async GLM client. Use as ``async with GlmClient(endpoint) as client``.

    Args:
        endpoint: Where to send requests.
        max_attempts: Spent attempts per request before ``GlmUnavailable``; holds do not count.
        hold_timeout: Seconds an infrastructure hold may last before ``GlmUnavailable``.
        backoff: Delay schedule between attempts and between health polls during a hold.
        max_connections: httpx connection-pool size; sized for hundreds of concurrent calls.
    """

    def __init__(
        self,
        endpoint: GlmEndpoint,
        *,
        max_attempts: int = 8,
        hold_timeout: float = 3600.0,
        backoff: ExponentialBackoff | None = None,
        max_connections: int = 512,
    ):
        self.endpoint = endpoint
        self._max_attempts = max_attempts
        self._hold_timeout = hold_timeout
        self._backoff = backoff or ExponentialBackoff(initial=1.0, maximum=60.0, factor=2.0)
        self._http = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {endpoint.token}"},
            limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        await self._http.aclose()

    async def complete(
        self,
        messages: Sequence[Message],
        policy: LLMPolicy,
        request_fields: Mapping[str, object] | None = None,
    ) -> Completion:
        """Run one logical call, continuing on ``finish_reason == "length"`` per ``policy``.

        A continuation that no longer fits the context window ends the call with the segments
        gathered so far and ``finish_reason == "length"``.

        Args:
            messages: Chat messages.
            policy: Sampling, stall, and continuation policy.
            request_fields: Extra request-body fields such as ``tools`` and ``tool_choice``.
        """
        started = time.monotonic()
        segments: list[_Stream] = []
        attempts: list[Attempt] = []
        conversation = list(messages)
        fields: Mapping[str, object] = request_fields or {}
        while True:
            try:
                stream = await self._segment(conversation, policy, fields, len(segments), attempts)
            except GlmContextExhausted:
                if not segments:
                    raise
                logger.warning("GLM context is full after %d segments; returning the partial output", len(segments))
                break
            segments.append(stream)
            if (
                stream.finish_reason is not FinishReason.LENGTH
                or stream.tool_names
                or len(segments) > policy.max_continuations
            ):
                break
            content = "".join("".join(s.content) for s in segments)
            if content:
                conversation = continuation_messages(messages, content)
                fields = request_fields or {}
            else:
                reasoning = "".join("".join(s.reasoning) for s in segments)
                conversation = reasoning_continuation_messages(messages, reasoning)
                fields = {**(request_fields or {}), **CONTINUE_FINAL_MESSAGE_FIELDS}
        usage = segments[0].usage
        assert usage is not None and segments[-1].finish_reason is not None
        for stream in segments[1:]:
            assert stream.usage is not None
            usage = usage + stream.usage
        return Completion(
            content="".join("".join(s.content) for s in segments),
            reasoning="".join("".join(s.reasoning) for s in segments),
            tool_calls=segments[-1].tool_calls(),
            finish_reason=segments[-1].finish_reason,
            usage=usage,
            wall_time=time.monotonic() - started,
            ttft=segments[0].ttft or 0.0,
            decode_time=sum((s.finished or s.started) - s.started - (s.ttft or 0.0) for s in segments),
            continuations=len(segments) - 1,
            attempts=tuple(attempts),
        )

    async def _segment(
        self,
        messages: Sequence[Message],
        policy: LLMPolicy,
        request_fields: Mapping[str, object],
        segment: int,
        attempts: list[Attempt],
    ) -> _Stream:
        """Send one request with retries and holds, appending every attempt to ``attempts``.

        Attempt retries and health polls during holds follow separate copies of the backoff
        schedule, so a hold does not inflate the delay before the next spent attempt.
        """
        retry_backoff = self._backoff.copy()
        hold_backoff = self._backoff.copy()
        max_tokens = policy.max_tokens
        spent = 0
        hold_started: float | None = None
        while True:
            body = request_body(self.endpoint.model, messages, max_tokens, policy, request_fields)
            events: list[str] = []
            started_wall, started = time.time(), time.monotonic()
            try:
                stream = await self._stream(body, policy.stall_timeout, events)
            except _AttemptFailed as failure:
                attempts.append(
                    Attempt(
                        segment=segment,
                        outcome=failure.outcome,
                        max_tokens=max_tokens,
                        started=started_wall,
                        duration=time.monotonic() - started,
                        http_status=failure.http_status,
                        detail=failure.detail,
                        events=tuple(events),
                    )
                )
                if failure.outcome is AttemptOutcome.CONTEXT_OVERFLOW:
                    if max_tokens == 1:
                        raise GlmContextExhausted(failure.detail) from failure
                    remaining = await self._remaining_context(messages, policy, request_fields, failure.detail, attempts)
                    if remaining >= max_tokens:
                        # The router's 400 and vLLM's usage disagree on the prompt length; retrying
                        # at the same budget would get the same 400 forever.
                        raise GlmContextExhausted(
                            f"context overflow at max_tokens={max_tokens}, measured remaining {remaining}: "
                            f"{failure.detail}"
                        ) from failure
                    max_tokens = remaining
                    continue
                if failure.outcome is AttemptOutcome.ROUTE_MISSING or await self._pool_empty():
                    if hold_started is None:
                        hold_started = time.monotonic()
                    await self._hold(hold_started + self._hold_timeout, hold_backoff, attempts)
                    continue
                hold_started = None
                hold_backoff = self._backoff.copy()
                spent += 1
                if spent >= self._max_attempts:
                    raise GlmUnavailable(f"GLM attempts exhausted after {spent}: {failure}", attempts) from failure
                delay = max(retry_backoff.next_interval(), failure.retry_after)
                logger.warning(
                    "GLM attempt %d/%d failed (%s); retrying in %.1fs", spent, self._max_attempts, failure, delay
                )
                await asyncio.sleep(delay)
                continue
            assert stream.usage is not None
            if stream.finish_reason is FinishReason.TOOL_CALLS and stream.usage.completion_tokens >= max_tokens:
                stream.finish_reason = FinishReason.LENGTH
            attempts.append(
                Attempt(
                    segment=segment,
                    outcome=AttemptOutcome.COMPLETED,
                    max_tokens=max_tokens,
                    started=started_wall,
                    duration=time.monotonic() - started,
                    http_status=200,
                    detail="",
                    events=tuple(events),
                )
            )
            return stream

    async def _stream(self, body: Mapping[str, object], stall_timeout: float, events: list[str]) -> _Stream:
        """Run one streamed request, appending each raw ``data:`` payload to ``events``."""
        timeout = httpx.Timeout(connect=CONNECT_TIMEOUT, read=stall_timeout, write=WRITE_TIMEOUT, pool=None)
        stream = _Stream(started=time.monotonic())
        try:
            async with self._http.stream(
                "POST", f"{self.endpoint.base_url}/chat/completions", json=body, timeout=timeout
            ) as response:
                if response.status_code != 200:
                    text = (await response.aread()).decode(errors="replace")[:ERROR_BODY_LIMIT]
                    raise _status_failure(response, text)
                await self._read_events(response, stream, stall_timeout, events)
                return stream
        except httpx.ReadTimeout as error:
            raise _AttemptFailed(AttemptOutcome.STALLED, f"no bytes for {stall_timeout}s") from error
        except httpx.TransportError as error:
            raise _AttemptFailed(AttemptOutcome.TRANSPORT_ERROR, f"{type(error).__name__}: {error}") from error

    async def _read_events(
        self, response: httpx.Response, stream: _Stream, stall_timeout: float, events: list[str]
    ) -> None:
        last_token = time.monotonic()
        lines = response.aiter_lines()
        while True:
            remaining = stall_timeout - (time.monotonic() - last_token)
            try:
                line = await asyncio.wait_for(anext(lines), timeout=max(remaining, 0.0))
            except TimeoutError as error:
                raise _AttemptFailed(AttemptOutcome.STALLED, f"no token for {stall_timeout}s") from error
            except StopAsyncIteration as error:
                raise _AttemptFailed(AttemptOutcome.INCOMPLETE_STREAM, "stream ended before [DONE]") from error
            if not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                break
            events.append(data)
            try:
                event = json.loads(data)
            except json.JSONDecodeError as error:
                raise _AttemptFailed(AttemptOutcome.STREAM_ERROR, f"malformed SSE data: {error}") from error
            if "error" in event:
                raise _AttemptFailed(AttemptOutcome.STREAM_ERROR, json.dumps(event["error"])[:ERROR_BODY_LIMIT])
            if stream.add(event):
                last_token = time.monotonic()
            if stream.finish_reason is FinishReason.ABORT:
                raise _AttemptFailed(AttemptOutcome.ABORTED, "server aborted the request")
        if stream.finish_reason is None or stream.usage is None:
            raise _AttemptFailed(AttemptOutcome.INCOMPLETE_STREAM, "stream lacked finish_reason or usage")
        stream.finished = time.monotonic()

    async def _remaining_context(
        self,
        messages: Sequence[Message],
        policy: LLMPolicy,
        request_fields: Mapping[str, object],
        detail: str,
        attempts: list[Attempt],
    ) -> int:
        """Measure the prompt with a one-token request; return the context left for output."""
        match = CONTEXT_LIMIT_PATTERN.search(detail)
        assert match is not None
        probe = replace(policy, max_tokens=1, max_continuations=0)
        stream = await self._segment(messages, probe, request_fields, PROBE_SEGMENT, attempts)
        assert stream.usage is not None
        remaining = int(match.group(1)) - stream.usage.prompt_tokens
        if remaining < 1:
            raise GlmContextExhausted(detail)
        logger.info("GLM prompt has %d tokens; lowering max_tokens to %d", stream.usage.prompt_tokens, remaining)
        return remaining

    async def _pool_empty(self) -> bool:
        """Whether ``/health`` answers 200 and reports zero workers in this endpoint's pool.

        An unreachable ``/health``, a non-200 answer, or one that does not name the pool returns
        False, so the failure that prompted the check spends an attempt instead of holding.
        """
        try:
            response = await self._http.get(
                f"{self.endpoint.base_url.removesuffix('/v1')}/health", timeout=HEALTH_TIMEOUT
            )
        except httpx.TransportError:
            return False
        if response.status_code != 200:
            return False
        workers = response.json().get("workers", {})
        return str(self.endpoint.pool) in workers and workers[str(self.endpoint.pool)] == 0

    async def _hold(self, deadline: float, backoff: ExponentialBackoff, attempts: Sequence[Attempt]) -> None:
        """Wait, polling ``/health``, until the pool is no longer reported empty; raise once ``deadline`` passes."""
        while True:
            if time.monotonic() >= deadline:
                raise GlmUnavailable(f"GLM infrastructure hold exceeded {self._hold_timeout}s", attempts)
            await asyncio.sleep(backoff.next_interval())
            if not await self._pool_empty():
                return


def _joined_prefilled(parts: Sequence[Completion], content: str, started: float) -> Completion:
    """One ``Completion`` from prefilled segments, renumbering each segment's attempts in order."""
    attempts = tuple(
        replace(attempt, segment=index if attempt.segment != PROBE_SEGMENT else PROBE_SEGMENT)
        for index, part in enumerate(parts)
        for attempt in part.attempts
    )
    usage = parts[0].usage
    for part in parts[1:]:
        usage = usage + part.usage
    return Completion(
        content=content,
        reasoning="".join(part.reasoning for part in parts),
        tool_calls=parts[-1].tool_calls,
        finish_reason=parts[-1].finish_reason,
        usage=usage,
        wall_time=time.monotonic() - started,
        ttft=parts[0].ttft,
        decode_time=sum(part.decode_time for part in parts),
        continuations=len(parts) - 1,
        attempts=attempts,
    )


async def complete_prefilled(
    client: GlmClient,
    messages: Sequence[Message],
    policy: LLMPolicy,
    prefix: str,
    request_fields: Mapping[str, object] | None = None,
) -> Completion:
    """Run one logical call whose answer starts with ``prefix``; the returned ``content`` starts with it.

    ``prefix`` is sent as the start of the assistant turn (``prefilled_messages``) with
    ``CONTINUE_FINAL_MESSAGE_FIELDS``, and the model writes the rest. Measured live on GLM-5.3:

    * A prefilled turn does not think. The chat template renders an empty ``<think></think>`` before
      the prefilled text (the prompt has the same tokens with and without an explicit empty think
      block), so the model continues the answer directly. Without ``enable_thinking: false`` vLLM's
      reasoning parser, which waits for a ``</think>`` the output never contains, files the whole
      continuation under ``reasoning`` and leaves ``content`` empty; with it the continuation is
      ``content`` and ``reasoning_tokens`` is 0. This function sends ``enable_thinking: false``
      beside the policy's ``reasoning_effort``, so ``chat_template_kwargs`` in ``request_fields`` is
      replaced.
    * The template strips the prefilled text, so ``"id: "`` and ``"id:"`` render the same prompt. A
      ``prefix`` with leading or trailing whitespace is rejected, so ``content`` is exactly what the
      model saw followed by what it wrote.

    A reply cut off by ``max_tokens`` is continued, up to ``policy.max_continuations`` times, by
    prefilling everything written so far (trailing whitespace removed, since the template strips it
    and the model writes it again). A continuation that no longer fits the context window ends the
    call with the output gathered so far, as in ``GlmClient.complete``.

    Raises:
        ValueError: ``prefix`` is empty or has leading or trailing whitespace.
    """
    if not prefix or prefix != prefix.strip():
        raise ValueError(f"prefix must be non-empty without surrounding whitespace, got {prefix!r}")
    template_kwargs = {**policy.template_kwargs(), **PREFILL_TEMPLATE_KWARGS}
    fields = {**(request_fields or {}), **CONTINUE_FINAL_MESSAGE_FIELDS, "chat_template_kwargs": template_kwargs}
    segment_policy = replace(policy, max_continuations=0)
    started = time.monotonic()
    parts: list[Completion] = []
    written = prefix
    while True:
        try:
            part = await client.complete(prefilled_messages(messages, written), segment_policy, fields)
        except GlmContextExhausted:
            if not parts:
                raise
            logger.warning("GLM context is full after %d prefilled segments; returning the partial output", len(parts))
            break
        parts.append(part)
        if part.finish_reason is not FinishReason.LENGTH or part.tool_calls or len(parts) > policy.max_continuations:
            return _joined_prefilled(parts, written + part.content, started)
        written = (written + part.content).rstrip()
    return _joined_prefilled(parts, written, started)
