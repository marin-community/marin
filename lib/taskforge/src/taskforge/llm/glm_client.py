# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The single GLM-5.3 transport: streaming chat completions through the ``openai`` SDK.

``GlmClient`` implements ``taskforge.llm.client.ModelClient``; ``LLMProviderConfig`` sets how it
calls the provider. The SDK builds and sends the request (its own retries are off); this module
owns retries, holds and stall detection. Every request streams, so a stall (no token for
``stall_timeout`` seconds) is detected without a wall-clock cap: the SDK's read timeout covers
bytes, and the client's loop over chunks covers tokens. Each retryable failure (408, 429, 5xx,
connection errors, stalls, truncated, malformed, or aborted streams) counts against
``max_attempts`` and is retried with exponential backoff. A missing relay route (404 whose body
names the route), or a retryable failure while ``/health`` answers 200 and reports zero workers
for the endpoint's pool, is an infrastructure hold: the client waits for the pool to come back, up
to ``hold_timeout``, and the failure does not count against ``max_attempts``. A ``/health`` that is
unreachable or does not report the pool never causes a hold.

The client asks its ``EndpointSource`` where GLM is when a hold starts, on every hold poll, and
after a connection error (a relay that moved shows up as connection refused). When the answer
changes it rebuilds the SDK client on the same connection pool, so the next attempt goes to the
new endpoint.

A 400 saying the prompt plus ``max_tokens`` exceeds the context window raises ``ContextExhausted``
at once; the client never lowers ``max_tokens`` and never truncates the prompt. A prompt that long
is a sign that its parts belong in files, read by the model in an agentic environment, rather than
in one request. A continuation that overflows the window ends the call with the output gathered so
far and ``finish_reason == "length"``.

vLLM's GLM tool parser reports a reply cut off by ``max_tokens`` inside a tool call as
``finish_reason == "tool_calls"`` with the partial arguments (measured live). A ``tool_calls``
segment that used its whole ``max_tokens`` is therefore reported as ``LENGTH``; the chunks in
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
from typing import Any, Protocol, Self

import httpx
import openai
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletionChunk
from pydantic import BaseModel
from rigging.timing import ExponentialBackoff

from taskforge.llm.client import (
    Attempt,
    AttemptOutcome,
    Completion,
    ContextExhausted,
    FinishReason,
    ModelRequestRejected,
    ModelUnavailable,
    StructuredResult,
    ToolCall,
    Usage,
)
from taskforge.llm.policy import (
    CONTINUE_FINAL_MESSAGE_FIELDS,
    LLMPolicy,
    LLMProviderConfig,
    Message,
    continuation_messages,
    reasoning_continuation_messages,
)
from taskforge.llm.structured import complete_structured, response_format

logger = logging.getLogger(__name__)

API_ROOT = "/v1"
"""The path suffix of an OpenAI-compatible base URL; the router's ``/health`` sits beside it."""
RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
CONTEXT_LIMIT_PATTERN = re.compile(r"maximum context length is (\d+) tokens")
SDK_REQUEST_FIELDS = frozenset({"tools", "tool_choice", "response_format", "parallel_tool_calls"})
"""Request fields passed to ``chat.completions.create`` by name; every other field goes in ``extra_body``."""
ERROR_BODY_LIMIT = 2000
CONNECT_TIMEOUT = 30.0
WRITE_TIMEOUT = 120.0
HEALTH_TIMEOUT = 20.0


class Pool(StrEnum):
    """Router worker pool; the bearer token binds which one serves a request."""

    HIGH = "high"
    BULK = "bulk"


@dataclass(frozen=True)
class GlmEndpoint:
    """An OpenAI-compatible base URL ending in ``/v1``, its bearer token, and the pool it serves.

    The model requests name is ``LLMProviderConfig.model``.
    """

    base_url: str
    token: str = field(repr=False)
    pool: Pool

    def __post_init__(self) -> None:
        if not self.base_url.endswith(API_ROOT):
            raise ValueError(f"base_url must end in {API_ROOT}: {self.base_url}")


class EndpointSource(Protocol):
    """Where the client reaches GLM now.

    The client calls ``current`` again when a hold starts, so a relay that moves is picked up.
    """

    def current(self) -> GlmEndpoint: ...


@dataclass(frozen=True)
class FixedEndpoint:
    """An endpoint that never moves."""

    endpoint: GlmEndpoint

    def current(self) -> GlmEndpoint:
        return self.endpoint


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

    def add(self, chunk: ChatCompletionChunk) -> bool:
        """Fold one chunk in; return whether it carried a token."""
        if chunk.usage is not None:
            self.usage = _usage(chunk.usage)
        carried = False
        for choice in chunk.choices:
            delta = choice.delta
            if delta.content:
                self.content.append(delta.content)
                carried = True
            # vLLM's reasoning parser streams the reasoning block in a field the SDK does not model.
            reasoning = (delta.model_extra or {}).get("reasoning")
            if reasoning:
                self.reasoning.append(reasoning)
                carried = True
            for call in delta.tool_calls or []:
                if call.id:
                    self.tool_ids[call.index] = call.id
                if call.function is not None and call.function.name:
                    self.tool_names[call.index] = call.function.name
                arguments = call.function.arguments if call.function is not None else None
                self.tool_arguments.setdefault(call.index, []).append(arguments or "")
                carried = True
            if choice.finish_reason:
                self.finish_reason = FinishReason(choice.finish_reason)
        if carried and self.ttft is None:
            self.ttft = time.monotonic() - self.started
        return carried

    def tool_calls(self) -> tuple[ToolCall, ...]:
        return tuple(
            ToolCall(id=self.tool_ids[i], name=self.tool_names[i], arguments="".join(self.tool_arguments[i]))
            for i in sorted(self.tool_names)
        )


def _usage(usage: CompletionUsage) -> Usage:
    completion_details = usage.completion_tokens_details
    prompt_details = usage.prompt_tokens_details
    return Usage(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        reasoning_tokens=(completion_details.reasoning_tokens if completion_details else None) or 0,
        cached_tokens=(prompt_details.cached_tokens if prompt_details else None) or 0,
    )


def _retry_after(response: httpx.Response) -> float:
    value = response.headers.get("retry-after", "")
    return float(value) if value.replace(".", "", 1).isdigit() else 0.0


def _status_failure(error: openai.APIStatusError) -> _AttemptFailed:
    status = error.status_code
    body = error.response.text[:ERROR_BODY_LIMIT]
    if status == 404 and "route" in body.lower():
        return _AttemptFailed(AttemptOutcome.ROUTE_MISSING, body, status)
    if status == 400 and CONTEXT_LIMIT_PATTERN.search(body):
        return _AttemptFailed(AttemptOutcome.CONTEXT_OVERFLOW, body, status)
    if status in RETRYABLE_STATUSES:
        return _AttemptFailed(AttemptOutcome.RETRYABLE_STATUS, body, status, _retry_after(error.response))
    return _AttemptFailed(AttemptOutcome.REJECTED, body, status)


def request_kwargs(
    model: str,
    messages: Sequence[Message],
    max_tokens: int,
    policy: LLMPolicy,
    request_fields: Mapping[str, object],
) -> dict[str, Any]:
    """The exact ``chat.completions.create`` arguments the client sends.

    ``request_fields`` named in ``SDK_REQUEST_FIELDS`` are passed by name; the rest (vLLM extras
    such as ``continue_final_message``) go in ``extra_body``, which the SDK merges into the body.
    """
    sdk_fields = {key: value for key, value in request_fields.items() if key in SDK_REQUEST_FIELDS}
    extra_fields = {key: value for key, value in request_fields.items() if key not in SDK_REQUEST_FIELDS}
    return {
        "model": model,
        "messages": list(messages),
        "max_tokens": max_tokens,
        "temperature": policy.temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
        **sdk_fields,
        "extra_body": {"chat_template_kwargs": policy.template_kwargs(), **extra_fields},
    }


class GlmClient:
    """Async GLM client. Use as ``async with GlmClient(source, provider) as client``.

    Args:
        source: Where to send requests; asked again when a hold starts or a connection fails.
        provider: The model, retries, backoff, the infrastructure hold, the stall timeout, the
            connection pool and the structured-call repair budget.
    """

    def __init__(self, source: EndpointSource, provider: LLMProviderConfig):
        self._source = source
        self._provider = provider
        self._backoff = ExponentialBackoff(
            initial=provider.backoff_initial, maximum=provider.backoff_maximum, factor=provider.backoff_factor
        )
        connections = provider.max_connections
        self._http = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=connections, max_keepalive_connections=connections)
        )
        self._endpoint = source.current()
        self._sdk = self._sdk_client(self._endpoint)

    @property
    def model(self) -> str:
        return self._provider.model

    @property
    def http(self) -> httpx.AsyncClient:
        """The connection pool the SDK is built on, for the router's non-chat endpoints such as ``/health``."""
        return self._http

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        await self._http.aclose()

    def _sdk_client(self, endpoint: GlmEndpoint) -> openai.AsyncOpenAI:
        timeout = httpx.Timeout(
            connect=CONNECT_TIMEOUT, read=self._provider.stall_timeout, write=WRITE_TIMEOUT, pool=None
        )
        return openai.AsyncOpenAI(
            base_url=endpoint.base_url, api_key=endpoint.token, max_retries=0, timeout=timeout, http_client=self._http
        )

    def _refresh_endpoint(self) -> None:
        """Ask the source where GLM is; rebuild the SDK client if the endpoint moved."""
        endpoint = self._source.current()
        if endpoint == self._endpoint:
            return
        logger.warning("GLM endpoint moved from %s to %s", self._endpoint.base_url, endpoint.base_url)
        self._endpoint = endpoint
        self._sdk = self._sdk_client(endpoint)

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
            except ContextExhausted:
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

    async def complete_schema(
        self, messages: Sequence[Message], output_type: type[BaseModel], policy: LLMPolicy
    ) -> Completion:
        """One request whose reply the server constrains to ``output_type``'s JSON schema.

        The request is never continued: a JSON object cut by the output limit cannot be resumed
        under the schema's grammar, so the reply ends with ``finish_reason == "length"``.
        """
        fields = {"response_format": response_format(output_type)}
        return await self.complete(messages, replace(policy, max_continuations=0), fields)

    async def complete_structured[T: BaseModel](
        self, messages: Sequence[Message], output_type: type[T], policy: LLMPolicy
    ) -> StructuredResult[T]:
        """Return a reply that ``output_type`` validates, requested with ``response_format=output_type``.

        vLLM enforces ``output_type``'s JSON schema by structured decoding (xgrammar), applied after
        the reasoning block. The client still validates the reply with ``output_type``, which covers
        what the grammar cannot: pydantic validators, and a reply cut off by the output limit. An
        invalid reply is sent back for correction up to the provider's ``structured_repairs`` times
        before ``StructuredOutputError``.
        """
        return await complete_structured(self, messages, output_type, policy, self._provider.structured_repairs)

    async def _segment(
        self,
        messages: Sequence[Message],
        policy: LLMPolicy,
        request_fields: Mapping[str, object],
        segment: int,
        attempts: list[Attempt],
    ) -> _Stream:
        """Send one request with retries and holds, appending every attempt to ``attempts``."""
        # Separate schedules, so a hold does not inflate the delay before the next counted retry.
        retry_backoff = self._backoff.copy()
        hold_backoff = self._backoff.copy()
        max_tokens = policy.max_tokens
        failures = 0
        hold_started: float | None = None
        while True:
            kwargs = request_kwargs(self._provider.model, messages, max_tokens, policy, request_fields)
            events: list[str] = []
            started_wall, started = time.time(), time.monotonic()
            try:
                stream = await self._stream(kwargs, events)
            except _AttemptFailed as failure:
                duration = time.monotonic() - started
                status, detail = failure.http_status, failure.detail
                attempts.append(
                    Attempt(segment, failure.outcome, max_tokens, started_wall, duration, status, detail, tuple(events))
                )
                if failure.outcome is AttemptOutcome.REJECTED:
                    assert failure.http_status is not None
                    raise ModelRequestRejected(failure.http_status, failure.detail, attempts) from failure
                if failure.outcome is AttemptOutcome.CONTEXT_OVERFLOW:
                    raise ContextExhausted(failure.detail, attempts) from failure
                if failure.outcome is AttemptOutcome.TRANSPORT_ERROR:
                    self._refresh_endpoint()
                if failure.outcome is AttemptOutcome.ROUTE_MISSING or await self._pool_empty():
                    if hold_started is None:
                        hold_started = time.monotonic()
                        self._refresh_endpoint()
                    await self._hold(hold_started + self._provider.hold_timeout, hold_backoff, attempts)
                    continue
                hold_started = None
                hold_backoff = self._backoff.copy()
                failures += 1
                if failures >= self._provider.max_attempts:
                    raise ModelUnavailable(f"GLM attempts exhausted after {failures}: {failure}", attempts) from failure
                delay = max(retry_backoff.next_interval(), failure.retry_after)
                logger.warning(
                    "GLM attempt %d/%d failed (%s); retrying in %.1fs",
                    failures,
                    self._provider.max_attempts,
                    failure,
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            assert stream.usage is not None
            if stream.finish_reason is FinishReason.TOOL_CALLS and stream.usage.completion_tokens >= max_tokens:
                stream.finish_reason = FinishReason.LENGTH
            duration = time.monotonic() - started
            attempts.append(
                Attempt(segment, AttemptOutcome.COMPLETED, max_tokens, started_wall, duration, 200, "", tuple(events))
            )
            return stream

    async def _stream(self, kwargs: Mapping[str, Any], events: list[str]) -> _Stream:
        """Run one streamed request, appending each chunk's JSON to ``events``."""
        stall_timeout = self._provider.stall_timeout
        stream = _Stream(started=time.monotonic())
        try:
            chunks = await self._sdk.chat.completions.create(**kwargs)
            assert isinstance(chunks, openai.AsyncStream)
            async with chunks:
                last_token = time.monotonic()
                while True:
                    remaining = stall_timeout - (time.monotonic() - last_token)
                    try:
                        chunk = await asyncio.wait_for(anext(chunks), timeout=max(remaining, 0.0))
                    except StopAsyncIteration:
                        break
                    events.append(chunk.model_dump_json())
                    if stream.add(chunk):
                        last_token = time.monotonic()
                    if stream.finish_reason is FinishReason.ABORT:
                        raise _AttemptFailed(AttemptOutcome.ABORTED, "server aborted the request")
        except TimeoutError as error:
            raise _AttemptFailed(AttemptOutcome.STALLED, f"no token for {stall_timeout}s") from error
        except openai.APIStatusError as error:
            raise _status_failure(error) from error
        except openai.APITimeoutError as error:
            raise _AttemptFailed(AttemptOutcome.STALLED, f"no response within the timeout: {error}") from error
        except openai.APIConnectionError as error:
            raise _AttemptFailed(AttemptOutcome.TRANSPORT_ERROR, f"{error}: {error.__cause__!r}") from error
        except openai.APIError as error:
            # The SDK raises the base class for an ``error`` event inside the stream.
            raise _AttemptFailed(AttemptOutcome.STREAM_ERROR, json.dumps(error.body)[:ERROR_BODY_LIMIT]) from error
        except json.JSONDecodeError as error:
            raise _AttemptFailed(AttemptOutcome.STREAM_ERROR, f"malformed SSE data: {error}") from error
        # Errors while reading the body come from httpx unwrapped.
        except httpx.ReadTimeout as error:
            raise _AttemptFailed(AttemptOutcome.STALLED, f"no bytes for {stall_timeout}s") from error
        except httpx.TransportError as error:
            raise _AttemptFailed(AttemptOutcome.TRANSPORT_ERROR, f"{type(error).__name__}: {error}") from error
        if stream.finish_reason is None or stream.usage is None:
            raise _AttemptFailed(AttemptOutcome.INCOMPLETE_STREAM, "stream lacked finish_reason or usage")
        stream.finished = time.monotonic()
        return stream

    async def _pool_empty(self) -> bool:
        """Whether ``/health`` answers 200 and reports zero workers in this endpoint's pool.

        An unreachable ``/health``, a non-200 answer, or one that does not name the pool returns
        False, so the failure that prompted the check counts against ``max_attempts`` instead of holding.
        """
        endpoint = self._endpoint
        try:
            response = await self._http.get(
                f"{endpoint.base_url.removesuffix(API_ROOT)}/health",
                headers={"Authorization": f"Bearer {endpoint.token}"},
                timeout=HEALTH_TIMEOUT,
            )
        except httpx.TransportError:
            return False
        if response.status_code != 200:
            return False
        workers = response.json().get("workers", {})
        return str(endpoint.pool) in workers and workers[str(endpoint.pool)] == 0

    async def _hold(self, deadline: float, backoff: ExponentialBackoff, attempts: Sequence[Attempt]) -> None:
        """Wait, polling the source and ``/health``, until the pool is no longer reported empty.

        Raises ``ModelUnavailable`` once ``deadline`` passes.
        """
        while True:
            if time.monotonic() >= deadline:
                raise ModelUnavailable(f"GLM infrastructure hold exceeded {self._provider.hold_timeout}s", attempts)
            await asyncio.sleep(backoff.next_interval())
            self._refresh_endpoint()
            if not await self._pool_empty():
                return
