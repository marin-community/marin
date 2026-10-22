# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.3 chat completions through the ``openai`` SDK, retried by one loop.

``GlmClient`` implements ``taskforge.llm.client.ModelClient``, with ``LLMProviderConfig`` supplying
the model, the retry budget and backoff, the hold and stall timeouts and the connection pool. The
client builds each request and sends it through the SDK with the SDK's retries turned off, then
retries failed attempts itself. The SDK retries only before a response starts, which rules out
retrying a stall, an abort, a mid-stream error or a stream ending without usage. The SDK also cannot
hold on an empty pool or a missing route and returns no record of the individual attempts.

Every chat-completions request streams. The SDK's read timeout abandons a connection after
``stall_timeout`` seconds without bytes, and the client's loop over chunks abandons a stream after
as long without a token, so a slow reply still producing tokens runs without a wall-clock cap. The
SDK's ``ChatCompletionStreamState`` accumulates the chunks, and the client reads content,
reasoning, tool calls, finish reason and usage from its snapshot.

After a failed attempt, the client handles the failure by kind:

* A 408, 429 or 5xx status, a connection error, a stall, or a stream cut short, malformed or aborted
  counts against ``max_attempts``. The client retries it after an exponential backoff.
* A 404 mentioning the route starts an infrastructure hold. So does a retryable failure while
  ``/health`` answers 200 with zero workers in the endpoint's pool. The client then polls ``/health``
  until the pool has workers again, for up to ``hold_timeout``, without spending an attempt. An
  unreachable ``/health``, or a report without the pool, leads to an ordinary counted retry instead.
* A 400 saying the prompt plus ``max_tokens`` exceeds the context window raises ``ContextExhausted``
  at once. The client always sends the whole prompt with the policy's full ``max_tokens``. A prompt
  that long usually means its parts belong in files, for a model in an agentic environment to read.
  When a continuation overflows the window, the call returns the output gathered so far with
  ``finish_reason == "length"``.

The client asks its ``EndpointSource`` for the endpoint when a hold starts, on each hold poll and
after a connection error, since a moved relay shows up as a refused connection. When the answer
changes, each event loop's SDK client is rebuilt on that loop's connection pool at its next request,
which goes to the new endpoint.

An httpx pool works only in the event loop that opened it, so the client opens one connection pool,
with its SDK client, in each event loop that uses it, and drops the pools of loops that have closed.
A pickled client holds only its endpoint source and provider config, so a copy in another process
opens its own pools. After a stalled read the client replaces the running loop's pool, at most once
per ``POOL_RESET_INTERVAL``, so the retry cannot reuse a connection that may be dead. Requests still
streaming on the old pool keep it: the old pool is retired and closes when the last of them ends.

vLLM's GLM tool parser reports a reply cut off by ``max_tokens`` inside a tool call as
``finish_reason == "tool_calls"`` with partial arguments (measured live). The client reports a
``tool_calls`` segment as ``LENGTH`` whenever its completion tokens use up ``max_tokens``, which
also catches a reply ending exactly on the budget. ``Attempt.events`` holds the chunks as the server
sent them.

``complete_structured`` sends ``output_type``'s JSON schema as the ``response_format`` with
continuations off, then validates the reply and asks for repairs through ``taskforge.llm.structured``.
That module also appends a user turn describing each field, since the chat template renders nothing
from ``response_format``.
"""

import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from enum import StrEnum
from types import TracebackType
from typing import Any, Protocol, Self

import httpx
import openai
from openai.lib.streaming.chat import ChatCompletionStreamState
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletionChunk
from pydantic import BaseModel
from rigging.timing import ExponentialBackoff

from taskforge.llm.client import (
    RETRYABLE_STATUSES,
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
"""The path suffix of an OpenAI-compatible base URL. The router serves ``/health`` at the URL without it."""
CONTEXT_LIMIT_PATTERN = re.compile(r"maximum context length is (\d+) tokens")
SDK_REQUEST_FIELDS = frozenset({"tools", "tool_choice", "response_format", "parallel_tool_calls"})
"""Fields the client passes to ``chat.completions.create`` as keyword arguments, putting the rest in ``extra_body``."""
ERROR_BODY_LIMIT = 2000
CONNECT_TIMEOUT = 10.0
WRITE_TIMEOUT = 30.0
KEEPALIVE_EXPIRY = 30.0
"""Seconds an idle pooled connection may be reused, so a connection idle through a long author turn is not
reused blindly; a stalled read also replaces the whole pool (``_reset_pool``)."""
POOL_RESET_INTERVAL = 30.0
"""Least seconds between two pool resets, so a burst of stalls on one bad path replaces the pool once."""
HEALTH_TIMEOUT = 20.0


class Pool(StrEnum):
    """Router worker pool; the bearer token selects the pool serving a request."""

    HIGH = "high"
    BULK = "bulk"


@dataclass(frozen=True)
class GlmEndpoint:
    """An OpenAI-compatible base URL ending in ``/v1``, with its bearer token and the pool behind it.

    Requests take the model id from ``LLMProviderConfig.model``.
    """

    base_url: str
    token: str = field(repr=False)
    pool: Pool

    def __post_init__(self) -> None:
        if not self.base_url.endswith(API_ROOT):
            raise ValueError(f"base_url must end in {API_ROOT}: {self.base_url}")


class EndpointSource(Protocol):
    """Supplies the GLM endpoint for the client's requests.

    The client calls ``current`` at construction and again when a hold starts, on each hold poll and
    after a connection error. When the answer changes, the client sends its next attempt to the new
    endpoint.
    """

    def current(self) -> GlmEndpoint: ...


@dataclass(frozen=True)
class FixedEndpoint:
    """Supplies the same endpoint on every call."""

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


@dataclass(frozen=True)
class _Segment:
    """A streamed response as read from the SDK's snapshot, with the client's monotonic timings."""

    content: str
    reasoning: str
    tool_calls: tuple[ToolCall, ...]
    finish_reason: FinishReason
    usage: Usage
    ttft: float | None
    started: float
    finished: float


def _carries_token(chunk: ChatCompletionChunk) -> bool:
    """Check a chunk for generated text in its content, reasoning or tool-call deltas.

    vLLM's reasoning parser streams reasoning in ``delta.reasoning``, a field absent from the SDK's
    ``ChoiceDelta`` schema. A chunk holding only reasoning counts as a token for the stall clock and
    for ``ttft``.
    """
    return any(
        choice.delta.content or choice.delta.tool_calls or (choice.delta.model_extra or {}).get("reasoning")
        for choice in chunk.choices
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


def _status_failure(response: httpx.Response) -> _AttemptFailed:
    status = response.status_code
    body = response.text[:ERROR_BODY_LIMIT]
    if status == 404 and "route" in body.lower():
        return _AttemptFailed(AttemptOutcome.ROUTE_MISSING, body, status)
    if status == 400 and CONTEXT_LIMIT_PATTERN.search(body):
        return _AttemptFailed(AttemptOutcome.CONTEXT_OVERFLOW, body, status)
    if status in RETRYABLE_STATUSES:
        return _AttemptFailed(AttemptOutcome.RETRYABLE_STATUS, body, status, _retry_after(response))
    return _AttemptFailed(AttemptOutcome.REJECTED, body, status)


def request_kwargs(
    model: str,
    messages: Sequence[Message],
    max_tokens: int,
    policy: LLMPolicy,
    request_fields: Mapping[str, object],
) -> dict[str, Any]:
    """Build the exact ``chat.completions.create`` arguments the client sends.

    Fields listed in ``SDK_REQUEST_FIELDS`` go in as keyword arguments. The rest, vLLM extras such as
    ``continue_final_message`` among them, go in ``extra_body`` for the SDK to merge into the body.
    """
    sdk_fields = {key: value for key, value in request_fields.items() if key in SDK_REQUEST_FIELDS}
    extra_fields = {key: value for key, value in request_fields.items() if key not in SDK_REQUEST_FIELDS}
    return {
        "model": model,
        "messages": list(messages),
        "max_tokens": max_tokens,
        **policy.sampling_params(),
        "stream": True,
        "stream_options": {"include_usage": True},
        **sdk_fields,
        "extra_body": {"chat_template_kwargs": policy.template_kwargs(), **extra_fields},
    }


@dataclass(eq=False)
class _LoopPool:
    """The connection pool of one event loop, the SDK client on it for one endpoint, and its requests in flight."""

    http: httpx.AsyncClient
    sdk: openai.AsyncOpenAI
    endpoint: GlmEndpoint
    in_flight: int = 0


class GlmClient:
    """Async GLM client. Use as ``async with GlmClient(source, provider) as client``.

    The client opens one connection pool in each event loop that uses it, and a pickled client holds only
    ``source`` and ``provider``.

    Args:
        source: Supplies the endpoint. The client asks it again during holds and after a connection
            error.
        provider: The model, the retry budget and backoff, the hold and stall timeouts, the connection
            pool size and the structured-call repair budget.
    """

    def __init__(self, source: EndpointSource, provider: LLMProviderConfig):
        self._source = source
        self._provider = provider
        self._backoff = ExponentialBackoff(
            initial=provider.backoff_initial, maximum=provider.backoff_maximum, factor=provider.backoff_factor
        )
        self._endpoint = source.current()
        self._pools: dict[asyncio.AbstractEventLoop, _LoopPool] = {}
        self._retired: dict[asyncio.AbstractEventLoop, list[_LoopPool]] = {}
        self._pool_reset_at = -POOL_RESET_INTERVAL

    def __reduce__(self) -> tuple[type[Self], tuple[EndpointSource, LLMProviderConfig]]:
        return type(self), (self._source, self._provider)

    @property
    def model(self) -> str:
        return self._provider.model

    @property
    def http(self) -> httpx.AsyncClient:
        """The running event loop's connection pool, for requests to the router's other endpoints such as ``/health``."""
        return self._pool().http

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        loop = asyncio.get_running_loop()
        pools = self._retired.pop(loop, [])
        current = self._pools.pop(loop, None)
        if current is not None:
            pools.append(current)
        for pool in pools:
            await pool.http.aclose()

    def _open_pool(self) -> _LoopPool:
        """A fresh connection pool for the running event loop, with an SDK client on it for the current endpoint."""
        limit = self._provider.max_connections
        http = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=limit, max_keepalive_connections=limit, keepalive_expiry=KEEPALIVE_EXPIRY
            )
        )
        return _LoopPool(http, self._sdk_client(self._endpoint, http), self._endpoint)

    def _pool(self) -> _LoopPool:
        """The running event loop's connection pool and SDK client, opened on first use there.

        An httpx pool works only in the event loop that opened it. The pools of closed loops, retired ones
        included, are dropped.
        """
        for closed in [loop for loop in self._pools.keys() | self._retired.keys() if loop.is_closed()]:
            self._pools.pop(closed, None)
            self._retired.pop(closed, None)
        loop = asyncio.get_running_loop()
        pool = self._pools.get(loop)
        if pool is None:
            pool = self._open_pool()
            self._pools[loop] = pool
        elif pool.endpoint != self._endpoint:
            pool.sdk = self._sdk_client(self._endpoint, pool.http)
            pool.endpoint = self._endpoint
        return pool

    async def _reset_pool(self) -> None:
        """Open a fresh connection pool after a stalled read, so the retry cannot reuse a connection that may be dead.

        A stall has other causes the reset does not touch, such as the relay refusing connections or this process's
        own event loop blocking for longer than the connect timeout; the bounded retry with backoff carries the call
        through those, and the reset interval keeps the two from compounding into a reconnect storm.

        The reset replaces the running event loop's pool. Other calls in that loop may still be streaming on the old
        pool's connections, so the old pool is retired and closes when the last request leased from it ends
        (``_leased``). httpx drops an expired idle connection only when a request next uses its pool, so a retired
        pool left open would hold its sockets until the client exits.
        """
        now = time.monotonic()
        if now - self._pool_reset_at < POOL_RESET_INTERVAL:
            return
        self._pool_reset_at = now
        loop = asyncio.get_running_loop()
        old = self._pool()
        logger.warning(
            "GLM read stalled; replacing the connection pool (%d requests still on the old one)", old.in_flight
        )
        self._pools[loop] = self._open_pool()
        if old.in_flight:
            self._retired.setdefault(loop, []).append(old)
        else:
            await old.http.aclose()

    @asynccontextmanager
    async def _leased(self) -> AsyncIterator[tuple[openai.AsyncOpenAI, httpx.AsyncClient]]:
        """The running event loop's SDK client and its pool, which stays open until the request using them ends."""
        pool = self._pool()
        pool.in_flight += 1
        try:
            yield pool.sdk, pool.http
        finally:
            pool.in_flight -= 1
            retired = self._retired.get(asyncio.get_running_loop(), [])
            if not pool.in_flight and pool in retired:
                retired.remove(pool)
                await pool.http.aclose()

    def _sdk_client(self, endpoint: GlmEndpoint, http: httpx.AsyncClient) -> openai.AsyncOpenAI:
        timeout = httpx.Timeout(
            connect=CONNECT_TIMEOUT, read=self._provider.stall_timeout, write=WRITE_TIMEOUT, pool=None
        )
        return openai.AsyncOpenAI(
            base_url=endpoint.base_url, api_key=endpoint.token, max_retries=0, timeout=timeout, http_client=http
        )

    def _refresh_endpoint(self) -> None:
        """Ask the source for the endpoint. Each loop's SDK client moves to a new endpoint at its next request."""
        endpoint = self._source.current()
        if endpoint == self._endpoint:
            return
        logger.warning("GLM endpoint moved from %s to %s", self._endpoint.base_url, endpoint.base_url)
        self._endpoint = endpoint

    async def complete(
        self,
        messages: Sequence[Message],
        policy: LLMPolicy,
        request_fields: Mapping[str, object] | None = None,
    ) -> Completion:
        """Run a logical call, continuing after ``finish_reason == "length"`` as ``policy`` allows.

        When a continuation overflows the context window, the call ends with the segments gathered so
        far and ``finish_reason == "length"``.

        Args:
            messages: Chat messages.
            policy: Sampling and continuation settings.
            request_fields: Extra request-body fields such as ``tools`` and ``tool_choice``.
        """
        started = time.monotonic()
        segments: list[_Segment] = []
        attempts: list[Attempt] = []
        conversation = list(messages)
        fields: Mapping[str, object] = request_fields or {}
        while True:
            try:
                result = await self._segment(conversation, policy, fields, len(segments), attempts)
            except ContextExhausted:
                if not segments:
                    raise
                logger.warning("GLM context is full after %d segments; returning the partial output", len(segments))
                break
            segments.append(result)
            if (
                result.finish_reason is not FinishReason.LENGTH
                or result.tool_calls
                or len(segments) > policy.max_continuations
            ):
                break
            content = "".join(s.content for s in segments)
            if content:
                conversation = continuation_messages(messages, content)
                fields = request_fields or {}
            else:
                reasoning = "".join(s.reasoning for s in segments)
                conversation = reasoning_continuation_messages(messages, reasoning)
                fields = {**(request_fields or {}), **CONTINUE_FINAL_MESSAGE_FIELDS}
        return Completion(
            content="".join(s.content for s in segments),
            reasoning="".join(s.reasoning for s in segments),
            tool_calls=segments[-1].tool_calls,
            finish_reason=segments[-1].finish_reason,
            usage=sum((s.usage for s in segments[1:]), segments[0].usage),
            wall_time=time.monotonic() - started,
            ttft=segments[0].ttft or 0.0,
            decode_time=sum(s.finished - s.started - (s.ttft or 0.0) for s in segments),
            continuations=len(segments) - 1,
            attempts=tuple(attempts),
        )

    async def complete_schema(
        self, messages: Sequence[Message], output_type: type[BaseModel], policy: LLMPolicy
    ) -> Completion:
        """Request a reply constrained to ``output_type``'s JSON schema, with continuations off.

        vLLM applies the schema's grammar from the start of each request, so a JSON object cut off by
        the output limit cannot resume in a second request. A cut reply returns with
        ``finish_reason == "length"``.
        """
        fields = {"response_format": response_format(output_type)}
        return await self.complete(messages, replace(policy, max_continuations=0), fields)

    async def complete_structured[T: BaseModel](
        self, messages: Sequence[Message], output_type: type[T], policy: LLMPolicy
    ) -> StructuredResult[T]:
        """Request a reply as ``output_type`` and return it validated, repairing invalid replies.

        vLLM decodes the answer under ``output_type``'s JSON schema with xgrammar, after the reasoning
        block. The model reads the fields from the last user turn, holding
        ``taskforge.llm.structured.schema_instruction(output_type)``. The client then validates the
        reply with ``output_type`` to run its pydantic validators. It sends an invalid reply back for
        correction up to the provider's ``structured_repairs`` times, then raises
        ``StructuredOutputError``. A reply cut off by the output limit raises ``StructuredOutputError``
        at once, with no repair.
        """
        return await complete_structured(self, messages, output_type, policy, self._provider.structured_repairs)

    async def _segment(
        self,
        messages: Sequence[Message],
        policy: LLMPolicy,
        request_fields: Mapping[str, object],
        segment: int,
        attempts: list[Attempt],
    ) -> _Segment:
        """Stream a chat-completions request through ``_retrying``, appending each attempt to ``attempts``."""
        kwargs = request_kwargs(self._provider.model, messages, policy.max_tokens, policy, request_fields)

        async def send(events: list[str]) -> _Segment:
            return await self._stream(kwargs, events)

        result = await self._retrying(send, segment, policy.max_tokens, attempts)
        if result.finish_reason is FinishReason.TOOL_CALLS and result.usage.completion_tokens >= policy.max_tokens:
            return replace(result, finish_reason=FinishReason.LENGTH)
        return result

    async def _retrying[T](
        self,
        send: Callable[[list[str]], Awaitable[T]],
        segment: int,
        max_tokens: int,
        attempts: list[Attempt],
    ) -> T:
        """Call ``send`` until an attempt succeeds, retrying and holding as needed, and return its result.

        ``send(events)`` makes one request, appends the raw payloads it receives to ``events``, and
        raises ``_AttemptFailed`` for a failure this loop handles. Each attempt, failed or not, is
        appended to ``attempts``.
        """
        # Retries and hold polls back off on separate schedules, so a hold leaves the next retry's delay as it was.
        retry_backoff = self._backoff.copy()
        hold_backoff = self._backoff.copy()
        failures = 0
        hold_started: float | None = None
        while True:
            events: list[str] = []
            started_wall, started = time.time(), time.monotonic()
            try:
                result = await send(events)
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
                if failure.outcome is AttemptOutcome.STALLED:
                    await self._reset_pool()
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
            duration = time.monotonic() - started
            attempts.append(
                Attempt(segment, AttemptOutcome.COMPLETED, max_tokens, started_wall, duration, 200, "", tuple(events))
            )
            return result

    async def _stream(self, kwargs: Mapping[str, Any], events: list[str]) -> _Segment:
        """Stream a request into a segment, appending each chunk's JSON to ``events``."""
        stall_timeout = self._provider.stall_timeout
        # A ``length`` finish drives continuations, but ``get_final_completion`` raises on it, as does
        # ``handle_chunk`` once the state holds strict tools. The client therefore builds the state without
        # tools or a response format and reads its snapshot.
        state = ChatCompletionStreamState()
        started = time.monotonic()
        ttft: float | None = None
        try:
            async with self._leased() as (sdk, _):
                chunks = await sdk.chat.completions.create(**kwargs)
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
                        state.handle_chunk(chunk)
                        if _carries_token(chunk):
                            last_token = time.monotonic()
                            if ttft is None:
                                ttft = last_token - started
                        if any(choice.finish_reason == FinishReason.ABORT for choice in chunk.choices):
                            raise _AttemptFailed(AttemptOutcome.ABORTED, "server aborted the request")
        except TimeoutError as error:
            raise _AttemptFailed(AttemptOutcome.STALLED, f"no token for {stall_timeout}s") from error
        except openai.APIStatusError as error:
            raise _status_failure(error.response) from error
        except openai.APITimeoutError as error:
            raise _AttemptFailed(AttemptOutcome.STALLED, f"no response within the timeout: {error}") from error
        except openai.APIConnectionError as error:
            raise _AttemptFailed(AttemptOutcome.TRANSPORT_ERROR, f"{error}: {error.__cause__!r}") from error
        except openai.APIError as error:
            # The SDK raises the base ``APIError`` for an ``error`` event inside the stream.
            raise _AttemptFailed(AttemptOutcome.STREAM_ERROR, json.dumps(error.body)[:ERROR_BODY_LIMIT]) from error
        except json.JSONDecodeError as error:
            raise _AttemptFailed(AttemptOutcome.STREAM_ERROR, f"malformed SSE data: {error}") from error
        # httpx raises its own errors, unwrapped, for failures while reading the body.
        except httpx.ReadTimeout as error:
            raise _AttemptFailed(AttemptOutcome.STALLED, f"no bytes for {stall_timeout}s") from error
        except httpx.TransportError as error:
            raise _AttemptFailed(AttemptOutcome.TRANSPORT_ERROR, f"{type(error).__name__}: {error}") from error
        if not events:
            raise _AttemptFailed(AttemptOutcome.INCOMPLETE_STREAM, "stream ended before its first chunk")
        snapshot = state.current_completion_snapshot
        if not snapshot.choices or snapshot.choices[0].finish_reason is None or snapshot.usage is None:
            raise _AttemptFailed(AttemptOutcome.INCOMPLETE_STREAM, "stream lacked finish_reason or usage")
        choice = snapshot.choices[0]
        message = choice.message
        return _Segment(
            content=message.content or "",
            # The SDK merges deltas as dicts, so vLLM's extra ``reasoning`` field arrives concatenated.
            reasoning=(message.model_extra or {}).get("reasoning") or "",
            tool_calls=tuple(
                ToolCall(id=call.id, name=call.function.name, arguments=call.function.arguments or "")
                for call in message.tool_calls or ()
            ),
            finish_reason=FinishReason(choice.finish_reason),
            usage=_usage(snapshot.usage),
            ttft=ttft,
            started=started,
            finished=time.monotonic(),
        )

    async def _pool_empty(self) -> bool:
        """Check whether ``/health`` answers 200 with zero workers in this endpoint's pool.

        The check returns False for an unreachable ``/health``, a non-200 answer or a report without
        the pool. The caller then counts the failure against ``max_attempts`` instead of holding.
        """
        endpoint = self._endpoint
        try:
            async with self._leased() as (_, pool):
                response = await pool.get(
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
        """Poll the source and ``/health`` with backoff until ``/health`` stops reporting the pool empty.

        Raises:
            ModelUnavailable: ``deadline`` passed first.
        """
        while True:
            if time.monotonic() >= deadline:
                raise ModelUnavailable(f"GLM infrastructure hold exceeded {self._provider.hold_timeout}s", attempts)
            await asyncio.sleep(backoff.next_interval())
            self._refresh_endpoint()
            if not await self._pool_empty():
                return
