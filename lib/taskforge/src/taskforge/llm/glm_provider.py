# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The library's default ``Provider``: the GLM router's chat completions through the ``openai`` SDK.

Each ``stream`` call is one streamed request with the SDK's retries off; ``ModelClient`` owns every retry. A stream
is abandoned after ``stall_timeout`` seconds without a token, never on wall-clock time, and a request that finds no
free pooled connection within ``POOL_TIMEOUT`` is a transport failure. A 400 naming the context window (vLLM's and
sglang's wordings, or the code ``context_length_exceeded``) is a context overflow; a 404 naming the route is a missing
route; every 5xx is an outage.

The endpoint resolves at the first request, and again after a refused connection or a missing route (a moved
relay), so an unregistered relay is unavailability the client waits out. The lookup runs in a thread, because an
Iris registry lookup blocks and retries for up to 30 minutes; one runs at a time, and each request waits for it at
most ``RESOLVE_TIMEOUT`` seconds. An httpx pool works only in the event loop
that opened it, so each loop gets its own pool and SDK client, and a pickled provider carries only its settings. A
stalled read replaces the loop's pool, at most once per ``POOL_RESET_INTERVAL``; requests still streaming on the
old pool keep it until they end.
"""

import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from types import TracebackType
from typing import Any, Protocol, Self, override

import httpx
import openai
from openai.lib.streaming.chat import ChatCompletionStreamState
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletionChunk

from taskforge.llm.client import RETRYABLE_STATUSES, AttemptFailed, AttemptOutcome, Progress, Provider, Segment
from taskforge.model.llm import FinishReason, ToolCall, Usage

logger = logging.getLogger(__name__)

API_ROOT = "/v1"
"""The path suffix of an OpenAI-compatible base URL. The router serves ``/health`` at the URL without it."""
CONTEXT_OVERFLOW = re.compile(r"context length|maximum model length", re.IGNORECASE)
CONTEXT_OVERFLOW_CODE = "context_length_exceeded"
ERROR_BODY_LIMIT = 2000
CONNECT_TIMEOUT = 10.0
WRITE_TIMEOUT = 30.0
POOL_TIMEOUT = 30.0
"""Seconds a request waits for a free connection in the local pool; longer is local admission, not the model."""
RESOLVE_TIMEOUT = 30.0
"""Seconds a request waits for the endpoint lookup in flight before failing as unavailable; the lookup keeps running."""
KEEPALIVE_EXPIRY = 30.0
"""Seconds an idle pooled connection may be reused, so a connection idle through a long author turn is not
reused blindly; a stalled read also replaces the whole pool (``_reset_pool``)."""
POOL_RESET_INTERVAL = 30.0
"""Least seconds between two pool resets, so a burst of stalls on one bad path replaces the pool once."""
HEALTH_TIMEOUT = 20.0
RERESOLVE = frozenset({AttemptOutcome.TRANSPORT_ERROR, AttemptOutcome.ROUTE_MISSING})
"""Failures after which the endpoint is resolved again: a relay that moved is refused or has no route."""


class Pool(StrEnum):
    """Router worker pool; the bearer token selects the pool serving a request."""

    HIGH = "high"
    BULK = "bulk"


@dataclass(frozen=True)
class GlmEndpoint:
    """An OpenAI-compatible base URL ending in ``/v1``, with its bearer token and the pool behind it."""

    base_url: str
    token: str = field(repr=False)
    pool: Pool

    def __post_init__(self) -> None:
        if not self.base_url.endswith(API_ROOT):
            raise ValueError(f"base_url must end in {API_ROOT}: {self.base_url}")


class EndpointUnavailable(Exception):
    """The source has no endpoint right now, such as a relay not yet registered with a fresh Iris controller."""


class EndpointSource(Protocol):
    """Supplies the GLM endpoint; raises ``EndpointUnavailable`` while there is none."""

    def current(self) -> GlmEndpoint: ...


@dataclass(frozen=True)
class FixedEndpoint:
    """Supplies the same endpoint on every call."""

    endpoint: GlmEndpoint

    def current(self) -> GlmEndpoint:
        return self.endpoint


def _carries_token(chunk: ChatCompletionChunk) -> bool:
    """Whether a chunk carries generated text, including vLLM's ``delta.reasoning`` (absent from the SDK schema)."""
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


def overflowed(status: int, body: str) -> bool:
    """Whether a rejection is of a conversation longer than the model's context window, by error code or message."""
    try:
        detail = json.loads(body)["error"]
        code, message = detail.get("code"), str(detail.get("message", ""))
    except (ValueError, KeyError, TypeError, AttributeError):
        code, message = None, body
    return status == 400 and (code == CONTEXT_OVERFLOW_CODE or bool(CONTEXT_OVERFLOW.search(message)))


def _status_failure(response: httpx.Response) -> AttemptFailed:
    status = response.status_code
    body = response.text[:ERROR_BODY_LIMIT]
    if status == 404 and "route" in body.lower():
        return AttemptFailed(AttemptOutcome.ROUTE_MISSING, body, status)
    if overflowed(status, body):
        return AttemptFailed(AttemptOutcome.CONTEXT_OVERFLOW, body, status)
    if status in RETRYABLE_STATUSES or status >= 500:
        return AttemptFailed(AttemptOutcome.RETRYABLE_STATUS, body, status, _retry_after(response))
    return AttemptFailed(AttemptOutcome.REJECTED, body, status)


@dataclass(eq=False)
class _LoopPool:
    """The connection pool of one event loop, the SDK client on it for the latest endpoint, and requests in flight."""

    http: httpx.AsyncClient
    sdk: openai.AsyncOpenAI | None = None
    endpoint: GlmEndpoint | None = None
    in_flight: int = 0


class GlmProvider(Provider):
    """The GLM router as a ``Provider``. Use as ``async with GlmProvider(...) as provider``.

    Args:
        source: Supplies the endpoint, at the first request and again after a refused connection or missing route.
        model: The model id sent with each request; it must match the endpoint's model.
        stall_timeout: Seconds without a streamed token before an attempt is abandoned.
        max_connections: httpx connection-pool size; sized for hundreds of concurrent calls.
    """

    def __init__(self, source: EndpointSource, model: str, stall_timeout: float, max_connections: int) -> None:
        self._source, self._model = source, model
        self.stall_timeout, self.max_connections = stall_timeout, max_connections
        self._endpoint: GlmEndpoint | None = None
        self._stale = True
        self._lookup: asyncio.Future[GlmEndpoint] | None = None
        self._pools: dict[asyncio.AbstractEventLoop, _LoopPool] = {}
        self._retired: dict[asyncio.AbstractEventLoop, list[_LoopPool]] = {}
        self._pool_reset_at = -POOL_RESET_INTERVAL

    def __reduce__(self) -> tuple[type[Self], tuple[EndpointSource, str, float, int]]:
        return type(self), (self._source, self._model, self.stall_timeout, self.max_connections)

    @property
    @override
    def model(self) -> str:
        return self._model

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

    async def _current(self) -> GlmEndpoint:
        """The endpoint, resolved at first use and again after a failure that a moved relay causes."""
        if not self._stale and self._endpoint is not None:
            return self._endpoint
        loop = asyncio.get_running_loop()
        if self._lookup is None or self._lookup.get_loop() is not loop:
            self._lookup = loop.run_in_executor(None, self._source.current)
        lookup = self._lookup
        try:
            endpoint = await asyncio.wait_for(asyncio.shield(lookup), RESOLVE_TIMEOUT)
        except TimeoutError as error:
            raise AttemptFailed(AttemptOutcome.TRANSPORT_ERROR, f"no endpoint within {RESOLVE_TIMEOUT}s") from error
        except EndpointUnavailable as error:
            raise AttemptFailed(AttemptOutcome.TRANSPORT_ERROR, f"no endpoint: {error}") from error
        finally:
            if lookup.done() and self._lookup is lookup:
                self._lookup = None
        if self._endpoint is not None and endpoint != self._endpoint:
            logger.warning("GLM endpoint moved from %s to %s", self._endpoint.base_url, endpoint.base_url)
        self._endpoint, self._stale = endpoint, False
        return endpoint

    def _open_pool(self) -> _LoopPool:
        limit = self.max_connections
        limits = httpx.Limits(max_connections=limit, max_keepalive_connections=limit, keepalive_expiry=KEEPALIVE_EXPIRY)
        return _LoopPool(httpx.AsyncClient(limits=limits))

    def _pool(self) -> _LoopPool:
        """The running event loop's pool, opened on first use; closed loops' pools are dropped."""
        for closed in [loop for loop in self._pools.keys() | self._retired.keys() if loop.is_closed()]:
            self._pools.pop(closed, None)
            self._retired.pop(closed, None)
        loop = asyncio.get_running_loop()
        if loop not in self._pools:
            self._pools[loop] = self._open_pool()
        return self._pools[loop]

    async def _reset_pool(self) -> None:
        """Open a fresh pool after a stalled read, so the retry cannot reuse a dead connection.

        The old pool is retired and closes when the last request leased from it ends (``_leased``).
        """
        now = time.monotonic()
        loop = asyncio.get_running_loop()
        old = self._pools.get(loop)
        if old is None or now - self._pool_reset_at < POOL_RESET_INTERVAL:
            return
        self._pool_reset_at = now
        logger.warning(
            "GLM read stalled; replacing the connection pool (%d requests still on the old one)", old.in_flight
        )
        self._pools[loop] = self._open_pool()
        if old.in_flight:
            self._retired.setdefault(loop, []).append(old)
        else:
            await old.http.aclose()

    @asynccontextmanager
    async def _leased(self) -> AsyncIterator[tuple[openai.AsyncOpenAI, httpx.AsyncClient, GlmEndpoint]]:
        """The running event loop's SDK client and its pool, which stays open until the request using them ends."""
        endpoint = await self._current()
        pool = self._pool()
        sdk = pool.sdk if pool.sdk is not None and pool.endpoint == endpoint else self._sdk_client(endpoint, pool.http)
        pool.sdk, pool.endpoint = sdk, endpoint
        pool.in_flight += 1
        try:
            yield sdk, pool.http, endpoint
        finally:
            pool.in_flight -= 1
            retired = self._retired.get(asyncio.get_running_loop(), [])
            if not pool.in_flight and pool in retired:
                retired.remove(pool)
                await pool.http.aclose()

    def _sdk_client(self, endpoint: GlmEndpoint, http: httpx.AsyncClient) -> openai.AsyncOpenAI:
        timeout = httpx.Timeout(connect=CONNECT_TIMEOUT, read=self.stall_timeout, write=WRITE_TIMEOUT, pool=POOL_TIMEOUT)
        return openai.AsyncOpenAI(
            base_url=endpoint.base_url, api_key=endpoint.token, max_retries=0, timeout=timeout, http_client=http
        )

    @override
    async def stream(self, body: Mapping[str, Any], progress: Progress) -> Segment:
        try:
            return await self._stream(body, progress)
        except AttemptFailed as failure:
            self._stale = self._stale or failure.outcome in RERESOLVE
            if failure.outcome is AttemptOutcome.STALLED:
                await self._reset_pool()
            raise

    async def _stream(self, body: Mapping[str, Any], progress: Progress) -> Segment:
        """Stream a request into a segment, reporting each chunk's JSON to ``progress``."""
        stall_timeout = self.stall_timeout
        # ``get_final_completion`` raises on a ``length`` finish, as does ``handle_chunk`` once the state holds
        # strict tools, so the provider builds the state without tools or a response format and reads its snapshot.
        state = ChatCompletionStreamState()
        started = time.monotonic()
        ttft: float | None = None
        try:
            async with self._leased() as (sdk, _, _):
                model, messages = body["model"], body["messages"]
                extra = {key: value for key, value in body.items() if key not in ("model", "messages")}
                chunks = await sdk.chat.completions.create(
                    model=model, messages=messages, stream=True, stream_options={"include_usage": True}, extra_body=extra
                )
                assert isinstance(chunks, openai.AsyncStream)
                progress.admit()
                async with chunks:
                    last_token = time.monotonic()
                    while True:
                        remaining = stall_timeout - (time.monotonic() - last_token)
                        try:
                            chunk = await asyncio.wait_for(anext(chunks), timeout=max(remaining, 0.0))
                        except StopAsyncIteration:
                            break
                        progress.event(chunk.model_dump_json(), _carries_token(chunk))
                        try:
                            state.handle_chunk(chunk)
                        except IndexError as error:
                            # The accumulator indexes tool-call deltas by their index and raises on one
                            # whose earlier deltas never arrived; such a stream is the server's fault.
                            raise AttemptFailed(
                                AttemptOutcome.STREAM_ERROR, f"tool-call delta without its earlier deltas: {error}"
                            ) from error
                        if _carries_token(chunk):
                            last_token = time.monotonic()
                            if ttft is None:
                                ttft = last_token - started
                        if any(choice.finish_reason == FinishReason.ABORT for choice in chunk.choices):
                            raise AttemptFailed(AttemptOutcome.ABORTED, "server aborted the request")
        except TimeoutError as error:
            raise AttemptFailed(AttemptOutcome.STALLED, f"no token for {stall_timeout}s") from error
        except openai.APIStatusError as error:
            raise _status_failure(error.response) from error
        except openai.APITimeoutError as error:
            if isinstance(error.__cause__, httpx.PoolTimeout):
                raise AttemptFailed(AttemptOutcome.TRANSPORT_ERROR, f"no free connection in {POOL_TIMEOUT}s") from error
            raise AttemptFailed(AttemptOutcome.STALLED, f"no response within the timeout: {error}") from error
        except openai.APIConnectionError as error:
            raise AttemptFailed(AttemptOutcome.TRANSPORT_ERROR, f"{error}: {error.__cause__!r}") from error
        except openai.APIError as error:
            # The SDK raises the base ``APIError`` for an ``error`` event inside the stream.
            raise AttemptFailed(AttemptOutcome.STREAM_ERROR, json.dumps(error.body)[:ERROR_BODY_LIMIT]) from error
        except json.JSONDecodeError as error:
            raise AttemptFailed(AttemptOutcome.STREAM_ERROR, f"malformed SSE data: {error}") from error
        # httpx raises its own errors, unwrapped, for failures while reading the body.
        except httpx.ReadTimeout as error:
            raise AttemptFailed(AttemptOutcome.STALLED, f"no bytes for {stall_timeout}s") from error
        except httpx.TransportError as error:
            raise AttemptFailed(AttemptOutcome.TRANSPORT_ERROR, f"{type(error).__name__}: {error}") from error
        if not progress.events:
            raise AttemptFailed(AttemptOutcome.INCOMPLETE_STREAM, "stream ended before its first chunk")
        snapshot = state.current_completion_snapshot
        if not snapshot.choices or snapshot.choices[0].finish_reason is None or snapshot.usage is None:
            raise AttemptFailed(AttemptOutcome.INCOMPLETE_STREAM, "stream lacked finish_reason or usage")
        choice = snapshot.choices[0]
        message = choice.message
        return Segment(
            content=message.content or "",
            # The SDK merges deltas as dicts, so vLLM's extra ``reasoning`` field arrives concatenated.
            reasoning=(message.model_extra or {}).get("reasoning") or "",
            tool_calls=tuple(
                ToolCall(call.id, call.function.name, call.function.arguments or "") for call in message.tool_calls or ()
            ),
            finish_reason=FinishReason(choice.finish_reason),
            usage=_usage(snapshot.usage),
            ttft=ttft,
            started=started,
            finished=time.monotonic(),
        )

    @override
    async def no_workers(self) -> bool:
        """Whether ``/health`` answers 200 with zero workers in this endpoint's pool.

        False, unknown, for an unresolvable endpoint, an unreachable ``/health``, a non-200 answer or a report that is
        not JSON or lacks the pool.
        """
        try:
            async with self._leased() as (_, http, endpoint):
                response = await http.get(
                    f"{endpoint.base_url.removesuffix(API_ROOT)}/health",
                    headers={"Authorization": f"Bearer {endpoint.token}"},
                    timeout=HEALTH_TIMEOUT,
                )
        except (AttemptFailed, httpx.TransportError):
            return False
        if response.status_code != 200:
            return False
        try:
            return response.json()["workers"][str(endpoint.pool)] == 0
        except (ValueError, KeyError, TypeError):
            return False
