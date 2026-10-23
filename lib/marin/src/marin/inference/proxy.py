# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import html
import json
import logging
import socket
import threading
import time
import uuid
from collections import deque
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterator, Mapping, Sequence
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass, replace

import anyio
import uvicorn
from rigging.timing import Duration, ExponentialBackoff
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Route
from starlette.types import Receive, Scope, Send

from marin.inference.http_proxy import forwardable_request_headers
from marin.inference.types import (
    STREAM_BUFFER_BYTES,
    InferenceBackendState,
    InferenceRequest,
    InferenceResponse,
    InferenceResponseProvider,
    OpenAIEndpoint,
    RunningModel,
    StreamChunk,
    StreamEnd,
    StreamError,
    StreamEvent,
    StreamStart,
    format_request_ids,
)

logger = logging.getLogger(__name__)
TOKEN_RATE_WINDOW_SECONDS = 60.0


@dataclass
class PendingStream:
    headers: Future[StreamStart | InferenceResponse]
    chunks: asyncio.Queue[StreamEvent]
    loop: asyncio.AbstractEventLoop
    buffered_bytes: int = 0
    terminal: bool = False


class BrokerStreamingResponse(StreamingResponse):
    """Close the upstream request even if HTTP sending fails outside its iterator."""

    def __init__(self, chunks: AsyncGenerator[bytes], start: StreamStart, cleanup: Callable[[], Awaitable[None]]):
        super().__init__(chunks, status_code=start.status_code, headers=dict(start.headers))
        self._chunks = chunks
        self._cleanup = cleanup

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):
                try:
                    await self._chunks.aclose()
                finally:
                    await self._cleanup()


@dataclass
class ProxyStats:
    matched_responses: int = 0
    dropped_responses: int = 0
    timed_out_requests: int = 0
    rejected_requests: int = 0
    generation_responses: int = 0
    responses_with_usage: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


def response_token_usage(payload: bytes) -> tuple[int, int] | None:
    """Read optional OpenAI usage without disrupting response forwarding."""
    try:
        document = json.loads(payload)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(document, dict) or not isinstance(document.get("usage"), dict):
        return None
    usage = document["usage"]
    prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if not isinstance(prompt, int) or not isinstance(completion, int):
        return None
    if isinstance(prompt, bool) or isinstance(completion, bool) or prompt < 0 or completion < 0:
        return None
    return prompt, completion


class InferenceProxy:
    """OpenAI-compatible local HTTP proxy backed by an inference response provider."""

    def __init__(
        self,
        *,
        broker: InferenceResponseProvider,
        model: str,
        request_timeout_seconds: float,
        readiness_timeout_seconds: float,
        max_pending_requests: int,
        response_fetch_batch_size: int,
        ignored_request_fields: Sequence[str] = (),
        backoff: ExponentialBackoff | None = None,
    ) -> None:
        if max_pending_requests < 1:
            raise ValueError("max_pending_requests must be at least 1")
        if response_fetch_batch_size < 1:
            raise ValueError("response_fetch_batch_size must be at least 1")
        self._broker = broker
        self._started_at = time.monotonic()
        self._token_samples: deque[tuple[float, int, int]] = deque()
        self._model = model
        self._request_timeout_seconds = request_timeout_seconds
        self._readiness_timeout_seconds = readiness_timeout_seconds
        self._max_pending_requests = max_pending_requests
        self._response_fetch_batch_size = response_fetch_batch_size
        self._ignored_request_fields = frozenset(ignored_request_fields)
        if backoff is None:
            backoff = ExponentialBackoff(initial=0.01, maximum=0.25, factor=2.0)
        self._backoff = backoff
        self._pending: dict[str, Future[InferenceResponse]] = {}
        self._streams: dict[str, PendingStream] = {}
        self._lock = threading.Lock()
        self._poll_stop_event: threading.Event | None = None
        self._poll_thread: threading.Thread | None = None
        self.stats = ProxyStats()
        self.app = Starlette(
            routes=[
                Route("/", self._status),
                Route("/status", self._status),
                Route("/health", self._health),
                Route("/ready", self._ready),
                Route("/tokenize", self._forward, methods=["POST"]),
                Route("/metrics", self._forward, methods=["GET"]),
                Route("/v1/{path:path}", self._forward, methods=["GET", "POST", "OPTIONS"]),
            ],
        )

    def __enter__(self) -> "InferenceProxy":
        self.start_response_poller()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.stop_response_poller()

    def start_response_poller(self) -> None:
        logger.info(
            "InferenceProxy starting model=%s timeout_seconds=%.1f max_pending_requests=%d",
            self._model,
            self._request_timeout_seconds,
            self._max_pending_requests,
        )
        stop_event = threading.Event()
        self._poll_stop_event = stop_event
        self._poll_thread = threading.Thread(
            target=lambda: self.run_forever(stop_event=stop_event),
            name="inference-proxy-response-poller",
        )
        self._poll_thread.start()

    def stop_response_poller(self) -> None:
        thread = self._poll_thread
        stop_event = self._poll_stop_event
        if thread is None or stop_event is None:
            return
        logger.info("InferenceProxy stopping response poller")
        stop_event.set()
        thread.join()
        self._poll_thread = None
        self._poll_stop_event = None
        logger.info("InferenceProxy stopped stats=%s", self.stats)

    def _health(self, _request: Request) -> Response:
        ready = self._poll_thread is not None and self._poll_thread.is_alive()
        return JSONResponse({"status": "ok" if ready else "stopped"}, status_code=200 if ready else 503)

    async def _ready(self, _request: Request) -> Response:
        snapshot = await anyio.to_thread.run_sync(self._broker.status)
        return JSONResponse(
            {"status": snapshot.readiness.value, "recovery_count": snapshot.recovery_count},
            status_code=200 if snapshot.readiness is InferenceBackendState.READY else 503,
        )

    async def _status(self, _request: Request) -> Response:
        snapshot = await anyio.to_thread.run_sync(self._broker.status)
        with self._lock:
            pending = len(self._pending)
            stats = replace(self.stats)
            now = time.monotonic()
            while self._token_samples and self._token_samples[0][0] <= now - TOKEN_RATE_WINDOW_SECONDS:
                self._token_samples.popleft()
            window_prompt = sum(sample[1] for sample in self._token_samples)
            window_completion = sum(sample[2] for sample in self._token_samples)
        elapsed = max(now - self._started_at, 0.001)
        window = min(elapsed, TOKEN_RATE_WINDOW_SECONDS)
        metrics = (
            ("Proxy uptime", f"{elapsed:.1f} seconds"),
            ("Prompt tokens", stats.prompt_tokens),
            ("Output tokens", stats.completion_tokens),
            ("Total tokens", stats.prompt_tokens + stats.completion_tokens),
            ("Prompt tokens/s (last 60s)", f"{window_prompt / window:.1f}"),
            ("Output tokens/s (last 60s)", f"{window_completion / window:.1f}"),
            ("Total tokens/s (last 60s)", f"{(window_prompt + window_completion) / window:.1f}"),
            ("Output tokens/s (lifetime)", f"{stats.completion_tokens / elapsed:.1f}"),
            ("Prompt tokens/s (lifetime)", f"{stats.prompt_tokens / elapsed:.1f}"),
            ("Total tokens/s (lifetime)", f"{(stats.prompt_tokens + stats.completion_tokens) / elapsed:.1f}"),
            ("Generation responses with usage", f"{stats.responses_with_usage} / {stats.generation_responses}"),
            ("Proxy outstanding", f"{pending} / {self._max_pending_requests}"),
            ("Broker queued", snapshot.queued_requests),
            ("Leased to workers", snapshot.leased_requests),
            ("Responses awaiting proxy", snapshot.buffered_responses),
            ("Submitted", snapshot.submitted_requests),
            ("Dispatched", snapshot.dispatched_requests),
            ("Completed", snapshot.completed_requests),
            ("HTTP errors from workers", snapshot.failed_responses),
            ("Mean queue wait", f"{snapshot.mean_queue_wait_seconds:.3f} seconds"),
            ("Max queue wait", f"{snapshot.max_queue_wait_seconds:.3f} seconds"),
            ("Proxy timeouts", stats.timed_out_requests),
            ("Proxy rejections", stats.rejected_requests),
            ("Broker stale responses", snapshot.dropped_responses),
            ("Proxy stale responses", stats.dropped_responses),
        )
        rows = "".join(f"<tr><th>{label}</th><td>{value}</td></tr>" for label, value in metrics)
        workers = "".join(
            f"<tr><td>{html.escape(worker)}</td><td>{html.escape(metadata.backend_name)}</td>"
            f"<td>{metadata.tensor_parallel_size}</td></tr>"
            for worker, metadata in sorted(snapshot.workers.items())
        )
        return HTMLResponse(
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<meta http-equiv="refresh" content="5"><title>Inference broker</title>'
            "<style>body{font:16px system-ui;max-width:900px;margin:2rem auto;padding:0 1rem}"
            "table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{text-align:left;"
            "padding:.4rem;border-bottom:1px solid #ddd}th{font-weight:500}</style></head><body>"
            f"<h1>Inference broker</h1><p>Model: {html.escape(self._model)}</p>"
            "<p>Refreshes every 5 seconds. Counts cover this broker process. Queue wait measures "
            "submission to first worker lease; inference time is excluded.</p>"
            "<p>Token counts come from successful completion responses with valid usage. Rates count "
            "tokens when responses finish, across all workers; they are not instantaneous GPU rates. "
            "The 60-second window uses elapsed proxy lifetime until the first minute. "
            "Missing usage is excluded. Totals reset when the proxy restarts.</p>"
            f"<table>{rows}</table><h2>Registered workers ({len(snapshot.workers)})</h2>"
            "<p>Registration does not imply current health.</p><table><tr><th>Worker</th>"
            f"<th>Backend</th><th>Tensor parallel size</th></tr>{workers}</table></body></html>"
        )

    async def _forward(self, request: Request) -> Response:
        body = await request.body()
        streaming = False
        timeout_seconds = (
            self._readiness_timeout_seconds if request.url.path == "/v1/models" else self._request_timeout_seconds
        )
        content_type = request.headers.get("content-type", "")
        if body and "application/json" in content_type:
            try:
                request_json = json.loads(body)
            except ValueError:
                return JSONResponse({"error": "request body must be valid JSON"}, status_code=400)
            if not isinstance(request_json, dict):
                return JSONResponse({"error": "request body must be a JSON object"}, status_code=400)
            streaming = request_json.get("stream") is True
            if self._ignored_request_fields:
                request_json = {
                    key: value for key, value in request_json.items() if key not in self._ignored_request_fields
                }
                body = json.dumps(request_json, separators=(",", ":")).encode()
        response: Response = JSONResponse({"error": "client disconnected"}, status_code=499)
        async with anyio.create_task_group() as tasks:

            async def disconnected() -> None:
                while True:
                    if (await request.receive())["type"] == "http.disconnect":
                        tasks.cancel_scope.cancel()
                        return

            tasks.start_soon(disconnected)
            response = await self.forward_raw_request(
                request.url.path,
                body,
                method=request.method,
                query_string=request.url.query,
                headers=forwardable_request_headers(request.headers),
                timeout_seconds=timeout_seconds,
                streaming=streaming,
            )
            tasks.cancel_scope.cancel()
        return response

    async def forward_raw_request(
        self,
        path: str,
        body: bytes,
        *,
        method: str,
        query_string: str = "",
        headers: Mapping[str, str] | None = None,
        timeout_seconds: float | None = None,
        streaming: bool = False,
    ) -> Response:
        if streaming:
            return await self._forward_stream(
                path,
                body,
                method=method,
                query_string=query_string,
                headers=headers,
                timeout_seconds=self._request_timeout_seconds if timeout_seconds is None else timeout_seconds,
            )
        request_id = str(uuid.uuid4())
        future: Future[InferenceResponse] = Future()
        pending_count = 0
        timeout_seconds = self._request_timeout_seconds if timeout_seconds is None else timeout_seconds
        with self._lock:
            if len(self._pending) + len(self._streams) >= self._max_pending_requests:
                self.stats.rejected_requests += 1
                logger.warning(
                    "InferenceProxy rejecting request method=%s path=%s pending=%d/%d",
                    method,
                    path,
                    len(self._pending),
                    self._max_pending_requests,
                )
                return JSONResponse(
                    {"error": "too many pending proxy requests; back off and retry"},
                    status_code=429,
                    headers={"Retry-After": "1"},
                )
            self._pending[request_id] = future
            pending_count = len(self._pending)

        try:
            broker_request = InferenceRequest(
                request_id=request_id,
                method=method,
                path=path,
                payload=body,
                query_string=query_string,
                headers=tuple((headers or {}).items()),
            )
            await anyio.to_thread.run_sync(lambda: self._broker.submit_request(broker_request))
            logger.info(
                "InferenceProxy submitted request request_id=%s method=%s path=%s pending=%d/%d",
                request_id,
                method,
                path,
                pending_count,
                self._max_pending_requests,
            )
            # Waiting for inference must not occupy AnyIO's bounded thread pool.
            response = await asyncio.wait_for(asyncio.shield(asyncio.wrap_future(future)), timeout=timeout_seconds)
        except TimeoutError:
            self.stats.timed_out_requests += 1
            logger.warning(
                "InferenceProxy timed out waiting for response request_id=%s method=%s path=%s timeout_seconds=%.1f",
                request_id,
                method,
                path,
                timeout_seconds,
            )
            return JSONResponse({"error": f"timed out waiting for inference response {request_id}"}, status_code=504)
        finally:
            with self._lock:
                self._pending.pop(request_id, None)
                if not future.done():
                    future.cancel()
            with anyio.CancelScope(shield=True):
                await anyio.to_thread.run_sync(lambda: self._broker.cancel_request(request_id))

        logger.info(
            "InferenceProxy returning response request_id=%s method=%s path=%s status_code=%d",
            request_id,
            method,
            path,
            response.status_code,
        )
        if path in {"/v1/completions", "/v1/chat/completions"} and 200 <= response.status_code < 300:
            usage = response_token_usage(response.payload)
            with self._lock:
                self.stats.generation_responses += 1
                if usage is not None:
                    prompt, completion = usage
                    self.stats.responses_with_usage += 1
                    self.stats.prompt_tokens += prompt
                    self.stats.completion_tokens += completion
                    now = time.monotonic()
                    while self._token_samples and self._token_samples[0][0] <= now - TOKEN_RATE_WINDOW_SECONDS:
                        self._token_samples.popleft()
                    self._token_samples.append((now, prompt, completion))
        return Response(content=response.payload, status_code=response.status_code, headers=dict(response.headers))

    async def _forward_stream(
        self,
        path: str,
        body: bytes,
        *,
        method: str,
        query_string: str,
        headers: Mapping[str, str] | None,
        timeout_seconds: float,
    ) -> Response:
        request_id = str(uuid.uuid4())
        stream = PendingStream(Future(), asyncio.Queue(), asyncio.get_running_loop())
        with self._lock:
            if len(self._pending) + len(self._streams) >= self._max_pending_requests:
                return JSONResponse(
                    {"error": "too many pending proxy requests"}, status_code=429, headers={"Retry-After": "1"}
                )
            self._streams[request_id] = stream
        deadline = stream.loop.time() + timeout_seconds

        async def cleanup() -> None:
            with anyio.CancelScope(shield=True):
                with self._lock:
                    self._streams.pop(request_id, None)
                await anyio.to_thread.run_sync(lambda: self._broker.cancel_request(request_id))

        try:
            await anyio.to_thread.run_sync(
                lambda: self._broker.submit_request(
                    InferenceRequest(
                        request_id,
                        method,
                        path,
                        body,
                        query_string,
                        tuple((headers or {}).items()),
                        streaming=True,
                    )
                )
            )
            response = await asyncio.wait_for(
                asyncio.wrap_future(stream.headers), timeout=max(0, deadline - stream.loop.time())
            )
        except TimeoutError:
            await cleanup()
            return JSONResponse({"error": "timed out waiting for inference"}, status_code=504)
        except BaseException:
            await cleanup()
            raise
        if isinstance(response, InferenceResponse):
            await cleanup()
            return Response(response.payload, status_code=response.status_code, headers=dict(response.headers))

        async def chunks() -> AsyncGenerator[bytes]:
            async with asyncio.timeout(max(0, deadline - stream.loop.time())):
                while True:
                    event = await stream.chunks.get()
                    if isinstance(event, StreamEnd):
                        return
                    if isinstance(event, StreamError):
                        raise RuntimeError(event.message)
                    if isinstance(event, StreamChunk):
                        with self._lock:
                            stream.buffered_bytes -= len(event.payload)
                        yield event.payload

        return BrokerStreamingResponse(chunks(), response, cleanup)

    def _stream_event(self, response: InferenceResponse | StreamEvent, stream: PendingStream) -> None:
        if stream.terminal:
            return
        if isinstance(response, (StreamStart, InferenceResponse)):
            if not stream.headers.done():
                stream.headers.set_result(response)
            return
        if isinstance(response, StreamChunk):
            stream.buffered_bytes += len(response.payload)
            if stream.buffered_bytes > STREAM_BUFFER_BYTES:
                response = StreamError(response.request_id, response.lease_id, response.sequence, "stream buffer full")
        if isinstance(response, (StreamEnd, StreamError)):
            stream.terminal = True
        stream.loop.call_soon_threadsafe(stream.chunks.put_nowait, response)

    def run_forever(
        self,
        *,
        stop_event: threading.Event | None = None,
        backoff: ExponentialBackoff | None = None,
    ) -> None:
        if stop_event is None:
            stop_event = threading.Event()
        backoff = self._backoff.copy() if backoff is None else backoff.copy()
        while not stop_event.is_set():
            response_count = self.tick()
            if response_count:
                backoff.reset()
                continue
            stop_event.wait(backoff.next_interval())

    def tick(self, *, max_responses: int | None = None) -> int:
        max_responses = self._response_fetch_batch_size if max_responses is None else max_responses
        responses = self._broker.fetch_responses(max_items=max_responses)
        matched_ids: list[str] = []
        dropped_ids: list[str] = []
        with self._lock:
            for response in responses:
                stream = self._streams.get(response.request_id)
                if stream is not None:
                    self._stream_event(response, stream)
                    continue
                if not isinstance(response, InferenceResponse):
                    dropped_ids.append(response.request_id)
                    continue
                future = self._pending.pop(response.request_id, None)
                if future is None or future.done():
                    dropped_ids.append(response.request_id)
                    continue
                future.set_result(response)
                matched_ids.append(response.request_id)
            pending_count = len(self._pending)
            self.stats.matched_responses += len(matched_ids)
            self.stats.dropped_responses += len(dropped_ids)
        if responses:
            logger.info(
                "InferenceProxy fetched responses count=%d matched=%d dropped=%d "
                "pending=%d matched_ids=%s dropped_ids=%s",
                len(responses),
                len(matched_ids),
                len(dropped_ids),
                pending_count,
                format_request_ids(matched_ids),
                format_request_ids(dropped_ids),
            )
        if dropped_ids:
            logger.warning(
                "InferenceProxy dropped stale or duplicate inference responses count=%d request_ids=%s; "
                "likely causes are proxy timeout or request lease expiry before worker response. "
                "Increase request_timeout_seconds or investigate slow/unhealthy workers if this repeats.",
                len(dropped_ids),
                format_request_ids(dropped_ids),
            )
        return len(responses)


@contextmanager
def serve_inference_proxy(
    *,
    broker: InferenceResponseProvider,
    model: str,
    host: str = "127.0.0.1",
    port: int = 0,
    request_timeout_seconds: float,
    readiness_timeout_seconds: float,
    max_pending_requests: int,
    response_fetch_batch_size: int,
    server_start_timeout_seconds: float,
    ignored_request_fields: Sequence[str] = (),
    backoff: ExponentialBackoff | None = None,
) -> Iterator[RunningModel]:
    actual_port = _reserve_port(host, port)
    proxy = InferenceProxy(
        broker=broker,
        model=model,
        request_timeout_seconds=request_timeout_seconds,
        readiness_timeout_seconds=readiness_timeout_seconds,
        max_pending_requests=max_pending_requests,
        response_fetch_batch_size=response_fetch_batch_size,
        ignored_request_fields=ignored_request_fields,
        backoff=backoff,
    )
    config = uvicorn.Config(proxy.app, host=host, port=actual_port, log_level="error", log_config=None)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="inference-proxy")
    logger.info("Starting InferenceProxy server url=http://%s:%d/v1 model=%s", host, actual_port, model)
    with proxy:
        thread.start()
        started = ExponentialBackoff(initial=0.01, maximum=1, jitter=0).wait_until(
            lambda: server.started or not thread.is_alive(),
            timeout=Duration.from_seconds(server_start_timeout_seconds),
        )
        if not started or not server.started:
            server.should_exit = True
            thread.join()
            raise RuntimeError("Inference proxy failed to start")
        try:
            yield RunningModel(endpoint=OpenAIEndpoint(base_url=f"http://{host}:{actual_port}/v1", model=model))
        finally:
            logger.info("Stopping InferenceProxy server url=http://%s:%d/v1", host, actual_port)
            server.should_exit = True
            thread.join()


def _reserve_port(host: str, port: int) -> int:
    if port != 0:
        return port
    # Mirrors the existing Iris embedded-server pattern: reserve a port before uvicorn starts.
    with socket.socket() as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])
