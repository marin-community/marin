# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
import socket
import threading
import uuid
from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager, suppress
from dataclasses import dataclass
from typing import Any

import uvicorn
from rigging.timing import Duration, ExponentialBackoff
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from marin.inference.types import OpenAIEndpoint, RunningModel
from marin.inference.workload_broker import (
    WorkloadBroker,
    WorkloadRequest,
    WorkloadResponse,
    format_request_ids,
    pack_json_payload,
    unpack_json_payload,
)

logger = logging.getLogger(__name__)


@dataclass
class ProxyStats:
    matched_responses: int = 0
    dropped_responses: int = 0
    timed_out_requests: int = 0
    rejected_requests: int = 0


class VllmHttpProxy:
    """OpenAI-compatible local HTTP proxy backed by a WorkloadBroker."""

    def __init__(
        self,
        *,
        broker: WorkloadBroker,
        model: str,
        request_timeout_seconds: float,
        readiness_timeout_seconds: float,
        max_pending_requests: int,
        response_fetch_batch_size: int,
        backoff: ExponentialBackoff | None = None,
    ) -> None:
        if max_pending_requests < 1:
            raise ValueError("max_pending_requests must be at least 1")
        if response_fetch_batch_size < 1:
            raise ValueError("response_fetch_batch_size must be at least 1")
        self._broker = broker
        self._model = model
        self._request_timeout_seconds = request_timeout_seconds
        self._readiness_timeout_seconds = readiness_timeout_seconds
        self._max_pending_requests = max_pending_requests
        self._response_fetch_batch_size = response_fetch_batch_size
        if backoff is None:
            backoff = ExponentialBackoff(initial=0.01, maximum=0.25, factor=2.0)
        self._backoff = backoff
        self._pending: dict[str, asyncio.Future[WorkloadResponse]] = {}
        self._lock = asyncio.Lock()
        self.stats = ProxyStats()
        self.app = Starlette(
            routes=[
                Route("/v1/models", self._models),
                Route("/v1/completions", self._forward, methods=["POST"]),
                Route("/v1/chat/completions", self._forward, methods=["POST"]),
            ],
            lifespan=self._lifespan,
        )

    @asynccontextmanager
    async def _lifespan(self, _app: Starlette) -> AsyncIterator[None]:
        logger.info(
            "VllmHttpProxy starting model=%s timeout_seconds=%.1f max_pending_requests=%d",
            self._model,
            self._request_timeout_seconds,
            self._max_pending_requests,
        )
        poll_task = asyncio.create_task(self.run_forever())
        try:
            yield
        finally:
            logger.info("VllmHttpProxy stopping response poller")
            poll_task.cancel()
            with suppress(asyncio.CancelledError):
                await poll_task
            logger.info("VllmHttpProxy stopped stats=%s", self.stats)

    async def _models(self, request: Request) -> Response:
        return await self.forward_request(
            request.url.path,
            {},
            method="GET",
            timeout_seconds=self._readiness_timeout_seconds,
        )

    async def _forward(self, request: Request) -> Response:
        request_json = await request.json()
        if not isinstance(request_json, dict):
            return JSONResponse({"error": "request body must be a JSON object"}, status_code=400)
        return await self.forward_request(request.url.path, request_json, method=request.method)

    async def forward_request(
        self,
        path: str,
        request_json: Mapping[str, Any],
        *,
        method: str,
        timeout_seconds: float | None = None,
    ) -> Response:
        request_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        future: asyncio.Future[WorkloadResponse] = loop.create_future()
        pending_count = 0
        timeout_seconds = self._request_timeout_seconds if timeout_seconds is None else timeout_seconds
        async with self._lock:
            if len(self._pending) >= self._max_pending_requests:
                self.stats.rejected_requests += 1
                logger.warning(
                    "VllmHttpProxy rejecting request method=%s path=%s pending=%d/%d",
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
            await asyncio.to_thread(
                self._broker.submit_request,
                WorkloadRequest(
                    request_id=request_id,
                    method=method,
                    path=path,
                    payload=pack_json_payload(request_json),
                ),
            )
            logger.info(
                "VllmHttpProxy submitted request request_id=%s method=%s path=%s pending=%d/%d",
                request_id,
                method,
                path,
                pending_count,
                self._max_pending_requests,
            )
            async with asyncio.timeout(timeout_seconds):
                response = await future
        except TimeoutError:
            self.stats.timed_out_requests += 1
            logger.warning(
                "VllmHttpProxy timed out waiting for response request_id=%s method=%s path=%s timeout_seconds=%.1f",
                request_id,
                method,
                path,
                timeout_seconds,
            )
            return JSONResponse({"error": f"timed out waiting for workload response {request_id}"}, status_code=504)
        finally:
            async with self._lock:
                self._pending.pop(request_id, None)
                if not future.done():
                    future.cancel()

        logger.info(
            "VllmHttpProxy returning response request_id=%s method=%s path=%s status_code=%d",
            request_id,
            method,
            path,
            response.status_code,
        )
        return JSONResponse(unpack_json_payload(response.payload), status_code=response.status_code)

    async def run_forever(self, *, backoff: ExponentialBackoff | None = None) -> None:
        backoff = self._backoff.copy() if backoff is None else backoff.copy()
        while True:
            response_count = await self.tick()
            if response_count:
                backoff.reset()
                continue
            await asyncio.sleep(backoff.next_interval())

    async def tick(self, *, max_responses: int | None = None) -> int:
        max_responses = self._response_fetch_batch_size if max_responses is None else max_responses
        responses = await asyncio.to_thread(self._broker.fetch_responses, max_items=max_responses)
        matched_ids: list[str] = []
        dropped_ids: list[str] = []
        async with self._lock:
            for response in responses:
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
                "VllmHttpProxy fetched responses count=%d matched=%d dropped=%d "
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
                "VllmHttpProxy dropped stale or duplicate workload responses count=%d request_ids=%s; "
                "likely causes are proxy timeout or request lease expiry before worker response. "
                "Increase request_timeout_seconds or investigate slow/unhealthy workers if this repeats.",
                len(dropped_ids),
                format_request_ids(dropped_ids),
            )
        return len(responses)


@contextmanager
def serve_vllm_http_proxy(
    *,
    broker: WorkloadBroker,
    model: str,
    host: str = "127.0.0.1",
    port: int = 0,
    request_timeout_seconds: float,
    readiness_timeout_seconds: float,
    max_pending_requests: int,
    response_fetch_batch_size: int,
    server_start_timeout_seconds: float,
    backoff: ExponentialBackoff | None = None,
) -> Iterator[RunningModel]:
    actual_port = _reserve_port(host, port)
    proxy = VllmHttpProxy(
        broker=broker,
        model=model,
        request_timeout_seconds=request_timeout_seconds,
        readiness_timeout_seconds=readiness_timeout_seconds,
        max_pending_requests=max_pending_requests,
        response_fetch_batch_size=response_fetch_batch_size,
        backoff=backoff,
    )
    config = uvicorn.Config(proxy.app, host=host, port=actual_port, log_level="error", log_config=None)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="vllm-http-proxy")
    logger.info("Starting VllmHttpProxy server url=http://%s:%d/v1 model=%s", host, actual_port, model)
    thread.start()
    started = ExponentialBackoff(initial=0.01, maximum=1, jitter=0).wait_until(
        lambda: server.started or not thread.is_alive(),
        timeout=Duration.from_seconds(server_start_timeout_seconds),
    )
    if not started or not server.started:
        server.should_exit = True
        thread.join()
        raise RuntimeError("vLLM HTTP proxy failed to start")
    try:
        yield RunningModel(endpoint=OpenAIEndpoint(base_url=f"http://{host}:{actual_port}/v1", model=model))
    finally:
        logger.info("Stopping VllmHttpProxy server url=http://%s:%d/v1", host, actual_port)
        server.should_exit = True
        thread.join()


def _reserve_port(host: str, port: int) -> int:
    if port != 0:
        return port
    # Mirrors the existing Iris embedded-server pattern: reserve a port before uvicorn starts.
    with socket.socket() as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])
