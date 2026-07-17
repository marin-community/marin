# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import socket
import threading
import uuid
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import anyio
import uvicorn
from rigging.timing import Duration, ExponentialBackoff
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from marin.inference.types import (
    InferenceRequest,
    InferenceResponse,
    InferenceResponseProvider,
    OpenAIEndpoint,
    RunningModel,
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
        self._lock = threading.Lock()
        self._poll_stop_event: threading.Event | None = None
        self._poll_thread: threading.Thread | None = None
        self.stats = ProxyStats()
        self.app = Starlette(
            routes=[
                Route("/v1/models", self._models),
                Route("/v1/completions", self._forward, methods=["POST"]),
                Route("/v1/chat/completions", self._forward, methods=["POST"]),
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

    def _models(self, request: Request) -> Response:
        return self.forward_request(
            request.url.path,
            {},
            method="GET",
            timeout_seconds=self._readiness_timeout_seconds,
        )

    def _forward(self, request: Request) -> Response:
        # Starlette runs sync endpoints in a worker thread, but request.json()
        # still reads the async ASGI body stream.
        request_json = anyio.from_thread.run(request.json)
        if not isinstance(request_json, dict):
            return JSONResponse({"error": "request body must be a JSON object"}, status_code=400)
        return self.forward_request(request.url.path, request_json, method=request.method)

    def forward_request(
        self,
        path: str,
        request_json: Mapping[str, Any],
        *,
        method: str,
        timeout_seconds: float | None = None,
    ) -> Response:
        request_id = str(uuid.uuid4())
        future: Future[InferenceResponse] = Future()
        pending_count = 0
        timeout_seconds = self._request_timeout_seconds if timeout_seconds is None else timeout_seconds
        with self._lock:
            if len(self._pending) >= self._max_pending_requests:
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
            forwarded_json = {
                key: value for key, value in request_json.items() if key not in self._ignored_request_fields
            }
            self._broker.submit_request(
                InferenceRequest(
                    request_id=request_id,
                    method=method,
                    path=path,
                    payload=pack_json_payload(forwarded_json),
                ),
            )
            logger.info(
                "InferenceProxy submitted request request_id=%s method=%s path=%s pending=%d/%d",
                request_id,
                method,
                path,
                pending_count,
                self._max_pending_requests,
            )
            response = future.result(timeout=timeout_seconds)
        except FutureTimeoutError:
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

        logger.info(
            "InferenceProxy returning response request_id=%s method=%s path=%s status_code=%d",
            request_id,
            method,
            path,
            response.status_code,
        )
        return JSONResponse(unpack_json_payload(response.payload), status_code=response.status_code)

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
