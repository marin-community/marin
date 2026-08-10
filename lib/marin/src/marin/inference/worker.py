# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import threading
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from typing import Any

import httpx
from rigging.timing import ExponentialBackoff

from marin.inference.http_proxy import forwardable_response_headers
from marin.inference.types import (
    InferenceRequest,
    InferenceRequestProvider,
    InferenceResponse,
    LeasedInferenceRequest,
    LeasedInferenceResponse,
    RunningModel,
    format_request_ids,
)

logger = logging.getLogger(__name__)


class InferenceWorker:
    """Poll brokered requests and forward them to an OpenAI-compatible endpoint."""

    def __init__(
        self,
        *,
        broker: InferenceRequestProvider,
        upstream: RunningModel,
        request_timeout_seconds: float,
    ) -> None:
        self._broker = broker
        self._upstream = upstream
        self._request_timeout_seconds = request_timeout_seconds

    def run_forever(
        self,
        *,
        max_in_flight: int,
        stop_event: threading.Event | None = None,
        backoff: ExponentialBackoff | None = None,
    ) -> None:
        if max_in_flight < 1:
            raise ValueError("max_in_flight must be at least 1")
        if stop_event is None:
            stop_event = threading.Event()
        backoff = ExponentialBackoff() if backoff is None else backoff.copy()
        in_flight: set[Future[LeasedInferenceResponse]] = set()
        logger.info(
            "InferenceWorker starting upstream=%s model=%s max_in_flight=%d timeout_seconds=%.1f",
            self._upstream.endpoint.base_url,
            self._upstream.endpoint.model,
            max_in_flight,
            self._request_timeout_seconds,
        )
        try:
            with (
                httpx.Client(timeout=self._request_timeout_seconds) as client,
                ThreadPoolExecutor(max_workers=max_in_flight, thread_name_prefix="inference-worker-request") as pool,
            ):
                while not stop_event.is_set():
                    available_slots = max_in_flight - len(in_flight)
                    if available_slots:
                        # `max_in_flight` counts individual HTTP requests; vLLM does its own continuous batching.
                        leased_requests = self._broker.fetch_requests(max_items=available_slots)
                        for leased_request in leased_requests:
                            in_flight.add(pool.submit(self._forward_one, client, leased_request))
                        if leased_requests:
                            logger.info(
                                "InferenceWorker fetched requests count=%d in_flight=%d/%d request_ids=%s",
                                len(leased_requests),
                                len(in_flight),
                                max_in_flight,
                                format_request_ids(
                                    [leased_request.request.request_id for leased_request in leased_requests]
                                ),
                            )
                            backoff.reset()

                    if not in_flight:
                        stop_event.wait(backoff.next_interval())
                        continue

                    done, _ = wait(
                        in_flight,
                        timeout=backoff.next_interval(),
                        return_when=FIRST_COMPLETED,
                    )
                    in_flight.difference_update(done)
                    if done:
                        responses = [future.result() for future in done]
                        self._broker.submit_responses(responses)
                        logger.info(
                            "InferenceWorker submitted responses count=%d in_flight=%d/%d statuses=%s request_ids=%s",
                            len(responses),
                            len(in_flight),
                            max_in_flight,
                            dict(Counter(response.response.status_code for response in responses)),
                            format_request_ids([response.response.request_id for response in responses]),
                        )
                        backoff.reset()
        finally:
            logger.info("InferenceWorker stopping in_flight=%d", len(in_flight))

    def _forward_one(self, client: httpx.Client, leased_request: LeasedInferenceRequest) -> LeasedInferenceResponse:
        request = leased_request.request
        # The proxy receives /v1/... paths, while RunningModel.endpoint.url() already points at /v1.
        upstream_path = request.path.removeprefix("/v1/")
        url = self._upstream.endpoint.url(upstream_path)
        try:
            response = self._send(client, request, url)
            inference_response = _response_from_upstream(request, response)
        except Exception as exc:
            inference_response = _response_from_exception(request, exc, timeout_seconds=self._request_timeout_seconds)
        return LeasedInferenceResponse(lease_id=leased_request.lease_id, response=inference_response)

    def _send(self, client: httpx.Client, request: InferenceRequest, url: str) -> httpx.Response:
        method = request.method.upper()
        if method not in {"GET", "POST", "OPTIONS"}:
            return httpx.Response(
                status_code=405,
                json={"error": f"unsupported brokered request method {request.method!r}"},
                request=httpx.Request(method, url),
            )
        if request.query_string:
            url = f"{url}?{request.query_string}"
        return client.request(
            method,
            url,
            content=request.payload,
            headers=dict(request.headers),
        )


@contextmanager
def run_inference_worker(
    worker: InferenceWorker,
    *,
    max_in_flight: int,
    backoff: ExponentialBackoff | None = None,
) -> Iterator[None]:
    stop_event = threading.Event()
    thread = threading.Thread(
        target=lambda: worker.run_forever(stop_event=stop_event, max_in_flight=max_in_flight, backoff=backoff),
        name="inference-worker",
    )
    logger.info("Starting InferenceWorker thread max_in_flight=%d", max_in_flight)
    thread.start()
    try:
        yield
    finally:
        logger.info("Stopping InferenceWorker thread")
        stop_event.set()
        thread.join()


def _response_from_upstream(request: InferenceRequest, response: httpx.Response) -> InferenceResponse:
    return InferenceResponse(
        request_id=request.request_id,
        status_code=response.status_code,
        payload=response.content,
        headers=tuple(forwardable_response_headers(response.headers).items()),
    )


def _response_from_exception(
    request: InferenceRequest,
    exc: Exception,
    *,
    timeout_seconds: float,
) -> InferenceResponse:
    if isinstance(exc, httpx.TimeoutException):
        return _inference_error_response(
            request,
            504,
            "timed out forwarding request to upstream endpoint",
            detail=f"timeout_seconds={timeout_seconds:.1f}",
        )
    if isinstance(exc, httpx.HTTPError):
        return _inference_error_response(
            request, 502, "failed forwarding request to upstream endpoint", detail=repr(exc)
        )
    return _inference_error_response(
        request,
        502,
        "unexpected worker failure while forwarding request to upstream endpoint",
        detail=repr(exc),
        exc_info=True,
    )


def _inference_error_response(
    request: InferenceRequest,
    status_code: int,
    message: str,
    *,
    detail: str | None = None,
    exc_info: bool = False,
) -> InferenceResponse:
    logger.warning(
        "InferenceWorker returning error response request_id=%s method=%s path=%s status_code=%d error=%s detail=%s",
        request.request_id,
        request.method,
        request.path,
        status_code,
        message,
        detail or "-",
        exc_info=exc_info,
    )
    error: dict[str, Any] = {"message": message}
    payload: dict[str, Any] = {"error": error}
    return InferenceResponse(
        request_id=request.request_id,
        status_code=status_code,
        payload=json.dumps(payload).encode(),
        headers=(("content-type", "application/json"),),
    )
