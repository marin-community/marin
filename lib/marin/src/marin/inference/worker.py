# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import threading
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
from rigging.timing import ExponentialBackoff

from marin.inference.types import (
    InferenceRequest,
    InferenceRequestProvider,
    InferenceResponse,
    LeasedInferenceRequest,
    LeasedInferenceResponse,
    RunningModel,
    format_request_ids,
    pack_json_payload,
    unpack_json_payload,
)

logger = logging.getLogger(__name__)

# Keep brokered error payloads bounded when upstream returns arbitrary text.
ERROR_BODY_LIMIT_BYTES = 1000


class InferenceWorker:
    """Poll brokered requests and process them with a request handler."""

    def __init__(
        self,
        *,
        broker: InferenceRequestProvider,
        handler: "InferenceRequestHandler",
        request_timeout_seconds: float,
    ) -> None:
        self._broker = broker
        self._handler = handler
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
        logger.info("InferenceWorker starting max_in_flight=%d", max_in_flight)
        try:
            with (
                httpx.Client(timeout=self._request_timeout_seconds) as client,
                ThreadPoolExecutor(max_workers=max_in_flight, thread_name_prefix="inference-worker-request") as pool,
            ):
                while not stop_event.is_set():
                    available_slots = max_in_flight - len(in_flight)
                    if available_slots:
                        leased_requests = self._broker.fetch_requests(max_items=available_slots)
                        for leased_request in leased_requests:
                            in_flight.add(pool.submit(self._handle_one, client, leased_request))
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

    def _handle_one(self, client: httpx.Client, leased_request: LeasedInferenceRequest) -> LeasedInferenceResponse:
        request = leased_request.request
        try:
            inference_response = self._handler(client, request)
        except Exception as exc:
            inference_response = _response_from_exception(request, exc, timeout_seconds=self._request_timeout_seconds)
        return LeasedInferenceResponse(lease_id=leased_request.lease_id, response=inference_response)


class InferenceRequestHandler(Protocol):
    """Process one brokered inference request."""

    def __call__(self, client: httpx.Client, request: InferenceRequest) -> InferenceResponse: ...


@dataclass(frozen=True)
class ForwardingInferenceHandler:
    """Forward brokered requests to an OpenAI-compatible endpoint."""

    upstream: RunningModel

    def __call__(self, client: httpx.Client, request: InferenceRequest) -> InferenceResponse:
        upstream_path = request.path.removeprefix("/v1/")
        url = self.upstream.endpoint.url(upstream_path)
        method = request.method.upper()
        if method == "GET":
            return _response_from_upstream(request, client.get(url))
        if method == "POST":
            return _response_from_upstream(
                request,
                client.post(url, json=unpack_json_payload(request.payload)),
            )
        return _inference_error_response(request, 405, f"unsupported brokered request method {request.method!r}")


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
    try:
        payload = response.json()
    except ValueError:
        return _inference_error_response(
            request,
            response.status_code,
            "upstream endpoint returned a non-JSON response",
            body=response.content[:ERROR_BODY_LIMIT_BYTES].decode(errors="replace"),
        )
    if not isinstance(payload, dict):
        return _inference_error_response(
            request, response.status_code, "upstream endpoint returned a non-object JSON response"
        )
    return InferenceResponse(
        request_id=request.request_id,
        status_code=response.status_code,
        payload=pack_json_payload(payload),
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
    body: str | None = None,
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
    if body is not None:
        error["body"] = body
    payload: dict[str, Any] = {"error": error}
    return InferenceResponse(request_id=request.request_id, status_code=status_code, payload=pack_json_payload(payload))
