# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
import threading
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import httpx
from rigging.timing import ExponentialBackoff

from marin.inference.types import RunningModel
from marin.inference.workload_broker import (
    LeasedWorkloadRequest,
    LeasedWorkloadResponse,
    WorkloadBroker,
    WorkloadRequest,
    WorkloadResponse,
    format_request_ids,
    pack_json_payload,
    unpack_json_payload,
)

logger = logging.getLogger(__name__)

_UPSTREAM_TEXT_PREVIEW_CHARS = 1000


class VllmWorker:
    """Poll brokered requests and forward them to a local OpenAI-compatible vLLM endpoint."""

    def __init__(
        self,
        *,
        broker: WorkloadBroker,
        upstream: RunningModel,
        request_timeout_seconds: float,
    ) -> None:
        self._broker = broker
        self._upstream = upstream
        self._request_timeout_seconds = request_timeout_seconds

    async def run_forever(
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
        in_flight: set[asyncio.Task[LeasedWorkloadResponse]] = set()
        logger.info(
            "VllmWorker starting upstream=%s model=%s max_in_flight=%d timeout_seconds=%.1f",
            self._upstream.endpoint.base_url,
            self._upstream.endpoint.model,
            max_in_flight,
            self._request_timeout_seconds,
        )
        try:
            async with httpx.AsyncClient(timeout=self._request_timeout_seconds) as client:
                while not stop_event.is_set():
                    available_slots = max_in_flight - len(in_flight)
                    if available_slots:
                        # `max_in_flight` counts individual HTTP requests; vLLM does its own continuous batching.
                        leased_requests = await asyncio.to_thread(self._broker.fetch_requests, max_items=available_slots)
                        for leased_request in leased_requests:
                            in_flight.add(asyncio.create_task(self._forward_one(client, leased_request)))
                        if leased_requests:
                            logger.info(
                                "VllmWorker fetched requests count=%d in_flight=%d/%d request_ids=%s",
                                len(leased_requests),
                                len(in_flight),
                                max_in_flight,
                                format_request_ids(
                                    [leased_request.request.request_id for leased_request in leased_requests]
                                ),
                            )
                            backoff.reset()

                    if not in_flight:
                        await asyncio.sleep(backoff.next_interval())
                        continue

                    done, _ = await asyncio.wait(
                        in_flight,
                        timeout=backoff.next_interval(),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    in_flight.difference_update(done)
                    if done:
                        responses = [task.result() for task in done]
                        await asyncio.to_thread(self._broker.submit_responses, responses)
                        logger.info(
                            "VllmWorker submitted responses count=%d in_flight=%d/%d statuses=%s request_ids=%s",
                            len(responses),
                            len(in_flight),
                            max_in_flight,
                            dict(Counter(response.response.status_code for response in responses)),
                            format_request_ids([response.response.request_id for response in responses]),
                        )
                        backoff.reset()
        finally:
            logger.info("VllmWorker stopping in_flight=%d", len(in_flight))
            if in_flight:
                for task in in_flight:
                    task.cancel()
                await asyncio.gather(*in_flight, return_exceptions=True)

    async def _forward_one(
        self, client: httpx.AsyncClient, leased_request: LeasedWorkloadRequest
    ) -> LeasedWorkloadResponse:
        request = leased_request.request
        # The proxy receives /v1/... paths, while RunningModel.endpoint.url() already points at /v1.
        upstream_path = request.path.removeprefix("/v1/")
        url = self._upstream.endpoint.url(upstream_path)
        try:
            response = await self._send(client, request, url)
            workload_response = _response_from_upstream(request, response)
        except Exception as exc:
            workload_response = _response_from_exception(request, exc, timeout_seconds=self._request_timeout_seconds)
        return LeasedWorkloadResponse(lease_id=leased_request.lease_id, response=workload_response)

    async def _send(self, client: httpx.AsyncClient, request: WorkloadRequest, url: str) -> httpx.Response:
        method = request.method.upper()
        if method == "GET":
            return await client.get(url)
        if method == "POST":
            return await client.post(url, json=unpack_json_payload(request.payload))
        return httpx.Response(
            status_code=405,
            json={"error": f"unsupported brokered request method {request.method!r}"},
            request=httpx.Request(method, url),
        )


@contextmanager
def run_vllm_worker(
    worker: VllmWorker,
    *,
    max_in_flight: int,
    backoff: ExponentialBackoff | None = None,
) -> Iterator[None]:
    stop_event = threading.Event()
    thread = threading.Thread(
        target=lambda: asyncio.run(
            worker.run_forever(stop_event=stop_event, max_in_flight=max_in_flight, backoff=backoff)
        ),
        name="vllm-worker",
    )
    logger.info("Starting VllmWorker thread max_in_flight=%d", max_in_flight)
    thread.start()
    try:
        yield
    finally:
        logger.info("Stopping VllmWorker thread")
        stop_event.set()
        thread.join()


def _response_from_upstream(request: WorkloadRequest, response: httpx.Response) -> WorkloadResponse:
    try:
        payload = response.json()
    except ValueError:
        return _workload_error_response(
            request,
            response.status_code,
            "vLLM returned a non-JSON response",
            body=response.text[:_UPSTREAM_TEXT_PREVIEW_CHARS],
        )
    if not isinstance(payload, dict):
        return _workload_error_response(request, response.status_code, "vLLM returned a non-object JSON response")
    return WorkloadResponse(
        request_id=request.request_id,
        status_code=response.status_code,
        payload=pack_json_payload(payload),
    )


def _response_from_exception(
    request: WorkloadRequest,
    exc: Exception,
    *,
    timeout_seconds: float,
) -> WorkloadResponse:
    if isinstance(exc, httpx.TimeoutException):
        return _workload_error_response(
            request,
            504,
            "timed out forwarding request to vLLM",
            detail=f"timeout_seconds={timeout_seconds:.1f}",
        )
    if isinstance(exc, httpx.HTTPError):
        return _workload_error_response(request, 502, "failed forwarding request to vLLM", detail=repr(exc))
    return _workload_error_response(
        request,
        502,
        "unexpected worker failure while forwarding request to vLLM",
        detail=repr(exc),
        exc_info=True,
    )


def _workload_error_response(
    request: WorkloadRequest,
    status_code: int,
    message: str,
    *,
    body: str | None = None,
    detail: str | None = None,
    exc_info: bool = False,
) -> WorkloadResponse:
    logger.warning(
        "VllmWorker returning error response request_id=%s method=%s path=%s status_code=%d error=%s detail=%s",
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
    return WorkloadResponse(request_id=request.request_id, status_code=status_code, payload=pack_json_payload(payload))
