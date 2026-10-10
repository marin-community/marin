# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded direct chat requests with returned request and response observations."""

import json
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import BoundedSemaphore
from typing import Any, Protocol

from zephyr import counters

from taskcompendium.pipeline.review_requests import (
    DEFAULT_MAX_BATCH_BYTES,
    DEFAULT_MAX_BATCH_REQUESTS,
    RequestObservation,
    RequestOutput,
    _unavailable_output,
    request_batches,
)

MAX_DIRECT_CONCURRENT_REQUESTS = 8
# Concurrent reviewer invocations share this process-wide admission limit.
# Zephyr subprocesses each have their own limit; this does not cap the pool.
_DIRECT_REQUEST_CAPACITY = BoundedSemaphore(MAX_DIRECT_CONCURRENT_REQUESTS)


class ChatClient(Protocol):
    def complete(self, body: Mapping[str, Any]) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class ChatResult:
    observation: RequestObservation
    provider_seconds: float
    admission_seconds: float


def _chat_request(client: ChatClient, request: dict[str, Any]) -> ChatResult:
    admission_started = time.monotonic()
    with _DIRECT_REQUEST_CAPACITY:
        admission_seconds = time.monotonic() - admission_started
        started = time.monotonic()
        try:
            response = dict(client.complete(request["body"]))
        except (ConnectionError, TimeoutError) as error:
            output = _unavailable_output([request], "chat_request_failed", str(error))
        else:
            output = json.dumps({"custom_id": request["custom_id"], "response": {"status_code": 200, "body": response}})
        elapsed = time.monotonic() - started
    return ChatResult(
        RequestObservation((request["custom_id"],), output),
        provider_seconds=elapsed,
        admission_seconds=admission_seconds,
    )


def chat_output(
    client: ChatClient,
    requests: Sequence[dict[str, Any]],
    *,
    max_concurrent: int,
    max_batch_bytes: int = DEFAULT_MAX_BATCH_BYTES,
) -> RequestOutput:
    """Issue one direct call per request; the reviewer owns finite retries."""
    if not 1 <= max_concurrent <= MAX_DIRECT_CONCURRENT_REQUESTS:
        raise ValueError("Direct concurrency must be between one and the worker admission cap")
    metrics = counters.current_stage()
    observations: dict[str, RequestObservation] = {}

    def oversized(request: dict[str, Any], request_bytes: int) -> None:
        observations[request["custom_id"]] = RequestObservation(
            (request["custom_id"],),
            _unavailable_output(
                [request],
                "chat_request_too_large",
                f"Request requires {request_bytes} bytes; budget is {max_batch_bytes}",
            ),
        )
        metrics.update_counter("review/direct/oversized_requests", 1)

    groups = request_batches(
        requests, max_requests=DEFAULT_MAX_BATCH_REQUESTS, max_bytes=max_batch_bytes, oversized=oversized
    )
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        for group in groups:
            futures = []
            for request in group:
                futures.append((request, executor.submit(_chat_request, client, request)))
            for request, future in futures:
                result = future.result()
                observations[request["custom_id"]] = result.observation
                metrics.update_counter("review/direct/submitted_requests", 1)
                metrics.update_counter("review/direct/provider_seconds", result.provider_seconds)
                metrics.update_counter("review/direct/admission_seconds", result.admission_seconds)
    ordered = [observations[request["custom_id"]] for request in requests]
    return RequestOutput(tuple(requests), ordered)
