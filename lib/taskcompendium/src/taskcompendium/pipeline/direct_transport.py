# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded direct chat requests with local request and response evidence."""

import json
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import BoundedSemaphore
from typing import Any, Protocol

from zephyr import counters
from zephyr.writers import write_jsonl_file

from taskcompendium.pipeline.review_transport import (
    DEFAULT_MAX_BATCH_BYTES,
    DEFAULT_MAX_BATCH_REQUESTS,
    RAW_OUTPUT_FILENAME,
    REQUEST_EVIDENCE_FILENAME,
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
class DirectResult:
    output: str
    provider_seconds: float
    admission_seconds: float


def _direct_request(client: ChatClient, request: dict[str, Any], output_path: Path) -> DirectResult:
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "request.json").write_text(json.dumps(request, indent=2))
    admission_started = time.monotonic()
    with _DIRECT_REQUEST_CAPACITY:
        admission_seconds = time.monotonic() - admission_started
        started = time.monotonic()
        try:
            response = dict(client.complete(request["body"]))
        except (ConnectionError, TimeoutError) as error:
            output = _unavailable_output([request], "direct_transport_failed", str(error))
        else:
            (output_path / "response.json").write_text(json.dumps(response, indent=2))
            output = json.dumps({"custom_id": request["custom_id"], "response": {"status_code": 200, "body": response}})
        elapsed = time.monotonic() - started
    (output_path / RAW_OUTPUT_FILENAME).write_text(output)
    return DirectResult(output, provider_seconds=elapsed, admission_seconds=admission_seconds)


def direct_output(
    client: ChatClient,
    requests: Sequence[dict[str, Any]],
    output_path: Path,
    *,
    max_concurrent: int,
    max_batch_bytes: int = DEFAULT_MAX_BATCH_BYTES,
) -> str:
    """Issue one direct call per request; the reviewer owns finite retries."""
    if not 1 <= max_concurrent <= MAX_DIRECT_CONCURRENT_REQUESTS:
        raise ValueError("Direct concurrency must be between one and the worker admission cap")
    output_path.mkdir(parents=True, exist_ok=True)
    write_jsonl_file(requests, str(output_path / REQUEST_EVIDENCE_FILENAME))
    metrics = counters.current_stage()
    outputs: dict[str, str] = {}

    def oversized(request: dict[str, Any], request_bytes: int) -> None:
        outputs[request["custom_id"]] = _unavailable_output(
            [request], "direct_request_too_large", f"Request requires {request_bytes} bytes; budget is {max_batch_bytes}"
        )
        metrics.update_counter("review/direct/oversized_requests", 1)

    groups = request_batches(
        requests, max_requests=DEFAULT_MAX_BATCH_REQUESTS, max_bytes=max_batch_bytes, oversized=oversized
    )
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        index = 0
        for group in groups:
            futures = []
            for request in group:
                futures.append(
                    (request, executor.submit(_direct_request, client, request, output_path / "direct" / str(index)))
                )
                index += 1
            for request, future in futures:
                result = future.result()
                outputs[request["custom_id"]] = result.output
                metrics.update_counter("review/direct/submitted_requests", 1)
                metrics.update_counter("review/direct/provider_seconds", result.provider_seconds)
                metrics.update_counter("review/direct/admission_seconds", result.admission_seconds)
    output = "\n".join(outputs[request["custom_id"]] for request in requests)
    (output_path / RAW_OUTPUT_FILENAME).write_text(output)
    return output
