# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""An httpx transport that sends the chat completions of many concurrent callers to a GLM relay's batch API.

``GlmBatchTransport`` takes each ``POST .../chat/completions`` of an ``openai.AsyncOpenAI`` client, groups the
requests that arrive within ``flush`` seconds, at most ``batch_size`` of them, into one batch of the relay's
``/bulk/v1`` API at ``priority=batch``, and answers each request with its row of the batch output. One transport
serves all the agent sessions of a process, so their turns share batches. Each batch writes one log line with its
size, tokens, latency and errors, which Iris forwards to Finelog.
"""

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
from marin.inference.openai_batch import CHAT_COMPLETIONS_ENDPOINT, BatchOutput, BatchSubmission, OpenAIBatchClient

logger = logging.getLogger(__name__)

BATCH_PRIORITY = "batch"
BULK_API_ROOT = "/bulk/v1"
CHAT_COMPLETIONS_PATH = "/chat/completions"
BATCH_LOG_EVENT = "glm_batch"
"""The ``event`` of the JSON log line of each batch."""


class BatchApi(Protocol):
    """The calls of an OpenAI-compatible batch API that the transport makes, as ``OpenAIBatchClient`` has them."""

    def submit(self, requests: Sequence[Mapping[str, Any]], filename: str) -> BatchSubmission: ...

    def wait(self, batch_id: str, poll_seconds: float) -> dict[str, Any]: ...

    def output(self, batch: Mapping[str, Any]) -> BatchOutput: ...


def bulk_batch_api(base_url: str, token: str) -> OpenAIBatchClient:
    """The batch API of the relay at ``base_url``, the URL without ``/v1``, at ``priority=batch``."""
    return OpenAIBatchClient(base_url.rstrip("/") + BULK_API_ROOT, token, priority=BATCH_PRIORITY)


@dataclass(frozen=True)
class _Pending:
    custom_id: str
    body: dict[str, Any]
    request: httpx.Request
    result: asyncio.Future[httpx.Response]


class GlmBatchTransport(httpx.AsyncBaseTransport):
    """Answer chat completion requests from the batches of ``api``, for example ``bulk_batch_api``.

    A request that the batch does not answer, or a batch that fails, gives a 502 response, which the OpenAI client
    retries. At most ``concurrent_batches`` batches wait at the same time; each waits in a thread of its own.
    """

    def __init__(
        self,
        api: BatchApi,
        *,
        batch_size: int,
        flush: float,
        poll: float,
        concurrent_batches: int,
    ):
        if min(batch_size, concurrent_batches) < 1:
            raise ValueError("batch_size and concurrent_batches must be positive")
        self._client = api
        self._batch_size = batch_size
        self._flush = flush
        self._poll = poll
        self._concurrent_batches = concurrent_batches
        self._queue: asyncio.Queue[_Pending] = asyncio.Queue()
        self._executor = ThreadPoolExecutor(max_workers=concurrent_batches, thread_name_prefix="glm-batch")
        self._consumers: list[asyncio.Task[None]] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.method != "POST" or not request.url.path.endswith(CHAT_COMPLETIONS_PATH):
            raise ValueError(f"The batch transport serves only chat completions, not {request.method} {request.url}")
        if not self._consumers:
            # The consumers start on the first request, in the event loop of the client.
            self._consumers = [asyncio.create_task(self._consume()) for _ in range(self._concurrent_batches)]
        body = json.loads(await request.aread())
        body.pop("stream", None)
        future: asyncio.Future[httpx.Response] = asyncio.get_running_loop().create_future()
        await self._queue.put(_Pending(uuid.uuid4().hex, body, request, future))
        return await future

    async def aclose(self) -> None:
        for consumer in self._consumers:
            consumer.cancel()
        await asyncio.gather(*self._consumers, return_exceptions=True)
        self._consumers = []
        self._executor.shutdown(wait=False, cancel_futures=True)

    async def _consume(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            pending = [await self._queue.get()]
            deadline = loop.time() + self._flush
            while len(pending) < self._batch_size and (remaining := deadline - loop.time()) > 0:
                try:
                    pending.append(await asyncio.wait_for(self._queue.get(), timeout=remaining))
                except TimeoutError:
                    break
            started = time.monotonic()
            try:
                rows, errors = await loop.run_in_executor(self._executor, self._run_batch, pending)
            except Exception as error:
                logger.exception("A GLM batch of %d requests failed", len(pending))
                rows, errors = {}, str(error)
            _log_batch(pending, rows, errors, time.monotonic() - started)
            for item in pending:
                if not item.result.done():
                    item.result.set_result(_response(item, rows.get(item.custom_id)))

    def _run_batch(self, pending: list[_Pending]) -> tuple[dict[str, dict[str, Any]], str | None]:
        requests = [
            {"custom_id": item.custom_id, "method": "POST", "url": CHAT_COMPLETIONS_ENDPOINT, "body": item.body}
            for item in pending
        ]
        submission = self._client.submit(requests, f"taskforge-drafts-{uuid.uuid4().hex}.jsonl")
        batch = self._client.wait(submission.batch_id, self._poll)
        if batch.get("status") != "completed":
            return {}, f"batch {submission.batch_id} ended in state {batch.get('status')}"
        output = self._client.output(batch)
        rows = {
            row["custom_id"]: row for line in output.output.splitlines() if line.strip() and (row := json.loads(line))
        }
        return rows, output.errors


def _response(item: _Pending, row: dict[str, Any] | None) -> httpx.Response:
    """The HTTP response of one request from its row of the batch output, or a 502 when it has none."""
    response = row.get("response") if row is not None else None
    if row is None or row.get("error") is not None or not isinstance(response, dict):
        error = row.get("error") if row is not None else None
        return httpx.Response(
            502, json={"error": error or "The GLM batch has no response for this request"}, request=item.request
        )
    body = response.get("body")
    if not isinstance(body, dict):
        return httpx.Response(502, json={"error": "The GLM batch response has no body"}, request=item.request)
    return httpx.Response(response.get("status_code") or 200, json=body, request=item.request)


def _log_batch(pending: list[_Pending], rows: dict[str, dict[str, Any]], errors: str | None, seconds: float) -> None:
    usages = [
        ((row.get("response") or {}).get("body") or {}).get("usage") or {} for row in rows.values() if row is not None
    ]
    cached = sum(((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0) for usage in usages)
    record = {
        "event": BATCH_LOG_EVENT,
        "requests": len(pending),
        "answered": sum(1 for item in pending if item.custom_id in rows and rows[item.custom_id].get("error") is None),
        "prompt_tokens": sum(usage.get("prompt_tokens") or 0 for usage in usages),
        "cached_tokens": cached,
        "completion_tokens": sum(usage.get("completion_tokens") or 0 for usage in usages),
        "seconds": round(seconds, 2),
        "errors": (errors or "")[:500],
    }
    logger.info(json.dumps(record))
