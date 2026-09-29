# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Batch-priority GLM transport for concurrent science conversion requests."""

import asyncio
import json
import logging
from dataclasses import dataclass
from uuid import uuid4

import httpx
from marin.inference.openai_batch import CHAT_COMPLETIONS_ENDPOINT, OpenAIBatchClient

logger = logging.getLogger(__name__)

BATCH_POLL_SECONDS = 5.0
BATCH_TIMEOUT_SECONDS = 3600.0
BATCH_FLUSH_SECONDS = 1.0


@dataclass(frozen=True)
class PendingCompletion:
    custom_id: str
    request: httpx.Request
    body: dict
    result: asyncio.Future[httpx.Response]


class GLMBatchChatClient:
    """Route concurrent chat requests through Ortet's batch API in bounded groups."""

    def __init__(self, base_url: str, token: str, batch_size: int, workers: int) -> None:
        if min(batch_size, workers) < 1:
            raise ValueError("batch_size and workers must be positive")
        self._client = OpenAIBatchClient(base_url.rstrip("/") + "/bulk/v1", token, priority="batch")
        self._queue: asyncio.Queue[PendingCompletion | None] = asyncio.Queue()
        self._batch_size = batch_size
        self._workers = workers
        self._tasks: list[asyncio.Task[None]] = []

    async def __aenter__(self) -> "GLMBatchChatClient":
        self._tasks = [asyncio.create_task(self._consume()) for _ in range(self._workers)]
        return self

    async def __aexit__(self, exception_type, exception, traceback) -> None:
        if exception_type is not None:
            for task in self._tasks:
                task.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
            return
        for _ in self._tasks:
            await self._queue.put(None)
        await asyncio.gather(*self._tasks)

    async def post(self, url: str, *, json: dict) -> httpx.Response:
        if not self._tasks:
            raise RuntimeError("GLM batch client must be entered before use")
        request = httpx.Request("POST", url)
        future: asyncio.Future[httpx.Response] = asyncio.get_running_loop().create_future()
        await self._queue.put(PendingCompletion(uuid4().hex, request, json, future))
        return await future

    def _submit_and_read(self, pending: list[PendingCompletion]) -> tuple[str, str | None]:
        requests = [
            {"custom_id": item.custom_id, "method": "POST", "url": CHAT_COMPLETIONS_ENDPOINT, "body": item.body}
            for item in pending
        ]
        submission = self._client.submit(requests, f"science-sft-{uuid4().hex}.jsonl")
        logger.info("Submitted GLM batch %s with %d requests", submission.batch_id, len(requests))
        batch = self._client.wait(submission.batch_id, BATCH_POLL_SECONDS, timeout_seconds=BATCH_TIMEOUT_SECONDS)
        if batch.get("status") != "completed":
            raise RuntimeError(f"GLM batch {submission.batch_id} ended in state {batch.get('status')}")
        output = self._client.output(batch)
        return output.output, output.errors

    async def _consume(self) -> None:
        while True:
            first = await self._queue.get()
            if first is None:
                return
            pending = [first]
            stop_after_batch = False
            deadline = asyncio.get_running_loop().time() + BATCH_FLUSH_SECONDS
            while len(pending) < self._batch_size:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                except TimeoutError:
                    break
                if item is None:
                    stop_after_batch = True
                    break
                pending.append(item)

            try:
                output, errors = await asyncio.to_thread(self._submit_and_read, pending)
                responses = {row["custom_id"]: row for line in output.splitlines() if (row := json.loads(line))}
                if errors:
                    logger.warning("GLM batch returned request errors: %s", errors[:1000])
                for item in pending:
                    row = responses.get(item.custom_id)
                    response = row.get("response") if row is not None else None
                    body = response.get("body") if isinstance(response, dict) else None
                    status = response.get("status_code") if isinstance(response, dict) else None
                    if row is None or row.get("error") is not None or not isinstance(body, dict):
                        status = 502
                        body = {"error": row.get("error") if row is not None else "Missing GLM batch response"}
                    if not item.result.done():
                        item.result.set_result(httpx.Response(status or 502, json=body, request=item.request))
            except Exception as error:
                logger.exception("GLM batch request failed")
                for item in pending:
                    if not item.result.done():
                        item.result.set_exception(httpx.TransportError(str(error), request=item.request))
            if stop_after_batch:
                return
