# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Web search and page extraction through the REST API of Parallel (https://docs.parallel.ai).

``ParallelWeb`` implements ``taskforge.llm.web.Web``. ``search`` posts to ``/v1/search`` and ``extract``
to ``/v1/extract``. A 400 or 422 means the arguments were rejected and raises ``WebRequestRejected``.
A 408, 429, 5xx or transport error is retried with exponential backoff, up to ``MAX_ATTEMPTS``
requests; after that, and on every other non-2xx status, the ``httpx`` error raises for the caller.
"""

import asyncio
import logging
from collections.abc import Sequence
from types import TracebackType
from typing import Any, Self

import httpx
from rigging.timing import ExponentialBackoff
from taskforge.llm.client import RETRYABLE_STATUSES
from taskforge.llm.web import WebRequestRejected

logger = logging.getLogger(__name__)

PARALLEL_API_URL = "https://api.parallel.ai/v1"
REQUEST_TIMEOUT = 180.0
MAX_ATTEMPTS = 6
MODEL_ERROR_STATUSES = frozenset({400, 422})


class ParallelWeb:
    """An async client for ``/search`` and ``/extract``. Use it as an async context manager."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = PARALLEL_API_URL,
        timeout: float = REQUEST_TIMEOUT,
        backoff: ExponentialBackoff | None = None,
    ):
        self._client = httpx.AsyncClient(base_url=base_url, headers={"x-api-key": api_key}, timeout=timeout)
        self._backoff = backoff or ExponentialBackoff(initial=1.0, maximum=30.0, factor=2.0)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, kind: type[BaseException] | None, error: BaseException | None, traceback: TracebackType | None
    ) -> None:
        await self._client.aclose()

    async def search(self, objective: str, queries: Sequence[str]) -> dict[str, Any]:
        """Results with URLs, titles and excerpts for a natural-language objective and its search queries."""
        return await self._post("/search", {"objective": objective, "search_queries": list(queries)})

    async def extract(self, urls: Sequence[str], objective: str | None) -> dict[str, Any]:
        """The excerpts of each page, focused on ``objective`` when given, and an error for each page that failed."""
        body: dict[str, Any] = {"urls": list(urls)}
        if objective is not None:
            body["objective"] = objective
        return await self._post("/extract", body)

    async def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        delays = self._backoff.copy()
        attempt = 0
        while True:
            attempt += 1
            try:
                response = await self._client.post(path, json=body)
                if response.status_code in MODEL_ERROR_STATUSES:
                    raise WebRequestRejected(response.status_code, response.text)
                response.raise_for_status()
                return response.json()
            except (httpx.HTTPStatusError, httpx.TransportError) as error:
                retryable = isinstance(error, httpx.TransportError) or error.response.status_code in RETRYABLE_STATUSES
                if not retryable or attempt == MAX_ATTEMPTS:
                    raise
                delay = delays.next_interval()
                logger.warning("Parallel %s attempt %d failed (%s); retrying in %.1fs", path, attempt, error, delay)
                await asyncio.sleep(delay)
