# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Web search and page extraction through the REST API of Parallel (https://docs.parallel.ai)."""

from collections.abc import Sequence
from types import TracebackType
from typing import Any, Self

import httpx

PARALLEL_API_URL = "https://api.parallel.ai/v1"


class ParallelWeb:
    """An async client for ``/search`` and ``/extract``. Use it as an async context manager.

    A response with an HTTP error status raises ``httpx.HTTPStatusError``.
    """

    def __init__(self, api_key: str, *, base_url: str = PARALLEL_API_URL, timeout: float = 120.0):
        self._client = httpx.AsyncClient(base_url=base_url, headers={"x-api-key": api_key}, timeout=timeout)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, kind: type[BaseException] | None, error: BaseException | None, traceback: TracebackType | None
    ) -> None:
        await self._client.aclose()

    async def search(self, objective: str, queries: Sequence[str]) -> dict[str, Any]:
        """Results with URLs, titles and excerpts for a natural-language objective and its search queries."""
        return await self._post("/search", {"objective": objective, "search_queries": list(queries)})

    async def extract(self, urls: Sequence[str], objective: str) -> dict[str, Any]:
        """The excerpts of each page that are relevant to the objective, and an error for each page that failed."""
        return await self._post("/extract", {"urls": list(urls), "objective": objective})

    async def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        response = await self._client.post(path, json=body)
        response.raise_for_status()
        return response.json()
