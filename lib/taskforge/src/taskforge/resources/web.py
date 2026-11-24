# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replayable web search and extraction."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, override

import httpx

from taskforge.effects.events import Event, Fingerprint, ReplayPolicy, digest
from taskforge.effects.issues import Hold, Transient
from taskforge.resources.execution import Notify, retry_effect


@dataclass(frozen=True, slots=True)
class WebSearch(Event[str]):
    kind = "web.search"
    query: str


@dataclass(frozen=True, slots=True)
class WebFetch(Event[str]):
    kind = "web.fetch"
    url: str


class WebClient(Protocol):
    def fingerprint(self) -> Fingerprint: ...
    async def search(self, query: str) -> str: ...
    async def fetch(self, url: str) -> str: ...


@dataclass(frozen=True, slots=True)
class ParallelWeb(WebClient):
    """The caller owns the httpx client and its API-key header and timeout."""

    client: httpx.AsyncClient

    @override
    def fingerprint(self) -> Fingerprint:
        return Fingerprint("web", digest(str(self.client.base_url)), ReplayPolicy.CACHE)

    async def _post(self, path: str, body: Mapping[str, object]) -> str:
        try:
            response = await self.client.post(path, json=dict(body))
            response.raise_for_status()
        except httpx.TransportError as error:
            raise Transient("Web transport unavailable") from error
        except httpx.HTTPStatusError as error:
            if error.response.status_code == 429 or error.response.status_code >= 500:
                raise Transient(f"Web status {error.response.status_code}") from error
            if error.response.status_code in (401, 403):
                raise Hold(f"Web credentials rejected: {error.response.status_code}") from error
            return f"Web request rejected ({error.response.status_code}): {error.response.text}"
        return response.text

    @override
    async def search(self, query: str) -> str:
        return await self._post("/search", {"objective": query, "search_queries": [query]})

    @override
    async def fetch(self, url: str) -> str:
        return await self._post("/extract", {"urls": [url]})


async def search(event: WebSearch, client: WebClient, *, notify: Notify) -> str:
    return await retry_effect(event, lambda: client.search(event.query), notify)


async def fetch(event: WebFetch, client: WebClient, *, notify: Notify) -> str:
    return await retry_effect(event, lambda: client.fetch(event.url), notify)
