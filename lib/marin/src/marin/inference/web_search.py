# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Optional dashboard ``web_search`` tool backed by Parallel's search API (https://docs.parallel.ai).

The dashboard server holds the API key, so the browser never receives it. Parallel's balance endpoint
accepts only OAuth account tokens, not API keys, so availability comes from search responses instead:
HTTP 402 means the account has no credit and 401/403 that the key was rejected. The server probes once at
start-up, updates availability from every search, and re-probes while search is unavailable.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import StrEnum

import httpx
from pydantic import BaseModel, ValidationError
from rigging.secrets import get_secret
from starlette.datastructures import State
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

PARALLEL_API_KEY_SECRET = "PARALLEL_API_KEY"
PARALLEL_API_URL = "https://api.parallel.ai/v1"
PARALLEL_REQUEST_TIMEOUT = 60.0
# Interval between availability probes while web search is unavailable.
WEB_SEARCH_RECHECK_SECONDS = 600.0
# The smallest valid search, sent at start-up and while search is unavailable.
PROBE_SEARCH = {
    "objective": "Check that web search is available",
    "search_queries": ["web search availability"],
    "advanced_settings": {"max_results": 1},
}


class WebSearchAvailability(StrEnum):
    """Whether Parallel accepted this server's key on the last search."""

    AVAILABLE = "available"
    NO_CREDIT = "no_credit"
    KEY_REJECTED = "key_rejected"


@dataclass(frozen=True)
class ParallelWebSearch:
    """Parallel credentials and API root used by the dashboard's ``web_search`` tool."""

    api_key: str = field(repr=False)
    base_url: str = PARALLEL_API_URL


class WebSearchArguments(BaseModel):
    """Arguments of one ``web_search`` call. Unknown fields are dropped, so callers cannot set other Parallel options."""

    objective: str
    search_queries: list[str]


def availability_from_status(status_code: int) -> WebSearchAvailability | None:
    """Availability implied by a Parallel search response; None for statuses that say nothing about the key."""
    if 200 <= status_code < 300:
        return WebSearchAvailability.AVAILABLE
    if status_code == 402:
        return WebSearchAvailability.NO_CREDIT
    if status_code in (401, 403):
        return WebSearchAvailability.KEY_REJECTED
    return None


def _record_availability(state: State, availability: WebSearchAvailability | None) -> None:
    if availability is None or availability is state.web_search_availability:
        return
    logger.info("Parallel web search is now %s", availability.value)
    state.web_search_availability = availability


async def _probe_availability(search: ParallelWebSearch, client: httpx.AsyncClient) -> WebSearchAvailability | None:
    try:
        response = await client.post(
            f"{search.base_url}/search",
            json=PROBE_SEARCH,
            headers={"x-api-key": search.api_key},
            timeout=PARALLEL_REQUEST_TIMEOUT,
        )
    except httpx.HTTPError as exc:
        logger.warning("Parallel availability probe failed: %s", type(exc).__name__)
        return None
    return availability_from_status(response.status_code)


async def monitor_web_search_availability(
    state: State, search: ParallelWebSearch, client: httpx.AsyncClient, recheck_seconds: float
) -> None:
    """Probe Parallel now, then every ``recheck_seconds`` while ``state.web_search_availability`` is not available.

    While search is available, the searches users make keep the state current, so nothing is probed.
    """
    _record_availability(state, await _probe_availability(search, client))
    while True:
        await asyncio.sleep(recheck_seconds)
        if state.web_search_availability is not WebSearchAvailability.AVAILABLE:
            _record_availability(state, await _probe_availability(search, client))


def parallel_web_search_from_secret() -> ParallelWebSearch | None:
    """Return web search configured from the ``PARALLEL_API_KEY`` secret, or None when it is absent or empty."""
    api_key = get_secret(PARALLEL_API_KEY_SECRET)
    return ParallelWebSearch(api_key) if api_key else None


async def web_search_response(request: Request) -> Response:
    """Run one ``web_search`` call against Parallel ``/v1/search`` and return Parallel's response unchanged."""
    search: ParallelWebSearch | None = request.app.state.web_search
    if search is None:
        return JSONResponse({"error": "web search is not configured on this server"}, status_code=404)
    try:
        arguments = WebSearchArguments.model_validate_json(await request.body())
    except ValidationError as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)

    client: httpx.AsyncClient = request.app.state.web_search_client
    try:
        response = await client.post(
            f"{search.base_url}/search",
            json=arguments.model_dump(),
            headers={"x-api-key": search.api_key},
            timeout=PARALLEL_REQUEST_TIMEOUT,
        )
    except httpx.HTTPError as exc:
        logger.warning("Parallel search request failed: %s", type(exc).__name__)
        return JSONResponse({"error": f"Parallel search request failed: {type(exc).__name__}"}, status_code=502)
    if not response.is_success:
        logger.warning("Parallel search returned HTTP %d", response.status_code)
    _record_availability(request.app.state, availability_from_status(response.status_code))
    return Response(response.content, status_code=response.status_code, media_type="application/json")
