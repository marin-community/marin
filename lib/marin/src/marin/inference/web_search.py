# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Optional dashboard ``web_search`` tool backed by Parallel's search API (https://docs.parallel.ai).

The dashboard server holds the API key, so the browser never receives it.
"""

import logging
from dataclasses import dataclass, field

import httpx
from pydantic import BaseModel, ValidationError
from rigging.secrets import get_secret
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

PARALLEL_API_KEY_SECRET = "PARALLEL_API_KEY"
PARALLEL_API_URL = "https://api.parallel.ai/v1"
PARALLEL_REQUEST_TIMEOUT = 60.0


@dataclass(frozen=True)
class ParallelWebSearch:
    """Parallel credentials and API root used by the dashboard's ``web_search`` tool."""

    api_key: str = field(repr=False)
    base_url: str = PARALLEL_API_URL


class WebSearchArguments(BaseModel):
    """Arguments of one ``web_search`` call. Unknown fields are dropped, so callers cannot set other Parallel options."""

    objective: str
    search_queries: list[str]


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
    return Response(response.content, status_code=response.status_code, media_type="application/json")
