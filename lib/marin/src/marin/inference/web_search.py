# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Optional dashboard web search through Parallel's search API (https://docs.parallel.ai).

The dashboard server holds the Parallel API key; the browser never receives it. ``/web-search``
forwards only ``objective`` and ``search_queries`` to Parallel ``/v1/search``, so a caller cannot
select other Parallel options, and returns Parallel's JSON response body unchanged. A 400 or 422
from Parallel means the model's arguments were rejected, so the error body goes back as a 422.
Every other failure returns a 502 without retrying.
"""

import json
import logging
from dataclasses import dataclass, field

import httpx
from rigging.secrets import get_secret
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

PARALLEL_API_KEY_SECRET = "PARALLEL_API_KEY"
PARALLEL_API_URL = "https://api.parallel.ai/v1"
PARALLEL_REQUEST_TIMEOUT = 60.0
_MODEL_ERROR_STATUSES = frozenset({400, 422})
_MAX_ERROR_LENGTH = 4_000


@dataclass(frozen=True)
class ParallelWebSearch:
    """Parallel credentials and API root used by the dashboard's ``web_search`` tool."""

    api_key: str = field(repr=False)
    base_url: str = PARALLEL_API_URL


def parallel_web_search_from_secret() -> ParallelWebSearch | None:
    """Return web search configured from the ``PARALLEL_API_KEY`` secret, or None when it is absent or empty.

    Iris delivers job secrets as files; ``get_secret`` reads the file and falls back to the environment.
    """
    api_key = get_secret(PARALLEL_API_KEY_SECRET)
    return ParallelWebSearch(api_key) if api_key else None


async def web_search_response(request: Request) -> Response:
    """Run one ``web_search`` tool call against Parallel."""
    search: ParallelWebSearch | None = request.app.state.web_search
    if search is None:
        return JSONResponse({"error": "web search is not configured on this server"}, status_code=404)
    try:
        payload = await request.json()
    except json.JSONDecodeError:
        return JSONResponse({"error": "web search request must be JSON"}, status_code=400)
    if not isinstance(payload, dict):
        return JSONResponse({"error": "web search request must be a JSON object"}, status_code=400)

    body = {key: payload[key] for key in ("objective", "search_queries") if key in payload}
    client: httpx.AsyncClient = request.app.state.web_search_client
    try:
        response = await client.post(
            f"{search.base_url}/search",
            json=body,
            headers={"x-api-key": search.api_key},
            timeout=PARALLEL_REQUEST_TIMEOUT,
        )
    except httpx.HTTPError as exc:
        logger.warning("Parallel search request failed: %s", type(exc).__name__)
        return JSONResponse({"error": f"Parallel search request failed: {type(exc).__name__}"}, status_code=502)
    if response.status_code in _MODEL_ERROR_STATUSES:
        return JSONResponse(
            {
                "error": f"Parallel rejected the search arguments (HTTP {response.status_code})",
                "details": response.text[:_MAX_ERROR_LENGTH],
            },
            status_code=422,
        )
    if not response.is_success:
        logger.warning("Parallel search returned HTTP %d", response.status_code)
        return JSONResponse({"error": f"Parallel search returned HTTP {response.status_code}"}, status_code=502)
    return Response(response.content, media_type="application/json")
