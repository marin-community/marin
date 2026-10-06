# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Web search and page extraction tools for the agent loop, over Parallel's REST API.

``web_search`` posts to ``/v1/search`` and ``web_fetch`` to ``/v1/extract`` through the caller's
``httpx.AsyncClient``, so concurrent agents share one connection pool; the model sees Parallel's
JSON response body unchanged. ``web_fetch`` returns extracted page content that may be Parallel's
cached copy of the page; its description says so and leaves it to the agent to make a direct network
call from the sandbox shell when it decides it needs current data. A 400 or 422 means the model's arguments were rejected, so the body
goes back to the model as an error result. A 408, 429, 5xx or transport error is retried with
exponential backoff, up to ``MAX_ATTEMPTS`` requests; after that, and on every other non-2xx
status, the error raises for the caller to classify.
"""

import asyncio
import logging
from collections.abc import Mapping

import httpx
from rigging.timing import ExponentialBackoff

from taskforge.llm.agent import AgentTool

logger = logging.getLogger(__name__)

PARALLEL_API_URL = "https://api.parallel.ai/v1"
REQUEST_TIMEOUT = 180.0
MAX_ATTEMPTS = 6
MODEL_ERROR_STATUSES = frozenset({400, 422})
RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504})


def web_tools(
    http: httpx.AsyncClient,
    api_key: str,
    *,
    base_url: str = PARALLEL_API_URL,
    backoff: ExponentialBackoff | None = None,
) -> tuple[AgentTool, AgentTool]:
    """Return the ``web_search`` and ``web_fetch`` tools authenticated with ``api_key``.

    Args:
        http: Client the tools send through; the caller owns its lifetime.
        api_key: Parallel API key, sent as ``x-api-key``.
        base_url: Parallel API root ending in ``/v1``.
        backoff: Delay schedule between retried requests.
    """
    schedule = backoff or ExponentialBackoff(initial=1.0, maximum=30.0, factor=2.0)

    async def post(path: str, body: Mapping[str, object]) -> str:
        delays = schedule.copy()
        attempt = 0
        while True:
            attempt += 1
            try:
                response = await http.post(
                    f"{base_url}/{path}", json=body, headers={"x-api-key": api_key}, timeout=REQUEST_TIMEOUT
                )
                if response.status_code in MODEL_ERROR_STATUSES:
                    return f"error: Parallel rejected the request (HTTP {response.status_code}): {response.text}"
                response.raise_for_status()
                return response.text
            except (httpx.HTTPStatusError, httpx.TransportError) as error:
                retryable = isinstance(error, httpx.TransportError) or error.response.status_code in RETRYABLE_STATUSES
                if not retryable or attempt == MAX_ATTEMPTS:
                    raise
                delay = delays.next_interval()
                logger.warning("Parallel %s attempt %d failed (%s); retrying in %.1fs", path, attempt, error, delay)
                await asyncio.sleep(delay)

    async def search(arguments: Mapping[str, object]) -> str:
        return await post("search", {"objective": arguments["objective"], "search_queries": arguments["search_queries"]})

    async def fetch(arguments: Mapping[str, object]) -> str:
        body = {"urls": arguments["urls"]}
        if "objective" in arguments:
            body["objective"] = arguments["objective"]
        return await post("extract", body)

    search_tool = AgentTool(
        name="web_search",
        description="Search the web. Returns ranked URLs with excerpts relevant to the objective.",
        parameters={
            "type": "object",
            "properties": {
                "objective": {"type": "string", "description": "What you are trying to find, in one sentence."},
                "search_queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Two or three keyword queries.",
                },
            },
            "required": ["objective", "search_queries"],
            "additionalProperties": False,
        },
        handler=search,
    )
    fetch_tool = AgentTool(
        name="web_fetch",
        description=(
            "Fetch extracted content from up to 20 URLs, focused on an optional objective. The content may "
            "be a cached copy of the page, which is fine for most reference lookups. When you need current "
            "data from a fast-moving source, such as a PyPI release page, a direct network call from the "
            "shell (for example curl) is the better path."
        ),
        parameters={
            "type": "object",
            "properties": {
                "urls": {"type": "array", "items": {"type": "string"}, "maxItems": 20},
                "objective": {"type": "string"},
            },
            "required": ["urls"],
            "additionalProperties": False,
        },
        handler=fetch,
    )
    return search_tool, fetch_tool
