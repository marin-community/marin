# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Web search and page reading: the ``Web`` protocol and the agent loop's tools over it.

``Web`` is what a search provider's client implements (``taskforge_adapters.parallel.ParallelWeb``).
``web_tools(web)`` returns the ``web_search`` and ``web_fetch`` tools for ``llm.agent.run_agent``; each
returns the provider's JSON response. ``web_fetch`` returns extracted page content that may be the
provider's cached copy of the page; its description says so and leaves it to the agent to make a direct
network call from the sandbox shell when it decides it needs current data. A request the provider
rejects as malformed (``WebRequestRejected``) goes back to the model as an error result, so it can
correct its arguments; any other failure raises for the caller to classify.
"""

import json
from collections.abc import Awaitable, Mapping, Sequence
from typing import Any, Protocol, cast

from taskforge.llm.agent import AgentTool


class WebRequestRejected(Exception):
    """The provider rejected the request's arguments (HTTP ``status``); ``body`` is its reply."""

    def __init__(self, status: int, body: str):
        super().__init__(f"the web service rejected the request (HTTP {status}): {body}")
        self.status = status
        self.body = body


class Web(Protocol):
    """Web search and page extraction. Methods raise ``WebRequestRejected`` for arguments the provider refuses."""

    async def search(self, objective: str, queries: Sequence[str]) -> dict[str, Any]: ...

    async def extract(self, urls: Sequence[str], objective: str | None) -> dict[str, Any]: ...


def web_tools(web: Web) -> tuple[AgentTool, AgentTool]:
    """Return the ``web_search`` and ``web_fetch`` tools over ``web``.

    ``run_agent`` validates each call's arguments against the tool's parameter schema before the handler runs.
    """

    async def answered(request: Awaitable[dict[str, Any]]) -> str:
        try:
            return json.dumps(await request)
        except WebRequestRejected as error:
            return f"error: {error}"

    async def search(arguments: Mapping[str, object]) -> str:
        return await answered(
            web.search(cast(str, arguments["objective"]), cast(list[str], arguments["search_queries"]))
        )

    async def fetch(arguments: Mapping[str, object]) -> str:
        return await answered(
            web.extract(cast(list[str], arguments["urls"]), cast(str | None, arguments.get("objective")))
        )

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
