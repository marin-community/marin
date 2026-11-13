# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serve the ``fetch-v1`` tool from the pages a task declares, without network access."""

import json
from dataclasses import dataclass

from pydantic import JsonValue
from taskcompendium.fetch import FETCH_TOOL, NOT_FOUND_STATUS, FetchState, NetworkFailure
from taskcompendium.models import FunctionDefinition, ProviderRequirement

from rolloutengine.harness.shell import ToolCallResult, ToolContext


@dataclass
class FetchSession:
    state: FetchState

    async def call(self, name: str, arguments: dict[str, JsonValue]) -> ToolCallResult:
        url = arguments.get("url")
        if name != FETCH_TOOL.name or set(arguments) != {"url"} or not isinstance(url, str):
            return ToolCallResult(json.dumps({"error": f"{FETCH_TOOL.name} requires one string url"}))
        page = self.state.pages.get(url)
        if page is None:
            return ToolCallResult(json.dumps({"url": url, "status": NOT_FOUND_STATUS, "error": "Not found"}))
        if isinstance(page, NetworkFailure):
            return ToolCallResult(json.dumps({"url": url, "error": page.error}))
        return ToolCallResult(json.dumps({"url": url, "status": page.status, "body": page.body}))

    async def close(self) -> None:
        pass


class FetchProvider:
    """Serve ``fetch_url`` for ``fetch-v1`` requirements from their declared pages."""

    def tools(self, requirement: ProviderRequirement) -> tuple[FunctionDefinition, ...]:
        return (FETCH_TOOL,)

    async def open(self, context: ToolContext) -> FetchSession:
        return FetchSession(FetchState.model_validate(context.requirement.initial_state))
