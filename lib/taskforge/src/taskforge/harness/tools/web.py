# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import override

from taskforge.effects.context import Context
from taskforge.harness.protocol import ToolResult
from taskforge.harness.tools.base import ValidatedTool
from taskforge.resources.web import WebFetch, WebSearch


@dataclass(frozen=True, slots=True)
class SearchArgs:
    query: str


@dataclass(frozen=True, slots=True)
class FetchArgs:
    url: str


@dataclass(frozen=True, slots=True)
class WebSearchTool(ValidatedTool[SearchArgs]):
    name = "web_search"
    description = "Search the web for a query."
    args_type = SearchArgs

    @override
    async def run(self, ctx: Context, args: SearchArgs) -> ToolResult:
        return ToolResult(await ctx.emit(WebSearch(args.query)))


@dataclass(frozen=True, slots=True)
class WebFetchTool(ValidatedTool[FetchArgs]):
    name = "web_fetch"
    description = "Fetch the contents of a web page."
    args_type = FetchArgs

    @override
    async def run(self, ctx: Context, args: FetchArgs) -> ToolResult:
        return ToolResult(await ctx.emit(WebFetch(args.url)))
