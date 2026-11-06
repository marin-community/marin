# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import override

from forge.effects.context import Context
from forge.harness.protocol import ToolResult
from forge.harness.tools.base import ValidatedTool
from forge.harness.web import WebFetch, WebSearch
from forge.resources.env import Env


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
    async def run(self, ctx: Context, args: SearchArgs, env: Env) -> ToolResult:
        return ToolResult(await ctx.emit(WebSearch(args.query)))


@dataclass(frozen=True, slots=True)
class WebFetchTool(ValidatedTool[FetchArgs]):
    name = "web_fetch"
    description = "Fetch the contents of a web page."
    args_type = FetchArgs

    @override
    async def run(self, ctx: Context, args: FetchArgs, env: Env) -> ToolResult:
        return ToolResult(await ctx.emit(WebFetch(args.url)))
