# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run BrowseComp's original retriever inside the task's declared actor machine."""

import json
from dataclasses import dataclass

from pydantic import JsonValue
from shellbox.machine import Command, Machine
from taskcompendium.browsecomp import BROWSECOMP_MODULE, BrowseCompState, browsecomp_state, browsecomp_tools
from taskcompendium.models import FunctionDefinition, ProviderRequirement

from rolloutengine.harness.shell import ToolCallResult, ToolContext

BROWSECOMP_OUTPUT_LIMIT = 16 * 1024 * 1024


@dataclass(frozen=True)
class BrowseCompSession:
    machine: Machine
    state: BrowseCompState
    command_timeout: float | None

    async def call(self, name: str, arguments: dict[str, JsonValue]) -> ToolCallResult:
        parameter = "query" if name == "search" else "docid"
        allowed = {tool.name for tool in browsecomp_tools(self.state)}
        if name not in allowed or set(arguments) != {parameter} or not isinstance(arguments.get(parameter), str):
            return ToolCallResult(json.dumps({"error": "Invalid offline search tool or arguments"}))
        result = await self.machine.run(
            Command(
                argv=("python3", "-m", BROWSECOMP_MODULE),
                stdin=json.dumps(
                    {"name": name, "arguments": arguments, "state": self.state.model_dump(mode="json")}
                ).encode(),
                timeout=self.command_timeout,
                output_limit_bytes=BROWSECOMP_OUTPUT_LIMIT,
            )
        )
        if result.exit_code != 0 or result.stdout_truncated:
            raise RuntimeError(
                f"Offline BrowseComp retrieval failed ({result.reason}, exit={result.exit_code}, "
                f"truncated={result.stdout_truncated}): {result.stderr.decode(errors='replace')[-2000:]}"
            )
        # Invalid producer output is an execution error, never a fabricated empty search result.
        observation = json.loads(result.stdout)
        return ToolCallResult(json.dumps(observation, ensure_ascii=False))

    async def close(self) -> None:
        pass  # The enclosing attempt owns the actor machine.


@dataclass(frozen=True)
class BrowseCompProvider:
    command_timeout: float | None

    def tools(self, requirement: ProviderRequirement) -> tuple[FunctionDefinition, ...]:
        return browsecomp_tools(browsecomp_state(requirement))

    async def open(self, context: ToolContext) -> BrowseCompSession:
        if context.machine is None:
            raise ValueError("Offline BrowseComp requires its declared actor machine")
        return BrowseCompSession(context.machine, browsecomp_state(context.requirement), self.command_timeout)
