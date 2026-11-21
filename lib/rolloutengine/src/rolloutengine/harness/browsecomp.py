# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run offline retrieval inside the task's declared actor machine."""

import json
from collections.abc import Collection
from dataclasses import dataclass

from pydantic import BaseModel, JsonValue
from shellbox.machine import Command, Machine, Result
from taskcompendium.browsecomp import BROWSECOMP_MODULE, browsecomp_state, browsecomp_tools
from taskcompendium.models import FunctionDefinition, ProviderRequirement

from rolloutengine.harness.shell import ToolCallResult, ToolContext

BROWSECOMP_OUTPUT_LIMIT = 16 * 1024 * 1024
INVALID_ARGUMENTS = json.dumps({"error": "Invalid offline search tool or arguments"})


def valid_retrieval_call(name: str, arguments: dict[str, JsonValue], tools: Collection[str]) -> bool:
    parameter = "query" if name == "search" else "docid"
    return name in tools and set(arguments) == {parameter} and isinstance(arguments.get(parameter), str)


def retrieval_output(result: Result) -> bytes:
    if result.exit_code != 0 or result.stdout_truncated:
        raise RuntimeError(
            f"Offline retrieval failed ({result.reason}, exit={result.exit_code}, "
            f"truncated={result.stdout_truncated}): {result.stderr.decode(errors='replace')[-2000:]}"
        )
    return result.stdout


@dataclass(frozen=True)
class OfflineSearchSession:
    machine: Machine
    state: BaseModel
    tools: tuple[FunctionDefinition, ...]
    command_timeout: float | None

    def request_command(self, name: str, arguments: dict[str, JsonValue]) -> Command | None:
        """Build a supported call, or return None for an unknown tool or invalid arguments."""
        if not valid_retrieval_call(name, arguments, {tool.name for tool in self.tools}):
            return None
        return Command(
            argv=("python3", "-m", BROWSECOMP_MODULE),
            stdin=json.dumps(
                {"name": name, "arguments": arguments, "state": self.state.model_dump(mode="json")}
            ).encode(),
            timeout=self.command_timeout,
            output_limit_bytes=BROWSECOMP_OUTPUT_LIMIT,
        )

    async def call(self, name: str, arguments: dict[str, JsonValue]) -> ToolCallResult:
        command = self.request_command(name, arguments)
        if command is None:
            return ToolCallResult(INVALID_ARGUMENTS)
        # Invalid producer output is an execution error, never a fabricated empty search result.
        observation = json.loads(retrieval_output(await self.machine.run(command)))
        return ToolCallResult(json.dumps(observation, ensure_ascii=False))

    async def close(self) -> None:
        pass  # The enclosing attempt owns the actor machine.


@dataclass(frozen=True)
class BrowseCompProvider:
    command_timeout: float | None

    def tools(self, requirement: ProviderRequirement) -> tuple[FunctionDefinition, ...]:
        return browsecomp_tools(browsecomp_state(requirement))

    async def open(self, context: ToolContext) -> OfflineSearchSession:
        if context.machine is None:
            raise ValueError("Offline BrowseComp requires its declared actor machine")
        state = browsecomp_state(context.requirement)
        return OfflineSearchSession(context.machine, state, browsecomp_tools(state), self.command_timeout)
