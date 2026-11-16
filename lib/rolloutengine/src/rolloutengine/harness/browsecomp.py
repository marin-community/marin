# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run offline retrieval inside the task's declared actor machine."""

import json
import shlex
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import BaseModel, JsonValue
from shellbox.machine import Command, Machine, Result
from taskcompendium.browsecomp import BROWSECOMP_MODULE, browsecomp_state, browsecomp_tools
from taskcompendium.models import FunctionDefinition, ProviderRequirement
from taskcompendium.openresearcher import OPENRESEARCHER_MODULE, openresearcher_state

from rolloutengine.harness.shell import ToolCallResult, ToolContext

BROWSECOMP_OUTPUT_LIMIT = 16 * 1024 * 1024
INVALID_ARGUMENTS = json.dumps({"error": "Invalid offline search tool or arguments"})


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
    module: str
    command_timeout: float | None

    def request_command(self, name: str, arguments: dict[str, JsonValue]) -> Command | None:
        parameter = "query" if name == "search" else "docid"
        allowed = {tool.name for tool in self.tools}
        if name not in allowed or set(arguments) != {parameter} or not isinstance(arguments.get(parameter), str):
            return None
        return Command(
            argv=("python3", "-m", self.module),
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
class OpenResearcherSession:
    captured: OfflineSearchSession

    async def call(self, name: str, arguments: dict[str, JsonValue]) -> ToolCallResult:
        if name != "get_document":
            return await self.captured.call(name, arguments)
        command = self.captured.request_command(name, arguments)
        if command is None:
            return ToolCallResult(INVALID_ARGUMENTS)
        remote = f"/tmp/openresearcher-document-{uuid.uuid4().hex}.json"
        producer = replace(command, argv=("sh", "-c", f"{shlex.join(command.argv)} > {shlex.quote(remote)}"))
        machine = self.captured.machine
        try:
            retrieval_output(await machine.run(producer))
            with TemporaryDirectory(prefix="openresearcher-document-") as directory:
                local = Path(directory) / "document.json"
                await machine.download(remote, local)
                observation = json.loads(local.read_bytes())
        finally:
            retrieval_output(await machine.run(Command(("rm", "-f", remote), timeout=command.timeout)))
        return ToolCallResult(json.dumps(observation, ensure_ascii=False))

    async def close(self) -> None:
        await self.captured.close()


@dataclass(frozen=True)
class BrowseCompProvider:
    command_timeout: float | None

    def tools(self, requirement: ProviderRequirement) -> tuple[FunctionDefinition, ...]:
        return browsecomp_tools(browsecomp_state(requirement))

    async def open(self, context: ToolContext) -> OfflineSearchSession:
        if context.machine is None:
            raise ValueError("Offline BrowseComp requires its declared actor machine")
        state = browsecomp_state(context.requirement)
        return OfflineSearchSession(
            context.machine, state, browsecomp_tools(state), BROWSECOMP_MODULE, self.command_timeout
        )


@dataclass(frozen=True)
class OpenResearcherProvider:
    command_timeout: float | None

    def tools(self, requirement: ProviderRequirement) -> tuple[FunctionDefinition, ...]:
        return browsecomp_tools(openresearcher_state(requirement))

    async def open(self, context: ToolContext) -> OpenResearcherSession:
        if context.machine is None:
            raise ValueError("Offline OpenResearcher requires its declared actor machine")
        state = openresearcher_state(context.requirement)
        return OpenResearcherSession(
            OfflineSearchSession(
                context.machine, state, browsecomp_tools(state), OPENRESEARCHER_MODULE, self.command_timeout
            )
        )
