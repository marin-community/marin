# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serve ``shell:v1`` on the attempt's existing actor machine."""

from dataclasses import dataclass

from pydantic import JsonValue
from shellbox.machine import DEFAULT_MACHINE_OUTPUT_LIMIT_BYTES, Machine
from taskcompendium.models import FunctionCall, FunctionDefinition, ProviderRequirement
from taskcompendium.repository_shell import SHELL_TOOL, RepositoryShellState, repository_shell_state
from taskcompendium.runtime.shell import BoundShellTool, run_shell_call

from rolloutengine.harness.shell import ToolCallResult, ToolContext


@dataclass(frozen=True)
class RepositoryShellSession:
    machine: Machine
    state: RepositoryShellState
    command_timeout: float | None
    output_limit_bytes: int

    async def call(self, name: str, arguments: dict[str, JsonValue]) -> ToolCallResult:
        observation = await run_shell_call(
            self.machine,
            FunctionCall(name=name, arguments=arguments),
            BoundShellTool(SHELL_TOOL, "command"),
            timeout=self.command_timeout,
            output_limit_bytes=self.output_limit_bytes,
            cwd=self.state.workspace,
        )
        return ToolCallResult(observation)

    async def close(self) -> None:
        # The enclosing attempt owns the machine and its bounded cleanup.
        pass


@dataclass(frozen=True)
class RepositoryShellProvider:
    command_timeout: float | None
    output_limit_bytes: int = DEFAULT_MACHINE_OUTPUT_LIMIT_BYTES

    def tools(self, requirement: ProviderRequirement) -> tuple[FunctionDefinition, ...]:
        repository_shell_state(requirement)
        return (SHELL_TOOL,)

    async def open(self, context: ToolContext) -> RepositoryShellSession:
        if context.machine is None:
            raise ValueError("shell:v1 requires the declared actor machine")
        return RepositoryShellSession(
            context.machine,
            repository_shell_state(context.requirement),
            self.command_timeout,
            self.output_limit_bytes,
        )
