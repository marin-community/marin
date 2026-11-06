# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import dataclass
from typing import override

from forge.effects.context import Context
from forge.harness.protocol import ToolResult
from forge.harness.tools.base import ValidatedTool
from forge.resources.env import Env, RunCommand


@dataclass(frozen=True, slots=True)
class ShellArgs:
    command: str
    timeout: float = 30.0


@dataclass(frozen=True, slots=True)
class ShellTool(ValidatedTool[ShellArgs]):
    output_chars: int
    name = "shell"
    description = "Run a shell command with a timeout in seconds."
    args_type = ShellArgs

    @override
    async def run(self, ctx: Context, args: ShellArgs, env: Env) -> ToolResult:
        result = await ctx.emit(RunCommand(args.command, args.timeout))
        return ToolResult(
            json.dumps(
                {
                    "exit_code": result.exit_code,
                    "stdout": result.stdout[: self.output_chars],
                    "stderr": result.stderr[: self.output_chars],
                    "truncated": (
                        result.truncated
                        or len(result.stdout) > self.output_chars
                        or len(result.stderr) > self.output_chars
                    ),
                }
            )
        )
