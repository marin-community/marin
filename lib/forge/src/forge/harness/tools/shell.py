# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import math
from dataclasses import dataclass
from typing import override

from forge.effects.context import Context
from forge.harness.protocol import ToolResult
from forge.harness.tools.base import ValidatedTool
from forge.resources.env import RunCommand


@dataclass(frozen=True, slots=True)
class ShellArgs:
    command: str
    timeout: float | None = None


@dataclass(frozen=True, slots=True)
class ShellTool(ValidatedTool[ShellArgs]):
    default_timeout: float
    max_timeout: float
    name = "shell"
    description = "Run a shell command with a timeout in seconds."
    args_type = ShellArgs

    @override
    async def run(self, ctx: Context, args: ShellArgs) -> ToolResult:
        timeout = self.default_timeout if args.timeout is None else args.timeout
        if not math.isfinite(timeout) or timeout <= 0:
            return ToolResult("timeout must be a finite positive number")
        result = await ctx.emit(RunCommand(args.command, min(timeout, self.max_timeout)))
        return ToolResult(
            json.dumps(
                {
                    "exit_code": result.exit_code,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "truncated": result.truncated,
                }
            )
        )
