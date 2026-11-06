# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import override

from forge.effects.context import Context
from forge.harness.protocol import FinishRequest
from forge.harness.tools.base import ValidatedTool
from forge.resources.env import Env


@dataclass(frozen=True, slots=True)
class FinishArgs:
    reason: str


@dataclass(frozen=True, slots=True)
class FinishTool(ValidatedTool[FinishArgs]):
    name = "finish"
    description = "Stop this session without submitting a task."
    args_type = FinishArgs

    @override
    async def run(self, ctx: Context, args: FinishArgs, env: Env) -> FinishRequest:
        return FinishRequest(args.reason)
