# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import override

from forge.effects.context import Context
from forge.harness.protocol import ReturnRequest, ToolResult
from forge.harness.tools.base import ValidatedTool
from forge.harness.tools.submit import Gate, read_submission
from forge.model.workpiece import Submission
from forge.resources.env import Env


@dataclass(frozen=True, slots=True)
class ValidateRequest:
    submission: Submission


@dataclass(frozen=True, slots=True)
class ValidateArgs:
    pass


@dataclass(frozen=True, slots=True)
class ValidateTool(ValidatedTool[ValidateArgs]):
    gate: Gate
    task_path: str = "/workspace/task.json"
    controls_path: str = "/workspace/controls.json"
    name = "validate"
    description = "Validate the current task and controls, then continue with the evidence."
    args_type = ValidateArgs

    @override
    async def run(self, ctx: Context, args: ValidateArgs, env: Env) -> ToolResult | ReturnRequest[ValidateRequest]:
        result = await read_submission(ctx, self.task_path, self.controls_path, "", self.gate)
        return result if isinstance(result, ToolResult) else ReturnRequest("validate", ValidateRequest(result))
