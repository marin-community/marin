# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import override

from taskforge.effects.context import Context
from taskforge.harness.protocol import ReturnRequest, ToolResult
from taskforge.harness.tools.base import ValidatedTool
from taskforge.harness.tools.submit import Gate, read_submission
from taskforge.model.workpiece import Submission


@dataclass(frozen=True, slots=True)
class ValidateRequest:
    submission: Submission


@dataclass(frozen=True, slots=True)
class ValidateArgs:
    pass


@dataclass(frozen=True, slots=True)
class ValidateTool(ValidatedTool[ValidateArgs]):
    gate: Gate
    task_path: str
    controls_path: str
    name = "validate"
    description = "Validate the current task and controls, then continue with the evidence."
    args_type = ValidateArgs

    @override
    async def run(self, ctx: Context, args: ValidateArgs) -> ToolResult | ReturnRequest[ValidateRequest]:
        result = await read_submission(ctx, self.task_path, self.controls_path, "", self.gate)
        return result if isinstance(result, ToolResult) else ReturnRequest("validate", ValidateRequest(result))
