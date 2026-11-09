# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import override

from taskcompendium.models import TaskSpec

from taskforge.effects.context import Context
from taskforge.harness.protocol import SubmitRequest, ToolResult
from taskforge.harness.tools.base import ValidatedTool
from taskforge.lowering import check_candidates
from taskforge.model.controls import CONTROLS
from taskforge.model.workpiece import Submission
from taskforge.resources.env import ReadFile

type Gate = Callable[[Submission], tuple[str, ...]]


async def read_submission(
    ctx: Context, task_path: str, controls_path: str, notes: str, gate: Gate
) -> Submission | ToolResult:
    try:
        task = TaskSpec.model_validate_json(await ctx.emit(ReadFile(PurePosixPath(task_path))))
        controls = CONTROLS.validate_json(await ctx.emit(ReadFile(PurePosixPath(controls_path))))
        check_candidates(task, tuple(control.candidate for control in controls))
    except ValueError as error:
        return ToolResult(f"Invalid draft: {error}")
    submission = Submission(task, controls, notes)
    findings = gate(submission)
    return ToolResult("\n".join(findings)) if findings else submission


@dataclass(frozen=True, slots=True)
class SubmitArgs:
    notes: str = ""


@dataclass(frozen=True, slots=True)
class SubmitTool(ValidatedTool[SubmitArgs]):
    gate: Gate
    task_path: str
    controls_path: str
    name = "submit"
    description = "Submit the task JSON and controls JSON after validation."
    args_type = SubmitArgs

    @override
    async def run(self, ctx: Context, args: SubmitArgs) -> ToolResult | SubmitRequest:
        result = await read_submission(ctx, self.task_path, self.controls_path, args.notes, self.gate)
        return result if isinstance(result, ToolResult) else SubmitRequest(result)
