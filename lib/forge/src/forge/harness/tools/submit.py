# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import override

from forge.effects.context import Context
from forge.harness.protocol import SubmitRequest, ToolResult
from forge.harness.tools.base import ValidatedTool
from forge.lowering import check_candidates
from forge.model.serialization import CONTROLS, TASK_SPEC
from forge.model.workpiece import Submission
from forge.resources.env import ReadFile

type Gate = Callable[[Submission], tuple[str, ...]]


async def read_submission(
    ctx: Context, task_path: str, controls_path: str, notes: str, gate: Gate
) -> Submission | ToolResult:
    try:
        task = TASK_SPEC.validate_json(await ctx.emit(ReadFile(PurePosixPath(task_path))))
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
