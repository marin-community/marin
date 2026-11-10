# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import override

from taskcompendium.models import TaskSpec

from forge.effects.context import Context
from forge.harness.protocol import SubmitRequest, ToolResult
from forge.harness.tools.base import ValidatedTool
from forge.lowering import check_candidates
from forge.model.controls import CONTROLS
from forge.model.proposal import TASK_PROPOSAL, Verification
from forge.model.workpiece import Submission
from forge.resources.env import ReadFile

type Gate = Callable[[Submission, Verification], tuple[str, ...]]
PROPOSAL_FILE = "proposal.json"
"""The proposal the author stage writes beside the controls; validation rechecks the workpiece's own."""


async def read_submission(
    ctx: Context, task_path: str, controls_path: str, notes: str, gate: Gate
) -> Submission | ToolResult:
    try:
        task = TaskSpec.model_validate_json(await ctx.emit(ReadFile(PurePosixPath(task_path))))
        controls = CONTROLS.validate_json(await ctx.emit(ReadFile(PurePosixPath(controls_path))))
        check_candidates(task, tuple(control.candidate for control in controls))
        proposal = TASK_PROPOSAL.validate_json(
            await ctx.emit(ReadFile(PurePosixPath(controls_path).with_name(PROPOSAL_FILE)))
        )
    except ValueError as error:
        return ToolResult(f"Invalid draft: {error}")
    submission = Submission(task, controls, notes)
    findings = gate(submission, proposal.header.verification)
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
