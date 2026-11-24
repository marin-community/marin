# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import PurePosixPath
from typing import override

from taskcompendium.models import TaskSpec

from taskforge.effects.context import Context
from taskforge.effects.events import HANDLED_FAILURE
from taskforge.harness.protocol import SubmitRequest, ToolResult
from taskforge.harness.tools.base import ValidatedTool
from taskforge.lowering import check_candidates
from taskforge.model.controls import CONTROLS
from taskforge.model.review import Decline, ReviewFinding, Text
from taskforge.model.workpiece import Submission
from taskforge.resources.env import ReadFile

type Gate = Callable[[Submission], tuple[str, ...]]


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
        try:
            task = TaskSpec.model_validate_json(await ctx.emit(ReadFile(PurePosixPath(self.task_path))))
            controls = CONTROLS.validate_json(await ctx.emit(ReadFile(PurePosixPath(self.controls_path))))
            check_candidates(task, tuple(control.candidate for control in controls))
        except (ValueError, HANDLED_FAILURE) as error:
            return ToolResult(f"Invalid draft: {error}")
        submission = Submission(task, controls, args.notes)
        findings = self.gate(submission)
        return ToolResult("\n".join(findings)) if findings else SubmitRequest(submission)


@dataclass(frozen=True, slots=True)
class DeclineArgs:
    finding: int
    reason: Text


@dataclass(frozen=True, slots=True)
class RevisedArgs:
    notes: str = ""
    declined: tuple[DeclineArgs, ...] = ()


@dataclass(frozen=True, slots=True)
class RevisedSubmitTool(ValidatedTool[RevisedArgs]):
    """``submit`` in a revision after a solve-review: it also records the findings the author declines, and why."""

    submit: SubmitTool
    findings: tuple[ReviewFinding, ...]
    name = "submit"
    description = (
        f"{SubmitTool.description} `declined` lists each solve-review finding you decline by its number, with a reason."
    )
    args_type = RevisedArgs

    @override
    async def run(self, ctx: Context, args: RevisedArgs) -> ToolResult | SubmitRequest:
        numbers = [decline.finding for decline in args.declined]
        if any(not 0 <= number < len(self.findings) for number in numbers) or len(set(numbers)) != len(numbers):
            return ToolResult(f"declined names each finding at most once, by its number 0..{len(self.findings) - 1}")
        outcome = await self.submit.run(ctx, SubmitArgs(args.notes))
        if not isinstance(outcome, SubmitRequest):
            return outcome
        declined = tuple(Decline(self.findings[decline.finding], decline.reason) for decline in args.declined)
        return SubmitRequest(replace(outcome.submission, declined=declined))
