# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import override

from shellbox.machine import MachineFactory
from taskcompendium.models import TaskSpec
from taskcompendium.runtime.resources import inline_resource

from taskforge.effects.context import Context
from taskforge.effects.issues import Hold, OutputBudgetExhausted
from taskforge.harness.protocol import ToolResult
from taskforge.harness.tools.base import FILE_ERRORS, ValidatedTool
from taskforge.lowering import InvalidTaskSpec, LoweringSettings, check_candidates, lower_with_settings
from taskforge.model.content_hash import pretty_json
from taskforge.model.controls import Candidate
from taskforge.model.verdict import grader_checks
from taskforge.resources.env import ReadFile
from taskforge.resources.grading import GradeCandidate, candidate_root


@dataclass(frozen=True, slots=True)
class CandidateArgs:
    reply: str
    files: tuple[str, ...] = ()


def resource_path(path: str, root: PurePosixPath) -> str:
    """``path`` relative to ``root``, the directory grading stages candidate files under."""
    absolute = PurePosixPath(path)
    if not absolute.is_absolute():
        return path
    if not absolute.is_relative_to(root):
        raise ValueError(f"candidate file {path} is outside the workspace {root}; pass a path relative to {root}")
    return str(absolute.relative_to(root))


async def read_candidate(ctx: Context, args: CandidateArgs, task: TaskSpec) -> Candidate:
    root = candidate_root(task)
    files = tuple(
        [
            inline_resource(resource_path(path, root), await ctx.emit(ReadFile(PurePosixPath(path))))
            for path in args.files
        ]
    )
    return Candidate(args.reply, files)


@dataclass(frozen=True, slots=True)
class GradeTool(ValidatedTool[CandidateArgs]):
    task_path: str
    factories: Mapping[str, MachineFactory]
    settings: LoweringSettings
    name = "grade"
    description = "Grade a candidate reply and workspace files against the current task draft."
    args_type = CandidateArgs

    def __post_init__(self) -> None:
        object.__setattr__(self, "factories", MappingProxyType(dict(self.factories)))

    @override
    async def run(self, ctx: Context, args: CandidateArgs) -> ToolResult:
        try:
            task = TaskSpec.model_validate_json(await ctx.emit(ReadFile(PurePosixPath(self.task_path))))
            candidate = await read_candidate(ctx, args, task)
            check_candidates(task, (candidate,))
        except (ValueError, *FILE_ERRORS) as error:
            return ToolResult(f"Cannot grade draft: {error}")
        try:
            spec = lower_with_settings(task, self.settings, self.factories)
        except InvalidTaskSpec as error:
            return ToolResult(f"Cannot grade draft: {error}")
        except (ValueError, NotImplementedError) as error:
            # A host that cannot run a valid draft is not the author's defect; submit's gate holds the same way.
            raise Hold(f"Lowering failed: {error}") from error
        try:
            result = await ctx.emit(GradeCandidate(spec, candidate))
        except OutputBudgetExhausted as error:
            # The grading machine's budget, not the caller's session: raised, it would read as the caller's own
            # exhausted search. The caller may grade another candidate; reproducing this one stays inconclusive.
            return ToolResult(f"Grading this candidate failed and produced no grade: {error} on the grading machine")
        try:
            checks = grader_checks(result)
        except ValueError as error:
            return ToolResult(
                "Grader checks must be a list of {name: string, passed: boolean, weight: number}; "
                f"fix the grader output: {error}"
            )
        return ToolResult(pretty_json({"reward": result.reward, "status": result.status, "grader_checks": checks}))
