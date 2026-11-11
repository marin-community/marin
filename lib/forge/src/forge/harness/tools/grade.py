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

from forge.effects.context import Context
from forge.harness.protocol import ToolResult
from forge.harness.tools.base import ValidatedTool
from forge.lowering import LoweringSettings, check_candidates, lower_with_settings
from forge.model.content_hash import pretty_json
from forge.model.controls import Candidate
from forge.model.verdict import grader_checks
from forge.resources.env import ReadFile
from forge.resources.grading import GradeCandidate


@dataclass(frozen=True, slots=True)
class CandidateArgs:
    reply: str
    files: tuple[str, ...] = ()


async def read_candidate(ctx: Context, args: CandidateArgs) -> Candidate:
    files = tuple(
        [inline_resource(path.lstrip("/"), await ctx.emit(ReadFile(PurePosixPath(path)))) for path in args.files]
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
            candidate = await read_candidate(ctx, args)
            check_candidates(task, (candidate,))
            spec = lower_with_settings(task, self.settings, self.factories)
        except (ValueError, NotImplementedError) as error:
            return ToolResult(f"Cannot grade draft: {error}")
        result = await ctx.emit(GradeCandidate(spec, candidate))
        try:
            checks = grader_checks(result)
        except ValueError as error:
            return ToolResult(
                "Grader checks must be a list of {name: string, passed: boolean, weight: number}; "
                f"fix the grader output: {error}"
            )
        output: dict[str, object] = {"reward": result.reward, "status": result.status, "grader_checks": checks}
        if read := (result.detail or {}).get("decision_model"):
            output["decision_model"] = read
        return ToolResult(pretty_json(output))
