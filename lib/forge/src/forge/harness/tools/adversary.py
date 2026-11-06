# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import override

from forge.effects.context import Context
from forge.harness.protocol import ReturnRequest, ToolResult
from forge.harness.tools.base import ValidatedTool
from forge.harness.tools.grade import CandidateArgs, read_candidate
from forge.model.controls import Candidate
from forge.resources.env import Env


@dataclass(frozen=True, slots=True)
class AdversaryReport:
    findings: tuple[str, ...]
    candidates: tuple[Candidate, ...]


@dataclass(frozen=True, slots=True)
class ReportArgs:
    findings: tuple[str, ...]
    candidates: tuple[CandidateArgs, ...] = ()


@dataclass(frozen=True, slots=True)
class ReportTool(ValidatedTool[ReportArgs]):
    name = "report"
    description = "Finish the adversarial review with findings and candidate replies/files for validation."
    args_type = ReportArgs

    @override
    async def run(self, ctx: Context, args: ReportArgs, env: Env) -> ToolResult | ReturnRequest[AdversaryReport]:
        try:
            candidates = tuple([await read_candidate(ctx, candidate) for candidate in args.candidates])
        except (FileNotFoundError, KeyError) as error:
            return ToolResult(f"Candidate file not found: {error}")
        return ReturnRequest("report", AdversaryReport(args.findings, candidates))
