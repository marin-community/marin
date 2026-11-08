# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import override

from taskcompendium.models import TaskSpec

from forge.effects.context import Context
from forge.harness.protocol import ReturnRequest, ToolResult
from forge.harness.tools.base import ValidatedTool
from forge.harness.tools.grade import CandidateArgs, read_candidate
from forge.lowering import check_candidates
from forge.model.controls import Candidate, RewardBand


@dataclass(frozen=True, slots=True)
class AdversaryClaim:
    claim: str
    candidate: Candidate
    expected: RewardBand


@dataclass(frozen=True, slots=True)
class AdversaryReport:
    findings: tuple[AdversaryClaim, ...]


@dataclass(frozen=True, slots=True)
class ClaimArgs:
    claim: str
    candidate: CandidateArgs
    expected: RewardBand


@dataclass(frozen=True, slots=True)
class ReportArgs:
    findings: tuple[ClaimArgs, ...]


@dataclass(frozen=True, slots=True)
class ReportTool(ValidatedTool[ReportArgs]):
    task: TaskSpec
    name = "report"
    description = (
        "Report demonstrated grading defects. Each finding needs a claim, candidate reply/files, and expected "
        "status and reward band showing the defect (for example, a wrong answer receiving full credit). "
        "Validation grades each candidate against the original task. Use empty findings when no defect was found."
    )
    args_type = ReportArgs

    def __post_init__(self) -> None:
        object.__setattr__(self, "task", self.task.model_copy(deep=True))

    @override
    async def run(self, ctx: Context, args: ReportArgs) -> ToolResult | ReturnRequest[AdversaryReport]:
        try:
            findings = tuple(
                [
                    AdversaryClaim(finding.claim, await read_candidate(ctx, finding.candidate), finding.expected)
                    for finding in args.findings
                ]
            )
            check_candidates(self.task, tuple(finding.candidate for finding in findings))
        except ValueError as error:
            return ToolResult(f"Invalid report: {error}")
        return ReturnRequest("report", AdversaryReport(findings))
