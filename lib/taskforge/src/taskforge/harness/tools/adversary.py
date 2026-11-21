# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Annotated, override

from pydantic import Field
from taskcompendium.models import TaskSpec

from taskforge.effects.context import Context
from taskforge.effects.events import HANDLED_FAILURE
from taskforge.harness.protocol import ReturnRequest, ToolResult
from taskforge.harness.tools.base import ValidatedTool
from taskforge.harness.tools.grade import CandidateArgs, read_candidate
from taskforge.lowering import check_candidates
from taskforge.model.controls import Candidate
from taskforge.model.review import Defect, DefectClaim, Observation, ObservedStatus


@dataclass(frozen=True, slots=True)
class AdversaryClaim:
    claim: str
    candidate: Candidate
    defect: DefectClaim


@dataclass(frozen=True, slots=True)
class AdversaryReport:
    findings: tuple[AdversaryClaim, ...]


@dataclass(frozen=True, slots=True)
class FindingArgs:
    defect: Defect
    why_the_grade_is_wrong: str
    candidate: CandidateArgs
    grader_observed_status: ObservedStatus
    grader_observed_reward: Annotated[float, Field(ge=0, le=1)] | None
    deserved_reward: Annotated[float, Field(ge=0, le=1)] | None


@dataclass(frozen=True, slots=True)
class ReportArgs:
    findings: tuple[FindingArgs, ...]


def defect_claim(args: FindingArgs) -> DefectClaim:
    """Reject a finding whose own numbers describe a correct grade or mix up observed and deserved."""
    if (args.grader_observed_status is ObservedStatus.GRADED) != (args.grader_observed_reward is not None):
        raise ValueError("grader_observed_reward is the graded reward, and null unless the status is graded")
    if (args.defect is Defect.GRADER_CRASH) != (args.grader_observed_status is ObservedStatus.GRADER_ERROR):
        raise ValueError("grader_observed_status is grader_error exactly for a grader_crash finding")
    if args.defect is not Defect.GRADER_CRASH and args.deserved_reward is None:
        raise ValueError(f"a {args.defect} finding needs deserved_reward: the reward correct grading would give")
    observed = Observation(args.grader_observed_status, args.grader_observed_reward)
    claim = DefectClaim(args.defect, observed, args.deserved_reward)
    if not claim.shown_by(observed):
        relation = "more" if args.defect is Defect.FALSE_ACCEPT else "less"
        raise ValueError(
            f"a {args.defect} finding needs the grader to give {relation} credit than deserved; grader_observed "
            f"{args.grader_observed_status} {args.grader_observed_reward} with deserved_reward {args.deserved_reward} "
            "describes a correct grade"
        )
    return claim


@dataclass(frozen=True, slots=True)
class ReportTool(ValidatedTool[ReportArgs]):
    task: TaskSpec
    name = "report"
    description = (
        "Report demonstrated grading defects, or an empty findings list when no defect was demonstrated. "
        "Each finding is one concrete candidate (reply and workspace files) with two DIFFERENT grades: "
        "grader_observed_status and grader_observed_reward copy what your grade call actually returned for that "
        "exact candidate; deserved_reward is what a correct grader would give it. defect names the direction. "
        "false_accept: the grader gives more credit than deserved, e.g. a wrong answer observed graded 1.0, "
        "deserved 0.0. false_reject: the grader gives less, e.g. a correct answer observed graded 0.0 (or "
        "submission_failure with null reward), deserved 1.0. grader_crash: the grader errors instead of grading, "
        "observed grader_error with both rewards null. why_the_grade_is_wrong says why the observed grade is wrong. "
        "Validation replays each candidate against the original task, then on a fresh verifier, and keeps a finding "
        "only when the replayed grade is wrong in the stated direction."
    )
    args_type = ReportArgs

    def __post_init__(self) -> None:
        object.__setattr__(self, "task", self.task.model_copy(deep=True))

    @override
    async def run(self, ctx: Context, args: ReportArgs) -> ToolResult | ReturnRequest[AdversaryReport]:
        findings: list[AdversaryClaim] = []
        try:
            for index, finding in enumerate(args.findings):
                try:
                    claim = defect_claim(finding)
                except ValueError as error:
                    raise ValueError(f"finding {index}: {error}") from error
                candidate = await read_candidate(ctx, finding.candidate, self.task)
                findings.append(AdversaryClaim(finding.why_the_grade_is_wrong, candidate, claim))
            check_candidates(self.task, tuple(finding.candidate for finding in findings))
        except (ValueError, HANDLED_FAILURE) as error:
            return ToolResult(f"Invalid report: {error}")
        return ReturnRequest("report", AdversaryReport(tuple(findings)))
