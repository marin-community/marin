# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import override

from taskcompendium.models import TaskSpec

from forge.effects.context import Context
from forge.harness.protocol import ReturnRequest, ToolResult
from forge.harness.tools.base import ValidatedTool
from forge.harness.tools.grade import CandidateArgs, read_candidate
from forge.lowering import check_candidates
from forge.model.content_hash import digest
from forge.model.controls import Candidate, RewardBand
from forge.model.review import Defect, DefectClaim, Observation, ObservedStatus


@dataclass(frozen=True, slots=True)
class AdversaryClaim:
    """A reported finding; ``expected`` is the legacy observed-grade band, superseded by ``defect``."""

    claim: str
    candidate: Candidate
    expected: RewardBand | None = None
    defect: DefectClaim | None = None


@dataclass(frozen=True, slots=True)
class AdversaryReport:
    findings: tuple[AdversaryClaim, ...]


@dataclass(frozen=True, slots=True)
class FindingArgs:
    defect: Defect
    why_the_grade_is_wrong: str
    candidate: CandidateArgs
    grader_observed_status: ObservedStatus
    grader_observed_reward: float | None
    deserved_reward: float | None


@dataclass(frozen=True, slots=True)
class ReportArgs:
    findings: tuple[FindingArgs, ...]


def defect_claim(args: FindingArgs) -> DefectClaim:
    """Reject a finding whose own numbers describe a correct grade or mix up observed and deserved."""
    for name, value in (
        ("grader_observed_reward", args.grader_observed_reward),
        ("deserved_reward", args.deserved_reward),
    ):
        if value is not None and not (math.isfinite(value) and 0 <= value <= 1):
            raise ValueError(f"{name} {value} is outside [0, 1]")
    if (args.grader_observed_status is ObservedStatus.GRADED) != (args.grader_observed_reward is not None):
        raise ValueError("grader_observed_reward is the graded reward, and null unless the status is graded")
    if args.defect is Defect.GRADER_CRASH and args.grader_observed_status is not ObservedStatus.GRADER_ERROR:
        raise ValueError("a grader_crash finding needs grader_observed_status grader_error")
    if args.defect is not Defect.GRADER_CRASH and args.grader_observed_status is ObservedStatus.GRADER_ERROR:
        raise ValueError("a grader that errors instead of grading is a grader_crash finding")
    if args.defect is not Defect.GRADER_CRASH and args.deserved_reward is None:
        raise ValueError(f"a {args.defect} finding needs deserved_reward: the reward correct grading would give")
    claim = DefectClaim(
        args.defect, Observation(args.grader_observed_status, args.grader_observed_reward), args.deserved_reward
    )
    if not claim.shown_by(claim.observed):
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
    # This session's own grades by candidate digest, filled by the stage; reports must agree with them.
    graded: Mapping[str, Observation]
    name = "report"
    description = (
        "Report demonstrated grading defects, or an empty findings list when the grader behaved correctly. "
        "Each finding is one concrete candidate (reply and workspace files) with two DIFFERENT grades: "
        "grader_observed_status and grader_observed_reward copy what your grade call actually returned for that "
        "exact candidate; deserved_reward is what a correct grader would give it. defect names the direction. "
        "false_accept: the grader gives more credit than deserved, e.g. a wrong answer observed graded 1.0, "
        "deserved 0.0. false_reject: the grader gives less, e.g. a correct answer observed graded 0.0 (or "
        "submission_failure with null reward), deserved 1.0. grader_crash: the grader errors instead of grading, "
        "observed grader_error with both rewards null. why_the_grade_is_wrong says why the observed grade is wrong. "
        "Validation replays each candidate against the original task and keeps a finding only when the replayed "
        "grade is wrong in the stated direction."
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
                candidate = await read_candidate(ctx, finding.candidate)
                own = self.graded.get(digest(candidate))
                if own is not None and not own.matches(claim.observed):
                    raise ValueError(
                        f"finding {index}: your grade call on this exact candidate returned {own.status} with reward "
                        f"{own.reward}, not {claim.observed.status} with reward {claim.observed.reward}. The "
                        "grader_observed fields copy what the grader returned; the reward the candidate should get "
                        "goes in deserved_reward"
                    )
                findings.append(AdversaryClaim(finding.why_the_grade_is_wrong, candidate, defect=claim))
            check_candidates(self.task, tuple(finding.candidate for finding in findings))
        except ValueError as error:
            return ToolResult(f"Invalid report: {error}")
        return ReturnRequest("report", AdversaryReport(tuple(findings)))
