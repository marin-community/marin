# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compute execution oversight and informational review results for Atlas."""

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from finestore.rl.curation import (
    AssessmentItem,
    AssessmentPopulation,
    JudgmentStatus,
    QualityResultFinalization,
    SourceDeclaration,
    SourceQualityResult,
    Verdict,
)


class Quality(StrEnum):
    QUICK = "QUICK"
    SAMPLE = "SAMPLE"
    FULL = "FULL"


@dataclass(frozen=True)
class ReviewCompleteness:
    reviewed: int
    required: int
    coverage_complete: bool

    @property
    def provisional(self) -> bool:
        return not self.coverage_complete or self.reviewed < self.required


def review_completeness(
    declaration: SourceDeclaration, populations: list[AssessmentPopulation], items: list[AssessmentItem]
) -> dict[str, Any] | None:
    """Bind Result completeness to its required review panel, independently of Quality."""
    aggregate = declaration.quality_result
    if aggregate is not None:
        assert aggregate.sample_count is not None
        completeness = ReviewCompleteness(
            reviewed=aggregate.reviewed,
            required=aggregate.sample_count,
            coverage_complete=aggregate.unavailable_count == 0,
        )
    elif populations:
        required = next(row.sample_count for row in populations if row.cohort == "admitted")
        panel = [row for row in items if row.cohort == "admitted"]
        completeness = ReviewCompleteness(
            reviewed=sum(
                row.status == JudgmentStatus.REVIEWED and row.has_task and row.verdict != Verdict.UNKNOWN
                for row in panel
            ),
            required=required,
            coverage_complete=len(panel) == required and all(row.status == JudgmentStatus.REVIEWED for row in panel),
        )
    else:
        return None
    return {
        **asdict(completeness),
        "provisional": (
            completeness.provisional
            and not (
                declaration.quality_result_finalization is QualityResultFinalization.COMPLETE
                or (
                    declaration.quality_result_finalization is QualityResultFinalization.ACCEPTED_INCOMPLETE
                    and declaration.quality_result_waiver is not None
                )
            )
        ),
    }


def quality(declaration: dict[str, Any], curation: dict[str, Any] | None) -> Quality:
    """Report whole-source review coverage, including explicitly accepted FULL waivers."""
    review = declaration.get("quality_result")
    qc = declaration.get("independent_qc")
    retained = declaration.get("within_source_dedup", {}).get("unique_count")
    if (
        retained
        and qc is not None
        and review is not None
        and declaration.get("quality_result_finalization") == QualityResultFinalization.ACCEPTED_INCOMPLETE
        and declaration.get("quality_result_waiver") is not None
        and qc["reviewed"] == review["reviewed"] + review.get("uncertain_count", 0)
        and qc["reviewed"] + review.get("unavailable_count", 0) == retained
    ):
        return Quality.FULL
    if (
        retained
        and qc is not None
        and qc["reviewed"] == retained
        and review is not None
        and review["reviewed"] + review.get("uncertain_count", 0) == retained
        and review.get("unavailable_count", 0) == 0
    ):
        return Quality.FULL
    if declaration.get("oversight") in {"sample", "full"}:
        return Quality.SAMPLE
    if (curation or {}).get("mode") in {"sample", "full"}:
        controls = (curation or {}).get("verification") or {}
        if controls.get("status") in {"passed", "validated"}:
            return Quality.SAMPLE
    return Quality.QUICK


def measured_result(declaration: dict[str, Any], review: dict[str, Any] | None) -> SourceQualityResult | None:
    """Return the current quality result, or None when no review is recorded."""
    stored = declaration.get("quality_result")
    if stored is not None:
        measured = SourceQualityResult.model_validate(stored)
    elif review is not None:
        measured = SourceQualityResult(
            reviewed=review["result_reviewed"],
            issue_count=review["issue_count"],
            issues=[],
            repair_recommendation="See the sampled assessment evidence",
        )
    else:
        return None
    return measured


def result(declaration: dict[str, Any], review: dict[str, Any] | None) -> str | None:
    """Compute Result from conclusive issue counts using inclusive boundaries."""
    measured = measured_result(declaration, review)
    return measured.verdict.value.upper() if measured is not None else None


def issue_rate(declaration: dict[str, Any], review: dict[str, Any] | None) -> float | None:
    measured = measured_result(declaration, review)
    return measured.issue_rate if measured is not None else None


def intended_use(declaration: dict[str, Any]) -> str:
    """Return eval for benchmark or evaluation evidence, otherwise train."""
    receipt = (declaration.get("lexical_contamination") or {}).get("receipt")
    if declaration["is_benchmark"] or (receipt is not None and receipt["use"] in {"eval", "benchmark"}):
        return "eval"
    return "train"
