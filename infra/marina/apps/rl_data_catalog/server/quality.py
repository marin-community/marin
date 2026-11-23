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
    OversightLevel,
    QualityResultFinalization,
    SourceDeclaration,
    SourceQualityResult,
    SourceUse,
    Verdict,
)

SAMPLE_MAX_MISSING_FRACTION = 0.05


class Quality(StrEnum):
    QUICK = "QUICK"
    SAMPLE = "SAMPLE"
    FULL = "FULL"


@dataclass(frozen=True)
class QualityDisplay:
    quality: Quality
    review_status: str | None


def sample_coverage_status(declaration: dict[str, Any], level: Quality) -> str | None:
    """Show the reviewed/retained coverage a SAMPLE source's evidence records.

    The oversight evidence carries the reviewed coverage that carried prior
    reviews joined onto the fixed panel; the Atlas shows it beside the SAMPLE
    label. Sources without recorded coverage keep a bare label.
    """
    if level is not Quality.SAMPLE:
        return None
    evidence = declaration.get("oversight_evidence") or {}
    reviewed = evidence.get("reviewed")
    retained = (declaration.get("within_source_dedup") or {}).get("unique_count")
    if not isinstance(reviewed, int) or reviewed <= 0 or not isinstance(retained, int) or retained <= 0:
        return None
    return f"SAMPLE, {reviewed:,} of {retained:,} reviewed"


def quality_display(
    declaration: dict[str, Any], curation: dict[str, Any] | None, assessment_key: str | None
) -> QualityDisplay:
    """Keep unfinished reviews pending without changing declared oversight or evidence."""
    aggregate = declaration.get("quality_result")
    if aggregate is not None and aggregate["reviewed"] == 0:
        return QualityDisplay(Quality.QUICK, "SAMPLE review in progress")
    level = quality(declaration, curation)
    if level != Quality.QUICK and declaration.get("quality_result") is None and assessment_key is None:
        return QualityDisplay(Quality.QUICK, "SAMPLE review in progress")
    return QualityDisplay(level, sample_coverage_status(declaration, level))


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
    sample_qualified = (
        declaration.oversight is OversightLevel.SAMPLE
        and declaration.quality_result_finalization is None
        and aggregate is not None
        and declaration.independent_qc is not None
        and declaration.independent_qc.reviewed == aggregate.reviewed
        and completeness.required > 0
        and completeness.required - completeness.reviewed <= SAMPLE_MAX_MISSING_FRACTION * completeness.required
    )
    return {
        **asdict(completeness),
        "provisional": (
            completeness.provisional
            and not sample_qualified
            and not final_whole_source_result(declaration.model_dump(mode="json"))
        ),
    }


def final_whole_source_result(declaration: dict[str, Any]) -> bool:
    """Accept finality only when its state and evidence account for the retained population."""
    review = declaration.get("quality_result")
    qc = declaration.get("independent_qc")
    retained = (declaration.get("within_source_dedup") or {}).get("unique_count")
    if review is None or qc is None or retained is None:
        return False
    unavailable = review.get("unavailable_count", 0)
    if (
        review.get("sample_count") != retained
        or qc["reviewed"] != review["reviewed"] + review.get("uncertain_count", 0)
        or qc["reviewed"] + unavailable != retained
    ):
        return False
    finalization = declaration.get("quality_result_finalization")
    waiver = declaration.get("quality_result_waiver")
    if finalization == QualityResultFinalization.COMPLETE:
        return unavailable == 0 and waiver is None
    # An accepted waiver may cover at most 1% of the retained population
    # (FULL_WAIVER_MAX_UNREVIEWED_FRACTION); a larger waiver is not whole-source
    # finality and displays as its measured coverage. Integer math keeps the exact bound.
    return (
        finalization == QualityResultFinalization.ACCEPTED_INCOMPLETE
        and unavailable > 0
        and waiver is not None
        and unavailable * 100 <= retained
    )


def review_coverage(declaration: dict[str, Any], completeness: dict[str, Any] | None) -> dict[str, Any]:
    """Review coverage numbers for a source: reviewed / required population, waiver if any."""
    review = declaration.get("quality_result")
    return {
        "reviewed": (completeness or {}).get("reviewed"),
        "required": (completeness or {}).get("required"),
        "coverage_complete": (completeness or {}).get("coverage_complete"),
        "provisional": (completeness or {}).get("provisional"),
        "uncertain": review.get("uncertain_count") if review else None,
        "unavailable": review.get("unavailable_count") if review else None,
        "finalization": declaration.get("quality_result_finalization"),
        "waiver": declaration.get("quality_result_waiver"),
    }


def quality(declaration: dict[str, Any], curation: dict[str, Any] | None) -> Quality:
    """Report whole-source review coverage, including explicitly accepted FULL waivers."""
    review = declaration.get("quality_result")
    qc = declaration.get("independent_qc")
    retained = declaration.get("within_source_dedup", {}).get("unique_count")
    if final_whole_source_result(declaration):
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
    if review is not None and qc is not None and review.get("sample_count") and qc["reviewed"] == review["sample_count"]:
        return Quality.SAMPLE
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
    return measured if measured.reviewed else None


def result(declaration: dict[str, Any], review: dict[str, Any] | None) -> str | None:
    """Compute Result from conclusive issue counts using inclusive boundaries."""
    measured = measured_result(declaration, review)
    return measured.verdict.value.upper() if measured is not None else None


def issue_rate(declaration: dict[str, Any], review: dict[str, Any] | None) -> float | None:
    measured = measured_result(declaration, review)
    return measured.issue_rate if measured is not None else None


def intended_use(declaration: dict[str, Any]) -> SourceUse:
    """Return the converter's explicit Use policy, refusing any other value.

    The catalog writer resolves Use from the registered converter and the load
    refuses a declaration without one, so a missing or free-string value here
    is a serving bug, never a display fallback.
    """
    value = declaration.get("intended_use")
    if value is None:
        raise ValueError(f"Declaration carries no intended_use: {declaration.get('id')}")
    return SourceUse(value)
