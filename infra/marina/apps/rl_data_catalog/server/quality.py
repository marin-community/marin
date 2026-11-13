# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compute execution oversight and informational review results for Atlas."""

from typing import Any

from finestore.rl.curation import SourceQualityResult


def quality(declaration: dict[str, Any], curation: dict[str, Any] | None) -> str:
    """Reserve FULL for a separately verified complete content-review assembly."""
    review = declaration.get("quality_result")
    qc = declaration.get("independent_qc")
    retained = declaration.get("within_source_dedup", {}).get("unique_count")
    if (
        retained
        and qc is not None
        and qc["reviewed"] == retained
        and review is not None
        and review["reviewed"] + review.get("uncertain_count", 0) == retained
        and review.get("unavailable_count", 0) == 0
    ):
        return "FULL"
    if declaration.get("oversight") in {"sample", "full"}:
        return "SAMPLE"
    if (curation or {}).get("mode") in {"sample", "full"}:
        controls = (curation or {}).get("verification") or {}
        if controls.get("status") in {"passed", "validated"}:
            return "SAMPLE"
    return "QUICK"


def result(declaration: dict[str, Any], review: dict[str, Any] | None) -> str | None:
    """Compute Result from conclusive issue counts using inclusive boundaries."""
    stored = declaration.get("quality_result")
    if stored is not None:
        measured = SourceQualityResult.model_validate(stored)
    elif review is not None:
        counts = review["verdict_counts"]
        measured = SourceQualityResult(
            reviewed=sum(counts.get(key, 0) for key in ("good", "some_issues", "bad")),
            issue_count=counts.get("some_issues", 0) + counts.get("bad", 0),
            issues=[],
            repair_recommendation="See the sampled assessment evidence",
        )
    else:
        return None
    return measured.verdict.value.upper() if measured.reviewed else None


def intended_use(declaration: dict[str, Any]) -> str:
    """Use the declaration's lexical policy and explicit benchmark marker."""
    receipt = (declaration.get("lexical_contamination") or {}).get("receipt")
    if declaration["is_benchmark"] or (receipt is not None and receipt["use"] in {"eval", "benchmark"}):
        return "eval"
    return "train"
