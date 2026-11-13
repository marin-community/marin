# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Admission checks over the census evidence carried by each source."""

from finestore.rl.curation import (
    ActorCensusInput,
    CurationSummary,
    GradingEnvironmentEvidence,
    OversightLevel,
    SourceDeclaration,
)

MISSING_GRADER_REASON = "No executable grader is declared"


def population_admission_blocker(
    binding: ActorCensusInput | None, retained_tasks: int, grading: GradingEnvironmentEvidence | None
) -> str | None:
    if retained_tasks <= 0:
        return "admission requires a nonempty retained population"
    if binding is None or binding.measured_tasks != retained_tasks:
        return "actor census does not cover the retained population"
    if grading is None or grading.assessed_tasks != retained_tasks:
        return "grading census does not cover the complete retained population"
    if any(
        sum(getattr(grading, field).values()) != retained_tasks
        for field in ("required_counts", "declared_counts", "status_counts", "reason_counts")
    ):
        return "grading census counts do not cover the complete retained population"
    if grading.reason_counts.get(MISSING_GRADER_REASON, 0):
        return "admitted source has tasks without runnable graders"
    return None


def require_declaration_admission(declaration: SourceDeclaration) -> None:
    """Reject incomplete or differently pinned evidence before publication or loading."""
    binding, pipeline = declaration.actor_census_input, declaration.pipeline
    if (
        binding is None
        or pipeline is None
        or (binding.dataset_id, binding.dataset_revision, binding.converter_version)
        != (declaration.dataset_id, declaration.dataset_revision, pipeline.version)
    ):
        raise ValueError(f"Admission census belongs to another current population: {declaration.id}")
    if blocker := population_admission_blocker(
        binding, declaration.within_source_dedup.unique_count, declaration.grading_environment
    ):
        raise ValueError(f"Admission census blocked: {declaration.id}: {blocker}")


def require_content_result(
    declaration: SourceDeclaration, curation: CurationSummary | None, assessment_key: str | None
) -> None:
    """Require a computed Result for SAMPLE/FULL snapshots before selecting them."""
    if assessment_key is not None or declaration.quality_result is not None:
        return
    controls = curation.verification if curation is not None else None
    reviewed = declaration.oversight in {OversightLevel.SAMPLE, OversightLevel.FULL} or (
        curation is not None
        and curation.mode in {"sample", "full"}
        and controls is not None
        and controls.status in {"passed", "validated"}
    )
    if reviewed:
        raise ValueError(f"SAMPLE/FULL quality requires a computed Result: {declaration.id}")
