# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind exact-input sampled content Results without changing execution qualification."""

from dataclasses import replace

from finestore.rl.curation import ContentEvidenceReference, IndependentQC
from pydantic import BaseModel, ConfigDict, model_validator
from taskcompendium.pipeline.models import ReviewRecord

from experiments.post_training.task_curation.full_content_binding import (
    RetainedLineageProof,
    clear_stale_full_content_result,
    lineage_note,
)
from experiments.post_training.task_curation.sample_qc_artifacts import SampleQCReview
from experiments.post_training.task_curation.source import CatalogConfig, RlDataSource
from infra.marina.apps.rl_data_catalog.server.quality import SAMPLE_MAX_MISSING_FRACTION
from infra.marina.apps.rl_data_catalog.server.source_identity import SourceId


def sample_matches_lineage(receipt: "SampleQCReceipt", lineage: RetainedLineageProof) -> bool:
    """Whether an archived panel is exactly the retained release's byte-identical original panel."""
    panel = receipt.review.panel
    return receipt.source_id == lineage.source_id and (
        panel.dataset_id,
        panel.dataset_revision,
        panel.converter_version,
        panel.task_specs_sha256,
        panel.retained_tasks,
    ) == (
        lineage.original_dataset_id,
        lineage.original_dataset_revision,
        lineage.original_input.converter_version,
        lineage.original_input.task_specs_sha256,
        lineage.original_input.measured_tasks,
    )


class SampleMissingObservation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    review: ReviewRecord
    batch_receipt: ContentEvidenceReference


class SampleQCReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: SourceId
    review: SampleQCReview
    artifact_sha256: str
    missing_observations: tuple[SampleMissingObservation, ...] = ()

    @model_validator(mode="after")
    def _source_identity(self) -> "SampleQCReceipt":
        if self.source_id != self.review.panel.source_id:
            raise ValueError("Sample Result belongs to another source")
        return self


def require_sample_qualification(receipt: SampleQCReceipt) -> None:
    """Require at least 95% conclusive reviews of the unchanged attempted panel."""
    review = receipt.review
    result = review.result
    missing = result.unavailable_count + result.uncertain_count
    if (
        review.attempted_count != review.panel.sample_size
        or result.reviewed + missing != review.attempted_count
        or missing > SAMPLE_MAX_MISSING_FRACTION * review.attempted_count
    ):
        raise ValueError("SAMPLE requires at least 95% reviewed rows in its fixed panel (at most 5% missing)")
    ids = [row.review.task_id for row in receipt.missing_observations]
    if len(ids) != missing or len(set(ids)) != missing:
        raise ValueError("SAMPLE receipt must enumerate every missing task and its batch evidence")


def bind_sample_result[ConfigT: CatalogConfig](
    source: RlDataSource[ConfigT],
    receipt: SampleQCReceipt | None,
    *,
    retained_lineage: RetainedLineageProof | None = None,
) -> RlDataSource[ConfigT]:
    """Attach exact-input content evidence while preserving execution qualification."""
    if receipt is None:
        return clear_stale_full_content_result(source)
    panel = receipt.review.panel
    dataset, actor = source.dataset, source.info.actor_census_input
    if (
        dataset is None
        or actor is None
        or (
            source.info.id,
            dataset.name,
            dataset.revision,
            source.version,
            actor.task_specs_sha256,
            actor.measured_tasks,
        )
        != (
            receipt.source_id,
            panel.dataset_id,
            panel.dataset_revision,
            panel.converter_version,
            panel.task_specs_sha256,
            panel.retained_tasks,
        )
    ):
        if retained_lineage is not None and sample_matches_lineage(receipt, retained_lineage):
            result = receipt.review.result
            return replace(
                source,
                info=replace(
                    source.info,
                    quality_result=result,
                    independent_qc=IndependentQC(
                        verdict=result.verdict,
                        method=lineage_note("", retained_lineage, "Archived SAMPLE content panel"),
                        reviewed=result.reviewed,
                        evidence_url=receipt.review.path,
                    ),
                ),
            )
        raise ValueError(f"Sample Result belongs to another current population: {source.info.id}")
    source = clear_stale_full_content_result(source)
    if source.info.full_content_result_binding is not None:
        return source
    result = receipt.review.result
    return replace(
        source,
        info=replace(
            source.info,
            quality_result=result,
            independent_qc=IndependentQC(
                verdict=result.verdict,
                method=(
                    f"SAMPLE content panel: seed {panel.sample_seed}, {panel.sample_size} selected / "
                    f"{panel.retained_tasks} current tasks; pinned GLM-5.3 v3; execution qualification pending"
                ),
                reviewed=result.reviewed,
                evidence_url=receipt.review.path,
            ),
        ),
    )
