# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compose whole-source content evidence after SAMPLE without retaining stale Results."""

from dataclasses import replace

from finestore.rl.curation import ActorCensusInput, IndependentQC
from pydantic import BaseModel, ConfigDict

from experiments.post_training.task_curation.full_content_results import FullContentResultBinding
from experiments.post_training.task_curation.source import CatalogConfig, RlDataSource


class RetainedLineageProof(BaseModel):
    """A zero-drop retained release whose population is byte-identical to its original.

    Presence of this proof records the publication readback that verified every
    retained task hash against the frozen original conversion, so archived
    original-pin evidence stays valid for the retained source.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str
    original_dataset_id: str
    original_dataset_revision: str
    original_input: ActorCensusInput
    original_unique_count: int
    retained_content_sha256: str


def lineage_note(notes: str, lineage: RetainedLineageProof, label: str) -> str:
    """Record a lineage-validated receipt on the retained declaration."""
    return (
        f"{label} validated against the original pin {lineage.original_dataset_id}"
        f"@{lineage.original_dataset_revision} through the zero-drop retained lineage; "
        f"retained readback {lineage.retained_content_sha256[:12]} proves byte-identical "
        "retained tasks. " + notes
    ).strip()


def full_content_matches_lineage(binding: FullContentResultBinding, lineage: RetainedLineageProof) -> bool:
    """Whether an archived whole-source Result is exactly the retained original's review."""
    return (
        binding.source_id == lineage.source_id
        and (binding.input.dataset_id, binding.input.dataset_revision)
        == (
            lineage.original_dataset_id,
            lineage.original_dataset_revision,
        )
        and binding.input == lineage.original_input
        and lineage.original_unique_count == binding.input.measured_tasks
    )


def matches_full_content_input(source: RlDataSource, binding: FullContentResultBinding) -> bool:
    dataset, actor, dedup = source.dataset, source.info.actor_census_input, source.info.within_source_dedup
    return (
        dataset is not None
        and actor is not None
        and dedup is not None
        and (source.info.id, dataset.name, dataset.revision, source.version, dedup.unique_count)
        == (
            binding.source_id,
            binding.input.dataset_id,
            binding.input.dataset_revision,
            binding.input.converter_version,
            binding.input.measured_tasks,
        )
        and actor == binding.input
    )


def clear_stale_full_content_result[ConfigT: CatalogConfig](source: RlDataSource[ConfigT]) -> RlDataSource[ConfigT]:
    """Clear carried whole-source evidence before attaching a current SAMPLE panel."""
    binding = source.info.full_content_result_binding
    if binding is not None and matches_full_content_input(source, binding):
        return source
    if binding is None and source.info.quality_result_finalization is None:
        return source
    return replace(
        source,
        info=replace(
            source.info,
            quality_result=None,
            independent_qc=None,
            quality_result_finalization=None,
            quality_result_waiver=None,
            full_content_result_binding=None,
        ),
    )


def bind_full_content_result[ConfigT: CatalogConfig](
    source: RlDataSource[ConfigT],
    binding: FullContentResultBinding | None,
    *,
    retained_lineage: RetainedLineageProof | None = None,
) -> RlDataSource[ConfigT]:
    """Attach a verified exact-input final Result while preserving execution evidence."""
    if binding is None:
        source = clear_stale_full_content_result(source)
        binding = source.info.full_content_result_binding
        if binding is None:
            return source
    if not matches_full_content_input(source, binding):
        if retained_lineage is not None and full_content_matches_lineage(binding, retained_lineage):
            result = binding.result
            return replace(
                source,
                info=replace(
                    source.info,
                    quality_result=result,
                    quality_result_finalization=binding.finalization,
                    quality_result_waiver=binding.waiver,
                    independent_qc=IndependentQC(
                        verdict=result.verdict,
                        method=lineage_note("", retained_lineage, "Archived whole-source Result"),
                        reviewed=result.reviewed + result.uncertain_count,
                        evidence_url=binding.review_receipt.uri,
                    ),
                ),
            )
        raise ValueError(f"Whole-source Result belongs to another current retained population: {source.info.id}")
    source = clear_stale_full_content_result(source)
    result = binding.result
    return replace(
        source,
        info=replace(
            source.info,
            quality_result=result,
            quality_result_finalization=binding.finalization,
            quality_result_waiver=binding.waiver,
            full_content_result_binding=binding,
            independent_qc=IndependentQC(
                verdict=result.verdict,
                method=f"Final whole-source content review: {binding.finalization.value}; execution unchanged",
                reviewed=result.reviewed + result.uncertain_count,
                evidence_url=binding.review_receipt.uri,
            ),
        ),
    )
