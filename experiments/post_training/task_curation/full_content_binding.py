# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compose whole-source content evidence after SAMPLE without retaining stale Results."""

from dataclasses import replace

from finestore.rl.curation import IndependentQC

from experiments.post_training.task_curation.full_content_results import FullContentResultBinding
from experiments.post_training.task_curation.source import CatalogConfig, RlDataSource


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
    source: RlDataSource[ConfigT], binding: FullContentResultBinding | None
) -> RlDataSource[ConfigT]:
    """Attach a verified exact-input final Result while preserving execution evidence."""
    source = clear_stale_full_content_result(source)
    binding = binding or source.info.full_content_result_binding
    if binding is None:
        return source
    if not matches_full_content_input(source, binding):
        raise ValueError(f"Whole-source Result belongs to another current retained population: {source.info.id}")
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
