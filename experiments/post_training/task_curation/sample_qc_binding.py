# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind exact-input sampled content Results without changing execution qualification."""

from dataclasses import replace

from finestore.rl.curation import IndependentQC
from infra.marina.apps.rl_data_catalog.server.source_identity import SourceId
from pydantic import BaseModel, ConfigDict, model_validator

from experiments.post_training.task_curation.sample_qc_artifacts import SampleQCReview
from experiments.post_training.task_curation.source import CatalogConfig, RlDataSource


class SampleQCReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: SourceId
    review: SampleQCReview
    artifact_sha256: str

    @model_validator(mode="after")
    def _source_identity(self) -> "SampleQCReceipt":
        if self.source_id != self.review.panel.source_id:
            raise ValueError("Sample Result belongs to another source")
        return self


def bind_sample_result[ConfigT: CatalogConfig](
    source: RlDataSource[ConfigT], receipt: SampleQCReceipt | None
) -> RlDataSource[ConfigT]:
    """Attach exact-input content evidence while preserving execution qualification."""
    if receipt is None:
        return source
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
        raise ValueError(f"Sample Result belongs to another current population: {source.info.id}")
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
                reviewed=receipt.review.attempted_count,
                evidence_url=receipt.review.path,
            ),
        ),
    )
