# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind complete grading censuses to the measured source population."""

from dataclasses import replace

from finestore.rl.curation import ActorCensusInput, GradingEnvironmentEvidence
from finestore.rl.source_identity import SourceId
from pydantic import BaseModel, ConfigDict, model_validator

from experiments.post_training.task_curation.source import RlDataSource


class GradingCensusReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: SourceId
    input: ActorCensusInput
    population_path: str
    scanned_task_specs_sha256: str
    evidence: GradingEnvironmentEvidence
    task_evidence_uri: str
    unassessable_reasons: dict[str, int]

    @model_validator(mode="after")
    def _complete_population(self) -> "GradingCensusReceipt":
        assessed = self.evidence.assessed_tasks
        if assessed + sum(self.unassessable_reasons.values()) != self.input.measured_tasks:
            raise ValueError("Grading census does not cover the complete input population")
        for field in ("required_counts", "declared_counts", "status_counts", "reason_counts"):
            if sum(getattr(self.evidence, field).values()) != assessed:
                raise ValueError(f"Grading census {field} does not cover all assessed tasks")
        return self


def bind_grading_census(source: RlDataSource, receipt: GradingCensusReceipt) -> RlDataSource:
    """Expose evidence only while its complete actor input binding remains current."""
    dataset = source.dataset
    if (
        dataset is None
        or (source.info.id, dataset.name, dataset.revision, source.version)
        != (receipt.source_id, receipt.input.dataset_id, receipt.input.dataset_revision, receipt.input.converter_version)
        or source.info.actor_census_input != receipt.input
    ):
        return replace(source, info=replace(source.info, grading_environment=None))
    return replace(source, info=replace(source.info, grading_environment=receipt.evidence))
