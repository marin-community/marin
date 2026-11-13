# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact-input final content Results and their immutable evidence references."""

from finestore.rl.curation import ActorCensusInput, QualityResultFinalization, SourceQualityResult
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContentEvidenceReference(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    uri: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class FullContentResultBinding(BaseModel):
    """A verified whole-source Result, including explicitly accepted missing reviews."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str
    input: ActorCensusInput
    contract_index: ContentEvidenceReference
    review_receipt: ContentEvidenceReference
    records: ContentEvidenceReference
    result: SourceQualityResult
    coverage_complete: bool
    finalization: QualityResultFinalization
    waiver: ContentEvidenceReference | None = None

    @model_validator(mode="after")
    def _whole_source_result(self) -> "FullContentResultBinding":
        if self.result.sample_count != self.input.measured_tasks:
            raise ValueError("Whole-source Result does not account for its retained population")
        if self.coverage_complete != (self.result.unavailable_count == 0):
            raise ValueError("Content coverage differs from unavailable observations")
        if self.finalization is QualityResultFinalization.COMPLETE:
            if not self.coverage_complete or self.waiver is not None:
                raise ValueError("Complete finalization requires complete coverage without a waiver")
        elif self.coverage_complete or self.waiver is None:
            raise ValueError("Accepted incomplete finalization requires missing reviews and waiver evidence")
        return self
