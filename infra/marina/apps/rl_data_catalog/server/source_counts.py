# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Current-input population counts under the explicit no-oracle policy."""

from typing import Literal

from finestore.rl.curation import ActorCensusInput, ContentEvidenceReference, SourceUse
from pydantic import BaseModel, ConfigDict, Field, model_validator

NO_ORACLE_POLICY_SHA256 = "028c7dfbc7800350eb6bf1ccb2be61b1810447f3d7fb76596318764c0e2f3953"


class SourceCountEvidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str
    input: ActorCensusInput
    intended_use: SourceUse
    oracle_tasks: int = Field(ge=0)
    static_blocked_tasks: int | None = Field(default=None, ge=0)
    static_reason_counts: dict[str, int] = Field(default_factory=dict)
    policy: ContentEvidenceReference
    oracle_inventory: ContentEvidenceReference
    static_audit: ContentEvidenceReference | None = None

    @model_validator(mode="after")
    def check_population(self) -> "SourceCountEvidence":
        if self.policy.sha256 != NO_ORACLE_POLICY_SHA256:
            raise ValueError("Count evidence requires the explicit no-oracle policy ruling")
        if self.oracle_tasks > self.input.measured_tasks:
            raise ValueError("Oracle count exceeds the measured population")
        if self.static_blocked_tasks is not None and (
            self.oracle_tasks != 0 or self.static_blocked_tasks > self.input.measured_tasks or self.static_audit is None
        ):
            raise ValueError("Static retention requires a zero-oracle population and a conserved partition")
        return self


class SourceCounts(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal[1] = 1
    sources: tuple[SourceCountEvidence, ...]

    @model_validator(mode="after")
    def check_source_ids(self) -> "SourceCounts":
        if len({source.source_id for source in self.sources}) != len(self.sources):
            raise ValueError("Count evidence contains duplicate source IDs")
        return self
