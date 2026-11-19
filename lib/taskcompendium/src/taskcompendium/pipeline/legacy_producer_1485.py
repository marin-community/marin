# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded qualification of the separately approved producer-1485 original passes."""

from typing import Literal

from finestore.rl.curation import ContentEvidenceReference
from pydantic import BaseModel, ConfigDict, Field, model_validator

from taskcompendium.harbor.runtime import TaskSpecRuntimeBinding
from taskcompendium.pipeline.legacy_roles import LegacyRoleAllocation, require_legacy_allocation_contract


class LegacyProducer1485AdoptionDecision(BaseModel):
    """The separately approved completed-pass and unscored partition."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    version: Literal[1]
    disposition: Literal["adopted"]
    producer_revision: Literal["1485f1b6c64640b5c15a1bffc113d49eefdc60b6"]
    input_index_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    adopted_count: int = Field(gt=0)
    excluded_unscored: int = Field(ge=0)
    ruling: ContentEvidenceReference


class LegacyProducer1485QualificationReceipt(BaseModel):
    """Qualify an original pass using only its actual producer's constructor capture."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    version: Literal[1]
    producer_revision: Literal["1485f1b6c64640b5c15a1bffc113d49eefdc60b6"]
    source_task_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_task_checksum: str
    consumed_task_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    consumed_task_checksum: str
    published_task_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_cohort_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_runtime: TaskSpecRuntimeBinding
    actor: LegacyRoleAllocation | None
    oracle_helper: LegacyRoleAllocation | None
    verifier: LegacyRoleAllocation
    producer_manifest: ContentEvidenceReference
    producer_closure: ContentEvidenceReference
    input_partition: ContentEvidenceReference
    audit: ContentEvidenceReference
    decision: ContentEvidenceReference
    adoption_decision: LegacyProducer1485AdoptionDecision

    @model_validator(mode="after")
    def validate_frozen_contract(self):
        if (
            self.adoption_decision.input_index_sha256 != self.input_partition.sha256
            or self.adoption_decision.audit_sha256 != self.audit.sha256
        ):
            raise ValueError("Producer-1485 qualification differs from its exact adopted scope")
        require_legacy_allocation_contract(self)
        return self
