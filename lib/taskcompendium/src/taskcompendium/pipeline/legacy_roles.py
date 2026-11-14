# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded allocation qualification for completed trials from the frozen producer."""

from typing import Literal

from finestore.rl.curation import ContentEvidenceReference
from pydantic import BaseModel, ConfigDict, Field, model_validator
from shellbox.machine import NetworkPolicy

from taskcompendium.harbor.runtime import MachineRole, TaskSpecRuntimeBinding
from taskcompendium.models import ExecutionRequirements


class LegacyRoleAllocation(BaseModel):
    """An actual frozen production MachineSpec, observed before provider creation."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    execution_requirements: ExecutionRequirements
    network: NetworkPolicy
    gpus: Literal[0]
    machine_spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class LegacyRoleAdoptionDecision(BaseModel):
    """The exact frozen partition and audit authorized by an adoption ruling."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    version: Literal[1]
    disposition: Literal["adopted"]
    producer_revision: Literal["2c5eea40f264158596f62e593dd21ecaeaa4dc52"]
    input_partition_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ruling: ContentEvidenceReference


class LegacyRoleQualificationReceipt(BaseModel):
    """Qualify one immutable completed pass without changing its original receipt."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    version: Literal[1]
    producer_revision: Literal["2c5eea40f264158596f62e593dd21ecaeaa4dc52"]
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
    adoption_decision: LegacyRoleAdoptionDecision

    @model_validator(mode="after")
    def validate_frozen_contract(self):
        if (
            self.adoption_decision.producer_revision != self.producer_revision
            or self.adoption_decision.input_partition_sha256 != self.input_partition.sha256
            or self.adoption_decision.audit_sha256 != self.audit.sha256
        ):
            raise ValueError("Legacy adoption decision scope differs from its frozen partition or audit")
        if (
            self.source_runtime.task_sha256 != self.source_task_sha256
            or self.consumed_task_sha256 != self.source_task_sha256
            or self.consumed_task_checksum != self.source_task_checksum
        ):
            raise ValueError("Legacy qualification requires the unchanged original packet")
        if (self.actor is None) == (self.oracle_helper is None):
            raise ValueError("Legacy qualification requires the actual actor or its actorless oracle helper")
        allocations: tuple[tuple[MachineRole, LegacyRoleAllocation | None], ...] = (
            ("task_machine", self.actor),
            ("verifier_machine", self.oracle_helper),
            ("verifier_machine", self.verifier),
        )
        for role, allocation in allocations:
            if allocation is None:
                continue
            for name, value in self.source_runtime.resource_limits(role).items():
                if getattr(allocation.execution_requirements, name) != value:
                    raise ValueError(f"Legacy allocation differs from original {role}.{name}")
            network = self.source_runtime.network(role)
            if network is not None and allocation.network != network:
                raise ValueError(f"Legacy allocation differs from original {role}.network")
            original = self.source_runtime.runtime.get(role) or {}
            if original.get("gpus") not in (None, allocation.gpus):
                raise ValueError(f"Legacy allocation differs from original {role}.gpus")
        return self
