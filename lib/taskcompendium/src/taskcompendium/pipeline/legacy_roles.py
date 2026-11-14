# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded allocation qualification for completed trials from the frozen producer."""

from typing import Literal, Protocol

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


class LegacyAllocationContract(Protocol):
    """Frozen packet and role observations shared by the producer-specific receipts."""

    @property
    def source_task_sha256(self) -> str: ...

    @property
    def source_task_checksum(self) -> str: ...

    @property
    def consumed_task_sha256(self) -> str: ...

    @property
    def consumed_task_checksum(self) -> str: ...

    @property
    def source_runtime(self) -> TaskSpecRuntimeBinding: ...

    @property
    def actor(self) -> LegacyRoleAllocation | None: ...

    @property
    def oracle_helper(self) -> LegacyRoleAllocation | None: ...

    @property
    def verifier(self) -> LegacyRoleAllocation: ...


def require_legacy_allocation_contract(qualification: LegacyAllocationContract) -> None:
    """Require the unchanged packet and allocations consistent with its frozen runtime."""
    if (
        qualification.source_runtime.task_sha256 != qualification.source_task_sha256
        or qualification.consumed_task_sha256 != qualification.source_task_sha256
        or qualification.consumed_task_checksum != qualification.source_task_checksum
    ):
        raise ValueError("Legacy qualification requires the unchanged original packet")
    if (qualification.actor is None) == (qualification.oracle_helper is None):
        raise ValueError("Legacy qualification requires the actual actor or its actorless oracle helper")
    allocations: tuple[tuple[MachineRole, LegacyRoleAllocation | None], ...] = (
        ("task_machine", qualification.actor),
        ("verifier_machine", qualification.oracle_helper),
        ("verifier_machine", qualification.verifier),
    )
    for role, allocation in allocations:
        if allocation is None:
            continue
        for name, value in qualification.source_runtime.resource_limits(role).items():
            if getattr(allocation.execution_requirements, name) != value:
                raise ValueError(f"Legacy allocation differs from original {role}.{name}")
        network = qualification.source_runtime.network(role)
        if network is not None and allocation.network != network:
            raise ValueError(f"Legacy allocation differs from original {role}.network")
        original = qualification.source_runtime.runtime.get(role) or {}
        if original.get("gpus") not in (None, allocation.gpus):
            raise ValueError(f"Legacy allocation differs from original {role}.gpus")


class LegacyRoleAdoptionDecision(BaseModel):
    """The exact frozen partition and audit authorized by an adoption ruling."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    version: Literal[1]
    disposition: Literal["adopted"]
    producer_revision: Literal["2c5eea40f264158596f62e593dd21ecaeaa4dc52"]
    input_partition_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ruling: ContentEvidenceReference


class LegacySmallRoleRuling(BaseModel):
    """Preserve the local ruling anchor in the original small-partition decision."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class LegacySmallRoleAdoptionDecision(BaseModel):
    """Adopt completed passes while retaining the partition's scored negatives."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    version: Literal[1]
    disposition: Literal["adopted"]
    producer_revision: Literal["2c5eea40f264158596f62e593dd21ecaeaa4dc52"]
    input_index_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_partition_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    adopted_count: int = Field(gt=0)
    excluded_scored_failures: int = Field(ge=0)
    ruling: LegacySmallRoleRuling


class LegacySmallRoleScope(BaseModel):
    """Published evidence for the small-partition decision's exact anchors."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    audit_manifest: ContentEvidenceReference
    ruling: ContentEvidenceReference


def require_legacy_adoption_scope(
    decision: LegacyRoleAdoptionDecision | LegacySmallRoleAdoptionDecision,
    input_partition: ContentEvidenceReference,
    audit: ContentEvidenceReference,
    small_scope: LegacySmallRoleScope | None,
) -> None:
    """Bind each authority format to its original index, audit and ruling hashes."""
    if isinstance(decision, LegacySmallRoleAdoptionDecision):
        matches = (
            small_scope is not None
            and decision.input_index_sha256 == input_partition.sha256
            and decision.input_partition_sha256 == audit.sha256
            and decision.audit_sha256 == small_scope.audit_manifest.sha256
            and decision.ruling.sha256 == small_scope.ruling.sha256
        )
    else:
        matches = (
            small_scope is None
            and decision.input_partition_sha256 == input_partition.sha256
            and decision.audit_sha256 == audit.sha256
        )
    if not matches:
        raise ValueError("Legacy adoption decision scope differs from its frozen partition or audit")


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
    adoption_decision: LegacyRoleAdoptionDecision | LegacySmallRoleAdoptionDecision
    small_scope: LegacySmallRoleScope | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def validate_frozen_contract(self):
        require_legacy_adoption_scope(self.adoption_decision, self.input_partition, self.audit, self.small_scope)
        if self.adoption_decision.producer_revision != self.producer_revision:
            raise ValueError("Legacy adoption decision scope differs from its frozen partition or audit")
        require_legacy_allocation_contract(self)
        return self
