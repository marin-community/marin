# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact-input evidence for admitting a nonempty population with executable graders."""

from collections.abc import Mapping

from finestore.rl.curation import ActorCensusInput, SourceDeclaration
from finestore.rl.source_identity import SourceId
from pydantic import BaseModel, ConfigDict, Field, model_validator


class AdmissionCensusReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: SourceId
    input: ActorCensusInput
    population_path: str
    scanned_task_specs_sha256: str
    retained_tasks: int = Field(ge=0)
    runnable_grader_tasks: int = Field(ge=0)
    no_grader_tasks: int = Field(ge=0)
    grader_kinds: dict[str, int]
    missing_grader_reasons: dict[str, int]
    evidence_url: str

    @model_validator(mode="after")
    def _complete_counts(self) -> "AdmissionCensusReceipt":
        if (
            self.retained_tasks != self.input.measured_tasks
            or self.runnable_grader_tasks + self.no_grader_tasks != self.retained_tasks
            or sum(self.grader_kinds.values()) != self.retained_tasks
            or sum(self.missing_grader_reasons.values()) != self.no_grader_tasks
        ):
            raise ValueError("Admission census does not account for its complete input population")
        return self


def require_source_admission(
    source_id: str, binding: ActorCensusInput, retained_tasks: int, receipt: AdmissionCensusReceipt
) -> None:
    """Reject empty, ungraded, incomplete or differently pinned admission evidence."""
    if receipt.source_id != source_id or receipt.input != binding or receipt.retained_tasks != retained_tasks:
        raise ValueError(f"Admission census belongs to another current population: {source_id}")
    if retained_tasks <= 0:
        raise ValueError(f"Admitted source requires a nonempty retained population: {source_id}")
    if receipt.no_grader_tasks or receipt.runnable_grader_tasks != retained_tasks:
        raise ValueError(f"Admitted source has tasks without runnable graders: {source_id}: {receipt.no_grader_tasks}")


def require_declaration_admission(
    declaration: SourceDeclaration, receipts: Mapping[str, AdmissionCensusReceipt]
) -> None:
    """Require complete current-input evidence before publication, loading or selection."""
    receipt = receipts.get(declaration.id)
    binding = declaration.actor_census_input
    if receipt is None or binding is None:
        raise ValueError(f"Admission census is missing for the current source: {declaration.id}")
    require_source_admission(declaration.id, binding, declaration.within_source_dedup.unique_count, receipt)
