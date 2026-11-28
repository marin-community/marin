# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Separate unknown-source ingest from repair of pinned Atlas exclusions."""

from collections.abc import Mapping

from finestore.rl.status_taxonomy import ExclusionClaim, SourceStatus
from pydantic import BaseModel, ConfigDict, Field
from taskcompendium.identity import canonical_sha256

from experiments.post_training.task_curation.ingest_stages.ingest_config import GIT_REVISION, IngestConfig, IngestMode
from infra.marina.apps.rl_data_catalog.server.source_identity import canonical_source_id, canonical_source_map


class ExcludedSource(BaseModel):
    """Authoritative starting declaration, carried into the worker's provenance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str
    dataset_id: str
    dataset_revision: str = Field(pattern=GIT_REVISION)
    exclusion_claims: tuple[ExclusionClaim, ...] = Field(min_length=1)
    declaration_sha256: str


def eligible_source(policy: IngestConfig, declarations: Mapping[str, dict]) -> ExcludedSource:
    """Reject unknown, unpinned, non-Excluded or reasonless repair inputs.

    The config cannot create a declaration or supply its own status or Reasons.
    Dataset identity must agree with the existing declaration's immutable pin.
    """
    source_id = canonical_source_id(policy.source_id)
    current = canonical_source_map(declarations)
    declaration = current.get(source_id)
    if declaration is None:
        raise ValueError("atlas-repair requires a source already declared in Atlas")
    if canonical_source_id(declaration["id"]) != source_id:
        raise ValueError("Atlas declaration identity differs from its source key")
    if declaration["status"] != SourceStatus.EXCLUDED:
        raise ValueError("atlas-repair requires an Excluded Atlas source")
    starting = ExcludedSource(
        source_id=source_id,
        dataset_id=declaration["dataset_id"],
        dataset_revision=declaration["dataset_revision"],
        exclusion_claims=declaration.get("exclusion_claims", ()),
        declaration_sha256=canonical_sha256(declaration),
    )
    if (starting.dataset_id, starting.dataset_revision) != (policy.upstream.repository, policy.upstream.revision):
        raise ValueError("ingest config differs from the existing Atlas dataset pin")
    return starting


def require_starting_state(policy: IngestConfig, declarations: Mapping[str, dict]) -> ExcludedSource | None:
    """Ingest accepts unknown sources; repair accepts only pinned exclusions."""
    if policy.mode is IngestMode.REPAIR:
        return eligible_source(policy, declarations)
    if canonical_source_id(policy.source_id) in canonical_source_map(declarations):
        raise ValueError("atlas-ingest requires an unknown source; use atlas-repair for an Excluded source")
    return None
