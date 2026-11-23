# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed release manifest and per-stage receipt artifacts (DESIGN.md §3).

Every release-identifying value — release id, prior/target load IDs, snapshot names,
plan hashes — lives on these typed models and flows through the graph as artifact
fields. Stage worker source never carries such literals (``guard.py`` enforces it).
"""

from marin.execution.artifact import Artifact
from pydantic import BaseModel


class ArtifactRef(BaseModel):
    """A pointer to a produced artifact: location plus content hash."""

    uri: str
    sha256: str


class PriorPin(BaseModel):
    """The prior release a new release builds on."""

    release_id: str
    load_id: str
    snapshot_uri: str
    snapshot_sha256: str
    catalog_uri: str


class CensusRef(BaseModel):
    """Census receipts with the population totals that drive sizing."""

    receipt: ArtifactRef
    population_total: int
    per_source: dict[str, int]


class PlanRef(BaseModel):
    """One named plan artifact (corpus, curriculum compose, QC browse)."""

    uri: str
    sha256: str


class IdentityRef(BaseModel):
    """Bound identity survey."""

    receipt: ArtifactRef
    survey_count: int


class BackendRef(BaseModel):
    """Backend validation with the parity receipt that decides GPU vs CPU embedding."""

    runtime: str
    parity_receipt: ArtifactRef | None
    validation_receipt: ArtifactRef
    gpu_eligible: bool


class SnapshotRef(BaseModel):
    """The published admissions assembly snapshot."""

    name: str
    receipt: ArtifactRef


class LoadRef(BaseModel):
    """One staged load (catalog or curriculum) with its typed load ID."""

    load_id: str
    receipt: ArtifactRef


class GuardReceipt(BaseModel):
    """The submission guard's record for one release."""

    repo_sha: str
    landed_sha: str
    checked_at: str


class ReceiptRef(BaseModel):
    """A per-stage receipt artifact under the release prefix."""

    stage: str
    artifact: ArtifactRef


class ReleaseManifest(Artifact):
    """The typed release manifest consumed by every stage (DESIGN.md §3)."""

    release_id: str
    prior: PriorPin
    declarations: ArtifactRef
    census: CensusRef
    plans: dict[str, PlanRef]
    identity: IdentityRef
    backend: BackendRef
    snapshot: SnapshotRef
    loads: dict[str, LoadRef]
    guard: GuardReceipt
    stage_receipts: dict[str, ReceiptRef]


class HistoryStagingReceipt(Artifact):
    """Output of the history-staging stage (DESIGN.md §2)."""

    kind: str
    release_id: str
    prior_load_id: str
    target_load_id: str
    snapshot_id: str
    source_revision: str
    prior_source_count: int
    copied: dict[str, int]
    workers: int
    item_batch_rows: int
    receipt_uri: str
    completed_at: str
