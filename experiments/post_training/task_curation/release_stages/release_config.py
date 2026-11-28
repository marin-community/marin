# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The release config: the tiny reviewed document a human writes for one release.

The config carries only true policy choices — the release id, the cluster, the
explicit freeze revision, a pointer to the prior release's state, and the
deliberate removals and QC exclusions with their reasons. Everything else in the
release plan is derived or measured by :mod:`plan_derivation` from authoritative
state (the declarations layer at the freeze revision, the census registry, the
prior catalog and the applet SQL), so no count or identity is ever typed by a
person.

The config is reviewed configuration like the old release plan document, but
diffable in ~20 lines: a release is a policy statement, not a data dump.
"""

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, Field

from experiments.post_training.task_curation.release_stages.main_sources import MainSourceSyncPlan
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef

RELEASE_CONFIG_KIND = "atlas-release-config"


class PriorRelease(BaseModel):
    """The prior release's sealed state this release builds on.

    ``release_id`` names the prior release; every pin points at one of its
    prior-state documents that has no catalog-derivable home (the served
    declarations export, the identity/backend surveys the compose retains rows
    from, the composed curriculum release and coverage, the package artifacts).
    The derivation digest-verifies every pin it consumes; URIs are pointers, the
    digests arrive with them and are never re-typed. ``source_counts`` is the
    prior load's counts document, carried only for its ruling citations — the
    package's oracle inventory and source counts derive from the release's own
    sealed census, never from the prior release.
    """

    release_id: str
    declarations: ArtifactRef | None = None
    identity: ArtifactRef | None = None
    backend: ArtifactRef | None = None
    composed_release: ArtifactRef | None = None
    composed_coverage: ArtifactRef | None = None
    source_counts: ArtifactRef | None = None

    def validate_shape(self) -> None:
        if not self.release_id:
            raise ValueError("prior_release.release_id is required")
        for name in (
            "declarations",
            "identity",
            "backend",
            "composed_release",
            "composed_coverage",
            "source_counts",
        ):
            ref = getattr(self, name)
            if ref is not None and (not ref.uri or not ref.sha256):
                raise ValueError(f"prior_release.{name} must carry a uri and a sha256 when set")


class RemovedSource(BaseModel):
    """One deliberate removal, with the reason a reviewer reads."""

    source_id: str
    reason: str = Field(min_length=1)


class ExcludedSource(BaseModel):
    """One deliberate QC exclusion (its prior panel is not copied), with the reason."""

    source_id: str
    reason: str = Field(min_length=1)


class ReleaseConfig(BaseModel):
    """The typed release config: the only human input to the derived plan."""

    kind: str = RELEASE_CONFIG_KIND
    release_id: str
    cluster: str
    freeze_source_revision: str
    prior_release: PriorRelease
    prior_snapshot_id: str
    """The snapshot id the prior release served. When the catalog's current snapshot
    differs, the derivation re-cuts this release: the retained prior state pins come
    from what is SERVED, not from the config's prior-release documents."""
    removed_sources: tuple[RemovedSource, ...] = ()
    qc_exclusions: tuple[ExcludedSource, ...] = ()
    production_version: int | None = None
    """The production deployment version the hosted checks verify against (deployment state)."""
    preview_base_version: int | None = None
    """The preview revision the release publishes on top of (deployment state)."""
    gpu_runtime: str | None = None
    parity_receipt: ArtifactRef | None = None
    exposure_input_check: ArtifactRef | None = None
    main_source_sync: MainSourceSyncPlan | None = None
    migrations: tuple[str, ...] = ()
    rebound_source: str = ""
    rebound_tasks: int = 0
    strict_audit_local: bool = False
    package_output_root: str = ""

    def validate_shape(self) -> None:
        """Refuse a config that cannot drive the derivation."""
        if self.kind != RELEASE_CONFIG_KIND:
            raise ValueError(f"release config kind is {self.kind!r}, expected {RELEASE_CONFIG_KIND!r}")
        if not self.release_id or not self.cluster:
            raise ValueError("release_id and cluster are required")
        if len(self.freeze_source_revision) != 40 or not all(
            c in "0123456789abcdef" for c in self.freeze_source_revision
        ):
            raise ValueError("freeze_source_revision must be a 40-hex git revision")
        self.prior_release.validate_shape()
        ids = [row.source_id for row in self.removed_sources]
        if len(ids) != len(set(ids)):
            raise ValueError("removed_sources repeats a source id")
        if "" in ids or any(not row.reason.strip() for row in self.removed_sources):
            raise ValueError("removed_sources entries need a source id and a non-empty reason")
        excluded = [row.source_id for row in self.qc_exclusions]
        if len(excluded) != len(set(excluded)):
            raise ValueError("qc_exclusions repeats a source id")
        if any(not row.reason.strip() for row in self.qc_exclusions):
            raise ValueError("qc_exclusions entries need a non-empty reason")
        if self.rebound_tasks < 0 or (bool(self.rebound_source) != (self.rebound_tasks > 0)):
            raise ValueError("rebound pins must be empty (no rebound) or a source with a positive task count")
        if self.main_source_sync is not None and self.main_source_sync.queue_path == "":
            raise ValueError("main_source_sync.queue_path must be a path or unset, not an empty string")
        for name in ("parity_receipt", "exposure_input_check"):
            ref = getattr(self, name)
            if ref is not None and (not ref.uri or not ref.sha256):
                raise ValueError(f"{name} must carry a uri and a sha256 when set")


def load_release_config(path: Path) -> ReleaseConfig:
    """Load and validate the reviewed release config document."""
    config = ReleaseConfig.model_validate_json(path.read_text(encoding="utf-8"))
    config.validate_shape()
    return config


def config_document(config: ReleaseConfig) -> dict:
    """The canonical config document bytes' source (unset opt-ins absent)."""
    data = config.model_dump(mode="json")
    if config.main_source_sync is None:
        data.pop("main_source_sync", None)
    for name in ("parity_receipt", "exposure_input_check"):
        if getattr(config, name) is None:
            data.pop(name, None)
    if config.gpu_runtime is None:
        data.pop("gpu_runtime", None)
    if config.production_version is None:
        data.pop("production_version", None)
    if config.preview_base_version is None:
        data.pop("preview_base_version", None)
    return data


def config_fingerprint(config: ReleaseConfig) -> str:
    """The digest of the canonical config document; the derivation receipt binds it."""
    document = json.dumps(config_document(config), sort_keys=True)
    return hashlib.sha256(document.encode()).hexdigest()
