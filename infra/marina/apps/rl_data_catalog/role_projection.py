# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify a metadata-only curriculum role correction against frozen releases and current policy."""

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

from marina.client import AppletQuery
from pydantic import BaseModel, ConfigDict
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.identity import canonical_sha256

from experiments.post_training.task_curation.embedding import COVERAGE_VERSION
from experiments.post_training.task_curation.source_identity import SourceId, canonical_source_id, original_source_id
from infra.marina.apps.rl_data_catalog.server.quality import intended_use


class ManifestReference(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    uri: str
    sha256: str


class CurriculumManifests(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    release_uri: str
    release_sha256: str
    release_id: str
    coverage_uri: str
    coverage_sha256: str
    coverage_id: str


class PolicyRuling(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    session_id: str
    message_id: str
    timestamp: str


class RoleChange(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    source_id: SourceId
    source_key: str
    population: Literal["converted", "admitted"]
    task_count: int
    dataset_id: str
    dataset_revision: str
    recorded_use: Literal["train"]
    effective_use: Literal["eval"]


class VectorCache(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    manifest_uri: str
    manifest_sha256: str
    cache_hits: int
    cache_misses: Literal[0]


class RoleProjection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal[1]
    kind: Literal["source-use-correction"]
    policy_ruling: PolicyRuling
    raw: CurriculumManifests
    corrected: CurriculumManifests
    changes: list[RoleChange]
    vector_cache: VectorCache


def manifest_bytes(reference: ManifestReference) -> bytes:
    payload = StoragePath(reference.uri).read_bytes()
    if hashlib.sha256(payload).hexdigest() != reference.sha256:
        raise ValueError(f"Role-projection artifact hash differs: {reference.uri}")
    return payload


@dataclass(frozen=True)
class ManifestDocuments:
    release: dict[str, Any]
    coverage: dict[str, Any]


def manifest_documents(reference: CurriculumManifests) -> ManifestDocuments:
    release = json.loads(manifest_bytes(ManifestReference(uri=reference.release_uri, sha256=reference.release_sha256)))
    coverage = json.loads(
        manifest_bytes(ManifestReference(uri=reference.coverage_uri, sha256=reference.coverage_sha256))
    )
    if (
        release["release_id"] != reference.release_id
        or coverage["coverage_id"] != reference.coverage_id
        or coverage["alignment_release_id"] != reference.release_id
        or canonical_sha256({key: release[key] for key in ("lineage", "sources", "inventory")}) != reference.release_id
        or canonical_sha256(
            {
                **{
                    key: coverage[key]
                    for key in (
                        "alignment_release_id",
                        "subject_distance",
                        "capability_distance",
                        "source_sample_size",
                        "area_sample_size",
                    )
                },
                "version": COVERAGE_VERSION,
            }
        )
        != reference.coverage_id
    ):
        raise ValueError("Role-projection release or coverage identity differs")
    return ManifestDocuments(release, coverage)


def validated_role_projection(
    reference: ManifestReference, release_uri: str, coverage_uri: str, query: AppletQuery
) -> RoleProjection:
    """Require exact metadata changes and current catalog authority before accepting historical roles."""
    receipt = RoleProjection.model_validate_json(manifest_bytes(reference))
    manifest_bytes(ManifestReference(uri=receipt.vector_cache.manifest_uri, sha256=receipt.vector_cache.manifest_sha256))
    if (receipt.corrected.release_uri, receipt.corrected.coverage_uri) != (release_uri, coverage_uri):
        raise ValueError("Role-projection receipt targets different loader inputs")
    raw = manifest_documents(receipt.raw)
    corrected = manifest_documents(receipt.corrected)
    raw_release, raw_coverage = raw.release, raw.coverage
    corrected_release, corrected_coverage = corrected.release, corrected.coverage
    changes = {change.source_id: change for change in receipt.changes}
    if not changes or len(changes) != len(receipt.changes):
        raise ValueError("Role-projection changes must identify distinct sources")
    sources = {canonical_source_id(row["source_id"]): row for row in raw_release["sources"]}
    inventory = {canonical_source_id(row["id"]): row for row in raw_release["inventory"]}
    for source_id, change in changes.items():
        source = sources.get(source_id)
        declaration = inventory.get(source_id)
        if (
            source is None
            or declaration is None
            or source["basis"] != "matched"
            or (source["source_key"], source["population"], source["task_count"])
            != (change.source_key, change.population, change.task_count)
            or source["intended_use"] != change.recorded_use
            or (declaration["dataset_id"], declaration["dataset_revision"])
            != (change.dataset_id, change.dataset_revision)
        ):
            raise ValueError(f"Role-projection source lineage differs: {source_id}")
        rows = query(
            "SELECT s.declaration FROM atlas_sources s JOIN atlas_current c USING(load_id) "
            "WHERE c.name='latest' AND s.id IN (:source_id, :original_source_id)",
            {"source_id": source_id, "original_source_id": original_source_id(source_id)},
        )["rows"]
        if len(rows) != 1:
            raise ValueError(f"Role projection requires one current catalog source: {source_id}")
        current = rows[0]["declaration"]
        if (current["dataset_id"], current["dataset_revision"], intended_use(current)) != (
            change.dataset_id,
            change.dataset_revision,
            change.effective_use,
        ):
            raise ValueError(f"Role projection differs from current catalog policy or pin: {source_id}")
    expected_release = {
        **raw_release,
        "release_id": corrected_release["release_id"],
        "sources": [
            (
                {**source, "intended_use": changes[canonical_source_id(source["source_id"])].effective_use}
                if canonical_source_id(source["source_id"]) in changes
                else source
            )
            for source in raw_release["sources"]
        ],
    }
    expected_coverage = {
        **raw_coverage,
        "coverage_id": corrected_coverage["coverage_id"],
        "alignment_release_id": corrected_release["release_id"],
    }
    if canonical_sha256(expected_release) != canonical_sha256(corrected_release) or canonical_sha256(
        expected_coverage
    ) != canonical_sha256(corrected_coverage):
        raise ValueError("Role projection changes frozen data or metadata beyond the approved source roles")
    return receipt
