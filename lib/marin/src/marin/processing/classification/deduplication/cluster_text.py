# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Artifact contract for fuzzy-duplicate text grouped by cluster."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from rigging.filesystem.storage_path import StoragePath, prefix_join

CLUSTER_TEXT_MANIFEST_FILENAME = "manifest.json"
CLUSTER_TEXT_MANIFEST_VERSION = "v1"
CLUSTER_TEXT_SUBDIRECTORY = "text"
CLUSTER_TEXT_SUCCESS_FILENAME = "_SUCCESS"
MAXIMUM_VERIFICATION_TEXT_CHARS = 8 * 1024 * 1024


class ClusterTextShard(BaseModel):
    """One normalized shard in a cluster-text artifact."""

    model_config = ConfigDict(frozen=True)

    file_idx: int = Field(ge=0)
    source_key: str
    source_tag: str
    basename: str


class ClusterTextManifest(BaseModel):
    """Inputs, split parameters, and shard layout for grouped cluster text."""

    model_config = ConfigDict(frozen=True)

    version: str = CLUSTER_TEXT_MANIFEST_VERSION
    candidates: str
    max_cluster_size: int = Field(ge=1)
    output_shards: int = Field(ge=1)
    groups_per_shard: int = Field(ge=1)
    split_ngram_size: int = Field(ge=1)
    split_strategy: Literal["minhash_jaccard_v1"] = "minhash_jaccard_v1"
    oversized_clusters: dict[str, int]
    oversized_cluster_members: int = Field(ge=0)
    shards: list[ClusterTextShard]

    @model_validator(mode="after")
    def _valid_layout(self) -> "ClusterTextManifest":
        file_indices = [shard.file_idx for shard in self.shards]
        if file_indices != list(range(len(self.shards))):
            raise ValueError("cluster-text file indices must be contiguous and start at zero")
        shard_keys = [(shard.source_key, shard.basename) for shard in self.shards]
        if len(shard_keys) != len(set(shard_keys)):
            raise ValueError("cluster-text manifest contains a duplicate source shard")
        source_tags: dict[str, str] = {}
        tag_sources: dict[str, str] = {}
        for shard in self.shards:
            if not _is_path_component(shard.source_tag) or not _is_path_component(shard.basename):
                raise ValueError("cluster-text source tags and basenames must be single path components")
            if source_tags.setdefault(shard.source_key, shard.source_tag) != shard.source_tag:
                raise ValueError(f"cluster-text source {shard.source_key!r} has more than one source tag")
            if tag_sources.setdefault(shard.source_tag, shard.source_key) != shard.source_key:
                raise ValueError(f"cluster-text source tag {shard.source_tag!r} names more than one source")
        if any(splits < 2 for splits in self.oversized_clusters.values()):
            raise ValueError("cluster-text oversized cluster split counts must be at least two")
        return self


def _is_path_component(value: str) -> bool:
    return bool(value) and value not in {".", ".."} and "/" not in value and "\\" not in value


def resolve_data_path(prefix: str, path: str) -> str:
    """Resolve a stored relative path against an explicit data prefix."""
    storage_path = StoragePath(path)
    if storage_path.scheme or path.startswith("/"):
        return str(storage_path)
    return prefix_join(prefix, path)


def read_cluster_text_manifest(cluster_text: str) -> ClusterTextManifest:
    """Read the manifest from a cluster-text artifact."""
    path = StoragePath(prefix_join(cluster_text, CLUSTER_TEXT_MANIFEST_FILENAME))
    return ClusterTextManifest.model_validate_json(path.read_bytes())


def write_cluster_text_manifest(cluster_text: str, manifest: ClusterTextManifest) -> None:
    """Write the manifest to a cluster-text artifact."""
    path = StoragePath(prefix_join(cluster_text, CLUSTER_TEXT_MANIFEST_FILENAME))
    path.parent.mkdirs()
    path.write_text(manifest.model_dump_json(indent=2) + "\n")


def write_cluster_text_success(cluster_text: str) -> None:
    """Mark a cluster-text artifact complete after all data and metadata exist."""
    path = StoragePath(prefix_join(cluster_text, CLUSTER_TEXT_SUCCESS_FILENAME))
    path.write_text(CLUSTER_TEXT_MANIFEST_VERSION + "\n")
