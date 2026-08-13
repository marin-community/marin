# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Manifest snapshots and compare-and-swap commit publication."""

from __future__ import annotations

import hashlib
import threading
import uuid
from dataclasses import dataclass, field

from rigging.filesystem import StoragePath
from rigging.filesystem.conditional_object import ConditionalWriteError, conditional_object

from finestore.layout import (
    FORMAT_VERSION,
    ArchiveMetadata,
    CommitToken,
    FineStoreLayout,
    HeadMetadata,
    Manifest,
    ManifestTable,
    SealMarker,
    Shard,
    TableMetadata,
)

_MAX_COMMIT_ATTEMPTS = 32
_ROOT_COMMIT_ID = "root"


class CommitConflict(RuntimeError):
    """A manifest delta can no longer be applied to the current snapshot."""


@dataclass(frozen=True)
class ArchiveSnapshot:
    """A manifest and the storage version of the HEAD object that selected it."""

    manifest: Manifest
    token: CommitToken | None


@dataclass(frozen=True)
class TableAddition:
    """Immutable shards and table metadata added by one transaction."""

    metadata_path: str
    shards: tuple[Shard, ...]


@dataclass(frozen=True)
class TableReplacement:
    """A logical compaction replacing exact input paths with output shards."""

    input_paths: frozenset[str]
    output_shards: tuple[Shard, ...]


@dataclass(frozen=True)
class ClearSeal:
    """Remove the current seal marker as part of a commit."""


@dataclass(frozen=True)
class CommitDelta:
    """The changes one manifest commit applies atomically."""

    additions: dict[str, TableAddition] = field(default_factory=dict)
    replacements: dict[str, TableReplacement] = field(default_factory=dict)
    metadata_updates: dict[str, str] = field(default_factory=dict)
    removals: frozenset[str] = frozenset()
    seal_update: SealMarker | ClearSeal | None = None


def _empty_manifest() -> Manifest:
    return Manifest(commit_id=_ROOT_COMMIT_ID, sequence=0)


def initialize_archive(layout: FineStoreLayout) -> ArchiveMetadata:
    """Create the immutable archive marker, or validate the marker already present."""
    marker = conditional_object(layout.archive_path)
    existing = marker.read()
    expected = ArchiveMetadata()
    if existing is None:
        try:
            marker.write(expected.model_dump_json(indent=2).encode(), expected_version=None)
            return expected
        except ConditionalWriteError:
            existing = marker.read()
            assert existing is not None
    found = ArchiveMetadata.model_validate_json(existing.data)
    if found.format_version != FORMAT_VERSION:
        raise ValueError(
            f"archive at {layout.root} is FineStore format v{found.format_version}; expected v{FORMAT_VERSION}"
        )
    return found


def validate_archive(layout: FineStoreLayout) -> ArchiveMetadata | None:
    """Read and validate the archive marker, returning ``None`` when it is absent."""
    try:
        data = StoragePath(layout.archive_path).read_bytes()
    except FileNotFoundError:
        return None
    found = ArchiveMetadata.model_validate_json(data)
    if found.format_version != FORMAT_VERSION:
        raise ValueError(
            f"archive at {layout.root} is FineStore format v{found.format_version}; expected v{FORMAT_VERSION}"
        )
    return found


def write_schema(layout: FineStoreLayout, metadata: TableMetadata) -> str:
    """Publish immutable logical table metadata and return its content-addressed path."""
    data = metadata.model_dump_json(indent=2).encode()
    schema_id = hashlib.blake2b(data, digest_size=16).hexdigest()
    path = layout.schema_path(schema_id)
    obj = conditional_object(path)
    existing = obj.read()
    if existing is None:
        try:
            obj.write(data, expected_version=None)
            return path
        except ConditionalWriteError:
            existing = obj.read()
            assert existing is not None
    if existing.data != data:
        raise ValueError(f"schema digest collision at {path}")
    return path


def read_snapshot(layout: FineStoreLayout) -> ArchiveSnapshot:
    """Pin and return the manifest selected by one HEAD read."""
    head_object = conditional_object(layout.head_path)
    versioned = head_object.read()
    if versioned is None:
        return ArchiveSnapshot(manifest=_empty_manifest(), token=None)
    head = HeadMetadata.model_validate_json(versioned.data)
    if head.format_version != FORMAT_VERSION:
        raise ValueError(f"HEAD at {layout.root} is FineStore format v{head.format_version}; expected v{FORMAT_VERSION}")
    manifest = Manifest.model_validate_json(StoragePath(head.manifest_path).read_bytes())
    if manifest.format_version != FORMAT_VERSION:
        raise ValueError(
            f"manifest {head.manifest_path} is FineStore format v{manifest.format_version}; expected v{FORMAT_VERSION}"
        )
    if (manifest.commit_id, manifest.sequence) != (head.commit_id, head.sequence):
        raise ValueError(f"HEAD at {layout.root} does not match manifest {head.manifest_path}")
    token = CommitToken(
        commit_id=head.commit_id,
        sequence=head.sequence,
        version=versioned.version,
        manifest_path=head.manifest_path,
    )
    return ArchiveSnapshot(manifest=manifest, token=token)


def _apply_delta(base: Manifest, delta: CommitDelta) -> Manifest:
    sequence = base.sequence + 1
    tables = dict(base.tables)
    for name, addition in delta.additions.items():
        current = tables.get(name)
        if current is not None and current.metadata_path != addition.metadata_path:
            raise CommitConflict(f"table {name!r} was registered with different metadata")
        prior = current.shards if current is not None else ()
        paths = {shard.path for shard in prior}
        added = tuple(
            shard.model_copy(update={"commit_sequence": sequence})
            for shard in addition.shards
            if shard.path not in paths
        )
        tables[name] = ManifestTable(metadata_path=addition.metadata_path, shards=prior + added)

    for name, replacement in delta.replacements.items():
        current = tables.get(name)
        if current is None:
            raise CommitConflict(f"table {name!r} no longer exists")
        active_paths = {shard.path for shard in current.shards}
        if not replacement.input_paths.issubset(active_paths):
            raise CommitConflict(f"compaction inputs for table {name!r} are no longer active")
        survivors = tuple(shard for shard in current.shards if shard.path not in replacement.input_paths)
        outputs = tuple(shard.model_copy(update={"commit_sequence": sequence}) for shard in replacement.output_shards)
        tables[name] = ManifestTable(metadata_path=current.metadata_path, shards=survivors + outputs)

    for name, metadata_path in delta.metadata_updates.items():
        current = tables.get(name)
        if current is None:
            raise CommitConflict(f"table {name!r} no longer exists")
        tables[name] = ManifestTable(metadata_path=metadata_path, shards=current.shards)

    for name in delta.removals:
        tables.pop(name, None)

    sealed = base.sealed
    if isinstance(delta.seal_update, ClearSeal):
        sealed = None
    elif delta.seal_update is not None:
        sealed = delta.seal_update
    return Manifest(
        commit_id=uuid.uuid4().hex,
        parent_commit_id=None if base.commit_id == _ROOT_COMMIT_ID else base.commit_id,
        sequence=sequence,
        tables=tables,
        sealed=sealed,
    )


class CommitCoordinator:
    """Serialize local commits and rebase manifest deltas after remote HEAD races."""

    def __init__(self, layout: FineStoreLayout) -> None:
        self._layout = layout
        self._head = conditional_object(layout.head_path)
        self._lock = threading.Lock()

    def snapshot(self) -> ArchiveSnapshot:
        return read_snapshot(self._layout)

    def commit(self, delta: CommitDelta, *, base: ArchiveSnapshot | None = None) -> CommitToken:
        """Publish ``delta`` and return the new durable commit token."""
        with self._lock:
            current = base or read_snapshot(self._layout)
            for _attempt in range(_MAX_COMMIT_ATTEMPTS):
                manifest = _apply_delta(current.manifest, delta)
                manifest_path = self._layout.manifest_path(manifest.commit_id)
                StoragePath(manifest_path).write_text(manifest.model_dump_json(indent=2))
                head = HeadMetadata(
                    commit_id=manifest.commit_id,
                    sequence=manifest.sequence,
                    manifest_path=manifest_path,
                )
                expected_version = current.token.version if current.token is not None else None
                try:
                    version = self._head.write(head.model_dump_json().encode(), expected_version=expected_version)
                except ConditionalWriteError:
                    current = read_snapshot(self._layout)
                    continue
                return CommitToken(
                    commit_id=manifest.commit_id,
                    sequence=manifest.sequence,
                    version=version,
                    manifest_path=manifest_path,
                )
        raise CommitConflict(f"HEAD at {self._layout.root} changed {_MAX_COMMIT_ATTEMPTS} consecutive times")
