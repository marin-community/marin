# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Manifest snapshots and compare-and-swap commit publication."""

from __future__ import annotations

import hashlib
import threading
import uuid
from dataclasses import dataclass, field

from rigging import telemetry
from rigging.filesystem.conditional_object import ConditionalWriteError, conditional_object
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import ExponentialBackoff, retry_with_backoff

from finestore.layout import (
    CACHE_HISTORY_FLOOR_FEATURE,
    FORMAT_VERSION,
    SUPPORTED_MANIFEST_FEATURES,
    ArchiveMetadata,
    CommitToken,
    FineStoreLayout,
    FormatVersionError,
    HeadMetadata,
    Manifest,
    ManifestTable,
    SealMarker,
    Shard,
    TableMetadata,
)

# A lost compare-and-swap means another writer published, so the retry delay only needs to
# spread concurrent writers apart, not wait out an outage. Shorter delays raise the request
# count faster than they lower latency: in the simulator, 50 ms tripled requests at 64 writers.
# The retry budget is wall time, so many writers can share one HEAD without a flush failing.
_COMMIT_BACKOFF_INITIAL = 0.5
_COMMIT_BACKOFF_MAXIMUM = 5.0
_RETRY_BACKOFF_FACTOR = 2.0
_RETRY_BACKOFF_JITTER = 0.25
_COMMIT_RETRY_BUDGET = 60 * 60.0
_MAX_COMMIT_ATTEMPTS = 100_000
_SCHEMA_RETRY_BUDGET = 60.0
_MAX_SCHEMA_ATTEMPTS = 1000
_ROOT_COMMIT_ID = "root"

_COMMITS = telemetry.counter("finestore_commits", unit="{commit}")
_COMMIT_CONFLICTS = telemetry.counter("finestore_commit_conflicts", unit="{conflict}")


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
    required_features: frozenset[str] = frozenset()
    seal_update: SealMarker | ClearSeal | None = None
    # ``expected_epoch`` rejects the commit, without rebasing, once another maintainer has
    # taken over; ``epoch_update`` is how a new maintainer announces itself.
    expected_epoch: int | None = None
    epoch_update: int | None = None
    cache_history_floor: int = 0


def _empty_manifest() -> Manifest:
    return Manifest(commit_id=_ROOT_COMMIT_ID, sequence=0)


def initialize_archive(layout: FineStoreLayout) -> ArchiveMetadata:
    """Create the archive marker, or validate the marker already present."""
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
        raise FormatVersionError(layout.root, found=found.format_version)
    return found


def validate_archive(layout: FineStoreLayout) -> ArchiveMetadata | None:
    """Read and validate the archive marker, returning ``None`` when it is absent."""
    try:
        data = StoragePath(layout.archive_path).read_bytes()
    except FileNotFoundError:
        return None
    found = ArchiveMetadata.model_validate_json(data)
    if found.format_version != FORMAT_VERSION:
        raise FormatVersionError(layout.root, found=found.format_version)
    return found


def write_schema(layout: FineStoreLayout, metadata: TableMetadata) -> str:
    """Publish immutable logical table metadata and return its content-addressed path."""
    data = metadata.model_dump_json(indent=2).encode()
    schema_id = hashlib.blake2b(data, digest_size=16).hexdigest()
    path = layout.schema_path(schema_id)
    obj = conditional_object(path)

    def publish() -> str:
        existing = obj.read()
        if existing is None:
            try:
                obj.write(data, expected_version=None)
                return path
            except ConditionalWriteError:
                existing = obj.read()
                if existing is None:
                    raise
        if existing.data != data:
            raise ValueError(f"schema digest collision at {path}")
        return path

    return retry_with_backoff(
        publish,
        retryable=lambda exc: isinstance(exc, ConditionalWriteError),
        max_attempts=_MAX_SCHEMA_ATTEMPTS,
        max_elapsed=_SCHEMA_RETRY_BUDGET,
        backoff=ExponentialBackoff(initial=0.1, maximum=1.0, factor=_RETRY_BACKOFF_FACTOR, jitter=_RETRY_BACKOFF_JITTER),
        operation=f"FineStore schema publication at {path}",
    )


def read_snapshot(layout: FineStoreLayout) -> ArchiveSnapshot:
    """Pin and return the manifest selected by one HEAD read.

    A populated archive is validated from HEAD alone. Only an absent HEAD reads the format
    marker, which distinguishes an empty root from a format-v1 archive that has no HEAD.
    """
    head_object = conditional_object(layout.head_path)
    versioned = head_object.read()
    if versioned is None:
        validate_archive(layout)
        return ArchiveSnapshot(manifest=_empty_manifest(), token=None)
    head = HeadMetadata.model_validate_json(versioned.data)
    if head.format_version != FORMAT_VERSION:
        raise FormatVersionError(layout.root, found=head.format_version)
    manifest = Manifest.model_validate_json(StoragePath(head.manifest_path).read_bytes())
    if manifest.format_version != FORMAT_VERSION:
        raise ValueError(
            f"manifest {head.manifest_path} is FineStore format v{manifest.format_version}; expected v{FORMAT_VERSION}"
        )
    if (manifest.commit_id, manifest.sequence) != (head.commit_id, head.sequence):
        raise ValueError(f"HEAD at {layout.root} does not match manifest {head.manifest_path}")
    unsupported = set(manifest.required_features) - SUPPORTED_MANIFEST_FEATURES
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise ValueError(f"manifest {head.manifest_path} requires unsupported FineStore features: {names}")
    token = CommitToken(
        commit_id=head.commit_id,
        sequence=head.sequence,
        version=versioned.version,
        manifest_path=head.manifest_path,
    )
    return ArchiveSnapshot(manifest=manifest, token=token)


def _apply_delta(base: Manifest, delta: CommitDelta) -> Manifest:
    if delta.expected_epoch is not None and base.maintainer_epoch != delta.expected_epoch:
        raise CommitConflict(
            f"maintenance epoch is {base.maintainer_epoch}, commit was prepared under {delta.expected_epoch}"
        )
    if delta.epoch_update is not None and delta.epoch_update <= base.maintainer_epoch:
        raise CommitConflict(f"maintenance epoch {base.maintainer_epoch} already passed {delta.epoch_update}")
    sequence = base.sequence + 1
    tables = dict(base.tables)
    for name, addition in delta.additions.items():
        current = tables.get(name)
        if current is not None and current.metadata_path != addition.metadata_path:
            raise CommitConflict(f"table {name!r} was registered with different metadata")
        prior = current.shards if current is not None else ()
        paths = {shard.path for shard in prior}
        added = tuple(
            shard.model_copy(
                update={"commit_sequence": sequence, "min_row_commit": sequence, "max_row_commit": sequence}
            )
            for shard in addition.shards
            if shard.path not in paths
        )
        if current is None:
            tables[name] = ManifestTable(metadata_path=addition.metadata_path, shards=added)
        else:
            tables[name] = current.model_copy(update={"shards": prior + added})

    for name, replacement in delta.replacements.items():
        current = tables.get(name)
        if current is None:
            raise CommitConflict(f"table {name!r} no longer exists")
        active_paths = {shard.path for shard in current.shards}
        if not replacement.input_paths.issubset(active_paths):
            raise CommitConflict(f"compaction inputs for table {name!r} are no longer active")
        survivors = tuple(shard for shard in current.shards if shard.path not in replacement.input_paths)
        outputs = tuple(shard.model_copy(update={"commit_sequence": sequence}) for shard in replacement.output_shards)
        tables[name] = current.model_copy(update={"shards": survivors + outputs})

    for name, metadata_path in delta.metadata_updates.items():
        current = tables.get(name)
        if current is None:
            raise CommitConflict(f"table {name!r} no longer exists")
        tables[name] = current.model_copy(update={"metadata_path": metadata_path})

    for name in delta.removals:
        tables.pop(name, None)

    sealed = base.sealed
    if isinstance(delta.seal_update, ClearSeal):
        sealed = None
    elif delta.seal_update is not None:
        sealed = delta.seal_update
    history_floor = max(base.cache_history_floor, delta.cache_history_floor)
    required_features = set(base.required_features) | delta.required_features
    if history_floor:
        required_features.add(CACHE_HISTORY_FLOOR_FEATURE)
    return base.model_copy(
        update={
            "commit_id": uuid.uuid4().hex,
            "parent_commit_id": None if base.commit_id == _ROOT_COMMIT_ID else base.commit_id,
            "sequence": sequence,
            "required_features": tuple(sorted(required_features)),
            "cache_history_floor": history_floor,
            "tables": tables,
            "sealed": sealed,
            "maintainer_epoch": base.maintainer_epoch if delta.epoch_update is None else delta.epoch_update,
        }
    )


class CommitCoordinator:
    """Serialize local commits and rebase manifest deltas after remote HEAD races."""

    def __init__(self, layout: FineStoreLayout) -> None:
        self._layout = layout
        self._head = conditional_object(layout.head_path)
        self._lock = threading.Lock()
        self._published: ArchiveSnapshot | None = None

    def snapshot(self) -> ArchiveSnapshot:
        """Read the current HEAD and manifest."""
        return read_snapshot(self._layout)

    def last_snapshot(self) -> ArchiveSnapshot:
        """Return the snapshot this coordinator last published, reading HEAD only when it has none.

        The result is current for this writer's own commits and may lag other writers; a commit
        based on it rebases after a lost compare-and-swap exactly as a fresh read would.
        """
        with self._lock:
            if self._published is None:
                self._published = read_snapshot(self._layout)
            return self._published

    def commit(self, delta: CommitDelta, *, base: ArchiveSnapshot | None = None) -> CommitToken:
        """Publish ``delta`` and return the new durable commit token.

        Without ``base`` the commit starts from the last snapshot this coordinator published,
        so an uncontended commit reads nothing before its compare-and-swap.
        """
        with self._lock:
            if base is None and self._published is None:
                self._published = read_snapshot(self._layout)
            current = base or self._published
            assert current is not None
            refresh = False

            def publish() -> CommitToken:
                nonlocal current, refresh
                if refresh:
                    current = read_snapshot(self._layout)
                    refresh = False
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
                    _COMMIT_CONFLICTS.add()
                    refresh = True
                    raise
                _COMMITS.add()
                token = CommitToken(
                    commit_id=manifest.commit_id,
                    sequence=manifest.sequence,
                    version=version,
                    manifest_path=manifest_path,
                )
                self._published = ArchiveSnapshot(manifest=manifest, token=token)
                return token

            try:
                return retry_with_backoff(
                    publish,
                    retryable=lambda exc: isinstance(exc, ConditionalWriteError),
                    max_attempts=_MAX_COMMIT_ATTEMPTS,
                    max_elapsed=_COMMIT_RETRY_BUDGET,
                    backoff=ExponentialBackoff(
                        initial=_COMMIT_BACKOFF_INITIAL,
                        maximum=_COMMIT_BACKOFF_MAXIMUM,
                        factor=_RETRY_BACKOFF_FACTOR,
                        jitter=_RETRY_BACKOFF_JITTER,
                    ),
                    operation=f"FineStore commit at {self._layout.root}",
                )
            except ConditionalWriteError as exc:
                raise CommitConflict(
                    f"HEAD at {self._layout.root} kept changing for {_COMMIT_RETRY_BUDGET:.0f} s"
                ) from exc
