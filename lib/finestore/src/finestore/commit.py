# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Manifest snapshots and compare-and-swap commit publication."""

from __future__ import annotations

import hashlib
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

from rigging import telemetry
from rigging.filesystem.conditional_object import ConditionalWriteError, conditional_object
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import ExponentialBackoff, retry_with_backoff

from finestore.layout import (
    CACHE_HISTORY_FLOOR_FEATURE,
    FORMAT_VERSION,
    INITIAL_TABLE_INCARNATION,
    SUPPORTED_MANIFEST_FEATURES,
    TABLE_INCARNATIONS_FEATURE,
    ArchiveMetadata,
    CommitInfo,
    CommitToken,
    FineStoreLayout,
    FormatVersionError,
    HeadMetadata,
    Manifest,
    ManifestTable,
    SealMarker,
    Shard,
    TableContentVersion,
    TableMetadata,
    TableVersion,
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
_CREATE_RETRY_BUDGET = 60.0
_MAX_CREATE_ATTEMPTS = 1000
_ROOT_COMMIT_ID = "root"

_COMMITS = telemetry.counter("finestore_commits", unit="{commit}")
_COMMIT_CONFLICTS = telemetry.counter("finestore_commit_conflicts", unit="{conflict}")


class CommitConflict(RuntimeError):
    """A manifest delta can no longer be applied to the current snapshot."""


@dataclass(frozen=True)
class Attribution:
    """Who makes a commit and why; its manifest records them as ``commit_info``."""

    author: str
    message: str


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
    incarnation: int


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
    reset_tables: frozenset[str] = frozenset()
    table_creations: dict[str, ManifestTable] = field(default_factory=dict)
    table_moves: dict[str, str] = field(default_factory=dict)
    expected_tables: dict[str, TableVersion] = field(default_factory=dict)
    expected_contents: dict[str, TableContentVersion] = field(default_factory=dict)
    expected_incarnations: dict[str, int] | None = None
    required_features: frozenset[str] = frozenset()
    seal_update: SealMarker | ClearSeal | None = None
    # ``expected_epoch`` rejects the commit, without rebasing, once another maintainer has
    # taken over; ``epoch_update`` is how a new maintainer announces itself.
    expected_epoch: int | None = None
    epoch_update: int | None = None
    cache_history_floor: int = 0
    attribution: Attribution | None = None


def _empty_manifest() -> Manifest:
    return Manifest(commit_id=_ROOT_COMMIT_ID, sequence=0)


def _create_or_read(path: str, data: bytes, *, operation: str) -> bytes:
    """Create the object at ``path`` with ``data`` unless it exists, and return the stored bytes."""
    obj = conditional_object(path)

    def attempt() -> bytes:
        existing = obj.read()
        if existing is not None:
            return existing.data
        try:
            obj.write(data, expected_version=None)
            return data
        except ConditionalWriteError:
            existing = obj.read()
            if existing is None:
                # The competing create is still in flight; retry until its object is visible.
                raise
            return existing.data

    return retry_with_backoff(
        attempt,
        retryable=lambda exc: isinstance(exc, ConditionalWriteError),
        max_attempts=_MAX_CREATE_ATTEMPTS,
        max_elapsed=_CREATE_RETRY_BUDGET,
        backoff=ExponentialBackoff(initial=0.1, maximum=1.0, factor=_RETRY_BACKOFF_FACTOR, jitter=_RETRY_BACKOFF_JITTER),
        operation=operation,
    )


def initialize_archive(layout: FineStoreLayout) -> ArchiveMetadata:
    """Create the archive marker, or validate the marker already present."""
    data = _create_or_read(
        layout.archive_path,
        ArchiveMetadata().model_dump_json(indent=2).encode(),
        operation=f"FineStore archive initialization at {layout.root}",
    )
    found = ArchiveMetadata.model_validate_json(data)
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
    if _create_or_read(path, data, operation=f"FineStore schema publication at {path}") != data:
        raise ValueError(f"schema digest collision at {path}")
    return path


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


def _commit_info(attribution: Attribution | None, sequence: int) -> CommitInfo | None:
    if attribution is None:
        return None
    return CommitInfo(
        author=attribution.author,
        message=attribution.message,
        sequence=sequence,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )


def check_preconditions(base: Manifest, delta: CommitDelta) -> None:
    """Reject a delta whose guards fail on this manifest."""
    if delta.expected_epoch is not None and base.maintainer_epoch != delta.expected_epoch:
        raise CommitConflict(
            f"maintenance epoch is {base.maintainer_epoch}, commit was prepared under {delta.expected_epoch}"
        )
    if delta.epoch_update is not None and delta.epoch_update <= base.maintainer_epoch:
        raise CommitConflict(f"maintenance epoch {base.maintainer_epoch} already passed {delta.epoch_update}")
    if delta.expected_incarnations is not None and base.table_incarnations != delta.expected_incarnations:
        raise CommitConflict("a table was dropped or replaced during sealing")
    for name, expected in delta.expected_tables.items():
        if base.table_version(name) != expected:
            raise CommitConflict(f"table {name!r} changed since the expected snapshot")
    for name, expected_content in delta.expected_contents.items():
        current = base.table_content_version(name)
        # Compaction can tighten a legacy upper bound. Appends always advance it.
        if (
            current.incarnation != expected_content.incarnation
            or current.metadata_path != expected_content.metadata_path
            or current.history_floor != expected_content.history_floor
            or current.row_commit_upper_bound > expected_content.row_commit_upper_bound
        ):
            raise CommitConflict(f"table {name!r} changed since the expected snapshot")


def _published_shards(shards: tuple[Shard, ...], sequence: int) -> tuple[Shard, ...]:
    return tuple(
        shard.model_copy(update={"commit_sequence": sequence, "min_row_commit": sequence, "max_row_commit": sequence})
        for shard in shards
    )


def _append_tables(base: Manifest, additions: dict[str, TableAddition], sequence: int) -> dict[str, ManifestTable]:
    tables = dict(base.tables)
    for name, addition in additions.items():
        if base.table_incarnations.get(name, INITIAL_TABLE_INCARNATION) != addition.incarnation:
            raise CommitConflict(f"table {name!r} was dropped or replaced after the writer registered it")
        current = tables.get(name)
        if current is not None and current.metadata_path != addition.metadata_path:
            raise CommitConflict(f"table {name!r} was registered with different metadata")
        prior = current.shards if current is not None else ()
        paths = {shard.path for shard in prior}
        added = _published_shards(tuple(shard for shard in addition.shards if shard.path not in paths), sequence)
        if current is None:
            tables[name] = ManifestTable(metadata_path=addition.metadata_path, shards=added)
        else:
            tables[name] = current.model_copy(update={"shards": prior + added})
    return tables


def _apply_table_ddl(
    tables: dict[str, ManifestTable], base: Manifest, delta: CommitDelta, sequence: int
) -> dict[str, int]:
    incarnations = dict(base.table_incarnations)
    moved_tables = {destination: tables[source] for destination, source in delta.table_moves.items()}
    for name in delta.reset_tables:
        tables.pop(name, None)
        if name in base.tables or name in incarnations:
            incarnations[name] = sequence
    tables.update(moved_tables)
    for name, table in delta.table_creations.items():
        tables[name] = table.model_copy(update={"shards": _published_shards(table.shards, sequence)})
    return incarnations


def _apply_delta(base: Manifest, delta: CommitDelta) -> Manifest:
    check_preconditions(base, delta)
    sequence = base.sequence + 1
    tables = _append_tables(base, delta.additions, sequence)
    incarnations = _apply_table_ddl(tables, base, delta, sequence)

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

    sealed = base.sealed
    if isinstance(delta.seal_update, ClearSeal) or delta.table_creations or delta.reset_tables.intersection(base.tables):
        sealed = None
    elif delta.seal_update is not None:
        sealed = delta.seal_update
    history_floor = max(base.cache_history_floor, delta.cache_history_floor)
    required_features = set(base.required_features) | delta.required_features
    if history_floor:
        required_features.add(CACHE_HISTORY_FLOOR_FEATURE)
    if incarnations:
        required_features.add(TABLE_INCARNATIONS_FEATURE)
    return base.model_copy(
        update={
            "commit_id": uuid.uuid4().hex,
            "parent_commit_id": None if base.commit_id == _ROOT_COMMIT_ID else base.commit_id,
            "sequence": sequence,
            "required_features": tuple(sorted(required_features)),
            "cache_history_floor": history_floor,
            "tables": tables,
            "table_incarnations": incarnations,
            "sealed": sealed,
            "maintainer_epoch": base.maintainer_epoch if delta.epoch_update is None else delta.epoch_update,
            "commit_info": _commit_info(delta.attribution, sequence),
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
                refreshed = refresh
                if refresh:
                    current = read_snapshot(self._layout)
                    refresh = False
                try:
                    manifest = _apply_delta(current.manifest, delta)
                except CommitConflict:
                    if refreshed:
                        raise
                    # A precondition can fail on the cached manifest without reaching HEAD.
                    current = read_snapshot(self._layout)
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
