# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Non-destructive production smoke test for a FineStore v1-to-v2 migration."""

from __future__ import annotations

import hashlib
import uuid
from collections import defaultdict
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.dataset as pds
import pyarrow.parquet as pq
from finestore.layout import COMMIT_COLUMN, GEN_COLUMN, SEQ_COLUMN, ArchiveMetadata, FineStoreLayout
from finestore.migrations import migrate
from finestore.migrations.m0001_manifest import LegacyArchive, LegacyTable, inspect_legacy_archive
from finestore.reader import ReadView
from marin.evaluation.records import list_records
from pyarrow.fs import FSSpecHandler, PyFileSystem
from rigging.filesystem import StoragePath, marin_temp_bucket, url_to_fs

_COPY_BUFFER_BYTES = 8 * 1024 * 1024
_FORMAT_V1 = 1
_FORMAT_V2 = 2
_EVAL_SAMPLES_TABLE = "samples"


@dataclass(frozen=True)
class FileFingerprint:
    """Content identity for one FineStore-owned object."""

    relative_path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class TableValidation:
    """Logical equality result for one migrated table."""

    name: str
    rows: int
    sha256: str


@dataclass(frozen=True)
class SmokeUpgradeResult:
    """Structured evidence produced by a successful non-destructive smoke upgrade."""

    source: str
    destination: str
    source_objects: int
    source_bytes: int
    source_sha256: str
    tables: tuple[TableValidation, ...]
    rows: int
    commit_id: str
    commit_sequence: int


class SmokeUpgradeValidationError(ValueError):
    """The cloned or migrated archive differs from its v1 source."""


def _legacy_paths(root: StoragePath, archive: LegacyArchive) -> tuple[str, ...]:
    paths = [FineStoreLayout(str(root)).archive_path, archive.seal_path]
    for table in archive.tables.values():
        paths.append(table.metadata_path)
        paths.extend(shard.path for shard in table.shards)
    return tuple(sorted(paths))


def _sha256(path: StoragePath) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_COPY_BUFFER_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprints(root: StoragePath, relative_paths: tuple[str, ...]) -> tuple[FileFingerprint, ...]:
    fingerprints = []
    for relative_path in relative_paths:
        path = root / relative_path
        fingerprints.append(FileFingerprint(relative_path=relative_path, size=path.size(), sha256=_sha256(path)))
    return tuple(fingerprints)


def _archive_digest(files: tuple[FileFingerprint, ...]) -> str:
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.relative_path.encode())
        digest.update(b"\0")
        digest.update(str(file.size).encode())
        digest.update(b"\0")
        digest.update(file.sha256.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _relative_paths(root: StoragePath, paths: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(StoragePath(path).relative_to(root) for path in paths)


def _copy_files(source: StoragePath, destination: StoragePath, relative_paths: tuple[str, ...]) -> None:
    same_storage = source.is_local and destination.is_local
    if source.is_remote or destination.is_remote:
        same_storage = source.scheme == destination.scheme and source.bucket == destination.bucket
    if not same_storage:
        raise ValueError(f"smoke migration must stay on one storage backend and bucket: {source} -> {destination}")
    if destination.exists():
        raise FileExistsError(f"smoke migration destination already exists: {destination}")

    for relative_path in relative_paths:
        source_path = source / relative_path
        destination_path = destination / relative_path
        source_fs, source_key = url_to_fs(str(source_path))
        _destination_fs, destination_key = url_to_fs(str(destination_path))
        destination_path.parent.mkdirs()
        source_fs.cp_file(source_key, destination_key)


def _deduplicate_v1(table: pa.Table, primary_key: tuple[str, ...]) -> pa.Table:
    if table.num_rows == 0:
        return table
    key_columns = [table.column(name).to_pylist() for name in primary_key]
    generations = table.column(GEN_COLUMN).to_pylist()
    sequences = table.column(SEQ_COLUMN).to_pylist() if SEQ_COLUMN in table.column_names else [0] * table.num_rows
    order = sorted(range(table.num_rows), key=lambda index: (sequences[index], generations[index]))
    winners: dict[tuple, int] = {}
    for index in order:
        winners[tuple(column[index] for column in key_columns)] = index
    keep = sorted(winners.values())
    return table if len(keep) == table.num_rows else table.take(pa.array(keep, pa.int64()))


def _read_v1_table(root: str, table: LegacyTable) -> pa.Table | None:
    if not table.shards:
        return None
    fs, _ = url_to_fs(root)
    pa_fs = PyFileSystem(FSSpecHandler(fs))
    unified = pa.unify_schemas(
        [pq.read_schema(shard.path, filesystem=pa_fs) for shard in table.shards],
        promote_options="permissive",
    )
    paths_by_generation: dict[int, list[str]] = defaultdict(list)
    for shard in table.shards:
        paths_by_generation[shard.generation].append(shard.path)

    parts = []
    for generation, paths in sorted(paths_by_generation.items()):
        part = pds.dataset(paths, filesystem=pa_fs, format="parquet", schema=unified).to_table()
        part = part.append_column(GEN_COLUMN, pa.array([generation] * part.num_rows, pa.int32()))
        parts.append(part)
    combined = parts[0] if len(parts) == 1 else pa.concat_tables(parts, promote_options="permissive")
    if all(name in combined.column_names for name in table.metadata.primary_key):
        combined = _deduplicate_v1(combined, table.metadata.primary_key)
    return _canonical_table(combined, table.metadata.primary_key)


def _canonical_table(table: pa.Table, primary_key: tuple[str, ...]) -> pa.Table:
    if COMMIT_COLUMN in table.column_names:
        table = table.drop_columns([COMMIT_COLUMN])
    table = table.select(sorted(table.column_names))
    if table.num_rows and all(name in table.column_names for name in primary_key):
        table = table.sort_by([(name, "ascending") for name in primary_key])
    return table.combine_chunks()


def _table_digest(table: pa.Table) -> str:
    output = pa.BufferOutputStream()
    with pa.ipc.new_stream(output, table.schema) as writer:
        writer.write_table(table)
    return hashlib.sha256(output.getvalue()).hexdigest()


def _validate_tables(
    source: str,
    archive: LegacyArchive,
    expected: dict[str, pa.Table | None],
    reader: ReadView,
) -> tuple[TableValidation, ...]:
    if reader.table_names() != tuple(sorted(archive.tables)):
        raise SmokeUpgradeValidationError(
            f"table set changed during migration: {tuple(sorted(archive.tables))!r} -> {reader.table_names()!r}"
        )
    validations = []
    for name, legacy_table in sorted(archive.tables.items()):
        if reader.table_metadata(name) != legacy_table.metadata:
            raise SmokeUpgradeValidationError(f"table metadata changed during migration for {name!r}")
        before = expected[name]
        after = reader.scan(name)
        if before is None or after is None:
            if before is not None or after is not None:
                raise SmokeUpgradeValidationError(f"empty table state changed during migration for {name!r}")
            validations.append(TableValidation(name=name, rows=0, sha256=hashlib.sha256(b"").hexdigest()))
            continue
        after = _canonical_table(after, legacy_table.metadata.primary_key)
        if not before.equals(after):
            raise SmokeUpgradeValidationError(f"logical rows changed during migration for {name!r} from {source!r}")
        validations.append(TableValidation(name=name, rows=before.num_rows, sha256=_table_digest(before)))
    return tuple(validations)


def _assert_fingerprints_equal(
    expected: tuple[FileFingerprint, ...],
    actual: tuple[FileFingerprint, ...],
    *,
    context: str,
) -> None:
    if expected != actual:
        raise SmokeUpgradeValidationError(f"FineStore-owned objects changed during {context}")


def select_v1_archive(records_prefix: str, *, max_bytes: int) -> str:
    """Choose a sealed v1 eval archive with sample rows and at most ``max_bytes`` of data."""
    prefix = StoragePath(records_prefix)
    for results_path in sorted({record.results_path.rstrip("/") for record in list_records(records_prefix)}):
        root = StoragePath(results_path)
        if root.scheme != prefix.scheme or root.bucket != prefix.bucket:
            continue
        archive_path = StoragePath(FineStoreLayout(results_path).archive_path)
        if not archive_path.exists():
            continue
        metadata = ArchiveMetadata.model_validate_json(archive_path.read_bytes())
        if metadata.format_version != _FORMAT_V1:
            continue
        seal_path = root / "SEALED"
        if not seal_path.exists():
            continue
        archive = inspect_legacy_archive(results_path)
        samples = archive.tables.get(_EVAL_SAMPLES_TABLE)
        if samples is None or not any(shard.rows for shard in samples.shards):
            continue
        paths = _relative_paths(root, _legacy_paths(root, archive))
        total_bytes = sum((root / relative_path).size() for relative_path in paths)
        if total_bytes <= max_bytes:
            return results_path
    raise ValueError(
        f"no sealed FineStore v1 archive with sample rows at or below {max_bytes} bytes under {records_prefix!r}"
    )


def smoke_destination(source: str, *, ttl_days: int) -> str:
    """Return a unique same-region lifecycle-managed prefix for one smoke migration."""
    root = StoragePath(source)
    run_name = root.parent.name if root.name == "results" else root.name
    identity = hashlib.sha256(source.encode()).hexdigest()[:12]
    prefix = f"finestore-migration-smoke/{run_name}-{identity}-{uuid.uuid4().hex[:12]}"
    return marin_temp_bucket(ttl_days, prefix=prefix, source_prefix=source)


def smoke_upgrade(source: str, destination: str) -> SmokeUpgradeResult:
    """Clone one sealed v1 archive, migrate the clone, and prove its contents unchanged."""
    source_root = StoragePath(source)
    destination_root = StoragePath(destination)
    archive_path = StoragePath(FineStoreLayout(source).archive_path)
    metadata = ArchiveMetadata.model_validate_json(archive_path.read_bytes())
    if metadata.format_version != _FORMAT_V1:
        raise ValueError(f"smoke migration source must be FineStore format v1; found v{metadata.format_version}")

    legacy = inspect_legacy_archive(source)
    source_paths = _legacy_paths(source_root, legacy)
    relative_paths = _relative_paths(source_root, source_paths)
    expected_tables = {name: _read_v1_table(source, table) for name, table in legacy.tables.items()}
    source_files = _fingerprints(source_root, relative_paths)

    _copy_files(source_root, destination_root, relative_paths)
    copied_files = _fingerprints(destination_root, relative_paths)
    _assert_fingerprints_equal(source_files, copied_files, context="the source-to-temp copy")

    migration = migrate(destination)
    token = migration.token
    if migration.applied != ("0001_manifest",) or token is None:
        raise SmokeUpgradeValidationError(f"unexpected migration result: {migration}")

    reader = ReadView(destination)
    if not reader.is_sealed() or reader.seal_marker() != legacy.seal:
        raise SmokeUpgradeValidationError("seal state changed during migration")
    validations = _validate_tables(source, legacy, expected_tables, reader)

    source_after = _fingerprints(source_root, relative_paths)
    _assert_fingerprints_equal(source_files, source_after, context="migration of the temporary clone")
    immutable_paths = tuple(path for path in relative_paths if path != "_archive.json")
    copied_immutable = tuple(file for file in copied_files if file.relative_path != "_archive.json")
    migrated_immutable = _fingerprints(destination_root, immutable_paths)
    _assert_fingerprints_equal(copied_immutable, migrated_immutable, context="v2 manifest publication")

    migrated_metadata = ArchiveMetadata.model_validate_json(
        StoragePath(FineStoreLayout(destination).archive_path).read_bytes()
    )
    if migrated_metadata.format_version != _FORMAT_V2:
        raise SmokeUpgradeValidationError(
            f"temporary archive format marker is v{migrated_metadata.format_version}, expected v{_FORMAT_V2}"
        )

    return SmokeUpgradeResult(
        source=source,
        destination=destination,
        source_objects=len(source_files),
        source_bytes=sum(file.size for file in source_files),
        source_sha256=_archive_digest(source_files),
        tables=validations,
        rows=sum(table.rows for table in validations),
        commit_id=token.commit_id,
        commit_sequence=token.sequence,
    )
