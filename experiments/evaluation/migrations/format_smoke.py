# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Non-destructive production smoke test for a FineStore v1-to-v2 migration."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.dataset as pds
import pyarrow.parquet as pq
from finestore.layout import ARCHIVE_FILE, COMMIT_COLUMN, GEN_COLUMN, SEQ_COLUMN, ArchiveMetadata, FineStoreLayout
from finestore.migrations import migrate
from finestore.migrations.m0001_manifest import LEGACY_SEAL_FILE, LegacyArchive, LegacyTable, inspect_legacy_archive
from finestore.reader import ReadView
from marin.evaluation.records import list_records, read_records
from pyarrow.fs import FSSpecHandler, PyFileSystem
from rigging.filesystem import StoragePath, marin_temp_bucket, url_to_fs

_COPY_BUFFER_BYTES = 8 * 1024 * 1024
_FORMAT_V1 = 1
_FORMAT_V2 = 2
_EVAL_SAMPLES_TABLE = "samples"
_FLEET_DIRECTORY = "finestore-migration-fleet"
_FLEET_MARKER = "_fleet.json"
_FLEET_ID = re.compile(r"^[0-9a-f]{32}$")
_TTL_DIRECTORY = re.compile(r"^ttl=[1-9][0-9]*d$")
_RESULTS_DIRECTORY = "results"


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


@dataclass(frozen=True)
class FleetSelection:
    """FineStore archives referenced by one or more record prefixes."""

    records: int
    unique_results: int
    sources: tuple[str, ...]
    already_current: int
    missing_archive: int
    foreign_results: int
    record_failures: tuple[str, ...]
    unsealed_v1: tuple[str, ...]
    unsupported: tuple[str, ...]


@dataclass(frozen=True)
class FleetSmokeResult:
    """Aggregate evidence from a successful fleet-wide migration rehearsal."""

    destination_root: str
    results: tuple[SmokeUpgradeResult, ...]
    source_objects: int
    source_bytes: int
    rows: int
    cleaned_up: bool


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
    if not _same_storage(source, destination):
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


def _same_storage(left: StoragePath, right: StoragePath) -> bool:
    if left.is_local and right.is_local:
        return True
    return left.scheme == right.scheme and left.bucket == right.bucket


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
    for results_path in sorted({str(StoragePath(record.results_path)) for record in list_records(records_prefix)}):
        root = StoragePath(results_path)
        if root.scheme != prefix.scheme or root.bucket != prefix.bucket:
            continue
        archive_path = StoragePath(FineStoreLayout(results_path).archive_path)
        if not archive_path.exists():
            continue
        metadata = ArchiveMetadata.model_validate_json(archive_path.read_bytes())
        if metadata.format_version != _FORMAT_V1:
            continue
        seal_path = root / LEGACY_SEAL_FILE
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
    run_name = root.parent.name if root.name == _RESULTS_DIRECTORY else root.name
    identity = hashlib.sha256(source.encode()).hexdigest()[:12]
    prefix = f"finestore-migration-smoke/{run_name}-{identity}-{uuid.uuid4().hex[:12]}"
    return marin_temp_bucket(ttl_days, prefix=prefix, source_prefix=source)


def select_v1_fleet(records_prefixes: tuple[str, ...]) -> FleetSelection:
    """Find every sealed v1 archive referenced by records on the scanned storage backends."""
    if not records_prefixes:
        raise ValueError("at least one records prefix is required")

    records = []
    record_failures = []
    for prefix in records_prefixes:
        prefix_records, failures = read_records(prefix)
        records.extend(prefix_records)
        record_failures.extend(failure.path for failure in failures)

    backends = {(StoragePath(prefix).scheme, StoragePath(prefix).bucket) for prefix in records_prefixes}
    results_paths = sorted({str(StoragePath(record.results_path)) for record in records})
    sources = []
    unsealed_v1 = []
    unsupported = []
    already_current = 0
    missing_archive = 0
    foreign_results = 0
    for results_path in results_paths:
        root = StoragePath(results_path)
        if (root.scheme, root.bucket) not in backends:
            foreign_results += 1
            continue
        archive_path = StoragePath(FineStoreLayout(results_path).archive_path)
        if not archive_path.exists():
            missing_archive += 1
            continue
        metadata = ArchiveMetadata.model_validate_json(archive_path.read_bytes())
        if metadata.format_version == _FORMAT_V2:
            already_current += 1
            continue
        if metadata.format_version != _FORMAT_V1:
            unsupported.append(f"v{metadata.format_version}:{results_path}")
            continue
        if not (root / LEGACY_SEAL_FILE).exists():
            unsealed_v1.append(results_path)
            continue
        sources.append(results_path)

    return FleetSelection(
        records=len(records),
        unique_results=len(results_paths),
        sources=tuple(sources),
        already_current=already_current,
        missing_archive=missing_archive,
        foreign_results=foreign_results,
        record_failures=tuple(sorted(record_failures)),
        unsealed_v1=tuple(unsealed_v1),
        unsupported=tuple(unsupported),
    )


def fleet_destination(source_prefix: str, *, ttl_days: int) -> str:
    """Return a unique lifecycle-managed root for a complete migration rehearsal."""
    fleet_id = uuid.uuid4().hex
    return marin_temp_bucket(
        ttl_days,
        prefix=f"{_FLEET_DIRECTORY}/{fleet_id}",
        source_prefix=source_prefix,
    )


def _fleet_archive_destination(root: StoragePath, source: str) -> StoragePath:
    source_root = StoragePath(source)
    run_name = source_root.parent.name if source_root.name == _RESULTS_DIRECTORY else source_root.name
    identity = hashlib.sha256(source.encode()).hexdigest()[:16]
    return root / f"{run_name}-{identity}"


def _validate_fleet_root(root: StoragePath) -> None:
    if len(root.segments) < 4:
        raise ValueError(f"fleet cleanup root is too broad: {root}")
    if root.segments[-2] != _FLEET_DIRECTORY or _FLEET_ID.fullmatch(root.name) is None:
        raise ValueError(f"fleet cleanup root lacks a generated fleet identity: {root}")
    if root.segments[-4] != "tmp" or _TTL_DIRECTORY.fullmatch(root.segments[-3]) is None:
        raise ValueError(f"fleet cleanup root is not lifecycle-managed temporary storage: {root}")


def _prepare_fleet_root(root: StoragePath, sources: tuple[str, ...]) -> None:
    _validate_fleet_root(root)
    if root.exists():
        raise FileExistsError(f"fleet migration destination already exists: {root}")
    for source in sources:
        if not _same_storage(StoragePath(source), root):
            raise ValueError(f"fleet migration must stay on one storage backend and bucket: {source} -> {root}")
    marker = root / _FLEET_MARKER
    marker.parent.mkdirs()
    marker.write_text(json.dumps({"fleet_id": root.name, "root": str(root)}, sort_keys=True))


def _cleanup_fleet_root(root: StoragePath) -> None:
    _validate_fleet_root(root)
    marker = root / _FLEET_MARKER
    expected = {"fleet_id": root.name, "root": str(root)}
    if not marker.exists() or json.loads(marker.read_text()) != expected:
        raise ValueError(f"fleet cleanup marker does not match the requested root: {root}")
    root.rmtree()
    if root.exists():
        raise RuntimeError(f"fleet cleanup did not remove {root}")


def smoke_upgrade_fleet(
    sources: tuple[str, ...],
    destination_root: str,
    *,
    workers: int,
    cleanup: bool,
    on_result: Callable[[SmokeUpgradeResult], None] | None = None,
) -> FleetSmokeResult:
    """Rehearse every source migration and optionally remove the fleet after all validate."""
    if not sources:
        raise ValueError("no sealed FineStore v1 archives were selected")
    if workers < 1:
        raise ValueError("workers must be positive")

    root = StoragePath(destination_root)
    _prepare_fleet_root(root, sources)
    results = []
    executor = ThreadPoolExecutor(max_workers=workers)
    futures = {
        executor.submit(smoke_upgrade, source, str(_fleet_archive_destination(root, source))): source
        for source in sources
    }
    try:
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if on_result is not None:
                on_result(result)
    except BaseException:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)

    ordered = tuple(sorted(results, key=lambda result: result.source))
    if cleanup:
        _cleanup_fleet_root(root)
    return FleetSmokeResult(
        destination_root=destination_root,
        results=ordered,
        source_objects=sum(result.source_objects for result in ordered),
        source_bytes=sum(result.source_bytes for result in ordered),
        rows=sum(result.rows for result in ordered),
        cleaned_up=cleanup,
    )


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
    immutable_paths = tuple(path for path in relative_paths if path != ARCHIVE_FILE)
    copied_immutable = tuple(file for file in copied_files if file.relative_path != ARCHIVE_FILE)
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
