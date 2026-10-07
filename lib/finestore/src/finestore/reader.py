# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Snapshot reads over the immutable shards selected by one FineStore manifest."""

from __future__ import annotations

import hashlib
import heapq
import io
import itertools
import logging
import sqlite3
import time
from collections import defaultdict
from collections.abc import Generator, Iterator, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, ClassVar, Protocol

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pds
import pyarrow.parquet as pq
import rigging.filesystem.factory as factory
from pyarrow.fs import FSSpecHandler, PyFileSystem
from rigging.filesystem.storage_path import StoragePath

from finestore.commit import ArchiveSnapshot, read_snapshot, validate_archive
from finestore.layout import (
    BlobColumns,
    BlobTables,
    CommitToken,
    FineStoreLayout,
    SealMarker,
    Shard,
    SystemColumns,
    TableMetadata,
    parse_uri,
)

_SUPPORTED_OPS = frozenset({"==", "!=", "in"})
_KEY_INDEX_BYTES = 128 * 1024 * 1024
_KEY_INDEX_BATCH_ROWS = 4096
logger = logging.getLogger(__name__)


@dataclass
class BlobReadDiagnostics:
    """Work performed by blob reads; returned bytes exclude storage read amplification."""

    index_seconds: float = 0.0
    descriptor_seconds: float = 0.0
    payload_seconds: float = 0.0
    indexed_reads: int = 0
    scan_reads: int = 0
    index_fallbacks: int = 0
    index_warm_reads: int = 0
    # Valid name scans performed, including work rolled back after a later failure.
    index_refreshed_shards: int = 0
    index_lock_seconds: float = 0.0
    index_refresh_seconds: float = 0.0
    selected_shards: int = 0
    bytes_returned: int = 0


class _BlobReadHandler(FSSpecHandler):
    def open_input_file(self, path: str) -> pa.PythonFile:
        # Arrow's pre_buffer=False does not disable fsspec's independent 50 MiB
        # read-ahead, which can fetch unrelated inline blobs after a small read.
        return pa.PythonFile(self.fs.open(path, mode="rb", cache_type="none"), mode="r")


@dataclass(frozen=True)
class _ScanProfile:
    batch_rows: int
    batch_readahead: int
    fragment_readahead: int


_TABLE_SCAN_PROFILE = _ScanProfile(batch_rows=16_384, batch_readahead=16, fragment_readahead=4)
_BLOB_DESCRIPTOR_SCAN_PROFILE = _ScanProfile(batch_rows=64, batch_readahead=1, fragment_readahead=1)
_BLOB_SCAN_PROFILE = _ScanProfile(batch_rows=1, batch_readahead=1, fragment_readahead=1)


@dataclass(frozen=True)
class VersionedRow:
    """One row tagged with its key and manifest ordering coordinates."""

    key: tuple
    commit_sequence: int
    sequence: int
    generation: int
    row: dict


@dataclass(frozen=True)
class MergedRow:
    """The winning row for one key and the number of older rows it replaced."""

    row: dict
    superseded: int


class BlobCorruptionError(ValueError):
    """A committed blob is missing data or has inconsistent size metadata."""


@dataclass(frozen=True)
class BlobDescriptor:
    """The descriptor fields needed to read an inline or chunked blob."""

    COLUMNS: ClassVar[tuple[BlobColumns, ...]] = (
        BlobColumns.NAME,
        BlobColumns.SIZE,
        BlobColumns.METADATA,
        BlobColumns.DATA,
        BlobColumns.PART_COUNT,
    )

    name: str
    size: int | None
    metadata_json: str | None
    data: bytes | None
    part_count: int | None

    @classmethod
    def from_row(cls, row: Mapping[str, object]) -> BlobDescriptor:
        """Validate and convert one blob descriptor row."""
        name = row.get(BlobColumns.NAME)
        if not isinstance(name, str):
            raise BlobCorruptionError(f"blob descriptor has invalid name {name!r}")
        size = row.get(BlobColumns.SIZE)
        if size is not None and not isinstance(size, int):
            raise BlobCorruptionError(f"blob {name!r} has invalid size {size!r}")
        metadata_json = row.get(BlobColumns.METADATA)
        if metadata_json is not None and not isinstance(metadata_json, str):
            raise BlobCorruptionError(f"blob {name!r} has invalid metadata {metadata_json!r}")
        data = row.get(BlobColumns.DATA)
        if data is not None and not isinstance(data, bytes):
            raise BlobCorruptionError(f"blob {name!r} has invalid inline data")
        part_count = row.get(BlobColumns.PART_COUNT)
        if part_count is not None and not isinstance(part_count, int):
            raise BlobCorruptionError(f"blob {name!r} has invalid part count {part_count!r}")
        return cls(name=name, size=size, metadata_json=metadata_json, data=data, part_count=part_count)


class _BlobReader(io.RawIOBase):
    """Forward-only file interface over a sequence of bounded blob parts."""

    def __init__(self, parts: Generator[bytes, None, None]) -> None:
        self._parts: Generator[bytes, None, None] | None = parts
        self._current = memoryview(b"")

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        if self.closed:
            raise ValueError("I/O operation on closed blob")
        target = memoryview(buffer).cast("B")
        written = 0
        while written < len(target):
            if not self._current:
                try:
                    assert self._parts is not None
                    self._current = memoryview(next(self._parts))
                except StopIteration:
                    break
            count = min(len(target) - written, len(self._current))
            target[written : written + count] = self._current[:count]
            self._current = self._current[count:]
            written += count
        return written

    def close(self) -> None:
        if self._parts is not None:
            self._parts.close()
            self._parts = None
        self._current = memoryview(b"")
        super().close()


class _ReadableShard(Protocol):
    """The shard coordinates needed to scan one archive snapshot."""

    @property
    def path(self) -> str: ...

    @property
    def generation(self) -> int: ...

    @property
    def commit_sequence(self) -> int: ...

    @property
    def primary_key_sorted(self) -> bool: ...


@dataclass(frozen=True)
class _ReadPlan:
    shards: tuple[_ReadableShard, ...]
    primary_key: tuple[str, ...]
    pushdown_where: list[tuple[str, str, object]]
    post_dedup_where: list[tuple[str, str, object]]
    filesystem: PyFileSystem
    schema: pa.Schema
    columns: list[str] | None


def _build_filter(where: list[tuple[str, str, object]] | None) -> pds.Expression | None:
    if not where:
        return None
    expr: pds.Expression | None = None
    for column, operator, value in where:
        if operator not in _SUPPORTED_OPS:
            raise ValueError(f"unsupported filter op {operator!r}; expected one of {sorted(_SUPPORTED_OPS)}")
        field = pds.field(column)
        clause = field.isin(value) if operator == "in" else (field == value if operator == "==" else field != value)
        expr = clause if expr is None else expr & clause
    return expr


def _partition_filter(
    where: list[tuple[str, str, object]] | None, primary_key: tuple[str, ...]
) -> tuple[list[tuple[str, str, object]], list[tuple[str, str, object]]]:
    primary_key_columns = set(primary_key)
    pushdown = []
    after_deduplication = []
    for clause in where or []:
        if clause[1] not in _SUPPORTED_OPS:
            raise ValueError(f"unsupported filter op {clause[1]!r}; expected one of {sorted(_SUPPORTED_OPS)}")
        target = pushdown if clause[0] in primary_key_columns else after_deduplication
        target.append(clause)
    return pushdown, after_deduplication


def _matches_filter(row: dict, where: list[tuple[str, str, object]]) -> bool:
    for column, operator, value in where:
        candidate = row.get(column)
        if candidate is None:
            return False
        if operator == "==" and candidate != value:
            return False
        if operator == "!=" and candidate == value:
            return False
        if operator == "in" and candidate not in value:
            return False
    return True


def iter_shard_rows(
    shard: _ReadableShard,
    unified: pa.Schema,
    primary_key: tuple[str, ...],
    pa_fs: PyFileSystem,
    columns: list[str] | None = None,
    where: list[tuple[str, str, object]] | None = None,
    scan_profile: _ScanProfile = _TABLE_SCAN_PROFILE,
) -> Iterator[VersionedRow]:
    """Yield rows from one shard in primary-key order with version coordinates."""
    dataset = pds.dataset([shard.path], filesystem=pa_fs, format="parquet", schema=unified)
    if shard.primary_key_sorted:
        batches = dataset.scanner(
            columns=columns,
            filter=_build_filter(where),
            use_threads=False,
            batch_size=scan_profile.batch_rows,
            batch_readahead=scan_profile.batch_readahead,
            fragment_readahead=scan_profile.fragment_readahead,
        ).to_batches()
    else:
        source = dataset.to_table(columns=columns, filter=_build_filter(where))
        sort_columns = [(name, "ascending") for name in primary_key if name in source.column_names]
        batches = source.sort_by(sort_columns).to_batches(max_chunksize=scan_profile.batch_rows)
    for batch in batches:
        for row in batch.to_pylist():
            commit_sequence = row.get(SystemColumns.COMMIT)
            if commit_sequence is None:
                commit_sequence = shard.commit_sequence
            row[SystemColumns.COMMIT] = commit_sequence
            key = tuple((row.get(name) is None, row.get(name)) for name in primary_key)
            yield VersionedRow(
                key=key,
                commit_sequence=commit_sequence,
                sequence=row.get(SystemColumns.SEQUENCE) or 0,
                generation=shard.generation,
                row=row,
            )


def merge_deduplicated_rows(streams: list[Iterator[VersionedRow]]) -> Iterator[MergedRow]:
    """Merge sorted shard streams and retain the newest row for each key."""
    merged = heapq.merge(*streams, key=lambda item: item.key)
    for _key, group in itertools.groupby(merged, key=lambda item: item.key):
        items = list(group)
        winner = max(items, key=lambda item: (item.commit_sequence, item.sequence, item.generation))
        yield MergedRow(row=winner.row, superseded=len(items) - 1)


def _read_plan(
    filesystem: PyFileSystem,
    shards: tuple[_ReadableShard, ...],
    primary_key: tuple[str, ...],
    columns: Sequence[str] | None,
    where: list[tuple[str, str, object]] | None,
) -> _ReadPlan | None:
    if not shards:
        return None
    pushdown_where, post_dedup_where = _partition_filter(where, primary_key)
    schema = pa.unify_schemas(
        [pq.read_schema(shard.path, filesystem=filesystem) for shard in shards],
        promote_options="permissive",
    )
    read_columns = None
    if columns is not None:
        filter_columns = {name for name, _operator, _value in post_dedup_where}
        needed = (
            set(columns)
            | set(primary_key)
            | filter_columns
            | {
                SystemColumns.SEQUENCE,
                SystemColumns.COMMIT,
            }
        )
        read_columns = [name for name in schema.names if name in needed]
    return _ReadPlan(
        shards=shards,
        primary_key=primary_key,
        pushdown_where=pushdown_where,
        post_dedup_where=post_dedup_where,
        filesystem=filesystem,
        schema=schema,
        columns=read_columns,
    )


def _scan_plan(plan: _ReadPlan, columns: Sequence[str] | None, *, scan_profile: _ScanProfile | None = None) -> pa.Table:
    by_version: dict[tuple[int, int], list[str]] = defaultdict(list)
    for shard in plan.shards:
        by_version[(shard.commit_sequence, shard.generation)].append(shard.path)

    parts: list[pa.Table] = []
    for (commit_sequence, generation), paths in sorted(by_version.items()):
        dataset = pds.dataset(paths, filesystem=plan.filesystem, format="parquet", schema=plan.schema)
        if scan_profile is None:
            part = dataset.to_table(columns=plan.columns, filter=_build_filter(plan.pushdown_where))
        else:
            # Inline blob values can make a compacted shard gigabytes larger than the
            # requested results. Bound decoding and disable whole-fragment prefetch.
            part = dataset.scanner(
                columns=plan.columns,
                filter=_build_filter(plan.pushdown_where),
                batch_size=scan_profile.batch_rows,
                batch_readahead=scan_profile.batch_readahead,
                fragment_readahead=scan_profile.fragment_readahead,
                fragment_scan_options=pds.ParquetFragmentScanOptions(pre_buffer=False),
                use_threads=False,
            ).to_table()
        part = part.append_column(
            SystemColumns.GENERATION,
            pa.array([generation] * part.num_rows, pa.int32()),
        )
        commit_values = pa.array([commit_sequence] * part.num_rows, pa.int64())
        if SystemColumns.COMMIT in part.column_names:
            commit_index = part.schema.get_field_index(SystemColumns.COMMIT)
            part = part.set_column(
                commit_index,
                SystemColumns.COMMIT,
                pc.coalesce(part[SystemColumns.COMMIT], commit_values),
            )
        else:
            part = part.append_column(SystemColumns.COMMIT, commit_values)
        parts.append(part)

    combined = parts[0] if len(parts) == 1 else pa.concat_tables(parts, promote_options="permissive")
    if all(name in combined.column_names for name in plan.primary_key):
        combined = _deduplicate(combined, plan.primary_key)
    if plan.post_dedup_where:
        combined = pds.dataset(combined).to_table(filter=_build_filter(plan.post_dedup_where))
    if columns is not None:
        combined = combined.select([name for name in columns if name in combined.column_names])
    return combined


class _ReadOperations:
    """Read operations shared by manifest and legacy listing snapshots."""

    root: str

    def primary_key(self, table: str) -> tuple[str, ...]:
        raise NotImplementedError

    def list_shards(self, table: str) -> Sequence[_ReadableShard]:
        raise NotImplementedError

    def _read_plan(
        self,
        table: str,
        columns: Sequence[str] | None,
        where: list[tuple[str, str, object]] | None,
    ) -> _ReadPlan | None:
        shards = tuple(self.list_shards(table))
        if not shards:
            return None
        fs, _ = factory.url_to_fs(self.root)
        handler = _BlobReadHandler(fs) if table in (BlobTables.DESCRIPTORS, BlobTables.PARTS) else FSSpecHandler(fs)
        return _read_plan(PyFileSystem(handler), shards, self.primary_key(table), columns, where)

    def scan(
        self,
        table: str,
        *,
        columns: Sequence[str] | None = None,
        where: list[tuple[str, str, object]] | None = None,
    ) -> pa.Table | None:
        """Read a deduplicated table, or ``None`` when the table is unknown or has no shards."""
        plan = self._read_plan(table, columns, where)
        if plan is None:
            return None
        return _scan_plan(plan, columns)

    def iter_rows(
        self,
        table: str,
        *,
        columns: Sequence[str] | None = None,
        where: list[tuple[str, str, object]] | None = None,
    ) -> Iterator[dict]:
        """Yield deduplicated rows from this view in primary-key order."""
        return self._iter_rows(
            table,
            columns=columns,
            where=where,
            scan_profile=_TABLE_SCAN_PROFILE,
        )

    def _iter_rows(
        self,
        table: str,
        *,
        columns: Sequence[str] | None,
        where: list[tuple[str, str, object]] | None,
        scan_profile: _ScanProfile,
    ) -> Iterator[dict]:
        plan = self._read_plan(table, columns, where)
        if plan is None:
            return
        streams = [
            iter_shard_rows(
                shard,
                plan.schema,
                plan.primary_key,
                plan.filesystem,
                plan.columns,
                plan.pushdown_where,
                scan_profile,
            )
            for shard in plan.shards
        ]
        for merged in merge_deduplicated_rows(streams):
            row = merged.row
            if not _matches_filter(row, plan.post_dedup_where):
                continue
            if columns is not None:
                row = {name: row[name] for name in columns if name in row}
            yield row

    def point(self, table: str, **keys) -> dict | None:
        result = self.scan(table, where=[(key, "==", value) for key, value in keys.items()])
        if result is None or result.num_rows == 0:
            return None
        return result.slice(0, 1).to_pylist()[0]

    def keys(self, table: str) -> set[tuple]:
        if not self.list_shards(table):
            return set()
        primary_key = self.primary_key(table)
        result = self.scan(table, columns=list(primary_key))
        if result is None:
            return set()
        values = [result.column(name).to_pylist() for name in primary_key]
        return set(zip(*values, strict=True))

    def read_blob(self, name: str) -> bytes | None:
        """Return a named blob, or ``None`` when it is absent."""
        stream = self.open_blob(name)
        if stream is None:
            return None
        with stream:
            return stream.read()

    def read_blobs(self, names: Sequence[str], *, diagnostics: BlobReadDiagnostics | None = None) -> dict[str, bytes]:
        """Read named blobs, omitting absent names."""
        if not names:
            return {}
        rows = self._blob_descriptors(names, diagnostics=diagnostics)
        if rows is None:
            return {}
        started = time.monotonic()
        values = {}
        try:
            for row in rows.to_pylist():
                descriptor = BlobDescriptor.from_row(row)
                with io.BufferedReader(_BlobReader(self.blob_parts(descriptor))) as stream:
                    values[descriptor.name] = stream.read()
        finally:
            if diagnostics is not None:
                diagnostics.payload_seconds += time.monotonic() - started
        if diagnostics is not None:
            diagnostics.bytes_returned += sum(len(value) for value in values.values())
        return values

    def _blob_descriptors(
        self, names: Sequence[str], *, diagnostics: BlobReadDiagnostics | None = None
    ) -> pa.Table | None:
        started = time.monotonic()
        if diagnostics is not None:
            diagnostics.scan_reads += 1
        try:
            plan = self._read_plan(BlobTables.DESCRIPTORS, None, [(BlobColumns.NAME, "in", list(names))])
            if plan is None:
                return None
            if diagnostics is not None:
                diagnostics.selected_shards += len(plan.shards)
            return _scan_plan(plan, None, scan_profile=_BLOB_DESCRIPTOR_SCAN_PROFILE)
        finally:
            if diagnostics is not None:
                diagnostics.descriptor_seconds += time.monotonic() - started

    def open_blob(self, name: str) -> BinaryIO | None:
        """Open a named blob as a forward-only stream, or return ``None`` when absent.

        Chunk and size validation completes when the caller reads through EOF.
        """
        rows = self._blob_descriptors((name,))
        if rows is None or rows.num_rows == 0:
            return None
        row = rows.slice(0, 1).to_pylist()[0]
        return io.BufferedReader(_BlobReader(self.blob_parts(BlobDescriptor.from_row(row))))

    def blob_parts(self, descriptor: BlobDescriptor) -> Generator[bytes, None, None]:
        """Yield a pinned descriptor's inline value or ordered chunked parts."""
        name = descriptor.name
        part_count = descriptor.part_count
        if part_count is None:
            data = descriptor.data
            if data is None:
                raise BlobCorruptionError(f"blob {name!r} has neither inline data nor parts")
            if descriptor.size is not None and len(data) != descriptor.size:
                raise BlobCorruptionError(f"blob {name!r} declares {descriptor.size} bytes but stores {len(data)}")
            yield data
            return
        if part_count <= 0:
            raise BlobCorruptionError(f"blob {name!r} has invalid part count {part_count!r}")
        expected_part = 0
        total_bytes = 0
        for row in self._iter_rows(
            BlobTables.PARTS,
            columns=[BlobColumns.PART, BlobColumns.DATA],
            where=[(BlobColumns.NAME, "==", name)],
            scan_profile=_BLOB_SCAN_PROFILE,
        ):
            part = row.get(BlobColumns.PART)
            if part is not None and part >= part_count:
                break
            if part != expected_part:
                raise BlobCorruptionError(f"blob {name!r} is missing part {expected_part}")
            data = row.get(BlobColumns.DATA)
            if data is None:
                raise BlobCorruptionError(f"blob {name!r} part {part} has no data")
            value = bytes(data)
            total_bytes += len(value)
            expected_part += 1
            yield value
        if expected_part != part_count:
            raise BlobCorruptionError(f"blob {name!r} has {expected_part} of {part_count} parts")
        if descriptor.size is not None and total_bytes != descriptor.size:
            raise BlobCorruptionError(f"blob {name!r} declares {descriptor.size} bytes but stores {total_bytes}")

    def resolve(self, uri: str) -> bytes | None:
        """Resolve a blob URI, returning ``None`` when absent and rejecting unsupported references."""
        ref = parse_uri(uri)
        if ref is None:
            raise ValueError(f"not a finestore:// reference: {uri!r}")
        if ref.table != BlobTables.DESCRIPTORS:
            raise ValueError(f"finestore:// resolution supports the blobs table only, got {ref.table!r}")
        return self.read_blob(ref.key)


def _key_index_read_plan(
    connection: sqlite3.Connection,
    shards: Sequence[Shard],
    identities: Sequence[str],
    indexed: Mapping[str, tuple[int, bytes]],
    names: Sequence[str],
    filesystem: PyFileSystem,
) -> _ReadPlan | None:
    matching_ids = set()
    for start in range(0, len(names), _KEY_INDEX_BATCH_ROWS):
        batch_names = names[start : start + _KEY_INDEX_BATCH_ROWS]
        placeholders = ",".join("?" for _ in batch_names)
        matching_ids.update(
            row[0]
            for row in connection.execute(
                f"SELECT DISTINCT shard_id FROM names WHERE name IN ({placeholders})", batch_names
            )
        )
    selected = []
    schemas = []
    # The local index may also contain another pinned view's identities.
    # Only this manifest's shards participate in version selection.
    for shard, identity in zip(shards, identities, strict=True):
        shard_id, schema_bytes = indexed[identity]
        if shard_id in matching_ids:
            selected.append(shard)
            schemas.append(pa.ipc.read_schema(pa.BufferReader(schema_bytes)))
    if not selected:
        return None
    return _ReadPlan(
        shards=tuple(selected),
        primary_key=(BlobColumns.NAME,),
        pushdown_where=[(BlobColumns.NAME, "in", list(names))],
        post_dedup_where=[],
        filesystem=filesystem,
        schema=pa.unify_schemas(schemas, promote_options="permissive"),
        columns=None,
    )


class BlobKeyIndex:
    """Index descriptor names and schemas without caching payloads or row versions.

    Reads select matching shards from the current manifest. Each archive's index
    uses at most 128 MiB; refresh journals may consume another 128 MiB.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def read_plan(
        self, view: ReadView, names: Sequence[str], *, diagnostics: BlobReadDiagnostics | None = None
    ) -> _ReadPlan | None:
        """Select descriptor shards for names in this view, refreshing new identities."""
        shards = view.list_shards(BlobTables.DESCRIPTORS)
        primary_key = view.primary_key(BlobTables.DESCRIPTORS) if shards else (BlobColumns.NAME,)
        if primary_key != (BlobColumns.NAME,):
            raise ValueError(f"unexpected descriptor primary key: {primary_key}")
        self.directory.mkdir(parents=True, exist_ok=True)
        root_digest = hashlib.sha256(view.root.encode()).hexdigest()
        database = self.directory / f"blob-keys-v1-{root_digest}.sqlite"
        fs, _ = factory.url_to_fs(view.root)
        filesystem = PyFileSystem(_BlobReadHandler(fs))
        identities = [
            hashlib.sha256(
                f"{shard.path}\0{shard.content_sha256}\0{shard.rows}\0{shard.size_bytes}".encode()
            ).hexdigest()
            for shard in shards
        ]
        with closing(sqlite3.connect(database, timeout=10, isolation_level=None)) as connection, connection:
            connection.execute("PRAGMA cache_size = -8192")
            # A read transaction keeps membership and name selection consistent
            # while another subprocess refreshes the shared local index.
            connection.execute("BEGIN")
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' AND name IN ('shards', 'names')"
                )
            }
            if tables == {"shards", "names"}:
                indexed = {
                    identity: (shard_id, schema)
                    for shard_id, identity, schema in connection.execute("SELECT id, identity, schema FROM shards")
                }
                if all(identity in indexed for identity in identities):
                    plan = _key_index_read_plan(connection, shards, identities, indexed, names, filesystem)
                    if diagnostics is not None:
                        diagnostics.index_warm_reads += 1
                    return plan
            connection.commit()
            page_size = connection.execute("PRAGMA page_size").fetchone()[0]
            connection.execute(f"PRAGMA max_page_count = {_KEY_INDEX_BYTES // page_size}")
            connection.execute("PRAGMA foreign_keys = ON")
            started = time.monotonic()
            try:
                connection.execute("BEGIN IMMEDIATE")
            finally:
                if diagnostics is not None:
                    diagnostics.index_lock_seconds += time.monotonic() - started
            refresh_started = time.monotonic()
            try:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS shards "
                    "(id INTEGER PRIMARY KEY, identity TEXT UNIQUE, schema BLOB NOT NULL)"
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS names "
                    "(name TEXT, shard_id INTEGER REFERENCES shards(id) ON DELETE CASCADE, "
                    "PRIMARY KEY (name, shard_id)) WITHOUT ROWID"
                )
                connection.execute("CREATE INDEX IF NOT EXISTS names_by_shard ON names(shard_id)")
                connection.execute("CREATE TEMP TABLE active (identity TEXT PRIMARY KEY)")
                connection.executemany("INSERT INTO active VALUES (?)", ((identity,) for identity in identities))
                connection.execute("DELETE FROM shards WHERE identity NOT IN (SELECT identity FROM active)")
                indexed = {
                    identity: (shard_id, schema)
                    for shard_id, identity, schema in connection.execute("SELECT id, identity, schema FROM shards")
                }
                for shard, identity in zip(shards, identities, strict=True):
                    if identity in indexed:
                        continue
                    # Avoid fsspec's payload readahead while indexing only the name column.
                    with (
                        filesystem.open_input_file(shard.path) as source,
                        pq.ParquetFile(source, pre_buffer=False) as parquet,
                    ):
                        schema = parquet.schema_arrow.serialize().to_pybytes()
                        cursor = connection.execute(
                            "INSERT INTO shards(identity, schema) VALUES (?, ?)", (identity, schema)
                        )
                        shard_id = cursor.lastrowid
                        row_count = 0
                        for batch in parquet.iter_batches(
                            columns=[BlobColumns.NAME], batch_size=_KEY_INDEX_BATCH_ROWS, use_threads=False
                        ):
                            keys = batch.column(0).to_pylist()
                            if any(not isinstance(key, str) for key in keys):
                                raise BlobCorruptionError(f"invalid descriptor name in {shard.path}")
                            row_count += len(keys)
                            connection.executemany(
                                "INSERT OR IGNORE INTO names VALUES (?, ?)", ((key, shard_id) for key in keys)
                            )
                        if row_count != shard.rows:
                            raise BlobCorruptionError(f"descriptor row count differs from manifest for {shard.path}")
                        indexed[identity] = (shard_id, schema)
                        if diagnostics is not None:
                            diagnostics.index_refreshed_shards += 1
            finally:
                if diagnostics is not None:
                    diagnostics.index_refresh_seconds += time.monotonic() - refresh_started
            return _key_index_read_plan(connection, shards, identities, indexed, names, filesystem)


class ReadView(_ReadOperations):
    """A read-only archive view pinned to one commit token."""

    def __init__(
        self, root: str, snapshot: ArchiveSnapshot | None = None, *, blob_key_index: BlobKeyIndex | None = None
    ) -> None:
        self.root = root
        self._blob_key_index = blob_key_index
        self._layout = FineStoreLayout(self.root)
        # The marker only distinguishes a v1 archive from an empty root: v1 archives have no
        # HEAD, and read_snapshot validates the format version HEAD carries. A missing marker
        # is expected under a lifecycle rule that expires write-once objects, and the next
        # writer open recreates it.
        validate_archive(self._layout)
        self._snapshot = snapshot or read_snapshot(self._layout)
        self._meta_cache: dict[str, TableMetadata] = {}

    def _blob_descriptors(
        self, names: Sequence[str], *, diagnostics: BlobReadDiagnostics | None = None
    ) -> pa.Table | None:
        if self._blob_key_index is None:
            return super()._blob_descriptors(names, diagnostics=diagnostics)
        started = time.monotonic()
        try:
            plan = self._blob_key_index.read_plan(self, names, diagnostics=diagnostics)
        except Exception as exc:
            # Local index failure must not hide a valid expensive inference completion.
            logger.warning("FineStore key index is unavailable, falling back to descriptor scan: %s", exc)
            if diagnostics is not None:
                diagnostics.index_seconds += time.monotonic() - started
                diagnostics.index_fallbacks += 1
            return super()._blob_descriptors(names, diagnostics=diagnostics)
        if diagnostics is not None:
            diagnostics.index_seconds += time.monotonic() - started
            diagnostics.indexed_reads += 1
        if plan is None:
            return None
        started = time.monotonic()
        if diagnostics is not None:
            diagnostics.selected_shards += len(plan.shards)
        try:
            return _scan_plan(plan, None, scan_profile=_BLOB_DESCRIPTOR_SCAN_PROFILE)
        finally:
            if diagnostics is not None:
                diagnostics.descriptor_seconds += time.monotonic() - started

    @property
    def token(self) -> CommitToken | None:
        """The HEAD version that selected this view, or ``None`` for an empty archive."""
        return self._snapshot.token

    def primary_key(self, table: str) -> tuple[str, ...]:
        return self._meta(table).primary_key

    def table_metadata(self, table: str) -> TableMetadata:
        """Return the logical metadata pinned for ``table`` in this view."""
        return self._meta(table)

    def table_metadata_path(self, table: str) -> str | None:
        """Return the immutable metadata path selected for ``table``."""
        state = self._snapshot.manifest.tables.get(table)
        return None if state is None else state.metadata_path

    def table_names(self) -> tuple[str, ...]:
        """Return the table names selected by this view's manifest."""
        return tuple(sorted(self._snapshot.manifest.tables))

    def schema_version(self, table: str) -> int | None:
        """Return the table schema version, or ``None`` when it has no active shards."""
        state = self._snapshot.manifest.tables.get(table)
        if state is None or not state.shards:
            return None
        return self._meta(table).schema_version

    def _meta(self, table: str) -> TableMetadata:
        cached = self._meta_cache.get(table)
        if cached is not None:
            return cached
        state = self._snapshot.manifest.tables.get(table)
        if state is None:
            raise KeyError(f"table {table!r} is not present in commit {self._snapshot.manifest.commit_id}")
        metadata = TableMetadata.model_validate_json(StoragePath(state.metadata_path).read_bytes())
        self._meta_cache[table] = metadata
        return metadata

    def is_sealed(self) -> bool:
        return self._snapshot.manifest.sealed is not None

    def seal_marker(self) -> SealMarker | None:
        """Return the committed seal metadata, if this view is sealed."""
        return self._snapshot.manifest.sealed

    def max_seq(self, table: str) -> int:
        """Return the greatest sequence in active shards, or ``-1`` when there are none."""
        shards = self.list_shards(table)
        return max((shard.max_seq for shard in shards), default=-1)

    def list_shards(self, table: str) -> list[Shard]:
        state = self._snapshot.manifest.tables.get(table)
        return [] if state is None else list(state.shards)


def _deduplicate(table: pa.Table, primary_key: tuple[str, ...]) -> pa.Table:
    """Keep the latest committed row per primary key."""
    if table.num_rows == 0:
        return table
    key_columns = [table.column(name).to_pylist() for name in primary_key]
    commits = table.column(SystemColumns.COMMIT).to_pylist()
    generations = table.column(SystemColumns.GENERATION).to_pylist()
    sequences = (
        table.column(SystemColumns.SEQUENCE).to_pylist()
        if SystemColumns.SEQUENCE in table.column_names
        else [0] * table.num_rows
    )
    order = sorted(range(table.num_rows), key=lambda index: (commits[index], sequences[index], generations[index]))
    winners: dict[tuple, int] = {}
    for index in order:
        winners[tuple(column[index] for column in key_columns)] = index
    keep = sorted(winners.values())
    return table if len(keep) == table.num_rows else table.take(pa.array(keep, pa.int64()))
