# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Snapshot reads over the immutable shards selected by one FineStore manifest."""

from __future__ import annotations

import functools
import heapq
import io
import itertools
import threading
import time
from collections import OrderedDict, defaultdict
from collections.abc import Generator, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, closing
from dataclasses import dataclass, replace
from typing import BinaryIO, ClassVar, Protocol, TypeVar, cast

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pds
import pyarrow.parquet as pq
import rigging.filesystem.factory as factory
from fsspec.spec import AbstractBufferedFile, AbstractFileSystem
from pyarrow.fs import FileInfo, FileType, FSSpecHandler, PyFileSystem
from rigging import telemetry
from rigging.filesystem.storage_path import StoragePath

from finestore.commit import ArchiveSnapshot, read_snapshot
from finestore.layout import (
    BlobColumns,
    BlobTables,
    CommitToken,
    FineStoreLayout,
    KeyRange,
    SealMarker,
    Shard,
    SystemColumns,
    TableMetadata,
    parse_uri,
)

_SUPPORTED_OPS = frozenset({"==", "!=", "in"})
MAX_DESCRIPTOR_READS = 4
_READ_CACHE_BYTES = 64 * 1024 * 1024


@dataclass
class BlobReadDiagnostics:
    """Cumulative batch-read work; returned bytes exclude storage read amplification."""

    read_calls: int = 0
    requested_names: int = 0
    found_names: int = 0
    descriptor_seconds: float = 0.0
    payload_seconds: float = 0.0
    descriptor_lookups: int = 0
    selected_shards: int = 0
    bytes_returned: int = 0

    def add(self, other: BlobReadDiagnostics) -> None:
        self.read_calls += other.read_calls
        self.requested_names += other.requested_names
        self.found_names += other.found_names
        self.descriptor_seconds += other.descriptor_seconds
        self.payload_seconds += other.payload_seconds
        self.descriptor_lookups += other.descriptor_lookups
        self.selected_shards += other.selected_shards
        self.bytes_returned += other.bytes_returned


_BLOB_READ_CALLS = telemetry.counter("finestore_blob_read_calls", unit="{call}")
_BLOB_REQUESTED_NAMES = telemetry.counter("finestore_blob_requested_names", unit="{name}")
_BLOB_FOUND_NAMES = telemetry.counter("finestore_blob_found_names", unit="{name}")
_BLOB_RETURNED_BYTES = telemetry.counter("finestore_blob_returned_bytes", unit="By")
_BLOB_DESCRIPTOR_LOOKUPS = telemetry.counter("finestore_blob_descriptor_lookups", unit="{lookup}")
_BLOB_DESCRIPTOR_SECONDS = telemetry.histogram("finestore_blob_descriptor_seconds", unit="s")
_BLOB_PAYLOAD_SECONDS = telemetry.histogram("finestore_blob_payload_seconds", unit="s")
_BLOB_SELECTED_SHARDS = telemetry.histogram("finestore_blob_selected_shards", unit="{shard}")


class _ShardHandler(FSSpecHandler):
    """Open manifest shards without the existence check Arrow's handler issues first.

    Every shard in a snapshot is immutable and already published, so the HEAD request
    Arrow would send before each open is wasted on object storage.
    """

    def open_input_file(self, path: str) -> pa.PythonFile:
        return pa.PythonFile(self.fs.open(path, mode="rb"), mode="r")


class _NoReadAheadHandler(FSSpecHandler):
    def open_input_file(self, path: str) -> pa.PythonFile:
        # Arrow's pre_buffer=False does not disable fsspec's independent 50 MiB
        # read-ahead, which can fetch unrelated inline blobs after a small read.
        return pa.PythonFile(self.fs.open(path, mode="rb", cache_type="none"), mode="r")


class _BoundedReadAheadHandler(FSSpecHandler):
    def __init__(self, fs: AbstractFileSystem, block_size: int) -> None:
        super().__init__(fs)
        self._block_size = block_size

    def open_input_file(self, path: str) -> pa.PythonFile:
        return pa.PythonFile(
            self.fs.open(path, mode="rb", cache_type="readahead", block_size=self._block_size), mode="r"
        )


class _ReadRangeCache:
    """A byte-bounded LRU shared by all shard handles in one snapshot."""

    def __init__(self, filesystem: AbstractFileSystem, shards: Sequence[Shard], capacity: int) -> None:
        self._filesystem = filesystem
        self._capacity = capacity
        self._bytes = 0
        self._ranges: dict[str, dict[tuple[int, int], bytes]] = {shard.path: {} for shard in shards}
        self._entries: OrderedDict[tuple[str, int, int], bytes] = OrderedDict()
        self._file_locks = {path: threading.Lock() for path in self._ranges}
        self._lock = threading.Lock()

    def read(self, path: str, start: int, end: int) -> bytes:
        # Serialize misses for the same file while allowing different shards to load concurrently.
        with self._file_locks[path]:
            with self._lock:
                for (cached_start, cached_end), data in self._ranges[path].items():
                    if cached_start <= start and end <= cached_end:
                        self._entries.move_to_end((path, cached_start, cached_end))
                        return data[start - cached_start : end - cached_start]
            data = cast(bytes, self._filesystem.cat_file(path, start=start, end=end))
            if len(data) != end - start:
                raise ValueError(f"shard {path!r}: read {len(data)} of {end - start} bytes at {start}")
            # Oversized caller reads remain transient and do not evict the working set.
            if len(data) > self._capacity:
                return data
            with self._lock:
                while self._bytes + len(data) > self._capacity:
                    (old_path, old_start, old_end), old = self._entries.popitem(last=False)
                    del self._ranges[old_path][old_start, old_end]
                    self._bytes -= len(old)
                self._entries[path, start, end] = data
                self._ranges[path][start, end] = data
                self._bytes += len(data)
            return data


class _CachedReadFile(AbstractBufferedFile):
    def __init__(self, filesystem: AbstractFileSystem, path: str, size: int, cache: _ReadRangeCache) -> None:
        super().__init__(filesystem, path, mode="rb", size=size, cache_type="none")
        self._range_cache = cache

    def _fetch_range(self, start: int, end: int) -> bytes:
        return self._range_cache.read(self.path, start, min(end, self.size))


class _CachedReadHandler(FSSpecHandler):
    def __init__(self, filesystem: AbstractFileSystem, shards: Sequence[Shard], capacity: int) -> None:
        super().__init__(filesystem)
        self._sizes = {shard.path: shard.size_bytes for shard in shards}
        self._cache = _ReadRangeCache(filesystem, shards, capacity)

    def get_file_info(self, paths: list[str]) -> list[FileInfo]:
        # The pinned manifest already identifies these immutable files and their sizes.
        return [FileInfo(path, FileType.File, size=self._sizes[path]) for path in paths]

    def open_input_file(self, path: str) -> pa.PythonFile:
        return pa.PythonFile(_CachedReadFile(self.fs, path, self._sizes[path], self._cache), mode="r")

    def open_input_stream(self, path: str) -> pa.PythonFile:
        return self.open_input_file(path)


@dataclass(frozen=True)
class _ScanProfile:
    batch_rows: int
    batch_readahead: int
    fragment_readahead: int
    pre_buffer: bool = True


_TABLE_SCAN_PROFILE = _ScanProfile(batch_rows=16_384, batch_readahead=16, fragment_readahead=4)
_BLOB_DESCRIPTOR_SCAN_PROFILE = _ScanProfile(batch_rows=64, batch_readahead=1, fragment_readahead=1, pre_buffer=False)
_BLOB_PART_SCAN_PROFILE = _ScanProfile(batch_rows=1, batch_readahead=1, fragment_readahead=1, pre_buffer=False)
_ArrowRows = TypeVar("_ArrowRows", pa.Table, pa.RecordBatch)


@dataclass(frozen=True)
class VersionedRow:
    """One row tagged with its key and manifest ordering coordinates."""

    key: tuple
    commit_sequence: int
    sequence: int
    generation: int
    row: dict


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


def _validated_blob_parts(descriptor: BlobDescriptor, rows: Iterable[Mapping[str, object]]) -> Iterator[bytes]:
    """Yield a descriptor's bytes while checking its committed part and size metadata."""
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
    for row in rows:
        part = row.get(BlobColumns.PART)
        if not isinstance(part, int):
            raise BlobCorruptionError(f"blob {name!r} has invalid part number {part!r}")
        if part >= part_count:
            break
        if part != expected_part:
            raise BlobCorruptionError(f"blob {name!r} is missing part {expected_part}")
        data = row.get(BlobColumns.DATA)
        if not isinstance(data, bytes):
            raise BlobCorruptionError(f"blob {name!r} part {part} has invalid data")
        total_bytes += len(data)
        expected_part += 1
        yield data
    if expected_part != part_count:
        raise BlobCorruptionError(f"blob {name!r} has {expected_part} of {part_count} parts")
    if descriptor.size is not None and total_bytes != descriptor.size:
        raise BlobCorruptionError(f"blob {name!r} declares {descriptor.size} bytes but stores {total_bytes}")


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

    @property
    def key_range(self) -> KeyRange | None: ...


_SCHEMA_CACHE_SIZE = 8192


# Shard paths are unique and their contents immutable, so one process-wide cache keyed by path
# alone serves every view and compaction without invalidation; the size bound only limits memory.
# A miss reads the footer through the caller's filesystem so its range cache and transfer
# metering apply.
_schema_cache: OrderedDict[str, pa.Schema] = OrderedDict()
_schema_cache_lock = threading.Lock()


def shard_schema(path: str, filesystem: PyFileSystem) -> pa.Schema:
    """Return the Arrow schema of an immutable shard, served from the process cache when present."""
    with _schema_cache_lock:
        cached = _schema_cache.get(path)
        if cached is not None:
            _schema_cache.move_to_end(path)
            return cached
    schema = pq.read_schema(path, filesystem=filesystem)
    with _schema_cache_lock:
        _schema_cache[path] = schema
        while len(_schema_cache) > _SCHEMA_CACHE_SIZE:
            _schema_cache.popitem(last=False)
    return schema


@dataclass(frozen=True)
class FilterClause:
    """One ``where`` clause, ``column <operator> value``, with ``==``, ``!=``, or ``in``."""

    column: str
    operator: str
    value: object

    @classmethod
    def parse(cls, clause: tuple[str, str, object]) -> FilterClause:
        column, operator, value = clause
        if operator not in _SUPPORTED_OPS:
            raise ValueError(f"unsupported filter op {operator!r}; expected one of {sorted(_SUPPORTED_OPS)}")
        return cls(column, operator, value)

    def expression(self) -> pds.Expression:
        field = pds.field(self.column)
        if self.operator == "in":
            return field.isin(self.value)
        return field == self.value if self.operator == "==" else field != self.value

    def matches(self, row: Mapping[str, object]) -> bool:
        candidate = row.get(self.column)
        if candidate is None:
            return False
        if self.operator == "in":
            return candidate in cast(Iterable[object], self.value)
        return (candidate == self.value) == (self.operator == "==")


def shard_may_contain(shard: _ReadableShard, primary_key: tuple[str, ...], clauses: Sequence[FilterClause]) -> bool:
    """Whether ``clauses`` can match a row inside the shard's recorded primary-key range.

    Equality clauses on a prefix of the primary key are compared as a tuple against the
    range, and a membership clause on the first key column is checked value by value. An
    unbounded shard always matches.
    """
    key_range = shard.key_range
    if key_range is None:
        return True
    equalities = {clause.column: clause.value for clause in clauses if clause.operator == "=="}
    prefix = tuple(equalities[column] for column in itertools.takewhile(equalities.__contains__, primary_key))
    if prefix:
        return key_range.may_contain_prefix(prefix)
    for clause in clauses:
        if clause.column == primary_key[0] and clause.operator == "in":
            return key_range.may_contain_first(cast(Iterable[object], clause.value))
    return True


@dataclass(frozen=True)
class _ReadPlan:
    shards: tuple[_ReadableShard, ...]
    primary_key: tuple[str, ...]
    pushdown_where: list[FilterClause]
    post_dedup_where: list[FilterClause]
    filesystem: PyFileSystem
    schema: pa.Schema
    columns: list[str] | None


def _build_filter(clauses: Sequence[FilterClause] | None) -> pds.Expression | None:
    if not clauses:
        return None
    return functools.reduce(lambda expr, clause: expr & clause.expression(), clauses[1:], clauses[0].expression())


def _partition_filter(
    where: list[tuple[str, str, object]] | None, primary_key: tuple[str, ...]
) -> tuple[list[FilterClause], list[FilterClause]]:
    """Split ``where`` into clauses pushed down to shard scans and clauses applied after deduplication."""
    primary_key_columns = set(primary_key)
    pushdown = []
    after_deduplication = []
    for clause in map(FilterClause.parse, where or []):
        target = pushdown if clause.column in primary_key_columns else after_deduplication
        target.append(clause)
    return pushdown, after_deduplication


def iter_shard_batches(
    shard: _ReadableShard,
    unified: pa.Schema,
    pa_fs: PyFileSystem,
    columns: list[str] | None = None,
    where: Sequence[FilterClause] | None = None,
    scan_profile: _ScanProfile = _TABLE_SCAN_PROFILE,
) -> Generator[pa.RecordBatch, None, None]:
    """Yield shard batches with resolved commit and generation coordinates."""
    dataset = pds.dataset([shard.path], filesystem=pa_fs, format="parquet", schema=unified)
    with dataset.scanner(
        columns=columns,
        filter=_build_filter(where),
        use_threads=False,
        batch_size=scan_profile.batch_rows,
        batch_readahead=scan_profile.batch_readahead,
        fragment_readahead=scan_profile.fragment_readahead,
        fragment_scan_options=pds.ParquetFragmentScanOptions(pre_buffer=scan_profile.pre_buffer),
    ).to_reader() as batches:
        for batch in batches:
            yield _resolved_version_columns(batch, shard.commit_sequence, shard.generation)


def iter_shard_row_groups(
    shard: _ReadableShard,
    unified: pa.Schema,
    pa_fs: PyFileSystem,
) -> Generator[pa.Table, None, None]:
    """Yield one resolved Arrow table per Parquet row group."""
    dataset = pds.dataset([shard.path], filesystem=pa_fs, format="parquet", schema=unified)
    fragment = next(dataset.get_fragments())
    assert isinstance(fragment, pds.ParquetFileFragment)
    fragment.ensure_complete_metadata()
    assert fragment.metadata is not None
    for index in range(fragment.metadata.num_row_groups):
        row_group = fragment.subset(row_group_ids=[index])
        group = row_group.scanner(
            schema=unified,
            use_threads=False,
            batch_readahead=1,
            fragment_readahead=1,
            fragment_scan_options=pds.ParquetFragmentScanOptions(pre_buffer=False),
        ).to_table()
        if group.num_rows:
            yield _resolved_version_columns(group, shard.commit_sequence, shard.generation)


def _resolved_version_columns(rows: _ArrowRows, commit_sequence: int, generation: int) -> _ArrowRows:
    """Fill missing version coordinates without changing the Arrow row container type."""
    result: pa.Table | pa.RecordBatch = rows
    for name, value, data_type in (
        (SystemColumns.GENERATION, generation, pa.int32()),
        (SystemColumns.COMMIT, commit_sequence, pa.int64()),
    ):
        index = result.schema.get_field_index(name)
        if index < 0:
            result = result.append_column(name, pa.array([value] * result.num_rows, type=data_type))
        elif result.column(index).null_count:
            result = result.set_column(index, name, pc.fill_null(result.column(index), value))
    return cast(_ArrowRows, result)


def iter_shard_rows(
    shard: _ReadableShard,
    unified: pa.Schema,
    primary_key: tuple[str, ...],
    pa_fs: PyFileSystem,
    columns: list[str] | None = None,
    where: Sequence[FilterClause] | None = None,
    scan_profile: _ScanProfile = _TABLE_SCAN_PROFILE,
) -> Generator[VersionedRow, None, None]:
    """Yield rows from one shard in primary-key order with version coordinates."""
    with closing(iter_shard_batches(shard, unified, pa_fs, columns, where, scan_profile)) as batches:
        if shard.primary_key_sorted:
            ordered_batches = batches
        else:
            buffered = list(batches)
            if not buffered:
                return
            sort_columns = [(name, "ascending") for name in primary_key if name in buffered[0].schema.names]
            ordered_batches = (
                pa.Table.from_batches(buffered).sort_by(sort_columns).to_batches(max_chunksize=scan_profile.batch_rows)
            )
        for batch in ordered_batches:
            for row in batch.to_pylist():
                key = tuple((row.get(name) is None, row.get(name)) for name in primary_key)
                generation = row.pop(SystemColumns.GENERATION)
                yield VersionedRow(
                    key=key,
                    commit_sequence=row[SystemColumns.COMMIT],
                    sequence=row.get(SystemColumns.SEQUENCE) or 0,
                    generation=generation,
                    row=row,
                )


def merge_deduplicated_rows(streams: list[Iterator[VersionedRow]]) -> Iterator[dict]:
    """Merge sorted shard streams and retain the newest row for each key."""
    merged = heapq.merge(*streams, key=lambda item: item.key)
    for _key, group in itertools.groupby(merged, key=lambda item: item.key):
        winner: VersionedRow | None = None
        for item in group:
            if winner is None or (item.commit_sequence, item.sequence, item.generation) > (
                winner.commit_sequence,
                winner.sequence,
                winner.generation,
            ):
                winner = item
        assert winner is not None
        yield winner.row


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
    if pushdown_where:
        shards = tuple(shard for shard in shards if shard_may_contain(shard, primary_key, pushdown_where))
        if not shards:
            return None
    schema = pa.unify_schemas(
        [shard_schema(shard.path, filesystem) for shard in shards],
        promote_options="permissive",
    )
    read_columns = None
    if columns is not None:
        filter_columns = {clause.column for clause in post_dedup_where}
        needed = (
            set(columns)
            | set(primary_key)
            | filter_columns
            | {
                SystemColumns.SEQUENCE,
                SystemColumns.COMMIT,
                SystemColumns.GENERATION,
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
                fragment_scan_options=pds.ParquetFragmentScanOptions(pre_buffer=scan_profile.pre_buffer),
                use_threads=False,
            ).to_table()
        parts.append(_resolved_version_columns(part, commit_sequence, generation))

    combined = parts[0] if len(parts) == 1 else pa.concat_tables(parts, promote_options="permissive")
    if all(name in combined.column_names for name in plan.primary_key):
        combined = _deduplicate(combined, plan.primary_key)
    if plan.post_dedup_where:
        combined = pds.dataset(combined).to_table(filter=_build_filter(plan.post_dedup_where))
    if columns is not None:
        combined = combined.select([name for name in columns if name in combined.column_names])
    return combined


def _blob_shard_descriptors(
    filesystem: PyFileSystem, primary_key: tuple[str, ...], names: Sequence[str], shard: _ReadableShard
) -> pa.Table:
    plan = _read_plan(filesystem, (shard,), primary_key, None, [(BlobColumns.NAME, "in", list(names))])
    assert plan is not None
    return _scan_plan(plan, None, scan_profile=_BLOB_DESCRIPTOR_SCAN_PROFILE)


class _ReadOperations:
    """Read operations shared by manifest and legacy listing snapshots."""

    def __init__(self) -> None:
        self._read_diagnostics = BlobReadDiagnostics()
        self._read_diagnostics_lock = threading.Lock()

    def read_diagnostics(self) -> BlobReadDiagnostics:
        """Return a snapshot of cumulative ``read_blobs`` work on this view."""
        with self._read_diagnostics_lock:
            return replace(self._read_diagnostics)

    def _record_blob_read(self, diagnostics: BlobReadDiagnostics) -> None:
        with self._read_diagnostics_lock:
            self._read_diagnostics.add(diagnostics)
        _BLOB_READ_CALLS.add(diagnostics.read_calls)
        _BLOB_REQUESTED_NAMES.add(diagnostics.requested_names)
        _BLOB_FOUND_NAMES.add(diagnostics.found_names)
        _BLOB_RETURNED_BYTES.add(diagnostics.bytes_returned)
        _BLOB_DESCRIPTOR_LOOKUPS.add(diagnostics.descriptor_lookups)
        _BLOB_DESCRIPTOR_SECONDS.record(diagnostics.descriptor_seconds)
        _BLOB_PAYLOAD_SECONDS.record(diagnostics.payload_seconds)
        _BLOB_SELECTED_SHARDS.record(diagnostics.selected_shards)

    root: str

    def primary_key(self, table: str) -> tuple[str, ...]:
        raise NotImplementedError

    def list_shards(self, table: str) -> Sequence[_ReadableShard]:
        raise NotImplementedError

    def _filesystem(self, table: str) -> PyFileSystem:
        fs, _ = factory.url_to_fs(self.root)
        handler = _NoReadAheadHandler(fs) if table in (BlobTables.DESCRIPTORS, BlobTables.PARTS) else _ShardHandler(fs)
        return PyFileSystem(handler)

    def _read_plan(
        self,
        table: str,
        columns: Sequence[str] | None,
        where: list[tuple[str, str, object]] | None,
    ) -> _ReadPlan | None:
        shards = tuple(self.list_shards(table))
        if not shards:
            return None
        return _read_plan(self._filesystem(table), shards, self.primary_key(table), columns, where)

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
    ) -> Generator[dict, None, None]:
        plan = self._read_plan(table, columns, where)
        if plan is None:
            return
        with ExitStack() as stack:
            streams = [
                stack.enter_context(
                    closing(
                        iter_shard_rows(
                            shard,
                            plan.schema,
                            plan.primary_key,
                            plan.filesystem,
                            plan.columns,
                            plan.pushdown_where,
                            scan_profile,
                        )
                    )
                )
                for shard in plan.shards
            ]
            for row in merge_deduplicated_rows(streams):
                if not all(clause.matches(row) for clause in plan.post_dedup_where):
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
        return self.read_blobs((name,)).get(name)

    def read_blobs(self, names: Sequence[str]) -> dict[str, bytes]:
        """Read named blobs, omitting absent names."""
        diagnostics = BlobReadDiagnostics(read_calls=1, requested_names=len(set(names)))
        values: dict[str, bytes] = {}
        try:
            if not names:
                return {}
            rows = self._blob_descriptors(names, diagnostics=diagnostics)
            if rows is None:
                return {}
            started = time.monotonic()
            chunked: dict[str, BlobDescriptor] = {}
            try:
                for row in rows.to_pylist():
                    descriptor = BlobDescriptor.from_row(row)
                    if descriptor.part_count is None:
                        values[descriptor.name] = b"".join(_validated_blob_parts(descriptor, ()))
                    else:
                        chunked[descriptor.name] = descriptor
                if chunked:
                    with closing(
                        self._iter_rows(
                            BlobTables.PARTS,
                            columns=[BlobColumns.NAME, BlobColumns.PART, BlobColumns.DATA],
                            where=[(BlobColumns.NAME, "in", list(chunked))],
                            scan_profile=_BLOB_PART_SCAN_PROFILE,
                        )
                    ) as part_rows:
                        for name, group in itertools.groupby(part_rows, key=lambda row: row[BlobColumns.NAME]):
                            values[name] = b"".join(_validated_blob_parts(chunked[name], group))
                    for name, descriptor in chunked.items():
                        if name not in values:
                            values[name] = b"".join(_validated_blob_parts(descriptor, ()))
            finally:
                diagnostics.payload_seconds += time.monotonic() - started
            diagnostics.found_names = len(values)
            diagnostics.bytes_returned = sum(len(value) for value in values.values())
            return values
        finally:
            self._record_blob_read(diagnostics)

    def _blob_descriptors(
        self, names: Sequence[str], *, diagnostics: BlobReadDiagnostics | None = None
    ) -> pa.Table | None:
        started = time.monotonic()
        if diagnostics is not None:
            diagnostics.descriptor_lookups += 1
        try:
            if not self.list_shards(BlobTables.DESCRIPTORS):
                return None
            primary_key = self.primary_key(BlobTables.DESCRIPTORS)
            membership = [FilterClause(BlobColumns.NAME, "in", list(names))]
            shards = tuple(
                shard
                for shard in self.list_shards(BlobTables.DESCRIPTORS)
                if shard_may_contain(shard, primary_key, membership)
            )
            if not shards:
                return None
            if diagnostics is not None:
                diagnostics.selected_shards += len(shards)
            filesystem = self._filesystem(BlobTables.DESCRIPTORS)
            # Read footers and matching rows together, with bounded concurrent decoding.
            # Version resolution spans every shard whose key bounds can hold a requested name.
            with ThreadPoolExecutor(max_workers=min(MAX_DESCRIPTOR_READS, len(shards))) as executor:
                parts = list(
                    executor.map(lambda shard: _blob_shard_descriptors(filesystem, primary_key, names, shard), shards)
                )
            return _deduplicate(pa.concat_tables(parts, promote_options="permissive"), primary_key)
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
        if descriptor.part_count is None:
            yield from _validated_blob_parts(descriptor, ())
            return
        with closing(
            self._iter_rows(
                BlobTables.PARTS,
                columns=[BlobColumns.PART, BlobColumns.DATA],
                where=[(BlobColumns.NAME, "==", descriptor.name)],
                scan_profile=_BLOB_PART_SCAN_PROFILE,
            )
        ) as part_rows:
            yield from _validated_blob_parts(descriptor, part_rows)

    def resolve(self, uri: str) -> bytes | None:
        """Resolve a blob URI, returning ``None`` when absent and rejecting unsupported references."""
        ref = parse_uri(uri)
        if ref is None:
            raise ValueError(f"not a finestore:// reference: {uri!r}")
        if ref.table != BlobTables.DESCRIPTORS:
            raise ValueError(f"finestore:// resolution supports the blobs table only, got {ref.table!r}")
        return self.read_blob(ref.key)


class ReadView(_ReadOperations):
    """A read-only archive view pinned to one commit token."""

    def __init__(
        self, root: str, snapshot: ArchiveSnapshot | None = None, *, cache_bytes: int = _READ_CACHE_BYTES
    ) -> None:
        super().__init__()
        if cache_bytes < 0:
            raise ValueError("cache_bytes must be nonnegative")
        self.root = root
        self._layout = FineStoreLayout(self.root)
        self._snapshot = snapshot or read_snapshot(self._layout)
        self._meta_cache: dict[str, TableMetadata] = {}
        self._read_filesystem: PyFileSystem | None = None
        self._cache_bytes = cache_bytes
        self._filesystem_lock = threading.Lock()

    def _filesystem(self, table: str) -> PyFileSystem:
        if not self._cache_bytes:
            return super()._filesystem(table)
        with self._filesystem_lock:
            if self._read_filesystem is None:
                fs, _ = factory.url_to_fs(self.root)
                shards = [shard for state in self._snapshot.manifest.tables.values() for shard in state.shards]
                self._read_filesystem = PyFileSystem(_CachedReadHandler(fs, shards, self._cache_bytes))
            return self._read_filesystem

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
