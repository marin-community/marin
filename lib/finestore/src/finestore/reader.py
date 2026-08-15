# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Snapshot reads over the immutable shards selected by one FineStore manifest."""

from __future__ import annotations

import heapq
import itertools
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pds
import pyarrow.parquet as pq
from pyarrow.fs import FSSpecHandler, PyFileSystem
from rigging.filesystem import StoragePath, factory

from finestore.commit import ArchiveSnapshot, read_snapshot, validate_archive
from finestore.layout import (
    BLOB_DATA_COLUMN,
    BLOB_NAME_COLUMN,
    BLOBS_TABLE,
    COMMIT_COLUMN,
    GEN_COLUMN,
    SEQ_COLUMN,
    CommitToken,
    FineStoreLayout,
    SealMarker,
    Shard,
    TableMetadata,
    parse_uri,
)

_SUPPORTED_OPS = frozenset({"==", "!=", "in"})
_STREAM_BATCH_ROWS = 16_384


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
    shard: Shard,
    unified: pa.Schema,
    primary_key: tuple[str, ...],
    pa_fs,
    columns: list[str] | None = None,
    where: list[tuple[str, str, object]] | None = None,
) -> Iterator[VersionedRow]:
    """Yield rows from one shard in primary-key order with version coordinates."""
    dataset = pds.dataset([shard.path], filesystem=pa_fs, format="parquet", schema=unified)
    if shard.primary_key_sorted:
        batches = dataset.scanner(
            columns=columns,
            filter=_build_filter(where),
            use_threads=False,
            batch_size=_STREAM_BATCH_ROWS,
        ).to_batches()
    else:
        source = dataset.to_table(columns=columns, filter=_build_filter(where))
        sort_columns = [(name, "ascending") for name in primary_key if name in source.column_names]
        batches = source.sort_by(sort_columns).to_batches(max_chunksize=_STREAM_BATCH_ROWS)
    for batch in batches:
        for row in batch.to_pylist():
            commit_sequence = row.get(COMMIT_COLUMN)
            if commit_sequence is None:
                commit_sequence = shard.commit_sequence
            row[COMMIT_COLUMN] = commit_sequence
            key = tuple((row.get(name) is None, row.get(name)) for name in primary_key)
            yield VersionedRow(
                key=key,
                commit_sequence=commit_sequence,
                sequence=row.get(SEQ_COLUMN) or 0,
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


class ReadView:
    """A read-only archive view pinned to one commit token."""

    def __init__(self, root: str, snapshot: ArchiveSnapshot | None = None) -> None:
        self.root = root
        self._layout = FineStoreLayout(self.root)
        archive_metadata = validate_archive(self._layout)
        self._snapshot = snapshot or read_snapshot(self._layout)
        if archive_metadata is None and self._snapshot.token is not None:
            raise ValueError(f"archive at {self.root} has HEAD but no format marker")
        self._meta_cache: dict[str, TableMetadata] = {}

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

    def scan(
        self,
        table: str,
        *,
        columns: Sequence[str] | None = None,
        where: list[tuple[str, str, object]] | None = None,
    ) -> pa.Table | None:
        """Read this view's deduplicated table, or ``None`` when it has no active shards."""
        shards = self.list_shards(table)
        if not shards:
            return None
        primary_key = self.primary_key(table)
        pushdown_where, post_dedup_where = _partition_filter(where, primary_key)
        fs, _ = factory.url_to_fs(self.root)
        pa_fs = PyFileSystem(FSSpecHandler(fs))
        unified = pa.unify_schemas(
            [pq.read_schema(shard.path, filesystem=pa_fs) for shard in shards], promote_options="permissive"
        )

        read_columns = None
        if columns is not None:
            filter_columns = {name for name, _operator, _value in post_dedup_where}
            needed = set(columns) | set(primary_key) | filter_columns | {SEQ_COLUMN, COMMIT_COLUMN}
            read_columns = [name for name in unified.names if name in needed]

        by_version: dict[tuple[int, int], list[str]] = defaultdict(list)
        for shard in shards:
            by_version[(shard.commit_sequence, shard.generation)].append(shard.path)

        parts: list[pa.Table] = []
        for (commit_sequence, generation), paths in sorted(by_version.items()):
            dataset = pds.dataset(paths, filesystem=pa_fs, format="parquet", schema=unified)
            part = dataset.to_table(columns=read_columns, filter=_build_filter(pushdown_where))
            part = part.append_column(GEN_COLUMN, pa.array([generation] * part.num_rows, pa.int32()))
            commit_values = pa.array([commit_sequence] * part.num_rows, pa.int64())
            if COMMIT_COLUMN in part.column_names:
                commit_index = part.schema.get_field_index(COMMIT_COLUMN)
                part = part.set_column(commit_index, COMMIT_COLUMN, pc.coalesce(part[COMMIT_COLUMN], commit_values))
            else:
                part = part.append_column(COMMIT_COLUMN, commit_values)
            parts.append(part)

        combined = parts[0] if len(parts) == 1 else pa.concat_tables(parts, promote_options="permissive")
        if all(name in combined.column_names for name in primary_key):
            combined = _deduplicate(combined, primary_key)
        if post_dedup_where:
            combined = pds.dataset(combined).to_table(filter=_build_filter(post_dedup_where))
        if columns is not None:
            combined = combined.select([name for name in columns if name in combined.column_names])
        return combined

    def iter_rows(
        self,
        table: str,
        *,
        columns: Sequence[str] | None = None,
        where: list[tuple[str, str, object]] | None = None,
    ) -> Iterator[dict]:
        """Yield deduplicated rows from this view in primary-key order."""
        shards = self.list_shards(table)
        if not shards:
            return
        primary_key = self.primary_key(table)
        pushdown_where, post_dedup_where = _partition_filter(where, primary_key)
        fs, _ = factory.url_to_fs(self.root)
        pa_fs = PyFileSystem(FSSpecHandler(fs))
        unified = pa.unify_schemas(
            [pq.read_schema(shard.path, filesystem=pa_fs) for shard in shards], promote_options="permissive"
        )
        read_columns = None
        if columns is not None:
            filter_columns = {name for name, _operator, _value in where or []}
            needed = set(columns) | set(primary_key) | filter_columns | {SEQ_COLUMN, COMMIT_COLUMN}
            read_columns = [name for name in unified.names if name in needed]
        streams = [iter_shard_rows(shard, unified, primary_key, pa_fs, read_columns, pushdown_where) for shard in shards]
        for merged in merge_deduplicated_rows(streams):
            row = merged.row
            if not _matches_filter(row, post_dedup_where):
                continue
            if columns is not None:
                row = {name: row[name] for name in columns if name in row}
            yield row

    def point(self, table: str, **keys) -> dict | None:
        result = self.scan(table, where=[(key, "==", value) for key, value in keys.items()])
        if result is None or result.num_rows == 0:
            return None
        return result.slice(0, 1).to_pylist()[0]

    def max_seq(self, table: str) -> int:
        """Return the greatest sequence in active shards, or ``-1`` when there are none."""
        shards = self.list_shards(table)
        return max((shard.max_seq for shard in shards), default=-1)

    def keys(self, table: str) -> set[tuple]:
        state = self._snapshot.manifest.tables.get(table)
        if state is None or not state.shards:
            return set()
        primary_key = self.primary_key(table)
        result = self.scan(table, columns=list(primary_key))
        if result is None:
            return set()
        values = [result.column(name).to_pylist() for name in primary_key]
        return set(zip(*values, strict=True))

    def read_blob(self, name: str) -> bytes | None:
        row = self.point(BLOBS_TABLE, **{BLOB_NAME_COLUMN: name})
        if row is None:
            return None
        data = row.get(BLOB_DATA_COLUMN)
        return bytes(data) if data is not None else None

    def resolve(self, uri: str) -> bytes | None:
        ref = parse_uri(uri)
        if ref is None:
            raise ValueError(f"not a finestore:// reference: {uri!r}")
        if ref.table != BLOBS_TABLE:
            raise ValueError(f"finestore:// resolution supports the blobs table only, got {ref.table!r}")
        return self.read_blob(ref.key)

    def list_shards(self, table: str) -> list[Shard]:
        state = self._snapshot.manifest.tables.get(table)
        return [] if state is None else list(state.shards)


def _deduplicate(table: pa.Table, primary_key: tuple[str, ...]) -> pa.Table:
    """Keep the latest committed row per primary key."""
    if table.num_rows == 0:
        return table
    key_columns = [table.column(name).to_pylist() for name in primary_key]
    commits = table.column(COMMIT_COLUMN).to_pylist()
    generations = table.column(GEN_COLUMN).to_pylist()
    sequences = table.column(SEQ_COLUMN).to_pylist() if SEQ_COLUMN in table.column_names else [0] * table.num_rows
    order = sorted(range(table.num_rows), key=lambda index: (commits[index], sequences[index], generations[index]))
    winners: dict[tuple, int] = {}
    for index in order:
        winners[tuple(column[index] for column in key_columns)] = index
    keep = sorted(winners.values())
    return table if len(keep) == table.num_rows else table.take(pa.array(keep, pa.int64()))
