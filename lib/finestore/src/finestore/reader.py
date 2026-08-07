# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The reader: compose every writer's Parquet shards for a table, deduplicate, and project.

A read lists the table directory, unifies the shards' self-describing schemas (so a column a run
added later is null for older shards), reads only the projected columns, and collapses duplicates by
the table's primary key — keeping, for each key, the row with the highest ``_seq`` and then the highest
compaction generation. Because a writer resumes its ``_seq`` above every persisted row, a later write
always outranks an earlier one, so nothing can shadow it; the generation only breaks the exact-``_seq``
tie a compaction creates when it re-emits a row unchanged. That single rule makes a crash
mid-compaction (level-0 and its merge both present), a duplicate delivery, and a retried flush all
converge to one row without any manifest.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence

import pyarrow as pa
import pyarrow.dataset as pds
import pyarrow.parquet as pq
from pyarrow.fs import FSSpecHandler, PyFileSystem
from rigging.filesystem import StoragePath, factory

from finestore.layout import (
    BLOB_NAME_COLUMN,
    BLOBS_TABLE,
    FORMAT_VERSION,
    GEN_COLUMN,
    SEQ_COLUMN,
    ArchiveMetadata,
    FineStoreLayout,
    Shard,
    TableMetadata,
    parse_shard_path,
    parse_uri,
)

logger = logging.getLogger(__name__)

_SUPPORTED_OPS = frozenset({"==", "!=", "in"})


def _build_filter(where: list[tuple[str, str, object]] | None) -> pds.Expression | None:
    """Translate ``where`` — a list of ``(col, op, val)`` conditions ANDed together — to a filter."""
    if not where:
        return None
    expr: pds.Expression | None = None
    for col, op, val in where:
        if op not in _SUPPORTED_OPS:
            raise ValueError(f"unsupported filter op {op!r}; expected one of {sorted(_SUPPORTED_OPS)}")
        field = pds.field(col)
        clause = field.isin(val) if op == "in" else (field == val if op == "==" else field != val)
        expr = clause if expr is None else expr & clause
    return expr


class CompositeReader:
    """A read-only view over an archive that composes and deduplicates all of its shards."""

    def __init__(self, root: str) -> None:
        self.root = root.rstrip("/")
        self._layout = FineStoreLayout(self.root)
        self._meta_cache: dict[str, TableMetadata] = {}
        self._archive_meta: ArchiveMetadata | None = None

    # -- discovery ---------------------------------------------------------------------------------

    def _list_shards(self, fs, table: str) -> list[Shard]:
        _fs, base_key = factory.url_to_fs(self._layout.table_dir(table))
        try:
            keys = fs.find(base_key)
        except FileNotFoundError:
            return []
        shards = [parse_shard_path(key) for key in keys]
        return [shard for shard in shards if shard is not None]

    def primary_key(self, table: str) -> tuple[str, ...]:
        """The columns identifying a row of ``table``, read from its ``_schema.json``."""
        return tuple(self._meta(table).primary_key)

    def schema_version(self, table: str) -> int | None:
        """The caller's logical schema version for ``table``, or ``None`` if it holds no shards.

        A writer compares this against its own contract version to tell a fresh archive from one
        whose rows predate the current row shape. ``None`` means only that the table has no shards:
        a table that has them but no ``_schema.json`` raises, because reading a corrupt archive as
        "fresh" would skip the rebuild a contract change needs.
        """
        fs, _ = factory.url_to_fs(self.root)
        if not self._list_shards(fs, table):
            return None
        return self._meta(table).schema_version

    def _meta(self, table: str) -> TableMetadata:
        if table in self._meta_cache:
            return self._meta_cache[table]
        # The store writes _schema.json when it registers a table, before any of that table's shards.
        # So a table with shards always has one; a missing file is a corrupt or foreign archive and
        # propagates rather than silently defaulting the primary key (which would mis-deduplicate).
        meta = TableMetadata.model_validate_json(StoragePath(self._layout.schema_path(table)).read_text())
        self._meta_cache[table] = meta
        return meta

    def archive_metadata(self) -> ArchiveMetadata:
        """The archive-wide metadata the writer stamped at open (the on-disk format version)."""
        if self._archive_meta is None:
            text = StoragePath(self._layout.archive_path).read_text()
            self._archive_meta = ArchiveMetadata.model_validate_json(text)
        return self._archive_meta

    def _check_format(self) -> None:
        """Refuse an archive written in a newer on-disk format than this reader understands."""
        found = self.archive_metadata().format_version
        if found > FORMAT_VERSION:
            raise ValueError(
                f"archive at {self.root} is finestore format v{found}; this reader supports up to v{FORMAT_VERSION}"
            )

    def is_sealed(self) -> bool:
        return StoragePath(self._layout.sealed_path).exists()

    # -- scan / point ------------------------------------------------------------------------------

    def scan(
        self,
        table: str,
        *,
        columns: Sequence[str] | None = None,
        where: list[tuple[str, str, object]] | None = None,
    ) -> pa.Table | None:
        """Return the deduplicated rows of ``table``, projected to ``columns`` and filtered by ``where``.

        ``where`` is a list of ``(column, op, value)`` conditions (``==``, ``!=``, ``in``) ANDed
        together and pushed into the Parquet scan. Returns ``None`` when the table has no shards. When
        ``columns`` is given, only those column chunks are read (fat columns a query does not select
        are never fetched), plus whatever the primary key needs; the result is projected back to
        ``columns``.
        """
        fs, _ = factory.url_to_fs(self.root)
        shards = self._list_shards(fs, table)
        if not shards:
            return None
        self._check_format()
        key = self.primary_key(table)
        pa_fs = PyFileSystem(FSSpecHandler(fs))

        schemas = [pq.read_schema(shard.path, filesystem=pa_fs) for shard in shards]
        unified = pa.unify_schemas(schemas, promote_options="permissive")

        read_columns = None
        if columns is not None:
            needed = set(columns) | set(key) | {SEQ_COLUMN}
            read_columns = [name for name in unified.names if name in needed]

        filter_expr = _build_filter(where)
        by_generation: dict[int, list[str]] = defaultdict(list)
        for shard in shards:
            by_generation[shard.generation].append(shard.path)

        parts: list[pa.Table] = []
        for generation in sorted(by_generation):
            dataset = pds.dataset(by_generation[generation], filesystem=pa_fs, format="parquet", schema=unified)
            part = dataset.to_table(columns=read_columns, filter=filter_expr)
            part = part.append_column(GEN_COLUMN, pa.array([generation] * part.num_rows, pa.int32()))
            parts.append(part)

        combined = parts[0] if len(parts) == 1 else pa.concat_tables(parts, promote_options="permissive")
        if all(name in combined.column_names for name in key):
            combined = _deduplicate(combined, key)
        if columns is not None:
            combined = combined.select([name for name in columns if name in combined.column_names])
        return combined

    def point(self, table: str, **keys) -> dict | None:
        """Return the single row of ``table`` matching every ``key=value``, or ``None``."""
        where = [(key, "==", value) for key, value in keys.items()]
        result = self.scan(table, where=where)
        if result is None or result.num_rows == 0:
            return None
        return result.slice(0, 1).to_pylist()[0]

    # -- resume primitives -------------------------------------------------------------------------

    def max_seq(self, table: str) -> int:
        """The highest ``_seq`` any shard of ``table`` has persisted, or ``-1`` if it has none.

        A resuming writer starts its sequence counter one above this, so a row it appends now outranks
        every row a prior session left behind. Reads only each shard's Parquet footer -- the
        per-row-group ``_seq`` max statistic -- never the column data, so a resume is cheap however
        large the archive.
        """
        fs, _ = factory.url_to_fs(self.root)
        shards = self._list_shards(fs, table)
        if not shards:
            return -1
        pa_fs = PyFileSystem(FSSpecHandler(fs))
        highest = -1
        for shard in shards:
            metadata = pq.read_metadata(shard.path, filesystem=pa_fs)
            if SEQ_COLUMN not in metadata.schema.names:
                continue
            column = metadata.schema.names.index(SEQ_COLUMN)
            for group in range(metadata.num_row_groups):
                stats = metadata.row_group(group).column(column).statistics
                if stats is not None and stats.has_min_max:
                    highest = max(highest, stats.max)
        return highest

    def keys(self, table: str) -> set[tuple]:
        """The deduplicated set of primary-key tuples durably present in ``table``.

        The resume primitive a writer reads to skip work it already committed (e.g. Harbor asking
        which trials are done). Reads only the primary-key columns; returns an empty set for a table
        with no shards.
        """
        fs, _ = factory.url_to_fs(self.root)
        if not self._list_shards(fs, table):
            return set()
        primary_key = self.primary_key(table)
        result = self.scan(table, columns=list(primary_key))
        if result is None:
            return set()
        columns = [result.column(name).to_pylist() for name in primary_key]
        return set(zip(*columns, strict=True))

    # -- blobs / references ------------------------------------------------------------------------

    def read_blob(self, name: str) -> bytes | None:
        """Return the inline bytes of the blob named ``name``, or ``None`` if absent.

        The blobs table is keyed by name, so the ``name ==`` filter this issues prunes the scan to the
        one row group whose footer statistics bracket the name (log-n over a compacted, name-sorted
        shard) rather than reading every blob's payload.
        """
        row = self.point(BLOBS_TABLE, **{BLOB_NAME_COLUMN: name})
        if row is None:
            return None
        data = row.get("data")
        return bytes(data) if data is not None else None

    def resolve(self, uri: str) -> bytes | None:
        """Resolve a ``finestore://<table>/<key>`` reference to its bytes (blobs only, today)."""
        ref = parse_uri(uri)
        if ref is None:
            raise ValueError(f"not a finestore:// reference: {uri!r}")
        if ref.table != BLOBS_TABLE:
            raise ValueError(f"finestore:// resolution supports the blobs table only, got {ref.table!r}")
        return self.read_blob(ref.key)

    def list_shards(self, table: str) -> list[Shard]:
        """Every discovered shard of ``table``, across writers and generations."""
        fs, _ = factory.url_to_fs(self.root)
        return self._list_shards(fs, table)


def _deduplicate(table: pa.Table, primary_key: tuple[str, ...]) -> pa.Table:
    """Keep one row per primary key: the highest ``(_seq, _gen)`` wins.

    ``_seq`` decides — a writer resumes it above every persisted row, so a later write outranks an
    earlier one and cannot be shadowed. Generation only breaks the exact-``_seq`` tie compaction
    leaves when it re-emits a row unchanged.
    """
    num_rows = table.num_rows
    if num_rows == 0:
        return table
    key_columns = [table.column(name).to_pylist() for name in primary_key]
    generations = table.column(GEN_COLUMN).to_pylist()
    sequences = table.column(SEQ_COLUMN).to_pylist() if SEQ_COLUMN in table.column_names else [0] * num_rows
    order = sorted(range(num_rows), key=lambda i: (sequences[i], generations[i]))
    winners: dict[tuple, int] = {}
    for i in order:
        winners[tuple(column[i] for column in key_columns)] = i
    keep = sorted(winners.values())
    if len(keep) == num_rows:
        return table
    return table.take(pa.array(keep, pa.int64()))
