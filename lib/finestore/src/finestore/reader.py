# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The reader: compose every writer's Parquet shards for a table, deduplicate, and project.

A read lists the table directory, unifies the shards' self-describing schemas (so a column a run
added later is null for older shards), reads only the projected columns, and deduplicates by the
table's primary key — keeping, for each key, the row from the highest compaction generation and then
the highest ``_seq``. That single rule makes a crash mid-compaction (level-0 and its merge both
present), a duplicate delivery, and a retried flush all converge to one row without any manifest.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from collections.abc import Sequence

import pyarrow as pa
import pyarrow.dataset as pds
import pyarrow.parquet as pq
from pyarrow.fs import FSSpecHandler, PyFileSystem
from rigging.filesystem import StoragePath, url_to_fs

from finestore.layout import (
    BLOB_NAME_COLUMN,
    BLOBS_TABLE,
    GEN_COLUMN,
    SEQ_COLUMN,
    WRITER_COLUMN,
    Shard,
    archive_dir,
    parse_shard_path,
    parse_uri,
    schema_path,
    sealed_path,
)

logger = logging.getLogger(__name__)

# Deduplication falls back to this always-unique key when a table declares no primary key: it still
# collapses a re-published shard (same writer + seq) but not a duplicate domain delivery.
_DEFAULT_PRIMARY_KEY = (WRITER_COLUMN, SEQ_COLUMN)

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
        self._meta_cache: dict[str, dict] = {}

    # -- discovery ---------------------------------------------------------------------------------

    def _list_shards(self, fs, table: str) -> list[Shard]:
        base = f"{archive_dir(self.root)}/{table}"
        _fs, base_key = url_to_fs(base)
        try:
            keys = fs.find(base_key)
        except FileNotFoundError:
            return []
        shards = [parse_shard_path(key) for key in keys]
        return [shard for shard in shards if shard is not None]

    def primary_key(self, table: str) -> tuple[str, ...]:
        """The table's dedup primary key, read from ``_schema.json`` (or the default fallback)."""
        meta = self._meta(table)
        pk = meta.get("primary_key") if meta else None
        return tuple(pk) if pk else _DEFAULT_PRIMARY_KEY

    def _meta(self, table: str) -> dict | None:
        if table in self._meta_cache:
            return self._meta_cache[table]
        # An absent _schema.json means the table was written without registered metadata; fall back
        # to the default primary key. A present-but-corrupt file is a real fault and propagates.
        try:
            text = StoragePath(schema_path(self.root, table)).read_text()
        except FileNotFoundError:
            meta = {}
        else:
            meta = json.loads(text)
        self._meta_cache[table] = meta
        return meta

    def is_sealed(self) -> bool:
        return StoragePath(sealed_path(self.root)).exists()

    # -- scan / point ------------------------------------------------------------------------------

    def scan(
        self,
        table: str,
        *,
        columns: Sequence[str] | None = None,
        where: list[tuple[str, str, object]] | None = None,
        primary_key: Sequence[str] | None = None,
        dedup: bool = True,
    ) -> pa.Table | None:
        """Return the deduplicated rows of ``table``, projected to ``columns`` and filtered by ``where``.

        ``where`` is a list of ``(column, op, value)`` conditions (``==``, ``!=``, ``in``) ANDed
        together and pushed into the Parquet scan. Returns ``None`` when the table has no shards. When
        ``columns`` is given, only those column chunks are read (fat columns a query does not select
        are never fetched), plus whatever the deduplication key needs; the result is projected back to
        ``columns``.
        """
        fs, _ = url_to_fs(self.root)
        shards = self._list_shards(fs, table)
        if not shards:
            return None
        pk = tuple(primary_key) if primary_key is not None else self.primary_key(table)
        pa_fs = PyFileSystem(FSSpecHandler(fs))

        schemas = [pq.read_schema(shard.path, filesystem=pa_fs) for shard in shards]
        unified = pa.unify_schemas(schemas, promote_options="permissive")

        read_columns = None
        if columns is not None:
            needed = set(columns) | set(pk) | {SEQ_COLUMN}
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
        if dedup and all(name in combined.column_names for name in pk):
            combined = _deduplicate(combined, pk)
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

    # -- blobs / references ------------------------------------------------------------------------

    def read_blob(self, name: str) -> bytes | None:
        """Return the inline bytes of the blob named ``name``, or ``None`` if absent."""
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
        """Every shard of ``table`` (used by compaction)."""
        fs, _ = url_to_fs(self.root)
        return self._list_shards(fs, table)


def _deduplicate(table: pa.Table, primary_key: tuple[str, ...]) -> pa.Table:
    """Keep one row per primary key: the highest ``(_gen, _seq)`` wins."""
    num_rows = table.num_rows
    if num_rows == 0:
        return table
    key_columns = [table.column(name).to_pylist() for name in primary_key]
    generations = table.column(GEN_COLUMN).to_pylist()
    sequences = table.column(SEQ_COLUMN).to_pylist() if SEQ_COLUMN in table.column_names else [0] * num_rows
    order = sorted(range(num_rows), key=lambda i: (generations[i], sequences[i]))
    winners: dict[tuple, int] = {}
    for i in order:
        winners[tuple(column[i] for column in key_columns)] = i
    keep = sorted(winners.values())
    if len(keep) == num_rows:
        return table
    return table.take(pa.array(keep, pa.int64()))
