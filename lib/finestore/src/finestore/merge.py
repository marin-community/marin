# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Newest-version selection and sorted-run merging.

Rows order by primary key ascending, with nulls after every value and two nulls equal, and
then by version: a greater ``(_commit, _seq, _gen)`` is newer, and a missing ``_seq``
counts as zero. The newest row of each key is the visible one.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence

import pyarrow as pa
import pyarrow.compute as pc

from finestore.layout import SystemColumns


def _ordering_key_names(schema: pa.Schema, primary_key: Sequence[str]) -> list[str]:
    """Key columns that can differ between rows; an absent or all-null-typed column is null throughout."""
    return [name for name in primary_key if name in schema.names and not pa.types.is_null(schema.field(name).type)]


def newest_indices(rows: pa.Table, primary_key: Sequence[str]) -> pa.Array:
    """Return the index of the newest row for each primary key, in primary-key order."""
    if rows.num_rows == 0:
        return pa.array([], pa.int64())
    key_names = _ordering_key_names(rows.schema, primary_key)
    order_columns: dict[str, pa.ChunkedArray] = {f"key{index}": rows[name] for index, name in enumerate(key_names)}
    sort_keys = [(column, "ascending") for column in order_columns]
    order_columns["commit"] = rows[SystemColumns.COMMIT]
    if SystemColumns.SEQUENCE in rows.column_names:
        order_columns["sequence"] = pc.fill_null(rows[SystemColumns.SEQUENCE], 0)
    order_columns["generation"] = rows[SystemColumns.GENERATION]
    sort_keys.extend((name, "descending") for name in ("commit", "sequence", "generation") if name in order_columns)
    ordering = pa.table(order_columns)
    order = pc.sort_indices(ordering, sort_keys=sort_keys, null_placement="at_end")
    keys = ordering.select([f"key{index}" for index in range(len(key_names))]).take(order)
    return order.filter(_starts_key(keys))


def newest_rows(rows: pa.Table, primary_key: Sequence[str]) -> pa.Table:
    """Return the newest row for each primary key, in primary-key order."""
    return rows.take(newest_indices(rows, primary_key))


def sorted_by_key(tables: Iterable[pa.Table], primary_key: Sequence[str]) -> Iterator[pa.Table]:
    """Gather ``tables`` into one run in merge key order; memory holds all of their rows."""
    rows = list(tables)
    if rows:
        combined = pa.concat_tables(rows)
        sort_keys = [(name, "ascending") for name in _ordering_key_names(combined.schema, primary_key)]
        yield combined.sort_by(sort_keys, null_placement="at_end") if sort_keys else combined


def _starts_key(keys: pa.Table) -> pa.Array:
    """Mark each row of key-sorted ``keys`` whose key differs from the previous row's."""
    count = keys.num_rows
    differs = pa.repeat(pa.scalar(False), count - 1)
    for column in keys.columns:
        values = column.combine_chunks()
        current, previous = values.slice(1), values.slice(0, count - 1)
        equal = pc.fill_null(pc.equal(current, previous), False)
        both_null = pc.and_(pc.is_null(current), pc.is_null(previous))
        differs = pc.or_(differs, pc.invert(pc.or_(equal, both_null)))
    return pa.concat_arrays([pa.array([True]), differs])


def key_values(rows: pa.Table, index: int, primary_key: Sequence[str]) -> tuple:
    """Return a row's primary-key values, with ``None`` for a key column the rows lack."""
    return tuple(rows[name][index].as_py() if name in rows.column_names else None for name in primary_key)


def _ordered(values: Iterable[object]) -> tuple:
    """Make key values compare the way the merge orders keys, nulls after every value."""
    return tuple((value is None, value) for value in values)


def _key_arrays(rows: pa.Table, primary_key: Sequence[str]) -> list[pa.Array]:
    return [
        rows[name].combine_chunks() if name in rows.column_names else pa.nulls(rows.num_rows) for name in primary_key
    ]


def _first_at_or_above(keys: Sequence[pa.Array], count: int, key: tuple) -> int:
    low, high = 0, count
    while low < high:
        middle = (low + high) // 2
        if _ordered(column[middle].as_py() for column in keys) < key:
            low = middle + 1
        else:
            high = middle
    return low


class _RunCursor:
    """Buffers the unmerged rows of one sorted run, reading the run a table at a time.

    The buffer's key columns are kept contiguous so the frontier search probes plain arrays.
    """

    def __init__(self, tables: Iterator[pa.Table], primary_key: Sequence[str]) -> None:
        self._tables = tables
        self._primary_key = primary_key
        self._rows: pa.Table | None = None
        self._keys: list[pa.Array] = []
        self.last_key: tuple = ()
        self.exhausted = False
        self.pull()

    def pull(self) -> None:
        """Append the run's next non-empty table to the buffer, or mark the run exhausted.

        Buffered rows all share the last key when this is called, so they shrink to the newest
        of them first; a key with many versions therefore holds one row, not all of them.
        """
        for table in self._tables:
            if not table.num_rows:
                continue
            if self._rows is not None and self._rows.num_rows:
                held = self._rows if self._rows.num_rows == 1 else newest_rows(self._rows, self._primary_key)
                table = pa.concat_tables([held, table])
            self._rows = table
            self._keys = _key_arrays(table, self._primary_key)
            self.last_key = _ordered(column[table.num_rows - 1].as_py() for column in self._keys)
            return
        self.exhausted = True

    def take_below(self, key: tuple) -> pa.Table | None:
        """Remove and return the buffered rows whose key is below ``key``."""
        if self._rows is None:
            return None
        cut = _first_at_or_above(self._keys, self._rows.num_rows, key)
        if not cut:
            return None
        below, self._rows = self._rows.slice(0, cut), self._rows.slice(cut)
        self._keys = [column.slice(cut) for column in self._keys]
        return below

    def take_rest(self) -> pa.Table | None:
        rest, self._rows = self._rows, None
        return rest if rest is not None and rest.num_rows else None


def merge_sorted_runs(runs: Sequence[Iterable[pa.Table]], primary_key: Sequence[str]) -> Iterator[pa.Table]:
    """Merge runs into the newest row per primary key, yielding tables in ascending key order.

    Each run yields tables whose rows ascend by primary key within and across tables. A run
    can hold several versions of a key, including across consecutive tables. Every yielded
    table's keys are strictly greater than the previous table's, so no key appears twice in
    the output. Memory holds about one table per run, and rows are sorted when they become
    final rather than on every round.
    """
    cursors = [_RunCursor(iter(run), primary_key) for run in runs]
    while active := [cursor for cursor in cursors if not cursor.exhausted]:
        # Every run's unread rows are at or above its last buffered key, so buffered rows below
        # the smallest such key are final. Only the runs at that frontier read further.
        frontier = min(cursor.last_key for cursor in active)
        final = [rows for cursor in cursors if (rows := cursor.take_below(frontier)) is not None]
        if final:
            yield newest_rows(pa.concat_tables(final), primary_key)
        for cursor in active:
            if cursor.last_key == frontier:
                cursor.pull()
    rest = [rows for cursor in cursors if (rows := cursor.take_rest()) is not None]
    if rest:
        yield newest_rows(pa.concat_tables(rest), primary_key)
