# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Newest-version selection and the sorted-run merge shared by reads and compaction.

Rows order by primary key ascending, with nulls after every value and two nulls equal, and
then by version: a greater ``(_commit, _seq, _gen)`` is newer, and a missing ``_seq``
counts as zero. The newest row of each key is the visible one.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence

import pyarrow as pa
import pyarrow.compute as pc

from finestore.layout import SystemColumns


def newest_indices(rows: pa.Table, primary_key: Sequence[str]) -> pa.Array:
    """Return the index of the newest row for each primary key, in primary-key order."""
    if rows.num_rows == 0:
        return pa.array([], pa.int64())
    order_columns: dict[str, pa.ChunkedArray] = {}
    sort_keys: list[tuple[str, str]] = []
    for index, name in enumerate(primary_key):
        order_columns[f"key{index}"] = rows[name]
        sort_keys.append((f"key{index}", "ascending"))
    order_columns["commit"] = rows[SystemColumns.COMMIT]
    if SystemColumns.SEQUENCE in rows.column_names:
        order_columns["sequence"] = pc.fill_null(rows[SystemColumns.SEQUENCE], 0)
    order_columns["generation"] = rows[SystemColumns.GENERATION]
    sort_keys.extend((name, "descending") for name in ("commit", "sequence", "generation") if name in order_columns)
    order = pc.sort_indices(pa.table(order_columns), sort_keys=sort_keys, null_placement="at_end")
    return order.filter(_starts_key(rows.select(list(primary_key)).take(order)))


def newest_rows(rows: pa.Table, primary_key: Sequence[str]) -> pa.Table:
    """Return the newest row for each primary key, in primary-key order."""
    return rows.take(newest_indices(rows, primary_key))


def sorted_by_key(tables: Iterable[pa.Table], primary_key: Sequence[str]) -> Iterator[pa.Table]:
    """Gather ``tables`` into one run in merge key order; memory holds all of their rows."""
    rows = list(tables)
    if rows:
        yield pa.concat_tables(rows).sort_by([(name, "ascending") for name in primary_key], null_placement="at_end")


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


def row_key(rows: pa.Table, index: int, primary_key: Sequence[str]) -> tuple:
    """Return a row's primary key as a tuple that sorts the way the merge orders keys."""
    return tuple((value is None, value) for value in (rows[name][index].as_py() for name in primary_key))


def _first_at_or_above(rows: pa.Table, primary_key: Sequence[str], key: tuple) -> int:
    low, high = 0, rows.num_rows
    while low < high:
        middle = (low + high) // 2
        if row_key(rows, middle, primary_key) < key:
            low = middle + 1
        else:
            high = middle
    return low


class _RunCursor:
    """Reads one sorted run a table at a time and remembers the last key it returned."""

    def __init__(self, tables: Iterator[pa.Table], primary_key: Sequence[str]) -> None:
        self._tables = tables
        self._primary_key = primary_key
        self.last_key: tuple = ()
        self.exhausted = False

    def advance(self) -> pa.Table | None:
        for table in self._tables:
            if table.num_rows:
                self.last_key = row_key(table, table.num_rows - 1, self._primary_key)
                return table
        self.exhausted = True
        return None


def merge_sorted_runs(runs: Sequence[Iterable[pa.Table]], primary_key: Sequence[str]) -> Iterator[pa.Table]:
    """Merge runs into the newest row per primary key, yielding tables in ascending key order.

    Each run yields tables whose rows ascend by primary key within and across tables. A run
    can hold several versions of a key, including across consecutive tables. Every yielded
    table's keys are strictly greater than the previous table's, so no key appears twice in
    the output. Memory holds about one table per run.
    """
    cursors = [_RunCursor(iter(run), primary_key) for run in runs]
    pending = cursors
    carry: pa.Table | None = None
    while True:
        tables = [table for cursor in pending if (table := cursor.advance()) is not None]
        if carry is not None:
            tables.insert(0, carry)
        merged = newest_rows(pa.concat_tables(tables), primary_key) if tables else None
        active = [cursor for cursor in cursors if not cursor.exhausted]
        if not active:
            if merged is not None and merged.num_rows:
                yield merged
            return
        # Every run's remaining rows are at or above its own last key, so rows below the smallest
        # last key are final. Rows at or above it carry over, already one row per key, and only
        # the runs at that frontier are read next.
        frontier = min(cursor.last_key for cursor in active)
        pending = [cursor for cursor in active if cursor.last_key == frontier]
        if merged is None:
            continue
        final = _first_at_or_above(merged, primary_key, frontier)
        if final:
            yield merged.slice(0, final)
        carry = merged.slice(final)
