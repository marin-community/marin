# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Byte-budgeted Parquet writer with background GCS uploads.

SpillWriter wraps pq.ParquetWriter and accumulates Arrow tables, flushing
them as row groups when the accumulated bytes exceed a configurable threshold.
Writes happen in a background thread so the caller can overlap production
with I/O (pq.ParquetWriter.write_table releases the GIL).

TableAccumulator is a standalone helper that accumulates Arrow tables until
a byte threshold is reached, then yields the concatenated result.
"""

import logging
import queue
import threading

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

_SENTINEL = object()


class TableAccumulator:
    """Accumulates Arrow tables and yields merged results when a byte threshold is reached.

    Unlike row-count batching, byte-budgeted batching produces uniformly-sized
    output regardless of row width, which matters for write performance and
    memory predictability.
    """

    def __init__(self, byte_threshold: int) -> None:
        self._byte_threshold = byte_threshold
        self._tables: list[pa.Table] = []
        self._nbytes: int = 0

    def add(self, table: pa.Table) -> pa.Table | None:
        """Accumulate a table. Returns a merged table when the threshold is exceeded, else None."""
        self._tables.append(table)
        self._nbytes += table.nbytes
        if self._nbytes >= self._byte_threshold:
            return self._take()
        return None

    def flush(self) -> pa.Table | None:
        """Return any remaining accumulated data, or None if empty."""
        if not self._tables:
            return None
        return self._take()

    def _take(self) -> pa.Table:
        result = pa.concat_tables(self._tables, promote_options="default")
        self._tables.clear()
        self._nbytes = 0
        return result

    def pending_bytes(self) -> int:
        return self._nbytes

    def __len__(self) -> int:
        return sum(len(t) for t in self._tables)


def _background_writer_loop(
    write_queue: "queue.Queue[pa.Table | object]",
    writer: pq.ParquetWriter,
    error_box: list[Exception],
) -> None:
    """Drain write_queue, writing each table as a row group. Stops on _SENTINEL."""
    while True:
        item = write_queue.get()
        if item is _SENTINEL:
            return
        try:
            writer.write_table(item)
        except Exception as exc:
            error_box.append(exc)
            return


class SpillWriter:
    """Byte-budgeted ParquetWriter with background I/O.

    Row groups are accumulated via an internal TableAccumulator and flushed
    to a pq.ParquetWriter in a background thread, overlapping one write
    with the next produce cycle.

    Two write modes:
    - write_table(table): accumulates rows, flushes a row group when
      accumulated bytes exceed row_group_bytes.
    - write_row_group(table): writes the table as its own row group immediately
      (no accumulation). Used by the scatter path where each sorted chunk must
      be a separate row group.
    """

    def __init__(
        self,
        path: str,
        schema: pa.Schema,
        *,
        row_group_bytes: int = 8 * 1024 * 1024,
        compression: str = "zstd",
        compression_level: int = 1,
    ) -> None:
        self._writer = pq.ParquetWriter(path, schema, compression=compression, compression_level=compression_level)
        self._accumulator = TableAccumulator(row_group_bytes)
        # Single-slot queue: allows one write to be in-flight while the caller
        # produces the next batch. Backpressure is automatic — put() blocks when
        # the slot is occupied.
        self._queue: queue.Queue[pa.Table | object] = queue.Queue(maxsize=1)
        self._error_box: list[Exception] = []
        self._thread = threading.Thread(
            target=_background_writer_loop,
            args=(self._queue, self._writer, self._error_box),
            daemon=True,
        )
        self._thread.start()
        self._closed = False

    def _check_error(self) -> None:
        if self._error_box:
            raise self._error_box[0]

    def write_table(self, table: pa.Table) -> None:
        """Accumulate rows; flush a row group to the background writer when threshold is exceeded."""
        self._check_error()
        merged = self._accumulator.add(table)
        if merged is not None:
            self._queue.put(merged)
            self._check_error()

    def write_row_group(self, table: pa.Table) -> None:
        """Write the table as its own row group immediately (no accumulation)."""
        self._check_error()
        self._queue.put(table)
        self._check_error()

    def close(self) -> None:
        """Flush remaining accumulated data and wait for the background thread to finish."""
        self._close(abort=False)

    def _close(self, *, abort: bool) -> None:
        # abort=True (caller raised inside `with`): skip the final flush — the
        # partial file will never be read. Either way, we must still drain the
        # queue so the background thread exits and the parquet file handle is
        # closed cleanly.
        #
        # If the background thread has already errored, its `get()` consumed the
        # last item and then the thread returned, leaving the queue empty. So a
        # single `put(_SENTINEL)` into the maxsize=1 queue does not block —
        # but putting anything *before* that sentinel would, because nothing is
        # draining. Hence we only enqueue `remaining` when the thread is still
        # healthy (error_box empty) and we're not aborting.
        if self._closed:
            return
        self._closed = True
        try:
            if not abort and not self._error_box:
                remaining = self._accumulator.flush()
                if remaining is not None:
                    self._queue.put(remaining)
            self._queue.put(_SENTINEL)
            self._thread.join()
        finally:
            self._writer.close()
        if not abort:
            self._check_error()

    def __enter__(self) -> "SpillWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._close(abort=exc_type is not None)
