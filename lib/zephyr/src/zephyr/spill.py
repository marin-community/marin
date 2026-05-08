# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Opaque chunked row format for zephyr spill files.

SpillWriter and SpillReader hide the on-disk representation from callers.
Items are pickled into an opaque binary payload and written as chunks of a
chunked row format. Callers do not see the schema, serialization, or storage
format — they append items and read back items (or chunks of items) in the
same order.

Currently backed by Parquet with a single binary payload column, a background
I/O thread, and byte-budgeted row groups. The file format is an implementation
detail; do not rely on it outside this module.
"""

import logging
import pickle
from collections.abc import Iterable, Iterator
from typing import Any

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq

from zephyr.writers import ThreadedBatchWriter

logger = logging.getLogger(__name__)

# Single binary payload column. Not part of the public API.
_PAYLOAD_COL = "_zephyr_payload"
_SCHEMA = pa.schema([pa.field(_PAYLOAD_COL, pa.binary())])


class _TableAccumulator:
    """Accumulates Arrow tables and yields merged results when a byte threshold is reached.

    Byte-budgeted batching produces uniformly-sized output regardless of row
    width, which matters for write performance and memory predictability.
    """

    def __init__(self, byte_threshold: int) -> None:
        self._byte_threshold = byte_threshold
        self._tables: list[pa.Table] = []
        self._nbytes: int = 0

    def add(self, table: pa.Table) -> pa.Table | None:
        self._tables.append(table)
        self._nbytes += table.nbytes
        if self._nbytes >= self._byte_threshold:
            return self._take()
        return None

    def flush(self) -> pa.Table | None:
        if not self._tables:
            return None
        return self._take()

    def _take(self) -> pa.Table:
        result = pa.concat_tables(self._tables, promote_options="default")
        self._tables.clear()
        self._nbytes = 0
        return result


def _items_to_table(items: Iterable[Any]) -> pa.Table:
    payloads = [pickle.dumps(item, protocol=pickle.HIGHEST_PROTOCOL) for item in items]
    return pa.table({_PAYLOAD_COL: pa.array(payloads, type=pa.binary())})


class SpillWriter:
    """Writes items to an opaque chunked row-format spill file.

    Use ``write`` to stream items; the writer accumulates a byte budget and
    emits chunks when the budget is exceeded. Use ``write_chunk`` to commit
    a batch of items as its own chunk immediately (no accumulation) — useful
    when the caller wants each logical batch to round-trip as one chunk.

    Writes are offloaded to a :class:`ThreadedBatchWriter` so one write can be
    in-flight while the caller produces the next batch. Backpressure, error
    propagation, and clean teardown on the exception path are delegated to it.
    """

    def __init__(
        self,
        path: str,
        *,
        row_group_bytes: int = 8 * 1024 * 1024,
        compression: str = "zstd",
        compression_level: int = 1,
    ) -> None:
        self._writer = pq.ParquetWriter(path, _SCHEMA, compression=compression, compression_level=compression_level)
        self._accumulator = _TableAccumulator(row_group_bytes)

        def _drain(tables: Iterable[pa.Table]) -> None:
            for table in tables:
                self._writer.write_table(table)

        # maxsize=1: at most one chunk in-flight so memory stays bounded while
        # the producer keeps working on the next batch.
        self._threaded = ThreadedBatchWriter(_drain, maxsize=1)
        self._closed = False

    def write(self, items: Iterable[Any]) -> None:
        """Append items. Emits a chunk when the accumulated byte budget is exceeded."""
        table = _items_to_table(items)
        if len(table) == 0:
            return
        merged = self._accumulator.add(table)
        if merged is not None:
            self._threaded.submit(merged)

    def write_chunk(self, items: Iterable[Any]) -> None:
        """Commit items as their own chunk immediately (no accumulation)."""
        table = _items_to_table(items)
        if len(table) == 0:
            return
        self._threaded.submit(table)

    def close(self) -> None:
        """Flush remaining buffered items and wait for the background writer."""
        if self._closed:
            return
        self._closed = True
        try:
            remaining = self._accumulator.flush()
            if remaining is not None:
                self._threaded.submit(remaining)
            self._threaded.close()
        finally:
            self._writer.close()

    def __enter__(self) -> "SpillWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if exc_type is not None:
                # Error path: skip final flush (partial file will never be read)
                # and let ThreadedBatchWriter.__exit__ tear down the thread
                # without blocking the caller.
                self._threaded.__exit__(exc_type, exc_val, exc_tb)
            else:
                remaining = self._accumulator.flush()
                if remaining is not None:
                    self._threaded.submit(remaining)
                self._threaded.close()
        finally:
            self._writer.close()


class SpillReader:
    """Reads items from an opaque chunked row-format spill file.

    Iteration yields items one at a time in write order. ``iter_chunks`` yields
    lists of items grouped by the on-disk chunks; callers that want a specific
    batch size can pass ``batch_size`` to re-batch.
    """

    def __init__(self, path: str, *, batch_size: int | None = None) -> None:
        self._path = path
        self._batch_size = batch_size

    @property
    def path(self) -> str:
        return self._path

    @property
    def num_rows(self) -> int:
        with fsspec.open(self._path, "rb") as f:
            return pq.ParquetFile(f).metadata.num_rows

    @property
    def approx_item_bytes(self) -> int:
        """Uncompressed payload bytes per item, read from file metadata.

        Returns 0 for an empty spill. Useful as a memory-budgeting hint without
        exposing the underlying format.
        """
        with fsspec.open(self._path, "rb") as f:
            md = pq.ParquetFile(f).metadata
            if md.num_rows <= 0:
                return 0
            total = sum(md.row_group(i).column(0).total_uncompressed_size for i in range(md.num_row_groups))
            return total // md.num_rows

    def iter_chunks(self) -> Iterator[list[Any]]:
        """Yield chunks of items (lists).

        Chunk boundaries follow the on-disk layout unless ``batch_size`` was
        set on the reader, in which case items are re-batched to approximately
        that size.
        """
        with fsspec.open(self._path, "rb") as f:
            pf = pq.ParquetFile(f)
            if self._batch_size is None:
                for i in range(pf.num_row_groups):
                    table = pf.read_row_group(i, columns=[_PAYLOAD_COL])
                    payloads = table.column(_PAYLOAD_COL).to_pylist()
                    yield [pickle.loads(p) for p in payloads]
            else:
                for record_batch in pf.iter_batches(batch_size=self._batch_size, columns=[_PAYLOAD_COL]):
                    payloads = record_batch.column(_PAYLOAD_COL).to_pylist()
                    yield [pickle.loads(p) for p in payloads]

    def __iter__(self) -> Iterator[Any]:
        for chunk in self.iter_chunks():
            yield from chunk
