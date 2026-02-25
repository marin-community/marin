# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Writers for common output formats."""

from __future__ import annotations

import queue
import threading
import uuid
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
import itertools
import os
from typing import Any

import fsspec
import msgspec
import logging

logger = logging.getLogger(__name__)


def unique_temp_path(output_path: str) -> str:
    """Return a unique temporary path derived from ``output_path``.

    Appends ``.tmp.<uuid>`` to avoid collisions when multiple writers target the
    same output path (e.g. during network-partition induced worker races).
    """
    return f"{output_path}.tmp.{uuid.uuid4().hex}"


@contextmanager
def atomic_rename(output_path: str) -> Iterable[str]:
    """Context manager for atomic write-and-rename with UUID collision avoidance.

    Yields a unique temporary path to write to. On successful exit, atomically
    renames the temp file to the final path. On failure, cleans up the temp file.

    Example:
        with atomic_rename("output.jsonl.gz") as tmp_path:
            write_data(tmp_path)
        # File is now at output.jsonl.gz
    """
    temp_path = unique_temp_path(output_path)
    fs = fsspec.core.url_to_fs(output_path)[0]

    try:
        yield temp_path
        fs.mv(temp_path, output_path, recursive=True)
    except Exception:
        # Try to cleanup if something went wrong
        try:
            if fs.exists(temp_path):
                fs.rm(temp_path)
        except Exception:
            pass
        raise


def ensure_parent_dir(path: str) -> None:
    """Create directories for `path` if necessary."""
    # Use os.path.dirname for local paths, otherwise use fsspec
    if "://" in path:
        output_dir = path.rsplit("/", 1)[0]
        fs, dir_path = fsspec.core.url_to_fs(output_dir)
        if not fs.exists(dir_path):
            fs.mkdirs(dir_path, exist_ok=True)
    else:
        output_dir = os.path.dirname(path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)


def write_jsonl_file(records: Iterable, output_path: str) -> dict:
    """Write records to a JSONL file with automatic compression."""
    ensure_parent_dir(output_path)

    count = 0
    encoder = msgspec.json.Encoder()

    with atomic_rename(output_path) as temp_path:
        if output_path.endswith(".zst"):
            import zstandard as zstd

            cctx = zstd.ZstdCompressor(level=2, threads=1)
            with fsspec.open(temp_path, "wb", block_size=64 * 1024 * 1024) as raw_f:
                with cctx.stream_writer(raw_f) as f:
                    for record in records:
                        f.write(encoder.encode(record) + b"\n")
                        count += 1
        elif output_path.endswith(".gz"):
            with fsspec.open(temp_path, "wb", compression="gzip", compresslevel=1, block_size=64 * 1024 * 1024) as f:
                for record in records:
                    f.write(encoder.encode(record) + b"\n")
                    count += 1
        else:
            with fsspec.open(temp_path, "wb", block_size=64 * 1024 * 1024) as f:
                for record in records:
                    f.write(encoder.encode(record) + b"\n")
                    count += 1

    return {"path": output_path, "count": count}


def infer_parquet_type(value):
    """Recursively infer PyArrow type from a Python value."""
    import pyarrow as pa

    if isinstance(value, bool):
        # Check bool before int since bool is a subclass of int
        return pa.bool_()
    elif isinstance(value, str):
        return pa.string()
    elif isinstance(value, int):
        return pa.int64()
    elif isinstance(value, float):
        return pa.float64()
    elif isinstance(value, dict):
        nested_fields = []
        for k, v in value.items():
            nested_fields.append((k, infer_parquet_type(v)))
        return pa.struct(nested_fields)
    elif isinstance(value, list):
        # Simple list of strings for now
        return pa.list_(pa.string())
    else:
        return pa.string()


def infer_parquet_schema(record: dict[str, Any] | Any):
    """Infer PyArrow schema from a dictionary record."""
    import pyarrow as pa

    if is_dataclass(record):
        record = asdict(record)

    fields = []
    for key, value in record.items():
        fields.append((key, infer_parquet_type(value)))

    return pa.schema(fields)


def write_parquet_file(
    records: Iterable, output_path: str, schema: object | None = None, batch_size: int = 1000
) -> dict:
    """Write records to a Parquet file.

    Args:
        records: Records to write (iterable of dicts)
        output_path: Path to output file
        schema: PyArrow schema (optional, will be inferred from first record if None)
        batch_size: Number of records per batch (default: 1000)

    Returns:
        Dict with metadata: {"path": output_path, "count": num_records}
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    ensure_parent_dir(output_path)

    record_iter = iter(records)

    first_record = None
    try:
        first_record = next(record_iter)
    except StopIteration:
        # Empty dataset case - write directly without temp file
        actual_schema = schema or pa.schema([])
        table = pa.Table.from_pylist([], schema=actual_schema)
        pq.write_table(table, output_path)
        return {"path": output_path, "count": 0}

    actual_schema = schema or infer_parquet_schema(first_record)
    maybe_map_to_dict = asdict if is_dataclass(first_record) else (lambda x: x)

    count = 1
    with atomic_rename(output_path) as temp_path:
        with pq.ParquetWriter(temp_path, actual_schema) as writer:
            batch = [maybe_map_to_dict(first_record)]
            for record in record_iter:
                batch.append(maybe_map_to_dict(record))
                count += 1
                if len(batch) >= batch_size:
                    table = pa.Table.from_pylist(batch, schema=actual_schema)
                    writer.write_table(table)
                    batch = []

            if batch:
                table = pa.Table.from_pylist(batch, schema=actual_schema)
                writer.write_table(table)

    return {"path": output_path, "count": count}


def write_vortex_file(records: Iterable, output_path: str) -> dict:
    """Write records to a Vortex file.

    Args:
        records: Records to write (iterable of dicts)
        output_path: Path to output .vortex file

    Returns:
        Dict with metadata: {"path": output_path, "count": num_records}
    """
    import pyarrow as pa
    import vortex

    ensure_parent_dir(output_path)

    record_iter = iter(records)

    try:
        first_record = next(record_iter)
    except StopIteration:
        # Empty case - write empty vortex file
        empty_table = pa.Table.from_pylist([])
        with atomic_rename(output_path) as temp_path:
            vortex.io.write(empty_table, temp_path)
        return {"path": output_path, "count": 0}

    # Accumulate all records and write
    all_records = [first_record]
    for record in record_iter:
        all_records.append(record)

    count = len(all_records)
    table = pa.Table.from_pylist(all_records)

    with atomic_rename(output_path) as temp_path:
        vortex.io.write(table, temp_path)

    return {"path": output_path, "count": count}


def batchify(batch: Iterable, n: int = 1024) -> Iterable:
    iterator = iter(batch)
    while batch := tuple(itertools.islice(iterator, n)):
        yield batch


_SENTINEL = object()


def _queue_iterable(q: queue.Queue) -> Iterable:
    """Yield items from a bounded queue until the sentinel is received.

    Designed for use with ``ThreadedBatchWriter``: the background thread passes
    this iterable to a writer function so the writer can consume items naturally
    as they arrive through the queue.
    """
    while True:
        item = q.get()
        if item is _SENTINEL:
            return
        yield item


class ThreadedBatchWriter:
    """Offloads batch writes to a background thread so the producer isn't blocked on IO.

    Uses a bounded queue for backpressure: the producer blocks when the writer
    falls behind, preventing unbounded memory growth.

    The ``write_fn`` receives an iterable that yields submitted items from the
    internal queue, allowing the writer to consume items as a natural stream
    rather than via per-item callbacks.
    """

    def __init__(self, write_fn: Callable[[Iterable], None], maxsize: int = 128):
        self._write_fn = write_fn
        self._queue_maxsize = maxsize
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, daemon=True, name="ZephyrWriter")
        self._thread.start()

    def _run(self) -> None:
        try:
            self._write_fn(_queue_iterable(self._queue))
        except Exception as e:
            self._error = e

    def submit(self, batch: Any) -> None:
        """Enqueue *batch* for writing. Raises if the background thread failed."""
        # Poll so we detect background-thread failures even when the queue is
        # full (a plain ``put`` would block forever if the consumer died).
        while True:
            if self._error is not None:
                raise self._error
            try:
                self._queue.put(batch, timeout=1.0)
                return
            except queue.Full:
                logger.warning(f"ThreadedBatchWriter queue is full (size={self._queue_maxsize}), waiting ...")
                continue

    def close(self) -> None:
        """Wait for all pending writes and propagate any error."""
        self._queue.put(_SENTINEL)
        self._thread.join()
        if self._error is not None:
            raise self._error

    def __enter__(self) -> ThreadedBatchWriter:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None:
            # Signal the thread to stop without blocking the caller.
            try:
                self._queue.put_nowait(_SENTINEL)
            except queue.Full:
                pass
            self._thread.join(timeout=5.0)
            return False
        self.close()
        return False


def write_levanter_cache(
    records: Iterable[dict[str, Any]], output_path: str, metadata: dict[str, Any], batch_size: int = 1024
) -> dict:
    """Write tokenized records to Levanter cache format."""
    from levanter.store.cache import CacheMetadata, SerialCacheWriter

    ensure_parent_dir(output_path)
    record_iter = iter(records)

    try:
        exemplar = next(record_iter)
    except StopIteration:
        return {"path": output_path, "count": 0}

    count = 1
    logger.info("write_levanter_cache: starting write to %s", output_path)

    with atomic_rename(output_path) as tmp_path:
        with SerialCacheWriter(tmp_path, exemplar, shard_name=output_path, metadata=CacheMetadata(metadata)) as writer:

            def _drain_batches(batches: Iterable) -> None:
                for batch in batches:
                    writer.write_batch(batch)

            with ThreadedBatchWriter(_drain_batches) as threaded:
                threaded.submit([exemplar])
                for batch in batchify(record_iter, n=batch_size):
                    threaded.submit(batch)
                    count += len(batch)
                    logger.info("write_levanter_cache: %s — %d records so far", output_path, count)

    logger.info("write_levanter_cache: finished %s — %d records", output_path, count)

    # write success sentinel
    with fsspec.open(f"{output_path}/.success", "w") as f:
        f.write("")

    return {"path": output_path, "count": count}


def write_binary_file(records: Iterable[bytes], output_path: str) -> dict:
    """Write binary records to a file."""
    ensure_parent_dir(output_path)

    count = 0
    with atomic_rename(output_path) as temp_path:
        with fsspec.open(temp_path, "wb", block_size=64 * 1024 * 1024) as f:
            for record in records:
                f.write(record)
                count += 1

    return {"path": output_path, "count": count}
