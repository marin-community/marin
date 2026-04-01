# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Writers for common output formats."""

from __future__ import annotations

import queue
import tempfile
import threading
import uuid
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
import itertools
import os
from typing import Any

import pyarrow as pa
from rigging.filesystem import open_url, url_to_fs
import msgspec
import logging
from zephyr import counters

logger = logging.getLogger(__name__)

# 64 MB write blocks — controls S3 multipart upload part size.
_WRITE_BLOCK_SIZE = 64 * 1024 * 1024

# Default target buffer size for writer batching. Writers accumulate
# micro-batches until accumulated nbytes reaches this threshold, then yield
# a single concatenated table.
DEFAULT_TARGET_BUFFER_BYTES = 64 * 1024 * 1024  # 64 MB

# Number of records converted to PyArrow at a time. Small enough that
# ``pa.Table.from_pylist`` is fast; large enough to amortise per-call overhead.
_MICRO_BATCH_SIZE = 8

# Fixed batch size for Levanter cache writes (2^14).
_LEVANTER_BATCH_SIZE = 16384


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

    For S3-compatible stores, writes to a local temp directory first, then uploads
    via fs.put() to avoid server-side multipart copy which is unreliable on some
    providers (e.g. Cloudflare R2).

    Example:
        with atomic_rename("output.jsonl.gz") as tmp_path:
            write_data(tmp_path)
        # File is now at output.jsonl.gz
    """
    if output_path.startswith("s3://"):
        fs, resolved_path = url_to_fs(output_path)
        with tempfile.TemporaryDirectory() as local_tmp_dir:
            local_path = os.path.join(local_tmp_dir, "output")
            yield local_path
            if os.path.isdir(local_path):
                # Trailing slash prevents fsspec from nesting under an extra
                # "output/" level when the destination already exists.
                fs.put(local_path + "/", resolved_path, recursive=True)
            else:
                fs.put(local_path, resolved_path)
        return

    temp_path = unique_temp_path(output_path)
    fs = url_to_fs(output_path)[0]

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
        fs, dir_path = url_to_fs(output_dir)
        if not fs.exists(dir_path):
            fs.mkdirs(dir_path, exist_ok=True)
    else:
        output_dir = os.path.dirname(path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)


@contextmanager
def _open_write_stream(fs, resolved_path: str, output_path: str):
    """Open a binary write stream with compression inferred from ``output_path``."""
    if output_path.endswith(".zst"):
        import zstandard as zstd

        cctx = zstd.ZstdCompressor(level=2, threads=1)
        with fs.open(resolved_path, "wb", block_size=_WRITE_BLOCK_SIZE) as raw_f:
            with cctx.stream_writer(raw_f) as f:
                yield f
    elif output_path.endswith(".gz"):
        with fs.open(resolved_path, "wb", block_size=_WRITE_BLOCK_SIZE, compression="gzip") as f:
            yield f
    else:
        with fs.open(resolved_path, "wb", block_size=_WRITE_BLOCK_SIZE) as f:
            yield f


def write_jsonl_file(records: Iterable, output_path: str) -> dict:
    """Write records to a JSONL file with automatic compression."""
    ensure_parent_dir(output_path)

    count = 0
    encoder = msgspec.json.Encoder()

    with atomic_rename(output_path) as temp_path:
        fs, resolved_temp = url_to_fs(temp_path)
        with _open_write_stream(fs, resolved_temp, output_path) as f:
            for record in records:
                f.write(encoder.encode(record) + b"\n")
                count += 1
                counters.increment("zephyr/records_out")

    return {"path": output_path, "count": count}


def infer_arrow_schema(records: list[dict[str, Any]]) -> Any:
    """Infer a PyArrow schema from a batch of record dicts"""
    return pa.Table.from_pylist(records).schema


def batchify(batch: Iterable, n: int = 1024) -> Iterable:
    iterator = iter(batch)
    while batch := tuple(itertools.islice(iterator, n)):
        yield batch


def _accumulate_tables(
    records: Iterable,
    *,
    schema: pa.Schema | None = None,
    target_bytes: int = DEFAULT_TARGET_BUFFER_BYTES,
) -> Iterable[pa.Table]:
    """Yield PyArrow tables of approximately ``target_bytes`` each.

    Converts records to PyArrow in micro-batches of ``_MICRO_BATCH_SIZE``,
    tracks byte size incrementally, and yields a single ``concat_tables``
    result each time the threshold is reached.
    """
    chunks: list[pa.Table] = []
    bytesize = 0
    convert: Callable | None = None

    for micro_batch in batchify(records, n=_MICRO_BATCH_SIZE):
        if convert is None:
            convert = asdict if is_dataclass(micro_batch[0]) else (lambda x: x)
        dicts = [convert(r) for r in micro_batch]
        if schema is None:
            # NOTE: the _MICRO_BATCH_SIZE is fairly small, here we hope it's enough to infer "real" schema
            schema = infer_arrow_schema(dicts)
        table = pa.Table.from_pylist(dicts, schema=schema)
        chunks.append(table)
        bytesize += table.nbytes
        if bytesize >= target_bytes:
            yield pa.concat_tables(chunks)
            chunks = []
            bytesize = 0

    if chunks:
        yield pa.concat_tables(chunks)


def write_parquet_file(
    records: Iterable,
    output_path: str,
    *,
    schema: pa.Schema | None = None,
    target_buffer_bytes: int = DEFAULT_TARGET_BUFFER_BYTES,
) -> dict:
    """Write records to a Parquet file.

    Args:
        records: Records to write (iterable of dicts)
        output_path: Path to output file
        schema: PyArrow schema (optional, will be inferred from first batch if None)
        target_buffer_bytes: Target buffer size in bytes for accumulating records
            before flushing to the writer.

    Returns:
        Dict with metadata: {"path": output_path, "count": num_records}
    """
    import pyarrow.parquet as pq

    ensure_parent_dir(output_path)
    count = 0

    with atomic_rename(output_path) as temp_path:
        writer: pq.ParquetWriter | None = None
        try:
            for table in _accumulate_tables(records, schema=schema, target_bytes=target_buffer_bytes):
                if writer is None:
                    writer = pq.ParquetWriter(temp_path, table.schema)
                writer.write_table(table)
                count += len(table)
                counters.increment("zephyr/records_out", len(table))
        finally:
            if writer is not None:
                writer.close()

        if writer is None:
            actual_schema = schema or pa.schema([])
            pq.write_table(pa.Table.from_pylist([], schema=actual_schema), temp_path)

    return {"path": output_path, "count": count}


def write_vortex_file(
    records: Iterable,
    output_path: str,
    *,
    schema: pa.Schema | None = None,
    target_buffer_bytes: int = DEFAULT_TARGET_BUFFER_BYTES,
) -> dict:
    """Write records to a Vortex file using streaming writes.

    Args:
        records: Records to write (iterable of dicts)
        output_path: Path to output .vortex file
        schema: PyArrow schema (optional, will be inferred from first batch if None)
        target_buffer_bytes: Target buffer size in bytes for accumulating records
            before flushing to the writer.

    Returns:
        Dict with metadata: {"path": output_path, "count": num_records}
    """
    import vortex

    ensure_parent_dir(output_path)

    table_iter = _accumulate_tables(records, schema=schema, target_bytes=target_buffer_bytes)
    first_table = next(table_iter, None)

    if first_table is None:
        actual_schema = schema or pa.schema([])
        empty_table = pa.Table.from_pylist([], schema=actual_schema)
        with atomic_rename(output_path) as temp_path:
            vortex.io.write(empty_table, temp_path)
        return {"path": output_path, "count": 0}

    actual_schema = first_table.schema
    dtype = vortex.DType.from_arrow(actual_schema, non_nullable=True)
    count = 0

    def _array_batches():
        nonlocal count
        count += len(first_table)
        counters.increment("zephyr/records_out", len(first_table))
        yield vortex.Array.from_arrow(first_table)
        for table in table_iter:
            count += len(table)
            counters.increment("zephyr/records_out", len(table))
            yield vortex.Array.from_arrow(table)

    array_iter = vortex.ArrayIterator.from_iter(dtype, _array_batches())

    with atomic_rename(output_path) as temp_path:
        vortex.io.write(array_iter, temp_path)

    return {"path": output_path, "count": count}


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
    records: Iterable[dict[str, Any]],
    output_path: str,
    *,
    metadata: dict[str, Any],
) -> dict:
    """Write tokenized records to Levanter cache format.

    Args:
        records: Tokenized records (iterable of dicts with array values)
        output_path: Path to output cache directory
        metadata: Metadata for the cache
    """
    from levanter.store.cache import CacheMetadata, SerialCacheWriter

    ensure_parent_dir(output_path)
    record_iter = iter(records)

    try:
        exemplar = next(record_iter)
    except StopIteration:
        return {"path": output_path, "count": 0}

    count = 0
    logger.info("write_levanter_cache: starting write to %s (batch_size=%d)", output_path, _LEVANTER_BATCH_SIZE)

    with atomic_rename(output_path) as tmp_path:
        with SerialCacheWriter(tmp_path, exemplar, shard_name=output_path, metadata=CacheMetadata(metadata)) as writer:

            def _drain_batches(batches: Iterable) -> None:
                for batch in batches:
                    writer.write_batch(batch)

            with ThreadedBatchWriter(_drain_batches) as threaded:
                threaded.submit([exemplar])
                count += 1
                counters.increment("zephyr/records_out")
                for batch in batchify(record_iter, n=_LEVANTER_BATCH_SIZE):
                    threaded.submit(batch)
                    count += len(batch)
                    counters.increment("zephyr/records_out", len(batch))
                    logger.info("write_levanter_cache: %s — %d records so far", output_path, count)

    logger.info("write_levanter_cache: finished %s — %d records", output_path, count)

    # write success sentinel
    with open_url(f"{output_path}/.success", "w") as f:
        f.write("")

    return {"path": output_path, "count": count}


def write_binary_file(records: Iterable[bytes], output_path: str) -> dict:
    """Write binary records to a file."""
    ensure_parent_dir(output_path)

    count = 0
    with atomic_rename(output_path) as temp_path:
        fs, resolved_temp = url_to_fs(temp_path)
        with fs.open(resolved_temp, "wb", block_size=_WRITE_BLOCK_SIZE) as f:
            for record in records:
                f.write(record)
                count += 1
                counters.increment("zephyr/records_out")

    return {"path": output_path, "count": count}
