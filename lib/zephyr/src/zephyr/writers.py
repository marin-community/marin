# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Writers for common output formats."""

from __future__ import annotations

import uuid
from dataclasses import asdict, is_dataclass
import itertools
import os
from collections.abc import Iterable
from contextlib import contextmanager
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


def _get_existing_row_count(tmp_path: str, exemplar: dict[str, Any]) -> int:
    """Read the number of rows already written in a partial .tmp cache directory.

    Returns 0 if the path doesn't exist, has no data, or can't be read.
    """
    fs = fsspec.core.url_to_fs(tmp_path)[0]
    if not fs.exists(tmp_path):
        return 0
    try:
        from levanter.store.tree_store import TreeStore

        store = TreeStore.open(exemplar, tmp_path, mode="r", cache_metadata=False)
        return len(store)
    except Exception:
        logger.debug("Could not read existing rows from %s, starting fresh", tmp_path, exc_info=True)
        return 0


def _promote_tmp_cache(fs, tmp_path: str, output_path: str) -> None:
    """Promote a temporary cache directory to the final output path.

    If a previous output exists, move it aside first and restore it on failure.
    """
    backup_path = None
    if fs.exists(output_path):
        backup_path = f"{output_path}.bak"
        if fs.exists(backup_path):
            fs.rm(backup_path, recursive=True)
        fs.mv(output_path, backup_path, recursive=True)

    try:
        fs.mv(tmp_path, output_path, recursive=True)
    except Exception:
        if backup_path is not None and fs.exists(backup_path):
            try:
                fs.mv(backup_path, output_path, recursive=True)
            except Exception as restore_exc:
                raise RuntimeError(
                    f"Failed to promote {tmp_path} to {output_path} and failed to restore {backup_path}"
                ) from restore_exc
        raise
    else:
        if backup_path is not None and fs.exists(backup_path):
            fs.rm(backup_path, recursive=True)


def write_levanter_cache(records: Iterable[dict[str, Any]], output_path: str, metadata: dict[str, Any]) -> dict:
    """Write tokenized records to Levanter cache format."""
    from levanter.store.cache import CacheMetadata, SerialCacheWriter

    ensure_parent_dir(output_path)
    record_iter = iter(records)
    tmp_path = f"{output_path}.tmp"
    fs = fsspec.core.url_to_fs(output_path)[0]

    if fs.exists(output_path) and fs.exists(tmp_path):
        logger.info("Removing stale temporary cache %s because %s already exists", tmp_path, output_path)
        fs.rm(tmp_path, recursive=True)

    try:
        exemplar = next(record_iter)
    except StopIteration:
        return {"path": output_path, "count": 0}

    count = 1
    logger.info("write_levanter_cache: starting write to %s", output_path)

    existing_rows = 0 if fs.exists(output_path) else _get_existing_row_count(tmp_path, exemplar)

    if existing_rows > 0:
        logger.info("Resuming write to %s from %d existing rows", output_path, existing_rows)
        # we already consumed 1 record (exemplar), skip existing_rows - 1 more
        rows_to_skip = existing_rows - 1
        skipped_rows = 0
        for _record in itertools.islice(record_iter, rows_to_skip):
            skipped_rows += 1
            count += 1
        if skipped_rows != rows_to_skip:
            raise ValueError(
                f"Temporary cache at {tmp_path} has {existing_rows} rows, but input has only {skipped_rows + 1} rows"
            )
        mode = "a"
        write_exemplar = False
    else:
        mode = "w"
        write_exemplar = True

    with SerialCacheWriter(
        tmp_path, exemplar, shard_name=output_path, metadata=CacheMetadata(metadata), mode=mode
    ) as writer:
        if write_exemplar:
            writer.write_batch([exemplar])
        for batch in batchify(record_iter):
            writer.write_batch(batch)
            count += len(batch)
            if count % 1000 == 0:
                logger.info("write_levanter_cache: %s — %d records so far", output_path, count)

    logger.info("write_levanter_cache: finished %s — %d records", output_path, count)

    _promote_tmp_cache(fs, tmp_path, output_path)

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
