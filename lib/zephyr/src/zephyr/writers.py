# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Writers for common output formats."""

from __future__ import annotations

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


@contextmanager
def atomic_rename(output_path: str) -> Iterable[str]:
    """Context manager for atomic write-and-rename.

    Yields a temporary path to write to. On successful exit, atomically renames
    the temp file to the final path. On failure, cleans up the temp file.

    Example:
        with atomic_rename("output.jsonl.gz") as tmp_path:
            write_data(tmp_path)
        # File is now at output.jsonl.gz
    """
    temp_path = f"{output_path}.tmp"
    fs = fsspec.core.url_to_fs(output_path)[0]

    try:
        yield temp_path
        # not so atomic if on a remote FS and recursive, but ...
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


def write_levanter_cache(records: Iterable[dict[str, Any]], output_path: str, metadata: dict[str, Any]) -> dict:
    """Write tokenized records to Levanter cache format."""
    from levanter.store.cache import SerialCacheWriter

    ensure_parent_dir(output_path)
    from levanter.store.cache import CacheMetadata

    try:
        exemplar = next(iter(records))
    except StopIteration:
        return {"path": output_path, "count": 0}

    count = 1
    with atomic_rename(output_path) as tmp_path:
        with SerialCacheWriter(tmp_path, exemplar, shard_name=output_path, metadata=CacheMetadata(metadata)) as writer:
            writer.write_batch([exemplar])
            for batch in batchify(records):
                writer.write_batch(batch)
                count += len(batch)

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
