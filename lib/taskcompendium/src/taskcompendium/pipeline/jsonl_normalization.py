# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Normalize mixed JSONL inputs into component-filterable Parquet."""

import hashlib
import io
import json
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.atomic import atomic_rename
from rigging.filesystem.storage_path import StoragePath
from zephyr.readers import open_file
from zephyr.writers import ensure_parent_dir, parquet_sink

ROW_GROUP_BYTES = 8 * 1024 * 1024
READ_BUFFER_BYTES = 1024 * 1024
NORMALIZED_SCHEMA = pa.schema(
    [("component", pa.string()), ("source_index", pa.int64()), ("raw_json", pa.large_binary())]
)


def normalized_jsonl_path(file: StoragePath) -> StoragePath:
    return file.parent / ".curation-parquet" / (file.name + ".parquet")


@dataclass
class _Buffer:
    rows: list[dict] = field(default_factory=list)
    bytes: int = 0
    count: int = 0


def _flush(buffer: _Buffer, writer: pq.ParquetWriter) -> None:
    if not buffer.rows:
        return
    writer.write_table(pa.Table.from_pylist(buffer.rows, schema=NORMALIZED_SCHEMA), row_group_size=len(buffer.rows))
    buffer.rows.clear()
    buffer.bytes = 0


def normalize_jsonl(path: str, component: Callable[[bytes], str], *, row_group_bytes: int = ROW_GROUP_BYTES) -> dict:
    """Write one shared Parquet file, retaining original record bytes and native row indices.

    Blank lines do not advance the native row index. Groups contain one component;
    ``row_group_bytes`` bounds their raw payload, except for an oversized single record.
    Return the input hash, byte/row counts and per-component counts.

    Acquisition runs this once per file as a Zephyr task. Its generic Parquet
    writer combines batches across component boundaries, defeating component
    filter pushdown, so this writer flushes each component separately.
    """
    destination = normalized_jsonl_path(StoragePath(path))
    ensure_parent_dir(str(destination))
    buffers: dict[str, _Buffer] = defaultdict(_Buffer)
    source_hash = hashlib.sha256()
    source_bytes = 0
    count = 0
    with atomic_rename(str(destination)) as temporary:
        with parquet_sink(temporary) as (output, filesystem):
            # Exclude large raw values from footer statistics and dictionaries.
            with pq.ParquetWriter(
                output,
                NORMALIZED_SCHEMA,
                filesystem=filesystem,
                compression="zstd",
                use_dictionary=["component"],
                write_statistics=["component", "source_index"],
            ) as writer:
                # fsspec's binary readline fetches a whole cache block, then seeks
                # back per record; buffering keeps the underlying reads sequential.
                with (
                    open_file(path, "rb") as stream,
                    io.BufferedReader(cast(io.RawIOBase, stream), buffer_size=READ_BUFFER_BYTES) as records,
                ):
                    for line in records:
                        source_hash.update(line)
                        source_bytes += len(line)
                        if not line.strip():
                            continue
                        label = component(line)
                        buffer = buffers[label]
                        if buffer.bytes + len(line) > row_group_bytes:
                            _flush(buffer, writer)
                        buffer.rows.append({"component": label, "source_index": count, "raw_json": line})
                        buffer.bytes += len(line)
                        buffer.count += 1
                        count += 1
                        if buffer.bytes >= row_group_bytes:
                            _flush(buffer, writer)
                for buffer in buffers.values():
                    _flush(buffer, writer)
                result = {
                    "source_sha256": source_hash.hexdigest(),
                    "source_bytes": source_bytes,
                    "source_rows": count,
                    "components": {label: buffer.count for label, buffer in buffers.items()},
                }
                writer.add_key_value_metadata({"curation_source": json.dumps(result, sort_keys=True)})
    return result
