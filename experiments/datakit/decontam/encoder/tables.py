# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Parquet tables on object storage or local disk, read and written whole, and shard file names."""

import io

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath, prefix_join


def read_table(path: str, columns: list[str] | None = None) -> pa.Table:
    return pq.read_table(io.BytesIO(StoragePath(path).read_bytes()), columns=columns)


def read_frame(path: str, columns: list[str] | None = None) -> pd.DataFrame:
    return read_table(path, columns).to_pandas()


def write_table(table: pa.Table, path: str) -> None:
    buffer = io.BytesIO()
    pq.write_table(table, buffer, compression="zstd")
    StoragePath(path).write_bytes(buffer.getvalue())


def write_frame(frame: pd.DataFrame, path: str) -> None:
    write_table(pa.Table.from_pandas(frame, preserve_index=False), path)


def parquet_files(prefix: str, pattern: str = "*.parquet") -> list[str]:
    """The sorted paths under *prefix* that match *pattern*."""
    return sorted(str(path) for path in StoragePath(prefix_join(prefix, pattern)).glob())


# Path segment under which the normalized pretraining shards live.
NORMALIZED = "/normalized/"


def shard_name(path: str) -> str:
    """A flat name for one normalized shard: its path below ``normalized/``, with ``/`` written as ``__``."""
    if NORMALIZED not in path:
        raise ValueError(f"{path} is not under a {NORMALIZED} prefix")
    return path.split(NORMALIZED, 1)[1].replace("/", "__")
