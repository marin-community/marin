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

"""Readers for common input formats."""

from __future__ import annotations

import fnmatch
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager

import msgspec


@contextmanager
def open_file(file_path: str, mode: str = "rb"):
    """Open file using fsspec, handling both local and remote paths.

    Supports automatic decompression for .gz, .zst, and .xz files.
    Uses background caching for remote files to improve read performance.

    Args:
        file_path: Path to file (local or remote via fsspec)
        mode: File mode ('rb' for binary, 'rt' for text)

    Yields:
        File-like object (binary or text depending on mode)
    """
    import fsspec

    compression = None
    if file_path.endswith(".gz"):
        compression = "gzip"
    elif file_path.endswith(".zst"):
        compression = "zstd"
    elif file_path.endswith(".xz"):
        compression = "xz"

    with fsspec.open(
        file_path,
        mode,
        compression=compression,
        block_size=16_000_000,
        cache_type="background",
        maxblocks=2,
    ) as f:
        yield f


def load_jsonl(file_path: str) -> Iterator[dict]:
    """Load JSONL file and yield records.
    Handles gzip, zstd, and xz compression automatically.

    Args:
        file_path: Path to JSONL file (local or remote, .gz, .zst, and .xz supported)

    Yields:
        Parsed JSON records as dictionaries

    Example:
        >>> backend = create_backend("ray", max_parallelism=10)
        >>> ds = (Dataset
        ...     .from_files("/input", "**/*.jsonl.gz")
        ...     .flat_map(load_jsonl)  # Each file yields many records
        ...     .filter(lambda r: r["score"] > 0.5)
        ...     .write_jsonl("/output/filtered-{shard:05d}.jsonl.gz")
        ... )
        >>> output_files = list(backend.execute(ds))
    """
    decoder = msgspec.json.Decoder()

    with open_file(file_path, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                yield decoder.decode(line)


def load_parquet(file_path: str, **kwargs) -> Iterator[dict]:
    """Load Parquet file and yield records as dicts.

    Args:
        file_path: Path to Parquet file (local or remote)
        **kwargs: Additional arguments for pd.read_parquet (e.g., columns, engine)

    Yields:
        Records as dictionaries

    Example:
        >>> backend = create_backend("ray", max_parallelism=10)
        >>> ds = (Dataset
        ...     .from_files("/input", "**/*.parquet")
        ...     .flat_map(load_parquet)
        ...     .map(lambda r: transform_record(r))
        ...     .write_jsonl("/output/data-{shard:05d}.jsonl.gz")
        ... )
        >>> output_files = list(backend.execute(ds))
    """
    import pandas as pd

    with open_file(file_path, "rb") as f:
        df = pd.read_parquet(f, **kwargs)
        for _, row in df.iterrows():
            yield row.to_dict()


def load_file(file_path: str, **parquet_kwargs) -> Iterator[dict]:
    """Load records from file, auto-detecting JSONL or Parquet format.

    Args:
        file_path: Path to file (local or remote)
        **parquet_kwargs: Additional arguments for load_parquet (ignored for JSONL)

    Yields:
        Parsed records as dictionaries

    Raises:
        ValueError: If file extension is not supported

    Example:
        >>> backend = create_backend("ray", max_parallelism=10)
        >>> # Works with mixed JSONL and Parquet files
        >>> ds = (Dataset
        ...     .from_files("/input", "**/*")
        ...     .flat_map(load_file)
        ...     .filter(lambda r: r["score"] > 0.5)
        ...     .write_jsonl("/output/data-{shard:05d}.jsonl.gz")
        ... )
        >>> output_files = list(backend.execute(ds))
    """
    if (
        file_path.endswith(".jsonl")
        or file_path.endswith(".jsonl.gz")
        or file_path.endswith(".jsonl.zstd")
        or file_path.endswith(".jsonl.xz")
        or file_path.endswith(".json")
        or file_path.endswith(".json.gz")
        or file_path.endswith(".json.zstd")
        or file_path.endswith(".json.xz")
    ):
        yield from load_jsonl(file_path)
    elif file_path.endswith(".parquet"):
        yield from load_parquet(file_path, **parquet_kwargs)
    else:
        raise ValueError(f"Unsupported extension: {file_path}.")


def load_zip_members(zip_path: str, pattern: str = "*") -> Iterator[dict]:
    """Load zip members matching pattern, yielding filename and content.

    Opens zip file (supports fsspec paths like gs://), finds members matching
    the pattern, and yields dicts with 'filename' and 'content' (bytes).

    Args:
        zip_path: Path to zip file (local or remote via fsspec)
        pattern: Glob pattern to match member names (default: "*")

    Yields:
        Dicts with 'filename' (str) and 'content' (bytes)

    Example:
        >>> backend = create_backend("ray", max_parallelism=10)
        >>> ds = (Dataset
        ...     .from_list(["gs://bucket/data.zip"])
        ...     .flat_map(lambda p: load_zip_members(p, pattern="test.jsonl"))
        ...     .map(lambda m: process_file(m["filename"], m["content"]))
        ... )
        >>> output_files = list(backend.execute(ds))
    """

    import fsspec

    with fsspec.open(zip_path, "rb") as f:
        with zipfile.ZipFile(f) as zf:
            for member_name in zf.namelist():
                if not member_name.endswith("/") and fnmatch.fnmatch(member_name, pattern):
                    with zf.open(member_name, "r") as member_file:
                        yield {
                            "filename": member_name,
                            "content": member_file.read(),
                        }
