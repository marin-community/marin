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

import msgspec


def load_jsonl(file_path: str) -> Iterator[dict]:
    """Load JSONL file and yield records.

    Handles gzip and zstd compression automatically.
    Use with .flat_map() to read files containing multiple records.

    Args:
        file_path: Path to JSONL file (local or remote, .gz and .zst supported)

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
    import gzip
    import io

    decoder = msgspec.json.Decoder()

    if file_path.endswith(".zst"):
        import zstandard as zstd

        dctx = zstd.ZstdDecompressor()
        with open(file_path, "rb") as raw_f:
            with dctx.stream_reader(raw_f) as reader:
                text_f = io.TextIOWrapper(reader, encoding="utf-8")
                for line in text_f:
                    line = line.strip()
                    if line:
                        yield decoder.decode(line)
    elif file_path.endswith(".gz"):
        with gzip.open(file_path, "rt") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield decoder.decode(line)
    else:
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield decoder.decode(line)


def load_parquet(file_path: str, **kwargs) -> Iterator[dict]:
    """Load Parquet file and yield records as dicts.

    Use with .flat_map() to read Parquet files containing multiple records.

    Args:
        file_path: Path to Parquet file
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

    df = pd.read_parquet(file_path, **kwargs)
    for _, row in df.iterrows():
        yield row.to_dict()


def load_file(file_path: str, **parquet_kwargs) -> Iterator[dict]:
    """Load records from file, auto-detecting JSONL or Parquet format.

    Detects format based on file extension and delegates to load_jsonl or load_parquet.
    Supports .jsonl, .jsonl.gz, .jsonl.zstd, jsonl.xz, and .parquet files.

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
    ):
        yield from load_jsonl(file_path)
    elif file_path.endswith(".parquet"):
        yield from load_parquet(file_path, **parquet_kwargs)
    else:
        raise ValueError(
            f"Unsupported extension: {file_path}. Supported formats: .jsonl, .jsonl.gz, .jsonl.zstd, .jsonl.xz, .parquet"
        )


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
