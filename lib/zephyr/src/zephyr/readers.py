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

"""Readers for common input formats.

Supports reading from local filesystems, cloud storage (gs://, s3://) and HuggingFace Hub (hf://) via fsspec.
"""

from __future__ import annotations

import fnmatch
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager

import fsspec
import msgspec

# Register HuggingFace filesystem with authentication if HF_TOKEN is available
# This enables reading from hf:// URLs throughout the codebase
try:
    from huggingface_hub import HfFileSystem

    fsspec.register_implementation("hf", HfFileSystem, clobber=True)
except ImportError:
    # HuggingFace Hub is optional - only needed for hf:// URLs
    pass


@contextmanager
def open_file(file_path: str, mode: str = "rb"):
    """Open `file_path` with sensible defaults for compression and caching."""

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
    """Load a JSONL file and yield parsed records as dictionaries.

    If the input file is compressed (.gz, .zst, .xz), it will be automatically
    decompressed during loading.

    Args:
        file_path: Path to JSONL file (local, remote, or HuggingFace Hub)
            Supports: local paths, gs://, s3://, hf://datasets/{repo}@{rev}/{path}

    Yields:
        Parsed JSON records as dictionaries

    Example:
        >>> backend = create_backend("ray", max_parallelism=10)
        >>> # Load from cloud storage
        >>> ds = (Dataset
        ...     .from_files("gs://bucket/data", "**/*.jsonl.gz")
        ...     .flat_map(load_jsonl)  # Each file yields many records
        ...     .filter(lambda r: r["score"] > 0.5)
        ...     .write_jsonl("/output/filtered-{shard:05d}.jsonl.gz")
        ... )
        >>> output_files = list(backend.execute(ds))
        >>>
        >>> # Load from HuggingFace Hub (requires HF_TOKEN env var)
        >>> hf_url = "hf://datasets/username/dataset@main/data/train.jsonl.gz"
        >>> ds = Dataset.from_list([hf_url]).flat_map(load_jsonl)
        >>> records = list(backend.execute(ds))
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
        **kwargs: Additional arguments to the ParquetFile.iter_batches() method

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
    import pyarrow.parquet as pq

    with open_file(file_path, "rb") as f:
        # TODO: should we also expose kwargs for ParquetFile constructor?
        parquet_file = pq.ParquetFile(f)
        for batch in parquet_file.iter_batches(**kwargs):
            yield from batch.to_pylist()


SUPPORTED_EXTENSIONS = tuple(
    [
        ".json",
        ".json.gz",
        ".json.xz",
        ".json.zst",
        ".json.zstd",
        ".jsonl",
        ".jsonl.gz",
        ".jsonl.xz",
        ".jsonl.zst",
        ".jsonl.zstd",
        ".parquet",
    ]
)


def load_file(file_path: str, columns: list[str] | None = None, **parquet_kwargs) -> Iterator[dict]:
    """Load records from file, auto-detecting JSONL or Parquet format.

    Args:
        file_path: Path to file (local or remote)
        columns: Optional list of column names to select. For Parquet files, this
            enables efficient column pushdown. For JSONL files, only specified
            columns that exist in each record are included.
        **parquet_kwargs: Additional arguments for load_parquet (ignored for JSONL)

    Yields:
        Parsed records as dictionaries

    Raises:
        ValueError: If file extension is not supported

    Example:
        >>> backend = create_backend("ray", max_parallelism=10)
        >>> # Works with mixed JSONL and Parquet files
        >>> ds = (Dataset
        ...     .from_files("/input/**/*.jsonl")
        ...     .flat_map(load_file)
        ...     .filter(lambda r: r["score"] > 0.5)
        ...     .write_jsonl("/output/data-{shard:05d}.jsonl.gz")
        ... )
        >>> output_files = list(backend.execute(ds))
        >>> # Select specific columns
        >>> ds = (Dataset
        ...     .from_files("/input/**/*.parquet")
        ...     .flat_map(lambda p: load_file(p, columns=["id", "text", "score"]))
        ...     .write_jsonl("/output/data-{shard:05d}.jsonl.gz")
        ... )
    """
    if not file_path.endswith(SUPPORTED_EXTENSIONS):
        raise ValueError(f"Unsupported extension: {file_path}.")
    if file_path.endswith(".parquet"):
        if columns is not None:
            parquet_kwargs = {**parquet_kwargs, "columns": columns}
        yield from load_parquet(file_path, **parquet_kwargs)
    else:
        for record in load_jsonl(file_path):
            if columns is not None:
                yield {k: v for k, v in record.items() if k in columns}
            else:
                yield record


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
    with fsspec.open(zip_path, "rb") as f:
        with zipfile.ZipFile(f) as zf:
            for member_name in zf.namelist():
                if not member_name.endswith("/") and fnmatch.fnmatch(member_name, pattern):
                    with zf.open(member_name, "r") as member_file:
                        yield {
                            "filename": member_name,
                            "content": member_file.read(),
                        }
