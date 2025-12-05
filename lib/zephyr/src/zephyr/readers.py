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
import logging
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal

import fsspec
import msgspec
import numpy as np
import pyarrow as pa

from zephyr.expr import Expr

logger = logging.getLogger(__name__)


@dataclass
class InputFileSpec:
    """Specification for reading a file or portion of a file.

    Used by LoadFileOp to specify what file to read and optional chunking.

    Attributes:
        path: Path to the file
        format: File format ("parquet", "jsonl", or "auto" to detect)
        columns: Optional column projection (for parquet)
        row_start: Optional start row for chunked reading
        row_end: Optional end row for chunked reading
        filter_expr: Optional filter expression for pushdown
    """

    path: str
    format: Literal["parquet", "jsonl", "auto"] = "auto"
    columns: list[str] | None = None
    row_start: int | None = None
    row_end: int | None = None
    filter_expr: Expr | None = field(default=None)


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


def load_jsonl(source: str | InputFileSpec) -> Iterator[dict]:
    """Load a JSONL file and yield parsed records as dictionaries.

    If the input file is compressed (.gz, .zst, .xz), it will be automatically
    decompressed during loading.

    Args:
        source: Path to JSONL file or InputFileSpec containing the path.
            Supports: local paths, gs://, s3://, hf://datasets/{repo}@{rev}/{path}

    Yields:
        Parsed JSON records as dictionaries

    Example:
        >>> backend = create_backend("ray", max_parallelism=10)
        >>> # Load from cloud storage
        >>> ds = (Dataset
        ...     .from_files("gs://bucket/data", "**/*.jsonl.gz")
        ...     .load_jsonl()
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
    if isinstance(source, InputFileSpec):
        file_path = source.path
    else:
        file_path = source
    decoder = msgspec.json.Decoder()

    with open_file(file_path, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                yield decoder.decode(line)


def load_parquet(source: str | InputFileSpec, **kwargs) -> Iterator[dict]:
    """Load Parquet file and yield records as dicts.

    When given an InputFileSpec with row_start/row_end, reads only the exact rows
    in that range. Row groups are read efficiently (only overlapping groups are loaded),
    then rows are filtered to the precise range. When filter_expr is provided, the filter
    is pushed down to PyArrow for efficient filtering at read time.

    Args:
        source: Path to Parquet file or InputFileSpec containing the path and optional row range.
        **kwargs: Additional arguments to the ParquetFile.iter_batches() method

    Yields:
        Records as dictionaries

    Example:
        >>> backend = create_backend("ray", max_parallelism=10)
        >>> ds = (Dataset
        ...     .from_files("/input", "**/*.parquet")
        ...     .load_parquet()
        ...     .map(lambda r: transform_record(r))
        ...     .write_jsonl("/output/data-{shard:05d}.jsonl.gz")
        ... )
        >>> output_files = list(backend.execute(ds))
    """
    import pyarrow.parquet as pq

    filter_expr = None
    if isinstance(source, InputFileSpec):
        file_path = source.path
        row_start = source.row_start
        row_end = source.row_end
        filter_expr = source.filter_expr
        if source.columns is not None:
            kwargs = {**kwargs, "columns": source.columns}
    else:
        file_path = source
        row_start = None
        row_end = None

    # Convert filter expression to PyArrow if provided
    pa_filter = None
    if filter_expr is not None:
        from zephyr.expr import to_pyarrow_expr

        pa_filter = to_pyarrow_expr(filter_expr)

    with open_file(file_path, "rb") as f:
        parquet_file = pq.ParquetFile(f)

        if row_start is not None and row_end is not None:
            # Read only row groups that overlap with [row_start, row_end)
            row_groups_to_read = []
            first_rg_start = 0  # Global row index where first selected row group starts
            cumulative_rows = 0
            for i in range(parquet_file.metadata.num_row_groups):
                rg = parquet_file.metadata.row_group(i)
                rg_start = cumulative_rows
                rg_end = cumulative_rows + rg.num_rows

                if rg_end > row_start and rg_start < row_end:
                    if not row_groups_to_read:
                        first_rg_start = rg_start
                    row_groups_to_read.append(i)

                cumulative_rows = rg_end
                if cumulative_rows >= row_end:
                    break

            if pa_filter is not None:
                # Use read_table with filter for row group subset
                table = pq.read_table(
                    f,
                    columns=kwargs.get("columns"),
                    filters=pa_filter,
                )
                # Slice to exact row range
                slice_start = row_start - first_rg_start
                slice_end = row_end - first_rg_start
                yield from table.slice(slice_start, slice_end - slice_start).to_pylist()
            else:
                # Read selected row groups and filter to exact row range
                global_row_idx = first_rg_start
                for batch in parquet_file.iter_batches(row_groups=row_groups_to_read, **kwargs):
                    for record in batch.to_pylist():
                        if global_row_idx >= row_start and global_row_idx < row_end:
                            yield record
                        global_row_idx += 1
                        if global_row_idx >= row_end:
                            return
        elif pa_filter is not None:
            # Use read_table with filter (ParquetFile.iter_batches doesn't support filter)
            table = pq.read_table(
                f,
                columns=kwargs.get("columns"),
                filters=pa_filter,
            )
            yield from table.to_pylist()
        else:
            for batch in parquet_file.iter_batches(**kwargs):
                yield from batch.to_pylist()


def load_vortex(
    source: str | InputFileSpec,
    columns: list[str] | None = None,
) -> Iterator[dict]:
    """Load records from a Vortex file with optional pushdown.

    Uses Vortex's PyArrow Dataset interface for filter/column pushdown.
    Supports row-range reading via take() for chunked parallel execution.

    Args:
        source: Path to .vortex file or InputFileSpec
        columns: Optional column projection

    Yields:
        Records as dictionaries

    Example:
        >>> backend = create_backend("ray", max_parallelism=10)
        >>> ds = (Dataset
        ...     .from_files("/input/**/*.vortex")
        ...     .load_vortex()
        ...     .filter(lambda r: r["score"] > 0.5)
        ...     .write_jsonl("/output/filtered-{shard:05d}.jsonl.gz")
        ... )
        >>> output_files = list(backend.execute(ds))
    """
    import vortex

    if isinstance(source, InputFileSpec):
        file_path = source.path
        columns = source.columns or columns
        filter_expr = source.filter_expr
        row_start = source.row_start
        row_end = source.row_end
    else:
        file_path = source
        filter_expr = None
        row_start = None
        row_end = None

    # Convert filter to PyArrow expression if provided
    pa_filter = None
    if filter_expr is not None:
        from zephyr.expr import to_pyarrow_expr

        pa_filter = to_pyarrow_expr(filter_expr)

    # Open vortex file and get PyArrow Dataset interface
    logger.info("Loading: %s", file_path)
    vf = vortex.open(file_path)
    dataset = vf.to_dataset()

    if row_start is not None and row_end is not None:
        # convert [row_start, row_end) to list of indices, using a np.uint64 array
        indices = np.arange(row_start, row_end, dtype=np.uint64)
        indices = pa.array(indices)
        table = dataset.take(indices, columns=columns, filter=pa_filter)
        yield from table.to_pylist()
    else:
        # Full file scan
        table = dataset.to_table(columns=columns, filter=pa_filter)
        yield from table.to_pylist()


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
        ".vortex",
    ]
)


def load_file(source: str | InputFileSpec, columns: list[str] | None = None, **parquet_kwargs) -> Iterator[dict]:
    """Load records from file, auto-detecting JSONL or Parquet format.

    Args:
        source: Path to file or InputFileSpec containing the path.
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
        ...     .load_file()
        ...     .filter(lambda r: r["score"] > 0.5)
        ...     .write_jsonl("/output/data-{shard:05d}.jsonl.gz")
        ... )
        >>> output_files = list(backend.execute(ds))
        >>> # Select specific columns
        >>> ds = (Dataset
        ...     .from_files("/input/**/*.parquet")
        ...     .load_file(columns=["id", "text", "score"])
        ...     .write_jsonl("/output/data-{shard:05d}.jsonl.gz")
        ... )
    """
    # Extract file path and optional filter/columns from InputFileSpec
    filter_expr = None
    spec_columns = None
    if isinstance(source, InputFileSpec):
        file_path = source.path
        filter_expr = source.filter_expr
        spec_columns = source.columns
    else:
        file_path = source

    # Merge columns from source spec and explicit argument
    effective_columns = columns if columns is not None else spec_columns

    if not file_path.endswith(SUPPORTED_EXTENSIONS):
        raise ValueError(f"Unsupported extension: {file_path}.")
    if file_path.endswith(".parquet"):
        if effective_columns is not None:
            parquet_kwargs = {**parquet_kwargs, "columns": effective_columns}
        yield from load_parquet(source, **parquet_kwargs)
    elif file_path.endswith(".vortex"):
        yield from load_vortex(source, columns=effective_columns)
    else:
        # For JSONL, apply filter and column selection manually
        filter_fn = filter_expr.evaluate if filter_expr is not None else None
        for record in load_jsonl(source):
            # Apply filter if provided
            if filter_fn is not None and not filter_fn(record):
                continue
            # Apply column selection if provided
            if effective_columns is not None:
                yield {k: v for k, v in record.items() if k in effective_columns}
            else:
                yield record


def load_zip_members(source: str | InputFileSpec, pattern: str = "*") -> Iterator[dict]:
    """Load zip members matching pattern, yielding filename and content.

    Opens zip file (supports fsspec paths like gs://), finds members matching
    the pattern, and yields dicts with 'filename' and 'content' (bytes).

    Args:
        source: Path to zip file or InputFileSpec containing the path.
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
    zip_path = source.path if isinstance(source, InputFileSpec) else source
    with fsspec.open(zip_path, "rb") as f:
        with zipfile.ZipFile(f) as zf:
            for member_name in zf.namelist():
                if not member_name.endswith("/") and fnmatch.fnmatch(member_name, pattern):
                    with zf.open(member_name, "r") as member_file:
                        yield {
                            "filename": member_name,
                            "content": member_file.read(),
                        }
