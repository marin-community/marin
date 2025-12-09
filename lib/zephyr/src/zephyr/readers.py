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

    Attributes:
        path: Path to the file
        format: File format ("parquet", "jsonl", or "auto" to detect)
        columns: List of columns to read
        row_start: Optional start row for chunked reading
        row_end: Optional end row for chunked reading
        filter_expr: Optional filter expression to apply
    """

    path: str
    format: Literal["parquet", "jsonl", "auto"] = "auto"
    columns: list[str] | None = None
    row_start: int | None = None
    row_end: int | None = None
    filter_expr: Expr | None = field(default=None)


def _as_spec(source: str | InputFileSpec) -> InputFileSpec:
    """Normalize source to InputFileSpec for consistent downstream handling."""
    if isinstance(source, InputFileSpec):
        return source
    return InputFileSpec(path=source)


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
    spec = _as_spec(source)
    decoder = msgspec.json.Decoder()

    with open_file(spec.path, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                yield decoder.decode(line)


def load_parquet(source: str | InputFileSpec) -> Iterator[dict]:
    """Load Parquet file and yield records as dicts.

    When given an InputFileSpec with row_start/row_end, reads only the exact rows
    in that range. Row groups are read efficiently (only overlapping groups are loaded),
    then rows are filtered to the precise range. When filter_expr is provided, the filter
    is pushed down to PyArrow for efficient filtering at read time.

    Args:
        source: Path to Parquet file or InputFileSpec containing the path, columns,
            row range, and filter expression.

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
    import pyarrow.dataset as pads

    spec = _as_spec(source)
    columns = spec.columns

    pa_filter = None
    if spec.filter_expr is not None:
        from zephyr.expr import to_pyarrow_expr

        pa_filter = to_pyarrow_expr(spec.filter_expr)

    dataset = pads.dataset(spec.path, format="parquet")

    if spec.row_start is not None and spec.row_end is not None:
        # Row range first: select rows by position, then apply filter
        cumulative_rows = 0
        for fragment in dataset.get_fragments():
            for rg_fragment in fragment.split_by_row_group():
                # Get row group size from RowGroupInfo (no data read)
                rg_info = rg_fragment.row_groups[0]
                rg_num_rows = rg_info.num_rows
                rg_start = cumulative_rows
                rg_end = cumulative_rows + rg_num_rows

                if rg_end > spec.row_start and rg_start < spec.row_end:
                    is_interior = rg_start >= spec.row_start and rg_end <= spec.row_end

                    if is_interior:
                        # Entirely within range: push filter down, yield all
                        table = rg_fragment.to_table(columns=columns, filter=pa_filter)
                        yield from table.to_pylist()
                    else:
                        # Boundary row group: slice first, then filter
                        table = rg_fragment.to_table(columns=columns)
                        local_start = max(0, spec.row_start - rg_start)
                        local_end = min(rg_num_rows, spec.row_end - rg_start)
                        sliced = table.slice(local_start, local_end - local_start)

                        if pa_filter is not None:
                            yield from sliced.filter(pa_filter).to_pylist()
                        else:
                            yield from sliced.to_pylist()

                cumulative_rows = rg_end
                if cumulative_rows >= spec.row_end:
                    return
    elif pa_filter is not None:
        table = dataset.to_table(columns=columns, filter=pa_filter)
        yield from table.to_pylist()
    else:
        for batch in dataset.to_batches(columns=columns):
            yield from batch.to_pylist()


def load_vortex(source: str | InputFileSpec) -> Iterator[dict]:
    """Load records from a Vortex file with optional pushdown.

    Uses Vortex's PyArrow Dataset interface for filter/column pushdown.
    Supports row-range reading via take() for chunked parallel execution.

    Args:
        source: Path to .vortex file or InputFileSpec containing the path,
            columns, row range, and filter expression.

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

    spec = _as_spec(source)
    columns = spec.columns

    # Convert filter to PyArrow expression if provided
    pa_filter = None
    if spec.filter_expr is not None:
        from zephyr.expr import to_pyarrow_expr

        pa_filter = to_pyarrow_expr(spec.filter_expr)

    # Open vortex file and get PyArrow Dataset interface
    logger.info("Loading: %s", spec.path)
    vf = vortex.open(spec.path)
    dataset = vf.to_dataset()

    if spec.row_start is not None and spec.row_end is not None:
        indices = np.arange(spec.row_start, spec.row_end, dtype=np.uint64)
        indices = pa.array(indices)
        table = dataset.take(indices, columns=columns, filter=pa_filter)
        yield from table.to_pylist()
    else:
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


def load_file(source: str | InputFileSpec) -> Iterator[dict]:
    """Load records from file, auto-detecting JSONL, Parquet, or Vortex format.

    Args:
        source: Path to file or InputFileSpec containing the path, columns,
            row range, and filter expression.

    Yields:
        Parsed records as dictionaries

    Raises:
        ValueError: If file extension is not supported

    Example:
        >>> backend = create_backend("ray", max_parallelism=10)
        >>> ds = (Dataset
        ...     .from_files("/input/**/*.jsonl")
        ...     .load_file()
        ...     .filter(lambda r: r["score"] > 0.5)
        ...     .write_jsonl("/output/data-{shard:05d}.jsonl.gz")
        ... )
        >>> output_files = list(backend.execute(ds))
    """
    spec = _as_spec(source)

    if not spec.path.endswith(SUPPORTED_EXTENSIONS):
        raise ValueError(f"Unsupported extension: {spec.path}.")

    if spec.path.endswith(".parquet"):
        yield from load_parquet(spec)
    elif spec.path.endswith(".vortex"):
        yield from load_vortex(spec)
    else:
        # For JSONL, apply filter and column selection manually
        filter_fn = spec.filter_expr.evaluate if spec.filter_expr is not None else None
        for record in load_jsonl(spec):
            if filter_fn is not None and not filter_fn(record):
                continue
            if spec.columns is not None:
                yield {k: v for k, v in record.items() if k in spec.columns}
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
    spec = _as_spec(source)
    with fsspec.open(spec.path, "rb") as f:
        with zipfile.ZipFile(f) as zf:
            for member_name in zf.namelist():
                if not member_name.endswith("/") and fnmatch.fnmatch(member_name, pattern):
                    with zf.open(member_name, "r") as member_file:
                        yield {
                            "filename": member_name,
                            "content": member_file.read(),
                        }
