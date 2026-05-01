# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Readers for common input formats.

Supports reading from local filesystems, cloud storage (gs://, s3://) and HuggingFace Hub (hf://) via fsspec.
"""

from __future__ import annotations

import fnmatch
import logging
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

import fsspec
import msgspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem import open_url, url_to_fs

from zephyr import counters
from zephyr.expr import Expr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared Parquet row-group reader
# ---------------------------------------------------------------------------


def _check_row_group_statistics(
    rg_meta: pq.RowGroupMetaData,
    equality_predicates: dict[str, object],
) -> bool:
    """Return False if row group min/max statistics prove no rows can match."""
    for col_idx in range(rg_meta.num_columns):
        col_meta = rg_meta.column(col_idx)
        name = col_meta.path_in_schema
        if name not in equality_predicates:
            continue
        stats = col_meta.statistics
        if stats is None or not stats.has_min_max:
            continue  # no stats — assume it could match
        value = equality_predicates[name]
        if value < stats.min or value > stats.max:
            return False
    return True


def iter_parquet_row_groups(
    source: str | pq.ParquetFile,
    *,
    columns: list[str] | None = None,
    row_start: int | None = None,
    row_end: int | None = None,
    equality_predicates: dict[str, object] | None = None,
) -> Iterator[pa.Table]:
    """Yield one ``pa.Table`` per qualifying row group with O(row_group) memory.

    Uses ``pq.ParquetFile`` instead of ``pyarrow.dataset`` to avoid the
    upstream memory leak (https://github.com/apache/arrow/issues/39808).

    Args:
        source: Path to parquet file or an already-open ``pq.ParquetFile``.
        columns: Columns to read (``None`` for all).
        row_start: First row to include (inclusive, before filtering).
        row_end: Last row to include (exclusive, before filtering).
        equality_predicates: Column-value pairs for row group skipping and
            row-level filtering.  Row groups whose min/max statistics exclude
            the target value are not read at all, and within matching groups
            only rows where every predicate column equals its target are kept.
    """
    pf = pq.ParquetFile(source) if isinstance(source, str) else source
    has_row_range = row_start is not None and row_end is not None

    # If caller requests specific columns, ensure predicate columns are
    # also read so row-level filtering works; drop them before yielding.
    read_columns = columns
    drop_columns: list[str] = []
    if columns is not None and equality_predicates:
        extra = [c for c in equality_predicates if c not in columns]
        if extra:
            read_columns = list(columns) + extra
            drop_columns = extra

    cumulative_rows = 0

    for i in range(pf.metadata.num_row_groups):
        rg_meta = pf.metadata.row_group(i)
        rg_num_rows = rg_meta.num_rows
        rg_start = cumulative_rows
        rg_end = cumulative_rows + rg_num_rows
        cumulative_rows = rg_end

        if equality_predicates and not _check_row_group_statistics(rg_meta, equality_predicates):
            continue

        if has_row_range:
            assert row_start is not None and row_end is not None
            if rg_end <= row_start:
                continue
            if rg_start >= row_end:
                return

        table = pf.read_row_group(i, columns=read_columns)

        if has_row_range:
            assert row_start is not None and row_end is not None
            is_interior = rg_start >= row_start and rg_end <= row_end
            if not is_interior:
                local_start = max(0, row_start - rg_start)
                local_end = min(rg_num_rows, row_end - rg_start)
                table = table.slice(local_start, local_end - local_start)

        if equality_predicates:
            for col_name, value in equality_predicates.items():
                mask = pa.compute.equal(table.column(col_name), value)
                table = table.filter(mask)
            if drop_columns:
                table = table.drop(drop_columns)

        if len(table) > 0:
            yield table


# 16 MB read blocks with background prefetch for S3/remote reads.
_READ_BLOCK_SIZE = 16_000_000
_READ_CACHE_TYPE = "background"
_READ_MAX_BLOCKS = 2


@dataclass
class InputFileSpec:
    """Specification for reading a file or portion of a file.

    Pure read-spec: everything here is caller-supplied. Discovered metadata
    (e.g. file size from a bulk listing) lives on ``FileEntry`` instead.

    Attributes:
        path: Path to the file
        format: File format ("parquet", "jsonl", or "auto" to detect)
        columns: List of columns to read
        row_start: Optional start row for chunked reading
        row_end: Optional end row for chunked reading
        filter_expr: Optional filter expression to apply
    """

    path: str
    format: Literal["parquet", "jsonl", "vortex", "auto"] = "auto"
    columns: list[str] | None = None
    row_start: int | None = None
    row_end: int | None = None
    filter_expr: Expr | None = None


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

    # Use url_to_fs + fs.open so that block_size/cache_type reach the file
    # opener (AbstractBufferedFile) rather than the filesystem constructor.
    # fsspec.open() routes all **kwargs to the FS constructor, where S3's
    # AioSession rejects unknown kwargs like block_size.
    fs, resolved_path = url_to_fs(file_path)
    with fs.open(
        resolved_path,
        mode,
        block_size=_READ_BLOCK_SIZE,
        cache_type=_READ_CACHE_TYPE,
        cache_options={"maxblocks": _READ_MAX_BLOCKS},
        compression=compression,
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
        >>> # Load from cloud storage
        >>> ds = (Dataset
        ...     .from_files("gs://bucket/data", "**/*.jsonl.gz")
        ...     .load_jsonl()
        ...     .filter(lambda r: r["score"] > 0.5)
        ...     .write_jsonl("/output/filtered-{shard:05d}.jsonl.gz")
        ... )
        >>> output_files = ctx.execute(ds).results
        >>>
        >>> # Load from HuggingFace Hub (requires HF_TOKEN env var)
        >>> hf_url = "hf://datasets/username/dataset@main/data/train.jsonl.gz"
        >>> ds = Dataset.from_list([hf_url]).flat_map(load_jsonl)
        >>> records = ctx.execute(ds).results
    """
    spec = _as_spec(source)
    decoder = msgspec.json.Decoder()

    try:
        with open_file(spec.path, "rt") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        record = decoder.decode(line)
                    except Exception as e:
                        error_msg = str(e).lower()
                        if "truncated" in error_msg or "corrupt" in error_msg:
                            logger.warning(
                                "Skipping rest of %s: truncated/corrupt data at line %d (%s)",
                                spec.path,
                                line_num,
                                e,
                            )
                            return
                        logger.error(f"Error decoding line {line_num} in {spec.path}: {e}")
                        raise RuntimeError(f"Error decoding line {line_num} in {spec.path}: {e}") from e
                    counters.increment("zephyr/records_in")
                    yield record
    except RuntimeError:
        raise  # Re-raise wrapped error
    except Exception as e:
        error_msg = str(e).lower()
        if "truncated" in error_msg or "corrupt" in error_msg:
            logger.warning("Skipping corrupted file %s: %s", spec.path, e)
            return
        logger.error(f"Error reading JSONL file {spec.path}: {e}")
        raise RuntimeError(f"Error reading JSONL file {spec.path}: {e}") from e


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
        >>> ds = (Dataset
        ...     .from_files("/input", "**/*.parquet")
        ...     .load_parquet()
        ...     .map(lambda r: transform_record(r))
        ...     .write_jsonl("/output/data-{shard:05d}.jsonl.gz")
        ... )
        >>> output_files = ctx.execute(ds).results
    """
    spec = _as_spec(source)
    logger.info("Loading: %s", spec.path)

    pa_filter = None
    if spec.filter_expr is not None:
        from zephyr.expr import to_pyarrow_expr

        pa_filter = to_pyarrow_expr(spec.filter_expr)

    # Determine columns to read: include any filter-referenced columns
    # so post-hoc filtering works, then project down afterwards.
    read_columns = spec.columns
    need_project = False
    if spec.columns is not None and spec.filter_expr is not None:
        from zephyr.expr import referenced_columns

        filter_cols = referenced_columns(spec.filter_expr) - set(spec.columns)
        if filter_cols:
            read_columns = list(spec.columns) + sorted(filter_cols)
            need_project = True

    for table in iter_parquet_row_groups(
        spec.path,
        columns=read_columns,
        row_start=spec.row_start,
        row_end=spec.row_end,
    ):
        if pa_filter is not None:
            table = table.filter(pa_filter)
        if need_project:
            table = table.select(spec.columns)
        counters.increment("zephyr/records_in", len(table))
        yield from table.to_pylist()


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
        >>> ds = (Dataset
        ...     .from_files("/input/**/*.vortex")
        ...     .load_vortex()
        ...     .filter(lambda r: r["score"] > 0.5)
        ...     .write_jsonl("/output/filtered-{shard:05d}.jsonl.gz")
        ... )
        >>> output_files = ctx.execute(ds).results
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

    # Empty vortex files have no schema, so column projection would fail
    if dataset.count_rows() == 0:
        return

    if spec.row_start is not None and spec.row_end is not None:
        indices = np.arange(spec.row_start, spec.row_end, dtype=np.uint64)
        indices = pa.array(indices)
        table = dataset.take(indices, columns=columns, filter=pa_filter)
        counters.increment("zephyr/records_in", len(table))
        yield from table.to_pylist()
    else:
        table = dataset.to_table(columns=columns, filter=pa_filter)
        counters.increment("zephyr/records_in", len(table))
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
        >>> ds = (Dataset
        ...     .from_files("/input/**/*.jsonl")
        ...     .load_file()
        ...     .filter(lambda r: r["score"] > 0.5)
        ...     .write_jsonl("/output/data-{shard:05d}.jsonl.gz")
        ... )
        >>> output_files = ctx.execute(ds).results
    """
    spec = _as_spec(source)
    logger.info("Loading file: %s", spec.path)

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
        >>> ds = (Dataset
        ...     .from_list(["gs://bucket/data.zip"])
        ...     .flat_map(lambda p: load_zip_members(p, pattern="test.jsonl"))
        ...     .map(lambda m: process_file(m["filename"], m["content"]))
        ... )
        >>> output_files = ctx.execute(ds).results
    """
    spec = _as_spec(source)
    with open_url(spec.path, "rb") as f:
        with zipfile.ZipFile(f) as zf:
            for member_name in zf.namelist():
                if not member_name.endswith("/") and fnmatch.fnmatch(member_name, pattern):
                    with zf.open(member_name, "r") as member_file:
                        counters.increment("zephyr/records_in")
                        yield {
                            "filename": member_name,
                            "content": member_file.read(),
                        }
