# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Readers for common input formats.

Supports reading from local filesystems, cloud storage (gs://, s3://) and HuggingFace Hub (hf://) via fsspec.
"""

import fnmatch
import logging
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Literal

import fsspec
import msgspec
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import vortex
from rigging.filesystem import open_url, url_to_fs

from zephyr import counters
from zephyr.expr import Expr, referenced_columns, to_pyarrow_expr

logger = logging.getLogger(__name__)

DEFAULT_FILE_PATH_COLUMN = "__file_path"


# ---------------------------------------------------------------------------
# Shared Parquet row-group reader
# ---------------------------------------------------------------------------


def iter_parquet_row_groups(
    source: str | pq.ParquetFile,
    *,
    columns: list[str] | None = None,
    row_start: int | None = None,
    row_end: int | None = None,
) -> Iterator[pa.Table]:
    """Yield one ``pa.Table`` per qualifying row group with O(row_group) memory.

    Uses ``pq.ParquetFile`` instead of ``pyarrow.dataset`` to avoid the
    upstream memory leak (https://github.com/apache/arrow/issues/39808).

    Args:
        source: Path to parquet file or an already-open ``pq.ParquetFile``.
        columns: Columns to read (``None`` for all).
        row_start: First row to include (inclusive, before filtering).
        row_end: Last row to include (exclusive, before filtering).
    """
    # Open string paths through open_file (rigging fsspec) rather than letting
    # pyarrow build a native S3FileSystem from the URI: pyarrow's default S3
    # client uses path-style addressing, which CoreWeave object storage rejects
    # with HTTP 400. open_file routes through the correctly-addressed fsspec fs
    # (same path the JSONL reader uses). The `with` stays open for the lifetime
    # of the delegated generator.
    if isinstance(source, str):
        with open_file(source, "rb") as f:
            yield from iter_parquet_row_groups(pq.ParquetFile(f), columns=columns, row_start=row_start, row_end=row_end)
        return

    pf = source
    has_row_range = row_start is not None and row_end is not None

    cumulative_rows = 0

    for i in range(pf.metadata.num_row_groups):
        rg_meta = pf.metadata.row_group(i)
        rg_num_rows = rg_meta.num_rows
        rg_start = cumulative_rows
        rg_end = cumulative_rows + rg_num_rows
        cumulative_rows = rg_end

        if has_row_range:
            assert row_start is not None and row_end is not None
            if rg_end <= row_start:
                continue
            if rg_start >= row_end:
                return

        table = pf.read_row_group(i, columns=columns)

        if has_row_range:
            assert row_start is not None and row_end is not None
            is_interior = rg_start >= row_start and rg_end <= row_end
            if not is_interior:
                local_start = max(0, row_start - rg_start)
                local_end = min(rg_num_rows, row_end - rg_start)
                table = table.slice(local_start, local_end - local_start)

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


def _strip_injected_file_path_column(spec: InputFileSpec, file_path_column: str) -> InputFileSpec:
    """Drop the injected file-path column from a reader projection.

    When a caller projects ``columns`` and also requests ``include_file_paths``,
    the path column must appear in the projection (it is appended after the
    read) but must not be passed to the underlying reader (it is not in the
    file). Returns a spec with that column stripped from ``columns``.

    Raises:
        RuntimeError: If ``file_path_column`` is missing from the projection.
    """
    assert spec.columns is not None
    if file_path_column not in spec.columns:
        raise RuntimeError(f"Column filter must include file path column '{file_path_column}'.")
    reader_columns = [c for c in spec.columns if c != file_path_column]
    return replace(spec, columns=reader_columns or None)


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


def compute_parquet_splits(path: str, approx_shard_bytes: int) -> list[tuple[int, int]]:
    """Compute row-range split points from Parquet footer metadata.

    Reads only the file footer — no data is transferred. Splits are aligned to
    row-group boundaries, so actual shard sizes may exceed approx_shard_bytes when
    a single row group is larger than the target. Files whose total compressed size
    is below approx_shard_bytes return a single span.

    Args:
        path: Path to the Parquet file (local or remote via fsspec).
        approx_shard_bytes: Approximate target split size in bytes. Best-effort:
            a row group will never be split, so individual shards may be larger.

    Returns:
        List of (row_start, row_end) tuples where row_end is exclusive.
    """
    # Read the footer through open_file (fsspec) so CoreWeave object storage's
    # virtual-host addressing is honored; a raw path makes pyarrow use its native
    # path-style S3 client, which CW rejects with HTTP 400.
    with open_file(path, "rb") as f:
        metadata = pq.ParquetFile(f).metadata
    splits: list[tuple[int, int]] = []
    split_start = 0
    split_bytes = 0
    cumulative_rows = 0

    for i in range(metadata.num_row_groups):
        rg = metadata.row_group(i)
        rg_bytes = rg.total_byte_size

        if split_bytes > 0 and split_bytes + rg_bytes > approx_shard_bytes:
            splits.append((split_start, cumulative_rows))
            split_start = cumulative_rows
            split_bytes = 0

        split_bytes += rg_bytes
        cumulative_rows += rg.num_rows

    splits.append((split_start, cumulative_rows))
    return splits


def load_jsonl(source: str | InputFileSpec) -> Iterator[dict]:
    """Load a JSONL file and yield parsed records as dictionaries.

    If the input file is compressed (.gz, .zst, .xz), it will be automatically
    decompressed during loading. When given an InputFileSpec, ``filter_expr``
    and ``columns`` are honored at read time (mirroring ``load_parquet`` /
    ``load_vortex``); ``row_start`` / ``row_end`` are ignored because JSONL
    has no random-access index.

    Args:
        source: Path to JSONL file or InputFileSpec containing the path,
            optional filter expression, and optional column projection.
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
    filter_fn = spec.filter_expr.evaluate if spec.filter_expr is not None else None
    columns = spec.columns

    with open_file(spec.path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = decoder.decode(line)
            if filter_fn is not None and not filter_fn(record):
                continue
            if columns is not None:
                record = {k: record[k] for k in columns if k in record}
            counters.pipeline.update_counter(counters.RECORDS_IN, 1)
            yield record


def load_parquet_batch(source: str | InputFileSpec) -> Iterator[pa.RecordBatch]:
    """Load a Parquet file and yield one ``pa.RecordBatch`` per row group.

    Applies the same column projection, row-range slicing, and filter pushdown
    as ``load_parquet``, but returns Arrow batches rather than Python dicts so
    callers can stay in the columnar world.

    Args:
        source: Path to Parquet file or InputFileSpec containing the path, columns,
            row range, and filter expression.

    Yields:
        One ``pa.RecordBatch`` per qualifying row group.
    """
    spec = _as_spec(source)
    logger.info("Loading: %s", spec.path)

    pa_filter = None
    if spec.filter_expr is not None:
        pa_filter = to_pyarrow_expr(spec.filter_expr)

    # Determine columns to read: include any filter-referenced columns
    # so post-hoc filtering works, then project down afterwards.
    read_columns = spec.columns
    need_project = False
    if spec.columns is not None and spec.filter_expr is not None:
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
        counters.pipeline.update_counter(counters.RECORDS_IN, len(table))
        yield from table.to_batches()


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
    for batch in load_parquet_batch(source):
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
        >>> ds = (Dataset
        ...     .from_files("/input/**/*.vortex")
        ...     .load_vortex()
        ...     .filter(lambda r: r["score"] > 0.5)
        ...     .write_jsonl("/output/filtered-{shard:05d}.jsonl.gz")
        ... )
        >>> output_files = ctx.execute(ds).results
    """
    spec = _as_spec(source)
    columns = spec.columns

    # Convert filter to PyArrow expression if provided
    pa_filter = None
    if spec.filter_expr is not None:
        pa_filter = to_pyarrow_expr(spec.filter_expr)

    # Open vortex file and get PyArrow Dataset interface
    logger.info("Loading: %s", spec.path)
    vf = vortex.open(spec.path)
    dataset = vf.to_dataset()

    # Empty vortex files have no schema, so column projection would fail
    if dataset.count_rows() == 0:
        return

    if spec.row_start is not None and spec.row_end is not None:
        indices = pa.array(np.arange(spec.row_start, spec.row_end, dtype=np.uint64))
        table = dataset.take(indices, columns=columns, filter=pa_filter)
    else:
        table = dataset.to_table(columns=columns, filter=pa_filter)

    counters.pipeline.update_counter(counters.RECORDS_IN, len(table))
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


def load_file(
    source: str | InputFileSpec,
    include_file_paths: bool = False,
    file_path_column: str = DEFAULT_FILE_PATH_COLUMN,
) -> Iterator[dict]:
    """Load records from file, auto-detecting JSONL, Parquet, or Vortex format.

    Args:
        source: Path to file or InputFileSpec containing the path, columns,
            row range, and filter expression.
        include_file_paths: If True, inject the source file path into each record
            under file_path_column.
        file_path_column: Key to add when include_file_paths is True.

    Yields:
        Parsed records as dictionaries

    Raises:
        ValueError: If file extension is not supported
        RuntimeError: If file_path_column already exists in a record.

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

    if include_file_paths and spec.columns is not None:
        spec = _strip_injected_file_path_column(spec, file_path_column)

    if spec.path.endswith(".parquet"):
        records = load_parquet(spec)
    elif spec.path.endswith(".vortex"):
        records = load_vortex(spec)
    else:
        records = load_jsonl(spec)

    if not include_file_paths:
        yield from records
        return

    for record in records:
        if file_path_column in record:
            raise RuntimeError(f"Cannot add file path column '{file_path_column}': key already exists in record")
        record[file_path_column] = spec.path
        yield record


def load_file_batch(
    source: str | InputFileSpec,
    include_file_paths: bool = False,
    file_path_column: str = DEFAULT_FILE_PATH_COLUMN,
) -> Iterator[pa.RecordBatch]:
    """Load a Parquet file and yield ``pa.RecordBatch`` objects.

    Only Parquet files are supported. Raises ``RuntimeError`` for any other
    file type so callers get a clear error rather than silent dict conversion.

    Args:
        source: Path to Parquet file or InputFileSpec containing the path, columns,
            row range, and filter expression.
        include_file_paths: If True, append a string column named file_path_column
            containing the source file path to each batch.
        file_path_column: Name of the column to add when include_file_paths is True.

    Yields:
        One ``pa.RecordBatch`` per qualifying row group.

    Raises:
        RuntimeError: If the file is not a Parquet file, or if file_path_column
            already exists in the batch schema.
    """
    spec = _as_spec(source)
    if not spec.path.endswith(".parquet"):
        raise RuntimeError(f"load_file_batch only supports Parquet files, got: {spec.path}")
    if include_file_paths and spec.columns is not None:
        spec = _strip_injected_file_path_column(spec, file_path_column)
    for batch in load_parquet_batch(spec):
        if include_file_paths:
            if file_path_column in batch.schema.names:
                raise RuntimeError(
                    f"Cannot add file path column '{file_path_column}': column already exists in batch schema"
                )
            batch = batch.append_column(
                file_path_column,
                pa.array([spec.path] * len(batch), type=pa.string()),
            )
        yield batch


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
                        counters.pipeline.update_counter(counters.RECORDS_IN, 1)
                        yield {
                            "filename": member_name,
                            "content": member_file.read(),
                        }
