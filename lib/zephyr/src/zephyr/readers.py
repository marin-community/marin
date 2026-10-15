# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Readers for common input formats.

Supports reading from local filesystems, cloud storage (gs://, s3://) and HuggingFace Hub (hf://) via fsspec.
"""

import fnmatch
import io
import logging
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from typing import BinaryIO, Literal

import fsspec
import msgspec
import numpy as np
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import vortex
import zstandard as zstd
from rigging.filesystem.factory import open_url, url_to_fs

from zephyr import counters
from zephyr.expr import Expr, referenced_columns, to_pyarrow_expr
from zephyr.input_file import DEFAULT_FILE_PATH_COLUMN, InputFileSpec
from zephyr.worker_context import input_progress

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared Parquet row-group reader
# ---------------------------------------------------------------------------


def iter_parquet_row_groups(
    source: str | pq.ParquetFile,
    *,
    columns: list[str] | None = None,
    row_start: int | None = None,
    row_end: int | None = None,
    row_groups: frozenset[int] | None = None,
) -> Iterator[pa.Table]:
    """Yield one ``pa.Table`` per qualifying row group with O(row_group) memory.

    Uses ``pq.ParquetFile`` instead of ``pyarrow.dataset`` to avoid the
    upstream memory leak (https://github.com/apache/arrow/issues/39808).

    Args:
        source: Path to parquet file or an already-open ``pq.ParquetFile``.
        columns: Columns to read (``None`` for all).
        row_start: First row to include (inclusive, before filtering).
        row_end: Last row to include (exclusive, before filtering).
        row_groups: Metadata-selected row group IDs (``None`` for all).
    """
    # Open string paths through open_file (rigging fsspec) rather than letting
    # pyarrow build a native S3FileSystem from the URI: pyarrow's default S3
    # client uses path-style addressing, which CoreWeave object storage rejects
    # with HTTP 400. open_file routes through the correctly-addressed fsspec fs
    # (same path the JSONL reader uses). The `with` stays open for the lifetime
    # of the delegated generator.
    if isinstance(source, str):
        with open_file(source, "rb") as f:
            yield from iter_parquet_row_groups(
                pq.ParquetFile(f), columns=columns, row_start=row_start, row_end=row_end, row_groups=row_groups
            )
        return

    pf = source
    has_row_range = row_start is not None or row_end is not None
    start = 0 if row_start is None else row_start
    end = pf.metadata.num_rows if row_end is None else row_end

    # Count only the selected row ranges/groups, using the already-loaded footer.
    # Advance after yield so a fused mapper consuming the group is included.
    selected_rows = 0
    group_start = 0
    for i in range(pf.metadata.num_row_groups):
        group_end = group_start + pf.metadata.row_group(i).num_rows
        if row_groups is None or i in row_groups:
            selected_rows += max(0, min(group_end, end) - max(group_start, start))
        group_start = group_end
    with input_progress(selected_rows) as progress:
        cumulative_rows = 0

        for i in range(pf.metadata.num_row_groups):
            rg_meta = pf.metadata.row_group(i)
            rg_num_rows = rg_meta.num_rows
            rg_start = cumulative_rows
            rg_end = cumulative_rows + rg_num_rows
            cumulative_rows = rg_end

            if has_row_range:
                if rg_end <= start:
                    continue
                if rg_start >= end:
                    return

            if row_groups is not None and i not in row_groups:
                continue

            table = pf.read_row_group(i, columns=columns)

            if has_row_range:
                is_interior = rg_start >= start and rg_end <= end
                if not is_interior:
                    local_start = max(0, start - rg_start)
                    local_end = min(rg_num_rows, end - rg_start)
                    table = table.slice(local_start, local_end - local_start)

            if len(table) > 0:
                yield table
                progress.completed += len(table)


# 16 MB read blocks with background prefetch for S3/remote reads.
_READ_BLOCK_SIZE = 16_000_000
_READ_CACHE_TYPE = "background"
_READ_MAX_BLOCKS = 2
# Small cache blocks keep footer reads and filtered payloads from fetching many
# unrelated row groups. Unfiltered scans retain the larger streaming blocks.
_PARQUET_READ_BLOCK_SIZE = 1_048_576
# Bound the index array and result materialized by each Vortex read.
_VORTEX_READ_BATCH_ROWS = 65_536


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
    return replace(spec, columns=reader_columns)


# Register HuggingFace filesystem with authentication if HF_TOKEN is available
# This enables reading from hf:// URLs throughout the codebase
try:
    from huggingface_hub import HfFileSystem

    fsspec.register_implementation("hf", HfFileSystem, clobber=True)
except ImportError:
    # HuggingFace Hub is optional - only needed for hf:// URLs
    pass


@contextmanager
def open_file(file_path: str, mode: str = "rb", *, block_size: int = _READ_BLOCK_SIZE):
    """Open ``file_path``, decompressing by extension, with caching defaults.

    ``.zst`` / ``.zstd`` decode via zstandard; ``.gz`` / ``.xz`` via fsspec's
    built-in codecs. ``mode`` may be binary (``rb``) or text (``rt``).
    """
    # Use url_to_fs + fs.open so that block_size/cache_type reach the file
    # opener (AbstractBufferedFile) rather than the filesystem constructor.
    # fsspec.open() routes all **kwargs to the FS constructor, where S3's
    # AioSession rejects unknown kwargs like block_size.
    fs, resolved_path = url_to_fs(file_path)
    read_kwargs = {
        "block_size": block_size,
        "cache_type": _READ_CACHE_TYPE,
        "cache_options": {"maxblocks": _READ_MAX_BLOCKS},
    }

    if file_path.endswith((".zst", ".zstd")):
        # Decompress with the zstandard package directly (as the writer and
        # shuffle/spill already do), NOT fsspec's compression="zstd": fsspec's
        # zstd codec is backed by stdlib compression.zstd (Python 3.14+) or
        # backports.zstd, neither of which zephyr requires, so routing through
        # it fails wherever those are absent. Open the raw bytes (keeping the
        # prefetch cache); read_across_frames decodes concatenated zstd frames;
        # text modes are wrapped so line-iterating callers get str, like "rt".
        with fs.open(resolved_path, "rb", **read_kwargs) as raw_f:
            reader = zstd.ZstdDecompressor().stream_reader(raw_f, read_across_frames=True)
            yield reader if "b" in mode else io.TextIOWrapper(reader, encoding="utf-8")
        return

    compression = None
    if file_path.endswith(".gz"):
        compression = "gzip"
    elif file_path.endswith(".xz"):
        compression = "xz"

    with fs.open(resolved_path, mode, compression=compression, **read_kwargs) as f:
        yield f


def _matching_row_groups(stream: BinaryIO, schema: pa.Schema, filter_expr: Expr) -> frozenset[int]:
    # Use the Dataset API only for footer statistics, never its scanner: payload
    # reads stay with ParquetFile to avoid the scanner memory leak above.
    fragment = pads.ParquetFileFormat().make_fragment(stream)
    return frozenset(
        group.id
        for part in fragment.split_by_row_group(filter=to_pyarrow_expr(filter_expr), schema=schema)
        for group in part.row_groups
    )


def compute_parquet_splits(
    path: str, approx_shard_bytes: int | None, *, filter_expr: Expr | None = None
) -> list[tuple[int, int]]:
    """Compute row-range split points from Parquet footer metadata.

    Reads only the file footer — no data is transferred. Splits are aligned to
    row-group boundaries, so actual shard sizes may exceed approx_shard_bytes when
    a single row group is larger than the target. Without a byte target, only
    excluded row groups separate spans.

    Args:
        path: Path to the Parquet file (local or remote via fsspec).
        approx_shard_bytes: Approximate target split size in uncompressed bytes.
            A row group will never be split, so individual shards may be larger.
            None disables byte-based splitting.
        filter_expr: Exclude groups whose footer statistics rule out this predicate.

    Returns:
        List of (row_start, row_end) tuples where row_end is exclusive.
    """
    # Read the footer through open_file (fsspec) so CoreWeave object storage's
    # virtual-host addressing is honored; a raw path makes pyarrow use its native
    # path-style S3 client, which CW rejects with HTTP 400.
    with open_file(path, "rb", block_size=_PARQUET_READ_BLOCK_SIZE) as f:
        parquet = pq.ParquetFile(f)
        metadata = parquet.metadata
        matching = _matching_row_groups(f, parquet.schema_arrow, filter_expr) if filter_expr is not None else None
    splits: list[tuple[int, int]] = []
    split_start = 0
    split_bytes = 0
    cumulative_rows = 0

    for i in range(metadata.num_row_groups):
        rg = metadata.row_group(i)
        rg_bytes = rg.total_byte_size

        if matching is not None and i not in matching:
            if split_bytes:
                splits.append((split_start, cumulative_rows))
                split_bytes = 0
            cumulative_rows += rg.num_rows
            split_start = cumulative_rows
            continue

        if approx_shard_bytes is not None and split_bytes > 0 and split_bytes + rg_bytes > approx_shard_bytes:
            splits.append((split_start, cumulative_rows))
            split_start = cumulative_rows
            split_bytes = 0

        split_bytes += rg_bytes
        cumulative_rows += rg.num_rows

    if split_bytes or matching is None:
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

    with input_progress(None) as progress:
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
                progress.completed += 1


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

    block_size = _PARQUET_READ_BLOCK_SIZE if spec.filter_expr is not None else _READ_BLOCK_SIZE
    with open_file(spec.path, "rb", block_size=block_size) as stream:
        parquet = pq.ParquetFile(stream)
        matching = (
            _matching_row_groups(stream, parquet.schema_arrow, spec.filter_expr)
            if spec.filter_expr is not None
            else None
        )
        for table in iter_parquet_row_groups(
            parquet,
            columns=read_columns,
            row_start=spec.row_start,
            row_end=spec.row_end,
            row_groups=matching,
        ):
            if pa_filter is not None:
                table = table.filter(pa_filter)
            if need_project:
                table = table.select(spec.columns)
            counters.pipeline.update_counter(counters.RECORDS_IN, len(table))
            yield from table.to_batches()


def load_parquet(source: str | InputFileSpec) -> Iterator[dict]:
    """Load Parquet file and yield records as dicts.

    When given an InputFileSpec with row_start or row_end, reads only the exact rows
    in that range. Row groups are read efficiently (only overlapping groups are loaded),
    then rows are filtered to the precise range. When filter_expr is provided, the filter
    excludes row groups using footer statistics before payload reads; remaining rows
    are filtered by PyArrow after reading.

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
    num_rows = dataset.count_rows()
    start = 0 if spec.row_start is None else spec.row_start
    end = num_rows if spec.row_end is None else min(spec.row_end, num_rows)
    # Filtered output batches do not expose how many source rows were scanned.
    with input_progress(max(0, end - start) if pa_filter is None else None) as progress:
        # Schema-less empty files cannot be projected by Vortex.
        if not dataset.schema.names and num_rows == 0:
            return

        # Vortex rejects empty projections. Read one column to retain row counts,
        # then drop it after applying the row range and filter.
        empty_projection = columns == []
        if empty_projection:
            columns = [dataset.schema.names[0]]

        if spec.row_start is not None or spec.row_end is not None:
            batches = (
                dataset.take(
                    pa.array(np.arange(batch_start, min(batch_start + _VORTEX_READ_BATCH_ROWS, end), dtype=np.uint64)),
                    columns=columns,
                    filter=pa_filter,
                )
                for batch_start in range(start, end, _VORTEX_READ_BATCH_ROWS)
            )
        else:
            batches = dataset.to_batches(columns=columns, filter=pa_filter, batch_size=_VORTEX_READ_BATCH_ROWS)

        for batch in batches:
            if empty_projection:
                batch = batch.select([])
            counters.pipeline.update_counter(counters.RECORDS_IN, len(batch))
            yield from batch.to_pylist()
            progress.completed += len(batch)


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


def _resolve_read_format(spec: InputFileSpec) -> Literal["parquet", "jsonl", "vortex"]:
    """Return the reader format for *spec*.

    An explicit ``format`` wins. ``Dataset.load_parquet`` and its siblings record
    the caller's choice on the spec, so a source whose name lacks the matching
    extension — Spark-style ``part-00000``, a ``.bin`` shard — still reads as the
    requested format instead of being rejected or handed to the wrong reader.
    ``auto`` infers the format from the path extension.

    Raises:
        ValueError: If ``format`` is ``auto`` and the extension is unsupported.
    """
    if spec.format != "auto":
        return spec.format
    if spec.path.endswith(".parquet"):
        return "parquet"
    if spec.path.endswith(".vortex"):
        return "vortex"
    if spec.path.endswith(SUPPORTED_EXTENSIONS):
        return "jsonl"
    raise ValueError(f"Unsupported extension: {spec.path}.")


def load_file(
    source: str | InputFileSpec,
    include_file_paths: bool = False,
    file_path_column: str = DEFAULT_FILE_PATH_COLUMN,
) -> Iterator[dict]:
    """Load records from file as JSONL, Parquet, or Vortex.

    The spec's ``format`` selects the reader; ``auto`` (the default for a bare
    path) infers it from the file extension.

    Args:
        source: Path to file or InputFileSpec containing the path, format, columns,
            row range, and filter expression.
        include_file_paths: If True, inject the source file path into each record
            under file_path_column.
        file_path_column: Key to add when include_file_paths is True.

    Yields:
        Parsed records as dictionaries

    Raises:
        ValueError: If the format is ``auto`` and the file extension is not supported.
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

    read_format = _resolve_read_format(spec)

    if include_file_paths and spec.columns is not None:
        spec = _strip_injected_file_path_column(spec, file_path_column)

    if read_format == "parquet":
        records = load_parquet(spec)
    elif read_format == "vortex":
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
    A spec with ``format="parquet"`` is honored whatever the extension.

    Args:
        source: Path to Parquet file or InputFileSpec containing the path, format,
            columns, row range, and filter expression.
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
    # Inlined rather than routed through _resolve_read_format so an unsupported
    # extension reports "not Parquet" instead of that helper's ValueError.
    is_parquet = spec.format == "parquet" or (spec.format == "auto" and spec.path.endswith(".parquet"))
    if not is_parquet:
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
