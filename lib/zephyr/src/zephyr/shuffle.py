# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scatter/shuffle for Zephyr group-by.

Mappers hash each row's key to a target shard, buffer DataFrames in memory,
and flush sorted Parquet chunks (plus a small msgpack sidecar). Reducers
scan those files with a shard predicate, merge-sort the matching rows, and
group by key. Accepts Python items or columnar batches (``pl.DataFrame`` /
``pa.RecordBatch``).
"""

import concurrent.futures
import functools
import gc
import io
import logging
import math
import os
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any, overload
from urllib.parse import urlparse

import cloudpickle
import humanfriendly
import msgspec
import polars as pl
import pyarrow as pa
from iris.env_resources import TaskResources
from rigging.filesystem import StoragePath, open_url, url_to_fs
from rigging.filesystem.s3_compat import needs_virtual_host_addressing
from rigging.timing import RateLimiter, log_time

from zephyr.expr import ColumnExpr
from zephyr.external_sort import external_sort_merge
from zephyr.shard_keys import encode_key
from zephyr.worker_context import _worker_ctx_var
from zephyr.writers import ensure_parent_dir

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core shard data types
# ---------------------------------------------------------------------------


@dataclass
class MemChunk:
    """In-memory chunk."""

    items: list[Any]

    def __iter__(self) -> Iterator:
        return iter(self.items)


@dataclass
class ListShard:
    """Shard backed by a list of iterable references (PickleDiskChunk, MemChunk, etc.)."""

    refs: list[Iterable]

    def __iter__(self) -> Iterator:
        for ref in self.refs:
            yield from ref


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SCATTER_METADATA_FILENAME = "metadata.msgpack"

# Number of parallel sidecar reads each reducer issues when building its
# ScatterReader. Sidecars are small msgpack files (a few KB) and reads are
# GCS GET-bound, so a modest pool keeps latency low without thrashing.
_SIDECAR_READ_CONCURRENCY = 32
# Fraction of available memory available for merging.
_SCATTER_READ_MEMORY_FRACTION = 0.4

# Memory overhead multiple per row in Polars DataFrame.
_SCATTER_READ_POLARS_ROW_OVERHEAD = 2
# Memory overhead multiple per row in the Python iterator when the reducer is not Polars-based.
_SCATTER_READ_PYTHON_ROW_OVERHEAD = 2

_PROGRESS_LOG_INTERVAL_SECONDS = 60.0
# Polars streaming chunk size, important to avoid excessive memory usage during merge.
_POLARS_STREAMING_CHUNK_SIZE = 10000
# Maximum run files that Polars merges at one time during an external sort.
_EXTERNAL_SORT_MAX_MERGE_FAN_IN = 32
# Seed for the vectorized shard-routing hash used by ScatterWriter.write_batch.
_SCATTER_HASH_SEED = 0

# Helper column names injected by _items_to_dataframe and stripped before
# writing to disk.  Both are internal implementation details; user schemas must
# not collide with these names.
_SHARD_COL = "__zephyr_shard__"
_SORT_KEY_COL = "__zephyr_sort_key__"
# A cloudpickle-serialized Python object representing the item
_PAYLOAD_COL = "__payload__"
# Temporary flat columns folded into the _SORT_KEY_COL struct during
# _items_to_dataframe; never present in written chunks.
_KEY_TMP_COL = "__zephyr_key_tmp__"
_SORT_VALUE_TMP_COL = "__zephyr_sort_value_tmp__"

# Python items consumed before creating a DataFrame.
_DATAFRAME_ROW_COUNT = 1000
# Flush all scatter buffers when the buffer's estimated size exceeds this fraction of task memory.
_SCATTER_FLUSH_THRESHOLD = 0.20
# Empirically measured ratio of DataFrame.estimated_size() to actual per-shard process RSS growth,
# across three datasets (nemotron: 0.54-0.60, skewed: 0.59-0.66, FineWeb-Edu: 0.66-0.70).
# Applied to _SCATTER_FLUSH_THRESHOLD so the effective trigger is 12% of task memory. The 2.27x
# flush peak was measured during the 2026-08-02 fuzzy-dedup incident: https://echo.oa.dev/wiki/68.
_ESTIMATED_SIZE_CORRECTION_FACTOR = 0.60
# Threshold for triggering a gc.collect() after a flush.
_GC_FLUSH_SIZE_THRESHOLD_BYTES = 8 * 1024 * 1024


def _task_memory_bytes() -> int:
    ctx = _worker_ctx_var.get()
    if ctx is not None and ctx.task_memory_bytes > 0:
        return ctx.task_memory_bytes

    memory_bytes = TaskResources.from_environment().memory_bytes
    if memory_bytes <= 0:
        logger.warning("No task memory is available. Using a 1 GiB memory budget.")
        return 1024 * 1024 * 1024
    return memory_bytes


def _dataframe_to_items(df: pl.DataFrame) -> Iterator[Any]:
    """Yield Python items from a DataFrame, stripping routing columns."""
    if _PAYLOAD_COL in df.columns:
        for p in df[_PAYLOAD_COL].to_list():
            yield cloudpickle.loads(p)
    else:
        drop_cols = [c for c in (_SHARD_COL, _SORT_KEY_COL) if c in df.columns]
        yield from df.drop(drop_cols).to_dicts()


def _with_routing_columns(
    df: pl.DataFrame,
    key_expr: pl.Expr,
    sort_value_expr: pl.Expr,
    num_output_shards: int,
) -> pl.DataFrame:
    """Append ``_SHARD_COL`` / ``_SORT_KEY_COL`` from a key and sort-value expression."""
    return df.with_columns(
        (key_expr.hash(seed=_SCATTER_HASH_SEED) % num_output_shards).cast(pl.Int32).alias(_SHARD_COL),
        pl.struct(key_expr.alias("key"), sort_value_expr.alias("sort_value")).alias(_SORT_KEY_COL),
    )


def _column_routing_exprs(key: ColumnExpr, sort_by: ColumnExpr | None) -> tuple[pl.Expr, pl.Expr]:
    """Polars expressions for ``col(...)`` key/sort columns already present on a frame."""
    return pl.col(key.name), pl.col(sort_by.name) if sort_by is not None else pl.lit(None)


def _columns_to_dataframe(
    payloads: list[bytes],
    key_bytes: list[bytes],
    sort_values: list[Any],
    num_output_shards: int,
) -> pl.DataFrame:
    """Build the scatter DataFrame from pre-computed flat columns.

    The sort-key struct is folded from two flat columns rather than per-row
    Python dicts: series construction from homogeneous lists is the native
    fast path, and ``pl.struct`` over existing columns is cheap. Field order
    (key first) drives the (key, sort_value) sort order.

    ``key_bytes`` must be pre-encoded via :func:`~zephyr.shard_keys.encode_key` so
    that ``_KEY_TMP_COL`` is always ``Binary`` — preventing sort-key schema
    mismatches when different mapper shards produce keys of different Python
    types. Shard routing hashes those bytes with Polars (see
    :func:`_with_routing_columns`).
    """
    try:
        df = pl.DataFrame(
            {
                _PAYLOAD_COL: pl.Series(payloads, dtype=pl.Binary),
                _KEY_TMP_COL: pl.Series(key_bytes, dtype=pl.Binary),
                _SORT_VALUE_TMP_COL: sort_values,
            }
        )
        # Non-serializable sort_values surface as TypeError from Series
        # construction or InvalidOperationError ("nested objects are not
        # allowed") when a column lands as Object dtype and pl.struct rejects it.
        return _with_routing_columns(
            df,
            pl.col(_KEY_TMP_COL),
            pl.col(_SORT_VALUE_TMP_COL),
            num_output_shards,
        ).select(_PAYLOAD_COL, _SHARD_COL, _SORT_KEY_COL)
    except (TypeError, pl.exceptions.InvalidOperationError) as err:
        raise ValueError("sort_fn must return an Arrow-serializable object.") from err


def _items_to_dataframe(
    items: list[Any],
    key_fn: Callable,
    sort_fn: Callable | None,
    num_output_shards: int,
) -> pl.DataFrame:
    """Convert a list of Python items to a DataFrame with routing columns.

    Cloudpickle-serializes items into ``_PAYLOAD_COL`` and adds ``_SHARD_COL``
    (int32 target shard index) and ``_SORT_KEY_COL``. This is the adapter
    between Python-item pipelines and the DataFrame-based
    :class:`ScatterWriter`; DataFrame-native pipelines can feed the writer
    directly. Keys are msgpack-encoded to Binary before routing so every
    mapper writes the same sort-key dtype; shard indices then use the same
    Polars hash as :meth:`ScatterWriter.write_batch`.
    """
    key_bytes: list[bytes] = []
    sort_values: list[Any] = []
    for item in items:
        key = key_fn(item)
        try:
            key_bytes.append(encode_key(key))
        except TypeError as err:
            raise ValueError(f"key_fn must return a msgpack-serializable object; got {type(key).__name__!r}.") from err
        sort_values.append(sort_fn(item) if sort_fn is not None else None)
    payloads = [cloudpickle.dumps(item) for item in items]
    return _columns_to_dataframe(payloads, key_bytes, sort_values, num_output_shards)


# ---------------------------------------------------------------------------
# Sidecar / manifest helpers
# ---------------------------------------------------------------------------


@functools.cache
def _sidecar_encoder() -> msgspec.msgpack.Encoder:
    return msgspec.msgpack.Encoder()


@functools.cache
def _sidecar_decoder() -> msgspec.msgpack.Decoder:
    return msgspec.msgpack.Decoder()


def _scatter_meta_path(data_path: str) -> str:
    return f"{data_path}{_SCATTER_METADATA_FILENAME}"


def _write_scatter_meta(data_path: str, sidecar: dict) -> None:
    meta_path = _scatter_meta_path(data_path)
    payload = _sidecar_encoder().encode(sidecar)
    with log_time(f"Writing scatter meta for {data_path} to {meta_path}", level=logging.DEBUG):
        StoragePath(meta_path).write_bytes(payload)


@dataclass(frozen=True)
class _SidecarSlice:
    """Chunk paths and metadata from one mapper's sidecar.

    Each entry in ``chunk_paths`` is one combined Parquet file written during a
    flush; the file contains data for all target shards sorted by
    ``(_SHARD_COL, _SORT_KEY_COL)``. ``target_bytes`` is the exact payload
    bytes this mapper wrote for the reader's target shard, summed across all
    chunks.
    """

    path: str
    chunk_paths: list[str]  # GCS parquet paths, one per flush event
    avg_item_bytes: float
    target_bytes: int


def _read_sidecar_slice(path: str, target_shard: int) -> _SidecarSlice | None:
    """Read one sidecar and return its file list plus the target shard's payload bytes.

    Returns ``None`` if the sidecar has no files (empty writer).

    Uses ``fs.cat_file`` rather than ``open_url`` — one direct GET returning
    bytes is ~25% faster than going through ``TextIOWrapper(BufferedFile)``
    for small sidecars, and msgpack decodes bytes directly.
    """
    meta_path = _scatter_meta_path(path)
    fs, fs_path = url_to_fs(meta_path)
    meta = _sidecar_decoder().decode(fs.cat_file(fs_path))
    files = meta.get("files", [])
    if not files:
        return None
    return _SidecarSlice(
        path=path,
        chunk_paths=[str(f) for f in files],
        avg_item_bytes=float(meta.get("avg_item_bytes", 0)),
        target_bytes=int(meta.get("shard_bytes", {}).get(str(target_shard), 0)),
    )


def _read_sidecar_slices_parallel(scatter_paths: list[str], target_shard: int) -> list[_SidecarSlice]:
    """Read every sidecar concurrently and return slices in input order.

    Empty sidecars (no files written) are dropped from the result.
    """
    ordered: list[_SidecarSlice | None] = [None] * len(scatter_paths)
    with concurrent.futures.ThreadPoolExecutor(max_workers=_SIDECAR_READ_CONCURRENCY) as pool:
        futures = {pool.submit(_read_sidecar_slice, p, target_shard): i for i, p in enumerate(scatter_paths)}
        for fut in concurrent.futures.as_completed(futures):
            idx = futures[fut]
            ordered[idx] = fut.result()
    return [s for s in ordered if s is not None]


def _unify_frame_schemas(frames: list[pl.LazyFrame]) -> list[pl.LazyFrame]:
    """Cast frames to a common supertype schema so pl.merge_sorted doesn't fail.

    Different source shards may write the same column with different dtypes when
    Polars infers from Python values — most commonly a sort_value field that is
    Null on an all-None batch from one shard and Int64 from another.  This also
    handles arbitrary user-column dtype drift when DataFrames are written directly.

    collect_schema() reads only parquet file-footer metadata (no row data).
    The limit(0) concat derives the supertype schema without any I/O.  Casting
    is applied as a lazy expression, so no data is scanned here.
    """
    if len(frames) <= 1:
        return frames
    schemas = [f.collect_schema() for f in frames]
    if all(s == schemas[0] for s in schemas[1:]):
        return frames
    unified = pl.concat([f.limit(0) for f in frames], how="diagonal_relaxed").collect_schema()
    return [f.cast(dict(unified)) for f in frames]


def _scan_scatter_parquet(path: str) -> pl.LazyFrame:
    """Scan a scatter chunk with the addressing required by CoreWeave object storage."""
    endpoint = os.environ.get("AWS_ENDPOINT_URL_S3") or os.environ.get("AWS_ENDPOINT_URL")
    if not path.startswith("s3://") or not endpoint or not needs_virtual_host_addressing(endpoint):
        return pl.scan_parquet(path)

    bucket = urlparse(path).netloc
    parsed_endpoint = urlparse(endpoint)
    hostname = parsed_endpoint.hostname or ""
    if not hostname.startswith(f"{bucket}."):
        endpoint = parsed_endpoint._replace(netloc=f"{bucket}.{parsed_endpoint.netloc}").geturl()

    # Polars' Rust object_store client reads credentials and region from the
    # environment, but unlike fsspec it cannot infer CoreWeave's virtual-host
    # requirement. In virtual-host mode object_store uses the endpoint verbatim,
    # so the bucket must be part of the endpoint host.
    return pl.scan_parquet(
        path,
        storage_options={
            "aws_endpoint_url": endpoint,
            "aws_virtual_hosted_style_request": "true",
        },
    )


# ---------------------------------------------------------------------------
# ScatterReader: built from manifest, fed to Reduce
# ---------------------------------------------------------------------------


class ScatterReader:
    """All scatter chunks for one target shard, across all source files.

    ``_files`` is a list of ``(source_path, chunk_paths)`` pairs — one entry
    per source shard — where ``chunk_paths`` is the list of GCS parquet file
    paths that source shard wrote for ``_target_shard``.

    Construct via :meth:`from_sidecars` for production use, or pass fields
    directly for testing.
    """

    def __init__(
        self,
        files: list[tuple[str, list[str]]],
        target_shard: int,
        avg_item_bytes: float,
        shard_payload_bytes: float = 0.0,
    ) -> None:
        self._files = files
        self._target_shard = target_shard
        self.avg_item_bytes = avg_item_bytes
        self.shard_payload_bytes = shard_payload_bytes

    @classmethod
    def from_sidecars(cls, scatter_paths: list[str], target_shard: int) -> "ScatterReader":
        """Build a ScatterReader by reading per-mapper sidecars directly.

        Each reducer reads every mapper's ``.scatter_meta`` sidecar in parallel
        and filters for its own ``target_shard``. No coordinator-written manifest
        is needed, which eliminates a serialization bottleneck when there are
        thousands of mappers.
        """
        files: list[tuple[str, list[str]]] = []
        weighted_bytes = 0.0
        total_chunks = 0
        shard_payload_bytes = 0.0

        with log_time(
            f"Building ScatterReader for target shard {target_shard} "
            f"from {len(scatter_paths)} sidecars (concurrency={_SIDECAR_READ_CONCURRENCY})"
        ):
            for slice_ in _read_sidecar_slices_parallel(scatter_paths, target_shard):
                files.append((slice_.path, slice_.chunk_paths))
                weighted_bytes += slice_.avg_item_bytes * len(slice_.chunk_paths)
                total_chunks += len(slice_.chunk_paths)
                shard_payload_bytes += slice_.target_bytes

        avg_item_bytes = weighted_bytes / total_chunks if total_chunks > 0 else 0.0

        logger.info(
            "ScatterReader for shard %d: %d source files, %d total chunks, "
            "avg_item_bytes=%.1f, shard_payload_bytes=%.0f",
            target_shard,
            len(files),
            total_chunks,
            avg_item_bytes,
            shard_payload_bytes,
        )
        return cls(
            files=files,
            target_shard=target_shard,
            avg_item_bytes=avg_item_bytes,
            shard_payload_bytes=shard_payload_bytes,
        )

    def get_frames(self) -> list[pl.LazyFrame]:
        frames = [
            _scan_scatter_parquet(path).filter(pl.col(_SHARD_COL) == self._target_shard).drop(_SHARD_COL)
            for _, chunk_paths in self._files
            for path in chunk_paths
        ]
        return _unify_frame_schemas(frames)

    @property
    def total_chunks(self) -> int:
        return sum(len(chunks) for _, chunks in self._files)

    def merge_sorted_chunks(self, external_sort_dir: str) -> Iterator[Any]:
        """Merge sorted chunks using k-way merge, yielding items in global sort order.

        Each chunk file is assumed to be sorted by ``_SORT_KEY_COL`` (key plus optional
        secondary sort). Performs a k-way merge across all chunks.
        Args:
            external_sort_dir: If set and the shard exceeds the memory budget,
                spill intermediate runs.

        Yields:
            Deserialized Python items in merged sort order.
        """

        with pl.Config() as polars_config:
            polars_config.set_streaming_chunk_size(_POLARS_STREAMING_CHUNK_SIZE)

            if self.total_chunks == 0:
                return

            # Upper bound on merge memory: the target shard's entire data
            # resident at once (the streaming merge holds strictly less in
            # flight), using the exact per-shard payload bytes recorded in the
            # sidecars — no row-count estimation.
            estimated_merge_memory_bytes = self.shard_payload_bytes
            # Overhead per row in the Polars DataFrame plus the deserialized Python object.
            # Future Polars-only processing would remove the Python overhead.
            overhead = _SCATTER_READ_POLARS_ROW_OVERHEAD * _SCATTER_READ_PYTHON_ROW_OVERHEAD
            memory_bytes = _task_memory_bytes()

            if estimated_merge_memory_bytes * overhead > memory_bytes * _SCATTER_READ_MEMORY_FRACTION:
                fan_in = math.ceil(math.sqrt(self.total_chunks))

                logger.info(
                    "[shard %d] Merging %d chunks via external sort "
                    "(%s memory needed > %s memory available); fan_in=%d",
                    self._target_shard,
                    self.total_chunks,
                    humanfriendly.format_size(estimated_merge_memory_bytes * overhead, binary=True),
                    humanfriendly.format_size(memory_bytes * _SCATTER_READ_MEMORY_FRACTION, binary=True),
                    fan_in,
                )

                batches = external_sort_merge(
                    input_frames=self.get_frames(),
                    sort_key=_SORT_KEY_COL,
                    external_sort_dir=external_sort_dir,
                    fan_in=fan_in,
                    max_merge_fan_in=_EXTERNAL_SORT_MAX_MERGE_FAN_IN,
                    shard=self._target_shard,
                )

            else:
                logger.info(
                    "[shard %d] Merging %d chunks in memory (%s memory needed < %s memory available)",
                    self._target_shard,
                    self.total_chunks,
                    humanfriendly.format_size(estimated_merge_memory_bytes * overhead, binary=True),
                    humanfriendly.format_size(memory_bytes * _SCATTER_READ_MEMORY_FRACTION, binary=True),
                )
                batches = pl.merge_sorted(self.get_frames(), key=_SORT_KEY_COL).collect_batches()

            for batch in batches:
                yield from _dataframe_to_items(batch)


# ---------------------------------------------------------------------------
# Scatter writer
# ---------------------------------------------------------------------------


def _apply_combiner(buffer: list, key_fn: Callable, combiner_fn: Callable) -> list:
    """Group buffer by key and reduce locally."""
    by_key: dict[object, list] = defaultdict(list)
    with log_time(f"Applying combiner to buffer of size {len(buffer)}", level=logging.DEBUG):
        for item in buffer:
            by_key[key_fn(item)].append(item)
        combined: list = []
        for key, items in by_key.items():
            combined.extend(combiner_fn(key, iter(items)))
    return combined


def _combine_scatter_buffer(
    buffer: pl.DataFrame,
    key: Callable | ColumnExpr,
    sort_by: Callable | ColumnExpr | None,
    combiner_fn: Callable,
) -> pl.DataFrame | None:
    """Run *combiner_fn* per shard and rebuild routing columns. Empty → ``None``."""
    frames: list[pl.DataFrame] = []
    key_fn = key.evaluate if isinstance(key, ColumnExpr) else key
    for (shard_val,), group in buffer.partition_by(_SHARD_COL, as_dict=True).items():
        rows = list(_dataframe_to_items(group))
        rows = _apply_combiner(rows, key_fn, combiner_fn)
        if not rows:
            continue
        # Shard is already assigned; rebuild sort keys then restore it.
        if _PAYLOAD_COL in group.columns:
            df = _items_to_dataframe(rows, key, sort_by, num_output_shards=1)
        else:
            assert isinstance(key, ColumnExpr)
            assert sort_by is None or isinstance(sort_by, ColumnExpr)
            key_expr, sort_value_expr = _column_routing_exprs(key, sort_by)
            df = _with_routing_columns(pl.DataFrame(rows), key_expr, sort_value_expr, num_output_shards=1)
        frames.append(df.with_columns(pl.lit(shard_val, dtype=pl.Int32).alias(_SHARD_COL)))
    if not frames:
        return None
    return pl.concat(frames, how="vertical_relaxed", rechunk=True)


class ScatterWriter:
    """Writes scatter chunk files as zstd-compressed Parquet, one combined file per flush.

    Accepts routing-column DataFrames via :meth:`write` (see
    ``_items_to_dataframe`` for the Python-items adapter) as well as raw
    batch DataFrames via :meth:`write_batch`, which computes the routing
    columns directly in Polars and ingests the batch's own columns with no
    Python-object round trip. Frames are buffered as a list — appends are
    free — and combined with one concat per flush.

    Each flush writes a single ``c{chunk:04d}.parquet`` file sorted by
    ``[_SHARD_COL, _SORT_KEY_COL]`` with row groups sized so Polars predicate
    pushdown skips non-target row groups on the read side.

    Flushing is estimated-size-based: when the sum of ``DataFrame.estimated_size()``
    across buffered frames exceeds ``_SCATTER_FLUSH_THRESHOLD * _ESTIMATED_SIZE_CORRECTION_FACTOR``
    of available task memory, all buffered frames are flushed together into one combined file.
    """

    def __init__(
        self,
        data_path: str,
        key: Callable | ColumnExpr,
        source_shard: int,
        num_output_shards: int,
        sort_by: Callable | ColumnExpr | None = None,
        combiner_fn: Callable | None = None,
    ) -> None:
        self._data_path = data_path if data_path.endswith("/") else f"{data_path}/"
        self._key = key
        self._sort_by = sort_by
        self._num_output_shards = num_output_shards

        self._source_shard = source_shard
        self._combiner_fn = combiner_fn
        self._memory_available_bytes = _task_memory_bytes()
        # estimated_size() measures only the buffer columns, not process overhead (Python runtime,
        # Arrow allocator, Parquet scan). The correction factor maps estimated_size to estimated RSS.
        self._flush_threshold_bytes = int(
            self._memory_available_bytes * _SCATTER_FLUSH_THRESHOLD * _ESTIMATED_SIZE_CORRECTION_FACTOR
        )

        # Buffered DataFrames, combined into one file per flush. Buffering
        # frames (not Python items) keeps the writer format-agnostic: a future
        # RecordBatch/DataFrame-native pipeline can feed frames directly.
        self._frames: list[pl.DataFrame] = []
        self._chunk_paths: list[str] = []
        # Payload bytes written per target shard, recorded in the sidecar so
        # reducers know their shard's exact data size for the external-sort
        # decision without opening any chunk files.
        self._shard_bytes: defaultdict[int, int] = defaultdict(int)
        self._avg_item_bytes: float = 0.0
        self._total_bytes_written: int = 0
        self._total_rows_written: int = 0
        self._n_chunks_written = 0
        # Throttles the per-flush progress log so high-fanout workloads don't log too often
        self._progress_log_limiter = RateLimiter(interval_seconds=_PROGRESS_LOG_INTERVAL_SECONDS)
        # Running estimated_size() total of unflushed frames; reset to 0 on flush.
        self._buffer_estimated_bytes: int = 0

        ensure_parent_dir(self._data_path)
        self._result: ListShard | None = None

    def _flush(self) -> None:
        """Flush the accumulated buffer into one combined Parquet file sorted by [_SHARD_COL, _SORT_KEY_COL]."""
        if not self._frames:
            return

        buffer = pl.concat(self._frames, how="vertical_relaxed", rechunk=False)
        self._frames = []
        self._buffer_estimated_bytes = 0

        if self._combiner_fn is not None:
            combined = _combine_scatter_buffer(buffer, self._key, self._sort_by, self._combiner_fn)
            if combined is None:
                return
            buffer = combined

        buffer_sorted = buffer.sort([_SHARD_COL, _SORT_KEY_COL])
        del buffer

        flushed_bytes = int(buffer_sorted.estimated_size())
        self._total_bytes_written += flushed_bytes
        self._total_rows_written += len(buffer_sorted)
        if _PAYLOAD_COL in buffer_sorted.columns:
            shard_sizes = buffer_sorted.group_by(_SHARD_COL).agg(pl.col(_PAYLOAD_COL).bin.size().sum().alias("bytes"))
            for shard_val, nbytes in shard_sizes.iter_rows():
                self._shard_bytes[shard_val] += int(nbytes)
        else:
            for (shard_val,), group in buffer_sorted.partition_by(_SHARD_COL, as_dict=True).items():
                self._shard_bytes[shard_val] += int(group.estimated_size())

        # Size row groups so each target shard fits in roughly one row group,
        # enabling Polars predicate pushdown to skip non-matching groups.
        num_targets = buffer_sorted[_SHARD_COL].n_unique()
        row_group_size = max(1, len(buffer_sorted) // num_targets)
        chunk_path = f"{self._data_path}c{self._n_chunks_written:04d}.parquet"
        # Ideally we'd call write_parquet directly with the GCS path, but it occationally fails with a generic error.
        buf = io.BytesIO()
        buffer_sorted.write_parquet(buf, compression="zstd", row_group_size=row_group_size)
        with open_url(chunk_path, "wb") as f:
            f.write(buf.getvalue())

        self._chunk_paths.append(chunk_path)
        self._n_chunks_written += 1

        if self._progress_log_limiter.should_run():
            logger.info(
                "[shard %d] Wrote %d scatter chunks so far (latest chunk size: %d items, %d targets)",
                self._source_shard,
                self._n_chunks_written,
                len(buffer_sorted),
                num_targets,
            )

        del buffer_sorted
        if flushed_bytes >= _GC_FLUSH_SIZE_THRESHOLD_BYTES:
            gc.collect()

    def write(self, df: pl.DataFrame) -> None:
        """Buffer a DataFrame, flushing on memory pressure.

        The DataFrame must contain ``_SHARD_COL`` (int32) and ``_SORT_KEY_COL``
        columns as produced by ``_items_to_dataframe``.
        """
        if len(df) == 0:
            return

        self._frames.append(df)
        self._buffer_estimated_bytes += int(df.estimated_size())

        if self._buffer_estimated_bytes > self._flush_threshold_bytes:
            logger.info(
                "[shard %d] Buffer estimated at %s (threshold %s); flushing scatter buffers",
                self._source_shard,
                humanfriendly.format_size(self._buffer_estimated_bytes, binary=True),
                humanfriendly.format_size(self._flush_threshold_bytes, binary=True),
            )
            self._flush()

    def write_batch(self, df: pl.DataFrame) -> None:
        """Ingest a raw batch DataFrame directly — no per-row Python conversion.

        ``key``/``sort_by`` must be a ``zephyr.expr.col(...)`` (a
        ``ColumnExpr``): an arbitrary Python callable can't be turned into a
        vectorized Polars expression, which is the whole point of this path.
        ``_SHARD_COL``/``_SORT_KEY_COL`` are computed directly from *df*'s own
        columns via Polars expressions and appended; *df*'s remaining columns
        are buffered as-is (no ``_PAYLOAD_COL``, no cloudpickle) and become
        the written chunk file's schema. If a ``combiner_fn`` is configured, it
        runs at flush time over dict rows from those columns.
        """
        assert isinstance(
            self._key, ColumnExpr
        ), f"DataFrame/RecordBatch scatter items require key=zephyr.expr.col(...), got {self._key!r}"
        if self._sort_by is not None:
            assert isinstance(
                self._sort_by, ColumnExpr
            ), f"DataFrame/RecordBatch scatter items require sort_by=zephyr.expr.col(...), got {self._sort_by!r}"

        key_expr, sort_value_expr = _column_routing_exprs(self._key, self._sort_by)
        self.write(_with_routing_columns(df, key_expr, sort_value_expr, self._num_output_shards))

    def close(self) -> ListShard:
        """Flush remaining buffers, write sidecar, return ListShard.

        Idempotent: the committed result is cached, so a second call (e.g. the
        context-manager exit after an explicit ``close()``) is a no-op.
        """
        if self._result is not None:
            return self._result
        pre_close_flushes = self._n_chunks_written
        with log_time(f"Flushing remaining buffer for {self._data_path}"):
            self._flush()

        self._avg_item_bytes = (
            self._total_bytes_written / self._total_rows_written if self._total_rows_written > 0 else 0.0
        )

        logger.info(
            "[shard %d] scatter write done: %d pre-close flushes + %d at close = %d total; avg_item_bytes=%.0f B",
            self._source_shard,
            pre_close_flushes,
            self._n_chunks_written - pre_close_flushes,
            self._n_chunks_written,
            self._avg_item_bytes,
        )

        sidecar: dict = {
            "files": list(self._chunk_paths),
            "avg_item_bytes": round(self._avg_item_bytes, 1),
            "shard_bytes": {str(k): v for k, v in self._shard_bytes.items()},
        }

        with log_time(f"Writing scatter meta for {self._data_path}"):
            _write_scatter_meta(self._data_path, sidecar)

        self._result = ListShard(refs=[MemChunk(items=[self._data_path])])
        return self._result

    def cleanup(self) -> None:
        """Discard the output instead of committing it."""
        try:
            path = StoragePath(self._data_path)
            if path.exists():
                path.rmtree()
        except Exception as e:
            logger.warning(f"Failed to cleanup scatter directory {self._data_path}: {e}")

    def __enter__(self) -> "ScatterWriter":
        return self

    def __exit__(self, exc_type: type[BaseException] | None, *exc: Any) -> None:
        if exc_type is not None:
            self.cleanup()
            return
        try:
            self.close()
        except BaseException:
            # A failure while committing (e.g. a dead R2 connection) would
            # otherwise leave the multipart upload open. Discard, then re-raise.
            self.cleanup()
            raise


@overload
def _write_scatter(
    items: Iterator,
    source_shard: int,
    data_path: str,
    key: Callable,
    num_output_shards: int,
    sort_by: Callable | None = None,
    combiner_fn: Callable | None = None,
) -> ListShard: ...


@overload
def _write_scatter(
    items: Iterator[pl.DataFrame | pa.RecordBatch],
    source_shard: int,
    data_path: str,
    key: ColumnExpr,
    num_output_shards: int,
    sort_by: ColumnExpr | None = None,
    combiner_fn: Callable | None = None,
) -> ListShard: ...


def _write_scatter(
    items: Iterator,
    source_shard: int,
    data_path: str,
    key: Callable | ColumnExpr,
    num_output_shards: int,
    sort_by: Callable | ColumnExpr | None = None,
    combiner_fn: Callable | None = None,
) -> ListShard:
    """Route items to target shards, buffer, sort, and flush as Parquet chunk files.

    Two distinct modes, selected by item type:

    * **Python items** — ``key``/``sort_by`` are Callables invoked per item;
      items are batched into DataFrames up to ``_DATAFRAME_ROW_COUNT``.
    * **DataFrame / RecordBatch items** — ``key``/``sort_by`` are
      ``zephyr.expr.col(...)`` values; each item is ingested via
      ``ScatterWriter.write_batch`` with no per-row Python conversion.

    Writes Parquet chunk files plus one ``metadata.msgpack`` sidecar.

    Returns:
        A ListShard wrapping the data file path (as the existing scatter
        plumbing expects a list of paths).
    """
    with ScatterWriter(
        data_path=data_path,
        key=key,
        source_shard=source_shard,
        num_output_shards=num_output_shards,
        sort_by=sort_by,
        combiner_fn=combiner_fn,
    ) as writer:
        pending: list[Any] = []
        for item in items:
            if isinstance(item, pa.RecordBatch):
                item = pl.from_arrow(item)
            if isinstance(item, pl.DataFrame):
                assert isinstance(
                    key, ColumnExpr
                ), "DataFrame/RecordBatch scatter items require key=zephyr.expr.col(...)"
                assert sort_by is None or isinstance(
                    sort_by, ColumnExpr
                ), "DataFrame/RecordBatch scatter items require sort_by=zephyr.expr.col(...)"

                writer.write_batch(item)
                continue

            pending.append(item)
            if len(pending) >= _DATAFRAME_ROW_COUNT:
                writer.write(_items_to_dataframe(pending, key, sort_by, num_output_shards))
                pending.clear()
        if pending:
            writer.write(_items_to_dataframe(pending, key, sort_by, num_output_shards))
        return writer.close()
