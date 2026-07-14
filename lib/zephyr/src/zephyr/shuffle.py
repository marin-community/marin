# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scatter/shuffle support for Zephyr pipelines.

Each source-shard's scatter output is a set of zstd-compressed Parquet files,
one combined file per flush (``c{chunk:04d}.parquet``) containing all target
shards' data sorted by ``(_SHARD_COL, _SORT_KEY_COL)``.  A msgpack sidecar
(``.scatter_meta``) records ``files -> [{path, bytes}, ...]``, a global
``avg_item_bytes`` estimate, and exact per-target-shard payload bytes
(``shard_bytes``) used by reducers to size the external-sort decision.

On the read side, each reducer scans only its target shard via
``pl.scan_parquet(path).filter(pl.col(_SHARD_COL) == target).drop(_SHARD_COL)``.
Polars predicate pushdown with row-group statistics skips non-matching row
groups via byte-range GETs, so each reducer reads roughly 1/N of each file.
The resulting LazyFrames are merged via ``external_sort_merge`` (two-pass
fan-in merge with ``sink_parquet`` pass-1, fully streaming).

Write-side memory is bounded by cgroup memory usage: when the process exceeds
``_SCATTER_FLUSH_THRESHOLD`` of the container limit, all buffers are flushed
together into one combined file and usage drops to ``_SCATTER_FLUSH_TARGET``.

Routing columns (``__zephyr_shard__``, ``__zephyr_sort_key__``) are added
in ``_items_to_dataframe``; ``__zephyr_shard__`` is stripped on read,
``__zephyr_sort_key__`` is consumed by the merge and stripped after.

``_write_scatter`` accepts plain Python items (one row each) as well as
``pl.DataFrame``/``pa.RecordBatch`` items (one full batch of rows each, as
produced by a ``.map()`` step chained after ``load_parquet(batch_mode=True)``).
A batch item is ingested directly — no per-row Python conversion, no
``_PAYLOAD_COL``: the batch's own columns become the chunk file's schema, and
``_SHARD_COL``/``_SORT_KEY_COL`` are computed as vectorized Polars
expressions from ``key``/``sort_by`` (which must be a ``zephyr.expr.col(...)``
for batch items, since an arbitrary Python callable can't be vectorized).
On read, ``_dataframe_to_items`` reconstructs dict rows from the batch's own
columns when ``_PAYLOAD_COL`` is absent. See ``ScatterWriter.write_batch``.
"""

import concurrent.futures
import functools
import gc
import io
import logging
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import cloudpickle
import humanfriendly
import msgspec
import polars as pl
import psutil
import pyarrow as pa
from iris.env_resources import TaskResources
from rigging.filesystem import StoragePath, open_url, url_to_fs
from rigging.timing import RateLimiter, log_time

from zephyr.expr import ColumnExpr, Expr
from zephyr.external_sort import external_sort_merge
from zephyr.shard_keys import deterministic_hash
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

    def get_iterators(self) -> Iterator[Iterator]:
        for ref in self.refs:
            yield iter(ref)


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
# Fraction of local disk space to use for shuffle.
_LOCAL_DISK_SHUFFLE_UTILIZATION = 0.9
# Polars streaming chunk size, important to avoid excessive memory usage during merge.
_POLARS_STREAMING_CHUNK_SIZE = 10000
# Seed for the vectorized shard-routing hash used by ScatterWriter.write_batch.
# Deterministic across processes for a fixed Polars version (verified); does
# not need to match deterministic_hash's algorithm — see write_batch's docstring.
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
# Number of write() calls between buffer compaction passes.
_BUFFER_COMPACTION_INTERVAL = 100
# Number of write() calls between memory checks.
_MEMORY_CHECK_INTERVAL = 10
# Flush scatter buffers when cgroup memory exceeds this fraction of the container
# limit, and keep flushing until usage drops to _SCATTER_FLUSH_TARGET.
_SCATTER_FLUSH_THRESHOLD = 0.75
_SCATTER_FLUSH_TARGET = 0.60


def _read_cgroup_memory_bytes() -> int:
    """Read current memory usage in bytes from the cgroup controller.

    Falls back to process RSS when running outside a cgroup (e.g., local dev).
    """
    try:
        with open("/sys/fs/cgroup/memory.current") as f:
            return int(f.read().strip())
    except OSError:
        pass
    try:
        with open("/sys/fs/cgroup/memory/memory.usage_in_bytes") as f:
            return int(f.read().strip())
    except OSError:
        pass
    return int(psutil.Process().memory_info().rss)


def _dataframe_to_items(df: pl.DataFrame) -> Iterator[Any]:
    """Yield Python items from a DataFrame, stripping routing columns.

    Payload-bearing chunks (Python-item writes) deserialize ``_PAYLOAD_COL``
    via cloudpickle. Batch-ingested chunks (``ScatterWriter.write_batch``, no
    ``_PAYLOAD_COL``) have no payload to deserialize — their own columns
    already are the item, so rows are reconstructed directly.
    """
    if _PAYLOAD_COL in df.columns:
        for p in df[_PAYLOAD_COL].to_list():
            yield cloudpickle.loads(p)
    else:
        yield from df.drop(_SORT_KEY_COL).to_dicts()


def _columns_to_dataframe(
    payloads: list[bytes],
    shards: list[int],
    keys: list[Any],
    sort_values: list[Any],
) -> pl.DataFrame:
    """Build the scatter DataFrame from pre-computed flat columns.

    The sort-key struct is folded from two flat columns rather than per-row
    Python dicts: series construction from homogeneous lists is the native
    fast path, and ``pl.struct`` over existing columns is cheap. Field order
    (key first) drives the (key, sort_value) sort order.
    """
    try:
        return pl.DataFrame(
            {
                _PAYLOAD_COL: pl.Series(payloads, dtype=pl.Binary),
                _SHARD_COL: pl.Series(shards, dtype=pl.Int32),
                _KEY_TMP_COL: keys,
                _SORT_VALUE_TMP_COL: sort_values,
            }
        ).select(
            _PAYLOAD_COL,
            _SHARD_COL,
            pl.struct(
                pl.col(_KEY_TMP_COL).alias("key"),
                pl.col(_SORT_VALUE_TMP_COL).alias("sort_value"),
            ).alias(_SORT_KEY_COL),
        )
    except (TypeError, pl.exceptions.InvalidOperationError) as err:
        # Non-serializable keys surface as TypeError from Series construction
        # or InvalidOperationError ("nested objects are not allowed") when the
        # key column lands as Object dtype and pl.struct rejects it.
        raise ValueError("key_fn must return an Arrow-serializable object.") from err


def _as_row_fn(value: Callable | Expr | None) -> Callable | None:
    """Normalize a ``key``/``sort_by`` value to a plain row-wise callable.

    ``key``/``sort_by`` may be a ``zephyr.expr.col(...)`` (for the vectorized
    batch-ingestion path) or a plain ``Callable`` (row-wise, as before).
    Everywhere a single Python item must be evaluated against it — the
    Python-item write path, the combiner, and the reduce-side ``groupby`` —
    this reduces both forms to one calling convention.
    """
    if isinstance(value, Expr):
        return value.evaluate
    return value


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
    directly.
    """
    shards: list[int] = []
    keys: list[Any] = []
    sort_values: list[Any] = []
    for item in items:
        key = key_fn(item)
        shards.append(deterministic_hash(key) % num_output_shards if num_output_shards > 0 else 0)
        keys.append(key)
        sort_values.append(sort_fn(item) if sort_fn is not None else None)
    payloads = [cloudpickle.dumps(item) for item in items]
    return _columns_to_dataframe(payloads, shards, keys, sort_values)


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
        return [
            pl.scan_parquet(path).filter(pl.col(_SHARD_COL) == self._target_shard).drop(_SHARD_COL)
            for _, chunk_paths in self._files
            for path in chunk_paths
        ]

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
            ctx = _worker_ctx_var.get()
            if ctx is not None and ctx.task_memory_bytes > 0:
                memory_bytes = ctx.task_memory_bytes
            else:
                memory_bytes = TaskResources.from_environment().memory_bytes
                if memory_bytes <= 0:
                    memory_bytes = 1024 * 1024 * 1024

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

    Flushing is cgroup-memory-based: when the container memory usage exceeds
    ``_SCATTER_FLUSH_THRESHOLD``, all buffers are flushed together into one
    combined file until usage drops to ``_SCATTER_FLUSH_TARGET``.
    """

    def __init__(
        self,
        data_path: str,
        key: Callable | Expr,
        source_shard: int,
        num_output_shards: int,
        sort_by: Callable | Expr | None = None,
        combiner_fn: Callable | None = None,
    ) -> None:
        self._data_path = data_path if data_path.endswith("/") else f"{data_path}/"
        # Raw key/sort_by, used by write_batch's vectorized Polars path.
        self._key = key
        self._sort_by = sort_by
        # Row-wise callable form (Expr.evaluate if an Expr, else unchanged),
        # used by the combiner path, which operates on individual Python items.
        self._key_fn = _as_row_fn(key)
        self._sort_fn = _as_row_fn(sort_by)
        self._num_output_shards = num_output_shards

        self._source_shard = source_shard
        self._combiner_fn = combiner_fn
        self._memory_available_bytes = TaskResources.from_environment().memory_bytes
        if self._memory_available_bytes == 0:
            logger.warning("No memory available for scatter write, defaulting to 1GB. This will likely fail.")
            self._memory_available_bytes = 1024 * 1024 * 1024
        self._flush_threshold_bytes = int(self._memory_available_bytes * _SCATTER_FLUSH_THRESHOLD)
        self._flush_target_bytes = int(self._memory_available_bytes * _SCATTER_FLUSH_TARGET)

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
        self._peak_rss_bytes: int = 0
        self._write_calls: int = 0

        ensure_parent_dir(self._data_path)
        self._result: ListShard | None = None

    def _flush(self) -> None:
        """Flush the accumulated buffer into one combined Parquet file sorted by [_SHARD_COL, _SORT_KEY_COL]."""
        if not self._frames:
            gc.collect()
            return

        buffer = pl.concat(self._frames, rechunk=False)
        self._frames = []

        if self._combiner_fn is not None:
            # combiner_fn re-groups and reduces individual Python items, so it
            # requires a per-row payload to operate on. write_batch() asserts
            # this case can't arise for batch-ingested frames; this is a
            # defensive second check in case some other path buffers a
            # payload-less frame while a combiner is configured.
            assert _PAYLOAD_COL in buffer.columns, (
                "combiner_fn is not supported for DataFrame/RecordBatch-sourced scatter items "
                "(no per-row payload to combine over)"
            )
            frames: list[pl.DataFrame] = []
            for (shard_val,), group in buffer.partition_by(_SHARD_COL, as_dict=True).items():
                rows = list(_dataframe_to_items(group))
                rows = _apply_combiner(rows, self._key_fn, self._combiner_fn)
                if not rows:
                    continue
                df = _items_to_dataframe(rows, self._key_fn, self._sort_fn, num_output_shards=0)
                frames.append(df.with_columns(pl.lit(shard_val, dtype=pl.Int32).alias(_SHARD_COL)))
            if not frames:
                gc.collect()
                return
            buffer = pl.concat(frames, rechunk=True)

        buffer_sorted = buffer.sort([_SHARD_COL, _SORT_KEY_COL])
        del buffer

        self._total_bytes_written += int(buffer_sorted.estimated_size())
        self._total_rows_written += len(buffer_sorted)
        if _PAYLOAD_COL in buffer_sorted.columns:
            shard_sizes = buffer_sorted.group_by(_SHARD_COL).agg(pl.col(_PAYLOAD_COL).bin.size().sum().alias("bytes"))
            for shard_val, nbytes in shard_sizes.iter_rows():
                self._shard_bytes[shard_val] += int(nbytes)
        else:
            # No per-row payload to measure exactly; Polars' own per-shard
            # memory estimate is the same kind of estimate already used for
            # _total_bytes_written above.
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
        gc.collect()

    def write(self, df: pl.DataFrame) -> None:
        """Buffer a DataFrame, flushing on memory pressure.

        The DataFrame must contain ``_SHARD_COL`` (int32) and ``_SORT_KEY_COL``
        columns as produced by ``_items_to_dataframe``.
        """
        if len(df) == 0:
            return

        self._frames.append(df)
        self._write_calls += 1

        if self._write_calls % _MEMORY_CHECK_INTERVAL == 0:
            mem = _read_cgroup_memory_bytes()
            if mem > self._peak_rss_bytes:
                self._peak_rss_bytes = mem

            if mem > self._flush_threshold_bytes:
                logger.info(
                    "[shard %d] Memory at %s (%.0f%% of %s); flushing scatter buffers to %.0f%%",
                    self._source_shard,
                    humanfriendly.format_size(mem, binary=True),
                    100.0 * mem / self._memory_available_bytes,
                    humanfriendly.format_size(self._memory_available_bytes, binary=True),
                    100.0 * _SCATTER_FLUSH_TARGET,
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
        the written chunk file's schema.

        The shard hash uses Polars' native ``Expr.hash()`` rather than
        ``deterministic_hash`` (no vectorized equivalent of the latter
        exists). This is safe because ``_SHARD_COL`` is computed once here
        and only ever read back (never recomputed) on the reduce side — the
        only requirement is that every mapper task in this Scatter stage
        route a given key to the same shard, which holds since they all run
        this same method with the same hash function.
        """
        assert self._combiner_fn is None, "combiner_fn is not supported for DataFrame/RecordBatch-sourced scatter items"
        assert isinstance(
            self._key, ColumnExpr
        ), f"DataFrame/RecordBatch scatter items require key=zephyr.expr.col(...), got {self._key!r}"
        if self._sort_by is not None:
            assert isinstance(
                self._sort_by, ColumnExpr
            ), f"DataFrame/RecordBatch scatter items require sort_by=zephyr.expr.col(...), got {self._sort_by!r}"

        key_expr = pl.col(self._key.name)
        sort_value_expr = pl.col(self._sort_by.name) if self._sort_by is not None else pl.lit(None)
        shard_expr = (key_expr.hash(seed=_SCATTER_HASH_SEED) % self._num_output_shards).cast(pl.Int32)
        sort_key_expr = pl.struct(key_expr.alias("key"), sort_value_expr.alias("sort_value"))
        self.write(df.with_columns(shard_expr.alias(_SHARD_COL), sort_key_expr.alias(_SORT_KEY_COL)))

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
            "[shard %d] scatter write done: %d pre-close flushes + %d at close = %d total; "
            "avg_item_bytes=%.0f B, peak_rss=%d MB",
            self._source_shard,
            pre_close_flushes,
            self._n_chunks_written - pre_close_flushes,
            self._n_chunks_written,
            self._avg_item_bytes,
            self._peak_rss_bytes // (1024 * 1024),
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


def _write_scatter(
    items: Iterator,
    source_shard: int,
    data_path: str,
    key_fn: Callable | Expr,
    num_output_shards: int,
    sort_fn: Callable | Expr | None = None,
    combiner_fn: Callable | None = None,
) -> ListShard:
    """Route items to target shards, buffer, sort, and flush as Parquet chunk files.

    Plain Python items are routed by calling ``key_fn``/``sort_fn`` per item
    (in Python, since arbitrary callables can't be vectorized) and batched
    into DataFrames up to ``_DATAFRAME_ROW_COUNT`` at a time. A
    ``pl.DataFrame`` or ``pa.RecordBatch`` item (e.g. from an upstream
    ``.map()`` over ``load_parquet(batch_mode=True)``) already represents a
    full batch of rows and is ingested directly via
    ``ScatterWriter.write_batch`` — no per-row Python conversion — which
    requires ``key_fn``/``sort_fn`` to be a ``zephyr.expr.col(...)``.
    Writes Parquet chunk files plus one ``metadata.msgpack`` sidecar.

    Returns:
        A ListShard wrapping the data file path (as the existing scatter
        plumbing expects a list of paths).
    """
    # Row-wise form for the plain-Python-item path below; write_batch (used
    # for DataFrame/RecordBatch items) uses the raw key_fn/sort_fn directly.
    row_key_fn = _as_row_fn(key_fn)
    row_sort_fn = _as_row_fn(sort_fn)
    with ScatterWriter(
        data_path=data_path,
        key=key_fn,
        source_shard=source_shard,
        num_output_shards=num_output_shards,
        sort_by=sort_fn,
        combiner_fn=combiner_fn,
    ) as writer:
        pending: list[Any] = []
        for item in items:
            if isinstance(item, pa.RecordBatch):
                item = pl.from_arrow(item)
            if isinstance(item, pl.DataFrame):
                writer.write_batch(item)
                continue
            pending.append(item)
            if len(pending) >= _DATAFRAME_ROW_COUNT:
                writer.write(_items_to_dataframe(pending, row_key_fn, row_sort_fn, num_output_shards))
                pending.clear()
        if pending:
            writer.write(_items_to_dataframe(pending, row_key_fn, row_sort_fn, num_output_shards))
        return writer.close()
