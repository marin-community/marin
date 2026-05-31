# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scatter/shuffle support for Zephyr pipelines.

Each source-shard's scatter output is a set of zstd-compressed Parquet files,
one combined file per flush (``c{chunk:04d}.parquet``) containing all target
shards' data sorted by ``(_SHARD_COL, _SORT_KEY_COL)``.  A msgpack sidecar
(``.scatter_meta``) records ``files -> [{path, bytes}, ...]`` plus a global
``avg_item_bytes`` estimate.

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
"""

from __future__ import annotations

import concurrent.futures
import functools
import gc
import itertools
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
from iris.env_resources import TaskResources
from rigging.filesystem import open_url, url_to_fs
from rigging.timing import RateLimiter, log_time

from zephyr.external_sort import external_sort_merge
from zephyr.shard_keys import composite_sort_key, deterministic_hash
from zephyr.worker_context import zephyr_worker_ctx
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

# Helper column names injected by _items_to_dataframe and stripped before
# writing to disk.  Both are internal implementation details; user schemas must
# not collide with these names.
_SHARD_COL = "__zephyr_shard__"
_SORT_KEY_COL = "__zephyr_sort_key__"
# A cloudpickle-serialized Python object representing the item
_PAYLOAD_COL = "__payload__"

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
    """Yield Python items from a DataFrame, stripping routing columns and deserializing payloads."""
    for p in df[_PAYLOAD_COL].to_list():
        yield cloudpickle.loads(p)


def _items_to_dataframe(
    items: list[Any],
    key_fn: Callable,
    sort_fn: Callable | None,
    num_output_shards: int,
) -> pl.DataFrame:
    """Convert a list of Python items to a DataFrame with routing columns.

    Cloudpickle-serializes items into ``_PAYLOAD_COL`` and adds ``_SHARD_COL``
    (int32 target shard index) and ``_SORT_KEY_COL``.  These columns are
    consumed by :class:`ScatterWriter`; ``_SHARD_COL`` is stripped on write,
    ``_SORT_KEY_COL`` is stripped on read.
    """
    shards: list[int] = []
    sort_keys: list[dict[str, object | None]] = []
    for item in items:
        key = key_fn(item)
        shards.append(deterministic_hash(key) % num_output_shards if num_output_shards > 0 else 0)
        sort_value = sort_fn(item) if sort_fn is not None else None
        sort_keys.append({"key": key, "sort_value": sort_value})

    try:
        payloads = [cloudpickle.dumps(item) for item in items]
        return pl.DataFrame({_PAYLOAD_COL: pl.Series(payloads, dtype=pl.Binary)}).with_columns(
            [
                pl.Series(_SHARD_COL, shards, dtype=pl.Int32),
                pl.Series(_SORT_KEY_COL, sort_keys),
            ]
        )
    except TypeError as err:
        raise ValueError("key_fn must return an Arrow-serializable object.") from err


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
        with open_url(meta_path, "wb") as f:
            f.write(payload)


@dataclass(frozen=True)
class _SidecarSlice:
    """Chunk paths and metadata from one mapper's sidecar.

    Each entry in ``chunk_paths`` is one combined Parquet file written during a
    flush; the file contains data for all target shards sorted by
    ``(_SHARD_COL, _SORT_KEY_COL)``.
    """

    path: str
    chunk_paths: list[str]  # GCS parquet paths, one per flush event
    avg_item_bytes: float


def _read_sidecar_slice(path: str) -> _SidecarSlice | None:
    """Read one sidecar and return its file list.

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
    )


def _read_sidecar_slices_parallel(scatter_paths: list[str]) -> list[_SidecarSlice]:
    """Read every sidecar concurrently and return slices in input order.

    Empty sidecars (no files written) are dropped from the result.
    """
    ordered: list[_SidecarSlice | None] = [None] * len(scatter_paths)
    with concurrent.futures.ThreadPoolExecutor(max_workers=_SIDECAR_READ_CONCURRENCY) as pool:
        futures = {pool.submit(_read_sidecar_slice, p): i for i, p in enumerate(scatter_paths)}
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
    ) -> None:
        self._files = files
        self._target_shard = target_shard
        self.avg_item_bytes = avg_item_bytes

    @classmethod
    def from_sidecars(cls, scatter_paths: list[str], target_shard: int) -> ScatterReader:
        """Build a ScatterReader by reading per-mapper sidecars directly.

        Each reducer reads every mapper's ``.scatter_meta`` sidecar in parallel
        and filters for its own ``target_shard``. No coordinator-written manifest
        is needed, which eliminates a serialization bottleneck when there are
        thousands of mappers.
        """
        files: list[tuple[str, list[str]]] = []
        weighted_bytes = 0.0
        total_chunks = 0

        with log_time(
            f"Building ScatterReader for target shard {target_shard} "
            f"from {len(scatter_paths)} sidecars (concurrency={_SIDECAR_READ_CONCURRENCY})"
        ):
            for slice_ in _read_sidecar_slices_parallel(scatter_paths):
                files.append((slice_.path, slice_.chunk_paths))
                weighted_bytes += slice_.avg_item_bytes * len(slice_.chunk_paths)
                total_chunks += len(slice_.chunk_paths)

        avg_item_bytes = weighted_bytes / total_chunks if total_chunks > 0 else 0.0

        logger.info(
            "ScatterReader for shard %d: %d source files, %d total chunks, avg_item_bytes=%.1f",
            target_shard,
            len(files),
            total_chunks,
            avg_item_bytes,
        )
        return cls(
            files=files,
            target_shard=target_shard,
            avg_item_bytes=avg_item_bytes,
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

            estimated_merge_memory_bytes = self.avg_item_bytes * self.total_chunks * _POLARS_STREAMING_CHUNK_SIZE
            # Overhead per row in the Polars DataFrame plus the deserialized Python object.
            # Future Polars-only processing would remove the Python overhead.
            overhead = _SCATTER_READ_POLARS_ROW_OVERHEAD * _SCATTER_READ_PYTHON_ROW_OVERHEAD
            num_workers = zephyr_worker_ctx().num_workers

            task_resources = TaskResources.from_environment()
            memory_bytes = task_resources.memory_bytes / num_workers

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

                merged = external_sort_merge(
                    input_frames=self.get_frames(),
                    sort_key=_SORT_KEY_COL,
                    external_sort_dir=external_sort_dir,
                    fan_in=fan_in,
                    shard=self._target_shard,
                )

                yield from itertools.chain.from_iterable(map(_dataframe_to_items, merged))

            else:
                logger.info(
                    "[shard %d] Merging %d chunks in memory (%s memory needed < %s memory available)",
                    self._target_shard,
                    self.total_chunks,
                    humanfriendly.format_size(estimated_merge_memory_bytes * overhead, binary=True),
                    humanfriendly.format_size(memory_bytes * _SCATTER_READ_MEMORY_FRACTION, binary=True),
                )
                merged_lf = pl.merge_sorted(self.get_frames(), key=_SORT_KEY_COL)

                yield from itertools.chain.from_iterable(
                    _dataframe_to_items(batch) for batch in merged_lf.collect_batches()
                )


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

    DataFrames are routed to per-target buffers via ``_SHARD_COL``. Each flush
    combines all buffered targets into a single ``c{chunk:04d}.parquet`` file,
    sorted by ``[_SHARD_COL, _SORT_KEY_COL]`` with row groups sized so Polars
    predicate pushdown skips non-target row groups on the read side.

    Flushing is cgroup-memory-based: when the container memory usage exceeds
    ``_SCATTER_FLUSH_THRESHOLD``, all buffers are flushed together into one
    combined file until usage drops to ``_SCATTER_FLUSH_TARGET``.
    """

    def __init__(
        self,
        data_path: str,
        key_fn: Callable,
        source_shard: int,
        sort_fn: Callable | None = None,
        combiner_fn: Callable | None = None,
    ) -> None:
        self._data_path = data_path if data_path.endswith("/") else f"{data_path}/"
        self._key_fn = key_fn
        self._sort_fn = sort_fn
        self._sort_key = composite_sort_key(key_fn, sort_fn)

        self._source_shard = source_shard
        self._combiner_fn = combiner_fn
        self._memory_available_bytes = TaskResources.from_environment().memory_bytes
        if self._memory_available_bytes == 0:
            logger.warning("No memory available for scatter write, defaulting to 1GB. This will likely fail.")
            self._memory_available_bytes = 1024 * 1024 * 1024
        self._flush_threshold_bytes = int(self._memory_available_bytes * _SCATTER_FLUSH_THRESHOLD)
        self._flush_target_bytes = int(self._memory_available_bytes * _SCATTER_FLUSH_TARGET)

        self._buffer: pl.DataFrame | None = None
        self._chunk_paths: list[str] = []
        self._avg_item_bytes: float = 0.0
        self._total_bytes_written: int = 0
        self._total_rows_written: int = 0
        self._n_chunks_written = 0
        # Throttles the per-flush progress log so high-fanout workloads don't log too often
        self._progress_log_limiter = RateLimiter(interval_seconds=_PROGRESS_LOG_INTERVAL_SECONDS)
        self._peak_rss_bytes: int = 0
        self._write_calls: int = 0

        ensure_parent_dir(self._data_path)

    def _flush(self) -> None:
        """Flush the accumulated buffer into one combined Parquet file sorted by [_SHARD_COL, _SORT_KEY_COL]."""
        if self._buffer is None:
            gc.collect()
            return

        buffer = self._buffer
        self._buffer = None

        if self._combiner_fn is not None:
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

        self._total_bytes_written += buffer_sorted.estimated_size()
        self._total_rows_written += len(buffer_sorted)

        # Size row groups so each target shard fits in roughly one row group,
        # enabling Polars predicate pushdown to skip non-matching groups.
        num_targets = buffer_sorted[_SHARD_COL].n_unique()
        row_group_size = max(1, len(buffer_sorted) // num_targets)
        chunk_path = f"{self._data_path}c{self._n_chunks_written:04d}.parquet"
        buffer_sorted.write_parquet(chunk_path, compression="zstd", row_group_size=row_group_size, use_pyarrow=True)

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
        """Accumulate a DataFrame into the buffer, flushing on memory pressure.

        The DataFrame must contain ``_SHARD_COL`` (int32) and ``_SORT_KEY_COL``
        columns produced by ``_items_to_dataframe``.
        """
        if len(df) == 0:
            return

        if self._buffer is None:
            self._buffer = df
        else:
            self._buffer.extend(df)
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

    def close(self) -> ListShard:
        """Flush remaining buffers, write sidecar."""
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
        }

        with log_time(f"Writing scatter meta for {self._data_path}"):
            _write_scatter_meta(self._data_path, sidecar)

        return ListShard(refs=[MemChunk(items=[self._data_path])])

    def __enter__(self) -> ScatterWriter:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _write_scatter(
    items: Iterator,
    source_shard: int,
    data_path: str,
    key_fn: Callable,
    num_output_shards: int,
    sort_fn: Callable | None = None,
    combiner_fn: Callable | None = None,
) -> ListShard:
    """Route items to target shards, buffer, sort, and flush as Parquet chunk files.

    Routing and sort keys are computed here (in Python, since ``key_fn`` and
    ``sort_fn`` are arbitrary callables) and embedded as helper columns in the DataFrame.
    Items are batched into DataFrames.
    Writes Parquet chunk files plus one ``metadata.msgpack`` sidecar.

    Returns:
        A ListShard wrapping the data file path (as the existing scatter
        plumbing expects a list of paths).
    """
    writer = ScatterWriter(
        data_path=data_path,
        key_fn=key_fn,
        source_shard=source_shard,
        sort_fn=sort_fn,
        combiner_fn=combiner_fn,
    )
    pending: list[Any] = []
    for item in items:
        pending.append(item)
        if len(pending) >= _DATAFRAME_ROW_COUNT:
            writer.write(_items_to_dataframe(pending, key_fn, sort_fn, num_output_shards))
            pending.clear()
    if pending:
        writer.write(_items_to_dataframe(pending, key_fn, sort_fn, num_output_shards))
    return writer.close()
