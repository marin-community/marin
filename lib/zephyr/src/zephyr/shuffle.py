# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scatter/shuffle support for Zephyr pipelines.

Each source-shard's scatter output is a set of zstd-compressed Parquet files,
one per flush per target shard (``{stem}-t{target:04d}-c{chunk:04d}.parquet``).
A msgpack sidecar (``.scatter_meta``) maps
``target_shard -> [parquet_path, ...]`` plus a global ``avg_item_bytes``
estimate.

On the read side, each reducer scans its target shard's parquet paths via
``pl.scan_parquet``, which supports Polars' streaming engine. The resulting
LazyFrames are merged via ``external_sort_merge`` (two-pass fan-in merge with
``sink_parquet`` pass-1, fully streaming).

Write-side memory is bounded by cgroup memory usage: when the process exceeds
``_SCATTER_FLUSH_THRESHOLD`` of the container limit, the largest buffers are
flushed until usage drops to ``_SCATTER_FLUSH_TARGET``.

Routing columns (``__zephyr_shard__``, ``__zephyr_sort_key__``) are added
in ``_items_to_dataframe`` and stripped before writing to disk.
"""

from __future__ import annotations

import concurrent.futures
import functools
import gc
import io
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
_SCATTER_READ_PYTHON_ROW_OVERHEAD = 10

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
# Flush any shard buffer over this limit
_SCATTER_MAX_BUFFER_BYTES = 512 * 1024 * 1024  # 512 MiB


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
    """One reducer's slice of a mapper sidecar.

    A full sidecar is ~hundreds of KB and carries chunk paths for every
    target shard (tens of thousands on large jobs). A reducer only consumes
    its own shard's paths plus one scalar, so the worker extracts just
    those fields and discards the parsed dict before returning.
    """

    path: str
    chunks: list[str]  # GCS parquet paths, one per flush for this target
    chunk_bytes: list[int]  # compressed byte size of each chunk file
    avg_item_bytes: float


def _read_sidecar_slice(path: str, shard_key: str) -> _SidecarSlice | None:
    """Read one sidecar and extract only the fields for ``shard_key``.

    Returns ``None`` if the sidecar has no entry for this shard.

    Uses ``fs.cat_file`` rather than ``open_url`` — one direct GET returning
    bytes is ~25% faster than going through ``TextIOWrapper(BufferedFile)``
    for small sidecars, and msgpack decodes bytes directly.
    """
    meta_path = _scatter_meta_path(path)
    fs, fs_path = url_to_fs(meta_path)
    meta = _sidecar_decoder().decode(fs.cat_file(fs_path))
    raw = meta.get("shards", {}).get(shard_key)
    if not raw:
        return None
    if "avg_item_bytes" not in meta:
        raise ValueError(f"Sidecar {meta_path} has entry for shard {shard_key} but no avg_item_bytes.")
    return _SidecarSlice(
        path=path,
        chunks=[str(p) for p in raw["paths"]],
        chunk_bytes=[int(b) for b in raw["bytes"]],
        avg_item_bytes=float(meta["avg_item_bytes"]),
    )


def _read_sidecar_slices_parallel(scatter_paths: list[str], target_shard: int) -> list[_SidecarSlice]:
    """Read every sidecar concurrently and return per-shard slices in input order.

    Extraction happens inside the worker so full sidecar dicts never
    accumulate in the reducer process. Sidecars with no ranges for
    ``target_shard`` are dropped.

    TODO(rav): each reducer subprocess re-reads every sidecar even though only
    one shard's byte ranges are used. A worker-level sidecar cache (or a shared
    read across colocated reducers) would avoid the redundant GCS GETs when
    many reducers run on the same host.
    """
    shard_key = str(target_shard)
    ordered: list[_SidecarSlice | None] = [None] * len(scatter_paths)
    with concurrent.futures.ThreadPoolExecutor(max_workers=_SIDECAR_READ_CONCURRENCY) as pool:
        futures = {pool.submit(_read_sidecar_slice, p, shard_key): i for i, p in enumerate(scatter_paths)}
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
        total_chunk_bytes: int = 0,
    ) -> None:
        self._files = files
        self._target_shard = target_shard
        self.avg_item_bytes = avg_item_bytes
        self.total_chunk_bytes = total_chunk_bytes

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
        total_chunk_bytes = 0

        with log_time(
            f"Building ScatterReader for target shard {target_shard} "
            f"from {len(scatter_paths)} sidecars (concurrency={_SIDECAR_READ_CONCURRENCY})"
        ):
            for slice_ in _read_sidecar_slices_parallel(scatter_paths, target_shard):
                files.append((slice_.path, slice_.chunks))
                weighted_bytes += slice_.avg_item_bytes * len(slice_.chunks)
                total_chunks += len(slice_.chunks)
                total_chunk_bytes += sum(slice_.chunk_bytes)

        avg_item_bytes = weighted_bytes / total_chunks if total_chunks > 0 else 0.0

        logger.info(
            "ScatterReader for shard %d: %d source files, %d total chunks, " "total_compressed=%s, avg_item_bytes=%.1f",
            target_shard,
            len(files),
            total_chunks,
            humanfriendly.format_size(total_chunk_bytes, binary=True),
            avg_item_bytes,
        )
        return cls(
            files=files,
            target_shard=target_shard,
            avg_item_bytes=avg_item_bytes,
            total_chunk_bytes=total_chunk_bytes,
        )

    def get_frames(self) -> list[pl.LazyFrame]:
        return [pl.scan_parquet(path) for _, chunk_paths in self._files for path in chunk_paths]

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
    """Writes scatter chunk files, one per flush, as zstd-compressed Parquet.

    DataFrames are routed to target shards via the ``_SHARD_COL`` column, buffered
    per shard as Polars DataFrames, optionally combined, sorted by ``_SORT_KEY_COL``,
    and written to individual ``{stem}-t{target:04d}-c{chunk:04d}.parquet`` files.
    A msgpack sidecar with per-target parquet paths is written on close.

    Flushing is cgroup-memory-based: when the container memory usage exceeds
    ``_SCATTER_FLUSH_THRESHOLD``, the largest buffers are flushed until usage
    drops to ``_SCATTER_FLUSH_TARGET``.  Cgroup memory is used because that is
    what the OOM killer enforces and it captures all memory in the process,
    including the reduce phase running concurrently.
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

        self._buffers: dict[int, pl.DataFrame] = {}
        self._chunk_paths: dict[int, list[str]] = defaultdict(list)
        self._chunk_bytes: dict[int, list[int]] = defaultdict(list)
        self._avg_item_bytes: float = 0.0
        self._total_bytes_written: int = 0
        self._total_rows_written: int = 0
        self._n_chunks_written = 0
        # Throttles the per-flush progress log so high-fanout workloads don't log too often
        self._progress_log_limiter = RateLimiter(interval_seconds=_PROGRESS_LOG_INTERVAL_SECONDS)
        self._peak_rss_bytes: int = 0
        self._write_calls: int = 0
        self._total_compactions: int = 0

        ensure_parent_dir(data_path)

    def _flush(self, target: int) -> None:
        try:
            df = self._buffers[target]

            if df.is_empty():
                return

            if self._combiner_fn is not None:
                rows = list(_dataframe_to_items(df))
                rows = _apply_combiner(rows, self._key_fn, self._combiner_fn)

                if len(rows) == 0:
                    return

                df = _items_to_dataframe(rows, self._key_fn, self._sort_fn, num_output_shards=0).drop(_SHARD_COL)

            sorted_df = df.sort(_SORT_KEY_COL)

            self._total_bytes_written += sorted_df.estimated_size()
            self._total_rows_written += len(sorted_df)

            # Write each flush as its own parquet file so we can use scan_parquet to stream it lazily in the reducer.
            chunk_path = f"{self._data_path}t{target:04d}-c{self._n_chunks_written:04d}.parquet"
            buf = io.BytesIO()
            sorted_df.write_parquet(buf, compression="zstd")
            chunk_data = buf.getvalue()
            with open_url(chunk_path, "wb") as f:
                f.write(chunk_data)

            self._chunk_paths[target].append(chunk_path)
            self._chunk_bytes[target].append(len(chunk_data))

            self._n_chunks_written += 1
            if self._progress_log_limiter.should_run():
                logger.info(
                    "[shard %d] Wrote %d scatter chunks so far (latest chunk size: %d items)",
                    self._source_shard,
                    self._n_chunks_written,
                    len(df),
                )
        finally:
            del self._buffers[target]
            gc.collect()  # Force gc, since flush creates copies of the DataFrame and buffers

    def _compact_buffers(self) -> None:
        """Rechunk each shard's DataFrame to consolidate accumulated Arrow chunks.

        extend() appends new Arrow chunks rather than extending the buffer in-place.
        Without rechunking, O(calls) tiny chunks accumulate per shard, increasing
        memory overhead rechunk() collapses all chunks into one contiguous buffer.
        """
        for shard, df in self._buffers.items():
            self._buffers[shard] = df.rechunk()
        self._total_compactions += 1

    def write(self, df: pl.DataFrame) -> None:
        """Route a DataFrame to per-shard buffers, flushing when over budget.

        The DataFrame must contain ``_SHARD_COL`` (int32) and ``_SORT_KEY_COL``
        (binary) columns produced by ``_items_to_dataframe``.
        """
        if len(df) == 0:
            return

        for (shard_val,), partition in df.partition_by(_SHARD_COL, as_dict=True).items():
            stripped = partition.drop(_SHARD_COL)
            if shard_val in self._buffers:
                self._buffers[shard_val].extend(stripped)
            else:
                self._buffers[shard_val] = stripped

            if self._buffers[shard_val].estimated_size() > _SCATTER_MAX_BUFFER_BYTES:
                self._flush(shard_val)

        self._write_calls += 1
        if self._write_calls % _BUFFER_COMPACTION_INTERVAL == 0:
            self._compact_buffers()

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
                while self._buffers and _read_cgroup_memory_bytes() > self._flush_target_bytes:
                    largest = max(self._buffers, key=lambda t: self._buffers[t].estimated_size())
                    self._flush(largest)

    def close(self) -> ListShard:
        """Flush remaining buffers, write sidecar."""
        pre_close_flushes = self._n_chunks_written
        with log_time(f"Flushing remaining buffers for {self._data_path}"):
            for target in sorted(self._buffers.keys()):
                self._flush(target)

        self._avg_item_bytes = (
            self._total_bytes_written / self._total_rows_written if self._total_rows_written > 0 else 0.0
        )

        logger.info(
            "[shard %d] scatter write done: %d pre-close flushes + %d at close = %d total; "
            "avg_item_bytes=%.0f B, peak_rss=%d MB, compactions=%d",
            self._source_shard,
            pre_close_flushes,
            self._n_chunks_written - pre_close_flushes,
            self._n_chunks_written,
            self._avg_item_bytes,
            self._peak_rss_bytes // (1024 * 1024),
            self._total_compactions,
        )

        sidecar: dict = {
            "shards": {
                str(t): {"paths": paths, "bytes": self._chunk_bytes[t]} for t, paths in self._chunk_paths.items()
            },
        }
        if self._avg_item_bytes > 0:
            sidecar["avg_item_bytes"] = round(self._avg_item_bytes, 1)

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
