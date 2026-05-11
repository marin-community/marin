# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scatter/shuffle support for Zephyr pipelines.

Each source-shard's scatter output is a collection of zstd-compressed Polars
IPC files, one per flush (chunk), named ``{stem}-{target:04d}-{chunk:05d}.arrow``.
A msgpack sidecar (``.scatter_meta``) maps ``target_shard -> num_files`` plus
per-shard ``max_chunk_rows`` and a global ``avg_item_bytes`` estimate.  Reducers
read sidecars, reconstruct the file list, and scan all chunks via
``pl.scan_ipc([paths])``.

Write-side memory is bounded by a byte budget (``_SCATTER_WRITE_BUFFER_BYTES``)
rather than a fixed row count.  When the estimated total bytes across all
shard buffers exceeds the budget, the largest buffer is flushed as a new chunk
file.  This prevents OOM on skewed or large-item workloads where a row-count
limit provides no reliable bound.

Routing columns (``__zephyr_shard__``, ``__zephyr_sort_key__``) are added
in ``_items_to_dataframe`` and stripped before writing to disk.
"""

from __future__ import annotations

import concurrent.futures
import functools
import logging
import os
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from itertools import groupby
from typing import Any

import cloudpickle
import msgspec
import polars as pl
import pyarrow as pa
import xxhash
from iris.env_resources import TaskResources
from rigging.filesystem import open_url, url_to_fs
from rigging.timing import log_time

from zephyr.external_sort import (
    _dataframe_to_items,
    compute_fan_in,
    external_sort_merge,
    polars_unify_schemas,
)
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

_SCATTER_META_SUFFIX = ".scatter_meta"
_SCATTER_CHUNK_SUFFIX = ".parquet"

# Number of parallel sidecar reads each reducer issues when building its
# ScatterReader. Sidecars are small msgpack files (a few KB) and reads are
# GCS GET-bound, so a modest pool keeps latency low without thrashing.
_SIDECAR_READ_CONCURRENCY = 32
# Items sampled on the first flush to establish an avg_item_bytes baseline.
_SCATTER_SAMPLE_SIZE = 100
# Items sampled on each subsequent flush to track item-size drift cheaply.
_SCATTER_ONGOING_SAMPLE_SIZE = 10
# How often (in items written) to re-sample one item's size and update
# the EMA estimate in write(). This is independent of flush-time sampling and
# ensures the estimate tracks drift even when no flush has fired yet.
_ESTIMATE_WRITE_SAMPLE_INTERVAL = 10
# EMA weight given to each new observation. 0.3 converges to a 2x step-change
# in item size within ~3 samples while staying stable under small fluctuations.
_ESTIMATE_EMA_ALPHA = 0.3
# Fraction of total memory budgeted for read-side decompression buffers.
_SCATTER_READ_BUFFER_FRACTION = 0.25

_ZSTD_COMPRESS_LEVEL = 3
# Upper bound on rows per chunk file, used for the external-sort memory estimate.
_SUB_BATCH_SIZE = 1024

# Helper column names injected by _items_to_dataframe and stripped before
# writing to disk.  Both are internal implementation details; user schemas must
# not collide with these names.
_SHARD_COL = "__zephyr_shard__"
_SORT_KEY_COL = "__zephyr_sort_key__"
# Written when item values are not Polars-representable (e.g. frozenset).
# On read, each row's bytes are deserialized via cloudpickle.
_PAYLOAD_COL = "_payload"

# Fraction of cgroup memory allocated to scatter write buffers.
_SCATTER_WRITE_BUFFER_FRACTION = 0.25
# Static fallback used when the cgroup memory limit cannot be determined.
_SCATTER_WRITE_BUFFER_BYTES_FALLBACK = 256 * 1024 * 1024  # 256 MB


def _default_scatter_write_buffer_bytes() -> int:
    """Return the scatter write buffer budget based on the cgroup memory limit.

    Uses 25% of the container memory limit so the budget scales with the
    worker size. Falls back to 256 MB when the limit cannot be read.
    """
    memory = TaskResources.from_environment().memory_bytes
    if memory > 0:
        return int(memory * _SCATTER_WRITE_BUFFER_FRACTION)
    return _SCATTER_WRITE_BUFFER_BYTES_FALLBACK


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
    """``shard-0000.shuffle`` -> ``shard-0000.scatter_meta``."""
    stem, _ = os.path.splitext(data_path)
    return stem + _SCATTER_META_SUFFIX


def _chunk_path(data_path: str, target: int, chunk_idx: int) -> str:
    """Return the path for chunk file ``chunk_idx`` of target shard ``target``."""
    stem, _ = os.path.splitext(data_path)
    return f"{stem}-{target:04d}-{chunk_idx:05d}{_SCATTER_CHUNK_SUFFIX}"


def deterministic_hash(obj: object) -> int:
    """Compute a deterministic hash for an object."""
    s = msgspec.msgpack.encode(obj, order="deterministic")
    return xxhash.xxh3_64_intdigest(s)


def _write_scatter_meta(data_path: str, sidecar: dict) -> None:
    meta_path = _scatter_meta_path(data_path)
    payload = _sidecar_encoder().encode(sidecar)
    with log_time(f"Writing scatter meta for {data_path} to {meta_path}", level=logging.DEBUG):
        with open_url(meta_path, "wb") as f:
            f.write(payload)


@dataclass(frozen=True)
class _SidecarSlice:
    """One reducer's slice of a mapper sidecar.

    A full sidecar is ~hundreds of KB and carries file counts for every
    target shard (tens of thousands on large jobs). A reducer only consumes
    its own shard's count plus two scalars, so the worker extracts just
    those fields and discards the parsed dict before returning. This keeps
    the reducer's resident memory proportional to the number of mappers
    instead of mappers * sidecar size.
    """

    path: str
    num_files: int
    max_chunk_rows: int
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
    num_files_raw = meta.get("shards", {}).get(shard_key)
    if not num_files_raw:
        return None
    max_rows_map = meta.get("max_chunk_rows", {})
    if shard_key not in max_rows_map:
        raise ValueError(f"Sidecar {meta_path} has entry for shard {shard_key} but no max_chunk_rows entry.")
    if "avg_item_bytes" not in meta:
        raise ValueError(f"Sidecar {meta_path} has entry for shard {shard_key} but no avg_item_bytes.")
    return _SidecarSlice(
        path=path,
        num_files=int(num_files_raw),
        max_chunk_rows=int(max_rows_map[shard_key]),
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


def _dtype_contains_list(dtype: pl.DataType) -> bool:
    """True if ``dtype`` is or nests a :class:`pl.List` (including under struct fields)."""
    if isinstance(dtype, pl.List):
        return True
    if isinstance(dtype, pl.Struct):
        return any(_dtype_contains_list(field.dtype) for field in dtype.fields)
    return False


def _schema_contains_list(schema: pl.Schema) -> bool:
    """True if any column type contains a :class:`pl.List` anywhere (including nested fields)."""
    return any(_dtype_contains_list(dt) for dt in schema.values())


# ---------------------------------------------------------------------------
# ScatterReader: built from manifest, fed to Reduce
# ---------------------------------------------------------------------------


class ScatterReader:
    """All scatter chunks for one target shard, across all source files.

    ``_files`` is a list of ``(path, num_files)`` pairs — one entry per source
    shard — where ``num_files`` is the count of ``.arrow`` chunk files that
    source shard wrote for ``_target_shard``.  Chunk paths are derived from
    ``path`` at read time via :func:`_chunk_path`.

    Construct via :meth:`from_sidecars` for production use, or pass fields
    directly for testing.
    """

    def __init__(
        self,
        files: list[tuple[str, int]],
        target_shard: int,
        max_chunk_rows: int,
        avg_item_bytes: float,
    ) -> None:
        self._files = files
        self._target_shard = target_shard
        self.max_chunk_rows = max_chunk_rows
        self.avg_item_bytes = avg_item_bytes

    @classmethod
    def from_sidecars(cls, scatter_paths: list[str], target_shard: int) -> ScatterReader:
        """Build a ScatterReader by reading per-mapper sidecars directly.

        Each reducer reads every mapper's ``.scatter_meta`` sidecar in parallel
        and filters for its own ``target_shard``. No coordinator-written manifest
        is needed, which eliminates a serialization bottleneck when there are
        thousands of mappers.
        """
        files: list[tuple[str, int]] = []
        max_rows = 0
        weighted_bytes = 0.0
        total_chunks_for_avg = 0

        with log_time(
            f"Building ScatterReader for target shard {target_shard} "
            f"from {len(scatter_paths)} sidecars (concurrency={_SIDECAR_READ_CONCURRENCY})"
        ):
            for slice_ in _read_sidecar_slices_parallel(scatter_paths, target_shard):
                files.append((slice_.path, slice_.num_files))
                max_rows = max(max_rows, slice_.max_chunk_rows)
                if slice_.avg_item_bytes > 0:
                    weighted_bytes += slice_.avg_item_bytes * slice_.num_files
                    total_chunks_for_avg += slice_.num_files

        avg_item_bytes = weighted_bytes / total_chunks_for_avg if total_chunks_for_avg > 0 else 0.0

        logger.info(
            "ScatterReader for shard %d: %d source files, %d total chunks, max_chunk_rows=%d, avg_item_bytes=%.1f",
            target_shard,
            len(files),
            sum(n for _, n in files),
            max_rows,
            avg_item_bytes,
        )
        return cls(files=files, target_shard=target_shard, max_chunk_rows=max_rows, avg_item_bytes=avg_item_bytes)

    def get_iterators(self) -> Iterator[pl.LazyFrame]:
        for path, num_files in self._files:
            for i in range(num_files):
                yield pl.scan_parquet(_chunk_path(path, self._target_shard, i))

    @property
    def total_chunks(self) -> int:
        return sum(n for _, n in self._files)

    _MAX_IN_MEMORY_ITERATORS = 10_000

    def needs_external_sort(self, memory_limit: int, memory_fraction: float = 0.33) -> bool:
        """Return True if opening all chunks at once would blow the budget."""
        total_chunks = self.total_chunks
        if total_chunks == 0:
            return False

        # Too many open iterators -- force external sort regardless of memory.
        if total_chunks > self._MAX_IN_MEMORY_ITERATORS:
            logger.info(
                "needs_external_sort: total_chunks=%d > %d -> forced external sort",
                total_chunks,
                self._MAX_IN_MEMORY_ITERATORS,
            )
            return True

        if self.avg_item_bytes <= 0:
            raise ValueError(
                "avg_item_bytes not available in scatter manifest. "
                "Re-run the scatter stage with a version that records avg_item_bytes."
            )
        # Estimate merge memory per open iterator (one scan_ipc LazyFrame per source shard):
        #
        # 1. Decompressed chunk data: max_chunk_rows * avg_item_bytes.
        # 2. One sub-batch of Python objects: _SUB_BATCH_SIZE items with dict overhead.
        _FIXED_OVERHEAD_PER_ITEM = 512
        in_memory_multiplier = (self.avg_item_bytes + _FIXED_OVERHEAD_PER_ITEM) / self.avg_item_bytes
        avg_decompressed = self.max_chunk_rows * self.avg_item_bytes
        sub_batch_mem = _SUB_BATCH_SIZE * self.avg_item_bytes * in_memory_multiplier
        per_iterator = avg_decompressed + sub_batch_mem
        estimated = total_chunks * per_iterator
        budget = memory_limit * memory_fraction
        triggered = estimated > budget
        logger.info(
            "needs_external_sort: %d chunks x %.1f MB/iter "
            "(%.1f MB decompressed + %.1f MB sub-batch [%.1fx]) "
            "= %.1f GB estimated vs %.1f GB budget (%.1f GB * %.2f) -> %s",
            total_chunks,
            per_iterator / 1e6,
            avg_decompressed / 1e6,
            sub_batch_mem / 1e6,
            in_memory_multiplier,
            estimated / 1e9,
            budget / 1e9,
            memory_limit / 1e9,
            memory_fraction,
            triggered,
        )
        return triggered

    def merge_sorted_chunks(
        self,
        key_fn: Callable,
        sort_fn: Callable | None = None,
        external_sort_dir: str | None = None,
    ) -> Iterator[tuple[object, Iterator]]:
        """Merge sorted chunks using k-way merge, yielding (key, items_iterator) groups.

        Each chunk file is assumed to be sorted by key (and optionally by sort_fn within key).
        Performs a k-way merge across all chunks and groups consecutive items with the
        same key.

        Args:
            key_fn: Function to extract grouping key from item.
            sort_fn: Optional secondary sort key. When provided, the merge uses
                (key_fn, sort_fn) for ordering but still groups by key_fn alone.
            external_sort_dir: If set and the shard exceeds the memory budget,
                spill intermediate runs to this directory instead of merging in memory.

        Yields:
            Tuples of (key, iterator_of_items) for each unique key.
        """
        use_external = external_sort_dir is not None and self.needs_external_sort(
            TaskResources.from_environment().memory_bytes
        )

        if use_external:
            memory_limit = TaskResources.from_environment().memory_bytes
            per_iter_bytes = int(self.max_chunk_rows * self.avg_item_bytes)
            fan_in = compute_fan_in(per_iter_bytes, memory_limit)
            logger.info(
                "External sort triggered for shard with %d chunks, fan_in=%d (per_iter≈%dKB), spilling to %s",
                self.total_chunks,
                fan_in,
                per_iter_bytes // 1024,
                external_sort_dir,
            )
            merged = external_sort_merge(
                self.get_iterators(),
                sort_key=_SORT_KEY_COL,
                external_sort_dir=external_sort_dir,
                fan_in=fan_in,
            )
            yield from groupby(merged, key=key_fn)
        else:
            all_frames = list(self.get_iterators())
            if not all_frames:
                return

            requires_cast, unified_schema = polars_unify_schemas(all_frames)
            if requires_cast:
                logger.info("Casting frames to unified schema %s", unified_schema)
                all_frames = [frame.cast(unified_schema) for frame in all_frames]

            logger.info("Merging in memory with %d frames across %d chunks", len(all_frames), self.total_chunks)
            logger.info("Merging in memory with schema %s", unified_schema)

            #merged_lf = pl.concat(all_frames, how="vertical_relaxed").sort(_SORT_KEY_COL)
            merged_lf = pl.merge_sorted(all_frames, key=_SORT_KEY_COL)

            yield from groupby(_dataframe_to_items(merged_lf.collect()), key=key_fn)


# ---------------------------------------------------------------------------
# Combiner / sort helper
# ---------------------------------------------------------------------------


def _dataframe_from_pylist(items: list[Any]) -> pl.DataFrame:
    """Build a Polars DataFrame from Python row objects, with cloudpickle fallback.

    Uses ``pl.from_arrow(pa.Table.from_pylist(...))`` because constructing
    ``pl.DataFrame(items)`` directly was unusably slow. Falls back to a binary
    ``_PAYLOAD_COL`` when Arrow cannot encode the rows or when the schema
    contains lists (Polars bug: https://github.com/pola-rs/polars/issues/27563).
    """
    encoding_failed = False
    df: pl.DataFrame | None = None
    try:
        df = pl.from_arrow(pa.Table.from_pylist(items))
    except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError, AttributeError, TypeError):
        encoding_failed = True

    if not encoding_failed and df is not None and _schema_contains_list(df.schema):
        encoding_failed = True

    if encoding_failed:
        payloads = [cloudpickle.dumps(item) for item in items]
        df = pl.DataFrame({_PAYLOAD_COL: pl.Series(payloads, dtype=pl.Binary)})

    return df


def _items_to_dataframe(
    items: list[Any],
    key_fn: Callable,
    sort_fn: Callable | None,
    num_output_shards: int,
) -> pl.DataFrame:
    """Convert a list of Python items to a DataFrame with routing columns.

    Adds ``_SHARD_COL`` (int32 target shard index) and ``_SORT_KEY_COL``
    (binary msgpack-encoded sort key) to every batch.  These columns are
    consumed by :class:`ScatterWriter` and stripped before writing to disk.

    Items not representable in Polars are cloudpickle-serialized into a
    binary ``_PAYLOAD_COL`` column; they are deserialized transparently on
    read.
    """
    shards: list[int] = []
    sort_keys: list[tuple[object, object | None]] = []
    for item in items:
        key = key_fn(item)
        shards.append(deterministic_hash(key) % num_output_shards)
        sort_value = sort_fn(item) if sort_fn is not None else None
        sort_keys.append(OrderedDict([("key", key), ("sort_value", sort_value)]))

    return _dataframe_from_pylist(items).with_columns(
        [
            pl.Series(_SHARD_COL, shards, dtype=pl.Int32),
            pl.Series(_SORT_KEY_COL, sort_keys),
        ]
    )


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


# ---------------------------------------------------------------------------
# Scatter writer
# ---------------------------------------------------------------------------


class ScatterWriter:
    """Writes scatter chunk files, one per flush, as zstd-compressed Polars IPC.

    DataFrames are routed to target shards via the ``_SHARD_COL`` column, buffered
    per shard as Polars DataFrames, optionally combined, sorted by ``_SORT_KEY_COL``,
    and flushed as individual ``.arrow`` IPC files named by target shard and chunk
    index.  A msgpack sidecar recording chunk counts is written on close.

    Flushing is byte-budget-based: when the estimated total uncompressed bytes
    across all shard buffers exceeds ``buffer_limit_bytes``, the largest buffer
    is flushed.  This bounds peak RSS regardless of item count or output shard
    count.
    """

    def __init__(
        self,
        data_path: str,
        key_fn: Callable,
        num_output_shards: int,
        source_shard: int = 0,
        sort_fn: Callable | None = None,
        combiner_fn: Callable | None = None,
        buffer_limit_bytes: int | None = None,
    ) -> None:
        self._data_path = data_path
        self._key_fn = key_fn
        self._num_output_shards = num_output_shards
        self._source_shard = source_shard
        self._combiner_fn = combiner_fn
        self._buffer_limit_bytes = (
            buffer_limit_bytes if buffer_limit_bytes is not None else _default_scatter_write_buffer_bytes()
        )

        if sort_fn is not None:
            captured_sort_fn = sort_fn

            def _sort_key(item: Any) -> Any:
                return (key_fn(item), captured_sort_fn(item))

            self._sort_key = _sort_key
        else:
            self._sort_key = key_fn

        self._buffers: dict[int, list[pl.DataFrame]] = defaultdict(list)
        self._per_shard_chunk_count: dict[int, int] = defaultdict(int)
        self._per_shard_max_rows: dict[int, int] = defaultdict(int)
        self._avg_item_bytes: float = 0.0
        self._sampled_avg = False
        self._n_chunks_written = 0
        self._mid_write_flushes: int = 0
        self._total_buffer_rows: int = 0
        self._peak_buffer_rows: int = 0
        self._item_bytes_estimate: float = 0.0
        self._first_item_bytes: float = 0.0

        ensure_parent_dir(data_path)

    def _flush(self, target: int, frames: list[pl.DataFrame]) -> None:
        combined = pl.concat(frames, how="diagonal_relaxed")

        if self._combiner_fn is not None:
            # Combiner requires Python callables — handle mixed schemas
            # where some rows are Arrow-typed and some are cloudpickled.
            rows = list(_dataframe_to_items(combined))
            rows = _apply_combiner(rows, self._key_fn, self._combiner_fn)

            sort_keys = []
            for item in rows:
                key = self._key_fn(item)
                sort_value = self._sort_key(item) if self._sort_key is not None else None
                sort_keys.append(OrderedDict([("key", key), ("sort_value", sort_value)]))

            combined = _dataframe_from_pylist(rows).with_columns(pl.Series(_SORT_KEY_COL, sort_keys))

        if len(combined) > 0:
            n = _SCATTER_SAMPLE_SIZE if not self._sampled_avg else _SCATTER_ONGOING_SAMPLE_SIZE
            n = min(len(combined), n)
            observed = combined[:n].estimated_size() / n
            if not self._sampled_avg:
                self._avg_item_bytes = observed
                self._sampled_avg = True
            else:
                self._avg_item_bytes = (1 - _ESTIMATE_EMA_ALPHA) * self._avg_item_bytes + _ESTIMATE_EMA_ALPHA * observed
            self._item_bytes_estimate = self._avg_item_bytes

        # Sort by sort key column; keep it in the written frame so merge_sorted_chunks
        # can sort across chunks using Polars rather than Python-level heapq.
        sorted_df = combined.sort(_SORT_KEY_COL)

        chunk_idx = self._per_shard_chunk_count[target]
        chunk_path = _chunk_path(self._data_path, target, chunk_idx)
        chunk_fs, chunk_fs_path = url_to_fs(chunk_path)
        with chunk_fs.open(chunk_fs_path, "wb") as f:
            sorted_df.write_parquet(f, compression="zstd")

        self._per_shard_chunk_count[target] += 1
        self._per_shard_max_rows[target] = max(self._per_shard_max_rows[target], len(sorted_df))

        self._n_chunks_written += 1
        if self._n_chunks_written % 10 == 0:
            logger.info(
                "[shard %d] Wrote %d scatter chunks so far (latest chunk size: %d items)",
                self._source_shard,
                self._n_chunks_written,
                len(sorted_df),
            )

    def write(self, df: pl.DataFrame) -> None:
        """Route a DataFrame to per-shard buffers, flushing when over budget.

        The DataFrame must contain ``_SHARD_COL`` (int32) and ``_SORT_KEY_COL``
        (binary) columns produced by ``_items_to_dataframe``.
        """
        if len(df) == 0:
            return

        # Update item-size estimate from Polars estimated size. This runs on every
        # write() call so the estimate tracks size drift between flush cycles.
        observed = df.estimated_size() / len(df)
        if self._total_buffer_rows == 0:
            self._item_bytes_estimate = observed
            self._first_item_bytes = observed
        else:
            self._item_bytes_estimate = (
                1 - _ESTIMATE_EMA_ALPHA
            ) * self._item_bytes_estimate + _ESTIMATE_EMA_ALPHA * observed

        # Partition by shard in a single pass.
        for (shard_val,), partition in df.partition_by(_SHARD_COL, as_dict=True).items():
            self._buffers[shard_val].append(partition.drop(_SHARD_COL))

        self._total_buffer_rows += len(df)
        if self._total_buffer_rows > self._peak_buffer_rows:
            self._peak_buffer_rows = self._total_buffer_rows

        if self._total_buffer_rows * self._item_bytes_estimate > self._buffer_limit_bytes:
            largest = max(self._buffers, key=lambda t: sum(len(tbl) for tbl in self._buffers[t]))
            rows_flushed = sum(len(tbl) for tbl in self._buffers[largest])
            self._flush(largest, self._buffers[largest])
            del self._buffers[largest]
            self._total_buffer_rows -= rows_flushed
            self._mid_write_flushes += 1

    def close(self) -> ListShard:
        """Flush remaining buffers, write sidecar, return ListShard."""
        close_flushes = 0
        with log_time(f"Flushing remaining buffers for {self._data_path}"):
            for target, frames in sorted(self._buffers.items()):
                if frames:
                    self._flush(target, frames)
                    close_flushes += 1

        measured_avg = self._avg_item_bytes if self._sampled_avg else self._item_bytes_estimate
        logger.info(
            "[shard %d] scatter write done: %d mid-write flushes + %d at close = %d total; "
            "first-item estimate=%.0f B, measured avg=%.0f B (%.1fx), "
            "peak buffered=%d rows, budget=%d MB",
            self._source_shard,
            self._mid_write_flushes,
            close_flushes,
            self._mid_write_flushes + close_flushes,
            self._first_item_bytes,
            measured_avg,
            measured_avg / self._first_item_bytes if self._first_item_bytes > 0 else 0.0,
            self._peak_buffer_rows,
            self._buffer_limit_bytes // (1024 * 1024),
        )

        sidecar: dict = {
            "shards": {str(k): v for k, v in self._per_shard_chunk_count.items()},
            "max_chunk_rows": {str(k): v for k, v in self._per_shard_max_rows.items() if v > 0},
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
    buffer_limit_bytes: int | None = None,
) -> ListShard:
    """Route items to target shards, buffer, sort, and flush as Polars IPC chunk files.

    Items are batched into DataFrames of up to ``_SUB_BATCH_SIZE`` rows.
    Routing and sort keys are computed here (in Python, since ``key_fn`` and
    ``sort_fn`` are arbitrary callables) and embedded as helper columns before
    passing the DataFrame to :meth:`ScatterWriter.write`.

    Writes one binary data file plus one ``.scatter_meta`` sidecar.

    Returns:
        A ListShard wrapping the data file path (as the existing scatter
        plumbing expects a list of paths).
    """
    writer = ScatterWriter(
        data_path=data_path,
        key_fn=key_fn,
        num_output_shards=num_output_shards,
        source_shard=source_shard,
        sort_fn=sort_fn,
        combiner_fn=combiner_fn,
        buffer_limit_bytes=buffer_limit_bytes,
    )
    pending: list[Any] = []
    for item in items:
        pending.append(item)
        if len(pending) >= _SUB_BATCH_SIZE:
            writer.write(_items_to_dataframe(pending, key_fn, sort_fn, num_output_shards))
            pending.clear()
    if pending:
        writer.write(_items_to_dataframe(pending, key_fn, sort_fn, num_output_shards))
    return writer.close()
