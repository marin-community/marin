# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scatter/shuffle support for Zephyr pipelines.

Each source-shard's scatter output is a single binary file containing a
sequence of Arrow IPC stream frames, each compressed with zstd at the
buffer level (via ``pa.Codec("zstd")``).  Within one chunk's IPC frame,
items are written in batches of up to ``_SUB_BATCH_SIZE`` rows.

A msgpack sidecar (``.scatter_meta``) maps ``target_shard -> [(offset, length)]``
byte ranges into the data file, plus per-shard ``max_chunk_rows`` and a global
``avg_item_bytes`` estimate (in Arrow uncompressed buffer bytes).  Sidecars
from all source shards are aggregated into a single ``scatter_metadata``
manifest at the end of the scatter stage, which reducers consume to build
:class:`ScatterReader` instances.

On read, each chunk is fetched with a single ``cat_file`` range GET, wrapped
in a ``pa.BufferReader``, and streamed via ``pa.ipc.open_stream``.  The
reader yields Python dicts for backward compatibility with existing consumers
(``_merge_sorted_chunks``, ``external_sort``).  Items whose values are not
Arrow-representable were written via a cloudpickle fallback column
(``_payload: binary``) and are deserialized on read.

Write-side memory is bounded by a byte budget (``_SCATTER_WRITE_BUFFER_BYTES``)
rather than a fixed row count.  When the estimated total bytes across all
shard buffers exceeds the budget, the largest buffer is flushed.  This prevents
OOM on skewed or large-item workloads where a row-count limit provides no
reliable bound.

Routing columns (``__zephyr_shard__``, ``__zephyr_sort_key__``) are added
in ``_items_to_record_batch`` and stripped before writing to disk.
"""

from __future__ import annotations

import concurrent.futures
import functools
import logging
import os
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from itertools import groupby
from typing import Any

import cloudpickle
import msgspec
import pyarrow as pa
import pyarrow.compute as pc
import xxhash
from iris.env_resources import TaskResources
from rigging.filesystem import open_url, url_to_fs
from rigging.timing import log_time

from zephyr.external_sort import compute_fan_in, external_sort_merge, in_memory_k_way_merge
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
_SCATTER_DATA_SUFFIX = ".shuffle"

# Number of parallel sidecar reads each reducer issues when building its
# ScatterReader. Sidecars are small msgpack files (a few KB) and reads are
# GCS GET-bound, so a modest pool keeps latency low without thrashing.
_SIDECAR_READ_CONCURRENCY = 32
# Items sampled on the first flush to establish an avg_item_bytes baseline.
_SCATTER_SAMPLE_SIZE = 100
# Items sampled on each subsequent flush to track item-size drift cheaply.
_SCATTER_ONGOING_SAMPLE_SIZE = 10
# How often (in items written) to re-sample one item's pickle size and update
# the EMA estimate in write(). This is independent of flush-time sampling and
# ensures the estimate tracks drift even when no flush has fired yet.
_ESTIMATE_WRITE_SAMPLE_INTERVAL = 10
# EMA weight given to each new observation. 0.3 converges to a 2x step-change
# in item size within ~3 samples while staying stable under small fluctuations.
_ESTIMATE_EMA_ALPHA = 0.3
# Fraction of total memory budgeted for read-side decompression buffers.
_SCATTER_READ_BUFFER_FRACTION = 0.25

_ZSTD_COMPRESS_LEVEL = 3
# Rows per IPC RecordBatch written inside one chunk frame.
_SUB_BATCH_SIZE = 1024

# Helper column names injected by _items_to_record_batch and stripped before
# writing to disk.  Both are internal implementation details; user schemas must
# not collide with these names.
_SHARD_COL = "__zephyr_shard__"
_SORT_KEY_COL = "__zephyr_sort_key__"
# Written when item values are not Arrow-representable (e.g. frozenset).
# On read, each row's bytes are deserialized via cloudpickle.
_PAYLOAD_COL = "_payload"

# Reused across all writes to avoid re-allocating the C++ codec object.
_IPC_CODEC = pa.Codec("zstd", compression_level=_ZSTD_COMPRESS_LEVEL)
_IPC_WRITE_OPTIONS = pa.ipc.IpcWriteOptions(compression=_IPC_CODEC)

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

    A full sidecar is ~hundreds of KB and carries byte ranges for every
    target shard (tens of thousands on large jobs). A reducer only consumes
    its own shard's ranges plus two scalars, so the worker extracts just
    those fields and discards the parsed dict before returning. This keeps
    the reducer's resident memory proportional to the number of mappers
    instead of mappers * sidecar size.
    """

    path: str
    ranges: tuple[tuple[int, int], ...]
    max_chunk_rows: int
    avg_item_bytes: float


def _read_sidecar_slice(path: str, shard_key: str) -> _SidecarSlice | None:
    """Read one sidecar and extract only the fields for ``shard_key``.

    Returns ``None`` if the sidecar has no ranges for this shard. The parsed
    dict is released when this function returns. Once we confirm this shard
    has ranges, ``max_chunk_rows[shard_key]`` and ``avg_item_bytes`` must
    also be present — ``ScatterWriter`` records both in the same ``_flush``
    that appends to ``shards[shard_key]``. A missing field here means the
    sidecar is corrupt or was written by an incompatible version, and we
    fail rather than silently substituting zero.

    Uses ``fs.cat_file`` rather than ``open_url`` — one direct GET returning
    bytes is ~25% faster than going through ``TextIOWrapper(BufferedFile)``
    for small sidecars, and msgpack decodes bytes directly.
    """
    meta_path = _scatter_meta_path(path)
    fs, fs_path = url_to_fs(meta_path)
    meta = _sidecar_decoder().decode(fs.cat_file(fs_path))
    ranges_raw = meta.get("shards", {}).get(shard_key)
    if not ranges_raw:
        return None
    max_rows_map = meta.get("max_chunk_rows", {})
    if shard_key not in max_rows_map:
        raise ValueError(f"Sidecar {meta_path} has ranges for shard {shard_key} but no max_chunk_rows entry.")
    if "avg_item_bytes" not in meta:
        raise ValueError(f"Sidecar {meta_path} has ranges for shard {shard_key} but no avg_item_bytes.")
    ranges = tuple((int(off), int(length)) for off, length in ranges_raw)
    return _SidecarSlice(
        path=path,
        ranges=ranges,
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


def _batch_to_items(batch: pa.RecordBatch | pa.Table) -> Iterator:
    """Convert a single RecordBatch or Table to Python items, handling mixed schemas."""
    sort_key_idx = batch.schema.get_field_index(_SORT_KEY_COL)
    if sort_key_idx >= 0:
        batch = batch.remove_column(sort_key_idx)

    if _PAYLOAD_COL in batch.schema.names:
        payload_col = batch.column(_PAYLOAD_COL)
        if payload_col.null_count > 0:
            # Mixed batch: some rows are Arrow-typed, some are cloudpickled.
            # payload_col has None for Arrow-typed rows.
            payload_list = payload_col.to_pylist()
            data_batch = batch.remove_column(batch.schema.get_field_index(_PAYLOAD_COL))
            data_rows = data_batch.to_pylist()
            for p, row in zip(payload_list, data_rows, strict=True):
                if p is not None:
                    yield cloudpickle.loads(p)
                else:
                    yield row
        else:
            # Pure payload batch: all rows are cloudpickled.
            for p in payload_col.to_pylist():
                yield cloudpickle.loads(p)
    else:
        # Pure Arrow batch.
        yield from batch.to_pylist()


def _batches_to_items(batches: Iterable[pa.RecordBatch]) -> Iterator:
    """Convert RecordBatches to Python items, stripping internal columns.

    Strips ``_SORT_KEY_COL`` (present on disk for merge-sort) and deserializes
    the cloudpickle payload column when present. handles mixed schemas where
    some rows are Arrow-typed and some are cloudpickled.
    """
    for batch in batches:
        yield from _batch_to_items(batch)


# ---------------------------------------------------------------------------
# ScatterReader: built from manifest, fed to Reduce
# ---------------------------------------------------------------------------


class ScatterReader:
    """All scatter chunks for one target shard, across all source files.

    ``_files`` is a list of ``(path, chunks)`` pairs where ``chunks`` is a
    tuple of ``(offset, length)`` byte ranges within that file.

    Construct via :meth:`from_sidecars` for production use, or pass fields
    directly for testing.
    """

    def __init__(
        self,
        files: list[tuple[str, tuple[tuple[int, int], ...]]],
        max_chunk_rows: int,
        avg_item_bytes: float,
    ) -> None:
        self._files = files
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
        files: list[tuple[str, tuple[tuple[int, int], ...]]] = []
        max_rows = 0
        weighted_bytes = 0.0
        total_chunks_for_avg = 0

        with log_time(
            f"Building ScatterReader for target shard {target_shard} "
            f"from {len(scatter_paths)} sidecars (concurrency={_SIDECAR_READ_CONCURRENCY})"
        ):
            for slice_ in _read_sidecar_slices_parallel(scatter_paths, target_shard):
                files.append((slice_.path, slice_.ranges))
                max_rows = max(max_rows, slice_.max_chunk_rows)
                if slice_.avg_item_bytes > 0:
                    count = len(slice_.ranges)
                    weighted_bytes += slice_.avg_item_bytes * count
                    total_chunks_for_avg += count

        avg_item_bytes = weighted_bytes / total_chunks_for_avg if total_chunks_for_avg > 0 else 0.0

        logger.info(
            "ScatterReader for shard %d: %d files, %d total chunks, max_chunk_rows=%d, avg_item_bytes=%.1f",
            target_shard,
            len(files),
            sum(len(chunks) for _, chunks in files),
            max_rows,
            avg_item_bytes,
        )
        return cls(files=files, max_chunk_rows=max_rows, avg_item_bytes=avg_item_bytes)

    def get_iterators(self) -> Iterator[pa.RecordBatch]:
        for path, chunks in self._files:
            fs, fs_path = url_to_fs(path)
            for offset, length in chunks:
                blob = fs.cat_file(fs_path, start=offset, end=offset + length)
                with pa.ipc.open_stream(pa.BufferReader(blob)) as reader:
                    yield from reader

    @property
    def total_chunks(self) -> int:
        return sum(len(chunks) for _, chunks in self._files)

    @property
    def max_compressed_chunk_bytes(self) -> int:
        """Return the largest compressed chunk length across all files."""
        if not self._files:
            return 0
        return max(length for _, chunks in self._files for _, length in chunks)

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
        # Estimate merge memory per open iterator:
        #
        # 1. Compressed chunk blob: fetched via range GET, held in a BytesIO.
        # 2. Decompressed frame: zstd stream_reader decompresses the full
        #    frame into an internal buffer (~max_chunk_rows * avg_item_bytes).
        # 3. One unpickled sub-batch: _SUB_BATCH_SIZE Python objects in memory.
        #    Python object overhead is fixed per item (~500 bytes for a dict:
        #    object header, hash table, key/value strings) so the multiplier
        #    scales inversely with item size.
        _FIXED_OVERHEAD_PER_ITEM = 512
        in_memory_multiplier = (self.avg_item_bytes + _FIXED_OVERHEAD_PER_ITEM) / self.avg_item_bytes
        total_compressed = sum(length for _, chunks in self._files for _, length in chunks)
        avg_compressed = total_compressed / total_chunks
        avg_decompressed = self.max_chunk_rows * self.avg_item_bytes
        sub_batch_mem = _SUB_BATCH_SIZE * self.avg_item_bytes * in_memory_multiplier
        per_iterator = avg_compressed + avg_decompressed + sub_batch_mem
        estimated = total_chunks * per_iterator
        budget = memory_limit * memory_fraction
        triggered = estimated > budget
        logger.info(
            "needs_external_sort: %d chunks x %.1f MB/iter "
            "(%.1f MB compressed + %.1f MB decompressed + %.1f MB sub-batch [%.1fx]) "
            "= %.1f GB estimated vs %.1f GB budget (%.1f GB * %.2f) -> %s",
            total_chunks,
            per_iterator / 1e6,
            avg_compressed / 1e6,
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

        Each chunk is assumed to be sorted by key (and optionally by sort_fn within key).
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
            per_iter_bytes = self.max_compressed_chunk_bytes
            fan_in = compute_fan_in(per_iter_bytes, memory_limit)
            logger.info(
                "External sort triggered for shard with %d chunks, " "fan_in=%d (per_iter≈%dKB), spilling to %s",
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
            all_batches = list(self.get_iterators())
            logger.info("Merging %d batches across %d chunks", len(all_batches), self.total_chunks)
            if len(all_batches) == 0:
                return
            merged = in_memory_k_way_merge(all_batches, key=_SORT_KEY_COL)
            yield from groupby(_batches_to_items(merged), key=key_fn)


# ---------------------------------------------------------------------------
# Combiner / sort helper
# ---------------------------------------------------------------------------


def _items_to_record_batch(
    items: list[Any],
    key_fn: Callable,
    sort_fn: Callable | None,
    num_output_shards: int,
) -> pa.RecordBatch:
    """Convert a list of Python items to a RecordBatch with routing columns.

    Adds ``_SHARD_COL`` (int32 target shard index) and ``_SORT_KEY_COL``
    (binary msgpack-encoded sort key) to every batch.  These columns are
    consumed by :class:`ScatterWriter` and stripped before writing to disk.

    Items not representable in Arrow are cloudpickle-serialized into a
    binary ``_PAYLOAD_COL`` column; they are deserialized transparently on
    read.
    """
    shards: list[int] = []
    sort_keys: list[bytes] = []
    for item in items:
        key = key_fn(item)
        shards.append(deterministic_hash(key) % num_output_shards)
        sort_key_obj = (key, sort_fn(item)) if sort_fn is not None else key
        sort_keys.append(msgspec.msgpack.encode(sort_key_obj, order="deterministic"))

    try:
        table = pa.Table.from_pylist(items)
    except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError, AttributeError, TypeError):
        payloads = [cloudpickle.dumps(item) for item in items]
        table = pa.table({_PAYLOAD_COL: pa.array(payloads, type=pa.binary())})

    table = table.append_column(_SHARD_COL, pa.array(shards, type=pa.int32()))
    table = table.append_column(_SORT_KEY_COL, pa.array(sort_keys, type=pa.binary()))
    # from_pylist always produces a single-batch table for ≤ _SUB_BATCH_SIZE rows.
    return table.to_batches()[0]


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
    """Writes RecordBatches to a scatter data file as zstd-compressed Arrow IPC.

    Batches are routed to target shards via the ``_SHARD_COL`` column, buffered
    per shard as Arrow Tables, optionally combined, sorted by ``_SORT_KEY_COL``,
    and flushed as Arrow IPC stream frames.  A msgpack sidecar is written on
    close.

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

        self._buffers: dict[int, list[pa.Table]] = defaultdict(list)
        self._shard_ranges: dict[int, list[tuple[int, int]]] = defaultdict(list)
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
        fs, fs_path = url_to_fs(data_path)
        self._out = fs.open(fs_path, "wb")

    def _flush(self, target: int, tables: list[pa.Table]) -> None:
        combined = pa.concat_tables(tables, promote_options="permissive")

        if self._combiner_fn is not None:
            # Combiner requires Python callables — handle mixed schemas
            # where some rows are Arrow-typed and some are cloudpickled.
            rows = list(_batch_to_items(combined))
            rows = _apply_combiner(rows, self._key_fn, self._combiner_fn)
            sort_keys_bytes = [msgspec.msgpack.encode(self._sort_key(row), order="deterministic") for row in rows]
            try:
                combined = pa.Table.from_pylist(rows)
            except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError):
                payloads = [cloudpickle.dumps(r) for r in rows]
                combined = pa.table({_PAYLOAD_COL: pa.array(payloads, type=pa.binary())})
            combined = combined.append_column(_SORT_KEY_COL, pa.array(sort_keys_bytes, type=pa.binary()))

        if len(combined) > 0:
            n = _SCATTER_SAMPLE_SIZE if not self._sampled_avg else _SCATTER_ONGOING_SAMPLE_SIZE
            n = min(len(combined), n)
            observed = combined.slice(0, n).get_total_buffer_size() / n
            if not self._sampled_avg:
                self._avg_item_bytes = observed
                self._sampled_avg = True
            else:
                self._avg_item_bytes = (1 - _ESTIMATE_EMA_ALPHA) * self._avg_item_bytes + _ESTIMATE_EMA_ALPHA * observed
            self._item_bytes_estimate = self._avg_item_bytes

        # Sort by sort key column; keep it in the written frame so merge_sorted_chunks
        # can sort across chunks using Arrow rather than Python-level heapq.
        sort_indices = pc.sort_indices(combined, sort_keys=[(_SORT_KEY_COL, "ascending")])
        sorted_table = combined.take(sort_indices)

        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, sorted_table.schema, options=_IPC_WRITE_OPTIONS) as ipc_writer:
            for batch in sorted_table.to_batches(max_chunksize=_SUB_BATCH_SIZE):
                ipc_writer.write_batch(batch)

        frame = sink.getvalue().to_pybytes()
        offset = self._out.tell()
        self._out.write(frame)
        self._shard_ranges[target].append((offset, len(frame)))
        self._per_shard_max_rows[target] = max(self._per_shard_max_rows[target], len(sorted_table))

        self._n_chunks_written += 1
        if self._n_chunks_written % 10 == 0:
            logger.info(
                "[shard %d] Wrote %d scatter chunks so far (latest chunk size: %d items, %d bytes)",
                self._source_shard,
                self._n_chunks_written,
                len(sorted_table),
                len(frame),
            )

    def write(self, batch: pa.RecordBatch) -> None:
        """Route a RecordBatch to per-shard buffers, flushing when over budget.

        The batch must contain ``_SHARD_COL`` (int32) and ``_SORT_KEY_COL``
        (binary) columns produced by ``_items_to_record_batch``.
        """
        if batch.num_rows == 0:
            return

        # Update item-size estimate from Arrow buffer size.  This runs on every
        # write() call so the estimate tracks size drift between flush cycles —
        # the same role pickle.dumps sampling played in the old implementation.
        observed = batch.get_total_buffer_size() / batch.num_rows
        if self._total_buffer_rows == 0:
            self._item_bytes_estimate = observed
            self._first_item_bytes = observed
        else:
            self._item_bytes_estimate = (
                1 - _ESTIMATE_EMA_ALPHA
            ) * self._item_bytes_estimate + _ESTIMATE_EMA_ALPHA * observed

        # Partition batch by shard using PyArrow compute to preserve the schema
        # exactly (going via Polars would coerce string → large_string).
        shard_col = batch.column(_SHARD_COL)
        shard_col_idx = batch.schema.get_field_index(_SHARD_COL)
        for shard_val in shard_col.unique():
            mask = pc.equal(shard_col, shard_val)
            partition = pa.Table.from_batches([batch.filter(mask)]).remove_column(shard_col_idx)
            self._buffers[shard_val.as_py()].append(partition)

        self._total_buffer_rows += batch.num_rows
        if self._total_buffer_rows > self._peak_buffer_rows:
            self._peak_buffer_rows = self._total_buffer_rows

        if self._total_buffer_rows * self._item_bytes_estimate > self._buffer_limit_bytes:
            largest = max(self._buffers, key=lambda t: sum(tbl.num_rows for tbl in self._buffers[t]))
            rows_flushed = sum(tbl.num_rows for tbl in self._buffers[largest])
            self._flush(largest, self._buffers[largest])
            del self._buffers[largest]
            self._total_buffer_rows -= rows_flushed
            self._mid_write_flushes += 1

    def close(self) -> ListShard:
        """Flush remaining buffers, write sidecar, return ListShard."""
        close_flushes = 0
        with log_time(f"Flushing remaining buffers for {self._data_path}"):
            for target, tables in sorted(self._buffers.items()):
                if tables:
                    self._flush(target, tables)
                    close_flushes += 1
        self._out.close()

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
            "shards": {str(k): v for k, v in self._shard_ranges.items()},
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
    """Route items to target shards, buffer, sort, and flush as Arrow IPC chunks.

    Items are batched into RecordBatches of up to ``_SUB_BATCH_SIZE`` rows.
    Routing and sort keys are computed here (in Python, since ``key_fn`` and
    ``sort_fn`` are arbitrary callables) and embedded as helper columns before
    passing the batch to :meth:`ScatterWriter.write`.

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
            writer.write(_items_to_record_batch(pending, key_fn, sort_fn, num_output_shards))
            pending.clear()
    if pending:
        writer.write(_items_to_record_batch(pending, key_fn, sort_fn, num_output_shards))
    return writer.close()
