# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scatter/shuffle Parquet support for Zephyr pipelines.

Handles the full scatter pipeline: hash-routing items to target shards,
buffering per-shard, applying an optional combiner, sorting each buffer, and
writing sorted chunks as Parquet row groups with envelope wrapping.

Also provides ScatterShard, which reads back scatter data for the reduce stage.
"""

from __future__ import annotations

import concurrent.futures
import functools
import json
import logging
import os
import pickle
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import cloudpickle
import fsspec
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pad
import pyarrow.parquet as pq
from iris.env_resources import TaskResources as _TaskResources
from rigging.filesystem import open_url, url_to_fs
from rigging.timing import log_time

from zephyr.plan import deterministic_hash
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
# Column names and constants
# ---------------------------------------------------------------------------

_ZEPHYR_SHUFFLE_SHARD_IDX_COL = "shard_idx"
_ZEPHYR_SHUFFLE_CHUNK_IDX_COL = "chunk_idx"
_ZEPHYR_SHUFFLE_ITEM_COL = "item"
_ZEPHYR_SHUFFLE_PICKLED_COL = "pickled"
_SCATTER_META_SUFFIX = ".scatter_meta"
_SCATTER_MANIFEST_NAME = "scatter_metadata"

_ZEPHYR_SHUFFLE_SORT_KEY_COL = "_sort_key"
_ZEPHYR_SHUFFLE_SORT_SECONDARY_COL = "_sort_secondary"

_SCATTER_META_READ_CONCURRENCY = 256
# Number of items sampled from the first flush to estimate avg_item_bytes at scatter-write time
_SCATTER_SAMPLE_SIZE = 100
# Conservative item-bytes fallback when avg_item_bytes is not in the manifest
_ITEM_BYTES_FALLBACK = 500.0
# Fraction of total memory limit to budget for scatter read buffers
_SCATTER_READ_BUFFER_FRACTION = 0.25
# Number of rows to accumulate in Python before converting to an Arrow micro-batch
_SCATTER_MICRO_BATCH_SIZE = 64
# Byte threshold per shard buffer before flushing a sorted batch to the Parquet writer
_SCATTER_ROW_GROUP_BYTES = 2 * 1024 * 1024  # 2 MB


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


@functools.cache
def _get_scatter_read_fs(num_files: int, sample_path: str, memory_fraction: float = 0.05) -> pa.fs.FileSystem:
    """Return a pyarrow filesystem with per-file block_size budgeted from available memory.

    Caps total fsspec buffer memory at ``memory_fraction`` of the worker's RAM,
    split evenly across ``num_files``.  Falls back to a default fsspec-backed
    filesystem when the budget is large enough or ``block_size`` is not supported.
    """
    fs, _ = fsspec.core.url_to_fs(sample_path)
    default_fs = pa.fs.PyFileSystem(pa.fs.FSSpecHandler(fs))

    if num_files <= 0:
        return default_fs

    total_mem = _TaskResources.from_environment().memory_bytes
    budget = int(total_mem * memory_fraction)
    per_file = max(budget // num_files, 64 * 1024)  # floor at 64 KB

    # Only override when we would meaningfully reduce the default (~5 MB).
    if per_file >= 5 * 1024 * 1024:
        return default_fs

    if not hasattr(fs, "blocksize"):
        return default_fs

    # Recreate the filesystem with the budgeted block_size.
    fsspec_fs = type(fs)(block_size=per_file, **{k: v for k, v in fs.storage_options.items() if k != "block_size"})
    logger.info(
        "Scatter read: %d files, per-file block_size=%d KB (total budget=%.1f GB)",
        num_files,
        per_file // 1024,
        budget / 1024**3,
    )
    return pa.fs.PyFileSystem(pa.fs.FSSpecHandler(fsspec_fs))


# ---------------------------------------------------------------------------
# ScatterParquetIterator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScatterParquetIterator:
    """Reference to sorted chunks for one target shard in one Parquet file.

    Creates a ``pyarrow.dataset`` once (caching file metadata) and yields
    lazy per-chunk iterators via Scanner with predicate pushdown.
    """

    path: str
    shard_idx: int
    chunk_count: int
    is_pickled: bool
    filesystem: pa.fs.FileSystem
    has_sort_key: bool = False

    def __iter__(self) -> Iterator:
        for chunk_iter in self.get_chunk_iterators():
            yield from chunk_iter

    def get_chunk_iterators(self, batch_size: int = 1024) -> Iterator[Iterator]:
        """Yield one lazy iterator per sorted chunk.

        Opens the file once via ``pyarrow.dataset`` and creates a Scanner
        per chunk with predicate pushdown on ``(shard_idx, chunk_idx)``.
        """
        _, fs_path = url_to_fs(self.path)
        dataset: pad.FileSystemDataset = pad.dataset(fs_path, format="parquet", filesystem=self.filesystem)
        col = _ZEPHYR_SHUFFLE_PICKLED_COL if self.is_pickled else _ZEPHYR_SHUFFLE_ITEM_COL

        for chunk_idx in range(self.chunk_count):
            scanner = dataset.scanner(
                columns=[col],
                filter=(
                    (pc.field(_ZEPHYR_SHUFFLE_SHARD_IDX_COL) == self.shard_idx)
                    & (pc.field(_ZEPHYR_SHUFFLE_CHUNK_IDX_COL) == chunk_idx)
                ),
                batch_size=batch_size,
                use_threads=False,
            )
            yield self._iter_scanner(scanner, col)

    def get_chunk_tables(self, batch_size: int = 1024) -> Iterator[pa.Table]:
        """Yield Arrow tables per sorted chunk (no Python materialization)."""
        _, fs_path = url_to_fs(self.path)
        dataset: pad.FileSystemDataset = pad.dataset(fs_path, format="parquet", filesystem=self.filesystem)
        item_col = _ZEPHYR_SHUFFLE_PICKLED_COL if self.is_pickled else _ZEPHYR_SHUFFLE_ITEM_COL
        columns = [item_col, _ZEPHYR_SHUFFLE_SORT_KEY_COL]
        if _ZEPHYR_SHUFFLE_SORT_SECONDARY_COL in dataset.schema.names:
            columns.append(_ZEPHYR_SHUFFLE_SORT_SECONDARY_COL)

        for chunk_idx in range(self.chunk_count):
            scanner = dataset.scanner(
                columns=columns,
                filter=(
                    (pc.field(_ZEPHYR_SHUFFLE_SHARD_IDX_COL) == self.shard_idx)
                    & (pc.field(_ZEPHYR_SHUFFLE_CHUNK_IDX_COL) == chunk_idx)
                ),
                batch_size=batch_size,
                use_threads=False,
            )
            batches = list(scanner.to_batches())
            if batches:
                yield pa.Table.from_batches(batches)

    def _iter_scanner(self, scanner: pad.Scanner, col: str) -> Iterator:
        for batch in scanner.to_batches():
            items = batch.column(col).to_pylist()
            if self.is_pickled:
                yield from (pickle.loads(b) for b in items)
            else:
                yield from items


# ---------------------------------------------------------------------------
# ScatterShard
# ---------------------------------------------------------------------------


@dataclass
class ScatterShard:
    """Shard backed by scatter Parquet files for one target shard.

    Each ``iterator`` is a ScatterParquetIterator pointing to sorted chunks
    in a single Parquet file. ``get_iterators`` yields per-sorted-chunk
    iterators across all files for the k-way merge in Reduce.
    """

    iterators: list[ScatterParquetIterator]
    max_row_group_rows: int = 100_000  # conservative default = chunk_size
    avg_item_bytes: float = 0.0  # 0.0 = unknown, will probe on demand
    has_sort_key: bool = False

    def __iter__(self) -> Iterator:
        for it in self.iterators:
            yield from it

    def get_iterators(self) -> Iterator[Iterator]:
        batch_size = self._compute_batch_size()
        for it in self.iterators:
            yield from it.get_chunk_iterators(batch_size=batch_size)

    def needs_external_sort(self, memory_limit: int, memory_fraction: float = 0.5) -> bool:
        """Return True if opening all chunk iterators simultaneously would exceed memory_fraction of memory_limit."""
        total_chunks = sum(it.chunk_count for it in self.iterators)
        if total_chunks == 0 or memory_limit <= 0:
            return False
        item_bytes = self.avg_item_bytes if self.avg_item_bytes > 0 else _ITEM_BYTES_FALLBACK
        estimated = total_chunks * self.max_row_group_rows * item_bytes
        return estimated > memory_limit * memory_fraction

    def _compute_batch_size(self) -> int:
        """Compute a safe batch_size that keeps scatter read buffers within budget.

        With N total chunk iterators in the k-way merge, each holding one
        batch of batch_size deserialized items in memory simultaneously,
        the total buffer footprint is N * batch_size * bytes_per_item.
        We cap this at _SCATTER_READ_BUFFER_FRACTION of the worker's memory limit.
        """
        total_chunks = sum(it.chunk_count for it in self.iterators)
        if total_chunks == 0:
            return 1024
        bytes_per_item = self.avg_item_bytes if self.avg_item_bytes > 0 else _ITEM_BYTES_FALLBACK
        memory_limit = _TaskResources.from_environment().memory_bytes
        buffer_budget = int(memory_limit * _SCATTER_READ_BUFFER_FRACTION)
        safe = max(1, int(buffer_budget // (total_chunks * bytes_per_item)))
        safe = min(safe, 8192)
        logger.info(
            "ScatterShard batch_size=%d (total_chunks=%d, bytes_per_item=%.0f, buffer_budget=%dMB, memory_limit=%dMB)",
            safe,
            total_chunks,
            bytes_per_item,
            buffer_budget // (1024 * 1024),
            memory_limit // (1024 * 1024),
        )
        return safe


# ---------------------------------------------------------------------------
# Scatter write helpers
# ---------------------------------------------------------------------------


def _scatter_meta_path(parquet_path: str) -> str:
    """Return the sidecar metadata path for a scatter Parquet file.

    Replaces the ``.parquet`` extension: ``shard-0000-seg0000.parquet`` →
    ``shard-0000-seg0000.scatter_meta``.
    """
    stem, _ = os.path.splitext(parquet_path)
    return stem + _SCATTER_META_SUFFIX


def _write_scatter_meta(
    parquet_path: str,
    chunk_counts: dict[int, int],
    is_pickled: bool,
    max_chunk_rows: dict[int, int] | None = None,
    avg_item_bytes: float = 0.0,
    has_sort_key: bool = False,
) -> None:
    """Write a ``.scatter_meta`` sidecar alongside a scatter Parquet file."""
    meta_path = _scatter_meta_path(parquet_path)
    payload_dict: dict = {
        "chunk_counts": {str(k): v for k, v in chunk_counts.items()},
    }
    if is_pickled:
        payload_dict["is_pickled"] = True
    if max_chunk_rows:
        payload_dict["max_chunk_rows"] = {str(k): v for k, v in max_chunk_rows.items() if v > 0}
    if avg_item_bytes > 0:
        payload_dict["avg_item_bytes"] = round(avg_item_bytes, 1)
    if has_sort_key:
        payload_dict["has_sort_key"] = True
    payload = json.dumps(payload_dict)
    with log_time(f"Writing scatter meta for {parquet_path} to {meta_path}", level=logging.DEBUG):
        with open_url(meta_path, "w") as f:
            f.write(payload)


# Per-worker cache for scatter sidecar metadata (populated on first read, shared across tasks)
_scatter_meta_cache: dict[str, dict] = {}


def _read_scatter_meta(parquet_path: str) -> dict:
    """Read a ``.scatter_meta`` sidecar, cached per-worker."""
    meta_path = _scatter_meta_path(parquet_path)
    if meta_path not in _scatter_meta_cache:
        with open_url(meta_path, "r") as f:
            _scatter_meta_cache[meta_path] = json.loads(f.read())
    return _scatter_meta_cache[meta_path]


def _write_scatter_manifest(scatter_paths: list[str], output_path: str) -> None:
    """Write a consolidated scatter manifest combining all sidecar metadata.

    The manifest is a JSON array of objects, each containing a scatter file
    path alongside its chunk_counts, chunk_offsets, and is_pickled metadata.
    Reducers read this single file instead of N individual sidecars.

    Sidecar reads are parallelised via a thread pool since each is an
    independent GCS fetch (I/O bound, O(N) sequential latency otherwise).
    """

    def _read_entry(path: str) -> tuple[str, dict]:
        meta = _read_scatter_meta(path)
        entry: dict = {
            "path": path,
            "chunk_counts": meta["chunk_counts"],
            "is_pickled": meta.get("is_pickled", False),
        }
        if "max_chunk_rows" in meta:
            entry["max_chunk_rows"] = meta["max_chunk_rows"]
        if "avg_item_bytes" in meta:
            entry["avg_item_bytes"] = meta["avg_item_bytes"]
        if meta.get("has_sort_key", False):
            entry["has_sort_key"] = True
        return path, entry

    results: dict[str, dict] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=_SCATTER_META_READ_CONCURRENCY) as pool:
        for path, entry in pool.map(_read_entry, scatter_paths):
            results[path] = entry

    entries = [results[path] for path in scatter_paths]

    ensure_parent_dir(output_path)
    payload = json.dumps(entries)
    with log_time(f"Writing scatter manifest ({len(entries)} files) to {output_path}"):
        with open_url(output_path, "w") as f:
            f.write(payload)


# Per-worker cache for scatter manifests (populated on first read, shared across tasks)
_scatter_manifest_cache: dict[str, list[dict]] = {}


def _read_scatter_manifest(manifest_path: str) -> list[dict]:
    """Read a consolidated scatter manifest, cached per-worker."""
    if manifest_path not in _scatter_manifest_cache:
        with open_url(manifest_path, "r") as f:
            _scatter_manifest_cache[manifest_path] = json.loads(f.read())
    return _scatter_manifest_cache[manifest_path]


def _build_scatter_shard_from_manifest(manifest_path: str, target_shard: int) -> ScatterShard:
    """Build a ScatterShard for one target shard from a consolidated scatter manifest."""
    entries = _read_scatter_manifest(manifest_path)
    iterators: list[ScatterParquetIterator] = []
    with log_time(f"Building ScatterShard for target shard {target_shard} from manifest ({len(entries)} files)"):
        # First pass: count files that have data for this shard
        file_entries = []
        for entry in entries:
            count = entry["chunk_counts"].get(str(target_shard), 0)
            if count == 0:
                continue
            file_entries.append(entry)

        # has_sort_key is True only if ALL entries with data for this shard have it
        has_sort_key = bool(file_entries) and all(entry.get("has_sort_key", False) for entry in file_entries)

        sample_path = file_entries[0]["path"] if file_entries else ""
        filesystem = _get_scatter_read_fs(len(file_entries), sample_path)
        for entry in file_entries:
            count = entry["chunk_counts"].get(str(target_shard), 0)
            iterators.append(
                ScatterParquetIterator(
                    path=entry["path"],
                    shard_idx=target_shard,
                    chunk_count=count,
                    is_pickled=entry.get("is_pickled", False),
                    filesystem=filesystem,
                    has_sort_key=entry.get("has_sort_key", False),
                )
            )

        # Aggregate stats from manifest entries for this shard.
        # max_chunk_rows is a per-shard dict so we only look at target_shard's value.
        # Fall back to the old scalar max_row_group_rows for pre-migration manifests.
        max_rg_rows = 0
        for entry in file_entries:
            per_shard = entry.get("max_chunk_rows", {})
            if per_shard:
                max_rg_rows = max(max_rg_rows, per_shard.get(str(target_shard), 0))
            else:
                # old manifest: scalar max across all shards — use as conservative fallback
                max_rg_rows = max(max_rg_rows, entry.get("max_row_group_rows", 0))
        if max_rg_rows == 0:
            max_rg_rows = 100_000  # fallback for old manifests without stats

        # Weighted avg item bytes (weight by chunk_count for this shard)
        total_chunks_for_avg = 0
        weighted_bytes = 0.0
        for entry in file_entries:
            count = entry["chunk_counts"].get(str(target_shard), 0)
            ab = entry.get("avg_item_bytes", 0.0)
            if ab > 0:
                weighted_bytes += ab * count
                total_chunks_for_avg += count
        avg_item_bytes = weighted_bytes / total_chunks_for_avg if total_chunks_for_avg > 0 else 0.0

    return ScatterShard(
        iterators=iterators, max_row_group_rows=max_rg_rows, avg_item_bytes=avg_item_bytes, has_sort_key=has_sort_key
    )


# ---------------------------------------------------------------------------
# Envelope helpers
# ---------------------------------------------------------------------------


def _make_envelope(
    items: list,
    target_shard: int,
    chunk_idx: int,
    key_values: list | None = None,
    sort_values: list | None = None,
) -> list[dict]:
    enveloped = []
    for i, item in enumerate(items):
        row: dict[str, Any] = {
            _ZEPHYR_SHUFFLE_SHARD_IDX_COL: target_shard,
            _ZEPHYR_SHUFFLE_CHUNK_IDX_COL: chunk_idx,
            _ZEPHYR_SHUFFLE_ITEM_COL: item,
        }
        if key_values is not None:
            row[_ZEPHYR_SHUFFLE_SORT_KEY_COL] = key_values[i]
        if sort_values is not None:
            row[_ZEPHYR_SHUFFLE_SORT_SECONDARY_COL] = sort_values[i]
        enveloped.append(row)
    return enveloped


def _make_pickle_envelope(
    items: list,
    target_shard: int,
    chunk_idx: int,
    key_values: list | None = None,
    sort_values: list | None = None,
) -> list[dict]:
    """Wrap items as pickle-serialized bytes for Arrow-incompatible types."""
    enveloped = []
    for i, item in enumerate(items):
        row: dict[str, Any] = {
            _ZEPHYR_SHUFFLE_SHARD_IDX_COL: target_shard,
            _ZEPHYR_SHUFFLE_CHUNK_IDX_COL: chunk_idx,
            _ZEPHYR_SHUFFLE_PICKLED_COL: cloudpickle.dumps(item),
        }
        if key_values is not None:
            row[_ZEPHYR_SHUFFLE_SORT_KEY_COL] = key_values[i]
        if sort_values is not None:
            row[_ZEPHYR_SHUFFLE_SORT_SECONDARY_COL] = sort_values[i]
        enveloped.append(row)
    return enveloped


def _segment_path(base_path: str, seg_idx: int) -> str:
    """Return the file path for a given segment index.

    ``shard-0000.parquet`` → ``shard-0000-seg0000.parquet``
    """
    stem, ext = os.path.splitext(base_path)
    return f"{stem}-seg{seg_idx:04d}{ext}"


@dataclass
class _ShardBuffer:
    """Per-shard buffer that accumulates Arrow micro-batches and flushes when a byte threshold is reached.

    Items are appended one at a time with their sort key (and optional secondary sort value).
    Every _SCATTER_MICRO_BATCH_SIZE items, they are converted to an Arrow RecordBatch.
    When total buffered bytes exceed _SCATTER_ROW_GROUP_BYTES, take_sorted_batch() drains
    the buffer, sorts in Arrow, and returns a single RecordBatch for the Parquet writer.
    """

    shard_idx: int
    pickled: bool = False
    has_sort: bool = False
    pending: list[tuple[Any, Any, Any | None]] = field(default_factory=list)
    tables: list[pa.RecordBatch] = field(default_factory=list)
    nbytes: int = 0
    chunk_idx: int = 0
    schema: pa.Schema | None = None
    max_rows: int = 0

    def append(self, item: Any, key_value: Any, sort_value: Any | None = None) -> None:
        self.pending.append((item, key_value, sort_value))
        if len(self.pending) >= _SCATTER_MICRO_BATCH_SIZE:
            self._flush_micro()

    def _flush_micro(self) -> None:
        if not self.pending:
            return
        items, keys, sorts = zip(*self.pending, strict=True)
        envelope_fn = _make_pickle_envelope if self.pickled else _make_envelope
        enveloped = envelope_fn(
            list(items),
            self.shard_idx,
            self.chunk_idx,
            list(keys),
            list(sorts) if self.has_sort else None,
        )
        batch = pa.RecordBatch.from_pylist(enveloped, schema=self.schema)
        if self.schema is None:
            self.schema = batch.schema
        self.tables.append(batch)
        self.nbytes += batch.nbytes
        self.pending = []

    def should_flush(self) -> bool:
        return self.nbytes >= _SCATTER_ROW_GROUP_BYTES

    def take_sorted_batch(self) -> pa.RecordBatch | None:
        """Drain buffer, sort by _sort_key in Arrow, return single batch."""
        self._flush_micro()
        if not self.tables:
            return None
        table = pa.Table.from_batches(self.tables, schema=self.schema)
        sort_cols: list[tuple[str, str]] = [(_ZEPHYR_SHUFFLE_SORT_KEY_COL, "ascending")]
        if _ZEPHYR_SHUFFLE_SORT_SECONDARY_COL in table.column_names:
            sort_cols.append((_ZEPHYR_SHUFFLE_SORT_SECONDARY_COL, "ascending"))
        indices = pc.sort_indices(table, sort_keys=sort_cols)
        sorted_table = table.take(indices)
        num_rows = len(sorted_table)
        self.max_rows = max(self.max_rows, num_rows)
        self.chunk_idx += 1
        self.tables = []
        self.nbytes = 0
        return sorted_table.to_batches()[0]

    @property
    def item_count(self) -> int:
        return len(self.pending) + sum(len(t) for t in self.tables)


def _apply_combiner(buffer: list, key_fn: Callable, combiner_fn: Callable) -> list:
    """Apply combiner to a buffer, grouping by key and reducing locally."""
    by_key: dict[object, list] = defaultdict(list)
    with log_time(f"Applying combiner to buffer of size {len(buffer)}", level=logging.DEBUG):
        for item in buffer:
            by_key[key_fn(item)].append(item)
        combined: list = []
        for key, items in by_key.items():
            combined.extend(combiner_fn(key, iter(items)))
    return combined


def _write_parquet_scatter(
    items: Iterator,
    source_shard: int,
    parquet_path: str,
    key_fn: Callable,
    num_output_shards: int,
    sort_fn: Callable | None = None,
    combiner_fn: Callable | None = None,
    pickled: bool = False,
) -> ListShard:
    """Route items to target shards, buffer, sort, and write as Parquet row groups.

    Items are accumulated incrementally in per-shard _ShardBuffer instances,
    converted to Arrow micro-batches every _SCATTER_MICRO_BATCH_SIZE items.
    When a buffer exceeds _SCATTER_ROW_GROUP_BYTES, it is drained, sorted in
    Arrow by _sort_key (and optionally _sort_secondary), and written as a
    Parquet row group.

    Writes ``.scatter_meta`` sidecar files alongside each Parquet segment.

    Returns:
        A ListShard containing the segment file paths.
    """
    seg_shard_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    buffers: dict[int, _ShardBuffer] = {}
    n_chunks_flushed = 0
    seg_idx = 0
    seg_paths: list[str] = []
    schema: pa.Schema | None = None
    writer: pq.ParquetWriter | None = None
    seg_file = ""

    pending_chunk: pa.RecordBatch | None = None
    pending_target: int = -1
    pending_cnt: int = 0

    avg_item_bytes: float = 0.0
    _sampled_avg = False

    def _get_buffer(target: int) -> _ShardBuffer:
        if target not in buffers:
            buffers[target] = _ShardBuffer(shard_idx=target, pickled=pickled, has_sort=sort_fn is not None)
        return buffers[target]

    def _flush_pending() -> None:
        nonlocal n_chunks_flushed, pending_chunk
        if pending_chunk is None:
            return
        writer.write_batch(pending_chunk)
        seg_shard_counts[seg_idx][pending_target] = seg_shard_counts[seg_idx].get(pending_target, 0) + 1
        n_chunks_flushed += 1
        pending_chunk = None
        if n_chunks_flushed % 10 == 0:
            logger.info(
                "[shard %d segment %d] Wrote %d parquet chunks so far (latest chunk size: %d items)",
                source_shard,
                seg_idx,
                n_chunks_flushed,
                pending_cnt,
            )

    def _ensure_writer(chunk_schema: pa.Schema) -> pa.Schema:
        nonlocal schema, writer, seg_file, seg_idx
        if schema is None:
            schema = chunk_schema
            seg_file = _segment_path(parquet_path, seg_idx)
            seg_paths.append(seg_file)
            ensure_parent_dir(seg_file)
            writer = pq.ParquetWriter(seg_file, schema)
        elif chunk_schema != schema:
            _flush_pending()
            writer.close()
            schema = pa.unify_schemas([schema, chunk_schema])
            seg_idx += 1
            # Reset chunk_idx for all buffers in new segment
            for buf in buffers.values():
                buf.chunk_idx = 0
            seg_file = _segment_path(parquet_path, seg_idx)
            seg_paths.append(seg_file)
            ensure_parent_dir(seg_file)
            writer = pq.ParquetWriter(seg_file, schema)
            logger.info(
                "[shard %d] Schema evolved after %d chunks; starting segment %d",
                source_shard,
                n_chunks_flushed,
                seg_idx,
            )
        else:
            _flush_pending()
        return schema

    def _flush_buffer(buf: _ShardBuffer) -> None:
        nonlocal pending_chunk, pending_target, pending_cnt, avg_item_bytes, _sampled_avg

        if combiner_fn is not None:
            # Combiner path: drain buffer to Python, apply combiner, re-sort in Arrow
            buf._flush_micro()
            if not buf.tables:
                return
            table = pa.Table.from_batches(buf.tables, schema=buf.schema)
            item_col = _ZEPHYR_SHUFFLE_PICKLED_COL if pickled else _ZEPHYR_SHUFFLE_ITEM_COL
            py_items = table.column(item_col).to_pylist()
            if pickled:
                py_items = [pickle.loads(b) for b in py_items]
            combined = _apply_combiner(py_items, key_fn, combiner_fn)
            # Re-create a fresh buffer from combined items
            combined_buf = _ShardBuffer(shard_idx=buf.shard_idx, pickled=pickled, has_sort=sort_fn is not None)
            combined_buf.chunk_idx = buf.chunk_idx
            for item in combined:
                k = key_fn(item)
                sv = sort_fn(item) if sort_fn else None
                combined_buf.append(item, k, sv)
            batch = combined_buf.take_sorted_batch()
            buf.chunk_idx = combined_buf.chunk_idx
            buf.tables = []
            buf.nbytes = 0
            buf.pending = []
            buf.max_rows = max(buf.max_rows, combined_buf.max_rows)
        else:
            batch = buf.take_sorted_batch()

        if batch is None:
            return

        write_schema = _ensure_writer(batch.schema)
        if batch.schema != write_schema:
            batch = batch.cast(write_schema)
        pending_chunk = batch
        pending_target = buf.shard_idx
        pending_cnt = len(batch)

        if not _sampled_avg and len(batch) > 0:
            avg_item_bytes = batch.nbytes / len(batch)
            _sampled_avg = True

    for item in items:
        key = key_fn(item)
        target = deterministic_hash(key) % num_output_shards
        sort_val = sort_fn(item) if sort_fn else None
        buf = _get_buffer(target)
        buf.append(item, key, sort_val)
        if buf.should_flush():
            _flush_buffer(buf)

    # Flush remaining buffers — write each shard as its own row group so PyArrow
    # can use min/max statistics on shard_idx to skip non-matching row groups on read.
    with log_time(f"Flushing remaining buffers for {parquet_path}"):
        _flush_pending()
        for target in sorted(buffers.keys()):
            buf = buffers[target]
            if buf.item_count == 0:
                continue
            _flush_buffer(buf)
        _flush_pending()

    if writer is not None:
        writer.close()

    # Collect per-shard max rows from buffers
    per_shard_max_rows: dict[int, int] = {target: buf.max_rows for target, buf in buffers.items() if buf.max_rows > 0}

    with log_time(f"Writing scatter meta for {parquet_path}"):
        for i, path in enumerate(seg_paths):
            counts = dict(seg_shard_counts.get(i, {}))
            seg_max_rows = {shard: per_shard_max_rows[shard] for shard in counts if per_shard_max_rows.get(shard, 0) > 0}
            _write_scatter_meta(path, counts, pickled, seg_max_rows, avg_item_bytes, has_sort_key=True)

    return ListShard(refs=[MemChunk(items=seg_paths)])
