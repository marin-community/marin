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
from iris.marin_fs import open_url, url_to_fs
from iris.time_utils import log_time

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
    """Shard backed by a list of iterable references (MemChunk, etc.)."""

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

_ZEPHYR_SHARD_IDX = "_zephyr_shard_idx"
_ZEPHYR_CHUNK_IDX = "_zephyr_chunk_idx"
_ZEPHYR_PAYLOAD = "_zephyr_payload"
_ZEPHYR_SORT_KEY = "_zephyr_sort_key"
_ZEPHYR_SORT_SECONDARY = "_zephyr_sort_secondary"

_SCATTER_META_SUFFIX = ".scatter_meta"
_SCATTER_MANIFEST_NAME = "scatter_metadata"

_SCATTER_META_READ_CONCURRENCY = 256
_SCATTER_READ_BUFFER_FRACTION = 0.25
_SCATTER_MICRO_BATCH_SIZE = 64
_SCATTER_ROW_GROUP_BYTES = 2 * 1024 * 1024  # 2 MB


def is_zephyr_column(name: str) -> bool:
    """Return True if the column name is a zephyr metadata column."""
    return name.startswith("_zephyr_")


# ---------------------------------------------------------------------------
# Envelope helpers
# ---------------------------------------------------------------------------


def make_envelope_batch(
    items: list,
    shard_idx: int,
    chunk_idx: int,
    key_values: list,
    sort_values: list | None,
    pickled: bool,
) -> pa.RecordBatch:
    """Build an Arrow batch wrapping items with zephyr metadata columns.

    Flat mode (pickled=False): each item dict's fields become columns alongside _zephyr_* metadata.
    Pickle mode (pickled=True): items become _zephyr_payload bytes alongside _zephyr_* metadata.
    """
    n = len(items)
    rows = []
    for i in range(n):
        row: dict[str, Any] = {
            _ZEPHYR_SHARD_IDX: shard_idx,
            _ZEPHYR_CHUNK_IDX: chunk_idx,
            _ZEPHYR_SORT_KEY: key_values[i],
        }
        if sort_values is not None:
            row[_ZEPHYR_SORT_SECONDARY] = sort_values[i]
        if pickled:
            row[_ZEPHYR_PAYLOAD] = cloudpickle.dumps(items[i])
        else:
            row.update(items[i])
        rows.append(row)
    return pa.RecordBatch.from_pylist(rows)


def unwrap_items(table_or_batch: pa.Table | pa.RecordBatch, pickled: bool) -> list:
    """Extract user items from an envelope table/batch.

    Flat mode: drop _zephyr_* columns, return remaining fields as dicts.
    Pickle mode: deserialize _zephyr_payload column.
    """
    if pickled:
        payload_col = table_or_batch.column(_ZEPHYR_PAYLOAD)
        return [pickle.loads(b) for b in payload_col.to_pylist()]

    if isinstance(table_or_batch, pa.RecordBatch):
        table_or_batch = pa.Table.from_batches([table_or_batch])
    user_cols = [name for name in table_or_batch.column_names if not is_zephyr_column(name)]
    return table_or_batch.select(user_cols).to_pylist()


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

    if per_file >= 5 * 1024 * 1024:
        return default_fs

    # blocksize is a gcsfs/s3fs attribute; not all fsspec implementations have it.
    # We check via the storage_options dict rather than hasattr to stay explicit.
    if "block_size" not in getattr(fs, "storage_options", {}):
        return default_fs

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

    def __iter__(self) -> Iterator:
        for table in self.get_chunk_tables():
            yield from unwrap_items(table, self.is_pickled)

    def get_chunk_iterators(self, batch_size: int = 1024) -> Iterator[Iterator]:
        """Yield one lazy iterator per sorted chunk, each backed by get_chunk_tables + unwrap_items."""
        for table in self.get_chunk_tables(batch_size=batch_size):
            yield iter(unwrap_items(table, self.is_pickled))

    def get_chunk_tables(self, batch_size: int = 1024) -> Iterator[pa.Table]:
        """Yield Arrow tables per sorted chunk (no Python materialization).

        Always selects all columns; the caller (unwrap_items or the Arrow reduce
        path) handles column filtering.
        """
        _, fs_path = url_to_fs(self.path)
        dataset: pad.FileSystemDataset = pad.dataset(fs_path, format="parquet", filesystem=self.filesystem)

        # Select item columns + sort columns; filter on shard/chunk metadata
        if self.is_pickled:
            item_cols = [_ZEPHYR_PAYLOAD]
        else:
            item_cols = [name for name in dataset.schema.names if not is_zephyr_column(name)]

        columns = [*item_cols, _ZEPHYR_SORT_KEY]
        if _ZEPHYR_SORT_SECONDARY in dataset.schema.names:
            columns.append(_ZEPHYR_SORT_SECONDARY)

        for chunk_idx in range(self.chunk_count):
            scanner = dataset.scanner(
                columns=columns,
                filter=((pc.field(_ZEPHYR_SHARD_IDX) == self.shard_idx) & (pc.field(_ZEPHYR_CHUNK_IDX) == chunk_idx)),
                batch_size=batch_size,
                use_threads=False,
            )
            batches = list(scanner.to_batches())
            if batches:
                yield pa.Table.from_batches(batches)


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
        if total_chunks == 0 or memory_limit <= 0 or self.avg_item_bytes <= 0:
            return False
        estimated = total_chunks * self.max_row_group_rows * self.avg_item_bytes
        return estimated > memory_limit * memory_fraction

    def _compute_batch_size(self) -> int:
        """Compute a safe batch_size that keeps scatter read buffers within budget.

        With N total chunk iterators in the k-way merge, each holding one
        batch of batch_size deserialized items in memory simultaneously,
        the total buffer footprint is N * batch_size * bytes_per_item.
        We cap this at _SCATTER_READ_BUFFER_FRACTION of the worker's memory limit.
        """
        total_chunks = sum(it.chunk_count for it in self.iterators)
        if total_chunks == 0 or self.avg_item_bytes <= 0:
            return 1024
        bytes_per_item = self.avg_item_bytes
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

    Replaces the ``.parquet`` extension: ``shard-0000-seg0000.parquet`` ->
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
    payload = json.dumps(payload_dict)
    with log_time(f"Writing scatter meta for {parquet_path} to {meta_path}", level=logging.DEBUG):
        with open_url(meta_path, "w") as f:
            f.write(payload)


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
        file_entries = []
        for entry in entries:
            count = entry["chunk_counts"].get(str(target_shard), 0)
            if count == 0:
                continue
            file_entries.append(entry)

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
                )
            )

        max_rg_rows = 0
        for entry in file_entries:
            per_shard = entry.get("max_chunk_rows", {})
            max_rg_rows = max(max_rg_rows, per_shard.get(str(target_shard), 0))
        if max_rg_rows == 0:
            max_rg_rows = 100_000

        total_chunks_for_avg = 0
        weighted_bytes = 0.0
        for entry in file_entries:
            count = entry["chunk_counts"].get(str(target_shard), 0)
            ab = entry.get("avg_item_bytes", 0.0)
            if ab > 0:
                weighted_bytes += ab * count
                total_chunks_for_avg += count
        avg_item_bytes = weighted_bytes / total_chunks_for_avg if total_chunks_for_avg > 0 else 0.0

    return ScatterShard(iterators=iterators, max_row_group_rows=max_rg_rows, avg_item_bytes=avg_item_bytes)


# ---------------------------------------------------------------------------
# Internal write machinery
# ---------------------------------------------------------------------------


def _segment_path(base_path: str, seg_idx: int) -> str:
    """``shard-0000.parquet`` -> ``shard-0000-seg0000.parquet``"""
    stem, ext = os.path.splitext(base_path)
    return f"{stem}-seg{seg_idx:04d}{ext}"


@dataclass
class _ShardBuffer:
    """Per-shard buffer that accumulates Arrow micro-batches and flushes when a byte threshold is reached.

    Items are appended one at a time with their sort key (and optional secondary sort value).
    Every _SCATTER_MICRO_BATCH_SIZE items, they are converted to an Arrow RecordBatch via
    make_envelope_batch. When total buffered bytes exceed _SCATTER_ROW_GROUP_BYTES,
    take_sorted_batch() drains the buffer, sorts in Arrow, and returns a single RecordBatch.
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
        batch = make_envelope_batch(
            list(items),
            self.shard_idx,
            self.chunk_idx,
            list(keys),
            list(sorts) if self.has_sort else None,
            pickled=self.pickled,
        )
        if self.schema is None:
            self.schema = batch.schema
        self.tables.append(batch)
        self.nbytes += batch.nbytes
        self.pending = []

    def should_flush(self) -> bool:
        return self.nbytes >= _SCATTER_ROW_GROUP_BYTES

    def take_sorted_batch(self) -> pa.RecordBatch | None:
        """Drain buffer, sort by _zephyr_sort_key in Arrow, return single batch."""
        self._flush_micro()
        if not self.tables:
            return None
        table = pa.Table.from_batches(self.tables, schema=self.schema)
        sort_cols: list[tuple[str, str]] = [(_ZEPHYR_SORT_KEY, "ascending")]
        if _ZEPHYR_SORT_SECONDARY in table.column_names:
            sort_cols.append((_ZEPHYR_SORT_SECONDARY, "ascending"))
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
    Arrow by _zephyr_sort_key (and optionally _zephyr_sort_secondary), and
    written as a Parquet row group.

    Writes ``.scatter_meta`` sidecar files alongside each Parquet segment.
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
            writer = pq.ParquetWriter(seg_file, schema, compression="zstd", compression_level=1)
        elif chunk_schema != schema:
            _flush_pending()
            writer.close()
            schema = pa.unify_schemas([schema, chunk_schema])
            seg_idx += 1
            for buf in buffers.values():
                buf.chunk_idx = 0
            seg_file = _segment_path(parquet_path, seg_idx)
            seg_paths.append(seg_file)
            ensure_parent_dir(seg_file)
            writer = pq.ParquetWriter(seg_file, schema, compression="zstd", compression_level=1)
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
            py_items = unwrap_items(table, pickled)
            combined = _apply_combiner(py_items, key_fn, combiner_fn)
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

    per_shard_max_rows: dict[int, int] = {target: buf.max_rows for target, buf in buffers.items() if buf.max_rows > 0}

    with log_time(f"Writing scatter meta for {parquet_path}"):
        for i, path in enumerate(seg_paths):
            counts = dict(seg_shard_counts.get(i, {}))
            seg_max_rows = {shard: per_shard_max_rows[shard] for shard in counts if per_shard_max_rows.get(shard, 0) > 0}
            _write_scatter_meta(path, counts, pickled, seg_max_rows, avg_item_bytes)

    return ListShard(refs=[MemChunk(items=seg_paths)])
