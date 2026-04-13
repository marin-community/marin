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
from dataclasses import dataclass
from typing import Any

import cloudpickle
import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
from iris.env_resources import TaskResources as _TaskResources
from rigging.filesystem import open_url, url_to_fs
from rigging.timing import log_time

from zephyr.plan import deterministic_hash
from zephyr.writers import INTERMEDIATE_CHUNK_SIZE, ensure_parent_dir

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

_SCATTER_META_READ_CONCURRENCY = 256
# Number of items sampled from the first flush to estimate avg_item_bytes at scatter-write time
_SCATTER_SAMPLE_SIZE = 100
# Fraction of total memory limit to budget for scatter read buffers
_SCATTER_READ_BUFFER_FRACTION = 0.25


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

    Opens the file via ``pq.ParquetFile`` and uses Parquet row-group
    statistics on ``(shard_idx, chunk_idx)`` for predicate pushdown,
    avoiding the ``pyarrow.dataset`` memory leak (apache/arrow#39808).
    """

    path: str
    shard_idx: int
    chunk_count: int
    is_pickled: bool
    filesystem: pa.fs.FileSystem

    def __iter__(self) -> Iterator:
        for chunk_iter in self.get_chunk_iterators():
            yield from chunk_iter

    def get_chunk_iterators(self, batch_size: int = 1024) -> Iterator[Iterator]:
        """Yield one lazy iterator per sorted chunk.

        Opens the file once via ``pq.ParquetFile`` and uses row-group
        statistics to skip non-matching row groups (equivalent to dataset
        Scanner predicate pushdown for the scatter envelope columns).
        """

        _, fs_path = url_to_fs(self.path)
        pf = pq.ParquetFile(self.filesystem.open_input_file(fs_path))
        col = _ZEPHYR_SHUFFLE_PICKLED_COL if self.is_pickled else _ZEPHYR_SHUFFLE_ITEM_COL

        for chunk_idx in range(self.chunk_count):
            yield self._iter_chunk(pf, col, chunk_idx)

    def _iter_chunk(self, pf: pq.ParquetFile, col: str, chunk_idx: int) -> Iterator:
        from zephyr.readers import iter_parquet_row_groups

        # The scatter writer writes one (shard_idx, chunk_idx) per row group,
        # so equality_predicates on min/max statistics skip non-matching row
        # groups without reading data — equivalent to dataset predicate pushdown.
        for table in iter_parquet_row_groups(
            pf,
            columns=[col],
            equality_predicates={
                _ZEPHYR_SHUFFLE_SHARD_IDX_COL: self.shard_idx,
                _ZEPHYR_SHUFFLE_CHUNK_IDX_COL: chunk_idx,
            },
        ):
            items = table.column(col).to_pylist()
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
        if total_chunks == 0:
            return False
        if self.avg_item_bytes <= 0:
            raise ValueError(
                "avg_item_bytes not available in scatter manifest. "
                "Re-run the scatter stage with a version that records avg_item_bytes."
            )
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
        if total_chunks == 0:
            return 1024
        if self.avg_item_bytes <= 0:
            raise ValueError(
                "avg_item_bytes not available in scatter manifest. "
                "Re-run the scatter stage with a version that records avg_item_bytes."
            )
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
        # Filter to entries that have data for this shard
        shard_key = str(target_shard)
        file_entries = []
        for entry in entries:
            count = entry["chunk_counts"].get(shard_key, 0)
            if count > 0:
                file_entries.append((entry, count))

        sample_path = file_entries[0][0]["path"] if file_entries else ""
        filesystem = _get_scatter_read_fs(len(file_entries), sample_path)

        # Single pass: build iterators and aggregate stats
        max_rg_rows = 0
        total_chunks_for_avg = 0
        weighted_bytes = 0.0
        for entry, count in file_entries:
            iterators.append(
                ScatterParquetIterator(
                    path=entry["path"],
                    shard_idx=target_shard,
                    chunk_count=count,
                    is_pickled=entry.get("is_pickled", False),
                    filesystem=filesystem,
                )
            )

            # max_chunk_rows is a per-shard dict; fall back to old scalar max_row_group_rows
            per_shard = entry.get("max_chunk_rows", {})
            if per_shard:
                max_rg_rows = max(max_rg_rows, per_shard.get(shard_key, 0))
            else:
                max_rg_rows = max(max_rg_rows, entry.get("max_row_group_rows", 0))

            # Weighted avg item bytes (weight by chunk_count for this shard)
            ab = entry.get("avg_item_bytes", 0.0)
            if ab > 0:
                weighted_bytes += ab * count
                total_chunks_for_avg += count

        if max_rg_rows == 0:
            max_rg_rows = 100_000  # fallback for old manifests without stats
        avg_item_bytes = weighted_bytes / total_chunks_for_avg if total_chunks_for_avg > 0 else 0.0

    return ScatterShard(iterators=iterators, max_row_group_rows=max_rg_rows, avg_item_bytes=avg_item_bytes)


# ---------------------------------------------------------------------------
# Envelope helpers
# ---------------------------------------------------------------------------


def _make_envelope(items: list, target_shard: int, chunk_idx: int) -> list[dict]:
    return [
        {
            _ZEPHYR_SHUFFLE_SHARD_IDX_COL: target_shard,
            _ZEPHYR_SHUFFLE_CHUNK_IDX_COL: chunk_idx,
            _ZEPHYR_SHUFFLE_ITEM_COL: item,
        }
        for item in items
    ]


def _make_pickle_envelope(items: list, target_shard: int, chunk_idx: int) -> list[dict]:
    """Wrap items as pickle-serialized bytes for Arrow-incompatible types."""
    return [
        {
            _ZEPHYR_SHUFFLE_SHARD_IDX_COL: target_shard,
            _ZEPHYR_SHUFFLE_CHUNK_IDX_COL: chunk_idx,
            _ZEPHYR_SHUFFLE_PICKLED_COL: cloudpickle.dumps(item),
        }
        for item in items
    ]


def _segment_path(base_path: str, seg_idx: int) -> str:
    """Return the file path for a given segment index.

    ``shard-0000.parquet`` → ``shard-0000-seg0000.parquet``
    """
    stem, ext = os.path.splitext(base_path)
    return f"{stem}-seg{seg_idx:04d}{ext}"


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

    Handles the full scatter pipeline: hash-routing each item to a target shard,
    buffering per-shard, applying an optional combiner, sorting each buffer, and
    writing sorted chunks as Parquet row groups with envelope wrapping.

    Writes ``.scatter_meta`` sidecar files alongside each Parquet segment.

    Returns:
        A ListShard containing the segment file paths.
    """
    if sort_fn is not None:
        captured_sort_fn = sort_fn

        def _sort_key(item):
            return (key_fn(item), captured_sort_fn(item))

    else:
        _sort_key = key_fn

    chunk_size = INTERMEDIATE_CHUNK_SIZE

    # Per-segment per-shard chunk counts
    seg_shard_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    per_shard_chunk_cnt: dict[int, int] = defaultdict(int)
    buffers: dict[int, list] = defaultdict(list)
    n_chunks_flushed = 0
    seg_idx = 0
    seg_paths: list[str] = []
    schema: pa.Schema | None = None
    writer: pq.ParquetWriter | None = None
    seg_file = ""

    pending_chunk: pa.RecordBatch | None = None
    pending_target: int = -1
    pending_cnt: int = 0

    per_shard_max_rows: dict[int, int] = defaultdict(int)
    avg_item_bytes: float = 0.0
    _sampled_avg = False

    def _flush_pending():
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

    def _prepare_batch(target_shard: int, buf: list) -> list[dict]:
        """Apply combiner, sort, envelope a buffer. Returns enveloped rows."""
        if combiner_fn is not None:
            buf = _apply_combiner(buf, key_fn, combiner_fn)
        buf.sort(key=_sort_key)
        shard_chunk_idx = per_shard_chunk_cnt[target_shard]
        per_shard_chunk_cnt[target_shard] += 1
        envelope_fn = _make_pickle_envelope if pickled else _make_envelope
        return envelope_fn(buf, target_shard, shard_chunk_idx)

    def _ensure_writer(chunk_schema: pa.Schema) -> pa.Schema:
        """Ensure Parquet writer is open and compatible. Returns the active write schema."""
        nonlocal schema, writer, seg_file, seg_idx, per_shard_chunk_cnt
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
            per_shard_chunk_cnt = defaultdict(int)  # chunk_idx restarts at 0 in new segment
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

    def _write_buffer(target_shard: int, buf: list) -> None:
        """Sort a buffer and write it as a Parquet row group."""
        nonlocal pending_chunk, pending_target, pending_cnt, avg_item_bytes, _sampled_avg
        enveloped = _prepare_batch(target_shard, buf)
        chunk_arrow = pa.RecordBatch.from_pylist(enveloped)
        write_schema = _ensure_writer(chunk_arrow.schema)
        if chunk_arrow.schema != write_schema:
            chunk_arrow = chunk_arrow.cast(write_schema)
        pending_chunk = chunk_arrow
        pending_target = target_shard
        pending_cnt = len(buf)
        per_shard_max_rows[target_shard] = max(per_shard_max_rows[target_shard], len(buf))

        # Sample avg_item_bytes once on first flush
        if not _sampled_avg and len(enveloped) > 0:
            sample_size = min(len(enveloped), _SCATTER_SAMPLE_SIZE)
            sample_rows = enveloped[:sample_size]
            if pickled:
                total_bytes = sum(len(row[_ZEPHYR_SHUFFLE_PICKLED_COL]) for row in sample_rows)
            else:
                total_bytes = sum(len(pickle.dumps(row[_ZEPHYR_SHUFFLE_ITEM_COL])) for row in sample_rows)
            avg_item_bytes = total_bytes / len(sample_rows)
            _sampled_avg = True

    # Route items to target shards, flush buffers at chunk_size
    for item in items:
        key = key_fn(item)
        target = deterministic_hash(key) % num_output_shards
        buffers[target].append(item)
        if chunk_size > 0 and len(buffers[target]) >= chunk_size:
            _write_buffer(target, buffers[target])
            buffers[target] = []

    # Flush remaining buffers — write each shard as its own row group so PyArrow
    # can use min/max statistics on shard_idx to skip non-matching row groups on read.
    with log_time(f"Flushing remaining buffers for {parquet_path}"):
        _flush_pending()
        for target, buf in sorted(buffers.items()):
            if not buf:
                continue
            _write_buffer(target, buf)
        _flush_pending()

    if writer is not None:
        writer.close()

    # Write sidecar metadata for each segment.
    # chunk_offsets track where each segment's chunks start in the global
    # chunk_idx space (cumulative across segments from this source shard).
    with log_time(f"Writing scatter meta for {parquet_path}"):
        for i, path in enumerate(seg_paths):
            counts = dict(seg_shard_counts.get(i, {}))
            seg_max_rows = {shard: per_shard_max_rows[shard] for shard in counts if per_shard_max_rows[shard] > 0}
            _write_scatter_meta(path, counts, pickled, seg_max_rows, avg_item_bytes)

    return ListShard(refs=[MemChunk(items=seg_paths)])
