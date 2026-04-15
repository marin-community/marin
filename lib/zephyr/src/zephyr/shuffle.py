# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scatter/shuffle support for Zephyr pipelines.

Each source-shard's scatter output is a single binary file containing a
sequence of zstd-compressed frames. Within one chunk's zstd frame, items
are written in sub-batches of ``_SUB_BATCH_SIZE`` — each sub-batch is a
single ``pickle.dump(list_of_items)`` into the zstd stream. This amortises
per-item pickle/zstd dispatch over a sub-batch while still letting the
reader stream sub-batches lazily without materialising the full chunk.

A JSON sidecar (``.scatter_meta``) maps ``target_shard -> [(offset, length)]``
byte ranges into the data file, plus per-shard ``max_chunk_rows`` and a global
``avg_item_bytes`` estimate. Sidecars from all source shards are aggregated
into a single ``scatter_metadata`` manifest at the end of the scatter stage,
which reducers consume to build :class:`ScatterShard` instances.

On read, each chunk is fetched with a single ``cat_file`` range GET (one
HTTP request, no per-chunk file handle), then streamed via
``pickle.load`` on a length-bounded zstd reader. Per-iterator memory stays
near-constant: one buffered item plus the zstd decoder state plus the
chunk's compressed bytes (typically a few MB). This bound is essential for
skewed shuffles where one reducer pulls disproportionate data and the
external-sort fan-in opens hundreds of chunk iterators at once.

Compared to the previous Parquet-based layout this drops Arrow from the
shuffle data plane, removes schema-evolution segment splits, and replaces
row-group statistics with explicit byte ranges.
"""

from __future__ import annotations

import concurrent.futures
import io
import json
import logging
import os
import pickle
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import zstandard as zstd
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
# Constants
# ---------------------------------------------------------------------------

_SCATTER_META_SUFFIX = ".scatter_meta"
_SCATTER_MANIFEST_NAME = "scatter_metadata"
_SCATTER_DATA_SUFFIX = ".shuffle"

_SCATTER_META_READ_CONCURRENCY = 256
# Number of items sampled from the first flush to estimate avg_item_bytes.
_SCATTER_SAMPLE_SIZE = 100
# Fraction of total memory budgeted for read-side decompression buffers.
_SCATTER_READ_BUFFER_FRACTION = 0.25

_ZSTD_COMPRESS_LEVEL = 3
# Items per pickle.dump call within a chunk. Larger = faster (less per-call
# dispatch overhead), smaller = lower per-iterator read memory.
_SUB_BATCH_SIZE = 1024


# ---------------------------------------------------------------------------
# Sidecar / manifest helpers
# ---------------------------------------------------------------------------


def _scatter_meta_path(data_path: str) -> str:
    """``shard-0000.shuffle`` -> ``shard-0000.scatter_meta``."""
    stem, _ = os.path.splitext(data_path)
    return stem + _SCATTER_META_SUFFIX


def _write_scatter_meta(data_path: str, sidecar: dict) -> None:
    meta_path = _scatter_meta_path(data_path)
    payload = json.dumps(sidecar)
    with log_time(f"Writing scatter meta for {data_path} to {meta_path}", level=logging.DEBUG):
        with open_url(meta_path, "w") as f:
            f.write(payload)


# Per-worker caches for sidecar + manifest reads.
_scatter_meta_cache: dict[str, dict] = {}
_scatter_manifest_cache: dict[str, list[dict]] = {}


def _read_scatter_meta(data_path: str) -> dict:
    meta_path = _scatter_meta_path(data_path)
    if meta_path not in _scatter_meta_cache:
        with open_url(meta_path, "r") as f:
            _scatter_meta_cache[meta_path] = json.loads(f.read())
    return _scatter_meta_cache[meta_path]


def _read_scatter_manifest(manifest_path: str) -> list[dict]:
    if manifest_path not in _scatter_manifest_cache:
        with open_url(manifest_path, "r") as f:
            _scatter_manifest_cache[manifest_path] = json.loads(f.read())
    return _scatter_manifest_cache[manifest_path]


def _write_scatter_manifest(scatter_paths: list[str], output_path: str) -> None:
    """Aggregate ``.scatter_meta`` sidecars into a single manifest.

    Sidecar reads run in parallel since each is an independent GCS GET.
    """

    def _read_entry(path: str) -> tuple[str, dict]:
        meta = _read_scatter_meta(path)
        return path, {"path": path, **meta}

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


# ---------------------------------------------------------------------------
# Reader: one source-file's chunks for one target shard
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScatterFileIterator:
    """Reads chunks for one target shard from one scatter file.

    ``chunks`` is a tuple of ``(offset, length)`` byte ranges. Each chunk is
    fetched on demand via a single ``cat_file`` and streamed item-by-item.
    Per-iterator memory is bounded by the chunk's compressed size (typically
    a few MB) plus tiny zstd/pickle state.
    """

    path: str
    chunks: tuple[tuple[int, int], ...]

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    def __iter__(self) -> Iterator:
        for chunk_iter in self.get_chunk_iterators():
            yield from chunk_iter

    def get_chunk_iterators(self) -> Iterator[Iterator]:
        """Yield one lazy iterator per chunk, in write order."""
        for offset, length in self.chunks:
            yield _iter_chunk(self.path, offset, length)


def _iter_chunk(path: str, offset: int, length: int) -> Iterator:
    """Fetch one chunk's compressed bytes via cat_file and stream items.

    Each chunk is a zstd frame containing a sequence of pickled sub-batches
    (lists of up to ``_SUB_BATCH_SIZE`` items). The reader streams one
    sub-batch at a time, so per-iterator memory is bounded by the
    sub-batch size plus the chunk's compressed bytes.
    """
    fs, fs_path = url_to_fs(path)
    blob = fs.cat_file(fs_path, start=offset, end=offset + length)
    with zstd.ZstdDecompressor().stream_reader(io.BytesIO(blob)) as reader:
        while True:
            try:
                sub_batch = pickle.load(reader)
            except EOFError:
                return
            yield from sub_batch


# ---------------------------------------------------------------------------
# ScatterShard: built from manifest, fed to Reduce
# ---------------------------------------------------------------------------


@dataclass
class ScatterShard:
    """All scatter chunks for one target shard, across all source files."""

    iterators: list[ScatterFileIterator]
    max_chunk_rows: int = 100_000
    avg_item_bytes: float = 0.0

    def __iter__(self) -> Iterator:
        for it in self.iterators:
            yield from it

    def get_iterators(self) -> Iterator[Iterator]:
        for it in self.iterators:
            yield from it.get_chunk_iterators()

    def needs_external_sort(self, memory_limit: int, memory_fraction: float = 0.5) -> bool:
        """Return True if opening all chunks at once would blow the budget."""
        total_chunks = sum(it.chunk_count for it in self.iterators)
        if total_chunks == 0:
            return False
        if self.avg_item_bytes <= 0:
            raise ValueError(
                "avg_item_bytes not available in scatter manifest. "
                "Re-run the scatter stage with a version that records avg_item_bytes."
            )
        # Heuristic: assume each open chunk could hold up to max_chunk_rows
        # items in the worst case (e.g. if downstream materialises chunks).
        estimated = total_chunks * self.max_chunk_rows * self.avg_item_bytes
        return estimated > memory_limit * memory_fraction


def _build_scatter_shard_from_manifest(manifest_path: str, target_shard: int) -> ScatterShard:
    """Build a ScatterShard for one target shard from the consolidated manifest."""
    entries = _read_scatter_manifest(manifest_path)
    iterators: list[ScatterFileIterator] = []
    shard_key = str(target_shard)
    max_rows = 0
    weighted_bytes = 0.0
    total_chunks_for_avg = 0

    with log_time(f"Building ScatterShard for target shard {target_shard} from manifest ({len(entries)} files)"):
        for entry in entries:
            shards = entry.get("shards", {})
            ranges = shards.get(shard_key)
            if not ranges:
                continue

            iterators.append(
                ScatterFileIterator(
                    path=entry["path"],
                    chunks=tuple((int(off), int(length)) for off, length in ranges),
                )
            )

            per_shard_max = entry.get("max_chunk_rows", {})
            max_rows = max(max_rows, per_shard_max.get(shard_key, 0))

            ab = entry.get("avg_item_bytes", 0.0)
            if ab > 0:
                count = len(ranges)
                weighted_bytes += ab * count
                total_chunks_for_avg += count

    if max_rows == 0:
        max_rows = 100_000
    avg_item_bytes = weighted_bytes / total_chunks_for_avg if total_chunks_for_avg > 0 else 0.0

    return ScatterShard(iterators=iterators, max_chunk_rows=max_rows, avg_item_bytes=avg_item_bytes)


# ---------------------------------------------------------------------------
# Combiner / sort helper
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


# ---------------------------------------------------------------------------
# Scatter writer
# ---------------------------------------------------------------------------


def _write_chunk_frame(items: list) -> bytes:
    """Encode a list of items as one zstd frame of pickled sub-batches.

    Items are split into sub-batches of ``_SUB_BATCH_SIZE`` and each
    sub-batch is written as a single ``pickle.dump(sublist)`` into the
    same zstd stream. This batches per-call dispatch overhead while
    keeping per-iterator read memory bounded by the sub-batch size.
    """
    raw = io.BytesIO()
    cctx = zstd.ZstdCompressor(level=_ZSTD_COMPRESS_LEVEL)
    with cctx.stream_writer(raw, closefd=False) as zf:
        for i in range(0, len(items), _SUB_BATCH_SIZE):
            pickle.dump(items[i : i + _SUB_BATCH_SIZE], zf, protocol=pickle.HIGHEST_PROTOCOL)
    return raw.getvalue()


def _write_scatter(
    items: Iterator,
    source_shard: int,
    data_path: str,
    key_fn: Callable,
    num_output_shards: int,
    sort_fn: Callable | None = None,
    combiner_fn: Callable | None = None,
) -> ListShard:
    """Route items to target shards, buffer, sort, and append zstd chunks.

    Writes one binary data file plus one ``.scatter_meta`` sidecar.

    Returns:
        A ListShard wrapping the data file path (as the existing scatter
        plumbing expects a list of paths).
    """
    if sort_fn is not None:
        captured_sort_fn = sort_fn

        def _sort_key(item):
            return (key_fn(item), captured_sort_fn(item))

    else:
        _sort_key = key_fn

    chunk_size = INTERMEDIATE_CHUNK_SIZE

    buffers: dict[int, list] = defaultdict(list)
    # Per-shard list of (offset, length, n_items) for the sidecar.
    shard_ranges: dict[int, list[tuple[int, int]]] = defaultdict(list)
    per_shard_max_rows: dict[int, int] = defaultdict(int)

    avg_item_bytes: float = 0.0
    sampled_avg = False
    n_chunks_written = 0

    ensure_parent_dir(data_path)
    fs, fs_path = url_to_fs(data_path)

    def _flush(target: int, buf: list) -> None:
        nonlocal avg_item_bytes, sampled_avg, n_chunks_written

        if combiner_fn is not None:
            buf = _apply_combiner(buf, key_fn, combiner_fn)
        buf.sort(key=_sort_key)

        if not sampled_avg and buf:
            sample = buf[: min(len(buf), _SCATTER_SAMPLE_SIZE)]
            total_bytes = sum(len(pickle.dumps(item, protocol=pickle.HIGHEST_PROTOCOL)) for item in sample)
            avg_item_bytes = total_bytes / len(sample)
            sampled_avg = True

        frame = _write_chunk_frame(buf)
        offset = out.tell()
        out.write(frame)
        shard_ranges[target].append((offset, len(frame)))
        per_shard_max_rows[target] = max(per_shard_max_rows[target], len(buf))

        n_chunks_written += 1
        if n_chunks_written % 10 == 0:
            logger.info(
                "[shard %d] Wrote %d scatter chunks so far (latest chunk size: %d items, %d bytes)",
                source_shard,
                n_chunks_written,
                len(buf),
                len(frame),
            )

    with fs.open(fs_path, "wb") as out:
        # Route items, flush per-shard buffers when they hit chunk_size.
        for item in items:
            key = key_fn(item)
            target = deterministic_hash(key) % num_output_shards
            buffers[target].append(item)
            if chunk_size > 0 and len(buffers[target]) >= chunk_size:
                _flush(target, buffers[target])
                buffers[target] = []

        # Flush remaining per-shard buffers in shard order so the file
        # has a stable layout.
        with log_time(f"Flushing remaining buffers for {data_path}"):
            for target, buf in sorted(buffers.items()):
                if buf:
                    _flush(target, buf)

    sidecar: dict = {
        "shards": {str(k): v for k, v in shard_ranges.items()},
        "max_chunk_rows": {str(k): v for k, v in per_shard_max_rows.items() if v > 0},
    }
    if avg_item_bytes > 0:
        sidecar["avg_item_bytes"] = round(avg_item_bytes, 1)

    with log_time(f"Writing scatter meta for {data_path}"):
        _write_scatter_meta(data_path, sidecar)

    return ListShard(refs=[MemChunk(items=[data_path])])
