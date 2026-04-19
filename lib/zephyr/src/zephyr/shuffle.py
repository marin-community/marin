# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scatter/shuffle support for Zephyr pipelines.

Each source-shard's scatter output is a single binary file containing a
sequence of zstd-compressed frames. Within one chunk's zstd frame, items
are written in sub-batches of ``_SUB_BATCH_SIZE`` — each sub-batch is a
single ``pickle.dump(list_of_items)`` into the zstd stream. This amortises
per-item pickle/zstd dispatch over a sub-batch while still letting the
reader stream sub-batches lazily without materialising the full chunk.

A Parquet sidecar (``.scatter_meta``) stores one row per (target_shard, chunk)
pair with columns ``target_shard``, ``chunk_offset``, ``chunk_length``,
``max_chunk_rows``, and ``avg_item_bytes``. Rows are sorted by ``target_shard``
and written with small row groups (64 rows) so row-group min/max statistics
enable predicate pushdown -- each reducer reads only the 1-2 row groups
that match its shard, skipping ~99% of the sidecar.

On read, each chunk is fetched with a single ``cat_file`` range GET (one
HTTP request, no per-chunk file handle), then streamed via
``pickle.load`` on a length-bounded zstd reader. Per-iterator memory stays
near-constant: one buffered item plus the zstd decoder state plus the
chunk's compressed bytes (typically a few MB). This bound is essential for
skewed shuffles where one reducer pulls disproportionate data and the
external-sort fan-in opens hundreds of chunk iterators at once.
"""

from __future__ import annotations

import concurrent.futures
import io
import logging
import os
import pickle
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import cloudpickle
import pyarrow as pa
import pyarrow.parquet as pq
import zstandard as zstd
from rigging.filesystem import url_to_fs
from rigging.timing import log_time

from zephyr.plan import deterministic_hash
from zephyr.readers import iter_parquet_row_groups
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
_SCATTER_DATA_SUFFIX = ".shuffle"

# Number of parallel sidecar reads each reducer issues when building its
# ScatterReader. Sidecars are small Parquet files (a few KB) and reads are
# GCS GET-bound, so a modest pool keeps latency low without thrashing.
_SIDECAR_READ_CONCURRENCY = 32
# Number of items sampled from the first flush to estimate avg_item_bytes.
_SCATTER_SAMPLE_SIZE = 100
# Fraction of total memory budgeted for read-side decompression buffers.
_SCATTER_READ_BUFFER_FRACTION = 0.25

_ZSTD_COMPRESS_LEVEL = 3
# Items per pickle.dump call within a chunk. Larger = faster (less per-call
# dispatch overhead), smaller = lower per-iterator read memory.
_SUB_BATCH_SIZE = 1024


# ---------------------------------------------------------------------------
# Sidecar format (Parquet with predicate pushdown)
# ---------------------------------------------------------------------------
#
# Each mapper writes a ``.scatter_meta`` Parquet file with one row per
# (target_shard, chunk) pair. Rows are sorted by ``target_shard`` and
# written with ``row_group_size=64`` so row-group min/max statistics
# on ``target_shard`` allow the reader to skip ~99% of groups.
#
# Schema:
#   target_shard:   int32   (which reducer owns this chunk)
#   chunk_offset:   int64   (byte offset into the .shuffle data file)
#   chunk_length:   int32   (byte length of the zstd frame)
#   max_chunk_rows: int32   (max items in any chunk for this target)
#   avg_item_bytes: float64 (pickle-serialized item size estimate)
#
# Reader for target T uses iter_parquet_row_groups with
# equality_predicates={"target_shard": T} to read only the matching
# row group(s).

_SIDECAR_ROW_GROUP_SIZE = 64


def _scatter_meta_path(data_path: str) -> str:
    """``shard-0000.shuffle`` -> ``shard-0000.scatter_meta``."""
    stem, _ = os.path.splitext(data_path)
    return stem + _SCATTER_META_SUFFIX


def _write_scatter_meta(
    data_path: str,
    shard_ranges: dict[int, list[tuple[int, int]]],
    per_shard_max_rows: dict[int, int],
    avg_item_bytes: float,
) -> None:
    """Write a Parquet sidecar with per-chunk rows sorted by target shard."""
    meta_path = _scatter_meta_path(data_path)

    # Flatten into rows sorted by target for good row-group statistics.
    rows_target: list[int] = []
    rows_offset: list[int] = []
    rows_length: list[int] = []
    rows_max_rows: list[int] = []
    rows_avg_bytes: list[float] = []

    for target in sorted(shard_ranges.keys()):
        max_rows = per_shard_max_rows.get(target, 0)
        for chunk_offset, chunk_length in shard_ranges[target]:
            rows_target.append(target)
            rows_offset.append(chunk_offset)
            rows_length.append(chunk_length)
            rows_max_rows.append(max_rows)
            rows_avg_bytes.append(avg_item_bytes)

    table = pa.table(
        {
            "target_shard": pa.array(rows_target, type=pa.int32()),
            "chunk_offset": pa.array(rows_offset, type=pa.int64()),
            "chunk_length": pa.array(rows_length, type=pa.int32()),
            "max_chunk_rows": pa.array(rows_max_rows, type=pa.int32()),
            "avg_item_bytes": pa.array(rows_avg_bytes, type=pa.float64()),
        }
    )

    with log_time(f"Writing scatter meta for {data_path} to {meta_path}", level=logging.DEBUG):
        fs, fs_path = url_to_fs(meta_path)
        with fs.open(fs_path, "wb") as f:
            pq.write_table(
                table,
                f,
                row_group_size=_SIDECAR_ROW_GROUP_SIZE,
                write_statistics=True,
            )


@dataclass(frozen=True)
class _SidecarSlice:
    """One reducer's slice of a mapper sidecar."""

    path: str
    ranges: tuple[tuple[int, int], ...]
    max_chunk_rows: int
    avg_item_bytes: float


def _read_sidecar_slice(path: str, target_shard: int) -> _SidecarSlice | None:
    """Read a Parquet sidecar using row-group predicate pushdown.

    Row groups whose ``target_shard`` min/max statistics exclude the
    target are skipped entirely. Only matching groups are read and
    decoded — typically 1 out of ~100 groups.
    """
    meta_path = _scatter_meta_path(path)

    ranges: list[tuple[int, int]] = []
    max_chunk_rows = 0
    avg_item_bytes = 0.0

    pf = pq.ParquetFile(meta_path)
    for table in iter_parquet_row_groups(
        pf,
        equality_predicates={"target_shard": target_shard},
    ):
        for row in table.to_pylist():
            ranges.append((row["chunk_offset"], row["chunk_length"]))
            max_chunk_rows = max(max_chunk_rows, row["max_chunk_rows"])
            avg_item_bytes = row["avg_item_bytes"]

    if not ranges:
        return None

    return _SidecarSlice(
        path=path,
        ranges=tuple(ranges),
        max_chunk_rows=max_chunk_rows,
        avg_item_bytes=avg_item_bytes,
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
    ordered: list[_SidecarSlice | None] = [None] * len(scatter_paths)
    with concurrent.futures.ThreadPoolExecutor(max_workers=_SIDECAR_READ_CONCURRENCY) as pool:
        futures = {pool.submit(_read_sidecar_slice, p, target_shard): i for i, p in enumerate(scatter_paths)}
        for fut in concurrent.futures.as_completed(futures):
            idx = futures[fut]
            ordered[idx] = fut.result()
    return [s for s in ordered if s is not None]


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
    _fs: Any = None
    _fs_path: str = ""

    def __post_init__(self) -> None:
        if self._fs is None:
            fs, fs_path = url_to_fs(self.path)
            object.__setattr__(self, "_fs", fs)
            object.__setattr__(self, "_fs_path", fs_path)

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    def __iter__(self) -> Iterator:
        for chunk_iter in self.get_chunk_iterators():
            yield from chunk_iter

    def get_chunk_iterators(self) -> Iterator[Iterator]:
        """Yield one lazy iterator per chunk, in write order."""
        for offset, length in self.chunks:
            yield _iter_chunk(self._fs, self._fs_path, offset, length)


def _iter_chunk(fs: Any, fs_path: str, offset: int, length: int) -> Iterator:
    """Fetch one chunk's compressed bytes via cat_file and stream items.

    Each chunk is a zstd frame containing a sequence of pickled sub-batches
    (lists of up to ``_SUB_BATCH_SIZE`` items). The reader streams one
    sub-batch at a time, so per-iterator memory is bounded by the
    sub-batch size plus the chunk's compressed bytes.
    """
    blob = fs.cat_file(fs_path, start=offset, end=offset + length)
    with zstd.ZstdDecompressor().stream_reader(io.BytesIO(blob)) as reader:
        while True:
            try:
                sub_batch = pickle.load(reader)
            except EOFError:
                return
            yield from sub_batch


# ---------------------------------------------------------------------------
# ScatterReader: built from manifest, fed to Reduce
# ---------------------------------------------------------------------------


class ScatterReader:
    """All scatter chunks for one target shard, across all source files.

    Construct via :meth:`from_sidecars` for production use, or pass fields
    directly for testing.
    """

    def __init__(
        self,
        iterators: list[ScatterFileIterator],
        max_chunk_rows: int,
        avg_item_bytes: float,
    ) -> None:
        self.iterators = iterators
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
        iterators: list[ScatterFileIterator] = []
        max_rows = 0
        weighted_bytes = 0.0
        total_chunks_for_avg = 0

        with log_time(
            f"Building ScatterReader for target shard {target_shard} "
            f"from {len(scatter_paths)} sidecars (concurrency={_SIDECAR_READ_CONCURRENCY})"
        ):
            for slice_ in _read_sidecar_slices_parallel(scatter_paths, target_shard):
                iterators.append(ScatterFileIterator(path=slice_.path, chunks=slice_.ranges))
                max_rows = max(max_rows, slice_.max_chunk_rows)
                if slice_.avg_item_bytes > 0:
                    count = len(slice_.ranges)
                    weighted_bytes += slice_.avg_item_bytes * count
                    total_chunks_for_avg += count

        avg_item_bytes = weighted_bytes / total_chunks_for_avg if total_chunks_for_avg > 0 else 0.0

        logger.info(
            "ScatterReader for shard %d: %d files, %d total chunks, " "max_chunk_rows=%d, avg_item_bytes=%.1f",
            target_shard,
            len(iterators),
            sum(it.chunk_count for it in iterators),
            max_rows,
            avg_item_bytes,
        )
        return cls(iterators=iterators, max_chunk_rows=max_rows, avg_item_bytes=avg_item_bytes)

    def __iter__(self) -> Iterator:
        for it in self.iterators:
            yield from it

    def get_iterators(self) -> Iterator[Iterator]:
        for it in self.iterators:
            yield from it.get_chunk_iterators()

    @property
    def total_chunks(self) -> int:
        return sum(it.chunk_count for it in self.iterators)

    @property
    def max_compressed_chunk_bytes(self) -> int:
        """Return the largest compressed chunk length across all files."""
        if not self.iterators:
            return 0
        return max(length for file_iter in self.iterators for _, length in file_iter.chunks)

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
        total_compressed = sum(length for it in self.iterators for _, length in it.chunks)
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
    sub-batch is written as a single ``cloudpickle.dump(sublist)`` into the
    same zstd stream. This batches per-call dispatch overhead while
    keeping per-iterator read memory bounded by the sub-batch size.
    """
    raw = io.BytesIO()
    cctx = zstd.ZstdCompressor(level=_ZSTD_COMPRESS_LEVEL)
    with cctx.stream_writer(raw, closefd=False) as zf:
        for i in range(0, len(items), _SUB_BATCH_SIZE):
            cloudpickle.dump(items[i : i + _SUB_BATCH_SIZE], zf, protocol=pickle.HIGHEST_PROTOCOL)
    return raw.getvalue()


class ScatterWriter:
    """Writes items to a scatter data file with zstd-compressed chunks.

    Items are routed to target shards by ``key_fn``, buffered, optionally
    combined and sorted, then flushed as zstd frames. A JSON sidecar is
    written on close.
    """

    def __init__(
        self,
        data_path: str,
        key_fn: Callable,
        num_output_shards: int,
        source_shard: int = 0,
        sort_fn: Callable | None = None,
        combiner_fn: Callable | None = None,
    ) -> None:
        self._data_path = data_path
        self._key_fn = key_fn
        self._num_output_shards = num_output_shards
        self._source_shard = source_shard
        self._combiner_fn = combiner_fn
        self._chunk_size = INTERMEDIATE_CHUNK_SIZE

        if sort_fn is not None:
            captured_sort_fn = sort_fn

            def _sort_key(item: Any) -> Any:
                return (key_fn(item), captured_sort_fn(item))

            self._sort_key = _sort_key
        else:
            self._sort_key = key_fn

        self._buffers: dict[int, list] = defaultdict(list)
        self._shard_ranges: dict[int, list[tuple[int, int]]] = defaultdict(list)
        self._per_shard_max_rows: dict[int, int] = defaultdict(int)
        self._avg_item_bytes: float = 0.0
        self._sampled_avg = False
        self._n_chunks_written = 0

        ensure_parent_dir(data_path)
        fs, fs_path = url_to_fs(data_path)
        self._out = fs.open(fs_path, "wb")

    def _flush(self, target: int, buf: list) -> None:
        if self._combiner_fn is not None:
            buf = _apply_combiner(buf, self._key_fn, self._combiner_fn)
        buf.sort(key=self._sort_key)

        if not self._sampled_avg and buf:
            sample = buf[: min(len(buf), _SCATTER_SAMPLE_SIZE)]
            total_bytes = sum(len(pickle.dumps(item, protocol=pickle.HIGHEST_PROTOCOL)) for item in sample)
            self._avg_item_bytes = total_bytes / len(sample)
            self._sampled_avg = True

        frame = _write_chunk_frame(buf)
        offset = self._out.tell()
        self._out.write(frame)
        self._shard_ranges[target].append((offset, len(frame)))
        self._per_shard_max_rows[target] = max(self._per_shard_max_rows[target], len(buf))

        self._n_chunks_written += 1
        if self._n_chunks_written % 10 == 0:
            logger.info(
                "[shard %d] Wrote %d scatter chunks so far (latest chunk size: %d items, %d bytes)",
                self._source_shard,
                self._n_chunks_written,
                len(buf),
                len(frame),
            )

    def write(self, item: Any) -> None:
        """Route a single item to its target shard buffer, flushing when full."""
        key = self._key_fn(item)
        target = deterministic_hash(key) % self._num_output_shards
        self._buffers[target].append(item)
        if self._chunk_size > 0 and len(self._buffers[target]) >= self._chunk_size:
            self._flush(target, self._buffers[target])
            self._buffers[target] = []

    def close(self) -> ListShard:
        """Flush remaining buffers, write sidecar, return ListShard."""
        with log_time(f"Flushing remaining buffers for {self._data_path}"):
            for target, buf in sorted(self._buffers.items()):
                if buf:
                    self._flush(target, buf)
        self._out.close()

        with log_time(f"Writing scatter meta for {self._data_path}"):
            _write_scatter_meta(
                self._data_path,
                shard_ranges=dict(self._shard_ranges),
                per_shard_max_rows=dict(self._per_shard_max_rows),
                avg_item_bytes=self._avg_item_bytes,
            )

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
    """Route items to target shards, buffer, sort, and append zstd chunks.

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
    )
    for item in items:
        writer.write(item)
    return writer.close()
