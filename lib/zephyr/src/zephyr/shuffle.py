# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scatter/shuffle support for Zephyr pipelines.

Each source shard's scatter output is a ``mapper-XXXX.shuffle`` binary file
containing zstd-compressed frames; each frame is a sequence of
``pickle.dump``ed sub-batches of up to ``_SUB_BATCH_SIZE`` items.

Two manifest flavours share the ``.scatter_meta`` suffix in the same
directory:

* ``mapper-XXXX.scatter_meta`` — written by each mapper. Maps
  ``target_shard -> [(offset, length)]`` plus per-shard ``max_chunk_rows``
  and a global ``avg_item_bytes``.
* ``reducer-YYYY.scatter_meta`` — written by the ``CombineMeta`` stage.
  Pre-filtered list of ``{path, ranges, max_chunk_rows, avg_item_bytes}``
  entries for reducer ``YYYY``, so reducers read one file instead of M.

Chunks are fetched via a single ``cat_file`` range GET and streamed through
a zstd decoder, so per-iterator memory stays near-constant — important for
skewed shuffles where the external-sort fan-in opens many chunk iterators.
"""

from __future__ import annotations

import concurrent.futures
import functools
import io
import logging
import os
import pickle
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import cloudpickle
import msgspec
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

_MAPPER_PREFIX = "mapper-"
_REDUCER_PREFIX = "reducer-"
_SCATTER_DATA_SUFFIX = ".shuffle"
_SCATTER_META_SUFFIX = ".scatter_meta"

# Parallelism for manifest reads/writes — GCS GET/PUT-bound.
_MANIFEST_IO_CONCURRENCY = 32
_SCATTER_SAMPLE_SIZE = 100
_SCATTER_READ_BUFFER_FRACTION = 0.25

_ZSTD_COMPRESS_LEVEL = 3
# Items per pickle.dump within a chunk; trades write speed for read memory.
_SUB_BATCH_SIZE = 1024


# ---------------------------------------------------------------------------
# Sidecar / manifest helpers
# ---------------------------------------------------------------------------


def mapper_data_path(stage_dir: str, shard_idx: int) -> str:
    """``{stage_dir}/mapper-0000.shuffle``."""
    return f"{stage_dir}/{_MAPPER_PREFIX}{shard_idx:04d}{_SCATTER_DATA_SUFFIX}"


def mapper_manifest_path(data_path: str) -> str:
    """``mapper-0000.shuffle`` -> ``mapper-0000.scatter_meta``."""
    stem, _ = os.path.splitext(data_path)
    return stem + _SCATTER_META_SUFFIX


def reducer_manifest_path(stage_dir: str, shard_idx: int) -> str:
    """``{stage_dir}/reducer-0000.scatter_meta``."""
    return f"{stage_dir}/{_REDUCER_PREFIX}{shard_idx:04d}{_SCATTER_META_SUFFIX}"


@functools.cache
def _manifest_decoder() -> msgspec.json.Decoder:
    return msgspec.json.Decoder()


@functools.cache
def _manifest_encoder() -> msgspec.json.Encoder:
    return msgspec.json.Encoder()


def _write_mapper_manifest(data_path: str, sidecar: dict) -> None:
    meta_path = mapper_manifest_path(data_path)
    payload = _manifest_encoder().encode(sidecar)
    with log_time(f"Writing mapper manifest for {data_path} to {meta_path}", level=logging.DEBUG):
        with open_url(meta_path, "wb") as f:
            f.write(payload)


@dataclass(frozen=True)
class _SidecarSlice:
    """One reducer's slice of a mapper sidecar (freed eagerly to bound memory)."""

    path: str
    ranges: tuple[tuple[int, int], ...]
    max_chunk_rows: int
    avg_item_bytes: float


def _read_mapper_manifest(path: str) -> dict:
    """Read a mapper's manifest as a dict (``cat_file`` + ``msgspec.json``)."""
    meta_path = mapper_manifest_path(path)
    fs, fs_path = url_to_fs(meta_path)
    return _manifest_decoder().decode(fs.cat_file(fs_path))


def _sidecar_slice_from_manifest(path: str, meta: dict, shard_key: str) -> _SidecarSlice | None:
    ranges_raw = meta.get("shards", {}).get(shard_key)
    if not ranges_raw:
        return None
    max_rows_map = meta.get("max_chunk_rows", {})
    if shard_key not in max_rows_map:
        raise ValueError(f"Manifest for {path} has ranges for shard {shard_key} but no max_chunk_rows entry.")
    if "avg_item_bytes" not in meta:
        raise ValueError(f"Manifest for {path} has ranges for shard {shard_key} but no avg_item_bytes.")
    ranges = tuple((int(off), int(length)) for off, length in ranges_raw)
    return _SidecarSlice(
        path=path,
        ranges=ranges,
        max_chunk_rows=int(max_rows_map[shard_key]),
        avg_item_bytes=float(meta["avg_item_bytes"]),
    )


def _read_sidecar_slice(path: str, shard_key: str) -> _SidecarSlice | None:
    return _sidecar_slice_from_manifest(path, _read_mapper_manifest(path), shard_key)


def _read_sidecar_slices_parallel(scatter_paths: list[str], target_shard: int) -> list[_SidecarSlice]:
    """Read every mapper manifest and return slices for ``target_shard``.

    Used by ``ScatterReader.from_sidecars``; the ``CombineMeta`` stage
    replaces this with a single pre-built manifest per reducer.
    """
    shard_key = str(target_shard)
    ordered: list[_SidecarSlice | None] = [None] * len(scatter_paths)
    with concurrent.futures.ThreadPoolExecutor(max_workers=_MANIFEST_IO_CONCURRENCY) as pool:
        futures = {pool.submit(_read_sidecar_slice, p, shard_key): i for i, p in enumerate(scatter_paths)}
        for fut in concurrent.futures.as_completed(futures):
            idx = futures[fut]
            ordered[idx] = fut.result()
    return [s for s in ordered if s is not None]


def combine_sidecars(
    scatter_paths: list[str],
    num_output_shards: int,
    out_dir: str,
    task_idx: int = 0,
    num_combine_tasks: int = 1,
) -> str:
    """Invert mapper manifests into this task's slice of reducer manifests.

    Task ``task_idx`` of ``num_combine_tasks`` owns target shards in
    ``range(task_idx, num_output_shards, num_combine_tasks)`` — a strided
    slice so every task reads the full M mappers but only holds/writes
    ``ceil(R/K)`` reducers' ranges. Peak memory is the inverted
    ``(R/K) x M`` entry matrix plus the per-parse manifest buffer.
    """
    if num_output_shards < 0:
        raise ValueError(f"num_output_shards must be >= 0, got {num_output_shards}")
    if num_combine_tasks < 1 or not (0 <= task_idx < num_combine_tasks):
        raise ValueError(f"invalid task_idx={task_idx} / num_combine_tasks={num_combine_tasks}")

    owned_targets = set(range(task_idx, num_output_shards, num_combine_tasks))
    per_reducer: dict[int, list[dict]] = defaultdict(list)

    def _extract_entries(path: str) -> list[tuple[int, dict]]:
        meta = _read_mapper_manifest(path)
        shards = meta.get("shards", {})
        max_rows_map = meta.get("max_chunk_rows", {})
        avg_bytes = float(meta.get("avg_item_bytes", 0.0))
        out: list[tuple[int, dict]] = []
        for shard_key, ranges in shards.items():
            if not ranges:
                continue
            target = int(shard_key)
            if target not in owned_targets:
                continue
            if shard_key not in max_rows_map:
                raise ValueError(f"Manifest for {path} has ranges for shard {shard_key} but no max_chunk_rows entry.")
            out.append(
                (
                    target,
                    {
                        "path": path,
                        "ranges": ranges,
                        "max_chunk_rows": int(max_rows_map[shard_key]),
                        "avg_item_bytes": avg_bytes,
                    },
                )
            )
        return out

    with log_time(f"combine_sidecars[{task_idx}/{num_combine_tasks}]: reading {len(scatter_paths)} mapper manifests"):
        with concurrent.futures.ThreadPoolExecutor(max_workers=_MANIFEST_IO_CONCURRENCY) as pool:
            futures = [pool.submit(_extract_entries, p) for p in scatter_paths]
            for fut in concurrent.futures.as_completed(futures):
                for target, entry in fut.result():
                    per_reducer[target].append(entry)

    def _write_reducer_manifest(shard_idx: int) -> None:
        entries = per_reducer.get(shard_idx, [])
        manifest_path = reducer_manifest_path(out_dir, shard_idx)
        payload = _manifest_encoder().encode({"entries": entries})
        ensure_parent_dir(manifest_path)
        with open_url(manifest_path, "wb") as f:
            f.write(payload)

    with log_time(f"combine_sidecars[{task_idx}/{num_combine_tasks}]: writing {len(owned_targets)} reducer manifests"):
        with concurrent.futures.ThreadPoolExecutor(max_workers=_MANIFEST_IO_CONCURRENCY) as pool:
            for _ in pool.map(_write_reducer_manifest, sorted(owned_targets)):
                pass

    logger.info(
        "combine_sidecars[%d/%d]: %d mappers -> %d reducer manifests under %s",
        task_idx,
        num_combine_tasks,
        len(scatter_paths),
        len(owned_targets),
        out_dir,
    )
    return out_dir


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

    Prefer :meth:`from_combined_manifest` (one read per reducer, written by
    ``CombineMeta``); :meth:`from_sidecars` bypasses the combine stage.
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
        """Build a ScatterReader by reading every mapper manifest directly."""
        iterators: list[ScatterFileIterator] = []
        max_rows = 0
        weighted_bytes = 0.0
        total_chunks_for_avg = 0

        with log_time(
            f"Building ScatterReader for target shard {target_shard} "
            f"from {len(scatter_paths)} manifests (concurrency={_MANIFEST_IO_CONCURRENCY})"
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
            "ScatterReader for shard %d: %d files, %d total chunks, max_chunk_rows=%d, avg_item_bytes=%.1f",
            target_shard,
            len(iterators),
            sum(it.chunk_count for it in iterators),
            max_rows,
            avg_item_bytes,
        )
        return cls(iterators=iterators, max_chunk_rows=max_rows, avg_item_bytes=avg_item_bytes)

    @classmethod
    def from_combined_manifest(cls, manifest_path: str) -> ScatterReader:
        """Build a ScatterReader from one pre-built reducer manifest."""
        fs, fs_path = url_to_fs(manifest_path)
        with log_time(f"Reading combined manifest {manifest_path}"):
            payload = _manifest_decoder().decode(fs.cat_file(fs_path))

        entries = payload.get("entries", [])
        iterators: list[ScatterFileIterator] = []
        max_rows = 0
        weighted_bytes = 0.0
        total_chunks_for_avg = 0
        for entry in entries:
            ranges = tuple((int(off), int(length)) for off, length in entry["ranges"])
            iterators.append(ScatterFileIterator(path=entry["path"], chunks=ranges))
            max_rows = max(max_rows, int(entry["max_chunk_rows"]))
            avg = float(entry.get("avg_item_bytes", 0.0))
            if avg > 0:
                count = len(ranges)
                weighted_bytes += avg * count
                total_chunks_for_avg += count

        avg_item_bytes = weighted_bytes / total_chunks_for_avg if total_chunks_for_avg > 0 else 0.0

        logger.info(
            "ScatterReader from combined manifest %s: %d files, %d total chunks, "
            "max_chunk_rows=%d, avg_item_bytes=%.1f",
            manifest_path,
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

        sidecar: dict = {
            "shards": {str(k): v for k, v in self._shard_ranges.items()},
            "max_chunk_rows": {str(k): v for k, v in self._per_shard_max_rows.items() if v > 0},
        }
        if self._avg_item_bytes > 0:
            sidecar["avg_item_bytes"] = round(self._avg_item_bytes, 1)

        with log_time(f"Writing mapper manifest for {self._data_path}"):
            _write_mapper_manifest(self._data_path, sidecar)

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
