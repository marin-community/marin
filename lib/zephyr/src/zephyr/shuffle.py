# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scatter/shuffle support for Zephyr pipelines.

Each source-shard's scatter output is a set of zstd-compressed Parquet files,
one combined file per flush (``c{chunk:04d}.parquet``) containing all target
shards' data sorted by ``(_SHARD_COL, _SORT_KEY_COL)``.  A msgpack sidecar
(``metadata.msgpack``) records ``files -> [{path, schema}, ...]`` plus exact
per-target-shard payload bytes and row counts (``shard_bytes``,
``shard_rows``). A reducer's :class:`ScatterReader` sums those across every
source shard's sidecar to get its own target's exact payload bytes and
``avg_item_bytes``, fed into :func:`zephyr.memory_budget.read_merge_fan_in`
to size the merge plan. Row width can vary sharply by target shard, so this
average is computed per target rather than across a mapper's whole output.

On the read side, each reducer scans only its target shard via
``pl.scan_parquet(path).filter(pl.col(_SHARD_COL) == target).drop(_SHARD_COL)``.
Polars predicate pushdown with row-group statistics skips non-matching row
groups via byte-range GETs, so each reducer reads roughly 1/N of each file.
The resulting LazyFrames are merged via ``_merge_sorted_frames``: a fully
streaming merge that spills to Parquet runs with ``sink_parquet`` only above
:func:`zephyr.memory_budget.read_merge_fan_in`, and otherwise merges directly.

Write-side memory is bounded by buffer estimated size: when the sum of
``DataFrame.estimated_size()`` across buffered frames exceeds
:func:`zephyr.memory_budget.write_flush_threshold_bytes`, all buffers are
flushed together into one combined file. See ``zephyr/memory_budget.py`` for
the shared model behind both thresholds.

Routing columns (``__zephyr_shard__``, ``__zephyr_sort_key__``) are added
in ``_items_to_dataframe``; ``__zephyr_shard__`` is stripped on read,
``__zephyr_sort_key__`` is consumed by the merge and stripped after.
"""

import concurrent.futures
import gc
import io
import logging
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

import cloudpickle
import humanfriendly
import msgspec
import polars as pl
import psutil
import pyarrow as pa
from iris.env_resources import TaskResources
from rigging.filesystem.factory import open_url, url_to_fs
from rigging.filesystem.s3_errors import is_transient_s3_error
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import RateLimiter, log_time, retry_with_backoff

from zephyr import memory_budget
from zephyr.parquet_scan import scan_parquet
from zephyr.shard_keys import encode_key, hash_encoded_key
from zephyr.worker_context import _worker_ctx_var
from zephyr.writers import ensure_parent_dir

logger = logging.getLogger(__name__)


def _process_rss_bytes() -> int:
    return psutil.Process().memory_info().rss


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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SCATTER_METADATA_FILENAME = "metadata.msgpack"

# Number of parallel sidecar reads each reducer issues while building its
# ScatterReader. These reads are GCS
# GET-bound, so a modest pool keeps latency low without thrashing. The bound is
# per task, and a wave multiplies it: 2,048 reducers at 32 offered ~65,000
# simultaneous connections, more than a pod has ephemeral ports (#8402).
_SIDECAR_READ_CONCURRENCY = 8

_PROGRESS_LOG_INTERVAL_SECONDS = 60.0
# Bound Parquet footer size when a shuffle has thousands of target shards. One
# row group per target gives ideal predicate pruning, but makes every reducer
# read multi-megabyte footers from every mapper chunk before it can read data.
_SCATTER_MAX_ROW_GROUPS_PER_CHUNK = 512

# Helper column names injected by _items_to_dataframe and stripped before
# writing to disk.  Both are internal implementation details; user schemas must
# not collide with these names.
_SHARD_COL = "__zephyr_shard__"
_SORT_KEY_COL = "__zephyr_sort_key__"
# A cloudpickle-serialized Python object representing the item
_PAYLOAD_COL = "__payload__"
# Temporary flat columns folded into the _SORT_KEY_COL struct during
# _items_to_dataframe; never present in written chunks.
_KEY_TMP_COL = "__zephyr_key_tmp__"
_SORT_VALUE_TMP_COL = "__zephyr_sort_value_tmp__"

# Python items consumed before creating a DataFrame.
_DATAFRAME_ROW_COUNT = 1000
# Threshold for triggering a gc.collect() after a flush.
_GC_FLUSH_SIZE_THRESHOLD_BYTES = 8 * 1024 * 1024


def _task_memory_bytes() -> int:
    ctx = _worker_ctx_var.get()
    if ctx is not None and ctx.task_memory_bytes > 0:
        return ctx.task_memory_bytes

    memory_bytes = TaskResources.from_environment().memory_bytes
    if memory_bytes <= 0:
        logger.warning("No task memory is available. Using a 1 GiB memory budget.")
        return 1024 * 1024 * 1024
    return memory_bytes


def _dataframe_to_items(df: pl.DataFrame) -> Iterator[Any]:
    """Yield Python items from a DataFrame, stripping routing columns and deserializing payloads."""
    for p in df[_PAYLOAD_COL].to_list():
        yield cloudpickle.loads(p)


def _columns_to_dataframe(
    payloads: list[bytes],
    shards: list[int],
    key_bytes: list[bytes],
    sort_values: list[Any],
) -> pl.DataFrame:
    """Build the scatter DataFrame from pre-computed flat columns.

    The sort-key struct is folded from two flat columns rather than per-row
    Python dicts: series construction from homogeneous lists is the native
    fast path, and ``pl.struct`` over existing columns is cheap. Field order
    (key first) drives the (key, sort_value) sort order.

    ``key_bytes`` must be pre-encoded via :func:`~zephyr.shard_keys.encode_key` so that
    ``_KEY_TMP_COL`` is always ``Binary`` — preventing struct schema mismatches
    when different mapper shards produce keys of different Python types.
    """
    try:
        return pl.DataFrame(
            {
                _PAYLOAD_COL: pl.Series(payloads, dtype=pl.Binary),
                _SHARD_COL: pl.Series(shards, dtype=pl.Int32),
                _KEY_TMP_COL: pl.Series(key_bytes, dtype=pl.Binary),
                _SORT_VALUE_TMP_COL: sort_values,
            }
        ).select(
            _PAYLOAD_COL,
            _SHARD_COL,
            pl.struct(
                pl.col(_KEY_TMP_COL).alias("key"),
                pl.col(_SORT_VALUE_TMP_COL).alias("sort_value"),
            ).alias(_SORT_KEY_COL),
        )
    except (TypeError, pl.exceptions.InvalidOperationError) as err:
        # Non-serializable sort_values surface as TypeError from Series construction
        # or InvalidOperationError ("nested objects are not allowed") when the
        # sort_value column lands as Object dtype and pl.struct rejects it.
        raise ValueError("sort_fn must return an Arrow-serializable object.") from err


def _items_to_dataframe(
    items: list[Any],
    key_fn: Callable,
    sort_fn: Callable | None,
    num_output_shards: int,
) -> pl.DataFrame:
    """Convert a list of Python items to a DataFrame with routing columns.

    Cloudpickle-serializes items into ``_PAYLOAD_COL`` and adds ``_SHARD_COL``
    (int32 target shard index) and ``_SORT_KEY_COL``. This is the adapter
    between Python-item pipelines and the DataFrame-based
    :class:`ScatterWriter`; DataFrame-native pipelines can feed the writer
    directly.

    ``num_output_shards=0`` means the caller assigns ``_SHARD_COL`` itself (the
    combiner path in :meth:`ScatterWriter._flush`, whose rows are already
    routed); routing is then skipped rather than computed and discarded.
    """
    shards: list[int] = []
    key_bytes: list[bytes] = []
    sort_values: list[Any] = []
    for item in items:
        key = key_fn(item)
        try:
            kb = encode_key(key)
        except TypeError as err:
            raise ValueError(f"key_fn must return a msgpack-serializable object; got {type(key).__name__!r}.") from err
        # Route from the bytes we just encoded: deterministic_hash(key) would
        # msgpack-encode the same key a second time for every scattered item.
        shards.append(hash_encoded_key(kb) % num_output_shards if num_output_shards > 0 else 0)
        key_bytes.append(kb)
        sort_values.append(sort_fn(item) if sort_fn is not None else None)
    payloads = [cloudpickle.dumps(item) for item in items]
    return _columns_to_dataframe(payloads, shards, key_bytes, sort_values)


class _SidecarFilesystem(Protocol):
    """A protocol because ``url_to_fs`` returns a bare fsspec filesystem for
    ``s3://`` and a ``CrossRegionGuardedFS`` wrapper for ``gs://``, which share
    no base class."""

    def _strip_protocol(self, path: str) -> str: ...

    def cat_file(self, path: str) -> bytes: ...


@dataclass(frozen=True)
class _ChunkFile:
    """A scatter Parquet file and the schema recorded when it was written."""

    path: str
    schema: pl.Schema

    _path_field: ClassVar[str] = "path"
    _schema_field: ClassVar[str] = "schema"

    def to_metadata(self) -> dict[str, str | bytes]:
        return {
            self._path_field: self.path,
            self._schema_field: self.schema.to_arrow().serialize().to_pybytes(),
        }

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> "_ChunkFile":
        schema = pa.ipc.read_schema(pa.BufferReader(metadata[cls._schema_field]))
        return cls(path=str(metadata[cls._path_field]), schema=pl.Schema(schema))


@dataclass(frozen=True)
class _FrameWithSchema:
    frame: pl.LazyFrame
    schema: pl.Schema


@dataclass(frozen=True)
class _Sidecar:
    """One mapper's scatter metadata (``metadata.msgpack``).

    ``files`` lists the combined Parquet paths and schemas written during flushes;
    each file contains data for all target shards sorted by
    ``(_SHARD_COL, _SORT_KEY_COL)``.
    ``shard_bytes`` and ``shard_rows`` map target shard index to the exact
    payload bytes and row count written for that shard across all files, used
    by reducers to size the merge-memory plan. ``path`` is the mapper output
    directory the sidecar lives under.
    """

    path: str
    files: list[_ChunkFile]
    shard_bytes: dict[int, int]
    shard_rows: dict[int, int]

    _encoder: ClassVar[msgspec.msgpack.Encoder] = msgspec.msgpack.Encoder()
    _decoder: ClassVar[msgspec.msgpack.Decoder] = msgspec.msgpack.Decoder()
    _files_field: ClassVar[str] = "files"
    _shard_bytes_field: ClassVar[str] = "shard_bytes"
    _shard_rows_field: ClassVar[str] = "shard_rows"

    @staticmethod
    def meta_path(data_path: str) -> str:
        return f"{data_path}{_SCATTER_METADATA_FILENAME}"

    def target_bytes(self, target_shard: int) -> int:
        return self.shard_bytes.get(target_shard, 0)

    def target_rows(self, target_shard: int) -> int:
        return self.shard_rows.get(target_shard, 0)

    def write(self) -> None:
        """Serialize this sidecar to ``path/metadata.msgpack``."""
        meta_path = self.meta_path(self.path)
        payload = self._encoder.encode(
            {
                self._files_field: [file.to_metadata() for file in self.files],
                self._shard_bytes_field: {str(k): v for k, v in self.shard_bytes.items()},
                self._shard_rows_field: {str(k): v for k, v in self.shard_rows.items()},
            }
        )
        with log_time(f"Writing scatter meta for {self.path} to {meta_path}", level=logging.DEBUG):
            StoragePath(meta_path).write_bytes(payload)

    @classmethod
    def read(cls, fs: _SidecarFilesystem, data_path: str) -> "_Sidecar | None":
        """Load one non-empty sidecar, returning ``None`` for an empty writer."""
        source_path = cls.meta_path(data_path)
        meta_path = fs._strip_protocol(source_path)
        # Avoid buffered-file overhead for these small payloads.
        try:
            payload = retry_with_backoff(
                lambda: fs.cat_file(meta_path),
                retryable=is_transient_s3_error,
                max_attempts=4,
                operation=f"read scatter sidecar {source_path}",
            )
        except Exception as error:
            error.add_note(f"while reading scatter sidecar {source_path}")
            raise
        data = cls._decoder.decode(payload)
        files = data.get(cls._files_field, [])
        if not files:
            return None
        raw_shard_bytes = data.get(cls._shard_bytes_field, {})
        raw_shard_rows = data.get(cls._shard_rows_field, {})
        return cls(
            path=data_path,
            files=[_ChunkFile.from_metadata(file) for file in files],
            shard_bytes={int(k): int(v) for k, v in raw_shard_bytes.items()},
            shard_rows={int(k): int(v) for k, v in raw_shard_rows.items()},
        )

    @classmethod
    def read_all(cls, scatter_paths: list[str]) -> list["_Sidecar"]:
        """Read every non-empty sidecar concurrently in input order."""
        if not scatter_paths:
            return []
        ordered: list[_Sidecar | None] = [None] * len(scatter_paths)
        # Resolve before creating the pool so worker threads share one client.
        fs, _ = url_to_fs(cls.meta_path(scatter_paths[0]))
        with concurrent.futures.ThreadPoolExecutor(max_workers=_SIDECAR_READ_CONCURRENCY) as pool:
            futures = {pool.submit(cls.read, fs, p): i for i, p in enumerate(scatter_paths)}
            for fut in concurrent.futures.as_completed(futures):
                ordered[futures[fut]] = fut.result()
        return [s for s in ordered if s is not None]


def _unify_frame_schemas(frames: list[_FrameWithSchema]) -> list[pl.LazyFrame]:
    """Cast frames to a common supertype schema for sorted merging."""
    if len(frames) <= 1:
        return [frame.frame for frame in frames]
    if all(frame.schema == frames[0].schema for frame in frames[1:]):
        return [frame.frame for frame in frames]
    # Build the supertype from sidecar schemas so drift such as Null versus
    # Int64 is resolved without reading the Parquet footer for every input.
    unified = pl.concat([pl.DataFrame(schema=frame.schema) for frame in frames], how="diagonal_relaxed").schema
    return [frame.frame.cast(dict(unified)) for frame in frames]


def _fan_in_groups(frames: list[pl.LazyFrame], fan_in: int) -> list[list[pl.LazyFrame]]:
    """Split frames into consecutive groups of at most fan_in, preserving order."""
    return [frames[i : i + fan_in] for i in range(0, len(frames), fan_in)]


def _merge_sorted_frames(
    frames: list[pl.LazyFrame],
    sort_key: str,
    external_sort_dir: str,
    fan_in: int,
    shard: int,
) -> Iterator[pl.DataFrame]:
    """Merge sorted LazyFrames, spilling to Parquet runs only above ``fan_in``.

    Repeatedly spills groups of at most ``fan_in`` frames to sorted
    zstd-compressed Parquet runs under ``external_sort_dir`` until at most
    ``fan_in`` frames remain, then streams the final merge of those frames.
    An input with ``len(frames) <= fan_in`` never touches ``external_sort_dir``
    at all — it goes straight to the same streaming merge a plain
    ``pl.merge_sorted`` call would produce. Deletes run files after
    completion or an error.

    Args:
        frames: LazyFrames already sorted ascending on ``sort_key``. Order
            within the list does not matter (:func:`polars.merge_sorted`
            merges by key, not position).
        sort_key: Column name to merge on. Frames must already be sorted by
            this key ascending.
        external_sort_dir: Directory or URL prefix for spill files (e.g. a
            temp dir or ``gs://.../stage1-external-sort/shard-NNNN``); only
            accessed if a spill is actually needed.
        fan_in: Maximum frames merged in any one pass; bounds peak memory.
        shard: Target shard id for log messages only.

    Yields:
        DataFrame batches in ``sort_key`` order.
    """
    if len(frames) == 0:
        return
    if fan_in < memory_budget.MIN_MERGE_FAN_IN:
        raise ValueError(f"fan_in must be at least {memory_budget.MIN_MERGE_FAN_IN}, got {fan_in}")

    # Created lazily, on the first actual spill, so a shard that fits within
    # fan_in never pays for a filesystem round trip to external_sort_dir.
    spill_dir: StoragePath | None = None
    spill_files: set[StoragePath] = set()

    try:
        prior_runs: list[StoragePath] = []
        pass_index = 0
        while len(frames) > fan_in:
            if spill_dir is None:
                spill_dir = StoragePath(external_sort_dir)
                spill_dir.mkdirs()

            logger.info(
                "[shard %d] External sort: pass %d merging %d frames with fan_in=%d",
                shard,
                pass_index,
                len(frames),
                fan_in,
            )
            groups = _fan_in_groups(frames, fan_in)
            runs: list[StoragePath] = []
            for run_index, group in enumerate(groups):
                run_name = f"pass-{pass_index:04d}-run-{run_index:04d}.spill"
                run = spill_dir / run_name
                spill_files.add(run)
                with run.open("wb") as output:
                    pl.merge_sorted(group, key=sort_key).sink_parquet(output, compression="zstd")
                runs.append(run)

            for prior_run in prior_runs:
                prior_run.rm()
                spill_files.remove(prior_run)

            frames = [scan_parquet(str(run)) for run in runs]
            prior_runs = runs
            pass_index += 1

        logger.info("[shard %d] Final merge of %d frames (%d spill pass(es))", shard, len(frames), pass_index)
        yield from pl.merge_sorted(frames, key=sort_key).collect_batches()
    finally:
        if spill_files:
            try:
                for spill_file in sorted(spill_files, key=str):
                    spill_file.rm()
            except Exception:
                logger.warning("Failed to delete external-sort run files under %s", spill_dir, exc_info=True)


# ---------------------------------------------------------------------------
# ScatterReader: built from manifest, fed to Reduce
# ---------------------------------------------------------------------------


class ScatterReader:
    """All scatter chunks for one target shard, across all source shards.

    ``_chunk_files`` lists every combined Parquet file the mappers wrote. Each
    file holds rows for *all* target shards, so :meth:`get_frames` filters each
    scan down to ``_target_shard``.

    Construct via :meth:`from_sidecars` for production use, or pass fields
    directly for testing.
    """

    def __init__(
        self,
        chunk_files: list[_ChunkFile],
        target_shard: int,
        avg_item_bytes: float,
        shard_payload_bytes: float = 0.0,
    ) -> None:
        self._chunk_files = chunk_files
        self._target_shard = target_shard
        self.avg_item_bytes = avg_item_bytes
        self.shard_payload_bytes = shard_payload_bytes

    @classmethod
    def from_sidecars(cls, scatter_paths: list[str], target_shard: int) -> "ScatterReader":
        """Build a ScatterReader by reading per-mapper sidecars directly.

        Each reducer reads every mapper's ``metadata.msgpack`` sidecar in parallel
        and filters for its own ``target_shard``. No coordinator-written manifest
        is needed, which eliminates a serialization bottleneck when there are
        thousands of mappers.
        """
        chunk_files: list[_ChunkFile] = []
        shard_payload_bytes = 0.0
        shard_payload_rows = 0

        with log_time(
            f"Building ScatterReader for target shard {target_shard} "
            f"from {len(scatter_paths)} sidecars (concurrency={_SIDECAR_READ_CONCURRENCY})"
        ):
            sidecars = _Sidecar.read_all(scatter_paths)
            for sidecar in sidecars:
                chunk_files.extend(sidecar.files)
                shard_payload_bytes += sidecar.target_bytes(target_shard)
                shard_payload_rows += sidecar.target_rows(target_shard)

        # Computed from this target's own exact bytes and row count, not a
        # mapper-wide average, so row width that varies by target (e.g. a
        # skewed shuffle key) doesn't bias the merge-memory prediction.
        avg_item_bytes = shard_payload_bytes / shard_payload_rows if shard_payload_rows > 0 else 0.0

        logger.info(
            "ScatterReader for shard %d: %d source shards, %d total chunks, "
            "avg_item_bytes=%.1f, shard_payload_bytes=%.0f",
            target_shard,
            len(sidecars),
            len(chunk_files),
            avg_item_bytes,
            shard_payload_bytes,
        )
        return cls(
            chunk_files=chunk_files,
            target_shard=target_shard,
            avg_item_bytes=avg_item_bytes,
            shard_payload_bytes=shard_payload_bytes,
        )

    def get_frames(self) -> list[pl.LazyFrame]:
        frames: list[_FrameWithSchema] = []
        for chunk_file in self._chunk_files:
            frame = (
                scan_parquet(chunk_file.path, schema=chunk_file.schema)
                .filter(pl.col(_SHARD_COL) == self._target_shard)
                .drop(_SHARD_COL)
            )
            schema = pl.Schema({name: dtype for name, dtype in chunk_file.schema.items() if name != _SHARD_COL})
            frames.append(_FrameWithSchema(frame=frame, schema=schema))
        return _unify_frame_schemas(frames)

    @property
    def total_chunks(self) -> int:
        return len(self._chunk_files)

    def merge_sorted_chunks(self, external_sort_dir: str) -> Iterator[Any]:
        """Merge sorted chunks using k-way merge, yielding items in global sort order.

        Each chunk file is assumed to be sorted by ``_SORT_KEY_COL`` (key plus optional
        secondary sort). Performs a k-way merge across all chunks.
        Args:
            external_sort_dir: Directory for intermediate run files, used only
                if the shard's chunk count exceeds the computed fan-in budget.

        Yields:
            Deserialized Python items in merged sort order.
        """

        with pl.Config() as polars_config:
            polars_config.set_streaming_chunk_size(memory_budget.STREAMING_CHUNK_SIZE_ROWS)

            if self.total_chunks == 0:
                return
            if self.shard_payload_bytes == 0:
                return

            frames = self.get_frames()
            memory_bytes = _task_memory_bytes()
            baseline_rss_bytes = _process_rss_bytes()
            polars_threads = pl.thread_pool_size()
            fan_in = memory_budget.read_merge_fan_in(
                memory_bytes,
                baseline_rss_bytes,
                self.avg_item_bytes,
                self.total_chunks,
                self.shard_payload_bytes,
                polars_threads,
            )
            logger.info(
                "[shard %d] Merging %d chunks with fan_in=%d "
                "(baseline_rss=%s, shard_payload_bytes=%s, avg_item_bytes=%.1f, polars_threads=%d)",
                self._target_shard,
                self.total_chunks,
                fan_in,
                humanfriendly.format_size(baseline_rss_bytes, binary=True),
                humanfriendly.format_size(self.shard_payload_bytes, binary=True),
                self.avg_item_bytes,
                polars_threads,
            )

            batches = _merge_sorted_frames(
                frames=frames,
                sort_key=_SORT_KEY_COL,
                external_sort_dir=external_sort_dir,
                fan_in=fan_in,
                shard=self._target_shard,
            )

            for batch in batches:
                yield from _dataframe_to_items(batch)


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

    Accepts routing-column DataFrames (see ``_items_to_dataframe`` for the
    Python-items adapter) and buffers them as a frame list — appends are free,
    and the frames are combined with one concat per flush. Buffering frames
    keeps the interface ready for DataFrame/RecordBatch-native pipelines.

    Each flush writes a single ``c{chunk:04d}.parquet`` file sorted by
    ``[_SHARD_COL, _SORT_KEY_COL]`` with bounded, target-local row groups so
    Polars predicate pushdown skips most unrelated data without creating
    unbounded Parquet footers.

    Flushing is estimated-size-based: when the sum of ``DataFrame.estimated_size()``
    across buffered frames exceeds :func:`zephyr.memory_budget.write_flush_threshold_bytes`,
    all buffered frames are flushed together into one combined file.
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

        self._source_shard = source_shard
        self._combiner_fn = combiner_fn
        self._memory_available_bytes = _task_memory_bytes()
        self._flush_threshold_bytes: int | None = None

        # Buffered DataFrames, combined into one file per flush. Buffering
        # frames (not Python items) keeps the writer format-agnostic: a future
        # RecordBatch/DataFrame-native pipeline can feed frames directly.
        self._frames: list[pl.DataFrame] = []
        self._chunk_files: list[_ChunkFile] = []
        # Payload bytes and row counts written per target shard, recorded in
        # the sidecar so reducers know their own shard's exact data size and
        # row width for the merge-memory plan without opening any chunk files.
        self._shard_bytes: defaultdict[int, int] = defaultdict(int)
        self._shard_rows: defaultdict[int, int] = defaultdict(int)
        self._total_bytes_written: int = 0
        self._total_rows_written: int = 0
        self._n_chunks_written = 0
        # Throttles the per-flush progress log so high-fanout workloads don't log too often
        self._progress_log_limiter = RateLimiter(interval_seconds=_PROGRESS_LOG_INTERVAL_SECONDS)
        # Running estimated_size() total of unflushed frames; reset to 0 on flush.
        self._buffer_estimated_bytes: int = 0

        ensure_parent_dir(self._data_path)
        self._result: ListShard | None = None

    def _flush(self) -> None:
        """Flush the accumulated buffer into one combined Parquet file sorted by [_SHARD_COL, _SORT_KEY_COL]."""
        if not self._frames:
            return

        buffer = pl.concat(self._frames, how="vertical_relaxed", rechunk=False)
        self._frames = []
        self._buffer_estimated_bytes = 0

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
                return
            buffer = pl.concat(frames, how="vertical_relaxed", rechunk=True)

        buffer_sorted = buffer.sort([_SHARD_COL, _SORT_KEY_COL])
        del buffer

        flushed_bytes = int(buffer_sorted.estimated_size())
        self._total_bytes_written += flushed_bytes
        self._total_rows_written += len(buffer_sorted)
        shard_sizes = buffer_sorted.group_by(_SHARD_COL).agg(
            pl.col(_PAYLOAD_COL).bin.size().sum().alias("bytes"), pl.len().alias("rows")
        )
        for shard_val, nbytes, nrows in shard_sizes.iter_rows():
            self._shard_bytes[shard_val] += int(nbytes)
            self._shard_rows[shard_val] += int(nrows)

        # Keep target shards local to as few row groups as practical so Polars
        # predicate pushdown can skip unrelated data. Cap the group count because
        # every reducer must read every chunk footer before applying that filter.
        num_targets = buffer_sorted[_SHARD_COL].n_unique()
        num_row_groups = min(num_targets, _SCATTER_MAX_ROW_GROUPS_PER_CHUNK)
        row_group_size = max(1, math.ceil(len(buffer_sorted) / num_row_groups))
        chunk_path = f"{self._data_path}c{self._n_chunks_written:04d}.parquet"
        # Ideally we'd call write_parquet directly with the GCS path, but it occationally fails with a generic error.
        buf = io.BytesIO()
        buffer_sorted.write_parquet(buf, compression="zstd", row_group_size=row_group_size)
        with open_url(chunk_path, "wb") as f:
            # getbuffer() views the serialized chunk in place; getvalue() would copy the
            # whole compressed payload, and this flush is exactly what memory_budget.py
            # sizes peak RSS against.
            f.write(buf.getbuffer())

        self._chunk_files.append(_ChunkFile(path=chunk_path, schema=buffer_sorted.schema))
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
        if flushed_bytes >= _GC_FLUSH_SIZE_THRESHOLD_BYTES:
            gc.collect()

    def write(self, df: pl.DataFrame) -> None:
        """Buffer a DataFrame, flushing on memory pressure.

        The DataFrame must contain ``_SHARD_COL`` (int32) and ``_SORT_KEY_COL``
        columns as produced by ``_items_to_dataframe``.
        """
        if len(df) == 0:
            return

        if self._flush_threshold_bytes is None:
            baseline_rss_bytes = _process_rss_bytes()
            self._flush_threshold_bytes = memory_budget.write_flush_threshold_bytes(
                self._memory_available_bytes, baseline_rss_bytes
            )
            logger.info(
                "[shard %d] Scatter memory baseline %s; flush threshold %s",
                self._source_shard,
                humanfriendly.format_size(baseline_rss_bytes, binary=True),
                humanfriendly.format_size(self._flush_threshold_bytes, binary=True),
            )

        self._frames.append(df)
        self._buffer_estimated_bytes += int(df.estimated_size())

        if self._buffer_estimated_bytes > self._flush_threshold_bytes:
            logger.info(
                "[shard %d] Buffer estimated at %s (threshold %s); flushing scatter buffers",
                self._source_shard,
                humanfriendly.format_size(self._buffer_estimated_bytes, binary=True),
                humanfriendly.format_size(self._flush_threshold_bytes, binary=True),
            )
            self._flush()

    def close(self) -> ListShard:
        """Flush remaining buffers, write sidecar, return ListShard.

        Idempotent: the committed result is cached, so a second call (e.g. the
        context-manager exit after an explicit ``close()``) is a no-op.
        """
        if self._result is not None:
            return self._result
        pre_close_flushes = self._n_chunks_written
        with log_time(f"Flushing remaining buffer for {self._data_path}"):
            self._flush()

        mapper_avg_item_bytes = (
            self._total_bytes_written / self._total_rows_written if self._total_rows_written > 0 else 0.0
        )

        logger.info(
            "[shard %d] scatter write done: %d pre-close flushes + %d at close = %d total; avg_item_bytes=%.0f B",
            self._source_shard,
            pre_close_flushes,
            self._n_chunks_written - pre_close_flushes,
            self._n_chunks_written,
            mapper_avg_item_bytes,
        )

        with log_time(f"Writing scatter meta for {self._data_path}"):
            _Sidecar(
                path=self._data_path,
                files=list(self._chunk_files),
                shard_bytes=dict(self._shard_bytes),
                shard_rows=dict(self._shard_rows),
            ).write()

        self._result = ListShard(refs=[MemChunk(items=[self._data_path])])
        return self._result

    def cleanup(self) -> None:
        """Discard the output instead of committing it."""
        try:
            path = StoragePath(self._data_path)
            if path.exists():
                path.rmtree()
        except Exception as e:
            logger.warning(f"Failed to cleanup scatter directory {self._data_path}: {e}")

    def __enter__(self) -> "ScatterWriter":
        return self

    def __exit__(self, exc_type: type[BaseException] | None, *exc: Any) -> None:
        if exc_type is not None:
            self.cleanup()
            return
        try:
            self.close()
        except BaseException:
            # A failure while committing (e.g. a dead R2 connection) would
            # otherwise leave the multipart upload open. Discard, then re-raise.
            self.cleanup()
            raise


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
    with ScatterWriter(
        data_path=data_path,
        key_fn=key_fn,
        source_shard=source_shard,
        sort_fn=sort_fn,
        combiner_fn=combiner_fn,
    ) as writer:
        pending: list[Any] = []
        for item in items:
            pending.append(item)
            if len(pending) >= _DATAFRAME_ROW_COUNT:
                writer.write(_items_to_dataframe(pending, key_fn, sort_fn, num_output_shards))
                pending.clear()
        if pending:
            writer.write(_items_to_dataframe(pending, key_fn, sort_fn, num_output_shards))
        return writer.close()
