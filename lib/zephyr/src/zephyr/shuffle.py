# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Scatter/shuffle support for Zephyr pipelines.

Each source-shard's scatter output is a set of zstd-compressed Parquet files.
A flush writes one file per target-shard range, sorted by
``(_SHARD_COL, _SORT_KEY_COL)``. A msgpack sidecar records each file's Arrow
schema and target range plus exact per-target-shard payload bytes and row
counts. Reducers use it to select one range file per mapper flush without
opening unrelated Parquet footers.

On the read side, each reducer uses DataFusion predicate pushdown to scan only
its target shard. Row-group statistics skip non-matching groups via byte-range
GETs, so each reducer reads roughly 1/N of each file. DataFusion preserves the
declared file ordering and performs the global k-way merge. Large merges use a
bounded multi-pass merge with Parquet runs on the stage filesystem. DataFusion
native disk spill is disabled; Zephyr owns every external run and writes it to
the same filesystem as the stage.

Write-side memory is bounded by buffer size: when the sum of Arrow table
buffers exceeds ``_SCATTER_FLUSH_MEMORY_FRACTION`` of available task memory,
all buffers are flushed as bounded target-range files.

Routing columns (``__zephyr_shard__``, ``__zephyr_sort_key__``) are added
in ``_items_to_table``; ``__zephyr_shard__`` is stripped on read,
``__zephyr_sort_key__`` is consumed by the merge and stripped after.
"""

import concurrent.futures
import gc
import logging
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

import cloudpickle
import humanfriendly
import msgspec
import pyarrow as pa
import pyarrow.compute as pc
from datafusion import DataFrame, DataFrameWriteOptions, ParquetWriterOptions, SessionContext, col, lit
from iris.env_resources import TaskResources
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import RateLimiter, log_time

from zephyr.parquet_scan import datafusion_context, register_object_stores, scan_parquet
from zephyr.shard_keys import encode_key, hash_encoded_key
from zephyr.worker_context import _worker_ctx_var
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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SCATTER_METADATA_FILENAME = "metadata.msgpack"

# Number of parallel sidecar reads each reducer issues. These reads are GCS
# GET-bound, so a modest pool keeps latency low without thrashing.
_SIDECAR_READ_CONCURRENCY = 8
# Fraction of available memory available for merging.
_SCATTER_READ_MEMORY_FRACTION = 0.4

# Memory overhead multiple per row in the DataFusion/Arrow representation.
_SCATTER_READ_DATAFUSION_ROW_OVERHEAD = 2
# Memory overhead multiple per row in the deserialized Python iterator.
_SCATTER_READ_PYTHON_ROW_OVERHEAD = 2

_PROGRESS_LOG_INTERVAL_SECONDS = 60.0
# Bound Parquet footer size when a shuffle has thousands of target shards. One
# row group per target gives ideal predicate pruning, but makes every reducer
# read multi-megabyte footers from every mapper chunk before it can read data.
_SCATTER_MAX_ROW_GROUPS_PER_CHUNK = 512
# Split a flush into target-shard ranges. DataFusion's sort keeps several
# copies of its input columns, so bounding each sort is substantially cheaper
# than sorting the whole multiplexed buffer. Each reducer still opens one file
# per mapper flush because sidecars identify the range that contains it.
_SCATTER_FILE_PARTITIONS = 8

# Internal routing columns injected by _items_to_table.
_SHARD_COL = "__zephyr_shard__"
_SORT_KEY_COL = "__zephyr_sort_key__"
# A cloudpickle-serialized Python object representing the item
_PAYLOAD_COL = "__payload__"
_PAYLOAD_BYTES_COL = "payload_bytes"
_PAYLOAD_BYTES_SUM_COL = f"{_PAYLOAD_BYTES_COL}_sum"
_PAYLOAD_ROWS_COL = f"{_PAYLOAD_BYTES_COL}_count"

_PARQUET_WRITE_OPTIONS = DataFrameWriteOptions(single_file_output=True)
_PARQUET_COMPRESSION = "zstd(1)"

# Python items consumed before creating an Arrow table.
_TABLE_ROW_COUNT = 10_000
# Leave headroom for Arrow concatenation, DataFusion sorting, and Parquet
# encoding. A scatter flush has reached 2.27x the buffered size at peak RSS:
# https://echo.oa.dev/wiki/68.
_SCATTER_FLUSH_MEMORY_FRACTION = 0.12
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


def _table_to_items(table: pa.Table | pa.RecordBatch) -> Iterator[Any]:
    """Yield deserialized Python items from an Arrow table or record batch."""
    for payload in table[_PAYLOAD_COL].to_pylist():
        yield cloudpickle.loads(payload)


def _columns_to_table(
    payloads: list[bytes],
    shards: list[int],
    key_bytes: list[bytes],
    sort_values: list[Any],
) -> pa.Table:
    """Build a scatter Arrow table from pre-computed flat columns.

    Field order in the sort-key struct (key first) drives the
    ``(key, sort_value)`` ordering used by DataFusion.

    ``key_bytes`` must be pre-encoded via :func:`~zephyr.shard_keys.encode_key` so that
    the key field is always ``Binary`` — preventing struct schema mismatches
    when different mapper shards produce keys of different Python types.
    """
    try:
        sort_key = pa.StructArray.from_arrays(
            [pa.array(key_bytes, type=pa.binary()), pa.array(sort_values)],
            names=["key", "sort_value"],
        )
        return pa.table(
            {
                _PAYLOAD_COL: pa.array(payloads, type=pa.binary()),
                _SHARD_COL: pa.array(shards, type=pa.int32()),
                _SORT_KEY_COL: sort_key,
            }
        )
    except (TypeError, pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError) as err:
        raise ValueError("sort_fn must return an Arrow-serializable object.") from err


def _file_partition_count(num_output_shards: int) -> int:
    if num_output_shards <= 0:
        raise ValueError(f"num_output_shards must be positive, got {num_output_shards}")
    return min(num_output_shards, _SCATTER_FILE_PARTITIONS)


def _file_partition(shard: int, num_output_shards: int, num_file_partitions: int) -> int:
    return min(num_file_partitions - 1, shard * num_file_partitions // num_output_shards)


def _file_partition_bounds(partition: int, num_output_shards: int, num_file_partitions: int) -> tuple[int, int]:
    shard_start = (partition * num_output_shards + num_file_partitions - 1) // num_file_partitions
    shard_end = ((partition + 1) * num_output_shards + num_file_partitions - 1) // num_file_partitions
    return shard_start, shard_end


def _items_to_partitioned_tables(
    items: list[Any],
    key_fn: Callable,
    sort_fn: Callable | None,
    num_output_shards: int,
    num_file_partitions: int,
) -> list[tuple[int, pa.Table]]:
    """Convert Python items directly into target-range Arrow tables."""
    columns: list[tuple[list[Any], list[int], list[bytes], list[Any]]] = [
        ([], [], [], []) for _ in range(num_file_partitions)
    ]
    for item in items:
        key = key_fn(item)
        try:
            key_bytes = encode_key(key)
        except TypeError as err:
            raise ValueError(f"key_fn must return a msgpack-serializable object; got {type(key).__name__!r}.") from err
        shard = hash_encoded_key(key_bytes) % num_output_shards
        partition = _file_partition(shard, num_output_shards, num_file_partitions)
        partition_items, shards, keys, sort_values = columns[partition]
        partition_items.append(item)
        shards.append(shard)
        keys.append(key_bytes)
        sort_values.append(sort_fn(item) if sort_fn is not None else None)

    return [
        (
            partition,
            _columns_to_table(
                [cloudpickle.dumps(item) for item in partition_items],
                shards,
                keys,
                sort_values,
            ),
        )
        for partition, (partition_items, shards, keys, sort_values) in enumerate(columns)
        if partition_items
    ]


def _items_to_table(
    items: list[Any],
    key_fn: Callable,
    sort_fn: Callable | None,
    num_output_shards: int,
) -> pa.Table:
    """Convert Python items to an Arrow table with routing columns.

    Cloudpickle-serializes items into ``_PAYLOAD_COL`` and adds ``_SHARD_COL``
    (int32 target shard index) and ``_SORT_KEY_COL``. This is the adapter
    between Python-item pipelines and the Arrow-based :class:`ScatterWriter`.

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
    return _columns_to_table(payloads, shards, key_bytes, sort_values)


class _SidecarFilesystem(Protocol):
    """A protocol because ``url_to_fs`` returns a bare fsspec filesystem for
    ``s3://`` and a ``CrossRegionGuardedFS`` wrapper for ``gs://``, which share
    no base class."""

    def _strip_protocol(self, path: str) -> str: ...

    def cat_file(self, path: str) -> bytes: ...


@dataclass(frozen=True)
class _ChunkFile:
    """A scatter Parquet file and the Arrow schema recorded when it was written."""

    path: str
    schema: pa.Schema
    shard_start: int
    shard_end: int

    _path_field: ClassVar[str] = "path"
    _schema_field: ClassVar[str] = "schema"
    _shard_start_field: ClassVar[str] = "shard_start"
    _shard_end_field: ClassVar[str] = "shard_end"

    def to_metadata(self) -> dict[str, str | bytes | int]:
        return {
            self._path_field: self.path,
            self._schema_field: self.schema.serialize().to_pybytes(),
            self._shard_start_field: self.shard_start,
            self._shard_end_field: self.shard_end,
        }

    def contains(self, target_shard: int) -> bool:
        return self.shard_start <= target_shard < self.shard_end

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> "_ChunkFile":
        schema = pa.ipc.read_schema(pa.BufferReader(metadata[cls._schema_field]))
        return cls(
            path=str(metadata[cls._path_field]),
            schema=schema,
            shard_start=int(metadata[cls._shard_start_field]),
            shard_end=int(metadata[cls._shard_end_field]),
        )


@dataclass(frozen=True)
class _Sidecar:
    """One mapper's scatter metadata (``metadata.msgpack``).

    ``files`` lists the combined Parquet paths and schemas written during
    flushes. ``shard_bytes`` and ``shard_rows`` contain exact per-target totals
    used to plan reducer merges without reading Parquet metadata.
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
        meta_path = fs._strip_protocol(cls.meta_path(data_path))
        # Avoid buffered-file overhead for these small payloads.
        data = cls._decoder.decode(fs.cat_file(meta_path))
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


def _fan_in_groups(frames: list[DataFrame], fan_in: int) -> list[list[DataFrame]]:
    return [frames[i : i + fan_in] for i in range(0, len(frames), fan_in)]


def _merged_frame(frames: list[DataFrame], sort_key: str) -> DataFrame:
    """Return a streaming merge over inputs already ordered by ``sort_key``."""
    if not frames:
        raise ValueError("frames must not be empty")
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.union(frame)
    return merged.sort(col(sort_key).sort())


def _merge_sorted_frames(
    context: SessionContext,
    frames: list[DataFrame],
    sort_key: str,
    external_sort_dir: str,
    fan_in: int,
    shard: int,
) -> Iterator[pa.RecordBatch]:
    """Merge ordered frames, writing bounded intermediate runs when needed."""
    if not frames:
        return
    if fan_in < 2:
        raise ValueError(f"fan_in must be at least 2, got {fan_in}")

    spill_dir: StoragePath | None = None
    spill_files: set[StoragePath] = set()
    schema = frames[0].schema()

    def write_run(group: list[DataFrame], pass_index: int, run_index: int) -> StoragePath:
        assert spill_dir is not None
        run = spill_dir / f"pass-{pass_index:04d}-run-{run_index:04d}.parquet"
        spill_files.add(run)
        merged = _merged_frame(group, sort_key)
        merged.write_parquet_with_options(
            str(run),
            ParquetWriterOptions(compression=_PARQUET_COMPRESSION),
            _PARQUET_WRITE_OPTIONS,
        )
        return run

    try:
        prior_runs: list[StoragePath] = []
        pass_index = 0
        while len(frames) > fan_in:
            if spill_dir is None:
                spill_dir = StoragePath(external_sort_dir)
                spill_dir.mkdirs()
                register_object_stores(context, [external_sort_dir])

            logger.info(
                "[shard %d] External sort: pass %d merging %d frames with fan_in=%d",
                shard,
                pass_index,
                len(frames),
                fan_in,
            )
            runs = [write_run(group, pass_index, i) for i, group in enumerate(_fan_in_groups(frames, fan_in))]

            for prior_run in prior_runs:
                prior_run.rm()
                spill_files.remove(prior_run)

            frames = [scan_parquet(context, str(run), schema=schema, sorted_by=(sort_key,)) for run in runs]
            prior_runs = runs
            pass_index += 1

        logger.info("[shard %d] Final merge of %d frames (%d spill pass(es))", shard, len(frames), pass_index)
        for batch in _merged_frame(frames, sort_key).execute_stream():
            yield batch.to_pyarrow()
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

    ``_chunk_files`` contains the one target-range Parquet file selected from
    each mapper flush. :meth:`get_frames` filters each range down to
    ``_target_shard``.

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
                chunk_files.extend(file for file in sidecar.files if file.contains(target_shard))
                shard_payload_bytes += sidecar.target_bytes(target_shard)
                shard_payload_rows += sidecar.target_rows(target_shard)

        avg_item_bytes = shard_payload_bytes / shard_payload_rows if shard_payload_rows else 0.0

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

    def get_frames(self, context: SessionContext) -> list[DataFrame]:
        """Build ordered DataFusion scans for this reducer's target shard."""
        if not self._chunk_files:
            return []

        register_object_stores(context, [chunk.path for chunk in self._chunk_files])
        schemas = [chunk.schema for chunk in self._chunk_files]
        schema = (
            schemas[0]
            if all(value.equals(schemas[0]) for value in schemas[1:])
            else pa.unify_schemas(schemas, promote_options="permissive")
        )
        frames = [
            scan_parquet(
                context,
                chunk.path,
                schema=schema,
                sorted_by=(_SHARD_COL, _SORT_KEY_COL),
            )
            for chunk in self._chunk_files
        ]
        return [frame.filter(col(_SHARD_COL) == lit(self._target_shard)).drop(_SHARD_COL) for frame in frames]

    @property
    def total_chunks(self) -> int:
        return len(self._chunk_files)

    def merge_sorted_chunks(self, external_sort_dir: str) -> Iterator[Any]:
        """Merge sorted chunks using k-way merge, yielding items in global sort order.

        Each chunk file is assumed to be sorted by ``_SORT_KEY_COL`` (key plus optional
        secondary sort). Performs a k-way merge across all chunks.
        Args:
            external_sort_dir: Stage-filesystem directory for intermediate runs
                when the shard exceeds the memory budget.

        Yields:
            Deserialized Python items in merged sort order.
        """

        if self.total_chunks == 0:
            return
        if self.shard_payload_bytes == 0:
            return

        shard_payload_bytes = self.shard_payload_bytes
        overhead = _SCATTER_READ_DATAFUSION_ROW_OVERHEAD * _SCATTER_READ_PYTHON_ROW_OVERHEAD
        memory_bytes = _task_memory_bytes()
        merge_memory_bytes = int(memory_bytes * _SCATTER_READ_MEMORY_FRACTION)

        # Each input is one independently sorted Parquet stream. More target
        # partitions make DataFusion round-robin repartition every file before
        # the merge, multiplying queues without exposing additional scan
        # parallelism for the one-file plans.
        context = datafusion_context(memory_limit_bytes=merge_memory_bytes, target_partitions=1)
        frames = self.get_frames(context)

        needs_external_sort = shard_payload_bytes * overhead > merge_memory_bytes
        fan_in = max(2, math.ceil(math.sqrt(self.total_chunks))) if needs_external_sort else max(2, self.total_chunks)
        logger.info(
            "[shard %d] Merging %d chunks with fan_in=%d (%s memory needed, %s memory available)",
            self._target_shard,
            self.total_chunks,
            fan_in,
            humanfriendly.format_size(shard_payload_bytes * overhead, binary=True),
            humanfriendly.format_size(merge_memory_bytes, binary=True),
        )
        batches = _merge_sorted_frames(
            context=context,
            frames=frames,
            sort_key=_SORT_KEY_COL,
            external_sort_dir=external_sort_dir,
            fan_in=fan_in,
            shard=self._target_shard,
        )

        for batch in batches:
            yield from _table_to_items(batch)


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
    """Write sorted, multiplexed scatter chunks and their reducer sidecar.

    Input tables contain payload, target-shard, and sort-key columns. The writer
    buffers them against the shard task's memory budget and writes target-local
    Parquet row groups for reducer predicate pushdown.
    """

    def __init__(
        self,
        data_path: str,
        key_fn: Callable,
        source_shard: int,
        num_output_shards: int,
        sort_fn: Callable | None = None,
        combiner_fn: Callable | None = None,
    ) -> None:
        self._data_path = data_path if data_path.endswith("/") else f"{data_path}/"
        self._key_fn = key_fn
        self._sort_fn = sort_fn

        self._source_shard = source_shard
        self._num_output_shards = num_output_shards
        self._num_file_partitions = _file_partition_count(num_output_shards)
        self._combiner_fn = combiner_fn
        self._memory_available_bytes = _task_memory_bytes()
        self._flush_threshold_bytes = int(self._memory_available_bytes * _SCATTER_FLUSH_MEMORY_FRACTION)

        # Buffered Arrow tables, grouped by target range and combined into one
        # file per range per flush.
        self._partition_tables: list[list[pa.Table]] = [[] for _ in range(self._num_file_partitions)]
        self._chunk_files: list[_ChunkFile] = []
        # Exact per-target payload sizes and row counts are recorded in the
        # sidecar for reducer planning.
        self._shard_bytes: defaultdict[int, int] = defaultdict(int)
        self._shard_rows: defaultdict[int, int] = defaultdict(int)
        self._n_chunks_written = 0
        self._n_flushes = 0
        # Throttles the per-flush progress log so high-fanout workloads don't log too often
        self._progress_log_limiter = RateLimiter(interval_seconds=_PROGRESS_LOG_INTERVAL_SECONDS)
        # Running Arrow buffer-size total of unflushed tables; reset to 0 on flush.
        self._buffer_estimated_bytes: int = 0

        ensure_parent_dir(self._data_path)
        self._result: ListShard | None = None

    def _flush(self) -> None:
        """Flush target-range buffers into sorted Parquet files."""
        if not any(self._partition_tables):
            return

        partition_tables = self._partition_tables
        self._partition_tables = [[] for _ in range(self._num_file_partitions)]
        self._buffer_estimated_bytes = 0
        total_flushed_bytes = 0
        files_written = 0

        for partition, tables in enumerate(partition_tables):
            if not tables:
                continue
            buffer = pa.concat_tables(tables, promote_options="permissive")
            tables.clear()
            if self._num_file_partitions > 1:
                buffer = buffer.combine_chunks()

            if self._combiner_fn is not None:
                by_shard: defaultdict[int, list[Any]] = defaultdict(list)
                for shard_val, payload in zip(
                    buffer[_SHARD_COL].to_pylist(),
                    buffer[_PAYLOAD_COL].to_pylist(),
                    strict=True,
                ):
                    by_shard[int(shard_val)].append(cloudpickle.loads(payload))

                combined_tables: list[pa.Table] = []
                for shard_val, rows in by_shard.items():
                    rows = _apply_combiner(rows, self._key_fn, self._combiner_fn)
                    if not rows:
                        continue
                    table = _items_to_table(rows, self._key_fn, self._sort_fn, num_output_shards=0)
                    shard_index = table.schema.get_field_index(_SHARD_COL)
                    combined_tables.append(
                        table.set_column(
                            shard_index,
                            _SHARD_COL,
                            pa.array([shard_val] * len(table), type=pa.int32()),
                        )
                    )
                if not combined_tables:
                    continue
                buffer = pa.concat_tables(combined_tables, promote_options="permissive")

            flushed_bytes = int(buffer.nbytes)
            total_flushed_bytes += flushed_bytes
            shard_sizes = (
                pa.table(
                    {
                        _SHARD_COL: buffer[_SHARD_COL],
                        _PAYLOAD_BYTES_COL: pc.binary_length(buffer[_PAYLOAD_COL]),
                    }
                )
                .group_by(_SHARD_COL, use_threads=False)
                .aggregate([(_PAYLOAD_BYTES_COL, "sum"), (_PAYLOAD_BYTES_COL, "count")])
            )
            for row in shard_sizes.to_pylist():
                self._shard_bytes[int(row[_SHARD_COL])] += int(row[_PAYLOAD_BYTES_SUM_COL])
                self._shard_rows[int(row[_SHARD_COL])] += int(row[_PAYLOAD_ROWS_COL])

            num_targets = int(pc.count_distinct(buffer[_SHARD_COL]).as_py())
            num_row_groups = min(num_targets, _SCATTER_MAX_ROW_GROUPS_PER_CHUNK)
            row_count = len(buffer)
            row_group_size = max(1, math.ceil(row_count / num_row_groups))
            chunk_path = f"{self._data_path}c{self._n_chunks_written:04d}.parquet"
            schema = buffer.schema
            sort_context = datafusion_context(memory_limit_bytes=self._memory_available_bytes, target_partitions=1)
            register_object_stores(sort_context, [chunk_path])
            sorted_frame = sort_context.from_arrow(buffer).sort(_SHARD_COL, _SORT_KEY_COL)
            del buffer
            sorted_frame.write_parquet_with_options(
                chunk_path,
                ParquetWriterOptions(
                    compression=_PARQUET_COMPRESSION,
                    max_row_group_size=row_group_size,
                ),
                _PARQUET_WRITE_OPTIONS,
            )

            shard_start, shard_end = _file_partition_bounds(
                partition, self._num_output_shards, self._num_file_partitions
            )
            self._chunk_files.append(
                _ChunkFile(path=chunk_path, schema=schema, shard_start=shard_start, shard_end=shard_end)
            )
            self._n_chunks_written += 1
            files_written += 1

            if self._progress_log_limiter.should_run():
                logger.info(
                    "[shard %d] Wrote %d scatter files so far (latest file: %d items, %d targets)",
                    self._source_shard,
                    self._n_chunks_written,
                    row_count,
                    num_targets,
                )

        if files_written:
            self._n_flushes += 1
        if total_flushed_bytes >= _GC_FLUSH_SIZE_THRESHOLD_BYTES:
            gc.collect()

    def write(self, table: pa.Table) -> None:
        """Partition and buffer an Arrow table, flushing on memory pressure.

        The table must contain ``_PAYLOAD_COL``, ``_SHARD_COL`` (int32), and
        ``_SORT_KEY_COL`` columns as produced by ``_items_to_table``.
        """
        if len(table) == 0:
            return

        if self._num_file_partitions == 1:
            self.write_partitioned([(0, table)])
            return

        partitioned: list[tuple[int, pa.Table]] = []
        for partition in range(self._num_file_partitions):
            shard_start, shard_end = _file_partition_bounds(
                partition, self._num_output_shards, self._num_file_partitions
            )
            mask = pc.and_(
                pc.greater_equal(table[_SHARD_COL], shard_start),
                pc.less(table[_SHARD_COL], shard_end),
            )
            partition_table = table.filter(mask)
            if len(partition_table):
                partitioned.append((partition, partition_table))
        self.write_partitioned(partitioned)

    def write_partitioned(self, tables: list[tuple[int, pa.Table]]) -> None:
        """Buffer tables already routed to target-range file partitions."""
        for partition, table in tables:
            if not 0 <= partition < self._num_file_partitions:
                raise ValueError(f"file partition {partition} is out of range")
            if len(table) == 0:
                continue
            self._partition_tables[partition].append(table)
            self._buffer_estimated_bytes += int(table.nbytes)

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
        pre_close_flushes = self._n_flushes
        with log_time(f"Flushing remaining buffer for {self._data_path}"):
            self._flush()

        logger.info(
            "[shard %d] scatter write done: %d pre-close flushes + %d at close = %d total",
            self._source_shard,
            pre_close_flushes,
            self._n_flushes - pre_close_flushes,
            self._n_flushes,
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
    """Write items as sorted Parquet chunks routed across target shards.

    Returns:
        A shard containing the scatter data path.
    """
    with ScatterWriter(
        data_path=data_path,
        key_fn=key_fn,
        source_shard=source_shard,
        num_output_shards=num_output_shards,
        sort_fn=sort_fn,
        combiner_fn=combiner_fn,
    ) as writer:
        num_file_partitions = _file_partition_count(num_output_shards)
        pending: list[Any] = []
        for item in items:
            pending.append(item)
            if len(pending) >= _TABLE_ROW_COUNT:
                writer.write_partitioned(
                    _items_to_partitioned_tables(
                        pending,
                        key_fn,
                        sort_fn,
                        num_output_shards,
                        num_file_partitions,
                    )
                )
                pending.clear()
        if pending:
            writer.write_partitioned(
                _items_to_partitioned_tables(
                    pending,
                    key_fn,
                    sort_fn,
                    num_output_shards,
                    num_file_partitions,
                )
            )
        return writer.close()
