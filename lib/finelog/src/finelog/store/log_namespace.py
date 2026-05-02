# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-namespace log storage state.

A :class:`LogNamespace` owns everything that used to be top-level state on
``DuckDBLogStore``: chunked Arrow tables, the in-flight sealed buffer, the
local segment registry, the sequence counter, the background flush thread,
and the registered Arrow schema.

Every namespace carries its own registered
:class:`finelog.store.schema.Schema`, and the flush / compaction pipeline
projects to that schema's column order. The ``log`` namespace continues to
expose the old key/source/data read path — that's specific to the log RPC,
not to the storage layer — but its on-disk columns are driven by the
registered schema like any other namespace.

Concurrency
-----------

The two global locks (insertion mutex + query-visibility rwlock) live on the
:class:`NamespaceRegistry` and are passed to each ``LogNamespace`` at
construction. The namespace's mutating methods (append, flush, compaction,
GC) acquire those locks; the namespace itself does not own additional locks
beyond a per-namespace flush mutex that prevents two writers from racing on
the same ``tmp_*.parquet`` filename when both the test ``_force_flush`` hook
and the bg thread fire concurrently.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import deque
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from threading import Condition, Lock
from typing import Protocol

import duckdb
import fsspec.core
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rigging.timing import RateLimiter

from finelog.rpc import logging_pb2
from finelog.store.rwlock import RWLock
from finelog.store.schema import (
    Column,
    ColumnType,
    Schema,
    duckdb_type_for,
    schema_to_arrow,
)
from finelog.types import REGEX_META_RE, LogReadResult, parse_attempt_id, str_to_log_level

logger = logging.getLogger(__name__)

# The registered schema for the "log" namespace. Mirrors the Parquet
# layout the log RPC has always written, so its read path is unaffected
# by going through the namespace registry.
LOG_REGISTERED_SCHEMA = Schema(
    columns=(
        Column(name="seq", type=ColumnType.INT64, nullable=False),
        Column(name="key", type=ColumnType.STRING, nullable=False),
        Column(name="source", type=ColumnType.STRING, nullable=False),
        Column(name="data", type=ColumnType.STRING, nullable=False),
        Column(name="epoch_ms", type=ColumnType.INT64, nullable=False),
        Column(name="level", type=ColumnType.INT32, nullable=False),
    ),
    key_column="epoch_ms",
)

# Both prefixes keyed by min_seq, so sort-by-filename yields chronological order.
_TMP_PREFIX = "tmp_"
_LOG_PREFIX = "logs_"

_ROW_GROUP_SIZE = 16_384

# Hard ceiling on the per-read parquet working set. Caps cumulative on-disk
# bytes opened in a single query; safety net for pathological body-LIKE
# queries that cannot be pruned by row-group statistics.
_MAX_PARQUET_BYTES_PER_READ = 2_500 * 1024 * 1024

_FILENAME_SEQ_RE = re.compile(rf"^(?:{_TMP_PREFIX}|{_LOG_PREFIX})(\d+)\.parquet$")


def _tmp_filename(min_seq: int) -> str:
    return f"{_TMP_PREFIX}{min_seq:019d}.parquet"


def _log_filename(min_seq: int) -> str:
    return f"{_LOG_PREFIX}{min_seq:019d}.parquet"


def _is_tmp_path(path: str) -> bool:
    # ``Path.name`` strips the synthetic ``<mem>/`` prefix used for
    # in-memory segments, so the same check works for both backings.
    return Path(path).name.startswith(_TMP_PREFIX)


# Synthetic path scheme for in-memory segments. The basename matches the
# on-disk filename scheme so ``_is_tmp_path`` and the segment-ordering
# logic don't need to know about the storage backing.
_MEM_PATH_PREFIX = "<mem>/"


def _mem_segment_path(filename: str) -> str:
    return _MEM_PATH_PREFIX + filename


def _min_seq_from_filename(name: str) -> int | None:
    match = _FILENAME_SEQ_RE.match(name)
    if not match:
        return None
    return int(match.group(1))


def _read_seq_bounds(path: Path) -> tuple[int, int]:
    """Compute (min_seq, max_seq) for a segment using filename + parquet metadata.

    The ``min_seq`` is encoded in the filename. ``max_seq = min_seq + num_rows - 1``
    where ``num_rows`` comes from the Parquet footer. This works uniformly for
    both the log namespace (where seq is a row-level column) and other
    namespaces (where the filename counter is per-batch).
    """
    min_seq = _min_seq_from_filename(path.name)
    if min_seq is None:
        return 0, 0
    try:
        meta = pq.read_metadata(path)
        num_rows = meta.num_rows
    except Exception:
        return 0, 0
    if num_rows <= 0:
        return min_seq, min_seq
    return min_seq, min_seq + num_rows - 1


def _discover_segments(log_dir: Path) -> list[Path]:
    """Return every on-disk segment (tmp + log), chronological by filename."""
    return sorted(list(log_dir.glob(f"{_TMP_PREFIX}*.parquet")) + list(log_dir.glob(f"{_LOG_PREFIX}*.parquet")))


def _recover_next_seq(log_dir: Path) -> int:
    """Walk segments and return the next sequence to assign.

    Returns ``max(min_seq + num_rows) + 1`` across all segments, or 1 if
    none exist. Sequence values are dense (no gaps) by construction; recovery
    only needs the global maximum.
    """
    next_seq = 1
    for p in _discover_segments(log_dir):
        _, max_seq = _read_seq_bounds(p)
        if max_seq + 1 > next_seq:
            next_seq = max_seq + 1
    return next_seq


def _merge_chunks(chunks: list[pa.Table]) -> list[pa.Table]:
    """Compact the chunk list by merging adjacent same-order-of-magnitude tables.

    Maintains the invariant that each chunk is at least 2x the size of the
    previous one (log-structured merge), keeping ``len(chunks)`` logarithmic
    in total row count.
    """
    if len(chunks) < 2:
        return chunks
    merged = [chunks[0]]
    for chunk in chunks[1:]:
        if merged[-1].num_rows <= chunk.num_rows:
            merged[-1] = pa.concat_tables([merged[-1], chunk])
        else:
            merged.append(chunk)
    return merged


@dataclass
class _SealedBuffer:
    """Pre-flush snapshot being written to Parquet by the bg thread.

    Visible to readers via the active RAM tables so data in flight isn't
    invisible during the write.
    """

    table: pa.Table
    min_seq: int
    max_seq: int


@dataclass
class LocalSegment:
    """Metadata for a sealed segment.

    Backed either by a Parquet file (``table is None``, ``path`` is the
    filesystem path) or by an in-memory Arrow table (``table`` set,
    ``path`` is a synthetic ``mem:tmp_…`` / ``mem:logs_…`` identifier
    used purely for ``_is_tmp_path`` checks and as a stable key for
    eviction).
    """

    path: str
    size_bytes: int
    min_seq: int = 0
    max_seq: int = 0
    table: pa.Table | None = None


def _build_log_table(buffer: list[tuple], arrow_schema: pa.Schema) -> pa.Table:
    """Build an Arrow table for the log namespace's row tuples.

    The tuple layout is fixed at ``(seq, key, source, data, epoch_ms,
    level)`` — see :func:`LogNamespace.append_log_batch`. Other namespaces
    don't go through this function; they call :func:`append_record_batch`
    with a pre-aligned RecordBatch.
    """
    if not buffer:
        return arrow_schema.empty_table()
    n = 6
    cols: list[list] = [[] for _ in range(n)]
    for row in buffer:
        for i, val in enumerate(row):
            cols[i].append(val)
    arrays = [
        pa.array(cols[0], type=pa.int64()),
        pa.array(cols[1], type=pa.string()),
        pa.array(cols[2], type=pa.string()),
        pa.array(cols[3], type=pa.string()),
        pa.array(cols[4], type=pa.int64()),
        pa.array(cols[5], type=pa.int32()),
    ]
    return pa.table(arrays, schema=arrow_schema)


class LogNamespace:
    """Per-namespace storage state.

    Owns the in-memory write buffer, the on-disk segment registry, the flush
    thread, and the compaction state for a single namespace. The registered
    schema (an Arrow schema derived from :class:`finelog.store.schema.Schema`)
    drives the flush and compaction pipelines.

    The ``log`` namespace exposes a key/source/data read API on top of the
    same storage; that path is hardcoded for log columns and is not used by
    other namespaces.
    """

    def __init__(
        self,
        *,
        name: str,
        schema: Schema,
        data_dir: Path | None,
        remote_log_dir: str,
        segment_target_bytes: int,
        flush_interval_sec: float,
        compaction_interval_sec: float,
        max_tmp_segments_before_compact: int,
        insertion_lock: Lock,
        query_visibility_lock: RWLock,
        compaction_conn: duckdb.DuckDBPyConnection,
        read_pool: _ReadPoolProtocol,
        evict_hook: Callable[[], None] = lambda: None,
    ) -> None:
        self.name = name
        self.schema = schema
        self._arrow_schema = schema_to_arrow(schema)
        # ``data_dir is None`` selects in-memory mode: segments are held as
        # Arrow tables in :class:`LocalSegment.table`, no parquet files are
        # written, GCS offload is forced off. The compaction and eviction
        # machinery is otherwise unchanged.
        self._data_dir = data_dir
        self._in_memory = data_dir is None
        if data_dir is not None:
            data_dir.mkdir(parents=True, exist_ok=True)

        # Force-off GCS in memory mode: archiving in-memory segments would
        # require materializing them, which defeats the purpose.
        self._remote_log_dir = "" if self._in_memory else remote_log_dir
        self._segment_target_bytes = segment_target_bytes
        self._max_tmp_segments_before_compact = max_tmp_segments_before_compact
        self._evict_hook = evict_hook

        # Locks supplied by the registry. The insertion lock is acquired by
        # every mutator (append, compact step, flush snapshot, segment-list
        # mutations); the rwlock guards file ops against in-flight reads.
        self._insertion_lock = insertion_lock
        self._query_visibility_lock = query_visibility_lock
        # Local flush mutex prevents two flushers (test hook + bg thread)
        # racing on the same tmp filename. This is namespace-local because
        # filenames are derived from the namespace's own _next_seq.
        self._flush_lock = Lock()

        self._compaction_conn = compaction_conn
        self._read_pool = read_pool

        self._next_seq = 1 if self._in_memory else _recover_next_seq(data_dir)
        self._chunks: list[pa.Table] = []
        self._flushing: _SealedBuffer | None = None
        self._local_segments: deque[LocalSegment] = deque()

        if not self._in_memory:
            # Drop stale tmp segments left behind by a prior compaction that
            # crashed between rename and unlink: any tmp whose [min, max] is
            # fully covered by a logs_ segment is a duplicate.
            discovered: list[LocalSegment] = []
            for p in _discover_segments(data_dir):
                min_seq, max_seq = _read_seq_bounds(p)
                discovered.append(
                    LocalSegment(
                        path=str(p),
                        size_bytes=p.stat().st_size,
                        min_seq=min_seq,
                        max_seq=max_seq,
                    )
                )
            log_ranges = [(s.min_seq, s.max_seq) for s in discovered if not _is_tmp_path(s.path)]
            for s in discovered:
                if _is_tmp_path(s.path) and any(lo <= s.min_seq and s.max_seq <= hi for lo, hi in log_ranges):
                    logger.info("Dropping stale tmp segment %s covered by compacted logs_ range", s.path)
                    try:
                        Path(s.path).unlink()
                    except Exception:
                        logger.warning("Failed to unlink stale tmp segment %s", s.path, exc_info=True)
                    continue
                self._local_segments.append(s)

        self._flush_rl = RateLimiter(flush_interval_sec)
        self._compaction_rl = RateLimiter(compaction_interval_sec)
        # Mark all rate limiters as just-run so the bg loop doesn't fire a
        # spurious tick at startup. Without this, the very first iteration
        # would race the caller's first ``append_*`` and could compact a
        # partially-written set of tmp segments.
        self._flush_rl.mark_run()
        self._compaction_rl.mark_run()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._flush_generation = 0
        self._flush_generation_cond = Condition(Lock())
        self._compaction_generation = 0
        self._compaction_generation_cond = Condition(Lock())
        self._bg_thread = threading.Thread(
            target=self._bg_loop,
            name=f"finelog_flush_{self.name}",
            daemon=True,
        )
        self._bg_thread.start()

    # ------------------------------------------------------------------
    # Append API
    # ------------------------------------------------------------------

    def append_log_batch(self, items: list[tuple[str, list]]) -> None:
        """Log-namespace-only append for ``PushLogs`` RPCs.

        Each ``(key, entries)`` pair becomes a contiguous run of rows with
        synthesized ``seq`` values starting at the namespace's
        ``_next_seq``. The register-time Schema for the log namespace is
        :data:`LOG_REGISTERED_SCHEMA`.
        """
        with self._insertion_lock:
            for key, entries in items:
                if not entries:
                    continue
                first_seq = self._next_seq
                self._next_seq += len(entries)
                rows = [
                    (first_seq + i, key, e.source, e.data, e.timestamp.epoch_ms, e.level) for i, e in enumerate(entries)
                ]
                self._chunks.append(_build_log_table(rows, self._arrow_schema))
            self._chunks = _merge_chunks(self._chunks)
            needs_drain = self._ram_bytes_locked() >= self._segment_target_bytes
        if needs_drain:
            self._wake.set()

    def append_record_batch(self, batch: pa.RecordBatch) -> None:
        """Generic append for stats namespaces.

        ``batch`` must already be aligned to ``self._arrow_schema`` (caller
        runs :func:`finelog.store.schema.validate_and_align_batch` first). The
        batch is concatenated onto the in-RAM chunk list and a flush is
        signaled if the total RAM working set exceeds ``segment_target_bytes``.
        """
        if batch.num_rows == 0:
            return
        table = pa.Table.from_batches([batch], schema=self._arrow_schema)
        with self._insertion_lock:
            self._next_seq += batch.num_rows
            self._chunks.append(table)
            self._chunks = _merge_chunks(self._chunks)
            needs_drain = self._ram_bytes_locked() >= self._segment_target_bytes
        if needs_drain:
            self._wake.set()

    # ------------------------------------------------------------------
    # Read API (log-namespace-specific)
    # ------------------------------------------------------------------

    def get_logs(
        self,
        key: str,
        *,
        since_ms: int = 0,
        cursor: int = 0,
        substring_filter: str = "",
        max_lines: int = 0,
        tail: bool = False,
        min_level: str = "",
    ) -> LogReadResult:
        min_level_enum = str_to_log_level(min_level)
        is_pattern = bool(REGEX_META_RE.search(key))

        if not is_pattern:
            where_parts = ["key = $key", "seq > $cursor"]
            params: dict = {"key": key, "cursor": cursor}
            _add_common_filters(where_parts, params, since_ms, substring_filter, min_level_enum)
            return self._execute_read(
                where_parts,
                params,
                max_lines,
                tail,
                cursor,
                include_key_in_select=False,
                exact_key=key,
            )

        where_parts, params = _regex_query(key, cursor)
        _add_common_filters(where_parts, params, since_ms, substring_filter, min_level_enum)
        return self._execute_read(
            where_parts,
            params,
            max_lines,
            tail,
            cursor,
            include_key_in_select=True,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Stop the bg thread and drain any remaining data."""
        self._stop.set()
        self._wake.set()
        self._bg_thread.join()
        self._flush_step()
        self._compaction_step(compact_single=True)

    def stop_and_join(self) -> None:
        """Stop the bg thread and join it, without flushing or compacting.

        Used by ``DropTable``: the local segment directory is about to be
        deleted, so flushing in-memory data to disk would be wasted I/O.
        After this call no further file writes happen for this namespace.
        """
        self._stop.set()
        self._wake.set()
        self._bg_thread.join()

    # ------------------------------------------------------------------
    # Schema evolution
    # ------------------------------------------------------------------

    def update_schema(self, new_schema: Schema) -> None:
        """Replace the registered schema with an additively-evolved version.

        Caller (the registry) is responsible for verifying the new schema is
        an additive-nullable extension of the current one. Stats writes for
        the previously-registered subset continue to work; missing columns
        on the wire are filled with NULL during validate-and-align.
        """
        old_columns = {c.name: c for c in self.schema.columns}
        new_columns = {c.name: c for c in new_schema.columns}
        for name, old in old_columns.items():
            new = new_columns.get(name)
            assert new is not None, f"update_schema: column {name!r} dropped (must be additive)"
            assert new.type == old.type, f"update_schema: column {name!r} type changed {old.type}->{new.type}"
            assert new.nullable == old.nullable, f"update_schema: column {name!r} nullability changed"
        for name, new in new_columns.items():
            if name not in old_columns:
                assert new.nullable, f"update_schema: new column {name!r} must be nullable"
        self.schema = new_schema
        self._arrow_schema = schema_to_arrow(new_schema)

    # ------------------------------------------------------------------
    # Internal: background thread
    # ------------------------------------------------------------------

    def _ram_bytes_locked(self) -> int:
        """Total bytes across every in-RAM holder. Caller holds the insertion lock."""
        chunks_b = sum(t.nbytes for t in self._chunks)
        flushing_b = self._flushing.table.nbytes if self._flushing is not None else 0
        return chunks_b + flushing_b

    def _bg_loop(self) -> None:
        """Drive compact, flush, and compaction on rate-limited schedules."""
        while not self._stop.is_set():
            with self._insertion_lock:
                force_drain = self._ram_bytes_locked() >= self._segment_target_bytes
                tmp_count = sum(1 for s in self._local_segments if _is_tmp_path(s.path))
                force_compaction = tmp_count > self._max_tmp_segments_before_compact
            if force_drain:
                self._flush_step()
                self._flush_rl.mark_run()
            elif self._flush_rl.should_run():
                self._flush_step()
            if force_compaction or self._compaction_rl.should_run():
                self._compaction_step()
                self._compaction_rl.mark_run()

            self._wake.wait(timeout=min(self._flush_rl.time_until_next(), 1.0))
            self._wake.clear()

    def _compact_step(self) -> None:
        """Compatibility hook: appends already materialize Arrow chunks."""

    def _flush_step(self) -> None:
        """Seal any RAM data into a Parquet segment on disk."""
        with self._flush_lock:
            with self._insertion_lock:
                if not self._chunks:
                    return
                tables = list(self._chunks)
                self._chunks = []
                visible_table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
                min_seq, max_seq = self._derive_seq_bounds_locked(visible_table)
                visible = _SealedBuffer(table=visible_table, min_seq=min_seq, max_seq=max_seq)
                self._flushing = visible

            sealed = _SealedBuffer(
                table=self._sort_for_flush(visible_table),
                min_seq=min_seq,
                max_seq=max_seq,
            )

            try:
                self._write_new_segment(sealed)
            except Exception:
                logger.warning("Flush failed, restoring data to chunks", exc_info=True)
                with self._insertion_lock:
                    self._chunks.insert(0, visible.table)
                    self._flushing = None
                return

            with self._flush_generation_cond:
                self._flush_generation += 1
                self._flush_generation_cond.notify_all()

            self._evict_hook()

    def _derive_seq_bounds_locked(self, table: pa.Table) -> tuple[int, int]:
        """Choose ``(min_seq, max_seq)`` for the sealed buffer.

        For the log namespace, the seq column is per-row and bounds come
        from the column itself. For other namespaces, the filename counter
        is per-batch — bounds are derived from ``_next_seq`` and the table's
        row count. Caller holds the insertion lock.
        """
        if "seq" in table.column_names:
            seq_col = table.column("seq")
            return pc.min(seq_col).as_py(), pc.max(seq_col).as_py()
        max_seq = self._next_seq - 1
        min_seq = max_seq - table.num_rows + 1
        return min_seq, max_seq

    def _sort_for_flush(self, table: pa.Table) -> pa.Table:
        """Sort a sealed buffer by its compaction order.

        For the log namespace this is ``(key, seq)`` so per-key reads scan
        contiguous runs. Other namespaces rely on the registered key column.
        """
        if "seq" in table.column_names and "key" in table.column_names:
            return table.sort_by([("key", "ascending"), ("seq", "ascending")])
        return table.sort_by([(self.schema.key_column or "timestamp_ms", "ascending")])

    def _write_new_segment(self, sealed: _SealedBuffer) -> None:
        filename = _tmp_filename(sealed.min_seq)
        write_start = time.monotonic()

        if self._in_memory:
            seg = LocalSegment(
                path=_mem_segment_path(filename),
                size_bytes=sealed.table.nbytes,
                min_seq=sealed.min_seq,
                max_seq=sealed.max_seq,
                table=sealed.table,
            )
        else:
            filepath = self._data_dir / filename
            tmp_path = filepath.with_suffix(".parquet.tmp")
            pq.write_table(
                sealed.table,
                tmp_path,
                compression="zstd",
                row_group_size=_ROW_GROUP_SIZE,
                write_page_index=True,
            )
            tmp_path.rename(filepath)
            seg = LocalSegment(
                path=str(filepath),
                size_bytes=filepath.stat().st_size,
                min_seq=sealed.min_seq,
                max_seq=sealed.max_seq,
            )

        with self._insertion_lock:
            self._local_segments.append(seg)
            self._flushing = None

        logger.info(
            "Wrote tmp segment %s: rows=%d bytes=%d seq=[%d,%d] elapsed_ms=%d",
            filename,
            sealed.table.num_rows,
            seg.size_bytes,
            sealed.min_seq,
            sealed.max_seq,
            int((time.monotonic() - write_start) * 1000),
        )

    def _compaction_step(self, *, compact_single: bool = False) -> None:
        with self._insertion_lock:
            tmps = [s for s in self._local_segments if _is_tmp_path(s.path)]
        if not tmps:
            return
        if len(tmps) < 2 and not compact_single:
            return

        tmps.sort(key=lambda s: s.min_seq)
        min_seq = tmps[0].min_seq
        max_seq = max(t.max_seq for t in tmps)
        merged_filename = _log_filename(min_seq)
        compaction_start = time.monotonic()

        if self._in_memory:
            merged_seg = self._compact_in_memory(tmps, min_seq, max_seq, merged_filename)
            if merged_seg is None:
                return
            merged_target = "<memory>"
        else:
            merged_path = self._data_dir / merged_filename
            staging_path = merged_path.with_suffix(".parquet.tmp")
            sql = self._build_compaction_sql([Path(t.path) for t in tmps], staging_path)
            try:
                with self._insertion_lock:
                    self._compaction_conn.execute(sql)
            except Exception:
                logger.warning("Compaction failed, leaving tmp segments in place", exc_info=True)
                staging_path.unlink(missing_ok=True)
                return
            merged_seg = LocalSegment(
                path=str(merged_path),
                size_bytes=staging_path.stat().st_size,
                min_seq=min_seq,
                max_seq=max_seq,
            )
            merged_target = str(merged_path)

        tmp_paths = {t.path for t in tmps}

        self._query_visibility_lock.write_acquire()
        try:
            if not self._in_memory:
                staging_path.rename(merged_path)
            with self._insertion_lock:
                new_segments: deque[LocalSegment] = deque()
                merged_inserted = False
                for s in self._local_segments:
                    if s.path in tmp_paths:
                        if not merged_inserted:
                            new_segments.append(merged_seg)
                            merged_inserted = True
                    else:
                        new_segments.append(s)
                if not merged_inserted:
                    new_segments.append(merged_seg)
                self._local_segments = new_segments
            if not self._in_memory:
                for t in tmps:
                    try:
                        Path(t.path).unlink(missing_ok=True)
                    except Exception:
                        logger.warning("Failed to unlink tmp segment %s", t.path, exc_info=True)
        finally:
            self._query_visibility_lock.write_release()

        with self._compaction_generation_cond:
            self._compaction_generation += 1
            self._compaction_generation_cond.notify_all()

        logger.info(
            "Compacted %d tmp segments into %s: bytes=%d seq=[%d,%d] elapsed_ms=%d",
            len(tmps),
            merged_filename,
            merged_seg.size_bytes,
            min_seq,
            max_seq,
            int((time.monotonic() - compaction_start) * 1000),
        )
        if not self._in_memory:
            self._offload_to_gcs(merged_filename, Path(merged_target))
        self._evict_hook()

    def _compact_in_memory(
        self,
        tmps: list[LocalSegment],
        min_seq: int,
        max_seq: int,
        merged_filename: str,
    ) -> LocalSegment | None:
        """Concat + project + sort the input tables; return one merged segment.

        Mirrors ``_build_compaction_sql`` semantics: columns missing from a
        given input are filled with NULL, the output is projected to the
        registered schema's column order, and rows are ordered by the
        compaction key.
        """
        target_schema = self._arrow_schema
        try:
            with self._insertion_lock:
                aligned = [_project_to_schema(t.table, target_schema) for t in tmps if t.table is not None]
            if not aligned:
                return None
            merged = pa.concat_tables(aligned)
            merged = self._sort_for_flush(merged)
        except Exception:
            logger.warning("In-memory compaction failed, leaving tmp segments in place", exc_info=True)
            return None
        return LocalSegment(
            path=_mem_segment_path(merged_filename),
            size_bytes=merged.nbytes,
            min_seq=min_seq,
            max_seq=max_seq,
            table=merged,
        )

    def _build_compaction_sql(self, input_paths: list[Path], staging_path: Path) -> str:
        """Compose the COPY statement that compacts ``input_paths`` to ``staging_path``.

        Generates one SELECT expression per registered column. Columns
        missing from *every* input segment are synthesized as
        ``NULL::TYPE AS name``; columns present in at least one input are
        projected by name and ``union_by_name=true`` fills NULL where they
        are absent from individual segments.
        """
        present_columns = self._present_input_columns(input_paths)
        select_exprs: list[str] = []
        for col in self.schema.columns:
            ident = _sql_quote_ident(col.name)
            if col.name in present_columns:
                select_exprs.append(ident)
            else:
                select_exprs.append(f"NULL::{duckdb_type_for(col)} AS {ident}")

        # Self-generated paths from _tmp_filename — no SQL injection surface.
        paths_sql = ", ".join(f"'{_sql_escape_str(str(p))}'" for p in input_paths)
        select_clause = ", ".join(select_exprs)
        order_clause = self._compaction_order_clause()
        return (
            f"COPY (SELECT {select_clause} "
            f"FROM read_parquet([{paths_sql}], union_by_name=true) "
            f"{order_clause}) "
            f"TO '{_sql_escape_str(str(staging_path))}' "
            f"(FORMAT 'parquet', ROW_GROUP_SIZE {_ROW_GROUP_SIZE}, COMPRESSION 'zstd', COMPRESSION_LEVEL 1)"
        )

    def _present_input_columns(self, input_paths: list[Path]) -> set[str]:
        """Return the union of column names present across all input segments."""
        present: set[str] = set()
        for p in input_paths:
            schema = pq.read_schema(p)
            present.update(schema.names)
        return present

    def _compaction_order_clause(self) -> str:
        """Return the ORDER BY clause for compaction.

        For the log namespace the order is ``(key, seq)`` so per-key reads
        get contiguous row ranges. Other namespaces sort by their key column.
        """
        names = self.schema.column_names()
        if "seq" in names and "key" in names:
            return "ORDER BY key, seq"
        order_col = _sql_quote_ident(self.schema.key_column or "timestamp_ms")
        return f"ORDER BY {order_col}"

    def sealed_segments(self) -> list[LocalSegment]:
        """Return flushed (logs_*) segments only, oldest first.

        Caller must hold the registry's query-visibility read lock so the
        list isn't mutated mid-snapshot. Tmp segments are deliberately
        excluded — queries see only flushed/compacted data.

        Each returned segment is either parquet-backed (``table is None``,
        ``path`` is the filesystem path) or in-memory backed (``table``
        set; ``path`` is a synthetic ``<mem>/...`` identifier).
        """
        with self._insertion_lock:
            return [s for s in self._local_segments if not _is_tmp_path(s.path)]

    def query_snapshot(self) -> tuple[list[LocalSegment], list[pa.Table]]:
        """Return all currently queryable local segments and RAM tables.

        The stats SQL path should see rows as soon as ``write_rows`` accepts
        them, not only after background flush/compaction. The caller holds the
        registry's query-visibility read lock across query execution so parquet
        files in the returned segment list cannot be unlinked mid-scan.
        """
        with self._insertion_lock:
            ram_tables = list(self._chunks)
            if self._flushing is not None:
                ram_tables.append(self._flushing.table)
            return list(self._local_segments), ram_tables

    def all_segments_unlocked(self) -> list[LocalSegment]:
        """Return a snapshot of every locally-tracked segment (tmp + sealed).

        Caller MUST hold the registry's insertion lock — this method does
        not take it. Used by the registry's eviction pass which already
        owns the lock for the snapshot phase.
        """
        return list(self._local_segments)

    def evict_segment(self, path: str) -> int:
        """Remove ``path`` from local tracking and unlink the file.

        Caller holds the registry's query-visibility write lock so a
        concurrent query cannot observe a torn segment list. Returns the
        size in bytes freed (or 0 if the segment was not tracked).
        """
        with self._insertion_lock:
            new: deque[LocalSegment] = deque()
            removed_bytes = 0
            for s in self._local_segments:
                if s.path == path:
                    removed_bytes = s.size_bytes
                    continue
                new.append(s)
            self._local_segments = new
        if not self._in_memory:
            try:
                Path(path).unlink(missing_ok=True)
            except Exception:
                logger.warning("Failed to delete evicted segment %s", path, exc_info=True)
        return removed_bytes

    def remove_local_storage(self) -> None:
        """Delete every tracked segment file plus the namespace directory.

        Used by ``DropTable``. Caller holds the query-visibility write
        lock; the namespace must already be detached from the registry
        (the bg flush thread joined, the registry row removed) so no
        concurrent code holds a reference to this namespace.

        In-memory mode: just clear the segment list — there is no
        directory or files to remove.
        """
        if self._in_memory:
            self._local_segments.clear()
            return
        for s in list(self._local_segments):
            try:
                Path(s.path).unlink(missing_ok=True)
            except Exception:
                logger.warning("Failed to delete %s during drop", s.path, exc_info=True)
        self._local_segments.clear()
        # Remove any stragglers (e.g. half-written .parquet.tmp) plus the
        # directory itself. ``rmdir`` only removes an empty directory; if
        # something unexpected is left we log and leave the directory.
        for p in list(self._data_dir.glob("*")):
            try:
                p.unlink()
            except Exception:
                logger.warning("Failed to delete stray file %s during drop", p, exc_info=True)
        try:
            self._data_dir.rmdir()
        except OSError:
            logger.warning("Namespace dir %s not empty after drop", self._data_dir)

    def _offload_to_gcs(self, filename: str, filepath: Path) -> None:
        """Copy a Parquet file to GCS (best-effort).

        Per spec, the remote layout mirrors local: per-namespace prefix at
        ``{remote_log_dir}/{namespace}/{filename}``.
        """
        if not self._remote_log_dir:
            return

        remote_path = f"{self._remote_log_dir.rstrip('/')}/{self.name}/{filename}"
        upload_start = time.monotonic()
        try:
            with fsspec.core.open(str(filepath), "rb") as f_src, fsspec.core.open(remote_path, "wb") as f_dst:
                f_dst.write(f_src.read())
        except Exception:
            logger.warning("Failed to offload %s to GCS", filepath, exc_info=True)
            return
        logger.info(
            "Offloaded %s to %s: bytes=%d elapsed_ms=%d",
            filename,
            remote_path,
            filepath.stat().st_size,
            int((time.monotonic() - upload_start) * 1000),
        )

    # ------------------------------------------------------------------
    # Internal: read
    # ------------------------------------------------------------------

    def _execute_read(
        self,
        where_parts: list[str],
        params: dict,
        max_lines: int,
        tail: bool,
        default_cursor: int,
        include_key_in_select: bool,
        exact_key: str | None = None,
    ) -> LogReadResult:
        # Hold the rwlock across the whole query so GC / compaction can't
        # unlink a file that DuckDB may still open lazily.
        self._query_visibility_lock.read_acquire()
        try:
            rows = self._run_read_locked(
                where_parts=where_parts,
                params=params,
                max_lines=max_lines,
                tail=tail,
                include_key_in_select=include_key_in_select,
            )
        finally:
            self._query_visibility_lock.read_release()

        if tail and max_lines > 0:
            rows.reverse()

        if not rows:
            return LogReadResult(entries=[], cursor=default_cursor)

        max_seq = max(r[0] for r in rows)

        if include_key_in_select:
            entries = []
            for r in rows:
                # r: (seq, key, source, data, epoch_ms, level)
                entry = logging_pb2.LogEntry(source=r[2], data=r[3], level=r[5])
                entry.timestamp.epoch_ms = r[4]
                entry.key = r[1]
                entry.attempt_id = parse_attempt_id(r[1])
                entries.append(entry)
        else:
            entries = []
            attempt_id = parse_attempt_id(exact_key) if exact_key else 0
            for r in rows:
                # r: (seq, source, data, epoch_ms, level)
                entry = logging_pb2.LogEntry(source=r[1], data=r[2], level=r[4])
                entry.timestamp.epoch_ms = r[3]
                entry.attempt_id = attempt_id
                entries.append(entry)

        return LogReadResult(entries=entries, cursor=max_seq)

    def _run_read_locked(
        self,
        *,
        where_parts: list[str],
        params: dict,
        max_lines: int,
        tail: bool,
        include_key_in_select: bool,
    ) -> list[tuple]:
        """Snapshot RAM + segments, run one DuckDB query."""
        with self._insertion_lock:
            segments = list(self._local_segments)
            ram_tables: list[pa.Table] = list(self._chunks)
            if self._flushing is not None:
                ram_tables.append(self._flushing.table)

        segments = _cap_segments(segments)
        parquet_files = [s.path for s in segments if s.table is None]
        # In-memory segments come into the query as registered views,
        # alongside the in-flight RAM tables. Order is irrelevant — the
        # final SQL is a UNION ALL.
        mem_segment_tables = [s.table for s in segments if s.table is not None]

        where_clause = " AND ".join(where_parts)
        select_cols = (
            "seq, key, source, data, epoch_ms, level" if include_key_in_select else "seq, source, data, epoch_ms, level"
        )
        order = "ORDER BY seq DESC" if (tail and max_lines > 0) else "ORDER BY seq"
        limit = f"LIMIT {max_lines}" if max_lines > 0 else ""

        with self._read_pool.checkout(ram_tables + mem_segment_tables) as (conn, ram_names):
            source = _build_union_source(parquet_files, ram_names, self._arrow_schema)
            sql = f"SELECT {select_cols} FROM ({source}) WHERE {where_clause} {order} {limit}"
            return conn.execute(sql, params).fetchall()


# ---------------------------------------------------------------------------
# Helpers (private)
# ---------------------------------------------------------------------------


class _ReadPoolProtocol(Protocol):
    """Structural type the ``LogNamespace`` uses for read connections.

    The registry's :class:`_ConnectionPool` implements this; tests can
    substitute any object that yields a ``(cursor, ram_table_names)`` pair.
    """

    def checkout(
        self, buffer_tables: list[pa.Table]
    ) -> AbstractContextManager[tuple[duckdb.DuckDBPyConnection, list[str]]]: ...


def _cap_segments(segments: list[LocalSegment]) -> list[LocalSegment]:
    """Cap a segment list at the per-read working-set ceiling, newest-first."""
    if not segments:
        return segments
    newest_first = sorted(segments, key=lambda s: s.min_seq, reverse=True)
    capped: list[LocalSegment] = []
    total = 0
    for seg in newest_first:
        if capped and total + seg.size_bytes > _MAX_PARQUET_BYTES_PER_READ:
            break
        capped.append(seg)
        total += seg.size_bytes
    capped.sort(key=lambda s: s.min_seq)
    return capped


def _regex_literal_prefix(pattern: str) -> str:
    """Extract the literal prefix from a regex pattern."""
    match = REGEX_META_RE.search(pattern)
    if match is None:
        return pattern
    return pattern[: match.start()]


def _regex_query(pattern: str, cursor: int) -> tuple[list[str], dict]:
    literal_prefix = _regex_literal_prefix(pattern)
    suffix = pattern[len(literal_prefix) :]
    is_pure_prefix = suffix in (".*", "")

    where_parts = ["seq > $cursor"]
    params: dict = {"cursor": cursor}

    if literal_prefix:
        where_parts.append("prefix(key, $prefix_lo)")
        params["prefix_lo"] = literal_prefix

    if not is_pure_prefix:
        where_parts.append("regexp_matches(key, $key_pattern)")
        params["key_pattern"] = pattern

    return where_parts, params


def _add_common_filters(
    where_parts: list[str],
    params: dict,
    since_ms: int,
    substring_filter: str,
    min_level_enum: int,
) -> None:
    if since_ms > 0:
        where_parts.append("epoch_ms > $since_ms")
        params["since_ms"] = since_ms
    if substring_filter:
        where_parts.append("contains(data, $substring)")
        params["substring"] = substring_filter
    if min_level_enum > 0:
        where_parts.append("(level = 0 OR level >= $min_level)")
        params["min_level"] = min_level_enum


def _project_to_schema(table: pa.Table, target: pa.Schema) -> pa.Table:
    """Cast/extend ``table`` to match ``target`` column order, filling NULL.

    Mirrors :meth:`LogNamespace._build_compaction_sql` for in-memory
    compaction: columns absent from ``table`` are added as nulls; columns
    present are cast to the target type if needed; columns not in
    ``target`` are dropped.
    """
    cols = []
    for field in target:
        if field.name in table.schema.names:
            col = table.column(field.name)
            if col.type != field.type:
                col = col.cast(field.type)
            cols.append(col)
        else:
            cols.append(pa.nulls(table.num_rows, type=field.type))
    return pa.Table.from_arrays(cols, schema=target)


def _build_union_source(parquet_files: list[str], ram_table_names: list[str], arrow_schema: pa.Schema) -> str:
    """SQL source: local Parquet files UNION ALL ram tables.

    File paths are self-generated (``tmp_*.parquet`` / ``logs_*.parquet``) so
    no SQL injection risk from f-string embedding. RAM table names are
    generated internally (``_ram_<cid>_<i>``).
    """
    parts: list[str] = []
    if parquet_files:
        file_list = ", ".join(f"'{f}'" for f in parquet_files)
        parts.append(f"SELECT * FROM read_parquet([{file_list}])")
    for name in ram_table_names:
        parts.append(f"SELECT * FROM {name}")
    if not parts:
        col_defs = ", ".join(f"NULL::{_arrow_to_duckdb_type(f.type)} AS {f.name}" for f in arrow_schema)
        return f"SELECT {col_defs} WHERE false"
    return " UNION ALL ".join(parts)


_ARROW_TO_DUCKDB: dict[pa.DataType, str] = {
    pa.int64(): "BIGINT",
    pa.int32(): "INTEGER",
    pa.string(): "VARCHAR",
    pa.float64(): "DOUBLE",
    pa.bool_(): "BOOLEAN",
    pa.timestamp("ms"): "TIMESTAMP_MS",
    pa.binary(): "BLOB",
}


def _arrow_to_duckdb_type(arrow_type: pa.DataType) -> str:
    return _ARROW_TO_DUCKDB[arrow_type]


def _sql_quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _sql_escape_str(text: str) -> str:
    return text.replace("'", "''")
