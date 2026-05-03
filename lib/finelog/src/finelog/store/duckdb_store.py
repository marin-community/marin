# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Log store backed by rotating RAM buffers + two-tier Parquet + DuckDB reads.

Two on-disk tiers:

    ``tmp_{min_seq}.parquet`` -- written by every flush; local-only.
    ``logs_{min_seq}.parquet`` -- produced by periodic compaction that merges
    all current tmps into one archive and uploads only that to GCS.

Lifecycle of a log entry:

    1. Appended to ``_pending`` (plain Python list) under ``_memory_lock``.
    2. A single background thread wakes periodically and:
       a. Every ``compact_interval_sec`` (default 1s): convert ``_pending``
          to an Arrow ``Table`` and append to ``_chunks`` (power-of-2 merged).
       b. Every ``flush_interval_sec`` (default 60s), or under backpressure:
          seal RAM data into ``_flushing`` and write a standalone
          ``tmp_*.parquet`` segment to local disk. Existing segments are
          never rewritten — in-line consolidation retained ~45 MB per flush
          of arrow-pool memory (identified via targeted reproducer), so the
          write path is strictly append-only.
       c. Every ``compaction_interval_sec`` (default 10 min) or when tmp
          count exceeds ``max_tmp_segments_before_compact`` (default 10):
          merge all tmp_ files into a single ``logs_*.parquet`` via DuckDB
          ``COPY (SELECT ... ORDER BY key, seq) TO ... (FORMAT parquet)``,
          then upload the merged file to GCS and unlink the tmps. DuckDB's
          streaming COPY avoids the pyarrow concat/sort/write leak path.
    3. Backpressure: if ``append()`` sees total RAM holders exceed
       ``segment_target_bytes``, it wakes the bg thread to bypass the rate
       limit and drain immediately.

Read path: DuckDB ``read_parquet()`` over local segments (tmp + log), UNION
ALL pyarrow tables for in-RAM data (``_chunks``, ``_flushing.table`` if
present). Relies entirely on DuckDB's per-row-group parquet
stats for pruning — files are sorted by ``(key, seq)`` so row-group ``key``
bounds are tight enough that file-level Python filtering would add nothing.
The per-read working set is capped by ``_MAX_PARQUET_BYTES_PER_READ``, which
handles a mix of small tmps and large logs gracefully.

Locking:
    ``_memory_lock``  -- protects all mutable RAM state: ``_pending``, ``_chunks``,
    ``_flushing``, ``_local_segments``, and ``_next_seq``. Held briefly for
    snapshots and swaps; never held across I/O.

    ``_segments_rwlock`` -- readers hold the *shared* lock while DuckDB has
    parquet files open; GC / compaction hold the *exclusive* lock before
    unlinking or renaming files. This prevents file ops from disrupting
    in-flight queries.
"""

from __future__ import annotations

import logging
import tempfile
import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Condition, Lock

import duckdb
import fsspec.core
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rigging.timing import RateLimiter

from finelog.rpc import logging_pb2
from finelog.types import _EST_BYTES_PER_ROW, REGEX_META_RE, LogReadResult, parse_attempt_id, str_to_log_level

logger = logging.getLogger(__name__)

_PARQUET_SCHEMA = pa.schema(
    [
        ("seq", pa.int64()),
        ("key", pa.string()),
        ("source", pa.string()),
        ("data", pa.string()),
        ("epoch_ms", pa.int64()),
        ("level", pa.int32()),
    ]
)

# Arrow type → DuckDB type mapping for empty-source fallback SQL.
_DUCKDB_TYPE_MAP: dict[pa.DataType, str] = {
    pa.int64(): "BIGINT",
    pa.int32(): "INTEGER",
    pa.string(): "VARCHAR",
}

# ---------------------------------------------------------------------------
# Heuristic thresholds
# ---------------------------------------------------------------------------

# Target size for a single Parquet segment on disk. New data is concatenated
# onto the latest segment until it reaches this size, then a new file starts.
SEGMENT_TARGET_BYTES = 100 * 1024 * 1024  # 100 MB

# Background step cadences.
DEFAULT_COMPACT_INTERVAL_SEC = 1.0
DEFAULT_FLUSH_INTERVAL_SEC = 60.0
DEFAULT_COMPACTION_INTERVAL_SEC = 600.0  # merge small tmp parquets into one archive

# Trigger compaction when this many tmp segments accumulate, even before the
# time-based interval fires. Keeps per-read fanout bounded under high ingest.
DEFAULT_MAX_TMP_SEGMENTS_BEFORE_COMPACT = 10

# Default caps for local Parquet retention.
# The read path has no remote fallback yet (see module docstring TODO): once a
# parquet is GC'd locally, its rows are unreachable via FetchLogs even though
# they're durable on GCS. Sized to keep ~2 weeks of the production `marin`
# cluster's ingest (~6-7 GB/day, ~30 GB bucket total at 2026-04-21) fully
# local. The per-read working set is still bounded by _MAX_PARQUET_BYTES_PER_READ
# (2.5 GB, newest-first), so raising the retention cap does not affect query
# time — verified via lib/iris/scripts/benchmark_log_store.py --corpus-dir.
DEFAULT_MAX_LOCAL_SEGMENTS = 1000
DEFAULT_MAX_LOCAL_BYTES = 100 * 1024**3  # 100 GB

_ROW_GROUP_SIZE = 16_384

# Hard ceiling on the per-read parquet working set. Caps cumulative on-disk
# bytes opened in a single query; kept as a safety net for pathological
# body-LIKE queries that cannot be pruned by row-group statistics. Because
# small tmp segments coexist with larger compacted log segments, we cap by
# size only — file count doesn't cleanly distinguish the two.
_MAX_PARQUET_BYTES_PER_READ = 2_500 * 1024 * 1024

# Both prefixes keyed by min_seq, so sort-by-filename yields chronological order.
_TMP_PREFIX = "tmp_"
_LOG_PREFIX = "logs_"


def _tmp_filename(min_seq: int) -> str:
    return f"{_TMP_PREFIX}{min_seq:019d}.parquet"


def _log_filename(min_seq: int) -> str:
    return f"{_LOG_PREFIX}{min_seq:019d}.parquet"


def _is_tmp_path(path: str) -> bool:
    return Path(path).name.startswith(_TMP_PREFIX)


def _fsspec_copy(src: str, dst: str) -> None:
    """Copy a file using fsspec so either path can be remote (e.g. GCS)."""
    with fsspec.core.open(src, "rb") as f_src, fsspec.core.open(dst, "wb") as f_dst:
        f_dst.write(f_src.read())


def _read_seq_bounds(path: Path) -> tuple[int, int]:
    """Read min/max seq from Parquet row-group statistics."""
    try:
        meta = pq.read_metadata(path)
        schema = meta.schema.to_arrow_schema()
        seq_idx = schema.get_field_index("seq")
        min_seq = 0
        max_seq = 0
        for i in range(meta.num_row_groups):
            col = meta.row_group(i).column(seq_idx)
            if col.statistics is not None and col.statistics.has_min_max:
                if not min_seq or col.statistics.min < min_seq:
                    min_seq = col.statistics.min
                if col.statistics.max > max_seq:
                    max_seq = col.statistics.max
        return min_seq, max_seq
    except Exception:
        return 0, 0


def _discover_segments(log_dir: Path) -> list[Path]:
    """Return every on-disk segment (tmp + log), chronological by filename."""
    return sorted(list(log_dir.glob(f"{_TMP_PREFIX}*.parquet")) + list(log_dir.glob(f"{_LOG_PREFIX}*.parquet")))


def _recover_max_seq(log_dir: Path) -> int:
    """Recover the max sequence number across both tmp and log Parquet files.

    Returns max_seq + 1 so the counter can resume, or 1 if no files exist.
    """
    max_seen = -1
    for p in _discover_segments(log_dir):
        _, max_seq = _read_seq_bounds(p)
        if max_seq > max_seen:
            max_seen = max_seq
    return max_seen + 1 if max_seen >= 0 else 1


def _build_buffer_table(buffer: list[tuple]) -> pa.Table:
    """Convert a list of row tuples into a pyarrow Table with the log schema."""
    if not buffer:
        return _PARQUET_SCHEMA.empty_table()
    cols: list[list] = [[] for _ in range(6)]
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
    return pa.table(arrays, schema=_PARQUET_SCHEMA)


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------


def _merge_chunks(chunks: list[pa.Table]) -> list[pa.Table]:
    """Compact the chunk list by merging adjacent same-order-of-magnitude tables.

    Maintains the invariant: each chunk is at least 2x the size of the
    previous one (like a log-structured merge). This keeps len(chunks)
    logarithmic in total row count.
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

    Visible to readers via ``ram_tables`` so data in flight isn't invisible
    during the write.
    """

    table: pa.Table
    min_seq: int
    max_seq: int


@dataclass
class _LocalSegment:
    """Metadata for a Parquet file on local disk."""

    path: str
    size_bytes: int
    min_seq: int = 0
    max_seq: int = 0


class _RWLock:
    """Simple readers-writer lock.

    Multiple readers can hold the lock concurrently. A writer must wait for
    all readers to release before acquiring exclusive access. Used to prevent
    GC or consolidation from unlinking / renaming parquet files while DuckDB
    reads are in flight.
    """

    def __init__(self):
        self._cond = Condition(Lock())
        self._readers = 0
        self._writer = False

    def read_acquire(self) -> None:
        with self._cond:
            while self._writer:
                self._cond.wait()
            self._readers += 1

    def read_release(self) -> None:
        with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def write_acquire(self) -> None:
        with self._cond:
            while self._writer or self._readers > 0:
                self._cond.wait()
            self._writer = True

    def write_release(self) -> None:
        with self._cond:
            self._writer = False
            self._cond.notify_all()


# Per-connection memory ceiling for DuckDB. Tight limits (e.g. 256MB) caused
# spill-to-disk loops under concurrent tail reads over large row groups,
# wedging the controller. 4GB is generous against realistic working sets
# (5 segments x 500MB + zstd decompression scratch).
_DEFAULT_DUCKDB_MEMORY_LIMIT = "4GB"

_cursor_counter = 0
_cursor_counter_lock = Lock()


def _next_cursor_id() -> int:
    global _cursor_counter
    with _cursor_counter_lock:
        _cursor_counter += 1
        return _cursor_counter


class _ConnectionPool:
    """Two DuckDB databases: one for reads, one for compaction.

    Reads share a single ``duckdb.connect()`` with ``enable_object_cache``
    so parquet footer / row-group stats are cached across queries. Callers
    get cursors via ``conn.cursor()`` which share that connection's thread
    pool and buffer pool.

    Compaction runs on a second, isolated connection. Sharing one connection
    across reads and compaction starves the compaction COPY's sort phase:
    DuckDB schedules concurrent cursors cooperatively over a fixed worker
    pool, so ~16 in-flight fetch_logs cursors stretch a 2s compaction past
    10s and reads on top of it time out. The dedicated connection has its
    own thread pool so compaction's sort cost is independent of reader load.

    RAM tables are registered with unique names (incorporating a monotonic
    counter) so concurrent cursors don't collide on table names.
    """

    def __init__(self, memory_limit: str = _DEFAULT_DUCKDB_MEMORY_LIMIT):
        self._conn = duckdb.connect(config={"memory_limit": memory_limit, "threads": "4"})
        # Cache parquet footers / row-group stats across queries so repeated
        # reads over the same segment set don't re-parse metadata. This is the
        # single most impactful setting for read latency on the log store.
        self._conn.execute("SET enable_object_cache=true")
        # Compaction writes a fresh file each run, so no object cache benefit.
        # Bounded memory so it can't starve the read path under pressure.
        self._compaction_conn = duckdb.connect(config={"memory_limit": memory_limit, "threads": "4"})

    @contextmanager
    def checkout(self, buffer_tables: list[pa.Table]) -> Iterator[tuple[duckdb.DuckDBPyConnection, list[str]]]:
        """Create a cursor and register each RAM table under a unique name.

        Yields ``(cursor, list_of_table_names)`` so callers can UNION ALL
        the names into their SQL without a ``pa.concat_tables`` copy.
        """
        cid = _next_cursor_id()
        cursor = self._conn.cursor()
        names: list[str] = []
        try:
            for i, table in enumerate(buffer_tables):
                name = f"_ram_{cid}_{i}"
                cursor.register(name, table)
                names.append(name)
            yield cursor, names
        finally:
            for name in names:
                cursor.unregister(name)
            cursor.close()

    @contextmanager
    def compaction_checkout(self) -> Iterator[duckdb.DuckDBPyConnection]:
        """Cursor on the dedicated compaction connection. No RAM tables needed —
        compaction only reads already-flushed tmp_*.parquet files.
        """
        cursor = self._compaction_conn.cursor()
        try:
            yield cursor
        finally:
            cursor.close()

    def close(self) -> None:
        self._conn.close()
        self._compaction_conn.close()


class DuckDBLogStore:
    """Log store backed by rotating RAM buffers + Parquet segments.

    Thread-safe. ``_memory_lock`` protects all mutable RAM state; ``_segments_rwlock``
    serializes file ops against in-flight DuckDB reads.
    """

    def __init__(
        self,
        log_dir: Path | None = None,
        *,
        remote_log_dir: str = "",
        max_local_segments: int = DEFAULT_MAX_LOCAL_SEGMENTS,
        max_local_bytes: int = DEFAULT_MAX_LOCAL_BYTES,
        compact_interval_sec: float = DEFAULT_COMPACT_INTERVAL_SEC,
        flush_interval_sec: float = DEFAULT_FLUSH_INTERVAL_SEC,
        compaction_interval_sec: float = DEFAULT_COMPACTION_INTERVAL_SEC,
        max_tmp_segments_before_compact: int = DEFAULT_MAX_TMP_SEGMENTS_BEFORE_COMPACT,
        segment_target_bytes: int = SEGMENT_TARGET_BYTES,
        duckdb_memory_limit: str = _DEFAULT_DUCKDB_MEMORY_LIMIT,
    ):
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
        if log_dir is not None:
            self._log_dir = log_dir
        else:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="iris_controller_logs_")
            self._log_dir = Path(self._temp_dir.name) / "parquet_logs"
        self._log_dir.mkdir(parents=True, exist_ok=True)

        self._remote_log_dir = remote_log_dir
        self._max_local_segments = max_local_segments
        self._max_local_bytes = max_local_bytes
        self._segment_target_bytes = segment_target_bytes
        self._max_tmp_segments_before_compact = max_tmp_segments_before_compact

        # ---- shared mutable state (all guarded by _memory_lock) ----
        self._memory_lock = Lock()
        # Serializes _flush_step so test _force_flush + bg thread can't both
        # be mid-write against the same tmp filename simultaneously.
        self._flush_lock = Lock()
        self._next_seq = _recover_max_seq(self._log_dir)
        self._pending: list[tuple] = []  # hot write list, converted by bg thread
        self._chunks: list[pa.Table] = []  # power-of-2 merged arrow tables
        self._flushing: _SealedBuffer | None = None  # pre-flush snapshot being written
        self._local_segments: deque[_LocalSegment] = deque()

        self._segments_rwlock = _RWLock()

        # Seq numbers are monotonic across the controller's lifetime and each
        # flush assigns a unique range, so any tmp whose [min, max] is fully
        # contained in a logs_ segment's range is a leftover from a prior
        # compaction that crashed between rename and unlink. Drop it on
        # startup so restart reads don't double-count those rows.
        discovered = []
        for p in _discover_segments(self._log_dir):
            min_seq, max_seq = _read_seq_bounds(p)
            discovered.append(
                _LocalSegment(
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

        self._pool = _ConnectionPool(memory_limit=duckdb_memory_limit)

        # ---- background compact + flush + compaction thread ----
        self._compact_rl = RateLimiter(compact_interval_sec)
        self._flush_rl = RateLimiter(flush_interval_sec)
        self._compaction_rl = RateLimiter(compaction_interval_sec)
        self._stop = threading.Event()
        self._wake = threading.Event()
        # Bumped on each successful flush so tests can wait for one to land.
        self._flush_generation = 0
        self._flush_generation_cond = Condition(Lock())
        # Bumped on each successful compaction; symmetric with flush generation.
        self._compaction_generation = 0
        self._compaction_generation_cond = Condition(Lock())
        self._bg_thread = threading.Thread(target=self._bg_loop, name="log_flush", daemon=True)
        self._bg_thread.start()

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def append(self, key: str, entries: list) -> None:
        if not entries:
            return
        self.append_batch([(key, entries)])

    def append_batch(self, items: list[tuple[str, list]]) -> None:
        """Write log entries from multiple keys in a single operation."""
        with self._memory_lock:
            for key, entries in items:
                if not entries:
                    continue
                first_seq = self._next_seq
                self._next_seq += len(entries)
                self._pending.extend(
                    (first_seq + i, key, e.source, e.data, e.timestamp.epoch_ms, e.level) for i, e in enumerate(entries)
                )
            needs_drain = self._ram_bytes_locked() >= self._segment_target_bytes
        if needs_drain:
            self._wake.set()

    # ------------------------------------------------------------------
    # Read API
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
        """Fetch logs for a key or regex pattern.

        If the key contains regex metacharacters, it is interpreted as a
        regular expression and matched with DuckDB's ``regexp_matches()``.
        Otherwise it is treated as an exact key lookup.
        """
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

    def has_logs(self, key: str) -> bool:
        """Check whether any log entries exist for the given key."""
        result = self.get_logs(key, max_lines=1)
        return len(result.entries) > 0

    def cursor(self, key: str):
        """Return a stateful cursor for incremental reads on *key*."""
        from finelog.store import LogCursor

        return LogCursor(self, key)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Stop the bg thread, drain + compact remaining data, and clean up."""
        self._stop.set()
        self._wake.set()
        self._bg_thread.join()
        # Final drain + compaction in the foreground so any lingering tmp
        # segments get merged and uploaded before shutdown. ``compact_single``
        # ensures the last tmp is rewritten to a logs_ segment and offloaded
        # to GCS even when only one exists — otherwise a low-volume shutdown
        # leaves only local tmp_*.parquet and loses data on fresh restart.
        self._compact_step()
        self._flush_step()
        self._compaction_step(compact_single=True)
        self._pool.close()
        if self._temp_dir is not None:
            self._temp_dir.cleanup()

    # ------------------------------------------------------------------
    # Internal: background thread
    # ------------------------------------------------------------------

    def _ram_bytes_locked(self) -> int:
        """Total bytes across every in-RAM holder (pending + chunks + flushing)."""
        chunks_b = sum(t.nbytes for t in self._chunks)
        flushing_b = self._flushing.table.nbytes if self._flushing is not None else 0
        return len(self._pending) * _EST_BYTES_PER_ROW + chunks_b + flushing_b

    def _bg_loop(self) -> None:
        """Drive compact, flush, and compaction on rate-limited schedules.

        Three steps run on distinct cadences:
            - ``_compact_step`` (1s): drain ``_pending`` → Arrow chunk.
            - ``_flush_step`` (60s or 100 MB backpressure): RAM → tmp parquet.
            - ``_compaction_step`` (10 min or >N tmp files): merge tmps into
              one log parquet via DuckDB ``COPY ... ORDER BY ...``, upload to
              GCS, unlink tmps.
        """
        while not self._stop.is_set():
            # Backpressure: if total RAM holders blew past the segment target,
            # drain immediately and reset both rate limiters so the next idle
            # tick doesn't fire redundantly.
            with self._memory_lock:
                force_drain = self._ram_bytes_locked() >= self._segment_target_bytes
                tmp_count = sum(1 for s in self._local_segments if _is_tmp_path(s.path))
                force_compaction = tmp_count > self._max_tmp_segments_before_compact
            if force_drain:
                self._compact_step()
                self._flush_step()
                self._compact_rl.mark_run()
                self._flush_rl.mark_run()
            else:
                if self._compact_rl.should_run():
                    self._compact_step()
                if self._flush_rl.should_run():
                    self._flush_step()
            if force_compaction or self._compaction_rl.should_run():
                self._compaction_step()
                self._compaction_rl.mark_run()

            self._wake.wait(timeout=min(self._compact_rl.time_until_next(), 1.0))
            self._wake.clear()

    def _compact_step(self) -> None:
        """Graduate ``_pending`` row tuples into an Arrow chunk.

        Always drains: readers see ``_chunks`` directly, so leaving rows in
        ``_pending`` would hide them until the next flush. ``_merge_chunks``
        keeps the chunk list logarithmic even under low-rate writes.
        """
        with self._memory_lock:
            rows = self._pending
            if not rows:
                return
            self._pending = []
        table = _build_buffer_table(rows)
        with self._memory_lock:
            self._chunks.append(table)
            self._chunks = _merge_chunks(self._chunks)

    def _flush_step(self) -> None:
        """Seal any RAM data into a Parquet segment on disk.

        The flush lock serializes with any concurrent caller (test
        ``_force_flush`` vs bg thread) so two writers can't race on the same
        tmp filename.
        """
        with self._flush_lock:
            with self._memory_lock:
                if not self._chunks and not self._pending:
                    return
                tables = list(self._chunks)
                if self._pending:
                    tables.append(_build_buffer_table(self._pending))
                    self._pending = []

            # Concat + sort outside the memory lock — hundreds of ms on large
            # flushes. Sort by (key, seq) so Parquet row-group stats on `key`
            # are tight, letting DuckDB skip row groups that don't contain
            # the target key.
            new_table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
            new_table = new_table.sort_by([("key", "ascending"), ("seq", "ascending")])
            seq_col = new_table.column("seq")
            sealed = _SealedBuffer(
                table=new_table,
                min_seq=pc.min(seq_col).as_py(),
                max_seq=pc.max(seq_col).as_py(),
            )

            with self._memory_lock:
                self._chunks = []
                self._flushing = sealed

            try:
                self._write_new_segment(sealed)
            except Exception:
                logger.warning("Flush failed, restoring data to chunks", exc_info=True)
                with self._memory_lock:
                    self._chunks.insert(0, sealed.table)
                    self._flushing = None
                return

            with self._flush_generation_cond:
                self._flush_generation += 1
                self._flush_generation_cond.notify_all()

            self._gc_local_segments()

    def _write_new_segment(self, sealed: _SealedBuffer) -> None:
        """Write a sealed buffer as a new tmp_ Parquet file.

        Not uploaded to GCS — the periodic compaction step will merge this and
        any other tmps into a log_ file and upload that instead. That keeps
        GCS object count bounded regardless of flush frequency.
        """
        filename = _tmp_filename(sealed.min_seq)
        filepath = self._log_dir / filename
        write_start = time.monotonic()
        tmp_path = filepath.with_suffix(".parquet.tmp")
        pq.write_table(
            sealed.table,
            tmp_path,
            compression="zstd",
            row_group_size=_ROW_GROUP_SIZE,
            write_page_index=True,
        )
        tmp_path.rename(filepath)

        seg = _LocalSegment(
            path=str(filepath),
            size_bytes=filepath.stat().st_size,
            min_seq=sealed.min_seq,
            max_seq=sealed.max_seq,
        )
        with self._memory_lock:
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
        """Merge all tmp_ segments into a single logs_ segment, upload, unlink.

        Uses ``COPY (SELECT ... ORDER BY key, seq)`` so the merge streams
        inside DuckDB — never touches pyarrow's concat/sort path, which
        leaks ~45 MB per invocation when reading parquet back.

        By default, a single tmp is left alone — rewriting it buys nothing at
        steady state. ``close()`` passes ``compact_single=True`` so the final
        tmp on shutdown still becomes a logs_ segment and reaches GCS.
        """
        with self._memory_lock:
            tmps = [s for s in self._local_segments if _is_tmp_path(s.path)]
        if not tmps:
            return
        if len(tmps) < 2 and not compact_single:
            return

        tmps.sort(key=lambda s: s.min_seq)
        min_seq = tmps[0].min_seq
        max_seq = max(t.max_seq for t in tmps)
        merged_filename = _log_filename(min_seq)
        merged_path = self._log_dir / merged_filename
        staging_path = merged_path.with_suffix(".parquet.tmp")

        compaction_start = time.monotonic()
        # Self-generated paths from _tmp_filename — no SQL injection surface.
        paths_sql = ",".join(f"'{t.path}'" for t in tmps)
        sql = (
            f"COPY (SELECT * FROM read_parquet([{paths_sql}]) ORDER BY key, seq) "
            f"TO '{staging_path}' "
            f"(FORMAT 'parquet', ROW_GROUP_SIZE {_ROW_GROUP_SIZE}, COMPRESSION 'zstd', COMPRESSION_LEVEL 1)"
        )
        try:
            with self._pool.compaction_checkout() as conn:
                conn.execute(sql)
        except Exception:
            logger.warning("Compaction failed, leaving tmp segments in place", exc_info=True)
            staging_path.unlink(missing_ok=True)
            return

        merged_seg = _LocalSegment(
            path=str(merged_path),
            size_bytes=staging_path.stat().st_size,
            min_seq=min_seq,
            max_seq=max_seq,
        )
        tmp_paths = {t.path for t in tmps}

        self._segments_rwlock.write_acquire()
        try:
            staging_path.rename(merged_path)
            with self._memory_lock:
                new_segments: deque[_LocalSegment] = deque()
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
            for t in tmps:
                try:
                    Path(t.path).unlink(missing_ok=True)
                except Exception:
                    logger.warning("Failed to unlink tmp segment %s", t.path, exc_info=True)
        finally:
            self._segments_rwlock.write_release()

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
        self._offload_to_gcs(merged_filename, merged_path)
        self._gc_local_segments()

    def _gc_local_segments(self) -> None:
        """Drop oldest local Parquet segments if count or size exceeds limits.

        Takes the _segments_rwlock exclusively before unlinking files so that
        in-progress DuckDB reads (which hold the shared read lock) are not
        disrupted by file deletion.
        """
        with self._memory_lock:
            total_bytes = sum(s.size_bytes for s in self._local_segments)
            to_delete: list[tuple[str, int]] = []
            remaining_count = len(self._local_segments)
            remaining_bytes = total_bytes

            while self._local_segments and (
                len(self._local_segments) > self._max_local_segments or total_bytes > self._max_local_bytes
            ):
                oldest = self._local_segments.popleft()
                total_bytes -= oldest.size_bytes
                to_delete.append((oldest.path, oldest.size_bytes))
                remaining_count -= 1
                remaining_bytes -= oldest.size_bytes

        if not to_delete:
            return

        self._segments_rwlock.write_acquire()
        try:
            for path, _ in to_delete:
                try:
                    Path(path).unlink(missing_ok=True)
                except Exception:
                    logger.warning("Failed to delete old segment %s", path, exc_info=True)
        finally:
            self._segments_rwlock.write_release()

        logger.info(
            "GC'd %d local segment(s), freed=%d bytes, remaining=%d segments / %d bytes",
            len(to_delete),
            sum(b for _, b in to_delete),
            remaining_count,
            remaining_bytes,
        )

    def _offload_to_gcs(self, filename: str, filepath: Path) -> None:
        """Copy a Parquet file to GCS (best-effort)."""
        if not self._remote_log_dir:
            return
        remote_path = f"{self._remote_log_dir.rstrip('/')}/{filename}"
        upload_start = time.monotonic()
        try:
            _fsspec_copy(str(filepath), remote_path)
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
    # Test hooks
    # ------------------------------------------------------------------

    def _force_flush(self) -> None:
        """Synchronously compact + flush. For tests only."""
        self._compact_step()
        self._flush_step()

    def _wait_for_flush(self, timeout: float = 10.0) -> None:
        """Block until at least one more flush has landed. For tests only."""
        start_gen = self._flush_generation
        deadline = time.monotonic() + timeout
        with self._flush_generation_cond:
            while self._flush_generation == start_gen:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("timed out waiting for flush")
                self._flush_generation_cond.wait(timeout=remaining)

    def _force_compaction(self) -> None:
        """Synchronously merge all tmp segments into one log segment. For tests only."""
        self._compaction_step()

    def _wait_for_compaction(self, timeout: float = 10.0) -> None:
        """Block until at least one more compaction has landed. For tests only."""
        start_gen = self._compaction_generation
        deadline = time.monotonic() + timeout
        with self._compaction_generation_cond:
            while self._compaction_generation == start_gen:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("timed out waiting for compaction")
                self._compaction_generation_cond.wait(timeout=remaining)

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
        self._segments_rwlock.read_acquire()
        try:
            rows = self._run_read_locked(
                where_parts=where_parts,
                params=params,
                max_lines=max_lines,
                tail=tail,
                include_key_in_select=include_key_in_select,
            )
        finally:
            self._segments_rwlock.read_release()

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
            # Parse attempt_id from the exact key once for all entries.
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
        """Snapshot RAM + segments, run one DuckDB query. Caller holds the
        segments read lock. All pruning is delegated to DuckDB.
        """
        with self._memory_lock:
            segments = list(self._local_segments)
            ram_tables: list[pa.Table] = list(self._chunks)
            if self._flushing is not None:
                ram_tables.append(self._flushing.table)

        segments = _cap_segments(segments)
        parquet_files = [s.path for s in segments]

        where_clause = " AND ".join(where_parts)
        select_cols = (
            "seq, key, source, data, epoch_ms, level" if include_key_in_select else "seq, source, data, epoch_ms, level"
        )
        order = "ORDER BY seq DESC" if (tail and max_lines > 0) else "ORDER BY seq"
        limit = f"LIMIT {max_lines}" if max_lines > 0 else ""

        with self._pool.checkout(ram_tables) as (conn, ram_names):
            source = _build_union_source(parquet_files, ram_names)
            sql = f"SELECT {select_cols} FROM ({source}) WHERE {where_clause} {order} {limit}"
            return conn.execute(sql, params).fetchall()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cap_segments(segments: list[_LocalSegment]) -> list[_LocalSegment]:
    """Cap a segment list at _MAX_PARQUET_BYTES_PER_READ cumulative bytes,
    taking the newest segments first and returning them in ascending min_seq
    order.

    No file-count cap: tmp (small) and log (large) segments coexist after
    compaction, so bytes are the only stable bound on read working-set size.
    """
    if not segments:
        return segments
    newest_first = sorted(segments, key=lambda s: s.min_seq, reverse=True)
    capped: list[_LocalSegment] = []
    total = 0
    for seg in newest_first:
        if capped and total + seg.size_bytes > _MAX_PARQUET_BYTES_PER_READ:
            break
        capped.append(seg)
        total += seg.size_bytes
    capped.sort(key=lambda s: s.min_seq)
    return capped


def _regex_literal_prefix(pattern: str) -> str:
    """Extract the literal prefix from a regex pattern.

    Returns the leading portion of *pattern* that contains no regex
    metacharacters, so it can be used for Parquet range pushdown.
    """
    match = REGEX_META_RE.search(pattern)
    if match is None:
        return pattern
    return pattern[: match.start()]


def _regex_query(pattern: str, cursor: int) -> tuple[list[str], dict]:
    """Build WHERE clauses for a regex pattern.

    Emits ``prefix(key, $prefix_lo)`` on the literal leading portion so DuckDB
    prunes parquet row groups via min/max stats. If the pattern has a
    non-trivial suffix (e.g. ``\\d+:.*``), adds ``regexp_matches()`` as a
    residual filter — the prefix still gets pushed down, the regex only runs
    on surviving rows.
    """
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
    """Append shared WHERE clauses for since_ms, substring, and min_level."""
    if since_ms > 0:
        where_parts.append("epoch_ms > $since_ms")
        params["since_ms"] = since_ms
    if substring_filter:
        where_parts.append("contains(data, $substring)")
        params["substring"] = substring_filter
    if min_level_enum > 0:
        where_parts.append("(level = 0 OR level >= $min_level)")
        params["min_level"] = min_level_enum


def _build_union_source(parquet_files: list[str], ram_table_names: list[str]) -> str:
    """Build a SQL source expression: local Parquet files UNION ALL ram tables.

    File paths are self-generated (``tmp_*.parquet`` / ``logs_*.parquet``) so
    no SQL injection risk from the f-string embedding. RAM table names are
    generated internally (``_ram_<cid>_<i>``).
    """
    parts: list[str] = []
    if parquet_files:
        file_list = ", ".join(f"'{f}'" for f in parquet_files)
        parts.append(f"SELECT * FROM read_parquet([{file_list}])")
    for name in ram_table_names:
        parts.append(f"SELECT * FROM {name}")
    if not parts:
        col_defs = ", ".join(f"NULL::{_DUCKDB_TYPE_MAP[f.type]} AS {f.name}" for f in _PARQUET_SCHEMA)
        return f"SELECT {col_defs} WHERE false"
    return " UNION ALL ".join(parts)
