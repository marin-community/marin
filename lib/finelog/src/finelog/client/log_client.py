# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Top-level finelog client.

:class:`LogClient` is the single client surface for both the legacy log
service (``write_batch`` / ``query``, mirroring today's ``LogPusher`` and
``LogServiceProxy.fetch_logs``) and the new stats service
(``get_table`` / ``drop_table``). Internally every write — log entries
included — flows through a per-namespace :class:`Table`, which owns the
in-memory buffer and the background flush thread that used to live in
``LogPusher``.

The two namespaces presented to callers are:

* ``log`` — the privileged log namespace. ``write_batch`` / ``query``
  delegate to it via the legacy ``LogService`` RPCs (PushLogs / FetchLogs),
  preserving the on-the-wire shape and the existing server impl.
* any other registered namespace — created via ``get_table`` and backed by
  ``StatsService.WriteRows`` for writes and a per-table client-side
  buffer.

The implementation is sync and uses one ``LogServiceClientSync`` plus one
``StatsServiceClientSync`` against the resolved endpoint. Resolver
invalidation on connection-refused mirrors the old ``LogPusher`` behavior;
the cached client is dropped and re-resolved on the next attempt.
"""

from __future__ import annotations

import dataclasses
import io
import logging
import sys
import threading
import time
import types
import typing
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime
from typing import Any

import pyarrow as pa
import pyarrow.ipc as paipc
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.interceptor import Interceptor
from rigging.timing import ExponentialBackoff, RateLimiter

from finelog.errors import (
    InvalidNamespaceError,
    NamespaceNotFoundError,
    QueryResultTooLargeError,
    SchemaConflictError,
    SchemaValidationError,
    StatsError,
)
from finelog.rpc import finelog_stats_pb2 as stats_pb2
from finelog.rpc import logging_pb2
from finelog.rpc.finelog_stats_connect import StatsServiceClientSync
from finelog.rpc.logging_connect import LogServiceClientSync
from finelog.store.schema import (
    Column,
    ColumnType,
    Schema,
    schema_from_proto,
    schema_to_arrow,
    schema_to_proto,
)
from finelog.types import is_retryable_error

# Detached from the root logger: ``RemoteLogHandler`` lives on the root
# logger and writes through the ``log`` Table, so any diagnostics that
# reached the root would feed back into the same buffer — a re-entrant
# loop that silently amplifies during failure storms. Diagnostics go to
# stderr directly with ``propagate = False``.


class _QuietStreamHandler(logging.StreamHandler):
    """StreamHandler that drops emit failures silently.

    The flush thread is a daemon that outlives pytest's stderr capture
    (and interpreter shutdown), so any emit failure is a dead-stream
    symptom of teardown. Swallowing avoids the cascade of
    "--- Logging error ---" tracebacks during teardown.
    """

    def handleError(self, record: logging.LogRecord) -> None:
        pass


logger = logging.getLogger(__name__)
logger.propagate = False
if not logger.handlers:
    _stderr_handler = _QuietStreamHandler(sys.stderr)
    _stderr_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(_stderr_handler)
    if logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Top-level constants matching today's LogPusher behavior.
# ---------------------------------------------------------------------------

LOG_NAMESPACE = "log"
"""Privileged namespace name. ``write_batch`` and ``query`` route here."""

DEFAULT_FLUSH_INTERVAL = 1.0
"""Default time-based flush interval for a Table's bg thread."""

DEFAULT_BATCH_ROWS = 10_000
"""Default row threshold that wakes the bg thread."""

DEFAULT_MAX_BUFFER_BYTES = 16 * 1024 * 1024
"""Default per-Table queue cap in bytes (matches WriteRows max body size)."""

_BACKOFF_INITIAL = 0.5
_BACKOFF_MAX = 30.0

# Floor on per-row byte cost applied to log entries only.
_EST_BYTES_PER_LOG_ENTRY = 256

# Throttle overflow warnings.
_OVERFLOW_LOG_INTERVAL = 5.0


def _format_exc_summary(exc: BaseException) -> str:
    """Collapse a ConnectError to ``ClassName(CODE)``; otherwise ``ClassName: msg``."""
    if isinstance(exc, ConnectError):
        return f"{type(exc).__name__}({exc.code.name})"
    return f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Dataclass schema inference.
# ---------------------------------------------------------------------------

_PRIMITIVE_TYPE_MAP: dict[Any, ColumnType] = {
    str: ColumnType.STRING,
    int: ColumnType.INT64,
    float: ColumnType.FLOAT64,
    bool: ColumnType.BOOL,
    bytes: ColumnType.BYTES,
    datetime: ColumnType.TIMESTAMP_MS,
}


def _strip_optional(annotation: Any) -> tuple[Any, bool]:
    """Return ``(inner, nullable)`` for ``T | None`` annotations.

    Accepts both ``typing.Optional[T]`` and PEP 604 ``T | None`` forms.
    Handles arbitrary union order (``T | None`` and ``None | T``).
    Multi-arm unions other than ``T | None`` are not supported.
    """
    origin = typing.get_origin(annotation)
    if origin is typing.Union or _is_pep604_union(annotation):
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        nullable = len(args) != len(typing.get_args(annotation))
        if nullable and len(args) == 1:
            return args[0], True
        if not nullable:
            return annotation, False
        raise SchemaValidationError(f"unsupported union annotation: {annotation!r}")
    return annotation, False


def _is_pep604_union(annotation: Any) -> bool:
    """Return True for the PEP 604 ``X | Y`` union form."""
    return isinstance(annotation, types.UnionType)


def schema_from_dataclass(cls: type) -> Schema:
    """Infer a :class:`Schema` from a dataclass class.

    Each dataclass field maps to a Column in declaration order. Unsupported
    field types (collections, nested dataclasses, custom classes) raise
    :class:`SchemaValidationError`.

    The inferred ``key_column`` is taken from a ``ClassVar[str]`` named
    ``key_column`` if present, otherwise empty (the server falls back to the
    implicit ``timestamp_ms`` rule).
    """
    if not dataclasses.is_dataclass(cls):
        raise SchemaValidationError(f"{cls!r} is not a dataclass")
    columns: list[Column] = []
    type_hints = typing.get_type_hints(cls, include_extras=False)
    for field in dataclasses.fields(cls):
        annotation = type_hints.get(field.name, field.type)
        inner, nullable = _strip_optional(annotation)
        col_type = _PRIMITIVE_TYPE_MAP.get(inner)
        if col_type is None:
            raise SchemaValidationError(
                f"dataclass {cls.__name__}: field {field.name!r} has unsupported "
                f"type {annotation!r} (supported: str, int, float, bool, bytes, datetime)"
            )
        columns.append(Column(name=field.name, type=col_type, nullable=nullable))
    key_column = getattr(cls, "key_column", "")
    if not isinstance(key_column, str):
        raise SchemaValidationError(
            f"dataclass {cls.__name__}: key_column ClassVar must be a str, got {type(key_column).__name__}"
        )
    return Schema(columns=tuple(columns), key_column=key_column)


# ---------------------------------------------------------------------------
# Table — per-namespace buffered writer.
# ---------------------------------------------------------------------------


class _PendingItem:
    """One queued row payload plus an estimated byte cost."""

    __slots__ = ("payload", "seq", "size_bytes")

    def __init__(self, seq: int, payload: Any, size_bytes: int) -> None:
        self.seq = seq
        self.payload = payload
        self.size_bytes = size_bytes


class Table:
    """Handle to a registered namespace.

    Each Table owns:

    - A bounded in-memory queue (default: 10k rows or 16 MiB, whichever
      first), oldest-drop on overflow.
    - A background flush thread that flushes on size threshold, time
      interval (default 1s), or explicit ``flush()``.
    - Retry/backoff with resolver invalidation on transient server
      failures.

    The Table is created via :meth:`LogClient.get_table`. Closing the
    LogClient drains every Table and joins all flush threads.
    """

    def __init__(
        self,
        *,
        namespace: str,
        schema: Schema,
        flusher: Callable[[str, list[Any]], None],
        querier: Callable[[str], pa.Table] | None = None,
        flush_interval: float = DEFAULT_FLUSH_INTERVAL,
        batch_rows: int = DEFAULT_BATCH_ROWS,
        max_buffer_bytes: int = DEFAULT_MAX_BUFFER_BYTES,
        max_buffer_rows: int = DEFAULT_BATCH_ROWS,
        thread_name: str | None = None,
        row_encoder: Callable[[Any], tuple[Any, int]] | None = None,
    ) -> None:
        self._namespace = namespace
        self._schema = schema
        self._flusher = flusher
        self._querier = querier
        self._flush_interval = flush_interval
        self._batch_rows = batch_rows
        self._max_buffer_bytes = max_buffer_bytes
        self._max_buffer_rows = max_buffer_rows
        self._row_encoder = row_encoder

        self._cond = threading.Condition()
        self._queue: deque[_PendingItem] = deque()
        self._queue_bytes = 0
        self._closing = False
        self._closed = False

        self._pushed_seq = 0
        self._processed_seq = 0

        self._overflow_dropped_pending = 0
        self._overflow_log_limiter = RateLimiter(interval_seconds=_OVERFLOW_LOG_INTERVAL)
        self._backoff = ExponentialBackoff(initial=_BACKOFF_INITIAL, maximum=_BACKOFF_MAX, factor=2.0)

        self._thread = threading.Thread(
            target=self._run,
            name=thread_name or f"finelog-table-{namespace}",
            daemon=True,
        )
        self._thread.start()

    # ------------------------------------------------------------------
    # Public API.
    # ------------------------------------------------------------------

    @property
    def namespace(self) -> str:
        return self._namespace

    @property
    def schema(self) -> Schema:
        return self._schema

    def write(self, rows: Iterable[Any]) -> None:
        """Buffer ``rows`` for write. Never blocks the caller."""
        rows_list = list(rows)
        if not rows_list:
            return
        with self._cond:
            if self._closing or self._closed:
                raise RuntimeError(f"Table({self._namespace}) is closed")
            for row in rows_list:
                if self._row_encoder is not None:
                    payload, size = self._row_encoder(row)
                else:
                    # Log path: payload is (key, [LogEntry, ...]); size is the
                    # sum of the raw entry bytes plus a fixed header per entry.
                    payload = row
                    _key, entries = row
                    size = sum(_EST_BYTES_PER_LOG_ENTRY + len(e.data) for e in entries)
                self._pushed_seq += 1
                self._queue.append(_PendingItem(self._pushed_seq, payload, size))
                self._queue_bytes += size
            self._trim_oldest_locked()
            if len(self._queue) >= self._batch_rows or self._queue_bytes >= self._max_buffer_bytes:
                self._cond.notify_all()

    def query(self, sql: str, *, max_rows: int = 100_000) -> pa.Table:
        """Run Postgres-flavored SQL against the stats service.

        Reference namespaces by name in the FROM clause (e.g.
        ``FROM "iris.worker"``); the server registers a DuckDB view per
        registered namespace before executing and never rewrites the SQL.

        The result is materialized into a ``pa.Table``. If the row count
        exceeds ``max_rows``, raises :class:`QueryResultTooLargeError`
        rather than silently truncating — callers can re-issue with a
        higher cap or add a LIMIT/aggregation.

        DuckDB-side errors (catalog, parser, binder) propagate as
        :class:`SchemaValidationError` (the proto carries them as
        INVALID_ARGUMENT). Network-level failures propagate as
        ConnectionError / ConnectError.
        """
        if self._querier is None:
            raise StatsError(f"Table({self._namespace}) has no query path (log namespace?)")
        result = self._querier(sql)
        if result.num_rows > max_rows:
            raise QueryResultTooLargeError(
                f"query returned {result.num_rows} rows, exceeds max_rows={max_rows} "
                f"(add a LIMIT or pass a higher max_rows)"
            )
        return result

    def flush(self, timeout: float | None = None) -> bool:
        """Block until rows enqueued before this call have been processed."""
        with self._cond:
            target = self._pushed_seq
            if target == 0 or self._processed_seq >= target:
                return True
            self._cond.notify_all()
            deadline = (time.monotonic() + timeout) if timeout is not None else None
            while self._processed_seq < target:
                if self._closed:
                    return self._processed_seq >= target
                if deadline is None:
                    self._cond.wait(timeout=1.0)
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    self._cond.wait(timeout=remaining)
            return True

    def close(self) -> None:
        """Stop the flush thread after one best-effort drain."""
        with self._cond:
            if self._closed:
                return
            self._closing = True
            self._cond.notify_all()
        self._thread.join(timeout=max(self._flush_interval * 2, 10.0))
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    # ------------------------------------------------------------------
    # Internal — buffer management (caller holds ``_cond``).
    # ------------------------------------------------------------------

    def _trim_oldest_locked(self) -> None:
        dropped = 0
        max_dropped_seq = 0
        while len(self._queue) > self._max_buffer_rows or self._queue_bytes > self._max_buffer_bytes:
            if not self._queue:
                break
            item = self._queue.popleft()
            self._queue_bytes -= item.size_bytes
            if item.seq > max_dropped_seq:
                max_dropped_seq = item.seq
            dropped += 1
        if dropped:
            self._overflow_dropped_pending += dropped
            if self._overflow_log_limiter.should_run():
                logger.warning(
                    "Table(%s) buffer overflow: dropped %d oldest rows (rows=%d/%d, bytes=%d/%d)",
                    self._namespace,
                    self._overflow_dropped_pending,
                    len(self._queue),
                    self._max_buffer_rows,
                    self._queue_bytes,
                    self._max_buffer_bytes,
                )
                self._overflow_dropped_pending = 0
            if max_dropped_seq > self._processed_seq:
                self._processed_seq = max_dropped_seq
                self._cond.notify_all()

    def _take_queue_locked(self) -> list[_PendingItem]:
        items = list(self._queue)
        self._queue.clear()
        self._queue_bytes = 0
        return items

    def _rebuffer_at_head_locked(self, items: list[_PendingItem]) -> None:
        for item in reversed(items):
            self._queue.appendleft(item)
            self._queue_bytes += item.size_bytes
        self._trim_oldest_locked()

    # ------------------------------------------------------------------
    # Internal — drain thread.
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._closing and not self._queue:
                    self._cond.wait(timeout=self._flush_interval)
                if not self._queue:
                    return
                items = self._take_queue_locked()

            sent_max_seq, unsent = self._send(items)
            with self._cond:
                if sent_max_seq > self._processed_seq:
                    self._processed_seq = sent_max_seq
                    self._cond.notify_all()
            if not unsent:
                self._backoff.reset()
                continue

            with self._cond:
                if self._closing:
                    if unsent[-1].seq > self._processed_seq:
                        self._processed_seq = unsent[-1].seq
                        self._cond.notify_all()
                    return
                self._rebuffer_at_head_locked(unsent)
            deadline = time.monotonic() + self._backoff.next_interval()
            with self._cond:
                while not self._closing:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._cond.wait(timeout=remaining)

    def _send(self, items: list[_PendingItem]) -> tuple[int, list[_PendingItem]]:
        """Send ``items`` via the flusher. Returns ``(max_sent_seq, unsent)``."""
        if not items:
            return 0, []
        payloads = [item.payload for item in items]
        try:
            self._flusher(self._namespace, payloads)
        except Exception as exc:
            retryable = is_retryable_error(exc) or isinstance(exc, (ConnectionError, OSError, TimeoutError))
            summary = _format_exc_summary(exc)
            logger.warning(
                "Table(%s) send failure (%d rows, retryable=%s): %s",
                self._namespace,
                len(items),
                retryable,
                summary,
            )
            if not retryable:
                # Non-retryable failures drop the batch — surfacing them as
                # blocked queue would back up indefinitely. Mirrors the old
                # LogPusher (which only re-buffered for retryable errors).
                return items[-1].seq, []
            return 0, items
        return items[-1].seq, []


# ---------------------------------------------------------------------------
# LogClient — top-level client.
# ---------------------------------------------------------------------------


class LogClient:
    """Domain client for the finelog process.

    Hides Connect/RPC details. Both LogService methods (``write_batch``,
    ``query``) and StatsService methods (``get_table``, ``drop_table``) are
    exposed through this single surface.

    Construct with :meth:`connect`. ``close()`` drains every open Table and
    closes the underlying RPC connections; subsequent writes raise.
    """

    def __init__(
        self,
        *,
        server_url: str,
        resolver: Callable[[str], str] | None = None,
        timeout_ms: int = 10_000,
        interceptors: Iterable[Interceptor] = (),
    ) -> None:
        self._server_url = server_url
        self._resolver: Callable[[str], str] = resolver if resolver is not None else (lambda url: url)
        self._timeout_ms = timeout_ms
        self._interceptors = tuple(interceptors)

        self._lock = threading.Lock()
        self._closed = False
        self._log_client: LogServiceClientSync | None = None
        self._stats_client: StatsServiceClientSync | None = None

        # Open Tables keyed by namespace. The log namespace's Table is
        # constructed lazily on first write_batch / query so connect()
        # does not pay the resolver cost when a caller only needs stats.
        self._tables: dict[str, Table] = {}

    # ------------------------------------------------------------------
    # Construction / lifecycle.
    # ------------------------------------------------------------------

    @staticmethod
    def connect(
        endpoint: str | tuple[str, int],
        *,
        resolver: Callable[[str], str] | None = None,
        timeout_ms: int = 10_000,
        interceptors: Iterable[Interceptor] = (),
    ) -> LogClient:
        """Construct a LogClient against ``endpoint``.

        ``endpoint`` is either an HTTP URL string or a ``(host, port)``
        tuple. The optional ``resolver`` mirrors today's ``LogPusher``: the
        client passes ``server_url`` through it before constructing the
        underlying RPC client, so callers who advertise their endpoint via
        a registry (e.g. iris's ``/system/log-server``) can plug that lookup
        in here.
        """
        if isinstance(endpoint, tuple):
            host, port = endpoint
            server_url = f"http://{host}:{port}"
        else:
            server_url = endpoint
        return LogClient(
            server_url=server_url,
            resolver=resolver,
            timeout_ms=timeout_ms,
            interceptors=interceptors,
        )

    def close(self) -> None:
        """Drain and join every open Table, then close the RPC clients.

        Tables are drained *before* the client is marked closed so the bg
        flush threads can complete one final send through the cached RPC
        clients. Marking ``_closed`` first would race with the drain and
        leave queued rows undelivered.
        """
        with self._lock:
            if self._closed:
                return
            tables = list(self._tables.values())
            self._tables.clear()
        for tbl in tables:
            tbl.close()
        with self._lock:
            self._closed = True
            log_client = self._log_client
            stats_client = self._stats_client
            self._log_client = None
            self._stats_client = None
        if log_client is not None:
            try:
                log_client.close()
            except Exception:
                logger.debug("LogClient.close: log client close raised", exc_info=True)
        if stats_client is not None:
            try:
                stats_client.close()
            except Exception:
                logger.debug("LogClient.close: stats client close raised", exc_info=True)

    # ------------------------------------------------------------------
    # Log-side surface (legacy LogService).
    # ------------------------------------------------------------------

    def write_batch(self, key: str, messages: Sequence[logging_pb2.LogEntry]) -> None:
        """Append ``messages`` to the ``log`` namespace under ``key``."""
        if not messages:
            return
        table = self._get_log_table()
        table.write([(key, list(messages))])

    def query(self, request: logging_pb2.FetchLogsRequest) -> logging_pb2.FetchLogsResponse:
        """Read from the ``log`` namespace via the legacy FetchLogs RPC."""
        client = self._get_log_client()
        return client.fetch_logs(request)

    def flush(self, timeout: float | None = None) -> bool:
        """Flush the ``log`` namespace's Table, if any."""
        table = self._tables.get(LOG_NAMESPACE)
        if table is None:
            return True
        return table.flush(timeout=timeout)

    # ------------------------------------------------------------------
    # Stats-side surface.
    # ------------------------------------------------------------------

    def get_table(self, namespace: str, schema: type | Schema) -> Table:
        """Idempotently register ``namespace`` and return a Table handle."""
        if namespace == LOG_NAMESPACE:
            raise InvalidNamespaceError("use write_batch/query for the privileged 'log' namespace")
        if isinstance(schema, Schema):
            requested = schema
        elif isinstance(schema, type):
            requested = schema_from_dataclass(schema)
        else:
            raise SchemaValidationError(f"schema must be a Schema or a dataclass class, got {type(schema).__name__}")

        existing = self._tables.get(namespace)
        if existing is not None:
            return existing

        client = self._get_stats_client()
        try:
            response = client.register_table(
                stats_pb2.RegisterTableRequest(
                    namespace=namespace,
                    schema=schema_to_proto(requested),
                )
            )
        except ConnectError as exc:
            raise _translate_connect_error(exc) from exc
        effective = schema_from_proto(response.effective_schema)
        arrow_schema = schema_to_arrow(effective)
        table = Table(
            namespace=namespace,
            schema=effective,
            flusher=lambda ns, rows: self._stats_flush(ns, rows),
            querier=self._stats_query,
            row_encoder=_make_stats_row_encoder(arrow_schema, effective),
        )
        with self._lock:
            if self._closed:
                table.close()
                raise RuntimeError("LogClient is closed")
            existing = self._tables.get(namespace)
            if existing is not None:
                # Lost the race; close ours and return the winner.
                table.close()
                return existing
            self._tables[namespace] = table
        return table

    def drop_table(self, namespace: str) -> None:
        """Remove ``namespace`` from the registry and delete its local data."""
        if namespace == LOG_NAMESPACE:
            raise InvalidNamespaceError("cannot drop the privileged 'log' namespace")
        # Close any local Table first so its in-flight rows do not race the
        # registry deletion. Per the design's drop contract, in-flight stats
        # data is not durable — closing here makes that explicit on the
        # client side too.
        with self._lock:
            tbl = self._tables.pop(namespace, None)
        if tbl is not None:
            tbl.close()
        client = self._get_stats_client()
        try:
            client.drop_table(stats_pb2.DropTableRequest(namespace=namespace))
        except ConnectError as exc:
            translated = _translate_connect_error(exc)
            if isinstance(translated, NamespaceNotFoundError):
                # Spec: "No-op (does not raise) if the namespace was not registered."
                return
            raise translated from exc

    # ------------------------------------------------------------------
    # Internals.
    # ------------------------------------------------------------------

    def _get_log_table(self) -> Table:
        with self._lock:
            tbl = self._tables.get(LOG_NAMESPACE)
            if tbl is not None:
                return tbl
            if self._closed:
                raise RuntimeError("LogClient is closed")
            tbl = Table(
                namespace=LOG_NAMESPACE,
                schema=Schema(columns=()),  # log table schema is server-managed
                flusher=self._log_flush,
                thread_name="finelog-log-client",
            )
            self._tables[LOG_NAMESPACE] = tbl
            return tbl

    def _get_log_client(self) -> LogServiceClientSync:
        with self._lock:
            if self._closed:
                raise RuntimeError("LogClient is closed")
            if self._log_client is not None:
                return self._log_client
            address = self._resolve()
            self._log_client = LogServiceClientSync(
                address=address,
                timeout_ms=self._timeout_ms,
                interceptors=self._interceptors,
            )
            logger.info("LogClient resolved %s -> %s (log)", self._server_url, address)
            return self._log_client

    def _get_stats_client(self) -> StatsServiceClientSync:
        with self._lock:
            if self._closed:
                raise RuntimeError("LogClient is closed")
            if self._stats_client is not None:
                return self._stats_client
            address = self._resolve()
            self._stats_client = StatsServiceClientSync(
                address=address,
                timeout_ms=self._timeout_ms,
                interceptors=self._interceptors,
            )
            logger.info("LogClient resolved %s -> %s (stats)", self._server_url, address)
            return self._stats_client

    def _resolve(self) -> str:
        address = self._resolver(self._server_url)
        if not address:
            raise ConnectionError(f"LogClient resolver returned empty address for {self._server_url!r}")
        return address

    def _invalidate(self, reason: str) -> None:
        with self._lock:
            log_client = self._log_client
            stats_client = self._stats_client
            self._log_client = None
            self._stats_client = None
        if log_client is None and stats_client is None:
            return
        logger.info("LogClient: invalidating cached endpoint for %s (%s)", self._server_url, reason)
        for c in (log_client, stats_client):
            if c is None:
                continue
            try:
                c.close()
            except Exception:
                logger.debug("LogClient invalidate: cached client close raised", exc_info=True)

    # --- log namespace flusher ---

    def _log_flush(self, _ns: str, payloads: list[Any]) -> None:
        """Flush log-namespace items. Each payload is ``(key, [LogEntry, ...])``."""
        # Group by key — multiple writes to different keys may have piled up.
        grouped: dict[str, list[logging_pb2.LogEntry]] = {}
        for key, entries in payloads:
            grouped.setdefault(key, []).extend(entries)
        client = self._get_log_client()
        try:
            for key, entries in grouped.items():
                client.push_logs(logging_pb2.PushLogsRequest(key=key, entries=entries))
        except ConnectError as exc:
            if is_retryable_error(exc):
                self._invalidate(_format_exc_summary(exc))
            raise
        except (ConnectionError, OSError, TimeoutError) as exc:
            self._invalidate(_format_exc_summary(exc))
            raise

    # --- stats namespace query ---

    def _stats_query(self, sql: str) -> pa.Table:
        """Run a SQL query and decode the Arrow IPC response into a pa.Table.

        Used by every Table.query — the underlying connection is shared
        across namespaces because the server resolves namespaces from the
        FROM clause, not the request.
        """
        client = self._get_stats_client()
        try:
            response = client.query(stats_pb2.QueryRequest(sql=sql))
        except ConnectError as exc:
            if is_retryable_error(exc):
                self._invalidate(_format_exc_summary(exc))
            raise _translate_connect_error(exc) from exc
        except (ConnectionError, OSError, TimeoutError) as exc:
            self._invalidate(_format_exc_summary(exc))
            raise
        reader = paipc.open_stream(pa.BufferReader(bytes(response.arrow_ipc)))
        return reader.read_all()

    # --- stats namespace flusher ---

    def _stats_flush(self, namespace: str, batches: list[Any]) -> None:
        """Flush pre-built Arrow RecordBatches to the stats service."""
        combined = pa.concat_batches(batches)
        sink = io.BytesIO()
        with paipc.new_stream(sink, combined.schema) as writer:
            writer.write_batch(combined)
        batch_bytes = sink.getvalue()
        client = self._get_stats_client()
        try:
            client.write_rows(stats_pb2.WriteRowsRequest(namespace=namespace, arrow_ipc=batch_bytes))
        except ConnectError as exc:
            if is_retryable_error(exc):
                self._invalidate(_format_exc_summary(exc))
            raise
        except (ConnectionError, OSError, TimeoutError) as exc:
            self._invalidate(_format_exc_summary(exc))
            raise


# ---------------------------------------------------------------------------
# Row → Arrow conversion for stats Tables.
# ---------------------------------------------------------------------------


def _row_to_record_batch(row: Any, arrow_schema: pa.Schema, schema: Schema) -> pa.RecordBatch:
    """Convert a single dataclass (or attribute-bearing) row to a 1-row RecordBatch.

    Missing nullable columns are filled with NULL; missing non-nullable columns
    raise :class:`SchemaValidationError`. ``datetime`` values are accepted
    directly by pyarrow for timestamp(ms) columns.
    """
    columns: list[pa.Array] = []
    for col, field in zip(schema.columns, arrow_schema, strict=True):
        value = _extract_row_value(row, col.name)
        if value is _MISSING:
            if not col.nullable:
                raise SchemaValidationError(f"row missing required (non-nullable) column {col.name!r}")
            raw: list[Any] = [None]
        elif value is None:
            if not col.nullable:
                raise SchemaValidationError(f"row has None for non-nullable column {col.name!r}")
            raw = [None]
        else:
            raw = [value]
        try:
            arr = pa.array(raw, type=field.type, from_pandas=False)
        except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError, ValueError) as exc:
            raise SchemaValidationError(f"column {col.name!r}: failed to encode row as {col.type.value}: {exc}") from exc
        columns.append(arr)
    return pa.RecordBatch.from_arrays(columns, schema=arrow_schema)


def _make_stats_row_encoder(arrow_schema: pa.Schema, schema: Schema) -> Callable[[Any], tuple[pa.RecordBatch, int]]:
    """Return a ``row_encoder`` for stats tables.

    The encoder converts each dataclass row to a 1-row RecordBatch and
    returns ``(batch, batch.nbytes)`` so the byte cap tracks the actual
    Arrow wire representation.
    """

    def encode(row: Any) -> tuple[pa.RecordBatch, int]:
        batch = _row_to_record_batch(row, arrow_schema, schema)
        return batch, batch.nbytes

    return encode


_MISSING = object()


def _extract_row_value(row: Any, name: str) -> Any:
    return getattr(row, name, _MISSING)


# ---------------------------------------------------------------------------
# ConnectError translation.
# ---------------------------------------------------------------------------


def _translate_connect_error(exc: ConnectError) -> Exception:
    """Map server-side Connect codes to public error types."""
    msg = str(exc)
    if exc.code == Code.NOT_FOUND:
        return NamespaceNotFoundError(msg)
    if exc.code == Code.INVALID_ARGUMENT:
        # Cannot tell SchemaValidation vs InvalidNamespace from Connect codes;
        # use SchemaValidationError as the structural error and let callers
        # match on the message if they need finer discrimination. Tests assert
        # against either. Both are subclasses of StatsError.
        if "namespace" in msg.lower() and "name" in msg.lower():
            return InvalidNamespaceError(msg)
        return SchemaValidationError(msg)
    if exc.code == Code.FAILED_PRECONDITION:
        return SchemaConflictError(msg)
    return StatsError(msg)
