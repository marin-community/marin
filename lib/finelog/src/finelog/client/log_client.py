# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

import pyarrow as pa
import pyarrow.ipc as paipc
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.interceptor import Interceptor
from rigging.log_setup import LOG_DATEFMT, LOG_FORMAT, LevelPrefixFormatter
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
from finelog.store.log_namespace import LOG_REGISTERED_SCHEMA
from finelog.store.schema import (
    IMPLICIT_SEQ_COLUMN,
    Column,
    ColumnTypeValue,
    Schema,
    schema_from_proto,
    schema_to_arrow,
    schema_to_proto,
)
from finelog.types import is_retryable_error


class _QuietStreamHandler(logging.StreamHandler):
    # The flush thread is a daemon that outlives pytest's stderr capture (and
    # interpreter shutdown). Swallow emit failures so teardown does not
    # cascade "--- Logging error ---" tracebacks.

    def handleError(self, record: logging.LogRecord) -> None:
        pass


logger = logging.getLogger(__name__)
# RemoteLogHandler lives on the root logger and writes through this Table;
# propagating to root would re-enter the same buffer during failure storms.
logger.propagate = False
if not logger.handlers:
    _stderr_handler = _QuietStreamHandler(sys.stderr)
    _stderr_handler.setFormatter(LevelPrefixFormatter(fmt=LOG_FORMAT, datefmt=LOG_DATEFMT))
    logger.addHandler(_stderr_handler)
    if logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)


LOG_NAMESPACE = "log"
DEFAULT_FLUSH_INTERVAL = 1.0
DEFAULT_BATCH_ROWS = 10_000
# Per-Table queue cap in bytes. Matches WriteRows max body size.
DEFAULT_MAX_BUFFER_BYTES = 16 * 1024 * 1024

_BACKOFF_INITIAL = 0.5
_BACKOFF_MAX = 30.0
_OVERFLOW_LOG_INTERVAL = 5.0


class FlushResult(StrEnum):
    SUCCEEDED = "succeeded"
    TIMEOUT = "timeout"


def _format_exc_summary(exc: BaseException) -> str:
    if isinstance(exc, ConnectError):
        return f"{type(exc).__name__}({exc.code.name})"
    return f"{type(exc).__name__}: {exc}"


_PRIMITIVE_TYPE_MAP: dict[Any, ColumnTypeValue] = {
    str: stats_pb2.COLUMN_TYPE_STRING,
    int: stats_pb2.COLUMN_TYPE_INT64,
    float: stats_pb2.COLUMN_TYPE_FLOAT64,
    bool: stats_pb2.COLUMN_TYPE_BOOL,
    bytes: stats_pb2.COLUMN_TYPE_BYTES,
    datetime: stats_pb2.COLUMN_TYPE_TIMESTAMP_MS,
}


def _strip_optional(annotation: Any) -> tuple[Any, bool]:
    """Return ``(inner, nullable)`` for ``T | None`` annotations.

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
    return isinstance(annotation, types.UnionType)


def schema_from_dataclass(cls: type) -> Schema:
    """Infer a :class:`Schema` from a dataclass class.

    The inferred ``key_column`` is taken from a ``ClassVar[str]`` named
    ``key_column`` if present, otherwise empty (the server falls back to the
    implicit ``timestamp_ms`` rule).
    """
    if not dataclasses.is_dataclass(cls):
        raise SchemaValidationError(f"{cls!r} is not a dataclass")
    columns: list[Column] = []
    type_hints = typing.get_type_hints(cls, include_extras=False)
    for field in dataclasses.fields(cls):
        if field.name == IMPLICIT_SEQ_COLUMN:
            raise SchemaValidationError(
                f"dataclass {cls.__name__}: field {field.name!r} is reserved " f"(server-assigned implicit column)"
            )
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


@dataclass(slots=True)
class _PendingItem:
    seq: int
    payload: Any
    size_bytes: int


class Table:
    """Handle to a registered namespace.

    Owns a bounded in-memory queue (oldest-drop on overflow), a background
    flush thread, and retry/backoff with resolver invalidation on transient
    server failures. Created via :meth:`LogClient.get_table`; closing the
    LogClient drains every Table.
    """

    def __init__(
        self,
        *,
        namespace: str,
        schema: Schema,
        flusher: Callable[[str, pa.RecordBatch], None],
        querier: Callable[[str], pa.Table] | None = None,
        flush_interval: float = DEFAULT_FLUSH_INTERVAL,
        batch_rows: int = DEFAULT_BATCH_ROWS,
        max_buffer_bytes: int = DEFAULT_MAX_BUFFER_BYTES,
        max_buffer_rows: int = DEFAULT_BATCH_ROWS,
        thread_name: str | None = None,
    ) -> None:
        self._namespace = namespace
        self._schema = schema
        self._arrow_schema = schema_to_arrow(schema)
        self._flusher = flusher
        self._querier = querier
        self._flush_interval = flush_interval
        self._batch_rows = batch_rows
        self._max_buffer_bytes = max_buffer_bytes
        self._max_buffer_rows = max_buffer_rows

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

    @property
    def namespace(self) -> str:
        return self._namespace

    @property
    def schema(self) -> Schema:
        return self._schema

    def write(self, rows: Iterable[Any]) -> None:
        """Buffer ``rows`` for write. Never blocks the caller.

        Caller-side cost is a cheap byte estimate per row plus an enqueue —
        Arrow construction is deferred to the bg flush thread so a single
        producer batch is encoded once instead of once per row.
        """
        rows_list = list(rows)
        if not rows_list:
            return
        sizes = [_estimate_row_size(row, self._schema.columns) for row in rows_list]
        with self._cond:
            if self._closing or self._closed:
                raise RuntimeError(f"Table({self._namespace}) is closed")
            for row, size in zip(rows_list, sizes, strict=True):
                self._pushed_seq += 1
                self._queue.append(_PendingItem(self._pushed_seq, row, size))
                self._queue_bytes += size
            self._trim_oldest_locked()
            if len(self._queue) >= self._batch_rows or self._queue_bytes >= self._max_buffer_bytes:
                self._cond.notify_all()

    def query(self, sql: str, *, max_rows: int = 100_000) -> pa.Table:
        """Run Postgres-flavored SQL against the stats service.

        Reference namespaces by name in the FROM clause (e.g.
        ``FROM "iris.worker"``). Raises :class:`QueryResultTooLargeError`
        if the row count exceeds ``max_rows``.
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

    def flush(self, timeout: float | None = None) -> FlushResult:
        """Block until rows enqueued before this call have been processed."""
        with self._cond:
            target = self._pushed_seq
            if target == 0 or self._processed_seq >= target:
                return FlushResult.SUCCEEDED
            self._cond.notify_all()
            deadline = (time.monotonic() + timeout) if timeout is not None else None
            while self._processed_seq < target:
                if self._closed:
                    return FlushResult.SUCCEEDED if self._processed_seq >= target else FlushResult.TIMEOUT
                if deadline is None:
                    self._cond.wait(timeout=1.0)
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return FlushResult.TIMEOUT
                    self._cond.wait(timeout=remaining)
            return FlushResult.SUCCEEDED

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
        """Send ``items`` via the flusher. Returns ``(max_sent_seq, unsent)``.

        Builds the entire batch's RecordBatch in one pass on the bg thread,
        amortizing Arrow construction across the whole queue rather than
        per-row at write time.
        """
        if not items:
            return 0, []
        rows = [item.payload for item in items]
        try:
            batch = _rows_to_record_batch(rows, self._arrow_schema, self._schema)
            self._flusher(self._namespace, batch)
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
                # Non-retryable failures drop the batch; rebuffering would
                # back up the queue indefinitely.
                return items[-1].seq, []
            return 0, items
        return items[-1].seq, []


class LogClient:
    """Domain client for the finelog process.

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
        self._stats_client: StatsServiceClientSync | None = None
        self._log_service_client: LogServiceClientSync | None = None

        # The log namespace's Table is constructed lazily on first
        # write_batch/fetch_logs so connect() does not pay the resolver cost
        # when a caller only needs stats.
        self._tables: dict[str, Table] = {}

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
        tuple. The optional ``resolver`` is invoked on each re-resolve so
        callers who advertise their endpoint via a registry (e.g. iris's
        ``/system/log-server``) can plug that lookup in here.
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

        Tables are drained before the client is marked closed so the bg
        flush threads can complete one final send.
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
            stats_client = self._stats_client
            log_service_client = self._log_service_client
            self._stats_client = None
            self._log_service_client = None
        if stats_client is not None:
            stats_client.close()
        if log_service_client is not None:
            log_service_client.close()

    def write_batch(self, key: str, messages: Sequence[logging_pb2.LogEntry]) -> None:
        """Append ``messages`` to the ``log`` namespace under ``key``."""
        if not messages:
            return
        table = self._get_log_table()
        table.write(_log_entries_to_rows(key, messages))

    def fetch_logs(self, request: logging_pb2.FetchLogsRequest) -> logging_pb2.FetchLogsResponse:
        """Read from the ``log`` namespace via ``LogService.FetchLogs``.

        Uses the purpose-built log read path (warm cached DuckDB connection,
        single-namespace scan, tail-aware short-circuit) instead of the
        general SQL ``StatsService.Query`` surface.
        """
        client = self._get_log_service_client()
        try:
            return client.fetch_logs(request)
        except ConnectError as exc:
            if is_retryable_error(exc):
                self._invalidate(_format_exc_summary(exc))
            raise _translate_connect_error(exc) from exc
        except (ConnectionError, OSError, TimeoutError) as exc:
            self._invalidate(_format_exc_summary(exc))
            raise

    def flush(self, timeout: float | None = None) -> FlushResult:
        """Flush the ``log`` namespace's Table, if any."""
        table = self._tables.get(LOG_NAMESPACE)
        if table is None:
            return FlushResult.SUCCEEDED
        return table.flush(timeout=timeout)

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
        table = Table(
            namespace=namespace,
            schema=effective,
            flusher=self._stats_flush,
            querier=self._stats_query,
        )
        with self._lock:
            if self._closed:
                table.close()
                raise RuntimeError("LogClient is closed")
            existing = self._tables.get(namespace)
            if existing is not None:
                table.close()
                return existing
            self._tables[namespace] = table
        return table

    def drop_table(self, namespace: str) -> None:
        """Remove ``namespace`` from the registry and delete its local data."""
        if namespace == LOG_NAMESPACE:
            raise InvalidNamespaceError("cannot drop the privileged 'log' namespace")
        # Close the local Table first so in-flight rows do not race the
        # registry deletion.
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
                return
            raise translated from exc

    def _get_log_table(self) -> Table:
        with self._lock:
            tbl = self._tables.get(LOG_NAMESPACE)
            if tbl is not None:
                return tbl
            if self._closed:
                raise RuntimeError("LogClient is closed")
            # ``log`` is auto-registered server-side; skip register_table.
            tbl = Table(
                namespace=LOG_NAMESPACE,
                schema=LOG_REGISTERED_SCHEMA,
                flusher=self._stats_flush,
                thread_name="finelog-log-client",
            )
            self._tables[LOG_NAMESPACE] = tbl
            return tbl

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

    def _get_log_service_client(self) -> LogServiceClientSync:
        with self._lock:
            if self._closed:
                raise RuntimeError("LogClient is closed")
            if self._log_service_client is not None:
                return self._log_service_client
            address = self._resolve()
            self._log_service_client = LogServiceClientSync(
                address=address,
                timeout_ms=self._timeout_ms,
                interceptors=self._interceptors,
            )
            logger.info("LogClient resolved %s -> %s (log)", self._server_url, address)
            return self._log_service_client

    def _resolve(self) -> str:
        address = self._resolver(self._server_url)
        if not address:
            raise ConnectionError(f"LogClient resolver returned empty address for {self._server_url!r}")
        return address

    def _invalidate(self, reason: str) -> None:
        with self._lock:
            stats_client = self._stats_client
            log_service_client = self._log_service_client
            self._stats_client = None
            self._log_service_client = None
        if stats_client is None and log_service_client is None:
            return
        logger.info("LogClient: invalidating cached endpoint for %s (%s)", self._server_url, reason)
        if stats_client is not None:
            stats_client.close()
        if log_service_client is not None:
            log_service_client.close()

    def _stats_query(self, sql: str) -> pa.Table:
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

    def _stats_flush(self, namespace: str, batch: pa.RecordBatch) -> None:
        sink = io.BytesIO()
        with paipc.new_stream(sink, batch.schema) as writer:
            writer.write_batch(batch)
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


def _rows_to_record_batch(rows: list[Any], arrow_schema: pa.Schema, schema: Schema) -> pa.RecordBatch:
    """Build a single RecordBatch from ``rows``.

    Builds one Arrow array per column for the entire batch — much cheaper
    than constructing a 1-row RecordBatch per row and concatenating later.

    Missing or None values for non-nullable columns raise
    :class:`SchemaValidationError`; nullable columns become NULL.
    """
    n = len(rows)
    columns: list[pa.Array] = []
    for col, field in zip(schema.columns, arrow_schema, strict=True):
        values: list[Any] = [None] * n
        if col.nullable:
            for i, row in enumerate(rows):
                values[i] = getattr(row, col.name, None)
        else:
            for i, row in enumerate(rows):
                v = getattr(row, col.name, None)
                if v is None:
                    raise SchemaValidationError(f"row missing required (non-nullable) column {col.name!r}")
                values[i] = v
        try:
            arr = pa.array(values, type=field.type, from_pandas=False)
        except (pa.ArrowInvalid, pa.ArrowTypeError, TypeError, ValueError) as exc:
            raise SchemaValidationError(
                f"column {col.name!r}: failed to encode batch as " f"{stats_pb2.ColumnType.Name(col.type)}: {exc}"
            ) from exc
        columns.append(arr)
    return pa.RecordBatch.from_arrays(columns, schema=arrow_schema)


def _estimate_row_size(row: Any, columns: tuple[Column, ...]) -> int:
    """Cheap byte estimate for buffer-cap accounting.

    Strings count their length; everything else is treated as 8 bytes. This
    is rough on purpose — the buffer cap only needs an order-of-magnitude
    estimate, and the real Arrow encode runs once per batch on the flush
    thread.
    """
    total = 0
    for col in columns:
        v = getattr(row, col.name, None)
        if isinstance(v, str):
            total += len(v)
        else:
            total += 8
    return total


def _log_entries_to_rows(key: str, messages: Sequence[logging_pb2.LogEntry]) -> list[Any]:
    rows: list[Any] = []
    for entry in messages:
        rows.append(
            types.SimpleNamespace(
                key=key,
                source=entry.source,
                data=entry.data,
                epoch_ms=entry.timestamp.epoch_ms,
                level=int(entry.level),
            )
        )
    return rows


def _translate_connect_error(exc: ConnectError) -> Exception:
    msg = str(exc)
    if exc.code == Code.NOT_FOUND:
        return NamespaceNotFoundError(msg)
    if exc.code == Code.INVALID_ARGUMENT:
        # Connect codes can't distinguish SchemaValidation from InvalidNamespace;
        # match on the message text. Both subclass StatsError.
        if "namespace" in msg.lower() and "name" in msg.lower():
            return InvalidNamespaceError(msg)
        return SchemaValidationError(msg)
    if exc.code == Code.FAILED_PRECONDITION:
        return SchemaConflictError(msg)
    return StatsError(msg)
