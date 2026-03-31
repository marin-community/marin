# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SQLite access layer and typed query models for controller state."""

from __future__ import annotations

import json
import logging
import queue
import sqlite3
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import MISSING, dataclass, field, fields, replace as dc_replace
from pathlib import Path
from threading import Lock, RLock
from typing import Any, TypeVar

from iris.cluster.constraints import AttributeValue
from iris.cluster.types import JobName, WorkerId, get_gpu_count, get_tpu_count
from iris.rpc import cluster_pb2
from rigging.timing import Deadline, Duration, Timestamp

logger = logging.getLogger(__name__)

T = TypeVar("T")
RowDecoder = Callable[[sqlite3.Row], Any]


class ProtoCache:
    """Thread-safe bounded cache for deserialized protobuf blobs.

    Keyed by raw proto bytes — no explicit invalidation needed for immutable
    columns (job protos). Changed bytes (worker heartbeat) naturally miss.
    """

    def __init__(self, max_size: int = 8192):
        self._cache: dict[bytes, Any] = {}
        self._lock = Lock()
        self._max_size = max_size

    def get_or_decode(self, blob: bytes, decoder: Callable[[bytes], Any]) -> Any:
        with self._lock:
            result = self._cache.get(blob)
            if result is not None:
                return result
        decoded = decoder(blob)
        with self._lock:
            if len(self._cache) >= self._max_size:
                to_evict = self._max_size // 4
                for k in list(self._cache.keys())[:to_evict]:
                    del self._cache[k]
            self._cache[blob] = decoded
        return decoded

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


# Module-level singleton — used automatically by decode_rows/_decode_row for
# fields marked with cached=True. Keyed by raw bytes so immutable blobs
# (job protos) get permanent cache hits and changed blobs naturally miss.
_proto_cache = ProtoCache()


def _identity(value: Any) -> Any:
    return value


def _nullable(decoder: RowDecoder) -> RowDecoder:
    def inner(value: Any) -> Any:
        if value is None:
            return None
        return decoder(value)

    return inner


def _decode_worker_id(value: Any) -> WorkerId:
    return WorkerId(str(value))


def _decode_timestamp_ms(value: Any) -> Timestamp:
    return Timestamp.from_ms(int(value))


def _decode_bool_int(value: Any) -> bool:
    return bool(int(value))


def _decode_int(value: Any) -> int:
    return int(value)


def _decode_str(value: Any) -> str:
    return str(value)


def _decode_json_dict(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    return json.loads(str(value))


def _decode_json_list(value: Any) -> list[str]:
    if not value:
        return []
    return json.loads(str(value))


def _proto_decoder(proto_factory: Callable[[], T]) -> Callable[[Any], T]:
    def decode(value: Any) -> T:
        proto = proto_factory()
        proto.ParseFromString(value)
        return proto

    return decode


def db_field(
    column: str,
    decoder: Callable[[Any], Any] = _identity,
    *,
    default: Any = MISSING,
    default_factory: Callable[[], Any] | None = None,
    cached: bool = False,
):
    metadata = {"db_column": column, "db_decoder": decoder, "db_cached": cached}
    kwargs: dict[str, Any] = {"metadata": metadata}
    if default_factory is not None:
        kwargs["default_factory"] = default_factory
    elif default is not MISSING:
        kwargs["default"] = default
    return field(**kwargs)


def db_row_model(cls: type[T]) -> type[T]:
    cls = dataclass(frozen=True)(cls)
    db_fields = tuple(f for f in fields(cls) if "db_column" in f.metadata)
    cls.__db_fields__ = db_fields
    # Pre-computed parallel tuples for fast row decoding (avoids per-row metadata lookups).
    cls.__db_names__ = tuple(f.name for f in db_fields)
    cls.__db_columns__ = tuple(f.metadata["db_column"] for f in db_fields)
    cls.__db_decoders__ = tuple(f.metadata.get("db_decoder", _identity) for f in db_fields)
    cls.__db_cached__ = tuple(f.metadata.get("db_cached", False) for f in db_fields)
    # Pre-computed defaults for fields that have them (keyed by column name).
    defaults: dict[str, tuple[str, Any | Callable[[], Any], bool]] = {}
    for f in db_fields:
        col = f.metadata["db_column"]
        if f.default is not MISSING:
            defaults[col] = (f.name, f.default, False)
        elif f.default_factory is not MISSING:
            defaults[col] = (f.name, f.default_factory, True)
    cls.__db_defaults__ = defaults
    cls.__db_required_columns__ = tuple(
        f.metadata["db_column"] for f in db_fields if f.metadata["db_column"] not in defaults
    )
    return cls


def _decode_row(model_cls: type[T], row: sqlite3.Row) -> T:
    """Decode a sqlite3.Row into a model instance, filling defaults for missing optional columns."""
    row_keys = set(row.keys())

    # Check required columns up front.
    for col in model_cls.__db_required_columns__:
        if col not in row_keys:
            raise KeyError(f"Missing required column {col!r} for {model_cls.__name__}")

    names = model_cls.__db_names__
    columns = model_cls.__db_columns__
    decoders = model_cls.__db_decoders__
    cached_flags = model_cls.__db_cached__
    defaults = model_cls.__db_defaults__
    values: dict[str, Any] = {}
    for name, col, decoder, is_cached in zip(names, columns, decoders, cached_flags, strict=True):
        if col in row_keys:
            raw = row[col]
            if is_cached and raw is not None:
                values[name] = _proto_cache.get_or_decode(raw, decoder)
            else:
                values[name] = decoder(raw)
        else:
            field_name, default_val, is_factory = defaults[col]
            values[field_name] = default_val() if is_factory else default_val
    return model_cls(**values)


def decode_rows(model_cls: type[T], rows: Iterable[sqlite3.Row]) -> list[T]:
    """Decode sqlite3.Row objects into model instances."""
    names = model_cls.__db_names__
    columns = model_cls.__db_columns__
    decoders = model_cls.__db_decoders__
    cached_flags = model_cls.__db_cached__
    cls = model_cls

    # Build effective decoders: wrap cached fields through the global proto cache.
    has_cached = any(cached_flags)
    if has_cached:
        effective_decoders = tuple(
            (lambda d: lambda v: _proto_cache.get_or_decode(v, d) if v is not None else d(v))(dec) if is_cached else dec
            for dec, is_cached in zip(decoders, cached_flags, strict=True)
        )
    else:
        effective_decoders = decoders

    zipped = tuple(zip(names, columns, effective_decoders, strict=True))

    result = []
    # Detect on first row whether all columns are present and pick a strategy.
    it = iter(rows)
    first = next(it, None)
    if first is None:
        return result

    first_keys = set(first.keys())
    all_present = all(col in first_keys for col in columns)

    if all_present:
        # All columns present — tight loop, no per-row key checks.
        result.append(cls(**{name: decoder(first[col]) for name, col, decoder in zipped}))
        for row in it:
            result.append(cls(**{name: decoder(row[col]) for name, col, decoder in zipped}))
    else:
        # Some columns missing — use default-filling path for every row.
        result.append(_decode_row(cls, first))
        for row in it:
            result.append(_decode_row(cls, row))
    return result


def decode_one(model_cls: type[T], rows: Iterable[sqlite3.Row]) -> T | None:
    """Decode a single row, returning None if empty."""
    for row in rows:
        return _decode_row(model_cls, row)
    return None


class Row:
    """Lightweight result row with attribute access for raw query results."""

    __slots__ = ("_data",)

    def __init__(self, data: dict[str, Any]):
        object.__setattr__(self, "_data", data)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(f"Row has no column {name!r}") from None

    def __repr__(self) -> str:
        return f"Row({self._data!r})"


class QuerySnapshot:
    """Read-only snapshot over the controller DB."""

    def __init__(self, conn: sqlite3.Connection, lock: RLock | None):
        self._conn = conn
        self._lock = lock

    def __enter__(self) -> QuerySnapshot:
        if self._lock is not None:
            self._lock.acquire()
        self._conn.execute("BEGIN")
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        try:
            self._conn.rollback()
        finally:
            if self._lock is not None:
                self._lock.release()

    def execute_sql(self, sql: str, params: tuple[object, ...] = ()) -> sqlite3.Cursor:
        """Execute raw SQL and return the cursor for result inspection."""
        return self._conn.execute(sql, params)

    def fetchall(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        """Execute SQL and return all rows."""
        return self._fetchall(sql, list(params))

    def fetchone(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        """Execute SQL and return the first row, or None."""
        return self._conn.execute(sql, params).fetchone()

    def _fetchall(self, sql: str, params: Sequence[object]) -> list[sqlite3.Row]:
        return list(self._conn.execute(sql, tuple(params)).fetchall())

    def raw(
        self,
        sql: str,
        params: tuple = (),
        decoders: dict[str, Callable] | None = None,
    ) -> list[Row]:
        """Execute raw SQL and return decoded rows with attribute access.

        Each key in `decoders` maps a column name to a decoder function.
        Columns without decoders are returned as-is from SQLite.
        """
        cursor = self._conn.execute(sql, params)
        col_names = [desc[0] for desc in cursor.description]
        active_decoders = decoders or {}
        rows = []
        for raw_row in cursor.fetchall():
            data = {
                name: active_decoders[name](raw_row[name]) if name in active_decoders else raw_row[name]
                for name in col_names
            }
            rows.append(Row(data))
        return rows


# ---------------------------------------------------------------------------
# Shared predicate functions for Task/TaskRow and Worker/WorkerRow.
# Placed above the class definitions so both full and lightweight models
# can delegate to the same logic without duplication.
# ---------------------------------------------------------------------------


def task_is_finished(
    state: int, failure_count: int, max_retries_failure: int, preemption_count: int, max_retries_preemption: int
) -> bool:
    """Whether a task has reached a terminal state with no remaining retries."""
    if state == cluster_pb2.TASK_STATE_SUCCEEDED:
        return True
    if state in (cluster_pb2.TASK_STATE_KILLED, cluster_pb2.TASK_STATE_UNSCHEDULABLE):
        return True
    if state == cluster_pb2.TASK_STATE_FAILED:
        return failure_count > max_retries_failure
    if state == cluster_pb2.TASK_STATE_WORKER_FAILED:
        return preemption_count > max_retries_preemption
    return False


def task_can_be_scheduled(
    state: int,
    current_attempt_id: int,
    failure_count: int,
    max_retries_failure: int,
    preemption_count: int,
    max_retries_preemption: int,
) -> bool:
    if state != cluster_pb2.TASK_STATE_PENDING:
        return False
    return current_attempt_id < 0 or not task_is_finished(
        state, failure_count, max_retries_failure, preemption_count, max_retries_preemption
    )


def task_is_retry_exhausted(
    state: int, failure_count: int, max_retries_failure: int, preemption_count: int, max_retries_preemption: int
) -> bool:
    if state == cluster_pb2.TASK_STATE_FAILED:
        return failure_count > max_retries_failure
    if state == cluster_pb2.TASK_STATE_WORKER_FAILED:
        return preemption_count > max_retries_preemption
    return False


def worker_available_cpu_millicores(total_cpu_millicores: int, committed_cpu_millicores: int) -> int:
    return total_cpu_millicores - committed_cpu_millicores


def worker_available_memory(total_memory_bytes: int, committed_mem_bytes: int) -> int:
    return total_memory_bytes - committed_mem_bytes


def worker_available_gpus(total_gpu_count: int, committed_gpu: int) -> int:
    return total_gpu_count - committed_gpu


def worker_available_tpus(total_tpu_count: int, committed_tpu: int) -> int:
    return total_tpu_count - committed_tpu


TERMINAL_TASK_STATES: frozenset[int] = frozenset(
    {
        cluster_pb2.TASK_STATE_SUCCEEDED,
        cluster_pb2.TASK_STATE_FAILED,
        cluster_pb2.TASK_STATE_KILLED,
        cluster_pb2.TASK_STATE_UNSCHEDULABLE,
        cluster_pb2.TASK_STATE_WORKER_FAILED,
    }
)

TERMINAL_JOB_STATES: frozenset[int] = frozenset(
    {
        cluster_pb2.JOB_STATE_SUCCEEDED,
        cluster_pb2.JOB_STATE_FAILED,
        cluster_pb2.JOB_STATE_KILLED,
        cluster_pb2.JOB_STATE_WORKER_FAILED,
        cluster_pb2.JOB_STATE_UNSCHEDULABLE,
    }
)

ACTIVE_TASK_STATES: frozenset[int] = frozenset(
    {
        cluster_pb2.TASK_STATE_ASSIGNED,
        cluster_pb2.TASK_STATE_BUILDING,
        cluster_pb2.TASK_STATE_RUNNING,
    }
)

# Tasks executing on a worker (subset of ACTIVE that excludes ASSIGNED).
EXECUTING_TASK_STATES: frozenset[int] = frozenset(
    {
        cluster_pb2.TASK_STATE_BUILDING,
        cluster_pb2.TASK_STATE_RUNNING,
    }
)

# Failure states that trigger coscheduled sibling cascades.
FAILURE_TASK_STATES: frozenset[int] = frozenset(
    {
        cluster_pb2.TASK_STATE_FAILED,
        cluster_pb2.TASK_STATE_WORKER_FAILED,
    }
)


@db_row_model
class Attempt:
    task_id: JobName = db_field("task_id", JobName.from_wire)
    attempt_id: int = db_field("attempt_id", _decode_int)
    worker_id: WorkerId | None = db_field("worker_id", _nullable(_decode_worker_id))
    state: int = db_field("state", _decode_int)
    created_at: Timestamp = db_field("created_at_ms", _decode_timestamp_ms)
    started_at: Timestamp | None = db_field("started_at_ms", _nullable(_decode_timestamp_ms))
    finished_at: Timestamp | None = db_field("finished_at_ms", _nullable(_decode_timestamp_ms))
    exit_code: int | None = db_field("exit_code", _nullable(_decode_int))
    error: str | None = db_field("error", _nullable(_decode_str))

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_TASK_STATES

    @property
    def is_worker_failure(self) -> bool:
        return self.state == cluster_pb2.TASK_STATE_WORKER_FAILED


@db_row_model
class JobDetail:
    job_id: JobName = db_field("job_id", JobName.from_wire)
    request: cluster_pb2.Controller.LaunchJobRequest = db_field(
        "request_proto",
        _proto_decoder(cluster_pb2.Controller.LaunchJobRequest),
        cached=True,
    )
    state: int = db_field("state", _decode_int)
    submitted_at: Timestamp = db_field("submitted_at_ms", _decode_timestamp_ms)
    root_submitted_at: Timestamp = db_field("root_submitted_at_ms", _decode_timestamp_ms)
    started_at: Timestamp | None = db_field("started_at_ms", _nullable(_decode_timestamp_ms))
    finished_at: Timestamp | None = db_field("finished_at_ms", _nullable(_decode_timestamp_ms))
    scheduling_deadline_epoch_ms: int | None = db_field("scheduling_deadline_epoch_ms", _nullable(_decode_int))
    error: str | None = db_field("error", _nullable(_decode_str))
    exit_code: int | None = db_field("exit_code", _nullable(_decode_int))
    num_tasks: int = db_field("num_tasks", _decode_int)
    is_reservation_holder: bool = db_field("is_reservation_holder", _decode_bool_int)
    has_reservation: bool = db_field("has_reservation", _decode_bool_int, default=False)
    name: str = db_field("name", _decode_str, default="")
    depth: int = db_field("depth", _decode_int, default=0)

    def is_finished(self) -> bool:
        return self.state in TERMINAL_JOB_STATES

    @property
    def is_coscheduled(self) -> bool:
        return self.request.HasField("coscheduling")

    @property
    def resources(self) -> cluster_pb2.ResourceSpecProto | None:
        if not self.request.HasField("resources"):
            return None
        return self.request.resources

    @property
    def constraints(self) -> list[cluster_pb2.Constraint]:
        return list(self.request.constraints)

    @property
    def scheduling_deadline(self) -> Deadline | None:
        if self.scheduling_deadline_epoch_ms is None:
            return None
        return Deadline.after(Timestamp.from_ms(self.scheduling_deadline_epoch_ms), Duration.from_ms(0))

    @property
    def coscheduling_group_by(self) -> str | None:
        if self.request.HasField("coscheduling"):
            return self.request.coscheduling.group_by
        return None


@db_row_model
class TaskDetail:
    task_id: JobName = db_field("task_id", JobName.from_wire)
    job_id: JobName = db_field("job_id", JobName.from_wire)
    state: int = db_field("state", _decode_int)
    error: str | None = db_field("error", _nullable(_decode_str))
    exit_code: int | None = db_field("exit_code", _nullable(_decode_int))
    submitted_at: Timestamp = db_field("submitted_at_ms", _decode_timestamp_ms)
    started_at: Timestamp | None = db_field("started_at_ms", _nullable(_decode_timestamp_ms))
    finished_at: Timestamp | None = db_field("finished_at_ms", _nullable(_decode_timestamp_ms))
    max_retries_failure: int = db_field("max_retries_failure", _decode_int)
    max_retries_preemption: int = db_field("max_retries_preemption", _decode_int)
    failure_count: int = db_field("failure_count", _decode_int)
    preemption_count: int = db_field("preemption_count", _decode_int)
    current_attempt_id: int = db_field("current_attempt_id", _decode_int)
    resource_usage: cluster_pb2.ResourceUsage | None = db_field(
        "resource_usage_proto",
        _nullable(_proto_decoder(cluster_pb2.ResourceUsage)),
    )
    current_worker_id: WorkerId | None = db_field("current_worker_id", _nullable(_decode_worker_id), default=None)
    current_worker_address: str | None = db_field("current_worker_address", _nullable(_decode_str), default=None)
    container_id: str | None = db_field("container_id", _nullable(_decode_str), default=None)
    attempts: tuple[Attempt, ...] = field(default_factory=tuple)

    def is_finished(self) -> bool:
        return task_is_finished(
            self.state, self.failure_count, self.max_retries_failure, self.preemption_count, self.max_retries_preemption
        )

    @property
    def current_attempt(self) -> Attempt | None:
        if not self.attempts:
            return None
        return self.attempts[-1]

    @property
    def worker_id(self) -> WorkerId | None:
        current = self.current_attempt
        if current is None:
            return self.current_worker_id
        return current.worker_id

    @property
    def active_worker_id(self) -> WorkerId | None:
        if self.state == cluster_pb2.TASK_STATE_PENDING:
            return None
        return self.worker_id

    @property
    def task_index(self) -> int:
        return int(self.task_id.to_wire().rsplit("/", 1)[-1])

    def can_be_scheduled(self) -> bool:
        return task_can_be_scheduled(
            self.state,
            self.current_attempt_id,
            self.failure_count,
            self.max_retries_failure,
            self.preemption_count,
            self.max_retries_preemption,
        )

    def is_live(self) -> bool:
        return self.state not in TERMINAL_TASK_STATES

    def is_dead(self) -> bool:
        return self.state in TERMINAL_TASK_STATES

    def is_retry_exhausted(self) -> bool:
        return task_is_retry_exhausted(
            self.state, self.failure_count, self.max_retries_failure, self.preemption_count, self.max_retries_preemption
        )


@db_row_model
class WorkerDetail:
    worker_id: WorkerId = db_field("worker_id", _decode_worker_id)
    address: str = db_field("address", _decode_str)
    metadata: cluster_pb2.WorkerMetadata = db_field("metadata_proto", _proto_decoder(cluster_pb2.WorkerMetadata))
    healthy: bool = db_field("healthy", _decode_bool_int)
    consecutive_failures: int = db_field("consecutive_failures", _decode_int)
    last_heartbeat: Timestamp = db_field("last_heartbeat_ms", _decode_timestamp_ms)
    committed_cpu_millicores: int = db_field("committed_cpu_millicores", _decode_int)
    committed_mem: int = db_field("committed_mem_bytes", _decode_int)
    committed_gpu: int = db_field("committed_gpu", _decode_int)
    committed_tpu: int = db_field("committed_tpu", _decode_int)
    active: bool = db_field("active", _decode_bool_int, default=True)
    attributes: dict[str, AttributeValue] = field(default_factory=dict)

    @property
    def available_cpu_millicores(self) -> int:
        return worker_available_cpu_millicores(self.metadata.cpu_count * 1000, self.committed_cpu_millicores)

    @property
    def available_memory(self) -> int:
        return worker_available_memory(self.metadata.memory_bytes, self.committed_mem)

    @property
    def available_gpus(self) -> int:
        return worker_available_gpus(get_gpu_count(self.metadata.device), self.committed_gpu)

    @property
    def available_tpus(self) -> int:
        return worker_available_tpus(get_tpu_count(self.metadata.device), self.committed_tpu)

    @property
    def device_variant(self) -> str:
        if self.metadata.device.HasField("gpu"):
            return str(self.metadata.device.gpu.variant)
        if self.metadata.device.HasField("tpu"):
            return str(self.metadata.device.tpu.variant)
        return "cpu"


@db_row_model
class Endpoint:
    endpoint_id: str = db_field("endpoint_id", _decode_str)
    name: str = db_field("name", _decode_str)
    address: str = db_field("address", _decode_str)
    job_id: JobName = db_field("job_id", JobName.from_wire)
    metadata: dict[str, str] = db_field("metadata_json", _decode_json_dict)
    registered_at: Timestamp = db_field("registered_at_ms", _decode_timestamp_ms)


# ---------------------------------------------------------------------------
# Lightweight row models -- scalar-only projections for hot-path queries.
# These avoid decoding proto blobs (request_proto, metadata_proto, etc.).
# ---------------------------------------------------------------------------


def _constraint_list_decoder(blob: bytes | None) -> list[cluster_pb2.Constraint]:
    if blob is None:
        return []
    cl = cluster_pb2.ConstraintList()
    cl.ParseFromString(blob)
    return list(cl.constraints)


@db_row_model
class JobRow:
    """Scalar-only job row for queries that don't need request_proto."""

    job_id: JobName = db_field("job_id", JobName.from_wire)
    state: int = db_field("state", _decode_int)
    submitted_at: Timestamp = db_field("submitted_at_ms", _decode_timestamp_ms)
    root_submitted_at: Timestamp = db_field("root_submitted_at_ms", _decode_timestamp_ms)
    started_at: Timestamp | None = db_field("started_at_ms", _nullable(_decode_timestamp_ms))
    finished_at: Timestamp | None = db_field("finished_at_ms", _nullable(_decode_timestamp_ms))
    scheduling_deadline_epoch_ms: int | None = db_field("scheduling_deadline_epoch_ms", _nullable(_decode_int))
    error: str | None = db_field("error", _nullable(_decode_str))
    exit_code: int | None = db_field("exit_code", _nullable(_decode_int))
    num_tasks: int = db_field("num_tasks", _decode_int)
    is_reservation_holder: bool = db_field("is_reservation_holder", _decode_bool_int)
    has_reservation: bool = db_field("has_reservation", _decode_bool_int, default=False)
    name: str = db_field("name", _decode_str, default="")
    depth: int = db_field("depth", _decode_int, default=0)
    resources: cluster_pb2.ResourceSpecProto | None = db_field(
        "resources_proto", _nullable(_proto_decoder(cluster_pb2.ResourceSpecProto)), default=None, cached=True
    )
    constraints: list[cluster_pb2.Constraint] = db_field(
        "constraints_proto", _constraint_list_decoder, default_factory=list, cached=True
    )
    has_coscheduling: bool = db_field("has_coscheduling", _decode_bool_int, default=False)
    coscheduling_group_by: str = db_field("coscheduling_group_by", _decode_str, default="")
    scheduling_timeout_ms: int | None = db_field("scheduling_timeout_ms", _nullable(_decode_int), default=None)
    max_task_failures: int = db_field("max_task_failures", _decode_int, default=0)

    def is_finished(self) -> bool:
        return self.state in TERMINAL_JOB_STATES

    @property
    def is_coscheduled(self) -> bool:
        return self.has_coscheduling

    @property
    def scheduling_deadline(self) -> Deadline | None:
        if self.scheduling_deadline_epoch_ms is None:
            return None
        return Deadline.after(Timestamp.from_ms(self.scheduling_deadline_epoch_ms), Duration.from_ms(0))


@db_row_model
class WorkerRow:
    """Scalar-only worker row for queries that don't need metadata_proto."""

    worker_id: WorkerId = db_field("worker_id", _decode_worker_id)
    address: str = db_field("address", _decode_str)
    healthy: bool = db_field("healthy", _decode_bool_int)
    consecutive_failures: int = db_field("consecutive_failures", _decode_int)
    last_heartbeat: Timestamp = db_field("last_heartbeat_ms", _decode_timestamp_ms)
    committed_cpu_millicores: int = db_field("committed_cpu_millicores", _decode_int)
    committed_mem: int = db_field("committed_mem_bytes", _decode_int)
    committed_gpu: int = db_field("committed_gpu", _decode_int)
    committed_tpu: int = db_field("committed_tpu", _decode_int)
    active: bool = db_field("active", _decode_bool_int, default=True)
    total_cpu_millicores: int = db_field("total_cpu_millicores", _decode_int, default=0)
    total_memory_bytes: int = db_field("total_memory_bytes", _decode_int, default=0)
    total_gpu_count: int = db_field("total_gpu_count", _decode_int, default=0)
    total_tpu_count: int = db_field("total_tpu_count", _decode_int, default=0)
    device_type: str = db_field("device_type", _decode_str, default="")
    device_variant: str = db_field("device_variant", _decode_str, default="")
    attributes: dict[str, AttributeValue] = field(default_factory=dict)

    @property
    def available_cpu_millicores(self) -> int:
        return worker_available_cpu_millicores(self.total_cpu_millicores, self.committed_cpu_millicores)

    @property
    def available_memory(self) -> int:
        return worker_available_memory(self.total_memory_bytes, self.committed_mem)

    @property
    def available_gpus(self) -> int:
        return worker_available_gpus(self.total_gpu_count, self.committed_gpu)

    @property
    def available_tpus(self) -> int:
        return worker_available_tpus(self.total_tpu_count, self.committed_tpu)


@db_row_model
class TaskRow:
    """Scalar-only task row for scheduling -- no resource_usage_proto, no attempts."""

    task_id: JobName = db_field("task_id", JobName.from_wire)
    job_id: JobName = db_field("job_id", JobName.from_wire)
    state: int = db_field("state", _decode_int)
    current_attempt_id: int = db_field("current_attempt_id", _decode_int)
    failure_count: int = db_field("failure_count", _decode_int)
    preemption_count: int = db_field("preemption_count", _decode_int)
    max_retries_failure: int = db_field("max_retries_failure", _decode_int)
    max_retries_preemption: int = db_field("max_retries_preemption", _decode_int)
    submitted_at: Timestamp = db_field("submitted_at_ms", _decode_timestamp_ms)

    def can_be_scheduled(self) -> bool:
        return task_can_be_scheduled(
            self.state,
            self.current_attempt_id,
            self.failure_count,
            self.max_retries_failure,
            self.preemption_count,
            self.max_retries_preemption,
        )

    def is_finished(self) -> bool:
        return task_is_finished(
            self.state, self.failure_count, self.max_retries_failure, self.preemption_count, self.max_retries_preemption
        )


JOB_ROW_COLUMNS = (
    "job_id, state, submitted_at_ms, root_submitted_at_ms, started_at_ms, "
    "finished_at_ms, scheduling_deadline_epoch_ms, error, exit_code, "
    "num_tasks, is_reservation_holder, has_reservation, name, depth, "
    "resources_proto, constraints_proto, has_coscheduling, "
    "coscheduling_group_by, scheduling_timeout_ms, max_task_failures"
)

# Same as JOB_ROW_COLUMNS but without constraints_proto — used for listing
# paths where constraints are never accessed, avoiding the blob fetch entirely.
JOB_LISTING_COLUMNS = (
    "job_id, state, submitted_at_ms, root_submitted_at_ms, started_at_ms, "
    "finished_at_ms, scheduling_deadline_epoch_ms, error, exit_code, "
    "num_tasks, is_reservation_holder, has_reservation, name, depth, "
    "resources_proto, has_coscheduling, "
    "coscheduling_group_by, scheduling_timeout_ms, max_task_failures"
)

WORKER_ROW_COLUMNS = (
    "worker_id, address, healthy, active, consecutive_failures, "
    "last_heartbeat_ms, committed_cpu_millicores, committed_mem_bytes, "
    "committed_gpu, committed_tpu, total_cpu_millicores, total_memory_bytes, "
    "total_gpu_count, total_tpu_count, device_type, device_variant"
)

TASK_ROW_COLUMNS = (
    "task_id, job_id, state, current_attempt_id, failure_count, "
    "preemption_count, max_retries_failure, max_retries_preemption, submitted_at_ms"
)


@db_row_model
class TransactionAction:
    timestamp: Timestamp = db_field("created_at_ms", _decode_timestamp_ms)
    action: str = db_field("action", _decode_str)
    entity_id: str = db_field("entity_id", _decode_str)
    details: dict[str, object] = db_field("details_json", _decode_json_dict)


@dataclass(frozen=True)
class UserStats:
    user: str
    task_state_counts: dict[int, int] = field(default_factory=dict)
    job_state_counts: dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskJobSummary:
    job_id: JobName
    task_count: int = 0
    completed_count: int = 0
    failure_count: int = 0
    preemption_count: int = 0
    task_state_counts: dict[int, int] = field(default_factory=dict)


@db_row_model
class ApiKey:
    key_id: str = db_field("key_id", _decode_str)
    key_hash: str = db_field("key_hash", _decode_str)
    key_prefix: str = db_field("key_prefix", _decode_str)
    user_id: str = db_field("user_id", _decode_str)
    name: str = db_field("name", _decode_str)
    created_at: Timestamp = db_field("created_at_ms", _decode_timestamp_ms)
    last_used_at: Timestamp | None = db_field("last_used_at_ms", _nullable(_decode_timestamp_ms), default=None)
    expires_at: Timestamp | None = db_field("expires_at_ms", _nullable(_decode_timestamp_ms), default=None)
    revoked_at: Timestamp | None = db_field("revoked_at_ms", _nullable(_decode_timestamp_ms), default=None)


@dataclass(frozen=True)
class EndpointQuery:
    endpoint_ids: tuple[str, ...] = ()
    name_prefix: str | None = None
    exact_name: str | None = None
    job_ids: tuple[JobName, ...] = ()
    job_id: JobName | None = None
    task_ids: tuple[JobName, ...] = ()
    limit: int | None = None


def _decode_attribute_rows(rows: Sequence[Any]) -> dict[WorkerId, dict[str, AttributeValue]]:
    attrs_by_worker: dict[WorkerId, dict[str, AttributeValue]] = {}
    for row in rows:
        worker_attrs = attrs_by_worker.setdefault(row.worker_id, {})
        if row.value_type == "int":
            worker_attrs[row.key] = AttributeValue(int(row.int_value))
        elif row.value_type == "float":
            worker_attrs[row.key] = AttributeValue(float(row.float_value))
        else:
            worker_attrs[row.key] = AttributeValue(str(row.str_value or ""))
    return attrs_by_worker


def _tasks_with_attempts(tasks: Sequence[TaskDetail], attempts: Sequence[Attempt]) -> list[TaskDetail]:
    attempts_by_task: dict[JobName, list[Attempt]] = {}
    for attempt in attempts:
        attempts_by_task.setdefault(attempt.task_id, []).append(attempt)
    return [TaskDetail(**{**task.__dict__, "attempts": tuple(attempts_by_task.get(task.task_id, ()))}) for task in tasks]


def endpoint_query_sql(query: EndpointQuery) -> tuple[str, list[object]]:
    """Build SQL query for endpoint lookups."""
    from_clause = "SELECT e.* FROM endpoints e"
    conditions: list[str] = []
    params: list[object] = []

    if query.task_ids:
        from_clause += " JOIN endpoints et ON e.endpoint_id = et.endpoint_id"
        placeholders = ",".join("?" for _ in query.task_ids)
        conditions.append(f"et.task_id IN ({placeholders})")
        params.extend(tid.to_wire() for tid in query.task_ids)

    if query.endpoint_ids:
        placeholders = ",".join("?" for _ in query.endpoint_ids)
        conditions.append(f"e.endpoint_id IN ({placeholders})")
        params.extend(query.endpoint_ids)

    if query.name_prefix:
        conditions.append("e.name LIKE ?")
        params.append(f"{query.name_prefix}%")

    if query.exact_name:
        conditions.append("e.name = ?")
        params.append(query.exact_name)

    job_ids = list(query.job_ids)
    if query.job_id is not None:
        job_ids.append(query.job_id)
    if job_ids:
        placeholders = ",".join("?" for _ in job_ids)
        conditions.append(f"e.job_id IN ({placeholders})")
        params.extend(jid.to_wire() for jid in job_ids)

    sql = from_clause
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    if query.limit is not None:
        sql += " LIMIT ?"
        params.append(query.limit)
    return sql, params


class TransactionCursor:
    """Wraps a raw sqlite3.Cursor for use within controller transactions."""

    def __init__(self, cursor: sqlite3.Cursor):
        self._cursor = cursor

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Raw SQL escape hatch."""
        return self._cursor.execute(sql, params)

    def executemany(self, sql: str, params: Iterable[tuple]) -> sqlite3.Cursor:
        """Raw SQL batch escape hatch."""
        return self._cursor.executemany(sql, params)

    def executescript(self, sql: str) -> sqlite3.Cursor:
        """Raw SQL script escape hatch."""
        return self._cursor.executescript(sql)

    @property
    def lastrowid(self) -> int | None:
        return self._cursor.lastrowid

    @property
    def rowcount(self) -> int:
        return self._cursor.rowcount


class ControllerDB:
    """Thread-safe SQLite wrapper with typed query and migration helpers."""

    _READ_POOL_SIZE = 8
    DB_FILENAME = "controller.sqlite3"
    AUTH_DB_FILENAME = "auth.sqlite3"

    def __init__(self, db_dir: Path):
        import time

        self._db_dir = db_dir
        self._db_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._db_dir / self.DB_FILENAME
        self._auth_db_path = self._db_dir / self.AUTH_DB_FILENAME
        self._lock = RLock()

        t0 = time.monotonic()
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._configure(self._conn)
        self._conn.execute("ATTACH DATABASE ? AS auth", (str(self._auth_db_path),))
        logger.info("DB opened in %.2fs (path=%s)", time.monotonic() - t0, self._db_path)

        t0 = time.monotonic()
        self.apply_migrations()
        logger.info("Migrations applied in %.2fs", time.monotonic() - t0)

        # Populate sqlite_stat1 so the query planner picks good join orders.
        # Without this, queries like running_tasks_by_worker scan thousands of
        # rows instead of using the narrower index path.
        t0 = time.monotonic()
        self._conn.execute("ANALYZE")
        logger.info("ANALYZE completed in %.2fs", time.monotonic() - t0)

        t0 = time.monotonic()
        self._read_pool: queue.Queue[sqlite3.Connection] = queue.Queue()
        self._init_read_pool()
        logger.info("Read pool initialized in %.2fs", time.monotonic() - t0)
        # Lazily populated cache of worker attributes, keyed by worker_id.
        # Eliminates the per-cycle attribute SQL query from the scheduling hot path.
        self._attr_cache: dict[WorkerId, dict[str, AttributeValue]] | None = None
        self._attr_cache_lock = Lock()

    def _populate_attr_cache(self) -> dict[WorkerId, dict[str, AttributeValue]]:
        """Load all worker attributes from the DB into the cache.

        Called once on cold start (first access). The caller must NOT hold
        _attr_cache_lock when calling this, because the DB read can be slow.
        """
        with self.read_snapshot() as q:
            rows = q.raw(
                "SELECT worker_id, key, value_type, str_value, int_value, float_value FROM worker_attributes",
            )
        return _decode_attribute_rows(rows)

    def get_worker_attributes(self) -> dict[WorkerId, dict[str, AttributeValue]]:
        """Return cached worker attributes, populating from DB on first call."""
        cache = self._attr_cache
        if cache is not None:
            return cache
        fresh = self._populate_attr_cache()
        with self._attr_cache_lock:
            if self._attr_cache is None:
                self._attr_cache = fresh
            return self._attr_cache

    def set_worker_attributes(self, worker_id: WorkerId, attrs: dict[str, AttributeValue]) -> None:
        """Update the cached attributes for a single worker after registration."""
        with self._attr_cache_lock:
            if self._attr_cache is None:
                return
            self._attr_cache[worker_id] = attrs

    def remove_worker_from_attr_cache(self, worker_id: WorkerId) -> None:
        """Remove a single worker from the attribute cache."""
        with self._attr_cache_lock:
            if self._attr_cache is None:
                return
            self._attr_cache.pop(worker_id, None)

    def _init_read_pool(self) -> None:
        """Create (or recreate) the read-only connection pool."""
        while True:
            try:
                self._read_pool.get_nowait().close()
            except queue.Empty:
                break
        for _ in range(self._READ_POOL_SIZE):
            conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._configure(conn)
            conn.execute("PRAGMA query_only = ON")
            self._read_pool.put(conn)

    @property
    def db_dir(self) -> Path:
        return self._db_dir

    @property
    def db_path(self) -> Path:
        return self._db_path

    @property
    def auth_db_path(self) -> Path:
        return self._auth_db_path

    @staticmethod
    def _configure(conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA foreign_keys = ON")

    def optimize(self) -> None:
        """Run PRAGMA optimize to refresh statistics for tables with stale data.

        Lightweight operation that SQLite recommends running periodically or on
        connection close. Only re-analyzes tables whose stats have drifted.
        """
        with self._lock:
            self._conn.execute("PRAGMA optimize")

    def close(self) -> None:
        with self._lock:
            self._conn.close()
        for _ in range(self._READ_POOL_SIZE):
            try:
                self._read_pool.get(timeout=1).close()
            except queue.Empty:
                break

    @contextmanager
    def transaction(self):
        """Open an IMMEDIATE transaction and yield a TransactionCursor."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("BEGIN IMMEDIATE")
            try:
                yield TransactionCursor(cur)
            except Exception:
                self._conn.rollback()
                raise
            else:
                self._conn.commit()

    def fetchall(self, query: str, params: tuple | list = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(query, params).fetchall())

    def fetchone(self, query: str, params: tuple | list = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(query, params).fetchone()

    def execute(self, query: str, params: tuple | list = ()) -> None:
        with self.transaction() as cur:
            cur.execute(query, params)

    def snapshot(self) -> QuerySnapshot:
        return QuerySnapshot(self._conn, self._lock)

    @contextmanager
    def read_snapshot(self) -> Iterator[QuerySnapshot]:
        """Read-only snapshot that does NOT acquire the write lock.

        Uses a pooled read-only connection with WAL isolation. Safe for
        concurrent use from dashboard/RPC threads while the scheduling
        loop holds the write lock.
        """
        conn = self._read_pool.get()
        try:
            conn.execute("BEGIN")
            yield QuerySnapshot(conn, lock=None)
        finally:
            try:
                conn.rollback()
            except sqlite3.OperationalError:
                logging.getLogger(__name__).warning("read_snapshot rollback failed", exc_info=True)
            self._read_pool.put(conn)

    @staticmethod
    def decode_task(row: sqlite3.Row) -> TaskDetail:
        return _decode_row(TaskDetail, row)

    def apply_migrations(self) -> None:
        """Apply pending migrations from the migrations/ directory.

        Supports Python migration files that define a ``migrate(conn)``
        function. Migration names are matched by stem so that a migration
        previously applied as .sql is not re-run when converted to .py.

        Migrations run outside a transaction because executescript() implicitly
        commits. This is fine: migrations only run at startup before any
        concurrent access. Each migration is applied then recorded; if the
        process crashes mid-migration the partially-applied file won't be in
        schema_migrations and the next startup will re-run it (migrations must
        be idempotent via IF NOT EXISTS / IF EXISTS guards).
        """
        import importlib.util

        migrations_dir = Path(__file__).with_name("migrations")
        migrations_dir.mkdir(parents=True, exist_ok=True)

        with self.transaction() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    name TEXT PRIMARY KEY,
                    applied_at_ms INTEGER NOT NULL
                )
                """
            )
            applied = {row[0] for row in cur.execute("SELECT name FROM schema_migrations ORDER BY name").fetchall()}

        # Match by stem so a migration previously recorded as .sql is not
        # re-run after conversion to .py.
        applied_stems = {Path(name).stem for name in applied}

        import time

        pending = []
        for path in sorted(migrations_dir.glob("*.py")):
            if path.name.startswith("__"):
                continue
            if path.stem in applied_stems:
                continue
            pending.append(path)

        if pending:
            logger.info("Applying %d pending migration(s): %s", len(pending), [p.name for p in pending])

        for path in pending:
            t0 = time.monotonic()
            spec = importlib.util.spec_from_file_location(path.stem, path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.migrate(self._conn)
            # Commit any implicit transaction left open by migrate() (e.g.
            # row-by-row UPDATEs in 0008) so the next BEGIN IMMEDIATE succeeds.
            self._conn.commit()
            logger.info("Migration %s applied in %.2fs", path.name, time.monotonic() - t0)

            with self.transaction() as cur:
                cur.execute(
                    "INSERT INTO schema_migrations(name, applied_at_ms) VALUES (?, ?)",
                    (path.name, Timestamp.now().epoch_ms()),
                )

    @property
    def api_keys_table(self) -> str:
        return "auth.api_keys"

    @property
    def secrets_table(self) -> str:
        return "auth.controller_secrets"

    def ensure_user(self, user_id: str, now: Timestamp, role: str = "user") -> None:
        """Create user if not exists. Does not update role for existing users."""
        self.execute(
            "INSERT OR IGNORE INTO users (user_id, created_at_ms, role) VALUES (?, ?, ?)",
            (user_id, now.epoch_ms(), role),
        )

    def set_user_role(self, user_id: str, role: str) -> None:
        """Update the role for an existing user."""
        self.execute("UPDATE users SET role = ? WHERE user_id = ?", (role, user_id))

    def get_user_role(self, user_id: str) -> str:
        """Get a user's role. Returns 'user' if not found."""
        with self.snapshot() as q:
            rows = q.raw(
                "SELECT role FROM users WHERE user_id = ?",
                (user_id,),
                decoders={"role": str},
            )
            return rows[0].role if rows else "user"

    def next_sequence(self, key: str, *, cur: TransactionCursor) -> int:
        row = cur.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        if row is None:
            cur.execute("INSERT INTO meta(key, value) VALUES (?, ?)", (key, 1))
            return 1
        value = int(row[0]) + 1
        cur.execute("UPDATE meta SET value = ? WHERE key = ?", (value, key))
        return value

    def backup_to(self, destination: Path) -> None:
        """Create a hot backup to ``destination`` using SQLite backup API.

        The source DB uses WAL journal mode, but the backup API copies
        the WAL flag into the destination header.  We switch the
        destination to DELETE mode so the result is a single
        self-contained file (no -wal/-shm sidecars) that survives
        compression and remote upload without corruption.

        The backup is also VACUUMed with auto_vacuum=INCREMENTAL so that
        controllers restoring from this checkpoint start in incremental
        mode without needing a full VACUUM at boot.
        """
        import time

        destination.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            dest = sqlite3.connect(str(destination))
            try:
                self._conn.backup(dest)
                dest.execute("PRAGMA journal_mode = DELETE")
                dest.commit()
            finally:
                dest.close()

        # VACUUM INTO a compacted copy with incremental auto_vacuum enabled.
        # Runs outside the lock since it operates on the already-written backup.
        t0 = time.monotonic()
        vacuumed = destination.with_suffix(".vacuumed.sqlite3")
        conn = sqlite3.connect(str(destination))
        try:
            conn.execute("PRAGMA auto_vacuum = INCREMENTAL")
            conn.execute(f"VACUUM INTO '{vacuumed}'")
        finally:
            conn.close()
        vacuumed.rename(destination)
        logger.info("Checkpoint vacuumed in %.1fs", time.monotonic() - t0)

    def replace_from(self, source_dir: str | Path) -> None:
        """Replace current DB files from ``source_dir`` and reopen connection.

        ``source_dir`` is a directory (local or remote) containing
        ``controller.sqlite3`` and optionally ``auth.sqlite3``. Files are
        downloaded via fsspec so remote paths (e.g. ``gs://...``) work.
        Only called at startup before concurrent access begins.
        """
        import fsspec.core

        source_dir_str = str(source_dir).rstrip("/")

        with self._lock:
            # Download main DB
            main_source = f"{source_dir_str}/{self.DB_FILENAME}"
            tmp_path = self._db_path.with_suffix(".tmp")
            with fsspec.core.open(main_source, "rb") as src, open(tmp_path, "wb") as dst:
                dst.write(src.read())
            self._conn.close()
            tmp_path.rename(self._db_path)

            # Download auth DB if present in source
            auth_source = f"{source_dir_str}/{self.AUTH_DB_FILENAME}"
            fs, fs_path = fsspec.core.url_to_fs(auth_source)
            if fs.exists(fs_path):
                auth_tmp = self._auth_db_path.with_suffix(".tmp")
                with fsspec.core.open(auth_source, "rb") as src, open(auth_tmp, "wb") as dst:
                    dst.write(src.read())
                auth_tmp.rename(self._auth_db_path)

            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._configure(self._conn)
            self._conn.execute("ATTACH DATABASE ? AS auth", (str(self._auth_db_path),))
            self._init_read_pool()
        self.apply_migrations()

    # SQL-canonical read access is exposed through ``snapshot()`` and typed table
    # metadata at module scope. Legacy list/get/count helper methods were removed
    # to keep relation assembly explicit in controller/service/state query flows.

    def delete_endpoint(self, endpoint_id: str) -> Endpoint | None:
        with self.transaction() as cur:
            row = cur.execute(
                "SELECT endpoint_id, name, address, job_id, metadata_json, registered_at_ms "
                "FROM endpoints WHERE endpoint_id = ?",
                (endpoint_id,),
            ).fetchone()
            if row is None:
                return None
            cur.execute("DELETE FROM endpoints WHERE endpoint_id = ?", (endpoint_id,))
            return _decode_row(Endpoint, row)

    def delete_endpoints(self, endpoint_ids: Sequence[str]) -> None:
        if not endpoint_ids:
            return
        placeholders = ",".join("?" for _ in endpoint_ids)
        self.execute(f"DELETE FROM endpoints WHERE endpoint_id IN ({placeholders})", tuple(endpoint_ids))


# ---------------------------------------------------------------------------
# Shared read-only query helpers
#
# Pure DB reads that are used by both controller.py and service.py.
# Each takes a ControllerDB and returns domain objects.
# ---------------------------------------------------------------------------


def running_tasks_by_worker(db: ControllerDB, worker_ids: set[WorkerId]) -> dict[WorkerId, set[JobName]]:
    """Return the set of currently-running task IDs for each worker.

    Uses the denormalized current_worker_id column instead of joining task_attempts.
    """
    if not worker_ids:
        return {}
    placeholders = ",".join("?" for _ in worker_ids)
    with db.read_snapshot() as q:
        rows = q.raw(
            f"SELECT t.current_worker_id AS worker_id, t.task_id FROM tasks t "
            f"WHERE t.current_worker_id IN ({placeholders}) AND t.state IN (?, ?, ?)",
            (*[str(wid) for wid in worker_ids], *ACTIVE_TASK_STATES),
            decoders={"worker_id": _decode_worker_id, "task_id": JobName.from_wire},
        )
    running: dict[WorkerId, set[JobName]] = {wid: set() for wid in worker_ids}
    for row in rows:
        running[row.worker_id].add(row.task_id)
    return running


def tasks_for_job_with_attempts(db: ControllerDB, job_id: JobName) -> list[TaskDetail]:
    """Fetch all tasks for a job with their attempt history."""
    with db.read_snapshot() as q:
        tasks = decode_rows(
            TaskDetail,
            q.fetchall(
                "SELECT * FROM tasks WHERE job_id = ? ORDER BY task_index, task_id",
                (job_id.to_wire(),),
            ),
        )
        if not tasks:
            return []
        placeholders = ",".join("?" for _ in tasks)
        attempts = decode_rows(
            Attempt,
            q.fetchall(
                f"SELECT * FROM task_attempts WHERE task_id IN ({placeholders}) ORDER BY task_id, attempt_id",
                tuple(t.task_id.to_wire() for t in tasks),
            ),
        )
    return _tasks_with_attempts(tasks, attempts)


def healthy_active_workers_with_attributes(db: ControllerDB) -> list[WorkerRow]:
    """Fetch all healthy, active workers with their attributes populated.

    Returns WorkerRow (scalar-only) so the scheduling loop never decodes metadata_proto.
    Uses the in-memory attribute cache to avoid a per-cycle SQL join.
    """
    with db.read_snapshot() as q:
        workers = decode_rows(
            WorkerRow,
            q.fetchall(f"SELECT {WORKER_ROW_COLUMNS} FROM workers WHERE healthy = 1 AND active = 1"),
        )
        if not workers:
            return []
    attrs_by_worker = db.get_worker_attributes()
    return [dc_replace(w, attributes=attrs_by_worker.get(w.worker_id, {})) for w in workers]


def insert_task_profile(
    db: ControllerDB, task_id: str, profile_data: bytes, captured_at: Timestamp, profile_kind: str = "cpu"
) -> None:
    """Insert a captured profile snapshot for a task.

    The DB trigger caps profiles at 10 per (task_id, profile_kind), evicting the oldest automatically.
    """
    db.execute(
        "INSERT INTO task_profiles (task_id, profile_data, captured_at_ms, profile_kind) VALUES (?, ?, ?, ?)",
        (task_id, profile_data, captured_at.epoch_ms(), profile_kind),
    )


def get_task_profiles(
    db: ControllerDB, task_id: str, profile_kind: str | None = None
) -> list[tuple[bytes, Timestamp, str]]:
    """Return stored profile snapshots for a task, newest first.

    Args:
        db: Controller database.
        task_id: Task wire string.
        profile_kind: If set, filter to this kind (e.g. "cpu", "memory"). Returns all kinds when None.
    """
    if profile_kind is not None:
        query = (
            "SELECT profile_data, captured_at_ms, profile_kind FROM task_profiles"
            " WHERE task_id = ? AND profile_kind = ? ORDER BY id DESC"
        )
        params: tuple[str, ...] = (task_id, profile_kind)
    else:
        query = (
            "SELECT profile_data, captured_at_ms, profile_kind FROM task_profiles" " WHERE task_id = ? ORDER BY id DESC"
        )
        params = (task_id,)
    with db.snapshot() as q:
        rows = q.raw(query, params, decoders={"captured_at_ms": _decode_timestamp_ms})
    return [(row.profile_data, row.captured_at_ms, row.profile_kind) for row in rows]
