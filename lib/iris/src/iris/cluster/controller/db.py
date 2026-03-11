# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SQLite access layer and typed query models for controller state."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterable, Sequence
from contextlib import contextmanager
from dataclasses import MISSING, dataclass, field, fields, replace as dc_replace
from pathlib import Path
from threading import RLock
from typing import Any, Generic, Literal, TypeVar, overload

from iris.cluster.constraints import AttributeValue
from iris.cluster.types import JobName, WorkerId, get_gpu_count, get_tpu_count
from iris.rpc import cluster_pb2
from iris.time_utils import Deadline, Duration, Timestamp

T = TypeVar("T")
V = TypeVar("V")
RowDecoder = Callable[[sqlite3.Row], Any]


def _identity(value: Any) -> Any:
    return value


def _nullable(decoder: RowDecoder) -> RowDecoder:
    def inner(value: Any) -> Any:
        if value is None:
            return None
        return decoder(value)

    return inner


def _decode_job_name(value: Any) -> JobName:
    return JobName.from_wire(str(value))


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
):
    metadata = {"db_column": column, "db_decoder": decoder}
    kwargs: dict[str, Any] = {"metadata": metadata}
    if default_factory is not None:
        kwargs["default_factory"] = default_factory
    elif default is not MISSING:
        kwargs["default"] = default
    return field(**kwargs)


def db_row_model(cls: type[T]) -> type[T]:
    cls = dataclass(frozen=True)(cls)
    cls.__db_fields__ = tuple(f for f in fields(cls) if "db_column" in f.metadata)
    return cls


def _decode_row(model_cls: type[T], row: sqlite3.Row) -> T:
    values: dict[str, Any] = {}
    row_keys = set(row.keys())
    for item in model_cls.__db_fields__:
        column = item.metadata["db_column"]
        if column not in row_keys:
            if item.default is not MISSING:
                values[item.name] = item.default
                continue
            if item.default_factory is not MISSING:
                values[item.name] = item.default_factory()
                continue
            raise KeyError(f"Missing required column {column!r} for {model_cls.__name__}.{item.name}")
        decoder = item.metadata.get("db_decoder", _identity)
        values[item.name] = decoder(row[column])
    return model_cls(**values)


class Predicate:
    def compile(self) -> tuple[str, list[object]]:
        raise NotImplementedError

    def __and__(self, other: Predicate) -> Predicate:
        return _CompositePredicate("AND", (self, other))

    def __or__(self, other: Predicate) -> Predicate:
        return _CompositePredicate("OR", (self, other))

    def __invert__(self) -> Predicate:
        return _NotPredicate(self)


@dataclass(frozen=True)
class _SqlPredicate(Predicate):
    sql: str
    params: tuple[object, ...] = ()

    def compile(self) -> tuple[str, list[object]]:
        return self.sql, list(self.params)


@dataclass(frozen=True)
class _CompositePredicate(Predicate):
    op: Literal["AND", "OR"]
    parts: tuple[Predicate, ...]

    def compile(self) -> tuple[str, list[object]]:
        sql_parts: list[str] = []
        params: list[object] = []
        for part in self.parts:
            compiled_sql, compiled_params = part.compile()
            sql_parts.append(f"({compiled_sql})")
            params.extend(compiled_params)
        return f" {self.op} ".join(sql_parts), params


@dataclass(frozen=True)
class _NotPredicate(Predicate):
    inner: Predicate

    def compile(self) -> tuple[str, list[object]]:
        sql, params = self.inner.compile()
        return f"NOT ({sql})", params


class SelectExpr(Generic[V]):
    def __init__(self, sql: str, decoder: Callable[[Any], V], label: str):
        self.sql = sql
        self.decoder = decoder
        self.label = label

    def aliased_sql(self) -> str:
        return f"{self.sql} AS {self.label}"


class Column(SelectExpr[V]):
    def __init__(self, table_alias: str, column_name: str, decoder: Callable[[Any], V], label: str | None = None):
        self.table_alias = table_alias
        self.column_name = column_name
        super().__init__(f"{table_alias}.{column_name}", decoder, label or column_name)

    def _cmp(self, op: str, value: object) -> Predicate:
        if isinstance(value, SelectExpr):
            return _SqlPredicate(f"{self.sql} {op} {value.sql}", ())
        return _SqlPredicate(f"{self.sql} {op} ?", (value,))

    def __eq__(self, other: object) -> Predicate:  # type: ignore[override]
        return self.is_null() if other is None else self._cmp("=", other)

    def __ne__(self, other: object) -> Predicate:  # type: ignore[override]
        return self.not_null() if other is None else self._cmp("!=", other)

    def __lt__(self, other: object) -> Predicate:
        return self._cmp("<", other)

    def __le__(self, other: object) -> Predicate:
        return self._cmp("<=", other)

    def __gt__(self, other: object) -> Predicate:
        return self._cmp(">", other)

    def __ge__(self, other: object) -> Predicate:
        return self._cmp(">=", other)

    def in_(self, values: Sequence[object]) -> Predicate:
        if not values:
            return _SqlPredicate("0")
        placeholders = ",".join("?" for _ in values)
        return _SqlPredicate(f"{self.sql} IN ({placeholders})", tuple(values))

    def like(self, pattern: str) -> Predicate:
        return _SqlPredicate(f"{self.sql} LIKE ?", (pattern,))

    def is_null(self) -> Predicate:
        return _SqlPredicate(f"{self.sql} IS NULL")

    def not_null(self) -> Predicate:
        return _SqlPredicate(f"{self.sql} IS NOT NULL")

    def desc(self) -> Order:
        return Order(self, descending=True)

    def asc(self) -> Order:
        return Order(self, descending=False)

    def as_(self, label: str) -> SelectExpr[V]:
        return SelectExpr(self.sql, self.decoder, label)


@dataclass(frozen=True)
class Order:
    expr: SelectExpr[Any]
    descending: bool = False

    def compile(self) -> str:
        direction = "DESC" if self.descending else "ASC"
        return f"{self.expr.sql} {direction}"


@dataclass(frozen=True)
class Join:
    table: Table[Any]
    on: Predicate
    kind: Literal["JOIN", "LEFT JOIN"] = "JOIN"

    def compile(self) -> tuple[str, list[object]]:
        on_sql, params = self.on.compile()
        return f"{self.kind} {self.table.sql_name} {self.table.alias} ON {on_sql}", params


@dataclass(frozen=True)
class JoinedQuery:
    """Fluent join chain starting from a base table."""

    from_: Table[Any]
    joins: tuple[Join, ...]

    def join(self, table: Table[Any], *, on: Predicate, kind: Literal["JOIN", "LEFT JOIN"] = "JOIN") -> JoinedQuery:
        return JoinedQuery(self.from_, (*self.joins, Join(table, on, kind)))


class ColumnAccessor:
    """Attribute-style access to table columns. Raises AttributeError for unknown names."""

    def __init__(self, columns: dict[str, Column[Any]]):
        self._columns = columns

    def __getattr__(self, name: str) -> Column[Any]:
        if name not in self._columns:
            raise AttributeError(f"No column {name!r} in table")
        return self._columns[name]


@dataclass(frozen=True)
class Table(Generic[T]):
    sql_name: str
    alias: str
    model_cls: type[T] | None = None
    columns: dict[str, Column[Any]] = field(default_factory=dict)
    field_columns: tuple[tuple[str, str], ...] = ()

    @property
    def c(self) -> ColumnAccessor:
        return ColumnAccessor(self.columns)

    def all_columns(self) -> list[SelectExpr[Any]]:
        if self.model_cls is None:
            raise TypeError(f"Table {self.sql_name} does not have a row model")
        result: list[SelectExpr[Any]] = []
        for _field_name, column_name in self.field_columns:
            column = self.columns[column_name]
            result.append(SelectExpr(column.sql, column.decoder, column_name))
        return result

    def with_alias(self, alias: str) -> Table[T]:
        columns = {
            name: Column(alias, column.column_name, column.decoder, label=column.label)
            for name, column in self.columns.items()
        }
        return Table(
            sql_name=self.sql_name,
            alias=alias,
            model_cls=self.model_cls,
            columns=columns,
            field_columns=self.field_columns,
        )

    def join(self, other: Table[Any], *, on: Predicate, kind: Literal["JOIN", "LEFT JOIN"] = "JOIN") -> JoinedQuery:
        return JoinedQuery(from_=self, joins=(Join(other, on, kind),))


def _table_for_model(model_cls: type[T], sql_name: str, alias: str) -> Table[T]:
    columns: dict[str, Column[Any]] = {}
    field_columns: list[tuple[str, str]] = []
    for item in model_cls.__db_fields__:
        column_name = item.metadata["db_column"]
        decoder = item.metadata.get("db_decoder", _identity)
        columns[column_name] = Column(alias, column_name, decoder)
        field_columns.append((item.name, column_name))
    return Table(
        sql_name=sql_name,
        alias=alias,
        model_cls=model_cls,
        columns=columns,
        field_columns=tuple(field_columns),
    )


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

    def __init__(self, conn: sqlite3.Connection, lock: RLock):
        self._conn = conn
        self._lock = lock

    def __enter__(self) -> QuerySnapshot:
        self._lock.acquire()
        self._conn.execute("BEGIN")
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        try:
            self._conn.rollback()
        finally:
            self._lock.release()

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

    @overload
    def select(
        self,
        from_: Table[T] | JoinedQuery,
        *,
        where: Predicate | None = None,
        joins: Sequence[Join] = (),
        order_by: Sequence[Order] = (),
        limit: int | None = None,
    ) -> list[T]: ...

    @overload
    def select(
        self,
        from_: Table[Any] | JoinedQuery,
        *,
        columns: Sequence[SelectExpr[Any]],
        where: Predicate | None = None,
        joins: Sequence[Join] = (),
        order_by: Sequence[Order] = (),
        limit: int | None = None,
    ) -> list[Any]: ...

    def select(
        self,
        from_: Table[Any] | JoinedQuery,
        *,
        columns: Sequence[SelectExpr[Any]] | None = None,
        where: Predicate | None = None,
        joins: Sequence[Join] = (),
        order_by: Sequence[Order] = (),
        limit: int | None = None,
    ) -> list[Any]:
        base_table: Table[Any]
        all_joins: list[Join]
        if isinstance(from_, JoinedQuery):
            base_table = from_.from_
            all_joins = list(from_.joins) + list(joins)
        else:
            base_table = from_
            all_joins = list(joins)

        params: list[object] = []
        select_exprs = list(columns) if columns is not None else base_table.all_columns()
        cols_sql = ", ".join(expr.aliased_sql() for expr in select_exprs)
        sql = f"SELECT {cols_sql} FROM {base_table.sql_name} {base_table.alias}"
        for join in all_joins:
            join_sql, join_params = join.compile()
            sql += f" {join_sql}"
            params.extend(join_params)
        if where is not None:
            where_sql, where_params = where.compile()
            sql += f" WHERE {where_sql}"
            params.extend(where_params)
        if order_by:
            sql += " ORDER BY " + ", ".join(order.compile() for order in order_by)
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = self._fetchall(sql, params)
        if columns is None:
            return [_decode_row(base_table.model_cls, row) for row in rows]
        return [Row({expr.label: expr.decoder(row[expr.label]) for expr in select_exprs}) for row in rows]

    def one(
        self,
        from_: Table[Any],
        *,
        columns: Sequence[SelectExpr[Any]] | None = None,
        where: Predicate | None = None,
        joins: Sequence[Join] = (),
        order_by: Sequence[Order] = (),
    ) -> Any | None:
        rows = self.select(from_, columns=columns, where=where, joins=joins, order_by=order_by, limit=1)
        return rows[0] if rows else None

    def scalar(
        self,
        expr: SelectExpr[V],
        *,
        from_: Table[Any],
        where: Predicate | None = None,
        joins: Sequence[Join] = (),
    ) -> V | None:
        row = self.one(from_, columns=(expr,), where=where, joins=joins)
        if row is None:
            return None
        return getattr(row, expr.label)

    def count(self, from_: Table[Any], *, where: Predicate | None = None, joins: Sequence[Join] = ()) -> int:
        count_expr = SelectExpr[int]("COUNT(*)", int, "count_value")
        value = self.scalar(count_expr, from_=from_, where=where, joins=joins)
        return int(value or 0)

    def exists(self, from_: Table[Any], *, where: Predicate | None = None, joins: Sequence[Join] = ()) -> bool:
        return self.count(from_, where=where, joins=joins) > 0


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


@db_row_model
class Attempt:
    task_id: JobName = db_field("task_id", _decode_job_name)
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
class Job:
    job_id: JobName = db_field("job_id", _decode_job_name)
    request: cluster_pb2.Controller.LaunchJobRequest = db_field(
        "request_proto",
        _proto_decoder(cluster_pb2.Controller.LaunchJobRequest),
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

    def is_finished(self) -> bool:
        return self.state in (
            cluster_pb2.JOB_STATE_SUCCEEDED,
            cluster_pb2.JOB_STATE_FAILED,
            cluster_pb2.JOB_STATE_KILLED,
            cluster_pb2.JOB_STATE_UNSCHEDULABLE,
        )

    @property
    def is_coscheduled(self) -> bool:
        return self.request.HasField("coscheduling")

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
class Task:
    task_id: JobName = db_field("task_id", _decode_job_name)
    job_id: JobName = db_field("job_id", _decode_job_name)
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
    attempts: tuple[Attempt, ...] = field(default_factory=tuple)

    def is_finished(self) -> bool:
        if self.state == cluster_pb2.TASK_STATE_SUCCEEDED:
            return True
        if self.state in (cluster_pb2.TASK_STATE_KILLED, cluster_pb2.TASK_STATE_UNSCHEDULABLE):
            return True
        if self.state == cluster_pb2.TASK_STATE_FAILED:
            return self.failure_count > self.max_retries_failure
        if self.state == cluster_pb2.TASK_STATE_WORKER_FAILED:
            return self.preemption_count > self.max_retries_preemption
        return False

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
        if self.state in TERMINAL_TASK_STATES:
            return False
        if self.current_attempt_id < 0:
            return True
        return self.state == cluster_pb2.TASK_STATE_PENDING and not self.is_finished()

    def is_live(self) -> bool:
        return self.state not in TERMINAL_TASK_STATES

    def is_dead(self) -> bool:
        return self.state in TERMINAL_TASK_STATES

    def is_retry_exhausted(self) -> bool:
        if self.state == cluster_pb2.TASK_STATE_FAILED:
            return self.failure_count > self.max_retries_failure
        if self.state == cluster_pb2.TASK_STATE_WORKER_FAILED:
            return self.preemption_count > self.max_retries_preemption
        return False


@db_row_model
class Worker:
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
        return self.metadata.cpu_count * 1000 - self.committed_cpu_millicores

    @property
    def available_memory(self) -> int:
        return self.metadata.memory_bytes - self.committed_mem

    @property
    def available_gpus(self) -> int:
        return get_gpu_count(self.metadata.device) - self.committed_gpu

    @property
    def available_tpus(self) -> int:
        return get_tpu_count(self.metadata.device) - self.committed_tpu

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
    job_id: JobName = db_field("job_id", _decode_job_name)
    metadata: dict[str, str] = db_field("metadata_json", _decode_json_dict)
    registered_at: Timestamp = db_field("registered_at_ms", _decode_timestamp_ms)


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


@dataclass(frozen=True)
class EndpointQuery:
    endpoint_ids: tuple[str, ...] = ()
    name_prefix: str | None = None
    exact_name: str | None = None
    job_ids: tuple[JobName, ...] = ()
    job_id: JobName | None = None
    task_ids: tuple[JobName, ...] = ()
    include_terminal_jobs: bool = False
    limit: int | None = None


JOBS = _table_for_model(Job, "jobs", "j")
TASKS = _table_for_model(Task, "tasks", "t")
ATTEMPTS = _table_for_model(Attempt, "task_attempts", "a")
WORKERS = _table_for_model(Worker, "workers", "w")
ENDPOINTS = _table_for_model(Endpoint, "endpoints", "e")
TXN_ACTIONS = _table_for_model(TransactionAction, "txn_actions", "ta")
WORKER_ATTRIBUTES = Table[tuple[str, str]](
    sql_name="worker_attributes",
    alias="wa",
    columns={
        "worker_id": Column("wa", "worker_id", _decode_worker_id),
        "key": Column("wa", "key", _decode_str),
        "value_type": Column("wa", "value_type", _decode_str),
        "str_value": Column("wa", "str_value", _nullable(_decode_str)),
        "int_value": Column("wa", "int_value", _nullable(_decode_int)),
        "float_value": Column("wa", "float_value", _nullable(float)),
    },
)
WORKER_TASK_HISTORY = Table[tuple[str, str]](
    sql_name="worker_task_history",
    alias="wth",
    columns={
        "worker_id": Column("wth", "worker_id", _decode_worker_id),
        "task_id": Column("wth", "task_id", _decode_job_name),
        "assigned_at_ms": Column("wth", "assigned_at_ms", _decode_timestamp_ms),
    },
)
WORKER_RESOURCE_HISTORY = Table[tuple[str, str]](
    sql_name="worker_resource_history",
    alias="wrh",
    columns={
        "worker_id": Column("wrh", "worker_id", _decode_worker_id),
        "snapshot_proto": Column(
            "wrh",
            "snapshot_proto",
            _proto_decoder(cluster_pb2.WorkerResourceSnapshot),
        ),
        "timestamp_ms": Column("wrh", "timestamp_ms", _decode_timestamp_ms),
    },
)
RESERVATION_CLAIMS = Table[tuple[str, str]](
    sql_name="reservation_claims",
    alias="rc",
    columns={
        "worker_id": Column("rc", "worker_id", _decode_worker_id),
        "job_id": Column("rc", "job_id", _decode_str),
        "entry_idx": Column("rc", "entry_idx", _decode_int),
    },
)
SCALING_GROUPS = Table(
    sql_name="scaling_groups",
    alias="sg",
    columns={
        "name": Column("sg", "name", _decode_str),
        "consecutive_failures": Column("sg", "consecutive_failures", _decode_int),
        "backoff_until_ms": Column("sg", "backoff_until_ms", _decode_timestamp_ms),
        "last_scale_up_ms": Column("sg", "last_scale_up_ms", _decode_timestamp_ms),
        "last_scale_down_ms": Column("sg", "last_scale_down_ms", _decode_timestamp_ms),
        "quota_exceeded_until_ms": Column("sg", "quota_exceeded_until_ms", _decode_timestamp_ms),
        "quota_reason": Column("sg", "quota_reason", _decode_str),
        "updated_at_ms": Column("sg", "updated_at_ms", _decode_timestamp_ms),
    },
)
SLICES = Table(
    sql_name="slices",
    alias="sl",
    columns={
        "slice_id": Column("sl", "slice_id", _decode_str),
        "scale_group": Column("sl", "scale_group", _decode_str),
        "lifecycle": Column("sl", "lifecycle", _decode_str),
        "vm_addresses": Column("sl", "vm_addresses", _decode_json_list),
        "created_at_ms": Column("sl", "created_at_ms", _decode_timestamp_ms),
        "last_active_ms": Column("sl", "last_active_ms", _decode_timestamp_ms),
        "error_message": Column("sl", "error_message", _decode_str),
    },
)
TRACKED_WORKERS = Table[tuple[str, str]](
    sql_name="tracked_workers",
    alias="tw",
    columns={
        "worker_id": Column("tw", "worker_id", _decode_str),
        "slice_id": Column("tw", "slice_id", _decode_str),
        "scale_group": Column("tw", "scale_group", _decode_str),
        "internal_address": Column("tw", "internal_address", _decode_str),
    },
)
ENDPOINT_TASKS = Table[tuple[str, str]](
    sql_name="endpoints",
    alias="et",
    columns={
        "endpoint_id": Column("et", "endpoint_id", _decode_str),
        "task_id": Column("et", "task_id", _decode_job_name),
    },
)
JOBS.columns["parent_job_id"] = Column("j", "parent_job_id", _nullable(_decode_job_name))
TASKS.columns["priority_neg_depth"] = Column("t", "priority_neg_depth", _decode_int)
TASKS.columns["priority_root_submitted_ms"] = Column("t", "priority_root_submitted_ms", _decode_timestamp_ms)
TASKS.columns["task_index"] = Column("t", "task_index", _decode_int)
TASKS.columns.pop("current_worker_id", None)
TASKS.columns.pop("current_worker_address", None)
object.__setattr__(
    TASKS,
    "field_columns",
    tuple(
        (field_name, column_name)
        for field_name, column_name in TASKS.field_columns
        if column_name not in {"current_worker_id", "current_worker_address"}
    ),
)


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


def _tasks_with_attempts(tasks: Sequence[Task], attempts: Sequence[Attempt]) -> list[Task]:
    attempts_by_task: dict[JobName, list[Attempt]] = {}
    for attempt in attempts:
        attempts_by_task.setdefault(attempt.task_id, []).append(attempt)
    return [Task(**{**task.__dict__, "attempts": tuple(attempts_by_task.get(task.task_id, ()))}) for task in tasks]


def endpoint_query_predicate(query: EndpointQuery) -> tuple[list[Join], Predicate | None]:
    """Translate EndpointQuery to (joins, where) for snapshot.select(ENDPOINTS, ...)."""
    joins: list[Join] = []
    where: Predicate | None = None
    if query.endpoint_ids:
        where = ENDPOINTS.c.endpoint_id.in_(list(query.endpoint_ids))
    if query.name_prefix:
        predicate = ENDPOINTS.c.name.like(f"{query.name_prefix}%")
        where = predicate if where is None else where & predicate
    if query.exact_name:
        predicate = ENDPOINTS.c.name == query.exact_name
        where = predicate if where is None else where & predicate
    job_ids = list(query.job_ids)
    if query.job_id is not None:
        job_ids.append(query.job_id)
    if job_ids:
        predicate = ENDPOINTS.c.job_id.in_([job_id.to_wire() for job_id in job_ids])
        where = predicate if where is None else where & predicate
    if query.task_ids:
        joins.append(
            Join(
                table=ENDPOINT_TASKS,
                on=ENDPOINTS.c.endpoint_id == ENDPOINT_TASKS.c.endpoint_id,
            )
        )
        predicate = ENDPOINT_TASKS.c.task_id.in_([task_id.to_wire() for task_id in query.task_ids])
        where = predicate if where is None else where & predicate
    if not query.include_terminal_jobs:
        joins.append(Join(table=JOBS, on=ENDPOINTS.c.job_id == JOBS.c.job_id))
        predicate = ~JOBS.c.state.in_(list(TERMINAL_JOB_STATES))
        where = predicate if where is None else where & predicate
    return joins, where


class TransactionCursor:
    """Wraps a raw sqlite3.Cursor and adds typed mutation helpers.

    The insert/update/delete methods are thin SQL builders that accept
    SQL-compatible Python values directly. Use execute/executemany/executescript
    as an escape hatch when the builders don't cover the needed SQL shape.
    """

    def __init__(self, cursor: sqlite3.Cursor):
        self._cursor = cursor

    def insert(self, table: str, values: dict[str, Any]) -> None:
        """Insert a single row. Values must already be SQL-compatible types."""
        cols = ", ".join(values.keys())
        placeholders = ", ".join("?" for _ in values)
        self._cursor.execute(
            f"INSERT INTO {table} ({cols}) VALUES ({placeholders})",
            tuple(values.values()),
        )

    def update(self, table: str, updates: dict[str, Any], where: Predicate) -> int:
        """Update rows matching predicate. Returns number of rows affected."""
        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
        where_sql, where_params = where.compile()
        self._cursor.execute(
            f"UPDATE {table} SET {set_clause} WHERE {where_sql}",
            tuple(updates.values()) + tuple(where_params),
        )
        return self._cursor.rowcount

    def delete(self, table: str, where: Predicate) -> int:
        """Delete rows matching predicate. Returns number of rows affected."""
        where_sql, where_params = where.compile()
        self._cursor.execute(
            f"DELETE FROM {table} WHERE {where_sql}",
            tuple(where_params),
        )
        return self._cursor.rowcount

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

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._configure(self._conn)
        self.apply_migrations()

    @property
    def db_path(self) -> Path:
        return self._db_path

    @staticmethod
    def _configure(conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA foreign_keys = ON")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

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

    def decode_worker(self, row: sqlite3.Row) -> Worker:
        return _decode_row(Worker, row)

    def decode_job(self, row: sqlite3.Row) -> Job:
        return _decode_row(Job, row)

    def decode_task(self, row: sqlite3.Row) -> Task:
        return _decode_row(Task, row)

    def apply_migrations(self) -> None:
        """Apply pending SQL migrations from the migrations/ directory.

        Migrations run outside a transaction because executescript() implicitly
        commits. This is fine: migrations only run at startup before any
        concurrent access. Each migration is applied then recorded; if the
        process crashes mid-migration the partially-applied file won't be in
        schema_migrations and the next startup will re-run it (migrations must
        be idempotent via IF NOT EXISTS / IF EXISTS guards).
        """
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

        for path in sorted(migrations_dir.glob("*.sql")):
            if path.name in applied:
                continue
            sql = path.read_text(encoding="utf-8")
            self._conn.executescript(sql)
            with self.transaction() as cur:
                cur.execute(
                    "INSERT INTO schema_migrations(name, applied_at_ms) VALUES (?, ?)",
                    (path.name, Timestamp.now().epoch_ms()),
                )

    def next_sequence(self, key: str, *, cur: TransactionCursor) -> int:
        row = cur.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        if row is None:
            cur.execute("INSERT INTO meta(key, value) VALUES (?, ?)", (key, 1))
            return 1
        value = int(row[0]) + 1
        cur.execute("UPDATE meta SET value = ? WHERE key = ?", (value, key))
        return value

    def backup_to(self, destination: Path) -> None:
        """Create a hot backup to ``destination`` using SQLite backup API."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            dest = sqlite3.connect(str(destination))
            try:
                self._conn.backup(dest)
                dest.commit()
            finally:
                dest.close()

    def replace_from(self, source: str | Path) -> None:
        """Replace current DB file with ``source`` and reopen connection.

        ``source`` may be a remote path (e.g. ``gs://...``) thanks to fsspec.
        Only called at startup before concurrent access begins.
        """
        import fsspec.core

        with self._lock:
            # Download to a temp file first so a failed copy doesn't leave
            # the DB connection closed with no file to reopen.
            tmp_path = self._db_path.with_suffix(".tmp")
            with fsspec.core.open(str(source), "rb") as src, open(tmp_path, "wb") as dst:
                dst.write(src.read())
            self._conn.close()
            tmp_path.rename(self._db_path)
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._configure(self._conn)
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

    Derived from tasks JOIN task_attempts rather than a materialized view.
    """
    if not worker_ids:
        return {}
    placeholders = ",".join("?" for _ in worker_ids)
    with db.snapshot() as q:
        rows = q.raw(
            f"SELECT a.worker_id, t.task_id FROM tasks t "
            f"JOIN task_attempts a ON t.task_id = a.task_id AND t.current_attempt_id = a.attempt_id "
            f"WHERE a.worker_id IN ({placeholders}) AND t.state IN (?, ?, ?)",
            (*[str(wid) for wid in worker_ids], *ACTIVE_TASK_STATES),
            decoders={"worker_id": _decode_worker_id, "task_id": _decode_job_name},
        )
    running: dict[WorkerId, set[JobName]] = {wid: set() for wid in worker_ids}
    for row in rows:
        running[row.worker_id].add(row.task_id)
    return running


def tasks_for_job_with_attempts(db: ControllerDB, job_id: JobName) -> list[Task]:
    """Fetch all tasks for a job with their attempt history."""
    with db.snapshot() as q:
        tasks = q.select(
            TASKS,
            where=TASKS.c.job_id == job_id.to_wire(),
            order_by=(TASKS.c.job_id.asc(), TASKS.c.task_index.asc()),
        )
        if not tasks:
            return []
        attempts = q.select(
            ATTEMPTS,
            where=ATTEMPTS.c.task_id.in_([t.task_id.to_wire() for t in tasks]),
            order_by=(ATTEMPTS.c.task_id.asc(), ATTEMPTS.c.attempt_id.asc()),
        )
    return _tasks_with_attempts(tasks, attempts)


def healthy_active_workers_with_attributes(db: ControllerDB) -> list[Worker]:
    """Fetch all healthy, active workers with their attributes populated."""
    with db.snapshot() as q:
        workers = q.select(WORKERS, where=(WORKERS.c.healthy == 1) & (WORKERS.c.active == 1))
        if not workers:
            return []
        attrs = q.select(
            WORKER_ATTRIBUTES,
            columns=(
                WORKER_ATTRIBUTES.c.worker_id,
                WORKER_ATTRIBUTES.c.key,
                WORKER_ATTRIBUTES.c.value_type,
                WORKER_ATTRIBUTES.c.str_value,
                WORKER_ATTRIBUTES.c.int_value,
                WORKER_ATTRIBUTES.c.float_value,
            ),
            where=WORKER_ATTRIBUTES.c.worker_id.in_([str(w.worker_id) for w in workers]),
        )
    attrs_by_worker = _decode_attribute_rows(attrs)
    return [dc_replace(w, attributes=attrs_by_worker.get(w.worker_id, {})) for w in workers]
