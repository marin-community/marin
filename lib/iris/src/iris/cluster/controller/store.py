# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed entity operations for task state transitions.

Sits between transitions.py (state machine logic) and db.py (SQLite logistics).
Methods take and return dataclasses, not raw SQL rows. The store enforces
multi-table invariants (e.g. terminate = attempt terminal + task update + worker
decommit + endpoint delete) in a single method instead of scattering across
inline SQL blocks.

Stores are process-scoped: a single instance is constructed by
``ControllerDB.__init__`` and reused across transactions. Every method takes
the open ``Cursor`` as its first argument so writes land inside
the caller's transaction. Process-scoped caches (worker attributes,
job_config) survive across transactions and are invalidated via cursor
post-commit hooks.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field, replace as dc_replace
from enum import StrEnum
from threading import Lock, RLock
from typing import Any, Literal, Protocol, TypeVar, overload
from iris.cluster.constraints import AttributeValue
from iris.cluster.controller.budget import UserBudgetDefaults
from iris.cluster.controller.codec import resource_spec_from_scalars
from iris.cluster.controller.schema import (
    ACTIVE_TASK_STATES,
    ATTEMPT_PROJECTION,
    ENDPOINT_PROJECTION,
    EXECUTING_TASK_STATES,
    JOB_CONFIG_JOIN,
    JOB_CONFIG_PROJECTION,
    JOB_DETAIL_PROJECTION,
    JOB_ROW_PROJECTION,
    TASK_DETAIL_PROJECTION,
    TASK_DETAIL_SELECT_T,
    WORKER_DETAIL_PROJECTION,
    WORKER_ROW_PROJECTION,
    EndpointRow,
    JobConfigRow,
    JobDetailRow,
    JobRow,
    ResourceSpec,
    TaskDetailRow,
    WorkerActiveRow,
    WorkerDetailRow,
    WorkerRow,
    decode_timestamp_ms,
    decode_worker_id,
    tasks_with_attempts,
)
from iris.cluster.types import (
    TERMINAL_JOB_STATES,
    TERMINAL_TASK_STATES,
    JobName,
    JobState,
    TaskState,
    WorkerId,
    get_gpu_count,
    get_tpu_count,
)
from iris.rpc import job_pb2
from rigging.timing import Deadline, Duration, Timestamp

logger = logging.getLogger(__name__)


def sql_placeholders(n: int) -> str:
    """Return ``?,?,?`` style placeholder string for SQL ``IN`` lists."""
    return ",".join("?" * n)


class WhereBuilder:
    """Accumulate conditional WHERE clauses + params without repetitive boilerplate.

    Usage:
        wb = WhereBuilder()
        if flt.worker_id is not None:
            wb.eq("t.current_worker_id", str(flt.worker_id))
        wb.in_("t.state", states)   # skips if states is empty/None
        sql, params = wb.build()    # ("WHERE ...", (..., ...)) or ("", ())
    """

    def __init__(self) -> None:
        self._clauses: list[str] = []
        self._params: list[object] = []

    def eq(self, col: str, value: object) -> None:
        self._clauses.append(f"{col} = ?")
        self._params.append(value)

    def is_null(self, col: str) -> None:
        self._clauses.append(f"{col} IS NULL")

    def in_(self, col: str, values: Iterable[object] | None) -> None:
        if not values:
            return
        values = tuple(values)
        self._clauses.append(f"{col} IN ({sql_placeholders(len(values))})")
        self._params.extend(values)

    def raw(self, clause: str, *params: object) -> None:
        """Escape hatch for non-trivial conditions; keeps param alignment."""
        self._clauses.append(clause)
        self._params.extend(params)

    def build(self) -> tuple[str, tuple[object, ...]]:
        if not self._clauses:
            return "", ()
        return "WHERE " + " AND ".join(self._clauses), tuple(self._params)


# ---------------------------------------------------------------------------
# Protocols — structural interfaces used by stores to decouple from db.py.
# ControllerDB and Cursor satisfy these structurally; no explicit
# declaration is needed on those classes.
# ---------------------------------------------------------------------------


class Cursor(Protocol):
    """Methods that store operations call on a Cursor.

    Historically includes both read and write shape — tightened in a later pass.
    A read-only scope (``QuerySnapshot``) satisfies only the ``execute`` slice;
    calling write-only members on one fails at runtime, which is intentional.
    """

    def execute(self, sql: str, params: tuple = ...) -> sqlite3.Cursor: ...

    def executemany(self, sql: str, params: Iterable[tuple]) -> sqlite3.Cursor: ...

    def on_commit(self, hook: Callable[[], None]) -> None: ...

    @property
    def rowcount(self) -> int: ...

    @property
    def lastrowid(self) -> int | None: ...


class WriteCursor(Cursor, Protocol):
    """Cursor inside an IMMEDIATE transaction — supports commit hooks."""

    def on_commit(self, hook: Callable[[], None]) -> None: ...

    @property
    def lastrowid(self) -> int | None: ...

    @property
    def rowcount(self) -> int: ...


class DbBackend(Protocol):
    """Methods that stores call on a ControllerDB."""

    def read_snapshot(self) -> AbstractContextManager[Cursor]: ...

    def transaction(self) -> AbstractContextManager[WriteCursor]: ...

    def execute(self, query: str, params: tuple | list = ...) -> None: ...


class _DecodedRow:
    """Lightweight attribute-access wrapper over a decoded SQL row."""

    __slots__ = ("_data",)

    def __init__(self, data: dict[str, Any]) -> None:
        object.__setattr__(self, "_data", data)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def decoded_rows(
    cur: Cursor,
    sql: str,
    params: tuple = (),
    decoders: dict[str, Callable] | None = None,
) -> list[_DecodedRow]:
    """Execute ``sql`` on ``cur`` and return decoded rows with attribute access.

    Replaces the ``QuerySnapshot.raw`` helper so read methods can operate on
    any ``Cursor`` — including a write cursor when the caller is already
    inside a transaction.
    """
    cursor = cur.execute(sql, params)
    col_names = [desc[0] for desc in cursor.description]
    active_decoders = decoders or {}
    out: list[_DecodedRow] = []
    for raw_row in cursor.fetchall():
        data = {
            name: active_decoders[name](raw_row[name]) if name in active_decoders else raw_row[name]
            for name in col_names
        }
        out.append(_DecodedRow(data))
    return out


# ---------------------------------------------------------------------------
# Domain predicates — logic about task/job/attempt states.
# ---------------------------------------------------------------------------


def task_is_finished(
    state: int, failure_count: int, max_retries_failure: int, preemption_count: int, max_retries_preemption: int
) -> bool:
    """Whether a task has reached a terminal state with no remaining retries."""
    if state == job_pb2.TASK_STATE_SUCCEEDED:
        return True
    if state in (job_pb2.TASK_STATE_KILLED, job_pb2.TASK_STATE_UNSCHEDULABLE):
        return True
    if state == job_pb2.TASK_STATE_FAILED:
        return failure_count > max_retries_failure
    if state in (job_pb2.TASK_STATE_WORKER_FAILED, job_pb2.TASK_STATE_PREEMPTED):
        return preemption_count > max_retries_preemption
    return False


class TaskRowLike(Protocol):
    """Structural interface for task rows used by scheduling predicates.

    Satisfied by ``TaskRow`` and ``TaskDetailRow`` — any row carrying state
    plus retry counters and the current attempt id can be evaluated for
    finish/schedulability without coupling to a concrete row type.
    """

    state: int
    failure_count: int
    max_retries_failure: int
    preemption_count: int
    max_retries_preemption: int
    current_attempt_id: int


def task_row_is_finished(task: TaskRowLike) -> bool:
    return task_is_finished(
        task.state, task.failure_count, task.max_retries_failure, task.preemption_count, task.max_retries_preemption
    )


def task_row_can_be_scheduled(task: TaskRowLike) -> bool:
    if task.state != job_pb2.TASK_STATE_PENDING:
        return False
    return task.current_attempt_id < 0 or not task_is_finished(
        task.state, task.failure_count, task.max_retries_failure, task.preemption_count, task.max_retries_preemption
    )


def job_scheduling_deadline(scheduling_deadline_epoch_ms: int | None) -> Deadline | None:
    """Compute scheduling deadline from epoch ms."""
    if scheduling_deadline_epoch_ms is None:
        return None
    return Deadline.after(Timestamp.from_ms(scheduling_deadline_epoch_ms), Duration.from_ms(0))


def attempt_is_terminal(state: int) -> bool:
    """Check if an attempt is in a terminal state."""
    return state in TERMINAL_TASK_STATES


def attempt_is_worker_failure(state: int) -> bool:
    """Check if an attempt is a worker failure or preemption."""
    return state in (job_pb2.TASK_STATE_WORKER_FAILED, job_pb2.TASK_STATE_PREEMPTED)


# ---------------------------------------------------------------------------
# Domain summary dataclasses
# ---------------------------------------------------------------------------


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
class UserBudget:
    user_id: str
    budget_limit: int
    max_band: int
    updated_at: Timestamp


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


# ---------------------------------------------------------------------------
# Shared data types (used by both store.py and transitions.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskUpdate:
    """Single task state update applied in a batch."""

    task_id: JobName
    attempt_id: int
    new_state: int
    error: str | None = None
    exit_code: int | None = None
    resource_usage: job_pb2.ResourceUsage | None = None
    container_id: str | None = None


@dataclass(frozen=True)
class HeartbeatApplyRequest:
    """Batch of worker heartbeat updates applied atomically."""

    worker_id: WorkerId
    worker_resource_snapshot: job_pb2.WorkerResourceSnapshot | None
    updates: list[TaskUpdate]


# ---------------------------------------------------------------------------
# Dataclasses — read-only views and write inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TimedOutTask:
    """A running task that has exceeded its execution timeout."""

    task_id: JobName
    worker_id: WorkerId | None


@dataclass(frozen=True, slots=True)
class TaskSnapshot:
    """Read-only view of a task row for the transition planner.

    Built from a task row, its current attempt, and the cached job_config.
    Provides everything needed to decide the next transition without going
    back to the DB.
    """

    task_id: str
    job_id: JobName
    state: TaskState
    attempt_id: int
    attempt_state: TaskState
    failure_count: int
    preemption_count: int
    max_retries_failure: int
    max_retries_preemption: int
    worker_id: str | None
    has_coscheduling: bool
    resources: job_pb2.ResourceSpecProto | None


@dataclass(frozen=True, slots=True)
class SiblingSnapshot:
    """Read-only view of a coscheduled sibling for cascade decisions."""

    task_id: str
    attempt_id: int
    max_retries_preemption: int
    worker_id: str | None


@dataclass(frozen=True, slots=True)
class TaskTermination:
    """All inputs needed to terminate a task.

    ``finalize`` may be None when no attempt exists; the attempt UPDATE is skipped.
    ``finalize.attempt_state`` overrides the state written to the attempt row when
    it differs from the task state (e.g. attempt=WORKER_FAILED while task retries
    to PENDING). ``error`` is written to the task row; if the same string should
    apply to the attempt row, set ``finalize.error`` too.
    """

    task_id: str
    state: TaskState
    now_ms: int
    error: str | None = None
    finalize: AttemptFinalizer | None = None
    worker_id: str | None = None
    resources: job_pb2.ResourceSpecProto | None = None
    failure_count: int | None = None
    preemption_count: int | None = None
    exit_code: int | None = None


@dataclass(frozen=True, slots=True)
class TaskRetry:
    """All inputs needed to requeue a task to PENDING.

    Terminates the current attempt but resets the task row to PENDING
    so the scheduler can create a new attempt.
    """

    task_id: str
    finalize: AttemptFinalizer
    worker_id: str | None = None
    resources: job_pb2.ResourceSpecProto | None = None
    failure_count: int = 0
    preemption_count: int = 0


@dataclass(frozen=True, slots=True)
class ActiveStateUpdate:
    """Non-terminal task state update (BUILDING, RUNNING)."""

    task_id: str
    attempt_id: int
    state: TaskState
    error: str | None = None
    exit_code: int | None = None
    started_ms: int | None = None
    failure_count: int = 0
    preemption_count: int = 0


@dataclass(frozen=True, slots=True)
class TaskInsert:
    """All columns for an INSERT INTO tasks row."""

    task_id: str
    job_id: str
    task_index: int
    state: int
    submitted_at_ms: int
    max_retries_failure: int
    max_retries_preemption: int
    priority_neg_depth: int
    priority_root_submitted_ms: int
    priority_insertion: int
    priority_band: int


@dataclass(frozen=True, slots=True)
class WorkerAssignment:
    """Assign a task to a worker-backed slot (worker_id and address known)."""

    task_id: str
    attempt_id: int
    worker_id: str
    worker_address: str
    now_ms: int


@dataclass(frozen=True, slots=True)
class DirectAssignment:
    """Assign a task to a direct-provider slot (no backing worker daemon)."""

    task_id: str
    attempt_id: int
    now_ms: int


@dataclass(frozen=True, slots=True)
class AttemptFinalizer:
    """Fields needed to write a terminal row to task_attempts.

    Shared by TaskTermination and TaskRetry so both can delegate to a single
    ``_finalize_attempt`` helper.
    """

    task_id: str
    attempt_id: int
    attempt_state: TaskState
    now_ms: int
    error: str | None = None
    exit_code: int | None = None

    @classmethod
    def build(
        cls,
        task_id: str,
        attempt_id: int,
        state: TaskState | int,
        now_ms: int,
        error: str | None = None,
    ) -> AttemptFinalizer:
        return cls(
            task_id=task_id,
            attempt_id=attempt_id,
            attempt_state=state,
            now_ms=now_ms,
            error=error,
        )


@dataclass(frozen=True, slots=True)
class KillResult:
    """Tasks that need kill RPCs after a cascade."""

    tasks_to_kill: frozenset[JobName]
    task_kill_workers: dict[JobName, WorkerId]


@dataclass(frozen=True, slots=True)
class JobInsert:
    """All columns for an INSERT INTO jobs row.

    Covers both the main job insert and the reservation holder insert.
    """

    job_id: str
    user_id: str
    parent_job_id: str | None
    root_job_id: str
    depth: int
    state: int
    submitted_at_ms: int
    root_submitted_at_ms: int
    finished_at_ms: int | None
    scheduling_deadline_epoch_ms: int | None
    error: str | None
    num_tasks: int
    is_reservation_holder: bool
    name: str
    has_reservation: bool


@dataclass(frozen=True, slots=True)
class JobConfigInsert:
    """All columns for an INSERT INTO job_config row."""

    job_id: str
    name: str
    has_reservation: bool
    resources: ResourceSpec
    constraints_json: str | None
    has_coscheduling: int
    coscheduling_group_by: str
    scheduling_timeout_ms: int | None
    max_task_failures: int
    entrypoint_json: str
    environment_json: str
    bundle_id: str
    ports_json: str
    max_retries_failure: int
    max_retries_preemption: int
    timeout_ms: int | None
    preemption_policy: int
    existing_job_policy: int
    priority_band: int
    task_image: str
    submit_argv_json: str = "[]"
    reservation_json: str | None = None
    fail_if_exists: int = 0


@dataclass(frozen=True)
class EndpointQuery:
    endpoint_ids: tuple[str, ...] = ()
    name_prefix: str | None = None
    exact_name: str | None = None
    task_ids: tuple[JobName, ...] = ()
    limit: int | None = None


class TaskProjection(StrEnum):
    DETAIL = "detail"
    WITH_JOB = "with_job"
    WITH_JOB_CONFIG = "with_job_config"


@dataclass(frozen=True, slots=True)
class TaskFilter:
    """Closed WHERE-clause predicate for the tasks table.

    All set fields AND together; unset fields are not filtered on. Used by
    :meth:`TaskStore.query` as a single entry point for simple non-snapshot
    reads that differ only in their WHERE clause.

    ``worker_id`` and ``worker_is_null`` are mutually exclusive; setting both
    at construction raises ``ValueError``.
    """

    task_ids: tuple[str, ...] | None = None
    job_ids: tuple[str, ...] | None = None
    worker_id: WorkerId | None = None
    worker_is_null: bool = False
    states: frozenset[int] | None = None
    limit: int | None = None

    def __post_init__(self) -> None:
        if self.worker_id is not None and self.worker_is_null:
            raise ValueError("TaskFilter: worker_id and worker_is_null are mutually exclusive")


@dataclass(frozen=True, slots=True)
class JobDetailFilter:
    """Closed WHERE-clause predicate for the jobs table (with config join).

    All set fields AND together; unset fields are not filtered on. Used by
    :meth:`JobStore.query` as the single entry point for critical-path reads.
    """

    job_ids: tuple[str, ...] | None = None
    states: frozenset[int] | None = None
    has_reservation: bool | None = None
    limit: int | None = None


@dataclass(frozen=True, slots=True)
class WorkerFilter:
    """Closed WHERE-clause predicate for the workers table.

    All set fields AND together; unset fields are not filtered on. Used by
    :meth:`WorkerStore.query` for scheduling-tick reads that differ only in
    their WHERE clause.
    """

    worker_ids: tuple[WorkerId, ...] | None = None
    active: bool | None = None
    healthy: bool | None = None


@dataclass(frozen=True, slots=True)
class WorkerMetadata:
    """Worker environment metadata extracted from the registration RPC."""

    hostname: str
    ip_address: str
    cpu_count: int
    memory_bytes: int
    disk_bytes: int
    tpu_name: str
    tpu_worker_hostnames: str
    tpu_worker_id: int
    tpu_chips_per_host_bounds: str
    gpu_count: int
    gpu_name: str
    gpu_memory_mb: int
    gce_instance_name: str
    gce_zone: str
    git_hash: str
    device_json: str


@dataclass(frozen=True, slots=True)
class WorkerUpsert:
    """All inputs needed to insert or update a worker row."""

    worker_id: str
    address: str
    now_ms: int
    total_cpu_millicores: int
    total_memory_bytes: int
    total_gpu_count: int
    total_tpu_count: int
    device_type: str
    device_variant: str
    slice_id: str
    scale_group: str
    metadata: WorkerMetadata
    attributes: list[tuple[str, str, str | None, int | None, float | None]]


# SQLite caps host parameters at ~999. Leave headroom for fixed-position
# filter params (states, worker_id, limit) by chunking ID IN-lists well below
# that cap.
_ID_IN_CHUNK = 900


_T = TypeVar("_T")
_R = TypeVar("_R")


def chunk_ids(ids: Sequence[_T] | None, size: int = _ID_IN_CHUNK) -> list[tuple[_T, ...]] | None:
    """Chunk ids for IN-list queries. Returns None when ids is None (no filter applied)."""
    if ids is None:
        return None
    return [tuple(ids[i : i + size]) for i in range(0, len(ids), size)]


def run_chunked(
    chunks: list[tuple[_T, ...]] | None,
    limit: int | None,
    fetch: Callable[[tuple[_T, ...] | None, int | None], list[_R]],
) -> list[_R]:
    """Run a chunked IN-list query with optional row limit.

    `chunks is None` means no id filter — fetch is called once with (None, limit).
    When chunks are present, fetch is called per chunk with the remaining limit,
    stopping early once the limit is reached.
    Callers must short-circuit before calling this when chunks == [].
    """
    if chunks is None:
        return fetch(None, limit)
    results: list[_R] = []
    remaining = limit
    for chunk in chunks:
        if remaining is not None and remaining <= 0:
            break
        batch = fetch(chunk, remaining)
        results.extend(batch)
        if remaining is not None:
            remaining -= len(batch)
    return results


# ---------------------------------------------------------------------------
# EndpointStore
#
# Process-local in-memory cache for the ``endpoints`` table.
#
# Profiling showed that ``ListEndpoints`` dominated controller CPU — not because
# the SQL was slow per se, but because every call serialized through the
# read-connection pool and walked a large WAL to build a snapshot. The endpoints
# table is tiny (hundreds of rows) and only changes on explicit register /
# unregister, so it is a natural fit for a write-through in-memory cache.
#
# Design invariants:
#
# * Reads never touch the DB. All lookups are served from in-memory maps
#   guarded by an ``RLock`` — readers observe a consistent snapshot of the
#   indexes, never a torn state mid-update.
# * Writes execute the SQL inside the caller's transaction. The in-memory
#   update is scheduled as a post-commit hook on the cursor so memory only
#   changes after the DB has committed. If the transaction rolls back, the
#   hook never fires.
# * N is small enough (≈ hundreds) that linear scans for prefix / task / id
#   lookups are simpler and plenty fast. Extra indexes (by name, by task_id)
#   speed the two common cases.
#
# The store is the sole source of truth for endpoint reads; nothing else in
# the controller tree should SELECT from ``endpoints``.
# ---------------------------------------------------------------------------


class EndpointStore:
    """In-memory index of endpoint rows, kept in sync with the DB.

    Construct with a ``ControllerDB``; the store loads all existing rows at
    init time. Callers mutate through ``add`` / ``remove*`` methods that take
    the open ``Cursor`` so the SQL lands inside the caller's
    transaction. Memory is only updated after a successful commit via a
    cursor post-commit hook.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._by_id: dict[str, EndpointRow] = {}
        # One name can map to multiple endpoint_ids — the schema does not enforce
        # uniqueness on ``name``, and ``INSERT OR REPLACE`` keys off endpoint_id.
        self._by_name: dict[str, set[str]] = {}
        self._by_task: dict[JobName, set[str]] = {}

    # -- Loading --------------------------------------------------------------

    def _load_all(self, cur: Cursor) -> None:
        rows = ENDPOINT_PROJECTION.decode(
            cur.execute(f"SELECT {ENDPOINT_PROJECTION.select_clause()} FROM endpoints e").fetchall(),
        )
        with self._lock:
            self._by_id.clear()
            self._by_name.clear()
            self._by_task.clear()
            for row in rows:
                self._index(row)
        logger.info("EndpointStore loaded %d endpoint(s) from DB", len(rows))

    def _commit_index_update(
        self,
        cur: Cursor,
        *,
        add: EndpointRow | None = None,
        remove_ids: Iterable[str] = (),
    ) -> None:
        """Register index mutations to fire when ``cur``'s transaction commits."""
        # Capture mutable locals into the closure now, before the transaction commits.
        remove_list = list(remove_ids)

        def apply() -> None:
            with self._lock:
                for eid in remove_list:
                    self._unindex(eid)
                if add is not None:
                    self._unindex(add.endpoint_id)
                    self._index(add)

        cur.on_commit(apply)

    def _index(self, row: EndpointRow) -> None:
        self._by_id[row.endpoint_id] = row
        self._by_name.setdefault(row.name, set()).add(row.endpoint_id)
        self._by_task.setdefault(row.task_id, set()).add(row.endpoint_id)

    def _unindex(self, endpoint_id: str) -> EndpointRow | None:
        row = self._by_id.pop(endpoint_id, None)
        if row is None:
            return None
        name_ids = self._by_name.get(row.name)
        if name_ids is not None:
            name_ids.discard(endpoint_id)
            if not name_ids:
                self._by_name.pop(row.name, None)
        task_ids = self._by_task.get(row.task_id)
        if task_ids is not None:
            task_ids.discard(endpoint_id)
            if not task_ids:
                self._by_task.pop(row.task_id, None)
        return row

    # -- Reads ----------------------------------------------------------------

    def query(self, query: EndpointQuery = EndpointQuery()) -> list[EndpointRow]:
        """Return endpoint rows matching ``query``.

        All filters AND together, matching the semantics of the original SQL
        in :func:`iris.cluster.controller.db.endpoint_query_sql`.
        """
        with self._lock:
            # Narrow the candidate set using the most selective index available.
            if query.endpoint_ids:
                candidates: Iterable[EndpointRow] = (
                    self._by_id[eid] for eid in query.endpoint_ids if eid in self._by_id
                )
            elif query.task_ids:
                task_set = set(query.task_ids)
                candidates = (self._by_id[eid] for task_id in task_set for eid in self._by_task.get(task_id, ()))
            elif query.exact_name is not None:
                candidates = (self._by_id[eid] for eid in self._by_name.get(query.exact_name, ()))
            else:
                candidates = self._by_id.values()

            results: list[EndpointRow] = []
            for row in candidates:
                if query.name_prefix is not None and not row.name.startswith(query.name_prefix):
                    continue
                if query.exact_name is not None and row.name != query.exact_name:
                    continue
                if query.task_ids and row.task_id not in query.task_ids:
                    continue
                if query.endpoint_ids and row.endpoint_id not in query.endpoint_ids:
                    continue
                results.append(row)
                if query.limit is not None and len(results) >= query.limit:
                    break
            return results

    def resolve(self, name: str) -> EndpointRow | None:
        """Return any endpoint with exact ``name``, or None. Used by the actor proxy."""
        with self._lock:
            ids = self._by_name.get(name)
            if not ids:
                return None
            # Arbitrary but stable pick — the original SQL did not specify ORDER BY.
            return self._by_id[next(iter(ids))]

    def get(self, endpoint_id: str) -> EndpointRow | None:
        with self._lock:
            return self._by_id.get(endpoint_id)

    def all(self) -> list[EndpointRow]:
        with self._lock:
            return list(self._by_id.values())

    # -- Writes ---------------------------------------------------------------

    def add(self, cur: Cursor, endpoint: EndpointRow) -> bool:
        """Insert ``endpoint`` into the DB and schedule the memory update.

        Returns False (and writes nothing) if the owning task is already
        terminal. Otherwise inserts / replaces and schedules a post-commit
        hook that updates the in-memory indexes.
        """
        task_id = endpoint.task_id
        job_id, _ = task_id.require_task()
        row = cur.execute("SELECT state FROM tasks WHERE task_id = ?", (task_id.to_wire(),)).fetchone()
        if row is not None and int(row["state"]) in TERMINAL_TASK_STATES:
            return False

        cur.execute(
            "INSERT OR REPLACE INTO endpoints("
            "endpoint_id, name, address, job_id, task_id, metadata_json, registered_at_ms"
            ") VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                endpoint.endpoint_id,
                endpoint.name,
                endpoint.address,
                job_id.to_wire(),
                task_id.to_wire(),
                json.dumps(endpoint.metadata),
                endpoint.registered_at.epoch_ms(),
            ),
        )

        self._commit_index_update(cur, add=endpoint)
        return True

    def remove(self, cur: Cursor, endpoint_id: str) -> EndpointRow | None:
        """Remove a single endpoint by id. Returns the removed row snapshot, if any."""
        existing = self.get(endpoint_id)
        if existing is None:
            return None
        cur.execute("DELETE FROM endpoints WHERE endpoint_id = ?", (endpoint_id,))

        self._commit_index_update(cur, remove_ids=[endpoint_id])
        return existing

    def remove_by_task(self, cur: Cursor, task_id: JobName) -> list[str]:
        """Remove all endpoints owned by a task. Returns the removed endpoint_ids."""
        with self._lock:
            ids = list(self._by_task.get(task_id, ()))
        if not ids:
            # Still issue the DELETE to stay consistent with any rows the
            # store might not have observed yet (belt-and-suspenders for
            # the unlikely race of an in-flight concurrent writer). This
            # costs nothing on the common path.
            cur.execute("DELETE FROM endpoints WHERE task_id = ?", (task_id.to_wire(),))
            return []
        cur.execute("DELETE FROM endpoints WHERE task_id = ?", (task_id.to_wire(),))

        self._commit_index_update(cur, remove_ids=ids)
        return ids

    def remove_by_job_ids(self, cur: Cursor, job_ids: Sequence[JobName]) -> list[str]:
        """Remove all endpoints owned by any of ``job_ids``. Used by cancel_job and prune."""
        if not job_ids:
            return []
        wire_ids = [jid.to_wire() for jid in job_ids]
        with self._lock:
            to_remove: list[str] = []
            for row in self._by_id.values():
                owning_job, _ = row.task_id.require_task()
                if owning_job.to_wire() in wire_ids:
                    to_remove.append(row.endpoint_id)
        placeholders = sql_placeholders(len(wire_ids))
        cur.execute(
            f"DELETE FROM endpoints WHERE job_id IN ({placeholders})",
            tuple(wire_ids),
        )
        if not to_remove:
            return []

        self._commit_index_update(cur, remove_ids=to_remove)
        return to_remove


# ---------------------------------------------------------------------------
# TaskStore
# ---------------------------------------------------------------------------


class TaskStore:
    """Typed read/write operations for task entities.

    Process-scoped: a single instance lives on the ``ControllerDB``. Every
    method takes the open ``Cursor`` as its first argument so
    writes land inside the caller's transaction.
    """

    def __init__(self, endpoints: EndpointStore, jobs: JobStore) -> None:
        self._endpoints = endpoints
        self._jobs = jobs

    # ── Reads ────────────────────────────────────────────────────────

    def get_task(self, cur: Cursor, task_id: JobName) -> TaskSnapshot | None:
        """Load a task + its current attempt + job_config into a snapshot.

        Returns None if the task doesn't exist. Reads the job_config through
        ``JobStore.get_config`` which caches lookups process-wide.
        """
        task_row = cur.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id.to_wire(),)).fetchone()
        if task_row is None:
            return None

        task = TASK_DETAIL_PROJECTION.decode_one([task_row])
        attempt_id = int(task_row["current_attempt_id"])

        # Load the current attempt to get its state and worker_id.
        attempt_row = cur.execute(
            "SELECT state, worker_id FROM task_attempts WHERE task_id = ? AND attempt_id = ?",
            (task_id.to_wire(), attempt_id),
        ).fetchone()
        if attempt_row is None:
            attempt_state = int(task_row["state"])
            worker_id = task_row["current_worker_id"]
            worker_id = str(worker_id) if worker_id is not None else None
        else:
            attempt_state = int(attempt_row["state"])
            worker_id = str(attempt_row["worker_id"]) if attempt_row["worker_id"] is not None else None

        # Fetch job_config via the process-scoped JobStore cache.
        jc = self._jobs.get_config(cur, task.job_id.to_wire())

        has_coscheduling = False
        resources: job_pb2.ResourceSpecProto | None = None
        if jc is not None:
            has_coscheduling = jc.has_coscheduling
            resources = resource_spec_from_scalars(
                jc.resources.cpu_millicores,
                jc.resources.memory_bytes,
                jc.resources.disk_bytes,
                jc.resources.device_json,
            )

        return TaskSnapshot(
            task_id=task_id.to_wire(),
            job_id=task.job_id,
            state=int(task_row["state"]),
            attempt_id=attempt_id,
            attempt_state=attempt_state,
            failure_count=int(task_row["failure_count"]),
            preemption_count=int(task_row["preemption_count"]),
            max_retries_failure=int(task_row["max_retries_failure"]),
            max_retries_preemption=int(task_row["max_retries_preemption"]),
            worker_id=worker_id,
            has_coscheduling=has_coscheduling,
            resources=resources,
        )

    def get_attempt_state(self, cur: Cursor, task_id: str, attempt_id: int) -> int | None:
        """Load the state of a specific attempt. Used for stale-attempt checks."""
        row = cur.execute(
            "SELECT state FROM task_attempts WHERE task_id = ? AND attempt_id = ?",
            (task_id, attempt_id),
        ).fetchone()
        if row is None:
            return None
        return int(row["state"])

    def find_coscheduled_siblings(
        self,
        cur: Cursor,
        job_id: JobName,
        exclude_task_id: JobName,
        has_coscheduling: bool,
    ) -> list[SiblingSnapshot]:
        """Find active siblings in a coscheduled job.

        Returns an empty list when the job has no coscheduling config.
        Active means ASSIGNED, BUILDING, or RUNNING.
        """
        if not has_coscheduling:
            return []
        rows = cur.execute(
            "SELECT t.task_id, t.current_attempt_id, t.max_retries_preemption, "
            "t.current_worker_id AS worker_id "
            "FROM tasks t "
            "WHERE t.job_id = ? AND t.task_id != ? AND t.state IN (?, ?, ?)",
            (
                job_id.to_wire(),
                exclude_task_id.to_wire(),
                job_pb2.TASK_STATE_ASSIGNED,
                job_pb2.TASK_STATE_BUILDING,
                job_pb2.TASK_STATE_RUNNING,
            ),
        ).fetchall()
        return [
            SiblingSnapshot(
                task_id=str(r["task_id"]),
                attempt_id=int(r["current_attempt_id"]),
                max_retries_preemption=int(r["max_retries_preemption"]),
                worker_id=str(r["worker_id"]) if r["worker_id"] is not None else None,
            )
            for r in rows
        ]

    # ── Read-pool helpers ─────────────────────────────────────────────
    # These operate on the ControllerDB read pool, not a write transaction.

    def running_tasks_by_worker(self, cur: Cursor, worker_ids: set[WorkerId]) -> dict[WorkerId, set[JobName]]:
        """Return the set of currently-running task IDs for each worker.

        Uses the denormalized current_worker_id column instead of joining task_attempts.
        """
        if not worker_ids:
            return {}
        placeholders = sql_placeholders(len(worker_ids))
        rows = decoded_rows(
            cur,
            f"SELECT t.current_worker_id AS worker_id, t.task_id FROM tasks t "
            f"WHERE t.current_worker_id IN ({placeholders}) AND t.state IN (?, ?, ?)",
            (*[str(wid) for wid in worker_ids], *ACTIVE_TASK_STATES),
            decoders={"worker_id": decode_worker_id, "task_id": JobName.from_wire},
        )
        running: dict[WorkerId, set[JobName]] = {wid: set() for wid in worker_ids}
        for row in rows:
            running[row.worker_id].add(row.task_id)
        return running

    def timed_out_executing_tasks(self, cur: Cursor, now: Timestamp) -> list[TimedOutTask]:
        """Find executing tasks whose current attempt has exceeded the job's execution timeout.

        Reads the timeout from job_config.timeout_ms. Uses the current attempt's
        started_at_ms so that retried tasks get a fresh timeout budget per attempt.
        """
        now_ms = now.epoch_ms()
        executing_states = tuple(sorted(EXECUTING_TASK_STATES))
        placeholders = sql_placeholders(len(executing_states))
        rows = decoded_rows(
            cur,
            f"SELECT t.task_id, t.current_worker_id AS worker_id, "
            f"ta.started_at_ms AS attempt_started_at_ms, jc.timeout_ms "
            f"FROM tasks t "
            f"JOIN job_config jc ON jc.job_id = t.job_id "
            f"JOIN task_attempts ta ON ta.task_id = t.task_id AND ta.attempt_id = t.current_attempt_id "
            f"WHERE t.state IN ({placeholders}) "
            f"AND jc.timeout_ms IS NOT NULL AND jc.timeout_ms > 0 "
            f"AND ta.started_at_ms IS NOT NULL",
            (*executing_states,),
            decoders={
                "task_id": JobName.from_wire,
                "worker_id": lambda v: WorkerId(v) if v is not None else None,
                "attempt_started_at_ms": int,
                "timeout_ms": int,
            },
        )
        result: list[TimedOutTask] = []
        for row in rows:
            if row.attempt_started_at_ms + row.timeout_ms <= now_ms:
                result.append(TimedOutTask(task_id=row.task_id, worker_id=row.worker_id))
        return result

    def tasks_for_job_with_attempts(self, cur: Cursor, job_id: JobName) -> list:
        """Fetch all tasks for a job with their attempt history."""
        tasks = TASK_DETAIL_PROJECTION.decode(
            cur.execute(
                "SELECT * FROM tasks WHERE job_id = ? ORDER BY task_index, task_id",
                (job_id.to_wire(),),
            ).fetchall(),
        )
        if not tasks:
            return []
        placeholders = sql_placeholders(len(tasks))
        attempts = ATTEMPT_PROJECTION.decode(
            cur.execute(
                f"SELECT * FROM task_attempts WHERE task_id IN ({placeholders}) ORDER BY task_id, attempt_id",
                tuple(t.task_id.to_wire() for t in tasks),
            ).fetchall(),
        )
        return tasks_with_attempts(tasks, attempts)

    def insert_task_profile(
        self, cur: Cursor, task_id: str, profile_data: bytes, captured_at: Timestamp, profile_kind: str = "cpu"
    ) -> None:
        """Insert a captured profile snapshot for a task.

        The DB trigger caps profiles at 10 per (task_id, profile_kind), evicting the oldest automatically.
        """
        cur.execute(
            "INSERT INTO profiles.task_profiles "
            "(task_id, profile_data, captured_at_ms, profile_kind) VALUES (?, ?, ?, ?)",
            (task_id, profile_data, captured_at.epoch_ms(), profile_kind),
        )

    def get_task_profiles(
        self, cur: Cursor, task_id: str, profile_kind: str | None = None
    ) -> list[tuple[bytes, Timestamp, str]]:
        """Return stored profile snapshots for a task, newest first.

        Args:
            task_id: Task wire string.
            profile_kind: If set, filter to this kind (e.g. "cpu", "memory"). Returns all kinds when None.
        """
        if profile_kind is not None:
            query = (
                "SELECT profile_data, captured_at_ms, profile_kind FROM profiles.task_profiles"
                " WHERE task_id = ? AND profile_kind = ? ORDER BY id DESC"
            )
            params: tuple[str, ...] = (task_id, profile_kind)
        else:
            query = (
                "SELECT profile_data, captured_at_ms, profile_kind FROM profiles.task_profiles"
                " WHERE task_id = ? ORDER BY id DESC"
            )
            params = (task_id,)
        rows = decoded_rows(cur, query, params, decoders={"captured_at_ms": decode_timestamp_ms})
        return [(row.profile_data, row.captured_at_ms, row.profile_kind) for row in rows]

    # ── Writes ───────────────────────────────────────────────────────

    def terminate(self, cur: Cursor, t: TaskTermination) -> None:
        """Move a task (and its current attempt) to terminal state consistently.

        Enforces the multi-table invariant: attempt marked terminal, task
        state/error/finished_at updated, worker columns cleared, endpoints
        deleted, worker resources released.
        """
        finished_at_ms = None if t.state in ACTIVE_TASK_STATES or t.state == job_pb2.TASK_STATE_PENDING else t.now_ms

        if t.finalize is not None and t.finalize.attempt_id >= 0:
            self._finalize_attempt(cur, t.finalize)

        # Build the UPDATE tasks statement dynamically based on optional counters.
        if finished_at_ms is not None:
            set_clauses = ["state = ?", "error = ?", "finished_at_ms = COALESCE(finished_at_ms, ?)"]
        else:
            set_clauses = ["state = ?", "error = ?", "finished_at_ms = ?"]
        exit_code = t.finalize.exit_code if t.finalize is not None else None
        params: list[object] = [int(t.state), t.error, finished_at_ms]

        if t.failure_count is not None:
            set_clauses.append("failure_count = ?")
            params.append(t.failure_count)
        if t.preemption_count is not None:
            set_clauses.append("preemption_count = ?")
            params.append(t.preemption_count)
        if exit_code is not None:
            set_clauses.append("exit_code = COALESCE(?, exit_code)")
            params.append(exit_code)

        # Clear worker columns when leaving active state.
        if t.state not in ACTIVE_TASK_STATES:
            set_clauses.append("current_worker_id = NULL")
            set_clauses.append("current_worker_address = NULL")

        params.append(t.task_id)
        cur.execute(
            f"UPDATE tasks SET {', '.join(set_clauses)} WHERE task_id = ?",
            tuple(params),
        )

        self._remove_task_endpoints(cur, t.task_id)

        if t.worker_id is not None and t.resources is not None:
            self._decommit_worker_resources(cur, t.worker_id, t.resources)

    def requeue(self, cur: Cursor, r: TaskRetry) -> None:
        """Terminate the current attempt but reset the task to PENDING.

        The attempt is marked with the given terminal state, but the task
        row reverts to PENDING so the scheduler can create a fresh attempt.
        """
        self._finalize_attempt(cur, r.finalize)

        # Reset task to PENDING, clear worker columns, update counters,
        # and clear finished_at_ms.
        cur.execute(
            "UPDATE tasks SET state = ?, error = NULL, finished_at_ms = NULL, "
            "failure_count = ?, preemption_count = ?, "
            "current_worker_id = NULL, current_worker_address = NULL "
            "WHERE task_id = ?",
            (
                int(job_pb2.TASK_STATE_PENDING),
                r.failure_count,
                r.preemption_count,
                r.task_id,
            ),
        )

        self._remove_task_endpoints(cur, r.task_id)

        if r.worker_id is not None and r.resources is not None:
            self._decommit_worker_resources(cur, r.worker_id, r.resources)

    def update_active(self, cur: Cursor, u: ActiveStateUpdate) -> None:
        """Non-terminal state update (BUILDING, RUNNING).

        Updates both the attempt and task rows. Does not clear worker
        columns since the task remains active.
        """
        cur.execute(
            "UPDATE task_attempts SET state = ?, started_at_ms = COALESCE(started_at_ms, ?), "
            "exit_code = COALESCE(?, exit_code), error = COALESCE(?, error) "
            "WHERE task_id = ? AND attempt_id = ?",
            (
                int(u.state),
                u.started_ms,
                u.exit_code,
                u.error,
                u.task_id,
                u.attempt_id,
            ),
        )

        cur.execute(
            "UPDATE tasks SET state = ?, error = COALESCE(?, error), "
            "exit_code = COALESCE(?, exit_code), "
            "started_at_ms = COALESCE(started_at_ms, ?), finished_at_ms = ?, "
            "failure_count = ?, preemption_count = ? "
            "WHERE task_id = ?",
            (
                int(u.state),
                u.error,
                u.exit_code,
                u.started_ms,
                None,  # finished_at_ms — active tasks are not finished
                u.failure_count,
                u.preemption_count,
                u.task_id,
            ),
        )

    def assign_to_worker(self, cur: Cursor, a: WorkerAssignment) -> None:
        """Create an attempt bound to a worker and mark the task ASSIGNED."""
        cur.execute(
            "INSERT INTO task_attempts(task_id, attempt_id, worker_id, state, created_at_ms) VALUES (?, ?, ?, ?, ?)",
            (a.task_id, a.attempt_id, a.worker_id, int(job_pb2.TASK_STATE_ASSIGNED), a.now_ms),
        )
        cur.execute(
            "UPDATE tasks SET state = ?, current_attempt_id = ?, "
            "current_worker_id = ?, current_worker_address = ?, "
            "started_at_ms = COALESCE(started_at_ms, ?) WHERE task_id = ?",
            (
                int(job_pb2.TASK_STATE_ASSIGNED),
                a.attempt_id,
                a.worker_id,
                a.worker_address,
                a.now_ms,
                a.task_id,
            ),
        )

    def assign_direct(self, cur: Cursor, a: DirectAssignment) -> None:
        """Create an attempt with no backing worker and mark the task ASSIGNED."""
        cur.execute(
            "INSERT INTO task_attempts(task_id, attempt_id, worker_id, state, created_at_ms) VALUES (?, ?, ?, ?, ?)",
            (a.task_id, a.attempt_id, None, int(job_pb2.TASK_STATE_ASSIGNED), a.now_ms),
        )
        cur.execute(
            "UPDATE tasks SET state = ?, current_attempt_id = ?, "
            "started_at_ms = COALESCE(started_at_ms, ?) WHERE task_id = ?",
            (int(job_pb2.TASK_STATE_ASSIGNED), a.attempt_id, a.now_ms, a.task_id),
        )

    def terminate_coscheduled_siblings(
        self,
        cur: Cursor,
        siblings: list[SiblingSnapshot],
        cause_task_id: JobName,
        resources: job_pb2.ResourceSpecProto,
        now_ms: int,
    ) -> KillResult:
        """Terminate coscheduled siblings and decommit their resources.

        Each sibling is marked WORKER_FAILED with exhausted preemption count
        so it will not be retried. Returns tasks that need kill RPCs.
        """
        tasks_to_kill: set[JobName] = set()
        task_kill_workers: dict[JobName, WorkerId] = {}
        error = f"Coscheduled sibling {cause_task_id.to_wire()} failed"

        for sib in siblings:
            self.terminate(
                cur,
                TaskTermination(
                    task_id=sib.task_id,
                    state=job_pb2.TASK_STATE_WORKER_FAILED,
                    now_ms=now_ms,
                    error=error,
                    finalize=AttemptFinalizer.build(
                        sib.task_id, sib.attempt_id, job_pb2.TASK_STATE_WORKER_FAILED, now_ms, error=error
                    ),
                    worker_id=sib.worker_id,
                    resources=resources if sib.worker_id is not None else None,
                    preemption_count=sib.max_retries_preemption + 1,
                ),
            )
            if sib.worker_id is not None:
                task_kill_workers[JobName.from_wire(sib.task_id)] = WorkerId(sib.worker_id)
            tasks_to_kill.add(JobName.from_wire(sib.task_id))

        return KillResult(
            tasks_to_kill=frozenset(tasks_to_kill),
            task_kill_workers=task_kill_workers,
        )

    def insert_resource_usage(
        self,
        cur: Cursor,
        task_id: str,
        attempt_id: int,
        usage: job_pb2.ResourceUsage,
        now_ms: int,
    ) -> None:
        """Write a single task_resource_history row."""
        cur.execute(
            "INSERT INTO task_resource_history"
            "(task_id, attempt_id, cpu_millicores, memory_mb, disk_mb, memory_peak_mb, timestamp_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                task_id,
                attempt_id,
                usage.cpu_millicores,
                usage.memory_mb,
                usage.disk_mb,
                usage.memory_peak_mb,
                now_ms,
            ),
        )

    def insert_resource_usage_batch(self, cur: Cursor, params: list[tuple]) -> None:
        """Batch insert task_resource_history via executemany for steady-state updates."""
        cur.executemany(
            "INSERT INTO task_resource_history"
            "(task_id, attempt_id, cpu_millicores, memory_mb, disk_mb, memory_peak_mb, timestamp_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            params,
        )

    # ── Internal helpers ─────────────────────────────────────────────

    def _finalize_attempt(self, cur: Cursor, fin: AttemptFinalizer) -> None:
        """Write terminal state to a task_attempts row."""
        cur.execute(
            "UPDATE task_attempts SET state = ?, "
            "finished_at_ms = COALESCE(finished_at_ms, ?), error = ?, "
            "exit_code = COALESCE(?, exit_code) "
            "WHERE task_id = ? AND attempt_id = ?",
            (int(fin.attempt_state), fin.now_ms, fin.error, fin.exit_code, fin.task_id, fin.attempt_id),
        )

    def _remove_task_endpoints(self, cur: Cursor, task_id: str) -> None:
        """Remove all registered endpoints for a task."""
        self._endpoints.remove_by_task(cur, JobName.from_wire(task_id))

    def _decommit_worker_resources(self, cur: Cursor, worker_id: str, resources: job_pb2.ResourceSpecProto) -> None:
        """Subtract a task's resource reservation from a worker, flooring at zero."""
        cur.execute(
            "UPDATE workers SET committed_cpu_millicores = MAX(0, committed_cpu_millicores - ?), "
            "committed_mem_bytes = MAX(0, committed_mem_bytes - ?), "
            "committed_gpu = MAX(0, committed_gpu - ?), committed_tpu = MAX(0, committed_tpu - ?) "
            "WHERE worker_id = ?",
            (
                int(resources.cpu_millicores),
                int(resources.memory_bytes),
                int(get_gpu_count(resources.device)),
                int(get_tpu_count(resources.device)),
                worker_id,
            ),
        )

    # ── Extended reads (migrated from transitions.py inline SQL) ─────

    def get_for_assignment(self, cur: Cursor, task_id: str) -> sqlite3.Row | None:
        """Fetch task detail row for assignment validation."""
        return cur.execute(
            f"SELECT {TASK_DETAIL_SELECT_T} FROM tasks t WHERE t.task_id = ?",
            (task_id,),
        ).fetchone()

    def query(
        self, cur: Cursor, flt: TaskFilter, *, projection: TaskProjection = TaskProjection.DETAIL
    ) -> list[TaskDetailRow]:
        """Return rows matching ``flt`` under the requested projection.

        Builds SQL from ``flt`` by AND-ing every set field. Large
        ``task_ids`` / ``job_ids`` lists are chunked under SQLite's
        host-parameter cap (~999); chunk results are concatenated preserving
        ORDER BY task_id ASC within each chunk.

        ``projection=DETAIL`` (default) returns plain task columns.
        ``WITH_JOB`` joins ``jobs`` and populates ``is_reservation_holder``/``num_tasks``.
        ``WITH_JOB_CONFIG`` additionally joins ``job_config`` and populates resource/timeout fields.
        """
        return self._run_query(cur, flt, projection=projection)

    def _run_query(self, cur: Cursor, flt: TaskFilter, *, projection: TaskProjection) -> list:
        # Short-circuit empty IN-lists: SQLite forbids the empty literal.
        if flt.task_ids is not None and not flt.task_ids:
            return []
        if flt.job_ids is not None and not flt.job_ids:
            return []

        task_chunks = chunk_ids(flt.task_ids)
        job_chunks = chunk_ids(flt.job_ids)
        if task_chunks is None and job_chunks is None:
            return self._query_chunk(
                cur, flt, chunk_task_ids=None, chunk_job_ids=None, projection=projection, limit=flt.limit
            )

        # Pair each chunk with None for the other field; task_ids takes precedence
        # when both are set (callers pre-validate mutual exclusion).
        id_chunks: list[tuple[tuple[str, ...] | None, tuple[str, ...] | None]]
        if task_chunks is not None:
            id_chunks = [(c, None) for c in task_chunks]
        else:
            assert job_chunks is not None
            id_chunks = [(None, c) for c in job_chunks]

        results: list = []
        remaining_limit = flt.limit
        for chunk_task_ids, chunk_job_ids in id_chunks:
            if remaining_limit is not None and remaining_limit <= 0:
                break
            chunk_rows = self._query_chunk(
                cur,
                flt,
                chunk_task_ids=chunk_task_ids,
                chunk_job_ids=chunk_job_ids,
                projection=projection,
                limit=remaining_limit,
            )
            results.extend(chunk_rows)
            if remaining_limit is not None:
                remaining_limit -= len(chunk_rows)
        return results

    def _query_chunk(
        self,
        cur: Cursor,
        flt: TaskFilter,
        *,
        chunk_task_ids: tuple[str, ...] | None,
        chunk_job_ids: tuple[str, ...] | None,
        projection: TaskProjection,
        limit: int | None,
    ) -> list:
        if projection == TaskProjection.DETAIL:
            sql_parts = [f"SELECT {TASK_DETAIL_SELECT_T} FROM tasks t"]
        elif projection == TaskProjection.WITH_JOB:
            sql_parts = [
                "SELECT t.task_id, t.job_id, t.state, t.current_attempt_id, "
                "t.failure_count, t.preemption_count, t.max_retries_failure, t.max_retries_preemption, "
                "t.submitted_at_ms, t.priority_band, t.error, t.exit_code, "
                "t.started_at_ms, t.finished_at_ms, t.current_worker_id, t.current_worker_address, "
                "t.container_id, j.is_reservation_holder, j.num_tasks "
                "FROM tasks t"
            ]
            sql_parts.append("JOIN jobs j ON j.job_id = t.job_id")
        else:  # with_job_config
            sql_parts = [
                "SELECT t.task_id, t.job_id, t.state, t.current_attempt_id, "
                "t.failure_count, t.preemption_count, t.max_retries_failure, t.max_retries_preemption, "
                "t.submitted_at_ms, t.priority_band, t.error, t.exit_code, "
                "t.started_at_ms, t.finished_at_ms, t.current_worker_id, t.current_worker_address, "
                "t.container_id, j.is_reservation_holder, j.num_tasks, "
                "jc.res_cpu_millicores, jc.res_memory_bytes, jc.res_disk_bytes, jc.res_device_json, "
                "jc.has_coscheduling, jc.timeout_ms "
                "FROM tasks t"
            ]
            sql_parts.append("JOIN jobs j ON j.job_id = t.job_id")
            sql_parts.append(JOB_CONFIG_JOIN)

        wb = WhereBuilder()
        wb.in_("t.task_id", chunk_task_ids)
        wb.in_("t.job_id", chunk_job_ids)
        if flt.worker_id is not None:
            wb.eq("t.current_worker_id", str(flt.worker_id))
        if flt.worker_is_null:
            wb.is_null("t.current_worker_id")
        if flt.states is not None:
            wb.in_("t.state", tuple(sorted(flt.states)))

        where_sql, where_params = wb.build()
        params: list[object] = list(where_params)
        if where_sql:
            sql_parts.append(where_sql)
        sql_parts.append("ORDER BY t.task_id ASC")
        if limit is not None:
            sql_parts.append("LIMIT ?")
            params.append(limit)

        rows = cur.execute(" ".join(sql_parts), tuple(params)).fetchall()
        return TASK_DETAIL_PROJECTION.decode(rows)

    def update_container_id(self, cur: Cursor, task_id: str, container_id: str) -> None:
        """Set container_id on a task row."""
        cur.execute(
            "UPDATE tasks SET container_id = ? WHERE task_id = ?",
            (container_id, task_id),
        )

    def get_job_id(self, cur: Cursor, task_id: str) -> str | None:
        """Read job_id for a task. Returns None if the task does not exist."""
        row = cur.execute("SELECT job_id FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
        if row is None:
            return None
        return str(row["job_id"])

    def get_attempt_worker(self, cur: Cursor, task_id: str, attempt_id: int) -> str | None:
        """Worker_id from a specific attempt. Returns None if missing."""
        row = cur.execute(
            "SELECT worker_id FROM task_attempts WHERE task_id = ? AND attempt_id = ?",
            (task_id, attempt_id),
        ).fetchone()
        if row is None or row["worker_id"] is None:
            return None
        return str(row["worker_id"])

    def insert_task(self, cur: Cursor, req: TaskInsert) -> None:
        """Insert a single task row with all priority columns."""
        cur.execute(
            "INSERT INTO tasks("
            "task_id, job_id, task_index, state, error, exit_code, submitted_at_ms, started_at_ms, "
            "finished_at_ms, max_retries_failure, max_retries_preemption, failure_count, preemption_count, "
            "current_attempt_id, priority_neg_depth, priority_root_submitted_ms, "
            "priority_insertion, priority_band"
            ") VALUES (?, ?, ?, ?, NULL, NULL, ?, NULL, NULL, ?, ?, 0, 0, -1, ?, ?, ?, ?)",
            (
                req.task_id,
                req.job_id,
                req.task_index,
                req.state,
                req.submitted_at_ms,
                req.max_retries_failure,
                req.max_retries_preemption,
                req.priority_neg_depth,
                req.priority_root_submitted_ms,
                req.priority_insertion,
                req.priority_band,
            ),
        )

    def delete_attempt(self, cur: Cursor, task_id: str, attempt_id: int) -> None:
        """Delete a task attempt row (used for reservation holder reset)."""
        cur.execute(
            "DELETE FROM task_attempts WHERE task_id = ? AND attempt_id = ?",
            (task_id, attempt_id),
        )

    def reset_reservation_holder(self, cur: Cursor, task_id: str, state: int) -> None:
        """Reset a reservation holder task to pristine PENDING state."""
        cur.execute(
            "UPDATE tasks SET state = ?, current_attempt_id = -1, started_at_ms = NULL, "
            "finished_at_ms = NULL, error = NULL, preemption_count = 0, "
            "current_worker_id = NULL, current_worker_address = NULL WHERE task_id = ?",
            (state, task_id),
        )

    def bulk_cancel(self, cur: Cursor, job_ids: list[str], reason: str, now_ms: int) -> None:
        """Bulk UPDATE tasks to KILLED across multiple job IDs.

        Skips tasks already in terminal states. Clears worker columns.
        """
        placeholders = sql_placeholders(len(job_ids))
        task_terminal_placeholders = sql_placeholders(len(TERMINAL_TASK_STATES))
        cur.execute(
            f"UPDATE tasks SET state = ?, error = ?, finished_at_ms = COALESCE(finished_at_ms, ?), "
            f"current_worker_id = NULL, current_worker_address = NULL "
            f"WHERE job_id IN ({placeholders}) AND state NOT IN ({task_terminal_placeholders})",
            (
                job_pb2.TASK_STATE_KILLED,
                reason,
                now_ms,
                *job_ids,
                *TERMINAL_TASK_STATES,
            ),
        )

    def get_pending_for_direct_provider(self, cur: Cursor, limit: int) -> list:
        """Pending tasks for direct provider promotion (non-reservation-holder only)."""
        return cur.execute(
            "SELECT t.task_id, t.job_id, t.current_attempt_id, j.num_tasks, j.is_reservation_holder, "
            "jc.res_cpu_millicores, jc.res_memory_bytes, jc.res_disk_bytes, jc.res_device_json, "
            "jc.entrypoint_json, jc.environment_json, jc.bundle_id, jc.ports_json, "
            "jc.constraints_json, jc.task_image, jc.timeout_ms "
            f"FROM tasks t JOIN jobs j ON j.job_id = t.job_id {JOB_CONFIG_JOIN} "
            "WHERE t.state = ? AND j.is_reservation_holder = 0 "
            "LIMIT ?",
            (job_pb2.TASK_STATE_PENDING, limit),
        ).fetchall()

    def prune_task_resource_history(self, cur: Cursor, retention: int) -> int:
        """Logarithmic downsampling: when a (task, attempt) exceeds 2*retention rows,
        thin the older half by deleting every other row.

        Over repeated compaction cycles older data becomes exponentially sparser,
        preserving long-term trends while bounding total row count.
        """
        threshold = retention * 2
        overflows = cur.execute(
            "SELECT task_id, attempt_id, COUNT(*) as cnt "
            "FROM task_resource_history "
            "GROUP BY task_id, attempt_id HAVING cnt > ?",
            (threshold,),
        ).fetchall()
        ids_to_delete: list[int] = []
        for row in overflows:
            tid, aid = row["task_id"], row["attempt_id"]
            all_ids = [
                r["id"]
                for r in cur.execute(
                    "SELECT id FROM task_resource_history WHERE task_id = ? AND attempt_id = ? ORDER BY id ASC",
                    (tid, aid),
                ).fetchall()
            ]
            older = all_ids[: len(all_ids) - retention]
            ids_to_delete.extend(older[1::2])

        total_deleted = 0
        for chunk_start in range(0, len(ids_to_delete), 900):
            chunk = ids_to_delete[chunk_start : chunk_start + 900]
            ph = sql_placeholders(len(chunk))
            cur.execute(f"DELETE FROM task_resource_history WHERE id IN ({ph})", tuple(chunk))
            total_deleted += cur.rowcount
        if total_deleted > 0:
            logger.info("Pruned %d task_resource_history rows (log downsampling)", total_deleted)
        return total_deleted


# ---------------------------------------------------------------------------
# Pure job-state derivation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class JobContext:
    """Read-only snapshot of a job row for state derivation."""

    state: int
    started_at_ms: int | None
    max_task_failures: int


def derive_job_state(
    current: int,
    counts: dict[int, int],
    max_task_failures: int,
    has_started: bool,
) -> int:
    """Pure function: derive job state from task state counts.

    Priority ordering:
    SUCCEEDED (all tasks) > FAILED (exceeded budget) > UNSCHEDULABLE > KILLED >
    WORKER_FAILED/PREEMPTED (all terminal) > RUNNING > PENDING.

    Returns the current state unchanged when no transition is warranted.
    """
    total = sum(counts.values())

    if total > 0 and counts.get(job_pb2.TASK_STATE_SUCCEEDED, 0) == total:
        return job_pb2.JOB_STATE_SUCCEEDED
    if counts.get(job_pb2.TASK_STATE_FAILED, 0) > max_task_failures:
        return job_pb2.JOB_STATE_FAILED
    if counts.get(job_pb2.TASK_STATE_UNSCHEDULABLE, 0) > 0:
        return job_pb2.JOB_STATE_UNSCHEDULABLE
    if counts.get(job_pb2.TASK_STATE_KILLED, 0) > 0:
        return job_pb2.JOB_STATE_KILLED
    if (
        total > 0
        and (counts.get(job_pb2.TASK_STATE_WORKER_FAILED, 0) + counts.get(job_pb2.TASK_STATE_PREEMPTED, 0)) > 0
        and all(s in TERMINAL_TASK_STATES for s in counts)
    ):
        return job_pb2.JOB_STATE_WORKER_FAILED
    if (
        counts.get(job_pb2.TASK_STATE_ASSIGNED, 0) > 0
        or counts.get(job_pb2.TASK_STATE_BUILDING, 0) > 0
        or counts.get(job_pb2.TASK_STATE_RUNNING, 0) > 0
    ):
        return job_pb2.JOB_STATE_RUNNING
    if has_started:
        # Retries put tasks back into PENDING; keep job running once it has started.
        return job_pb2.JOB_STATE_RUNNING
    if total > 0:
        return job_pb2.JOB_STATE_PENDING

    return current


# ---------------------------------------------------------------------------
# JobStore
# ---------------------------------------------------------------------------


class JobStore:
    """Typed read/write operations for job entities.

    Process-scoped: a single instance lives on the ``ControllerDB``. Every
    method takes the open ``Cursor`` as its first argument.

    Owns a process-scoped cache of ``job_config`` rows. The cache is
    populated on ``insert_job_config`` and invalidated on ``delete_job``
    via cursor post-commit hooks — memory only diverges from disk on
    successful commit. The cache lets hot scheduling paths read
    resource/coscheduling config without re-hitting SQLite.
    """

    def __init__(self, endpoints: EndpointStore) -> None:
        self._endpoints = endpoints
        self._job_config_cache: dict[str, JobConfigRow] = {}
        self._job_config_lock = Lock()

    # ── Reads ────────────────────────────────────────────────────────

    def get_config(self, cur: Cursor, job_id_wire: str) -> JobConfigRow | None:
        """Fetch a job_config row by job_id, caching process-wide.

        Cache is populated here on hit, on ``insert_job_config``, and invalidated
        on ``delete_job`` (FK cascade deletes the config row). Misses are not
        cached so a later insert is observed immediately.
        """
        with self._job_config_lock:
            cached = self._job_config_cache.get(job_id_wire)
            if cached is not None:
                return cached
        row = cur.execute(
            f"SELECT {JOB_CONFIG_PROJECTION.select_clause(prefix=False)} FROM job_config WHERE job_id = ?",
            (job_id_wire,),
        ).fetchone()
        if row is None:
            return None
        value = JOB_CONFIG_PROJECTION.decode_one([row])
        assert value is not None
        with self._job_config_lock:
            self._job_config_cache.setdefault(job_id_wire, value)
            return self._job_config_cache[job_id_wire]

    def get_state(self, cur: Cursor, job_id: JobName) -> int | None:
        """Read current job state."""
        row = cur.execute(
            "SELECT state FROM jobs WHERE job_id = ?",
            (job_id.to_wire(),),
        ).fetchone()
        if row is None:
            return None
        return int(row["state"])

    def get_preemption_policy(self, cur: Cursor, job_id: JobName) -> int:
        """Resolve the effective preemption policy for a job.

        Defaults: single-task jobs use TERMINATE_CHILDREN, multi-task use
        PRESERVE_CHILDREN.
        """
        row = cur.execute(
            f"SELECT jc.preemption_policy, j.num_tasks FROM jobs j {JOB_CONFIG_JOIN} WHERE j.job_id = ?",
            (job_id.to_wire(),),
        ).fetchone()
        if row is None:
            return job_pb2.JOB_PREEMPTION_POLICY_TERMINATE_CHILDREN
        policy = int(row["preemption_policy"])
        if policy != job_pb2.JOB_PREEMPTION_POLICY_UNSPECIFIED:
            return policy
        if int(row["num_tasks"]) <= 1:
            return job_pb2.JOB_PREEMPTION_POLICY_TERMINATE_CHILDREN
        return job_pb2.JOB_PREEMPTION_POLICY_PRESERVE_CHILDREN

    # ── Writes ───────────────────────────────────────────────────────

    def update_state(
        self,
        cur: Cursor,
        job_id: JobName,
        state: JobState,
        now_ms: int,
        error: str | None = None,
    ) -> None:
        """Direct job state update with COALESCE patterns for timestamps and error."""
        terminal_placeholders = sql_placeholders(len(TERMINAL_JOB_STATES))
        cur.execute(
            "UPDATE jobs SET state = ?, "
            "started_at_ms = CASE WHEN ? = ? THEN COALESCE(started_at_ms, ?) ELSE started_at_ms END, "
            f"finished_at_ms = CASE WHEN ? IN ({terminal_placeholders}) THEN ? ELSE finished_at_ms END, "
            "error = CASE WHEN ? IN (?, ?, ?, ?) THEN ? ELSE error END "
            "WHERE job_id = ?",
            (
                state,
                state,
                job_pb2.JOB_STATE_RUNNING,
                now_ms,
                state,
                *TERMINAL_JOB_STATES,
                now_ms,
                state,
                job_pb2.JOB_STATE_FAILED,
                job_pb2.JOB_STATE_KILLED,
                job_pb2.JOB_STATE_UNSCHEDULABLE,
                job_pb2.JOB_STATE_WORKER_FAILED,
                error,
                job_id.to_wire(),
            ),
        )

    def get_job_context(self, cur: Cursor, job_id: JobName) -> JobContext | None:
        """Read-only: fetch current state, started_at_ms, and max_task_failures.

        Returns None if the job doesn't exist.
        """
        row = cur.execute(
            f"SELECT j.state, j.started_at_ms, jc.max_task_failures "
            f"FROM jobs j {JOB_CONFIG_JOIN} WHERE j.job_id = ?",
            (job_id.to_wire(),),
        ).fetchone()
        if row is None:
            return None
        return JobContext(
            state=int(row["state"]),
            started_at_ms=int(row["started_at_ms"]) if row["started_at_ms"] is not None else None,
            max_task_failures=int(row["max_task_failures"]),
        )

    def get_task_state_counts(self, cur: Cursor, job_id: JobName) -> dict[int, int]:
        """Read-only: GROUP BY state count query for a job's tasks."""
        rows = cur.execute(
            "SELECT state, COUNT(*) AS c FROM tasks WHERE job_id = ? GROUP BY state",
            (job_id.to_wire(),),
        ).fetchall()
        return {int(r["state"]): int(r["c"]) for r in rows}

    def get_first_task_error(self, cur: Cursor, job_id: JobName) -> str | None:
        """Read-only: fetch the error from the first failing task by task_index."""
        row = cur.execute(
            "SELECT error FROM tasks WHERE job_id = ? AND error IS NOT NULL ORDER BY task_index LIMIT 1",
            (job_id.to_wire(),),
        ).fetchone()
        if row is None:
            return None
        return str(row["error"])

    def recompute_state(self, cur: Cursor, job_id: JobName) -> int | None:
        """Derive job state from task state counts and update the row.

        Uses the pure derive_job_state function for the decision logic,
        then writes the result if the state changed.
        """
        ctx = self.get_job_context(cur, job_id)
        if ctx is None:
            return None
        if ctx.state in TERMINAL_JOB_STATES:
            return ctx.state

        counts = self.get_task_state_counts(cur, job_id)
        new_state = derive_job_state(
            current=ctx.state,
            counts=counts,
            max_task_failures=ctx.max_task_failures,
            has_started=ctx.started_at_ms is not None,
        )

        if new_state == ctx.state:
            return new_state

        error = self.get_first_task_error(cur, job_id)
        now_ms = Timestamp.now().epoch_ms()
        self.update_state(cur, job_id, new_state, now_ms, error)
        return new_state

    def kill_non_terminal_tasks(
        self,
        cur: Cursor,
        tasks: TaskStore,
        job_id: str,
        reason: str,
        now_ms: int,
    ) -> KillResult:
        """Kill all non-terminal tasks for a job, decommit resources, and delete endpoints."""
        terminal_states = tuple(sorted(TERMINAL_TASK_STATES))
        placeholders = sql_placeholders(len(terminal_states))
        rows = cur.execute(
            "SELECT t.task_id, t.current_attempt_id, t.current_worker_id, "
            "jc.res_cpu_millicores, jc.res_memory_bytes, jc.res_disk_bytes, jc.res_device_json "
            "FROM tasks t "
            "JOIN jobs j ON j.job_id = t.job_id "
            f"{JOB_CONFIG_JOIN} "
            f"WHERE t.job_id = ? AND t.state NOT IN ({placeholders})",
            (job_id, *terminal_states),
        ).fetchall()

        tasks_to_kill: set[JobName] = set()
        task_kill_workers: dict[JobName, WorkerId] = {}

        for row in rows:
            task_id = str(row["task_id"])
            worker_id = row["current_worker_id"]
            task_name = JobName.from_wire(task_id)
            resources = None
            if worker_id is not None:
                resources = resource_spec_from_scalars(
                    int(row["res_cpu_millicores"]),
                    int(row["res_memory_bytes"]),
                    int(row["res_disk_bytes"]),
                    row["res_device_json"],
                )
                task_kill_workers[task_name] = WorkerId(str(worker_id))
            attempt_id = int(row["current_attempt_id"])
            tasks.terminate(
                cur,
                TaskTermination(
                    task_id=task_id,
                    state=job_pb2.TASK_STATE_KILLED,
                    now_ms=now_ms,
                    error=reason,
                    finalize=(
                        AttemptFinalizer.build(task_id, attempt_id, job_pb2.TASK_STATE_KILLED, now_ms, error=reason)
                        if attempt_id >= 0
                        else None
                    ),
                    worker_id=str(worker_id) if worker_id is not None else None,
                    resources=resources,
                ),
            )
            tasks_to_kill.add(task_name)

        return KillResult(tasks_to_kill=frozenset(tasks_to_kill), task_kill_workers=task_kill_workers)

    def cascade_children(
        self,
        cur: Cursor,
        tasks: TaskStore,
        job_id: JobName,
        reason: str,
        now_ms: int,
        *,
        exclude_reservation_holders: bool = False,
    ) -> KillResult:
        """Kill descendant jobs (not the job itself) when a parent reaches terminal state.

        When exclude_reservation_holders is True, reservation holder jobs and their
        descendants are left alive.  Used during preemption retry so the parent's
        reservation survives for re-scheduling.
        """
        tasks_to_kill: set[JobName] = set()
        task_kill_workers: dict[JobName, WorkerId] = {}

        if exclude_reservation_holders:
            descendants = cur.execute(
                "WITH RECURSIVE subtree(job_id) AS ("
                "  SELECT job_id FROM jobs WHERE parent_job_id = ? AND is_reservation_holder = 0 "
                "  UNION ALL "
                "  SELECT j.job_id FROM jobs j JOIN subtree s ON j.parent_job_id = s.job_id"
                "   WHERE j.is_reservation_holder = 0"
                ") SELECT job_id FROM subtree",
                (job_id.to_wire(),),
            ).fetchall()
        else:
            descendants = cur.execute(
                "WITH RECURSIVE subtree(job_id) AS ("
                "  SELECT job_id FROM jobs WHERE parent_job_id = ? "
                "  UNION ALL "
                "  SELECT j.job_id FROM jobs j JOIN subtree s ON j.parent_job_id = s.job_id"
                ") SELECT job_id FROM subtree",
                (job_id.to_wire(),),
            ).fetchall()

        for child_row in descendants:
            child_job_id = str(child_row["job_id"])
            child_result = self.kill_non_terminal_tasks(cur, tasks, child_job_id, reason, now_ms)
            tasks_to_kill.update(child_result.tasks_to_kill)
            task_kill_workers.update(child_result.task_kill_workers)

            terminal_placeholders = sql_placeholders(len(TERMINAL_JOB_STATES))
            cur.execute(
                "UPDATE jobs SET state = ?, error = ?, finished_at_ms = COALESCE(finished_at_ms, ?) "
                f"WHERE job_id = ? AND state NOT IN ({terminal_placeholders})",
                (
                    job_pb2.JOB_STATE_KILLED,
                    reason,
                    now_ms,
                    child_job_id,
                    *TERMINAL_JOB_STATES,
                ),
            )

        return KillResult(tasks_to_kill=frozenset(tasks_to_kill), task_kill_workers=task_kill_workers)

    # ── Job submission and lifecycle ────────────────────────────────

    def insert_job(self, cur: Cursor, job: JobInsert) -> None:
        """Insert a row into the jobs table."""
        cur.execute(
            "INSERT INTO jobs("
            "job_id, user_id, parent_job_id, root_job_id, depth, state, submitted_at_ms, "
            "root_submitted_at_ms, started_at_ms, finished_at_ms, scheduling_deadline_epoch_ms, "
            "error, exit_code, num_tasks, is_reservation_holder, name, has_reservation"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, NULL, ?, ?, ?, ?)",
            (
                job.job_id,
                job.user_id,
                job.parent_job_id,
                job.root_job_id,
                job.depth,
                job.state,
                job.submitted_at_ms,
                job.root_submitted_at_ms,
                job.finished_at_ms,
                job.scheduling_deadline_epoch_ms,
                job.error,
                job.num_tasks,
                1 if job.is_reservation_holder else 0,
                job.name,
                1 if job.has_reservation else 0,
            ),
        )

    def insert_job_config(self, cur: Cursor, cfg: JobConfigInsert) -> None:
        """Insert a row into the job_config table and cache it on commit."""
        cur.execute(
            "INSERT INTO job_config("
            "job_id, name, has_reservation, "
            "res_cpu_millicores, res_memory_bytes, res_disk_bytes, res_device_json, "
            "constraints_json, has_coscheduling, coscheduling_group_by, "
            "scheduling_timeout_ms, max_task_failures, "
            "entrypoint_json, environment_json, bundle_id, ports_json, "
            "max_retries_failure, max_retries_preemption, timeout_ms, "
            "preemption_policy, existing_job_policy, priority_band, "
            "task_image, submit_argv_json, reservation_json, fail_if_exists"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                cfg.job_id,
                cfg.name,
                1 if cfg.has_reservation else 0,
                cfg.resources.cpu_millicores,
                cfg.resources.memory_bytes,
                cfg.resources.disk_bytes,
                cfg.resources.device_json,
                cfg.constraints_json,
                cfg.has_coscheduling,
                cfg.coscheduling_group_by,
                cfg.scheduling_timeout_ms,
                cfg.max_task_failures,
                cfg.entrypoint_json,
                cfg.environment_json,
                cfg.bundle_id,
                cfg.ports_json,
                cfg.max_retries_failure,
                cfg.max_retries_preemption,
                cfg.timeout_ms,
                cfg.preemption_policy,
                cfg.existing_job_policy,
                cfg.priority_band,
                cfg.task_image,
                cfg.submit_argv_json,
                cfg.reservation_json,
                cfg.fail_if_exists,
            ),
        )
        # Populate the process-scoped cache on commit, constructing the typed
        # row directly from the insert payload so we do not need a re-read.
        cached = JobConfigRow(
            job_id=JobName.from_wire(cfg.job_id),
            name=cfg.name,
            has_reservation=bool(cfg.has_reservation),
            resources=cfg.resources,
            constraints_json=cfg.constraints_json,
            has_coscheduling=bool(cfg.has_coscheduling),
            coscheduling_group_by=cfg.coscheduling_group_by,
            scheduling_timeout_ms=cfg.scheduling_timeout_ms,
            max_task_failures=cfg.max_task_failures,
            entrypoint_json=cfg.entrypoint_json,
            environment_json=cfg.environment_json,
            bundle_id=cfg.bundle_id,
            ports_json=cfg.ports_json,
            max_retries_failure=cfg.max_retries_failure,
            max_retries_preemption=cfg.max_retries_preemption,
            timeout_ms=cfg.timeout_ms,
            preemption_policy=cfg.preemption_policy,
            existing_job_policy=cfg.existing_job_policy,
            priority_band=cfg.priority_band,
            task_image=cfg.task_image,
            submit_argv_json=cfg.submit_argv_json,
            reservation_json=cfg.reservation_json,
            fail_if_exists=bool(cfg.fail_if_exists),
        )

        def apply() -> None:
            with self._job_config_lock:
                self._job_config_cache[cfg.job_id] = cached

        cur.on_commit(apply)

    def insert_workdir_files(self, cur: Cursor, job_id: str, files: list[tuple[str, bytes]]) -> None:
        """Insert workdir file entries for a job."""
        for filename, data in files:
            cur.execute(
                "INSERT INTO job_workdir_files(job_id, filename, data) VALUES (?, ?, ?)",
                (job_id, filename, data),
            )

    def get_workdir_files(self, cur: Cursor, job_id: str) -> dict[str, bytes]:
        """Fetch workdir files for a job, keyed by filename."""
        rows = cur.execute(
            "SELECT filename, data FROM job_workdir_files WHERE job_id = ?",
            (job_id,),
        ).fetchall()
        return {str(row["filename"]): bytes(row["data"]) for row in rows}

    def exists(self, cur: Cursor, job_id: str) -> bool:
        """Check whether a job row exists."""
        row = cur.execute("SELECT 1 FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return row is not None

    def get_root_submitted_ms(self, cur: Cursor, parent_job_id: str) -> int | None:
        """Read root_submitted_at_ms for a parent job. Returns None if not found."""
        row = cur.execute(
            "SELECT root_submitted_at_ms FROM jobs WHERE job_id = ?",
            (parent_job_id,),
        ).fetchone()
        if row is None:
            return None
        return int(row["root_submitted_at_ms"])

    def get_parent_band(self, cur: Cursor, parent_job_id: str) -> int | None:
        """Read priority_band from the parent's first task. Returns None if not found."""
        row = cur.execute(
            "SELECT priority_band FROM tasks WHERE job_id = ? LIMIT 1",
            (parent_job_id,),
        ).fetchone()
        if row is None:
            return None
        return int(row["priority_band"])

    def get_subtree_ids(self, cur: Cursor, job_id: str) -> list[str]:
        """Recursive CTE returning all job IDs in the subtree rooted at job_id (inclusive)."""
        rows = cur.execute(
            "WITH RECURSIVE subtree(job_id) AS ("
            "  SELECT job_id FROM jobs WHERE job_id = ? "
            "  UNION ALL "
            "  SELECT j.job_id FROM jobs j JOIN subtree s ON j.parent_job_id = s.job_id"
            ") SELECT job_id FROM subtree",
            (job_id,),
        ).fetchall()
        return [str(row["job_id"]) for row in rows]

    def bulk_cancel(self, cur: Cursor, job_ids: list[str], reason: str, now_ms: int) -> None:
        """Bulk UPDATE jobs to KILLED, skipping already-terminal jobs.

        Deliberately excludes JOB_STATE_WORKER_FAILED from the guard set so
        worker-failed jobs can still be cancelled.
        """
        if not job_ids:
            return
        placeholders = sql_placeholders(len(job_ids))
        cancel_guard_states = TERMINAL_JOB_STATES - {job_pb2.JOB_STATE_WORKER_FAILED}
        guard_placeholders = sql_placeholders(len(cancel_guard_states))
        cur.execute(
            f"UPDATE jobs SET state = ?, error = ?, finished_at_ms = COALESCE(finished_at_ms, ?) "
            f"WHERE job_id IN ({placeholders}) AND state NOT IN ({guard_placeholders})",
            (
                job_pb2.JOB_STATE_KILLED,
                reason,
                now_ms,
                *job_ids,
                *cancel_guard_states,
            ),
        )

    def start_if_pending(self, cur: Cursor, job_id: str, now_ms: int) -> None:
        """Transition a job from PENDING to RUNNING. No-op if already started."""
        cur.execute(
            "UPDATE jobs SET state = CASE WHEN state = ? THEN ? ELSE state END, "
            "started_at_ms = COALESCE(started_at_ms, ?) WHERE job_id = ?",
            (job_pb2.JOB_STATE_PENDING, job_pb2.JOB_STATE_RUNNING, now_ms, job_id),
        )

    def get_job_detail(self, cur: Cursor, job_id: str) -> JobDetailRow | None:
        """Fetch full job detail with config join for scheduling/dispatch."""
        row = cur.execute(
            f"SELECT {JOB_DETAIL_PROJECTION.select_clause()} " f"FROM jobs j {JOB_CONFIG_JOIN} WHERE j.job_id = ?",
            (job_id,),
        ).fetchone()
        if row is None:
            return None
        return JOB_DETAIL_PROJECTION.decode_one([row])

    @overload
    def query(self, cur: Cursor, flt: JobDetailFilter, *, detail: Literal[False] = ...) -> list[JobRow]: ...

    @overload
    def query(self, cur: Cursor, flt: JobDetailFilter, *, detail: Literal[True]) -> list[JobDetailRow]: ...

    def query(
        self,
        cur: Cursor,
        flt: JobDetailFilter,
        *,
        detail: bool = False,
    ) -> list[JobRow] | list[JobDetailRow]:
        """Query jobs matching ``flt``.

        Executes inside the caller's cursor (read snapshot or write
        transaction). ``detail=True`` selects the full :class:`JobDetailRow`
        projection (with config join); ``False`` selects the lightweight
        :class:`JobRow` projection.

        Large ``job_ids`` lists are chunked to respect SQLite's host-parameter cap.
        """
        if flt.job_ids is not None and not flt.job_ids:
            return []
        projection = JOB_DETAIL_PROJECTION if detail else JOB_ROW_PROJECTION

        def fetch(sql: str, params: tuple[object, ...]) -> list[tuple]:
            return cur.execute(sql, params).fetchall()

        chunks = chunk_ids(flt.job_ids)
        return run_chunked(  # type: ignore[return-value]
            chunks,
            flt.limit,
            lambda chunk, limit: self._query_job_chunk(fetch, flt, projection, chunk_job_ids=chunk, limit=limit),
        )

    def _where_for(self, flt: JobDetailFilter, chunk_job_ids: tuple[str, ...] | None) -> tuple[str, list[object]]:
        """Build the WHERE fragment and params for a JobDetailFilter.

        Returns ``("", [])`` when no predicates are set (no WHERE clause).
        """
        wb = WhereBuilder()
        wb.in_("j.job_id", chunk_job_ids)
        if flt.states is not None:
            wb.in_("j.state", tuple(sorted(flt.states)))
        if flt.has_reservation is not None:
            wb.eq("j.has_reservation", 1 if flt.has_reservation else 0)
        where_sql, where_params = wb.build()
        return where_sql, list(where_params)

    def _query_job_chunk(
        self,
        fetch: Callable[[str, tuple[object, ...]], list[tuple]],
        flt: JobDetailFilter,
        projection: Any,
        *,
        chunk_job_ids: tuple[str, ...] | None,
        limit: int | None = None,
    ) -> list:
        effective_limit = limit if limit is not None else flt.limit
        sql_parts = [f"SELECT {projection.select_clause()} FROM jobs j {JOB_CONFIG_JOIN}"]
        where_clause, params = self._where_for(flt, chunk_job_ids)
        if where_clause:
            sql_parts.append(where_clause)
        if effective_limit is not None:
            sql_parts.append("LIMIT ?")
            params.append(effective_limit)
        rows = fetch(" ".join(sql_parts), tuple(params))
        return projection.decode(rows)

    def delete_job(self, cur: Cursor, job_id: str) -> None:
        """DELETE FROM jobs WHERE job_id = ?. Cascades to tasks, attempts, endpoints, job_config.

        The DELETE cascades via FK to job_config; the in-memory cache entry
        is popped on commit.
        """
        cur.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))

        def apply() -> None:
            with self._job_config_lock:
                self._job_config_cache.pop(job_id, None)

        cur.on_commit(apply)

    def get_reservation_holder_ids(self, cur: Cursor, job_ids: set[str]) -> set[str]:
        """Filter a set of job IDs to those that are reservation holders."""
        if not job_ids:
            return set()
        placeholders = sql_placeholders(len(job_ids))
        rows = cur.execute(
            f"SELECT job_id FROM jobs WHERE job_id IN ({placeholders}) AND is_reservation_holder = 1",
            tuple(job_ids),
        ).fetchall()
        return {str(r["job_id"]) for r in rows}

    def get_finished_jobs_before(self, cur: Cursor, cutoff_ms: int) -> list[str]:
        """Return job_ids of terminal jobs finished before the cutoff, one at a time."""
        terminal_states = tuple(TERMINAL_JOB_STATES)
        placeholders = sql_placeholders(len(terminal_states))
        row = cur.execute(
            f"SELECT job_id FROM jobs WHERE state IN ({placeholders})"
            " AND finished_at_ms IS NOT NULL AND finished_at_ms < ? LIMIT 1",
            (*terminal_states, cutoff_ms),
        ).fetchone()
        if row is None:
            return []
        return [str(row["job_id"])]

    # ---------------------------------------------------------------------------


# WorkerStore
# ---------------------------------------------------------------------------


class WorkerStore:
    """Typed read/write operations for worker entities.

    Process-scoped: a single instance lives on the ``ControllerDB``. Every
    write takes the open ``Cursor`` as its first argument. Owns
    the lazy worker-attributes cache used by the scheduling hot path.
    """

    def __init__(self, endpoints: EndpointStore, dispatch: DispatchStore):
        self._endpoints = endpoints
        self._dispatch = dispatch

    # ── Reads ────────────────────────────────────────────────────────

    def healthy_active_with_attributes(self, cur: Cursor) -> list[WorkerRow]:
        """Fetch all healthy, active workers with their attributes and available resources.

        Both the worker rows and their attributes are read through the caller's
        cursor so the result is coherent with any outer snapshot/transaction.
        """
        workers = WORKER_ROW_PROJECTION.decode(
            cur.execute(
                f"SELECT {WORKER_ROW_PROJECTION.select_clause()} " "FROM workers w WHERE w.healthy = 1 AND w.active = 1"
            ).fetchall(),
        )
        if not workers:
            return []
        worker_ids = tuple(str(w.worker_id) for w in workers)
        placeholders = sql_placeholders(len(worker_ids))
        attr_rows = decoded_rows(
            cur,
            f"SELECT worker_id, key, value_type, str_value, int_value, float_value "
            f"FROM worker_attributes WHERE worker_id IN ({placeholders})",
            worker_ids,
        )
        attrs_by_worker = _decode_attribute_rows(attr_rows)
        return [
            dc_replace(
                w,
                attributes=attrs_by_worker.get(w.worker_id, {}),
                available_cpu_millicores=w.total_cpu_millicores - w.committed_cpu_millicores,
                available_memory=w.total_memory_bytes - w.committed_mem,
                available_gpus=w.total_gpu_count - w.committed_gpu,
                available_tpus=w.total_tpu_count - w.committed_tpu,
            )
            for w in workers
        ]

    def query(self, cur: Cursor, flt: WorkerFilter) -> list[WorkerRow]:
        """Return :class:`WorkerRow` instances matching ``flt``.

        Executes inside the caller's cursor. Large ``worker_ids`` lists are
        chunked under SQLite's host-parameter cap. Worker attributes are NOT
        loaded — use :meth:`healthy_active_with_attributes` when attributes
        are needed.
        """
        if flt.worker_ids is not None and not flt.worker_ids:
            return []
        chunks = chunk_ids(flt.worker_ids)
        return run_chunked(
            chunks,
            limit=None,
            fetch=lambda chunk, _limit: self._query_worker_chunk(cur, flt, chunk_worker_ids=chunk),
        )

    def _query_worker_chunk(
        self,
        cur: Cursor,
        flt: WorkerFilter,
        *,
        chunk_worker_ids: tuple[WorkerId, ...] | None,
    ) -> list[WorkerRow]:
        sql_parts = [f"SELECT {WORKER_ROW_PROJECTION.select_clause()} FROM workers w"]
        wb = WhereBuilder()
        if chunk_worker_ids is not None:
            wb.in_("w.worker_id", tuple(str(wid) for wid in chunk_worker_ids))
        if flt.active is not None:
            wb.eq("w.active", 1 if flt.active else 0)
        if flt.healthy is not None:
            wb.eq("w.healthy", 1 if flt.healthy else 0)
        where_sql, params = wb.build()
        if where_sql:
            sql_parts.append(where_sql)
        rows = cur.execute(" ".join(sql_parts), params).fetchall()
        return WORKER_ROW_PROJECTION.decode(rows)

    # ── Writes ───────────────────────────────────────────────────────

    def update_health_batch(self, cur: Cursor, requests: list[HeartbeatApplyRequest], now_ms: int) -> set[str]:
        """Batch-update worker health, resource snapshots, and history.

        Returns the set of worker IDs that actually exist in the DB so callers
        can skip updates from stale/removed workers.
        """
        worker_ids = [str(req.worker_id) for req in requests]
        if not worker_ids:
            return set()

        placeholders = sql_placeholders(len(worker_ids))
        rows = cur.execute(
            f"SELECT worker_id FROM workers WHERE worker_id IN ({placeholders})",
            tuple(worker_ids),
        ).fetchall()
        existing = {str(r["worker_id"]) for r in rows}

        health_params_no_snap: list[tuple] = []
        health_params_with_snap: list[tuple] = []
        history_params: list[tuple] = []
        for req in requests:
            wid = str(req.worker_id)
            if wid not in existing:
                continue
            snap = req.worker_resource_snapshot
            if snap is not None:
                snap_fields = (
                    snap.host_cpu_percent,
                    snap.memory_used_bytes,
                    snap.memory_total_bytes,
                    snap.disk_used_bytes,
                    snap.disk_total_bytes,
                    snap.running_task_count,
                    snap.total_process_count,
                    snap.net_recv_bps,
                    snap.net_sent_bps,
                )
                health_params_with_snap.append((now_ms, *snap_fields, wid))
                history_params.append((wid, *snap_fields, now_ms))
            else:
                health_params_no_snap.append((now_ms, wid))

        if health_params_no_snap:
            cur.executemany(
                "UPDATE workers SET healthy = 1, active = 1, consecutive_failures = 0, "
                "last_heartbeat_ms = ? WHERE worker_id = ?",
                health_params_no_snap,
            )
        if health_params_with_snap:
            cur.executemany(
                "UPDATE workers SET healthy = 1, active = 1, consecutive_failures = 0, "
                "last_heartbeat_ms = ?, "
                "snapshot_host_cpu_percent = ?, snapshot_memory_used_bytes = ?, "
                "snapshot_memory_total_bytes = ?, snapshot_disk_used_bytes = ?, "
                "snapshot_disk_total_bytes = ?, snapshot_running_task_count = ?, "
                "snapshot_total_process_count = ?, snapshot_net_recv_bps = ?, "
                "snapshot_net_sent_bps = ? WHERE worker_id = ?",
                health_params_with_snap,
            )
        if history_params:
            cur.executemany(
                "INSERT INTO worker_resource_history("
                "worker_id, snapshot_host_cpu_percent, snapshot_memory_used_bytes, "
                "snapshot_memory_total_bytes, snapshot_disk_used_bytes, snapshot_disk_total_bytes, "
                "snapshot_running_task_count, snapshot_total_process_count, "
                "snapshot_net_recv_bps, snapshot_net_sent_bps, timestamp_ms"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                history_params,
            )
        return existing

    def record_heartbeat_failure(self, cur: Cursor, worker_id: WorkerId, failures: int, threshold: int) -> None:
        """Increment consecutive_failures and mark unhealthy if threshold reached.

        The caller is responsible for reading the current failure count and
        computing `failures` (old count + 1) before calling this method.
        """
        cur.execute(
            "UPDATE workers SET consecutive_failures = ?, "
            "healthy = CASE WHEN ? >= ? THEN 0 ELSE healthy END "
            "WHERE worker_id = ?",
            (failures, failures, threshold, str(worker_id)),
        )

    def record_worker_task_history(self, cur: Cursor, worker_id: str, task_id: str, now_ms: int) -> None:
        """Insert a worker_task_history row recording an assignment."""
        cur.execute(
            "INSERT INTO worker_task_history(worker_id, task_id, assigned_at_ms) VALUES (?, ?, ?)",
            (worker_id, task_id, now_ms),
        )

    def remove(self, cur: Cursor, worker_id: str) -> None:
        """Remove a worker and sever all its foreign-key references.

        Nullifies worker_id in task_attempts and tasks, removes dispatch_queue
        entries, and deletes the worker row.
        """
        cur.execute(
            "UPDATE task_attempts SET worker_id = NULL WHERE worker_id = ?",
            (worker_id,),
        )
        cur.execute(
            "UPDATE tasks SET current_worker_id = NULL WHERE current_worker_id = ?",
            (worker_id,),
        )
        self._dispatch.delete_for_worker(cur, worker_id)
        cur.execute("DELETE FROM workers WHERE worker_id = ?", (worker_id,))

    def decommit_resources(self, cur: Cursor, worker_id: str, resources: job_pb2.ResourceSpecProto) -> None:
        """Subtract a task's resource reservation from a worker, flooring at zero."""
        cur.execute(
            "UPDATE workers SET committed_cpu_millicores = MAX(0, committed_cpu_millicores - ?), "
            "committed_mem_bytes = MAX(0, committed_mem_bytes - ?), "
            "committed_gpu = MAX(0, committed_gpu - ?), "
            "committed_tpu = MAX(0, committed_tpu - ?) "
            "WHERE worker_id = ?",
            (
                int(resources.cpu_millicores),
                int(resources.memory_bytes),
                int(get_gpu_count(resources.device)),
                int(get_tpu_count(resources.device)),
                worker_id,
            ),
        )

    def commit_resources(self, cur: Cursor, worker_id: str, resources: job_pb2.ResourceSpecProto) -> None:
        """Add a task's resource reservation to a worker's committed totals."""
        cur.execute(
            "UPDATE workers SET committed_cpu_millicores = committed_cpu_millicores + ?, "
            "committed_mem_bytes = committed_mem_bytes + ?, "
            "committed_gpu = committed_gpu + ?, "
            "committed_tpu = committed_tpu + ? "
            "WHERE worker_id = ?",
            (
                int(resources.cpu_millicores),
                int(resources.memory_bytes),
                int(get_gpu_count(resources.device)),
                int(get_tpu_count(resources.device)),
                worker_id,
            ),
        )

    def upsert(self, cur: Cursor, req: WorkerUpsert) -> None:
        """Insert or update a worker row and replace its attributes.

        Performs the INSERT...ON CONFLICT UPDATE for the workers table,
        then deletes and re-inserts all worker_attributes rows.
        """
        md = req.metadata
        cur.execute(
            "INSERT INTO workers("
            "worker_id, address, healthy, active, consecutive_failures, last_heartbeat_ms, "
            "committed_cpu_millicores, committed_mem_bytes, committed_gpu, committed_tpu, "
            "total_cpu_millicores, total_memory_bytes, total_gpu_count, total_tpu_count, "
            "device_type, device_variant, slice_id, scale_group, "
            "md_hostname, md_ip_address, md_cpu_count, md_memory_bytes, md_disk_bytes, "
            "md_tpu_name, md_tpu_worker_hostnames, md_tpu_worker_id, md_tpu_chips_per_host_bounds, "
            "md_gpu_count, md_gpu_name, md_gpu_memory_mb, "
            "md_gce_instance_name, md_gce_zone, md_git_hash, md_device_json"
            ") VALUES (?, ?, 1, 1, 0, ?, 0, 0, 0, 0, ?, ?, ?, ?, ?, ?, ?, ?, "
            "?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(worker_id) DO UPDATE SET "
            "address=excluded.address, healthy=1, active=1, "
            "consecutive_failures=0, last_heartbeat_ms=excluded.last_heartbeat_ms, "
            "total_cpu_millicores=excluded.total_cpu_millicores, total_memory_bytes=excluded.total_memory_bytes, "
            "total_gpu_count=excluded.total_gpu_count, total_tpu_count=excluded.total_tpu_count, "
            "device_type=excluded.device_type, device_variant=excluded.device_variant, "
            "slice_id=excluded.slice_id, scale_group=excluded.scale_group, "
            "md_hostname=excluded.md_hostname, md_ip_address=excluded.md_ip_address, "
            "md_cpu_count=excluded.md_cpu_count, md_memory_bytes=excluded.md_memory_bytes, "
            "md_disk_bytes=excluded.md_disk_bytes, md_tpu_name=excluded.md_tpu_name, "
            "md_tpu_worker_hostnames=excluded.md_tpu_worker_hostnames, "
            "md_tpu_worker_id=excluded.md_tpu_worker_id, "
            "md_tpu_chips_per_host_bounds=excluded.md_tpu_chips_per_host_bounds, "
            "md_gpu_count=excluded.md_gpu_count, md_gpu_name=excluded.md_gpu_name, "
            "md_gpu_memory_mb=excluded.md_gpu_memory_mb, "
            "md_gce_instance_name=excluded.md_gce_instance_name, md_gce_zone=excluded.md_gce_zone, "
            "md_git_hash=excluded.md_git_hash, md_device_json=excluded.md_device_json",
            (
                req.worker_id,
                req.address,
                req.now_ms,
                req.total_cpu_millicores,
                req.total_memory_bytes,
                req.total_gpu_count,
                req.total_tpu_count,
                req.device_type,
                req.device_variant,
                req.slice_id,
                req.scale_group,
                md.hostname,
                md.ip_address,
                md.cpu_count,
                md.memory_bytes,
                md.disk_bytes,
                md.tpu_name,
                md.tpu_worker_hostnames,
                md.tpu_worker_id,
                md.tpu_chips_per_host_bounds,
                md.gpu_count,
                md.gpu_name,
                md.gpu_memory_mb,
                md.gce_instance_name,
                md.gce_zone,
                md.git_hash,
                md.device_json,
            ),
        )
        cur.execute("DELETE FROM worker_attributes WHERE worker_id = ?", (req.worker_id,))
        for key, value_type, str_value, int_value, float_value in req.attributes:
            cur.execute(
                "INSERT INTO worker_attributes(worker_id, key, value_type, str_value, int_value, float_value) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (req.worker_id, key, value_type, str_value, int_value, float_value),
            )

    def get_active_row(self, cur: Cursor, worker_id: str) -> WorkerActiveRow | None:
        """Fetch consecutive_failures and last_heartbeat_ms for an active worker.

        Returns None if the worker doesn't exist or is inactive.
        """
        row = cur.execute(
            "SELECT consecutive_failures, last_heartbeat_ms FROM workers WHERE worker_id = ? AND active = 1",
            (worker_id,),
        ).fetchone()
        if row is None:
            return None
        return WorkerActiveRow(
            consecutive_failures=int(row["consecutive_failures"]),
            last_heartbeat_ms=int(row["last_heartbeat_ms"]) if row["last_heartbeat_ms"] is not None else None,
        )

    def get_row(self, cur: Cursor, worker_id: str) -> WorkerDetailRow | None:
        """Fetch the full worker row. Returns None if not found."""
        row = cur.execute(
            f"SELECT {WORKER_DETAIL_PROJECTION.select_clause(prefix=False)} FROM workers WHERE worker_id = ?",
            (worker_id,),
        ).fetchone()
        if row is None:
            return None
        return WORKER_DETAIL_PROJECTION.decode_one([row])

    def get_healthy_active(self, cur: Cursor, worker_id: str) -> dict | None:
        """Fetch worker_id and address for a healthy active worker.

        Returns None if the worker is missing, inactive, or unhealthy.
        """
        row = cur.execute(
            "SELECT worker_id, address FROM workers WHERE worker_id = ? AND active = 1 AND healthy = 1",
            (worker_id,),
        ).fetchone()
        if row is None:
            return None
        return row

    def prune_task_history(self, cur: Cursor, retention: int) -> int:
        """Trim worker_task_history to *retention* rows per worker."""
        return self._prune_per_worker_history(
            cur, "worker_task_history", retention, order_by="assigned_at_ms DESC, id DESC"
        )

    def prune_resource_history(self, cur: Cursor, retention: int) -> int:
        """Trim worker_resource_history to *retention* rows per worker."""
        return self._prune_per_worker_history(cur, "worker_resource_history", retention)

    def _prune_per_worker_history(self, cur: Cursor, table: str, retention: int, order_by: str = "id DESC") -> int:
        """Trim a per-worker history table to *retention* rows per worker."""
        rows = cur.execute(
            f"SELECT worker_id, COUNT(*) as cnt FROM {table} GROUP BY worker_id HAVING cnt > ?",
            (retention,),
        ).fetchall()
        total_deleted = 0
        for row in rows:
            wid = row["worker_id"]
            cur.execute(
                f"DELETE FROM {table} "
                "WHERE worker_id = ? "
                f"AND id NOT IN ("
                f"  SELECT id FROM {table} "
                "  WHERE worker_id = ? "
                f"  ORDER BY {order_by} LIMIT ?"
                ")",
                (wid, wid, retention),
            )
            total_deleted += cur.rowcount
        if total_deleted > 0:
            logger.info("Pruned %d %s rows", total_deleted, table)
        return total_deleted

    def get_inactive_worker_before(self, cur: Cursor, cutoff_ms: int) -> str | None:
        """Return a single inactive/unhealthy worker_id with heartbeat before the cutoff."""
        row = cur.execute(
            "SELECT worker_id FROM workers WHERE (active = 0 OR healthy = 0) AND last_heartbeat_ms < ? LIMIT 1",
            (cutoff_ms,),
        ).fetchone()
        if row is None:
            return None
        return str(row["worker_id"])


# ---------------------------------------------------------------------------
# DispatchStore
# ---------------------------------------------------------------------------


class DispatchStore:
    """Typed operations for the dispatch_queue table.

    Process-scoped: every method takes the open ``Cursor``.
    Encapsulates enqueue, drain, and delete so callers don't scatter raw
    dispatch_queue SQL.
    """

    def enqueue_run(self, cur: Cursor, worker_id: str, payload: bytes, now_ms: int) -> None:
        """Queue a 'run' dispatch entry for delivery on the next heartbeat."""
        cur.execute(
            "INSERT INTO dispatch_queue(worker_id, kind, payload_proto, task_id, created_at_ms) "
            "VALUES (?, 'run', ?, NULL, ?)",
            (worker_id, payload, now_ms),
        )

    def enqueue_kill(self, cur: Cursor, worker_id: str | None, task_id: str, now_ms: int) -> None:
        """Queue a 'kill' dispatch entry for delivery on the next heartbeat."""
        cur.execute(
            "INSERT INTO dispatch_queue(worker_id, kind, payload_proto, task_id, created_at_ms) "
            "VALUES (?, 'kill', NULL, ?, ?)",
            (worker_id, task_id, now_ms),
        )

    def drain_for_worker(self, cur: Cursor, worker_id: str) -> list[tuple[str, bytes | None, str | None]]:
        """SELECT and DELETE dispatch rows for one worker.

        Returns list of (kind, payload_proto, task_id) tuples ordered by id ASC.
        """
        rows = cur.execute(
            "SELECT kind, payload_proto, task_id FROM dispatch_queue WHERE worker_id = ? ORDER BY id ASC",
            (worker_id,),
        ).fetchall()
        if rows:
            cur.execute("DELETE FROM dispatch_queue WHERE worker_id = ?", (worker_id,))
        return [(str(r["kind"]), r["payload_proto"], r["task_id"]) for r in rows]

    def drain_for_workers(
        self, cur: Cursor, worker_ids: list[str]
    ) -> dict[str, list[tuple[str, bytes | None, str | None]]]:
        """Batch drain dispatch rows for multiple workers.

        Returns a dict mapping worker_id to list of (kind, payload_proto, task_id) tuples.
        """
        if not worker_ids:
            return {}
        placeholders = sql_placeholders(len(worker_ids))
        rows = cur.execute(
            f"SELECT worker_id, kind, payload_proto, task_id FROM dispatch_queue "
            f"WHERE worker_id IN ({placeholders}) ORDER BY id ASC",
            tuple(worker_ids),
        ).fetchall()
        if rows:
            cur.execute(
                f"DELETE FROM dispatch_queue WHERE worker_id IN ({placeholders})",
                tuple(worker_ids),
            )
        result: dict[str, list[tuple[str, bytes | None, str | None]]] = {}
        for r in rows:
            wid = str(r["worker_id"])
            if wid not in result:
                result[wid] = []
            result[wid].append((str(r["kind"]), r["payload_proto"], r["task_id"]))
        return result

    def drain_direct_kills(self, cur: Cursor) -> list[str]:
        """Drain NULL-worker kill entries. Returns list of task_ids."""
        rows = cur.execute(
            "SELECT task_id FROM dispatch_queue WHERE worker_id IS NULL AND kind = 'kill'",
        ).fetchall()
        task_ids = [str(r["task_id"]) for r in rows if r["task_id"] is not None]
        if rows:
            cur.execute("DELETE FROM dispatch_queue WHERE worker_id IS NULL AND kind = 'kill'")
        return task_ids

    def delete_for_worker(self, cur: Cursor, worker_id: str) -> None:
        """Delete all dispatch entries for a worker."""
        cur.execute("DELETE FROM dispatch_queue WHERE worker_id = ?", (worker_id,))

    def replace_claims(self, cur: Cursor, claims: dict[WorkerId, tuple[str, int]]) -> None:
        """Replace all reservation claims atomically.

        Args:
            claims: Mapping of worker_id -> (job_id, entry_idx).
        """
        cur.execute("DELETE FROM reservation_claims")
        cur.executemany(
            "INSERT INTO reservation_claims(worker_id, job_id, entry_idx) VALUES (?, ?, ?)",
            [(str(worker_id), job_id, entry_idx) for worker_id, (job_id, entry_idx) in claims.items()],
        )


# ---------------------------------------------------------------------------
# UserStore
# ---------------------------------------------------------------------------


class UserStore:
    """User and budget table operations."""

    def ensure_user_and_budget(
        self,
        cur: Cursor,
        user: str,
        now_ms: int,
        budget_defaults: UserBudgetDefaults,
    ) -> None:
        """Create user and default budget row if they don't already exist."""
        cur.execute(
            "INSERT OR IGNORE INTO users(user_id, created_at_ms) VALUES (?, ?)",
            (user, now_ms),
        )
        cur.execute(
            "INSERT OR IGNORE INTO user_budgets(user_id, budget_limit, max_band, updated_at_ms) " "VALUES (?, ?, ?, ?)",
            (user, budget_defaults.budget_limit, budget_defaults.max_band, now_ms),
        )


# ---------------------------------------------------------------------------
# ControllerStore — transaction-scoped handle bundling cursor + store access.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ControllerStore:
    """Transaction-scoped handle bundling the cursor with typed store access.

    Stores are process-scoped; ControllerStore is the thin per-transaction view
    binding them to the active cursor. State-machine code takes ControllerStore
    as its single DB argument.
    """

    cur: Cursor
    tasks: TaskStore
    jobs: JobStore
    workers: WorkerStore
    endpoints: EndpointStore
    dispatch: DispatchStore
    users: UserStore


# ---------------------------------------------------------------------------
# ControllerStores — process-scoped bundle owning the stores (and the db).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ControllerStores:
    """Process-scoped bundle of stores, constructed on top of a ``ControllerDB``.

    This is the single ownership layer for domain access. ``ControllerDB`` knows
    nothing about stores; callers construct a ``ControllerDB`` (pure infra) and
    then wrap it with ``ControllerStores.from_db(db)``.
    """

    db: DbBackend
    endpoints: EndpointStore
    dispatch: DispatchStore
    users: UserStore
    workers: WorkerStore
    jobs: JobStore
    tasks: TaskStore

    @classmethod
    def from_db(cls, db: DbBackend) -> ControllerStores:
        # Construction order encodes the store dependency graph:
        #   endpoints, dispatch, users: no dependencies (leaf caches/stores)
        #   workers depends on endpoints (for _remove_task_endpoints on dead
        #     workers) and dispatch (to drain pending dispatches on remove)
        #   jobs depends on endpoints (to cascade endpoint cleanup on job kill)
        #   tasks depends on endpoints (same reason) and jobs (to look up the
        #     job row while transitioning a task)
        # Adding a new store? Place it by its inward-edge count — leaves first.
        endpoints = EndpointStore()
        with db.read_snapshot() as snap:
            endpoints._load_all(snap)
        dispatch = DispatchStore()
        users = UserStore()
        workers = WorkerStore(endpoints, dispatch)
        jobs = JobStore(endpoints)
        tasks = TaskStore(endpoints, jobs)
        return cls(
            db=db,
            endpoints=endpoints,
            dispatch=dispatch,
            users=users,
            workers=workers,
            jobs=jobs,
            tasks=tasks,
        )

    @contextmanager
    def transact(self) -> Iterator[ControllerStore]:
        """Open an IMMEDIATE transaction and yield a per-txn ``ControllerStore``."""
        with self.db.transaction() as cur:
            yield ControllerStore(
                cur=cur,
                tasks=self.tasks,
                jobs=self.jobs,
                workers=self.workers,
                endpoints=self.endpoints,
                dispatch=self.dispatch,
                users=self.users,
            )

    @contextmanager
    def read(self) -> Iterator[ControllerStore]:
        """Open a read-only snapshot and yield a ControllerStore bound to it.

        Store methods that only read can be called against ``ctx.cur`` inside
        this scope. Write methods will fail at runtime (QuerySnapshot lacks
        on_commit/rowcount) — that's intentional.
        """
        with self.db.read_snapshot() as snap:
            yield ControllerStore(
                cur=snap,
                tasks=self.tasks,
                jobs=self.jobs,
                workers=self.workers,
                endpoints=self.endpoints,
                dispatch=self.dispatch,
                users=self.users,
            )
