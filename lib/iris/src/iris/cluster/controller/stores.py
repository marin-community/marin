# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed store layer over :mod:`iris.cluster.controller.db`.

Stores group related SQL against a single entity (jobs, tasks, workers,
endpoints, ...) and expose a typed API that callers invoke inside an open
transaction (read or write). :class:`ControllerStore` bundles every per-entity
store and forwards ``transaction()`` / ``read_snapshot()`` to the underlying
:class:`ControllerDB`.

Dependency chain (target state)::

    db.py        — connections, migrations, transaction context managers
    schema.py    — table DDL, row dataclasses, projections
    stores.py    — depends on { db, schema }; per-entity stores
    transitions.py — depends on stores; stores own the SQL

The layer is introduced incrementally. The current state is mid-migration:
``EndpointStore`` and ``JobStore`` are populated, while ``TaskStore``,
``TaskAttemptStore``, ``WorkerStore``, ``DispatchQueueStore`` and
``ReservationStore`` are still empty skeletons. ``ControllerTransitions``
keeps a temporary ``self._db`` backdoor for SQL that has not yet been
moved (tasks, workers, dispatch queue, reservations, the ``meta`` table,
worker-attribute cache). That backdoor is removed in a later phase once
every entity has a store.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from threading import RLock

from rigging.timing import Duration, Timestamp

from iris.cluster.constraints import AttributeValue
from iris.cluster.controller.codec import resource_spec_from_scalars
from iris.cluster.controller.db import (
    ACTIVE_TASK_STATES,
    ControllerDB,
    EndpointQuery,
    QuerySnapshot,
    TransactionCursor,
)
from iris.cluster.controller.schema import (
    ATTEMPT_PROJECTION,
    ENDPOINT_PROJECTION,
    JOB_CONFIG_JOIN,
    JOB_DETAIL_PROJECTION,
    TASK_DETAIL_PROJECTION,
    WORKER_DETAIL_PROJECTION,
    AttemptRow,
    EndpointRow,
    JobDetailRow,
    TaskDetailRow,
    WorkerDetailRow,
)
from iris.cluster.types import TERMINAL_JOB_STATES, TERMINAL_TASK_STATES, JobName, WorkerId, get_gpu_count, get_tpu_count
from iris.rpc import job_pb2

logger = logging.getLogger(__name__)

WORKER_TASK_HISTORY_RETENTION = 500
"""Maximum worker_task_history rows retained per worker."""

WORKER_RESOURCE_HISTORY_RETENTION = 500
"""Maximum worker_resource_history rows retained per worker."""

TASK_RESOURCE_HISTORY_RETENTION = 50
"""Maximum task_resource_history rows retained per (task_id, attempt_id)."""

TASK_RESOURCE_HISTORY_TERMINAL_TTL = Duration.from_hours(1)
"""After a task reaches a terminal state, fully evict its resource history after this delay."""

TASK_RESOURCE_HISTORY_DELETE_CHUNK = 1000
"""Maximum task_ids per DELETE in task_resource_history pruning."""


# Store read methods accept either a write cursor or a read snapshot. Writes
# require ``TransactionCursor`` explicitly so a ``QuerySnapshot`` can't be
# accidentally passed to a mutating API. (This alias does *not* prevent a store
# read method from issuing writes internally — it just polices the caller-side
# direction. A read-only ``Protocol`` would be stricter; not yet worth the
# plumbing.)
Tx = TransactionCursor | QuerySnapshot


# =============================================================================
# EndpointStore
# =============================================================================


class EndpointStore:
    """Process-local write-through cache over the ``endpoints`` table.

    Profiling showed ``ListEndpoints`` dominated controller CPU — not because
    the SQL was slow per se, but because every call serialized through the
    read-connection pool and walked a large WAL to build a snapshot. The
    endpoints table is tiny (hundreds of rows) and only changes on explicit
    register / unregister, so it is a natural fit for a write-through
    in-memory cache.

    Design invariants:

    * Reads never touch the DB. All lookups are served from in-memory maps
      guarded by an ``RLock`` — readers observe a consistent snapshot of the
      indexes, never a torn state mid-update.
    * Writes execute the SQL inside the caller's transaction. The in-memory
      update is scheduled as a post-commit hook on the cursor so memory only
      changes after the DB has committed. If the transaction rolls back, the
      hook never fires.
    * N is small enough (≈ hundreds) that linear scans for prefix / task / id
      lookups are simpler and plenty fast. Extra indexes (by name, by task_id)
      speed the two common cases.

    The store is the sole source of truth for endpoint reads; nothing else in
    the controller tree should SELECT from ``endpoints``.
    """

    def __init__(self, db: ControllerDB) -> None:
        self._db = db
        self._lock = RLock()
        self._by_id: dict[str, EndpointRow] = {}
        # One name can map to multiple endpoint_ids — the schema does not enforce
        # uniqueness on ``name``, and ``INSERT OR REPLACE`` keys off endpoint_id.
        self._by_name: dict[str, set[str]] = {}
        self._by_task: dict[JobName, set[str]] = {}
        self._load_all()

    # -- Loading --------------------------------------------------------------

    def _load_all(self) -> None:
        with self._db.read_snapshot() as q:
            rows = ENDPOINT_PROJECTION.decode(
                q.fetchall(f"SELECT {ENDPOINT_PROJECTION.select_clause()} FROM endpoints e"),
            )
        with self._lock:
            self._by_id.clear()
            self._by_name.clear()
            self._by_task.clear()
            for row in rows:
                self._index(row)
        logger.info("EndpointStore loaded %d endpoint(s) from DB", len(rows))

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
        """Return endpoint rows matching ``query``; all filters AND together."""
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

    def add(self, cur: TransactionCursor, endpoint: EndpointRow) -> bool:
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

        def apply() -> None:
            with self._lock:
                # Replace: drop any previous row with this id first so the
                # name/task indexes stay consistent on overwrite.
                self._unindex(endpoint.endpoint_id)
                self._index(endpoint)

        cur.on_commit(apply)
        return True

    def remove(self, cur: TransactionCursor, endpoint_id: str) -> EndpointRow | None:
        """Remove a single endpoint by id. Returns the removed row snapshot, if any."""
        existing = self.get(endpoint_id)
        if existing is None:
            return None
        cur.execute("DELETE FROM endpoints WHERE endpoint_id = ?", (endpoint_id,))

        def apply() -> None:
            with self._lock:
                self._unindex(endpoint_id)

        cur.on_commit(apply)
        return existing

    def remove_by_task(self, cur: TransactionCursor, task_id: JobName) -> list[str]:
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

        def apply() -> None:
            with self._lock:
                for eid in ids:
                    self._unindex(eid)

        cur.on_commit(apply)
        return ids

    def remove_by_job_ids(self, cur: TransactionCursor, job_ids: Sequence[JobName]) -> list[str]:
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
        placeholders = ",".join("?" for _ in wire_ids)
        cur.execute(
            f"DELETE FROM endpoints WHERE job_id IN ({placeholders})",
            tuple(wire_ids),
        )
        if not to_remove:
            return []

        def apply() -> None:
            with self._lock:
                for eid in to_remove:
                    self._unindex(eid)

        cur.on_commit(apply)
        return to_remove


# =============================================================================
# Phase-1 skeletons for the remaining per-entity stores.
#
# These exist so callers can already reference ``store.jobs`` etc. and so that
# subsequent phases (moving SQL out of transitions.py) land as additive
# changes to these classes rather than needing new plumbing each time.
# Methods are added as the corresponding SQL migrates out of transitions.py.
# =============================================================================


@dataclass(frozen=True, slots=True)
class JobInsertParams:
    """Fields needed to insert one row into the ``jobs`` table.

    Holder jobs set ``is_reservation_holder=True`` and leave ``error`` /
    ``exit_code`` / ``finished_at_ms`` / ``scheduling_deadline_epoch_ms`` None;
    the regular path passes the corresponding submit-time values.
    """

    job_id: JobName
    user_id: str
    parent_job_id: str | None
    root_job_id: str
    depth: int
    state: int
    submitted_at_ms: int
    root_submitted_at_ms: int
    started_at_ms: int | None
    finished_at_ms: int | None
    scheduling_deadline_epoch_ms: int | None
    error: str | None
    exit_code: int | None
    num_tasks: int
    is_reservation_holder: bool
    name: str
    has_reservation: bool


@dataclass(frozen=True, slots=True)
class JobConfigInsertParams:
    """Fields needed to insert one row into the ``job_config`` table.

    Holder jobs do not set ``submit_argv`` / ``reservation`` / ``fail_if_exists``;
    those have defaults so the holder path can omit them.
    """

    job_id: JobName
    name: str
    has_reservation: bool
    res_cpu_millicores: int
    res_memory_bytes: int
    res_disk_bytes: int
    res_device_json: str | None
    constraints_json: str
    has_coscheduling: bool
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
    fail_if_exists: bool = False


@dataclass(frozen=True, slots=True)
class JobRecomputeBasis:
    state: int
    started_at_ms: int | None
    max_task_failures: int


@dataclass(frozen=True, slots=True)
class TaskInsertParams:
    """Fields needed to insert one row into the ``tasks`` table."""

    task_id: JobName
    job_id: JobName
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
class TaskAttemptInsertParams:
    """Fields needed to insert one row into ``task_attempts``."""

    task_id: JobName
    attempt_id: int
    worker_id: WorkerId | None
    state: int
    created_at_ms: int


@dataclass(frozen=True, slots=True)
class TaskAttemptUpdateParams:
    """Fields for applying a worker/direct-provider attempt update."""

    task_id: JobName
    attempt_id: int
    state: int
    started_at_ms: int | None
    finished_at_ms: int | None
    exit_code: int | None
    error: str | None


@dataclass(frozen=True, slots=True)
class TaskStateUpdateParams:
    """Fields for applying a computed task state update."""

    task_id: JobName
    state: int
    error: str | None
    exit_code: int | None
    started_at_ms: int | None
    finished_at_ms: int | None
    failure_count: int
    preemption_count: int


@dataclass(frozen=True, slots=True)
class ResourceUsageInsertParams:
    task_id: JobName
    attempt_id: int
    cpu_millicores: int
    memory_mb: int
    disk_mb: int
    memory_peak_mb: int
    timestamp_ms: int


@dataclass(frozen=True, slots=True)
class WorkerAttributeParams:
    key: str
    value_type: str
    str_value: str | None
    int_value: int | None
    float_value: float | None


@dataclass(frozen=True, slots=True)
class WorkerUpsertParams:
    """All scalar columns written by a worker registration/refresh.

    The upsert leaves ``committed_*`` counters and attributes untouched —
    attributes are replaced via :meth:`WorkerStore.replace_attributes` and
    resource commitment is tracked incrementally via
    :meth:`WorkerStore.add_committed_resources` / ``decommit_resources``.
    """

    worker_id: WorkerId
    address: str
    last_heartbeat_ms: int
    total_cpu_millicores: int
    total_memory_bytes: int
    total_gpu_count: int
    total_tpu_count: int
    device_type: str
    device_variant: str
    slice_id: str
    scale_group: str
    md_hostname: str
    md_ip_address: str
    md_cpu_count: int
    md_memory_bytes: int
    md_disk_bytes: int
    md_tpu_name: str
    md_tpu_worker_hostnames: str
    md_tpu_worker_id: str
    md_tpu_chips_per_host_bounds: str
    md_gpu_count: int
    md_gpu_name: str
    md_gpu_memory_mb: int
    md_gce_instance_name: str
    md_gce_zone: str
    md_git_hash: str
    md_device_json: str


@dataclass(frozen=True, slots=True)
class ActiveWorkerStatus:
    """Minimal row used by the worker-failure path: confirms the worker is
    active (non-None return) and reports its last heartbeat timestamp.
    """

    last_heartbeat_ms: int | None


@dataclass(frozen=True, slots=True)
class TaskScope:
    """Scope predicate for :meth:`TaskStore.list_active`.

    Exactly one field must be set. The store validates at the call boundary.
    ``null_worker=True`` matches rows where ``current_worker_id IS NULL``
    (direct-provider-promoted tasks).
    """

    job_id: JobName | None = None
    job_subtree: Sequence[JobName] | None = None
    worker_id: WorkerId | None = None
    worker_ids: Sequence[WorkerId] | None = None
    task_ids: Sequence[JobName] | None = None
    null_worker: bool = False


@dataclass(frozen=True, slots=True)
class ActiveTaskRow:
    """Task projection joined with ``jobs`` + ``job_config``.

    Shared by every cascade/scheduling query (``_kill_non_terminal_tasks``,
    ``_find_coscheduled_siblings``, ``cancel_job``, ``preempt_task``,
    ``cancel_tasks_for_timeout``, ``_remove_failed_worker``, poll paths). The
    resource columns are decoded into a single ``ResourceSpecProto`` so
    callers stop re-running ``resource_spec_from_scalars(...)`` at every
    site. Reservation-holder rows carry a populated ``resources`` that
    callers are expected to ignore (they never commit resources).
    """

    task_id: JobName
    job_id: JobName
    state: int
    current_attempt_id: int
    current_worker_id: WorkerId | None
    failure_count: int
    preemption_count: int
    max_retries_failure: int
    max_retries_preemption: int
    is_reservation_holder: bool
    has_coscheduling: bool
    resources: job_pb2.ResourceSpecProto


_ACTIVE_TASK_PROJECTION = (
    "t.task_id, t.job_id, t.state, t.current_attempt_id, t.current_worker_id, "
    "t.failure_count, t.preemption_count, t.max_retries_failure, t.max_retries_preemption, "
    "j.is_reservation_holder, "
    "jc.has_coscheduling, "
    "jc.res_cpu_millicores, jc.res_memory_bytes, jc.res_disk_bytes, jc.res_device_json"
)


def _decode_active_task_row(row) -> ActiveTaskRow:
    worker_id = row["current_worker_id"]
    return ActiveTaskRow(
        task_id=JobName.from_wire(str(row["task_id"])),
        job_id=JobName.from_wire(str(row["job_id"])),
        state=int(row["state"]),
        current_attempt_id=int(row["current_attempt_id"]),
        current_worker_id=WorkerId(str(worker_id)) if worker_id is not None else None,
        failure_count=int(row["failure_count"]),
        preemption_count=int(row["preemption_count"]),
        max_retries_failure=int(row["max_retries_failure"]),
        max_retries_preemption=int(row["max_retries_preemption"]),
        is_reservation_holder=bool(int(row["is_reservation_holder"])),
        has_coscheduling=bool(int(row["has_coscheduling"])),
        resources=resource_spec_from_scalars(
            int(row["res_cpu_millicores"]),
            int(row["res_memory_bytes"]),
            int(row["res_disk_bytes"]),
            row["res_device_json"],
        ),
    )


@dataclass(frozen=True, slots=True)
class PendingDispatchRow:
    """Scheduling payload for a pending task awaiting direct-provider promotion.

    Unlike :class:`ActiveTaskRow`, this row carries the full serialized
    runtime configuration (entrypoint / environment / ports / constraints
    / task_image / timeout) so the caller can assemble a
    ``RunTaskRequest``. Kept separate so other active-task queries don't
    pay for loading these JSON blobs.
    """

    task_id: JobName
    job_id: JobName
    current_attempt_id: int
    num_tasks: int
    resources: job_pb2.ResourceSpecProto
    entrypoint_json: str
    environment_json: str
    bundle_id: str
    ports_json: str
    constraints_json: str | None
    task_image: str
    timeout_ms: int | None


class JobStore:
    """Jobs, job_config, users, user_budgets.

    Holds the SQL for the four tables the controller uses to track a submitted
    job's lifecycle. Reads take a ``Tx`` (read snapshot or write cursor);
    writes require a ``TransactionCursor`` so static typing rules out
    mutations through a read-only snapshot.
    """

    def __init__(self, db: ControllerDB) -> None:
        self._db = db

    # -- Reads ---------------------------------------------------------------

    def get_state(self, tx: Tx, job_id: JobName) -> int | None:
        row = tx.fetchone("SELECT state FROM jobs WHERE job_id = ?", (job_id.to_wire(),))
        return int(row["state"]) if row is not None else None

    def get_root_submitted_at_ms(self, tx: Tx, job_id: JobName) -> int | None:
        row = tx.fetchone("SELECT root_submitted_at_ms FROM jobs WHERE job_id = ?", (job_id.to_wire(),))
        return int(row["root_submitted_at_ms"]) if row is not None else None

    def get_preemption_info(self, tx: Tx, job_id: JobName) -> tuple[int, int] | None:
        """Return ``(preemption_policy, num_tasks)`` or None if the job is gone."""
        row = tx.fetchone(
            f"SELECT jc.preemption_policy, j.num_tasks FROM jobs j {JOB_CONFIG_JOIN} WHERE j.job_id = ?",
            (job_id.to_wire(),),
        )
        if row is None:
            return None
        return int(row["preemption_policy"]), int(row["num_tasks"])

    def get_recompute_basis(self, tx: Tx, job_id: JobName) -> JobRecomputeBasis | None:
        row = tx.fetchone(
            f"SELECT j.state, j.started_at_ms, jc.max_task_failures "
            f"FROM jobs j {JOB_CONFIG_JOIN} WHERE j.job_id = ?",
            (job_id.to_wire(),),
        )
        if row is None:
            return None
        return JobRecomputeBasis(
            state=int(row["state"]),
            started_at_ms=int(row["started_at_ms"]) if row["started_at_ms"] is not None else None,
            max_task_failures=int(row["max_task_failures"]),
        )

    def get_detail(self, tx: Tx, job_id: JobName) -> JobDetailRow | None:
        row = tx.fetchone(
            f"SELECT {JOB_DETAIL_PROJECTION.select_clause()} " f"FROM jobs j {JOB_CONFIG_JOIN} WHERE j.job_id = ?",
            (job_id.to_wire(),),
        )
        if row is None:
            return None
        return JOB_DETAIL_PROJECTION.decode_one([row])

    def get_config(self, tx: Tx, job_id: JobName) -> dict | None:
        """Return the raw ``job_config`` row as a dict, or None.

        Callers currently access fields by string key (e.g. ``jc["res_cpu_millicores"]``);
        returning a dict keeps the existing consumers working while SQL moves
        behind the store.
        """
        row = tx.fetchone("SELECT * FROM job_config WHERE job_id = ?", (job_id.to_wire(),))
        return dict(row) if row is not None else None

    def list_descendants(
        self,
        tx: Tx,
        parent_id: JobName,
        *,
        exclude_reservation_holders: bool = False,
    ) -> list[JobName]:
        """Return all transitive descendants of ``parent_id`` (not ``parent_id`` itself).

        When ``exclude_reservation_holders`` is True, reservation-holder jobs and
        anything below them are skipped — used during preemption retry, where the
        parent goes back to PENDING and needs its reservation subtree preserved.
        """
        if exclude_reservation_holders:
            rows = tx.fetchall(
                "WITH RECURSIVE subtree(job_id) AS ("
                "  SELECT job_id FROM jobs WHERE parent_job_id = ? AND is_reservation_holder = 0 "
                "  UNION ALL "
                "  SELECT j.job_id FROM jobs j JOIN subtree s ON j.parent_job_id = s.job_id"
                "   WHERE j.is_reservation_holder = 0"
                ") SELECT job_id FROM subtree",
                (parent_id.to_wire(),),
            )
        else:
            rows = tx.fetchall(
                "WITH RECURSIVE subtree(job_id) AS ("
                "  SELECT job_id FROM jobs WHERE parent_job_id = ? "
                "  UNION ALL "
                "  SELECT j.job_id FROM jobs j JOIN subtree s ON j.parent_job_id = s.job_id"
                ") SELECT job_id FROM subtree",
                (parent_id.to_wire(),),
            )
        return [JobName.from_wire(str(row["job_id"])) for row in rows]

    def list_subtree(self, tx: Tx, root_id: JobName) -> list[JobName]:
        """Return ``root_id`` and all its transitive descendants."""
        rows = tx.fetchall(
            "WITH RECURSIVE subtree(job_id) AS ("
            "  SELECT job_id FROM jobs WHERE job_id = ? "
            "  UNION ALL "
            "  SELECT j.job_id FROM jobs j JOIN subtree s ON j.parent_job_id = s.job_id"
            ") SELECT job_id FROM subtree",
            (root_id.to_wire(),),
        )
        return [JobName.from_wire(str(row["job_id"])) for row in rows]

    def find_prunable(self, tx: Tx, before_ms: int) -> JobName | None:
        """Return one terminal job whose ``finished_at_ms`` predates ``before_ms``, or None."""
        placeholders = ",".join("?" for _ in TERMINAL_JOB_STATES)
        row = tx.fetchone(
            f"SELECT job_id FROM jobs WHERE state IN ({placeholders})"
            " AND finished_at_ms IS NOT NULL AND finished_at_ms < ? LIMIT 1",
            (*TERMINAL_JOB_STATES, before_ms),
        )
        return JobName.from_wire(str(row["job_id"])) if row is not None else None

    def get_workdir_files(self, tx: Tx, job_id: JobName) -> dict[str, bytes]:
        """Return ``{filename: data}`` for all workdir files attached to a job."""
        rows = tx.fetchall(
            "SELECT filename, data FROM job_workdir_files WHERE job_id = ?",
            (job_id.to_wire(),),
        )
        return {str(row["filename"]): bytes(row["data"]) for row in rows}

    # -- Writes --------------------------------------------------------------

    def update_state_if_not_terminal(
        self,
        cur: TransactionCursor,
        job_id: JobName,
        new_state: int,
        error: str | None,
        finished_at_ms: int | None,
    ) -> None:
        """Set a new state on a single job, skipping rows already in a terminal state."""
        placeholders = ",".join("?" for _ in TERMINAL_JOB_STATES)
        cur.execute(
            "UPDATE jobs SET state = ?, error = ?, finished_at_ms = COALESCE(finished_at_ms, ?) "
            f"WHERE job_id = ? AND state NOT IN ({placeholders})",
            (new_state, error, finished_at_ms, job_id.to_wire(), *TERMINAL_JOB_STATES),
        )

    def bulk_update_state(
        self,
        cur: TransactionCursor,
        job_ids: Sequence[JobName],
        new_state: int,
        error: str | None,
        finished_at_ms: int | None,
        guard_states: Iterable[int],
    ) -> None:
        """Set state on many jobs; rows in any of ``guard_states`` are skipped."""
        if not job_ids:
            return
        wire_ids = [jid.to_wire() for jid in job_ids]
        guard = tuple(guard_states)
        job_placeholders = ",".join("?" for _ in wire_ids)
        guard_placeholders = ",".join("?" for _ in guard)
        cur.execute(
            f"UPDATE jobs SET state = ?, error = ?, finished_at_ms = COALESCE(finished_at_ms, ?) "
            f"WHERE job_id IN ({job_placeholders}) AND state NOT IN ({guard_placeholders})",
            (new_state, error, finished_at_ms, *wire_ids, *guard),
        )

    def mark_running_if_pending(self, cur: TransactionCursor, job_id: JobName, now_ms: int) -> None:
        """Advance PENDING → RUNNING and set ``started_at_ms`` if not already populated."""
        cur.execute(
            "UPDATE jobs SET state = CASE WHEN state = ? THEN ? ELSE state END, "
            "started_at_ms = COALESCE(started_at_ms, ?) WHERE job_id = ?",
            (job_pb2.JOB_STATE_PENDING, job_pb2.JOB_STATE_RUNNING, now_ms, job_id.to_wire()),
        )

    def apply_recomputed_state(
        self,
        cur: TransactionCursor,
        job_id: JobName,
        new_state: int,
        now_ms: int,
        error: str | None,
    ) -> None:
        """Write the result of ``_recompute_job_state`` back to the row.

        Sets ``started_at_ms`` (if moving to RUNNING), ``finished_at_ms`` (if
        moving to a terminal state), and ``error`` (if the terminal reason
        warrants one). The caller has already decided ``new_state`` differs
        from the current state.
        """
        terminal_placeholders = ",".join("?" for _ in TERMINAL_JOB_STATES)
        cur.execute(
            "UPDATE jobs SET state = ?, "
            "started_at_ms = CASE WHEN ? = ? THEN COALESCE(started_at_ms, ?) ELSE started_at_ms END, "
            f"finished_at_ms = CASE WHEN ? IN ({terminal_placeholders}) THEN ? ELSE finished_at_ms END, "
            "error = CASE WHEN ? IN (?, ?, ?, ?) THEN ? ELSE error END "
            "WHERE job_id = ?",
            (
                new_state,
                new_state,
                job_pb2.JOB_STATE_RUNNING,
                now_ms,
                new_state,
                *TERMINAL_JOB_STATES,
                now_ms,
                new_state,
                job_pb2.JOB_STATE_FAILED,
                job_pb2.JOB_STATE_KILLED,
                job_pb2.JOB_STATE_UNSCHEDULABLE,
                job_pb2.JOB_STATE_WORKER_FAILED,
                error,
                job_id.to_wire(),
            ),
        )

    def insert(self, cur: TransactionCursor, params: JobInsertParams) -> None:
        cur.execute(
            "INSERT INTO jobs("
            "job_id, user_id, parent_job_id, root_job_id, depth, state, submitted_at_ms, "
            "root_submitted_at_ms, started_at_ms, finished_at_ms, scheduling_deadline_epoch_ms, "
            "error, exit_code, num_tasks, is_reservation_holder, name, has_reservation"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                params.job_id.to_wire(),
                params.user_id,
                params.parent_job_id,
                params.root_job_id,
                params.depth,
                params.state,
                params.submitted_at_ms,
                params.root_submitted_at_ms,
                params.started_at_ms,
                params.finished_at_ms,
                params.scheduling_deadline_epoch_ms,
                params.error,
                params.exit_code,
                params.num_tasks,
                1 if params.is_reservation_holder else 0,
                params.name,
                1 if params.has_reservation else 0,
            ),
        )

    def insert_config(self, cur: TransactionCursor, params: JobConfigInsertParams) -> None:
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
                params.job_id.to_wire(),
                params.name,
                1 if params.has_reservation else 0,
                params.res_cpu_millicores,
                params.res_memory_bytes,
                params.res_disk_bytes,
                params.res_device_json,
                params.constraints_json,
                1 if params.has_coscheduling else 0,
                params.coscheduling_group_by,
                params.scheduling_timeout_ms,
                params.max_task_failures,
                params.entrypoint_json,
                params.environment_json,
                params.bundle_id,
                params.ports_json,
                params.max_retries_failure,
                params.max_retries_preemption,
                params.timeout_ms,
                params.preemption_policy,
                params.existing_job_policy,
                params.priority_band,
                params.task_image,
                params.submit_argv_json,
                params.reservation_json,
                1 if params.fail_if_exists else 0,
            ),
        )

    def delete(self, cur: TransactionCursor, job_id: JobName) -> None:
        """Delete a job row. ON DELETE CASCADE handles tasks, attempts, endpoints."""
        cur.execute("DELETE FROM jobs WHERE job_id = ?", (job_id.to_wire(),))

    def insert_workdir_files(
        self,
        cur: TransactionCursor,
        job_id: JobName,
        files: Mapping[str, bytes],
    ) -> None:
        """Insert each ``{filename: data}`` pair as a row in ``job_workdir_files``."""
        if not files:
            return
        cur.executemany(
            "INSERT INTO job_workdir_files(job_id, filename, data) VALUES (?, ?, ?)",
            [(job_id.to_wire(), name, data) for name, data in files.items()],
        )

    def reserve_priority_insertion_base(self, cur: TransactionCursor) -> int:
        """Bump the ``task_priority_insertion`` sequence and return the new value.

        Callers reserving N task slots use ``base + i`` for ``i in range(N)``.
        """
        return self._db.next_sequence("task_priority_insertion", cur=cur)

    # -- users / user_budgets ------------------------------------------------

    def ensure_user(self, cur: TransactionCursor, user_id: str, now_ms: int) -> None:
        """Idempotently create a ``users`` row at submission time."""
        cur.execute(
            "INSERT OR IGNORE INTO users(user_id, created_at_ms) VALUES (?, ?)",
            (user_id, now_ms),
        )


class TaskStore:
    """Tasks and task_resource_history."""

    def __init__(self, db: ControllerDB) -> None:
        self._db = db
        self._status_text_detail: dict[str, str] = {}  # task_id wire → detail markdown
        self._status_text_summary: dict[str, str] = {}  # task_id wire → summary markdown

    def set_status_text(self, task_id: str, detail_md: str, summary_md: str) -> None:
        """Store the latest markdown status text for a task (in memory only)."""
        self._status_text_detail[task_id] = detail_md
        self._status_text_summary[task_id] = summary_md

    def get_status_text_detail(self, task_id: str) -> str:
        """Return the latest detail markdown for a task, or empty string if none."""
        return self._status_text_detail.get(task_id, "")

    def get_status_text_summary(self, task_id: str) -> str:
        """Return the latest summary markdown for a task, or empty string if none."""
        return self._status_text_summary.get(task_id, "")

    def remove_status_text_by_job_ids(self, job_ids: Sequence[JobName]) -> None:
        """Evict status-text cache entries for all tasks owned by any of ``job_ids``."""
        if not job_ids:
            return
        prefixes = tuple(f"{jid.to_wire()}/" for jid in job_ids)
        for key in [k for k in self._status_text_detail if k.startswith(prefixes)]:
            del self._status_text_detail[key]
        for key in [k for k in self._status_text_summary if k.startswith(prefixes)]:
            del self._status_text_summary[key]

    # -- Reads ---------------------------------------------------------------

    def get_detail(self, tx: Tx, task_id: JobName) -> TaskDetailRow | None:
        row = tx.fetchone(
            f"SELECT {TASK_DETAIL_PROJECTION.select_clause()} FROM tasks t WHERE t.task_id = ?",
            (task_id.to_wire(),),
        )
        if row is None:
            return None
        return TASK_DETAIL_PROJECTION.decode_one([row])

    def bulk_get_detail(self, tx: Tx, task_ids: Iterable[JobName]) -> dict[JobName, TaskDetailRow]:
        """Return ``{task_id: TaskDetailRow}`` for all ``task_ids`` that exist.

        Missing ids are silently absent from the result. Chunks internally
        to stay under SQLite's statement-parameter limit.
        """
        result: dict[JobName, TaskDetailRow] = {}
        ids = list(task_ids)
        for chunk_start in range(0, len(ids), 900):
            chunk = ids[chunk_start : chunk_start + 900]
            if not chunk:
                continue
            placeholders = ",".join("?" for _ in chunk)
            rows = tx.fetchall(
                f"SELECT {TASK_DETAIL_PROJECTION.select_clause()} " f"FROM tasks t WHERE t.task_id IN ({placeholders})",
                tuple(tid.to_wire() for tid in chunk),
            )
            for task in TASK_DETAIL_PROJECTION.decode(rows):
                result[task.task_id] = task
        return result

    def get_job_id(self, tx: Tx, task_id: JobName) -> JobName | None:
        row = tx.fetchone("SELECT job_id FROM tasks WHERE task_id = ?", (task_id.to_wire(),))
        return JobName.from_wire(str(row["job_id"])) if row is not None else None

    def get_current_attempt_id(self, tx: Tx, task_id: JobName) -> int | None:
        row = tx.fetchone("SELECT current_attempt_id FROM tasks WHERE task_id = ?", (task_id.to_wire(),))
        return int(row["current_attempt_id"]) if row is not None else None

    def get_priority_band_for_job(self, tx: Tx, job_id: JobName) -> int | None:
        row = tx.fetchone(
            "SELECT priority_band FROM tasks WHERE job_id = ? LIMIT 1",
            (job_id.to_wire(),),
        )
        return int(row["priority_band"]) if row is not None else None

    def state_counts_for_job(self, tx: Tx, job_id: JobName) -> dict[int, int]:
        rows = tx.fetchall(
            "SELECT state, COUNT(*) AS c FROM tasks WHERE job_id = ? GROUP BY state",
            (job_id.to_wire(),),
        )
        return {int(row["state"]): int(row["c"]) for row in rows}

    def first_error_for_job(self, tx: Tx, job_id: JobName) -> str | None:
        row = tx.fetchone(
            "SELECT error FROM tasks WHERE job_id = ? AND error IS NOT NULL ORDER BY task_index LIMIT 1",
            (job_id.to_wire(),),
        )
        return str(row["error"]) if row is not None else None

    def list_active(
        self,
        tx: Tx,
        scope: TaskScope,
        *,
        states: Iterable[int],
        exclude_task_id: JobName | None = None,
        exclude_reservation_holders: bool = False,
        order_by_task_id: bool = False,
        limit: int | None = None,
    ) -> list[ActiveTaskRow]:
        """Return :class:`ActiveTaskRow` rows matching ``scope`` and ``states``.

        ``scope`` picks which side of the query the filter binds to
        (single job, job subtree, worker, explicit task list, or NULL
        worker). ``states`` is the required ``tasks.state`` filter —
        typical values are ``ACTIVE_TASK_STATES``,
        ``EXECUTING_TASK_STATES``, or ``NON_TERMINAL_TASK_STATES``. Pass
        an empty ``states`` (or an empty ``task_ids``/``job_subtree``
        scope) to short-circuit to an empty list.
        """
        scope_set = sum(
            1
            for x in (scope.job_id, scope.job_subtree, scope.worker_id, scope.worker_ids, scope.task_ids)
            if x is not None
        ) + (1 if scope.null_worker else 0)
        if scope_set != 1:
            raise ValueError(
                "TaskScope must set exactly one of: " "job_id, job_subtree, worker_id, worker_ids, task_ids, null_worker"
            )

        where_parts: list[str] = []
        params: list[object] = []

        if scope.job_id is not None:
            where_parts.append("t.job_id = ?")
            params.append(scope.job_id.to_wire())
        elif scope.job_subtree is not None:
            if not scope.job_subtree:
                return []
            wires = [jid.to_wire() for jid in scope.job_subtree]
            ph = ",".join("?" for _ in wires)
            where_parts.append(f"t.job_id IN ({ph})")
            params.extend(wires)
        elif scope.worker_id is not None:
            where_parts.append("t.current_worker_id = ?")
            params.append(str(scope.worker_id))
        elif scope.worker_ids is not None:
            if not scope.worker_ids:
                return []
            wids = [str(wid) for wid in scope.worker_ids]
            ph = ",".join("?" for _ in wids)
            where_parts.append(f"t.current_worker_id IN ({ph})")
            params.extend(wids)
        elif scope.task_ids is not None:
            if not scope.task_ids:
                return []
            wires = [tid.to_wire() for tid in scope.task_ids]
            ph = ",".join("?" for _ in wires)
            where_parts.append(f"t.task_id IN ({ph})")
            params.extend(wires)
        else:  # null_worker
            where_parts.append("t.current_worker_id IS NULL")

        if exclude_task_id is not None:
            where_parts.append("t.task_id != ?")
            params.append(exclude_task_id.to_wire())

        if exclude_reservation_holders:
            where_parts.append("j.is_reservation_holder = 0")

        states_tuple = tuple(states)
        if not states_tuple:
            return []
        state_ph = ",".join("?" for _ in states_tuple)
        where_parts.append(f"t.state IN ({state_ph})")
        params.extend(states_tuple)

        sql = (
            f"SELECT {_ACTIVE_TASK_PROJECTION} "
            f"FROM tasks t JOIN jobs j ON j.job_id = t.job_id {JOB_CONFIG_JOIN} "
            f"WHERE {' AND '.join(where_parts)}"
        )
        if order_by_task_id:
            sql += " ORDER BY t.task_id ASC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)

        rows = tx.fetchall(sql, tuple(params))
        return [_decode_active_task_row(row) for row in rows]

    def get_with_resources(self, tx: Tx, task_id: JobName) -> ActiveTaskRow | None:
        """Fetch a single task with its job_config resource projection.

        Unlike :meth:`list_active`, no state filter is applied; callers
        (``preempt_task``) check the returned ``state`` themselves.
        """
        row = tx.fetchone(
            f"SELECT {_ACTIVE_TASK_PROJECTION} "
            f"FROM tasks t JOIN jobs j ON j.job_id = t.job_id {JOB_CONFIG_JOIN} "
            f"WHERE t.task_id = ?",
            (task_id.to_wire(),),
        )
        return _decode_active_task_row(row) if row is not None else None

    def list_pending_for_direct_provider(
        self,
        tx: Tx,
        limit: int,
    ) -> list[PendingDispatchRow]:
        """Return pending non-holder tasks eligible for direct-provider dispatch.

        Joins ``job_config`` to return the full runtime payload (entrypoint,
        environment, ports, constraints, task_image, timeout) that the caller
        needs to assemble a ``RunTaskRequest``. Returns at most ``limit`` rows.
        """
        if limit <= 0:
            return []
        rows = tx.fetchall(
            "SELECT t.task_id, t.job_id, t.current_attempt_id, j.num_tasks, "
            "jc.res_cpu_millicores, jc.res_memory_bytes, jc.res_disk_bytes, jc.res_device_json, "
            "jc.entrypoint_json, jc.environment_json, jc.bundle_id, jc.ports_json, "
            "jc.constraints_json, jc.task_image, jc.timeout_ms "
            f"FROM tasks t JOIN jobs j ON j.job_id = t.job_id {JOB_CONFIG_JOIN} "
            "WHERE t.state = ? AND j.is_reservation_holder = 0 "
            "LIMIT ?",
            (job_pb2.TASK_STATE_PENDING, limit),
        )
        result: list[PendingDispatchRow] = []
        for row in rows:
            timeout_ms = row["timeout_ms"]
            result.append(
                PendingDispatchRow(
                    task_id=JobName.from_wire(str(row["task_id"])),
                    job_id=JobName.from_wire(str(row["job_id"])),
                    current_attempt_id=int(row["current_attempt_id"]),
                    num_tasks=int(row["num_tasks"]),
                    resources=resource_spec_from_scalars(
                        int(row["res_cpu_millicores"]),
                        int(row["res_memory_bytes"]),
                        int(row["res_disk_bytes"]),
                        row["res_device_json"],
                    ),
                    entrypoint_json=str(row["entrypoint_json"]),
                    environment_json=str(row["environment_json"]),
                    bundle_id=str(row["bundle_id"]),
                    ports_json=str(row["ports_json"]),
                    constraints_json=row["constraints_json"],
                    task_image=str(row["task_image"]),
                    timeout_ms=int(timeout_ms) if timeout_ms is not None else None,
                )
            )
        return result

    # -- Writes --------------------------------------------------------------

    def insert(self, cur: TransactionCursor, params: TaskInsertParams) -> None:
        cur.execute(
            "INSERT INTO tasks("
            "task_id, job_id, task_index, state, error, exit_code, submitted_at_ms, started_at_ms, "
            "finished_at_ms, max_retries_failure, max_retries_preemption, failure_count, preemption_count, "
            "current_attempt_id, priority_neg_depth, priority_root_submitted_ms, "
            "priority_insertion, priority_band"
            ") VALUES (?, ?, ?, ?, NULL, NULL, ?, NULL, NULL, ?, ?, 0, 0, -1, ?, ?, ?, ?)",
            (
                params.task_id.to_wire(),
                params.job_id.to_wire(),
                params.task_index,
                params.state,
                params.submitted_at_ms,
                params.max_retries_failure,
                params.max_retries_preemption,
                params.priority_neg_depth,
                params.priority_root_submitted_ms,
                params.priority_insertion,
                params.priority_band,
            ),
        )

    def mark_assigned(
        self,
        cur: TransactionCursor,
        task_id: JobName,
        attempt_id: int,
        worker_id: WorkerId | None,
        worker_address: str | None,
        now_ms: int,
    ) -> None:
        if worker_id is not None:
            cur.execute(
                "UPDATE tasks SET state = ?, current_attempt_id = ?, "
                "current_worker_id = ?, current_worker_address = ?, "
                "started_at_ms = COALESCE(started_at_ms, ?) WHERE task_id = ?",
                (job_pb2.TASK_STATE_ASSIGNED, attempt_id, str(worker_id), worker_address, now_ms, task_id.to_wire()),
            )
            return
        cur.execute(
            "UPDATE tasks SET state = ?, current_attempt_id = ?, "
            "started_at_ms = COALESCE(started_at_ms, ?) WHERE task_id = ?",
            (job_pb2.TASK_STATE_ASSIGNED, attempt_id, now_ms, task_id.to_wire()),
        )

    def assign(
        self,
        cur: TransactionCursor,
        attempts: TaskAttemptStore,
        task_id: JobName,
        worker_id: WorkerId | None,
        worker_address: str | None,
        attempt_id: int,
        now_ms: int,
    ) -> None:
        attempts.insert(
            cur,
            TaskAttemptInsertParams(
                task_id=task_id,
                attempt_id=attempt_id,
                worker_id=worker_id,
                state=job_pb2.TASK_STATE_ASSIGNED,
                created_at_ms=now_ms,
            ),
        )
        self.mark_assigned(cur, task_id, attempt_id, worker_id, worker_address, now_ms)

    def apply_state_update(
        self,
        cur: TransactionCursor,
        params: TaskStateUpdateParams,
        active_states: set[int],
    ) -> None:
        if params.state in active_states:
            cur.execute(
                "UPDATE tasks SET state = ?, error = COALESCE(?, error), exit_code = COALESCE(?, exit_code), "
                "started_at_ms = COALESCE(started_at_ms, ?), finished_at_ms = ?, "
                "failure_count = ?, preemption_count = ? "
                "WHERE task_id = ?",
                (
                    params.state,
                    params.error,
                    params.exit_code,
                    params.started_at_ms,
                    params.finished_at_ms,
                    params.failure_count,
                    params.preemption_count,
                    params.task_id.to_wire(),
                ),
            )
            return
        cur.execute(
            "UPDATE tasks SET state = ?, error = COALESCE(?, error), exit_code = COALESCE(?, exit_code), "
            "started_at_ms = COALESCE(started_at_ms, ?), finished_at_ms = ?, "
            "failure_count = ?, preemption_count = ?, "
            "current_worker_id = NULL, current_worker_address = NULL "
            "WHERE task_id = ?",
            (
                params.state,
                params.error,
                params.exit_code,
                params.started_at_ms,
                params.finished_at_ms,
                params.failure_count,
                params.preemption_count,
                params.task_id.to_wire(),
            ),
        )

    def mark_terminal(
        self,
        cur: TransactionCursor,
        task_id: JobName,
        state: int,
        error: str | None,
        finished_at_ms: int | None,
        *,
        failure_count: int | None = None,
        preemption_count: int | None = None,
        active_states: set[int],
    ) -> None:
        if finished_at_ms is not None:
            set_clauses = ["state = ?", "error = ?", "finished_at_ms = COALESCE(finished_at_ms, ?)"]
        else:
            set_clauses = ["state = ?", "error = ?", "finished_at_ms = ?"]
        params: list[object] = [state, error, finished_at_ms]

        if failure_count is not None:
            set_clauses.append("failure_count = ?")
            params.append(failure_count)
        if preemption_count is not None:
            set_clauses.append("preemption_count = ?")
            params.append(preemption_count)
        if state not in active_states:
            set_clauses.append("current_worker_id = NULL")
            set_clauses.append("current_worker_address = NULL")

        params.append(task_id.to_wire())
        cur.execute(
            f"UPDATE tasks SET {', '.join(set_clauses)} WHERE task_id = ?",
            tuple(params),
        )

    def bulk_kill_non_terminal(
        self,
        cur: TransactionCursor,
        job_ids: Sequence[JobName],
        reason: str,
        finished_at_ms: int,
        terminal_states: set[int],
    ) -> None:
        if not job_ids:
            return
        wire_ids = [jid.to_wire() for jid in job_ids]
        job_placeholders = ",".join("?" for _ in wire_ids)
        terminal_placeholders = ",".join("?" for _ in terminal_states)
        cur.execute(
            f"UPDATE tasks SET state = ?, error = ?, finished_at_ms = COALESCE(finished_at_ms, ?), "
            "current_worker_id = NULL, current_worker_address = NULL "
            f"WHERE job_id IN ({job_placeholders}) AND state NOT IN ({terminal_placeholders})",
            (
                job_pb2.TASK_STATE_KILLED,
                reason,
                finished_at_ms,
                *wire_ids,
                *terminal_states,
            ),
        )

    def update_container_id(self, cur: TransactionCursor, task_id: JobName, container_id: str) -> None:
        cur.execute(
            "UPDATE tasks SET container_id = ? WHERE task_id = ?",
            (container_id, task_id.to_wire()),
        )

    def insert_resource_usage(self, cur: TransactionCursor, params: ResourceUsageInsertParams) -> None:
        cur.execute(
            "INSERT INTO task_resource_history"
            "(task_id, attempt_id, cpu_millicores, memory_mb, disk_mb, memory_peak_mb, timestamp_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                params.task_id.to_wire(),
                params.attempt_id,
                params.cpu_millicores,
                params.memory_mb,
                params.disk_mb,
                params.memory_peak_mb,
                params.timestamp_ms,
            ),
        )

    def insert_resource_usage_many(
        self,
        cur: TransactionCursor,
        params: Sequence[ResourceUsageInsertParams],
    ) -> None:
        if not params:
            return
        cur.executemany(
            "INSERT INTO task_resource_history"
            "(task_id, attempt_id, cpu_millicores, memory_mb, disk_mb, memory_peak_mb, timestamp_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    p.task_id.to_wire(),
                    p.attempt_id,
                    p.cpu_millicores,
                    p.memory_mb,
                    p.disk_mb,
                    p.memory_peak_mb,
                    p.timestamp_ms,
                )
                for p in params
            ],
        )

    def prune_resource_history(self) -> int:
        """Prune task_resource_history rows using TTL eviction and logarithmic downsampling."""
        now_ms = Timestamp.now().epoch_ms()
        ttl_cutoff_ms = now_ms - TASK_RESOURCE_HISTORY_TERMINAL_TTL.to_ms()
        terminal_placeholders = ",".join("?" for _ in TERMINAL_TASK_STATES)

        with self._db.read_snapshot() as snap:
            terminal_ids = [
                str(row["task_id"])
                for row in snap.fetchall(
                    f"SELECT task_id FROM tasks "
                    f"WHERE state IN ({terminal_placeholders}) "
                    f"AND finished_at_ms IS NOT NULL AND finished_at_ms < ?",
                    (*TERMINAL_TASK_STATES, ttl_cutoff_ms),
                )
            ]

        evicted_terminal = 0
        for chunk_start in range(0, len(terminal_ids), TASK_RESOURCE_HISTORY_DELETE_CHUNK):
            chunk = terminal_ids[chunk_start : chunk_start + TASK_RESOURCE_HISTORY_DELETE_CHUNK]
            placeholders = ",".join("?" * len(chunk))
            with self._db.transaction() as cur:
                cur.execute(f"DELETE FROM task_resource_history WHERE task_id IN ({placeholders})", tuple(chunk))
                evicted_terminal += cur.rowcount

        threshold = TASK_RESOURCE_HISTORY_RETENTION * 2
        with self._db.transaction() as cur:
            overflows = cur.execute(
                "SELECT task_id, attempt_id, COUNT(*) as cnt "
                "FROM task_resource_history "
                "GROUP BY task_id, attempt_id HAVING cnt > ?",
                (threshold,),
            ).fetchall()
            ids_to_delete: list[int] = []
            for row in overflows:
                task_id, attempt_id = row["task_id"], row["attempt_id"]
                all_ids = [
                    r["id"]
                    for r in cur.execute(
                        "SELECT id FROM task_resource_history WHERE task_id = ? AND attempt_id = ? ORDER BY id ASC",
                        (task_id, attempt_id),
                    ).fetchall()
                ]
                older = all_ids[: len(all_ids) - TASK_RESOURCE_HISTORY_RETENTION]
                ids_to_delete.extend(older[1::2])

            total_deleted = 0
            for chunk_start in range(0, len(ids_to_delete), 900):
                chunk = ids_to_delete[chunk_start : chunk_start + 900]
                placeholders = ",".join("?" * len(chunk))
                cur.execute(f"DELETE FROM task_resource_history WHERE id IN ({placeholders})", tuple(chunk))
                total_deleted += cur.rowcount
        if evicted_terminal > 0:
            logger.info("Evicted %d task_resource_history rows (terminal TTL)", evicted_terminal)
        if total_deleted > 0:
            logger.info("Pruned %d task_resource_history rows (log downsampling)", total_deleted)
        return evicted_terminal + total_deleted

    def _batch_prune_profiles(
        self,
        sql: str,
        params: tuple[object, ...],
        *,
        stopped: Callable[[], bool],
        pause_between_s: float,
    ) -> int:
        """Repeatedly delete one batch per transaction, sleeping between commits.

        Each iteration commits its batch before sleeping so the writer lock
        is released and other RPCs can interleave with pruning.
        """
        total = 0
        while not stopped():
            with self._db.transaction() as cur:
                batch = cur.execute(sql, params).rowcount
            if batch == 0:
                break
            total += batch
            time.sleep(pause_between_s)
        return total

    def prune_stale_profiles(
        self,
        *,
        cutoff_ms: int,
        stopped: Callable[[], bool],
        pause_between_s: float,
    ) -> int:
        """Delete ``task_profiles`` rows older than ``cutoff_ms`` in 1000-row batches."""
        return self._batch_prune_profiles(
            "DELETE FROM profiles.task_profiles WHERE rowid IN "
            "(SELECT rowid FROM profiles.task_profiles WHERE captured_at_ms < ? LIMIT 1000)",
            (cutoff_ms,),
            stopped=stopped,
            pause_between_s=pause_between_s,
        )

    def prune_orphan_profiles(
        self,
        *,
        stopped: Callable[[], bool],
        pause_between_s: float,
    ) -> int:
        """Delete ``task_profiles`` rows whose task has been pruned."""
        return self._batch_prune_profiles(
            "DELETE FROM profiles.task_profiles WHERE rowid IN "
            "(SELECT p.rowid FROM profiles.task_profiles p"
            " LEFT JOIN tasks t ON p.task_id = t.task_id"
            " WHERE t.task_id IS NULL LIMIT 1000)",
            (),
            stopped=stopped,
            pause_between_s=pause_between_s,
        )

    def set_state_for_test(
        self,
        cur: TransactionCursor,
        task_id: JobName,
        state: int,
        *,
        error: str | None = None,
        exit_code: int | None = None,
    ) -> None:
        """Test helper: overwrite ``state`` / ``error`` / ``exit_code`` directly.

        For non-active target states, also clears ``current_worker_id`` /
        ``current_worker_address`` so the row is consistent with production
        terminal-transition writes.
        """
        if state in ACTIVE_TASK_STATES:
            cur.execute(
                "UPDATE tasks SET state = ?, error = ?, exit_code = ? WHERE task_id = ?",
                (state, error, exit_code, task_id.to_wire()),
            )
            return
        cur.execute(
            "UPDATE tasks SET state = ?, error = ?, exit_code = ?, "
            "current_worker_id = NULL, current_worker_address = NULL WHERE task_id = ?",
            (state, error, exit_code, task_id.to_wire()),
        )


class TaskAttemptStore:
    """Task attempts."""

    def __init__(self, db: ControllerDB) -> None:
        self._db = db

    # -- Reads ---------------------------------------------------------------

    def get(self, tx: Tx, task_id: JobName, attempt_id: int) -> AttemptRow | None:
        row = tx.fetchone(
            f"SELECT {ATTEMPT_PROJECTION.select_clause()} FROM task_attempts ta "
            "WHERE ta.task_id = ? AND ta.attempt_id = ?",
            (task_id.to_wire(), attempt_id),
        )
        if row is None:
            return None
        return ATTEMPT_PROJECTION.decode_one([row])

    def get_state(self, tx: Tx, task_id: JobName, attempt_id: int) -> int | None:
        row = tx.fetchone(
            "SELECT state FROM task_attempts WHERE task_id = ? AND attempt_id = ?",
            (task_id.to_wire(), attempt_id),
        )
        return int(row["state"]) if row is not None else None

    def get_worker_id(self, tx: Tx, task_id: JobName, attempt_id: int) -> WorkerId | None:
        row = tx.fetchone(
            "SELECT worker_id FROM task_attempts WHERE task_id = ? AND attempt_id = ?",
            (task_id.to_wire(), attempt_id),
        )
        if row is None or row["worker_id"] is None:
            return None
        return WorkerId(str(row["worker_id"]))

    # -- Writes --------------------------------------------------------------

    def insert(self, cur: TransactionCursor, params: TaskAttemptInsertParams) -> None:
        cur.execute(
            "INSERT INTO task_attempts(task_id, attempt_id, worker_id, state, created_at_ms) VALUES (?, ?, ?, ?, ?)",
            (
                params.task_id.to_wire(),
                params.attempt_id,
                str(params.worker_id) if params.worker_id is not None else None,
                params.state,
                params.created_at_ms,
            ),
        )

    def mark_finished(
        self,
        cur: TransactionCursor,
        task_id: JobName,
        attempt_id: int,
        state: int,
        finished_at_ms: int,
        error: str | None,
    ) -> None:
        cur.execute(
            "UPDATE task_attempts SET state = ?, finished_at_ms = COALESCE(finished_at_ms, ?), error = ? "
            "WHERE task_id = ? AND attempt_id = ?",
            (state, finished_at_ms, error, task_id.to_wire(), attempt_id),
        )

    def apply_update(self, cur: TransactionCursor, params: TaskAttemptUpdateParams) -> None:
        cur.execute(
            "UPDATE task_attempts SET state = ?, started_at_ms = COALESCE(started_at_ms, ?), "
            "finished_at_ms = COALESCE(finished_at_ms, ?), exit_code = COALESCE(?, exit_code), "
            "error = COALESCE(?, error) WHERE task_id = ? AND attempt_id = ?",
            (
                params.state,
                params.started_at_ms,
                params.finished_at_ms,
                params.exit_code,
                params.error,
                params.task_id.to_wire(),
                params.attempt_id,
            ),
        )


class WorkerStore:
    """Workers, worker_attributes, worker_task_history, worker_resource_history."""

    def __init__(self, db: ControllerDB) -> None:
        self._db = db

    def active_healthy_address(self, tx: Tx, worker_id: WorkerId) -> str | None:
        row = tx.fetchone(
            "SELECT address FROM workers WHERE worker_id = ? AND active = 1 AND healthy = 1",
            (str(worker_id),),
        )
        return str(row["address"]) if row is not None else None

    def address(self, tx: Tx, worker_id: WorkerId) -> str | None:
        row = tx.fetchone("SELECT address FROM workers WHERE worker_id = ?", (str(worker_id),))
        return str(row["address"]) if row is not None else None

    def get_detail(self, tx: Tx, worker_id: WorkerId) -> WorkerDetailRow | None:
        row = tx.fetchone(
            f"SELECT {WORKER_DETAIL_PROJECTION.select_clause()} FROM workers w WHERE w.worker_id = ?",
            (str(worker_id),),
        )
        return WORKER_DETAIL_PROJECTION.decode_one([row]) if row is not None else None

    def get_active_status(self, tx: Tx, worker_id: WorkerId) -> ActiveWorkerStatus | None:
        """Return heartbeat info for an active worker, or None if missing/inactive."""
        row = tx.fetchone(
            "SELECT last_heartbeat_ms FROM workers WHERE worker_id = ? AND active = 1",
            (str(worker_id),),
        )
        if row is None:
            return None
        hb = row["last_heartbeat_ms"]
        return ActiveWorkerStatus(last_heartbeat_ms=int(hb) if hb is not None else None)

    def list_active_healthy(self, tx: Tx) -> dict[WorkerId, str]:
        """Return ``{worker_id: address}`` for all active+healthy workers."""
        rows = tx.fetchall("SELECT worker_id, address FROM workers WHERE active = 1 AND healthy = 1")
        return {WorkerId(str(row["worker_id"])): str(row["address"]) for row in rows}

    def list_active_by_ids(self, tx: Tx, worker_ids: Iterable[str]) -> list[WorkerDetailRow]:
        """Return :class:`WorkerDetailRow` for all active workers whose id is in ``worker_ids``."""
        ids = sorted({str(wid) for wid in worker_ids})
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = tx.fetchall(
            f"SELECT {WORKER_DETAIL_PROJECTION.select_clause()} "
            f"FROM workers w WHERE w.active = 1 AND w.worker_id IN ({placeholders})",
            tuple(ids),
        )
        return WORKER_DETAIL_PROJECTION.decode(rows)

    def filter_existing(self, tx: Tx, worker_ids: Iterable[WorkerId]) -> set[str]:
        """Return the subset of ``worker_ids`` (as strings) that have a ``workers`` row."""
        ids = [str(wid) for wid in worker_ids]
        if not ids:
            return set()
        placeholders = ",".join("?" for _ in ids)
        rows = tx.fetchall(
            f"SELECT worker_id FROM workers WHERE worker_id IN ({placeholders})",
            tuple(ids),
        )
        return {str(r["worker_id"]) for r in rows}

    def upsert(self, cur: TransactionCursor, params: WorkerUpsertParams) -> None:
        """Insert a new worker row or refresh every field of an existing one.

        On conflict the row is reset to healthy/active with zero
        consecutive_failures (registration re-establishes a worker as good).
        ``committed_*`` counters are left untouched because they reflect
        concurrent scheduling decisions, not registration metadata.
        """
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
                str(params.worker_id),
                params.address,
                params.last_heartbeat_ms,
                params.total_cpu_millicores,
                params.total_memory_bytes,
                params.total_gpu_count,
                params.total_tpu_count,
                params.device_type,
                params.device_variant,
                params.slice_id,
                params.scale_group,
                params.md_hostname,
                params.md_ip_address,
                params.md_cpu_count,
                params.md_memory_bytes,
                params.md_disk_bytes,
                params.md_tpu_name,
                params.md_tpu_worker_hostnames,
                params.md_tpu_worker_id,
                params.md_tpu_chips_per_host_bounds,
                params.md_gpu_count,
                params.md_gpu_name,
                params.md_gpu_memory_mb,
                params.md_gce_instance_name,
                params.md_gce_zone,
                params.md_git_hash,
                params.md_device_json,
            ),
        )

    def mark_unhealthy(self, cur: TransactionCursor, worker_id: WorkerId) -> None:
        cur.execute("UPDATE workers SET healthy = 0 WHERE worker_id = ?", (str(worker_id),))

    def record_task_assignment(
        self,
        cur: TransactionCursor,
        worker_id: WorkerId,
        task_id: JobName,
        now_ms: int,
    ) -> None:
        """Append a row to ``worker_task_history`` at task-assign time."""
        cur.execute(
            "INSERT INTO worker_task_history(worker_id, task_id, assigned_at_ms) VALUES (?, ?, ?)",
            (str(worker_id), task_id.to_wire(), now_ms),
        )

    def find_prunable(self, tx: Tx, before_ms: int) -> WorkerId | None:
        """Return one inactive-or-unhealthy worker whose heartbeat predates ``before_ms``."""
        row = tx.fetchone(
            "SELECT worker_id FROM workers " "WHERE (active = 0 OR healthy = 0) AND last_heartbeat_ms < ? LIMIT 1",
            (before_ms,),
        )
        return WorkerId(str(row["worker_id"])) if row is not None else None

    def set_health_for_test(self, cur: TransactionCursor, worker_id: WorkerId, healthy: bool) -> None:
        """Test helper: overwrite ``healthy`` and reset/raise ``consecutive_failures``."""
        cur.execute(
            "UPDATE workers SET healthy = ?, consecutive_failures = ? WHERE worker_id = ?",
            (1 if healthy else 0, 0 if healthy else 1, str(worker_id)),
        )

    def set_consecutive_failures_for_test(self, cur: TransactionCursor, worker_id: WorkerId, count: int) -> None:
        """Test helper: overwrite ``consecutive_failures`` directly."""
        cur.execute(
            "UPDATE workers SET consecutive_failures = ? WHERE worker_id = ?",
            (count, str(worker_id)),
        )

    def apply_snapshots(
        self,
        cur: TransactionCursor,
        items: Sequence[tuple[WorkerId, job_pb2.WorkerResourceSnapshot | None]],
        now_ms: int,
        *,
        reset_health: bool,
    ) -> None:
        """Bump ``last_heartbeat_ms`` for every worker; for entries with a
        snapshot also rewrite ``snapshot_*`` columns and append a
        ``worker_resource_history`` row.

        A ``None`` snapshot is a liveness-only update: the heartbeat path emits
        these for workers that skipped their resource refresh this cycle, and
        the ping path emits these on cycles where it skips the resource
        refresh.

        ``reset_health=True`` also clears ``healthy``/``active``/
        ``consecutive_failures`` because a successful heartbeat proves
        recovery. Ping path passes ``False`` — the ping loop tracks failures
        in-memory and removes workers via ``fail_workers_batch``.
        """
        if not items:
            return

        health_prefix = "healthy = 1, active = 1, consecutive_failures = 0, " if reset_health else ""

        liveness_only = [(now_ms, str(wid)) for wid, snap in items if snap is None]
        if liveness_only:
            cur.executemany(
                f"UPDATE workers SET {health_prefix}last_heartbeat_ms = ? WHERE worker_id = ?",
                liveness_only,
            )

        snapshot_binds = [
            {
                "worker_id": str(wid),
                "now_ms": now_ms,
                "host_cpu_percent": snap.host_cpu_percent,
                "memory_used_bytes": snap.memory_used_bytes,
                "memory_total_bytes": snap.memory_total_bytes,
                "disk_used_bytes": snap.disk_used_bytes,
                "disk_total_bytes": snap.disk_total_bytes,
                "running_task_count": snap.running_task_count,
                "total_process_count": snap.total_process_count,
                "net_recv_bps": snap.net_recv_bps,
                "net_sent_bps": snap.net_sent_bps,
            }
            for wid, snap in items
            if snap is not None
        ]
        if not snapshot_binds:
            return

        cur.executemany(
            f"UPDATE workers SET {health_prefix}last_heartbeat_ms = :now_ms, "
            "snapshot_host_cpu_percent = :host_cpu_percent, "
            "snapshot_memory_used_bytes = :memory_used_bytes, "
            "snapshot_memory_total_bytes = :memory_total_bytes, "
            "snapshot_disk_used_bytes = :disk_used_bytes, "
            "snapshot_disk_total_bytes = :disk_total_bytes, "
            "snapshot_running_task_count = :running_task_count, "
            "snapshot_total_process_count = :total_process_count, "
            "snapshot_net_recv_bps = :net_recv_bps, "
            "snapshot_net_sent_bps = :net_sent_bps "
            "WHERE worker_id = :worker_id",
            snapshot_binds,
        )
        cur.executemany(
            "INSERT INTO worker_resource_history ("
            "worker_id, snapshot_host_cpu_percent, snapshot_memory_used_bytes, "
            "snapshot_memory_total_bytes, snapshot_disk_used_bytes, snapshot_disk_total_bytes, "
            "snapshot_running_task_count, snapshot_total_process_count, "
            "snapshot_net_recv_bps, snapshot_net_sent_bps, timestamp_ms"
            ") VALUES ("
            ":worker_id, :host_cpu_percent, :memory_used_bytes, "
            ":memory_total_bytes, :disk_used_bytes, :disk_total_bytes, "
            ":running_task_count, :total_process_count, "
            ":net_recv_bps, :net_sent_bps, :now_ms"
            ")",
            snapshot_binds,
        )

    def add_committed_resources(
        self,
        cur: TransactionCursor,
        worker_id: WorkerId,
        resources: job_pb2.ResourceSpecProto,
    ) -> None:
        cur.execute(
            "UPDATE workers SET committed_cpu_millicores = committed_cpu_millicores + ?, "
            "committed_mem_bytes = committed_mem_bytes + ?, committed_gpu = committed_gpu + ?, "
            "committed_tpu = committed_tpu + ? WHERE worker_id = ?",
            (
                int(resources.cpu_millicores),
                int(resources.memory_bytes),
                int(get_gpu_count(resources.device)),
                int(get_tpu_count(resources.device)),
                str(worker_id),
            ),
        )

    def decommit_resources(
        self,
        cur: TransactionCursor,
        worker_id: WorkerId,
        resources: job_pb2.ResourceSpecProto,
    ) -> None:
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
                str(worker_id),
            ),
        )

    def replace_attributes(
        self,
        cur: TransactionCursor,
        worker_id: WorkerId,
        attrs: Sequence[WorkerAttributeParams],
    ) -> None:
        cur.execute("DELETE FROM worker_attributes WHERE worker_id = ?", (str(worker_id),))
        for attr in attrs:
            cur.execute(
                "INSERT INTO worker_attributes(worker_id, key, value_type, str_value, int_value, float_value) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (str(worker_id), attr.key, attr.value_type, attr.str_value, attr.int_value, attr.float_value),
            )

    def set_attribute_for_test(
        self,
        cur: TransactionCursor,
        worker_id: WorkerId,
        attr: WorkerAttributeParams,
    ) -> None:
        cur.execute(
            "INSERT INTO worker_attributes(worker_id, key, value_type, str_value, int_value, float_value) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(worker_id, key) DO UPDATE SET "
            "value_type=excluded.value_type, "
            "str_value=excluded.str_value, "
            "int_value=excluded.int_value, "
            "float_value=excluded.float_value",
            (str(worker_id), attr.key, attr.value_type, attr.str_value, attr.int_value, attr.float_value),
        )

    def update_attr_cache(self, worker_id: WorkerId, attrs: dict[str, AttributeValue]) -> None:
        self._db.set_worker_attributes(worker_id, attrs)

    def remove_from_attr_cache(self, worker_id: WorkerId) -> None:
        self._db.remove_worker_from_attr_cache(worker_id)

    def remove(self, cur: TransactionCursor, worker_id: WorkerId) -> None:
        cur.execute("UPDATE task_attempts SET worker_id = NULL WHERE worker_id = ?", (str(worker_id),))
        cur.execute("UPDATE tasks SET current_worker_id = NULL WHERE current_worker_id = ?", (str(worker_id),))
        cur.execute("DELETE FROM dispatch_queue WHERE worker_id = ?", (str(worker_id),))
        cur.execute("DELETE FROM workers WHERE worker_id = ?", (str(worker_id),))

    def prune_task_history(self) -> int:
        return self._prune_per_worker_history(
            "worker_task_history",
            WORKER_TASK_HISTORY_RETENTION,
            order_by="assigned_at_ms DESC, id DESC",
        )

    def prune_resource_history(self) -> int:
        return self._prune_per_worker_history(
            "worker_resource_history",
            WORKER_RESOURCE_HISTORY_RETENTION,
        )

    def _prune_per_worker_history(
        self,
        table: str,
        retention: int,
        order_by: str = "id DESC",
    ) -> int:
        with self._db.transaction() as cur:
            rows = cur.execute(
                f"SELECT worker_id, COUNT(*) as cnt FROM {table} GROUP BY worker_id HAVING cnt > ?",
                (retention,),
            ).fetchall()
            total_deleted = 0
            for row in rows:
                worker_id = row["worker_id"]
                cur.execute(
                    f"DELETE FROM {table} "
                    "WHERE worker_id = ? "
                    f"AND id NOT IN ("
                    f"  SELECT id FROM {table} "
                    "  WHERE worker_id = ? "
                    f"  ORDER BY {order_by} LIMIT ?"
                    ")",
                    (worker_id, worker_id, retention),
                )
                total_deleted += cur.rowcount
        if total_deleted > 0:
            logger.info("Pruned %d %s rows", total_deleted, table)
        return total_deleted


class DispatchQueueStore:
    """The dispatch_queue table."""

    def __init__(self, db: ControllerDB) -> None:
        self._db = db

    def enqueue_run(self, cur: TransactionCursor, worker_id: WorkerId, payload_proto: bytes, now_ms: int) -> None:
        cur.execute(
            "INSERT INTO dispatch_queue(worker_id, kind, payload_proto, task_id, created_at_ms) "
            "VALUES (?, 'run', ?, NULL, ?)",
            (str(worker_id), payload_proto, now_ms),
        )

    def enqueue_kill(self, cur: TransactionCursor, worker_id: WorkerId | None, task_id: str, now_ms: int) -> None:
        """Insert a kill entry. ``task_id`` is stored verbatim — direct-provider
        callers may pass non-canonical IDs (e.g. K8s pod-derived strings) and
        the dispatch_queue.task_id column is plain TEXT."""
        cur.execute(
            "INSERT INTO dispatch_queue(worker_id, kind, payload_proto, task_id, created_at_ms) "
            "VALUES (?, 'kill', NULL, ?, ?)",
            (str(worker_id) if worker_id is not None else None, task_id, now_ms),
        )

    def drain_direct_kills(self, cur: TransactionCursor) -> list[str]:
        rows = cur.execute(
            "SELECT task_id FROM dispatch_queue WHERE worker_id IS NULL AND kind = 'kill'",
        ).fetchall()
        tasks_to_kill = [str(row["task_id"]) for row in rows if row["task_id"] is not None]
        if rows:
            cur.execute("DELETE FROM dispatch_queue WHERE worker_id IS NULL AND kind = 'kill'")
        return tasks_to_kill


class ReservationStore:
    """Reservation claims and the meta(last_submission_ms) counter."""

    def __init__(self, db: ControllerDB) -> None:
        self._db = db

    def replace_claims(self, cur: TransactionCursor, claims: dict[WorkerId, tuple[str, int]]) -> None:
        cur.execute("DELETE FROM reservation_claims")
        for worker_id, (job_id, entry_idx) in claims.items():
            cur.execute(
                "INSERT INTO reservation_claims(worker_id, job_id, entry_idx) VALUES (?, ?, ?)",
                (str(worker_id), job_id, entry_idx),
            )

    def next_submission_ms(self, cur: TransactionCursor, submitted_ms: int) -> int:
        row = cur.execute("SELECT value FROM meta WHERE key = 'last_submission_ms'").fetchone()
        last_submission_ms = int(row["value"]) if row is not None else 0
        effective_submission_ms = max(submitted_ms, last_submission_ms + 1)
        if row is None:
            cur.execute("INSERT INTO meta(key, value) VALUES ('last_submission_ms', ?)", (effective_submission_ms,))
        else:
            cur.execute("UPDATE meta SET value = ? WHERE key = 'last_submission_ms'", (effective_submission_ms,))
        return effective_submission_ms


# =============================================================================
# ControllerStore
# =============================================================================


class ControllerStore:
    """Bundle of per-entity stores with direct access to transactions/snapshots."""

    def __init__(self, db: ControllerDB) -> None:
        self._db = db
        self.jobs = JobStore(db)
        self.tasks = TaskStore(db)
        self.attempts = TaskAttemptStore(db)
        self.workers = WorkerStore(db)
        self.endpoints = EndpointStore(db)
        self.dispatch = DispatchQueueStore(db)
        self.reservations = ReservationStore(db)
        # Caches reload after a checkpoint restore via db.replace_from(). The
        # hook fires only in that flow; normal startup loads caches in the
        # store constructors above.
        db.register_reopen_hook(self.endpoints._load_all)

    def transaction(self):
        return self._db.transaction()

    def read_snapshot(self):
        return self._db.read_snapshot()

    def optimize(self) -> None:
        self._db.optimize()
