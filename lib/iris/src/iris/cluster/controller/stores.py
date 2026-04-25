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
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from threading import RLock

from iris.cluster.controller.db import ControllerDB, EndpointQuery, QuerySnapshot, TransactionCursor
from iris.cluster.controller.schema import (
    ENDPOINT_PROJECTION,
    JOB_CONFIG_JOIN,
    JOB_DETAIL_PROJECTION,
    EndpointRow,
    JobDetailRow,
)
from iris.cluster.types import TERMINAL_JOB_STATES, TERMINAL_TASK_STATES, JobName
from iris.rpc import job_pb2

logger = logging.getLogger(__name__)


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


class TaskAttemptStore:
    """Task attempts."""

    def __init__(self, db: ControllerDB) -> None:
        self._db = db


class WorkerStore:
    """Workers, worker_attributes, worker_task_history, worker_resource_history."""

    def __init__(self, db: ControllerDB) -> None:
        self._db = db


class DispatchQueueStore:
    """The dispatch_queue table."""

    def __init__(self, db: ControllerDB) -> None:
        self._db = db


class ReservationStore:
    """Reservation claims and the meta(last_submission_ms) counter."""

    def __init__(self, db: ControllerDB) -> None:
        self._db = db


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
