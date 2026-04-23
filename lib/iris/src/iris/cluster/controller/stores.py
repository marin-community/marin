# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed store layer over :mod:`iris.cluster.controller.db`.

Stores group related SQL against a single entity (jobs, tasks, workers,
endpoints, ...) and expose a typed API that callers invoke inside an open
transaction (read or write). :class:`ControllerStore` bundles every per-entity
store and forwards ``transaction()`` / ``read_snapshot()`` to the underlying
:class:`ControllerDB`.

Dependency chain::

    db.py        — connections, migrations, transaction context managers
    schema.py    — table DDL, row dataclasses, projections
    stores.py    — depends on { db, schema }; per-entity stores
    transitions.py — depends on stores; never calls db.py directly

Stores are the only place outside of ``db.py`` / ``schema.py`` that build
SQL strings for the controller tables. ``transitions.py`` uses the store
API; other callers (``service.py``, ``controller.py``) are migrated
later as the pattern proves out.

The layer is introduced incrementally. Phase 1 (this module as it stands)
adds the scaffolding and folds the previous ``EndpointRegistry`` in as
:class:`EndpointStore`. Subsequent phases move per-entity SQL out of
``transitions.py`` into the relevant store class.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from threading import RLock

from iris.cluster.controller.db import ControllerDB, EndpointQuery, QuerySnapshot, TransactionCursor
from iris.cluster.controller.schema import ENDPOINT_PROJECTION, EndpointRow
from iris.cluster.types import TERMINAL_TASK_STATES, JobName

logger = logging.getLogger(__name__)


# Store read methods accept either a write cursor or a read snapshot. Writes
# require ``TransactionCursor`` explicitly so static typing prevents issuing
# mutations through a read-only snapshot.
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


class JobStore:
    """Jobs, job_config, users, user_budgets, job_workdir_files."""

    def __init__(self, db: ControllerDB) -> None:
        self._db = db


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
