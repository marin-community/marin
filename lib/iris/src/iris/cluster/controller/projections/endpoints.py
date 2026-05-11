# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""EndpointsProjection — write-through in-memory cache over the ``endpoints`` table.

Stage 6 of the SA Core migration. Same atomicity model as the original
``EndpointStore``: writes execute SQL inside the caller's transaction and
schedule the in-memory update via ``TransactionCursor.on_commit``. Hooks
fire while the write lock is still held, so concurrent readers cannot
observe the new state until the SQL has committed and the dicts have been
updated. A rollback suppresses the hook and the dicts stay in sync with
disk.

Stage 11 will flip every write through the SA Core ``Tx`` wrapper; this
stage is intentionally a code-reorg + atomicity-test fortification under
the existing ``TransactionCursor`` mechanism.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from enum import StrEnum
from threading import RLock
from typing import ClassVar

from iris.cluster.controller.db import ControllerDB, EndpointQuery, TransactionCursor
from iris.cluster.controller.schema import ENDPOINT_PROJECTION, EndpointRow
from iris.cluster.controller.schema_v2 import endpoints_table
from iris.cluster.types import TERMINAL_TASK_STATES, JobName

logger = logging.getLogger(__name__)


# Module-level registry of all projection instances. Stage 12 will iterate
# this to enforce the @writes_to invariant; here we register every instance
# at construction so the registry is ready when later stages land.
PROJECTIONS: list[EndpointsProjection] = []


class AddEndpointOutcome(StrEnum):
    """Result of :meth:`EndpointsProjection.add`.

    String values are stable for logging; compare against the enum
    members rather than the literal strings.
    """

    OK = "ok"
    NOT_FOUND = "not_found"
    STALE_ATTEMPT = "stale_attempt"
    TERMINAL = "terminal"


class EndpointsProjection:
    """Process-local write-through cache over the ``endpoints`` table.

    Reads serve the latest committed state from in-memory dicts guarded
    by an ``RLock``. Writes execute SQL on the caller's
    :class:`TransactionCursor` and register an ``on_commit`` hook that
    re-indexes the dicts; the hook fires under the DB write lock after the
    SQL commit so readers cannot observe a torn state.

    Method signatures are identical to the legacy ``EndpointStore``: this
    stage is a code reorg and atomicity-test fortification, not a
    transaction-layer migration. Stage 11 swaps every mutating call site
    to the SA Core ``Tx`` wrapper in one commit.
    """

    sources: ClassVar = (endpoints_table,)

    def __init__(self, db: ControllerDB) -> None:
        self._db = db
        self._lock = RLock()
        self._by_id: dict[str, EndpointRow] = {}
        # One name can map to multiple endpoint_ids — the schema does not
        # enforce uniqueness on ``name`` and ``INSERT OR REPLACE`` keys off
        # endpoint_id.
        self._by_name: dict[str, set[str]] = {}
        self._by_task: dict[JobName, set[str]] = {}
        PROJECTIONS.append(self)
        self.rehydrate()
        # Caches reload after a checkpoint restore via db.replace_from().
        db.register_reopen_hook(self.rehydrate)

    # -- Loading --------------------------------------------------------------

    def rehydrate(self) -> None:
        """Reload the dicts from SQL.

        Called once at construction and again after ``ControllerDB.replace_from``
        has swapped the underlying database file. Uses the legacy
        ``read_snapshot`` path; Stage 11 will switch to SA Core.
        """
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
        logger.info("EndpointsProjection loaded %d endpoint(s) from DB", len(rows))

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

    def add(
        self,
        cur: TransactionCursor,
        endpoint: EndpointRow,
        *,
        expected_attempt_id: int | None = None,
    ) -> AddEndpointOutcome:
        """Insert ``endpoint`` into the DB and schedule the memory update.

        All task validation runs inside this transaction so the RPC handler
        does not need a separate read snapshot. Returns:

        - ``NOT_FOUND`` if the task row does not exist.
        - ``TERMINAL`` if the task is in a terminal state.
        - ``STALE_ATTEMPT`` if ``expected_attempt_id`` doesn't match the
          task's current attempt.
        - ``OK`` after a successful insert; the in-memory index is updated
          via a post-commit hook.
        """
        task_id = endpoint.task_id
        job_id, _ = task_id.require_task()
        row = cur.execute(
            "SELECT state, current_attempt_id FROM tasks WHERE task_id = ?", (task_id.to_wire(),)
        ).fetchone()
        if row is None:
            return AddEndpointOutcome.NOT_FOUND
        if int(row["state"]) in TERMINAL_TASK_STATES:
            return AddEndpointOutcome.TERMINAL
        if expected_attempt_id is not None and int(row["current_attempt_id"]) != int(expected_attempt_id):
            return AddEndpointOutcome.STALE_ATTEMPT

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
        return AddEndpointOutcome.OK

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
            # projection might not have observed yet (belt-and-suspenders
            # for the unlikely race of an in-flight concurrent writer).
            # This costs nothing on the common path.
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
