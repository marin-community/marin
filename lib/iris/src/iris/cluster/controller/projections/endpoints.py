# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""EndpointsProjection — write-through in-memory cache over the ``endpoints`` table.

TypeDecorators on ``endpoints_table`` (``JobNameType``, ``TimestampMsType``)
handle column decoding transparently so ``rehydrate`` can build
``EndpointRow`` directly from SA row attributes.

Atomicity model: mutating methods execute SQL inside the caller's ``Tx`` and
register an in-memory update via ``cur.register``. Hooks fire under the write
lock after COMMIT; a ROLLBACK suppresses them so the dicts stay in sync with
disk.
"""

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from threading import RLock
from typing import ClassVar

from rigging.timing import Timestamp
from sqlalchemy import bindparam, delete, insert, select

from iris.cluster.controller import db
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.projections import PROJECTIONS
from iris.cluster.controller.schema import endpoints_table, tasks_table
from iris.cluster.types import TERMINAL_TASK_STATES, EndpointAccess, JobName


@dataclass(frozen=True)
class EndpointQuery:
    endpoint_ids: tuple[str, ...] = ()
    name_prefix: str | None = None
    exact_name: str | None = None
    task_ids: tuple[JobName, ...] = ()
    limit: int | None = None


def access_from_db(value: int | None) -> int:
    """Decode a stored ``access`` column (NULL ⇒ PRIVATE) to an EndpointAccess value."""
    return EndpointAccess.ENDPOINT_ACCESS_PRIVATE if value is None else value


@dataclass(frozen=True, slots=True)
class EndpointRow:
    """Registered service endpoint (in-memory write-through cache row)."""

    endpoint_id: str
    name: str
    address: str
    task_id: JobName
    metadata: dict
    registered_at: Timestamp
    # Lease expiry; ``None`` never expires (only fixtures that skip leasing).
    # A passed deadline is hidden from reads and swept by ``sweep_expired``.
    lease_deadline: Timestamp | None = None
    # A Controller.EndpointAccess value; who may reach this endpoint via /proxy.
    access: int = EndpointAccess.ENDPOINT_ACCESS_PRIVATE

    def is_expired(self, now: Timestamp) -> bool:
        return self.lease_deadline is not None and self.lease_deadline <= now


logger = logging.getLogger(__name__)


# Module-level INSERT OR REPLACE. SA Core caches its compiled SQL across calls;
# the SQLite-dialect ``insert(...).on_conflict_do_update(...)`` form is *not*
# cacheable (``_generate_cache_key()`` returns None) and was re-compiling the
# statement once per row on burst writes. Safe to use INSERT OR REPLACE here
# because no other table holds an FK referencing ``endpoints``.
_INSERT_OR_REPLACE_ENDPOINT = insert(endpoints_table).prefix_with("OR REPLACE")

# Built once so the SELECT cache key is computed at import time; rebuilding it
# inside ``add()`` paid a ~50µs cache-key tax per call on burst writes.
_TASK_STATE_FOR_ENDPOINT = select(tasks_table.c.state, tasks_table.c.current_attempt_id).where(
    tasks_table.c.task_id == bindparam("task_id", type_=tasks_table.c.task_id.type)
)


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
    by an ``RLock``. Mutating methods accept a :class:`db.Tx`; the
    TypeDecorators on ``endpoints_table`` handle all encode/decode so no
    manual wire-format conversion appears here.
    """

    sources: ClassVar = (endpoints_table,)

    def __init__(self, db: ControllerDB) -> None:
        self._db = db
        self._lock = RLock()
        self._by_id: dict[str, EndpointRow] = {}
        # One name can map to multiple endpoint_ids — the schema does not
        # enforce uniqueness on ``name`` and the upsert keys off endpoint_id.
        self._by_name: dict[str, set[str]] = {}
        self._by_task: dict[JobName, set[str]] = {}
        PROJECTIONS.append(self)
        self.rehydrate()
        # Caches reload after a checkpoint restore via db.replace_from().
        db.register_reopen_hook(self.rehydrate)

    # -- Loading --------------------------------------------------------------

    def rehydrate(self) -> None:
        """Reload the dicts from SQL via the SA read engine.

        Called once at construction and again after ``ControllerDB.replace_from``
        has swapped the underlying database file. TypeDecorators on
        ``endpoints_table`` decode ``JobNameType`` and ``TimestampMsType``
        columns; ``metadata_json`` is decoded by the ``JSONDict`` TypeDecorator.
        """
        with self._lock:
            self._by_id.clear()
            self._by_name.clear()
            self._by_task.clear()
            with db.read_snapshot(self._db.sa_read_engine) as tx:
                for row in tx.execute(select(endpoints_table)).all():
                    endpoint = EndpointRow(
                        endpoint_id=row.endpoint_id,
                        name=row.name,
                        address=row.address,
                        task_id=row.task_id,
                        metadata=row.metadata_json,
                        registered_at=row.registered_at_ms,
                        lease_deadline=row.lease_deadline_ms,
                        access=access_from_db(row.access),
                    )
                    self._index(endpoint)
        logger.info("EndpointsProjection loaded %d endpoint(s) from DB", len(self._by_id))

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
        """Return live endpoint rows matching ``query``; all filters AND together.

        Rows whose lease has expired are treated as gone and never returned,
        even before the pruner sweeps them from storage.
        """
        now = Timestamp.now()
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
                if row.is_expired(now):
                    continue
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
        """Return any live endpoint with exact ``name``, or None. Used by the actor proxy."""
        now = Timestamp.now()
        with self._lock:
            ids = self._by_name.get(name)
            if not ids:
                return None
            # Arbitrary but stable pick — the original SQL did not specify ORDER BY.
            # Skip expired leases so a dead registrant's address is never served.
            for eid in ids:
                row = self._by_id[eid]
                if not row.is_expired(now):
                    return row
            return None

    def get(self, endpoint_id: str) -> EndpointRow | None:
        with self._lock:
            row = self._by_id.get(endpoint_id)
        if row is None or row.is_expired(Timestamp.now()):
            return None
        return row

    def all(self) -> list[EndpointRow]:
        now = Timestamp.now()
        with self._lock:
            return [row for row in self._by_id.values() if not row.is_expired(now)]

    # -- Writes ---------------------------------------------------------------

    def add(
        self,
        cur: db.Tx,
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
        - ``OK`` after a successful upsert; the in-memory index is updated
          via a post-commit hook.
        """
        task_id = endpoint.task_id
        job_id, _ = task_id.require_task()
        task_row = cur.execute(_TASK_STATE_FOR_ENDPOINT, {"task_id": task_id}).fetchone()
        if task_row is None:
            return AddEndpointOutcome.NOT_FOUND
        if int(task_row.state) in TERMINAL_TASK_STATES:
            return AddEndpointOutcome.TERMINAL
        if expected_attempt_id is not None and int(task_row.current_attempt_id) != int(expected_attempt_id):
            return AddEndpointOutcome.STALE_ATTEMPT

        cur.execute(
            _INSERT_OR_REPLACE_ENDPOINT,
            {
                "endpoint_id": endpoint.endpoint_id,
                "name": endpoint.name,
                "address": endpoint.address,
                "job_id": job_id,
                "task_id": task_id,
                "metadata_json": endpoint.metadata,
                "registered_at_ms": endpoint.registered_at,
                "lease_deadline_ms": endpoint.lease_deadline,
                "access": endpoint.access,
            },
        )

        def apply() -> None:
            with self._lock:
                # Replace: drop any previous row with this id first so the
                # name/task indexes stay consistent on overwrite.
                self._unindex(endpoint.endpoint_id)
                self._index(endpoint)

        cur.register(apply)
        return AddEndpointOutcome.OK

    def remove(self, cur: db.Tx, endpoint_id: str) -> EndpointRow | None:
        """Remove a single endpoint by id. Returns the removed row snapshot, if any."""
        existing = self.get(endpoint_id)
        if existing is None:
            return None
        cur.execute(delete(endpoints_table).where(endpoints_table.c.endpoint_id == endpoint_id))

        def apply() -> None:
            with self._lock:
                self._unindex(endpoint_id)

        cur.register(apply)
        return existing

    def remove_by_task(self, cur: db.Tx, task_id: JobName) -> list[str]:
        """Remove all endpoints owned by a task. Returns the removed endpoint_ids."""
        with self._lock:
            ids = list(self._by_task.get(task_id, ()))
        # Empty index means no committed rows (the index mirrors committed state
        # under the write lock), so the DELETE would be a no-op — skip it, as
        # remove() does.
        if not ids:
            return []
        cur.execute(delete(endpoints_table).where(endpoints_table.c.task_id == task_id))

        def apply() -> None:
            with self._lock:
                for eid in ids:
                    self._unindex(eid)

        cur.register(apply)
        return ids

    def remove_by_job_ids(self, cur: db.Tx, job_ids: Sequence[JobName]) -> list[str]:
        """Remove all endpoints owned by any of ``job_ids``. Returns the removed endpoint_ids."""
        if not job_ids:
            return []
        job_id_set = set(job_ids)
        with self._lock:
            to_remove: list[str] = []
            for row in self._by_id.values():
                owning_job, _ = row.task_id.require_task()
                if owning_job in job_id_set:
                    to_remove.append(row.endpoint_id)
        # The scan keys on the task's owning job; the DELETE keys on the persisted
        # endpoints.job_id (the same value, written at add() time), so both select the same rows.
        cur.execute(
            delete(endpoints_table).where(endpoints_table.c.job_id.in_(bindparam("job_ids", expanding=True))),
            {"job_ids": list(job_ids)},
        )
        if not to_remove:
            return []

        def apply() -> None:
            with self._lock:
                for eid in to_remove:
                    self._unindex(eid)

        cur.register(apply)
        return to_remove

    def sweep_expired(self, cur: db.Tx, now: Timestamp) -> list[str]:
        """Delete endpoints whose lease deadline has passed; return removed ids.

        Reads already hide expired rows, so this only reclaims storage, making
        the lease (not the FK ``CASCADE``) the GC trigger: a crashed task's
        endpoint expires here even while its task row still exists.
        """
        with self._lock:
            expired = [row.endpoint_id for row in self._by_id.values() if row.is_expired(now)]
        if not expired:
            return []
        cur.execute(
            delete(endpoints_table).where(endpoints_table.c.endpoint_id.in_(bindparam("ids", expanding=True))),
            {"ids": expired},
        )

        def apply() -> None:
            with self._lock:
                for eid in expired:
                    self._unindex(eid)

        cur.register(apply)
        return expired
