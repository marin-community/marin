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
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from threading import RLock
from typing import ClassVar

from rigging.timing import Timestamp
from sqlalchemy import bindparam, delete, insert, select

from iris.cluster.controller import db
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.projections.base import Projection
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
    # Owning peer cluster id when this row is mirrored from a federated child; None
    # for a locally-registered endpoint. A remote row's ``address`` is the peer-side
    # bind (display only); /proxy forwards to the peer's controller instead.
    peer_id: str | None = None

    def is_expired(self, now: Timestamp) -> bool:
        return self.lease_deadline is not None and self.lease_deadline <= now


@dataclass(frozen=True, slots=True)
class EndpointDelta:
    """One committed, atomic delta to the endpoint projection."""

    upserts: tuple[EndpointRow, ...]
    deletes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EndpointReset:
    """Notification that the endpoint projection was rehydrated."""


@dataclass(slots=True)
class _PendingEndpointMutation:
    upserts: dict[str, EndpointRow] = field(default_factory=dict)
    deletes: set[str] = field(default_factory=set)

    def upsert(self, row: EndpointRow) -> None:
        self.deletes.discard(row.endpoint_id)
        self.upserts[row.endpoint_id] = row

    def delete(self, endpoint_id: str) -> None:
        self.upserts.pop(endpoint_id, None)
        self.deletes.add(endpoint_id)


logger = logging.getLogger(__name__)
_PENDING_MUTATION_KEY = "endpoints_projection.pending_mutation"


# Module-level INSERT OR REPLACE. SA Core caches its compiled SQL across calls;
# the SQLite-dialect ``insert(...).on_conflict_do_update(...)`` form is *not*
# cacheable (``_generate_cache_key()`` returns None) and was re-compiling the
# statement once per row on burst writes. Safe to use INSERT OR REPLACE here
# because no other table holds an FK referencing ``endpoints``.
_INSERT_OR_REPLACE_ENDPOINT = insert(endpoints_table).prefix_with("OR REPLACE")

# Built once so the SELECT cache key is computed at import time; rebuilding it
# inside ``add()`` paid a ~50µs cache-key tax per call on burst writes.
_TASK_STATE_FOR_ENDPOINT = select(tasks_table.c.state).where(
    tasks_table.c.task_id == bindparam("task_id", type_=tasks_table.c.task_id.type)
)


class AddEndpointOutcome(StrEnum):
    """Result of :meth:`EndpointsProjection.add`.

    String values are stable for logging; compare against the enum
    members rather than the literal strings.
    """

    OK = "ok"
    NOT_FOUND = "not_found"
    TERMINAL = "terminal"


class EndpointsProjection(Projection):
    """Process-local write-through cache over the ``endpoints`` table.

    Reads serve the latest committed state from in-memory dicts guarded
    by an ``RLock``. Mutating methods accept a :class:`db.Tx`; the
    TypeDecorators on ``endpoints_table`` handle all encode/decode so no
    manual wire-format conversion appears here.
    """

    owns: ClassVar = (endpoints_table,)

    def __init__(self, db: ControllerDB) -> None:
        self._lock = RLock()
        self._by_id: dict[str, EndpointRow] = {}
        # One name can map to multiple endpoint_ids — the schema does not
        # enforce uniqueness on ``name`` and the upsert keys off endpoint_id.
        self._by_name: dict[str, set[str]] = {}
        self._by_task: dict[JobName, set[str]] = {}
        self._listeners: list[Callable[[EndpointDelta | EndpointReset], None]] = []
        super().__init__(db)

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
            with self._db.read_snapshot() as tx:
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
                        peer_id=row.peer_id,
                    )
                    self._index(endpoint)
        logger.info("EndpointsProjection loaded %d endpoint(s) from DB", len(self._by_id))
        self._notify(EndpointReset())

    def subscribe(self, listener: Callable[[EndpointDelta | EndpointReset], None]) -> None:
        """Receive one notification for each committed transaction."""
        with self._lock:
            self._listeners.append(listener)

    def _pending_mutation(self, cur: db.Tx) -> _PendingEndpointMutation:
        existing = cur.memo.get(_PENDING_MUTATION_KEY)
        if existing is not None:
            assert isinstance(existing, _PendingEndpointMutation)
            return existing
        pending = _PendingEndpointMutation()
        cur.memo[_PENDING_MUTATION_KEY] = pending
        cur.register(lambda: self._apply_pending(pending))
        return pending

    def _apply_pending(self, pending: _PendingEndpointMutation) -> None:
        with self._lock:
            for endpoint_id in pending.deletes:
                self._unindex(endpoint_id)
            for row in pending.upserts.values():
                self._unindex(row.endpoint_id)
                self._index(row)
        self._notify(
            EndpointDelta(
                upserts=tuple(pending.upserts.values()),
                deletes=tuple(pending.deletes),
            )
        )

    def _notify(self, mutation: EndpointDelta | EndpointReset) -> None:
        with self._lock:
            listeners = tuple(self._listeners)
        for listener in listeners:
            listener(mutation)

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

    def resolve_all(self, name: str) -> list[EndpointRow]:
        """Every live (unexpired) endpoint row with exact ``name``.

        A name is not unique in the schema — a local registration and rows mirrored
        from one or more peers can share a name — so all live matches are returned,
        not just the first. Expired-lease rows are omitted."""
        now = Timestamp.now()
        with self._lock:
            ids = self._by_name.get(name, ())
            return [self._by_id[eid] for eid in ids if not self._by_id[eid].is_expired(now)]

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
    ) -> AddEndpointOutcome:
        """Insert ``endpoint`` into the DB and schedule the memory update.

        Task validation runs inside this transaction so the RPC handler does
        not need a separate read snapshot. Returns:

        - ``NOT_FOUND`` if the task row does not exist.
        - ``TERMINAL`` if the task is in a terminal state; registration is
          refused so an endpoint isn't served for a task that is already gone.
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
                "peer_id": endpoint.peer_id,
            },
        )

        self._pending_mutation(cur).upsert(endpoint)
        return AddEndpointOutcome.OK

    def replace_remote_for_peer(self, cur: db.Tx, peer_id: str, rows: Sequence[EndpointRow]) -> None:
        """Set-replace all endpoints mirrored from ``peer_id`` with ``rows``.

        The federation sync loop reports a peer's full current endpoint set every
        tick; this makes the parent's mirror match it. Rows previously mirrored under
        this peer but absent from ``rows`` are deleted; the rest are upserted. Every
        change registers a post-commit cache update so ``_by_name`` never serves a
        dropped remote row (raw CASCADE would desync the cache — see pruner).

        Each row must reference a task already mirrored on this controller (the job
        deltas in the same sync batch create them). A row whose task row is absent is
        skipped defensively — the FK would otherwise abort the whole sync batch.
        """
        with self._lock:
            existing_ids = {eid for eid, row in self._by_id.items() if row.peer_id == peer_id}

        present = self._present_task_ids(cur, [row.task_id for row in rows])
        # Stamp peer_id so the persisted column and the cached row never disagree,
        # whatever peer_id the caller's row objects carried.
        keep = [replace(row, peer_id=peer_id) for row in rows if row.task_id in present]
        for row in rows:
            if row.task_id not in present:
                logger.debug("skipping remote endpoint %s: task %s not mirrored yet", row.name, row.task_id)

        new_by_id = {row.endpoint_id: row for row in keep}
        stale = existing_ids - new_by_id.keys()
        if stale:
            cur.execute(
                delete(endpoints_table).where(endpoints_table.c.endpoint_id.in_(bindparam("ids", expanding=True))),
                {"ids": list(stale)},
            )
        for row in keep:
            job_id, _ = row.task_id.require_task()
            cur.execute(
                _INSERT_OR_REPLACE_ENDPOINT,
                {
                    "endpoint_id": row.endpoint_id,
                    "name": row.name,
                    "address": row.address,
                    "job_id": job_id,
                    "task_id": row.task_id,
                    "metadata_json": row.metadata,
                    "registered_at_ms": row.registered_at,
                    "lease_deadline_ms": row.lease_deadline,
                    "access": row.access,
                    "peer_id": peer_id,
                },
            )

        pending = self._pending_mutation(cur)
        for endpoint_id in stale:
            pending.delete(endpoint_id)
        for row in keep:
            pending.upsert(row)

    def _present_task_ids(self, cur: db.Tx, task_ids: Sequence[JobName]) -> set[JobName]:
        """Which of ``task_ids`` have a persisted task row (FK target for an endpoint)."""
        if not task_ids:
            return set()
        found = cur.execute(
            select(tasks_table.c.task_id).where(tasks_table.c.task_id.in_(bindparam("task_ids", expanding=True))),
            {"task_ids": list(task_ids)},
        ).all()
        return {r.task_id for r in found}

    def remove(self, cur: db.Tx, endpoint_id: str) -> EndpointRow | None:
        """Remove a single endpoint by id. Returns the removed row snapshot, if any."""
        existing = self.get(endpoint_id)
        if existing is None:
            return None
        cur.execute(delete(endpoints_table).where(endpoints_table.c.endpoint_id == endpoint_id))

        self._pending_mutation(cur).delete(endpoint_id)
        return existing

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

        pending = self._pending_mutation(cur)
        for endpoint_id in to_remove:
            pending.delete(endpoint_id)
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

        pending = self._pending_mutation(cur)
        for endpoint_id in expired:
            pending.delete(endpoint_id)
        return expired
