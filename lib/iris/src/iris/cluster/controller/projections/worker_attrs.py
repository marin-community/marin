# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""WorkerAttrsProjection — write-through in-memory cache over ``worker_attributes``.

``WorkerIdType`` on the ``worker_id`` column decodes the string to ``WorkerId``
automatically. The ``value_type`` column has no TypeDecorator (it encodes a
three-way dispatch among int/float/str columns) so the decode branch is handled
explicitly by :func:`_decode_value`.

Unlike :class:`EndpointsProjection`, mutating methods do not issue SQL — the
corresponding ``worker_attributes`` writes are emitted by
``writes.workers.replace_attributes``. This projection owns only the in-memory
cache that ``healthy_active_workers_with_attributes`` reads on the scheduler hot
path.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import ClassVar, Protocol

from sqlalchemy import select

from iris.cluster.constraints import AttributeValue
from iris.cluster.controller import db
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.projections import PROJECTIONS
from iris.cluster.controller.schema import worker_attributes_table
from iris.cluster.types import WorkerId


class PostCommitRegistrar(Protocol):
    """Structural type for any transaction wrapper that schedules post-commit hooks.

    :class:`db.Tx` exposes a ``register(callable)`` method that fires after
    the surrounding write transaction commits, under the write lock. Projection
    invalidation methods accept this Protocol so the hook works for any
    transaction wrapper that follows the same shape.
    """

    def register(self, hook: Callable[[], None]) -> None: ...


logger = logging.getLogger(__name__)


def _decode_value(row) -> AttributeValue:
    """Decode a single ``worker_attributes`` SA row to an ``AttributeValue``.

    ``value_type`` is a plain string column (CHECK 'str'/'int'/'float');
    no TypeDecorator handles the three-way dispatch so it is done here.
    """
    if row.value_type == "int":
        return AttributeValue(int(row.int_value))
    if row.value_type == "float":
        return AttributeValue(float(row.float_value))
    return AttributeValue(str(row.str_value or ""))


class WorkerAttrsProjection:
    """Process-local write-through cache over the ``worker_attributes`` table.

    Reads serve the latest committed snapshot from an in-memory dict guarded
    by a ``threading.Lock``. Writes register an ``on_commit`` hook that
    updates the dict atomically with the surrounding SQL commit; the hook
    fires under the DB write lock so concurrent readers cannot observe
    torn state.
    """

    sources: ClassVar = (worker_attributes_table,)

    def __init__(self, db: ControllerDB) -> None:
        self._db = db
        self._lock = threading.Lock()
        self._cache: dict[WorkerId, dict[str, AttributeValue]] = {}
        PROJECTIONS.append(self)
        self.rehydrate()
        # Caches reload after a checkpoint restore via db.replace_from().
        db.register_reopen_hook(self.rehydrate)

    # -- Loading --------------------------------------------------------------

    def rehydrate(self) -> None:
        """Reload the cache from SQL via the SA read engine.

        Called once at construction and again after ``ControllerDB.replace_from``
        has swapped the underlying database file. ``WorkerIdType`` on
        ``worker_id`` decodes the string automatically; ``value_type`` dispatch
        is handled by :func:`_decode_value`.
        """
        decoded: dict[WorkerId, dict[str, AttributeValue]] = {}
        with db.read_snapshot(self._db.sa_read_engine) as tx:
            for row in tx.execute(select(worker_attributes_table)).all():
                decoded.setdefault(row.worker_id, {})[row.key] = _decode_value(row)
        with self._lock:
            self._cache.clear()
            self._cache.update(decoded)
        logger.info("WorkerAttrsProjection loaded attributes for %d worker(s) from DB", len(decoded))

    # -- Reads ----------------------------------------------------------------

    def get(self, worker_id: WorkerId) -> dict[str, AttributeValue]:
        """Return ``worker_id``'s attributes, or ``{}`` if none are recorded."""
        with self._lock:
            attrs = self._cache.get(worker_id)
            if attrs is None:
                return {}
            # Copy so callers cannot mutate the cached dict.
            return dict(attrs)

    def all(self) -> dict[WorkerId, dict[str, AttributeValue]]:
        """Snapshot of every worker's attributes. Copies to avoid mutation leaks."""
        with self._lock:
            return {wid: dict(attrs) for wid, attrs in self._cache.items()}

    # -- Writes ---------------------------------------------------------------

    def set(
        self,
        cur: PostCommitRegistrar,
        worker_id: WorkerId,
        attrs: dict[str, AttributeValue],
    ) -> None:
        """Schedule a dict update for ``worker_id``'s attributes after commit.

        Does not issue SQL — the corresponding ``worker_attributes`` writes
        are still emitted by `writes.workers.replace_attributes`. The
        new value replaces any prior entry for ``worker_id`` so the dict
        matches the SQL post-image.
        """
        snapshot = dict(attrs)

        def apply() -> None:
            with self._lock:
                self._cache[worker_id] = snapshot

        cur.register(apply)

    def remove(self, cur: PostCommitRegistrar, worker_id: WorkerId) -> None:
        """Schedule a dict pop for ``worker_id`` after commit."""

        def apply() -> None:
            with self._lock:
                self._cache.pop(worker_id, None)

        cur.register(apply)

    def invalidate_for_worker(self, tx: PostCommitRegistrar, worker_id: WorkerId) -> None:
        """Drop ``worker_id`` from the cache after commit (FK-cascade hook).

        Semantically distinct from :meth:`remove`: ``remove`` is used by the
        explicit worker-removal path, while ``invalidate_for_worker`` is the
        hook for callers that delete from ``workers`` and rely on the
        ``ON DELETE CASCADE`` to clear ``worker_attributes``.
        """

        def apply() -> None:
            with self._lock:
                self._cache.pop(worker_id, None)

        tx.register(apply)
