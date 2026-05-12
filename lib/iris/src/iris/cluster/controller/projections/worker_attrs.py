# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""WorkerAttrsProjection — write-through in-memory cache over ``worker_attributes``.

Stage 7 of the SA Core migration. Mirrors :class:`EndpointsProjection`'s
atomicity model: every mutating method registers an ``on_commit`` hook on
the caller's :class:`TransactionCursor` so the dict update fires under the
DB write lock after the SQL commit. Rollbacks suppress the hook and the
dict stays in sync with disk.

Unlike :class:`EndpointsProjection`, mutating methods here do not issue
their own SQL — today the ``worker_attributes`` writes happen in
:class:`WorkerStore.replace_attributes`. This projection is responsible
only for the in-memory cache that ``healthy_active_workers_with_attributes``
reads on the scheduler hot path. Stage 11 may pull the SQL inside the
projection; for now the surface matches the legacy ``_attr_cache``
shape exactly so call sites change minimally.
"""

from __future__ import annotations

import logging
import threading
from typing import ClassVar

from iris.cluster.constraints import AttributeValue
from iris.cluster.controller.db import ControllerDB, TransactionCursor, _decode_attribute_rows
from iris.cluster.controller.projections import PROJECTIONS
from iris.cluster.controller.schema_v2 import worker_attributes_table
from iris.cluster.types import WorkerId

logger = logging.getLogger(__name__)


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
        """Reload the cache from SQL.

        Called once at construction and again after ``ControllerDB.replace_from``
        has swapped the underlying database file. Uses the legacy
        ``read_snapshot`` path; Stage 11 will switch to SA Core.
        """
        with self._db.read_snapshot() as q:
            rows = q.raw(
                "SELECT worker_id, key, value_type, str_value, int_value, float_value FROM worker_attributes",
            )
        decoded = _decode_attribute_rows(rows)
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
        cur: TransactionCursor,
        worker_id: WorkerId,
        attrs: dict[str, AttributeValue],
    ) -> None:
        """Schedule a dict update for ``worker_id``'s attributes after commit.

        Does not issue SQL — the corresponding ``worker_attributes`` writes
        are still emitted by :class:`WorkerStore.replace_attributes`. The
        new value replaces any prior entry for ``worker_id`` so the dict
        matches the SQL post-image.
        """
        snapshot = dict(attrs)

        def apply() -> None:
            with self._lock:
                self._cache[worker_id] = snapshot

        cur.on_commit(apply)

    def remove(self, cur: TransactionCursor, worker_id: WorkerId) -> None:
        """Schedule a dict pop for ``worker_id`` after commit."""

        def apply() -> None:
            with self._lock:
                self._cache.pop(worker_id, None)

        cur.on_commit(apply)

    def invalidate_for_worker(self, cur: TransactionCursor, worker_id: WorkerId) -> None:
        """Drop ``worker_id`` from the cache after commit (FK-cascade hook).

        Semantically distinct from :meth:`remove`: ``remove`` is used by the
        explicit worker-removal path, while ``invalidate_for_worker`` is the
        Stage 12 hook for callers that delete from ``workers`` and rely on
        the ``ON DELETE CASCADE`` to clear ``worker_attributes``.
        """
        self.remove(cur, worker_id)
