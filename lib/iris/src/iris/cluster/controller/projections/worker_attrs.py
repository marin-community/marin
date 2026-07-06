# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""WorkerAttrsProjection — write-through in-memory cache over ``worker_attributes``.

``WorkerIdType`` on the ``worker_id`` column decodes the string to ``WorkerId``
automatically. The ``value_type`` column has no TypeDecorator (it encodes a
three-way dispatch among int/float/str columns) so the decode branch is handled
explicitly by :func:`_decode_value`.

:meth:`WorkerAttrsProjection.set` is the sole writer of ``worker_attributes``
rows: it issues a DELETE for the worker followed by an INSERT for each attribute,
then registers a post-commit hook that atomically installs the new dict in the
in-memory cache. :meth:`WorkerAttrsProjection.invalidate_for_worker` handles the
FK-cascade case: deleting a ``workers`` row cascades into ``worker_attributes``
and the cache entry is dropped post-commit.
"""

import logging
import threading
from typing import ClassVar

from sqlalchemy import delete, insert, select

from iris.cluster.constraints import AttributeValue
from iris.cluster.controller.codec import WorkerAttributeRow, attribute_value_from_row
from iris.cluster.controller.db import ControllerDB, Tx
from iris.cluster.controller.projections.base import Projection
from iris.cluster.controller.schema import worker_attributes_table
from iris.cluster.types import WorkerId

logger = logging.getLogger(__name__)


def _attribute_value_cols(value: str | int | float) -> dict:
    if isinstance(value, int):
        return {"value_type": "int", "str_value": None, "int_value": int(value), "float_value": None}
    if isinstance(value, float):
        return {"value_type": "float", "str_value": None, "int_value": None, "float_value": float(value)}
    return {"value_type": "str", "str_value": str(value), "int_value": None, "float_value": None}


def _decode_value(row: WorkerAttributeRow) -> AttributeValue:
    return AttributeValue(attribute_value_from_row(row))


class WorkerAttrsProjection(Projection):
    """Process-local write-through cache over the ``worker_attributes`` table.

    Owns the table: :meth:`set` is the sole writer and issues DELETE + INSERT
    before registering the post-commit cache update. Reads serve the latest
    committed snapshot from an in-memory dict guarded by a ``threading.Lock``.
    The hook fires under the DB write lock so concurrent readers cannot observe
    torn state.
    """

    owns: ClassVar = (worker_attributes_table,)

    def __init__(self, db: ControllerDB) -> None:
        self._lock = threading.Lock()
        self._cache: dict[WorkerId, dict[str, AttributeValue]] = {}
        super().__init__(db)

    def rehydrate(self) -> None:
        decoded: dict[WorkerId, dict[str, AttributeValue]] = {}
        with self._db.read_snapshot() as tx:
            for row in tx.execute(select(worker_attributes_table)).all():
                decoded.setdefault(row.worker_id, {})[row.key] = _decode_value(row)
        with self._lock:
            self._cache.clear()
            self._cache.update(decoded)
        logger.info("WorkerAttrsProjection loaded attributes for %d worker(s) from DB", len(decoded))

    def get(self, worker_id: WorkerId) -> dict[str, AttributeValue]:
        """Return ``worker_id``'s attributes, or ``{}`` if none are recorded."""
        with self._lock:
            attrs = self._cache.get(worker_id)
            if attrs is None:
                return {}
            return dict(attrs)

    def all(self) -> dict[WorkerId, dict[str, AttributeValue]]:
        """Snapshot of every worker's attributes. Copies to avoid mutation leaks."""
        with self._lock:
            return {wid: dict(attrs) for wid, attrs in self._cache.items()}

    def set(self, cur: Tx, worker_id: WorkerId, attrs: dict[str, AttributeValue]) -> None:
        """Replace ``worker_id``'s attributes in SQL and update the cache at commit."""
        cur.execute(delete(worker_attributes_table).where(worker_attributes_table.c.worker_id == worker_id))
        if attrs:
            rows = [{"worker_id": worker_id, "key": key, **_attribute_value_cols(av.value)} for key, av in attrs.items()]
            cur.execute(insert(worker_attributes_table), rows)
        snapshot = dict(attrs)

        def apply() -> None:
            with self._lock:
                self._cache[worker_id] = snapshot

        cur.register(apply)

    def invalidate_for_worker(self, cur: Tx, worker_id: WorkerId) -> None:
        """Drop ``worker_id`` from the cache after commit (FK-cascade hook).

        Used by callers that delete from ``workers``; the ``ON DELETE CASCADE``
        clears ``worker_attributes`` in SQL and this call keeps the cache in sync.
        """

        def apply() -> None:
            with self._lock:
                self._cache.pop(worker_id, None)

        cur.register(apply)
