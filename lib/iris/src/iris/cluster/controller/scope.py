# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ControllerScope — the cursor waist that recombines raw DB access with caches.

The controller's storage stack is a reverse hourglass: :class:`ControllerDB` is
raw SQLite (engines, write-lock, migrations) and knows nothing of tables,
projections, or caches; the typed tier (``schema``/``reads``/``writes``/
``projections``) fans up from it on the raw :class:`~iris.cluster.controller.db.Tx`
cursor; and this module is the narrow waist where the two recombine into a single
object the app threads.

:class:`ScopedTx` is that cursor: a raw ``Tx`` plus the per-controller projection
caches. :class:`ControllerScope` owns the ``ControllerDB`` and those projections
and hands out ``ScopedTx`` cursors via the same four snapshot managers the raw DB
exposes (it injects a ``ScopedTx`` factory through the DB's ``tx_factory`` seam,
so ``db.py`` never imports upward). Every read path and write chokepoint that
touches derived counts reaches them through ``cur.attempt_counts`` — no global
registry, no cache threaded as a constructor argument.

Only per-controller caches belong here. ``worker_attrs`` is per-backend and
``endpoints``' writers already hold it, so they stay where they are; the cursor
is simply the home a per-controller memo hangs off of when its writers would
otherwise have no way to reach it.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import cast

from sqlalchemy.engine import Connection

from iris.cluster.controller.db import ControllerDB, Tx
from iris.cluster.controller.projections.attempt_counts import AttemptCountsProjection


class ScopedTx(Tx):
    """A raw :class:`Tx` enriched with the controller's per-controller caches.

    Threaded through the app in place of the raw cursor; because it *is* a ``Tx``,
    every ``reads``/``writes``/projection signature typed ``tx: Tx`` accepts it
    unchanged. Code that needs derived counts uses ``cur.attempt_counts``.
    """

    def __init__(self, conn: Connection, attempt_counts: AttemptCountsProjection) -> None:
        super().__init__(conn)
        self.attempt_counts = attempt_counts


class ControllerScope:
    """Owns a :class:`ControllerDB` and its per-controller projections, and hands
    out :class:`ScopedTx` cursors.

    Mirrors the DB's four snapshot managers 1:1 so call sites migrate from
    ``self._db.<snapshot>()`` to ``self._scope.<snapshot>()`` with no body edits;
    :attr:`db` is the escape hatch for the raw-DB surface (checkpoint, engines,
    backup) that does not belong on the cursor.
    """

    def __init__(self, db: ControllerDB) -> None:
        self._db = db
        # Per-controller derived-count memo. Constructed here (registers its own
        # reopen hook on ``db``) and exposed on every ``ScopedTx`` this scope yields.
        self._attempt_counts = AttemptCountsProjection(db)

    @property
    def db(self) -> ControllerDB:
        """The raw DB, for surfaces that are not part of the cursor (engines,
        ``wal_checkpoint``, backup/restore)."""
        return self._db

    @property
    def attempt_counts(self) -> AttemptCountsProjection:
        """The derived-count projection (also reachable via any ``ScopedTx``)."""
        return self._attempt_counts

    def _cursor(self, conn: Connection) -> ScopedTx:
        return ScopedTx(conn, self._attempt_counts)

    @contextmanager
    def transaction(self) -> Iterator[ScopedTx]:
        with self._db.transaction(self._cursor) as tx:
            yield cast(ScopedTx, tx)

    @contextmanager
    def read_snapshot(self) -> Iterator[ScopedTx]:
        with self._db.read_snapshot(self._cursor) as tx:
            yield cast(ScopedTx, tx)

    @contextmanager
    def control_read_snapshot(self) -> Iterator[ScopedTx]:
        with self._db.control_read_snapshot(self._cursor) as tx:
            yield cast(ScopedTx, tx)

    @contextmanager
    def auth_read_snapshot(self) -> Iterator[ScopedTx]:
        with self._db.auth_read_snapshot(self._cursor) as tx:
            yield cast(ScopedTx, tx)
