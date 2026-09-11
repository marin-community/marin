# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Projection concept: in-memory materialized views over controller DB tables.

A projection is in-memory state whose ground truth is one or more DB tables — a
materialized view. It declares the base tables it derives from, split by the
relationship that :func:`iris.cluster.controller.writes.validate` enforces:

- ``owns``: the projection is the sole writer of these tables; its mutating
  methods issue the SQL and post the updated in-memory image via a post-commit
  hook (eager write-through).
- ``watches``: the tables are written elsewhere; every writer must declare that
  it invalidates this projection, which then recomputes lazily on the next read
  (cache-aside / invalidate-and-recompute).

Among the controller's projections that single axis also fixes the maintenance
strategy — ``owns`` ⇒ eager, ``watches`` ⇒ lazy — so the declaration *is* the
strategy; there is no separate knob. Eviction policy (clear-at-N vs. LRU) is a
per-view concern, orthogonal to the concept.

Two neighbours are deliberately *not* projections:

- a **tracker** (``WorkerHealthTracker``) is in-memory state that is its own
  ground truth — no table backs it, so it cannot be rehydrated from one.
- a **memo** (``Tx.memo``) is transaction-lifetime state that may reflect the
  transaction's own *uncommitted* rows; no committed-state view can do that.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import ClassVar, Generic, TypeVar

from sqlalchemy import Table

from iris.cluster.controller.db import ControllerDB

K = TypeVar("K")


class Projection(ABC):
    """An in-memory materialized view over controller DB tables.

    Construction is the single wiring point: the projection registers itself into
    ``db.caches`` (reached everywhere by concrete type through any ``Tx`` or the
    ``ControllerDB`` — never threaded through call signatures), loads its initial
    state via :meth:`rehydrate`, and arranges to reload after a checkpoint restore
    swaps the DB file.
    """

    #: Tables this projection is the sole writer of (eager write-through).
    owns: ClassVar[tuple[Table, ...]] = ()
    #: Tables written elsewhere that this projection derives from (lazy invalidate).
    watches: ClassVar[tuple[Table, ...]] = ()

    def __init__(self, db: ControllerDB) -> None:
        self._db = db
        db.caches.register(self)
        self.rehydrate()
        db.register_reopen_hook(self.rehydrate)

    @abstractmethod
    def rehydrate(self) -> None:
        """Make in-memory state consistent with the current DB file.

        Called once at construction and again after ``ControllerDB.replace_from``
        installs a new file. An eager projection reloads from SQL; a lazy one drops
        its memo (and resets its :class:`LazyFillGuard`) so it refills against the
        new rows.
        """


class LazyFillGuard(Generic[K]):
    """Sequence-stamp guard against the stale-set race for a lazy projection.

    A lazy projection fills a read-miss by recomputing from the caller's read
    snapshot, which does not hold the write lock. Absent a guard, this interleaving
    installs a stale value: a reader opens its snapshot, a concurrent write commits
    and its post-commit hook drops the key, then the reader — still on the older
    snapshot — recomputes the pre-write value and stores it, where it survives until
    the key's next write. This is the "stale set" of *Scaling Memcache at Facebook*.

    The guard closes it by comparing the reader's ``tx.seq`` (the commit sequence
    sampled before its snapshot opened — a conservative lower bound on what the
    snapshot sees) against the commit seq of each key's most recent invalidation: a
    value may be cached only from a snapshot at least as fresh as that invalidation.
    A global ``floor``, advanced whenever the memo is cleared wholesale (eviction or
    checkpoint reopen), does the same for drops that aren't per-key.

    Not internally locked: the caller holds the projection's own lock across
    :meth:`may_store`, :meth:`note_invalidated`, and :meth:`reset`, so the guard's
    decisions stay consistent with the value dict they gate.
    """

    #: Cap on tracked per-key seqs; past it, fold into the floor instead of growing.
    _MAX_TRACKED: ClassVar[int] = 100_000

    def __init__(self) -> None:
        self._inval_seq: dict[K, int] = {}
        self._floor: int = 0

    def may_store(self, tx_seq: int, key: K) -> bool:
        """Whether a value for ``key`` recomputed at snapshot ``tx_seq`` may be cached."""
        return tx_seq >= self._floor and tx_seq >= self._inval_seq.get(key, 0)

    def note_invalidated(self, commit_seq: int, keys: Iterable[K]) -> None:
        """Record that ``keys`` were invalidated by the commit at ``commit_seq``.

        Called from the post-commit hook, where ``commit_seq`` is already ticked.
        Self-bounds: once the per-key map is full, fold it into the floor.
        """
        for key in keys:
            if len(self._inval_seq) >= self._MAX_TRACKED and key not in self._inval_seq:
                self.reset(commit_seq)
            self._inval_seq[key] = commit_seq

    def reset(self, commit_seq: int) -> None:
        """Drop per-key tracking; refuse any fill from a snapshot older than ``commit_seq``."""
        self._floor = commit_seq
        self._inval_seq.clear()
