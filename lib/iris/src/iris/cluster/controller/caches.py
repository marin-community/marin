# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A type-keyed lookaside for a controller's per-controller memos.

A single :class:`~iris.cluster.controller.db.ControllerDB` owns one
:class:`CacheRegistry` and exposes it as ``db.caches``; every ``Tx`` the DB mints
mirrors the same registry as ``tx.caches``. So a write sink holding only a cursor
(``commit_effects``, ``delete_job``) reaches a cache the same way a reader holding
the DB does — no cache reference threaded through call signatures.

Caches register themselves by their concrete type and are retrieved by that type
(``caches[AttemptCountsProjection]``), so access is typed end-to-end: no string
keys, and the one unavoidable heterogeneous-container ``cast`` lives here rather
than at every call site. Contrast ``Tx.memo``, the per-*transaction*
string-keyed slot for values that must not outlive a single transaction; this
registry holds per-*controller* memos that persist across transactions.

Deliberately minimal: it stores whatever projection registers itself without
``db.py`` needing to import — or name — the concrete projection types, which keeps
the raw storage layer free of upward deps.

The registry also carries ``commit_seq``, the DB's monotonic write-commit counter.
It is the one piece of per-controller mutable state a lazy projection's
:class:`~iris.cluster.controller.projections.base.LazyFillGuard` needs, and it
lives here because the registry is already the per-DB object threaded onto every
``Tx``: ``write_transaction`` ticks it once per commit (under the write lock,
before invalidation hooks run) and a checkpoint restore ticks it before firing
reopen hooks, so every event that changes what a snapshot sees advances it.
"""

from collections.abc import Iterator
from typing import TypeVar, cast

T = TypeVar("T")


class CacheRegistry:
    """A ``{type: instance}`` map keyed and retrieved by the instance's own type."""

    def __init__(self) -> None:
        self._by_type: dict[type, object] = {}
        # Monotonic DB write-commit counter (see module docstring). Sampled onto
        # each ``Tx`` at mint as ``tx.seq`` and stamped by lazy-fill guards.
        self.commit_seq: int = 0

    def register(self, cache: object) -> None:
        """Register ``cache`` under its concrete type, superseding any prior entry.

        Re-registration is intentional: reconstructing a projection over the same
        DB (e.g. a second test harness on one ``ControllerDB``) replaces the old
        entry so every lookup resolves to one canonical instance.
        """
        self._by_type[type(cache)] = cache

    def tick(self) -> int:
        """Advance and return ``commit_seq`` (once per commit / per DB-file swap)."""
        self.commit_seq += 1
        return self.commit_seq

    def __getitem__(self, key: type[T]) -> T:
        # Registration keys by ``type(cache)``, so the stored object is always an
        # instance of ``key``; the cast just recovers that for the type checker.
        return cast(T, self._by_type[key])

    def __iter__(self) -> Iterator[object]:
        """Iterate the registered projection instances."""
        return iter(self._by_type.values())
