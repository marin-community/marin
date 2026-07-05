# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A type-keyed lookaside for a controller's per-controller memos.

A single :class:`~iris.cluster.controller.db.ControllerDB` owns one
:class:`CacheRegistry` and exposes it as ``db.caches``; every ``Tx`` the DB mints
mirrors the same registry as ``tx.caches``. So a write sink holding only a cursor
(``commit_effects``, ``purge_job``) reaches a cache the same way a reader holding
the DB does — no cache reference threaded through call signatures.

Caches register themselves by their concrete type and are retrieved by that type
(``caches[AttemptCountsProjection]``), so access is typed end-to-end: no string
keys, and the one unavoidable heterogeneous-container ``cast`` lives here rather
than at every call site. Contrast ``Tx.memo``, the per-*transaction*
string-keyed slot for values that must not outlive a single transaction; this
registry holds per-*controller* memos that persist across transactions.

Deliberately minimal: it stores whatever memo registers itself (today only
``AttemptCountsProjection``) without ``db.py`` needing to import — or name — the
concrete cache type, which keeps the raw storage layer free of upward deps.
"""

from typing import TypeVar, cast

T = TypeVar("T")


class CacheRegistry:
    """A ``{type: instance}`` map keyed and retrieved by the instance's own type."""

    def __init__(self) -> None:
        self._by_type: dict[type, object] = {}

    def register(self, cache: object) -> None:
        """Register ``cache`` under its concrete type, superseding any prior entry.

        Re-registration is intentional: reconstructing a memo over the same DB
        (e.g. a second test harness on one ``ControllerDB``) replaces the old
        entry so every lookup resolves to one canonical instance.
        """
        self._by_type[type(cache)] = cache

    def __getitem__(self, key: type[T]) -> T:
        # Registration keys by ``type(cache)``, so the stored object is always an
        # instance of ``key``; the cast just recovers that for the type checker.
        return cast(T, self._by_type[key])
