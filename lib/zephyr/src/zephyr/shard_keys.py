# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic hashing and sort-key helpers for Zephyr shard routing."""

from __future__ import annotations

from collections.abc import Callable

import msgspec
import xxhash


def deterministic_hash(obj: object) -> int:
    """Compute a deterministic hash for an object."""
    s = msgspec.msgpack.encode(obj, order="deterministic")
    return xxhash.xxh3_64_intdigest(s)


def composite_sort_key(key_fn: Callable, sort_fn: Callable | None) -> Callable:
    """Build a merge/sort key from a grouping key and an optional secondary sort.

    Returns ``key_fn`` unchanged when ``sort_fn`` is None. Otherwise returns a
    callable producing ``(key_fn(item), sort_fn(item))`` so items order first by
    group key and then by the secondary key; grouping should still use
    ``key_fn`` alone. Used by both the scatter writer (pre-sort within a chunk)
    and the reduce-side k-way merge so the two stay consistent.
    """
    if sort_fn is None:
        return key_fn
    # Bind to a non-Optional local so the closure captures a narrowed type.
    secondary = sort_fn
    return lambda item: (key_fn(item), secondary(item))
