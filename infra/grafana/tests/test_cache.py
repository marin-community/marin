# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavioural tests for the bridge's result cache."""

import threading

from cache import TtlCache


def test_returns_cached_value_within_the_ttl():
    cache: TtlCache[int] = TtlCache(ttl=60.0)
    calls = []

    def compute():
        calls.append(1)
        return 42

    assert cache.get_or_compute("k", compute) == 42
    assert cache.get_or_compute("k", compute) == 42
    assert len(calls) == 1


def test_recomputes_once_the_ttl_has_passed():
    cache: TtlCache[int] = TtlCache(ttl=0.0)
    assert cache.get_or_compute("k", lambda: 1) == 1
    assert cache.get_or_compute("k", lambda: 2) == 2


def test_concurrent_misses_on_one_key_compute_once():
    # N viewers opening the same dashboard at once should cost one finelog query,
    # not N — the coalescing is the point, not just the TTL.
    #
    # Held deterministic by pinning the order: the first caller is inside compute
    # (holding the key's lock) before the others start, so they must contend for a
    # key whose value does not exist yet. Without coalescing each would compute.
    cache: TtlCache[int] = TtlCache(ttl=60.0)
    computing = threading.Event()
    release = threading.Event()
    calls: list[int] = []
    results: list[int] = []

    def compute():
        calls.append(1)
        computing.set()
        release.wait(timeout=5)
        return 7

    def worker():
        results.append(cache.get_or_compute("k", compute))

    first = threading.Thread(target=worker)
    first.start()
    assert computing.wait(timeout=5), "first caller never entered compute"

    others = [threading.Thread(target=worker) for _ in range(3)]
    for t in others:
        t.start()
    release.set()
    for t in [first, *others]:
        t.join(timeout=10)

    assert len(calls) == 1
    assert results == [7, 7, 7, 7]


def test_expired_entries_are_evicted_rather_than_accumulating():
    # Cache keys embed a rotating time bucket, so an insert-only cache would grow
    # without bound on a process that is deliberately never restarted. At ttl=0
    # every entry is stale the moment it lands, so 50 distinct keys must leave
    # nothing behind — an unpruned cache would hold all 50.
    cache: TtlCache[int] = TtlCache(ttl=0.0)
    for i in range(50):
        cache.get_or_compute(f"bucket-{i}", lambda i=i: i)
    assert len(cache) == 0


def test_live_entries_survive_eviction():
    cache: TtlCache[int] = TtlCache(ttl=60.0)
    cache.get_or_compute("a", lambda: 1)
    cache.get_or_compute("b", lambda: 2)
    assert len(cache) == 2
    assert cache.get_or_compute("a", lambda: 99) == 1
