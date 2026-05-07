# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for :class:`iris.cluster.controller.stores.SnapshotView`."""

from __future__ import annotations

import threading
import time

import pytest
from iris.cluster.controller.stores import SnapshotView


def test_first_read_calls_build_and_returns_value() -> None:
    calls = 0

    def build() -> str:
        nonlocal calls
        calls += 1
        return f"v{calls}"

    view = SnapshotView[str](name="t", ttl_s=60.0, build=build)
    assert view.read() == "v1"
    assert calls == 1


def test_read_within_ttl_returns_cached_value() -> None:
    calls = 0

    def build() -> int:
        nonlocal calls
        calls += 1
        return calls

    view = SnapshotView[int](name="t", ttl_s=60.0, build=build)
    assert view.read() == 1
    assert view.read() == 1
    assert view.read() == 1
    assert calls == 1


def test_read_past_ttl_rebuilds() -> None:
    calls = 0

    def build() -> int:
        nonlocal calls
        calls += 1
        return calls

    view = SnapshotView[int](name="t", ttl_s=0.05, build=build)
    assert view.read() == 1
    time.sleep(0.07)
    assert view.read() == 2
    assert calls == 2


def test_invalidate_forces_rebuild() -> None:
    calls = 0

    def build() -> int:
        nonlocal calls
        calls += 1
        return calls

    view = SnapshotView[int](name="t", ttl_s=60.0, build=build)
    assert view.read() == 1
    view.invalidate()
    assert view.read() == 2


def test_concurrent_readers_share_one_rebuild() -> None:
    """Past-TTL reads from many threads must not all run ``build`` in parallel.

    The view serializes rebuilds on its lock; concurrent callers wait for the
    in-flight rebuild to complete and observe the freshly-built value.
    """
    calls = 0
    builds_in_flight = 0
    max_concurrent = 0
    lock = threading.Lock()
    barrier = threading.Barrier(8)

    def build() -> int:
        nonlocal calls, builds_in_flight, max_concurrent
        with lock:
            calls += 1
            builds_in_flight += 1
            max_concurrent = max(max_concurrent, builds_in_flight)
        # Hold long enough that any racing readers would observe concurrency
        # if the view didn't serialize them.
        time.sleep(0.05)
        with lock:
            builds_in_flight -= 1
        return calls

    view = SnapshotView[int](name="t", ttl_s=60.0, build=build)
    results: list[int] = [0] * 8

    def worker(i: int) -> None:
        barrier.wait()
        results[i] = view.read()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert max_concurrent == 1, f"expected serialized rebuild, saw {max_concurrent} concurrent"
    # All workers see the same value (whatever rebuild number won the race).
    assert len(set(results)) == 1


def test_build_error_propagates_and_next_read_retries() -> None:
    calls = 0

    def build() -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("transient")
        return calls

    view = SnapshotView[int](name="t", ttl_s=60.0, build=build)
    with pytest.raises(RuntimeError, match="transient"):
        view.read()
    # Cached value is still None, so the next read retries instead of returning stale.
    assert view.read() == 2
