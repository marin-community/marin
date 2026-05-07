# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for :class:`iris.cluster.controller.stores.SnapshotView`."""

from __future__ import annotations

import threading

import pytest
from iris.cluster.controller.stores import SnapshotView


class FakeClock:
    """Deterministic monotonic clock for tests.

    Drives ``SnapshotView`` TTL expiry without ``time.sleep``: tests advance
    the clock manually, so behavior is independent of wall time and CI load.
    """

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        self.now += dt


def test_first_read_calls_build_and_returns_value() -> None:
    calls = 0

    def build() -> str:
        nonlocal calls
        calls += 1
        return f"v{calls}"

    view = SnapshotView[str](name="t", ttl_s=60.0, build=build, clock=FakeClock())
    assert view.read() == "v1"
    assert calls == 1


def test_read_within_ttl_returns_cached_value() -> None:
    calls = 0

    def build() -> int:
        nonlocal calls
        calls += 1
        return calls

    clock = FakeClock()
    view = SnapshotView[int](name="t", ttl_s=60.0, build=build, clock=clock)
    assert view.read() == 1
    clock.advance(10.0)
    assert view.read() == 1
    clock.advance(10.0)
    assert view.read() == 1
    assert calls == 1


def test_read_past_ttl_rebuilds() -> None:
    calls = 0

    def build() -> int:
        nonlocal calls
        calls += 1
        return calls

    clock = FakeClock()
    view = SnapshotView[int](name="t", ttl_s=60.0, build=build, clock=clock)
    assert view.read() == 1
    clock.advance(60.0)
    assert view.read() == 2
    assert calls == 2


def test_invalidate_forces_rebuild() -> None:
    calls = 0

    def build() -> int:
        nonlocal calls
        calls += 1
        return calls

    # Clock starts at 0, so a backdate-based invalidate would not force a
    # rebuild within the TTL window. The flag-based implementation must.
    view = SnapshotView[int](name="t", ttl_s=60.0, build=build, clock=FakeClock())
    assert view.read() == 1
    view.invalidate()
    assert view.read() == 2


def test_concurrent_readers_share_one_rebuild() -> None:
    """Concurrent past-TTL reads must call ``build`` exactly once.

    The view's lock serializes rebuilds: the first reader runs ``build``;
    later readers wait for the lock, then observe the freshly-built value
    within TTL and skip rebuild. We assert on call count rather than
    instantaneous concurrency so the test does not depend on timing.
    """
    calls = 0
    call_lock = threading.Lock()
    # Gate the first build so the other 7 threads pile up on the view's lock
    # before it completes. This makes the "many threads enter read()
    # simultaneously" condition deterministic without sleeping.
    release_build = threading.Event()
    first_in_build = threading.Event()

    def build() -> int:
        nonlocal calls
        with call_lock:
            calls += 1
            my_call = calls
        if my_call == 1:
            first_in_build.set()
            release_build.wait()
        return my_call

    view = SnapshotView[int](name="t", ttl_s=60.0, build=build, clock=FakeClock())
    results: list[int] = [0] * 8
    barrier = threading.Barrier(8)

    def worker(i: int) -> None:
        barrier.wait()
        results[i] = view.read()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    # Block until the first build is in flight. The other 7 threads pile up
    # on the view's lock; we then release the build so it returns and they
    # observe the freshly-cached value. Timeouts guard against deadlock if
    # the view stops serializing rebuilds.
    assert first_in_build.wait(timeout=5.0), "first build never entered"
    release_build.set()
    for t in threads:
        t.join(timeout=5.0)
        assert not t.is_alive(), "worker thread did not finish"

    # Only the first thread ran ``build``; the rest got the cached value.
    assert calls == 1
    assert results == [1] * 8


def test_build_error_propagates_and_next_read_retries() -> None:
    calls = 0

    def build() -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("transient")
        return calls

    view = SnapshotView[int](name="t", ttl_s=60.0, build=build, clock=FakeClock())
    with pytest.raises(RuntimeError, match="transient"):
        view.read()
    # Cached value is still None, so the next read retries instead of returning stale.
    assert view.read() == 2
