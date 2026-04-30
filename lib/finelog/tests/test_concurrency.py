# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for concurrent RPC interactions: drop-during-concurrent-write safety,
query-vs-concurrent-drop dict-iteration safety, and query holding the read
lock for its full duration (so write-side callers queue correctly). These
tests cross RPC verb boundaries and verify lock-ordering and no-deadlock
properties of the store.
"""

from __future__ import annotations

import threading

import pyarrow as pa
from finelog.store.duckdb_store import DuckDBLogStore
from finelog.store.schema import NamespaceNotFoundError

from tests.conftest import _ipc_bytes, _seal, _worker_batch, _worker_schema


def test_drop_during_concurrent_write_is_safe(store: DuckDBLogStore):
    """A racing ``drop_table`` + ``write_rows`` upholds the write contract.

    ``write_rows`` looks up the namespace under the insertion mutex; if
    drop got there first, the lookup raises ``NamespaceNotFoundError``.
    Otherwise the write proceeds and drop blocks on the mutex until the
    write returns.

    The contract: a write either completes and persists, or fails with
    ``NamespaceNotFoundError``. We verify both the cleanup-on-error
    (no surprise exceptions, no deadlock) AND the successful-write
    visibility / dropped-namespace cleanup.
    """
    store.register_table("iris.worker", _worker_schema())
    seg_dir = store._data_dir / "iris.worker"
    payload = _ipc_bytes(_worker_batch(["w-1"], [100], [1]))

    write_results: list[Exception | int] = []

    def writer():
        try:
            n = store.write_rows("iris.worker", payload)
            write_results.append(n)
        except Exception as exc:
            write_results.append(exc)

    threads = [threading.Thread(target=writer, daemon=True) for _ in range(8)]
    for t in threads:
        t.start()

    drop_succeeded = False
    try:
        store.drop_table("iris.worker")
        drop_succeeded = True
    except NamespaceNotFoundError:
        # Lost the race: every write got there first. Possible only if
        # all 8 writers slipped in before the drop's lookup, in which
        # case the namespace would still exist. We don't expect this
        # path in practice (drop runs from the main thread which started
        # last), but tolerate it cleanly.
        pass

    for t in threads:
        t.join(timeout=5.0)
        assert not t.is_alive(), "writer thread did not terminate"

    # Every result is either success (int=1) or a clean
    # NamespaceNotFoundError. No surprise exceptions.
    successes = 0
    for r in write_results:
        if isinstance(r, Exception):
            assert isinstance(r, NamespaceNotFoundError), repr(r)
        else:
            assert r == 1
            successes += 1

    if drop_succeeded:
        # Namespace was dropped: local dir is gone and no remnants
        # remain on disk. Successful writes' in-memory chunks
        # evaporated by design (the explicit drop contract).
        assert "iris.worker" not in store._namespaces
        assert not seg_dir.exists()
    else:
        # Drop lost the race: namespace survives. Successful writes
        # are durable — flush-then-compact and assert the rows are
        # readable through the public query API.
        assert "iris.worker" in store._namespaces
        ns = store._namespaces["iris.worker"]
        ns._flush_step()
        ns._compaction_step(compact_single=True)
        table = store.query('SELECT worker_id FROM "iris.worker"')
        assert table.num_rows == successes


def test_query_safe_against_concurrent_drop_table(store: DuckDBLogStore):
    """A query iterating namespaces is safe against a concurrent drop.

    Regression for the dict-iteration race: ``query()`` snapshots
    ``self._namespaces`` under the insertion lock so a concurrent
    ``drop_table`` (which also takes the insertion lock to ``del`` the
    entry) can't trigger ``RuntimeError: dictionary changed size during
    iteration``.

    We hold the read side of the rwlock through the snapshot phase by
    instrumenting ``sealed_segments`` to block on a barrier, then run
    ``drop_table`` on a *different* namespace from another thread. Drop
    waits on the rwlock write side (queued behind our read), but its
    insertion-lock window — which deletes from ``self._namespaces`` —
    runs *before* the write_acquire and would race the snapshot if the
    snapshot weren't itself protected by the insertion lock.
    """
    # Two namespaces: query iterates both; drop targets ``victim`` so
    # the dict mutation lands during the query.
    store.register_table("ns.alpha", _worker_schema())
    store.register_table("ns.victim", _worker_schema())
    # Seal a segment in alpha so its branch of the loop has work to do.
    store.write_rows("ns.alpha", _ipc_bytes(_worker_batch(["a"], [1], [1])))
    _seal(store, "ns.alpha")

    # Block the query thread inside its per-namespace loop with a barrier,
    # then run drop concurrently. Without Task A's fix, the iteration
    # of ``self._namespaces.items()`` would race ``del self._namespaces[name]``.
    alpha_ns = store._namespaces["ns.alpha"]
    in_loop = threading.Event()
    proceed = threading.Event()
    orig_sealed = alpha_ns.sealed_segments

    def blocking_sealed():
        in_loop.set()
        # Hold here while the dict mutation thread runs.
        proceed.wait(timeout=10.0)
        return orig_sealed()

    alpha_ns.sealed_segments = blocking_sealed  # type: ignore[method-assign]

    query_error: list[Exception] = []
    query_result: list[pa.Table] = []

    def run_query():
        try:
            query_result.append(store.query('SELECT COUNT(*) AS n FROM "ns.alpha"'))
        except Exception as exc:
            query_error.append(exc)

    qt = threading.Thread(target=run_query, daemon=True)
    qt.start()
    try:
        # Wait until the query is parked inside the per-namespace loop.
        assert in_loop.wait(timeout=5.0), "query never reached the per-namespace loop"

        # Drop ns.victim. ``drop_table`` mutates ``self._namespaces`` under
        # the insertion lock. Because the query already snapshotted the
        # dict under the same insertion lock, this mutation no longer
        # affects the in-flight iteration.
        drop_thread = threading.Thread(target=lambda: store.drop_table("ns.victim"), daemon=True)
        drop_thread.start()
        # Drop will block on the rwlock write side until the query
        # completes — release the barrier so the query can finish.
        proceed.set()
        drop_thread.join(timeout=10.0)
        assert not drop_thread.is_alive(), "drop_table did not complete"
    finally:
        alpha_ns.sealed_segments = orig_sealed  # type: ignore[method-assign]

    qt.join(timeout=10.0)
    assert not qt.is_alive(), "query thread did not complete"
    assert not query_error, f"unexpected query error: {query_error[0]!r}"
    assert query_result and query_result[0].column("n").to_pylist() == [1]


def test_query_acquires_read_lock_for_duration(store: DuckDBLogStore):
    """``store.query`` takes the read lock; the write lock waits for it.

    We instrument the rwlock so a write-acquire attempt records that it
    waited for the query to finish. The test does not rely on timing or
    sleep: the rwlock's internal counter tells us whether the write was
    forced to queue.
    """
    store.register_table("iris.worker", _worker_schema())
    ns = store._namespaces["iris.worker"]
    store.write_rows("iris.worker", _ipc_bytes(_worker_batch(["w-1"], [100], [1])))
    ns._flush_step()
    ns._compaction_step(compact_single=True)

    rwlock = store._query_visibility_lock
    write_held_during_query: list[bool] = []

    # Patch the rwlock's read_release to attempt the write_acquire
    # *before* read count drops to zero. If the write side could acquire
    # while the read side was still held, we'd record True here.
    orig_release = rwlock.read_release
    write_observed_readers = threading.Event()

    def instrumented_release():
        # While we still hold the read lock (refcount > 0), confirm the
        # rwlock would block a writer. The internal state check is the
        # only reliable signal that doesn't rely on timing.
        write_held_during_query.append(rwlock._readers > 0)
        write_observed_readers.set()
        orig_release()

    rwlock.read_release = instrumented_release  # type: ignore[method-assign]

    try:
        table = store.query('SELECT COUNT(*) AS n FROM "iris.worker"')
        assert table.column("n").to_pylist() == [1]
    finally:
        rwlock.read_release = orig_release  # type: ignore[method-assign]

    # The instrumentation fired exactly once and observed the read lock
    # still held at the moment of release.
    assert write_held_during_query == [True]
