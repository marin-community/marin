# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading

import pyarrow as pa
from finelog.store.duckdb_store import DuckDBLogStore
from finelog.store.schema import NamespaceNotFoundError

from tests.conftest import _ipc_bytes, _seal, _worker_batch, _worker_schema


def test_drop_during_concurrent_write_is_safe(store: DuckDBLogStore):
    """A racing ``drop_table`` + ``write_rows`` upholds the write contract.

    A write either completes and persists, or fails with
    ``NamespaceNotFoundError`` — never a surprise exception or deadlock.
    """
    store.register_table("iris.worker", _worker_schema())
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
        # All 8 writers slipped in before the drop's lookup. Possible but
        # rare — drop runs from the main thread which started last.
        pass

    for t in threads:
        t.join(timeout=5.0)
        assert not t.is_alive(), "writer thread did not terminate"

    successes = 0
    for r in write_results:
        if isinstance(r, Exception):
            assert isinstance(r, NamespaceNotFoundError), repr(r)
        else:
            assert r == 1
            successes += 1

    if drop_succeeded:
        # In-memory chunks for successful writes evaporate by design.
        assert "iris.worker" not in store._namespaces
    else:
        assert "iris.worker" in store._namespaces
        ns = store._namespaces["iris.worker"]
        ns._flush_step()
        ns._compaction_step(compact_single=True)
        table = store.query('SELECT worker_id FROM "iris.worker"')
        assert table.num_rows == successes


def test_query_safe_against_concurrent_drop_table(store: DuckDBLogStore):
    """A query iterating namespaces is safe against a concurrent drop.

    Regression: ``query()`` must snapshot ``self._namespaces`` under the
    insertion lock so a concurrent ``drop_table`` can't trigger
    ``RuntimeError: dictionary changed size during iteration``.
    """
    store.register_table("ns.alpha", _worker_schema())
    store.register_table("ns.victim", _worker_schema())
    store.write_rows("ns.alpha", _ipc_bytes(_worker_batch(["a"], [1], [1])))
    _seal(store, "ns.alpha")

    alpha_ns = store._namespaces["ns.alpha"]
    in_loop = threading.Event()
    proceed = threading.Event()
    orig_query_snapshot = alpha_ns.query_snapshot

    def blocking_query_snapshot():
        in_loop.set()
        proceed.wait(timeout=10.0)
        return orig_query_snapshot()

    alpha_ns.query_snapshot = blocking_query_snapshot  # type: ignore[method-assign]

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
        assert in_loop.wait(timeout=5.0), "query never reached the per-namespace loop"

        drop_thread = threading.Thread(target=lambda: store.drop_table("ns.victim"), daemon=True)
        drop_thread.start()
        # Release the barrier so the query can finish; drop is queued
        # on the rwlock write side until then.
        proceed.set()
        drop_thread.join(timeout=10.0)
        assert not drop_thread.is_alive(), "drop_table did not complete"
    finally:
        alpha_ns.query_snapshot = orig_query_snapshot  # type: ignore[method-assign]

    qt.join(timeout=10.0)
    assert not qt.is_alive(), "query thread did not complete"
    assert not query_error, f"unexpected query error: {query_error[0]!r}"
    assert query_result and query_result[0].column("n").to_pylist() == [1]


def test_query_acquires_read_lock_for_duration(store: DuckDBLogStore):
    """``store.query`` takes the read lock; the write lock waits for it."""
    store.register_table("iris.worker", _worker_schema())
    ns = store._namespaces["iris.worker"]
    store.write_rows("iris.worker", _ipc_bytes(_worker_batch(["w-1"], [100], [1])))
    ns._flush_step()
    ns._compaction_step(compact_single=True)

    rwlock = store._query_visibility_lock
    write_held_during_query: list[bool] = []

    # Inspect the rwlock's reader count from inside read_release: if the
    # writer could acquire here, _readers would already be 0.
    orig_release = rwlock.read_release
    write_observed_readers = threading.Event()

    def instrumented_release():
        write_held_during_query.append(rwlock._readers > 0)
        write_observed_readers.set()
        orig_release()

    rwlock.read_release = instrumented_release  # type: ignore[method-assign]

    try:
        table = store.query('SELECT COUNT(*) AS n FROM "iris.worker"')
        assert table.column("n").to_pylist() == [1]
    finally:
        rwlock.read_release = orig_release  # type: ignore[method-assign]

    assert write_held_during_query == [True]
