# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end coverage of the durable-write contract:

* writers don't observe success until their rows are on an L0 segment,
* the persistence cursor exposed to handlers reflects parquet state,
* timeouts surface to RPC callers without dropping buffered data,
* memory-mode namespaces treat every append as immediately persisted.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from finelog.store.duckdb_store import DuckDBLogStore
from finelog.store.log_namespace import DiskLogNamespace

from tests.conftest import _ipc_bytes, _worker_batch, _worker_schema


def _segments_dir(store: DuckDBLogStore, namespace: str) -> Path:
    return store._namespace_dir(namespace)  # type: ignore[return-value]


def test_max_persisted_seq_advances_after_flush(store: DuckDBLogStore):
    store.register_table("iris.worker", _worker_schema())
    rows, last_seq = store.write_rows("iris.worker", _ipc_bytes(_worker_batch(["w-1", "w-2"], [1, 2], [10, 20])))
    assert rows == 2
    assert last_seq == 2

    store._wait_persisted("iris.worker", last_seq, timeout=5.0)

    parquet_files = list(_segments_dir(store, "iris.worker").glob("*.parquet"))
    assert parquet_files, "expected at least one L0 segment after persistence wait"
    assert store.max_persisted_seq("iris.worker") >= last_seq


def test_memory_namespace_persisted_seq_is_next_seq_minus_one(tmp_path: Path):
    # log_dir=None selects in-memory mode for every namespace.
    store = DuckDBLogStore(log_dir=None)
    try:
        store.register_table("iris.worker", _worker_schema())
        _, last_seq = store.write_rows("iris.worker", _ipc_bytes(_worker_batch(["w-1"], [1], [10])))
        assert last_seq == 1
        # Memory namespaces have no flush boundary — the cursor must already
        # be past the just-allocated seq when the append returns.
        assert store.max_persisted_seq("iris.worker") >= last_seq
    finally:
        store.close()


def test_request_persistance_returns_immediately_when_already_persisted(store: DuckDBLogStore):
    store.register_table("iris.worker", _worker_schema())
    _, last_seq = store.write_rows("iris.worker", _ipc_bytes(_worker_batch(["w-1"], [1], [10])))
    store._wait_persisted("iris.worker", last_seq, timeout=5.0)

    assert store.request_persistance("iris.worker", last_seq, timeout=0.01) == last_seq


def test_request_persistance_returns_immediately_for_empty_append(store: DuckDBLogStore):
    store.register_table("iris.worker", _worker_schema())
    # Sentinel for "no rows appended" — handler must not block.
    assert store.request_persistance("iris.worker", -1, timeout=0.01) == -1


def test_request_persistance_times_out_when_flush_blocked(store: DuckDBLogStore, monkeypatch):
    store.register_table("iris.worker", _worker_schema())
    # Stall the flush so persistence never advances inside the timeout.
    stalled = threading.Event()
    release = threading.Event()
    original_write = DiskLogNamespace._write_new_segment

    def stalling_write(self, visible):
        stalled.set()
        # Cap the stall so test teardown doesn't deadlock if release isn't
        # signalled (the test itself sets release in the finally block).
        release.wait(timeout=2.0)
        return original_write(self, visible)

    monkeypatch.setattr(DiskLogNamespace, "_write_new_segment", stalling_write)
    try:
        _, last_seq = store.write_rows("iris.worker", _ipc_bytes(_worker_batch(["w-1"], [1], [10])))
        with pytest.raises(TimeoutError):
            store.request_persistance("iris.worker", last_seq, timeout=0.1)
        assert stalled.is_set(), "bg loop never started the flush"
    finally:
        # Let the bg loop unblock so close() can join cleanly.
        release.set()


def test_request_persistance_flushes_buffered_writers_once(store: DuckDBLogStore, monkeypatch):
    """A persistence wait drains all buffered writers in one L0 segment."""
    store.register_table("iris.worker", _worker_schema())

    flush_count = 0
    original_write = DiskLogNamespace._write_new_segment

    def counting_write(self, visible):
        nonlocal flush_count
        flush_count += 1
        return original_write(self, visible)

    monkeypatch.setattr(DiskLogNamespace, "_write_new_segment", counting_write)

    # Issue several concurrent writes from worker threads. One persistence
    # wait should wake one background flush that drains all buffered rows.
    num_writers = 8
    results: list[int] = []
    results_lock = threading.Lock()

    def writer(i: int):
        _, last_seq = store.write_rows("iris.worker", _ipc_bytes(_worker_batch([f"w-{i}"], [i], [i])))
        with results_lock:
            results.append(last_seq)

    threads = [threading.Thread(target=writer, args=(i,), daemon=True) for i in range(num_writers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)
        assert not t.is_alive()

    assert len(results) == num_writers
    final_target = max(results)
    store._wait_persisted("iris.worker", final_target, timeout=5.0)

    assert flush_count == 1
