# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Async copy worker behavior.

Compaction must not block on the GCS upload; the worker drains on close
and on the test ``_wait_for_copies`` hook.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from finelog.store import duckdb_store
from finelog.store.duckdb_store import DuckDBLogStore

from tests.conftest import _ipc_bytes, _worker_batch, _worker_schema


def test_compaction_returns_immediately_when_upload_is_slow(tmp_path: Path, monkeypatch):
    """A multi-second upload must not stall the compaction thread.

    The copy worker polls the catalog independently; compaction just
    wakes it. So compaction returns regardless of upload duration.
    """
    remote = tmp_path / "remote"
    remote.mkdir()

    release_upload = threading.Event()
    real_upload = duckdb_store.CopyWorker._upload

    def gated_upload(self, namespace, local_path):
        if not release_upload.wait(timeout=5.0):
            raise AssertionError("test forgot to release the copy gate")
        return real_upload(self, namespace, local_path)

    monkeypatch.setattr(duckdb_store.CopyWorker, "_upload", gated_upload)

    store = DuckDBLogStore(log_dir=tmp_path / "data", remote_log_dir=str(remote))
    try:
        store.register_table("iris.worker", _worker_schema())
        store.write_rows("iris.worker", _ipc_bytes(_worker_batch(["w-1"], [100], [1])))
        ns = store._namespaces["iris.worker"]
        ns._flush_step()

        compaction_start = time.monotonic()
        ns._force_compact_l0()
        compaction_elapsed = time.monotonic() - compaction_start

        assert compaction_elapsed < 1.0, f"compaction blocked on upload (elapsed={compaction_elapsed:.3f}s)"

        remote_ns_dir = remote / "iris.worker"
        if remote_ns_dir.exists():
            assert not list(remote_ns_dir.glob("*.parquet"))

        release_upload.set()
        assert store._wait_for_copies(timeout=5.0)
        assert sorted((remote / "iris.worker").glob("*.parquet"))
    finally:
        release_upload.set()
        store.close()


def test_close_drains_pending_copies(tmp_path: Path):
    """``close`` blocks until the copy worker finishes queued uploads."""
    remote = tmp_path / "remote"
    remote.mkdir()

    store = DuckDBLogStore(log_dir=tmp_path / "data", remote_log_dir=str(remote))
    try:
        store.register_table("iris.worker", _worker_schema())
        store.write_rows("iris.worker", _ipc_bytes(_worker_batch(["w-1"], [100], [1])))
        ns = store._namespaces["iris.worker"]
        ns._flush_step()
        ns._force_compact_l0()
    except Exception:
        store.close()
        raise

    # close() should drain the queue before tearing down the worker.
    store.close()

    remote_files = sorted((remote / "iris.worker").glob("*.parquet"))
    assert remote_files, "expected close() to drain pending uploads"


def test_copy_worker_not_started_without_remote_dir(tmp_path: Path):
    """No remote_log_dir => no worker thread, even with a disk store."""
    store = DuckDBLogStore(log_dir=tmp_path / "data")
    try:
        assert store._copy_worker is None
        # Wait hook is a no-op when the worker is absent.
        assert store._wait_for_copies(timeout=0.0)
    finally:
        store.close()
