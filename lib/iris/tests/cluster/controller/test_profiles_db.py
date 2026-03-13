# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for profile storage: ring buffer, store/query round-trips, migration idempotency."""

from pathlib import Path

import pytest

from iris.cluster.controller.db import (
    ControllerDB,
    recent_profiles,
    store_profile,
)


@pytest.fixture
def db(tmp_path: Path) -> ControllerDB:
    return ControllerDB(tmp_path / "test.db")


def test_store_and_retrieve_profile(db: ControllerDB) -> None:
    store_profile(db, "worker-1", "threads", b"stack trace data", 1000)

    profiles = recent_profiles(db, "worker-1", "threads")
    assert len(profiles) == 1
    assert profiles[0].target_id == "worker-1"
    assert profiles[0].profile_type == "threads"
    assert profiles[0].data == b"stack trace data"
    assert profiles[0].captured_at.epoch_ms() == 1000


def test_recent_profiles_ordered_newest_first(db: ControllerDB) -> None:
    for i in range(5):
        store_profile(db, "worker-1", "threads", f"data-{i}".encode(), 1000 + i)

    profiles = recent_profiles(db, "worker-1", "threads")
    timestamps = [p.captured_at.epoch_ms() for p in profiles]
    assert timestamps == sorted(timestamps, reverse=True)
    assert len(profiles) == 5


def test_recent_profiles_respects_limit(db: ControllerDB) -> None:
    for i in range(5):
        store_profile(db, "worker-1", "threads", f"data-{i}".encode(), 1000 + i)

    profiles = recent_profiles(db, "worker-1", "threads", limit=3)
    assert len(profiles) == 3
    assert profiles[0].captured_at.epoch_ms() == 1004


def test_ring_buffer_keeps_last_10(db: ControllerDB) -> None:
    for i in range(15):
        store_profile(db, "worker-1", "threads", f"data-{i}".encode(), 1000 + i)

    profiles = recent_profiles(db, "worker-1", "threads")
    assert len(profiles) == 10
    # Should keep the newest 10 (ids 6-14)
    timestamps = [p.captured_at.epoch_ms() for p in profiles]
    assert min(timestamps) == 1005
    assert max(timestamps) == 1014


def test_ring_buffer_independent_per_target(db: ControllerDB) -> None:
    for i in range(15):
        store_profile(db, "worker-1", "threads", f"data-{i}".encode(), 1000 + i)
    for i in range(3):
        store_profile(db, "worker-2", "threads", f"data-{i}".encode(), 2000 + i)

    assert len(recent_profiles(db, "worker-1", "threads")) == 10
    assert len(recent_profiles(db, "worker-2", "threads")) == 3


def test_ring_buffer_independent_per_profile_type(db: ControllerDB) -> None:
    for i in range(15):
        store_profile(db, "worker-1", "threads", f"data-{i}".encode(), 1000 + i)
    for i in range(5):
        store_profile(db, "worker-1", "cpu", f"cpu-{i}".encode(), 2000 + i)

    assert len(recent_profiles(db, "worker-1", "threads")) == 10
    assert len(recent_profiles(db, "worker-1", "cpu")) == 5


def test_recent_profiles_filters_by_target_and_type(db: ControllerDB) -> None:
    store_profile(db, "worker-1", "threads", b"w1-threads", 1000)
    store_profile(db, "worker-1", "cpu", b"w1-cpu", 1001)
    store_profile(db, "worker-2", "threads", b"w2-threads", 1002)

    w1_threads = recent_profiles(db, "worker-1", "threads")
    assert len(w1_threads) == 1
    assert w1_threads[0].data == b"w1-threads"

    w2_threads = recent_profiles(db, "worker-2", "threads")
    assert len(w2_threads) == 1
    assert w2_threads[0].data == b"w2-threads"


def test_migration_idempotency(tmp_path: Path) -> None:
    """Applying migration twice should not fail (IF NOT EXISTS guards)."""
    db = ControllerDB(tmp_path / "test.db")
    store_profile(db, "worker-1", "threads", b"data", 1000)

    # Re-create DB pointing to same file — migrations re-apply idempotently
    db2 = ControllerDB(tmp_path / "test.db")
    profiles = recent_profiles(db2, "worker-1", "threads")
    assert len(profiles) == 1
