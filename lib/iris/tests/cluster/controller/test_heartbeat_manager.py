# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for HeartbeatManager."""

from __future__ import annotations

import time

from iris.cluster.controller.heartbeat_manager import HeartbeatManager


def test_record_alive_resets_failure_count() -> None:
    hbm = HeartbeatManager()
    hbm.record_failure("w1")
    hbm.record_failure("w1")
    hbm.record_alive("w1")
    # Subsequent failure starts over at 1.
    assert hbm.record_failure("w1") == 1


def test_record_failure_increments_and_returns_new_count() -> None:
    hbm = HeartbeatManager()
    assert hbm.record_failure("w1") == 1
    assert hbm.record_failure("w1") == 2
    assert hbm.record_failure("w2") == 1


def test_age_ms_reports_freshly_recorded_liveness() -> None:
    hbm = HeartbeatManager()
    hbm.record_alive("w1")
    age = hbm.age_ms("w1")
    assert age is not None
    assert age < 100


def test_age_ms_none_for_unknown_worker() -> None:
    hbm = HeartbeatManager()
    assert hbm.age_ms("never-seen") is None


def test_seed_from_db_preserves_wallclock_age() -> None:
    hbm = HeartbeatManager()
    wall_now_ms = int(time.time() * 1000)
    # Worker last seen 2 s ago.
    hbm.seed_from_db([("w1", wall_now_ms - 2000)])
    age = hbm.age_ms("w1")
    assert age is not None
    # Allow generous slack for clock drift and test scheduling jitter.
    assert 1500 <= age <= 3500


def test_seed_from_db_with_null_last_heartbeat_starts_fresh() -> None:
    hbm = HeartbeatManager()
    hbm.seed_from_db([("w1", None)])
    age = hbm.age_ms("w1")
    assert age is not None
    assert age < 500


def test_retain_only_drops_missing_workers() -> None:
    hbm = HeartbeatManager()
    hbm.record_alive("w1")
    hbm.record_alive("w2")
    hbm.record_alive("w3")
    hbm.retain_only({"w1", "w3"})
    assert hbm.age_ms("w1") is not None
    assert hbm.age_ms("w2") is None
    assert hbm.age_ms("w3") is not None


def test_writes_avoided_counts_successful_pings() -> None:
    hbm = HeartbeatManager()
    assert hbm.writes_avoided == 0
    hbm.record_alive("w1")
    hbm.record_alive("w2")
    hbm.record_alive("w1")
    assert hbm.writes_avoided == 3


def test_db_failure_writes_counts_reconciler_writes() -> None:
    hbm = HeartbeatManager()
    hbm.note_db_failure_write(3)
    hbm.note_db_failure_write()
    assert hbm.db_failure_writes == 4


def test_remove_drops_worker() -> None:
    hbm = HeartbeatManager()
    hbm.record_alive("w1")
    hbm.remove("w1")
    assert hbm.age_ms("w1") is None
