# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the in-memory heartbeat store."""

import pytest

from iris.cluster.controller.heartbeat_manager import (
    HeartbeatManager,
    heartbeat_inmemory_enabled,
)


@pytest.fixture
def hbm() -> HeartbeatManager:
    return HeartbeatManager()


def test_record_alive_sets_recent_age(hbm: HeartbeatManager) -> None:
    hbm.record_alive("w1")
    age = hbm.age_ms("w1")
    assert age is not None
    assert 0 <= age < 100


def test_record_failure_increments_then_record_alive_resets(hbm: HeartbeatManager) -> None:
    assert hbm.record_failure("w1") == 1
    assert hbm.record_failure("w1") == 2
    assert hbm.record_failure("w1") == 3
    hbm.record_alive("w1")
    # Next failure restarts the count.
    assert hbm.record_failure("w1") == 1


def test_age_ms_for_absent_worker_is_none(hbm: HeartbeatManager) -> None:
    # Absence is "no data yet", not a failure / not staleness.
    assert hbm.age_ms("ghost") is None


def test_retain_only_drops_others(hbm: HeartbeatManager) -> None:
    hbm.record_alive("w1")
    hbm.record_alive("w2")
    hbm.record_alive("w3")
    hbm.retain_only({"w1", "w3"})
    assert hbm.age_ms("w1") is not None
    assert hbm.age_ms("w2") is None
    assert hbm.age_ms("w3") is not None


def test_remove_drops_entry(hbm: HeartbeatManager) -> None:
    hbm.record_alive("w1")
    hbm.remove("w1")
    assert hbm.age_ms("w1") is None


def test_record_failure_on_absent_creates_entry_without_staleness(hbm: HeartbeatManager) -> None:
    count = hbm.record_failure("fresh")
    assert count == 1
    # Entry exists now, age should be tiny — absence didn't look stale.
    age = hbm.age_ms("fresh")
    assert age is not None
    assert age < 100


def test_writes_avoided_counter_tracks_successful_pings(hbm: HeartbeatManager) -> None:
    hbm.record_alive("w1")
    hbm.record_alive("w1")
    hbm.record_alive("w2")
    assert hbm.writes_avoided == 3


def test_note_db_failure_write_accumulates(hbm: HeartbeatManager) -> None:
    hbm.note_db_failure_write(2)
    hbm.note_db_failure_write(3)
    assert hbm.db_failure_writes == 5


def test_enabled_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IRIS_HEARTBEAT_INMEMORY", raising=False)
    assert heartbeat_inmemory_enabled() is False
    for truthy in ("1", "true", "yes", "True", "YES"):
        monkeypatch.setenv("IRIS_HEARTBEAT_INMEMORY", truthy)
        assert heartbeat_inmemory_enabled() is True
    for falsy in ("0", "false", "no", ""):
        monkeypatch.setenv("IRIS_HEARTBEAT_INMEMORY", falsy)
        assert heartbeat_inmemory_enabled() is False
