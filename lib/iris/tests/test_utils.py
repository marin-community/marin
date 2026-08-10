# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for iris.test_util synchronization helpers."""

from collections.abc import Callable
from pathlib import Path

import iris.test_util as test_util
import pytest
import rigging.timing as timing
from iris.test_util import SentinelFile, wait_for_condition
from rigging.timing import Duration


class FakeClock:
    def __init__(self) -> None:
        self.current = 0.0
        self.sleeps: list[float] = []
        self.after_sleep: Callable[[], None] | None = None

    def monotonic(self) -> float:
        return self.current

    def sleep(self, interval: float) -> None:
        self.sleeps.append(interval)
        self.current += interval
        if self.after_sleep is not None:
            self.after_sleep()


@pytest.fixture
def fake_clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    clock = FakeClock()
    monkeypatch.setattr(test_util, "time", clock)
    monkeypatch.setattr(timing, "time", clock)
    return clock


def test_sentinel_file_signal_and_wait(tmp_path: Path, fake_clock: FakeClock) -> None:
    """Test SentinelFile signal and wait work together."""
    sentinel = SentinelFile(str(tmp_path / "nested" / "dir" / "sentinel.txt"))

    sentinel.signal()
    assert sentinel.is_set()

    sentinel.wait(timeout=Duration.from_seconds(1.0))
    assert fake_clock.sleeps == []


def test_sentinel_file_timeout(tmp_path: Path, fake_clock: FakeClock) -> None:
    """Test SentinelFile.wait raises TimeoutError when file doesn't appear."""
    sentinel = SentinelFile(str(tmp_path / "nonexistent.txt"))

    with pytest.raises(TimeoutError, match="not signalled within"):
        sentinel.wait(timeout=Duration.from_seconds(0.1))
    assert fake_clock.current == 0.1


def test_sentinel_file_wait_observes_file_created_between_polls(tmp_path: Path, fake_clock: FakeClock) -> None:
    sentinel = SentinelFile(str(tmp_path / "concurrent.txt"))
    fake_clock.after_sleep = sentinel.signal

    sentinel.wait(timeout=Duration.from_seconds(1.0))

    assert sentinel.is_set()
    assert fake_clock.sleeps == [0.1]


def test_sentinel_file_reset(tmp_path: Path) -> None:
    """Test SentinelFile.reset removes the file."""
    sentinel = SentinelFile(str(tmp_path / "sentinel.txt"))

    sentinel.signal()
    assert sentinel.is_set()

    sentinel.reset()
    assert not sentinel.is_set()

    # Reset is idempotent
    sentinel.reset()
    assert not sentinel.is_set()


def test_wait_for_condition_immediate(fake_clock: FakeClock) -> None:
    """Test wait_for_condition returns immediately when condition is already true."""
    wait_for_condition(lambda: True, timeout=Duration.from_seconds(1.0))
    assert fake_clock.sleeps == []


def test_wait_for_condition_timeout(fake_clock: FakeClock) -> None:
    """Test wait_for_condition raises TimeoutError when condition never becomes true."""
    with pytest.raises(TimeoutError, match="did not become true within"):
        wait_for_condition(lambda: False, timeout=Duration.from_seconds(0.1))
    assert fake_clock.current >= 0.1


def test_wait_for_condition_becomes_true(fake_clock: FakeClock) -> None:
    """Test wait_for_condition succeeds when condition becomes true."""
    ready = False

    def mark_ready() -> None:
        nonlocal ready
        ready = True

    fake_clock.after_sleep = mark_ready
    wait_for_condition(lambda: ready, timeout=Duration.from_seconds(1.0))
    assert fake_clock.sleeps == [0.01]
