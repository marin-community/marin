# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for iris.test_util synchronization helpers."""

import threading
import time
from pathlib import Path

import pytest

from iris.test_util import SentinelFile, wait_for_condition
from rigging.timing import Duration


def test_sentinel_file_signal_and_wait(tmp_path: Path) -> None:
    """Test SentinelFile signal and wait work together."""
    sentinel = SentinelFile(str(tmp_path / "nested" / "dir" / "sentinel.txt"))

    sentinel.signal()
    assert sentinel.is_set()

    start = time.monotonic()
    sentinel.wait(timeout=Duration.from_seconds(1.0))
    elapsed = time.monotonic() - start

    assert elapsed < 0.1


def test_sentinel_file_timeout(tmp_path: Path) -> None:
    """Test SentinelFile.wait raises TimeoutError when file doesn't appear."""
    sentinel = SentinelFile(str(tmp_path / "nonexistent.txt"))

    start = time.monotonic()
    with pytest.raises(TimeoutError, match="not signalled within"):
        sentinel.wait(timeout=Duration.from_seconds(0.1))
    elapsed = time.monotonic() - start

    assert 0.09 < elapsed < 0.2


def test_sentinel_file_concurrent_creation(tmp_path: Path) -> None:
    """Test SentinelFile.wait works when file is created concurrently."""
    sentinel = SentinelFile(str(tmp_path / "concurrent.txt"))

    def create_after_delay():
        time.sleep(0.05)
        sentinel.signal()

    thread = threading.Thread(target=create_after_delay, daemon=True)
    thread.start()

    start = time.monotonic()
    sentinel.wait(timeout=Duration.from_seconds(1.0))
    elapsed = time.monotonic() - start

    assert 0.04 < elapsed < 0.2
    assert sentinel.is_set()

    thread.join()


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


def test_wait_for_condition_immediate() -> None:
    """Test wait_for_condition returns immediately when condition is already true."""
    start = time.monotonic()
    wait_for_condition(lambda: True, timeout=Duration.from_seconds(1.0))
    elapsed = time.monotonic() - start

    assert elapsed < 0.1


def test_wait_for_condition_timeout() -> None:
    """Test wait_for_condition raises TimeoutError when condition never becomes true."""
    start = time.monotonic()
    with pytest.raises(TimeoutError, match="did not become true within"):
        wait_for_condition(lambda: False, timeout=Duration.from_seconds(0.1))
    elapsed = time.monotonic() - start

    assert 0.09 < elapsed < 1.0


def test_wait_for_condition_becomes_true() -> None:
    """Test wait_for_condition succeeds when condition becomes true."""
    flag = threading.Event()

    def set_after_delay():
        time.sleep(0.05)
        flag.set()

    thread = threading.Thread(target=set_after_delay, daemon=True)
    thread.start()

    start = time.monotonic()
    wait_for_condition(lambda: flag.is_set(), timeout=Duration.from_seconds(1.0))
    elapsed = time.monotonic() - start

    assert 0.04 < elapsed < 0.2

    thread.join()
