# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test utilities for reliable synchronization without time.sleep().

Best Practices for Test Synchronization
========================================

PREFER SentinelFile and wait_for_condition over time.sleep:

    BAD - flaky and slow:
        worker.start()
        time.sleep(0.5)  # Hope worker is ready
        assert worker.is_ready()

    GOOD - reliable and fast:
        sentinel = SentinelFile(str(tmp_path / "ready"))
        worker.start(ready_sentinel=sentinel)
        sentinel.wait(timeout=5.0)
        assert worker.is_ready()

    ALSO GOOD - for condition-based waiting:
        worker.start()
        wait_for_condition(lambda: worker.is_ready(), timeout=5.0)

Thread Safety
=============

All managed threads should:
1. Accept stop_event as first parameter
2. Check stop_event regularly in loops (every ~0.1-1s)
3. Exit promptly when stop_event is set
4. Be created via ThreadContainer for lifecycle management

Example thread pattern:
    def worker_loop(stop_event: threading.Event, config: Config) -> None:
        while not stop_event.is_set():
            do_work()
            stop_event.wait(timeout=1.0)  # Sleep but check stop_event
"""

import time
from collections.abc import Callable
from pathlib import Path

from iris.test_util import SentinelFile


def wait_for_condition(condition: Callable[[], bool], timeout: float = 10.0, poll_interval: float = 0.01) -> None:
    """Wait for a condition to become true.

    Polls the condition callable at regular intervals until it returns True
    or timeout is reached. More flexible than wait_for_sentinel for complex conditions.

    Args:
        condition: Callable that returns True when the condition is satisfied
        timeout: Maximum time to wait in seconds (default: 10.0)
        poll_interval: Time between condition checks in seconds (default: 0.01)

    Raises:
        TimeoutError: If the condition does not become true within timeout

    Example:
        >>> controller.start()
        >>> # Wait for controller to be ready
        >>> wait_for_condition(lambda: controller.is_ready(), timeout=5.0)
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(poll_interval)
    raise TimeoutError(f"Condition did not become true within {timeout}s")


# Tests for sentinel utilities


def test_sentinel_file_signal_and_wait(tmp_path: Path) -> None:
    """Test SentinelFile signal and wait work together."""
    sentinel = SentinelFile(str(tmp_path / "nested" / "dir" / "sentinel.txt"))

    # Signal creates file
    sentinel.signal()
    assert sentinel.is_set()

    # Wait returns immediately when file exists
    start = time.monotonic()
    sentinel.wait(timeout=1.0)
    elapsed = time.monotonic() - start

    # Should return almost immediately (< 100ms)
    assert elapsed < 0.1


def test_sentinel_file_timeout(tmp_path: Path) -> None:
    """Test SentinelFile.wait raises TimeoutError when file doesn't appear."""
    sentinel = SentinelFile(str(tmp_path / "nonexistent.txt"))

    import pytest

    start = time.monotonic()
    with pytest.raises(TimeoutError, match="not signalled within"):
        sentinel.wait(timeout=0.1)
    elapsed = time.monotonic() - start

    # Should wait approximately the timeout duration
    assert 0.09 < elapsed < 0.2


def test_sentinel_file_concurrent_creation(tmp_path: Path) -> None:
    """Test SentinelFile.wait works when file is created concurrently."""
    import threading

    sentinel = SentinelFile(str(tmp_path / "concurrent.txt"))

    def create_after_delay():
        time.sleep(0.05)
        sentinel.signal()

    thread = threading.Thread(target=create_after_delay, daemon=True)
    thread.start()

    # Wait should succeed once thread creates the file
    start = time.monotonic()
    sentinel.wait(timeout=1.0)
    elapsed = time.monotonic() - start

    # Should return shortly after file creation (~50-100ms)
    assert 0.04 < elapsed < 0.2
    assert sentinel.is_set()

    thread.join()


def test_sentinel_file_reset(tmp_path: Path) -> None:
    """Test SentinelFile.reset removes the file."""
    sentinel = SentinelFile(str(tmp_path / "sentinel.txt"))

    # Signal then reset
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
    wait_for_condition(lambda: True, timeout=1.0)
    elapsed = time.monotonic() - start

    # Should return almost immediately
    assert elapsed < 0.1


def test_wait_for_condition_timeout() -> None:
    """Test wait_for_condition raises TimeoutError when condition never becomes true."""
    import pytest

    start = time.monotonic()
    with pytest.raises(TimeoutError, match="did not become true within"):
        wait_for_condition(lambda: False, timeout=0.1)
    elapsed = time.monotonic() - start

    # Should wait approximately the timeout duration
    assert 0.09 < elapsed < 0.2


def test_wait_for_condition_becomes_true() -> None:
    """Test wait_for_condition succeeds when condition becomes true."""
    import threading

    flag = threading.Event()

    def set_after_delay():
        time.sleep(0.05)
        flag.set()

    thread = threading.Thread(target=set_after_delay, daemon=True)
    thread.start()

    start = time.monotonic()
    wait_for_condition(lambda: flag.is_set(), timeout=1.0)
    elapsed = time.monotonic() - start

    # Should return shortly after flag is set (~50-100ms)
    assert 0.04 < elapsed < 0.2

    thread.join()
