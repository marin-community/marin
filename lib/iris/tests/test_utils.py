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

PREFER sentinel files and wait_for_condition over time.sleep:

    BAD - flaky and slow:
        worker.start()
        time.sleep(0.5)  # Hope worker is ready
        assert worker.is_ready()

    GOOD - reliable and fast:
        with sentinel_file(tmp_path / "ready") as sentinel:
            worker.start(ready_sentinel=sentinel)
            wait_for_sentinel(sentinel, timeout=5.0)
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
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def sentinel_file(path: Path) -> Iterator[Path]:
    """Create a sentinel file for signaling test completion.

    This context manager creates parent directories for the sentinel file if needed,
    yields the path for use during testing, and ensures cleanup in the finally block.

    Args:
        path: Path to the sentinel file

    Yields:
        The Path object for the sentinel file

    Example:
        >>> with sentinel_file(tmp_path / "controller_ready") as sentinel:
        ...     controller.start()
        ...     # Controller will signal readiness by creating sentinel
        ...     wait_for_sentinel(sentinel, timeout=5.0)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        if path.exists():
            path.unlink()


def wait_for_sentinel(path: Path, timeout: float = 10.0) -> None:
    """Wait for sentinel file to appear.

    Polls every 0.01 seconds until the sentinel file exists or timeout is reached.

    Args:
        path: Path to the sentinel file to wait for
        timeout: Maximum time to wait in seconds (default: 10.0)

    Raises:
        TimeoutError: If the sentinel file does not appear within timeout

    Example:
        >>> sentinel = tmp_path / "job_complete"
        >>> # Start job in background that will create sentinel when done
        >>> wait_for_sentinel(sentinel, timeout=30.0)
        >>> # Job is now complete
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.01)
    raise TimeoutError(f"Sentinel file {path} did not appear within {timeout}s")


def signal_sentinel(path: Path) -> None:
    """Signal completion by creating sentinel file.

    Creates parent directories if needed, then creates (touches) the sentinel file.

    Args:
        path: Path to the sentinel file to create

    Example:
        >>> def background_task(sentinel_path: Path):
        ...     # Do work
        ...     process_data()
        ...     # Signal completion
        ...     signal_sentinel(sentinel_path)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def wait_for_condition(condition: callable, timeout: float = 10.0, poll_interval: float = 0.01) -> None:
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


def test_sentinel_file_creates_and_cleans_up(tmp_path: Path) -> None:
    """Test that sentinel_file context manager creates parent dirs and cleans up."""
    sentinel = tmp_path / "subdir" / "sentinel.txt"

    # Initially doesn't exist
    assert not sentinel.exists()

    with sentinel_file(sentinel) as path:
        assert path == sentinel
        # Parent directory is created
        assert sentinel.parent.exists()
        # File doesn't exist yet (caller creates it)
        assert not sentinel.exists()

        # Simulate signaling
        sentinel.touch()
        assert sentinel.exists()

    # Cleanup happens automatically
    assert not sentinel.exists()
    # Parent dir still exists (we only clean up the file)
    assert sentinel.parent.exists()


def test_sentinel_file_cleans_up_on_exception(tmp_path: Path) -> None:
    """Test that sentinel_file cleans up even when exception occurs."""
    sentinel = tmp_path / "sentinel.txt"

    try:
        with sentinel_file(sentinel):
            sentinel.touch()
            assert sentinel.exists()
            raise ValueError("Test exception")
    except ValueError:
        pass

    # Cleanup still happens
    assert not sentinel.exists()


def test_signal_and_wait_sentinel(tmp_path: Path) -> None:
    """Test signal_sentinel and wait_for_sentinel work together."""
    sentinel = tmp_path / "nested" / "dir" / "sentinel.txt"

    # Signal creates parent dirs and file
    signal_sentinel(sentinel)
    assert sentinel.exists()

    # Wait returns immediately when file exists
    start = time.monotonic()
    wait_for_sentinel(sentinel, timeout=1.0)
    elapsed = time.monotonic() - start

    # Should return almost immediately (< 100ms)
    assert elapsed < 0.1


def test_wait_for_sentinel_timeout(tmp_path: Path) -> None:
    """Test wait_for_sentinel raises TimeoutError when file doesn't appear."""
    sentinel = tmp_path / "nonexistent.txt"

    import pytest

    start = time.monotonic()
    with pytest.raises(TimeoutError, match="did not appear within"):
        wait_for_sentinel(sentinel, timeout=0.1)
    elapsed = time.monotonic() - start

    # Should wait approximately the timeout duration
    assert 0.09 < elapsed < 0.2


def test_wait_for_sentinel_concurrent_creation(tmp_path: Path) -> None:
    """Test wait_for_sentinel works when file is created concurrently."""
    import threading

    sentinel = tmp_path / "concurrent.txt"

    def create_after_delay():
        time.sleep(0.05)
        signal_sentinel(sentinel)

    thread = threading.Thread(target=create_after_delay, daemon=True)
    thread.start()

    # Wait should succeed once thread creates the file
    start = time.monotonic()
    wait_for_sentinel(sentinel, timeout=1.0)
    elapsed = time.monotonic() - start

    # Should return shortly after file creation (~50-100ms)
    assert 0.04 < elapsed < 0.2
    assert sentinel.exists()

    thread.join()


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
