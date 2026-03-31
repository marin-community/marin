# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test utilities for iris. Kept in src so cloudpickle can resolve references."""

import os
import time
from collections.abc import Callable

from rigging.timing import Deadline, Duration


class SentinelFile:
    """File-based signal for cross-thread/cross-process coordination in tests.

    Stores a plain str path so it passes through cloudpickle without issues.
    Can be passed directly into submitted jobs.
    """

    def __init__(self, path: str):
        self._path = path

    @property
    def path(self) -> str:
        return self._path

    def signal(self) -> None:
        dirname = os.path.dirname(self._path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(self._path, "w"):
            pass

    def reset(self) -> None:
        try:
            os.unlink(self._path)
        except FileNotFoundError:
            pass

    def is_set(self) -> bool:
        return os.path.exists(self._path)

    def wait(self, poll_interval: float = 0.1, timeout: Duration | None = None) -> None:
        """Block until the sentinel file exists."""
        deadline = Deadline.from_now(timeout) if timeout is not None else None
        while not os.path.exists(self._path):
            if deadline is not None and deadline.expired():
                raise TimeoutError(f"SentinelFile {self._path} not signalled within {timeout}")
            time.sleep(poll_interval)


def wait_for_condition(
    condition: Callable[[], bool], timeout: Duration = Duration.from_seconds(10.0), poll_interval: float = 0.01
) -> None:
    """Poll condition at regular intervals until it returns True or timeout is reached.

    Raises:
        TimeoutError: If the condition does not become true within timeout.
    """
    deadline = Deadline.from_now(timeout)
    while not deadline.expired():
        if condition():
            return
        time.sleep(poll_interval)
    raise TimeoutError(f"Condition did not become true within {timeout}")
