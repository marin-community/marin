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

"""Test utilities for iris. Kept in src so cloudpickle can resolve references."""

import os
import time


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
        with open(self._path, "w"):
            pass

    def reset(self) -> None:
        try:
            os.unlink(self._path)
        except FileNotFoundError:
            pass

    def is_set(self) -> bool:
        return os.path.exists(self._path)

    def wait(self, poll_interval: float = 0.1, timeout: float | None = None) -> None:
        """Block until the sentinel file exists."""
        deadline = time.monotonic() + timeout if timeout is not None else None
        while not os.path.exists(self._path):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"SentinelFile {self._path} not signalled within {timeout}s")
            time.sleep(poll_interval)
