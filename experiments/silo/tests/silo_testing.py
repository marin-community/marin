# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Small helpers shared by the silo tests."""

import time


def wait_until(predicate, timeout: float = 10.0, interval: float = 0.02) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError("condition not met in time")
