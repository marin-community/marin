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

"""Global chaos injection for testing distributed failure scenarios.

Usage:
    from iris.chaos import chaos, chaos_raise, enable_chaos, reset_chaos

    enable_chaos("controller.dispatch", failure_rate=0.3)

    # Site decides what to do:
    if chaos("controller.dispatch"):
        raise Exception("chaos: dispatch failed")

    # Or use helper for simple raise cases:
    chaos_raise("worker.bundle_download")

    reset_chaos()
"""

import random
import time
import threading
from dataclasses import dataclass, field


@dataclass
class ChaosRule:
    failure_rate: float = 1.0
    error: Exception | None = None
    delay_seconds: float = 0.0
    max_failures: int | None = None
    _failure_count: int = field(default=0, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def should_fire(self) -> bool:
        with self._lock:
            if self.max_failures is not None and self._failure_count >= self.max_failures:
                return False
            return random.random() < self.failure_rate

    def fire(self) -> None:
        with self._lock:
            self._failure_count += 1
        if self.delay_seconds > 0:
            time.sleep(self.delay_seconds)


_rules: dict[str, ChaosRule] = {}


def enable_chaos(
    key: str,
    failure_rate: float = 1.0,
    error: Exception | None = None,
    delay_seconds: float = 0.0,
    max_failures: int | None = None,
) -> None:
    _rules[key] = ChaosRule(
        failure_rate=failure_rate,
        error=error,
        delay_seconds=delay_seconds,
        max_failures=max_failures,
    )


def chaos(key: str) -> bool:
    """Check if chaos should fire for this key.

    Returns True if chaos is active and fires, False otherwise.
    Injection sites decide what to do with the signal.
    """
    rule = _rules.get(key)
    if rule is None:
        return False
    if rule.should_fire():
        rule.fire()  # increment counter, apply delay
        return True
    return False


def chaos_raise(key: str) -> None:
    """Convenience: raise an exception if chaos fires for this key."""
    if chaos(key):
        rule = _rules[key]
        if rule.error:
            raise rule.error
        raise RuntimeError(f"chaos: {key}")


def reset_chaos() -> None:
    _rules.clear()
