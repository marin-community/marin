# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Global chaos injection for testing distributed failure scenarios.

Usage:
    from iris.chaos import chaos, chaos_raise, enable_chaos, reset_chaos

    enable_chaos("controller.dispatch", failure_rate=0.3)

    # Site decides what to do:
    if chaos("controller.dispatch") is not None:
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

    def try_fire(self) -> bool:
        """Atomically check and increment the failure counter.

        Returns True if chaos should fire, False otherwise.
        Does NOT sleep - call sites must explicitly handle delay_seconds.
        """
        with self._lock:
            if self.max_failures is not None and self._failure_count >= self.max_failures:
                return False
            if random.random() >= self.failure_rate:
                return False
            self._failure_count += 1
        return True


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


def chaos(key: str) -> ChaosRule | None:
    """Check if chaos should fire for this key.

    Returns the fired rule (with delay_seconds, error, etc.), or None.
    Call sites must explicitly handle delay_seconds and error.
    No side effects - use walrus operator pattern:

        if rule := chaos("worker.building_delay"):
            time.sleep(rule.delay_seconds)
    """
    rule = _rules.get(key)
    if rule is None:
        return None
    if rule.try_fire():
        return rule
    return None


def chaos_raise(key: str) -> None:
    """Convenience: raise an exception if chaos fires for this key.

    Handles delay_seconds before raising.
    """
    if rule := chaos(key):
        time.sleep(rule.delay_seconds)
        raise rule.error or RuntimeError(f"chaos: {key}")


def reset_chaos() -> None:
    _rules.clear()
