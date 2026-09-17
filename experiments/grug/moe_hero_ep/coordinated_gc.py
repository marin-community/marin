# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Overlap cyclic-GC scans across training ranks at shared step boundaries."""

import gc
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from jax.experimental import multihost_utils

GC_WARMUP_STEPS = 10


@contextmanager
def coordinated_gc(interval: int) -> Iterator[Callable[[int], float]]:
    """Collect at shared completed-step numbers and restore the caller's GC policy.

    Enter after compilation warmup. Every rank must invoke the yielded function
    with identical step numbers. Its returned time includes both barriers.
    """
    if interval <= 0:
        raise ValueError("GC interval must be positive")
    was_enabled = gc.isenabled()
    gc.disable()

    def collect(step: int) -> float:
        if step % interval:
            return 0.0
        started = time.perf_counter()
        multihost_utils.sync_global_devices(f"gc-start-{step}")
        gc.collect()
        multihost_utils.sync_global_devices(f"gc-done-{step}")
        return time.perf_counter() - started

    try:
        yield collect
    finally:
        if was_enabled:
            gc.enable()
        else:
            gc.disable()
