# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Overlap cyclic-GC scans across training ranks at shared step boundaries."""

import gc
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import jax

GC_WARMUP_STEPS = 10


def collect_garbage() -> float:
    """Collect locally and return the pause duration, without a cross-rank barrier."""
    with jax.profiler.TraceAnnotation("garbage_collection"):
        started = time.perf_counter()
        gc.collect()
        return time.perf_counter() - started


@contextmanager
def coordinated_gc(interval: int) -> Iterator[Callable[[int], float]]:
    """Collect after warmup and at shared steps; restore the caller's GC policy.

    The caller validates a positive interval before dispatch. Training collectives
    bound rank skew; adding GC barriers would delay ranks that could collect early.
    """
    was_enabled = gc.isenabled()
    gc.disable()

    def collect(step: int) -> float:
        if step % interval:
            return 0.0
        return collect_garbage()

    try:
        collect_garbage()
        yield collect
    finally:
        if was_enabled:
            gc.enable()
        else:
            gc.disable()
