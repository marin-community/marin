# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Overlap cyclic-GC scans across training ranks at shared step boundaries."""

import gc
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import jax
import levanter.tracker
from levanter.callbacks import StepInfo

GC_WARMUP_STEPS = 10
GC_TIME_METRIC = "throughput/gc_time"


def collect_garbage() -> float:
    """Collect locally and return the pause duration, without a cross-rank barrier."""
    with jax.profiler.TraceAnnotation("garbage_collection"):
        started = time.perf_counter()
        gc.collect()
        return time.perf_counter() - started


@contextmanager
def coordinated_gc() -> Iterator[Callable[[StepInfo], None]]:
    """Collect after warmup and at shared steps; restore the caller's GC policy.

    Enter after warmup and register the yielded hook at the collection interval.
    The hook also collects on the runner's forced final callback pass.
    """
    was_enabled = gc.isenabled()
    gc.disable()

    def collect(info: StepInfo) -> None:
        # Training collectives bound rank skew; early arrivals can start collecting immediately.
        levanter.tracker.log({GC_TIME_METRIC: collect_garbage()}, step=info.step)

    try:
        collect_garbage()
        yield collect
    finally:
        if was_enabled:
            gc.enable()
        else:
            gc.disable()
