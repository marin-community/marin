# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""User-defined counters for Zephyr tasks.

Task code can increment named counters during execution; counters are
aggregated across all tasks and exposed in ``JobStatus.counters``.

Usage::

    from zephyr import counters

    counters.increment("documents_processed", 100)
    counters.increment("validation_errors")

Counter values are accumulated in-memory on each worker and sent to the
coordinator via the heartbeat loop. The coordinator aggregates per-task
counters and exposes them through ``get_status()``.

Outside of a Zephyr worker context, all calls are silent no-ops.
"""

import logging
import threading

logger = logging.getLogger(__name__)


def increment(name: str, value: int = 1) -> None:
    """Increment a named counter by ``value`` (default 1).

    O(1) in-memory update. Thread-safe. No-op outside a Zephyr worker.
    """
    from zephyr.execution import _worker_ctx_var

    worker = _worker_ctx_var.get()
    if worker is None:
        return
    worker.increment_counter(name, value)


def get_counters() -> dict[str, int]:
    """Return a snapshot of the current task's counters.

    Returns an empty dict outside a Zephyr worker context.
    """
    from zephyr.execution import _worker_ctx_var

    worker = _worker_ctx_var.get()
    if worker is None:
        return {}
    return worker.get_counter_snapshot()


_report_lock = threading.Lock()
_last_reported: dict[str, dict[str, int]] = {}


def counters_changed_since_last_report(worker_id: str, current: dict[str, int]) -> bool:
    """Return True if *current* differs from the last snapshot we reported for *worker_id*."""
    with _report_lock:
        prev = _last_reported.get(worker_id)
        if prev == current:
            return False
        _last_reported[worker_id] = dict(current)
        return True
