# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""User-defined counters for Zephyr tasks.

Task code can increment named counters during execution; counters are
aggregated across all tasks and exposed via the coordinator's ``get_counters()``
actor method.

Usage::

    from zephyr import counters

    counters.increment("documents_processed", 100)
    counters.increment("validation_errors")

Counter values are accumulated in-memory on each worker and sent to the
coordinator periodically via heartbeats and as a final snapshot on task
completion.

Outside of a Zephyr worker context, all calls are silent no-ops.
"""

import logging

from zephyr.execution import _worker_ctx_var

logger = logging.getLogger(__name__)


def increment(name: str, value: int = 1) -> None:
    """Increment a named counter by ``value`` (default 1).

    O(1) lock-free in-memory update. No-op outside a Zephyr worker.
    """
    worker = _worker_ctx_var.get()
    if worker is None:
        return
    worker.increment_counter(name, value)


def get_counters() -> dict[str, int]:
    """Return a snapshot of the current task's counters.

    Returns an empty dict outside a Zephyr worker context.
    """
    worker = _worker_ctx_var.get()
    if worker is None:
        return {}
    return worker.get_counter_snapshot().counters
