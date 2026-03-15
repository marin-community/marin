# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Task identity for multi-replica jobs.

Each replica within a multi-replica job gets a unique 0-based task index.
Workers use ``get_task_index()`` to discover their index at runtime,
regardless of which backend (Local, Iris, Ray) is running the job.
"""

import threading

_local = threading.local()


def get_task_index() -> int:
    """Return the current task's replica index (0-based).

    Set by the backend before invoking the job entrypoint:
    - LocalClient: sets per-thread before calling the callable.
    - Iris: set from ``get_job_info().task_index`` by the entrypoint.
    - Ray: similar to Iris.

    Falls back to 0 for single-replica jobs.
    """
    return getattr(_local, "task_index", 0)


def set_task_index(index: int) -> None:
    """Set the current task's replica index (called by backends)."""
    _local.task_index = index
