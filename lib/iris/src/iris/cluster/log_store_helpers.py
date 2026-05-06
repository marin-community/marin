# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris-domain helpers that produce opaque string keys for the finelog store.

Keys are plain strings on the finelog side; this module owns the mapping from
iris-domain values (`JobName`, `TaskAttempt`) to those strings.
"""

from __future__ import annotations

import re

from iris.cluster.types import JobName, TaskAttempt

CONTROLLER_LOG_KEY = "/system/controller"
_WORKER_LOG_PREFIX = "/system/worker/"


def worker_log_key(worker_id: str) -> str:
    """Build the log store key for a worker's process logs."""
    return f"{_WORKER_LOG_PREFIX}{worker_id}"


def task_log_key(task_attempt: TaskAttempt) -> str:
    """Build a hierarchical key for task attempt logs."""
    task_attempt.require_attempt()
    return task_attempt.to_wire()


def build_log_source(target: JobName, attempt_id: int = -1) -> str:
    """Build a FetchLogs source regex pattern from a JobName.

    Escapes regex metacharacters in the job name so they match literally,
    then appends the appropriate wildcard suffix.

    - Task + specific attempt: /user/job/0:<attempt_id>  (exact match)
    - Task + all attempts:     /user/job/0:.*
    - Job (all tasks):         /user/job/.*
    """
    wire = re.escape(target.to_wire())
    if target.is_task:
        if attempt_id >= 0:
            return f"{wire}:{attempt_id}"
        return f"{wire}:.*"
    return f"{wire}/.*"
