# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Structural Protocols for SA Row shapes that cross module boundaries.

Protocols here describe the attribute contract of SQLAlchemy ``Row`` objects
returned by specific read helpers.  They are static typing aids only; no
``@runtime_checkable`` decorator is used because no ``isinstance`` check is
needed anywhere in the controller.
"""

from typing import Protocol

from rigging.timing import Timestamp

from iris.cluster.types import JobName


class SchedulerTaskRow(Protocol):
    """Shape of the rows returned by ``_schedulable_tasks`` in ``controller.py``.

    Consumed by ``task_row_can_be_scheduled``, ``_sort_pending_tasks_by_resolved_band``,
    and the scheduler gating/ordering logic in the scheduling loop.
    """

    task_id: JobName
    job_id: JobName
    state: int
    current_attempt_id: int
    failure_count: int
    preemption_count: int
    max_retries_failure: int
    max_retries_preemption: int
    submitted_at_ms: Timestamp
    priority_band: int
    priority_neg_depth: int
    priority_root_submitted_ms: int
    priority_insertion: int
