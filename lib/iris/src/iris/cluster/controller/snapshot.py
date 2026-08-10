# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Database-neutral inputs for one controller control cycle."""

from collections.abc import Sequence
from dataclasses import dataclass, field

from rigging.timing import Timestamp

from iris.cluster.controller.reconcile.worker import ReconcileRow
from iris.cluster.controller.task_state import RunningTaskEntry
from iris.resources.attempt import AttemptLaunch, AttemptLaunchTemplate
from iris.resources.names import JobName, WorkerId


@dataclass(frozen=True, slots=True)
class ExecutionTimeoutRow:
    """Persisted execution-deadline fields consumed by the control loop."""

    task_id: JobName
    started_at_ms: Timestamp
    timeout_ms: int


@dataclass(frozen=True, slots=True)
class ControlSnapshot:
    """The database-neutral per-tick input consumed by execution backends.

    Persistence builds this value in one read transaction. Backends receive only
    native records and never need to know how those records were stored.
    """

    worker_addresses: dict[WorkerId, str]
    reconcile_rows: list[ReconcileRow]
    timeout_rows: Sequence[ExecutionTimeoutRow]
    launch_templates: dict[JobName, AttemptLaunchTemplate] = field(default_factory=dict)
    tasks_to_run: list[AttemptLaunch] = field(default_factory=list)
    running_tasks: list[RunningTaskEntry] = field(default_factory=list)
