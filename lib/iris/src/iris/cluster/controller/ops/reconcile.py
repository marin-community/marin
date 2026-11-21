# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Apply one backend reconciliation observation to controller-owned state."""

from dataclasses import dataclass

from rigging.timing import Timestamp

from iris.cluster.controller.ops.task import apply_reconcile_updates
from iris.cluster.controller.reconcile.effects import ControllerEffects
from iris.cluster.controller.reconcile.loader import TransitionReader
from iris.cluster.controller.reconcile.snapshot import TaskUpdate
from iris.cluster.controller.worker_health import (
    WorkerHealthEvent,
    WorkerHealthEventKind,
    WorkerHealthTracker,
)
from iris.cluster.types import WorkerId


@dataclass(frozen=True, slots=True)
class ReconcileApplication:
    effects: ControllerEffects
    reaped_workers: list[WorkerId]


def apply_observation(
    source: TransitionReader,
    task_updates: list[TaskUpdate],
    health_events: list[WorkerHealthEvent],
    *,
    worker_health: WorkerHealthTracker | None,
    now: Timestamp,
) -> ReconcileApplication:
    """Apply exact task facts and orthogonal worker-health facts once."""
    effects = apply_reconcile_updates(source, task_updates, now=now)
    events = health_events + [
        WorkerHealthEvent(worker_id, WorkerHealthEventKind.BUILD_FAILED) for worker_id in effects.health.build_failed
    ]
    if not events:
        return ReconcileApplication(effects=effects, reaped_workers=[])
    if worker_health is None:
        raise ValueError("backend reported worker health without a worker health tracker")
    return ReconcileApplication(
        effects=effects,
        reaped_workers=worker_health.apply(events, now_ms=now.epoch_ms()),
    )
