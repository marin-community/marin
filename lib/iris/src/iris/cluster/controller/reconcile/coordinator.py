# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controller-owned folding of backend observations into Iris effects."""

from dataclasses import dataclass, field

from rigging.timing import Timestamp

from iris.cluster.controller.backend import DirectTaskObservation, ReconcileObservation, WorkerFleetObservation
from iris.cluster.controller.ops.task import apply_dispatch_updates
from iris.cluster.controller.ops.worker import apply_reconcile
from iris.cluster.controller.reconcile import ControllerEffects
from iris.cluster.controller.reconcile.loader import TransitionReader
from iris.cluster.controller.worker_health import WorkerHealthEvent, WorkerHealthEventKind, WorkerHealthTracker
from iris.cluster.types import WorkerId


@dataclass(frozen=True)
class ReconcileFold:
    """Controller effects and post-commit teardown work from one observation."""

    effects: ControllerEffects = field(default_factory=ControllerEffects)
    dead_workers: list[WorkerId] = field(default_factory=list)


def fold_reconcile(
    source: TransitionReader,
    observation: ReconcileObservation,
    *,
    worker_health: WorkerHealthTracker | None,
    now: Timestamp,
) -> ReconcileFold:
    """Resolve backend facts against a fresh controller snapshot."""
    if isinstance(observation, DirectTaskObservation):
        if worker_health is not None:
            raise ValueError("Kubernetes reconciliation cannot use worker liveness")
        return ReconcileFold(effects=apply_dispatch_updates(source, observation.updates, now=now))

    assert isinstance(observation, WorkerFleetObservation)
    if worker_health is None:
        raise ValueError("worker reconciliation requires a liveness tracker")
    effects = apply_reconcile(source, observation.worker_results, now=now)
    events = observation.transport_events + [
        WorkerHealthEvent(worker_id, WorkerHealthEventKind.BUILD_FAILED) for worker_id in effects.health.build_failed
    ]
    return ReconcileFold(
        effects=effects,
        dead_workers=worker_health.apply(events, now_ms=now.epoch_ms()),
    )
