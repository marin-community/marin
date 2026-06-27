# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""RemoteTaskBackend: the root-side CLUSTER_VIEW backend for a remote cluster.

The root never touches the remote cluster directly. Each tick ``reconcile``
publishes the desired attempt set into the backend's :class:`BackendRelay` and
drains the observations a remote agent has reported through the poll RPC,
turning them into neutral ``TaskUpdate``s. The agent owns the real in-cluster
backend; this backend is the root's view of it.
"""

import logging
from typing import ClassVar

from iris.cluster.backends.remote.relay import BackendRelay
from iris.cluster.controller.autoscaler import Autoscaler
from iris.cluster.controller.autoscaler.models import DemandEntry
from iris.cluster.controller.backend import (
    AutoscaleResult,
    BackendCapability,
    ProviderUnsupportedError,
    ReconcileResult,
    ScheduleInput,
    ScheduleResult,
    TaskTarget,
)
from iris.cluster.controller.reads import ControlSnapshot
from iris.cluster.controller.reconcile.snapshot import TaskUpdate
from iris.cluster.types import WorkerId
from iris.rpc import job_pb2, worker_pb2

logger = logging.getLogger(__name__)


class RemoteTaskBackend:
    """Root-side view of a remote cluster's task execution.

    A CLUSTER_VIEW backend: the remote agent owns placement, so ``schedule`` and
    ``autoscale`` are no-ops and ``reconcile`` exchanges desired state for
    observations through the relay. Interactive ops are not supported on the
    root side.
    """

    capabilities: ClassVar[frozenset[BackendCapability]] = frozenset({BackendCapability.CLUSTER_VIEW})

    def __init__(self, name: str = "remote") -> None:
        self.name = name
        self.relay = BackendRelay()
        # A remote CLUSTER_VIEW backend manages its own capacity through the
        # agent; the Iris autoscaler never drives it.
        self.autoscaler: Autoscaler | None = None

    def schedule(self, snapshot: ScheduleInput) -> ScheduleResult:
        """No-op: the remote agent owns placement."""
        return ScheduleResult()

    def reconcile(self, snapshot: ControlSnapshot) -> ReconcileResult:
        """Publish the desired set and fold drained agent observations into updates."""
        self.relay.set_desired(snapshot.tasks_to_run, snapshot.running_tasks)

        updates: list[TaskUpdate] = []
        for obs in self.relay.take_observations():
            resolved = self.relay.resolve(obs.attempt_uid)
            if resolved is None:
                logger.debug("remote relay: observation for unknown uid %s; dropping", obs.attempt_uid)
                continue
            task_id, attempt_id = resolved
            updates.append(
                TaskUpdate(
                    task_id=task_id,
                    attempt_id=attempt_id,
                    new_state=obs.state,
                    exit_code=(obs.exit_code or None),
                    error=(obs.message or None),
                )
            )
        return ReconcileResult(updates=updates)

    def autoscale(
        self,
        snapshot: ControlSnapshot,
        residual_demand: list[DemandEntry],
        dead_workers: list[WorkerId],
    ) -> AutoscaleResult:
        """No-op: the remote agent provisions its own capacity."""
        return AutoscaleResult()

    def attach_autoscaler(self, autoscaler: Autoscaler) -> None:
        """Never called: a remote CLUSTER_VIEW backend never receives an autoscaler."""
        raise ProviderUnsupportedError("RemoteTaskBackend manages capacity through its agent; no autoscaler is attached")

    def get_process_status(
        self,
        target: TaskTarget,
        request: job_pb2.GetProcessStatusRequest,
    ) -> job_pb2.GetProcessStatusResponse:
        raise ProviderUnsupportedError("remote backend does not support interactive ops")

    def profile_task(
        self,
        target: TaskTarget,
        request: job_pb2.ProfileTaskRequest,
        timeout_ms: int,
    ) -> job_pb2.ProfileTaskResponse:
        raise ProviderUnsupportedError("remote backend does not support interactive ops")

    def exec_in_container(
        self,
        target: TaskTarget,
        request: worker_pb2.Worker.ExecInContainerRequest,
        timeout_seconds: int = 60,
    ) -> worker_pb2.Worker.ExecInContainerResponse:
        raise ProviderUnsupportedError("remote backend does not support interactive ops")

    def close(self) -> None:
        """No-op: the backend owns no connections or background threads."""
