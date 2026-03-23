# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskProvider adapter implementations.

WorkerProviderAdapter: wraps WorkerProvider (heartbeat RPC model)
DirectProviderAdapter: wraps KubernetesProvider (direct-pod model)

Each adapter translates between the controller's unified TaskProvider
interface and the underlying provider's native types.
"""

import logging
from dataclasses import dataclass

from iris.cluster.controller.provider import (
    FailedWorker,
    ProviderSyncOutcome,
    ProviderUnsupportedError,
)
from iris.cluster.controller.transitions import (
    ControllerTransitions,
    HeartbeatAction,
)
from iris.cluster.controller.worker_provider import WorkerProvider
from iris.cluster.k8s.provider import KubernetesProvider
from iris.cluster.types import JobName, WorkerId
from iris.rpc import cluster_pb2, logging_pb2

logger = logging.getLogger(__name__)


@dataclass
class WorkerProviderAdapter:
    """TaskProvider backed by worker daemons via heartbeat RPC.

    Wraps a WorkerProvider and implements the unified sync cycle:
    drain per-worker batches -> parallel heartbeat dispatch -> apply results.
    """

    _inner: WorkerProvider

    @property
    def has_workers(self) -> bool:
        return True

    def sync(self, transitions: ControllerTransitions) -> ProviderSyncOutcome:
        """Drain per-worker batches, dispatch heartbeats, apply results."""
        batches = transitions.drain_dispatch_all()
        if not batches:
            return ProviderSyncOutcome()

        results = self._inner.sync(batches)

        # Separate successes from failures for batch application.
        success_reqs = []
        failure_entries = []
        for batch, apply_req, error in results:
            if apply_req is not None:
                success_reqs.append(apply_req)
            else:
                failure_entries.append((batch, error or "unknown error"))

        # Batch all successful heartbeats in one transaction.
        all_tasks_to_kill: set[JobName] = set()
        if success_reqs:
            batch_results = transitions.apply_heartbeats_batch(success_reqs)
            for result in batch_results:
                all_tasks_to_kill.update(result.tasks_to_kill)

        # Handle failures individually (rare, need per-worker side effects).
        failed_workers: list[FailedWorker] = []
        error_count = 0
        error_worker_ids: list[str] = []
        for batch, error in failure_entries:
            logger.debug("Sync error for %s: %s", batch.worker_id, error)
            action = transitions.fail_heartbeat(batch, error)
            if action == HeartbeatAction.WORKER_FAILED:
                error_count += 1
                error_worker_ids.append(batch.worker_id)
                failed_workers.append(
                    FailedWorker(
                        worker_id=batch.worker_id,
                        address=batch.worker_address,
                        is_permanent=True,
                    )
                )
            elif action == HeartbeatAction.TRANSIENT_FAILURE:
                error_count += 1
                error_worker_ids.append(batch.worker_id)
                failed_workers.append(
                    FailedWorker(
                        worker_id=batch.worker_id,
                        address=batch.worker_address,
                        is_permanent=False,
                    )
                )

        return ProviderSyncOutcome(
            tasks_to_kill=all_tasks_to_kill,
            failed_workers=failed_workers,
            batch_count=len(batches),
            error_count=error_count,
            error_worker_ids=error_worker_ids,
        )

    def kill_unmapped_tasks(
        self,
        task_ids: set[JobName],
        transitions: ControllerTransitions,
    ) -> bool:
        # Worker providers don't have unmapped tasks to kill.
        return False

    def fetch_live_logs(
        self,
        task_id: str,
        attempt_id: int,
        cursor: int,
        max_lines: int,
        worker_id: WorkerId | None = None,
        address: str | None = None,
    ) -> tuple[list[logging_pb2.LogEntry], int]:
        return self._inner.fetch_live_logs(worker_id, address, task_id, attempt_id, cursor, max_lines)

    def fetch_process_logs(
        self,
        worker_id: WorkerId,
        address: str | None,
        request: cluster_pb2.FetchLogsRequest,
    ) -> tuple[list[logging_pb2.LogEntry], int]:
        return self._inner.fetch_process_logs(worker_id, address, request)

    def get_process_status(
        self,
        worker_id: WorkerId,
        address: str | None,
        request: cluster_pb2.GetProcessStatusRequest,
    ) -> cluster_pb2.GetProcessStatusResponse:
        return self._inner.get_process_status(worker_id, address, request)

    def on_worker_failed(self, worker_id: WorkerId, address: str | None) -> None:
        self._inner.on_worker_failed(worker_id, address)

    def profile_task(
        self,
        task_id: str,
        attempt_id: int,
        address: str | None,
        request: cluster_pb2.ProfileTaskRequest,
        timeout_ms: int,
    ) -> cluster_pb2.ProfileTaskResponse:
        if not address:
            raise ProviderUnsupportedError("Worker provider requires an address for profiling")
        return self._inner.profile_task(address, request, timeout_ms)

    def exec_in_container(
        self,
        address: str,
        request: cluster_pb2.Worker.ExecInContainerRequest,
        timeout_seconds: int,
    ) -> cluster_pb2.Worker.ExecInContainerResponse:
        return self._inner.exec_in_container(address, request, timeout_seconds)

    def get_cluster_status(self) -> cluster_pb2.Controller.GetKubernetesClusterStatusResponse:
        return cluster_pb2.Controller.GetKubernetesClusterStatusResponse()

    def close(self) -> None:
        self._inner.close()


@dataclass
class DirectProviderAdapter:
    """TaskProvider backed by Kubernetes pods without worker daemons.

    Wraps a KubernetesProvider and implements the unified sync cycle:
    drain direct batch -> sync pods -> apply updates.
    """

    _inner: KubernetesProvider

    @property
    def has_workers(self) -> bool:
        return False

    def sync(self, transitions: ControllerTransitions) -> ProviderSyncOutcome:
        """Drain direct batch, sync with k8s, apply updates."""
        batch = transitions.drain_for_direct_provider()
        if not batch.tasks_to_run and not batch.running_tasks and not batch.tasks_to_kill:
            return ProviderSyncOutcome()

        result = self._inner.sync(batch)
        tx_result = transitions.apply_direct_provider_updates(result.updates)

        return ProviderSyncOutcome(
            tasks_to_kill=tx_result.tasks_to_kill,
            scheduling_events=list(result.scheduling_events) if result.scheduling_events else [],
            capacity=result.capacity,
            batch_count=1,
        )

    def kill_unmapped_tasks(
        self,
        task_ids: set[JobName],
        transitions: ControllerTransitions,
    ) -> bool:
        any_buffered = False
        for task_id in task_ids:
            transitions.buffer_direct_kill(task_id.to_wire())
            any_buffered = True
        return any_buffered

    def fetch_live_logs(
        self,
        task_id: str,
        attempt_id: int,
        cursor: int,
        max_lines: int,
        worker_id: WorkerId | None = None,
        address: str | None = None,
    ) -> tuple[list[logging_pb2.LogEntry], int]:
        return self._inner.fetch_live_logs(task_id, attempt_id, cursor, max_lines)

    def fetch_process_logs(
        self,
        worker_id: WorkerId,
        address: str | None,
        request: cluster_pb2.FetchLogsRequest,
    ) -> tuple[list[logging_pb2.LogEntry], int]:
        raise ProviderUnsupportedError("Direct provider does not have worker process logs")

    def get_process_status(
        self,
        worker_id: WorkerId,
        address: str | None,
        request: cluster_pb2.GetProcessStatusRequest,
    ) -> cluster_pb2.GetProcessStatusResponse:
        raise ProviderUnsupportedError("Direct provider does not have worker processes")

    def on_worker_failed(self, worker_id: WorkerId, address: str | None) -> None:
        pass  # No persistent worker connections to evict.

    def profile_task(
        self,
        task_id: str,
        attempt_id: int,
        address: str | None,
        request: cluster_pb2.ProfileTaskRequest,
        timeout_ms: int,
    ) -> cluster_pb2.ProfileTaskResponse:
        return self._inner.profile_task(task_id, attempt_id, request)

    def exec_in_container(
        self,
        address: str,
        request: cluster_pb2.Worker.ExecInContainerRequest,
        timeout_seconds: int,
    ) -> cluster_pb2.Worker.ExecInContainerResponse:
        raise ProviderUnsupportedError("Direct provider does not support exec_in_container")

    def get_cluster_status(self) -> cluster_pb2.Controller.GetKubernetesClusterStatusResponse:
        return self._inner.get_cluster_status()

    def close(self) -> None:
        self._inner.close()
