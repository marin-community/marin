# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tracked worker registry for the autoscaler."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from rigging.timing import Duration

from iris.cluster.providers.types import (
    CloudWorkerState,
    CommandResult,
    RemoteWorkerHandle,
    WorkerStatus,
)
from iris.rpc import vm_pb2


class _RestoredWorkerHandle:
    """Minimal handle placeholder used for restored tracked workers."""

    def __init__(self, worker_id: str, internal_address: str) -> None:
        self._worker_id = worker_id
        self._internal_address = internal_address

    @property
    def worker_id(self) -> str:
        return self._worker_id

    @property
    def vm_id(self) -> str:
        return self._worker_id

    @property
    def internal_address(self) -> str:
        return self._internal_address

    @property
    def external_address(self) -> str | None:
        return None

    @property
    def bootstrap_log(self) -> str:
        return ""

    def status(self) -> WorkerStatus:
        return WorkerStatus(state=CloudWorkerState.RUNNING)

    def run_command(
        self,
        command: str,
        timeout: Duration | None = None,
        on_line: Callable[[str], None] | None = None,
    ) -> CommandResult:
        del command, timeout, on_line
        raise NotImplementedError("RestoredWorkerHandle does not support run_command")

    def reboot(self) -> None:
        raise NotImplementedError("RestoredWorkerHandle does not support reboot")


@dataclass(frozen=True)
class TrackedWorkerRow:
    """Tracked worker row restored from the workers table."""

    worker_id: str
    slice_id: str
    scale_group: str
    address: str


@dataclass
class TrackedWorker:
    """Per-worker state tracked by the autoscaler across bootstrap and lifecycle."""

    worker_id: str
    slice_id: str
    scale_group: str
    handle: RemoteWorkerHandle
    bootstrap_log: str = ""


@dataclass
class WorkerRegistry:
    """In-memory registry for live and restored worker handles."""

    workers: dict[str, TrackedWorker] = field(default_factory=dict)

    def register_slice_workers(self, workers: list[RemoteWorkerHandle], slice_id: str, scale_group: str) -> None:
        """Register all workers from a slice into the handle cache."""

        for worker in workers:
            self.workers[worker.worker_id] = TrackedWorker(
                worker_id=worker.worker_id,
                slice_id=slice_id,
                scale_group=scale_group,
                handle=worker,
                bootstrap_log=worker.bootstrap_log,
            )

    def unregister_slice_workers(self, slice_id: str, worker_ids: Sequence[str] | None = None) -> None:
        """Remove tracked workers belonging to a slice from the handle cache."""

        to_remove = (
            list(worker_ids)
            if worker_ids is not None
            else [worker_id for worker_id, tracked in self.workers.items() if tracked.slice_id == slice_id]
        )
        for worker_id in to_remove:
            self.workers.pop(worker_id, None)

    def restore(self, workers: dict[str, TrackedWorker]) -> None:
        """Restore tracked worker state from a snapshot."""

        self.workers.update(workers)

    def tracked_worker(self, worker_id: str) -> TrackedWorker | None:
        """Look up a tracked worker by ID."""

        return self.workers.get(worker_id)

    def vm_info(self, vm_id: str) -> vm_pb2.VmInfo | None:
        """Build VM status for a tracked worker."""

        tracked = self.workers.get(vm_id)
        if tracked is None:
            return None

        worker_status = tracked.handle.status()
        if worker_status.state == CloudWorkerState.RUNNING:
            iris_state = vm_pb2.VM_STATE_READY
        elif worker_status.state == CloudWorkerState.STOPPED:
            iris_state = vm_pb2.VM_STATE_FAILED
        elif worker_status.state == CloudWorkerState.TERMINATED:
            iris_state = vm_pb2.VM_STATE_TERMINATED
        else:
            iris_state = vm_pb2.VM_STATE_BOOTING

        return vm_pb2.VmInfo(
            vm_id=tracked.worker_id,
            state=iris_state,
            address=tracked.handle.internal_address,
            scale_group=tracked.scale_group,
            slice_id=tracked.slice_id,
        )

    def init_log(self, vm_id: str, tail: int | None = None) -> str:
        """Get bootstrap log for a tracked worker."""

        tracked = self.workers.get(vm_id)
        if tracked is None:
            return ""
        log = tracked.bootstrap_log
        if tail and log:
            lines = log.splitlines()
            return "\n".join(lines[-tail:])
        return log


def restore_tracked_workers(rows: list[TrackedWorkerRow]) -> dict[str, TrackedWorker]:
    """Restore tracked workers from DB rows."""

    workers: dict[str, TrackedWorker] = {}
    for row in rows:
        handle = _RestoredWorkerHandle(worker_id=row.worker_id, internal_address=row.address)
        workers[row.worker_id] = TrackedWorker(
            worker_id=row.worker_id,
            slice_id=row.slice_id,
            scale_group=row.scale_group,
            handle=handle,
        )
    return workers
