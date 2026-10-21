# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tracked worker registry for the autoscaler."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from rigging.timing import Duration

from iris.cluster.platforms.types import (
    CloudWorkerState,
    CommandResult,
    RemoteWorkerHandle,
    WorkerStatus,
)


class _RestoredWorkerHandle:
    """Minimal handle placeholder used for restored tracked workers."""

    def __init__(self, worker_id: str, internal_address: str, port: int) -> None:
        self._worker_id = worker_id
        self._internal_address = internal_address
        self._port = port

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
    def worker_url(self) -> str:
        return f"http://{self._internal_address}:{self._port}"

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


def restore_tracked_workers(rows: list[TrackedWorkerRow]) -> dict[str, TrackedWorker]:
    """Restore tracked workers from DB rows.

    ``row.address`` is the ``host:port`` the worker self-reported at
    registration; it is split into the host/port pair the handle carries.
    """

    workers: dict[str, TrackedWorker] = {}
    for row in rows:
        host, sep, port = row.address.rpartition(":")
        if not sep or not host or not port.isdigit():
            raise ValueError(f"restored worker {row.worker_id} has malformed address: {row.address!r}")
        handle = _RestoredWorkerHandle(worker_id=row.worker_id, internal_address=host, port=int(port))
        workers[row.worker_id] = TrackedWorker(
            worker_id=row.worker_id,
            slice_id=row.slice_id,
            scale_group=row.scale_group,
            handle=handle,
        )
    return workers
