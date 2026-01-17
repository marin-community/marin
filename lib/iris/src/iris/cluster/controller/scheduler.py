# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pure job-to-worker matching without threading, dispatch, or state mutation."""

import logging
from dataclasses import dataclass, field

from iris.cluster.controller.state import ControllerJob, ControllerState, ControllerWorker
from iris.cluster.types import WorkerId
from iris.rpc import cluster_pb2

logger = logging.getLogger(__name__)


def get_device_type(device: cluster_pb2.DeviceConfig) -> str:
    if device.HasField("cpu"):
        return "cpu"
    elif device.HasField("gpu"):
        return "gpu"
    elif device.HasField("tpu"):
        return "tpu"
    return "cpu"  # Default to CPU if no device specified


def get_device_variant(device: cluster_pb2.DeviceConfig) -> str | None:
    if device.HasField("gpu"):
        return device.gpu.variant if device.gpu.variant else None
    elif device.HasField("tpu"):
        return device.tpu.variant if device.tpu.variant else None
    return None


def get_gpu_count(device: cluster_pb2.DeviceConfig) -> int:
    if device.HasField("gpu"):
        return device.gpu.count or 1
    return 0


@dataclass
class WorkerCapacity:
    """Available resources on a worker (mutable during scheduling)."""

    available_cpu: int
    available_memory: int  # bytes
    available_gpus: int
    device_type: str  # "cpu", "gpu", "tpu"
    device_variant: str | None


def worker_can_fit_job(capacity: WorkerCapacity, job: ControllerJob) -> bool:
    """Check if worker capacity can fit job. Pure function, no state."""
    res = job.request.resources

    if res.cpu > capacity.available_cpu:
        return False

    if res.memory_bytes > capacity.available_memory:
        return False

    job_device_type = get_device_type(res.device)
    if job_device_type != capacity.device_type:
        return False

    job_variant = get_device_variant(res.device)
    if job_variant and job_variant != "auto" and job_variant != capacity.device_variant:
        return False

    if job_device_type == "gpu" and get_gpu_count(res.device) > capacity.available_gpus:
        return False

    return True


def deduct_job_from_capacity(capacity: WorkerCapacity, job: ControllerJob) -> None:
    """Deduct job's resources from capacity (mutates capacity)."""
    res = job.request.resources
    capacity.available_cpu -= res.cpu
    capacity.available_memory -= res.memory_bytes
    capacity.available_gpus -= get_gpu_count(res.device)


def compute_worker_capacity(state: ControllerState, worker: ControllerWorker) -> WorkerCapacity:
    """Compute current available capacity for a worker."""
    committed_cpu, committed_mem, committed_gpu = state.get_committed_resources(worker)
    res = worker.resources
    return WorkerCapacity(
        available_cpu=res.cpu - committed_cpu,
        available_memory=res.memory_bytes - committed_mem,
        available_gpus=get_gpu_count(res.device) - committed_gpu,
        device_type=get_device_type(res.device),
        device_variant=get_device_variant(res.device),
    )


@dataclass
class SchedulingTransaction:
    """Accumulates tentative assignments that can be rolled back."""

    state: ControllerState
    assignments: list[tuple[ControllerJob, ControllerWorker]] = field(default_factory=list)
    timed_out_jobs: list[ControllerJob] = field(default_factory=list)

    def tentatively_assign(self, job: ControllerJob, worker: ControllerWorker) -> None:
        """Assign job to worker immediately (updates running_jobs and removes from queue)."""
        self.state.assign_job_to_worker(worker.worker_id, job.job_id)
        self.assignments.append((job, worker))

    def rollback_assignment(self, job: ControllerJob, worker: ControllerWorker) -> None:
        """Rollback a single failed assignment."""
        self.state.rollback_assignment(worker.worker_id, job)


def build_capacity_map(state: ControllerState, workers: list[ControllerWorker]) -> dict[WorkerId, WorkerCapacity]:
    """Build capacity map for all healthy workers."""
    return {w.worker_id: compute_worker_capacity(state, w) for w in workers if w.healthy}


class Scheduler:
    """Pure job-to-worker matching logic. Does not dispatch jobs, modify state, or run threads."""

    def __init__(self, state: ControllerState):
        self._state = state

    def find_assignments(
        self,
        pending_jobs: list[ControllerJob],
        workers: list[ControllerWorker],
        now_ms: int,
    ) -> SchedulingTransaction:
        """Match pending jobs to available workers one at a time.

        Uses first-fit algorithm, skipping jobs that don't fit any worker.
        Also identifies jobs that have exceeded their scheduling timeout.

        The algorithm prevents head-of-line blocking: if a large job at the
        front of the queue doesn't fit, smaller jobs behind it can still be
        scheduled.

        Args:
            pending_jobs: Jobs waiting to be scheduled (in FIFO order)
            workers: Available workers (only healthy ones should be passed)
            now_ms: Current timestamp in milliseconds

        Returns:
            SchedulingTransaction with assignments and timed-out jobs
        """
        transaction = SchedulingTransaction(self._state)
        capacities = build_capacity_map(self._state, workers)

        for job in pending_jobs:
            if self._is_job_timed_out(job, now_ms):
                transaction.timed_out_jobs.append(job)
                continue

            for worker in workers:
                if not worker.healthy:
                    continue
                capacity = capacities[worker.worker_id]
                if worker_can_fit_job(capacity, job):
                    deduct_job_from_capacity(capacity, job)
                    transaction.tentatively_assign(job, worker)
                    break
            else:
                logger.debug(
                    "No suitable worker for job %s (cpu=%d, memory_bytes=%d)",
                    job.job_id,
                    job.request.resources.cpu,
                    job.request.resources.memory_bytes,
                )

        if transaction.assignments or transaction.timed_out_jobs:
            logger.debug(
                "Scheduling cycle: %d pending, %d assigned, %d timed_out",
                len(pending_jobs),
                len(transaction.assignments),
                len(transaction.timed_out_jobs),
            )
        return transaction

    def _is_job_timed_out(self, job: ControllerJob, now_ms: int) -> bool:
        timeout_seconds = job.request.scheduling_timeout_seconds
        if timeout_seconds <= 0:
            return False

        pending_duration_ms = now_ms - job.submitted_at_ms
        timeout_ms = timeout_seconds * 1000
        return pending_duration_ms > timeout_ms
