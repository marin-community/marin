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

"""Pure task-to-worker matching without threading, dispatch, or state mutation."""

import logging
from dataclasses import dataclass, field

from iris.cluster.controller.state import (
    ControllerJob,
    ControllerState,
    ControllerTask,
    ControllerWorker,
    get_device_type,
    get_device_variant,
    get_gpu_count,
)
from iris.cluster.types import WorkerId

logger = logging.getLogger(__name__)


@dataclass
class WorkerCapacity:
    """Available resources on a worker (mutable during scheduling)."""

    available_cpu: int
    available_memory: int  # bytes
    available_gpus: int
    device_type: str  # "cpu", "gpu", "tpu"
    device_variant: str | None


def worker_can_fit_task(capacity: WorkerCapacity, job: ControllerJob) -> bool:
    """Check if worker capacity can fit task. Tasks use the job's resource spec."""
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


def deduct_task_from_capacity(capacity: WorkerCapacity, job: ControllerJob) -> None:
    """Deduct task's resources from capacity (mutates capacity). Uses job's resource spec."""
    res = job.request.resources
    capacity.available_cpu -= res.cpu
    capacity.available_memory -= res.memory_bytes
    capacity.available_gpus -= get_gpu_count(res.device)


def compute_worker_capacity(worker: ControllerWorker) -> WorkerCapacity:
    """Compute current available capacity for a worker."""
    committed_cpu, committed_mem, committed_gpu = worker.get_committed_resources()
    metadata = worker.metadata
    return WorkerCapacity(
        available_cpu=metadata.cpu_count - committed_cpu,
        available_memory=metadata.memory_bytes - committed_mem,
        available_gpus=get_gpu_count(metadata.device) - committed_gpu,
        device_type=get_device_type(metadata.device),
        device_variant=get_device_variant(metadata.device),
    )


@dataclass
class SchedulingTransaction:
    """Accumulates tentative task assignments that can be rolled back."""

    state: ControllerState
    assignments: list[tuple[ControllerTask, ControllerWorker]] = field(default_factory=list)
    timed_out_tasks: list[ControllerTask] = field(default_factory=list)

    def tentatively_assign(self, task: ControllerTask, worker: ControllerWorker) -> None:
        """Assign task to worker immediately (updates running_tasks and removes from queue)."""
        self.state.assign_task_to_worker(worker.worker_id, task.task_id)
        self.assignments.append((task, worker))

    def rollback_assignment(self, task: ControllerTask, worker: ControllerWorker) -> None:
        """Rollback a single failed assignment."""
        self.state.rollback_task_assignment(worker.worker_id, task)


def build_capacity_map(workers: list[ControllerWorker]) -> dict[WorkerId, WorkerCapacity]:
    """Build capacity map for all healthy workers."""
    return {w.worker_id: compute_worker_capacity(w) for w in workers if w.healthy}


class Scheduler:
    """Pure task-to-worker matching logic. Does not dispatch tasks, modify state, or run threads."""

    def __init__(self, state: ControllerState):
        self._state = state

    def find_assignments(
        self,
        pending_tasks: list[ControllerTask],
        workers: list[ControllerWorker],
        now_ms: int,
    ) -> SchedulingTransaction:
        """Match pending tasks to available workers one at a time.

        Uses first-fit algorithm, skipping tasks that don't fit any worker.
        Also identifies tasks that have exceeded their scheduling timeout.

        The algorithm prevents head-of-line blocking: if a large task at the
        front of the queue doesn't fit, smaller tasks behind it can still be
        scheduled.

        Args:
            pending_tasks: Tasks waiting to be scheduled (in FIFO order)
            workers: Available workers (only healthy ones should be passed)
            now_ms: Current timestamp in milliseconds

        Returns:
            SchedulingTransaction with assignments and timed-out tasks
        """
        transaction = SchedulingTransaction(self._state)
        capacities = build_capacity_map(workers)

        for task in pending_tasks:
            job = self._state.get_job(task.job_id)
            if not job:
                continue

            if self._is_task_timed_out(task, job, now_ms):
                transaction.timed_out_tasks.append(task)
                self._state.log_action(
                    "task_timeout",
                    job_id=task.job_id,
                    details=f"task={task.task_id} attempt={task.current_attempt_id}",
                )
                continue

            for worker in workers:
                if not worker.healthy:
                    continue
                capacity = capacities[worker.worker_id]
                if worker_can_fit_task(capacity, job):
                    deduct_task_from_capacity(capacity, job)
                    transaction.tentatively_assign(task, worker)
                    break
            else:
                self._state.log_action(
                    "task_unschedulable",
                    job_id=task.job_id,
                    details=f"task={task.task_id} no_worker_has_capacity",
                )
                logger.debug(
                    "No suitable worker for task %s (cpu=%d, memory_bytes=%d)",
                    task.task_id,
                    job.request.resources.cpu,
                    job.request.resources.memory_bytes,
                )

        if transaction.assignments or transaction.timed_out_tasks:
            logger.debug(
                "Scheduling cycle: %d pending, %d assigned, %d timed_out",
                len(pending_tasks),
                len(transaction.assignments),
                len(transaction.timed_out_tasks),
            )
        return transaction

    def _is_task_timed_out(self, task: ControllerTask, job: ControllerJob, now_ms: int) -> bool:
        """Check if a task has exceeded its scheduling timeout."""
        timeout_seconds = job.request.scheduling_timeout_seconds
        if timeout_seconds <= 0:
            return False

        pending_duration_ms = now_ms - task.submitted_at_ms
        timeout_ms = timeout_seconds * 1000
        return pending_duration_ms > timeout_ms
