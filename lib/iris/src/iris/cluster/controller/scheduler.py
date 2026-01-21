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
from iris.cluster.types import AttributeValue, WorkerId
from iris.time_utils import now_ms

logger = logging.getLogger(__name__)


@dataclass
class WorkerCapacity:
    """Available capacity on a worker for scheduling.

    Initialized from worker's current available resources. The deduct() method
    reduces capacity as tasks are tentatively assigned during a scheduling cycle.
    """

    worker: ControllerWorker
    available_cpu: int
    available_memory: int
    available_gpus: int
    device_type: str
    device_variant: str | None
    attributes: dict[str, AttributeValue] = field(default_factory=dict)

    @staticmethod
    def from_worker(worker: ControllerWorker) -> "WorkerCapacity":
        """Create capacity snapshot from a worker's current state."""
        return WorkerCapacity(
            worker=worker,
            available_cpu=worker.available_cpu,
            available_memory=worker.available_memory,
            available_gpus=worker.available_gpus,
            device_type=worker.device_type,
            device_variant=worker.device_variant,
            attributes=dict(worker.attributes),
        )

    def can_fit_job(self, job: ControllerJob) -> bool:
        """Check if this capacity can fit the job's resource requirements."""
        res = job.request.resources

        if res.cpu > self.available_cpu:
            return False

        if res.memory_bytes > self.available_memory:
            return False

        job_device_type = get_device_type(res.device)
        if job_device_type != self.device_type:
            return False

        job_variant = get_device_variant(res.device)
        if job_variant and job_variant != "auto" and job_variant != self.device_variant:
            return False

        if job_device_type == "gpu" and get_gpu_count(res.device) > self.available_gpus:
            return False

        return True

    def deduct(self, job: ControllerJob) -> None:
        """Deduct job's resources from available capacity."""
        res = job.request.resources
        self.available_cpu -= res.cpu
        self.available_memory -= res.memory_bytes
        self.available_gpus -= get_gpu_count(res.device)


@dataclass
class TaskScheduleResult:
    """Result of attempting to schedule a single task.

    Either contains a successful assignment (worker is set) or explains
    why scheduling failed (failure_reason is set).
    """

    task: ControllerTask
    worker: ControllerWorker | None = None
    failure_reason: str | None = None
    timed_out: bool = False

    @property
    def success(self) -> bool:
        return self.worker is not None


@dataclass
class SchedulingResult:
    """Result of a scheduling cycle - pure data, no state mutation.

    Only contains successful assignments and timed-out tasks.
    Failure details are available via get_task_schedule_status() for dashboard use.
    """

    assignments: list[tuple[ControllerTask, ControllerWorker]] = field(default_factory=list)
    timed_out_tasks: list[ControllerTask] = field(default_factory=list)


def build_capacity_map(workers: list[ControllerWorker]) -> dict[WorkerId, WorkerCapacity]:
    """Build capacity map for all healthy workers."""
    return {w.worker_id: WorkerCapacity.from_worker(w) for w in workers if w.healthy}


class Scheduler:
    """Pure task-to-worker matching logic. Does not dispatch tasks, modify state, or run threads."""

    def __init__(self, state: ControllerState):
        self._state = state

    def try_schedule_task(
        self,
        task: ControllerTask,
        capacities: dict[WorkerId, WorkerCapacity],
    ) -> TaskScheduleResult:
        """Attempt to schedule a single task.

        Returns a TaskScheduleResult indicating success (with assigned worker)
        or failure (with reason).

        Args:
            task: The task to schedule
            capacities: Current worker capacities (may have tentative deductions)

        Returns:
            TaskScheduleResult with either worker assignment or failure reason
        """
        if not task.can_be_scheduled():
            return TaskScheduleResult(
                task=task,
                failure_reason="Task has non-terminal attempt (waiting for worker to report state)",
            )

        job = self._state.get_job(task.job_id)
        if not job:
            return TaskScheduleResult(task=task, failure_reason="Job not found")

        if self._is_task_timed_out(task, job):
            return TaskScheduleResult(task=task, timed_out=True)

        if not capacities:
            return TaskScheduleResult(task=task, failure_reason="No healthy workers available")

        # Try to find a worker that can fit this task
        for capacity in capacities.values():
            if capacity.can_fit_job(job):
                capacity.deduct(job)
                return TaskScheduleResult(task=task, worker=capacity.worker)

        # No worker could fit the task - build detailed reason
        res = job.request.resources
        return TaskScheduleResult(
            task=task,
            failure_reason=(f"No worker has sufficient resources " f"(need cpu={res.cpu}, memory={res.memory_bytes})"),
        )

    def find_assignments(
        self,
        pending_tasks: list[ControllerTask],
        workers: list[ControllerWorker],
    ) -> SchedulingResult:
        """Match pending tasks to available workers.

        Pure function - does not mutate ControllerState. Returns assignments
        for the controller to execute.

        Uses first-fit algorithm, skipping tasks that don't fit any worker.
        Also identifies tasks that have exceeded their scheduling timeout.

        The algorithm prevents head-of-line blocking: if a large task at the
        front of the queue doesn't fit, smaller tasks behind it can still be
        scheduled.

        Args:
            pending_tasks: Tasks waiting to be scheduled (in FIFO order)
            workers: Available workers (only healthy ones should be passed)

        Returns:
            SchedulingResult with successful assignments and timed-out tasks only
        """
        result = SchedulingResult()
        capacities = build_capacity_map(workers)

        for task in pending_tasks:
            task_result = self.try_schedule_task(task, capacities)

            if task_result.success and task_result.worker:
                result.assignments.append((task, task_result.worker))
            elif task_result.timed_out:
                result.timed_out_tasks.append(task)
            else:
                logger.debug(
                    "Task %s not scheduled: %s",
                    task.task_id,
                    task_result.failure_reason,
                )

        if result.assignments or result.timed_out_tasks:
            logger.debug(
                "Scheduling cycle: %d pending, %d assigned, %d timed_out",
                len(pending_tasks),
                len(result.assignments),
                len(result.timed_out_tasks),
            )
        return result

    def _is_task_timed_out(self, task: ControllerTask, job: ControllerJob) -> bool:
        """Check if a task has exceeded its scheduling timeout."""
        timeout_seconds = job.request.scheduling_timeout_seconds
        if timeout_seconds <= 0:
            return False

        pending_duration_ms = now_ms() - task.submitted_at_ms
        timeout_ms = timeout_seconds * 1000
        return pending_duration_ms > timeout_ms

    def get_task_schedule_status(self, task: ControllerTask) -> TaskScheduleResult:
        """Get the current scheduling status of a task.

        Used by the dashboard to show why a task is pending.
        Builds fresh capacity map to reflect current state.
        """
        workers = self._state.get_available_workers()
        capacities = build_capacity_map(workers)
        return self.try_schedule_task(task, capacities)
