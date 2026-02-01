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
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

from iris.cluster.controller.state import (
    ControllerJob,
    ControllerState,
    ControllerTask,
    ControllerWorker,
    get_device_type,
    get_device_variant,
    get_gpu_count,
    get_tpu_chip_count,
)
from iris.cluster.types import AttributeValue, JobId, WorkerId
from iris.rpc import cluster_pb2
from iris.time_utils import now_ms

logger = logging.getLogger(__name__)


def device_compatible(job_device_type: str, worker_device_type: str) -> bool:
    """Check if a job's device requirement is compatible with a worker's device.

    CPU jobs can run on any worker since every host has a CPU.
    Accelerator jobs (GPU, TPU) require the specific hardware.
    """
    if job_device_type == "cpu":
        return True
    return job_device_type == worker_device_type


def _compare_ordered(
    attr_value: str | int | float,
    target_value: str | int | float,
    op: str,
) -> bool:
    """Compare two attribute values with an ordering operator.

    Only numeric types (int, float) support ordered comparisons.
    Strings are not orderable (comparing "v4-8" > "v5" is not meaningful).

    Raises:
        ValueError: If either value is a string (ordered comparison not supported).
    """
    if isinstance(attr_value, str) or isinstance(target_value, str):
        raise ValueError(
            f"Ordered comparison ({op}) not supported for string attributes: "
            f"{attr_value!r} vs {target_value!r}. Use EQ or NE operators instead."
        )

    attr_num: int | float = attr_value
    target_num: int | float = target_value

    if op == "gt":
        return attr_num > target_num
    elif op == "ge":
        return attr_num >= target_num
    elif op == "lt":
        return attr_num < target_num
    elif op == "le":
        return attr_num <= target_num
    return False


def _evaluate_constraint(
    attr: AttributeValue | None,
    constraint: cluster_pb2.Constraint,
) -> bool:
    """Evaluate a single constraint against a worker attribute.

    Args:
        attr: Worker attribute value (None if attribute doesn't exist)
        constraint: Constraint to evaluate

    Returns:
        True if constraint is satisfied, False otherwise
    """
    op = constraint.op

    # EXISTS/NOT_EXISTS don't need a value comparison
    if op == cluster_pb2.CONSTRAINT_OP_EXISTS:
        return attr is not None
    if op == cluster_pb2.CONSTRAINT_OP_NOT_EXISTS:
        return attr is None

    # All other operators require the attribute to exist
    if attr is None:
        return False

    target = AttributeValue.from_proto(constraint.value)

    match op:
        case cluster_pb2.CONSTRAINT_OP_EQ:
            return attr.value == target.value
        case cluster_pb2.CONSTRAINT_OP_NE:
            return attr.value != target.value
        case cluster_pb2.CONSTRAINT_OP_GT:
            return _compare_ordered(attr.value, target.value, "gt")
        case cluster_pb2.CONSTRAINT_OP_GE:
            return _compare_ordered(attr.value, target.value, "ge")
        case cluster_pb2.CONSTRAINT_OP_LT:
            return _compare_ordered(attr.value, target.value, "lt")
        case cluster_pb2.CONSTRAINT_OP_LE:
            return _compare_ordered(attr.value, target.value, "le")
        case _:
            return False


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
    available_tpus: int
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
            available_tpus=worker.available_tpus,
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
        if not device_compatible(job_device_type, self.device_type):
            return False

        job_variant = get_device_variant(res.device)
        if job_variant and job_variant != "auto" and job_variant != self.device_variant:
            return False

        if job_device_type == "gpu" and get_gpu_count(res.device) > self.available_gpus:
            return False

        if job_device_type == "tpu" and get_tpu_chip_count(res.device) > self.available_tpus:
            return False

        return True

    def deduct(self, job: ControllerJob) -> None:
        """Deduct job's resources from available capacity."""
        res = job.request.resources
        self.available_cpu -= res.cpu
        self.available_memory -= res.memory_bytes
        self.available_gpus -= get_gpu_count(res.device)
        self.available_tpus -= get_tpu_chip_count(res.device)

    def matches_constraints(self, constraints: Sequence[cluster_pb2.Constraint]) -> bool:
        """Check if this worker matches all given constraints."""
        for constraint in constraints:
            attr = self.attributes.get(constraint.key)
            if not _evaluate_constraint(attr, constraint):
                return False
        return True


@dataclass
class SchedulingContext:
    """Transient index for a single scheduling cycle.

    Built from worker capacities at cycle start. Provides O(1) constraint
    matching for common cases (EQ on string attributes, EXISTS/NOT_EXISTS)
    via posting lists. Falls back to linear scan for numeric comparisons.

    The posting lists are read-only after construction. As workers are
    tentatively assigned, we track capacity changes in the capacities dict,
    but do not update the posting lists. This is safe because posting lists
    are only used for attribute matching, not capacity checks.

    Workers are tracked in scheduled_workers to ensure each worker receives
    at most one task per cycle, providing round-robin distribution.
    """

    all_worker_ids: set[WorkerId]

    # Posting lists for fast constraint matching
    # Maps: attribute_key -> attribute_value -> set of worker IDs
    discrete_lists: dict[str, dict[str | int | float, set[WorkerId]]]

    # Worker capacities indexed by worker ID
    capacities: dict[WorkerId, WorkerCapacity]

    # Workers already assigned a task this cycle (for round-robin)
    scheduled_workers: set[WorkerId] = field(default_factory=set)

    @classmethod
    def from_workers(cls, workers: list[ControllerWorker]) -> "SchedulingContext":
        """Build scheduling context from worker list.

        Creates capacity snapshots for healthy workers and constructs posting
        lists for all worker attributes. String, int, and float values are
        indexed for fast EQ lookups.
        """
        # Build capacity map for healthy workers
        capacities = {w.worker_id: WorkerCapacity.from_worker(w) for w in workers if w.healthy}
        discrete_lists: dict[str, dict[str | int | float, set[WorkerId]]] = {}

        for worker_id, cap in capacities.items():
            for key, attr_value in cap.attributes.items():
                if key not in discrete_lists:
                    discrete_lists[key] = {}
                value = attr_value.value
                if value not in discrete_lists[key]:
                    discrete_lists[key][value] = set()
                discrete_lists[key][value].add(worker_id)

        return cls(
            all_worker_ids=set(capacities.keys()),
            discrete_lists=discrete_lists,
            capacities=capacities,
        )

    def matching_workers(self, constraints: Sequence[cluster_pb2.Constraint]) -> set[WorkerId]:
        """Get workers matching ALL constraints.

        Uses posting lists for fast EQ/EXISTS/NOT_EXISTS lookups.
        Falls back to linear scan for NE, GT, GE, LT, LE operators.
        """
        if not constraints:
            return self.all_worker_ids.copy()

        result: set[WorkerId] | None = None

        for constraint in constraints:
            matches = self._evaluate_constraint_set(constraint)

            if result is None:
                result = matches
            else:
                result &= matches

            # Short-circuit if no workers match
            if not result:
                return set()

        return result or set()

    def _evaluate_constraint_set(self, constraint: cluster_pb2.Constraint) -> set[WorkerId]:
        """Evaluate a single constraint, returning matching worker IDs."""
        key = constraint.key
        op = constraint.op

        # Fast path: EQ on discrete attribute with posting list
        if op == cluster_pb2.CONSTRAINT_OP_EQ and key in self.discrete_lists:
            target = AttributeValue.from_proto(constraint.value).value
            return self.discrete_lists[key].get(target, set()).copy()

        # Fast path: EXISTS check - union all workers that have this attribute
        if op == cluster_pb2.CONSTRAINT_OP_EXISTS:
            if key in self.discrete_lists:
                result: set[WorkerId] = set()
                for workers in self.discrete_lists[key].values():
                    result.update(workers)
                return result
            # Attribute doesn't exist for any worker
            return set()

        # Fast path: NOT_EXISTS - all workers minus those with the attribute
        if op == cluster_pb2.CONSTRAINT_OP_NOT_EXISTS:
            if key in self.discrete_lists:
                has_attr: set[WorkerId] = set()
                for workers in self.discrete_lists[key].values():
                    has_attr.update(workers)
                return self.all_worker_ids - has_attr
            # Attribute doesn't exist for any worker, so all workers match
            return self.all_worker_ids.copy()

        # Slow path: linear scan for NE, GT, GE, LT, LE, or non-indexed attributes
        result_set: set[WorkerId] = set()
        for worker_id, cap in self.capacities.items():
            attr = cap.attributes.get(key)
            if _evaluate_constraint(attr, constraint):
                result_set.add(worker_id)
        return result_set

    def workers_by_group(
        self,
        group_by: str,
        matching_worker_ids: set[WorkerId],
    ) -> dict[str, list[WorkerId]]:
        """Group workers by the specified attribute value.

        Args:
            group_by: Attribute key to group by
            matching_worker_ids: Set of worker IDs to consider

        Returns:
            Dict mapping group key (str representation) to list of worker IDs
        """
        groups: dict[str, list[WorkerId]] = defaultdict(list)

        if group_by not in self.discrete_lists:
            return groups

        # Use posting list to efficiently find workers in each group
        for value, workers in self.discrete_lists[group_by].items():
            for worker_id in workers:
                if worker_id in matching_worker_ids:
                    groups[str(value)].append(worker_id)

        return groups


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
    Failure details are available via task_schedule_status() for dashboard use.
    """

    assignments: list[tuple[ControllerTask, ControllerWorker]] = field(default_factory=list)
    timed_out_tasks: list[ControllerTask] = field(default_factory=list)


class Scheduler:
    """Computes optimal task-to-worker assignments based on constraints and capacity.

    Pure functional scheduler that does not dispatch tasks, modify state, or run threads.
    Each call to find_assignments() returns assignments for a single scheduling cycle.
    """

    def __init__(self, state: ControllerState):
        self._state = state

    def try_schedule_task(
        self,
        task: ControllerTask,
        context: SchedulingContext,
    ) -> TaskScheduleResult:
        """Attempt to schedule a single task.

        Returns a TaskScheduleResult indicating success (with assigned worker)
        or failure (with reason).

        Args:
            task: The task to schedule
            context: Scheduling context with posting lists and capacities

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

        if not context.capacities:
            return TaskScheduleResult(task=task, failure_reason="No healthy workers available")

        constraints = list(job.request.constraints)

        # Use posting lists for fast constraint matching
        matching_worker_ids = context.matching_workers(constraints)

        # Try to find a worker that can fit this task among matching workers
        for worker_id in matching_worker_ids:
            if worker_id in context.scheduled_workers:
                continue
            capacity = context.capacities[worker_id]
            if capacity.can_fit_job(job):
                capacity.deduct(job)
                context.scheduled_workers.add(worker_id)
                return TaskScheduleResult(task=task, worker=capacity.worker)

        # No worker could fit the task - build detailed reason
        res = job.request.resources
        if constraints:
            return TaskScheduleResult(
                task=task,
                failure_reason=(
                    f"No worker matches constraints and has sufficient resources "
                    f"(need cpu={res.cpu}, memory={res.memory_bytes}, "
                    f"constraints={[c.key for c in constraints]})"
                ),
            )
        return TaskScheduleResult(
            task=task,
            failure_reason=(f"No worker has sufficient resources (need cpu={res.cpu}, memory={res.memory_bytes})"),
        )

    def find_assignments(
        self,
        pending_tasks: list[ControllerTask],
        workers: list[ControllerWorker],
    ) -> SchedulingResult:
        """Match pending tasks to available workers.

        Pure function - does not mutate ControllerState. Returns assignments
        for the controller to execute.

        Coscheduled jobs are processed first: all tasks must be assigned atomically
        to workers sharing the same group_by attribute value. If not enough workers
        are available in any group, the job stays pending.

        Non-coscheduled jobs use first-fit algorithm, skipping tasks that don't
        fit any worker. The algorithm prevents head-of-line blocking: if a large
        task at the front of the queue doesn't fit, smaller tasks behind it can
        still be scheduled.

        Args:
            pending_tasks: Tasks waiting to be scheduled (in FIFO order)
            workers: Available workers (only healthy ones should be passed)

        Returns:
            SchedulingResult with successful assignments and timed-out tasks only
        """
        result = SchedulingResult()
        context = SchedulingContext.from_workers(workers)
        scheduled_task_ids: set[str] = set()

        # Group tasks by job for coscheduled handling
        tasks_by_job: dict[JobId, list[ControllerTask]] = defaultdict(list)
        for task in pending_tasks:
            tasks_by_job[task.job_id].append(task)

        # Handle coscheduled jobs first (all-or-nothing assignment)
        for job_id, job_tasks in tasks_by_job.items():
            job = self._state.get_job(job_id)
            if job is None or not job.is_coscheduled:
                continue

            coscheduled_result = self._find_coscheduled_assignments(context, job_tasks, job)
            if coscheduled_result:
                result.assignments.extend(coscheduled_result)
                for task, _ in coscheduled_result:
                    scheduled_task_ids.add(task.task_id)

        # Handle remaining non-coscheduled tasks (first-fit)
        for task in pending_tasks:
            if task.task_id in scheduled_task_ids:
                continue

            # Skip coscheduled jobs entirely - they were handled above
            job = self._state.get_job(task.job_id)
            if job is not None and job.is_coscheduled:
                continue

            task_result = self.try_schedule_task(task, context)

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

    def _find_coscheduled_assignments(
        self,
        context: SchedulingContext,
        tasks: list[ControllerTask],
        job: ControllerJob,
    ) -> list[tuple[ControllerTask, ControllerWorker]] | None:
        """Find atomic assignment for a coscheduled task group.

        All tasks must be assigned to workers sharing the same group_by attribute
        value. Tasks are sorted by task_index and assigned to workers sorted by
        tpu-worker-id for deterministic ordering.

        Returns None if no valid worker group exists with sufficient capacity.
        """
        group_by = job.coscheduling_group_by
        if group_by is None:
            return None

        # Filter to only schedulable tasks
        schedulable_tasks = [t for t in tasks if t.can_be_scheduled()]
        if not schedulable_tasks:
            return None

        num_tasks = len(schedulable_tasks)
        constraints = list(job.request.constraints)

        matching_worker_ids = context.matching_workers(constraints)
        groups = context.workers_by_group(group_by, matching_worker_ids)

        # Find first group with enough workers that have capacity.
        # Note: matching_worker_ids passed attribute constraints (e.g., tpu-name=my-tpu),
        # but we still need to check resource capacity (CPU, memory, GPU). These are
        # orthogonal: a worker can match constraints but lack available resources.
        for group_key, group_worker_ids in groups.items():
            available = [worker_id for worker_id in group_worker_ids if context.capacities[worker_id].can_fit_job(job)]

            if len(available) < num_tasks:
                continue

            # Sort workers by tpu-worker-id for deterministic task-to-worker mapping
            available.sort(key=lambda w: context.capacities[w].attributes.get("tpu-worker-id", AttributeValue(0)).value)

            # Sort tasks by task_index
            sorted_tasks = sorted(schedulable_tasks, key=lambda t: t.task_index)

            # Assign tasks to workers in order
            assignments: list[tuple[ControllerTask, ControllerWorker]] = []
            for task, worker_id in zip(sorted_tasks, available[:num_tasks], strict=False):
                context.capacities[worker_id].deduct(job)
                assignments.append((task, context.capacities[worker_id].worker))

            logger.debug(
                "Coscheduled job %s: assigned %d tasks to group %s",
                job.job_id,
                len(assignments),
                group_key,
            )
            return assignments

        # No group had enough capacity
        logger.debug(
            "Coscheduled job %s: no group with %d available workers for group_by=%s",
            job.job_id,
            num_tasks,
            group_by,
        )
        return None

    def _is_task_timed_out(self, task: ControllerTask, job: ControllerJob) -> bool:
        """Check if a task has exceeded its scheduling timeout."""
        timeout_seconds = job.request.scheduling_timeout_seconds
        if timeout_seconds <= 0:
            return False

        pending_duration_ms = now_ms() - task.submitted_at_ms
        timeout_ms = timeout_seconds * 1000
        return pending_duration_ms > timeout_ms

    def task_schedule_status(self, task: ControllerTask, context: SchedulingContext) -> TaskScheduleResult:
        """Get the current scheduling status of a task.

        Used by the dashboard to show why a task is pending.
        Caller provides the SchedulingContext so it can be reused across
        multiple calls (e.g. when listing all tasks for a job).
        """
        job = self._state.get_job(task.job_id)
        if job and job.is_coscheduled:
            return self._coscheduled_schedule_status(task, job, context)

        return self.try_schedule_task(task, context)

    def _coscheduled_schedule_status(
        self,
        task: ControllerTask,
        job: ControllerJob,
        context: SchedulingContext,
    ) -> TaskScheduleResult:
        """Detailed scheduling status for a coscheduled task."""
        constraints = list(job.request.constraints)
        matching_ids = context.matching_workers(constraints)
        group_by = job.coscheduling_group_by
        num_tasks = len(self._state.get_job_tasks(task.job_id))

        if not matching_ids:
            constraint_keys = [c.key for c in constraints]
            return TaskScheduleResult(
                task=task,
                failure_reason=f"No workers match constraints: {constraint_keys}",
            )

        if not group_by:
            return self.try_schedule_task(task, context)

        groups = context.workers_by_group(group_by, matching_ids)

        if not groups:
            return TaskScheduleResult(
                task=task,
                failure_reason=(
                    f"Coscheduling: {len(matching_ids)} workers match constraints "
                    f"but none have '{group_by}' attribute"
                ),
            )

        best = max(len(wids) for wids in groups.values())
        if best < num_tasks:
            return TaskScheduleResult(
                task=task,
                failure_reason=(
                    f"Coscheduling: need {num_tasks} workers in same '{group_by}' group, " f"largest group has {best}"
                ),
            )

        # Workers exist in theory, check capacity
        return self.try_schedule_task(task, context)
