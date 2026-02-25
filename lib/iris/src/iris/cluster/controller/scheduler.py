# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure task-to-worker matching without threading, dispatch, or state mutation.

Implements scheduler back-pressure to limit concurrent setup operations per worker.
When many tasks are assigned simultaneously, their uv sync commands can overwhelm
the worker. The max_building_tasks_per_worker setting limits how many tasks can
be in BUILDING state on each worker, preventing resource exhaustion.

The scheduler operates exclusively on scheduler-owned types (JobRequirements,
WorkerCapacity, SchedulingContext) and has ZERO runtime imports from controller
state. The boundary conversion from ControllerWorker to WorkerCapacity happens
via the WorkerSnapshot protocol in create_scheduling_context.
"""

import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from iris.cluster.types import (
    AttributeValue,
    JobName,
    WorkerId,
    get_device_type,
    get_device_variant,
    get_gpu_count,
    get_tpu_count,
)
from iris.rpc import cluster_pb2

logger = logging.getLogger(__name__)

DEFAULT_MAX_BUILDING_TASKS_PER_WORKER = 4
"""Default limit for concurrent BUILDING tasks per worker.

When many tasks start simultaneously, their setup commands (uv sync, pip install)
can overwhelm the worker. This limit provides back-pressure by deferring new
task assignments until existing tasks complete their build phase.
"""


class WorkerSnapshot(Protocol):
    """What the scheduler needs from a worker to build a capacity snapshot.

    This protocol decouples the scheduler from ControllerWorker. Any object
    exposing these fields can be used (ControllerWorker satisfies this).
    """

    worker_id: WorkerId
    available_cpu_millicores: int
    available_memory: int
    available_gpus: int
    available_tpus: int
    device_type: str
    device_variant: str | None
    attributes: dict[str, AttributeValue]
    healthy: bool


class RejectionKind(StrEnum):
    """Types of reasons a job can be rejected from a worker."""

    CPU = "cpu"
    MEMORY = "memory"
    DEVICE_TYPE = "device_type"
    DEVICE_VARIANT = "device_variant"
    GPU_COUNT = "gpu_count"
    TPU_COUNT = "tpu_count"
    BUILDING_LIMIT = "building_limit"


@dataclass
class RejectionReason:
    """Lazy-formatted rejection reason for scheduler diagnostics.

    The message is only formatted when converted to string, avoiding cost
    when the reason is never displayed (e.g., during successful scheduling).
    """

    kind: RejectionKind
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        match self.kind:
            case RejectionKind.CPU:
                need_cores = self.details["need"] / 1000
                have_cores = self.details["have"] / 1000
                return f"Insufficient CPU (need {need_cores:g} cores, available {have_cores:g} cores)"
            case RejectionKind.MEMORY:
                need_gb = self.details["need"] / (1024**3)
                have_gb = self.details["have"] / (1024**3)
                return f"Insufficient memory (need {need_gb:.1f}GB, available {have_gb:.1f}GB)"
            case RejectionKind.DEVICE_TYPE:
                return f"Device type mismatch (need {self.details['need']}, worker has {self.details['have']})"
            case RejectionKind.DEVICE_VARIANT:
                return f"Device variant mismatch (need {self.details['need']}, worker has {self.details['have']})"
            case RejectionKind.GPU_COUNT:
                return f"Insufficient GPUs (need {self.details['need']}, available {self.details['have']})"
            case RejectionKind.TPU_COUNT:
                return f"Insufficient TPUs (need {self.details['need']}, available {self.details['have']})"
            case RejectionKind.BUILDING_LIMIT:
                return (
                    f"Worker at building task limit ({self.details['current']}/{self.details['max']} concurrent builds)"
                )
            case _:
                return f"Unknown rejection: {self.kind}"


@dataclass
class JobRequirements:
    """What a job needs from a worker. Scheduler's input type.

    Protos are readonly -- shared by reference, no copy needed.
    """

    resources: cluster_pb2.ResourceSpecProto
    constraints: list[cluster_pb2.Constraint]
    is_coscheduled: bool
    coscheduling_group_by: str | None


def device_compatible(job_device_type: str, worker_device_type: str) -> bool:
    """Check if a job's device requirement is compatible with a worker's device.

    CPU jobs can run on any worker since every host has a CPU.
    Accelerator jobs (GPU, TPU) require the specific hardware.
    """
    if job_device_type == "cpu":
        return True
    return job_device_type == worker_device_type


def device_variant_matches(job_variant: str, worker_variant: str | None) -> bool:
    """Check if a job's requested device variant matches a worker's reported variant.

    Uses case-insensitive substring matching so that short config names (e.g. "H100")
    match full nvidia-smi names (e.g. "NVIDIA H100 80GB HBM3").
    """
    if not worker_variant:
        return False
    return job_variant.lower() in worker_variant.lower() or worker_variant.lower() in job_variant.lower()


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
        case cluster_pb2.CONSTRAINT_OP_IN:
            target_values = {AttributeValue.from_proto(v).value for v in constraint.values}
            return attr.value in target_values
        case _:
            return False


@dataclass
class WorkerCapacity:
    """Available capacity on a worker for scheduling.

    Initialized from worker's current available resources. The deduct() method
    reduces capacity as tasks are tentatively assigned during a scheduling cycle.

    Tracks building task count for back-pressure: workers with too many tasks
    in BUILDING state won't receive new assignments until builds complete.
    """

    worker_id: WorkerId
    available_cpu_millicores: int
    available_memory: int
    available_gpus: int
    available_tpus: int
    device_type: str
    device_variant: str | None
    attributes: dict[str, AttributeValue] = field(default_factory=dict)
    building_task_count: int = 0
    max_building_tasks: int = DEFAULT_MAX_BUILDING_TASKS_PER_WORKER

    @staticmethod
    def from_worker(
        worker: WorkerSnapshot,
        building_count: int = 0,
        max_building_tasks: int = DEFAULT_MAX_BUILDING_TASKS_PER_WORKER,
    ) -> "WorkerCapacity":
        """Create capacity snapshot from a worker's current state.

        Args:
            worker: The worker to snapshot (any object satisfying WorkerSnapshot)
            building_count: Number of tasks currently in BUILDING state on this worker
            max_building_tasks: Maximum allowed building tasks per worker
        """
        return WorkerCapacity(
            worker_id=worker.worker_id,
            available_cpu_millicores=worker.available_cpu_millicores,
            available_memory=worker.available_memory,
            available_gpus=worker.available_gpus,
            available_tpus=worker.available_tpus,
            device_type=worker.device_type,
            device_variant=worker.device_variant,
            attributes=dict(worker.attributes),
            building_task_count=building_count,
            max_building_tasks=max_building_tasks,
        )

    def can_accept_building_task(self) -> bool:
        """Check if this worker can accept another BUILDING task.

        Back-pressure mechanism: limits concurrent uv sync operations.
        """
        return self.building_task_count < self.max_building_tasks

    def can_fit(self, req: JobRequirements) -> RejectionReason | None:
        """Check if this capacity can fit the job's resource requirements.

        Args:
            req: The job requirements to check

        Returns:
            None if job fits, otherwise RejectionReason with lazy-formatted details
        """
        # Check building task back-pressure first
        if not self.can_accept_building_task():
            return RejectionReason(
                kind=RejectionKind.BUILDING_LIMIT,
                details={"current": self.building_task_count, "max": self.max_building_tasks},
            )

        res = req.resources

        if res.cpu_millicores > self.available_cpu_millicores:
            return RejectionReason(
                kind=RejectionKind.CPU, details={"need": res.cpu_millicores, "have": self.available_cpu_millicores}
            )

        if res.memory_bytes > self.available_memory:
            return RejectionReason(
                kind=RejectionKind.MEMORY, details={"need": res.memory_bytes, "have": self.available_memory}
            )

        job_device_type = get_device_type(res.device)
        if not device_compatible(job_device_type, self.device_type):
            return RejectionReason(
                kind=RejectionKind.DEVICE_TYPE, details={"need": job_device_type, "have": self.device_type}
            )

        job_variant = get_device_variant(res.device)
        if job_variant and job_variant != "auto" and not device_variant_matches(job_variant, self.device_variant):
            return RejectionReason(
                kind=RejectionKind.DEVICE_VARIANT, details={"need": job_variant, "have": self.device_variant}
            )

        if job_device_type == "gpu":
            gpu_count = get_gpu_count(res.device)
            if gpu_count > self.available_gpus:
                return RejectionReason(
                    kind=RejectionKind.GPU_COUNT, details={"need": gpu_count, "have": self.available_gpus}
                )

        if job_device_type == "tpu":
            tpu_count = get_tpu_count(res.device)
            if tpu_count > self.available_tpus:
                return RejectionReason(
                    kind=RejectionKind.TPU_COUNT, details={"need": tpu_count, "have": self.available_tpus}
                )

        return None

    def deduct(self, req: JobRequirements) -> None:
        """Deduct job's resources from available capacity."""
        res = req.resources
        self.available_cpu_millicores -= res.cpu_millicores
        self.available_memory -= res.memory_bytes
        self.available_gpus -= get_gpu_count(res.device)
        self.available_tpus -= get_tpu_count(res.device)
        # Increment building count since new tasks start in BUILDING state
        self.building_task_count += 1

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

    # Device index for fast device-based filtering.
    # Key (device_type, variant) -> exact match; key (device_type, None) -> all workers of that type.
    device_index: dict[tuple[str, str | None], set[WorkerId]] = field(default_factory=dict)

    # Workers that have already been assigned a task this cycle
    scheduled_workers: set[WorkerId] = field(default_factory=set)

    # Task IDs of pending tasks, in scheduling priority order
    pending_tasks: list[JobName] = field(default_factory=list)

    # Job requirements indexed by job ID
    jobs: dict[JobName, JobRequirements] = field(default_factory=dict)

    @classmethod
    def from_workers(
        cls,
        workers: list[WorkerSnapshot],
        building_counts: dict[WorkerId, int] | None = None,
        max_building_tasks: int = DEFAULT_MAX_BUILDING_TASKS_PER_WORKER,
        pending_tasks: list[JobName] | None = None,
        jobs: dict[JobName, JobRequirements] | None = None,
    ) -> "SchedulingContext":
        """Build scheduling context from worker list.

        Creates capacity snapshots for healthy workers and constructs posting
        lists for all worker attributes. String, int, and float values are
        indexed for fast EQ lookups.

        Args:
            workers: List of workers to include in scheduling context
            building_counts: Map of worker_id -> count of tasks in BUILDING state
            max_building_tasks: Maximum building tasks allowed per worker
            pending_tasks: Task IDs in scheduling priority order
            jobs: Job requirements indexed by job ID
        """
        building_counts = building_counts or {}

        # Build capacity map for healthy workers
        capacities = {
            w.worker_id: WorkerCapacity.from_worker(
                w,
                building_count=building_counts.get(w.worker_id, 0),
                max_building_tasks=max_building_tasks,
            )
            for w in workers
            if w.healthy
        }
        discrete_lists: dict[str, dict[str | int | float, set[WorkerId]]] = {}

        device_index: dict[tuple[str, str | None], set[WorkerId]] = {}
        for worker_id, cap in capacities.items():
            for key, attr_value in cap.attributes.items():
                if key not in discrete_lists:
                    discrete_lists[key] = {}
                value = attr_value.value
                if value not in discrete_lists[key]:
                    discrete_lists[key][value] = set()
                discrete_lists[key][value].add(worker_id)

            # Build device index: (type, variant) for exact match, (type, None) for type-wide
            dt = cap.device_type
            dv = cap.device_variant
            device_index.setdefault((dt, dv), set()).add(worker_id)
            device_index.setdefault((dt, None), set()).add(worker_id)

        return cls(
            all_worker_ids=set(capacities.keys()),
            discrete_lists=discrete_lists,
            capacities=capacities,
            device_index=device_index,
            pending_tasks=pending_tasks or [],
            jobs=jobs or {},
        )

    def workers_for_device(self, device_type: str, device_variant: str | None) -> set[WorkerId]:
        """Get workers compatible with the given device requirement.

        CPU jobs can run on any worker. For accelerator jobs, returns workers
        matching the variant when specified (substring match to handle short
        config names vs full nvidia-smi names), or all workers of that device
        type when variant is None or "auto".
        """
        if device_type == "cpu":
            return self.all_worker_ids
        all_of_type = self.device_index.get((device_type, None), set())
        if not device_variant or device_variant == "auto":
            return all_of_type
        # Try exact match first (fast path for local/test workers)
        exact = self.device_index.get((device_type, device_variant))
        if exact:
            return exact
        # Fall back to substring match (nvidia-smi full names vs short config names)
        return {
            wid for wid in all_of_type if device_variant_matches(device_variant, self.capacities[wid].device_variant)
        }

    def matching_workers(self, constraints: Sequence[cluster_pb2.Constraint]) -> set[WorkerId]:
        """Get workers matching ALL constraints.

        Uses posting lists for fast EQ/EXISTS/NOT_EXISTS lookups.
        Falls back to linear scan for NE, GT, GE, LT, LE operators.
        """
        if not constraints:
            return self.all_worker_ids

        result: set[WorkerId] | None = None

        for constraint in constraints:
            matches = self._evaluate_constraint_set(constraint)

            if result is None:
                result = matches
            else:
                result = result & matches

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
            return self.discrete_lists[key].get(target, set())

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
            return self.all_worker_ids

        # Fast path: IN on discrete attribute — union of posting lists for each value
        if op == cluster_pb2.CONSTRAINT_OP_IN and key in self.discrete_lists:
            in_result: set[WorkerId] = set()
            for av in constraint.values:
                target_val = AttributeValue.from_proto(av).value
                in_result |= self.discrete_lists[key].get(target_val, set())
            return in_result

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

    Either contains a successful assignment (worker_id is set) or explains
    why scheduling failed (failure_reason is set).
    """

    task_id: JobName
    worker_id: WorkerId | None = None
    failure_reason: str | None = None

    @property
    def success(self) -> bool:
        return self.worker_id is not None


@dataclass
class SchedulingResult:
    """Result of a scheduling cycle - pure data, no state mutation.

    Only contains successful assignments.
    Failure details are available via get_job_scheduling_diagnostics() for dashboard use.
    """

    assignments: list[tuple[JobName, WorkerId]] = field(default_factory=list)


class Scheduler:
    """Computes optimal task-to-worker assignments based on constraints and capacity.

    Pure functional scheduler that does not dispatch tasks, modify state, or run threads.
    Each call to find_assignments() returns assignments for a single scheduling cycle.

    Implements back-pressure by limiting concurrent BUILDING tasks per worker. This
    prevents resource exhaustion when many tasks start simultaneously and run uv sync.
    """

    def __init__(
        self,
        max_building_tasks_per_worker: int = DEFAULT_MAX_BUILDING_TASKS_PER_WORKER,
    ):
        self._max_building_tasks_per_worker = max_building_tasks_per_worker

    def try_schedule_task(
        self,
        task_id: JobName,
        req: JobRequirements,
        context: SchedulingContext,
        collect_details: bool = False,
    ) -> TaskScheduleResult:
        """Attempt to schedule a single task.

        Returns a TaskScheduleResult indicating success (with assigned worker)
        or failure (with reason).

        Args:
            task_id: The task ID to schedule
            req: Job requirements for the task
            context: Scheduling context with posting lists and capacities
            collect_details: If True, collect detailed rejection reasons (expensive).
                           If False (default), return generic failure with no details (fast).

        Returns:
            TaskScheduleResult with either worker assignment or failure reason
        """
        if not context.capacities:
            return TaskScheduleResult(task_id=task_id, failure_reason="No healthy workers available")

        constraints = list(req.constraints)

        # Pre-filter by device type/variant before constraint matching
        job_device_type = get_device_type(req.resources.device)
        job_device_variant = get_device_variant(req.resources.device)
        device_candidates = context.workers_for_device(job_device_type, job_device_variant)
        if not device_candidates:
            if collect_details:
                # Report available variants so the user can see what the cluster has
                available_variants = sorted(
                    {
                        cap.device_variant or "unknown"
                        for cap in context.capacities.values()
                        if cap.device_type == job_device_type
                    }
                )
                if available_variants:
                    return TaskScheduleResult(
                        task_id=task_id,
                        failure_reason=(
                            f"Device variant mismatch (need {job_device_variant}, "
                            f"cluster has {', '.join(available_variants)})"
                        ),
                    )
                return TaskScheduleResult(
                    task_id=task_id,
                    failure_reason=f"No workers with device type {job_device_type}",
                )
            return TaskScheduleResult(task_id=task_id, failure_reason=None)

        # Use posting lists for fast constraint matching, intersected with device candidates
        matching_worker_ids = context.matching_workers(constraints)
        candidate_ids = matching_worker_ids & device_candidates

        # Cheap mode: try all matching workers, no detailed rejection tracking
        if not collect_details:
            for worker_id in candidate_ids:
                if worker_id in context.scheduled_workers:
                    continue
                capacity = context.capacities[worker_id]
                rejection = capacity.can_fit(req)
                if rejection is None:
                    capacity.deduct(req)
                    context.scheduled_workers.add(worker_id)
                    return TaskScheduleResult(task_id=task_id, worker_id=worker_id)
            # No matching worker had capacity
            return TaskScheduleResult(task_id=task_id, failure_reason=None)

        # Expensive mode: collect all rejection reasons with counts
        rejection_counts: dict[RejectionKind, int] = defaultdict(int)
        rejection_samples: dict[RejectionKind, RejectionReason] = {}
        for worker_id in candidate_ids:
            if worker_id in context.scheduled_workers:
                continue
            capacity = context.capacities[worker_id]
            rejection = capacity.can_fit(req)
            if rejection is None:
                capacity.deduct(req)
                context.scheduled_workers.add(worker_id)
                return TaskScheduleResult(task_id=task_id, worker_id=worker_id)
            rejection_counts[rejection.kind] += 1
            # Keep first sample of each rejection kind for formatting
            if rejection.kind not in rejection_samples:
                rejection_samples[rejection.kind] = rejection

        # No worker could fit the task - build detailed reason
        res = req.resources

        # Report all rejection reasons with their counts
        if rejection_counts:
            # Special handling for building limit
            if RejectionKind.BUILDING_LIMIT in rejection_counts:
                # Check if workers would otherwise have capacity
                workers_with_capacity = sum(
                    1
                    for check_wid in candidate_ids
                    if check_wid not in context.scheduled_workers
                    and context.capacities[check_wid].available_cpu_millicores >= res.cpu_millicores
                    and context.capacities[check_wid].available_memory >= res.memory_bytes
                )
                if workers_with_capacity > 0:
                    count = rejection_counts[RejectionKind.BUILDING_LIMIT]
                    return TaskScheduleResult(
                        task_id=task_id,
                        failure_reason=(
                            f"Waiting for build slots: {count} worker(s) at building limit "
                            f"(max {self._max_building_tasks_per_worker} concurrent builds per worker), "
                            f"but have sufficient resources for this task"
                        ),
                    )

            # Format all rejection reasons with counts
            reason_lines = []
            for kind in sorted(rejection_counts.keys(), key=lambda k: rejection_counts[k], reverse=True):
                count = rejection_counts[kind]
                sample = rejection_samples[kind]
                reason_lines.append(f"{sample} - {count} worker(s)")

            failure_reason = "\n".join(reason_lines)
            if constraints:
                constraint_keys = [c.key for c in constraints]
                failure_reason = f"{failure_reason}\n(with constraints={constraint_keys})"
            return TaskScheduleResult(task_id=task_id, failure_reason=failure_reason)

        if constraints:
            return TaskScheduleResult(
                task_id=task_id,
                failure_reason=(
                    f"No worker matches constraints and has sufficient resources "
                    f"(need cpu={res.cpu_millicores / 1000:g} cores, memory={res.memory_bytes}, "
                    f"constraints={[c.key for c in constraints]})"
                ),
            )
        return TaskScheduleResult(
            task_id=task_id,
            failure_reason=(
                f"No worker has sufficient resources "
                f"(need cpu={res.cpu_millicores / 1000:g} cores, memory={res.memory_bytes})"
            ),
        )

    def find_assignments(
        self,
        context: SchedulingContext,
    ) -> SchedulingResult:
        """Match pending tasks to available workers.

        Pure function - does not mutate any external state. Returns assignments
        for the controller to execute.

        Coscheduled jobs are processed first: all tasks must be assigned atomically
        to workers sharing the same group_by attribute value. If not enough workers
        are available in any group, the job stays pending.

        Non-coscheduled jobs use first-fit algorithm, skipping tasks that don't
        fit any worker. The algorithm prevents head-of-line blocking: if a large
        task at the front of the queue doesn't fit, smaller tasks behind it can
        still be scheduled.

        Implements back-pressure by limiting concurrent BUILDING tasks per worker.
        Workers with too many tasks in BUILDING state won't receive new assignments.

        Args:
            context: Scheduling context with workers, pending tasks, and job requirements

        Returns:
            SchedulingResult with successful assignments
        """
        result = SchedulingResult()
        scheduled_task_ids: set[JobName] = set()

        # Group tasks by job for coscheduled handling
        tasks_by_job: dict[JobName, list[JobName]] = defaultdict(list)
        for task_id in context.pending_tasks:
            job_id = task_id.parent
            if job_id is not None:
                tasks_by_job[job_id].append(task_id)

        # Handle coscheduled jobs first (all-or-nothing assignment)
        for job_id, task_ids in tasks_by_job.items():
            req = context.jobs.get(job_id)
            if req is None or not req.is_coscheduled:
                continue

            coscheduled_result = self._find_coscheduled_assignments(context, task_ids, req)
            if coscheduled_result:
                result.assignments.extend(coscheduled_result)
                for task_id, _ in coscheduled_result:
                    scheduled_task_ids.add(task_id)

        # Handle remaining non-coscheduled tasks (first-fit)
        for task_id in context.pending_tasks:
            if task_id in scheduled_task_ids:
                continue

            # Skip coscheduled jobs entirely - they were handled above
            job_id = task_id.parent
            if job_id is not None:
                req = context.jobs.get(job_id)
                if req is not None and req.is_coscheduled:
                    continue

            req = context.jobs.get(job_id) if job_id is not None else None
            if req is None:
                logger.debug("Task %s has no job requirements, skipping", task_id)
                continue

            task_result = self.try_schedule_task(task_id, req, context)

            if task_result.success and task_result.worker_id:
                result.assignments.append((task_id, task_result.worker_id))
            else:
                logger.debug(
                    "Task %s not scheduled: %s",
                    task_id,
                    task_result.failure_reason,
                )

        if result.assignments:
            logger.debug(
                "Scheduling cycle: %d pending, %d assigned",
                len(context.pending_tasks),
                len(result.assignments),
            )
        return result

    def _find_coscheduled_assignments(
        self,
        context: SchedulingContext,
        task_ids: list[JobName],
        req: JobRequirements,
    ) -> list[tuple[JobName, WorkerId]] | None:
        """Find atomic assignment for a coscheduled task group.

        All tasks must be assigned to workers sharing the same group_by attribute
        value. Tasks are sorted by task_index and assigned to workers sorted by
        tpu-worker-id for deterministic ordering.

        Returns None if no valid worker group exists with sufficient capacity.
        """
        group_by = req.coscheduling_group_by
        if group_by is None:
            return None

        if not task_ids:
            return None

        num_tasks = len(task_ids)
        constraints = list(req.constraints)

        # Pre-filter by device before constraint matching
        job_device_type = get_device_type(req.resources.device)
        job_device_variant = get_device_variant(req.resources.device)
        device_candidates = context.workers_for_device(job_device_type, job_device_variant)
        if not device_candidates:
            return None

        matching_worker_ids = context.matching_workers(constraints) & device_candidates
        groups = context.workers_by_group(group_by, matching_worker_ids)

        # Find first group with enough workers that have capacity.
        # Note: matching_worker_ids passed attribute constraints (e.g., tpu-name=my-tpu),
        # but we still need to check resource capacity (CPU, memory, GPU). These are
        # orthogonal: a worker can match constraints but lack available resources.
        for group_key, group_worker_ids in groups.items():
            available = [
                worker_id for worker_id in group_worker_ids if context.capacities[worker_id].can_fit(req) is None
            ]

            if len(available) < num_tasks:
                continue

            # Sort workers by tpu-worker-id for deterministic task-to-worker mapping
            available.sort(key=lambda w: context.capacities[w].attributes.get("tpu-worker-id", AttributeValue(0)).value)

            # Sort tasks by task_index
            sorted_task_ids = sorted(task_ids, key=lambda t: t.require_task()[1])

            # Assign tasks to workers in order
            assignments: list[tuple[JobName, WorkerId]] = []
            for task_id, worker_id in zip(sorted_task_ids, available[:num_tasks], strict=False):
                context.capacities[worker_id].deduct(req)
                context.scheduled_workers.add(worker_id)
                assignments.append((task_id, worker_id))

            logger.debug(
                "Coscheduled job: assigned %d tasks to group %s",
                len(assignments),
                group_key,
            )
            return assignments

        # No group had enough capacity
        logger.debug(
            "Coscheduled job: no group with %d available workers for group_by=%s",
            num_tasks,
            group_by,
        )
        return None

    def create_scheduling_context(
        self,
        workers: list[WorkerSnapshot],
        building_counts: dict[WorkerId, int] | None = None,
        pending_tasks: list[JobName] | None = None,
        jobs: dict[JobName, JobRequirements] | None = None,
    ) -> SchedulingContext:
        """Create a scheduling context for the given workers.

        This is the boundary conversion point: accepts WorkerSnapshot-compatible
        objects (e.g. ControllerWorker) and converts them to scheduler-internal types.

        Args:
            workers: Workers to include (any objects satisfying WorkerSnapshot)
            building_counts: Map of worker_id -> count of tasks in BUILDING state
            pending_tasks: Task IDs in scheduling priority order
            jobs: Job requirements indexed by job ID
        """
        return SchedulingContext.from_workers(
            workers,
            building_counts=building_counts,
            max_building_tasks=self._max_building_tasks_per_worker,
            pending_tasks=pending_tasks,
            jobs=jobs,
        )

    def get_job_scheduling_diagnostics(
        self,
        req: JobRequirements,
        context: SchedulingContext,
        schedulable_task_id: JobName | None,
        num_tasks: int,
    ) -> str:
        """Get detailed diagnostics for why a job cannot be scheduled.

        This is expensive - it collects rejection reasons from all workers.
        Only call this for displaying to users (e.g., job detail page).

        Args:
            req: The job's requirements
            context: Scheduling context with posting lists and capacities
            schedulable_task_id: A representative schedulable task ID, or None
            num_tasks: Total number of tasks in the job

        Returns:
            Human-readable string explaining why the job cannot be scheduled
        """
        if req.is_coscheduled:
            return self._diagnose_coscheduled_job(req, context, schedulable_task_id, num_tasks)

        if num_tasks == 0:
            return "No tasks found for job"

        if schedulable_task_id is None:
            return "No schedulable tasks (all tasks have non-terminal attempts)"

        # Use expensive mode to collect detailed rejection reasons
        result = self.try_schedule_task(schedulable_task_id, req, context, collect_details=True)
        if result.success:
            return "Schedulable — waiting for next scheduling cycle"
        return result.failure_reason or "Unknown scheduling failure"

    def _diagnose_coscheduled_job(
        self,
        req: JobRequirements,
        context: SchedulingContext,
        schedulable_task_id: JobName | None,
        num_tasks: int,
    ) -> str:
        """Get detailed diagnostics for why a coscheduled job cannot be scheduled."""
        constraints = list(req.constraints)
        matching_ids = context.matching_workers(constraints)
        group_by = req.coscheduling_group_by

        if not matching_ids:
            constraint_keys = [c.key for c in constraints]
            return f"No workers match constraints: {constraint_keys}"

        if not group_by:
            if schedulable_task_id:
                result = self.try_schedule_task(schedulable_task_id, req, context, collect_details=True)
                if result.success:
                    return "Schedulable — waiting for next scheduling cycle"
                return result.failure_reason or "Unknown scheduling failure"
            return "No schedulable tasks"

        groups = context.workers_by_group(group_by, matching_ids)

        if not groups:
            return f"Coscheduling: {len(matching_ids)} workers match constraints but none have '{group_by}' attribute"

        best = max(len(wids) for wids in groups.values())
        if best < num_tasks:
            return f"Coscheduling: need {num_tasks} workers in same '{group_by}' group, largest group has {best}"

        # Workers exist in theory, check capacity within each group
        for group_key, group_worker_ids in groups.items():
            # Count how many workers in this group have capacity
            available = []
            rejection_counts: dict[RejectionKind, int] = defaultdict(int)
            rejection_samples: dict[RejectionKind, RejectionReason] = {}
            for worker_id in group_worker_ids:
                rejection = context.capacities[worker_id].can_fit(req)
                if rejection is None:
                    available.append(worker_id)
                else:
                    rejection_counts[rejection.kind] += 1
                    if rejection.kind not in rejection_samples:
                        rejection_samples[rejection.kind] = rejection

            # If this is the largest group, report why it doesn't have capacity
            if len(group_worker_ids) == best and rejection_counts:
                # Format all rejection reasons with counts
                reason_lines = []
                for kind in sorted(rejection_counts.keys(), key=lambda k: rejection_counts[k], reverse=True):
                    count = rejection_counts[kind]
                    sample = rejection_samples[kind]
                    reason_lines.append(f"{sample} - {count} worker(s)")
                reasons = "\n".join(reason_lines)
                return (
                    f"Coscheduling: need {num_tasks} workers in '{group_by}' group '{group_key}', "
                    f"only {len(available)} of {len(group_worker_ids)} have capacity:\n{reasons}"
                )

        return "Unable to schedule (no clear reason found)"
