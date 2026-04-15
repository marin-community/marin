# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure task-to-worker matching without threading, dispatch, or state mutation.

Implements scheduler back-pressure to limit concurrent setup operations per worker.
When many tasks are assigned simultaneously, their uv sync commands can overwhelm
the worker. The max_building_tasks_per_worker setting limits how many tasks can
be in BUILDING state on each worker, preventing resource exhaustion.

The scheduler operates exclusively on scheduler-owned types (JobRequirements,
WorkerCapacity, SchedulingContext) and has ZERO runtime imports from controller
state. The boundary conversion from worker rows to WorkerCapacity happens
via the WorkerSnapshot protocol in create_scheduling_context.
"""

import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from iris.cluster.constraints import (
    AttributeValue,
    Constraint,
    ConstraintIndex,
    ResourceCapacity,
    WellKnownAttribute,
    check_resource_fit,
    evaluate_constraint,
    soft_constraint_score,
    split_hard_soft,
)
from iris.cluster.types import (
    JobName,
    WorkerId,
    get_gpu_count,
    get_tpu_count,
)
from iris.rpc import job_pb2

logger = logging.getLogger(__name__)

DEFAULT_MAX_BUILDING_TASKS_PER_WORKER = 8
"""Default limit for concurrent BUILDING tasks per worker.

When many tasks start simultaneously, their setup commands (uv sync, pip install)
can overwhelm the worker. This limit provides back-pressure by deferring new
task assignments until existing tasks complete their build phase.
"""

DEFAULT_MAX_ASSIGNMENTS_PER_WORKER = 1
"""Default limit for task assignments per worker per scheduling cycle.

Set to 1 for normal scheduling (round-robin distribution). The dry-run in
compute_demand_entries sets this to sys.maxsize so that a big worker with
spare CPU can absorb multiple tasks, preventing false demand signals.
"""


class WorkerSnapshot(Protocol):
    """What the scheduler needs from a worker to build a capacity snapshot.

    This protocol decouples the scheduler from a concrete worker row type. Any object
    exposing these fields can be used.  Fields mirror the DB column names so that
    projection row classes satisfy this protocol without computed properties.
    """

    worker_id: WorkerId
    total_cpu_millicores: int
    committed_cpu_millicores: int
    total_memory_bytes: int
    committed_mem: int
    total_gpu_count: int
    committed_gpu: int
    total_tpu_count: int
    committed_tpu: int
    attributes: dict[str, AttributeValue]
    healthy: bool


class RejectionKind(StrEnum):
    """Types of reasons a job can be rejected from a worker."""

    CPU = "cpu"
    MEMORY = "memory"
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
    """What a job needs from a worker. Scheduler's input type."""

    resources: job_pb2.ResourceSpecProto
    constraints: list[Constraint]
    is_coscheduled: bool
    coscheduling_group_by: str | None


_evaluate_constraint = evaluate_constraint


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
            available_cpu_millicores=worker.total_cpu_millicores - worker.committed_cpu_millicores,
            available_memory=worker.total_memory_bytes - worker.committed_mem,
            available_gpus=worker.total_gpu_count - worker.committed_gpu,
            available_tpus=worker.total_tpu_count - worker.committed_tpu,
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

        Only checks resource capacity (CPU, memory, device count, building limit).
        Device type and variant matching is handled by matches_constraints() via
        the posting-list index in SchedulingContext.

        Returns:
            None if job fits, otherwise RejectionReason with lazy-formatted details
        """
        if not self.can_accept_building_task():
            return RejectionReason(
                kind=RejectionKind.BUILDING_LIMIT,
                details={"current": self.building_task_count, "max": self.max_building_tasks},
            )

        res = req.resources
        gpu_count = get_gpu_count(res.device)
        tpu_count = get_tpu_count(res.device)

        available = ResourceCapacity(
            cpu_millicores=self.available_cpu_millicores,
            memory_bytes=self.available_memory,
            gpu_count=self.available_gpus,
            tpu_count=self.available_tpus,
        )
        required = ResourceCapacity(
            cpu_millicores=res.cpu_millicores,
            memory_bytes=res.memory_bytes,
            gpu_count=gpu_count,
            tpu_count=tpu_count,
        )

        reason = check_resource_fit(available, required)
        if reason is None:
            return None

        return self._reason_to_rejection(reason, res, gpu_count, tpu_count)

    def _reason_to_rejection(
        self, reason: str, res: job_pb2.ResourceSpecProto, gpu_count: int, tpu_count: int
    ) -> RejectionReason:
        """Map a check_resource_fit reason string to a RejectionReason."""
        if reason.startswith("cpu:"):
            return RejectionReason(
                kind=RejectionKind.CPU, details={"need": res.cpu_millicores, "have": self.available_cpu_millicores}
            )
        if reason.startswith("memory:"):
            return RejectionReason(
                kind=RejectionKind.MEMORY, details={"need": res.memory_bytes, "have": self.available_memory}
            )
        if reason.startswith("gpu:"):
            return RejectionReason(
                kind=RejectionKind.GPU_COUNT, details={"need": gpu_count, "have": self.available_gpus}
            )
        if reason.startswith("tpu:"):
            return RejectionReason(
                kind=RejectionKind.TPU_COUNT, details={"need": tpu_count, "have": self.available_tpus}
            )
        return RejectionReason(kind=RejectionKind.CPU, details={"need": 0, "have": 0})

    def deduct(self, req: JobRequirements) -> None:
        """Deduct job's resources from available capacity."""
        res = req.resources
        self.available_cpu_millicores -= res.cpu_millicores
        self.available_memory -= res.memory_bytes
        self.available_gpus -= get_gpu_count(res.device)
        self.available_tpus -= get_tpu_count(res.device)
        # Increment building count since new tasks start in BUILDING state
        self.building_task_count += 1

    def matches_constraints(self, constraints: Sequence[Constraint]) -> bool:
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

    Workers are tracked via assignment_counts to limit how many tasks each
    worker receives per cycle (default 1 for round-robin distribution).
    """

    index: ConstraintIndex

    # Worker capacities indexed by worker ID
    capacities: dict[WorkerId, WorkerCapacity]

    # Reverse map from string ID back to WorkerId
    _str_to_wid: dict[str, WorkerId]

    # Per-worker assignment count this cycle (replaces scheduled_workers set)
    assignment_counts: dict[WorkerId, int] = field(default_factory=dict)

    # Maximum assignments per worker per cycle
    max_assignments_per_worker: int = DEFAULT_MAX_ASSIGNMENTS_PER_WORKER

    # Task IDs of pending tasks, in scheduling priority order
    pending_tasks: list[JobName] = field(default_factory=list)

    # Job requirements indexed by job ID
    jobs: dict[JobName, JobRequirements] = field(default_factory=dict)

    @property
    def all_worker_ids(self) -> set[WorkerId]:
        return {self._str_to_wid[s] for s in self.index._all_ids}

    @classmethod
    def from_workers(
        cls,
        workers: list[WorkerSnapshot],
        building_counts: dict[WorkerId, int] | None = None,
        max_building_tasks: int = DEFAULT_MAX_BUILDING_TASKS_PER_WORKER,
        pending_tasks: list[JobName] | None = None,
        jobs: dict[JobName, JobRequirements] | None = None,
        max_assignments_per_worker: int = DEFAULT_MAX_ASSIGNMENTS_PER_WORKER,
    ) -> "SchedulingContext":
        """Build scheduling context from worker list.

        Creates capacity snapshots for healthy workers and builds a
        ConstraintIndex for fast attribute matching.

        Args:
            workers: List of workers to include in scheduling context
            building_counts: Map of worker_id -> count of tasks in BUILDING state
            max_building_tasks: Maximum building tasks allowed per worker
            pending_tasks: Task IDs in scheduling priority order
            jobs: Job requirements indexed by job ID
            max_assignments_per_worker: Maximum task assignments per worker per cycle
        """
        building_counts = building_counts or {}

        capacities = {
            w.worker_id: WorkerCapacity.from_worker(
                w,
                building_count=building_counts.get(w.worker_id, 0),
                max_building_tasks=max_building_tasks,
            )
            for w in workers
            if w.healthy
        }

        str_to_wid: dict[str, WorkerId] = {}
        entity_attrs: dict[str, dict[str, AttributeValue]] = {}
        for wid, cap in capacities.items():
            key = str(wid)
            str_to_wid[key] = wid
            entity_attrs[key] = dict(cap.attributes)

        index = ConstraintIndex.build(entity_attrs)

        return cls(
            index=index,
            capacities=capacities,
            _str_to_wid=str_to_wid,
            pending_tasks=pending_tasks or [],
            jobs=jobs or {},
            max_assignments_per_worker=max_assignments_per_worker,
        )

    def matching_workers(self, constraints: Sequence[Constraint]) -> set[WorkerId]:
        """Get workers matching ALL constraints.

        Uses posting lists for fast EQ/EXISTS/NOT_EXISTS lookups.
        Falls back to linear scan for NE, GT, GE, LT, LE operators.
        """
        matched_strs = self.index.matching_entities(constraints)
        return {self._str_to_wid[s] for s in matched_strs}

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
        matching_strs = {str(wid) for wid in matching_worker_ids}
        str_groups = self.index.entities_by_group(group_by, matching_strs)
        return {key: [self._str_to_wid[s] for s in ids] for key, ids in str_groups.items()}


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


def _rank_by_soft_score(
    candidate_ids: set[WorkerId],
    soft_constraints: list[Constraint],
    context: SchedulingContext,
) -> list[WorkerId]:
    """Sort candidate workers by soft-constraint satisfaction (descending).

    Workers satisfying more soft constraints are tried first. Workers with the
    same score retain arbitrary (set) order.
    """
    scored: list[tuple[int, WorkerId]] = []
    for wid in candidate_ids:
        cap = context.capacities.get(wid)
        if cap is None:
            continue
        score = soft_constraint_score(dict(cap.attributes), soft_constraints)
        scored.append((score, wid))
    # Sort descending by score so soft-preferred workers are tried first
    scored.sort(key=lambda t: t[0], reverse=True)
    return [wid for _, wid in scored]


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

        all_constraints = list(req.constraints)
        hard_constraints, soft_constraints = split_hard_soft(all_constraints)

        # Use posting lists for fast constraint matching on hard constraints only.
        # Soft constraints do not filter — they only influence candidate
        # ranking so that matching workers are tried first.
        candidate_ids = context.matching_workers(hard_constraints)

        # When soft constraints are present, sort candidates so workers
        # satisfying more soft constraints are tried first.
        if soft_constraints:
            candidate_ids = _rank_by_soft_score(candidate_ids, soft_constraints, context)

        # Cheap mode: try all matching workers, no detailed rejection tracking
        if not collect_details:
            for worker_id in candidate_ids:
                if context.assignment_counts.get(worker_id, 0) >= context.max_assignments_per_worker:
                    continue
                capacity = context.capacities[worker_id]
                rejection = capacity.can_fit(req)
                if rejection is None:
                    capacity.deduct(req)
                    context.assignment_counts[worker_id] = context.assignment_counts.get(worker_id, 0) + 1
                    return TaskScheduleResult(task_id=task_id, worker_id=worker_id)
            # No matching worker had capacity
            return TaskScheduleResult(task_id=task_id, failure_reason=None)

        # Expensive mode: collect all rejection reasons with counts
        rejection_counts: dict[RejectionKind, int] = defaultdict(int)
        rejection_samples: dict[RejectionKind, RejectionReason] = {}
        for worker_id in candidate_ids:
            if context.assignment_counts.get(worker_id, 0) >= context.max_assignments_per_worker:
                continue
            capacity = context.capacities[worker_id]
            rejection = capacity.can_fit(req)
            if rejection is None:
                capacity.deduct(req)
                context.assignment_counts[worker_id] = context.assignment_counts.get(worker_id, 0) + 1
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
                    if context.assignment_counts.get(check_wid, 0) < context.max_assignments_per_worker
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
            if hard_constraints:
                constraint_keys = [c.key for c in hard_constraints]
                failure_reason = f"{failure_reason}\n(with constraints={constraint_keys})"
            return TaskScheduleResult(task_id=task_id, failure_reason=failure_reason)

        if hard_constraints:
            return TaskScheduleResult(
                task_id=task_id,
                failure_reason=(
                    f"No worker matches constraints and has sufficient resources "
                    f"(need cpu={res.cpu_millicores / 1000:g} cores, memory={res.memory_bytes}, "
                    f"constraints={[c.key for c in hard_constraints]})"
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
        all_constraints = list(req.constraints)
        hard_constraints, soft_constraints = split_hard_soft(all_constraints)

        # Only hard constraints filter candidates; soft constraints rank groups.
        matching_worker_ids = context.matching_workers(hard_constraints)
        groups = context.workers_by_group(group_by, matching_worker_ids)

        # Sort groups so those satisfying more soft constraints are tried first.
        def _group_soft_score(group_worker_ids: list[WorkerId]) -> int:
            if not soft_constraints:
                return 0
            total = 0
            for wid in group_worker_ids:
                cap = context.capacities.get(wid)
                if cap is not None:
                    total += soft_constraint_score(dict(cap.attributes), soft_constraints)
            return total

        sorted_groups = sorted(groups.items(), key=lambda kv: _group_soft_score(kv[1]), reverse=True)

        # Find first group with enough workers that have capacity.
        # Note: matching_worker_ids passed attribute constraints (e.g., tpu-name=my-tpu),
        # but we still need to check resource capacity (CPU, memory, GPU). These are
        # orthogonal: a worker can match constraints but lack available resources.
        for group_key, group_worker_ids in sorted_groups:
            available = [
                worker_id for worker_id in group_worker_ids if context.capacities[worker_id].can_fit(req) is None
            ]

            if len(available) < num_tasks:
                continue

            # Sort workers by tpu-worker-id for deterministic task-to-worker mapping
            available.sort(
                key=lambda w: context.capacities[w]
                .attributes.get(WellKnownAttribute.TPU_WORKER_ID, AttributeValue(0))
                .value
            )

            # Sort tasks by task_index
            sorted_task_ids = sorted(task_ids, key=lambda t: t.require_task()[1])

            # Assign tasks to workers in order
            assignments: list[tuple[JobName, WorkerId]] = []
            for task_id, worker_id in zip(sorted_task_ids, available[:num_tasks], strict=False):
                context.capacities[worker_id].deduct(req)
                context.assignment_counts[worker_id] = context.assignment_counts.get(worker_id, 0) + 1
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
        max_building_tasks: int | None = None,
        max_assignments_per_worker: int | None = None,
    ) -> SchedulingContext:
        """Create a scheduling context for the given workers.

        This is the boundary conversion point: accepts WorkerSnapshot-compatible
        objects (e.g. worker rows) and converts them to scheduler-internal types.

        Args:
            workers: Workers to include (any objects satisfying WorkerSnapshot)
            building_counts: Map of worker_id -> count of tasks in BUILDING state
            pending_tasks: Task IDs in scheduling priority order
            jobs: Job requirements indexed by job ID
            max_building_tasks: Override for max building tasks per worker.
                If None, uses the scheduler's configured default.
            max_assignments_per_worker: Override for max assignments per worker per cycle.
                If None, uses DEFAULT_MAX_ASSIGNMENTS_PER_WORKER.
        """
        limit = max_building_tasks if max_building_tasks is not None else self._max_building_tasks_per_worker
        assignments_limit = (
            max_assignments_per_worker if max_assignments_per_worker is not None else DEFAULT_MAX_ASSIGNMENTS_PER_WORKER
        )
        return SchedulingContext.from_workers(
            workers,
            building_counts=building_counts,
            max_building_tasks=limit,
            pending_tasks=pending_tasks,
            jobs=jobs,
            max_assignments_per_worker=assignments_limit,
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
        all_constraints = list(req.constraints)
        hard_constraints, _soft_constraints = split_hard_soft(all_constraints)
        # Only hard constraints filter — soft constraints are preferences, not filters.
        matching_ids = context.matching_workers(hard_constraints)
        group_by = req.coscheduling_group_by

        if not matching_ids:
            constraint_keys = [c.key for c in hard_constraints]
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
