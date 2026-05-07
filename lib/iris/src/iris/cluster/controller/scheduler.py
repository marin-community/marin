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
    WellKnownAttribute,
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
    """What a job needs from a worker. Scheduler's input type.

    The four cached scalars (`req_cpu_millicores`, `req_memory_bytes`,
    `req_gpu_count`, `req_tpu_count`) are derived once from the proto in
    `__post_init__` so the scheduler's per-(task, worker) `can_fit` hot loop
    does not pay protobuf attribute access overhead. The hot loop runs
    ~pending x workers times per scheduling cycle (≈10^5 on the marin cluster).
    """

    resources: job_pb2.ResourceSpecProto
    constraints: list[Constraint]
    is_coscheduled: bool
    coscheduling_group_by: str | None

    req_cpu_millicores: int = field(init=False)
    req_memory_bytes: int = field(init=False)
    req_gpu_count: int = field(init=False)
    req_tpu_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.req_cpu_millicores = self.resources.cpu_millicores
        self.req_memory_bytes = self.resources.memory_bytes
        self.req_gpu_count = get_gpu_count(self.resources.device)
        self.req_tpu_count = get_tpu_count(self.resources.device)


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

    def can_fit(self, req: JobRequirements) -> RejectionReason | None:
        """Check if this capacity can fit the job's resource requirements.

        Only checks resource capacity (CPU, memory, device count, building limit).
        Device type and variant matching is handled by matches_constraints() via
        the posting-list index in SchedulingContext.

        Hot path: called O(pending x workers) per scheduling cycle. Consumes
        the cached integer fields on `JobRequirements` so we never touch the
        proto here, and returns early on the first failing dimension instead
        of routing through `check_resource_fit`'s string-reason indirection.

        Returns:
            None if job fits, otherwise RejectionReason with lazy-formatted details
        """
        if self.building_task_count >= self.max_building_tasks:
            return RejectionReason(
                kind=RejectionKind.BUILDING_LIMIT,
                details={"current": self.building_task_count, "max": self.max_building_tasks},
            )

        cpu_need = req.req_cpu_millicores
        if cpu_need > 0 and cpu_need > self.available_cpu_millicores:
            return RejectionReason(
                kind=RejectionKind.CPU,
                details={"need": cpu_need, "have": self.available_cpu_millicores},
            )

        mem_need = req.req_memory_bytes
        if mem_need > 0 and mem_need > self.available_memory:
            return RejectionReason(
                kind=RejectionKind.MEMORY,
                details={"need": mem_need, "have": self.available_memory},
            )

        gpu_need = req.req_gpu_count
        if gpu_need > 0 and gpu_need > self.available_gpus:
            return RejectionReason(
                kind=RejectionKind.GPU_COUNT,
                details={"need": gpu_need, "have": self.available_gpus},
            )

        tpu_need = req.req_tpu_count
        if tpu_need > 0 and tpu_need > self.available_tpus:
            return RejectionReason(
                kind=RejectionKind.TPU_COUNT,
                details={"need": tpu_need, "have": self.available_tpus},
            )

        return None

    def deduct(self, req: JobRequirements) -> None:
        """Deduct job's resources from available capacity."""
        self.available_cpu_millicores -= req.req_cpu_millicores
        self.available_memory -= req.req_memory_bytes
        self.available_gpus -= req.req_gpu_count
        self.available_tpus -= req.req_tpu_count
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
class SchedulingResult:
    """Result of a scheduling cycle - pure data, no state mutation.

    Only contains successful assignments.
    Failure details are available via get_job_scheduling_diagnostics() for dashboard use.
    """

    assignments: list[tuple[JobName, WorkerId]] = field(default_factory=list)


def rank_by_soft_score(
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


def compute_candidates(req: JobRequirements, context: SchedulingContext) -> list[WorkerId]:
    """Constraint-filter and soft-rank workers for a given req.

    Used by both the hot find_assignments path and diagnostics, so that the
    "which workers are even candidates for this req?" decision exists in
    exactly one place.
    """
    hard_constraints, soft_constraints = split_hard_soft(list(req.constraints))
    matching = context.matching_workers(hard_constraints)
    if soft_constraints:
        return rank_by_soft_score(matching, soft_constraints, context)
    return list(matching)


def first_fitting_worker(
    candidates: Sequence[WorkerId],
    context: SchedulingContext,
    req: JobRequirements,
) -> WorkerId | None:
    """Return the first candidate that has capacity and is below the per-cycle cap.

    Sole authority on "is there a worker that can take this req right now?".
    Pure read: does not mutate context. Callers are responsible for deducting
    capacity / bumping assignment_counts on success.
    """
    max_per_worker = context.max_assignments_per_worker
    for worker_id in candidates:
        if context.assignment_counts.get(worker_id, 0) >= max_per_worker:
            continue
        if context.capacities[worker_id].can_fit(req) is None:
            return worker_id
    return None


def explain_unfittable(
    req: JobRequirements,
    context: SchedulingContext,
    max_building_tasks_per_worker: int,
) -> str:
    """Build a human-readable failure reason for a non-coscheduled req.

    Diagnostics-only — walks candidates a second time accumulating rejection
    counts and formatting per-dimension messages with totals. Returns
    "Schedulable — waiting for next scheduling cycle" if the req does fit
    after all (race against `find_assignments`).
    """
    if not context.capacities:
        return "No healthy workers available"

    hard_constraints, soft_constraints = split_hard_soft(list(req.constraints))
    matching = context.matching_workers(hard_constraints)
    if soft_constraints:
        candidates = rank_by_soft_score(matching, soft_constraints, context)
    else:
        candidates = list(matching)

    rejection_counts: dict[RejectionKind, int] = defaultdict(int)
    rejection_samples: dict[RejectionKind, RejectionReason] = {}
    max_per_worker = context.max_assignments_per_worker
    for worker_id in candidates:
        if context.assignment_counts.get(worker_id, 0) >= max_per_worker:
            continue
        rejection = context.capacities[worker_id].can_fit(req)
        if rejection is None:
            return "Schedulable — waiting for next scheduling cycle"
        rejection_counts[rejection.kind] += 1
        if rejection.kind not in rejection_samples:
            rejection_samples[rejection.kind] = rejection

    res = req.resources

    if rejection_counts:
        if RejectionKind.BUILDING_LIMIT in rejection_counts:
            workers_with_capacity = sum(
                1
                for cwid in candidates
                if context.assignment_counts.get(cwid, 0) < max_per_worker
                and context.capacities[cwid].available_cpu_millicores >= res.cpu_millicores
                and context.capacities[cwid].available_memory >= res.memory_bytes
            )
            if workers_with_capacity > 0:
                count = rejection_counts[RejectionKind.BUILDING_LIMIT]
                return (
                    f"Waiting for build slots: {count} worker(s) at building limit "
                    f"(max {max_building_tasks_per_worker} concurrent builds per worker), "
                    f"but have sufficient resources for this task"
                )

        reason_lines = []
        for kind in sorted(rejection_counts.keys(), key=lambda k: rejection_counts[k], reverse=True):
            count = rejection_counts[kind]
            sample = rejection_samples[kind]
            reason_lines.append(f"{sample} - {count} worker(s)")
        failure_reason = "\n".join(reason_lines)
        if hard_constraints:
            constraint_keys = [c.key for c in hard_constraints]
            failure_reason = f"{failure_reason}\n(with constraints={constraint_keys})"
        return failure_reason

    if hard_constraints:
        constraint_keys = [c.key for c in hard_constraints]
        return (
            f"No worker matches constraints and has sufficient resources "
            f"(need cpu={res.cpu_millicores / 1000:g} cores, memory={res.memory_bytes}, "
            f"constraints={constraint_keys})"
        )
    return (
        f"No worker has sufficient resources "
        f"(need cpu={res.cpu_millicores / 1000:g} cores, memory={res.memory_bytes})"
    )


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

        Per-job dedup: tasks of the same non-coscheduled job share one
        `JobRequirements`, so constraint matching and soft-score ranking are
        hoisted once per job. Once a job's first task in this pass fails to
        find a worker, capacities only decrease and assignment_counts only
        increase for the remainder of the pass — so all of that job's
        remaining same-req tasks would also fail. We mark the job exhausted
        and skip them, turning the inner work from O(pending x workers) into
        O(jobs x workers + min(pending, workers)).

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

        # Per-job memo for the non-coscheduled fan-out below.
        candidate_lists: dict[JobName, list[WorkerId]] = {}
        exhausted_jobs: set[JobName] = set()

        # Handle remaining non-coscheduled tasks (first-fit, in priority order).
        for task_id in context.pending_tasks:
            if task_id in scheduled_task_ids:
                continue

            job_id = task_id.parent
            if job_id is None:
                continue
            if job_id in exhausted_jobs:
                continue

            req = context.jobs.get(job_id)
            if req is None:
                logger.debug("Task %s has no job requirements, skipping", task_id)
                continue
            if req.is_coscheduled:
                continue

            candidates = candidate_lists.get(job_id)
            if candidates is None:
                candidates = compute_candidates(req, context)
                candidate_lists[job_id] = candidates

            worker_id = first_fitting_worker(candidates, context, req)
            if worker_id is None:
                exhausted_jobs.add(job_id)
                continue

            capacity = context.capacities[worker_id]
            capacity.deduct(req)
            context.assignment_counts[worker_id] = context.assignment_counts.get(worker_id, 0) + 1
            result.assignments.append((task_id, worker_id))

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

        candidates = compute_candidates(req, context)
        if first_fitting_worker(candidates, context, req) is not None:
            return "Schedulable — waiting for next scheduling cycle"
        return explain_unfittable(req, context, self._max_building_tasks_per_worker)

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
                candidates = compute_candidates(req, context)
                if first_fitting_worker(candidates, context, req) is not None:
                    return "Schedulable — waiting for next scheduling cycle"
                return explain_unfittable(req, context, self._max_building_tasks_per_worker)
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
