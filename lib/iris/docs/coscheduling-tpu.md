# TPU Coscheduling Design

## Overview

Add coscheduling support to Iris so that multi-host TPU jobs can be scheduled atomically onto complete TPU slices. A coscheduled job specifies a `group_by` attribute (e.g., `tpu-name`); all tasks must land on workers sharing that attribute value.

This design also removes the existing `gang_id` infrastructure, which is superseded by coscheduling.

## Design Principles

1. **Scheduler remains stateless**: The `Scheduler` is a pure function that takes state as input and returns assignments. All persistent state lives in `ControllerState`.

2. **Single source of truth**: Worker capacity is tracked exclusively in `ControllerWorker.committed_*` fields within `ControllerState`. The scheduler builds transient snapshots each cycle.

3. **Synchronous dispatch**: The `Controller` is synchronous with threading (not async). Dispatch happens under `_scheduler_lock`.

4. **Commit-then-dispatch**: Resources are committed via `TaskAssignedEvent` before RPC. On RPC failure, resources are released via state events.

## Data Model

### Proto Changes (`cluster.proto`)

```protobuf
// Attribute value - workers report these, jobs filter on them
message AttributeValue {
  oneof value {
    string string_value = 1;
    int64 int_value = 2;
    double float_value = 3;
  }
}

// Constraint operators for job scheduling
enum ConstraintOp {
  CONSTRAINT_OP_EQ = 0;
  CONSTRAINT_OP_NE = 1;
  CONSTRAINT_OP_EXISTS = 2;
  CONSTRAINT_OP_NOT_EXISTS = 3;
  CONSTRAINT_OP_GT = 4;
  CONSTRAINT_OP_GE = 5;
  CONSTRAINT_OP_LT = 6;
  CONSTRAINT_OP_LE = 7;
}

// A single scheduling constraint
message Constraint {
  string key = 1;
  ConstraintOp op = 2;
  AttributeValue value = 3;
}

// Coscheduling configuration
message CoschedulingConfig {
  string group_by = 1;  // Attribute key to group workers by (e.g., "tpu-name")
}

// Add to WorkerMetadata:
message WorkerMetadata {
  // ... existing fields ...
  map<string, AttributeValue> attributes = 40;
}

// Add to LaunchJobRequest:
message LaunchJobRequest {
  // ... existing fields ...
  repeated Constraint constraints = 14;
  CoschedulingConfig coscheduling = 15;
}

// REMOVED: gang_id field from ControllerJob - superseded by coscheduling
```

### Python Types (`types.py`)

```python
@dataclass(frozen=True)
@functools.total_ordering
class AttributeValue:
    """Comparable attribute value for constraint evaluation."""
    value: str | int | float

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AttributeValue):
            return NotImplemented
        return self.value == other.value

    def __lt__(self, other: "AttributeValue") -> bool:
        if type(self.value) != type(other.value):
            raise TypeError(f"Cannot compare {type(self.value)} with {type(other.value)}")
        return self.value < other.value

    def __hash__(self) -> int:
        return hash(self.value)

@dataclass
class Constraint:
    key: str
    op: ConstraintOp
    value: AttributeValue | None = None  # None for EXISTS/NOT_EXISTS

@dataclass
class CoschedulingConfig:
    group_by: str  # e.g., "tpu-name"
```

### Taints as Constraints

Taints are modeled as constraints using `NOT_EXISTS`. A worker with taint `"maintenance"` has attribute `taint:maintenance = true`. Jobs that don't tolerate this taint have an implicit constraint:

```python
Constraint(key="taint:maintenance", op=ConstraintOp.NOT_EXISTS)
```

### Worker Attribute Population

Workers populate attributes from their environment:

```python
def build_worker_attributes(metadata: WorkerMetadata) -> dict[str, AttributeValue]:
    attrs = {}
    if metadata.tpu_name:
        attrs["tpu-name"] = AttributeValue(metadata.tpu_name)
        attrs["tpu-worker-id"] = AttributeValue(int(metadata.tpu_worker_id))
        if metadata.device.HasField("tpu"):
            attrs["tpu-topology"] = AttributeValue(metadata.device.tpu.variant)
            topo = get_tpu_topology(metadata.device.tpu.variant)
            attrs["tpu-vm-count"] = AttributeValue(topo.vm_count)

    for taint in metadata.taints:
        attrs[f"taint:{taint}"] = AttributeValue(True)

    return attrs
```

## Scheduler Design

### Stateless Architecture

The scheduler is a **pure function** that:
1. Takes worker list and pending tasks as input
2. Builds transient `WorkerCapacity` snapshots
3. Returns `SchedulingResult` with proposed assignments
4. Never mutates any persistent state

All state mutations happen in `ControllerState` via events, triggered by `Controller`.

### Concurrency Model

The scheduling loop operates under `Controller._scheduler_lock`:
1. Lock acquired
2. `peek_pending_tasks()` and `get_available_workers()` called on `ControllerState`
3. `Scheduler.find_assignments()` returns proposed assignments
4. For each assignment: `TaskAssignedEvent` commits resources (synchronous)
5. All RPCs dispatched in parallel via thread pool (with 5s timeout)
6. Failed dispatches release resources via `TaskDispatchFailedEvent`
7. Lock released

The lock is held for the entire cycle, but RPC latency is bounded because:
- RPCs are dispatched in parallel via `ThreadPoolExecutor`
- Worker's `run_task` returns immediately (execution is async on worker)
- Short timeout (5s) prevents slow workers from blocking the control plane
- Worker heartbeats and task completions are processed on separate threads via state events

### WorkerCapacity: Transient Snapshot

```python
@dataclass
class WorkerCapacity:
    """
    Immutable snapshot of worker resources for scheduling.

    Built fresh each scheduling cycle from ControllerWorker.
    Used for tentative assignment decisions within a single cycle.
    """
    worker_id: WorkerId
    worker: ControllerWorker  # Reference to source (for returning in assignments)
    available_cpu: int
    available_memory: int
    available_gpus: int
    device_type: str
    device_variant: str
    attributes: dict[str, AttributeValue]

    @classmethod
    def from_worker(cls, worker: ControllerWorker) -> "WorkerCapacity":
        """Build capacity snapshot from current worker state."""
        return cls(
            worker_id=worker.worker_id,
            worker=worker,
            available_cpu=worker.available_cpu,
            available_memory=worker.available_memory,
            available_gpus=worker.available_gpus,
            device_type=worker.device_type,
            device_variant=worker.device_variant,
            attributes=dict(worker.attributes),
        )

    def can_fit(self, cpu: int, memory: int, gpus: int = 0) -> bool:
        """Check if this worker can fit the given resource requirements."""
        return (
            self.available_cpu >= cpu
            and self.available_memory >= memory
            and self.available_gpus >= gpus
        )

    def deduct(self, cpu: int, memory: int, gpus: int = 0) -> "WorkerCapacity":
        """Return new capacity with resources tentatively deducted."""
        return dataclasses.replace(
            self,
            available_cpu=self.available_cpu - cpu,
            available_memory=self.available_memory - memory,
            available_gpus=self.available_gpus - gpus,
        )
```

### SchedulingContext: Per-Cycle Index

The `SchedulingContext` is built fresh each scheduling cycle. It provides fast constraint matching via posting lists, but these are read-only after construction. As workers are tentatively assigned, we track capacity changes in a separate dict.

```python
@dataclass
class SchedulingContext:
    """
    Transient index for a single scheduling cycle.

    Built from worker capacities at cycle start.
    Posting lists are read-only (not updated during assignment).
    Capacity is tracked separately and verified before final assignment.
    """
    # Original capacities (immutable reference)
    original_capacities: dict[WorkerId, WorkerCapacity]

    # Tentative capacities (modified during scheduling)
    capacities: dict[WorkerId, WorkerCapacity]

    # All worker IDs for NOT_EXISTS queries
    all_worker_ids: set[WorkerId]

    # Posting lists for fast constraint matching (read-only after construction)
    discrete_lists: dict[str, dict[str, set[WorkerId]]]  # attr -> value -> workers

    @classmethod
    def from_workers(cls, workers: list[ControllerWorker]) -> "SchedulingContext":
        """Build context from worker list."""
        capacities = {w.worker_id: WorkerCapacity.from_worker(w) for w in workers if w.healthy}

        # Build posting lists for discrete attributes
        discrete_lists: dict[str, dict[str, set[WorkerId]]] = {}
        for worker_id, cap in capacities.items():
            for key, attr_value in cap.attributes.items():
                if isinstance(attr_value.value, str):
                    if key not in discrete_lists:
                        discrete_lists[key] = {}
                    value = attr_value.value
                    if value not in discrete_lists[key]:
                        discrete_lists[key][value] = set()
                    discrete_lists[key][value].add(worker_id)

        return cls(
            original_capacities=capacities,
            capacities=dict(capacities),  # Mutable copy
            all_worker_ids=set(capacities.keys()),
            discrete_lists=discrete_lists,
        )

    def get_matching_workers(self, constraints: Sequence[Constraint]) -> set[WorkerId]:
        """
        Get workers matching ALL constraints.

        Uses posting lists for fast EQ lookups, falls back to linear scan for others.
        """
        if not constraints:
            return self.all_worker_ids.copy()

        result: set[WorkerId] | None = None

        for constraint in constraints:
            matches = self._evaluate_constraint(constraint)

            if result is None:
                result = matches
            else:
                result &= matches

            if not result:
                return set()

        return result or set()

    def _evaluate_constraint(self, constraint: Constraint) -> set[WorkerId]:
        """Evaluate a single constraint, returning matching worker IDs."""
        key = constraint.key
        op = constraint.op

        # Fast path: EQ on discrete attribute with posting list
        if op == ConstraintOp.EQ and key in self.discrete_lists:
            target = constraint.value.value if constraint.value else None
            return self.discrete_lists[key].get(target, set()).copy()

        # Fast path: EXISTS check
        if op == ConstraintOp.EXISTS:
            if key in self.discrete_lists:
                result = set()
                for workers in self.discrete_lists[key].values():
                    result.update(workers)
                return result
            # Fall through to linear scan for numeric attributes

        # Fast path: NOT_EXISTS
        if op == ConstraintOp.NOT_EXISTS:
            if key in self.discrete_lists:
                has_attr = set()
                for workers in self.discrete_lists[key].values():
                    has_attr.update(workers)
                return self.all_worker_ids - has_attr
            # Attribute doesn't exist for any worker
            return self.all_worker_ids.copy()

        # Slow path: linear scan for NE, GT, GE, LT, LE, or non-indexed attributes
        result = set()
        for worker_id, cap in self.capacities.items():
            attr = cap.attributes.get(key)
            if self._matches(attr, op, constraint.value):
                result.add(worker_id)
        return result

    def _matches(
        self,
        attr: AttributeValue | None,
        op: ConstraintOp,
        target: AttributeValue | None,
    ) -> bool:
        """Check if attribute matches constraint."""
        if op == ConstraintOp.EXISTS:
            return attr is not None
        if op == ConstraintOp.NOT_EXISTS:
            return attr is None
        if attr is None:
            return False
        if target is None:
            return False

        match op:
            case ConstraintOp.EQ:
                return attr.value == target.value
            case ConstraintOp.NE:
                return attr.value != target.value
            case ConstraintOp.GT:
                return attr.value > target.value
            case ConstraintOp.GE:
                return attr.value >= target.value
            case ConstraintOp.LT:
                return attr.value < target.value
            case ConstraintOp.LE:
                return attr.value <= target.value
        return False

    def deduct_capacity(self, worker_id: WorkerId, cpu: int, memory: int, gpus: int = 0):
        """Tentatively deduct capacity during scheduling."""
        cap = self.capacities[worker_id]
        self.capacities[worker_id] = cap.deduct(cpu, memory, gpus)

    def verify_capacity(self, worker_id: WorkerId, cpu: int, memory: int, gpus: int = 0) -> bool:
        """Verify worker still has capacity (guards against stale posting lists)."""
        cap = self.capacities.get(worker_id)
        if cap is None:
            return False
        return cap.can_fit(cpu, memory, gpus)
```

### Scheduler Interface

```python
class Scheduler:
    """
    Stateless scheduler - pure function for finding task assignments.

    All state is passed as parameters. Returns proposed assignments
    without mutating any persistent state.
    """

    def find_assignments(
        self,
        pending_tasks: list[ControllerTask],
        workers: list[ControllerWorker],
        get_job: Callable[[JobId], ControllerJob | None],
    ) -> SchedulingResult:
        """
        Find task-to-worker assignments.

        Args:
            pending_tasks: Tasks ready for scheduling (from state.peek_pending_tasks())
            workers: Available workers (from state.get_available_workers())
            get_job: Lookup function for jobs (typically state.get_job)

        Returns:
            SchedulingResult with proposed assignments and timed-out tasks.
            Caller commits assignments via TaskAssignedEvent.
        """
        ctx = SchedulingContext.from_workers(workers)

        assignments: list[tuple[ControllerTask, ControllerWorker]] = []
        scheduled_task_ids: set[TaskId] = set()
        timed_out_tasks: list[ControllerTask] = []

        # Group tasks by job for coscheduled handling
        tasks_by_job: dict[JobId, list[ControllerTask]] = defaultdict(list)
        for task in pending_tasks:
            tasks_by_job[task.job_id].append(task)

        # Handle coscheduled jobs first (all-or-nothing)
        for job_id, job_tasks in tasks_by_job.items():
            job = get_job(job_id)
            if job is None or not job.is_coscheduled:
                continue

            result = self._find_coscheduled_assignments(ctx, job_tasks, job)
            if result:
                assignments.extend(result)
                scheduled_task_ids.update(t.task_id for t, _ in result)

        # Handle remaining non-coscheduled tasks (first-fit)
        for task in pending_tasks:
            if task.task_id in scheduled_task_ids:
                continue

            job = get_job(task.job_id)
            if job is None:
                continue

            # Check for scheduling timeout
            if self._is_timed_out(task, job):
                timed_out_tasks.append(task)
                continue

            worker = self._find_worker_for_task(ctx, job)
            if worker:
                assignments.append((task, worker))
                scheduled_task_ids.add(task.task_id)

        return SchedulingResult(assignments=assignments, timed_out_tasks=timed_out_tasks)

    def _find_worker_for_task(
        self,
        ctx: SchedulingContext,
        job: ControllerJob,
    ) -> ControllerWorker | None:
        """Find first worker that matches constraints and has capacity."""
        matching = ctx.get_matching_workers(list(job.request.constraints))
        resources = job.request.resources

        for worker_id in matching:
            # Verify capacity (guards against stale posting lists)
            if ctx.verify_capacity(worker_id, resources.cpu, resources.memory_bytes):
                # Deduct tentatively for subsequent assignments in this cycle
                ctx.deduct_capacity(worker_id, resources.cpu, resources.memory_bytes)
                return ctx.capacities[worker_id].worker

        return None

    def _find_coscheduled_assignments(
        self,
        ctx: SchedulingContext,
        tasks: list[ControllerTask],
        job: ControllerJob,
    ) -> list[tuple[ControllerTask, ControllerWorker]] | None:
        """
        Find atomic assignment for a coscheduled task group.

        Returns None if no valid worker group exists with sufficient capacity.
        All tasks must land on workers sharing the same group_by attribute value.
        """
        group_by = job.coscheduling_group_by
        if group_by is None:
            return None

        num_tasks = len(tasks)
        resources = job.request.resources
        constraints = list(job.request.constraints)

        # Get workers matching all constraints
        matching = ctx.get_matching_workers(constraints)

        # Group by coscheduling key
        groups: dict[str, list[WorkerId]] = defaultdict(list)
        for worker_id in matching:
            cap = ctx.capacities[worker_id]
            key_value = cap.attributes.get(group_by)
            if key_value is not None:
                groups[str(key_value.value)].append(worker_id)

        # Find first group with enough workers that have capacity
        for group_key, group_worker_ids in groups.items():
            available = []
            for worker_id in group_worker_ids:
                if ctx.verify_capacity(worker_id, resources.cpu, resources.memory_bytes):
                    available.append(worker_id)

            if len(available) >= num_tasks:
                # Sort by tpu-worker-id for deterministic task-to-worker mapping
                available.sort(
                    key=lambda w: ctx.capacities[w].attributes.get(
                        "tpu-worker-id", AttributeValue(0)
                    ).value
                )
                sorted_tasks = sorted(tasks, key=lambda t: t.task_index)

                result = []
                for task, worker_id in zip(sorted_tasks, available[:num_tasks], strict=False):
                    # Deduct tentatively
                    ctx.deduct_capacity(worker_id, resources.cpu, resources.memory_bytes)
                    result.append((task, ctx.original_capacities[worker_id].worker))

                return result

        return None

    def _is_timed_out(self, task: ControllerTask, job: ControllerJob) -> bool:
        """Check if task has exceeded scheduling timeout."""
        timeout_ms = job.request.scheduling_timeout_seconds * 1000
        if timeout_ms <= 0:
            return False
        elapsed_ms = time.time() * 1000 - task.created_at_ms
        return elapsed_ms > timeout_ms
```

## Controller Integration

### Dispatch Flow

The `Controller` orchestrates scheduling and dispatch. Resource commitment happens synchronously under the lock, but RPC dispatch happens in parallel with short timeouts.

**Key insight**: The worker's `run_task` RPC returns immediately after queuing the task for execution - actual execution happens in a separate thread on the worker. This means RPC latency should be minimal (network round-trip only), so we can safely dispatch in parallel with a short timeout (e.g., 5 seconds).

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

# RPC timeout - worker should respond immediately since execution is async
DISPATCH_RPC_TIMEOUT_SECONDS = 5.0


class Controller:
    def __init__(
        self,
        config: ControllerConfig,
        state: ControllerState | None = None,
        scheduler: Scheduler | None = None,
    ):
        self._config = config
        self._state = state or ControllerState()
        self._scheduler = scheduler or Scheduler()
        self._scheduler_lock = threading.RLock()
        self._dispatch_executor = ThreadPoolExecutor(
            max_workers=config.max_dispatch_parallelism,  # e.g., 32
            thread_name_prefix="dispatch",
        )

    def _run_scheduling(self) -> None:
        """Run one scheduling cycle. Called from background thread."""
        with self._scheduler_lock:
            pending_tasks = self._state.peek_pending_tasks()
            workers = self._state.get_available_workers()

            result = self._scheduler.find_assignments(
                pending_tasks,
                workers,
                self._state.get_job,
            )

            self._apply_scheduling_result(result)

    def _apply_scheduling_result(self, result: SchedulingResult) -> None:
        """Apply scheduling result: dispatch assignments, handle timeouts."""
        # Group assignments by job for coscheduled handling
        by_job: dict[JobId, list[tuple[ControllerTask, ControllerWorker]]] = defaultdict(list)
        for task, worker in result.assignments:
            by_job[task.job_id].append((task, worker))

        for job_id, job_assignments in by_job.items():
            job = self._state.get_job(job_id)
            if job is None:
                continue

            if job.is_coscheduled:
                self._dispatch_coscheduled_group(job_assignments, job)
            else:
                self._dispatch_tasks_parallel(job_assignments)

        for task in result.timed_out_tasks:
            self._mark_task_unschedulable(task)

    def _dispatch_tasks_parallel(
        self,
        assignments: list[tuple[ControllerTask, ControllerWorker]],
    ) -> None:
        """
        Dispatch multiple independent tasks in parallel.

        Flow:
        1. Commit all resources synchronously (under lock)
        2. Send all RPCs in parallel with timeout
        3. Handle failures by releasing resources
        """
        if not assignments:
            return

        # Phase 1: Commit all resources synchronously
        committed: list[tuple[ControllerTask, ControllerWorker]] = []
        for task, worker in assignments:
            self._state.handle_event(TaskAssignedEvent(
                task_id=task.task_id,
                worker_id=worker.worker_id,
            ))
            committed.append((task, worker))

        # Phase 2: Send RPCs in parallel
        futures = {
            self._dispatch_executor.submit(
                self._send_run_task_rpc, task, worker
            ): (task, worker)
            for task, worker in committed
        }

        # Phase 3: Collect results and handle failures
        for future in as_completed(futures, timeout=DISPATCH_RPC_TIMEOUT_SECONDS + 1):
            task, worker = futures[future]
            try:
                future.result(timeout=0)  # Already completed
            except Exception as e:
                logger.warning(f"Dispatch failed for {task.task_id}: {e}")
                self._state.handle_event(TaskDispatchFailedEvent(
                    task_id=task.task_id,
                    worker_id=worker.worker_id,
                    error=str(e),
                ))

    def _send_run_task_rpc(
        self,
        task: ControllerTask,
        worker: ControllerWorker,
    ) -> None:
        """Send run_task RPC with timeout. Called from thread pool."""
        stub = self._get_worker_stub(worker)
        stub.run_task(
            RunTaskRequest(
                task_id=str(task.task_id),
                attempt_id=str(task.current_attempt_id),
                # ... other fields
            ),
            timeout=DISPATCH_RPC_TIMEOUT_SECONDS,
        )

    def _dispatch_coscheduled_group(
        self,
        assignments: list[tuple[ControllerTask, ControllerWorker]],
        job: ControllerJob,
    ) -> None:
        """
        Dispatch all tasks in a coscheduled group in parallel.

        Commits all resources first, then sends all RPCs in parallel.
        On any RPC failure, releases resources for failed tasks.
        Successfully dispatched tasks continue running.
        """
        if not assignments:
            return

        # Phase 1: Commit all resources synchronously
        for task, worker in assignments:
            self._state.handle_event(TaskAssignedEvent(
                task_id=task.task_id,
                worker_id=worker.worker_id,
            ))

        # Phase 2: Send all RPCs in parallel
        futures = {
            self._dispatch_executor.submit(
                self._send_run_task_rpc, task, worker
            ): (task, worker)
            for task, worker in assignments
        }

        # Phase 3: Collect results
        failed_tasks: list[tuple[ControllerTask, ControllerWorker, str]] = []
        for future in as_completed(futures, timeout=DISPATCH_RPC_TIMEOUT_SECONDS + 1):
            task, worker = futures[future]
            try:
                future.result(timeout=0)
            except Exception as e:
                failed_tasks.append((task, worker, str(e)))

        # Phase 4: Handle failures
        # For coscheduled jobs, if ANY task fails to dispatch, we have a problem.
        # The successfully dispatched tasks will start running, but the group is incomplete.
        # We release resources for failed tasks; the running tasks will eventually fail
        # due to missing peers (e.g., collective ops will timeout).
        for task, worker, error in failed_tasks:
            logger.warning(f"Coscheduled dispatch failed for {task.task_id}: {error}")
            self._state.handle_event(TaskDispatchFailedEvent(
                task_id=task.task_id,
                worker_id=worker.worker_id,
                error=error,
            ))
```

### State Events for Dispatch

```python
# In events.py

@dataclass
class TaskDispatchFailedEvent:
    """Fired when RPC dispatch fails. Releases committed resources."""
    task_id: TaskId
    worker_id: WorkerId
    error: str


# In state.py

class ControllerState:
    def _on_task_dispatch_failed(
        self,
        txn: TransactionLog,
        event: TaskDispatchFailedEvent,
    ) -> None:
        """Handle failed dispatch - revert attempt and release resources."""
        task = self._tasks.get(event.task_id)
        if task is None:
            return

        worker = self._workers.get(event.worker_id)
        job = self._jobs.get(task.job_id)

        # Revert the attempt (sets task back to PENDING)
        if task.current_attempt is not None:
            task.revert_attempt()
            txn.log("dispatch_reverted", task.task_id)

        # Release resources if worker still exists
        if worker is not None and job is not None:
            resources = job.request.resources
            worker.unassign_task(task.task_id, resources)
            txn.log("resources_released", task.task_id, worker_id=str(worker.worker_id))
```

## Failure Handling

### Coscheduled Job Properties

```python
@dataclass
class ControllerJob:
    # ... existing fields ...

    @property
    def is_coscheduled(self) -> bool:
        return self.request.HasField("coscheduling")

    @property
    def coscheduling_group_by(self) -> str | None:
        if self.is_coscheduled:
            return self.request.coscheduling.group_by
        return None

    # REMOVED: gang_id field - superseded by coscheduling
```

### Group Failure Cascade

When one task in a coscheduled job fails, all running siblings are killed:

```python
class ControllerState:
    def _on_task_state_changed(self, txn: TransactionLog, event: TaskStateChangedEvent) -> None:
        task = self._tasks[event.task_id]
        job = self._jobs[task.job_id]
        old_state = task.state

        # Process this task's state change via proper state machine
        task.handle_attempt_result(event.new_state, event.error, event.exit_code)
        txn.task_state_changes.append((task.task_id, task.state))

        # Coscheduled group failure: if one task fails terminally, fail all siblings
        if job.is_coscheduled and task.is_finished() and task.state in (
            cluster_pb2.TASK_STATE_FAILED,
            cluster_pb2.TASK_STATE_WORKER_FAILED,
        ):
            for sibling_id in self._tasks_by_job.get(job.job_id, []):
                if sibling_id == task.task_id:
                    continue
                sibling = self._tasks[sibling_id]
                if sibling.state == cluster_pb2.TASK_STATE_RUNNING:
                    sibling_old = sibling.state
                    sibling.handle_attempt_result(
                        cluster_pb2.TASK_STATE_WORKER_FAILED,
                        error=f"Coscheduled sibling {task.task_id} failed",
                    )
                    job.on_task_transition(sibling_old, sibling.state)
                    txn.tasks_to_kill.add(sibling_id)
                    txn.log("coscheduled_sibling_killed", sibling_id, trigger_task=str(task.task_id))

        # ... rest of existing code ...
```

## Validation

```python
def validate_coscheduled_job(request: LaunchJobRequest):
    if not request.HasField("coscheduling"):
        return

    if not request.resources.device.HasField("tpu"):
        raise ValueError("Coscheduled jobs require TPU device")

    tpu_variant = request.resources.device.tpu.variant
    topo = get_tpu_topology(tpu_variant)

    if request.resources.replicas != topo.vm_count:
        raise ValueError(
            f"TPU {tpu_variant} requires {topo.vm_count} replicas, "
            f"got {request.resources.replicas}"
        )
```

## Migration: Removing Gang Scheduling

Remove `gang_id` field and related logic entirely:

1. **Proto**: Remove `gang_id` from `ControllerJob` message
2. **State**: Remove `gang_id` from `ControllerJob` dataclass, remove `get_gang_jobs()` method
3. **Events**: Remove any gang-related event handling
4. **Tests**: Update tests to use coscheduling instead

## Out of Scope: Preemption

This design uses first-fit scheduling without preemption. Large coscheduled jobs may wait if the cluster is fragmented with smaller jobs.

**Mitigations**:
- Use dedicated TPU pools for large jobs via constraints (e.g., `tpu-pool=large-jobs`)
- Monitor pending coscheduled jobs for starvation alerts
- Consider time-based fairness policies in future work

Preemption support (evicting lower-priority jobs to make room for higher-priority coscheduled jobs) is deferred to a future design.

## Implementation Plan (Staged)

Each stage follows the SPIRAL approach: proto → types → worker → controller → scheduler → test.
Each stage is independently testable and deployable without breaking existing functionality.

---

### Stage 1: Worker Attributes

**Goal**: Workers report attributes; controller stores them; scheduler can access them.

**Files Changed**:
- `lib/iris/protos/cluster.proto`
- `lib/iris/src/iris/cluster/types.py`
- `lib/iris/src/iris/cluster/worker/env_probe.py`
- `lib/iris/src/iris/cluster/controller/state.py`
- `lib/iris/src/iris/cluster/controller/scheduler.py`
- `lib/iris/tests/cluster/controller/test_state.py`

**Proto** (`cluster.proto`):
```protobuf
message AttributeValue {
  oneof value {
    string string_value = 1;
    int64 int_value = 2;
    double float_value = 3;
  }
}

message WorkerMetadata {
  // ... existing fields ...
  map<string, AttributeValue> attributes = 40;
}
```

**Types** (`types.py`):
```python
@dataclass(frozen=True)
class AttributeValue:
    value: str | int | float

    def to_proto(self) -> cluster_pb2.AttributeValue:
        proto = cluster_pb2.AttributeValue()
        if isinstance(self.value, str):
            proto.string_value = self.value
        elif isinstance(self.value, int):
            proto.int_value = self.value
        else:
            proto.float_value = self.value
        return proto

    @staticmethod
    def from_proto(proto: cluster_pb2.AttributeValue) -> "AttributeValue":
        if proto.HasField("string_value"):
            return AttributeValue(proto.string_value)
        elif proto.HasField("int_value"):
            return AttributeValue(proto.int_value)
        else:
            return AttributeValue(proto.float_value)
```

**Worker** (`env_probe.py`):
```python
def probe(self) -> cluster_pb2.WorkerMetadata:
    attributes = {}
    if tpu_name:
        attributes["tpu-name"] = cluster_pb2.AttributeValue(string_value=tpu_name)
        attributes["tpu-worker-id"] = cluster_pb2.AttributeValue(int_value=int(tpu_worker_id or "0"))

    return cluster_pb2.WorkerMetadata(..., attributes=attributes)
```

**State** (`state.py`):
```python
@dataclass
class ControllerWorker:
    # ... existing fields ...
    attributes: dict[str, AttributeValue] = field(default_factory=dict)

# In _on_worker_registered:
worker.attributes = {k: AttributeValue.from_proto(v) for k, v in event.metadata.attributes.items()}
```

**Scheduler** (`scheduler.py`):
```python
@dataclass
class WorkerCapacity:
    # ... existing fields ...
    attributes: dict[str, AttributeValue] = field(default_factory=dict)
```

**Test** (`test_state.py`):
```python
def test_worker_registers_with_attributes(worker_metadata):
    state = ControllerState()
    metadata = worker_metadata()
    metadata.attributes["tpu-name"].string_value = "my-tpu"
    metadata.attributes["tpu-worker-id"].int_value = 0

    register_worker(state, "w1", "host:8080", metadata)

    worker = state.get_worker(WorkerId("w1"))
    assert worker.attributes["tpu-name"].value == "my-tpu"
    assert worker.attributes["tpu-worker-id"].value == 0
```

---

### Stage 2: Constraints on Jobs

**Goal**: Jobs can specify constraints; scheduler filters workers.

**Files Changed**:
- `lib/iris/protos/cluster.proto`
- `lib/iris/src/iris/cluster/controller/scheduler.py`
- `lib/iris/tests/cluster/controller/test_scheduler.py`

**Proto** (`cluster.proto`):
```protobuf
enum ConstraintOp {
  CONSTRAINT_OP_EQ = 0;
  CONSTRAINT_OP_NE = 1;
  CONSTRAINT_OP_EXISTS = 2;
  CONSTRAINT_OP_NOT_EXISTS = 3;
  CONSTRAINT_OP_GT = 4;
  CONSTRAINT_OP_GE = 5;
  CONSTRAINT_OP_LT = 6;
  CONSTRAINT_OP_LE = 7;
}

message Constraint {
  string key = 1;
  ConstraintOp op = 2;
  AttributeValue value = 3;
}

message LaunchJobRequest {
  // ... existing fields ...
  repeated Constraint constraints = 14;
}
```

**Scheduler** (`scheduler.py`):
```python
def _worker_matches_constraints(
    self,
    cap: WorkerCapacity,
    constraints: Sequence[cluster_pb2.Constraint],
) -> bool:
    """Check if worker matches all constraints."""
    for c in constraints:
        attr = cap.attributes.get(c.key)
        if not self._evaluate_constraint(attr, c):
            return False
    return True

def _evaluate_constraint(
    self,
    attr: AttributeValue | None,
    c: cluster_pb2.Constraint,
) -> bool:
    op = c.op
    if op == cluster_pb2.CONSTRAINT_OP_EXISTS:
        return attr is not None
    if op == cluster_pb2.CONSTRAINT_OP_NOT_EXISTS:
        return attr is None
    if attr is None:
        return False

    target = AttributeValue.from_proto(c.value)
    match op:
        case cluster_pb2.CONSTRAINT_OP_EQ:
            return attr.value == target.value
        case cluster_pb2.CONSTRAINT_OP_NE:
            return attr.value != target.value
        case cluster_pb2.CONSTRAINT_OP_GT:
            return attr.value > target.value
        case cluster_pb2.CONSTRAINT_OP_GE:
            return attr.value >= target.value
        case cluster_pb2.CONSTRAINT_OP_LT:
            return attr.value < target.value
        case cluster_pb2.CONSTRAINT_OP_LE:
            return attr.value <= target.value
    return False
```

**Test** (`test_scheduler.py`):
```python
def test_constraint_filters_workers_by_attribute(scheduler, state, job_request, worker_metadata):
    """Job with constraint only schedules on workers with matching attribute."""
    meta1 = worker_metadata()
    meta1.attributes["tpu-name"].string_value = "tpu-a"
    register_worker(state, "w1", "addr1", meta1)

    meta2 = worker_metadata()
    meta2.attributes["tpu-name"].string_value = "tpu-b"
    register_worker(state, "w2", "addr2", meta2)

    req = job_request()
    c = req.constraints.add()
    c.key = "tpu-name"
    c.op = cluster_pb2.CONSTRAINT_OP_EQ
    c.value.string_value = "tpu-a"
    submit_job(state, "j1", req)

    result = scheduler.find_assignments(
        state.peek_pending_tasks(),
        state.get_available_workers(),
        state.get_job,
    )
    assert len(result.assignments) == 1
    assert result.assignments[0][1].worker_id == WorkerId("w1")
```

---

### Stage 3: Coscheduling

**Goal**: Jobs can request coscheduling via `group_by`; scheduler assigns all tasks atomically.

**Files Changed**:
- `lib/iris/protos/cluster.proto`
- `lib/iris/src/iris/cluster/controller/state.py`
- `lib/iris/src/iris/cluster/controller/scheduler.py`
- `lib/iris/src/iris/cluster/controller/controller.py`
- `lib/iris/tests/cluster/controller/test_scheduler.py`

**Proto** (`cluster.proto`):
```protobuf
message CoschedulingConfig {
  string group_by = 1;  // Attribute key to group workers by (e.g., "tpu-name")
}

message LaunchJobRequest {
  // ... existing fields ...
  CoschedulingConfig coscheduling = 15;
}
```

**State** (`state.py`):
```python
@dataclass
class ControllerJob:
    # ... existing fields ...

    @property
    def is_coscheduled(self) -> bool:
        return self.request.HasField("coscheduling")

    @property
    def coscheduling_group_by(self) -> str | None:
        if self.is_coscheduled:
            return self.request.coscheduling.group_by
        return None
```

**Test** (`test_scheduler.py`):
```python
def test_coscheduled_job_assigns_all_tasks_atomically(scheduler, state, worker_metadata):
    """Coscheduled job assigns all tasks to workers in the same group."""
    # Create 4 workers on tpu-a
    for i in range(4):
        meta = worker_metadata()
        meta.attributes["tpu-name"].string_value = "tpu-a"
        meta.attributes["tpu-worker-id"].int_value = i
        register_worker(state, f"w{i}", f"addr{i}", meta)

    # Create coscheduled job with 4 replicas
    req = cluster_pb2.LaunchJobRequest(
        name="coschedule-test",
        serialized_entrypoint=b"test",
        resources=cluster_pb2.ResourceSpecProto(cpu=1, memory_bytes=1024**3, replicas=4),
        environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
    )
    req.coscheduling.group_by = "tpu-name"
    submit_job(state, "j1", req)

    result = scheduler.find_assignments(
        state.peek_pending_tasks(),
        state.get_available_workers(),
        state.get_job,
    )

    # All 4 tasks should be assigned
    assert len(result.assignments) == 4

    # All assigned to workers with same tpu-name
    assigned_tpu_names = {w.attributes["tpu-name"].value for _, w in result.assignments}
    assert assigned_tpu_names == {"tpu-a"}

    # Tasks assigned in order: task-0 -> worker-0, task-1 -> worker-1, etc.
    for task, worker in result.assignments:
        expected_worker_id = f"w{task.task_index}"
        assert worker.worker_id == WorkerId(expected_worker_id)


def test_coscheduled_job_waits_when_insufficient_workers(scheduler, state, worker_metadata):
    """Coscheduled job stays pending when not enough workers in any group."""
    # Only 2 workers on tpu-a
    for i in range(2):
        meta = worker_metadata()
        meta.attributes["tpu-name"].string_value = "tpu-a"
        meta.attributes["tpu-worker-id"].int_value = i
        register_worker(state, f"w{i}", f"addr{i}", meta)

    # Job requires 4 replicas
    req = cluster_pb2.LaunchJobRequest(
        name="coschedule-test",
        serialized_entrypoint=b"test",
        resources=cluster_pb2.ResourceSpecProto(cpu=1, memory_bytes=1024**3, replicas=4),
        environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
    )
    req.coscheduling.group_by = "tpu-name"
    submit_job(state, "j1", req)

    result = scheduler.find_assignments(
        state.peek_pending_tasks(),
        state.get_available_workers(),
        state.get_job,
    )

    # No assignments - job stays pending
    assert len(result.assignments) == 0
```

---

### Stage 4: Coscheduled Failure Handling

**Goal**: When one task in a coscheduled job fails, all sibling tasks are killed.

**Files Changed**:
- `lib/iris/src/iris/cluster/controller/state.py`
- `lib/iris/tests/cluster/controller/test_state.py`

**Test** (`test_state.py`):
```python
def test_coscheduled_task_failure_kills_siblings(state, worker_metadata):
    """When one coscheduled task fails, all running siblings are killed."""
    # Register 4 workers
    for i in range(4):
        meta = worker_metadata()
        meta.attributes["tpu-name"].string_value = "tpu-a"
        meta.attributes["tpu-worker-id"].int_value = i
        register_worker(state, f"w{i}", f"addr{i}", meta)

    # Create coscheduled job with 4 tasks
    req = cluster_pb2.LaunchJobRequest(
        name="coschedule-test",
        serialized_entrypoint=b"test",
        resources=cluster_pb2.ResourceSpecProto(cpu=1, memory_bytes=1024**3, replicas=4),
        environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
    )
    req.coscheduling.group_by = "tpu-name"
    tasks = submit_job(state, "j1", req)

    # Dispatch all tasks
    for i, task in enumerate(tasks):
        dispatch_task(state, task, WorkerId(f"w{i}"))

    # Fail task-0
    txn = state.handle_event(
        TaskStateChangedEvent(
            task_id=tasks[0].task_id,
            new_state=cluster_pb2.TASK_STATE_FAILED,
            error="OOM",
        )
    )

    # All other tasks should be marked for killing
    assert tasks[0].state == cluster_pb2.TASK_STATE_FAILED
    for task in tasks[1:]:
        assert task.state == cluster_pb2.TASK_STATE_WORKER_FAILED
        assert task.task_id in txn.tasks_to_kill
```

---

### Stage 5: Remove Gang Scheduling

**Goal**: Clean up superseded code now that coscheduling is complete.

**Files Changed**:
- `lib/iris/protos/cluster.proto`
- `lib/iris/src/iris/cluster/controller/state.py`
- `lib/iris/tests/cluster/controller/test_state.py`

**Changes**:
```python
# Remove from ControllerJob:
#   gang_id: str | None = None

# Remove from ControllerState:
#   self._gangs: dict[str, set[JobId]] = {}
#   def get_gang_jobs(self, gang_id: str) -> list[ControllerJob]: ...
```

---

### Stage 6: Posting Lists (Performance Optimization)

**Goal**: O(1) constraint matching for large clusters.

This is a **performance optimization** that doesn't change observable behavior.
Only implement if profiling shows constraint matching is a bottleneck.

The design above already includes posting lists in `SchedulingContext.discrete_lists`.
This stage would add:
- Numeric posting lists for range queries (if needed)
- Benchmark tests with 1000+ workers

**Note**: We intentionally avoid external dependencies like `sortedcontainers`.
The simple dict-based posting list is sufficient for EQ/EXISTS queries which
are the common case for TPU scheduling.

---

## Stage Summary

| Stage | Deliverable | Test Behavior |
|-------|-------------|---------------|
| 1 | Worker attributes (`AttributeValue`) | Worker registration stores typed attributes |
| 2 | Constraints on jobs (all operators) | Job only schedules on matching workers |
| 3 | Coscheduling (`group_by`) | All tasks assigned atomically to same group |
| 4 | Coscheduled failure cascade | One task fails → siblings killed |
| 5 | Remove `gang_id` | Legacy code cleaned up |
| 6 | Posting lists (optional) | Same behavior, faster at scale |

Each stage can be merged independently and provides incremental value.

## Files Affected Summary

| File | Stages |
|------|--------|
| `lib/iris/protos/cluster.proto` | 1, 2, 3, 5 |
| `lib/iris/src/iris/cluster/types.py` | 1 |
| `lib/iris/src/iris/cluster/worker/env_probe.py` | 1 |
| `lib/iris/src/iris/cluster/controller/state.py` | 1, 3, 4, 5 |
| `lib/iris/src/iris/cluster/controller/scheduler.py` | 1, 2, 3 |
| `lib/iris/src/iris/cluster/controller/controller.py` | 3 |
| `lib/iris/src/iris/cluster/controller/events.py` | 3 (TaskDispatchFailedEvent) |
| `lib/iris/tests/cluster/controller/test_state.py` | 1, 4, 5 |
| `lib/iris/tests/cluster/controller/test_scheduler.py` | 2, 3 |

# Review Feedback

> **Verdict**: ✅ **Approve with Warnings**
>
> The revised design addresses the critical stability and data integrity issues identified in the previous review. The shift to a stateless `Scheduler` and a synchronous dispatch model eliminates the split-brain risk.

## Resolved Issues

1.  **Architecture**: The `Scheduler` is now correctly defined as a pure, stateless function. This ensures the `ControllerState` remains the single source of truth.
2.  **Concurrency**: The synchronous dispatch model with `TaskDispatchFailedEvent` rollback is a safe, albeit lower-throughput, choice. It prioritizes correctness over scheduling latency, which is appropriate for TPU workloads where job durations are long (hours/days) and scheduling latency (milliseconds) is negligible.
3.  **Complexity**: The removal of posting lists for the initial implementation is a wise choice to reduce complexity.

## Remaining Risks & Warnings

1.  **Blocking RPCs**:
    -   **Warning**: Holding `_scheduler_lock` during `stub.run_task()` RPCs means the controller is fully blocked during dispatch.
    -   **Impact**: If a worker is slow to accept a request (e.g., network timeout), the entire cluster control plane freezes.
    -   **Mitigation**: Ensure concise timeouts on the `run_task` RPC (e.g., 5 seconds). Do not allow it to block indefinitely.

2.  **Hardcoded Topologies**:
    -   **Nit**: Relying on `get_tpu_topology` lookup tables restricts the control plane to knowing about every possible hardware variation in advance. A dynamic reporting model would be more robust long-term.

## Verdict
The design is now theoretically sound and safe to implement. Proceed with implementation, keeping the above warnings in mind regarding timeouts and multi-job support.
