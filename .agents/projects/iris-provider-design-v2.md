# Iris Provider Redesign (v2) — DirectTaskProvider

This document supersedes the original design. The key change: **no synthetic
worker entries** for k8s. The `KubernetesProvider` implements a new
`DirectTaskProvider` protocol where the provider handles its own scheduling
and the controller is a thin observation/routing layer.

---

## 1. Core Problem with v1

The v1 design registered a synthetic `Worker` row in the DB for the k8s
cluster (`execution_unit_id = WorkerId("k8s-cluster")`). The scheduler
assigned tasks to this synthetic worker, and the heartbeat loop sent
`DispatchBatch` objects to the provider.

This is the "hacky fake worker abstraction" the user rejected. The k8s provider
should own scheduling, autoscaling, and execution entirely — the controller
reflects its state.

---

## 2. Architecture

### Two execution models

| | WorkerProvider | KubernetesProvider |
|---|---|---|
| Protocol | `TaskProvider` (existing) | `DirectTaskProvider` (new) |
| Workers in DB | Yes (real daemons register) | No |
| Controller scheduler | Runs | Skipped |
| Controller autoscaler | Runs | Skipped |
| Sync loop | Drains dispatch queue per worker | Drains all PENDING tasks |
| Dashboard | Worker list + autoscaler panel | Provider status + scheduling events |

### DirectTaskProvider protocol

```python
# lib/iris/src/iris/cluster/controller/direct_provider.py

@dataclass
class SchedulingEvent:
    """K8s pod scheduling event (e.g. FailedScheduling, Scheduled)."""
    task_id: str
    attempt_id: int
    event_type: str    # "Scheduled", "FailedScheduling", "BackOff", etc.
    reason: str
    message: str
    timestamp: Timestamp

@dataclass
class ClusterCapacity:
    """Cluster-wide resource summary from the provider."""
    schedulable_nodes: int = 0
    total_cpu_millicores: int = 0
    available_cpu_millicores: int = 0
    total_memory_bytes: int = 0
    available_memory_bytes: int = 0

@dataclass
class DirectProviderBatch:
    """Snapshot passed to DirectTaskProvider.sync()."""
    tasks_to_run: list[cluster_pb2.Worker.RunTaskRequest]
    running_tasks: list[RunningTaskEntry]   # ASSIGNED/BUILDING/RUNNING with no worker_id
    tasks_to_kill: list[str]               # task IDs

@dataclass
class DirectProviderSyncResult:
    updates: list[TaskUpdate]
    scheduling_events: list[SchedulingEvent] = field(default_factory=list)
    capacity: ClusterCapacity | None = None

class DirectTaskProvider(Protocol):
    """Provider that manages its own scheduling and execution.

    No workers in Iris DB. Controller creates attempt rows for pending tasks,
    then passes them to the provider. Provider creates pods/jobs and returns
    state updates each sync cycle.
    """

    def sync(
        self,
        batch: DirectProviderBatch,
    ) -> DirectProviderSyncResult:
        """Sync state with the backend.

        tasks_to_run: Tasks with freshly-created attempt rows (ASSIGNED state,
                      NULL worker_id). Provider should start execution.
        running_tasks: Tasks previously dispatched, currently tracked as active.
        tasks_to_kill: Task IDs to cancel.
        """
        ...

    def fetch_live_logs(
        self,
        task_id: str,
        attempt_id: int,
        cursor: int,
        max_lines: int,
    ) -> tuple[list[logging_pb2.LogEntry], int]:
        """Fetch live logs for a running task."""
        ...

    def close(self) -> None: ...
```

---

## 3. Transitions Changes

Add to `transitions.py`:

```python
@dataclass
class DirectProviderBatch:
    """Batch from drain_for_direct_provider()."""
    tasks_to_run: list[cluster_pb2.Worker.RunTaskRequest]
    running_tasks: list[RunningTaskEntry]
    tasks_to_kill: list[str]
```

### New method: `drain_for_direct_provider()`

```python
def drain_for_direct_provider(self) -> DirectProviderBatch:
    """Drain tasks for direct provider mode.

    1. Reads all PENDING tasks and creates attempt rows (NULL worker_id, ASSIGNED).
    2. Reads all tasks in ASSIGNED/BUILDING/RUNNING state with NULL worker_id.
    3. Drains the global kill queue.

    Returns a batch for DirectTaskProvider.sync().
    """
    with self._db.transaction() as cur:
        now_ms = Timestamp.now().epoch_ms()

        # --- Pending tasks: create attempt rows ---
        pending_rows = cur.execute(
            "SELECT t.task_id, t.current_attempt_id, j.request_proto, j.num_tasks "
            "FROM tasks t JOIN jobs j ON j.job_id = t.job_id "
            "WHERE t.state = ? AND j.is_reservation_holder = 0",
            (cluster_pb2.TASK_STATE_PENDING,),
        ).fetchall()

        tasks_to_run: list[cluster_pb2.Worker.RunTaskRequest] = []
        for row in pending_rows:
            task_id_wire = str(row["task_id"])
            attempt_id = int(row["current_attempt_id"]) + 1
            job_req = cluster_pb2.Controller.LaunchJobRequest()
            job_req.ParseFromString(bytes(row["request_proto"]))

            cur.execute(
                "INSERT INTO task_attempts(task_id, attempt_id, worker_id, state, created_at_ms) "
                "VALUES (?, ?, NULL, ?, ?)",
                (task_id_wire, attempt_id, cluster_pb2.TASK_STATE_ASSIGNED, now_ms),
            )
            cur.execute(
                "UPDATE tasks SET state = ?, current_attempt_id = ?, "
                "started_at_ms = COALESCE(started_at_ms, ?) WHERE task_id = ?",
                (cluster_pb2.TASK_STATE_ASSIGNED, attempt_id, now_ms, task_id_wire),
            )

            run_req = cluster_pb2.Worker.RunTaskRequest(
                task_id=task_id_wire,
                num_tasks=int(row["num_tasks"]),
                entrypoint=job_req.entrypoint,
                environment=job_req.environment,
                bundle_id=job_req.bundle_id,
                resources=job_req.resources,
                ports=list(job_req.ports),
                attempt_id=attempt_id,
                constraints=list(job_req.constraints),
            )
            if job_req.timeout.milliseconds > 0:
                run_req.timeout.CopyFrom(job_req.timeout)
            tasks_to_run.append(run_req)

        # --- Running tasks: already dispatched ---
        running_rows = cur.execute(
            "SELECT t.task_id, t.current_attempt_id "
            "FROM tasks t "
            "JOIN task_attempts ta ON t.task_id = ta.task_id AND t.current_attempt_id = ta.attempt_id "
            "WHERE ta.worker_id IS NULL AND t.state IN (?, ?, ?) "
            "ORDER BY t.task_id ASC",
            (cluster_pb2.TASK_STATE_ASSIGNED, cluster_pb2.TASK_STATE_BUILDING, cluster_pb2.TASK_STATE_RUNNING),
        ).fetchall()
        running_tasks = [
            RunningTaskEntry(
                task_id=JobName.from_wire(str(row["task_id"])),
                attempt_id=int(row["current_attempt_id"]),
            )
            for row in running_rows
            if row not in [RunningTaskEntry(JobName.from_wire(str(r["task_id"])), int(r["current_attempt_id"])) for r in pending_rows]
            # Exclude tasks we just promoted (they'll be in tasks_to_run, not running_tasks)
        ]
        # Simpler: exclude task_ids that were just promoted
        promoted_ids = {str(r["task_id"]) for r in pending_rows}
        running_tasks = [
            RunningTaskEntry(JobName.from_wire(str(row["task_id"])), int(row["current_attempt_id"]))
            for row in running_rows
            if str(row["task_id"]) not in promoted_ids
        ]

        # --- Kill queue ---
        kill_rows = cur.execute(
            "SELECT task_id FROM dispatch_queue WHERE worker_id IS NULL AND kind = 'kill'",
        ).fetchall()
        tasks_to_kill = [str(row["task_id"]) for row in kill_rows]
        if kill_rows:
            cur.execute("DELETE FROM dispatch_queue WHERE worker_id IS NULL AND kind = 'kill'")

    return DirectProviderBatch(
        tasks_to_run=tasks_to_run,
        running_tasks=running_tasks,
        tasks_to_kill=tasks_to_kill,
    )
```

### New method: `apply_direct_provider_updates()`

```python
def apply_direct_provider_updates(self, updates: list[TaskUpdate]) -> TxResult:
    """Apply task state updates from a direct provider (no worker context).

    Like apply_task_updates but without worker health tracking or resource
    decommitment (direct providers don't use the worker resource model).
    """
    # Very similar to apply_task_updates but:
    # - No worker lookup or update
    # - No _decommit_worker_resources calls
    # - Same task state machine logic (FAILED retry, coscheduling cascade, etc.)
    ...
```

Also add `buffer_direct_kill(task_id: str)` for direct kills without a worker:
```python
def buffer_direct_kill(self, task_id: str) -> None:
    """Buffer a kill request for a directly-managed task."""
    with self._db.transaction() as cur:
        cur.execute(
            "INSERT INTO dispatch_queue(worker_id, kind, task_id, created_at_ms) VALUES (NULL, 'kill', ?, ?)",
            (task_id, Timestamp.now().epoch_ms()),
        )
```

---

## 4. Controller Changes

### `controller.py`

```python
# In Controller.__init__:
# The provider can be TaskProvider (WorkerProvider) or DirectTaskProvider (KubernetesProvider)
# Use isinstance to dispatch to the right sync path.
from iris.cluster.controller.direct_provider import DirectTaskProvider

if isinstance(provider, DirectTaskProvider):
    self._direct_provider = provider
    self._worker_provider = None
else:
    self._direct_provider = None
    self._worker_provider = provider  # same as self._provider

# In start():
if self._direct_provider:
    # Direct provider: no scheduler, no autoscaler
    self._provider_thread = self._threads.spawn(self._run_direct_provider_loop, ...)
else:
    # Worker provider: full scheduler + autoscaler + heartbeat
    self._scheduling_thread = ...
    self._autoscaler_thread = ...
    self._heartbeat_thread = ...
```

```python
def _run_direct_provider_loop(self, stop_event: threading.Event) -> None:
    limiter = RateLimiter(interval_seconds=self._config.heartbeat_interval.to_seconds())
    while not stop_event.is_set():
        stop_event.wait(timeout=limiter.time_until_next())
        if stop_event.is_set():
            break
        try:
            self._sync_direct_provider()
        except Exception:
            logger.exception("Direct provider loop iteration failed")

def _sync_direct_provider(self) -> None:
    assert self._direct_provider is not None
    with self._heartbeat_lock:
        batch = self._transitions.drain_for_direct_provider()
        if not batch.tasks_to_run and not batch.running_tasks and not batch.tasks_to_kill:
            return
        result = self._direct_provider.sync(batch)
        tx_result = self._transitions.apply_direct_provider_updates(result.updates)
        # Cache scheduling events for dashboard display
        self._provider_scheduling_events = result.scheduling_events
        self._provider_capacity = result.capacity
        if tx_result.tasks_to_kill:
            self.kill_tasks_on_workers(tx_result.tasks_to_kill)
```

### Kill routing for direct provider

When `kill_tasks_on_workers` is called, if the task has no worker (NULL worker_id in task_attempts), route to `buffer_direct_kill()` instead.

---

## 5. KubernetesProvider Rewrite

The rewritten `KubernetesProvider` implements `DirectTaskProvider`:

```python
# lib/iris/src/iris/cluster/controller/kubernetes_provider.py

@dataclass
class KubernetesProvider:
    """DirectTaskProvider that executes tasks as Kubernetes Pods.

    No worker daemons. Each sync cycle:
    1. Creates pods for tasks_to_run
    2. Deletes pods for tasks_to_kill
    3. Polls pod phase for running_tasks
    4. Fetches scheduling events for dashboard display

    Pod management code taken from runtime/kubernetes.py.
    """
    kubectl: Kubectl
    namespace: str
    default_image: str

    def sync(self, batch: DirectProviderBatch) -> DirectProviderSyncResult:
        for run_req in batch.tasks_to_run:
            self._apply_pod(run_req)
        for task_id in batch.tasks_to_kill:
            self._delete_pods_by_task_id(task_id)
        updates = self._poll_pods(batch.running_tasks)
        events = self._fetch_scheduling_events()
        capacity = self._query_node_resources()
        return DirectProviderSyncResult(updates=updates, scheduling_events=events, capacity=capacity)
```

Key: **No `execution_unit_id` field**. No worker registration.

---

## 6. Runtime Code Migration

`lib/iris/src/iris/cluster/runtime/kubernetes.py` contains pod management code
used by the worker-daemon-based runtime. The relevant pod management helpers
(`_sanitize_label_value`, `_build_gpu_resources`, pod manifest building) are
duplicated or referenced from `kubernetes_provider.py`.

After this change:
- `runtime/kubernetes.py` stays for the existing worker-pod runtime (used by
  `KubernetesRuntime` and worker daemons that run tasks inside pods)
- `kubernetes_provider.py` has its own pod management code for the direct
  provider (different pod spec: no init container, no bundle staging)
- Remove the import of `_sanitize_label_value` from `runtime/kubernetes.py`
  in `kubernetes_provider.py` — copy it directly or put in shared `k8s/util.py`

---

## 7. Dashboard Changes

### Controller service (`service.py`)

Add `GetProviderStatus` RPC:
```proto
message GetProviderStatusRequest {}
message GetProviderStatusResponse {
  // Non-empty when using a DirectTaskProvider (e.g. KubernetesProvider)
  bool has_direct_provider = 1;
  // Recent scheduling events from the provider
  repeated SchedulingEvent scheduling_events = 2;
  // Current cluster capacity
  ClusterCapacity capacity = 3;
}
message SchedulingEvent {
  string task_id = 1;
  int32 attempt_id = 2;
  string event_type = 3;
  string reason = 4;
  string message = 5;
  iris.time.Timestamp timestamp = 6;
}
message ClusterCapacity {
  int32 schedulable_nodes = 1;
  int64 total_cpu_millicores = 2;
  int64 available_cpu_millicores = 3;
  int64 total_memory_bytes = 4;
  int64 available_memory_bytes = 5;
}
```

Add to `ControllerProtocol`:
```python
@property
def has_direct_provider(self) -> bool: ...
@property
def provider_scheduling_events(self) -> list[SchedulingEvent]: ...
@property
def provider_capacity(self) -> ClusterCapacity | None: ...
```

### Dashboard routing

- `GetAutoscalerStatus`: return empty response when `has_direct_provider=True`
- `ListWorkers`: return empty when `has_direct_provider=True`
- `GetWorkerStatus`: return UNIMPLEMENTED when `has_direct_provider=True`
- `FetchLogs` for worker source: route to `direct_provider.fetch_live_logs()`
  when `has_direct_provider=True`

---

## 8. Proto Changes

### `cluster.proto`

Add to `Controller` service:
```proto
message GetProviderStatusRequest {}
message GetProviderStatusResponse {
  bool has_direct_provider = 1;
  repeated SchedulingEvent scheduling_events = 2;
  ClusterCapacity capacity = 3;
}
message SchedulingEvent {
  string task_id = 1;
  int32 attempt_id = 2;
  string event_type = 3;
  string reason = 4;
  string message = 5;
  iris.time.Timestamp timestamp = 6;
}
message ClusterCapacity {
  int32 schedulable_nodes = 1;
  int64 total_cpu_millicores = 2;
  int64 available_cpu_millicores = 3;
  int64 total_memory_bytes = 4;
  int64 available_memory_bytes = 5;
}

// Add to ControllerService:
rpc GetProviderStatus(Controller.GetProviderStatusRequest) returns (Controller.GetProviderStatusResponse);
```

### `config.proto`

The `KubernetesProviderConfig` message stays the same, but remove
`execution_unit_id` / any synthetic worker fields.

---

## 9. Tests to Write

1. `test_direct_provider.py`: Unit tests for `drain_for_direct_provider()` and
   `apply_direct_provider_updates()` covering PENDING→ASSIGNED→RUNNING→SUCCEEDED,
   failure retry, and kill routing.

2. `test_kubernetes_provider.py`: Rewrite existing tests to use new `DirectProviderBatch`
   interface — no `worker_id`, no `DispatchBatch`.

3. Integration test: Full task lifecycle using `KubernetesProvider` with mock kubectl,
   without any Worker row in DB.

---

## 10. What Does NOT Change

- `TaskProvider` protocol (used by `WorkerProvider`) — unchanged
- `WorkerProvider` implementation — unchanged
- `transitions.py` existing methods (`drain_dispatch`, `apply_heartbeat`, etc.) — unchanged
- All existing tests for worker-based path — unchanged
- `runtime/kubernetes.py` — unchanged (still used by worker daemon k8s runtime)
