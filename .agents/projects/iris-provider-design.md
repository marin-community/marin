# Iris Provider Abstraction Design

Design for introducing a `TaskProvider` protocol that decouples the controller from
the heartbeat-to-worker-daemon execution model, enabling direct Kubernetes Pod dispatch
(needed for CoreWeave bare-metal without worker daemons).

---

## 1. Problem

The Iris controller is hard-wired to send tasks via a heartbeat RPC to worker daemons.
Every task execution path passes through:

- `controller.py:1436` `_heartbeat_all_workers_inner()` — pulls `DispatchBatch` from the
  DB, fires `_do_heartbeat_rpc()` per worker, writes results back via
  `complete_heartbeat()` / `fail_heartbeat()`.
- `controller.py:1553` `_do_heartbeat_rpc()` — calls `WorkerStubFactory.get_stub(address)`
  and sends `HeartbeatRequest(tasks_to_run, tasks_to_kill, expected_tasks)`.
- `transitions.py:1308` `drain_dispatch()` — serialises the DB dispatch queue into a
  `DispatchBatch` keyed by `WorkerId + worker_address`.
- `db.py:668` `Worker` row — contains `address: str` (host:port), `metadata: WorkerMetadata`,
  `consecutive_failures`, `last_heartbeat`, committed resources.

The scheduler (`scheduler.py:59` `WorkerSnapshot`) and autoscaler
(`autoscaler.py` `TrackedWorker`) also assume workers are long-lived registered daemons.

The proxy RPCs in `service.py:1457` `fetch_logs`, `service.py:1345` `profile_task`,
`service.py:1574` `get_process_status` all hard-code the pattern:
`target="/system/worker/<worker_id>"` → HTTP to the worker's address.

**Consequence**: on CoreWeave CKS, where tasks run directly as Pods without a worker
daemon (c.f. `runtime/kubernetes.py`), there is no worker daemon to heartbeat. Iris
cannot be used for direct-Pod task dispatch today.

---

## 2. Proposed Solution

Introduce a `TaskProvider` Protocol that the controller delegates all task lifecycle
operations to. The protocol has two implementations:

| Implementation | Execution model |
|---|---|
| `WorkerProvider` | existing heartbeat-to-daemon (GCP, Manual, Local, CoreWeave-with-daemon) |
| `KubernetesProvider` | direct kubectl Pod lifecycle (CoreWeave bare-metal, GKE) |

The controller's heartbeat loop becomes a **provider sync loop**: it calls
`provider.sync()` instead of `_do_heartbeat_rpc()`. The sync loop applies results
using the same `complete_heartbeat` / `fail_heartbeat` state machine. This minimises
the state machine changes required.

**Why not restructure the scheduler?**
The scheduler's `WorkerSnapshot` Protocol (`scheduler.py:59`) is already clean — it
only requires `available_cpu_millicores`, `available_memory`, `available_gpus`,
`available_tpus`, `attributes`, `healthy`. A `KubernetesProvider` can expose
`ExecutionUnit` objects satisfying `WorkerSnapshot` without changing the scheduler.

**Why keep `Worker` in the DB for KubernetesProvider?**
The transitions layer (`transitions.py`) implements robust retry logic, resource
accounting, and log buffering via the `Worker` table. Rather than duplicating this,
`KubernetesProvider` registers a **synthetic worker** per execution unit (e.g. one per
k8s node pool or one "virtual worker" representing the whole cluster). The synthetic
worker has capacity derived from node availability and no `address`. The provider's
`sync()` method produces `HeartbeatApplyRequest` objects; the controller applies them
via `ControllerTransitions.apply_heartbeat()` (already exists at `transitions.py:1133`,
skipping `complete_heartbeat()` which takes a `HeartbeatResponse` proto).

This means `dispatch_batch.worker_address` becomes `str | None` — `None` for
KubernetesProvider execution units where there is no daemon to contact.

---

## 3. Provider Protocol

**New file:** `lib/iris/src/iris/cluster/controller/provider.py`

```python
from typing import Iterator, Protocol
from iris.cluster.controller.transitions import DispatchBatch, HeartbeatApplyRequest
from iris.cluster.types import WorkerId
from iris.rpc import cluster_pb2, logging_pb2


class TaskProvider(Protocol):
    """Abstraction over a task execution backend.

    The controller calls sync() in a loop (replacing _do_heartbeat_rpc). The provider
    is responsible for submitting/cancelling tasks and collecting their state. It returns
    HeartbeatApplyRequest batches — one per execution unit — which the controller applies
    via ControllerTransitions.apply_heartbeat() directly (not complete_heartbeat(), which
    takes a HeartbeatResponse proto).

    Log fetching for live tasks is provider-specific. Completed task logs are always
    available from the controller's local LogStore (written via HeartbeatApplyRequest
    log_entries), so the protocol only needs to cover live log streaming.
    """

    def sync(
        self,
        batches: list[DispatchBatch],
    ) -> list[tuple[DispatchBatch, HeartbeatApplyRequest | None, str | None]]:
        """Sync task state with the execution backend.

        Args:
            batches: One DispatchBatch per active execution unit, drained from the DB.
                     tasks_to_run and tasks_to_kill come from the DB dispatch queue.
                     running_tasks is the reconciliation set.

        Returns:
            For each batch: (batch, apply_request | None, error_str | None).
            apply_request is None on communication failure (caller uses fail_heartbeat).
        """
        ...

    def fetch_live_logs(
        self,
        worker_id: WorkerId,
        task_id: str,
        attempt_id: int,
        cursor: int,
        max_lines: int,
    ) -> tuple[list[logging_pb2.LogEntry], int]:
        """Fetch live logs for a running task from the execution backend.

        Returns (entries, next_cursor). next_cursor == cursor means no new logs.
        Raises ProviderError if the backend is unreachable (caller falls back to
        controller's local LogStore for completed tasks).

        **Cursor semantics**: For `WorkerProvider`, cursor is a line-number offset
        (integer byte/line count as used by worker daemon). For `KubernetesProvider`,
        `kubectl logs` supports `--since` (time-based) not line-number offsets; cursor
        should be a Unix timestamp (float seconds) or RFC3339 string. Implementations
        must document their cursor format; callers must not mix providers. Consider
        using byte-offset cursors (`kubectl logs --limit-bytes`) as a more portable
        alternative.
        """
        ...

    def fetch_process_logs(
        self,
        worker_id: WorkerId,
        cursor: int,
        max_lines: int,
    ) -> tuple[list[logging_pb2.LogEntry], int]:
        """Fetch execution-unit process logs (daemon or pod logs).

        Returns (entries, next_cursor). Not all providers support this.
        Raises ProviderUnsupportedError if not applicable.
        """
        ...

    def on_worker_failed(self, worker_id: WorkerId, address: str | None) -> None:
        """Called by the controller when a worker is definitively failed.

        Providers should evict any cached connection state for this worker.
        WorkerProvider evicts the stub keyed by `address` from stub_factory.
        KubernetesProvider is a no-op (no stub cache).
        """
        ...
```

**Autoscaling** is intentionally not part of `TaskProvider`. The existing `Autoscaler` /
`Platform` protocol manages slice lifecycle for `WorkerProvider`. `KubernetesProvider`
does not need autoscaling (k8s handles node scheduling). The controller's autoscaler
loop is a no-op when `autoscaler is None` — this remains true for `KubernetesProvider`.

---

## 4. WorkerProvider Implementation

**New file:** `lib/iris/src/iris/cluster/controller/worker_provider.py`

`WorkerProvider` wraps the existing `WorkerStubFactory` + `_do_heartbeat_rpc` logic
behind the `TaskProvider` protocol. The implementation is essentially a copy of
`_heartbeat_all_workers_inner()` lifted into the provider.

```python
@dataclass
class WorkerProvider:
    """TaskProvider backed by worker daemons via heartbeat RPC.

    Drop-in replacement for the direct _do_heartbeat_rpc path. Concurrency
    model unchanged: ThreadPoolExecutor, one future per dispatch batch.
    """
    stub_factory: WorkerStubFactory
    parallelism: int = 32

    def sync(
        self, batches: list[DispatchBatch]
    ) -> list[tuple[DispatchBatch, HeartbeatApplyRequest | None, str | None]]:
        results = []
        with ThreadPoolExecutor(max_workers=min(self.parallelism, len(batches))) as pool:
            futures = {pool.submit(self._heartbeat_one, b): b for b in batches}
            for fut, batch in futures.items():
                try:
                    apply_req = fut.result()
                    results.append((batch, apply_req, None))
                except Exception as e:
                    results.append((batch, None, str(e)))
        return results

    def _heartbeat_one(self, batch: DispatchBatch) -> HeartbeatApplyRequest:
        stub = self.stub_factory.get_stub(batch.worker_address)
        request = HeartbeatRequest(
            tasks_to_run=batch.tasks_to_run,
            tasks_to_kill=batch.tasks_to_kill,
            expected_tasks=[...],  # from batch.running_tasks
        )
        response = stub.heartbeat(request)
        # worker_healthy==False means the daemon detected an unrecoverable state
        # (e.g. OOM). Raise so sync() returns (batch, None, error), causing the
        # controller to call fail_heartbeat() instead of apply_heartbeat().
        if not response.worker_healthy:
            raise WorkerUnhealthyError(f"worker {batch.worker_id} reported unhealthy")
        return _apply_request_from_response(batch.worker_id, response)

    def fetch_live_logs(self, worker_id, task_id, attempt_id, cursor, max_lines):
        stub = self.stub_factory.get_stub(_address_for_worker(worker_id))
        resp = stub.fetch_logs(FetchLogsRequest(source=task_log_key(...), cursor=cursor, max_lines=max_lines))
        return resp.entries, resp.next_cursor

    def fetch_process_logs(self, worker_id, cursor, max_lines):
        stub = self.stub_factory.get_stub(_address_for_worker(worker_id))
        resp = stub.fetch_logs(FetchLogsRequest(source=PROCESS_LOG_KEY, cursor=cursor, max_lines=max_lines))
        return resp.entries, resp.next_cursor
```

**Required changes to make existing code conform:**

1. `controller.py:1553` `_do_heartbeat_rpc()` → deleted; its body moves into `WorkerProvider._heartbeat_one()`.
2. `controller.py:1436` `_heartbeat_all_workers_inner()` → replaced by the loop over `provider.sync()` results,
   calling `apply_heartbeat()` (success) / `fail_heartbeat()` (failure); `_reap_stale_workers()` called before sync.
3. `controller.py:708` `Controller.__init__` adds `provider: TaskProvider` parameter;
   `worker_stub_factory` is removed (it is an implementation detail of `WorkerProvider`).
4. `service.py:1457` `fetch_logs` and `service.py:1345` `profile_task`: proxy via
   `self._provider.fetch_live_logs()` instead of directly resolving worker address.
   The `/system/worker/<id>` target convention can be preserved at the RPC surface.

**What does NOT change:**
- `ControllerTransitions.drain_dispatch()` — no change; still produces `DispatchBatch`.
- `ControllerTransitions.apply_heartbeat()` — called directly by the sync loop with `HeartbeatApplyRequest`.
  `complete_heartbeat()` still exists but takes a `HeartbeatResponse` proto and is not called from the
  sync loop (it was the old RPC-facing entry point that performed the proto→apply conversion internally).
- `ControllerTransitions.fail_heartbeat()` / `fail_workers_by_ids()` — no change.
- `DispatchBatch.worker_address` — kept as `str | None`; `WorkerProvider` always has a non-None address.
- The autoscaler: `_autoscaler.notify_worker_failed()` still called from the same place.

---

## 5. KubernetesProvider Implementation

**New file:** `lib/iris/src/iris/cluster/controller/kubernetes_provider.py`

For each pending `RunTaskRequest` in a batch, `KubernetesProvider` applies a Pod spec.
Pod status is polled each sync cycle. Log streaming uses `kubectl logs --follow`.

```python
@dataclass
class KubernetesProvider:
    """TaskProvider that executes tasks as Kubernetes Pods directly.

    No worker daemon. Each execution unit is a synthetic entry in the WORKERS table
    representing the k8s cluster (or a node pool). Capacity comes from node allocatable
    resources queried via kubectl.

    Pod naming: iris-{task_id_sanitized}-{attempt_id}
    Namespace: configurable (default: "iris")
    """
    kubectl: Kubectl
    namespace: str
    # execution_unit_id registered in WORKERS — one per KubernetesProvider instance
    execution_unit_id: WorkerId

    def sync(
        self, batches: list[DispatchBatch]
    ) -> list[tuple[DispatchBatch, HeartbeatApplyRequest | None, str | None]]:
        assert len(batches) <= 1, "KubernetesProvider has one execution unit"
        if not batches:
            return []
        batch = batches[0]
        try:
            # Apply new tasks
            for run_req in batch.tasks_to_run:
                self._apply_pod(run_req)
            # Kill cancelled tasks
            for task_id in batch.tasks_to_kill:
                self._delete_pod(task_id)
            # Query state of all expected tasks
            updates = self._poll_pods(batch.running_tasks)
            resource_snapshot = self._query_node_resources()
            apply_req = HeartbeatApplyRequest(
                worker_id=batch.worker_id,
                worker_resource_snapshot=resource_snapshot,
                updates=updates,
            )
            return [(batch, apply_req, None)]
        except Exception as e:
            return [(batch, None, str(e))]
```

**Task submission** (`_apply_pod`):

```python
def _apply_pod(self, run_req: cluster_pb2.Worker.RunTaskRequest) -> None:
    pod_spec = _build_pod_spec(run_req, namespace=self.namespace)
    self.kubectl.apply(pod_spec)
```

Pod spec follows the same pattern as `KubernetesRuntime.run()` in `runtime/kubernetes.py:~120`.
Key fields: `metadata.name = _pod_name(task_id, attempt_id)`, `spec.restartPolicy = "Never"`,
`spec.containers[0].resources` from `run_req.resources`.
Labels: `iris/task-id`, `iris/attempt-id` for field selectors.

**State polling** (`_poll_pods`):

```python
def _poll_pods(self, running: list[RunningTaskEntry]) -> list[TaskUpdate]:
    updates = []
    pod_statuses = self.kubectl.get_pods(
        namespace=self.namespace,
        label_selector="iris/managed=true",
    )
    by_name = {p.name: p for p in pod_statuses}
    for entry in running:
        pod_name = _pod_name(entry.task_id, entry.attempt_id)
        pod = by_name.get(pod_name)
        if pod is None:
            updates.append(TaskUpdate(task_id=entry.task_id, attempt_id=entry.attempt_id,
                                      new_state=TASK_STATE_WORKER_FAILED, error="Pod not found"))
        else:
            updates.append(_task_update_from_pod(entry, pod))
    return updates
```

Pod phase → task state mapping:
- `Pending` → `TASK_STATE_RUNNING` (not yet started; no distinct "pending" state in iris)
- `Running` → `TASK_STATE_RUNNING`
- `Succeeded` → `TASK_STATE_COMPLETE`
- `Failed` / `Unknown` → `TASK_STATE_FAILED` (with `exit_code` from `containerStatuses`)

**Log access** (`fetch_live_logs`):

```python
def fetch_live_logs(self, worker_id, task_id, attempt_id, cursor, max_lines):
    pod_name = _pod_name(task_id, attempt_id)
    lines = self.kubectl.logs(
        pod=pod_name, namespace=self.namespace,
        since_line=cursor, max_lines=max_lines, timestamps=True,
    )
    entries = [_log_entry_from_line(l) for l in lines]
    return entries, cursor + len(entries)
```

**Process logs** (`fetch_process_logs`): raises `ProviderUnsupportedError` — there is
no daemon process to inspect. The controller dashboard should hide the "Worker process"
tab for `KubernetesProvider` execution units.

**Node capacity** (`_query_node_resources`): one `kubectl get nodes -o json` call per
sync cycle, summing `allocatable` across schedulable nodes. Updates
`WorkerResourceSnapshot` so the scheduler sees current available capacity.

**No bundle staging**: `KubernetesProvider` puts the image and command directly in the
Pod spec from `run_req.image` / `run_req.command`. Bundle sync is a no-op (tasks use
pre-built images). This means `TASK_STATE_BUILDING` is skipped — tasks go directly
from `TASK_STATE_ASSIGNED` to `TASK_STATE_RUNNING`.

---

## 6. Controller Changes

### `controller.py`

| Current | Change |
|---|---|
| `__init__(worker_stub_factory: WorkerStubFactory)` | `__init__(provider: TaskProvider)` |
| `_heartbeat_all_workers_inner()` (lines 1436–1553) | Replace with `_sync_all_execution_units()` calling `provider.sync()` |
| `_do_heartbeat_rpc()` (lines 1553–1583) | Delete; logic moves to `WorkerProvider._heartbeat_one()` |
| `self.stub_factory` field | `self._provider: TaskProvider` |
| `_capture_one_profile()` (line 1045) — calls `stub.profile_task()` directly | Delegate to provider or remove for K8s (profile via kubectl exec, optional) |
| `_run_heartbeat_loop` name | Rename to `_run_provider_loop` (cosmetic but clarifying) |

The result of `provider.sync()` is a list of `(batch, apply_req | None, error_str | None)`.
The existing `apply_heartbeat()` / `fail_heartbeat()` / autoscaler notification logic
is unchanged and called in the same phase-3 loop.

```python
def _sync_all_execution_units(self) -> None:
    with slow_log(logger, "provider sync phase 1 (snapshot)", threshold_ms=100):
        workers = healthy_active_workers_with_attributes(self._db)
        batches = [batch for w in workers if (batch := self._transitions.drain_dispatch(w.worker_id))]
    if not batches:
        return

    # Preserve existing behaviour: reap stale workers before syncing.
    # WorkerProvider benefits from this; KubernetesProvider treats it as a no-op
    # (pods self-clean; there are no stale worker daemon registrations to reap).
    self._reap_stale_workers()

    results = self._provider.sync(batches)

    fail_count = 0
    for batch, apply_req, error in results:
        if apply_req is not None:
            # Call apply_heartbeat() directly — complete_heartbeat() takes a
            # HeartbeatResponse proto and would do an extra conversion step.
            result = self._transitions.apply_heartbeat(apply_req)
            # ... identical to current phase 3
        else:
            action = self._transitions.fail_heartbeat(batch, error or "unknown")
            if action == HeartbeatAction.WORKER_FAILED:
                # ... autoscaler notification (WorkerProvider) or log+retry (K8sProvider)
```

**One subtle change**: `fail_heartbeat()` currently evicts `stub_factory.evict(address)`.
`stub_factory` is keyed by **address** (not `worker_id`), so the protocol must carry
the address. The eviction call should be moved into `WorkerProvider` (called from
`_sync_all_execution_units` when `action == WORKER_FAILED` via
`provider.on_worker_failed(worker_id, address=batch.worker_address)`).

Add `on_worker_failed(worker_id: WorkerId, address: str | None) -> None` to the `TaskProvider` Protocol:
- `WorkerProvider`: evicts the stub by `address` from the `stub_factory` cache (address-keyed).
- `KubernetesProvider`: no-op (no stub cache; next sync retries pod queries).

### `scheduler.py`

`WorkerSnapshot` Protocol (`scheduler.py:59`) is already clean. No changes needed.
The scheduler receives `Worker` rows (which satisfy `WorkerSnapshot` via duck typing)
for both providers — `KubernetesProvider` registers a synthetic `Worker` row with
`address=""` and `metadata` derived from node capacity queries.

### `transitions.py`

`DispatchBatch.worker_address` (`transitions.py:185`) changes type from `str` to
`str | None`. `drain_dispatch()` sets it to `None` for synthetic execution units
(where `Worker.address == ""`). This is the only change needed — nothing else in
`transitions.py` reads `worker_address`.

### `service.py`

The proxy RPCs (`fetch_logs`, `profile_task`, `get_process_status`) currently resolve
the worker address then call the worker stub directly. With a `TaskProvider`, they
delegate to the provider:

```python
# Old (service.py:1468–1480):
worker_id = _parse_worker_target(request.source)
stub = self._resolve_worker_stub(worker_id)
return stub.fetch_logs(FetchLogsRequest(source="/system/process", ...))

# New:
worker_id = _parse_worker_target(request.source)
entries, next_cursor = self._provider.fetch_process_logs(
    WorkerId(worker_id), request.cursor, request.max_lines
)
# Build FetchLogsResponse from entries
```

For task log fetching (non-`/system/worker/` source), the existing path — look up
completed logs in the controller's `_log_store`, then fall through to live provider
logs — remains correct but the "live" branch calls `provider.fetch_live_logs()`.

`_WORKER_TARGET_PREFIX` and `_parse_worker_target()` remain; the target format
`/system/worker/<id>` is still the right convention for routing log/status RPCs.

---

## 7. Dashboard Changes

### Controller Dashboard (`controller/dashboard.py`)

- `GET /worker/{worker_id:path}` → rename to `GET /execution-unit/{id:path}` or keep
  as `/worker/` with the understanding that "worker" means "execution unit" in the URL.
  Recommendation: keep the URL as-is (avoid breaking existing bookmarks) but change
  the page title and labels.
- "Worker list" table: add a `provider_type` column (`worker` vs `kubernetes`). Show
  `address` only if non-empty. Hide "consecutive failures / last heartbeat" for
  `KubernetesProvider` execution units (these are meaningless; pod failures are per-task).

### Controller Service RPC

`ListWorkers` response already returns `worker_id`, `address`, `metadata`, `attributes`,
`healthy`. For `KubernetesProvider` units, `address` is `""` and `healthy` reflects
whether the node pool is reachable. No proto changes needed for the basic case.

For the dashboard "Worker detail" page, the "Process status" tab calls `GetProcessStatus`
with `target=/system/worker/<id>`. `KubernetesProvider.fetch_process_logs()` raises
`ProviderUnsupportedError`; the controller service should return a gRPC `UNIMPLEMENTED`
status. The dashboard should hide the tab when `UNIMPLEMENTED` is returned.

### Worker Dashboard (`worker/dashboard.py`)

Unchanged — `WorkerProvider` still uses worker daemons that serve `WorkerService` at
their own HTTP endpoint. `KubernetesProvider` has no worker daemon and no per-unit
dashboard; the controller dashboard is the only UI.

---

## 8. Config Changes

### Proto (`rpc/config.proto`)

Add a `KubernetesProviderConfig` message and a `provider` oneof to `IrisClusterConfig`:

```protobuf
message KubernetesProviderConfig {
  string namespace = 1;                // default: "iris"
  string kubeconfig = 2;               // path or "" for in-cluster
  string node_selector_json = 3;       // JSON map for pod nodeSelector
  string image_pull_policy = 4;        // default: "IfNotPresent"
  repeated string image_pull_secrets = 5;
}

// Add to IrisClusterConfig:
message IrisClusterConfig {
  // ... existing fields ...
  oneof provider {
    // Worker provider (existing default): uses Platform + WorkerConfig
    // Selected implicitly when platform is set.
    bool worker_provider = 50;         // explicit opt-in (no-op for now)
    KubernetesProviderConfig kubernetes_provider = 51;
  }
}
```

**Selection logic in `cluster/config.py`:**

```python
def make_provider(config: IrisClusterConfig, platform: Platform | None) -> TaskProvider:
    if config.HasField("kubernetes_provider"):
        kp = config.kubernetes_provider
        return KubernetesProvider(
            kubectl=Kubectl(namespace=kp.namespace, kubeconfig=kp.kubeconfig or None),
            namespace=kp.namespace or "iris",
            execution_unit_id=WorkerId("k8s-cluster"),
        )
    # Default: worker provider using the platform's stub factory
    assert platform is not None
    return WorkerProvider(stub_factory=RpcWorkerStubFactory())
```

### `ControllerConfig` dataclass (`controller.py:649`)

Remove `worker_stub_factory` as a constructor parameter. Add it back only inside
`make_provider()` / `WorkerProvider`. The `Controller.__init__` signature becomes:

```python
def __init__(
    self,
    config: ControllerConfig,
    provider: TaskProvider,
    autoscaler: Autoscaler | None = None,
    ...
)
```

`worker_access_address` (currently in `ControllerConfig` at line 680, never read) should
be deleted.

---

## 9. Migration Path (Minimal for CoreWeave)

The minimal first step to unblock CoreWeave bare-metal (direct Pod dispatch, no worker daemon):

**Step 1 — Introduce `TaskProvider` protocol and `WorkerProvider`** _(~200 lines changed)_
- Create `provider.py` with the `TaskProvider` Protocol and `ProviderUnsupportedError`.
- Create `worker_provider.py` with `WorkerProvider` wrapping `WorkerStubFactory`.
- Remove `_do_heartbeat_rpc()` from `controller.py`; replace `_heartbeat_all_workers_inner()`
  with `_sync_all_execution_units()` calling `provider.sync()`.
- Update `Controller.__init__` signature.
- Update `ControllerServiceImpl` to use `provider.fetch_live_logs()` and
  `provider.fetch_process_logs()` in `fetch_logs()` and `get_process_status()`.
- Change `DispatchBatch.worker_address: str | None`.
- All existing tests still pass (WorkerProvider is behaviorally identical).

**Step 2 — Add `KubernetesProvider`** _(new file, ~300 lines)_
- `kubernetes_provider.py`: pod apply, poll, log fetch using existing `Kubectl` wrapper.
- Register synthetic `Worker` row on startup (one per k8s cluster).
- Add `KubernetesProviderConfig` proto message + `make_provider()` factory.

**Step 3 — Dashboard updates** _(frontend + service changes, ~50 lines)_
- Hide worker address, consecutive failures for K8s execution units.
- Return `UNIMPLEMENTED` for `GetProcessStatus` on K8s units.

Steps 1 and 2 are independent enough for parallel implementation once Step 1's
interfaces are finalised. Step 3 depends on Step 2 being testable.

---

## 10. Risks and Open Questions

**Risk: BUILDING state skip for KubernetesProvider**
The task state machine goes `ASSIGNED → BUILDING → RUNNING → COMPLETE`. `BUILDING` is
the bundle-prep phase (uv sync). For direct-Pod dispatch, there is no bundle prep —
pods use pre-built images. The provider's `sync()` needs to emit `TASK_STATE_RUNNING`
immediately when a pod is `Pending` (or skip `BUILDING` entirely). Check
`transitions.py` for guards that require `BUILDING` → `RUNNING` sequencing.

**Risk: Log completeness for short-lived pods**
`KubernetesRuntime.logs()` already handles `--previous` for completed pods
(`runtime/kubernetes.py`). `KubernetesProvider.fetch_live_logs()` must do the same —
after pod termination, fall back to `kubectl logs --previous`. Current worker heartbeat
model drains logs into the controller's LogStore during execution; with direct Pods,
the controller's LogStore won't have the logs unless they are fetched post-completion.
Consider: on pod completion, eagerly fetch and store all logs into `LogStore` during
the sync cycle that detects `Succeeded`/`Failed`.

**Risk: Crash-unsafe log loss (streaming gap)**
Worker daemons stream log entries into the controller's LogStore via each heartbeat
response. `KubernetesProvider` has no equivalent unless it actively fetches and buffers
logs on each sync cycle — not just at pod completion. If the controller crashes between
pod completion and log fetch, all logs for that run are lost. Mitigation: on every sync
cycle, for each running pod, fetch new log lines since the last cursor and write them
into `LogStore` via `apply_heartbeat`'s `log_entries` field. This mirrors what worker
daemons do today and ensures logs survive a controller restart.

**Risk: Synthetic worker re-registration on controller restart**
`WorkerProvider` relies on workers re-registering via `RegisterRequest` after a
controller restart (their `Worker` rows are marked `active=false`). `KubernetesProvider`
must re-register its synthetic `Worker` row on startup. Pods that were running before
the restart remain running; the first `sync()` call will re-discover them via the
`running_tasks` reconciliation set. The reconciliation in `complete_heartbeat()` marks
tasks as `WORKER_FAILED` if they appear in `expected_tasks` but not in the response —
so the first sync must include all running pods in the `updates` list to prevent
spurious failures.

**Open question: Multi-node-pool scheduling**
Should `KubernetesProvider` register one synthetic `Worker` per node pool, or one for
the whole cluster? One-per-pool allows the scheduler's constraint matching
(e.g. `gpu_type=A100`) to work correctly. But querying per-pool capacity requires
node label selectors. The design above uses one provider per instance; the config can
have multiple `ScaleGroupConfig` entries each pointing to a separate `KubernetesProvider`
with a different `node_selector_json`. This matches the existing multi-scale-group model
and requires no scheduler changes.

**Open question: CoreWeave worker daemon mode**
CoreWeave currently runs Iris workers as Pods (`platform/coreweave.py:create_slice()`).
With `KubernetesProvider`, these worker Pods are no longer needed. The
`CoreweavePlatform` would be replaced by `KubernetesProvider` in the config.
Existing CoreWeave deployments using worker Pods must migrate to
`kubernetes_provider` config. No backward-compat shim — update configs directly.
