## Overview

`iris.cluster.controller` is the Iris cluster control plane: it accepts job submissions, schedules tasks onto workers, drives VM autoscaling, and persists all state in a WAL-mode SQLite database with periodic GCS checkpoints. The data flow is: client → `ControllerServiceImpl` (RPC layer) → `ControllerTransitions` (atomic DB mutations) → `Scheduler` (task-to-worker matching) → `WorkerProvider` (parallel heartbeat dispatch) → `Autoscaler` / `ScalingGroup` (VM lifecycle). The `Controller` class owns three background threads (scheduling, provider, autoscaler) and coordinates them.

Default parameter values encode operational experience: `heartbeat_failure_threshold` tolerates transient network blips before evicting a worker; `unresolvable_timeout` absorbs slow cloud provisioning before treating a slice as failed; `DEFAULT_JWT_TTL_SECONDS` balances token freshness against login friction; checkpoint interval defaults to hourly as a recovery-cost/write-cost tradeoff. Change these only with cluster-specific knowledge — they are not arbitrary.

Entry point: `run_controller_serve(cluster_config)` → blocks. For programmatic use: construct `Controller`, call `.start()`, submit via `.launch_job(request)`, poll with `.get_job_status(job_id)`.

## API

```python
from iris.cluster.controller.main import run_controller_serve

def run_controller_serve(
    cluster_config: config_pb2.IrisClusterConfig,
    *,
    host: str = "0.0.0.0",
    port: int = 10000,
    checkpoint_path: str | None = None,
    checkpoint_interval: float | None = None,
    dry_run: bool = False,
) -> None: ...
```
Daemon entry point; blocks until SIGTERM. `lib/iris/src/iris/cluster/controller/main.py:1`

---

```python
from iris.cluster.controller.controller import Controller

class Controller:
    def __init__(
        self,
        config: ControllerConfig,
        provider: TaskProvider | K8sTaskProvider,
        autoscaler: Autoscaler | None = None,
        threads: ThreadContainer | None = None,
        db: ControllerDB | None = None,
    ) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def launch_job(self, request: LaunchJobRequest) -> LaunchJobResponse: ...
    def get_job_status(self, job_id: str) -> GetJobStatusResponse: ...
    def terminate_job(self, job_id: str) -> Empty: ...
    def begin_checkpoint(self) -> tuple[str, CheckpointResult]: ...
    def wake(self) -> None: ...
```
Unified control plane; `launch_job`/`get_job_status` are the primary call sites. `lib/iris/src/iris/cluster/controller/controller.py:1`

---

```python
from iris.cluster.controller.autoscaler import route_demand, Autoscaler

def route_demand(
    groups: list[ScalingGroup],
    demand_entries: list[DemandEntry],
    timestamp: Timestamp | None = None,
) -> RoutingDecision: ...
```
Two-phase priority waterfall; phase 1 uses committed budgets, phase 2 uses full capacity. `lib/iris/src/iris/cluster/controller/autoscaler.py:1`

---

```python
from iris.cluster.controller.autoscaler import Autoscaler

class Autoscaler:
    def update(self, demand_entries: list[DemandEntry], timestamp: Timestamp | None) -> list[ScalingDecision]: ...
    def notify_worker_failed(self, worker_id: str) -> list[str]: ...
    def restore_from_db(self, db: ControllerDB, platform: WorkerInfraProvider) -> None: ...
    def shutdown(self) -> None: ...
```
Orchestrates evaluate/execute/refresh; call `restore_from_db` before starting loop to avoid duplicate VM provisioning. `lib/iris/src/iris/cluster/controller/autoscaler.py:1`

---

```python
from iris.cluster.controller.db import ControllerDB

class ControllerDB:
    def __init__(self, db_dir: Path) -> None: ...
    def transaction(self) -> ContextManager[TransactionCursor]: ...
    def read_snapshot(self) -> ContextManager[QuerySnapshot]: ...
    def snapshot(self) -> QuerySnapshot: ...
    def backup_to(self, destination: Path) -> None: ...
```
WAL SQLite with 8-connection read pool. `read_snapshot()` is lock-free; `snapshot()` acquires write lock. `lib/iris/src/iris/cluster/controller/db.py:1`

---

```python
from iris.cluster.controller.transitions import ControllerTransitions

class ControllerTransitions:
    def submit_job(self, job_id, request, ts) -> SubmitJobResult: ...
    def queue_assignments(self, assignments: list[Assignment]) -> AssignmentResult: ...
    def apply_heartbeat(self, req: HeartbeatApplyRequest) -> HeartbeatApplyResult: ...
    def drain_dispatch_all(self) -> list[DispatchBatch]: ...
    def fail_heartbeat_for_worker(self, worker_id, error, snapshot, *, force_remove: bool) -> HeartbeatFailureResult: ...
    def prune_old_data(self, *, job_retention, worker_retention, log_retention, txn_action_retention, profile_retention, stop_event, pause_between_s) -> PruneResult: ...
```
Sole write path for all state machine transitions. `lib/iris/src/iris/cluster/controller/transitions.py:1`

---

```python
from iris.cluster.controller.checkpoint import write_checkpoint, download_checkpoint_to_local

def write_checkpoint(db: ControllerDB, remote_state_dir: str) -> tuple[str, CheckpointResult]: ...
def download_checkpoint_to_local(remote_state_dir: str, local_db_dir: Path, checkpoint_dir: str | None = None) -> bool: ...
```
`write_checkpoint` = backup + compress + upload + prune. Returns `False` if no checkpoint found. `lib/iris/src/iris/cluster/controller/checkpoint.py:1`

---

```python
from iris.cluster.controller.scheduler import Scheduler

class Scheduler:
    def __init__(self, max_building_tasks_per_worker: int = DEFAULT) -> None: ...
    def find_assignments(self, context: SchedulingContext) -> SchedulingResult: ...
    def create_scheduling_context(self, workers, building_counts, pending_tasks, jobs, max_building_tasks, max_assignments_per_worker) -> SchedulingContext: ...
```
Pure-functional; coscheduled jobs assigned atomically before non-coscheduled. `lib/iris/src/iris/cluster/controller/scheduler.py:1`

---

Omitted: `ActorProxy`, `auth.*`, `dashboard.*`, `pending_diagnostics`, `query`, `schema` row types, `service` helpers, `vm_lifecycle`, `worker_provider` — useful internals but not primary call sites.

## Gotchas

- **`remote_state_dir` is mandatory**: `run_controller_serve` and `Controller` raise immediately if `storage.remote_state_dir` is unset — there is no in-memory-only mode for production.
- **Write path is `ControllerTransitions` only**: after any transaction that returns `TxResult`, the *caller* must issue kill RPCs for every `task_id` in `TxResult.tasks_to_kill` — the DB commit does not send them.
- **`backup_databases` requires the write lock; `upload_checkpoint` does not**: calling them in the wrong order (or holding no lock during backup) can produce an inconsistent snapshot.
- **`read_snapshot()` lags writes**: pooled read connections are not coordinated with the write lock — never use `read_snapshot()` to verify something you just wrote inside a `transaction()`.
- **`scale_up()` does not self-register**: callers must bracket with `begin_scale_up` / `complete_scale_up` (or `cancel_scale_up`); forgetting leaves the group in a permanently inflated requesting count.
