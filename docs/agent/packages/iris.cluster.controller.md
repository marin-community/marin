## Overview

`iris.cluster.controller` is the Iris cluster control plane: it schedules tasks onto workers, heartbeats running tasks, autoscales VM slices, and persists state to a WAL-mode SQLite DB with periodic GCS checkpoints. The data flow is: `ControllerDB` (schema + queries) → `ControllerTransitions` (atomic state-machine mutations) → `Controller` (orchestrates scheduling/heartbeat loops) → `ControllerServiceImpl` (gRPC RPC surface). The `Autoscaler` runs a parallel loop: `compute_demand_entries` → `route_demand` → `ScalingGroup.scale_up/down`.

Default values encode production operating experience: `heartbeat_failure_threshold=3` before marking a worker dead; `DEFAULT_UNRESOLVABLE_TIMEOUT` before giving up on unrouteable tasks; `DEFAULT_JWT_TTL_SECONDS` balances security and re-login friction. The `max_building_tasks_per_worker` cap prevents thundering-herd task starts on fresh workers. Do not change these without understanding their backoff/cooldown interactions in `ScalingGroup`.

## API

```python
from iris.cluster.controller.controller import Controller, ControllerConfig, compute_demand_entries

@dataclass
class ControllerConfig:
    remote_state_dir: str          # required — raises ValueError if empty
    port: int = 0                  # 0 = OS auto-assign; read actual via Controller.port
    ...

class Controller:
    def __init__(self, config: ControllerConfig, provider: TaskProvider,
                 autoscaler: Autoscaler | None = None,
                 threads: ThreadContainer | None = None,
                 db: ControllerDB | None = None): ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def launch_job(self, request: LaunchJobRequest) -> LaunchJobResponse: ...
    def begin_checkpoint(self) -> tuple[str, CheckpointResult]: ...
```
Top-level controller; call `start()` then interact via RPC or direct methods. `lib/iris/src/iris/cluster/controller/controller.py:1`

```python
from iris.cluster.controller.autoscaler import Autoscaler, route_demand, DemandEntry

class Autoscaler:
    def update(self, demand_entries: list[DemandEntry],
               timestamp: Timestamp | None = None) -> list[ScalingDecision]: ...
    def run_once(self, demand_entries, worker_status_map, timestamp) -> list[ScalingDecision]: ...

def route_demand(groups: list[ScalingGroup], demand_entries: list[DemandEntry],
                 timestamp: Timestamp | None = None) -> RoutingDecision: ...
```
Two-phase priority routing; `update` = evaluate + execute combined. `lib/iris/src/iris/cluster/controller/autoscaler.py:1`

```python
from iris.cluster.controller.db import ControllerDB, QuerySnapshot

class ControllerDB:
    def __init__(self, db_dir: Path): ...
    def transaction(self) -> ContextManager[TransactionCursor]: ...
    def read_snapshot(self) -> ContextManager[QuerySnapshot]: ...
```
Use `read_snapshot()` for concurrent reads (pooled WAL); `transaction()` for writes. `lib/iris/src/iris/cluster/controller/db.py:1`

```python
from iris.cluster.controller.transitions import ControllerTransitions, HeartbeatApplyRequest

class ControllerTransitions:
    def __init__(self, db: ControllerDB, log_store: LogStore,
                 heartbeat_failure_threshold: int = HEARTBEAT_FAILURE_THRESHOLD): ...
    def submit_job(self, job_id: JobName, request: LaunchJobRequest, ts: Timestamp) -> SubmitJobResult: ...
    def apply_heartbeat(self, req: HeartbeatApplyRequest) -> HeartbeatApplyResult: ...
    def drain_dispatch(self, worker_id: WorkerId) -> DispatchBatch | None: ...
    def queue_assignments(self, assignments: list[Assignment]) -> AssignmentResult: ...
```
All SQLite state-machine mutations. `lib/iris/src/iris/cluster/controller/transitions.py:1`

```python
from iris.cluster.controller.checkpoint import write_checkpoint, download_checkpoint_to_local

def write_checkpoint(db: ControllerDB, remote_state_dir: str) -> tuple[str, CheckpointResult]: ...
def download_checkpoint_to_local(remote_state_dir: str, local_db_dir: Path,
                                  checkpoint_dir: str | None = None) -> bool: ...
```
`write_checkpoint` = backup (under write lock) + compress + upload. Returns `False` if no checkpoint found. `lib/iris/src/iris/cluster/controller/checkpoint.py:1`

```python
from iris.cluster.controller.scheduler import Scheduler, SchedulingContext

class Scheduler:
    def find_assignments(self, context: SchedulingContext) -> SchedulingResult: ...
    def get_job_scheduling_diagnostics(self, req, context, schedulable_task_id, num_tasks) -> str: ...
```
Pure functional; mutates `SchedulingContext` in-place — do not reuse across cycles. `lib/iris/src/iris/cluster/controller/scheduler.py:1`

```python
from iris.cluster.controller.main import run_controller_serve

def run_controller_serve(cluster_config: config_pb2.IrisClusterConfig, *,
                          host: str = "0.0.0.0", port: int = 10000,
                          checkpoint_path: str | None = None,
                          checkpoint_interval: float | None = None,
                          dry_run: bool = False) -> None: ...
```
Daemon entry point; blocks until SIGTERM. `lib/iris/src/iris/cluster/controller/main.py:1`

## Gotchas

- **`remote_state_dir` is mandatory** — `Controller.__init__` and `run_controller_serve` both raise `ValueError` if it is empty or absent in config.
- **`ScalingGroup.scale_up()` does not self-register** — callers must bracket with `begin_scale_up()` / `complete_scale_up()` or `cancel_scale_up()`; omitting these causes phantom capacity in `slice_count()`.
- **`read_snapshot()` vs `snapshot()`** — `read_snapshot()` uses a pooled read connection (safe for concurrent threads); `snapshot()` acquires the write lock and will deadlock if called from multiple threads simultaneously.
- **`add_endpoint` returns `False` silently** when the associated task is already terminal — check the return value before assuming registration succeeded.
- **`begin_checkpoint()` stalls scheduling** — it sets `_checkpoint_in_progress`, causing the scheduling and autoscaler loops to skip work until the checkpoint completes. Do not call it on a hot path.
- **`sync()` in `WorkerProvider` swallows RPC errors** into the third tuple element (`str | None`); callers must inspect it and call `fail_heartbeat`, not assume success.
