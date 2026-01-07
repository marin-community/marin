# Controller Class

The `Controller` class provides a unified interface for managing all controller components and their lifecycle.

## Overview

Instead of manually initializing and managing 5+ controller components, the `Controller` class encapsulates:

- **ControllerState**: Thread-safe job and worker state
- **Scheduler**: Background thread for job scheduling
- **HeartbeatMonitor**: Background thread for worker health checks
- **ControllerServiceImpl**: RPC service implementation
- **ControllerDashboard**: Web dashboard and HTTP server

## Basic Usage

```python
from pathlib import Path
from fluster.cluster.controller import Controller, ControllerConfig
from fluster.cluster.controller.state import ControllerJob, ControllerWorker
from fluster.cluster.types import JobId, WorkerId
from fluster import cluster_pb2

# Define callbacks
def dispatch_job(job: ControllerJob, worker: ControllerWorker) -> bool:
    """Dispatch a job to a worker."""
    # Send RPC to worker
    return True

def send_heartbeat(address: str) -> cluster_pb2.HeartbeatResponse | None:
    """Check worker health."""
    # Send heartbeat RPC
    return response

def on_worker_failed(worker_id: WorkerId, job_ids: list[JobId]) -> None:
    """Handle worker failure."""
    # Retry jobs, log failure, etc.
    pass

# Configure controller
config = ControllerConfig(
    host="127.0.0.1",
    port=8080,
    bundle_dir=Path("/tmp/bundles"),
    scheduler_interval_seconds=0.5,
    heartbeat_interval_seconds=2.0,
)

# Create and start controller
controller = Controller(
    config=config,
    dispatch_fn=dispatch_job,
    heartbeat_fn=send_heartbeat,
    on_worker_failed=on_worker_failed,
)
controller.start()

try:
    # Submit jobs
    response = controller.launch_job(job_request)
    job_id = response.job_id

    # Query status
    status = controller.get_job_status(job_id)

    # Register workers
    controller.register_worker(worker_request)

finally:
    controller.stop()
```

## Configuration

The `ControllerConfig` dataclass provides all configuration options:

```python
@dataclass
class ControllerConfig:
    host: str = "127.0.0.1"
    port: int = 0  # 0 for auto-assign
    bundle_dir: Path | None = None
    scheduler_interval_seconds: float = 0.5
    heartbeat_interval_seconds: float = 2.0
```

## Methods

### Lifecycle Methods

- **`start()`**: Start all background components (scheduler, heartbeat monitor, dashboard server)
- **`stop()`**: Stop all background components gracefully

### Job Management

- **`launch_job(request)`**: Submit a new job
- **`get_job_status(job_id)`**: Query job status
- **`terminate_job(job_id)`**: Terminate a running job

### Worker Management

- **`register_worker(request)`**: Register a worker with the controller

## Properties

- **`state`**: Access to the underlying `ControllerState` for advanced usage
- **`url`**: HTTP URL of the controller dashboard and RPC service

## Callbacks

The Controller requires three callback functions:

### dispatch_fn

```python
def dispatch_fn(job: ControllerJob, worker: ControllerWorker) -> bool:
    """Dispatch a job to a worker.

    Args:
        job: Job to dispatch
        worker: Worker to dispatch to

    Returns:
        True if dispatch succeeded, False otherwise
    """
```

Called by the scheduler when a job should be dispatched to a worker. Should send an RPC to the worker to start the job.

### heartbeat_fn

```python
def heartbeat_fn(address: str) -> cluster_pb2.HeartbeatResponse | None:
    """Check worker health.

    Args:
        address: Worker address (host:port)

    Returns:
        HeartbeatResponse on success, None on failure
    """
```

Called by the heartbeat monitor to check worker health. Should send an RPC to the worker.

### on_worker_failed

```python
def on_worker_failed(worker_id: WorkerId, job_ids: list[JobId]) -> None:
    """Handle worker failure.

    Args:
        worker_id: Failed worker ID
        job_ids: Jobs that were running on the worker
    """
```

Called when a worker exceeds the heartbeat failure threshold. Should handle job retry logic.

## Benefits Over Manual Composition

### Before (Manual)

```python
# Create all components manually
state = ControllerState()
scheduler = Scheduler(state, dispatch_fn, interval_seconds=0.5)
heartbeat_monitor = HeartbeatMonitor(state, heartbeat_fn, on_worker_failed, interval_seconds=2.0)
service = ControllerServiceImpl(state, scheduler, bundle_dir=bundle_dir)
dashboard = ControllerDashboard(service, host="127.0.0.1", port=8080)

# Start each component
scheduler.start()
heartbeat_monitor.start()
server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()
time.sleep(1.0)

# Use components
response = service.launch_job(request, None)

# Stop everything
heartbeat_monitor.stop()
scheduler.stop()
```

### After (Controller)

```python
# Create and configure
config = ControllerConfig(port=8080, bundle_dir=bundle_dir)
controller = Controller(config, dispatch_fn, heartbeat_fn, on_worker_failed)

# Start
controller.start()

# Use
response = controller.launch_job(request)

# Stop
controller.stop()
```

## Dashboard Access

When the controller is running, the dashboard is accessible at:

- **`/`**: Web dashboard with auto-refresh
- **`/health`**: Health check endpoint
- **`/api/stats`**: Statistics JSON
- **`/api/jobs`**: Jobs list JSON
- **`/api/workers`**: Workers list JSON
- **`/api/actions`**: Recent actions log JSON
- **`/fluster.cluster.ControllerService/*`**: Connect RPC endpoints

## Testing

The Controller class is designed to be easily testable by accepting callbacks in the constructor:

```python
def test_controller():
    dispatch_calls = []

    def mock_dispatch(job, worker):
        dispatch_calls.append((job, worker))
        return True

    config = ControllerConfig(port=0)
    controller = Controller(config, mock_dispatch, mock_heartbeat, mock_failure)

    # Test functionality
    controller.launch_job(request)
    assert len(dispatch_calls) > 0
```

## Advanced Usage

For advanced use cases, access the underlying state:

```python
# Access all jobs
all_jobs = controller.state.list_all_jobs()

# Access all workers
all_workers = controller.state.list_all_workers()

# Get recent actions
actions = controller.state.get_recent_actions()

# Direct state manipulation (use with caution)
job = controller.state.get_job(JobId(job_id))
if job:
    job.state = cluster_pb2.JOB_STATE_FAILED
```

## See Also

- [Controller State](../src/fluster/cluster/controller/state.py): In-memory state management
- [Scheduler](../src/fluster/cluster/controller/scheduler.py): Job scheduling algorithm
- [Heartbeat Monitor](../src/fluster/cluster/controller/heartbeat.py): Worker health checking
- [Controller Service](../src/fluster/cluster/controller/service.py): RPC implementation
- [Dashboard](../src/fluster/cluster/controller/dashboard.py): Web UI and HTTP server
