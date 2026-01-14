# Worker Overview

The Worker is the execution agent in Fluster. It registers with the Controller, receives job assignments, prepares execution environments, runs jobs in isolated containers, and reports status back to the Controller. Workers are stateless—they can be added or removed from the cluster without affecting other workers or requiring coordination.

## Responsibilities

| Responsibility | Description |
|----------------|-------------|
| Job execution | Runs job entrypoints in isolated Docker containers |
| Environment preparation | Downloads bundles, builds images, sets up dependencies |
| Status reporting | Reports job progress and completion to the Controller |
| Log collection | Captures and serves job stdout/stderr |
| Port allocation | Assigns ephemeral ports for actor servers within jobs |

## RPC Interface

The Worker exposes a single RPC service (`WorkerService`) with these methods:

| Method | Description |
|--------|-------------|
| `RunJob(JobRequest)` | Execute a job on this worker |
| `GetJobStatus(JobId)` | Query current status of a job |
| `ListJobs()` | List all jobs on this worker |
| `FetchLogs(JobId, options)` | Retrieve job logs with optional filtering |
| `KillJob(JobId)` | Terminate a running job |
| `HealthCheck()` | Liveness probe |

## Job Execution Flow

When the Controller dispatches a job to a worker:

```
1. Download bundle ──► 2. Build image ──► 3. Start container ──► 4. Monitor
        │                    │                    │                  │
        ▼                    ▼                    ▼                  ▼
   Cache lookup        Cache lookup         Port allocation     Log collection
   or fetch URL        or docker build      Env var injection   Status updates
```

1. **Download bundle**: Fetch the workspace archive containing code and dependencies
2. **Build image**: Create a Docker image with the required Python environment
3. **Start container**: Launch the container with allocated ports and environment variables
4. **Monitor**: Track container status, collect logs, report completion

## Environment Variables

The Worker injects these environment variables into every job container:

| Variable | Description |
|----------|-------------|
| `FLUSTER_JOB_ID` | Unique job identifier |
| `FLUSTER_WORKER_ID` | ID of the executing worker |
| `FLUSTER_CONTROLLER_ADDRESS` | Controller URL for actor registration |
| `FLUSTER_NAMESPACE` | Namespace for actor isolation |
| `FLUSTER_PORT_<NAME>` | Allocated ports (e.g., `FLUSTER_PORT_ACTOR`) |

Jobs that run actor servers use `FLUSTER_PORT_ACTOR` to bind their server and `FLUSTER_CONTROLLER_ADDRESS` to register with the endpoint registry.

## Integration Points

```
┌──────────────────────────────────────────────────────────┐
│                       Controller                          │
│                                                           │
│  ◄───────────────────┬────────────────────────────────►  │
│     RegisterWorker   │   RunJob, GetJobStatus, KillJob   │
└──────────────────────┼───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│                        Worker                             │
│                                                           │
│   JobManager ──► Docker ──► Container                    │
│                                  │                        │
│                                  ▼                        │
│                           Job Entrypoint                  │
│                           (user code)                     │
│                                  │                        │
│                                  ▼                        │
│                           ActorServer                     │
│                           (optional)                      │
└──────────────────────────────────────────────────────────┘
```

- The **Controller** dispatches jobs and polls for status updates
- The **JobManager** orchestrates the full job lifecycle
- **Docker** provides container isolation
- **Job Entrypoint** is user code that may optionally start an **ActorServer**

## Job States on Worker

| State | Description |
|-------|-------------|
| `BUILDING` | Downloading bundle, building image |
| `RUNNING` | Container executing |
| `SUCCEEDED` | Exited with code 0 |
| `FAILED` | Exited with non-zero code or error |
| `KILLED` | Terminated by request |

## File Summary

| File | Purpose |
|------|---------|
| `worker.py` | Main `Worker` class, configuration, startup |
| `manager.py` | `JobManager` lifecycle orchestration, port allocation |
| `service.py` | RPC method implementations |
| `docker.py` | Container runtime interface |
| `builder.py` | Image building |
| `bundle.py` | Bundle download |
| `worker_types.py` | Internal job tracking types |
| `dashboard.py` | Web monitoring UI |
| `main.py` | CLI entry point |
