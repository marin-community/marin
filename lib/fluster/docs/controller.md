# Controller Overview

The Controller is the central coordination service in Fluster. It accepts job submissions from clients, assigns jobs to available workers, tracks job status through completion, and maintains an endpoint registry for actor discovery. All cluster-wide state flows through the Controller.

## Responsibilities

| Responsibility | Description |
|----------------|-------------|
| Job scheduling | Accepts job requests, queues them, assigns to workers with available capacity |
| Worker management | Tracks registered workers and their health status |
| Status tracking | Maintains authoritative job state, reports status to clients |
| Endpoint registry | Stores actor endpoints for service discovery |

## RPC Interface

The Controller exposes a single RPC service (`ControllerService`) with these methods:

### Job Management

| Method | Description |
|--------|-------------|
| `LaunchJob(JobRequest)` | Submit a job for execution, returns `JobId` |
| `GetJobStatus(JobId)` | Query current status of a job |
| `TerminateJob(JobId)` | Request termination of a running job |
| `ListJobs(filter)` | List jobs matching optional filter criteria |

### Worker Management

| Method | Description |
|--------|-------------|
| `RegisterWorker(WorkerInfo)` | Add a worker to the scheduling pool |
| `ListWorkers()` | List all registered workers |

### Endpoint Registry

| Method | Description |
|--------|-------------|
| `RegisterEndpoint(name, address, metadata)` | Register an actor endpoint |
| `UnregisterEndpoint(endpoint_id)` | Remove an actor endpoint |
| `LookupEndpoint(name)` | Find endpoints by actor name |
| `ListEndpoints(prefix)` | List all endpoints, optionally filtered by name prefix |

## Job Lifecycle

Jobs progress through these states:

```
PENDING ──► BUILDING ──► RUNNING ──► SUCCEEDED
                │           │
                ▼           ▼
             FAILED      FAILED
```

| State | Description |
|-------|-------------|
| `PENDING` | Job submitted, waiting for worker assignment |
| `BUILDING` | Assigned to worker, environment being prepared |
| `RUNNING` | Job entrypoint executing |
| `SUCCEEDED` | Completed with exit code 0 |
| `FAILED` | Completed with error or non-zero exit |
| `KILLED` | Terminated by user request |
| `WORKER_FAILED` | Worker became unresponsive |
| `UNSCHEDULABLE` | No worker can satisfy resource requirements |

## Scheduling Behavior

The Controller assigns pending jobs to workers using first-fit scheduling:

1. Jobs are processed in submission order (FIFO)
2. Each job is assigned to the first worker with sufficient capacity
3. Jobs remain pending until a suitable worker is available
4. Failed jobs may be retried based on failure type and retry policy

## Integration Points

```
┌──────────────┐         ┌──────────────┐
│    Client    │         │    Worker    │
│              │         │              │
│ submit()  ───┼────────►│              │
│ status()  ───┼────────►│◄─────────────┼── RegisterWorker
│ wait()    ───┼────────►│              │
│              │         │◄─────────────┼── job dispatch
└──────────────┘         │              │
                         │◄─────────────┼── status updates
┌──────────────┐         │              │
│ ActorServer  │         └──────────────┘
│              │                │
│ register() ──┼────────────────┘
└──────────────┘
        ▲
        │
┌──────────────┐
│   Resolver   │
│              │
│ lookup() ────┼── LookupEndpoint
└──────────────┘
```

- **Clients** submit jobs and query status via the Controller
- **Workers** register themselves and receive job assignments
- **ActorServers** register endpoints for discovery
- **Resolvers** query the endpoint registry to locate actors

## File Summary

| File | Purpose |
|------|---------|
| `controller.py` | Main `Controller` class and startup |
| `state.py` | State management and data types |
| `scheduler.py` | Job-to-worker assignment logic |
| `service.py` | RPC method implementations |
| `dashboard.py` | Web monitoring UI |
| `retry.py` | Retry policy for failed jobs |
| `workers.py` | Worker capacity evaluation |
