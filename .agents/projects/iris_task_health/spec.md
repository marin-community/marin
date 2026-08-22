# Iris Task Health Contract

## HTTP Contract

An enabled Iris task exposes one HTTP route:

```text
GET /healthz
```

Status codes from 200 through 399 mean healthy. Other status codes mean unhealthy. A connection error or timeout also means unhealthy. Iris ignores the response body for the verdict. Iris includes at most 4096 bytes of the last failure body in diagnostics.

The probe does not follow redirects. A redirect status is healthy without a second request.

The task server binds `IRIS_BIND_HOST`. It reads `IRIS_PORT_HEALTHZ` as the requested port. A value of `0` permits kernel port selection. After listen starts, the task publishes the real port through the Python API below.

## Public Python API

### Iris client

File: `lib/iris/src/iris/cluster/health.py`

```python
from dataclasses import dataclass

from rigging.timing import Duration


HEALTH_PATH = "/healthz"
HEALTH_PORT_NAME = "healthz"


@dataclass(frozen=True)
class TaskHealthCheck:
    startup_timeout: Duration
    period: Duration
    request_timeout: Duration
    failure_threshold: int


def task_health_port() -> int:
    """Return the Iris health port for the current task.

    The function returns a positive allocated port on worker-daemon backends.
    It returns zero on Kubernetes, where the server selects an available port.
    It raises RuntimeError when the task has no health contract.
    """


def task_health_enabled() -> bool:
    """Return true when the current task has an Iris health contract."""


def publish_task_health(port: int) -> None:
    """Publish the listening health port.

    The function writes the port atomically to IRIS_HEALTH_PORT_FILE. When Iris
    allocates a positive port, the published port must have the same value. The
    function raises ValueError for an invalid or different port.
    """
```

`IrisClient.submit`, `RemoteClusterClient.submit_job`, and their protocols add this parameter:

```python
health_check: TaskHealthCheck | None = None
```

`None` disables all health behavior and does not reserve a port.

### Fray

File: `lib/fray/src/fray/types.py`

```python
@dataclass(frozen=True)
class TaskHealthCheck:
    startup_timeout: Duration
    period: Duration
    request_timeout: Duration
    failure_threshold: int


@dataclass
class JobRequest:
    # Existing fields stay unchanged.
    health_check: TaskHealthCheck | None = None
```

`FrayIrisClient.submit` converts the Fray type to the Iris type. `LocalClient` rejects a non-null field because it does not run probes.

## Wire Contract

File: `lib/iris/src/iris/rpc/job.proto`

```proto
message TaskHealthCheck {
  iris.time.Duration startup_timeout = 1;
  iris.time.Duration period = 2;
  iris.time.Duration request_timeout = 3;
  int32 failure_threshold = 4;
}

message RunTaskRequest {
  // Fields 1 through 15 stay unchanged.
  TaskHealthCheck health_check = 16;
}
```

File: `lib/iris/src/iris/rpc/controller.proto`

```proto
message Controller {
  message LaunchJobRequest {
    // Fields 1 through 38 stay unchanged.
    iris.job.TaskHealthCheck health_check = 39;
  }
}
```

The controller rejects these values with `INVALID_ARGUMENT`:

- A zero or negative duration.
- A `failure_threshold` less than one.
- A `request_timeout` greater than or equal to the `period`.

The controller rejects duplicate port names. It also rejects a user-supplied `healthz` name when `health_check` is present. The controller then adds the reserved `healthz` port itself.

## Persisted Shape

File: `lib/iris/src/iris/cluster/controller/schema.py`

```sql
ALTER TABLE job_config ADD COLUMN health_check_json TEXT;
```

`NULL` means that the job has no task health contract. A non-null value is the protobuf JSON form of `iris.job.TaskHealthCheck`. The next numbered migration adds the column after a `PRAGMA table_info(job_config)` check.

`writes.insert_job_config` stores this value for local and incoming federated jobs. Display-only mirror rows leave it null. Job-config reads and `RunTemplatesProjection` restore it. `reconstruct_launch_job_request` copies it into outbound federation requests. The run-template projection copies it into `RunTaskRequest.health_check`.

## Runtime Files and Environment

Iris adds these environment values for enabled tasks:

```text
IRIS_PORT_HEALTHZ=<allocated-port-or-zero>
IRIS_HEALTH_PORT_FILE=/tmp/iris/health-port
IRIS_HEALTH_FAILURE_COUNT_FILE=/tmp/iris/health-failures
IRIS_HEALTH_TERMINATION_FILE=/tmp/iris/health-termination-log
```

`publish_task_health` creates the parent directory and replaces the port file atomically. The file contains one base-10 port and one newline. Its mode is `0644`.

The Kubernetes probe helper lives at `lib/iris/src/iris/runtime/health_probe.py`. It reads the port file and requests `http://127.0.0.1:<port>/healthz`. It returns process status zero for HTTP 200 through 399. It returns status one for all failures.

The helper accepts `--timeout <seconds>` and applies the configured request timeout. It does not follow redirects. It uses the standard library and has no extra runtime dependency.

An enabled Kubernetes task must provide the `iris.runtime.health_probe` module through `$IRIS_VENV` or `$IRIS_PYTHON`. A missing module stays a startup failure.

On a live threshold failure, the helper writes a bounded failure message to `IRIS_HEALTH_TERMINATION_FILE`:

```text
Task health check failed <count> consecutive times: <detail>
```

The live helper keeps its diagnostic count in `IRIS_HEALTH_FAILURE_COUNT_FILE`. Success resets the count and removes the Iris termination file. Startup probes do not change either file.

## Backend Behavior

### Worker daemon

The monitor does not probe during the container build phase. After the run container starts, it enters startup state. The first success changes the state to live. Startup expires at `startup_timeout`.

The first probe starts immediately. Later probes use monotonic, start-to-start scheduling. The monitor never runs two probes at once. It checks process status before and after each request.

In live state, a success sets the consecutive-failure count to zero. A failure adds one. At the threshold, the monitor checks process status again. If the process still runs, the monitor saves the health terminal cause as `TASK_STATE_FAILED` and stops it.

The monitor checks for process exit before each probe. A process that exits before the threshold keeps its exit code and normal terminal error.

Docker run containers add one label:

```text
iris.health_check=<protobuf-JSON>
```

The monitor writes `.iris_health_live` in the task work directory after the first success. Container adoption restores the health contract and the existing `iris.ports` label. The live marker restores the phase. Without the marker, the original container start time defines the remaining startup window.

### Kubernetes

The task container adds these probe entries:

```yaml
startupProbe:
  exec:
    command:
      - bash
      - -lc
      - >-
        if [ -x "$IRIS_VENV/bin/python" ]; then
          exec "$IRIS_VENV/bin/python" -m iris.runtime.health_probe --phase startup --timeout <request_timeout>;
        fi;
        exec "$IRIS_PYTHON" -m iris.runtime.health_probe --phase startup --timeout <request_timeout>
  periodSeconds: <period>
  timeoutSeconds: <request_timeout + 1>
  initialDelaySeconds: 0
  failureThreshold: <ceil(startup_timeout / period) + 1>
livenessProbe:
  exec:
    command:
      - bash
      - -lc
      - >-
        if [ -x "$IRIS_VENV/bin/python" ]; then
          exec "$IRIS_VENV/bin/python" -m iris.runtime.health_probe --phase live --timeout <request_timeout> --failure-threshold <failure_threshold>;
        fi;
        exec "$IRIS_PYTHON" -m iris.runtime.health_probe --phase live --timeout <request_timeout> --failure-threshold <failure_threshold>
  periodSeconds: <period>
  timeoutSeconds: <request_timeout + 1>
  initialDelaySeconds: 0
  failureThreshold: <failure_threshold>
```

All duration fields must have an integer number of seconds. Submission rejects sub-second values. The Pod keeps `restartPolicy: Never`. A probe kill thus creates one failed task attempt. Iris applies the current failure retry limits.

The extra kubelet second lets the helper record its own request timeout before kubelet stops the probe process.

The task container sets `terminationMessagePath` to `/tmp/iris/health-termination-log`. It keeps `terminationMessagePolicy: FallbackToLogsOnError`. An empty health file thus preserves the current log-tail fallback.

The startup threshold assumes an immediate first probe. The added probe count makes termination occur on or after `startup_timeout`.

## Levanter Training Consumer

File: `lib/levanter/src/levanter/training_control.py`

`TrainingDashboard` accepts the local `ProgressWatchdog`. It serves the PR #8554 JSON report at `/healthz`. It binds `task_health_port()` and enters `publish_task_health(actual_port)` after the server listens.

One local task leader starts the server. Global JAX process zero also registers `training-control` with `EndpointAccess.LINK`. Other task leaders do not register an endpoint.

Without an Iris health contract, only global process zero starts the current public dashboard. Its server remains best-effort. With a contract, `task_health_enabled()` selects one local leader per task. An enabled task without a watchdog fails at process startup.

With a contract, health server bind and port publication errors propagate. Endpoint-registry errors only disable the public link. Health startup does not depend on a checkpointer.

The health response keeps the PR #8554 JSON shape:

```json
{
  "run_id": "<run-id>",
  "job_id": "<job-id>",
  "task_id": "<task-id>",
  "monitored": true,
  "state": "starting",
  "event": "process_started",
  "elapsed": 12.5,
  "timeout": 4800.0
}
```

`event`, `elapsed`, and `timeout` can be null. An enabled Iris health route always has `monitored: true`. State `stalled` returns 503. Other states return 200.

`ProgressWatchdogConfig.create` returns a watchdog on every JAX process. It attaches the diagnostic function and timeout only on global process zero. All watchdogs keep the same progress deadlines and exit code 124.

Files: `lib/iris/src/iris/hooks/multigpu.py` and `lib/iris/src/iris/hooks/multigpu_main.py`

```python
IRIS_MULTIGPU_LOCAL_PROCESS_INDEX_ENV = "IRIS_MULTIGPU_LOCAL_PROCESS_INDEX"
```

`multigpu_main` sets this value for each child. `TrainingDashboard` treats value zero as the local task leader. A process without the value is the task leader.

File: `experiments/grug/dispatch.py`

```python
GRUG_TRAINING_HEALTH_CHECK = TaskHealthCheck(
    startup_timeout=Duration.from_minutes(30),
    period=Duration.from_seconds(10),
    request_timeout=Duration.from_seconds(3),
    failure_threshold=13,
)
```

The first rollout sets this value on Grug EP hero `JobRequest` objects.

At 10-second intervals, 13 failures give at least 120 seconds after the first failure. This exceeds the 60-second poll, the 20-second diagnostic budget, and a 30-second margin.

## Errors

- Invalid configuration fails job submission with `INVALID_ARGUMENT`.
- A missing published port counts as a startup failure.
- A worker-daemon threshold failure reports `TASK_STATE_FAILED` with the last probe detail.
- A Kubernetes live-threshold failure reports `TASK_STATE_FAILED` with the health termination detail.
- A Kubernetes startup failure uses the current log-tail fallback.
- Probe failure never reports `WORKER_FAILED` or `PREEMPTED`.
- A process exit before the threshold keeps its normal exit code and terminal error.
- One unhealthy coscheduled attempt makes siblings `COSCHED_FAILED`. Only the unhealthy attempt charges the failure budget.

## File Summary

| File | Contract change |
|---|---|
| `lib/iris/src/iris/rpc/job.proto` | Shared health message and run field. |
| `lib/iris/src/iris/rpc/controller.proto` | Launch field. |
| `lib/iris/src/iris/cluster/health.py` | Client config and port publication API. |
| `lib/iris/src/iris/runtime/health_probe.py` | Kubernetes local probe command. |
| `lib/iris/src/iris/cluster/worker/task_attempt.py` | Worker-daemon state and failure action. |
| `lib/iris/src/iris/cluster/runtime/docker.py` | Adoption label. |
| `lib/iris/src/iris/cluster/runtime/types.py` | Adopted health metadata. |
| `lib/iris/src/iris/cluster/backends/k8s/tasks.py` | Kubernetes startup and liveness probes. |
| `lib/iris/src/iris/cluster/controller/schema.py` | Persisted health config. |
| `lib/iris/src/iris/cluster/controller/migrations/0051_task_health.py` | Health-config migration. |
| `lib/iris/src/iris/cluster/controller/writes.py` | Local and mirrored config writes. |
| `lib/iris/src/iris/cluster/controller/reads.py` | Persisted config reads. |
| `lib/iris/src/iris/cluster/controller/codec.py` | Federation request reconstruction. |
| `lib/iris/src/iris/cluster/controller/projections/run_templates.py` | Per-attempt request projection. |
| `lib/iris/src/iris/cluster/client/protocol.py` | Client submission protocol. |
| `lib/iris/src/iris/cluster/client/remote_client.py` | Remote submission mapping. |
| `lib/iris/src/iris/client/client.py` | High-level submission mapping. |
| `lib/fray/src/fray/types.py` | Backend-neutral submission field. |
| `lib/fray/src/fray/iris_backend.py` | Iris conversion. |
| `lib/levanter/src/levanter/training_control.py` | Training health server. |
| `lib/levanter/src/levanter/callbacks/progress_watchdog.py` | Per-rank watchdog creation. |
| `lib/levanter/src/levanter/trainer.py` | Training-dashboard watchdog access. |
| `lib/iris/src/iris/hooks/multigpu.py` | Local task-leader environment name. |
| `lib/iris/src/iris/hooks/multigpu_main.py` | Local task-leader environment value. |
| `lib/levanter/src/levanter/main/train_lm.py` | Standard training consumer. |
| `experiments/grug/base/train.py` | Base Grug training consumer. |
| `experiments/grug/moe/train.py` | Grug MoE training consumer. |
| `experiments/grug/moe_hero_ep/train.py` | EP hero training consumer. |
| `experiments/grug/moe_hero_fsdp/train.py` | FSDP hero training consumer. |
| `experiments/grug/dispatch.py` | First training rollout values. |

## Out of Scope

- Controller, worker, and cluster synthetic health endpoints.
- Endpoint-registry health metadata or proxy-based probes.
- Kubernetes readiness probes and Service traffic.
- More than one health route per task.
- Controller storage of each live probe result.
- Automatic health checks for jobs that do not opt in.
