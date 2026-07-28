# Distributed training diagnostics: contract

## 1. NCCL RAS client

Location: `lib/iris/src/iris/cluster/runtime/nccl_ras.py`.

```python
class NcclRasFormat(StrEnum):
    JSON = "json"
    TEXT = "text"


class NcclRasFormatError(ValueError):
    """The RAS service did not return the requested machine-readable payload."""


@dataclass(frozen=True)
class CollectiveCountSkew:
    communicator_hash: str
    collective: str
    minimum: int
    maximum: int
    lagging_ranks: tuple[int, ...]


@dataclass(frozen=True)
class NcclRasSnapshot:
    captured_at: str
    response_format: NcclRasFormat
    raw_response: str
    report: dict[str, Any] | None

    def record(self) -> dict[str, Any]:
        """Return a JSON-serializable capture envelope."""


def query_nccl_ras(
    *,
    host: str,
    port: int,
    timeout: float,
    response_format: NcclRasFormat,
) -> bytes:
    """Send TIMEOUT, SET FORMAT, and VERBOSE STATUS; return the raw response."""


def parse_json_response(response: bytes) -> dict[str, Any]:
    """Parse a JSON object after zero or more text command acknowledgements."""


def capture_nccl_ras(
    *,
    host: str = "localhost",
    port: int = 28028,
    timeout: float = 5.0,
) -> NcclRasSnapshot:
    """Request JSON once and retry with text only when JSON parsing is unavailable."""


def collective_count_skews(report: Mapping[str, Any]) -> list[CollectiveCountSkew]:
    """Return count differences; an empty result does not assert job health."""
```

Network errors and timeouts propagate. Invalid vendor JSON raises `NcclRasFormatError` from `parse_json_response`; `capture_nccl_ras` converts only that error into a text retry. `raw_response` is always preserved.

## 2. Iris profile RPC

Location: `lib/iris/src/iris/rpc/job.proto`.

```proto
message DistributedProfile {
  bool include_threads = 1;
  bool include_process_waits = 2;
  bool include_gpu = 3;
  bool include_environment = 4;
  int32 collector_timeout_seconds = 5;
}

message ProfileType {
  oneof profiler {
    CpuProfile cpu = 1;
    MemoryProfile memory = 2;
    ThreadsProfile threads = 3;
    DistributedProfile distributed = 4;
  }
}
```

`collector_timeout_seconds` must be between 1 and 30; zero resolves to five. The CLI sets every `include_*` field to true. The thread collector keeps its existing 30-second ptrace watchdog even when the other collector timeout is lower.

`ProfileTaskResponse.profile_data` is the exact UTF-8 JSON written to `iris.profile`. A target without NCCL or GPUs succeeds with structured `unavailable` sections. Failure to enter the task namespace, failure to identify the process, or a payload above 4 MiB returns `ProfileTaskResponse.error` and writes no row.

## 3. Capture envelope

Schema version 1:

```json
{
  "schema_version": 1,
  "captured_at": "2026-07-28T01:10:00.000000",
  "source": "/power/job/task/0",
  "attempt_id": 1,
  "hostname": "pod-or-node",
  "runtime": {
    "pid": 1,
    "cuda_driver_version": "13.2",
    "cuda_runtime_version": "13.0",
    "jax_version": "0.x",
    "nccl_packages": ["nvidia-nccl-cu13==2.28.9"]
  },
  "nccl_ras": {
    "status": "ok",
    "response_format": "json",
    "raw_response": "OK\nOK\n{...}",
    "report": {},
    "collective_count_skews": []
  },
  "threads": {
    "status": "ok",
    "format": "py-spy",
    "text": "..."
  },
  "process_waits": {
    "status": "ok",
    "threads": [{"tid": 1, "name": "python", "wchan": "futex_wait_queue"}]
  },
  "gpu": {
    "status": "ok",
    "devices": [
      {
        "index": 0,
        "uuid": "GPU-...",
        "utilization_pct": 100.0,
        "memory_used_bytes": 157000000000,
        "power_w": 205.0,
        "power_limit_w": 1200.0
      }
    ]
  },
  "environment": {
    "NCCL_DEBUG": "INFO",
    "NCCL_DEBUG_SUBSYS": "INIT,BOOTSTRAP,ENV,NET,GRAPH,TUNING,RAS",
    "NCCL_RAS_ENABLE": "1",
    "XLA_FLAGS": "..."
  },
  "errors": []
}
```

Each collector section has `status` equal to `ok`, `unavailable`, `timeout`, or `error`. Errors include collector name, stable error class, and a message capped at 2 KiB. Environment capture allows only keys prefixed `NCCL_` plus `XLA_FLAGS`; it never serializes credentials or the full process environment.

## 4. Finelog profile shape

Location: `lib/iris/src/iris/cluster/stats/tables.py`.

Add:

```python
class ProfileType(StrEnum):
    CPU = "cpu"
    MEMORY = "memory"
    THREAD = "thread"
    DISTRIBUTED = "distributed"


class ProfileFormat(StrEnum):
    # existing values...
    JSON = "json"
```

The existing `IrisProfile` row is unchanged. A distributed capture sets:

- `type="distributed"`
- `format="json"`
- `trigger="on_demand"`
- `duration_seconds=0`
- CPU/memory/thread-specific nullable columns to `None`
- `profile_data` to the capture envelope bytes.

Seven-day `iris.profile` retention remains unchanged.

## 5. CLI

Location: `lib/iris/src/iris/cli/process_status.py`.

```text
iris process profile distributed \
  --target <task-id-or-job-id> \
  [--output <path>] \
  [--collector-timeout 5]
```

A task target writes one envelope. A job target lists the current running tasks and captures them with concurrency 16. The command exits nonzero when no task produced a capture. Partial task failures are printed beside successful output paths and produce a nonzero exit.

## 6. Progress metrics

Location: `lib/levanter/src/levanter/tracker/telltale.py`.

```python
class TrainingPhase(IntEnum):
    INITIALIZING = 0
    TRAINING = 1
    EVALUATION = 2
    CHECKPOINTING = 3
    FINISHED = 4


def set_training_phase(phase: TrainingPhase) -> None:
    """Publish the current phase to the process-global Telltale registry."""
```

Metrics:

- `levanter_progress_time_seconds`: Unix seconds, set after each completed training step.
- `levanter_phase`: one `TrainingPhase` numeric value.

`TelltaleTracker.__init__` initializes the phase to `INITIALIZING` and progress time to zero. The training loop sets `TRAINING` before dispatch, updates progress time after a returned step, brackets evaluation/checkpoint work with their phases, and sets `FINISHED` at clean shutdown.

## 7. GPU power normalization

Location: `lib/iris/src/iris/cluster/stats/tables.py`.

Add `gpu_power_limit_w: float | None = None` to `IrisWorkerStat`. `parse_dcgm` reads `DCGM_FI_DEV_ENFORCED_POWER_LIMIT`, sums it per node, and leaves the field `None` when DCGM does not expose it.

## 8. Grafana bridge alert projection

Location: `infra/grafana/src/server.py`.

```python
@dataclass(frozen=True)
class TrainingStall:
    cluster: str
    job_id: str
    run: str
    phase: str
    last_step: int
    stalled_seconds: int
    reporting_nodes: int
    high_utilization_nodes: int
    median_gpu_util_pct: float | None
    median_power_ratio: float | None
    classification: str
    value: int


def training_stall_rows(source: MetricSource, now: datetime) -> list[TrainingStall]:
    """Project fresh finelog progress, job state, utilization, and power into alerts."""
```

Route: `GET /finelog/marin/alerts/training_stalls`.

The route always returns at least one row. With no candidates it returns:

```json
{
  "cluster": "all",
  "job_id": "",
  "run": "",
  "phase": "",
  "last_step": 0,
  "stalled_seconds": 0,
  "reporting_nodes": 0,
  "high_utilization_nodes": 0,
  "median_gpu_util_pct": null,
  "median_power_ratio": null,
  "classification": "healthy",
  "value": 0
}
```

Candidate thresholds:

- `training`: 900 seconds without progress
- `initializing`: 2700 seconds since first fresh Telltale sample
- `evaluation` and `checkpointing`: 3600 seconds
- Telltale freshness: 90 seconds
- job-state freshness: 90 seconds and state `RUNNING`
- GPU window: 300 seconds
- high-utilization node: mean `gpu_util_pct >= 90`
- required high-utilization fraction: `>= 0.75`
- `collective_like`: median power ratio `< 0.35`; otherwise `generic_stall`.

Location: `infra/grafana/provisioning/alerting/rules.yaml`.

Add warning rule `TrainingProgressStalled`, evaluated every minute with `for: 5m`, `noDataState: Alerting`, and `execErrState: Alerting`. It fires when `value > 0` and routes to Slack and Loom through the existing warning policy. Grafana receives no Iris write credential.

## 9. Runtime environment

Normal multi-host training jobs explicitly set:

```text
NCCL_RAS_ENABLE=1
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=INIT,BOOTSTRAP,ENV,NET,GRAPH,TUNING,RAS
NCCL_DEBUG_FILE=/dev/stderr
```

Launch helpers record the resolved CUDA runtime, driver, JAX, and NCCL package versions as W&B config and Telltale labels. Debug reproduction jobs may add `COLL,PROXY,NVLS,REG`; no checked-in production config sets `NCCL_DEBUG=TRACE` or adds `CALL`.

## 10. Out of scope

- Automatic task kick, cluster restart, or job termination.
- Grafana calling an Iris mutation RPC.
- Periodic NCCL RAS polling from every process.
- A generic Telltale HTTP callback registry.
- Parsing NCCL text output into stable fields.
- Supporting GPU collectives outside NCCL in the first version.
