# Iris-to-Rust Migration: Step-by-Step Plan

Each step is a single PR-sized change. The system works after every step with a mix of Rust and Python.

---

## Phase 1: Foundation (Steps 1–5)

### Step 1: Rust Project Scaffolding

**What**: Create `lib/iris-native/` with Cargo.toml, pyproject.toml, and build integration.

**Source files**:
- New: `lib/iris-native/Cargo.toml`, `lib/iris-native/pyproject.toml`, `lib/iris-native/src/lib.rs`
- Modified: root `pyproject.toml` (workspace member), `Cargo.toml` (workspace)

**Rust crates**: `pyo3 0.26` (abi3-py311), `prost 0.13`, `prost-build`

**Python interface**: `from iris._native import hello_world` — a trivial test function that returns a string, verifying the build pipeline works end-to-end.

**Tests**: `tests/test_native_import.py` — import `iris._native`, call `hello_world()`, assert return value. CI build verifies maturin compilation.

**Dependencies**: None.

---

### Step 2: Proto Codegen in Rust

**What**: Generate Rust types from the existing `.proto` files using `prost-build`.

**Source files**:
- New: `lib/iris-native/build.rs` (prost-build script), `lib/iris-native/src/proto/` (generated)
- Read: `lib/iris/src/iris/rpc/time.proto`, `errors.proto`, `logging.proto`, `query.proto`, `vm.proto`, `config.proto`, `cluster.proto`, `actor.proto`

**Rust crates**: `prost 0.13`, `prost-build 0.13`, `prost-types`

**Python interface**: None directly. This step generates Rust types that later steps expose.

**Tests**: Rust `#[cfg(test)]` tests that construct each major proto message type (Timestamp, Duration, JobStatus, ResourceSpecProto, LaunchJobRequest) and verify roundtrip encode/decode via `prost::Message`.

**Dependencies**: Step 1.

---

### Step 3: Time Utilities in Rust

**What**: Port `Timestamp`, `Duration`, `Deadline`, `Timer`, `RateLimiter`, `TokenBucket`, `ExponentialBackoff` from `lib/iris/src/iris/time_utils.py:1-556`.

**Source files**:
- New: `lib/iris-native/src/time_utils.rs`
- Modified: `lib/iris/src/iris/time_utils.py` (add import shim with `IRIS_USE_NATIVE` fallback)
- Preserved: `lib/iris/src/iris/_time_utils_py.py` (copy of original for rollback)

**Rust crates**: `pyo3`, `prost` (for proto conversion), `std::time` (Instant, SystemTime)

**Python interface**: PyO3 `#[pyclass]` for each type. All existing methods exposed as `#[pymethods]`. The Python module re-exports from `iris._native`:
```python
# time_utils.py
from iris._native import Timestamp, Duration, Deadline, Timer, RateLimiter, TokenBucket, ExponentialBackoff
```

**Tests**:
- Rust unit tests mirroring `tests/test_time_utils.py` cases
- Python integration test calling both `_time_utils_py` and `_native` versions, asserting identical results
- Existing `tests/` that use `time_utils` pass unchanged

**Dependencies**: Step 2 (proto types for `to_proto()`/`from_proto()` methods).

---

### Step 4: Core Enum and Value Types

**What**: Port `AttributeValue`, `ConstraintOp`, `WellKnownAttribute`, `DeviceType` from `lib/iris/src/iris/cluster/constraints.py:30-145`.

**Source files**:
- New: `lib/iris-native/src/types.rs`
- Modified: `lib/iris/src/iris/cluster/constraints.py` (import shim for ported types)

**Rust crates**: `pyo3`, `prost`

**Python interface**: `#[pyclass]` with `#[pymethods]` for `to_proto()`, `from_proto()`, comparison operators, hash. `WellKnownAttribute` as a Rust enum with `#[pyclass]`.

**Tests**:
- Rust tests for AttributeValue proto roundtrips, ConstraintOp mapping
- Python test asserting `_native.AttributeValue(42).to_proto()` matches Python version

**Dependencies**: Step 2.

---

### Step 5: Constraint Evaluation Engine

**What**: Port `Constraint`, `evaluate_constraint`, `check_resource_fit`, `ConstraintIndex`, `ResourceCapacity` from `lib/iris/src/iris/cluster/constraints.py:145-end`.

**Source files**:
- New: `lib/iris-native/src/constraints.rs`
- Modified: `lib/iris/src/iris/cluster/constraints.py` (import shim for evaluation functions)

**Rust crates**: `pyo3`, `prost`

**Python interface**:
- `evaluate_constraint(constraint, attributes) -> bool` as `#[pyfunction]`
- `check_resource_fit(required, available) -> bool` as `#[pyfunction]`
- `ConstraintIndex` as `#[pyclass]` with `matches()` method

**Tests**:
- Parametric Rust tests for all constraint operators (EQ, NE, EXISTS, NOT_EXISTS, GT, GE, LT, LE, IN)
- Python integration test running the scheduler test suite's constraint cases against both implementations
- Existing `tests/cluster/controller/test_scheduler.py` passes unchanged

**Dependencies**: Step 4.

---

## Phase 2: Data Layer (Steps 6–10)

### Step 6: JobName and WorkerId Types

**What**: Port `JobName`, `WorkerId`, `Namespace` from `lib/iris/src/iris/cluster/types.py:29-200`.

**Source files**:
- New: `lib/iris-native/src/job_name.rs`
- Modified: `lib/iris/src/iris/cluster/types.py` (import shim)

**Rust crates**: `pyo3`

**Python interface**: `#[pyclass]` for `JobName` with all methods: `from_string()`, `child()`, `task()`, `parent`, `user`, `root_job`, `namespace`, `is_root`, `task_index`, `depth`, `wire_format`. `WorkerId` as a newtype wrapper.

**Tests**: Rust tests for parsing, hierarchy navigation, wire format roundtrips. Python test asserting `_native.JobName.from_string("/alice/job")` matches Python `JobName.from_string("/alice/job")`.

**Dependencies**: Step 1.

---

### Step 7: ResourceSpec and EnvironmentSpec

**What**: Port `ResourceSpec`, `EnvironmentSpec` from `lib/iris/src/iris/cluster/types.py:200-500` (excluding `Entrypoint` which uses cloudpickle).

**Source files**:
- New: `lib/iris-native/src/resource_spec.rs`
- Modified: `lib/iris/src/iris/cluster/types.py` (import shim for ResourceSpec, EnvironmentSpec)

**Rust crates**: `pyo3`, `prost`, `humansize` (for human-readable byte parsing)

**Python interface**: `#[pyclass]` with `to_proto()`, `from_proto()` and all builder methods. `Entrypoint` stays in Python.

**Tests**: Rust tests for proto conversion, human-readable parsing ("4GB" -> bytes). Python test asserting `ResourceSpec(cpu=4, memory="8GB").to_proto()` matches.

**Dependencies**: Step 2, Step 4.

---

### Step 8: SQLite DB Foundation

**What**: Create the Rust DB access layer — connection management, read snapshots, migration runner. Port the `Predicate` query builder.

**Source files**:
- New: `lib/iris-native/src/db.rs`, `lib/iris-native/src/db/predicate.rs`, `lib/iris-native/src/db/migrations.rs`
- Read: `lib/iris/src/iris/cluster/controller/db.py:1-350` (Predicate, db_row_model, ControllerDB connection handling)
- Read: `lib/iris/src/iris/cluster/controller/migrations/` (all 9 migration files)

**Rust crates**: `rusqlite 0.32` (bundled), `pyo3`

**Python interface**: Not exposed via PyO3 yet. This step builds the Rust-internal DB layer that later steps use. A `#[pyfunction]` `verify_db_schema(path: &str) -> bool` is exposed for testing.

**Tests**: Rust tests that create an in-memory SQLite DB, run all migrations, insert test data, and query using the Predicate builder. Python test that creates a DB via Python, then calls `verify_db_schema()` from Rust to confirm compatibility.

**Dependencies**: Step 1.

---

### Step 9: DB Row Models in Rust

**What**: Port the typed row models (`JobRow`, `TaskRow`, `WorkerRow`, `EndpointRow`) from `lib/iris/src/iris/cluster/controller/db.py:350-end`.

**Source files**:
- New: `lib/iris-native/src/db/models.rs`
- Read: `lib/iris/src/iris/cluster/controller/db.py:350-1334`

**Rust crates**: `rusqlite`, `pyo3`, `serde`

**Python interface**: `#[pyclass]` for each row model with field accessors. A `read_workers(db_path: str) -> list[WorkerRow]` function exposed via PyO3 for the scheduler to consume.

**Tests**: Roundtrip test: Python inserts rows → Rust reads them → assert field values match. Rust inserts rows → Python reads them → assert match.

**Dependencies**: Step 8, Step 6, Step 3 (Timestamp fields in rows).

---

### Step 10: DB Write Operations

**What**: Port write operations: `insert_job`, `update_task_state`, `insert_worker`, `upsert_endpoint`, `insert_log_entries`, transaction support.

**Source files**:
- Extended: `lib/iris-native/src/db.rs`, `lib/iris-native/src/db/models.rs`
- Modified: `lib/iris/src/iris/cluster/controller/db.py` (delegate writes to Rust via PyO3)

**Rust crates**: `rusqlite`

**Python interface**: `#[pyfunction]` for each write operation, called from `ControllerDB` methods.

**Tests**: Integration test: Rust writes → Python reads and vice versa. All existing `tests/cluster/controller/test_db.py` pass with Rust backend.

**Dependencies**: Step 9.

---

## Phase 3: RPC Layer (Steps 11–15)

### Step 11: Connect-RPC Framework Crate

**What**: Build the `iris-connect` Rust crate — a Connect-RPC server adapter on `axum`.

**Source files**:
- New: `lib/iris-native/src/connect/mod.rs`, `lib/iris-native/src/connect/handler.rs`, `lib/iris-native/src/connect/error.rs`, `lib/iris-native/src/connect/codec.rs`

**Rust crates**: `axum 0.8`, `hyper 1.0`, `tokio 1`, `prost`, `prost-reflect` (for JSON encoding), `serde_json`

**Python interface**: None — this is Rust-internal infrastructure.

**Tests**: Rust integration tests that start an axum server with a dummy service, send Connect-RPC requests (both binary protobuf and JSON), and verify correct responses. Test error code mapping (NOT_FOUND, ALREADY_EXISTS, INTERNAL, etc.).

**Dependencies**: Step 2.

---

### Step 12: Auth Middleware

**What**: Port JWT authentication from `lib/iris/src/iris/cluster/controller/auth.py:1-387` and `lib/iris/src/iris/rpc/auth.py`.

**Source files**:
- New: `lib/iris-native/src/auth.rs`
- Read: `lib/iris/src/iris/cluster/controller/auth.py` (JWT signing key management, token validation)
- Read: `lib/iris/src/iris/rpc/auth.py` (RPC interceptor, token providers)

**Rust crates**: `jsonwebtoken 9`, `axum` (middleware layer)

**Python interface**: `#[pyfunction]` `verify_jwt(token: str, signing_key: bytes) -> dict` for use in the Python controller during transition. Also used as `axum` middleware in the Rust controller.

**Tests**: Rust tests for JWT create/verify roundtrip. Python test: create JWT with Python, verify with Rust and vice versa. Test GCP identity token validation flow.

**Dependencies**: Step 11.

---

### Step 13: ControllerService Stubs (Read-Only RPCs)

**What**: Implement read-only RPCs in Rust: `GetJobStatus`, `ListJobs`, `GetTaskStatus`, `ListTasks`, `ListWorkers`, `ListEndpoints`, `GetAutoscalerStatus`, `GetTransactions`, `ListUsers`.

**Source files**:
- New: `lib/iris-native/src/controller_service.rs`
- Read: `lib/iris/src/iris/cluster/controller/service.py:1-1759` (RPC implementations)

**Rust crates**: `axum`, `prost`, `rusqlite`, `iris-connect` (from Step 11)

**Python interface**: None — these are served by the Rust binary directly.

**Tests**: For each RPC: populate SQLite DB with known state, send Connect-RPC request, verify response matches expected proto. Run against both Python and Rust servers with identical DB state.

**Dependencies**: Step 10 (DB reads), Step 11 (Connect framework), Step 12 (auth).

---

### Step 14: ControllerService Write RPCs

**What**: Implement write RPCs: `LaunchJob`, `TerminateJob`, `Register`, `RegisterEndpoint`, `UnregisterEndpoint`, `Login`, `CreateApiKey`, `RevokeApiKey`, `BeginCheckpoint`.

**Source files**:
- Extended: `lib/iris-native/src/controller_service.rs`
- New: `lib/iris-native/src/controller_service/launch.rs`, `lib/iris-native/src/controller_service/worker_mgmt.rs`

**Rust crates**: Same as Step 13.

**Python interface**: None.

**Tests**: End-to-end: Python client → Rust controller → verify DB state. Specifically: submit job, check status, terminate job, verify state transitions. All existing `tests/cluster/controller/test_service.py` tests adapted to run against Rust server.

**Dependencies**: Step 13.

---

### Step 15: WorkerService in Rust

**What**: Implement the WorkerService RPCs: `Heartbeat`, `GetTaskStatus`, `ListTasks`, `HealthCheck`, `FetchLogs`, `ProfileTask`, `GetProcessStatus`.

**Source files**:
- New: `lib/iris-native/src/worker_service.rs`
- Read: `lib/iris/src/iris/cluster/worker/service.py`

**Rust crates**: `axum`, `prost`, `tokio`, `iris-connect`

**Python interface**: None — the Rust worker service binary handles heartbeats directly.

**Tests**: Mock task execution, verify heartbeat request/response proto compatibility. Test that Python controller can heartbeat a Rust worker and vice versa.

**Dependencies**: Step 11, Step 2.

---

## Phase 4: Core Logic (Steps 16–20)

### Step 16: Scheduler in Rust

**What**: Port the pure scheduler from `lib/iris/src/iris/cluster/controller/scheduler.py:1-840`.

**Source files**:
- New: `lib/iris-native/src/scheduler.rs`
- Modified: `lib/iris/src/iris/cluster/controller/scheduler.py` (delegate to Rust via PyO3)

**Rust crates**: `pyo3`

**Python interface**: `#[pyfunction]` `schedule(context: SchedulingContext) -> list[Assignment]` where `SchedulingContext` is constructed from Python `WorkerSnapshot` objects. The Python scheduler module becomes a thin wrapper.

**Tests**: Port all cases from `tests/cluster/controller/test_scheduler.py` to Rust. Python integration test: run scheduler with identical input on both backends, assert identical assignments (order-independent).

**Dependencies**: Step 5 (constraints), Step 6 (JobName), Step 9 (WorkerRow for snapshot construction).

---

### Step 17: State Transitions in Rust

**What**: Port the state machine from `lib/iris/src/iris/cluster/controller/transitions.py:1-1679`.

**Source files**:
- New: `lib/iris-native/src/transitions.rs`
- Read: `lib/iris/src/iris/cluster/controller/transitions.py`

**Rust crates**: `pyo3`, `rusqlite`

**Python interface**: `#[pyfunction]` for each transition function: `apply_heartbeat`, `apply_task_completion`, `apply_job_timeout`, etc. Called from `Controller.tick()`.

**Tests**: Port `tests/cluster/controller/test_transitions.py`. Critical: test every state edge (PENDING→RUNNING, RUNNING→SUCCEEDED, RUNNING→FAILED, RUNNING→WORKER_FAILED, etc.).

**Dependencies**: Step 10 (DB writes), Step 16 (scheduler for assignment application).

---

### Step 18: Scaling Group Management

**What**: Port scaling group logic from `lib/iris/src/iris/cluster/controller/scaling_group.py:1-1275`.

**Source files**:
- New: `lib/iris-native/src/scaling_group.rs`
- Read: `lib/iris/src/iris/cluster/controller/scaling_group.py`

**Rust crates**: `pyo3`, `prost`

**Python interface**: `#[pyclass]` for `ScalingGroup` with methods for demand computation, waterfall routing, and VM lifecycle tracking.

**Tests**: Port `tests/cluster/platform/test_scaling_group.py`. Test demand computation with various job mixes.

**Dependencies**: Step 5 (constraints), Step 7 (ResourceSpec), Step 2 (config protos).

---

### Step 19: Autoscaler in Rust

**What**: Port the autoscaler from `lib/iris/src/iris/cluster/controller/autoscaler.py:1-1566`.

**Source files**:
- New: `lib/iris-native/src/autoscaler.rs`
- Read: `lib/iris/src/iris/cluster/controller/autoscaler.py`

**Rust crates**: `pyo3`, `tokio` (for async platform calls)

**Python interface**: `#[pyclass]` for `Autoscaler` with `evaluate()` method. During transition, platform calls are delegated back to Python via PyO3 callback.

**Tests**: Port `tests/cluster/controller/test_autoscaler.py`. Test scale-up/scale-down decisions with mock platform responses.

**Dependencies**: Step 18 (scaling groups), Step 17 (transitions for VM lifecycle).

---

### Step 20: Controller Coordinator in Rust

**What**: Port the main controller loop from `lib/iris/src/iris/cluster/controller/controller.py:1-1646`.

**Source files**:
- New: `lib/iris-native/src/controller.rs`
- Read: `lib/iris/src/iris/cluster/controller/controller.py`

**Rust crates**: `tokio`, `pyo3`

**Python interface**: `#[pyclass]` for `RustController` with `tick()` and `start()`/`stop()` methods. The Python `Controller` class delegates to `RustController` for scheduling and transitions, keeping platform/autoscaler calls in Python initially.

**Tests**: Integration test using `LocalCluster` harness: start cluster with Rust controller, submit job, verify execution. All existing `tests/cluster/controller/test_job.py` must pass.

**Dependencies**: Step 16 (scheduler), Step 17 (transitions), Step 19 (autoscaler), Step 14 (RPC service).

---

## Phase 5: Platform & Runtime (Steps 21–25)

### Step 21: Platform Abstraction Trait

**What**: Define the `Platform` trait in Rust mirroring the Protocol in `lib/iris/src/iris/cluster/platform/base.py:1-587`.

**Source files**:
- New: `lib/iris-native/src/platform/mod.rs`, `lib/iris-native/src/platform/traits.rs`
- Read: `lib/iris/src/iris/cluster/platform/base.py` (Platform Protocol, SliceInfo, VmInfo)

**Rust crates**: `async-trait`, `tokio`

**Python interface**: None initially. The trait defines the Rust-side abstraction. Platform implementations that call cloud APIs will be added in subsequent steps.

**Tests**: Compile-time verification that the trait is implementable. A `MockPlatform` struct for testing.

**Dependencies**: Step 2 (config protos), Step 7 (ResourceSpec).

---

### Step 22: Local Platform Implementation

**What**: Port `LocalPlatform` from `lib/iris/src/iris/cluster/platform/local.py:1-581`.

**Source files**:
- New: `lib/iris-native/src/platform/local.rs`
- Read: `lib/iris/src/iris/cluster/platform/local.py`

**Rust crates**: `tokio::process`, `sysinfo`

**Python interface**: `#[pyclass]` for `RustLocalPlatform` implementing the Platform trait. Used for testing and local development.

**Tests**: Test VM create/delete/list lifecycle on localhost. Integration test: run `LocalCluster` with Rust platform backend.

**Dependencies**: Step 21.

---

### Step 23: GCP Platform Implementation

**What**: Port GCP VM/TPU management from `lib/iris/src/iris/cluster/platform/gcp.py:1-1715`.

**Source files**:
- New: `lib/iris-native/src/platform/gcp.rs`
- Read: `lib/iris/src/iris/cluster/platform/gcp.py`

**Rust crates**: `reqwest` (for GCE/TPU REST APIs), `google-cloud-auth`, `tokio`, `serde_json`

**Python interface**: `#[pyclass]` for `RustGcpPlatform`. Initially, complex operations (TPU slice management) can delegate to Python via PyO3 callback while simpler operations (VM create/delete) are native Rust.

**Tests**: Unit tests with mock HTTP responses for VM create/delete/list. Integration test against GCP (gated behind `IRIS_GCP_INTEGRATION_TEST` env var).

**Dependencies**: Step 21.

---

### Step 24: Runtime Abstraction and Process Runtime

**What**: Port the `Runtime` abstraction and `ProcessRuntime` from `lib/iris/src/iris/cluster/runtime/types.py` and `lib/iris/src/iris/cluster/runtime/process.py:1-671`.

**Source files**:
- New: `lib/iris-native/src/runtime/mod.rs`, `lib/iris-native/src/runtime/process.rs`
- Read: `lib/iris/src/iris/cluster/runtime/process.py`, `lib/iris/src/iris/cluster/runtime/types.py`

**Rust crates**: `tokio::process`, `nix` (signal handling)

**Python interface**: `#[pyclass]` for `RustProcessRuntime` with `start_task()`, `kill_task()`, `get_status()`.

**Tests**: Port `tests/cluster/runtime/test_entrypoint.py` cases. Test process lifecycle: start, monitor, kill, timeout.

**Dependencies**: Step 7 (ResourceSpec), Step 6 (JobName).

---

### Step 25: Docker Runtime

**What**: Port Docker container management from `lib/iris/src/iris/cluster/runtime/docker.py:1-1013`.

**Source files**:
- New: `lib/iris-native/src/runtime/docker.rs`
- Read: `lib/iris/src/iris/cluster/runtime/docker.py`

**Rust crates**: `bollard` (Docker API), `tokio`

**Python interface**: `#[pyclass]` for `RustDockerRuntime` with same interface as `ProcessRuntime`.

**Tests**: Port `tests/cluster/runtime/test_docker_runtime.py`. Test container create/start/stop/logs lifecycle. Requires Docker daemon (CI must have Docker).

**Dependencies**: Step 24.

---

## Phase 6: Client & CLI (Steps 26–30+)

### Step 26: Rust Worker Binary

**What**: Build a standalone Rust binary that runs the full worker agent — combining WorkerService (Step 15), runtime (Steps 24-25), and env probe.

**Source files**:
- New: `lib/iris-native/src/bin/iris-worker.rs`
- Read: `lib/iris/src/iris/cluster/worker/worker.py`, `lib/iris/src/iris/cluster/worker/main.py`

**Rust crates**: `tokio`, `axum`, `clap`, `tracing`

**Python interface**: None — standalone binary replaces `python -m iris.cluster.worker.main`.

**Tests**: Integration test: Rust worker registers with Python controller, receives heartbeat, executes process task, reports completion. Test with Docker runtime.

**Dependencies**: Step 15, Step 24, Step 25.

---

### Step 27: Rust Controller Binary

**What**: Build a standalone Rust binary that runs the full controller — combining ControllerService (Steps 13-14), scheduler (Step 16), transitions (Step 17), autoscaler (Step 19), DB (Steps 8-10).

**Source files**:
- New: `lib/iris-native/src/bin/iris-controller.rs`
- Read: `lib/iris/src/iris/cluster/controller/main.py`

**Rust crates**: `tokio`, `axum`, `clap`, `tracing`, `rusqlite`

**Python interface**: None — standalone binary replaces `python -m iris.cluster.controller.main`.

**Tests**: Full integration: Rust controller + Rust worker. Submit job via Python client, verify execution. Run the full `tests/cluster/controller/` test suite against Rust controller.

**Dependencies**: Steps 13-20.

---

### Step 28: CLI in Rust

**What**: Port the CLI from `lib/iris/src/iris/cli/` (~3600 LOC) to Rust using `clap`.

**Source files**:
- New: `lib/iris-native/src/bin/iris.rs`, `lib/iris-native/src/cli/`
- Read: `lib/iris/src/iris/cli/main.py`, `job.py`, `cluster.py`, `query.py`, `build.py`, `rpc.py`

**Rust crates**: `clap 4` (derive), `tabled` (table formatting), `tokio`, `reqwest`

**Python interface**: None — replaces `python -m iris.cli.main`.

**Tests**: CLI integration tests: `iris job list`, `iris job submit`, `iris cluster status`. Compare output format with Python CLI.

**Dependencies**: Step 27 (needs running controller to test against).

---

### Step 29: Python Client Library Compatibility Layer

**What**: Ensure the Python client library (`lib/iris/src/iris/client/client.py:1-1092`) works seamlessly with Rust controller. Add any missing wire-format compatibility fixes.

**Source files**:
- Modified: `lib/iris/src/iris/client/client.py` (if any fixes needed)
- Modified: `lib/iris/src/iris/cluster/client/remote_client.py`

**Rust crates**: N/A (this is Python-side)

**Python interface**: The existing `IrisClient` class, unchanged.

**Tests**: Full client test suite against Rust controller. Test all client methods: `submit()`, `get_status()`, `terminate()`, `list_jobs()`, `get_logs()`, `wait_for_completion()`.

**Dependencies**: Step 27.

---

### Step 30: Actor System RPC Passthrough

**What**: Implement ActorService (`actor.proto:108-118`) in Rust as a bytes passthrough — Rust handles the Connect-RPC transport but delegates serialization/deserialization to Python cloudpickle.

**Source files**:
- New: `lib/iris-native/src/actor_service.rs`
- Read: `lib/iris/src/iris/actor/server.py`, `lib/iris/src/iris/actor/client.py`

**Rust crates**: `axum`, `prost`, `iris-connect`

**Python interface**: The actor server runs in a Python process. The Rust Connect-RPC handler receives `ActorCall` messages, forwards `serialized_args`/`serialized_kwargs` bytes to the Python process (via a local socket or PyO3 callback), and returns the `ActorResponse`.

**Tests**: End-to-end actor test: Python client → Rust RPC layer → Python actor → response. All existing actor tests pass.

**Dependencies**: Step 11.

---

### Step 31: Kubernetes Runtime

**What**: Port Kubernetes runtime from `lib/iris/src/iris/cluster/runtime/kubernetes.py:1-696` and CoreWeave platform from `lib/iris/src/iris/cluster/platform/coreweave.py:1-1577`.

**Source files**:
- New: `lib/iris-native/src/runtime/kubernetes.rs`, `lib/iris-native/src/platform/coreweave.rs`
- Read: `lib/iris/src/iris/cluster/runtime/kubernetes.py`, `lib/iris/src/iris/cluster/platform/coreweave.py`

**Rust crates**: `kube-rs`, `k8s-openapi`, `tokio`

**Python interface**: `#[pyclass]` for `RustKubernetesRuntime` and `RustCoreweavePlatform`.

**Tests**: Port `tests/cluster/runtime/test_kubernetes_runtime.py` and `tests/cluster/platform/test_coreweave_platform.py`.

**Dependencies**: Step 24 (runtime trait), Step 21 (platform trait).

---

### Step 32: Dashboard Integration

**What**: Port dashboard endpoints from `lib/iris/src/iris/cluster/controller/dashboard.py` and `lib/iris/src/iris/cluster/worker/dashboard.py`.

**Source files**:
- New: `lib/iris-native/src/dashboard.rs`
- Read: Controller and worker dashboard files

**Rust crates**: `axum`, `askama` or `maud` (HTML templating), `tokio`

**Python interface**: None — served directly by Rust controller/worker.

**Tests**: HTTP test: GET dashboard endpoints, verify HTML response contains expected elements.

**Dependencies**: Step 27.

---

### Step 33: Remove Python Controller and Worker

**What**: Delete the Python controller and worker implementations. The Rust binaries are now the sole implementation.

**Source files**:
- Deleted: `lib/iris/src/iris/cluster/controller/controller.py`, `service.py`, `main.py`
- Deleted: `lib/iris/src/iris/cluster/worker/worker.py`, `service.py`, `main.py`
- Modified: Deployment configs, Dockerfiles, CI

**Rust crates**: N/A

**Python interface**: Python client library (`iris.client`) remains. CLI is Rust. Controller/worker are Rust.

**Tests**: Full test suite passes with only Rust binaries. No Python controller/worker code remains.

**Dependencies**: Steps 26-32. All tests passing on Rust binaries.

---

## Dependency Graph

```
Step 1 ─→ Step 2 ─→ Step 3
  │         │ │       │
  │         │ ├→ Step 4 → Step 5
  │         │ │               │
  │         │ └→ Step 7       │
  │         │     │           │
  ├→ Step 6 │     │           │
  │    │    │     │           │
  │    └────┼─────┼───→ Step 9 → Step 10
  │         │     │      │         │
  ├→ Step 8─┘     │      │         │
  │               │      │         │
  │    Step 11 ←──┘      │         │
  │      │               │         │
  │      ├→ Step 12      │         │
  │      │    │          │         │
  │      ├────┼→ Step 13─┘─────────┘
  │      │    │    │
  │      │    │    └→ Step 14
  │      │    │         │
  │      └→ Step 15     │
  │           │         │
  │    Step 16 ←────────┤ (needs 5, 6, 9)
  │      │              │
  │    Step 17 ←────────┤ (needs 10, 16)
  │      │              │
  │    Step 18          │ (needs 5, 7)
  │      │              │
  │    Step 19 ←── Step 18, Step 17
  │      │
  │    Step 20 ←── Steps 14, 16, 17, 19
  │      │
  │    Step 21 (needs 2, 7)
  │      │
  │      ├→ Step 22
  │      ├→ Step 23
  │      │
  │    Step 24 (needs 7, 6)
  │      │
  │      └→ Step 25
  │           │
  │    Step 26 ←── Steps 15, 24, 25
  │    Step 27 ←── Steps 13-20
  │    Step 28 ←── Step 27
  │    Step 29 ←── Step 27
  │    Step 30 ←── Step 11
  │    Step 31 ←── Steps 24, 21
  │    Step 32 ←── Step 27
  │    Step 33 ←── Steps 26-32
```

## Timeline Estimate

| Phase | Steps | Estimated Duration | Parallelism |
|-------|-------|--------------------|-------------|
| Phase 1 | 1–5 | 4–6 weeks | Steps 4-5 parallel with Step 3 |
| Phase 2 | 6–10 | 4–6 weeks | Steps 6-7 parallel; Steps 9-10 sequential |
| Phase 3 | 11–15 | 6–8 weeks | Step 11 first; Steps 13-15 partially parallel |
| Phase 4 | 16–20 | 6–8 weeks | Steps 16-18 partially parallel; 19-20 sequential |
| Phase 5 | 21–25 | 4–6 weeks | Steps 22-23 parallel; Steps 24-25 sequential |
| Phase 6 | 26–33 | 6–8 weeks | Steps 26-27 parallel; 28-32 partially parallel |
| **Total** | | **~8–12 months** | |

## Milestones

1. **M1 (end of Phase 1)**: `iris._native` module builds and passes CI. Time utils and constraint evaluation run in Rust. Measurable speedup on scheduler benchmarks.

2. **M2 (end of Phase 2)**: Rust can read and write the controller SQLite database. Schema compatibility verified.

3. **M3 (end of Phase 3)**: A Rust binary serves ControllerService RPCs. Python clients work against it. Shadow mode deployed.

4. **M4 (end of Phase 4)**: Scheduler, transitions, and autoscaler run in Rust. Controller loop is Rust-native.

5. **M5 (end of Phase 5)**: Worker agent runs as a Rust binary with Docker and process runtimes.

6. **M6 (end of Phase 6)**: Full Rust deployment. Python code remains only for client library and actor system.
