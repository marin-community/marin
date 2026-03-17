# Iris-to-Rust Migration: Step-by-Step Plan

Each step is a single PR-sized change. The system works after every step with a mix of Rust and Python.

---

## Phase 1: Foundation (Steps 1–6, 6b)

### Step 1: Rust Project Scaffolding

**What**: Create `lib/iris/rust/` with Cargo.toml, pyproject.toml, and build integration.

**Source files**:
- New: `lib/iris/rust/Cargo.toml`, `lib/iris/rust/pyproject.toml`, `lib/iris/rust/src/lib.rs`
- Modified: root `pyproject.toml` (workspace member), `Cargo.toml` (workspace)

**Rust crates**: `pyo3 0.26` (abi3-py311), `prost 0.13`, `prost-build`

**Python interface**: `from iris._native import hello_world` — a trivial test function that returns a string, verifying the build pipeline works end-to-end.

**Tests**: `tests/test_native_import.py` — import `iris._native`, call `hello_world()`, assert return value. CI build verifies maturin compilation.

**Dependencies**: None.

---

### Step 2: Dev/User Mode Build Infrastructure

**What**: Set up dual build modes for iris-rust. Dev mode: `make iris-dev` runs `maturin develop --release` to build and install the native extension locally. User mode: `make iris-user` or `pip install iris[native]` installs a pre-built wheel from GitHub Releases. No Rust toolchain needed for user mode.

**Source files**:
- New: `Makefile` targets (`iris-dev`, `iris-user`)
- New: `lib/iris/rust/pyproject.toml` (maturin config with `[tool.maturin]` settings)
- New: `.github/workflows/iris-rust-wheels.yml` (CI wheel build + Docker image push)
- New: `lib/iris/src/iris/_native_loader.py` (importlib-based loader with fallback)

**Env var**: `IRIS_BUILD_MODE=dev|user` — `dev` triggers `maturin develop` on `make iris-dev`; `user` skips Rust compilation and installs from wheel.

**Detection logic** in `_native_loader.py`:

```python
import logging
import os

logger = logging.getLogger(__name__)

def load_native():
    try:
        import iris._native
        return iris._native
    except ImportError:
        if os.environ.get("IRIS_USE_NATIVE", "1") != "0":
            logger.warning(
                "iris._native not found. Run 'make iris-dev' to build from source "
                "or 'pip install iris[native]' for the pre-built wheel."
            )
        return None
```

**CI**: GitHub Actions workflow using `PyO3/maturin-action@v1` to build wheels on every push to `main` that touches `lib/iris/rust/` (not just tags). Targets: `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`, `x86_64-apple-darwin`, `aarch64-apple-darwin`. Wheels uploaded to GitHub Releases with `latest` tag. `make iris-user` downloads the latest wheel from GitHub Releases via `gh release download` — no PyPI needed. Tagged releases (`iris-rust-v*`) additionally publish to PyPI. Docker images with the Rust extension baked in are pushed to the container registry on each merge.

**Tests**:
- `make iris-dev` produces a working `iris._native` module (import succeeds, `hello_world()` returns expected value)
- Importing without the native extension falls back cleanly (no crash, warning logged)
- CI wheel artifact is installable in a clean venv

**Dependencies**: Step 1 (needs the Cargo project to exist).

---

### Step 3: Proto Codegen in Rust

**What**: Generate Rust types from the existing `.proto` files using `prost-build`.

**Source files**:
- New: `lib/iris/rust/build.rs` (prost-build script), `lib/iris/rust/src/proto/` (generated)
- Read: `lib/iris/src/iris/rpc/time.proto`, `errors.proto`, `logging.proto`, `query.proto`, `vm.proto`, `config.proto`, `cluster.proto`, `actor.proto`

**Rust crates**: `prost 0.13`, `prost-build 0.13`, `prost-types`

**Python interface**: None directly. This step generates Rust types that later steps expose.

**Tests**: Rust `#[cfg(test)]` tests that construct each major proto message type (Timestamp, Duration, JobStatus, ResourceSpecProto, LaunchJobRequest) and verify roundtrip encode/decode via `prost::Message`.

**Dependencies**: Step 1.

---

### Step 4: Time Utilities in Rust

**What**: Port `Timestamp`, `Duration`, `Deadline`, `Timer`, `RateLimiter`, `TokenBucket`, `ExponentialBackoff` from `lib/iris/src/iris/time_utils.py:1-556`.

**Source files**:
- New: `lib/iris/rust/src/time_utils.rs`
- Modified: `lib/iris/src/iris/time_utils.py` (replace implementation with re-exports from `iris._native`)

**Rust crates**: `pyo3`, `prost` (for proto conversion), `std::time` (Instant, SystemTime)

**Python interface**: PyO3 `#[pyclass]` for each type. All existing methods exposed as `#[pymethods]`. The Python module re-exports from `iris._native`:
```python
# time_utils.py — the Python implementation is deleted, not preserved
from iris._native import Timestamp, Duration, Deadline, Timer, RateLimiter, TokenBucket, ExponentialBackoff
```

**Tests**:
- Rust unit tests mirroring `tests/test_time_utils.py` cases
- Existing `tests/` that use `time_utils` pass unchanged (they now exercise the Rust implementation)

**Dependencies**: Step 3 (proto types for `to_proto()`/`from_proto()` methods).

---

### Step 5: Core Enum and Value Types

**What**: Port `AttributeValue`, `ConstraintOp`, `WellKnownAttribute`, `DeviceType` from `lib/iris/src/iris/cluster/constraints.py:30-145`.

**Source files**:
- New: `lib/iris/rust/src/types.rs`
- Modified: `lib/iris/src/iris/cluster/constraints.py` (import shim for ported types)

**Rust crates**: `pyo3`, `prost`

**Python interface**: `#[pyclass]` with `#[pymethods]` for `to_proto()`, `from_proto()`, comparison operators, hash. `WellKnownAttribute` as a Rust enum with `#[pyclass]`.

**Tests**:
- Rust tests for AttributeValue proto roundtrips, ConstraintOp mapping
- Python test asserting `_native.AttributeValue(42).to_proto()` matches Python version

**Dependencies**: Step 3.

---

### Step 6: Constraint Evaluation Engine

**What**: Port `Constraint`, `evaluate_constraint`, `check_resource_fit`, `ConstraintIndex`, `ResourceCapacity` from `lib/iris/src/iris/cluster/constraints.py:145-end`.

**Source files**:
- New: `lib/iris/rust/src/constraints.rs`
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

**Dependencies**: Step 5.

---

### Step 6b: Remove Python Leaf Module Implementations

**What**: Delete the Python implementations of `time_utils`, `constraints`, and enum types that are now fully served by Rust. The Python modules become thin re-export shims (`from iris._native import ...`). CI runs the full test suite against `make iris-user` (pre-built wheel) to verify user-mode works.

**Source files**:
- Modified: `lib/iris/src/iris/time_utils.py` — strip to re-exports only, delete all class/function bodies
- Modified: `lib/iris/src/iris/cluster/constraints.py` — strip to re-exports for ported types and functions
- Deleted: Any `_*_py.py` fallback copies if they were created

**Tests**: Full test suite passes with `IRIS_USE_NATIVE=1` (required, no fallback). CI builds wheel, installs in clean venv, runs tests.

**Dependencies**: Steps 4, 5, 6 all passing.

---

## Phase 2: Data Layer (Steps 7–11, 11b)

### Step 7: JobName and WorkerId Types

**What**: Port `JobName`, `WorkerId`, `Namespace` from `lib/iris/src/iris/cluster/types.py:29-200`.

**Source files**:
- New: `lib/iris/rust/src/job_name.rs`
- Modified: `lib/iris/src/iris/cluster/types.py` (import shim)

**Rust crates**: `pyo3`

**Python interface**: `#[pyclass]` for `JobName` with all methods: `from_string()`, `child()`, `task()`, `parent`, `user`, `root_job`, `namespace`, `is_root`, `task_index`, `depth`, `wire_format`. `WorkerId` as a newtype wrapper.

**Tests**: Rust tests for parsing, hierarchy navigation, wire format roundtrips. Python test asserting `_native.JobName.from_string("/alice/job")` matches Python `JobName.from_string("/alice/job")`.

**Dependencies**: Step 1.

---

### Step 8: ResourceSpec and EnvironmentSpec

**What**: Port `ResourceSpec`, `EnvironmentSpec` from `lib/iris/src/iris/cluster/types.py:200-500` (excluding `Entrypoint` which uses cloudpickle).

**Source files**:
- New: `lib/iris/rust/src/resource_spec.rs`
- Modified: `lib/iris/src/iris/cluster/types.py` (import shim for ResourceSpec, EnvironmentSpec)

**Rust crates**: `pyo3`, `prost`, `humansize` (for human-readable byte parsing)

**Python interface**: `#[pyclass]` with `to_proto()`, `from_proto()` and all builder methods. `Entrypoint` stays in Python.

**Tests**: Rust tests for proto conversion, human-readable parsing ("4GB" -> bytes). Python test asserting `ResourceSpec(cpu=4, memory="8GB").to_proto()` matches.

**Dependencies**: Step 3, Step 5.

---

### Step 9: SQLite DB Foundation

**What**: Create the Rust DB access layer — connection management, read snapshots, migration runner. Port the `Predicate` query builder.

**Source files**:
- New: `lib/iris/rust/src/db.rs`, `lib/iris/rust/src/db/predicate.rs`, `lib/iris/rust/src/db/migrations.rs`
- Read: `lib/iris/src/iris/cluster/controller/db.py:1-350` (Predicate, db_row_model, ControllerDB connection handling)
- Read: `lib/iris/src/iris/cluster/controller/migrations/` (all 9 migration files)

**Rust crates**: `rusqlite 0.32` (bundled), `pyo3`

**Python interface**: Not exposed via PyO3 yet. This step builds the Rust-internal DB layer that later steps use. A `#[pyfunction]` `verify_db_schema(path: &str) -> bool` is exposed for testing.

**Tests**: Rust tests that create an in-memory SQLite DB, run all migrations, insert test data, and query using the Predicate builder. Python test that creates a DB via Python, then calls `verify_db_schema()` from Rust to confirm compatibility.

**Dependencies**: Step 1.

---

### Step 10: DB Row Models in Rust

**What**: Port the typed row models (`JobRow`, `TaskRow`, `WorkerRow`, `EndpointRow`) from `lib/iris/src/iris/cluster/controller/db.py:350-end`.

**Source files**:
- New: `lib/iris/rust/src/db/models.rs`
- Read: `lib/iris/src/iris/cluster/controller/db.py:350-1334`

**Rust crates**: `rusqlite`, `pyo3`, `serde`

**Python interface**: `#[pyclass]` for each row model with field accessors. A `read_workers(db_path: str) -> list[WorkerRow]` function exposed via PyO3 for the scheduler to consume.

**Tests**: Roundtrip test: Python inserts rows → Rust reads them → assert field values match. Rust inserts rows → Python reads them → assert match.

**Dependencies**: Step 9, Step 7, Step 4 (Timestamp fields in rows).

---

### Step 11: DB Write Operations

**What**: Port write operations: `insert_job`, `update_task_state`, `insert_worker`, `upsert_endpoint`, `insert_log_entries`, transaction support.

**Source files**:
- Extended: `lib/iris/rust/src/db.rs`, `lib/iris/rust/src/db/models.rs`
- Modified: `lib/iris/src/iris/cluster/controller/db.py` (delegate writes to Rust via PyO3)

**Rust crates**: `rusqlite`

**Python interface**: `#[pyfunction]` for each write operation, called from `ControllerDB` methods.

**Tests**: Integration test: Rust writes → Python reads and vice versa. All existing `tests/cluster/controller/test_db.py` pass with Rust backend.

**Dependencies**: Step 10.

---

### Step 11b: Remove Python DB Layer

**What**: Delete the Python `ControllerDB` implementation, `db_row_model` decorator, and `Predicate` query builder. The Python `db.py` becomes a thin wrapper that delegates all operations to `iris._native` DB functions. Remove the custom Python ORM code entirely.

**Source files**:
- Modified: `lib/iris/src/iris/cluster/controller/db.py` — strip to Rust delegation shim
- Modified: `lib/iris/src/iris/cluster/types.py` — strip `JobName`, `WorkerId`, `ResourceSpec`, `EnvironmentSpec` to re-exports

**Tests**: All `tests/cluster/controller/test_db.py` pass. All downstream tests (scheduler, transitions, service) pass.

**Dependencies**: Steps 7–11 all passing.

---

## Phase 3: RPC Layer (Steps 12–16, 16b)

### Step 12: Connect-RPC Framework Crate

**What**: Build the `iris-connect` Rust crate — a Connect-RPC server adapter on `axum`.

**Source files**:
- New: `lib/iris/rust/src/connect/mod.rs`, `lib/iris/rust/src/connect/handler.rs`, `lib/iris/rust/src/connect/error.rs`, `lib/iris/rust/src/connect/codec.rs`

**Rust crates**: `axum 0.8`, `hyper 1.0`, `tokio 1`, `prost`, `prost-reflect` (for JSON encoding), `serde_json`

**Python interface**: None — this is Rust-internal infrastructure.

**Tests**: Rust integration tests that start an axum server with a dummy service, send Connect-RPC requests (both binary protobuf and JSON), and verify correct responses. Test error code mapping (NOT_FOUND, ALREADY_EXISTS, INTERNAL, etc.).

**Dependencies**: Step 3.

---

### Step 13: Auth Middleware

**What**: Port JWT authentication from `lib/iris/src/iris/cluster/controller/auth.py:1-387` and `lib/iris/src/iris/rpc/auth.py`.

**Source files**:
- New: `lib/iris/rust/src/auth.rs`
- Read: `lib/iris/src/iris/cluster/controller/auth.py` (JWT signing key management, token validation)
- Read: `lib/iris/src/iris/rpc/auth.py` (RPC interceptor, token providers)

**Rust crates**: `jsonwebtoken 9`, `axum` (middleware layer)

**Python interface**: `#[pyfunction]` `verify_jwt(token: str, signing_key: bytes) -> dict` for use in the Python controller during transition. Also used as `axum` middleware in the Rust controller.

**Tests**: Rust tests for JWT create/verify roundtrip. Python test: create JWT with Python, verify with Rust and vice versa. Test GCP identity token validation flow.

**Dependencies**: Step 12.

---

### Step 14: ControllerService Stubs (Read-Only RPCs)

**What**: Implement read-only RPCs in Rust: `GetJobStatus`, `ListJobs`, `GetTaskStatus`, `ListTasks`, `ListWorkers`, `ListEndpoints`, `GetAutoscalerStatus`, `GetTransactions`, `ListUsers`.

**Source files**:
- New: `lib/iris/rust/src/controller_service.rs`
- Read: `lib/iris/src/iris/cluster/controller/service.py:1-1759` (RPC implementations)

**Rust crates**: `axum`, `prost`, `rusqlite`, `iris-connect` (from Step 12)

**Python interface**: None — these are served by the Rust binary directly.

**Tests**: For each RPC: populate SQLite DB with known state, send Connect-RPC request, verify response matches expected proto. Run against both Python and Rust servers with identical DB state.

**Dependencies**: Step 11 (DB reads), Step 12 (Connect framework), Step 13 (auth).

---

### Step 15: ControllerService Write RPCs

**What**: Implement write RPCs: `LaunchJob`, `TerminateJob`, `Register`, `RegisterEndpoint`, `UnregisterEndpoint`, `Login`, `CreateApiKey`, `RevokeApiKey`, `BeginCheckpoint`.

**Source files**:
- Extended: `lib/iris/rust/src/controller_service.rs`
- New: `lib/iris/rust/src/controller_service/launch.rs`, `lib/iris/rust/src/controller_service/worker_mgmt.rs`

**Rust crates**: Same as Step 14.

**Python interface**: None.

**Tests**: End-to-end: Python client → Rust controller → verify DB state. Specifically: submit job, check status, terminate job, verify state transitions. All existing `tests/cluster/controller/test_service.py` tests adapted to run against Rust server.

**Dependencies**: Step 14.

---

### Step 16: WorkerService in Rust

**What**: Implement the WorkerService RPCs: `Heartbeat`, `GetTaskStatus`, `ListTasks`, `HealthCheck`, `FetchLogs`, `ProfileTask`, `GetProcessStatus`.

**Source files**:
- New: `lib/iris/rust/src/worker_service.rs`
- Read: `lib/iris/src/iris/cluster/worker/service.py`

**Rust crates**: `axum`, `prost`, `tokio`, `iris-connect`

**Python interface**: None — the Rust worker service binary handles heartbeats directly.

**Tests**: Mock task execution, verify heartbeat request/response proto compatibility. Test that Python controller can heartbeat a Rust worker and vice versa.

**Dependencies**: Step 12, Step 3.

---

### Step 16b: Remove Python RPC Service Implementations

**What**: Delete the Python `ControllerService` and `WorkerService` RPC handler implementations. The Rust binary now serves all RPCs. Python `service.py` files are deleted entirely. Update Dockerfiles and deployment configs to use the Rust controller/worker binaries. Push updated Docker images to the container registry.

**Source files**:
- Deleted: `lib/iris/src/iris/cluster/controller/service.py` (Python RPC handlers)
- Deleted: `lib/iris/src/iris/cluster/worker/service.py` (Python RPC handlers)
- Modified: `lib/iris/src/iris/cluster/controller/auth.py` — strip to re-exports from `iris._native`
- Modified: Dockerfile(s) — switch entrypoint from Python to Rust binary
- Modified: `.github/workflows/` — add Docker image build+push step

**Tests**: Full test suite runs against Rust-served RPCs. CI builds and pushes Docker image with Rust binary. `make iris-user` installs the latest wheel and Docker image works out of the box.

**Dependencies**: Steps 12–16 all passing.

---

## Phase 4: Core Logic (Steps 17–21, 21b)

### Step 17: Scheduler in Rust

**What**: Port the pure scheduler from `lib/iris/src/iris/cluster/controller/scheduler.py:1-840`.

**Source files**:
- New: `lib/iris/rust/src/scheduler.rs`
- Modified: `lib/iris/src/iris/cluster/controller/scheduler.py` (delegate to Rust via PyO3)

**Rust crates**: `pyo3`

**Python interface**: `#[pyfunction]` `schedule(context: SchedulingContext) -> list[Assignment]` where `SchedulingContext` is constructed from Python `WorkerSnapshot` objects. The Python scheduler module becomes a thin wrapper.

**Tests**: Port all cases from `tests/cluster/controller/test_scheduler.py` to Rust. Python integration test: run scheduler with identical input on both backends, assert identical assignments (order-independent).

**Dependencies**: Step 6 (constraints), Step 7 (JobName), Step 10 (WorkerRow for snapshot construction).

---

### Step 18: State Transitions in Rust

**What**: Port the state machine from `lib/iris/src/iris/cluster/controller/transitions.py:1-1679`.

**Source files**:
- New: `lib/iris/rust/src/transitions.rs`
- Read: `lib/iris/src/iris/cluster/controller/transitions.py`

**Rust crates**: `pyo3`, `rusqlite`

**Python interface**: `#[pyfunction]` for each transition function: `apply_heartbeat`, `apply_task_completion`, `apply_job_timeout`, etc. Called from `Controller.tick()`.

**Tests**: Port `tests/cluster/controller/test_transitions.py`. Critical: test every state edge (PENDING→RUNNING, RUNNING→SUCCEEDED, RUNNING→FAILED, RUNNING→WORKER_FAILED, etc.).

**Dependencies**: Step 11 (DB writes), Step 17 (scheduler for assignment application).

---

### Step 19: Scaling Group Management

**What**: Port scaling group logic from `lib/iris/src/iris/cluster/controller/scaling_group.py:1-1275`.

**Source files**:
- New: `lib/iris/rust/src/scaling_group.rs`
- Read: `lib/iris/src/iris/cluster/controller/scaling_group.py`

**Rust crates**: `pyo3`, `prost`

**Python interface**: `#[pyclass]` for `ScalingGroup` with methods for demand computation, waterfall routing, and VM lifecycle tracking.

**Tests**: Port `tests/cluster/platform/test_scaling_group.py`. Test demand computation with various job mixes.

**Dependencies**: Step 6 (constraints), Step 8 (ResourceSpec), Step 3 (config protos).

---

### Step 20: Autoscaler in Rust

**What**: Port the autoscaler from `lib/iris/src/iris/cluster/controller/autoscaler.py:1-1566`.

**Source files**:
- New: `lib/iris/rust/src/autoscaler.rs`
- Read: `lib/iris/src/iris/cluster/controller/autoscaler.py`

**Rust crates**: `pyo3`, `tokio` (for async platform calls)

**Python interface**: `#[pyclass]` for `Autoscaler` with `evaluate()` method. During transition, platform calls are delegated back to Python via PyO3 callback.

**Tests**: Port `tests/cluster/controller/test_autoscaler.py`. Test scale-up/scale-down decisions with mock platform responses.

**Dependencies**: Step 19 (scaling groups), Step 18 (transitions for VM lifecycle).

---

### Step 21: Controller Coordinator in Rust

**What**: Port the main controller loop from `lib/iris/src/iris/cluster/controller/controller.py:1-1646`.

**Source files**:
- New: `lib/iris/rust/src/controller.rs`
- Read: `lib/iris/src/iris/cluster/controller/controller.py`

**Rust crates**: `tokio`, `pyo3`

**Python interface**: `#[pyclass]` for `RustController` with `tick()` and `start()`/`stop()` methods. The Python `Controller` class delegates to `RustController` for scheduling and transitions, keeping platform/autoscaler calls in Python initially.

**Tests**: Integration test using `LocalCluster` harness: start cluster with Rust controller, submit job, verify execution. All existing `tests/cluster/controller/test_job.py` must pass.

**Dependencies**: Step 17 (scheduler), Step 18 (transitions), Step 20 (autoscaler), Step 15 (RPC service).

---

### Step 21b: Remove Python Core Logic

**What**: Delete the Python scheduler, transitions, autoscaler, scaling group, and controller coordinator implementations. These are now fully served by Rust via PyO3 or the standalone Rust binary.

**Source files**:
- Deleted: `lib/iris/src/iris/cluster/controller/scheduler.py`
- Deleted: `lib/iris/src/iris/cluster/controller/transitions.py`
- Deleted: `lib/iris/src/iris/cluster/controller/autoscaler.py`
- Deleted: `lib/iris/src/iris/cluster/controller/scaling_group.py`
- Deleted: `lib/iris/src/iris/cluster/controller/controller.py`

**Tests**: Full test suite passes. No Python core logic remains in the controller.

**Dependencies**: Steps 17–21 all passing.

---

## Phase 5: Platform & Runtime (Steps 22–26, 26b)

### Step 22: Platform Abstraction Trait

**What**: Define the `Platform` trait in Rust mirroring the Protocol in `lib/iris/src/iris/cluster/platform/base.py:1-587`.

**Source files**:
- New: `lib/iris/rust/src/platform/mod.rs`, `lib/iris/rust/src/platform/traits.rs`
- Read: `lib/iris/src/iris/cluster/platform/base.py` (Platform Protocol, SliceInfo, VmInfo)

**Rust crates**: `async-trait`, `tokio`

**Python interface**: None initially. The trait defines the Rust-side abstraction. Platform implementations that call cloud APIs will be added in subsequent steps.

**Tests**: Compile-time verification that the trait is implementable. A `MockPlatform` struct for testing.

**Dependencies**: Step 3 (config protos), Step 8 (ResourceSpec).

---

### Step 23: Local Platform Implementation

**What**: Port `LocalPlatform` from `lib/iris/src/iris/cluster/platform/local.py:1-581`.

**Source files**:
- New: `lib/iris/rust/src/platform/local.rs`
- Read: `lib/iris/src/iris/cluster/platform/local.py`

**Rust crates**: `tokio::process`, `sysinfo`

**Python interface**: `#[pyclass]` for `RustLocalPlatform` implementing the Platform trait. Used for testing and local development.

**Tests**: Test VM create/delete/list lifecycle on localhost. Integration test: run `LocalCluster` with Rust platform backend.

**Dependencies**: Step 22.

---

### Step 24: GCP Platform Implementation

**What**: Port GCP VM/TPU management from `lib/iris/src/iris/cluster/platform/gcp.py:1-1715`.

**Source files**:
- New: `lib/iris/rust/src/platform/gcp.rs`
- Read: `lib/iris/src/iris/cluster/platform/gcp.py`

**Rust crates**: `reqwest` (for GCE/TPU REST APIs), `google-cloud-auth`, `tokio`, `serde_json`

**Python interface**: `#[pyclass]` for `RustGcpPlatform`. Initially, complex operations (TPU slice management) can delegate to Python via PyO3 callback while simpler operations (VM create/delete) are native Rust.

**Tests**: Unit tests with mock HTTP responses for VM create/delete/list. Integration test against GCP (gated behind `IRIS_GCP_INTEGRATION_TEST` env var).

**Dependencies**: Step 22.

---

### Step 25: Runtime Abstraction and Process Runtime

**What**: Port the `Runtime` abstraction and `ProcessRuntime` from `lib/iris/src/iris/cluster/runtime/types.py` and `lib/iris/src/iris/cluster/runtime/process.py:1-671`.

**Source files**:
- New: `lib/iris/rust/src/runtime/mod.rs`, `lib/iris/rust/src/runtime/process.rs`
- Read: `lib/iris/src/iris/cluster/runtime/process.py`, `lib/iris/src/iris/cluster/runtime/types.py`

**Rust crates**: `tokio::process`, `nix` (signal handling)

**Python interface**: `#[pyclass]` for `RustProcessRuntime` with `start_task()`, `kill_task()`, `get_status()`.

**Tests**: Port `tests/cluster/runtime/test_entrypoint.py` cases. Test process lifecycle: start, monitor, kill, timeout.

**Dependencies**: Step 8 (ResourceSpec), Step 7 (JobName).

---

### Step 26: Docker Runtime

**What**: Port Docker container management from `lib/iris/src/iris/cluster/runtime/docker.py:1-1013`.

**Source files**:
- New: `lib/iris/rust/src/runtime/docker.rs`
- Read: `lib/iris/src/iris/cluster/runtime/docker.py`

**Rust crates**: `bollard` (Docker API), `tokio`

**Python interface**: `#[pyclass]` for `RustDockerRuntime` with same interface as `ProcessRuntime`.

**Tests**: Port `tests/cluster/runtime/test_docker_runtime.py`. Test container create/start/stop/logs lifecycle. Requires Docker daemon (CI must have Docker).

**Dependencies**: Step 25.

---

### Step 26b: Remove Python Platform & Runtime Implementations

**What**: Delete the Python platform and runtime implementations. The Rust binaries handle all platform and runtime operations.

**Source files**:
- Deleted: `lib/iris/src/iris/cluster/platform/local.py`, `gcp.py`, `coreweave.py`, `manual.py`
- Deleted: `lib/iris/src/iris/cluster/runtime/process.py`, `docker.py`, `kubernetes.py`
- Keep: `lib/iris/src/iris/cluster/platform/base.py` (Protocol definition, still used by Python client)

**Tests**: Full test suite passes with Rust platform/runtime. Integration tests against GCP and Docker confirm production readiness.

**Dependencies**: Steps 22–26 all passing.

---

## Phase 6: Client & CLI (Steps 27–34)

### Step 27: Rust Worker Binary

**What**: Build a standalone Rust binary that runs the full worker agent — combining WorkerService (Step 16), runtime (Steps 25-26), and env probe.

**Source files**:
- New: `lib/iris/rust/src/bin/iris-worker.rs`
- Read: `lib/iris/src/iris/cluster/worker/worker.py`, `lib/iris/src/iris/cluster/worker/main.py`

**Rust crates**: `tokio`, `axum`, `clap`, `tracing`

**Python interface**: None — standalone binary replaces `python -m iris.cluster.worker.main`.

**Tests**: Integration test: Rust worker registers with Python controller, receives heartbeat, executes process task, reports completion. Test with Docker runtime.

**Dependencies**: Step 16, Step 25, Step 26.

---

### Step 28: Rust Controller Binary

**What**: Build a standalone Rust binary that runs the full controller — combining ControllerService (Steps 14-15), scheduler (Step 17), transitions (Step 18), autoscaler (Step 20), DB (Steps 9-11).

**Source files**:
- New: `lib/iris/rust/src/bin/iris-controller.rs`
- Read: `lib/iris/src/iris/cluster/controller/main.py`

**Rust crates**: `tokio`, `axum`, `clap`, `tracing`, `rusqlite`

**Python interface**: None — standalone binary replaces `python -m iris.cluster.controller.main`.

**Tests**: Full integration: Rust controller + Rust worker. Submit job via Python client, verify execution. Run the full `tests/cluster/controller/` test suite against Rust controller.

**Dependencies**: Steps 14-21.

---

### Step 29: CLI in Rust

**What**: Port the CLI from `lib/iris/src/iris/cli/` (~3600 LOC) to Rust using `clap`.

**Source files**:
- New: `lib/iris/rust/src/bin/iris.rs`, `lib/iris/rust/src/cli/`
- Read: `lib/iris/src/iris/cli/main.py`, `job.py`, `cluster.py`, `query.py`, `build.py`, `rpc.py`

**Rust crates**: `clap 4` (derive), `tabled` (table formatting), `tokio`, `reqwest`

**Python interface**: None — replaces `python -m iris.cli.main`.

**Tests**: CLI integration tests: `iris job list`, `iris job submit`, `iris cluster status`. Compare output format with Python CLI.

**Dependencies**: Step 28 (needs running controller to test against).

---

### Step 30: Python Client Library Compatibility Layer

**What**: Ensure the Python client library (`lib/iris/src/iris/client/client.py:1-1092`) works seamlessly with Rust controller. Add any missing wire-format compatibility fixes.

**Source files**:
- Modified: `lib/iris/src/iris/client/client.py` (if any fixes needed)
- Modified: `lib/iris/src/iris/cluster/client/remote_client.py`

**Rust crates**: N/A (this is Python-side)

**Python interface**: The existing `IrisClient` class, unchanged.

**Tests**: Full client test suite against Rust controller. Test all client methods: `submit()`, `get_status()`, `terminate()`, `list_jobs()`, `get_logs()`, `wait_for_completion()`.

**Dependencies**: Step 28.

---

### Step 31: Actor System RPC Passthrough

**What**: Implement ActorService (`actor.proto:108-118`) in Rust as a bytes passthrough — Rust handles the Connect-RPC transport but delegates serialization/deserialization to Python cloudpickle.

**Source files**:
- New: `lib/iris/rust/src/actor_service.rs`
- Read: `lib/iris/src/iris/actor/server.py`, `lib/iris/src/iris/actor/client.py`

**Rust crates**: `axum`, `prost`, `iris-connect`

**Python interface**: The actor server runs in a Python process. The Rust Connect-RPC handler receives `ActorCall` messages, forwards `serialized_args`/`serialized_kwargs` bytes to the Python process (via a local socket or PyO3 callback), and returns the `ActorResponse`.

**Tests**: End-to-end actor test: Python client → Rust RPC layer → Python actor → response. All existing actor tests pass.

**Dependencies**: Step 12.

---

### Step 32: Kubernetes Runtime

**What**: Port Kubernetes runtime from `lib/iris/src/iris/cluster/runtime/kubernetes.py:1-696` and CoreWeave platform from `lib/iris/src/iris/cluster/platform/coreweave.py:1-1577`.

**Source files**:
- New: `lib/iris/rust/src/runtime/kubernetes.rs`, `lib/iris/rust/src/platform/coreweave.rs`
- Read: `lib/iris/src/iris/cluster/runtime/kubernetes.py`, `lib/iris/src/iris/cluster/platform/coreweave.py`

**Rust crates**: `kube-rs`, `k8s-openapi`, `tokio`

**Python interface**: `#[pyclass]` for `RustKubernetesRuntime` and `RustCoreweavePlatform`.

**Tests**: Port `tests/cluster/runtime/test_kubernetes_runtime.py` and `tests/cluster/platform/test_coreweave_platform.py`.

**Dependencies**: Step 25 (runtime trait), Step 22 (platform trait).

---

### Step 33: Dashboard Integration

**What**: Port dashboard endpoints from `lib/iris/src/iris/cluster/controller/dashboard.py` and `lib/iris/src/iris/cluster/worker/dashboard.py`.

**Source files**:
- New: `lib/iris/rust/src/dashboard.rs`
- Read: Controller and worker dashboard files

**Rust crates**: `axum`, `askama` or `maud` (HTML templating), `tokio`

**Python interface**: None — served directly by Rust controller/worker.

**Tests**: HTTP test: GET dashboard endpoints, verify HTML response contains expected elements.

**Dependencies**: Step 28.

---

### Step 34: Final Python Cleanup

**What**: Remove remaining Python scaffolding: worker main/entry, CLI, managed_thread, chaos, marin_fs (replaced by `object_store` in Rust), logging shims, and any remaining re-export shim modules. The Python `iris` package shrinks to: client library, actor system, and proto stubs.

**Source files**:
- Deleted: `lib/iris/src/iris/cluster/worker/worker.py`, `main.py`
- Deleted: `lib/iris/src/iris/cluster/controller/main.py`
- Deleted: `lib/iris/src/iris/cli/` (all CLI modules)
- Deleted: `lib/iris/src/iris/managed_thread.py`, `chaos.py`, `marin_fs.py`
- Deleted: All `_native_loader.py` and re-export shims (import directly from `iris._native`)
- Modified: `lib/iris/src/iris/__init__.py` — re-export from Rust

**Python interface**: Python client library (`iris.client`) and actor system remain in Python. Everything else is Rust.

**Tests**: Full test suite passes. `pip install iris` installs the Rust extension wheel. `make iris-dev` and `make iris-user` both work. Docker image contains only the Rust binaries.

**Dependencies**: Steps 27-33. All tests passing on Rust binaries.

---

## Dependency Graph

```
Step 1 ─→ Step 2 (CI wheels + Docker image pipeline)
  │         │
  ├─→ Step 3 (proto codegen)
  │    │ │
  │    │ ├→ Step 4 (time utils, needs 3)
  │    │ ├→ Step 5 → Step 6 (enums → constraints)
  │    │ └→ Step 8 (ResourceSpec, needs 3, 5)
  │    │
  │    └→ Step 12 (Connect-RPC, needs 3)
  │         │
  │         ├→ Step 13 (auth)
  │         ├→ Step 16 (WorkerService, needs 3)
  │         └→ Step 31 (actor passthrough)
  │
  │    Step 6b (delete Python leaf modules, needs 4, 5, 6)
  │
  ├─→ Step 7 (JobName)
  │
  ├─→ Step 9 (SQLite DB)
  │    │
  │    └→ Step 10 (DB rows, needs 9, 7, 4)
  │         │
  │         └→ Step 11 (DB writes)
  │              │
  │              ├→ Step 14 (read RPCs, needs 11, 12, 13)
  │              │    │
  │              │    └→ Step 15 (write RPCs)
  │              │
  │              └→ Step 18 (transitions, needs 11, 17)
  │
  │    Step 11b (delete Python DB layer, needs 7-11)
  │
  │    Step 17 (scheduler, needs 6, 7, 10)
  │    Step 19 (scaling groups, needs 6, 8, 3)
  │    Step 20 (autoscaler, needs 19, 18)
  │    Step 21 (controller, needs 17, 18, 20, 15)
  │
  │    Step 16b (delete Python RPC services + push Docker image, needs 12-16)
  │    Step 21b (delete Python core logic, needs 17-21)
  │
  │    Step 22 (platform trait, needs 3, 8)
  │      ├→ Step 23 (local platform)
  │      └→ Step 24 (GCP platform)
  │
  │    Step 25 (process runtime, needs 8, 7)
  │      └→ Step 26 (Docker runtime)
  │
  │    Step 26b (delete Python platform/runtime, needs 22-26)
  │
  │    Step 27 (worker binary, needs 16, 25, 26)
  │    Step 28 (controller binary, needs 14-21)
  │    Step 29 (CLI, needs 28)
  │    Step 30 (client compat, needs 28)
  │    Step 32 (K8s runtime, needs 25, 22)
  │    Step 33 (dashboard, needs 28)
  │    Step 34 (final Python cleanup, needs 27-33)
```

## Incremental Python Removal Strategy

Each phase ends with a `*b` cleanup step that deletes the Python code replaced by Rust in that phase. This ensures:

1. **No dead code accumulates.** Python implementations are deleted as soon as Rust replacements pass the full test suite.
2. **CI validates the Rust path.** After each cleanup step, CI runs the test suite with `make iris-user` (pre-built wheel), confirming that user-mode installation works end-to-end.
3. **Docker images stay current.** Each cleanup step triggers a Docker image rebuild+push so deployed clusters use the latest Rust code automatically.
4. **No fallback path.** We do NOT keep `_*_py.py` backup copies or `IRIS_USE_NATIVE=0` fallback. Once Rust passes tests, the Python code is deleted. Rollback is via `git revert`.

## CI/CD Pipeline

Set up in Step 2 and used throughout:

- **On every push to `main` touching `lib/iris/rust/`**: Build manylinux + macOS wheels, upload to GitHub Releases (`latest` tag). Build and push Docker image with iris-rust baked in.
- **`make iris-user`**: Downloads the latest wheel from GitHub Releases via `gh release download`, installs it. No Rust toolchain needed.
- **`make iris-dev`**: Runs `maturin develop --release` from source. Requires Rust toolchain.
- **On tag `iris-rust-v*`**: Additionally publish to PyPI for external consumers.

## Timeline Estimate

| Phase | Steps | Estimated Duration | Parallelism |
|-------|-------|--------------------|-------------|
| Phase 1 | 1–6, 6b | 4–6 weeks | Step 2 ∥ Step 3; Steps 5-6 ∥ Step 4; 6b after all |
| Phase 2 | 7–11, 11b | 4–6 weeks | Steps 7-8 ∥; Steps 10-11 sequential; 11b after |
| Phase 3 | 12–16, 16b | 6–8 weeks | Step 12 first; Steps 14-16 partially ∥; 16b after |
| Phase 4 | 17–21, 21b | 6–8 weeks | Steps 17-19 partially ∥; 20-21 sequential; 21b after |
| Phase 5 | 22–26, 26b | 4–6 weeks | Steps 23-24 ∥; Steps 25-26 sequential; 26b after |
| Phase 6 | 27–34 | 6–8 weeks | Steps 27-28 ∥; 29-33 partially ∥; 34 last |
| **Total** | | **~8–12 months** | |

## Milestones

1. **M1 (end of Phase 1)**: `iris._native` module builds and passes CI. Dev/user build modes work. Time utils and constraint evaluation run in Rust. **Python leaf module code deleted.** CI publishes wheels on every merge.

2. **M2 (end of Phase 2)**: Rust owns the controller SQLite database. **Python DB/ORM code deleted.**

3. **M3 (end of Phase 3)**: A Rust binary serves all ControllerService and WorkerService RPCs. **Python RPC handler code deleted.** Docker images with Rust binary pushed to registry.

4. **M4 (end of Phase 4)**: Scheduler, transitions, and autoscaler run in Rust. Controller loop is Rust-native. **Python core logic deleted.**

5. **M5 (end of Phase 5)**: Platform and runtime implementations in Rust. **Python platform/runtime code deleted.**

6. **M6 (end of Phase 6)**: Full Rust deployment. Python `iris` package contains only: client library, actor system, and proto stubs. CLI is a Rust binary.
