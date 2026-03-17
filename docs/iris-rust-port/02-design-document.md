# Iris-to-Rust Migration: Design Document

## Problem

Iris is a ~40K LOC Python system that runs as a long-lived distributed controller managing cloud VMs, scheduling tasks, and coordinating workers. Python's GIL, startup latency, memory overhead, and lack of static types create operational pain at scale:

- Controller process uses ~500MB RSS idle, growing with job count
- GIL contention in the heartbeat/scheduler/autoscaler loop (all threading-based: `managed_thread.py:58-149`)
- No compile-time guarantees on the complex state machine (`transitions.py:1-1679`)
- Cold start for workers is dominated by Python environment setup, not actual Rust startup

The goal: incrementally replace Python with Rust, starting from leaf modules and working inward, while maintaining the existing Connect-RPC wire protocol and CLI interface throughout.

## Interop Strategy

Two complementary approaches, used at different architectural layers:

### Approach 1: PyO3/maturin for In-Process Hot Paths

Following the `lib/dupekit/` precedent (`dupekit/Cargo.toml:1-25`, `dupekit/pyproject.toml:1-55`):

- A new `lib/iris/rust/` Rust crate compiled as a cdylib via maturin
- Exposed as `iris._native` Python module (like `dupekit._native`)
- Used for CPU-bound hot paths: scheduler, constraint evaluation, time utilities
- Python code calls Rust functions that return Python-compatible types (via PyO3 `#[pyfunction]`/`#[pyclass]`)

```toml
# lib/iris/rust/Cargo.toml
[package]
name = "iris-rust"
version = "0.1.0"
edition = "2021"

[lib]
name = "_native"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.26", features = ["extension-module", "abi3-py311"] }
prost = "0.13"
rusqlite = { version = "0.32", features = ["bundled"] }
jsonwebtoken = "9"
tokio = { version = "1", features = ["full"] }
```

The Python shim pattern:

```python
# iris/time_utils.py (after migration)
from iris._native import Timestamp, Duration, Deadline, Timer
# All existing call sites unchanged — same API, Rust implementation
```

### Approach 2: RPC Boundary for Service Replacement

For replacing entire services (controller, worker), use the existing Connect-RPC protocol boundary:

- Build a Rust binary that implements `ControllerService` (the 25 RPCs in `cluster.proto:970-1025`)
- Serve on the same port with the same Connect-RPC wire format
- Python workers and clients connect without code changes
- Gradual migration: start with a few RPCs forwarded to Rust, expand over time

```
                     Connect-RPC (HTTP + protobuf)

[Python CLI] ──────> [Rust Controller] ──────> [Python Worker]
[Python Client] ──/  (axum + prost)            (unchanged)
```

### When to Use Each Approach

| Scenario | Approach |
|----------|----------|
| Porting `time_utils`, `constraints`, `scheduler` | PyO3 (same process, zero latency) |
| Porting `controller/db` | PyO3 (shared SQLite connection) |
| Replacing `controller/service` | RPC boundary (separate binary) |
| Replacing `worker/service` | RPC boundary (separate binary) |
| Porting `auth` | PyO3 first, then RPC boundary |

## Migration Ordering Rationale

The dependency graph flows inward:

```
                      CLI
                       |
                    Client
                       |
            Controller Service
           /        |         \
      Scheduler  Transitions  Autoscaler
           \        |         /
              Controller DB
                    |
         Types + Constraints
                    |
              Time Utilities
                    |
               Proto Types
```

We start at the bottom (leaf modules) and work upward because:

1. **Leaf modules have zero internal consumers to update.** Porting `time_utils` requires changing zero other files — the Python shim re-exports the same API.

2. **Each layer validates the interop mechanism.** If PyO3 works for `Timestamp`/`Duration`, we know it'll work for `Constraint`/`AttributeValue`.

3. **Performance gains compound from the bottom.** The scheduler calls `evaluate_constraint` (`constraints.py:~300`) thousands of times per cycle. Porting constraints to Rust speeds up the scheduler even before the scheduler itself is ported.

4. **Risk is contained.** If the Rust `time_utils` has a bug, only time-related behavior is affected. If the Rust controller service has a bug, everything breaks.

## State Management Migration

### Current State Architecture

The controller stores all state in a single SQLite database (`controller/db.py:1-1334`):

- Custom ORM: `db_row_model` decorator (`db.py:99-102`) creates frozen dataclasses with column metadata
- Predicate query builder: `Predicate` base class (`db.py:123-134`) with `&`, `|`, `~` operators
- Thread-safe access: `RLock` on write path, `read_snapshot()` context manager for consistent reads
- Tables: `jobs`, `tasks`, `workers`, `endpoints`, `log_entries`, `api_keys`, `transactions`
- 9 migration files in `controller/migrations/`

### Migration Plan

**Phase A: Rust reads the same SQLite file** (PyO3 approach)

```rust
// Rust reads from the same DB file, Python still writes
fn read_workers(db_path: &str) -> Vec<WorkerSnapshot> {
    let conn = Connection::open_with_flags(db_path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    // ... query workers table, return PyO3-compatible types
}
```

- Rust opens the SQLite file read-only (WAL mode supports concurrent readers)
- Python controller continues to own writes
- Scheduler reads worker snapshots from Rust, eliminating Python ORM overhead

**Phase B: Rust owns reads AND writes** (dual-access)

- Rust `ControllerDB` implementation with the same schema
- Python code calls Rust DB functions via PyO3
- Migrations run from Rust (`rusqlite-migration` crate)

**Phase C: Rust controller binary owns the DB exclusively**

- Rust controller service runs as a standalone binary
- Python code interacts only via Connect-RPC, never touches SQLite
- DB schema can evolve independently

### Schema Compatibility

The SQLite schema is defined by migration files in `controller/migrations/`. During dual-access phases:

- **Constraint**: Python and Rust must agree on the schema version
- **Mechanism**: Shared migration numbering. Rust applies migrations from a matching set of `.sql` files generated from the existing Python migration scripts.
- **Validation**: Integration test opens the DB with both Python and Rust, verifies read/write compatibility

## RPC Compatibility

### Connect Protocol Implementation

The Connect protocol (`cluster.proto:970-1037`) uses:

- **Unary RPCs only** — no streaming (simplifies implementation)
- **URL format**: `POST /iris.cluster.ControllerService/LaunchJob`
- **Binary encoding**: `Content-Type: application/proto`, body = serialized protobuf
- **JSON encoding**: `Content-Type: application/json`, body = protobuf JSON mapping
- **Error format**: JSON body `{"code": "not_found", "message": "..."}`

### Rust Server Implementation

Build a `connect-axum` adapter crate:

```rust
// Sketch of the Connect-RPC adapter
async fn connect_handler<S: Service>(
    method: &str,
    body: Bytes,
    content_type: &str,
) -> Response {
    let request_msg = match content_type {
        "application/proto" => S::decode_request(method, &body)?,
        "application/json" => S::decode_request_json(method, &body)?,
        _ => return error_response(Code::InvalidArgument, "unsupported content type"),
    };
    let response_msg = service.dispatch(method, request_msg).await?;
    // Encode response in the same format as the request
    encode_response(response_msg, content_type)
}
```

### Compatibility Testing

Each migrated RPC gets a compatibility test:

```python
# Test: Python client → Rust server produces identical results to Python client → Python server
def test_launch_job_compat(python_controller, rust_controller):
    request = make_launch_request()
    py_response = python_controller.LaunchJob(request)
    rs_response = rust_controller.LaunchJob(request)
    assert py_response == rs_response
```

The existing test suite (`tests/cluster/controller/test_service.py`, `test_job.py`, `test_scheduler.py`, etc.) runs against both Python and Rust implementations via a pytest fixture that selects the backend.

## Testing Strategy

### Layer 1: Unit Tests (Rust-native)

Each Rust module has `#[cfg(test)]` tests covering the same cases as the Python tests:

```rust
#[test]
fn test_timestamp_roundtrip() {
    let ts = Timestamp::now();
    let proto = ts.to_proto();
    let recovered = Timestamp::from_proto(&proto);
    assert_eq!(ts, recovered);
}
```

### Layer 2: PyO3 Integration Tests

Python tests call Rust functions via the `iris._native` module and verify identical behavior:

```python
def test_constraint_evaluation_matches():
    from iris._native import evaluate_constraint as rust_eval
    from iris.cluster.constraints import evaluate_constraint as py_eval

    for constraint, attributes in CONSTRAINT_TEST_CASES:
        assert rust_eval(constraint, attributes) == py_eval(constraint, attributes)
```

### Layer 3: RPC Compatibility Tests

End-to-end tests that submit jobs through the full Python client → Rust server → Python worker pipeline:

- Reuse existing `tests/cluster/controller/test_job.py` test scenarios
- Parameterize on backend: `@pytest.fixture(params=["python", "rust"])`
- The `LocalCluster` test harness (`cluster/local_cluster.py`) starts either a Python or Rust controller

### Layer 4: Shadow Mode

Before cutting over a production RPC:

1. Run Rust server alongside Python server
2. Mirror all requests to both
3. Compare responses (log divergences, don't fail)
4. Cut over when divergence rate hits zero

## Rollback Strategy

### No Fallback Path — Delete Python Once Verified

Once Rust passes the full test suite for a module, the Python implementation is **deleted** (not preserved as a fallback). This keeps the codebase clean and avoids maintaining two implementations.

- **Rollback mechanism**: `git revert` the cleanup commit that deleted the Python code. Each cleanup step is a single commit/PR, making reverts surgical.
- **Risk mitigation**: Each Rust module passes the existing Python test suite before the Python code is deleted. CI validates both `make iris-dev` (source build) and `make iris-user` (pre-built wheel).

### RPC Services: Docker Image Versioning

Since both Python and Rust controllers speak the same Connect-RPC protocol, rollback at the deployment level is a Docker image tag change:

```yaml
# Kubernetes deployment — roll back by reverting image tag
image: iris-controller:v2.3-rust  # or iris-controller:v2.2-python
```

### Database: Forward-Compatible Migrations

All Rust schema changes must be forward-compatible (additive only). This ensures the Python controller can read any DB written by the Rust controller if we need to revert to an older image.

## Distribution: Dev Mode vs User Mode

Two build/install modes for `iris._native`, switchable via env var or Makefile target.

### Dev Mode

`maturin develop --release` compiles the Rust extension in-place from source. Requires a local Rust toolchain. Hot-reload friendly — edit Rust, `make iris-dev`, re-run Python.

```bash
# Triggered by:
export IRIS_BUILD_MODE=dev
make iris-dev  # installs rustup if needed, runs maturin develop --release
```

### User Mode

Pre-built manylinux/macOS wheels published to PyPI. No Rust toolchain required. Default when no Rust toolchain is detected.

```bash
# Triggered by:
export IRIS_BUILD_MODE=user
make iris-user  # downloads pre-built wheel from GitHub Releases
```

### Switching Mechanism

The env var `IRIS_BUILD_MODE=dev|user` selects the mode:

```makefile
# Makefile targets
iris-dev:
	@command -v rustup >/dev/null || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
	cd lib/iris/rust && maturin develop --release
	@echo "iris._native built from source"

iris-user:
	gh release download iris-rust-latest --pattern '*.whl' --dir /tmp/iris-wheels --clobber
	uv pip install /tmp/iris-wheels/iris_rust-*.whl
	@echo "iris._native installed from pre-built wheel"
```

### CI/CD

GitHub Actions builds wheels on **every push to `main`** that touches `lib/iris/rust/`, not just tags. This ensures `make iris-user` always has a fresh wheel available.

```yaml
# .github/workflows/iris-rust-wheels.yml
on:
  push:
    branches: [main]
    paths: ['lib/iris/rust/**']
    tags: ['iris-rust-v*']

jobs:
  build-wheels:
    strategy:
      matrix:
        target: [x86_64-unknown-linux-gnu, aarch64-unknown-linux-gnu,
                 x86_64-apple-darwin, aarch64-apple-darwin]
    runs-on: ${{ contains(matrix.target, 'linux') && 'ubuntu-latest' || 'macos-latest' }}
    steps:
      - uses: PyO3/maturin-action@v1
        with:
          target: ${{ matrix.target }}
          args: --release -m lib/iris/rust/Cargo.toml
      - uses: actions/upload-artifact@v4
        with:
          name: wheel-${{ matrix.target }}
          path: target/wheels/*.whl

  publish-wheels:
    needs: build-wheels
    runs-on: ubuntu-latest
    steps:
      # Always upload to GitHub Releases (latest tag)
      - uses: softprops/action-gh-release@v2
        with:
          tag_name: iris-rust-latest
          files: target/wheels/*.whl
          prerelease: true
      # Only publish to PyPI on version tags
      - if: startsWith(github.ref, 'refs/tags/iris-rust-v')
        uses: pypa/gh-action-pypi-publish@release/v1

  build-docker:
    needs: build-wheels
    runs-on: ubuntu-latest
    steps:
      # Build Docker image with Rust extension baked in
      - uses: docker/build-push-action@v6
        with:
          push: true
          tags: |
            ghcr.io/${{ github.repository }}/iris:latest
            ghcr.io/${{ github.repository }}/iris:${{ github.sha }}
```

The `make iris-user` target downloads the latest wheel from GitHub Releases:

```makefile
iris-user:
	gh release download iris-rust-latest --pattern '*.whl' --dir /tmp/iris-wheels --clobber
	uv pip install /tmp/iris-wheels/iris_rust-*.whl
	@echo "iris._native installed from pre-built wheel"
```

### No Fallback

Once Python code for a module is deleted (per the incremental removal strategy), `iris._native` is **required**. There is no pure-Python fallback. If the extension is missing, import fails loudly. Users must run `make iris-dev` or `make iris-user` to install the extension.

## Key Design Decisions

### 1. Connect-RPC over gRPC
**Decision**: Implement Connect protocol in Rust rather than switching to standard gRPC.
**Why**: Existing Python clients use `connect-python`. Switching to gRPC would require updating all clients simultaneously — violating the incremental constraint. The Connect protocol is simple (HTTP POST + protobuf) and can be implemented in ~1000 LOC on `axum`.

### 2. `tokio` Runtime for Rust Services
**Decision**: Use async Rust (`tokio`) rather than mirroring Python's threading model.
**Why**: The controller is I/O-bound (DB queries, heartbeat RPCs, platform API calls). `tokio` is the standard for async Rust services. `axum` (HTTP framework) requires `tokio`. The Python threading model was a pragmatic choice for Python, not an architectural one.

### 3. Keep Actor System in Python
**Decision**: The actor system (`actor/server.py`, `actor/client.py`) stays in Python.
**Why**: Its core function is serializing/deserializing Python callables via `cloudpickle`. There is no Rust equivalent. The RPC transport (Connect-RPC) can be served by Rust as a bytes passthrough, but the serialization boundary must remain Python.

### 4. Shared Proto Files
**Decision**: Both Python and Rust generate types from the same `.proto` files.
**Why**: Ensures wire-format compatibility by construction. No manual type mapping. The proto files are the contract.

### 5. `rusqlite` with Bundled SQLite
**Decision**: Use `rusqlite` with the `bundled` feature (compiles SQLite from source).
**Why**: Guarantees the same SQLite version across all deployments. Avoids system SQLite version mismatches. WAL mode support is built-in.

### 6. Dual Dev/User Build Modes
**Decision**: `maturin develop` for developers, pre-built PyPI wheels for users, switchable via `IRIS_BUILD_MODE` env var or `make iris-dev`/`make iris-user`.
**Why**: Mirrors the `lib/dupekit/` precedent. Developers need fast iteration with source builds; users need zero-Rust-toolchain installs. The env var / Makefile target keeps switching trivial. CI publishes wheels automatically on tagged releases.
