# Iris-to-Rust Migration: Research Findings

## Module Inventory

### Total Codebase Size
~39,700 LOC across ~106 Python files in `lib/iris/src/iris/`, plus ~1,930 lines of protobuf definitions across 8 `.proto` files.

### Module-by-Module Assessment

| Module | Files | LOC | Complexity | External Deps | Port Difficulty |
|--------|-------|-----|-----------|---------------|----------------|
| **time_utils** | 1 | 556 | Low | None (stdlib only + `time_pb2`) | Easy |
| **managed_thread** | 1 | 349 | Medium | None (stdlib `threading`) | Medium — Rust has `tokio::task` but paradigm differs |
| **distributed_lock** | 1 | 342 | Medium | `fsspec`, `google.cloud.storage` | Medium — GCS client needed |
| **chaos** | 1 | ~100 | Low | None | Easy |
| **logging** | 1 | ~200 | Low | stdlib | Easy |
| **marin_fs** | 1 | 878 | Medium | `fsspec`, `gcsfs`, `s3fs` | Hard — fsspec abstraction is Python-specific |
| **rpc/** | 5 py + 8 proto | ~400 py + 1928 proto | High | `connectrpc`, `starlette`, `uvicorn` | High — Connect-RPC is the migration crux |
| **cluster/types** | 1 | 800 | Medium | `cloudpickle`, `humanfriendly` | Medium — cloudpickle is Python-only |
| **cluster/constraints** | 1 | 1001 | Medium | None (only `cluster_pb2`) | Easy |
| **cluster/config** | 1 | 1094 | Medium | `pydantic`, `PyYAML` | Medium |
| **cluster/controller/db** | 1 | 1334 | High | `sqlite3` (stdlib) | Medium — `rusqlite` is mature |
| **cluster/controller/scheduler** | 1 | 840 | High | None (pure logic) | Easy — pure algorithm, no I/O |
| **cluster/controller/transitions** | 1 | 1679 | High | None (depends on db types) | Medium |
| **cluster/controller/controller** | 1 | 1646 | Very High | threading, all internal modules | Hard — central coordinator |
| **cluster/controller/service** | 1 | 1759 | Very High | `connectrpc`, starlette | Hard — RPC surface area |
| **cluster/controller/autoscaler** | 1 | 1566 | High | Platform abstractions | Hard |
| **cluster/controller/scaling_group** | 1 | 1275 | High | Config protos | Medium |
| **cluster/controller/auth** | 1 | 387 | Medium | `PyJWT` | Easy — `jsonwebtoken` crate |
| **cluster/controller/checkpoint** | 1 | ~300 | Low | `fsspec` | Medium |
| **cluster/controller/query** | 1 | 81 | Low | sqlite3 | Easy |
| **cluster/worker/worker** | 1 | 780 | High | threading, internal | Hard |
| **cluster/worker/task_attempt** | 1 | 825 | High | Docker/K8s runtime | Hard |
| **cluster/worker/service** | 1 | ~300 | Medium | `connectrpc` | Medium |
| **cluster/worker/env_probe** | 1 | 613 | Medium | subprocess, GCE metadata | Medium |
| **cluster/platform/base** | 1 | 587 | Medium | None (Protocol definitions) | Easy |
| **cluster/platform/gcp** | 1 | 1715 | Very High | `google.cloud.compute`, `google.cloud.tpu` | Hard |
| **cluster/platform/coreweave** | 1 | 1577 | Very High | Kubernetes client | Hard |
| **cluster/platform/local** | 1 | 581 | Medium | subprocess | Medium |
| **cluster/platform/manual** | 1 | 507 | Medium | SSH (paramiko/subprocess) | Medium |
| **cluster/runtime/docker** | 1 | 1013 | High | Docker SDK | Hard |
| **cluster/runtime/kubernetes** | 1 | 696 | High | K8s client | Hard |
| **cluster/runtime/process** | 1 | 671 | Medium | subprocess | Medium |
| **client/client** | 1 | 1092 | High | connectrpc, all types | Medium |
| **client/worker_pool** | 1 | 607 | Medium | internal | Medium |
| **actor/server** | 1 | ~300 | Medium | `cloudpickle`, `connectrpc` | Hard — cloudpickle is fundamental |
| **actor/client** | 1 | ~250 | Medium | `cloudpickle`, `connectrpc` | Hard |
| **cli/** | 7 | ~3600 | Medium | `click`, `tabulate` | Medium — `clap` is mature |

## Dependency Analysis

### Leaf Modules (fewest dependencies, easiest to port)

1. **`time_utils`** — Only depends on `time_pb2` proto. Pure value types (Timestamp, Duration, Deadline, Timer, RateLimiter, TokenBucket, ExponentialBackoff). Zero external deps. ~556 LOC.

2. **`cluster/constraints`** — Depends only on `cluster_pb2`. Core scheduling primitives (AttributeValue, Constraint, ConstraintOp, ResourceCapacity, ConstraintIndex). ~1001 LOC of pure logic.

3. **`cluster/controller/scheduler`** — Pure algorithm. Depends on constraints + types. No I/O, no threading, no external deps. ~840 LOC. Uses `WorkerSnapshot` Protocol for input decoupling.

4. **`cluster/controller/auth`** — JWT signing/verification. Depends only on `PyJWT`. ~387 LOC. Direct equivalent: `jsonwebtoken` crate.

5. **`rpc/proto_utils`** — Trivial enum-to-string helpers. ~68 LOC.

6. **`cluster/controller/query`** — Raw SQL executor, ~81 LOC. Depends on db layer.

### Mid-Tier Modules (moderate dependencies)

7. **`cluster/types`** — Depends on `cloudpickle` and `humanfriendly` for `Entrypoint` serialization. The `JobName`, `WorkerId`, `ResourceSpec`, `EnvironmentSpec` types are pure data and easy to port. `Entrypoint` wraps a Python callable via cloudpickle — this must remain Python.

8. **`cluster/controller/db`** — Custom ORM over sqlite3 with `db_row_model` decorator, `Predicate` query builder, and `ControllerDB` class. ~1334 LOC. `rusqlite` is a direct equivalent, but the ORM pattern needs reimplementation.

9. **`cluster/controller/transitions`** — State machine for jobs/tasks/workers. ~1679 LOC. Depends on db layer + types. Pure logic once db reads/writes are abstracted.

10. **`cluster/config`** — YAML/proto config loading. ~1094 LOC. Depends on `pydantic` validation and proto merging.

### Hard Modules (deep dependencies, I/O-heavy)

11. **`cluster/controller/controller`** + **`service`** — Central coordinator. ~3405 LOC combined. Depends on everything: db, scheduler, transitions, autoscaler, threading, RPC.

12. **`cluster/platform/*`** — Cloud provider integrations. ~4400 LOC total. Each platform has deep SDK dependencies (google-cloud-compute, kubernetes-client).

13. **`cluster/runtime/*`** — Container execution. ~2380 LOC. Docker SDK, K8s API, subprocess management.

14. **`actor/*`** — Actor system. ~550 LOC. Fundamentally tied to `cloudpickle` for Python callable serialization. This should be the **last** thing ported or potentially kept in Python permanently.

## Risk Assessment

### Low Risk
- **time_utils**: Pure value types, 1:1 Rust mapping. `std::time` covers everything.
- **constraints**: Pure data + matching logic. Rust enums and pattern matching are superior.
- **scheduler**: Stateless pure function. Rust's ownership model naturally prevents mutation bugs.
- **auth (JWT)**: `jsonwebtoken` crate is battle-tested.
- **proto codegen**: `prost` + `prost-build` generate Rust types from the same `.proto` files.

### Medium Risk
- **DB layer**: `rusqlite` is mature, but the custom ORM (`db_row_model`, `Predicate`) needs careful reimplementation. Risk: schema drift between Python and Rust versions during migration.
- **Config loading**: YAML + proto merging has edge cases. `serde_yaml` + `prost` cover this.
- **State transitions**: Complex state machine with many edge cases. Risk: behavioral divergence during dual-language period.

### High Risk
- **Connect-RPC protocol compatibility**: The Python client uses `connect-python` which speaks the [Connect protocol](https://connectrpc.com/docs/protocol/) — HTTP/1.1 or HTTP/2 with JSON or binary protobuf encoding. The Rust ecosystem has `tonic` (standard gRPC) but NOT a mature Connect-RPC implementation. This is the **critical path risk**.
- **Actor system (cloudpickle)**: The actor system serializes Python callables with cloudpickle. There is no Rust equivalent. Actor system must either stay in Python permanently or be redesigned with a language-agnostic serialization approach.
- **Platform integrations**: GCP, CoreWeave, and manual SSH platforms have complex cloud SDK interactions. Google Cloud Rust SDK is less mature than the Python SDK.

### Critical Risk
- **Threading model migration**: Iris uses `ManagedThread` + `ThreadPoolExecutor` (non-async). Rust would use `tokio` (async) or `std::thread`. The paradigm shift affects every module. **Recommendation**: Port to Rust's async model (`tokio`) since the controller is I/O-bound, but preserve the non-async Python API via PyO3 blocking wrappers.

## Rust Crate Equivalents

| Python Dependency | Rust Crate | Maturity | Notes |
|-------------------|-----------|----------|-------|
| `sqlite3` | `rusqlite` | Excellent | Bundles SQLite, supports WAL mode |
| `PyJWT` | `jsonwebtoken` | Excellent | JWT signing/verification |
| `click` | `clap` | Excellent | Derive macros for CLI |
| `tabulate` | `tabled` or `comfy-table` | Good | Table formatting |
| `humanfriendly` | `humansize` + custom | Fair | Partial coverage |
| `pydantic` | `serde` + `validator` | Excellent | `serde` is superior |
| `PyYAML` | `serde_yaml` | Excellent | |
| `protobuf` | `prost` + `prost-build` | Excellent | Code generation from .proto |
| `connectrpc` (Python) | See analysis below | **Gap** | No mature Connect-RPC for Rust |
| `starlette` / `uvicorn` | `axum` + `hyper` + `tokio` | Excellent | HTTP framework |
| `cloudpickle` | N/A | **No equivalent** | Python-specific |
| `fsspec` / `gcsfs` | `object_store` (Apache) | Good | GCS, S3, local FS |
| `google.cloud.compute` | `google-cloud-compute` (unofficial) | Fair | Less mature |
| `google.cloud.tpu` | Direct REST API via `reqwest` | Fair | No official crate |
| `threading` / `ThreadPoolExecutor` | `tokio` / `rayon` | Excellent | Different paradigm |
| `docker` (Python SDK) | `bollard` | Good | Docker Engine API |
| `kubernetes` (Python) | `kube-rs` | Excellent | Mature K8s client |

## Connect-RPC Analysis

### Current Protocol
Iris uses the [Connect protocol](https://connectrpc.com/docs/protocol/), which is an HTTP-based RPC protocol designed by Buf. Key characteristics:

- **Wire format**: HTTP/1.1 or HTTP/2 with either JSON or binary protobuf bodies
- **URL scheme**: `POST /<package>.<Service>/<Method>` (e.g., `POST /iris.cluster.ControllerService/LaunchJob`)
- **Content-Type**: `application/proto` (binary) or `application/json` (JSON)
- **Error model**: Uses `Connect-Error` trailer/header with error codes
- **Streaming**: Supports server-streaming via chunked encoding with envelope framing

### Rust Alternatives

1. **`tonic` (standard gRPC)** — Mature, production-grade. But speaks gRPC protocol (HTTP/2 with `application/grpc+proto`), NOT Connect protocol. **Wire-incompatible** with existing Python Connect clients.

2. **`connect-rpc` Rust crate** — Does not exist as a mature library. The Connect protocol has official implementations in Go, TypeScript, Java, Swift, and Kotlin — but NOT Rust.

3. **Custom Connect-RPC implementation on `axum`** — Since Connect is a simple HTTP protocol (not a complex binary protocol like gRPC), it can be implemented as middleware on top of `axum` + `prost`:
   - Route: `POST /<package>.<service>/<method>`
   - Deserialize: `prost::Message::decode()` for binary, `prost-reflect` + `serde_json` for JSON
   - Serialize response the same way
   - Error mapping to Connect error codes
   - ~500-1000 LOC of framework code

4. **Dual-protocol server** — Run both Connect (for existing Python clients) and gRPC (for new Rust clients) on the same port using content-type sniffing. `tonic` + custom Connect handler.

### Recommendation
**Option 3 (custom Connect-RPC on axum)** is the best path. The Connect protocol is intentionally simple — it's HTTP POST with protobuf bodies. Building a server-side Connect handler on `axum` is straightforward and ensures wire compatibility. This can be built as a reusable `iris-connect` crate.

The alternative — switching to standard gRPC — would require updating ALL Python clients simultaneously, which violates the incremental migration constraint.

## Proto File Structure

The existing proto files define the complete wire protocol:

| Proto File | Package | Messages | Services | LOC |
|-----------|---------|----------|----------|-----|
| `time.proto` | `iris.time` | Timestamp, Duration | — | 47 |
| `errors.proto` | `iris.errors` | Error | — | 28 |
| `logging.proto` | `iris.logging` | LogEntry | — | 71 |
| `query.proto` | `iris.query` | RawQueryRequest/Response, ColumnMeta | — | 36 |
| `vm.proto` | `iris.vm` | VmInfo, AutoscalerStatus, etc. | — | 189 |
| `config.proto` | `iris.config` | IrisClusterConfig and all sub-configs | — | 402 |
| `cluster.proto` | `iris.cluster` | All job/task/worker types | ControllerService (25 RPCs), WorkerService (7 RPCs) | 1037 |
| `actor.proto` | `iris.actor` | ActorCall, ActorResponse, Operation | ActorService (6 RPCs) | 118 |

All protos use `edition = "2023"` (cluster, config, time, vm, query) or `syntax = "proto3"` (actor). `prost-build` handles both. Proto import chain: `time` -> `config` -> `vm` -> `query` -> `cluster`.

## Existing Rust Precedent: dupekit Distribution Model

`lib/dupekit/` uses `maturin` for its PyO3/Rust native extension (`dupekit._native`). The distribution model:

- **Dev mode**: `maturin develop --release` builds the extension in-place from Rust source. Used by developers iterating on the Rust code.
- **User mode**: Pre-built wheels published via CI. Users install with `pip install dupekit` — no Rust toolchain required.
- **CI**: GitHub Actions builds manylinux/macOS wheels using `PyO3/maturin-action@v1`, uploads to PyPI on tagged releases.

This pattern should be replicated for `iris-native`. The same dual dev/user mode approach with `IRIS_BUILD_MODE` env var and `make iris-dev`/`make iris-user` targets keeps the workflow consistent across the repo.

## Key Architectural Observations

1. **The scheduler is the ideal first complex port target.** It's a pure function (`lib/iris/src/iris/cluster/controller/scheduler.py:1-840`) operating on immutable snapshots (`WorkerSnapshot` Protocol, `JobRequirements` dataclass). Zero I/O. Zero threading. The `WorkerSnapshot` protocol boundary makes it trivially callable from Python via PyO3.

2. **The DB layer is the migration bottleneck.** The custom ORM (`db_row_model` at `db.py:99-102`, `Predicate` at `db.py:123-134`, `ControllerDB` at `db.py:~300+`) is used by transitions, controller, and service. Porting this to `rusqlite` is medium complexity but high value — it unlocks porting transitions and the controller.

3. **The actor system should stay Python.** Its core value proposition is transparently calling Python methods on remote objects via `cloudpickle` serialization (`actor.proto:26-30`). There is no Rust equivalent for cloudpickle. The actor RPC can be served from Rust (just bytes passthrough), but the serialization/deserialization must remain Python.

4. **The Connect-RPC layer is architecturally advantageous for migration.** Because services communicate over HTTP + protobuf, a Rust controller can serve Python workers and clients without any shared-memory interop. This is cleaner than PyO3 for the service boundary.

5. **The `ManagedThread` pattern maps to `tokio::task::JoinSet`.** The `ThreadContainer` hierarchy (`managed_thread.py:156-307`) with `spawn()` / `stop()` / `join()` maps naturally to `tokio::task::JoinSet` with cancellation tokens.

6. **Config loading is proto-native.** The cluster config (`config.proto:385-402`) is already defined in protobuf. `prost` generates the Rust types directly — no need for `serde` roundtripping except for YAML file loading at the CLI boundary.
