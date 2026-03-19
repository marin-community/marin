# Iris Test Codebase Audit

## 1. Test File Inventory

**Total**: ~76 Python files in `lib/iris/tests/`, ~38,210 lines, ~1,418 `def test_` functions.

### actor/ (4 files, 470 lines, 14 tests)

| File | Lines | Tests | Summary |
|------|-------|-------|---------|
| `test_actor_e2e.py` | 52 | 2 | Basic actor RPC calls and exception propagation |
| `test_actor_lro.py` | 108 | 4 | Long-running operations: basic, failure, cancel, not found |
| `test_actor_pool.py` | 79 | 2 | ActorPool round-robin and broadcast |
| `test_actor_retry.py` | 231 | 6 | Retry on transient errors, cache clearing, exhaustion |

### cli/ (6 files, 450 lines, 27 tests)

| File | Lines | Tests | Summary |
|------|-------|-------|---------|
| `test_build_config_regions.py` | 41 | 1 | Image building with multi-region config |
| `test_image_tag_parsing.py` | 33 | 2 | GHCR image tag parsing and non-GHCR rejection |
| `test_job.py` | 64 | 8 | Region/zone validation with closest-match suggestions |
| `test_job_multinode.py` | 25 | 1 | Multinode job resolution |
| `test_local_cluster.py` | 151 | 1 | CLI local cluster e2e (boots real cluster) |
| `test_token_store.py` | 136 | 14 | Auth token persistence and format preservation |

### client/ (1 file, 98 lines, 4 tests)

| File | Lines | Tests | Summary |
|------|-------|-------|---------|
| `test_worker_pool.py` | 98 | 4 | Worker pool endpoint tracking |

### cluster/client/ (3 files, 437 lines, 19 tests)

| File | Lines | Tests | Summary |
|------|-------|-------|---------|
| `test_bundle.py` | 87 | 4 | Client-side bundle creation (zip packaging) |
| `test_job_info.py` | 59 | 7 | JobInfo parsing and validation |
| `test_local_client.py` | 291 | 8 | LocalClusterClient integration (boots local cluster) |

### cluster/controller/ (16 files, 12,726 lines, 545 tests) — **largest group**

| File | Lines | Tests | Summary |
|------|-------|-------|---------|
| `conftest.py` | 101 | 0 | Shared fixtures: `make_controller_state`, `fake_provider`, `make_test_entrypoint` |
| `test_api_keys.py` | 489 | 22 | API key auth: creation, rotation, verification, admin access |
| `test_autoscaler.py` | 3833 | 132 | Autoscaler: scale up/down, quota, failure injection, backoff, zone selection |
| `test_bundle_store.py` | 79 | 6 | Controller-side bundle store: write/read/idempotency |
| `test_checkpoint.py` | 223 | 11 | Job checkpoint save/restore |
| `test_dashboard.py` | 1062 | 32 | Dashboard RPC: list jobs/workers, autoscaler status, auth config |
| `test_db.py` | 330 | 21 | ControllerDB schema: CRUD, migrations, queries |
| `test_direct_controller.py` | 355 | 13 | DirectProvider: task dispatch, completion, multinode |
| `test_heartbeat.py` | 282 | 8 | Heartbeat timeout, worker reconciliation, failure cascades |
| `test_job.py` | 176 | 5 | Job lifecycle helpers, state transitions |
| `test_logs.py` | 641 | 30 | LogStore: write/read, rotation, level filtering, retention |
| `test_pending_diagnostics.py` | 62 | 3 | Scheduling diagnostics for pending tasks |
| `test_reservation.py` | 1778 | 55 | Reservation system: slots, quota, priority, preemption |
| `test_scheduler.py` | 2150 | 51 | Scheduler: constraint matching, resource accounting, priority, affinity |
| `test_service.py` | 1098 | 38 | ControllerService RPC: launch, terminate, list, status, pagination |
| `test_task_profiles.py` | 68 | 5 | Task profiling data collection |
| `test_transitions.py` | 3873 | 104 | Core state machine: submit→schedule→dispatch→complete/fail/retry/cancel |
| `test_vm_lifecycle.py` | 325 | 9 | VM start/stop/replace with mocked GCP platform |

### cluster/platform/ (7 files, 4,988 lines, 208 tests)

| File | Lines | Tests | Summary |
|------|-------|-------|---------|
| `conftest.py` | 2 | 0 | Empty (conftest in parent) |
| `fakes.py` | 788 | 0 | FakePlatform, FakeGcloud, FakeKubectl implementations |
| `test_bootstrap.py` | 218 | 17 | Worker/controller bootstrap script generation |
| `test_config.py` | 1447 | 55 | Scale group config validation across all providers |
| `test_coreweave_platform.py` | 1046 | 28 | CoreWeave: RBAC, nodepools, controller deploy, image resolution |
| `test_platform.py` | 924 | 36 | Cross-platform contract tests (GCP, Manual, Local) via FakeGcloud |
| `test_remote_exec.py` | 16 | 2 | SSH command string building |
| `test_scaling_group.py` | 1547 | 70 | Scale group demand routing, constraint matching |

### cluster/runtime/ (5 files, 682 lines, 45 tests)

| File | Lines | Tests | Summary |
|------|-------|-------|---------|
| `test_docker_runtime.py` | 106 | 6 | Docker container lifecycle |
| `test_entrypoint.py` | 104 | 6 | RuntimeEntrypoint proto construction |
| `test_env_parity.py` | 268 | 20 | Environment variable parity between runtimes |
| `test_process_runtime.py` | 69 | 3 | Process-based task execution |
| `test_profile.py` | 135 | 10 | Profiling hooks and data collection |

### cluster/worker/ (5 files, 1,652 lines, 61 tests)

| File | Lines | Tests | Summary |
|------|-------|-------|---------|
| `conftest.py` | 16 | 0 | DockerRuntime fixture |
| `test_bundle_store.py` | 98 | 4 | Worker-side bundle extraction and LRU cache |
| `test_dashboard.py` | 272 | 8 | Worker dashboard: task list/detail, port allocation |
| `test_env_probe.py` | 402 | 20 | Environment probe: TPU metadata, GPU attrs, worker ID |
| `test_worker.py` | 864 | 29 | Worker lifecycle: port allocation, task phases, heartbeat |

### cluster/ (6 files, 1,941 lines, 100 tests)

| File | Lines | Tests | Summary |
|------|-------|-------|---------|
| `test_attempt_logs.py` | 268 | 4 | Attempt log storage and retrieval |
| `test_client.py` | 160 | 8 | Cluster client integration |
| `test_constraints.py` | 161 | 9 | Constraint expression evaluation |
| `test_env_propagation.py` | 418 | 10 | Environment variable propagation end-to-end |
| `test_snapshot_reconciliation.py` | 367 | 15 | Controller state snapshot and reconciliation |
| `test_types.py` | 567 | 54 | Type utilities: JobName, topology, parsing |

### e2e/ (8 files, 3,483 lines, 57 tests)

| File | Lines | Tests | Summary |
|------|-------|-------|---------|
| `_docker_cluster.py` | 332 | 0 | Docker cluster fixture helper |
| `benchmark_controller.py` | 836 | 0 | Controller benchmark (not a test file) |
| `chronos.py` | 89 | 0 | Virtual time fixture |
| `conftest.py` | 528 | 0 | Core e2e fixtures: `cluster`, `smoke_cluster`, Playwright |
| `helpers.py` | 141 | 0 | Test helpers: `wait_for_condition`, etc. |
| `test_chaos.py` | 356 | 25 | Chaos: worker crashes, timeouts, capacity waits, retries |
| `test_docker.py` | 185 | 2 | Docker container integration e2e |
| `test_smoke.py` | 1113 | 27 | Full smoke suite: dashboard, jobs, logs, constraints, auth |
| `test_vm_lifecycle.py` | 93 | 3 | VM lifecycle failures: quota, stall, preemption |

### kubernetes/ (4 files, 1,891 lines, 110 tests)

| File | Lines | Tests | Summary |
|------|-------|-------|---------|
| `conftest.py` | 119 | 0 | mock_kubectl and KubernetesProvider fixtures |
| `test_kubectl.py` | 87 | 6 | kubectl output parsing: CPU/memory, top pods |
| `test_pod_manifest.py` | 775 | 64 | Pod manifest: naming, env, volumes, tolerations, affinity |
| `test_provider.py` | 910 | 40 | KubernetesProvider: sync, logs, capacity, scheduling |

### rpc/ (4 files, 967 lines, 68 tests)

| File | Lines | Tests | Summary |
|------|-------|-------|---------|
| `test_auth.py` | 754 | 54 | Auth: token verification, API keys, middleware |
| `test_errors.py` | 102 | 6 | RPC error handling and mapping |
| `test_interceptors.py` | 60 | 3 | RPC interceptor chain |
| `test_proto_utils.py` | 51 | 5 | Protobuf utility functions |

### Root-level (12 files, 2,834 lines, 157 tests)

| File | Lines | Tests | Summary |
|------|-------|-------|---------|
| `conftest.py` | 160 | 0 | Global fixtures: thread cleanup, sentinel, logging health |
| `test_dev_tpu.py` | 55 | 5 | Dev TPU utilities |
| `test_distributed_lock.py` | 240 | 16 | Distributed lock acquisition/release |
| `test_env_resources.py` | 107 | 8 | Environment resource detection |
| `test_iris_run.py` | 351 | 19 | Iris run command and configuration |
| `test_jax_init.py` | 238 | 10 | JAX initialization |
| `test_jax_init_integration.py` | 306 | 7 | JAX init integration with real envs |
| `test_logging.py` | 195 | 10 | Logging framework |
| `test_marin_fs.py` | 431 | 33 | Filesystem abstraction |
| `test_mirror_fs.py` | 106 | 7 | Mirror filesystem |
| `test_query.py` | 146 | 8 | Query parsing |
| `test_time_utils.py` | 411 | 27 | Time utilities: Timestamp, Duration, Deadline |
| `test_utils.py` | 198 | 7 | General utilities |

---

## 2. Test Overlap Analysis

### Mostly Distinct — Same Names, Different Layers

The codebase is well-factored: files with similar names test different layers:

- **`cluster/controller/test_bundle_store.py`** (6 tests) — controller-side write/read/idempotency
- **`cluster/worker/test_bundle_store.py`** (4 tests) — worker-side extraction, LRU cache
- → **No overlap.** Different concerns (server vs client).

- **`cluster/controller/test_dashboard.py`** (32 tests) — controller dashboard RPC
- **`cluster/worker/test_dashboard.py`** (8 tests) — worker dashboard RPC
- → **No overlap.** Different services.

- **`cluster/controller/test_vm_lifecycle.py`** (9 tests) — GCP VM start/stop with mocked subprocess
- **`e2e/test_vm_lifecycle.py`** (3 tests) — FakePlatform-based failure injection
- → **Complementary.** Unit vs integration level.

### Potential Overlap Groups

#### Group A: Scheduling & Resource Allocation

| File | Tests | Focus |
|------|-------|-------|
| `controller/test_scheduler.py` | 51 | Core scheduling algorithm: constraint matching, resource accounting |
| `controller/test_reservation.py` | 55 | Reservation slots and priority preemption |
| `platform/test_scaling_group.py` | 70 | Scale group demand routing, same constraint matching |
| `controller/test_pending_diagnostics.py` | 3 | Scheduling failure explanations |

Overlap risk: `test_scheduler` and `test_scaling_group` both exercise constraint matching. The scheduler tests focus on worker→task assignment; scaling group tests focus on demand→group routing. Some constraint expression tests may test the same parsing logic.

#### Group B: Job Lifecycle State Machine

| File | Tests | Focus |
|------|-------|-------|
| `controller/test_transitions.py` | 104 | Full state machine via ControllerTransitions |
| `controller/test_service.py` | 38 | Same transitions via ControllerService RPC layer |
| `controller/test_job.py` | 5 | Job row helpers |
| `controller/test_direct_controller.py` | 13 | DirectProvider integration |

Overlap risk: `test_transitions` and `test_service` both test submit→schedule→dispatch→complete paths. The service tests go through the RPC proto layer; transitions tests call the state machine directly. Some scenarios (launch, terminate, list) appear in both.

#### Group C: Autoscaler + Platform Lifecycle

| File | Tests | Focus |
|------|-------|-------|
| `controller/test_autoscaler.py` | 132 | Autoscaler decisions with FakePlatform |
| `controller/test_heartbeat.py` | 8 | Heartbeat timeout → worker failure |
| `controller/test_vm_lifecycle.py` | 9 | VM creation/replacement |
| `e2e/test_vm_lifecycle.py` | 3 | VM lifecycle failure scenarios |
| `cluster/test_snapshot_reconciliation.py` | 15 | State reconciliation from platform snapshot |

Low overlap — each file tests a distinct aspect, though heartbeat/failure scenarios touch similar state transitions.

#### Group D: Auth

| File | Tests | Focus |
|------|-------|-------|
| `rpc/test_auth.py` | 54 | Token verification, API key middleware |
| `controller/test_api_keys.py` | 22 | API key CRUD and rotation |
| `controller/test_dashboard.py` | 2 | Auth config endpoint |

Minimal overlap — auth middleware vs key management vs config reporting.

#### Group E: Config Validation

| File | Tests | Focus |
|------|-------|-------|
| `platform/test_config.py` | 55 | Scale group config across all providers |
| `platform/test_scaling_group.py` | 70 | Scale group behavior with validated config |

Some overlap in config validation, but test_config focuses on invalid configs / error paths while test_scaling_group focuses on runtime behavior.

---

## 3. Fixture Analysis

### Expensive Resources Created Per Test

| Resource | Creation Cost | Files Using It |
|----------|--------------|----------------|
| `ControllerDB` (SQLite) | Cheap (tmp_path) | 15+ controller/ files via `make_controller_state` |
| `LogStore` (DuckDB) | Moderate | 15+ controller/ files via `make_controller_state` |
| `ControllerTransitions` | Moderate (wraps DB + LogStore) | 10 files via `make_controller_state` |
| `ControllerServiceImpl` | Moderate (wraps Transitions + Scheduler) | `test_service.py`, `test_dashboard.py`, `test_api_keys.py` |
| `FakePlatform` | Cheap (in-memory) | `test_autoscaler.py`, `test_platform.py`, `e2e/test_vm_lifecycle.py` |
| `LocalCluster` (Controller + Workers) | **Expensive** (threads, ports, temp dirs) | `test_local_client.py`, `cli/test_local_cluster.py`, all e2e/ |
| `Worker` instance | Moderate (port allocation, runtime) | `test_worker.py`, `worker/test_dashboard.py` |
| `ActorServer` | Cheap (bind port) | All actor/ files |
| `KubernetesProvider` | Cheap (mocked kubectl) | `kubernetes/test_provider.py`, `platform/conftest.py` |
| Dashboard npm build | **Very expensive** (session-scoped) | `e2e/conftest.py` — built once per session |

### Fixture Sharing Opportunities

**Already well-shared:**
- `make_controller_state()` in `cluster/controller/conftest.py` — used by 10+ test files
- `FakePlatform`, `FakeGcloud`, `FakeKubectl` in `cluster/platform/fakes.py` — centralized fakes
- Global `conftest.py` — thread cleanup, logging health

**Could be improved:**
1. **ControllerServiceImpl setup** is repeated in `test_service.py`, `test_dashboard.py`, `test_api_keys.py` with slightly different configurations. A parametrized fixture in `conftest.py` could reduce duplication.
2. **Worker fixture** creation in `test_worker.py` involves constructing a Worker with mock runtime, bundle store, and port allocator. The `cluster/worker/conftest.py` only has a `docker_runtime` fixture — it could host a shared `mock_worker` fixture.
3. **E2e cluster fixtures** (`cluster`, `smoke_cluster`, `multi_worker_cluster`) share 80% of setup code. The `IrisTestCluster` wrapper helps, but the underlying `connect_cluster` context manager is configured slightly differently each time.

---

## 4. Current Service Boundaries

### Service Boundary Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Test Layer                                    │
│                                                                       │
│  FakePlatform ◄── test_autoscaler, test_platform, e2e/test_vm_lifecycle │
│  FakeGcloud  ◄── test_platform (GCP path)                            │
│  FakeKubectl ◄── test_coreweave_platform, kubernetes/*               │
│  MockRuntime ◄── test_worker                                          │
│  LocalPlatform ◄── e2e/* (real workers as threads)                   │
└──────────────────────────────────────────────────────────────────────┘
         │                │               │              │
         ▼                ▼               ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌─────────────┐ ┌──────────────┐
│   Platform   │ │  Controller  │ │   Worker     │ │    K8s       │
│   Protocol   │ │              │ │              │ │  Provider    │
│              │ │ Transitions  │ │ Task phases  │ │              │
│ create_slice │ │ Scheduler    │ │ Port alloc   │ │ Pod manifests│
│ list_slices  │ │ Autoscaler   │ │ Bundle store │ │ kubectl exec │
│ create_vm    │ │ Dashboard    │ │ Env probe    │ │ Log fetch    │
│ tunnel       │ │ Service RPC  │ │ Dashboard    │ │              │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │                │
       ▼                ▼                ▼                ▼
┌──────────────────────────────────────────────────────────────────┐
│                    External Services                              │
│                                                                   │
│  gcloud tpu/compute (subprocess)  ──── GcpPlatform               │
│  kubectl (subprocess)             ──── CoreweavePlatform, K8sProv │
│  ssh (subprocess)                 ──── ManualPlatform             │
│  Docker daemon (subprocess)       ──── DockerRuntime              │
│  GCP metadata server (HTTP)       ──── env_probe                 │
│  SQLite (file)                    ──── ControllerDB              │
│  DuckDB (file)                    ──── LogStore                  │
│  Filesystem (tmp_path)            ──── BundleStore               │
└──────────────────────────────────────────────────────────────────┘
```

### GCP TPU Operations

**Wrapper**: `GcpPlatform` in `src/iris/cluster/platform/gcp.py` (~1500 lines)

All GCP calls go through subprocess execution of `gcloud`:
- `gcloud alpha compute tpus tpu-vm create` — slice creation
- `gcloud alpha compute tpus tpu-vm delete` — slice deletion
- `gcloud alpha compute tpus tpu-vm describe` — status query
- `gcloud alpha compute tpus tpu-vm list` — listing
- `gcloud compute instances create/delete/list/describe` — GCE VMs
- `gcloud compute ssh` — remote execution and tunneling

**Fake**: `FakeGcloud` in `tests/cluster/platform/fakes.py` — intercepts subprocess.run calls, maintains in-memory VM/TPU state, simulates create/delete/list/describe responses as JSON.

### K8s Operations

**Wrapper**: `Kubectl` class in `src/iris/cluster/k8s/kubectl.py`

All K8s calls go through subprocess execution of `kubectl`:
- `kubectl apply -f` — resource creation
- `kubectl delete` — resource deletion
- `kubectl get -o json` — resource queries
- `kubectl logs` — log fetching
- `kubectl port-forward` — tunneling
- `kubectl rollout status` — deployment readiness

**Fakes**:
- `FakeKubectl` in `tests/cluster/platform/fakes.py` — records applied manifests, serves from in-memory dict
- `mock_kubectl` in `tests/kubernetes/conftest.py` — MagicMock for simpler test scenarios

### Local Cluster

`LocalPlatform` (`src/iris/cluster/platform/local.py`, ~580 lines) provides two modes:

1. **Unit test mode** (no `controller_address`): `create_slice()` returns stubs with fake workers. Workers are in-memory objects that don't actually execute tasks.
2. **E2E mode** (with `controller_address`): Creates real `Worker` threads that register with a real Controller, execute tasks via `ProcessRuntime`, and use real `BundleStore` with temp directories.

`LocalCluster` (`src/iris/cluster/local_cluster.py`) orchestrates the full in-process stack:
- Creates Controller instance
- Creates LocalPlatform in E2E mode
- Starts Autoscaler
- Manages lifecycle via ThreadContainer

### Existing Fake/Mock Implementations

| Fake | Location | What it simulates |
|------|----------|-------------------|
| `FakePlatform` | `tests/cluster/platform/fakes.py` | Full Platform protocol with tick()-based state machine |
| `FakeGcloud` | `tests/cluster/platform/fakes.py` | gcloud CLI subprocess responses |
| `FakeKubectl` | `tests/cluster/platform/fakes.py` | kubectl CLI subprocess responses |
| `FakeProvider` | `tests/cluster/controller/conftest.py` | Minimal TaskProvider (noop dispatch) |
| `MockRuntime` | `tests/cluster/worker/test_worker.py` | ContainerRuntime with configurable success/failure |
| `FixedEnvironmentProvider` | `src/iris/cluster/worker/env_probe.py` | Hardcoded worker attributes (used in LocalPlatform) |
| `LocalPlatform` | `src/iris/cluster/platform/local.py` | In-process platform (production code, but serves as test fake) |

---

## 5. Platform Abstraction Architecture

### Protocol Hierarchy

```
Platform (Protocol) — 15 methods
├── RemoteWorkerHandle (Protocol) — single worker: status, run_command, reboot
│   └── StandaloneWorkerHandle (Protocol) — + terminate, set_labels, set_metadata
└── SliceHandle (Protocol) — atomic group: describe, terminate

4 implementations:
├── GcpPlatform      → GcpSliceHandle, GcpWorkerHandle, GcpStandaloneWorkerHandle
├── CoreweavePlatform → (uses K8s pods, no traditional slices)
├── ManualPlatform   → ManualSliceHandle, ManualWorkerHandle
└── LocalPlatform    → LocalSliceHandle, _LocalWorkerHandle, _LocalStandaloneWorkerHandle
```

### Key Differences: Local vs GCP

| Concern | Local | GCP |
|---------|-------|-----|
| Worker creation | Thread in same process | `gcloud compute tpus tpu-vm create` subprocess |
| Remote execution | `subprocess.run` on localhost | `gcloud compute ssh` subprocess |
| Bootstrap | Noop or `ProcessRuntime` start | SSH script execution with retry |
| Slice state | In-memory enum | Parsed from `gcloud describe` JSON |
| Tunneling | Direct localhost | `gcloud compute ssh -L` port forward |
| Image resolution | Identity function | GCP Artifact Registry rewrite |
| Worker discovery | Known localhost ports | GCP metadata + label queries |
| Failure modes | Thread exceptions | SSH timeouts, quota errors, preemption |

### Factory

`create_platform()` in `src/iris/cluster/platform/factory.py` dispatches on `config.cluster_type`:
- `"gcp"` → `GcpPlatform`
- `"coreweave"` → `CoreweavePlatform`
- `"manual"` → `ManualPlatform`
- `"local"` → `LocalPlatform`

---

## 6. E2E Test Gap Analysis

### What Unit Tests Cover Well
- State machine transitions (104 tests in `test_transitions.py`)
- Scheduling logic (51 tests in `test_scheduler.py`, 70 in `test_scaling_group.py`)
- Autoscaler decisions (132 tests in `test_autoscaler.py`)
- RPC API surface (`test_service.py`, `test_dashboard.py`)
- Config validation (55 tests in `test_config.py`)
- Auth (54 + 22 tests)
- K8s manifest generation (64 tests in `test_pod_manifest.py`)

### What Slips Through to Cloud E2E

1. **Real subprocess interaction with `gcloud`/`kubectl`**: Unit tests use FakeGcloud/FakeKubectl which simulate responses but don't validate actual CLI flag combinations, version-specific behavior, or error message formats.

2. **Network/SSH connectivity**: Bootstrap waits for SSH connectivity with backoff. Unit tests skip this entirely. Bugs in SSH key setup, firewall rules, or metadata propagation would be missed.

3. **GCP Artifact Registry image resolution**: `resolve_image()` rewrites GHCR tags to regional AR mirrors. Unit tests mock this; a misconfigured AR mirror or missing image would only surface in cloud.

4. **TPU topology ↔ VM count mapping**: `get_tpu_topology()` maps variant strings to VM counts. If GCP changes topology naming or adds new variants, tests won't catch the mismatch.

5. **Port forwarding / tunneling**: `GcpPlatform.tunnel()` creates SSH tunnels. Local tests use direct localhost connections.

6. **Concurrent worker bootstrap timing**: Manual/GCP platforms bootstrap workers via parallel daemon threads. The thread timing/failure behavior is simplified in fakes.

7. **Kubernetes RBAC and node pool lifecycle**: CoreWeave tests use FakeKubectl which just records manifests. Actual RBAC permission errors, node pool scheduling delays, and pod eviction aren't exercised.

8. **Cross-zone / cross-region behavior**: Autoscaler tests use fake zones. Real cross-region latency, zone-specific quotas, and GCE API rate limits aren't tested.

9. **Controller persistence across restarts**: `test_checkpoint.py` tests save/restore in-memory, but real controller VM restarts (disk state, PID files, port re-acquisition) aren't tested.

10. **Docker image pull and registry auth**: DockerRuntime tests are marked `docker` and mostly test process-based execution. Actual image pulls from GHCR/AR with auth tokens aren't tested in CI.

11. **Log aggregation at scale**: `test_logs.py` tests LogStore with small data. Real DuckDB behavior with millions of log lines, concurrent writes from many workers, and retention policies at scale aren't exercised.

12. **Worker pre-emption recovery**: `test_autoscaler.py` simulates preemption via FakePlatform state changes, but real GCP preemption signals (maintenance events, capacity reclaims) have specific timing characteristics.

### Specific Gap Categories

| Gap | Impact | Mitigation |
|-----|--------|------------|
| gcloud CLI changes | Broken deployments | Pin gcloud version, add smoke test with real gcloud |
| SSH bootstrap failure | Workers stuck in BOOTSTRAPPING | E2E test with real SSH (exists in cloud mode) |
| AR image resolution | Image not found on workers | Integration test with real AR registry |
| K8s RBAC | Pods fail to create | Integration test on kind/k3d cluster |
| TPU create race | Duplicate slices | Hard to test without real TPU quota |
| DuckDB concurrent writes | Log corruption | Stress test with multi-threaded writer |
| Controller restart | State loss | Integration test with process restart |

---

## 7. AGENTS.md / TESTING.md Guidance

From `lib/iris/AGENTS.md`:
- Unit tests: `uv run pytest lib/iris/tests/ -m "not e2e" -o "addopts="`
- E2E tests: `uv run pytest lib/iris/tests/e2e/ -m e2e -o "addopts="`
- Uses `iris.time_utils` for all time operations
- Fakes preferred over mocks

From `lib/iris/TESTING.md`:
- Test stable behavior, not implementation details
- No tests for constructor round-trips, attribute existence, or constants
- No assertions on `_`-prefixed attributes
- No `assert_called_once_with` on internal helpers (only on external boundaries like subprocess)
- Use fakes backed by in-memory state over mocks where reasonable
- Shared fakes live in `tests/cluster/platform/fakes.py` or `src/iris/test_util.py`
- E2E tests marked `@pytest.mark.e2e`, docker tests also `@pytest.mark.docker`
- `test_smoke.py` uses module-scoped shared cluster; `test_chaos.py` uses function-scoped fresh cluster
- No `time.sleep()` in polling — use `Deadline`, `ExponentialBackoff.wait_until()`, or `wait_for_condition`
- No permanently skipped tests

---

## 8. Summary Recommendations

### Fixture Sharing
1. Extract `ControllerServiceImpl` setup from `test_service.py`/`test_dashboard.py`/`test_api_keys.py` into a shared fixture in `conftest.py`.
2. Add a `mock_worker` fixture to `cluster/worker/conftest.py` for the Worker+MockRuntime+PortAllocator combo.
3. Consider module-scoping the `make_controller_state` fixture for test files that don't mutate state between tests.

### Overlap Reduction
1. `test_scheduler.py` and `test_scaling_group.py` both test constraint matching. If these test the same `Constraint` evaluation code, consider consolidating constraint expression tests.
2. `test_transitions.py` and `test_service.py` cover overlapping submit/terminate/list paths at different layers. This is intentional (unit vs integration) and should be kept, but document the layering.

### Gap Priorities
1. **Kind/k3d integration test** for KubernetesProvider — would catch RBAC and manifest issues without real cloud.
2. **Controller restart test** — save checkpoint, kill process, restore, verify state.
3. **Concurrent LogStore writer stress test** — exercise DuckDB under write contention.
4. **Bootstrap SSH test** with real subprocess (not gcloud, but raw SSH to localhost) — would catch command-string bugs.
