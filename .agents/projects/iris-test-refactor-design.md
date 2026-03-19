# Iris Test Refactor Design

## Problem

The Iris test suite (76 files, ~1418 tests, ~38K lines) has three structural problems:

1. **Overlapping test coverage** — Groups A-E from the audit show duplicate constraint-matching tests across `test_scheduler.py` and `test_scaling_group.py`, and overlapping job-lifecycle tests between `test_transitions.py` and `test_service.py`.

2. **No clean service boundary below Platform** — `GcpPlatform` (`lib/iris/src/iris/cluster/platform/gcp.py:681`, ~1050 lines) makes ~50 direct `subprocess.run("gcloud", ...)` calls. `CoreweavePlatform` (`lib/iris/src/iris/cluster/platform/coreweave.py:130`) uses `Kubectl` wrapper (`lib/iris/src/iris/cluster/k8s/kubectl.py:61`). Testing either requires patching subprocess or using MagicMock, neither of which validates request structure.

3. **Fake/real divergence** — `FakePlatform` (`lib/iris/tests/cluster/platform/fakes.py:245`) operates at the Platform protocol level (too high), while `FakeGcloud` (`lib/iris/tests/cluster/platform/fakes.py:426`) operates at subprocess level (too low). No fake exists at the right abstraction — the *service* level between Platform and subprocess.

## Proposed Solution

Three workstreams, two of which can proceed in parallel:

```
Workstream 1 (cleanup)  ──────────────────────────────────►
Workstream 2 (protocols) ──────► Workstream 3 (fakes) ────►
```

### Why this layering

The current architecture:
```
Platform (GcpPlatform/CoreweavePlatform)
    └── subprocess.run("gcloud ...")/Kubectl.run(subprocess)
```

The target architecture:
```
Platform (GcpPlatform/CoreweavePlatform)
    └── GcpService / K8sService (Protocol)
        └── RealGcpService(subprocess) / RealK8sService(subprocess)
        └── FakeGcpService(in-memory) / FakeK8sService(in-memory)
```

This puts the testing seam at the service boundary, where requests are structured (not raw strings) and responses are typed (not raw JSON). FakeGcloud currently does this at subprocess level, which means every test must construct the exact `gcloud` CLI flag format.

### Why not just improve FakePlatform?

FakePlatform hides all the GCP/K8s-specific logic — zone routing, image resolution, label filtering, bootstrap sequencing. Tests using FakePlatform can't verify that GcpPlatform correctly handles quota errors from zone A and retries in zone B. The service-level fake preserves this logic while eliminating subprocess.

---

## Workstream 1: Unit Test Cleanup

### Group A: Scheduling & Constraints

**Files:**
- `lib/iris/tests/cluster/controller/test_scheduler.py` (51 tests, 2150 lines)
- `lib/iris/tests/cluster/platform/test_scaling_group.py` (70 tests, 1547 lines)
- `lib/iris/tests/cluster/test_constraints.py` (9 tests, 161 lines)
- `lib/iris/tests/cluster/controller/test_pending_diagnostics.py` (3 tests, 62 lines)

**Analysis:** `test_scheduler.py` and `test_scaling_group.py` both test constraint matching but at different levels — scheduler matches workers→tasks, scaling groups match demand→groups. `test_constraints.py` tests constraint expression evaluation in isolation.

**Actions:**
- **Keep all three files.** The overlap is intentional layering (unit vs integration).
- **Extract shared constraint fixtures** from `test_scheduler.py` and `test_scaling_group.py` into `lib/iris/tests/cluster/conftest.py`. Currently both files construct `ResourceSpec`, `Constraint`, and worker attribute dicts inline.
- **Merge `test_pending_diagnostics.py`** (3 tests, 62 lines) into `test_scheduler.py` — pending diagnostics are a scheduler feature and the tests share the same setup.

**Shared fixtures to add to `lib/iris/tests/cluster/conftest.py`:**
```python
@pytest.fixture
def cpu_resource_spec() -> cluster_pb2.ResourceSpecProto:
    """Standard CPU resource spec for scheduling tests."""
    return cluster_pb2.ResourceSpecProto(
        cpu_millicores=1000, memory_bytes=4 * 1024**3
    )

@pytest.fixture
def gpu_resource_spec() -> cluster_pb2.ResourceSpecProto:
    """GPU resource spec with device type constraint."""
    spec = cluster_pb2.ResourceSpecProto(
        cpu_millicores=1000, memory_bytes=4 * 1024**3
    )
    spec.device_type = config_pb2.ACCELERATOR_TYPE_GPU
    return spec

def make_worker_attrs(
    region: str = "us-central1",
    device_type: str = "cpu",
    **extras: str,
) -> dict[str, cluster_pb2.AttributeValue]:
    """Build worker attributes dict for scheduling tests."""
    ...
```

**Estimated reduction:** ~65 lines from merging pending_diagnostics, ~100 lines from deduplicating constraint setup across files. Net: ~165 lines.

### Group B: Job Lifecycle State Machine

**Files:**
- `lib/iris/tests/cluster/controller/test_transitions.py` (104 tests, 3873 lines)
- `lib/iris/tests/cluster/controller/test_service.py` (38 tests, 1098 lines)
- `lib/iris/tests/cluster/controller/test_job.py` (5 tests, 176 lines)
- `lib/iris/tests/cluster/controller/test_direct_controller.py` (13 tests, 355 lines)

**Analysis:** `test_transitions.py` exercises the state machine directly via `ControllerTransitions`. `test_service.py` exercises the same transitions through the RPC layer (`ControllerServiceImpl`). Some submit→complete scenarios appear in both — this is intentional (unit vs integration layering), not duplicate coverage.

**Actions:**
- **Keep both files.** The layering is correct per TESTING.md ("integration-style tests which exercise behavior").
- **Extract `ControllerServiceImpl` fixture** — currently `test_service.py`, `test_dashboard.py`, and `test_api_keys.py` each construct `ControllerServiceImpl` with slightly different configs. Extract to `lib/iris/tests/cluster/controller/conftest.py`.
- **Merge `test_job.py`** (5 tests, 176 lines) into `test_transitions.py` — these test job row helpers which are consumed by transitions.

**Shared fixture to add to `lib/iris/tests/cluster/controller/conftest.py`:**
```python
@pytest.fixture
def controller_service(fake_provider, tmp_path) -> ControllerServiceImpl:
    """Create a ControllerServiceImpl with fresh DB, log store, and fake provider."""
    state = make_controller_state()
    scheduler = Scheduler(state)
    return ControllerServiceImpl(
        transitions=state,
        scheduler=scheduler,
        provider=fake_provider,
    )
```

**Estimated reduction:** ~180 lines from merging test_job.py and ~90 lines from deduplicating service construction. Net: ~270 lines.

### Group C: Autoscaler + Platform Lifecycle

**Files:**
- `lib/iris/tests/cluster/controller/test_autoscaler.py` (132 tests, 3833 lines)
- `lib/iris/tests/cluster/controller/test_heartbeat.py` (8 tests, 282 lines)
- `lib/iris/tests/cluster/controller/test_vm_lifecycle.py` (9 tests, 325 lines)
- `lib/iris/tests/e2e/test_vm_lifecycle.py` (3 tests, 93 lines)
- `lib/iris/tests/cluster/test_snapshot_reconciliation.py` (15 tests, 367 lines)

**Analysis:** Low overlap. Each tests a distinct aspect: autoscaler decisions, heartbeat detection, VM creation/replacement, failure injection, state reconciliation.

**Actions:**
- **No merges.** Tests are complementary.
- **Share `FakePlatformConfig` factory fixture** — `test_autoscaler.py` and `test_vm_lifecycle.py` both construct `FakePlatformConfig` with similar boilerplate. Add a parameterizable factory to `lib/iris/tests/cluster/platform/conftest.py`.

**Estimated reduction:** ~40 lines from config factory dedup. Minimal.

### Group D: Auth

**Files:**
- `lib/iris/tests/rpc/test_auth.py` (54 tests, 754 lines)
- `lib/iris/tests/cluster/controller/test_api_keys.py` (22 tests, 489 lines)

**Analysis:** Minimal overlap — auth middleware vs key management. Different concerns.

**Actions:** No changes needed.

### Group E: Config Validation

**Files:**
- `lib/iris/tests/cluster/platform/test_config.py` (55 tests, 1447 lines)
- `lib/iris/tests/cluster/platform/test_scaling_group.py` (70 tests, 1547 lines)

**Analysis:** `test_config.py` focuses on invalid configs/error paths. `test_scaling_group.py` focuses on runtime behavior with valid configs. Some config validation tests in `test_scaling_group.py` could be parameterized.

**Actions:**
- **Identify and consolidate duplicate validation tests** — scan for tests in `test_scaling_group.py` that assert `ValueError` on invalid configs and move them to `test_config.py`.
- **Parameterize provider-specific config validation** in `test_config.py` — many tests follow the pattern "for provider X, setting Y should raise ValueError".

**Estimated reduction:** ~100 lines from parameterization and consolidation.

### Worker Test Cleanup

**Files:**
- `lib/iris/tests/cluster/worker/test_worker.py` (29 tests, 864 lines)
- `lib/iris/tests/cluster/worker/conftest.py` (16 lines — only `docker_runtime` fixture)

**Actions:**
- **Extract shared `mock_worker` fixture** to `lib/iris/tests/cluster/worker/conftest.py`:

```python
@pytest.fixture
def mock_runtime() -> MockRuntime:
    return MockRuntime()

@pytest.fixture
def mock_worker(tmp_path, mock_runtime) -> Worker:
    """Construct a Worker with mock runtime and local bundle store."""
    port_allocator = PortAllocator(base_port=find_free_port())
    bundle_store = BundleStore(tmp_path / "bundles")
    return Worker(
        runtime=mock_runtime,
        bundle_store=bundle_store,
        port_allocator=port_allocator,
    )
```

**Estimated reduction:** ~50 lines from deduplicating Worker construction.

### Summary: Workstream 1 Totals

| Action | Files Touched | Lines Saved |
|--------|--------------|-------------|
| Merge `test_pending_diagnostics.py` → `test_scheduler.py` | 2 (1 deleted) | ~65 |
| Merge `test_job.py` → `test_transitions.py` | 2 (1 deleted) | ~180 |
| Extract constraint fixtures to `cluster/conftest.py` | 3 | ~100 |
| Extract `ControllerServiceImpl` fixture | 4 | ~90 |
| Parameterize config validation | 2 | ~100 |
| Extract `mock_worker` fixture | 2 | ~50 |
| **Total** | **~12 files** | **~585 lines** |

---

## Workstream 2: Service Boundary Protocols

### GcpService Protocol

The `GcpPlatform` class at `lib/iris/src/iris/cluster/platform/gcp.py:681` makes direct subprocess calls for TPU operations (create/delete/list/describe), GCE VM operations (create/delete/list/describe/update), SSH, and metadata. These should be behind a protocol.

**Proposed protocol** — `lib/iris/src/iris/cluster/platform/gcp_service.py`:

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol

from iris.time_utils import Timestamp


@dataclass
class TpuInfo:
    """Parsed TPU state from GCP API."""
    name: str
    state: str  # "CREATING", "READY", "DELETING", etc.
    accelerator_type: str
    zone: str
    labels: dict[str, str]
    metadata: dict[str, str]
    network_endpoints: list[str]  # IP addresses
    created_at: Timestamp


@dataclass
class VmInfo:
    """Parsed GCE VM state from GCP API."""
    name: str
    status: str  # "RUNNING", "TERMINATED", etc.
    zone: str
    internal_ip: str
    external_ip: str | None
    labels: dict[str, str]
    metadata: dict[str, str]


@dataclass
class TpuCreateRequest:
    """Parameters for creating a TPU slice."""
    name: str
    zone: str
    accelerator_type: str
    runtime_version: str
    labels: dict[str, str]
    metadata: dict[str, str]
    preemptible: bool
    service_account: str | None = None
    network: str | None = None
    subnetwork: str | None = None


@dataclass
class VmCreateRequest:
    """Parameters for creating a GCE VM."""
    name: str
    zone: str
    machine_type: str
    labels: dict[str, str]
    metadata: dict[str, str]
    startup_script: str | None = None
    service_account: str | None = None
    disk_size_gb: int = 200
    image_family: str = "cos-stable"
    image_project: str = "cos-cloud"


class GcpService(Protocol):
    """Service boundary for GCP operations.

    All methods raise PlatformError (or subclass) on failure.
    Implementations: SubprocessGcpService (real gcloud), FakeGcpService (in-memory).
    """

    def tpu_create(self, request: TpuCreateRequest) -> TpuInfo: ...
    def tpu_delete(self, name: str, zone: str) -> None: ...
    def tpu_describe(self, name: str, zone: str) -> TpuInfo | None: ...
    def tpu_list(self, zones: list[str], labels: dict[str, str] | None = None) -> list[TpuInfo]: ...

    def vm_create(self, request: VmCreateRequest) -> VmInfo: ...
    def vm_delete(self, name: str, zone: str) -> None: ...
    def vm_describe(self, name: str, zone: str) -> VmInfo | None: ...
    def vm_list(self, zones: list[str], labels: dict[str, str] | None = None) -> list[VmInfo]: ...
    def vm_update_labels(self, name: str, zone: str, labels: dict[str, str]) -> None: ...
    def vm_set_metadata(self, name: str, zone: str, metadata: dict[str, str]) -> None: ...
    def vm_get_serial_port_output(self, name: str, zone: str, start: int = 0) -> str: ...
```

**Key design decisions:**

1. **Structured request/response objects** instead of raw dicts/CLI flags. `TpuCreateRequest` captures all parameters needed for `gcloud compute tpus tpu-vm create`. `TpuInfo` captures the parsed JSON response. This makes fake validation trivial.

2. **SSH/remote execution stays separate.** The existing `RemoteExec` protocol (`lib/iris/src/iris/cluster/platform/remote_exec.py:35`) already abstracts SSH. `GcpService` handles resource CRUD; `RemoteExec` handles command execution. No change needed to `RemoteExec`.

3. **No image resolution in GcpService.** Image resolution (`rewrite_ghcr_to_ar_remote` at `lib/iris/src/iris/cluster/platform/bootstrap.py`) is a string transformation, not a GCP API call. It stays in `GcpPlatform`.

### K8sService Protocol

`Kubectl` at `lib/iris/src/iris/cluster/k8s/kubectl.py:61` already provides a clean interface. However, it's a concrete class, not a protocol. Both `CoreweavePlatform` and `KubernetesProvider` depend on it directly.

**Proposed protocol** — `lib/iris/src/iris/cluster/k8s/k8s_service.py`:

```python
from __future__ import annotations
import subprocess
from typing import Protocol

from iris.cluster.k8s.kubectl import KubectlLogResult


class K8sService(Protocol):
    """Protocol for Kubernetes operations.

    Implementations: Kubectl (real subprocess), FakeK8sService (in-memory).
    Consumed by CoreweavePlatform and KubernetesProvider.
    """

    @property
    def namespace(self) -> str: ...

    def apply_json(self, manifest: dict) -> None: ...
    def get_json(self, resource: str, name: str, *, cluster_scoped: bool = False) -> dict | None: ...
    def list_json(self, resource: str, *, labels: dict[str, str] | None = None, cluster_scoped: bool = False) -> list[dict]: ...
    def delete(self, resource: str, name: str, *, cluster_scoped: bool = False, force: bool = False, wait: bool = True) -> None: ...
    def logs(self, pod_name: str, *, container: str | None = None, tail: int = 50, previous: bool = False) -> str: ...
    def stream_logs(self, pod_name: str, *, container: str | None = None, byte_offset: int = 0) -> KubectlLogResult: ...
    def exec(self, pod_name: str, cmd: list[str], *, container: str | None = None, timeout: float | None = None) -> subprocess.CompletedProcess[str]: ...
    def set_image(self, resource: str, name: str, container: str, image: str, *, namespaced: bool = False) -> None: ...
    def rollout_restart(self, resource: str, name: str, *, namespaced: bool = False) -> None: ...
    def rollout_status(self, resource: str, name: str, *, timeout: float = 600.0, namespaced: bool = False) -> None: ...
    def get_events(self, field_selector: str | None = None) -> list[dict]: ...
    def top_pod(self, pod_name: str) -> tuple[int, int] | None: ...
```

**Key design decision:** K8sService is a thin protocol that matches `Kubectl`'s existing public API. This is the right level — it already structures request parameters and handles JSON parsing. The protocol just enables swapping the implementation.

### Refactoring GcpPlatform

`GcpPlatform.__init__` at `lib/iris/src/iris/cluster/platform/gcp.py:681` currently takes `gcp_config` and constructs subprocess commands inline. After refactoring:

```python
@dataclass
class GcpPlatform:
    def __init__(
        self,
        gcp_config: config_pb2.GcpPlatformConfig,
        label_prefix: str,
        ssh_config: config_pb2.SshConfig | None = None,
        gcp_service: GcpService | None = None,  # NEW: injectable
    ):
        self._gcp = gcp_service or SubprocessGcpService(
            project_id=gcp_config.project_id,
        )
        ...
```

Every `subprocess.run(["gcloud", ...])` call in `GcpPlatform` methods becomes a call to `self._gcp.tpu_create(...)`, `self._gcp.vm_list(...)`, etc.

### Refactoring CoreweavePlatform and KubernetesProvider

Both already use `Kubectl`. The refactoring is mechanical: change the type annotation from `Kubectl` to `K8sService`.

- `lib/iris/src/iris/cluster/platform/coreweave.py:148` — `self._kubectl: K8sService`
- `lib/iris/src/iris/cluster/k8s/provider.py:597` — `kubectl: K8sService`

### Files to create/modify

| File | Action |
|------|--------|
| `lib/iris/src/iris/cluster/platform/gcp_service.py` | **Create** — `GcpService` protocol, `TpuInfo`, `VmInfo`, `TpuCreateRequest`, `VmCreateRequest` dataclasses |
| `lib/iris/src/iris/cluster/platform/gcp_service_subprocess.py` | **Create** — `SubprocessGcpService` implementation (extracted from GcpPlatform) |
| `lib/iris/src/iris/cluster/k8s/k8s_service.py` | **Create** — `K8sService` protocol |
| `lib/iris/src/iris/cluster/platform/gcp.py` | **Modify** — inject `GcpService`, replace all subprocess calls |
| `lib/iris/src/iris/cluster/platform/coreweave.py` | **Modify** — type `self._kubectl` as `K8sService` |
| `lib/iris/src/iris/cluster/k8s/provider.py` | **Modify** — type `kubectl` field as `K8sService` |
| `lib/iris/src/iris/cluster/platform/factory.py` | **Modify** — pass `GcpService` through factory if needed |

---

## Workstream 3: Fake Service Implementation

### FakeGcpService

**Location:** `lib/iris/tests/cluster/platform/fake_gcp_service.py`

Replaces `FakeGcloud` (`lib/iris/tests/cluster/platform/fakes.py:426`). Key differences:

1. **Typed requests** — validates `TpuCreateRequest` fields, not raw CLI flag strings
2. **Request validation** — rejects unknown accelerator types, missing zones, etc.
3. **Failure injection** — quota exhaustion, resource not found, transient errors
4. **No subprocess patching** — no `unittest.mock.patch("subprocess.run")`

```python
@dataclass
class FakeGcpConfig:
    """Configuration for FakeGcpService behavior."""
    available_zones: list[str] = field(default_factory=lambda: ["us-central1-a", "us-central2-b"])
    valid_accelerator_types: list[str] = field(default_factory=lambda: ["v4-8", "v5litepod-16", "v5litepod-256"])
    zone_quotas: dict[str, int] = field(default_factory=dict)  # zone -> max TPUs


class FakeGcpService:
    """In-memory GcpService for testing.

    Validates requests, maintains TPU/VM state, supports failure injection.
    """

    def __init__(self, config: FakeGcpConfig | None = None):
        self._config = config or FakeGcpConfig()
        self._tpus: dict[tuple[str, str], TpuInfo] = {}  # (name, zone) -> info
        self._vms: dict[tuple[str, str], VmInfo] = {}
        self._failures: dict[str, PlatformError] = {}

    def inject_failure(self, operation: str, error: PlatformError) -> None:
        """Make the next call to `operation` raise `error`."""
        self._failures[operation] = error

    def tpu_create(self, request: TpuCreateRequest) -> TpuInfo:
        if err := self._failures.pop("tpu_create", None):
            raise err
        if request.zone not in self._config.available_zones:
            raise PlatformError(f"Zone {request.zone} not available")
        if request.accelerator_type not in self._config.valid_accelerator_types:
            raise ResourceNotFoundError(f"Unknown accelerator: {request.accelerator_type}")
        # Check quota
        zone_count = sum(1 for (_, z) in self._tpus if z == request.zone)
        max_quota = self._config.zone_quotas.get(request.zone, 100)
        if zone_count >= max_quota:
            raise QuotaExhaustedError(f"Quota exhausted in {request.zone}")
        # Create
        topology = get_tpu_topology(request.accelerator_type)
        endpoints = [f"10.0.{len(self._tpus)}.{i}" for i in range(topology.vm_count)]
        info = TpuInfo(
            name=request.name, state="READY",
            accelerator_type=request.accelerator_type,
            zone=request.zone, labels=dict(request.labels),
            metadata=dict(request.metadata),
            network_endpoints=endpoints, created_at=Timestamp.now(),
        )
        self._tpus[(request.name, request.zone)] = info
        return info
    ...
```

### FakeK8sService

**Location:** `lib/iris/tests/cluster/platform/fake_k8s_service.py`

Replaces the `mock_kubectl` MagicMock in `lib/iris/tests/kubernetes/conftest.py:20`. Key improvement: validates manifest structure and maintains state.

```python
@dataclass
class FakeK8sConfig:
    """Configuration for FakeK8sService."""
    available_node_pools: list[str] = field(default_factory=list)
    pod_scheduling_delay_ms: int = 0


class FakeK8sService:
    """In-memory K8sService for testing.

    Tracks applied manifests, serves pods/resources from in-memory state,
    validates resource types and manifest structure.
    """

    def __init__(self, namespace: str = "iris", config: FakeK8sConfig | None = None):
        self._namespace = namespace
        self._config = config or FakeK8sConfig()
        self._resources: dict[tuple[str, str], dict] = {}  # (kind, name) -> manifest
        self._log_cursors: dict[str, int] = {}

    @property
    def namespace(self) -> str:
        return self._namespace

    def apply_json(self, manifest: dict) -> None:
        kind = manifest.get("kind", "").lower()
        name = manifest.get("metadata", {}).get("name", "")
        if not kind or not name:
            raise KubectlError("Invalid manifest: missing kind or metadata.name")
        self._resources[(kind, name)] = manifest

    def get_json(self, resource: str, name: str, *, cluster_scoped: bool = False) -> dict | None:
        return self._resources.get((resource, name))

    def list_json(self, resource: str, *, labels: dict[str, str] | None = None, cluster_scoped: bool = False) -> list[dict]:
        results = []
        for (kind, _), manifest in self._resources.items():
            if kind != resource:
                continue
            if labels:
                res_labels = manifest.get("metadata", {}).get("labels", {})
                if not all(res_labels.get(k) == v for k, v in labels.items()):
                    continue
            results.append(manifest)
        return results
    ...
```

### Relationship to existing fakes

| Existing Fake | Disposition | Rationale |
|---------------|-------------|-----------|
| `FakePlatform` (`fakes.py:245`) | **Keep** | Used by autoscaler tests. Tests autoscaler decisions, not GCP service calls. Different abstraction level. |
| `FakeGcloud` (`fakes.py:426`) | **Deprecate → Remove** | Replaced by `FakeGcpService`. After GcpPlatform uses `GcpService`, tests pass `FakeGcpService` instead of patching subprocess. |
| `mock_kubectl` (`kubernetes/conftest.py:20`) | **Replace** | MagicMock provides no validation. `FakeK8sService` validates manifests and maintains state. |
| `FakeProvider` (`controller/conftest.py:20`) | **Keep** | Tests controller transitions without provider. Different layer. |

### Migration: FakeGcloud → FakeGcpService

The `FakeGcloud` fixture (`fakes.py:783`) is used by:
- `test_platform.py` — GcpPlatform contract tests with FakeGcloud patching subprocess
- `test_coreweave_platform.py` — uses FakeKubectl (MagicMock)
- `controller/test_vm_lifecycle.py` — VM creation with FakeGcloud

After workstream 2 (GcpPlatform takes injectable `GcpService`):

```python
# Before (test_platform.py)
@pytest.fixture
def fake_gcloud():
    fake = FakeGcloud()
    with patch("iris.cluster.platform.gcp.subprocess.run", fake):
        yield fake

def test_create_tpu_slice(fake_gcloud):
    platform = GcpPlatform(gcp_config=..., label_prefix="test")
    ...

# After
@pytest.fixture
def fake_gcp_service():
    return FakeGcpService()

def test_create_tpu_slice(fake_gcp_service):
    platform = GcpPlatform(gcp_config=..., label_prefix="test", gcp_service=fake_gcp_service)
    ...
```

No more `unittest.mock.patch`. No more subprocess interception. Tests construct `GcpPlatform` with the fake service directly.

---

## Implementation Plan

### Task 1.1: Extract shared scheduling fixtures
- **Files:** Create `lib/iris/tests/cluster/conftest.py` with constraint/resource fixtures. Modify `test_scheduler.py`, `test_scaling_group.py`, `test_constraints.py` to use them.
- **Tests:** Run `uv run pytest lib/iris/tests/cluster/controller/test_scheduler.py lib/iris/tests/cluster/platform/test_scaling_group.py lib/iris/tests/cluster/test_constraints.py -m "not e2e" -o "addopts="` — all existing tests must pass unchanged.
- **Scope:** 4 files, ~100 lines changed.

### Task 1.2: Merge test_pending_diagnostics into test_scheduler
- **Files:** Move 3 test functions from `lib/iris/tests/cluster/controller/test_pending_diagnostics.py` into `lib/iris/tests/cluster/controller/test_scheduler.py`. Delete `test_pending_diagnostics.py`.
- **Tests:** Run scheduler tests — all must pass.
- **Scope:** 2 files (1 deleted), ~65 lines.

### Task 1.3: Extract ControllerServiceImpl fixture
- **Files:** Add `controller_service` fixture to `lib/iris/tests/cluster/controller/conftest.py`. Modify `test_service.py`, `test_dashboard.py`, `test_api_keys.py` to use it.
- **Tests:** Run `uv run pytest lib/iris/tests/cluster/controller/test_service.py lib/iris/tests/cluster/controller/test_dashboard.py lib/iris/tests/cluster/controller/test_api_keys.py -m "not e2e" -o "addopts="`.
- **Scope:** 4 files, ~90 lines changed.

### Task 1.4: Merge test_job into test_transitions
- **Files:** Move 5 test functions from `lib/iris/tests/cluster/controller/test_job.py` into `lib/iris/tests/cluster/controller/test_transitions.py`. Delete `test_job.py`.
- **Tests:** Run transitions tests — all must pass.
- **Scope:** 2 files (1 deleted), ~180 lines.

### Task 1.5: Extract mock_worker fixture and parameterize config validation
- **Files:** Add `mock_worker` and `mock_runtime` fixtures to `lib/iris/tests/cluster/worker/conftest.py`. Parameterize duplicate validation in `test_config.py`.
- **Tests:** Run worker tests and config tests.
- **Scope:** 4 files, ~150 lines.

### Task 2.1: Define GcpService protocol and data classes
- **Files:** Create `lib/iris/src/iris/cluster/platform/gcp_service.py` with `GcpService` protocol, `TpuInfo`, `VmInfo`, `TpuCreateRequest`, `VmCreateRequest` dataclasses.
- **Tests:** Type-check with `uv run pyrefly`.
- **Scope:** 1 new file, ~150 lines.
- **Dependencies:** None.

### Task 2.2: Implement SubprocessGcpService
- **Files:** Create `lib/iris/src/iris/cluster/platform/gcp_service_subprocess.py`. Extract all `subprocess.run(["gcloud", ...])` calls from `gcp.py` into methods on `SubprocessGcpService`.
- **Tests:** Existing `test_platform.py` tests must still pass with subprocess-patched approach (temporary — will migrate in task 3.2).
- **Scope:** 1 new file (~400 lines), modify `gcp.py` (~500 lines changed).
- **Dependencies:** Task 2.1.

### Task 2.3: Refactor GcpPlatform to use GcpService
- **Files:** Modify `lib/iris/src/iris/cluster/platform/gcp.py` to accept `GcpService` parameter and delegate all GCP operations to it. Modify `factory.py` if needed.
- **Tests:** All platform tests must pass.
- **Scope:** Modify `gcp.py` (~300 lines changed), `factory.py` (~10 lines).
- **Dependencies:** Task 2.2.

### Task 2.4: Define K8sService protocol
- **Files:** Create `lib/iris/src/iris/cluster/k8s/k8s_service.py` with `K8sService` protocol. Modify `coreweave.py` and `provider.py` to type `kubectl` as `K8sService`.
- **Tests:** Type-check + existing tests.
- **Scope:** 1 new file (~50 lines), modify 2 files (~20 lines each).
- **Dependencies:** None (parallel with Task 2.1).

### Task 3.1: Implement FakeGcpService
- **Files:** Create `lib/iris/tests/cluster/platform/fake_gcp_service.py`.
- **Tests:** Write unit tests for FakeGcpService itself — validate request rejection, quota enforcement, state tracking.
- **Scope:** 1 new file (~300 lines), 1 test file (~150 lines).
- **Dependencies:** Task 2.1.

### Task 3.2: Migrate GcpPlatform tests to FakeGcpService
- **Files:** Modify `lib/iris/tests/cluster/platform/test_platform.py` and `lib/iris/tests/cluster/controller/test_vm_lifecycle.py` to use `FakeGcpService` instead of `FakeGcloud` + subprocess patch.
- **Tests:** All existing platform and VM lifecycle tests must pass.
- **Scope:** 2 files (~200 lines changed).
- **Dependencies:** Tasks 2.3, 3.1.

### Task 3.3: Implement FakeK8sService
- **Files:** Create `lib/iris/tests/cluster/platform/fake_k8s_service.py`.
- **Tests:** Write unit tests for FakeK8sService — validate manifest apply, get, list, delete.
- **Scope:** 1 new file (~250 lines), 1 test file (~100 lines).
- **Dependencies:** Task 2.4.

### Task 3.4: Migrate K8s tests to FakeK8sService
- **Files:** Modify `lib/iris/tests/kubernetes/conftest.py` to provide `FakeK8sService` instead of `mock_kubectl`. Update `test_provider.py` and `test_coreweave_platform.py`.
- **Tests:** All kubernetes and coreweave tests must pass.
- **Scope:** 3-4 files (~300 lines changed).
- **Dependencies:** Tasks 2.4, 3.3.

### Task 3.5: Remove FakeGcloud
- **Files:** Remove `FakeGcloud` class and `fake_gcloud` fixture from `lib/iris/tests/cluster/platform/fakes.py`. Remove associated helpers (`_parse_flag`, `_parse_labels_string`, etc.) if no longer used.
- **Tests:** Full test suite pass.
- **Scope:** 1 file (~400 lines removed).
- **Dependencies:** Task 3.2 (all consumers migrated).

---

## Dependency Graph

```
Task 1.1 ─┐
Task 1.2 ─┤
Task 1.3 ─┼─── (all independent, can run in parallel)
Task 1.4 ─┤
Task 1.5 ─┘

Task 2.1 ──► Task 2.2 ──► Task 2.3 ──┐
Task 2.4 ─────────────────────────────┤
                                       │
Task 2.1 ──► Task 3.1 ───────────────►├── Task 3.2 ──► Task 3.5
Task 2.4 ──► Task 3.3 ──► Task 3.4 ──┘

Workstream 1: [1.1, 1.2, 1.3, 1.4, 1.5] — fully parallel, no deps on WS2/WS3
Workstream 2: 2.1 → 2.2 → 2.3 (GCP chain) | 2.4 (K8s, parallel with GCP)
Workstream 3: depends on WS2 protocols, then: 3.1 → 3.2 → 3.5 | 3.3 → 3.4
```

## Risks and Open Questions

1. **GcpStandaloneWorkerHandle subprocess calls** — `GcpStandaloneWorkerHandle` at `gcp.py:264` makes direct subprocess calls for `status()`, `terminate()`, `set_labels()`, `set_metadata()`. These need to go through `GcpService` too, but the handle holds a reference to the service. This means `GcpService` must be injected into handle construction, which happens inside `create_vm()` and `create_slice()`. This is straightforward but adds complexity to the handle constructors.

2. **RemoteExec stays separate from GcpService** — SSH operations (`run_command`, `bootstrap`) use `RemoteExec` protocol, which is already well-abstracted. Merging SSH into `GcpService` would make the protocol too broad. But this means tests that need both GCP resource operations AND SSH must inject two fakes. Acceptable — the concerns are genuinely separate.

3. **FakeGcpService vs FakePlatform scope** — Tests that only care about autoscaler decisions should continue using `FakePlatform`. Tests that need to verify GcpPlatform's zone-routing, error-classification, or label-management logic should use `FakeGcpService`. Both fakes will coexist permanently.

4. **Kubectl already is the K8sService** — `Kubectl` class already provides exactly the right interface. Introducing a `K8sService` protocol adds a file but no real abstraction gap. The value is purely testability (swapping `FakeK8sService`). If this is deemed over-engineering, skip Task 2.4 and just have `FakeK8sService` be a drop-in `Kubectl` replacement that doesn't subclass it.

5. **E2E test migration** — E2E tests in `tests/e2e/` use `LocalPlatform` with real worker threads. These are NOT candidates for FakeGcpService/FakeK8sService — they test the full stack. The fakes are for unit/integration tests only.

6. **FakeGcloud removal timeline** — FakeGcloud is well-tested and works. Removing it (Task 3.5) should only happen after ALL consumers are migrated and the full test suite passes with FakeGcpService. If migration hits issues, FakeGcloud can coexist temporarily.
