# Iris Test Refactor Design

## Problem

The Iris test suite (76 files, ~1418 tests, ~38K lines) has three structural problems:

1. **Overlapping test coverage** — Groups A-E from the audit show duplicate constraint-matching tests across `test_scheduler.py` and `test_scaling_group.py`, and overlapping job-lifecycle tests between `test_transitions.py` and `test_service.py`.

2. **No clean service boundary below Platform** — `GcpPlatform` (`lib/iris/src/iris/cluster/platform/gcp.py:681`, ~1050 lines) makes ~50 direct `subprocess.run("gcloud", ...)` calls. `CoreweavePlatform` (`lib/iris/src/iris/cluster/platform/coreweave.py:130`) uses `Kubectl` wrapper (`lib/iris/src/iris/cluster/k8s/kubectl.py:61`). Testing either requires patching subprocess or using MagicMock, neither of which validates request structure.

3. **Fake/real divergence** — `FakePlatform` (`lib/iris/tests/cluster/platform/fakes.py:245`) operates at the Platform protocol level (too high), while `FakeGcloud` (`lib/iris/tests/cluster/platform/fakes.py:426`) operates at subprocess level (too low). No fake exists at the right abstraction — the *service* level between Platform and subprocess.

4. **LocalPlatform is a separate Platform implementation** — `LocalPlatform` (`lib/iris/src/iris/cluster/platform/local.py:294`) duplicates the Platform interface for local testing. Its two modes (in-memory stubs vs real worker threads) should be capabilities of the service layer, not a separate platform.

## Proposed Solution

Three workstreams, two of which can proceed in parallel:

```
Workstream 1 (cleanup)  ──────────────────────────────────►
Workstream 2 (protocols) ──────► Workstream 3 (fakes) ────►
```

### Core idea: ServiceMode enum

Instead of separate classes per environment (SubprocessGcpService, FakeGcpService, LocalPlatform), a single `ServiceMode` enum determines behavior:

```python
class ServiceMode(StrEnum):
    DRY_RUN = "dry_run"   # Validate request, return synthetic response. No side effects.
    LOCAL = "local"       # Validate + create real local workers (replaces LocalPlatform).
    CLOUD = "cloud"       # Validate + call real gcloud/kubectl.
```

The mode is set at service construction time, not per-call. A `GcpService` in DRY_RUN mode validates TPU create requests (zone, accelerator type, labels) and returns a synthetic `TpuInfo` — catching 99% of silly config bugs. In LOCAL mode it does the same validation AND spawns real worker threads (extracting the logic from `LocalPlatform._create_slice_with_workers` at `local.py:379`). In CLOUD mode it shells out to gcloud.

### Why this over separate fake classes

1. **Validation parity.** A single validation codepath runs in all modes. DRY_RUN and LOCAL can't drift from CLOUD because they share the same request validation.
2. **LocalPlatform removal.** `GcpPlatform(gcp_service=GcpService(mode=LOCAL))` replaces `LocalPlatform`. E2E tests use `GcpPlatform` with LOCAL-mode service instead of a separate platform class.
3. **Hookable testing surface.** The service exposes methods to inject failures (quota exhaustion, unavailable TPU types, scheduling failures) in DRY_RUN/LOCAL modes. Tests construct `GcpPlatform` with a service they control.

### Target architecture

```
Platform (GcpPlatform / CoreweavePlatform)
    └── GcpService / K8sService  (Protocol)
        └── GcpServiceImpl(mode=CLOUD)   — subprocess calls
        └── GcpServiceImpl(mode=LOCAL)   — validate + local workers
        └── GcpServiceImpl(mode=DRY_RUN) — validate + synthetic responses
```

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

### ServiceMode Enum

Shared enum used by both GCP and K8s service implementations:

```python
# lib/iris/src/iris/cluster/platform/service_mode.py

from enum import StrEnum

class ServiceMode(StrEnum):
    DRY_RUN = "dry_run"   # Validate only, return synthetic responses
    LOCAL = "local"       # Validate + create real local resources
    CLOUD = "cloud"       # Validate + call real cloud APIs
```

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
    Implementations: GcpServiceImpl with ServiceMode determining behavior.
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

1. **Structured request/response objects** instead of raw dicts/CLI flags. `TpuCreateRequest` captures all parameters needed for `gcloud compute tpus tpu-vm create`. `TpuInfo` captures the parsed JSON response. This makes validation in DRY_RUN/LOCAL modes trivial.

2. **SSH/remote execution stays separate.** The existing `RemoteExec` protocol (`lib/iris/src/iris/cluster/platform/remote_exec.py:35`) already abstracts SSH. `GcpService` handles resource CRUD; `RemoteExec` handles command execution. No change needed to `RemoteExec`.

3. **No image resolution in GcpService.** Image resolution (`rewrite_ghcr_to_ar_remote` at `lib/iris/src/iris/cluster/platform/bootstrap.py`) is a string transformation, not a GCP API call. It stays in `GcpPlatform`.

### GcpServiceImpl — Single Class, Three Modes

Instead of separate `SubprocessGcpService` and `FakeGcpService` classes, a single `GcpServiceImpl` handles all modes:

```python
# lib/iris/src/iris/cluster/platform/gcp_service_impl.py

class GcpServiceImpl:
    """GcpService implementation supporting DRY_RUN, LOCAL, and CLOUD modes.

    Validation runs in ALL modes. The mode determines what happens after validation:
    - DRY_RUN: return synthetic response
    - LOCAL: create real local workers via worker thread spawning
    - CLOUD: shell out to gcloud CLI
    """

    def __init__(
        self,
        mode: ServiceMode,
        project_id: str = "",
        # LOCAL mode params (extracted from LocalPlatform constructor)
        controller_address: str | None = None,
        cache_path: Path | None = None,
        port_allocator: PortAllocator | None = None,
        threads: ThreadContainer | None = None,
    ):
        self._mode = mode
        self._project_id = project_id
        # Shared validation state
        self._valid_zones = KNOWN_GCP_ZONES  # or configurable
        self._valid_accelerator_types = KNOWN_TPU_TYPES
        # In-memory state for DRY_RUN/LOCAL
        self._tpus: dict[tuple[str, str], TpuInfo] = {}
        self._vms: dict[tuple[str, str], VmInfo] = {}
        # Failure injection (DRY_RUN/LOCAL only)
        self._injected_failures: dict[str, PlatformError] = {}
        self._zone_quotas: dict[str, int] = {}
        # LOCAL mode worker spawning (from LocalPlatform)
        self._controller_address = controller_address
        self._cache_path = cache_path
        self._port_allocator = port_allocator
        self._threads = threads

    # -- Validation (shared across all modes) --

    def _validate_tpu_create(self, request: TpuCreateRequest) -> None:
        """Validate a TPU create request against known GCP constraints.

        Checks zone, accelerator type, label format, name length.
        Raises PlatformError/ValueError for invalid requests.
        """
        if request.zone not in self._valid_zones:
            raise PlatformError(f"Zone {request.zone} not available")
        if request.accelerator_type not in self._valid_accelerator_types:
            raise ResourceNotFoundError(f"Unknown accelerator: {request.accelerator_type}")
        if len(request.name) > 63:
            raise ValueError(f"TPU name exceeds 63 chars: {request.name!r}")
        for key, val in request.labels.items():
            if not re.match(r'^[a-z][a-z0-9_-]*$', key):
                raise ValueError(f"Invalid label key: {key!r}")

    def _validate_vm_create(self, request: VmCreateRequest) -> None:
        """Validate a VM create request."""
        if request.zone not in self._valid_zones:
            raise PlatformError(f"Zone {request.zone} not available")
        if len(request.name) > 63:
            raise ValueError(f"VM name exceeds 63 chars: {request.name!r}")

    # -- Failure injection (DRY_RUN/LOCAL) --

    def inject_failure(self, operation: str, error: PlatformError) -> None:
        """Make the next call to `operation` raise `error`.

        Only meaningful in DRY_RUN/LOCAL modes. Ignored in CLOUD mode.
        """
        self._injected_failures[operation] = error

    def set_zone_quota(self, zone: str, max_tpus: int) -> None:
        """Set TPU quota for a zone. Enforced in DRY_RUN/LOCAL modes."""
        self._zone_quotas[zone] = max_tpus

    def set_tpu_type_unavailable(self, accelerator_type: str) -> None:
        """Remove an accelerator type from the valid set."""
        self._valid_accelerator_types = self._valid_accelerator_types - {accelerator_type}

    def add_tpu_type(self, accelerator_type: str) -> None:
        """Add an accelerator type to the valid set."""
        self._valid_accelerator_types = self._valid_accelerator_types | {accelerator_type}

    # -- Core operations --

    def tpu_create(self, request: TpuCreateRequest) -> TpuInfo:
        if err := self._injected_failures.pop("tpu_create", None):
            raise err
        self._validate_tpu_create(request)

        if self._mode == ServiceMode.CLOUD:
            return self._tpu_create_cloud(request)

        # DRY_RUN and LOCAL: check quota, then create in-memory
        zone_count = sum(1 for (_, z) in self._tpus if z == request.zone)
        max_quota = self._zone_quotas.get(request.zone, 100)
        if zone_count >= max_quota:
            raise QuotaExhaustedError(f"Quota exhausted in {request.zone}")

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

        if self._mode == ServiceMode.LOCAL:
            self._spawn_local_workers(request, info)

        return info

    def _tpu_create_cloud(self, request: TpuCreateRequest) -> TpuInfo:
        """Shell out to gcloud compute tpus tpu-vm create."""
        # Extracted from GcpPlatform._create_tpu_slice (gcp.py:870-898)
        ...

    def _spawn_local_workers(self, request: TpuCreateRequest, info: TpuInfo) -> None:
        """Spawn real local worker threads. Extracted from LocalPlatform._create_slice_with_workers."""
        # Uses self._controller_address, self._cache_path, self._port_allocator, self._threads
        # Follows the same logic as local.py:379-507
        ...
```

### K8sService Protocol

`Kubectl` at `lib/iris/src/iris/cluster/k8s/kubectl.py:61` already provides a clean interface. The protocol enables swapping the implementation.

**Proposed protocol** — `lib/iris/src/iris/cluster/k8s/k8s_service.py`:

```python
from __future__ import annotations
import subprocess
from typing import Protocol

from iris.cluster.k8s.kubectl import KubectlLogResult


class K8sService(Protocol):
    """Protocol for Kubernetes operations.

    Implementations: Kubectl (CLOUD), K8sServiceImpl (DRY_RUN/LOCAL).
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

### K8sServiceImpl — DRY_RUN/LOCAL modes

```python
# lib/iris/tests/cluster/k8s/k8s_service_impl.py

class K8sServiceImpl:
    """In-memory K8sService for DRY_RUN/LOCAL testing.

    Validates manifests, tracks state, supports failure injection.
    In LOCAL mode, could spawn local processes for pods (future work).
    """

    def __init__(
        self,
        namespace: str = "iris",
        mode: ServiceMode = ServiceMode.DRY_RUN,
        available_node_pools: list[str] | None = None,
    ):
        self._namespace = namespace
        self._mode = mode
        self._available_node_pools = set(available_node_pools or [])
        self._resources: dict[tuple[str, str], dict] = {}  # (kind, name) -> manifest
        self._injected_failures: dict[str, Exception] = {}

    @property
    def namespace(self) -> str:
        return self._namespace

    # -- Validation --

    def _validate_manifest(self, manifest: dict) -> None:
        """Validate pod spec structure, node selectors, resource requests."""
        kind = manifest.get("kind", "")
        if not kind:
            raise KubectlError("Manifest missing 'kind'")
        name = manifest.get("metadata", {}).get("name", "")
        if not name:
            raise KubectlError("Manifest missing 'metadata.name'")
        # Validate node selectors against available pools
        spec = manifest.get("spec", {})
        if kind.lower() == "pod":
            node_selector = spec.get("nodeSelector", {})
            pool = node_selector.get("cloud.google.com/gke-nodepool")
            if pool and self._available_node_pools and pool not in self._available_node_pools:
                raise KubectlError(f"Node pool {pool!r} not found")
        # Validate resource requests are parseable
        containers = spec.get("containers", [])
        for c in containers:
            resources = c.get("resources", {})
            for section in ("requests", "limits"):
                for key in resources.get(section, {}):
                    if key not in ("cpu", "memory", "nvidia.com/gpu", "google.com/tpu", "ephemeral-storage"):
                        raise KubectlError(f"Unknown resource type: {key!r}")

    # -- Failure injection --

    def inject_failure(self, operation: str, error: Exception) -> None:
        self._injected_failures[operation] = error

    def remove_node_pool(self, pool_name: str) -> None:
        """Simulate a node pool disappearing."""
        self._available_node_pools.discard(pool_name)

    def add_node_pool(self, pool_name: str) -> None:
        self._available_node_pools.add(pool_name)

    # -- Protocol methods --

    def apply_json(self, manifest: dict) -> None:
        if err := self._injected_failures.pop("apply_json", None):
            raise err
        self._validate_manifest(manifest)
        kind = manifest["kind"].lower()
        name = manifest["metadata"]["name"]
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

    def delete(self, resource: str, name: str, **kwargs) -> None:
        self._resources.pop((resource, name), None)
    ...
```

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
        self._gcp = gcp_service or GcpServiceImpl(
            mode=ServiceMode.CLOUD,
            project_id=gcp_config.project_id,
        )
        ...
```

Every `subprocess.run(["gcloud", ...])` call in `GcpPlatform` methods becomes a call to `self._gcp.tpu_create(...)`, `self._gcp.vm_list(...)`, etc.

### LocalPlatform Removal

After WS2+WS3, `LocalPlatform` is replaced by:

```python
# Before (e2e tests)
platform = LocalPlatform(
    label_prefix=label_prefix,
    threads=threads,
    controller_address=controller_address,
    cache_path=cache_path,
    ...
)

# After
gcp_service = GcpServiceImpl(
    mode=ServiceMode.LOCAL,
    controller_address=controller_address,
    cache_path=cache_path,
    port_allocator=port_allocator,
    threads=threads,
)
platform = GcpPlatform(
    gcp_config=local_gcp_config,
    label_prefix=label_prefix,
    gcp_service=gcp_service,
)
```

`local_cluster.py:50` (`create_local_autoscaler`) constructs a `GcpServiceImpl(mode=LOCAL)` instead of `LocalPlatform`. The `LocalCluster` class continues to work unchanged — it just delegates to GcpPlatform with a LOCAL-mode service.

Key migration points:
- `local.py:294` (`LocalPlatform`) — entire class removed
- `local.py:379` (`_create_slice_with_workers`) — logic moves into `GcpServiceImpl._spawn_local_workers`
- `local_cluster.py:98` (`LocalPlatform(...)` construction) — becomes `GcpPlatform(gcp_service=GcpServiceImpl(mode=LOCAL, ...))`
- `local.py:337` (`create_vm`) — in-memory VM creation moves to `GcpServiceImpl.vm_create` in LOCAL mode
- `local.py:509-576` (list_slices, list_vms, tunnel, shutdown, etc.) — handled by GcpPlatform delegating to the service

**What stays in GcpPlatform (not in GcpService):**
- `resolve_image` — string transformation, not a cloud API call
- `tunnel` — SSH tunneling, uses GceRemoteExec
- `discover_controller` — VM discovery + tunnel setup
- `stop_all` — orchestration over list + delete
- Bootstrap sequencing (`_bootstrap_workers`, `_wait_for_workers`) — orchestration logic on top of service calls

### Refactoring CoreweavePlatform and KubernetesProvider

Both already use `Kubectl`. The refactoring is mechanical: change the type annotation from `Kubectl` to `K8sService`.

- `lib/iris/src/iris/cluster/platform/coreweave.py:148` — `self._kubectl: K8sService`
- `lib/iris/src/iris/cluster/k8s/provider.py:597` — `kubectl: K8sService`

### Files to create/modify

| File | Action |
|------|--------|
| `lib/iris/src/iris/cluster/platform/service_mode.py` | **Create** — `ServiceMode` enum |
| `lib/iris/src/iris/cluster/platform/gcp_service.py` | **Create** — `GcpService` protocol, data classes |
| `lib/iris/src/iris/cluster/platform/gcp_service_impl.py` | **Create** — `GcpServiceImpl` (all three modes) |
| `lib/iris/src/iris/cluster/k8s/k8s_service.py` | **Create** — `K8sService` protocol |
| `lib/iris/src/iris/cluster/k8s/k8s_service_impl.py` | **Create** — `K8sServiceImpl` (DRY_RUN/LOCAL) |
| `lib/iris/src/iris/cluster/platform/gcp.py` | **Modify** — inject `GcpService`, replace subprocess calls |
| `lib/iris/src/iris/cluster/platform/coreweave.py` | **Modify** — type `self._kubectl` as `K8sService` |
| `lib/iris/src/iris/cluster/k8s/provider.py` | **Modify** — type `kubectl` field as `K8sService` |
| `lib/iris/src/iris/cluster/platform/factory.py` | **Modify** — pass `GcpService` through factory if needed |
| `lib/iris/src/iris/cluster/local_cluster.py` | **Modify** — use `GcpPlatform(gcp_service=GcpServiceImpl(mode=LOCAL))` |
| `lib/iris/src/iris/cluster/platform/local.py` | **Delete** — replaced by LOCAL mode in GcpServiceImpl |

---

## Workstream 3: Validation, Hooks, and Migration

### Validation in GcpServiceImpl

The validation layer mirrors what the real GCP API rejects, catching config bugs without making cloud calls. Shared across all modes via `_validate_*` methods.

**TPU validation** (from `gcp.py:200-231` `_validate_slice_config` and real gcloud behavior):
- Zone must be a known GCP zone (e.g. `us-central1-a`, `us-central2-b`)
- Accelerator type must be a known TPU type (e.g. `v4-8`, `v5litepod-16`, `v5litepod-256`)
- Name must be ≤63 chars, lowercase alphanumeric + hyphens
- Labels must be valid GCP label format (`^[a-z][a-z0-9_-]*$`)
- Runtime version must be non-empty
- Quota check: zone TPU count vs configured max

**VM validation** (from `gcp.py:234-242` `_validate_vm_config`):
- Zone must be known
- Name must be ≤63 chars, valid GCE instance name format
- Machine type should be a recognized format (e.g. `n1-standard-4`)
- Disk size must be > 0

**K8s validation** (from real kubectl behavior):
- Manifest must have `kind` and `metadata.name`
- Pod specs must have valid node selectors (pool must exist if `available_node_pools` configured)
- Resource requests must use known resource types
- Container images must be non-empty strings

### Hookable Testing Surface

The service impl exposes methods for injecting test behaviors:

```python
# GcpServiceImpl hooks
service.inject_failure("tpu_create", QuotaExhaustedError("zone full"))
service.set_zone_quota("us-central1-a", max_tpus=0)
service.set_tpu_type_unavailable("v4-8")
service.add_tpu_type("v6e-256")

# K8sServiceImpl hooks
k8s.inject_failure("apply_json", KubectlError("scheduling failed"))
k8s.remove_node_pool("gpu-pool")
k8s.add_node_pool("new-tpu-pool")
```

**Autoscaler test example — quota fallback:**
```python
def test_autoscaler_quota_fallback(fake_gcp_service):
    """Autoscaler tries zone A, hits quota, falls back to zone B."""
    fake_gcp_service.set_zone_quota("us-central1-a", max_tpus=0)
    # Zone B has quota
    platform = GcpPlatform(gcp_config=..., gcp_service=fake_gcp_service)
    # ... trigger autoscaler, verify it creates in zone B
```

### Relationship to existing fakes

| Existing Fake | Disposition | Rationale |
|---------------|-------------|-----------|
| `FakePlatform` (`fakes.py:245`) | **Keep** | Used by autoscaler tests. Tests autoscaler decisions, not GCP service calls. Different abstraction level. |
| `FakeGcloud` (`fakes.py:426`) | **Deprecate → Remove** | Replaced by `GcpServiceImpl(mode=DRY_RUN)`. After GcpPlatform uses `GcpService`, tests inject the DRY_RUN service instead of patching subprocess. |
| `mock_kubectl` (`kubernetes/conftest.py:20`) | **Replace** | MagicMock provides no validation. `K8sServiceImpl` validates manifests and maintains state. |
| `FakeProvider` (`controller/conftest.py:20`) | **Keep** | Tests controller transitions without provider. Different layer. |
| `LocalPlatform` (`local.py:294`) | **Remove** | Replaced by `GcpServiceImpl(mode=LOCAL)` injected into `GcpPlatform`. |

### Migration: FakeGcloud → GcpServiceImpl(DRY_RUN)

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
def gcp_service():
    return GcpServiceImpl(mode=ServiceMode.DRY_RUN)

def test_create_tpu_slice(gcp_service):
    platform = GcpPlatform(gcp_config=..., label_prefix="test", gcp_service=gcp_service)
    ...
```

No more `unittest.mock.patch`. No more subprocess interception.

### Migration: LocalPlatform → GcpPlatform + LOCAL service

```python
# Before (local_cluster.py:98)
platform = LocalPlatform(
    label_prefix=label_prefix,
    threads=threads,
    controller_address=controller_address,
    cache_path=cache_path,
    fake_bundle=fake_bundle,
    port_allocator=port_allocator,
    worker_attributes_by_group=worker_attributes_by_group,
    gpu_count_by_group=gpu_count_by_group,
    storage_prefix=storage_prefix,
)

# After
gcp_service = GcpServiceImpl(
    mode=ServiceMode.LOCAL,
    controller_address=controller_address,
    cache_path=cache_path,
    port_allocator=port_allocator,
    threads=threads,
    worker_attributes_by_group=worker_attributes_by_group,
    gpu_count_by_group=gpu_count_by_group,
)
local_gcp_config = config_pb2.GcpPlatformConfig(project_id="local")
platform = GcpPlatform(
    gcp_config=local_gcp_config,
    label_prefix=label_prefix,
    gcp_service=gcp_service,
)
```

**GcpPlatform changes for LOCAL compatibility:**
- `tunnel()` — must return `nullcontext(address)` when service mode is LOCAL (no SSH needed). This could be handled by checking `self._gcp._mode` or by having GcpPlatform's tunnel behavior be configurable via constructor.
- `discover_controller()` — must return configured address directly in LOCAL mode.
- `shutdown()` — must call through to service to stop worker threads.

These are methods that live in GcpPlatform, not GcpService. The cleanest approach: add a `mode` property to the `GcpService` protocol so GcpPlatform can branch on it for these orchestration methods. Alternatively, pass a separate `local_mode: bool` to GcpPlatform constructor.

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

### Task 2.1: Define ServiceMode enum and GcpService protocol + data classes
- **Files:** Create `lib/iris/src/iris/cluster/platform/service_mode.py`, `lib/iris/src/iris/cluster/platform/gcp_service.py`.
- **Tests:** Type-check with `uv run pyrefly`.
- **Scope:** 2 new files, ~180 lines.
- **Dependencies:** None.

### Task 2.2: Implement GcpServiceImpl (CLOUD mode)
- **Files:** Create `lib/iris/src/iris/cluster/platform/gcp_service_impl.py`. Extract all `subprocess.run(["gcloud", ...])` calls from `gcp.py` into CLOUD-mode methods on `GcpServiceImpl`.
- **Tests:** Existing `test_platform.py` tests must still pass with subprocess-patched approach (temporary — will migrate in task 3.2).
- **Scope:** 1 new file (~500 lines), modify `gcp.py` (~500 lines changed).
- **Dependencies:** Task 2.1.

### Task 2.3: Add DRY_RUN + LOCAL modes to GcpServiceImpl
- **Files:** Extend `gcp_service_impl.py` with in-memory state, validation, failure injection, and local worker spawning (extracted from `local.py:379-507`).
- **Tests:** Write unit tests for DRY_RUN validation (zone checks, accelerator types, quota). Write unit tests for LOCAL mode worker spawning.
- **Scope:** Modify `gcp_service_impl.py` (~300 lines added), 1 test file (~200 lines).
- **Dependencies:** Task 2.2.

### Task 2.4: Refactor GcpPlatform to use GcpService
- **Files:** Modify `lib/iris/src/iris/cluster/platform/gcp.py` to accept `GcpService` parameter and delegate all GCP operations to it. Modify `factory.py` if needed.
- **Tests:** All platform tests must pass.
- **Scope:** Modify `gcp.py` (~300 lines changed), `factory.py` (~10 lines).
- **Dependencies:** Task 2.2.

### Task 2.5: Define K8sService protocol
- **Files:** Create `lib/iris/src/iris/cluster/k8s/k8s_service.py` with `K8sService` protocol. Modify `coreweave.py` and `provider.py` to type `kubectl` as `K8sService`.
- **Tests:** Type-check + existing tests.
- **Scope:** 1 new file (~50 lines), modify 2 files (~20 lines each).
- **Dependencies:** None (parallel with Task 2.1).

### Task 3.1: Implement K8sServiceImpl (DRY_RUN/LOCAL)
- **Files:** Create `lib/iris/src/iris/cluster/k8s/k8s_service_impl.py` with manifest validation, state tracking, failure injection.
- **Tests:** Write unit tests for K8sServiceImpl — validate manifest apply, get, list, delete, node pool constraints.
- **Scope:** 1 new file (~300 lines), 1 test file (~150 lines).
- **Dependencies:** Task 2.5.

### Task 3.2: Migrate GcpPlatform tests to GcpServiceImpl(DRY_RUN)
- **Files:** Modify `lib/iris/tests/cluster/platform/test_platform.py` and `lib/iris/tests/cluster/controller/test_vm_lifecycle.py` to use `GcpServiceImpl(mode=DRY_RUN)` instead of `FakeGcloud` + subprocess patch.
- **Tests:** All existing platform and VM lifecycle tests must pass.
- **Scope:** 2 files (~200 lines changed).
- **Dependencies:** Tasks 2.3, 2.4.

### Task 3.3: Migrate K8s tests to K8sServiceImpl
- **Files:** Modify `lib/iris/tests/kubernetes/conftest.py` to provide `K8sServiceImpl` instead of `mock_kubectl`. Update `test_provider.py` and `test_coreweave_platform.py`.
- **Tests:** All kubernetes and coreweave tests must pass.
- **Scope:** 3-4 files (~300 lines changed).
- **Dependencies:** Tasks 2.5, 3.1.

### Task 3.4: Migrate LocalPlatform → GcpPlatform + LOCAL service
- **Files:** Modify `lib/iris/src/iris/cluster/local_cluster.py` to construct `GcpPlatform(gcp_service=GcpServiceImpl(mode=LOCAL))`. Update `lib/iris/src/iris/cluster/platform/__init__.py` exports.
- **Tests:** All E2E tests must pass — `uv run pytest lib/iris/tests/e2e/ -m "not cloud"`.
- **Scope:** 2-3 files, ~100 lines changed.
- **Dependencies:** Tasks 2.3, 2.4.

### Task 3.5: Remove LocalPlatform and FakeGcloud
- **Files:** Delete `lib/iris/src/iris/cluster/platform/local.py`. Remove `FakeGcloud` class and `fake_gcloud` fixture from `lib/iris/tests/cluster/platform/fakes.py`. Remove associated helpers (`_parse_flag`, `_parse_labels_string`, etc.) if no longer used. Update `__init__.py` exports.
- **Tests:** Full test suite pass.
- **Scope:** 2 files deleted/heavily modified (~800 lines removed).
- **Dependencies:** Tasks 3.2, 3.4 (all consumers migrated).

---

## Dependency Graph

```
Task 1.1 ─┐
Task 1.2 ─┤
Task 1.3 ─┼─── (all independent, can run in parallel)
Task 1.4 ─┤
Task 1.5 ─┘

Task 2.1 ──► Task 2.2 ──► Task 2.3 ──► Task 2.4 ──┐
Task 2.5 ───────────────────────────────────────────┤
                                                     │
Task 2.3 ──────────────────────► Task 3.2 ──────────┤
Task 2.3 + 2.4 ──────────────► Task 3.4 ───────────┼──► Task 3.5
Task 2.5 ──► Task 3.1 ──► Task 3.3 ────────────────┘

Workstream 1: [1.1, 1.2, 1.3, 1.4, 1.5] — fully parallel, no deps on WS2/WS3
Workstream 2: 2.1 → 2.2 → 2.3 → 2.4 (GCP chain) | 2.5 (K8s, parallel with GCP)
Workstream 3: depends on WS2 protocols, then:
  - 3.2 (migrate GCP tests) needs 2.3 + 2.4
  - 3.4 (migrate LocalPlatform) needs 2.3 + 2.4
  - 3.1 → 3.3 (K8s impl + migration) needs 2.5
  - 3.5 (cleanup) needs 3.2 + 3.4 (all consumers migrated)
```

## Risks and Open Questions

1. **GcpPlatform orchestration methods need LOCAL awareness.** `tunnel()`, `discover_controller()`, `shutdown()` in GcpPlatform need to behave differently in LOCAL mode (no SSH, direct address, stop worker threads). Options: (a) add `mode` property to `GcpService` protocol, (b) pass `local_mode: bool` to GcpPlatform, (c) make `tunnel()` / `discover_controller()` configurable via strategy injection. Option (a) is simplest and keeps mode in one place.

2. **GcpStandaloneWorkerHandle subprocess calls.** `GcpStandaloneWorkerHandle` at `gcp.py:264` makes direct subprocess calls for `status()`, `terminate()`, `set_labels()`, `set_metadata()`. These need to go through `GcpService` too, but the handle holds a reference to the service. This means `GcpService` must be injected into handle construction, which happens inside `create_vm()` and `create_slice()`. Straightforward but adds complexity to handle constructors.

3. **Validation fidelity.** The DRY_RUN validation should catch "99% of silly bugs" (invalid zones, bad TPU types, malformed labels) but doesn't need to replicate GCP's exact error messages. We maintain a `KNOWN_GCP_ZONES` and `KNOWN_TPU_TYPES` set that gets updated periodically. If a new TPU type is added to GCP, tests using DRY_RUN need the set updated — this is a feature (forces explicit acknowledgment of new types).

4. **LOCAL mode worker lifecycle ownership.** Currently `LocalPlatform._threads` owns all worker threads and `shutdown()` stops them. In the new design, `GcpServiceImpl` in LOCAL mode owns the `ThreadContainer`. `GcpPlatform.shutdown()` must delegate to the service's shutdown. The `GcpService` protocol needs a `shutdown()` method (no-op in CLOUD/DRY_RUN modes).

5. **FakePlatform coexistence.** `FakePlatform` (`fakes.py:245`) is used by autoscaler tests that operate at the Platform level and don't care about GCP specifics. It stays permanently. Tests that need GCP-specific behavior (zone routing, quota handling) use `GcpPlatform` with a DRY_RUN/LOCAL service.

6. **RemoteExec stays separate from GcpService.** SSH operations (`run_command`, `bootstrap`) use `RemoteExec` protocol, already well-abstracted. In LOCAL mode, commands run via `subprocess.run(["bash", "-c", ...])` as `_LocalWorkerHandle` does today (`local.py:98`). This logic moves into the LOCAL-mode handle that GcpServiceImpl creates.

7. **Kubectl already is the K8sService.** `Kubectl` class already provides exactly the right interface — the protocol just enables swapping in `K8sServiceImpl` for tests. If `Kubectl` API changes, the protocol must track it. Low risk since the API is stable.
