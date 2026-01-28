# Smoke Test Dry-Run Mode (v2)

This is a revised design incorporating feedback from the [codex review](smoke-test-dry-run-review.md).
The v1 design underestimated proto/config wiring, omitted worker bootstrap dependencies
(`_LocalBundleProvider`, `_LocalImageProvider`, `_LocalContainerRuntime`, `LocalEnvironmentProvider`),
and proposed a `LocalController` that didn't implement the full `ControllerProtocol`.

This revision:
- **Replaces `demo_cluster.py`** with shared infrastructure in `local_platform.py`
- Shows **complete control-flow diagrams** for GCP vs local mode
- Identifies every GCP call site and the corresponding local shim
- Resolves the open questions from v1

## Motivation

Same as v1 — local smoke test without GCP. See the v1 doc for context.

---

## Current Control Flow: GCP Mode

### Full call graph for `smoke-test.py --config eu-west4.yaml`

```
smoke-test.py main()
│
├── Phase 0a: _build_images()
│   └── subprocess: uv run iris build {controller,worker}-image --push
│       (GCP: pushes to Artifact Registry)
│
├── Phase 0b: _cleanup_existing()
│   └── cleanup_iris_resources(zone, project)
│       └── gcloud compute instances delete ...
│       └── gcloud compute tpus tpu-vm delete ...
│
├── Phase 1: _start_cluster(cluster_config)
│   └── create_controller(cluster_config)        # controller.py:931
│       ├── config.controller_vm.WhichOneof("controller")
│       │   ├── "gcp"    → GcpController(config)
│       │   └── "manual" → ManualController(config)
│       └── GcpController.start()
│           ├── gcloud compute instances create ...   ← GCP CALL
│           ├── _wait_for_health(address)
│           └── returns "http://<internal-ip>:10000"
│
├── Phase 2: SSH Tunnel
│   └── controller_tunnel(zone, project)          # debug.py
│       └── gcloud compute ssh ... -L 10000:localhost:10000  ← GCP CALL
│
├── Phase 3: _run_tests(tunnel_url)
│   └── IrisClient.remote(tunnel_url)
│       └── client.submit(entrypoint, resources)
│           └── gRPC → Controller.LaunchJob()
│
│   [Inside Controller VM - separate process on GCE]
│   Controller._run_scheduling_loop()
│   ├── _run_scheduling()
│   │   └── Scheduler.find_assignments() → dispatch RPCs to workers
│   └── _run_autoscaler_once()
│       ├── compute_demand_entries(state)
│       ├── _build_vm_status_map()
│       └── Autoscaler.run_once(demand, vm_status_map)
│           ├── group.cleanup_failed_slices()
│           ├── evaluate() → route_demand() → ScalingDecision(SCALE_UP)
│           ├── execute() → group.scale_up()
│           │   └── VmManager.create_vm_group()     ← KEY ABSTRACTION
│           │       └── TpuVmManager._gcloud_create_tpu()  ← GCP CALL
│           │           └── gcloud compute tpus tpu-vm create ...
│           │       └── _make_vm_group()
│           │           └── ManagedVm lifecycle thread
│           │               └── GcloudSshConnection.bootstrap()  ← GCP CALL
│           └── group.scale_down_if_idle()
│               └── VmGroupProtocol.terminate()
│                   └── TpuVmGroup.terminate()
│                       └── gcloud compute tpus tpu-vm delete ...  ← GCP CALL
│
└── Phase 4 + Cleanup
    └── controller.stop()
        └── gcloud compute instances delete ...   ← GCP CALL
```

### Summary of GCP call sites

| Call Site | File | What it does |
|-----------|------|--------------|
| `GcpController.start()` | `controller.py` | Creates controller GCE VM |
| `GcpController.stop()` | `controller.py` | Deletes controller GCE VM |
| `controller_tunnel()` | `debug.py` | SSH port forwarding |
| `TpuVmManager._gcloud_create_tpu()` | `gcp_tpu_platform.py` | Creates TPU VMs |
| `TpuVmGroup.terminate()` | `gcp_tpu_platform.py` | Deletes TPU VMs |
| `GcloudSshConnection.bootstrap()` | `ssh.py` | Runs bootstrap script on VMs |
| `TpuVmManager._gcloud_list_tpus()` | `gcp_tpu_platform.py` | Discovery/reconcile |
| `cleanup_iris_resources()` | `debug.py` | Cleanup on start/failure |
| `discover_controller_vm()` | `debug.py` | Find controller for tunnel/logs |
| `collect_docker_logs()` | `debug.py` | SSH to collect logs |
| `list_iris_tpus()` | `debug.py` | Find TPUs for log collection |

### Key insight: what needs to be replaced

The GCP calls fall into two categories:

1. **Controller lifecycle** (controller VM create/delete/ssh) — replaced by running `Controller` in-process
2. **Worker lifecycle** (TPU create/delete/bootstrap/ssh) — replaced by `LocalVmManager` creating in-process `Worker` instances

Everything between these boundaries is unchanged: the `Controller`, `Autoscaler`, `ScalingGroup`, `Scheduler`, and job execution code all run identically.

---

## Proposed Control Flow: Local Mode

```
smoke-test.py main(--local)
│
├── Phase 0a: SKIP (no images to build)
├── Phase 0b: SKIP (no GCP resources to clean)
│
├── Phase 1: _start_cluster(cluster_config)
│   └── create_controller(cluster_config)        # controller.py:931
│       └── config.controller_vm.WhichOneof("controller")
│           └── "local" → LocalController(config)
│       └── LocalController.start()
│           ├── Create temp dir for bundles/cache/fake_bundle
│           ├── create_local_autoscaler(config, controller_address, cache, bundle)
│           │   └── For each scale_group in config:
│           │       └── LocalVmManager(scale_group, controller_address, cache, bundle, ...)
│           │       └── ScalingGroup(config, vm_manager=local_vm_manager)
│           │   └── Autoscaler(scale_groups, vm_registry)
│           ├── Controller(config, RpcWorkerStubFactory(), autoscaler)
│           ├── controller.start()
│           └── returns "http://127.0.0.1:{port}"
│
├── Phase 2: SKIP (no SSH tunnel — controller is localhost)
│
├── Phase 3: _run_tests(controller_url)   ← IDENTICAL TO GCP MODE
│   └── IrisClient.remote(controller_url)
│       └── client.submit(entrypoint, resources)
│           └── gRPC → Controller.LaunchJob()
│
│   [Inside Controller — same process, same code path]
│   Controller._run_scheduling_loop()
│   ├── _run_scheduling()                 ← IDENTICAL
│   └── _run_autoscaler_once()            ← IDENTICAL
│       └── Autoscaler.run_once()         ← IDENTICAL
│           └── group.scale_up()          ← IDENTICAL (calls VmManager interface)
│               └── LocalVmManager.create_vm_group()  ← LOCAL SHIM
│                   ├── Determine worker_count from topology
│                   ├── For each worker:
│                   │   ├── _LocalBundleProvider(fake_bundle)
│                   │   ├── _LocalImageProvider()
│                   │   ├── _LocalContainerRuntime()
│                   │   ├── LocalEnvironmentProvider(cpu, mem, device, attrs)
│                   │   └── Worker(config, bundle_provider, image_provider,
│                   │             container_runtime, environment_provider,
│                   │             port_allocator)
│                   │       └── worker.start()  (gRPC server on localhost)
│                   └── LocalVmGroup(workers, vm_registry)
│                       └── _StubManagedVm(vm_info) for each worker
│
└── Phase 4 + Cleanup
    └── controller.stop()
        └── LocalController.stop()
            ├── Controller.stop()  (stops scheduling loop, uvicorn server)
            └── temp_dir.cleanup()
            (Autoscaler terminating workers happens via ScalingGroup cleanup)
```

### What changes, what stays the same

| Component | GCP Mode | Local Mode | Changed? |
|-----------|----------|------------|----------|
| `Controller` | Runs on GCE VM | Runs in-process | **No** — same class |
| `ControllerConfig` | bundle_prefix=gs://... | bundle_prefix=file:///tmp/... | Config only |
| `Autoscaler` | Same | Same | **No** |
| `ScalingGroup` | Same | Same | **No** |
| `Scheduler` | Same | Same | **No** |
| `VmManager` | `TpuVmManager` | `LocalVmManager` | **Replaced** |
| `VmGroup` | `TpuVmGroup` | `LocalVmGroup` | **Replaced** |
| `ManagedVm` | Real lifecycle threads | `_StubManagedVm` | **Replaced** |
| `Worker` | Docker + bootstrap | In-process threads | **Same class**, different providers |
| `BundleProvider` | GCS download | Returns fake path | **Replaced** |
| `ImageProvider` | Docker build | Returns no-op result | **Replaced** |
| `ContainerRuntime` | Docker containers | Thread-based execution | **Replaced** |
| `EnvironmentProvider` | Probes real hardware | Returns fake metadata | **Replaced** |
| `PortAllocator` | Per-worker | Shared (same process) | Config only |
| SSH tunnel | gcloud ssh | Not needed | **Skipped** |
| Image build | Docker + push | Not needed | **Skipped** |

---

## Detailed Design

### 1. Proto Changes: `config.proto`

Add `local` variant to both oneofs:

```protobuf
// ProviderConfig — per-scale-group
message ProviderConfig {
  oneof provider {
    TpuProvider tpu = 1;
    ManualProvider manual = 2;
    LocalProvider local = 3;      // NEW
  }
}

message LocalProvider {
  // No fields needed — local provider uses in-process workers.
  // Scale group config (accelerator_type, accelerator_variant, etc.)
  // drives worker attribute simulation.
}

// ControllerVmConfig — controller lifecycle
message ControllerVmConfig {
  string image = 10;
  string bundle_prefix = 11;
  oneof controller {
    GcpControllerConfig gcp = 1;
    ManualControllerConfig manual = 2;
    LocalControllerConfig local = 3;  // NEW
  }
}

message LocalControllerConfig {
  int32 port = 1;  // Port to bind (default 10000)
}
```

After editing, run `scripts/generate-protos.py`.

### 2. New Module: `src/iris/cluster/vm/local_platform.py`

This extracts and generalizes code from `demo_cluster.py` and `local_client.py`.

The worker bootstrap dependencies are the key thing the v1 design missed
(flagged by the codex review). Each in-process worker needs four provider
implementations already defined in `local_client.py`:

- `_LocalBundleProvider` — returns a fake bundle path
- `_LocalImageProvider` — returns a no-op `BuildResult`
- `_LocalContainerRuntime` — executes entrypoints in threads
- `LocalEnvironmentProvider` — returns fake CPU/memory/device metadata

```python
"""Local platform: in-process VmManager for testing without GCP.

Provides LocalVmManager (VmManagerProtocol) and LocalVmGroup (VmGroupProtocol)
that create real Worker instances running in-process with thread-based execution
instead of Docker containers.

Worker dependencies (bundle, image, container, environment providers) come from
iris.cluster.client.local_client — the same providers used by LocalClusterClient.
"""

from iris.cluster.client.local_client import (
    LocalEnvironmentProvider,
    _LocalBundleProvider,
    _LocalContainerRuntime,
    _LocalImageProvider,
)
from iris.cluster.vm.managed_vm import ManagedVm, VmRegistry
from iris.cluster.vm.vm_platform import VmGroupProtocol, VmGroupStatus, VmSnapshot
from iris.cluster.worker.worker import PortAllocator, Worker, WorkerConfig


class _StubManagedVm(ManagedVm):
    """Minimal ManagedVm that holds VmInfo without lifecycle threads.

    In-process workers don't need SSH bootstrap or health polling —
    they are immediately READY. This stub satisfies the VmGroupProtocol
    interface for the autoscaler's VmRegistry tracking.
    """

    def __init__(self, info: vm_pb2.VmInfo):
        self.info = info
        self._log_lines: list[str] = []

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def init_log(self, tail: int | None = None) -> str:
        return ""

    def check_health(self) -> bool:
        return True


class LocalVmGroup(VmGroupProtocol):
    """VM group backed by in-process Worker instances.

    Each "VM" is a Worker running on localhost with a unique port.
    Workers become ready immediately (no bootstrap delay).
    The group tracks workers and their stub ManagedVm instances
    for autoscaler compatibility.
    """

    def __init__(
        self,
        group_id: str,
        scale_group: str,
        workers: list[Worker],
        worker_ids: list[str],
        vm_registry: VmRegistry,
    ):
        ...
        # Create _StubManagedVm for each worker
        # Register each with vm_registry
        # Workers are already started

    # Properties: group_id, slice_id, scale_group, created_at_ms
    # Methods: status(), vms(), terminate(), to_proto()
    # terminate() calls worker.stop() + vm_registry.unregister()


class LocalVmManager:
    """VmManager for in-process workers. Implements VmManagerProtocol.

    Creates LocalVmGroup instances containing real Worker instances.
    Workers use local providers (from local_client.py) for bundle/image/
    container/environment — no Docker, no GCS, no SSH.

    Key difference from demo_cluster: accepts controller_address and
    temp paths as constructor args instead of computing them internally,
    so it can be wired into the standard config → autoscaler → controller
    pipeline via create_autoscaler_from_config().
    """

    def __init__(
        self,
        scale_group_config: config_pb2.ScaleGroupConfig,
        controller_address: str,
        cache_path: Path,
        fake_bundle: Path,          # ← v1 missed this (flagged by review)
        vm_registry: VmRegistry,
        port_allocator: PortAllocator,  # ← shared across groups (flagged by review)
    ):
        ...

    def create_vm_group(self, tags: dict[str, str] | None = None) -> VmGroupProtocol:
        """Create a new group of in-process workers.

        1. Determine worker_count from accelerator_variant topology
           (get_tpu_topology(variant).vm_count, or 1 for CPU)
        2. For each worker:
           a. Allocate port via find_free_port()
           b. Create providers:
              - _LocalBundleProvider(self._fake_bundle)
              - _LocalImageProvider()
              - _LocalContainerRuntime()
              - LocalEnvironmentProvider(cpu=1000, memory_gb=1000,
                  attributes={"tpu-name": slice_id, "tpu-worker-id": i, ...},
                  device=tpu_device(variant) if TPU else None)
           c. Create Worker(config, providers..., port_allocator=self._port_allocator)
           d. worker.start()
        3. Return LocalVmGroup(workers, vm_registry)
        """
        ...

    def discover_vm_groups(self) -> list[VmGroupProtocol]:
        return []  # No persistence — local workers are ephemeral
```

### 3. `LocalController` in `controller.py`

The v1 design only had `start()/stop()/status()`. The review flagged that
`ControllerProtocol` requires `restart()`, `reload()`, `discover()`, and
`fetch_startup_logs()`.

```python
class LocalController:
    """In-process controller for local testing and demos.

    Unlike GcpController (creates GCE VM) or ManualController (SSHs to host),
    LocalController runs the Controller class directly in the current process.

    Worker creation happens via LocalVmManager through the standard autoscaler
    pipeline — the Controller doesn't know it's running locally.

    Used by:
    - smoke-test.py --local
    - demo_cluster.py (replaces DemoCluster)
    """

    def __init__(self, config: config_pb2.IrisClusterConfig):
        self._config = config
        self._controller: Controller | None = None
        self._temp_dir: tempfile.TemporaryDirectory | None = None
        self._port: int = 0

    def start(self) -> str:
        """Start in-process controller with local autoscaler.

        Control flow:
        1. Create temp directory for bundles, cache, fake_bundle
        2. Create Autoscaler via _create_local_autoscaler()
           └── For each scale_group:
               └── LocalVmManager (reads accelerator config from proto)
               └── ScalingGroup(config, vm_manager)
           └── Autoscaler(scale_groups, vm_registry)
        3. Create Controller(config, RpcWorkerStubFactory(), autoscaler)
        4. controller.start()
        5. Return controller.url
        """
        self._temp_dir = tempfile.TemporaryDirectory(prefix="iris_local_")
        temp_path = Path(self._temp_dir.name)

        bundle_dir = temp_path / "bundles"
        bundle_dir.mkdir()
        cache_path = temp_path / "cache"
        cache_path.mkdir()
        fake_bundle = temp_path / "fake_bundle"
        fake_bundle.mkdir()
        (fake_bundle / "pyproject.toml").write_text("[project]\nname = 'local'\n")

        # Read port from config (local controller config)
        local_config = self._config.controller_vm.local
        self._port = local_config.port or find_free_port()

        controller_address = f"http://127.0.0.1:{self._port}"

        # Create autoscaler using the standard pipeline but with LocalVmManagers
        autoscaler = _create_local_autoscaler(
            config=self._config,
            controller_address=controller_address,
            cache_path=cache_path,
            fake_bundle=fake_bundle,
        )

        # Wire bundle_prefix to local filesystem
        bundle_prefix = self._config.controller_vm.bundle_prefix or f"file://{bundle_dir}"

        controller_config = ControllerConfig(
            host="127.0.0.1",
            port=self._port,
            bundle_prefix=bundle_prefix,
        )
        self._controller = Controller(
            config=controller_config,
            worker_stub_factory=RpcWorkerStubFactory(),
            autoscaler=autoscaler,
        )
        self._controller.start()
        return self._controller.url

    def stop(self) -> None:
        if self._controller:
            self._controller.stop()
        if self._temp_dir:
            self._temp_dir.cleanup()

    def restart(self) -> str:
        self.stop()
        return self.start()

    def reload(self) -> str:
        # For local mode, reload == restart (no SSH, no Docker pull)
        return self.restart()

    def discover(self) -> str | None:
        if self._controller:
            return self._controller.url
        return None

    def fetch_startup_logs(self) -> str:
        return "(local controller — no startup logs)"

    def status(self) -> ControllerStatus:
        if self._controller:
            return ControllerStatus(
                running=True,
                address=self._controller.url,
                healthy=True,  # In-process controller is always healthy if running
            )
        return ControllerStatus(running=False, address="", healthy=False)
```

### 4. Autoscaler Factory for Local Mode

New function in `config.py` (or `local_platform.py`):

```python
def _create_local_autoscaler(
    config: config_pb2.IrisClusterConfig,
    controller_address: str,
    cache_path: Path,
    fake_bundle: Path,
) -> Autoscaler:
    """Create Autoscaler with LocalVmManagers for all scale groups.

    This parallels create_autoscaler_from_config() but creates LocalVmManagers
    instead of TpuVmManager/ManualVmManager.

    Control flow:
    1. Create shared VmRegistry + PortAllocator
    2. For each scale_group in config:
       a. Create LocalVmManager with the group's config
       b. Wrap in ScalingGroup with fast cooldowns
    3. Return Autoscaler(scale_groups, vm_registry)
    """
    vm_registry = VmRegistry()
    shared_port_allocator = PortAllocator(port_range=(30000, 40000))

    scale_groups: dict[str, ScalingGroup] = {}
    for name, sg_config in config.scale_groups.items():
        manager = LocalVmManager(
            scale_group_config=sg_config,
            controller_address=controller_address,
            cache_path=cache_path,
            fake_bundle=fake_bundle,
            vm_registry=vm_registry,
            port_allocator=shared_port_allocator,
        )
        scale_groups[name] = ScalingGroup(
            config=sg_config,
            vm_manager=manager,
            scale_up_cooldown_ms=1000,       # Fast for local
            scale_down_cooldown_ms=300_000,   # Keep workers alive
        )

    return Autoscaler(
        scale_groups=scale_groups,
        vm_registry=vm_registry,
        config=AutoscalerConfig(evaluation_interval_seconds=2.0),
    )
```

### 5. Config Wiring: `config.py`

Add local provider support to the existing factory functions:

```python
# In _get_provider_info():
def _get_provider_info(group_config):
    ...
    which = provider.WhichOneof("provider")
    if which == "tpu":
        ...
    if which == "manual":
        ...
    if which == "local":                           # NEW
        return ("local", None, None)
    raise ValueError(...)

# In _create_manager_from_config():
def _create_manager_from_config(group_name, cluster_config, vm_factory, *, dry_run=False):
    ...
    if provider_type == "local":                    # NEW
        # LocalVmManager doesn't use vm_factory (no SSH/lifecycle threads)
        # It creates its own workers directly.
        # The controller_address and paths are provided by LocalController,
        # not by this factory. So we defer LocalVmManager creation to
        # _create_local_autoscaler() which has the necessary context.
        raise ValueError(
            "Local provider is handled by LocalController, not by "
            "create_autoscaler_from_config(). Use LocalController instead."
        )
    ...

# In create_controller():
def create_controller(config):
    which = config.controller_vm.WhichOneof("controller")
    if which == "gcp":
        return GcpController(config)
    if which == "manual":
        return ManualController(config)
    if which == "local":                            # NEW
        return LocalController(config)
    raise ValueError(...)
```

**Design note:** The local provider in `_create_manager_from_config` raises
because `LocalVmManager` needs runtime context (controller address, temp paths)
that aren't available at config-load time. The `LocalController` constructs the
entire autoscaler itself via `_create_local_autoscaler()`. This is analogous to
how `DemoCluster._create_autoscaler()` works today.

### 6. Smoke Test Changes

The smoke test already uses `create_controller(config)` which dispatches on the
config's controller type. With `local` support wired in, the smoke test needs
**minimal changes** — just skipping GCP-only phases:

```python
# In SmokeTestRunner.run():
def run(self) -> bool:
    cluster_config = load_config(self.config.config_path)

    is_local = cluster_config.controller_vm.WhichOneof("controller") == "local"

    # Phase 0a: Build images (skip for local)
    if self.config.build_images and not is_local:
        ...

    # Phase 0b: Clean start (skip for local)
    if self.config.clean_start and not is_local:
        ...

    # Phase 1: Start cluster (works for both — create_controller dispatches)
    self._start_cluster(cluster_config)

    if is_local:
        # Direct connection, no tunnel needed
        controller_url = self._controller.discover()
        # Skip log streaming (no SSH/Docker)
        self._run_tests(controller_url)
        success = self._print_results()
    else:
        # GCP: SSH tunnel + log streaming (existing code)
        with controller_tunnel(...) as tunnel_url:
            self._run_tests(tunnel_url)
            success = self._print_results()

    return success
```

The test job definitions (`_hello_tpu_job`, `_distributed_work_job`, etc.) and
`_run_tests()` method work **unchanged** — they use `IrisClient.remote()` which
connects via gRPC regardless of whether the controller is local or remote.

### 7. Delete `demo_cluster.py` / Replace with Local Config

The current `demo_cluster.py` (845 lines) duplicates most of this infrastructure.
After extracting to `local_platform.py`, `demo_cluster.py` becomes:

```python
"""Demo cluster launcher.

Usage:
    uv run python examples/demo_cluster.py
    uv run python examples/demo_cluster.py --config examples/local.yaml
"""

@click.command()
@click.option("--config", default="examples/local.yaml")
@click.option("--no-browser", is_flag=True)
def main(config: str, no_browser: bool):
    cluster_config = load_config(config)
    controller = create_controller(cluster_config)
    controller_url = controller.start()
    print(f"Controller: {controller_url}")

    # Seed cluster
    client = IrisClient.remote(controller_url, workspace=IRIS_ROOT)
    # ... submit demo jobs (reuse from existing seed_cluster) ...

    # Launch Jupyter
    # ... existing launch_jupyter() code ...

    controller.stop()
```

This shrinks demo_cluster.py from ~845 lines to ~100 lines by delegating all
infrastructure to the shared `local_platform.py` + `LocalController`.

### Example Local Config: `examples/local.yaml`

```yaml
# Local development cluster — no GCP, no Docker
# Usage: uv run python scripts/smoke-test.py --config examples/local.yaml

controller_vm:
  bundle_prefix: "file:///tmp/iris-local/bundles"
  local:
    port: 0  # 0 = auto-assign

scale_groups:
  tpu_v5e_16:
    provider:
      local: {}
    accelerator_type: tpu
    accelerator_variant: v5litepod-16
    min_slices: 0
    max_slices: 4

  cpu:
    provider:
      local: {}
    accelerator_type: cpu
    min_slices: 0
    max_slices: 4
```

---

## Implementation Plan (Spiral)

Following the AGENTS.md guidance for spiral plans where each step is independently testable.

### Step 1: Proto + LocalVmManager extraction + test

**Files changed:**
- `src/iris/rpc/config.proto` — add `LocalProvider`, `LocalControllerConfig`
- `scripts/generate-protos.py` — run to regenerate
- `src/iris/cluster/vm/local_platform.py` — **new**, extracted from `demo_cluster.py`
- `tests/cluster/vm/test_local_platform.py` — **new**

**Test:** Create a `LocalVmManager`, call `create_vm_group()`, verify workers
start and are reachable via gRPC health check.

### Step 2: LocalController + config wiring

**Files changed:**
- `src/iris/cluster/vm/controller.py` — add `LocalController`
- `src/iris/cluster/vm/config.py` — add local provider handling
- `tests/cluster/vm/test_local_controller.py` — **new**

**Test:** Load a local.yaml config, call `create_controller()`, verify it returns
a `LocalController`. Start it, submit a job via `IrisClient.remote()`, verify it
completes.

### Step 3: Smoke test integration

**Files changed:**
- `scripts/smoke-test.py` — detect `local` mode, skip GCP phases
- `examples/local.yaml` — **new** config file

**Test:** Run `uv run python scripts/smoke-test.py --config examples/local.yaml`.
All three test jobs (simple, concurrent, coscheduled) should pass locally.

### Step 4: Replace demo_cluster.py

**Files changed:**
- `examples/demo_cluster.py` — slim down to use `LocalController`
- Delete duplicated code (~700 lines removed)

**Test:** Run `uv run python examples/demo_cluster.py` and verify demo jobs complete.

### Step 5: Documentation

**Files changed:**
- `README.md` — add local mode section
- `AGENTS.md` — add local platform to key modules

---

## Resolved Questions (from v1 + review)

### Q1: Should local mode use the same config file format?
**A:** Yes. Same YAML schema, with `controller_vm.local` and `provider.local`.
The config file fully determines the mode — no `--local` flag needed.

### Q2: How should local mode handle bundle storage?
**A:** `file://` prefix with temp directory. `LocalController.start()` creates
the temp directory and sets `bundle_prefix=file://{temp}/bundles`. This exercises
the same `fsspec`-based bundle storage path as GCP mode.

### Q3: Should local workers simulate TPU attributes?
**A:** Yes. `LocalVmManager` reads `accelerator_type` and `accelerator_variant`
from the scale group config and passes them to `LocalEnvironmentProvider` as
attributes (`tpu-name`, `tpu-worker-id`, `tpu-topology`) and device config
(`tpu_device(variant)`). This is what `demo_cluster.py` already does.

### Q4 (review): What about worker bootstrap dependencies?
**A:** `LocalVmManager` imports and uses the four providers from
`iris.cluster.client.local_client`: `_LocalBundleProvider`, `_LocalImageProvider`,
`_LocalContainerRuntime`, `LocalEnvironmentProvider`. These are already tested
by `LocalClusterClient` and `demo_cluster.py`.

### Q5 (review): ControllerProtocol completeness?
**A:** `LocalController` implements all protocol methods. `restart()` = stop+start,
`reload()` = restart (no Docker to pull), `discover()` = return url if running,
`fetch_startup_logs()` = no-op string.

### Q6 (review): PortAllocator shared namespace?
**A:** `_create_local_autoscaler()` creates a single shared `PortAllocator(port_range=(30000, 40000))`
and passes it to all `LocalVmManager` instances, preventing port collisions.

---

## References

- [Codex Review](smoke-test-dry-run-review.md) — review of v1 design
- `examples/demo_cluster.py` — existing local cluster (to be replaced)
- `src/iris/cluster/client/local_client.py` — local worker providers
- `src/iris/cluster/vm/vm_platform.py` — `VmManagerProtocol`, `VmGroupProtocol`
- `src/iris/cluster/vm/controller.py` — `ControllerProtocol`, `create_controller()`
- `src/iris/cluster/vm/config.py` — `create_autoscaler_from_config()`, `_get_provider_info()`
- `src/iris/cluster/vm/gcp_tpu_platform.py` — `TpuVmManager` (the thing we're replacing)
- `src/iris/cluster/controller/controller.py` — `Controller`, `_run_autoscaler_once()`
