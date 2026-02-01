# Smoke Test Dry-Run Mode (v3) — ClusterManager

This revision introduces `ClusterManager` as the central abstraction for cluster
lifecycle, replacing the v2 `LocalController` approach. The key insight: callers
(smoke-test, demo, CLI) don't care _how_ a controller runs — they need a URL and
a clean shutdown. `ClusterManager` provides that uniformly.

## Design Principles

1. **No new config files** — `--local` flag overrides any existing config at runtime
2. **Maximize reuse** — extract `LocalVmManager`/`LocalVmGroup` from `demo_cluster.py`, don't duplicate
3. **Explicit lifecycle** — `start()` returns references; no work in `__init__`
4. **Callers stay readable** — smoke-test and demo use the same `ClusterManager` API

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        Callers                              │
│  smoke-test.py  │  demo_cluster.py  │  cli.py (iris cluster)│
└────────┬────────┴─────────┬─────────┴──────────┬────────────┘
         │                  │                    │
         ▼                  ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│                     ClusterManager                          │
│  vm/cluster_manager.py                                      │
│                                                             │
│  start() → str              # start controller, return URL  │
│  stop()                     # stop controller + cleanup     │
│  connect() → ctx[str]       # start + tunnel/direct → URL   │
│  is_local: bool             # mode query                    │
│                                                             │
│  Owns:                                                      │
│  ┌──────────────────┐  ┌─────────────────────┐              │
│  │ControllerProtocol│  │ SSH Tunnel (GCP only)│              │
│  │ GcpController    │  │ controller_tunnel()  │              │
│  │ ManualController  │  │ (no-op for local)   │              │
│  │ LocalController   │  └─────────────────────┘              │
│  └──────┬───────────┘                                       │
│         │ delegates to                                       │
│  ┌──────▼───────────────────────────────────────────┐       │
│  │ Controller (the real gRPC server)                 │       │
│  │  + Autoscaler                                     │       │
│  │    + ScalingGroup[]                               │       │
│  │      + VmManager (TpuVmManager | LocalVmManager)  │       │
│  └───────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

### What `ClusterManager` replaces in each caller

| Caller | Before | After |
|--------|--------|-------|
| `smoke-test.py` | `create_controller()` + `controller_tunnel()` + manual phase skipping | `ClusterManager(config).connect()` |
| `demo_cluster.py` | `DemoCluster.__enter__` builds Controller/Autoscaler/Workers manually | `ClusterManager(config).start()` |
| `cli.py` (`cluster start`) | `create_controller(config)` + `ctrl.start()` | `ClusterManager(config).start()` |

---

## Control Flow: GCP vs Local

### GCP Mode

```
ClusterManager(config)
│
├── start()
│   ├── create_controller(config) → GcpController
│   └── GcpController.start()
│       ├── gcloud compute instances create ...
│       ├── _wait_for_health()
│       └── returns "http://<internal-ip>:10000"
│
├── connect()  [context manager]
│   ├── start()
│   ├── controller_tunnel(zone, project) → SSH port forward
│   ├── yield "http://localhost:10000"
│   └── stop()
│
└── stop()
    └── GcpController.stop()
        └── gcloud compute instances delete ...
```

### Local Mode

```
ClusterManager(config)   # config has local controller + local providers
│
├── start()
│   ├── create_controller(config) → LocalController
│   └── LocalController.start()
│       ├── Create temp dir (bundles, cache, fake_bundle)
│       ├── _create_local_autoscaler()
│       │   └── For each scale_group:
│       │       └── LocalVmManager → ScalingGroup
│       │   └── Autoscaler(scale_groups, vm_registry)
│       ├── Controller(config, RpcWorkerStubFactory(), autoscaler)
│       ├── controller.start()
│       └── returns "http://127.0.0.1:{port}"
│
├── connect()  [context manager]
│   ├── start()
│   ├── (no tunnel — direct connection)
│   ├── yield "http://127.0.0.1:{port}"
│   └── stop()
│
└── stop()
    └── LocalController.stop()
        ├── Controller.stop()
        └── temp_dir.cleanup()
```

### Local Mode: Inside the Autoscaler (on job submit)

```
Controller._run_scheduling_loop()         ← IDENTICAL to GCP
├── _run_scheduling()                     ← IDENTICAL
└── _run_autoscaler_once()                ← IDENTICAL
    └── Autoscaler.run_once()             ← IDENTICAL
        └── group.scale_up()              ← IDENTICAL (calls VmManager interface)
            └── LocalVmManager.create_vm_group()  ← LOCAL SHIM
                ├── Determine worker_count from topology
                ├── For each worker:
                │   ├── _LocalBundleProvider(fake_bundle)
                │   ├── _LocalImageProvider()
                │   ├── _LocalContainerRuntime()
                │   ├── LocalEnvironmentProvider(...)
                │   └── Worker(...).start()  (gRPC on localhost)
                └── LocalVmGroup(workers, vm_registry)
```

---

## Module Layout

```
src/iris/cluster/vm/
├── cluster_manager.py      # NEW — ClusterManager + make_local_config()
├── local_platform.py       # NEW — extracted from demo_cluster.py
│   ├── LocalVmManager      (VmManagerProtocol)
│   ├── LocalVmGroup        (VmGroupProtocol)
│   └── _StubManagedVm
├── controller.py           # MODIFIED — add LocalController + factory branch
│   ├── ControllerProtocol
│   ├── GcpController
│   ├── ManualController
│   ├── LocalController     # NEW
│   └── create_controller()
├── config.py               # MODIFIED — add local provider branch
├── vm_platform.py          # UNCHANGED
├── gcp_tpu_platform.py     # UNCHANGED
├── manual_platform.py      # UNCHANGED
└── ...

examples/
└── demo_cluster.py         # SIMPLIFIED — uses ClusterManager

scripts/
└── smoke-test.py           # SIMPLIFIED — uses ClusterManager

src/iris/rpc/
└── config.proto            # MODIFIED — add LocalProvider, LocalControllerConfig
```

---

## Detailed Design

### 1. `ClusterManager` — `vm/cluster_manager.py`

```python
"""Cluster lifecycle manager.

Provides a uniform interface for starting/stopping/connecting to an Iris
cluster regardless of backend (GCP, manual, local). Callers get a URL;
ClusterManager handles tunnel setup, mode detection, and cleanup.
"""

from contextlib import contextmanager
from collections.abc import Iterator

from iris.cluster.vm.config import load_config
from iris.cluster.vm.controller import ControllerProtocol, create_controller
from iris.cluster.vm.debug import controller_tunnel
from iris.rpc import config_pb2


def make_local_config(
    base_config: config_pb2.IrisClusterConfig,
) -> config_pb2.IrisClusterConfig:
    """Override a GCP/manual config to run locally.

    Replaces the controller oneof with LocalControllerConfig and every
    scale group's provider oneof with LocalProvider. Everything else
    (accelerator_type, accelerator_variant, min/max_slices) is preserved.

    Usage:
        config = load_config("eu-west4.yaml")
        local = make_local_config(config)
        manager = ClusterManager(local)
    """
    config = config_pb2.IrisClusterConfig()
    config.CopyFrom(base_config)
    config.controller_vm.ClearField("controller")
    config.controller_vm.local.port = 0  # auto-assign
    config.controller_vm.bundle_prefix = ""  # LocalController will set temp path
    for sg in config.scale_groups.values():
        sg.provider.ClearField("provider")
        sg.provider.local.SetInParent()
    return config


class ClusterManager:
    """Manages the full cluster lifecycle: controller + connectivity.

    Provides explicit start/stop methods and a connect() context manager
    that handles tunnel setup for GCP or direct connection for local mode.

    Example (smoke test / demo):
        manager = ClusterManager(config)
        with manager.connect() as url:
            client = IrisClient.remote(url)
            client.submit(...)

    Example (CLI - long-running):
        manager = ClusterManager(config)
        url = manager.start()
        # ... use cluster ...
        manager.stop()
    """

    def __init__(self, config: config_pb2.IrisClusterConfig):
        self._config = config
        self._controller: ControllerProtocol | None = None

    @property
    def is_local(self) -> bool:
        return self._config.controller_vm.WhichOneof("controller") == "local"

    def start(self) -> str:
        """Start the controller. Returns the controller address.

        For GCP: creates a GCE VM, bootstraps, returns internal IP.
        For local: starts in-process Controller, returns localhost URL.
        """
        self._controller = create_controller(self._config)
        return self._controller.start()

    def stop(self) -> None:
        """Stop the controller and clean up resources."""
        if self._controller:
            self._controller.stop()
            self._controller = None

    @contextmanager
    def connect(self) -> Iterator[str]:
        """Start controller, yield a usable URL, stop on exit.

        For GCP: establishes SSH tunnel, yields tunnel URL.
        For local: yields direct localhost URL (no tunnel).
        """
        address = self.start()
        try:
            if self.is_local:
                yield address
            else:
                zone = self._config.zone
                project = self._config.project_id
                label_prefix = self._config.label_prefix or "iris"
                with controller_tunnel(
                    zone, project, label_prefix=label_prefix
                ) as tunnel_url:
                    yield tunnel_url
        finally:
            self.stop()

    @property
    def controller(self) -> ControllerProtocol:
        """Access the underlying controller (must call start() first)."""
        if self._controller is None:
            raise RuntimeError("ClusterManager.start() not called")
        return self._controller
```

### 2. `LocalController` — added to `controller.py`

```python
class LocalController:
    """In-process controller for local testing.

    Runs Controller + Autoscaler(LocalVmManagers) in the current process.
    Workers are threads, not VMs. No Docker, no GCS, no SSH.
    """

    def __init__(self, config: config_pb2.IrisClusterConfig):
        self._config = config
        self._controller: Controller | None = None
        self._temp_dir: tempfile.TemporaryDirectory | None = None

    def start(self) -> str:
        self._temp_dir = tempfile.TemporaryDirectory(prefix="iris_local_")
        temp = Path(self._temp_dir.name)
        bundle_dir = temp / "bundles"; bundle_dir.mkdir()
        cache_path = temp / "cache"; cache_path.mkdir()
        fake_bundle = temp / "fake_bundle"; fake_bundle.mkdir()
        (fake_bundle / "pyproject.toml").write_text("[project]\nname='local'\n")

        port = self._config.controller_vm.local.port or find_free_port()
        address = f"http://127.0.0.1:{port}"

        autoscaler = _create_local_autoscaler(
            self._config, address, cache_path, fake_bundle,
        )
        self._controller = Controller(
            config=ControllerConfig(
                host="127.0.0.1",
                port=port,
                bundle_prefix=self._config.controller_vm.bundle_prefix
                    or f"file://{bundle_dir}",
            ),
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
        return self.restart()

    def discover(self) -> str | None:
        return self._controller.url if self._controller else None

    def status(self) -> ControllerStatus:
        if self._controller:
            return ControllerStatus(
                running=True, address=self._controller.url, healthy=True,
            )
        return ControllerStatus(running=False, address="", healthy=False)

    def fetch_startup_logs(self, tail_lines: int = 100) -> str | None:
        return "(local controller — no startup logs)"
```

### 3. `local_platform.py` — extracted from `demo_cluster.py`

This is a **move**, not new code. The classes `LocalVmManager`, `LocalVmGroup`,
and `_StubManagedVm` are extracted verbatim from `demo_cluster.py` lines 89–332
into `src/iris/cluster/vm/local_platform.py`. The `_create_local_autoscaler()`
function lives here too.

```python
"""Local platform: in-process VmManager for testing without GCP.

Extracted from demo_cluster.py. Provides LocalVmManager (VmManagerProtocol)
and LocalVmGroup (VmGroupProtocol) that create real Worker instances running
in-process with thread-based execution instead of Docker containers.

Worker dependencies (bundle, image, container, environment providers) come
from iris.cluster.client.local_client.
"""

# _StubManagedVm — holds VmInfo, immediately READY, no lifecycle threads
# LocalVmGroup   — VmGroupProtocol backed by in-process Workers
# LocalVmManager  — VmManagerProtocol, creates LocalVmGroups

def _create_local_autoscaler(
    config: config_pb2.IrisClusterConfig,
    controller_address: str,
    cache_path: Path,
    fake_bundle: Path,
) -> Autoscaler:
    """Create Autoscaler with LocalVmManagers for all scale groups.

    Parallels create_autoscaler_from_config() but uses LocalVmManagers.
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
            scale_up_cooldown_ms=1000,
            scale_down_cooldown_ms=300_000,
        )

    return Autoscaler(
        scale_groups=scale_groups,
        vm_registry=vm_registry,
        config=AutoscalerConfig(evaluation_interval_seconds=2.0),
    )
```

### 4. Proto Changes — `config.proto`

```protobuf
message ProviderConfig {
  oneof provider {
    TpuProvider tpu = 1;
    ManualProvider manual = 2;
    LocalProvider local = 3;      // NEW
  }
}

message LocalProvider {}  // No fields — config drives simulation via scale group

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
  int32 port = 1;  // 0 = auto-assign
}
```

### 5. Config Wiring — `config.py`

```python
# create_controller() — add "local" branch:
def create_controller(config):
    which = config.controller_vm.WhichOneof("controller")
    if which == "gcp":    return GcpController(config)
    if which == "manual": return ManualController(config)
    if which == "local":  return LocalController(config)
    raise ValueError(...)

# _get_provider_info() — add "local" branch:
#   Returns ("local", None, None)
# _create_manager_from_config() — NOT used for local
#   LocalVmManager needs runtime context (address, paths) that
#   LocalController provides. The standard factory path is not used.
```

---

## Caller Rewrites

### smoke-test.py — Before vs After

**Before** (current code, ~120 lines of lifecycle management):

```python
def run(self):
    cluster_config = load_config(self.config.config_path)
    zone = cluster_config.zone
    project = cluster_config.project_id

    # Phase 0a: Build images
    if not self._build_images(): return False

    # Phase 0b: Cleanup
    if self.config.clean_start:
        self._cleanup_existing(zone, project)

    # Phase 1: Start cluster
    controller = create_controller(cluster_config)
    controller.start()

    # Phase 2: SSH tunnel + log streaming
    self._start_log_streaming(zone, project)
    with controller_tunnel(zone, project) as tunnel_url:
        # Phase 3: Run tests
        self._run_tests(tunnel_url)
        success = self._print_results()

    # Cleanup
    controller.stop()
    return success
```

**After** (~60 lines, mode-aware at top only):

```python
def run(self):
    cluster_config = load_config(self.config.config_path)

    # Apply --local override if requested
    if self.config.local:
        cluster_config = make_local_config(cluster_config)

    manager = ClusterManager(cluster_config)

    # GCP-only setup phases
    if not manager.is_local:
        if not self._build_images(): return False
        if self.config.clean_start:
            self._cleanup_existing(cluster_config.zone, cluster_config.project_id)

    # Run tests (connect() handles tunnel for GCP, direct for local)
    with manager.connect() as url:
        if not manager.is_local:
            self._start_log_streaming(cluster_config.zone, cluster_config.project_id)
        self._run_tests(url)
        return self._print_results()
```

The `--local` flag is a one-liner that transforms any config:

```python
@click.option("--local", is_flag=True, help="Run locally without GCP")
```

### demo_cluster.py — Before vs After

**Before** (~845 lines, builds everything manually):

```python
class DemoCluster:
    def __enter__(self):
        self._temp_dir = tempfile.TemporaryDirectory(...)
        # ... 80 lines of manual Controller/Autoscaler/Worker wiring ...
        self._controller.start()
        return self

    def __exit__(self, ...):
        self._controller.stop()
        self._temp_dir.cleanup()
```

**After** (~150 lines total, Jupyter + seed logic preserved):

```python
class DemoCluster:
    def __init__(self, config_path: str | None = None, controller_url: str | None = None):
        self._remote_url = controller_url
        self._config_path = config_path
        self._manager: ClusterManager | None = None

    def __enter__(self):
        if self._remote_url:
            return self  # remote mode, no local infra

        config = self._load_or_default_config()
        config = make_local_config(config)
        self._manager = ClusterManager(config)
        self._manager.start()
        return self

    def __exit__(self, ...):
        self._stop_jupyter()
        if self._manager:
            self._manager.stop()

    @property
    def controller_url(self) -> str:
        if self._remote_url:
            return self._remote_url
        return self._manager.controller.discover()

    # seed_cluster(), launch_jupyter(), etc. — UNCHANGED
```

The ~700 lines of `LocalVmManager`, `LocalVmGroup`, `_StubManagedVm`, and manual
autoscaler wiring are deleted from `demo_cluster.py` — they now live in
`local_platform.py` and are used via `LocalController` inside `ClusterManager`.

### cli.py — Before vs After

The CLI commands that use `create_controller()` directly can optionally migrate to
`ClusterManager`, but this is not required in v3. The main win is that `cluster start`
and `cluster stop` could use `ClusterManager` to gain local mode support for free:

```python
# Before:
@cluster.command("start")
def cluster_start(ctx):
    config = ctx.obj["config"]
    _build_cluster_images(config)
    ctrl = create_controller(config)
    address = ctrl.start()
    click.echo(f"Controller started at {address}")

# After:
@cluster.command("start")
@click.option("--local", is_flag=True, help="Run locally without GCP")
def cluster_start(ctx, local: bool):
    config = ctx.obj["config"]
    if local:
        config = make_local_config(config)
    if not ClusterManager(config).is_local:
        _build_cluster_images(config)
    manager = ClusterManager(config)
    address = manager.start()
    click.echo(f"Controller started at {address}")
    # Note: for long-running CLI, caller manages stop
```

The `run-local` command becomes a thin wrapper around `ClusterManager` with
`make_local_config()`.

---

## What Changes, What Stays the Same

| Component | GCP Mode | Local Mode | Changed? |
|-----------|----------|------------|----------|
| `ClusterManager` | tunnel + GcpController | direct + LocalController | **NEW** |
| `Controller` | Runs on GCE VM | Runs in-process | **No** — same class |
| `Autoscaler` | Same | Same | **No** |
| `ScalingGroup` | Same | Same | **No** |
| `Scheduler` | Same | Same | **No** |
| `VmManager` | `TpuVmManager` | `LocalVmManager` | **Extracted** from demo_cluster |
| `VmGroup` | `TpuVmGroup` | `LocalVmGroup` | **Extracted** from demo_cluster |
| `Worker` | Docker + bootstrap | In-process threads | **No** — same class, different providers |
| SSH tunnel | gcloud ssh | Not needed | Handled by `ClusterManager.connect()` |
| Image build | Docker + push | Not needed | Skipped when `is_local` |

---

## Implementation Plan (Spiral)

### Step 1: Extract `local_platform.py` from `demo_cluster.py`

**Files changed:**
- `src/iris/cluster/vm/local_platform.py` — **new** (extracted code)
- `examples/demo_cluster.py` — imports from `local_platform` instead of defining inline
- `tests/cluster/vm/test_local_platform.py` — **new**

**Test:** `demo_cluster.py` works identically after extraction. Unit test creates
`LocalVmManager`, calls `create_vm_group()`, verifies workers are gRPC-reachable.

### Step 2: Proto + `LocalController` + config wiring

**Files changed:**
- `src/iris/rpc/config.proto` — add `LocalProvider`, `LocalControllerConfig`
- `scripts/generate-protos.py` — run
- `src/iris/cluster/vm/controller.py` — add `LocalController`
- `src/iris/cluster/vm/config.py` — add local provider branch
- `src/iris/cluster/vm/local_platform.py` — add `_create_local_autoscaler()`

**Test:** `create_controller(local_config)` returns `LocalController`.
Start it, submit a job via `IrisClient.remote()`, verify completion.

### Step 3: `ClusterManager` + `make_local_config()`

**Files changed:**
- `src/iris/cluster/vm/cluster_manager.py` — **new**
- `tests/cluster/vm/test_cluster_manager.py` — **new**

**Test:** `ClusterManager(local_config).connect()` yields a working URL.
Submit and complete a job through it.

### Step 4: Smoke test integration

**Files changed:**
- `scripts/smoke-test.py` — use `ClusterManager`, add `--local` flag

**Test:** `uv run python scripts/smoke-test.py --config examples/eu-west4.yaml --local`
passes all three test jobs locally.

### Step 5: Simplify `demo_cluster.py`

**Files changed:**
- `examples/demo_cluster.py` — use `ClusterManager` + `make_local_config()`

**Test:** `uv run python examples/demo_cluster.py` works as before.

### Step 6: CLI integration (optional)

**Files changed:**
- `src/iris/cli.py` — add `--local` to `cluster start`/`stop`

---

## `--local` Override: How It Works

The `make_local_config()` function transforms any config to local mode:

```
Input (eu-west4.yaml):                Output (in memory):
┌────────────────────────────┐        ┌────────────────────────────┐
│ controller_vm:             │        │ controller_vm:             │
│   image: ...docker.pkg...  │   →    │   local:                   │
│   gcp:                     │        │     port: 0                │
│     zone: europe-west4-b   │        │                            │
│     machine_type: n1-std-4 │        │                            │
│                            │        │                            │
│ scale_groups:              │        │ scale_groups:              │
│   tpu_v5e_16:              │        │   tpu_v5e_16:              │
│     provider:              │        │     provider:              │
│       tpu:                 │   →    │       local: {}            │
│         project: hai-gcp   │        │     accelerator_type: tpu  │  ← preserved
│     accelerator_type: tpu  │        │     accelerator_variant:   │  ← preserved
│     accelerator_variant:   │        │       v5litepod-16         │
│       v5litepod-16         │        │     max_slices: 4          │  ← preserved
│     max_slices: 4          │        │                            │
└────────────────────────────┘        └────────────────────────────┘
```

Scale group topology (accelerator_type, variant, min/max slices) is preserved
so local workers simulate the same TPU attributes. Only the provider and
controller backend change.

---

## Resolved Questions

**Q: Should local mode have its own config file?**
No. `make_local_config()` transforms any existing config. A standalone
`local.yaml` can exist for convenience but is not required.

**Q: Where does `ClusterManager` live?**
`src/iris/cluster/vm/cluster_manager.py`. It sits alongside `controller.py`
and `config.py` in the `vm` package — it orchestrates VM-level lifecycle.

**Q: Does `ClusterManager` replace `create_controller()`?**
No. `create_controller()` remains the factory for `ControllerProtocol`.
`ClusterManager` wraps it with tunnel/lifecycle management. Low-level callers
(e.g., unit tests) can still use `create_controller()` directly.

**Q: How does `demo_cluster.py` handle its hardcoded scale groups?**
`DemoCluster._load_or_default_config()` builds a config proto programmatically
(same as today) when no config file is provided. `make_local_config()` is then
applied. When a config file is provided, it's loaded and overridden.

**Q: Port allocation across multiple LocalVmManagers?**
`_create_local_autoscaler()` creates a single shared `PortAllocator(30000, 40000)`
passed to all `LocalVmManager` instances, preventing collisions.

---

## References

- `examples/demo_cluster.py` — existing local cluster (source for extraction)
- `src/iris/cluster/client/local_client.py` — local worker providers
- `src/iris/cluster/vm/vm_platform.py` — `VmManagerProtocol`, `VmGroupProtocol`
- `src/iris/cluster/vm/controller.py` — `ControllerProtocol`, `create_controller()`
- `src/iris/cluster/vm/config.py` — `create_autoscaler_from_config()`
- `src/iris/cluster/vm/debug.py` — `controller_tunnel()`
- `src/iris/cli.py` — CLI commands that use `create_controller()`
