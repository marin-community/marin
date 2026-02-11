# Iris Platform Redesign

## Problem

The current `cluster/vm/` layer conflates several concerns:

1. **Zone baked into platform**: `GcpPlatformConfig` has `zone` and `default_zones` fields. A multi-zone cluster must cascade through `group.zones → platform.default_zones → platform.zone`. The platform shouldn't own zone — it's per-slice.

2. **Confusing naming**: `vm/controller_vm.py` means "scripts to start the controller node" but shares namespace with the Controller service. `VmManagerProtocol` is really "slice factory" but the name doesn't convey that.

3. **Mixed responsibilities**: Controller VM lifecycle (`GcpController`), worker slice lifecycle (`TpuVmManager`), SSH utilities, bootstrap scripts, and platform dispatch are all tangled in `cluster/vm/`.

4. **No clear provider boundary**: Adding CoreWeave requires touching `platform.py`, `vm_platform.py`, `controller_vm.py`, and config protos in uncoordinated ways.

## Design Goals

- **Platform = infrastructure provider**: Create VMs, create slices, list/terminate resources. No lifecycle state machines.
- **Platform is zone/region agnostic**: Platform config has only `project_id`. Zone and region live on `SliceConfig` / `VmConfig`.
- **Controller is just a VM**: Eliminate `ControllerProtocol` — use `platform.create_vm()` + bootstrap uniformly across all platforms. GCP creates a GCE instance; Manual allocates from host list; Local starts in-process.
- **Adding a provider = one file**: `platform/coreweave.py` implements `Platform`, done.

## Directory Structure

Replace `cluster/vm/` with `cluster/platform/`:

```
src/iris/cluster/platform/
├── __init__.py         # Exports: Platform, SliceHandle, VmHandle, etc.
├── base.py             # Protocol definitions + status types
├── gcp.py              # GcpPlatform implementation
├── coreweave.py        # CoreweavePlatform stub (NotImplementedError)
├── manual.py           # ManualPlatform implementation
├── local.py            # LocalPlatform — in-process threads, full Platform interface
├── bootstrap.py        # Bootstrap script generation (worker + controller)
└── ssh.py              # SSH connection implementations (from vm/ssh.py)
```

Deleted: `cluster/vm/` (entirely replaced after migration).

## Protocol Definitions

### Platform

```python
class Platform(Protocol):
    """Infrastructure provider abstraction.

    Handles resource allocation (VMs, slices) and infrastructure operations.
    Does NOT manage lifecycle state machines — that's the controller layer's job.
    """

    def create_vm(self, config: VmConfig) -> StandaloneVmHandle:
        """Create a single VM (e.g., for the controller)."""
        ...

    def create_slice(self, config: SliceConfig) -> SliceHandle:
        """Create a slice of connected VMs (e.g., TPU pod, IB GPU cluster).

        The slice is the atomic scaling unit — it succeeds or fails as a whole.
        """
        ...

    def list_slices(
        self,
        zones: list[str],
        labels: dict[str, str] | None = None,
    ) -> list[SliceHandle]:
        """List existing slices, filtered by zone and optionally by labels.

        Labels are exact key=value match. This matches how all providers work:
        - GCP: `gcloud --filter="labels.key=value"` is exact equality
        - Manual: no labels (static host list, no filtering needed)
        - CoreWeave: Kubernetes label selectors are exact-match by default

        No config language or substring matching is needed — every current use
        case is exact-match (e.g., `{prefix}-scale-group={name}`).

        Zones are required because GCP TPU listing is per-zone. Callers always
        know which zones to query (from ScaleGroupConfig.zones). Making this
        optional with a silent default would hide bugs.

        Used by:
        - Autoscaler for discovery/adoption at startup
        - CLI for cleanup operations
        """
        ...

    def list_vms(
        self,
        zones: list[str],
        labels: dict[str, str] | None = None,
    ) -> list[VmHandle]:
        """List existing VMs, filtered by zone and optionally by labels.

        Labels are exact key=value match (same semantics as list_slices).
        Zones are required for the same reason as list_slices — consistent
        interface, and GCP operations are per-zone for TPU VMs even if GCE
        instances can technically be listed project-wide.

        Used by CLI and lifecycle functions to discover the controller VM.
        """
        ...

    def tunnel(
        self,
        address: str,
        local_port: int | None = None,
    ) -> AbstractContextManager[str]:
        """Create a tunnel to a remote address if needed.

        GCP: SSH tunnel with port forwarding.
        Manual/Local: returns address directly (nullcontext).
        """
        ...

    def shutdown(self) -> None:
        """Release platform-owned resources (threads, connections, caches).

        Distinct from terminate() on handles — shutdown() doesn't destroy
        cloud resources. It cleans up the Platform object itself.

        For LocalPlatform this stops worker threads managed by ThreadContainer.
        For GCP/Manual this is typically a no-op.

        Called by the autoscaler during Autoscaler.shutdown() — replaces the
        current `group._vm_manager.stop()` call.
        """
        ...


def create_platform(
    config: PlatformConfig,
    ssh_config: config_pb2.SshConfig | None = None,
) -> Platform:
    """Factory: dispatch on config oneof to create the right Platform.

    Standalone function rather than staticmethod on Platform — protocols
    describe interface requirements for implementors, not construction logic.

    ssh_config is needed by GCP (for gcloud SSH) and Manual (for direct SSH).
    Local platform doesn't use it.
    """
    label_prefix = config.label_prefix or "iris"
    which = config.WhichOneof("platform")
    if which == "gcp":
        return GcpPlatform(config.gcp, label_prefix, ssh_config)
    elif which == "manual":
        return ManualPlatform(label_prefix, ssh_config)
    elif which == "local":
        return LocalPlatform(label_prefix)
    elif which == "coreweave":
        return CoreweavePlatform(config.coreweave, label_prefix)
    raise ValueError(f"Unknown platform type: {which}")
```

### SliceHandle

```python
class SliceHandle(Protocol):
    """Handle to an allocated slice of connected VMs.

    A slice is the atomic scaling unit. For TPUs, it's a complete pod.
    For GPUs, it could be a set of IB-connected nodes.
    """

    @property
    def slice_id(self) -> str:
        """Unique identifier (e.g., 'iris-tpu_v5e_16-1738000000000')."""
        ...

    @property
    def zone(self) -> str:
        """Zone where this slice is allocated."""
        ...

    @property
    def scale_group(self) -> str:
        """Name of the scale group this slice belongs to.

        Extracted from labels (e.g., labels["{prefix}-scale-group"]).
        Needed by the autoscaler to associate discovered slices with groups.
        """
        ...

    @property
    def labels(self) -> dict[str, str]:
        """Labels/tags set on this slice at creation time.

        Used for filtering in list_slices() and for diagnostics.
        """
        ...

    @property
    def created_at(self) -> Timestamp:
        """When this slice was created. Used for age-based scaling decisions."""
        ...

    def list_vms(self) -> list[VmHandle]:
        """Live query: individual VMs in this slice (one per worker in multi-host).

        Always queries the cloud for current state. For GCP TPUs, this calls
        `gcloud compute tpus describe` to get current network endpoints.
        If the cloud has repaired/replaced a worker, this reflects it.

        The controller layer (ScalingGroup) maintains its own WorkerVm registry
        for lifecycle tracking — it does not depend on this returning a cached list.
        """
        ...

    def terminate(self) -> None:
        """Destroy the slice and all its VMs."""
        ...

    def status(self) -> SliceStatus:
        """Cloud-level status of the slice (e.g., CREATING, READY, DELETING)."""
        ...
```

### VmHandle

```python
class VmHandle(Protocol):
    """Handle to a single VM within a slice. Provides raw infrastructure operations.

    The orchestration layer (WorkerVm) wraps this with lifecycle state machines,
    retries, and health checking.

    No terminate — slices are the atomic unit. Individual slice members cannot
    be terminated independently (e.g., TPU pod workers). Termination lives on
    SliceHandle.
    """

    @property
    def vm_id(self) -> str: ...

    @property
    def internal_address(self) -> str:
        """Internal/private IP address of this VM.

        For GCP: VPC-internal IP (used for worker-to-worker and controller
        communication within the same network).
        For Manual: the configured host address.
        For CoreWeave: pod-internal IP.

        This is the primary address used for all intra-cluster communication.
        """
        ...

    @property
    def external_address(self) -> str | None:
        """External/public IP address of this VM, if available.

        Returns None when the VM has no external IP (e.g., private GCE instances,
        manual hosts behind NAT). The tunnel() method on Platform is the
        preferred way to reach VMs externally — this property is for cases
        where the caller needs the raw address (e.g., diagnostics, logging).
        """
        ...

    def status(self) -> VmStatus:
        """Cloud-level VM status."""
        ...

    def wait_for_connection(
        self,
        timeout: Duration,
        poll_interval: Duration = Duration.from_seconds(5),
    ) -> bool:
        """Poll until SSH/connection is available. Returns False on timeout."""
        ...

    def run_command(
        self,
        command: str,
        timeout: Duration | None = None,
        on_line: Callable[[str], None] | None = None,
    ) -> CommandResult:
        """Run a command on the VM via SSH. Optionally stream output lines."""
        ...

    def bootstrap(self, script: str) -> None:
        """Run a bootstrap script on the VM.

        Equivalent to run_command(f'bash -c {script}') with streaming.
        Raises on non-zero exit.
        """
        ...

    def reboot(self) -> None:
        """Reboot the VM."""
        ...


class StandaloneVmHandle(VmHandle, Protocol):
    """Handle to a standalone VM (e.g., controller). Can be terminated and labeled.

    Returned by platform.create_vm(). Extends VmHandle with operations that
    only make sense for independently-managed VMs:
    - terminate(): Destroy the VM
    - set_labels(): Tag for discovery (controller discovery uses labels)
    - set_metadata(): Pass data to the VM (controller address)

    Slice member VMs (from SliceHandle.list_vms()) are plain VmHandle —
    they can't be individually terminated, and their labels/metadata are
    set on the slice at creation time.
    """

    def terminate(self) -> None:
        """Destroy the VM."""
        ...

    def set_labels(self, labels: dict[str, str]) -> None:
        """Set labels on the VM (for discovery via list_vms).

        GCE label values: lowercase alphanumeric + hyphens, max 63 chars.
        """
        ...

    def set_metadata(self, metadata: dict[str, str]) -> None:
        """Set arbitrary key-value metadata on the VM.

        Unlike labels, metadata values have no character restrictions.
        On GCP, accessible via the metadata server from within the VM.
        Used for passing data to VMs (e.g., controller address).
        """
        ...
```

`create_vm()` returns `StandaloneVmHandle`. `SliceHandle.list_vms()` returns
`list[VmHandle]` (no terminate, no labeling). This makes ownership explicit.

### Status Types

These are cloud-level status types — distinct from the Iris lifecycle states
in `vm.proto` (BOOTING, INITIALIZING, READY, etc.). Cloud status represents
the infrastructure provider's view; Iris lifecycle represents the controller's
view after bootstrap.

These should NOT be protos. They are ephemeral, returned by platform operations,
never stored or transmitted over RPC. But the state field should use enums
(not raw strings) for type safety. Define them in `platform/base.py`:

```python
class CloudSliceState(StrEnum):
    """Cloud-level slice states. Provider implementations map to these."""
    CREATING = "CREATING"
    READY = "READY"
    REPAIRING = "REPAIRING"
    DELETING = "DELETING"
    UNKNOWN = "UNKNOWN"

class CloudVmState(StrEnum):
    """Cloud-level VM states."""
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"
    TERMINATED = "TERMINATED"
    UNKNOWN = "UNKNOWN"

@dataclass
class SliceStatus:
    """Cloud-level slice status."""
    state: CloudSliceState
    vm_count: int

@dataclass
class VmStatus:
    """Cloud-level VM status."""
    state: CloudVmState

@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str
```

## Config Types

### VmConfig

Both `VmConfig` and `SliceConfig` are proto messages with `oneof` for the
platform-specific portion. This matches the existing proto patterns
(`PlatformConfig`, `ControllerVmConfig` both use `oneof`), gives proper
tagged union semantics via `WhichOneof()`, and enables proto serialization
for logging/debugging. Defined in `config.proto`.

**Labels vs Metadata**: Both are `map<string, string>` on VmConfig, but serve
different purposes. GCE supports both independently — labels can be set at
creation time and updated later via `gcloud compute instances update --update-labels`.

- **Labels**: For resource filtering/discovery (used in `list_vms(labels=...)`).
  On GCP these map to GCE instance labels, filtered via `--filter="labels.key=value"`.
  Label values must be lowercase alphanumeric + hyphens (max 63 chars).
- **Metadata**: For passing arbitrary data to VMs (accessible via GCP metadata
  server or `gcloud compute instances list --format`). No value restrictions.

Controller discovery uses **both**:
- **Label** `iris-controller-{prefix}=true` for discovery via `list_vms(labels=...)`
- **Metadata** `iris-controller-address-{prefix}=http://10.x.x.x:10000` for the
  address (which contains colons/dots — invalid in label values)

This is a change from the current code, which uses metadata for both. The
migration tags controller VMs with a label (for `list_vms`) AND metadata
(for the address value that workers read during bootstrap).

```protobuf
// --- VmConfig: used with platform.create_vm() ---

message GcpVmConfig {
  string zone = 1;
  string machine_type = 2;      // Default: "n2-standard-4"
  int32 boot_disk_size_gb = 3;  // Default: 50
}

message ManualVmConfig {
  string host = 1;
  string ssh_user = 2;          // Default: "root"
  string ssh_key_file = 3;
}

message VmConfig {
  string name = 1;
  map<string, string> labels = 2;
  map<string, string> metadata = 3;
  oneof platform {
    GcpVmConfig gcp = 10;
    ManualVmConfig manual = 11;
  }
}

// --- SliceConfig: used with platform.create_slice() ---

message GcpSliceConfig {
  string zone = 1;
  string runtime_version = 2;
  bool preemptible = 3;
  string topology = 4;
}

message CoreweaveSliceConfig {
  string region = 1;
  string instance_type = 2;
}

message ManualSliceConfig {
  repeated string hosts = 1;
  string ssh_user = 2;          // Default: "root"
  string ssh_key_file = 3;
}

message SliceConfig {
  string name_prefix = 1;
  int32 slice_size = 2;
  AcceleratorType accelerator_type = 3;
  string accelerator_variant = 4;
  map<string, string> labels = 5;
  oneof platform {
    GcpSliceConfig gcp = 10;
    CoreweaveSliceConfig coreweave = 11;
    ManualSliceConfig manual = 12;
  }
}
```

### Proto Changes

The proto `PlatformConfig` changes to remove `zone` from platform level:

```protobuf
// config.proto changes

message GcpPlatformConfig {
  string project_id = 1;
  // REMOVED: region (field 2) — now per-slice
  // REMOVED: zone (field 3) — now per-slice
  // REMOVED: default_zones (field 4) — now per-slice
}

// New: CoreWeave platform
message CoreweavePlatformConfig {
  string region = 1;
}

message PlatformConfig {
  string label_prefix = 10;
  oneof platform {
    GcpPlatformConfig gcp = 1;
    ManualPlatformConfig manual = 2;
    LocalPlatformConfig local = 3;
    CoreweavePlatformConfig coreweave = 4;  // NEW
  }
}
```

`ScaleGroupConfig` is restructured to embed a `SliceConfig` directly rather
than duplicating all the same fields. Currently `ScaleGroupConfig` has
`accelerator_variant`, `runtime_version`, `preemptible`, `zones`, `topology` —
all of which are just slice creation params. The restructured proto separates:

1. **Scaling policy** (min_slices, max_slices, priority) — autoscaler's concern
2. **Slice template** (SliceConfig) — what to create when scaling up
3. **Resource declaration** (resources) — for scheduling/demand matching

This eliminates the `scale_group_to_slice_config()` conversion function entirely.
When scaling up, the autoscaler stamps the template with labels and a unique name:

```protobuf
message ScaleGroupConfig {
  string name = 1;

  // Scaling policy
  int32 min_slices = 3 [features.field_presence = EXPLICIT];
  int32 max_slices = 4 [features.field_presence = EXPLICIT];
  int32 priority = 30 [features.field_presence = EXPLICIT];

  // What to create when scaling up — the slice template
  SliceConfig slice_template = 50;

  // Resource declarations for demand matching
  ScaleGroupResources resources = 14;
}
```

The autoscaler stamps labels and generates a unique name at scale-up time:

```python
def prepare_slice_config(
    template: config_pb2.SliceConfig,
    group_name: str,
    label_prefix: str,
) -> config_pb2.SliceConfig:
    """Stamp a slice template with autoscaler labels and unique name."""
    config = config_pb2.SliceConfig()
    config.CopyFrom(template)
    config.name_prefix = f"{label_prefix}-{group_name}"
    config.labels[f"{label_prefix}-managed"] = "true"
    config.labels[f"{label_prefix}-scale-group"] = group_name
    return config
```

The `VmType` enum is eliminated — the `SliceConfig.WhichOneof("platform")`
provides the same dispatch.

## How Components Map

### Current → New

| Current | New | Notes |
|---------|-----|-------|
| `Platform` protocol (platform.py) | `Platform` protocol (base.py) | Expanded API: create_vm, create_slice, list_slices, list_vms |
| `VmManagerProtocol` (vm_platform.py) | Eliminated | Absorbed into `Platform.create_slice()` + `Platform.list_slices()` |
| `VmGroupProtocol` (vm_platform.py) | `SliceHandle` (base.py) | Renamed. Status computation stays, but WorkerVm wrapping moves to controller layer |
| `TpuVmGroup` (gcp_tpu_platform.py) | `GcpSliceHandle` (gcp.py) | Implements SliceHandle for GCP TPU pods |
| `TpuVmManager` (gcp_tpu_platform.py) | Absorbed into `GcpPlatform` | create_slice() and list_slices() methods |
| `ManualVmManager/Group` (manual_platform.py) | `ManualPlatform` + `ManualSliceHandle` (manual.py) | Fresh implementation, full Platform interface |
| `ManagedVm` (managed_vm.py) | `WorkerVm` in controller layer | Renamed. Wraps `VmHandle` instead of `SshConnection`. Lifecycle thread unchanged |
| `VmRegistry` / `TrackedVmFactory` (managed_vm.py) | Stays in controller layer | Wraps `VmHandle` instead of directly creating `WorkerVm` |
| `ControllerProtocol` (controller_vm.py) | Eliminated | Controller lifecycle becomes top-level functions using `Platform` |
| `GcpController` (controller_vm.py) | `start_controller()` function | Uses `platform.create_vm()` + `platform.list_vms()` |
| `ManualController` (controller_vm.py) | `start_controller()` function | Same function, different VmConfig |
| `PlatformOps` (platform.py) | Eliminated | Absorbed into `Platform.list_slices()` + `SliceHandle.terminate()` |
| `create_platform()` (platform.py) | `create_platform()` (base.py) | Standalone factory function (not a staticmethod on Protocol) |
| Bootstrap scripts (managed_vm.py, controller_vm.py) | `platform/bootstrap.py` | Centralized script generation |
| SSH connections (ssh.py) | `platform/ssh.py` | Moved, API unchanged |

### Controller VM Lifecycle (New)

The controller lifecycle becomes a set of functions in `cluster/controller/lifecycle.py`
rather than a class hierarchy. These functions use `platform.create_vm()` uniformly
across all platforms — every platform implements the full interface:

- **GCP**: Creates a GCE instance, SSHs in, bootstraps the controller container.
- **Manual**: Allocates a host from the configured host list, SSHs in, bootstraps.
- **Local**: Starts the controller in-process, returns a stub VmHandle.

The CLI dispatches based on `config.controller.WhichOneof("controller")` only to
build the right `VmConfig` — the lifecycle functions themselves are platform-agnostic.

```python
# cluster/controller/lifecycle.py

def start_controller(
    platform: Platform,
    config: config_pb2.IrisClusterConfig,
) -> tuple[str, StandaloneVmHandle]:
    """Start or discover existing controller. Returns (address, vm_handle).

    1. Try to discover an existing healthy controller
    2. If found and healthy, return it
    3. Otherwise, create a new VM and bootstrap it
    """
    label_prefix = config.platform.label_prefix or "iris"
    zones = _controller_zones(config)

    # Check for existing controller
    existing_vm = _discover_controller_vm(platform, zones, label_prefix)
    if existing_vm:
        ip = existing_vm.internal_address
        port = _controller_port(config)
        if _check_controller_health(existing_vm, port):
            return f"http://{ip}:{port}", existing_vm
        # Unhealthy — terminate and recreate
        existing_vm.terminate()

    # Create new controller VM
    vm_config = _build_controller_vm_config(config)
    vm = platform.create_vm(vm_config)

    # Bootstrap
    vm.wait_for_connection(timeout=Duration.from_seconds(300))
    script = build_controller_bootstrap_script(config)
    vm.bootstrap(script)

    # Health check
    port = _controller_port(config)
    address = f"http://{vm.internal_address}:{port}"
    if not _wait_healthy(vm, port):
        raise RuntimeError(f"Controller at {address} failed health check")

    # Tag for discovery: label for list_vms(), metadata for address
    # (address contains colons/dots which are invalid in GCE label values)
    vm.set_labels({f"iris-controller-{label_prefix}": "true"})
    vm.set_metadata({f"iris-controller-address-{label_prefix}": address})

    return address, vm


def stop_controller(platform: Platform, config: config_pb2.IrisClusterConfig) -> None:
    """Find and terminate the controller VM."""
    label_prefix = config.platform.label_prefix or "iris"
    zones = _controller_zones(config)
    vm = _discover_controller_vm(platform, zones, label_prefix)
    if vm:
        vm.terminate()


def reload_controller(
    platform: Platform,
    config: config_pb2.IrisClusterConfig,
) -> str:
    """Re-bootstrap the controller on existing VM."""
    label_prefix = config.platform.label_prefix or "iris"
    zones = _controller_zones(config)
    vm = _discover_controller_vm(platform, zones, label_prefix)
    if not vm:
        raise RuntimeError("Controller VM not found")

    script = build_controller_bootstrap_script(config)
    vm.bootstrap(script)

    port = _controller_port(config)
    address = f"http://{vm.internal_address}:{port}"
    if not _wait_healthy(vm, port):
        raise RuntimeError(f"Controller at {address} failed health check")

    return address


def _discover_controller_vm(
    platform: Platform, zones: list[str], label_prefix: str,
) -> StandaloneVmHandle | None:
    """Find existing controller VM by labels.

    list_vms returns VmHandle, but controller VMs were created by create_vm()
    and support terminate/set_labels. We cast to StandaloneVmHandle since we
    know the controller VM supports these operations.
    """
    vms = platform.list_vms(
        zones=zones,
        labels={f"iris-controller-{label_prefix}": "true"},
    )
    return cast(StandaloneVmHandle, vms[0]) if vms else None
```

### ScalingGroup Changes

ScalingGroup switches from holding a `VmManagerProtocol` to holding a `Platform` reference:

```python
class ScalingGroup:
    def __init__(
        self,
        config: config_pb2.ScaleGroupConfig,
        platform: Platform,                    # NEW: replaces vm_manager
        label_prefix: str,                     # NEW: for label generation
        ...
    ):
        self._config = config
        self._platform = platform
        self._label_prefix = label_prefix
        ...

    def scale_up(self, ...) -> SliceHandle:
        slice_config = prepare_slice_config(
            self._config.slice_template, self._config.name, self._label_prefix,
        )
        slice_handle = self._platform.create_slice(slice_config)

        # Query current VMs (live) and wrap in WorkerVm for lifecycle tracking.
        # From here on, ScalingGroup owns the WorkerVm registry — it doesn't
        # re-query list_vms() until reconcile.
        managed_vms = []
        for vm in slice_handle.list_vms():
            managed_vm = self._vm_factory.create_vm(vm, ...)
            managed_vms.append(managed_vm)

        # Track the slice
        self._slices[slice_handle.slice_id] = SliceState(
            slice_handle=slice_handle,
            managed_vms=managed_vms,
        )
        return slice_handle

    def reconcile(self) -> None:
        """Discover existing slices at startup."""
        zones = self._zones_from_template()
        for slice_handle in self._platform.list_slices(
            zones=zones,
            labels={f"{self._label_prefix}-scale-group": self.name},
        ):
            managed_vms = [
                self._vm_factory.create_vm(vm, ...)
                for vm in slice_handle.list_vms()
            ]
            self._slices[slice_handle.slice_id] = SliceState(
                slice_handle=slice_handle,
                managed_vms=managed_vms,
            )

    def scale_down(self, slice_id: str, ...) -> None:
        state = self._slices.pop(slice_id, None)
        if state:
            for vm in state.managed_vms:
                vm.stop()
            state.slice_handle.terminate()
```

### WorkerVm Changes (renamed from ManagedVm)

`ManagedVm` is renamed to `WorkerVm`. The only consumer is the autoscaler's
ScalingGroup, and the only VMs it manages are workers. The name should reflect
what it is, not hedge on hypothetical reuse.

WorkerVm wraps `VmHandle` instead of `SshConnection`:

```python
class WorkerVm:
    def __init__(
        self,
        vm_handle: VmHandle,               # NEW: replaces SshConnection
        vm_id: str,
        slice_id: str,
        scale_group: str,
        bootstrap_config: config_pb2.BootstrapConfig,
        timeouts: config_pb2.TimeoutConfig,
        discovery_preamble: str = "",
    ):
        self._handle = vm_handle
        ...
        # Start lifecycle thread as before

    def _run(self, stop_event: threading.Event) -> None:
        """BOOTING → INITIALIZING → READY."""
        boot_timeout = Duration.from_proto(self._timeouts.boot_timeout)

        # Wait for connection (delegates to VmHandle)
        if not self._handle.wait_for_connection(boot_timeout):
            self._transition(VM_STATE_FAILED)
            return

        self._transition(VM_STATE_INITIALIZING)

        # Build and run bootstrap script
        script = _build_bootstrap_script(self._bootstrap_config, ...)
        self._handle.run_command(script, on_line=self._log)

        self._transition(VM_STATE_READY)
```

**Config threading**: The key simplification is that `ssh_config` no longer threads
through WorkerVm. The current flow is:

```
create_platform(platform_config, bootstrap_config, timeout_config, ssh_config)
  → Platform.__init__ stores ssh_config
    → Platform.vm_manager(group_config, vm_factory)
      → VmManager.__init__ stores ssh_config
        → VmManager.create_vm_group() creates SshConnection from ssh_config
          → ManagedVm.__init__ receives SshConnection
```

The new flow:

```
create_platform(platform_config, ssh_config)
  → Platform.__init__ stores ssh_config internally
    → Platform.create_slice(slice_config) creates VmHandles with SSH baked in
      → VmHandle.run_command() uses internal SSH connection
        → WorkerVm.__init__ receives VmHandle (no ssh_config needed)
```

WorkerVm only needs `BootstrapConfig` (for the bootstrap script content) and
`TimeoutConfig` (for boot/connection timeouts). SSH is the Platform's concern.
`TrackedVmFactory` creates WorkerVm instances given a VmHandle + bootstrap/timeout config.

### ClusterManager → Free Functions

ClusterManager is eliminated. With `start_controller()` / `stop_controller()`
as free functions and `create_platform()` as a standalone factory, the class
adds no value — it's just holding config and a platform reference that callers
already have.

Replace with a `connect_cluster()` context manager for the CLI's primary
use case (start + tunnel + stop):

```python
# cluster/manager.py

@contextmanager
def connect_cluster(config: config_pb2.IrisClusterConfig) -> Iterator[str]:
    """Start controller, open tunnel, yield address, stop on exit."""
    platform = create_platform(config.platform, ssh_config=config.ssh)
    address, vm = start_controller(platform, config)
    try:
        with platform.tunnel(address) as tunnel_url:
            yield tunnel_url
    finally:
        stop_controller(platform, config)
        platform.shutdown()


def stop_all(config: config_pb2.IrisClusterConfig) -> None:
    """Stop controller and all worker slices."""
    platform = create_platform(config.platform, ssh_config=config.ssh)
    label_prefix = config.platform.label_prefix or "iris"
    stop_controller(platform, config)
    for group_config in config.scale_groups.values():
        zones = [group_config.slice_template.gcp.zone] if group_config.slice_template.HasField("gcp") else []
        for slice_handle in platform.list_slices(
            zones=zones,
            labels={f"{label_prefix}-scale-group": group_config.name},
        ):
            slice_handle.terminate()
```

## GcpPlatform Implementation Sketch

```python
class GcpPlatform:
    def __init__(self, gcp_config: GcpPlatformConfig, label_prefix: str, ssh_config: SshConfig):
        self._project_id = gcp_config.project_id
        self._label_prefix = label_prefix
        self._ssh_config = ssh_config

    def create_vm(self, config: VmConfig) -> StandaloneVmHandle:
        gcp = config.gcp
        # gcloud compute instances create ...
        cmd = [
            "gcloud", "compute", "instances", "create", config.name,
            f"--project={self._project_id}",
            f"--zone={gcp.zone}",
            f"--machine-type={gcp.machine_type}",
            ...
        ]
        subprocess.run(cmd, ...)

        # Get internal IP
        ip = _get_vm_ip(self._project_id, gcp.zone, config.name)
        ssh_conn = GceSshConnection(self._project_id, gcp.zone, config.name)
        # GcpStandaloneVmHandle implements StandaloneVmHandle (terminate + labels + metadata)
        return GcpStandaloneVmHandle(
            vm_id=config.name,
            internal_address=ip,
            zone=gcp.zone,
            project_id=self._project_id,
            ssh=ssh_conn,
        )

    def create_slice(self, config: SliceConfig) -> SliceHandle:
        gcp = config.gcp
        slice_id = f"{config.name_prefix}-{Timestamp.now().epoch_ms()}"

        # gcloud compute tpus tpu-vm create ...
        cmd = [
            "gcloud", "compute", "tpus", "tpu-vm", "create", slice_id,
            f"--zone={gcp.zone}",
            f"--project={self._project_id}",
            f"--accelerator-type={config.accelerator_variant}",
            f"--version={gcp.runtime_version}",
            "--labels", _format_labels(config.labels),
        ]
        if gcp.preemptible:
            cmd.append("--preemptible")
        subprocess.run(cmd, ...)

        # Return a handle — list_vms() will do a live query when called
        return GcpSliceHandle(
            slice_id=slice_id,
            zone=gcp.zone,
            project_id=self._project_id,
        )

    def list_slices(self, zones: list[str], labels=None) -> list[SliceHandle]:
        results = []
        for zone in zones:
            tpus = _gcloud_list_tpus(self._project_id, zone, labels)
            for tpu in tpus:
                results.append(GcpSliceHandle(...))
        return results

    def list_vms(self, zones: list[str], labels=None) -> list[VmHandle]:
        results = []
        for zone in zones:
            instances = _gcloud_list_instances(self._project_id, zone, labels)
            for inst in instances:
                results.append(GcpVmHandle(...))
        return results

    def tunnel(self, address, local_port=None) -> AbstractContextManager[str]:
        return controller_tunnel(
            project=self._project_id,
            label_prefix=self._label_prefix,
            local_port=local_port,
        )

    def shutdown(self) -> None:
        pass  # No platform-owned resources to clean up
```

## LocalPlatform Design

LocalPlatform implements the full `Platform` interface — no exceptions, no
stubs. The difference is that "VMs" are in-process threads, not remote hosts.

1. **`create_vm()` starts an in-process controller**: Returns a `_LocalVmHandle`
   where `run_command()` executes locally and `internal_address` is `localhost:{port}`.
   This lets `start_controller()` work uniformly across all platforms.
2. **`create_slice()` spawns worker threads**: Each worker runs in a
   `ThreadContainer`. Returns a `LocalSliceHandle` whose `list_vms()` returns
   `_LocalVmHandle` instances.
3. **`shutdown()` is critical**: Stops worker threads owned by the ThreadContainer.
   For GcpPlatform, `shutdown()` is a no-op. For LocalPlatform, it replaces the
   current `LocalVmManager.stop()` call.
4. **No discovery/reconcile**: `list_slices()` / `list_vms()` return resources
   created in this process. There's no cloud state to rediscover after a restart.

```python
class LocalPlatform:
    """Platform for local testing — workers run as in-process threads.

    Implements the full Platform interface. "VMs" are threads, "slices"
    are groups of worker threads, and "standalone VMs" (controller) are
    in-process server instances.
    """

    def __init__(self, label_prefix: str):
        self._label_prefix = label_prefix
        self._threads = ThreadContainer()
        self._slices: dict[str, LocalSliceHandle] = {}
        self._vms: dict[str, _LocalVmHandle] = {}

    def create_vm(self, config: VmConfig) -> StandaloneVmHandle:
        """Start an in-process 'VM'. Used by start_controller() for local mode."""
        handle = _LocalStandaloneVmHandle(
            vm_id=config.name,
            internal_address="localhost",  # port assigned after bootstrap
            labels=dict(config.labels),
            metadata=dict(config.metadata),
        )
        self._vms[config.name] = handle
        return handle

    def create_slice(self, config: SliceConfig) -> SliceHandle:
        slice_id = f"{config.name_prefix}-{Timestamp.now().epoch_ms()}"
        workers = []
        for i in range(config.slice_size):
            worker = Worker(...)
            self._threads.spawn(worker.run, name=f"{slice_id}-{i}")
            workers.append(_LocalVmHandle(
                vm_id=f"{slice_id}-worker-{i}",
                internal_address=f"localhost:{worker.port}",
                worker=worker,
            ))
        handle = LocalSliceHandle(
            slice_id=slice_id,
            vms=workers,
            labels=dict(config.labels),
        )
        self._slices[slice_id] = handle
        return handle

    def list_slices(self, zones: list[str], labels=None) -> list[SliceHandle]:
        results = list(self._slices.values())
        if labels:
            results = [s for s in results if all(
                s.labels.get(k) == v for k, v in labels.items()
            )]
        return results

    def list_vms(self, zones: list[str], labels=None) -> list[VmHandle]:
        results = list(self._vms.values())
        if labels:
            results = [v for v in results if all(
                v.labels.get(k) == val for k, val in labels.items()
            )]
        return results

    def tunnel(self, address, local_port=None) -> AbstractContextManager[str]:
        return nullcontext(address)

    def shutdown(self) -> None:
        """Stop all worker threads. Critical for clean test teardown."""
        self._threads.stop(timeout=Duration.from_seconds(5.0))
        self._slices.clear()
        self._vms.clear()
```

`_LocalVmHandle` implements `VmHandle` — `run_command()` executes locally
(subprocess or in-process), `wait_for_connection()` returns True immediately.
`_LocalStandaloneVmHandle` extends it with `terminate()`, `set_labels()`,
`set_metadata()` (just updates in-memory dicts).

In tests, use a pytest fixture for cleanup:
```python
@pytest.fixture
def local_platform():
    p = LocalPlatform("test")
    yield p
    p.shutdown()
```

The autoscaler calls `platform.shutdown()` during `Autoscaler.shutdown()`,
replacing the current per-group `_vm_manager.stop()` loop.

## Migration Plan (Spiral)

Each step is independently testable and leaves the build green.

### Step 1: Proto additions (additive only)

Add all new proto messages without removing or renaming anything. This is
purely additive — existing code continues to compile unchanged.

**Additions to `rpc/config.proto`**:
- Add `CoreweavePlatformConfig` with `region` field
- Add `coreweave` branch to `PlatformConfig` oneof
- Add `VmConfig`, `GcpVmConfig`, `ManualVmConfig` messages
- Add `SliceConfig`, `GcpSliceConfig`, `CoreweaveSliceConfig`, `ManualSliceConfig` messages
- Add `SliceConfig slice_template` field to `ScaleGroupConfig`
  (existing flat fields remain for now — they'll be removed in Step 8)

Old fields (`zone`, `region`, `default_zones` on `GcpPlatformConfig`, `VmType`
enum, flat fields on `ScaleGroupConfig`) are **not removed yet**. They are
ignored by new code but still present for old code to compile against.

**Run**: `uv run python scripts/generate_protos.py` to regenerate.

**Test**: `uv run pytest` — full suite green. No code changes, only new messages added.

### Step 2: Create `platform/` with protocols + SSH

- Create `cluster/platform/` directory
- Define `Platform`, `SliceHandle`, `VmHandle`, `StandaloneVmHandle` protocols in `base.py`
- Define `CloudSliceState`, `CloudVmState`, `SliceStatus`, `VmStatus`, `CommandResult` in `base.py`
- Define `PlatformError`, `QuotaExhaustedError`, `ResourceNotFoundError`, `PlatformUnavailableError` in `base.py`
- Move SSH implementations to `platform/ssh.py` (update imports in `vm/` — old paths re-export during migration)
- Move bootstrap script generation to `platform/bootstrap.py`
- **Test**: All existing tests still pass. New protocols are importable.

### Step 3: GcpPlatform

- Implement `GcpPlatform` in `platform/gcp.py` (fresh implementation, not a wrapper):
  - `__init__(gcp_config, label_prefix, ssh_config)` — no region/zone
  - `create_slice()` — gcloud TPU create, label attachment, returns `GcpSliceHandle`
  - `list_slices(zones, labels)` — gcloud TPU list with label filter
  - `create_vm()` — gcloud compute instances create, returns `GcpStandaloneVmHandle`
  - `list_vms(zones, labels)` — gcloud compute instances list with label filter.
    **Note**: Current controller discovery uses metadata filters; this migrates to
    labels. Existing controllers will need to be re-tagged with labels (or torn down
    and recreated — acceptable since we're replacing all clusters).
  - `tunnel()` — SSH tunnel with port forwarding
  - `shutdown()` — no-op
- `GcpSliceHandle` — manages TPU pod lifecycle (list workers, terminate, status)
- `GcpVmHandle` / `GcpStandaloneVmHandle` — SSH via gcloud, `set_labels()` via
  `gcloud compute instances update --update-labels`, `set_metadata()` via
  `gcloud compute instances add-metadata`
- **Test**: Unit tests with mocked gcloud calls — create_slice, list_slices, create_vm, list_vms, label operations.

### Step 4: Manual + Local platforms

All platforms implement the full `Platform` interface — no `NotImplementedError`,
no stubs, no wrapping of old managers.

- `ManualPlatform` in `platform/manual.py`:
  - `create_vm()` — allocates a host from the configured host list, returns a
    handle with SSH to that host. Used by `start_controller()` for manual controllers.
  - `create_slice()` — allocates N hosts from the list, returns ManualSliceHandle
  - `shutdown()` — no-op
- `LocalPlatform` in `platform/local.py` — in-process threads:
  - `create_vm()` — starts an in-process "VM" (e.g., local controller), returns
    `_LocalStandaloneVmHandle` with in-memory label/metadata tracking
  - `create_slice()` — spawns worker threads in ThreadContainer, returns `LocalSliceHandle`
  - `list_vms(labels)` — returns in-process VMs matching labels
  - `shutdown()` — stops ThreadContainer (critical — replaces `LocalVmManager.stop()`)
- **Test**: Unit tests for both. LocalPlatform test verifies `shutdown()` stops threads.
  Use `local_platform` pytest fixture for cleanup.

### Step 5: CoreWeave stub

- `CoreweavePlatform` in `platform/coreweave.py` — raises `NotImplementedError` for all methods
- **Test**: Importing works, calling methods raises correctly.

### Step 6: Migrate controller lifecycle

- Rewrite `controller/lifecycle.py`: `start_controller()`, `stop_controller()`, `reload_controller()` as free functions taking `Platform`
- Controller discovery: use `platform.list_vms(zones, labels={...})` + `vm.set_labels()` + `vm.set_metadata()`
- **Bootstrap/SSH flow**: `start_controller()` receives `IrisClusterConfig`, builds the bootstrap
  script via `build_controller_bootstrap_script(config)`, calls `vm.bootstrap(script)`.
  The VmHandle's SSH connection is internal — the Platform constructs it from `ssh_config`
  passed to `create_platform()`.
- Replace `ClusterManager` class with `connect_cluster()` + `stop_all()` free functions in `cluster/manager.py`
- Delete `ControllerProtocol`, `GcpController`, `ManualController` from `controller_vm.py`
- **Files modified**: `controller/lifecycle.py`, `cluster/manager.py`, `controller_vm.py`
- **Test**: Integration test with LocalPlatform: start/stop controller. CLI `iris cluster start/stop/reload` works.

### Step 7: Migrate autoscaler/scaling groups

- `ScalingGroup.__init__` takes `Platform` + `label_prefix` instead of `VmManagerProtocol`
- `ScalingGroup.scale_up()` calls `platform.create_slice(prepare_slice_config(...))`
- `ScalingGroup.reconcile()` calls `platform.list_slices(zones, labels)`
- `ScalingGroup.scale_down()` calls `slice_handle.terminate()` after stopping WorkerVms
- Rename `ManagedVm` → `WorkerVm` — wraps `VmHandle` instead of `SshConnection`
- **Bootstrap/SSH flow for workers**: WorkerVm receives a `VmHandle` (which already has
  SSH internalized). Bootstrap script is built from `BootstrapConfig` + `TimeoutConfig`
  passed through `TrackedVmFactory` → `WorkerVm.__init__`. The `ssh_config` is no longer
  threaded through WorkerVm — it lives in the Platform, which owns the SSH connections.
  WorkerVm calls `vm_handle.run_command(script)` instead of `ssh_conn.run()`.
- `Autoscaler.shutdown()` refactor: Currently the shutdown sequence is:
  1. Stop all WorkerVm bootstrap threads (per-group, per-slice, per-vm)
  2. Terminate all slices (per-group `terminate_all()`)
  3. Stop VM managers (per-group `_vm_manager.stop()` — this is what cleans up
     LocalVmManager's ThreadContainer)
  After migration, `ScalingGroup` no longer has `_vm_manager`. Steps 1-2 stay the
  same (they operate on WorkerVms and SliceHandles). Step 3 becomes a single
  `platform.shutdown()` call on the Autoscaler, since Platform is shared across
  groups. The Autoscaler holds a `Platform` reference (not each ScalingGroup).
- Update `create_autoscaler()` in `controller/config.py`: take `Platform` instead of
  constructing `VmManagerProtocol` per group.
- Delete `VmManagerProtocol`, `VmGroupProtocol`, `TpuVmManager`, `ManualVmManager`, `LocalVmManager`
- **Files modified**: `controller/scaling_group.py`, `controller/config.py`, `controller/autoscaler.py`,
  `vm/managed_vm.py` (rename + refactor → move to `controller/worker_vm.py`)
- **Test**: Autoscaler tests pass. ScalingGroup scale_up/reconcile/scale_down with mocked Platform.

### Step 8: Proto removals + cleanup

Now that all code uses the new Platform API, remove old proto fields and code.

**Proto removals from `rpc/config.proto`**:
- Remove `zone`, `region`, `default_zones` from `GcpPlatformConfig` (keep only `project_id`)
- Remove `VmType` enum
- Remove flat fields from `ScaleGroupConfig` (now using `slice_template`)

**Code cleanup**:
- Delete `cluster/vm/` directory entirely
- Update all imports project-wide (grep for `cluster.vm.`)
- Update YAML configs in `configs/` for new `ScaleGroupConfig` shape (embedded `slice_template`)
- Update `cluster/config.py`: YAML parsing for new config shape, remove `_VM_TYPE_MAP`/`_normalize_vm_types`/`_validate_vm_types`
- Update `cli/debug.py`: read zones from scale group templates instead of `platform.gcp.zone`
- Update `cli/cluster.py`: use `stop_all()` free function with Platform API
- Update `lib/iris/AGENTS.md`: new directory structure
- Run `scripts/generate_protos.py` to regenerate clean protos
- **Test**: Full test suite passes. `iris cluster start --local` end-to-end.
  CLI `iris cluster stop` cleans up all resources.

## Files Modified

| File | Change | Step |
|------|--------|------|
| `rpc/config.proto` | **MODIFY** — Add new messages (VmConfig, SliceConfig, CoreweaveConfig, slice_template on ScaleGroupConfig) | 1 |
| `rpc/config.proto` | **MODIFY** — Remove old fields (zone/region/default_zones from GcpPlatformConfig, VmType enum, flat ScaleGroupConfig fields) | 8 |
| `cluster/config.py` | **MODIFY** — Remove `_VM_TYPE_MAP`, update YAML parsing, update `make_local_config()` | 8 |
| `cluster/controller/config.py` | **MODIFY** — Take Platform instead of constructing VmManagers | 7 |
| `cli/cluster.py` | **MODIFY** — Use Platform API for cleanup | 8 |
| `cli/debug.py` | **MODIFY** — Read zones from scale group templates | 8 |
| `cluster/platform/__init__.py` | **NEW** — Exports | 2 |
| `cluster/platform/base.py` | **NEW** — Protocol definitions (Platform, SliceHandle, VmHandle, StandaloneVmHandle), status types, exception types | 2 |
| `cluster/platform/bootstrap.py` | **NEW** — Bootstrap scripts (from managed_vm.py + controller_vm.py) | 2 |
| `cluster/platform/ssh.py` | **MOVED** from vm/ssh.py | 2 |
| `cluster/platform/gcp.py` | **NEW** — GcpPlatform, GcpSliceHandle, GcpVmHandle | 3 |
| `cluster/platform/manual.py` | **NEW** — ManualPlatform, ManualSliceHandle | 4 |
| `cluster/platform/local.py` | **NEW** — LocalPlatform with ThreadContainer ownership | 4 |
| `cluster/platform/coreweave.py` | **NEW** — CoreWeave stub | 5 |
| `cluster/controller/lifecycle.py` | **REWRITE** — start/stop/reload as free functions using Platform | 6 |
| `cluster/manager.py` | **REWRITE** — ClusterManager class → `connect_cluster()` + `stop_all()` free functions | 6 |
| `cluster/controller/scaling_group.py` | **MODIFY** — Platform + label_prefix instead of VmManagerProtocol | 7 |
| `cluster/controller/autoscaler.py` | **MODIFY** — `shutdown()` calls `platform.shutdown()` instead of per-group `_vm_manager.stop()` | 7 |
| `cluster/controller/worker_vm.py` | **NEW** (moved from vm/managed_vm.py) — Renamed to WorkerVm, wraps VmHandle | 7 |
| `cluster/vm/` | **DELETE** — entirely replaced by platform/ | 8 |
| `lib/iris/AGENTS.md` | **UPDATE** — new directory structure | 8 |
| `scripts/generate_protos.py` | **RUN** — regenerate after proto changes | 1, 8 |

## Verification

Concrete integration matrix — every path must be tested:

### Unit tests (mocked infrastructure)
1. **GcpPlatform**: create_slice returns SliceHandle, list_slices discovers TPUs, create_vm returns StandaloneVmHandle, list_vms discovers controller by labels, set_labels/set_metadata operations
2. **ManualPlatform**: create_slice wraps static hosts, list_slices filters by labels
3. **LocalPlatform**: create_slice spawns workers, shutdown() stops ThreadContainer, list_slices returns in-process slices
4. **CoreweavePlatform**: all methods raise NotImplementedError

### Integration tests (LocalPlatform, no cloud)
5. **Controller create/discover/reload**: `start_controller()` with LocalPlatform, verify discovery via `list_vms(labels=...)`, `reload_controller()` re-bootstraps
6. **Autoscaler reconcile + adoption**: Autoscaler discovers existing slices via `platform.list_slices()`, adopts them into ScalingGroup state
7. **Autoscaler scale_up/scale_down cycle**: ScalingGroup creates slice, wraps VMs in WorkerVm, terminates on scale down
8. **Autoscaler shutdown**: Verify `platform.shutdown()` stops LocalPlatform worker threads
9. **Local mode end-to-end**: `iris cluster start --local` through new Platform path

### CLI tests
10. **`iris cluster stop`**: `stop_all()` terminates controller + all slices
11. **`iris cluster status`**: reads from Platform API
12. **`cli/debug.py` paths**: zone/project references work with new config shape

### Smoke tests (GCP, if available)
13. **GCP controller create/discover/reload**: Full lifecycle with real GCE VM
14. **GCP TPU slice create/list/terminate**: Real TPU pod lifecycle
15. **Manual host adoption and teardown**: SSH to manual hosts

## Additional Design Concerns

### 1. VmHandle.terminate() is invalid for TPU slice members

**Resolved** in the main protocol definitions above. `VmHandle` has no
`terminate()`, `set_labels()`, or `set_metadata()`. These operations live
on `StandaloneVmHandle` (returned by `create_vm()`), since only standalone
VMs (like the controller) can be independently terminated and labeled. Slice
member VMs (from `SliceHandle.list_vms()`) are plain `VmHandle` — termination
goes through `SliceHandle.terminate()`.

Note: `_discover_controller_vm()` uses `cast(StandaloneVmHandle, ...)` when
retrieving controller VMs from `list_vms()`, since `list_vms()` returns
`list[VmHandle]` but the controller VM was created with `create_vm()` and
actually implements the full `StandaloneVmHandle` interface.

### 2. Thread safety for VmHandle

The current `WorkerVm` (née ManagedVm) creates its own `SshConnection` internally
and owns it exclusively. With `VmHandle`, the connection is owned by the platform
layer and the same handle may be accessed from multiple threads (WorkerVm's lifecycle
thread, health check calls, the autoscaler thread).

**Resolution**: `VmHandle` implementations must be thread-safe for concurrent
`run_command()` calls. For SSH-based handles, this means each `run_command()`
creates a new SSH process (which is what `GcloudSshConnection` already does —
it shells out to `gcloud compute ssh` each time). Document this requirement
on the protocol.

### 3. Error taxonomy for create_slice

The current `TpuVmManager._gcloud_create_tpu()` treats all gcloud failures
the same. But the autoscaler should handle quota exhaustion differently from
transient errors (retry in a different zone vs back off and wait).

**Resolution**: Define platform exception types that the autoscaler can
dispatch on:

```python
class PlatformError(Exception):
    """Base for platform operation failures."""

class QuotaExhaustedError(PlatformError):
    """No capacity in the requested zone. Try another zone or wait."""

class ResourceNotFoundError(PlatformError):
    """The requested resource type/variant doesn't exist."""

class PlatformUnavailableError(PlatformError):
    """Transient platform failure. Retry with backoff."""
```

`GcpPlatform.create_slice()` parses gcloud stderr to classify the error.
The autoscaler's zone rotation logic uses `QuotaExhaustedError` to skip
zones that are full.

### 4. SliceHandle.list_vms() is always a live query

**Resolved** in the protocol definitions above. `SliceHandle.list_vms()` always
queries the cloud for current state (e.g., `gcloud compute tpus describe` for
GCP). The Platform layer is a thin infrastructure layer — no caching, no
stale snapshots. If the cloud repairs/replaces a worker, `list_vms()` reflects it.

The controller layer (ScalingGroup) maintains its own WorkerVm registry for
lifecycle tracking. It does not depend on `list_vms()` being cached — it calls
`list_vms()` during reconcile to discover current state, then updates its own
registry accordingly.

### 5. discovery_preamble coupling

The current design keeps `discovery_preamble` on WorkerVm — a bash snippet
that workers run at bootstrap to discover the controller address via GCP
metadata. This is GCP-specific logic leaking into the controller layer.

**Resolution**: Move discovery_preamble generation into the Platform. The
Platform knows how workers discover the controller:
- GCP: query instance metadata via gcloud
- Manual: controller address is passed directly in config
- CoreWeave: Kubernetes service discovery

```python
class Platform(Protocol):
    ...
    def controller_discovery_script(self, label_prefix: str) -> str:
        """Generate a bash snippet that sets CONTROLLER_ADDRESS.

        Workers run this at bootstrap to discover the controller.
        GCP: queries instance metadata. Manual: returns static address.
        """
        ...
```

WorkerVm's bootstrap script calls this instead of owning the preamble.

### 6. Config migration for embedded SliceConfig

The restructured `ScaleGroupConfig` (embedding `SliceConfig` directly) changes
the YAML config format. Current:

```yaml
scale_groups:
  tpu-group:
    vm_type: TPU_VM
    accelerator_variant: v5litepod-16
    runtime_version: tpu-ubuntu2204-base
    zones: [us-central2-b]
    preemptible: true
    slice_size: 1
    min_slices: 0
    max_slices: 2
```

New:

```yaml
scale_groups:
  tpu-group:
    min_slices: 0
    max_slices: 2
    slice_template:
      accelerator_type: ACCELERATOR_TYPE_TPU
      accelerator_variant: v5litepod-16
      slice_size: 1
      gcp:
        zone: us-central2-b
        runtime_version: tpu-ubuntu2204-base
        preemptible: true
```

This is a breaking config change. All existing YAML configs and documentation
must be updated in the same step. The migration plan Step 8 should explicitly
call out updating all YAML configs in `configs/`, test fixtures, and CI.

### 7. `reboot()` on VmHandle is GCP-specific

Not all platforms support VM reboot as a concept. Manual hosts may not allow
it. CoreWeave pods can be restarted but the semantics differ.

**Resolution**: Make `reboot()` optional — raise `NotImplementedError` on
platforms that don't support it. The WorkerVm lifecycle can fall back to
terminate + recreate the slice if reboot isn't available. Alternatively,
remove `reboot()` from VmHandle entirely and let the autoscaler handle
recovery by replacing the slice.
