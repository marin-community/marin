# Iris Configuration System: Analysis and Refactoring Plan

## Executive Summary

This document proposes a complete refactoring of Iris configuration to address issues identified in PR #2644:

**Key Changes:**
1. **Flat Factories** - `create_autoscaler(platform, config, groups)` instead of `create_autoscaler_from_config(config)`
2. **Direct Field Access** - `config.defaults.autoscaler` instead of helper functions
3. **Single DEFAULT_CONFIG** - All hardcoded defaults in one constant
4. **Rename vm/controller.py → vm/controller_vm.py** - Avoid confusion with controller/controller.py
5. **Remove Duplicate Proto Fields** - Only `defaults.*`, remove top-level bootstrap/timeouts/ssh/autoscaler

**Design Rationale:**
- **Flat factories** allow platform reuse, easier testing, explicit dependencies
- **Direct field access** is simpler than helpers (config already has defaults applied)
- **Renaming** makes the codebase easier to navigate
- Research shows this pattern scales well (Flask application factory, Kubernetes config patterns)

---

## Research: Configuration Patterns in Large-Scale Systems

### Pattern Analysis

I researched configuration patterns in production systems ([Flask Application Factory](https://flask.palletsprojects.com/en/stable/patterns/appfactories/), [Python Dependency Injection](https://python-dependency-injector.ets-labs.org/introduction/di_in_python.html), [Kubernetes Architecture](https://www.portainer.io/blog/kubernetes-architecture), [Factory Pattern Best Practices](https://dagster.io/blog/python-factory-patterns)):

**Key Patterns Found:**

1. **Factory Functions Over God Objects** ([Flask](https://flask.palletsprojects.com/en/stable/patterns/appfactories/))
   - `create_app(config)` returns configured app
   - Config is passed in, not cached in a wrapper
   - Multiple instances possible with different configs
   - Extensions use deferred binding: `db = SQLAlchemy()` then `db.init_app(app)`

2. **Explicit Constructor Injection** ([Dependency Injector](https://python-dependency-injector.ets-labs.org/introduction/di_in_python.html))
   - Components declare what they need via constructor parameters
   - Dependencies passed explicitly, not looked up
   - "Objects do not create each other anymore. They provide a way to inject the dependencies instead."

3. **Hierarchical Config Extraction** (Kubernetes)
   - Large config broken into domain-specific pieces
   - Each component extracts what it needs from relevant section
   - No component sees the entire config unless necessary

**Key Insight**: Config should be a **pure data structure** (proto). Factory functions **extract** what they need and **pass** it explicitly to components. No caching, no god objects.

### Recommendation: Explicit Configuration Threading

For Iris, use:
- **Factory functions** that take explicit config parameters
- **Pure proto config** - no Python wrapper with caching
- **Extract-and-pass pattern** - Each factory extracts what it needs from proto
- **Explicit dependencies** - Thread config through constructors
- **Helper functions** for common extraction patterns

This is Python-native, explicit, and scales well without framework overhead.

---

## Current State Analysis

### Configuration Flow Problems

```
CURRENT (PROBLEMATIC):

load_config("cluster.yaml")
    ↓
IrisClusterConfig proto (has duplicate fields: bootstrap/defaults.bootstrap, timeouts/defaults.timeouts)
    ↓
create_platform(config: IrisClusterConfig)  ← Takes ENTIRE config
    ↓
GcpPlatform.__init__(config: IrisClusterConfig)  ← Stores ENTIRE config
    ↓
platform.vm_manager(group_config, factory)
    ↓
Creates VmManager with bootstrap/timeouts extracted from stored config
```

**Problems:**
1. Platform stores entire config when it only needs platform + bootstrap + timeouts
2. Duplicate fields (bootstrap vs defaults.bootstrap) create ambiguity
3. Hardcoded defaults scattered across multiple files
4. No clear extraction pattern - each module re-implements default resolution

### Current Class Dependencies

```python
# Current (implicit, scattered)
class GcpPlatform:
    def __init__(self, config: IrisClusterConfig):  # Takes EVERYTHING
        self._platform = config.platform.gcp
        self._bootstrap = config.bootstrap  # Which one? Top-level or defaults?
        self._timeouts = config.timeouts
        self._label_prefix = config.platform.label_prefix or "iris"  # Default here

class Autoscaler:
    def __init__(
        self,
        scale_groups: dict[str, ScalingGroup],
        vm_registry: VmRegistry,
        config: AutoscalerConfig | None = None,  # Nullable with hardcoded defaults
    ):
        if config is None:
            config = AutoscalerConfig()  # Empty proto
        if not config.HasField("evaluation_interval"):
            config.evaluation_interval = Duration.from_seconds(10).to_proto()  # Hardcoded
        # ... more hardcoded defaults

class ScalingGroup:
    def __init__(
        self,
        config: ScaleGroupConfig,
        vm_manager: VmManagerProtocol,
        scale_up_cooldown: Duration = DEFAULT_SCALE_UP_COOLDOWN,  # Module constant
        scale_down_cooldown: Duration = DEFAULT_SCALE_DOWN_COOLDOWN,
        # ... 5 more optional params with defaults
    ):
```

**Problems:**
1. Too many optional parameters with module-level defaults
2. Autoscaler has hardcoded fallbacks instead of using config
3. Platform needs entire IrisClusterConfig
4. Unclear where defaults come from

---

## Proposed Solution: Explicit Configuration Threading

### Design Decision: Flat Factories vs Extraction-Inside-Factory

**Option A: Extraction Inside Factory (Rejected)**
```python
# One-line creation, factory does extraction
autoscaler = create_autoscaler_from_config(config, dry_run=False)
```

**Option B: Flat Factories (Chosen)**
```python
# Caller extracts, factory takes explicit dependencies
platform = create_platform(
    platform_config=config.platform,
    bootstrap_config=config.defaults.bootstrap,
    timeout_config=config.defaults.timeouts,
    ssh_config=config.defaults.ssh,
)

autoscaler = create_autoscaler(
    platform=platform,
    autoscaler_config=config.defaults.autoscaler,
    scale_groups=config.scale_groups,
)
```

**Why Flat Factories?**

1. **Single Responsibility** - Each factory creates one component, doesn't extract config
2. **Platform Reuse** - Create platform once, use for autoscaler, controller VM, debug tools
3. **Testability** - Easy to inject test platform: `create_autoscaler(mock_platform, test_config, {})`
4. **Explicit Dependencies** - Reader sees that autoscaler needs platform
5. **Composition** - Natural dependency order visible at call site

The tradeoff is verbosity at call sites, but this happens in ~3 places (CLI startup, controller bootstrap, tests), so the benefits outweigh the cost.

### Core Principles

1. **Config is Pure Data** - `IrisClusterConfig` proto is the single source of truth (after `apply_defaults()`)
2. **Flat Factories** - Each factory creates one thing from explicit dependencies (not mega-factories that extract everything)
3. **Direct Field Access** - Use `config.defaults.autoscaler` directly, no helper functions
4. **Explicit Dependencies** - Caller extracts from config and passes to factories
5. **Single Defaults Constant** - All hardcoded defaults in one place
6. **Clear Naming** - Rename `vm/controller.py` → `vm/controller_vm.py` to avoid confusion

### Phase 1: Clean Proto Schema

**Remove duplicate fields:**

```protobuf
message IrisClusterConfig {
  PlatformConfig platform = 10;
  DefaultsConfig defaults = 11;  // Single source of defaults
  ControllerVmConfig controller = 31;
  map<string, ScaleGroupConfig> scale_groups = 50;

  // REMOVE these duplicate fields:
  // BootstrapConfig bootstrap = 80;
  // TimeoutConfig timeouts = 81;
  // SshConfig ssh = 82;
  // AutoscalerConfig autoscaler = 83;
}
```

**Consolidate in DefaultsConfig:**

```protobuf
message DefaultsConfig {
  TimeoutConfig timeouts = 1;
  SshConfig ssh = 2;
  AutoscalerDefaults autoscaler = 3;
  BootstrapConfig bootstrap = 4;
}
```

### Phase 2: Single Defaults Constant

All hardcoded defaults in one place:

```python
# config.py

# Single source of truth for all default values
DEFAULT_CONFIG = config_pb2.DefaultsConfig(
    timeouts=config_pb2.TimeoutConfig(
        boot_timeout=Duration.from_seconds(300).to_proto(),
        init_timeout=Duration.from_seconds(600).to_proto(),
        ssh_poll_interval=Duration.from_seconds(5).to_proto(),
    ),
    ssh=config_pb2.SshConfig(
        user="root",
        port=22,
        connect_timeout=Duration.from_seconds(30).to_proto(),
    ),
    autoscaler=config_pb2.AutoscalerDefaults(
        evaluation_interval=Duration.from_seconds(10).to_proto(),
        requesting_timeout=Duration.from_seconds(120).to_proto(),
        scale_up_delay=Duration.from_seconds(60).to_proto(),
        scale_down_delay=Duration.from_seconds(300).to_proto(),
    ),
    bootstrap=config_pb2.BootstrapConfig(
        worker_port=10001,
        cache_dir="/var/cache/iris",
    ),
)


def apply_defaults(config: config_pb2.IrisClusterConfig) -> config_pb2.IrisClusterConfig:
    """Apply defaults to config and return merged result.

    Resolution order:
    1. Explicit field in config.defaults.* (if set)
    2. DEFAULT_CONFIG constant (hardcoded defaults)

    This function is called once during load_config().
    """
    merged = config_pb2.IrisClusterConfig()
    merged.CopyFrom(config)

    # Start with DEFAULT_CONFIG, then overlay user-provided defaults
    result_defaults = config_pb2.DefaultsConfig()
    result_defaults.CopyFrom(DEFAULT_CONFIG)

    # Merge each section
    if config.HasField("defaults"):
        _deep_merge_defaults(result_defaults, config.defaults)

    merged.defaults.CopyFrom(result_defaults)

    # Apply scale group defaults
    for group in merged.scale_groups.values():
        if not group.HasField("priority"):
            group.priority = 100

    return merged


def _deep_merge_defaults(target: config_pb2.DefaultsConfig, source: config_pb2.DefaultsConfig) -> None:
    """Deep merge source defaults into target, field by field."""
    if source.HasField("timeouts"):
        _merge_proto_fields(target.timeouts, source.timeouts)
    if source.HasField("ssh"):
        _merge_proto_fields(target.ssh, source.ssh)
    if source.HasField("autoscaler"):
        _merge_proto_fields(target.autoscaler, source.autoscaler)
    if source.HasField("bootstrap"):
        _merge_proto_fields(target.bootstrap, source.bootstrap)


def _merge_proto_fields(target, source) -> None:
    """Merge non-empty fields from source into target."""
    for field in source.DESCRIPTOR.fields:
        if source.HasField(field.name):
            value = getattr(source, field.name)
            # For Duration, check milliseconds > 0
            if hasattr(value, "milliseconds") and value.milliseconds == 0:
                continue
            # For string, check non-empty
            if isinstance(value, str) and not value:
                continue
            # For int, check non-zero (except port which can be 0)
            if isinstance(value, int) and value == 0 and field.name != "port":
                continue
            getattr(target, field.name).CopyFrom(value) if hasattr(value, "CopyFrom") else setattr(
                target, field.name, value
            )
```

### Phase 3: Explicit Platform Factory

Platform should receive **only** what it needs:

```python
# platform.py

def create_platform(
    platform_config: config_pb2.PlatformConfig,
    bootstrap_config: config_pb2.BootstrapConfig,
    timeout_config: config_pb2.TimeoutConfig,
    ssh_config: config_pb2.SshConfig,
) -> Platform:
    """Create platform from explicit config sections.

    Args:
        platform_config: Platform type and settings (gcp/manual/local)
        bootstrap_config: Worker bootstrap settings
        timeout_config: VM lifecycle timeouts
        ssh_config: SSH connection settings

    Returns:
        Platform instance for the configured platform type

    Raises:
        ValueError: If platform type is unspecified or invalid
    """
    if not platform_config.HasField("platform"):
        raise ValueError("platform is required")

    which = platform_config.WhichOneof("platform")

    if which == "gcp":
        if not platform_config.gcp.project_id:
            raise ValueError("platform.gcp.project_id is required")
        return GcpPlatform(
            gcp_config=platform_config.gcp,
            label_prefix=platform_config.label_prefix or "iris",
            bootstrap_config=bootstrap_config,
            timeout_config=timeout_config,
        )

    if which == "manual":
        return ManualPlatform(
            label_prefix=platform_config.label_prefix or "iris",
            bootstrap_config=bootstrap_config,
            timeout_config=timeout_config,
            ssh_config=ssh_config,
        )

    if which == "local":
        return LocalPlatform()

    raise ValueError(f"Unknown platform: {which}")


class GcpPlatform:
    """GCP platform for TPU and GCE VM management."""

    def __init__(
        self,
        gcp_config: config_pb2.GcpPlatformConfig,
        label_prefix: str,
        bootstrap_config: config_pb2.BootstrapConfig,
        timeout_config: config_pb2.TimeoutConfig,
    ):
        """Create GCP platform with explicit config sections.

        No defaults here - all defaults resolved before this point.
        """
        self._gcp = gcp_config
        self._label_prefix = label_prefix
        self._bootstrap = bootstrap_config
        self._timeouts = timeout_config

    def vm_ops(self) -> PlatformOps:
        return _GcpPlatformOps(self._gcp, self._label_prefix)

    def vm_manager(
        self,
        group_config: config_pb2.ScaleGroupConfig,
        vm_factory: TrackedVmFactory,
        *,
        dry_run: bool = False,
    ) -> VmManagerProtocol:
        if group_config.vm_type == config_pb2.VM_TYPE_TPU_VM:
            # Resolve zones once
            zones = _resolve_zones(group_config, self._gcp)

            return TpuVmManager(
                project_id=self._gcp.project_id,
                group_config=group_config,
                zones=zones,
                bootstrap_config=self._bootstrap,
                timeout_config=self._timeouts,
                vm_factory=vm_factory,
                label_prefix=self._label_prefix,
                dry_run=dry_run,
            )

        if group_config.vm_type == config_pb2.VM_TYPE_GCE_VM:
            raise NotImplementedError("GCE VMs not yet implemented")

        raise ValueError(f"Unsupported vm_type for GCP: {group_config.vm_type}")


class ManualPlatform:
    """Manual platform for pre-existing hosts."""

    def __init__(
        self,
        label_prefix: str,
        bootstrap_config: config_pb2.BootstrapConfig,
        timeout_config: config_pb2.TimeoutConfig,
        ssh_config: config_pb2.SshConfig,
    ):
        self._label_prefix = label_prefix
        self._bootstrap = bootstrap_config
        self._timeouts = timeout_config
        self._ssh = ssh_config

    # ... similar structure
```

### Phase 4: Flat Autoscaler Factory

Autoscaler receives platform and config, creates scale groups:

```python
# config.py

def create_autoscaler(
    platform: Platform,
    autoscaler_config: config_pb2.AutoscalerDefaults,
    scale_groups: dict[str, config_pb2.ScaleGroupConfig],
    dry_run: bool = False,
) -> Autoscaler:
    """Create autoscaler from platform and explicit config.

    Args:
        platform: Platform instance for creating VM managers
        autoscaler_config: Autoscaler settings (already resolved with defaults)
        scale_groups: Map of scale group name to config
        dry_run: If True, don't actually provision VMs

    Returns:
        Configured Autoscaler instance
    """
    # Create shared infrastructure
    vm_registry = VmRegistry()
    vm_factory = TrackedVmFactory(vm_registry)

    # Extract autoscaler settings from config
    scale_up_delay = Duration.from_proto(autoscaler_config.scale_up_delay)
    scale_down_delay = Duration.from_proto(autoscaler_config.scale_down_delay)
    evaluation_interval = Duration.from_proto(autoscaler_config.evaluation_interval)
    requesting_timeout = Duration.from_proto(autoscaler_config.requesting_timeout)

    # Create scale groups using provided platform
    scaling_groups: dict[str, ScalingGroup] = {}
    for name, group_config in scale_groups.items():
        vm_manager = platform.vm_manager(group_config, vm_factory=vm_factory, dry_run=dry_run)

        scaling_groups[name] = ScalingGroup(
            config=group_config,
            vm_manager=vm_manager,
            scale_up_cooldown=scale_up_delay,
            scale_down_cooldown=scale_down_delay,
        )
        logger.info("Created scale group %s", name)

    # Create autoscaler with explicit parameters
    return Autoscaler(
        scale_groups=scaling_groups,
        vm_registry=vm_registry,
        evaluation_interval=evaluation_interval,
        requesting_timeout=requesting_timeout,
    )


# Usage example:
config = load_config("cluster.yaml")  # Defaults already applied

# Create platform once
platform = create_platform(
    platform_config=config.platform,
    bootstrap_config=config.defaults.bootstrap,
    timeout_config=config.defaults.timeouts,
    ssh_config=config.defaults.ssh,
)

# Create autoscaler using platform
autoscaler = create_autoscaler(
    platform=platform,
    autoscaler_config=config.defaults.autoscaler,
    scale_groups=config.scale_groups,
    dry_run=False,
)


# autoscaler.py

class Autoscaler:
    """Manages scaling across scale groups.

    The autoscaler:
    - Receives demand from a DemandSource
    - Evaluates scaling decisions based on demand vs capacity
    - Executes decisions by calling ScalingGroup.scale_up/scale_down
    - Reports status via VmRegistry
    """

    def __init__(
        self,
        scale_groups: dict[str, ScalingGroup],
        vm_registry: VmRegistry,
        evaluation_interval: Duration,
        requesting_timeout: Duration,
        threads: ThreadContainer | None = None,
    ):
        """Create autoscaler with explicit parameters.

        No defaults, no nullable config - all parameters required.

        Args:
            scale_groups: Map of scale group name to ScalingGroup instance
            vm_registry: Shared VM registry for tracking all VMs
            evaluation_interval: How often to evaluate scaling decisions
            requesting_timeout: How long to wait for VMs to provision before timing out
            threads: Optional thread container for testing
        """
        self._groups = scale_groups
        self._vm_registry = vm_registry
        self._evaluation_interval = evaluation_interval
        self._requesting_timeout = requesting_timeout

        # Track slice creation times
        self._slice_created_at: dict[str, int] = {}

        # Action log for dashboard
        self._action_log: deque[vm_pb2.AutoscalerAction] = deque(maxlen=100)

        # Thread management
        self._threads = threads if threads is not None else get_thread_container()

    # ... rest of implementation
```

### Phase 5: Explicit ScalingGroup Constructor

ScalingGroup receives exactly what it needs, no module-level defaults:

```python
# scaling_group.py

class ScalingGroup:
    """Manages a group of VMs for a specific accelerator type."""

    def __init__(
        self,
        config: config_pb2.ScaleGroupConfig,
        vm_manager: VmManagerProtocol,
        scale_up_cooldown: Duration,
        scale_down_cooldown: Duration,
        backoff_initial: Duration | None = None,
        backoff_max: Duration | None = None,
        backoff_factor: float = 2.0,
        idle_threshold: Duration | None = None,
        quota_timeout: Duration | None = None,
    ):
        """Create scaling group with explicit parameters.

        Args:
            config: Scale group configuration (min/max slices, accelerator type, etc.)
            vm_manager: VM manager for this group (platform-specific)
            scale_up_cooldown: Minimum time between scale-ups
            scale_down_cooldown: Minimum time between scale-downs
            backoff_initial: Initial backoff on failure (default: scale_up_cooldown)
            backoff_max: Maximum backoff (default: 5 * backoff_initial)
            backoff_factor: Backoff multiplier on repeated failures
            idle_threshold: How long a slice must be idle before scale-down (default: scale_down_cooldown)
            quota_timeout: How long to wait before retrying after quota error (default: scale_down_cooldown)
        """
        self._config = config
        self._vm_manager = vm_manager
        self._vm_groups: dict[str, VmGroupProtocol] = {}
        self._vm_groups_lock = threading.Lock()

        # Demand tracking
        self._current_demand: int = 0
        self._peak_demand: int = 0
        self._slice_last_active: dict[str, Timestamp] = {}

        # Rate limiting
        self._scale_up_cooldown = scale_up_cooldown
        self._scale_down_cooldown = scale_down_cooldown
        self._last_scale_up: Timestamp = Timestamp.from_ms(0)
        self._last_scale_down: Timestamp = Timestamp.from_ms(0)

        # Backoff (use scale_up_cooldown as default initial backoff)
        self._backoff_initial = backoff_initial or scale_up_cooldown
        self._backoff_max = backoff_max or Duration.from_seconds(self._backoff_initial.seconds * 5)
        self._backoff_factor = backoff_factor
        self._consecutive_failures: int = 0
        self._backoff_until: Timestamp = Timestamp.from_ms(0)

        # Idle threshold (use scale_down_cooldown as default)
        self._idle_threshold = idle_threshold or scale_down_cooldown

        # Quota timeout (use scale_down_cooldown as default)
        self._quota_timeout = quota_timeout or scale_down_cooldown
        self._quota_exceeded_until: Timestamp = Timestamp.from_ms(0)
        self._quota_reason: str = ""

        # Requesting state
        self._requesting_until: Timestamp = Timestamp.from_ms(0)
```

### Phase 6: Rename controller.py → controller_vm.py and Flat Factories

**RENAME FIRST** to avoid confusion:
- `cluster/vm/controller.py` → `cluster/vm/controller_vm.py`
- Update all imports

Then implement flat factories:

```python
# controller/controller.py (unchanged - the actual Controller service)

@dataclass
class ControllerConfig:
    """Controller configuration - all required fields."""

    bundle_prefix: str
    """URI prefix for job bundles (required)."""

    host: str = "127.0.0.1"
    """Host to bind HTTP server."""

    port: int = 0
    """Port to bind HTTP server (0 = auto-assign)."""

    scheduler_interval: Duration = field(default_factory=lambda: Duration.from_seconds(0.5))
    """Scheduling loop interval."""

    worker_timeout: Duration = field(default_factory=lambda: Duration.from_seconds(60.0))
    """Worker heartbeat timeout."""

    max_dispatch_parallelism: int = 32
    """Max concurrent RPC dispatches."""


class Controller:
    """Unified controller managing scheduling, workers, and autoscaling."""

    def __init__(
        self,
        config: ControllerConfig,
        worker_stub_factory: WorkerStubFactory,
        autoscaler: Autoscaler | None = None,
        threads: ThreadContainer | None = None,
    ):
        """Create controller with explicit dependencies.

        Args:
            config: Controller configuration (all defaults resolved)
            worker_stub_factory: Factory for worker RPC stubs
            autoscaler: Optional autoscaler for VM management
            threads: Optional thread container for testing
        """
        if not config.bundle_prefix:
            raise ValueError("bundle_prefix is required in ControllerConfig")

        self._config = config
        self._stub_factory = worker_stub_factory
        self._autoscaler = autoscaler

        # Create internal components
        self._state = ControllerState()
        self._scheduler = Scheduler(self._state)
        self._service = ControllerServiceImpl(
            self._state,
            self,
            bundle_prefix=config.bundle_prefix,
            log_buffer=get_global_buffer(),
        )
        self._dashboard = ControllerDashboard(
            self._service,
            host=config.host,
            port=config.port,
        )

        # ... rest of initialization


# vm/controller_vm.py (RENAMED - manages controller VM lifecycle)

def create_controller_vm(
    controller_vm_config: config_pb2.ControllerVmConfig,
    platform_config: config_pb2.PlatformConfig,
    bootstrap_config: config_pb2.BootstrapConfig,
) -> ControllerVmProtocol:
    """Create controller VM manager from explicit config sections.

    This creates the VM MANAGER (GcpControllerVm, ManualControllerVm, LocalControllerVm),
    NOT the Controller service itself.

    Args:
        controller_vm_config: Controller VM configuration
        platform_config: Platform configuration (for project_id, region, etc.)
        bootstrap_config: Bootstrap configuration for workers

    Returns:
        ControllerVmProtocol implementation for managing controller VM lifecycle
    """
    if not controller_vm_config.HasField("controller"):
        raise ValueError("controller configuration is required")

    which = controller_vm_config.WhichOneof("controller")

    if which == "gcp":
        if not platform_config.HasField("gcp"):
            raise ValueError("GCP controller requires platform.gcp configuration")

        return GcpControllerVm(
            project_id=platform_config.gcp.project_id,
            region=platform_config.gcp.region,
            zone=platform_config.gcp.zone,
            label_prefix=platform_config.label_prefix or "iris",
            controller_config=controller_vm_config.gcp,
            controller_image=controller_vm_config.image,
            bundle_prefix=controller_vm_config.bundle_prefix,
            bootstrap_config=bootstrap_config,
        )

    if which == "manual":
        return ManualControllerVm(
            host=controller_vm_config.manual.host,
            port=controller_vm_config.manual.port or 10000,
        )

    if which == "local":
        return LocalControllerVm(
            port=controller_vm_config.local.port,
            bundle_prefix=controller_vm_config.bundle_prefix,
        )

    raise ValueError(f"Unknown controller type: {which}")


# Usage example (in CLI or main.py):
config = load_config("cluster.yaml")

# Create controller VM manager (for VM lifecycle)
controller_vm = create_controller_vm(
    controller_vm_config=config.controller,
    platform_config=config.platform,
    bootstrap_config=config.defaults.bootstrap,
)

# Start the controller VM
controller_url = controller_vm.start()

# Inside the controller container, create the Controller service:
autoscaler = create_autoscaler(platform, config.defaults.autoscaler, config.scale_groups)

controller = Controller(
    config=ControllerConfig(bundle_prefix=config.controller.bundle_prefix),
    worker_stub_factory=RpcWorkerStubFactory(),
    autoscaler=autoscaler,
)
```

---

## Complete Initialization Flow

### Full Controller VM Bootstrap (GCP Example)

```
User:
  uv run iris cluster --config=cluster.yaml start

CLI (cli/cluster.py):
  config = load_config("cluster.yaml")
    ↓
  load_config():
    1. Load YAML
    2. Normalize enums
    3. ParseDict → IrisClusterConfig proto
    4. apply_defaults(config) → merge with DEFAULT_CONFIG
    5. Validate vm_types and accelerator_types
    ↓
  Returns: IrisClusterConfig (defaults fully resolved)

  # Caller extracts from config and calls flat factory
  controller_vm = create_controller_vm(
    controller_vm_config=config.controller,  # Direct field access
    platform_config=config.platform,
    bootstrap_config=config.defaults.bootstrap,
  )
    ↓
  create_controller_vm() in vm/controller_vm.py:
    which = controller_vm_config.WhichOneof("controller")  # "gcp"
    ↓
    return GcpControllerVm(
      project_id=platform_config.gcp.project_id,
      region=platform_config.gcp.region,
      zone=platform_config.gcp.zone,
      label_prefix=platform_config.label_prefix or "iris",
      controller_config=controller_vm_config.gcp,
      controller_image=controller_vm_config.image,
      bundle_prefix=controller_vm_config.bundle_prefix,
      bootstrap_config=bootstrap_config,
    )

  controller_vm.start()
    ↓
  GcpControllerVm.start():
    1. Create GCE VM via gcloud
    2. Wait for SSH
    3. Install Docker
    4. Write bootstrap script with:
       - BUNDLE_PREFIX
       - CONTROLLER_ADDRESS (internal IP)
       - AUTOSCALER_CONFIG (as JSON)
    5. Run: docker run iris-controller:latest controller serve
    ↓
  Returns: controller_url

  Inside Controller Container:
    controller serve (cluster/controller/main.py)
      ↓
    load_config_from_env():
      - Read AUTOSCALER_CONFIG JSON from env
      - ParseDict → IrisClusterConfig
      ↓
    # Caller extracts from config and creates dependencies
    config = load_config_from_env()

    # Step 1: Create platform (once, reusable)
    platform = create_platform(
      platform_config=config.platform,  # Direct field access
      bootstrap_config=config.defaults.bootstrap,
      timeout_config=config.defaults.timeouts,
      ssh_config=config.defaults.ssh,
    )
      ↓
    create_platform() in vm/platform.py:
      which = platform_config.WhichOneof("platform")  # "gcp"
      ↓
      return GcpPlatform(
        gcp_config=platform_config.gcp,
        label_prefix=platform_config.label_prefix or "iris",
        bootstrap_config=bootstrap_config,
        timeout_config=timeout_config,
      )

    # Step 2: Create autoscaler using platform
    autoscaler = create_autoscaler(
      platform=platform,  # Pass created platform
      autoscaler_config=config.defaults.autoscaler,  # Direct field access
      scale_groups=config.scale_groups,  # Direct field access
      dry_run=False,
    )
      ↓
    create_autoscaler() in vm/config.py:
      1. Create shared infrastructure:
         vm_registry = VmRegistry()
         vm_factory = TrackedVmFactory(vm_registry)

      2. Extract autoscaler settings from autoscaler_config:
         scale_up_delay = Duration.from_proto(autoscaler_config.scale_up_delay)
         scale_down_delay = Duration.from_proto(autoscaler_config.scale_down_delay)
         evaluation_interval = Duration.from_proto(autoscaler_config.evaluation_interval)
         requesting_timeout = Duration.from_proto(autoscaler_config.requesting_timeout)

      3. Create scale groups using platform:
         for name, group_config in scale_groups.items():
           vm_manager = platform.vm_manager(
             group_config,
             vm_factory=vm_factory,
             dry_run=False,
           )
           ↓
         platform.vm_manager():
           if group_config.vm_type == VM_TYPE_TPU_VM:
             zones = _resolve_zones(group_config, self._gcp)
             return TpuVmManager(
               project_id=self._gcp.project_id,
               group_config=group_config,
               zones=zones,
               bootstrap_config=self._bootstrap,
               timeout_config=self._timeouts,
               vm_factory=vm_factory,
               label_prefix=self._label_prefix,
               dry_run=False,
             )

           scaling_groups[name] = ScalingGroup(
             config=group_config,
             vm_manager=vm_manager,
             scale_up_cooldown=scale_up_delay,
             scale_down_cooldown=scale_down_delay,
           )

      4. Return autoscaler:
         return Autoscaler(
           scale_groups=scaling_groups,
           vm_registry=vm_registry,
           evaluation_interval=evaluation_interval,
           requesting_timeout=requesting_timeout,
         )

    # Step 3: Create controller service
    controller_config = ControllerConfig(
      bundle_prefix=config.controller.bundle_prefix,  # Direct field access
      port=10000,
      scheduler_interval=Duration.from_seconds(0.5),
      worker_timeout=Duration.from_seconds(60),
    )

    controller = Controller(
      config=controller_config,
      worker_stub_factory=RpcWorkerStubFactory(),
      autoscaler=autoscaler,  # Pass created autoscaler
    )

    controller.start()
      ↓
    Controller.start():
      1. Start HTTP server (dashboard + RPC)
      2. Start scheduling loop:
         - Read demand from jobs
         - Call scheduler.schedule()
         - Call autoscaler.evaluate(demand_source)
         - Send heartbeats to workers
      3. Autoscaler provisions VMs as needed
```

### Configuration Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      cluster.yaml (User Input)                   │
│  platform:                                                       │
│    gcp:                                                          │
│      project_id: my-project                                      │
│      region: us-central1                                         │
│  defaults:                                                       │
│    autoscaler:                                                   │
│      scale_up_delay: {milliseconds: 120000}  # Override default │
│    bootstrap:                                                    │
│      docker_image: gcr.io/my-project/iris-worker:v1              │
│  controller:                                                     │
│    bundle_prefix: gs://my-bucket/iris/bundles                    │
│    gcp:                                                          │
│      machine_type: n2-standard-4                                 │
│  scale_groups:                                                   │
│    tpu_v5e_4:                                                    │
│      vm_type: tpu_vm                                             │
│      accelerator_type: tpu                                       │
│      accelerator_variant: v5litepod-4                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               load_config() + apply_defaults()                   │
│                                                                  │
│  Merges user config with DEFAULT_CONFIG:                        │
│                                                                  │
│  defaults.autoscaler.scale_up_delay = 120s (from user)          │
│  defaults.autoscaler.scale_down_delay = 300s (from DEFAULT)     │
│  defaults.timeouts.boot_timeout = 300s (from DEFAULT)           │
│  defaults.ssh.user = "root" (from DEFAULT)                      │
│  defaults.bootstrap.worker_port = 10001 (from DEFAULT)          │
│  defaults.bootstrap.docker_image = gcr.io/... (from user)       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│            IrisClusterConfig (Fully Resolved Proto)              │
│  platform.gcp.project_id = "my-project"                          │
│  platform.label_prefix = "iris"                                  │
│  defaults.autoscaler.scale_up_delay.milliseconds = 120000        │
│  defaults.bootstrap.docker_image = "gcr.io/..."                  │
│  defaults.timeouts.boot_timeout.milliseconds = 300000            │
│  controller.bundle_prefix = "gs://my-bucket/iris/bundles"        │
│  scale_groups["tpu_v5e_4"].vm_type = VM_TYPE_TPU_VM              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ├──────────────────────────────────┐
                              │                                  │
                              ▼                                  ▼
┌──────────────────────────────────────────────────────────────┐
│   Caller extracts from config (direct field access):         │
│                                                               │
│   platform = create_platform(                                │
│     platform_config=config.platform,                         │
│     bootstrap_config=config.defaults.bootstrap,              │
│     timeout_config=config.defaults.timeouts,                 │
│     ssh_config=config.defaults.ssh,                          │
│   )                                                           │
│     ↓                                                         │
│   GcpPlatform(                                               │
│     gcp_config=config.platform.gcp,                          │
│     label_prefix="iris",                                     │
│     bootstrap_config=config.defaults.bootstrap,              │
│     timeout_config=config.defaults.timeouts,                 │
│   )                                                           │
└──────────────────────────────────────────────────────────────┘
              │
              ├───────────────────────────────────────────┐
              │                                           │
              ▼                                           ▼
┌─────────────────────────────────┐  ┌──────────────────────────────────┐
│   create_controller_vm(...)      │  │   create_autoscaler(...)         │
│                                  │  │                                  │
│   Caller extracts:               │  │   Caller passes:                 │
│   - config.controller            │  │   - platform (already created)   │
│   - config.platform              │  │   - config.defaults.autoscaler   │
│   - config.defaults.bootstrap    │  │   - config.scale_groups          │
│                                  │  │                                  │
│   GcpControllerVm(               │  │   Uses platform to create        │
│     project_id="my-project",     │  │   vm_managers for each group     │
│     controller_config=...,       │  │                                  │
│     bootstrap_config=...,        │  │                                  │
│   )                              │  │                                  │
└─────────────────────────────────┘  └──────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────────────────────────────────┐
│   platform.vm_manager(group_config, vm_factory)                 │
│                                                                 │
│   Uses platform's stored:                                      │
│   - self._bootstrap (bootstrap_config)                         │
│   - self._timeouts (timeout_config)                            │
│                                                                 │
│   Creates:                                                     │
│   TpuVmManager(                                                │
│     project_id=self._gcp.project_id,                           │
│     group_config=group_config,  # Just this group's config     │
│     zones=_resolve_zones(group_config, self._gcp),             │
│     bootstrap_config=self._bootstrap,  # From platform         │
│     timeout_config=self._timeouts,     # From platform         │
│     vm_factory=vm_factory,                                     │
│     label_prefix=self._label_prefix,                           │
│   )                                                            │
└────────────────────────────────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────────────────────────────────┐
│   ScalingGroup(                                                │
│     config=group_config,        # Just this group's config     │
│     vm_manager=vm_manager,      # Platform-created manager     │
│     scale_up_cooldown=120s,     # Extracted from defaults      │
│     scale_down_cooldown=300s,   # Extracted from defaults      │
│   )                                                            │
└────────────────────────────────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────────────────────────────────┐
│   Autoscaler(                                                  │
│     scale_groups={...},                                        │
│     vm_registry=vm_registry,                                   │
│     evaluation_interval=10s,    # Extracted from defaults      │
│     requesting_timeout=120s,    # Extracted from defaults      │
│   )                                                            │
└────────────────────────────────────────────────────────────────┘
              │
              ▼
┌────────────────────────────────────────────────────────────────┐
│   Controller(                                                  │
│     config=ControllerConfig(                                   │
│       bundle_prefix="gs://my-bucket/iris/bundles",             │
│       port=10000,                                              │
│     ),                                                         │
│     worker_stub_factory=RpcWorkerStubFactory(),                │
│     autoscaler=autoscaler,                                     │
│   )                                                            │
└────────────────────────────────────────────────────────────────┘
```

### Key Points

1. **Config is Pure Data**: `IrisClusterConfig` proto is just data, no methods
2. **Single Defaults Pass**: `apply_defaults()` runs once in `load_config()`
3. **Explicit Extraction**: Each factory extracts exactly what it needs
4. **Thread Configuration Through**: Config pieces passed explicitly to constructors
5. **No Caching**: Factories create fresh instances, don't cache them
6. **Platform Abstraction**: Platform stores bootstrap/timeouts and creates managers on demand
7. **Required Parameters**: Components use required params, not nullable configs with defaults

---

## Migration Path

### Step 0: Rename files for clarity (non-breaking but do first)
- **Rename** `cluster/vm/controller.py` → `cluster/vm/controller_vm.py`
  - Update all imports: `from cluster.vm.controller_vm import GcpControllerVm`
  - Update factory function name: `create_controller()` → `create_controller_vm()`
- **Optional**: Rename `cluster/worker/worker.py` → `cluster/worker/worker_vm.py` for consistency
- Test that all imports work

### Step 1: Add DEFAULT_CONFIG constant (non-breaking)
- Add `DEFAULT_CONFIG` in `config.py`
- Update `apply_defaults()` to use it instead of inline defaults
- Test that defaults are correctly applied

### Step 2: Update factory signatures to flat pattern (non-breaking)
- Add new explicit factory signatures alongside existing ones
- Mark old signatures as deprecated
- Update tests to use new signatures

### Step 3: Update Platform classes (non-breaking)
- Add new Platform.__init__ signatures with explicit params
- Keep old signature working via adapter
- Update create_platform() to use new signature

### Step 4: Update CLI (non-breaking)
- Update CLI commands to use new factory signatures
- Remove direct proto field access, use extraction helpers
- Test all CLI commands

### Step 5: Clean proto schema (breaking)
- Remove duplicate top-level fields from `IrisClusterConfig`
- Update examples to use only `defaults.*`
- Regenerate protos

### Step 6: Remove deprecated code (breaking)
- Remove old factory signatures
- Remove adapter code
- Update all tests

---

## Benefits

1. **Explicit Over Implicit**: Every dependency is visible in constructors
2. **No God Objects**: No config wrapper that knows everything
3. **Single Source of Truth**: `DEFAULT_CONFIG` for all hardcoded defaults
4. **Easy Testing**: Create components with explicit params, no config file needed
5. **Clear Ownership**: Platform owns bootstrap/timeouts, factories extract what they need
6. **No Magic**: Configuration flows explicitly through constructors
7. **Pythonic**: Uses standard patterns (factory functions, dataclasses), no DI framework

---

## Alternative: Protocol-Based Config Access

### The Suggestion

Instead of threading individual config pieces, use Protocols to define config interfaces:

```python
class PlatformConfigProvider(Protocol):
    """Protocol for objects that provide platform configuration."""

    @property
    def platform(self) -> config_pb2.PlatformConfig: ...

    @property
    def bootstrap(self) -> config_pb2.BootstrapConfig: ...

    @property
    def timeouts(self) -> config_pb2.TimeoutConfig: ...

    @property
    def ssh(self) -> config_pb2.SshConfig: ...


def create_platform(config: PlatformConfigProvider) -> Platform:
    """Create platform from config provider."""
    which = config.platform.WhichOneof("platform")

    if which == "gcp":
        return GcpPlatform(
            gcp_config=config.platform.gcp,
            label_prefix=config.platform.label_prefix or "iris",
            bootstrap_config=config.bootstrap,
            timeout_config=config.timeouts,
        )
    # ...


# IrisClusterConfig (after defaults applied) implements PlatformConfigProvider automatically
# through structural subtyping (duck typing)
```

### Comparison: Explicit Params vs Protocols

#### Approach 1: Explicit Parameters (Proposed in Document)

```python
# Factory signature
def create_platform(
    platform_config: config_pb2.PlatformConfig,
    bootstrap_config: config_pb2.BootstrapConfig,
    timeout_config: config_pb2.TimeoutConfig,
    ssh_config: config_pb2.SshConfig,
) -> Platform:
    ...

# Usage
platform = create_platform(
    platform_config=config.platform,
    bootstrap_config=config.defaults.bootstrap,
    timeout_config=config.defaults.timeouts,
    ssh_config=config.defaults.ssh,
)
```

**Pros:**
- **Maximum explicitness**: Every dependency is a separate parameter
- **No coupling to config structure**: Factory doesn't know about parent config object
- **Easy to test**: Can pass mock protos directly without creating full config
- **No protocols needed**: Uses standard Python types (proto messages)
- **Clear at call site**: You see exactly what's being passed
- **Refactoring friendly**: If you need fewer fields, remove parameters

**Cons:**
- **Verbose**: Many parameters for complex components
- **Repetitive**: Similar extraction patterns in multiple places
- **Boilerplate**: Each factory needs to extract the same fields from config

#### Approach 2: Protocol-Based (Alternative)

```python
# Protocol definition
class PlatformConfigProvider(Protocol):
    @property
    def platform(self) -> config_pb2.PlatformConfig: ...
    @property
    def bootstrap(self) -> config_pb2.BootstrapConfig: ...
    @property
    def timeouts(self) -> config_pb2.TimeoutConfig: ...
    @property
    def ssh(self) -> config_pb2.SshConfig: ...

# Factory signature
def create_platform(config: PlatformConfigProvider) -> Platform:
    ...

# Usage (IrisClusterConfig implements protocol implicitly)
platform = create_platform(config)

# Testing with custom provider
@dataclass
class TestPlatformConfig:
    platform: config_pb2.PlatformConfig
    bootstrap: config_pb2.BootstrapConfig
    timeouts: config_pb2.TimeoutConfig
    ssh: config_pb2.SshConfig

test_config = TestPlatformConfig(...)
platform = create_platform(test_config)
```

**Pros:**
- **Concise call sites**: Just pass config object
- **Type-safe**: Protocol defines exactly what's needed
- **Flexible implementation**: Any object with matching properties works
- **Grouped dependencies**: Related config pieces stay together
- **Less repetition**: Don't repeat field access in every factory call

**Cons:**
- **Less explicit**: Dependencies hidden behind protocol
- **Coupling to structure**: Protocol defines a specific config shape
- **Indirection**: Reader must check protocol to understand dependencies
- **More types**: Need to define protocols for each component
- **Can encourage god objects**: Easy to add more properties to protocol over time

### Hybrid Approach: Protocols for Grouping, Explicit for Core

Use protocols for naturally grouped config, explicit params for independent values:

```python
class PlatformConfigProvider(Protocol):
    """Config needed by platform - naturally cohesive group."""

    @property
    def platform(self) -> config_pb2.PlatformConfig: ...

    @property
    def bootstrap(self) -> config_pb2.BootstrapConfig: ...

    @property
    def timeouts(self) -> config_pb2.TimeoutConfig: ...

    @property
    def ssh(self) -> config_pb2.SshConfig: ...


def create_autoscaler_from_config(
    config: PlatformConfigProvider,  # Protocol for platform-related config
    scale_up_delay: Duration,  # Explicit for autoscaler-specific config
    scale_down_delay: Duration,
    evaluation_interval: Duration,
    requesting_timeout: Duration,
    scale_groups_config: dict[str, config_pb2.ScaleGroupConfig],
) -> Autoscaler:
    """Create autoscaler with hybrid approach."""
    # Create platform using protocol
    platform = create_platform(config)

    # Use explicit params for autoscaler settings
    # ...
```

### Recommendation

**For Iris, use explicit parameters** because:

1. **Clarity over conciseness**: Config initialization happens once, explicitness is valuable
2. **Avoid accidental coupling**: Don't want components to "know about" the full config structure
3. **Easier refactoring**: Adding/removing config fields doesn't require protocol changes
4. **Simpler testing**: Can create minimal config pieces without implementing protocols
5. **Matches factory pattern**: Factories take explicit inputs, produce specific outputs

**When to consider protocols:**
- If the same 4+ config fields are always passed together to multiple factories
- If you're creating a config "view" for a specific domain (e.g., `VmBootstrapConfig` protocol)
- If you want to support alternative config sources (e.g., environment variables)

**Example where protocols might help:**

```python
# If you find yourself repeatedly passing these exact 4 fields together:
create_platform(bootstrap, timeouts, ssh, platform)
create_vm_manager(bootstrap, timeouts, ssh, ...)
create_worker_factory(bootstrap, timeouts, ssh, ...)

# Then a protocol makes sense:
class VmLifecycleConfig(Protocol):
    @property
    def bootstrap(self) -> BootstrapConfig: ...
    @property
    def timeouts(self) -> TimeoutConfig: ...
    @property
    def ssh(self) -> SshConfig: ...

# But for Iris, these fields are NOT always used together:
# - Platform needs bootstrap + timeouts + ssh + platform_config
# - Autoscaler needs bootstrap + timeouts + autoscaler_config
# - Controller needs just bundle_prefix
# So explicit params are clearer.
```

---

## Files to Change

### Step 0: Renaming (do first)
- **RENAME** `src/iris/cluster/vm/controller.py` → `src/iris/cluster/vm/controller_vm.py`
  - Rename `create_controller()` → `create_controller_vm()`
  - Update all imports in:
    - `src/iris/cli/cluster.py`
    - `src/iris/cluster/vm/cluster_manager.py`
    - `tests/cluster/vm/test_controller.py`
    - Any other files importing from `cluster.vm.controller`

### Proto
- `src/iris/rpc/config.proto` - Remove duplicate fields (bootstrap, timeouts, ssh, autoscaler from top-level)

### Core Config
- `src/iris/cluster/vm/config.py`
  - Add `DEFAULT_CONFIG` constant
  - Update `apply_defaults()` to use deep merge with `DEFAULT_CONFIG`
  - **REPLACE** `create_autoscaler_from_config(config)` → `create_autoscaler(platform, autoscaler_config, scale_groups)`
  - Update `create_platform()` signature to take explicit params (not full config)

### Platform
- `src/iris/cluster/vm/platform.py`
  - Update `create_platform()` to take explicit params:
    ```python
    def create_platform(
        platform_config: PlatformConfig,
        bootstrap_config: BootstrapConfig,
        timeout_config: TimeoutConfig,
        ssh_config: SshConfig,
    ) -> Platform
    ```
  - Update `GcpPlatform.__init__()` to take explicit params
  - Update `ManualPlatform.__init__()` to take explicit params

### Autoscaler
- `src/iris/cluster/vm/autoscaler.py`
  - Update `Autoscaler.__init__()` to take explicit params (no nullable config)
  - Remove hardcoded defaults from constructor (lines 185-194)

### ScalingGroup
- `src/iris/cluster/vm/scaling_group.py`
  - Ensure `ScalingGroup.__init__()` uses explicit params for cooldowns (already does this)

### Controller VM
- `src/iris/cluster/vm/controller_vm.py` (RENAMED)
  - Update `create_controller_vm()` to take explicit params:
    ```python
    def create_controller_vm(
        controller_vm_config: ControllerVmConfig,
        platform_config: PlatformConfig,
        bootstrap_config: BootstrapConfig,
    ) -> ControllerVmProtocol
    ```

### Controller Service
- `src/iris/cluster/controller/controller.py` - Ensure `ControllerConfig` is explicit (already is)

### CLI
- `src/iris/cli/cluster.py`
  - Update to use flat factories with direct field access
  - Extract from config: `config.platform`, `config.defaults.bootstrap`, etc.
  - Call `create_controller_vm(config.controller, config.platform, config.defaults.bootstrap)`

- `src/iris/cli/main.py`
  - Update tunnel logic to use `config.platform` directly

- `src/iris/cli/run.py`
  - Update job submission to use `config.platform.gcp` directly

- `src/iris/cli/debug.py`
  - Update debug commands to use `config.platform` directly

### Tests
- `tests/cluster/vm/test_config.py`
  - Test new flat factory signatures
  - Test `DEFAULT_CONFIG` merging

- `tests/cluster/vm/test_controller.py`
  - Update imports: `from cluster.vm.controller_vm import ...`

- `tests/cluster/test_e2e.py`
  - Update to use flat factory pattern

### Examples
- `examples/eu-west4.yaml` - Update to use only `defaults.*` fields (remove top-level bootstrap, timeouts, ssh, autoscaler)
