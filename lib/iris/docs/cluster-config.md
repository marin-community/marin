# Cluster Configuration V2: Platform + Defaults

## Summary

This document proposes a new Iris cluster configuration model that combines the
platform refactor strategy with the requirement that the configuration file be a
complete, canonical description of a cluster (Issue #2569). The design introduces
(1) a top-level `platform` selection that produces a provider-specific Platform
object, (2) typed defaults in `config.proto`, and (3) explicit separation between
platform direct operations (CLI cleanup) and lifecycle-managed VM creation
(autoscaler). The goal is to fix the cluster-stop lifecycle bug while making the
configuration more expressive and reliable.

## Goals

- Make the cluster configuration a complete description of cluster setup:
  controller, worker, autoscaler, and VM lifecycle behavior.
- Unify platform discovery with scale-group provisioning (TPU, GCE, manual, local).
- Provide typed defaults in `config.proto` and a single, predictable override
  path in `config.py`.
- Allow CLI operations (cluster stop/cleanup) to query and delete VMs without
  bootstrapping managed VMs or starting lifecycle threads.
- Support multiple scale-group types in a single cluster under a single
  platform, with per-group VM manager selection and per-group zones.

## Non-goals

- Maintaining backward compatibility with the existing YAML schema or old
  config parsing behavior. Call sites are updated instead.
- Introducing a new orchestration backend beyond the existing GCP/manual/local
  providers (future work).
- Reworking the autoscaler algorithm itself.

## Background and Problem Statement

- The current Iris configuration describes scale groups and VM behavior, but
  does not fully specify the controller and worker configurations. A number of
  non-trivial constants remain hardcoded rather than configurable.
- The cluster stop path incorrectly boots managed VMs and starts lifecycle
  threads, causing a race between bootstrap and termination.
- The platform refactor strategy proposes a `Platform` abstraction to centralize
  provider-specific logic and avoid lifecycle side effects in CLI operations.
- Issue #2569 requests that `config.proto` become the canonical cluster
  description, with typed defaults and a structured override mechanism.

## Current State (Brief)

- `config.proto` includes `IrisClusterConfig`, `ScaleGroupConfig`, and related
  controller/autoscaler/timeout configs, but defaults are not consistently
  expressed or overridable.
- CLI `cluster stop` can create `VmGroup` and `ManagedVm` lifecycle threads,
  which is actively harmful for cleanup.
- Platform logic is distributed across `*_VmManager` implementations, with
  direct ops (gcloud/ssh) interleaved with lifecycle behavior.

## Proposal

### 1) New Configuration Shape

Introduce a platform-first configuration with explicit defaults and a map of
scale groups keyed by name. The design preserves the current conceptual pieces
but makes the platform the root of discovery and allows scale groups to select
VM manager type independently from the platform.

Example YAML (illustrative):

```yaml
platform:
  gcp:
    project_id: hai-gcp-models
    region: europe-west4
    default_zones: [europe-west4-b]
    label_prefix: iris

defaults:
  autoscaler:
    evaluation_interval: { seconds: 10 }
    requesting_timeout: { seconds: 120 }
    scale_up_delay: { seconds: 1000 }
    scale_down_delay: { seconds: 1000 }
  timeouts:
    boot_timeout: { seconds: 300 }
    init_timeout: { seconds: 600 }
    ssh_poll_interval: { seconds: 5 }
  ssh:
    user: root
    port: 22
    connect_timeout: { seconds: 30 }

controller:
  image: europe-west4-docker.pkg.dev/hai-gcp-models/marin/iris-controller:latest
  bundle_prefix: gs://marin-us-central2/tmp/iris/bundles
  gcp:
    machine_type: n2-standard-4
    port: 10000

scale_groups:
  tpu_v5e_16:
    type: tpu_vm
    topology: 4x4
    accelerator_variant: v5litepod-16
    runtime_version: v2-alpha-tpuv5-lite
    min_slices: 1
    max_slices: 2
    zones: [europe-west4-b]
    preemptible: true
    priority: 100
  manual_hosts:
    type: manual_vm
    hosts: [10.0.0.1, 10.0.0.2]
    ssh_user: ubuntu
    ssh_key_file: ~/.ssh/manual_key
```

Notes:
- `platform` is a oneof (gcp/manual/local) and is the root for shared provider
  settings and defaulting.
- `defaults` are strongly typed and apply to nested configs unless overridden
  at the group or controller level.
- `scale_groups[*].type` selects the VM manager type (tpu_vm, gce_vm, manual_vm,
  local_vm). The platform provides the provider-specific capabilities.

### 2) `config.proto` Sketch (proto3 optional + HasField)

We keep `IrisClusterConfig` but restructure it to be platform-first and add a
`DefaultsConfig` message. Presence is critical: defaults only apply when fields
are unset.

```proto
message PlatformConfig {
  oneof platform {
    GcpPlatformConfig gcp = 1;
    ManualPlatformConfig manual = 2;
    LocalPlatformConfig local = 3;
  }
}

message DefaultsConfig {
  TimeoutConfig timeouts = 1;
  SshConfig ssh = 2;
  AutoscalerDefaults autoscaler = 3;
  BootstrapDefaults bootstrap = 4;
}

message AutoscalerDefaults {
  iris.time.Duration evaluation_interval = 1;
  iris.time.Duration requesting_timeout = 2;
  iris.time.Duration scale_up_delay = 3;
  iris.time.Duration scale_down_delay = 4;
}

message ScaleGroupConfig {
  string name = 1;
  string type = 2;  // "tpu_vm", "gce_vm", "manual_vm", "local_vm"
  optional int32 min_slices = 3;
  optional int32 max_slices = 4;
  string accelerator_variant = 5;
  string runtime_version = 6;
  optional bool preemptible = 7;
  repeated string zones = 8;
  optional int32 priority = 9;

  // Provider-specific settings (for manual, etc.)
  ManualProvider manual = 20;
}

message IrisClusterConfig {
  PlatformConfig platform = 1;
  DefaultsConfig defaults = 2;
  ControllerVmConfig controller = 3;
  map<string, ScaleGroupConfig> scale_groups = 4;
  BootstrapConfig bootstrap = 5;
  AutoscalerConfig autoscaler = 6;
}
```

Notes:
- Defaulting is explicit in the schema, not scattered in code. The merge logic
  lives in `config.py` and is applied once at load time.
- Scalar fields that need default/override semantics are declared `optional` and
  we use `HasField(...)` to distinguish unset from explicitly set values.

### 3) Platform Abstraction

Introduce a `Platform` interface that can serve both lifecycle (controller)
paths and direct ops (CLI cleanup) without starting ManagedVm threads.

```python
class Platform(Protocol):
    def vm_ops(self) -> PlatformOps: ...
    def vm_manager(self, vm_type: str, vm_factory: TrackedVmFactory) -> VmManager: ...

class PlatformOps(Protocol):
    def list_slices(self) -> list[str]: ...
    def delete_slice(self, slice_id: str) -> None: ...
```

- `Platform` is constructed from `IrisClusterConfig.platform` and used by all
  scale groups. Group-specific details (zones, labels, topology) are passed into
  `vm_manager(...)` when the autoscaler builds the group.
- `PlatformOps` is intentionally narrow to avoid lifecycle creation. This
  replaces the current CLI pattern of building autoscalers just to list/delete
  slices.
- `VmManager` remains the lifecycle abstraction used only by the autoscaler.

### 4) Scale-Group Types and Platform Interaction

- Scale groups remain responsible for expressing accelerator type, topology,
  runtime version, and min/max slices.
- Provisioning details are split:
  - `Platform` handles provider-specific API access and VM primitives.
  - `ScaleGroupConfig.type` selects the `VmManager` implementation for the
    group via `platform.vm_manager(type, factory)`.
- This allows heterogenous groups: TPU pods and GCE VM pools within one cluster.

### 5) CLI Behavior

- `cluster stop` should attempt an RPC query first (if controller is reachable)
  and fall back to `PlatformOps.list_slices()` to discover resources.
- Deletion uses `PlatformOps.delete_slice()` directly (gcloud delete, SSH kill,
  etc.) without invoking ManagedVm lifecycle.
- No VM bootstrap or autoscaler should be created for CLI cleanup paths.

## Compare/Contrast with the Platform Refactor Strategy

| Aspect | Platform Refactor (current proposal) | This Design |
|---|---|---|
| Primary goal | Fix cluster stop bug | Fix bug + make config canonical |
| Platform scope | Per scale-group factory | Platform-first, root config |
| Defaults | Implicit in code | Typed defaults in proto + merge logic |
| CLI cleanup | Direct ops on Platform | Direct ops via PlatformOps interface |
| Multi-region | Not explicit | Single platform + per-group zones |

The refactor strategy is a good minimal fix, but it does not solve the
configuration completeness requirement. This design folds the platform approach
into a more explicit, typed configuration that can be used across controller,
worker, autoscaler, and CLI flows.

## Recommendations

1. Adopt the platform-first config shape with typed defaults in `config.proto`.
2. Implement `Platform` + `PlatformOps` and use it in both autoscaler creation
   and CLI cleanup paths.
3. Keep `ClusterManager` for now; it can remain a CLI-focused lifecycle helper
   that does not understand providers.
4. Use proto3 `optional` and `HasField(...)` for defaultable scalars.
5. Update docs/examples to reflect the new config shape and remove old YAMLs.

## Risks and Mitigations

- **Large breaking change**: migrate all YAMLs and update config loaders in one
  sweep, remove compatibility shims.
- **Defaulting complexity**: centralize defaults in `config.py` with a single
  merge function that applies `defaults` to all nested fields, keyed on
  `HasField` to preserve explicit overrides.
- **Provider leakage**: keep direct ops in `PlatformOps` only, avoid exposing
  gcloud/ssh helpers from `VmManager`.

## Migration Plan (High-Level)

1. Update `config.proto` and regenerate protos (`scripts/generate_protos.py`).
2. Update config loading in `config.py` to merge defaults and produce a
   normalized `IrisClusterConfig` object.
3. Refactor autoscaler creation to use `Platform.create_vm_manager()`.
4. Update CLI `cluster stop` to use `PlatformOps` fallback.
5. Update docs/examples and README.

## New Configuration and Class Topology

```
IrisClusterConfig
  ├── platform: PlatformConfig
  │     └── create_platform(...) -> Platform
  │           ├── vm_ops() -> PlatformOps
  │           └── vm_manager(vm_type, factory) -> VmManager
  ├── defaults: DefaultsConfig
  ├── controller: ControllerVmConfig
  ├── autoscaler: AutoscalerConfig
  └── scale_groups: map<string, ScaleGroupConfig>

Autoscaler creation
  ├── platform = create_platform(config.platform, config.defaults)
  ├── for group in scale_groups:
  │     manager = platform.vm_manager(group.type, vm_factory)
  │     ScalingGroup(group_config, manager)
  └── Autoscaler(groups)

CLI cleanup
  ├── platform = create_platform(config.platform, config.defaults)
  ├── ops = platform.vm_ops()
  ├── ops.list_slices() -> slice ids
  └── ops.delete_slice(id)
```

## Required Changes (Outline)

- `lib/iris/src/iris/rpc/config.proto`
  - Add `PlatformConfig`, `DefaultsConfig`, `AutoscalerDefaults` (with
    `scale_up_delay`/`scale_down_delay`).
  - Convert scalar override fields to `optional` where defaults apply.
  - Update `IrisClusterConfig` to `platform` + `defaults` shape.
- `lib/iris/src/iris/cluster/vm/config.py`
  - Add `create_platform(config.platform, config.defaults)`.
  - Implement a single `apply_defaults(config)` that merges defaults using
    `HasField(...)` for scalar overrides.
- `lib/iris/src/iris/cluster/vm/platform.py` (new)
  - Implement `Platform` + `PlatformOps` per provider.
  - Move shared direct-ops helpers out of `VmManager` where appropriate.
- `lib/iris/src/iris/cluster/vm/scaling_group.py`
  - Accept `vm_type` or full `ScaleGroupConfig` and call
    `platform.vm_manager(vm_type, factory)`.
- `lib/iris/src/iris/cluster/vm/autoscaler.py` / `config.py`
  - Build `ScalingGroup` via platform+vm_type rather than `_create_manager_from_config`.
- `lib/iris/src/iris/cli/cluster.py`
  - Update `cluster stop` to use `platform.vm_ops()` for list/delete fallback.
- `lib/iris/src/iris/README.md`, `lib/iris/examples/*.yaml`
  - Update docs/examples to new config shape.
- `lib/iris/scripts/generate_protos.py`
  - Regenerate proto outputs after schema changes.

## Spiral Plan (Per AGENTS.md)

Step 1: Minimal, end-to-end platform + config scaffolding
- Add proto messages + optional fields (no behavioral changes yet).
- Regenerate protos.
- Add `Platform` + `PlatformOps` interfaces and a single GCP implementation
  stub that compiles but is not wired into autoscaler.
- Add a unit-style config loader test that asserts `HasField` behavior for
  optional scalars.

Step 2: Wire autoscaler to platform.vm_manager by vm_type
- Update `create_autoscaler_from_config` to call `platform.vm_manager(...)`.
- Update `ScalingGroup` constructor to take `vm_type` (or full group config).
- Add a lightweight integration test with a fake platform + fake vm_manager.

Step 3: CLI cleanup without lifecycle
- Update `cluster stop` to use `platform.vm_ops()` fallback.
- Add a test that ensures cleanup path does not instantiate `ManagedVm`.

Step 4: Defaults merge + documentation
- Implement `apply_defaults` merge using `HasField` for scalar overrides.
- Update README and example YAMLs to new schema.
- Add an end-to-end config round-trip test (load YAML -> proto -> defaults).

## Open Questions

1. Should `PlatformOps` expand beyond list/delete for cleanup and diagnostics,
   or remain intentionally minimal?
2. Do we want a single enum for `vm_type` (tpu_vm, gce_vm, manual_vm) in proto
   rather than a string to avoid typos?
3. Where should `scale_up_delay` / `scale_down_delay` live: defaults only or
   allow per-group overrides with `optional` fields?
