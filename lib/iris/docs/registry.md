# Per-Scale-Group Provider Config for Iris

**Issue:** wv-88e8 - iris provider should be per scaling group & controller, not top-level

---

## Goal

- **What**: Per-scale-group provider configuration via protobuf, plus dead code cleanup
- **Why**: Current top-level provider config prevents mixed-provider clusters (e.g., TPU + manual hosts in same cluster)
- **Scope**: `lib/iris/src/iris/rpc/vm.proto`, `lib/iris/src/iris/cluster/vm/config.py`, SSH cleanup

---

## Non-Goals

- **Python-side registry pattern**: Config is protobuf-first; no separate Python dataclasses for provider config
- **Removing `_create_manager()` dispatch**: Simple if/elif is appropriate for 2 provider types

---

## Design Overview

### Current State

Config is now protobuf-based:
- `IrisClusterConfig` proto holds top-level `provider_type`, `project_id`, `manual_hosts`
- `ScaleGroupConfig` proto has no provider info
- `ScaleGroupSpec` Python dataclass wraps proto with `provider` and `hosts` fields
- `load_config()` uses `ParseDict()` to parse YAML → proto

### Proposed Change

Add `ProviderConfig` message to proto with `oneof` for provider-specific fields:

```protobuf
message ProviderConfig {
  oneof provider {
    TpuProvider tpu = 1;
    ManualProvider manual = 2;
  }
}

message TpuProvider {
  string project_id = 1;
}

message ManualProvider {
  repeated string hosts = 1;
  string ssh_user = 2;
  string ssh_key_file = 3;
  int32 ssh_port = 4;
}
```

Then add `ProviderConfig provider = 90;` to `ScaleGroupConfig`.

### Data Flow

```
YAML → ParseDict() → IrisClusterConfig (proto)
                           ↓
         scale_groups[name].provider (ProviderConfig)
                           ↓
         _create_manager() [if/elif on provider.WhichOneof()]
                           ↓
         TpuVmManager | ManualVmManager
```

### Key Decisions

| Decision | Rationale |
|----------|-----------|
| Proto-first | Config already uses `ParseDict()`; keep single source of truth |
| `oneof` for providers | Type-safe, extensible, auto-generates Python accessors |
| Keep `_create_manager()` | Simple if/elif for 2 providers; add registry when third appears |
| Remove top-level provider | Per AGENTS.md: no backward compat; update all call sites |

---

## Files Modified or Created

### Directory Tree

```
lib/iris/src/iris/
  rpc/
    vm.proto <modified>
  cluster/vm/
    config.py <modified>
    ssh.py <modified>
lib/iris/examples/
  demo.yaml <modified>
  eu-west4.yaml <modified>
lib/iris/tests/cluster/vm/
  test_config.py <modified>
```

### File-by-File Overview

#### `vm.proto` (modified)

**Purpose**: Protobuf definitions for VM management.

**Changes**:
1. Add `TpuProvider` message with `project_id`
2. Add `ManualProvider` message with `hosts`, `ssh_user`, `ssh_key_file`, `ssh_port`
3. Add `ProviderConfig` message with `oneof provider`
4. Add `ProviderConfig provider = 90` field to `ScaleGroupConfig`

**Key interfaces**:
```protobuf
message TpuProvider {
  string project_id = 1;
}

message ManualProvider {
  repeated string hosts = 1;
  string ssh_user = 2;           // Default: "root"
  string ssh_key_file = 3;
  int32 ssh_port = 4;            // Default: 22
}

message ProviderConfig {
  oneof provider {
    TpuProvider tpu = 1;
    ManualProvider manual = 2;
  }
}

message ScaleGroupConfig {
  // ... existing fields ...

  // Per-group provider config (overrides top-level if set)
  ProviderConfig provider = 90;
}
```

#### `config.py` (modified)

**Purpose**: Cluster configuration loading and autoscaler creation.

**Changes**:
1. Update `create_autoscaler_from_config()` to read `group_config.provider`
2. Extract provider-specific fields from proto messages
3. Remove `ScaleGroupSpec` dataclass (no longer needed - proto has all info)

**Key logic**:
```python
def _get_provider_config(
    group_config: vm_pb2.ScaleGroupConfig,
) -> tuple[str, str | None, list[str] | None]:
    """Extract provider type and config from scale group.

    Returns: (provider_type, project_id, hosts)
    Raises: ValueError if provider not set
    """
    if not group_config.HasField("provider"):
        raise ValueError(f"Scale group {group_config.name} missing provider config")

    provider = group_config.provider
    which = provider.WhichOneof("provider")
    if which == "tpu":
        return ("tpu", provider.tpu.project_id, None)
    if which == "manual":
        return ("manual", None, list(provider.manual.hosts))

    raise ValueError(f"Unknown provider type in {group_config.name}")
```

#### `ssh.py` (modified)

**Purpose**: SSH connection implementations.

**Changes**: Delete unused factory code:
- `SshConnectionFactory` type alias
- `GcloudSshConnectionFactory` Protocol
- `make_direct_ssh_factory()`
- `make_in_memory_connection_factory()`
- `make_gcloud_ssh_factory()`

**Keep**: `SshConnection` Protocol and all connection classes.

#### `examples/demo.yaml`, `examples/eu-west4.yaml` (modified)

**Changes**: Move provider config into each scale group, remove top-level provider fields:

```yaml
scale_groups:
  tpu_v5e_16:
    provider:
      tpu:
        project_id: hai-gcp-models
    accelerator_type: v5litepod-16
    zones: [europe-west4-b]

  manual_workers:
    provider:
      manual:
        hosts: [10.0.0.1, 10.0.0.2]
        ssh_user: ray
    accelerator_type: cpu
    zones: [local]
```

#### `test_config.py` (modified)

**Changes**: Add tests for per-group provider parsing.

---

## Alternatives Considered

### Alternative A: Python Dataclasses for Provider Config

Add `TpuProviderConfig` and `ManualProviderConfig` Python dataclasses alongside proto.

**Trade-offs**: Duplicates type definitions. Config already uses `ParseDict()` which handles YAML → proto directly. Adding Python dataclasses creates two sources of truth.

### Alternative B: Keep Provider in ScaleGroupSpec Only

Keep `ScaleGroupSpec` Python dataclass, don't modify proto.

**Trade-offs**: Provider config isn't serializable. Can't round-trip config to YAML. `ScaleGroupSpec` becomes the "real" config instead of the proto.

---

## Implementation Steps

### Step 1: Delete Dead SSH Factory Code

**Files**: `ssh.py`

**Actions**:
1. Remove `SshConnectionFactory` type alias
2. Remove `GcloudSshConnectionFactory` Protocol
3. Remove `make_direct_ssh_factory()`, `make_in_memory_connection_factory()`, `make_gcloud_ssh_factory()`

**Verify**: `uv run pytest lib/iris/tests/ -v` — all tests pass

### Step 2: Add Provider Messages to Proto

**Files**: `vm.proto`

**Actions**:
1. Add `TpuProvider` message
2. Add `ManualProvider` message
3. Add `ProviderConfig` message with `oneof`
4. Add `provider` field to `ScaleGroupConfig`

**Verify**: `uv run python -c "from iris.rpc import vm_pb2; print(vm_pb2.ProviderConfig.DESCRIPTOR.fields_by_name)"`

### Step 3: Update Config Parsing

**Files**: `config.py`

**Actions**:
1. Add `_get_provider_config()` helper to extract provider from group
2. Update `create_autoscaler_from_config()` to use new helper
3. Remove `ScaleGroupSpec` dataclass
4. Remove top-level `provider_type`, `project_id`, `manual_hosts` handling

**Verify**: Load example YAML with per-group providers

### Step 4: Update Example Configs

**Files**: `examples/demo.yaml`, `examples/eu-west4.yaml`

**Actions**:
1. Move provider config into each scale group
2. Remove top-level `provider_type`, `project_id`, `manual_hosts`

**Verify**: `uv run python -c "from iris.cluster.vm.config import load_config; print(load_config('lib/iris/examples/eu-west4.yaml'))"`

### Step 5: Update All Call Sites

**Files**: Any code creating `IrisClusterConfig` or `ScaleGroupConfig` directly

**Actions**:
1. Search for usages of `provider_type`, `project_id`, `manual_hosts` on `IrisClusterConfig`
2. Update to use per-group `provider` field instead
3. Update any tests that use old config format

**Verify**: `uv run pytest lib/iris/tests/ -v`

### Step 6: Add Tests

**Files**: `test_config.py`

**Actions**:
1. Test per-group TPU provider parsing
2. Test per-group manual provider parsing
3. Test mixed providers in same cluster
4. Test error when provider missing

**Verify**: `uv run pytest lib/iris/tests/cluster/vm/test_config.py -v`

---

## Cross-Cutting Concerns

### Testing Strategy

- **Unit tests**: Provider parsing, error cases
- **Integration tests**: Full autoscaler creation with mixed providers
- **Proto tests**: Verify `WhichOneof()` works correctly

### Error Handling

- `ValueError` if scale group missing `provider` field
- `ValueError` if TPU provider missing `project_id`
- `ValueError` if manual provider missing `hosts`

---

## Open Questions

1. **Remove `ScaleGroupSpec`?** With provider in proto, `ScaleGroupSpec` is only needed for `create_autoscaler_from_specs()`. Options:
   - Keep for programmatic API
   - Remove and have callers build proto directly

2. **Remove top-level provider fields from proto?** Fields `provider_type`, `project_id`, `manual_hosts` on `IrisClusterConfig` become unused. Options:
   - Remove from proto (clean)
   - Leave in proto but ignore (avoid proto version bump)

---

## Plan Review

**Potential issues**:
- Proto `oneof` requires checking `WhichOneof()` at runtime — slightly more verbose than Python union types
- Need to regenerate proto stubs after modifying `vm.proto`

**Alignment with guidelines**:
- Protobuf-first matches existing `load_config()` pattern
- Simple if/elif dispatch for 2 providers
- No unnecessary Python abstractions
