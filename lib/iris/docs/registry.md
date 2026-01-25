# Per-Scale-Group Provider Config for Iris

**Issue:** wv-88e8 - iris provider should be per scaling group & controller, not top-level

---

## Goal

- **What**: Per-scale-group provider configuration with typed dataclasses, plus dead code cleanup
- **Why**: Current top-level provider config prevents mixed-provider clusters (e.g., TPU + manual hosts)
- **Scope**: `lib/iris/src/iris/cluster/vm/` — config parsing, provider dataclasses, SSH cleanup

---

## Non-Goals

- **Full registry pattern**: Only 2 providers exist. Add registry infrastructure when a third provider is needed.
- **Removing `_create_manager()` dispatch**: Simple if/elif is appropriate for 2 types.
- **Breaking change to YAML**: Support both top-level and per-group provider config during transition.

---

## Design Overview

### Architecture

1. **Typed provider config dataclasses** (`TpuProviderConfig`, `ManualProviderConfig`) hold provider-specific fields
2. **Per-scale-group `provider:` section** in YAML allows mixed-provider clusters
3. **Existing `_create_manager()` if/elif dispatch** remains — no registry pattern
4. **Backward compatibility**: Top-level `provider:` applies to groups without explicit provider

### Data Flow

```
YAML → load_config() → IrisClusterConfig
                            ↓
                    ScaleGroupSpec (with parsed ProviderConfig)
                            ↓
                    _create_manager() [if/elif dispatch]
                            ↓
                    TpuVmManager | ManualVmManager
```

### Key Decisions

| Decision | Rationale |
|----------|-----------|
| No registry pattern | Only 2 providers; if/elif is 40 lines vs 150+ for registry |
| Typed dataclasses | Type safety for provider-specific fields without runtime overhead |
| Backward compat | Existing configs continue to work; gradual migration |

---

## Files Modified or Created

### Directory Tree

```
lib/iris/src/iris/cluster/vm/
  config.py <modified>
  ssh.py <modified>
lib/iris/examples/
  demo.yaml <modified>
  eu-west4.yaml <modified>
lib/iris/tests/cluster/vm/
  test_config.py <modified>
```

### File-by-File Overview

#### `config.py` (modified)

**Purpose**: Cluster configuration loading and autoscaler creation.

**Changes**:
1. Add `TpuProviderConfig` and `ManualProviderConfig` dataclasses
2. Add `parse_provider_config()` function to dispatch on `type` field
3. Update `load_config()` to parse per-group provider config
4. Update `_create_manager()` to accept typed config instead of kwargs

**Key interfaces**:
```python
@dataclass
class TpuProviderConfig:
    type: Literal["tpu"] = "tpu"
    project_id: str = ""

@dataclass
class ManualProviderConfig:
    type: Literal["manual"] = "manual"
    hosts: list[str] = field(default_factory=list)
    ssh_user: str = "root"
    ssh_key_file: str | None = None
    ssh_port: int = 22

ProviderConfig = TpuProviderConfig | ManualProviderConfig

def parse_provider_config(data: dict) -> ProviderConfig:
    """Parse dict into typed provider config."""
    type_name = data.get("type", "manual")
    if type_name == "tpu":
        return TpuProviderConfig(**data)
    if type_name == "manual":
        return ManualProviderConfig(**data)
    raise ValueError(f"Unknown provider: {type_name}")
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

**Changes**: Add per-group provider format (top-level still works as fallback):

```yaml
# Old format (still supported as fallback)
provider:
  type: tpu
  project_id: hai-gcp-models

# New format (per-group)
scale_groups:
  tpu_v5e_16:
    provider:
      type: tpu
      project_id: hai-gcp-models
    accelerator_type: v5litepod-16

  manual_workers:
    provider:
      type: manual
      hosts: [10.0.0.1, 10.0.0.2]
      ssh_user: ray
```

#### `test_config.py` (modified)

**Changes**: Add tests for provider config parsing and backward compatibility.

---

## Alternatives Considered

### Alternative A: Full Registry Pattern

Config classes register themselves via `ProviderConfig.register()`. Dispatch uses registry lookup.

**Trade-offs**: Adds ~100 lines of infrastructure (Protocol with ClassVar registry, register/parse methods). Only 2 providers exist; unlikely to add more soon. Deferred until third provider needed.

### Alternative B: Union Type Without Dataclasses

Use `dict` with runtime type checking instead of typed dataclasses.

**Trade-offs**: Loses type safety and IDE support. Current kwargs-based approach already has this problem.

---

## Implementation Steps

### Step 1: Delete Dead SSH Factory Code

**Files**: `ssh.py`

**Actions**:
1. Remove `SshConnectionFactory` type alias
2. Remove `GcloudSshConnectionFactory` Protocol
3. Remove `make_direct_ssh_factory()`, `make_in_memory_connection_factory()`, `make_gcloud_ssh_factory()`

**Verify**: `uv run pytest lib/iris/tests/ -v` — all tests pass (code is unused)

### Step 2: Add Provider Config Dataclasses

**Files**: `config.py`

**Actions**:
1. Add `TpuProviderConfig` dataclass with `type`, `project_id` fields
2. Add `ManualProviderConfig` dataclass with `type`, `hosts`, `ssh_*` fields
3. Add `ProviderConfig` type alias as union
4. Add `parse_provider_config()` function

**Verify**: Type checker passes

### Step 3: Update Config Parsing

**Files**: `config.py`

**Actions**:
1. Update `load_config()` to parse `scale_groups[name].provider` into typed config
2. Fall back to top-level `provider:` if group has no explicit provider
3. Store parsed `ProviderConfig` in `ScaleGroupSpec`

**Verify**: `uv run python -c "from iris.cluster.vm.config import load_config; c = load_config('lib/iris/examples/eu-west4.yaml'); print(c)"`

### Step 4: Update Manager Creation

**Files**: `config.py`

**Actions**:
1. Update `_create_manager()` to accept `ProviderConfig` instead of kwargs
2. Use `isinstance()` or match on `config.type` to dispatch

**Verify**: `uv run pytest lib/iris/tests/cluster/vm/test_controller.py -v`

### Step 5: Add Tests

**Files**: `test_config.py`

**Actions**:
1. Test `parse_provider_config()` for tpu, manual, unknown
2. Test per-group provider parsing
3. Test fallback to top-level provider

**Verify**: `uv run pytest lib/iris/tests/cluster/vm/test_config.py -v`

---

## Cross-Cutting Concerns

### Testing Strategy

- **Unit tests**: `parse_provider_config()`, config loading with per-group providers
- **Integration tests**: Existing autoscaler tests continue to pass
- **Manual verification**: Load example YAML files

### Error Handling

- `ValueError` for unknown provider type (lists available: tpu, manual)
- `ValueError` for missing required fields (e.g., `project_id` for TPU, `hosts` for manual)
- Errors propagate to caller

### Migration

Backward compatible:
1. If `scale_groups[name].provider` exists, use it
2. Else if top-level `provider:` exists, use it as default
3. Else raise error (no provider specified)

---

## Open Questions

1. **Remove redundant top-level fields?** With per-group providers, `IrisClusterConfig.project_id`, `manual_hosts`, etc. are redundant. Options:
   - Keep as defaults (current plan)
   - Deprecate with warning
   - Remove (breaking change)

---

## Plan Review

**Potential issues**:
- Union type `TpuProviderConfig | ManualProviderConfig` requires Python 3.10+ or `Union[]`
- `isinstance()` checks in `_create_manager()` are slightly less elegant than registry dispatch, but appropriate for 2 types

**Alignment with guidelines**:
- Follows AGENTS.md preference for simple solutions
- No unnecessary abstraction
- Tests verify behavior
