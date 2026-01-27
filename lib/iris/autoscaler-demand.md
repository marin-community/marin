# Fix Autoscaler Demand Routing for CPU-only Jobs

## Overview

### The Bug
When a job is submitted with `ResourceSpec(cpu=1, memory="512m")` (no accelerator), the autoscaler fails to match it to any scale group:

```
Demand overflow: 1/1 slices for accelerator_type= unmet (tried 0 groups)
```

### Root Cause
The code conflates `None` and `""` (empty string) with opposite semantics:

| Location | Behavior |
|----------|----------|
| `compute_demand_entries()` | `get_device_variant()` returns `None` for CPU jobs, then `accelerator_type or ""` converts it to `""` |
| `matches_requirements()` | Treats `None` as "any accelerator" (returns True), but `""` as "exact match to empty string" |

**Result**: CPU jobs create `DemandEntry(accelerator_type="")`. The matching logic checks `group.accelerator_type == ""`, which fails for TPU groups (`"v5litepod-16"`).

### The Fix: Explicit Device Type in Config and Demand

1. **Add `accelerator_type` enum to `ScaleGroupConfig`** - Groups declare their device type (CPU/GPU/TPU)
2. **Rename current `accelerator_type` → `accelerator_variant`** - The specific variant (e.g., "v5litepod-16")
3. **Introduce `DeviceType` enum** - Type-safe matching in Python code
4. **Handle "any variant" semantics** - `device_variant=None` matches any group of the same type

---

## Behavior: CPU Demand Policy

**CPU-only jobs can scale ANY group.** This is intentional.

All VMs have CPUs, so a CPU-only job can run on any scale group. The autoscaler routes CPU demand to groups by priority order (lower priority value = higher preference).

**Operator guidance:**
- Set lower `priority` values on cheaper/CPU-focused groups to prefer them for CPU-only work
- TPU/GPU groups with higher priority values will only receive CPU jobs when lower-priority groups are at capacity
- If you want to exclude a group from CPU demand entirely, this requires a future `cpu_demand_eligible` config field (not in this PR)

---

## Detailed Changes

### File 1: `src/iris/rpc/config.proto`

**Add `AcceleratorType` enum and update `ScaleGroupConfig`**

```protobuf
// Device/accelerator type for scale groups
enum AcceleratorType {
  ACCELERATOR_TYPE_UNSPECIFIED = 0;
  ACCELERATOR_TYPE_CPU = 1;
  ACCELERATOR_TYPE_GPU = 2;
  ACCELERATOR_TYPE_TPU = 3;
}

message ScaleGroupConfig {
  string name = 1;
  int32 min_slices = 2;
  int32 max_slices = 3;

  // Device configuration
  AcceleratorType accelerator_type = 9;   // NEW: cpu, gpu, or tpu
  string accelerator_variant = 10;        // RENAMED from accelerator_type: e.g., "v5litepod-16", "A100"
  string runtime_version = 11;
  bool preemptible = 12;

  // ... rest unchanged
}
```

### File 2: `examples/demo.yaml` and `examples/eu-west4.yaml`

**Update YAML configs to use new field names**

```yaml
# demo.yaml
scale_groups:
  tpu_v5e_16:
    provider:
      tpu:
        project_id: hai-gcp-models
    accelerator_type: tpu              # NEW
    accelerator_variant: v5litepod-16  # RENAMED
    runtime_version: v2-alpha-tpuv5-lite
    min_slices: 1
    max_slices: 2
    zones: [europe-west4-b]
    preemptible: true
```

```yaml
# eu-west4.yaml
scale_groups:
  tpu_v5e_16:
    provider:
      tpu:
        project_id: hai-gcp-models
    accelerator_type: tpu              # NEW
    accelerator_variant: v5litepod-16  # RENAMED
    runtime_version: v2-alpha-tpuv5-lite
    min_slices: 0
    max_slices: 2
    zones: [europe-west4-b]
    preemptible: true
    priority: 100
```

---

### File 3: `src/iris/cluster/controller/state.py`

**Add `DeviceType` enum**

```python
from enum import Enum

class DeviceType(Enum):
    """Device type for demand routing."""
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"


def get_device_type_enum(device: cluster_pb2.DeviceConfig) -> DeviceType:
    """Extract device type as enum from config."""
    if device.HasField("gpu"):
        return DeviceType.GPU
    elif device.HasField("tpu"):
        return DeviceType.TPU
    return DeviceType.CPU
```

---

### File 4: `src/iris/cluster/vm/autoscaler.py`

**Update `DemandEntry` dataclass**

```python
from iris.cluster.controller.state import DeviceType

@dataclass
class DemandEntry:
    """A demand entry specifying resource requirements and count."""
    device_type: DeviceType = DeviceType.CPU
    device_variant: str | None = None  # None = any variant of this type
    count: int = 0
    total_cpu: int = 0
    total_memory_bytes: int = 0
```

**Update `route_demand()`**

```python
for entry in demand_entries:
    matching = [g for g in groups if g.matches_device_requirement(entry.device_type, entry.device_variant)]
    matching.sort(key=lambda g: g.config.priority or 100)
    # ... rest unchanged
```

---

### File 5: `src/iris/cluster/vm/scaling_group.py`

**Replace `matches_requirements()` with `matches_device_requirement()`**

```python
from iris.cluster.controller.state import DeviceType
from iris.rpc import config_pb2

def matches_device_requirement(self, device_type: DeviceType, device_variant: str | None) -> bool:
    """Check if this group can satisfy the given device requirements.

    Matching rules:
    - CPU demand: matches ANY group (all VMs have CPUs)
    - GPU/TPU with variant=None: matches any group of the same device type
    - GPU/TPU with specific variant: requires exact variant match
    """
    if device_type == DeviceType.CPU:
        return True  # CPU jobs can run on ANY group

    # Check device type matches
    group_type = self._get_device_type()
    if group_type != device_type:
        return False

    # None variant = any group of this type; specific variant = exact match
    if device_variant is None:
        return True
    return self._config.accelerator_variant == device_variant


def _get_device_type(self) -> DeviceType:
    """Get device type from config."""
    accel = self._config.accelerator_type
    if accel == config_pb2.ACCELERATOR_TYPE_GPU:
        return DeviceType.GPU
    elif accel == config_pb2.ACCELERATOR_TYPE_TPU:
        return DeviceType.TPU
    return DeviceType.CPU
```

---

### File 6: `src/iris/cluster/controller/controller.py`

**Update `compute_demand_entries()`**

```python
def compute_demand_entries(state: ControllerState) -> list[DemandEntry]:
    """Compute demand entries from controller state."""
    from iris.cluster.vm.autoscaler import DemandEntry
    from iris.cluster.controller.state import DeviceType, get_device_type_enum, get_device_variant

    @dataclass
    class DemandAccumulator:
        count: int = 0
        total_cpu: int = 0
        total_memory_bytes: int = 0

    demand_by_device: dict[tuple[DeviceType, str | None], DemandAccumulator] = {}

    for task in state.peek_pending_tasks():
        job = state.get_job(task.job_id)
        if not job:
            continue

        device = job.request.resources.device
        device_type = get_device_type_enum(device)
        device_variant = get_device_variant(device) if device_type != DeviceType.CPU else None

        key = (device_type, device_variant)
        if key not in demand_by_device:
            demand_by_device[key] = DemandAccumulator()

        acc = demand_by_device[key]
        acc.count += 1
        acc.total_cpu += job.request.resources.cpu
        acc.total_memory_bytes += job.request.resources.memory_bytes

    return [
        DemandEntry(
            device_type=dt,
            device_variant=variant,
            count=acc.count,
            total_cpu=acc.total_cpu,
            total_memory_bytes=acc.total_memory_bytes,
        )
        for (dt, variant), acc in demand_by_device.items()
    ]
```

---

## Testing Plan

---

## Migration Checklist (Repo-wide Touch Points)

Renaming `accelerator_type` (variant string) to `accelerator_variant` and adding the new enum field requires a repo-wide migration. Based on current code usage, these areas must be updated:

### Core code paths

- `src/iris/cluster/vm/gcp_tpu_platform.py` (uses `accelerator_type` as TPU variant for topology/flags)
- `src/iris/cluster/vm/config.py` (config examples and parsing expectations)
- `src/iris/cluster/controller/dashboard.py` (status display strings)
- `src/iris/cli.py` (CLI output fields)
- `src/iris/cluster/vm/scaling_group.py` (matching logic and type decoding)
- `src/iris/cluster/vm/autoscaler.py` (demand entry schema and routing)
- `src/iris/cluster/controller/controller.py` (demand computation)

### Examples, docs, and scripts

- `examples/demo.yaml`
- `examples/eu-west4.yaml`
- `examples/demo_cluster.py`
- `docs/boot-log.md`
- `README.md`
- `scripts/cluster-tools.py`

### Tests (broad updates required)

- `tests/cluster/vm/test_autoscaler.py`
- `tests/cluster/vm/test_scaling_group.py`
- `tests/cluster/vm/test_vm_platform.py`
- `tests/cluster/vm/test_controller.py`
- `tests/cluster/vm/test_controller_boot.py`
- `tests/cluster/vm/test_config.py`
- `tests/cluster/controller/test_dashboard.py`
- `tests/cli/test_cli.py`

### Generated protobufs

- Regenerate `config_pb2.py` / `config_pb2.pyi` / `vm_pb2.pyi` after `config.proto` changes.
- Update any code that assumes `ScaleGroupConfig.accelerator_type` is a string.

---

### New Tests

**`tests/cluster/controller/test_state.py`** - DeviceType enum extraction

```python
def test_cpu_is_default_for_empty_device():
    device = cluster_pb2.DeviceConfig()
    assert get_device_type_enum(device) == DeviceType.CPU

def test_tpu_device_returns_tpu():
    device = cluster_pb2.DeviceConfig(tpu=cluster_pb2.TpuDevice(variant="v5litepod-16"))
    assert get_device_type_enum(device) == DeviceType.TPU
```

**`tests/cluster/vm/test_scaling_group.py`** - Device requirement matching

```python
def test_cpu_matches_any_scale_group():
    """DeviceType.CPU matches any scale group."""
    config = config_pb2.ScaleGroupConfig(
        name="tpu",
        accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
        accelerator_variant="v5litepod-16",
        max_slices=5,
    )
    group = ScalingGroup(config, make_mock_vm_manager())
    assert group.matches_device_requirement(DeviceType.CPU, None)

def test_tpu_none_variant_matches_any_tpu_group():
    """DeviceType.TPU with variant=None matches any TPU group."""
    config = config_pb2.ScaleGroupConfig(
        name="tpu",
        accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
        accelerator_variant="v5litepod-16",
        max_slices=5,
    )
    group = ScalingGroup(config, make_mock_vm_manager())
    assert group.matches_device_requirement(DeviceType.TPU, None)
    assert group.matches_device_requirement(DeviceType.TPU, "v5litepod-16")
    assert not group.matches_device_requirement(DeviceType.TPU, "v5litepod-8")

def test_gpu_does_not_match_tpu_group():
    """DeviceType.GPU does not match TPU group even with None variant."""
    config = config_pb2.ScaleGroupConfig(
        name="tpu",
        accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
        accelerator_variant="v5litepod-16",
        max_slices=5,
    )
    group = ScalingGroup(config, make_mock_vm_manager())
    assert not group.matches_device_requirement(DeviceType.GPU, None)
    assert not group.matches_device_requirement(DeviceType.GPU, "v5litepod-16")
```

**`tests/cluster/vm/test_autoscaler.py`** - CPU demand routing

```python
def test_cpu_demand_routes_to_highest_priority():
    """CPU demand routes by priority since it matches all groups."""
    config_tpu = config_pb2.ScaleGroupConfig(
        name="tpu-group",
        accelerator_type=config_pb2.ACCELERATOR_TYPE_TPU,
        accelerator_variant="v5litepod-16",
        max_slices=5,
        priority=20,
    )
    config_gpu = config_pb2.ScaleGroupConfig(
        name="gpu-group",
        accelerator_type=config_pb2.ACCELERATOR_TYPE_GPU,
        accelerator_variant="A100",
        max_slices=5,
        priority=10,
    )

    group_tpu = ScalingGroup(config_tpu, make_mock_vm_manager(), scale_up_cooldown_ms=0)
    group_gpu = ScalingGroup(config_gpu, make_mock_vm_manager(), scale_up_cooldown_ms=0)
    autoscaler = make_autoscaler({"tpu-group": group_tpu, "gpu-group": group_gpu})

    demand = [DemandEntry(device_type=DeviceType.CPU, device_variant=None, count=1)]
    decisions = autoscaler.evaluate(demand)

    assert len(decisions) == 1
    assert decisions[0].scale_group == "gpu-group"  # Higher priority (lower value)
```

---

## Verification Commands

```bash
# Regenerate protos after config.proto change
uv run python -m grpc_tools.protoc ...

# Type check
uv run pyright src/iris/cluster/vm/autoscaler.py src/iris/cluster/vm/scaling_group.py

# Run tests
uv run pytest tests/cluster/vm/test_autoscaler.py tests/cluster/vm/test_scaling_group.py -v
```

---

## Summary of Files to Modify

| File | Changes |
|------|---------|
| `src/iris/rpc/config.proto` | Add `AcceleratorType` enum, add `accelerator_type` field, rename old field to `accelerator_variant` |
| `examples/demo.yaml` | Update to use `accelerator_type` + `accelerator_variant` |
| `examples/eu-west4.yaml` | Update to use `accelerator_type` + `accelerator_variant` |
| `src/iris/cluster/controller/state.py` | Add `DeviceType` enum and `get_device_type_enum()` |
| `src/iris/cluster/vm/autoscaler.py` | Update `DemandEntry` to use `device_type`/`device_variant` |
| `src/iris/cluster/vm/scaling_group.py` | Add `matches_device_requirement()` with type-safe matching |
| `src/iris/cluster/controller/controller.py` | Update `compute_demand_entries()` to use enum |
| Tests | Add tests for enum, matching logic, and CPU demand routing |
