# Design: Preemptible as a Schedulable Constraint

**Issue**: #2553 — Expose worker preemptible status as a schedulable attribute

## Problem

Today, `ResourceSpec.preemptible` and `ResourceSpecProto.preemptible` exist as fields but have **no effect on scheduling**. The scheduler never reads them. The autoscaler's `ScaleGroupConfig` also has a `preemptible` field, which is used only for VM creation (passing `--preemptible` to `gcloud`), not for routing demand or matching workers.

Users cannot currently say "run this job only on non-preemptible workers" or "prefer preemptible workers for this batch job." The constraint system already supports arbitrary key-value matching on worker attributes, but preemptibility is not wired through it.

### What needs to happen

1. **Workers** detect whether they are preemptible and register `preemptible=true/false` as a worker attribute.
2. **The scheduler** uses existing constraint matching to filter workers by preemptibility.
3. **The autoscaler** routes demand to the correct scale group (preemptible vs on-demand) based on job requirements.

## Current State

### Worker registration (env_probe.py)

Workers build attributes from TPU environment variables and `IRIS_WORKER_ATTRIBUTES`:

```python
def _build_worker_attributes(tpu_name, tpu_worker_id, device, extra_attributes):
    attributes = {}
    if tpu_name:
        attributes["tpu-name"] = AttributeValue(string_value=tpu_name)
        attributes["tpu-worker-id"] = AttributeValue(int_value=int(tpu_worker_id))
    for key, value in extra_attributes.items():
        attributes[key] = AttributeValue(string_value=value)
    return attributes
```

No preemptible detection exists today.

### Scheduler (scheduler.py)

The scheduler evaluates `job.request.constraints` against `worker.attributes` using posting lists. It **never reads** `resources.preemptible`. The constraint system is fully generic and already handles EQ, NE, EXISTS, etc.

### Autoscaler demand routing (autoscaler.py)

`route_demand()` matches demand to scale groups by `device_type` and `device_variant` only. There is no consideration of preemptibility when routing demand.

### ResourceSpec (types.py)

```python
@dataclass
class ResourceSpec:
    cpu: int = 0
    memory: str | int = 0
    device: DeviceConfig | None = None
    replicas: int = 0
    preemptible: bool = False   # <-- exists but unused by scheduler
    regions: Sequence[str] | None = None
```

### Client API (client.py)

`IrisClient.remote()` takes both `resources: ResourceSpec` and `constraints: list[Constraint]`. These are independent — `ResourceSpec.preemptible` is serialized to the proto but never acts as a constraint.

---

## Approach A: Keep `ResourceSpec.preemptible` as sugar

Keep the `preemptible` field on `ResourceSpec`. Internally, convert it to a constraint before submission.

### User-facing API

```python
# Option 1: Sugar (unchanged API)
client.remote(
    fn,
    resources=ResourceSpec(device=tpu_device("v4-8"), preemptible=True),
)

# Option 2: Explicit constraint (also works, always has)
client.remote(
    fn,
    resources=ResourceSpec(device=tpu_device("v4-8")),
    constraints=[Constraint(key="preemptible", op=ConstraintOp.EQ, value="true")],
)
```

### Implementation

In `IrisClient.remote()` (or in `ResourceSpec.to_constraints()`), auto-generate a constraint:

```python
def _resource_constraints(resources: ResourceSpec) -> list[Constraint]:
    """Convert ResourceSpec scheduling preferences to constraints."""
    constraints = []
    if resources.preemptible:
        constraints.append(Constraint(key="preemptible", op=ConstraintOp.EQ, value="true"))
    else:
        constraints.append(Constraint(key="preemptible", op=ConstraintOp.EQ, value="false"))
    return constraints
```

The proto field `ResourceSpecProto.preemptible` becomes informational / can be removed from the wire format since the constraint carries the scheduling intent.

### Worker registration

```python
# In env_probe.py, detect preemptibility from GCP metadata or scale group config
def _detect_preemptible() -> bool:
    # GCP: query metadata server scheduling/preemptible
    # Or: read from IRIS_WORKER_PREEMPTIBLE env var set by bootstrap
    ...

attributes["preemptible"] = AttributeValue(string_value=str(_detect_preemptible()).lower())
```

### Autoscaler demand routing

`DemandEntry` gains a `preemptible: bool | None` field. `route_demand()` filters scale groups:

```python
# In route_demand(), when matching groups:
if entry.preemptible is not None:
    matching = [g for g in matching if g.config.preemptible == entry.preemptible]
```

`compute_demand_entries()` reads the constraint (or the proto field) to determine preemptible preference.

### Pros

- Zero API change for users who already use `ResourceSpec(preemptible=True)`
- Discoverable: `preemptible` appears in IDE autocomplete
- Consistent with `regions` (also a scheduling preference on ResourceSpec)

### Cons

- Two ways to express the same thing (field + constraint) — potential for conflicts
- Need conflict resolution: what if user passes `preemptible=True` AND `Constraint(key="preemptible", op=EQ, value="false")`?
- `ResourceSpec` accumulates scheduling fields that are really constraints in disguise

---

## Approach B: Remove `preemptible` from `ResourceSpec`

Remove `ResourceSpec.preemptible` and `ResourceSpecProto.preemptible`. Users express preemptibility purely through constraints.

### User-facing API

```python
client.remote(
    fn,
    resources=ResourceSpec(device=tpu_device("v4-8")),
    constraints=[Constraint(key="preemptible", op=ConstraintOp.EQ, value="true")],
)

# Or with a helper:
from iris.cluster.types import preemptible_constraint

client.remote(
    fn,
    resources=ResourceSpec(device=tpu_device("v4-8")),
    constraints=[preemptible_constraint(True)],
)
```

Where:

```python
def preemptible_constraint(preemptible: bool = True) -> Constraint:
    """Constraint requiring workers to be preemptible (or not)."""
    return Constraint(key="preemptible", op=ConstraintOp.EQ, value=str(preemptible).lower())
```

### Implementation

1. Remove `preemptible` from `ResourceSpec` and `ResourceSpecProto`.
2. Worker registration: same as Approach A (detect and register attribute).
3. Autoscaler: extract preemptible preference from job constraints:

```python
def _extract_preemptible_preference(constraints: list[cluster_pb2.Constraint]) -> bool | None:
    """Extract preemptible preference from constraints, if any."""
    for c in constraints:
        if c.key == "preemptible" and c.op == cluster_pb2.CONSTRAINT_OP_EQ:
            return AttributeValue.from_proto(c.value).value == "true"
    return None  # No preference
```

4. `compute_demand_entries()` groups demand by `(device_type, device_variant, preemptible)`.

### Pros

- Single source of truth: constraints are the only scheduling filter mechanism
- No conflict resolution needed
- `ResourceSpec` stays focused on resource quantities
- Consistent with how other scheduling attributes (zone, tpu-name) already work via constraints

### Cons

- Slightly more verbose for the common case (mitigated by helper function)
- Breaking change: callers using `ResourceSpec(preemptible=True)` must migrate
- `preemptible` is arguably more discoverable as a field than as a constraint key string

---

## Recommendation: Approach B (constraints only)

**Remove `preemptible` from `ResourceSpec`.** Rationale:

1. **Single mechanism**: The constraint system already exists and works well. Every other worker attribute (tpu-name, zone, topology) is matched through constraints. Preemptible should not be special-cased.

2. **No conflict resolution**: Approach A introduces an ambiguity that must be resolved and documented. With Approach B, the constraint is the single source of truth.

3. **The field is currently dead code**: Nobody is relying on `ResourceSpec.preemptible` for scheduling today (it does nothing). This is the ideal time to remove it. The AGENTS.md says "NO BACKWARD COMPATIBILITY" — update all call sites.

4. **`regions` should also migrate to constraints eventually**: Having `preemptible` as a constraint sets the right precedent. `ResourceSpec` should describe resource *quantities* (cpu, memory, device, replicas), not scheduling *preferences*.

5. **The helper function is sufficient sugar**: `preemptible_constraint(True)` is clear and discoverable, and can be documented alongside other common constraints.

### Migration

Since `preemptible` is currently unused by the scheduler, migration is straightforward:

1. Remove `preemptible` from `ResourceSpec`, `ResourceSpecProto`, `cluster.proto`
2. Add `preemptible_constraint()` helper to `iris.cluster.types`
3. Update any call sites (grep shows none using it for scheduling)
4. Add worker preemptible detection in `env_probe.py`
5. Update `compute_demand_entries()` to extract preemptible preference from constraints
6. Update `route_demand()` to filter groups by preemptible

### Autoscaler integration detail

The autoscaler needs to route preemptible demand to preemptible scale groups. The key change is in `compute_demand_entries()`:

```python
# Group key becomes (device_type, device_variant, preemptible)
key = (device_type, device_variant, preemptible_pref)
```

And `route_demand()` filters:

```python
for entry in demand_entries:
    matching = [g for g in groups if g.matches_device_requirement(entry.device_type, entry.device_variant)]
    if entry.preemptible is not None:
        matching = [g for g in matching if g.config.preemptible == entry.preemptible]
```

This ensures preemptible jobs are routed to preemptible scale groups and vice versa, while jobs with no preemptible constraint can be routed to any matching group.

---

## Review Notes

Reviewed by: senior engineer review pass

### 1. Approach B is correct

Agree with the recommendation. The constraint system is the right place for this. The `ResourceSpec.preemptible` field is dead code today and removing it is the clean path forward per the project's no-backward-compat policy.

### 2. Critical gap: `compute_demand_entries` does not have access to constraints

The design says `compute_demand_entries()` should group by `(device_type, device_variant, preemptible)` and extract preemptible preference from constraints. However, looking at the actual code in `controller.py` (lines 75-117), `compute_demand_entries()` iterates over pending **tasks**, gets the **job** for each task, and reads `job.request.resources`. It does **not** currently read `job.request.constraints`.

The function signature is:

```python
def compute_demand_entries(state: ControllerState) -> list[DemandEntry]:
```

To extract preemptible preference, the implementation needs to also read `job.request.constraints` (or the task's constraints) from the state. This is straightforward but should be called out as a concrete code change: the function must iterate constraints for each job to extract the preemptible key.

Additionally, the grouping key change means that two jobs requesting the same `(v4-8, None)` but with different preemptible preferences will produce **two separate `DemandEntry` objects**. This is correct behavior but is a behavioral change to the demand computation that should be tested.

### 3. Worker detection: GCP metadata is the right approach, but specify the mechanism

The doc says "detect preemptibility from GCP metadata or scale group config" but does not commit to a mechanism. Two concrete options:

- **GCP metadata server**: `curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/scheduling/preemptible` returns `TRUE` or `FALSE`. This is the most reliable source and requires no changes to the bootstrap.
- **`IRIS_WORKER_ATTRIBUTES` env var**: The bootstrap script (in `gcp_tpu_platform.py`) would need to be modified to pass `preemptible=true` in the `IRIS_WORKER_ATTRIBUTES` env var. This is simpler but requires the bootstrap to know the scale group config.

**Recommendation**: Use the GCP metadata server. It is authoritative and does not require bootstrap changes. The env_probe already runs on the worker and can make this HTTP call. For non-GCP environments (local, manual), default to `preemptible=false` or let users set it via `IRIS_WORKER_ATTRIBUTES`.

Note: for **TPU VMs** specifically, the metadata endpoint may differ. Verify that `scheduling/preemptible` works on TPU VMs, not just standard GCE VMs. If not available, fall back to checking `IRIS_WORKER_ATTRIBUTES`.

### 4. Proto field removal: safe but needs sequenced rollout consideration

The doc says to remove `preemptible` from `cluster.proto` (field 7 on `ResourceSpec`) and `config.proto` (field 12 on `ScaleGroupConfig`).

- **`cluster.proto` ResourceSpec field 7**: Safe to remove. No running system reads this for scheduling. However, if there are serialized jobs in flight (e.g. in the controller's state), removing the field means proto3 will silently ignore it on deserialization. This is fine — no data loss, just the field disappears.
- **`config.proto` ScaleGroupConfig field 12**: Do **NOT** remove this. The autoscaler needs `ScaleGroupConfig.preemptible` to know which scale groups create preemptible VMs (used in `gcp_tpu_platform.py` line 326: `if self._config.preemptible: cmd.append("--preemptible")`). The design doc should clarify: remove from `ResourceSpec` only, keep on `ScaleGroupConfig`.

### 5. `DemandEntry` needs a new field

The `DemandEntry` dataclass (autoscaler.py line 77) currently has: `device_type`, `device_variant`, `count`, `total_cpu`, `total_memory_bytes`. It needs a `preemptible: bool | None = None` field. This is mentioned in passing but should be explicit in the migration steps.

### 6. Default behavior for jobs without preemptible constraint

The design says jobs with no preemptible constraint "can be routed to any matching group." This is correct and important. But consider: if a cluster has both preemptible and on-demand scale groups, and a job has no preference, `route_demand` will route to whichever group has higher priority. This means the `priority` field on `ScaleGroupConfig` becomes the mechanism for "prefer preemptible" vs "prefer on-demand" for unconstrained jobs. This interaction should be documented.

### 7. Test coverage

Existing tests in `test_controller.py` cover scale group config parsing (including `preemptible: true` in YAML). The autoscaler tests (`test_autoscaler.py` presumably) test `route_demand`. New tests needed:

- `compute_demand_entries` produces separate entries when jobs differ only by preemptible constraint
- `route_demand` filters groups by preemptible when `DemandEntry.preemptible` is set
- `route_demand` does NOT filter when `DemandEntry.preemptible` is None
- Worker attribute includes `preemptible` key after detection
- `preemptible_constraint()` helper produces correct `Constraint` proto

### 8. Migration step ordering (per AGENTS.md spiral plan preference)

The migration steps should follow the spiral approach from AGENTS.md. Suggested reorder:

**Step 1** (minimal end-to-end):
- Add `preemptible: bool | None` to `DemandEntry`
- Add `preemptible_constraint()` helper to types.py
- Add preemptible detection to `env_probe.py`
- Add test: worker registers preemptible attribute, job with preemptible constraint matches

**Step 2** (autoscaler integration):
- Update `compute_demand_entries` to extract preemptible from constraints
- Update `route_demand` to filter by preemptible
- Add autoscaler routing tests

**Step 3** (cleanup):
- Remove `ResourceSpec.preemptible` field
- Remove `ResourceSpecProto.preemptible` from `cluster.proto`
- Update all call sites (grep confirms none use it for scheduling)

### Summary

The design is sound. The main issues are:
1. **Config proto**: Do not remove `ScaleGroupConfig.preemptible` — it controls VM creation, not scheduling.
2. **Specify detection mechanism**: Commit to GCP metadata server as primary, env var as fallback.
3. **Document priority interaction**: Unconstrained jobs route by group priority, which is the implicit "preference" mechanism.
4. **Spiral the implementation**: Follow AGENTS.md planning guidance.
