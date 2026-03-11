# Constraint System

Iris uses a unified constraint system for both **worker scheduling** (assigning
tasks to workers) and **scaling group routing** (deciding which group should
provision capacity for a demand entry). The shared primitives live in
`iris.cluster.constraints`.

## Core types

### `AttributeValue`
A typed wrapper around `str | int | float`. Workers and scaling groups expose
their properties as `dict[str, AttributeValue]` — the same representation for
both entity types.

### `evaluate_constraint(attr, constraint) -> bool`
The single evaluation function used everywhere. Takes an `AttributeValue | None`
and a proto `Constraint`, returns whether the constraint is satisfied. Supports
EQ, NE, EXISTS, NOT_EXISTS, GT/GE/LT/LE (numeric only), and IN.

### `ConstraintIndex`
Posting-list index for fast constraint matching over any set of entities
(workers or scaling groups). Built once per scheduling/routing cycle from
`dict[str, dict[str, AttributeValue]]`.

Fast paths (O(1) per constraint):
- **EQ**: direct posting-list lookup
- **IN**: union of posting lists for each value
- **EXISTS**: union of all posting lists for the key
- **NOT_EXISTS**: complement of EXISTS

Slow path (linear scan): NE, GT, GE, LT, LE — rare in practice.

### `ResourceCapacity` and `check_resource_fit`
Frozen dataclass with cpu/memory/disk/gpu/tpu dimensions (each `int | None`,
where `None` means "not configured / unlimited" on the available side and
"not needed" on the required side). The shared `check_resource_fit(available,
required)` function is used by both `ScalingGroup.check_resource_fit()` and
`WorkerCapacity.can_fit()`.

## Constraint registry

`CONSTRAINT_REGISTRY` maps well-known attribute keys to `ConstraintDescriptor`
objects declaring type, allowed operators, and whether the constraint is:

- **canonical**: child constraints override parent (used by `merge_constraints`)
- **routing**: evaluated during scaling group matching (vs scheduler-only)

| Key | Routing | Canonical | Notes |
|---|---|---|---|
| `device-type` | yes | yes | `cpu`, `gpu`, `tpu` |
| `device-variant` | yes | yes | e.g. `h100`, `v5litepod-16` (always lowercase) |
| `preemptible` | yes | yes | `"true"` or `"false"` |
| `region` | yes | yes | e.g. `us-central1` |
| `zone` | yes | yes | e.g. `us-central1-a` |
| `tpu-name` | no | no | scheduler-only, matches individual workers |
| `tpu-worker-id` | no | no | scheduler-only |
| `tpu-topology` | no | no | scheduler-only |
| `tpu-vm-count` | no | no | scheduler-only |
| `gpu-variant` | no | no | scheduler-only |
| `gpu-count` | no | no | scheduler-only (consumable) |

## How constraints flow through the system

### Job submission → proto constraints

1. `constraints_from_resources()` auto-generates `device-type` and
   `device-variant` constraints from the job's `ResourceSpecProto`.
2. User-supplied constraints are merged via `merge_constraints()`, where
   canonical keys in the child override the parent.
3. The final `list[cluster_pb2.Constraint]` is stored on the job and carried
   through to both the scheduler and autoscaler.

### Autoscaler: scaling group routing

`route_demand()` builds a `ConstraintIndex` over all scaling groups at the
start of each cycle:

```
group_attrs = {g.name: g.to_attributes() for g in sorted_groups}
group_index = ConstraintIndex.build(group_attrs)
```

`ScalingGroup.to_attributes()` returns the group's routing properties as a
`dict[str, AttributeValue]` — the same shape as worker attributes. This is
what makes the shared index work for both entity types.

For each demand entry, `routing_constraints()` filters the entry's constraints
to routing-only ones (stripping `tpu-name`, `tpu-worker-id`, etc.) and removes
`device-type=cpu` constraints (CPU jobs match any group). The filtered
constraints are evaluated against the group index:

```
routing_cs = routing_constraints(entry.constraints)
matching_names = group_index.matching_entities(routing_cs)
```

Matched groups are then tried in priority order via the waterfall budget system.

### Scheduler: worker matching

`SchedulingContext.from_workers()` builds a `ConstraintIndex` over all healthy
workers. `matching_workers()` evaluates the job's full constraint list
(including non-routing constraints like `tpu-name`) against the worker index.

The scheduler uses all constraints — not just routing ones — because workers
have attributes for everything (tpu-name, tpu-worker-id, zone, etc.).

## Example: flexible TPU variant scheduling

A job requesting either `v4-8` or `v5-8` TPUs:

```python
constraints = [
    device_variant_constraint(["v4-8", "v5-8"]),  # generates IN constraint
]
# With auto-generated device-type from resources:
# final constraints = [device-type EQ "tpu", device-variant IN ("v4-8", "v5-8")]
```

**Scaling group routing** (which group to provision):

Given three groups with attributes:
```
tpu-v4-8-us:  {device-type: "tpu", device-variant: "v4-8", region: "us-central1"}
tpu-v5-8-us:  {device-type: "tpu", device-variant: "v5-8", region: "us-central2"}
gpu-h100-us:  {device-type: "gpu", device-variant: "h100", region: "us-central1"}
```

The `ConstraintIndex` posting lists:
```
device-type:    {"tpu": {tpu-v4-8-us, tpu-v5-8-us}, "gpu": {gpu-h100-us}}
device-variant: {"v4-8": {tpu-v4-8-us}, "v5-8": {tpu-v5-8-us}, "h100": {gpu-h100-us}}
```

Evaluation:
1. `device-type EQ "tpu"` → posting list lookup → `{tpu-v4-8-us, tpu-v5-8-us}`
2. `device-variant IN ("v4-8", "v5-8")` → union of posting lists → `{tpu-v4-8-us, tpu-v5-8-us}`
3. Intersection → `{tpu-v4-8-us, tpu-v5-8-us}`

Both TPU groups match. The waterfall routes demand to the higher-priority group
first; overflow cascades to the next.

**Worker scheduling** (which worker to assign the task to):

The same `ConstraintIndex` mechanism runs over workers. Workers probed from
`v4-8` TPUs have `device-variant: "v4-8"` in their attributes. The `IN`
constraint matches any worker whose variant is in the set.

## CPU job routing

CPU-only jobs do **not** receive any device-related constraints from
`constraints_from_resources()`. That helper only emits device constraints for
non-CPU accelerators (e.g., GPU or TPU resources).

As a result, CPU jobs have no device constraints and match all scaling groups
by default. The waterfall then routes them to the highest-priority group with
available CPU capacity.

`ScalingGroup.matches_constraints()` applies the same stripping via
`is_cpu_device_type_constraint()`, so the per-budget `try_assign()` check
is consistent.

## Lowercase normalization

Device variants are normalized to lowercase at every entry point:
- `device_variant_constraint()` lowercases values
- `constraints_from_resources()` lowercases the variant from proto
- `worker_attributes_from_resources()` lowercases the variant
- `ScalingGroup.to_attributes()` lowercases the variant
- Worker `env_probe.py` lowercases the probed accelerator variant

This means `evaluate_constraint` does exact string comparison — no
case-insensitive flag needed.
