# TPU Pool Expansion & Allocation Tiers

**Status:** implementation plan
**Last updated:** 2026-04-03

## Problem

When the autoscaler can't allocate a TPU of size X in a zone, it falls through the priority waterfall and tries size 2X, 4X, etc. This is wasteful and dangerous:

1. **Wasted API calls**: GCP TPU capacity is monotonic — if v5p-8 is unavailable, v5p-16 will also be unavailable. Each failed attempt burns rate limit tokens and adds latency.
2. **Accidental over-allocation**: If a larger slice transiently succeeds, the job gets more resources than intended and is more likely to be preempted.
3. **Config verbosity**: Each TPU size × zone is a separate scale group entry. The production config has ~35 nearly-identical entries that differ only in size-derived fields.

## Design

Two changes:

### 1. TPU Pool Config Sugar (`tpu_pools`)

A new top-level YAML key that expands into scale groups. Each pool defines shared properties for a TPU family; the `sizes` map lists per-size overrides.

```yaml
tpu_pools:
  v5e-preemptible:
    family: v5e
    zones: [europe-west4-b, us-west4-a]
    base_priority: 10
    resources: { cpu: 112, ram: 192GB, disk: 100GB, preemptible: true }
    slice_template:
      gcp:
        service_account: iris-worker@hai-gcp-models.iam.gserviceaccount.com
        runtime_version: v2-alpha-tpuv5-lite
    sizes:
      4:   { buffer_slices: 3, max_slices: 1024 }
      8:   { max_slices: 512 }
      16:  { max_slices: 256 }
```

The pool name (`v5e-preemptible`) is an operator-chosen label, independent of the TPU family. This allows multiple pools for the same family (e.g., `v5e-preemptible` vs `v5e-reserved` with different zones, priorities, and preemptibility).

**Expansion** (`_expand_tpu_pools`): For each pool × size × zone, emit a scale group:

- **name**: `tpu_{pool}_{size}-{zone}` (e.g., `tpu_v5e-preemptible_16-europe-west4-b`)
- **device_variant**: looked up from `TpuTopologyInfo` via `family` (e.g., `v5e` → `v5litepod-16`)
- **num_vms**: `TpuTopologyInfo.vm_count`
- **device_count**: `TpuTopologyInfo.chips_per_vm`
- **device_type**: `tpu` (injected)
- **priority**: `base_priority + (tier_index × 10)` where tier_index is the 0-based position in sorted sizes
- **quota_pool**: `{pool_name}/{zone}` (e.g., `v5e-preemptible/europe-west4-b`). Per-zone because GCP quota is per-zone — a failure in one zone should not block allocation in another.
- **allocation_tier**: `tier_index + 1` (1-based)
- **buffer_slices**: from size entry, default 0
- **max_slices**: from size entry (required)
- **zone, region**: set on `slice_template.gcp.zone` and `worker.attributes`

The function runs before `_expand_multi_zone_groups` and handles zone expansion itself (TPU pools don't go through the generic zone expander).

**Family → variant mapping**: A dict in `types.py`:

```python
TPU_FAMILY_VARIANT_PREFIX: dict[str, str] = {
    "v4": "v4",
    "v5e": "v5litepod",
    "v5p": "v5p",
    "v6e": "v6e",
}
```

The variant name for a pool with `family: v5e` and size `16` is `v5litepod-16`. This is validated against `TPU_TOPOLOGIES` — unknown family/size combinations are rejected at config load time.

### 2. Allocation Tiers (`quota_pool` + `allocation_tier`)

Two new fields on `ScaleGroupConfig`:

```protobuf
message ScaleGroupConfig {
  // Groups sharing a quota_pool propagate quota-exceeded state together.
  // When tier N in a pool hits quota, tiers > N are blocked.
  string quota_pool = 80;
  int32 allocation_tier = 81;
}
```

**Autoscaler behavior** change in `route_demand`:

When filtering matching groups for a demand entry, skip groups where **any lower-tier group in the same quota_pool** is in `QUOTA_EXCEEDED` or `BACKOFF` state. This is a filter applied after hard constraint matching and before budget assignment.

```python
def _pool_blocked_tiers(groups: list[ScalingGroup], ts: Timestamp) -> dict[str, int]:
    """Return the minimum blocked tier per quota_pool.

    If pool "v5e" has tier 1 in QUOTA_EXCEEDED, returns {"v5e": 1},
    meaning tiers >= 1 should be skipped.
    """
    blocked: dict[str, int] = {}
    for g in groups:
        pool = g.config.quota_pool
        tier = g.config.allocation_tier
        if not pool or not tier:
            continue
        avail = g.availability(ts)
        if avail.status in (GroupAvailability.QUOTA_EXCEEDED, GroupAvailability.BACKOFF):
            if pool not in blocked or tier < blocked[pool]:
                blocked[pool] = tier
    return blocked
```

In `route_demand`, after `matching_groups` is computed:

```python
blocked = _pool_blocked_tiers(sorted_groups, ts)
matching_groups = [
    g for g in matching_groups
    if not g.config.quota_pool
    or g.config.allocation_tier < blocked.get(g.config.quota_pool, float('inf'))
]
```

**Dashboard**: The AutoscalerTab groups scale groups by `quota_pool` when present, showing a visual tier chain: `[v5p-8 ✓] → [v5p-16 ⊘] → [v5p-32 ⊘]`.

## Implementation Plan

### Stage 1: Proto + config expansion

1. Add `TPU_FAMILY_VARIANT_PREFIX` dict to `types.py`
2. Add `quota_pool` and `allocation_tier` fields to `ScaleGroupConfig` in `config.proto`, regenerate
3. Implement `_expand_tpu_pools()` in `config.py`
4. Wire into `load_config()` before `_expand_multi_zone_groups()`
5. Tests: expansion correctness, topology derivation, validation errors
6. Migrate `examples/marin.yaml` to `tpu_pools` format

### Stage 2: Autoscaler tier blocking

7. Implement `_pool_blocked_tiers()` in `autoscaler.py`
8. Add tier filtering to `route_demand()`
9. Tests: tier blocking on quota exceeded, independent pools, groups without pools

### Stage 3: Dashboard

10. Group autoscaler view by `quota_pool`
11. Show tier chain with blocked/available visual state
