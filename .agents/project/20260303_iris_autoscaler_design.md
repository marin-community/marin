# Autoscaler Fix: Priority Walk-Up, Cooldown, and Additive Resource Packing

**Status:** implementation plan (intended to be executed immediately)
**Last updated:** 2026-03-03
**Canonical references:** `docs/autoscaler-v2.md`, `docs/controller-flow.md`, `docs/worker-flow.md`

This document is an implementation-oriented plan. It is written to be actionable against the current codebase (including line-number callouts), not as a timeless design spec.

## System Model (What the Autoscaler Is Actually Doing)

### Demand entries are *unscheduled work*, not “tasks that exist”

The controller builds `DemandEntry` objects from:

- pending (unscheduled) tasks, and
- reservation entries (persistent demand that prevents scale-down).

The autoscaler does **not** try to “balance load” across ready workers. That is the scheduler’s job. The autoscaler’s job is to decide whether to request more slices.

Availability and scale decisions are computed per scaling group; multiple groups can be in `REQUESTING` simultaneously.

### Fungible vs non-fungible resources (device requirements)

Iris treats CPU as a fungible baseline resource:

- **CPU demand** (`device=cpu`) can be routed to *any* scaling group, including GPU/TPU groups, as long as the group’s per-VM resources and constraints match.
- **GPU/TPU demand** is non-fungible: it requires a matching device type, and (if specified) a matching variant.

This is intentional: it enables “spillover” when dedicated CPU groups are exhausted. It also means priority configuration matters: CPU-oriented groups should usually have higher routing priority than accelerator groups, otherwise CPU-only demand can consume expensive workers.

### Additive vs categorical requirements (what “fungible” means for capacity)

For autoscaling math, Iris must distinguish:

- **Additive capacity**: resources that add linearly with the number of VMs.
  - Example: if each VM has `cpu_millicores` and `memory_bytes` capacity, then a slice with `num_vms=N` adds `N` VMs worth of capacity.
  - Important: “additive” does **not** mean “pool into one giant RAM bucket”. It means we can add capacity across *independent VMs*, and should estimate how many VMs are required via packing.

- **Categorical eligibility**: requirements that are not meaningfully “summed” and instead gate whether a group can ever satisfy an entry.
  - Examples: device type (CPU vs GPU vs TPU), TPU/GPU variant strings (e.g. `tpuv5e-litepod16`), region/zone constraints, and preemptibility preference.
  - These should be treated as *filters* (match / no-match), not as capacity arithmetic.

### Capacity units: slices vs VMs (and why we need packing)

Within a scaling group, a “slice” can contain multiple VMs (`num_vms`). For routing and scale-up math:

- A **coscheduled** `DemandEntry` consumes **1 full slice** (all `num_vms` VMs) and cannot be packed with other work.
- A **non-coscheduled** `DemandEntry` consumes **some fraction of a VM** (CPU/memory), and multiple entries may fit on one VM.

This is the core bug in the “walk up” behavior: the current routing math treats each demand entry as “1 slice”, ignoring that multi-VM slices can serve multiple independent CPU tasks and that multiple tasks can pack onto a single VM when their additive resource requests allow it.

### Disk (current behavior and plan scope)

Disk requests exist in the demand `resources` proto and are checked for **per-VM fit** in `ScalingGroup.check_resource_fit()` when the scale group config sets a non-zero disk limit.

However, disk is not currently part of:

- scheduler capacity accounting (worker available disk is not tracked in `WorkerCapacity`), or
- autoscaler packing (until this change).

So disk is treated as an **eligibility gate** (“can a single VM of this group satisfy a single task’s disk request?”) but not as an additive dimension for packing or live placement.

#### Make disk additive in autoscaler packing (plan scope)

For now, implement disk additivity **only** in autoscaler packing by treating `resources.disk_bytes` as a third additive dimension in the bin packer:

- **Per-VM capacity for packing**: use the scale group’s configured per-VM disk capacity (`ScaleGroupConfig.resources.disk_bytes`).
  - If `disk_bytes == 0` (not configured), treat disk as **unbounded** for packing (consistent with `check_resource_fit()` semantics).
- **Per-entry request**: use `DemandEntry.resources.disk_bytes`.
- **Eligibility remains per-VM**: keep `check_resource_fit()` as the hard gate. Packing is an estimate for required capacity, not a replacement for fit checks.

This is intentionally conservative and configuration-driven (not runtime-driven): it assumes disk requests are enforceable enough at the VM level that summing requested bytes is a reasonable proxy for “how many VMs do we need?”. It will not perfectly match runtime behavior (e.g., if the runtime does not enforce disk limits), but it will prevent the autoscaler from assuming “infinite disk” when the config intends disk to be capacity-limiting.

Implementation steps for disk in packing:

1. Extend `AdditiveReq` with `disk_bytes`.
2. Extend `VmBin` with `disk_remaining`.
3. Update `additive_req(entry)` to include disk.
4. Update bin sort key for FFD to reduce fragmentation across three dimensions:
   - sort by `(disk_bytes, memory_bytes, cpu_millicores)` descending.

If this proves too noisy in practice (disk requests vary widely and dominate the sort), prefer a weighted key (e.g. disk first only when disk is configured for the group).

## Problem

A large tokenization job requesting many CPU-only workers causes the autoscaler
to "walk up" the priority tree — distributing demand across increasingly larger
TPU slice groups (v5e-4 → v5e-8 → ... → v5e-256) all in a single cycle.

### Root Causes

1. **Capacity accounting ignores `num_vms`**: A v5e-64 slice has 16 VMs
   (`num_vms=16`), meaning it can serve 16 CPU tasks. But routing treats each
   slice as capacity for only 1 demand entry. So a group with `max_slices=2`
   and `num_vms=4` is treated as capacity for 2 entries when it actually serves
   8 (2×4).

2. **No explicit COOLDOWN state**: The cooldown period (rate-limiting between
   scale-ups) is invisible in routing — `availability()` returns AVAILABLE
   during cooldown. While `can_scale_up()` correctly blocks the actual scale-up,
   there's no visibility or semantic distinction from a truly available group.

3. **AT_CAPACITY accepts demand but shouldn't**: When a group hits
   `max_slices`, it's currently in the ACCEPTING set (demand stays). This means
   demand is absorbed by the full group's existing slices, preventing it from
   falling through. For the walk-up scenario, a full group should REJECT demand
   so it cascades to the next priority level — identical to QUOTA_EXCEEDED
   semantics.

### Key Semantic Rules

- **COOLDOWN** = demand STAYS (group is healthy; scale-up is rate-limited).
- **REQUESTING** = demand STAYS (scale-up already in flight; capacity is incoming).
- **BACKOFF / QUOTA_EXCEEDED / AT_MAX_SLICES** = demand falls through (group cannot provide additional capacity).

**Precedence rule:** `AT_MAX_SLICES` must override `COOLDOWN`. If a group is both “cooling down” and already at `max_slices`, routing must treat it as blocked (otherwise a maxed-out group can keep absorbing demand, hiding the need to fall through).

`max_slices` is a pre-known capacity limit. Once hit, demand should fall
through to the next group — same semantics as QUOTA_EXCEEDED.

## Changes

## Detailed Flow Diagram (Demand → Routing → Scaling)

The following diagram is intentionally verbose. It is meant to document *exactly* how demand is produced, how groups accept/fall-through demand, and how backoff/cooldown/max-slices interact.

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│ Controller loop (daemon thread(s))                                                   │
│                                                                                      │
│ 1) Scheduler loop (fast, frequent)                                                   │
│    ┌──────────────────────────────────────────────────────────────────────────────┐ │
│    │ controller.py scheduling cycle                                                 │ │
│    │                                                                              │ │
│    │ Inputs:                                                                       │ │
│    │ - pending tasks from ControllerState                                           │ │
│    │ - available workers from ControllerState                                       │ │
│    │ - per-worker available CPU/memory/GPU/TPU (committed-resources model)          │ │
│    │ - constraints + reservation taints/claims (if any)                             │ │
│    │                                                                              │ │
│    │ Core behavior:                                                                │ │
│    │ - attempts to place tasks on existing READY workers                            │ │
│    │ - rejects if insufficient per-worker available capacity                         │ │
│    │ - rejected tasks remain pending                                                │ │
│    │                                                                              │ │
│    │ Output: assignments (task → worker) and state mutations                         │ │
│    └──────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                      │
│ 2) Autoscaler loop (slower, periodic)                                                │
│    ┌──────────────────────────────────────────────────────────────────────────────┐ │
│    │ controller.py:_run_autoscaler_once                                              │ │
│    │                                                                              │ │
│    │ Phase A: refresh (state-read / I/O)                                            │ │
│    │   vm_status_map = controller._build_vm_status_map()                            │ │
│    │   autoscaler.refresh(vm_status_map)                                            │ │
│    │                                                                              │ │
│    │   refresh() does:                                                             │ │
│    │   - polls non-ready slices via platform SliceHandle.describe()                 │ │
│    │   - marks slices READY/FAILED, registers workers, records failures/backoff     │ │
│    │   - computes per-group target_capacity and may scale down idle slices          │ │
│    │                                                                              │ │
│    │ Phase B: update (CPU)                                                         │ │
│    │   demand_entries = compute_demand_entries(state)                               │ │
│    │   autoscaler.update(demand_entries)                                            │ │
│    └──────────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────────┐
│ compute_demand_entries(state)                                                        │
│                                                                                      │
│ Produces DemandEntry objects representing “work the cluster wants capacity for”.     │
│                                                                                      │
│ - Pending tasks: one entry per (unscheduled) task (or per coscheduled group)         │
│ - Reservation entries (if enabled): one entry per reservation entry                  │
│                                                                                      │
│ Each DemandEntry includes:                                                          │
│ - resources: cpu_millicores, memory_bytes, disk_bytes, device (cpu/gpu/tpu+variant)  │
│ - constraints: region/zone/attributes, reservation taints, etc.                      │
│ - coschedule_group_id: set iff atomic coscheduled group                              │
└──────────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────────┐
│ autoscaler.update(demand_entries)                                                    │
│                                                                                      │
│ Step 1: route_demand(groups, demand_entries, ts)                                     │
│   - sorts groups by priority (lower number first)                                    │
│   - for each demand entry, finds the first group that “can fit”                      │
│                                                                                      │
│   can_fit_group(group, entry) checks:                                                │
│   - categorical eligibility:                                                         │
│     - device type/variant match rules                                                │
│     - preemptible preference                                                         │
│     - required region/zone constraints                                               │
│     - constraint filters                                                             │
│   - per-VM fit (not additive packing):                                               │
│     - ScalingGroup.check_resource_fit() verifies entry fits in ONE VM                │
│       (cpu/mem/disk limits, accelerator counts)                                      │
│   - acceptance / fallthrough:                                                        │
│     - ScalingGroup.can_accept_demand(ts)                                             │
│                                                                                      │
│   ScalingGroup.availability(ts) computes a single status by precedence:              │
│     QUOTA_EXCEEDED > BACKOFF > REQUESTING > AT_MAX_SLICES > COOLDOWN > AVAILABLE     │
│                                                                                      │
│   Semantics:                                                                         │
│   - ACCEPTING (demand stays attributed here): AVAILABLE, COOLDOWN, REQUESTING        │
│   - REJECTING (demand falls through): BACKOFF, QUOTA_EXCEEDED, AT_MAX_SLICES         │
│                                                                                      │
│   Why AT_MAX_SLICES overrides COOLDOWN:                                              │
│   - if both are true, the group cannot create more slices anyway, so it must not     │
│     keep absorbing demand that needs to spill to lower-priority groups.              │
│                                                                                      │
│ Step 2: compute required capacity per group (this plan changes this step)           │
│   - For each group, consider routed entries for that group:                          │
│     - coscheduled entries: 1 slice each                                              │
│     - non-coscheduled entries: pack by additive resources                            │
│       (cpu_millicores + memory_bytes; disk excluded unless explicitly extended)      │
│   - Run first-fit decreasing (FFD) packing to compute required_vms                   │
│   - Convert required_vms → required_slices = ceil(required_vms / group.num_vms)      │
│   - total_required_slices = required_slices + coscheduled_slices                     │
│                                                                                      │
│ Step 3: evaluate scaling (per group)                                                 │
│   - counts = group.slice_state_counts()                                              │
│   - capacity_slices = READY + (BOOTING + INITIALIZING + REQUESTING)                  │
│   - target = min(required_slices + buffer_slices, max_slices)                          │
│   - slices_needed = max(demand_gap, buffer_gap)                                        │
│   - if slices_needed > 0 and total < max_slices: scale up (if can_scale_up)            │
│   - can_scale_up gates on quota/backoff/cooldown/max_slices                          │
│                                                                                      │
│ Step 4: execute scale-ups                                                            │
│   - begin_scale_up increments REQUESTING count and starts a background thread        │
│   - complete/cancel updates group state; failures record backoff                      │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

### 1. Add COOLDOWN State, Rename AT_CAPACITY → AT_MAX_SLICES

**File:** `lib/iris/src/iris/cluster/controller/scaling_group.py`

#### 1a. Update `GroupAvailability` Enum

```python
class GroupAvailability(Enum):
    """Availability state for waterfall routing.

    ACCEPTING states (demand stays here, scale-up may be deferred):
    - AVAILABLE: can create new slices immediately
    - COOLDOWN: recently scaled up, next scale-up deferred until cooldown expires
    - REQUESTING: scale-up in progress, capacity incoming

    REJECTING states (demand falls through to lower-priority groups):
    - BACKOFF: slice creation failed, exponential backoff active
    - QUOTA_EXCEEDED: cloud quota exhausted
    - AT_MAX_SLICES: configured slice limit reached (pre-known capacity ceiling)
    """

    AVAILABLE = "available"
    COOLDOWN = "cooldown"
    REQUESTING = "requesting"
    AT_MAX_SLICES = "at_max_slices"
    BACKOFF = "backoff"
    QUOTA_EXCEEDED = "quota_exceeded"
```

#### 1b. Update `availability()`

Insert `AT_MAX_SLICES` check **before** `COOLDOWN` so `AT_MAX_SLICES` overrides cooldown.

Priority order becomes:

`QUOTA_EXCEEDED > BACKOFF > REQUESTING > AT_MAX_SLICES > COOLDOWN > AVAILABLE`

```python
def availability(self, timestamp=None):
    timestamp = timestamp or Timestamp.now()

    # ... quota_exceeded and backoff checks unchanged ...

    with self._slices_lock:
        pending = self._pending_scale_ups
        count = len(self._slices) + pending
    if pending > 0:
        return AvailabilityState(GroupAvailability.REQUESTING, "scale-up in progress")

    # At max slices: demand falls through (pre-known capacity ceiling)
    if count >= self._config.max_slices:
        return AvailabilityState(GroupAvailability.AT_MAX_SLICES)

    # Cooldown: recently scaled up, next scale-up deferred. Demand stays.
    cooldown_end = self._last_scale_up.add(self._scale_up_cooldown)
    if self._last_scale_up.epoch_ms() > 0 and timestamp.before(cooldown_end):
        return AvailabilityState(GroupAvailability.COOLDOWN, "scale-up cooldown", cooldown_end)

    return AvailabilityState(GroupAvailability.AVAILABLE)
```

#### 1c. Update `can_accept_demand()`

Only ACCEPTING states keep demand. AT_MAX_SLICES is now REJECTING:

```python
def can_accept_demand(self, timestamp=None):
    return self.availability(timestamp).status in {
        GroupAvailability.AVAILABLE,
        GroupAvailability.COOLDOWN,
        GroupAvailability.REQUESTING,
    }
```

**Important consideration:** Removing AT_MAX_SLICES from the accept set means
`current_demand` will drop to 0 for a group at max_slices. The current code has
a comment (autoscaler.py:285-288) warning that this would cause immediate
scale-down of newly ready slices. However, this is handled correctly because:

1. The scheduler still assigns tasks to existing workers independently of demand
   routing. Workers run tasks regardless of what the routing layer does.
2. Scale-down safety is maintained by `scale_down_if_idle()`, which checks that
   workers have no running tasks before terminating a slice.
3. The `target_capacity` in `refresh()` uses `min(demand + buffer_slices, max_slices)`.
   When demand=0, target becomes buffer_slices. If buffer_slices=0, scale-down can
   occur but only for truly idle slices.

For groups at max_slices with active workers, the idle check prevents premature
termination. The change in demand tracking is acceptable because the routing
information is no longer actionable (no new slices can be created).

#### 1d. Simplify `can_scale_up()`

Do **not** derive `can_scale_up()` from `availability()`.

These two concepts are intentionally different:

- `availability()` exists to decide whether *routing should keep demand here* vs fall through.
- `can_scale_up()` exists to decide whether we are allowed to start *another* scale-up action.

In particular, `REQUESTING` is an ACCEPTING routing state but is not “scale-up allowed”.

**Suggested change:** keep `can_scale_up()` as a dedicated predicate (quota/backoff/cooldown/max_slices), and optionally refactor it to share helper methods with `availability()` (without coupling the semantics).

```python
def can_scale_up(self, timestamp=None):
    timestamp = timestamp or Timestamp.now()
    # Keep the explicit checks (quota/backoff/cooldown/max_slices) rather than
    # delegating to availability().
    if self._quota_exceeded_until is not None and not self._quota_exceeded_until.expired(now=timestamp):
        return False
    if self._backoff_until is not None and not self._backoff_until.expired(now=timestamp):
        return False
    cooldown_end = self._last_scale_up.add(self._scale_up_cooldown)
    if self._last_scale_up.epoch_ms() > 0 and timestamp.before(cooldown_end):
        return False
    with self._slices_lock:
        count = len(self._slices) + self._pending_scale_ups
    if count >= self._config.max_slices:
        return False
    return True
```

### 2. Bin-Pack Additive Resources in `route_demand`

**File:** `lib/iris/src/iris/cluster/controller/autoscaler.py`

The current code treats each demand entry as consuming 1 slice. This is wrong in two ways:

1. It ignores `num_vms` (multi-VM slices can host more than 1 independent task).
2. It ignores additive packing (multiple tasks can fit on one VM if CPU/memory allow).

We should change routing capacity accounting for **non-coscheduled** entries to estimate **how many VMs are needed** using a simple bin-packing heuristic. First-fit decreasing (FFD) is sufficient.

#### Packing Model

- Pack **non-coscheduled** entries into VMs using additive resources (`cpu_millicores`, `memory_bytes`).
- Treat **coscheduled** entries as consuming a full slice (one slice per entry).
- Continue to treat categorical fields (device type/variant, preemptible, region/zone) as eligibility filters.

#### What counts as a “bin”?

A bin represents a single VM with per-VM additive capacity.

**Key point:** bin packing here is used to estimate **required capacity**, not to simulate exact placement onto existing workers.

- The controller constructs demand entries from pending tasks and (for jobs with reservations) reservation entries (`controller.py:compute_demand_entries`).
- The autoscaler already compares demand-derived requirements against **existing capacity** (`ready + pending`) when deciding to scale up (`autoscaler.py:_evaluate_group`).

So the packing algorithm should answer: “How many VMs (and thus slices) would it take to fit these demand entries, given per-VM CPU/memory limits for this scale group?”

Then the autoscaler can compare `required_slices` to `capacity_slices = ready + pending/requesting` and scale up only if required exceeds current capacity and `max_slices` headroom exists.

#### Heuristic choice (clean + deterministic)

Multi-dimensional bin packing is NP-hard. We want a stable, deterministic heuristic that:

- is monotonic (more demand never yields fewer required VMs),
- is fast (runs every autoscaler loop), and
- is “good enough” to avoid systematic over-provisioning on packable workloads.

First-fit decreasing (FFD) is a standard choice. For 2D packing (CPU + memory), sort by descending memory then CPU to reduce fragmentation, then place into the first bin that fits, opening a new bin when needed.

#### 2a. Add bin packing helpers (module-level)

```python
@dataclass(frozen=True)
class AdditiveReq:
    cpu_millicores: int
    memory_bytes: int
    disk_bytes: int

def additive_req(entry: DemandEntry) -> AdditiveReq:
    """Additive (packable) resource request for one non-coscheduled entry."""
    return AdditiveReq(
        cpu_millicores=entry.resources.cpu_millicores,
        memory_bytes=entry.resources.memory_bytes,
        disk_bytes=entry.resources.disk_bytes,
    )

@dataclass
class VmBin:
    cpu_remaining: int
    memory_remaining: int
    disk_remaining: int

    def can_fit(self, req: AdditiveReq) -> bool:
        return (
            req.cpu_millicores <= self.cpu_remaining
            and req.memory_bytes <= self.memory_remaining
            and req.disk_bytes <= self.disk_remaining
        )

    def place(self, req: AdditiveReq) -> None:
        self.cpu_remaining -= req.cpu_millicores
        self.memory_remaining -= req.memory_bytes
        self.disk_remaining -= req.disk_bytes

def first_fit_decreasing(bins: list[VmBin], reqs: list[AdditiveReq]) -> int:
    """Pack requests into bins, returning the number of VMs required.

    This computes required capacity. It does not attempt to model which existing
    READY workers have free space; the controller/scheduler already tried that.
    """
    reqs_sorted = sorted(
        reqs,
        key=lambda r: (r.disk_bytes, r.memory_bytes, r.cpu_millicores),
        reverse=True,
    )
    if not bins:
        raise ValueError("bins must contain one template VmBin with per-VM capacity")
    used: list[VmBin] = []
    for req in reqs_sorted:
        for b in used:
            if b.can_fit(req):
                b.place(req)
                break
        else:
            template = bins[0]
            b = VmBin(
                cpu_remaining=template.cpu_remaining,
                memory_remaining=template.memory_remaining,
                disk_remaining=template.disk_remaining,
            )
            b.place(req)
            used.append(b)
    return len(used)
```

Notes:

- This models additive packing for CPU and memory only. Disk is currently not tracked as an “available” quantity in controller worker capacity, so disk remains an eligibility gate (`check_resource_fit`) but is not used for packing math.
- This is intentionally a heuristic. We prefer a stable, monotonic estimate to a “perfect” packer.

#### 2b. Update `PendingGroup` and `make_pending()`

Rename fields from slice-based to slot-based units:

```python
@dataclass
class PendingGroup:
    name: str
    pending_vms: int     # in-flight capacity in VMs
    remaining_vms: int   # VMs available to pack into (existing + inflight + headroom)
    assigned_entries: list[DemandEntry]
    reason: str
```

```python
def make_pending(group: ScalingGroup) -> PendingGroup:
    counts = group.slice_state_counts()
    inflight = (
        counts.get(SliceLifecycleState.REQUESTING, 0)
        + counts.get(SliceLifecycleState.BOOTING, 0)
        + counts.get(SliceLifecycleState.INITIALIZING, 0)
    )
    current = sum(counts.values())
    headroom = group.max_slices - current
    return PendingGroup(
        name=group.name,
        pending_vms=inflight * group.num_vms,
        remaining_vms=(inflight + headroom) * group.num_vms,
        assigned_entries=[],
        reason="demand-routed",
    )
```

Note: The old code had a branch for `headroom <= 0` that included ready slices
in remaining capacity. With AT_MAX_SLICES now REJECTING, a group at max_slices
will not accept demand at all (via `can_accept_demand`), so the `headroom <= 0`
branch in `make_pending` is no longer reachable via the normal `can_fit_group`
path. The only way `make_pending` is called for a group at max is through the
pending-first loop (line 330), which already has the group in the `pending`
dict from a previous cycle when it was still REQUESTING. At that point
`remaining_vms = inflight * num_vms` is correct.

#### 2c. Update `can_fit_pending()` and `assign()`

```python
def can_fit_pending(pg: PendingGroup, group: ScalingGroup, entry: DemandEntry) -> bool:
    if pg.remaining_vms <= 0:
        return False
    return can_fit_group(group, entry, check_accept=False)

def assign(pg: PendingGroup, group: ScalingGroup, entry: DemandEntry) -> None:
    # We don't decrement remaining_vms here. Bin packing determines how many VMs
    # are actually used once all entries for the group are known.
    pg.assigned_entries.append(entry)
    routed.setdefault(pg.name, []).append(entry)
```

The `assign` inner function gains a `group` parameter. Both call sites must be
updated:
- Line ~335: `assign(pg, group_by_name[name], entry)`
- Line ~349: `assign(pending[group.name], group, entry)`

#### 2d. Compute required VMs via packing, then derive `group_to_launch`

For each group, compute:

- `existing_bins`: bins representing available capacity on existing READY workers (preferred), plus bins for inflight workers, plus bins for headroom workers.
- `reqs`: additive requests for all assigned **non-coscheduled** entries.
- `coscheduled_count`: number of assigned coscheduled entries (each consumes 1 slice).

Then:

1. Run FFD packing of `reqs` into per-VM bins to compute `required_vms`.
2. Compute `required_slices_for_noncsc = ceil(used_vms / num_vms)`.
3. Compute `required_slices_for_csc = coscheduled_count` (each needs its own slice).
4. Total required slices = `required_slices_for_noncsc + required_slices_for_csc`.
5. Additional slices to launch (diagnostic) = `max(0, total_required_slices - (ready_slices + inflight_slices))`.

The important difference from the “1 entry = 1 VM” approximation is step (1): we pack by CPU/memory, so (for example) four `32GiB` tasks can fit on one `128GiB` VM.

```python
group_to_launch: dict[str, int] = {}
for name, pg in pending.items():
    if not pg.assigned_entries:
        continue
    group = group_by_name[name]
    # 1) Split assigned entries into coscheduled vs non-coscheduled.
    # 2) Build initial bins from existing READY workers (preferred) + inflight + headroom.
    # 3) Pack additive reqs via FFD to estimate VMs needed.
    # 4) Convert VMs to slices and subtract existing slice count to compute launch.
    ...
```

### 3. Fix `_evaluate_group` to Use Packed Capacity

**File:** `lib/iris/src/iris/cluster/controller/autoscaler.py`

Currently `_evaluate_group` compares `demand` (number of entries) to `capacity`
(number of slices). With packing, `_evaluate_group` should compare slices to slices:

- compute `required_slices` from packing for that group’s currently-routed demand (as in 2d), and
- compare to existing slices (ready + pending/requesting).

**Important:** if `current_demand` is redefined to mean “required VMs”, then *every* place that displays or logs demand must reflect that unit (dashboard text, status protos, reason strings).

```python
def _evaluate_group(self, group, required_slices, ts):
    """Evaluate scaling for a group. required_slices is derived from packing."""
    counts = group.slice_state_counts()
    ready = counts[SliceLifecycleState.READY]
    pending = (
        counts[SliceLifecycleState.BOOTING]
        + counts[SliceLifecycleState.INITIALIZING]
        + counts[SliceLifecycleState.REQUESTING]
    )
    total = sum(counts.values())
    capacity_slices = ready + pending

    target = min(required_slices + group.buffer_slices, group.max_slices)
    demand_gap = max(0, required_slices - pending)
    buffer_gap = max(0, target - total)
    slices_needed = max(demand_gap, buffer_gap)

    if slices_needed > 0 and total < group.max_slices:
        if not group.can_scale_up(ts):
            return []
        slices_to_add = min(slices_needed, group.max_slices - total)
        return [ScalingDecision(
            scale_group=group.name,
            action=ScalingAction.SCALE_UP,
            reason=f"required_slices={required_slices} > capacity_slices={capacity_slices}",
        )
    return None
```

Caller in `evaluate()`:

```python
for name, group in self._groups.items():
    allocated = result.routed_entries.get(name, [])
    # Compute required_slices via (2d) packing for this group's allocated entries.
    required_vms, required_slices = pack_required_capacity(group, allocated, ts)
    group.update_demand(required_vms)  # interpret current_demand as "required VMs"
    decision = self._evaluate_group(group, required_slices, ts)
```

### 4. Fix Scale-Down Target Capacity

**File:** `lib/iris/src/iris/cluster/controller/autoscaler.py`, `refresh()`

`scale_down_if_idle()` compares `target_capacity` against slice counts (ready +
pending). If `current_demand` is updated to “required VMs”, convert it to slices:

```python
for group in self._groups.values():
    required_vms = group.current_demand
    required_slices = (required_vms + group.num_vms - 1) // group.num_vms if required_vms > 0 else 0
    target_capacity = min(group.current_demand + group.buffer_slices, group.max_slices)
    scaled_down = group.scale_down_if_idle(vm_status_map, target_capacity, timestamp)
```

### 5. Update Dashboard Routing Status

**File:** `lib/iris/src/iris/cluster/controller/autoscaler.py`

Add COOLDOWN and AT_MAX_SLICES to the `group_statuses` loop:

```python
elif availability.status == GroupAvailability.COOLDOWN:
    decision = "cooldown"
    reason = availability.reason
elif availability.status == GroupAvailability.AT_MAX_SLICES:
    decision = "blocked"
    reason = "at max_slices"
```

### 6. Rename All AT_CAPACITY → AT_MAX_SLICES References

| File | Location |
|------|----------|
| `scaling_group.py` | Enum value (line 64), availability() docstring (line 738), availability() return (line 770), can_accept_demand() docstring (line 778), can_accept_demand() set (line 785) |
| `autoscaler.py` | make_pending comment (line 285), group_statuses check (line 380) |
| `test_scaling_group.py` | `test_at_capacity_when_at_max_slices` docstring/assertion (lines 751, 769), `test_can_accept_demand_true_when_at_capacity` docstring (line 792) |
| `docs/autoscaler-improvements.md` | Line 56 |

## Testing

### New Tests — `test_autoscaler.py`

1. **`test_packing_allows_multiple_tasks_per_vm`**: Group with `num_vms=64`, 1 slice, per-VM memory=128GiB.
   Submit 128 CPU demand entries each requesting 32GiB. Packing fits 4 per VM → 32 VMs → required_slices=1 → `group_to_launch` = 0.

2. **`test_packing_prevents_unnecessary_overflow`**: Group A (priority=10, larger VMs) and B (priority=20).
   Submit CPU demand entries that pack within A’s VMs. Verify routing/launch stays on A and does not spill to B.

3. **`test_cooldown_does_not_cause_fallthrough`**: Groups A (priority=10,
   cooldown=5min) and B (priority=20). Scale up A once, put it in cooldown.
   Submit demand. Verify routes to A, not B.

4. **`test_backoff_causes_fallthrough`**: Same setup. Put A in backoff.
   Verify demand routes to B.

5. **`test_at_max_slices_causes_fallthrough`**: Group A at `max_slices`, group B
   available. Verify demand falls through from A to B.

6. **`test_evaluate_uses_packed_capacity`**: Group `num_vms=64`, 1 ready slice, per-VM memory=128GiB.
   128×32GiB entries → required_slices=1 → no scale-up.
   1025×32GiB entries → required_vms=257 → required_slices=5 → scale-up triggered when capacity_slices < 5.

7. **`test_launch_count_derived_from_packing`**: Vary entry sizes so packing requires a known VM count, then verify `group_to_launch` uses `ceil(required_vms/num_vms)` (plus coscheduled slices).

8. **`test_scale_down_target_uses_required_vms`**: Group `num_vms=4`, 2 ready slices.
   required_vms=5 → target_slices=2 → no scale-down.
   required_vms=3 → target_slices=1 → scale-down one idle.

### New Tests — `test_scaling_group.py`

9. **`test_cooldown_availability_state`**: After scale-up + complete, verify
   `availability()` returns COOLDOWN until expiry, then AVAILABLE.

10. **`test_can_scale_up_only_when_available`**: Verify returns True only
    for AVAILABLE, False for COOLDOWN/BACKOFF/QUOTA_EXCEEDED/AT_MAX_SLICES.

### Existing Tests to Update

- Rename `AT_CAPACITY` → `AT_MAX_SLICES` everywhere in tests
- `test_at_capacity_when_at_max_slices` → update docstring and assertion to use
  `GroupAvailability.AT_MAX_SLICES`
- `test_can_accept_demand_true_when_at_capacity` → this test's assertion
  **changes behavior**: AT_MAX_SLICES now rejects demand. The test should be
  renamed and updated to verify `can_accept_demand() is False` when at max
  slices.
- `test_demand_overflows_to_lower_priority_when_at_capacity` → behavior is
  preserved (overflow still happens), just rename references
- Update any `PendingGroup` field assertions from `remaining_slices` →
  `remaining_vms`, `pending_slices` → `pending_vms`
- `_evaluate_group` tests asserting on reason strings should be updated to the new
  `required_slices=... > capacity_slices=...` format (or equivalent).

### Verification

```bash
cd lib/iris
uv run pytest tests/cluster/controller/test_autoscaler.py -x -v
uv run pytest tests/providers/test_scaling_group.py -x -v
uv run pytest -m 'not slow' -x
```

## Files Modified

| File | Changes |
|------|---------|
| `scaling_group.py` | Add COOLDOWN, rename AT_CAPACITY→AT_MAX_SLICES, update availability/can_scale_up/can_accept_demand |
| `autoscaler.py` | Bin packing helpers, PendingGroup field renames, packed-capacity make_pending/launch/evaluate/refresh, dashboard status |
| `test_autoscaler.py` | New tests 1-8, update existing tests for renames/packing |
| `test_scaling_group.py` | New tests 9-10, rename AT_CAPACITY, update can_accept_demand_at_capacity test |
| `docs/autoscaler-improvements.md` | Rename AT_CAPACITY reference |

## Execution Order

### Step 1: COOLDOWN State + AT_CAPACITY Rename (No Capacity Math Changes)

1. Update `GroupAvailability` enum (add COOLDOWN, rename AT_CAPACITY→AT_MAX_SLICES)
2. Update `availability()`, `can_scale_up()`, `can_accept_demand()`
3. Rename all AT_CAPACITY references in code and tests
4. Update `test_can_accept_demand_true_when_at_capacity` to reflect new
   behavior (AT_MAX_SLICES rejects)
5. Add tests 3, 4, 5, 9, 10
6. Run `test_scaling_group.py` and `test_autoscaler.py`

### Step 2: Additive Resource Packing

1. Add packing helpers (FFD) and a `pack_required_capacity(...)` helper for a group
2. Plumb existing worker available CPU/memory into packing bins (preferred); otherwise implement a conservative fallback (idle-only capacity)
3. Update `PendingGroup` fields (`remaining_slices`→`remaining_vms`, `pending_slices`→`pending_vms`)
4. Update `make_pending`, `assign`, and `group_to_launch` to derive slices to launch from packed required VMs (plus coscheduled slices)
5. Update `_evaluate_group` and `evaluate()` to use `required_slices`
6. Update `refresh()` target_capacity using `required_vms` → slices
7. Update dashboard status for COOLDOWN/AT_MAX_SLICES and update demand-unit labels (required VMs)
8. Add tests 1, 2, 6, 7, 8 and update existing tests for renamed fields/reason strings
9. Run full test suite

---

## Addendum: Plan Review

### Correctness Issues

**AT_MAX_SLICES demand rejection and scale-down safety.** The most critical
behavioral change is removing AT_MAX_SLICES from the acceptance set. The plan
acknowledges the risk (the existing comment at autoscaler.py:285-288) and argues
it's safe because `scale_down_if_idle` checks worker activity. This is correct:
the idle check in `_verify_slice_idle` requires at least one known worker to be
non-idle, and `update_slice_activity` keeps `last_active` current for busy
slices. However, there's a subtle gap: **between the time demand drops to 0 and
the next `run_once` cycle, `target_capacity` will be `min(0 + buffer_slices, max_slices)`**.  If
`buffer_slices=0`, the autoscaler will attempt to scale down on every cycle,
relying solely on the idle check. This is fine for slices with active workers,
but could cause premature termination of slices that just became ready and
haven't received tasks yet (their `last_active` was set at mark_ready, so
they'd need `idle_threshold` minutes to elapse — the default 5 minutes is
usually sufficient, but worth confirming in tests).

**Recommendation:** Test 8 should include a case where demand drops to 0 for a
group at max_slices with active workers, confirming no scale-down occurs.

### Potential Gaps

1. **`make_pending` branch removal.** The plan's `make_pending` no longer has
   the `headroom <= 0` branch that included ready slices. The plan claims this
   path is unreachable when AT_MAX_SLICES rejects demand. But there's still the
   pending-first loop (line 330) that checks `pending` dict entries created in
   earlier iterations. A group could be in `pending` from a REQUESTING state
   that transitions to AT_MAX_SLICES between cycles. This edge case should be
   tested or the code should defensively handle `headroom <= 0`.

   **Recommendation:** Add a defensive `max(0, ...)` for remaining_vms in
   make_pending, or at least add a comment + test.

2. **`update_demand` unit change.** `update_demand(demand)` currently stores
   demand as an integer with implicit "number of entries" semantics. After
   step 2, the plan proposes storing **required VMs** (derived from packing).
   All consumers of `current_demand` and `peak_demand` need to be aware of the
   unit change. The `to_status()` method exposes `current_demand` in the proto —
   dashboard consumers may interpret this differently.

   **Recommendation:** treat the unit change as an API change:
   - update dashboard labels / tooltips to say “required VMs” (or introduce parallel fields),
   - update reason strings (`required_slices=...`),
   - update any tests that implicitly interpret demand as “entries”.

3. **Coscheduled entries and packing.** Coscheduled demand cannot be packed and
   should be treated as “1 slice per coscheduled entry.” The existing code
   already enforces `group.num_vms == len(entry.task_ids)` in `can_fit_group`,
   so “1 slice per entry” is consistent.

### Style / Approach

- The two-step spiral execution order is well-chosen. Separating the enum/state
  changes from the math changes reduces blast radius per step.
- The plan correctly identifies all AT_CAPACITY references that need renaming.
- Keeping routing availability separate from `can_scale_up()` avoids conflating
  “accept demand for attribution” with “allowed to start a new scale-up”.

### Summary

The plan is solid and well-structured. The main risk is the AT_MAX_SLICES
rejection causing unexpected scale-down for groups with `buffer_slices=0` and
recently-ready-but-not-yet-assigned slices. The `idle_threshold` default of 5
minutes provides adequate protection, but this should be explicitly tested.
The `make_pending` branch removal and `update_demand` unit change are minor
loose ends that should be addressed during implementation.
