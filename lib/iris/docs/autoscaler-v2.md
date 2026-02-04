# Autoscaler v2 Design

## Context

This document proposes a simplified autoscaler design for Iris. It responds to issues about confusing autoscaler behavior and unclear dashboard visibility for scaling decisions and slice states. The goal is to make scaling decisions explainable, the UI inspectable, and the routing logic straightforward.

See:
- Issue #2580: clarify/simplify autoscaler logic and surface decisions in the dashboard
- Issue #2648: show slice lifecycle states (requesting -> initializing -> ready -> failed)

## Goals

- Make scale-up decisions deterministic and explainable, with clear logs and dashboard state.
- Make demand routing explicit and simple, with a minimal algorithm.
- Track unsatisfied demand and surface it to the user.
- Preserve existing responsibilities: autoscaler decides scale up, scaling groups manage per-slice lifecycle.

## Non-Goals

- No consolidation or packing across demand entries in this first version.
- No new scale-down behavior beyond existing idle logic.
- No inference about task duration or cost optimization.

## Current Behavior (v0)

The autoscaler currently:
- Accepts `DemandEntry` with `device_type`, `device_variant`, `count`, and preemptible preference.
- Routes demand to groups by accelerator type/variant, preemptible, and priority. This produces `allocations` per group.
- For each group, computes a single `ScalingDecision` if demand exceeds capacity or min_slices is violated.
- Executes scale-up in a background thread (`scale_up` per group) and relies on group-level state for scale-down.

Important details in `lib/iris/src/iris/cluster/vm/autoscaler.py`:
- The routing logic is `route_demand`, which chooses groups by matching device type/variant and preemptible, then uses priority and available headroom.
- The evaluate step only produces at most one scale-up per group per tick.
- `DemandEntry.total_cpu` and `.total_memory_bytes` exist but are unused in routing.

This is conceptually simple but lacks:
- Fine-grained demand-to-group resource matching based on attribute constraints.
- A clear model for “pending” capacity and in-flight slices.
- A way to communicate to users why demand is not being satisfied.

## Proposed Design (v2)

### Key Idea

Demand is a list of `DemandEntry` objects, each describing unscheduled work and its constraints. We do not attempt to consolidate or pack demand across unrelated tasks; each entry is processed independently. For coscheduled jobs, we route the entire coscheduled group together as a single demand unit (mirroring scheduler behavior). We allocate demand to scale groups using a simple two-phase routing process:

1. Prefer existing pending groups (in-flight or already provisioned) that can satisfy the demand.
2. If no pending group matches, select the first matching scale group by priority and mark it as pending.

This makes behavior predictable and explainable, and it mirrors the scheduler’s “deduct capacity” model at the scale-group level.

### Data Model

Extend `DemandEntry` (or derive a new type) to include:
- `device_type` and optional `device_variant`.
- `constraints`: list of attribute constraints (reuse `cluster_pb2.Constraint` if it covers needs).
- `resources`: cpu, memory, and device-specific resource requirements.
- `preemptible` preference (optional).
- `coschedule_group_id`: identifier for coscheduled tasks (None for non-coscheduled entries).
- `task_ids`: list of task ids included in this demand entry (size 1 for non-coscheduled).

Introduce a `PendingGroup` accounting model:
- Tracks scale group name.
- Tracks remaining capacity for the group (cpu, memory, accelerators) as demand entries are assigned.
- Tracks the reason why the group is pending (first demand entry assigned, min_slices, etc.).

Scale group capacity is defined per VM via `ScaleGroupConfig.resources`:
- `cpu`, `memory_bytes`, `disk_bytes`, `gpu_count`, `tpu_count`
- `slice_size` is a separate field indicating how many tasks comprise a coscheduled slice.

### Algorithm

Given `demand_entries` (each entry represents one unit of unscheduled work):

1. Build `pending_groups` as an empty list/map.
2. For each demand entry in order:
   - Check `pending_groups` first. If a pending group satisfies all constraints and has enough remaining capacity, assign demand to it and deduct capacity.
   - If none matches, iterate scale groups (sorted by priority) and choose the first that satisfies constraints and has capacity headroom.
   - When a group is selected, add it to `pending_groups` (if not already present) and deduct capacity.
   - If no group matches, record the demand entry as `unmet` with a reason.
3. At the end, autoscaler has:
   - `pending_groups`: scale groups that need a scale-up action, along with the number of slices needed (derived from demand count and capacity headroom).
   - `unmet_demand`: list of demand entries with a reason code.

### Notes on Capacity Deduction

This mirrors scheduler logic but at scale-group granularity:
- `pending_groups` represents the planned capacity we expect to exist after scale-up.
- Capacity is deducted as demand entries are assigned to groups.
- This is not exact consolidation; it is a simple accounting model to avoid double-counting demand and to explain why a group was selected.

Pending capacity should include only slices that are in-flight (`REQUESTING`, `BOOTING`, `INITIALIZING`),
not slices that are `READY` but not yet registered with the scheduler. `READY` slices are accounted
for as normal capacity, not as pending.

For TPU multi-host pods, represent capacity the same way the scheduler does: coscheduled tasks
are grouped into a single demand entry (“pack”), and the pack consumes one slice worth of TPU
capacity. This avoids partial allocation of a pod and keeps routing aligned with scheduler semantics.

### Handling In-Flight Slices

The pending set includes any scale group that:
- Is already in `requesting`, `booting`, or `initializing` state.
- Has an in-flight scale-up action in the action log.

This ensures the autoscaler uses already-provisioning capacity before starting new slices.

### Slice Lifecycle States

Add an explicit `REQUESTING` state (if not already present) to represent:
- The autoscaler has asked the VM manager for a slice
- The slice has not yet been created or returned by the provider

Lifecycle state ordering is:
`REQUESTING -> BOOTING -> INITIALIZING -> READY -> (FAILED | TERMINATED)`

### Logging and Dashboard Visibility

Expose and log:
- Decisions per demand entry: which group was chosen and why.
- `pending_groups` with remaining capacity and number of assigned demand entries.
- `unmet_demand` entries with a reason code.
- Slice lifecycle states per group: requesting, initializing, ready, failed, terminated.

The autoscaler decision output should include:
- Which scale groups to launch (and how many slices).
- Which demand entries (and task ids) were routed to each scale group.
- Which demand entries were not scheduled (with reason codes).
This output is the primary source for the dashboard and user-facing diagnostics.

### Detailed Pseudocode

Below is a detailed, step-by-step pseudocode sketch. It is intended to be explicit about
inputs, outputs, and the internal accounting used for pending capacity. It avoids
backwards compatibility behavior: any new data required by the algorithm must be
added to the demand model, and all call sites updated.

```python
# Types:
# DemandEntry:
#   - task_ids: list[str]
#   - coschedule_group_id: str | None
#   - device_type: DeviceType
#   - device_variant: str | None
#   - constraints: list[cluster_pb2.Constraint]
#   - resources: ResourceSpec
#   - preemptible: bool | None
#
# PendingGroup:
#   - name: str
#   - remaining_cpu: int
#   - remaining_memory: int
#   - remaining_gpus: int
#   - remaining_tpus: int
#   - remaining_slices: int
#   - assigned_entries: list[DemandEntry]
#   - reason: str
#
# RoutingDecision:
#   - group_to_launch: dict[str, int]  # group -> slices to launch
#   - routed_entries: dict[str, list[DemandEntry]]  # group -> demand entries
#   - unmet_entries: list[UnmetDemand]
#
# UnmetDemand:
#   - entry: DemandEntry
#   - reason: str

def build_demand_entries(unscheduled_tasks):
    # Group coscheduled tasks together (mirror scheduler behavior).
    # Tasks without coschedule_group_id become single-entry demand.
    groups = {}
    for task in unscheduled_tasks:
        gid = task.coschedule_group_id
        if gid is None:
            groups[task.task_id] = [task]
        else:
            groups.setdefault(gid, []).append(task)

    entries = []
    for _, tasks in groups.items():
        # All tasks in a group share resource requirements and constraints.
        # If they do not, fail fast and mark unmet with a reason.
        entry = DemandEntry(
            task_ids=[t.task_id for t in tasks],
            coschedule_group_id=tasks[0].coschedule_group_id,
            device_type=tasks[0].device_type,
            device_variant=tasks[0].device_variant,
            constraints=tasks[0].constraints,
            resources=tasks[0].resources,
            preemptible=tasks[0].preemptible,
        )
        entries.append(entry)
    return entries

def route_demand(groups, demand_entries, timestamp):
    # Pre-sort scale groups once by priority.
    sorted_groups = sorted(groups, key=lambda g: g.config.priority or 100)

    pending = {}  # name -> PendingGroup
    routed = {}   # name -> list[DemandEntry]
    unmet = []    # list[UnmetDemand]

    def can_fit(capacity_view, entry):
        # capacity_view may be a scale group or a pending group. The interface is the same:
        # - matches_device_requirement
        # - preemptible
        # - can_accept_demand(timestamp)
        # - remaining_* resource fields
        # - remaining_slices
        if not capacity_view.matches_device_requirement(entry.device_type, entry.device_variant):
            return False
        if entry.preemptible is not None and capacity_view.preemptible != entry.preemptible:
            return False
        if not capacity_view.can_accept_demand(timestamp):
            return False
        if entry.resources.cpu > capacity_view.remaining_cpu:
            return False
        if entry.resources.memory_bytes > capacity_view.remaining_memory:
            return False
        if entry.resources.gpu_count > capacity_view.remaining_gpus:
            return False
        if entry.resources.tpu_chip_count > capacity_view.remaining_tpus:
            return False
        if capacity_view.remaining_slices <= 0:
            return False
        return True

    def deduct(capacity_view, entry):
        capacity_view.remaining_cpu -= entry.resources.cpu
        capacity_view.remaining_memory -= entry.resources.memory_bytes
        capacity_view.remaining_gpus -= entry.resources.gpu_count
        capacity_view.remaining_tpus -= entry.resources.tpu_chip_count
        capacity_view.assigned_entries.append(entry)

    def make_pending_group(group):
        # Headroom in slices minus already reserved slices.
        counts = group.slice_state_counts()
        current = sum(counts.values())
        headroom = group.max_slices - current

        return PendingGroup(
            name=group.name,
            # CPU and RAM should always be known and are required.
            remaining_cpu=group.capacity_cpu,
            remaining_memory=group.capacity_memory_bytes,
            # If device-specific capacity is unknown, treat as 0.
            # This forces routing to rely on device_type/variant and slices only.
            remaining_gpus=group.capacity_gpus or 0,
            remaining_tpus=group.capacity_tpus or 0,
            # Pending capacity only includes in-flight slices (REQUESTING/BOOTING/INITIALIZING).
            remaining_slices=counts["requesting"] + counts["booting"] + counts["initializing"],
            assigned_entries=[],
            reason="demand-routed",
        )

    for entry in demand_entries:
        # 1) Prefer pending groups.
        matched_pending = False
        for pg in pending.values():
            if can_fit(pg, entry):
                deduct(pg, entry)
                routed.setdefault(pg.name, []).append(entry)
                matched_pending = True
                break
        if matched_pending:
            continue

        # 2) Find a new group by priority.
        matched_group = False
        for group in sorted_groups:
            if not can_fit(group, entry):
                continue
            if group.name not in pending:
                pending[group.name] = make_pending_group(group)
            pg = pending[group.name]
            if can_fit(pg, entry):
                deduct(pg, entry)
                routed.setdefault(pg.name, []).append(entry)
                matched_group = True
                break

        if not matched_group:
            unmet.append(UnmetDemand(entry=entry, reason="no_matching_group"))

    # Compute scale-up plan: count how many slices to launch from pending groups.
    group_to_launch = {}
    for name, pg in pending.items():
        # If there are assigned entries, scale up at least one slice.
        # If remaining_slices already covers demand, scale up may be zero.
        if pg.assigned_entries:
            needed = max(1, len(pg.assigned_entries) - pg.remaining_slices)
            group_to_launch[name] = max(0, needed)

    return RoutingDecision(
        group_to_launch=group_to_launch,
        routed_entries=routed,
        unmet_entries=unmet,
    )
```

These should be surfaced via existing autoscaler status RPC and dashboard rendering.

## Comparison Summary

Current behavior:
- Demand routing is per-device-type count, not per-demand entry.
- Priority + headroom routing only considers slice count, not detailed constraints.
- The autoscaler may start new scaling groups even while adoption/inflight is ongoing, which is confusing to users.

Proposed behavior:
- Demand is processed entry-by-entry, matching constraints, with a clear pending set.
- Pending/in-flight groups are preferred for subsequent demand entries.
- Unmet demand is explicit and visible, with reason codes.

## Slotting Into Controller Flow

This design replaces the existing routing logic (no v2 flag or backward compatibility). The call flow remains the same, but the content of `demand_entries` and the autoscaler evaluation changes:

1. `Controller._run_autoscaler_once()` calls `compute_demand_entries(state)`.
2. `compute_demand_entries` will be updated to return *per-demand-entry* objects, including:
   - `task_ids` (1 for non-coscheduled, N for coscheduled jobs)
   - constraints and resource specs (from the job/task)
   - device type/variant and preemptible preference
3. `Autoscaler.run_once(...)` calls `evaluate(...)`.
4. `evaluate(...)` calls the new `route_demand(...)` (replacing the current `route_demand` implementation).
5. `evaluate(...)` stores the `RoutingDecision` (see below) on the autoscaler for status reporting.
6. `execute(...)` launches the scale groups indicated by `RoutingDecision.group_to_launch`.

The scheduler loop and the autoscaler invocation points in
`lib/iris/src/iris/cluster/controller/controller.py` stay the same; only the demand
construction and routing algorithm change.

## Status and Dashboard Reporting

The dashboard already fetches data via the Controller RPCs (no REST layer). The autoscaler status
response should be extended to include the most recent `RoutingDecision`, and the autoscaler should
cache the last decision for display. This allows the UI to render:

- The list of scale groups to launch (with slice counts).
- The mapping of demand entries (task ids) to scale groups.
- The list of unscheduled demand entries with reason codes.

Suggested API flow:

1. `Autoscaler.evaluate(...)` saves `self._last_routing_decision`.
2. `Autoscaler.get_status()` includes a `last_routing_decision` field in `vm_pb2.AutoscalerStatus`.
3. The dashboard’s autoscaler tab reads this via `GetAutoscalerStatus` and renders it.

Because the dashboard is driven by RPCs, the only server-side change needed is to populate the
status proto. The JS client can then render it without further Python changes.

## Open Questions

- How should we validate that all tasks in a coschedule group have identical resource/constraint requirements, and how should we report mismatches to the user?

## Implementation Sketch

Suggested code changes:
- Extend `DemandEntry` to include `constraints` and `resources`.
- Add a `PendingGroup` struct in `autoscaler.py` for capacity tracking.
- Replace `route_demand` with a new `route_demand_v2` that processes entries with the pending-first strategy.
- Add structured log events for each demand entry routing decision.
- Update `vm_pb2.AutoscalerStatus` and dashboard to include per-slice state and unmet demand reasons.

## Migration Strategy

- Implement v2 routing behind a config flag or new autoscaler mode.
- Validate with the smoke test and with synthetic demand scenarios.
- Update dashboard to show the new per-group and per-slice visibility.
