# Iris Autoscaler Refactor Plan

## Problem

`lib/iris/src/iris/cluster/controller/autoscaler.py` currently mixes five different concerns:

- Pure demand routing and capacity estimation (`route_demand()` at `autoscaler.py:645`)
- Scale-up decision derivation (`_evaluate_group()` at `autoscaler.py:996`)
- Cloud lifecycle polling and idle scale-down (`refresh()` at `autoscaler.py:1204`)
- Worker registry and operator helpers (`restart_worker()` at `autoscaler.py:1305`)
- Status/debug projection (`_routing_decision_to_proto()` at `autoscaler.py:1548`)

That makes the code hard to reason about because the main invariants are split across files:

- Demand generation happens in `controller.py:241`
- Live worker capacity is modeled in `scheduler.py:152`
- Group availability and cooldown/backoff state live in `scaling_group.py:991`
- Pending-job UI messaging consumes routing status in `pending_diagnostics.py:84`

The most confusing point is that `route_demand()` computes `group_to_launch` (`autoscaler.py:774`), but actual scale-up execution ignores that value and instead recomputes launch decisions in `_evaluate_group()` (`autoscaler.py:996`). This creates two overlapping notions of "how much to launch", one for status and one for execution.

Current behavior is defensible, but the structure is not. The file is carrying runtime orchestration, planning, diagnostics, restore logic, and API projection in one place.

Backwards compatibility: this plan preserves current routing semantics, quota-pool tier blocking, cooldown/backoff behavior, and public status fields unless explicitly renamed behind a compatibility shim.

## Goals

- Separate pure planning logic from runtime side effects
- Establish one canonical source of truth for scale-up counts
- Make routing, availability gating, and diagnostics readable without tracing the whole controller
- Preserve current scheduling and autoscaling semantics while reducing cognitive load
- Improve testability of routing and scale-up decisions

Non-goals:

- Changing quota-pool monotonicity semantics
- Reworking scheduler dry-run behavior in `controller.py`
- Introducing backward-compatibility layers for internal Python APIs unless needed for incremental landing
- Fixing all known routing-policy issues in the same refactor

## Proposed Solution

Split the current file into three modules plus one small status adapter:

- `autoscaler_runtime.py`
  Owns the `Autoscaler` class, `refresh()`, `update()`, `execute()`, worker tracking, restore, and operator helpers.
- `demand_router.py`
  Owns `DemandEntry`, packing helpers, routing budgets, `route_demand()`, unmet reasons, and quota-pool tier filtering.
- `scaling_plan.py`
  Owns the canonical conversion from routed demand plus group state into actionable scale-up counts.
- `routing_status.py`
  Owns user-facing projection helpers and proto/status shaping.

The key architectural change is to replace the current `group_to_launch` hint with a canonical `ScalePlan` produced once and consumed both by execution and by status rendering.

```python
@dataclass(frozen=True)
class GroupScalePlan:
    group: str
    required_slices: int
    target_slices: int
    pending_slices: int
    total_slices: int
    slices_to_add: int
    reason: str


@dataclass(frozen=True)
class ScalePlan:
    routing: RoutingDecision
    groups: dict[str, GroupScalePlan]


def build_scale_plan(groups: list[ScalingGroup], demand_entries: list[DemandEntry], ts: Timestamp) -> ScalePlan:
    routing = route_demand(groups, demand_entries, ts)
    group_plans = {
        group.name: evaluate_group_scale(group, routing.group_required_slices.get(group.name, 0), ts)
        for group in groups
    }
    return ScalePlan(routing=routing, groups=group_plans)
```

This removes the current split-brain:

- `route_demand()` remains responsible for "where demand would go"
- `evaluate_group_scale()` becomes the only place that answers "how many slices should we add now"
- status code renders `ScalePlan.groups[*].slices_to_add` instead of inferring meaning from `RoutingDecision.group_to_launch`

### Module Boundaries

`demand_router.py`

- `DemandEntry`
- `AdditiveReq`
- `VmBin`
- `RoutingBudget`
- `UnmetDemand`
- `RoutingDecision`
- `GroupRoutingStatus`
- `first_fit_decreasing()`
- `route_demand()`
- `_diagnose_no_matching_group()`
- `_diagnose_no_capacity()`
- quota-pool tier helpers

`scaling_plan.py`

- `GroupScalePlan`
- `ScalePlan`
- `evaluate_group_scale()`
- `build_scale_plan()`

`routing_status.py`

- Routing/proto conversion helpers now in `_routing_decision_to_proto()`
- pending-diagnostic helpers currently in `pending_diagnostics.py`
- optional rename of `group_to_launch` to `launch_hint` in the proto later, after call sites are migrated

`autoscaler_runtime.py`

- `Autoscaler`
- refresh / execute / restore / restart-worker logic
- `TrackedWorker` and restored handle types

### Canonical Invariants

After the refactor, the code should enforce these invariants explicitly:

1. Routing answers assignment feasibility by group, not launch count.
2. Scale-up count is derived exactly once from `required_slices`, `pending`, `total`, and buffer.
3. Status/UI code never recomputes control-plane decisions from partial routing fields.
4. Runtime/cloud code does not depend on routing internals like `VmBin`.
5. Tests assert policy at the appropriate layer:
   routing tests for placement, scale-plan tests for launch counts, runtime tests for side effects.

## Implementation Outline

1. Extract pure routing code from [`autoscaler.py`](/Users/power/code/marin/lib/iris/src/iris/cluster/controller/autoscaler.py) into `demand_router.py` without semantic changes, keeping imports and tests green.
2. Introduce `GroupScalePlan` and `ScalePlan`, move `_evaluate_group()` logic into `scaling_plan.py`, and make `Autoscaler.evaluate()` consume `ScalePlan`.
3. Remove `group_to_launch` as an execution input and make status/pending diagnostics read canonical `slices_to_add` data.
4. Move runtime-only behavior (`refresh`, restore, worker tracking, restart) into `autoscaler_runtime.py` so the orchestrator file contains only loop control.
5. Split tests by layer: routing tests stay pure, add focused scale-plan tests, and keep runtime tests around `run_once()` and quota/backoff transitions.
6. After behavior is preserved, simplify names and delete dead compatibility helpers such as duplicate required-slice computations that are no longer used.

## Notes

### Landing order

The safest landing sequence is:

1. Move code without behavior changes.
2. Introduce `ScalePlan` while preserving old status fields.
3. Switch all internal readers to `ScalePlan`.
4. Delete redundant fields and helpers.

This avoids a risky "rewrite in place" and keeps each diff reviewable.

### `group_to_launch` should become status-only or go away

Today `RoutingDecision.group_to_launch` looks authoritative, but it is only a routing-time estimate computed in `route_demand()` (`autoscaler.py:774`) and is not what `execute()` uses. That field either needs to:

- become a clearly named status hint such as `launch_hint`, or
- be replaced with `GroupScalePlan.slices_to_add`

The second option is cleaner.

### Keep quota-pool policy isolated

The tier monotonicity logic in `autoscaler.py:614` and `autoscaler.py:633` is small but high-impact. It should remain pure and be grouped under a named policy section or helper module so readers can see that it is a policy gate, not an incidental filter.

### Keep routing pure

`route_demand()` should remain free of DB access, cloud calls, thread state, and worker-handle mutation. That function is the easiest part of the autoscaler to test and reason about; the refactor should strengthen that property rather than weaken it.

### Recommended file sketch

```python
# autoscaler_runtime.py
class Autoscaler:
    def run_once(...): ...
    def refresh(...): ...
    def update(...):
        plan = build_scale_plan(list(self._groups.values()), demand_entries, timestamp)
        self._last_scale_plan = plan
        self.execute(scale_plan_to_decisions(plan), timestamp)
```

## Future Work

- Revisit whether routing budgets for READY slices should seed from residual worker capacity instead of nominal per-VM capacity
- Rename or redesign the routing proto to better separate control-plane decisions from diagnostics
- Collapse duplicated status generation between autoscaler status APIs and pending diagnostics
- Consider a dedicated `QuotaPoolPolicy` type if more pool-level rules are added
- Audit whether `compute_required_slices()` should remain public or become an internal helper used only by routing tests
