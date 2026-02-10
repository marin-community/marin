# Design: Autoscaler Improvements

N.B. This design doc was machine generated and is primarily used for agent task tracking.
DO NOT use it as a canonical source of information about the project.

## Problem

The Iris autoscaler has two issues affecting controller responsiveness:

1. **Blocking scale-up**: When `_execute_scale_up()` (autoscaler.py:361) triggers `ScalingGroup.scale_up()` (scaling_group.py:174), it synchronously calls `VmManager.create_vm_group()`, which for TPUs blocks on a `subprocess.run()` gcloud call for 10-60+ seconds. This blocks the entire scheduling loop (`controller.py:350`), preventing task dispatch and worker timeout checks.

2. **Excessive evaluation**: The autoscaler runs every scheduling cycle (0.5s default) via `_run_autoscaler_once()` (controller.py:616). `AutoscalerConfig.evaluation_interval_seconds` exists (autoscaler.py:98) and `_last_evaluation_ms` is tracked (autoscaler.py:187), but the interval is never enforced.

## Goals

- Unblock the scheduling thread from VM creation latency
- Rate-limit autoscaler evaluation to a configurable interval
- Add autoscaler config to `config.proto` so it's YAML-configurable
- Introduce a REQUESTING group state so demand isn't double-routed to groups with pending scale-ups

**Non-goals**: Changing VmManager interfaces, parallel multi-group scale-up, changing scale-down behavior

## Proposed Solution

### Part 1: Rate-Limiting Autoscaler Evaluation

Enforce `evaluation_interval_seconds` in `run_once()`:

```python
def run_once(self, demand_entries, vm_status_map, timestamp_ms=None):
    timestamp_ms = timestamp_ms or now_ms()
    if timestamp_ms - self._last_evaluation_ms < self._config.evaluation_interval_seconds * 1000:
        return []
    self._last_evaluation_ms = timestamp_ms
    # ... existing logic
```

Add an `AutoscalerConfig` message to `config.proto`:

```protobuf
message AutoscalerConfig {
  float evaluation_interval_seconds = 1;  // Default: 10.0
  float requesting_timeout_seconds = 2;   // Default: 120.0
}
```

Wire this into `IrisClusterConfig` and thread through controller startup.

### Part 2: REQUESTING State and Async Scale-Up

Add `REQUESTING` to `GroupAvailability`:

```python
class GroupAvailability(Enum):
    AVAILABLE = "available"
    BACKOFF = "backoff"
    AT_CAPACITY = "at_capacity"
    QUOTA_EXCEEDED = "quota_exceeded"
    REQUESTING = "requesting"  # Scale-up in progress
```

Track pending requests in `ScalingGroup`:

```python
# In ScalingGroup.__init__:
self._requesting_until_ms: int = 0

def mark_requesting(self, ts: int, timeout_ms: int) -> None:
    self._requesting_until_ms = ts + timeout_ms

def availability(self, ts: int) -> AvailabilityState:
    # ... existing checks ...
    if ts < self._requesting_until_ms:
        return AvailabilityState(GroupAvailability.REQUESTING, "scale-up in progress")
    # ... rest
```

Offload `_execute_scale_up` to a `ThreadPoolExecutor` in the `Autoscaler`:

```python
# In Autoscaler.__init__:
self._scale_up_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="scale-up")

def _execute_scale_up(self, group, ts, reason=""):
    timeout_ms = int(self._config.requesting_timeout_seconds * 1000)
    group.mark_requesting(ts, timeout_ms)
    self._scale_up_executor.submit(self._do_scale_up, group, ts, reason)

def _do_scale_up(self, group, ts, reason):
    try:
        slice_obj = group.scale_up(ts=ts)
        # ... success handling
    except Exception:
        # ... failure handling
    finally:
        group.clear_requesting()
```

The REQUESTING state prevents `route_demand()` from allocating additional demand to a group that already has a scale-up in flight, since `can_accept_demand()` checks availability.

### Threading Safety

`ScalingGroup` state is currently accessed only from the scheduling thread. With async scale-up, `_do_scale_up` runs on a pool thread and mutates `_vm_groups` and availability state. We need minimal synchronization:

- `_requesting_until_ms` is set before submission and cleared after completion — the scheduling thread only reads it, so a simple atomic-style pattern (write-then-read) is safe.
- `scale_up()` mutates `_vm_groups` — we add a `threading.Lock` to `ScalingGroup` protecting `_vm_groups` mutations (scale_up, scale_down, reconcile). Reads of `_vm_groups` for status/counting are already dict snapshots.

## Implementation Outline

1. Add `AutoscalerConfig` to `config.proto`, regenerate, wire through YAML → controller → Autoscaler
2. Enforce `evaluation_interval_seconds` check in `run_once()`
3. Add `REQUESTING` to `GroupAvailability` and `mark_requesting`/`clear_requesting` to `ScalingGroup`
4. Move `_execute_scale_up` to use `ThreadPoolExecutor`, add lock to ScalingGroup for `_vm_groups`
5. Test: rate-limiting skips evaluation, REQUESTING blocks demand routing, async scale-up doesn't block caller

## Notes

- `VM_STATE_REQUESTING` already exists in `vm.proto:32` but is unused. We can use it for VM-level status alongside the group-level `REQUESTING` availability.
- The existing `_last_evaluation_ms` field and `AutoscalerConfig` class mean Part 1 is nearly a one-liner.
- `ThreadPoolExecutor` with `max_workers=4` allows concurrent scale-ups across different groups while bounding resource usage.
- The requesting timeout (default 120s) acts as a safety net — if a scale-up hangs, the group becomes available again after timeout.

## Future Work

- Per-group scale-up concurrency limits
- Async scale-down (currently fast since terminate is non-blocking for TPUs)
- Metrics/observability for scale-up latency
