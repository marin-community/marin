Here is the design document:

---

# Design: TPU Unresolvable → FAILED After Timeout

**Issue:** https://github.com/marin-community/marin/issues/3404

---

## Problem

When `gcloud compute tpus tpu-vm describe` fails (network hiccup, transient API error, or the TPU genuinely disappearing), `GcpSliceHandle._describe_cloud()` returns `SliceStatus(state=CloudSliceState.UNKNOWN)` (gcp.py:520).

`Autoscaler.refresh()` (autoscaler.py:1105–1127) handles `READY` and `FAILED` but has **no branch for `UNKNOWN`**. A slice stuck in `CloudSliceState.UNKNOWN` stays in `SliceLifecycleState.BOOTING` indefinitely — it never progresses, never fails, and blocks the group's capacity waterfall.

Concrete scenario: a TPU is preempted or never fully creates; `describe` starts returning errors; the slice occupies a slot in the scaling group forever.

---

## Goals

- After a configurable timeout, treat a persistently-UNKNOWN slice as FAILED
- Trigger the same cleanup path as a normal FAILED slice (scale-down + failure record)
- Default timeout of 15 min (quota timeout is 5 min, so this is deliberately conservative)
- Keep the happy path (transient glitch, next poll resolves) unchanged

**Non-goals:** Distinguishing "describe command timed out" from "slice genuinely deleted". Changing `CloudSliceState` semantics. Retry logic inside `_describe_cloud`.

---

## Proposed Solution

### Where to add the timeout

`Autoscaler.refresh()` is the right place. It already owns the READY/FAILED transitions. The slice's age is available via `handle.created_at` (base.py:304); `refresh` already receives a `timestamp` argument. No new state needed on `SliceState`.

### Core change (autoscaler.py:1093–1128)

```python
DEFAULT_UNRESOLVABLE_TIMEOUT = Duration.from_minutes(15)

# in Autoscaler.refresh():
elif status.state == CloudSliceState.UNKNOWN:
    age = timestamp - handle.created_at
    if age >= self._unresolvable_timeout:
        group.mark_slice_failed(slice_id, error_message="unresolvable after timeout")
        group.scale_down(slice_id)
        self._unregister_slice_workers(slice_id)
        group.record_failure()
        self._log_action(
            "slice_failed", group.name, slice_id,
            reason=f"TPU unresolvable for {age}", status="failed",
        )
    else:
        logger.debug("Slice %s UNKNOWN (age %s < timeout %s); will retry",
                     slice_id, age, self._unresolvable_timeout)
```

`_unresolvable_timeout` injected at construction with the default, consistent with how `quota_timeout` works in `ScalingGroup` (scaling_group.py:104, 231).

---

## Implementation Outline

1. **Add constant** — `DEFAULT_UNRESOLVABLE_TIMEOUT = Duration.from_minutes(15)` in `autoscaler.py` alongside existing defaults
2. **Constructor param** — Add `unresolvable_timeout: Duration = DEFAULT_UNRESOLVABLE_TIMEOUT` to `Autoscaler.__init__`
3. **Handle UNKNOWN in refresh()** — `elif status.state == CloudSliceState.UNKNOWN` branch with age-check; trigger mark_slice_failed + scale_down + record_failure when expired
4. **Tests** in `test_autoscaler.py`: (a) UNKNOWN before timeout → slice stays BOOTING, (b) UNKNOWN after timeout → FAILED + cleanup, (c) UNKNOWN then READY before timeout → recovers normally

---

## Notes

- **Age vs. first-seen-UNKNOWN**: Using `handle.created_at` starts the clock at slice creation, not the first UNKNOWN poll. This is intentional: avoids needing per-slice UNKNOWN tracking state, and a slice UNKNOWN from birth is more suspicious than one that briefly glitched.
- **Exception path is separate**: When `handle.describe()` raises (autoscaler.py:1101–1103), the code does `continue`. That path also needs a similar timeout but requires tracking last-known-state — leave for a follow-up.
- **No proto change needed** for MVP — the timeout is controller-internal config.

---

## Future Work

- Per-group `unresolvable_timeout` in `ScaleGroupConfig` proto for per-accelerator tuning
- Timeout on the exception path (describe throws repeatedly)
- Metric/alert on UNKNOWN-timeout transitions

---

Does this look right? Any preference on the 15-min default, or whether to also tackle the exception path (`describe()` raises) in the same PR?
