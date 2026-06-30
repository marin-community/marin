# Lighter-path implementation spec — ReservationLedger + DRAINING slice state

Scratch spec (untracked). Reworks PR #6531's cross-variant reserved-pool preemption so the
(pure) scheduler preemption pass and the autoscaler launch planner both ride on ONE explicit
per-tick capacity object (`ReservationLedger`), and a teardown-for-preemption is an explicit,
self-limiting `DRAINING` slice state instead of an immediate detach guarded by a hidden 5-minute
cooldown timer. Keep `DemandEntry.band` + band-ordered launch admission. Defer the task-binding
claim machine (provisional `claimed_by` slices) to a later PR.

All paths are under
`lib/iris/src/iris/cluster/controller/` unless noted. Line numbers are from the current branch
(approximate — re-grep before editing).

## Behavior being replaced

Today: the cross-variant pass returns `drain_workers`; the autoscaler `drain_slices_for_workers`
**detaches the victim slice immediately** (`operations.py:135` → `scaling_group.detach_slice` pops
it) and stamps a per-pool `_drain_cooldown` (`runtime.py:128,1051`). A 5-min `RESERVED_DRAIN_COOLDOWN`
then blocks the pass from re-preempting that pool while the replacement provisions
(`runtime.py:1068`, consumed in `policy.py:647` via `pools_on_cooldown`). Phase 1's band head-of-line
launch cap (`planning.py:_admit_in_band_order`) separately holds freed chips for the high band.

New: the victim slice is marked `DRAINING` and **kept counted** until reaped; the ledger exposes
its chips as `draining_chips`. The pass skips re-preempting a pool whose incoming capacity already
covers the deficit — `free + draining` (fungible, covers the drain-in-flight window) OR a booting
replacement slice of the preemptor's own variant (covers the replacement-booting window). That
direct observation replaces the cooldown. The launch cap still bands but reads the ledger's `free`.

## Two windows the pass must NOT re-preempt across (why the skip has two terms)

1. Victim chosen → reaped: victim is `DRAINING`, still allocated, `free` excludes it. `free + draining
   >= need` covers it. Chips are held automatically (draining slice still counted; no group can launch
   into them because `free` excludes them).
2. Victim reaped → replacement booted: replacement slice is `REQUESTING/BOOTING` of the preemptor's
   variant. `inflight_slices(pool, variant) >= 1` covers it.

Base case `free >= need` (autoscaler can already provision) is subsumed by term 1.

---

## 1. `scaling_group.py` — DRAINING state + mark + reap-eligibility

- `SliceLifecycleState` (≈49): add `DRAINING = "draining"`.
- `_lifecycle_to_vm_state` (≈260): add a mapping for `DRAINING`. Check `vm_pb2.VmState` for a
  TERMINATING-like value; if none exists use `VM_STATE_FAILED` (the slice is going away). Must not
  KeyError.
- `SliceState` (≈109): add `drain_started: Timestamp | None = None  # memory-only; set when marked
  DRAINING, used for the reap-timeout fallback`.
- New method `mark_slice_draining(self, slice_id, timestamp=None) -> SliceHandle | None`: under the
  lock set `state.lifecycle = DRAINING`, `state.drain_started = ts`, return `state.handle` (caller
  issues terminate). Do **not** pop from `_slices`. Return None if absent.
- `slices_needing_describe` (≈727): also include `DRAINING` slices, so `refresh` polls them and can
  observe deletion. (Add `or state.lifecycle == SliceLifecycleState.DRAINING`.)
- `ready_slice_probe_targets` (≈754) and `ready_slice_count` (≈749): already filter `== READY`, so
  DRAINING is excluded from health probing — leave as is, just confirm.
- Idle scale-down: a DRAINING slice must never be selected for idle scale-down. Check
  `get_idle_slices` / `scale_down_if_idle` (≈907-980) and exclude `lifecycle == DRAINING`
  (defensively — DRAINING slices have `quiet_since=None` so they're already excluded, but add an
  explicit guard).
- Chip accounting helper for the ledger: add (or extend) a per-group breakdown the ledger consumes —
  counts of chips by bucket {live=READY, inflight=REQUESTING/BOOTING/INITIALIZING (+pending_scale_ups),
  draining=DRAINING} and an inflight-slice-count-by-variant. See `reserved_pool_usage` rewrite below;
  prefer to compute buckets from `slice_state_counts()` (which already returns a dict keyed by
  `SliceLifecycleState`) so DRAINING appears automatically. Verify `slice_state_counts` buckets an
  unknown/new lifecycle correctly (it should, since it keys by the enum).
- `persistable_state` (≈365): no change — `lifecycle.value` already serializes `"draining"`.
- `restore_scaling_group` (≈1316): `SliceLifecycleState("draining")` now parses, so a restored
  DRAINING slice comes back DRAINING (drain_started lost → None). That's fine: refresh reaps it on
  the first describe-gone, or via the timeout measured from "now" if drain_started is None.

## 2. `reserved_pool.py` — `ReservationLedger` replaces `ReservedPoolView`

Replace `ReservedPoolView` / `reserved_pool_view` / `pools_on_cooldown` with:

```python
@dataclass(frozen=True)
class PoolLedger:
    """Chip accounting for one fungible reservation pool, one tick."""
    pool_id: str
    reservation_chips: int
    live_chips: int       # READY slices
    inflight_chips: int   # REQUESTING/BOOTING/INITIALIZING (+ pending_scale_ups), NOT draining
    draining_chips: int   # DRAINING slices (still physically allocated until reaped)
    inflight_slices_by_variant: dict[str, int]  # variant -> count of in-flight (non-draining) slices

    @property
    def allocated_chips(self) -> int: return self.live_chips + self.inflight_chips + self.draining_chips
    @property
    def free_chips(self) -> int: return self.reservation_chips - self.allocated_chips
    @property
    def incoming_chips(self) -> int: return self.free_chips + self.draining_chips  # free now or being freed
    @property
    def utilization(self) -> float: ...  # consumed/reservation, 0 when reservation<=0

@dataclass(frozen=True)
class ReservationLedger:
    """Single per-tick chip ledger over all fungible reservation pools.

    The one capacity view both the scheduler's cross-variant preemption pass and the autoscaler's
    launch planner read. The pass asks `incoming_chips`/`inflight_slices` ("is the deficit already
    being addressed?"); the planner asks `free_chips` ("how many new slices may I launch?").
    """
    pools: dict[str, PoolLedger]
    worker_pool: dict[str, str]    # worker_id -> pool_id
    worker_slice: dict[str, str]   # worker_id -> slice_id
    variant_pool: dict[str, str]   # device_variant -> pool_id
    chips_per_variant: dict[str, int]

    def is_empty(self) -> bool: return not self.variant_pool
    def free_chips(self, pool_id: str) -> int: ...        # 0 if absent
    def incoming_chips(self, pool_id: str) -> int: ...     # 0 if absent
    def inflight_slices(self, pool_id: str, variant: str) -> int: ...  # 0 if absent

def build_reservation_ledger(groups) -> ReservationLedger: ...
def log_reservation_ledger(ledger: ReservationLedger) -> None: ...  # one line per pool: res/live/inflight/draining/free
```

- `build_reservation_ledger`: bucket only groups with `reservation_chips > 0` by `quota_pool`. Same
  conflicting-budget validation as today (`reserved_pool_usage` raises on mismatch / missing
  quota_pool). live/inflight/draining chips = (per-bucket slice count) * `chip_count(variant)`.
  `pending_scale_ups` count as inflight. Build the worker/slice/variant maps exactly as
  `reserved_pool_view` did (`all_worker_ids`, `worker_slice_ids`, `get_tpu_topology(variant).chip_count`).
- Keep a small internal helper for the per-pool consumed total if useful, but the public object is
  `ReservationLedger`. Delete `ReservedPoolUsage`/`reserved_pool_usage`/`reserved_pool_view`/
  `log_reserved_pool_usage` and update all imports.

## 3. `policy.py` — `run_reserved_pool_preemption` rides the ledger (≈560-678)

- Signature: `view: ReservedPoolView` → `ledger: ReservationLedger`; guard `ledger.is_empty()`.
- Victim-slice construction (≈588-625): unchanged logic, source maps from `ledger`
  (`worker_slice`, `worker_pool`, `chips_per_variant`).
- Capacity bookkeeping for the skip decision: replace `free = dict(view.free_chips)` with
  `incoming = {pool: ledger.incoming_chips(pool) for pools that appear}` AND
  `replacement = {(pool): {variant: ledger.inflight_slices(pool, variant)}}` (lazily, per pool).
- Per preemptor (keep: BATCH never preempts; dedup coscheduled by parent; skip if variant None):
  - `need = ledger.chips_per_variant[variant]`; `pool = ledger.variant_pool.get(variant)`; skip if pool None.
  - **Remove** the `pool in view.pools_on_cooldown` check entirely.
  - if `incoming[pool] >= need`: `incoming[pool] -= need`; continue   # covered by free / drain-in-flight
  - elif `replacement[pool].get(variant, 0) >= 1`: `replacement[pool][variant] -= 1`; continue  # replacement already booting
  - else: select victims (UNCHANGED: lowest-band-first then smallest, `band > candidate.band`,
    minimal cover; if can't cover deficit, do nothing). On success: record pairs + `drain_workers`,
    and `incoming[pool] += freed - need` (the surplus freed chips stay available to later, lower-priority
    preemptors this tick — mirrors the old `free[pool] = free + freed - need`).
- Return `(pairs, drain_workers)` unchanged.

NOTE the deficit math: `deficit = need - incoming[pool]` (was `need - free`). Victims must cover the
deficit relative to incoming (free+draining), not raw free — so an already-draining victim isn't
double-counted.

## 4. `decision.py` — `apply_preemptions` (≈30-70)

- Param `reserved_view: ReservedPoolView | None` → `ledger: ReservationLedger | None`; pass to
  `run_reserved_pool_preemption`. Update the `is_empty()` guard.

## 5. `planning.py` — launch cap reads the ledger (≈171-234)

- `_cap_fungible_pool_launches`: build/receive a `ReservationLedger` instead of calling
  `reserved_pool_usage`. `usage[pool].free_chips` → `ledger.free_chips(pool)`. Keep `_PoolCandidate`,
  `_admit_in_band_order` (band-ordered admission), and band sourcing from
  `routing_decision.routed_entries[name]`. committable per pool = `ledger.free_chips(pool)`
  (NOT incoming — the planner can only launch into chips that are actually free now; draining chips
  aren't free until reaped).
- `build_scale_plan` (≈217): build the ledger once via `build_reservation_ledger(groups.values())`
  and pass it to `_cap_fungible_pool_launches`. Keep the existing call order (cap after per-group
  plans). DemandEntry.band stays.

## 6. `runtime.py` — drain = mark DRAINING; refresh reaps; delete cooldown

- Delete `RESERVED_DRAIN_COOLDOWN` (≈128), `self._drain_cooldown` (≈269), all cooldown stamping
  (≈1048-1051), and `pools_on_cooldown` plumbing.
- `reserved_pool_view(ts)` (≈1060) → `reservation_ledger(self) -> ReservationLedger` returning
  `build_reservation_ledger(self._groups.values())`. No ts, no cooldown.
- `drain_slices_for_workers(worker_ids)` (≈1031): change from detach to **mark DRAINING**:
  - For each unique slice holding a requested worker: `handle = group.mark_slice_draining(slice_id, ts)`
    (keeps it in `_slices` as DRAINING), collect the slice's worker ids (primary + sibling),
    `_unregister_slice_workers(slice_id, slice_worker_ids)` (so the scheduler stops placing — parity
    with today), and append a termination request.
  - Issue `terminate_slice_handle(handle, context="draining for preemption")` via the existing
    `_run_io_batch` (start VM deletion now; idempotent).
  - Return ALL drained slices' worker ids (primary + sibling) for the controller to fail+forget
    (today it returns siblings only and the controller already fails the primaries via the PREEMPT
    path — preserve the net effect: every worker on a drained slice ends up failed+forgotten and the
    SLICE lingers DRAINING). Reuse the `_detach_slices_for_workers` shape in `operations.py` but
    swapping `detach_slice` → `mark_slice_draining` and keeping the slice; rename to
    `mark_slices_draining_for_workers`. `feed_backoff=False` stays (no churn signal).
- `refresh()` fold (≈680-739): add a DRAINING branch BEFORE the READY/FAILED/UNKNOWN handling. For a
  slice whose tracked `lifecycle == DRAINING`:
  - If describe → FAILED or allocation-gone/zero-workers: **clean reap** — `group.detach_slice(slice_id)`
    (already terminated; do NOT `mark_slice_failed`, do NOT feed backoff), `_unregister_slice_workers`,
    log a `slice_drain_complete` action. (Reuses the `_log_action`/outcome machinery but as a clean
    completion, not a failure.)
  - If describe → READY/still-alive: keep DRAINING (VMs still deleting).
  - If still DRAINING after `drain_started` (or, if None, since first-seen) exceeds a reap timeout
    (reuse the unresolvable timeout, ≈15 min): force `detach_slice` + re-`terminate_slice_handle`.
  - DRAINING slices must be in `slices_needing_describe` (done in §1) so this branch actually runs.

## 7. `backend.py` — thread the ledger

- Import `ReservationLedger` instead of `ReservedPoolView`. Param on the schedule entry
  `reserved_view: ReservedPoolView | None` → `ledger: ReservationLedger | None` (≈244); pass to
  `apply_preemptions`. KEEP `autoscale_runs` gating (cross-variant pass only when the autoscaler runs
  this tick — regression `test_reserved_view_only_built_when_autoscale_runs`, rename if needed) and
  KEEP `_stamp_demand_bands` / `DemandEntry.band`.
- Where the backend obtains the capacity view from the autoscaler runtime, call
  `runtime.reservation_ledger()`.

## 8. `controller.py`

- `_remove_drained_workers` (≈1276): unchanged in spirit — still fails+forgets the workers the drain
  reports. The SLICE now lingers DRAINING and is reaped by refresh, so do not expect the slice to be
  gone here.
- Any reference to `reserved_pool_view` / cooldown: update to the ledger.

## 9. Persistence / restore

- `SlicePersist.lifecycle` already round-trips the string; `restore_scaling_group` parses "draining".
  No proto change. Confirm `_LIVE_CLOUD_STATES` handling in `recovery.py` doesn't discard a DRAINING
  slice whose cloud VMs are mid-delete (it may already be gone from cloud → reclaimed, which is fine).

## 10. Tests (lib/iris/tests/cluster/controller/)

Behavior-focused (read root `TESTING.md` + `lib/iris/AGENTS.md`/`TESTING.md` first). Cover:
- `test_reserved_pool.py`: rewrite for `ReservationLedger`/`PoolLedger` — bucket accounting
  (live/inflight/draining chips, free, incoming), `inflight_slices_by_variant`, conflicting-budget
  raise, the band-ordered `_admit_in_band_order` (now fed `ledger.free_chips`). Keep/port the existing
  `TestFungiblePoolLaunchCap` via `build_scale_plan` (high band wins; non-fungible untouched).
- `test_reserved_preemption.py`: pass skip across BOTH windows — (a) victim already DRAINING →
  `free+draining` covers → no re-preempt; (b) replacement booting (`inflight_slices`) → no re-preempt;
  base `free>=need` → no preempt. Victim selection minimal/lowest-band-first unchanged. Coscheduled
  dedup + entangled-slice exclusion unchanged. REMOVE cooldown tests; keep `_stamp_demand_bands` tests.
- New DRAINING lifecycle test (scaling_group / runtime): `mark_slice_draining` keeps the slice counted
  and excluded from health/idle; on describe-gone refresh reaps it cleanly (no backoff fed, no FAILED
  outcome); reap timeout forces teardown.
- Keep `test_reserved_view_only_built_when_autoscale_runs` behavior (rename to ledger).
- Delete/adjust any test importing `RESERVED_DRAIN_COOLDOWN`, `pools_on_cooldown`, `reserved_pool_view`,
  `reserved_pool_usage`, `ReservedPoolView`, `ReservedPoolUsage`.

## Invariants / gotchas

- `chip_count` = physical chips = HALF the GCP `vN-SIZE` suffix (v4-8 → 4 chips). Use
  `get_tpu_topology(variant).chip_count`.
- Lower `band_sort_key` = higher priority. Victim must be strictly lower priority (`band > candidate.band`).
- The pass runs in the SCHEDULE phase; victims become DRAINING in the AUTOSCALE phase of the SAME tick
  (the marking is driven by `drain_workers` → `mark_slices_draining_for_workers`). So a victim selected
  at tick N reads DRAINING in the ledger at tick N+1 — that's when term-1 of the skip engages. Within
  tick N, the pass's own `incoming[pool] += freed - need` bookkeeping prevents double-claiming across
  multiple preemptors. This is correct and intended.
- Launch cap uses `free` (not incoming); pass uses `incoming` (free+draining) + replacement slices.
- Do NOT feed the backoff detector when reaping a DRAINING slice (it's intentional).
- Keep the two prior codex P1 fixes intact: (1) cross-variant pass only when `autoscale_runs`;
  (2) victim grouping by physical slice, not task id.
- `./infra/pre-commit.py --all-files --fix` and `uv run pyrefly` must pass. No `:func:`/`:class:`
  docstring cross-references (describe the code at hand).
