# Iris: replace reservations with soft `availability:` constraints

**Status:** proposed (weaver #201)
**Author:** agent session `iris-no-more-reservations`
**Scope:** `lib/iris` — delete the reservation apparatus; add a zone-level soft placement hint.

## 1. Problem

The reservation system exists to do exactly one thing: **run a non-TPU job (a CPU
orchestrator) in a zone where some accelerator can be found**, so that when the
orchestrator launches its TPU work it lands in the same place.

To achieve that one goal it carries a large, brittle apparatus:

- A `LaunchJobRequest.reservation` (`ReservationConfig` / `ReservationEntry`) wire
  type and a `--reserve` CLI flag.
- A synthetic `:reservation:` **holder child job** spawned per reserved job
  (`ops/job.py`), with its own tasks and replicas, whose only purpose is to hold
  slices.
- A `reservation_claims` table plus a per-tick **claim loop** (`cleanup_stale_claims`
  + `claim_workers_for_reservations`) binding workers to entries.
- A **scheduling gate** (`reservation_unsatisfied`) that blocks the real job until
  every entry is claimed.
- **Worker taints** (`reservation-job` attribute) + **constraint injection**
  (`inject_reservation_taints`, `inject_taint_constraints`): claimed workers are
  tainted, non-reservation jobs get a `NOT_EXISTS(reservation-job)` constraint, and
  the reserving job gets a hard **zone pin** to the claimed zone.
- A **preference pass** (`preference_pass`) and **preemption exemption** for claimed
  workers.
- DB columns `jobs.is_reservation_holder`, `jobs.has_reservation`,
  `job_config.has_reservation`, `job_config.reservation_json`, three partial
  indexes, and a stack of reads/writes/codec helpers.

### This is now mostly unnecessary

1. We run `executor.main` **from the TPU job itself**, so resources are created in
   the region where the TPU already runs — no need to pre-pin a separate CPU job.
2. We almost always **pre-compute data in a region**, so region pinning is moot.

### And it actively hurts

`.agents/ops/2026-06-08-canary-ferry-reservation-taint-timeouts.md`: the daily TPU
canary was **red for 8 consecutive runs**. Two failure modes, both reservation
apparatus:

1. The `:reservation:` holder inherited the parent's `preemptible=false` and hunted
   for a non-existent non-preemptible v5p-8 group → stuck PENDING for 3.5h.
2. After that was patched, the CPU orchestrator parent is `has_direct_reservation`,
   so the scheduler injects a `reservation-job == <self>` **EQ taint** pinning it to
   the *v5p-8 workers claimed for the reservation*. But its task only needs CPU, and
   on-demand CPU workers never carry that taint — so the orchestrator **can never
   schedule on CPU** and burns the full 6h wall clock holding an idle TPU, then is
   force-cancelled.

The mechanism built to *help* co-location is the thing that prevents the
orchestrator from ever running.

## 2. Goal

Replace the whole apparatus with a **soft, zone-level placement hint**:

> "I will probably use accelerator resources like X — schedule me in a zone where
> X can be found, if one exists."

Concretely, a `--reserve v5p-8` becomes a **soft constraint** `availability:v5p-8`
on the job. The scheduler and autoscaler treat it as a *preference* (rank, never
block), evaluated as a **zone-level intersection**: a worker is a good placement if
*its zone* can provide the requested accelerators — not if that individual worker
*is* the accelerator.

Key properties:

- **Soft by default.** Never gates scheduling, never spawns holder jobs, never pins,
  never preempts. If no matching zone exists, the job still runs — anywhere.
- **Zone-level.** "Can this *zone* provide `v5p-8`?" is inferred from the autoscaler
  (which configured groups exist per zone, and which are not hard-quota-blocked).
- **Reuses existing machinery.** Iris already has soft constraints
  (`CONSTRAINT_MODE_PREFERRED`), `evaluate_constraint`, `ConstraintIndex`, and
  soft-constraint ranking in **both** the scheduler (`rank_by_soft_score`) and
  autoscaler routing (`route_demand` ranks groups by `soft_constraint_score`). We
  add almost no new mechanism — we feed those existing rankers a new attribute.
- **Accelerators only.** Availability markers are produced only for accelerator
  device variants; CPU/RAM/disk never produce them.

User-visible behavior is preserved: `--reserve v5p-8` still works and still steers
the job toward a v5p-8 zone — it just does so as a hint, not a hard reservation.

### 2.1 This is a deliberate semantic narrowing (not an accidental regression)

Today a reservation is a **hard hold**: `ReservationEntry` is "a demand anchor that
the autoscaler provisions before the reserving job schedules" (`types.py:452`); the
controller spawns a real `:reservation:` holder job (`ops/job.py:252`), gates the
parent until claims exist (`policy.py:142`), and suppresses parent demand until
claimed (`policy.py:344`). **We are intentionally dropping the hold.** Per the goal:
`executor.main` runs from the TPU job itself and data is pre-computed in-region, so
pre-provisioning a co-located CPU job is no longer needed.

Caller audit confirms nothing depends on the hold:

- **No production SDK caller** passes `submit(reservation=...)` (grep of
  `experiments/`, `scripts/`, `lib/marin`, `lib/iris/scripts` — zero hits).
- The only `--reserve` users are **CPU orchestrators** that co-locate accelerator
  work: the canary ferry and `experiments/grug/moe/agent.md`. Both want the *hint*,
  not the hold — the hold is exactly what broke the canary for 8 days.
- `dev_tpu.py` (the `reserve-tpu` skill) does **not** use `--reserve`; it submits a
  normal job with a `zone_constraint` and an accelerator resource spec
  (`scripts/iris/dev_tpu.py:747`). Unaffected.

So the lost hard guarantee has no remaining dependant. A user who still wants a hard
pin uses `--zone` (existing) or `availability_constraint(v, soft=False)`.

## 3. Design

### 3.1 The `availability:` attribute namespace

We model zone capability as a set of composite-key attributes that workers and
scaling groups carry:

```
availability:v5p-8      = "true"
availability:v6e-8      = "true"
availability:h100       = "true"
```

The key encodes the device variant (lowercased, the same canonical string used by
the existing `device-variant` attribute). The value is a presence marker; matching
uses `ConstraintOp.EXISTS`. Multiple variants per zone = multiple keys (the
`worker_attributes` table is `(worker_id, key)` single-valued, and these markers are
**injected at decision time, never persisted**, so the schema is irrelevant here).

A job hint is therefore:

```python
Constraint.create(key="availability:v5p-8", op=ConstraintOp.EXISTS,
                  mode=CONSTRAINT_MODE_PREFERRED)   # soft
```

New helper in `constraints.py`:

```python
AVAILABILITY_PREFIX = "availability:"

def availability_key(variant: str) -> str:
    return f"{AVAILABILITY_PREFIX}{variant.strip().lower()}"

def availability_constraint(variant: str, *, soft: bool = True) -> Constraint:
    """Prefer (soft) or require (hard) a zone that can provision `variant`."""
    mode = job_pb2.CONSTRAINT_MODE_PREFERRED if soft else job_pb2.CONSTRAINT_MODE_REQUIRED
    return Constraint.create(key=availability_key(variant), op=ConstraintOp.EXISTS, mode=mode)

def is_availability_key(key: str) -> bool:
    return key.startswith(AVAILABILITY_PREFIX)
```

**Why composite keys, not one multi-valued `availability` attribute?** `AttributeValue`
holds a single scalar and `ConstraintIndex`/`evaluate_constraint` are per-key, so a
multi-valued attribute would need new machinery. Composite keys drop straight into
the existing EXISTS posting-list path with zero new matching code, and read exactly
as the issue described it (`availability:tpuv5p-8`).

### 3.2 Where capability comes from: `Autoscaler.zone_capabilities()`

The autoscaler owns `self._groups: dict[str, ScalingGroup]`. Each group exposes
`.zone` and (for accelerator groups) `resources.device_variant` — there is **no
`.device_variant` property** on `ScalingGroup` (review finding #5), so we read
`group.resources.device_variant` behind a `None` guard, exactly as `to_attributes()`
does. `zone_capabilities()` computes the map **live, on demand**:

```python
def zone_capabilities(self, timestamp: Timestamp | None = None) -> dict[str, frozenset[str]]:
    """zone -> {device_variant} the cluster is configured to provision there.

    A variant counts for a zone if a configured accelerator group for it exists
    there and is not hard-quota-blocked (QUOTA_EXCEEDED). Transient states
    (AT_MAX_SLICES, BACKOFF, REQUESTING) still count — the zone *can* provide it.
    """
    ts = timestamp or Timestamp.now()
    caps: dict[str, set[str]] = defaultdict(set)
    for group in self._groups.values():
        zone = group.zone
        res = group.resources                     # None for CPU-only / unset
        variant = res.device_variant if res is not None else ""
        if zone is None or not variant:            # accelerators only
            continue
        if group.availability(ts).status is GroupAvailability.QUOTA_EXCEEDED:
            continue
        caps[zone].add(variant.lower())
    return {z: frozenset(v) for z, v in caps.items()}
```

Both readers call it directly: the autoscale phase passes
`self.zone_capabilities(ts)` into `route_demand(...)`, and `schedule()` reads
`self.autoscaler.zone_capabilities()` (or `None` when there is no autoscaler, e.g.
the K8s provider).

**Live read, no cache.** `single_control_tick` (`controller.py`) is the **only**
real path — schedule, reconcile, and autoscale run as sequential phases of one
driver thread, so a read is already race-free. (The legacy three-thread mode is dead
code, tracked for removal in #6434; we do not add threading defensiveness for it.)
Computing the map fresh each call is O(#groups) — a handful of dict appends per
tick — so there is no reason to cache it. An earlier revision cached an immutable
snapshot to defend the dead three-thread path; that complexity is gone.

Rationale for the QUOTA_EXCEEDED-only exclusion: the hint is soft, so flapping is
cheap and we prefer stability — a zone *configured* for v5p-8 but momentarily at max
slices is still the right place to steer a co-locating orchestrator. Only a
persistent quota block means "don't bother." Single, defensible predicate; open to
tightening to `can_accept_demand()` if review prefers liveness over stability.

### 3.3 Feeding the rankers

The `RpcTaskBackend` owns both the `Scheduler` and the `Autoscaler`, so it is the
natural seam. Both phases read the same `zone_capabilities()` map.

**Scheduler side (worker enrichment).** Each schedulable worker carries a `zone`
attribute. Before placement, enrich each worker's attributes with its zone's
availability markers:

```python
def enrich_workers_with_availability(
    workers: list[WorkerSnapshot], zone_caps: dict[str, frozenset[str]],
) -> list[WorkerSnapshot]:
    if not zone_caps:
        return workers
    out = []
    for w in workers:
        zone = w.attributes.get(WellKnownAttribute.ZONE)
        variants = zone_caps.get(str(zone.value)) if zone else None
        if not variants:
            out.append(w); continue
        attrs = dict(w.attributes)
        for v in variants:
            attrs[availability_key(v)] = AttributeValue("true")
        out.append(replace(w, attributes=attrs))
    return out
```

This keys on the **existing `WellKnownAttribute.ZONE` attribute** — the same one
`--zone`/`--region` matching already relies on, injected into worker config for
GCP/CoreWeave groups (`scaling_group.py:228`). Workers without a zone (local/manual,
review finding #6) are passed through unchanged: they simply get no markers and the
soft hint never ranks them up — correct, and explicitly tested.

This runs at the top of `run_scheduling_decision` (replacing the deleted
`inject_reservation_taints`), so the `ConstraintIndex` and `rank_by_soft_score`
built downstream already see the markers. A CPU worker in a v5p-8 zone now satisfies
the soft `availability:v5p-8` hint, so `rank_by_soft_score` orders it ahead of CPU
workers in non-v5p-8 zones — the orchestrator co-locates by *preference*, with no
taint and no gate.

`RpcTaskBackend.schedule()` passes the map in:

```python
def schedule(self, snapshot: ScheduleInput) -> ScheduleResult:
    zone_caps = self.autoscaler.zone_capabilities() if self.autoscaler is not None else None
    return run_scheduling_decision(self._scheduler, snapshot, zone_capabilities=zone_caps)
```

`run_scheduling_decision` treats `zone_capabilities=None` as "no enrichment" — the
worker rows pass through unchanged, so the K8s path (no autoscaler) is a clean no-op.

`ScheduleInput.claims` is **removed**; no new field is needed (the backend injects
from its own attached autoscaler — consistent with how `autoscale()` already calls
`self.autoscaler.*`). `K8sTaskProvider` has no autoscaler and no Iris workers, so it
is unaffected.

**Autoscaler side (group enrichment).** So the autoscaler *provisions* the CPU
orchestrator in the right zone, enrich each group's routing attributes with its
zone's markers inside `route_demand`. **Critically (review finding #2), the enriched
attrs must be threaded through BOTH the hard-filter index AND the soft-score sort** —
today the soft-rank step recomputes `group.to_attributes()` directly
(`routing.py:485`), so injecting markers only into a local `group_attrs` map would
let availability constraints survive filtering but score **zero** for every group.
Build the enriched map once and use it in both places:

```python
zone_caps = ...                                   # passed in from evaluate()
group_attrs = {
    g.name: {**g.to_attributes(),
             **{availability_key(v): AttributeValue("true")
                for v in zone_caps.get(g.zone or "", ())}}
    for g in sorted_groups
}
group_index = ConstraintIndex.build(group_attrs)
...
# soft-rank now reads the enriched map, NOT group.to_attributes():
matching_groups = sorted(
    matching_groups,
    key=lambda g: (-soft_constraint_score(group_attrs[g.name], soft_routing_cs),
                   g.config.priority or 100),
)
```

`route_demand` already splits hard/soft and ranks matching groups by
`soft_constraint_score`, so once both call sites read `group_attrs`, a CPU group in a
v5p-8 zone ranks above CPU groups elsewhere. (Switching the sort key from
`group.to_attributes()` to the precomputed `group_attrs[g.name]` also removes a
redundant per-comparison recompute — a small win.) No other routing logic changes.

**Routing filter fix (required).** `routing_constraints()` drops any key not in
`CONSTRAINT_REGISTRY` (`desc is None`). Without a change, `availability:*` would be
silently stripped before routing and the autoscaler would ignore the hint. Add a
prefix exception so availability keys survive as routing constraints:

```python
def routing_constraints(constraints):
    result = []
    for c in constraints:
        if is_cpu_device_type_constraint(c):
            continue
        if is_availability_key(c.key):           # NEW: dynamic routing key
            result.append(c); continue
        desc = CONSTRAINT_REGISTRY.get(c.key)
        if desc is None or not desc.routing:
            continue
        result.append(c)
    return result
```

`merge_constraints` treats non-canonical keys by accumulation, so availability keys
flow through untouched. `Constraint.__post_init__` validates only arity (EXISTS = 0
values), so no key validation rejects them. They are **not** in
`INHERITED_CONSTRAINT_KEYS` (`{region, zone}`), so they do not propagate to child
jobs — correct, since the hint targets the orchestrator and children re-derive their
own device constraints.

### 3.4 Ingestion: `--reserve` → soft availability hint

`cli/job.py`: keep `--reserve` (preserve user-facing behavior), but transform each
spec into a soft availability constraint instead of a `ReservationEntry`:

```python
for spec in reserve:
    variant = parse_reserve_variant(spec)      # extract device variant; ignore count
    constraints.append(availability_constraint(variant, soft=True))
```

- The count in `[COUNT:]DEVICE` is now meaningless (a zone hint is not a worker
  count) — parse and ignore it (warn if > 1).
- The **mutual exclusion** with `--region`/`--zone` is **dropped**: a soft
  availability hint composes cleanly with a hard region/zone constraint.
- Variant canonicalization reuses the existing path: TPU specs → topology name
  (`v5p-8`); GPU specs → gpu variant (`h100`), matching the `device-variant`
  attribute the groups already carry.

SDK `client.submit_job()` / `remote_client`: **remove** the
`reservation: list[ReservationEntry]` parameter (no production callers found).
Current clients add availability hints via the already-supported
`constraints=[availability_constraint("v5p-8")]`; the CLI does the transform for
`--reserve`.

**Server-side back-compat (added after review).** The `LaunchJobRequest.reservation`
wire field is *not* deleted — only omitted by current clients. A pre-availability
client that still sets it must not hard-fail. So the controller keeps the proto
messages and converts at the **single ingestion point** (`ops.job.submit`): each
`ReservationEntry`'s accelerator variant becomes one soft
`availability:<variant>` constraint (deduped, accelerators only), merged into the
job's constraints, with a deprecation `WARNING` logged so we can spot stragglers.
Nothing reservation-shaped is persisted — the field is consumed and discarded, so
the DB schema, codec, and replay paths stay clean. A scheduled follow-up (≈2 weeks)
deletes the proto field + the ingestion shim once clients have upgraded.

### 3.5 Deletions

**Proto** (`job.proto`, `controller.proto`): `ReservationConfig` /
`ReservationEntry` and `LaunchJobRequest.reservation = 30` are **kept but marked
deprecated** (back-compat shim above), not deleted — deleting them would break
in-flight old clients. Everything else reservation-shaped is deleted. Keep
`Constraint`.

**DB** (migration `0029_drop_reservations.py`). This is heavier than the doc first
implied (review finding #4). The iris convention is *table rebuild*, not in-place
`DROP COLUMN`, because "SQLite cannot drop a column referenced by an index in place
across all supported versions" (`0028` docstring) — and both `has_reservation` and
`is_reservation_holder` are referenced by partial indexes. Unlike `0028` (the
unreferenced `auth.api_keys`), `jobs` is **FK-referenced** by `tasks`, `job_config`,
`endpoints`, `workdir_files`, etc., so the rebuild must follow the **`0027`** pattern,
not `0028`:

1. **Graceful handling of in-flight reservation state** (codex P1; no idle-deploy
   requirement). A holder runs the *parent's entrypoint on the accelerator* with
   special dispatch/reconcile, so once the holder filters are deleted it would read
   as an ordinary job — a RUNNING holder keeps consuming its accelerator, a PENDING
   one gets dispatched. So the migration, *before* stripping the schema:
   (a) **deletes holder jobs** (`DELETE FROM jobs WHERE is_reservation_holder = 1`);
   `foreign_keys` is pinned ON at connect, so the cascade — including the
   self-referential `parent_job_id` FK — removes their tasks/attempts/config and any
   descendants. The deleted holder leaves the controller's reconcile `desired` set,
   so the worker **zombie-reaps** the still-running process (`worker.py:884-894`) and
   frees the accelerator; nothing leaks controller-side because no rows remain.
   (b) **converts each non-holder job's `reservation_json`** into soft
   `availability:<variant>` constraints merged into its `constraints_json` (the same
   conversion as the ingestion shim), preserving the zone-steering intent of a job
   that was mid-flight at upgrade. Idempotent: holder-delete re-runs as a no-op,
   conversion dedups by constraint key.
2. `raw_conn.commit(); PRAGMA foreign_keys=OFF; BEGIN IMMEDIATE` (toggle outside a
   txn, per `0027:88`).
3. Rebuild `jobs` without `is_reservation_holder` / `has_reservation` and without the
   `jobs_is_reservation_holder_check`; `INSERT INTO ... SELECT`; `DROP TABLE jobs`;
   `RENAME`; recreate **all surviving** `jobs` indexes (omit
   `idx_jobs_has_reservation`, `idx_jobs_reservation_holder`).
4. Rebuild `job_config` without `has_reservation` / `reservation_json`; recreate its
   surviving indexes (omit `idx_job_config_has_reservation`).
5. `DROP TABLE reservation_claims`.
6. `commit()`; re-enable FKs. Idempotency guard via `PRAGMA table_info` (no-op if the
   columns are already gone), so a mid-run crash is safe to retry — same shape as
   `0027`/`0028`. Add a migration round-trip test.

Update `schema.py` to match (remove the columns, indexes, check constraint, and the
`reservation_claims_table` + `ReservationClaim`).

**scheduling/policy.py**: delete `RESERVATION_TAINT_KEY`, `reservation_unsatisfied`,
`_worker_matches_reservation_entry`, `claim_workers_for_reservations`,
`cleanup_stale_claims`, `refresh_reservation_claims*`, `inject_reservation_taints`,
`inject_taint_constraints`, `_reservation_zone_constraints`,
`reservation_zones_from_claims`, `_find_reservation_ancestor`, `preference_pass`,
`_MAX_RESERVATION_PLACEMENTS_PER_WORKER_PER_CYCLE`, the reservation gate in
`apply_scheduling_gates`, reservation branches in `compute_demand_entries`, and the
claimed-worker preemption exemption (`get_running_tasks_with_band_and_value` no
longer takes `claimed_workers`).

**backend.py**: `run_scheduling_decision(scheduler, snapshot, zone_caps)` — enrich
workers, then gates → order → `find_assignments` → preemption (no taints, no
preference pass). `apply_placements` loses the claims/taint args. Remove `claims`
from `ScheduleInput`.

**reads.py / writes.py / codec.py / types.py**: delete `list_claims`,
`reserved_job_ids`, `reservation_entry_counts`, `jobs_with_reservations`,
`replace_reservation_claims`, the holder-exclusion in `resource_usage_by_worker`,
`reservation_to_json`, `reservation_entries_from_json`, `ReservationEntry`,
`ReservationClaim`, and the `is_reservation_holder`/`has_reservation` fields on
`PendingTask` / `insert_job`.

**ops/job.py**: delete `RESERVATION_HOLDER_JOB_NAME` holder-job creation,
`request_has_reservation`. **reconcile/policy.py**: delete the holder-name constant
and any holder special-casing.

**SchedulingContext**: drop `reserved_job_ids`, `reservation_entry_counts`,
`reservation_zones_by_job`.

**Dashboard**: remove reservation displays from `CapacityTab.vue` (and any
reservation column/tab). `npm run build:check`.

**Docs / scripts**: update `OPS.md` (drop the "Reservation system" section and the
`SELECT * FROM reservation_claims` recipe), `README.md`, `docs/architecture.md`;
simplify `scripts/workflows/iris_monitor.py` (delete `_RESERVATION_HOLDER_SUFFIX` /
`_is_reservation_holder` / holder-skipping in `_pick_child` — no more holder
children). `experiments/grug/moe/agent.md`'s `--reserve v5p-8` still works as-is.

**Tests**: delete `tests/cluster/controller/test_reservation.py` (2150 lines) and the
`reservation_worker_failure_*` replay golden; strip reservation cases from
`test_preemption.py`, `test_transitions.py`, `test_dashboard.py`,
`test_5470_preemption_reassignment.py`, `test_perf_baselines.py`, e2e smoke. Add new
tests (§5).

### 3.6 What we explicitly do NOT build

- **No pre-provisioning.** The autoscaler no longer provisions accelerator capacity
  ahead of the job. The orchestrator's own later TPU launch provisions it. This is
  the intended simplification (executor.main runs from the TPU job).
- **No child-zone inheritance work.** Co-locating the eventual TPU job is out of
  scope; the hint only steers the orchestrator's placement.
- **No hard gating.** A user who wants a hard pin uses `--zone` (existing) or
  `availability_constraint(v, soft=False)`.

## 4. End-to-end walkthrough (the canary case, fixed)

1. `iris job run --reserve v5p-8 -- python -m executor.main` → job carries one CPU
   task + soft constraint `availability:v5p-8`. No holder job, no claim, no wire
   reservation.
2. Autoscaler computes `zone_capabilities()` = `{us-central1-a: {v5p-8, ...},
   us-east5-a: {v5p-8, ...}, ...}`.
3. Scheduler enriches workers: CPU workers in those two zones get
   `availability:v5p-8=true`. `rank_by_soft_score` prefers them. The orchestrator
   schedules immediately on whatever CPU worker exists, preferring a v5p-8 zone.
4. If no CPU worker exists yet, CPU demand routes through the autoscaler; the v5p-8
   zone's CPU group ranks first and is provisioned there.
5. The orchestrator runs `executor.main`, which launches the v5p-8 TPU job; that TPU
   demand provisions v5p-8 in the same zone (capacity-driven, as today).

No idle TPU held for 6h; no untoppable EQ taint; no preemptible-mismatch holder.

## 5. Testing

- `constraints.py`: `availability_constraint`/`availability_key` shape; EXISTS arity;
  `routing_constraints` keeps `availability:*`; `merge_constraints` passes them;
  not inherited.
- `autoscaler`: `zone_capabilities()` over a multi-zone, multi-group fixture
  (CPU-only group contributes nothing; QUOTA_EXCEEDED group excluded; AT_MAX_SLICES
  still counts).
- scheduler: worker enrichment + `rank_by_soft_score` prefers a CPU worker in a
  capable zone; **never blocks** when no zone matches (regression for the canary
  bug — the orchestrator must schedule even with zero matching zones).
- routing: `route_demand` prefers the CPU group in a capable zone via group
  enrichment.
- ingestion: `--reserve v5p-8` → one soft `availability:v5p-8` constraint; composes
  with `--zone`; count ignored.
- migration round-trip test (existing migration test harness).
- Per `lib/iris/TESTING.md`: behavior-focused, no tautologies, use existing fakes.

## 6. Rollout / risk

- **Client/server split deploy (back-compat).** The wire contract is preserved on
  the server: pre-availability clients that still send `LaunchJobRequest.reservation`
  are converted at ingestion (§3.4), not rejected. Current clients omit the field.
  A scheduled follow-up (~2 weeks) removes the proto field + the ingestion/migration
  shims once stragglers have upgraded. Build the controller image *and* redeploy (per
  OPS.md — a restart alone re-pulls stale `:latest`).
- **In-flight reservations at migration (no idle-deploy needed).** A live
  `:reservation:` holder is a real accelerator job running the parent's entrypoint —
  *not* a harmless CPU job (codex P1 / review finding #4/#7). Rather than require an
  idle deploy, migration `0029` handles it: it **deletes holder jobs** (cascade) so
  the worker zombie-reaps the running process and frees the accelerator, and
  **converts real reservations** to soft availability hints (§3.5). Removing the
  reservation-only special cases (`resource_usage_by_worker` holder exclusion;
  preemption claimed-worker exclusion; reconcile holder no-cascade) is then a no-op,
  because no holders/claims remain after the migration.
- **Soft-hint blind spot.** Because the hint never blocks, a job whose zone has no
  matching capacity simply runs elsewhere and may fail later when it tries to launch
  the accelerator — same outcome as today minus the 6h idle-TPU burn. Acceptable and
  strictly better than the gate.
- **Index cost.** Worker/group enrichment adds O(workers × variants/zone) per tick
  and rebuilds an index that is already rebuilt each tick. Negligible at current
  fleet sizes.

## 7. Open questions for review

1. `zone_capabilities()` predicate: configured-minus-QUOTA_EXCEEDED (stable, chosen)
   vs `can_accept_demand()` (live, flappier)?
2. Keep `--reserve` as the flag name, or rename to `--prefer`/`--availability` with
   `--reserve` as a documented alias? (Proposal: keep `--reserve`, preserve behavior.)
3. Should we register a single `availability` prefix descriptor in
   `CONSTRAINT_REGISTRY` (formalize the namespace) instead of the prefix special-case
   in `routing_constraints`?

## 8. Adversarial review (codex) — dispositions

Ran `codex exec` against this doc and the real code. Eight findings; all addressed:

| # | Finding | Disposition |
|---|---------|-------------|
| 1 | Blocker: design drops the hard "hold capacity" semantics | **Intended**, documented in §2.1 with a caller audit — the hold is exactly what broke the canary; no production caller depends on it. |
| 2 | Blocker: routing soft-rank recomputes `to_attributes()`, so markers score 0 | **Fixed** in §3.3 — thread one enriched `group_attrs` map through both the index and the soft-score sort. (Real bug; good catch.) |
| 3 | Blocker: `schedule()` reading live autoscaler state isn't thread-safe (legacy 3-thread mode) | **Obsoleted.** The three-thread mode is confirmed dead code (single_control_tick is the only path; removal tracked in #6434), so there is no concurrent reader to defend against. Per maintainer direction we add no threading defensiveness: `zone_capabilities()` is a plain live read (§3.2). An interim revision cached an immutable snapshot for this finding; that was dropped as unnecessary. |
| 4 | Blocker: migration too casual; holders aren't harmless; `jobs` is FK-referenced | **Fixed** in §3.5/§6 — `0027`-style FK-off rebuild of `jobs`; and (raised again as a P1 on the PR) migration `0029` now **deletes holder jobs** (cascade → worker zombie-reaps the running process) and **converts real reservations** to availability hints, so no idle deploy is required. |
| 4b | PR P1: rebuild copies holder rows while dropping the only `is_reservation_holder` marker, so holders become ordinary jobs after restart | **Fixed** — see #4. Holders are deleted before the schema strip; conversion preserves real-job intent. Covered by `test_migration_0029`. |
| 5 | `ScalingGroup.device_variant` doesn't exist | **Fixed** in §3.2 — read `group.resources.device_variant` behind the `HasField` guard. |
| 6 | Worker `zone` not always present | **Fixed** in §3.3 — key on the existing `ZONE` attribute, tolerate absence, add a test. |
| 7 | Deleting reservation special-cases could affect normal flows if holders/claims exist | **Resolved** by #4's idle-deploy guarantee; no-op once no holders/claims remain. |
| 8 | GPU canonicalization must lowercase everywhere | **Confirmed/handled** — `availability_key()` lowercases; both job-side variant (`get_device_variant(tpu_device/gpu_device(...))`) and group-side (`device_variant.lower()`) funnel through the same lowercased string, so `--reserve H100x8` → `availability:h100` matches the group attribute. |

Confirmed code facts (review agreed): `routing_constraints()` drops unknown keys
(hence the §3.3 prefix exception); `EXISTS` arity is 0; soft constraints rank (never
filter) in both `scheduler.py:482` and `routing.py:485`.

**Verdict after revisions:** sound to implement. The remaining real work is
mechanical-but-large (the deletions) plus three precise new seams (live
`zone_capabilities()`, dual-path routing enrichment, worker enrichment) and one
careful migration.
