# SPIKE S2 — Two-level placement fidelity & the minimal CapacitySummary

## Question

Does splitting scheduling into two levels — **root** assigns `task→backend` from only a per-backend
**capacity summary** (no worker mirror), then the **agent** assigns `task→worker` with today's
`Scheduler` — lose anything meaningful vs today's single global scheduler with `backend` as a worker
attribute? And what is the **minimal sufficient** `CapacitySummary`? (spec.md §3.1 strawman: `static`,
`allocatable`, `pending_leases`, `largest_gang`, `stale_ms`, `backoff`.)

Global cross-backend preemption is an **accepted non-goal** and is not treated as a failure.

## Method / harness

Throwaway harness under this directory. It **imports the real scheduler** — no fork, no
reimplementation:

- `iris.cluster.controller.scheduling.scheduler.Scheduler.find_assignments` (task→worker),
- `...scheduling.policy.run_preemption_pass` (preemption),
- `iris.cluster.constraints` (`Constraint`, `ConstraintIndex`, `routing_constraints`, `split_hard_soft`).

Files (all new, this dir only):

| file | what |
|---|---|
| `harness.py` | fleet/workload dataclasses; worker→`WorkerSnapshot` projection; `CapacitySummary` + `compute_summary`; `RootRouter` (task→backend from summaries only, with per-field ablation switches); `schedule_and_preempt` (real `Scheduler` + one preemption round), shared by both modes |
| `sim.py` | multi-tick simulator: `run_global` (one `Scheduler` over all workers, `backend` a worker attr) and `run_two_level` (`RootRouter` + per-backend `Scheduler`), with poll-lag and partition injection; metrics |
| `workload.py` | seeded fleet (2 equivalent v5e TPU backends, 1 h100 k8s backend, 2 CPU pools = 96 workers) + mixed workload (explicit `--backend` pins, `device-variant` capability constraints, region pins, coscheduled gangs, varied bands/users) |
| `run.py` | scenarios + ablations; prints the numbers below |

Both modes call the **identical** `schedule_and_preempt` for task→worker, so the *only* variable is the
routing layer and the summary it reads. Jobs **pin to one backend** (design invariant). The root sees a
`CapacitySummary` computed at the start of each tick (optionally lagged by `poll_lag` polls); it never
sees a worker. Metrics: placed/starved, wait (arrival→first placement), utilization, queue depth,
preemptions, and a same-shape priority-inversion counter.

**Run:** `.venv/bin/python .agents/projects/iris_multi_backend/spikes/S2_two_level_placement/run.py`
(deterministic, seed 7, ~30 s).

## Fidelity verdict — two-level tracks global (numbers)

Mixed federation, 96 workers, 267 jobs / 512 tasks over 60 ticks, ~58% global utilization:

| mode | placed | starved | wait mean/p50/p95/max | util | preempt |
|---|---|---|---|---|---|
| GLOBAL (single scheduler) | 512/512 | 0 | 1.17 / 0 / 8 / 13 | 58.1% | 52 |
| TWO-LEVEL (full summary) | 512/512 | 0 | **1.32** / 0 / 8 / 13 | 57.5% | 54 |

Two-level matches global on placement and starvation, costs **+0.15 tick mean wait** (p50/p95/max
identical), and **−0.6 pts utilization**. Per-backend utilization differs (two-level spreads v5e more
evenly: west 49.7% vs global 56.7%; central 59.0% vs 54.4%) but aggregate throughput is identical. The
deterministic `cw-east-h100` figure (73.9% in both) is the single-h100-backend case where routing is
forced and the two modes are provably identical.

**Verdict: two-level loses nothing meaningful on a well-provisioned mixed workload** — provided the
summary carries the load-bearing fields below. (An earlier run showed 17 spurious "starvations"; that
was a harness bug — see *Two findings for the spec*, item 1 — not a property of two-level.)

### What two-level genuinely loses (all accepted / bounded)

1. **Cross-backend rebalancing of a queued task.** A job pins at route time; if its backend later
   congests while an equivalent one drains, global could float the task over but two-level cannot. In
   the mixed workload this cost **+0.15 tick mean wait, 0 extra starvation** — the pinning tax is real
   but small when the router balances at route time.
2. **Cross-backend preemption** — accepted non-goal. Within-backend preemption is preserved (the agent
   runs `run_preemption_pass`): two-level even did *more* preemptions (54 vs 52) because it concentrates
   each band's contention inside one backend.
3. **Priority inversion** is essentially equal (same-shape inversion counter 104 vs 103). The residual
   is within-backend (preemptable) in both; the only inversions two-level *can't* fix are cross-backend,
   which is (2).

## Which summary fields are load-bearing (ablation numbers)

At moderate load the mixed workload doesn't stress any single field (all ablations stay 512/512). Each
field's value shows in a **targeted stress scenario**:

| field | scenario | with field | without field | verdict |
|---|---|---|---|---|
| **in-tick decrement** | BURST: 48 v5e jobs in one tick, 2×24-unit backends | 48/48, util 83.3%, wait 0 | 48/48 but util **50%**, wait p95 **25** | **ESSENTIAL** (but root-side bookkeeping, *not* a summary field) |
| **max_free_cpu_bin** | CPU-BIN: 48-core jobs; TPU boxes have more *total* free cpu (192) but tiny bins (8) | 24/36 placed | **0/36 placed** (total starvation) | **ESSENTIAL** for fungible CPU/RAM |
| **max_gang (structural)** | GANG: gang-of-4 vs a backend whose biggest slice is 2 | 16/16 | **12/16, 4 starved** | **ESSENTIAL** — but config-derivable, not dynamic |
| **largest_gang (dynamic free)** | GANG-BALANCE: gang-of-4, one slice 2/4 busy, an idle peer | wait **0** | 6/6 but wait **19** (p95) | load-bearing for **latency**, not starvation |
| **stale_ms** | PARTITION: one backend's agent unreachable, summary looks idle | 60/60 | **42/60, 18 starved** into a black hole | **ESSENTIAL** for fault handling (ties to S3) |
| **pending_leases** | LAG: oversubscribed, summary 2 polls stale | 128/192 | **128/192 (identical)** with `pending_leases` *or* `neither` | **PRUNE** — redundant |
| **static (in the summary)** | — | — | — | **PRUNE** — it is config, not capacity |
| **backoff** | (quota exhaustion not exercised) | — | — | KEEP-tentative; couldn't falsify |

### `pending_leases` is redundant — the real guard is in-tick + the root ledger

The "don't double-count in-flight capacity" function splits in two:

- **In-tick** (within one routing pass, decrement after each route): **essential** — BURST shows
  dropping it halves utilization (83%→50%) and forces a 25-tick wait, because every job in a burst sees
  the same stale free count and piles onto one backend. This is **pure root-side bookkeeping**, not a
  field the agent reports.
- **Cross-tick** (the summary lags the agent's placements): the root is **authoritative** over
  `task→backend` and the leases (design §0/§5), so it already knows every route it has issued — the
  *root ledger* (queue backlog + outstanding launches), recomputed each tick from its own DB. In the
  LAG scenario, `root-ledger`, `pending_leases`, and `neither` are **byte-identical** (128/192) because
  in-tick decrement plus the root's own queue knowledge already balance routing.

`pending_leases` (the agent's "launched-but-unobserved" count) is a strict **subset** of the root
ledger — a scalar, blind to the queue backlog, and not attributable to a device-shape — so it can
never replace the ledger, and the ledger is free. **Prune it.**

## Minimal sufficient `CapacitySummary`

```python
class CapacitySummary(BaseModel):
    # --- dynamic, sent every Poll (this is the whole summary) ---
    allocatable: dict[str, int]        # free placeable units per device-variant (accelerator slots)
    free_cpu_millicores: int           # fungible CPU: total free across the whole backend
    max_free_cpu_bin: int              # fungible CPU: largest single-worker free  (bin-fit guard)
    free_memory_bytes: int             # fungible RAM: total free
    max_free_memory_bin: int           # fungible RAM: largest single-worker free  (bin-fit guard)
    largest_gang: dict[str, int]       # free coschedulable gang now, per variant  (latency/balance)
    stale_ms: int                      # summary age; root discounts/zeroes a stale backend
    backoff: dict[str, int]            # per scale-group cooldown-until-ms          (keep-tentative)
```

Knowledge the root already has **from `BackendConfig` (not the per-Poll summary)**, and uses for
constraint matching + `UNSCHEDULABLE`/gang eligibility:

```python
# config-static, set-valued — never travels in the summary:
static_attributes: dict[str, set[str]]   # {device-variant, region, zone, provider, backend, ...}
max_gang: dict[str, int]                 # structural max group size per variant (= scale-group num_vms)
```

Changes vs the spec.md §3.1 strawman:

- **PRUNE `static`** from the summary — it mirrors `BackendConfig.attributes`, which the root already
  has; sending it every Poll is redundant. (Live availability is captured by `allocatable[variant]==0`.)
- **PRUNE `pending_leases`** — the root's authoritative ledger subsumes it (above).
- **EXTEND `allocatable`**: a per-shape free *count* is sufficient for indivisible accelerators but
  **not** for fungible CPU/RAM. Add `(free_*, max_free_*_bin)` per fungible resource. Without the
  bin field the root routes a big CPU job to the backend with the most *total* free cores even though no
  single box can host it → permanent starvation (CPU-BIN: 0/36).
- **SPLIT `largest_gang`** into a **static structural** `max_gang` (config-derivable; the
  starvation-preventing eligibility guard — already mirrors `route_demand`'s `num_vms == len(task_ids)`
  check) and the **dynamic free** `largest_gang` (the latency/balance signal that stays in the summary).
  Make both **per-variant** dicts, not a single int — a backend can host several gang shapes.
- **KEEP `stale_ms`** (fault-handling; quantify the threshold in S3) and **`backoff`** (tentative:
  distinguishes "full but autoscalable" from "full and quota-stuck"; orthogonal to S2's question, could
  not be falsified here).

## Two findings for the spec beyond field selection

1. **`--backend X` is a routing directive, not a worker constraint.** It must be **consumed by the
   root** and **stripped from `DesiredAttempt.constraints`** before the agent places — agents do not
   advertise a `backend` attribute, so a forwarded `backend EQ X` matches **no worker** and the pinned
   task starves. (This was the entire "17 starved" gap in an early run.) spec.md §1/§3.1 should state
   that routing-only constraints are filtered out of the per-attempt local constraints.
2. **Backend static attributes are set-valued; the worker matcher is scalar-per-key.** §3.1 says to
   "reuse the existing `Constraint`/`ConstraintIndex` matcher". A backend advertises e.g.
   `device-variant: {v5e-4, v5p-8}`, but `ConstraintIndex` stores one `AttributeValue` per (entity,key)
   and `route_demand` has the same one-variant-per-group shape. Backend routing therefore needs either
   per-`(backend, variant)` entity expansion or a small set-membership evaluator (the harness uses
   `_backend_matches`, evaluating `EQ` as set membership, exactly as §3.1's prose intends). Note this so
   "reuse the matcher" isn't read as drop-in.

## Honest limits of this spike

- Synchronous placement (no build/launch latency); `poll_lag` models summary staleness but not partial
  per-tick placement, so the cross-tick ledger's value is likely understated — yet even so
  `pending_leases` added nothing, which only strengthens the prune.
- `backoff` and multi-variant backends were modeled but not stress-tested to failure.
- Utilization in the headline run is ~58%; the pinning tax grows with saturation, but the targeted
  saturated scenarios (BURST/LAG at 83–98% util) still show two-level matching global once in-tick
  decrement is present.
