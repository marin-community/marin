---
date: 2026-06-08
system: ci / canary-ferry / iris
severity: degraded
resolution: mitigated
pr: 6289
issue: none (iris reservation-job-taint follow-up recommended — draft below)
---

# TPU canary ferry: 5 days of timeouts (reservation-job taint + v5p stockout)

## TL;DR

- `marin-canary-ferry.yaml` (the daily TPU canary) has been **red for 8 consecutive
  scheduled runs** (Jun 1 → Jun 8). May 16–31 were all green (~2h each).
- There are **two distinct failure modes**, not one:
  1. **Jun 1–4 — reservation holder stuck PENDING (`failure` at ~3.5h).** The
     `:reservation:` holder for `--reserve v5p-8` inherited the parent's
     `preemptible=false` and searched for a non-existent non-preemptible v5p-8
     group. Code fix #6141 merged Jun 3, but the **marin controller was not
     redeployed**, so it recurred. Triaged as #6164 (closed Jun 4 after redeploy).
  2. **Jun 5–8 — CPU orchestrator stuck PENDING (`cancelled` at the 6h job
     timeout).** After the redeploy fixed the holder, the failure flipped: the
     v5p-8 `:reservation:` holder reaches RUNNING in seconds, but the **CPU
     orchestrator parent never schedules** and the whole job burns the full 6h
     GitHub wall clock holding a TPU reservation idle, then is force-cancelled.
- Root cause of mode 2 is an **Iris scheduler bug + a capacity squeeze, masked by a
  monitor blind spot**:
  - A `--reserve` parent is `has_direct_reservation`, so the scheduler injects a
    `reservation-job == <self>` **EQ taint** that pins it to the *workers claimed
    for the reservation* (the v5p-8 TPU). The parent's own task only needs CPU,
    and on-demand CPU workers never carry that taint, so it can **never** schedule
    on CPU. (`lib/iris/src/iris/cluster/controller/scheduling/policy.py:911-928`.)
    The intended design is that the orchestrator co-locates on the reserved v5p-8
    VM — which works *only when the reservation actually holds a healthy v5p-8
    worker*.
  - v5p-8 is **preemptible-only and lives in just two zones** (us-central1-a,
    us-east5-a). The Jun 8 controller log is wall-to-wall `Bootstrap failed … no
    more capacity in zone` for v5p, alongside a large competing v6e job — so the
    reservation churns and rarely holds a live worker, leaving the taint-pinned
    parent stranded.
  - The autoscaler can't see the injected taint (it's not persisted), so it routes
    **phantom CPU demand** to `cpu_vm_e2_highmem_2_ondemand`, boots CPU VMs that
    sit idle (`ready=3`, idle 306s) and get scaled down while the parent stays
    PENDING — an autoscaler/scheduler disagreement.
  - The monitor (`iris_monitor.py wait`) **drops its 3.5h queue timeout the instant
    the `:reservation:` holder reaches RUNNING** (`_child_started`, line 230) — at
    +6s on Jun 8 — so it never fails fast; it waits the full GitHub wall clock.
- **Shipped in this PR (ferry-local, validated):**
  1. **Monitor fix** — the synthetic `:reservation:` holder no longer counts as a
     started child, so a stranded parent now fails fast at `child-wait-timeout`
     (~3.5h) instead of hanging to the 6h cancel. The real-work child
     (`<parent>/grug-train-*`) is still watched.
  2. **Canary no longer hard-pins to v5p-8 and no longer reserves up front** — the
     training now requests `with_tpu(["v5p-8","v4-8"])` (a `device-variant IN`
     OR-match; v4-8 adds us-central2-b incl. the v4 **reserved** pool, same 4-chip
     slice shape), and `--reserve` is dropped so the orchestrator runs as a plain
     CPU job (no taint) while the training child acquires the TPU on its own.
- **Filed for follow-up:** an Iris-core issue for the taint bug (a directly-reserved
  parent whose own resource request doesn't match the reservation device should not
  be EQ-pinned to the reservation workers; and the autoscaler emits phantom demand
  for the unseen taint).

## Evidence

### Run history (`gh run list --workflow=marin-canary-ferry.yaml`)

| Date | Run | Conclusion | Wall | Mode |
| --- | --- | --- | --- | --- |
| 05-16 … 05-31 | — | success | ~2h | healthy |
| 06-01 | 26752277350 | failure | 1h10m | (fast fail) |
| 06-02 | 26813618112 | failure | 3h40m | holder PENDING (#5760/#6141) |
| 06-03 | 26880322441 | failure | 3h44m | holder PENDING |
| 06-04 | 26944533426 | failure | 3h47m | holder PENDING → #6164 |
| 06-05 | 27007932034 | **cancelled** | 6h05m | parent PENDING / taint |
| 06-06 | 27057525777 | **cancelled** | 6h05m | parent PENDING / taint |
| 06-07 | 27087927375 | **cancelled** | 6h05m | parent PENDING / taint |
| 06-08 | 27132773965 | **cancelled** | 6h05m | parent PENDING / taint |

The Jun 1–4 `failure`s land at ~3.5h = `CANARY_CHILD_WAIT_TIMEOUT` (12600s): the
holder never reached RUNNING, so the monitor's queue timeout correctly fired. The
Jun 5–8 `cancelled`s run the full 6h: the holder *did* reach RUNNING, so the queue
timeout was dropped and only the GitHub job timeout stopped the run.

### Jun 8 monitor + diagnostics (run 27132773965)

- `10:53:53` (+6s into the wait): `Child …/:reservation: reached JOB_STATE_RUNNING;
  dropping queue timeout.`
- Parent state across the whole run: **`JOB_STATE_PENDING` in 718/718 poll samples**
  — the orchestrator never ran.
- `job-tree.json`: parent PENDING, reason
  `Scheduler: No worker matches constraints … constraints=['preemptible',
  'reservation-job'] … Autoscaler: Waiting for workers in scale group
  'cpu_vm_e2_highmem_2_ondemand-us-east1-b' to become ready`; child `:reservation:`
  `JOB_STATE_RUNNING`.
- Controller log: CPU group `cpu_vm_e2_highmem_2_ondemand-us-east1-b` reaches
  `ready=3/1` and scales down a slice **idle for 306s** while the parent stays
  PENDING (definitive matching failure, not capacity); meanwhile relentless v5p
  `Bootstrap failed … no more capacity in zone us-east5-a / us-central1-a` plus a
  competing `tpu_v6e-preemptible_*-us-east5-b` job at demand 47–48.

### Why the parent can't schedule (Iris source)

- `preemptible=false` on the parent comes from the executor heuristic for a small
  job (`constraints.py:810-846`) — an on-demand CPU worker **does** satisfy it.
- The blocker is the `reservation-job` EQ taint injected for
  `has_direct_reservation` jobs (`policy.py:911-928`, classification at
  `policy.py:1158-1162`). It matches only workers claimed for the reservation
  (the v5p-8). An on-demand CPU worker has no such attribute →
  `evaluate_constraint(None, EQ)` is False (`constraints.py:911-912`) → never
  schedules.
- `cpu_vm_e2_highmem_2_ondemand` is the **only** `device_type: cpu` group
  (`lib/iris/config/marin.yaml:178-184`), and it is `capacity_type: on-demand`.

## Fixes in this PR

### 1. `scripts/workflows/iris_monitor.py` — stop the reservation holder masking a stuck parent

`_pick_child` now excludes the `:reservation:` holder, so the
`child-wait-timeout` is dropped only when a real-work child (the training job)
reaches RUNNING. A stranded parent fails fast at ~3.5h instead of burning 6h and
holding a TPU idle, and Claude triage runs with full signal. Tests added in
`tests/workflows/test_iris_monitor.py`.

This is independent of the canary changes and is safe for every `--reserve`
caller of the monitor.

### 2. `experiments/ferries/canary_ferry.py` + `marin-canary-ferry.yaml` — flexible TPU, no up-front reservation

- Training resource is now `ResourceConfig.with_tpu(_tpu_types_from_env())`,
  default `("v5p-8", "v4-8")` (override via `CANARY_TPU_TYPE`, comma-separated,
  primary first, or the `tpu_type` workflow_dispatch input). This becomes a single
  `device-variant IN {v5p-8, v4-8}` constraint that both the scheduler and
  autoscaler OR-match, so the run lands on whichever pool has capacity. v5p-8 and
  v4-8 are both single-VM 4-chip slices, so the training shape is unchanged.
- `--reserve v5p-8` is **removed** from the submit. The orchestrator is now a plain
  CPU job (no `reservation-job` taint → schedules on the CPU group), and the
  `grug-train-*` child acquires its TPU by emitting its own device demand
  (`--reserve` was only a pre-warm optimization, not a requirement — verified in
  `policy.py:167,290-353` and `iris/client/client.py:649-657`).
- The validate step's `continue-on-error` guard is extended to `tpu_type` overrides
  (the override changes the versioned output-path hash, which the validate step
  recomputes from defaults). mirror:// already scans us-central2, so a v4 run's
  output is readable.

Strictly safer than the status quo: v5p-8 is still primary (coverage preserved when
v5p is available), v4-8 only adds a same-shape fallback, and dropping `--reserve`
removes the taint that caused the 6h hang.

## Evaluating the two ideas from the request

- **"Make the ferry accept any TPU type."** Partially adopted. Literal "any type"
  is not possible: `with_tpu` forbids mixing slices with different `chips_per_vm`,
  and v5p-8 (4 chips/VM) cannot be pooled with v6e-8/v5litepod-8 (8 chips/VM) — the
  `-8` suffix means *cores* on v5p but *chips* on v6e, i.e. different compute. The
  topology-compatible fallback is `v4-8`, which is what we use. (To also accept
  v6e/v5e the canary would need a separate 8-chip slice config — possible, but it
  changes what's validated; left as an option via `CANARY_TPU_TYPE`.)
- **"Or not require to run on a TPU."** The orchestrator already runs on CPU; the
  TPU is the point of the canary, so the *training* must stay on a TPU. What we
  removed is the up-front **reservation** (`--reserve`), which is what pinned the
  CPU orchestrator to the TPU and caused the hang. So the spirit of this idea is
  adopted: the ferry no longer *reserves* a TPU, it just requests one when training.

## Follow-ups

- **Iris-core (filed):** a directly-reserved parent whose own resource request
  (CPU) does not match the reservation entry's device (TPU) should not be EQ-pinned
  to the reservation workers; and the autoscaler should not emit phantom demand for
  the unseen `reservation-job` taint. Fixing this would let `--reserve` be used
  safely again (e.g. to pre-warm scarce capacity) without stranding the parent.
- **Validation:** trigger one `workflow_dispatch` run (optionally with
  `tpu_type=v4-8`) to confirm the flexible request schedules and the canary passes
  end-to-end before relying on the daily cron.
- **Priority note:** without `--reserve`, the `grug-train-*` child competes for TPU
  at its default priority rather than holding pre-warmed production capacity. The
  v5p/v4 + reserved-pool fallback compensates, but if pass-rate under contention is
  still low, consider setting the child's priority explicitly.

## Draft Iris-core issue (file when ready)

> **Title:** [iris] --reserve EQ taint pins a CPU orchestrator parent to its TPU reservation workers (never schedules; phantom CPU demand)

A job submitted with `iris job run --reserve <tpu> --cpu=N --memory=M -- <cpu orchestrator>`
can be permanently stranded PENDING: the scheduler injects a `reservation-job == <self>`
EQ taint that pins the *parent* to the workers claimed for the reservation (the TPU), but
the parent's own task only requests CPU and can never run on a TPU-only claim — while no
CPU worker carries the taint either. The autoscaler never sees the injected taint, so it
routes phantom CPU demand to the on-demand CPU group, booting workers that idle and are
scaled down.

Mechanism: `policy.py:1158-1162` (classification) → `policy.py:911-928` (EQ-taint inject,
wired at `backend.py:316`) → taint only on claimed workers (`policy.py:858-884`, matched by
`policy.py:742-761`) → `constraints.py:911-912` (`evaluate_constraint(None, EQ)` is False) →
autoscaler routes from persisted constraints only (`policy.py:305-313`).

Expected: a directly-reserved parent whose own resource request does not match the
reservation entry's device should schedule on a worker matching its own request (only its
reservation-*descendant* should be admitted onto the claimed TPU workers); and demand
routing should not select a group the scheduler will then reject.

Workaround used by the canary: drop `--reserve` (orchestrator is a plain CPU job; training
child acquires the TPU on its own). Fixing this would let `--reserve` pre-warm scarce
capacity safely again.
