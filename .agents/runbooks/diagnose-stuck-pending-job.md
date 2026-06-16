---
name: diagnose-stuck-pending-job
description: A job sits PENDING — distinguish a genuine capacity/quota wait from a frozen scheduler from a reservation-taint stranding.
---

# Runbook: Diagnose a job stuck PENDING

**When you're here:** A job (or its `:reservation:` sub-job) has been PENDING far
longer than its resources warrant, and you need to know whether to wait, redeploy,
or resubmit. The three causes look identical from the dashboard — the fix for each
is different.

**TL;DR:**

- **First move is not a per-job command.** Look at the *spread* of pending-reason
  text across all pending jobs. Uniform text on many unrelated jobs => the
  scheduler is frozen, not a capacity problem.
- If reasons are varied and one names quota/capacity, it's a real wait — check
  `scaling_groups.quota_reason` and autoscaler backoff.
- If a `--reserve` parent is PENDING while its `:reservation:` holder is RUNNING,
  it's the EQ-taint bug — the parent is pinned to TPU workers and can never land
  on CPU.

## Before you touch anything

- **Read-only first.** Every diagnostic here is a query or RPC. Don't restart the
  controller, resubmit, or checkpoint until you know which branch you're in. A
  controller restart is destructive to in-flight scheduling state and needs a human
  yes (see [deploy-controller-fix](deploy-controller-fix.md)).
- **Don't trigger a live checkpoint to investigate.** `iris cluster controller
  checkpoint` briefly stalls the controller — wrong when scheduling is already
  wedged. Pull the last GCS checkpoint instead (lib/iris/OPS.md:151).
- **Baseline:** the job's pending_reason and queue_position from
  `get-scheduler-state`, plus the pending_reason of a handful of *other* pending
  jobs — you need the spread, not just this one.

## Diagnose

**Step 0 — read the spread, not the job.** Pull the pending queue and compare the
pending-reason text across jobs (see lib/iris/OPS.md:93 "Scheduler & Autoscaler"
for `get-scheduler-state`).

- **Many/all pending jobs carry the *same* fallback reason text** (e.g. the
  literal `"Pending scheduler feedback"`) => **Branch (b), frozen scheduler.**
  Uniform text means the per-job diagnostic cache stopped updating — the scheduling
  thread is dead and you're reading a stale snapshot. Skip capacity analysis.
- **Reasons are varied and job-specific** => a real per-job cause; go to (a) or (c).

### Branch (a) — genuine capacity / quota wait

Hypothesis: the scheduler is alive and correctly reporting that nothing can host
this job yet.

- `iris rpc controller get-scheduler-state` — confirm the constraint is resource/
  quota, not exotic.
- Quota-blocked groups: the `scaling_groups` query in lib/iris/OPS.md:123 ("Useful
  queries") — non-empty `quota_reason` or rising `consecutive_failures` means GCP
  quota is the wall.
- `iris rpc controller get-autoscaler-status` — check `backoff_until_ms` /
  `consecutive_failures`. The autoscaler backs off exponentially per group; quota
  is the primary scaling bottleneck (lib/iris/OPS.md:277 "GCP Gotchas").

If quota/backoff explains it, this is a wait, not a bug. Resolve via branch (a).

### Branch (b) — frozen scheduler (the net-new heuristic)

Triggered by step 0's uniform-pending-reason smell. The scheduling `ManagedThread`
caught an exception, logged it, and **returned without respawning** — the cluster
still heartbeats and looks alive, but no cycle runs and no diagnostics cache.

Confirm:

- READY idle slices exist in the job's pool yet queue_position never falls —
  capacity is not the issue.
- Grep recent controller parquet logs for the silent death and originating
  traceback — `ManagedThread.*crashed`, `scheduling-loop crashed` /
  `IntegrityError` on `key='/system/controller'`. For the duckdb-over-GCS recipe
  (no local download), see [offline-checkpoint-analysis](offline-checkpoint-analysis.md).
- Checkpoint cross-table check: any task where
  `max(task_attempts.attempt_id) > tasks.current_attempt_id` is a poisoned row that
  crashes the next assignment cycle.

A **stale controller** can present the same way (scheduling weirdness after a
deploy that never shipped). Rule it in/out via
[deploy-controller-fix](deploy-controller-fix.md).

### Branch (c) — reservation-taint stranding

Hypothesis: a `--reserve` parent is stuck PENDING while its `:reservation:` holder
is RUNNING.

- `iris query "SELECT * FROM reservation_claims"` and check the holder's state
  (lib/iris/OPS.md:281 "Reservation system"). Holder RUNNING + parent PENDING is
  the tell.
- The parent is `has_direct_reservation`, so the scheduler injects a
  `reservation-job == <self>` **EQ taint** pinning it to the reservation's workers
  (the TPU). If the parent's own task only needs CPU, on-demand CPU workers never
  carry that taint, so it can **never** schedule there
  (`lib/iris/src/iris/cluster/controller/scheduling/policy.py:911-928`).
- Corroborate phantom demand: the autoscaler can't see the injected taint, so it
  boots CPU VMs (`ready=N`, idle, then scaled down) while the parent stays PENDING —
  an autoscaler/scheduler disagreement, not a capacity shortfall.

## Resolve

- **(a) Capacity/quota:** wait, or raise quota / reduce request. Transient backoff
  clears on its own — don't poke it. If the pool genuinely can't grow, the job is
  correctly UNSCHEDULABLE; adjust the request.
- **(b) Frozen scheduler:** the thread does not self-respawn. Recovery is a
  controller restart — **human-approval gate** — to relaunch it; run the relevant
  migration if a poisoned cross-table row is the cause (the scheduler-freeze fix
  advances `current_attempt_id` to `max(attempt_id)` on restart). Restart on the
  *fixed* image via [deploy-controller-fix](deploy-controller-fix.md), not a stale
  `:latest`.
- **(c) Reservation-taint:** stop creating the taint — resubmit the orchestrator as
  a plain CPU job (drop `--reserve`) and let the training child acquire the TPU
  itself, or request a device-variant `IN` OR-match so the work isn't pinned to one
  churning pool. A controller restart will *not* fix this; it's a request-shape bug.

## Verify

- **(a):** queue_position falls and the job transitions to BUILDING/RUNNING once a
  slice is READY. `get-autoscaler-status` shows backoff clearing.
- **(b):** controller logs show `Scheduling cycle` lines resuming after restart;
  pending-reason text on other jobs becomes varied/job-specific again; the stuck
  job's queue_position starts moving. Confirm no new `ManagedThread.*crashed`.
- **(c):** the resubmitted parent reaches RUNNING on a CPU worker within seconds,
  and the autoscaler stops booting idle CPU VMs for phantom demand.

Active states are 2 (BUILDING), 3 (RUNNING), **and 9 (ASSIGNED)** — a job leaving
PENDING into ASSIGNED is progress, not a stall (lib/iris/OPS.md:117 "Sharp edges").

## Why this happens

- **Frozen scheduler:** `ManagedThread._safe_target` catches-logs-returns with no
  supervisor respawn. Apr 21: a `fail_worker` reservation-holder branch broke the
  `tasks.current_attempt_id` vs `task_attempts` invariant, the next cycle hit
  `sqlite3.IntegrityError`, and the thread died silently — ~90 min of no scheduling
  while the cluster looked healthy and every pending job showed one fallback string.
  `.agents/ops/2026-04-21-iris-scheduler-freeze.md` (cause `transitions.py:2170-2180`;
  fix `:2167-2186`).
- **Reservation-taint:** a directly-reserved CPU parent gets an EQ taint pinning it
  to the reservation's TPU workers and can never land; the autoscaler emits phantom
  CPU demand for a taint it can't see. Jun 8: the canary burned a full 6h wall clock
  holding an idle TPU. `.agents/ops/2026-06-08-canary-ferry-reservation-taint-timeouts.md`
  (mechanism `policy.py:911-928`).

## See also

- lib/iris/OPS.md:209 — Troubleshooting matrix ("Job stuck PENDING", "Autoscaler
  not scaling") and lib/iris/OPS.md:93 / lib/iris/OPS.md:123 for the exact
  `get-scheduler-state`, `quota_reason`, and `scaling_groups` commands.
- `.agents/ops/2026-04-21-iris-scheduler-freeze.md` — frozen scheduler.
- `.agents/ops/2026-06-08-canary-ferry-reservation-taint-timeouts.md` — reservation
  taint / phantom demand.
- [deploy-controller-fix](deploy-controller-fix.md) — a stale controller presents
  as scheduling weirdness; this is also how you restart on the fixed image.
- [offline-checkpoint-analysis](offline-checkpoint-analysis.md) — deep
  duckdb-over-GCS log and checkpoint queries.
- `babysit-job` skill — points operators here when a watched job won't leave
  PENDING.
