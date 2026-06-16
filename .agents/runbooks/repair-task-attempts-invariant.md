---
name: repair-task-attempts-invariant
description: Diagnose tasks vs task_attempts invariant breaks — ghost co-tenants, orphan attempts, split slices.
---

# Runbook: Diagnose tasks vs task_attempts invariant breaks

**When you're here:** The dashboard shows something impossible — a worker hosting
two TPU jobs at once, a task "running" on a worker that's actually free, or a
coscheduled job hanging with its tasks spread across two pods. Or `committed_*`
on a worker doesn't match its active tasks. These are almost always a broken
invariant between the `tasks` table and the `task_attempts` table, not a live
scheduling violation.

**TL;DR:**
- The dashboard worker-view reads `task_attempts.state`; the scheduler, PollTasks,
  and `committed_*` accounting read `tasks.state`. When a termination path forgets
  to finalize the attempt row, the two diverge and the dashboard lies.
- **Run the detection queries below** against an offline checkpoint to classify:
  stale active attempts on terminal tasks (**ghost co-tenants**), real split slices
  (coscheduled active tasks spanning >1 `md_tpu_name`), or a committed-resource leak.
- **Both bugs can be true at once.** A ghost UI artifact can sit on the same worker
  as a *real* split-slice hang. Don't stop at the first explanation.
- This is an **index/detection** runbook. The known instances are fixed by
  migrations 0036/0038/0039. Your job is to identify which one you're seeing and
  route to the fix or escalate — **not** to hand-write repair SQL.

## Before you touch anything

- **NEVER modify the controller database.** Read-only queries only, even on an
  offline checkpoint (lib/iris/OPS.md:113). The repairs in this family are *startup
  migrations* that run inside `apply_migrations()` before any worker reconnects —
  they are not ad-hoc UPDATEs you run by hand. If a snapshot shows broken rows,
  route to the migration; do not invent repair SQL.
- **Do these reads offline.** Run the detection queries against a checkpoint, not
  the live DB — these are joins over `task_attempts`, which can be large
  (30K+ rows seen in the wild). Expensive queries against the live DB stall the
  controller (lib/iris/OPS.md:151). Prefer copying the last GCS checkpoint over
  taking a fresh one — see `offline-checkpoint-analysis`.
- **Snapshot age matters.** A `/tmp/iris-debug/controller.sqlite3` from an earlier
  checkpoint may pre-date the incident. If a query returns "0 problems" against the
  snapshot, confirm against the live controller with a single cheap `iris query`
  before concluding it's clean.

## Diagnose

State codes you need (full legend lib/iris/OPS.md:115): active = **2 (BUILDING),
3 (RUNNING), 9 (ASSIGNED)**; terminal = 4..8 (SUCCEEDED/FAILED/KILLED/WORKER_FAILED/
UNSCHEDULABLE). Forgetting that ASSIGNED is active is the classic misdiagnosis
(lib/iris/OPS.md:119).

**1. Ghost co-tenants / orphan attempts — active attempt rows on terminal tasks.**
The dashboard says "task running on worker X" forever after the worker was freed.
The attempt row was never finalized; the `tasks` row is terminal.

```sql
SELECT count(*)
FROM task_attempts ta JOIN tasks t ON ta.task_id = t.task_id
WHERE ta.state IN (2,3,9) AND ta.finished_at_ms IS NULL
  AND t.state IN (4,5,6,7,8);
```

Non-zero ⇒ orphan attempts. If they're nearly all `t.state=6` (KILLED) with task
error "Terminated by user", they came from `cancel_job` (the 2026-04-28 bug). To
confirm a specific dashboard report is a ghost and not a real co-tenant,
**cross-check committed resources on the suspect worker**: if `committed_tpu`
equals exactly one task's worth and the worker's heartbeat is healthy, the extra
attempt is a ghost, not a co-tenant (lib/iris/OPS.md:120 for the committed schema).

A sibling orphan class is a **superseded attempt** — an active attempt whose
`attempt_id != tasks.current_attempt_id`. Migration 0038 covers both. The
`current_attempt_id` direction of this invariant (a stale `current_attempt_id`
pointing past surviving attempt rows) is the 2026-04-21 scheduler-freeze family,
healed by migration 0036.

**2. Real split slice — coscheduled active tasks spanning >1 TPU pod.**
The job *hangs* (SPMD collective can't form), independent of any UI ghost.

```sql
SELECT j.job_id, COUNT(DISTINCT w.md_tpu_name) AS slices,
       GROUP_CONCAT(DISTINCT w.md_tpu_name) AS names
FROM jobs j
JOIN job_config jc ON jc.job_id = j.job_id
JOIN tasks t ON t.job_id = j.job_id
JOIN workers w ON w.worker_id = t.current_worker_id
WHERE jc.has_coscheduling = 1 AND j.is_reservation_holder = 0
  AND t.state IN (2,3,9) AND w.md_tpu_name != ''
GROUP BY j.job_id HAVING COUNT(DISTINCT w.md_tpu_name) > 1;
```

Any row ⇒ a genuinely split coscheduled job. Migration 0039 requeues these.

**3. Committed-resource leak — workers carrying committed_* with no active tasks.**
Known Bug #1 (lib/iris/OPS.md:222). Join `workers` against active `task_attempts`:
a worker with high `committed_cpu`/`mem`/`tpu` but no active task is a leak. Symptom
overlaps with ghost co-tenants, so run query 1 first to rule that out.

**Both can be true.** On 2026-04-28 the suspect worker had a real co-tenant from a
preemption interaction *and* the orphan UI artifact, both surfaced by the same
dashboard query. Run all three; classify each finding separately.

## Resolve

You don't repair these by hand. Route by what the queries found:

- **Orphan / superseded attempts (query 1):** the fix is the startup migration —
  0038 (`finalize_orphan_attempts`) for terminal-task and superseded-attempt
  orphans, 0036 (`reconcile_reservation_holder_attempt_ids`) for the stale
  `current_attempt_id` holder case. If the migration already shipped, the rows heal
  on the next controller restart, before any worker reconnects. If it hasn't, the
  fix is a code change + migration, not a live edit — escalate to the iris owner.
- **Split slice (query 2):** migration 0039 (`requeue_split_coscheduled_jobs`)
  decommits each split task's resources and requeues to PENDING. Same deal — it runs
  at restart; if the live cluster has actively-hanging split jobs and no migration
  yet, escalate rather than touching the DB.
- **Committed leak (query 3):** this is a known unfixed leak path
  (lib/iris/OPS.md:222). Don't reconcile by hand — file/escalate with the worker id
  and the leaked `committed_*` values.

To deploy a shipped migration you rebuild + restart the **controller only** (it runs
migrations at startup) — see [deploy-controller-fix](deploy-controller-fix.md).
Never run a full `iris cluster restart` for this.

## Verify

- **Re-run the detection query** that flagged the problem against a *fresh* post-fix
  checkpoint (or a cheap live `iris query`) — it should return 0. The process coming
  back up is not proof; the row count is.
- **Cross-check the dashboard against accounting:** the worker-view (reads
  `task_attempts.state`) should now agree with `committed_*` and the scheduler's
  `tasks.state` view. If the dashboard still shows the ghost but query 1 is empty,
  you're looking at a stale checkpoint — re-pull.
- **For a split-slice fix:** confirm the previously-hung job either completed or its
  tasks now share a single `md_tpu_name` (re-run query 2 filtered to that job_id).

## Why this happens

The root class is a **cross-table invariant violation** between `tasks` /
`tasks.current_attempt_id` and `task_attempts`. Several termination paths historically
updated one side and not the other:

- `cancel_job` (`transitions.py:1083-1120`) decommitted worker resources and killed
  the `tasks` rows but never finalized the in-flight `task_attempts` — leaving them
  `state=RUNNING, finished_at_ms=NULL` forever. Because the dashboard worker-view
  reads `task_attempts.state`, those rows surfaced as phantom running tasks (30,382
  of them over one week). See `.agents/ops/2026-04-28-iris-split-slice-and-orphan-attempts.md`.
- The same incident's *second*, independent bug: a transient failure of one task in a
  coscheduled job downgraded it to PENDING without cascading to siblings, so the
  scheduler re-placed the lone task on any free worker group — splitting the SPMD mesh
  across pods (`scheduler.py:668-744`, gate at `transitions.py:1473`). A real hang, not
  a UI artifact — which is why the postmortem's load-bearing lesson is "both bugs are
  true."
- The holder variant: `fail_worker`'s reservation-holder branch reset
  `tasks.current_attempt_id = -1` while leaving old attempt rows in place; the next
  assignment computed a colliding `attempt_id` and the scheduler thread crashed with
  `IntegrityError: UNIQUE constraint failed: task_attempts.task_id, task_attempts.attempt_id`,
  silently freezing scheduling. See
  `.agents/ops/2026-04-21-iris-scheduler-freeze.md`.

The general guardrail these incidents bought: audit *every* code path that downgrades
a task to PENDING or kills a task for whether it also finalizes the attempt row and
cascades to coscheduled siblings.

## See also

- lib/iris/OPS.md:117 "Sharp edges" — active states {2,3,9}, `committed_*` vs
  `metadata_proto`, and the read-path divergence (dashboard reads `task_attempts.state`,
  scheduler/`committed_*` read `tasks.state`).
- lib/iris/OPS.md:222 "Known Bugs" #1 — the committed-resource leak.
- `offline-checkpoint-analysis` — pull the last GCS checkpoint and run these queries
  without stalling the live controller.
- [deploy-controller-fix](deploy-controller-fix.md) — rebuild + controller restart to
  apply a shipped migration.
- `.agents/ops/2026-04-28-iris-split-slice-and-orphan-attempts.md` — ghost co-tenants
  vs real split slice; migrations 0038/0039; the "both bugs are true" framing.
- `.agents/ops/2026-04-21-iris-scheduler-freeze.md` — the `current_attempt_id`
  IntegrityError / silent scheduler freeze; migration 0036.
