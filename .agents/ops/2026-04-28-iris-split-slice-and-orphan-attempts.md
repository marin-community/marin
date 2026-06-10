---
date: 2026-04-28
system: iris
severity: degraded
resolution: fixed
pr: none (branch: rjpower/20260428-fix-scheduling)
issue: none
---

## TL;DR

- Dashboard reported a v5p-64 worker hosting two TPU jobs simultaneously, suggesting a coscheduling violation.
- Two independent bugs were entangled. The "two jobs on one worker" report was a UI artifact from 30,382 stale `task_attempts` rows left active by `cancel_job` (`transitions.py:1083`) — workers and `committed_*` accounting were correct.
- The *real* incident — coscheduled v5p-64 training jobs hanging because their tasks ended up across two physical TPU pods — was a separate bug. Transient task failures (TPU init failure, host crash) returned the failed task to PENDING alone; siblings stayed RUNNING; the scheduler placed the retry on whichever slice had a free worker, splitting the SPMD mesh.
- Fixed both. Two new migrations (0038 finalize orphan attempts, 0039 requeue currently-split coscheduled jobs) plus a sibling-cascade in three failure paths in `transitions.py`. Drive-by: `apply_heartbeats_batch` was discarding `task_kill_workers` from the result, exposed by codex review.
- Live cluster post-investigation: 30K orphans waiting for migration 0038; 0 currently-split coscheduled jobs (incident jobs already terminated naturally).

## Original problem report

User shared the dashboard URL `https://iris.oa.dev/#/worker/marin-tpu-v5p-preemptible-64-us-central1-20260428-1031-d11be78c-worker-0` and stated: "we have a worker which seems to have been assigned 2 TPU tasks ... this should be forbidden by co-scheduling constraints". Asked for help debugging the scheduler in `lib/iris`. Subsequent assertion: "we had real problems from it" — confirming jobs were actually failing/hanging, not just looking wrong on a dashboard.

## Investigation path

1. Read `lib/iris/OPS.md` and listed the scheduler / constraints source. Located `_find_coscheduled_assignments` at `lib/iris/src/iris/cluster/controller/scheduler.py:668-744`.

2. Queried the live controller for active task_attempts on the suspect worker using `iris query` (config `lib/iris/examples/marin.yaml`, auto-tunnels). Found three: ahmed-r3 train_dpo/0 attempt 2, moojink-065137 train_lm/0 attempt 0, and a CPU rav/normalize task. The TPU pair looked like a real coscheduling violation.

3. Pulled all active attempts of both TPU jobs to map the slice topology. Ahmed-r3 train_dpo distributed across slice `1031-d11be78c` (workers 0,4,7) AND slice `1200-98ca1f3f` (workers 1,2,3,6) — a coscheduled job split across two TPU pods. Moojink-153141 likewise across slices `1031` and `1536-6b9deae1`. Three coscheduled jobs apparently co-tenanting slice `1031`.

4. Read `_find_coscheduled_assignments` (`scheduler.py:668-744`). The `num_tasks` value comes from `tasks_by_job[job_id]` which is built from `pending_tasks` only (line 614-618). For a job with N-1 RUNNING and 1 PENDING the singleton is placed on *any* worker group with ≥1 free slot, with no anchoring to where its siblings already live. Suspected that combined with the recent same-variant slice preemption (commit `e98304258`, #5240), preemption was producing partial-PENDING coscheduled state at scale.

5. Spawned two parallel agents: a senior-engineer agent to audit a week of iris commits for scheduling regressions; a general-purpose agent to grep controller logs for preemption-pass evidence around the suspect timeframe. Both came back with useful but partially misleading findings.

6. Senior-engineer agent reported `scheduler.py` had had zero changes in the past week; correctly fingered the partial-PENDING combine pattern and recommended changing the cascade in `apply_state_updates`.

7. Log agent reported no `task_preempted` log line for `/moojink/.../065137/train_lm-a561d9a157` anywhere in 12h of controller logs — and no termination event at all for that job, even though its `tasks` row was state KILLED. This led to a separate diagnosis (later confirmed): `cancel_job` (`transitions.py:1083-1120`) updates `tasks` / `jobs` / `workers` but never finalizes the corresponding `task_attempts` rows. Compare `_terminate_task` (`transitions.py:356-413`) which calls both `attempts.mark_finished` and `workers.decommit_resources`; `cancel_job` only does the latter.

8. User shared a controller checkpoint at `/tmp/iris-debug/controller.sqlite3`. Verified: `SELECT count(*) FROM task_attempts ta JOIN tasks t ON ta.task_id=t.task_id WHERE ta.state IN (2,3,9) AND ta.finished_at_ms IS NULL AND t.state IN (4,5,6,7,8)` returned **30,382 orphan rows**, all with task error "Terminated by user", spanning 22 users over a week (top: `/rav/` 15K, `/michaelryan/` 8K). Every orphan had `task_state=6` (KILLED) — confirming `cancel_job` was the sole source.

9. Briefly mis-attributed the entire incident to orphan attempts. User course-corrected ("i think both bugs are true", "wait so what was causing our split-slice issues, i thought that was a real bug?"). Re-separated the two bugs: orphan attempts explained the *dashboard* misread, but the actual job hangs were from the partial-PENDING split-slice path.

10. Designed the cascade fix: `_requeue_coscheduled_siblings` mirrors `_terminate_coscheduled_siblings` but bounces siblings to PENDING (attempt → PREEMPTED) with their `preemption_count` and `failure_count` untouched. Only the originally-failing task pays its retry budget. Wired into `apply_state_updates`, `apply_direct_provider_updates`, and `_remove_failed_worker`. Branch is gated on `update.new_state in FAILURE_TASK_STATES` (so the gate fires regardless of whether retry budget downgrades the task back to PENDING) and chooses the requeue path when `task_state == PENDING`, the existing terminate path otherwise.

11. Added `bulk_finalize_active` to `TaskAttemptStore` (mirrors `bulk_kill_non_terminal`) and called it from `cancel_job`. Wrote migration `0038_finalize_orphan_attempts.py` to heal the 30K historical orphans at controller startup. Initially shipped as a `scripts/fix-orphan-attempts.py` standalone; user redirected to migration form ("could we write it as a migration instead?") which is cleaner and runs before any worker reconnects.

12. After arguing migration 0039 wasn't needed because the live snapshot showed 0 currently-split coscheduled jobs, user pushed back with "change it TO FIX SPLIT SLICE". Wrote `0039_requeue_split_coscheduled_jobs.py` — finds coscheduled jobs whose active-task workers span >1 distinct `md_tpu_name`, decommits each task's resources from its worker (matching `workers.decommit_resources` math sourced from `job_config`), marks attempts PREEMPTED, resets tasks to PENDING. Idempotent.

13. Codex review (`/codex-review`) flagged: (a) `apply_heartbeats_batch` at `transitions.py:1711` wraps results as `TxResult(tasks_to_kill=...)` discarding `task_kill_workers` — pre-existing bug, exposed by every cascade going through batched heartbeats; (b) GPU decommit in 0039 used raw `json_extract($.gpu.count)` but runtime `types.get_gpu_count` returns `device.gpu.count or 1`; (c) direct-provider coscheduling concern. User confirmed (c) doesn't exist; fixed (a) and (b).

14. Verified live cluster via `iris query`: 0 currently-split coscheduled jobs across 10 active coscheduled jobs (all single-slice). The split-slice incidents from earlier in the week have all terminated naturally. Migration 0039 will be a no-op at restart against current state — kept as forward defense.

## User course corrections

- "send an engineer to look over the changes in Iris for the past week ... send another engineer to investigate the logs to see if we tried & failed to preempt the jobs" — Forced parallel agent dispatch instead of serial single-threaded investigation. Cut wall-time materially.

- "i think both bugs are true" + "wait so what was causing our split-slice issues, i thought that was a real bug?" — When the orphan-attempts theory landed, started conflating it with the split-slice bug and arguing the split was illusory. User pulled back to "both bugs are independently real, both need fixes". Saved from shipping only half the fix.

- "could we write it as a migration instead?" — Replaced a manual reconciliation script with a startup migration. Migrations run before any worker connects (`db.py:316 -> :519`), are idempotent, and don't require manual coordination during the restart window.

- "change it TO FIX SPLIT SLICE" — After arguing migration 0039 was zero-impact (current snapshot had 0 split jobs), user insisted it ship anyway as forward defense. Right call: migration is cheap, and any split between code-land and restart would have been left wedged.

- "direct provider doesn't coschedule, it's fine. aren't batched heartbeats deleted? i don't think they exist anymore. we don't have any gpus" — Pruned codex's three review concerns to the one real one (batched heartbeats — turned out they still exist, the bug was real). Saved time on cleanup of two non-issues.

- "try running the query against the active controller with uv run iris ... query" — When asserting "0 currently split jobs" based on the local `/tmp/iris-debug` snapshot, user demanded verification against the live DB. The snapshot pre-dated the incident timeframe; checking live data was the right move and confirmed the no-op behavior of 0039.

## Root cause

Two independent bugs that overlapped in observable symptoms:

**Bug 1 — `cancel_job` orphan task_attempts (`lib/iris/src/iris/cluster/controller/transitions.py:1083-1120`).** `cancel_job` calls `tasks.bulk_kill_non_terminal`, `workers.decommit_resources`, and `jobs.bulk_update_state` but never finalizes the in-flight `task_attempts` rows. Every cancel — whether user-initiated `iris job stop` or internal cascade — leaves the task's current attempt at `state=RUNNING, finished_at_ms=NULL` indefinitely. Class: state-invariant violation between `tasks` and `task_attempts` tables. Affected 30,382 attempt rows over the prior week. The dashboard's per-worker view reads `task_attempts.state` (not `tasks.state`), which surfaced the orphans as "task running on worker X" forever after the worker had been freed and re-leased.

**Bug 2 — coscheduled split-slice on transient task failure.** When one task of a coscheduled job hits a transient failure (TPU init failure, ASSIGNED→WORKER_FAILED, retriable WORKER_FAILED), `_resolve_task_failure_state` (`transitions.py:657-676`) downgrades it to PENDING. Pre-fix, the cascade gate at `transitions.py:1473` was `task_state in FAILURE_TASK_STATES` — but PENDING isn't in that set, so siblings stayed RUNNING. Next scheduling cycle, `_find_coscheduled_assignments` (`scheduler.py:668-744`) saw `num_tasks=1` (the lone PENDING task) and placed it on whichever worker group had ≥1 free slot — possibly a different `tpu-name`. The job's SPMD collective could not form across the split topology; runs hung. `_remove_failed_worker` (`transitions.py:1715`) had the same bug for host failures.

Bug 1 explained the worker-0 dashboard report. Bug 2 explained the actual job hangs (different jobs, different timeframe). They were tangled because the suspect worker had a real co-tenant from a separate concurrent issue (preemption interaction with #5240's same-variant slice preemption) plus the orphan UI artifact, both visible in the same dashboard query.

## Fix

Code (committed on `rjpower/20260428-fix-scheduling`):

- `lib/iris/src/iris/cluster/controller/transitions.py`:
  - New `_requeue_coscheduled_siblings` (parallel to `_terminate_coscheduled_siblings`). Bounces siblings to PENDING with `attempt_state=PREEMPTED`, leaves `preemption_count`/`failure_count` untouched, skips reservation holders.
  - Cascade gate replaced in three places (heartbeat `apply_state_updates`, direct-provider `apply_direct_provider_updates`, reaper `_remove_failed_worker`): if `update.new_state in FAILURE_TASK_STATES` and the job is coscheduled, branch on resolved `task_state` — terminal → existing `_terminate_coscheduled_siblings`, PENDING → new `_requeue_coscheduled_siblings`.
  - `cancel_job` now calls `attempts.bulk_finalize_active` after `tasks.bulk_kill_non_terminal`.
  - Drive-by: `apply_heartbeats_batch` at `transitions.py:1711` now returns the full `_apply_task_transitions` result instead of `TxResult(tasks_to_kill=...)` (was dropping `task_kill_workers`, breaking kill-RPC routing for every cascade through batched heartbeats).

- `lib/iris/src/iris/cluster/controller/stores.py`:
  - New `TaskAttemptStore.bulk_finalize_active` mirrors `TaskStore.bulk_kill_non_terminal`. Single UPDATE that marks active attempts under given job_ids terminal with COALESCE-protected `finished_at_ms` and `error`.

Migrations:

- `lib/iris/src/iris/cluster/controller/migrations/0038_finalize_orphan_attempts.py` — flips orphan attempts to PREEMPTED. Two orphan classes covered: task-terminal-but-attempt-active, and superseded-attempt (`attempt_id != tasks.current_attempt_id`). Idempotent.

- `lib/iris/src/iris/cluster/controller/migrations/0039_requeue_split_coscheduled_jobs.py` — finds coscheduled jobs spanning >1 `md_tpu_name`, decommits per-task resources from each worker (CPU millicores, mem bytes, TPU/GPU counts; GPU uses `COALESCE(NULLIF(json_extract($.gpu.count), 0), 1)` to mirror `types.get_gpu_count`'s `or 1` semantics), marks in-flight attempts PREEMPTED, resets tasks to PENDING with `current_worker_id` cleared. Logs each healed job + task to stdout for ops correlation. Idempotent.

Tests in `lib/iris/tests/cluster/controller/test_transitions.py` (sibling cascade behavior across all three failure paths, atomic re-coschedule via scheduler, cancel_job attempts finalize) and `lib/iris/tests/cluster/controller/test_db.py` (both migrations against synthetic split / orphan state).

Apply path: ship branch → `iris cluster controller checkpoint` → `iris cluster controller restart` (controller-only, seconds of downtime per `lib/iris/OPS.md`). Migrations run inside `apply_migrations()` at startup before any worker reconnects. Workers were stopped at original `cancel_job` time via `StopTasks` RPCs — migration only fixes DB rows, no kill RPCs issued. PollTasks reconciliation (`controller.py:1015`) was already keying off `tasks.state`, so the kill-side has been correct throughout.

## How OPS.md could have shortened this

- **Add to `lib/iris/OPS.md` "Sharp edges" under "SQL Queries":**

  > Dashboard worker-view reads `task_attempts.state`; scheduler / PollTasks / committed_* accounting reads `tasks.state`. A "two tasks on one worker" report from the slice view is consistent with stale attempt rows from a defunct termination path — cross-check `committed_tpu` on the worker. If it equals one task's worth and the workers' real heartbeat shows healthy, the extra attempts are ghosts, not co-tenants.

  Would have collapsed step 8's investigation into one query.

- **Add to `lib/iris/OPS.md` "Useful queries":**

  ```sql
  -- Coscheduled jobs whose active tasks span more than one TPU slice
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

  Detects any future split-slice incident in one shot. Generic — applies to any coscheduling failure, not just this bug.

- **Add to `lib/iris/OPS.md` "Sharp edges":**

  > Multi-task jobs default to `JOB_PREEMPTION_POLICY_PRESERVE_CHILDREN` (`transitions.py:_resolve_preemption_policy`). For coscheduled jobs that effectively means "any per-task failure can split the job across slices unless the failure path explicitly cascades". When debugging coscheduling oddities, audit every code path that downgrades a task to PENDING — check whether it cascades to siblings.

  Generic guidance for any future cascading-failure investigation in the scheduler.

- **Add to `lib/iris/OPS.md` "Sharp edges":**

  > Snapshot age matters when verifying scheduler hypotheses. The local `/tmp/iris-debug/controller.sqlite3` from a prior `iris cluster controller checkpoint` may pre-date the incident you're investigating. Run hypothesis queries against the live controller via `iris --config=... query` first; only fall back to the snapshot for slow / expensive scans.

  Saved time would have been step 14, where a stale snapshot produced misleading "0 splits" reads before the live query confirmed.

## Artifacts

- Branch: `rjpower/20260428-fix-scheduling`.
- Live cluster checkpoint snapshot used during investigation: `/tmp/iris-debug/controller.sqlite3` (date range 2026-04-09 → 2026-04-28 20:01 UTC; pre-dates the worker-0 incident; contains 30,382 orphan attempts).
- Codex review output: `/tmp/codex-reviews/split-slice-fix.md`.
- Original suspect worker (likely no longer present): `marin-tpu-v5p-preemptible-64-us-central1-20260428-1031-d11be78c-worker-0`.
- Recent commit that turned the latent split-slice bug into a routine occurrence: `e98304258 [iris] Same-variant slice preemption for coscheduled jobs (#5240)`.
