---
date: 2026-04-21
system: iris
severity: outage
resolution: fixed
pr: none
issue: none
---

# Iris scheduler thread freeze — "Pending scheduler feedback"

## TL;DR

- Dashboard showed **"Pending scheduler feedback"** on every pending job in the Marin cluster; a user-reported job had been pending 30+ min with no diagnostic text.
- The scheduling thread had crashed ~21 minutes before the reported job was submitted, with `sqlite3.IntegrityError: UNIQUE constraint failed: task_attempts.task_id, task_attempts.attempt_id` in `_dispatch_assignments_direct → queue_assignments → _assign_task → insert_task_attempt`.
- `ManagedThread._safe_target` logged the crash and returned — no respawn — so `_scheduling_diagnostics` cache stopped updating and every pending job fell back to the literal `"Pending scheduler feedback"` in `service.py:1224`.
- Real root cause: `fail_worker`'s reservation-holder branch at `transitions.py:2170-2180` reset `tasks.current_attempt_id = -1` while leaving old `task_attempts` rows (ids 0..N-1) in place. The next assignment computed `attempt_id = -1 + 1 = 0` and collided.
- Fix: route reservation holders through `_terminate_task` like non-holders, passing `preemption_count=0` to preserve the holder-specific retry-budget semantic. Companion migration `0036_reconcile_reservation_holder_attempt_ids.py` heals any stale row on restart. Regression test added.

## Original problem report

> help me debug why this job on lib/iris/examples/marin.yaml is showing "awaiting scheduler feedback". i think we might have a bug in the controller?
>
> https://iris.oa.dev/#/job/%2Fmichaelryan%2Fcuration-fm-157m-dclm-1p8e19-canonical

Job `/michaelryan/curation-fm-157m-dclm-1p8e19-canonical` — 2 CPU / 2 GiB, interactive band, 1 task — submitted 2026-04-22T00:53:34 UTC, pending 90+ minutes at queue_position 271 with pending_reason `"Pending scheduler feedback"`. CPU VM pool (`cpu_vm_e2_highmem_2_ondemand`) had 5 READY idle slices, so capacity was not the issue.

## Investigation path

1. Pulled `job bug-report` and `rpc controller get-scheduler-state`. Job was in `PRIORITY_BAND_INTERACTIVE` pending queue at position 271 with resource_value 12 — nothing exotic. Budget row showed `limit=0, spent=318036` for `michaelryan`; **user explicitly ruled this out** ("budgets aren't the problem").
2. Traced the fallback string `"Pending scheduler feedback"` to `service.py:1224` — set when `controller.get_job_scheduling_diagnostics()` returns None. Diagnostics only populated by `_cache_scheduling_diagnostics` inside `_run_scheduling` (controller.py:2007).
3. Dispatched a senior-engineer agent in the background to enumerate code paths that could suppress diagnostics for a long-pending task. Top hypotheses: `task_id.parent is None` (D), `reservation_unsatisfied` gate (A.3), `job_not_found` (A.2).
4. Downloaded the latest controller checkpoint from `gs://marin-us-central2/iris/marin/state/controller-state/` via `gcloud storage cp` + `zstd -df`. The checkpoint was from 00:41:28 — 12 minutes *before* the job was submitted, so `/michaelryan/...` wasn't in it. Hypotheses D, A.3, A.2 all refuted by live `iris query`: task had `/0` suffix (parent exists), `has_reservation=0`, `job_config` row present.
5. **User:** "it looks like the scheduling loop crashed, can you look for an exception in the parquet?" — pivoted to searching recent parquet log shards.
6. Downloaded 9 recent parquet log shards to `/tmp/iris-debug/` and queried with duckdb. **In hindsight this was unnecessary** — `pyarrow.dataset` + `gcsfs` + `duckdb.register` lets duckdb query `gs://...` directly with predicate pushdown; 15-shard searches complete in ~7s with no local storage. See the reference recipe in the OPS.md suggestions below. Found one hit on pattern `Traceback` in the controller process log: `scheduling-loop crashed` at `2026-04-22 00:32:23.242 UTC`, with full stack leading to `sqlite3.IntegrityError: UNIQUE constraint failed: task_attempts.task_id, task_attempts.attempt_id`.
7. Confirmed zero `Scheduling cycle` / `ManagedThread` / `_run_scheduling` log lines after the crash — thread never respawned.
8. Queried the checkpoint for tasks where `max(task_attempts.attempt_id) > tasks.current_attempt_id`. One result: `/larry/iris-run-job-20260421-153324/:reservation:/0` — 68 attempt rows (ids 0..67, all WORKER_FAILED), `current_attempt_id = -1`, state PENDING, `is_reservation_holder = 1`.
9. Read `fail_worker` at `transitions.py:2140-2210`. The reservation-holder branch at 2170-2180 was the only place in the codebase that set `current_attempt_id = -1`. It DELETEd only the current attempt row (`attempt_id = current_attempt_id`), leaving any accumulated earlier attempts behind.
10. Dispatched a second senior-engineer agent to review the proposed fix (replace branch with `_terminate_task`, pass `preemption_count=0`). Verdict: APPROVE WITH CHANGES — required a regression test and flagged that the change drops the `error = NULL` semantic.
11. **User:** "let's do the following" — write a migration, answer whether the try/except in `_run_scheduling_loop` would infinite-loop, send a third agent to add a scheduling regression test.
12. Dispatched third senior-engineer agent to apply the fix and add a multi-cycle regression test. Agent also updated a pre-existing test (`test_holder_task_worker_death_no_failure_record`) whose `len(attempts) == 0` assertion had been pinning the buggy behavior.

## User course corrections

- **"budgets aren't the problem"** — cut off a wrong branch early. User knew the cluster's budgets have always been zero-limit and they're advisory, not enforcing.
- **"don't take a checkpoint, copy the previous from GCS"** — rejected the default `iris cluster controller checkpoint` workflow (would have stalled the controller while the scheduling loop was already dead) in favor of pulling an existing checkpoint from GCS. Faster, zero cluster impact.
- **"gcloud storage cp"** — overrode reflex `gsutil` (deprecated tooling in this org).
- **"it looks like the scheduling loop crashed"** — pivoted the investigation. The model was still drilling into gate-by-gate reasons a diagnostic might be missing; the user's thread-level hypothesis was correct and shortened the path by ~30 minutes.
- **"is there a simple fix? e.g. never preempt reservation jobs"** — proposed a constraint that turned out to be unrelated to the bug (crash was in `fail_worker`, not `preempt_task`). Clarifying why it wouldn't help led to the actual one-line fix.
- **"reservation jobs shouldn't have any children"** — let us skip the `_cascade_children` / `TERMINATE_CHILDREN` concern the code-review agent flagged. Scoped the test correctly.

## Root cause

`lib/iris/src/iris/cluster/controller/transitions.py:2170-2180` — the `_remove_failed_worker` reservation-holder branch. It reset `tasks.current_attempt_id = -1` while DELETing only the single current-attempt row, leaving attempts `0..N-1` in the table. On the next scheduler cycle, `queue_assignments` at `transitions.py:1667` computed `attempt_id = current_attempt_id + 1 = 0` and `insert_task_attempt` raised `sqlite3.IntegrityError` against the still-present `(task_id, 0)` row.

The exception unwound out of `_run_scheduling_loop`, through `ManagedThread._safe_target`, which caught-and-logged-and-returned. No supervisor respawned the thread. For the following ~90 minutes, no scheduling cycle ran, no diagnostics were cached, no new assignments were made. Workers continued heartbeating (separate thread) so the cluster *looked* alive.

Class of bug: **cross-table invariant violation** between `tasks.current_attempt_id` and `task_attempts`. The holder-specific reset logic fell out of sync with the non-holder path that the rest of the codebase assumed.

## Fix

**Code** — `lib/iris/src/iris/cluster/controller/transitions.py:2167-2186`. Replaced the holder-specific DELETE+reset branch with a unified `_terminate_task` call; only `preemption_count` differs between holder (0) and non-holder (computed from retry budget). The holder attempt row is now preserved in `WORKER_FAILED` state like any other terminal attempt.

**Migration** — `lib/iris/src/iris/cluster/controller/migrations/0036_reconcile_reservation_holder_attempt_ids.py`. On next controller restart, advances `tasks.current_attempt_id` to `max(task_attempts.attempt_id)` for any task where the invariant was already broken. Dry-run on the live checkpoint healed the single poisoned row (`/larry/iris-run-job-20260421-153324/:reservation:/0`: `-1 → 67`).

**Regression test** — `lib/iris/tests/cluster/controller/test_reservation_holder_reset_regression.py::test_reservation_holder_reassignment_across_successive_worker_failures`. Fails a holder + non-holder across three successive workers, asserts `current_attempt_id` advances `0→1→2`, three `WORKER_FAILED` attempt rows accumulate, no IntegrityError raised, non-holder re-schedules each cycle.

**Contract change called out** — `test_holder_task_worker_death_no_failure_record` in `test_reservation.py` previously asserted `len(holder_task.attempts) == 0` on each worker-death cycle, pinning the buggy behavior. Updated to assert attempts accumulate as `WORKER_FAILED` rows while `preemption_count`/`failure_count` stay zero. Worth checking if any dashboard component relied on holders having zero attempt rows.

**Defense-in-depth not applied:** wrapping `outcome = self._run_scheduling()` at `controller.py:1372` in try/except was discussed. With the poisoned row, a bare try/except would hot-loop at the exponential backoff ceiling indefinitely. Correct form would be per-assignment `SAVEPOINT` inside `queue_assignments` so one bad task doesn't take down the cycle. Left as follow-up.

## How OPS.md could have shortened this

Generic patterns — suggestions apply beyond this incident.

1. **`lib/iris/OPS.md` → new section "Parquet log queries".** The investigation's biggest speedup came from duckdb over the controller's parquet log shards in GCS. Currently undocumented. The parquet schema (`seq, key, source, data, epoch_ms, level`) with the `key='/system/controller'` filter for controller-only lines should be named, and the runnable recipe below should be the canonical entrypoint. Querying directly against GCS (no local download) is ~10× faster and avoids the temptation to grab only a few shards:

   ```python
   import duckdb, gcsfs, pyarrow.dataset as ds

   fs = gcsfs.GCSFileSystem(project='hai-gcp-models')
   files = sorted(fs.glob('gs://marin-us-central2/iris/marin/state/logs/logs_*.parquet'))[-20:]
   dataset = ds.dataset(files, format='parquet', filesystem=fs)
   con = duckdb.connect()
   con.register('logs', dataset)

   con.execute("""
       SELECT epoch_ms, substr(data, 1, 200)
       FROM logs
       WHERE key='/system/controller'
         AND data LIKE '%Traceback%'
       ORDER BY epoch_ms DESC LIMIT 50
   """).fetchall()
   ```

   Auth is ADC (`gcloud auth application-default login`). DuckDB's native `gs://` httpfs extension requires HMAC keys and rejects ADC — the pyarrow-dataset + `con.register` path is the one that works. Document this explicitly; otherwise the next engineer (human or model) will burn 10 minutes on the `CREATE SECRET` dead end.

2. **`lib/iris/OPS.md` → "Offline checkpoint analysis": prefer copying an existing GCS checkpoint over triggering a new one.** Current guidance says "trigger a checkpoint, then query offline." That's fine when the controller is healthy but wrong when it's stuck — `iris cluster controller checkpoint` briefly stalls the controller, and checkpoints already land in GCS every ~hour. Add:

   ```bash
   # List existing checkpoints (timestamps are epoch-ms)
   gcloud storage ls gs://<bucket>/iris/<cluster>/state/controller-state/

   # Copy the most recent; decompress
   gcloud storage cp gs://<bucket>/.../<ts>/controller.sqlite3.zst /tmp/
   zstd -df /tmp/controller.sqlite3.zst
   ```

   Note `gcloud storage cp`, not `gsutil` (deprecated at this org).

3. **`lib/iris/OPS.md` → "Known Bugs" (or a new "Silent-thread failure modes" section): background threads managed by `ManagedThread` can die silently.** `ManagedThread._safe_target` catches exceptions, logs `<thread-name> crashed`, and returns — no supervisor respawns. The cluster looks alive (heartbeats continue; other threads run) but critical work silently halts. Generic diagnostic: when a subsystem appears frozen but the controller is reachable, check recent parquet logs for `ManagedThread.*crashed` before assuming the code path is just slow.

4. **`lib/iris/OPS.md` → "Troubleshooting" table: add a row for "same pending-reason text on many/all pending jobs."** This is a generic smell for cache-update failure in the controller — whatever populates the per-job diagnostic cache has stopped running. The specific string varies by code path, but the pattern (uniform fallback text across unrelated jobs) is diagnostic.

## Artifacts

- Checkpoint (decompressed): `gs://marin-us-central2/iris/marin/state/controller-state/1776818488274/controller.sqlite3.zst` — taken 2026-04-22T00:41:28 UTC, ~18 min after crash. Contains the poisoned row.
- Parquet log shards covering crash window: `gs://marin-us-central2/iris/marin/state/logs/logs_00000000029{66141366..91964120}.parquet`. Crash line: `seq=2984329155, epoch_ms=1776817943242`. Query directly via duckdb + gcsfs (see "Parquet log queries" above); no need to `gcloud storage cp` locally.
- Fix PR: (not yet filed)
- Migration: `lib/iris/src/iris/cluster/controller/migrations/0036_reconcile_reservation_holder_attempt_ids.py`
- Fix: `lib/iris/src/iris/cluster/controller/transitions.py:2167-2186`
- Test: `lib/iris/tests/cluster/controller/test_reservation_holder_reset_regression.py::test_reservation_holder_reassignment_across_successive_worker_failures`
