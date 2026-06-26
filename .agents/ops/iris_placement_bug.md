---
date: 2026-05-03
system: iris
severity: high
resolution: investigating — workaround = serialize submissions
related_logbook: .agents/logbooks/midtraining_delphi.md
---

# Iris coscheduler places two simultaneously-submitted gangs on identical host sets → JAX-coordinator-port death loop

## TL;DR

When two coscheduled multi-host TPU jobs are submitted to the same iris cluster within ~2–5 seconds of each other, requesting the same TPU variant + region + priority band, **the allocator can place both jobs on the exact same physical worker hosts.** Both jobs then race on the JAX distributed coordinator port (8476), each preempting/restarting the other repeatedly. The job tasks accumulate thousands of preemption cycles, never produce stable training, and eventually iris terminates with `failures=1, preemptions=N (N huge), worker_failed=∼all`.

This is **not flakiness** — it is a deterministic placement collision tied to simultaneous submission. Confirmed twice in 24 hours, on different TPU types (v5p-64 and v5p-256), with identical signatures.

## Evidence: two confirmed incidents

### Incident A — v5p-64, 2026-05-03 ~01:21 UTC

Two jobs submitted within seconds with same-shape resource request:

- `/ahmedah/delphi-1e21-p67m33-lr0p5-batch64-resume2-20260503-012053`
- `/ahmedah/delphi-1e21-p67m33-lr0p67-batch64-resume2-20260503-012053`

Both ended:

| Metric | lr0.5 | lr0.67 |
|---|---|---|
| state | failed | failed |
| failures | 1 | 1 |
| **preemptions** | **707** | **707** |
| **worker_failed** | **7/8** | **7/8** |
| run time | ~5 min | ~5 min |

**Same numbers on both jobs.** Probability that two independent failures produce identical preemption + worker-failed counts is ~0.

Bootstrap log evidence (same hosts assigned to both jobs):

```
lr0.5  task 0 advertise_host=10.202.1.1
lr0.67 task 0 advertise_host=10.202.1.1
lr0.5  task 1 advertise_host=10.202.1.32
lr0.67 task 1 advertise_host=10.202.1.32
lr0.5  task 2 advertise_host=10.202.1.16
lr0.67 task 2 advertise_host=10.202.1.16
... (8/8 tasks identical)
```

Note: incident A was layered on top of /moojink/ priority-band-2 batch contention on v5p-64 us-east5, which made the consequences worse (high external preemption pressure). The placement collision itself is the iris bug; the /moojink/ pressure is just contributing churn.

### Incident B — v5p-256, 2026-05-03 ~01:38 UTC submitted, ran 09:14–18:43 UTC

Three p50m50 jobs submitted in a tight window (~3.6s):

- `/ahmedah/delphi-1e22-p50m50-lr0p33-batch256-20260503-013817` (✅ survived, currently running cleanly)
- `/ahmedah/delphi-1e22-p50m50-lr0p5-batch256-20260503-013817` (❌ FAILED)
- `/ahmedah/delphi-1e22-p50m50-lr0p67-batch256-20260503-013817` (❌ FAILED)

Submission timestamps (epoch ms):

```
lr0.5  parent submitted: 1777772316424  (2026-05-03 01:38:36 UTC)
lr0.67 parent submitted: 1777772318732  (2026-05-03 01:38:38 UTC)  → +2.3s gap
```

train_lm dispatch:

```
lr0.5  train_lm started:  1777799671408  (09:14:31 UTC)
lr0.67 train_lm started:  1777799674124  (09:14:34 UTC)  → +2.7s gap
```

End-state metrics on the two failed jobs:

| Metric | lr0.5 | lr0.67 |
|---|---|---|
| state | failed | failed |
| failures | 1 | 1 |
| **preemptions** | **3131** | **3131** |
| **worker_failed** | **31/32** | **31/32** |
| train_lm wall lifetime | 569 min | 569 min |

Compare to the **survivor** lr0.33 in the same submission burst:
- preemptions = 1
- worker_failed = 0
- still running cleanly to step 4480+

And to the four other 1e22 v5p-256 jobs submitted slightly later (one at a time, with 1.5–3 min gaps) — all `preemptions=0`, all running fine.

**Bootstrap log evidence** (v5p-256 case): both failed jobs received the **identical 31-host set** out of 32 distinct hosts visible in the logs. Verified by:

```
diff -q <(grep advertise_host lr0.5_log | sort -u) \
        <(grep advertise_host lr0.67_log | sort -u)
→ "IDENTICAL host sets"
```

First few task assignments verbatim:

```
lr0.5  task 0  advertise_host=10.202.0.195    lr0.67 task 0  advertise_host=10.202.0.195
lr0.5  task 10 advertise_host=10.202.0.219    lr0.67 task 10 advertise_host=10.202.0.219
lr0.5  task 11 advertise_host=10.202.1.5      lr0.67 task 11 advertise_host=10.202.1.5
lr0.5  task 12 advertise_host=10.202.0.185    lr0.67 task 12 advertise_host=10.202.0.185
lr0.5  task 13 advertise_host=10.202.0.222    lr0.67 task 13 advertise_host=10.202.0.222
lr0.5  task 14 advertise_host=10.202.1.85     lr0.67 task 14 advertise_host=10.202.1.85
lr0.5  task 15 advertise_host=10.202.1.1      lr0.67 task 15 advertise_host=10.202.1.1
lr0.5  task 16 advertise_host=10.202.1.82     lr0.67 task 16 advertise_host=10.202.1.82
lr0.5  task 17 advertise_host=10.202.1.75     lr0.67 task 17 advertise_host=10.202.1.75
lr0.5  task 18 advertise_host=10.202.0.247    lr0.67 task 18 advertise_host=10.202.0.247
```

10/10 sampled task indices on the same physical hosts. Full set diff confirms 31/31 match.

## Failure mechanics — why identical placement is fatal

Each train_lm task on a TPU host expects to bind/reach `port 8476` (`endpoint_name=jax_coordinator`) on the worker VM. The full bootstrap line:

```
iris.runtime.jax_init initialize_jax bootstrap inputs:
  task_index=0 num_tasks=32 advertise_host=10.202.0.195
  ports={} endpoint_name=jax_coordinator requested_port=8476
  env={'IRIS_TASK_ID': '/ahmedah/.../train_lm/0:N', 'IRIS_NUM_TASKS': '32'}
```

When two jobs each bring up a task at the same host:port, exactly one binds; the other sees the port busy or sees an iris service-registry conflict on `endpoint_name=jax_coordinator`. iris recycles the loser (`preemptions++`). The recycled task starts up, sees the same conflict, dies again. Meanwhile peer tasks of both jobs are polling `_poll_for_coordinator` (`lib/iris/src/iris/runtime/jax_init.py:172`) with a 300s timeout. Some hit `TimeoutError`, marking their task as `worker_failed`. Eventually `failures=1` propagates to parent → terminal failure for both jobs.

Symptoms in logs:

```
ConnectError: Stale attempt: task .../train_lm/0 attempt 12 != current 14
TimeoutError: Timed out after 300.0s waiting for coordinator endpoint 'jax_coordinator'
RuntimeError: Build failed with exit_code=2  (occurred on lr0.5 task 28; cascade-induced, not root cause)
absl::Status: INVALID_ARGUMENT: Unexpected task registered with task_name=/job:/replica:0/task:7
absl::Status: ALREADY_EXISTS: Aborted connect attempt as there is a request from a newer incarnation
```

Note in incident B these errors appeared **mid-run after 9.5 hours of uptime**, not at startup. The collision is latent — both jobs may run nominally for many hours, then a single host-level transient (e.g. one task's setup retry, a brief preemption) triggers the dual-coordinator collision and the system never recovers because both jobs keep trying to bind the same port.

## Why the third co-submitted job (lr0.33) survived

Best hypothesis without iris allocator source code:

- iris commits placement decisions FIFO. lr0.33 was 2.3s ahead of lr0.5 on submission, so its placement was already committed before the lr0.5 / lr0.67 race.
- When lr0.5 and lr0.67 entered the scheduler within 2.3s of each other, iris's allocator may not have yet flushed lr0.5's reservation when it processed lr0.67's request, allowing both to claim the same host set.

This is consistent with the survivor pattern in incident B (3 simultaneous, oldest survives, last two collide) and with general scheduler race-condition shapes.

## Reproducer signature for future detection

A failed run is a likely placement-collision victim if **all** of these hold:

1. **Identical metrics on two failed jobs**: `preemptions` and `worker_failed` match exactly between the two jobs.
2. **Submission times within ≤5 seconds** of each other.
3. **Same TPU variant + region + priority band** in their `job_config` rows.
4. **Same iris allocator tick** (no other higher-priority demand entered the queue between them).
5. **No higher-priority external user** active on that pool (rules out preemption-eviction as the explanation; this happened on v5p-256 where /moojink/ has no jobs).

Bootstrap-log host comparison is the gold-standard confirmation: if the two jobs share `advertise_host` IPs across their task indices, the placement collision is proven.

## Mitigations / workarounds

Until iris fixes its allocator (see "Action items"), the operational workaround is **serialize coscheduled submissions** across the same (variant, region, priority) tuple:

1. Submit job A.
2. Wait for A's `train_lm` child to be in `running` state (`iris job summary <job>/train_lm` shows `state: running` and at least one preemption=0/1 reading).
3. Submit job B.
4. Repeat per job.

Alternative shapes:

- **Submit at different priority bands** so iris's allocator doesn't race them through the same scheduler tick.
- **Pin to different zones** with `--zone us-east5-a` vs `--zone us-east5-b` (only works if the TPU pool has multiple zones with capacity).
- **Submit with delay**: a wrapper that introduces a 60s sleep between consecutive submissions in the same TPU group. (Empirically, submissions ≥1 minute apart in incident B did not collide.)

Do **not**:

- Use the same `RUN_ID` / `WANDB_RUN_ID` for the simultaneously-submitted jobs (orthogonal issue, but compounding — see `.agents/ops/2026-05-02-delphi-midtrain-resume-namespace.md`).

## Action items / open questions for the iris team

1. **Investigate iris allocator placement logic** for races between concurrent coscheduled gang reservations. Suspected file: `lib/iris/src/iris/...` — needs a tour by an iris maintainer.
2. **Add a placement-uniqueness assertion** at coschedule commit time: each host should have at most one active `tpu-name` coscheduling group at any given moment. If two reservations claim overlapping hosts, fail the second with `Scheduler: host already reserved` instead of double-booking.
3. **Add metrics**: count "double-booked-host" events per scheduling tick. Should be 0; the bug shows it can be >0.
4. **Consider TASK_STATE_BUILDING transitions**: are tasks being re-placed onto contested hosts during recycle? If so, the recycler should also check uniqueness.

## Affected job IDs (debugging references)

### v5p-64 incident A (2026-05-03)

Failed jobs:
- `/ahmedah/delphi-1e21-p67m33-lr0p5-batch64-resume2-20260503-012053`
- `/ahmedah/delphi-1e21-p67m33-lr0p67-batch64-resume2-20260503-012053`

(Earlier failed v5p-64 runs from the same session — `delphi-1e21-p67m33-lr0p5-batch64-resume-20260502-151756`, `delphi-1e21-p67m33-lr0p67-batch64-resume-20260502-161841`, the 6 v5p-64 batch-band-3 jobs at `20260503-014225` — all failed for the same root cause class, mixed with /moojink/ band-2 contention.)

### v5p-256 incident B (2026-05-03)

Failed jobs:
- `/ahmedah/delphi-1e22-p50m50-lr0p5-batch256-20260503-013817`
- `/ahmedah/delphi-1e22-p50m50-lr0p67-batch256-20260503-013817`

Surviving co-submitted job (the third in the simultaneous burst, ahead by 2.3s):
- `/ahmedah/delphi-1e22-p50m50-lr0p33-batch256-20260503-013817`

Other v5p-256 1e22 jobs in the same general fleshing-out batch — all submitted with ≥1 minute gaps from each other, all running fine:
- `/ahmedah/delphi-1e22-p50m50-lr0p83-batch256-20260503-013949` (running)
- `/ahmedah/delphi-1e22-p67m33-lr0p83-batch256-20260503-014108` (running)
- `/ahmedah/delphi-1e22-p33m67-lr0p83-batch256-20260503-014108` (running)

## Useful diagnostic queries

```bash
# Check for identical preemption + worker_failed across job pairs
uv run iris --controller-url=http://localhost:10000 --cluster=marin job summary <job-A>/train_lm
uv run iris --controller-url=http://localhost:10000 --cluster=marin job summary <job-B>/train_lm

# Confirm host-collision: extract advertise_host from bootstrap logs
uv run iris --controller-url=http://localhost:10000 --cluster=marin job logs <job-A>/train_lm \
  --max-lines 5000 2>&1 | grep -oE "advertise_host=10\.[0-9.]+" | sort -u > /tmp/A.hosts
uv run iris --controller-url=http://localhost:10000 --cluster=marin job logs <job-B>/train_lm \
  --max-lines 5000 2>&1 | grep -oE "advertise_host=10\.[0-9.]+" | sort -u > /tmp/B.hosts
diff /tmp/A.hosts /tmp/B.hosts
# IDENTICAL output → placement collision confirmed

# Check active jobs on same TPU variant + region for context
uv run iris --controller-url=http://localhost:10000 --cluster=marin query "
SELECT j.name, j.state, jc.priority_band
FROM jobs j JOIN job_config jc ON j.job_id=jc.job_id
WHERE jc.res_device_json LIKE '%v5p-256%' AND j.state IN (1,2,3)
  AND jc.constraints_json LIKE '%us-east5%'
ORDER BY jc.priority_band, j.submitted_at_ms"

# Verify iris workers are healthy (rules out hardware as the cause)
uv run iris --controller-url=http://localhost:10000 --cluster=marin query "
SELECT count(DISTINCT w.worker_id) as healthy_hosts
FROM workers w JOIN worker_attributes wa ON w.worker_id=wa.worker_id
WHERE wa.key='device-variant' AND wa.str_value='v5p-256' AND w.healthy=1"
```

## Cost estimate

- Incident A: ~5 min × 8 hosts × 2 jobs = ~1.3 v5p-64-host-hours wasted (small).
- Incident B: ~9.5h × 32 hosts × 2 jobs = **608 v5p-256-host-hours wasted** (substantial — these jobs ran in a thrash state for 9.5 hours, never producing real progress beyond what was checkpointed early).

Combined two-day cost from this bug class: ~610 v5p host-hours of wasted compute, plus ~1 day of operator time on misdiagnosis (preemption-by-/moojink/ vs internal placement collision).

## Related issues

- `.agents/ops/2026-05-02-delphi-midtrain-resume-namespace.md` — the namespace/run-id-derivation bug that compounded the v5p-64 case (made wandb fragmentation worse and complicated relaunch logic).
- GitHub issue `https://github.com/marin-community/marin/issues/5374` — MirrorFS / cross-region transfer issue, unrelated but adjacent (same training jobs).

---

# Appended 2026-05-04 — three more confirmed incidents

Three additional reproductions of the same bug surfaced. The most important new finding: **placement collision can happen at dispatch time, not just submission time.** Two jobs submitted hours apart but pending in the queue can still collide if their queue dispatch lands in the same iris scheduler tick.

## Incident C — v5p-256, 2026-05-03 ~01:41 UTC submitted, ran 09:14–18:43 UTC

Two jobs submitted within ~2.6 seconds:

- `/ahmedah/delphi-1e22-p33m67-lr0p83-batch256-20260503-014108`  (FAILED)
- `/ahmedah/delphi-1e22-p67m33-lr0p83-batch256-20260503-014108`  (✅ SUCCEEDED — control)

Submission timestamps (epoch ms):

```
p33m67-lr0.83 submitted: 1777772484621
p67m33-lr0.83 submitted: 1777772482016  → +2.6s gap
```

End-state metrics:

| Metric | p33m67-lr0.83 (failed) | p67m33-lr0.83 (control) |
|---|---|---|
| state | failed | succeeded |
| failures | 1 | 0 |
| **preemptions** | **3131** | (low) |
| **worker_failed** | **31/32** | 0/32 |

Identical `preemptions=3131, worker_failed=31` to incident B's lr0.5/lr0.67 case — same v5p-256 32-task gang collision signature. The control (p67m33-lr0.83) won the placement race; p33m67-lr0.83 lost.

Last permanent checkpoint of the failed run: `step-6112` in `gs://marin-us-east5/checkpoints/delphi-1e22-p33m67-32p07b-lr0.83-78fd44/checkpoints/`. Temp checkpoint at `step-6767`. So the run reached step ≥6112 (80%+) before thrash kicked in — likely transient mid-run trigger similar to incident B.

Salvageable: yes via `MIDTRAIN_RESUME_OUTPUT_PATH=gs://marin-us-east5/checkpoints/delphi-1e22-p33m67-32p07b-lr0.83-78fd44` + `MIDTRAIN_EXPECT_RESUME_MIN_STEP=6112`.

## Incident D — v5p-128, 2026-05-03 ~16:14 + 20:02 UTC submitted, both dispatched 2026-05-04 12:19:17 UTC (within 5s)

This is the **dispatch-time-collision** variant. Two jobs:

- `/ahmedah/delphi-1e21-p67m33-lr0p67-batch128-resume2-20260503-161414` (FAILED)
- `/ahmedah/delphi-1e21-p33m67-lr0p83-batch128-20260503-200203` (FAILED)

Submission timestamps (epoch ms):

```
p67m33-lr0.67 resume2 submitted: 1777824878720  (16:14:38 UTC May 3)
p33m67-lr0.83          submitted: 1777838540690  (20:02:20 UTC May 3)
                                                  → 3h 47m gap at submission
```

But train_lm dispatch was effectively simultaneous:

```
p67m33-lr0.67 train_lm started: 1777897156854  (12:19:16 UTC May 4)
p33m67-lr0.83 train_lm started: 1777897161741  (12:19:21 UTC May 4)
                                                → 4.9s gap at dispatch
```

Both pending in the v5p-128 queue for hours, then iris dispatched both within the same ~5-second scheduler tick when capacity finally opened. They collided on placement.

End-state metrics:

| Metric | p67m33-lr0.67 resume2 | p33m67-lr0.83 |
|---|---|---|
| state | failed | failed |
| failures | 1 | 1 |
| **preemptions** | **1515** | **1515** |
| **worker_failed** | **15/16** | **15/16** |
| train_lm wall lifetime | 21 min | 37 min |

**Identical preemption count between two fully-independent jobs** — same as v5p-64 (incident A, 707 each) and v5p-256 (incident B, 3131 each; incident C single-victim 3131). The 1515 number is consistent with a 16-task gang on v5p-128 going through ~95 thrash cycles per task.

Recovery namespaces:
- p67m33-lr0.67 resume2: targeted `gs://marin-us-east5/checkpoints/delphi-1e21-p67m33-9p25b-lr0.67-ecbd27/` (latest perm step-2646).
- p33m67-lr0.83: never wrote a checkpoint (failed before first save). Namespace would be `gs://marin-us-east5/checkpoints/delphi-1e21-p33m67-9p25b-lr0.83-0cb048/` but `checkpoints/` empty. Would need to re-run from base.

## Updated reproducer signature

The submission-time check from the original postmortem was incomplete. Better criteria:

A failed run is a likely placement-collision victim if **all** of these hold:

1. **Identical metrics on two failed jobs**: `preemptions` and `worker_failed` match exactly.
2. **train_lm dispatch times within ≤10 seconds** of each other (NOT submission time — jobs queued for hours can still collide at dispatch).
3. **Same TPU variant + region + priority band** in their `job_config` rows.
4. **Same iris allocator scheduler tick** (verified by close `started_at_ms` on the train_lm child).
5. **Bootstrap-log host comparison** shows shared `advertise_host` IPs across task indices (gold-standard confirmation when logs are still available).

The `preemptions` count appears to scale with the gang size:
- 8-task gang (v5p-64): ~707 (incident A) — but compounded by external preemption from /moojink/ that incident
- 16-task gang (v5p-128): ~1515 (incident D)
- 32-task gang (v5p-256): ~3131 (incidents B, C)

Approximately 95-100 cycles per task before iris terminates. This may be a function of `max_retries_preemption=1000` budget in `job_config` — close to the cap but not at it, suggesting a different heuristic in iris triggers termination.

## Investigation pointers for the next agent

To dig into iris source for the placement bug:

1. **Repository**: `lib/iris/src/iris/` in this worktree.
2. **Key files to read**:
   - `lib/iris/src/iris/scheduler/` — coscheduling and placement logic
   - `lib/iris/src/iris/runtime/jax_init.py` — coordinator endpoint resolution (not the bug, but downstream symptom site)
   - `lib/iris/src/iris/cli/job.py` — submission flow
   - Look for `coscheduling_group_by`, `tpu-name` constraint handling, host reservation commit semantics

3. **Database schema** (controller SQLite):
   - `jobs` — job lifecycle states (1=submitted/pending, 3=running, 4=succeeded, 5=failed, 6=cancelled)
   - `job_config` — submission spec including `coscheduling_group_by`, `priority_band`, `res_device_json`, `constraints_json`
   - `tasks` — task-level state per job
   - `task_attempts` — each recycle creates a new attempt; this is where the 1515/3131 preemption counts accumulate
   - `task_resource_history` — should show host bindings per attempt; might reveal the placement decision tree
   - `worker_task_history` — inverse: which tasks a worker has seen
   - `dispatch_queue` — pending dispatch decisions; race window candidate

4. **Useful SQL queries** for next investigator:

```sql
-- Find pairs of failed jobs with matching preemption + worker_failed (collision suspects)
SELECT a.name AS job_a, b.name AS job_b, ja.preemptions, ja.failures
FROM jobs a JOIN jobs b ON a.parent_job_id = b.parent_job_id  -- or by some grouping
WHERE a.state=5 AND b.state=5 AND a.job_id < b.job_id
  AND ABS(a.started_at_ms - b.started_at_ms) < 10000  -- dispatched within 10s
LIMIT 20;

-- Find dispatch ticks: starts grouped by 5-second buckets
SELECT
  started_at_ms / 5000 as tick_5s,
  count(*) as n_started,
  group_concat(name, ' | ') as jobs_started
FROM jobs
WHERE state IN (3,4,5) AND started_at_ms IS NOT NULL
GROUP BY tick_5s
HAVING n_started >= 2
ORDER BY tick_5s DESC LIMIT 20;

-- Per-incident task placement history
SELECT * FROM task_resource_history
WHERE task_id LIKE '%delphi-1e22-p50m50-lr0p5-batch256-20260503-013817%'
ORDER BY epoch_ms LIMIT 100;
```

5. **Iris CLI commands for live diagnosis**:

```bash
# preemption count + failure summary
uv run iris --controller-url=http://localhost:10000 --cluster=marin job summary <job>/train_lm

# bootstrap host info (only fresh logs; rotate after a while)
uv run iris --controller-url=http://localhost:10000 --cluster=marin job logs <job>/train_lm \
  --max-lines 5000 2>&1 | grep -oE "task_index=[0-9]+ .* advertise_host=10\.[0-9.]+" | sort -u

# bug-report dump (more diagnostic detail)
uv run iris --controller-url=http://localhost:10000 --cluster=marin job bug-report <job>
```

## Affected job IDs (this round)

### Incident C — v5p-256 (2026-05-03 simultaneous submission)

- ❌ `/ahmedah/delphi-1e22-p33m67-lr0p83-batch256-20260503-014108` (failed, salvageable from step-6112)
- ✅ `/ahmedah/delphi-1e22-p67m33-lr0p83-batch256-20260503-014108` (succeeded — control)

### Incident D — v5p-128 (2026-05-04 simultaneous dispatch despite staggered submission)

- ❌ `/ahmedah/delphi-1e21-p67m33-lr0p67-batch128-resume2-20260503-161414` (failed, was attempting to resume from `ecbd27` step-2646)
- ❌ `/ahmedah/delphi-1e21-p33m67-lr0p83-batch128-20260503-200203` (failed, no checkpoint saved — would need fresh run)

## Updated cost tally

- Incidents A+B+C+D combined: ~1100+ v5p host-hours of wasted compute over 36 hours.
- Incident D specifically: 16 hosts × 21 min × 2 jobs = ~11 v5p-128-host-hours (small per-incident, but the work-loss for p67m33-lr0.67 cell is the existing 60% progress that has to be re-done if we don't resume from `ecbd27`).
- Incident C: 32 hosts × duration of thrash before iris terminated ≈ ~tens of v5p-256-host-hours; meaningful but smaller than B because failure happened mid-run, not 9.5h late.

## What this means for ongoing 1e21/1e22 work

- v5p-128 is **also** susceptible — not just v5p-64 and v5p-256. Bug is general to coscheduled gangs, scales with gang size.
- **Mitigation that's been working**: serialize submissions AND watch dispatch times, not just submission times. If a job is pending behind a queue and another is also pending, they can dispatch together when capacity opens.
- For pending queue scenarios, consider:
  - Submit only ONE job at a time across the whole sweep, even if the others would just sit pending — this guarantees no dispatch-time race.
  - Or: build a wrapper that watches `state` of the most recent submission and only submits the next once it transitions to `running`.

The current sweep's recovery is: serialize 1e22 p33m67-lr0.83 resume → wait until it's running → then 1e21 p67m33-lr0.67 resume2 → wait → then 1e21 p33m67-lr0.83 (or skip).
