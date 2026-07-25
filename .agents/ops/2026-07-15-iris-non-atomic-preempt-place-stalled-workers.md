---
date: 2026-07-15
system: iris
severity: degradation
resolution: fixed
pr: 7206
issue: weaver #439
---

# Iris scheduler: non-atomic preempt/place strands "free" workers and thrashes gangs

## TL;DR

- **Symptom (reported):** a worker on the `marin` cluster appears healthy with no
  running job — just a recently *preempted* one — while a higher-priority job sits
  pending. It looks like a resource-measurement glitch: a "free" worker the
  scheduler won't fill.
- **Root cause is not a measurement leak — it is a structural non-atomicity in the
  scheduler.** Placement and preemption are split across ticks by construction:
  `find_assignments` runs **first**, then the preemption pass runs **only over the
  tasks placement could not place** (`scheduling/backend.py:383-384`,
  `scheduling/decision.py:34-46`). So the preemptor is *never* placed in the same
  tick it triggers a preemption. It is placed in some *later* tick, after the
  victim's worker asynchronously stops and releases capacity. **Nothing reserves the
  freed worker for the preemptor** in the interim.
- Committed capacity frees only when the worker confirms the stop: `preempt_one`
  leaves the victim attempt unfinished (`finished_at_ms = NULL`,
  `reconcile/task.py:~275`), and committed capacity is computed from *unfinished
  worker-bound attempts* (`reconcile/reads.py:714-717`). A preemption-only tick does
  not even force a prompt reconcile (`controller.py:951-956` only sets
  `_force_reconcile` when there are assignments). So the freed-but-not-reclaimed
  window is seconds to minutes.
- Three consequences, all observed live:
  1. **Stranded "free" worker** during the preempt→free→place gap (the reported view).
  2. **Over-preemption across ticks** — the running-victim set is RUNNING-only
     (`reads.py:916`) while a just-preempted victim's chips stay committed, so an
     unplaced preemptor keeps selecting *fresh* victims on *other* workers tick after
     tick, stranding the extras. No "preemption in flight" marker guards this.
  3. **Gangs cannot re-form / workers stolen** — a coscheduled job needs all N
     workers free *simultaneously* (`scheduler.py:828-834`) but victims free
     asynchronously and independently (`reconcile/peers.py:44-49`); the freed workers
     idle and a solo first-fit task can steal one (`scheduler.py:759-789`), so the
     gang re-preempts. This is the churn engine.
- **Scale of the churn:** ~3,195 `event=task_preempted` in the recent log window;
  active tasks reaching attempt **134** (`/romain/dev-tpu-vllm-validation-sweep`),
  **66** (`/ryan/tomat-eval-maskgit-...mg-modal-v4`), **23**
  (`/michaelryan/10k-natural-fineweb-edu-r7`). 414 tasks currently in
  `TASK_STATE_COSCHED_FAILED (11)` — gang-unwind bookkeeping.
- **Fixed** (assign-and-defer-dispatch): the preemptor is committed ASSIGNED onto
  the worker its victim frees in the same transaction as the PREEMPT, and the
  reconcile loop defers its dispatch until the victim's attempt finalizes. See
  "Fix (implemented)" below.

## What the reported view actually is

There is **no** stale committed-resource leak. The two obvious "glitch" hypotheses
were checked and refuted:

- **Orphaned RUNNING task on a dead worker** — atomic single-snapshot query:
  481 RUNNING tasks, 477 with a worker set, **0** whose `current_worker_id` is
  missing from `workers`. None.
- **Ghost worker in `get_scheduler_state`** — an early cross-check flagged one worker
  the scheduler counted as running but with no DB task. It was a **snapshot-skew
  artifact**: `running_buckets` is a *DB projection* (`state=RUNNING AND
  current_worker_id IS NOT NULL`, `service.py:2981-2991`), and I had compared it
  against a *separately-timed* query. Within one snapshot it is self-consistent. The
  gang churns so fast (attempt 9+) that two RPCs taken seconds apart disagree.

The reported "free worker + one preempted job + pending higher-priority job" is a
**snapshot of the non-atomic preempt→free→place window**, made worse by the gang
all-or-nothing rule.

## Reproduction (live trace, all times 2026-07-15 UTC)

Victim: the `rohith` cross-region v6e-8 gang
`.../zephyr-xregion-pool-v6e-8-afaa222d-p0-workers-a0` (2 tasks, `/0` and `/1`).
Preemptor: `/ryan/tomat-eval-maskgit-train-mg-modal-v4-cont-clean-train_200-step-100000/0`
(a **single** v6e-8 task, at attempt 66).

```
15:15:39  task_preempted  gang/0  reason=Preempted by /ryan/tomat-...mg-modal-v4/0
15:15:49  task_preempted  gang/1  reason=Preempted by /ryan/tomat-...mg-modal-v4/0
15:15:51  "Dropping late update for terminal attempt ... attempt_state=10 reported=3"   # worker still running the killed task
15:16:02  assignment_queued  tomat/0        -> worker europe-west4-35c052b0   (== gang/0's old worker)
15:20:29  assignment_queued  gang/1         -> worker europe-west4-fd143cd5   (a BRAND-NEW worker, booted 15:15)
```

Reading:

- To free **one** worker for tomat, the scheduler had to preempt the **whole
  2-worker gang** (all-or-nothing). tomat reused gang/0's worker (`35c052b0`) but had
  no use for gang/1's worker (`60117dd7`) — that one was freed for nothing.
- gang/1 stayed pending **~4.5 minutes** (15:15:49 → 15:20:29) and then landed on a
  *newly autoscaled* worker rather than the one it was evicted from — the autoscaler
  is booting fresh v6e-8 slices (`35c052b0`, `60117dd7`, `fd143cd5`, `8c24319d` all
  booted 14:59-15:15) to chase demand that preemption keeps disrupting.
- The `Dropping late update ... reported=3` lines confirm the worker keeps running
  (and reporting) the preempted container after the controller has already freed it
  in accounting — the asynchronous-stop half of the window.

Worker `60117dd7` cycled through **three** owners in ~13 minutes (gang/1 @15:03 →
`/power/ft-fast-fork2` @15:11 → gang/0 @15:16), which is the churn from the same
mechanism seen from the worker's side.

## Investigation path

1. `cluster status` — controller healthy, 464/464 workers, running unmerged branch
   `weaver/fix-pending-job` @ `454ede7a5b`. Diffed it: **only federation/docs commits
   over main, no scheduler changes** — so this is current-main behavior.
2. Job/task state counts: 65 pending jobs, 1233 pending tasks; tasks by state
   included **414 in state 11** — undocumented in OPS.md. Confirmed via proto:
   `TASK_STATE_COSCHED_FAILED = 11`, `TASK_STATE_PREEMPTED = 10` (0 live).
3. Atomic checks refuted the leak hypotheses (above).
4. Read `get_scheduler_state` (`service.py:2952-3090`) — `running_buckets` is a DB
   projection, not in-memory. Explained the ghost as snapshot skew.
5. Used `iris process logs --substring=<X>` (finelog built-in filter; post-hoc
   `grep` on `--since` returns nothing) to trace `event=task_preempted` /
   `assignment_queued` for specific entities. Found the reason strings name the
   preemptor.
6. Traced tomat and the rohith gang end-to-end → the reproduction above.
7. Delegated a read-only scheduler-code map (subagent) confirming preempt/place is
   non-atomic by construction, with exact citations.
8. Quantified churn (3,195 preemptions; attempt-134 tail).

## Key code locations

- `scheduling/backend.py:383-384` — `apply_placements` then `apply_preemptions`;
  placement is the only placement pass.
- `scheduling/decision.py:34-46` — preemption candidates = tasks placement failed to
  place; the preemptor is unplaced by definition and never assigned here.
- `controller.py:1313-1322` (`_commit_schedule_decisions`) — assignments and
  preemptions commit in one txn, but the preemptor is not among the assignments.
- `reconcile/task.py` `preempt_one` (~246-284) — victim → PENDING, attempt left
  unfinished (`stamp_attempt_finished=False`).
- `reconcile/reads.py:714-717` — committed capacity = unfinished worker-bound
  attempts (so a preempted-but-not-stopped victim still holds its chips).
- `reconcile/reads.py:916` — victim set for preemption is `state = RUNNING` only.
- `controller.py:951-956` — `_force_reconcile` only on assignments, not preemptions.
- `scheduling/scheduler.py:828-834` — gang placement needs all N free at once.
- `scheduling/scheduler.py:759-789` — solo first-fit can steal a freed gang worker.
- `scheduling/policy.py:609` (`reserved_workers`) — per-tick only; no cross-tick
  reservation of freed capacity for a preemptor.
- `reconcile/peers.py:35-92` — gang unwind → siblings `COSCHED_FAILED` (not charged
  preemption budget); attempts left unfinished (async release).

## Fix (implemented — assign-and-defer-dispatch)

Option 2 above (transactional placement against the freed slot), realized as two
coordinated changes:

1. **Decision layer — co-assign the preemptor.** `run_preemption_pass`
   (`scheduling/policy.py`) now returns a `PreemptionPlan` carrying both the
   victim evictions *and* `placements`: each preemptor task bound to a worker its
   victim frees (solo → the victim's worker; gang whole-slice → the freed slice
   members, tpu-worker-id ordered; partial-host gang → the freed hosts plus the
   already-fitting ones). `run_scheduling_decision` (`backend.py`) appends those
   `placements` to the tick's assignments, so the preemptor commits ASSIGNED onto
   the freed worker in the *same* transaction as the victim's PREEMPT. It is no
   longer in the unscheduled set next tick, which kills over-preemption and
   worker-stealing; the worker is no longer "stranded free" (it visibly holds the
   incoming ASSIGNED preemptor).

2. **Reconcile layer — defer dispatch (`reconcile/worker.py`).** The freed worker
   is not physically empty until the victim's process dies. `_reconcile_worker`
   withholds the ASSIGNED preemptor's run-intent while a preemption victim
   (`_holds_preemption_victim` — an attempt in state PREEMPTED, which reconcile
   rows guarantee is still worker-bound) occupies the worker, so the preemptor
   never crash-loops against the live victim on the libtpu fixed port. The
   victim's terminal heartbeat stamps `finished_at_ms`, dropping it from the next
   reconcile snapshot and releasing the gate — a hold of one reconcile cycle in
   the healthy case.

`ops.task.assign` does not re-check capacity, so committing the preemptor onto the
still-committed victim worker is accepted; the worker only needs to be healthy.

Tests: `tests/cluster/controller/test_preempt_place_coassign.py` (end-to-end via
`ctrl._run_scheduling()`: co-assign + no cross-tick over-preemption — both red on
main), unit placement assertions in `test_preemption.py`
(`test_solo_preemptor_is_placed_on_the_freed_worker`,
`test_coscheduled_preemptor_is_placed_on_the_freed_slice`,
`test_gang_partial_host_places_gang_on_freed_and_fitting_hosts`), and dispatch-gate
units in `test_reconcile.py`.

**Remaining gap (not in this change):** a wedged victim whose process never
terminates never finalizes its attempt, so the gate holds the preemptor
indefinitely. This is a pre-existing infra failure mode (the reserved-until-
heartbeats contract already leaks a wedged worker's capacity); reclaim the worker
manually until an automatic stop-deadline watchdog lands. See OPS.md "Known Bug #3".

The extracted `_preempt_solo` / `_preempt_coscheduled` helpers came from sibling
branches (`iris-slice-preemption`, `weaver/coscheduled-task-preemption`); this
change only enriches their return type, so it composes with them.

## Caveats

- Attempt counts mix scheduler preemptions (`PREEMPTED`) with GCP preemptible
  reclaim (`WORKER_FAILED`); some churn is inherent to preemptible v6e capacity. But
  3,195 scheduler-driven `task_preempted` events is churn on top of that.
- The reporter's attached screenshots were not available to this session; the
  scenario was reconstructed from the live cluster.
