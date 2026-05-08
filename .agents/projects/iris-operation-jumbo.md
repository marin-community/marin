# Iris Controller: Reconcile Dispatch (`Operation Jumbo`)

## TL;DR

Delete `dispatch_queue`. Delete the kill dispatcher. Delete the start dispatcher's redispatch state machine. The whole control path is:

1. **Scheduler** writes `tasks.state = ASSIGNED` (existing state) atomically with `INSERT INTO task_attempts`. The attempt row is the resource ledger: an attempt holds worker resources while `task_attempts.worker_id IS NOT NULL AND task_attempts.finished_at_ms IS NULL`.
2. **Poll loop** every tick:
   - For each worker `W`: compute the expected set `E_W = {(t, a) : current attempt is on W and state IN {BUILDING, RUNNING}}`. Send `Poll(E_W)`. **Worker auto-kills any local task not in `E_W`.**
   - For each current attempt in `state=ASSIGNED`: send the start payload. Start is effectively fire-and-forget; the worker either starts/dedups the attempt or the normal heartbeat/liveness path catches failure.
3. **Producing transitions** (cancel, preempt, gang-cascade, worker-death) update task/attempt state but do **not** enqueue and do **not** stamp `task_attempts.finished_at_ms` for a still-running worker-bound attempt.
4. **Worker** auto-kills strays as a side effect of receiving `Poll(E_W)`. On terminal, it heartbeats; the heartbeat path stamps `task_attempts.finished_at_ms` and records the observed outcome. Resource release is derived from that timestamp. There is no separate decommit transition.

No `KILL_REQUESTED` state. No StopTasks RPC initiated by the controller. No `dispatch_queue`. No `pending_stop_resources`. No `workers.committed_*` cache in the correctness path. Scheduler availability is derived from unfinished `task_attempts`.

`current_attempt_id` (already exists, schema.py:737) is the per-task epoch. Job-name replacement uses soft-kill (state→terminal, prune later) — never CASCADE-delete a job whose tasks may still be running, because that destroys the `task_attempts` rows that describe resource ownership.

---

## 1. Goals / Non-goals

### Goals
- One control-plane path for reconcile. V1 can use existing `Poll` + `StartTasks`; the intended endpoint is a single `Reconcile` RPC that carries both expected tasks and start payloads. No `StopTasks` from controller.
- Worker auto-kills strays based on `Poll(expected)`. The kill is a worker-local operation, not a controller-driven RPC.
- `tasks.state` selects the current attempt; `task_attempts` is the dispatch and resource ledger.
- `available = total − SUM(resources for unfinished worker-bound attempts)`. No `pending_stop_resources` term and no `workers.committed_*` cache in the scheduler.
- Same model for worker-bound and K8s-bound tasks. K8s controller-side reconcile diffs `tasks` against the pod listing.
- Preserve all #5550 wins: heartbeat-confirmed resource release, single-snapshot scheduler read, three-phase RPC pattern (read → fan-out without lock → small write tx).

### Non-goals
- A new worker reconcile RPC in the first cut. `Poll` already carries `expected_tasks` and `StartTasks` already carries payloads. Once the state model is stable, replace both with one `Reconcile` RPC.
- New states. `ASSIGNED` (job.proto:205) and the existing terminals are enough.
- New columns on `task_attempts` for dispatch metadata (the prior version of this doc proposed `dispatched_at_ms` / `kill_dispatched_at_ms`; not needed — see §3.4 idempotency).

---

## 2. Background

The just-shipped PR (`agent/iris-kill-registry-refactor`, design `.agents/projects/scheduler-queue.md`) replaced direct dispatch with a SQLite `dispatch_queue` table, a kill dispatcher, a start dispatcher, and a `pending_stop_resources` join in the scheduler.

It closes #5470 by construction. It also carries complexity the design itself flags as not-strictly-load-bearing: after #5550, release is already supposed to wait for heartbeat-confirmed terminal state. The real canonical fact is the unfinished attempt row, not a second queue row or a mutable worker counter.

The reconcile design notices this and pushes further: if `task_attempts` is the source of truth for resource accounting, and worker auto-reconcile via `Poll` is the source of truth for "what should be running," then `dispatch_queue` is purely a redundant log of work that's also derivable from `tasks` and `task_attempts`. Drop it.

The simplification compounds. Without the queue:
- No kill dispatcher (controller never sends StopTasks; worker auto-kills).
- No redispatch state machine (worker reconciles every tick; if a kill is dropped, the next Poll re-sends the expected set).
- No CHECK constraint, no partial unique index, no `(worker_id, attempt_id)` discipline.
- No `pending_stop_resources`.
- No `workers.committed_*` cache in the scheduler. If scale tests need it later, reintroduce it as an in-memory scheduler abstraction derived from `task_attempts`, not as transition-maintained durable state.
- No `KILL_REQUESTED` state.
- No `task_attempts.dispatched_at_ms` columns.

What remains: scheduler writes `ASSIGNED`, poll loop sends `Poll` + `StartTasks`, heartbeat finalizes attempts, and scheduler capacity is derived from unfinished attempts.

---

## 3. The state machine

### 3.1 States

Existing states (job.proto:195-206) are sufficient:

```
TASK_STATE_PENDING        = 1   submitted, awaiting scheduler
TASK_STATE_ASSIGNED       = 9   scheduler picked target; start payload pending
TASK_STATE_BUILDING       = 2   worker received start; container starting
TASK_STATE_RUNNING        = 3   container running
TASK_STATE_SUCCEEDED      = 4   terminal
TASK_STATE_FAILED         = 5   terminal
TASK_STATE_KILLED         = 6   terminal
TASK_STATE_WORKER_FAILED  = 7   terminal
TASK_STATE_UNSCHEDULABLE  = 8   terminal
TASK_STATE_PREEMPTED      = 10  terminal-for-this-attempt; new attempt may exist
```

`ACTIVE_TASK_STATES = {ASSIGNED, BUILDING, RUNNING}` already exists (db.py:144-148).

No new state is added.

### 3.2 Cancel / preempt: tasks.state goes terminal directly

A "cancel" today writes a kill row to `dispatch_queue` and waits for heartbeat-confirmed terminal before flipping `tasks.state` to `KILLED`. Under reconcile, **the producing transition writes `tasks.state=KILLED` immediately**.

This sounds aggressive, but it isn't:
- The old attempt row remains unfinished. The producing transition does not stamp `task_attempts.finished_at_ms`.
- Worker still has the container running. Worker's next `Poll` excludes this task from `expected` (since `state ∉ ACTIVE`). Worker auto-kills.
- Worker heartbeats `(t, a) → KILLED`. Heartbeat path stamps `task_attempts.finished_at_ms = now_ms` and records the observed terminal state.
- The attempt no longer contributes to the scheduler's derived worker usage. Capacity returns. Scheduler can place new work.

The conservative-state property holds: `tasks.state=KILLED` does **not** mean "resources released." `task_attempts.finished_at_ms IS NULL` on a worker-bound attempt is the release-pending signal. Same property as #5550, just sourced from `task_attempts` instead of `dispatch_queue` or `workers.committed_*`.

### 3.3 Preempt with retry: old attempt stays unfinished until heartbeat

For preempt-with-retry: the producing transition (`preempt_task`, `_requeue_coscheduled_siblings`) writes:

```python
# Old attempt row stays with worker_id=W, attempt_id=N, finished_at_ms=NULL.
UPDATE task_attempts SET state=PREEMPTED
WHERE task_id=t AND attempt_id=N
# reconcile now excludes this attempt from the expected set

# Task row — back to pending. The scheduler later creates N+1.
UPDATE tasks
SET state=PENDING, current_worker_id=NULL, current_worker_address=NULL
WHERE task_id=t

# Later scheduler transaction, after choosing W':
INSERT INTO task_attempts (task_id=t, attempt_id=N+1, worker_id=W', state=ASSIGNED)
UPDATE tasks
SET state=ASSIGNED, current_attempt_id=N+1, current_worker_id=W'
WHERE task_id=t
```

How does worker `W` learn to kill the old attempt? The expected-set query filters on the task's current attempt and active state:

```sql
SELECT t.task_id, t.current_attempt_id
FROM tasks t
JOIN task_attempts ta
  ON ta.task_id = t.task_id AND ta.attempt_id = t.current_attempt_id
WHERE ta.worker_id = :worker
  AND t.state IN (TASK_STATE_ASSIGNED, TASK_STATE_BUILDING, TASK_STATE_RUNNING)
```

Before the retry is reassigned, `t.state=PENDING`, so `(t, N)` is excluded. After reassignment, `t.current_attempt_id=N+1` with `worker_id=W'`, so `(t, N)` is still excluded from W's expected set.

W's next Poll receives an expected set that excludes `(t, N)`. W auto-kills the container running `(t, N)`. Heartbeat path finalizes that attempt. The scheduler stops counting its resources because `finished_at_ms` is no longer NULL.

`current_attempt_id` is the per-task epoch. Worker dedup on `(task_id, attempt_id)` (`worker.py:679-692`) correctly distinguishes old vs new. No new state, no new column.

### 3.4 Idempotency: who handles redispatch?

Today's design has a 5s redispatch timer + 16-attempt poison pill on `dispatch_queue`. Under reconcile:

- **Start**: reconcile reads current `task_attempts.state=ASSIGNED` rows for the selected worker batch and sends start payloads. If the RPC is lost, the next reconcile for that worker re-sends the payload. Worker dedup on `(task_id, attempt_id)` makes duplicates harmless. The controller advances to BUILDING from the reconcile result or the normal worker status heartbeat. **No redispatch state machine needed** — worker-batch reconcile is the redispatch.
- **Kill**: kills are not RPC'd by the controller. Worker auto-kills based on Poll. If a Poll is dropped, the next tick sends another Poll. Eventually worker sees the expected set excludes the stray and kills it.
- **Stuck dispatch**: there is no meaningful `accepted=false` path. Start is delivered to the worker and either the worker starts/dedups it or the worker/task heartbeat path reports failure. If the worker is unreachable or wedged, heartbeat-fail-threshold trips and `_remove_failed_worker` synthesizes WORKER_FAILED. No `attempts` counter on the task; the worker-liveness check handles it.
- **Stuck kill (worker won't terminate)**: worker auto-kill is local; if the worker process is wedged, heartbeat-fail-threshold trips and `_remove_failed_worker` runs. Same path.

**Net: no `dispatched_at_ms` column. No `attempts` column. No poison-pill threshold on individual tasks.** The reconcile cadence + worker dedup gives idempotency for free; the heartbeat-fail-threshold gives liveness recovery for free.

If we observe in production that a *single* worker is otherwise healthy but one task is stuck (wedged container that won't stop), we can add a per-attempt watchdog later. For v1 it isn't needed.

---

## 4. The poll loop

```python
# controller.py
POLLING_TICK_INTERVAL = Duration.from_seconds(0.25)
RECONCILE_WORKER_BATCH_SIZE = 512

def _polling_tick(self) -> None:
    self._reconcile_worker_batch()  # Poll + starts from one DB snapshot.
    self._sync_direct_provider()    # K8s: same model, controller does the diff.
    self._drain_heartbeats()
    self._ping_worker_batch()
    self._fail_workers_over_threshold()
```

The controller does not have separate "start dispatcher", "kill dispatcher", and "full poll" stages. Each tick reconciles one bounded batch of active healthy workers. Over time the cursor rolls across the fleet, like the existing ping batching pattern. A wake can reset or prioritize the cursor for workers that just received assignments or terminal-producing transitions.

### 4.1 Worker-batch reconcile

```python
def _reconcile_worker_batch(self) -> None:
    # Phase 1: read snapshot, no write lock.
    with self._db.read_snapshot() as snap:
        workers = self._store.workers.next_reconcile_batch(
            snap,
            cursor=self._reconcile_cursor,
            limit=RECONCILE_WORKER_BATCH_SIZE,
        )
        worker_ids = [w.worker_id for w in workers]
        rows = self._store.attempts.reconcile_rows_for_workers(snap, worker_ids)

    actions = build_reconcile_actions(workers, rows)
    # actions[W] = {
    #   expected: list[(task_id, attempt_id)],  # BUILDING/RUNNING only
    #   starts: list[RunTaskRequest],
    # }

    # Phase 2: RPC fan-out. No DB lock. V1 may send Poll(E_W) and StartTasks;
    # the intended endpoint is one Reconcile RPC carrying both fields.
    results = self._provider.reconcile_workers(actions)

    # Phase 3: small write tx for Poll/status updates and start observations.
    with self._store.transaction() as cur:
        self._apply_reconcile_results(cur, results)
```

`reconcile_rows_for_workers` is the single task query for the worker batch:

```sql
SELECT
  ta.worker_id,
  t.task_id,
  ta.attempt_id,
  t.state AS task_state,
  ta.state AS attempt_state,
  j.num_tasks,
  jc.*
FROM tasks t
JOIN task_attempts ta
  ON ta.task_id = t.task_id AND ta.attempt_id = t.current_attempt_id
JOIN jobs j ON j.job_id = t.job_id
JOIN job_config jc ON jc.job_id = t.job_id
WHERE ta.worker_id IN (:worker_ids)
  AND ta.worker_id IS NOT NULL
  AND t.state IN (TASK_STATE_ASSIGNED, TASK_STATE_BUILDING, TASK_STATE_RUNNING)
```

Rows with `task_state IN {BUILDING, RUNNING}` become Poll `expected_tasks`. `ASSIGNED` rows do **not** go into Poll expected sets; they produce start payloads. This avoids the Poll-before-start race where a worker would report "task not found" for an attempt it has not received yet. Rows with `task_state=ASSIGNED AND attempt_state=ASSIGNED` become start payloads. `build_run_request(row)` reconstructs the `RunTaskRequest` from canonical tables (`jobs`, `job_config`, `job_workdir_files`) instead of relying on `dispatch_queue.payload_proto`. This is the same data currently serialized before `dispatch.enqueue_start`.

Every worker in the selected batch gets a Poll, including `Poll([])`. That is what kills local strays when the controller expects nothing on that worker.

`mark_building_if_current` updates both tables with attempt guards. In the interim `Poll` + `StartTasks` implementation this can run after the start RPC returns successfully. In the single-`Reconcile` version it runs from the worker's reconcile/status result.

```sql
UPDATE task_attempts
SET state = TASK_STATE_BUILDING, started_at_ms = COALESCE(started_at_ms, :now)
WHERE task_id = :task_id
  AND attempt_id = :attempt_id
  AND state = TASK_STATE_ASSIGNED;

UPDATE tasks
SET state = TASK_STATE_BUILDING
WHERE task_id = :task_id
  AND current_attempt_id = :attempt_id
  AND state = TASK_STATE_ASSIGNED;
```

Index: `idx_tasks_state_attempt` (schema.py:766) already covers `(state, task_id, current_attempt_id, job_id)`. Add an index on unfinished/assigned attempts if the join shows up in scale tests.

### 4.4 Wake events

Same wake-event split as #5550:

- `_scheduling_wake`: producing transitions that may free capacity (terminal heartbeats).
- `_polling_wake`: producing transitions that may produce new work to dispatch (any write to `tasks.state` that the dispatcher cares about — primarily new ASSIGNED, but also bulk state-changes like job cancellation).

The poll loop waits on `_polling_wake` with `POLLING_TICK_INTERVAL` timeout; `wait → clear → tick` order preserved.

### 4.5 Heartbeat drain

`_drain_heartbeats` (existing) does most of the work. The state-transition table gains no new entries — the path that takes RUNNING → KILLED on a terminal heartbeat already exists.

Two disciplines to enforce:

1. **Producing transitions never stamp `task_attempts.finished_at_ms` for worker-bound attempts.** The heartbeat path is the sole writer. This is a behavior change from today (`_requeue_coscheduled_siblings` at transitions.py:601-655 stamps `finished_at_ms`). Refactor: those transitions update `task_attempts.state=PREEMPTED` (for reporting) but leave `finished_at_ms` NULL. The heartbeat path stamps `finished_at_ms` when the worker confirms terminal, refining the state if needed.

2. **Attempt finalization is the resource-release transition.** There is no `decommit_resources` call. A worker-bound attempt holds resources while `finished_at_ms IS NULL`; once the heartbeat path sets `finished_at_ms`, scheduler usage derived from attempts drops automatically.

3. **Keep `UpdateTaskStatus` push in v1.** `PollTasksResponse` is a supplemental observation path, not the sole heartbeat. Today Poll returns statuses for expected tasks; unexpected local tasks are killed as a side effect and may not be represented as terminal confirmations in the response. Removing worker-push heartbeat requires a separate wire change.

The store API split should make this hard to misuse:

```python
attempts.mark_state(cur, task_id, attempt_id, state, error=None)
attempts.finalize_from_worker(cur, task_id, attempt_id, observed_state, finished_at_ms, error=None)
```

Only `finalize_from_worker` writes `finished_at_ms`.

### 4.6 K8s direct-provider

Same shape, controller-side:

```python
def _sync_direct_provider(self) -> None:
    if not isinstance(self._provider, K8sTaskProvider):
        return
    with self._db.read_snapshot() as snap:
        desired = self._store.attempts.list_active_direct_provider(snap)
        # state IN ACTIVE, current attempt, worker_id IS NULL
        to_start = [r for r in desired if r.state == TASK_STATE_ASSIGNED]
    pod_listing = self._provider.list_pods()
    actions = self._provider.diff(desired, pod_listing)
    # Pods in pod_listing but not in desired → DeletePod
    # Tasks in to_start but not in pod_listing → CreatePod
    self._provider.apply(actions)
    with self._store.transaction() as cur:
        for (task_id, attempt_id), kind in actions:
            if kind == "scheduled":
                self._store.attempts.mark_building_if_current(cur, task_id, attempt_id)
            elif kind == "deleted":
                self._store.attempts.finalize_from_worker(cur, task_id, attempt_id, TASK_STATE_KILLED, now_ms)
```

The K8s direct-provider's existing sync (`_run_direct_provider_loop` at controller.py:1601-1610, `drain_for_direct_provider` at transitions.py:2465-2543) is already this shape. The change is purely the input source: read `tasks` instead of `dispatch_queue` rows.

The pod-listing reconcile is the K8s analogue of `Poll(E_W)` for workers — controller diffs against external state to detect strays. Same model, controller is the actor instead of the worker.

---

## 5. Conservative scheduler state

```python
def _read_scheduling_state(self) -> SchedulingState:
    with self._db.read_snapshot() as snap:
        tasks = self._store.tasks.pending_for_scheduling(snap)
        workers = self._store.workers.list_active_healthy(snap)
        usage = self._store.attempts.resource_usage_by_worker(snap)
    return SchedulingState(tasks=tasks, workers=workers, usage=usage)
```

`available_R(W) = total_R(W) − resource_usage_by_worker(W)`. No `pending_stop_resources` join. No `dispatch_queue` read. No durable `workers.committed_*` cache. `resource_usage_by_worker` is a derived aggregate over worker-bound attempts with `finished_at_ms IS NULL`; it is not separate state.

Why #5470 still holds:

- Producing transition (preempt, cancel) writes `tasks.state=PREEMPTED|KILLED` on the to-be-stopped attempt's task. The old attempt row remains unfinished (`task_attempts.finished_at_ms IS NULL`).
- Scheduler derives usage from unfinished attempts → `available = 0` on the still-busy worker → can't double-book.
- Worker eventually heartbeats terminal → heartbeat tx finalizes the attempt → next scheduler tick can place.

Worker reconcile via `Poll` is what *causes* the worker to terminate; the conservative-state property is independent of when reconcile fires. Even if reconcile is delayed by a tick, scheduler decisions are correct because they're keyed on unfinished attempts, not on "kill RPC sent."

The usage query is the scheduler's resource ledger:

```sql
SELECT
  ta.worker_id,
  SUM(jc.res_cpu_millicores) AS cpu,
  SUM(jc.res_memory_bytes) AS mem,
  GROUP_CONCAT(jc.res_device_json) AS devices
FROM task_attempts ta
JOIN tasks t ON t.task_id = ta.task_id
JOIN job_config jc ON jc.job_id = t.job_id
WHERE ta.worker_id IS NOT NULL
  AND ta.finished_at_ms IS NULL
GROUP BY ta.worker_id
```

The controller parses `devices` in Python with a small cached helper in `codec.py`:

```python
class DeviceCounts(NamedTuple):
    gpu: int
    tpu: int


@lru_cache(maxsize=8192)
def device_counts_from_json(device_json: str | None) -> DeviceCounts:
    if not device_json:
        return DeviceCounts(gpu=0, tpu=0)
    device = proto_from_json(device_json, job_pb2.DeviceConfig)
    return DeviceCounts(gpu=get_gpu_count(device), tpu=get_tpu_count(device))
```

`resource_usage_by_worker()` should sum CPU/memory in SQL and use `device_counts_from_json()` for accelerator counts. This keeps JSON/proto parsing centralized and cached without introducing durable resource counters. If this is still too expensive at scale, reintroduce an in-memory per-scheduler-pass cache derived from this query. Do not reintroduce transition-maintained durable `committed_*` counters unless measurement shows the derived ledger is untenable.

---

## 6. Producing transitions: before/after

| Transition | Today (post-#5550) | Reconcile |
|---|---|---|
| `cancel_job(task)` (single) | enqueue_kill + state stays ACTIVE; heartbeat moves to KILLED | `UPDATE tasks SET state=KILLED WHERE task_id=t AND state IN ACTIVE` |
| `_kill_non_terminal_tasks(job)` | loop: enqueue_kill | `UPDATE tasks SET state=KILLED WHERE job_id=j AND state IN ACTIVE` (bulk) |
| `cancel_tasks_for_timeout` | loop: enqueue_kill | bulk UPDATE state=KILLED |
| `preempt_task(t)` | state=PREEMPTED on attempt + retry to PENDING or terminal + enqueue_kill on old | UPDATE old `task_attempts.state=PREEMPTED` without `finished_at_ms`; task goes PENDING if retry remains. Scheduler later inserts the next attempt. |
| `_requeue_coscheduled_siblings` | iterate: state=PREEMPTED + enqueue_kill per sibling | iterate: mark old attempt PREEMPTED without `finished_at_ms`; task goes PENDING. |
| `_terminate_coscheduled_siblings` | state=FAILED + enqueue_kill per sibling | UPDATE tasks SET state=FAILED per sibling. (Per-attempt per-worker reconcile kills the container.) |
| `_remove_failed_worker(W)` | synthesize WORKER_FAILED heartbeat for each non-terminal attempt; delete worker row | same |
| `queue_assignments` | INSERT task_attempts + state=ASSIGNED + committed_*+= + enqueue_run | INSERT task_attempts + state=ASSIGNED. No committed counter update. |

Every `dispatch.enqueue_*` call site collapses to a `tasks`/`task_attempts` UPDATE (or to nothing — if the existing UPDATE already happens, the enqueue was the only extra work). Incremental `add_committed_resources` and `decommit_resources` go away; scheduler usage is derived from unfinished attempts.

### 6.1 Subtle: producing transitions and `task_attempts.finished_at_ms`

Today (#5550), `_requeue_coscheduled_siblings` (transitions.py:601-655) stamps `task_attempts.state=PREEMPTED, finished_at_ms=now_ms` on the old attempt. Under reconcile, this is wrong — `finished_at_ms` means "the worker no longer holds resources for this attempt." If a producing transition stamps it before the worker actually exits, the scheduler can double-book the worker.

Refactor: producing transitions write `task_attempts.state` (for reporting) but never `finished_at_ms` for worker-bound attempts. The heartbeat path stamps `finished_at_ms` and confirms `task_attempts.state` when the worker confirms terminal.

This is a small but load-bearing change. Open question §10.3.

---

## 7. Job replacement and CASCADE-delete

You flagged: "when we cascade delete, we DELETE the task attempts. where do we store the KILL_REQUESTED in that case?"

Right answer: **don't CASCADE-delete tasks while their attempts may still be running.** Today's `submit_job` flow at service.py:1261-1346 does cleanup-then-resubmit; if a worker has a container running and the controller deletes the task row, CASCADE drops the `task_attempts` row. The eventual KILLED heartbeat lands at a controller that has no record of the old attempt, so the scheduler can no longer derive correct usage.

This is already a leak on today's design (independent of dispatch_queue). The reconcile design exposes it more clearly because reconcile is the only mechanism that tells the worker to stop.

### 7.1 Block replacement until the old job drains

When `submit_job` is called with an existing job name:

1. Mark all non-terminal tasks of the old job as terminal: `UPDATE tasks SET state=KILLED WHERE job_id=:old_job AND state IN ACTIVE`.
2. Workers running those tasks see them excluded from `Poll(E_W)` on the next tick. Workers auto-kill.
3. Workers heartbeat KILLED. Heartbeat path stamps `task_attempts.finished_at_ms`.
4. The launch RPC waits until the old job has no unfinished worker-bound attempts.
5. `service.py` deletes the old job rows and inserts the replacement using the same `job_id`.

Today `jobs.job_id` is not a UUID; it is a `JobName`, and for root jobs it is effectively the user-facing job name. Blocking the replacement RPC lets us keep that model. We do not need a `jobs.name` split for reconcile dispatch.

`service.py` waits for this drain condition:

```sql
SELECT 1
FROM tasks t
JOIN task_attempts ta ON ta.task_id = t.task_id
WHERE t.job_id = :job_id
  AND ta.worker_id IS NOT NULL
  AND ta.finished_at_ms IS NULL
LIMIT 1
```

When this query returns no rows, CASCADE delete is safe for accounting because no live worker-bound attempt rows remain. The wait must not hold the SQLite write transaction. Commit the soft-kill, wake reconcile, then poll with `rigging.timing.ExponentialBackoff`:

```python
def _wait_until_job_drained(self, job_id: JobName, timeout: Duration) -> None:
    def drained() -> bool:
        with self._store.read_snapshot() as snap:
            return not self._store.jobs.has_unfinished_worker_attempts(snap, job_id)

    ExponentialBackoff(initial=0.05, maximum=1.0, factor=1.5).wait_until_or_raise(
        drained,
        timeout=timeout,
        error_message=f"Timed out waiting for old job {job_id} to drain",
    )
```

On timeout, return `DEADLINE_EXCEEDED` from the launch RPC. The old job remains soft-killed; a later retry can finish the replacement after workers report terminal or the worker-failure path finalizes the attempts.

---

## 8. Worker side

### 8.1 `Poll(expected_tasks)` semantics

Already implemented (`worker.py:851-882`, `_reconcile_expected_tasks`). The worker:

1. Receives `expected_tasks: list[(task_id, attempt_id)]` (plus current state hint per entry, but the hint isn't load-bearing — local state is authoritative).
2. For every locally-running task not in `expected`: kill its container.
3. For every entry in `expected` that the worker doesn't have running: report missing. `ASSIGNED` attempts are not in `expected`; start payloads deliver them.
4. Reports back its current view of running tasks.

The 30s grace window for the StartTasks→PollTasks race (`_recent_submissions` in `worker.py`) can stay during the interim `Poll` + `StartTasks` implementation. It should become unnecessary once `ASSIGNED` attempts are excluded from Poll expected sets and then disappear with the single `Reconcile` RPC.

### 8.2 Start payload semantics

Interim wire shape: `StartTasks` carries `RunTaskRequest` payloads. Soon after, `StartTasks` and `Poll` collapse into a single `Reconcile` RPC with `expected_tasks` and `start_tasks` fields. Either way, start delivery is not a queue with accepted/rejected rows. The worker dedups on `(task_id, attempt_id)` (`worker.py:679-692`). The controller advances the selected attempt to `BUILDING` only if it is still the task's current attempt.

### 8.3 No new RPCs

The prior version of this doc proposed `GetTaskPayload(task_id, attempt_id)` for worker-pull-on-drift. It's not needed: the reconcile action carries start payloads and the next reconcile for that worker re-sends until the attempt leaves `ASSIGNED`. The only drift case is "worker has a container the controller doesn't know about" — and the worker's own auto-kill handles that (it's not in `expected`, so worker kills it).

---

## 9. Migration plan

Each step independently shippable and reversible.

### Step 0: Attempt ledger + delete guard
- Add `TaskAttemptStore.resource_usage_by_worker()`.
- Add cached `device_counts_from_json() -> DeviceCounts` in `codec.py`, where `DeviceCounts` is a `NamedTuple`.
- Change scheduler state reads to derive worker usage from unfinished attempts instead of `workers.committed_*`.
- Remove scheduler dependence on `workers.committed_cpu_millicores`, `workers.committed_mem_bytes`, `workers.committed_gpu`, and `workers.committed_tpu`.
- Add the unfinished-attempt guard to `remove_finished_job`, pruning, replacement, and any admin delete path.
- Keep the old columns only until the code no longer reads them; then drop them in a schema migration.

### Step 1: Same-name replacement policy
Per §7. Land early; independent correctness improvement on today's design. `service.py` should soft-kill the old job, wait with `rigging.timing.ExponentialBackoff` until no unfinished worker-bound attempts remain, then delete the old rows and insert the replacement with the same `job_id`.

### Step 2: Discipline producing transitions to never stamp `task_attempts.finished_at_ms`
Per §6.1. Audit and refactor every call site that stamps `finished_at_ms` outside the heartbeat path. Split `attempts.mark_state` from `attempts.finalize_from_worker`. Reversible: revert the call-site changes.

### Step 3: Producing transitions write `tasks.state=KILLED|PREEMPTED|FAILED` directly (in addition to enqueue)
Shadow mode. Existing dispatch_queue path still drives RPCs. Add an assertion-only check that the two paths agree. Logs only on mismatch.

### Step 4: Add worker-batch reconcile in shadow mode
- Add `_reconcile_worker_batch` reading `attempts.reconcile_rows_for_workers` for the next `N` active healthy workers.
- Build both `Poll(E_W)` and StartTasks payloads from that one snapshot.
- Keep existing dispatch_queue-driven start/kill paths enabled while logging mismatches between queue rows and reconcile-derived actions.
- Every worker in the selected batch receives `Poll(E_W)`, including `Poll([])` for workers with no expected tasks; auto-kill any local strays.
- Exclude `ASSIGNED` attempts from Poll expected sets; they are represented only as start payloads.
- Reversible: re-enable the queue-driven kill path.

### Step 5: Promote worker-batch reconcile to authoritative dispatch
- Disable `_dispatch_pending_kills_once` and `_dispatch_pending_starts_once`.
- Stop calling `dispatch.enqueue_stop` in producing transitions.
- Stop calling `dispatch.enqueue_start` in `queue_assignments`.
- Starts are driven by current `task_attempts.state=ASSIGNED` rows in the selected worker batch.
- Kills are driven by omission from `Poll(E_W)`.
- Once stable, replace the interim `Poll` + `StartTasks` fan-out with a single worker `Reconcile` RPC.
- Reversible: re-enable.

### Step 6: K8s direct-provider switches to `tasks` / `task_attempts` reads
- `_sync_direct_provider` reads active null-worker tasks/attempts, matching today's direct-provider identity.
- Stop writing K8s rows to `dispatch_queue`.
- Reversible: revert.

### Step 7: Drop `dispatch_queue`
- `0044_drop_dispatch_queue.py`: `DROP TABLE dispatch_queue`.
- Delete `DispatchQueueStore`, `pending_stop_resources_by_worker`, `enqueue_*`, `delete_*`, `mark_*`, `pending_*`.
- Cumulative point of no return; revert requires reverting Steps 6/5/4 as a unit.

---

## 10. Compare/contrast vs shipped #5550

| Concern | #5550 | Reconcile |
|---|---|---|
| Schema | `dispatch_queue` (7 columns, 2 indexes, CHECK) | no dispatch schema; keep `job_id == name` for root jobs by blocking replacement until drain |
| Store API | 9 worker-typed methods + `pending_stop_resources_by_worker` | no dispatch store; `attempts.reconcile_rows_for_workers`, `attempts.resource_usage_by_worker`, `tasks.bulk_set_state` |
| Producing transitions | enqueue scattered through transitions.py | `UPDATE tasks.state` only |
| Kill dispatch | controller fans out StopTasks via dispatcher loop | worker auto-kills based on `Poll(E_W)` |
| Start dispatch | controller fans out StartTasks via dispatcher loop, redispatch state machine | reconcile action carries start payloads for ASSIGNED attempts; repeated reconcile is redispatch |
| Scheduler read | tasks + workers + dispatch_queue (one snapshot) | tasks + workers + unfinished attempts (one snapshot) |
| Heartbeat path | delete kill row + decommit | finalize attempt (`finished_at_ms`); resource release is derived |
| K8s path | reads `dispatch_queue WHERE worker_id IS NULL` | reads active null-worker tasks/attempts |
| Worker side | `PollTasks` reconcile plus `StartTasks` | interim uses both; target is one `Reconcile` RPC |
| Wake events | `_scheduling_wake`, `_polling_wake` | same |
| Three-phase RPC | yes | yes |
| `pending_stop_resources` | subtracted from available | not needed (unfinished attempts are sufficient) |
| `workers.committed_*` | durable mutable cache | removed from scheduler semantics; reintroduce only as measured in-memory scheduler cache if needed |
| `KILL_REQUESTED` state | implied by kill row in queue | doesn't exist — kill is implied by absence from `expected` |
| Redispatch state | `dispatched_at_ms`, `attempts`, poison threshold on each row | none — the loop itself is the redispatch |
| Lines (estimated) | +1786 / −679, 6 new files | est. +600 / −1900: net negative vs main |

The reconcile design is structurally smaller than today's main *and* much smaller than #5550. The dominant savings come from deleting:
- The kill dispatcher (controller never sends StopTasks).
- The redispatch state machine (no `dispatched_at_ms`, no `attempts`, no poison-pill).
- `pending_stop_resources_by_worker` and the second-snapshot scheduler read.
- The `KILL_REQUESTED` state and its handling.
- The CHECK constraint and the K8s/non-K8s row split.

---

## 11. Walkthroughs

### 11.1 #5470: preempt-then-reassign

slice-3-w1 runs `gang-a-task-1` (attempt M), which requires 8 chips. Higher-priority job preempts. Producing transition fires:

```python
# preempt_task(gang-a-task-1)
UPDATE task_attempts SET state=PREEMPTED
  WHERE task_id='gang-a-task-1' AND attempt_id=M
UPDATE tasks SET state=PENDING, current_worker_id=NULL
  WHERE task_id='gang-a-task-1'
# task_attempts(M).finished_at_ms is NULL.
```

Scheduler derives `used_tpu(slice-3-w1) = 8` from unfinished attempts, so `available_tpu(slice-3-w1) = 8 - 8 = 0`. Cannot double-book.

slice-3-w1's next Poll excludes `(gang-a-task-1, M)` because the task is no longer active on that attempt. Worker kills the container.

Worker heartbeats: `(gang-a-task-1, M) → KILLED`. Heartbeat path:

```python
attempts.finalize_from_worker(cur, 'gang-a-task-1', M, KILLED, now_ms)
# task_attempts.state stays PREEMPTED (not overwritten by heartbeat in this case;
# the controller-set state is the higher-truth: this attempt was preempted.)
# task_attempts.finished_at_ms = now_ms
```

Next scheduler tick: derived resource usage on slice-3-w1 is 0, so `available_tpu(slice-3-w1) = 8`. Can place new work.

**Conservative state held throughout, identical to #5550.** Mechanism is simpler.

### 11.2 Coscheduled gang kill

Gang `gang-b` has 4 members on workers W1..W4. User cancels:

```python
UPDATE tasks SET state=KILLED
WHERE job_id IN (SELECT task_id FROM tasks WHERE parent_job_id=:job)
  AND state IN ACTIVE_TASK_STATES
# 4 rows updated, atomically. Attempt rows remain unfinished until worker terminal heartbeat.
self._polling_wake.set()
```

Next tick: each of W1..W4 receives `Poll(E_Wi)` where `E_Wi` excludes the killed task (since `state ∉ ACTIVE`). Each worker kills its container. Heartbeats trickle in independently. As each lands, the heartbeat path stamps `finished_at_ms`; the attempt drops out of scheduler usage.

All-or-nothing at request-level (single tx UPDATE). Per-worker independent at confirmation-level. Same as #5550.

### 11.3 Worker death

W3 misses N pings. Full-poll branch trips. `_remove_failed_worker(W3)`:

```python
for attempt in attempts_on_w3_active:
    self._heartbeat_worker_failed(cur, attempt.task_id, attempt.attempt_id, now_ms)
    # finalizes attempt and advances tasks.state to WORKER_FAILED
    # (or triggers retry → state=ASSIGNED on a new worker)
self._store.workers.remove(cur, W3)
```

W3's row gone. Tasks reassigned by next scheduler tick or terminal in WORKER_FAILED. No `dispatch_queue` rows to clean up. No `KILL_REQUESTED` rows to clean up. The `task_attempts.worker_id SET NULL` CASCADE is a no-op for finalized rows; only fires if the heartbeat synthesis missed a row, which the loop above prevents.

### 11.4 Job replacement

User submits `experiment-foo` with new code. The existing `experiment-foo` job has 50 tasks running on workers.

```python
# service.submit_job
with transaction() as cur:
    self._transitions.cancel_job(cur, JobName.root(user, 'experiment-foo'), "Replaced by new submission")
    self._polling_wake.set()

_wait_until_job_drained(JobName.root(user, 'experiment-foo'), timeout=Duration.from_minutes(10))

with transaction() as cur:
    self._transitions.remove_finished_job(cur, JobName.root(user, 'experiment-foo'))
    self._transitions.submit_job(cur, JobName.root(user, 'experiment-foo'), request, now)
```

Next Poll tick: the 50 workers see those tasks excluded from `expected`. They auto-kill. Heartbeats trickle in. Attempt finalization happens normally.

Once the old job's worker-bound attempts are all finalized, `service.py` deletes the old rows and inserts the replacement using the same `job_id`. The launch request blocks during the drain wait, but no SQLite write transaction is held while waiting.

### 11.5 Controller restart mid-dispatch

Controller restarts. `tasks.state=ASSIGNED, current_worker_id=W` for some `(t, a)`. Worker W may already have the container running because the start payload landed before the crash, but the controller did not persist BUILDING.

After restart:
- When W's reconcile batch runs, `_reconcile_worker_batch` sees the current assigned attempt row and sends its start payload. W starts or dedups `(t, a)`.
- The attempt is not included in Poll `expected_tasks` until it is BUILDING/RUNNING, so Poll-before-start cannot produce a false missing-task report.
- Phase 3 small write: advance `tasks.state=ASSIGNED→BUILDING`.

Or, worker had already heartbeated BUILDING/RUNNING before crash:
- Heartbeat write was committed → after restart, `tasks.state=BUILDING/RUNNING` already. Reconcile does not build a StartTasks payload for it.
- Either way: convergence.

No new state machine needed. Restart recovery is implicit in the per-tick read-and-dispatch loop.

---

## 12. Open questions

1. **Does Poll include per-task state confirmation?** Resolved for v1: keep `UpdateTaskStatus` push as the canonical heartbeat path. Poll remains reconcile plus supplemental observation. Removing worker-push heartbeat requires changing `PollTasksResponse` to include unexpected tasks killed by reconciliation and their eventual terminal outcomes.

2. **Single `Reconcile` RPC rollout.** Directionally resolved: collapse `Poll` and `StartTasks` into one RPC after the DB/state cleanup lands. The interim implementation can keep the existing wire methods to reduce blast radius, but the controller should already build one per-worker reconcile action containing both `expected_tasks` and start payloads.

3. **`finished_at_ms` discipline rollout.** Step 1 of the migration. Audit every producing-transition call site (`_requeue_coscheduled_siblings:601-655`, `_terminate_coscheduled_siblings`, others) for stamping `finished_at_ms` and refactor to leave it NULL. This is a behavior change; needs careful review. The Step-1 shadow mode catches mismatches.

4. **K8s pod-creation failure.** If `CreatePod` fails (image pull, namespace gone), what's the right state transition? Options: stay in ASSIGNED (next tick re-fires); transition to FAILED (gives up); transition to UNSCHEDULABLE. Today's K8s sync has its own retry; reconcile inherits it.

5. **Worker-batch sizing.** `RECONCILE_WORKER_BATCH_SIZE` controls the tradeoff between reconcile latency and RPC volume. At 4K workers and a batch size of 512 every 250ms, the whole fleet is reconciled roughly every 2s. Wake-triggered workers can be pushed into a priority lane so new assignments and kills do not wait for a full cursor rotation.

6. **Watchdog for wedged kills.** If a worker auto-kills but the container hangs in shutdown, what happens? Heartbeat-fail-threshold trips eventually (worker stops sending heartbeats because it's wedged on the kill); `_remove_failed_worker` synthesizes WORKER_FAILED. But that's catastrophic (loses the whole worker). For "container wedged but worker fine" — the worker process can implement its own per-container kill-with-timeout-then-SIGKILL. Existing worker behavior; out of scope for this doc.

7. **Atomicity guarantees for gang-kill across workers.** Single tx UPDATE bulks the state changes. Per-worker Poll fan-out is independent. If one Poll RPC fails, the next tick retries. Worst case: gang sibling A is killed, B is still running for a tick. B's unfinished attempt keeps its worker usage high; scheduler doesn't double-book; B's worker eventually kills via the next Poll. No correctness issue, only latency.

8. **Does `Poll(E_W)` need a sequence number?** For out-of-order delivery / split-brain detection. Today the client side is HTTP/2 over gRPC; ordering is per-stream. If worker process restarts mid-flight, the new process's local set is empty; first Poll auto-kills nothing; controller's expected set is delivered; worker pulls payloads via StartTasks dedup. No seq needed.

9. **Replacement wait timeout.** Existing jobs use `job_id == name` for root jobs. We keep that model by blocking same-name replacement until the old job drains. The open question is the timeout and user-facing error: the default should be long enough for normal worker shutdown, but finite so a launch RPC does not hang forever if worker failure finalization is broken.

---

## 13. Tunables

| Tunable | Default | Reasoning |
|---|---|---|
| `POLLING_TICK_INTERVAL` | 250ms | matches #5550 |
| `RECONCILE_WORKER_BATCH_SIZE` | 512 | bounds per-tick Poll/Start fan-out while keeping full-fleet reconcile latency low |
| heartbeat-fail-threshold | unchanged | sole liveness mechanism for stuck-RPC recovery |

No `DISPATCH_REDISPATCH`. No `DISPATCH_POISON_THRESHOLD`. Removed by deleting the redispatch state machine.
No durable `workers.committed_*` cache. If scheduling-state reads are too expensive, add an in-memory cache inside the scheduler read path after measurement.

---

## 14. Test strategy

### Unit
- State-transition coverage: PENDING → ASSIGNED → BUILDING → RUNNING → terminal. Cancel from each non-terminal state goes directly to KILLED.
- Producing transitions: assert `tasks.state` change post-cancel/preempt/gang-cascade.
- Heartbeat path: terminal advances `tasks.state` (if not already terminal) and stamps `task_attempts.finished_at_ms`. Idempotent on second heartbeat.
- Producing-transition `finished_at_ms` discipline: never stamped outside heartbeat path. Add a debug-build assertion.
- Scheduler usage: unfinished worker-bound attempts contribute resources; finalized attempts do not.
- Device count helper: `device_counts_from_json()` returns the expected `DeviceCounts(gpu, tpu)` for empty, GPU, TPU, and repeated JSON values.

### Integration
- #5470 regression: rewrite around unfinished-attempt conservative state.
- Counterfactual tripwire: prematurely stamp `finished_at_ms` in cascade and confirm the test fails.
- Stop-after-reassignment: preempt + reassign + heartbeat-from-old-worker-for-old-attempt → old-worker usage drops only after finalization.
- Three-phase RPC: writer lock available during fan-out.
- Single-snapshot scheduler read: ==1 read_snapshot per tick.
- Job replacement: old job's tasks reach KILLED; new job's tasks proceed independently; no leak; old job pruned after all-terminal.
- Delete/prune guard: terminal job with unfinished worker-bound attempt is not deleted.
- Controller restart with task in ASSIGNED: convergence to BUILDING on the next reconcile for that worker, with priority-lane coverage for newly assigned workers.
- Worker auto-kill via Poll: insert "stray" task on worker, controller's expected set excludes it, next tick auto-kills.
- Poll/Start race: ASSIGNED attempts are sent as start payloads but excluded from Poll expected sets until BUILDING/RUNNING.

### End-to-end
- Replay-golden regeneration. Diff: `state=KILLED` rows replace `dispatch_queue` rows; `finished_at_ms` only stamped at heartbeat.
- 4K-worker scale: Poll+Start RPC volume per tick. Confirm tick duration < 250ms in steady state.

### Coverage gaps
- Wake event shortens tick.
- K8s pod-creation failure semantics.
- Job replacement during in-flight scheduling tick.

---

## 15. Summary of deltas vs the prior version of this doc

The earlier draft of this doc proposed a `KILL_REQUESTED` state and a separate kill-dispatcher reading it. You correctly observed:

1. **The kill dispatcher is unnecessary if `Poll(E_W)` makes the worker auto-kill.** Removing it deletes a whole code path: no `_dispatch_kill_requested_workers`, no kill-side redispatch state, no per-attempt `kill_dispatched_at_ms` column.
2. **CASCADE delete of `task_attempts` destroys the row the heartbeat path needs to find.** The fix is soft-delete (state→terminal, prune later), not adding a column to track kill-pending across deletes.
3. **The poll loop reads better as one grouping.** Both expected-task reconciliation and start payload delivery group by worker; they share Phase 1 and Phase 2.

The result is meaningfully simpler than the prior draft, and structurally smaller than today's main. The control plane is: `tasks.state` selects the current attempt; `task_attempts` is the ledger; reconcile kills by omission and delivers starts; heartbeat closes the loop.
