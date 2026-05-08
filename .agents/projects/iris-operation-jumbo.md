# Iris Controller: Reconcile Dispatch (`Operation Jumbo`)

## TL;DR

Reorganize the controller around two invariants:

1. **`tasks` and `task_attempts` are the only state that matters.** Scheduler availability, worker reconcile, K8s reconcile, and restart recovery are all derived from those two tables. No `workers.committed_*` cache. No `dispatch_queue`. No in-memory mirror of intent.
2. **A `task_attempt` row holds worker resources iff `worker_id IS NOT NULL AND finished_at_ms IS NULL`.** Producing transitions write task/attempt state. The heartbeat path (and only the heartbeat path) stamps `finished_at_ms`.

The control path is:

1. **Scheduler** writes `tasks.state = ASSIGNED` atomically with `INSERT INTO task_attempts(state=ASSIGNED, worker_id=W)`. The new attempt row immediately holds W's resources in the scheduler's derived ledger.
2. **Poll loop** every tick, for a bounded batch of healthy workers:
   - Compute `E_W = {(t, a) : current attempt is on W and task state IN {BUILDING, RUNNING}}`.
   - Compute `S_W = {(t, a, RunTaskRequest) : current attempt is on W and task state = ASSIGNED}`.
   - Send `Reconcile(E_W, S_W)`. (Interim impl: `Poll(E_W)` + `StartTasks(S_W)`.)
   - **Worker auto-kills any local task not in `E_W ∪ {start payload task_ids}`.**
3. **Producing transitions** (cancel, preempt, gang cascade, worker death) update `tasks.state` and `task_attempts.state`. They do **not** stamp `task_attempts.finished_at_ms` for worker-bound attempts.
4. **Worker** auto-kills strays as a side effect of `Reconcile`. On terminal, it heartbeats; the heartbeat path stamps `finished_at_ms`. Resource release is derived from that timestamp.

`current_attempt_id` (schema.py:737) is the per-task epoch; worker dedup on `(task_id, attempt_id)` (worker.py:679-692) makes the rest follow.

Job-name replacement uses soft-kill plus a drain wait — never CASCADE-delete a job whose tasks may still be running, because that destroys the `task_attempts` rows that describe resource ownership.

---

## 1. Goals / Non-goals

### Goals
- One control-plane path for reconcile, replacing today's split between `_dispatch_assignments_direct` (scheduler thread, controller.py:2270), `_stop_tasks_direct` (heartbeat-updater thread, controller.py:2349), and inline `_poll_all_workers` in the scheduling tick (controller.py:2422).
- Worker auto-kill driven by reconcile expectation, not by a controller-issued `StopTasks` RPC.
- Scheduler availability derived from unfinished worker-bound `task_attempts`. No `workers.committed_*`.
- Single source of truth for state transitions: one `_terminate_task` helper, with explicit producer-vs-heartbeat parameterization controlling whether `finished_at_ms` is stamped.
- K8s direct-provider has the same shape as worker provider: scheduler writes `ASSIGNED`, reconcile diffs `tasks` against pod listing, applies pod creates/deletes, heartbeat-equivalent confirmation finalizes attempts. No buffered K8s kill queue.

### Non-goals
- A new worker reconcile RPC in the first cut. Interim wire shape: existing `Poll` + `StartTasks` (driven from one snapshot, one fan-out). Once the state model is stable, collapse them into a single `Reconcile` RPC.
- New task states. `ASSIGNED` (job.proto:205) and the existing terminals are sufficient.
- New columns on `task_attempts`. The current `(state, started_at_ms, finished_at_ms, exit_code, error)` set is enough.
- Watchdogs for "worker accepted the RPC but didn't act on it." If the worker process is sick, heartbeat-fail-threshold reaps it. If the RPC fails, the next reconcile re-sends. We don't model "worker is fine, but doing the wrong thing."

---

## 2. Background: where main is today

Current main (commit `040f97585`, this branch adds only design docs):

- **Worker dispatch is direct.** Scheduling loop calls `_dispatch_assignments_direct` (controller.py:2270), which opens a transaction, calls `transitions.queue_assignments(direct_dispatch=True)`, commits, then fans out `provider.start_tasks` against the addresses captured in `start_requests`.
- **Worker kill is direct.** `_process_heartbeat_updates` (controller.py:2443) collects `tasks_to_kill` from heartbeat-driven cascade transitions and calls `_stop_tasks_direct` (controller.py:2349).
- **Polling is inline.** `_poll_all_workers` (controller.py:2422) runs at the end of each scheduling tick. It reads `tasks WHERE current_worker_id IN (active_healthy) AND state IN ACTIVE_TASK_STATES` (`get_running_tasks_for_poll` at transitions.py:2220) and fans out `provider.poll_workers`. The worker side (`_reconcile_expected_tasks` at worker.py:854) already auto-kills local tasks not in `expected`; the 30s `_recent_submissions` grace window covers the StartTasks→PollTasks race.
- **`dispatch_queue` is K8s-only.** It exists in schema (schema.py:955) but only K8s uses it: `queue_assignments(direct_dispatch=False)` writes `enqueue_run` rows for K8s pods (transitions.py:1306), and `buffer_direct_kill` writes `enqueue_kill(worker_id=NULL)` rows for K8s task kills (transitions.py:2632). The K8s sync loop drains both. Worker-bound dispatch never touches the table.
- **`workers.committed_cpu_millicores | committed_mem_bytes | committed_gpu | committed_tpu`** are durable counters maintained by `add_committed_resources` / `decommit_resources` (stores.py:1762-1787). The scheduler reads them every tick (scheduler.py:189-192).
- **`_terminate_task` (transitions.py:353)** is the existing single helper for moving a task and its current attempt out of active state. It always stamps `task_attempts.finished_at_ms = now_ms` for terminal states (transitions.py:383-393), regardless of whether the worker is still holding the container. This is the source of the #5470 class of bug: capacity returns to the scheduler before the worker has actually exited.

The reconcile design takes the existing patterns at face value and rewires them around `task_attempts` as the resource ledger:

- `_dispatch_assignments_direct` → folded into the polling tick, driven by `tasks.state=ASSIGNED` rows.
- `_stop_tasks_direct` → deleted. Worker auto-kill via `Reconcile` is the only kill path.
- `_poll_all_workers` → generalized into per-worker reconcile that carries both expected set and start payloads.
- `workers.committed_*` → derived in the scheduler from unfinished worker-bound attempts.
- `_terminate_task` → keep as the single helper, but split the `finished_at_ms` write so producers cannot release resources before the worker confirms.
- K8s `dispatch_queue` rows + `buffer_direct_kill` → deleted. K8s reconcile reads `tasks` directly and diffs against pod listing.

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

### 3.2 Cancel / preempt: `tasks.state` goes terminal directly

The producing transition writes `tasks.state=KILLED|PREEMPTED` immediately and updates the attempt's reporting state, but **does not stamp `task_attempts.finished_at_ms`** while the attempt is worker-bound.

- Worker still has the container running. Worker's next reconcile excludes this task from `E_W` (state ∉ ACTIVE for the current attempt, or the current attempt has rolled). Worker auto-kills.
- Worker heartbeats `(t, a) → KILLED`. Heartbeat path stamps `finished_at_ms = now_ms`.
- The attempt no longer contributes to the scheduler's derived worker usage. Capacity returns. Scheduler can place new work on W.

The conservative-state property: `tasks.state=KILLED` means "the controller no longer wants this task to run"; it does **not** mean "resources released." `task_attempts.finished_at_ms IS NULL` on a worker-bound attempt is the release-pending signal. This is the same property #5550 attempted to enforce, sourced from `task_attempts` directly instead of via a kill queue or a mutable counter.

### 3.3 Preempt with retry

For preempt-with-retry, the producing transition leaves the old attempt unfinished:

```python
# preempt_task / _requeue_coscheduled_siblings
UPDATE task_attempts SET state=PREEMPTED
WHERE task_id=t AND attempt_id=N
# task_attempts(N).finished_at_ms stays NULL.

UPDATE tasks
SET state=PENDING, current_worker_id=NULL, current_worker_address=NULL
WHERE task_id=t

# Later scheduler tx, after picking W':
INSERT INTO task_attempts (task_id=t, attempt_id=N+1, worker_id=W', state=ASSIGNED)
UPDATE tasks
SET state=ASSIGNED, current_attempt_id=N+1, current_worker_id=W'
WHERE task_id=t
```

Worker W's reconcile expected-set query joins on `current_attempt_id`, so `(t, N)` is excluded once the task is back to PENDING and again once `current_attempt_id=N+1`. W auto-kills, heartbeats KILLED, heartbeat path stamps `finished_at_ms` on attempt N. Scheduler usage for W drops; W' already shows N+1 in its usage from the moment of insert.

### 3.4 Idempotency

- **Start**: reconcile reads `task_attempts.state=ASSIGNED` for the worker batch and sends start payloads. If the RPC fails, `task_attempts.state` is still ASSIGNED, and the next reconcile for that worker re-sends. Worker dedup on `(task_id, attempt_id)` makes duplicates harmless. The controller advances to BUILDING when the heartbeat reports it. **No redispatch state machine, no `dispatched_at_ms`, no `attempts` counter.**
- **Kill**: kills are not RPC'd. Worker auto-kills based on absence from `E_W`. Dropped reconcile → next tick re-sends.
- **Worker dies / wedges**: heartbeat-fail-threshold trips and `_remove_failed_worker` synthesizes WORKER_FAILED for each non-terminal attempt on that worker, finalizing them. Tasks bounce back to PENDING (or to terminal, depending on retry budget).

We do not model "worker accepted the RPC but didn't perform the work." The RPC either fails (handled by reconcile retry) or succeeds (worker eventually reports state). A genuinely sick worker stops heartbeating and the failure-threshold path catches it.

---

## 4. The poll loop

```python
# controller.py
POLLING_TICK_INTERVAL = Duration.from_seconds(0.25)
RECONCILE_WORKER_BATCH_SIZE = 512

def _polling_tick(self) -> None:
    self._reconcile_worker_batch()  # Phase 1-3 below.
    self._sync_direct_provider()    # K8s, same shape.
    self._drain_heartbeats()
```

The ping loop (`_run_ping_loop`, controller.py:2378) stays separate and on its own cadence; reconcile does not touch worker liveness.

### 4.1 Worker-batch reconcile

Three-phase RPC pattern (preserved from #5550 work):

```python
def _reconcile_worker_batch(self) -> None:
    # Phase 1: read snapshot, no write lock.
    with self._db.read_snapshot() as snap:
        workers = self._store.workers.next_reconcile_batch(
            snap, cursor=self._reconcile_cursor, limit=RECONCILE_WORKER_BATCH_SIZE,
        )
        rows = self._store.attempts.reconcile_rows_for_workers(snap, [w.worker_id for w in workers])
    actions = build_reconcile_actions(workers, rows)
    # actions[W] = ReconcileAction(expected=[(task_id, attempt_id)], starts=[RunTaskRequest])

    # Phase 2: RPC fan-out, no DB lock.
    results = self._provider.reconcile_workers(actions)

    # Phase 3: small write tx for status updates and start observations.
    with self._store.transaction() as cur:
        self._apply_reconcile_results(cur, results)
```

`reconcile_rows_for_workers`:

```sql
SELECT
  ta.worker_id, t.task_id, ta.attempt_id,
  t.state AS task_state, ta.state AS attempt_state,
  t.job_id
FROM tasks t
JOIN task_attempts ta
  ON ta.task_id = t.task_id AND ta.attempt_id = t.current_attempt_id
WHERE ta.worker_id IN (:worker_ids)
  AND t.state IN (TASK_STATE_ASSIGNED, TASK_STATE_BUILDING, TASK_STATE_RUNNING)
```

Rows with `task_state IN {BUILDING, RUNNING}` go into Poll `expected_tasks`. Rows with `task_state=ASSIGNED` produce start payloads and are **excluded** from `expected_tasks`. This avoids the Poll-before-start race; once `ASSIGNED` is no longer reported in `expected`, the 30s `_recent_submissions` grace window in `worker.py` becomes redundant and can be dropped together with the unified `Reconcile` RPC.

Every worker in the batch receives a reconcile action, including `Reconcile([], [])` for workers that should be empty — that is what kills strays.

### 4.2 RunTaskRequest cache

`RunTaskRequest` is heavy (entrypoint, environment, workdir files). Building it per-attempt per-tick on every reconcile pass would re-serialize the same job-level config repeatedly. Cache by `job_id`:

```python
@functools.lru_cache(maxsize=4096)
def _run_request_template(job_id_wire: str, snap_token: int) -> RunTaskRequestTemplate:
    """Build the shared (job-level) portion of RunTaskRequest. Per-attempt
    fields (task_id, attempt_id) are filled in at the call site.

    snap_token is invalidated when the job's config changes. Same-name
    replacement assigns a new job_id, so the cache key naturally rolls.
    """
    ...
```

The cache lives in the controller process and survives across ticks. Same-name replacement creates a new `job_id` (§6), so the cache key automatically rolls; no manual invalidation. Workdir files come from `job_workdir_files` and are stable for the life of a job.

Per-attempt fields (`task_id`, `attempt_id`) are stamped onto the cached template at fan-out time. Resources come from the job_config and are part of the cached template.

### 4.3 BUILDING transition

```sql
UPDATE task_attempts
SET state = TASK_STATE_BUILDING, started_at_ms = COALESCE(started_at_ms, :now)
WHERE task_id = :task_id AND attempt_id = :attempt_id AND state = TASK_STATE_ASSIGNED;

UPDATE tasks
SET state = TASK_STATE_BUILDING
WHERE task_id = :task_id AND current_attempt_id = :attempt_id AND state = TASK_STATE_ASSIGNED;
```

Driven by the worker's first BUILDING/RUNNING heartbeat for the attempt. The current attempt guard makes this idempotent and racy-safe against scheduler-rolled attempts.

### 4.4 Wake events

- `_scheduling_wake`: producing transitions that may free capacity (terminal heartbeats, attempt finalization).
- `_polling_wake`: producing transitions that may need a fresh reconcile pass (any write to `tasks.state` that affects `expected_tasks` — primarily new ASSIGNED, plus bulk state changes from cancellation).

The poll loop waits on `_polling_wake` with `POLLING_TICK_INTERVAL` timeout; `wait → clear → tick` order preserved. A wake can also push a worker to the front of the reconcile cursor so newly-assigned workers do not wait for a full rotation.

### 4.5 Heartbeat drain

`_drain_heartbeats` keeps its current shape. The state-transition table gains no new entries — RUNNING → KILLED on terminal heartbeat already exists.

The behavior change is in §5: producing transitions never stamp `task_attempts.finished_at_ms` for worker-bound attempts. The heartbeat path is the sole writer.

### 4.6 K8s direct-provider

Same shape, controller-side instead of worker-side:

```python
def _sync_direct_provider(self) -> None:
    if not isinstance(self._provider, K8sTaskProvider):
        return
    with self._db.read_snapshot() as snap:
        desired = self._store.attempts.list_active_direct_provider(snap)
        # tasks where current_worker_id IS NULL AND state IN ACTIVE
    pod_listing = self._provider.list_pods()
    actions = self._provider.diff(desired, pod_listing)
    # Pods in pod_listing but not in desired → DeletePod
    # Tasks in desired+ASSIGNED not in pod_listing → CreatePod
    self._provider.apply(actions)
    with self._store.transaction() as cur:
        for (task_id, attempt_id), kind in actions:
            if kind == "scheduled":
                self._store.attempts.mark_building_if_current(cur, task_id, attempt_id)
            elif kind == "deleted":
                self._store.attempts.finalize_from_worker(cur, task_id, attempt_id, TASK_STATE_KILLED, now_ms)
```

K8s reconcile is functionally identical to worker reconcile — start the pods that should exist, stop the pods that shouldn't, finalize attempts when pods exit. There is no buffered kill queue and no `dispatch_queue` rows. `buffer_direct_kill` is deleted; sites that called it (e.g. dispatch failure paths) instead update `tasks.state` directly, and the next sync diff picks up the change.

---

## 5. Producing transitions: single source of truth

### 5.1 The contract

Today's `_terminate_task` (transitions.py:353) is already the one helper that moves a task and its current attempt out of active state. The reconcile design keeps it as the single entry point, but makes the producer-vs-heartbeat distinction explicit:

```python
def _terminate_task(
    cur,
    attempts,
    tasks,
    workers,
    registry,
    task_id: str,
    attempt_id: int | None,
    state: int,
    error: str | None,
    now_ms: int,
    *,
    finalize_attempt: bool,            # see below
    attempt_state: int | None = None,
    failure_count: int | None = None,
    preemption_count: int | None = None,
) -> None:
    """Single source of truth for moving a task out of active state.

    Always:
      - tasks.state, error, exit_code, finished_at_ms updated
      - attempt's reporting state updated (PREEMPTED/KILLED/etc.)
      - endpoints deleted

    If finalize_attempt=True:
      - task_attempts.finished_at_ms = now_ms (resource release)

    Callers:
      finalize_attempt=True  -> heartbeat path, _remove_failed_worker
                                synthesis path, K8s pod-deleted observation
      finalize_attempt=False -> producing transitions: cancel_job,
                                cancel_tasks_for_timeout, preempt_task,
                                _requeue_coscheduled_siblings,
                                _terminate_coscheduled_siblings,
                                _kill_non_terminal_tasks
    """
```

`finalize_attempt` is not a stylistic flag; it directly controls whether the attempt's resources are released. Producers always pass `finalize_attempt=False` for worker-bound attempts. The heartbeat path and the worker-failure-synthesis path are the only `finalize_attempt=True` callers.

`workers.decommit_resources` and `workers.add_committed_resources` are removed; the scheduler derives usage from unfinished attempts (§6). `_terminate_task` no longer takes a `resources` parameter.

### 5.2 Call-site changes

| Caller | `finalize_attempt` |
|---|---|
| `_apply_task_transitions` (heartbeat) | `True` |
| `_remove_failed_worker` (synthesizes WORKER_FAILED for each non-terminal attempt on a dead worker) | `True` |
| K8s sync — pod-deleted observation | `True` |
| `cancel_job` / `_kill_non_terminal_tasks` / `cancel_tasks_for_timeout` | `False` |
| `preempt_task` | `False` |
| `_requeue_coscheduled_siblings` | `False` |
| `_terminate_coscheduled_siblings` | `False` |

Today's `_requeue_coscheduled_siblings` (transitions.py:592-642) and `_terminate_coscheduled_siblings` (transitions.py:548-589) both call `_terminate_task`, which today always stamps `finished_at_ms`. The change is the parameterization above.

`_remove_failed_worker` already has heartbeat-equivalent semantics (it synthesizes the worker's missing terminal heartbeats); it stays `finalize_attempt=True`.

---

## 6. Conservative scheduler state

```python
def _read_scheduling_state(self) -> SchedulingState:
    with self._db.read_snapshot() as snap:
        tasks = self._store.tasks.pending_for_scheduling(snap)
        workers = self._store.workers.list_active_healthy(snap)
        usage = self._store.attempts.resource_usage_by_worker(snap)
    return SchedulingState(tasks=tasks, workers=workers, usage=usage)
```

`available_R(W) = total_R(W) − usage[W][R]`. The scheduler does not read `workers.committed_*`; those columns are removed.

`resource_usage_by_worker`:

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

CPU/memory sum in SQL. Device counts are parsed in Python via a cached helper:

```python
class DeviceCounts(NamedTuple):
    gpu: int
    tpu: int

@functools.lru_cache(maxsize=8192)
def device_counts_from_json(device_json: str | None) -> DeviceCounts:
    if not device_json:
        return DeviceCounts(gpu=0, tpu=0)
    device = proto_from_json(device_json, job_pb2.DeviceConfig)
    return DeviceCounts(gpu=get_gpu_count(device), tpu=get_tpu_count(device))
```

If at scale the per-tick aggregate becomes a hot path, cache it in process for the life of one scheduler pass. Do not reintroduce transition-maintained durable counters.

### 6.1 Why #5470 holds

- Producing transition (preempt, cancel) writes `tasks.state=PREEMPTED|KILLED` and updates the attempt's reporting state. The old attempt row remains unfinished (`finished_at_ms IS NULL`).
- Scheduler derives usage from unfinished worker-bound attempts → `available = 0` on the still-busy worker → cannot double-book.
- Worker eventually heartbeats terminal → heartbeat tx finalizes the attempt → next scheduler tick can place new work.

The conservative property is independent of when reconcile fires. Even if reconcile is delayed by a tick, scheduler decisions are correct because they key on unfinished attempts, not on "kill RPC sent."

---

## 7. Job replacement

`jobs.job_id` is a `JobName`, which for root jobs is the user-facing name. CASCADE-deleting tasks while a worker still holds their containers destroys the `task_attempts` rows that the eventual heartbeat needs to find. The fix: block the launch RPC until the old job has drained.

`service.submit_job` for an existing job name, with `EXISTING_JOB_POLICY_RECREATE`:

```python
with self._store.transaction() as cur:
    self._transitions.cancel_job(cur, job_id, "Replaced by new submission")
    self._polling_wake.set()

self._wait_until_job_drained(job_id, timeout=Duration.from_minutes(10))

with self._store.transaction() as cur:
    self._transitions.remove_finished_job(cur, job_id)
    self._transitions.submit_job(cur, job_id, request, Timestamp.now())
```

Drain check (no write lock held):

```sql
SELECT 1
FROM tasks t
JOIN task_attempts ta ON ta.task_id = t.task_id
WHERE t.job_id = :job_id
  AND ta.worker_id IS NOT NULL
  AND ta.finished_at_ms IS NULL
LIMIT 1
```

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

The launch RPC blocks until the old job's worker-bound attempts are all finalized. This is the desired UX: clients wait until the new job is the live job. On timeout (worker-failure finalization is broken or stuck), return `DEADLINE_EXCEEDED`.

The blocking wait is also load-bearing for K8s — pods cannot be re-created with the same name until the prior generation has been observed deleted, and the scheduler cannot place the new job's tasks correctly until usage from the old job's attempts is gone.

---

## 8. Worker side

### 8.1 `Reconcile(expected_tasks, start_tasks)` semantics

Already implemented for the two halves separately:

- `_reconcile_expected_tasks` (worker.py:854-885): given `expected_tasks`, kills locally-running non-terminal tasks not in expected, returns status entries for expected tasks.
- `handle_start_tasks` (worker.py:942+): dedups on `(task_id, attempt_id)` via `_tasks` (worker.py:679-692).

The only worker-side change for v1 is conceptual: the controller stops including `ASSIGNED` attempts in `expected_tasks`. With that, the `_recent_submissions` grace window in worker.py becomes redundant once the unified `Reconcile` RPC lands and can be dropped.

### 8.2 No new worker RPCs

No `GetTaskPayload`, no per-task watchdog, no acceptance protocol. The reconcile action carries start payloads; reconcile is the redispatch.

---

## 9. Migration plan (delta from current main)

Each step independently shippable and reversible.

### Step 1 — `_terminate_task` parameterization
- Add `finalize_attempt: bool` to `_terminate_task`.
- Update call sites: producing transitions pass `False`; heartbeat path and `_remove_failed_worker` pass `True`.
- Behavior change: producing transitions stop stamping `task_attempts.finished_at_ms` for worker-bound attempts.
- Tests: existing transition replay tests should still pass. Add a regression for #5470: preempt + reassign on the same worker no longer sees freed capacity until heartbeat.

### Step 2 — Scheduler reads from `task_attempts`
- Add `TaskAttemptStore.resource_usage_by_worker()`.
- Add `device_counts_from_json` cache in `codec.py`.
- Replace scheduler reads of `workers.committed_*` with the derived usage.
- Drop `add_committed_resources` and `decommit_resources` calls (`_terminate_task` no longer takes a `resources` parameter).
- Schema migration drops `committed_cpu_millicores`, `committed_mem_bytes`, `committed_gpu`, `committed_tpu` from `workers`.

### Step 3 — Drain guard for delete paths
- `service.submit_job` `EXISTING_JOB_POLICY_RECREATE`: call `cancel_job`, commit, run `_wait_until_job_drained`, then `remove_finished_job`.
- Audit `remove_finished_job` callers (pruning, admin delete) and gate the same way.
- Reversible: revert the wait.

### Step 4 — Worker-batch reconcile
- Add `_reconcile_worker_batch`: one snapshot read of `attempts.reconcile_rows_for_workers` for the next batch of healthy workers; build `expected_tasks` and `start_tasks` actions; fan out via existing `Poll` + `StartTasks` provider methods; apply results in a small write tx.
- Add `_run_request_template` LRU cache.
- Drop scheduler-thread `_dispatch_assignments_direct` once the reconcile path covers ASSIGNED attempts. `queue_assignments` no longer fans out RPCs; it only writes state.
- Drop `_stop_tasks_direct` and the heartbeat-updater `tasks_to_kill` plumbing. Worker auto-kill is the only kill path.
- Drop `_poll_all_workers`; reconcile subsumes it.
- Reversible: re-enable old paths.

### Step 5 — K8s direct-provider on `tasks` / `task_attempts`
- `_sync_direct_provider` reads active null-worker rows from `tasks`/`task_attempts`, diffs against pod listing, applies pod creates/deletes.
- Drop K8s use of `dispatch_queue`: `queue_assignments(direct_dispatch=False)` no longer writes `enqueue_run`; `buffer_direct_kill` is deleted; sites that called it update `tasks.state` directly.

### Step 6 — Drop `dispatch_queue`
- `0044_drop_dispatch_queue.py`: `DROP TABLE dispatch_queue`.
- Delete `DispatchQueueStore`, `enqueue_run`, `enqueue_kill`, K8s drain helpers.

### Step 7 — Collapse `Poll` + `StartTasks` into `Reconcile`
- Single worker RPC carrying both fields.
- Drop `_recent_submissions` grace window in worker.py.

---

## 10. Walkthroughs

### 10.1 #5470: preempt-then-reassign

slice-3-w1 runs `gang-a-task-1` (attempt M), 8 chips. Higher-priority job preempts:

```python
# preempt_task
_terminate_task(..., state=PENDING, attempt_state=PREEMPTED, finalize_attempt=False)
# tasks: state=PENDING, current_worker_id=NULL
# task_attempts(M): state=PREEMPTED, finished_at_ms=NULL
```

Scheduler derives `used_tpu(slice-3-w1) = 8` from M's unfinished attempt. `available_tpu(slice-3-w1) = 0`. Cannot double-book.

slice-3-w1's next reconcile: M not in `E_W` (task is PENDING; current attempt is M but task not ACTIVE). Worker auto-kills.

Worker heartbeats KILLED. Heartbeat path:

```python
_terminate_task(..., state=KILLED, finalize_attempt=True)
# task_attempts(M): state=PREEMPTED preserved (controller-set is higher truth),
#                   finished_at_ms=now_ms
```

Next scheduler tick: derived usage on slice-3-w1 drops to 0. Can place new work.

### 10.2 Coscheduled gang kill

Gang `gang-b` has 4 members on W1..W4. User cancels:

```python
# cancel_job walks the parent_job_id subtree (existing recursive CTE)
# Each non-terminal task → _terminate_task(state=KILLED, finalize_attempt=False)
self._polling_wake.set()
```

Each of W1..W4's next reconcile excludes the killed task. Each worker auto-kills its container. Heartbeats trickle in independently. As each lands, heartbeat path stamps `finished_at_ms`; attempt drops out of usage.

### 10.3 Worker death

W3 misses N pings. Ping-loop terminate path runs `_remove_failed_worker(W3)`:

```python
for attempt in attempts_on_w3_active:
    _terminate_task(
        ..., state=WORKER_FAILED,
        finalize_attempt=True,  # heartbeat-equivalent synthesis
    )
    # task transitions per retry budget (back to PENDING or to terminal)
self._store.workers.remove(cur, W3)
```

`_remove_failed_worker` does **not** assign a new worker. It only marks the current attempt dead and lets the next scheduler tick pick up the now-PENDING task.

### 10.4 Job replacement

User submits `experiment-foo` with new code; existing `experiment-foo` has 50 tasks running:

```python
# service.submit_job EXISTING_JOB_POLICY_RECREATE
with transaction() as cur:
    cancel_job(cur, job_id, "Replaced by new submission")
    polling_wake.set()
_wait_until_job_drained(job_id, timeout=10min)  # blocks the launch RPC
with transaction() as cur:
    remove_finished_job(cur, job_id)
    submit_job(cur, job_id, request, now)
```

Workers see the 50 tasks excluded from `E_W` on their next reconcile, auto-kill, heartbeat KILLED. Heartbeat path finalizes attempts. Once drained, the new job inserts and is scheduled normally.

### 10.5 Controller restart mid-dispatch

Controller restarts while `tasks.state=ASSIGNED, current_attempt_id=N, current_worker_id=W`. Worker may already be running the container (start RPC landed before crash), or not.

After restart:
- Worker's reconcile batch fires. Reconcile sees ASSIGNED → sends start payload. Worker starts or dedups on `(t, N)`.
- If the worker had already heartbeated BUILDING/RUNNING before crash, that write committed; reconcile sees the new state and produces no start payload, just an `expected` entry.

Either way, convergence. Restart recovery is implicit in the reconcile loop.

---

## 11. Open questions

1. **K8s pod-creation failure semantics.** If `CreatePod` fails (image pull, namespace gone), what's the right state transition? Options: stay ASSIGNED (next sync re-fires); transition to FAILED; transition to UNSCHEDULABLE. Today's K8s sync has its own retry; reconcile inherits it.
2. **Replacement wait timeout.** Default 10 min is long enough for normal worker shutdown, finite enough that a stuck finalization doesn't hang launches forever. Should it be configurable per-launch?
3. **Worker-batch sizing.** `RECONCILE_WORKER_BATCH_SIZE = 512` at 4K workers and 250ms tick gives ~2s full-fleet rotation. Wake-triggered workers should be pushed into a priority lane so newly-assigned workers do not wait for full rotation.
4. **`reconcile_rows_for_workers` index.** Existing `idx_tasks_state_attempt` (schema.py:766) covers `(state, task_id, current_attempt_id, job_id)`. Add an index on `task_attempts(worker_id) WHERE finished_at_ms IS NULL` if the join shows up in scale tests.

---

## 12. Tunables

| Tunable | Default | Reasoning |
|---|---|---|
| `POLLING_TICK_INTERVAL` | 250ms | matches existing |
| `RECONCILE_WORKER_BATCH_SIZE` | 512 | bounds per-tick fan-out; ~2s full-fleet rotation at 4K workers |
| Replacement drain timeout | 10min | normal worker shutdown well under 1min; finite to surface stuck finalization |
| heartbeat-fail-threshold | unchanged | sole liveness mechanism for stuck-RPC recovery |

---

## 13. Test strategy

### Unit
- State-transition coverage: PENDING → ASSIGNED → BUILDING → RUNNING → terminal. Cancel from each non-terminal state goes directly to KILLED.
- `_terminate_task` parameterization: `finalize_attempt=True` stamps `finished_at_ms`; `finalize_attempt=False` does not.
- Heartbeat path: terminal advances `tasks.state` (if not already terminal) and stamps `task_attempts.finished_at_ms`. Idempotent on second heartbeat.
- Scheduler usage: unfinished worker-bound attempts contribute resources; finalized attempts do not.
- `device_counts_from_json` returns expected `DeviceCounts(gpu, tpu)` for empty, GPU, TPU, repeated values.

### Integration
- #5470 regression: preempt + reassign on the same worker; available capacity stays 0 until heartbeat.
- Three-phase RPC: writer lock available during reconcile fan-out.
- Single-snapshot scheduler read: ==1 read_snapshot per tick.
- Job replacement: launch RPC blocks until old job's tasks reach KILLED; new job's tasks proceed independently; no leak; old job pruned after drain.
- Delete/prune guard: terminal job with unfinished worker-bound attempt is not deleted.
- Controller restart with task in ASSIGNED: convergence to BUILDING on the next reconcile.
- Worker auto-kill via reconcile: insert "stray" task on worker, controller's expected set excludes it, next tick auto-kills.
- `RunTaskRequest` LRU cache: same `job_id` across many ASSIGNED attempts hits the cache; new `job_id` (via replacement) misses and rebuilds.

### End-to-end
- Replay-golden regeneration. Diff: producing transitions write `tasks.state` directly; `finished_at_ms` only stamped at heartbeat.
- 4K-worker scale: reconcile RPC volume per tick. Confirm tick duration < 250ms in steady state.
