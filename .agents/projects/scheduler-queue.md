# Iris Controller: SQL-Backed Dispatch Queue

The Iris controller currently dispatches StartTasks, StopTasks, Ping, and PollTasks RPCs from four disjoint loops (`_run_scheduling_loop`, `_run_ping_loop`, `_run_task_updater_loop`, plus the closed-PR-era `_run_kill_dispatcher_loop`). Resource accounting and RPC dispatch are decoupled across these loops in a way that lets the scheduler observe freed capacity before the corresponding kill RPCs land on workers, double-booking workers and producing two concurrent processes on the same TPU. Issue #5470 surfaced this; PR #5550 tried to patch it with `KillBuffer` / `KillRegistry` / `PendingKill` types and was closed because the abstraction was a Python-side reimplementation of what the controller actually needs — a transactional dispatch queue that the scheduler reads conservatively from.

This doc describes the redesign. It promotes the existing K8s-only `dispatch_queue` SQLite table to the single dispatch queue for *all* worker-directed RPCs, replaces the four loops with one staggered poll loop driving that queue, and moves resource decommit so it can never be observed before the worker has confirmed the task terminal.

## Goals

- **Close #5470 by construction, not by mirror.** The producing transition that decides "task X is dead on worker W" must not free W's capacity until the worker has confirmed the task is gone. Because the queue and the DB row are written in the *same SQLite transaction*, a scheduler tick between the producing transition's commit and the dispatch RPC sees both — committed resources still high *and* a pending stop. There is no in-memory mirror to fall out of sync with the DB.
- **Unify start, stop, and poll on a single dispatch path.** Every Controller→Worker side-effect (StartTasks, StopTasks, PollTasks, Ping) is driven by the poll loop. StartTasks and StopTasks are queue-driven; PollTasks/Ping are tick-driven. There is one place that turns intent into RPC, one place that handles failures, and one place that records observed worker state back into the DB.
- **Eliminate the post-commit-window class of bug.** `cur.on_commit(...)` hooks fire after the SQLite COMMIT returns. Anything that mutates an in-memory mirror in an `on_commit` hook is, by construction, observable to other threads later than the DB row. The new design puts the dispatch entry *inside* the writing transaction. There is no in-memory mirror that an `on_commit` hook would maintain.
- **Stay restart-safe.** Controller crashes are normal. The DB is the source of truth, including the queue. A controller restart re-reads the queue from SQL and resumes dispatching. No reconstruction-from-task-state is needed; the queue rows themselves persist. Idempotency on `(task_id, attempt_id)` covers the case where the worker already received an earlier identical RPC.
- **Do not depend on SQLite WAL behavior for dispatch correctness.** The current `_workers_pending_kill` set worked only because the read-snapshot transaction in the scheduler happened to start strictly after the writer's commit. The new design must be correct under any combination of WAL mode, snapshot timing, and reader/writer interleaving — i.e. not contain "this works because SQLite happens to serialize this way".

## Non-goals

- **Collapsing PollTasks/StartTasks/StopTasks into a single Poll RPC** is the long-term direction (Russell: "long term Poll() integrates both"). Out of scope for v1. The dispatch queue is the precondition for that change because it gives us a single point where we know everything pending for a worker, but the wire-protocol unification is a separate project.
- **Worker-side changes** are out of scope where avoidable. The one exception, flagged below, is `StartTasks` idempotency on `(task_id, attempt_id)`. If measurement shows the worker is already idempotent on those keys, no worker change is needed.
- **Per-task fairness across workers** is not changed. The scheduler still owns assignment policy. The queue is downstream of scheduling decisions.
- **Replacing the K8s direct-provider sync RPC.** K8s already uses `dispatch_queue` via `drain_for_direct_provider` / `buffer_direct_kill`; we keep that path. Step 4 (collapse) is not in this design.

## Background

Today the controller runs four loops in parallel against shared SQLite + in-memory state. The scheduling loop (`_run_scheduling_loop` at `controller.py:1473`) computes assignments and calls `_dispatch_assignments_direct` (`controller.py:2276`) which opens a transaction, calls `transitions.queue_assignments(direct_dispatch=True)` (`transitions.py:1231`), commits, then sends StartTasks RPCs out of band against the worker addresses captured in `start_requests`. The task-updater loop (`_run_task_updater_loop` at `controller.py:2476`) drains a `queue.Queue[HeartbeatApplyRequest]`, applies transitions in `_process_heartbeat_updates` (`controller.py:2450`), gets back a `tasks_to_kill` set with a parallel `task_kill_workers` map, and calls `_stop_tasks_direct` (`controller.py:2355`). The ping loop (`_run_ping_loop` at `controller.py:2384`) sends Ping RPCs and writes liveness verdicts. Inline at the end of every scheduling tick, `_poll_all_workers` (`controller.py:2429`) issues PollTasks. The K8s code path is a totally separate `_run_direct_provider_loop` (`controller.py:1575`) that uses the `dispatch_queue` SQLite table (schema at `schema.py:986`, store at `stores.py:1888`, drain at `transitions.py:2364`) to feed the K8s sync RPC.

The brittleness shows up at the seam between the producing transition and the StopTasks RPC. `_kill_non_terminal_tasks` (`transitions.py:412`) does the resource decommit (`workers.decommit_resources(...)` at `transitions.py:409` via `_terminate_task`) inside the same transaction as the task state change. The kill RPC is sent later, by another thread, after the transaction commits. Between commit and RPC send, the scheduling loop can read `worker.committed_*`, see freed capacity, and place a new task on a worker that still has the old gang's processes attached to its accelerators. PR #5550 tried to plug this by maintaining `_workers_pending_kill` (`controller.py:1245`) — a Python set populated under `_workers_pending_kill_lock` after the heartbeat transaction commits, drained after the StopTasks RPCs return. The reviewer correctly noted that there is still a microsecond window between `cur.commit()` returning (which makes the decommit visible to the next read snapshot) and the `_workers_pending_kill |= workers_to_kill` line under the lock. That window is small but it exists. There is also a duplication problem: K8s already has a buffered drain via `dispatch_queue` and `drain_for_direct_provider` (`transitions.py:2364`), and the new in-memory `_workers_pending_kill` in PR #5550 reimplemented part of that abstraction in Python. Two stores of "what RPCs do I owe a worker" are clearly worse than one.

## Proposed design

### Data model: extend the existing `dispatch_queue` table

`dispatch_queue` already exists (`schema.py:986`, migration `0011_direct_provider.py`). Today it carries:

```
id              INTEGER PRIMARY KEY AUTOINCREMENT
worker_id       TEXT NULLABLE FK -> workers(worker_id) ON DELETE CASCADE
kind            TEXT CHECK (kind IN ('run','kill'))
payload_proto   BLOB             -- RunTaskRequest for kind='run', NULL for kind='kill'
task_id         TEXT             -- wire ID; nullable in current schema, populated for both kinds going forward
created_at_ms   INTEGER NOT NULL
INDEX idx_dispatch_worker(worker_id, id)
```

We add four columns, a partial unique index, a covering index, and a CHECK constraint locking the two row shapes. Migration `0042_dispatch_queue_per_attempt.py`:

```sql
-- Two row shapes coexist; a CHECK enforces them so a future bug can't insert a hybrid:
--   "k8s"     : worker_id IS NULL  AND kind='kill' AND attempt_id IS NULL  -- legacy direct-provider sync
--   "worker"  : worker_id NOT NULL AND attempt_id NOT NULL                  -- new path
-- We add the columns first, then rebuild the table to attach the CHECK (sqlite cannot
-- ADD CONSTRAINT in-place). The rebuild also lets us drop legacy worker-typed rows
-- that were never per-attempt-keyed under the old schema.

ALTER TABLE dispatch_queue ADD COLUMN attempt_id INTEGER;        -- NULL only for k8s rows
ALTER TABLE dispatch_queue ADD COLUMN dispatched_at_ms INTEGER;  -- NULL = not yet dispatched
ALTER TABLE dispatch_queue ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0;
ALTER TABLE dispatch_queue ADD COLUMN last_error TEXT;

-- Scrub stale rows that predate per-attempt keying.
-- Worker-typed rows under the old schema have no attempt_id and cannot be safely
-- redispatched (we don't know which attempt they refer to). Drop them. K8s rows
-- (worker_id IS NULL) survive untouched.
DELETE FROM dispatch_queue WHERE worker_id IS NOT NULL AND attempt_id IS NULL;

-- Rebuild to attach the row-shape CHECK. The CHECK is the load-bearing invariant
-- preventing future hybrids; without it we'd rely on convention.
CREATE TABLE dispatch_queue_v42 (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    worker_id         TEXT REFERENCES workers(worker_id) ON DELETE CASCADE,
    kind              TEXT NOT NULL CHECK (kind IN ('run','kill')),
    payload_proto     BLOB,
    task_id           TEXT,
    attempt_id        INTEGER,
    created_at_ms     INTEGER NOT NULL,
    dispatched_at_ms  INTEGER,
    attempts          INTEGER NOT NULL DEFAULT 0,
    last_error        TEXT,
    CHECK (
      (worker_id IS NULL  AND attempt_id IS NULL  AND kind='kill') OR
      (worker_id IS NOT NULL AND attempt_id IS NOT NULL)
    )
);
INSERT INTO dispatch_queue_v42(id, worker_id, kind, payload_proto, task_id, attempt_id,
                               created_at_ms, dispatched_at_ms, attempts, last_error)
  SELECT id, worker_id, kind, payload_proto, task_id, attempt_id,
         created_at_ms, dispatched_at_ms, attempts, last_error
  FROM dispatch_queue;
DROP TABLE dispatch_queue;
ALTER TABLE dispatch_queue_v42 RENAME TO dispatch_queue;

-- Partial unique index for the worker-typed shape only.
CREATE UNIQUE INDEX IF NOT EXISTS idx_dispatch_unique
  ON dispatch_queue(worker_id, kind, task_id, attempt_id)
  WHERE worker_id IS NOT NULL;

-- Pending-rows scan for the poll loop: one global SELECT per tick (see Drain semantics).
CREATE INDEX IF NOT EXISTS idx_dispatch_pending
  ON dispatch_queue(dispatched_at_ms, kind, worker_id);

-- Per-worker fast lookup retained for the kill-cascade and worker-removal paths.
CREATE INDEX IF NOT EXISTS idx_dispatch_worker
  ON dispatch_queue(worker_id, id);
```

Per Iris convention (`schema.py:507-571`, `python_name=` everywhere), the SQL columns keep the `_ms` suffix but the Python `Column(...)` definitions in `schema.DISPATCH_QUEUE` use `python_name="dispatched_at"`, `python_type=Timestamp | None`, `decoder=_nullable(decode_timestamp_ms)`. Same pattern as `created_at_ms` already on this table.

Justifications:

- `attempt_id` is required because a task that has been requeued after a worker failure can have a stale `kind='run'` row for an old attempt and a fresh `kind='run'` row for the new one. Treating them as the same row is a bug. K8s rows have `worker_id IS NULL` and `attempt_id IS NULL`; the CHECK enforces this and the partial unique index excludes them.
- `dispatched_at_ms` separates "queued, not yet sent" from "sent, awaiting worker confirmation." The poll loop sets it on RPC ack; the row is *not* deleted at that point — see Drain semantics.
- `attempts` and `last_error` carry retry/backoff state that today lives in the `_workers_pending_kill` lock plus log_event calls. Durable across crashes.
- `idx_dispatch_pending` covers the *single global* scan that drives the poll-loop tick: `SELECT ... WHERE dispatched_at_ms IS NULL OR dispatched_at_ms < ?`. We do **not** scan per-worker per tick — see Load model.
- The CHECK is non-negotiable: with two shapes living in one table, the FK / unique-index machinery alone is not enough to keep a future bug from inserting a `worker_id NOT NULL, attempt_id NULL` row that the partial unique index would silently allow.

### `DispatchQueueStore` extensions

Today's store (`stores.py:1888`) has `enqueue_run`, `enqueue_kill` (K8s flavor with NULL worker_id), and `drain_direct_kills`. We add:

```python
class DispatchQueueStore:
    def enqueue_start(self, cur, worker_id: WorkerId, task_id: JobName, attempt_id: int,
                      payload_proto: bytes, now_ms: int) -> None:
        cur.execute(
            "INSERT OR IGNORE INTO dispatch_queue("
            "  worker_id, kind, payload_proto, task_id, attempt_id, created_at_ms"
            ") VALUES (?, 'run', ?, ?, ?, ?)",
            (str(worker_id), payload_proto, task_id.to_wire(), attempt_id, now_ms),
        )

    def enqueue_stop(self, cur, worker_id: WorkerId, task_id: JobName, attempt_id: int, now_ms: int) -> None:
        cur.execute(
            "INSERT OR IGNORE INTO dispatch_queue("
            "  worker_id, kind, payload_proto, task_id, attempt_id, created_at_ms"
            ") VALUES (?, 'kill', NULL, ?, ?, ?)",
            (str(worker_id), task_id.to_wire(), attempt_id, now_ms),
        )

    def list_pending_for_worker(self, cur, worker_id: WorkerId, *, limit: int) -> list[DispatchRow]: ...
    def list_pending_stops_by_worker(self, cur) -> dict[WorkerId, list[PendingStop]]: ...
    def mark_dispatched(self, cur, row_id: int, now_ms: int) -> None: ...
    def mark_dispatch_failed(self, cur, row_id: int, error: str) -> None: ...   # bumps attempts, sets last_error
    def delete(self, cur, row_id: int) -> None: ...
    def delete_for_attempt(self, cur, worker_id: WorkerId, task_id: JobName, attempt_id: int) -> None: ...
    def delete_all_for_worker(self, cur, worker_id: WorkerId) -> None: ...
```

`PendingStop` is `(worker_id, task_id, attempt_id, row_id, dispatched_at_ms, attempts)`. The K8s methods (`enqueue_run`, `enqueue_kill`, `drain_direct_kills`) stay; they continue to use `worker_id IS NULL`. A worker-typed `enqueue_start`/`enqueue_stop` is a strictly different shape (worker_id non-null, attempt_id non-null, partial unique index).

### Conservative scheduler state

This is the load-bearing invariant. The scheduler computes available capacity per worker as:

```
available_X(worker) = total_X(worker)
                    - committed_X(worker, from DB row)
                    - sum of resources for pending stops in dispatch_queue[worker]
```

Pending starts are *not* subtracted again — `queue_assignments` (`transitions.py:1285`) calls `add_committed_resources` in the same transaction as `enqueue_start`, and the row exists alongside the increased `committed_*`. Pending stops *are* subtracted because the corresponding decommit has not yet happened: in this design, `_terminate_task` (`transitions.py:408-409`), `cancel_job` (`transitions.py:1098-1100`), `_kill_non_terminal_tasks`, `_terminate_coscheduled_siblings`, `_requeue_coscheduled_siblings`, `cancel_tasks_for_timeout` (`transitions.py:2027-2046`), `preempt_task` (`transitions.py:1894-1911`), and `_remove_failed_worker` all stop decommitting. Decommit moves to the heartbeat path (`transitions.py:1500-1508`, unchanged).

`_enforce_execution_timeouts` (`controller.py:2198`) does not decommit on its own — it routes through `cancel_tasks_for_timeout` which calls `_terminate_task` with worker_id+resources. Once `_terminate_task` no longer decommits, this path is automatically fixed; there is no second decommit codepath to chase. (Iteration-1 flagged this as TBD; closed.)

#### How `pending_stop_resources` is computed

The query joins `dispatch_queue` to `tasks` on `task_id` (and validates `attempt_id` matches the row, see "Stop-after-reassignment"), then resolves resources from `job_config` — resources are immutable across attempts of the same task (they come from the job's `ResourceSpec`), so the join target does not depend on which attempt the row references:

```sql
SELECT q.worker_id,
       jc.res_cpu_millicores,
       jc.res_memory_bytes,
       jc.res_disk_bytes,
       jc.res_device_json
FROM dispatch_queue q
JOIN tasks t      ON t.task_id = q.task_id
JOIN jobs  j      ON j.job_id  = t.job_id
JOIN job_config jc ON jc.job_id = j.job_id
WHERE q.kind = 'kill' AND q.worker_id IS NOT NULL;
```

The Python side aggregates by `worker_id` into a `ResourceSpecProto` per worker. Reservation-holder tasks contribute nothing because they never enqueue stops (see "Reservation tasks").

The scheduler's `create_scheduling_context` (`scheduler.py:740`) takes a list of `WorkerSnapshot`. We add a new keyword:

```python
context = self._scheduler.create_scheduling_context(
    workers,
    building_counts=building_counts,
    pending_stop_resources=pending_stops,  # dict[WorkerId, ResourceSpecProto]
    pending_tasks=order.ordered_task_ids,
    jobs=modified_jobs,
)
```

`WorkerCapacity.from_worker` (`scheduler.py:175`) gains a `pending_stop_resources` argument and subtracts before producing `available_*`. The read snapshot used to compute `pending_stops` is the same `read_snapshot()` block that reads workers/tasks — one consistent view. Because the producing transition writes the queue row and the state change in the same SQLite transaction, the scheduler cannot see "task killed in DB, no queue row, committed unchanged" or any other intermediate state.

#### #5470 walkthrough under the new design

The reproducer (`test_5470_preemption_reassignment.py:236`) runs:

1. Two coscheduled gangs A and B running on `slice-1` and `slice-2`. Two extra workers `slice-3-w0..7` are healthy with no committed resources.
2. Both old slices are marked unhealthy. Gang A is reassigned to slice-3, transitioned to BUILDING.
3. One slice-3 worker reports `WORKER_FAILED`. This drives `_apply_task_transitions` → `_requeue_coscheduled_siblings` (`transitions.py:591`) on all 8 slice-3 workers.

Old behavior: the requeue calls `_terminate_task(... worker_id=W, resources=R)` which decommits. The transaction commits. Gang B's next scheduling tick observes `committed_tpu == 0` on slice-3 and places gang B there before the StopTasks RPCs land — the test asserts this is the bug.

New behavior:

- Inside the requeue transaction: `_terminate_task` no longer decommits. Instead, `_requeue_coscheduled_siblings` calls `dispatch.enqueue_stop(cur, w, task_id, attempt_id, now_ms)` for each affected `(worker, task, attempt)` triple. Both writes are part of the same atomic transaction.
- The transaction commits. Slice-3 workers' DB row still shows `committed_tpu = CHIPS_PER_VM`. The dispatch queue contains 8 kill rows.
- Gang B's scheduling tick opens a read snapshot. It reads workers (committed still high) and pending stops (8 rows, one per slice-3 worker). `WorkerCapacity.from_worker` subtracts pending-stop resources from available capacity → all slice-3 workers report `available_tpu = 0`. Gang B is not placed.
- Poll loop tick. It reads the kill rows, sends StopTasks RPCs to slice-3 workers, marks `dispatched_at_ms`.
- Workers ack StopTasks and eventually heartbeat the tasks as terminal. `apply_task_updates` decommits and deletes the kill row in one transaction (`transitions.py:1500-1508` plus `dispatch.delete_for_attempt`).
- Next scheduling tick: committed_* is now 0, queue row is gone, gang B can place on slice-3.

The window where committed-was-low-but-process-still-alive is gone. There is no Python set involved.

Reservation-holder tasks are unaffected: they never commit resources on assignment (`transitions.py:1278`), so they have nothing to decommit, and we do not enqueue stops for them (see "Reservation tasks").

### Stop RPCs key off DB state, not queue payload

User direction: *"stop RPCs are all keyed off of the DB state directly."* The right reading of "DB state" is: **the row identifies *which attempt on which worker* to stop; the worker address and the resource spec are looked up from the live DB at dispatch time, not from a payload serialized at enqueue.** The row carries `(worker_id, task_id, attempt_id)` because those three together name the historical attempt that needs to die — not the current attempt the scheduler may have since rolled to a new worker.

Concretely:

- `kind='kill'` carries no payload (`payload_proto IS NULL`). Identity columns: `(worker_id, task_id, attempt_id)`.
- The poll loop reads the kill row, looks up `workers.address` for `q.worker_id`, and sends `StopTasks((task_id, attempt_id))` to that address.
- We do **not** consult `tasks.current_worker_id` to decide where to send the stop. That would be wrong: when a task has been requeued, `tasks.current_worker_id` may be `NULL` or a *new* worker; the *old* worker's process is exactly what we need to kill, and the queue row keyed by `q.worker_id` is the durable record of who owns the kill.

#### Stop-after-reassignment

The case iteration-1 left open: kill is enqueued for `(W, T, N)`, then before the poll loop drains it, T's heartbeat reports terminal-on-W and the requeue + reassignment promotes T to attempt N+1 on a different worker W'. The kill row says `(W, T, N)`. With per-attempt keying:

- The worker `W` has already reported `(T, N)` terminal in the heartbeat. `apply_task_updates` decommitted resources for W and called `dispatch.delete_for_attempt(cur, W, T, N)` in the same transaction. The row is gone before any redispatch.
- If the heartbeat lands *between* the poll loop's read snapshot and its dispatch, the StopTasks RPC for `(T, N)` still fires against W. The worker is idempotent: stopping an already-terminal `(T, N)` is a no-op (the wire ID `(task_id, attempt_id)` is unique, and the worker has already cleaned up that attempt). This is harmless duplicate work.
- W' running attempt `N+1` is not affected; the kill row's `attempt_id=N` does not match the new attempt, and we never sent W' anything.

This is why the row carries `attempt_id`. Without it we'd have to choose between (a) sending `StopTasks(T)` to W and risking the worker honoring it against an unrelated future attempt of T it gets reassigned to (the worker cannot tell which generation of T we mean), or (b) silently dropping kills when `tasks.current_worker_id != W`, which is exactly the bug — the old W process is still alive but we no longer tell W to kill it.

A `kind='run'` row is different: by the time it dispatches, the assignment is fixed and the proto is non-trivial (Entrypoint + workdir files + bundle id + resources + constraints — see `transitions.queue_assignments` at `transitions.py:1290`). Reconstructing from DB would require re-reading job_config, workdir files, and constraint JSON each tick. The `payload_proto` BLOB is small (a few KB at most for typical jobs), and we already serialize it once at enqueue time. **Decision:** `kind='run'` carries the serialized `RunTaskRequest`; `kind='kill'` carries no payload, identity only.

### Idempotency: duplicate enqueues

Coscheduled-sibling cascades, double cancels, and chained transitions can each independently want to enqueue a stop for the same `(worker_id, task_id, attempt_id)`. We rely on the partial unique index plus `INSERT OR IGNORE` (see `enqueue_start`/`enqueue_stop` above). This is cheaper than a SELECT-then-INSERT, atomic without an extra round trip, and self-documenting: "the queue cannot hold two of the same identity."

For `kind='run'`: the `(task_id, attempt_id)` jumps on each retry (`attempt_id += 1` at `transitions.py:1268`), so duplicate-enqueue in a single transaction is genuinely a programming error. `INSERT OR IGNORE` makes it a no-op rather than a hard crash, and we add an assertion-level log when it fires (the `last_inserted_rowid` stays the same on IGNORE; we check `cur.rowcount == 0`).

For `kind='kill'`: the `(task_id, attempt_id)` is the live attempt at the moment of the kill cascade. A second cascade that reaches the same task will try to insert the same row; IGNORE makes it idempotent. If a *new* attempt has been issued in between (shouldn't happen — kill cascades are driven from terminal-attempt states — but defense in depth), the `attempt_id` will differ and a new row is correctly inserted.

A `kind='kill'` row arriving while a `kind='run'` row for the same `(task_id, attempt_id)` is still queued is allowed: they are separate rows distinguished by `kind` in the unique index. The poll loop will dispatch the run, the worker will accept it, then the next tick sees the kill and dispatches that. Worker-side StopTasks for an unknown task is a no-op today, and StopTasks for a just-started task is exactly what we want. We considered "kill replaces run in queue" semantics; rejected because it requires SELECT-DELETE-INSERT logic to handle correctly and the run-then-kill ordering is what already happens in normal operation.

### Lifecycle of a kill

```
service.terminate_job(...)
  -> transitions.cancel_job(cur, job_id, reason)
       within the transaction:
         - mark tasks KILLED, jobs KILLED in DB
         - DO NOT decommit resources
         - for each (task_id, attempt_id, worker_id):
             store.dispatch.enqueue_stop(cur, worker_id, task_id, attempt_id, now_ms)
       commit
poll loop tick (drain phase):
  for each healthy worker w with worker_id present in dispatch_queue:
    rows = dispatch.list_pending_stops_for_worker(snap, w.worker_id)
    if rows:
      stop_ids = [(r.task_id, r.attempt_id) for r in rows]
      provider.stop_tasks(w.worker_id, w.address, stop_ids)
      # On RPC success: mark dispatched_at_ms = now_ms.
      # The row stays in the queue until the worker confirms terminal.
worker eventually heartbeats with TASK_STATE_KILLED for those (task_id, attempt_id):
  service.update_task_status(...) -> task_update_queue.put(HeartbeatApplyRequest(...))
  task-updater drain (now part of the poll loop) calls apply_task_updates inside a tx:
    - decommit_resources for the worker
    - dispatch.delete_for_attempt(cur, worker_id, task_id, attempt_id)
  commit
scheduler tick after this:
  worker.committed_X is lower; queue row is gone; available capacity is correct.
```

If the worker is dead and never heartbeats, the worker-failure path runs (Worker death, below) and `WorkerStore.remove` cascades the queue row (it already does, `stores.py:1884`).

### Lifecycle of an assignment

```
scheduling loop tick:
  pending_stops = store.dispatch.list_pending_stops_by_worker(snap)
  context = scheduler.create_scheduling_context(workers, ..., pending_stop_resources=pending_stops)
  -> transitions.queue_assignments(cur, assignments)   # direct_dispatch=False, always
       within the transaction:
         - tasks.assign(...)                                                # ASSIGNED, attempt_id += 1
         - workers.add_committed_resources(...)
         - store.dispatch.enqueue_start(cur, worker_id, task_id, attempt_id,
             run_request.SerializeToString(), now_ms)
       commit
  controller wakes the poll loop via self._wake_event.set()

poll loop tick (drain phase):
  for each healthy worker:
    rows = dispatch.list_pending_for_worker(snap, w.worker_id, limit=BATCH)
    starts = [r for r in rows if r.kind == 'run']
    if starts:
      payloads = [parse(RunTaskRequest, r.payload_proto) for r in starts]
      acks, error = provider.start_tasks(w.worker_id, w.address, payloads)
      for ack:
        if ack.accepted:
          # mark dispatched_at_ms = now_ms, but keep the row until RUNNING heartbeat
          dispatch.mark_dispatched(cur, ack_row_id, now_ms)
        else:
          # synthesize WORKER_FAILED; the transition will delete the queue row.
          enqueue_task_update(WORKER_FAILED for (task_id, attempt_id))
      on transport error:
        dispatch.mark_dispatch_failed(cur, ack_row_id, error)  # bumps attempts; row stays

worker heartbeats RUNNING for (task_id, attempt_id):
  apply_task_updates:
    - update task state RUNNING (no decommit; resources already committed)
    - dispatch.delete_for_attempt(cur, worker_id, task_id, attempt_id)
```

The run row stays past `dispatched_at_ms` being set until the worker confirms RUNNING. Restart correctness: a controller crash between StartTasks ack and the first RUNNING heartbeat must re-issue the start. The queue row survives the crash; on restart the poll loop sees `dispatched_at_ms IS NOT NULL` but the worker has not yet reported RUNNING — we redispatch on a re-dispatch interval (see Drain semantics below). The worker's idempotency on `(task_id, attempt_id)` makes this safe.

Conservative-state confirmation for the gap between commit and dispatch: the start row contributes to `committed_*` already (we ran `add_committed_resources` in the transition), so the scheduler does not double-count. We do *not* subtract pending starts in the conservative pass.

### Drain semantics: when does a row die?

Two design questions, asked separately:

**1. Do we delete a stop row on RPC ack, or on terminal heartbeat?**

The user's direction is "stop RPCs are all keyed off DB state directly," which I read as: keep the row until the DB row says terminal. That gives us an automatic redispatch cycle: every kill-tick re-evaluates "is this task still alive on this worker?" and sends a stop if so. The risk is repeated stop RPCs to a healthy worker that has already accepted the stop but not yet flushed the kill to its container runtime.

**Decision (current):** Delete the queue row in the *heartbeat path* (`apply_task_updates` deletes the row when the worker reports the task terminal). The poll loop, between dispatch and terminal heartbeat, gates redispatch on a `redispatch_interval` (default 5s). A row with `dispatched_at_ms IS NOT NULL AND now_ms - dispatched_at_ms > redispatch_interval_ms` is redispatched; one with `dispatched_at_ms IS NOT NULL AND <= redispatch_interval_ms` is skipped. A row with `dispatched_at_ms IS NULL` is dispatched immediately.

This composes cleanly:
- Happy path: tick T sends stop, tick T+heartbeat (≈1s) gets terminal, queue row deleted. No redispatch.
- Slow worker: tick T sends stop, tick T+5s redispatches. Worker receives a duplicate stop RPC; idempotent.
- Wedged worker: redispatches every 5s until ping-threshold worker failure trips and `_remove_failed_worker` cascades the queue clear via `WorkerStore.remove`.

We add `attempts++` only on transport failure, not on redispatch. Poison-pill threshold is on `attempts` (sustained transport failure), not on redispatch count.

Counterpoint considered: delete on ack, re-enqueue from a heartbeat-watchdog if no terminal heartbeat within K ticks. Rejected: that puts the watchdog in a different place from the dispatch loop, and the row already encodes everything the watchdog needs (`dispatched_at_ms`, `attempts`). Keeping the row alive until DB terminal means there is one place to look.

**2. Do we delete a start row on RPC ack, or on RUNNING heartbeat?**

Same logic: keep until RUNNING (the heartbeat path deletes it). Redispatch on `dispatched_at_ms + redispatch_interval`. A start that was acked but never produced a RUNNING heartbeat (worker crashed mid-spawn) gets redispatched until the ping/heartbeat loop fails the worker, at which point the row goes via `WorkerStore.remove`. This relies on `(task_id, attempt_id)` start idempotency on the worker — same precondition as before.

### Staggered tick: how often does the poll loop wake?

The poll loop replaces `_run_ping_loop`, `_run_task_updater_loop`, and `_poll_all_workers`. It wakes frequently for low dispatch latency, but only does the heavy fan-out occasionally.

```python
@dataclass
class PollLoopConfig:
    tick_interval: Duration = Duration.from_ms(250)
    poll_every_n_ticks: int = 4   # full PollTasks fan-out cadence: 4 * 250ms = 1s
    ping_every_n_ticks: int = 4   # liveness Ping cadence: 1s (shares cadence with poll)
    redispatch_interval: Duration = Duration.from_ms(5000)
    dispatch_concurrency: int = 64
    dispatch_timeout: Duration = Duration.from_seconds(5.0)
    dispatch_batch_total: int = 4096  # cap rows per tick globally, not per worker
```

Defaults — to-measure markers explicit, no longer pure intuition:

- **`tick_interval=250ms`.** Balances kill-window latency (kill enqueued at tail of tick T fires by T+250ms) against SQL read load. With a *single* covered SELECT per tick (see Load model below) the cost is bounded and independent of worker count. **To-measure-during-step-2**: confirm the global SELECT stays under ~1ms at 4K workers under steady-state queue depth.
- **`poll_every_n_ticks=4` (≈1s).** Today's scheduler polls inline at the end of every scheduling tick gated by `poll_interval`. PollTasks fan-out is the actual cost driver. Match the existing 1s cadence so we are not regressing. Operator can tune `poll_interval` to dial it down for very large clusters; the staggered tick respects that.
- **`ping_every_n_ticks=4` (≈1s).** Matches today's `heartbeat_interval`. Cheap, no signal that 1s is too slow.
- **`redispatch_interval=5s`.** Long enough that a healthy worker's next heartbeat lands before we redispatch (heartbeat is ~1s, plus jitter); short enough that a wedged worker is reattempted within one ping-threshold window. **To-measure-during-step-2**: percentage of stop rows redispatched at all in steady state. If >5% of stop rows redispatch, raise the interval.
- **`dispatch_concurrency=64`.** Matches today's `profile_concurrency` and the K8s direct-provider sync. **To-measure-during-step-3**: at 4K workers with a 256-replica gang requeue, fan-out should complete in ≤2 ticks.
- **`dispatch_batch_total=4096`.** Global cap so a degenerate state (many thousands of pending rows after a long outage) doesn't block a tick. Per-worker fairness comes from `ORDER BY q.id` plus the round-trip — a worker with thousands of rows still gets ≤ batch_total / N_workers per tick.

#### Load model (4K workers)

The previous design implied per-worker SELECTs. That doesn't scale: 250ms × 4K = 16K SELECTs/sec just for dispatch reads, on top of writes. Switch to **one global SELECT per tick**, then partition in Python:

- **Reads.** One `SELECT id, worker_id, kind, task_id, attempt_id, payload_proto, dispatched_at_ms, attempts FROM dispatch_queue WHERE dispatched_at_ms IS NULL OR dispatched_at_ms < ? ORDER BY id LIMIT ?` per tick = 4 SELECTs/sec total. Covered by `idx_dispatch_pending`. At realistic queue depth (low hundreds in steady state, low thousands after a cascade) this is sub-millisecond.
- **Writes from poll loop.** `mark_dispatched` UPDATE per dispatched row. In steady state (no cascade) this is roughly the per-worker StartTasks/StopTasks rate — order-of-magnitude tens of writes/sec across the cluster.
- **Writes from heartbeat.** `apply_task_updates` already runs one transaction per heartbeat batch. Adding `dispatch.delete_for_attempt` for terminal updates is one extra DELETE per terminal task — same order as the existing `mark_terminal` UPDATEs.
- **Writes from transitions.** Producing transitions add 1 INSERT per affected `(worker, task, attempt)`. A 256-replica gang requeue adds 256 INSERTs in one transaction; the writer lock holds for the duration of that one transaction (already true today, since the cascade does 256 row mutations regardless).

Net: one new global SELECT per 250ms tick, ~1 INSERT per transition mutation, ~1 DELETE per terminal heartbeat, ~1 UPDATE per dispatched row. No per-worker fan-out at the SELECT level. SQLite WAL writer-lock contention does not get materially worse than today.

#### Tick cycle

The tick wakes via `self._wake_event.wait(timeout=tick_interval)`. Producing transitions call `self._wake_event.set()` post-commit so the next tick fires immediately.

```python
def _polling_tick(self, tick_count: int) -> None:
    # 1. Drain dispatch queue (every tick).
    self._drain_dispatch_queue()

    # 2. Heartbeat-derived task updates (every tick).
    requests = _drain_queue(self._task_update_queue, timeout=0)
    if requests:
        self._process_heartbeat_updates(requests)

    # 3. Liveness ping fan-out (every N ticks).
    if tick_count % self._config.ping_every_n_ticks == 0:
        self._ping_all_workers()

    # 4. PollTasks reconciliation fan-out (every N ticks).
    if tick_count % self._config.poll_every_n_ticks == 0:
        self._poll_all_workers()

def _drain_dispatch_queue(self) -> None:
    now_ms = Timestamp.now().epoch_ms()
    cutoff = now_ms - self._config.redispatch_interval.to_ms()
    with self._db.read_snapshot() as snap:
        rows = self._store.dispatch.list_pending(
            snap, cutoff_ms=cutoff, limit=self._config.dispatch_batch_total
        )
        addresses = self._store.workers.healthy_addresses(snap)

    by_worker: dict[WorkerId, list[DispatchRow]] = {}
    for r in rows:
        addr = addresses.get(r.worker_id)
        if addr is None:
            continue  # worker removal CASCADE will clean the row up
        by_worker.setdefault(r.worker_id, []).append(r)

    # RPC fan-out (no DB tx held).
    with ThreadPoolExecutor(max_workers=self._config.dispatch_concurrency) as pool:
        futures = [pool.submit(self._dispatch_one_worker, w, addresses[w], rs)
                   for w, rs in by_worker.items()]
        outcomes = [f.result(timeout=self._config.dispatch_timeout.to_seconds()) for f in futures]

    # Single transaction to apply all mark_dispatched / mark_dispatch_failed updates.
    with self._store.transaction() as cur:
        for outcome in outcomes:
            for u in outcome.updates:
                if u.kind == "ack":
                    self._store.dispatch.mark_dispatched(cur, u.row_id, now_ms)
                elif u.kind == "transport_fail":
                    self._store.dispatch.mark_dispatch_failed(cur, u.row_id, u.error)
                elif u.kind == "reject":
                    # Worker said "no". Synthesize WORKER_FAILED via the heartbeat queue;
                    # apply_task_updates will roll the task to PENDING and delete the row.
                    self._task_update_queue.put(u.synthesized_heartbeat)
```

### DB is the only source of truth (including the queue)

Since the queue lives in SQLite with FK CASCADE on workers, two failure modes that previously needed reasoning vanish:

1. *Controller crashes after committing a queue row but before sending the RPC.* The row survives. The next poll-loop tick after restart picks it up. If the worker had already received the RPC (impossible here — we crashed before sending), the worker's idempotency on `(task_id, attempt_id)` would absorb the duplicate. If it had not, we send it now. Either way, the controller's view converges with the worker's via the next heartbeat.

2. *The producing transition committed a queue row but the DB state and the queue row diverged across a crash.* This cannot happen — the row and the state change are in the same transaction. There is no "row committed, state not committed" intermediate. SQLite's transactional guarantees give us the invariant.

There is one remaining edge case: a queue row that points to a `(task_id, attempt_id)` no longer matching the live attempt. Concretely: a stop is enqueued for `(T, A)`, then the task is requeued via WORKER_FAILED → PENDING, the next scheduling tick assigns it to a different worker as `(T, A+1)`, and the original worker is still healthy. The original worker still has the original attempt running (the worker hasn't told us it's terminal yet, because we never told the worker), and we're now leaving a stop in the queue keyed at `(T, A)` while the live attempt is `(T, A+1)`. The DB-keyed dispatch (Stop RPCs key off DB state) handles this correctly: we read `tasks.current_worker_id` and only emit StopTasks if the original worker still owns the original attempt. **But:** rolling a task ASSIGNED→PENDING without first stopping the worker is exactly what produces #5470. The fix is conservative scheduler state; the queue row must remain (so the scheduler counts it) until the worker confirms terminal, even if the DB has a fresh attempt sitting in PENDING. This works because `apply_task_updates` only deletes the stop row on a terminal heartbeat for *this* attempt, not on a fresh assignment.

### Worker death

`WorkerStore.remove` already does `DELETE FROM dispatch_queue WHERE worker_id = ?` (`stores.py:1884`) — verified. The CASCADE FK from `dispatch_queue.worker_id` would also handle this if we relied on `DELETE FROM workers`, but the explicit DELETE in `remove` runs first; either way the queue is wiped for the dead worker. The cascade through `_remove_failed_worker` already requeues affected tasks (PENDING) or marks them WORKER_FAILED, and the new tasks-to-kill that come out of the cascade are coscheduled siblings on *other* workers, which go through the normal `enqueue_stop` path inside the same transaction.

If worker death is detected mid-tick — the poll loop is dispatching a StopTasks to a worker the ping path simultaneously decided to fail — the SQLite write-lock serialization protects us. The fail_workers transition deletes the queue rows; the poll loop's RPC dispatch is outside any DB transaction. The worst case is we successfully send a StopTasks to a worker we are about to forget about. The ack lands in our handler, the row no longer exists, `mark_dispatched` is a no-op (UPDATE WHERE id=? matches zero rows). No correctness violation.

### Cancelled-task-terminal-heartbeat edge case

A task that is KILLED in DB and the worker independently reports terminal: today's `_apply_task_transitions` early-returns on `task_row_is_finished(task)` (`transitions.py:1360`). With the new design, that early return must also delete the queue row for `(worker_id, task_id, attempt_id)`. Otherwise the row sits forever, the scheduler keeps subtracting nonexistent pending-stop resources from the worker's capacity, and the worker effectively shrinks until restart.

```python
if task_row_is_finished(task) or update.new_state in (TASK_STATE_UNSPECIFIED, TASK_STATE_PENDING):
    # If the worker reports terminal after we've already finalized, scrub the queue.
    if int(update.new_state) in TERMINAL_TASK_STATES and worker_id is not None:
        self._store.dispatch.delete_for_attempt(cur, worker_id, update.task_id, update.attempt_id)
    continue
```

Add a regression test (see Test strategy).

### Reservation tasks

Reservation-holder tasks never produce queue rows of either kind:

- **Start side.** `queue_assignments` already gates the start enqueue on `not job.is_reservation_holder` (`transitions.py:1278-1306`). The reservation-holder *attempt* is created (so the slot is held) but no RunTaskRequest is enqueued. Unchanged.
- **Stop side.** Today's `_kill_non_terminal_tasks` (`transitions.py:412`) already skips decommit for reservation holders (`transitions.py:444-446`) but it does populate `task_kill_workers[task_name] = worker_id` for them — meaning the legacy `_stop_tasks_direct` path *would* try to send StopTasks to the reservation slot's worker. There is no real worker-side process to kill; the reservation holder is metadata.

  In the new design, the kill enqueue must mirror the existing decommit predicate: skip the `enqueue_stop` call when `is_reservation_holder` is true. We do *not* want a row whose only effect is to telegraph StopTasks for a non-existent process and tie up `pending_stop_resources` accounting (the predicate would subtract zero anyway since the holder has no committed resources, but the row would exist and redispatch forever). Predicate at every kill enqueue site:

  ```python
  if not is_reservation_holder and worker_id is not None:
      self._store.dispatch.enqueue_stop(cur, worker_id, task_id, attempt_id, now_ms)
  ```

  Sites: `_kill_non_terminal_tasks`, `_terminate_coscheduled_siblings`, `_requeue_coscheduled_siblings`, `cancel_job`, `cancel_tasks_for_timeout`, `_remove_failed_worker`'s per-task loop, `preempt_task`. The reservation-holder branch in `_remove_failed_worker` (`transitions.py:1682-1684`) already does the right thing on the state side; we just don't enqueue.

### Concurrency invariants

The only lock in the new world is the SQLite writer lock. There is no in-memory queue, no `DispatchQueue._lock`, no `_workers_pending_kill_lock`.

**Atomic-with-respect-to-what.** Three pairs:

1. **Producing transition writes DB state change AND enqueues queue row in the same SQLite transaction.** Concretely, `_kill_non_terminal_tasks(cur, ...)` already takes a `TransactionCursor` (`transitions.py:412`); the new `enqueue_stop(cur, ...)` call uses the same cursor; SQLite holds the writer lock from BEGIN to COMMIT. There is no `on_commit` hook involved; the queue row is part of the transaction body. Rollback rolls back the row trivially. Verified: every kill enqueue site (`_terminate_task`, `_kill_non_terminal_tasks`, `_terminate_coscheduled_siblings`, `_requeue_coscheduled_siblings`, `cancel_job`, `cancel_tasks_for_timeout`, `_remove_failed_worker`'s loop, `preempt_task`) is already inside a transition method that owns a `cur`.

2. **Heartbeat-driven decommit AND queue-row deletion are in the same SQLite transaction.** `apply_task_updates` (`transitions.py:1579`) and `apply_heartbeats_batch` (`transitions.py:1588`) both take a `cur` and run inside it; we add `dispatch.delete_for_attempt(cur, ...)` next to the existing `decommit_resources(cur, ...)` call (`transitions.py:1500-1508`) and the `task_row_is_finished` early-return at `transitions.py:1360`.

3. **Scheduler reads dispatch queue rows AND `worker.committed_*` in the same read snapshot.** One `read_snapshot()` call, one consistent view. Because (1) and (2) are atomic, the snapshot cannot observe "row deleted, decommit not yet visible" or "row written, state change not yet visible."

The TransactionCursor's `on_commit` hook is *not* used for the dispatch queue, deliberately: the queue row is part of the transition, not a side effect. `on_commit` stays useful for in-memory caches that must lag the DB (endpoint cache, attribute cache) — unrelated.

#### Why this closes the iteration-1 microsecond window

PR #5550's `_workers_pending_kill |= workers_to_kill` ran *after* the SQLite COMMIT returned but *under* a separate `_workers_pending_kill_lock`. The microsecond window between the COMMIT making the decommit visible and the Python set update was the bug. In the new design there is no Python set: the row insertion *is* the visible signal of "this worker has a pending stop," and it lives in the same transaction as the state change. There is nothing to fall behind.

### RPC failure handling

Each row carries `attempts` and `last_error`. The poll loop on RPC failure:

- **Network/transport error.** `mark_dispatch_failed`: `attempts += 1`, set `last_error`, leave `dispatched_at_ms` unchanged. Per-worker exponential backoff (in-process state, recomputed from `attempts` — does not need to be persisted): `backoff_ms = min(2**attempts * 100, 30_000)`. The poll loop skips a worker whose last failure was less than `backoff_ms` ago.
- **Worker rejects (ack.accepted = False).** Synthesize a `HeartbeatApplyRequest` with `TASK_STATE_WORKER_FAILED` for that `(task_id, attempt_id)` exactly as `_dispatch_assignments_direct` does today (`controller.py:2316-2329`). `apply_task_updates` will roll the task back to PENDING (ASSIGNED → WORKER_FAILED is the existing pendable rollback at `transitions.py:1441`), decommit, and delete the queue row (via the `delete_for_attempt` call on terminal-heartbeat).
- **Poison pill (`attempts >= 16`, ≈8 minutes of backoff).** For starts: synthesize WORKER_FAILED as above; the task rolls to PENDING. For stops: log at WARNING and surface via dashboard diagnostics. The task is killed in our DB but we cannot tell the worker; rare today, leave it for an operator. v1 does not auto-fail the worker on stop poison pill (would cascade extra work); v2 can revisit.

### Controller restart

Boot sequence:

1. Run migrations.
2. Construct controller, scheduler, transitions.
3. Start poll loop. First tick selects the queue and dispatches anything pending.

There is no separate "reconstruct" phase — the queue rows are already there. The interesting question is what the poll loop does with each row given that the DB state may have advanced past it (or stalled before it) at the moment of crash. Enumerated:

| Queue row | Tasks-table state for `(task_id, attempt_id)` | Behavior on first tick after restart |
|---|---|---|
| `kind='run'`, `dispatched_at_ms IS NULL` | task ASSIGNED on this worker | dispatch StartTasks; worker idempotency absorbs any duplicate from a crash mid-RPC. |
| `kind='run'`, `dispatched_at_ms` set | task RUNNING (worker already accepted) | row would normally have been deleted by the RUNNING heartbeat. If we restarted between RPC ack and heartbeat, redispatch fires (post-cutoff) and the worker no-ops. Deletion will happen on the *next* RUNNING heartbeat — which is fine; the next heartbeat is within ~1s. |
| `kind='run'`, `dispatched_at_ms` set | task PENDING again (e.g. WORKER_FAILED rolled it back) | the row's `(task_id, attempt_id)` is stale. Must not redispatch. The poll loop's per-row predicate is "task row says current_attempt_id matches *or* exceeds q.attempt_id; if it exceeds, the row is stale — delete it." See "Stale-row scrub" below. |
| `kind='kill'`, `dispatched_at_ms IS NULL` | task killed in DB, attempt N still on worker W | dispatch StopTasks(T, N) to W. Same as steady-state. |
| `kind='kill'`, `dispatched_at_ms` set | terminal heartbeat for `(W, T, N)` already received and the apply_task_updates DELETE landed — but row exists | impossible: the DELETE is in the same tx as the decommit, both are durable on commit. If the row exists, the heartbeat hasn't been applied. Redispatch is correct. |
| `kind='kill'`, `dispatched_at_ms` set | DB says T is now on W' as attempt N+1, original W still healthy | the row is for the *old* attempt. We still need to kill `(T, N)` on W; the worker idempotency on `(task_id, attempt_id)` makes a duplicate stop harmless. The conservative scheduler still subtracts W's resources for this row, which is *correct* — W has not yet confirmed terminal for `(T, N)`, so the slot is not free. |
| `kind='kill'`, `worker_id` no longer in `workers` table | n/a | impossible: FK cascade and `WorkerStore.remove`'s explicit DELETE both wipe the row when the worker is removed. |

#### Stale-row scrub

The one row state the new design must explicitly handle on restart (and also during normal operation): `kind='run'` whose `attempt_id < tasks.current_attempt_id`. This can happen if the controller crashed between (a) the rollback transition incrementing `current_attempt_id` to N+1 and (b) the queue-row delete for the old attempt — but in this design, both writes are in the same transaction, so it cannot happen via the transitions path. It *can* happen if we add a future code path that promotes attempt N+1 without going through the rollback transition. Defense in depth: the poll loop's pre-dispatch predicate checks `q.attempt_id == tasks.current_attempt_id` (for `kind='run'` only — `kind='kill'` is intentionally allowed to lag, see the table above) and self-deletes mismatched run rows.

```sql
-- Pre-dispatch predicate for kind='run'
SELECT q.id, q.worker_id, q.task_id, q.attempt_id, q.payload_proto, q.dispatched_at_ms
FROM dispatch_queue q
JOIN tasks t ON t.task_id = q.task_id
WHERE q.kind = 'run'
  AND q.attempt_id = t.current_attempt_id   -- skip stale run rows
  AND (q.dispatched_at_ms IS NULL OR q.dispatched_at_ms < ?)
ORDER BY q.id
LIMIT ?;
```

Stale rows are deleted in a follow-up small transaction once per tick: `DELETE FROM dispatch_queue WHERE kind='run' AND attempt_id < (SELECT current_attempt_id FROM tasks WHERE tasks.task_id = dispatch_queue.task_id)`. The scrub is bounded; in the steady-state it is a no-op.

We also want one cheap startup scrub: `DELETE FROM dispatch_queue WHERE worker_id IS NOT NULL AND worker_id NOT IN (SELECT worker_id FROM workers)`. Defensive against the impossible "workers table loaded from a checkpoint that lags the queue table" — both tables are in the same DB so this should be unreachable, but the DELETE is cheap and removes a class of "what if".

### StartTasks idempotency (hard gate on Step 3)

The one worker-side requirement: sending the same `RunTaskRequest` (same `(task_id, attempt_id)`) twice must produce one container, not two. This is the precondition for:

- **Restart redispatch.** The queue row is durable; on restart we redispatch any row with `dispatched_at_ms` past the redispatch cutoff. Without idempotency, every restart spawns duplicate processes.
- **`redispatch_interval` redispatch.** A worker that ack'd StartTasks but hasn't yet sent the RUNNING heartbeat (slow build, slow image pull) will get a duplicate StartTasks at `redispatch_interval`. Without idempotency, that's a duplicate container.
- **Failure-then-retry on the same worker.** Cleared via the WORKER_FAILED rollback to PENDING, which assigns a *new* `attempt_id`. The old row is deleted in the same transaction as the rollback. Idempotency on `(task_id, attempt_id)` is sufficient; no global-task-level idempotency required.

**Step 3 cannot ship without measurement X = "send the same `(task_id, attempt_id)` twice to a worker; observe exactly one running container."** There is no in-memory dedup-set fallback in this design — the queue is durable across restarts, so restart correctness *requires* idempotency, and a process-local dedup-set would be lost on restart.

If the measurement fails, the fix lives in the worker, not the controller. Do not paper over with a Python set.

StopTasks for an unknown task is already a no-op; verified for both providers.

### K8s coexistence

Same table, same store, two row shapes — held apart by the table CHECK constraint, not just convention:

- **Worker-typed rows** (`worker_id IS NOT NULL`, `attempt_id IS NOT NULL`): produced by transitions for non-K8s workers, consumed by the poll loop.
- **K8s rows** (`worker_id IS NULL`, `attempt_id IS NULL`, `kind='kill'`): produced by `transitions.buffer_direct_kill` (`transitions.py:2630`), consumed by `transitions.drain_for_direct_provider` (`transitions.py:2364`) and the K8s sync loop.

The CHECK constraint added in migration 0042 makes this enforced rather than aspirational:

```sql
CHECK (
  (worker_id IS NULL AND attempt_id IS NULL AND kind='kill') OR
  (worker_id IS NOT NULL AND attempt_id IS NOT NULL)
)
```

The K8s side reads only `worker_id IS NULL` rows; the poll loop reads only `worker_id IS NOT NULL` rows; the partial unique index `WHERE worker_id IS NOT NULL` doesn't constrain K8s rows; `WorkerStore.remove`'s `DELETE WHERE worker_id = ?` only touches worker-typed rows.

#### Why one table, not two

We considered splitting into `dispatch_queue_workers` and `dispatch_queue_provider`. Rejected:

- Two stores with two migrations, two indexes, two test surfaces — and the "shared abstraction" gain is small because the two paths have different lifecycle anyway (K8s deletes on drain; worker rows survive until terminal heartbeat).
- The long-term direction (collapse the two into one `provider.dispatch(rows)` adapter — see Open Questions) is *easier* with one table than two.
- The CHECK constraint makes the cross-shape mutation impossible at the SQL layer, which removes the foot-gun.

Drain semantic asymmetry (K8s deletes on drain, worker rows survive until terminal heartbeat) is intentional: K8s acks via the next sync tick, non-K8s acks via heartbeat. We name the two store methods to surface this — `drain_direct_kills` (K8s, deletes) and `mark_dispatched` / `delete_for_attempt` (worker, two-phase).

## Migration plan

Four steps. Each is independently shippable; each is reversible by reverting the commit.

### Step 1: Schema migration + DispatchQueueStore extensions (no behavior change)

Land migration `0042_dispatch_queue_per_attempt.py` (the ALTER TABLEs and indexes above). Extend `DispatchQueueStore` with `enqueue_start`, `enqueue_stop`, `list_pending_for_worker`, `list_pending_stops_by_worker`, `mark_dispatched`, `mark_dispatch_failed`, `delete`, `delete_for_attempt`, `delete_all_for_worker`. **Wire transitions to also enqueue** (in addition to today's behavior — the existing direct dispatch still fires the RPCs):

- `queue_assignments` (`transitions.py:1231`): always call `store.dispatch.enqueue_start` (drop the `direct_dispatch` parameter's effect on enqueue — it now always enqueues, but still also returns `start_requests` so the existing caller still sends). `direct_dispatch` becomes a vestigial parameter we keep alive for one step and delete in Step 3.
- `_kill_non_terminal_tasks`, `_terminate_coscheduled_siblings`, `_requeue_coscheduled_siblings`, `cancel_job`, `cancel_tasks_for_timeout`, `_remove_failed_worker`: insert `enqueue_stop` calls for each `(worker_id, task_id, attempt_id)` that today gets propagated through `task_kill_workers`.
- `_apply_task_transitions` (`transitions.py:1500`) and the `task_row_is_finished` early return at `transitions.py:1360`: insert `dispatch.delete_for_attempt` calls when the worker confirms the task terminal.

No reader yet. Tests assert that the queue contains the expected rows after each transition (mirrors of existing transition tests).

Rollback: revert the commit. Migration is forward-only but the unused columns are harmless.

### Step 2: Switch the kill path through the queue

- Drop the `decommit_resources` calls inside `_kill_non_terminal_tasks` (`transitions.py:409`), `cancel_job` (`transitions.py:1098-1100`), `_terminate_coscheduled_siblings` (resources arg), `_requeue_coscheduled_siblings` (resources arg), `cancel_tasks_for_timeout` (`transitions.py:2024-2046`), `_remove_failed_worker`'s `_terminate_task` call (resources arg). The decommit call inside `apply_task_updates` (`transitions.py:1500-1508`) stays — that's the heartbeat-driven path.
- Replace the scheduler's `_workers_pending_kill` filter (`controller.py:2055-2064`) with a `pending_stop_resources` parameter passed into `create_scheduling_context`. Subtract resources in `WorkerCapacity.from_worker` (`scheduler.py:175`).
- Remove `_workers_pending_kill` and `_workers_pending_kill_lock` from `Controller`.
- Remove the post-commit `_workers_pending_kill |=` block in `_process_heartbeat_updates` (`controller.py:2467-2474`).
- The poll loop drains kills (initially: this step also adds the poll loop in a "stops only" mode; starts still go through `_dispatch_assignments_direct`).
- Direct stop dispatch (`_stop_tasks_direct` at `controller.py:2355` and the call site at `controller.py:2265`) is removed; `kill_tasks_on_workers` for non-K8s providers becomes a no-op (the queue row already exists, the poll loop handles it).

Affected regression tests: `tests/cluster/controller/test_5470_preemption_reassignment.py` updated to assert "scheduler sees no inflated capacity *before* the StopTasks RPC dispatches" without depending on a sleep or the poll loop.

Rollback: revert the commit. The decommit-in-transition behavior comes back; the queue continues to be populated but nothing reads it; identical to Step 1 state.

### Step 3: Switch the assignment path through the queue (gated by idempotency)

**Precondition:** StartTasks idempotency on `(task_id, attempt_id)` confirmed via gating measurement. If not confirmed, do not land this step; fix the worker first.

- Stop calling `provider.start_tasks` from `_dispatch_assignments_direct` (`controller.py:2301-2353`); the dispatch block goes away.
- Drop `direct_dispatch` from the `queue_assignments` signature.
- The poll loop now owns starts in addition to stops.
- Synthesizing WORKER_FAILED on RPC error/reject moves into the poll loop's failure handler.

Rollback: bring back the direct dispatch call in `_dispatch_assignments_direct`; queue continues to be populated but is shadowed by direct dispatch.

### Step 4: Collapse the four loops into the staggered poll loop

- Delete `_run_ping_loop` (`controller.py:2384`) and `_run_task_updater_loop` (`controller.py:2476`); fold their bodies into the poll loop's tick.
- Delete `_poll_all_workers`'s standalone caller (the inline `if poll_limiter.should_run(): self._poll_all_workers()` at `controller.py:1509-1513`); the staggered tick covers it.
- The scheduling loop stays separate. It still produces decisions and writes them to DB + queue.
- `_run_kill_dispatcher_loop`, if it exists, is gone.

This step is a refactor — all the wiring is in place from Steps 2 and 3; this just removes the now-duplicate scaffolding and consolidates the loop list.

Rollback: bring back the four loops by reverting the commit; the queue state is unaffected.

## Failure modes / risks

- **StartTasks idempotency unknown.** Gating measurement required before Step 3. There is no fallback in this design (a dispatched-set on the controller would lose restart correctness).
- **Restart redispatches all `dispatched_at_ms IS NOT NULL` rows.** A controller restart with N pending starts and M pending stops will fan out N+M RPCs as the poll loop spins up. At 4K workers with a few tasks each this is bounded by `dispatch_concurrency` and is fine; flag for measurement at scale.
- **SQLite write contention.** Producing transitions now write to `dispatch_queue` in addition to their existing tables. A coscheduled 256-replica gang requeue inserts 256 queue rows in one transaction; that's hundreds of microseconds of extra writer-lock time. The scheduler runs maybe once per second; the heartbeat path runs more often. Heartbeat-driven `delete_for_attempt` is a single DELETE WHERE per task — cheap. Poll-loop transactions are also small (UPDATE for dispatched_at_ms or attempts). Net: write rate goes up by ~one row per transition, not orders of magnitude. Measure under 4K-worker load before declaring done.
- **Read load.** The poll loop runs `SELECT WHERE id > last_seen LIMIT N` per tick, plus `SELECT WHERE kind='kill' GROUP BY worker_id` per scheduling tick. At 250ms tick and 4K workers, that's 4 selects/sec total scoped per-worker (after partitioning by worker for fan-out) plus the scheduler's 1/s scan. Indexes (`idx_dispatch_worker`, `idx_dispatch_kill_by_worker`) cover both. Should be fine; flag for measurement.
- **Stale stop rows from bizarre worker behavior.** A worker that ack's a stop, then never reports terminal, *and* never crosses the ping threshold, leaves a stop row alive forever. We redispatch every 5s, which is harmless but visible. Add a metric (`dispatch.queue_depth_by_kind`) so this is observable; an operator can prune via the existing tooling.
- **Coscheduled-job kill cascades enqueue many stops at once.** A 256-replica gang requeue produces 256 queue rows across (typically) 256 workers in one transition. Verified fine for the SQL writer (one INSERT per row under one transaction); for the poll loop's first dispatch tick, fan-out is bounded by `dispatch_concurrency`. Flag for scaled-test measurement.
- **`(task_id, attempt_id)` key on the K8s side is NULL.** The partial unique index `WHERE worker_id IS NOT NULL` excludes them. Existing K8s tests validate this, but the migration tests must explicitly cover "K8s rows do not collide with worker rows under the new unique constraint."

## Test strategy

- **`DispatchQueueStore` unit tests** (`tests/cluster/controller/test_dispatch_queue_store.py`, extending today's K8s coverage): `enqueue_start`/`enqueue_stop` insert; `INSERT OR IGNORE` collapses duplicate `(worker_id, kind, task_id, attempt_id)`; `list_pending` honors the `dispatched_at_ms` cutoff and the `worker_id IS NOT NULL` filter; `mark_dispatched`/`mark_dispatch_failed` update one row by id; `delete_for_attempt` is a no-op for unknown attempts; FK cascade on `WorkerStore.remove`; CHECK constraint rejects hybrid rows (worker_id NOT NULL + attempt_id NULL, etc.).
- **Migration `0042` test.** Start from a DB at migration 0041; insert K8s rows (`worker_id IS NULL`, `kind='kill'`, `attempt_id IS NULL`); insert a synthetic legacy worker-typed row (`worker_id NOT NULL`, `attempt_id IS NULL`) to verify the scrub. Apply migration; assert (a) K8s rows survive, (b) legacy worker-typed rows are deleted, (c) CHECK rejects new hybrid inserts, (d) partial unique index is present, (e) `idx_dispatch_pending` is present.
- **Conservative scheduler state integration test.** Stand up a controller with a fake provider whose `start_tasks`/`stop_tasks` are no-ops (queue fills up, never drained). Run a coscheduled-gang requeue. Assert `WorkerCapacity.available_*` matches `total - committed - sum(pending stop resources)` and that gang B is *not* placed on the still-pending workers — independent of any sleep.
- **#5470 regression test (`test_5470_preemption_reassignment.py`).** Replace the `intercepting_stop` mock with a paused poll loop (kill rows accumulate, never dispatched). Assert `result2 = _schedule_and_commit(scheduler, state)` returns zero assignments to the requeued slice. The legacy `test_pending_kill_guard_prevents_reassignment` is removed (the `_workers_pending_kill` set no longer exists).
- **Restart re-dispatch test.** Boot controller, submit a job, intercept the StartTasks RPC at the provider boundary (no actual worker), record that the row was inserted with `dispatched_at_ms` set, restart the controller process (in-test by reconstructing it against the same DB file), assert the row is dispatched again on the first poll tick after restart.
- **Stop-after-reassignment regression.** Enqueue a kill for `(W, T, N)`; before the poll loop runs, drive a heartbeat that fails T and a scheduling tick that promotes T to attempt N+1 on a fresh worker W'. Run the poll loop. Assert: (a) StopTasks RPC fires for `(T, N)` against W (not W'), (b) W's pending-stop accounting is still subtracted by the scheduler, (c) when W finally heartbeats `(T, N)` terminal, the kill row is deleted and W's capacity returns.
- **Cancelled-task-terminal-heartbeat test.** Worker reports SUCCEEDED for a task already KILLED in the controller's DB (cancel-job races worker completion). Verify the early-return path in `_apply_task_transitions` (`transitions.py:1360`) deletes the queue row for that `(task_id, attempt_id)` so the scheduler doesn't keep subtracting nonexistent pending-stop resources.
- **Poll-loop fan-out under failure.** Inject 30% RPC failure at the provider; assert eventual delivery via redispatch; assert `attempts` increments only on transport failure (not on no-op redispatch); assert poison-pill behavior at sustained 100% failure.
- **Worker death race.** Inject a ping-threshold worker failure mid-dispatch (`_remove_failed_worker` runs while a `dispatch_one_worker` is in flight against the same worker). Assert no stale queue rows for the dead worker (FK CASCADE), `mark_dispatched` for the now-deleted row is a no-op (UPDATE WHERE id=? matches zero rows), no spurious WORKER_FAILED transitions for tasks already terminated by `_remove_failed_worker`.
- **K8s coexistence test.** Run the K8s direct-provider sync side by side with worker-typed rows. Assert neither path observes the other's rows; assert the CHECK constraint rejects an attempt to enqueue a worker-typed row with NULL attempt_id.
- **Existing E2E suite (`lib/iris/tests/e2e/`)** must pass unchanged. Tests that depend on direct-dispatch timing relax to "within the next poll-loop tick" (~250ms).

## Open questions

Resolved by iteration 2 (closed, summarized in the iteration log): stop-after-reassignment semantics, default tick/redispatch values (with explicit to-measure markers), K8s-vs-non-K8s asymmetry foot-gun (CHECK constraint), second decommit codepath in `_enforce_execution_timeouts` (none — routes through `_terminate_task`), migration scrub of stale rows (`DELETE` worker-typed rows with NULL attempt_id during the rebuild), per-row vs per-worker quiet time (per-row is correct given the global-SELECT load model), reservation-holder kill semantics (do not enqueue), atomicity claims (verified at the cur boundary), restart story (enumerated table).

Remaining open:

1. **`payload_proto` size at scale.** `RunTaskRequest` includes inline `workdir_files`. For large workdirs (multi-MB), the BLOB grows. The dispatch queue is on the hot path of the writer lock. If it gets above ~100KB per row at scale, extract to a `dispatch_payloads` side-table referenced by id. Flag as a Step-3 measurement item.

2. **Per-step `dispatch_concurrency`.** `dispatch_concurrency=64` for both starts and stops. Stops are much cheaper than starts (no proto, no image pull). Splitting `dispatch_concurrency_starts` from `dispatch_concurrency_stops` is plausible but premature; defer until the staggered-tick benchmark says one is the bottleneck.

3. **Provider adapter extraction.** Once Step 4 lands and the poll-loop body is stable, the per-worker dispatch is the natural place to abstract over "non-K8s direct RPC" vs "K8s sync." Out of scope for v1 — but worth thinking about before we cement the store API.

4. **Stop-watchdog metric.** A worker that ack's a stop, then never reports terminal, *and* never crosses the ping threshold leaves a stop row redispatching every 5s. Harmless but visible. Add `dispatch.queue_depth_by_kind` and `dispatch.row_age_seconds_p99` so this is observable; consider a hard time-bound (e.g. force-fail the worker if a stop row stays alive >5min). Defer the hard bound until we have data.

5. **Drain-list query plan under SQLite WAL.** The single global `SELECT WHERE dispatched_at_ms IS NULL OR dispatched_at_ms < ?` is covered by `idx_dispatch_pending` but the `OR` may force two index scans depending on planner mood. If `EXPLAIN QUERY PLAN` shows a SCAN, rewrite as `UNION ALL` of two SARGable queries or store a synthetic `effective_dispatched_at_ms` (NULL → 0) so a single range scan covers both. Validate during Step 2 implementation.

6. **K8s `(task_id, attempt_id)` mapping.** K8s rows currently have `attempt_id IS NULL` because the direct-provider path uses `worker_id IS NULL` rows whose `task_id` is the wire ID. If we ever unify the two providers under a single dispatch adapter, K8s rows will need an `attempt_id` too. The CHECK constraint prevents accidental unification today. Track as a follow-up tied to Open Question 3.

## Iteration log

### Iteration 1 (2026-05-07, senior-engineer)

- Switched from in-memory `DispatchQueue` to SQL-backed `dispatch_queue` table. Per user direction: "let's just reintroduce the SQL dispatch queue." This eliminates the entire `on_commit` / post-commit-window class of bug because the queue row and the state change are in the same transaction.
- Reused the existing `dispatch_queue` schema + `DispatchQueueStore` rather than introducing a parallel table. Added migration `0042_dispatch_queue_per_attempt.py` for `attempt_id`, `dispatched_at_ms`, `attempts`, `last_error` columns plus a partial unique index.
- Introduced staggered poll-loop tick: 250ms tick interval, drain dispatch every tick, full PollTasks/Ping every 4 ticks (~1s). Justified defaults; flagged measurement at 4K-worker scale.
- Stop RPCs key off DB state at dispatch time per user direction. Start RPCs carry the serialized `RunTaskRequest` because rebuilding the proto from job_config + workdir_files + constraints each tick is non-trivial — captured this divergence explicitly.
- Resolved drain semantics: queue row dies on terminal heartbeat (not on RPC ack). Redispatch every 5s with idempotency on `(task_id, attempt_id)` covering duplicates. This is the clean reading of "stop RPCs are all keyed off DB state directly."
- Eliminated the in-memory `_workers_pending_kill` set + lock entirely.
- Restart correctness: queue rows survive crashes; first poll-loop tick after restart redispatches anything `dispatched_at_ms` plus `redispatch_interval` in the past. No reconstruction phase needed.
- Worker death: leveraged existing `WorkerStore.remove` → `DELETE FROM dispatch_queue WHERE worker_id` (`stores.py:1884`) plus FK CASCADE. No new wiring.
- Cancelled-task-terminal-heartbeat edge case: explicit `delete_for_attempt` call inside the early-return block in `_apply_task_transitions`. Easy to miss; flagged for regression test.
- K8s coexistence: same table, two row shapes (`worker_id IS NULL` for K8s, `worker_id IS NOT NULL` for worker dispatch). Partial unique index excludes K8s. K8s and the new poll loop see disjoint subsets of the table.
- Migration plan: rewrote four steps around the SQL queue. Step 1 (schema + double-write), Step 2 (kill path moves to queue, drop decommit-in-transition), Step 3 (assignment path moves to queue, gated on StartTasks idempotency measurement), Step 4 (collapse loops). Each step independently shippable.
- Counterpoint considered and rejected: "delete on ack, watchdog-redispatch on missing terminal heartbeat" — rejected because it splits state across the dispatch path and a watchdog. Putting redispatch in the row's own state (`dispatched_at_ms` + `attempts`) keeps everything in one place.
- Counterpoint considered and rejected: "kill replaces run in queue" semantics. Rejected: requires SELECT-DELETE-INSERT logic and the run-then-kill ordering is what already happens in normal operation; worker-side StopTasks for an unknown-or-just-started task is already a no-op or correct behavior.
- Open question 1 still open: per-row redispatch cadence vs per-worker quiet time.
- Open question 3 still open: `payload_proto` size at scale; may need to side-table large workdirs.
- Open question 7 flagged but low-risk: reservation-holder kill semantics under the new path.

### Iteration 2 (2026-05-07, senior-engineer)

Substantive doc changes:

- **Stop-after-reassignment closed.** Wrote out the full `(W, T, N)` vs `(W', T, N+1)` walkthrough; row carries `attempt_id`; StopTasks RPC fires for `(task_id, attempt_id)` against `q.worker_id` (the historical worker), not `tasks.current_worker_id`. The "key off DB state" reading is "look up the worker *address* and *resources* live; the row owns the attempt identity." Added a Stop-after-reassignment subsection.
- **Switched from per-worker SELECT to a single global SELECT per tick.** Iteration-1's per-worker scan was 16K SELECTs/sec at 4K workers and 250ms tick. The new index `idx_dispatch_pending(dispatched_at_ms, kind, worker_id)` covers a single global pending scan; we partition in Python after the read. Explicit Load model section: 4 SELECTs/sec, ~tens of UPDATE/INSERT/DELETE per second steady state.
- **Defaults marked to-measure-during-step-N.** Iteration-1 wrote defaults as if measured. Step 2 must validate `tick_interval` and `redispatch_interval`; Step 3 must validate `dispatch_concurrency` and `payload_proto` size.
- **K8s vs worker shape locked by CHECK constraint.** Iteration-1 left the partial unique index doing the work. CHECK now enforces `(worker_id IS NULL AND attempt_id IS NULL AND kind='kill') OR (worker_id IS NOT NULL AND attempt_id IS NOT NULL)`. The migration rebuilds the table to attach the CHECK.
- **Migration scrub spelled out.** `DELETE FROM dispatch_queue WHERE worker_id IS NOT NULL AND attempt_id IS NULL` runs before the table rebuild to drop legacy worker-typed rows. K8s rows survive. Migration body fully written in SQL.
- **Reservation-holder kill predicate explicit.** Iteration-1 said reservations don't enqueue; iteration-2 verified `_kill_non_terminal_tasks` (`transitions.py:444-446`) only skips *decommit* for reservation holders, not the `task_kill_workers` map population. The new design must add `is_reservation_holder` to the enqueue-stop predicate at every kill site (enumerated). Otherwise we'd accumulate forever-redispatching no-op kill rows for reservation holders.
- **Restart enumeration.** Replaced "trivial because rows persist" with a 7-row table covering each `(row state, DB state)` combination, including the stop-after-reassignment row that intentionally lags DB state.
- **Stale-row scrub for `kind='run'`.** New defense-in-depth subsection. Pre-dispatch predicate `q.attempt_id == tasks.current_attempt_id` for run rows; periodic delete of mismatched run rows. Kill rows are intentionally allowed to lag (we *want* to kill historical attempts).
- **Atomicity claim verified at the cur boundary.** Walked through `_kill_non_terminal_tasks(cur, ...)`, `cancel_job(cur, ...)`, `apply_task_updates(cur, ...)`, etc. — every kill enqueue site already takes a `TransactionCursor`; the new `enqueue_stop(cur, ...)` runs in the same transaction. No `on_commit` involved. Wrote the "Why this closes the iteration-1 microsecond window" subsection.
- **`_enforce_execution_timeouts` decommit codepath**: closed. It does not decommit on its own; it routes through `cancel_tasks_for_timeout` → `_terminate_task`. No second decommit codepath to chase.
- **House style on time columns**: iteration-1's question was wrongly stated. The SQL column is `dispatched_at_ms` (correct, matches the existing convention at `schema.py:507-571`); the Python attribute is `dispatched_at` via `python_name=`. Documented.
- **StartTasks idempotency promoted to a hard gate on Step 3.** Spelled out the three callers (restart redispatch, redispatch_interval redispatch, no third source). Removed any softer language about "fallback."
- **Test strategy refreshed.** Migration test now exercises CHECK + scrub. #5470 regression rewritten around the paused poll loop instead of the `intercepting_stop` mock that depended on `_workers_pending_kill`. Added explicit stop-after-reassignment regression. Restart re-dispatch test concrete (boot-from-DB-file).

Iteration-1 things found wrong (with citation):

- Iter-1 question 13: "house style is `dispatched_at` with a `Timestamp` decoder." Half right. The SQL column keeps `_ms` (`schema.py:507`, `submitted_at_ms`, `started_at_ms`, `finished_at_ms` everywhere), the Python alias drops it via `python_name`. Iter-1's claim that the SQL column should be `dispatched_at` is wrong; iter-2 corrects.
- Iter-1's per-worker SELECT pattern in the Drain-semantics section: would not scale. Iter-2 replaces with a global SELECT.
- Iter-1 said reservation-holder behavior is "unchanged." Half right. Decommit was already gated; *kill enqueue* would not be unless we added the predicate. Iter-2 makes the predicate explicit at all kill sites.

Considered and rejected (in addition to iter-1's rejections):

- **Two tables (`dispatch_queue_workers`, `dispatch_queue_provider`).** Rejected: doubles the migration/store/test surface, makes the long-term provider-adapter unification harder, and the CHECK constraint already prevents the cross-shape foot-gun.
- **Per-worker quiet-time tracking in a sidecar table.** Rejected: with the global SELECT load model, per-row `dispatched_at_ms` is already cheap to filter on. The sidecar would be redundant state.
- **Decommit on RPC ack instead of terminal heartbeat.** Same as iter-1's rejection, restated for completeness: ack is "I received your message," not "the task is gone." Decommit must wait for terminal heartbeat.
- **Putting `tasks.current_worker_id` in the kill predicate.** Rejected: stop-after-reassignment requires we still tell the *old* worker about the *old* attempt. The row's `q.worker_id` is the source of truth for that, not `tasks.current_worker_id`.

### Iteration 3 readiness check (2026-05-07, senior-engineer)

GO/NO-GO: **GO**.

- Migration 0042 is the next free number (verified — `0041_drop_worker_task_history.py` is the latest). `DISPATCH_QUEUE` at `schema.py:985` matches the doc. `_kill_non_terminal_tasks` at `transitions.py:412` already takes `cur: TransactionCursor` so the atomicity claim holds; reservation-holder gate at `transitions.py:434-446` matches the design's predicate.
- Doc is implementable as-is for Step 1. ml-engineer can land schema + DispatchQueueStore extensions + double-write transitions without further design clarification.
- Step ordering is correct: Step 1 additive (extra columns + unread rows), Step 2 introduces the conservative-state invariant gated by the paused-poll-loop regression test, Step 3 hard-gated on StartTasks idempotency (no Python-set fallback — correctly forbidden), Step 4 refactor only. Each step independently shippable and reversible.
- Test strategy covers both load-bearing scenarios from iteration 2's question 1: conservative-state integration test (paused poll, assert `available_* == total - committed - pending_stops`) and the explicit stop-after-reassignment regression for `(W, T, N)` vs `(W', T, N+1)`.
- Heads-up for the Step-1 implementor (not blockers): (a) `WorkerCapacity.from_worker` (`scheduler.py:156`) holds individual `available_cpu_millicores/memory/gpus/tpus` ints; the new `pending_stop_resources` proto needs to unpack into these four fields — Step-2 concern. (b) `payload_proto` is `expensive=True` in the existing schema; verify the covering index doesn't accidentally touch the BLOB. (c) `direct_dispatch` parameter staying vestigial for one step is fine but call it out in the PR description.
- No weasel-worded load-bearing claims. The to-measure markers are correctly on tunables, not correctness invariants. Open Question 5 (`OR` predicate planner-friendliness) is the one most likely to need `EXPLAIN QUERY PLAN` during Step 2 — doc already flags it.
- Long-form notes: `logs/scheduler-queue/iteration-3-readiness.md`.
