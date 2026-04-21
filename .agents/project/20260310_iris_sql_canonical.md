# SQL-Canonical Controller State

The controller's source of truth is a single SQLite database. All state—jobs,
tasks, workers, attempts, endpoints, dispatch queues, autoscaler checkpoints—lives
in SQL tables with foreign-key cascades and WAL-mode concurrency. There is no
in-memory state that must survive a restart.

## Previous Architecture

Controller state lived in Python dicts guarded by an `RLock` inside a monolithic
`ControllerState` class (2400+ lines). Reads and writes both acquired the same
lock. Persistence was a periodic protobuf snapshot dumped to disk; a restart
meant deserializing the entire snapshot back into memory. Query logic was
duplicated across `state.py`, `controller.py`, and `service.py`.

## File Layout

```
controller/
  db.py              # Table/column definitions, query DSL, row models, shared query helpers
  transitions.py     # All DB-mutating operations (state machine)
  controller.py      # Runtime loops, scheduling — reads DB directly
  service.py         # RPC handlers — reads DB directly
  scheduler.py       # Scheduling algorithm (pure logic, no DB access)
  autoscaler.py      # Scaling logic
  migrations/        # Numbered SQL migration files
```

## db.py — Query Layer

### Tables & Row Models

Tables are declared from frozen-dataclass row models annotated with `@db_row_model`.
Each field carries a `db_field()` with an optional decoder (e.g., `JobName.from_wire`,
`Timestamp.from_ms`, proto deserializer). Table objects are module-level constants:

```python
JOBS = _table_for_model(Job, "jobs", "j")
TASKS = _table_for_model(Task, "tasks", "t")
WORKERS = _table_for_model(Worker, "workers", "w")
ATTEMPTS = _table_for_model(Attempt, "task_attempts", "a")
ENDPOINTS = _table_for_model(Endpoint, "endpoints", "e")
```

Column access is `TABLE.c.column_name` (attribute access). Columns support
comparison operators (`==`, `!=`, `<`, `>`, `.in_()`) that produce composable
`Predicate` objects (`&`, `|`, `~`).

### Read Path — `db.snapshot()`

Opens a read transaction with snapshot isolation. Returns typed row-model
instances when selecting a full table, or lightweight decoded rows when
projecting specific columns.

```python
with db.snapshot() as q:
    tasks = q.select(TASKS, where=TASKS.c.state == TASK_STATE_PENDING)
    count = q.count(TASKS, where=...)
    job   = q.one(JOBS, where=JOBS.c.job_id == job_id)

    # Raw SQL escape hatch with per-column decoders
    rows = q.raw(
        "SELECT job_id, COUNT(*) as c FROM tasks GROUP BY job_id",
        decoders={"job_id": JobName.from_wire, "c": int},
    )
```

### Write Path — `db.transaction()`

Opens an IMMEDIATE write transaction with `insert()`, `update()`, `delete()`,
and `execute()` (raw SQL) on the cursor.

```python
with db.transaction() as tx:
    tx.insert("tasks", values={"task_id": tid, "state": 0, ...})
    tx.update("tasks", set={"state": 1}, where=TASKS.c.task_id == tid)
    tx.delete("tasks", where=TASKS.c.task_id == tid)
```

### Shared Query Helpers

A small set of multi-table read patterns that recur across `controller.py` and
`service.py` are top-level functions in `db.py` that accept a `QuerySnapshot`:

- `running_tasks_by_worker(q)` — tasks grouped by assigned worker
- `tasks_for_job_with_attempts(q, job_id)` — tasks with their current attempt
- `healthy_active_workers_with_attributes(q)` — workers joined with attributes

These are pure DSL calls, not methods on a stateful object.

## transitions.py — State Machine

`ControllerTransitions` owns every DB mutation. Each method runs inside a single
atomic transaction: read current state, validate, write new state, return a
frozen result dataclass. It has no query-only methods and does not expose its
`ControllerDB` handle.

Key method groups:

| Group | Methods |
|-------|---------|
| Job lifecycle | `submit_job`, `cancel_job`, `remove_finished_job` |
| Worker lifecycle | `register_worker`, `fail_workers_by_ids`, `remove_worker` |
| Heartbeat | `begin_heartbeat`, `apply_heartbeat`, `fail_heartbeat` |
| Assignment & dispatch | `queue_assignments`, `drain_dispatch`, `requeue_dispatch` |
| Endpoints | `add_endpoint`, `remove_endpoint`, `remove_endpoints_for_job` |
| Reservations | `replace_reservation_claims` |
| Checkpoint | `persist_checkpoint_state`, `restore_from` |

`_recompute_job_state` is called after task-state changes to derive the
parent job's state from aggregate task counts.

## controller.py & service.py — Direct Reads

Both hold a `ControllerDB` reference for reads and a `ControllerTransitions`
reference for writes. Reads use `db.snapshot()` inline—no wrapper functions,
no indirection through the transitions object.

```python
class ControllerRuntime:
    def __init__(self, transitions: ControllerTransitions, db: ControllerDB, ...):
        self._transitions = transitions
        self._db = db

    def _run_scheduling(self):
        with self._db.snapshot() as q:
            tasks = q.select(TASKS, where=...)
            workers = q.select(WORKERS, where=...)
            claims = q.select(RESERVATION_CLAIMS, ...)
        self._transitions.queue_assignments(assignments)
```

## Schema

Defined in `migrations/0001_init.sql` and applied automatically on startup.
Key tables: `jobs`, `tasks`, `task_attempts`, `workers`, `worker_attributes`,
`endpoints`, `dispatch_queue`, `reservation_claims`, `scaling_groups`,
`tracked_workers`, `txn_log`, `txn_actions`.

Notable schema features:
- **Foreign-key cascades**: deleting a job cascades to tasks, attempts, endpoints
- **Triggers**: `trg_txn_log_retention` caps the audit log at 1000 entries
- **Indexes**: `idx_tasks_pending` for scheduling queries, `idx_tasks_job_state`
  for job-state recomputation, `idx_dispatch_worker` for heartbeat draining
- **WAL mode + IMMEDIATE transactions**: concurrent readers, serialized writers

## What Was Removed

- **`state.py`** (2400 lines) — in-memory dicts, RLock, 30+ query methods, Protocol classes
- **`events.py`** — typed event classes and dispatch mechanism
- **`snapshot.py`** (bulk of it) — protobuf snapshot serialization/restore
- **`select_one_to_many`**, `_projection_cls`, `make_dataclass` — over-engineered query plumbing
- **Duplicate query wrappers** in `controller.py` and `service.py`
- **6 composite read-model classes** and 3 vestigial types from the old architecture
