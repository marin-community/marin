# SQL-Canonical Controller State: Rewrite Plan

## Current Problems

1. **Triple duplication of query helpers**: The same read logic exists in three places:
   - `state.py` methods like `jobs_by_id()`, `schedulable_tasks()`, `worker_roster()`
   - `service.py` module-level functions like `_read_job()`, `_tasks_for_listing()`, `_worker_roster()`
   - `controller.py` module-level functions like `_jobs_by_id()`, `_worker_capacities()`, `_schedulable_tasks()`

2. **db.py is over-engineered and under-powered**: The current query DSL has:
   - Verbose `db_row_model` / `db_field` metadata plumbing that's hard to follow
   - `select_one_to_many` that nobody uses well
   - Manual `_projection_cls` / `make_dataclass` for ad-hoc projections
   - No mutation DSL (all writes are raw SQL strings)

3. **state.py is 2259 lines mixing concerns**: It contains:
   - ~30 query methods that just wrap `db.snapshot().select(...)` (should not exist)
   - Complex state transitions (submit, cancel, heartbeat, assignment) — the real value
   - Dispatch buffering (heartbeat snapshot/drain/requeue)
   - Endpoint CRUD
   - Test helpers
   - Two Protocol classes that duplicate the method list

4. **No clear boundary**: Callers don't know whether to call `state.method()` or use
   `db.snapshot()` directly. Both paths exist and are used inconsistently.

## Golden Final State

### `db.py` — SQLAlchemy Core-like Query Layer

A clean, minimal query DSL with three responsibilities:

#### 1. Table & Column Definitions with Converters

```python
JOBS = Table("jobs", alias="j", columns={
    "job_id":       col("job_id",       decode=JobName.from_wire, encode=JobName.to_wire),
    "state":        col("state",        decode=int),
    "request_proto": col("request_proto", decode=_decode_launch_request),
    "submitted_at_ms": col("submitted_at_ms", decode=Timestamp.from_ms),
    ...
})

TASKS = Table("tasks", alias="t", columns={...})
WORKERS = Table("workers", alias="w", columns={...})
```

#### 2. Query DSL — Select, Where, Join, Order, Limit

Column access is via `TABLE.c.column_name` (attribute access, no `.col("name")` dict lookup).

```python
with db.snapshot() as q:
    # Select with row model — returns list[Task] with decoded fields
    tasks = q.select(TASKS,
        where=TASKS.c.state == cluster_pb2.TASK_STATE_PENDING,
        order_by=TASKS.c.submitted_at_ms.asc(),
    )

    # Projection — explicit columns, returns decoded rows with .attr access
    rows = q.select(TASKS,
        columns=[TASKS.c.task_id, TASKS.c.state],
        where=...,
    )

    # Join — returns flat rows, one per joined match
    rows = q.select(TASKS,
        columns=[TASKS.c.task_id, WORKERS.c.address],
        join=TASKS.join(WORKERS, on=TASKS.c.current_worker_id == WORKERS.c.worker_id),
        where=...,
    )

    # Scalar / count / exists
    count = q.count(TASKS, where=TASKS.c.state == ...)
    has_pending = q.exists(TASKS, where=TASKS.c.state == ...)
```

#### 3. Escape Hatch — Raw SQL with Decoders

```python
with db.snapshot() as q:
    rows = q.raw(
        "SELECT job_id, state, COUNT(*) as c FROM tasks GROUP BY job_id, state",
        decoders={"job_id": JobName.from_wire, "state": int, "c": int},
    )
```

#### 4. Mutation DSL — Thin Sugar

```python
with db.transaction() as tx:
    tx.insert("tasks", values={"task_id": "...", "state": 0, ...})
    tx.update("tasks", set={"state": new_state}, where=TASKS.c.task_id == "...")
    tx.delete("tasks", where=TASKS.c.task_id == "...")
    # Raw SQL when needed:
    tx.execute("UPDATE workers SET healthy = 0 WHERE worker_id = ?", (wid,))
```

#### 5. Row Model Types (Frozen Dataclasses)

Keep `Job`, `Task`, `Worker`, `Attempt`, `Endpoint` as frozen dataclasses because
they carry domain logic methods (`task.is_finished()`, `task.can_be_scheduled()`,
`worker.available_gpus`, `job.is_coscheduled`). A plain decoded row can't have these.

When you `q.select(TABLE)` without explicit `columns=`, the result is decoded into
the table's row model. When you specify `columns=`, you get flat decoded rows with
`.attr` access (no model instance).

#### 6. Joins — Flat Rows with Per-Column Decoding

Joins return flat rows. Each column in `columns=` carries its own decoder, so
the result has `.attr` access with decoded Python types. `columns=` is required
when using joins — omitting it is an error (we can't decode a joined row into
a single table's row model).

```python
with db.snapshot() as q:
    # Cross-table columns: task_id (JobName) + worker address (str)
    rows = q.select(TASKS,
        columns=[TASKS.c.task_id, WORKERS.c.address],
        join=TASKS.join(WORKERS, on=TASKS.c.current_worker_id == WORKERS.c.worker_id),
        where=TASKS.c.state == cluster_pb2.TASK_STATE_RUNNING,
    )
    for row in rows:
        print(row.task_id, row.address)  # JobName, str — decoded by column metadata

    # Aggregate with join: building counts per worker (derived from tasks + task_attempts)
    rows = q.raw(
        "SELECT a.worker_id, COUNT(*) as c FROM tasks t "
        "JOIN task_attempts a ON t.task_id = a.task_id AND t.current_attempt_id = a.attempt_id "
        "JOIN jobs j ON t.job_id = j.job_id "
        "WHERE t.state IN (?, ?) AND j.is_reservation_holder = 0 "
        "GROUP BY a.worker_id",
        (TASK_STATE_BUILDING, TASK_STATE_ASSIGNED),
        decoders={"worker_id": WorkerId, "c": int},
    )
    for row in rows:
        print(row.worker_id, row.c)  # WorkerId, int
```

No special decoding step — column metadata handles it. The result objects are
lightweight (named tuples or similar) with attribute access.

#### 7. Multi-Entity Reads ("Job + All Tasks")

No implicit grouping, no `select_one_to_many`, no ORM-style eager loading.
Use batched queries inside a single snapshot to avoid N+1:

```python
with db.snapshot() as q:
    jobs = q.select(JOBS, where=...)
    # One query for ALL tasks across all jobs — not one per job
    tasks = q.select(TASKS,
        where=TASKS.c.job_id.in_([j.job_id.to_wire() for j in jobs]),
    )
# Group in Python
tasks_by_job = defaultdict(list)
for t in tasks:
    tasks_by_job[t.job_id].append(t)
```

Two queries, one snapshot, O(1) round trips regardless of N. The snapshot
guarantees consistency. Joins are for when you need cross-table columns in
a single flat row (e.g., task_id + worker_address).

**What db.py does NOT have:**
- No `select_one_to_many` or implicit grouping
- No `_projection_cls` / `make_dataclass` magic
- No semantic query helpers (`schedulable_tasks`, `jobs_by_id`, etc.)
- No `endpoint_query_predicate` or other business-logic query builders

### `transitions.py` — State Machine (rename from `state.py`)

Owns all state transitions. Each method is a transaction that:
1. Reads current state
2. Validates the transition
3. Writes new state
4. Returns a result with side effects

```python
class ControllerTransitions:
    """State machine for controller entities.

    All methods that mutate DB state live here. Each is a single atomic
    transaction. Read-only queries do NOT belong here — callers use
    db.snapshot() directly.
    """
    def __init__(self, db: ControllerDB, log_store: LogStore): ...

    # --- Job Lifecycle ---
    def submit_job(job_id, request, ts) -> SubmitJobResult
    def cancel_job(job_id, reason) -> CancelJobResult

    # --- Task Assignment ---
    def queue_assignments(assignments) -> AssignmentResult
    def mark_task_unschedulable(task_id, reason) -> TxResult

    # --- Heartbeat / Worker Lifecycle ---
    def register_or_refresh_worker(worker_id, address, metadata, ts) -> TxResult
    def apply_heartbeat(req: HeartbeatApplyRequest) -> HeartbeatApplyResult
    def record_heartbeat_failure(worker_id, error, batch) -> HeartbeatFailureResult

    # --- Dispatch Buffer ---
    def buffer_dispatch(worker_id, task_request) -> None
    def buffer_kill(worker_id, task_id) -> None
    def drain_dispatch(worker_id) -> DispatchBatch | None
    def requeue_dispatch(batch) -> None

    # --- Reservation Claims ---
    def replace_reservation_claims(claims) -> None

    # --- Checkpoint ---
    def persist_checkpoint_state(scaling_groups, tracked_workers) -> None

    # --- Cleanup ---
    def remove_finished_job(job_id) -> bool
    def remove_worker(worker_id) -> Worker | None

    # --- Endpoints ---
    def add_endpoint(endpoint, task_id) -> None
    def remove_endpoint(endpoint_id) -> Endpoint | None
    def remove_endpoints_for_job(job_id) -> list[Endpoint]
```

**What transitions.py does NOT have:**
- No query methods (`reservation_claims()` is a read — it goes in the caller)
- No `checkpoint_state()` (the DB is the canonical state; query it directly)
- No `queries` / `db` property exposed to callers for reads
- No Protocol classes (callers depend on the concrete class)

### `controller.py`, `service.py` — Direct DB Reads

These files query the DB directly using the DSL. No indirection through
state/transitions. `reservation_claims()` is just a DSL read, inlined
wherever controller.py needs it.

```python
class ControllerRuntime:
    def __init__(self, transitions: ControllerTransitions, db: ControllerDB, ...):
        self._transitions = transitions
        self._db = db

    def _run_scheduling(self):
        with self._db.snapshot() as q:
            tasks = q.select(TASKS,
                where=~TASKS.c.state.in_(TERMINAL_TASK_STATES),
                order_by=[TASKS.c.priority_neg_depth.asc(), ...],
            )
            workers = q.select(WORKERS,
                where=(WORKERS.c.healthy == 1) & (WORKERS.c.active == 1),
            )
            # reservation claims — just a read, no special method
            claims = q.select(RESERVATION_CLAIMS,
                columns=[RESERVATION_CLAIMS.c.worker_id, ...],
            )
        self._transitions.queue_assignments(assignments)
```

Module-level query helper functions in controller.py and service.py are **deleted**.
If a multi-table read pattern recurs (e.g., "tasks with attempts for a set of
task IDs"), it's a small top-level function that takes a `QuerySnapshot` and
returns the composed result — but it's just DSL calls, not a method on a
stateful object.

### File Layout (Final)

```
controller/
  db.py              # Table defs, Column, query DSL, ControllerDB, row models
  transitions.py     # State machine (renamed from state.py)
  controller.py      # Runtime loops, scheduling (reads db directly)
  service.py         # RPC handlers (reads db directly)
  autoscaler.py      # Scaling logic
  migrations/        # SQL migration files
```

## Task Breakdown

Each task is independently implementable and testable. Tasks are ordered by
dependency — earlier tasks don't depend on later ones.

### Phase 1: Clean Up db.py Query DSL

**Task 1.1: Replace `.col("name")` with `.c.name` attribute access**

Currently columns are accessed via `TABLE.col("column_name")` (dict lookup, string-keyed).
Replace with `TABLE.c.column_name` attribute access. Remove `.col()` entirely.
Update all callers (db.py internal, state.py, controller.py, service.py, tests).

Files changed: `db.py`, all callers of `.col()`
Test: `uv run pytest lib/iris/tests/cluster/controller/ -x`

**Task 1.2: Add mutation helpers to ControllerDB**

Add `insert()`, `update()`, `delete()` methods to the transaction cursor.
These are thin SQL builders — values must already be SQL-compatible types.

```python
with db.transaction() as tx:
    tx.insert("tasks", values={"task_id": "...", "state": 0})
    tx.update("tasks", set={"state": 1}, where=TASKS.c.task_id == "...")
    tx.delete("tasks", where=TASKS.c.task_id == "...")
```

Files changed: `db.py`
Test: Unit test the builders in `test_state.py` or a new `test_db.py`

**Task 1.3: Add `raw()` escape hatch to QuerySnapshot**

```python
q.raw("SELECT ...", params=(...,), decoders={"col": decoder_fn})
```

Returns list of named-access objects with decoded values.

Files changed: `db.py`
Test: Add a test that uses `raw()` for a GROUP BY query

**Task 1.4: Remove `select_one_to_many` and `_projection_cls` / `make_dataclass`**

Nobody uses `select_one_to_many` well. The `_projection_cls` dynamically creates
dataclasses at runtime — replace with simple named tuples or just use the decoded
row objects directly.

Files changed: `db.py`
Test: Existing tests still pass; grep to confirm no callers

### Phase 2: Rename state.py → transitions.py, Strip Query Methods

**Task 2.1: Create `transitions.py` as a copy of `state.py`, remove all query methods**

1. Copy `state.py` → `transitions.py`
2. Rename class `ControllerState` → `ControllerTransitions`
3. Delete ALL query-only methods (see list below)
4. Delete the two Protocol classes (`ControllerRuntimeStateProtocol`, `ControllerServiceStateProtocol`)
5. Delete type aliases (`JobView`, `TaskView`, `WorkerView`, `EndpointRecord`)
6. Keep: all mutation methods, dispatch buffer, `_record_transaction`, `_recompute_job_state`
7. Do NOT expose a `db` / `queries` property — callers hold their own `ControllerDB` ref
8. Keep: `log_store` property, `db_path`, `close()`, `backup_to()`, `restore_from()`

Query methods to delete from transitions.py:
- `reservation_claims()` — inline DSL read at call sites in controller.py
- `checkpoint_state()` — delete entirely; callers read scaling_groups/tracked_workers tables directly
- `active_worker_ids()`
- `jobs_by_id()`
- `jobs_in_states()`
- `running_job_count()`
- `task_job_worker_counts()`
- `schedulable_tasks()`
- `healthy_active_workers_with_attributes()`
- `running_tasks_by_worker()`
- `task_worker_mapping()`
- `workers_by_id()`
- `tasks_by_ids_with_attempts()`
- `tasks_for_listing()`
- `worker_roster()`
- `worker_status_detail()`
- `worker_addresses()`
- `task_summaries_for_jobs()`
- `live_user_stats()`
- `descendant_jobs()`
- `child_jobs()`
- `task_by_id_with_attempts()`
- `tasks_for_job_with_attempts()`
- `tasks_for_worker()`
- `building_counts()`
- `endpoints()`
- `endpoint_task_mapping()`
- `transaction_actions()`
- `add_job()` (test helper that wraps submit_job)

Files changed: new `transitions.py`
Test: `uv run python -c "from iris.cluster.controller.transitions import ControllerTransitions"`

**Task 2.2: Update `controller.py` to use `ControllerTransitions` + direct DB reads**

1. Replace `from state import ControllerState` → `from transitions import ControllerTransitions`
2. Replace `self._state` → `self._transitions` for mutations
3. Hold a `self._db: ControllerDB` for reads (not via transitions)
4. Delete module-level query wrapper functions:
   - `_jobs_by_id()`, `_worker_capacities()`, `_schedulable_tasks()`
   - `_running_tasks_by_worker()`, `_tasks_by_ids_with_attempts()`
   - `_building_counts()`, `_workers_by_id()`, `_task_worker_mapping()`
   - `_tasks_for_job_with_attempts()`
5. Inline their DSL queries at the call sites
6. `reservation_claims()`: inline DSL read from RESERVATION_CLAIMS table
7. `checkpoint_state()`: inline DSL reads from SCALING_GROUPS + TRACKED_WORKERS tables

Files changed: `controller.py`
Test: `uv run pytest lib/iris/tests/cluster/controller/test_scheduler.py lib/iris/tests/cluster/controller/test_reservation.py -x`

**Task 2.3: Update `service.py` to use `ControllerTransitions` + direct DB reads**

1. Hold `self._transitions` for mutations, `self._db` for reads
2. Delete ALL module-level query wrapper functions:
   - `_read_job()`, `_read_task_with_attempts()`, `_read_worker()`, `_read_worker_detail()`
   - `_child_jobs()`, `_tasks_for_job_with_attempts()`, `_tasks_for_listing()`
   - `_worker_addresses()`, `_healthy_active_workers_with_attributes()`
   - `_jobs_in_states()`, `_task_summaries_for_jobs()`, `_worker_roster()`
   - `_running_tasks_by_worker()`, `_query_endpoints()`, `_descendant_jobs()`
   - `_transaction_actions()`, `_live_user_stats()`, `_tasks_for_worker()`
3. Inline DSL queries at each RPC handler

Files changed: `service.py`
Test: `uv run pytest lib/iris/tests/cluster/controller/test_job.py -x`

**Task 2.4: Update `autoscaler.py` and any other callers**

Grep for all imports of `ControllerState`, `ControllerRuntimeStateProtocol`,
`ControllerServiceStateProtocol` and update them.

Files changed: `autoscaler.py`, any other importers
Test: `uv run pytest lib/iris/tests/cluster/controller/ -x`

**Task 2.5: Delete `state.py`**

After all callers are migrated, delete state.py.

Files changed: delete `state.py`
Test: `uv run pytest lib/iris/tests/cluster/controller/ -x`

### Phase 3: Update Tests

**Task 3.1: Update `test_state.py` → `test_transitions.py`**

1. Rename file
2. Update imports: `ControllerState` → `ControllerTransitions`
3. Replace query helper calls with direct DB reads:
   - `_query_job(state, job_id)` → `db.snapshot().one(JOBS, where=JOBS.c.job_id == ...)`
   - `_query_task(state, task_id)` → inline DSL
   - `_query_worker(state, worker_id)` → inline DSL
   - `_query_tasks_for_job(state, job_id)` → inline DSL
   - `_schedulable_tasks(state)` → inline DSL
   - `_worker_running_tasks(state, worker_id)` → derived query via tasks JOIN task_attempts
4. Keep test helper functions that perform mutations (`submit_job`, `register_worker`,
   `dispatch_task`, `transition_task`, `fail_worker`)

Files changed: `test_state.py` → `test_transitions.py`
Test: `uv run pytest lib/iris/tests/cluster/controller/test_transitions.py -x`

**Task 3.2: Update `test_job.py`**

1. Update imports
2. Replace any `state.some_query()` calls with DSL reads

Files changed: `test_job.py`
Test: `uv run pytest lib/iris/tests/cluster/controller/test_job.py -x`

**Task 3.3: Update `test_scheduler.py`**

1. Update imports
2. Replace query helpers with DSL reads
3. Keep scheduling-specific helpers (`schedule_until_done`, `_build_context`)

Files changed: `test_scheduler.py`
Test: `uv run pytest lib/iris/tests/cluster/controller/test_scheduler.py -x`

**Task 3.4: Update `test_reservation.py`**

1. Update imports
2. Replace query helpers with DSL reads

Files changed: `test_reservation.py`
Test: `uv run pytest lib/iris/tests/cluster/controller/test_reservation.py -x`

### Phase 4: Final Cleanup

**Task 4.1: Remove hydrate fields from row models**

The `Worker` model has `attributes`, `running_tasks`, `task_history`, `resource_snapshot`,
`resource_history` as hydrate fields that require separate queries. Query these
separately and compose at the call site:

```python
with db.snapshot() as q:
    worker = q.one(WORKERS, where=...)
    attrs = q.select(WORKER_ATTRIBUTES, where=WORKER_ATTRIBUTES.c.worker_id == ...)
    running = q.raw("SELECT t.task_id FROM tasks t "
        "JOIN task_attempts a ON t.task_id = a.task_id AND t.current_attempt_id = a.attempt_id "
        "WHERE a.worker_id = ? AND t.state IN (?, ?, ?)", ...)
```

The `Task` model's `attempts` hydrate field follows the same pattern — query
attempts separately and compose at call site.

Files changed: `db.py`, callers that use hydrate fields
Test: All controller tests pass

**Task 4.2: Document the final architecture**

Update this document to reflect the completed state. Remove task checklists,
keep only the architecture description.

Files changed: this file
Test: N/A

## Execution Order

```
Phase 1 (db.py cleanup):     1.1 → 1.2 → 1.3 → 1.4
Phase 2 (state→transitions): 2.1 → 2.2, 2.3, 2.4 (parallel) → 2.5
Phase 3 (test updates):      3.1, 3.2, 3.3, 3.4 (parallel)
Phase 4 (final cleanup):     4.1 → 4.2
```

Phases 1 and 2.1 can be done first. Tasks 2.2-2.4 depend on 2.1 but are
independent of each other. Phase 3 can overlap with Phase 2 (update tests
as each caller is migrated). Phase 4 is optional polish.

## Checklist

### Phase 1: db.py
- [x] 1.1: Replace `.col("name")` with `.c.name`, remove `.col()`
- [x] 1.2: Add `insert()`, `update()`, `delete()` to transaction
- [x] 1.3: Add `raw()` to QuerySnapshot
- [x] 1.4: Remove `select_one_to_many`, `_projection_cls`, `make_dataclass`

### Phase 2: state.py → transitions.py
- [x] 2.1: Create transitions.py, strip query methods
- [x] 2.2: Update controller.py
- [x] 2.3: Update service.py
- [x] 2.4: Update autoscaler.py + other callers (no changes needed)
- [x] 2.5: Delete state.py

### Phase 3: Tests
- [x] 3.1: test_state.py → test_transitions.py
- [x] 3.2: test_job.py
- [x] 3.3: test_scheduler.py
- [x] 3.4: test_reservation.py

### Phase 4: Final Cleanup
- [x] 4.1: Remove hydrate fields from row models
- [x] 4.2: Update this document

## Additional work done beyond original plan
- Consolidated duplicated constants (`TERMINAL_TASK_STATES`, `TERMINAL_JOB_STATES`) into `db.py`
- Deduplicated shared query helpers (`running_tasks_by_worker`, `tasks_for_job_with_attempts`, `healthy_active_workers_with_attributes`) into `db.py`
- Removed 6 vestigial composite read-model classes from `db.py`
- Removed 3 vestigial types from `transitions.py`
- Inlined `_read_checkpoint_state` at call site
- Replaced inline terminal-state literals with constants in `transitions.py`
- Introduced `_WorkerDetail` composition type in `service.py` for worker detail reads
