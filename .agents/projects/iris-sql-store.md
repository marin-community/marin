# Iris: Introduce Stores Layer Between `transitions.py` and `db.py`

## Context

`lib/iris/src/iris/cluster/controller/transitions.py` has grown to ~3,350 lines containing ~174 inline SQL queries that operate directly against `ControllerDB`. SQL and domain logic are tangled, there's no typed API boundary around the DB, and write-through caches (like `EndpointRegistry`) live either inside `db.py` or floating beside `transitions.py`.

Goal: introduce a **stores layer** so the dependency chain is:

```
db.py        — connections, migrations, transaction/snapshot context managers, no schema knowledge
schema.py    — table DDL, row dataclasses, projections (unchanged)
stores.py    — depends on { db, schema }; typed per-entity stores + ControllerStore wrapper
transitions.py — depends on stores; NO direct db.py SQL
```

All store operations take a transaction (read or write) explicitly, e.g. `JobStore.list_task_attempts(tx, job_id)`. Stores own their write-through caches (e.g. `EndpointStore` absorbs today's `EndpointRegistry`).

This refactor is **low-risk and phased** — we do not try to migrate all 174 queries at once. Phase 1 scaffolds the layer and moves `EndpointRegistry` (the natural first candidate, since it's already a write-through cache). Subsequent phases migrate one entity at a time.

## Current State (verified during exploration)

- **DB access**: `ControllerDB.transaction()` yields `TransactionCursor`, `ControllerDB.read_snapshot()` yields `QuerySnapshot` (from a 32-conn pool). Post-commit hooks via `cur.on_commit(fn)` already underpin cache coherence.
- **Existing typed rows/projections in [schema.py](lib/iris/src/iris/cluster/controller/schema.py)**: `JobRow`, `JobSchedulingRow`, `JobDetailRow`, `TaskRow`, `TaskDetailRow`, `WorkerRow`, `WorkerDetailRow`, `AttemptRow`, `EndpointRow`, `ApiKeyRow`, `UserBudgetRow`, with matching `*_PROJECTION` objects. These are the return types we'll lean on.
- **Existing write-through cache**: [`EndpointRegistry`](lib/iris/src/iris/cluster/controller/endpoint_registry.py:43) — loads all rows at init, mutates memory in `cur.on_commit(...)` hooks. This is exactly the pattern stores will use.
- **Existing in-DB cache**: `ControllerDB._attr_cache` ([db.py:321](lib/iris/src/iris/cluster/controller/db.py:321)) is a worker-attribute map that belongs semantically in `WorkerStore`. Still actively used by `healthy_active_workers_with_attributes` ([db.py:908](lib/iris/src/iris/cluster/controller/db.py:908)) from multiple controller.py call sites and written from transitions.py on worker register/remove — stays put in Phase 1, relocates in Phase 5.
- **SQL entities in transitions.py** (count of queries per entity): jobs (~20), tasks (~35), task_attempts (~12), workers (~15), dispatch_queue (~5), endpoints (delegated to EndpointRegistry), task_resource_history (~6), worker_resource_history / worker_task_history (~6), reservation_claims (~2), meta (~3), users / user_budgets (~2). Full inventory captured during exploration.

## Design

### New file: `lib/iris/src/iris/cluster/controller/stores.py`

```python
from iris.cluster.controller.db import ControllerDB, TransactionCursor, QuerySnapshot
from iris.cluster.controller.schema import JobRow, TaskRow, WorkerRow, AttemptRow, EndpointRow, ...

# Type used by read methods that accept either a write cursor or a read snapshot.
# Writes require TransactionCursor explicitly.
Tx = TransactionCursor | QuerySnapshot


class JobStore:
    def __init__(self, db: ControllerDB) -> None:
        self._db = db  # opaque handle for the rare case a store needs the connection itself

    # reads
    def get(self, tx: Tx, job_id: JobName) -> JobRow | None: ...
    def get_config(self, tx: Tx, job_id: JobName) -> JobConfigRow | None: ...
    def list_descendants(self, tx: Tx, job_id: JobName) -> list[JobName]: ...
    def list_terminal_ids(self, tx: Tx) -> list[JobName]: ...

    # writes
    def insert(self, tx: TransactionCursor, job: JobInsert) -> None: ...
    def update_state(self, tx: TransactionCursor, job_id: JobName, state: JobState,
                     error: str | None, finished_at_ms: int | None) -> None: ...
    def delete(self, tx: TransactionCursor, job_id: JobName) -> None: ...


class TaskStore: ...          # tasks table + task_resource_history
class TaskAttemptStore: ...    # task_attempts
class WorkerStore: ...         # workers + worker_attributes (+ the attr cache currently in db.py)
class EndpointStore: ...       # former EndpointRegistry, renamed and relocated
class DispatchQueueStore: ...  # dispatch_queue
class ReservationStore: ...    # reservation_claims + meta(last_submission_ms)


class ControllerStore:
    """Bundle of per-entity stores with direct access to transactions/snapshots."""
    def __init__(self, db: ControllerDB) -> None:
        self._db = db
        self.jobs = JobStore(db)
        self.tasks = TaskStore(db)
        self.attempts = TaskAttemptStore(db)
        self.workers = WorkerStore(db)
        self.endpoints = EndpointStore(db)
        self.dispatch = DispatchQueueStore(db)
        self.reservations = ReservationStore(db)

    def transaction(self): return self._db.transaction()
    def read_snapshot(self): return self._db.read_snapshot()
```

### Transaction rule

- **Reads**: accept `Tx = TransactionCursor | QuerySnapshot`. Store methods internally call `tx.fetchall(...)` / `tx.fetchone(...)`. (Both types already expose these; where signatures diverge, the store normalizes.)
- **Writes**: require `TransactionCursor` specifically. Static typing enforces the invariant.
- **No store method opens its own transaction.** Callers are responsible for transaction scope. This matches today's pattern in `transitions.py` and keeps batching/atomicity in caller control.

### Validation in stores

Light and unambitious for phase 1:
- Reject writes with impossible state combinations (e.g., `update_state` asserting the new state is in `JobState`).
- Decode rows into the existing `*Row` dataclasses at the boundary — callers never see `sqlite3.Row`.
- No business rules (retry counts, cascade logic) — those stay in `transitions.py`.

### EndpointStore (rename of EndpointRegistry)

Move [`endpoint_registry.py`](lib/iris/src/iris/cluster/controller/endpoint_registry.py) → `EndpointStore` inside `stores.py`. Semantically identical: write-through cache keyed by id/name/task with post-commit hooks.

- Replace `db.endpoints` accessor in [db.py:334](lib/iris/src/iris/cluster/controller/db.py:334) with `store.endpoints`.
- Delete `endpoint_registry.py`; migrate its test file [test_endpoint_registry.py](lib/iris/tests/cluster/controller/test_endpoint_registry.py) to construct `EndpointStore` directly.

### transitions.py integration

Change the `ControllerTransitions` constructor:

```python
# before
def __init__(self, db: ControllerDB, ...): self._db = db

# after
def __init__(self, store: ControllerStore, ...):
    self._store = store
    self._db = store._db  # phased: kept only while unmigrated queries remain
```

The `self._db` escape hatch exists **only during the phased migration** and is deleted at the end. This lets each phase move a subset of queries without breaking the file.

## Phasing

Low-risk means small, verifiable PRs. Proposed sequence:

**Phase 1 (this PR) — scaffolding + EndpointStore**
1. Create `stores.py` with empty `JobStore`, `TaskStore`, `TaskAttemptStore`, `WorkerStore`, `DispatchQueueStore`, `ReservationStore` skeletons.
2. Fold `EndpointRegistry` into `stores.py` as `EndpointStore`.
3. Add `ControllerStore`; instantiate in [controller.py:~1035](lib/iris/src/iris/cluster/controller/controller.py:1035) and pass to `ControllerTransitions`.
4. Update transitions.py to take `store: ControllerStore`; route endpoint calls through `self._store.endpoints`; keep `self._db` as temporary escape hatch.
5. Update tests: `make_controller_state` in [conftest.py:186](lib/iris/tests/cluster/controller/conftest.py:186) constructs `ControllerStore`; update [test_endpoint_registry.py](lib/iris/tests/cluster/controller/test_endpoint_registry.py).

**Phase 2 — JobStore migration**
Move the ~20 jobs/job_config/users/user_budgets queries from transitions.py into `JobStore`/`ReservationStore`. Prefer one method per call site; collapse duplicates only when obvious.

**Phase 3 — TaskStore migration**
Move ~35 task and task_resource_history queries.

**Phase 4 — TaskAttemptStore migration**
Move ~12 task_attempts queries.

**Phase 5 — WorkerStore migration**
Move ~15 worker queries + `worker_attributes` + `_attr_cache` from `ControllerDB` into `WorkerStore`. Remove `_attr_cache` / `get_worker_attributes` / `set_worker_attributes` / `remove_worker_from_attr_cache` from [db.py](lib/iris/src/iris/cluster/controller/db.py:338).

**Phase 6 — Cleanup**
DispatchQueueStore + ReservationStore remaining queries. Drop `self._db` escape hatch on `ControllerTransitions`. Confirm transitions.py has zero `self._db.transaction()` / `self._db.fetchone()` calls — only `self._store.transaction()` + store method calls.

Out of scope for now: [service.py](lib/iris/src/iris/cluster/controller/service.py) (47 queries), [controller.py](lib/iris/src/iris/cluster/controller/controller.py) (22), [checkpoint.py](lib/iris/src/iris/cluster/controller/checkpoint.py), autoscaler. They keep using `ControllerDB` directly; the layering rule "transitions → stores, never db" is a per-file invariant, not global. We can migrate them later if the pattern proves out.

## Reuse Notes

- Row dataclasses + projections in [schema.py](lib/iris/src/iris/cluster/controller/schema.py) are the return types — do not invent parallel types.
- Post-commit hook pattern (`cur.on_commit(fn)`) already powers `EndpointRegistry` — reuse it verbatim for any write-through caches inside stores.
- `ProtoCache` ([schema.py:33](lib/iris/src/iris/cluster/controller/schema.py:33)) remains where it is; stores don't need to touch it.
- Predicate helpers (`task_is_finished`, `attempt_is_terminal`, etc. in [db.py:111-185](lib/iris/src/iris/cluster/controller/db.py:111)) stay as top-level functions; stores can call them when validating.
