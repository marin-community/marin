# SQL Layer Redesign

The controller database has grown organically through 19+ migrations and ad-hoc
patterns. This document proposes a multi-stage plan to rebuild it into a
consistent, clean, and fast system.

## Problems

### 1. Inconsistent normalization

The schema mixes normalization levels. Some choices are well-motivated, others
are haphazard:

- **Good denormalization**: `priority_neg_depth`, `priority_root_submitted_ms` on
  `tasks` — avoids joins in the scheduling hot path.
- **Good denormalization**: `current_worker_id` and `current_worker_address` on
  `tasks` — avoids a JOIN through `task_attempts` in the dispatch hot path
  (`drain_dispatch`, `fail_heartbeat`, `cancel_job`). Used in 25+ code paths in
  `transitions.py`. The benchmark script (`scripts/benchmark_db_queries.py`)
  confirms this is performance-critical: the single-worker running-tasks query
  uses `WHERE t.current_worker_id = ?` directly instead of a 2-way JOIN.
- **Good denormalization (recent)**: Migrations 0016-0018 promoted
  `total_cpu_millicores`, `device_type`, `device_variant` onto `workers`, and
  `resources_proto`, `constraints_proto`, `max_task_failures`, `has_coscheduling`
  onto `jobs` — avoiding proto deserialization in scheduling and dashboard queries.
- **Missing normalization**: `worker_ids` on `slices` is a JSON array of strings
  rather than a junction table. This prevents FK enforcement and makes
  worker-to-slice lookups require full-table scans with JSON parsing.

### 2. No consistent write paths or validation

Write operations are scattered across multiple patterns:

- `ControllerTransitions` owns most mutations via `db.transaction()`, but
  `ControllerDB` itself has `ensure_user`, `set_user_role`, `delete_endpoint`,
  `delete_endpoints`, and `execute` as standalone write methods.
- `auth.py` has its own set of write functions (`create_api_key`, `touch_api_key`,
  `revoke_api_key`) that bypass `ControllerTransitions` and call `db.execute()`
  directly.
- `scaling_group.py` writes to `scaling_groups`, `slices`, and `tracked_workers`
  through its own code path.
- State transitions have no formal validation. The task state machine
  (PENDING → ASSIGNED → BUILDING → RUNNING → terminal) is enforced only by
  application logic in `transitions.py` via implicit WHERE guards. Nothing
  prevents a direct UPDATE from moving a task from SUCCEEDED back to PENDING.

### 3. Inconsistent trigger usage

Three triggers exist, each following a different pattern:

- `trg_task_attempt_active_worker`: Cross-table BEFORE INSERT validation
  (correct use of triggers).
- `trg_txn_log_retention`: AFTER INSERT cleanup with a `NOT IN (SELECT ... LIMIT
  1000)` subquery that runs on every insert — O(n) on a table that could have
  thousands of rows before cleanup. Should be periodic batch cleanup.
- `trg_task_profiles_cap`: AFTER INSERT retention with a similar pattern.

Meanwhile, state machine transitions that would genuinely benefit from trigger
guards (terminal states are final, valid state transitions) have none.

### 4. Migration cruft

- 19 migration files, several of which only add indexes (`0002`, `0004_worker`,
  `0007`, `0009`, `0010_dashboard`), suggesting indexes were added reactively
  rather than designed with query patterns in mind.
- Duplicate migration number prefixes (`0004_*`, `0010_*`, `0012_*`) indicate
  concurrent development without coordination.
- `0001_init.py` no longer matches the actual schema (it lacks columns added by
  migrations 0003-0019: `name`, `has_reservation`, `container_id`, `display_name`,
  `role`, `profile_kind`, `current_worker_id`, `current_worker_address`,
  `resources_proto`, `constraints_proto`, `max_task_failures`, etc.). Fresh
  databases get a partial schema from `0001` plus 18 incremental patches, but
  there's no test that the result matches a known-good reference.
- `0010_purge_orphaned_endpoints.py` is a data cleanup, not a schema change,
  embedded in the migration chain.

### 5. Raw SQL strings everywhere

All queries are hand-written SQL strings with no compile-time validation:

```python
cur.execute(
    "UPDATE tasks SET state = ?, finished_at_ms = COALESCE(finished_at_ms, ?), error = ? WHERE task_id = ?",
    (cluster_pb2.TASK_STATE_KILLED, now_ms, reason, task_id),
)
```

Column names, table names, and parameter counts are all unchecked. A typo is only
caught at runtime. The `sql-canonical.md` doc describes a `TABLE.c.column`
predicate DSL that was never implemented.

### 6. Model-schema coupling is manual

`@db_row_model` dataclasses duplicate column names from the schema:

```python
@db_row_model
class Job:
    job_id: JobName = db_field("job_id", JobName.from_wire)
    state: int = db_field("state", _decode_int)
    ...
```

Adding a column requires changes in three places: the migration, the row model,
and every INSERT/SELECT statement that touches the table. No test catches drift
between these.

---

## ORM evaluation

We evaluated ORMs for two possible roles: **DDL generation** (schema-as-code) and
**row decoding** (mapping `sqlite3.Row` → typed Python dataclasses). No single
tool covers both without introducing the **dual schema problem** or ORM creep.

### The dual schema problem

SA Core and Peewee solve DDL but NOT row decoding. The custom decoders (proto
deserialization, `Timestamp.from_ms`, `JobName.from_wire`) are domain-specific
transformations that no ORM handles. Adopting SA Core or Peewee for DDL creates
two parallel schema definitions that must be kept in sync:

1. **ORM schema**: `Table("jobs", Column("state", Integer), ...)` — used for DDL
2. **Row model schema**: `db_field("state", _decode_int)` — used for decoding

This is arguably worse than the status quo, where the raw SQL DDL in
`0001_init.py` is a single source of truth that can be diffed against the
`@db_row_model` classes.

cattrs/Pydantic solve decoding but not DDL, avoiding the dual-schema problem but
also not addressing schema-as-code.

### Why partial ORM adoption fails

The project previously had an ORM layer that was removed because it was hard to
maintain. The fundamental issue: once ORM Model classes exist in the codebase,
the boundary between "ORM for DDL only" and "ORM for queries too" is a social
contract with no enforcement mechanism. Peewee Model classes have `.select()`,
`.create()`, `.save()` methods. SQLAlchemy Table objects support
`select(table).where(...)`. Within months, some queries migrate to the ORM while
others stay raw, producing inconsistency worse than the starting point.

### Detailed findings

| Library | DDL | Decode | Triggers | Connection compat | ORM creep risk |
|---------|-----|--------|----------|-------------------|----------------|
| **SA Core** | Yes | No (need custom decoders) | Raw DDL escape hatch only | Needs SA engine, conflicts with our sqlite3 pool | Medium |
| **Peewee** | Yes | No (returns tuples, not sqlite3.Row) | No support | Wants to own connection, fights read pool | High |
| **cattrs** | No | Partial (verbose rename boilerplate, loses fast-path) | N/A | Compatible | None |
| **Pydantic** | No | Yes (already a dependency, `Field(alias=...)`) | N/A | Compatible | None |

**SA Core specifics**: Triggers, PRAGMAs, and `ATTACH DATABASE` all require raw
SQL escape hatches. Alembic migration autogeneration doesn't detect trigger
changes. SQLite's ALTER TABLE limitations make Alembic's batch mode fragile with
FK cascades. The dependency weight (~1MB) is significant for an embedded tool.

**Peewee specifics**: `db.execute_sql()` returns raw cursor tuples, not
`sqlite3.Row` objects. The entire decode pipeline (`decode_rows`, `_decode_row`)
relies on `sqlite3.Row.keys()` for column-name lookup. Switching would require
rewriting all decoders.

**Pydantic specifics**: `model_validate(dict(row))` with `BeforeValidator` is
the closest drop-in replacement for `@db_row_model`, and it's already a
dependency. But Pydantic validation is ~5-10x slower than the current
pre-computed decoder tuples. For the scheduling loop decoding hundreds of Task
rows per cycle, this regression must be measured before adoption.

### Recommendation: no ORM, invest in a schema registry

Keep raw sqlite3 and the custom `@db_row_model` system (~120 lines, purpose-built,
zero overhead). Build a thin schema registry (~200 lines) that is the single source
of truth for both DDL generation AND row model field mappings, eliminating the
manual coupling without introducing a new dependency.

---

## Design principles

### Where validation lives

| Level | What belongs here | Examples |
|-------|-------------------|----------|
| **CHECK constraints** | Column domain invariants | `state IN (0..9)`, `healthy IN (0,1)`, `replicas >= 1` |
| **Foreign keys** | Referential integrity | `tasks.job_id → jobs.job_id` |
| **BEFORE triggers** | Cross-table invariants that prevent invalid writes | Worker must be active+healthy before attempt assignment |
| **Application code** | State machine transitions, business rules, anything requiring protobuf deserialization | Task state machine, job completion logic, retry budget checks |

**Explicitly not in triggers**: state machine transition validation. The task
state machine has ~15 valid transitions with context-dependent rules (retry
budgets, preemption policies). Encoding this in SQL triggers would duplicate the
Python logic, be harder to test, and add ~50% overhead per UPDATE. Instead, all
state-changing writes go through `ControllerTransitions` methods that use
`WHERE state = ?` guards and check `rowcount`.

### Denormalization policy

Denormalize when ALL of these hold:
1. The denormalized data is used in a hot path (scheduling, dispatch, heartbeat).
2. Removing the denormalization would require a JOIN that the benchmark shows is
   measurably slower.
3. There is a single, well-defined write path that updates both the canonical and
   denormalized copies atomically (within the same transaction).

Current denormalizations that meet this bar:
- `tasks.priority_*` fields (scheduling index scan)
- `tasks.current_worker_id/address` (dispatch, heartbeat failure, cancel)
- `jobs.num_tasks` (dashboard, state recomputation)
- `jobs.max_task_failures`, `jobs.resources_proto`, etc. (scheduling, dashboard)
- `workers.total_*`, `workers.device_type/variant` (scheduling constraint matching)

### FK and CASCADE policy

- All child tables CASCADE on DELETE from parent. This is already the case and is
  correct — deleting a job should atomically remove its tasks, attempts, endpoints,
  and dispatch entries.
- Every FK child column must have an index (for efficient cascade scans). Audit
  existing indexes against this rule.
- `tasks.current_worker_id` uses SET NULL on worker delete (migration 0019) rather
  than CASCADE, because deleting a worker should not delete the task.
- FK enforcement stays ON. The integrity benefit far outweighs the per-row lookup
  cost (single B-tree probe with PK index).

### Trigger policy

- **Keep**: `trg_task_attempt_active_worker` (cross-table invariant, correct use).
- **Replace**: `trg_txn_log_retention` and `trg_task_profiles_cap` with periodic
  application-level batch cleanup (every N inserts or every 60s). The `NOT IN
  (SELECT ... ORDER BY ... LIMIT)` pattern is O(n) per insert.
- **Add**: `trg_terminal_state_immutable` — a BEFORE UPDATE trigger on `tasks` and
  `jobs` that prevents transitions out of terminal states. This is a low-cost
  safety net (single integer comparison) that catches application bugs.

### Transaction discipline

All writes must go through one of:
1. `ControllerTransitions` methods (job/task/worker state)
2. `ControllerAuth` methods (API keys, secrets)
3. `ScalingGroupManager` methods (scaling state)

Each module owns specific tables and no other code may write to those tables.
`ControllerDB.execute()` as a general-purpose write API should be removed;
callers should use the appropriate module.

---

## Multi-stage plan

### Stage 0: Schema registry, projections, and canonical schema (foundation)

**Goal**: Single source of truth for table definitions. Eliminate hand-maintained
column strings and manual row-model/schema coupling. Enable on-the-fly typed
projections.

#### Current pain

Today there are three manually-synchronized artifacts per entity:

1. **Row model class** (`JobRow`, `JobDetail`, `TaskRow`, `TaskDetail`, etc.) —
   each duplicates column names via `db_field("column_name", decoder)`.
2. **Column strings** (`JOB_ROW_COLUMNS`, `JOB_LISTING_COLUMNS`,
   `TASK_ROW_COLUMNS`, `WORKER_ROW_COLUMNS`) — hand-written SELECT lists that
   must match the row model fields.
3. **Schema DDL** (in `0001_init.py` + 18 migrations) — the actual columns.

Adding a column requires touching all three. The `JOB_LISTING_COLUMNS` vs
`JOB_ROW_COLUMNS` distinction exists solely to drop one column
(`constraints_proto`) from listing queries — an entire constant for one field.
And `q.raw()` queries return untyped `Row` objects where field access is
stringly-typed (`row.worker_id`, `row.cnt`) with no IDE support.

#### Design

Create `lib/iris/src/iris/cluster/controller/schema.py` with a schema registry
that unifies DDL, decode metadata, and projection generation:

```python
@dataclass(frozen=True)
class Column:
    name: str
    sql_type: str
    constraints: str = ""
    # Decode metadata — used to generate row models and decoders
    python_name: str | None = None   # field name if different from column name
    python_type: type = object        # type hint for generated dataclass
    decoder: Callable = _identity
    default: Any = MISSING
    # Cost hints — used by projection system
    expensive: bool = False           # True for proto blobs

@dataclass(frozen=True)
class Table:
    name: str
    alias: str                        # e.g. "j" for jobs
    columns: tuple[Column, ...]
    table_constraints: tuple[str, ...] = ()
    indexes: tuple[str, ...] = ()
    triggers: tuple[str, ...] = ()

    def ddl(self) -> str:
        """Generate CREATE TABLE + CREATE INDEX + CREATE TRIGGER SQL."""
        ...

    def projection(self, *column_names: str) -> Projection:
        """Create a typed projection over a subset of columns."""
        ...

    def select_clause(self, *column_names: str) -> str:
        """Generate 'alias.col1, alias.col2, ...' for a SELECT."""
        ...

JOBS = Table("jobs", "j", columns=(
    Column("job_id", "TEXT", "PRIMARY KEY",
           python_type=JobName, decoder=JobName.from_wire),
    Column("state", "INTEGER", "NOT NULL",
           python_type=int, decoder=_decode_int),
    Column("request_proto", "BLOB", "NOT NULL",
           python_name="request",
           python_type=cluster_pb2.Controller.LaunchJobRequest,
           decoder=_proto_decoder(LaunchJobRequest),
           expensive=True),
    Column("resources_proto", "BLOB", "",
           python_type=cluster_pb2.ResourceSpecProto | None,
           decoder=_nullable(_proto_decoder(ResourceSpecProto)),
           default=None),
    Column("max_task_failures", "INTEGER", "NOT NULL DEFAULT 0",
           python_type=int, decoder=_decode_int, default=0),
    # ... all columns
))
```

#### On-the-fly typed projections

The key capability: generate typed result decoders from column subsets at
import time, rather than hand-maintaining `JobRow` / `JobDetail` / etc.

```python
class Projection(Generic[T]):
    """A typed subset of a Table's columns with a pre-compiled decoder."""

    def __init__(self, table: Table, columns: tuple[Column, ...]):
        self.table = table
        self.columns = columns
        # Pre-compute the same decoder tuples as @db_row_model
        self._names = tuple(c.python_name or c.name for c in columns)
        self._db_columns = tuple(c.name for c in columns)
        self._decoders = tuple(c.decoder for c in columns)
        # Generate a frozen dataclass at import time
        self._row_cls = _make_row_class(table.name, columns)
        # Pre-compute SELECT column string
        self._select = ", ".join(f"{table.alias}.{c.name}" for c in columns)

    @property
    def select_clause(self) -> str:
        return self._select

    def decode(self, rows: Iterable[sqlite3.Row]) -> list[T]:
        """Decode rows using the same fast-path as decode_rows()."""
        ...

    def decode_one(self, rows: Iterable[sqlite3.Row]) -> T | None:
        ...

# Define projections at module level — evaluated once at import time.

# Full row with proto blobs (detail views, job submission)
JobDetail = JOBS.projection("job_id", "request_proto", "state",
    "submitted_at_ms", "root_submitted_at_ms", "started_at_ms",
    "finished_at_ms", "scheduling_deadline_epoch_ms", "error",
    "exit_code", "num_tasks", "is_reservation_holder",
    "has_reservation", "name", "depth")

# Scalar-only (scheduling, dashboard lists)
JobRow = JOBS.projection("job_id", "state", "submitted_at_ms",
    "root_submitted_at_ms", "started_at_ms", "finished_at_ms",
    "scheduling_deadline_epoch_ms", "error", "exit_code",
    "num_tasks", "is_reservation_holder", "has_reservation",
    "name", "depth", "resources_proto", "constraints_proto",
    "has_coscheduling", "coscheduling_group_by",
    "scheduling_timeout_ms", "max_task_failures")

# Dashboard listing (no constraints blob)
JobListing = JOBS.projection("job_id", "state", "submitted_at_ms",
    "root_submitted_at_ms", "started_at_ms", "finished_at_ms",
    "scheduling_deadline_epoch_ms", "error", "exit_code",
    "num_tasks", "is_reservation_holder", "has_reservation",
    "name", "depth", "resources_proto", "has_coscheduling",
    "coscheduling_group_by", "scheduling_timeout_ms",
    "max_task_failures")

# Minimal for scheduling hot path
TaskScheduling = TASKS.projection("task_id", "job_id", "state",
    "current_attempt_id", "failure_count", "preemption_count",
    "max_retries_failure", "max_retries_preemption", "submitted_at_ms")
```

#### Convenience shortcuts

For truly ad-hoc queries (aggregates, JOINs across tables), the registry
provides a lightweight alternative to `q.raw()` with stringly-typed access:

```python
# Instead of:
rows = q.raw(
    "SELECT j.user_id, t.state, COUNT(*) as cnt FROM ...",
    decoders={"cnt": int}
)
for row in rows:
    user = row.user_id   # no type checking, no IDE completion

# The registry can generate a one-off decoder:
LiveStats = adhoc_projection(
    ("user_id", str),
    ("state", int),
    ("cnt", int),
)
rows = LiveStats.decode(q.fetchall("SELECT j.user_id, t.state, COUNT(*) as cnt FROM ..."))
for row in rows:
    user = row.user_id   # typed, IDE-completable
```

This replaces the `decoders={}` dict pattern with typed results while keeping
raw SQL for the query itself.

#### What projections replace

| Before | After |
|--------|-------|
| `class JobRow` (27-line `@db_row_model`) | `JobRow = JOBS.projection(...)` (one line) |
| `class JobDetail` (50-line `@db_row_model`) | `JobDetail = JOBS.projection(...)` (one line) |
| `JOB_ROW_COLUMNS` (hand-written string) | `JobRow.select_clause` (auto-generated) |
| `JOB_LISTING_COLUMNS` (same minus one col) | `JobListing.select_clause` |
| `decode_rows(JobRow, rows)` | `JobRow.decode(rows)` |
| `q.raw(..., decoders={"cnt": int})` | `adhoc_projection(("cnt", int)).decode(...)` |

Methods and computed properties that currently live on the `@db_row_model`
classes (`is_finished()`, `can_be_scheduled()`, `scheduling_deadline`, etc.)
move to standalone functions or Protocol-based mixins that accept the
generated row type. Several of these were already extracted in the recent
refactor (`task_is_finished`, `task_can_be_scheduled`, `worker_available_*`).

#### Validation at import time

Projections validate column names against the Table definition when they're
created (at module import). A typo like `JOBS.projection("staet", ...)` raises
`KeyError` immediately, not at query time. This is the compile-time safety that
raw SQL strings lack.

#### Performance

The projection decoder uses the same pre-computed `(name, column, decoder)`
tuple pattern as the current `decode_rows`. The `_make_row_class` function
generates a frozen `@dataclass` with `__slots__` at import time. Per-row
decoding overhead is identical to today: one dict-comprehension per row in
the fast path.

#### Tasks

1. Implement `schema.py` with `Table`, `Column`, `Projection`,
   `adhoc_projection`, and `generate_full_ddl()`.

2. Define all tables (JOBS, TASKS, WORKERS, TASK_ATTEMPTS, ENDPOINTS, etc.)
   with full column metadata.

3. Replace hand-maintained `@db_row_model` classes and `*_COLUMNS` strings with
   projections. Keep computed-property logic as standalone functions.

4. Write `test_schema_consistency`:
   (a) create a fresh `:memory:` DB via `generate_full_ddl()`,
   (b) create another via running all migrations,
   (c) assert both produce identical `sqlite_master` schemas (normalized).

5. Regenerate `0001_init.py` from `generate_full_ddl()` so fresh databases get
   the complete current schema in one step.

**Files changed**: new `schema.py`, modified `db.py` (projections replace row
models and column strings), modified callers in `controller.py`, `service.py`,
`transitions.py` (use `Projection.decode()` and `.select_clause`), new test,
updated `0001_init.py`.

**Risk**: Low. No runtime behavior changes. The decode fast-path is preserved.
Pure refactoring with import-time validation as a bonus.

---

### Stage 1: Write path consolidation

**Goal**: Every table has exactly one owner module. No more ad-hoc writes.

**Tasks**:

1. **Audit all write sites**. Categorize every `INSERT`, `UPDATE`, `DELETE` by
   table and calling module:

   | Table | Current writers | Target owner |
   |-------|----------------|--------------|
   | `users` | `ControllerDB.ensure_user`, `transitions.submit_job` | `ControllerTransitions` |
   | `jobs`, `tasks`, `task_attempts` | `transitions.py` | `ControllerTransitions` |
   | `workers`, `worker_attributes` | `transitions.py` | `ControllerTransitions` |
   | `dispatch_queue` | `transitions.py` | `ControllerTransitions` |
   | `endpoints` | `transitions.py`, `ControllerDB.delete_endpoint(s)` | `ControllerTransitions` |
   | `txn_log`, `txn_actions` | `transitions.py` | `ControllerTransitions` |
   | `reservation_claims` | `transitions.py` | `ControllerTransitions` |
   | `worker_task_history`, `worker_resource_history` | `transitions.py` | `ControllerTransitions` |
   | `meta` | `ControllerDB.next_sequence` | `ControllerTransitions` (via cursor) |
   | `scaling_groups`, `slices`, `tracked_workers` | `scaling_group.py` | `ScalingGroupManager` |
   | `api_keys`, `controller_secrets` | `auth.py` | `ControllerAuth` |
   | `task_profiles` | `db.py::insert_task_profile` | `ControllerTransitions` |
   | `logs` | `log_store.py` | `LogStore` |

2. **Move orphaned writes into their owner**:
   - `ControllerDB.ensure_user` → `ControllerTransitions.ensure_user`
   - `ControllerDB.set_user_role` → `ControllerTransitions.set_user_role`
   - `ControllerDB.delete_endpoint(s)` → `ControllerTransitions.delete_endpoint(s)`
   - `db.py::insert_task_profile` → `ControllerTransitions.insert_task_profile`

3. **Remove `ControllerDB.execute()`** as a public API. Replace with a private
   `_execute` for internal use by the migration system only.

4. **Add table ownership constants** to each module:
   ```python
   # transitions.py
   _OWNED_TABLES = frozenset({"jobs", "tasks", "task_attempts", "workers", ...})
   ```
   Add a debug-mode assertion in `TransactionCursor.execute()` that validates
   the target table is owned by the calling module (parse the SQL for the table
   name). This is a development-time safety net, not a production check.

**Files changed**: `db.py`, `transitions.py`, `auth.py`, `service.py`,
`controller.py`, `scaling_group.py`.

**Risk**: Medium. Moves code between modules but doesn't change SQL or logic.
Test with existing E2E suite.

---

### Stage 2: Schema cleanup

**Goal**: Fix remaining normalization issues and add missing constraints.

**Tasks**:

1. **Replace `slices.worker_ids` JSON array with a junction table**:
   ```sql
   CREATE TABLE slice_workers (
       slice_id TEXT NOT NULL REFERENCES slices(slice_id) ON DELETE CASCADE,
       worker_id TEXT NOT NULL,
       PRIMARY KEY (slice_id, worker_id)
   );
   CREATE INDEX idx_slice_workers_worker ON slice_workers(worker_id);
   ```
   Migration: parse existing JSON arrays and INSERT into junction table, then
   drop the `worker_ids` column.

2. **Consolidate state enums**. Job and task states are bare integers with no
   CHECK constraint in the schema. Add:
   ```sql
   CHECK (state IN (0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
   ```
   for both tables (enumerate all valid states from the protobuf definition).

3. **Index audit**. Run `EXPLAIN QUERY PLAN` on all queries in
   `benchmark_db_queries.py` against a production-sized checkpoint. Identify:
   - Missing indexes (full table scans where an index would help)
   - Unused indexes (present but never selected by the query planner)
   - Redundant indexes (one index is a prefix of another)

   The benchmark script already covers scheduling, dashboard, and heartbeat
   query groups with p50/p95 measurements.

4. **Regenerate `0001_init.py`** from `schema.py::generate_full_ddl()` to
   include all columns from migrations 0002-0019. This ensures fresh databases
   get the complete schema in one step.

**Files changed**: new migration(s), `schema.py`, `scaling_group.py` (junction
table), updated `0001_init.py`.

**Risk**: Medium. Schema changes require careful migration testing. Run
the Stage 0 schema-consistency test to validate.

---

### Stage 3: State machine hardening

**Goal**: Make invalid state transitions impossible, not just unlikely.

**Tasks**:

1. **Add terminal-state-immutability triggers**:
   ```sql
   CREATE TRIGGER trg_task_terminal_immutable
   BEFORE UPDATE OF state ON tasks
   FOR EACH ROW
   WHEN OLD.state IN (5, 6, 7, 8, 9)  -- terminal states
   AND NEW.state != OLD.state
   BEGIN
     SELECT RAISE(ABORT, 'cannot transition from terminal task state');
   END;

   CREATE TRIGGER trg_job_terminal_immutable
   BEFORE UPDATE OF state ON jobs
   FOR EACH ROW
   WHEN OLD.state IN (3, 4, 5, 6, 7)  -- terminal job states
   AND NEW.state != OLD.state
   BEGIN
     SELECT RAISE(ABORT, 'cannot transition from terminal job state');
   END;
   ```

   These are single-integer comparisons with zero overhead on non-terminal rows.
   They catch application bugs (e.g., a heartbeat handler accidentally resetting
   a succeeded task) that would otherwise cause silent corruption.

2. **Replace retention triggers with batch cleanup**:

   Remove `trg_txn_log_retention` and `trg_task_profiles_cap`. Add a
   `ControllerTransitions.prune_retention_tables()` method called from the
   controller's periodic maintenance loop (which already runs `prune_old_data`):

   ```python
   def prune_retention_tables(self) -> None:
       with self._db.transaction() as cur:
           # Keep newest 1000 txn_log entries
           cur.execute(
               "DELETE FROM txn_log WHERE id <= ("
               "  SELECT id FROM txn_log ORDER BY id DESC LIMIT 1 OFFSET 1000"
               ")"
           )
           # Keep newest 10 profiles per (task_id, profile_kind)
           cur.execute(
               "DELETE FROM task_profiles WHERE id IN ("
               "  SELECT p.id FROM task_profiles p"
               "  WHERE (SELECT COUNT(*) FROM task_profiles p2"
               "         WHERE p2.task_id = p.task_id AND p2.profile_kind = p.profile_kind"
               "         AND p2.id > p.id) >= 10"
               ")"
           )
   ```

   This runs once per maintenance cycle (~60s) instead of on every insert.

3. **Formalize the task state machine in code**. Create a state transition table:

   ```python
   VALID_TASK_TRANSITIONS: frozenset[tuple[int, int]] = frozenset({
       (TASK_STATE_PENDING, TASK_STATE_ASSIGNED),
       (TASK_STATE_ASSIGNED, TASK_STATE_BUILDING),
       (TASK_STATE_ASSIGNED, TASK_STATE_RUNNING),
       (TASK_STATE_BUILDING, TASK_STATE_RUNNING),
       (TASK_STATE_RUNNING, TASK_STATE_SUCCEEDED),
       (TASK_STATE_RUNNING, TASK_STATE_FAILED),
       (TASK_STATE_RUNNING, TASK_STATE_WORKER_FAILED),
       # Retry transitions
       (TASK_STATE_FAILED, TASK_STATE_PENDING),
       (TASK_STATE_WORKER_FAILED, TASK_STATE_PENDING),
       # Kill from any non-terminal state
       (TASK_STATE_PENDING, TASK_STATE_KILLED),
       (TASK_STATE_ASSIGNED, TASK_STATE_KILLED),
       (TASK_STATE_BUILDING, TASK_STATE_KILLED),
       (TASK_STATE_RUNNING, TASK_STATE_KILLED),
       # Unschedulable from pending
       (TASK_STATE_PENDING, TASK_STATE_UNSCHEDULABLE),
   })
   ```

   Add a helper used by all transition code:
   ```python
   def assert_valid_transition(entity: str, current: int, new: int) -> None:
       table = VALID_TASK_TRANSITIONS if entity == "task" else VALID_JOB_TRANSITIONS
       if (current, new) not in table:
           raise ValueError(f"Invalid {entity} transition: {current} → {new}")
   ```

**Files changed**: new migration (triggers), `transitions.py` (state machine
table, remove retention trigger setup), `controller.py` (call
`prune_retention_tables`).

**Risk**: Low-medium. Terminal-immutability triggers are purely defensive. If any
existing code path violates them, that's a bug we want to find. Run full E2E
suite to flush out any violations.

---

### Stage 4: Read path cleanup

**Goal**: Consistent, type-safe read patterns.

**Tasks**:

1. **Implement column-reference constants** from the schema registry:

   ```python
   # Generated from schema.py Table definitions
   class JobColumns:
       job_id = "j.job_id"
       state = "j.state"
       ...
   ```

   This gives import-time validation (attribute access fails if the column
   doesn't exist) without the overhead of a query builder. Raw SQL remains
   the query language; column references just avoid string typos.

2. **Standardize the read snapshot pattern**. Currently `service.py` uses
   `db.read_snapshot()` (pooled, no write lock) while `controller.py` sometimes
   uses `db.snapshot()` (write lock). Document when to use which:
   - `read_snapshot()`: Dashboard/RPC reads that tolerate slightly stale data.
   - `snapshot()`: Reads that inform an immediately-following write transaction
     (scheduling decisions, heartbeat processing). These need write-lock
     consistency to prevent TOCTOU bugs.

3. **Remove `ControllerDB.fetchall()` and `ControllerDB.fetchone()`**. These
   are convenience methods that acquire the write lock for reads, which is
   incorrect in the read-snapshot architecture. All reads should go through
   `snapshot()` or `read_snapshot()`.

**Files changed**: `schema.py` (column constants), `db.py` (remove
fetchall/fetchone), `service.py`, `controller.py`.

**Risk**: Low. Read-only refactoring. Existing tests validate correctness.

---

### Stage 5: Migration consolidation

**Goal**: Clean migration history, tested migration path.

**Tasks**:

1. **Regenerate `0001_init.py`** from `schema.py::generate_full_ddl()`. This
   produces the complete current schema for fresh databases.

2. **Squash migrations 0002-0019 into a compatibility marker**. For databases
   that already have `0001` applied, a new `0002_squash.py` migration:
   - Checks whether the current schema matches the fresh `0001` output.
   - If not (old database), applies the necessary ALTER TABLE / CREATE INDEX
     statements to bring it up to date.
   - Records all squashed migration names in `schema_migrations` so they
     won't be re-run.

3. **Add migration numbering policy**: sequential integers with no duplicates.
   Add a CI check that validates migration filenames are unique and sequential.

4. **Add migration test suite**:
   - `test_fresh_schema_matches_migrated`: Stage 0's consistency test, kept
     permanently.
   - `test_migrations_are_idempotent`: Run each migration twice on the same DB.
   - `test_migration_filenames_sequential`: CI lint for naming convention.

5. **Move data-only migrations to a separate system**. `0010_purge_orphaned_endpoints`
   is a data cleanup, not a schema change. Future data fixes should be separate
   scripts, not migrations.

**Files changed**: `0001_init.py` (regenerated), new `0002_squash.py`, old
migrations archived to `migrations/archived/`, new tests.

**Risk**: Medium. Must be tested against a snapshot of a production database
to verify the squash migration correctly handles existing state.

---

## Implementation order and dependencies

```
Stage 0 (schema registry)
  ├─→ Stage 1 (write consolidation)   ← can run in parallel with Stage 2
  └─→ Stage 2 (schema cleanup)
        └─→ Stage 3 (state machine hardening)
              └─→ Stage 4 (read path cleanup)
                    └─→ Stage 5 (migration consolidation)
```

Stages 1 and 2 can proceed in parallel after Stage 0. Stage 5 should be last
because it squashes all schema changes from earlier stages.

Each stage is independently shippable and testable. If we stop after Stage 1,
we still have a cleaner write path. If we stop after Stage 3, we have a hardened
state machine. The stages are designed so that partial completion still improves
the system.

## Performance considerations

- **Scheduling hot path** (runs every ~1s): `SELECT` pending tasks ordered by
  priority, `SELECT` healthy workers with attributes, `INSERT` task_attempts,
  `UPDATE` tasks/workers. This path must stay under 50ms for 1000 tasks.
  The benchmark script (`scripts/benchmark_db_queries.py`) covers this with
  `_schedulable_tasks`, `healthy_active_workers_with_attributes`,
  `_building_counts`, and `_jobs_by_id` benchmarks.

- **Dispatch hot path** (runs per-worker every heartbeat): `drain_dispatch`
  queries `tasks.current_worker_id` directly — this denormalization avoids
  the 2-way JOIN through `task_attempts` that the benchmark measures at
  significantly higher latency for the per-worker-loop case.

- **Heartbeat hot path** (runs per-worker every ~5s): `UPDATE` worker health,
  `UPDATE` task states, `INSERT` resource history. Stage 3's retention trigger
  removal reduces overhead on the history INSERT. The benchmark's "phase3
  two-pass" test measures the real `apply_heartbeats_batch` path.

- **Dashboard reads** (concurrent with scheduling): Already use `read_snapshot()`
  with the connection pool. Stage 4 formalizes this pattern and removes
  write-lock acquisition for reads.

## What we're not doing

- **Adopting an ORM.** We investigated SQLAlchemy Core, Peewee, cattrs, and
  Pydantic. The dual-schema problem (ORM for DDL, custom decoders for rows) makes
  partial adoption worse than the status quo. The project previously had an ORM and
  removed it because it was hard to maintain. A thin schema registry (~200 LOC)
  solves the DDL+decode coupling without a new dependency.

- **Removing `current_worker_id` denormalization.** This was initially proposed
  but is incorrect — the column is used in 25+ hot-path queries in
  `transitions.py` (dispatch, heartbeat failure, cancel, direct provider sync).
  The benchmark confirms the JOIN alternative is measurably slower.

- **Switching to PostgreSQL or DuckDB**. SQLite's single-file deployment,
  crash recovery, and zero-config operation are essential for the controller's
  operational model (backup via `sqlite3.backup()`, restore by file copy).

- **Adding a generic query builder**. Complex queries (CTEs, recursive subtree
  walks, aggregate job state) stay as raw SQL because they are write-once,
  read-many code paths where a query builder adds verbosity without safety.

- **Backward compatibility shims**. Per AGENTS.md policy, we update all call
  sites rather than maintaining compatibility layers.
