# Iris Controller Data Layer Refactor (SQLAlchemy Core)

**Date:** 2026-05-11
**Author:** russell.power@gmail.com
**Status:** Draft — proposal

---

## 1. Summary

Adopt **SQLAlchemy Core 2.x** as the engine, schema, query-construction, and connection-management layer for the Iris controller. Keep our hand-rolled migration system. Add two small custom pieces on top: a `Tx` wrapper that exposes post-commit hooks for atomic write-through cache updates, and a `Projection` class for the two tiny tables (`endpoints`, `worker_attributes`) that need in-memory write-through caches.

Today's data layer has five mechanisms accumulated to work around SQL performance issues: hand-rolled column-subset `Projection` constants, a bytes-keyed `ProtoCache`, an `EndpointStore` write-through dict, a `_attr_cache` dict bolted onto `ControllerDB`, and partial indexes serving as planner-level "poor man's materialized views." Each works in isolation; together they form a five-way decision tree for "how do I read entity X?" and writers must remember which caches their writes invalidate.

SA Core directly handles the parts that motivated most of the proliferation:

- **Composable column selection.** Today's hand-maintained `JOB_DETAIL_PROJECTION = Projection(JOBS, _job_detail_cols, ...)` constants disappear. Call sites write `select(jobs).where(...)` inline.
- **Aggregates, recursive CTEs, dynamic paging.** `func.count()`, `func.sum()`, `.cte(recursive=True)`, `.order_by()` / `.limit()` / `.offset()` are all first-class. The dashboard "list jobs with state filter, ordered, paginated" pattern becomes a one-line `select` composition.
- **Transparent column-level decoding (and caching).** `TypeDecorator` puts the decoder and the bytes-keyed memo on the column type itself — readers of `job_config.config_proto` get a fully-decoded, cached `JobConfigProto` with zero glue at the read site.
- **Statement compilation cache.** SA caches compiled SQL automatically. We don't have this today.
- **Connection pool + transaction management** as a well-tested standard library, replacing the hand-rolled 32-reader + 1-writer + RLock + IMMEDIATE-transaction machinery in today's `db.py`.

The two things SA Core does *not* solve — bytes-keyed proto memoization (handled by `TypeDecorator`, but the LRU policy is ours) and write-through tiny-table caches — are ~50 lines each, scoped to the column type or to a `Projection` class.

**Net effect**

| | Today | **After refactor** |
|---|---|---|
| Custom concepts in data layer | 5 (`Projection`, `ProtoCache`, `EndpointStore`, `_attr_cache`, partial indexes) | **2** (`Projection`, `Tx`) on top of SA Core |
| Custom LOC | ~2700 (`schema.py` + `stores.py` + `db.py` data-layer parts) | **~700–900** (schema declarations + 2 projection classes + Tx wrapper) |
| Composability for ad-hoc queries (dashboard, aggregates, paging) | Manual `Projection` constants; painful for one-offs | Native — write `select(...).where(...).limit(...)` inline |
| Read perf | Manual slim projections; partial indexes | Same SQL emitted; SA Core compilation cache helps cold queries |
| Write-through atomicity | `tx.on_commit` under write lock | Same, via `Tx.register` wrapped in `write_transaction()` |

**The mental model becomes:** SA Core handles "talk to SQL." Two small classes handle "talk to in-memory caches." Call sites use SA-idiomatic `select()` directly. Hand-tuned hot-path queries get named constants in `reads/<area>.py`; one-off and ad-hoc queries are inline.

---

## 2. Background

### 2.1 The five mechanisms today

**`Projection` (`schema.py:316–456`).** Compiled view of a `Table`'s columns. Resolves a column-name tuple at import time, validates types, and produces a tuple of decoder callables. Used by `JOB_DETAIL_PROJECTION`, `JOB_RESERVATION_PROJECTION`, `TASK_ROW_PROJECTION`, etc. Hand-maintained constants; adding a column to a query means editing the constant. Painful for one-off and dashboard queries.

**`ProtoCache` (`schema.py:34–66`).** Bounded LRU (8192 entries, 25% eviction batches) keyed on raw blob bytes. Wrapped into specific column decoders via `cached=True` on `Column`. A content-addressed identity map for immutable protos. Solves "don't re-decode the same proto blob 100 times per scheduler tick."

**`EndpointStore` (`stores.py:95–337`).** Tiny read-mostly table (hundreds of rows). Reads never touch SQL; three in-memory dicts (`_by_id`, `_by_name`, `_by_task`) are populated at startup via `_load_all()` and updated via post-commit hooks. Eliminated dashboard CPU dominated by `ListEndpoints` walking the WAL.

**`ControllerDB._attr_cache` (`db.py:329–379`).** Same shape as `EndpointStore` but bolted directly onto `ControllerDB`. Lazy-populated dict of `{worker_id: {attr_name: attr_value}}`. Written through via `set_worker_attributes` called from `on_commit` hooks in `transitions.py`.

**Partial indexes (`migrations/0045_*.py`).** Not a Python cache, but a planner-level mechanism playing the same role as a materialized view:

```sql
CREATE INDEX idx_task_attempts_live_workerbound
ON task_attempts(worker_id)
WHERE worker_id IS NOT NULL AND finished_at_ms IS NULL;
```

Drives `resource_usage_by_worker` from ~1k live rows instead of 24k jobs. 350 ms → 6.5 ms. **These stay** — they are an index optimization, not a Python-layer caching strategy, and SA generates compatible SQL.

### 2.2 What's wrong with this

Performance is fine. The problem is cognitive: five mechanisms means a new contributor meets five idioms before they ship one query, and writers must remember which caches their writes invalidate. Specific pain:

- Adding a new query means editing or adding a `Projection` constant in `schema.py` — even for one-off dashboard queries.
- `ProtoCache` is global, opt-in via a `cached=True` flag on `Column`, and threaded through the `Projection` decode path. The connection between "the column was declared cached" and "the cache is consulted on read" is non-local.
- `EndpointStore`-style write-through is the same pattern as `_attr_cache` but implemented twice in different shapes.
- The hand-rolled connection pool (`db.py:407`), `TransactionCursor` wrapper, and `on_commit` hook list reimplement primitives SA Core ships natively.

### 2.3 The audit

Findings that informed the design (full source: codebase audit of `stores.py`, `db.py`, `lib/iris/tests/`):

- **No write method touches more than 3 tables.** The largest is `WorkerStore.remove` (3 tables: `workers`, `task_attempts`, `tasks`). Module-level write functions handle this scale trivially.
- **Zero tests mock Store classes** (`grep mock.*Store` / `patch.*Store` over `lib/iris/tests/`: no matches). Tests construct a real `ControllerDB`. Refactor is test-safe.
- **`_attr_cache` and `EndpointStore` are the only two write-through caches today.** Refactor needs to preserve their atomicity contract (post-commit hooks fire under the write lock).
- **FK cascades silently invalidate caches.** `WorkerStore.remove` deletes from `workers`; FK cascade deletes from `worker_attributes` (the `_attr_cache`-backed table). Today this is handled by an explicit `_attr_cache.remove_worker(...)` hook in `transitions.py`. Refactor needs to address this generally.
- **`TransactionCursor` doesn't track written tables.** Auto-invalidation via SQL parsing was considered and rejected in favor of explicit `@writes_to(...)` declarations + a startup-time owned-table check.

---

## 3. Goals and Non-Goals

### Goals

- **Adopt SA Core 2.x** as the canonical engine, schema, and query-construction layer.
- **Eliminate manual column-subset `Projection`s.** Hand-written `JOB_DETAIL_PROJECTION = Projection(JOBS, _job_detail_cols, ...)` constants disappear. Call sites write `select(jobs).where(...)`.
- **Hoist column-level caching into `TypeDecorator`s.** `ProtoCache` becomes a per-column `CachedProto(message_cls)` type, transparent to call sites.
- **Preserve write-through cache semantics** for `endpoints` and `worker_attributes`, atomic w.r.t. transaction commits and the write lock.
- **Preserve hot-path perf.** Benchmarks gate the migration.
- **One stacked PR.** ~12 commits, each independently testable, end-to-end smoke before merge.
- **Common queries in a common library set.** `reads/<area>.py` holds named selects and helper functions for hot/shared queries. Ad-hoc and one-off queries live inline at the call site.

### Non-goals

- **No SA ORM** (mapped classes, identity map, Session UoW). Per the SA evaluation, the perf overhead and threading-model friction don't pay back.
- **No Alembic.** Hand-rolled `.py` migrations stay. Alembic autogen on SQLite has known footguns (`drop_column` generates wrong SQL; needs `batch_alter_table`), and we don't have declarative ORM models for Alembic to diff against.
- **No Postgres migration.** SQLite + WAL is right for a single-process controller.
- **No async.** Sync threading model stays.
- **No automatic SQL-parsing invalidation.** Explicit `@writes_to(...)` declarations + startup-time owned-table check (see §4.8).
- **No new query DSL.** SA Core's `select()` *is* the DSL.

---

## 4. Architecture

### 4.1 Component diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          ControllerDB                                   │
│  ┌─────────────────────┐    ┌────────────────────────────────────────┐  │
│  │   sqlalchemy.Engine │    │ Write lock (RLock)                     │  │
│  │  (SQLite, WAL, 32+1)│    │ Projections registry                   │  │
│  └──────────┬──────────┘    └────────────────────────────────────────┘  │
│             │                                                           │
│       ┌─────┴──────────────────┐                                        │
│       │                        │                                        │
│  read_snapshot()         write_transaction()                            │
│       │                        │                                        │
│       ▼                        ▼                                        │
│   ┌───────┐                ┌───────┐                                    │
│   │  Tx   │                │  Tx   │  ── holds write lock              │
│   │ (RO)  │                │  (RW) │     across commit + hooks          │
│   └───┬───┘                └───┬───┘                                    │
│       │                        │                                        │
└───────┼────────────────────────┼────────────────────────────────────────┘
        │                        │
        ▼                        ▼
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  reads/<area>.py             writes/<entity>.py    projections/      │
│  ──────────────────          ──────────────────    ──────────────    │
│  Named SA Core selects       @writes_to(...)       Owns in-memory    │
│  (hot/shared queries) +      def insert_job(tx,    dicts.            │
│  helpers for composite       …):                   Write-through:    │
│  reads.                          tx.execute(           SQL via tx,   │
│                                  insert(…))           dict update    │
│  Ad-hoc selects also           …                      via tx.register│
│  written inline at call                            Reads: dict.      │
│  sites — both are fine.                            Lifecycle:        │
│                                                    rehydrate().      │
└──────────────────────────────────────────────────────────────────────┘
                        ▲
                        │
                        │ All of the above use:
                        │
┌──────────────────────────────────────────────────────────────────────┐
│  schema.py                                                           │
│  ──────────                                                          │
│  metadata = MetaData()                                               │
│  jobs = Table("jobs", metadata, Column(...), ...)                    │
│  TypeDecorators: JobNameType, TimestampType, CachedProto(MsgCls)     │
│  Tables, indexes, partial indexes declared here.                     │
└──────────────────────────────────────────────────────────────────────┘
```

### 4.2 Engine, `Tx`, and transaction lifecycle

The engine is configured once in `ControllerDB.__init__`:

```python
# db.py
def _make_engine(db_path: Path) -> Engine:
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False, "timeout": 5.0},
        poolclass=QueuePool,
        pool_size=32,           # read connections
        max_overflow=4,
        future=True,
    )

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_conn, _):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode = WAL")
        cur.execute("PRAGMA synchronous = NORMAL")
        cur.execute("PRAGMA busy_timeout = 5000")
        cur.execute("PRAGMA foreign_keys = ON")
        cur.execute("PRAGMA cache_size = -65536")
        cur.close()

    return engine
```

`Tx` is a 30-line wrapper:

```python
class Tx:
    """Wraps a SA Connection. Adds post-commit hook registration."""
    def __init__(self, conn: Connection):
        self.conn = conn
        self._hooks: list[Callable[[], None]] = []

    def execute(self, stmt, params=None):
        return self.conn.execute(stmt, params or {})

    def executemany(self, stmt, params_list):
        return self.conn.execute(stmt, params_list)

    def register(self, hook: Callable[[], None]) -> None:
        """Hook fires once, after commit, under the write lock."""
        self._hooks.append(hook)

    def _fire_hooks(self) -> None:
        for h in self._hooks:
            h()
```

Two context managers expose `Tx` to callers:

```python
@contextmanager
def write_transaction(self) -> Iterator[Tx]:
    """RW transaction. Holds the write lock across commit + post-commit hooks."""
    with self._write_lock:
        with self._engine.begin() as conn:   # IMMEDIATE on SQLite
            tx = Tx(conn)
            yield tx
            # SA commits here on context exit (or rolls back on exception)
        # Lock still held — fire hooks atomically.
        tx._fire_hooks()

@contextmanager
def read_snapshot(self) -> Iterator[Tx]:
    """RO snapshot. No write lock, no hooks. PRAGMA query_only for safety."""
    with self._engine.connect() as conn:
        conn.execute(text("PRAGMA query_only = ON"))
        conn.execute(text("BEGIN"))
        try:
            yield Tx(conn)
        finally:
            conn.execute(text("ROLLBACK"))
            conn.execute(text("PRAGMA query_only = OFF"))
```

**Why holding the lock across hooks matters.** Today's `EndpointStore` invariant: a reader in another thread cannot observe the SQL-committed-but-dict-not-yet-updated state. Without holding the lock across hooks, there's a window where SQL is committed but the dict still has the old entry. By holding the lock until after `_fire_hooks()`, the window closes.

### 4.3 Schema with `TypeDecorator`s

Tables are SA Core `Table` constructs. Custom Python types become `TypeDecorator` subclasses:

```python
# schema.py
metadata = MetaData()


class JobNameType(TypeDecorator):
    """Adapts JobName <-> TEXT."""
    impl = String
    cache_ok = True

    def process_bind_param(self, value, dialect):
        return None if value is None else value.to_wire()

    def process_result_value(self, value, dialect):
        return None if value is None else JobName.from_wire(value)


class TimestampType(TypeDecorator):
    """Adapts Timestamp (ms since epoch) <-> INTEGER."""
    impl = Integer
    cache_ok = True

    def process_bind_param(self, v, _): return None if v is None else v.to_ms()
    def process_result_value(self, v, _): return None if v is None else Timestamp.from_ms(v)


class CachedProto(TypeDecorator):
    """Bytes-keyed LRU memo for proto blob columns. Transparent to call sites.

    Replaces today's ProtoCache + Column.cached=True mechanism. Two rows
    with identical blob bytes share the same decoded Python object — same
    invariant as today.
    """
    impl = LargeBinary
    cache_ok = True

    _MAX_SIZE = 8192
    _global_cache: ClassVar[dict[bytes, Any]] = {}
    _global_lock: ClassVar[Lock] = Lock()

    def __init__(self, message_cls: type[Message]):
        super().__init__()
        self._message_cls = message_cls

    def process_bind_param(self, value, _):
        return None if value is None else value.SerializeToString()

    def process_result_value(self, value, _):
        if value is None:
            return None
        with self._global_lock:
            hit = self._global_cache.get(value)
            if hit is not None:
                return hit
        decoded = self._message_cls.FromString(value)
        with self._global_lock:
            if len(self._global_cache) >= self._MAX_SIZE:
                # LRU-ish: drop oldest 25% (matches today's ProtoCache)
                for k in list(self._global_cache.keys())[: self._MAX_SIZE // 4]:
                    del self._global_cache[k]
            self._global_cache[value] = decoded
        return decoded


# Tables
jobs = Table(
    "jobs", metadata,
    Column("job_id", JobNameType, primary_key=True),
    Column("state", Integer, nullable=False),
    Column("submitted_at", TimestampType, nullable=False),
    Column("started_at", TimestampType),
    Column("finished_at", TimestampType),
    Column("error", String),
    Column("exit_code", Integer),
    Column("name", String, nullable=False),
    Column("depth", Integer, nullable=False),
    Column("res_cpu_millicores", Integer, nullable=False),
    Column("res_memory_bytes", Integer, nullable=False),
    Column("res_disk_bytes", Integer, nullable=False),
    Column("res_device_json", String),
    Column("is_reservation_holder", Boolean, nullable=False, server_default="0"),
    Column("reservation_json", String),
    # …
    Index("idx_jobs_reservation_holder", "job_id",
          sqlite_where=text("is_reservation_holder = 1")),  # partial index
)

job_config = Table(
    "job_config", metadata,
    Column("job_id", JobNameType, ForeignKey("jobs.job_id", ondelete="CASCADE"),
           primary_key=True),
    Column("config_proto", CachedProto(JobConfigProto), nullable=False),
)

# … other tables similarly
```

A read of `job_config.config_proto` now returns a fully-decoded `JobConfigProto` instance, with the bytes-keyed memo applied transparently. There's no `decode_one`, no `JOB_DETAIL_PROJECTION.decode`, no `cached=True` flag — the column type *is* the decoder.

### 4.4 Reads — SA Core `select` at call sites

The new convention:

- **Ad-hoc / one-off** queries: inline `select(...)` at the call site.
- **Hot or shared** queries (used in >1 place, or on a perf-critical path with a hand-tuned plan): named constants and helper functions in `reads/<area>.py`.

This is the "common/optimized queries remain in a common library set" point — `reads/` is the library, the rest is inline.

**Inline example (a one-off):**

```python
# In some RPC handler:
def get_job_count(tx, user_id):
    return tx.execute(
        select(func.count())
        .select_from(jobs)
        .where(jobs.c.user_id == user_id)
    ).scalar_one()
```

**Named-constant example (hot path):**

```python
# reads/scheduler.py — hot path, hand-tuned with partial index
JOBS_WITH_RESERVATIONS = (
    select(jobs.c.job_id, jobs.c.reservation_json)
    .where(jobs.c.is_reservation_holder == True)  # uses partial index
)

def jobs_with_reservations(tx):
    return tx.execute(JOBS_WITH_RESERVATIONS).all()
```

**Helper-function example (composite read):**

```python
# reads/dashboard.py
def task_summary_for_jobs(tx, job_ids: list[JobName]) -> dict[JobName, TaskSummary]:
    """Counts of tasks in each state, per job. One round-trip via GROUP BY."""
    rows = tx.execute(
        select(tasks.c.job_id, tasks.c.state, func.count())
        .where(tasks.c.job_id.in_(job_ids))
        .group_by(tasks.c.job_id, tasks.c.state)
    ).all()
    out: dict[JobName, TaskSummary] = {jid: TaskSummary() for jid in job_ids}
    for jid, state, n in rows:
        out[jid].set(state, n)
    return out
```

**Recursive CTE example (priority bands):**

```python
# reads/scheduler.py
def priority_band(tx, job_id: JobName) -> int | None:
    cte = (
        select(jobs.c.job_id, jobs.c.parent_id, jobs.c.priority_band)
        .where(jobs.c.job_id == job_id)
        .cte("ancestors", recursive=True)
    )
    cte = cte.union_all(
        select(jobs.c.job_id, jobs.c.parent_id, jobs.c.priority_band)
        .join(cte, jobs.c.job_id == cte.c.parent_id)
    )
    return tx.execute(
        select(cte.c.priority_band)
        .where(cte.c.priority_band.is_not(None))
        .order_by(cte.c.priority_band)
        .limit(1)
    ).scalar_one_or_none()
```

Aggregates, recursive CTEs, and dynamic ORDER BY / LIMIT / OFFSET are all first-class — no custom helpers, no API extensions to design.

#### What about row types?

SA Core returns `Row` objects from `tx.execute(...)`. These are tuple-like with attribute access:

```python
row = tx.execute(select(jobs).where(jobs.c.job_id == jid)).first()
row.state           # accessor
row.config_proto    # already a JobConfigProto thanks to CachedProto type
```

The shape of `Row` is dynamic at the type-system level — pyrefly/mypy don't know what attributes are present, so `row.state` is unchecked. For external API surfaces (RPC responses), we want named, typed shapes. Two patterns, both zero-runtime-cost:

**Pattern 1 — Bare `Row` (preferred for hot paths and internal helpers).** Return `Row` directly. Callers use `row.field` access. No type checking on the attributes; trade-off is acceptable for hot paths.

**Pattern 2 — `Protocol` + `cast` at the boundary.** When a function crosses an RPC handler or any place we want type checking and IDE autocomplete:

```python
# reads/jobs.py
class JobDetailRow(Protocol):
    job_id: JobName
    state: int
    submitted_at: Timestamp
    started_at: Timestamp | None
    finished_at: Timestamp | None
    config_proto: JobConfigProto
    # …

def get_detail(tx: Tx, job_id: JobName) -> JobDetailRow | None:
    row = tx.execute(select(jobs).where(jobs.c.job_id == job_id)).first()
    return cast(JobDetailRow | None, row)
```

The `cast` is a no-op at runtime — the returned object is still the SA `Row`. The Protocol gives pyrefly/mypy a structural description so callers get autocomplete and type errors on misspellings. **No object construction, no copying, no slots-vs-not-slots tradeoff.** Strictly better than the dataclass-at-boundary pattern we'd otherwise use.

When a Protocol gets reused in many call sites, hoist it to `reads/<area>.py` next to the `select` constant. One-shot Protocols can live next to the function that returns them.

Use **Pattern 1** by default. Reach for **Pattern 2** at RPC boundaries or wherever you'd want the type checker to catch a typo. There's no central decision; it's a per-call-site choice.

**Why not frozen dataclasses at the boundary?** They cost a per-row construction (small but real on hot paths) and they force a second source of truth for the field set (Protocol vs dataclass vs the SA `select` column list — three places to keep in sync). The Protocol version costs nothing and has only two sources of truth (the `select` and the Protocol), which is the same as today. If a particular boundary genuinely needs an immutable value with `__eq__` / `__hash__` (e.g. caching the return value in a Python-level dict), promote that one to a dataclass; otherwise stay with Protocol.

### 4.5 Writes — module functions

Writes are SA Core `insert` / `update` / `delete` constructs through `tx.execute`, in module-level functions under `writes/<entity>.py`:

```python
# writes/jobs.py
@writes_to(jobs, job_config, user_budgets)
def insert_job(tx, job, config, budget):
    tx.execute(insert(jobs).values(
        job_id=job.job_id,
        state=job.state,
        submitted_at=job.submitted_at,
        # …
    ))
    tx.execute(insert(job_config).values(
        job_id=job.job_id,
        config_proto=config,             # CachedProto handles serialization
    ))
    tx.execute(insert(user_budgets).values(
        user_id=budget.user_id,
        cpu_remaining=budget.cpu_remaining,
        # …
    ))


@writes_to(jobs)
def update_state_if_not_terminal(tx, job_id: JobName, new_state: int):
    tx.execute(
        update(jobs)
        .where(jobs.c.job_id == job_id, ~jobs.c.state.in_(TERMINAL_STATES))
        .values(state=new_state)
    )


@writes_to(jobs)
def bulk_update_state(tx, updates: list[tuple[JobName, int]]):
    tx.executemany(
        update(jobs).where(jobs.c.job_id == bindparam("jid")).values(state=bindparam("s")),
        [{"jid": jid, "s": state} for jid, state in updates],
    )
```

The `@writes_to(*tables)` decorator records the table set as `fn.writes_to` and registers `fn` in a module-level list. Used only at startup for the owned-table check (§4.8). Pure metadata.

### 4.6 Projections — write-through caches

The two tables that need in-memory write-through caches (`endpoints`, `worker_attributes`) get a `Projection` class each. The class:

- Owns the in-memory dict(s)
- Exposes mutating methods that do SQL + register a post-commit dict update
- Exposes read methods that hit the dict (no SQL)
- Implements `rehydrate(tx)` for startup and post-restore
- Declares its `sources` for the §4.8 owned-table check

```python
# projections/endpoints.py
class EndpointsProjection:
    """Write-through in-memory cache for the endpoints table."""

    sources = (endpoints,)

    def __init__(self) -> None:
        self._by_id: dict[str, EndpointRow] = {}
        self._by_name: dict[str, set[str]] = {}
        self._by_task: dict[JobName, set[str]] = {}
        self._lock = RLock()
        PROJECTIONS.append(self)

    def rehydrate(self, tx: Tx) -> None:
        """Populate state from SQL. Called at startup and after replace_from."""
        with self._lock:
            self._by_id.clear()
            self._by_name.clear()
            self._by_task.clear()
            for row in tx.execute(select(endpoints)):
                self._index(_row_to_endpoint(row))

    # ─── Mutations ──────────────────────────────────────────────────
    def add(self, tx: Tx, endpoint: EndpointRow) -> None:
        tx.execute(
            sqlite_insert(endpoints)
            .values(
                endpoint_id=endpoint.endpoint_id,
                name=endpoint.name,
                task_id=endpoint.task_id,
                # …
            )
            .on_conflict_do_update(
                index_elements=["endpoint_id"],
                set_={"name": endpoint.name, "task_id": endpoint.task_id, …},
            )
        )
        tx.register(lambda: self._reindex(endpoint))

    def remove(self, tx: Tx, endpoint_id: str) -> None:
        tx.execute(delete(endpoints).where(endpoints.c.endpoint_id == endpoint_id))
        tx.register(lambda: self._unindex(endpoint_id))

    # ─── Reads (take tx for API symmetry; serve from dict) ──────────
    def by_id(self, tx: Tx, endpoint_id: str) -> EndpointRow | None:
        del tx
        with self._lock:
            return self._by_id.get(endpoint_id)

    def by_name(self, tx: Tx, name: str) -> list[EndpointRow]:
        del tx
        with self._lock:
            return [self._by_id[i] for i in self._by_name.get(name, set())
                    if i in self._by_id]

    def all(self, tx: Tx) -> list[EndpointRow]:
        del tx
        with self._lock:
            return list(self._by_id.values())

    # ─── Helpers ────────────────────────────────────────────────────
    def _reindex(self, endpoint: EndpointRow) -> None:
        with self._lock:
            self._unindex(endpoint.endpoint_id)
            self._index(endpoint)
    # _index / _unindex as today
```

**No `Projection.query` companion.** A caller who needs snapshot-isolated reads against `endpoints_table` just writes `tx.execute(select(endpoints).where(...))` inline. SA selects are cheap; auto-generating a companion query object for each Projection would be ceremony with no payoff.

### 4.7 Consistency model

Two read flavors at call sites; readers pick by which handle / class they touch:

| Read | What it returns |
|---|---|
| `tx.execute(select(table).where(...))` inside `read_snapshot()` | Snapshot-isolated against everything else in the same `tx`. |
| `projection.by_id(tx, key)` | Latest-committed state. Not consulted to choose data. |

The `tx` parameter on `projection.by_id` is API ceremony — accepted so a reader scanning a call site sees `something.by_id(tx, ...)` everywhere and recognizes "this is a DB-layer access." Real call sites prefer latest-committed Projection reads anyway (`healthy_active_workers_with_attributes` in `db.py:908` is the canonical example — it deliberately mixes latest-committed worker-attribute reads with snapshot worker reads, and would break if the Projection were forced to be snapshot-isolated).

**The escape hatch is implicit.** Whoever wants snapshot-isolated reads against an endpoint just writes `tx.execute(select(endpoints).where(...))`. No `Projection.query` proxy needed.

### 4.8 Invalidation strategy — `@writes_to` + startup check + `cascades_into`

FK cascades can silently invalidate Projections: a write that targets one table may, via `ON DELETE CASCADE`, mutate a Projection-owned table without the write function knowing. `@writes_to` takes an optional `cascades_into=` argument to make this explicit:

```python
@writes_to(workers, cascades_into=(worker_attributes, task_attempts))
def remove_worker(tx, worker_id):
    tx.execute(delete(workers).where(workers.c.worker_id == worker_id))
    # FK cascades delete from worker_attributes (Projection-owned!) and task_attempts.
```

The startup check treats `cascades_into` tables as if the function had written to them directly. If a function cascade-deletes from a Projection-owned table, that function must be moved onto the Projection — or the Projection must add a `register` hook driven from the function's call site, with a comment explaining the linkage.

The startup-time check itself, run once after all `writes/` and `projections/` modules are imported:

```python
def assert_owned_tables_not_externally_written() -> None:
    owned: dict[Table, type[Projection]] = {}
    for P in PROJECTIONS:
        for table in P.sources:
            owned[table] = type(P)

    for fn in REGISTERED_WRITE_FUNCTIONS:
        for table in (*fn.writes_to, *fn.cascades_into):
            if table in owned:
                raise ConfigurationError(
                    f"Write function {fn.__qualname__} writes (or cascades) into "
                    f"{table.name}, which is owned by {owned[table].__name__}. "
                    f"Move this write onto the Projection or add an explicit "
                    f"invalidation hook."
                )
```

Zero runtime cost; fails loudly at controller startup if an invariant is violated.

For `remove_worker` specifically, the right shape is to call the projection from inside the function:

```python
@writes_to(workers)  # worker_attributes handled by the projection
def remove_worker(tx, worker_id):
    tx.execute(delete(workers).where(workers.c.worker_id == worker_id))
    # task_attempts cascade is OK — no projection on it
    # worker_attributes cascade: projection must update its dict
    projections.worker_attrs.invalidate_for_worker(tx, worker_id)
```

`invalidate_for_worker` registers a post-commit hook to drop the worker's attrs from the dict — symmetric with `remove` but without issuing the SQL DELETE (the cascade does that).

### 4.9 Startup / restore lifecycle

```
ControllerDB.__init__(db_path):
    ├── _engine = _make_engine(db_path)        # SA engine, pragmas on connect
    ├── _apply_migrations()                    # hand-rolled .py files, unchanged
    ├── _register_projections()                # instantiate EndpointsProjection,
    │                                          #              WorkerAttrsProjection
    │
    ├── with self.read_snapshot() as tx:
    │       for p in PROJECTIONS:
    │           p.rehydrate(tx)                # populate dicts
    │
    └── _assert_owned_tables_check()           # startup-time invariant check

ControllerDB.replace_from(backup):
    ├── _engine.dispose()                      # close pool
    ├── _replace_files(backup)
    ├── _engine = _make_engine(self._path)     # fresh engine for new files
    ├── _apply_migrations()                    # ensure schema is current
    └── with self.read_snapshot() as tx:
            for p in PROJECTIONS:
                p.rehydrate(tx)                # refresh dicts from new file
```

Critically, `_engine.dispose()` evicts all pooled connections — they're tied to the old SQLite file. The fresh engine reopens against the replaced file.

### 4.10 What lives where — file layout

```
lib/iris/src/iris/cluster/controller/
  db.py                       # Engine setup, Tx, write_transaction, read_snapshot, lifecycle
  schema.py                   # SA Core Tables + TypeDecorators (JobNameType, CachedProto, ...)
  reads/
    __init__.py
    scheduler.py              # JOBS_WITH_RESERVATIONS, priority_band, resource_usage_by_worker, ...
    dashboard.py              # task_summary_for_jobs, list_jobs_for_user_paginated, ...
    lifecycle.py              # bulk_get_for_updates, descendant tree walks, ...
    jobs.py                   # get_detail, get_config, count_by_state, ...
    tasks.py
    workers.py
  writes/
    __init__.py
    jobs.py                   # insert_job, update_state_if_not_terminal, bulk_update_state, ...
    tasks.py
    task_attempts.py
    workers.py
    reservations.py
  projections/
    __init__.py               # PROJECTIONS registry, rehydrate_all, owned-table check
    endpoints.py
    worker_attrs.py
  migrations/                 # unchanged
  controller.py               # runtime loops; reads through tx/db
  service.py                  # RPC handlers; reads through tx/db
  transitions.py              # state-machine, drives writes
  ...
```

Deletions: `stores.py` (~2100 lines), most of today's `schema.py` (the `Projection` / `ProtoCache` / `Column`-metadata machinery, ~600 lines), `_attr_cache` from `db.py` (~50 lines).

Additions: SA Core schema declarations (~400 lines, mostly verbose `Column(...)` lines), `Tx` wrapper (~50 lines), two Projection classes (~250 lines combined), `reads/` and `writes/` modules (~400 lines combined, mostly hand-tuned named queries hoisted from today's stores).

**Net code delta: estimated −1500 LOC.**

---

## 5. API Reference

### 5.1 `Tx`

```python
class Tx:
    conn: Connection                                 # raw SA Connection for advanced use
    def execute(self, stmt, params=None) -> CursorResult: ...
    def executemany(self, stmt, params_list) -> CursorResult: ...
    def register(self, hook: Callable[[], None]) -> None:
        """Hook fires once, after commit, under the write lock. Write tx only."""
```

### 5.2 `ControllerDB` methods

```python
class ControllerDB:
    @contextmanager
    def write_transaction(self) -> Iterator[Tx]:
        """RW transaction. Holds write lock across commit + post-commit hooks."""

    @contextmanager
    def read_snapshot(self) -> Iterator[Tx]:
        """RO snapshot. PRAGMA query_only = ON; BEGIN/ROLLBACK."""

    def backup_to(self, dst: Path) -> None: ...
    def replace_from(self, src: Path) -> None: ...
```

### 5.3 `@writes_to`

```python
def writes_to(
    *tables: Table,
    cascades_into: tuple[Table, ...] = (),
) -> Callable:
    """Marks a write function. Used by the startup-time owned-table check.

    Args:
        tables: Tables this function writes to directly (INSERT/UPDATE/DELETE).
        cascades_into: Tables whose rows may be deleted by FK cascade from
            writes to `tables`. The owned-table check treats these as direct
            writes for invalidation purposes.
    """
```

### 5.4 Projection base contract

```python
class Projection(Protocol):
    sources: ClassVar[tuple[Table, ...]]

    def rehydrate(self, tx: Tx) -> None: ...
```

Each Projection is otherwise free-form — methods are application-specific (`add`, `remove`, `by_id`, etc.).

---

## 6. Migration Plan — Single PR, Stacked Commits

One PR, ~12 stacked commits, each independently testable, end-to-end smoke before merge. The migration is invasive but mechanical; the structure below keeps each step reviewable and revertable.

| # | Commit | Touches | Net LOC | Tests after this commit |
|---|---|---|---|---|
| 1 | Add `sqlalchemy>=2.0` dep; add `db_v2.py` with engine + Tx + transaction contexts. | new file; `pyproject.toml` | +250 | new file's own unit tests |
| 2 | Add `schema_v2.py` with SA Core Tables, TypeDecorators. **No usages yet.** Schema must match today's DB exactly (verified by reading both and comparing CREATE TABLE statements). | new file | +400 | schema-equivalence test (round-trip DDL) |
| 3 | Add `Tx`, `write_transaction`, `read_snapshot`. Add tests for hook ordering, lock-held-across-hooks, rollback safety. | `db_v2.py`, tests | +100 | atomicity unit tests |
| 4 | **Port `_jobs_with_reservations` to SA Core (the perf canary).** Add it as a named select in `reads/scheduler.py`. Run benchmark. | new + 1 call site swap | ±60 | bench: ≤ 0.025 ms (today: 0.019 ms) |
| 5 | **Port `EndpointStore` → `EndpointsProjection` (the linchpin).** Add acceptance tests for atomic write-through, rollback safety, restore correctness, concurrency. | new file; delete `EndpointStore`; call-site swaps | ±300 | full atomicity suite (§7.4) |
| 6 | Port `_attr_cache` → `WorkerAttrsProjection`. | new file; delete `_attr_cache` from `db.py`; call sites | ±200 | atomicity tests |
| 7 | Port `JOB_DETAIL_PROJECTION` and related job-read paths. Hot reads (`get_detail`, `get_config`, `list_descendants`) become named selects in `reads/jobs.py`. | reads/, call sites | ±400 | integration tests pass |
| 8 | Port task / task_attempt reads. Heavy paths (`bulk_get_for_updates`, `reconcile_rows_for_workers`) become named selects in `reads/scheduler.py`. Benchmark. | reads/, call sites | ±450 | bench: `reconcile_rows_for_workers` ≤ 8 ms |
| 9 | Port worker reads, reservation reads, dashboard composites. | reads/, call sites | ±400 | integration |
| 10 | Move all writes from `stores.py` to `writes/*.py` with `@writes_to`. | new files; delete write methods | ±900 | integration |
| 11 | Add `@writes_to` decorator + `assert_owned_tables_not_externally_written()` startup check. Include `cascades_into` for `remove_worker`. | `db_v2.py`, `writes/*.py` | +100 | startup-check test |
| 12 | **Delete old infrastructure.** `stores.py` (most of it), today's `Projection`/`ProtoCache`/`Column`-metadata from `schema.py`, `_attr_cache` field. Rename `*_v2` files to drop the suffix. | broad delete | −1900 | full suite |

**Net (cumulative through commit 12): -1500 to -1800 LOC, 5 mechanisms → 2.**

### 6.1 Per-commit acceptance

For every commit:

1. `./infra/pre-commit.py --all-files --fix` passes.
2. `uv run pyrefly` passes.
3. `uv run pytest lib/iris/tests/` passes.
4. For commits 4, 5, 6, 8: perf benchmark in §7.3 passes its gate.
5. Commit is independently revertable.

### 6.2 The three high-risk commits

- **Commit 4** — first SA Core hot-path port. Validates the perf assumption. If `_jobs_with_reservations` regresses (>0.025 ms), the proposal is in trouble and we go back to homemade.
- **Commit 5** — `EndpointStore → EndpointsProjection`. Validates that SA `engine.begin()` + our `Tx.register` preserves today's atomicity. Acceptance tests in §7.4.
- **Commit 8** — heaviest hot path (`reconcile_rows_for_workers`). Final perf validation.

If any of these fail their gates, the design is wrong; rollback to the prior commit and rethink.

### 6.3 End-to-end smoke before merge

Before clicking merge:

1. Spin up a real controller against a real workload via `scripts/iris/dev_tpu.py` (the `dev-tpu` skill).
2. Submit a small batch of jobs and verify state transitions: submitted → scheduled → running → terminal.
3. Restart the controller from a backup. Verify all projections rehydrate; verify dashboard renders.
4. Run a 10-minute soak; confirm scheduler tick latency stays in the established envelope (`resource_usage_by_worker` < 10 ms p95; `reconcile_rows_for_workers` < 10 ms p95).
5. Compare wall-clock CPU profile of the controller (`agent-profiling` skill) before and after. Expect a small win from removed Python indirection plus SA's statement-compilation cache.

If any of these fail, the PR does not merge.

---

## 7. Testing Strategy

### 7.1 Unit tests

- `tests/cluster/controller/test_tx.py` — `Tx`, `write_transaction`, `read_snapshot`, post-commit hook ordering, lock-held-across-hooks, rollback safety.
- `tests/cluster/controller/test_schema.py` — `CachedProto` correctness and cache behavior, `JobNameType` round-tripping.
- `tests/cluster/controller/projections/test_endpoints_projection.py` — replaces today's `test_endpoint_store.py`; same assertions.
- `tests/cluster/controller/projections/test_worker_attrs_projection.py` — write-through correctness.
- `tests/cluster/controller/writes/test_jobs.py`, etc. — write-function smoke tests.

### 7.2 Integration tests

Existing integration tests in `tests/cluster/controller/test_controller.py`, `test_scheduler.py`, `test_transitions.py` must pass unchanged. These tests use a real `ControllerDB`; if they pass, the refactor is behaviorally transparent.

### 7.3 Performance benchmarks (gates for migration commits)

Added in commit 1 as `tests/cluster/controller/test_perf_baselines.py`:

| Benchmark | Today | v2 gate |
|---|---|---|
| `_jobs_with_reservations` (200 reservations) | 0.019 ms | ≤ 0.025 ms |
| `resource_usage_by_worker` (24k jobs, 1k live) | 6.5 ms | ≤ 8 ms |
| `reconcile_rows_for_workers` (200 workers) | 6.3 ms | ≤ 8 ms |
| `EndpointsProjection.by_id` (PK lookup, in-memory) | dict access | < 1 µs |
| `get_detail(tx, job_id)` (PK SELECT with proto decode) | 0.05 ms | ≤ 0.07 ms |

Gate failures block their respective commit.

### 7.4 Atomicity tests (commit 5 acceptance — the linchpin)

Commit 5 (`EndpointStore → EndpointsProjection`) proves the new `Tx.register` + `write_transaction()` machinery preserves today's atomicity invariant: post-commit hooks fire under the write lock, so another thread cannot observe the SQL-committed-but-dict-not-yet-updated state. If this commit cannot carry that guarantee, the whole design fails.

1. **Atomic write-through.** Insert an endpoint; before the `write_transaction` context exits, no thread can observe the new entry in the dict. After exit, it must be visible.
2. **Rollback safety.** Insert an endpoint, then raise inside the transaction. The dict must not contain the entry; SQL rolled back.
3. **Restore correctness.** Snapshot the DB, modify endpoints, then `replace_from(snapshot)`. The dict must reflect the snapshot state, not the post-modification state.
4. **Concurrency.** Two threads writing endpoints, one thread reading via `by_id` repeatedly for 5 seconds. No `KeyError`, no torn reads, no deadlock under the write lock.
5. **Listing.** `ListEndpoints` RPC returns the same result set before and after commit.

### 7.5 Owned-table check tests

- Test that a write function declaring `@writes_to(endpoints)` outside `EndpointsProjection` causes `assert_owned_tables_not_externally_written()` to raise at startup.
- Test that `cascades_into=(worker_attributes,)` is treated as a write for invalidation purposes.

### 7.6 Restore tests

- After `backup_to` + `replace_from`, every Projection's in-memory state matches the snapshot's SQL state, not any post-snapshot modification.

---

## 8. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| SA Core per-query overhead pushes scheduler tick over budget | Low | High | Commit 4 (canary) and commit 8 benchmark gates. If they fail, abort. |
| `engine.begin()` + `Tx.register` doesn't preserve `tx.on_commit` atomicity | Low | High | Commit 5 acceptance tests. Lock held across `_fire_hooks` keeps the window closed. |
| SA Core compilation cache cold-start adds startup latency | Low | Low | One-time cost. Warmup with a few representative selects at startup if needed. |
| Schema mismatch: SA-generated DDL ≠ today's hand-rolled DDL (incl. partial indexes) | Medium | Medium | Commit 2 includes a schema-equivalence test (compare actual `sqlite_master` against SA's `metadata.create_all()` output). Partial indexes are `Index(..., sqlite_where=text(...))`. |
| `CachedProto` global cache leaks memory across DB restores | Low | Low | Cache is bounded (LRU-style eviction). Restore doesn't invalidate; protos with the same bytes are identical, so cross-restore reuse is safe. |
| Future contributor mixes inline `select` and named-constant `select` confusingly | Medium | Low | Document the convention in `lib/iris/AGENTS.md`: "hot or shared → `reads/`; one-off → inline." |
| SA's `engine.begin()` opens a SAVEPOINT on top of an existing transaction by default if reused | Low | Medium | Use `engine.begin()` only from `write_transaction()`. `read_snapshot()` uses `engine.connect()` + explicit BEGIN/ROLLBACK. |
| Alembic-on-SQLite footguns | N/A | — | We're not using Alembic. Hand-rolled migrations stay. |
| PR size deters review | High | Medium | 12 small commits, each reviewable in isolation. Reviewers can focus on commit boundaries. |
| SA dependency adds package-install time / cold-start latency | Low | Low | SA 2.x is ~10MB; cold import is ~200 ms once. Acceptable for a long-running controller. |

---

## 9. Open Questions

### 9.1 Should `@writes_to` be derived automatically?

SA `insert(jobs).values(...)` already names the target table. A runtime hook on `Connection.execute` could inspect `stmt.table` and accumulate the table set per-tx. This eliminates the manual decorator.

**Recommendation:** add this in a follow-up after the refactor lands. The manual decorator is cheap to maintain, and writing the runtime hook against SA's internals carries non-trivial risk of breaking on a SA minor-version bump.

### 9.2 Pattern 1 (`Row`) vs Pattern 2 (dataclass at boundary)?

§4.4 sketches both. The choice is per-call-site.

**Recommendation:** start with Pattern 1 (return `Row` directly). Convert to Pattern 2 only at RPC handler boundaries where wire-format DTOs already exist or where the row type is reused widely.

### 9.3 Global vs per-column `CachedProto` cache?

Today's `ProtoCache` is one global LRU. `CachedProto` in §4.3 is also one global LRU (class-level `_global_cache`). Alternative: per-column-type cache so `JobConfigProto` decoding doesn't crowd out `TaskResultProto` decoding.

**Recommendation:** start global (matches today's behavior exactly). If profiling shows crowding, switch to per-column with separate `_MAX_SIZE` per type.

### 9.4 Should partial indexes be declared via SA `Index(sqlite_where=...)` or via migrations only?

`Index(..., sqlite_where=text("..."))` works in SA Core 2.x and is included in `metadata.create_all()`. But our partial indexes were added via migration files (e.g. `migrations/0045`) and the SQL there is authoritative.

**Recommendation:** declare indexes in `schema.py` for documentation, but the *migrations* remain the source of truth for what's actually on disk. Add a schema-equivalence test (commit 2) to ensure they agree.

### 9.5 Migration tooling: stick with hand-rolled or move to Alembic?

Per §3 non-goals: stick with hand-rolled. Revisit only if (a) we adopt SA ORM (we won't), or (b) migrations become a bottleneck (they aren't).

---

## 10. Alternatives Considered

| Alternative | Why rejected |
|---|---|
| **Homemade `Query` / `CachedQuery` / `Projection` class hierarchy** | Tried as a paper design and stress-tested against real call sites. The fixed-shape API didn't cover ~60% of real call sites cleanly (aggregates, recursive CTEs, dynamic paging, dashboard composites). Adding the required extensions recreated the very proliferation the refactor was trying to eliminate. SA Core's expression language handles all those cases natively. |
| **Full SA ORM** (mapped classes, identity map, Session UoW) | Identity map + UoW are optimized for entity-mutate-flush workloads we don't have. Mapped-instance construction adds an estimated 5–10 µs/row, putting scheduler ticks reading 1k+ rows near or over our 6 ms budget. Write-through atomicity becomes harder under SA `Session` events (which inspect `session.new/dirty/deleted` and miss bulk UPDATE statements). SA Core gets us the wins without the ORM friction. |
| **SA Core "engine only" — `text(...)` everywhere** | Possible but undersells SA Core. Loses composable `select()`, which is what makes ad-hoc aggregates, paging, and dashboard composites tractable. The verbosity of `text()` vs `select()` is roughly equal; `select()` gets us refactor safety. |
| **`sqlc-gen-better-python`** | Codegen step; immature; doesn't address the in-memory caching layer. Worth revisiting in 12 months once the ecosystem settles. |
| **Postgres** | Largest possible change. SQLite-on-one-machine is our deliberate design (single-process controller, file-based backup, predictable latency). |
| **Alembic for migrations** | Requires declarative ORM models to diff against; we're rejecting the ORM. Alembic on SQLite also has known footguns (`op.drop_column` generates wrong SQL; requires `batch_alter_table` wrapping). Hand-rolled migrations stay. |
| **No refactor (status quo)** | The five-mechanism cognitive load is the explicit problem statement. Performance is fine; the cost is teachability and the friction of adding new query patterns. |

---

## 11. Convention to Land in `lib/iris/AGENTS.md`

A short section to document the conventions for future contributors:

> ### Iris data layer (post-refactor)
>
> - **Schema:** `controller/schema.py` defines SA Core `Table` objects and `TypeDecorator`s. To add a column, edit `schema.py` *and* add a migration in `migrations/`.
> - **Reads:** for hot or shared queries, add a named `select(...)` constant in `controller/reads/<area>.py`. For one-off queries, write `tx.execute(select(...).where(...))` inline. Either is fine; the convention is "hot or shared → name it, otherwise inline."
> - **Writes:** module-level functions in `controller/writes/<entity>.py`, decorated with `@writes_to(...)`. The decorator's `cascades_into=` argument captures FK cascades into Projection-owned tables.
> - **Projections (`endpoints`, `worker_attributes`):** never write directly to these tables; always go through `projections.endpoints` / `projections.worker_attrs` methods. The startup check enforces this.
> - **Transactions:** `db.write_transaction()` for writes (holds the write lock across commit + post-commit hooks). `db.read_snapshot()` for reads (snapshot isolation; no lock). Every read and write takes a `tx` as first param.
> - **Returning rows:** by default, return SA `Row` objects from internal helpers. Convert to a frozen dataclass / DTO at RPC boundaries where a typed wire shape is required.

---

## 12. References

### SQLAlchemy
- [SQLAlchemy 2.0 Core — Tables and Schema](https://docs.sqlalchemy.org/en/20/core/metadata.html)
- [SQLAlchemy 2.0 Core — SQL Expression Language](https://docs.sqlalchemy.org/en/20/core/tutorial.html)
- [SQLAlchemy 2.0 — `TypeDecorator`](https://docs.sqlalchemy.org/en/20/core/custom_types.html#sqlalchemy.types.TypeDecorator)
- [SQLAlchemy 2.0 — Statement Compilation Cache](https://docs.sqlalchemy.org/en/20/core/connections.html#sql-compilation-caching)
- [SQLAlchemy 2.0 — `Engine.begin` and connection context](https://docs.sqlalchemy.org/en/20/core/connections.html#working-with-the-connection)
- [SQLAlchemy — SQLite Dialect (PRAGMA, WAL, partial indexes)](https://docs.sqlalchemy.org/en/20/dialects/sqlite.html)
- [SQLAlchemy — Event API](https://docs.sqlalchemy.org/en/20/core/events.html)

### Codebase
- Prior design (SQL as source of truth): `.agents/projects/20260310_iris_sql_canonical.md`
- Today's data layer: `lib/iris/src/iris/cluster/controller/{schema.py,stores.py,db.py,transitions.py}`
- Performance commits worth reading before porting hot paths:
  - `cb77a2877` — slim `JOB_RESERVATION_PROJECTION`, bulk prefetch in `apply_heartbeats_batch`
  - `6b117b882` — partial indexes (migration `0045`), `resource_usage_by_worker` 350 ms → 6.5 ms
  - `040f97585` — dropped `SnapshotView` overlay in favor of in-memory liveness tracking

### Industry context
- [Martin Fowler — CQRS](https://martinfowler.com/bliki/CQRS.html) — separate models for reads and writes; v2's `Projection`s are the "inline projection" of CQRS literature.
- [Marten — Projections](https://martendb.io/events/projections/) — the canonical write-through "inline projection" pattern in mainstream use.
- [Apache Airflow — Database ERD](https://airflow.apache.org/docs/apache-airflow/stable/database-erd-ref.html) — Python orchestrator using SA Core / ORM with `@provide_session` patterns; closest neighbor.
- [Materialize — Cache invalidation](https://materialize.com/blog/redis-cache-invalidation/) — write-through caches go silently stale when external writers touch the underlying table; the `@writes_to` + `cascades_into` mechanism in §4.8 is our explicit answer.
- [SQLite triggers as materialized views (madflex)](https://madflex.de/SQLite-triggers-as-replacement-for-a-materialized-view/) — alternative we don't need given Projections suffice.
