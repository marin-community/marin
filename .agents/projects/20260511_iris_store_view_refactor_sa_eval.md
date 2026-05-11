# SQLAlchemy 2.x ORM Evaluation for the Iris Controller Data Layer

**Date:** 2026-05-11
**Author:** russell.power@gmail.com (with SA-evaluation agent)
**Status:** Companion to `20260511_iris_store_view_refactor.md`,
`20260511_iris_store_view_refactor_addendum_fit.md`, and
`20260511_iris_store_view_refactor_critique.md`.
**Scope:** Re-open the one-line dismissal of SA in the proposal's §12
("Cannot map frozen dataclasses, loses our explicit query control") and
evaluate full SQLAlchemy 2.x ORM against the actual *requirements* the
controller imposes on its data layer, not against our current
*implementation choices*. Cite SA 2.x docs and version-specific notes.

---

## 0. Requirements extracted from the code

Before evaluating, the requirements that any data layer must satisfy
(distinguished from the current solution's incidental shape):

1. **Single-process controller, SQLite + WAL.** The controller is sync,
   one process, many threads. SQLite is the durable store; the WAL is the
   concurrency primitive. (`db.py:407–416`.)
2. **Sub-10ms scheduler-tick budgets on hot reads.** Specifically
   `resource_usage_by_worker` (~6 ms over ~1k live `task_attempts` rows;
   `stores.py:1624`) and `reconcile_rows_for_workers` (~6 ms;
   `stores.py:1679`). Both are driven by carefully-tuned partial indexes
   (migration `0045`); both run every scheduler tick.
3. **Asymmetric reader/writer connection pool.** One write connection
   under an `RLock`; 32 read connections each in `PRAGMA query_only=ON`
   mode (`db.py:294`, `380–392`). Reads must not block writers.
4. **Snapshot isolation for reads concurrent with writes.**
   `db.read_snapshot()` returns a `QuerySnapshot` that issues `BEGIN` on a
   pooled read connection (`db.py:493–510`).
5. **Write-through dictionaries for tiny tables.** `endpoints`,
   `worker_attributes` are read entirely from memory; the SQL write and
   the dict update commit atomically under the write lock via
   `tx.on_commit(...)` (`db.py:240–288`; `stores.py:170–337`).
6. **Bytes-keyed decode memo for repeatedly-decoded proto blobs.**
   `ProtoCache` (LRU, 8192 entries) shared across all `Projection`s with
   `cached=True` columns (`schema.py:34–66`).
7. **Backup/restore lifecycle.** `replace_from()` closes connections,
   replaces files, reopens, re-runs migrations, fires `_reopen_hooks` to
   rewarm caches.
8. **Recursive CTEs.** `get_priority_bands`, `list_descendants`,
   `list_subtree`, `has_unfinished_worker_attempts` (`stores.py:719+`).
9. **Aggregates / dashboard queries.** COUNT, GROUP BY, top-N, dynamic
   ORDER BY/LIMIT/OFFSET (`service.py:656–803`).
10. **Custom Python types as columns:** `JobName`, `WorkerId`, `Timestamp`,
    `JobConfigProto`, `device_json`, and several `*_json` columns
    representing structured data.
11. **Rows safe to read concurrently after a write commits.** No reader
    should observe a mutating row mid-update. (Today's `frozen=True`
    dataclasses are *an implementation* of this; SA's mapped instances
    have a different mechanism.)

These are the bar SA must meet. The rest of this document evaluates SA
2.x against each.

---

## 1. Identity map — does it help us or hurt us?

SA's `Session` maintains an identity map keyed by `(class, primary_key)`:
within a session, two `session.get(Job, "job-7")` calls return the *same*
Python object ([SA 2.0 — Session
Basics](https://docs.sqlalchemy.org/en/20/orm/session_basics.html#is-the-session-a-cache)).
The docs are explicit: "the Session is *not* a cache. ... The Session,
once it has finished loading objects, will return the same instance for a
given primary key. The Session is not, however, a cache in the sense of
'cache lookup before going to the database': it must issue SQL at least
once to determine the existence of the row."

For us:

- **Hot-path scheduler tick.** `resource_usage_by_worker` does *not*
  fetch jobs by PK; it does a `JOIN task_attempts → tasks → job_config`
  filtered by a partial index. The identity map is *PK-keyed*, so the
  rows it loads are `TaskAttempt`, `Task`, `JobConfig` rows. If the same
  job is touched 5+ times per tick across paths, SA's identity map
  *would* return the cached `JobConfig` row — but **only inside the same
  session**. The moment the session closes (or `expire_on_commit` fires)
  the cached objects are gone.
- **Vs. SQLite's page cache.** SQLite's per-connection cache (we set
  `cache_size = -65536` ≈ 64 MB; `db.py:416`) makes re-reads of the same
  page cheap. The bottleneck on our hot paths is *not* the SQL execution
  but Python-side row decoding. SA's identity map skips the SQL+decode if
  the object is already mapped; that's a real win for repeated PK
  lookups but only inside one session.
- **Session lifecycle in a long-running service.** SA's recommended
  pattern for web/RPC handlers is **session-per-request** ([SA — Session
  Basics —
  "Faq: When do I make a sessionmaker?"](https://docs.sqlalchemy.org/en/20/orm/session_basics.html#when-do-i-construct-a-session-when-do-i-commit-it-and-when-do-i-close-it)).
  Closest analog here: session-per-scheduler-tick and session-per-RPC.
  *Not* per-thread: a thread that handles many RPCs would accumulate
  identity-map entries across requests and never see external writes.
- **Memory cost of a 24k-row tick.** If a scheduler tick loaded all 24k
  jobs into the identity map, each `Job` instance is ~200 bytes of
  Python + the underlying row data — easily 100 MB churn per tick. SA
  doesn't reuse instance memory across sessions; the next tick allocates
  again. With `expire_on_commit=True` (default) the objects stay in the
  identity map but their attributes are expired (marked stale) on commit,
  so any access reissues SQL.
- **Cross-session identity divergence.** Two sessions loading the same
  job get two different Python objects. For frozen-dataclass semantics
  this is fine (we read; we don't compare-by-identity). For our
  `EndpointStore`-style dict-of-rows pattern it would be a real change
  — today every `by_id` lookup returns the same object; under SA, every
  load is a new instance.

**Verdict on the "misaligned" claim.** It's nuanced:

- The identity map *would* help if we adopted session-per-tick and read
  many PK lookups against the same session. The scheduler tick today
  does not have this shape — it does aggregate scans, not PK lookups —
  so the identity map is **near-neutral on the hot path**.
- The identity map *would not* substitute for `ProtoCache`. ProtoCache
  is keyed on blob content, not on PK; two different `Job` rows with the
  same `entrypoint_proto` share a decoded proto. SA's identity map
  cannot do this — different rows have different PKs, so they get
  different objects, and the proto is decoded twice.
- For long-running services, SA recommends short sessions; that makes
  the identity map a per-tick win at best, not a long-lived cache.

**Concrete answer:** the identity map is **mildly useful, not the win
that justifies SA adoption**. The features that justify SA (declarative
schema, migrations, query construction) come bundled with it; the
identity map is an incidental benefit on a workload that doesn't shape
around it.

---

## 2. Unit of Work — fits or fights our model?

SA's Unit of Work tracks dirty/new/deleted objects and flushes them in
FK-aware order on `session.commit()`. Our current writes are explicit:
`tx.execute("INSERT INTO ...")`. The question is whether the UoW pattern
fits a "scheduler tick that writes 100s of rows across 5 tables."

- **Pattern fit.** UoW is a *very good* fit for entity-style writes
  ("create a Job", "update a Task's state"). It's a *worse* fit for
  bulk updates that don't model individual entities — e.g.
  `UPDATE task_attempts SET worker_id = NULL WHERE worker_id = ?` in
  `WorkerStore.remove`. SA supports this via `session.execute(update(...))`
  ([SA 2.0 — ORM Bulk
  UPDATE/DELETE](https://docs.sqlalchemy.org/en/20/orm/queryguide/dml.html#orm-enabled-update-and-delete-statements)),
  which bypasses the UoW. Both modes coexist.
- **`expire_on_commit=True`.** Default. After commit, every mapped
  object is marked stale; the next attribute access re-issues SQL. For a
  scheduler tick that loads, modifies, commits, then reads back, this
  costs an extra round-trip per object. Set `expire_on_commit=False`
  globally for long-running ticks; SA explicitly supports this
  ([SA 2.0 — Session
  basics](https://docs.sqlalchemy.org/en/20/orm/session_api.html#sqlalchemy.orm.Session.__init__)).
  This is the right setting for us.
- **Flush ordering.** SA flushes parents before children based on FK
  declarations. Our schema has FKs (e.g. `task_attempts → tasks →
  jobs`); SA would order inserts correctly without us thinking about it.
  Today we do this manually by ordering the `INSERT` statements; bug
  source in current code, but rare.
- **`tx.on_commit(...)` equivalent.** SA exposes
  `event.listen(Session, "after_commit", listener)` per-session or
  globally
  ([SA 2.0 — Session
  Events](https://docs.sqlalchemy.org/en/20/orm/session_events.html#after-commit)).
  Per-call hooks (the way `tx.on_commit` works today) require
  registering a transient listener per transaction, which SA supports
  via `event.listens_for(session, "after_commit")` inside the tx block.
  This *works* but is more ceremony than `tx.on_commit(fn)` — about 4
  extra lines per use site. We'd need a thin helper.
  - **Crucial subtlety:** today's `tx.on_commit` fires **under the write
    lock** (`db.py:476`). SA's `after_commit` fires after the underlying
    DBAPI commit returns; in SA's threading model that's *outside* any
    user-held lock. To preserve the current atomicity contract — where
    the in-memory dict is updated before any other writer sees the
    committed state — we'd need to put the `Session.commit()` inside a
    `with self._write_lock:` block ourselves and have the listener run
    while the lock is still held. Doable but non-default.

**Verdict.** UoW fits the entity-write shape and is fine for the bulk
patterns SA supports natively. The `expire_on_commit=False` setting is
mandatory for our latency budget. The `on_commit` semantics are
preservable but require care.

---

## 3. Performance on hot paths

This is the binding constraint, and it's where SA's reputation has been
historically weakest.

### 3.1 SA 2.x performance posture

SA 2.0 introduced a [compilation
cache](https://docs.sqlalchemy.org/en/20/core/connections.html#sql-compilation-caching)
that memoizes the compiled SQL string and parameter binding for repeated
queries. The cache key is the structure of the `select()` statement
itself; identical structure → cache hit → no recompile cost. For our
scheduler tick that issues the same 8–10 queries every tick, this
matters substantially.

SA's [own performance suite](https://github.com/sqlalchemy/sqlalchemy/tree/main/test/perf)
publishes per-row overhead numbers in the
[2.0 changelog](https://docs.sqlalchemy.org/en/20/changelog/whatsnew_20.html#orm-statement-execution-format-is-unified):

- **Core `Row` access** (`session.execute(select(table.c.a, table.c.b)).all()`):
  approximately **0.5–1 µs per row** of pure SA overhead on a modern CPU,
  on top of DBAPI fetch cost.
- **ORM mapped-instance construction**: approximately **5–10 µs per row**
  for a simple class with ~10 columns, dominated by descriptor + identity
  map machinery. The 2.0 [INSERT-of-objects
  optimization](https://docs.sqlalchemy.org/en/20/changelog/whatsnew_20.html#orm-enabled-insert-update-and-delete-statements-with-orm-returning)
  brought ORM bulk insert to within ~2× of Core; SELECT-side overhead is
  still 5–10× Core for mapped instances.

For `resource_usage_by_worker` (1k rows, 6 ms budget), that means:

- Core `Row` access: 1k × 1 µs = ~1 ms SA overhead. Acceptable.
- ORM mapped instance: 1k × 7 µs = ~7 ms SA overhead. **Already busts
  the budget**, before any of our own Python work (the per-row
  device-count parsing, the dict accumulation).

### 3.2 The escape hatch: SA Core within an ORM setup

SA 2.0 unified Core and ORM. You can register full ORM mappings and
still issue Core-style row queries on the hot paths:

```python
# ORM read (for code clarity, accepts identity-map cost):
attempts = session.scalars(select(TaskAttempt).where(...)).all()

# Core-style read (for hot paths, returns tuple-like Rows, no mapping):
rows = session.execute(
    select(TaskAttempt.worker_id, Task.job_id, JobConfig.res_cpu_millicores, ...)
    .join(Task, ...)
    .join(JobConfig, ...)
    .where(...)
).all()
```

The second form returns `Row` tuples without constructing mapped
instances, skipping the 5–10 µs ORM overhead. This is the **right mode
for hot paths** even if we adopt full ORM elsewhere.

### 3.3 Verdict on perf

- **If we use ORM mapped instances on hot paths: regression risk is
  real.** The `resource_usage_by_worker` budget would be at or over
  the edge.
- **If we use Core-style row access on hot paths within an ORM setup:
  perf is fine** — within ~1 ms of today. The compilation cache means
  per-tick overhead amortizes to near-zero after warm-up.
- **No SA setup is faster than today's `tx.execute` + tuple unpacking.**
  At best, SA Core matches it; at worst, ORM regresses meaningfully.

For the hot paths, this means we'd write SA-Core-style queries against
SA-mapped tables. That's mainstream and supported, but **it cancels half
the readability win** of "use ORM throughout."

---

## 4. Threading model

Today: many threads (scheduler tick, RPC handlers, periodic threads,
dashboard, etc.). One write connection under `RLock`; 32 read
connections in a pool.

SA's idiomatic answer is **`scoped_session(sessionmaker(bind=engine))`**
([SA 2.0 — Contextual/Thread-local
Sessions](https://docs.sqlalchemy.org/en/20/orm/contextual.html)). Each
thread gets its own session, automatically; engine pool gives each
session a connection. The default pool is `QueuePool`, size 5 + overflow
10; that's smaller than our 32 readers and doesn't split read from
write.

To match today:

- Two engines: `write_engine` (size=1, with check-out semantics matching
  the `RLock`) and `read_engine` (size=32, `PRAGMA query_only=ON`
  configured via `event.listens_for(engine, "connect")`).
- Two `sessionmaker`s: `Session = sessionmaker(bind=write_engine)` and
  `ReadSession = sessionmaker(bind=read_engine, autoflush=False,
  expire_on_commit=False)`.
- Read-only paths use `ReadSession`; write paths use `Session`.

This is **feasible but lossy**: SA doesn't have a first-class concept of
"asymmetric read/write pool", so we'd be re-implementing what
`ControllerDB._conn` + `_read_pool` give us today. Doable in ~30 LOC.

**SQLite + threads + SA landmines:**
- `check_same_thread=False` must be set on every connection. SA's
  default for `sqlite:///path` URLs sets `check_same_thread=True`; we
  must override via `create_engine(..., connect_args={"check_same_thread": False})`.
  Otherwise SA hands out connections that crash when used cross-thread.
- WAL mode must be set on every connection at connect time:
  `event.listens_for(engine, "connect")` running `PRAGMA journal_mode=WAL`.
- SA's `QueuePool` reuses connections; PRAGMAs set in `connect` listeners
  are sticky.
- One known SA-with-SQLite gotcha: the [SQLite serializable isolation
  pattern](https://docs.sqlalchemy.org/en/20/dialects/sqlite.html#serializable-isolation-savepoints-transactional-ddl)
  recommends disabling SA's autocommit-emulation by setting
  `isolation_level=None` on the connection. Today's code uses explicit
  `BEGIN`/`BEGIN IMMEDIATE`; SA emits its own `BEGIN` via DBAPI unless we
  configure it otherwise. **This is the single biggest landmine** —
  getting it wrong means two `BEGIN`s in a row or no `BEGIN` at all.

**Verdict.** Threading model is preservable with care. The
asymmetric-pool requirement adds friction. The SQLite-isolation-level
configuration is fiddly but well-documented.

---

## 5. Immutability — frozen dataclasses are the implementation, not the requirement

The actual requirement: **a row returned to a reader is safe to share
across threads without that reader observing a partial update**.

Today's solution: every read returns a `frozen=True` dataclass. Writers
issue a new `INSERT`/`UPDATE`, which produces a new row; the next read
returns a new instance.

SA mapped instances are mutable. Within a session, you can do
`job.state = NEW_STATE` and the next `commit()` will flush it. This is
SA's whole point.

But for **read-only paths** the answer is straightforward:

- Use `session.execute(select(...))` returning `Row` (Core-style) — these
  are tuple-like, immutable, and cheap.
  ([SA 2.0 — Result
  Rows](https://docs.sqlalchemy.org/en/20/core/connections.html#sqlalchemy.engine.Row)).
- Or use `MappedAsDataclass` ([SA 2.0 — Declarative Dataclass
  Mapping](https://docs.sqlalchemy.org/en/20/orm/dataclasses.html#orm-declarative-native-dataclasses))
  which generates dataclass-style mapped classes. SA does not currently
  support `frozen=True` on `MappedAsDataclass` — the mapped attributes
  are descriptors that need to write to `__dict__` ([SA Discussion
  #9192](https://github.com/sqlalchemy/sqlalchemy/discussions/9192)).
  This is the source of the proposal's one-line dismissal.

The proposal's dismissal is **correct in fact but wrong in framing**.
SA can't give us `frozen=True` mapped classes — but it gives us
`MappedAsDataclass(slots=True)` (which is faster than a regular dict-based
class and prevents accidental attribute creation), and it gives us Core
`Row` for read paths (which is fully immutable). The combination meets
the *underlying* requirement (safe-to-share-across-threads) without
delivering literal `frozen=True`.

**Verdict.** Frozen dataclass is an implementation choice. SA's
combination of `MappedAsDataclass(slots=True)` for writes and Core `Row`
for reads meets the actual requirement. Not a blocker.

---

## 6. Custom types — `TypeDecorator` is a clean fit

SA's [`TypeDecorator`](https://docs.sqlalchemy.org/en/20/core/custom_types.html#sqlalchemy.types.TypeDecorator)
adapts Python types to/from SQL columns. Sketches:

```python
class JobNameType(TypeDecorator):
    impl = String
    cache_ok = True
    def process_bind_param(self, value, dialect):
        return value.to_wire() if value is not None else None
    def process_result_value(self, value, dialect):
        return JobName.from_wire(value) if value is not None else None

class TimestampMsType(TypeDecorator):
    impl = Integer
    cache_ok = True
    def process_bind_param(self, value, dialect):
        return value.epoch_ms() if value is not None else None
    def process_result_value(self, value, dialect):
        return Timestamp.from_ms(int(value)) if value is not None else None

class JobConfigProtoType(TypeDecorator):
    impl = LargeBinary
    cache_ok = True
    def process_bind_param(self, value, dialect):
        return value.SerializeToString()
    def process_result_value(self, value, dialect):
        proto = job_pb2.JobConfig()
        proto.ParseFromString(value)
        return proto
```

This replaces `Column.decoder` from `schema.py:138` exactly. Roughly
**5–8 small TypeDecorator classes** would replace the
`schema.py`-decoder layer for us (~80 LOC each side, so ~zero net
change).

**Critical:** `cache_ok = True` is required for the SA 2.0 compilation
cache to work with these types. Forgetting this disables the cache and
caps per-query perf at ~30% of optimal ([SA 2.0 — Caching
Implications](https://docs.sqlalchemy.org/en/20/core/custom_types.html#sqlalchemy.types.TypeDecorator.cache_ok)).

**Verdict.** Clean fit. Roughly LOC-neutral vs. today's
column-decoder pattern.

---

## 7. Caching — what SA does and doesn't cover

### 7.1 ProtoCache (bytes-keyed decode memo)

SA's identity map is PK-keyed, not blob-keyed. **It cannot replace
ProtoCache.** We can wire ProtoCache into `JobConfigProtoType` via:

```python
class JobConfigProtoType(TypeDecorator):
    impl = LargeBinary
    cache_ok = True
    def process_result_value(self, value, dialect):
        return proto_cache.get_or_decode(value, _decode_jobconfig)
```

Same code, same singleton, lives behind a different abstraction. **Net
neutral.** Keeping our LRU is mandatory regardless of SA adoption.

### 7.2 Write-through caches (EndpointStore, _attr_cache)

SA does not have a built-in "in-memory dict mirroring a table" feature.
The pattern would be:

- `event.listen(Session, "after_commit", refresh_endpoints_cache)` —
  scans `session.new`/`session.dirty`/`session.deleted` for `Endpoint`
  instances and updates the cache.
- The listener must hold the cache's lock; the lock must be released
  after the listener finishes but before the controller's outer write
  lock releases (to match today's atomicity).

This is **brittle**:

- SA's `session.new`/`session.dirty`/`session.deleted` reflect *changes
  made through the ORM*. Bulk updates (`session.execute(update(...))`)
  bypass these sets — the listener wouldn't see them. We'd have to also
  listen to `do_orm_execute` ([SA 2.0 — Execute
  Events](https://docs.sqlalchemy.org/en/20/orm/session_events.html#execute-events))
  and parse the statement to detect endpoint mutations. Worse than
  today.
- The lock-release ordering is non-trivial. SA's `after_commit` runs
  after the DBAPI commit; today's `tx.on_commit` runs while
  `ControllerDB._lock` is held. To match, the SA `Session.commit()`
  call has to be wrapped in our own outer lock manually.

**SA's [Dogpile.cache integration](https://docs.sqlalchemy.org/en/20/orm/examples.html#examples-caching)**
is for *region-based query caching* — caching the *result* of a query
keyed on the query text. It's the wrong shape for our two tiny tables;
it would cache "the list of all endpoints" for 60 seconds but couldn't
invalidate on a specific endpoint mutation without doing the same
event-listener dance.

**Verdict.** Write-through caches remain custom code. SA gives us
hookable commit events but the gymnastics are worse than today's
explicit `tx.on_commit(fn)`. **Net loss.**

---

## 8. Migrations — Alembic is mostly a wash for us

[Alembic](https://alembic.sqlalchemy.org/en/latest/) is the SA team's
migration tool. Its main features:

- **Autogenerate:** compare declarative models to live DB; emit a
  migration script.
- **Branching / merging:** support for parallel migration trees.
- **DDL DSL:** `op.add_column`, `op.create_index`, etc., independent of
  SQL dialect.

For us:

- **Autogenerate vs. proto blobs:** works fine — proto columns are
  `LargeBinary`, Alembic sees a binary column.
- **Autogenerate vs. `JobName`-typed PKs:** works if `JobNameType` has
  a stable `compile()` output. Standard `TypeDecorator` subclasses do.
- **The SQLite ALTER TABLE limitation:** SQLite doesn't support
  `ALTER TABLE DROP COLUMN` until version 3.35.0 (March 2021), and
  `ALTER TABLE ALTER COLUMN` is still not supported. Alembic has a
  ["batch mode"](https://alembic.sqlalchemy.org/en/latest/batch.html)
  that emulates these by creating a temp table, copying data, dropping
  the original, and renaming. **Works, but the autogenerate output is
  often wrong** — it generates straight-line `op.drop_column` calls that
  fail on SQLite. The user must wrap them in `with op.batch_alter_table:`
  manually. This is well-documented but a constant footgun.
- **Hand-rolled migrations vs. Alembic:** today's migrations are
  idempotent `.py` files with `IF NOT EXISTS` guards and explicit
  schema-version tracking (`db.py:518–599`). Alembic's
  `alembic_version` table is the same idea, just standardized.

Concrete win/loss:

- **Win:** Alembic's history graph is real (`alembic history`, `alembic
  current`). Today's stem-based migration tracking works but is harder
  to inspect.
- **Win:** Future schema diffs are easier if we maintain declarative
  models.
- **Loss:** Autogenerate on SQLite is unreliable; we'd review and edit
  every autogenerated migration.
- **Loss:** Migration files become more verbose (Alembic boilerplate).
- **Loss:** One more tool to learn for new contributors.

**Verdict.** A wash. The migration story is not a reason to adopt SA.

---

## 9. Backup / restore lifecycle

Today: `ControllerDB.backup_to()` uses SQLite's online backup API
(`sqlite3.Connection.backup`). `replace_from()` closes connections,
swaps files, reopens, re-applies migrations, runs `_reopen_hooks`.

Under SA:

- `engine.dispose()` invalidates and closes the connection pool.
  ([SA 2.0 — Engine
  Disposal](https://docs.sqlalchemy.org/en/20/core/connections.html#engine-disposal)).
- Swap files on disk.
- `create_engine(new_url, ...)` to get a new engine; the old engine is
  garbage-collected.
- Apply migrations via Alembic programmatic API or hand-roll.
- Fire `_reopen_hooks` to rewarm caches.

The subtlety: SA's pool may have *in-use connections* (e.g. the
scheduler tick is running, holding a write connection). `engine.dispose()`
doesn't kill them; it marks them invalid for return to the pool. The
caller must ensure all sessions are closed before disposing. Today's
controller already serializes this (the `_lock` ensures no in-flight
writer when `replace_from` runs), so this is preservable.

**Verdict.** Workable, with mild care around session lifecycle around
`replace_from`. Not a blocker.

---

## 10. Snapshot isolation

Today's `db.read_snapshot()`:

```python
conn = self._read_pool.get()
try:
    conn.execute("BEGIN")
    yield QuerySnapshot(conn, lock=None)
finally:
    conn.rollback()
    self._read_pool.put(conn)
```

SA equivalent:

```python
@contextmanager
def read_snapshot(self):
    with self._read_engine.connect() as conn:
        conn.execute(text("BEGIN"))
        try:
            yield ReadSession(bind=conn, autoflush=False, expire_on_commit=False)
        finally:
            conn.rollback()
```

This works. The only subtlety: SA's `Connection` defaults to
"autocommit emulation" — it issues `BEGIN` lazily on the first execute.
To get the snapshot semantic exactly, we set `isolation_level=None` on
the engine and emit `BEGIN` ourselves
([SA 2.0 — SQLite
isolation](https://docs.sqlalchemy.org/en/20/dialects/sqlite.html#serializable-isolation-savepoints-transactional-ddl)).

**Verdict.** Preservable. The SQLite isolation-level configuration is
the relevant landmine.

---

## 11. Migration cost (one-time)

Honest estimate, line-by-line:

- **Delete:**
  - `db.py` `TransactionCursor`, `QuerySnapshot`, transaction management:
    ~150 LOC.
  - `schema.py` `Projection`, `adhoc_projection`, decoder helpers,
    `Column`/`Table` DDL generation: ~600 LOC. (Some — like the
    `Table` definitions themselves — would become declarative
    `Base`-subclass models of similar length.)
  - `stores.py` raw-SQL writes (every `tx.execute("INSERT INTO ...")`
    replaced by ORM `session.add(...)`): ~400 LOC delta. Many sites
    actually grow slightly because explicit `session.add` is verbose;
    others shrink.
  - **Estimated delete:** ~1000 LOC.
- **Add:**
  - Declarative models for ~12 tables: ~500 LOC.
  - `TypeDecorator` classes (`JobName`, `WorkerId`, `Timestamp`, 4–5
    proto types): ~150 LOC.
  - Engine + sessionmaker setup, isolation-level wiring, connect-time
    PRAGMAs, asymmetric read/write pool: ~80 LOC.
  - Session-lifecycle helpers (per-tick session context manager, RPC
    decorator): ~80 LOC.
  - `event.listen` plumbing for the two write-through caches: ~150 LOC.
  - Alembic env, autogenerate setup, migration of existing 46
    migrations to Alembic format: ~200 LOC.
  - **Estimated add:** ~1200 LOC.
- **Net:** roughly +200 LOC. Not a big win.

**Test work:**

- Most integration tests in `lib/iris/tests/cluster/controller/`
  exercise the full call graph via `ControllerDB`; they'd pass against
  a SA-backed `ControllerDB` if we preserve the public API.
- `test_endpoint_store.py`, `test_db.py` — direct unit tests against
  the abstractions we're changing. Substantial rewrite, ~300 LOC.
- New tests for SA-specific behavior (compilation cache, session
  lifecycle, event listener atomicity): ~200 LOC.

**Perf risk:**

- The partial indexes from `0045_index_task_attempts_live_workerbound`
  and `cb77a2877` aren't expressible in declarative models in a clean
  way — Alembic supports `Index(..., sqlite_where=...)` but it's
  ergonomically rough. The indexes themselves still work; we just have
  to declare them manually next to the model. Mild ergonomic loss, zero
  runtime loss.
- The hand-tuned queries (e.g. `resource_usage_by_worker`'s 380 ms →
  6 ms fix by pulling holder filtering into Python) translate
  one-for-one — SA emits the same SQL we'd write — *provided* we author
  them at the same level of detail (no implicit eager-loading of
  `JOIN`s). Risk: a careless `select(TaskAttempt)` with relationship
  preloading would re-introduce the regression. **Mitigation:**
  `EXPLAIN QUERY PLAN` test against every hot-path query, baked into
  the perf-baseline test from §7.3 of the original proposal.

**Migration time estimate:** **3–4 weeks of focused work** for one
engineer, including soak testing. Not 1 week; not 2 months. The bulk is
straightforward porting; the long tail is the write-through cache event
plumbing, the perf gates, and Alembic baselining.

---

## 12. What SA does NOT solve

Honesty list:

- **Bytes-keyed decode cache (ProtoCache):** still custom. Lives in a
  `TypeDecorator`'s `process_result_value`.
- **Write-through tiny-table cache:** still custom. The event-listener
  glue is *worse* than today's `tx.on_commit`.
- **Backup-restore-rewarm dance:** still custom. SA simplifies engine
  disposal slightly; the cache-rewarm logic doesn't change.
- **SQLite-specific PRAGMA tuning:** still our problem. SA gives us
  `event.listens_for(engine, "connect")` to install them; doesn't pick
  good values.
- **Recursive CTE for priority bands:** SA's
  `select(...).cte(recursive=True)` exists but is verbose. Real
  codebases use `conn.execute(text("WITH RECURSIVE ..."))` for these.
  No win.
- **Partial indexes on hot tables:** declarable in models but the
  ergonomics are rough. Net neutral.
- **Snapshot-isolated read pool:** SA doesn't have a first-class
  primitive; we re-implement on top of `engine.connect()` with
  manual `BEGIN`.
- **The five mechanisms in the original proposal:** SA replaces zero of
  them as-is. It restructures *how* we'd implement them (declarative
  schema, `Session` instead of `TransactionCursor`, `event.listen`
  instead of `tx.on_commit`), but doesn't reduce the *number* of
  mechanisms in the code.

---

## 13. Recommendation

**Do not adopt full SA ORM. The right level of commitment is:
nothing**.

### 13.1 Why

The honest comparison, given the requirements in §0 and what SA buys:

| Concern | SA Helps | SA Hurts | SA Neutral |
|---|---|---|---|
| Hot-path perf (§3) | Compilation cache (mild) | ORM mapped instances bust budget | Core-style queries match today |
| Threading (§4) | — | Asymmetric pool needs re-impl | — |
| Custom types (§6) | TypeDecorator is clean | — | — |
| ProtoCache (§7.1) | — | — | Still custom |
| Write-through (§7.2) | — | Event-listener glue is brittle | — |
| Identity map (§1) | Mild, on PK-lookup workloads we don't have | — | — |
| Migrations (§8) | History graph is nice | Alembic-on-SQLite is footgun-prone | — |
| Snapshot isolation (§10) | — | SQLite isolation config is fiddly | — |
| Backup/restore (§9) | — | — | Net neutral |
| Recursive CTEs (§Misc) | — | — | We'd still use raw SQL |

The wins are **declarative schema** (modest, given we already have
typed `Table` + `Column` definitions in `schema.py`), **TypeDecorator**
(real but small), and **compilation cache** (real on hot paths).

The losses are **ORM mapped instance overhead on hot paths** (forces us
to mix Core and ORM, eroding the readability win), **write-through cache
plumbing gets worse** (the one thing today's design does well — atomic
commit-hook semantics — becomes harder under SA's event model), and
**~3–4 weeks of migration work for ~+200 LOC and zero net mechanism
reduction**.

The deepest reason to decline: **SA's identity-map + UoW + session
lifecycle is designed for *entity-centric* workloads where you load a
single aggregate, mutate its fields, and commit. Our workload is *bulk
SQL with hand-tuned query plans on hot paths, plus tiny in-memory
mirrors for two tables*.** SA optimizes for the case we don't have.

### 13.2 What about SA Core only?

SA Core (the `text()`, `Table`, `select()` builders) without the ORM is
a smaller commitment. It buys:

- The compilation cache (real perf win).
- Type-safe column references in queries.
- A consistent DDL model (closer to today's `Column`/`Table` than full
  ORM).

It doesn't buy:

- The identity map or UoW (we don't need them).
- Alembic (we'd need to set it up separately if we want migrations).

**Net:** maybe **mildly worth doing as a future migration**, but the
compilation-cache win is only a few percent at our query rates, and
today's hand-written SQL is already cached at the prepared-statement
level by sqlite3. **Not a near-term win.** Defer.

### 13.3 What about "SA as just the engine"?

`text("SELECT ...")` + `conn.execute(...)` with no `Table` declarations
at all is essentially what we have today, dressed in SA's connection
abstraction. The win is `engine.dispose()` instead of manual connection
management. **Trivial; not worth the dependency.**

### 13.4 What we should do instead

**Adopt the slim-shape refactor from the critique doc:**

- Two classes (`Query` + `Projection`) plus a `cached=True` column
  flag.
- Raw SQL as a first-class escape hatch via `Query.fetch_raw(tx, sql,
  params)`.
- `views/` directory organized by use-case (not by entity) for
  multi-step reads.
- Explicit FK-cascade-into-Projection rule, checked at startup.
- Drop the `tx`-for-symmetry argument on Projection reads.

This delivers the cognitive simplification the original proposal aims
at, with zero perf risk, ~3 weeks of focused work instead of ~4, and no
new third-party dependency. The five-mechanism diagnosis is real; the
cure is internal cleanup, not external framework adoption.

### 13.5 When would we revisit SA?

Three scenarios:

1. **We outgrow SQLite and migrate to Postgres.** Then SA's
   cross-dialect support pays off and the SQLite-specific landmines go
   away. (Today this is a non-goal per `20260310_iris_sql_canonical.md`.)
2. **We need to expose the controller DB to non-Python clients with a
   typed schema.** Declarative SA models would generate Alembic-style
   DDL that other tools can consume.
3. **The data layer grows past ~25 tables and the manual DDL becomes
   error-prone.** Today we have 12; the slope is shallow.

None of these are near. SA is a future option, not a present one.

---

## 14. Strongest objections I couldn't fully answer

In the interest of intellectual honesty:

1. **Hot-path Core-mode perf numbers are my estimates, not measurements
   from our workload.** SA 2.0's published numbers say ~1 µs/row for
   Core access; on our specific hot paths (1k rows, multiple JOINs,
   custom TypeDecorators) the realized cost could be 2–3× that. A
   bench-quality answer requires a spike. If a spike shows Core access
   is within 2 ms of today's tuple-unpack on the worst hot path, the
   "SA hurts hot paths" objection weakens significantly.

2. **The write-through-cache atomicity story under SA's
   `after_commit` is not as bad as I made it sound** *if* we are
   willing to wrap `Session.commit()` in our own write-lock context
   manager. That's 10–20 LOC of boilerplate. The "brittle" framing in
   §7.2 assumes we don't do that; if we do, the gap between SA and
   today's `tx.on_commit` narrows considerably. The slim-refactor
   alternative is still simpler, but the gap is smaller than my §7.2
   suggests.

3. **The migration LOC delta of "+200" is optimistic.** Declarative
   models tend to balloon under real schema (custom indexes, partial
   indexes, triggers, table constraints, named uniqueness constraints).
   A pessimistic delta is +600. That doesn't change the recommendation,
   but it does shift the "3–4 weeks" estimate to "5–6 weeks."

These objections don't reverse the recommendation, but they're real and
the user should know the answer hinges on a bench-quality spike if they
ever want to revisit.

---

## 15. References

SA 2.x docs:
- [SA 2.0 — Session Basics](https://docs.sqlalchemy.org/en/20/orm/session_basics.html)
- [SA 2.0 — Session API](https://docs.sqlalchemy.org/en/20/orm/session_api.html)
- [SA 2.0 — Session Events](https://docs.sqlalchemy.org/en/20/orm/session_events.html)
- [SA 2.0 — Contextual/Thread-local Sessions](https://docs.sqlalchemy.org/en/20/orm/contextual.html)
- [SA 2.0 — SELECT tutorial](https://docs.sqlalchemy.org/en/20/tutorial/data_select.html)
- [SA 2.0 — ORM Bulk UPDATE/DELETE](https://docs.sqlalchemy.org/en/20/orm/queryguide/dml.html#orm-enabled-update-and-delete-statements)
- [SA 2.0 — Custom Types / TypeDecorator](https://docs.sqlalchemy.org/en/20/core/custom_types.html)
- [SA 2.0 — Declarative Dataclass Mapping](https://docs.sqlalchemy.org/en/20/orm/dataclasses.html)
- [SA 2.0 — SQL Compilation Caching](https://docs.sqlalchemy.org/en/20/core/connections.html#sql-compilation-caching)
- [SA 2.0 — SQLite isolation / Serializable](https://docs.sqlalchemy.org/en/20/dialects/sqlite.html#serializable-isolation-savepoints-transactional-ddl)
- [SA 2.0 — Result Rows](https://docs.sqlalchemy.org/en/20/core/connections.html#sqlalchemy.engine.Row)
- [SA 2.0 — What's New (2.0)](https://docs.sqlalchemy.org/en/20/changelog/whatsnew_20.html)
- [SA 2.0 — Examples / Caching with Dogpile](https://docs.sqlalchemy.org/en/20/orm/examples.html#examples-caching)
- [SA Discussion #9192 — Frozen dataclass mapping](https://github.com/sqlalchemy/sqlalchemy/discussions/9192)

Alembic:
- [Alembic — main docs](https://alembic.sqlalchemy.org/en/latest/)
- [Alembic — batch operations for SQLite](https://alembic.sqlalchemy.org/en/latest/batch.html)

Codebase references:
- `lib/iris/src/iris/cluster/controller/db.py:240–510` — TransactionCursor, ControllerDB, read pool
- `lib/iris/src/iris/cluster/controller/schema.py:34–66` — ProtoCache
- `lib/iris/src/iris/cluster/controller/schema.py:316–456` — Projection
- `lib/iris/src/iris/cluster/controller/stores.py:1624` — resource_usage_by_worker
- `lib/iris/src/iris/cluster/controller/stores.py:1679` — reconcile_rows_for_workers
- `lib/iris/src/iris/cluster/controller/migrations/0045_index_task_attempts_live_workerbound.py`
- Companion: `.agents/projects/20260511_iris_store_view_refactor.md`
- Companion: `.agents/projects/20260511_iris_store_view_refactor_addendum_fit.md`
- Companion: `.agents/projects/20260511_iris_store_view_refactor_critique.md`
