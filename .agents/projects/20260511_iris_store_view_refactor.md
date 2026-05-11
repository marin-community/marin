# Iris Controller Data Layer Refactor: Query / CachedQuery / Projection

**Date:** 2026-05-11
**Author:** russell.power@gmail.com
**Status:** Draft — proposal
**Related prior work:** `.agents/projects/20260310_iris_sql_canonical.md` (SQL-as-source-of-truth), `.agents/projects/iris-sql-store.md` (initial Stores design)
**Companion research:** see "References" at the end of this document for the full survey of industry paradigms and codebase audit notes that informed the design.

---

## 1. Summary

The Iris controller's data layer has accumulated five distinct mechanisms for moving rows between SQLite and Python:

1. `Projection` (today's name for "select a column subset and decode into a frozen dataclass") — `schema.py:316–456`
2. `ProtoCache` (bounded LRU keyed by raw blob bytes) — `schema.py:34–66`
3. `EndpointStore` write-through dict cache — `stores.py:95–337`
4. `ControllerDB._attr_cache` write-through dict — `db.py:329–379`
5. Partial indexes as a substitute for materialized views — e.g. `migrations/0045_index_task_attempts_live_workerbound.py`

Each works in isolation. Together they form a five-way decision tree for "how do I read entity X?", and writers must remember which caches their writes invalidate. The cost is cognitive, not performance — performance is fine — but new contributors meet five idioms before they ship one query.

This document proposes collapsing these into **three named classes plus a `writes/` module convention**:

- **`Query[Row]`** — stateless read. Runs SQL each time. Replaces today's plain `Projection`.
- **`CachedQuery[Row]`** — read whose decoded rows are memoized by raw blob bytes. Replaces today's `Projection + ProtoCache` combination.
- **`Projection[Row]`** — write-through materialized view. Owns its mutating methods (because the SQL write and the in-memory dict update must commit atomically). Replaces `EndpointStore` and `_attr_cache`. Has lifecycle hooks: `rehydrate()` at startup and after restore.
- **`writes/<entity>.py`** — module-level functions taking a `tx`. Replaces the write methods on `JobStore`, `TaskStore`, `TaskAttemptStore`, `WorkerStore`. Class organization disappears.

**Uniform `tx` API.** Every read and write in the data layer takes a `tx` as its first parameter, including Projection lookups (which serve from in-memory dicts and don't *need* the tx, but accept it for API symmetry — see §4.6). The mental model is: "the tx is the handle for everything I do against the DB." The cache class of the read determines whether the tx is consulted for snapshot isolation; the doc spells out the contract explicitly.

Industry-wise this is conventional: it is **asymmetric CQRS with inline projections**, exactly the pattern Marten and other CQRS-aware systems use, with the Python-orchestrator-standard "writes are module functions" convention for the non-projected case. The originality is the explicit class split rather than a single `View` class with a policy enum — explicitly endorsed by `AGENTS.md`'s "use separate classes over boolean flags for variant behavior" rule.

**Execution plan**: one stacked PR, twelve commits, each independently revertable and tested. Final commit deletes the obsolete classes. End-to-end smoke test on a real controller runs before merge.

---

## 2. Motivation and Background

### 2.1 The five mechanisms in detail

**`Projection` (`schema.py:316–456`)** — Compiled view of a `Table`'s columns. Resolves a column-name tuple at import time, validates types, and produces a tuple of decoder callables. Used by `JOB_DETAIL_PROJECTION`, `JOB_RESERVATION_PROJECTION`, `TASK_ROW_PROJECTION`, etc. At read sites the pattern is:

```python
row = tx.fetchone(
    f"SELECT {JOB_DETAIL_PROJECTION.select_clause()} "
    f"FROM jobs j {JOB_CONFIG_JOIN} WHERE j.job_id = ?",
    (job_id.to_wire(),),
)
return JOB_DETAIL_PROJECTION.decode_one([row])
```

The Projection name is overloaded with CQRS's usage and will be renamed to `Query` in this proposal. The CQRS sense (write-through materialized read model) takes over the `Projection` name.

**`ProtoCache` (`schema.py:34–66`)** — Bounded LRU (8192 entries, 25% eviction batches) keyed on raw blob bytes:

```python
class ProtoCache:
    def get_or_decode(self, blob: bytes, decoder: Callable[[bytes], Any]) -> Any:
        with self._lock:
            result = self._cache.get(blob)
            if result is not None:
                return result
        decoded = decoder(blob)
        with self._lock:
            if len(self._cache) >= self._max_size:
                to_evict = self._max_size // 4
                for k in list(self._cache.keys())[:to_evict]:
                    del self._cache[k]
            self._cache[blob] = decoded
        return decoded
```

Wrapped into specific column decoders via `cached=True` on `Column`. Effectively a content-addressed identity map for immutable protos.

**`EndpointStore` (`stores.py:95–337`)** — Tiny read-mostly table (hundreds of rows). Reads never touch SQL; three in-memory dicts (`_by_id`, `_by_name`, `_by_task`) are populated at startup via `_load_all()` and updated via post-commit hooks:

```python
def add(self, cur: TransactionCursor, endpoint: EndpointRow) -> AddEndpointOutcome:
    cur.execute("INSERT OR REPLACE INTO endpoints(...) VALUES (...)", (...))
    def apply() -> None:
        with self._lock:
            self._unindex(endpoint.endpoint_id)
            self._index(endpoint)
    cur.on_commit(apply)
    return AddEndpointOutcome.OK
```

Eliminated dashboard CPU dominated by `ListEndpoints` walking the WAL.

**`ControllerDB._attr_cache` (`db.py:329–379`)** — Same shape as `EndpointStore` but bolted directly onto `ControllerDB` rather than a dedicated class. Lazy-populated dict of `{worker_id: {attr_name: attr_value}}`. Written through via `set_worker_attributes(worker_id, attrs)` called from `on_commit` hooks in `transitions.py:1296` (registration), `transitions.py:1911`/`transitions.py:2178` (deletion).

**Partial indexes (`migrations/0045*`)** — Not a Python cache, but a planner-level mechanism playing the same role as a materialized view:

```sql
CREATE INDEX idx_task_attempts_live_workerbound
ON task_attempts(worker_id)
WHERE worker_id IS NOT NULL AND finished_at_ms IS NULL;
```

Drives `resource_usage_by_worker` from ~1k live rows instead of 24k jobs. 350 ms → 6.5 ms. **These stay** — they are an index optimization, not a Python-layer caching strategy.

### 2.2 What the audit confirmed

(Detailed agent findings in §9.)

- **No write method touches more than 3 tables.** The largest is `WorkerStore.remove` (3 tables). Most are 1; a handful are 2. Class organization is unnecessary for transactional coherence.
- **`TransactionCursor` does not track written tables.** Adding auto-invalidation would require SQL parsing. We will not do this; explicit registration via Projection.sources is sufficient.
- **No test mocks any Store class.** Zero matches for `mock.*Store` / `patch.*Store` across `lib/iris/tests/`. Killing the Store classes is test-safe.
- **`_attr_cache` is a true write-through cache** with the same shape as `EndpointStore`. There are exactly two write-through caches in the system today.

### 2.3 What the industry survey concluded

(Detailed survey in §10.)

- **Repository pattern (Fowler/Evans):** Repository owns both reads and writes for one aggregate type. Conventional but not what we want — we want explicit caching variants.
- **CQRS (Greg Young):** Reads and writes use different models. The "soft CQRS" of same-DB, separate-classes is mainstream and recommended by Cosmic Python.
- **Inline projections (Marten):** A read model whose update commits in the same transaction as the source write. **This is exactly our WRITE_THROUGH semantics.** Marten calls these `Inline` projections.
- **Python orchestrator codebases (Airflow, Prefect, Dagster, Sentry):** All use module-level functions taking a session for writes. None have a Stores class layer. Dagster has Storage interfaces (RunStorage, EventLogStorage) but the internals are functions.
- **Style guides:** Cosmic Python and our own `AGENTS.md` both say "separate classes over boolean flags for variant behavior." A `View(cache=NONE)` and `View(cache=WRITE_THROUGH)` are *behaviorally different objects* — one is stateless, one owns mutations and lifecycle. Split them.

**Verdict from the survey:** mildly idiosyncratic but not eccentric. We are within the mainstream of CQRS-aware projections + Python-idiomatic free-function writes.

---

## 3. Goals and Non-Goals

### Goals

- **One way to read.** Every read in the controller goes through exactly one of `Query`, `CachedQuery`, or `Projection`. The cache policy is the *class*, not a configuration.
- **One way to write.** Plain writes are module-level functions. Writes that affect a `Projection` are methods on the Projection.
- **Reduce surface area.** Delete `ProtoCache` (replaced by `CachedQuery`), `EndpointStore` (replaced by a `Projection`), `_attr_cache` from `ControllerDB` (replaced by a `Projection`), and `JobStore` / `TaskStore` / `TaskAttemptStore` / `WorkerStore` (replaced by `writes/*.py` modules). Net deletion target: ~1500 lines.
- **Preserve performance.** Every benchmarked hot path (`resource_usage_by_worker`, `reconcile_rows_for_workers`, `_jobs_with_reservations`, `ListEndpoints`) must show no regression after the refactor.
- **Preserve atomicity.** The current `EndpointStore` semantics — SQL commit and in-memory dict update happen atomically under the write lock — must be preserved exactly.

### Non-goals

- **No ORM adoption.** SQLAlchemy ORM cannot map frozen dataclasses; we keep the typed-dataclass approach.
- **No async.** SQLite is sync; the controller's threading model is correct.
- **No new query DSL.** Raw SQL strings stay. This is a refactor of the row-decoding and caching layer, not the query-authoring layer.
- **No Postgres migration.** SQLite is the right backend for a single-process controller.
- **No automatic SQL-parsing invalidation.** Explicit Projection.sources declarations plus a startup-time owned-table guard. Lower magic.
- **No migration framework change.** Hand-rolled `.py` migrations stay.

---

## 4. Architecture

### 4.1 Class hierarchy

```
                ┌──────────────────────────────────┐
                │       ReadSpec[Row] (ABC)        │
                │ ──────────────────────────────── │
                │  name: str                       │
                │  sources: tuple[Table, ...]      │
                │  select: str       (SQL fragment) │
                │  row_cls: type[Row]              │
                │  decoders: tuple[Callable, ...]  │
                │  one(tx, ...) -> Row | None      │
                │  many(tx, ...) -> list[Row]      │
                └────────────────┬─────────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
┌───────▼─────────┐    ┌─────────▼─────────┐    ┌─────────▼──────────┐
│     Query       │    │   CachedQuery     │    │     Projection     │
│ ─────────────── │    │ ───────────────── │    │ ────────────────── │
│ Stateless.      │    │ Decoder result    │    │ In-memory dict(s). │
│ Always SQL.     │    │ memoized by blob  │    │ No SQL on read.    │
│                 │    │ bytes (LRU 8192). │    │ Owns mutations.    │
│                 │    │ Always SQL,       │    │ Atomic w/ on_commit│
│                 │    │ decode is cached. │    │ hooks.             │
│                 │    │                   │    │ rehydrate() on     │
│                 │    │                   │    │   startup/restore. │
└─────────────────┘    └───────────────────┘    └────────────────────┘
                                                       │
                                          ┌────────────┴────────────┐
                                          │                         │
                              ┌──────────────────────────┐   ┌────────────────────────────┐
                              │ EndpointsProjection      │   │ WorkerAttrsProjection      │
                              │   .add(tx, row)          │   │   .set(tx, wid, ...)       │
                              │   .remove(tx, eid)       │   │   .remove(tx, wid)         │
                              │   .by_id(tx, eid)        │   │   .get(tx, wid)            │
                              │   .by_name(tx, name)…    │   │   .query  (Query fallback) │
                              │   .query  (Query fallback)│   │                            │
                              └──────────────────────────┘   └────────────────────────────┘
```

`ReadSpec` is the shared base only because the three subclasses share metadata (sources, decoders, row_cls). The user-facing distinction is "which subclass did the author choose?" — not "which value of a field did they pass?"

### 4.2 Read flow

```
   caller (RPC / scheduler / dashboard)
        │
        │   <something>.one(tx, ...)   ← every read takes tx
        ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │                                                                    │
   │  Query.one(tx, **params):                                          │
   │      cursor = tx.execute(self.full_sql, params)                    │
   │      row = cursor.fetchone()                                       │
   │      return self._decode(row)                                      │
   │      # tx is the SQL execution context (snapshot-isolated).        │
   │                                                                    │
   │  CachedQuery.one(tx, **params):                                    │
   │      cursor = tx.execute(self.full_sql, params)                    │
   │      row = cursor.fetchone()                                       │
   │      return self._decode_with_cache(row)                           │
   │      # tx is the SQL execution context; cache is decode-only.      │
   │                                                                    │
   │  EndpointsProjection.by_id(tx, endpoint_id):                       │
   │      del tx  # accepted for API symmetry; not consulted            │
   │      with self._lock:                                              │
   │          return self._by_id.get(endpoint_id)                       │
   │      # Returns latest-committed state. See §4.6 for the contract.  │
   │                                                                    │
   └────────────────────────────────────────────────────────────────────┘
```

Every read takes the same `tx` handle. What the tx *does* differs by class (§4.6). A reader scanning a call site sees `something.one(tx, ...)` and knows "this is a DB-layer access"; whether the read is snapshot-isolated is then determined by which class `something` is.

### 4.3 Write flow

Two cases, distinguished by whether the table is owned by a Projection.

**Case A — unowned table (the common case):**

```
   caller
        │
        │   writes.jobs.insert_job(tx, job, config, budget)
        ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │ writes/jobs.py                                                  │
   │ ─────────────                                                   │
   │ @writes_to(JOBS, JOB_CONFIG, USER_BUDGETS)                      │
   │ def insert_job(tx, job, config, budget):                        │
   │     tx.execute("INSERT INTO jobs ...", (...))                   │
   │     tx.execute("INSERT INTO job_config ...", (...))             │
   │     tx.execute("INSERT INTO user_budgets ...", (...))           │
   │                                                                 │
   │ Plain function. No cache concerns. No class.                    │
   └─────────────────────────────────────────────────────────────────┘
```

The `@writes_to` decorator does not enforce or invalidate anything at runtime — it records metadata used by the startup-time owned-table check (§4.5).

**Case B — owned table (writes to a Projection-managed table):**

```
   caller
        │
        │   endpoints.add(tx, endpoint)
        ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │ EndpointsProjection.add(tx, endpoint):                          │
   │     tx.execute("INSERT OR REPLACE INTO endpoints ...", (...))   │
   │     def apply():                                                │
   │         with self._lock:                                        │
   │             self._unindex(endpoint.endpoint_id)                 │
   │             self._index(endpoint)                               │
   │     tx.on_commit(apply)                                         │
   │                                                                 │
   │ Atomic w.r.t. IMMEDIATE transaction commit.                     │
   └─────────────────────────────────────────────────────────────────┘
```

This is byte-for-byte equivalent to today's `EndpointStore.add`. The class moved; the semantics did not.

### 4.4 Lifecycle

```
┌─────────────────────────────────────────────────────────────────────┐
│  ControllerDB.__init__                                              │
│    │                                                                │
│    ├── open connections, set pragmas                                │
│    ├── apply_migrations()                                           │
│    ├── for P in PROJECTIONS:                                        │
│    │       P.rehydrate(self)        # populate in-memory state      │
│    │                                # equivalent to today's _load_all │
│    │                                # for EndpointStore, and lazy   │
│    │                                # _populate_attr_cache for      │
│    │                                # worker attrs (now eager).     │
│    └── register reopen-hook with self                               │
│                                                                     │
│  ControllerDB.replace_from(backup):                                 │
│    │                                                                │
│    ├── close, replace files, reopen                                 │
│    ├── apply_migrations()                                           │
│    ├── for P in PROJECTIONS:                                        │
│    │       P.rehydrate(self)        # refresh against new file      │
│    └── ready                                                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

`PROJECTIONS` is a module-level registry populated by `Projection.__init__` registering itself. The runtime does not look up Projections by name from caller code — it iterates the registry only for `rehydrate` and the startup check.

### 4.5 Invalidation strategy — explicit, not magic

A `Projection` declares its source tables (`sources: tuple[Table, ...]`). At startup, after all `writes/` modules are imported:

```python
def assert_owned_tables_not_externally_written() -> None:
    owned: dict[Table, type[Projection]] = {}
    for P in PROJECTIONS:
        for table in P.sources:
            owned[table] = type(P)

    for fn in REGISTERED_WRITE_FUNCTIONS:
        for table in fn.writes_to:
            if table in owned:
                raise ConfigurationError(
                    f"Write function {fn.__qualname__} declares writes to "
                    f"{table.name}, which is owned by {owned[table].__name__}. "
                    f"Move this write onto the Projection."
                )
```

This runs once at import time, fails the controller startup loudly if an invariant is violated, and has zero runtime cost. Together with the `@writes_to(...)` decorator, it provides a static guarantee that no plain write function silently invalidates a Projection's cache.

The check is conservative: a write function legitimately reading from an owned table but not writing it is fine, since `writes_to` declares writes only.

### 4.6 Consistency model — what `tx` means for each read class

All reads accept a `tx` (or read-snapshot) as their first argument. This is a deliberate API uniformity choice: a reader scanning code should see `xxx.one(tx, ...)` everywhere and immediately recognize it as a DB-layer call. The semantic of the tx, however, varies by class:

| Class | What `tx` does | Consistency offered |
|---|---|---|
| `Query` | Issues SQL through it. Inside a `QuerySnapshot`, the query reads from that snapshot. | Snapshot-isolated against everything else in the same snapshot. |
| `CachedQuery` | Same as `Query` — SQL goes through tx. The bytes-keyed memoization is orthogonal to the snapshot. | Snapshot-isolated against everything else in the same snapshot. |
| `Projection` | `tx` is accepted for API symmetry but **not consulted to choose what data to return**. Reads serve latest-committed state from the in-memory dict. | "At-least-as-recent-as the latest committed write to the source table." Not snapshot-isolated. |

This means a caller mixing `Query` and `Projection` reads inside a single `QuerySnapshot` may observe state that disagrees with itself: the `Query` sees rows at T₁ (snapshot start); the `Projection` reflects whatever is currently in the dict, which may be at T₂ ≥ T₁.

For our two Projections today (`endpoints`, `worker_attrs`) this is acceptable — neither is in a scheduling-tick loop that needs per-tick snapshot isolation, and the data they hold (endpoint metadata, worker attribute strings) doesn't participate in the consistency-critical state transitions.

**Escape hatch for snapshot-isolated reads of Projection-backed data.** Every `Projection` exposes a `.query` attribute — an auto-generated `Query` against the same underlying table that bypasses the dict:

```python
# Default — latest-committed (fast, in-memory):
endpoint = projections.endpoints.by_id(tx, endpoint_id)

# Snapshot-isolated against the rest of this tx (rare, hits SQL):
endpoint = projections.endpoints.query.one(
    tx, where="endpoint_id = ?", params=(endpoint_id,)
)
```

Both calls take the same `tx`. The difference is which read class the caller chose, and the caller is explicit about it.

**Why not make Projection reads snapshot-isolated?** Maintaining a versioned/MVCC history of the in-memory dict so it could serve old snapshots is implementable (each commit advances a version counter; the dict keeps recent past versions; tx records its start version) but the cost is real (memory, copy-on-write, eviction policy) and our hot Projections are tiny tables. The escape hatch — fall back to `.query` — is enough.

**Why accept `tx` if it's not consulted?** Two reasons. First, API symmetry: a Projection lookup that took *no* arguments would bifurcate the call-site surface ("some reads take tx, some don't") and reintroduce the cognitive overhead this refactor is trying to remove. Second, if we ever extend Projections to be snapshot-aware (a fourth read class, `VersionedProjection`, for example), no caller code needs to change — the signature is already correct.

**Worked example: dashboard "list endpoints" RPC.**

```python
def list_endpoints(request, tx):
    # Single read, no consistency concerns with other reads in this RPC.
    # Default path: in-memory dict. Microseconds.
    return [e for e in projections.endpoints.all(tx)
            if e.task_id == request.task_id]
```

**Worked example: scheduler tick mixing endpoint and job state.**

```python
def reconcile(tx_snap):
    # We want jobs and endpoints to agree on "what was true at the snapshot
    # start." Use the .query escape hatch on endpoints to get SQL-isolated
    # reads against the same tx_snap.
    jobs = JOB_DETAIL_QUERY.many(tx_snap, where="state = ?", params=(RUNNING,))
    endpoints = projections.endpoints.query.many(tx_snap)  # SQL, snapshot-iso
    ...
```

The choice between `.by_id(tx)` (fast, latest-committed) and `.query.one(tx, ...)` (snapshot-isolated) is local to each call site and reviewable in code.

### 4.7 What the `tx` interface looks like

Unchanged. `TransactionCursor` keeps its current `execute`, `executemany`, `fetchone`, `fetchall`, `on_commit` API (`db.py:240–288`). No table tracking is added. We do not add SQL parsing. This is deliberate — the explicit `@writes_to`/`Projection.sources` declarations are the source of truth.

---

## 5. API Sketches

### 5.1 `ReadSpec` and `Query`

```python
# In a new file: lib/iris/src/iris/cluster/controller/views.py

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar
import sqlite3

Row = TypeVar("Row")

@dataclass(frozen=True)
class ReadSpec(Generic[Row]):
    """Shared metadata for all read paths."""
    name: str                            # "jobs.detail", "endpoints.by_id"
    sources: tuple[Table, ...]
    select_columns: tuple[Column, ...]
    row_cls: type[Row]
    from_clause: str                     # "jobs j LEFT JOIN job_config jc ..."

    @property
    def select_clause(self) -> str:
        return ", ".join(f"{self._alias_for(c)}.{c.name}" for c in self.select_columns)


class Query(Generic[Row]):
    """Stateless read. No caching."""

    def __init__(self, spec: ReadSpec[Row]):
        self._spec = spec
        # Precompile decoders, validate columns at import time.
        self._decoders = tuple(_resolve_decoder(c) for c in spec.select_columns)

    def one(self, tx, where: str, params: tuple) -> Row | None:
        row = tx.fetchone(
            f"SELECT {self._spec.select_clause} "
            f"FROM {self._spec.from_clause} WHERE {where}",
            params,
        )
        return self._decode(row) if row is not None else None

    def many(self, tx, where: str = "1=1", params: tuple = ()) -> list[Row]:
        rows = tx.fetchall(
            f"SELECT {self._spec.select_clause} "
            f"FROM {self._spec.from_clause} WHERE {where}",
            params,
        )
        return [self._decode(r) for r in rows]

    def _decode(self, row: sqlite3.Row) -> Row:
        return self._spec.row_cls(*(dec(v) for dec, v in zip(self._decoders, row)))
```

Today's `JOB_DETAIL_PROJECTION` becomes `JOB_DETAIL_QUERY = Query(ReadSpec(...))`.

### 5.2 `CachedQuery`

```python
class CachedQuery(Query[Row]):
    """Read whose decoded rows are memoized by raw blob bytes.

    The cache is keyed on the tuple of bytes-valued column values, not on
    the row's primary key. This means two rows with identical blobs share
    a decoded Python object — the same invariant ProtoCache held.
    """

    _cache: dict[bytes, Row] = {}
    _lock = threading.Lock()
    _MAX_SIZE = 8192

    def _decode(self, row: sqlite3.Row) -> Row:
        cache_key = self._cache_key(row)
        if cache_key is not None:
            with self._lock:
                hit = self._cache.get(cache_key)
                if hit is not None:
                    return hit
        decoded = super()._decode(row)
        if cache_key is not None:
            with self._lock:
                if len(self._cache) >= self._MAX_SIZE:
                    # LRU-ish: drop oldest 25%
                    for k in list(self._cache.keys())[: self._MAX_SIZE // 4]:
                        del self._cache[k]
                self._cache[cache_key] = decoded
        return decoded
```

`JOB_RESERVATION_QUERY = CachedQuery(ReadSpec(...))`. Replaces `JOB_RESERVATION_PROJECTION` + the column-level `cached=True` flag + the global `ProtoCache` instance.

### 5.3 `Projection`

```python
class Projection(Generic[Row]):
    """A materialized read model with a write-through in-memory cache.

    Subclasses define the in-memory dict structure and the mutation methods
    that update both SQL and the dict atomically. All read methods take a
    `tx` as first argument for API symmetry; the tx is not consulted to
    select data (see §4.6). For snapshot-isolated reads against the same
    underlying table, callers use `self.query.one(tx, ...)`.
    """

    name: str                            # set by subclass
    sources: tuple[Table, ...]           # set by subclass
    query: Query[Row]                    # auto-generated SQL fallback

    def __init__(self, db: "ControllerDB", query: Query[Row]):
        self._db = db
        self._lock = threading.RLock()
        self.query = query               # snapshot-isolated escape hatch
        PROJECTIONS.append(self)

    def rehydrate(self, db: "ControllerDB") -> None:
        """Populate in-memory state from SQL. Called at startup and after restore."""
        raise NotImplementedError


class EndpointsProjection(Projection["EndpointRow"]):
    name = "endpoints"
    sources = (ENDPOINTS,)

    def __init__(self, db):
        super().__init__(db, query=ENDPOINTS_QUERY)
        self._by_id: dict[str, EndpointRow] = {}
        self._by_name: dict[str, set[str]] = {}
        self._by_task: dict[JobName, set[str]] = {}
        self.rehydrate(db)

    def rehydrate(self, db) -> None:
        with self._lock, db.read_snapshot() as snap:
            self._by_id.clear(); self._by_name.clear(); self._by_task.clear()
            for row in self.query.many(snap):
                self._index(row)

    # ─── Mutations (require tx) ────────────────────────────────────────
    def add(self, tx, endpoint: EndpointRow) -> AddEndpointOutcome:
        tx.execute("INSERT OR REPLACE INTO endpoints(...) VALUES (...)", (...))
        def apply():
            with self._lock:
                self._unindex(endpoint.endpoint_id)
                self._index(endpoint)
        tx.on_commit(apply)
        return AddEndpointOutcome.OK

    def remove(self, tx, endpoint_id: str) -> None:
        tx.execute("DELETE FROM endpoints WHERE endpoint_id = ?", (endpoint_id,))
        def apply():
            with self._lock:
                self._unindex(endpoint_id)
        tx.on_commit(apply)

    # ─── Reads (take tx for API symmetry; serve from dict) ─────────────
    def by_id(self, tx, endpoint_id: str) -> EndpointRow | None:
        del tx  # accepted for symmetry; not consulted (see §4.6)
        with self._lock:
            return self._by_id.get(endpoint_id)

    def by_name(self, tx, name: str) -> list[EndpointRow]:
        del tx
        with self._lock:
            ids = self._by_name.get(name, set())
            return [self._by_id[i] for i in ids if i in self._by_id]

    def all(self, tx) -> list[EndpointRow]:
        del tx
        with self._lock:
            return list(self._by_id.values())

    # ... _index, _unindex helpers as in today's EndpointStore
```

This is `EndpointStore` with the class renamed and the read methods now living on the same class as the writes — which is the truth they already represented (both touched the same dicts).

### 5.4 `writes/` module convention

```python
# lib/iris/src/iris/cluster/controller/writes/jobs.py

from iris.cluster.controller.views import writes_to
from iris.cluster.controller.schema import JOBS, JOB_CONFIG, USER_BUDGETS

@writes_to(JOBS, JOB_CONFIG, USER_BUDGETS)
def insert_job(tx, job, config, budget):
    tx.execute("INSERT INTO jobs(...) VALUES (...)", (...))
    tx.execute("INSERT INTO job_config(...) VALUES (...)", (...))
    tx.execute("INSERT INTO user_budgets(...) VALUES (...)", (...))

@writes_to(JOBS)
def update_state_if_not_terminal(tx, job_id, new_state, *, now_ms):
    tx.execute(
        "UPDATE jobs SET state = ? WHERE job_id = ? AND state NOT IN (?,?,?)",
        (new_state, job_id.to_wire(), *TERMINAL_STATES),
    )

@writes_to(JOBS)
def bulk_update_state(tx, updates):
    tx.executemany(
        "UPDATE jobs SET state = ? WHERE job_id = ?",
        [(u.state, u.job_id.to_wire()) for u in updates],
    )
```

`@writes_to(*tables)` decorator records the table set on the function as `fn.writes_to` and registers the function in a module-level list for the startup check. It does not wrap or alter call semantics — pure metadata.

Files: `writes/jobs.py`, `writes/tasks.py`, `writes/task_attempts.py`, `writes/workers.py`, `writes/reservations.py`. Each contains the write methods previously hosted on the corresponding Store class.

---

## 6. Migration Plan — Single PR, Stacked Commits

The migration is one PR. Each commit is independently testable; tests pass after every commit. The PR is not merged until a real controller has been smoke-tested with the entire stack applied.

### 6.1 Commit sequence

| # | Commit | Touches | Net LOC | Independently testable? |
|---|---|---|---|---|
| 1 | `[iris] scaffold Query / CachedQuery / Projection` | `views.py` (new), `writes/__init__.py` (new), `__init__.py` exports | +400 | yes — added but unused; tests for the new classes themselves |
| 2 | `[iris] port jobs.reservation read to CachedQuery` | `schema.py`, `stores.py:_jobs_with_reservations`, call sites | ±50 | yes — single hot path; benchmark must match |
| 3 | `[iris] port jobs.detail and jobs.config to Query` | `schema.py`, `stores.py:JobStore.get_detail/get_config/...` | ±200 | yes — broad read coverage on jobs |
| 4 | `[iris] port tasks and task_attempts read paths to Query` | `schema.py`, `stores.py`, call sites | ±400 | yes |
| 5 | `[iris] migrate EndpointStore → EndpointsProjection` | new `projections/endpoints.py`; delete `EndpointStore`; call sites | ±300 | **yes — this is the highest-risk commit; see §6.3** |
| 6 | `[iris] migrate _attr_cache → WorkerAttrsProjection` | new `projections/worker_attrs.py`; delete `_attr_cache` from `db.py`; call sites | ±150 | yes |
| 7 | `[iris] move JobStore writes → writes/jobs.py` | new file; delete write methods from `JobStore` | ±300 | yes |
| 8 | `[iris] move Task / TaskAttempt / Worker / Reservation writes → writes/*.py` | new files; delete write methods | ±600 | yes |
| 9 | `[iris] add startup-time owned-table check` | `views.py`, `ControllerDB.__init__` | +60 | yes — invariant test that intentional misuse fails loudly |
| 10 | `[iris] delete ProtoCache class and column-level cached flag` | `schema.py` | −80 | yes — must produce zero behavioral change |
| 11 | `[iris] delete now-empty Store classes` | `stores.py` mostly deleted | −1000 | yes |
| 12 | `[iris] rename today's Projection → Query in docstrings/comments` | doc-only sweep | ±50 | yes — pure cosmetic |

**Net change:** +1860 / −1380 added, with the file count growing slightly (`views.py`, `projections/`, `writes/`) and `stores.py` shrinking from 2100 lines to ~200 (or being deleted outright if it ends up empty).

### 6.2 Per-commit checklist

For every commit:

1. `./infra/pre-commit.py --all-files --fix` passes.
2. `uv run pyrefly` passes.
3. `uv run pytest lib/iris/tests/` passes.
4. For commits #2, #5, #6: run the targeted benchmark from §7.3 and confirm no regression beyond noise.
5. Commit is self-contained: revertable in isolation; does not break the build.

### 6.3 Commit 5 (`EndpointsProjection`) is the linchpin

This commit proves the `Projection` abstraction preserves the `EndpointStore`'s subtle atomicity property: the post-commit hook fires only after the `IMMEDIATE` transaction's commit lands, under the write lock, and only if the commit succeeds. If `Projection` cannot carry this guarantee, the whole design fails.

**Acceptance tests for commit 5:**

1. **Atomic write-through.** Insert an endpoint; before `on_commit` fires, the in-memory dict must not contain it. After commit returns, it must. (This already exists in `test_endpoint_store.py`; the test should pass unchanged with the call sites renamed.)
2. **Rollback safety.** Insert an endpoint, then raise inside the transaction. The in-memory dict must not contain the entry.
3. **Restore correctness.** Snapshot the DB, modify endpoints, then `replace_from` the snapshot. The in-memory dict must reflect the snapshot state, not the post-modification state. (`rehydrate` must run.)
4. **Concurrency.** Two threads writing endpoints, one thread reading via `by_id` repeatedly. No `KeyError`, no stale-read of an in-flight write, no deadlock under the write lock.
5. **Listing.** `ListEndpoints` RPC returns the same result before and after the commit.

### 6.4 End-to-end smoke before merge

Before clicking merge:

1. Spin up a real controller against a real workload via `scripts/iris/dev_tpu.py` (the `dev-tpu` skill).
2. Submit a small batch of jobs, watch them transition: submitted → scheduled → running → terminal.
3. Restart the controller from a backup. Verify all projections rehydrate; verify dashboard renders.
4. Run a 10-minute soak: confirm scheduler tick latency stays in the established envelope (resource_usage_by_worker: <10 ms p95; reconcile_rows_for_workers: <10 ms p95).
5. Compare wall-clock CPU profile of the controller (`agent-profiling` skill) before and after. The 5 mechanisms collapsing into 3 should at minimum not regress; we expect a small win from removed indirection.

If any of those fail, the PR does not merge.

### 6.5 Rollback strategy

Each commit is revertable in isolation, so partial rollback works. If commit 11 (the big delete) lands and a regression surfaces post-merge:

- The data layer's behavior is unchanged from pre-refactor (the Projections are functional equivalents of the Stores). Any regression is most likely in a call site that was rewritten.
- Bisect via `git bisect` to find the offending commit.
- For non-deletion commits, vanilla `git revert` works.
- For the deletion commits (#10, #11), revert means restoring the deleted classes; we will keep them available in git history but they should never come back into the working tree.

---

## 7. Testing Strategy

### 7.1 Unit tests

Each new class gets its own test file, mirroring today's `test_endpoint_store.py`:

- `tests/cluster/controller/test_query.py` — covers `Query` and `CachedQuery` behavior including LRU eviction and the bytes-keyed identity property.
- `tests/cluster/controller/test_projection.py` — covers `Projection.rehydrate`, `on_commit` atomicity, rollback safety, concurrent reads, and the startup-time owned-table check.
- `tests/cluster/controller/projections/test_endpoints_projection.py` — the existing `test_endpoint_store.py` renamed; same assertions, new class.
- `tests/cluster/controller/projections/test_worker_attrs_projection.py` — assertions previously in `test_db.py` for `_attr_cache`, moved.
- `tests/cluster/controller/writes/test_jobs.py` — new tests for write functions, distinguishing them from the old `JobStore` tests (which become projection tests where they touched the cache, or are deleted if they were testing the class structure itself).

### 7.2 Integration tests

Existing integration tests in `tests/cluster/controller/test_controller.py`, `test_scheduler.py`, `test_transitions.py` must pass unchanged at every commit. These tests use a real `ControllerDB` and exercise the full call graph; if they pass, the refactor is behaviorally transparent.

### 7.3 Performance benchmarks

Added in commit #1 alongside the scaffolding, as `tests/cluster/controller/test_perf_baselines.py`:

| Benchmark | Today's number | Acceptance window |
|---|---|---|
| `resource_usage_by_worker` (24k jobs, 1k live attempts) | 6.5 ms | ≤ 8 ms |
| `reconcile_rows_for_workers` (200 worker ids) | 6.3 ms | ≤ 8 ms |
| `_jobs_with_reservations` (200 reservations) | 0.019 ms | ≤ 0.025 ms |
| `EndpointsProjection.by_id` (single lookup) | dict access | < 1 µs |
| `JOB_DETAIL_QUERY.one` (job_id PK lookup) | 0.05 ms | ≤ 0.07 ms |

The benchmarks gate commit #2, #5, and #6. Out-of-window results block the commit.

### 7.4 Invariant tests

`test_projection.py::test_owned_table_externally_written_raises` — at import time, instantiate a Projection that owns `JOBS`, then register a write function that declares `writes_to=(JOBS,)`. Assert that `assert_owned_tables_not_externally_written()` raises.

`test_projection.py::test_concurrent_write_through_atomicity` — fire 100 writes and 1000 reads concurrently for 5 seconds against an `EndpointsProjection`. Assert: no observed `KeyError`, no read returns a row whose corresponding write has not yet committed, no read fails to return a row whose write has committed.

### 7.5 Restore tests

`test_projection.py::test_rehydrate_after_replace_from` — populate state, snapshot, modify, then `replace_from(snapshot)`. Assert every Projection's in-memory state matches the snapshot's SQL state, not the post-modification state.

### 7.6 Where tests don't change

Per audit Q3: there are zero `mock.*Store` / `patch.*Store` matches in `lib/iris/tests/`. No test mocks a Store class as an injection point, so the renaming is invisible to tests at the seam level. Tests that import `EndpointStore` directly (e.g. `test_endpoint_store.py:77`) get the class name swap in their import line; their assertions are unchanged.

---

## 8. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `Projection.on_commit` semantics differ subtly from `EndpointStore.on_commit` | Low | High | Commit #5 explicitly preserves the code path; acceptance tests in §6.3 cover the four atomicity cases. |
| `CachedQuery` blob-bytes cache key produces incorrect sharing across rows | Low | High | The cache is content-addressed on the *blob column value*, not on the row identity, exactly matching `ProtoCache` semantics. The blob is by definition the decoder input. Test that two rows with the same blob get the same decoded object. |
| Performance regression on the scheduler hot path | Medium | High | Benchmarks in §7.3 gate the relevant commits. Worst case: `CachedQuery`'s lock acquisition is observable. Mitigation: same lock+dict shape `ProtoCache` already has — should be perf-equivalent. |
| Startup-time owned-table check has false positives | Medium | Low | Conservative declaration: a function that *reads* an owned table without writing to it is allowed. Only `INSERT/UPDATE/DELETE` against the owned table is flagged. |
| Hidden caller that bypasses the public API (e.g., manual SQL in `transitions.py`) writes to an owned table | Medium | Medium | Audit `transitions.py` for raw `tx.execute` calls against `endpoints` or `worker_attributes`; move them onto the corresponding Projection in commit #5/#6. Grep is straightforward. |
| `WorkerAttrsProjection.rehydrate` is more expensive than today's lazy `_populate_attr_cache` | Low | Low | Eager rehydrate runs once at startup; the table is small (one row per (worker, attr_name) pair). If profiling shows a problem, make `rehydrate` lazy with a `_populated: bool` flag — small change. |
| PR size deters review | High | Medium | Twelve commits, each small and reviewable in isolation. Reviewers can focus on commit boundaries. The "delete" commits at the end (#10, #11) are trivial. |
| Future contributor adds a non-Projection write to a Projection-owned table | Low | Medium (silent stale cache) | Startup-time check catches this at import. If the contributor bypasses `@writes_to` entirely (just does `tx.execute` from somewhere unregistered), the check misses it — but writes through arbitrary call sites are already against convention. Document this in `AGENTS.md` under iris-specific guidance. |
| Style guide drift if a fourth caching strategy is needed later (e.g., TTL-based) | Low | Low | The `ReadSpec` base is the extension point — add `TTLQuery(ReadSpec, Query)` as a new sibling rather than adding a flag to an existing class. |

---

## 9. Codebase Audit (verbatim findings)

These are the audit results from the validation pass referenced in §2.2 and §4.5.

### 9.1 Cross-entity write counts

| Method | Tables written |
|---|---|
| `EndpointStore.add` / `remove` | 1 (`endpoints`) |
| `JobStore.insert` / `insert_config` / `insert_workdir_files` | 1 each |
| `TaskStore.insert` / `mark_assigned` / `apply_state_update` / `mark_terminal` / `bulk_kill_non_terminal` / `update_container_id` | 1 (`tasks`) |
| `TaskAttemptStore.insert` / `mark_finished` / `apply_attempt_state` / `apply_update` | 1 (`task_attempts`) |
| `TaskStore.assign` | 2 (`task_attempts`, `tasks`) |
| `TaskAttemptStore.bulk_apply_attempt_state` | 1 write (`task_attempts`); subquery reads `tasks` |
| `WorkerStore.remove` | **3** (`task_attempts`, `tasks`, `workers`) — single largest cross-table write |
| `ReservationStore.replace_claims` | 1 (`reservation_claims`) — multi-statement |
| `ReservationStore.next_submission_ms` | 1 (`meta`) — conditional INSERT-or-UPDATE |

Conclusion: zero writes touch more than 3 tables. Module-level functions handle this scale trivially.

### 9.2 Transaction cursor capabilities

`TransactionCursor` (`db.py:240–288`) wraps `sqlite3.Cursor` and adds only:
- `on_commit(fn)` — registers a callable invoked after `IMMEDIATE` commit, under the write lock.

No table tracking. No event emission. Adding table tracking would require either SQL parsing or a `tx.execute(table, sql, params)` wrapper — both more invasive than the explicit `@writes_to`/`Projection.sources` approach.

### 9.3 Test mocking patterns

Zero matches for `mock.*Store` / `patch.*Store` / `MagicMock.*Store` across `lib/iris/tests/`.

`test_endpoint_store.py:77` instantiates `EndpointStore(state._db)` directly for unit testing — not a mock, a real instance with a real DB. After refactor, this line becomes `EndpointsProjection(db)`.

### 9.4 `_attr_cache` access points

- **Reads:** `ControllerDB.get_worker_attributes` (`db.py:355`), called from scheduler logic via `controller.py` and `transitions.py`.
- **Writes:** `set_worker_attributes` called via `cur.on_commit(...)` at `transitions.py:1296` (registration), `remove_worker_from_attr_cache` called via `cur.on_commit(...)` at `transitions.py:1911` and `2178` (deletion).
- **Semantics:** true write-through, lazy populate, no invalidation path outside the two declared paths.

After refactor: this becomes `WorkerAttrsProjection` with `sources=(WORKER_ATTRIBUTES,)`. The two `on_commit` registrations move from `transitions.py` to method calls on the projection (`projection.set(tx, ...)`, `projection.remove(tx, ...)`).

---

## 10. Industry Survey (verbatim findings)

These are the survey results from §2.3, kept here for completeness.

### 10.1 Repository pattern (Fowler, Evans)

[Martin Fowler — Repository (PoEAA)](https://martinfowler.com/eaaCatalog/repository.html): "mediates between the domain and data mapping layers using a collection-like interface for accessing domain objects." Reads *and* writes on one class. Conventional. Not what we want — we want explicit caching variants by class.

### 10.2 CQRS (Greg Young, Bertrand Meyer's CQS generalized)

[Martin Fowler — CQRS](https://martinfowler.com/bliki/CQRS.html): "you can use a different model to update information than the model you use to read information." Spectrum from "same DB, separate classes" (soft) to "separate DBs + event sourcing" (extreme). The [Cosmic Python CQRS chapter](https://www.cosmicpython.com/book/chapter_12_cqrs.html) recommends soft CQRS as a starting point. Fowler explicitly: "for most systems CQRS adds risky complexity." We are within "soft CQRS" — same DB, separate classes per read variant.

### 10.3 Inline projections (Marten)

[Marten — Projections](https://martendb.io/events/projections/): `Inline` projection commits the read-model update in the same transaction as the source event write. This is **exactly** our `Projection` class's semantics, modulo that we are not event-sourced (we write directly to mutable rows rather than appending events).

### 10.4 Mainstream framework conventions

| Framework | Read/write split? |
|---|---|
| Rails ActiveRecord | Same `Model` class; community uses Query Objects ([Selleo](https://selleo.com/blog/essential-rubyonrails-patterns-part-2-query-objects)) for complex reads. |
| Django | Same `Model.objects` Manager. No built-in Repository. ([Luke Plant](https://lukeplant.me.uk/blog/posts/evolution-of-a-django-repository-pattern/) tried, retreated.) |
| SQLAlchemy | `Session` mixes reads and writes (Unit of Work). Repository layered on top is a user choice. |
| Spring Data JPA | `@Repository` extends `JpaRepository<T, ID>` exposing both reads and writes; mutating methods marked `@Modifying`. |

### 10.5 Python orchestrator production codebases

| System | Has Stores layer? | Pattern |
|---|---|---|
| Apache Airflow | No | SQLAlchemy ORM + `@provide_session` classmethods on model classes |
| Prefect | No | `prefect.server.models.*` modules of free functions taking a session |
| Dagster | Yes (`RunStorage`, `EventLogStorage`) | Pluggable storage interfaces, but interfaces mix reads and writes |
| Sentry | No | Django Managers + serializers framework for API shapes |

Pattern: **module-level functions taking a session for writes; class-based interfaces only when storage is pluggable across backends.** We do not have multiple backends; module-level functions are the right move.

### 10.6 Verdict from survey

The proposed design is "asymmetric CQRS with inline projections." Each piece is conventional:

- `Query` / `CachedQuery` — Query Object pattern (Rails community), or "live query model" in CQRS terms.
- `Projection` — inline projection (Marten), or write-through cache (generic).
- `writes/` modules — the Python orchestrator standard.

The one move that isn't in any textbook is the explicit class split (vs a `View(cache=enum)`). That split is endorsed by our own style guide and Cosmic Python.

### 10.7 Documented pitfalls

- [Materialize — Cache invalidation](https://materialize.com/blog/redis-cache-invalidation/): write-through caches go silently stale when an external writer touches the underlying table. → Mitigated by §4.5 startup-time check.
- [Cosmic Python — CQRS chapter](https://www.cosmicpython.com/book/chapter_12_cqrs.html): policy enums on one class invite mode-confused methods. → Mitigated by separate classes.
- Read/write convention drift: two write conventions (free functions vs. Projection methods) requires a written rule for which to use. → Documented in §11.1.

---

## 11. Open Questions / Decisions Pending

### 11.1 Where does "convention for which write style to use" live?

**Proposal:** add a short section to `lib/iris/AGENTS.md`:

> ### Iris data layer conventions
> - Reads: pick one of `Query`, `CachedQuery`, or `Projection` based on caching needs. Never write a read that bypasses these classes.
> - Writes: if the target table is owned by a `Projection`, the write is a method on that Projection. Otherwise, the write is a module-level function in `writes/<entity>.py` decorated with `@writes_to(...)`.
> - To answer "is this table owned?" — check `PROJECTIONS` in `views.py`.

### 11.2 Should `Query` and `CachedQuery` accept dynamic predicates?

Today's `Projection` only supplies the `SELECT` clause; callers concatenate `FROM` / `WHERE`. Should the new `Query` follow the same pattern (caller writes raw SQL after `select_clause`) or take a typed `where=Predicate`?

**Recommendation:** match today's convention (caller supplies SQL). The benefit of a typed predicate API is small and the cost is a new DSL. Defer.

### 11.3 Single shared `ProtoCache` instance or per-CachedQuery cache?

`ProtoCache` today is a singleton; all cached columns share its 8192-entry budget. `CachedQuery` could either (a) share one global cache (matches today) or (b) own a per-class cache (better isolation, harder to size globally).

**Recommendation:** start with (a) — preserves current behavior exactly. If we later see one Query crowding others out, switch to (b) with a per-class `_MAX_SIZE`.

### 11.4 Should `EndpointsProjection` reuse `EndpointStore`'s outcome enums?

`EndpointStore.add` returns `AddEndpointOutcome` (an enum). Should `EndpointsProjection.add` return the same enum, or simplify to raising on failure?

**Recommendation:** preserve the enum return; this is API-compatibility with callers, not a design choice worth debating in this refactor.

### 11.5 Naming: `views.py` vs `query.py` vs `read.py`?

The file holding `Query`, `CachedQuery`, `Projection` could be named various things. `views.py` clashes with HTTP-handler "views" in other frameworks; `read.py` is too generic; `query.py` undersells `Projection`.

**Recommendation:** `lib/iris/src/iris/cluster/controller/reads.py` — symmetric with `writes/` directory. Reviewers welcome to bikeshed in PR review.

### 11.6 Should the `@writes_to` decorator be enforcement, not just documentation?

We could have `@writes_to` actually instrument `tx` to assert at runtime that only the declared tables were written.

**Recommendation:** no — runtime cost on every write, false positives on metadata tables. The startup-time check is sufficient.

---

## 12. Alternatives Considered

| Alternative | Why rejected |
|---|---|
| Full SQLAlchemy ORM | Cannot map frozen dataclasses ([SQLAlchemy Discussion #9192](https://github.com/sqlalchemy/sqlalchemy/discussions/9192)). Loses our explicit query control. |
| SQLAlchemy Core only | Provides compiled-statement cache but doesn't address caching layer; the proliferation problem stays. |
| `sqlc-gen-better-python` codegen | Immature; doesn't address caching layer. Revisit in 12 months. |
| Postgres migration for real materialized views | Largest possible change. SQLite-on-one-machine is our deliberate design choice. `TRIGGER_MAINTAINED` views in SQLite ([madflex](https://madflex.de/SQLite-triggers-as-replacement-for-a-materialized-view/)) approximate what we'd want. |
| `View(cache=enum)` single class | Cosmic Python and AGENTS.md both say "separate classes over flags for variant behavior." Cache=NONE and cache=WRITE_THROUGH are *behaviorally different objects*. |
| `django-cacheops`-style automatic invalidation | Requires a global write chokepoint we don't have. Explicit declaration is lower magic. |
| Keep Stores, just add caching policies | Solves nothing — same five-mechanism problem with extra ceremony. |
| Multi-PR sequenced over weeks | Each intermediate state would be inconsistent (some entities ported, some not). One PR with reviewable commits is cleaner. |

---

## 13. References

### Codebase audit
- `lib/iris/src/iris/cluster/controller/schema.py:34–66` — `ProtoCache` definition
- `lib/iris/src/iris/cluster/controller/schema.py:316–456` — `Projection` (today's column-subset)
- `lib/iris/src/iris/cluster/controller/stores.py:95–337` — `EndpointStore`
- `lib/iris/src/iris/cluster/controller/db.py:240–288` — `TransactionCursor`
- `lib/iris/src/iris/cluster/controller/db.py:329–379` — `_attr_cache`
- `lib/iris/src/iris/cluster/controller/migrations/0045_*.py` — partial-index materialized-view pattern
- `lib/iris/tests/cluster/controller/test_endpoint_store.py:77` — sole direct Store instantiation in tests
- Prior design: `.agents/projects/20260310_iris_sql_canonical.md`
- Prior design: `.agents/projects/iris-sql-store.md`

### Industry survey
- [Martin Fowler — Repository (PoEAA)](https://martinfowler.com/eaaCatalog/repository.html)
- [Martin Fowler — CQRS](https://martinfowler.com/bliki/CQRS.html)
- [Microsoft — CQRS pattern](https://learn.microsoft.com/en-us/azure/architecture/patterns/cqrs)
- [Microsoft — Materialized View pattern](https://learn.microsoft.com/en-us/azure/architecture/patterns/materialized-view)
- [Kurrent — CQRS Pattern](https://www.kurrent.io/cqrs-pattern)
- [Event-Driven.io — CQRS facts and myths](https://event-driven.io/en/cqrs_facts_and_myths_explained/)
- [Event-Driven.io — Projections and Read Models](https://event-driven.io/en/projections_and_read_models_in_event_driven_architecture/)
- [Rinat Abdullin — Event Sourcing Projections](https://abdullin.com/post/event-sourcing-projections/)
- [Marten — Projections](https://martendb.io/events/projections/)
- [Marten — Read-Model Projections](https://martendb.io/tutorials/read-model-projections)
- [Cosmic Python — Repository Pattern](https://www.cosmicpython.com/book/chapter_02_repository.html)
- [Cosmic Python — Unit of Work](https://www.cosmicpython.com/book/chapter_06_uow.html)
- [Cosmic Python — CQRS](https://www.cosmicpython.com/book/chapter_12_cqrs.html)
- [Cosmic Python — Django appendix](https://www.cosmicpython.com/book/appendix_django.html)
- [Luke Plant — Evolution of a Django Repository pattern](https://lukeplant.me.uk/blog/posts/evolution-of-a-django-repository-pattern/)
- [SQLAlchemy 2.0 — Session Basics](https://docs.sqlalchemy.org/en/20/orm/session_basics.html)
- [SQLAlchemy Discussion #9192 — Frozen dataclass mapping](https://github.com/sqlalchemy/sqlalchemy/discussions/9192)
- [Spring Data JPA — Repository definition](https://docs.spring.io/spring-data/jpa/reference/repositories/definition.html)
- [Rails Guides — Active Record Querying](https://guides.rubyonrails.org/active_record_querying.html)
- [Selleo — Rails Query Objects](https://selleo.com/blog/essential-rubyonrails-patterns-part-2-query-objects)
- [Apache Airflow — Database ERD](https://airflow.apache.org/docs/apache-airflow/stable/database-erd-ref.html)
- [Astronomer — Airflow Metadata Database](https://www.astronomer.io/docs/learn/airflow-database)
- [Dagster — Storage and Persistence](https://deepwiki.com/dagster-io/dagster/5-storage-and-persistence)
- [Sentry — server models](https://github.com/getsentry/sentry/tree/master/src/sentry/models)
- [Materialize — Solving cache invalidation](https://materialize.com/blog/redis-cache-invalidation/)
- [Lu — Caching Partially Materialized Views Consistently](https://uvdn7.github.io/caching-partially-materialized-views-consistently/)
- [SQLite triggers as materialized views — madflex](https://madflex.de/SQLite-triggers-as-replacement-for-a-materialized-view/)
- [elmah.io — The Repository Pattern, simple yet misunderstood](https://blog.elmah.io/the-repository-pattern-is-simple-yet-misunderstood/)
- [Design Gurus — Cache Invalidation Strategies](https://www.designgurus.io/blog/cache-invalidation-strategies)
