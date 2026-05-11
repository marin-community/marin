# Addendum: Fit Analysis of Refactor Against 10 Real Call Sites

**Date:** 2026-05-11
**Author:** russell.power@gmail.com (with fit-analysis agent)
**Status:** Companion to `20260511_iris_store_view_refactor.md`

## Summary

The proposal in §5 holds up for the *simple* read shapes — PK lookups, cached
detail reads, bulk reads against a single table, endpoint reads — which are
roughly 60% of the call sites by count. The API breaks down at three places:
**(a)** aggregate / GROUP BY queries where the result shape is not a row of
the entity's table (dashboard summaries, `state_counts_for_job`,
`_live_user_stats`); **(b)** dashboard list-with-filter-and-pagination
queries that need dynamic ORDER BY / LIMIT / OFFSET and a parallel COUNT(*);
and **(c)** the §4.6 "Projection + Query mixed in one snapshot" worked
example, which on real call sites either (i) doesn't co-occur (Projection
reads stand alone) or (ii) when it does co-occur (worker registration tick),
demands snapshot isolation that today's lazy `_attr_cache` already provides
incidentally. The `tx`-symmetry argument is mostly defensible but is too
weak to justify forcing a second `.query` escape hatch on every Projection.

Concrete recommendations:

1. Add a `Query.scalar(tx, sql, params)` / `Query.scalars(tx, sql, params)` for COUNT and ad-hoc aggregate shapes that don't fit `row_cls`.
2. Replace the `where=` / `params=` interface in `Query.one`/`Query.many` with a more general `Query.fetchall(tx, *, where=None, order_by=None, limit=None, offset=None, params=())` builder, OR keep `where=` raw but also expose a `Query.raw(tx, sql_suffix, params)` escape hatch that appends to the canonical `SELECT … FROM … `.
3. Introduce `AggregateQuery[Row]` (sibling of `Query`) that takes a custom `row_cls` and an arbitrary SQL body, so `state_counts_for_job`, `_task_summaries_for_jobs`, `_live_user_stats`, and dashboard `list_jobs` aren't pushed into raw `tx.execute`.
4. Recursive CTEs (`get_priority_bands`, `list_descendants`, `has_unfinished_worker_attempts`) fit as plain `Query` *if* the API lets the caller supply a full custom body. The `where=` strawman is wrong for these — recommend an explicit `Query.with_sql(tx, sql, params)` method on `Query`.
5. Promote the `WorkerAttrsProjection` from a "Projection that takes `tx` for symmetry" to a `tx`-free read object. The §4.6 symmetry argument is unconvincing on the one real co-mixed call site (`healthy_active_workers_with_attributes` at `db.py:908`), which today *intentionally* uses the latest committed attrs alongside a snapshot worker read, and is correct because of it.

The rest of this document walks through ten call sites, with locations,
excerpts, proposed rewrites under the new design, and concerns.

---

## Call Site 1: `JobStore.get_detail` — simple PK lookup

- **Location:** `lib/iris/src/iris/cluster/controller/stores.py:700-707`
- **Excerpt:**

```python
def get_detail(self, tx: Tx, job_id: JobName) -> JobDetailRow | None:
    row = tx.fetchone(
        f"SELECT {JOB_DETAIL_PROJECTION.select_clause()} "
        f"FROM jobs j {JOB_CONFIG_JOIN} WHERE j.job_id = ?",
        (job_id.to_wire(),),
    )
    if row is None:
        return None
    return JOB_DETAIL_PROJECTION.decode_one([row])
```

- **Proposed shape under the refactor:**

```python
# views.py
JOB_DETAIL_QUERY: Query[JobDetailRow] = Query(ReadSpec(
    name="jobs.detail",
    sources=(JOBS, JOB_CONFIG),
    select_columns=_job_detail_cols,
    row_cls=JobDetailRow,
    from_clause=f"jobs j {JOB_CONFIG_JOIN}",
))

# call site
detail = JOB_DETAIL_QUERY.one(tx, where="j.job_id = ?", params=(job_id.to_wire(),))
```

- **Concerns:** Clean fit. `Query.one(tx, where=..., params=...)` covers this perfectly.
- **Recommended changes:** None.

---

## Call Site 2: `JOB_RESERVATION_PROJECTION` per-tick read — cached-blob read

- **Location:** `lib/iris/src/iris/cluster/controller/controller.py:418-433`
- **Excerpt:**

```python
def _jobs_with_reservations(queries: ControllerDB, states: tuple[int, ...]) -> list[JobReservationRow]:
    placeholders = ",".join("?" for _ in states)
    with queries.read_snapshot() as snapshot:
        rows = snapshot._fetchall(
            f"SELECT {JOB_RESERVATION_PROJECTION.select_clause()} "
            f"FROM jobs j {JOB_CONFIG_JOIN} "
            f"WHERE j.state IN ({placeholders}) AND j.has_reservation = 1",
            list(states),
        )
    return JOB_RESERVATION_PROJECTION.decode(rows)
```

The `reservation_json` column today is decoded through `proto_cache` because
`Column("reservation_json", …, cached=True)` — see `schema.py:664` — so
identical reservation blobs across reservation-holder jobs share their
decoded proto.

- **Proposed shape under the refactor:**

```python
JOB_RESERVATION_QUERY = CachedQuery(ReadSpec(
    name="jobs.reservation",
    sources=(JOBS, JOB_CONFIG),
    select_columns=_job_reservation_cols,
    row_cls=JobReservationRow,
    from_clause=f"jobs j {JOB_CONFIG_JOIN}",
))

def _jobs_with_reservations(db, states):
    placeholders = ",".join("?" for _ in states)
    with db.read_snapshot() as snap:
        return JOB_RESERVATION_QUERY.many(
            snap,
            where=f"j.state IN ({placeholders}) AND j.has_reservation = 1",
            params=tuple(states),
        )
```

- **Concerns:**
  1. The `CachedQuery` API as sketched in §5.2 caches the *whole decoded Row* keyed on a "blob bytes" cache key — but today's `ProtoCache` caches at the *column* granularity (each blob field decodes through the cache independently). For `JobReservationRow` that distinction is moot (one cached column). For a richer row (e.g. a hypothetical `JobConfigRow` with both `entrypoint_proto` and `reservation_proto`), the row-level key forces an all-or-nothing cache miss when *any* blob differs. That is a regression vs today.
  2. The §5.2 `CachedQuery._cache_key(row)` is called once with the full `sqlite3.Row`, but only some columns are blobs. Specification should be explicit: cache key is the tuple of bytes-typed (or `cached=True`-flagged) column values for that row, OR cache at the per-column decode level (matching ProtoCache).

- **Recommended changes:**
  - Spec §5.2 should clarify cache-key granularity: per-column (matching `ProtoCache`) is the right answer. Per-row is a behavioral regression with no upside.
  - Equivalent: keep `cached=True` on the `Column` and have `CachedQuery` simply use the column-level decoder wrappers that today's `Projection` already wires up at `schema.py:370-381`. That is what `CachedQuery` should compile to internally.

---

## Call Site 3: `TaskAttemptStore.bulk_get_for_updates` — bulk read

- **Location:** `lib/iris/src/iris/cluster/controller/stores.py:1580-1613`
- **Excerpt:**

```python
def bulk_get_for_updates(self, tx, keys):
    result = {}
    if not keys:
        return result
    unique = list({k: None for k in keys}.keys())
    chunk_size = 450
    for chunk_start in range(0, len(unique), chunk_size):
        chunk = unique[chunk_start : chunk_start + chunk_size]
        values_clause = ",".join("(?, ?)" for _ in chunk)
        params = []
        for task_id, attempt_id in chunk:
            params.append(task_id.to_wire())
            params.append(attempt_id)
        rows = tx.fetchall(
            f"SELECT {ATTEMPT_PROJECTION.select_clause()} FROM task_attempts ta "
            f"WHERE (ta.task_id, ta.attempt_id) IN (VALUES {values_clause})",
            tuple(params),
        )
        for attempt in ATTEMPT_PROJECTION.decode(rows):
            result[(attempt.task_id, attempt.attempt_id)] = attempt
    return result
```

- **Proposed shape under the refactor:**

```python
ATTEMPT_QUERY = Query(ReadSpec(name="attempts.bulk", sources=(TASK_ATTEMPTS,),
                               select_columns=_attempt_cols, row_cls=AttemptRow,
                               from_clause="task_attempts ta"))

def bulk_get_attempts(tx, keys):
    result = {}
    if not keys:
        return result
    unique = list({k: None for k in keys}.keys())
    for chunk_start in range(0, len(unique), 450):
        chunk = unique[chunk_start:chunk_start + 450]
        values_clause = ",".join("(?, ?)" for _ in chunk)
        params = tuple(p for tid, aid in chunk for p in (tid.to_wire(), aid))
        attempts = ATTEMPT_QUERY.many(
            tx,
            where=f"(ta.task_id, ta.attempt_id) IN (VALUES {values_clause})",
            params=params,
        )
        for a in attempts:
            result[(a.task_id, a.attempt_id)] = a
    return result
```

- **Concerns:**
  1. The chunk loop has to compose `where=` strings dynamically from `chunk` length — fine under §5.1's "caller writes raw `where`" decision (§11.2), but it forces a string-format step into every bulk caller. Almost every bulk read in `stores.py` repeats this pattern.
  2. The function moves to `writes/task_attempts.py`? No — it's a read. So where does it live now that `TaskAttemptStore` is gone? §5.4 talks about `writes/<entity>.py`; there is no parallel `reads/<entity>.py` defined. The proposal implicitly relies on the call site embedding the chunking loop inline against `ATTEMPT_QUERY`. With ~7 such bulk-with-chunking helpers across stores.py, that's ~150 LOC of inlined loops in the migration.

- **Recommended changes:**
  - Add a `reads/<entity>.py` convention (mirror of `writes/`) for read helpers that aren't a single `Query.one`/`many` call. Keep the chunking loop as a free function in `reads/task_attempts.py` taking `(tx, keys)`.
  - Or, add a `Query.many_in(tx, *, column_tuple: tuple[str, ...], values: Sequence[tuple])` helper that handles the chunking and `(...) IN (VALUES …)` boilerplate inside `Query`. This is the right level: SQLite's parameter limit is a property of the DB, not of every caller.

---

## Call Site 4: `EndpointStore.query` — endpoint store read (today bypasses SQL)

- **Location:** `lib/iris/src/iris/cluster/controller/stores.py:170-199`, called from `service.py:1745-1750`.
- **Excerpt (call site):**

```python
endpoints = self._store.endpoints.query(
    EndpointQuery(
        exact_name=prefix if request.exact else None,
        name_prefix=None if request.exact else prefix,
    ),
)
```

The internal `query(...)` body (stores.py:170) is the hand-coded multi-index
selector — it picks the most selective dict (`_by_id`, `_by_task`, or
`_by_name`) and Python-filters the rest. Reads never touch SQL.

- **Proposed shape under the refactor:**

```python
# projections/endpoints.py
class EndpointsProjection(Projection[EndpointRow]):
    ...
    def query(self, tx, q: EndpointQuery) -> list[EndpointRow]:
        del tx  # symmetry only
        with self._lock:
            # same dict-narrowing + Python filter as today
            ...
```

Caller becomes `self._store.endpoints.query(tx, EndpointQuery(...))`.

- **Concerns:**
  1. The §4.6 symmetry argument says "every read takes `tx` so a reader scanning code sees `xxx.one(tx, ...)`." Here the right argument signature is `query(tx, q)` — a *second* positional parameter `q`. The shape is `xxx.method(tx, …)`, not the uniform `xxx.one(tx, where=…, params=…)`. The "uniform shape" argument from §4.6 is therefore aspirational — Projection methods diverge in shape from `Query.one/many` because the dict-narrowing strategy is method-specific (`by_id`, `by_name`, `query`, `all`).
  2. The §5.3 sketch claims `by_id(tx, endpoint_id)`, `by_name(tx, name)`, `all(tx)`, `query.one(tx, where=…, params=…)` co-exist. That's four distinct call-site shapes on one class. The "API symmetry" benefit is largely cosmetic — the burden of remembering which method to call dominates the burden of remembering whether the first argument is `tx`.

- **Recommended changes:**
  - Drop `tx` from `Projection` reads entirely. The §4.6 argument is "future-proofing for a hypothetical `VersionedProjection`" — but YAGNI: the moment we need versioned projections, we add a new class with a different name and update the dozen call sites. The cost of forcing every call site to thread an unused `tx` through forever exceeds the cost of a future refactor.
  - Keep the `.query` (snapshot-isolated SQL fallback) escape hatch but make it explicitly `endpoints.snapshot_query.one(tx, ...)`. Naming the SQL escape hatch `snapshot_query` makes its semantics legible.

---

## Call Site 5: `_attr_cache` worker attribute read

- **Location:** `lib/iris/src/iris/cluster/controller/db.py:355-364` (impl), `db.py:929` (call site).
- **Excerpt (call site, inside `healthy_active_workers_with_attributes`):**

```python
with db.read_snapshot() as q:
    rows = WORKER_ROW_PROJECTION.decode(q.fetchall(
        f"SELECT {_worker_row_select()} FROM workers w WHERE w.worker_id IN ({placeholders})",
        tuple(str(wid) for wid in healthy_active),
    ))
    if not rows:
        return []
attrs_by_worker = db.get_worker_attributes()    # ← in-memory cache; NOT in snapshot
```

- **Proposed shape under the refactor:**

```python
def healthy_active_workers_with_attributes(db, projections, health):
    liveness = health.all()
    healthy_active = [wid for wid, l in liveness.items() if l.healthy and l.active]
    if not healthy_active:
        return []
    placeholders = ",".join("?" for _ in healthy_active)
    with db.read_snapshot() as snap:
        rows = WORKER_ROW_QUERY.many(
            snap,
            where=f"w.worker_id IN ({placeholders})",
            params=tuple(str(wid) for wid in healthy_active),
        )
    attrs_by_worker = projections.worker_attrs.all(snap)  # snap is closed; tx unused
    ...
```

- **Concerns (this is the §4.6 worked example, applied to the real code):**
  1. **The snapshot is closed when `projections.worker_attrs.all(snap)` is called.** Today's code reads `WORKER_ROW_PROJECTION` *inside* `with db.read_snapshot() as q:` and reads `_attr_cache` *outside* it. The semantics today are: workers come from snapshot at T₁; attrs come from the latest-committed state at T₂ ≥ T₁. Under the refactor, the same pattern still works — except that §4.6 says Projection reads take `tx` for "API symmetry", but here `tx` literally cannot be honored because the snapshot has been released. Passing the released `snap` as a placeholder is dishonest; passing nothing is what we want.
  2. **The mixed-consistency pattern is the *desired* behavior here**, not an accident. A newly-registered worker should be visible *now* (not at the snapshot start) because the scheduler tick's job is to dispatch tasks to it ASAP; an in-flight worker registration that committed at T₁.5 between snapshot open and `get_worker_attributes` should be observed. This is a counter-example to §4.6's worry about Projection-snapshot drift.

- **Recommended changes:**
  - Document this real call site in §4.6 alongside the "scheduler tick mixing endpoint and job state" worked example, explicitly as a case where the **latest-committed Projection read is the correct semantics**, not a hazard.
  - Remove `tx` from the `Projection.all` / `Projection.by_id` signatures. Replace the §4.6 "symmetry argument" with the simpler rule: *Projections serve latest-committed reads, no tx; use `Projection.snapshot_query` if you need a snapshot read*.

---

## Call Site 6: `WorkerStore.remove` — multi-table write

- **Location:** `lib/iris/src/iris/cluster/controller/stores.py:2066-2071`.
- **Excerpt:**

```python
def remove(self, cur: TransactionCursor, worker_id: WorkerId) -> None:
    cur.execute("UPDATE task_attempts SET worker_id = NULL WHERE worker_id = ?", (str(worker_id),))
    cur.execute("UPDATE tasks SET current_worker_id = NULL WHERE current_worker_id = ?", (str(worker_id),))
    cur.execute("DELETE FROM workers WHERE worker_id = ?", (str(worker_id),))
    cur.on_commit(lambda: self._health.forget(worker_id))
```

Touches `task_attempts`, `tasks`, `workers` — the single largest cross-table
write in the system. Also registers a post-commit hook on the in-memory
liveness tracker, which under the refactor moves into either a Projection or
stays put.

- **Proposed shape under the refactor:**

```python
# writes/workers.py
@writes_to(WORKERS, TASKS, TASK_ATTEMPTS)
def remove_worker(tx, worker_id, *, health: WorkerHealthTracker):
    tx.execute("UPDATE task_attempts SET worker_id = NULL WHERE worker_id = ?", (str(worker_id),))
    tx.execute("UPDATE tasks SET current_worker_id = NULL WHERE current_worker_id = ?", (str(worker_id),))
    tx.execute("DELETE FROM workers WHERE worker_id = ?", (str(worker_id),))
    tx.on_commit(lambda: health.forget(worker_id))
```

Plus, if `WorkerAttrsProjection` is introduced (§5.3) it owns `worker_attributes`,
which is **cascade-deleted via foreign key** when the `workers` row is
deleted (`schema.py:888`: `REFERENCES workers(worker_id) ON DELETE CASCADE`).

- **Concerns:**
  1. **Foreign-key cascade silently mutates a Projection-owned table.** The `DELETE FROM workers …` line causes SQLite to cascade-delete from `worker_attributes` — which is `WorkerAttrsProjection.sources`. The Projection's in-memory dict has no knowledge of the cascade. The §4.5 startup-time check (`assert_owned_tables_not_externally_written`) inspects declared `@writes_to(...)` — but this function declares `WORKERS, TASKS, TASK_ATTEMPTS`, not `WORKER_ATTRIBUTES`. The check passes, but the Projection's dict goes stale.
  2. This is the **exact failure mode** §4.5 was supposed to prevent. The proposal text says "a write function legitimately reading from an owned table but not writing it is fine" — but FK cascade is neither reading nor a direct write; it's a third category.
  3. Today's code at `transitions.py:1911` and `2178` handles this by explicitly calling `self._store.workers.remove_from_attr_cache(worker_id)` after the delete. Under the refactor, this becomes a manual `projections.worker_attrs.invalidate(tx, worker_id)` call that *every* place issuing a `DELETE FROM workers` must remember to make. That's a regression vs. today's already-manual pattern, with the added gotcha that `@writes_to` doesn't catch it.

- **Recommended changes:**
  - Add a `Projection.cascades_from(*tables)` declaration alongside `Projection.sources`. The startup check then also asserts: for any `@writes_to(T1, …)` where `T1` cascades into a Projection-owned table, either the write function lives on the cascading Projection, or it explicitly registers a follow-up hook on the projection.
  - Pragma: ban implicit FK cascade into Projection-owned tables. Simpler. The `worker_attributes` cascade is fine to keep at the SQL level but must be matched by an explicit `projections.worker_attrs.on_worker_delete(tx, worker_id)` call in `remove_worker`, registered via `tx.on_commit`. Document this rule in §11.1.
  - The §6.3 acceptance test list should include a "delete worker; assert WorkerAttrsProjection no longer reports stale attributes for that id" test.

---

## Call Site 7: `JobStore.get_priority_bands` — recursive CTE

- **Location:** `lib/iris/src/iris/cluster/controller/stores.py:719-770`.
- **Excerpt (the CTE):**

```python
rows = tx.fetchall(
    f"""
    WITH RECURSIVE chain(input_id, current_id, current_band, parent_id) AS (
        SELECT j.job_id, j.job_id, jc.priority_band, j.parent_job_id
        FROM jobs j JOIN job_config jc ON jc.job_id = j.job_id
        WHERE j.job_id IN ({placeholders})
        UNION ALL
        SELECT chain.input_id, j.job_id, jc.priority_band, j.parent_job_id
        FROM chain
        JOIN jobs j ON j.job_id = chain.parent_id
        JOIN job_config jc ON jc.job_id = j.job_id
        WHERE chain.current_band = 0
    )
    SELECT input_id, current_band FROM chain WHERE current_band != 0
    """,
    tuple(wire_ids),
)
```

- **Proposed shape under the refactor:** This does not fit `Query.one`/`many`
  cleanly. The `select_clause` of a Query is a fixed list of typed columns from
  a fixed `from_clause` — but a recursive CTE has a *named* working table
  (`chain`) and final `SELECT` against that named table, not against `jobs` or
  `job_config`. The result columns (`input_id`, `current_band`) are aliases,
  not table columns.

  Three options:
  - (a) Drop into `tx.execute` directly inside a free function `reads/jobs.py::resolve_priority_bands(tx, job_ids)`. The Query class doesn't help; the SQL is bespoke.
  - (b) Add a `RawQuery[Row]` that takes a row_cls and arbitrary SQL — basically `tx.execute` plus decoding into a dataclass. But the decode is trivial here (two ints+strings).
  - (c) Generalize `ReadSpec`/`Query` so that `from_clause` can be a `WITH RECURSIVE … (…) SELECT …` template with a fixed `?`-placeholdered parameter list. Awkward.

- **Concerns:**
  1. The §5.1 `Query.one(tx, where=..., params=...)` API assumes "the SQL is a `SELECT col_list FROM from_clause WHERE …`". Recursive CTEs are not that shape. There are at least 4 recursive CTEs in `stores.py` (`get_priority_bands`, `list_descendants`, `list_subtree`, `has_unfinished_worker_attempts`).
  2. Of those four, `list_descendants` and `list_subtree` return a single column (`job_id`), so they don't need a Projection at all — they're trivial helpers. `has_unfinished_worker_attempts` returns a bool. Only `get_priority_bands` needs decoded structured rows.

- **Recommended changes:**
  - Don't try to shoehorn recursive CTEs into `Query`. Add to §5.4 an explicit "raw" convention: ad-hoc SQL that doesn't fit `Query` lives as a free function in `reads/<entity>.py`, calls `tx.fetchall` directly, and decodes results inline. Roughly 8-12 such helpers across the codebase (recursive CTEs, GROUP BY shapes, scalar aggregates).
  - Document the rule in §11.1 alongside the read-class decision tree.

---

## Call Site 8: `TaskStore.state_counts_for_job` — COUNT / GROUP BY aggregate

- **Location:** `lib/iris/src/iris/cluster/controller/stores.py:1133-1138`.
- **Excerpt:**

```python
def state_counts_for_job(self, tx: Tx, job_id: JobName) -> dict[int, int]:
    rows = tx.fetchall(
        "SELECT state, COUNT(*) AS c FROM tasks WHERE job_id = ? GROUP BY state",
        (job_id.to_wire(),),
    )
    return {int(row["state"]): int(row["c"]) for row in rows}
```

Plus the dashboard sibling at `service.py:774-803` (`_task_summaries_for_jobs`)
which has the same shape with `SUM` aggregates:

```python
sql = f"""
    SELECT t.job_id, t.state, COUNT(*) as cnt,
           SUM(t.failure_count) as total_failures,
           SUM(t.preemption_count) as total_preemptions
    FROM tasks t WHERE t.job_id IN ({placeholders})
    GROUP BY t.job_id, t.state
"""
```

Plus the scalar `SELECT COUNT(*) FROM tasks WHERE job_id = ?` at
`service.py:1268`.

- **Proposed shape under the refactor:**
  - `Query.one(tx, where=..., params=...) -> Row | None` — does **not** work. Returns a `row_cls` instance; there is no row_cls for `(state: int, cnt: int)` because the columns are aliased aggregates.
  - `adhoc_projection(("state", int), ("cnt", int))` (today's helper at `schema.py:459`) plus `decode` works, but is exactly the "one more way to read" the refactor tries to eliminate.

- **Concerns:**
  1. **No path in the proposal handles aggregates.** The three read classes (`Query`, `CachedQuery`, `Projection`) all return rows of an entity's `row_cls`. Aggregate result shapes don't belong to any entity.
  2. Today's solution is `adhoc_projection`, which the refactor implicitly assumes will be deleted (it's based on the old `Projection` class). The proposal doesn't say what replaces it.
  3. `service.py:1268` is a scalar COUNT (returning `int`). Even more degenerate — no row_cls, just `int`.

- **Recommended changes:**
  - Add to §5.1 a `Query.scalar(tx, sql, params) -> Any` and `Query.scalars(tx, sql, params) -> list[Any]` for COUNT and SELECT-of-one-column.
  - Add an `AggregateQuery[Row]` (sibling to `Query`) for GROUP BY shapes:

    ```python
    @dataclass(frozen=True)
    class AggregateSpec(Generic[Row]):
        name: str
        sources: tuple[Table, ...]
        sql: str            # full SQL with named ? placeholders
        row_cls: type[Row]
        decoders: tuple[Callable, ...]

    class AggregateQuery(Generic[Row]):
        def __init__(self, spec: AggregateSpec[Row]): ...
        def many(self, tx, params: tuple = ()) -> list[Row]: ...
    ```

  - This is the same shape as `Query`, but with no `where=` / `from_clause` decomposition — caller supplies a full SQL string, plus a `row_cls` of their choice (typically a small dataclass declared next to the Query). Treat it as the canonical home for aggregates, COUNTs, recursive CTEs.

---

## Call Site 9: `list_jobs` — list-with-filters-and-pagination

- **Location:** `lib/iris/src/iris/cluster/controller/service.py:656-732`.
- **Excerpt (abridged):**

```python
conditions = ["j.depth = 0"]
params = []
if query.parent_job_id:
    conditions = ["j.parent_job_id = ?"]
    params = [query.parent_job_id]
... # more dynamic where-clause building
where_clause = " AND ".join(conditions)
order_expr = _SORT_FIELD_TO_SQL.get(sort_field, "j.submitted_at_ms")

count_sql = f"SELECT COUNT(*) FROM jobs j WHERE {where_clause}"

if needs_task_agg:
    select_sql = f"""
        SELECT {JOB_ROW_PROJECTION.select_clause()},
               COALESCE(SUM(t.failure_count), 0) AS agg_failures,
               COALESCE(SUM(t.preemption_count), 0) AS agg_preemptions
        FROM jobs j {JOB_CONFIG_JOIN}
        LEFT JOIN tasks t ON j.job_id = t.job_id
        WHERE {where_clause}
        GROUP BY j.job_id
        ORDER BY {order_expr} {direction}
    """
else:
    select_sql = f"""
        SELECT {JOB_ROW_PROJECTION.select_clause()}
        FROM jobs j {JOB_CONFIG_JOIN}
        WHERE {where_clause}
        ORDER BY {order_expr} {direction}
    """

if limit > 0:
    select_sql += " LIMIT ? OFFSET ?"
    select_params.extend([limit, offset])

rows = q.execute_sql(select_sql, tuple(select_params)).fetchall()
total = q.execute_sql(count_sql, tuple(params)).fetchone()[0]
return JOB_ROW_PROJECTION.decode(rows), total
```

- **Proposed shape under the refactor:** This barely fits `Query` at all.
  Dynamic WHERE construction, dynamic ORDER BY, dynamic GROUP BY / JOIN, and
  a paired COUNT(*) query.

- **Concerns:**
  1. **Dynamic ORDER BY / LIMIT / OFFSET.** §5.1's `Query.one(tx, where, params)` has no slot for `ORDER BY`, `LIMIT`, `OFFSET`. The caller would have to stuff `"… ORDER BY X DESC LIMIT ? OFFSET ?"` into the `where=` string and bind extra parameters, which is fragile (the `WHERE` keyword is fixed in the SQL template; `ORDER BY` after `WHERE` only works because there's nothing else in the template).
  2. **Optional JOIN + GROUP BY based on sort field.** The two variants (`needs_task_agg` vs. not) have different `from_clause`s and different result column shapes (`agg_failures` and `agg_preemptions` extra columns when aggregating). One `Query` can't span both. Today's code uses one `Projection` and ignores the extras.
  3. **Paired COUNT(*) for pagination.** The COUNT(*) is a *different* SQL than the listing SQL (no JOIN, no GROUP BY, no ORDER BY). Under the refactor, that's a separate scalar query (see Call Site 8).
  4. **The result decoder must handle 12 vs 14 columns** depending on the variant. Today's `Projection.decode` quietly tolerates extra columns (it picks the ones it knows about, see `schema.py:421-424`). `Query._decode` in §5.1 zips a fixed decoder tuple over `sqlite3.Row` positions — it can't tolerate extra columns without losing column-name addressing.

- **Recommended changes:**
  - Extend `ReadSpec`/`Query` API:
    ```python
    class Query(Generic[Row]):
        def many(self, tx, *,
                 where: str = "1=1",
                 params: tuple = (),
                 order_by: str | None = None,
                 limit: int | None = None,
                 offset: int | None = None,
                 extra_joins: str = "",
                 group_by: str | None = None) -> list[Row]: ...
    ```
    This grows the API but keeps dashboard call sites legible.
  - Or, accept that dashboard `list_jobs` is bespoke enough to go in `reads/jobs.py` as a free function, building raw SQL and decoding through a shared `JOB_ROW_QUERY._decode` — same pattern as the recursive CTE recommendation.
  - Pair the COUNT(*) using the new `Query.scalar` helper from Call Site 8.

---

## Call Site 10: §4.6 worked example — Projection + Query in same snapshot

The §4.6 sketch of a "scheduler tick mixing endpoint and job state" is:

```python
def reconcile(tx_snap):
    jobs = JOB_DETAIL_QUERY.many(tx_snap, where="state = ?", params=(RUNNING,))
    endpoints = projections.endpoints.query.many(tx_snap)  # SQL, snapshot-iso
```

**There is no real call site in today's codebase that matches this.** I
grepped `controller.py`, `service.py`, `transitions.py`, `stores.py`:

- Every `endpoints.*` read is *standalone* (RPC handler returning endpoint list, or transition deleting endpoints).
- Every `endpoints` write is inside a transaction *that does not read endpoints back in the same tx*.

The closest real co-mix is **Call Site 5** (`healthy_active_workers_with_attributes`):

- Read `workers` via SQL inside `read_snapshot()`.
- Read `_attr_cache` (the future `WorkerAttrsProjection`) *outside* the snapshot.

The "outside the snapshot" part is **load-bearing**. The scheduler tick
explicitly wants the latest committed attributes alongside the snapshot
worker set. If the refactor pushes `worker_attrs` reads inside the snapshot
via `projections.worker_attrs.query.many(snap)`, we get attributes at the
snapshot start time — which means a freshly-registered worker that
committed at T₁.5 sees an empty attrs dict, and the scheduler tick skips it
until the next tick. That is a behavior change.

- **Concerns:**
  1. The §4.6 worked example for "mixing Projection + Query in one snapshot" is **fictional**. There's no existing call site whose desired semantics is "snapshot-isolated Projection read mixed with snapshot-isolated Query read." The only co-mixing site (Call Site 5) wants the *opposite* — latest-committed Projection read alongside snapshot Query read.
  2. The escape hatch (§4.6 "every Projection exposes `.query`") therefore costs API surface for a use case that doesn't exist in production today. YAGNI applies.
  3. The §4.6 "scheduler tick mixing endpoint and job state" worked example should be replaced with the *actual* `healthy_active_workers_with_attributes` example, which has the opposite consistency requirement and motivates the design correctly: **Projection reads should default to latest-committed because that's what production code already relies on.**

- **Recommended changes:**
  - Replace the worked example in §4.6 with the real Call Site 5.
  - Remove the auto-generated `.query` attribute on `Projection`. If a caller eventually needs snapshot-isolated reads of Projection-backed data, they can declare a separate `WORKER_ATTRS_SNAPSHOT_QUERY = Query(…)` explicitly. Two named objects > one Projection with a hidden `.query` companion.
  - Update §11.1 to make this the rule: *Projections serve latest-committed reads only. If you need snapshot isolation, use `Query` against the underlying table directly.*

---

## Cross-Cutting Concerns

### 1. The `Query.one(tx, where, params)` API is too narrow

Across the ten call sites:

| Call site | Fits §5.1 `Query.one`? |
|---|---|
| 1. `get_detail` | Yes |
| 2. `_jobs_with_reservations` | Yes (with `CachedQuery`) |
| 3. `bulk_get_for_updates` | Needs `IN (VALUES …)` + chunking helper |
| 4. `EndpointStore.query` | N/A (Projection, but shape diverges from `Query.one`) |
| 5. `healthy_active_workers_with_attributes` | Worker read fits; attrs need rethinking |
| 6. `WorkerStore.remove` | N/A (write) |
| 7. `get_priority_bands` | **No** — recursive CTE |
| 8. `state_counts_for_job` | **No** — COUNT/GROUP BY |
| 9. `list_jobs` | **No** — dynamic ORDER BY/LIMIT/OFFSET/GROUP BY |
| 10. §4.6 mixed example | Fictional — no real call site |

The hit rate on `Query.one`/`many` as proposed is roughly 40% (sites 1, 2, 3
with help, plus a few of the simpler ones not enumerated). The remaining
60% need additional API surface: scalars, aggregates, recursive CTEs,
ORDER BY/LIMIT/OFFSET, dynamic JOIN composition.

Recommendation: extend §5.1 to declare the *full* read interface up front,
not just `one`/`many`. Specifically:

```python
class Query(Generic[Row]):
    def one(self, tx, *, where: str, params: tuple = ()) -> Row | None: ...
    def many(self, tx, *,
             where: str = "1=1", params: tuple = (),
             order_by: str | None = None,
             limit: int | None = None,
             offset: int | None = None) -> list[Row]: ...
    def scalar(self, tx, sql: str, params: tuple = ()) -> Any: ...
    def scalars(self, tx, sql: str, params: tuple = ()) -> list[Any]: ...

class AggregateQuery(Generic[Row]):
    def many(self, tx, params: tuple = ()) -> list[Row]: ...
```

### 2. The §4.6 consistency model needs reframing

The current §4.6 framing — "Projection reads accept `tx` for symmetry but
serve latest-committed state; use `.query` for snapshot isolation" — is
defensive against a hypothetical hazard ("a caller mixing Query and
Projection reads inside a single QuerySnapshot may observe state that
disagrees with itself"). The real call sites show:

- No production code mixes Projection + Query inside a snapshot today.
- The one near-miss (Call Site 5) **depends on** latest-committed Projection reads alongside snapshot Query reads.

Recommendation: simplify to "Projection reads do not take `tx`; they serve
latest-committed state always. If you need snapshot isolation of a
projected table, declare a separate `Query` against the underlying table."
This eliminates the `del tx  # accepted for symmetry` smell from every
Projection read method.

### 3. Aggregates and recursive CTEs need a home

Today's `adhoc_projection` (`schema.py:459`) handles ad-hoc aggregate
shapes. The refactor deletes the old `Projection` class but doesn't say
what replaces `adhoc_projection`. ~8-12 call sites (`state_counts_for_job`,
`_task_summaries_for_jobs`, `_live_user_stats`, `_parent_ids_with_children`,
budget aggregates, dashboard counts, `_get_running_tasks_with_band_and_value`,
`reservation_claims` scalar lookups, …) currently use raw SQL plus
`adhoc_projection` or hand-rolled dict comprehensions.

Recommendation:

- `AggregateQuery[Row]` (free-form SQL + custom row_cls) — see §5 above.
- `Query.scalar` / `Query.scalars` for one-column / single-value reads.
- For recursive CTEs that return decoded rows: also `AggregateQuery`.
- For recursive CTEs that return a single-column scalar list (`list_subtree`, `list_descendants`): `Query.scalars` (cleaner) or free function in `reads/jobs.py` (simpler).

### 4. Dashboard hot-path caching strategy is undefined

The proposal's `CachedQuery` caches *decoded rows* keyed on blob bytes —
good for proto-heavy reads (`JOB_RESERVATION_QUERY`) but useless for
dashboard queries where the cost is the SQL itself (COUNT over 24k jobs,
GROUP BY on tasks, dynamic-filter `list_jobs`). The decode cost is
negligible there.

No `TTLQuery` or page-level cache is proposed. Today there isn't one
either — dashboard latency is acceptable with raw SQL because SQLite is
fast on these shapes. But the proposal claims to "preserve performance"
and to give "one way to read" — and decode caching is not the answer for
dashboard endpoints.

Recommendation: explicitly document in §3 Non-Goals that **dashboard
SQL-level caching is out of scope**, and the answer for "the dashboard is
slow" is "tune the index or add a materialized view." This is consistent
with the §1 partial-index strategy. Avoid the temptation to add a
`TTLQuery` here; it would invite cache-staleness bugs.

### 5. Cascade-delete writes can silently invalidate Projections

Call Site 6 (`WorkerStore.remove`) demonstrates that FK-cascade deletes
from a `@writes_to`-declared table into a Projection-owned table bypass
the §4.5 startup check. This is a class of bug, not a one-off.

Recommendation: §4.5 needs a `cascades_into(*tables)` declaration on the
`Table` definitions (or equivalent), and the startup check must follow
cascades when validating that writes don't touch Projection-owned tables
without permission.

### 6. Reads need a home, too — not just writes

The §5.4 convention is `writes/<entity>.py`. There's no parallel `reads/`
directory in the proposal. But several call sites we examined have
read-side logic too complex for inline `Query.one` invocations:

- Bulk reads with chunking (Call Site 3).
- Recursive CTEs (Call Site 7).
- Aggregates with custom decoding (Call Site 8).
- Dashboard list builders (Call Site 9).

Recommendation: add `reads/<entity>.py` to the directory layout. It hosts
free functions that wrap one or more `Query`/`CachedQuery`/`AggregateQuery`
calls with whatever surrounding logic the read needs. Without this, the
inline-`Query.one` style implicit in §5.4 forces every caller to repeat
chunking loops, IN-clause building, and decode glue.

---

## Recommended Design Revisions

Targeted edits to `20260511_iris_store_view_refactor.md`:

1. **§5.1 — Extend `Query` API.** Add `scalar`, `scalars`, and `order_by`/`limit`/`offset` parameters to `many`. Add `Query.with_sql(tx, sql, params)` as the escape hatch for SQL that doesn't fit the `SELECT … FROM … WHERE …` template (recursive CTEs).

2. **§5 — New section §5.5 `AggregateQuery`.** Sibling class to `Query`/`CachedQuery` for GROUP BY / aggregate / recursive CTE result shapes. Takes a full SQL string and a `row_cls`. Documented as the replacement for `adhoc_projection`.

3. **§4.6 — Reframe consistency model.** Replace the "scheduler tick mixing endpoint and job state" worked example with the real `healthy_active_workers_with_attributes` example. Drop the auto-generated `.query` attribute on `Projection`. Restate the rule as: *Projections always serve latest-committed reads; snapshot isolation against a Projection-backed table is via a separately declared `Query`.*

4. **§4.6 / §5.3 — Drop `tx` from Projection reads.** The symmetry argument doesn't survive real call sites. Remove the `del tx  # accepted for symmetry` pattern. `Projection.by_id(endpoint_id)`, `Projection.all()`, etc.

5. **§4.5 — Cascade handling.** Add a paragraph on FK cascade: any FK cascade *into* a Projection-owned table must be explicitly mirrored by an `on_commit` hook on the cascading Projection. Add a startup-time check for declared FK cascades into Projection-owned tables.

6. **§5.4 — Add `reads/<entity>.py` convention.** Symmetric to `writes/<entity>.py`. Hosts read helpers for chunking, recursive CTEs, aggregates, and dashboard list builders.

7. **§6.1 — Adjust migration plan.** Commit #2 (port `jobs.reservation` to `CachedQuery`) needs the column-granularity cache key (see Call Site 2). Commit #5/6 (Endpoints/WorkerAttrs Projections) acceptance tests need to include FK-cascade scenarios (Call Site 6) and the latest-committed semantics test (Call Site 5).

8. **§3 Non-Goals — Dashboard SQL caching.** Explicitly call out that decode caching is not a substitute for SQL-level dashboard caching, and that dashboard latency is addressed via indexes/materialized views, not the new cache layer.

9. **§11 — Resolve open questions.** §11.2 (dynamic predicates): answer is "keep raw `where=` plus add `order_by`/`limit`/`offset` named kwargs"; §11.5 (file naming): `reads.py` is the right name for the views file — symmetric with `writes/` directory.

---

## Things That Worked Well

1. **Simple PK lookups (`get_detail`, `get_state`, `get_config`)** map cleanly to `Query.one(tx, where, params)`. About a third of the read sites in `stores.py` are this shape.

2. **`CachedQuery` for the per-tick `JOB_RESERVATION_PROJECTION` read** is the right idea — the existing `cached=True` + `ProtoCache` mechanism translates one-for-one. As long as the cache-key granularity stays per-column (Call Site 2), this is a clean win.

3. **Endpoint write-through semantics** map exactly onto `Projection` — atomic `on_commit` hook plus rehydrate at startup. This is the strongest part of the proposal, and the §6.3 acceptance test list is well-targeted.

4. **`writes/<entity>.py` for write functions** matches the production-orchestrator convention surveyed in §10.5. The `@writes_to(...)` decorator is genuinely lightweight (pure metadata) and gives a real safety property (Projection ownership) for free at startup.

5. **The class split (Query vs CachedQuery vs Projection)** is the right level of distinction for the read paths we examined. The behavioral difference between "always SQL", "SQL + decode cache", and "no SQL, dict only" is meaningful — a single `View(cache=enum)` class would conflate them, as §10.6 argues.

6. **Bulk operations against a single entity table** (e.g. Call Site 3) are *almost* clean — they just need a chunking helper. The decode part is shared with `Query.many`.

7. **Migration plan structure** (12 commits, each independently testable, with explicit perf gates on the hot-path commits) is sound and matches the project's commit-discipline norm.

The proposal's bones are right. The flesh on those bones — specifically the
`Query.one` API and the §4.6 consistency model — needs the revisions above
to handle the ~40% of call sites that aren't simple PK lookups.
