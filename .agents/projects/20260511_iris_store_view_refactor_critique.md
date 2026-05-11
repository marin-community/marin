# Critique: Best-Practices Review of the Iris Data-Layer Refactor

**Date:** 2026-05-11
**Author:** russell.power@gmail.com (with senior-review agent)
**Status:** Companion to `20260511_iris_store_view_refactor.md` and
`20260511_iris_store_view_refactor_addendum_fit.md`
**Scope:** This review reads the proposal and the fit addendum as a single
artifact and asks: given what the addendum surfaced, is the design still on
the right track? Where it isn't, what's the cleanest fix?

---

## Top-line verdict

The refactor is **worth doing**, but **not in the shape currently proposed**.
The original framing — "collapse five mechanisms to three named classes" —
is sound on writes and on the cache-policy axis (`Query` vs `CachedQuery` vs
`Projection` is a meaningful tripartition). It's wrong on two specific
counts: §4.6's "uniform `tx`" claim is a cosmetic constraint with no
production payoff, and the proposed `Query.one(tx, where=, params=)` API
covers about 40% of real reads while pretending to cover 100%. The
addendum's recommendations are roughly in the right direction but
over-correct in the other axis — adding `AggregateQuery`, `reads/`,
`cascades_into`, `Query.scalar`, `Query.with_sql`, plus four positional
kwargs to `many()` would re-create the proliferation the refactor was
explicitly trying to eliminate. The honest answer is: **two classes
(`Query` and `Projection`), one optional caching mixin, raw SQL as a
first-class citizen, an explicit cascade-mirror rule, and `tx` dropped from
Projection reads**. That's a smaller surface than today and smaller than
either prior draft. Migration should ship that surface first and only grow
it under measured pressure.

---

## Part 1: Industry practice for the gaps

### Aggregates and dashboards

The dominant pattern across mainstream ORMs is **aggregates live in the
same query layer as entity reads, returning ad-hoc tuples or
dictionaries** — not in a separate "AggregateQuery" class.

- **SQLAlchemy** exposes aggregates through `func.count()`, `func.sum()`,
  etc. on the same `select()`/`Query` object the user already constructs
  for entity reads. `session.query(func.count(User.id)).group_by(User.name)`
  returns a list of tuples. There is no separate aggregate class.
  Aggregates are just a different shape of `select()`. See [SQLAlchemy 2.0
  SELECT tutorial](https://docs.sqlalchemy.org/en/20/tutorial/data_select.html)
  and [SQLAlchemy ORM Query API](https://docs.sqlalchemy.org/en/14/orm/query.html).
- **Django** uses `.aggregate(...)` / `.annotate(...)` on the same
  `QuerySet` API. The result is a `dict` (for `.aggregate`) or a `QuerySet`
  of model instances with extra attributes (for `.annotate`). Same Manager,
  same chainable API, different terminal method.
- **Rails ActiveRecord** uses `count`, `sum`, `group(...)` directly on the
  Relation. No separate class.
- **Spring Data JPA** allows custom result projection interfaces and
  `@Query` annotations returning arbitrary DTOs; the repository interface
  hosts both entity and aggregate methods.
- **Marten**, the closest direct analog to our `Projection` class, treats
  aggregate projections as a **separate kind of projection** (built up by
  event streams) but exposes them through the same `IQuerySession` LINQ
  surface as entity reads. See [Marten Aggregate
  Projections](https://martendb.io/events/projections/aggregate-projections.html).

The **Rails Query Object pattern** is the relevant counter-example: when a
read gets complex enough that "where do I put it on the Model?" stops
having a clean answer, the community extracts a `*Query` class in
`app/queries/`. The defining feature is "one Query Object per
meaningful business question", not "one Query Object per result shape".
Dashboards typically get a `DashboardOverviewQuery` that internally calls
several smaller methods. See [Selleo — Rails Query
Objects](https://medium.com/selleo/essential-rubyonrails-patterns-part-2-query-objects-4b253f4f4539)
and [iRonin —
Query Objects](https://www.ironin.it/blog/design-patterns-in-large-rails-applications-query-objects.html).

**Takeaway for us.** The industry does not have a separate "AggregateQuery"
class. The closest model is "the same query API supports aggregates by
returning ad-hoc tuples / dataclasses", and complex multi-aggregate reads
get their own named function (or named class). The addendum's
`AggregateQuery[Row]` would be idiosyncratic. A cleaner shape is: `Query`
can return any `row_cls` (including a small dataclass declared next to the
query for the aggregate's shape), and `Query.scalar` exists for COUNT-style
single-value reads. Both reduce to "Query is the shape-and-decode
contract; the SQL can be anything".

### Recursive CTEs

The mainstream answer is **raw SQL plus a thin decoder**, regardless of
the ORM.

- **Django** doesn't natively support recursive CTEs. The community uses
  `django-cte` for a typed surface, or drops to `raw_cte_sql()` /
  `Manager.raw()` for one-off use. The django-cte docs explicitly note
  "each result field in the raw query must be explicitly mapped to a field
  type" — i.e., the decode layer is the only piece that's reused. See
  [django-cte](https://dimagi.github.io/django-cte/).
- **SQLAlchemy** has first-class CTE support via
  `select(...).cte(recursive=True)`, but real codebases overwhelmingly
  drop to `connection.execute(text("WITH RECURSIVE ..."))` for any
  non-trivial recursive query — the typed CTE API is significantly more
  verbose than the raw form for these shapes.
- **Rails ActiveRecord** has no native recursive support; everyone uses
  `find_by_sql` or raw `connection.execute` and constructs result objects
  manually.

**Takeaway for us.** There is no industry consensus on a "typed recursive
CTE API" because nobody has built one that's lighter than raw SQL. The
addendum's instinct — let recursive CTEs be raw `tx.execute` plus inline
decode, organized into `reads/<entity>.py` — is correct in spirit, but the
right primitive is "Query lets you pass a fully custom SQL body and
returns decoded `row_cls`". We don't need a new `RawQuery` class or
`AggregateQuery`; we need `Query` to accept arbitrary SQL.

### Pagination / dynamic ORDER BY

The dominant pattern is **builder-style chaining** (SQLAlchemy, Django,
ActiveRecord all do this): `query.order_by(...).limit(...).offset(...)`.
Each method returns a new query/select, parameters are validated at
chain-construction time, and ORDER BY is restricted to whitelisted
columns. See [SQLAlchemy pagination
patterns](https://docs.sqlalchemy.org/en/20/tutorial/data_select.html) and
[Flask-SQLAlchemy](https://flask-sqlalchemy.palletsprojects.com/en/stable/api/).

A non-chainable kwarg form (`query.many(where=..., order_by=..., limit=...)`)
is what FastAPI / DRF / similar HTTP frameworks expose at the request
boundary, but they almost always translate that into a builder call
underneath. The kwarg form is fine for our scale (a handful of dashboard
queries) but loses validation: `order_by="j.depth; DROP TABLE"` is a
SQL-injection sink unless the receiver whitelists.

The keyset / cursor pagination literature (e.g.
[sqlakeyset](https://github.com/djrobstep/sqlakeyset)) is interesting but
overkill for our scale. Offset-pagination on small dashboards is fine.

**Takeaway for us.** The addendum's `many(tx, where=, order_by=, limit=,
offset=)` extension is functionally OK but inherits the SQL-injection
exposure unless we ship a whitelist. The cleanest answer is **don't put
ORDER BY/LIMIT into the `Query` API at all** — let dashboard call sites
build their full SQL string from a Query's `select_clause` and dispatch
through a `Query.fetch_raw(tx, full_sql, params)` method that just runs
the decoder over the result rows. This matches what `list_jobs` already
does today (`service.py:656-732`).

### Multi-step reads / Query Services

In CQRS literature, multi-step reads belong to a **Query Handler** or
**Read Service** layer, distinct from both the Repository (write side)
and the Projection (materialized read model).

- [Microsoft's CQRS guidance](https://learn.microsoft.com/en-us/dotnet/architecture/microservices/microservice-ddd-cqrs-patterns/cqrs-microservice-reads)
  explicitly describes "the query model can use its own data schema",
  "lightweight DTOs", and a query handler layer separate from the domain
  model.
- [Cosmic Python chapter
  12](https://www.cosmicpython.com/book/chapter_12_cqrs.html) shows
  read-side handlers as free functions in a `views.py` module, taking a
  session and returning denormalized dicts. The pattern is: thin function
  per question, raw SQL inside, no class hierarchy.
- [Radek Maziarka's CQRS
  series](https://radekmaziarka.pl/2018/01/08/cqrs-third-step-simple-read-model/)
  is explicit: complex dashboards are "read models" — separate from both
  Repository and Projection, often built by raw SQL against the same
  database. The defining feature is that the read service is allowed to
  reach across aggregate boundaries that the write side keeps separate.
- The Rails Query Object pattern's "DashboardOverviewQuery" file under
  `app/queries/` is the same idea at a smaller scale.

**Takeaway for us.** The addendum's `reads/<entity>.py` directory is the
right shape — it matches Cosmic Python's `views.py`, Rails's
`app/queries/`, and the CQRS query-handler layer. The mistake is naming
it parallel to `writes/` ("symmetry") and organizing it by entity. **Read
services are organized by question, not by entity** — `dashboards.py`,
`scheduler.py`, `lifecycle.py` are better names than `reads/jobs.py`,
because a dashboard read pulls from `jobs`, `tasks`, and `task_attempts`
and there's no honest place for it under any single entity.

### Cascade-aware invalidation

Three mainstream answers, all imperfect:

1. **Trigger-based refresh.** Postgres materialized views combined with
   refresh triggers; see [Hashrocket's
   guide](https://hashrocket.com/blog/posts/materialized-view-strategies-using-postgresql)
   and [Netguru on Rails MV refresh
   triggers](https://www.netguru.com/blog/materialized-view-refresh-problem-and-how-to-solve-using-database-triggers-in-ror-based-application).
   The DB does the work. The cost is that `REFRESH MATERIALIZED VIEW`
   recomputes the whole MV, not the delta. `REFRESH CONCURRENTLY` helps
   for read latency but not for write cost.

2. **Event-handler subscription** (Axon, Marten, Kurrent, EventStoreDB).
   The projection subscribes to the underlying event stream; when an event
   touches an aggregate the projection cares about, the framework
   dispatches it to the projection's handler. See [Axon subscription
   queries](https://axoniq.io/blog-overview/introducing-subscription-queries)
   and [Marten event-driven
   projections](https://event-driven.io/en/projections_in_marten_explained/).
   This is the gold standard for cascade-correctness because the
   projection sees every state-changing event by construction. The cost is
   that it requires an event log.

3. **Manual mirror with discipline.** What we have today: explicit calls
   like `set_worker_attributes(...)` and `remove_from_attr_cache(...)`
   alongside any write that affects the cache. Brittle but cheap.

The Postgres MV `dbt-adapters` bug ([dbt-labs
#1714](https://github.com/dbt-labs/dbt-adapters/issues/1714)) — where
refreshing a MV cascade-deletes dependent views — is a real-world warning:
even the "the DB tracks it" approach has surprising failure modes.

**Takeaway for us.** Without an event log we can't get option 2. Trigger-
based refresh in SQLite is possible (see [SQLite triggers as MV
replacement](https://madflex.de/SQLite-triggers-as-replacement-for-a-materialized-view/))
but the cost of plumbing it for two tiny dictionaries (endpoints,
worker_attrs) is not worth it. The addendum's `cascades_into(*tables)`
declaration is option 3 with a static check — fine, and matches the
proposal's existing `@writes_to` philosophy. **But** the cleanest answer
is to recognize that the failure mode is specific to FK cascades, and the
simplest fix is a rule: **no `ON DELETE CASCADE` from outside-projection
tables into projection-owned tables**. If you need that cascade, declare
the projection as also owning the parent table, or do the deletion
manually inside the projection's write method. This is what the call-site
6 analysis was pushing toward; it just needs to be stated as a hard rule,
not as a runtime check.

### Uniform-handle ergonomics

The §4.6 "every read takes `tx` for API symmetry" claim is
**unprecedented** in mainstream ORMs, and that's evidence against it.

- **SQLAlchemy** doesn't pass `session` to every operation; the session is
  the unit of work and the work is methods on it (`session.execute(...)`,
  `session.scalars(...)`). Read-side helpers in `views.py` (Cosmic Python)
  *do* take a session because they emit SQL. But in-memory caches don't —
  and SQLAlchemy's `identity_map` is hidden behind the session and never
  exposed as a separate read class taking `session` "for symmetry".
- **Django** doesn't pass anything; the connection is thread-local and
  models are bound to a default manager.
- **Rails** is similar — connection is thread-global.
- **Marten** passes an `IDocumentSession` to every query, including
  projection reads, **because the session is also the snapshot boundary**.
  If you want a snapshot, you open a session; if you want latest-committed
  state, you open a different kind of session (`IQuerySession`). The
  important point: Marten doesn't pass the session to be symmetric. It
  passes it because the session **is** the consistency choice.

The addendum's recommendation — drop `tx` from `Projection` reads
entirely — aligns with this. The §4.6 argument that "future-proofing for
a hypothetical `VersionedProjection` means every Projection method should
take `tx`" is YAGNI: when versioned projections arrive, they'll be a new
class with a different name, and the 6-12 call sites that use the
versioned variant will be updated. **Forcing every Projection read site
forever to thread an unused tx is more code-cost than that future
refactor.**

---

## Part 2: Critique

### On §4.6's tx-uniformity (and the addendum's proposed fix)

The §4.6 argument breaks down in three ways:

1. **The worked example is fictional.** As the addendum showed, no
   production call site mixes `Projection` and `Query` reads inside a
   single snapshot today. The example the doc uses (`reconcile` mixing
   endpoint and job state via `endpoints.query.many(tx_snap)`) was
   constructed for the doc, not pulled from real code. Designing API
   shape against a fictional consumer is the wrong direction; we should
   design against `healthy_active_workers_with_attributes` (the one real
   co-mixing site), which **wants the opposite** — latest-committed
   Projection reads alongside snapshot Query reads.

2. **The "symmetry" rationale loses the moment Projection methods aren't
   `.one(...)` / `.many(...)`.** EndpointsProjection has `by_id`,
   `by_name`, `all`, `by_task`, `query`. WorkerAttrsProjection has `get`,
   `all`, `remove`. None of these have the same call-site shape as
   `Query.one(tx, where=, params=)`. The argument that "scanning code you
   see `xxx.method(tx, ...)` everywhere" is a degraded form of the
   stronger uniformity that doesn't actually exist in the API: the
   method-name space is already diverse, so adding `tx` as a leading
   positional doesn't recover scan-ability.

3. **`del tx  # accepted for symmetry; not consulted` is a code smell.**
   Any reader who lands on this line learns immediately that the argument
   is decorative. Once the code says "this parameter is ignored", the
   pretense of uniformity is broken; the cost of the argument is now
   non-zero (every caller threads it; every test passes it) with no
   conceptual benefit.

The addendum's fix — drop `tx` from Projection reads — is right. **There's
a third option neither agent surfaced**, though: rather than drop `tx`
unconditionally, **make `Projection` a stricter consistency contract**.
Specifically:

- `Projection.<read>(...)` always returns latest-committed state. No `tx`.
- `Projection.snapshot_view(snap)` returns a frozen snapshot of the
  projection's dict, captured at the moment the snapshot was taken. The
  snapshot-view object then has all the same read methods (`by_id`,
  `all`, etc.) but serves the snapshotted data.

This is a known pattern from immutable-data-structure literature (Clojure
atoms, Datomic's `db-as-of`, Marten's `IDocumentSession.QueryAt(...)`).
It would let the scheduler tick truly mix snapshot Query and snapshot
Projection reads if it ever needed to, without paying the cost on every
Projection call site that doesn't.

But the cost of building this is real (versioned dict, copy-on-write,
eviction policy). And the only call site that wants it (Call Site 5)
explicitly wants the *opposite* semantics. **Verdict: do the addendum's
fix. Drop tx. Move on.** If we ever discover a snapshot-Projection
consumer, we add `snapshot_view` then.

### On the proliferation of API surface (Query.scalar, AggregateQuery, kwargs)

The addendum surfaced real gaps, then proposed a solution that grows the
API roughly as follows:

| Today | Proposal §5 | Addendum extension |
|---|---|---|
| `Projection.decode_one`, `decode`, `adhoc_projection`, `ProtoCache`, `Column(cached=True)` | `Query.one`, `Query.many`, `CachedQuery.one`, `CachedQuery.many`, `Projection.<various>`, `Projection.query` | `Query.scalar`, `Query.scalars`, `Query.with_sql`, `Query.many_in`, `AggregateQuery.many`, plus `where=`, `order_by=`, `limit=`, `offset=`, `extra_joins=`, `group_by=` kwargs |

This is moving in the wrong direction. The original goal was "five
mechanisms → three." The addendum's extensions would land at roughly
**eight named entry points** plus a builder-style kwarg surface. The
cognitive overhead trends back toward today's, just with different
names. That's not a refactor, it's a rename plus accretion.

The right move is to **make Query do less, and let raw SQL be the escape
hatch for everything else**. Concretely:

```python
class Query(Generic[Row]):
    """Decoded read against SQL.

    The Query owns the decoder; it does NOT own the SQL. Callers supply
    SQL either via the convenience `select_clause` + `from_clause` (for
    simple shapes) or via `fetch_raw` (for everything else).
    """

    def __init__(self, spec: ReadSpec[Row]): ...

    @property
    def select_clause(self) -> str: ...      # for callers building SQL

    def one(self, tx, where: str, params: tuple = ()) -> Row | None: ...
    def many(self, tx, where: str = "1=1", params: tuple = ()) -> list[Row]: ...

    def fetch_raw(self, tx, sql: str, params: tuple = ()) -> list[Row]:
        """Run arbitrary SQL; decode each result row into Row.

        SQL must produce exactly the columns in `self.select_columns` in
        order. Used for recursive CTEs, dashboard list builders, anything
        with custom ORDER BY/LIMIT/GROUP BY, and bulk reads with custom
        IN-clauses.
        """
```

That gives us **three methods** on `Query` covering everything. No
`AggregateQuery`. No `scalar` (covered by `fetch_raw` + a trivial decoder
for `int`/`str`). No `where=/order_by=/limit=/offset=` kwarg explosion.
For aggregates and recursive CTEs, callers declare a small `row_cls`
alongside the SQL and reuse `Query`'s decoder machinery.

The addendum's `Query.scalar`/`scalars` is *the only* legitimate
add-on, because scalar reads are so common (COUNT, MAX, single-column
fetches) that decoding through a 1-field dataclass is friction. Add them.
But the other extensions are accretion. Push back on them.

### On the reads/ directory

The addendum's `reads/<entity>.py` proposal is half-right and half-wrong.

**Half-right:** complex multi-statement reads (bulk chunking, dashboard
list builders, recursive CTE wrappers) need a home. Inlining them at the
call site forces every consumer to re-implement chunking. Putting them on
the (now-deleted) Store classes isn't an option. They need a module.

**Half-wrong:** the entity-based naming (`reads/jobs.py`,
`reads/tasks.py`) is the wrong axis. Multi-step reads are organized by
**question**, not by entity. The `_task_summaries_for_jobs` helper at
`service.py:774-803` reads from `tasks` but is used by the dashboard list
endpoint that also reads from `jobs`; calling it `reads/tasks.py` mis-
organizes the cohesion. `_jobs_with_reservations` reads from `jobs +
job_config` but is used by the scheduler tick; putting it under
`reads/jobs.py` orphans it from the scheduler.

The clean shape is:

- `views/dashboard.py` — `list_jobs`, `_task_summaries_for_jobs`,
  `_live_user_stats`, the dashboard scalar counts. These compose multiple
  Queries.
- `views/scheduler.py` — `_jobs_with_reservations`, `bulk_get_attempts`
  (the chunking one), `healthy_active_workers_with_attributes`.
- `views/lifecycle.py` — `get_priority_bands`, `list_descendants`,
  `list_subtree`, `has_unfinished_worker_attempts`. These are the
  tree/recursive helpers that the scheduler+transitions reach for during
  job-state mutations.

This is Cosmic Python's `views.py` pattern applied at one level of
hierarchy. Each file is "the reads needed by this use-case", not "the
reads against this table".

(Naming: I called it `views/` to align with Cosmic Python. The proposal's
§11.5 considered `views.py` and rejected it as clashing with HTTP-handler
"views". That's a fair concern for the class-holding file but not for a
read-helper directory — the cosmic-python connotation is more apt than
"reads".)

### On the three-class taxonomy in light of the gaps

The taxonomy is **two-and-a-half** classes, not three. Let me unpack.

- `Query` is the row-decoding contract. It's a real thing — typed
  columns, decoder pipeline, optional cached-decode wrappers. We need
  this.
- `CachedQuery` is `Query` with a memoizing decoder. In the addendum's
  per-column cache shape (Call Site 2), `CachedQuery` is just `Query`
  with `cached=True` columns. There is no behavioral difference at the
  call site — every read shape works identically. **CachedQuery should be
  a flag on `ReadSpec` (or on `Column`), not a separate class.** The
  AGENTS.md "separate classes over boolean flags" rule applies to
  *behavioral* variants. Per-column cache is a *decoding optimization*, not
  a behavioral variant — read sites use it identically.
- `Projection` is the write-through materialized view. Genuinely
  different from `Query`: owns mutations, owns lifecycle, doesn't take
  `tx` on reads. We need this.

So the honest tripartition is **`Query` + `Projection` + a
caching-decoder mixin** (or `cached=True` columns reused from today).
That's two classes plus an attribute. Smaller surface than the proposal's
three. The addendum's per-column cache analysis (Call Site 2) is
implicitly arguing for this without saying it.

If we keep `CachedQuery` as a class, the only payoff is the visual
distinction at the read-site declaration — "this reader knows to use the
cache". That's worth something. But it's much weaker than the
`Query`/`Projection` distinction. Recommend collapsing.

### On scope and migration plan feasibility

Original migration: 12 commits, ~1860 added / 1380 removed.

After the addendum's recommendations, realistically:

- AggregateQuery class + tests: +200 LOC
- `reads/` directory and the 8-12 helpers it contains: +400 LOC
- `Query.scalar`/`scalars`/`with_sql`/`many_in`: +150 LOC
- `cascades_into` declaration + startup check: +80 LOC
- Documentation updates throughout: +200 LOC

Plus migration churn from the wider Query API surface. Net new code:
~+1000. Net delete still applies but doesn't grow.

That pushes the PR from "large but reviewable" to "very large." Two
specific risks:

1. **The migration is no longer one PR.** The original plan's "commit 11
   deletes 1000 LOC at the end" works because the intermediate state is
   well-defined: each commit ports one entity. With the addendum's
   extensions, commits 2-4 now also depend on AggregateQuery existing
   (because `state_counts_for_job` migrates as part of tasks), so the new
   scaffolding has to land in commit 1, not be added gradually. Commit 1
   becomes ~+800 LOC of scaffolding before any porting happens. Reviewers
   will struggle.

2. **The "preserve behavior" tests get harder.** Today's `adhoc_projection`
   is called from ~10 sites; each site has subtly different SQL. The
   addendum's `AggregateQuery` would normalize the API but each site
   needs a separate `AggregateSpec` declaration. That's 10 new
   declarations to write, each with its own row_cls. Higher chance of a
   transcription bug.

The recommended slimmer shape (Query + Projection + `cached=True` + raw
SQL via `fetch_raw` + a `views/` directory) is:

- Commit 1: ~+300 LOC scaffolding (smaller than even the original)
- Per-entity port commits: roughly unchanged
- `views/` directory: introduced gradually, one file per port commit
- Net: roughly matches the original plan's size, with cleaner endgame

**Migration is still feasible in one PR with the slim shape**. With the
addendum's full extension list, it isn't — it needs to be split into a
"scaffolding + Query/Projection ports" PR followed by a "harvest the
remaining ad-hoc reads into views/" PR.

### Missing concerns the prior agents didn't surface

A few things neither the proposal nor the addendum addressed:

1. **Observability when a Projection rehydrate is slow.** EndpointsProjection
   today is tiny (hundreds of rows). WorkerAttrsProjection is tiny (one row
   per (worker_id, attr_name)). If we add a third Projection in the future
   over a larger table (a tempting move once the pattern is in place), and
   its `rehydrate()` runs at every controller startup AND after every
   `replace_from`, startup latency grows silently. The proposal doesn't
   say where to put rehydrate timing or how to alert on regressions. Add a
   `rehydrate_ms` metric per Projection at startup.

2. **What does the controller log on a CachedQuery cache miss?** Today's
   `ProtoCache` is invisible. With `CachedQuery`, a miss is a decode
   operation — observable. But hit rate is only interesting in aggregate;
   logging every miss would be noise. The right shape is a periodic
   counter dump (cache_hits / cache_misses / size) into the structured
   log, maybe every 60s, gated on a debug flag. Worth one paragraph in
   the design doc.

3. **Concurrent rehydrate.** `Projection.rehydrate(db)` opens
   `db.read_snapshot()` and rebuilds the dict. If a write happens during
   rehydrate (impossible at startup, possible after `replace_from` if
   anything else is reading the controller), the dict could end up
   inconsistent. The proposal handwaves this with "under the lock", but
   the snapshot-read happens *outside* the lock (we don't hold the dict
   lock while doing SQL fetches — that would block reads for the duration
   of rehydrate). Need to spell out: rehydrate runs while the controller
   is otherwise quiesced; document this contract in
   `Projection.rehydrate` docstring.

4. **Debugging ergonomics for "the cache is wrong."** When (not if) a
   Projection's dict goes stale due to a missed cascade or a manual SQL
   write, what does an operator do? Today: SSH in and grep
   `/diagnostics/`-style endpoints. Under the refactor: same, but the
   Projection class should expose `assert_consistent(tx)` that
   re-rehydrates from SQL and compares. Useful as a periodic
   self-check; can be wired into the existing health endpoint.

5. **Migration safety for in-flight controllers.** The proposal's "one
   PR" path means a single deploy crosses the entire refactor. There's no
   "ramp" possible. For a single-process controller this is fine; for
   anyone running multiple controllers in different deploy states it
   isn't. We are single-process, so this is OK. But worth stating in §3
   Non-Goals as a deliberate choice.

6. **`@writes_to` decorator doesn't compose.** If a write function calls
   another write function (e.g., `remove_worker` internally calls
   `clear_worker_attrs` which is itself decorated), the `writes_to`
   metadata is on each individually — we don't union them. For the
   startup check this is fine (we walk every function). For runtime debug
   (if we ever add it), this would be a footgun. Note in the
   `@writes_to` docstring that it's per-function, not transitive.

---

## Part 3: Final feedback

### Is this refactor still worth doing?

**Yes, but slimmer than either prior draft.** The diagnosis in §1 of the
proposal is correct: five mechanisms is one too many, and the cognitive
cost is real even if the perf is fine. The cure is the right kind of
cure: collapse the policies into named types and make ownership of cache
state explicit.

Where the proposal goes wrong is §4.6 (uniform tx). Where the addendum
goes wrong is over-correcting by adding too much API surface. The right
shape is the **two-class shape** (Query + Projection), with raw SQL as
the explicit escape hatch, ad-hoc `row_cls` dataclasses for aggregate
shapes, and a `views/` directory organized by use-case for multi-step
reads.

### Top 3 design-doc revisions

In priority order:

1. **Drop the §4.6 uniform-`tx` claim entirely.** Replace with: "Reads
   that touch SQL take a `tx`. Reads that serve from in-memory state
   (Projections) don't." Update the worked example to be the real
   `healthy_active_workers_with_attributes` call site, which depends on
   latest-committed Projection reads alongside snapshot SQL reads.
   Remove the `Projection.query` auto-companion. This is a strict
   simplification — fewer arguments, fewer concepts, fewer misleading
   examples.

2. **Replace the `Query.one(where=, params=)` API with `one`/`many` for
   the simple shape plus `fetch_raw(tx, sql, params)` as the explicit
   escape hatch.** Add `Query.scalar` / `Query.scalars` only. Reject
   `AggregateQuery`, `Query.with_sql`, `Query.many_in`, and the
   `order_by=/limit=/offset=` kwargs. Aggregates and CTEs declare a
   small `row_cls` next to the Query and call `fetch_raw`. This handles
   100% of call sites with the smallest surface.

3. **Establish a `views/` directory organized by use-case** (not by
   entity), holding free functions for multi-step reads. Move
   `_jobs_with_reservations`, `state_counts_for_job`,
   `_task_summaries_for_jobs`, `list_jobs`, `bulk_get_attempts`, the
   recursive CTEs, and `healthy_active_workers_with_attributes` into
   files named for the use-case (`views/scheduler.py`,
   `views/dashboard.py`, `views/lifecycle.py`). Add a §11.1 rule:
   "Every multi-statement read lives in `views/`; single-statement reads
   live at the call site."

Lower-priority but still important:

4. Resolve `CachedQuery` to `cached=True` columns reused from today.
   Don't introduce a separate class for what is, behaviorally, a
   decoding optimization.

5. Add `Projection.cascades_from(*tables)` plus a hard rule: no
   `ON DELETE CASCADE` from a non-projection table into a
   projection-owned table. The startup check refuses to start the
   controller if this rule is violated. Real fix for the Call Site 6
   gap, no runtime cost.

6. Add `Projection.rehydrate_ms` metric and `Projection.assert_consistent(tx)`
   debug helper. Spell out the rehydrate-quiescence contract.

### Recommended next steps

1. **Update the design doc with the three top revisions above.** Roughly
   a 200-line diff to the proposal; the addendum can mostly be folded
   into the proposal's history section rather than living as a separate
   doc.

2. **Re-do the migration plan against the slimmer shape.** I expect 10
   commits instead of 12: scaffolding (1), per-entity ports (4 commits:
   jobs, tasks, attempts, workers/reservations), Projection ports (2:
   endpoints, worker_attrs), `views/` harvest (1 commit per views file
   ~= 3 commits), startup checks (1 commit), delete Stores (1 commit).

3. **Don't ship `AggregateQuery` or the `Query.*` kwarg explosion.**
   If after migration we discover that `fetch_raw` is being called with
   the same boilerplate at five sites, *then* extract a helper. Don't
   pre-extract.

4. **Validate against the 10 call sites in the addendum.** Re-walk each
   site under the slimmer shape and confirm it's cleaner than today. If
   any becomes meaningfully worse, the slim shape is wrong somewhere
   specific.

5. **Land the `views/` reorganization first, as a no-functional-change
   PR.** If we can move `_jobs_with_reservations`, `list_jobs`, and the
   CTE helpers out of `stores.py` and `service.py` into `views/*` files
   *before* the class refactor, the class refactor becomes much smaller
   and reviewable. This sequencing wasn't considered by either prior
   agent and is the single biggest reduction in PR risk.

---

## Sources

Industry survey:

- [Martin Fowler — Repository (PoEAA)](https://martinfowler.com/eaaCatalog/repository.html)
- [Martin Fowler — CQRS](https://martinfowler.com/bliki/CQRS.html)
- [Microsoft — CQRS pattern](https://learn.microsoft.com/en-us/azure/architecture/patterns/cqrs)
- [Microsoft — Implementing reads/queries in a CQRS microservice](https://learn.microsoft.com/en-us/dotnet/architecture/microservices/microservice-ddd-cqrs-patterns/cqrs-microservice-reads)
- [Cosmic Python — CQRS chapter](https://www.cosmicpython.com/book/chapter_12_cqrs.html)
- [Cosmic Python — Unit of Work](https://www.cosmicpython.com/book/chapter_06_uow.html)
- [Cosmic Python blog — Commands, Queries, Handlers, and Views](https://www.cosmicpython.com/blog/2017-09-13-commands-and-queries-handlers-and-views.html)
- [Radek Maziarka — CQRS, Third step: Simple read model](https://radekmaziarka.pl/2018/01/08/cqrs-third-step-simple-read-model/)
- [Marten — Projections](https://martendb.io/events/projections/)
- [Marten — Aggregate Projections](https://martendb.io/events/projections/aggregate-projections.html)
- [Marten — Read-Model Projections tutorial](https://martendb.io/tutorials/read-model-projections)
- [Event-Driven.io — Projections in Marten explained](https://event-driven.io/en/projections_in_marten_explained/)
- [Axon — Introducing Subscription Queries](https://axoniq.io/blog-overview/introducing-subscription-queries)
- [Baeldung — Persisting the Axon Query Model](https://www.baeldung.com/axon-persisting-query-model)
- [SQLAlchemy 2.0 — Session Basics](https://docs.sqlalchemy.org/en/20/orm/session_basics.html)
- [SQLAlchemy 2.0 — SELECT tutorial](https://docs.sqlalchemy.org/en/20/tutorial/data_select.html)
- [SQLAlchemy ORM Query API (1.4)](https://docs.sqlalchemy.org/en/14/orm/query.html)
- [django-cte — Common Table Expressions for Django](https://dimagi.github.io/django-cte/)
- [Django ticket #28919 — CTE support](https://code.djangoproject.com/ticket/28919)
- [Selleo — Rails Query Objects](https://medium.com/selleo/essential-rubyonrails-patterns-part-2-query-objects-4b253f4f4539)
- [iRonin — Query Objects in large Rails apps](https://www.ironin.it/blog/design-patterns-in-large-rails-applications-query-objects.html)
- [OneUptime — How to Implement Query Objects Pattern in Rails](https://oneuptime.com/blog/post/2025-07-02-rails-query-objects/view)
- [Hashrocket — Materialized View Strategies in PostgreSQL](https://hashrocket.com/blog/posts/materialized-view-strategies-using-postgresql)
- [Netguru — Materialized View Automatic Refresh via Triggers](https://www.netguru.com/blog/materialized-view-refresh-problem-and-how-to-solve-using-database-triggers-in-ror-based-application)
- [Postgres docs — REFRESH MATERIALIZED VIEW](https://www.postgresql.org/docs/current/sql-refreshmaterializedview.html)
- [scenic-views #275 — refresh with cascade](https://github.com/scenic-views/scenic/issues/275)
- [dbt-adapters #1714 — refresh-cascade deletes dependent views](https://github.com/dbt-labs/dbt-adapters/issues/1714)
- [sqlakeyset — offset-free paging for SQLAlchemy](https://github.com/djrobstep/sqlakeyset)
- [Flask-SQLAlchemy pagination API](https://flask-sqlalchemy.palletsprojects.com/en/stable/api/)
- [SQLite triggers as materialized-view replacement](https://madflex.de/SQLite-triggers-as-replacement-for-a-materialized-view/)
