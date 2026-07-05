# Unifying the controller's in-memory caches under one Projection concept (#6946)

Status: design locked, implementation in progress. Branch `weaver/issue-6946`.
Companion to `20260704_iris_derive_attempt_counts.md` (the attempt-counts work that
started this). Two independent model reviews (Fable) plus a full call-site map fed
this design; the external prior-art survey is in §2.

## Problem: five ad-hoc cache/memo patterns, two of them buggy

The controller accreted five different in-memory caches over its SQLite DB, each of
which independently improvised its **reachability** (threaded param vs. registry) and
its **maintenance** (write-through vs. invalidate vs. eager-pop):

| # | Cache | Derives from | Fill | Maintenance | Reach today |
|---|---|---|---|---|---|
| 1 | `Tx.memo` | (uncommitted tx state) | lazy per-tx | none (dies at commit) | `tx.memo[str]` |
| 2 | `EndpointsProjection` | `endpoints` table | eager rehydrate | write-through post-commit | threaded `endpoints=` (~21 sites) |
| 3 | `WorkerAttrsProjection` | `worker_attributes` table | eager rehydrate | write-through post-commit | threaded, **per-backend** |
| 4 | `AttemptCountsProjection` | `task_attempts` (aggregate) | lazy/cache-aside | invalidate post-commit | `db.caches[type]` registry |
| 5 | `RunTemplateCache` | `job_config` + workdir files | lazy/read-through (LRU) | **eager pre-commit pop** | threaded (~10 sites) |

Two of these carry live correctness bugs, both instances of the classic *stale set*
(a reader recomputes a value from a snapshot taken **before** a write and installs it
**after** the write's invalidation — the value is then stale until the next write):

- **#4** — `get_jobs` recomputes on a read-miss inside the caller's read snapshot,
  which does not hold the write lock; a concurrent attempt-commit's post-commit pop
  can be overtaken by the reader's write-back. Flagged by an automated reviewer on
  PR #6956; confirmed real.
- **#5** — `insert_job_and_config` pops the template **inside the write transaction,
  before the new `job_config` row commits** (`ops/job.py:147`). A concurrent dispatch
  reader refills from the pre-commit snapshot, so a resubmitted job (same `JobName`)
  serves the **prior** submission's payload. This is exactly the same-name-replacement
  case the pop was meant to guard.

The maintainer's direction: *"there must be a unifying caching abstraction… we've now
got 4 different ad-hoc patterns, hard to manage, vs something unifying like materialized
views… prefer one coherent concept and rework everything as needed."*

## 2. Prior art (external survey)

The pattern — *an in-memory value derived from base tables, kept consistent as those
tables change* — is a **materialized view**, and the maintenance question is the
**cache-strategy taxonomy**. Sources below are primary (official docs / the papers).

**Caching-strategy taxonomy.** Cache-aside (lazy/lookaside), read-through, write-through,
write-behind, refresh-ahead. Definitions: [Oracle Coherence, *Caching Data Sources*](https://docs.oracle.com/en/middleware/standalone/coherence/14.1.1.0/develop-applications/caching-data-sources.html)
(canonical origin of "refresh-ahead"); [AWS, *Database Caching Strategies Using Redis*](https://docs.aws.amazon.com/whitepapers/latest/database-caching-strategies-using-redis/caching-patterns.html);
[Azure, *Cache-Aside pattern*](https://learn.microsoft.com/en-us/azure/architecture/patterns/cache-aside).
"Derive an aggregate + memoize + invalidate on write" is **cache-aside with explicit,
event-driven invalidation** — the value is recomputed (not written) so write-through
does not apply, and invalidation is event-driven (not TTL) so it stays exact. Azure's
ordering rule is load-bearing: *update the store, then invalidate* (invalidate-first
leaves a window for a concurrent reader to refill the stale value) — the #4/#5 bug.

**ORM second-level caches** solve exactly our problem:
- **Hibernate L2 query cache** invalidates by **timestamp comparison**, not eviction:
  each cached result is tagged with its load time; an `UpdateTimestampsCache` records the
  last-write time per table; a read discards its entry if any source table changed after
  the entry's load time ([Hibernate User Guide — Caching](https://docs.hibernate.org/stable/orm/userguide/html_single/Hibernate_User_Guide.html);
  [Mihalcea, *How Hibernate query cache works*](https://vladmihalcea.com/how-does-hibernate-query-cache-work/)).
  This is a **version/sequence stamp** — the spine we adopt for lazy views (§4).
- **Rails ActiveRecord query cache** is a **per-request/per-transaction lookaside**,
  cleared on any write, destroyed at request end — *"query caches are created at the start
  of an action and destroyed at the end"* ([Rails Guides — Caching](https://guides.rubyonrails.org/caching_with_rails.html)).
  It is stale-free **by construction** because it never crosses a transaction. That is
  precisely what `Tx.memo` is, and why memo needs no invalidation machinery.
- **Django `cacheops`** does signal-based granular invalidation ([README](https://github.com/Suor/django-cacheops)).

**Invalidation at scale**: TTL/expiry; explicit event-driven (the cache-aside write path,
with Azure's ordering caveat); **generational/versioned keys** (Rails 5.2 recyclable
`cache_version` — a stale entry is never *read* because the version won't match:
[Rails 5.2 notes](https://guides.rubyonrails.org/5_2_release_notes.html)); **CDC/log-driven**
(Debezium binlog tailing invalidating a JPA L2 cache: [debezium cache-invalidation example](https://github.com/debezium/debezium-examples/tree/main/cache-invalidation)).

**Systems / academic prior art closest to "derive from a log, invalidate on write":**
- **Scaling Memcache at Facebook** (NSDI '13) — the **stale-set race** and its **lease**
  (CAS-token) fix are named and defined here: *"a stale set occurs when a web server sets
  a value in memcache that does not reflect the latest value… concurrent updates get
  reordered."* Leases bind a 64-bit token to a key; a `delete` invalidates outstanding
  tokens so a slow reader's stale `set` is rejected ([paper PDF](https://www.usenix.org/system/files/conference/nsdi13/nsdi13-final170_update.pdf)).
  Our #4/#5 bug **is** the stale set; our fix is the sequence-stamp variant of the lease.
- **TAO** (ATC '13) — read-through, write-through cache over MySQL; centralizes
  invalidation so no client races a derive-and-write-back ([paper PDF](https://www.usenix.org/system/files/conference/atc13/atc13-bronson.pdf)).
- **Noria** (OSDI '18) — partially-stateful dataflow / incremental view maintenance; the
  general framing of "cache a value derived from base tables and maintain it as they
  change" ([paper PDF](https://www.usenix.org/system/files/osdi18-gjengset.pdf)); IVM
  survey: Gupta & Mumick 1995.
- **Postgres materialized views** — the in-DB analog; stale-until-`REFRESH`, full recompute
  ([REFRESH MATERIALIZED VIEW](https://www.postgresql.org/docs/current/sql-refreshmaterializedview.html)).

**Lessons applied:** (a) never write back a value derived from a snapshot without proving
the snapshot is still current — via a lock, a **version/sequence compare** (Hibernate,
Rails cache_version), or a CAS token (memcache lease); (b) a **per-transaction** lookaside
is stale-free by construction and needs no invalidation (Rails query cache = `Tx.memo`).

## 3. The unifying concept: three categories, one crisp criterion

> **projection** = in-memory state whose ground truth is a **table** (a materialized view).
> **tracker**    = in-memory state that **is** its own ground truth (no table backs it).
> **memo**       = **transaction-lifetime** state that may include **uncommitted** rows.

By this criterion:
- #2 #3 #4 #5 are **projections** — materialized views over controller tables. They
  unify.
- `Tx.memo` (#1) is a **memo** and stays out, *honestly*: the federation changelog gate
  must reflect `insert_received_handle`'s **not-yet-committed** row to a later write in the
  same transaction (`writes.py:377`). No committed-state view can do that. Memo is the one
  category that is *not* a projection and cannot be. (Optionally type-key it —
  `tx.memo[_ChangelogGate]` — for ergonomic parity; no shared code.)
- `WorkerHealthTracker` is a **tracker** and stays out: no table stores liveness; the
  tracker's heartbeat/failure state *is* the ground truth, and per-backend ownership is
  semantic (each backend reaps its own fleet).

## 4. Target design

### `Projection` base (the concept's definition point)

```python
class Projection(ABC):
    """In-memory materialized view over controller DB tables.

    owns:    sole writer of these tables; mutating methods issue the SQL and post
             the in-memory image via post-commit hooks (eager write-through).
    watches: written elsewhere; every declared writer must invalidate this
             projection, which recomputes lazily on read-miss (cache-aside).
    """
    owns: ClassVar[tuple[Table, ...]] = ()
    watches: ClassVar[tuple[Table, ...]] = ()

    def __init__(self, db: ControllerDB) -> None:
        self._db = db
        db.projections.register(self)   # (rename of db.caches; see below)
        self.rehydrate()
        db.register_reopen_hook(self.rehydrate)

    @abstractmethod
    def rehydrate(self) -> None: ...    # eager: reload SQL; lazy: clear
```

The base kills the module-global `PROJECTIONS` list and the `projections/__init__`
import-order re-export hack (its only job was populating that global before `validate()`);
`validate(db.projections)` iterates the registry instead. It also makes forgetting the
reopen hook impossible.

**owns/watches is a single axis, not a 2×2**: among all four projections, `owns` ⟹ eager
write-through and `watches` ⟹ lazy invalidate-and-recompute — there is no eager-watcher
and no lazy-owner. So the declaration *is* the maintenance strategy; no separate strategy
enum. (Eviction policy — AttemptCounts' clear-at-100k vs. RunTemplates' LRU-4096 — is a
per-view knob, orthogonal to the concept.)

### Reach: deref, never store or thread

**One rule: reach a projection through the `Tx` or `ControllerDB` handle you already hold,
per use — never store a reference on an object, never accept one in a signature.**
Transaction code: `cur.projections[P]`. Cursor-less collaborators (endpoint_service,
dashboard, proxy) already hold `db`: `db.projections[P]`. The registry lives on the DB and
is *mirrored* onto each `Tx`, so both work. Per-use deref (not resolve-once-at-init) is
self-healing: `register` supersedes on re-registration, so a stored ref could go stale
against the canonical instance. `CacheRegistry` gains `__iter__`; `db.caches`→`db.projections`.

### Validator: owned-table check over the registry

`validate(caches)` runs in `Controller.__init__` right after the projections are built. It
iterates the registry (no module-global `PROJECTIONS`), builds `{table: owner}` from each
projection's `owns`, and raises if any `@writes_to`/`cascades_into` writer of an owned table
is not a method of the owner. That enforces the sole-writer invariant for the two `owns`
tables (`endpoints`, `worker_attributes`) — with `WorkerAttrsProjection.set` now issuing the
`worker_attributes` SQL, no external writer touches it.

**Rejected: an `invalidates=` declaration + commit-time obligation-settle for watched tables.**
It was the plan, but it does not fit this codebase and its failure mode is disproportionate:

- Invalidation is correctly *localized to chokepoints*, not co-located with writes.
  `task_attempts` is written from many low-level `@writes_to` helpers, but the
  `AttemptCountsProjection` invalidation is batched at `_flush_attempts` / the federation
  mirror / `purge_job`. A per-writer obligation would force every low-level writer to
  invalidate (or over-invalidate), fighting the batched design.
- A hard raise at commit on a missed invalidation would turn *display-cache staleness* (a
  stale dashboard count until the next write — the worst case for these two lazy views) into
  a *controller crash*. That trade is wrong for a memo whose ground truth is always one
  re-derivation away.

The realistic drift (a new watched-table write path that forgets to invalidate) is covered
by the `LazyFillGuard` (no torn/stale-set value can be stored) plus the chokepoint locality
and code review. The watched-table invalidation invariant is documented on each lazy
projection rather than machine-enforced.

### Lazy-fill guard: sequence-stamp, not per-fill CAS

The stale-set fix for lazy views (#4, #5), shared as a ~20-line `LazyFillGuard`:

- `ControllerDB.commit_seq`: a counter incremented post-commit **under the write lock,
  before invalidation hooks run**.
- every `Tx` samples `tx.seq = db.commit_seq` **at mint, before BEGIN** (so `tx.seq` ≤ the
  snapshot's true state — conservative in the safe direction). One integer, set in `db.py`;
  no call site sees it.
- a watcher's post-commit invalidation records `inval_seq[key] = db.commit_seq`.
- **fill rule**: store `filled[key]` iff `tx.seq >= max(floor, inval_seq.get(key, 0))`;
  otherwise serve the value but do not cache it. `clear()` sets `floor = db.commit_seq`
  and empties both maps.

Correct because if `tx.seq ≥ inval_seq[key]` the invalidating commit was visible at sample
time, hence to the later BEGIN, hence already in the computed value. This fixes the two
holes in the naive fix — a single global generation *starves* the memo under steady
commits, and a `clear()` (checkpoint reopen) racing an in-flight fill can store counts from
the **old DB file** — and needs **no second snapshot** (recomputes from the caller's `tx`,
which also keeps dashboard counts consistent with the page being rendered), avoiding a
nested-pool-checkout hazard on the dispatch hot path.

### The instances

| Projection | Declares | Strategy | Change |
|---|---|---|---|
| `EndpointsProjection` | `owns=(endpoints_table,)` | eager write-through | internals unchanged; delete ~21 threaded params; reach via registry |
| `WorkerAttrsProjection` | `owns=(worker_attributes_table,)` | eager write-through | **de-scoped to one global instance**; `set()` **absorbs** the delete+insert SQL from `ops/worker.py:119` (closes the validator blind spot); drop `owns_scale_group` (filter at read time); drop routing+assert (`service.py:2050`), the `TaskBackend.worker_attrs` property, and k8s's permanently-None field |
| `AttemptCountsProjection` | `watches=(task_attempts_table,)` | lazy, seq-guarded | keep clear-at-100k; add `invalidate_all`; adopt `LazyFillGuard` |
| `RunTemplatesProjection` (was `RunTemplateCache`) | `watches=(job_config_table, job_workdir_files_table)` | lazy, seq-guarded, LRU-4096 | replace the eager pre-commit pop with a **post-commit keyed invalidation** (fixes the same-name-replacement race); job purge invalidates too (pruned-id reuse); delete ~10 threaded params |

### Deleted

Module-global `PROJECTIONS` + the `projections/__init__` re-export hack + the
`controller.py:442` ordering comment; per-backend attrs instances and their process-global
`PROJECTIONS` leak; the eager pre-commit LRU pop; `BackendRuntime.endpoints` /
`.run_template_cache`; `TaskBackend.worker_attrs`; `writes.py`'s projection import (a
projection import in the raw-writes layer — a layering smell). Stale docstrings referencing
the nonexistent `writes.workers.replace_attributes` (`worker_attrs.py:13,131`).

### Out of scope, by the §3 criterion

`Tx.memo` (sees uncommitted state), `WorkerHealthTracker` (own ground truth),
`system_endpoints` (not table-derived).

## 5. Sequencing (one campaign; each step leaves green tests)

1. `Projection` base + registry-based `validate()` + `invalidates=`/settle machinery +
   `commit_seq`/`LazyFillGuard`; migrate `EndpointsProjection` and `AttemptCountsProjection`
   declarations onto it (AttemptCounts adopts the guard here — **fixes bug #4**).
2. `WorkerAttrsProjection`: absorb the SQL, de-scope to global, read-time filtering, delete
   the routing / protocol-property / k8s-None surface.
3. Endpoints reach: delete the ~21 threaded params; all reach via the registry.
4. `RunTemplateCache` → `RunTemplatesProjection`: convert, replace the pre-commit pop with
   post-commit invalidation (**fixes bug #5**), delete the ~10 threaded params.

Mechanical call-site sweeps (the ~21 `endpoints=` deletions, ~24 test files) delegated to
subagents; the base class, validator, and guard are hand-implemented.
