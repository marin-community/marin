# Iris Controller → SQLAlchemy Core: Staged Task Plan

**Date:** 2026-05-11
**Author:** russell.power@gmail.com
**Companion to:** `20260511_iris_store_view_refactor_v2.md`
**Branch:** `iris-sql-store-view`
**Shape:** one PR, ~14 commits (= 14 spirals). Each commit is independently
testable, builds cleanly, passes `./infra/pre-commit.py --all-files --fix`
and `uv run pytest lib/iris/tests/`, and is reviewed before the next is
attempted. Stages do not move on to the next until the prior commit lands.

---

## Current-state inventory (grounded against `iris-sql-store-view` HEAD)

- **Tables (17):** `schema_migrations`, `meta`, `users`, `jobs`, `job_config`,
  `job_workdir_files`, `tasks`, `task_attempts`, `workers`,
  `worker_attributes`, `endpoints`, `scaling_groups`, `slices`,
  `reservation_claims`, `auth.api_keys`, `user_budgets`,
  `auth.controller_secrets`. The auth-attached DB stays attached at startup
  via `ATTACH DATABASE`; SA must reproduce that.
- **Projection constants (12) at `schema.py:1411–1642`:** `JOB_ROW_PROJECTION`,
  `JOB_SCHEDULING_PROJECTION`, `WORKER_ROW_PROJECTION`, `TASK_ROW_PROJECTION`,
  `JOB_DETAIL_PROJECTION`, `JOB_RESERVATION_PROJECTION`,
  `TASK_DETAIL_PROJECTION`, `WORKER_DETAIL_PROJECTION`, `ATTEMPT_PROJECTION`,
  `ENDPOINT_PROJECTION`, `API_KEY_PROJECTION`, `USER_BUDGET_PROJECTION`.
  All disappear in stage 12 (replaced by inline `select(...)` or named
  `select` constants in `reads/<area>.py`).
- **Store classes (7) in `stores.py`:** `EndpointStore`, `JobStore`,
  `TaskStore`, `TaskAttemptStore`, `WorkerStore`, `ReservationStore`,
  `ControllerStore` (aggregator). All gone after stage 12.
- **Write-through caches (2):** `EndpointStore` triple-dict
  (`_by_id`/`_by_name`/`_by_task`) and `ControllerDB._attr_cache`. Both
  populate via post-commit hooks fired under the write lock.
- **Hook call sites:** `stores.py` (6), `transitions.py` (4). All are
  `cur.on_commit(...)` registrations for the two caches; the refactor must
  preserve atomicity (lock held across hook fire) byte-for-byte.
- **Hot paths to gate on perf:** `_jobs_with_reservations`
  (`controller.py:418`, 0.019 ms), `resource_usage_by_worker`
  (`stores.py:1624`, 6.5 ms), `reconcile_rows_for_workers`
  (`stores.py:1679`, 6.3 ms), `get_detail` (`stores.py:700`, 0.05 ms).

---

## Stage-by-stage plan

Every stage is a single commit. The commit subject is given in
backticks; the bullets after each commit list what lands and how it's
verified.

### Stage 1 — `[iris] add sqlalchemy>=2.0 dep + db_v2 skeleton`

**Goal:** Get the dep in `pyproject.toml`, scaffold the new module tree,
and prove SA imports correctly under `uv run pyrefly`. No behavior change.

- Add `sqlalchemy>=2.0` to `lib/iris/pyproject.toml`.
- Create empty modules with minimal stubs so the import graph compiles:
  - `controller/schema_v2.py` (just `metadata = MetaData()`)
  - `controller/db_v2.py` (just an empty `class Tx`)
  - `controller/reads/__init__.py`, `controller/writes/__init__.py`,
    `controller/projections/__init__.py`
- Add `tests/cluster/controller/test_db_v2_smoke.py` — imports succeed, SA
  version pin matches.

**Acceptance:**
- `./infra/pre-commit.py --all-files --fix` passes.
- `uv run pyrefly` passes (new modules type-check; nothing else touched).
- `uv run pytest lib/iris/tests/` passes (full suite must remain green —
  no behavior changed).

---

### Stage 2 — `[iris] schema_v2: SA Core Tables + typedecorators`

**Goal:** Translate all 17 tables to SA Core `Table` objects with
`TypeDecorator`s for `JobName`, `WorkerId`, `Timestamp`, and JSON-string
columns. Includes a schema-equivalence test.

- In `schema_v2.py`:
  - `JobNameType`, `WorkerIdType`, `TimestampMsType`,
    `BoolIntType` TypeDecorators (each ~10 lines, `cache_ok=True`).
  - All 17 SA Core `Table(...)` declarations matching today's columns,
    constraints, FKs (`ondelete="CASCADE"` where today's DDL has it),
    and indexes (including `Index(..., sqlite_where=text("..."))` for the
    partial indexes from migrations 0010, 0028, 0045).
  - `auth.api_keys` and `auth.controller_secrets` declared on a separate
    `auth_metadata = MetaData()` since they live in an attached DB.
- Add `tests/cluster/controller/test_schema_v2_equivalence.py`:
  - Spin up a fresh `ControllerDB` (hand-rolled migrations).
  - Pull `sqlite_master.sql` rows for every table and index.
  - Render `metadata.create_all()` against an in-memory SQLite and pull
    the same rows.
  - Normalize whitespace; assert the two sets are equivalent. Includes
    partial-index `WHERE` clause comparison.

**Acceptance:**
- `test_schema_v2_equivalence.py` passes — proves SA's DDL matches what
  migrations produce.
- Full pytest suite still green; `schema_v2.py` is not yet wired into
  `ControllerDB`.

---

### Stage 3 — `[iris] schema_v2: CachedProto TypeDecorator`

**Goal:** Port `ProtoCache` into a `CachedProto(message_cls)` TypeDecorator
on `LargeBinary`. Bytes-keyed LRU; 8192 entries; 25% eviction batches —
identical policy to `schema.py:34–66`.

- Add `CachedProto` in `schema_v2.py`. Per the v2 doc §4.3, start with a
  *global* class-level dict (`_global_cache`, `_global_lock`) — exact
  behavioral match to today's `proto_cache` singleton.
- Wire it onto the `job_config.config_proto`-equivalent columns
  (`reservation_json`, `entrypoint_json`, etc. that today have
  `cached=True`) by switching those columns from `String` to
  `CachedProto(...)` — but only in `schema_v2.py`, no call sites yet.
- Tests in `test_schema_v2_equivalence.py` (extended):
  - Two rows with identical blob bytes return the *same* Python proto
    instance.
  - Cache fills to 8192, evicts in 2048-entry batches, doesn't unbound.
  - `process_bind_param` round-trips a proto.

**Acceptance:**
- New tests pass. `CachedProto` not yet used at any read site — call
  sites still go through today's `Projection.decode`.

---

### Stage 4 — `[iris] db_v2: engine + Tx + write/read tx contexts`

**Goal:** Land the engine factory, the `Tx` wrapper, and the two
context managers (`write_transaction`, `read_snapshot`) with the exact
atomicity contract today's `ControllerDB.transaction()` carries.

- In `db_v2.py`:
  - `_make_engine(db_path, auth_db_path)` — creates the SA `Engine` with
    `QueuePool(pool_size=32, max_overflow=4)`,
    `connect_args={"check_same_thread": False, "timeout": 5.0}`,
    `isolation_level=None` (so SA does not autoBEGIN; we BEGIN
    explicitly to match today's `BEGIN IMMEDIATE`).
  - `event.listens_for(engine, "connect")` installs the four PRAGMAs
    from `ControllerDB._configure` plus `ATTACH DATABASE auth`.
  - `class Tx`: `conn`, `execute`, `executemany`, `register`,
    `_fire_hooks`.
  - `@contextmanager write_transaction(engine, write_lock)` — acquires
    the lock, opens `engine.connect()`, emits `BEGIN IMMEDIATE`,
    yields `Tx`, commits or rolls back, then fires hooks **while still
    holding the lock**.
  - `@contextmanager read_snapshot(engine)` — pool-connection,
    `PRAGMA query_only=ON`, `BEGIN`, yields `Tx`, `ROLLBACK`.
- New tests `tests/cluster/controller/test_tx.py`:
  - Hook fires once on commit; never on rollback.
  - Lock is held across `_fire_hooks` (verified by spawning a second
    thread that tries to acquire the lock and blocks).
  - Concurrent readers don't block writers (snapshot doesn't take the
    write lock).
  - PRAGMAs are set at connection time and persist across check-outs.

**Acceptance:**
- Atomicity tests pass under thread-race conditions.
- `db_v2.py` still independent of `db.py` — no production code touches it.

---

### Stage 5 — `[iris] reads/scheduler: port _jobs_with_reservations (perf canary)`

**Goal:** First real port. Validates the SA Core + `CachedProto`
read-path hits the perf budget on the hot tick query.

- Add `controller/reads/scheduler.py`:
  - `JOBS_WITH_RESERVATIONS = select(...)` constant — SA Core idiom.
  - `def jobs_with_reservations(tx, states) -> list[JobReservationRow]:`
    helper returning Protocol-typed rows (per v2 §4.4 pattern 2).
- Switch the one call site in `controller.py:418` from
  `JOB_RESERVATION_PROJECTION` to the new helper, opening a `read_snapshot`
  off a `ControllerDB` augmented to expose the new SA engine alongside
  the legacy `_conn`. Both data layers coexist in this stage.
- Add `tests/cluster/controller/test_perf_baselines.py` — pytest-benchmark
  style. Establish the table for all future commits:
  - `_jobs_with_reservations` (200 reservations): gate ≤ 0.025 ms.

**Acceptance:**
- Bench gate passes.
- Integration tests (`test_scheduler.py`, `test_reservation.py`) green.
- This stage proves the design's perf assumption. If it fails, the
  proposal is wrong and we stop here.

---

### Stage 6 — `[iris] projections/endpoints: EndpointStore → EndpointsProjection (linchpin)`

**Goal:** Highest-risk commit. Prove that
`engine.begin()` + `Tx.register` + write-lock-across-hooks preserves the
`EndpointStore` atomicity invariant.

- Add `controller/projections/endpoints.py`:
  - `class EndpointsProjection`: owns `_by_id`/`_by_name`/`_by_task`
    dicts and `_lock`. `sources = (endpoints_table,)`.
  - Mutating methods: `add`, `remove`, `remove_by_task`,
    `remove_by_job_ids` — each issues SA `insert/delete` via `tx.execute`
    and `tx.register(lambda: ...)`.
  - Read methods: `by_id`, `resolve`, `all`, `query(EndpointQuery)`. No
    `tx` parameter (per the critique doc's resolution of the §4.6
    debate — `tx` for symmetry is rejected).
  - `rehydrate(tx)` — populates dicts from `select(endpoints)` inside
    `read_snapshot`. Called at startup and after `replace_from`.
- Switch all call sites (`service.py:1745`, transitions, controller)
  from `self._store.endpoints` to the new projection.
- Delete `EndpointStore` from `stores.py` (the only Store deleted in
  this stage).
- Rewrite `tests/cluster/controller/test_endpoint_store.py` to drive
  `EndpointsProjection`. All existing assertions stay; new ones added:
  - **Atomicity:** before write-tx exits, no thread sees the new dict
    entry; after exit, all threads see it.
  - **Rollback safety:** raise mid-tx, assert dict unchanged.
  - **Restore correctness:** `backup_to` + modify + `replace_from` →
    dict reflects snapshot state.
  - **Concurrency:** 2 writers + 1 reader for 5s — no `KeyError`, no
    torn reads, no deadlock.

**Acceptance:**
- All atomicity tests pass. If any fail, the design is wrong and we
  back out and rethink before continuing.

---

### Stage 7 — `[iris] projections/worker_attrs: _attr_cache → WorkerAttrsProjection`

**Goal:** Replicate stage 6 for the second write-through cache. Smaller
than stage 6 (only `db.py:329–379` + 3 `transitions.py` call sites), but
brings the **cascade-into-projection** hazard surfaced in the addendum.

- Add `controller/projections/worker_attrs.py`:
  - `class WorkerAttrsProjection` with same shape as `EndpointsProjection`.
    `sources = (worker_attributes_table,)`.
  - `set(tx, worker_id, attrs)`, `remove_for_worker(tx, worker_id)`,
    `invalidate_for_worker(tx, worker_id)` (dict-only update for FK
    cascade), `get(worker_id)`, `all()`.
- Delete `_attr_cache`, `_attr_cache_lock`, `_populate_attr_cache`,
  `get_worker_attributes`, `set_worker_attributes`,
  `remove_worker_from_attr_cache` from `db.py`. Caller in
  `db.py:929` (`healthy_active_workers_with_attributes`) and the
  `transitions.py` registrations move to the projection.
- Tests: write-through atomicity, rollback, restore. Plus an explicit
  test that deleting a row from `workers` (which FK-cascades into
  `worker_attributes`) updates the projection — driving the need for
  the explicit invalidation hook on the cascading call site in stage 13.

**Acceptance:**
- Full atomicity suite passes for `WorkerAttrsProjection`.
- The cascade test fails *intentionally* before stage 13's invalidation
  is wired in. Mark it `xfail` here so the suite is green; stage 13
  removes the marker.

---

### Stage 8 — `[iris] reads/jobs: get_detail, get_config, list_descendants, get_priority_bands`

**Goal:** Largest-by-LOC read port. All `JobStore` read methods become
either inline `tx.execute(select(...))` at the call site or named
selects + helpers in `reads/jobs.py`.

- Hot/shared reads land as named constants in `reads/jobs.py`:
  - `JOB_DETAIL_QUERY`, `JOB_CONFIG_QUERY`, `JOB_PRIORITY_BANDS_CTE`
    (recursive CTE via SA `select(...).cte(recursive=True)`),
    `list_descendants(tx, ...)`, `list_subtree(tx, ...)`,
    `has_unfinished_worker_attempts(tx, ...)`.
- One-off reads (`get_state`, `get_root_submitted_at_ms`,
  `get_preemption_info`, `find_prunable`) become inline `select(...)`
  at the original `JobStore` method call sites. Keep `JobStore` as a
  thin shim that just calls into the new helpers — full deletion
  happens in stage 12 to keep this commit focused.
- Add `Protocol` types in `reads/jobs.py` (per v2 §4.4 pattern 2) for
  the row shapes that flow into RPC responses.
- Perf gate: `get_detail` ≤ 0.07 ms (today 0.05 ms).

**Acceptance:**
- All `test_transitions.py`, `test_direct_controller.py`,
  `test_service.py` tests pass unchanged.
- `JOB_DETAIL_PROJECTION` and `JOB_RESERVATION_PROJECTION` still exist
  in `schema.py` but have one fewer caller each.

---

### Stage 9 — `[iris] reads/scheduler+tasks: task/attempt reads + bulk paths`

**Goal:** Port the heavy scheduler-tick read paths
(`resource_usage_by_worker`, `reconcile_rows_for_workers`,
`bulk_get_for_updates`, `list_active`).

- `reads/scheduler.py`: `RESOURCE_USAGE_BY_WORKER`,
  `RECONCILE_ROWS_FOR_WORKERS` named selects. Helpers handle the
  chunking and `IN (VALUES …)` patterns from `stores.py:1580–1613`
  (`bulk_get_for_updates`).
- `reads/tasks.py`: `TASK_DETAIL_QUERY`, `BULK_TASK_DETAIL` (with
  chunking), `STATE_COUNTS_FOR_JOB` (aggregate with `func.count` +
  `group_by`), `FIRST_ERROR_FOR_JOB`, `list_active(tx, ...)`,
  `list_pending_for_direct_provider(tx, ...)`.
- Perf gates:
  - `resource_usage_by_worker` (24k jobs, 1k live) ≤ 8 ms.
  - `reconcile_rows_for_workers` (200 workers) ≤ 8 ms.
- If either gate fails, abort and rethink before committing.

**Acceptance:**
- All scheduler / autoscaler integration tests pass.
- Perf benches pass gates.

---

### Stage 10 — `[iris] reads/workers+dashboard: worker reads, reservation reads, dashboard composites`

**Goal:** Port the remaining reads. Dashboard `list_jobs`
(`service.py:656–732`) and its sibling helpers
(`_task_summaries_for_jobs`, `_live_user_stats`) get a home in
`reads/dashboard.py`, organized **by use-case** (per the critique doc's
fix — not `reads/jobs.py` for a dashboard query).

- `reads/workers.py`: worker detail/list reads,
  `WORKER_ROW_QUERY` for `healthy_active_workers_with_attributes`.
- `reads/reservations.py`: `RESERVATION_CLAIMS_QUERY`, lookups by job
  and slice.
- `reads/dashboard.py`: `list_jobs(tx, query)` (dynamic ORDER BY/LIMIT/
  OFFSET — caller composes via SA `select`, no kwarg explosion),
  `task_summaries_for_jobs(tx, job_ids)` (aggregate via SA
  `func.count`/`func.sum` + `group_by`).
- `reads/budgets.py`: user-budget reads currently inline in `db.py`
  (`set_user_budget`, `get_user_budget`, `list_user_budgets`).

**Acceptance:**
- `test_dashboard.py`, `test_main_endpoints.py`, `test_budgets.py`
  pass.

---

### Stage 11 — `[iris] writes: module-level @writes_to functions`

**Goal:** Move every write off the Store classes into module-level
functions under `writes/<entity>.py`, decorated with `@writes_to(...)`.

- New `controller/writes/__init__.py` defining `@writes_to(*tables,
  cascades_into=())` — pure metadata (records `fn.writes_to`,
  `fn.cascades_into`, appends to a module-level registry).
- New files (each lands the write methods currently on the
  corresponding Store):
  - `writes/jobs.py` — `insert_job`, `insert_job_config`,
    `update_state_if_not_terminal`, `bulk_update_state`,
    `mark_running_if_pending`, `apply_recomputed_state`,
    `delete_job`, `insert_workdir_files`, `ensure_user`,
    `reserve_priority_insertion_base`.
  - `writes/tasks.py` — `insert_task`, `mark_assigned`, `assign`,
    `apply_state_update`, `mark_terminal`, `bulk_kill_non_terminal`,
    `update_container_id`.
  - `writes/task_attempts.py` — `insert_attempt`,
    `apply_attempt_state`, `bulk_apply_attempt_state`,
    `mark_finished`, `apply_update`.
  - `writes/workers.py` — `upsert_worker`, `remove_worker`
    (with `cascades_into=(worker_attributes, task_attempts)` — see
    stage 13).
  - `writes/reservations.py` — `replace_claims`, `next_submission_ms`.
- Switch all `self._store.<entity>.<method>(cur, ...)` call sites in
  `transitions.py`, `controller.py`, `service.py`, `scheduler.py` to
  `writes.<entity>.<method>(tx, ...)`.
- Tests: `test_transitions.py`, `test_direct_controller.py`,
  `test_5470_preemption_reassignment.py` exercise the entire write
  graph. Must pass unchanged.

**Acceptance:**
- Stores remaining in `stores.py` after this commit: empty shells
  (all read paths use `reads/*`, all writes use `writes/*`). The
  `ControllerStore` aggregator is gone.

---

### Stage 12 — `[iris] startup: @writes_to owned-table check + cascade hook`

**Goal:** Wire in the §4.8 startup-time invariant check; resolve the
stage-7 `xfail` cascade test.

- In `controller/projections/__init__.py`:
  - `PROJECTIONS: list[Projection]` module-level registry.
  - `assert_owned_tables_not_externally_written()` per v2 §4.8.
- In `ControllerDB.__init__`, after the engine is up and all `writes/`
  + `projections/` modules are imported, call the check. Hard-fail
  startup on violation.
- In `writes/workers.py`'s `remove_worker`, call
  `projections.worker_attrs.invalidate_for_worker(tx, worker_id)` to
  mirror the FK cascade into the projection. Remove the `xfail` marker
  from the stage-7 cascade test.
- Add `tests/cluster/controller/test_writes_to_check.py`:
  - A write function decorated `@writes_to(endpoints)` outside
    `EndpointsProjection` causes startup to raise `ConfigurationError`.
  - `cascades_into=(worker_attributes,)` is treated as a write.

**Acceptance:**
- Startup check fires correctly on injected violations; quiet on the
  real codebase.

---

### Stage 13 — `[iris] delete legacy: stores.py, old Projection/ProtoCache, _attr_cache`

**Goal:** The big delete. Drops ~1900 LOC (the original v2 doc target).

- Delete `stores.py` entirely (all Store classes now empty).
- Delete from `schema.py`:
  - `ProtoCache` class and the `proto_cache` singleton.
  - `Column` field `cached` / `expensive` (now in `CachedProto`).
  - `Projection` class, `adhoc_projection`, `_make_row_class`,
    `_validate_row_cls`, `ExtraField`.
  - All 12 `*_PROJECTION` constants.
  - `Table` class and its DDL machinery — superseded by SA's
    `metadata.create_all` (but migrations remain authoritative for
    on-disk schema; SA tables are for SELECT generation only).
- Delete `TransactionCursor`, `QuerySnapshot`, `Row` from `db.py`.
  Delete `_conn`, the manual write lock plumbing, the manual read
  pool, and the `_configure` / `_init_read_pool` methods.
  `ControllerDB` becomes a thin wrapper: engine, projections,
  migration runner, backup/restore.
- Delete `decode_worker_id`, `decode_timestamp_ms`, the column-decoder
  helpers from `schema.py` (logic now in TypeDecorators).
- Keep the migrations directory and the migration runner intact —
  hand-rolled migrations stay per v2 §3 non-goals.

**Acceptance:**
- Full pytest suite passes.
- `wc -l` confirms ≥ -1500 net LOC delta (target -1900).
- Grep confirms no remaining references to `Projection`, `ProtoCache`,
  `TransactionCursor`, `_attr_cache`, `EndpointStore`, etc.

---

### Stage 14 — `[iris] rename: drop _v2 suffix; doc the convention in AGENTS.md`

**Goal:** Cosmetic + doc cleanup. Last commit before merge.

- Rename `schema_v2.py` → `schema.py` (the old file is now mostly
  gone; replace what remains). Same for `db_v2.py` → `db.py`.
- Update all imports.
- Add the §11 convention block from the v2 doc to
  `lib/iris/AGENTS.md` (under a new "Data layer" section): hot/shared
  → `reads/`; one-off → inline; writes module-level with `@writes_to`;
  projections own their tables.
- Update any stale references in `lib/iris/OPS.md` and the docs that
  point at `stores.py`.

**Acceptance:**
- Full suite green.
- `git diff main` shows the net -1500 LOC delta.

---

## End-to-end smoke before merge

Per v2 §6.3, before clicking merge on the assembled PR:

1. `scripts/iris/dev_tpu.py` to spin up a real controller; submit a
   small job batch; verify state transitions
   submitted → scheduled → running → terminal.
2. Restart the controller from a backup; verify both Projections
   rehydrate and the dashboard renders.
3. 10-minute soak; confirm scheduler-tick latency p95 stays inside the
   pre-refactor envelope (`resource_usage_by_worker` < 10 ms p95;
   `reconcile_rows_for_workers` < 10 ms p95).
4. Wall-clock CPU profile via the `agent-profiling` skill; expect a
   small win from removed Python indirection plus SA's compilation
   cache.

If any of these fail, the PR does not merge.

---

## Per-stage gate checklist (applies to every commit)

1. `./infra/pre-commit.py --all-files --fix` passes.
2. `uv run pyrefly` passes.
3. `uv run pytest lib/iris/tests/` passes.
4. Perf benches (stages 5, 6, 7, 9) inside their gate.
5. Commit revertable in isolation.
6. **User review on the actual diff before the next stage is started.**

---

## Risk-ordered stages

The three highest-risk stages (matching the v2 doc's "linchpin" framing):

- **Stage 5** — perf canary. If `_jobs_with_reservations` regresses,
  the SA Core perf assumption is wrong and the refactor is in trouble.
- **Stage 6** — `EndpointsProjection`. If atomicity slips here, the
  write-through model under SA's transaction context is broken.
- **Stage 9** — `resource_usage_by_worker` / `reconcile_rows_for_workers`.
  Heaviest hot paths; final perf validation.

Failing any of these gates is a stop-the-line event: back out the
commit, rethink, and either fix or abandon the refactor.

---

## Tracking

Each stage is a TaskCreate entry in the harness so progress survives
across sessions. The conventional flow:

1. Take the next pending task.
2. Implement the stage.
3. Run the gate checklist.
4. Open the commit; ping the user for review.
5. On approval, mark the task completed and proceed to the next one.
