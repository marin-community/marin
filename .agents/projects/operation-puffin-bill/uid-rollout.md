# UID Rollout — Operation Puffin Bill, Phase C

**Status:** Draft — 2026-05-19. Refines `spec.md` §4.1, §5.1, §5.7 into a concrete change.
**Owner:** Russell.
**Companion:** `spec.md` (canonical design), `worker-readiness.md` (gap audit this closes).

## 1. Shape of the work

**One PR. Two deploy steps.** The change is backward-compatible in both directions, so all of it — DB migration, controller, worker — lands in a single PR. The rollout is just:

1. Merge and deploy the controller. The migration runs on startup; the controller starts minting and emitting `attempt_uid`. Old workers ignore it.
2. Roll the workers. Each one picks up UID routing when it restarts onto the new binary.

No feature flag. No fleet-wide coordination. At every intermediate point — old workers + new controller, new workers + old controller, mixed fleet — the system is correct, because UID routing is additive and the controller keeps a composite-key fallback.

If you want a cleaner bisect story you *may* split it into two PRs (controller, then worker), but that is a reviewing convenience, not a correctness requirement.

This PR makes `attempt_uid` load-bearing for **worker routing**. It does **not** repoint the controller DB onto UIDs — that is a deliberate followup (§7) that makes `job_uid` exist and become the DB join key.

## 2. Why this is backward-compatible

The controller routes incoming observations by `attempt_uid` **when it is set**, and by `(task_id, attempt_id)` when it is not. Given that:

| Situation | Behavior |
|---|---|
| New controller, old worker | Controller emits UIDs; old worker ignores them and replies with empty `attempt_uid`; controller falls back to composite. Correct. |
| Old controller, new worker | Old controller never sends a UID; new worker sees empty `attempt_uid`, runs on composite. Correct. |
| New controller, mixed fleet | Each worker is independent — composite fallback covers the old ones, UID covers the new ones. Correct. |
| Worker downgrade at any time | Always safe — see above rows. Nothing about UID is load-bearing on a flag. |

The composite-key fallback in the controller can be deleted in a later cleanup once every worker is confirmed on the new binary, but leaving it costs nothing.

## 3. Identity model

This PR and the DB followup (§7) together move Iris onto UID identity. Splitting them by identifier:

| Identifier | This PR | After the followup |
|---|---|---|
| `job_id` (`JobName`) | `jobs` PK; reused on resubmit (prior row overwritten) | demoted to a non-unique indexed column — the human-facing name |
| `task_id` (`JobName`) | `tasks` PK | demoted to a display column |
| `attempt_id` (per-task counter) | part of `task_attempts` PK | display / ordering only |
| `attempt_uid` (16 hex, controller-minted) | **new** — `NOT NULL UNIQUE` column, load-bearing on the wire for worker routing | `task_attempts` PK |
| `job_uid` (controller-minted) | **not added** | **new** — `jobs` PK; the join key every other table FKs against |

`AttemptUid = NewType("AttemptUid", str)` already exists in `cluster/types.py:317`.

`job_uid` is deliberately **not** in this PR. Nothing reads it until the DB is repointed onto it (§7); a column with no reader is dead weight. This PR ships only `attempt_uid`, which has an immediate consumer — worker routing.

## 4. The change

### 4.1 DB migration `0027_attempt_uid` (runs on controller startup)

- `ALTER TABLE task_attempts ADD COLUMN attempt_uid TEXT` (nullable for the migration only).
- Backfill with `secrets.token_hex(8)`, 1k-row chunks (`UPDATE ... WHERE attempt_uid IS NULL LIMIT 1000` until zero rows update).
- Rebuild the table with the column promoted to `NOT NULL` (SQLite needs a table rebuild). Same migration — the backfill loop terminates only when no NULL rows remain, and the new INSERT path (§4.2) is in the same binary, so no row can land NULL afterward.
- `CREATE UNIQUE INDEX idx_task_attempts_uid ON task_attempts(attempt_uid)`.

### 4.2 Controller

- **Mint:** attempt placement generates `attempt_uid` — `secrets.token_hex(8)`, inside the same transaction as the row insert. The UNIQUE index turns the (astronomically unlikely) collision into a constraint error that retries with a fresh value.
- **Project:** `ReconcileRow` (`controller/reconcile.py:16`) gains `attempt_uid`. `reconcile_rows_for_workers` selects it.
- **Emit:** `controller/reconcile.py:_reconcile_worker` fills `DesiredAttempt.attempt_uid` at all five emit sites (currently hardcoded `""` at lines 91, 100, 109, 118, 127). The legacy translator (`worker_provider.py`) likewise.
- **`RunTaskRequest.attempt_uid`** — new field in `job.proto`. This is how the UID reaches the worker on the inline-spec dispatch *and* on the legacy `StartTasks` path, so the worker can stamp it on the container regardless of which wire delivered the task.
- **Route:** `transitions.py:_observations_to_updates` (line 2072) — when `obs.attempt_uid` is set, resolve it to the attempt row and route by it; otherwise fall back to the composite key (today the branch only logs — make it actually route).

### 4.3 Worker

- **`TaskAttempt` carries `attempt_uid`** alongside the `task_id` / `attempt_id` it already has. The attempt object is self-describing.
- **Task store becomes a `list[TaskAttempt]`** — replacing `self._tasks: dict[tuple[str, int], TaskAttempt]`. It cannot be keyed by UID: a worker booting during the rollover adopts containers created by a pre-upgrade worker that have no UID yet. A list needs no key. Two helpers over the O(10)-element list:
  ```python
  def task_by_uid(self, uid: AttemptUid) -> TaskAttempt | None: ...
  def task_by_composite(self, task_id: str, attempt_id: int) -> TaskAttempt | None: ...
  ```
  This also fixes the same-name-resubmit bug on the worker: today `submit_task` rejects a re-submitted `(task_id, attempt_id=0)` as a duplicate of a retained terminal attempt (`worker.py:706-728`). With UID identity, the new incarnation is a distinct `TaskAttempt`.
- **Delete `_spec_cache`.** It is write-only dead code today — `worker.py` writes it (line 1107) and evicts it (line 1052) but never reads it. The worker either holds a `TaskAttempt` (which already carries its `RunTaskRequest` via `TaskAttemptConfig`) or it does not (an inbound `run` intent carries the spec inline → create it, or does not → report `MISSING`). No third state. *(Independently shippable as a one-line-ish cleanup if you want it out of this PR.)*
- **Routing** (`_process_run_intent`, `_process_stop_intent`): delete the `del attempt_uid` lines; resolve via `task_by_uid`. In `_process_run_intent`, on a UID miss, fall back to `task_by_composite` once — this catches the rollover case of an adopted label-less attempt; on a hit, stamp the UID onto the `TaskAttempt` so later ticks resolve directly. Miss on both → create from inline spec, or `MISSING`.
- **`submit_task`** reads `RunTaskRequest.attempt_uid` into `TaskAttemptConfig`.
- **Container label** — `runtime/docker.py:_build_run_cmd` (near line 673) adds `--label iris.attempt_uid=<uid>`. `DiscoveredContainer` (`runtime/types.py`) gains an `attempt_uid` field read back from the label.
- **Observations** — `_build_observation` fills `AttemptObservation.attempt_uid` (composite fields stay populated too; cheap, and the controller fallback still wants them).
- **Adoption** — `adopt_running_containers`: if the container has an `iris.attempt_uid` label, the `TaskAttempt` gets that UID. If not (created by a pre-upgrade worker), it goes into the list with an empty UID; the first Reconcile tick composite-matches it and stamps the UID. One-time, bounded by the reconcile interval.

## 5. Container labels — the one Docker constraint

Docker labels are immutable after `docker create`; a worker cannot retroactively label a running container. Consequence:

- **New containers** carry `iris.attempt_uid` intrinsically.
- **Containers spanning the worker upgrade** have no label. They are adopted by composite key (the existing path, unchanged) and learn their UID from the first Reconcile tick. Composite is unique, so this is correct. No sidecar file, no `docker commit`, no out-of-band persistence — if a worker crashes in that <1 reconcile-interval window it just re-adopts by composite next start.

## 6. Out of scope for this PR

- **Deleting the legacy `StartTasks` / `StopTasks` / `PollTasks` / `UpdateTaskStatus` handlers and RPCs.** That is the tail of the *Reconcile RPC* migration, gated on `IRIS_RECONCILE_RPC_ENABLED`. A separate project; not UID work.

Repointing the controller DB onto UIDs is **not** out of scope in the sense of "skipped" — it is the planned followup, fully modeled in §7.

## 7. Followup PR — make UIDs the load-bearing DB identity

This PR fixes *worker* routing. It does **not** fix the controller DB. `jobs.job_id`, `tasks.task_id`, and `task_attempts(task_id, attempt_id)` are all `JobName`-derived primary keys (`schema.py:249, 340, 396`), and `JobName` is reused on resubmit. So the DB physically holds **exactly one row per name**: resubmitting a job whose name already exists CASCADE-overwrites the prior row (destroying its tasks and attempts) or is rejected outright. `existing_job_policy` is implemented today by PK collision, not by a real constraint.

Adding a passive `job_uid` column would not change this — there is still one row per name, so it has nothing to disambiguate. For the UID to be load-bearing it must be the **primary key** and the **FK target** every other table joins against. Hence `job_uid` is not in this PR at all; it is introduced here, where it does real work.

### 7.1 Target schema (`0028_uid_primary_keys`)

| Table | Today's PK | New PK | Notes |
|---|---|---|---|
| `jobs` | `job_id` (`JobName`) | `job_uid` | `job_id` demoted to a non-unique indexed column |
| `tasks` | `task_id` (`JobName`) | `(job_uid, task_index)` | the `(job_id, task_index)` UNIQUE constraint already exists (`schema.py:361`); `task_id` kept as a display column |
| `task_attempts` | `(task_id, attempt_id)` | `attempt_uid` | column added by this PR — promote to PK; carries `(job_uid, task_index)` for the task FK |

Repointed FKs — every `job_id` reference becomes `job_uid`:

- `jobs.parent_job_id` → `parent_job_uid` (self-ref); `root_job_id` → `root_job_uid`.
- `job_config.job_id`, `job_workdir_files.job_id`, `endpoints.job_id`, `reservation_claims.job_id` → `job_uid`.
- `endpoints.task_id` → `(job_uid, task_index)`.

`task_id` / `job_id` (`JobName`) survive as ordinary non-unique columns — they are the human-facing names the CLI and dashboards display, and the wire protocol's composite fallback still uses them. Only the **DB joins and PKs** move to UIDs.

### 7.2 The "one active job per name" constraint

A partial unique index:

```sql
CREATE UNIQUE INDEX idx_jobs_active_name ON jobs(job_id) WHERE finished_at_ms IS NULL;
```

`finished_at_ms IS NULL` is exactly "active" and is robust to `JobState` enum changes — no enum literals frozen into the DDL. It is the same "live" predicate the schema already uses for `idx_task_attempts_live_workerbound` (`schema.py:401`).

This makes resubmit semantics explicit instead of implicit in PK overwrite:

- **Prior job terminal** (`finished_at_ms` set): the index does not cover it; the new incarnation inserts cleanly and both rows coexist — history is preserved.
- **Prior job still active**: the INSERT hits the constraint. `existing_job_policy = FAIL` surfaces "already running"; `existing_job_policy = REPLACE` first transitions the old job to terminal (setting `finished_at_ms`, freeing the index) inside the same transaction, then inserts.

### 7.3 Effort

The migration DDL is mechanical — SQLite table rebuilds, one per repointed table. The real work is rewriting every controller query that joins or filters on `job_id` / `task_id` to use `job_uid` / `(job_uid, task_index)`. `schema.py` is SQLAlchemy Core, so this is a sweep of the data-access layer, not an ORM remap. Budget the followup around the query rewrites, not the schema change. `attempt_uid` shipped as `NOT NULL UNIQUE` by this PR is exactly the prerequisite — promoting it to PK in `0028` is a clean rebuild with no rework.

## 8. Testing

- **Migration:** on a prod-sized snapshot, the `0027` backfill leaves zero NULL `attempt_uid` rows and the NOT NULL rebuild succeeds.
- **Controller:** unit-test UID minting + collision retry; integration test that the controller emits a UID on every wire path and an unmodified worker ignores it cleanly.
- **Worker:** a same-name resubmit onto the same worker produces a distinct `TaskAttempt` in the list rather than hitting the duplicate-rejection path (the bug today). Adoption test: one container with the `iris.attempt_uid` label, one without (composite-match, UID learned on first reconcile).
- **Mixed fleet:** controller routes correctly when half the observations carry a UID and half don't.
- **Followup (`0028`):** resubmitting a name whose prior job is terminal yields two coexisting `jobs` rows; resubmitting while the prior job is active hits `idx_jobs_active_name` and is handled per `existing_job_policy`.

## 9. Operator-visible artifacts

- **Dashboard / `iris workers list`:** add an `attempt_uid` column to per-task panels (truncate to 8 chars in the UI) so operators can watch the fleet adopt UID routing during the rollout.
- **`docker inspect`:** UID visible in container labels on new-binary workers — free, delivered by §4.3's `iris.attempt_uid` label.

**Deferred to the §7 followup — log keys.** `task_log_key` currently returns a *hierarchical* key (`/user/job/0:<attempt_id>`); `build_log_source` relies on that hierarchy for prefix matching ("all logs for job X"). An opaque `attempt_uid` has no hierarchical relation to its job, so re-keying logs by UID would break prefix matching and needs its own design. It also delivers nothing until same-name jobs are *retained* — today a same-name resubmit destroys the prior job outright (§7), so there is no second job whose logs could collide. Re-key logs in the followup, alongside the DB repoint that makes retention real.
