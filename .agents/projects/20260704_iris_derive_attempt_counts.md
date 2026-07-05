# Derive task failure/preemption counts from attempts (#6946)

Status: complete. Branch `weaver/issue-6946`.

## Problem

`tasks.failure_count` / `tasks.preemption_count` are denormalized counters the
reconcile path increments in lockstep with attempt rows. They are a second
source of truth (drift-prone) and must be kept in sync on every read/write path
and mirrored over the federation sync wire. The attempt rows are already the
authoritative log, so the counters can be derived from them.

## Derivation (single source of truth)

Given a task's attempt rows (`task_attempts`), each carrying `state` and
`started_at_ms`:

- **failure_count** = number of attempts with `state == TASK_STATE_FAILED`.
  Application failures charge the failure budget regardless of phase (the
  reconcile kernel increments `failure_count` whenever `new_state == FAILED`,
  and `timeout_one` produces a FAILED attempt too).

- **preemption_count** = number of attempts in
  `{WORKER_FAILED, KILLED, PREEMPTED}` **with `started_at_ms IS NOT NULL`**
  (i.e. the attempt reached the executing phase: BUILDING/RUNNING both stamp
  `started_at_ms`). An ASSIGNED-phase worker/kill/preempt failure retries
  **without** charging the preemption budget (`resolve_task_failure_state`,
  `reconcile/task.py`), and such an attempt has `started_at_ms IS NULL`, so the
  phase predicate reproduces the increment semantics.

`started_at_ms IS NOT NULL` ⟺ the task was in `EXECUTING_TASK_STATES`
(`{BUILDING, RUNNING}`) when the attempt terminated, because BUILDING/RUNNING
are the only transitions that stamp the attempt's `started_at`. This is exactly
the condition `resolve_task_failure_state` gates the increment on.

### Intentional divergence (documented, permitted by the issue guidance)

Cancel/finalize (`_kill_non_terminal_tasks`) writes a KILLED attempt without
routing through `resolve_task_failure_state`, so today it does **not** bump
`preemption_count`. The derived count above **does** count an executing KILLED
attempt. Net effect: cancelling a task that was actively BUILDING/RUNNING adds
1 to that (now terminal, KILLED) task's `preemption_count` versus the old
lockstep counter. This is cosmetic (the job is being killed) and arguably more
truthful ("the run was preempted by the cancellation"). All non-cancel cases —
including a cancel of a previously-preempted PENDING task, whose prior PREEMPTED
attempt the cancel rewrites to KILLED — match the old counter exactly, since a
KILLED executing attempt still counts. The issue explicitly prefers a clean
derivation over bit-exact replication.

## Architecture

One semantic definition, two evaluation strategies:

- `reconcile/attempt_counts.py` (pure, import-clean): the state sets
  (`FAILURE_ATTEMPT_STATES`, `PREEMPTION_ATTEMPT_STATES`), a pure
  `counts_from_attempts(attempts) -> AttemptCounts` for callers that already
  hold the attempt rows, and SQLAlchemy column expressions
  (`failure_count_expr()`, `preemption_count_expr()`) that mirror the pure
  function for GROUP-BY aggregates. A unit test asserts the two agree.

- `projections/attempt_counts.py`: `AttemptCountsProjection` — a process-local
  in-memory cache of per-task `AttemptCounts`, following the
  `WorkerAttrsProjection` pattern (rehydrate on construction/reopen; post-commit
  invalidation under the DB write lock). Owns no table (`sources = ()`), so it
  imposes no `@writes_to` constraint; attempt-write sites call
  `attempt_counts.invalidate(tx, task_ids)`. Serves the standalone read-RPC hot
  paths (job summaries, task detail fill).

Consistency rule: paths that join derived counts with other reads from the
**same** DB snapshot (the reconcile loader; the `list_jobs` SQL sort) evaluate
via the SQL expressions in that snapshot, never the process-global cache, so a
concurrent commit can't desync the counts from the task/attempt rows they are
read alongside. The cache is only used where a slightly stale display value is
harmless.

### Cache reachability — the `CacheRegistry` lookaside

The projection is touched at only ~6 sites (3 reads in `service.py`; 3
invalidations: `commit_effects`, `purge_job`, `_mirror_delta`), but one sink
(`commit_effects`) fans out to four callers. Rather than thread a cache
reference through that call tree (or subclass the cursor), a per-controller
**`CacheRegistry`** (`caches.py`) — a type-keyed `{type: instance}` map — is
owned by `ControllerDB` (`db.caches`) and mirrored onto every `Tx` it mints
(`tx.caches`). `AttemptCountsProjection.__init__` self-registers into it, so:

- write sinks reach it via `cur.caches[AttemptCountsProjection]` — no param
  threaded, no cursor subtype;
- readers reach the same registry via `q.caches[AttemptCountsProjection]`;
- lookup is by concrete type (`caches[type[T]] -> T`), so there are no string
  keys and no `cast` at call sites (the one heterogeneous-container cast lives
  inside the registry).

This generalizes the existing per-transaction `Tx.memo` lookaside to a
per-controller scope. It replaced an earlier `ScopedTx(Tx)` subclass +
`ControllerScope` wrapper + `tx_factory` seam, which delivered the same
reachability with far more code for a single cache.

Invalidation is deferred to a `tx.register` post-commit hook that fires **under
the write lock**, so a concurrent reader either sees the pre-commit memo or
recomputes from the post-commit rows — never a torn value. `purge_job` is the
single job-deletion chokepoint, so a re-minted job id can't serve a dead job's
cached counts.

### Read/derive sites migrated

- Per-task RPC (`task_to_proto`, `TaskWithAttempts`): derive from the already
  loaded `attempts` via `counts_from_attempts`.
- `task_summaries_for_jobs`: task_count / completed_count / state histogram
  stay on the `tasks` table; failure/preemption sums come from the cache.
- `list_jobs` sort (`_AGG_FAILURES` / `_AGG_PREEMPTIONS`): SQL aggregate over
  `task_attempts` (can't drive a SQL ORDER BY from an in-memory cache).
- Reconcile snapshot loader: derive per-task counts via one SQL aggregate and
  populate the snapshot row shapes (`TaskDetailRow`, `ActiveTaskRow`,
  `TaskHistogramRow`). The kernel keeps its increment logic; the base value now
  comes from derivation instead of the column.

### Kernel / commit changes

- `TaskRowDelta.preemption_count` is removed: it was only ever consumed by the
  column write. `failure_count` stays on the delta as **overlay scratch** —
  `Overlay.job_basis` sums it for the job-level `max_task_failures` budget
  (`total_failures`), which must reflect in-batch failures not yet committed —
  but `commit._flush_tasks` no longer writes either column.
- `PendingTask` drops both count fields (the scheduler never reads them).

### Federation

Attempts are already fully mirrored (`mirror_federated_attempts`), including
`state` and `started_at_ms`, so the parent derives a federated task's counts
from the mirrored attempts. Removed: `TaskStatus.failure_count` (field 23) and
`TaskStatus.preemption_count` (field 28) from the proto, and the scalar writes
in `mirror_federated_task`. `JobStatus.failure_count/preemption_count`
(job-level aggregate, fields 17/18) stay; they are now sourced from the derived
job summary.

## Out of scope (evaluated, kept)

- **`tasks.current_attempt_id`** = `MAX(attempt_id)`. Kept: it is the task's
  live pointer to its in-flight attempt, written atomically with each attempt
  insert (`assign_to_worker` / `promote_for_dispatch`), read on the write hot
  path (dispatch, reconcile stale-attempt preconditions, endpoint validation)
  where it must reflect same-transaction inserts. Unlike the counters it is not
  incremented across many transitions, so it cannot drift the way they do.
- **`jobs.num_tasks`** = `COUNT(tasks)` for a materialized local job, **but** it
  doubles as the requested replica count before tasks exist and for a pre-sync
  federated job with no task rows (#6921). Deriving `COUNT(tasks)` would read 0
  in those states, losing the requested count. Kept.

## Schema / migration (0038)

- Drop `idx_tasks_job_failures` and `idx_tasks_job_state_counts` (both index the
  removed columns; `idx_tasks_job_state` already covers `(job_id, state)`).
- Drop columns `tasks.failure_count`, `tasks.preemption_count`.
- Add `idx_task_attempts_task_state` on `(task_id, state, started_at_ms)` to
  make the per-task/per-job derivation aggregates index-only.

## Spiral stages

1. `attempt_counts.py` semantic core + unit tests (pure vs SQL agreement).
2. `AttemptCountsProjection` cache + wiring into the store/controller.
3. Migrate read paths (reads.py, service.py) + federation derivation.
4. Migrate reconcile loader + kernel base; drop column writes.
5. Schema + migration 0038; drop proto fields; regenerate protos.
6. Dashboard/CLI: per-task counts derive from attempts.
7. Regenerate golden replays; add regression tests (mixed ASSIGNED/executing;
   federated mirror derivation; cache invalidation).
