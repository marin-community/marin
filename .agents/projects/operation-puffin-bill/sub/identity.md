# Sub-doc: Identity Model

Companion to `spec.md` §4.1. Draft 2.

## Changes from Draft 1

- **`attempt_uid` is the only wire-level routing key.** Carrying job_uid/task_uid on RPC messages was the bug class the doc warned about — three identity types to keep consistent. Deleted.
- **`job_uid` exists in the DB only.** Disambiguates same-name resubmissions on the storage side. Internal joins use it; the wire never sees it.
- **`task_uid` is not introduced at all.** Tasks are addressable via `attempt_uid` → `task_attempts.task_id`. No need for a separate task-level UID.

## The core distinction

| Kind | Type | Reusable? | Where it appears |
|---|---|---|---|
| **Name** (`JobName`) | hierarchical path like `/alice/train/0` | Yes — across submissions, retries, deletes | Dashboards, CLI, env vars, log filtering, search |
| **Job UID** | 128-bit random hex, server-minted | No | DB column on `jobs`. Used for incarnation disambiguation in joins. *Never on the wire.* |
| **Attempt UID** | 128-bit random hex, server-minted | No | The universal routing key. DB column, RPC field, container label, log key. |

Rule: **the wire carries `attempt_uid` and `JobName` (for display). Everything else is internal.**

## JobName API delta

Current (`lib/iris/src/iris/cluster/types.py:31`):

```python
@dataclass(frozen=True, slots=True)
class JobName:
    _parts: tuple[str, ...]

    def to_wire(self) -> str: ...
    def to_safe_token(self) -> str: ...
```

Proposed (Phase C):

```python
AttemptUid = NewType("AttemptUid", str)  # 16 hex chars

@dataclass(frozen=True, slots=True)
class JobName:
    _parts: tuple[str, ...]

    def to_wire(self) -> str: ...           # unchanged — display only
    def to_safe_token(self) -> str: ...     # unchanged — fs-safe hash of name
```

`JobName` is unchanged. UIDs ride on the row / proto / env — separately. This is the most important wire-discipline choice and the v1 spec wavered on it. v2 settles it: `JobName` carries names, period.

## Where UIDs go (Phase C+)

| Consumer | Today | After Phase C | Why |
|---|---|---|---|
| **`task_attempts` table** | PK `(task_id, attempt_id)` | + column `attempt_uid TEXT NOT NULL DEFAULT ''`. Promoted to PK in Phase D. | Routing key |
| **`tasks` table** | no UID | unchanged in Phase C (no need); Phase D may add `task_uid` for performance | Not needed for protocol |
| **`jobs` table** | no UID | + column `job_uid TEXT NOT NULL DEFAULT ''` | DB-only disambiguation |
| **Container labels** (`docker run --label`) | `task_id=<wire>, attempt_id=<int>` | + `attempt_uid=<hex>` | Adoption matches UID label preferentially |
| **Log key** (`task_log_key` in `log_store_helpers.py`) | `task_attempt.to_wire()` = `/alice/train/0:5` | `attempt_uid` directly | Disambiguates incarnations in finelog storage |
| **Workdir path** | `f"{safe_task_id}_attempt_{attempt_id}"` | `f"attempt_{attempt_uid}"` | Cross-incarnation safety |
| **`IRIS_TASK_ID` env var** | `/alice/train/0` | unchanged | Display string for task code |
| **`IRIS_ATTEMPT_UID` env var** (new) | n/a | `<16 hex>` | Lets task code echo incarnation in wandb runs, logs |
| **`Reconcile` RPC payload** | n/a | `attempt_uid` (primary) + `task_id` + `attempt_id` (compat through Phase D) | Routing |
| **`UpdateTaskStatus` push (legacy)** | `(task_id, attempt_id)` | dropped in Phase D | Replaced by worker-initiated Reconcile |
| **`GetTaskAttemptInfo` (legacy)** | `(task_id, attempt_id)` | dropped in Phase B | Spec rides inline in Reconcile |
| **Dashboard URLs** | `/jobs/<wire>/tasks/<idx>/attempts/<id>` | unchanged for permalink stability | URLs are user-facing; resolve to UID on the backend |

## What is *not* on the wire

- **`job_uid`** — not carried in any RPC message. Database-internal.
- **`task_uid`** — does not exist as a concept. Tasks are addressed by composite `(task_id, attempt_uid → task)` lookups internally.

This is a deliberate constraint. The v1 spec proposed carrying all three UIDs everywhere; the reviewer correctly identified this as recreating the bug class we were trying to eliminate. v2: **one UID type on the wire, period.**

## Wire-form discipline

```python
AttemptUid = NewType("AttemptUid", str)
```

`pyrefly` flags `dict[AttemptUid, X]` indexed with a plain `str`. Cast at the DB boundary exactly once.

## Migration of existing string-keyed consumers

Phase C migration adds UID columns and backfills (`sub/migration.md`). Code changes per consumer:

```python
# stores.py — added in Phase C
class TaskAttemptStore:
    def resolve_uid(self, snap, task_id: JobName, attempt_id: int) -> AttemptUid | None: ...
    def get_by_uid(self, snap, uid: AttemptUid) -> TaskAttemptRow | None: ...
    def list_by_worker(self, snap, worker_id: WorkerId) -> list[TaskAttemptRow]: ...
```

`get_by_uid` is the routing path Phase D promotes. `resolve_uid` is the compat path that disappears at Phase D.

## Container adoption across the deploy boundary

The hard case the v1 spec missed.

Phase C arrives. Existing containers (started before Phase C) have docker labels `(task_id=<wire>, attempt_id=<int>)`, no `attempt_uid` label. The migration backfills `attempt_uid` into `task_attempts` rows, including currently-running attempts. So the DB row has a UID but the running container does not.

Worker behavior on Phase C startup, encountering such a container:

1. Read container labels. Both `attempt_uid` and `(task_id, attempt_id)` are read.
2. If `attempt_uid` is present: route by UID.
3. Else: look up `task_attempts` by `(task_id, attempt_id)` and read the backfilled UID.
4. Tag the in-memory `TaskAttempt` with the UID.
5. Optionally relabel the docker container with `attempt_uid` (best-effort; not required for correctness — the in-memory tag is enough).

This works because composite keys remain unique across Phase C. Phase D promotes UID to PK, but by then all live containers have been started under Phase C+ (one full release cycle of Phase C running before Phase D) and have the UID label.

If a Phase B-era container survives all the way to Phase D (very long-running training job): the composite key still resolves via the secondary index, the worker tags the UID into the in-memory record, and the protocol works. Phase D's PK swap doesn't break the lookup; it just changes which index is primary.

## ExecInContainer, ProfileTask under UID-keyed world

`spec.md` §5.8 identifies these. Concretely:

```python
# Before (Phase A-B):
worker.exec_in_container(task_id="/alice/train/0", command=[...])

# After (Phase C+):
worker.exec_in_container(task_id="/alice/train/0", command=[...])           # name lookup; resolves to latest attempt
worker.exec_in_container(attempt_uid="a1b2c3...", command=[...])             # UID lookup; specific attempt
```

`task_id` keeps working — defaults to "most recent attempt" as today. `attempt_uid` adds the ability to address a specific historical attempt, useful for "debug this thing I see in logs from 5 minutes ago."

Same shape for `ProfileTask`: existing wire form `/job/.../task/N:attempt` is unchanged; an `attempt_uid` parameter is added.

## Why not bury UID in `JobName.to_wire()`?

The v1 spec considered this. Rejected because:

1. `to_wire()` is logged everywhere. Dashboards, log lines, container labels, env vars. Adding `#a1b2c3` to every identifier makes every log line wider and every grep less precise.
2. `to_wire()` is parsed by `from_wire()`. Dozens of callers do `JobName.from_wire(some_string)` on strings of unknown provenance.
3. Names and UIDs live on different cardinality scales. `JobName` queries want "all tasks of `/alice/train`" — that's the user mental model. UID is the implementation detail.

K8s' precedent: `metadata.name` and `metadata.uid` are sibling fields on every object, not concatenated. We do the same.

## The minimal Phase C migration

`sub/migration.md` has the SQL. The data-model summary:

```
ALTER TABLE jobs ADD COLUMN job_uid TEXT NOT NULL DEFAULT '';
ALTER TABLE task_attempts ADD COLUMN attempt_uid TEXT NOT NULL DEFAULT '';
-- backfill both with secrets.token_hex(8)
CREATE UNIQUE INDEX idx_jobs_uid ON jobs(job_uid);
CREATE UNIQUE INDEX idx_attempts_uid ON task_attempts(attempt_uid);
```

That's it. `tasks` doesn't get a UID column (not on the wire, not needed for incarnation disambiguation since `tasks.job_id` already references the disambiguated `jobs.job_id`).
