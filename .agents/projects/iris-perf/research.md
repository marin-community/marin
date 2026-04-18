# Iris Controller DB: Schema, Queries, and Performance Bottlenecks

## 1. Database Architecture

Single SQLite file with WAL mode, single `sqlite3.Connection` shared across all threads, serialized by a Python `RLock` (`db.py:1006-1007`).

Configuration (`db.py:1017-1021`):
```sql
PRAGMA journal_mode = WAL
PRAGMA synchronous = NORMAL
PRAGMA busy_timeout = 5000
PRAGMA foreign_keys = ON
```

### Connection Model & RLock Contention

`ControllerDB` holds exactly one `sqlite3.Connection` (`check_same_thread=False`) and one `RLock`. Every read and write acquires this lock:

- `snapshot()` → `QuerySnapshot.__enter__` acquires lock + `BEGIN` (`db.py:361-363`)
- `transaction()` → acquires lock + `BEGIN IMMEDIATE` (`db.py:1028-1039`)
- `fetchall()` / `fetchone()` → acquires lock for individual queries (`db.py:1041-1047`)

This means dashboard RPCs (reads) compete with the scheduling loop (reads + writes), heartbeat loop, and autoscaler for the same lock. With WAL mode, SQLite could support concurrent readers, but the Python RLock serializes them anyway.

## 2. Complete Table Schemas

### From `0001_init.sql`

```sql
CREATE TABLE schema_migrations (
    name TEXT PRIMARY KEY,
    applied_at_ms INTEGER NOT NULL
);

CREATE TABLE meta (
    key TEXT PRIMARY KEY,
    value INTEGER NOT NULL
);

CREATE TABLE users (
    user_id TEXT PRIMARY KEY,
    created_at_ms INTEGER NOT NULL
    -- Added by 0004_api_keys.sql:
    -- display_name TEXT
    -- role TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('admin', 'user', 'worker'))
);

CREATE TABLE jobs (
    job_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(user_id),
    parent_job_id TEXT REFERENCES jobs(job_id) ON DELETE CASCADE,
    root_job_id TEXT NOT NULL,
    depth INTEGER NOT NULL,
    request_proto BLOB NOT NULL,       -- serialized LaunchJobRequest protobuf
    state INTEGER NOT NULL,
    submitted_at_ms INTEGER NOT NULL,
    root_submitted_at_ms INTEGER NOT NULL,
    started_at_ms INTEGER,
    finished_at_ms INTEGER,
    scheduling_deadline_epoch_ms INTEGER,
    error TEXT,
    exit_code INTEGER,
    num_tasks INTEGER NOT NULL,
    is_reservation_holder INTEGER NOT NULL CHECK (is_reservation_holder IN (0, 1))
);

CREATE TABLE tasks (
    task_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    task_index INTEGER NOT NULL,
    state INTEGER NOT NULL,
    error TEXT,
    exit_code INTEGER,
    submitted_at_ms INTEGER NOT NULL,
    started_at_ms INTEGER,
    finished_at_ms INTEGER,
    max_retries_failure INTEGER NOT NULL,
    max_retries_preemption INTEGER NOT NULL,
    failure_count INTEGER NOT NULL,
    preemption_count INTEGER NOT NULL,
    resource_usage_proto BLOB,
    current_attempt_id INTEGER NOT NULL DEFAULT -1,
    priority_neg_depth INTEGER NOT NULL,
    priority_root_submitted_ms INTEGER NOT NULL,
    priority_insertion INTEGER NOT NULL,
    UNIQUE(job_id, task_index)
);

CREATE TABLE task_attempts (
    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    attempt_id INTEGER NOT NULL,
    worker_id TEXT REFERENCES workers(worker_id),
    state INTEGER NOT NULL,
    created_at_ms INTEGER NOT NULL,
    started_at_ms INTEGER,
    finished_at_ms INTEGER,
    exit_code INTEGER,
    error TEXT,
    PRIMARY KEY (task_id, attempt_id)
);

CREATE TABLE workers (
    worker_id TEXT PRIMARY KEY,
    address TEXT NOT NULL,
    metadata_proto BLOB NOT NULL,      -- serialized WorkerMetadata protobuf
    healthy INTEGER NOT NULL CHECK (healthy IN (0, 1)),
    active INTEGER NOT NULL CHECK (active IN (0, 1)),
    consecutive_failures INTEGER NOT NULL,
    last_heartbeat_ms INTEGER NOT NULL,
    committed_cpu_millicores INTEGER NOT NULL,
    committed_mem_bytes INTEGER NOT NULL,
    committed_gpu INTEGER NOT NULL,
    committed_tpu INTEGER NOT NULL,
    resource_snapshot_proto BLOB
);

CREATE TABLE worker_attributes (
    worker_id TEXT NOT NULL REFERENCES workers(worker_id) ON DELETE CASCADE,
    key TEXT NOT NULL,
    value_type TEXT NOT NULL CHECK (value_type IN ('str', 'int', 'float')),
    str_value TEXT,
    int_value INTEGER,
    float_value REAL,
    PRIMARY KEY (worker_id, key)
);

CREATE TABLE worker_task_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    worker_id TEXT NOT NULL REFERENCES workers(worker_id) ON DELETE CASCADE,
    task_id TEXT NOT NULL,
    assigned_at_ms INTEGER NOT NULL
);

CREATE TABLE worker_resource_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    worker_id TEXT NOT NULL REFERENCES workers(worker_id) ON DELETE CASCADE,
    snapshot_proto BLOB NOT NULL,
    timestamp_ms INTEGER NOT NULL
);

CREATE TABLE endpoints (
    endpoint_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    address TEXT NOT NULL,
    job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    task_id TEXT REFERENCES tasks(task_id) ON DELETE CASCADE,
    metadata_json TEXT NOT NULL,
    registered_at_ms INTEGER NOT NULL
);

CREATE TABLE dispatch_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    worker_id TEXT NOT NULL REFERENCES workers(worker_id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK (kind IN ('run', 'kill')),
    payload_proto BLOB,
    task_id TEXT,
    created_at_ms INTEGER NOT NULL
);

CREATE TABLE txn_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at_ms INTEGER NOT NULL
);

CREATE TABLE txn_actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    txn_id INTEGER NOT NULL REFERENCES txn_log(id) ON DELETE CASCADE,
    action TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    details_json TEXT NOT NULL,
    created_at_ms INTEGER NOT NULL
);

CREATE TABLE reservation_claims (
    worker_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    entry_idx INTEGER NOT NULL
);

CREATE TABLE logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL,
    source TEXT NOT NULL,
    data TEXT NOT NULL,
    epoch_ms INTEGER NOT NULL,
    level INTEGER NOT NULL DEFAULT 0
);
```

### From `0003_normalize_scaling_groups.sql`

```sql
CREATE TABLE scaling_groups (
    name TEXT PRIMARY KEY,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    backoff_until_ms INTEGER NOT NULL DEFAULT 0,
    last_scale_up_ms INTEGER NOT NULL DEFAULT 0,
    last_scale_down_ms INTEGER NOT NULL DEFAULT 0,
    quota_exceeded_until_ms INTEGER NOT NULL DEFAULT 0,
    quota_reason TEXT NOT NULL DEFAULT '',
    updated_at_ms INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE slices (
    slice_id TEXT PRIMARY KEY,
    scale_group TEXT NOT NULL,
    lifecycle TEXT NOT NULL,
    worker_ids TEXT NOT NULL DEFAULT '[]',    -- JSON array
    created_at_ms INTEGER NOT NULL DEFAULT 0,
    last_active_ms INTEGER NOT NULL DEFAULT 0,
    error_message TEXT NOT NULL DEFAULT ''
);
```

### From `0004_api_keys.sql`

```sql
CREATE TABLE api_keys (
    key_id TEXT PRIMARY KEY,
    key_hash TEXT NOT NULL UNIQUE,
    key_prefix TEXT NOT NULL,
    user_id TEXT NOT NULL REFERENCES users(user_id),
    name TEXT NOT NULL,
    created_at_ms INTEGER NOT NULL,
    last_used_at_ms INTEGER,
    expires_at_ms INTEGER,
    revoked_at_ms INTEGER
);
```

### From `0005_task_profiles.sql`

```sql
CREATE TABLE task_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    profile_data BLOB NOT NULL,
    captured_at_ms INTEGER NOT NULL
);
```

### From `0006_jwt_signing_key.sql`

```sql
CREATE TABLE controller_secrets (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    created_at_ms INTEGER NOT NULL
);
```

## 3. All Indices

### From `0001_init.sql`

```sql
CREATE INDEX idx_task_attempts_worker ON task_attempts(worker_id);
CREATE INDEX idx_tasks_pending ON tasks(state, priority_neg_depth, priority_root_submitted_ms, submitted_at_ms, priority_insertion);
CREATE INDEX idx_jobs_parent ON jobs(parent_job_id);
CREATE INDEX idx_endpoints_name ON endpoints(name);
CREATE INDEX idx_endpoints_task ON endpoints(task_id);
CREATE INDEX idx_dispatch_worker ON dispatch_queue(worker_id, id);
CREATE INDEX idx_txn_actions_txn ON txn_actions(txn_id, id);
CREATE INDEX idx_worker_task_history_worker ON worker_task_history(worker_id, assigned_at_ms DESC);
CREATE INDEX idx_worker_resource_history_worker ON worker_resource_history(worker_id, id DESC);
CREATE INDEX idx_logs_key ON logs(key, id);
```

### From `0002_read_indexes.sql`

```sql
CREATE INDEX idx_tasks_job_state ON tasks(job_id, state);
```

### From `0003_normalize_scaling_groups.sql`

```sql
CREATE INDEX idx_slices_scale_group ON slices(scale_group);
```

### From `0004_api_keys.sql`

```sql
CREATE INDEX idx_api_keys_hash ON api_keys(key_hash);
CREATE INDEX idx_api_keys_user ON api_keys(user_id);
```

### From `0004_worker_indexes.sql`

```sql
CREATE INDEX idx_workers_healthy_active ON workers(healthy, active);
```

### From `0005_task_profiles.sql`

```sql
CREATE INDEX idx_task_profiles_task ON task_profiles(task_id, id DESC);
```

### Triggers

```sql
-- Validates worker is active+healthy before inserting attempt
CREATE TRIGGER trg_task_attempt_active_worker
BEFORE INSERT ON task_attempts ...;

-- Caps txn_log at ~1000 rows (threshold check at 1100)
CREATE TRIGGER trg_txn_log_retention
AFTER INSERT ON txn_log WHEN (SELECT COUNT(*) FROM txn_log) > 1100 ...;

-- Caps task_profiles at 10 per task
CREATE TRIGGER trg_task_profiles_cap
AFTER INSERT ON task_profiles ...;
```

## 4. Scheduling Loop Hot-Path Queries

The scheduling loop runs every 0.5s (`controller.py:662`). Each cycle calls `_run_scheduling()` (`controller.py:1180+`).

### Phase: State reads (`controller.py:1208-1210`)

#### `_schedulable_tasks()` — `controller.py:289-301`

```sql
SELECT t.task_id, t.job_id, t.task_index, t.state, t.error, t.exit_code,
       t.submitted_at_ms, t.started_at_ms, t.finished_at_ms,
       t.max_retries_failure, t.max_retries_preemption,
       t.failure_count, t.preemption_count, t.resource_usage_proto,
       t.current_attempt_id, t.priority_neg_depth, t.priority_root_submitted_ms
FROM tasks t
WHERE t.state IS NOT NULL AND t.state NOT IN (succeeded, failed, killed, unschedulable, worker_failed)
ORDER BY t.priority_neg_depth ASC, t.priority_root_submitted_ms ASC,
         t.submitted_at_ms ASC, t.task_id ASC
```

**Bottleneck**: Fetches ALL non-terminal tasks. With 3500+ tasks (issue context), this decodes every row including `resource_usage_proto` (protobuf blob). Then Python-side `can_be_scheduled()` filters further. The `idx_tasks_pending` index covers this WHERE+ORDER but every matching row's full column set is fetched.

#### `healthy_active_workers_with_attributes()` — `db.py:1233-1252`

Two queries in one snapshot:
```sql
-- Query 1: workers
SELECT w.* FROM workers w WHERE w.healthy = 1 AND w.active = 1

-- Query 2: attributes (only if workers found)
SELECT wa.worker_id, wa.key, wa.value_type, wa.str_value, wa.int_value, wa.float_value
FROM worker_attributes wa
WHERE wa.worker_id IN (?, ?, ...)
```

Covered by `idx_workers_healthy_active`. Worker count is small (~100s), so this is fast.

### Phase: Job lookups (`controller.py:1226`)

#### `_jobs_by_id()` — `controller.py:281-286`

```sql
SELECT j.* FROM jobs j WHERE j.job_id IN (?, ?, ...)
```

Fetches jobs for all pending tasks' job_ids. Decodes `request_proto` (protobuf blob) per job.

### Phase: Building counts (`controller.py:1261-1262`)

#### `_building_counts()` — `controller.py:322-346`

**Three separate snapshot acquisitions** (3 lock acquire/release cycles):

1. `running_tasks_by_worker()` (`db.py:1193-1212`):
```sql
SELECT a.worker_id, t.task_id FROM tasks t
JOIN task_attempts a ON t.task_id = a.task_id AND t.current_attempt_id = a.attempt_id
WHERE a.worker_id IN (?, ...) AND t.state IN (assigned, building, running)
```

2. `_tasks_by_ids_with_attempts()` (`controller.py:304-319`):
```sql
-- Query A: tasks by IDs
SELECT t.* FROM tasks t WHERE t.task_id IN (?, ...) ORDER BY t.task_id ASC

-- Query B: attempts for those tasks
SELECT a.* FROM task_attempts a WHERE a.task_id IN (?, ...) ORDER BY a.task_id ASC, a.attempt_id ASC
```

3. `_jobs_by_id()` again (`controller.py:281-286`):
```sql
SELECT j.* FROM jobs j WHERE j.job_id IN (?, ...)
```

**Bottleneck**: 3 separate lock acquisitions and snapshot contexts. The running_tasks query could be combined with the task/attempt fetch. The job fetch here duplicates work done in the earlier phase.

## 5. Dashboard/RPC Query Patterns

All dashboard data fetching goes through Connect RPC calls from browser JS to `ControllerServiceImpl` methods in `service.py`.

### `list_jobs()` — `service.py:769-920`

The main jobs listing page. Calls:

1. `_jobs_in_states(db, USER_JOB_STATES)` (`service.py:316-318`):
```sql
SELECT j.* FROM jobs j WHERE j.state IN (?, ?, ?, ?, ?, ?, ?, ?)
```
Fetches ALL jobs in any state (8 states = all states). Every row decodes `request_proto` blob.

2. `_task_summaries_for_jobs()` (`service.py:321-346`):
```sql
SELECT t.job_id, t.state, t.failure_count, t.preemption_count
FROM tasks t WHERE t.job_id IN (?, ?, ...)
```
Fetches 4 columns per task for ALL jobs. With 3500+ tasks this is substantial but lightweight per row.

**Bottleneck**: Step 1 decodes protobuf for every job. Sorting and pagination happen in Python after full fetch. No SQL-level LIMIT/OFFSET.

### `get_job_status()` — `service.py:670-728`

Job detail page. Calls:

1. `_read_job()` — single job by ID
2. `tasks_for_job_with_attempts()` (`db.py:1215-1230`) — all tasks + attempts for one job
3. `_worker_addresses()` (`service.py:310-313`) — ALL workers' addresses:
```sql
SELECT w.worker_id, w.address FROM workers w
```

**Bottleneck**: `_worker_addresses()` fetches the entire workers table even though only a few workers are relevant.

### `list_tasks()` — `service.py:949-970`

1. `_tasks_for_listing()` (`service.py:293-307`) — all tasks (or filtered by job_id) with attempts:
```sql
-- Without job_id filter: fetches ALL tasks + ALL attempts
SELECT t.* FROM tasks t ORDER BY t.task_id ASC
SELECT a.* FROM task_attempts a WHERE a.task_id IN (?, ...) ORDER BY ...
```
2. `_worker_addresses()` — again fetches ALL workers.

**Bottleneck**: Without a job_id filter, fetches every task and every attempt in the system.

### `list_workers()` — `service.py:1008-1030`

1. `_worker_roster()` (`service.py:349-351`) — ALL workers:
```sql
SELECT w.* FROM workers w
```
2. `running_tasks_by_worker()` — running tasks for all workers.

### `list_endpoints()` — `service.py:1091-1115`

```sql
SELECT e.* FROM endpoints e
[JOIN jobs j ON e.job_id = j.job_id WHERE j.state NOT IN (terminal...)]
[WHERE e.name LIKE ? OR e.name = ?]
ORDER BY e.registered_at_ms DESC, e.endpoint_id ASC
```

Filtered by name, joins with jobs to exclude terminal. Reasonably efficient.

### `get_autoscaler_status()` — `service.py:1119-1152`

1. `_worker_roster()` — ALL workers
2. `running_tasks_by_worker()` — running tasks for all workers

Same pattern as `list_workers()`.

### `_live_user_stats()` — `service.py:381-400`

```sql
-- All jobs with user_id
SELECT j.user_id, j.state FROM jobs j

-- All tasks joined with jobs for user_id
SELECT ju.user_id, t.state FROM tasks t JOIN jobs ju ON t.job_id = ju.job_id
```

Scans entire jobs and tasks tables.

### `get_task_logs()` — `service.py:1158+`

Reads from in-memory log store, not the DB (logs table is for persistence).

### `execute_raw_query()` — `service.py:1681-1693`, `query.py:42-61`

Admin-only RPC that accepts raw SQL SELECT:
```python
def execute_raw_query(db: ControllerDB, sql: str) -> QueryResult:
    # Validates: must start with SELECT, no forbidden keywords
    with db.snapshot() as q:
        cursor = q.execute_sql(stripped)
        raw_rows = cursor.fetchall()
    # Returns columns + JSON-encoded rows
```

Forbidden keywords: INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, ATTACH, DETACH, PRAGMA, VACUUM, REINDEX, SAVEPOINT. Uses snapshot isolation (BEGIN + ROLLBACK).

## 6. Checkpoint/Restore

### Writing checkpoints (`checkpoint.py:55-93`)

```python
def write_checkpoint(db, remote_state_dir):
    db.backup_to(tmp_path)                     # SQLite backup API under RLock
    _fsspec_copy(tmp_path, remote_timestamped)  # gs://.../.../checkpoint-{epoch}.sqlite3
    _fsspec_copy(tmp_path, remote_latest)       # gs://.../.../latest.sqlite3
```

`backup_to()` (`db.py:1130-1139`) uses `self._conn.backup(dest)` under the RLock, creating a consistent snapshot.

### Downloading a checkpoint locally

```python
def download_checkpoint_to_local(remote_state_dir, local_db_path, checkpoint_path=None):
    source = checkpoint_path or f"{remote_state_dir}/controller-state/latest.sqlite3"
    _fsspec_copy(source, str(local_db_path))
```

To get a DB locally for benchmarking:
```bash
# Find the remote state dir from controller config, then:
gsutil cp gs://<bucket>/<prefix>/controller-state/latest.sqlite3 ./controller.sqlite3
# Or a timestamped checkpoint:
gsutil ls gs://<bucket>/<prefix>/controller-state/checkpoint-*.sqlite3
gsutil cp gs://<bucket>/<prefix>/controller-state/checkpoint-<ts>.sqlite3 ./controller.sqlite3
```

Checkpoints are written periodically (autoscaler loop, `controller.py:976-980`) and at exit (`controller.py:939-944`).

## 7. Identified Bottlenecks

### B1: RLock serializes all DB access (`db.py:1006`)

WAL mode allows concurrent readers, but the single-connection + RLock design means:
- Dashboard RPCs block on scheduling writes
- Multiple dashboard tabs/requests queue behind each other
- The scheduling loop (every 0.5s) holds the lock for state reads + writes

**Fix options**: Use a connection pool with separate read connections, or use `check_same_thread=False` with WAL mode's built-in reader concurrency.

### B2: `_schedulable_tasks()` full table scan (`controller.py:289-301`)

Fetches all non-terminal tasks including `resource_usage_proto` blobs, then Python-side filters with `can_be_scheduled()`. With 3500+ non-terminal tasks, this is the most expensive read per scheduling cycle.

**Missing**: The query doesn't exclude tasks in ACTIVE states (ASSIGNED, BUILDING, RUNNING) that can never be scheduled. Adding `state = PENDING OR current_attempt_id < 0` to the WHERE clause would reduce rows significantly.

### B3: `_building_counts()` 3-round-trip pattern (`controller.py:322-346`)

Three separate snapshot contexts means 3 lock acquire/release cycles. The data flow is:
1. running tasks by worker → set of task IDs
2. tasks + attempts by those IDs → task details
3. jobs by task.job_id → job details (is_reservation_holder check)

This could be a single snapshot with a single query joining tasks, attempts, and jobs.

### B4: `list_jobs()` fetches+decodes all jobs (`service.py:769-920`)

Decodes `request_proto` protobuf for every job, then sorts and paginates in Python. For a paginated dashboard showing 50 jobs, the server still processes all jobs.

**Missing**: SQL-level pagination. The sorting could partially move to SQL (by submitted_at_ms, state) with LIMIT/OFFSET, but state-priority sorting and name filtering would need careful handling.

### B5: `_task_summaries_for_jobs()` scans all tasks (`service.py:321-346`)

Fetches 4 columns per task for every job. Could be replaced with:
```sql
SELECT job_id, state, SUM(failure_count), SUM(preemption_count), COUNT(*)
FROM tasks WHERE job_id IN (...) GROUP BY job_id, state
```

### B6: `_worker_addresses()` fetches all workers for single-job views (`service.py:310-313`)

`get_job_status()` and `list_tasks()` both call this even when only a few workers are relevant.

### B7: Protobuf decoding cost

`Job.request_proto` contains the full `LaunchJobRequest` protobuf. Every `_decode_row(Job, row)` call deserializes this. For `list_jobs()` fetching 100s of jobs, this is significant CPU.

### B8: No index on `tasks.state` alone

The `idx_tasks_pending` composite index starts with `state` but includes 4 more columns for priority ordering. Queries that only filter by state (like `_task_summaries_for_jobs`) can use the leading `state` column, but the composite index is wider than needed for simple state lookups. The `idx_tasks_job_state` index covers `(job_id, state)` which helps per-job queries.

## 8. Summary of Dashboard Page → RPC → Query Chains

| Dashboard Page | RPC | Key Queries | Rows Touched |
|---|---|---|---|
| Jobs list | `list_jobs` | All jobs + all tasks (summary) | O(jobs + tasks) |
| Job detail | `get_job_status` | 1 job + job's tasks+attempts + all workers | O(tasks_in_job × attempts + workers) |
| Workers list | `list_workers` | All workers + running tasks | O(workers + active_tasks) |
| Worker detail | `get_worker_detail` | 1 worker + running tasks + resource history | O(history_limit) |
| Endpoints | `list_endpoints` | Endpoints + jobs join | O(endpoints) |
| Autoscaler | `get_autoscaler_status` | All workers + running tasks | O(workers + active_tasks) |
| User stats | `_live_user_stats` | All jobs + all tasks joined with jobs | O(jobs + tasks) |
| Raw query | `execute_raw_query` | Arbitrary SELECT | Unbounded |
