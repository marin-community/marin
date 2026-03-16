PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_migrations (
    name TEXT PRIMARY KEY,
    applied_at_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    created_at_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(user_id),
    parent_job_id TEXT REFERENCES jobs(job_id) ON DELETE CASCADE,
    root_job_id TEXT NOT NULL,
    depth INTEGER NOT NULL,
    request_proto BLOB NOT NULL,
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

CREATE TABLE IF NOT EXISTS tasks (
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

CREATE TABLE IF NOT EXISTS task_attempts (
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

CREATE TABLE IF NOT EXISTS workers (
    worker_id TEXT PRIMARY KEY,
    address TEXT NOT NULL,
    metadata_proto BLOB NOT NULL,
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

CREATE TABLE IF NOT EXISTS worker_attributes (
    worker_id TEXT NOT NULL REFERENCES workers(worker_id) ON DELETE CASCADE,
    key TEXT NOT NULL,
    value_type TEXT NOT NULL CHECK (value_type IN ('str', 'int', 'float')),
    str_value TEXT,
    int_value INTEGER,
    float_value REAL,
    PRIMARY KEY (worker_id, key)
);

CREATE TABLE IF NOT EXISTS worker_task_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    worker_id TEXT NOT NULL REFERENCES workers(worker_id) ON DELETE CASCADE,
    task_id TEXT NOT NULL,
    assigned_at_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS worker_resource_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    worker_id TEXT NOT NULL REFERENCES workers(worker_id) ON DELETE CASCADE,
    snapshot_proto BLOB NOT NULL,
    timestamp_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS endpoints (
    endpoint_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    address TEXT NOT NULL,
    job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    task_id TEXT REFERENCES tasks(task_id) ON DELETE CASCADE,
    metadata_json TEXT NOT NULL,
    registered_at_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS dispatch_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    worker_id TEXT NOT NULL REFERENCES workers(worker_id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK (kind IN ('run', 'kill')),
    payload_proto BLOB,
    task_id TEXT,
    created_at_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS txn_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS txn_actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    txn_id INTEGER NOT NULL REFERENCES txn_log(id) ON DELETE CASCADE,
    action TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    details_json TEXT NOT NULL,
    created_at_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS scaling_groups (
    name TEXT PRIMARY KEY,
    snapshot_proto BLOB NOT NULL,
    updated_at_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS tracked_workers (
    worker_id TEXT PRIMARY KEY,
    slice_id TEXT NOT NULL,
    scale_group TEXT NOT NULL,
    internal_address TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reservation_claims (
    worker_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    entry_idx INTEGER NOT NULL
);

-- NOTE: The ``logs`` table lives in a separate ATTACHed database (logs.db).
-- See ControllerDB._attach_log_db() for its schema.

CREATE INDEX IF NOT EXISTS idx_task_attempts_worker ON task_attempts(worker_id);
CREATE INDEX IF NOT EXISTS idx_tasks_pending ON tasks(state, priority_neg_depth, priority_root_submitted_ms, submitted_at_ms, priority_insertion);
CREATE INDEX IF NOT EXISTS idx_jobs_parent ON jobs(parent_job_id);
CREATE INDEX IF NOT EXISTS idx_endpoints_name ON endpoints(name);
CREATE INDEX IF NOT EXISTS idx_endpoints_task ON endpoints(task_id);
CREATE INDEX IF NOT EXISTS idx_dispatch_worker ON dispatch_queue(worker_id, id);
CREATE INDEX IF NOT EXISTS idx_txn_actions_txn ON txn_actions(txn_id, id);
CREATE INDEX IF NOT EXISTS idx_worker_task_history_worker ON worker_task_history(worker_id, assigned_at_ms DESC);
CREATE INDEX IF NOT EXISTS idx_worker_resource_history_worker ON worker_resource_history(worker_id, id DESC);

CREATE TRIGGER IF NOT EXISTS trg_task_attempt_active_worker
BEFORE INSERT ON task_attempts
FOR EACH ROW
WHEN NEW.worker_id IS NOT NULL
BEGIN
  SELECT
    CASE
      WHEN NOT EXISTS(
        SELECT 1 FROM workers w
        WHERE w.worker_id = NEW.worker_id
          AND w.active = 1
          AND w.healthy = 1
      )
      THEN RAISE(ABORT, 'task attempt worker must be active and healthy')
    END;
END;

CREATE TRIGGER IF NOT EXISTS trg_txn_log_retention
AFTER INSERT ON txn_log
BEGIN
  DELETE FROM txn_log
   WHERE id NOT IN (
     SELECT id FROM txn_log ORDER BY id DESC LIMIT 1000
   );
END;
