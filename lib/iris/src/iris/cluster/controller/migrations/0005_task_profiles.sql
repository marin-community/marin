-- Task profiles captured by the controller's periodic profiling loop.
-- Stores the last N (default 10) CPU profiles per task, automatically
-- evicting the oldest when the cap is exceeded.

CREATE TABLE IF NOT EXISTS task_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    profile_data BLOB NOT NULL,
    captured_at_ms INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_task_profiles_task ON task_profiles(task_id, id DESC);

-- Cap profiles at 10 per task: after each insert, delete the oldest
-- rows beyond the limit.
CREATE TRIGGER IF NOT EXISTS trg_task_profiles_cap
AFTER INSERT ON task_profiles
BEGIN
  DELETE FROM task_profiles
   WHERE task_id = NEW.task_id
     AND id NOT IN (
       SELECT id FROM task_profiles
        WHERE task_id = NEW.task_id
        ORDER BY id DESC
        LIMIT 10
     );
END;
