-- Performance indexes for scheduling and dashboard queries.

-- Covering index for _building_counts JOIN: task_attempts(worker_id) → tasks(task_id, current_attempt_id)
CREATE INDEX IF NOT EXISTS idx_task_attempts_worker_task
    ON task_attempts(worker_id, task_id, attempt_id);

-- Index for list_jobs sorting by state and date
CREATE INDEX IF NOT EXISTS idx_jobs_state
    ON jobs(state, submitted_at_ms DESC);

-- Index for list_jobs filtering by depth (top-level jobs)
CREATE INDEX IF NOT EXISTS idx_jobs_depth_state
    ON jobs(depth, state, submitted_at_ms DESC);
