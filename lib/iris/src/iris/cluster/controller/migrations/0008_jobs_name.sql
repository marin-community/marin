-- Denormalized name column for SQL-level filtering in list_jobs.
ALTER TABLE jobs ADD COLUMN name TEXT NOT NULL DEFAULT '';
CREATE INDEX IF NOT EXISTS idx_jobs_name ON jobs(name);
