CREATE TABLE IF NOT EXISTS controller_secrets (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    created_at_ms INTEGER NOT NULL
);
