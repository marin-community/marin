# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""App-owned storage; an import becomes visible only after all requests are loaded."""

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS knotmeter_imports (
    id TEXT PRIMARY KEY,
    run_key TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    run JSONB NOT NULL,
    request_count INTEGER NOT NULL,
    loaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS knotmeter_requests (
    import_id TEXT NOT NULL REFERENCES knotmeter_imports(id),
    ordinal INTEGER NOT NULL,
    payload JSONB NOT NULL,
    PRIMARY KEY (import_id, ordinal)
);
ALTER TABLE knotmeter_imports ADD COLUMN IF NOT EXISTS series_count INTEGER NOT NULL DEFAULT 0;
CREATE TABLE IF NOT EXISTS knotmeter_series (
    import_id TEXT NOT NULL REFERENCES knotmeter_imports(id),
    part INTEGER NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (import_id, part)
);
CREATE TABLE IF NOT EXISTS knotmeter_current (
    run_key TEXT PRIMARY KEY,
    import_id TEXT NOT NULL REFERENCES knotmeter_imports(id)
);
CREATE TABLE IF NOT EXISTS knotmeter_corpus (
    dataset TEXT NOT NULL,
    revision TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    structure JSONB NOT NULL,
    PRIMARY KEY (dataset, revision, conversation_id)
);
ALTER TABLE knotmeter_corpus ADD COLUMN IF NOT EXISTS summary JSONB;
CREATE TABLE IF NOT EXISTS knotmeter_datasets (
    dataset TEXT NOT NULL,
    revision TEXT NOT NULL,
    summary JSONB NOT NULL,
    distributions JSONB NOT NULL,
    PRIMARY KEY (dataset, revision)
);
"""
