# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Create the persistent codehealth review workbench tables."""

import sqlalchemy

DDL = """
CREATE TABLE codehealth_sync_runs (
    sync_id TEXT PRIMARY KEY,
    repository TEXT NOT NULL,
    window_start TIMESTAMPTZ NOT NULL,
    window_end TIMESTAMPTZ NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    status TEXT NOT NULL,
    attempt_count INTEGER NOT NULL,
    candidate_pull_requests INTEGER,
    github_usage JSONB,
    finelog_watermark JSONB,
    error TEXT
);
CREATE INDEX ix_codehealth_sync_runs_repository_started
    ON codehealth_sync_runs (repository, started_at);

CREATE TABLE codehealth_sync_pull_requests (
    sync_id TEXT NOT NULL REFERENCES codehealth_sync_runs(sync_id) ON DELETE CASCADE,
    pr_number INTEGER NOT NULL,
    synced_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (sync_id, pr_number)
);

CREATE TABLE codehealth_pull_requests (
    repository TEXT NOT NULL,
    pr_number INTEGER NOT NULL,
    node_id TEXT NOT NULL,
    author TEXT NOT NULL,
    state TEXT NOT NULL,
    head_sha TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    record_sha TEXT NOT NULL,
    record JSONB NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (repository, pr_number)
);
CREATE INDEX ix_codehealth_pull_requests_updated
    ON codehealth_pull_requests (updated_at);

CREATE TABLE codehealth_pull_request_versions (
    repository TEXT NOT NULL,
    pr_number INTEGER NOT NULL,
    record_sha TEXT NOT NULL,
    record JSONB NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (repository, pr_number, record_sha)
);

CREATE TABLE codehealth_review_events (
    event_id TEXT PRIMARY KEY,
    repository TEXT NOT NULL,
    pr_number INTEGER NOT NULL,
    kind TEXT NOT NULL,
    database_id BIGINT NOT NULL,
    author TEXT NOT NULL,
    is_human BOOLEAN NOT NULL,
    activity_at TIMESTAMPTZ NOT NULL,
    record_sha TEXT NOT NULL,
    record JSONB NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX ix_codehealth_review_events_pr_activity
    ON codehealth_review_events (repository, pr_number, activity_at);
CREATE INDEX ix_codehealth_review_events_author_activity
    ON codehealth_review_events (author, activity_at);

CREATE TABLE codehealth_review_event_versions (
    event_id TEXT NOT NULL,
    record_sha TEXT NOT NULL,
    record JSONB NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (event_id, record_sha)
);

CREATE TABLE codehealth_review_threads (
    repository TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    pr_number INTEGER NOT NULL,
    record_sha TEXT NOT NULL,
    record JSONB NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (repository, thread_id)
);

CREATE TABLE codehealth_changed_files (
    repository TEXT NOT NULL,
    pr_number INTEGER NOT NULL,
    filename TEXT NOT NULL,
    record JSONB NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (repository, pr_number, filename)
);
CREATE INDEX ix_codehealth_changed_files_filename
    ON codehealth_changed_files (filename);

CREATE TABLE codehealth_commits (
    repository TEXT NOT NULL,
    pr_number INTEGER NOT NULL,
    sha TEXT NOT NULL,
    position INTEGER NOT NULL,
    record JSONB NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (repository, pr_number, sha)
);

CREATE TABLE codehealth_pull_request_diffs (
    repository TEXT NOT NULL,
    pr_number INTEGER NOT NULL,
    head_sha TEXT NOT NULL,
    diff TEXT,
    observed_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (repository, pr_number, head_sha)
);

CREATE TABLE codehealth_source_contexts (
    event_id TEXT NOT NULL,
    commit_sha TEXT NOT NULL,
    path TEXT NOT NULL,
    anchor_line INTEGER NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    text TEXT,
    unavailable_reason TEXT,
    content_sha TEXT NOT NULL,
    fetched_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (event_id, commit_sha)
);

CREATE TABLE codehealth_lint_invocations (
    invocation_id TEXT NOT NULL,
    repository TEXT NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    pr_number INTEGER,
    head_sha TEXT,
    catalog_sha TEXT,
    successful BOOLEAN NOT NULL,
    finding_count INTEGER NOT NULL,
    record JSONB NOT NULL,
    PRIMARY KEY (repository, invocation_id)
);
CREATE INDEX ix_codehealth_lint_invocations_pr_ts
    ON codehealth_lint_invocations (repository, pr_number, ts);

CREATE TABLE codehealth_lint_findings (
    finding_id TEXT NOT NULL,
    invocation_id TEXT NOT NULL,
    repository TEXT NOT NULL,
    ts TIMESTAMPTZ NOT NULL,
    pr_number INTEGER,
    code TEXT NOT NULL,
    record JSONB NOT NULL,
    PRIMARY KEY (repository, finding_id),
    FOREIGN KEY (repository, invocation_id)
        REFERENCES codehealth_lint_invocations(repository, invocation_id) ON DELETE CASCADE
);
CREATE INDEX ix_codehealth_lint_findings_pr_code
    ON codehealth_lint_findings (repository, pr_number, code);

CREATE TABLE codehealth_lint_catalog_snapshots (
    catalog_sha TEXT PRIMARY KEY,
    observed_at TIMESTAMPTZ NOT NULL,
    record JSONB NOT NULL
);

CREATE TABLE codehealth_rule_probes (
    probe_id TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL,
    event_id TEXT NOT NULL,
    context_sha TEXT NOT NULL,
    rule_code TEXT NOT NULL,
    rule_sha TEXT NOT NULL,
    catalog_sha TEXT NOT NULL,
    model TEXT NOT NULL,
    effort TEXT NOT NULL,
    status TEXT NOT NULL,
    fired BOOLEAN,
    confidence DOUBLE PRECISION,
    finding TEXT,
    raw_output TEXT,
    error TEXT,
    elapsed DOUBLE PRECISION NOT NULL,
    record JSONB NOT NULL
);

GRANT SELECT ON
    codehealth_sync_runs,
    codehealth_sync_pull_requests,
    codehealth_pull_requests,
    codehealth_pull_request_versions,
    codehealth_review_events,
    codehealth_review_event_versions,
    codehealth_review_threads,
    codehealth_changed_files,
    codehealth_commits,
    codehealth_pull_request_diffs,
    codehealth_source_contexts,
    codehealth_lint_invocations,
    codehealth_lint_findings,
    codehealth_lint_catalog_snapshots,
    codehealth_rule_probes
TO "eng-all@openathena.ai";
GRANT SELECT, INSERT, UPDATE ON
    codehealth_sync_runs,
    codehealth_sync_pull_requests,
    codehealth_pull_requests,
    codehealth_pull_request_versions,
    codehealth_review_events,
    codehealth_review_event_versions,
    codehealth_review_threads,
    codehealth_changed_files,
    codehealth_commits,
    codehealth_pull_request_diffs,
    codehealth_source_contexts,
    codehealth_lint_invocations,
    codehealth_lint_findings,
    codehealth_lint_catalog_snapshots,
    codehealth_rule_probes
TO "loom-vm@hai-gcp-models.iam";
GRANT DELETE ON codehealth_review_events, codehealth_review_threads, codehealth_changed_files, codehealth_commits
TO "loom-vm@hai-gcp-models.iam";
"""


def upgrade(conn: sqlalchemy.Connection) -> None:
    for statement in DDL.strip().split(";"):
        if statement.strip():
            conn.execute(sqlalchemy.text(statement))
