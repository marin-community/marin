# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Postgres storage for one staged curation catalog load."""

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS atlas_loads (
    load_id TEXT PRIMARY KEY, catalog_root TEXT NOT NULL,
    snapshot_id TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL,
    loaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS atlas_current (
    name TEXT PRIMARY KEY, load_id TEXT
);
INSERT INTO atlas_current (name) VALUES ('latest') ON CONFLICT DO NOTHING;
CREATE TABLE IF NOT EXISTS atlas_sources (
    load_id TEXT NOT NULL, id TEXT NOT NULL, declaration JSONB NOT NULL,
    assessment_key TEXT, latest_attempt_key TEXT, latest_attempt_status TEXT NOT NULL,
    reason TEXT, curation JSONB, diagnostics JSONB NOT NULL,
    PRIMARY KEY (load_id, id)
);
CREATE TABLE IF NOT EXISTS atlas_assessments (
    load_id TEXT NOT NULL, pipeline_key TEXT NOT NULL, source_id TEXT NOT NULL,
    scope TEXT NOT NULL, model JSONB NOT NULL, rubric TEXT NOT NULL,
    curation JSONB NOT NULL, curation_uri TEXT NOT NULL, traces JSONB NOT NULL,
    PRIMARY KEY (load_id, pipeline_key)
);
CREATE TABLE IF NOT EXISTS atlas_populations (
    load_id TEXT NOT NULL, pipeline_key TEXT NOT NULL, cohort TEXT NOT NULL,
    population BIGINT NOT NULL, sample_count BIGINT NOT NULL,
    PRIMARY KEY (load_id, pipeline_key, cohort)
);
CREATE TABLE IF NOT EXISTS atlas_items (
    load_id TEXT NOT NULL, pipeline_key TEXT NOT NULL, task_id TEXT NOT NULL,
    cohort TEXT NOT NULL, status TEXT NOT NULL, verdict TEXT NOT NULL,
    explanation TEXT NOT NULL, evidence_ids JSONB NOT NULL, packet_uri TEXT NOT NULL,
    has_task BOOLEAN NOT NULL, omissions JSONB NOT NULL, exclusion_reason TEXT,
    PRIMARY KEY (load_id, pipeline_key, task_id)
);
CREATE TABLE IF NOT EXISTS atlas_packets (
    load_id TEXT NOT NULL, pipeline_key TEXT NOT NULL, task_id TEXT NOT NULL,
    packet TEXT NOT NULL,
    PRIMARY KEY (load_id, pipeline_key, task_id)
);
CREATE OR REPLACE VIEW atlas_cohort_summary AS
SELECT p.load_id, p.pipeline_key, p.cohort, p.population, p.sample_count,
    COUNT(i.task_id) FILTER (WHERE i.status = 'reviewed') AS reviewed,
    COUNT(i.task_id) FILTER (WHERE NOT i.has_task) AS missing_evidence,
    COUNT(i.task_id) FILTER (WHERE jsonb_array_length(i.omissions) > 0) AS packets_with_omissions,
    jsonb_build_object(
        'good', COUNT(i.task_id) FILTER (WHERE i.verdict = 'good'),
        'some_issues', COUNT(i.task_id) FILTER (WHERE i.verdict = 'some_issues'),
        'bad', COUNT(i.task_id) FILTER (WHERE i.verdict = 'bad'),
        'unknown', COUNT(i.task_id) FILTER (WHERE i.verdict = 'unknown')
    ) AS verdict_counts,
    CASE
        WHEN p.sample_count = 0 OR COUNT(i.task_id) <> p.sample_count
            OR COUNT(i.task_id) FILTER (WHERE i.status <> 'reviewed' OR NOT i.has_task) > 0 THEN 'unknown'
        WHEN COUNT(i.task_id) FILTER (WHERE i.verdict = 'bad') > 0 THEN 'bad'
        WHEN COUNT(i.task_id) FILTER (WHERE i.verdict = 'some_issues') > 0 THEN 'some_issues'
        WHEN COUNT(i.task_id) FILTER (WHERE i.verdict = 'unknown') > 0 THEN 'unknown'
        ELSE 'good'
    END AS grade
FROM atlas_populations p LEFT JOIN atlas_items i
    ON i.load_id = p.load_id AND i.pipeline_key = p.pipeline_key AND i.cohort = p.cohort
GROUP BY p.load_id, p.pipeline_key, p.cohort, p.population, p.sample_count;
"""
