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
CREATE TABLE IF NOT EXISTS curriculum_loads (
    load_id TEXT PRIMARY KEY, release_id TEXT NOT NULL, release_uri TEXT NOT NULL,
    manifest JSONB NOT NULL, loaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS curriculum_current (
    name TEXT PRIMARY KEY, load_id TEXT, previous_load_id TEXT
);
INSERT INTO curriculum_current (name) VALUES ('latest') ON CONFLICT DO NOTHING;
CREATE TABLE IF NOT EXISTS curriculum_sources (
    load_id TEXT NOT NULL, source_id TEXT NOT NULL, declaration JSONB NOT NULL,
    family TEXT NOT NULL, alignment JSONB,
    PRIMARY KEY (load_id, source_id)
);
CREATE TABLE IF NOT EXISTS curriculum_nodes (
    load_id TEXT NOT NULL, subject_id TEXT NOT NULL, node_id TEXT NOT NULL,
    parent_id TEXT, kind TEXT NOT NULL, name TEXT NOT NULL, description TEXT NOT NULL,
    PRIMARY KEY (load_id, subject_id, node_id)
);
CREATE TABLE IF NOT EXISTS curriculum_ancestors (
    load_id TEXT NOT NULL, subject_id TEXT NOT NULL, node_id TEXT NOT NULL, ancestor_id TEXT NOT NULL,
    PRIMARY KEY (load_id, subject_id, node_id, ancestor_id)
);
CREATE TABLE IF NOT EXISTS curriculum_tasks (
    load_id TEXT NOT NULL, source_id TEXT NOT NULL, task_id TEXT NOT NULL,
    content_hash TEXT NOT NULL, description_status TEXT NOT NULL, description TEXT NOT NULL,
    detail JSONB NOT NULL,
    PRIMARY KEY (load_id, source_id, task_id),
    FOREIGN KEY (load_id, source_id) REFERENCES curriculum_sources (load_id, source_id)
);
CREATE TABLE IF NOT EXISTS curriculum_matches (
    load_id TEXT NOT NULL, source_id TEXT NOT NULL, task_id TEXT NOT NULL,
    subject_id TEXT NOT NULL, capability_id TEXT NOT NULL, facet TEXT NOT NULL,
    membership_similarity DOUBLE PRECISION NOT NULL, capability_similarity DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (load_id, source_id, task_id, subject_id),
    FOREIGN KEY (load_id, source_id, task_id) REFERENCES curriculum_tasks (load_id, source_id, task_id)
);
CREATE INDEX IF NOT EXISTS curriculum_match_area
    ON curriculum_matches (load_id, subject_id, capability_id, source_id, task_id);
CREATE TABLE IF NOT EXISTS curriculum_statistics (
    load_id TEXT NOT NULL, source_id TEXT NOT NULL, metric TEXT NOT NULL,
    subject_id TEXT NOT NULL, node_id TEXT NOT NULL, instances BIGINT NOT NULL,
    PRIMARY KEY (load_id, source_id, metric, subject_id, node_id),
    FOREIGN KEY (load_id, source_id) REFERENCES curriculum_sources (load_id, source_id)
);
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
ALTER TABLE atlas_sources ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ;
ALTER TABLE atlas_loads ADD COLUMN IF NOT EXISTS loader_version INTEGER;
CREATE OR REPLACE VIEW atlas_cohort_summary AS
WITH counts AS (
SELECT p.load_id, p.pipeline_key, p.cohort, p.population, p.sample_count,
    COUNT(i.task_id) FILTER (WHERE i.status = 'reviewed') AS reviewed,
    COUNT(i.task_id) FILTER (WHERE NOT i.has_task) AS missing_evidence,
    COUNT(i.task_id) FILTER (WHERE jsonb_array_length(i.omissions) > 0) AS packets_with_omissions,
    jsonb_build_object(
        'good', COUNT(i.task_id) FILTER (WHERE i.status = 'reviewed' AND i.verdict = 'good'),
        'some_issues', COUNT(i.task_id) FILTER (WHERE i.status = 'reviewed' AND i.verdict = 'some_issues'),
        'bad', COUNT(i.task_id) FILTER (WHERE i.status = 'reviewed' AND i.verdict = 'bad'),
        'unknown', COUNT(i.task_id) FILTER (WHERE i.status = 'reviewed' AND i.verdict = 'unknown')
    ) AS verdict_counts
FROM atlas_populations p
LEFT JOIN atlas_items i
    ON i.load_id = p.load_id AND i.pipeline_key = p.pipeline_key AND i.cohort = p.cohort
GROUP BY p.load_id, p.pipeline_key, p.cohort, p.population, p.sample_count
)
SELECT load_id, pipeline_key, cohort, population, sample_count, reviewed,
    missing_evidence, packets_with_omissions, verdict_counts,
    CASE
        WHEN reviewed = 0 THEN 'unknown'
        WHEN 10 * (verdict_counts->>'good')::bigint >= 9 * reviewed THEN 'good'
        WHEN 2 * ((verdict_counts->>'some_issues')::bigint + (verdict_counts->>'bad')::bigint) >= reviewed THEN 'bad'
        ELSE 'some_issues'
    END AS grade,
    (verdict_counts->>'good')::double precision / NULLIF(reviewed, 0) AS good_fraction,
    CASE
        WHEN reviewed = 0 THEN 'No reviewed sample available'
        WHEN 10 * (verdict_counts->>'good')::bigint >= 9 * reviewed THEN 'At least 90% of reviewed tasks are clean'
        WHEN 2 * ((verdict_counts->>'some_issues')::bigint + (verdict_counts->>'bad')::bigint) >= reviewed
            THEN 'At least 50% of reviewed tasks have issues'
        ELSE 'Reviewed sample falls between the Good and Bad thresholds'
    END AS grade_reason
FROM counts;
"""
