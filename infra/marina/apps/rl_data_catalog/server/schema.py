# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Postgres storage for one staged curation catalog load."""

from .cohort_sql import COHORT_PROJECTION_SQL, cohort_counts_sql

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS atlas_review_batches (
    load_id TEXT NOT NULL,pipeline_key TEXT NOT NULL,batch_key TEXT NOT NULL,record JSONB NOT NULL,
    chunk_count INTEGER NOT NULL DEFAULT 0,PRIMARY KEY(load_id,pipeline_key,batch_key)
);
ALTER TABLE atlas_review_batches ADD COLUMN IF NOT EXISTS chunk_count INTEGER NOT NULL DEFAULT 0;
CREATE TABLE IF NOT EXISTS atlas_review_batch_chunks (
    load_id TEXT NOT NULL,pipeline_key TEXT NOT NULL,batch_key TEXT NOT NULL,chunk_index INTEGER NOT NULL,
    payload TEXT NOT NULL,PRIMARY KEY(load_id,pipeline_key,batch_key,chunk_index),
    FOREIGN KEY(load_id,pipeline_key,batch_key) REFERENCES atlas_review_batches(load_id,pipeline_key,batch_key)
);
CREATE TABLE IF NOT EXISTS atlas_review_batch_tasks (
    load_id TEXT NOT NULL,pipeline_key TEXT NOT NULL,task_id TEXT NOT NULL,batch_key TEXT NOT NULL,
    PRIMARY KEY(load_id,pipeline_key,task_id),
    FOREIGN KEY(load_id,pipeline_key,batch_key) REFERENCES atlas_review_batches(load_id,pipeline_key,batch_key)
);
CREATE TABLE IF NOT EXISTS atlas_rollout_sets (
    measurement_id TEXT PRIMARY KEY, metadata JSONB NOT NULL
);
CREATE TABLE IF NOT EXISTS atlas_rollout_attempts (
    measurement_id TEXT NOT NULL, attempt_id TEXT NOT NULL, task_id TEXT NOT NULL,
    summary JSONB NOT NULL, detail JSONB NOT NULL,
    PRIMARY KEY (measurement_id,attempt_id)
);
CREATE INDEX IF NOT EXISTS atlas_rollout_task_order
    ON atlas_rollout_attempts (measurement_id,task_id,attempt_id);
CREATE TABLE IF NOT EXISTS atlas_review_current (
    load_id TEXT NOT NULL, source_id TEXT NOT NULL, pipeline_key TEXT NOT NULL,
    receipt JSONB NOT NULL, PRIMARY KEY (load_id,source_id)
);
CREATE TABLE IF NOT EXISTS atlas_raw_reviews (
    load_id TEXT NOT NULL, pipeline_key TEXT NOT NULL, task_id TEXT NOT NULL,
    record JSONB NOT NULL, reviewer JSONB, PRIMARY KEY (load_id,pipeline_key,task_id)
);
CREATE TABLE IF NOT EXISTS atlas_derived_contracts (
    source_id TEXT NOT NULL,receipt_id TEXT NOT NULL,receipt JSONB NOT NULL,
    PRIMARY KEY (source_id,receipt_id)
);
CREATE TABLE IF NOT EXISTS atlas_difficulty (
    measurement_id TEXT PRIMARY KEY, source_id TEXT NOT NULL,
    protocol_id TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL,
    measurement JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS atlas_difficulty_source ON atlas_difficulty (source_id, protocol_id);
CREATE TABLE IF NOT EXISTS atlas_difficulty_seals (
    snapshot_id TEXT PRIMARY KEY, manifest_uri TEXT NOT NULL, manifest_sha256 TEXT NOT NULL,
    manifest JSONB NOT NULL
);
CREATE TABLE IF NOT EXISTS atlas_difficulty_seal_cells (
    snapshot_id TEXT NOT NULL REFERENCES atlas_difficulty_seals(snapshot_id), source_id TEXT NOT NULL,
    solver TEXT NOT NULL, registry_revision TEXT NOT NULL, cell JSONB NOT NULL,
    PRIMARY KEY (snapshot_id,source_id,solver)
);
CREATE TABLE IF NOT EXISTS atlas_difficulty_seal_loads (
    load_id TEXT PRIMARY KEY, snapshot_id TEXT NOT NULL REFERENCES atlas_difficulty_seals(snapshot_id)
);
CREATE TABLE IF NOT EXISTS atlas_difficulty_coverage (
    load_id TEXT PRIMARY KEY, coverage JSONB NOT NULL
);
CREATE TABLE IF NOT EXISTS atlas_loads (
    load_id TEXT PRIMARY KEY, catalog_root TEXT NOT NULL,
    snapshot_id TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL,
    loaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS atlas_current (
    name TEXT PRIMARY KEY, load_id TEXT
);
ALTER TABLE atlas_loads ADD COLUMN IF NOT EXISTS review_completeness_version INTEGER NOT NULL DEFAULT 0;
ALTER TABLE atlas_loads ADD COLUMN IF NOT EXISTS display_policy TEXT NOT NULL DEFAULT 'strict';
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
CREATE TABLE IF NOT EXISTS curriculum_source_areas (
    load_id TEXT NOT NULL, source_id TEXT NOT NULL, version INTEGER NOT NULL, area JSONB,
    PRIMARY KEY (load_id,source_id)
);
CREATE TABLE IF NOT EXISTS curriculum_confidence (
    load_id TEXT NOT NULL, scope_id TEXT NOT NULL, metric TEXT NOT NULL,
    subject_id TEXT NOT NULL, node_id TEXT NOT NULL, instances BIGINT[] NOT NULL,
    PRIMARY KEY (load_id, scope_id, metric, subject_id, node_id)
);
CREATE TABLE IF NOT EXISTS atlas_sources (
    load_id TEXT NOT NULL, id TEXT NOT NULL, declaration JSONB NOT NULL,
    assessment_key TEXT, latest_attempt_key TEXT, latest_attempt_status TEXT NOT NULL,
    reason TEXT, curation JSONB, diagnostics JSONB NOT NULL,
    PRIMARY KEY (load_id, id)
);
ALTER TABLE atlas_sources ADD COLUMN IF NOT EXISTS backend_verification_pending JSONB;
ALTER TABLE atlas_sources ADD COLUMN IF NOT EXISTS sample_pin_stale JSONB;
ALTER TABLE atlas_sources ADD COLUMN IF NOT EXISTS method_variants JSONB;
ALTER TABLE atlas_sources ADD COLUMN IF NOT EXISTS source_fingerprint JSONB;
ALTER TABLE atlas_sources ADD COLUMN IF NOT EXISTS duplicate_group JSONB;
ALTER TABLE atlas_sources ADD COLUMN IF NOT EXISTS review_completeness JSONB;
CREATE TABLE IF NOT EXISTS atlas_identity_surveys (
    load_id TEXT PRIMARY KEY, survey_id TEXT NOT NULL, metadata JSONB NOT NULL
);
CREATE TABLE IF NOT EXISTS atlas_duplicate_reviews (
    load_id TEXT NOT NULL, left_source TEXT NOT NULL, right_source TEXT NOT NULL, review JSONB NOT NULL,
    PRIMARY KEY (load_id, left_source, right_source)
);
CREATE TABLE IF NOT EXISTS atlas_identity_measurements (
    load_id TEXT NOT NULL, measurement_id TEXT NOT NULL, metadata JSONB NOT NULL,
    PRIMARY KEY (load_id,measurement_id)
);
CREATE TABLE IF NOT EXISTS atlas_identity_measurement_reviews (
    load_id TEXT NOT NULL, measurement_id TEXT NOT NULL, left_source TEXT NOT NULL,
    right_source TEXT NOT NULL, review JSONB NOT NULL,
    PRIMARY KEY (load_id,measurement_id,left_source,right_source)
);
CREATE TABLE IF NOT EXISTS atlas_identity_measurement_current (
    load_id TEXT PRIMARY KEY, measurement_id TEXT NOT NULL
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
CREATE TABLE IF NOT EXISTS atlas_cohort_summaries (
    load_id TEXT NOT NULL, pipeline_key TEXT NOT NULL, cohort TEXT NOT NULL,
    version INTEGER NOT NULL, summary JSONB NOT NULL,
    PRIMARY KEY (load_id,pipeline_key,cohort),
    FOREIGN KEY (load_id,pipeline_key,cohort)
        REFERENCES atlas_populations (load_id,pipeline_key,cohort) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS atlas_items (
    load_id TEXT NOT NULL, pipeline_key TEXT NOT NULL, task_id TEXT NOT NULL,
    cohort TEXT NOT NULL, status TEXT NOT NULL, verdict TEXT NOT NULL,
    explanation TEXT NOT NULL, evidence_ids JSONB NOT NULL, packet_uri TEXT NOT NULL,
    has_task BOOLEAN NOT NULL, omissions JSONB NOT NULL, exclusion_reason TEXT,
    PRIMARY KEY (load_id, pipeline_key, task_id)
);
ALTER TABLE atlas_items ADD COLUMN IF NOT EXISTS confidence TEXT;
ALTER TABLE atlas_items ADD COLUMN IF NOT EXISTS reference_status TEXT;
ALTER TABLE atlas_items ADD COLUMN IF NOT EXISTS defects JSONB;
CREATE INDEX IF NOT EXISTS atlas_items_cohort ON atlas_items (load_id, pipeline_key, cohort);
ALTER TABLE atlas_items ADD COLUMN IF NOT EXISTS provenance TEXT;
ALTER TABLE atlas_items ADD COLUMN IF NOT EXISTS replaced_task_sha256 TEXT;
ALTER TABLE atlas_items ADD COLUMN IF NOT EXISTS repair_advisories JSONB;
-- Whole-population FULL reviews make atlas_items far larger than SAMPLE panels;
-- the browse pages order, filter and count through these indexes.
CREATE INDEX IF NOT EXISTS atlas_items_cohort_order ON atlas_items (load_id, pipeline_key, cohort, task_id);
CREATE INDEX IF NOT EXISTS atlas_items_verdict ON atlas_items (load_id, pipeline_key, verdict);
CREATE INDEX IF NOT EXISTS atlas_items_defects ON atlas_items USING GIN (defects);
CREATE TABLE IF NOT EXISTS atlas_audit_samples (
    sample_id TEXT PRIMARY KEY, source_id TEXT NOT NULL, pipeline_key TEXT NOT NULL,
    load_id TEXT NOT NULL, strategy TEXT NOT NULL, seed TEXT NOT NULL,
    sample_size INTEGER NOT NULL, strata JSONB NOT NULL,
    created_by TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS atlas_audit_sample_tasks (
    sample_id TEXT NOT NULL, task_id TEXT NOT NULL, verdict TEXT NOT NULL, defects JSONB NOT NULL,
    PRIMARY KEY (sample_id, task_id),
    FOREIGN KEY (sample_id) REFERENCES atlas_audit_samples (sample_id)
);
CREATE TABLE IF NOT EXISTS atlas_audit_judgments (
    sample_id TEXT NOT NULL, task_id TEXT NOT NULL, user_id TEXT NOT NULL,
    agree BOOLEAN NOT NULL, corrected_verdict TEXT, corrected_defects JSONB,
    note TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (sample_id, task_id, user_id)
);
CREATE TABLE IF NOT EXISTS atlas_packets (
    load_id TEXT NOT NULL, pipeline_key TEXT NOT NULL, task_id TEXT NOT NULL,
    packet TEXT NOT NULL,
    PRIMARY KEY (load_id, pipeline_key, task_id)
);
CREATE INDEX IF NOT EXISTS atlas_packets_pipeline_task ON atlas_packets (pipeline_key,task_id);
CREATE TABLE IF NOT EXISTS atlas_backend_surveys (
    load_id TEXT PRIMARY KEY, survey_id TEXT NOT NULL, survey JSONB NOT NULL
);
CREATE TABLE IF NOT EXISTS atlas_backend_sources (
    load_id TEXT NOT NULL, source_id TEXT NOT NULL, capabilities JSONB NOT NULL,
    PRIMARY KEY (load_id, source_id)
);
CREATE TABLE IF NOT EXISTS atlas_backend_cells (
    load_id TEXT NOT NULL, backend TEXT NOT NULL, requirement TEXT NOT NULL, cell JSONB NOT NULL,
    PRIMARY KEY (load_id, backend, requirement)
);
"""


PROJECTION_REFRESH_SEAL_SQL = """
CREATE TABLE IF NOT EXISTS atlas_projection_refresh_seals (
    name TEXT PRIMARY KEY, load_id TEXT NOT NULL, audit JSONB NOT NULL,
    sealed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

SCHEMA_SQL += PROJECTION_REFRESH_SEAL_SQL

SCHEMA_SQL += (
    "CREATE OR REPLACE VIEW atlas_cohort_summary AS WITH counts AS (\n"
    + cohort_counts_sql()
    + ")\n"
    + COHORT_PROJECTION_SQL
)
