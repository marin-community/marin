# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Keep confirmed verifier defects authoritative across review publications."""

from sqlalchemy import text
from sqlalchemy.engine import Connection


def migrate_verifier_policy(connection: Connection) -> None:
    connection.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS catalog_verifier_issues (
                source_id TEXT NOT NULL REFERENCES catalog_sources(id),
                issue_url TEXT NOT NULL,
                review_id TEXT NOT NULL REFERENCES catalog_reviews(id),
                status TEXT NOT NULL CHECK (status IN ('open', 'resolved')),
                resolution_review_id TEXT REFERENCES catalog_reviews(id),
                created_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL,
                PRIMARY KEY (source_id, issue_url),
                CHECK (
                    (status = 'open' AND resolution_review_id IS NULL)
                    OR (status = 'resolved' AND resolution_review_id IS NOT NULL
                        AND resolution_review_id <> review_id)
                )
            )
            """
        )
    )
    connection.execute(
        text(
            """
            CREATE OR REPLACE FUNCTION enforce_verifier_quality() RETURNS TRIGGER AS $$
            BEGIN
                IF EXISTS (SELECT 1 FROM catalog_verifier_issues
                           WHERE source_id = NEW.id AND status = 'open') THEN
                    IF TG_OP = 'UPDATE' AND OLD.quality = 'bad' THEN
                        NEW.quality := 'bad';
                    ELSIF NEW.quality IS DISTINCT FROM 'bad' THEN
                        NEW.quality := 'some_issues';
                    END IF;
                    NEW.difficulty := NULL;
                END IF;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql
            """
        )
    )
    connection.execute(text("DROP TRIGGER IF EXISTS enforce_verifier_quality ON catalog_sources"))
    connection.execute(
        text(
            """
            CREATE TRIGGER enforce_verifier_quality
            BEFORE INSERT OR UPDATE ON catalog_sources
            FOR EACH ROW EXECUTE FUNCTION enforce_verifier_quality()
            """
        )
    )
    connection.execute(
        text(
            """
            CREATE OR REPLACE FUNCTION demote_verifier_source() RETURNS TRIGGER AS $$
            BEGIN
                IF NEW.status = 'open' THEN
                    UPDATE catalog_sources SET
                        quality = CASE WHEN quality = 'bad' THEN 'bad' ELSE 'some_issues' END,
                        difficulty = NULL
                    WHERE id = NEW.source_id;
                END IF;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql
            """
        )
    )
    connection.execute(text("DROP TRIGGER IF EXISTS demote_verifier_source ON catalog_verifier_issues"))
    connection.execute(
        text(
            """
            CREATE TRIGGER demote_verifier_source
            AFTER INSERT OR UPDATE ON catalog_verifier_issues
            FOR EACH ROW EXECUTE FUNCTION demote_verifier_source()
            """
        )
    )
    connection.execute(
        text(
            """
            CREATE OR REPLACE FUNCTION validate_verifier_resolution() RETURNS TRIGGER AS $$
            BEGIN
                IF NEW.status = 'resolved' AND NOT EXISTS (
                    SELECT 1 FROM catalog_reviews r
                    JOIN catalog_sources s ON s.id = r.source_id
                    WHERE r.id = NEW.resolution_review_id AND r.source_id = NEW.source_id
                        AND r.updated_at > NEW.created_at
                        AND EXISTS (
                            SELECT 1 FROM jsonb_array_elements(r.collection->'reviews') judgment,
                                jsonb_array_elements(judgment->'attributes'->'resolved_verifier_issues') resolution
                            WHERE resolution->>'issue_url' = NEW.issue_url
                                AND resolution ? 'verifier_revision'
                                AND (resolution->>'verifier_revision') IS NOT DISTINCT FROM
                                    (s.payload->>'verifier_revision')
                                AND resolution->>'dataset_revision' = COALESCE(
                                    s.payload->>'dataset_revision', s.payload->>'revision')
                                AND resolution->'fix_validated' = 'true'::jsonb
                        )
                        AND EXISTS (
                            SELECT 1 FROM jsonb_array_elements(r.collection->'reviews') judgment
                            WHERE judgment->>'method' = 'runtime_execution'
                                AND judgment->'tests_executed' = 'true'::jsonb
                                AND judgment->'attributes'->'verification'->>'status' = 'verified'
                        )
                ) THEN
                    RAISE EXCEPTION 'Resolve verifier defects with a fresh native review and applicable fix evidence';
                END IF;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql
            """
        )
    )
    connection.execute(text("DROP TRIGGER IF EXISTS validate_verifier_resolution ON catalog_verifier_issues"))
    connection.execute(
        text(
            """
            CREATE TRIGGER validate_verifier_resolution
            BEFORE INSERT OR UPDATE ON catalog_verifier_issues
            FOR EACH ROW EXECUTE FUNCTION validate_verifier_resolution()
            """
        )
    )
