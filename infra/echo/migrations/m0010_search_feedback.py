# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Store agent judgments about federated search results."""

import sqlalchemy

MIN_GRADE = 0
MAX_GRADE = 10

DDL = f"""
CREATE TABLE search_feedback (
    id BIGINT GENERATED ALWAYS AS IDENTITY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
    author TEXT NOT NULL,
    query TEXT NOT NULL,
    note TEXT,
    PRIMARY KEY (id)
);
CREATE INDEX idx_search_feedback_created_at ON search_feedback (created_at DESC);

CREATE TABLE search_feedback_grades (
    feedback_id BIGINT NOT NULL REFERENCES search_feedback(id) ON DELETE CASCADE,
    result_id TEXT NOT NULL,
    grade SMALLINT NOT NULL,
    PRIMARY KEY (feedback_id, result_id),
    CONSTRAINT search_feedback_grades_range CHECK (grade BETWEEN {MIN_GRADE} AND {MAX_GRADE})
);
CREATE INDEX idx_search_feedback_grades_result_id ON search_feedback_grades (result_id);

GRANT SELECT ON search_feedback, search_feedback_grades
    TO "eng-all@openathena.ai";
GRANT SELECT ON search_feedback, search_feedback_grades
    TO "loom-vm@hai-gcp-models.iam";
GRANT SELECT, INSERT ON search_feedback, search_feedback_grades
    TO "echo-api@hai-gcp-models.iam";
GRANT USAGE, SELECT ON SEQUENCE search_feedback_id_seq
    TO "echo-api@hai-gcp-models.iam";
"""


def upgrade(conn: sqlalchemy.Connection) -> None:
    for statement in DDL.strip().split(";"):
        if statement.strip():
            conn.execute(sqlalchemy.text(statement))
