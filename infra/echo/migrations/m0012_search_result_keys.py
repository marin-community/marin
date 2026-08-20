# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Give stored search results durable grading keys and retain reranker scores."""

import sqlalchemy

DDL = """
ALTER TABLE search_execution_results
    ADD COLUMN id BIGINT GENERATED ALWAYS AS IDENTITY;
ALTER TABLE search_execution_results
    DROP CONSTRAINT search_execution_results_pkey;
ALTER TABLE search_execution_results
    ADD PRIMARY KEY (id);
ALTER TABLE search_execution_results
    ADD CONSTRAINT search_execution_results_execution_rank UNIQUE (execution_id, rank);
ALTER TABLE search_execution_results
    ADD COLUMN rerank_score DOUBLE PRECISION;

ALTER TABLE search_feedback_grades
    ADD COLUMN search_result_id BIGINT REFERENCES search_execution_results(id) ON DELETE SET NULL;
UPDATE search_feedback_grades AS grade
SET search_result_id = result.id
FROM search_feedback AS feedback, search_execution_results AS result
WHERE feedback.id = grade.feedback_id
  AND result.execution_id = feedback.execution_id
  AND result.result_id = grade.result_id;
ALTER TABLE search_feedback_grades
    ADD CONSTRAINT search_feedback_grades_search_result UNIQUE (feedback_id, search_result_id);
CREATE INDEX idx_search_feedback_grades_search_result_id
    ON search_feedback_grades (search_result_id);

GRANT USAGE, SELECT ON SEQUENCE search_execution_results_id_seq
    TO "echo-api@hai-gcp-models.iam";
"""


def upgrade(conn: sqlalchemy.Connection) -> None:
    for statement in DDL.strip().split(";"):
        if statement.strip():
            conn.execute(sqlalchemy.text(statement))
