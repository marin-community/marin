# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""PostgreSQL statements for Echo's lexical/vector reciprocal-rank fusion."""

from collections.abc import Sequence

import sqlalchemy

RRF_K = 60
LEXICAL_WEIGHT = 2.0
MIN_CANDIDATES = 40
CANDIDATE_MULTIPLIER = 4


def candidate_limit(limit: int) -> int:
    return max(MIN_CANDIDATES, limit * CANDIDATE_MULTIPLIER)


def _where(clauses: Sequence[str], extra: str) -> str:
    return " AND ".join([*clauses, extra])


def chunk_search_statement(filter_clauses: Sequence[str] = ()) -> sqlalchemy.TextClause:
    semantic_where = _where(filter_clauses, "c.embedding IS NOT NULL")
    lexical_where = _where(filter_clauses, "c.search_document @@ input.query")
    return sqlalchemy.text(
        f"""
        WITH input AS (
            SELECT
                CAST(:embedding AS vector) AS embedding,
                websearch_to_tsquery('english', :q) AS query
        ),
        semantic AS (
            SELECT
                c.id,
                row_number() OVER (ORDER BY c.embedding <=> input.embedding) AS rank,
                c.embedding <=> input.embedding AS distance
            FROM chunks AS c CROSS JOIN input
            WHERE {semantic_where}
            ORDER BY c.embedding <=> input.embedding
            LIMIT :candidate_limit
        ),
        lexical AS (
            SELECT
                c.id,
                row_number() OVER (
                    ORDER BY ts_rank_cd(c.search_document, input.query, 32) DESC
                ) AS rank,
                ts_rank_cd(c.search_document, input.query, 32) AS lexical_score
            FROM chunks AS c CROSS JOIN input
            WHERE {lexical_where}
            ORDER BY lexical_score DESC
            LIMIT :candidate_limit
        ),
        candidate_ids AS (
            SELECT id FROM semantic
            UNION
            SELECT id FROM lexical
        )
        SELECT
            c.*,
            semantic.distance,
            lexical.lexical_score,
            (
                coalesce(1.0 / ({RRF_K} + semantic.rank), 0.0) +
                coalesce({LEXICAL_WEIGHT} / ({RRF_K} + lexical.rank), 0.0)
            ) AS score
        FROM candidate_ids
        JOIN chunks AS c USING (id)
        LEFT JOIN semantic USING (id)
        LEFT JOIN lexical USING (id)
        ORDER BY score DESC, c.date DESC NULLS LAST
        LIMIT :limit
        """
    )


def wiki_search_statement() -> sqlalchemy.TextClause:
    return sqlalchemy.text(
        f"""
        WITH input AS (
            SELECT
                CAST(:embedding AS vector) AS embedding,
                websearch_to_tsquery('english', :q) AS query
        ),
        semantic AS (
            SELECT
                w.id,
                row_number() OVER (ORDER BY w.embedding <=> input.embedding) AS rank,
                w.embedding <=> input.embedding AS distance
            FROM wiki_entries AS w CROSS JOIN input
            ORDER BY w.embedding <=> input.embedding
            LIMIT :candidate_limit
        ),
        lexical AS (
            SELECT
                w.id,
                row_number() OVER (
                    ORDER BY ts_rank_cd(w.search_document, input.query, 32) DESC
                ) AS rank,
                ts_rank_cd(w.search_document, input.query, 32) AS lexical_score
            FROM wiki_entries AS w CROSS JOIN input
            WHERE w.search_document @@ input.query
            ORDER BY lexical_score DESC
            LIMIT :candidate_limit
        ),
        candidate_ids AS (
            SELECT id FROM semantic
            UNION
            SELECT id FROM lexical
        )
        SELECT
            w.*,
            semantic.distance,
            lexical.lexical_score,
            (
                coalesce(1.0 / ({RRF_K} + semantic.rank), 0.0) +
                coalesce({LEXICAL_WEIGHT} / ({RRF_K} + lexical.rank), 0.0)
            ) AS score
        FROM candidate_ids
        JOIN wiki_entries AS w USING (id)
        LEFT JOIN semantic USING (id)
        LEFT JOIN lexical USING (id)
        ORDER BY score DESC, w.updated_at DESC
        LIMIT :limit
        """
    )
