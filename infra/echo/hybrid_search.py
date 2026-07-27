# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""PostgreSQL statements for Echo's lexical/vector reciprocal-rank fusion."""

from collections.abc import Sequence

import sqlalchemy
from search_config import TEXT_SEARCH_CONFIG, TS_RANK_NORMALIZATION

RRF_K = 60
LEXICAL_WEIGHT = 2.0
MIN_CANDIDATES = 40
CANDIDATE_MULTIPLIER = 4
HNSW_ITERATIVE_SCAN = sqlalchemy.text("SET hnsw.iterative_scan = relaxed_order")


def candidate_limit(limit: int) -> int:
    return max(MIN_CANDIDATES, limit * CANDIDATE_MULTIPLIER)


def _where(clauses: Sequence[str], extra: str) -> str:
    return " AND ".join([*clauses, extra])


def _search_statement(
    table: str,
    alias: str,
    semantic_where: str | None,
    lexical_where: str,
    final_order: str,
) -> sqlalchemy.TextClause:
    semantic_filter = f"WHERE {semantic_where}" if semantic_where else ""
    return sqlalchemy.text(
        f"""
        WITH input AS (
            SELECT
                CAST(:embedding AS vector) AS embedding,
                websearch_to_tsquery('{TEXT_SEARCH_CONFIG}', :q) AS query
        ),
        semantic AS (
            SELECT
                {alias}.id,
                row_number() OVER (ORDER BY {alias}.embedding <=> input.embedding) AS rank,
                {alias}.embedding <=> input.embedding AS distance
            FROM {table} AS {alias} CROSS JOIN input
            {semantic_filter}
            ORDER BY {alias}.embedding <=> input.embedding
            LIMIT :candidate_limit
        ),
        lexical AS (
            SELECT
                {alias}.id,
                row_number() OVER (
                    ORDER BY ts_rank_cd(
                        {alias}.search_document, input.query, {TS_RANK_NORMALIZATION}
                    ) DESC
                ) AS rank,
                ts_rank_cd(
                    {alias}.search_document, input.query, {TS_RANK_NORMALIZATION}
                ) AS lexical_score
            FROM {table} AS {alias} CROSS JOIN input
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
            {alias}.*,
            semantic.distance,
            lexical.lexical_score,
            (
                coalesce(1.0 / ({RRF_K} + semantic.rank), 0.0) +
                coalesce({LEXICAL_WEIGHT} / ({RRF_K} + lexical.rank), 0.0)
            ) AS score
        FROM candidate_ids
        JOIN {table} AS {alias} USING (id)
        LEFT JOIN semantic USING (id)
        LEFT JOIN lexical USING (id)
        ORDER BY score DESC, {final_order}
        LIMIT :limit
        """
    )


def chunk_search_statement(filter_clauses: Sequence[str] = ()) -> sqlalchemy.TextClause:
    return _search_statement(
        table="chunks",
        alias="c",
        semantic_where=_where(filter_clauses, "c.embedding IS NOT NULL"),
        lexical_where=_where(filter_clauses, "c.search_document @@ input.query"),
        final_order="c.date DESC NULLS LAST",
    )


def wiki_search_statement() -> sqlalchemy.TextClause:
    return _search_statement(
        table="wiki_entries",
        alias="w",
        semantic_where=None,
        lexical_where="w.search_document @@ input.query",
        final_order="w.updated_at DESC",
    )
