# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""PostgreSQL statements for Echo's lexical/vector reciprocal-rank fusion."""

from collections.abc import Sequence

import sqlalchemy
from search_config import (
    FILE_ADDITIONAL_HIT_MAX_FRACTION,
    FILE_ADDITIONAL_HIT_WEIGHT,
    MAX_SEMANTIC_DISTANCE,
    RRF_K,
    TEXT_SEARCH_CONFIG,
    TS_RANK_NORMALIZATION,
)

HNSW_ITERATIVE_SCAN = sqlalchemy.text("SET hnsw.iterative_scan = relaxed_order")


def _where(clauses: Sequence[str], extra: str) -> str:
    return " AND ".join([*clauses, extra])


def _search_statement(
    table: str,
    alias: str,
    semantic_where: str | None,
    lexical_where: str,
    lexical_score: str,
    result_key: str,
    final_order: str,
    additional_hit_weight: float = 0.0,
    additional_hit_max_fraction: float = 0.0,
) -> sqlalchemy.TextClause:
    semantic_filter = f"WHERE {semantic_where}" if semantic_where else ""
    return sqlalchemy.text(
        f"""
        WITH input AS (
            SELECT websearch_to_tsquery('{TEXT_SEARCH_CONFIG}', :q) AS query
        ),
        semantic AS (
            -- Rank the nearest neighbours the HNSW index returns. The distance must compare
            -- the indexed column against the bind parameter directly, and the LIMIT must sit
            -- below the window: an `ORDER BY col <=> value` sourced from a joined CTE, or a
            -- row_number() over the whole table, both defeat the index and force a full scan.
            SELECT id, row_number() OVER (ORDER BY distance) AS rank, distance
            FROM (
                SELECT {alias}.id, {alias}.embedding <=> CAST(:embedding AS vector) AS distance
                FROM {table} AS {alias}
                {semantic_filter}
                ORDER BY {alias}.embedding <=> CAST(:embedding AS vector)
                LIMIT :candidate_limit
            ) AS nearest
        ),
        lexical AS (
            SELECT
                {alias}.id,
                row_number() OVER (
                    ORDER BY {lexical_score} DESC
                ) AS rank,
                {lexical_score} AS lexical_score
            FROM {table} AS {alias} CROSS JOIN input
            WHERE {lexical_where}
            ORDER BY lexical_score DESC
            LIMIT :candidate_limit
        ),
        candidate_ids AS (
            SELECT id FROM semantic
            UNION
            SELECT id FROM lexical
        ),
        fused AS (
            SELECT
                {alias}.*,
                semantic.distance,
                lexical.lexical_score,
                (
                    coalesce(:semantic_weight / ({RRF_K} + semantic.rank), 0.0) +
                    coalesce(:lexical_weight / ({RRF_K} + lexical.rank), 0.0)
                ) AS raw_score
            FROM candidate_ids
            JOIN {table} AS {alias} USING (id)
            LEFT JOIN semantic USING (id)
            LEFT JOIN lexical USING (id)
            WHERE lexical.id IS NOT NULL
                OR semantic.distance <= {MAX_SEMANTIC_DISTANCE}
        ),
        ranked AS (
            SELECT
                fused.*,
                row_number() OVER (
                    PARTITION BY {result_key}
                    ORDER BY raw_score DESC, {final_order}
                ) AS result_rank,
                sum(raw_score) OVER (PARTITION BY {result_key}) AS evidence_score
            FROM fused
        )
        SELECT
            *,
            raw_score + least(
                raw_score * {additional_hit_max_fraction},
                greatest(evidence_score - raw_score, 0.0) * {additional_hit_weight}
            ) AS score
        FROM ranked
        WHERE result_rank = 1
        ORDER BY score DESC, {final_order}
        LIMIT :limit
        """
    )


def full_text_score_expression(alias: str) -> str:
    return f"ts_rank_cd({alias}.search_document, input.query, {TS_RANK_NORMALIZATION})"


def chunk_search_statement(filter_clauses: Sequence[str] = ()) -> sqlalchemy.TextClause:
    return _search_statement(
        table="chunks",
        alias="c",
        semantic_where=_where(filter_clauses, "c.embedding IS NOT NULL"),
        lexical_where=_where(filter_clauses, "c.search_document @@ input.query"),
        lexical_score=full_text_score_expression("c"),
        result_key="id",
        final_order="date DESC NULLS LAST",
    )


def wiki_search_statement(filter_clauses: Sequence[str] = ()) -> sqlalchemy.TextClause:
    return _search_statement(
        table="wiki_entries",
        alias="w",
        semantic_where=" AND ".join(filter_clauses) or None,
        lexical_where=_where(filter_clauses, "w.search_document @@ input.query"),
        lexical_score=full_text_score_expression("w"),
        result_key="id",
        final_order="updated_at DESC",
    )


def repository_file_search_statement() -> sqlalchemy.TextClause:
    filters = ["r.repository = :repository", "r.branch = :branch"]
    filename = "regexp_replace(r.path, '^.*/', '')"
    filename_exact = f"{filename} ILIKE :exact ESCAPE '\\'"
    filename_match = f"{filename} ILIKE :substring ESCAPE '\\'"
    path_match = "r.path ILIKE :substring ESCAPE '\\'"
    text_match = "r.text ILIKE :substring ESCAPE '\\'"
    lexical_score = (
        f"{full_text_score_expression('r')}"
        f" + CASE WHEN {filename_exact} THEN 8.0 ELSE 0.0 END"
        f" + CASE WHEN {filename_match} THEN 5.0 ELSE 0.0 END"
        f" + CASE WHEN {path_match} THEN 3.0 ELSE 0.0 END"
        f" + CASE WHEN {text_match} THEN 1.0 ELSE 0.0 END"
    )
    return _search_statement(
        table="repository_file_chunks",
        alias="r",
        semantic_where=_where(filters, "r.embedding IS NOT NULL"),
        lexical_where=_where(
            filters,
            f"(r.search_document @@ input.query OR {filename_match} OR {path_match} OR {text_match})",
        ),
        lexical_score=lexical_score,
        result_key="path",
        final_order="path",
        additional_hit_weight=FILE_ADDITIONAL_HIT_WEIGHT,
        additional_hit_max_fraction=FILE_ADDITIONAL_HIT_MAX_FRACTION,
    )
