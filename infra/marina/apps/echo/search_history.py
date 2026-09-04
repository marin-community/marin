# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persist Echo search executions."""

from dataclasses import dataclass
from typing import Literal

import sqlalchemy

from . import schema
from .search_config import normalize_query

SearchMode = Literal["federated", "activity", "grep"]


@dataclass(frozen=True)
class SearchResultRecord:
    result_id: str
    domain: str
    title: str | None
    url: str
    snippet: str
    score: float
    distance: float | None
    lexical_score: float | None
    rerank_score: float | None


@dataclass(frozen=True)
class StoredSearchExecution:
    id: int
    search_result_ids: tuple[int, ...]


@dataclass(frozen=True)
class SearchExecutionRecord:
    query: str
    mode: SearchMode
    domains: tuple[str, ...]
    filters: dict[str, object]
    requested_limit: int
    returned_count: int
    duration_ms: float
    author: str | None = None
    repository_commit: str | None = None
    service_revision: str | None = None
    results: tuple[SearchResultRecord, ...] = ()


def execution_values(record: SearchExecutionRecord) -> dict[str, object]:
    return {
        "author": record.author,
        "query": record.query,
        "normalized_query": normalize_query(record.query),
        "mode": record.mode,
        "domains": list(record.domains),
        "filters": record.filters,
        "requested_limit": record.requested_limit,
        "returned_count": record.returned_count,
        "duration_ms": record.duration_ms,
        "repository_commit": record.repository_commit,
        "service_revision": record.service_revision,
    }


def insert_execution(conn: sqlalchemy.Connection, record: SearchExecutionRecord) -> StoredSearchExecution:
    """Insert one execution and its ranked result snapshot.

    Returns:
        The new search execution and its result-row IDs in rank order.
    """
    row = conn.execute(
        schema.search_executions.insert().values(**execution_values(record)).returning(schema.search_executions.c.id)
    ).first()
    assert row is not None
    execution_id = row.id
    search_result_ids: tuple[int, ...] = ()
    if record.results:
        stored_results = conn.execute(
            schema.search_execution_results.insert()
            .values(
                [
                    {
                        "execution_id": execution_id,
                        "rank": rank,
                        "result_id": result.result_id,
                        "domain": result.domain,
                        "title": result.title,
                        "url": result.url,
                        "snippet": result.snippet,
                        "score": result.score,
                        "distance": result.distance,
                        "lexical_score": result.lexical_score,
                        "rerank_score": result.rerank_score,
                    }
                    for rank, result in enumerate(record.results, start=1)
                ]
            )
            .returning(schema.search_execution_results.c.id, schema.search_execution_results.c.rank)
        ).all()
        search_result_ids = tuple(row.id for row in sorted(stored_results, key=lambda row: row.rank))
    return StoredSearchExecution(execution_id, search_result_ids)
