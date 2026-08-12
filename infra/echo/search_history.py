# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persist Echo search executions."""

from dataclasses import dataclass
from typing import Literal

import schema
import sqlalchemy

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


def normalize_query(query: str) -> str:
    return " ".join(query.casefold().split())


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


def insert_execution(conn: sqlalchemy.Connection, record: SearchExecutionRecord) -> int:
    """Insert one execution and its ranked result snapshot."""
    row = conn.execute(
        schema.search_executions.insert().values(**execution_values(record)).returning(schema.search_executions.c.id)
    ).first()
    assert row is not None
    execution_id = row.id
    if record.results:
        conn.execute(
            schema.search_execution_results.insert().values(
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
                    }
                    for rank, result in enumerate(record.results, start=1)
                ]
            )
        )
    return execution_id
