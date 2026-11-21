# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The work-log API and its slug migration against a real Postgres."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import cast

import pytest
import sqlalchemy
from fastapi import FastAPI
from fastapi.testclient import TestClient

from echo import app as echo
from echo import schema
from echo.migrations import m0001_init, m0002_chunk_identity, m0003_work_log_project_case

CALLER = "agent@openathena.ai"


@dataclass(frozen=True)
class PostgresServices:
    """Hands the app the test's engine, so API calls and direct inserts share one fresh schema."""

    engine_: sqlalchemy.Engine
    name: str = "echo"
    data_url: str = "memory://echo"

    def engine(self) -> sqlalchemy.Engine:
        return self.engine_


@pytest.fixture
def client(schema_engine: sqlalchemy.Engine) -> TestClient:
    with schema_engine.begin() as conn:
        echo.migrate(conn)
    api = cast(FastAPI, echo.create_api(PostgresServices(schema_engine)).app)
    api.dependency_overrides[echo.caller] = lambda: CALLER
    return TestClient(api)


def insert_entries(engine: sqlalchemy.Engine, *entries: tuple[str, datetime]) -> None:
    with engine.begin() as conn:
        for project, at in entries:
            conn.execute(schema.work_log.insert().values(author=CALLER, project=project, title="t", at=at))


def test_work_log_stores_slugs_in_lowercase_and_filters_in_any_case(client):
    added = client.post("/work_log", json={"project": " MarinSkyRL:Hero ", "title": "t"})
    assert added.status_code == 201
    assert added.json()["project"] == "marinskyrl:hero"

    entries = client.get("/work_log", params={"project": "MARINSKYRL:hero"}).json()

    assert [entry["project"] for entry in entries] == ["marinskyrl:hero"]


def test_work_log_projects_counts_entries_in_window_most_recent_first(client, schema_engine):
    now = datetime.now(UTC)
    insert_entries(
        schema_engine,
        ("harbor:issue-161", now - timedelta(days=2)),
        ("harbor:issue-161", now - timedelta(days=1)),
        ("vllm:serve-smoke", now - timedelta(hours=3)),
        ("evalchemy:pr-212", now - timedelta(days=40)),
    )

    projects = client.get("/work_log/projects").json()

    assert [(project["project"], project["entries"]) for project in projects] == [
        ("vllm:serve-smoke", 1),
        ("harbor:issue-161", 2),
    ]


def test_project_case_migration_lowercases_existing_slugs(schema_engine):
    with schema_engine.begin() as conn:
        m0001_init.upgrade(conn)
        m0002_chunk_identity.upgrade(conn)
    insert_entries(
        schema_engine,
        (" MarinSkyRL:Score-Centering ", datetime(2026, 10, 20, tzinfo=UTC)),
        ("harbor:issue-161", datetime(2026, 10, 21, tzinfo=UTC)),
    )

    with schema_engine.begin() as conn:
        m0003_work_log_project_case.upgrade(conn)
        projects = set(conn.execute(sqlalchemy.select(schema.work_log.c.project)).scalars())

    assert projects == {"marinskyrl:score-centering", "harbor:issue-161"}
