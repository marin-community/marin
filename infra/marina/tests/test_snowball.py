# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Snowball summaries and curriculum search against real Postgres."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Barrier, Lock
from types import ModuleType
from typing import cast
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from marina.applets import AppletServices
from marina.db import UrlDatabase, engine_for
from rigging.server_auth import VerifiedIdentity, identity_scope
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine

from infra.marina.applets.snowball_conversation_lab.server import app as lab
from infra.marina.applets.snowball_curriculum.server import app as curriculum
from infra.marina.applets.snowball_token_comparison.server import app as comparison

OWNER = VerifiedIdentity(user_id="owner@example.com", role="user")


@dataclass
class SnowballServices:
    database_engine: Engine

    def engine(self) -> Engine:
        return self.database_engine


@dataclass
class ConversationApp:
    module: ModuleType
    client: TestClient
    engine: Engine


@pytest.fixture
def snowball_services(database_url):
    schema = "snowball_test_" + uuid4().hex
    engine = engine_for(UrlDatabase(database_url), schema)
    try:
        yield cast(AppletServices, SnowballServices(engine))
    finally:
        engine.dispose()
        admin = create_engine(database_url)
        with admin.begin() as connection:
            connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()


@pytest.fixture(params=[lab, comparison], ids=["conversation_lab", "token_comparison"])
def conversation(request, snowball_services):
    module = request.param
    with snowball_services.engine().begin() as connection:
        module.migrate(connection)
        connection.execute(
            text(
                """INSERT INTO runs (run_id,owner,config,token_cap,concurrency,state)
                VALUES ('one',:owner,'{}',1000,2,'running'), ('two',:owner,'{}',1000,2,'running')"""
            ),
            {"owner": OWNER.user_id},
        )
        connection.execute(
            text(
                """INSERT INTO cases (case_id,run_id,title,category,mode,state,slot,round,seed,review)
                VALUES ('chat','one','Chat','reasoning','chat','complete',0,0,'{}',
                    '{"assessments":[{"model":"candidate","score": 3}]}'),
                ('agent','one','Agent','coding','agentic','generating',1,0,'{}','{}'),
                ('other','two','Other','coding','chat','error',0,0,'{}','{}')"""
            )
        )
    with TestClient(module.create_api(snowball_services)) as client:
        yield ConversationApp(module, client, snowball_services.engine())


def summary(client, **parameters):
    with identity_scope(OWNER):
        response = client.get("/summary", params=parameters)
    assert response.status_code == 200, response.text
    return response.json()


def test_summary_cache_keeps_runs_and_modes_separate_then_refreshes(conversation, monkeypatch):
    module, client, engine = conversation.module, conversation.client, conversation.engine
    clock = [100.0]
    monkeypatch.setattr(module, "monotonic", lambda: clock[0])
    initial = summary(client, run_id="one")
    assert (initial["cases"], initial["completed"], initial["active"]) == (2, 1, 1)
    assert summary(client, run_id="one", mode="chat")["cases"] == 1
    assert summary(client, run_id="one", mode="agentic")["active"] == 1
    assert summary(client, run_id="two")["errors"] == 1
    with engine.begin() as connection:
        connection.execute(text("UPDATE cases SET state='complete' WHERE case_id='agent'"))
    assert summary(client, run_id="one") == initial
    clock[0] += 9
    refreshed = summary(client, run_id="one")
    assert (refreshed["cases"], refreshed["completed"], refreshed["active"]) == (2, 2, 0)
    assert refreshed["scores"] == [{"model": "candidate", "category": "reasoning", "score": 3.0, "n": 1}]


def test_concurrent_summary_misses_share_database_aggregations(conversation):
    client, engine = conversation.client, conversation.engine
    ready = Barrier(8)
    lock = Lock()
    aggregate_queries = 0

    def count_aggregation(_connection, _cursor, statement, _parameters, _context, _many):
        nonlocal aggregate_queries
        if "jsonb_array_elements" in statement:
            with lock:
                aggregate_queries += 1

    def request(_index):
        ready.wait()
        return summary(client, run_id="one")

    event.listen(engine, "before_cursor_execute", count_aggregation)
    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(request, range(8)))
    finally:
        event.remove(engine, "before_cursor_execute", count_aggregation)
    assert all(result["cases"] == 2 and result["completed"] == 1 for result in results)
    # Eight simultaneous viewers should cost at most one set of JSON aggregations.
    assert 0 < aggregate_queries <= 3


def test_cached_summary_still_checks_worker_and_session_access(conversation):
    module, client, engine = conversation.module, conversation.client, conversation.engine
    assert summary(client, run_id="one")["cases"] == 2
    worker = VerifiedIdentity(user_id="worker@example.iam.gserviceaccount.com", role="user")
    with identity_scope(worker):
        denied = client.get("/summary", params={"run_id": "one"})
    assert denied.status_code == 403
    with engine.begin() as connection:
        connection.execute(
            text("UPDATE runs SET worker_session_hash=:hash WHERE run_id='two'"),
            {"hash": module.key_hash("two-session")},
        )
    denied = client.get("/summary", params={"run_id": "one"}, headers={"X-Lab-Session-Key": "two-session"})
    assert denied.status_code == 403
    allowed = client.get("/summary", params={"run_id": "two"}, headers={"X-Lab-Session-Key": "two-session"})
    assert allowed.status_code == 200
    with engine.begin() as connection:
        connection.execute(text("UPDATE runs SET worker_session_hash=NULL WHERE run_id='two'"))
    revoked = client.get("/summary", params={"run_id": "two"}, headers={"X-Lab-Session-Key": "two-session"})
    assert revoked.status_code == 401


def test_summary_preserves_private_and_public_read_access(conversation):
    module, client = conversation.module, conversation.client
    response = client.get("/summary", params={"run_id": "one"})
    assert response.status_code == (401 if module is lab else 200)


def test_curriculum_index_preserves_case_insensitive_literal_search(snowball_services):
    engine = snowball_services.engine()
    with engine.begin() as connection:
        # Extensions belong to the server; release migrations only create application objects.
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm WITH SCHEMA public"))
        curriculum.migrate(connection)
        curriculum.migrate(connection)
        connection.execute(text("INSERT INTO pool_loads VALUES ('load','{}')"))
        connection.execute(text("UPDATE pool_current SET load_id='load' WHERE name='latest'"))
        connection.execute(
            text(
                """INSERT INTO pool_tasks
                (load_id,contract_id,task_id,source_id,family,difficulty,preview,search_text,detail)
                VALUES ('load','literal','one','source','family','easy','preview','Alpha_100%','{}'),
                    ('load','wildcard','two','source','family','easy','preview','AlphaZ100anything','{}')"""
            )
        )
    with TestClient(curriculum.create_api(snowball_services)) as client:
        response = client.get("/tasks", params={"q": "aLPHA_100%"})
    assert response.status_code == 200, response.text
    assert response.json()["total"] == 1
    assert [row["contract_id"] for row in response.json()["tasks"]] == ["literal"]
