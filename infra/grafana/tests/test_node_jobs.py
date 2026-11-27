# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Node Details job row's bounded dataset: a node's task attempts and their usage."""

from dataclasses import replace
from types import SimpleNamespace as Record

import duckdb
import pytest
from config import ClusterTarget
from conftest import bridge_config, install_finelog_dialect_macros
from server import create_app
from starlette.testclient import TestClient

CLUSTER = "cw-us-east-02a"
NODE = "g7387bc"
MINUTE = 60_000
PARAMS = {"clusters": CLUSTER, "node": NODE, "from": 0, "to": 10 * MINUTE, "bucket_ms": MINUTE}


def _database() -> duckdb.DuckDBPyConnection:
    database = duckdb.connect()
    install_finelog_dialect_macros(database)
    key = "cluster VARCHAR, root_job_id VARCHAR, root_submitted_at_ms BIGINT, job_id VARCHAR, submitted_at_ms BIGINT"
    database.execute(
        f"""CREATE TABLE "iris.controller_db__task_attempts"({key}, task_id VARCHAR, attempt_id BIGINT,
               attempt_uid VARCHAR, state VARCHAR, started_at TIMESTAMP, finished_at TIMESTAMP,
               node_name VARCHAR, worker_id VARCHAR, ts TIMESTAMP, seq BIGINT)"""
    )
    database.execute(f'CREATE TABLE "iris.controller_db__jobs"({key}, name VARCHAR, ts TIMESTAMP, seq BIGINT)')
    database.execute(
        f"""CREATE TABLE "iris.controller_db__job_config"({key}, res_cpu_millicores BIGINT, res_memory_bytes BIGINT,
               accelerator_kind VARCHAR, accelerator_type VARCHAR, accelerators_per_task BIGINT,
               ts TIMESTAMP, seq BIGINT)"""
    )
    database.execute(
        """CREATE TABLE "iris.task"(cluster VARCHAR, task_id VARCHAR, attempt_id BIGINT, worker_id VARCHAR,
               ts TIMESTAMP, cpu_millicores BIGINT, memory_mb BIGINT)"""
    )
    at = "TIMESTAMP '1970-01-01 00:00:00' + INTERVAL {} MINUTE"
    # Job a: one attempt, minutes 1-3. Job b: one attempt from minute 2, still running. Job c: another node.
    for job, attempt_uid, node, started, finished, seq in (
        ("/u/a", "uid-a", NODE, 1, 3, 1),
        ("/u/b", "uid-b", NODE, 2, None, 2),
        ("/u/c", "uid-c", "other-node", 1, None, 3),
    ):
        finished_sql = "NULL" if finished is None else at.format(finished)
        database.execute(
            f"""INSERT INTO "iris.controller_db__task_attempts" VALUES ('{CLUSTER}', '{job}', 0, '{job}', 0,
                   '{job}/0', 0, '{attempt_uid}', 'TASK_STATE_RUNNING', {at.format(started)}, {finished_sql},
                   '{node}', '', {at.format(started)}, {seq})"""
        )
        database.execute(
            f"""INSERT INTO "iris.controller_db__jobs" VALUES ('{CLUSTER}', '{job}', 0, '{job}', 0,
                   '{job.strip("/")}', {at.format(0)}, {seq})"""
        )
        database.execute(
            f"""INSERT INTO "iris.controller_db__job_config" VALUES ('{CLUSTER}', '{job}', 0, '{job}', 0,
                   2000, 1073741824, 'gpu', 'H100', 8, {at.format(0)}, {seq})"""
        )
    for minute in range(1, 5):
        database.execute(
            f"""INSERT INTO "iris.task" VALUES ('{CLUSTER}', '/u/b/0', 0, 'pod-b', {at.format(minute)}, 1500, 2048)"""
        )
    return database


def _client(database: duckdb.DuckDBPyConnection) -> tuple[TestClient, list[str]]:
    queries: list[str] = []

    def query(sql: str, *, max_rows: int):
        queries.append(sql)
        return database.execute(sql).fetch_arrow_table()

    source = Record(target=ClusterTarget("marin", "project", "zone", "fleet", "cluster"), query=query)
    app = create_app(replace(bridge_config(), max_rows=100_000), {"marin": source}, {}, None, None, None)
    return TestClient(app), queries


def test_node_jobs_serves_the_job_row_from_two_source_queries():
    client, queries = _client(_database())
    with client:
        views = {
            view: client.get("/finelog/marin/v1/node/jobs", params={**PARAMS, "view": view})
            for view in ("job_activity", "attempts", "attempt_usage")
        }

    assert all(response.status_code == 200 for response in views.values()), {v: r.text for v, r in views.items()}
    assert len(queries) == 2
    active = {}
    for row in views["job_activity"].json():
        active.setdefault(row["job_id"], {})[row["bucket"] // MINUTE] = row["active"]
    # The finished attempt is active in minutes 1 and 2; the running one from minute 2 to the range end.
    assert {minute for minute, on in active["/u/a"].items() if on} == {1, 2}
    assert {minute for minute, on in active["/u/b"].items() if on} == set(range(2, 10))
    assert active["/u/a"][2] + active["/u/b"][2] == 2
    attempts = views["attempts"].json()
    assert [row["job_id"] for row in attempts] == ["/u/a", "/u/b"]
    assert "user_id" not in attempts[0]
    usage = views["attempt_usage"].json()
    assert {row["attempt"] for row in usage} == {"/u/b/0:0"}
    assert {row["cpu_cores"] for row in usage} == {1.5}


@pytest.mark.parametrize(
    "params",
    [
        {**PARAMS, "node": ""},
        {**PARAMS, "node": "g73'; DROP TABLE x; --"},
        {**PARAMS, "clusters": "not-a-cluster"},
    ],
    ids=["empty node", "malformed node", "unknown cluster"],
)
def test_node_jobs_rejects_a_bad_node_or_cluster_before_any_hub_query(params):
    client, queries = _client(_database())
    with client:
        response = client.get("/finelog/marin/v1/node/jobs", params={**params, "view": "attempts"})

    assert response.status_code == 400
    assert queries == []
