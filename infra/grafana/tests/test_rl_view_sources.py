# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from types import SimpleNamespace

import duckdb
import pytest
from config import ClusterTarget
from conftest import bridge_config, install_finelog_dialect_macros
from rl_execution_observability import rl_execution_dataset
from server import create_app
from starlette.testclient import TestClient


@pytest.fixture
def telemetry_database():
    with duckdb.connect() as database:
        install_finelog_dialect_macros(database)
        database.execute(
            """CREATE TABLE "telemetry_v1.marinskyrl"(
                cluster VARCHAR, service VARCHAR, run_id VARCHAR, job_id VARCHAR, execution_uid VARCHAR,
                timestamp_ms BIGINT, seq BIGINT, name VARCHAR, value DOUBLE, attributes_json VARCHAR,
                resource_attributes_json VARCHAR, body_json VARCHAR, node_name VARCHAR, process_index BIGINT)
            """
        )
        database.execute(
            """INSERT INTO "telemetry_v1.marinskyrl" VALUES
                ('cw-a', 'marinskyrl', 'run', 'job', 'execution', 1000, 1,
                 'policy_step', 10, '{}', '{"role":"trainer"}', '{}', 'node', 0),
                ('cw-a', 'marinskyrl', 'run', 'job', 'execution', 31000, 2,
                 'policy_step', 11, '{}', '{"role":"trainer"}', '{}', 'node', 0),
                ('cw-a', 'marinskyrl', 'run', 'job', 'execution', 31000, 3,
                 'training_metric_value', 0.75,
                 '{"role":"trainer","payload_kind":"train","metric":"reward/avg_raw_reward","step":"11"}',
                 '{"role":"trainer"}', '{}', 'node', 0),
                ('cw-a', 'marinskyrl', 'run', 'job', 'execution', 31000, 4,
                 'phase_duration_seconds', 2,
                 '{"clock_domain":"critical_path","phase":"train_step","outcome":"success"}',
                 '{"role":"trainer"}', '{}', 'node', 0)
            """
        )
        yield database


def _client(database, queries):
    def query(sql, *, max_rows):
        queries.append(sql)
        return database.execute(sql.replace("FIRST_VALUE(", "FIRST(")).to_arrow_table()

    source = SimpleNamespace(target=ClusterTarget("marin", "project", "zone", "fleet", "cluster"), query=query)
    config = replace(bridge_config(), max_rows=200_000)
    return TestClient(create_app(config, {"marin": source}, {}, None, None, None))


PARAMS = {
    "clusters": "cw-a",
    "run": "run",
    "job": "job",
    "executions": "execution",
    "from": 0,
    "to": 60_000,
    "bucket_ms": 30_000,
}


def test_rl_core_chart_loads_without_engine_namespace(telemetry_database):
    queries = []
    with _client(telemetry_database, queries) as client:
        response = client.get("/finelog/marin/v1/rl/overview", params={**PARAMS, "view": "critical_path"})
        assert response.status_code == 200, response.text
        assert response.json() == [
            {"section": "critical_path", "t": 30_000, "series": "train_step · success", "value": 2.0}
        ]
        straggler = client.get("/finelog/marin/v1/rl/overview", params={**PARAMS, "view": "straggler"})
        assert straggler.status_code == 200, straggler.text
        assert straggler.json() == []
    assert len(queries) == 1


def test_rl_metric_chart_ignores_unrelated_invalid_memory_records(telemetry_database):
    telemetry_database.execute(
        """INSERT INTO "telemetry_v1.marinskyrl" VALUES
            ('cw-a', 'marinskyrl', 'run', 'job', 'execution', 31000, 5,
             'cuda_memory_observation', 1, '{"worker_role":"policy"}', '{}',
             '{"peak_allocated_bytes":"invalid"}', 'node', 0)
        """
    )
    queries = []
    with _client(telemetry_database, queries) as client:
        response = client.get("/finelog/marin/v1/rl/execution", params={**PARAMS, "view": "reward"})
        assert response.status_code == 200, response.text
        assert response.json() == [
            {
                "section": "reward",
                "t": 31_000,
                "series": "reward/avg_raw_reward train · execution",
                "value": 0.75,
            }
        ]
    assert len(queries) == 1


def test_rl_execution_grouped_views_match_full_projection_without_duplicate_scans(telemetry_database):
    baseline_queries = []
    with _client(telemetry_database, baseline_queries) as client:
        response = client.get("/finelog/marin/v1/rl/execution", params=PARAMS)
        assert response.status_code == 200, response.text
        baseline = response.json()

    # Read every declared panel, including views with no rows for this run.
    views = rl_execution_dataset(("cw-a",), "run", "job", ("execution",), 0, 60_000, 30_000).views
    queries = []
    with _client(telemetry_database, queries) as client:
        for view in views:
            response = client.get("/finelog/marin/v1/rl/execution", params={**PARAMS, "view": view})
            assert response.status_code == 200, response.text
            assert response.json() == [row for row in baseline if row["section"] == view]
    assert len(queries) == len(baseline_queries) == 10
