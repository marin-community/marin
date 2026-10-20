# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import duckdb
import pytest
from config import ClusterTarget
from conftest import bridge_config
from runs_observability import runs_overview_dataset
from server import create_app
from starlette.testclient import TestClient


@pytest.fixture
def runs_database():
    with duckdb.connect() as database:
        database.execute(
            """CREATE TABLE "levanter.metrics"(
                   cluster VARCHAR, node_name VARCHAR, run_id VARCHAR, step BIGINT,
                   process_index VARCHAR, name VARCHAR, value DOUBLE, timestamp_ms BIGINT)"""
        )
        database.executemany(
            """INSERT INTO "levanter.metrics"
               VALUES ('cw-a', 'node-a', 'selected', 10, '0', ?, ?, ?)""",
            [
                ("train_loss", 2.0, 10_000),
                ("train_loss", 4.0, 20_000),
                ("throughput_mfu", 0.5, 10_000),
                ("throughput_tokens_per_second", 100.0, 10_000),
                ("throughput_duration", 2.0, 10_000),
                ("throughput_loading_time", 0.1, 10_000),
                ("throughput_hook_time", 0.2, 10_000),
                ("eval_loss", 7.0, 30_000),
            ],
        )
        database.execute("CREATE MACRO json_get(document, key) AS json_extract_string(document, '$.' || key)")
        yield database


def runs_client(database, queries):
    def query(sql, *, max_rows):
        queries.append(sql)
        table = database.execute(sql).to_arrow_table()
        assert table.num_rows <= max_rows
        return table

    source = SimpleNamespace(
        target=ClusterTarget("marin", "project", "zone", "fleet", "cluster"),
        query=query,
    )
    app = create_app(bridge_config(), {"marin": source}, {}, None, None, None)
    return TestClient(app)


def test_runs_metric_panels_load_without_node_telemetry(runs_database):
    params = {"clusters": "cw-a", "runs": "selected", "from": 0, "to": 60_000, "bucket_ms": 30_000}
    queries = []
    with runs_client(runs_database, queries) as client:
        responses = {
            view: client.get("/finelog/marin/v1/runs/overview", params={**params, "view": view})
            for view in ("loss", "active", "mfu", "tokens", "step", "step_time", "evaluation")
        }

    assert all(response.status_code == 200 for response in responses.values())
    assert responses["loss"].json() == [{"section": "loss", "t": 0, "series": "selected", "value": 3.0}]
    assert responses["active"].json()[0]["nodes"] == 1
    assert responses["evaluation"].json()[0]["value"] == 7.0
    # The whole metrics group shares one fetch and does not require node telemetry.
    assert len(queries) == 1


def test_runs_all_panels_and_combined_response_share_the_two_sources(runs_database):
    runs_database.execute(
        """CREATE TABLE "telemetry_v1.node_agent"(
               cluster VARCHAR, node_name VARCHAR, service VARCHAR, name VARCHAR,
               attributes_json VARCHAR, value DOUBLE, timestamp_ms BIGINT)"""
    )
    runs_database.execute(
        """INSERT INTO "telemetry_v1.node_agent" VALUES
               ('cw-a', 'node-a', 'iris-node-agent', 'gpu_power_watts', '{"gpu_uuid":"a"}', 300, 10000),
               ('cw-a', 'node-a', 'iris-node-agent', 'gpu_power_watts', '{"gpu_uuid":"a"}', 500, 20000),
               ('cw-a', 'node-a', 'iris-node-agent', 'gpu_power_watts', '{"gpu_uuid":"b"}', 200, 10000)"""
    )
    params = {"clusters": "cw-a", "runs": "selected", "from": 0, "to": 60_000, "bucket_ms": 30_000}
    dataset = runs_overview_dataset(("cw-a",), ("selected",), 0, 60_000, 30_000)
    queries = []
    with runs_client(runs_database, queries) as client:
        responses = {
            view: client.get("/finelog/marin/v1/runs/overview", params={**params, "view": view})
            for view in dataset.views
        }
        combined = client.get("/finelog/marin/v1/runs/overview", params=params)

    assert combined.status_code == 200
    assert all(response.status_code == 200 for response in responses.values())
    assert responses["power"].json() == [{"section": "power", "t": 0, "series": "selected", "value": 0.6}]
    for view, response in responses.items():
        assert response.json() == [row for row in combined.json() if row["section"] == view]
    assert len(queries) == 2


def test_runs_power_panel_does_not_require_training_metric_columns():
    with duckdb.connect() as database:
        database.execute(
            """CREATE TABLE "levanter.metrics"(
                   cluster VARCHAR, node_name VARCHAR, run_id VARCHAR, step BIGINT, timestamp_ms BIGINT)"""
        )
        database.execute("""INSERT INTO "levanter.metrics" VALUES ('cw-a', 'node-a', 'selected', 1, 10000)""")
        database.execute(
            """CREATE TABLE "telemetry_v1.node_agent"(
                   cluster VARCHAR, node_name VARCHAR, service VARCHAR, name VARCHAR,
                   attributes_json VARCHAR, value DOUBLE, timestamp_ms BIGINT)"""
        )
        database.execute(
            """INSERT INTO "telemetry_v1.node_agent" VALUES
                   ('cw-a', 'node-a', 'iris-node-agent', 'gpu_power_watts', '{"gpu_uuid":"a"}', 400, 10000)"""
        )
        database.execute("CREATE MACRO json_get(document, key) AS json_extract_string(document, '$.' || key)")
        queries = []
        with runs_client(database, queries) as client:
            response = client.get(
                "/finelog/marin/v1/runs/overview",
                params={
                    "clusters": "cw-a",
                    "runs": "selected",
                    "from": 0,
                    "to": 60_000,
                    "bucket_ms": 30_000,
                    "view": "power",
                },
            )

    assert response.status_code == 200
    assert response.json() == [{"section": "power", "t": 0, "series": "selected", "value": 0.4}]
    assert len(queries) == 1
