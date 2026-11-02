# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import duckdb
import pyarrow.ipc as ipc
import pytest
import server
from benchmarks import dashboard_query_cost as benchmark
from starlette.testclient import TestClient


@pytest.mark.parametrize(
    "duration,points,minimum,expected",
    [
        (10_800_000, 100, "", 120_000),
        (10_800_000, 360, "", 30_000),
        (600_000, 360, "15s", 15_000),
        (10_800_000, 100, "5m", 300_000),
    ],
)
def test_dashboard_intervals_match_grafana_rounding_and_panel_minimum(duration, points, minimum, expected):
    assert benchmark.interval_ms(duration, points, minimum) == expected


def test_dashboard_discovery_replays_visible_targets_with_panel_time_and_variable_formats():
    dashboard = {
        "time": {"from": "now-3h", "to": "now"},
        "panels": [
            {
                "id": 1,
                "title": "mixed",
                "type": "table",
                "datasource": {"uid": "status"},
                "maxDataPoints": 100,
                "targets": [
                    {
                        "refId": "A",
                        "url": "/finelog/marin/query",
                        "url_options": {
                            "params": [
                                {"key": "from", "value": "${__from}"},
                                {"key": "to", "value": "${__to}"},
                                {"key": "sql", "value": "SELECT ${run:sqlstring}, '${cluster:csv}'"},
                                {"key": "bucket_ms", "value": "${__interval_ms}"},
                            ]
                        },
                    },
                    {"refId": "B", "url": "/k8s/nodes"},
                    {"refId": "C", "url": "/finelog/marin/query", "hide": True},
                ],
            },
            {
                "id": 2,
                "title": "override",
                "type": "table",
                "datasource": {"uid": "finelog-marin"},
                "timeFrom": "10m",
                "interval": "15s",
                "maxDataPoints": 100,
                "targets": [
                    {
                        "refId": "A",
                        "url": "/v1/training/overview",
                        "url_options": {
                            "params": [{"key": "run", "value": "${run}"}, {"key": "to", "value": "${__to}"}]
                        },
                    }
                ],
            },
            {
                "id": 3,
                "title": "collapsed",
                "type": "row",
                "collapsed": True,
                "panels": [{"id": 4, "targets": [{"url": "/finelog/marin/query"}]}],
            },
        ],
    }
    requests, skipped = benchmark.panel_requests(
        dashboard, {"run": ["O'Brien"], "cluster": ["cw-a", "marin"]}, 10_800_000
    )

    assert len(requests) == 2
    assert requests[0].params == {
        "from": "0",
        "to": "10800000",
        "sql": "SELECT 'O''Brien', 'cw-a,marin'",
        "bucket_ms": "120000",
    }
    assert requests[1].path == "/finelog/marin/v1/training/overview"
    assert requests[1].from_ms == 10_200_002
    assert requests[1].to_ms == 10_800_002
    assert requests[1].interval_ms == 15_000
    assert requests[1].params == {"run": "O'Brien", "to": "10800002"}
    assert {row["reason"] for row in skipped} == {"external datasource 'status'", "hidden target", "collapsed row"}


@pytest.mark.parametrize("value", ["${run:percentencode}", "$__interval", "${missing}"])
def test_dashboard_replay_rejects_unresolved_request_macros(value):
    with pytest.raises(ValueError, match="unsupported"):
        benchmark.substitute(value, {"run": ["selected"]}, {})


def test_dashboard_replay_writes_evidence_and_reuses_only_warm_bridge_caches(tmp_path, monkeypatch):
    grafana_dir = Path(server.__file__).resolve().parent.parent
    variables = tmp_path / "variables.json"
    variables.write_text(json.dumps({"nodes.json": {"cluster": ["cw-a"], "node": "node-a"}}))
    observed_clocks = []
    with duckdb.connect() as database:
        database.execute(
            """CREATE TABLE "telemetry_v1.node_agent"(
                   cluster VARCHAR, node_name VARCHAR, service VARCHAR, name VARCHAR,
                   attributes_json VARCHAR, timestamp_ms BIGINT, seq BIGINT, value DOUBLE)"""
        )
        database.execute(
            """CREATE MACRO named_struct(k1, v1, k2, v2, k3, v3)
                       AS struct_pack(timestamp_ms := v1, seq := v2, value := v3)"""
        )
        database.execute(
            """INSERT INTO "telemetry_v1.node_agent" VALUES
                   ('cw-a', 'node-a', 'iris-node-agent', 'gpu_power_watts',
                    '{"gpu_index":"0"}', 10770000, 1, 400)"""
        )

        class Upstream:
            def query(self, sql, *, max_rows):
                observed_clocks.append((server.time.time(), time.time()))
                table = database.execute(sql).to_arrow_table()
                assert table.num_rows <= max_rows
                return table

            def close(self):
                pass

        monkeypatch.setattr(benchmark.LogClient, "connect", lambda address, *, timeout_ms: Upstream())
        result = benchmark.run_replay(
            argparse.Namespace(
                grafana_dir=grafana_dir,
                dashboard=["nodes.json"],
                address="http://127.0.0.1:19001",
                to_ms=10_800_000,
                variables=variables,
                output=tmp_path / "evidence",
                iterations=2,
                workers=1,
                live_window=True,
                warm=True,
            )
        )

    assert [page["query_count"] for page in result["pages"]] == [1, 0, 1, 0]
    assert all(panel["status"] == 200 for page in result["pages"] for panel in page["panels"])
    power = next(panel for panel in result["pages"][0]["panels"] if panel["params"]["view"] == "power")
    assert len(power["response"]) == 1
    assert power["response"][0]["series"] == "GPU 0"
    assert power["response"][0]["value"] == 400.0
    assert all(bridge == 10801 and real > 1_000_000_000 for bridge, real in observed_clocks)
    assert server.time.time() > 1_000_000_000
    evidence = tmp_path / "evidence" / "nodes" / "iteration-0"
    with ipc.open_file(evidence / "query-0.arrow") as reader:
        assert reader.read_all().num_rows == result["pages"][0]["query_rows"]
    assert json.loads((evidence / "cold.json").read_text())["query_count"] == 1
    assert (evidence / "query-0.sql").read_text()


def test_dashboard_replay_producers_discovers_namespaces_and_counts_metadata_separately(tmp_path):
    with duckdb.connect() as database:
        database.execute(
            """CREATE TABLE "telemetry_v1.vllm"(
                   service VARCHAR, resource_attributes_json VARCHAR, execution_uid VARCHAR,
                   attributes_json VARCHAR, name VARCHAR, timestamp_ms BIGINT, run_id VARCHAR, cluster VARCHAR)"""
        )
        database.execute(
            """INSERT INTO "telemetry_v1.vllm" VALUES
                   ('vllm', '{"role":"inference"}', 'attempt', '{}', 'num_requests_running', 1000, 'run', 'cw-a')"""
        )
        database.execute("CREATE MACRO json_get(document, key) AS json_extract_string(document, '$.' || key)")

        class Upstream:
            def list_namespaces(self):
                return [SimpleNamespace(namespace="telemetry_v1.vllm"), SimpleNamespace(namespace="unrelated")]

            def query(self, sql, *, max_rows):
                table = database.execute(sql).to_arrow_table()
                assert table.num_rows <= max_rows
                return table

        source = benchmark.RecordingSource(Upstream(), tmp_path)
        app = server.create_app(benchmark.replay_bridge_config(), {"marin": source}, {}, None, None, None)
        request = benchmark.PanelRequest(
            1,
            "Services reporting this run",
            "A",
            "/finelog/marin/v1/rl/producers",
            {"run": "run", "clusters": "cw-a", "from": "0", "to": "60000"},
            0,
            60_000,
            15_000,
        )
        with TestClient(app) as client:
            cold = benchmark.replay_page(client, source, [request], 1)
            warm = benchmark.replay_page(client, source, [request], 1)

    assert cold["panels"][0]["status"] == 200
    assert cold["panels"][0]["response"] == [
        {
            "producer": "vllm",
            "role": "inference",
            "attempt": "attempt",
            "metric_source": "",
            "signals": 1,
            "records": 1,
            "last_record_ms": 1000,
        }
    ]
    assert cold["namespace_calls"] == 1
    assert cold["query_count"] == 1
    assert warm["namespace_calls"] == warm["query_count"] == 0


def test_dashboard_replay_cli_reports_failed_panels_after_saving_all_iterations(tmp_path, monkeypatch):
    variables = tmp_path / "variables.json"
    variables.write_text(json.dumps({"nodes.json": {"cluster": ["cw-a"], "node": ["node-a"]}}))
    output = tmp_path / "evidence"

    class UnavailableUpstream:
        def query(self, sql, *, max_rows):
            raise RuntimeError("upstream unavailable")

        def close(self):
            pass

    monkeypatch.setattr(benchmark.LogClient, "connect", lambda address, *, timeout_ms: UnavailableUpstream())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dashboard_query_cost",
            "--grafana-dir",
            str(Path(server.__file__).resolve().parent.parent),
            "--dashboard",
            "nodes.json",
            "--address",
            "http://127.0.0.1:19001",
            "--to-ms",
            "10800000",
            "--variables",
            str(variables),
            "--output",
            str(output),
            "--iterations",
            "2",
            "--workers",
            "1",
        ],
    )

    with pytest.raises(SystemExit) as error:
        benchmark.main()

    assert error.value.code == 1
    result = json.loads((output / "result.json").read_text())
    assert len(result["pages"]) == 2
    assert all(page["panels"] for page in result["pages"])
    assert all(panel["status"] == 500 for page in result["pages"] for panel in page["panels"])
    for iteration in range(2):
        page = json.loads((output / "nodes" / f"iteration-{iteration}" / "cold.json").read_text())
        assert page == result["pages"][iteration]
        assert page["queries"][0]["error"] == "upstream unavailable"
