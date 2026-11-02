# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import duckdb
import pyarrow as pa
import pyarrow.ipc as ipc
import pytest
from finelog.benchmarks import grafana_dashboard_bench as benchmark
from finelog.benchmarks.layout_candidates import table_digest
from finelog.rpc import finelog_stats_pb2 as stats_pb2


@pytest.fixture
def query_service(monkeypatch):
    statements = []
    with duckdb.connect() as database:
        database.execute("CREATE MACRO to_timestamp_millis(value) AS to_timestamp(value / 1000.0)")

        class QueryService:
            def query(self, request):
                statements.append(request.sql)
                table = database.execute(request.sql).to_arrow_table()
                sink = pa.BufferOutputStream()
                with ipc.new_stream(sink, table.schema) as writer:
                    writer.write_table(table)
                return stats_pb2.QueryResponse(arrow_ipc=sink.getvalue().to_pybytes(), row_count=table.num_rows)

        monkeypatch.setattr(benchmark, "stats_client", lambda address: QueryService())
        monkeypatch.setattr(benchmark, "server_info", lambda address: {"build": "test-server"})
        yield statements


@pytest.mark.parametrize("bounds", [[], ["--start-ms", "9999", "--end-ms", "19999"]])
def test_literal_query_workload_preserves_sql_without_claiming_rendered_bounds(tmp_path, query_service, capsys, bounds):
    sql = "SELECT 123 AS start_ms, 456 AS end_ms"
    query_file = tmp_path / "fixed.json"
    query_file.write_text(json.dumps([{"name": "fixed_window", "sql": sql}]))
    output = tmp_path / "result.json"

    assert (
        benchmark.main(
            [
                "--query-file",
                str(query_file),
                "--address",
                "http://existing-server",
                "--warmup",
                "0",
                "--iterations",
                "1",
                "--no-explain",
                "--output",
                str(output),
                *bounds,
            ]
        )
        == 0
    )

    payload = json.loads(output.read_text())
    (workload,) = payload["workloads"]
    assert "window" not in payload
    assert workload["window"] is None
    assert workload["queries"][0]["sql"] == sql
    assert query_service == [sql]
    assert workload["queries"][0]["result_digest"] == table_digest(pa.table({"start_ms": [123], "end_ms": [456]}))
    assert json.loads(capsys.readouterr().out) == {"output": str(output), "workloads": 1, "queries": 1}


def test_mixed_workloads_record_dashboard_rendering_separately_from_literal_sql(tmp_path, query_service):
    dashboard = tmp_path / "dashboard.json"
    dashboard_sql = (
        "SELECT CAST(EXTRACT(EPOCH FROM {{from}}) * 1000 AS BIGINT) AS start_ms, "
        "CAST(EXTRACT(EPOCH FROM {{to}}) * 1000 AS BIGINT) AS end_ms, ${__interval_ms} AS bucket_ms"
    )
    dashboard.write_text(
        json.dumps(
            {
                "uid": "dashboard",
                "title": "Dashboard",
                "refresh": "1m",
                "panels": [
                    {
                        "title": "Window",
                        "targets": [{"refId": "A", "url_options": {"params": [{"key": "sql", "value": dashboard_sql}]}}],
                    }
                ],
            }
        )
    )
    query_file = tmp_path / "fixed.json"
    literal_sql = "SELECT 7 AS bucket_ms"
    query_file.write_text(json.dumps([{"name": "fixed", "sql": literal_sql}]))
    output = tmp_path / "result.json"

    assert (
        benchmark.main(
            [
                "--dashboard",
                str(dashboard),
                "--query-file",
                str(query_file),
                "--start-ms",
                "1000",
                "--end-ms",
                "5000",
                "--interval-ms",
                "250",
                "--address",
                "http://existing-server",
                "--warmup",
                "0",
                "--iterations",
                "1",
                "--no-explain",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    rendered, literal = json.loads(output.read_text())["workloads"]
    assert rendered["window"] == {"start_ms": 1000, "end_ms": 5000, "interval_ms": 250, "variables": {}}
    assert rendered["queries"][0]["result_digest"] == table_digest(
        pa.table({"start_ms": [1000], "end_ms": [5000], "bucket_ms": [250]})
    )
    assert literal["window"] is None
    assert literal["queries"][0]["result_digest"] == table_digest(pa.table({"bucket_ms": [7]}))
    assert query_service[-1] == literal_sql
