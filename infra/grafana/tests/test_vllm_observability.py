# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from typing import NamedTuple

import duckdb
import pyarrow as pa
import pytest
from config import ClusterTarget
from conftest import bridge_config
from github_source import GithubSource
from k8s_source import K8sFleet
from server import create_app
from starlette.testclient import TestClient
from vllm_observability import (
    VLLM_MAX_POINTS,
    VLLM_MAX_WINDOW_MS,
    VLLM_OVERVIEW_SECTIONS,
    VllmIdentityField,
    vllm_overview_query,
)
from wandb_source import WandbSource


class TelemetryRow(NamedTuple):
    cluster: str
    service: str
    job_id: str
    run_id: str
    execution_uid: str
    name: str
    kind: str
    value: float
    resource_attributes_json: str
    attributes_json: str
    timestamp_ms: int


def _attributes(temporality: str | None = None, **labels: str) -> str:
    if temporality is not None:
        labels["source_temporality"] = temporality
    return json.dumps(labels, sort_keys=True, separators=(",", ":"))


def _record(
    service: str,
    job: str,
    run: str,
    name: str,
    value: float,
    timestamp_ms: int,
    attributes: str,
    *,
    replica: str = "driver",
) -> TelemetryRow:
    resource = json.dumps({"job_id": job, "worker": replica}, sort_keys=True, separators=(",", ":"))
    return TelemetryRow(
        "cw-a",
        service,
        job,
        run,
        "execution-1",
        name,
        "gauge",
        value,
        resource,
        attributes,
        timestamp_ms,
    )


def _database(rows: list[TelemetryRow]) -> duckdb.DuckDBPyConnection:
    database = duckdb.connect()
    columns = """
        cluster VARCHAR, service VARCHAR, job_id VARCHAR, run_id VARCHAR,
        execution_uid VARCHAR, name VARCHAR, kind VARCHAR, value DOUBLE,
        resource_attributes_json VARCHAR, attributes_json VARCHAR,
        timestamp_ms BIGINT, seq BIGINT
    """
    for service in ("vllm", "marinskyrl"):
        database.execute(f'CREATE TABLE "telemetry_v1.{service}"({columns})')
        service_rows = [row for row in rows if row.service == service]
        if service_rows:
            database.executemany(
                f'INSERT INTO "telemetry_v1.{service}" VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                [(*row, seq) for seq, row in enumerate(service_rows)],
            )
    database.execute('CREATE VIEW telemetry_v1 AS SELECT * FROM "telemetry_v1.vllm" WHERE FALSE')
    database.execute(
        "CREATE MACRO json_get(document, field_name) " "AS json_extract_string(document, concat('$.', field_name))"
    )
    return database


class _DuckDBFinelog:
    def __init__(self, database: duckdb.DuckDBPyConnection):
        self.target = ClusterTarget("marin", "p", "z", "f", "c")
        self._database = database

    def query(self, sql: str, *, max_rows: int) -> pa.Table:
        table = self._database.execute(sql).fetch_arrow_table()
        assert table.num_rows <= max_rows
        return table

    def health(self):
        raise AssertionError("these tests do not query source health")


def _app(database: duckdb.DuckDBPyConnection):
    return create_app(
        bridge_config(),
        {"marin": _DuckDBFinelog(database)},
        {},
        GithubSource(auth=None, timeout=5.0),
        K8sFleet(()),
        WandbSource(timeout=5.0),
    )


def _embedded_record(
    name: str,
    value: float,
    timestamp_ms: int,
    *,
    temporality: str = "cumulative_snapshot",
    engine: str | None = "physical-a",
    **labels: str,
) -> TelemetryRow:
    identity = {} if engine is None else {"engine": engine, "engine_index": "0"}
    return _record(
        "marinskyrl",
        "/train",
        "embedded-run",
        name,
        value,
        timestamp_ms,
        _attributes(temporality, metric_source="vllm", **identity, **labels),
    )


def _observability_records() -> list[TelemetryRow]:
    rows: list[TelemetryRow] = []
    for sample, timestamp_ms in enumerate(range(0, 150_000, 15_000)):
        finished = sample // 2
        cumulative = {
            "generation_tokens_total": sample * 150,
            "num_preemptions_total": sample // 3,
            "request_success_total": finished,
            "iteration_tokens_total_sum": sample * 100,
            "iteration_tokens_total_count": sample,
            "iteration_tokens_total_bucket": sample,
            "request_time_per_output_token_seconds_sum": finished * 0.1,
            "request_time_per_output_token_seconds_count": finished,
            "request_time_per_output_token_seconds_bucket": finished,
        }
        for name, value in cumulative.items():
            labels = {}
            if name.endswith("_bucket"):
                labels["le"] = "+Inf"
            if name == "request_success_total":
                labels["finished_reason"] = "stop"
            rows.append(_embedded_record(name, value, timestamp_ms, **labels))
        for name, value in (
            ("num_requests_running", sample + 1),
            ("num_requests_waiting", sample % 3),
            ("kv_cache_usage_perc", 0.5),
        ):
            rows.append(
                _embedded_record(
                    name,
                    value,
                    timestamp_ms,
                    temporality="current_snapshot",
                    step=str(sample),
                )
            )

    for component, value in (("sum", 20), ("count", 1), ("bucket", 1)):
        labels = {"le": "+Inf"} if component == "bucket" else {}
        rows.append(
            _embedded_record(
                f"request_time_per_output_token_seconds_{component}",
                value,
                15_000,
                engine="physical-b",
                **labels,
            )
        )
    for component in ("sum", "count", "bucket"):
        for timestamp_ms, value in ((0, 0), (15_000, 1_000)):
            labels = {"le": "+Inf"} if component == "bucket" else {}
            rows.append(
                _record(
                    "marinskyrl",
                    "/train",
                    "embedded-run",
                    f"request_time_per_output_token_seconds_{component}",
                    value,
                    timestamp_ms,
                    _attributes("cumulative_snapshot", metric_source="vllm", engine="legacy", **labels),
                )
            )
    rows.extend(
        _embedded_record(
            "metric_publication_dropped_records",
            0,
            135_000,
            temporality="current_snapshot",
            engine=None,
            drop_reason=reason,
        )
        for reason in ("sample_limit", "telemetry_loss")
    )
    rows.extend(
        [
            _record(
                "marinskyrl",
                "/train",
                "ray-run",
                "generation_tokens_total",
                100_000,
                135_000,
                _attributes("cumulative_snapshot", metric_source="ray"),
            ),
            _record(
                "marinskyrl",
                "/train",
                "ray-run",
                "metric_publication_dropped_records",
                99,
                135_000,
                _attributes("current_snapshot", metric_source="ray", drop_reason="telemetry_loss"),
            ),
        ]
    )
    rows.extend(
        _record(
            "vllm",
            "/standalone",
            "standalone-run",
            "generation_tokens_total",
            value,
            timestamp_ms,
            _attributes("cumulative_snapshot"),
        )
        for timestamp_ms, value in ((0, 0), (60_000, 600), (120_000, 1_200))
    )
    return rows


def _one(rows: list[dict], section: str, metric: str, stat: str, series: str | None = None) -> dict:
    matches = [
        row
        for row in rows
        if row["section"] == section
        and row["metric"] == metric
        and row["stat"] == stat
        and (series is None or row["series"] == series)
    ]
    assert len(matches) == 1
    return matches[0]


def test_dashboard_overview_end_to_end():
    dashboard = json.loads((Path(__file__).parents[1] / "dashboards" / "inference.json").read_text())
    targets = [
        target
        for panel in dashboard["panels"]
        for target in panel.get("targets", [])
        if target.get("url") == "/v1/vllm/overview"
    ]
    target_views = {
        next(p["value"] for p in target["url_options"]["params"] if p["key"] == "view") for target in targets
    }
    assert target_views == VLLM_OVERVIEW_SECTIONS

    database = _database(_observability_records())
    identity_variable = next(item for item in dashboard["templating"]["list"] if item["name"] == "identity")
    identity_sql = identity_variable["query"]["infinityQuery"]["url_options"]["params"][0]["value"]
    identity_sql = (
        identity_sql.replace("${identity_kind}", "run_id")
        .replace("{{from}}", "TIMESTAMP '1970-01-01 00:02:00'")
        .replace("{{to}}", "TIMESTAMP '1970-01-01 00:03:00'")
    )
    identities = {row[0] for row in database.execute(identity_sql).fetchall()}
    assert "embedded-run" in identities
    assert "ray-run" not in identities

    concrete = {"identity_kind": "job_id", "identity": "/train", "from": "0", "to": "150000", "bucket_ms": "15000"}
    rows_by_view = {}
    with TestClient(_app(database)) as client:
        for target in targets:
            params = {
                param["key"]: concrete.get(param["key"], param["value"]) for param in target["url_options"]["params"]
            }
            response = client.get(f"/finelog/marin{target['url']}", params=params)
            assert response.status_code == 200
            view = params["view"]
            assert response.json() and all(row["section"] == view for row in response.json())
            rows_by_view[view] = response.json()

        token_target = next(
            target for target in targets if any(p["value"] == "token_rate" for p in target["url_options"]["params"])
        )
        standalone = {**concrete, "identity": "/standalone", "to": "135000"}
        standalone_params = {
            param["key"]: standalone.get(param["key"], param["value"]) for param in token_target["url_options"]["params"]
        }
        standalone_response = client.get(f"/finelog/marin{token_target['url']}", params=standalone_params)

    schema = {"t", "section", "metric", "stat", "series", "value", "unit", "status", "samples", "gap_seconds"}
    generated_rates = [row for row in rows_by_view["token_rate"] if row["metric"] == "generated_tokens"]
    assert len(generated_rates) == 9
    assert all(set(row) == schema and row["value"] == pytest.approx(10) for row in generated_rates)

    saturation = rows_by_view["saturation"]
    running = sorted((row for row in saturation if row["metric"] == "num_requests_running"), key=lambda row: row["t"])
    iteration = [row for row in saturation if row["metric"] == "iteration_tokens"]
    assert [row["value"] for row in running] == list(range(1, 11))
    assert len(iteration) == 9 and all(row["value"] == pytest.approx(100) for row in iteration)

    tpot = [row for row in rows_by_view["latency"] if row["metric"] == "tpot" and row["stat"] == "mean_over_time"]
    assert len(tpot) == 4 and all(row["value"] == pytest.approx(0.1) for row in tpot)
    assert _one(rows_by_view["counter_total"], "counter_total", "generated_tokens", "total")["value"] == 1_350
    assert _one(rows_by_view["counter_total"], "counter_total", "preemptions", "total")["value"] == 3
    assert _one(rows_by_view["request_outcome"], "request_outcome", "requests", "total", "stop")["value"] == 4
    assert _one(rows_by_view["freshness"], "freshness", "telemetry", "latest_sample_age")["status"] == "fresh"
    health = rows_by_view["telemetry_health"]
    assert _one(health, "telemetry_health", "collector", "polls")["status"] == "healthy"
    assert not [row for row in health if row["metric"] == "dropped samples"]

    assert standalone_response.status_code == 200
    standalone_rates = [row for row in standalone_response.json() if row["metric"] == "generated_tokens"]
    assert len(standalone_rates) == 2
    assert all(row["value"] == pytest.approx(10) for row in standalone_rates)
    base = {"identity_kind": "job_id", "identity": "/missing", "from": "0", "to": "60000", "bucket_ms": "15000"}
    with TestClient(_app(_database([]))) as client:
        response = client.get("/finelog/marin/v1/vllm/overview", params=base)
        assert response.status_code == 200
        assert [(row["section"], row["status"]) for row in response.json()] == [
            ("freshness", "no_data"),
            ("telemetry_health", "unknown"),
        ]

        invalid = (
            ({"identity_kind": "invalid"}, "identity_kind"),
            ({"view": "invalid"}, "unknown vLLM overview view"),
            ({"bucket_ms": "0"}, "positive"),
            ({"to": str(VLLM_MAX_WINDOW_MS + 1)}, "7 days"),
        )
        for overrides, message in invalid:
            response = client.get("/finelog/marin/v1/vllm/overview", params={**base, **overrides})
            assert response.status_code == 400
            assert message in response.json()["error"]

    bounded = vllm_overview_query(VllmIdentityField.JOB_ID, "/serve", 0, VLLM_MAX_WINDOW_MS, 1)
    assert bounded.bucket_ms >= VLLM_MAX_WINDOW_MS / VLLM_MAX_POINTS
