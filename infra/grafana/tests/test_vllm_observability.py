# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace as Record

import duckdb
import pytest
from config import ClusterTarget
from conftest import bridge_config
from dashboard_stitch import stitch_all
from server import create_app
from starlette.testclient import TestClient
from vllm_observability import VLLM_OVERVIEW_SECTIONS


@pytest.mark.parametrize("invalid_histogram", ["reset", "missing_component"])
def test_dashboard_vllm_overview_end_to_end(invalid_histogram):
    database = duckdb.connect()
    columns = """cluster VARCHAR, service VARCHAR, job_id VARCHAR, name VARCHAR, kind VARCHAR,
        value DOUBLE, resource_attributes_json VARCHAR, attributes_json VARCHAR, timestamp_ms BIGINT, seq BIGINT"""
    database.execute(f'CREATE TABLE "telemetry_v1.marinskyrl"({columns})')
    database.execute(f'CREATE TABLE "telemetry_v1.vllm"({columns})')
    database.execute("CREATE MACRO json_get(d, f) AS json_extract_string(d, concat('$.', f))")

    def sample(name, value, timestamp, **labels):
        attributes = {"metric_source": "vllm", "engine": "physical-a", "engine_index": "0", **labels}
        return name, value, json.dumps(attributes), timestamp

    cumulative = {"source_temporality": "cumulative_snapshot"}
    snapshot = {"source_temporality": "current_snapshot"}
    samples = [
        sample("generation_tokens_total", value, timestamp, **cumulative)
        for timestamp, value in ((0, 0), (15_000, 150), (30_000, 300))
    ]
    samples += [
        sample("prompt_tokens_total", value, timestamp, **cumulative)
        for timestamp, value in ((0, 0), (15_000, 30), (30_000, 60))
    ]
    samples += [
        sample("request_success_total", value, timestamp, finished_reason="stop", **cumulative)
        for timestamp, value in ((0, 0), (15_000, 1))
    ]
    for family, mean in (
        ("request_time_per_output_token_seconds", 0.1),
        ("time_to_first_token_seconds", 2),
        ("request_prefill_time_seconds", 1),
        ("request_decode_time_seconds", 10),
        ("request_generation_tokens", 100),
    ):
        for component, final in (("sum", mean), ("count", 1), ("bucket", 1)):
            labels = {**cumulative, **({"le": str(mean)} if component == "bucket" else {})}
            samples += [
                sample(f"{family}_{component}", value, timestamp, **labels)
                for timestamp, value in ((0, 0), (15_000, final))
            ]
    # A second engine must not skew the fleet mean/tails with a reset or partial scrape.
    for component in ("sum", "count", "bucket"):
        labels = {**cumulative, "engine": "physical-b", "engine_index": "1"}
        if component == "bucket":
            labels["le"] = "100"
        samples.append(sample(f"request_prefill_time_seconds_{component}", 10, 0, **labels))
        if invalid_histogram == "missing_component" and component == "count":
            continue
        value = 5 if invalid_histogram == "reset" and component == "sum" else 110
        samples.append(sample(f"request_prefill_time_seconds_{component}", value, 15_000, **labels))
    samples += [
        sample("num_requests_running", 2, 15_000, **snapshot),
        sample("num_requests_waiting", 1, 15_000, **snapshot),
        sample("kv_cache_usage_perc", 0.25, 15_000, **snapshot),
        sample("kv_cache_usage_perc", 0.75, 15_000, engine="physical-b", engine_index="1", **snapshot),
    ]
    samples += [
        sample("num_requests_running", 1, timestamp, engine="physical-b", engine_index="1", **snapshot)
        for timestamp in range(0, 240_000, 15_000)
    ]
    samples += [
        sample("num_requests_waiting", 0, timestamp, engine="physical-b", engine_index="1", **snapshot)
        for timestamp in range(0, 240_000, 15_000)
    ]
    database.executemany(
        """INSERT INTO "telemetry_v1.marinskyrl"
           VALUES ('cw-a', 'marinskyrl', '/train', ?, 'gauge', ?, '{"worker":"driver"}', ?, ?, ?)""",
        [(*row, seq) for seq, row in enumerate(samples)],
    )

    source = Record(
        target=ClusterTarget("marin", "project", "zone", "fleet", "cluster"),
        query=lambda sql, *, max_rows: database.execute(sql).fetch_arrow_table(),
    )
    app = create_app(bridge_config(), {"marin": source}, {}, None, None, None)
    dashboard_dir = Path(__file__).parents[1] / "dashboards"
    dashboards = stitch_all(dashboard_dir, dashboard_dir / "panels")
    path = "/v1/vllm/overview"
    params = {"identity_kind": "job_id", "identity": "/train", "from": 0, "to": 240_000, "bucket_ms": 15_000}

    with TestClient(app) as client:
        response = client.get(f"/finelog/marin{path}", params={**params, "view": "token_rate"})
        saturation = client.get(f"/finelog/marin{path}", params={**params, "view": "saturation"})
        all_rows = client.get(f"/finelog/marin{path}", params=params)
        # Exercise the provisioned panel requests, including the new overview and shared fragments.
        for filename in ("inference.json", "inference_overview.json"):
            for panel in dashboards[filename]["panels"]:
                for target in panel.get("targets", []):
                    view = next(p["value"] for p in target["url_options"]["params"] if p["key"] == "view")
                    result = client.get(f"/finelog/marin{target['url']}", params={**params, "view": view})
                    assert result.status_code == 200, panel["title"]
                    for row in result.json():
                        assert row["section"] == view
                        assert all(column["selector"] in row for column in target["columns"])

    assert response.status_code == 200
    assert saturation.status_code == 200
    assert all_rows.status_code == 200
    token_rows = response.json()
    saturation_rows = saturation.json()
    assert [row["t"] for row in token_rows] == sorted(row["t"] for row in token_rows)
    assert [row["t"] for row in saturation_rows] == sorted(row["t"] for row in saturation_rows)
    assert {(row["metric"], row["t"]): row["value"] for row in token_rows} == {
        ("generated_tokens", 15_000): 10,
        ("generated_tokens", 30_000): 10,
        ("prompt_tokens", 15_000): 2,
        ("prompt_tokens", 30_000): 2,
    }
    rows = all_rows.json()
    assert {row["section"] for row in rows} == VLLM_OVERVIEW_SECTIONS
    values = {(row["section"], row["metric"], row["stat"], row["series"], row["t"]): row["value"] for row in rows}
    assert values[("request_outcome", "requests", "total", "stop", None)] == 1
    assert values[("request_rate", "requests", "rate", "stop", 15_000)] == pytest.approx(1 / 15)
    assert values[("latency", "tpot", "mean_over_time", "time per output token", 15_000)] == 0.1
    assert values[("latency", "ttft", "mean_over_time", "ttft", 15_000)] == 2
    assert values[("latency", "prefill", "mean", "prefill", None)] == 1
    assert values[("latency", "prefill", "p90", "prefill", None)] == 1
    assert values[("latency", "decode", "mean", "decode", None)] == 10
    assert values[("workload", "output_tokens", "mean", "output_tokens", None)] == 100
    histogram_rows = [row for row in rows if row["section"] in ("latency", "workload")]
    assert all(row["samples"] == 1 for row in histogram_rows)
    assert all(row["unit"] == "tokens" for row in histogram_rows if row["metric"] == "output_tokens")
    assert values[("saturation", "num_requests_running", "value", "num_requests_running", 15_000)] == 3
    assert values[("saturation", "num_requests_in_flight", "value", "num_requests_in_flight", 15_000)] == 4
    assert values[("saturation", "kv_cache_usage", "value", "kv_cache_usage", 15_000)] == 0.5
    assert values[("saturation", "kv_cache_usage_peak", "value", "kv_cache_usage_peak", 15_000)] == 0.75
    engines = {
        (row["series"].split(" @ ", 1)[0], row["metric"]): row["value"]
        for row in rows
        if row["section"] == "engine_summary"
    }
    assert engines[("physical-a", "running_mean")] == 2
    assert engines[("physical-b", "running_mean")] == 1
    assert engines[("physical-a", "waiting_mean")] == 1
    assert engines[("physical-b", "kv_cache_peak")] == 0.75
    assert engines[("physical-a", "generated_tokens_per_second")] == 10
    assert ("physical-b", "generated_tokens_per_second") not in engines  # Missing is not idle.
    freshness = {row["series"].rsplit(":", 1)[-1]: row["status"] for row in rows if row["section"] == "freshness_detail"}
    assert freshness == {"physical-a": "stale_or_stopped", "physical-b": "fresh"}
