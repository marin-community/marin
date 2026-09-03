# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace as Record

import duckdb
from config import ClusterTarget
from conftest import bridge_config
from server import create_app
from starlette.testclient import TestClient
from vllm_observability import VLLM_OVERVIEW_SECTIONS


def test_dashboard_vllm_overview_end_to_end():
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
    samples = [
        sample("generation_tokens_total", value, timestamp, **cumulative) for timestamp, value in ((0, 0), (15_000, 150))
    ]
    samples += [
        sample("request_success_total", value, timestamp, finished_reason="stop", **cumulative)
        for timestamp, value in ((0, 0), (15_000, 1))
    ]
    for component, final in (("sum", 0.1), ("count", 1), ("bucket", 1)):
        labels = {**cumulative, **({"le": "0.1"} if component == "bucket" else {})}
        samples += [
            sample(f"request_time_per_output_token_seconds_{component}", value, timestamp, **labels)
            for timestamp, value in ((0, 0), (15_000, final))
        ]
    samples.append(sample("num_requests_running", 2, 15_000, source_temporality="current_snapshot", step="1"))
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
    dashboard = json.loads((Path(__file__).parents[1] / "dashboards" / "inference.json").read_text())
    targets = (target for panel in dashboard["panels"] for target in panel.get("targets", []))
    path = next(target["url"] for target in targets if target.get("url") == "/v1/vllm/overview")
    params = {"identity_kind": "job_id", "identity": "/train", "from": 0, "to": 30_000, "bucket_ms": 15_000}

    with TestClient(app) as client:
        response = client.get(f"/finelog/marin{path}", params={**params, "view": "token_rate"})
        all_rows = client.get(f"/finelog/marin{path}", params=params)

    assert response.status_code == 200
    assert [(row["metric"], row["value"]) for row in response.json()] == [("generated_tokens", 10)]
    assert {row["section"] for row in all_rows.json()} == VLLM_OVERVIEW_SECTIONS
