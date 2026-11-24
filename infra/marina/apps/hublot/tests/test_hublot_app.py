# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import ClassVar

import pytest
from fastapi.testclient import TestClient
from finelog.client import LogClient
from finelog.embedded import require_embedded_server
from marina.testing import SourceServices

from infra.marina.apps.hublot.backend import app, queries

START = 1_800_000_000_000
END = START + 60_000


@dataclass
class Telemetry:
    key_column: ClassVar[str] = "timestamp_ms"
    timestamp_ms: int
    name: str
    attributes_json: str
    body_json: str = "{}"
    value: float = 1.0
    cluster: str = "fixture"
    service: str = "marinskyrl"
    execution_uid: str = "iris:fixture"
    run_id: str = "example-run"
    job_id: str = "/fixtures/trainer's-run"
    node_name: str = "worker"
    region: str = "fixture"
    process_index: int = 0
    resource_attributes_json: str = '{"internal_node_label":"worker-id"}'


@pytest.fixture
def api(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    server = require_embedded_server()(log_dir=str(tmp_path))
    url = f"http://127.0.0.1:{server.port}"
    client = LogClient.connect(url)
    try:
        table = client.get_table(queries.NAMESPACE, Telemetry)
        updates = [
            Telemetry(
                START + offset,
                "work_completed",
                json.dumps({"work_kind": "consumed_sample", "step": step}),
            )
            for step, offset in enumerate((5_000, 20_000, 20_000, 20_000, 55_000), start=1)
        ]
        labels = json.dumps(
            {
                "engine": "fixture-engine",
                "metric_source": "vllm",
                "source_kind": "histogram",
                "source_temporality": "cumulative_snapshot",
            }
        )
        timing = [
            Telemetry(START + offset, f"request_prefill_time_seconds_{sample}", labels, value=value)
            for offset, total, count in ((-5_000, 1.0, 10.0), (5_000, 2.0, 12.0))
            for sample, value in (("sum", total), ("count", count))
        ]
        wait = Telemetry(
            timestamp_ms=END + 5_000,
            name="async_phase_window",
            attributes_json='{"phase":"rollout_wait","step":5}',
            body_json=json.dumps(
                {
                    "started_unix_ms": END - 2_000,
                    "finished_unix_ms": END + 5_000,
                    "duration_seconds": 7,
                    "unrelated_text": "unneeded field",
                }
            ),
        )
        table.write(
            [
                *updates,
                *timing,
                wait,
                replace(wait, job_id=""),
                replace(updates[0], job_id="", value=99),
                replace(updates[0], timestamp_ms=START - 30_000, value=99),
                replace(updates[0], timestamp_ms=END + 30_000, value=99),
            ]
        )
        assert table.flush(timeout=10) is not None
        monkeypatch.setenv("HUBLOT_FINELOG_URL", url)
        monkeypatch.setattr(queries, "PAGE_ROWS", 2)
        with TestClient(app.create_api(SourceServices(name="hublot", data_url="", database=None))) as test_client:
            yield test_client
    finally:
        client.close()
        server.stop()


def test_full_window_reads_every_update_for_the_selected_execution(api: TestClient) -> None:
    window = {"from": START, "to": END}
    ambiguous = api.get("/run", params={"run": "example-run", **window})
    assert ambiguous.status_code == 409
    choices = {choice["job_id"]: choice["id"] for choice in ambiguous.json()["choices"]}

    response = api.get("/run", params={"run": choices["/fixtures/trainer's-run"], **window})
    assert response.status_code == 200, response.json()
    assert response.headers["cache-control"] == "private, no-store"
    payload = response.json()
    assert payload["window"] == window
    updates = [row for row in payload["rows"] if row["name"] == "work_completed"]
    assert [row["attributes_json"]["step"] for row in updates] == [1, 2, 3, 4, 5]
    assert [row["value"] for row in updates] == [1.0] * 5
    timing = [row for row in payload["rows"] if row["name"].startswith("request_prefill_time_seconds_")]
    assert [(row["timestamp_ms"], row["value"]) for row in timing] == [
        (START - 5_000, 1.0),
        (START - 5_000, 10.0),
        (START + 5_000, 2.0),
        (START + 5_000, 12.0),
    ]
    assert len({row["engineSeries"] for row in timing}) == 1
    waits = [row for row in payload["rows"] if row["name"] == "async_phase_window"]
    assert [row["body_json"] for row in waits] == [
        {"duration_seconds": 7, "started_unix_ms": END - 2_000, "finished_unix_ms": END + 5_000}
    ]
    assert "internal_node_label" not in response.text

    other = api.get("/run", params={"run": choices[""], **window})
    assert other.status_code == 200
    assert [row["value"] for row in other.json()["rows"] if row["name"] == "work_completed"] == [99.0]


def test_window_over_the_record_limit_returns_no_partial_run(api: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "MAX_WINDOW_ROWS", 4)
    response = api.get("/run", params={"run": "/fixtures/trainer's-run", "from": START, "to": END})
    assert response.status_code == 422
    assert "Load a shorter window" in response.json()["error"]
    assert "rows" not in response.json()
