# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Iterator
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from time import time
from typing import ClassVar

import pytest
from fastapi.testclient import TestClient
from finelog.client import LogClient
from finelog.embedded import require_embedded_server
from marina.testing import SourceServices

from infra.marina.apps.hublot.backend import app, queries

START = int(time()) * 1_000 - 4 * 3_600_000
END = START + 60_000
RUN_END = START + 7_200_000
JOB = "/fixtures/trainer's-run"


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
    job_id: str = JOB
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
            for step, offset in enumerate((5_000, 20_000, 20_000, 20_000, 55_000, RUN_END - START - 5_000), start=1)
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
        generation = [
            Telemetry(START + offset, "generation_tokens_total", labels, value=value)
            for offset, value in (
                (-5_000, 10),
                (5_000, 110),
                (15_000, 210),
                (25_000, 310),
                (35_000, 410),
                (40_000, 5),
                (50_000, 105),
            )
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
                *generation,
                Telemetry(
                    START + 20_000,
                    "kv_cache_usage_perc",
                    '{"metric_source":"vllm","source_temporality":"current_snapshot","engine":"fixture-engine"}',
                    value=0.2,
                ),
                Telemetry(
                    START,
                    "lifecycle",
                    '{"role":"controller"}',
                    body_json='{"state":"started"}',
                ),
                Telemetry(RUN_END, "terminal", '{"role":"controller"}'),
                Telemetry(START + 80_000, "async_run_configuration", "{}", body_json='{"mini_batch_size":128}'),
                Telemetry(START + 85_000, "rollout_capacity", "{}", value=256),
                Telemetry(
                    START + 90_000,
                    "training_metric_value",
                    '{"metric":"async/performance/configured_inference_gpus"}',
                    value=8,
                ),
                Telemetry(START + 95_000, "training_metric_value", '{"metric":"generate/outcome/graded"}', value=4),
                wait,
                replace(wait, job_id=""),
                replace(updates[0], job_id="", value=99),
                replace(updates[0], timestamp_ms=START - 30_000, value=99),
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
    ambiguous = api.get("/run", params={"run": "example-run"})
    assert ambiguous.status_code == 409
    choices = {choice["job_id"]: choice["id"] for choice in ambiguous.json()["choices"]}

    params = {"run": choices[JOB], **window}
    resolved = api.get("/run", params={"run": choices[JOB]})
    assert resolved.status_code == 200, resolved.json()
    assert resolved.json()["window"] == {"from": START, "to": RUN_END + 1}
    assert "rows" not in resolved.json()

    response = api.get("/window", params=params)
    assert response.status_code == 200, response.json()
    assert response.headers["cache-control"] == "private, no-store"
    payload = response.json()
    updates = [row for row in payload["rows"] if row["name"] == "work_completed"]
    assert [row["attributes_json"]["step"] for row in updates] == [1, 2, 3, 4, 5]
    assert [row["value"] for row in updates] == [1.0] * 5
    assert {row["name"] for row in payload["rows"]} == {"work_completed", "async_phase_window"}
    waits = [row for row in payload["rows"] if row["name"] == "async_phase_window"]
    assert [row["body_json"] for row in waits] == [
        {"duration_seconds": 7, "started_unix_ms": END - 2_000, "finished_unix_ms": END + 5_000}
    ]

    detail = api.get("/window", params={**params, "card": "inferenceTiming"})
    assert detail.status_code == 200, detail.json()
    timing = detail.json()["rows"]
    assert all(row["name"].startswith("request_prefill_time_seconds_") for row in timing)
    assert [(row["timestamp_ms"], row["value"]) for row in timing] == [
        (START - 5_000, 1.0),
        (START - 5_000, 10.0),
        (START + 5_000, 2.0),
        (START + 5_000, 12.0),
    ]
    assert len({row["engineSeries"] for row in timing}) == 1
    assert "internal_node_label" not in detail.text

    other = api.get("/window", params={"run": choices[""], **window})
    assert other.status_code == 200
    assert [row["value"] for row in other.json()["rows"] if row["name"] == "work_completed"] == [99.0]


def test_long_run_overview_and_selected_metrics_use_different_windows(
    api: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolved = api.get("/run", params={"run": "iris:fixture"})
    assert resolved.status_code == 200, resolved.json()
    payload = resolved.json()
    assert payload["window"] == {"from": START, "to": RUN_END + 1}
    params = {"run": payload["run"]["id"], **payload["window"]}

    overview = api.get("/window", params=params)
    assert overview.status_code == 200, overview.json()
    timeline = overview.json()
    steps = [row["attributes_json"]["step"] for row in timeline["rows"] if row["name"] == "work_completed"]
    assert steps == [1, 2, 3, 4, 5, 6]
    assert timeline["configuration"] == {"mini_batch_size": 128}
    assert timeline["runSettings"] == {
        "inferenceGpus": 8,
        "policyGpus": None,
        "bufferCapacity": 256,
        "separateGrading": True,
    }
    summary = api.get("/window", params={**params, "card": "generation"})
    assert summary.status_code == 200, summary.json()
    assert summary.json()["bucketMs"] == 30_000
    assert [row["value"] for row in summary.json()["buckets"]] == pytest.approx([300, 150])
    inference = api.get("/window", params={**params, "card": "inferenceTiming"})
    assert inference.status_code == 200, inference.json()
    [prefill] = inference.json()["buckets"]
    assert prefill["name"] == "request_prefill_time_seconds_sum"
    assert (prefill["value"], prefill["requests"]) == pytest.approx((0.5, 1))
    cache = api.get("/window", params={**params, "card": "inferenceCache"})
    assert cache.status_code == 200, cache.json()
    [bucket] = cache.json()["buckets"]
    assert bucket["name"] == "kv_cache_usage_perc"
    assert (bucket["value"], bucket["engines"]) == pytest.approx((0.2, 1))

    defaults = api.get("/window", params={**params, "card": "inferenceDefaults"})
    other = api.get("/window", params={**params, "card": "inferenceOtherTabs"})
    assert defaults.status_code == other.status_code == 200
    assert defaults.json()["buckets"] == summary.json()["buckets"]
    assert other.json()["buckets"] == sorted(
        inference.json()["buckets"] + cache.json()["buckets"], key=lambda row: (row["bucket_start"], row["name"])
    )
    cursor = defaults.json()["next"]
    assert cursor == START + app.INFERENCE_TAB_FIRST_PAGE_MS
    rest = api.get("/window", params={**params, "card": "inferenceDefaults", "after": cursor})
    assert rest.status_code == 200
    assert rest.json()["next"] is None
    assert rest.json()["buckets"] == []

    fleet = api.get("/window", params={**params, "to": END, "card": "fleet"})
    assert fleet.status_code == 200, fleet.json()
    buckets = fleet.json()["buckets"]
    assert {row["name"] for row in buckets} == {"generation_tokens_total"}
    assert [row["value"] for row in buckets] == pytest.approx([300, 150])

    monkeypatch.setattr(app, "BACKGROUND_READ_PAUSE_MS", 60_000)
    sample_params = {**params, "card": "fleet", "until": f"{START + 30_000},{START + 120_000}"}
    samples = api.get("/samples", params=sample_params)
    assert samples.status_code == 200, samples.json()
    assert samples.json()["samples"] == [
        {"from": START, "to": START + 30_000},
        {"from": START + 90_000, "to": START + 120_000},
    ]
    [bucket] = samples.json()["buckets"]
    assert bucket["bucket_start"] == START
    assert bucket["value"] == pytest.approx(300)
    paced = api.get("/samples", params=sample_params)
    assert paced.status_code == 429
    assert int(paced.headers["Retry-After"]) > 0
    paced_detail = api.get("/window", params={**params, "to": END, "card": "generation", "background": True})
    assert paced_detail.status_code == 429

    detail = api.get("/window", params={**params, "to": END, "card": "generation"})
    assert detail.status_code == 200, detail.json()
    assert [row["value"] for row in detail.json()["rows"]] == [10, 110, 210, 310, 410, 5, 105]


def test_date_hint_opens_the_full_older_run(api: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "LOOKBACK_DAYS", 0)
    assert api.get("/run", params={"run": "iris:fixture"}).status_code == 404
    date = datetime.fromtimestamp(START / 1000, UTC).date().isoformat()
    response = api.get("/run", params={"run": "iris:fixture", "date": date})
    assert response.status_code == 200, response.json()
    assert response.json()["window"] == {"from": START, "to": RUN_END + 1}


def test_metrics_pages_keep_counter_deltas_at_the_seam(api: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "RAW_METRIC_WINDOW_MS", 30_000)
    monkeypatch.setattr(app, "BACKGROUND_SUMMARY_SCAN_MS", 30_000)
    params = {"run": f"fixture@iris:fixture@{JOB}", "from": START, "to": END, "card": "generation"}
    first = api.get("/window", params={**params, "background": True})
    assert first.status_code == 200, first.json()
    assert first.json()["next"] == START + 30_000
    second = api.get("/window", params={**params, "after": first.json()["next"]})
    assert second.status_code == 200, second.json()
    assert second.json()["next"] is None
    rows = first.json()["buckets"] + second.json()["buckets"]
    assert [row["bucket_start"] for row in rows] == [START, START + 30_000]
    assert [row["value"] for row in rows] == pytest.approx([300, 150])


def test_window_over_the_record_limit_returns_no_partial_run(api: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "MAX_WINDOW_ROWS", 4)
    response = api.get("/window", params={"run": f"fixture@iris:fixture@{JOB}", "from": START, "to": END})
    assert response.status_code == 422
    assert "Load a shorter window" in response.json()["error"]
    assert "rows" not in response.json()
