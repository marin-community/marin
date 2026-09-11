# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from datetime import UTC, datetime
from unittest.mock import MagicMock

from finelog.client import LogClient
from starlette.testclient import TestClient
from zephyr.coordinator import PullStatus
from zephyr.dataset import Dataset
from zephyr.plan import compute_plan
from zephyr.shuffle import ListShard
from zephyr.stage_io import ShardTask
from zephyr.stats import StatsWriter
from zephyr.testing.coordinator import TEST_TASK_COST, make_test_coordinator, start_test_stage
from zephyr.worker_context import CounterEntry, CounterSnapshot


def _api(client: TestClient, path: str, **params) -> dict:
    response = client.get(f"/api/{path}", params=params)
    assert response.status_code == 200, response.text
    return response.json()


def _task(shard: int, total_shards: int = 1) -> ShardTask:
    return ShardTask(
        shard_idx=shard,
        total_shards=total_shards,
        shard=ListShard(refs=[]),
        operations=[],
        stage_name="stage0-Map",
        cost=TEST_TASK_COST,
    )


def test_dashboard_keeps_pipeline_data_separate(coordinator):
    secret = "source-value-must-not-leak"
    left = Dataset.from_list([{"id": 1}])
    right = Dataset.from_list([{"id": 1, "secret": secret}])
    joined = left.sorted_merge_join(right, left_key=lambda item: item["id"], right_key=lambda item: item["id"])
    start_test_stage(coordinator, [_task(0)], plan=compute_plan(joined), execution_id="join")
    start_test_stage(
        coordinator,
        [_task(0, 2), _task(1, 2)],
        plan=compute_plan(Dataset.from_list([1]).map(lambda value: value + 1)),
        execution_id="map",
    )
    coordinator.register_worker("worker", MagicMock())
    coordinator.heartbeat(
        "worker",
        {
            "join": CounterSnapshot(counters={"records": CounterEntry(12)}, generation=1),
            "map": CounterSnapshot(counters={"records": CounterEntry(40)}, generation=1),
        },
    )
    pull_status, work = coordinator.pull_task("worker", TEST_TASK_COST)
    assert pull_status is PullStatus.RUN_TASK
    assert work is not None

    with TestClient(coordinator.web_application) as client:
        assert {row["execution_id"] for row in _api(client, "pipelines")["pipelines"]} == {"join", "map"}
        plan = _api(client, "plan", execution_id="join")
        assert any(node["auxiliary"] for node in plan["nodes"])
        assert secret not in str(plan)
        other_plan = _api(client, "plan", execution_id="map")
        assert not any(node["auxiliary"] for node in other_plan["nodes"])

        for execution_id, total_shards, records in [("join", 1, 12), ("map", 2, 40)]:
            assert _api(client, "status", execution_id=execution_id)["total_shards"] == total_shards
            counters = _api(client, "counters", execution_id=execution_id)["counters"]
            assert {row["name"]: row["value"] for row in counters} == {"records": records}

        workers = _api(client, "workers")["workers"]
        assert workers[0]["assignments"] == [{"execution_id": work.execution_id, "shard": work.task.shard_idx}]
        page = client.get("/", headers={"x-forwarded-prefix": "/proxy/coordinator"})
        assert page.status_code == 200
        assert '<base href="/proxy/coordinator/"' in page.text


def test_dashboard_metrics_keep_recent_bins_and_reject_unknown_execution(actor_context, tmp_path, monkeypatch):
    log_client = MagicMock(spec=LogClient)
    log_client.query.return_value.to_pylist.return_value = [
        {
            "time_bin": datetime(2026, 8, 2, 10, 15, tzinfo=UTC),
            "stage_name": "older",
            "item_rate": 1.0,
            "byte_rate": 2.0,
            "cpu_cores": 0.5,
            "memory_bytes": 1024.0,
        },
        {
            "time_bin": datetime(2026, 8, 2, 10, 30),
            "stage_name": "stage-b",
            "item_rate": None,
            "byte_rate": None,
            "cpu_cores": None,
            "memory_bytes": None,
        },
        {
            "time_bin": datetime(2026, 8, 2, 10, 30, tzinfo=UTC),
            "stage_name": "stage-a",
            "item_rate": 125.5,
            "byte_rate": 4096.0,
            "cpu_cores": 1.25,
            "memory_bytes": 2048.0,
        },
    ]
    monkeypatch.setattr("zephyr.coordinator.StatsWriter.connect", lambda: StatsWriter(log_client))
    coordinator = make_test_coordinator(tmp_path)
    start_test_stage(coordinator, [], execution_id="active")
    try:
        with TestClient(coordinator.web_application) as client:
            metrics = _api(client, "metrics", execution_id="active", max_points=1)
            assert [point["stage"] for point in metrics["points"]] == ["stage-a", "stage-b"]
            assert metrics["points"][0]["cpu_cores"] == 1.25
            assert metrics["points"][1]["item_rate"] == 0

            unknown = _api(client, "metrics", execution_id="another-job")
            assert unknown["points"] == []
            assert unknown["warning"]
    finally:
        coordinator.shutdown()


def test_dashboard_reports_pipeline_failure(coordinator):
    plan = compute_plan(Dataset.from_list([1]).map(lambda value: value + 1))
    start_test_stage(coordinator, [_task(0)], plan=plan, execution_id="failed", stage_name="stage0-Map")
    coordinator.register_worker("worker", MagicMock())
    for _ in range(3):
        pull_status, work = coordinator.pull_task("worker", TEST_TASK_COST)
        assert pull_status is PullStatus.RUN_TASK
        assert work is not None
        coordinator.report_error(
            "worker", work.execution_id, work.task.shard_idx, work.attempt, "stage failed", work.stage_generation
        )

    with TestClient(coordinator.web_application) as client:
        status = _api(client, "status", execution_id="failed")
        assert status["phase"] == "failed"
        assert status["fatal_error"]
        assert [node["state"] for node in status["node_statuses"]] == ["succeeded", "failed"]


def test_dashboard_waits_for_worker_recovery(coordinator):
    start_test_stage(coordinator, [], execution_id="exec")
    with TestClient(coordinator.web_application) as client:
        assert _api(client, "status")["phase"] == "waiting_for_workers"
        coordinator.register_worker("worker", MagicMock())
        assert _api(client, "status")["phase"] == "running"
        coordinator.check_heartbeats(timeout=0.0)
        assert _api(client, "status")["phase"] == "waiting_for_workers"
        coordinator.register_worker("replacement", MagicMock())
        assert _api(client, "status")["phase"] == "running"


def test_dashboard_counter_stages_include_rows_outside_the_page(coordinator):
    start_test_stage(coordinator, [], execution_id="exec")
    coordinator.register_worker("worker", MagicMock())
    coordinator.heartbeat(
        "worker",
        {
            "exec": CounterSnapshot(
                counters={"a": CounterEntry(1, stage="first"), "b": CounterEntry(2, stage="second")},
                generation=1,
            )
        },
    )
    with TestClient(coordinator.web_application) as client:
        page = _api(client, "counters", execution_id="exec", limit=1)
        assert [row["stage"] for row in page["counters"]] == ["first"]
        assert page["stages"] == ["first", "second"]
        filtered = _api(client, "counters", execution_id="exec", stage="second", search="b")
        assert [row["name"] for row in filtered["counters"]] == ["b"]
        assert filtered["stages"] == ["first", "second"]
