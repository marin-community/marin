# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior of the coordinator dashboard's fleet snapshot."""

from dataclasses import replace
from unittest.mock import MagicMock

import pytest
from starlette.testclient import TestClient
from zephyr.dashboard.app import PlanNodeState
from zephyr.dashboard.coordinator import MAX_FINISHED_PIPELINES
from zephyr.dataset import Dataset
from zephyr.plan import compute_plan, plan_nodes
from zephyr.shuffle import ListShard
from zephyr.stage_io import ShardTask, TaskResult
from zephyr.testing.coordinator import TEST_TASK_COST, TEST_WORKER_AVAILABLE, start_test_stage
from zephyr.worker_context import CounterSnapshot, ShardProgress


def test_overview_reports_current_shard_heatmap_for_multiple_pipelines(coordinator):
    plan = compute_plan(Dataset.from_list(list(range(40))).map(lambda item: item))
    active_node = next(node for node in plan_nodes(plan) if node.stage_name)
    tasks = [
        ShardTask(
            shard_idx=index,
            total_shards=40,
            shard=ListShard(refs=[]),
            operations=[],
            stage_name=active_node.stage_name,
            cost=TEST_TASK_COST,
        )
        for index in range(40)
    ]
    run = start_test_stage(coordinator, tasks, stage_name=active_node.stage_name, execution_id="first")
    run.plan = plan
    run.pipeline_name = "pipeline with a long name"
    run.node_states[active_node.node_id] = PlanNodeState.RUNNING
    run.results = {index: MagicMock() for index in (0, 1, 39)}
    run.completed_shards = 3
    run.in_flight = {20: MagicMock(worker_id="worker-0")}
    start_test_stage(coordinator, [], execution_id="second")

    client = TestClient(coordinator.web_application)
    response = client.get("/api/overview")

    assert response.status_code == 200
    first, second = response.json()["pipelines"]
    assert second["status"]["execution_id"] == "second"
    assert first["plan"]["pipeline_name"] == "pipeline with a long name"
    node = next(item for item in first["status"]["node_statuses"] if item["node_id"] == active_node.node_id)
    assert node["total_shards"] == 40
    assert node["completed_shards"] == 3
    assert len(node["shard_buckets"]) == 32
    assert sum(bucket["completed"] for bucket in node["shard_buckets"]) == 3
    assert sum(bucket["running"] for bucket in node["shard_buckets"]) == 1
    assert sum(bucket["pending"] for bucket in node["shard_buckets"]) == 36

    shards = client.get("/api/shards", params={"execution_id": "first", "offset": 19, "limit": 22}).json()
    assert shards["total"] == 40
    assert [shard["state"] for shard in shards["shards"]] == [
        "pending",
        "running",
        *(["pending"] * 18),
        "succeeded",
    ]

    run.terminal_error = RuntimeError("failed stage")
    failed = client.get("/api/shards", params={"execution_id": "first", "offset": 19, "limit": 2}).json()
    assert [shard["state"] for shard in failed["shards"]] == ["stopped", "running"]


@pytest.mark.parametrize("failed", [False, True])
def test_released_pipeline_retains_terminal_summary_without_payloads(coordinator, tmp_path, failed):
    execution_id = "finished"
    run = start_test_stage(coordinator, [], execution_id=execution_id)
    run.plan = compute_plan(Dataset.from_list(["private-source-value"]).map(str.upper))
    run.started_at_ms = 10
    if failed:
        run.terminal_error = RuntimeError("grader failed")
    run.finish(storage_cleanup_safe=True)
    path = tmp_path / "chunks" / execution_id
    path.mkdir(parents=True)
    (path / "shared.pkl").write_text("private-source-value")
    start_test_stage(coordinator, [], execution_id="active").started_at_ms = 20
    client = TestClient(coordinator.web_application)
    before = client.get("/api/overview").json()["pipelines"]

    coordinator.release_execution(execution_id)

    response = client.get("/api/overview")
    finished, active = response.json()["pipelines"]
    assert finished["status"] == before[0]["status"]
    assert finished["plan"] == before[0]["plan"]
    assert finished["archived"]
    assert finished["status"]["phase"] == ("failed" if failed else "succeeded")
    assert active["status"]["execution_id"] == "active"
    assert not active["archived"]
    assert "private-source-value" not in response.text
    assert not path.exists()
    assert client.get("/api/counters", params={"execution_id": execution_id}).json()["counters"] == []


def test_finished_history_is_bounded_and_keeps_active_pipelines(coordinator):
    start_test_stage(coordinator, [], execution_id="active").started_at_ms = 1
    for index in range(MAX_FINISHED_PIPELINES + 2):
        execution_id = f"finished-{index}"
        run = start_test_stage(coordinator, [], execution_id=execution_id)
        run.started_at_ms = index + 2
        run.finish(storage_cleanup_safe=False)
        coordinator.release_execution(execution_id)

    pipelines = TestClient(coordinator.web_application).get("/api/overview").json()["pipelines"]
    assert [item["status"]["execution_id"] for item in pipelines] == [
        "active",
        *(f"finished-{index}" for index in range(2, MAX_FINISHED_PIPELINES + 2)),
    ]


def test_shards_keep_concurrent_mapper_progress_separate_and_reject_stale_attempts(coordinator):
    tasks = [
        ShardTask(
            shard_idx=index,
            total_shards=3,
            shard=ListShard(refs=[]),
            operations=[],
            stage_name="map",
            cost=TEST_TASK_COST,
        )
        for index in range(3)
    ]
    run = start_test_stage(coordinator, tasks, stage_name="map", execution_id="mappers")
    coordinator.register_worker("worker", MagicMock(), task_id="/test/job/worker/0")
    for _ in range(2):
        coordinator.pull_task("worker", TEST_WORKER_AVAILABLE)
    first = ShardProgress(run.stage_generation, 0, 20, 40, 40, 100)
    second = replace(first, records_in=80, input_rows_completed=80)
    coordinator.heartbeat("worker", {"mappers": CounterSnapshot({}, 2, {0: first, 1: second})})
    coordinator.heartbeat("worker", {"mappers": CounterSnapshot({}, 1, {0: second})})
    client = TestClient(coordinator.web_application)

    page = client.get("/api/shards", params={"execution_id": "mappers", "state": "running", "limit": 1}).json()
    assert (page["total"], page["matching"]) == (3, 2)
    assert page["shards"][0]["progress"]["input_rows_completed"] == 40
    assert page["shards"][0]["task_id"] == "/test/job/worker/0"
    other = client.get("/api/shards", params={"execution_id": "mappers", "state": "running", "offset": 1}).json()
    assert other["shards"][0]["progress"]["input_rows_completed"] == 80
    pending = client.get("/api/shards", params={"execution_id": "mappers", "state": "pending"}).json()
    assert [item["index"] for item in pending["shards"]] == [2]
    assert pending["shards"][0]["progress"] is None

    coordinator.report_error("worker", "mappers", 0, 0, "retry", run.stage_generation)
    # Dispatch the queued third shard, then the retry on the same worker.
    for _ in range(2):
        coordinator.pull_task("worker", TEST_WORKER_AVAILABLE)
    coordinator.heartbeat("worker", {"mappers": CounterSnapshot({}, 3, {0: first, 1: second})})
    retried = client.get("/api/shards", params={"execution_id": "mappers"}).json()["shards"]
    assert retried[0]["state"] == "running"
    assert retried[0]["progress"] is None
    assert retried[1]["progress"]["input_rows_completed"] == 80

    fresh = replace(first, attempt=1, input_rows_completed=10)
    old_stage = replace(fresh, stage_generation=run.stage_generation - 1)
    coordinator.heartbeat("worker", {"mappers": CounterSnapshot({}, 4, {0: old_stage, 1: second})})
    assert client.get("/api/shards", params={"execution_id": "mappers"}).json()["shards"][0]["progress"] is None
    coordinator.heartbeat("worker", {"mappers": CounterSnapshot({}, 5, {0: fresh, 1: second})})
    current = client.get("/api/shards", params={"execution_id": "mappers"}).json()["shards"][0]
    assert current["progress"]["input_rows_completed"] == 10
    coordinator.report_result(
        "worker",
        "mappers",
        0,
        1,
        TaskResult(shard=ListShard(refs=[])),
        CounterSnapshot.empty(6),
        run.stage_generation,
    )
    completed = client.get("/api/shards", params={"execution_id": "mappers", "state": "succeeded"}).json()
    assert completed["matching"] == 1
    assert completed["shards"][0]["state"] == "succeeded"
    assert completed["shards"][0]["progress"] is None
    remaining = client.get("/api/shards", params={"execution_id": "mappers", "state": "running"}).json()
    assert remaining["shards"][0]["index"] == 1
    assert remaining["shards"][0]["progress"]["input_rows_completed"] == 80
