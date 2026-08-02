# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the coordinator-owned dashboard boundary."""

from unittest.mock import MagicMock

from conftest import _TEST_TASK_COST, _make_test_coordinator, start_test_stage
from starlette.testclient import TestClient
from zephyr.coordinator import PullStatus
from zephyr.dashboard import service_path
from zephyr.dataset import Dataset
from zephyr.plan import PhysicalPlan, compute_plan
from zephyr.shuffle import ListShard
from zephyr.stage_io import ShardTask
from zephyr.stats import ZEPHYR_WORKER_CPU_PCT_CURRENT_KEY, ZEPHYR_WORKER_MEM_CURRENT_KEY
from zephyr.worker_context import CounterEntry, CounterSnapshot


def _rpc(client: TestClient, method: str, request: dict | None = None) -> dict:
    response = client.post(
        f"/{service_path(method)}",
        json=request or {},
        headers={"connect-protocol-version": "1"},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _start_pipeline(
    coordinator,
    plan: PhysicalPlan,
    execution_id: str,
    pipeline_name: str,
    *,
    stage_name: str = "stage0-Map",
    tasks: list[ShardTask] | None = None,
):
    return start_test_stage(
        coordinator,
        tasks or [],
        plan=plan,
        pipeline_name=pipeline_name,
        execution_id=execution_id,
        stage_name=stage_name,
    )


def _task(shard: int, total_shards: int) -> ShardTask:
    return ShardTask(
        shard_idx=shard,
        total_shards=total_shards,
        shard=ListShard(refs=[]),
        operations=[],
        stage_name="stage0-Map",
        cost=_TEST_TASK_COST,
    )


def test_dashboard_lists_and_selects_concurrent_pipelines(actor_context, tmp_path):
    secret_source_value = "source-value-must-not-leak"
    first_plan = compute_plan(Dataset.from_list([secret_source_value, "second"]).map(lambda value: value.upper()))
    second_plan = compute_plan(Dataset.from_list([1]).filter(lambda value: value > 0))
    coordinator = _make_test_coordinator(tmp_path)
    _start_pipeline(coordinator, first_plan, "exec-map", "document-cleaning")
    _start_pipeline(coordinator, second_plan, "exec-filter", "quality-filter", stage_name="stage0-Filter")

    try:
        with TestClient(coordinator.web_application) as client:
            pipelines = _rpc(client, "ListPipelines")
            assert [pipeline["executionId"] for pipeline in pipelines["pipelines"]] == [
                "exec-filter",
                "exec-map",
            ]
            assert [pipeline["pipelineName"] for pipeline in pipelines["pipelines"]] == [
                "quality-filter",
                "document-cleaning",
            ]

            response = _rpc(client, "GetPlan", {"executionId": "exec-map"})
            assert response["pipelineName"] == "document-cleaning"
            assert response["executionId"] == "exec-map"
            assert response["sourceItemCount"] == "2"
            assert response["nodes"][0]["stageType"] == "SOURCE"
            assert "Map" in response["nodes"][1]["operationTypes"]
            assert secret_source_value not in str(response)

            other = _rpc(client, "GetPlan", {"executionId": "exec-filter"})
            assert other["pipelineName"] == "quality-filter"
            assert other["executionId"] == "exec-filter"
            assert other["sourceItemCount"] == "1"

            page = client.get("/", headers={"x-forwarded-prefix": "/proxy/private-coordinator"})
            assert page.status_code == 200
            assert '<base href="/proxy/private-coordinator/"' in page.text
    finally:
        coordinator.shutdown()


def test_dashboard_scopes_live_counters_and_status_by_pipeline(actor_context, tmp_path, monkeypatch):
    plan = compute_plan(Dataset.from_list([1]).map(lambda value: value + 1))
    job_info = MagicMock()
    job_info.task_id.to_wire.return_value = "user/coordinator/0"
    monkeypatch.setattr("zephyr.coordinator.get_job_info", lambda: job_info)
    coordinator = _make_test_coordinator(tmp_path, expected_workers=2)
    _start_pipeline(coordinator, plan, "exec-a", "pipeline-a", tasks=[_task(0, 1)])
    _start_pipeline(coordinator, plan, "exec-b", "pipeline-b", tasks=[_task(0, 2), _task(1, 2)])
    coordinator.register_worker("worker-0", MagicMock(), "user/workers/0")
    coordinator.heartbeat(
        "worker-0",
        {
            "exec-a": CounterSnapshot(
                counters={
                    "records-a": CounterEntry(12, stage="Map"),
                    ZEPHYR_WORKER_CPU_PCT_CURRENT_KEY: CounterEntry(125.0, stage="Map"),
                    ZEPHYR_WORKER_MEM_CURRENT_KEY: CounterEntry(256, stage="Map"),
                },
                generation=1,
            ),
            "exec-b": CounterSnapshot(
                counters={"records-b": CounterEntry(40, stage="Map")},
                generation=1,
            ),
        },
    )

    try:
        with TestClient(coordinator.web_application) as client:
            status = _rpc(client, "GetStatus", {"executionId": "exec-a"})
            assert status["executionId"] == "exec-a"
            assert status.get("completedShards", 0) == 0
            assert status["totalShards"] == 1
            assert status["expectedWorkers"] == 2
            assert status["coordinatorTaskId"] == "user/coordinator/0"
            assert status["workerStates"] == [{"state": "active", "count": 1}]
            assert status["resources"]["cpuCores"] == 1.25
            assert status["resources"]["memoryBytes"] == "256"

            counters = _rpc(client, "ListCounters", {"executionId": "exec-a", "limit": 10})
            assert {counter["name"] for counter in counters["counters"]} == {
                ZEPHYR_WORKER_CPU_PCT_CURRENT_KEY,
                ZEPHYR_WORKER_MEM_CURRENT_KEY,
                "records-a",
            }
            assert "records-b" not in str(counters)

            other = _rpc(client, "ListCounters", {"executionId": "exec-b", "limit": 10})
            assert [counter["name"] for counter in other["counters"]] == ["records-b"]

            workers = _rpc(client, "ListWorkers", {"limit": 10})
            assert workers["total"] == 1
            assert workers["workers"][0]["taskId"] == "user/workers/0"
            assert workers["workers"][0]["cpuPercent"] == 125

            metrics = _rpc(client, "GetMetrics", {"executionId": "exec-a", "maxPoints": 10})
            assert metrics["warning"] == "Finelog is not available for this coordinator."
    finally:
        coordinator.shutdown()


def test_dashboard_worker_assignments_include_pipeline(actor_context, tmp_path):
    plan = compute_plan(Dataset.from_list([1]).map(lambda value: value + 1))
    coordinator = _make_test_coordinator(tmp_path)
    start_test_stage(
        coordinator,
        [_task(3, 4)],
        plan=plan,
        pipeline_name="pipeline-a",
        execution_id="exec-a",
        stage_name="stage0-Map",
    )
    coordinator.register_worker("worker-0", MagicMock())
    status, work = coordinator.pull_task("worker-0", _TEST_TASK_COST)
    assert status is PullStatus.RUN_TASK
    assert work is not None

    try:
        with TestClient(coordinator.web_application) as client:
            workers = _rpc(client, "ListWorkers", {"limit": 10})
            assert workers["workers"][0]["assignments"] == [{"executionId": "exec-a", "shard": 3}]
    finally:
        coordinator.shutdown()


def test_dashboard_plan_includes_join_input_without_source_values(actor_context, tmp_path):
    private_right_value = "right-source-value-must-not-leak"
    left = Dataset.from_list([{"id": 1, "value": "left"}])
    right = Dataset.from_list([{"id": 1, "value": private_right_value}])
    joined = left.sorted_merge_join(right, left_key=lambda item: item["id"], right_key=lambda item: item["id"])
    plan = compute_plan(joined)
    coordinator = _make_test_coordinator(tmp_path)
    _start_pipeline(coordinator, plan, "join-exec", "join")
    try:
        with TestClient(coordinator.web_application) as client:
            response = _rpc(client, "GetPlan", {"executionId": "join-exec"})
            auxiliary_nodes = [node for node in response["nodes"] if node.get("auxiliary")]
            assert auxiliary_nodes
            assert all("/join/" in node["nodeId"] for node in auxiliary_nodes)
            assert private_right_value not in str(response)
    finally:
        coordinator.shutdown()


def test_dashboard_reports_selected_pipeline_failure(actor_context, tmp_path):
    plan = compute_plan(Dataset.from_list([1]).map(lambda value: value + 1))
    coordinator = _make_test_coordinator(tmp_path)
    _start_pipeline(coordinator, plan, "failed-exec", "failed", tasks=[_task(0, 1)])
    coordinator.register_worker("worker-0", MagicMock())
    for _ in range(3):
        pull_status, work = coordinator.pull_task("worker-0", _TEST_TASK_COST)
        assert pull_status is PullStatus.RUN_TASK
        assert work is not None
        coordinator.report_error(
            "worker-0",
            work.execution_id,
            work.task.shard_idx,
            work.attempt,
            "test stage failed",
            work.stage_generation,
        )
    try:
        with TestClient(coordinator.web_application) as client:
            status = _rpc(client, "GetStatus", {"executionId": "failed-exec"})
            assert status["phase"] == "PIPELINE_PHASE_FAILED"
            assert "test stage failed" in status["fatalError"]
            failed_nodes = [node for node in status["nodeStatuses"] if node["state"] == "PLAN_NODE_STATE_FAILED"]
            assert [node["nodeId"] for node in failed_nodes] == ["main/stage/0"]
    finally:
        coordinator.shutdown()
