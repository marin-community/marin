# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the coordinator-owned dashboard boundary."""

import hashlib
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from conftest import _TEST_TASK_COST, _make_test_coordinator, start_test_stage
from rigging.timing import Duration, ExponentialBackoff
from starlette.testclient import TestClient
from zephyr.coordinator import PullStatus
from zephyr.dataset import Dataset
from zephyr.plan import PhysicalPlan, compute_plan
from zephyr.shuffle import ListShard
from zephyr.stage_io import ShardTask, ZephyrWorkerError
from zephyr.stats import ZEPHYR_WORKER_CPU_PCT_CURRENT_KEY, ZEPHYR_WORKER_MEM_CURRENT_KEY
from zephyr.worker_context import CounterEntry, CounterSnapshot

_DASHBOARD_ROOT = Path(__file__).parents[1] / "dashboard"
_DASHBOARD_ASSET = Path(__file__).parents[1] / "src" / "zephyr" / "dashboard.html"


def _api(client: TestClient, path: str, params: dict | None = None) -> dict:
    response = client.get(f"/api/{path}", params=params or {})
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


def test_dashboard_asset_matches_frontend_sources():
    source_files = [
        *(_DASHBOARD_ROOT / "src").rglob("*"),
        *(_DASHBOARD_ROOT / "scripts").rglob("*"),
        *(
            _DASHBOARD_ROOT / name
            for name in [
                "env.d.ts",
                "package-lock.json",
                "package.json",
                "postcss.config.cjs",
                "rsbuild.config.ts",
                "tailwind.config.ts",
                "tsconfig.json",
            ]
        ),
    ]
    digest = hashlib.sha256()
    for path in sorted(path for path in source_files if path.is_file()):
        digest.update(path.relative_to(_DASHBOARD_ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")

    match = re.search(r'data-source-hash="([0-9a-f]{64})"', _DASHBOARD_ASSET.read_text())
    assert match is not None
    assert match.group(1) == digest.hexdigest(), "dashboard.html is stale; run npm run build"


def test_dashboard_lists_and_selects_concurrent_pipelines(actor_context, tmp_path):
    secret_source_value = "source-value-must-not-leak"
    first_plan = compute_plan(Dataset.from_list([secret_source_value, "second"]).map(lambda value: value.upper()))
    second_plan = compute_plan(Dataset.from_list([1]).filter(lambda value: value > 0))
    coordinator = _make_test_coordinator(tmp_path)
    _start_pipeline(coordinator, first_plan, "exec-map", "document-cleaning")
    _start_pipeline(coordinator, second_plan, "exec-filter", "quality-filter", stage_name="stage0-Filter")

    try:
        with TestClient(coordinator.web_application) as client:
            pipelines = _api(client, "pipelines")
            assert [pipeline["execution_id"] for pipeline in pipelines["pipelines"]] == [
                "exec-filter",
                "exec-map",
            ]
            assert [pipeline["pipeline_name"] for pipeline in pipelines["pipelines"]] == [
                "quality-filter",
                "document-cleaning",
            ]

            response = _api(client, "plan", {"execution_id": "exec-map"})
            assert response["pipeline_name"] == "document-cleaning"
            assert response["execution_id"] == "exec-map"
            assert response["source_item_count"] == 2
            assert response["nodes"][0]["stage_type"] == "SOURCE"
            assert "Map" in response["nodes"][1]["operation_types"]
            assert secret_source_value not in str(response)

            other = _api(client, "plan", {"execution_id": "exec-filter"})
            assert other["pipeline_name"] == "quality-filter"
            assert other["execution_id"] == "exec-filter"
            assert other["source_item_count"] == 1

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
            status = _api(client, "status", {"execution_id": "exec-a"})
            assert status["execution_id"] == "exec-a"
            assert status["completed_shards"] == 0
            assert status["total_shards"] == 1
            assert status["expected_workers"] == 2
            assert status["coordinator_task_id"] == "user/coordinator/0"
            assert status["worker_states"] == [{"state": "active", "count": 1}]
            assert status["resources"]["cpu_cores"] == 1.25
            assert status["resources"]["memory_bytes"] == 256

            counters = _api(client, "counters", {"execution_id": "exec-a", "limit": 10})
            assert {counter["name"] for counter in counters["counters"]} == {
                ZEPHYR_WORKER_CPU_PCT_CURRENT_KEY,
                ZEPHYR_WORKER_MEM_CURRENT_KEY,
                "records-a",
            }
            assert "records-b" not in str(counters)

            other = _api(client, "counters", {"execution_id": "exec-b", "limit": 10})
            assert [counter["name"] for counter in other["counters"]] == ["records-b"]

            workers = _api(client, "workers", {"limit": 10})
            assert workers["total"] == 1
            assert workers["workers"][0]["task_id"] == "user/workers/0"
            assert workers["workers"][0]["cpu_percent"] == 125

            metrics = _api(client, "metrics", {"execution_id": "exec-a", "max_points": 10})
            assert metrics["warning"] == "Finelog is not available for this coordinator."
    finally:
        coordinator.shutdown()


def test_dashboard_rejects_metrics_for_unknown_execution(actor_context, tmp_path, monkeypatch):
    class RejectingStatsWriter:
        def query_pipeline_metrics(self, execution_id: str, max_points: int):
            raise AssertionError(f"unexpected metrics query for {execution_id=} and {max_points=}")

        def close(self) -> None:
            pass

    monkeypatch.setattr("zephyr.coordinator.StatsWriter.connect", lambda: RejectingStatsWriter())
    plan = compute_plan(Dataset.from_list([1]).map(lambda value: value + 1))
    coordinator = _make_test_coordinator(tmp_path)
    _start_pipeline(coordinator, plan, "exec-a", "pipeline-a")
    try:
        with TestClient(coordinator.web_application) as client:
            metrics = _api(client, "metrics", {"execution_id": "another-job", "max_points": 10})
            assert metrics == {"points": [], "warning": "The selected pipeline is not active."}
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
            workers = _api(client, "workers", {"limit": 10})
            assert workers["workers"][0]["assignments"] == [{"execution_id": "exec-a", "shard": 3}]
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
            response = _api(client, "plan", {"execution_id": "join-exec"})
            auxiliary_nodes = [node for node in response["nodes"] if node.get("auxiliary")]
            assert auxiliary_nodes
            assert all("/join/" in node["node_id"] for node in auxiliary_nodes)
            assert private_right_value not in str(response)
    finally:
        coordinator.shutdown()


def test_dashboard_reports_selected_pipeline_failure(actor_context, tmp_path):
    plan = compute_plan(Dataset.from_list([1]).map(lambda value: value + 1))
    coordinator = _make_test_coordinator(tmp_path)
    coordinator.register_worker("worker-0", MagicMock())
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            execution = executor.submit(
                coordinator.run_pipeline,
                plan,
                "failed-exec",
                "failed",
                _TEST_TASK_COST,
                _TEST_TASK_COST,
            )
            pulled_work = []

            def pull_first_task() -> bool:
                pull_status, work = coordinator.pull_task("worker-0", _TEST_TASK_COST)
                if pull_status is not PullStatus.RUN_TASK or work is None:
                    return False
                pulled_work.append(work)
                return True

            assert ExponentialBackoff(initial=0.01, maximum=0.1).wait_until(
                pull_first_task,
                Duration.from_seconds(5),
            )
            for failure in range(3):
                if failure:
                    pull_status, work = coordinator.pull_task("worker-0", _TEST_TASK_COST)
                    assert pull_status is PullStatus.RUN_TASK
                    assert work is not None
                else:
                    work = pulled_work[0]
                coordinator.report_error(
                    "worker-0",
                    work.execution_id,
                    work.task.shard_idx,
                    work.attempt,
                    "test stage failed",
                    work.stage_generation,
                )
            with pytest.raises(ZephyrWorkerError, match="test stage failed"):
                execution.result(timeout=5)

        with TestClient(coordinator.web_application) as client:
            status = _api(client, "status", {"execution_id": "failed-exec"})
            assert status["phase"] == "failed"
            assert "test stage failed" in status["fatal_error"]
            failed_nodes = [node for node in status["node_statuses"] if node["state"] == "failed"]
            assert [node["node_id"] for node in failed_nodes] == ["main/stage/0"]
    finally:
        coordinator.shutdown()
