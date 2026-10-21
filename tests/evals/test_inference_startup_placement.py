# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from types import SimpleNamespace
from typing import cast

import marin.inference.iris as iris_module
import pytest
import rigging.timing as timing_module
from fray.client import JobHandle
from fray.types import JobStatus
from iris.client.workload import TaskStatus
from iris.cluster.types import JobName
from iris.resources.state import TaskState


def _task(state: TaskState, message: str = "") -> TaskStatus:
    return TaskStatus(
        task_id=JobName.from_string("/test/serve/0"),
        state=state,
        worker_id="worker",
        worker_address="",
        exit_code=0,
        error_message="",
        submitted_at=None,
        started_at=None,
        finished_at=None,
        ports={},
        resource_usage=None,
        build_metrics=None,
        current_attempt_number=0,
        attempts=(),
        pending_reason="",
        can_be_scheduled=True,
        container_id="",
        backend_id="",
        execution_cluster_id="",
        status_message=message,
    )


@pytest.fixture
def clock(monkeypatch) -> SimpleNamespace:
    clock = SimpleNamespace(value=0.0)

    def sleep(seconds: float) -> None:
        clock.value += seconds

    simulated_time = SimpleNamespace(monotonic=lambda: clock.value, sleep=sleep)
    monkeypatch.setattr(iris_module, "time", simulated_time)
    monkeypatch.setattr(timing_module, "time", simulated_time)
    monkeypatch.setattr(iris_module, "_ENDPOINT_READY_POLL_SECONDS", 1.0)
    return clock


def _job(
    monkeypatch, clock: SimpleNamespace, task_at: Callable[[float], TaskStatus], ready_at: float | None
) -> JobHandle:
    endpoint = SimpleNamespace(address="http://test-server", metadata={"kind": "test"})
    client = SimpleNamespace(
        list_endpoint_instances=lambda _: [endpoint] if ready_at is not None and clock.value >= ready_at else [],
        list_tasks=lambda _: [task_at(clock.value)],
    )
    monkeypatch.setattr(iris_module, "iris_ctx", lambda: SimpleNamespace(client=client))
    return cast(JobHandle, SimpleNamespace(job_id="/test/serve", status=lambda: JobStatus.RUNNING))


def test_kueue_admission_does_not_consume_startup_budget(monkeypatch, clock) -> None:
    job = _job(
        monkeypatch,
        clock,
        lambda now: _task(TaskState.BUILDING, "SchedulingGated: waiting for Kueue admission" if now < 8 else ""),
        ready_at=12,
    )
    assert iris_module._wait_for_endpoint(job, "test", timeout_seconds=5) == (
        "http://test-server",
        {"kind": "test"},
    )
    assert clock.value == 12


def test_admitted_container_build_keeps_original_startup_limit(monkeypatch, clock) -> None:
    job = _job(monkeypatch, clock, lambda _: _task(TaskState.BUILDING, "ContainerCreating: pulling image"), None)
    with pytest.raises(TimeoutError, match="Timed out waiting for inference endpoint"):
        iris_module._wait_for_endpoint(job, "test", timeout_seconds=5)
    assert clock.value == 5


def test_long_kueue_wait_preserves_queue_position_until_admission(monkeypatch, clock) -> None:
    monkeypatch.setattr(iris_module, "_ENDPOINT_READY_POLL_SECONDS", 1800.0)
    job = _job(
        monkeypatch,
        clock,
        lambda now: _task(TaskState.BUILDING, "SchedulingGated: waiting for quota" if now < 18000 else ""),
        ready_at=19800,
    )
    assert iris_module._wait_for_endpoint(job, "test", timeout_seconds=2400)[0] == "http://test-server"
    assert clock.value == 19800


def test_kueue_wait_keeps_original_placement_limit(monkeypatch, clock) -> None:
    monkeypatch.setattr(iris_module, "_ENDPOINT_PLACEMENT_TIMEOUT_SECONDS", 6)
    job = _job(monkeypatch, clock, lambda _: _task(TaskState.BUILDING, "SchedulingGated: waiting for quota"), None)
    with pytest.raises(TimeoutError, match="queued for 6s"):
        iris_module._wait_for_endpoint(job, "test", timeout_seconds=5)
    assert clock.value == 6


def test_preemption_into_kueue_resets_startup_clock(monkeypatch, clock) -> None:
    job = _job(
        monkeypatch,
        clock,
        lambda now: (
            _task(TaskState.BUILDING, "SchedulingGated: requeued after preemption")
            if 3 <= now < 10
            else _task(TaskState.RUNNING)
        ),
        ready_at=12,
    )
    assert iris_module._wait_for_endpoint(job, "test", timeout_seconds=5)[0] == "http://test-server"
    assert clock.value == 12


def test_running_task_keeps_startup_limit_with_stale_admission_message(monkeypatch, clock) -> None:
    job = _job(monkeypatch, clock, lambda _: _task(TaskState.RUNNING, "SchedulingGated: stale status"), None)
    with pytest.raises(TimeoutError, match="Timed out waiting for inference endpoint"):
        iris_module._wait_for_endpoint(job, "test", timeout_seconds=5)
    assert clock.value == 5
