# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for user-defined counter support: task-side API, heartbeat plumbing, and job aggregation."""


from iris.cluster.controller.transitions import (
    Assignment,
    ControllerTransitions,
    HeartbeatApplyRequest,
    TaskUpdate,
)
from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.types import JobName, WorkerId
from iris.rpc import cluster_pb2
from iris.time_utils import Timestamp

from .conftest import make_controller_state, MockController, BundleStore


def _register_worker(state: ControllerTransitions, worker_id: str) -> None:
    state.register_or_refresh_worker(
        worker_id=WorkerId(worker_id),
        address=f"{worker_id}:8080",
        metadata=cluster_pb2.WorkerMetadata(
            hostname=worker_id,
            ip_address="127.0.0.1",
            cpu_count=4,
            memory_bytes=8 * 1024**3,
            disk_bytes=50 * 1024**3,
        ),
        ts=Timestamp.now(),
    )


def test_counters_increment_writes_file(tmp_path, monkeypatch):
    """increment() accumulates in-memory; flush() writes to the file at IRIS_COUNTER_FILE."""
    counter_file = tmp_path / "iris_counters.json"
    monkeypatch.setenv("IRIS_COUNTER_FILE", str(counter_file))

    from iris import counters

    # Reset module-level state so previous tests don't leak.
    with counters._lock:
        counters._counters.clear()
        counters._dirty = False
        counters._flush_thread = None

    counters.increment("docs", 10)
    counters.increment("docs", 5)
    counters.increment("errors", 1)

    # File should not exist yet (increments are in-memory only).
    assert not counter_file.exists()

    # Explicit flush writes to disk.
    counters.flush()

    import json

    data = json.loads(counter_file.read_text())
    assert data == {"docs": 15, "errors": 1}


def test_counters_noop_outside_task(monkeypatch):
    """increment() is a no-op when IRIS_COUNTER_FILE is not set."""
    monkeypatch.delenv("IRIS_COUNTER_FILE", raising=False)

    from iris import counters

    # Reset module-level state so previous tests don't leak.
    with counters._lock:
        counters._counters.clear()
        counters._dirty = False
        counters._flush_thread = None

    counters.increment("anything", 999)  # should not raise


def test_counters_flow_through_heartbeat_and_job_status(tmp_path):
    """Counters sent in a heartbeat update are stored and aggregated in get_job_status."""
    with make_controller_state() as state:
        log_store = state._log_store
        mock_controller = MockController()
        service = ControllerServiceImpl(
            state,
            state._db,
            controller=mock_controller,
            bundle_store=BundleStore(storage_dir=str(tmp_path / "bundles")),
            log_store=log_store,
        )

        _register_worker(state, "w1")
        _register_worker(state, "w2")

        job_id = JobName.from_wire("/user/counter-test")
        state.submit_job(
            job_id,
            cluster_pb2.Controller.LaunchJobRequest(name="/user/counter-test", replicas=2),
            Timestamp.now(),
        )
        task0 = job_id.task(0)
        task1 = job_id.task(1)

        state.queue_assignments(
            [
                Assignment(task_id=task0, worker_id=WorkerId("w1")),
                Assignment(task_id=task1, worker_id=WorkerId("w2")),
            ]
        )

        # Send heartbeat updates with counters for both tasks
        state.apply_task_updates(
            HeartbeatApplyRequest(
                worker_id=WorkerId("w1"),
                worker_resource_snapshot=None,
                updates=[
                    TaskUpdate(
                        task_id=task0,
                        attempt_id=0,
                        new_state=cluster_pb2.TASK_STATE_RUNNING,
                        counters={"docs": 100, "bytes": 1024},
                    )
                ],
            )
        )
        state.apply_task_updates(
            HeartbeatApplyRequest(
                worker_id=WorkerId("w2"),
                worker_resource_snapshot=None,
                updates=[
                    TaskUpdate(
                        task_id=task1,
                        attempt_id=0,
                        new_state=cluster_pb2.TASK_STATE_RUNNING,
                        counters={"docs": 200, "errors": 5},
                    )
                ],
            )
        )

        response = service.get_job_status(cluster_pb2.Controller.GetJobStatusRequest(job_id=job_id.to_wire()), None)

        assert response.job.counters["docs"] == 300
        assert response.job.counters["bytes"] == 1024
        assert response.job.counters["errors"] == 5

        # Per-task counters are also surfaced
        task_counters = {t.task_id: dict(t.counters) for t in response.job.tasks}
        assert task_counters[task0.to_wire()] == {"docs": 100, "bytes": 1024}
        assert task_counters[task1.to_wire()] == {"docs": 200, "errors": 5}


def test_counters_cleared_on_task_retry(tmp_path):
    """When a task fails and is retried, stale counters from the previous attempt are cleared."""
    with make_controller_state() as state:
        log_store = state._log_store
        mock_controller = MockController()
        service = ControllerServiceImpl(
            state,
            state._db,
            controller=mock_controller,
            bundle_store=BundleStore(storage_dir=str(tmp_path / "bundles")),
            log_store=log_store,
        )

        _register_worker(state, "w1")

        job_id = JobName.from_wire("/user/retry-test")
        state.submit_job(
            job_id,
            cluster_pb2.Controller.LaunchJobRequest(name="/user/retry-test", replicas=1, max_retries_failure=1),
            Timestamp.now(),
        )
        task0 = job_id.task(0)

        state.queue_assignments([Assignment(task_id=task0, worker_id=WorkerId("w1"))])

        # Attempt 0: report counters, then fail
        state.apply_task_updates(
            HeartbeatApplyRequest(
                worker_id=WorkerId("w1"),
                worker_resource_snapshot=None,
                updates=[
                    TaskUpdate(
                        task_id=task0,
                        attempt_id=0,
                        new_state=cluster_pb2.TASK_STATE_RUNNING,
                        counters={"docs": 500},
                    )
                ],
            )
        )
        state.apply_task_updates(
            HeartbeatApplyRequest(
                worker_id=WorkerId("w1"),
                worker_resource_snapshot=None,
                updates=[
                    TaskUpdate(
                        task_id=task0,
                        attempt_id=0,
                        new_state=cluster_pb2.TASK_STATE_FAILED,
                        error="boom",
                        exit_code=1,
                    )
                ],
            )
        )

        # Task should be PENDING (retry). Counters should be cleared.
        response = service.get_job_status(cluster_pb2.Controller.GetJobStatusRequest(job_id=job_id.to_wire()), None)
        assert response.job.counters.get("docs", 0) == 0, "Stale counters should be cleared on retry"

        # Attempt 1: assign and report new counters
        state.queue_assignments([Assignment(task_id=task0, worker_id=WorkerId("w1"))])
        state.apply_task_updates(
            HeartbeatApplyRequest(
                worker_id=WorkerId("w1"),
                worker_resource_snapshot=None,
                updates=[
                    TaskUpdate(
                        task_id=task0,
                        attempt_id=1,
                        new_state=cluster_pb2.TASK_STATE_RUNNING,
                        counters={"docs": 200},
                    )
                ],
            )
        )

        response = service.get_job_status(cluster_pb2.Controller.GetJobStatusRequest(job_id=job_id.to_wire()), None)
        assert response.job.counters["docs"] == 200, "Only attempt-1 counters should be counted"
