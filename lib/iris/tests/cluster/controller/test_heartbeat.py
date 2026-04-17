# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for worker heartbeat timeout handling, health checks, and unified heartbeat path."""

import logging
import time

import iris.cluster.controller.worker_provider as worker_provider_module
import pytest
from iris.cluster.controller.controller import Controller, ControllerConfig, _SyncFailureAccumulator
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.schema import (
    TASK_DETAIL_PROJECTION,
    WORKER_DETAIL_PROJECTION,
)
from tests.cluster.controller.conftest import FakeProvider
from iris.cluster.controller.transitions import (
    Assignment,
    ControllerTransitions,
    DispatchBatch,
    HEARTBEAT_STALENESS_THRESHOLD,
    HeartbeatAction,
    HeartbeatApplyRequest,
    RunningTaskEntry,
    TaskUpdate,
)
from iris.cluster.controller.worker_provider import RpcWorkerStubFactory, WorkerProvider
from iris.cluster.types import JobName, WorkerId
from iris.rpc import controller_pb2
from iris.rpc import job_pb2
from rigging.timing import Duration, Timestamp


@pytest.fixture
def state(tmp_path):
    db = ControllerDB(db_dir=tmp_path)
    s = ControllerTransitions(db=db)
    yield s
    db.close()


@pytest.fixture
def worker_metadata():
    return job_pb2.WorkerMetadata(
        hostname="test-host",
        ip_address="192.168.1.1",
        cpu_count=8,
        memory_bytes=16 * 1024**3,
        disk_bytes=100 * 1024**3,
    )


def _register_worker(state, worker_id, worker_metadata, address="host:8080"):
    state.register_or_refresh_worker(
        worker_id=WorkerId(worker_id),
        address=address,
        metadata=worker_metadata,
        ts=Timestamp.now(),
    )


def _make_snapshot(worker_id, address="host:8080", running_tasks=None):
    return DispatchBatch(
        worker_id=WorkerId(worker_id),
        worker_address=address,
        running_tasks=running_tasks or [],
        tasks_to_run=[],
        tasks_to_kill=[],
    )


def test_worker_heartbeat_expired_check(state, worker_metadata):
    """Test heartbeat expiration checks against worker row last_heartbeat."""
    state.register_or_refresh_worker(
        worker_id=WorkerId("worker1"),
        address="host:8080",
        metadata=worker_metadata,
        ts=Timestamp.now(),
    )

    with state._db.snapshot() as q:
        workers = WORKER_DETAIL_PROJECTION.decode(q.fetchall("SELECT * FROM workers"))
    worker = workers[0]

    # Short timeout should not expire immediately
    short_timeout = Duration.from_seconds(10.0)
    assert worker.last_heartbeat.age_ms() < short_timeout.to_ms()

    # Very short timeout might expire after a brief sleep
    very_short_timeout = Duration.from_ms(1)
    time.sleep(0.01)  # 10ms
    assert worker.last_heartbeat.age_ms() > very_short_timeout.to_ms()


def test_complete_heartbeat_success(state, worker_metadata):
    """Successful heartbeat with healthy worker returns OK."""
    _register_worker(state, "worker1", worker_metadata)
    snapshot = _make_snapshot("worker1")

    response = job_pb2.HeartbeatResponse(worker_healthy=True)
    result = state.complete_heartbeat(snapshot, response)

    assert result.action == HeartbeatAction.OK

    with state._db.snapshot() as q:
        worker = WORKER_DETAIL_PROJECTION.decode_one(
            q.fetchall("SELECT * FROM workers WHERE worker_id = ? LIMIT 1", ("worker1",)),
        )
    assert worker is not None
    assert worker.healthy


def test_fail_heartbeat_records_transient_failure(state, worker_metadata):
    """RPC failure without force_remove returns TRANSIENT_FAILURE and bumps counter.

    Repeated failures no longer trip inline removal — the reaper thread owns
    termination, driven by the in-memory health tracker.
    """
    _register_worker(state, "worker1", worker_metadata)
    snapshot = _make_snapshot("worker1")

    for expected_failures in range(1, 25):
        action = state.fail_heartbeat(snapshot, "connection refused")
        assert action == HeartbeatAction.TRANSIENT_FAILURE
        with state._db.snapshot() as q:
            worker = WORKER_DETAIL_PROJECTION.decode_one(
                q.fetchall("SELECT * FROM workers WHERE worker_id = ? LIMIT 1", ("worker1",)),
            )
        assert worker is not None
        assert worker.healthy
        assert worker.consecutive_failures == expected_failures


def test_force_remove_prunes_worker(state, worker_metadata):
    """force_remove=True is the explicit kill path the reaper uses."""
    _register_worker(state, "worker1", worker_metadata)
    snapshot = _make_snapshot("worker1")

    state.record_heartbeat_failure(
        WorkerId("worker1"),
        "health score exceeded threshold",
        snapshot,
        force_remove=True,
    )

    with state._db.snapshot() as q:
        assert q.fetchone("SELECT 1 FROM workers WHERE worker_id = ?", ("worker1",)) is None


def test_complete_heartbeat_unhealthy_worker_increments_failures(state, worker_metadata):
    """Worker reporting unhealthy increments failure count (not immediate removal)."""
    _register_worker(state, "worker1", worker_metadata)
    snapshot = _make_snapshot("worker1")

    response = job_pb2.HeartbeatResponse(
        worker_healthy=False,
        health_error="disk free space 2.1% below threshold 5%",
    )
    result = state.complete_heartbeat(snapshot, response)

    assert result.action == HeartbeatAction.TRANSIENT_FAILURE
    with state._db.snapshot() as q:
        assert q.fetchone("SELECT 1 FROM workers WHERE worker_id = ?", ("worker1",)) is not None


def test_fail_workers_batch_cascades_to_running_tasks(tmp_path):
    """When the reaper force-fails a worker via fail_workers_batch, its running
    tasks cascade to TASK_STATE_WORKER_FAILED. This is the primitive the
    reaper uses when the health tracker crosses threshold."""
    db = ControllerDB(db_dir=tmp_path)
    state = ControllerTransitions(db=db)
    worker_metadata = job_pb2.WorkerMetadata(
        hostname="test-host",
        ip_address="192.168.1.1",
        cpu_count=8,
        memory_bytes=16 * 1024**3,
        disk_bytes=100 * 1024**3,
    )
    _register_worker(state, "worker1", worker_metadata)

    job_id = JobName.from_wire("/user/test-job")
    state.submit_job(
        job_id,
        controller_pb2.Controller.LaunchJobRequest(
            name="/user/test-job",
            replicas=1,
        ),
        Timestamp.now(),
    )
    task_id = job_id.task(0)
    state.queue_assignments([Assignment(task_id=task_id, worker_id=WorkerId("worker1"))])
    state.apply_task_updates(
        HeartbeatApplyRequest(
            worker_id=WorkerId("worker1"),
            worker_resource_snapshot=None,
            updates=[
                TaskUpdate(
                    task_id=task_id,
                    attempt_id=0,
                    new_state=job_pb2.TASK_STATE_RUNNING,
                )
            ],
        )
    )

    result = state.fail_workers_batch(["worker1"], reason="health score exceeded threshold")
    assert [wid for wid, _ in result.removed_workers] == [WorkerId("worker1")]

    with state._db.snapshot() as q:
        task = TASK_DETAIL_PROJECTION.decode_one(
            q.fetchall("SELECT * FROM tasks WHERE task_id = ? LIMIT 1", (task_id.to_wire(),)),
        )
    assert task is not None
    assert task.state == job_pb2.TASK_STATE_WORKER_FAILED


def test_reap_stale_workers_removes_old_heartbeat(tmp_path, worker_metadata, caplog):
    """Workers restored from checkpoint with heartbeat older than the staleness
    threshold are failed immediately by the heartbeat loop's reap pass."""
    db = ControllerDB(db_dir=tmp_path)
    config = ControllerConfig(remote_state_dir="file:///tmp/iris-test-state", local_state_dir=tmp_path)
    controller = Controller(config=config, provider=FakeProvider(), db=db)
    state = controller.state

    # Register a worker with a timestamp well beyond the staleness threshold.
    stale_ts = Timestamp.from_ms(Timestamp.now().epoch_ms() - HEARTBEAT_STALENESS_THRESHOLD.to_ms() - 60_000)
    state.register_or_refresh_worker(
        worker_id=WorkerId("stale-worker"),
        address="10.0.0.1:10001",
        metadata=worker_metadata,
        ts=stale_ts,
    )
    # Register a fresh worker that should survive.
    state.register_or_refresh_worker(
        worker_id=WorkerId("fresh-worker"),
        address="10.0.0.2:10001",
        metadata=worker_metadata,
        ts=Timestamp.now(),
    )

    with db.snapshot() as q:
        assert q.fetchone("SELECT 1 FROM workers WHERE worker_id = ?", ("stale-worker",)) is not None
        assert q.fetchone("SELECT 1 FROM workers WHERE worker_id = ?", ("fresh-worker",)) is not None

    with caplog.at_level(logging.WARNING):
        controller._reap_stale_workers()

    with db.snapshot() as q:
        assert q.fetchone("SELECT 1 FROM workers WHERE worker_id = ?", ("stale-worker",)) is None
        assert q.fetchone("SELECT 1 FROM workers WHERE worker_id = ?", ("fresh-worker",)) is not None
    assert "stale-worker" in caplog.text
    assert "age_s" in caplog.text
    assert "10.0.0.1:10001" in caplog.text

    controller.stop()


def test_reap_stale_workers_no_op_when_all_fresh(tmp_path, worker_metadata):
    """When all workers have recent heartbeats, no workers are reaped."""
    db = ControllerDB(db_dir=tmp_path)
    config = ControllerConfig(remote_state_dir="file:///tmp/iris-test-state", local_state_dir=tmp_path)
    controller = Controller(config=config, provider=FakeProvider(), db=db)

    controller.state.register_or_refresh_worker(
        worker_id=WorkerId("worker1"),
        address="10.0.0.1:10001",
        metadata=worker_metadata,
        ts=Timestamp.now(),
    )

    controller._reap_stale_workers()

    with db.snapshot() as q:
        assert q.fetchone("SELECT 1 FROM workers WHERE worker_id = ?", ("worker1",)) is not None

    controller.stop()


class _FakeStub:
    """Stub that returns a canned HeartbeatResponse."""

    def __init__(self, response: job_pb2.HeartbeatResponse):
        self._response = response

    async def heartbeat(
        self,
        request: job_pb2.HeartbeatRequest,
        *,
        timeout_ms: int | None = None,
    ) -> job_pb2.HeartbeatResponse:
        return self._response


class _RaisingStub:
    def __init__(self, exc: Exception):
        self._exc = exc

    async def heartbeat(
        self,
        request: job_pb2.HeartbeatRequest,
        *,
        timeout_ms: int | None = None,
    ) -> job_pb2.HeartbeatResponse:
        raise self._exc


class _FakeStubFactory:
    def __init__(self, stub: _FakeStub):
        self._stub = stub

    def get_stub(self, address: str) -> _FakeStub:
        return self._stub

    def evict(self, address: str) -> None:
        pass

    def close(self) -> None:
        pass


def test_handle_failed_heartbeats_logs_diagnostics(tmp_path, worker_metadata, caplog):
    db = ControllerDB(db_dir=tmp_path)
    config = ControllerConfig(remote_state_dir="file:///tmp/iris-test-state", local_state_dir=tmp_path)
    controller = Controller(config=config, provider=FakeProvider(), db=db)
    state = controller.state
    _register_worker(state, "worker1", worker_metadata, address="10.0.0.1:10001")

    batch = DispatchBatch(
        worker_id=WorkerId("worker1"),
        worker_address="10.0.0.1:10001",
        running_tasks=[RunningTaskEntry(JobName.from_wire("/user/test-job/0"), 0)],
        tasks_to_run=[],
        tasks_to_kill=[],
    )
    acc = _SyncFailureAccumulator()
    with caplog.at_level(logging.WARNING):
        primary_failed_workers = controller._handle_failed_heartbeats(
            [(batch, "deadline exceeded after 12000ms")],
            acc,
        )

    assert primary_failed_workers == []
    assert acc.fail_count == 1
    assert "worker=worker1" in caplog.text
    assert "address=10.0.0.1:10001" in caplog.text
    assert "action=transient_failure" in caplog.text
    assert "failures=1/10" in caplog.text
    assert "last_success_age_s=" in caplog.text
    assert "deadline exceeded after 12000ms" in caplog.text

    controller.stop()


def test_rpc_worker_stub_factory_default_timeout(monkeypatch):
    captured: dict[str, object] = {}

    class _RecordingClient:
        def __init__(self, address: str, timeout_ms: int):
            captured["address"] = address
            captured["timeout_ms"] = timeout_ms

        def close(self) -> None:
            pass

    monkeypatch.setattr(worker_provider_module, "WorkerServiceClient", _RecordingClient)

    factory = RpcWorkerStubFactory()
    factory.get_stub("host:8080")

    assert captured["address"] == "http://host:8080"
    assert captured["timeout_ms"] == 10_000

    factory.close()


def test_heartbeat_failure_error_includes_rpc_context():
    provider = WorkerProvider(
        stub_factory=_FakeStubFactory(_RaisingStub(RuntimeError("deadline exceeded"))),
    )
    batch = DispatchBatch(
        worker_id=WorkerId("w1"),
        worker_address="host:8080",
        running_tasks=[RunningTaskEntry(JobName.from_wire("/user/test-job/0"), 0)],
        tasks_to_run=[],
        tasks_to_kill=[],
    )

    results = provider.sync([batch])

    assert len(results) == 1
    _, apply_req, error = results[0]
    assert apply_req is None
    assert error is not None
    assert "heartbeat RPC failed:" in error
    assert "worker=w1" in error
    assert "address=host:8080" in error
    assert "expected=1" in error

    provider.close()
