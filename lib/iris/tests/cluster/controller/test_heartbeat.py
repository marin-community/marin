# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for worker heartbeat timeout handling and health checks."""

import logging
import time

import pytest
from iris.cluster.controller.controller import Controller, ControllerConfig
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.schema import (
    WORKER_DETAIL_PROJECTION,
)
from tests.cluster.controller.conftest import FakeProvider
from iris.cluster.controller.transitions import (
    ControllerTransitions,
    HEARTBEAT_STALENESS_THRESHOLD,
)
from iris.cluster.types import WorkerId
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

    db.close()
