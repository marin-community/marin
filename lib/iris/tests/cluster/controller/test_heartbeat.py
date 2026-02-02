# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for worker heartbeat timeout handling with typed time primitives."""

import time

import pytest

from iris.cluster.controller.events import WorkerRegisteredEvent
from iris.cluster.controller.state import ControllerState
from iris.cluster.types import WorkerId
from iris.rpc import cluster_pb2
from iris.time_utils import Duration, Timestamp


@pytest.fixture
def state():
    return ControllerState()


@pytest.fixture
def worker_metadata():
    return cluster_pb2.WorkerMetadata(
        hostname="test-host",
        ip_address="192.168.1.1",
        cpu_count=8,
        memory_bytes=16 * 1024**3,
        disk_bytes=100 * 1024**3,
    )


def test_heartbeat_timeout_uses_duration(state, worker_metadata):
    """Test that check_worker_timeouts accepts Duration parameter."""
    # Register worker
    event = WorkerRegisteredEvent(
        worker_id=WorkerId("worker1"),
        address="host:8080",
        metadata=worker_metadata,
        timestamp=Timestamp.now(),
    )
    state.handle_event(event)

    # Check timeout with Duration
    timeout = Duration.from_seconds(60.0)
    tasks = state.check_worker_timeouts(timeout)

    # Worker should not timeout yet (just registered)
    assert len(tasks) == 0

    # Verify worker is healthy
    workers = list(state.list_all_workers())
    assert len(workers) == 1
    assert workers[0].healthy


def test_worker_heartbeat_deadline_helper(state, worker_metadata):
    """Test that ControllerWorker.heartbeat_deadline computes correct deadline."""
    event = WorkerRegisteredEvent(
        worker_id=WorkerId("worker1"),
        address="host:8080",
        metadata=worker_metadata,
        timestamp=Timestamp.now(),
    )
    state.handle_event(event)

    workers = list(state.list_all_workers())
    worker = workers[0]

    timeout = Duration.from_seconds(30.0)

    # Compute deadline
    deadline = worker.heartbeat_deadline(timeout)

    # Deadline should be approximately now + 30 seconds
    expected_deadline = worker.last_heartbeat.add(timeout)
    assert abs(deadline.epoch_ms() - expected_deadline.epoch_ms()) < 100  # Within 100ms


def test_worker_heartbeat_expired_check(state, worker_metadata):
    """Test that ControllerWorker.is_heartbeat_expired detects expired heartbeats."""
    event = WorkerRegisteredEvent(
        worker_id=WorkerId("worker1"),
        address="host:8080",
        metadata=worker_metadata,
        timestamp=Timestamp.now(),
    )
    state.handle_event(event)

    workers = list(state.list_all_workers())
    worker = workers[0]

    # Short timeout should not expire immediately
    short_timeout = Duration.from_seconds(10.0)
    assert not worker.is_heartbeat_expired(short_timeout)

    # Very short timeout might expire after a brief sleep
    very_short_timeout = Duration.from_ms(1)
    time.sleep(0.01)  # 10ms
    assert worker.is_heartbeat_expired(very_short_timeout)
