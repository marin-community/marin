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
