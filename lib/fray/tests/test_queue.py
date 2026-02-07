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

"""Tests for queue abstraction with lease semantics."""

import tempfile
import time

import pytest
from fray.v1.queue.file import FileQueue
from fray.v1.queue.http import HttpQueueServer


@pytest.fixture(params=["file", "http"])
def queue(request):
    """Parameterized fixture providing FileQueue and HttpQueue implementations."""
    if request.param == "file":
        with tempfile.TemporaryDirectory() as tmpdir:
            yield FileQueue(path=tmpdir)
    elif request.param == "http":
        with HttpQueueServer(host="127.0.0.1", port=9999) as server:
            yield server.new_queue("test-queue")
    else:
        raise ValueError(f"Unknown queue type: {request.param}")


def test_complete_workflow(queue):
    """Test a complete workflow with multiple workers."""
    # Add tasks
    for i in range(5):
        queue.push(f"task{i}")

    # Worker 1 processes tasks successfully
    lease1 = queue.pop()
    assert lease1 is not None
    assert lease1.item == "task0"
    queue.done(lease1)

    # Worker 2 gets a task but fails
    lease2 = queue.pop()
    assert lease2 is not None
    assert lease2.item == "task1"
    queue.release(lease2)  # Requeue the failed task

    # Worker 3 gets the requeued task (goes to front)
    lease3 = queue.pop()
    assert lease3 is not None
    assert lease3.item == "task1"  # Requeued task at front
    queue.done(lease3)


def test_worker_crash_simulation(queue):
    """Test abandoned leases (worker crashed without done/release)."""
    queue.push("task1")
    queue.push("task2")

    # Worker 1 pops task1 but crashes (never calls done/release)
    lease1 = queue.pop(lease_timeout=0.2)
    assert lease1 is not None
    assert lease1.item == "task1"

    # Worker 2 can immediately get task2
    lease2 = queue.pop()
    assert lease2 is not None
    assert lease2.item == "task2"
    queue.done(lease2)

    # After timeout, task1 is available again
    time.sleep(0.3)
    lease3 = queue.pop()
    assert lease3 is not None
    assert lease3.item == "task1"
