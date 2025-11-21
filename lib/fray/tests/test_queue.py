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
import ray
from fray import FileQueue, RayQueue


@pytest.fixture(scope="module")
def ray_context():
    """Initialize Ray once for all Ray-based tests."""
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.fixture(params=["file", "ray"])
def queue(request, ray_context):
    """Parameterized fixture providing FileQueue and RayQueue implementations."""
    if request.param == "file":
        with tempfile.TemporaryDirectory() as tmpdir:
            yield FileQueue(path=tmpdir)
    else:  # ray
        # Use unique name per test to avoid state pollution
        import uuid

        unique_name = f"test-queue-{uuid.uuid4()}"
        yield RayQueue(name=unique_name)


def test_push_and_pop(queue):
    """Test basic push and pop operations."""
    assert queue.pop() is None

    queue.push("task1")
    lease = queue.pop()
    assert lease is not None
    assert lease.item == "task1"
    assert isinstance(lease.lease_id, str)
    assert isinstance(lease.timestamp, float)
    assert lease.timestamp <= time.time()


def test_pop_returns_fifo(queue):
    """Test that pop returns items in FIFO order."""
    queue.push("task1")
    queue.push("task2")
    queue.push("task3")

    lease1 = queue.pop()
    assert lease1 is not None
    assert lease1.item == "task1"

    lease2 = queue.pop()
    assert lease2 is not None
    assert lease2.item == "task2"

    lease3 = queue.pop()
    assert lease3 is not None
    assert lease3.item == "task3"


def test_done_completes_lease(queue):
    """Test that done() removes the item from the queue."""
    queue.push("task1")
    queue.push("task2")

    lease = queue.pop()
    assert lease is not None
    assert queue.size() == 2  # task2 pending + task1 leased
    assert queue.pending() == 1  # only task2 pending

    queue.done(lease)
    assert queue.size() == 1  # only task2 remains
    assert queue.pending() == 1


def test_done_with_invalid_lease(queue):
    """Test that done() raises error for invalid lease."""
    queue.push("task1")
    lease = queue.pop()
    assert lease is not None

    # Complete the lease
    queue.done(lease)

    # Trying to complete again should raise
    with pytest.raises((ValueError, Exception), match="Invalid lease"):
        queue.done(lease)


def test_release_requeues_item(queue):
    """Test that release() requeues the item."""
    queue.push("task1")
    queue.push("task2")

    lease = queue.pop()
    assert lease is not None
    assert lease.item == "task1"
    assert queue.pending() == 1  # only task2 pending

    queue.release(lease)
    assert queue.pending() == 2  # both items pending again

    # Released items go to front for immediate retry
    next_lease = queue.pop()
    assert next_lease is not None
    assert next_lease.item == "task1"  # task1 was released, goes to front


def test_release_with_invalid_lease(queue):
    """Test that release() raises error for invalid lease."""
    queue.push("task1")
    lease = queue.pop()
    assert lease is not None

    # Release the lease
    queue.release(lease)

    # Trying to release again should raise
    with pytest.raises((ValueError, Exception), match="Invalid lease"):
        queue.release(lease)


def test_size_counts_all_items(queue):
    """Test that size() includes both pending and leased items."""
    assert queue.size() == 0

    queue.push("task1")
    queue.push("task2")
    queue.push("task3")
    assert queue.size() == 3

    lease1 = queue.pop()
    assert queue.size() == 3  # Still 3 items (2 pending, 1 leased)

    lease2 = queue.pop()
    assert queue.size() == 3  # Still 3 items (1 pending, 2 leased)

    queue.done(lease1)
    assert queue.size() == 2  # Now 2 items (1 pending, 1 leased)

    queue.release(lease2)
    assert queue.size() == 2  # Still 2 items (both pending)


def test_pending_counts_unleased_items(queue):
    """Test that pending() only counts unleased items."""
    assert queue.pending() == 0

    queue.push("task1")
    queue.push("task2")
    queue.push("task3")
    assert queue.pending() == 3

    lease1 = queue.pop()
    assert queue.pending() == 2  # 2 items still pending

    lease2 = queue.pop()
    assert queue.pending() == 1  # 1 item still pending

    queue.done(lease1)
    assert queue.pending() == 1  # Still 1 pending

    queue.release(lease2)
    assert queue.pending() == 2  # Released item is pending again


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

    # Continue with remaining tasks
    assert queue.pending() == 3  # task2, task3, task4
    assert queue.size() == 3

    # Process next task
    lease4 = queue.pop()
    assert lease4 is not None
    assert queue.size() == 3


def test_lease_expiration_recovery(queue):
    """Test that expired leases are automatically recovered."""
    queue.push("task1")
    lease = queue.pop(lease_timeout=0.2)
    assert lease is not None
    assert lease.item == "task1"
    assert queue.pending() == 0

    time.sleep(0.3)  # Wait for expiration

    # Should be available again
    lease2 = queue.pop()
    assert lease2 is not None
    assert lease2.item == "task1"


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

    # After timeout, task1 recovers
    time.sleep(0.3)
    lease3 = queue.pop()
    assert lease3 is not None
    assert lease3.item == "task1"
