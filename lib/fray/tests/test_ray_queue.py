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

"""Tests for Ray-based distributed queue implementation."""

import time

import pytest
import ray

from fray.cluster.ray.queue import RayQueue


@pytest.fixture(scope="module")
def ray_context():
    """Initialize Ray for testing."""
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.fixture
def queue(ray_context) -> RayQueue:
    """Create a Ray queue for testing."""
    return RayQueue("test-queue")


def test_push_and_peek(queue: RayQueue):
    """Test pushing items and peeking at the next item."""
    assert queue.peek() is None

    queue.push("task1")
    assert queue.peek() == "task1"

    queue.push("task2")
    assert queue.peek() == "task1"


def test_push_and_pop(queue: RayQueue):
    """Test pushing and popping items."""
    assert queue.pop() is None

    queue.push("task1")
    lease = queue.pop()
    assert lease is not None
    assert lease.item == "task1"
    assert isinstance(lease.lease_id, str)
    assert isinstance(lease.timestamp, float)
    assert lease.timestamp <= time.time()


def test_pop_returns_fifo(queue: RayQueue):
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


def test_done_completes_lease(queue: RayQueue):
    """Test that done() removes the item from the queue."""
    queue.push("task1")
    queue.push("task2")

    lease = queue.pop()
    assert lease is not None
    assert queue.size() == 2
    assert queue.pending() == 1

    queue.done(lease)
    assert queue.size() == 1
    assert queue.pending() == 1


def test_done_with_invalid_lease(queue: RayQueue):
    """Test that done() raises error for invalid lease."""
    queue.push("task1")
    lease = queue.pop()
    assert lease is not None

    queue.done(lease)

    with pytest.raises(Exception, match="Invalid lease"):
        queue.done(lease)


def test_release_requeues_item(queue: RayQueue):
    """Test that release() requeues the item."""
    queue.push("task1")
    queue.push("task2")

    lease = queue.pop()
    assert lease is not None
    assert lease.item == "task1"
    assert queue.pending() == 1

    queue.release(lease)
    assert queue.pending() == 2

    next_lease = queue.pop()
    assert next_lease is not None
    assert next_lease.item == "task2"


def test_release_with_invalid_lease(queue: RayQueue):
    """Test that release() raises error for invalid lease."""
    queue.push("task1")
    lease = queue.pop()
    assert lease is not None

    queue.release(lease)

    with pytest.raises(Exception, match="Invalid lease"):
        queue.release(lease)


def test_size_counts_all_items(queue: RayQueue):
    """Test that size() includes both pending and leased items."""
    assert queue.size() == 0

    queue.push("task1")
    queue.push("task2")
    queue.push("task3")
    assert queue.size() == 3

    lease1 = queue.pop()
    assert queue.size() == 3

    lease2 = queue.pop()
    assert queue.size() == 3

    queue.done(lease1)
    assert queue.size() == 2

    queue.release(lease2)
    assert queue.size() == 2


def test_pending_counts_unleased_items(queue: RayQueue):
    """Test that pending() only counts unleased items."""
    assert queue.pending() == 0

    queue.push("task1")
    queue.push("task2")
    queue.push("task3")
    assert queue.pending() == 3

    lease1 = queue.pop()
    assert queue.pending() == 2

    lease2 = queue.pop()
    assert queue.pending() == 1

    queue.done(lease1)
    assert queue.pending() == 1

    queue.release(lease2)
    assert queue.pending() == 2


def test_complete_workflow(queue: RayQueue):
    """Test a complete workflow with multiple workers."""
    for i in range(5):
        queue.push(f"task{i}")

    lease1 = queue.pop()
    assert lease1 is not None
    assert lease1.item == "task0"
    queue.done(lease1)

    lease2 = queue.pop()
    assert lease2 is not None
    assert lease2.item == "task1"
    queue.release(lease2)

    lease3 = queue.pop()
    assert lease3 is not None
    assert lease3.item == "task2"
    queue.done(lease3)

    assert queue.pending() == 3
    assert queue.size() == 3

    lease4 = queue.pop()
    assert lease4 is not None
    assert queue.size() == 3


def test_empty_queue_operations(queue: RayQueue):
    """Test operations on an empty queue."""
    assert queue.peek() is None
    assert queue.pop() is None
    assert queue.size() == 0
    assert queue.pending() == 0


def test_queue_with_different_types(ray_context):
    """Test that Queue works with different types."""
    int_queue = RayQueue("int-queue")
    int_queue.push(42)
    lease = int_queue.pop()
    assert lease is not None
    assert lease.item == 42
    assert isinstance(lease.item, int)

    str_queue = RayQueue("str-queue")
    str_queue.push("hello")
    lease2 = str_queue.pop()
    assert lease2 is not None
    assert lease2.item == "hello"
    assert isinstance(lease2.item, str)


def test_lease_timestamp_ordering(queue: RayQueue):
    """Test that lease timestamps are correctly ordered."""
    queue.push("task1")
    time.sleep(0.01)
    queue.push("task2")

    lease1 = queue.pop()
    time.sleep(0.01)
    lease2 = queue.pop()

    assert lease1 is not None
    assert lease2 is not None
    assert lease1.timestamp < lease2.timestamp


def test_multiple_done_calls(queue: RayQueue):
    """Test that calling done() on same lease twice fails."""
    queue.push("task1")
    lease = queue.pop()
    assert lease is not None

    queue.done(lease)

    with pytest.raises(Exception, match="Invalid lease"):
        queue.done(lease)


def test_done_after_release(queue: RayQueue):
    """Test that done() fails after release()."""
    queue.push("task1")
    lease = queue.pop()
    assert lease is not None

    queue.release(lease)

    with pytest.raises(Exception, match="Invalid lease"):
        queue.done(lease)


def test_release_after_done(queue: RayQueue):
    """Test that release() fails after done()."""
    queue.push("task1")
    lease = queue.pop()
    assert lease is not None

    queue.done(lease)

    with pytest.raises(Exception, match="Invalid lease"):
        queue.release(lease)


def test_multiple_queues_independent(ray_context):
    """Test that multiple queues maintain independent state."""
    queue1 = RayQueue("queue1")
    queue2 = RayQueue("queue2")

    queue1.push("task1")
    queue2.push("task2")

    assert queue1.size() == 1
    assert queue2.size() == 1

    lease1 = queue1.pop()
    assert lease1 is not None
    assert lease1.item == "task1"

    lease2 = queue2.pop()
    assert lease2 is not None
    assert lease2.item == "task2"

    assert queue1.pending() == 0
    assert queue2.pending() == 0
