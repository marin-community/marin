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

import time
from typing import Any

import pytest

from fray.cluster.queue import Lease, Queue


class SimpleQueue(Queue[Any]):
    """Simple in-memory implementation of Queue for testing the protocol."""

    def __init__(self):
        self._items: list[Any] = []
        self._leased: dict[str, tuple[Any, float]] = {}
        self._next_lease_id = 0

    def push(self, item: Any) -> None:
        self._items.append(item)

    def peek(self) -> Any | None:
        if not self._items:
            return None
        return self._items[0]

    def pop(self) -> Lease[Any] | None:
        if not self._items:
            return None
        item = self._items.pop(0)
        lease_id = str(self._next_lease_id)
        self._next_lease_id += 1
        timestamp = time.time()
        self._leased[lease_id] = (item, timestamp)
        return Lease(item=item, lease_id=lease_id, timestamp=timestamp)

    def done(self, lease: Lease[Any]) -> None:
        if lease.lease_id not in self._leased:
            raise ValueError(f"Invalid lease: {lease.lease_id}")
        del self._leased[lease.lease_id]

    def release(self, lease: Lease[Any]) -> None:
        if lease.lease_id not in self._leased:
            raise ValueError(f"Invalid lease: {lease.lease_id}")
        item, _ = self._leased[lease.lease_id]
        del self._leased[lease.lease_id]
        self._items.append(item)

    def size(self) -> int:
        return len(self._items) + len(self._leased)

    def pending(self) -> int:
        return len(self._items)


@pytest.fixture
def queue() -> SimpleQueue:
    """Create a simple queue for testing."""
    return SimpleQueue()


def test_lease_dataclass():
    """Test Lease dataclass creation."""
    lease = Lease(item="task1", lease_id="lease-123", timestamp=1234567890.0)
    assert lease.item == "task1"
    assert lease.lease_id == "lease-123"
    assert lease.timestamp == 1234567890.0


def test_push_and_peek(queue: SimpleQueue):
    """Test pushing items and peeking at the next item."""
    assert queue.peek() is None

    queue.push("task1")
    assert queue.peek() == "task1"

    queue.push("task2")
    assert queue.peek() == "task1"  # Still returns first item


def test_push_and_pop(queue: SimpleQueue):
    """Test pushing and popping items."""
    assert queue.pop() is None

    queue.push("task1")
    lease = queue.pop()
    assert lease is not None
    assert lease.item == "task1"
    assert isinstance(lease.lease_id, str)
    assert isinstance(lease.timestamp, float)
    assert lease.timestamp <= time.time()


def test_pop_returns_fifo(queue: SimpleQueue):
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


def test_done_completes_lease(queue: SimpleQueue):
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


def test_done_with_invalid_lease(queue: SimpleQueue):
    """Test that done() raises error for invalid lease."""
    queue.push("task1")
    lease = queue.pop()
    assert lease is not None

    # Complete the lease
    queue.done(lease)

    # Trying to complete again should raise
    with pytest.raises(ValueError, match="Invalid lease"):
        queue.done(lease)


def test_release_requeues_item(queue: SimpleQueue):
    """Test that release() requeues the item."""
    queue.push("task1")
    queue.push("task2")

    lease = queue.pop()
    assert lease is not None
    assert lease.item == "task1"
    assert queue.pending() == 1  # only task2 pending

    queue.release(lease)
    assert queue.pending() == 2  # both items pending again

    # task1 should be available again (at the end)
    next_lease = queue.pop()
    assert next_lease is not None
    assert next_lease.item == "task2"  # task2 was next in line


def test_release_with_invalid_lease(queue: SimpleQueue):
    """Test that release() raises error for invalid lease."""
    queue.push("task1")
    lease = queue.pop()
    assert lease is not None

    # Release the lease
    queue.release(lease)

    # Trying to release again should raise
    with pytest.raises(ValueError, match="Invalid lease"):
        queue.release(lease)


def test_size_counts_all_items(queue: SimpleQueue):
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


def test_pending_counts_unleased_items(queue: SimpleQueue):
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


def test_complete_workflow(queue: SimpleQueue):
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

    # Worker 3 processes successfully
    lease3 = queue.pop()
    assert lease3 is not None
    assert lease3.item == "task2"
    queue.done(lease3)

    # The requeued task should be available
    assert queue.pending() == 3  # task1 (requeued), task3, task4
    assert queue.size() == 3

    # Process the requeued task
    lease4 = queue.pop()
    assert lease4 is not None
    # Could be task3 or task4 depending on queue order, but task1 should still be there
    assert queue.size() == 3


def test_empty_queue_operations(queue: SimpleQueue):
    """Test operations on an empty queue."""
    assert queue.peek() is None
    assert queue.pop() is None
    assert queue.size() == 0
    assert queue.pending() == 0


def test_queue_with_different_types():
    """Test that Queue works with different types."""
    int_queue: Queue[int] = SimpleQueue()
    int_queue.push(42)
    lease = int_queue.pop()
    assert lease is not None
    assert lease.item == 42
    assert isinstance(lease.item, int)

    str_queue: Queue[str] = SimpleQueue()
    str_queue.push("hello")
    lease2 = str_queue.pop()
    assert lease2 is not None
    assert lease2.item == "hello"
    assert isinstance(lease2.item, str)


def test_lease_timestamp_ordering(queue: SimpleQueue):
    """Test that lease timestamps are correctly ordered."""
    queue.push("task1")
    time.sleep(0.01)  # Small delay to ensure different timestamps
    queue.push("task2")

    lease1 = queue.pop()
    time.sleep(0.01)
    lease2 = queue.pop()

    assert lease1 is not None
    assert lease2 is not None
    assert lease1.timestamp < lease2.timestamp


def test_multiple_done_calls(queue: SimpleQueue):
    """Test that calling done() on same lease twice fails."""
    queue.push("task1")
    lease = queue.pop()
    assert lease is not None

    queue.done(lease)

    with pytest.raises(ValueError, match="Invalid lease"):
        queue.done(lease)


def test_done_after_release(queue: SimpleQueue):
    """Test that done() fails after release()."""
    queue.push("task1")
    lease = queue.pop()
    assert lease is not None

    queue.release(lease)

    with pytest.raises(ValueError, match="Invalid lease"):
        queue.done(lease)


def test_release_after_done(queue: SimpleQueue):
    """Test that release() fails after done()."""
    queue.push("task1")
    lease = queue.pop()
    assert lease is not None

    queue.done(lease)

    with pytest.raises(ValueError, match="Invalid lease"):
        queue.release(lease)
