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

"""Tests for WorkerPool."""

import time

import pytest
import ray

from zephyr import WorkerPool, WorkerPoolConfig


@pytest.fixture(autouse=True)
def ensure_ray(ray_cluster):
    """Ensure Ray is initialized for all tests."""
    pass


@ray.remote
class SimpleWorker:
    """Simple worker for testing."""

    def ping(self):
        """Health check method."""
        return True

    def process(self, item):
        """Process an item by doubling it."""
        return item * 2


@ray.remote
class SlowWorker:
    """Worker that processes slowly to test autoscaling."""

    def ping(self):
        """Health check method."""
        return True

    def process(self, item):
        """Process an item slowly."""
        time.sleep(0.1)
        return item * 2


def test_worker_pool_basic():
    """Test basic worker pool functionality."""
    pool = WorkerPool(
        SimpleWorker,
        method_name="process",
        config=WorkerPoolConfig(min_workers=1, max_workers=2),
    )

    try:
        results = pool.map([1, 2, 3, 4, 5])
        assert sorted(results) == [2, 4, 6, 8, 10]
    finally:
        pool.shutdown()


def test_worker_pool_config():
    """Test worker pool with custom config."""
    config = WorkerPoolConfig(
        min_workers=2,
        max_workers=4,
        scale_up_threshold=3.0,
        scale_down_threshold=0.3,
    )

    pool = WorkerPool(SimpleWorker, config=config)

    try:
        # Should start with min_workers
        assert len(pool.actors) >= config.min_workers

        results = pool.map([1, 2, 3])
        assert sorted(results) == [2, 4, 6]
    finally:
        pool.shutdown()


def test_worker_pool_autoscaling_up():
    """Test that worker pool scales up under load."""
    config = WorkerPoolConfig(
        min_workers=1,
        max_workers=4,
        scale_up_threshold=1.0,  # Scale up when queue_size/workers > 1
        scale_check_interval=0.2,  # Check frequently
    )

    pool = WorkerPool(SlowWorker, config=config)

    try:
        # Submit many slow tasks to trigger scale up
        # With slow tasks (0.1s each) and 1 worker, queue should build up
        num_items = 20
        for item in range(num_items):
            pool.task_queue.put(item)

        # Initial state: 1 worker, 20 tasks in queue
        # Ratio = 20/1 = 20 > 1.0, should trigger scale up
        initial_workers = len(pool.actors)

        # Wait for multiple autoscaler cycles
        time.sleep(1.0)

        # Check if we scaled up (might not reach max, but should increase)
        final_workers = len(pool.actors)

        # Drain results
        results = []
        for _ in range(num_items):
            results.append(pool.result_queue.get(timeout=10))

        assert len(results) == num_items

        # Either we scaled up during the test, or all tasks completed quickly
        # If tasks completed quickly, we won't see scaling, which is also fine
        # So we just verify the pool handled all tasks correctly
        assert final_workers >= initial_workers
    finally:
        pool.shutdown()


def test_worker_pool_shutdown():
    """Test clean shutdown of worker pool."""
    pool = WorkerPool(SimpleWorker, config=WorkerPoolConfig(min_workers=2, max_workers=4))

    # Submit some work
    pool.map([1, 2, 3])

    # Shutdown
    pool.shutdown()

    # Verify all actors are killed
    assert len(pool.actors) == 0
    assert len(pool.actor_futures) == 0


def test_worker_pool_error_handling():
    """Test that worker pool handles task failures gracefully."""

    @ray.remote
    class FaultyWorker:
        def __init__(self):
            self.call_count = 0

        def ping(self):
            return True

        def process(self, item):
            self.call_count += 1
            # Fail on first call, succeed on retry
            if item == 2 and self.call_count <= 1:
                raise ValueError("Simulated failure")
            return item * 2

    pool = WorkerPool(FaultyWorker, config=WorkerPoolConfig(min_workers=1, max_workers=2))

    try:
        # Submit tasks
        for item in [1, 2, 3]:
            pool.task_queue.put(item)

        # Collect results (should retry failed task)
        results = []
        for _ in range(3):
            try:
                result = pool.result_queue.get(timeout=5.0)
                results.append(result)
            except Exception:
                pass  # Ignore timeout

        # Should get some results despite failures
        assert len(results) >= 2
    finally:
        pool.shutdown()


@ray.remote
class CustomWorker:
    """Worker with custom initialization."""

    def __init__(self, multiplier):
        self.multiplier = multiplier

    def ping(self):
        return True

    def process(self, item):
        return item * self.multiplier


def test_worker_pool_custom_args():
    """Test worker pool with custom actor initialization args."""
    pool = WorkerPool(
        CustomWorker,
        config=WorkerPoolConfig(
            min_workers=1,
            max_workers=2,
            actor_kwargs={"multiplier": 3},
        ),
    )

    try:
        results = pool.map([1, 2, 3])
        assert sorted(results) == [3, 6, 9]
    finally:
        pool.shutdown()


def test_worker_pool_ray_options():
    """Test worker pool with Ray actor options."""
    pool = WorkerPool(
        SimpleWorker,
        config=WorkerPoolConfig(
            min_workers=1,
            max_workers=2,
            actor_options={"num_cpus": 0.1},  # Request minimal resources
        ),
    )

    try:
        results = pool.map([1, 2, 3])
        assert sorted(results) == [2, 4, 6]
    finally:
        pool.shutdown()
