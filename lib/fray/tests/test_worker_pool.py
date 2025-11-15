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

"""Tests for worker pool with autoscaling."""

import time

import pytest

from fray.cluster.local import LocalCluster
from fray.worker_pool import WorkerPool, WorkerPoolConfig


@pytest.fixture
def local_cluster():
    """Fixture for local cluster."""
    cluster = LocalCluster()
    yield cluster
    # Cleanup any running jobs
    for job in cluster.list_jobs():
        if job.status == "running":
            cluster.terminate(job.job_id)


def test_worker_pool_local_inference(local_cluster):
    """Test worker pool with local cluster backend using fake LLM calls.

    This test exercises:
    - LocalCluster backend
    - Worker jobs launched via cluster.launch()
    - Fake LLM inference (mock completions)
    - Batch task submission via WorkerPool
    - Result collection and validation
    - Basic autoscaling (scale up)
    """
    # Create worker pool with autoscaling config
    config = WorkerPoolConfig(
        min_workers=1,
        max_workers=3,
        scale_up_threshold=0.5,  # Scale up when queue has >0.5 tasks per worker
        scale_down_threshold=0.1,  # Scale down when queue has <0.1 tasks per worker
        scale_check_interval=0.5,  # Check every 0.5 seconds
    )

    pool = WorkerPool(
        cluster=local_cluster,
        worker_module="fray.examples.fake_llm_worker",
        config=config,
    )

    try:
        # Submit a batch of fake math problems
        problems = [
            {"problem": "What is 2 + 2?", "id": 1},
            {"problem": "What is 5 * 3?", "id": 2},
            {"problem": "What is 10 - 7?", "id": 3},
            {"problem": "What is 16 / 4?", "id": 4},
            {"problem": "What is 9 + 6?", "id": 5},
        ]

        # Submit batch - this should trigger autoscaling
        results = pool.map(problems, timeout=30.0)

        # Validate results
        assert len(results) == len(problems)

        # Each result should have the expected structure
        for result in results:
            assert "problem_id" in result
            assert "answer" in result
            assert result["problem_id"] in [p["id"] for p in problems]

        # Verify all problems were processed
        result_ids = {r["problem_id"] for r in results}
        problem_ids = {p["id"] for p in problems}
        assert result_ids == problem_ids

    finally:
        # Cleanup
        pool.shutdown(timeout=10.0)


def test_worker_pool_autoscaling(local_cluster):
    """Test that worker pool scales up and down based on queue depth."""
    config = WorkerPoolConfig(
        min_workers=1,
        max_workers=4,
        scale_up_threshold=1.0,  # Scale up when >1 task per worker
        scale_down_threshold=0.2,  # Scale down when <0.2 tasks per worker
        scale_check_interval=0.5,
    )

    pool = WorkerPool(
        cluster=local_cluster,
        worker_module="fray.examples.fake_llm_worker",
        config=config,
    )

    try:
        # Start with min workers
        time.sleep(1.0)  # Let pool initialize
        initial_workers = pool.num_workers()
        assert initial_workers == config.min_workers

        # Submit a large batch to trigger scale-up
        large_batch = [{"problem": f"Problem {i}", "id": i} for i in range(20)]

        # Submit without waiting for completion
        for problem in large_batch[:10]:
            pool.submit(problem)

        # Give autoscaler time to react
        time.sleep(2.0)

        # Should have scaled up
        scaled_workers = pool.num_workers()
        assert scaled_workers > initial_workers
        assert scaled_workers <= config.max_workers

        # Collect results
        results = pool.collect(num_results=10, timeout=20.0)
        assert len(results) == 10

        # Wait for scale down
        time.sleep(5.0)

        # Should scale back down toward min_workers
        final_workers = pool.num_workers()
        assert final_workers <= scaled_workers

    finally:
        pool.shutdown(timeout=10.0)


def test_worker_pool_shutdown(local_cluster):
    """Test graceful shutdown of worker pool."""
    config = WorkerPoolConfig(
        min_workers=2,
        max_workers=4,
    )

    pool = WorkerPool(
        cluster=local_cluster,
        worker_module="fray.examples.fake_llm_worker",
        config=config,
    )

    # Submit some tasks
    tasks = [{"problem": f"Problem {i}", "id": i} for i in range(5)]
    for task in tasks:
        pool.submit(task)

    # Shutdown should wait for pending tasks
    pool.shutdown(timeout=15.0)

    # All workers should be terminated
    assert pool.num_workers() == 0

    # Cluster should have no running jobs
    running_jobs = [j for j in local_cluster.list_jobs() if j.status == "running"]
    # Filter to only worker jobs (they'll have specific naming pattern)
    worker_jobs = [j for j in running_jobs if "worker" in j.job_id.lower()]
    assert len(worker_jobs) == 0
