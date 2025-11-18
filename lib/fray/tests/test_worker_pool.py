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

import logging
import time

import pytest
import ray
from fray.cluster import set_current_cluster
from fray.cluster.local import LocalCluster
from fray.cluster.ray import RayCluster
from fray.examples.fake_llm_worker import worker_loop
from fray.worker_pool import WorkerPool, WorkerPoolConfig

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def ray_context():
    """Initialize Ray once for all Ray-based tests."""
    ray.init(
        address="local",
        ignore_reinit_error=True,
        namespace="fray_test",
        num_cpus=8,
        _memory=20 * 1024 * 1024 * 1024,  # 20GB total memory
        object_store_memory=2 * 1024 * 1024 * 1024,  # 2GB object store
        _system_config={"automatic_object_spilling_enabled": True},
    )
    yield
    ray.shutdown()


@pytest.fixture(params=["local", "ray"])
def cluster(request, ray_context):
    """Parameterized fixture providing LocalCluster and RayCluster implementations."""
    if request.param == "local":
        cluster = LocalCluster()
    else:  # ray
        cluster = RayCluster()

    # Set current cluster context for workers
    set_current_cluster(cluster)

    yield cluster

    # Cleanup any running jobs
    for job in cluster.list_jobs():
        if job.status == "running" and job.job_id is not None:
            cluster.terminate(job.job_id)


def test_worker_pool_inference(cluster):
    logger.info(f"Testing worker pool with cluster type: {type(cluster).__name__}")

    config = WorkerPoolConfig(
        worker_func=worker_loop,
        min_workers=1,
        max_workers=3,
        scale_up_threshold=0.5,  # Scale up when queue has >0.5 tasks per worker
        scale_down_threshold=0.1,  # Scale down when queue has <0.1 tasks per worker
        scale_check_interval=0.5,  # Check every 0.5 seconds
    )

    pool = WorkerPool(
        cluster=cluster,
        config=config,
    )

    logger.info(f"Pool created, num_workers: {pool.num_workers()}")

    jobs = cluster.list_jobs()
    logger.info(f"Cluster jobs: {len(jobs)}")
    for job in jobs:
        logger.info(f"  Job {job.job_id}: status={job.status}, name={job.name}")

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
        for problem in problems:
            pool.submit(problem)

        logger.info(f"Submitted {len(problems)} problems")
        logger.info(f"Task queue size: {pool._task_queue.size()}, pending: {pool._task_queue.pending()}")
        logger.info(f"Result queue size: {pool._result_queue.size()}, pending: {pool._result_queue.pending()}")

        results = pool.collect(num_results=len(problems), timeout=30.0)
        assert len(results) == len(problems)
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


def test_worker_pool_autoscaling(cluster):
    """Test that worker pool scales up and down based on queue depth."""
    config = WorkerPoolConfig(
        worker_func=worker_loop,
        min_workers=1,
        max_workers=4,
        scale_up_threshold=1.0,  # Scale up when >1 task per worker
        scale_down_threshold=0.2,  # Scale down when <0.2 tasks per worker
        scale_check_interval=0.5,
    )

    pool = WorkerPool(
        cluster=cluster,
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


def test_worker_pool_shutdown(cluster):
    """Test graceful shutdown of worker pool."""
    config = WorkerPoolConfig(
        worker_func=worker_loop,
        min_workers=2,
        max_workers=4,
    )

    pool = WorkerPool(
        cluster=cluster,
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
    running_jobs = [j for j in cluster.list_jobs() if j.status == "running"]
    # Filter to only worker jobs (they'll have specific naming pattern)
    worker_jobs = [j for j in running_jobs if j.job_id is not None and "worker" in j.job_id.lower()]
    assert len(worker_jobs) == 0
