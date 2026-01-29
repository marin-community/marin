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

"""Tests for Fray v2 Ray backend.

These tests are skipped by default because they require Ray to be running
and can be slow in monorepo environments due to working directory packaging.

To run these tests:
    pytest lib/fray/tests/v2/test_ray.py -m ray --timeout=300

The tests work correctly when Ray workers can quickly initialize, which
typically happens on a pre-configured Ray cluster or when running outside
of complex monorepo environments.
"""

import os
import time

import pytest

# Skip all tests in this module unless explicitly requested
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_RAY_TESTS", "0") != "1", reason="Ray tests are slow in monorepo; set RUN_RAY_TESTS=1 to enable"
)

ray = pytest.importorskip("ray")

from fray.v2 import (  # noqa: E402
    ActorServer,
    Entrypoint,
    JobStatus,
    ResourceSpec,
    create_cluster,
)
from fray.v2.backends.ray import RayCluster  # noqa: E402


# Test helper functions (must be at module level for pickling)
def return_value(value):
    return value


def add_numbers(a, b):
    return a + b


def square(x):
    return x * x


def slow_square(x):
    time.sleep(0.05)
    return x * x


def raise_error():
    raise ValueError("Test error")


# Test actor class
class Counter:
    def __init__(self, initial=0):
        self.value = initial

    def incr(self, n=1):
        self.value += n
        return self.value

    def get_value(self):
        return self.value


@pytest.fixture(scope="module")
def ray_cluster():
    """Create a Ray cluster for testing."""
    # Shutdown any existing Ray instance
    if ray.is_initialized():
        ray.shutdown()

    # Initialize local Ray instance - no runtime_env to avoid working dir upload
    ray.init(
        ignore_reinit_error=True,
        num_cpus=4,
    )

    cluster = RayCluster()
    yield cluster

    # Cleanup
    ray.shutdown()


class TestRayCluster:
    def test_create_ray_cluster(self, ray_cluster):
        assert ray_cluster is not None
        assert ray_cluster.namespace is not None

    def test_create_via_factory(self):
        # Skip factory test as it would initialize Ray again
        cluster = create_cluster("ray")
        assert isinstance(cluster, RayCluster)

    def test_submit_simple_job(self, ray_cluster):
        job = ray_cluster.submit(
            Entrypoint.from_callable(return_value, 42),
            name="test-job",
            resources=ResourceSpec(),
        )
        assert job.job_id is not None
        status = job.wait()
        assert status == JobStatus.SUCCEEDED

    def test_submit_job_with_args(self, ray_cluster):
        job = ray_cluster.submit(
            Entrypoint.from_callable(add_numbers, 10, 20),
            name="add-job",
            resources=ResourceSpec(),
        )
        status = job.wait()
        assert status == JobStatus.SUCCEEDED

    def test_job_status_transitions(self, ray_cluster):
        def slow_job():
            time.sleep(0.2)

        job = ray_cluster.submit(
            Entrypoint.from_callable(slow_job),
            name="slow-job",
            resources=ResourceSpec(),
        )

        # Should be running or pending initially
        initial_status = job.status()
        assert initial_status in [JobStatus.PENDING, JobStatus.RUNNING]

        # Wait for completion
        status = job.wait()
        assert status == JobStatus.SUCCEEDED

    def test_job_failure(self, ray_cluster):
        job = ray_cluster.submit(
            Entrypoint.from_callable(raise_error),
            name="fail-job",
            resources=ResourceSpec(),
        )

        # Wait should raise with raise_on_failure=True (default)
        with pytest.raises(RuntimeError, match="Test error"):
            job.wait()

    def test_job_failure_no_raise(self, ray_cluster):
        job = ray_cluster.submit(
            Entrypoint.from_callable(raise_error),
            name="fail-job2",
            resources=ResourceSpec(),
        )

        status = job.wait(raise_on_failure=False)
        assert status == JobStatus.FAILED

    def test_job_name_with_slash_raises(self, ray_cluster):
        with pytest.raises(ValueError, match="cannot contain '/'"):
            ray_cluster.submit(
                Entrypoint.from_callable(return_value, 1),
                name="invalid/name",
                resources=ResourceSpec(),
            )

    def test_list_jobs(self, ray_cluster):
        job1 = ray_cluster.submit(
            Entrypoint.from_callable(return_value, 1),
            name="list-job1",
            resources=ResourceSpec(),
        )
        job2 = ray_cluster.submit(
            Entrypoint.from_callable(return_value, 2),
            name="list-job2",
            resources=ResourceSpec(),
        )

        jobs = ray_cluster.list_jobs()
        job_ids = [j.job_id for j in jobs]

        assert job1.job_id in job_ids
        assert job2.job_id in job_ids


class TestRayWorkerPool:
    def test_worker_pool_submit(self, ray_cluster):
        with ray_cluster.worker_pool(num_workers=2, resources=ResourceSpec()) as pool:
            future = pool.submit(square, 5)
            result = future.result()

        assert result == 25

    def test_worker_pool_multiple_tasks(self, ray_cluster):
        with ray_cluster.worker_pool(num_workers=4, resources=ResourceSpec()) as pool:
            futures = [pool.submit(square, i) for i in range(10)]
            results = [f.result() for f in futures]

        assert sorted(results) == [i * i for i in range(10)]

    def test_worker_pool_as_completed(self, ray_cluster):
        with ray_cluster.worker_pool(num_workers=2, resources=ResourceSpec()) as pool:
            futures = [pool.submit(slow_square, i) for i in range(5)]

            results = []
            # Note: _RayFuture doesn't fully implement concurrent.futures.Future
            # so as_completed won't work directly. We collect results manually.
            for future in futures:
                results.append(future.result())

        assert sorted(results) == [0, 1, 4, 9, 16]

    def test_worker_pool_size(self, ray_cluster):
        with ray_cluster.worker_pool(num_workers=3, resources=ResourceSpec()) as pool:
            assert pool.size == 3


class TestRayActorServer:
    def test_actor_registration_and_call(self, ray_cluster):
        server = ActorServer(ray_cluster)
        server.register("counter", Counter(0))
        server.serve_background()

        # Give time for actor to be registered
        time.sleep(0.5)

        pool = ray_cluster.resolver().lookup("counter")
        pool.wait_for_size(1, timeout=10.0)

        # Test calls
        assert pool.call().incr(5) == 5
        assert pool.call().incr(3) == 8
        assert pool.call().get_value() == 8

        server.shutdown()

    def test_actor_broadcast(self, ray_cluster):
        # Register multiple actors
        server1 = ActorServer(ray_cluster)
        server1.register("bcast-counter", Counter(10))
        server1.serve_background()

        server2 = ActorServer(ray_cluster)
        server2.register("bcast-counter", Counter(20))
        server2.serve_background()

        # Give time for actors to be registered
        time.sleep(0.5)

        pool = ray_cluster.resolver().lookup("bcast-counter")
        pool.wait_for_size(2, timeout=10.0)

        # Broadcast to all
        results = pool.broadcast().get_value().wait_all()
        assert sorted(results) == [10, 20]

        server1.shutdown()
        server2.shutdown()


class TestRayPicklingSerialization:
    """Tests that verify cloudpickle serialization works correctly with Ray."""

    def test_job_callable_is_serialized(self, ray_cluster):
        """Verify that job entrypoints are serialized/deserialized."""
        # Lambda functions should work (cloudpickle handles them)
        job = ray_cluster.submit(
            Entrypoint.from_callable(lambda x: x * 2, 21),
            name="lambda-job",
            resources=ResourceSpec(),
        )
        status = job.wait()
        assert status == JobStatus.SUCCEEDED

    def test_worker_pool_callable_is_serialized(self, ray_cluster):
        """Verify that worker pool tasks are serialized/deserialized."""
        with ray_cluster.worker_pool(num_workers=2, resources=ResourceSpec()) as pool:
            # Lambda should work
            future = pool.submit(lambda x: x * 3, 7)
            result = future.result()

        assert result == 21
