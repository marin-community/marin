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

"""Tests for Fray v2 local backend."""

import concurrent.futures
import time

import pytest

from fray.v2 import (
    ActorServer,
    Entrypoint,
    JobStatus,
    LocalCluster,
    ResourceSpec,
    create_cluster,
    current_cluster,
    set_current_cluster,
)


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


class TestLocalCluster:
    def test_create_local_cluster(self):
        cluster = LocalCluster()
        assert cluster is not None
        assert cluster.namespace is not None

    def test_create_via_factory(self):
        cluster = create_cluster("local")
        assert isinstance(cluster, LocalCluster)

    def test_submit_simple_job(self):
        cluster = LocalCluster()
        job = cluster.submit(
            Entrypoint.from_callable(return_value, 42),
            name="test-job",
            resources=ResourceSpec(),
        )
        assert job.job_id is not None
        status = job.wait()
        assert status == JobStatus.SUCCEEDED

    def test_submit_job_with_args(self):
        cluster = LocalCluster()
        job = cluster.submit(
            Entrypoint.from_callable(add_numbers, 10, 20),
            name="add-job",
            resources=ResourceSpec(),
        )
        status = job.wait()
        assert status == JobStatus.SUCCEEDED

    def test_job_status_transitions(self):
        cluster = LocalCluster()

        def slow_job():
            time.sleep(0.2)

        job = cluster.submit(
            Entrypoint.from_callable(slow_job),
            name="slow-job",
            resources=ResourceSpec(),
        )

        # Should be running or pending
        initial_status = job.status()
        assert initial_status in [JobStatus.PENDING, JobStatus.RUNNING]

        # Wait for completion
        status = job.wait()
        assert status == JobStatus.SUCCEEDED

    def test_job_failure(self):
        cluster = LocalCluster()
        job = cluster.submit(
            Entrypoint.from_callable(raise_error),
            name="fail-job",
            resources=ResourceSpec(),
        )

        # Wait should raise with raise_on_failure=True (default)
        with pytest.raises(RuntimeError, match="Test error"):
            job.wait()

    def test_job_failure_no_raise(self):
        cluster = LocalCluster()
        job = cluster.submit(
            Entrypoint.from_callable(raise_error),
            name="fail-job",
            resources=ResourceSpec(),
        )

        status = job.wait(raise_on_failure=False)
        assert status == JobStatus.FAILED

    def test_job_name_with_slash_raises(self):
        cluster = LocalCluster()
        with pytest.raises(ValueError, match="cannot contain '/'"):
            cluster.submit(
                Entrypoint.from_callable(return_value, 1),
                name="invalid/name",
                resources=ResourceSpec(),
            )

    def test_list_jobs(self):
        cluster = LocalCluster()

        job1 = cluster.submit(
            Entrypoint.from_callable(return_value, 1),
            name="job1",
            resources=ResourceSpec(),
        )
        job2 = cluster.submit(
            Entrypoint.from_callable(return_value, 2),
            name="job2",
            resources=ResourceSpec(),
        )

        jobs = cluster.list_jobs()
        job_ids = [j.job_id for j in jobs]

        assert job1.job_id in job_ids
        assert job2.job_id in job_ids

    def test_terminate_job(self):
        cluster = LocalCluster()

        def long_running():
            time.sleep(10)

        job = cluster.submit(
            Entrypoint.from_callable(long_running),
            name="long-job",
            resources=ResourceSpec(),
        )

        cluster.terminate(job.job_id)
        assert job.status() == JobStatus.KILLED


class TestLocalWorkerPool:
    def test_worker_pool_submit(self):
        cluster = LocalCluster()

        with cluster.worker_pool(num_workers=2, resources=ResourceSpec()) as pool:
            future = pool.submit(square, 5)
            result = future.result()

        assert result == 25

    def test_worker_pool_multiple_tasks(self):
        cluster = LocalCluster()

        with cluster.worker_pool(num_workers=4, resources=ResourceSpec()) as pool:
            futures = [pool.submit(square, i) for i in range(10)]
            results = [f.result() for f in futures]

        assert sorted(results) == [i * i for i in range(10)]

    def test_worker_pool_as_completed(self):
        cluster = LocalCluster()

        with cluster.worker_pool(num_workers=4, resources=ResourceSpec()) as pool:
            futures = [pool.submit(slow_square, i) for i in range(5)]

            results = []
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

        assert sorted(results) == [0, 1, 4, 9, 16]

    def test_worker_pool_size(self):
        cluster = LocalCluster()

        with cluster.worker_pool(num_workers=3, resources=ResourceSpec()) as pool:
            assert pool.size == 3


class TestLocalActorServer:
    def test_actor_registration_and_call(self):
        cluster = LocalCluster()

        server = ActorServer(cluster)
        server.register("counter", Counter(0))
        server.serve_background()

        pool = cluster.resolver().lookup("counter")
        pool.wait_for_size(1)

        # Test calls
        assert pool.call().incr(5) == 5
        assert pool.call().incr(3) == 8
        assert pool.call().get_value() == 8

        server.shutdown()

    def test_actor_broadcast(self):
        cluster = LocalCluster()

        # Register multiple actors
        server1 = ActorServer(cluster)
        server1.register("counter", Counter(10))
        server1.serve_background()

        server2 = ActorServer(cluster)
        server2.register("counter", Counter(20))
        server2.serve_background()

        pool = cluster.resolver().lookup("counter")
        pool.wait_for_size(2)

        # Broadcast to all
        results = pool.broadcast().get_value().wait_all()
        assert sorted(results) == [10, 20]

        server1.shutdown()
        server2.shutdown()

    def test_actor_pool_round_robin(self):
        cluster = LocalCluster()

        server1 = ActorServer(cluster)
        server1.register("counter", Counter(100))
        server1.serve_background()

        server2 = ActorServer(cluster)
        server2.register("counter", Counter(200))
        server2.serve_background()

        pool = cluster.resolver().lookup("counter")
        pool.wait_for_size(2)

        # Round-robin should alternate between actors
        results = [pool.call().get_value() for _ in range(4)]
        # Should get both values
        assert 100 in results
        assert 200 in results

        server1.shutdown()
        server2.shutdown()


class TestCurrentCluster:
    def test_current_cluster_default(self):
        # Clear any existing context
        set_current_cluster(None)

        # Should create LocalCluster by default
        cluster = current_cluster()
        assert isinstance(cluster, LocalCluster)

    def test_set_current_cluster(self):
        cluster1 = LocalCluster()
        set_current_cluster(cluster1)

        assert current_cluster() is cluster1

        # Clean up
        set_current_cluster(None)


class TestPicklingSerialization:
    """Tests that verify cloudpickle serialization works correctly."""

    def test_job_callable_is_serialized(self):
        """Verify that job entrypoints are serialized/deserialized."""
        cluster = LocalCluster()

        # Lambda functions should work (cloudpickle handles them)
        job = cluster.submit(
            Entrypoint.from_callable(lambda x: x * 2, 21),
            name="lambda-job",
            resources=ResourceSpec(),
        )
        status = job.wait()
        assert status == JobStatus.SUCCEEDED

    def test_worker_pool_callable_is_serialized(self):
        """Verify that worker pool tasks are serialized/deserialized."""
        cluster = LocalCluster()

        with cluster.worker_pool(num_workers=2, resources=ResourceSpec()) as pool:
            # Lambda should work
            future = pool.submit(lambda x: x * 3, 7)
            result = future.result()

        assert result == 21

    def test_actor_args_are_serialized(self):
        """Verify that actor call args are serialized/deserialized."""
        cluster = LocalCluster()

        class Echo:
            def echo(self, data):
                return data

        server = ActorServer(cluster)
        server.register("echo", Echo())
        server.serve_background()

        pool = cluster.resolver().lookup("echo")
        pool.wait_for_size(1)

        # Complex data should be serialized correctly
        test_data = {"key": [1, 2, 3], "nested": {"a": "b"}}
        result = pool.call().echo(test_data)
        assert result == test_data

        server.shutdown()
