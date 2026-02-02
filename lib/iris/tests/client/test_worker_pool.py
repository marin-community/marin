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

"""E2E tests for WorkerPool using LocalClusterClient."""

import pytest

from iris.client import IrisClient
from iris.client.worker_pool import (
    WorkerPool,
    WorkerPoolConfig,
)
from iris.cluster.types import ResourceSpec


@pytest.fixture
def local_client():
    """Create a LocalClusterClient-backed IrisClient for true E2E testing.

    This fixture starts a real Controller and Worker with in-process execution,
    ensuring WorkerPool tests go through the full job submission infrastructure.
    """
    client = IrisClient.local()
    yield client
    client.shutdown()


class TestWorkerPoolE2E:
    """True end-to-end tests for WorkerPool using LocalClusterClient.

    These tests exercise the full job submission flow:
    WorkerPool -> IrisClient -> LocalClusterClient -> Controller -> Worker -> task execution.
    """

    def test_submit_executes_task(self, local_client):
        """submit() dispatches a task through real job infrastructure and returns correct result."""
        config = WorkerPoolConfig(
            num_workers=1,
            resources=ResourceSpec(cpu=1, memory="512m"),
        )

        with WorkerPool(local_client, config, timeout=30.0) as pool:

            def add(a, b):
                return a + b

            future = pool.submit(add, 10, 20)
            result = future.result(timeout=60.0)

            assert result == 30

    def test_map_executes_tasks(self, local_client):
        """map() distributes work through real job infrastructure."""
        config = WorkerPoolConfig(
            num_workers=2,
            resources=ResourceSpec(cpu=1, memory="512m"),
        )

        with WorkerPool(local_client, config, timeout=30.0) as pool:

            def square(x):
                return x * x

            futures = pool.map(square, [1, 2, 3, 4, 5])
            results = [f.result(timeout=60.0) for f in futures]

            assert results == [1, 4, 9, 16, 25]

    def test_exception_propagates_to_caller(self, local_client):
        """Exceptions raised by user code propagate through job infrastructure to caller."""
        config = WorkerPoolConfig(
            num_workers=1,
            resources=ResourceSpec(cpu=1, memory="512m"),
        )

        with WorkerPool(local_client, config, timeout=30.0) as pool:

            def fail():
                raise ValueError("intentional error")

            future = pool.submit(fail)

            with pytest.raises(ValueError, match="intentional error"):
                future.result(timeout=60.0)

    def test_shutdown_prevents_new_submissions(self, local_client):
        """After shutdown, submit() raises RuntimeError."""
        config = WorkerPoolConfig(
            num_workers=1,
            resources=ResourceSpec(cpu=1, memory="512m"),
        )

        pool = WorkerPool(local_client, config, timeout=30.0)
        pool.__enter__()

        pool.shutdown(wait=False)

        with pytest.raises(RuntimeError, match="shutdown"):
            pool.submit(lambda: 42)

    def test_multiple_sequential_tasks(self, local_client):
        """Multiple tasks can be submitted sequentially to the same pool."""
        config = WorkerPoolConfig(
            num_workers=1,
            resources=ResourceSpec(cpu=1, memory="512m"),
        )

        with WorkerPool(local_client, config, timeout=30.0) as pool:
            results = []
            for i in range(3):
                future = pool.submit(lambda x: x * 2, i)
                results.append(future.result(timeout=60.0))

            assert results == [0, 2, 4]
