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

"""Pytest fixtures for zephyr tests."""

import threading
import time

import pytest
import ray

from fray.job import create_job_ctx
from fray.job.rpc.controller import FrayControllerServer
from fray.job.rpc.worker import FrayWorker

from zephyr import load_file


@pytest.fixture(scope="module")
def ray_cluster():
    """Start Ray cluster for tests."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    yield
    # Don't shutdown - let pytest handle cleanup


@pytest.fixture(scope="module")
def rpc_infrastructure():
    """Start RPC controller and workers for Zephyr tests.

    Creates a controller server and 2 workers to match threadpool
    parallelism (max_workers=2).
    """
    # Start controller on random port
    server = FrayControllerServer(port=0)
    port = server.start()

    # Start 2 workers for parallel execution
    workers = []
    threads = []

    for i in range(2):
        worker = FrayWorker(f"http://localhost:{port}", port=0)
        workers.append(worker)
        thread = threading.Thread(target=worker.run, daemon=True, name=f"zephyr-rpc-worker-{i}")
        thread.start()
        threads.append(thread)

    # Give workers time to register
    time.sleep(0.3)

    yield port

    # Cleanup
    for worker in workers:
        worker.stop()
    server.stop()
    for thread in threads:
        thread.join(timeout=2.0)


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return list(range(1, 11))  # [1, 2, 3, ..., 10]


@pytest.fixture(
    params=[
        pytest.param("sync", id="sync"),
        pytest.param("threadpool", id="thread"),
        pytest.param("ray", id="ray"),
        pytest.param("rpc", id="rpc"),
    ]
)
def backend(request, ray_cluster, rpc_infrastructure):
    """Parametrized fixture providing all job contexts for testing.

    Tests run against all 4 backends: sync, threadpool, ray, and rpc.
    """
    backend_type = request.param

    if backend_type == "sync":
        return create_job_ctx("sync")
    elif backend_type == "threadpool":
        return create_job_ctx("threadpool", max_workers=2)
    elif backend_type == "ray":
        return create_job_ctx("ray")
    elif backend_type == "rpc":
        return create_job_ctx("fray", controller_address=f"http://localhost:{rpc_infrastructure}")
    else:
        raise ValueError(f"Unknown backend: {backend_type}")


class CallCounter:
    """Helper to track function calls across test scenarios."""

    def __init__(self):
        self.flat_map_count = 0
        self.map_count = 0
        self.processed_ids = []

    def reset(self):
        self.flat_map_count = 0
        self.map_count = 0
        self.processed_ids = []

    def counting_flat_map(self, path):
        self.flat_map_count += 1
        return load_file(path)

    def counting_map(self, x):
        self.map_count += 1
        self.processed_ids.append(x["id"])
        return {**x, "processed": True}
