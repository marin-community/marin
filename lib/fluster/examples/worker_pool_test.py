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

"""Minimal reproduction of WorkerPool timeout issue.

Usage:
    cd lib/fluster
    uv run python examples/worker_pool_test.py
"""

import logging

from fluster.client import FlusterClient, LocalClientConfig
from fluster.cluster.types import Entrypoint
from fluster.rpc import cluster_pb2

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")


def workerpool_coordinator():
    """Coordinator job that demonstrates WorkerPool usage."""
    from fluster.client import WorkerPool, WorkerPoolConfig, fluster_ctx

    ctx = fluster_ctx()
    print(f"Coordinator starting (job_id={ctx.job_id})")

    def square(n: int) -> int:
        return n * n

    config = WorkerPoolConfig(
        num_workers=2,
        resources=cluster_pb2.ResourceSpec(cpu=1, memory="512m"),
        name_prefix="pool-worker",
    )

    print(f"Creating WorkerPool with {config.num_workers} workers...")

    with WorkerPool(ctx.client, config, timeout=30.0) as pool:
        print(f"Pool ready: {pool.size} workers available")

        items = list(range(1, 5))
        print(f"Computing squares of {items}...")

        futures = pool.map(square, items)
        results = [f.result(timeout=30.0) for f in futures]

        print(f"Results: {results}")
        expected = [i * i for i in items]
        assert results == expected, f"Expected {expected}, got {results}"

    print("WorkerPool test completed!")


def main():
    print("Starting local cluster...")
    config = LocalClientConfig(max_workers=4)

    with FlusterClient.local(config) as client:
        print("Submitting coordinator job...")
        job_id = client.submit(
            entrypoint=Entrypoint.from_callable(workerpool_coordinator),
            name="workerpool-test",
            resources=cluster_pb2.ResourceSpec(cpu=1, memory="512m"),
        )
        print(f"Job submitted: {job_id}")

        print("Waiting with log streaming...")
        status = client.wait(job_id, timeout=60.0, stream_logs=True)

        state_name = cluster_pb2.JobState.Name(status.state)
        print(f"Final state: {state_name}")

        if status.state != cluster_pb2.JOB_STATE_SUCCEEDED:
            print(f"ERROR: {status.error}")
            return 1

    print("SUCCESS!")
    return 0


if __name__ == "__main__":
    exit(main())
