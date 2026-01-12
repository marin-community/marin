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

"""Example usage of WorkerContext for manual testing."""

import asyncio

from fluster.cluster.worker.worker_ctx import WorkerContext


async def example_basic(worker: WorkerContext):
    """Basic job submission example."""
    print("\n=== Example: Basic Job Submission ===\n")

    def hello_job():
        print("Hello from worker!")
        return 42

    job_id = await worker.submit(hello_job)
    print(f"Submitted: {job_id}")

    status = await worker.wait(job_id)
    print(f"Completed with state: {status.state}")

    logs = await worker.logs(job_id)
    print("Logs:")
    for log in logs:
        print(f"  {log}")


async def example_with_ports(worker: WorkerContext):
    """Example with port allocation."""
    print("\n=== Example: Port Allocation ===\n")

    def port_job():
        import os

        http_port = os.environ["FLUSTER_PORT_HTTP"]
        print(f"HTTP port: {http_port}")

    job_id = await worker.submit(port_job, ports=["http"])
    status = await worker.status(job_id)
    print(f"Allocated ports: {dict(status.ports)}")

    await worker.wait(job_id)
    print("Job completed!")


async def example_concurrent(worker: WorkerContext):
    """Example with multiple concurrent jobs."""
    print("\n=== Example: Concurrent Jobs ===\n")

    def slow_job(n):
        import time

        for i in range(5):
            print(f"Job {n}: iteration {i}")
            time.sleep(1)

    # Submit 5 jobs
    job_ids = []
    for i in range(5):
        job_id = await worker.submit(slow_job, i)
        job_ids.append(job_id)
        print(f"Submitted job {i}: {job_id}")

    # Wait for all
    for job_id in job_ids:
        await worker.wait(job_id)
        print(f"Job {job_id} completed")


async def example_kill(worker: WorkerContext):
    """Example of killing a job."""
    print("\n=== Example: Killing a Job ===\n")

    def long_job():
        import time

        for i in range(60):
            print(f"Tick {i}")
            time.sleep(1)

    job_id = await worker.submit(long_job)
    print(f"Started long job: {job_id}")

    # Wait for it to actually reach RUNNING state
    print("Waiting for job to start running...")
    max_wait = 60  # seconds
    for _ in range(max_wait * 2):  # Poll every 0.5 seconds
        status = await worker.status(job_id)
        from fluster import cluster_pb2

        if status.state == cluster_pb2.JOB_STATE_RUNNING:
            print("Job is now running!")
            break
        if status.state in (
            cluster_pb2.JOB_STATE_SUCCEEDED,
            cluster_pb2.JOB_STATE_FAILED,
            cluster_pb2.JOB_STATE_KILLED,
        ):
            print(f"Job completed before we could kill it (state: {status.state})")
            return
        await asyncio.sleep(0.5)
    else:
        print("Job did not start running in time")
        return

    # Kill it
    print("Killing job...")
    await worker.kill(job_id)

    status = await worker.status(job_id)
    print(f"Final state: {status.state}")


async def main():
    """Run all examples with a single shared worker."""
    print("=" * 60)
    print("Fluster Worker Context Manager Examples")
    print("=" * 60)

    try:
        async with WorkerContext(max_concurrent_jobs=3) as worker:
            print(f"\nDashboard available at: {worker.url}")
            print("Press Ctrl+C to stop.\n")

            await example_basic(worker)
            await example_with_ports(worker)
            await example_concurrent(worker)
            await example_kill(worker)

            print("\n" + "=" * 60)
            print("All examples completed. Dashboard still available.")
            print("Press Ctrl+C to stop.")
            print("=" * 60)

            # Keep the worker alive so you can monitor the dashboard
            while True:
                await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("\nNote: These examples require Docker to be running.")
        print("Start Docker and try again.")


if __name__ == "__main__":
    asyncio.run(main())
