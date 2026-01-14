#!/usr/bin/env python3
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

"""Simple example demonstrating the unified Controller class.

This example shows how to:
1. Initialize a Controller with configuration and callbacks
2. Start the controller and all its components
3. Register workers and submit jobs
4. Query job status
5. Cleanly shut down the controller

The Controller class encapsulates all controller components (state, scheduler,
heartbeat monitor, service, dashboard) and manages their lifecycle automatically.
"""

import time
from pathlib import Path

from fluster import cluster_pb2
from fluster.cluster.controller import Controller, ControllerConfig
from fluster.cluster.controller.state import ControllerJob, ControllerWorker
from fluster.cluster.types import JobId, WorkerId


def main():
    """Run the simple controller example."""
    print("=== Simple Controller Example ===\n")

    # Define callbacks for the controller
    def dispatch_job(job: ControllerJob, worker: ControllerWorker) -> bool:
        """Dispatch a job to a worker (mock implementation)."""
        print(f"[dispatch] Job {job.job_id[:8]} -> Worker {worker.worker_id}")
        # In a real implementation, this would send an RPC to the worker
        return True

    def send_heartbeat(address: str) -> cluster_pb2.HeartbeatResponse | None:
        """Send heartbeat to a worker (mock implementation)."""
        print(f"[heartbeat] Checking worker at {address}")
        # In a real implementation, this would send an RPC to the worker
        return cluster_pb2.HeartbeatResponse(jobs=[], timestamp_ms=int(time.time() * 1000))

    def on_worker_failed(worker_id: WorkerId, job_ids: list[JobId]) -> None:
        """Handle worker failure."""
        print(f"[failure] Worker {worker_id} failed with {len(job_ids)} jobs")

    # Create controller configuration
    print("Creating controller configuration...")
    config = ControllerConfig(
        host="127.0.0.1",
        port=8080,
        bundle_dir=Path("/tmp/controller_bundles"),
        scheduler_interval_seconds=0.5,
        heartbeat_interval_seconds=2.0,
    )

    # Create controller
    print("Creating controller...")
    controller = Controller(
        config=config,
        dispatch_fn=dispatch_job,
        heartbeat_fn=send_heartbeat,
        on_worker_failed=on_worker_failed,
    )

    # Start controller
    print("Starting controller...")
    controller.start()
    print(f"Controller started at {controller.url}\n")

    try:
        # Register a worker
        print("Registering worker...")
        worker_request = cluster_pb2.RegisterWorkerRequest(
            worker_id="worker-1",
            address="http://localhost:8081",
            resources=cluster_pb2.ResourceSpec(cpu=4, memory="8g"),
        )
        worker_response = controller.register_worker(worker_request)
        print(f"Worker registered: {worker_response.accepted}\n")

        # Submit a job
        print("Submitting job...")
        job_request = cluster_pb2.LaunchJobRequest(
            name="example-job",
            serialized_entrypoint=b"example",
            resources=cluster_pb2.ResourceSpec(cpu=1, memory="1g"),
            environment=cluster_pb2.EnvironmentConfig(workspace="/tmp"),
        )
        job_response = controller.launch_job(job_request)
        job_id = job_response.job_id
        print(f"Job submitted: {job_id[:8]}...\n")

        # Wait a bit for scheduling
        time.sleep(2.0)

        # Query job status
        print("Querying job status...")
        status_response = controller.get_job_status(job_id)
        job_status = status_response.job
        print(f"Job {job_id[:8]}... state: {job_status.state}")
        print(f"Job assigned to worker: {job_status.worker_id or 'None'}\n")

        # Access controller state for advanced usage
        print("Controller state summary:")
        all_jobs = controller.state.list_all_jobs()
        all_workers = controller.state.list_all_workers()
        print(f"  Total jobs: {len(all_jobs)}")
        print(f"  Total workers: {len(all_workers)}")
        print(f"  Recent actions: {len(controller.state.get_recent_actions())}\n")

        print("Dashboard is accessible at:")
        print(f"  {controller.url}/")
        print(f"  {controller.url}/health")
        print(f"  {controller.url}/api/stats\n")

        print("Controller is running. Press Ctrl+C to stop...")
        time.sleep(10)

    finally:
        # Clean shutdown
        print("\nStopping controller...")
        controller.stop()
        print("Controller stopped.")


if __name__ == "__main__":
    main()
