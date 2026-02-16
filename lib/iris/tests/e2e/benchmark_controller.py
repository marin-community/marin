#!/usr/bin/env python
# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark Iris controller performance under realistic load.

Simulates a cluster with 25 TPU slices of varying sizes and 100 training jobs,
measuring scheduler performance, job scheduling latency, and resource utilization.

Usage:
    uv run python lib/iris/tests/e2e/benchmark_controller.py
    uv run python lib/iris/tests/e2e/benchmark_controller.py --num-jobs 200 --num-slices 50
    uv run python lib/iris/tests/e2e/benchmark_controller.py --profile --profile-output ./profiles

This benchmark helps detect performance regressions like #2802 (SSL context overhead).
"""

import logging
import os
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import click
import psutil
import pytest
from iris.client.client import IrisClient, Job
from iris.cluster.config import load_config, make_local_config
from iris.cluster.manager import connect_cluster
from iris.rpc import cluster_pb2, config_pb2
from iris.rpc.cluster_connect import ControllerServiceClientSync

logger = logging.getLogger(__name__)

# Test root for relative imports
TEST_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class BenchmarkMetrics:
    """Performance metrics collected during benchmark."""

    num_jobs: int
    num_slices: int
    submission_time_seconds: float
    time_to_90_percent_running: float
    time_to_all_complete: float
    controller_memory_mb: float
    jobs_by_state: dict[str, int]


def _make_benchmark_config(num_slices: int) -> config_pb2.IrisClusterConfig:
    """Build a local cluster config with diverse TPU scale groups.

    Creates a realistic mix of TPU slice sizes to stress-test the scheduler:
    - 40% small slices (v5litepod-4: 2x2 cores)
    - 32% medium slices (v5litepod-8: 2x4 cores)
    - 20% large slices (v5litepod-16: 4x4 cores)
    - 8% xlarge slices (v5litepod-32: 4x8 cores)

    Args:
        num_slices: Total number of slices to create across all groups
    """
    config = load_config(TEST_ROOT / "examples" / "demo.yaml")
    config.scale_groups.clear()

    # Distribute slices across size categories
    num_small = int(num_slices * 0.40)
    num_medium = int(num_slices * 0.32)
    num_large = int(num_slices * 0.20)
    num_xlarge = num_slices - num_small - num_medium - num_large

    slice_configs = [
        ("v5litepod-4", 1, 4, num_small, "64", "64GB", "500GB"),
        ("v5litepod-8", 1, 8, num_medium, "96", "96GB", "1TB"),
        ("v5litepod-16", 4, 16, num_large, "128", "128GB", "1TB"),
        ("v5litepod-32", 4, 32, num_xlarge, "256", "256GB", "2TB"),
    ]

    for variant, num_vms, tpu_count, count, cpu, memory, disk in slice_configs:
        if count == 0:
            continue

        sg_name = f"tpu-{variant}"
        sg = config.scale_groups[sg_name]
        sg.name = sg_name
        sg.accelerator_type = config_pb2.ACCELERATOR_TYPE_TPU
        sg.accelerator_variant = variant
        sg.num_vms = num_vms
        sg.min_slices = count
        sg.max_slices = count
        sg.resources.cpu = int(cpu)
        sg.resources.memory_bytes = _parse_size(memory)
        sg.resources.disk_bytes = _parse_size(disk)
        sg.resources.tpu_count = tpu_count
        sg.slice_template.preemptible = True
        sg.slice_template.num_vms = num_vms
        sg.slice_template.accelerator_type = config_pb2.ACCELERATOR_TYPE_TPU
        sg.slice_template.accelerator_variant = variant
        sg.slice_template.local.SetInParent()

    return make_local_config(config)


def _parse_size(size_str: str) -> int:
    """Parse human-readable size string to bytes."""
    import humanfriendly

    return humanfriendly.parse_size(size_str)


def _dummy_training_task():
    """Simulate a training task that runs briefly."""
    time.sleep(0.1)
    return "done"


def _submit_job_mix(
    client: IrisClient, num_jobs: int, workspace: Path
) -> tuple[list[Job], list[Job]]:
    """Submit a realistic mix of training jobs.

    Job distribution:
    - 60% small jobs (1-4 TPU cores)
    - 25% medium jobs (4-8 TPU cores)
    - 10% large jobs (16-32 TPU cores)
    - 5% misconfigured jobs (wrong TPU variant, should fail scheduling)

    Returns:
        (schedulable_jobs, unschedulable_jobs) tuple
    """
    from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec

    num_small = int(num_jobs * 0.60)
    num_medium = int(num_jobs * 0.25)
    num_large = int(num_jobs * 0.10)
    num_bad = num_jobs - num_small - num_medium - num_large

    schedulable_jobs = []
    unschedulable_jobs = []

    job_configs = [
        # (name_prefix, cpu, memory, tpu_variant, tpu_count, count, schedulable)
        ("small", 2, "4g", "v5litepod-4", 2, num_small, True),
        ("medium", 4, "8g", "v5litepod-8", 4, num_medium, True),
        ("large", 8, "16g", "v5litepod-16", 8, num_large, True),
        ("bad", 4, "8g", "v5p-8", 4, num_bad, False),  # v5p doesn't exist
    ]

    job_idx = 0
    for name_prefix, cpu, memory, tpu_variant, tpu_count, count, schedulable in job_configs:
        for i in range(count):
            device = cluster_pb2.DeviceConfig()
            device.tpu.variant = tpu_variant
            device.tpu.count = tpu_count

            job = client.submit(
                entrypoint=Entrypoint.from_callable(_dummy_training_task),
                name=f"{name_prefix}-{job_idx:03d}",
                resources=ResourceSpec(cpu=cpu, memory=memory, device=device),
                environment=EnvironmentSpec(),
            )

            if schedulable:
                schedulable_jobs.append(job)
            else:
                unschedulable_jobs.append(job)
            job_idx += 1

    return schedulable_jobs, unschedulable_jobs


def _wait_for_workers(
    controller_client: ControllerServiceClientSync, min_workers: int, timeout: float = 120.0
) -> None:
    """Wait for workers to register with the controller."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        request = cluster_pb2.Controller.ListWorkersRequest()
        response = controller_client.list_workers(request)
        healthy = [w for w in response.workers if w.healthy]
        if len(healthy) >= min_workers:
            logger.info(f"Cluster ready: {len(healthy)} healthy workers registered")
            return
        time.sleep(1.0)
    raise TimeoutError(f"Only {len(healthy)} of {min_workers} workers registered in {timeout}s")


def _get_job_state_counts(
    controller_client: ControllerServiceClientSync, jobs: list[Job]
) -> dict[str, int]:
    """Count jobs in each state."""
    counts: dict[str, int] = defaultdict(int)
    for job in jobs:
        request = cluster_pb2.Controller.GetJobStatusRequest(job_id=job.job_id.to_wire())
        response = controller_client.get_job_status(request)
        state_name = cluster_pb2.JobState.Name(response.job.state)
        counts[state_name] += 1
    return dict(counts)


def _wait_for_running_fraction(
    controller_client: ControllerServiceClientSync,
    jobs: list[Job],
    target_fraction: float,
    timeout: float,
) -> float:
    """Wait until target_fraction of jobs are RUNNING or SUCCEEDED.

    Returns elapsed time in seconds.
    """
    start = time.monotonic()
    deadline = start + timeout
    target_count = int(len(jobs) * target_fraction)

    while time.monotonic() < deadline:
        counts = _get_job_state_counts(controller_client, jobs)
        running_or_done = counts.get("JOB_STATE_RUNNING", 0) + counts.get("JOB_STATE_SUCCEEDED", 0)

        if running_or_done >= target_count:
            elapsed = time.monotonic() - start
            logger.info(f"Reached {running_or_done}/{len(jobs)} running/succeeded in {elapsed:.2f}s")
            return elapsed

        time.sleep(0.5)

    raise TimeoutError(
        f"{target_fraction * 100:.0f}% of jobs did not reach RUNNING/SUCCEEDED in {timeout}s"
    )


def _wait_for_all_complete(
    controller_client: ControllerServiceClientSync, jobs: list[Job], timeout: float
) -> float:
    """Wait until all jobs reach terminal state (SUCCEEDED/FAILED).

    Returns elapsed time in seconds.
    """
    start = time.monotonic()
    deadline = start + timeout

    while time.monotonic() < deadline:
        counts = _get_job_state_counts(controller_client, jobs)
        terminal = counts.get("JOB_STATE_SUCCEEDED", 0) + counts.get("JOB_STATE_FAILED", 0)

        if terminal >= len(jobs):
            elapsed = time.monotonic() - start
            logger.info(f"All {len(jobs)} jobs completed in {elapsed:.2f}s")
            return elapsed

        time.sleep(1.0)

    raise TimeoutError(f"Not all jobs completed in {timeout}s")


def run_benchmark(
    num_jobs: int = 100,
    num_slices: int = 25,
) -> BenchmarkMetrics:
    """Run controller benchmark with specified configuration.

    Args:
        num_jobs: Number of training jobs to submit
        num_slices: Number of TPU slices to create

    Returns:
        BenchmarkMetrics with collected performance data
    """
    print("\n" + "=" * 70)
    print(f"Iris Controller Benchmark")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Jobs: {num_jobs}")
    print(f"  Slices: {num_slices}")
    print("=" * 70 + "\n")

    # Create cluster config
    config = _make_benchmark_config(num_slices)

    # Start cluster
    print("Starting local cluster...")
    with connect_cluster(config) as url:
        client = IrisClient.remote(url, workspace=TEST_ROOT)
        controller_client = ControllerServiceClientSync(address=url, timeout_ms=30000)

        try:
            # Wait for workers
            print(f"Waiting for {num_slices} workers to register...")
            _wait_for_workers(controller_client, num_slices, timeout=120.0)

            # Get controller process for memory tracking
            controller_proc = psutil.Process(os.getpid())
            mem_before = controller_proc.memory_info().rss

            # Submit jobs
            print(f"Submitting {num_jobs} jobs...")
            submit_start = time.time()
            schedulable_jobs, unschedulable_jobs = _submit_job_mix(client, num_jobs, TEST_ROOT)
            submission_time = time.time() - submit_start
            print(f"Submitted {len(schedulable_jobs)} schedulable + {len(unschedulable_jobs)} unschedulable jobs in {submission_time:.2f}s")

            # Wait for 90% to be running
            print("Waiting for 90% of jobs to reach RUNNING...")
            time_to_90_running = _wait_for_running_fraction(
                controller_client, schedulable_jobs, target_fraction=0.90, timeout=300.0
            )

            # Wait for all to complete
            print("Waiting for all jobs to complete...")
            time_to_complete = _wait_for_all_complete(controller_client, schedulable_jobs, timeout=600.0)

            # Collect final metrics
            mem_after = controller_proc.memory_info().rss
            memory_delta_mb = (mem_after - mem_before) / (1024 * 1024)

            final_counts = _get_job_state_counts(controller_client, schedulable_jobs + unschedulable_jobs)

            metrics = BenchmarkMetrics(
                num_jobs=num_jobs,
                num_slices=num_slices,
                submission_time_seconds=submission_time,
                time_to_90_percent_running=time_to_90_running,
                time_to_all_complete=time_to_complete,
                controller_memory_mb=memory_delta_mb,
                jobs_by_state=final_counts,
            )

            # Print results
            print("\n" + "=" * 70)
            print("Benchmark Results:")
            print("-" * 70)
            print(f"  Job submission time:       {metrics.submission_time_seconds:>10.2f}s")
            print(f"  Time to 90% running:       {metrics.time_to_90_percent_running:>10.2f}s")
            print(f"  Time to all complete:      {metrics.time_to_all_complete:>10.2f}s")
            print(f"  Controller memory delta:   {metrics.controller_memory_mb:>10.1f} MB")
            print(f"\nFinal job states:")
            for state, count in sorted(metrics.jobs_by_state.items()):
                print(f"  {state:<30} {count:>5}")
            print("=" * 70 + "\n")

            return metrics

        finally:
            controller_client.close()


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Benchmark Iris controller performance."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(benchmark)


@cli.command("benchmark")
@click.option("--num-jobs", type=int, default=100, help="Number of jobs to submit")
@click.option("--num-slices", type=int, default=25, help="Number of TPU slices to create")
@click.option("--profile", is_flag=True, help="Profile controller with py-spy")
@click.option(
    "--profile-output",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory for profile output (default: lib/iris/tests/e2e/profiles/)",
)
def benchmark(
    num_jobs: int = 100,
    num_slices: int = 25,
    profile: bool = False,
    profile_output: Path | None = None,
) -> None:
    """Run controller benchmark."""
    if profile:
        if profile_output is None:
            profile_output = Path(__file__).parent / "profiles"

        print(f"\nProfiling enabled: output will be saved to {profile_output}")
        print("Note: py-spy requires sudo permissions\n")

        profile_output.mkdir(parents=True, exist_ok=True)
        speedscope_file = profile_output / "controller_benchmark.speedscope"

        pyspy_cmd = [
            "sudo",
            "py-spy",
            "record",
            "--format",
            "speedscope",
            "--output",
            str(speedscope_file),
            "--rate",
            "100",
            "--subprocesses",
            "--",
            sys.executable,
            __file__,
            "benchmark",
            "--num-jobs",
            str(num_jobs),
            "--num-slices",
            str(num_slices),
        ]

        print(f"Running: {' '.join(pyspy_cmd)}\n")
        result = subprocess.run(pyspy_cmd)

        if result.returncode == 0:
            print(f"\nSpeedscope profile saved to {speedscope_file}")
            print("\nTo view:")
            print("  1. Visit https://www.speedscope.app/")
            print(f"  2. Upload {speedscope_file}")
        else:
            print(f"\npy-spy failed with return code {result.returncode}")

        return

    # Normal benchmark mode
    run_benchmark(num_jobs=num_jobs, num_slices=num_slices)


# Pytest integration
pytestmark = pytest.mark.slow


@pytest.mark.slow
def test_controller_benchmark():
    """Benchmark test for Iris controller performance.

    Marked as 'slow' to exclude from default CI runs.
    Run with: uv run pytest lib/iris/tests/e2e/benchmark_controller.py -m slow
    """
    metrics = run_benchmark(num_jobs=100, num_slices=25)

    # Sanity checks
    assert metrics.submission_time_seconds < 30, "Job submission should take < 30s"
    assert metrics.time_to_90_percent_running < 120, "90% of jobs should be running within 2 minutes"
    assert metrics.time_to_all_complete < 300, "All jobs should complete within 5 minutes"
    assert metrics.controller_memory_mb < 500, "Controller memory delta should be < 500MB"

    # Verify job states
    assert metrics.jobs_by_state.get("JOB_STATE_SUCCEEDED", 0) >= 90, "At least 90 jobs should succeed"
    assert metrics.jobs_by_state.get("JOB_STATE_UNSCHEDULABLE", 0) >= 1, "Some misconfigured jobs should be unschedulable"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cli()
