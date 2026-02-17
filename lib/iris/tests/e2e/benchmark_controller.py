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

import json
import logging
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import click
import psutil
from iris.client.client import IrisClient, Job, ResourceSpec
from iris.cluster.config import load_config, make_local_config
from iris.cluster.manager import connect_cluster
from iris.cluster.types import get_tpu_topology, tpu_device
from iris.rpc import cluster_pb2, config_pb2
from iris.rpc.cluster_connect import ControllerServiceClientSync

logger = logging.getLogger(__name__)

# Test root for relative imports
TEST_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class TpuJobSpec:
    """ResourceSpec + replica count for a TPU job.

    Mirrors fray v2's ResourceConfig.with_tpu() without depending on fray.
    Iris's ResourceSpec doesn't carry replicas (they're a submit-time concern),
    so we bundle them here.
    """

    resources: ResourceSpec
    replicas: int


def tpu_job_spec(
    tpu_type: str, *, slice_count: int = 1, cpu: int = 32, memory: str = "128g", disk: str = "50g"
) -> TpuJobSpec:
    """Build a TPU ResourceSpec and compute the replica count from topology."""
    topo = get_tpu_topology(tpu_type)
    replicas = slice_count * topo.vm_count
    resources = ResourceSpec(cpu=cpu, memory=memory, disk=disk, device=tpu_device(tpu_type))
    return TpuJobSpec(resources=resources, replicas=replicas)


@dataclass
class BenchmarkMetrics:
    """Performance metrics collected during benchmark."""

    num_jobs: int
    num_slices: int
    submission_time_seconds: float
    time_to_complete: float
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
    num_small = max(1, int(num_slices * 0.40))
    num_medium = max(1, int(num_slices * 0.32))
    num_large = max(1, int(num_slices * 0.20))
    num_xlarge = max(1, num_slices - num_small - num_medium - num_large)

    slice_configs = [
        ("v5litepod-4", 1, 4, num_small, "64", "64GB", "500GB"),
        ("v5litepod-8", 1, 8, num_medium, "96", "96GB", "1TB"),
        ("v5litepod-16", 4, 16, num_large, "128", "128GB", "1TB"),
        ("v5litepod-32", 8, 32, num_xlarge, "256", "128GB", "1TB"),
    ]

    for variant, num_vms, tpu_count, count, cpu, memory, disk in slice_configs:
        logger.info("Creating %d slices of %s", count, variant)
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


def _submit_job_mix(client: IrisClient, num_jobs: int, workspace: Path) -> tuple[list[Job], list[Job]]:
    """Submit a realistic mix of training jobs.

    Job distribution:
    - 60% small jobs (1-4 TPU cores)
    - 25% medium jobs (4-8 TPU cores)
    - 10% large jobs (16-32 TPU cores)
    - 5% misconfigured jobs (wrong TPU variant, should fail scheduling)

    Returns:
        (schedulable_jobs, unschedulable_jobs) tuple
    """
    from iris.cluster.types import Entrypoint, EnvironmentSpec

    num_small = int(num_jobs * 0.60)
    num_medium = int(num_jobs * 0.25)
    num_large = int(num_jobs * 0.10)
    num_bad = num_jobs - num_small - num_medium - num_large

    schedulable_jobs = []
    unschedulable_jobs = []

    job_configs = [
        ("small", tpu_job_spec("v5litepod-4"), num_small, True),
        ("medium", tpu_job_spec("v5litepod-8"), num_medium, True),
        ("large", tpu_job_spec("v5litepod-16"), num_large, True),
        ("bad", tpu_job_spec("v5p-8"), num_bad, False),
    ]

    job_idx = 0
    for name_prefix, spec, count, schedulable in job_configs:
        for _ in range(count):
            job = client.submit(
                entrypoint=Entrypoint.from_callable(_dummy_training_task),
                name=f"{name_prefix}-{job_idx:03d}",
                resources=spec.resources,
                replicas=spec.replicas,
                environment=EnvironmentSpec(),
            )

            if schedulable:
                schedulable_jobs.append(job)
            else:
                unschedulable_jobs.append(job)
            job_idx += 1

    return schedulable_jobs, unschedulable_jobs


def _wait_for_workers(controller_client: ControllerServiceClientSync, min_workers: int, timeout: float = 120.0) -> None:
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


@dataclass
class JobResult:
    """Result from waiting on a single job in a thread."""

    job: Job
    state_name: str
    elapsed: float
    error: Exception | None = None


def _wait_for_job(job: Job, timeout: float) -> JobResult:
    """Wait for a single job, streaming logs including children.

    Designed to be called from a thread pool. Each thread independently waits
    on its job and streams logs back through the logger.
    """
    start = time.monotonic()
    try:
        status = job.wait(
            timeout=timeout,
            poll_interval=2.0,
            raise_on_failure=False,
            stream_logs=True,
            include_children=True,
        )
        state_name = cluster_pb2.JobState.Name(status.state)
        return JobResult(job=job, state_name=state_name, elapsed=time.monotonic() - start)
    except Exception as e:
        return JobResult(job=job, state_name="UNKNOWN", elapsed=time.monotonic() - start, error=e)


def _wait_all_jobs_threaded(jobs: list[Job], timeout: float) -> list[JobResult]:
    """Wait for all jobs concurrently, one thread per job, streaming logs."""
    results: list[JobResult] = []

    with ThreadPoolExecutor(max_workers=min(len(jobs), 64)) as pool:
        futures = {pool.submit(_wait_for_job, job, timeout): job for job in jobs}
        for future in as_completed(futures):
            results.append(future.result())

    return results


def run_benchmark(num_jobs: int, num_slices: int) -> BenchmarkMetrics:
    """Run controller benchmark with specified configuration.

    Args:
        num_jobs: Number of training jobs to submit
        num_slices: Number of TPU slices to create

    Returns:
        BenchmarkMetrics with collected performance data
    """
    print("\n" + "=" * 70)
    print("Iris Controller Benchmark")
    print("=" * 70)
    print("Configuration:")
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
            logger.info(
                "Submitted %s schedulable + %s unschedulable jobs in %s seconds",
                len(schedulable_jobs),
                len(unschedulable_jobs),
                submission_time,
            )

            # Wait for schedulable jobs concurrently, streaming logs from each in its own thread.
            # Unschedulable ("bad") jobs are excluded since they'll never reach a terminal state
            # in a timely manner.
            print(f"Waiting for {len(schedulable_jobs)} schedulable jobs (streaming logs with children)...")
            wait_start = time.monotonic()
            results = _wait_all_jobs_threaded(schedulable_jobs, timeout=600.0)
            time_to_complete = time.monotonic() - wait_start

            for r in results:
                if r.error is not None:
                    logger.warning("Job %s errored during wait: %s", r.job.job_id, r.error)

            # Collect final metrics
            mem_after = controller_proc.memory_info().rss
            memory_delta_mb = (mem_after - mem_before) / (1024 * 1024)

            final_counts: dict[str, int] = defaultdict(int)
            for r in results:
                final_counts[r.state_name] += 1
            final_counts = dict(final_counts)

            metrics = BenchmarkMetrics(
                num_jobs=num_jobs,
                num_slices=num_slices,
                submission_time_seconds=submission_time,
                time_to_complete=time_to_complete,
                controller_memory_mb=memory_delta_mb,
                jobs_by_state=final_counts,
            )

            # Print results
            print("\n" + "=" * 70)
            print("Benchmark Results:")
            print("-" * 70)
            print(f"  Job submission time:       {metrics.submission_time_seconds:>10.2f}s")
            print(f"  Time to complete:          {metrics.time_to_complete:>10.2f}s")
            print(f"  Controller memory delta:   {metrics.controller_memory_mb:>10.1f} MB")
            print("\nFinal job states:")
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


def _print_profile_table(speedscope_path: Path, top_n: int = 30) -> None:
    """Parse a speedscope JSON file and print a text table of top functions by sample count."""
    with open(speedscope_path) as f:
        data = json.load(f)

    frames = data["shared"]["frames"]
    sample_counts: Counter[int] = Counter()

    for profile in data["profiles"]:
        for sample in profile.get("samples", []):
            for frame_idx in sample:
                sample_counts[frame_idx] += 1

    total_samples = sum(sample_counts.values())
    if total_samples == 0:
        print("No samples collected.")
        return

    print(f"\n{'=' * 90}")
    print(f"Profile Summary ({total_samples} total samples across {len(data['profiles'])} threads)")
    print(f"{'=' * 90}")
    print(f"{'Samples':>8}  {'%':>6}  {'Function':<40}  {'File'}")
    print(f"{'-' * 8}  {'-' * 6}  {'-' * 40}  {'-' * 30}")

    for frame_idx, count in sample_counts.most_common(top_n):
        frame = frames[frame_idx]
        name = frame.get("name", "???")
        file = frame.get("file", "???")
        # Shorten file paths for readability
        if "/site-packages/" in file:
            file = "..." + file.split("/site-packages/")[-1]
        elif "/lib/" in file:
            file = "..." + file.split("/lib/")[-1]
        pct = 100.0 * count / total_samples
        line = frame.get("line", "")
        loc = f"{file}:{line}" if line else file
        print(f"{count:>8}  {pct:>5.1f}%  {name:<40}  {loc}")

    print(f"{'=' * 90}\n")


@cli.command("benchmark")
@click.option("--num-jobs", type=int, default=100, help="Number of jobs to submit")
@click.option("--num-slices", type=int, default=25, help="Number of TPU slices to create")
@click.option("--profile", is_flag=True, help="Profile with py-spy (requires sudo)")
@click.option(
    "--profile-output",
    type=click.Path(path_type=Path),
    default=Path("/tmp/profiles"),
    help="Directory for profile output (default: /tmp/profiles/)",
)
def benchmark(
    num_jobs: int,
    num_slices: int,
    profile: bool = False,
    profile_output: Path | None = None,
) -> None:
    """Run controller benchmark."""
    if profile:
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
            "--gil",
            "--idle",
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
            _print_profile_table(speedscope_file)
            print(f"Speedscope profile saved to {speedscope_file}")
            print("To view: https://www.speedscope.app/")
        else:
            print(f"\npy-spy failed with return code {result.returncode}")

        return

    # Normal benchmark mode
    run_benchmark(num_jobs=num_jobs, num_slices=num_slices)


if __name__ == "__main__":
    from iris.logging import configure_logging

    configure_logging(level=logging.INFO)
    cli()
