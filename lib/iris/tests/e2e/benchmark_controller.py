#!/usr/bin/env python
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark Iris controller performance under realistic load.

Two benchmark modes:

1. ``benchmark`` (default): Simulates a cluster with 25 TPU slices of varying
   sizes and 100 training jobs, measuring scheduler performance, job scheduling
   latency, and resource utilization.

2. ``single-worker``: Submits many jobs to a single CPU worker as fast as
   possible while streaming logs from every job.  This exercises the controller
   hot-path that was overwhelmed in #3062 (125+ simultaneous task pods on one
   worker caused DEADLINE_EXCEEDED RPC timeouts).

Usage:
    uv run python lib/iris/tests/e2e/benchmark_controller.py
    uv run python lib/iris/tests/e2e/benchmark_controller.py --num-jobs 200 --num-slices 50
    uv run python lib/iris/tests/e2e/benchmark_controller.py --profile --profile-output ./profiles
    uv run python lib/iris/tests/e2e/benchmark_controller.py single-worker --num-jobs 100

This benchmark helps detect performance regressions like #2802 (SSL context overhead)
and #3062 (single-worker burst overwhelms controller).
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
import httpx
import humanfriendly
import psutil
from iris.client.client import IrisClient, Job, ResourceSpec
from iris.cluster.config import connect_cluster, load_config, make_local_config
from iris.cluster.types import Entrypoint, EnvironmentSpec, get_tpu_topology, tpu_device
from iris.logging import configure_logging
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
    - 40% small slices (v5litepod-4: 1x4-chip VM)
    - 32% medium slices (v5litepod-8: 1x8-chip VM)
    - 20% large slices (v5litepod-16: 4x4-chip VMs)
    - 8% xlarge slices (v5litepod-32: 8x4-chip VMs)

    Args:
        num_slices: Total number of slices to create across all groups
    """
    config = load_config(TEST_ROOT / "examples" / "test.yaml")
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
        sg.num_vms = num_vms
        sg.min_slices = count
        sg.max_slices = count
        sg.resources.cpu_millicores = int(float(cpu) * 1000)
        sg.resources.memory_bytes = _parse_size(memory)
        sg.resources.disk_bytes = _parse_size(disk)
        sg.resources.device_count = tpu_count
        sg.resources.device_type = config_pb2.ACCELERATOR_TYPE_TPU
        sg.resources.device_variant = variant
        sg.slice_template.preemptible = True
        sg.slice_template.num_vms = num_vms
        sg.slice_template.accelerator_type = config_pb2.ACCELERATOR_TYPE_TPU
        sg.slice_template.accelerator_variant = variant
        sg.slice_template.local.SetInParent()

    return make_local_config(config)


def _parse_size(size_str: str) -> int:
    """Parse human-readable size string to bytes."""
    return humanfriendly.parse_size(size_str)


def _dummy_training_task():
    """Simulate a training task that runs briefly."""
    time.sleep(10.0)
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


def _make_single_worker_config() -> config_pb2.IrisClusterConfig:
    """Build a local cluster config with a single CPU worker.

    This deliberately funnels all work onto one worker so we can observe how the
    controller handles a burst of task creation, scheduling, log streaming, and
    completion RPCs from a single source — the scenario that triggered #3062.
    """
    config = load_config(TEST_ROOT / "examples" / "test.yaml")
    config.scale_groups.clear()

    sg = config.scale_groups["local-cpu"]
    sg.name = "local-cpu"
    sg.num_vms = 1
    sg.min_slices = 1
    sg.max_slices = 1
    sg.resources.cpu_millicores = 128 * 1000
    sg.resources.memory_bytes = 256 * 1024**3
    sg.resources.disk_bytes = 500 * 1024**3
    sg.resources.device_type = config_pb2.ACCELERATOR_TYPE_CPU
    sg.slice_template.local.SetInParent()

    return make_local_config(config)


def _cpu_burn_task(seconds: float = 1.0):
    """CPU-bound task that burns cycles in a tight loop.

    Generates enough work to keep the subprocess busy for approximately
    *seconds*, giving the controller realistic scheduling and log-streaming
    pressure.
    """
    print(f"burning cpu for {seconds}s")
    deadline = time.monotonic() + seconds
    total = 0
    while time.monotonic() < deadline:
        for _ in range(10_000):
            total += 1
    print(f"done ({total} iterations)")
    return total


def run_single_worker_benchmark(num_jobs: int) -> BenchmarkMetrics:
    """Submit *num_jobs* to one worker as fast as possible, streaming logs.

    Every job is waited on concurrently with ``stream_logs=True`` and
    ``include_children=True``, mirroring how Marin's ferry driver monitors a
    batch of download tasks.  The goal is to stress the controller's RPC
    handling and task-dispatch path when a single worker is hit with many
    concurrent requests.
    """
    print("\n" + "=" * 70)
    print("Iris Controller Benchmark — single-worker burst")
    print("=" * 70)
    print("Configuration:")
    print(f"  Jobs:    {num_jobs}")
    print("  Workers: 1  (all jobs target the same worker)")
    print("  Log streaming: ON")
    print("=" * 70 + "\n")

    config = _make_single_worker_config()

    print("Starting local cluster with 1 CPU worker...")
    with connect_cluster(config) as url:
        client = IrisClient.remote(url, workspace=TEST_ROOT)
        controller_client = ControllerServiceClientSync(address=url, timeout_ms=30000)

        try:
            print("Waiting for worker to register...")
            _wait_for_workers(controller_client, 1, timeout=60.0)

            controller_proc = psutil.Process(os.getpid())
            mem_before = controller_proc.memory_info().rss

            # Submit all jobs as fast as possible — this is the burst.
            print(f"Submitting {num_jobs} jobs...")
            submit_start = time.time()
            jobs: list[Job] = []
            for i in range(num_jobs):
                job = client.submit(
                    entrypoint=Entrypoint.from_callable(_cpu_burn_task, 1.0),
                    name=f"burst-{i:04d}",
                    resources=ResourceSpec(cpu=0, memory="64m"),
                    environment=EnvironmentSpec(),
                )
                jobs.append(job)
            submission_time = time.time() - submit_start
            logger.info("Submitted %d jobs in %.3fs", num_jobs, submission_time)

            # Wait for every job with log streaming enabled, concurrently.
            print(f"Waiting for {num_jobs} jobs (streaming logs)...")
            wait_start = time.monotonic()
            results = _wait_all_jobs_threaded(jobs, timeout=300.0)
            time_to_complete = time.monotonic() - wait_start

            for r in results:
                if r.error is not None:
                    logger.warning("Job %s errored during wait: %s", r.job.job_id, r.error)

            mem_after = controller_proc.memory_info().rss
            memory_delta_mb = (mem_after - mem_before) / (1024 * 1024)

            final_counts: dict[str, int] = defaultdict(int)
            for r in results:
                final_counts[r.state_name] += 1
            final_counts = dict(final_counts)

            metrics = BenchmarkMetrics(
                num_jobs=num_jobs,
                num_slices=1,
                submission_time_seconds=submission_time,
                time_to_complete=time_to_complete,
                controller_memory_mb=memory_delta_mb,
                jobs_by_state=final_counts,
            )

            print("\n" + "=" * 70)
            print("Benchmark Results (single-worker burst):")
            print("-" * 70)
            print(f"  Job submission time:       {metrics.submission_time_seconds:>10.2f}s")
            print(f"  Time to complete:          {metrics.time_to_complete:>10.2f}s")
            print(f"  Controller memory delta:   {metrics.controller_memory_mb:>10.1f} MB")
            print(f"  Throughput:                {num_jobs / metrics.time_to_complete:>10.1f} jobs/s")
            print("\nFinal job states:")
            for state, count in sorted(metrics.jobs_by_state.items()):
                print(f"  {state:<30} {count:>5}")
            print("=" * 70 + "\n")

            return metrics

        finally:
            controller_client.close()


@click.group()
def cli() -> None:
    """Benchmark Iris controller performance."""
    pass


def _run_with_pyspy(
    subcommand: str,
    cli_args: list[str],
    profile_output: Path,
    speedscope_name: str,
) -> None:
    """Re-launch this script under py-spy for CPU profiling.

    Constructs a py-spy command that invokes ``__file__ <subcommand> <cli_args>``
    and writes the speedscope profile to *profile_output / speedscope_name*.
    """
    print(f"\nProfiling enabled: output will be saved to {profile_output}")
    print("Note: py-spy requires sudo permissions\n")

    profile_output.mkdir(parents=True, exist_ok=True)
    speedscope_file = profile_output / speedscope_name

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
        subcommand,
        *cli_args,
    ]

    print(f"Running: {' '.join(pyspy_cmd)}\n")
    result = subprocess.run(pyspy_cmd)

    if result.returncode == 0:
        _print_profile_table(speedscope_file)
        print(f"Speedscope profile saved to {speedscope_file}")
        print("To view: https://www.speedscope.app/")
    else:
        print(f"\npy-spy failed with return code {result.returncode}")


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
def multi_tpu(
    num_jobs: int,
    num_slices: int,
    profile: bool = False,
    profile_output: Path | None = None,
) -> None:
    """Run controller benchmark."""
    if profile:
        _run_with_pyspy(
            "multi_tpu",
            ["--num-jobs", str(num_jobs), "--num-slices", str(num_slices)],
            profile_output,
            "controller_benchmark.speedscope",
        )
        return

    run_benchmark(num_jobs=num_jobs, num_slices=num_slices)


def _time_rpc(label: str, fn, iterations: int = 100) -> float:
    """Time an RPC over many iterations, return p50 latency in ms."""
    latencies = []
    for _ in range(iterations):
        t0 = time.monotonic()
        fn()
        latencies.append((time.monotonic() - t0) * 1000)
    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p99 = latencies[int(len(latencies) * 0.99)]
    total = sum(latencies)
    print(f"  {label:<40}  p50={p50:>8.2f}ms  p99={p99:>8.2f}ms  total={total:>8.0f}ms  ({iterations} iters)")
    return p50


def run_rpc_stress_benchmark(num_jobs: int, tasks_per_job: int) -> None:
    """Stress-test controller RPC hot paths with many jobs, tasks, and endpoints.

    Simulates the load pattern from 7 Zephyr jobs x512 tasks: many endpoints
    registered via service discovery, frequent GetJobStatus and ListTasks calls,
    and endpoint lookups (exact + prefix). Measures RPC latency to detect
    regressions in lock contention and index performance.
    """
    total_tasks = num_jobs * tasks_per_job

    print("\n" + "=" * 70)
    print("Iris Controller Benchmark — RPC stress")
    print("=" * 70)
    print(f"  Jobs:           {num_jobs}")
    print(f"  Tasks per job:  {tasks_per_job}")
    print(f"  Total tasks:    {total_tasks}")
    print(f"  Endpoints:      {total_tasks} (1 per task)")
    print("=" * 70 + "\n")

    config = _make_single_worker_config()

    print("Starting local cluster...")
    with connect_cluster(config) as url:
        client = IrisClient.remote(url, workspace=TEST_ROOT)
        controller_client = ControllerServiceClientSync(address=url, timeout_ms=30000)

        try:
            print("Waiting for worker...")
            _wait_for_workers(controller_client, 1, timeout=60.0)

            # Submit jobs with many replicas to create many tasks.
            print(f"Submitting {num_jobs} jobs with {tasks_per_job} replicas each...")
            submit_start = time.monotonic()
            jobs: list[Job] = []
            for i in range(num_jobs):
                job = client.submit(
                    entrypoint=Entrypoint.from_callable(_dummy_training_task),
                    name=f"stress-{i:04d}",
                    resources=ResourceSpec(cpu=0, memory="64m"),
                    replicas=tasks_per_job,
                    environment=EnvironmentSpec(),
                )
                jobs.append(job)
            submission_time = time.monotonic() - submit_start
            print(f"  Submitted in {submission_time:.2f}s")

            # Register endpoints — one per task, mimicking Fray actor registration.
            print(f"Registering {total_tasks} endpoints...")
            reg_start = time.monotonic()
            endpoint_names: list[str] = []
            for job in jobs:
                for t in range(tasks_per_job):
                    ep_name = f"{job.job_id}/worker-{t}"
                    endpoint_names.append(ep_name)
                    controller_client.register_endpoint(
                        cluster_pb2.Controller.RegisterEndpointRequest(
                            name=ep_name,
                            address=f"10.0.0.{t % 256}:{10000 + t}",
                            job_id=job.job_id.to_wire(),
                        )
                    )
            reg_time = time.monotonic() - reg_start
            print(
                f"  Registered {len(endpoint_names)} endpoints in {reg_time:.2f}s "
                f"({len(endpoint_names) / reg_time:.0f} eps/s)"
            )

            # Benchmark RPCs
            print(f"\nRPC latencies ({total_tasks} tasks, {len(endpoint_names)} endpoints):")
            print("-" * 90)

            iters = 200

            # Exact endpoint lookup
            sample_name = endpoint_names[len(endpoint_names) // 2]
            _time_rpc(
                "ListEndpoints (exact)",
                lambda: controller_client.list_endpoints(
                    cluster_pb2.Controller.ListEndpointsRequest(prefix=sample_name, exact=True)
                ),
                iters,
            )

            # Prefix endpoint listing (all endpoints for one job)
            sample_prefix = f"{jobs[0].job_id}/"
            _time_rpc(
                f"ListEndpoints (prefix, ~{tasks_per_job} results)",
                lambda: controller_client.list_endpoints(
                    cluster_pb2.Controller.ListEndpointsRequest(prefix=sample_prefix)
                ),
                iters,
            )

            # GetJobStatus for a large job
            _time_rpc(
                f"GetJobStatus ({tasks_per_job} tasks)",
                lambda: controller_client.get_job_status(
                    cluster_pb2.Controller.GetJobStatusRequest(job_id=jobs[0].job_id.to_wire())
                ),
                iters,
            )

            # ListTasks for one job
            _time_rpc(
                f"ListTasks (job filter, {tasks_per_job} tasks)",
                lambda: controller_client.list_tasks(
                    cluster_pb2.Controller.ListTasksRequest(job_id=jobs[0].job_id.to_wire())
                ),
                iters,
            )

            # ListTasks unfiltered (all tasks)
            _time_rpc(
                f"ListTasks (all, {total_tasks} tasks)",
                lambda: controller_client.list_tasks(cluster_pb2.Controller.ListTasksRequest()),
                iters,
            )

            # ListWorkers
            _time_rpc(
                "ListWorkers",
                lambda: controller_client.list_workers(cluster_pb2.Controller.ListWorkersRequest()),
                iters,
            )

            # Health check
            health_client = httpx.Client(base_url=url)
            _time_rpc(
                "GET /health",
                lambda: health_client.get("/health"),
                iters,
            )
            health_client.close()

            print("-" * 90)
            print()

        finally:
            controller_client.close()


@cli.command("rpc-stress")
@click.option("--num-jobs", type=int, default=7, help="Number of jobs to submit")
@click.option("--tasks-per-job", type=int, default=512, help="Number of task replicas per job")
@click.option("--profile", is_flag=True, help="Profile with py-spy (requires sudo)")
@click.option(
    "--profile-output",
    type=click.Path(path_type=Path),
    default=Path("/tmp/profiles"),
    help="Directory for profile output (default: /tmp/profiles/)",
)
def rpc_stress(
    num_jobs: int,
    tasks_per_job: int,
    profile: bool = False,
    profile_output: Path | None = None,
) -> None:
    """Stress-test RPC hot paths with many jobs, tasks, and endpoints.

    Simulates the Zephyr pattern of 7 jobs x512 tasks with endpoint
    registration, measuring lookup/listing RPC latency under load.
    """
    if profile:
        _run_with_pyspy(
            "rpc-stress",
            ["--num-jobs", str(num_jobs), "--tasks-per-job", str(tasks_per_job)],
            profile_output,
            "rpc_stress_benchmark.speedscope",
        )
        return

    run_rpc_stress_benchmark(num_jobs=num_jobs, tasks_per_job=tasks_per_job)


@cli.command("single-worker")
@click.option("--num-jobs", type=int, default=100, help="Number of jobs to burst-submit to a single worker")
@click.option("--profile", is_flag=True, help="Profile with py-spy (requires sudo)")
@click.option(
    "--profile-output",
    type=click.Path(path_type=Path),
    default=Path("/tmp/profiles"),
    help="Directory for profile output (default: /tmp/profiles/)",
)
def single_worker(
    num_jobs: int,
    profile: bool = False,
    profile_output: Path | None = None,
) -> None:
    """Burst-submit jobs to a single worker with log streaming.

    Exercises the controller hot-path from #3062: many concurrent task
    creations, log-stream RPCs, and completions funneled through one worker.
    """
    if profile:
        _run_with_pyspy(
            "single-worker",
            ["--num-jobs", str(num_jobs)],
            profile_output,
            "single_worker_benchmark.speedscope",
        )
        return

    run_single_worker_benchmark(num_jobs=num_jobs)


if __name__ == "__main__":

    configure_logging(level=logging.INFO)
    cli()
