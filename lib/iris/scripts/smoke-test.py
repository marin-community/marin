#!/usr/bin/env python3
# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris cluster autoscaling smoke test.

This script provides end-to-end validation of an Iris cluster by exercising
the `iris` CLI for cluster lifecycle and `IrisClient` for job submission:

1. Cleans up existing resources (`iris cluster stop`)
2. Starts the cluster (`iris cluster start`)
3. Establishes connection (`iris cluster dashboard` or local address)
4. Submits TPU jobs to exercise autoscaling via IrisClient
5. Logs results to stdout via standard logging
6. Cleans up on success/failure/interrupt (`iris cluster stop`)

Usage:
    # Basic smoke test (uses examples/smoke.yaml)
    uv run python scripts/smoke-test.py

    # With custom config
    uv run python scripts/smoke-test.py --config examples/marin.yaml

    # Custom boot timeout for slow environments
    uv run python scripts/smoke-test.py --boot-timeout 600

    # Custom per-job timeout
    uv run python scripts/smoke-test.py --job-timeout 120

    # Keep cluster running on failure for debugging
    uv run python scripts/smoke-test.py --mode keep

    # Redeploy mode: reuse existing VMs (much faster for iteration)
    uv run python scripts/smoke-test.py --mode redeploy
"""

import json
import logging
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Literal, TextIO
from collections.abc import Callable

import click
from iris.client import IrisClient
from iris.cluster.config import load_config
import fsspec
from iris.cluster.types import (
    Constraint,
    CoschedulingConfig,
    Entrypoint,
    EnvironmentSpec,
    ResourceSpec,
    gpu_device,
    is_job_finished,
    preemptible_constraint,
    region_constraint,
    tpu_device,
)
from iris.rpc import cluster_pb2, config_pb2

logger = logging.getLogger("smoke-test")

IRIS_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG_PATH = IRIS_ROOT / "examples" / "smoke.yaml"

DEFAULT_JOB_TIMEOUT = 600  # 10 minutes; all jobs launch in parallel, this is the global wait ceiling
DEFAULT_BOOT_TIMEOUT = 300  # 5 minutes; cluster start + connection


def _log_section(title: str):
    logger.info("")
    logger.info("=" * 60)
    logger.info(" %s", title)
    logger.info("=" * 60)


# =============================================================================
# Diagnostics Formatting
# =============================================================================


def _format_resource(res: dict) -> str:
    """One-line summary of a resource spec, omitting zero-valued fields.

    Handles both job resources (device.gpu/tpu wrapper) and autoscaler demand
    entries (flat gpu_count/tpu_count fields).
    """
    parts: list[str] = []
    device = res.get("device", {})
    gpu = device.get("gpu")
    tpu = device.get("tpu")
    if gpu:
        parts.append(f"{gpu.get('count', 1)}x {gpu.get('variant', '?')}")
    elif tpu:
        parts.append(tpu.get("variant", "TPU"))
    elif int(res.get("gpu_count", 0)) > 0:
        parts.append(f"{res['gpu_count']} gpu")
    elif int(res.get("tpu_count", 0)) > 0:
        parts.append(f"{res['tpu_count']} tpu")
    cpu_millicores = res.get("cpuMillicores", res.get("cpu_millicores", 0))
    if cpu_millicores:
        parts.append(f"{int(cpu_millicores) / 1000:g} cpu")
    mem = int(res.get("memory_bytes", 0))
    if mem:
        parts.append(f"{mem / (1 << 30):.0f}GB mem")
    return ", ".join(parts) if parts else "none"


def _format_epoch_ms(epoch_ms: str | int) -> str:
    ms = int(epoch_ms)
    if ms == 0:
        return "never"
    return datetime.fromtimestamp(ms / 1000).strftime("%H:%M:%S")


def _worker_process_log_path(log_prefix: str, worker_id: str) -> str:
    return f"{log_prefix.rstrip('/')}/process/worker/{worker_id}/logs.jsonl"


def _load_worker_process_logs(log_prefix: str, worker_id: str, limit: int = 200) -> list[dict]:
    path = _worker_process_log_path(log_prefix, worker_id)
    try:
        with fsspec.open(path, "rb") as f:
            data = f.read().decode("utf-8", errors="replace")
    except FileNotFoundError:
        return []
    except Exception as e:
        logger.warning("Failed to fetch worker process logs from %s: %s", path, e)
        return []
    lines = [line for line in data.splitlines() if line.strip()]
    if limit > 0:
        lines = lines[-limit:]
    records: list[dict] = []
    for line in lines:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _log_autoscaler_pretty(status: dict) -> None:
    """Log autoscaler status as a readable summary."""
    groups = status.get("groups", [])
    if not groups:
        logger.info("  (no scale groups)")
        return

    logger.info("Scale groups:")
    for g in groups:
        name = g.get("name", "?")
        cfg = g.get("config", {})
        accel = cfg.get("accelerator_variant", cfg.get("accelerator_type", ""))
        slices = g.get("slices", [])
        counts = g.get("slice_state_counts", {})
        nonzero_states = {k: v for k, v in counts.items() if v}
        state_str = ", ".join(f"{k}={v}" for k, v in nonzero_states.items()) if nonzero_states else "idle"
        logger.info(
            "  %-20s  accel=%-10s  demand=%s  peak=%s  slices=%d (%s)  max=%s",
            name,
            accel,
            g.get("current_demand", 0),
            g.get("peak_demand", 0),
            len(slices),
            state_str,
            cfg.get("max_slices", "?"),
        )
        for s in slices:
            error_msg = s.get("error_message", "")
            if error_msg:
                logger.warning("    slice %s ERROR: %s", s.get("slice_id", "?"), error_msg)
        failures = g.get("consecutive_failures", 0)
        if failures:
            logger.warning(
                "    consecutive_failures=%d  backoff_until=%s",
                failures,
                _format_epoch_ms(g.get("backoff_until", {}).get("epoch_ms", "0")),
            )

    # Recent actions
    actions = status.get("recent_actions", [])
    if actions:
        logger.info("Recent autoscaler actions:")
        for a in actions:
            logger.info(
                "  %s  %-10s  group=%-15s  slice=%s  reason=%s  status=%s",
                _format_epoch_ms(a.get("timestamp", {}).get("epoch_ms", "0")),
                a.get("action_type", "?"),
                a.get("scale_group", "?"),
                a.get("slice_id", "?"),
                a.get("reason", ""),
                a.get("status", ""),
            )

    # Unmet demand
    routing = status.get("last_routing_decision", {})
    unmet = routing.get("unmet_entries", [])
    if unmet:
        logger.warning("Unmet demand (%d entries):", len(unmet))
        for u in unmet:
            entry = u.get("entry", {})
            task_ids = entry.get("task_ids", [])
            reason = u.get("reason", "?")
            res = _format_resource(entry.get("resources", {}))
            accel = entry.get("accelerator_variant", entry.get("accelerator_type", ""))
            logger.warning(
                "  reason=%-15s  accel=%-10s  resources=%-20s  tasks=%s",
                reason,
                accel,
                res,
                ", ".join(task_ids),
            )


def _log_jobs_pretty(jobs: list[dict]) -> None:
    """Log job list as a readable table."""
    if not jobs:
        logger.info("No jobs")
        return
    logger.info("Jobs (%d):", len(jobs))
    logger.info(
        "  %-40s  %-22s  %-20s  %s",
        "JOB",
        "STATE",
        "RESOURCES",
        "TASKS",
    )
    logger.info("  %s", "-" * 110)
    for j in jobs:
        job_id = j.get("job_id", "?")
        state = j.get("state", "?").replace("JOB_STATE_", "")
        res = _format_resource(j.get("resources", {}))
        task_counts = j.get("task_state_counts", {})
        task_str = ", ".join(f"{k}={v}" for k, v in task_counts.items() if v)
        error = j.get("error", "")
        line = f"  {job_id:<40s}  {state:<22s}  {res:<20s}  {task_str}"
        if error:
            line += f"  error={error}"
        failures = j.get("failure_count", 0)
        if failures:
            line += f"  failures={failures}"
        logger.info(line)


# =============================================================================
# Accelerator Detection
# =============================================================================


@dataclass
class AcceleratorConfig:
    """Accelerator configuration detected from cluster config."""

    device_type: str  # "gpu" or "tpu"
    variant: str  # e.g., "H100", "v5litepod-16"
    count: int  # GPUs per node or TPU chips per VM
    num_vms: int  # VMs per slice (replicas for coscheduled jobs)
    region: str  # for constraint tests

    @property
    def is_gpu(self) -> bool:
        return self.device_type == "gpu"

    @property
    def is_tpu(self) -> bool:
        return self.device_type == "tpu"

    def make_device(self) -> cluster_pb2.DeviceConfig:
        if self.is_gpu:
            return gpu_device(self.variant, self.count)
        return tpu_device(self.variant, self.count if self.count > 0 else None)

    def label(self) -> str:
        if self.is_gpu:
            return f"{self.count}x {self.variant}"
        return self.variant


def detect_accelerator(config_path: Path) -> AcceleratorConfig:
    """Detect the primary accelerator type and region from a cluster config.

    Scans scale groups for the first active (max_slices > 0) group and returns
    its accelerator type, variant, and region.
    """
    config = load_config(config_path)

    for _name, sg in config.scale_groups.items():
        if sg.HasField("max_slices") and sg.max_slices <= 0:
            continue
        if sg.accelerator_type not in (config_pb2.ACCELERATOR_TYPE_GPU, config_pb2.ACCELERATOR_TYPE_TPU):
            continue

        template = sg.slice_template
        platform = template.WhichOneof("platform")

        region = ""
        if platform == "coreweave" and template.coreweave.region:
            region = template.coreweave.region
        elif platform == "gcp" and template.gcp.zone:
            region = template.gcp.zone.rsplit("-", 1)[0]

        if sg.accelerator_type == config_pb2.ACCELERATOR_TYPE_GPU:
            return AcceleratorConfig(
                device_type="gpu",
                variant=sg.accelerator_variant,
                count=sg.resources.gpu_count or 1,
                num_vms=sg.num_vms,
                region=region,
            )

        return AcceleratorConfig(
            device_type="tpu",
            variant=sg.accelerator_variant,
            count=sg.resources.tpu_count,
            num_vms=sg.num_vms,
            region=region,
        )

    # Fallback for local/CPU configs — tests use tpu_device() which local workers accept
    return AcceleratorConfig(device_type="tpu", variant="v5litepod-16", count=4, num_vms=4, region="")


# =============================================================================
# CLI Helpers
# =============================================================================


DEFAULT_CLI_TIMEOUT = 300  # 5 minutes; CLI ops (stop/status) on a running cluster


def _run_iris(*args: str, config_path: Path, timeout: float = DEFAULT_CLI_TIMEOUT) -> subprocess.CompletedProcess[str]:
    """Run `uv run iris --config {config} ...` and return the result.

    Args:
        *args: CLI arguments to pass after `iris --config {config}`.
        config_path: Path to the cluster config YAML.
        timeout: Maximum seconds to wait for the command to finish.

    Raises subprocess.CalledProcessError on non-zero exit,
    subprocess.TimeoutExpired if the command exceeds timeout.
    """
    cmd = ["uv", "run", "iris", "--config", str(config_path), *args]
    logger.info("Running (timeout=%ds): %s", timeout, " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(IRIS_ROOT), timeout=timeout)
    if result.returncode != 0:
        logger.error(
            "Command failed (exit %d): %s\nstdout: %s\nstderr: %s",
            result.returncode,
            " ".join(cmd),
            result.stdout,
            result.stderr,
        )
        result.check_returncode()
    return result


def _run_iris_rpc(
    controller_url: str, service: str, method: str, *args: str, timeout: float = 30
) -> subprocess.CompletedProcess[str]:
    """Run `uv run iris --controller-url <url> rpc <service> <method> ...`.

    Returns the CompletedProcess; does not raise on non-zero exit so callers
    can handle failures gracefully during diagnostics.
    """
    cmd = ["uv", "run", "iris", "--controller-url", controller_url, "rpc", service, method, *args]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(IRIS_ROOT), timeout=timeout)


@dataclass
class BackgroundProc:
    """A background subprocess with its owned file handles.

    Owns the file handles opened for stdout/stderr redirection so they are
    closed when the process is terminated, preventing resource leaks.
    Tracks daemon threads (reader/drain) so terminate() can join them.
    """

    name: str
    proc: subprocess.Popen
    owned_fds: list[TextIO]
    _threads: list[threading.Thread] = field(default_factory=list)

    def terminate(self, timeout: float = 10.0) -> None:
        _terminate_process(self.proc, self.name, timeout)
        for t in self._threads:
            t.join(timeout=5.0)
        for fh in self.owned_fds:
            fh.close()


def _run_iris_background(
    *args: str,
    config_path: Path,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
) -> BackgroundProc:
    """Start `uv run iris --config {config} ...` as a background subprocess.

    Returns a BackgroundProc that owns any opened file handles. Caller is
    responsible for calling terminate() to stop the process and close handles.
    """
    cmd = ["uv", "run", "iris", "--config", str(config_path), *args]
    logger.info("Starting background: %s", " ".join(cmd))

    owned_fds: list[TextIO] = []
    if stdout_path:
        stdout_fh: TextIO | int = open(stdout_path, "w")
        owned_fds.append(stdout_fh)  # type: ignore[arg-type]
    else:
        stdout_fh = subprocess.PIPE

    if stderr_path:
        stderr_fh: TextIO | int = open(stderr_path, "w")
        owned_fds.append(stderr_fh)  # type: ignore[arg-type]
    else:
        stderr_fh = subprocess.STDOUT

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=stdout_fh,
            stderr=stderr_fh,
            text=True,
            cwd=str(IRIS_ROOT),
        )
    except Exception:
        for fh in owned_fds:
            fh.close()
        raise
    return BackgroundProc(name="", proc=proc, owned_fds=owned_fds)


def _wait_for_line(bg: BackgroundProc, pattern: str, timeout: float = 300) -> str:
    """Read stdout from a background subprocess until a line matches `pattern`.

    Uses a reader thread so the timeout is enforced even if readline blocks on a
    partial line.  After the match is found, a drain thread continues consuming
    stdout to prevent the pipe buffer from filling and blocking the subprocess.

    Returns the full matching line. Raises TimeoutError or RuntimeError on failure.
    """
    proc = bg.proc
    assert proc.stdout is not None, "subprocess stdout must be PIPE"

    line_queue: queue.Queue[str | None] = queue.Queue()

    def _reader():
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            line_queue.put(raw_line.rstrip("\n"))
        line_queue.put(None)  # EOF sentinel

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()
    bg._threads.append(reader_thread)

    deadline = time.monotonic() + timeout
    compiled = re.compile(pattern)
    # Docker push/pull output: 12-char hex hash followed by layer status.
    # These repeat hundreds of times and provide no useful signal.
    docker_layer_noise = re.compile(
        r"^[0-9a-f]{12}: (Waiting|Preparing|Pushing|Pulling|Layer already|Mounted|Pushed|Pulled)"
    )
    matched_line: str | None = None

    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        try:
            line = line_queue.get(timeout=min(remaining, 1.0))
        except queue.Empty:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"Subprocess exited with code {proc.returncode} before matching pattern: {pattern}"
                ) from None
            continue
        if line is None:
            raise RuntimeError(f"Subprocess EOF before matching pattern: {pattern}")
        if not docker_layer_noise.search(line):
            logger.info("  [bg] %s", line)
        if compiled.search(line):
            matched_line = line
            break

    if matched_line is None:
        raise TimeoutError(f"Timed out waiting for pattern '{pattern}' after {timeout}s")

    # Drain remaining stdout so the pipe never fills and blocks the subprocess.
    def _drain():
        while True:
            try:
                remaining_line = line_queue.get(timeout=1.0)
            except queue.Empty:
                if proc.poll() is not None:
                    break
                continue
            if remaining_line is None:
                break

    drain_thread = threading.Thread(target=_drain, daemon=True)
    drain_thread.start()
    bg._threads.append(drain_thread)

    return matched_line


def _parse_address_from_line(line: str, prefix: str) -> str:
    """Extract the address/URL that follows `prefix` in a line."""
    idx = line.index(prefix)
    return line[idx + len(prefix) :].strip()


def _terminate_process(proc: subprocess.Popen, name: str, timeout: float = 10.0) -> None:
    """Send SIGTERM to a subprocess and wait for it to exit."""
    if proc.poll() is not None:
        return
    logger.info("Terminating %s (pid=%d)...", name, proc.pid)
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning("Process %s did not exit after SIGTERM, sending SIGKILL", name)
        proc.kill()
        proc.wait(timeout=5.0)


# =============================================================================
# Test Job Definitions
# =============================================================================


def _hello_job():
    """Simple job that prints and returns."""
    print("Hello from accelerator!")
    return 42


def _quick_task_job(task_id: int):
    """Quick job that sleeps and returns."""
    import time as time_module

    time_module.sleep(2.0)
    print(f"Task {task_id} completed")
    return task_id


def _assert_region_child(expected_region: str):
    """Parent job that submits a child and asserts the child inherits the region constraint.

    Uses iris_ctx() to get the IrisClient and submit a child without explicit
    constraints. The child reads its inherited constraints from JobInfo and
    asserts the expected region is present.
    """
    from iris.client.client import iris_ctx
    from iris.cluster.types import ResourceSpec as RS

    ctx = iris_ctx()

    def _child_check_region():
        from iris.cluster.client import get_job_info
        from iris.cluster.types import REGION_ATTRIBUTE_KEY as RK

        info = get_job_info()
        if info is None:
            raise RuntimeError("Not running in an Iris job context")
        region_constraints = [c for c in info.constraints if c.key == RK]
        if not region_constraints:
            raise RuntimeError(f"No region constraint found. constraints={info.constraints}")
        actual = region_constraints[0].value
        if actual != expected_region:
            raise RuntimeError(f"Expected region {expected_region}, got {actual}")
        print(f"Child validated inherited region: {actual}")
        return actual

    child = ctx.client.submit(
        entrypoint=Entrypoint.from_callable(_child_check_region),
        name="smoke-inherited-child",
        resources=RS(cpu=1, memory="1GB", disk="1GB"),
    )
    child.wait(timeout=60, raise_on_failure=True)
    print(f"Parent: child completed with inherited region={expected_region}")


def _distributed_work_job():
    """Coscheduled job that uses job context."""
    from iris.cluster.client import get_job_info

    info = get_job_info()
    if info is None:
        raise RuntimeError("Not running in an Iris job context")
    print(f"Task {info.task_index} of {info.num_tasks} on worker {info.worker_id}")
    return f"Task {info.task_index} done"


def _jax_tpu_job():
    """Initialize JAX on TPU and run simple computation."""
    import os
    import sys

    print(f"Python: {sys.executable} {sys.version}", flush=True)
    print(f"JAX_PLATFORMS={os.environ.get('JAX_PLATFORMS', '<unset>')}", flush=True)
    print(f"PJRT_DEVICE={os.environ.get('PJRT_DEVICE', '<unset>')}", flush=True)
    print(f"TPU_NAME={os.environ.get('TPU_NAME', '<unset>')}", flush=True)
    print(f"TPU_WORKER_ID={os.environ.get('TPU_WORKER_ID', '<unset>')}", flush=True)
    print(f"TPU_WORKER_HOSTNAMES={os.environ.get('TPU_WORKER_HOSTNAMES', '<unset>')}", flush=True)
    print(f"TPU_CHIPS_PER_HOST_BOUNDS={os.environ.get('TPU_CHIPS_PER_HOST_BOUNDS', '<unset>')}", flush=True)

    # Check /dev/vfio exists
    vfio_path = "/dev/vfio"
    if os.path.exists(vfio_path):
        print(f"/dev/vfio exists, contents: {os.listdir(vfio_path)}", flush=True)
    else:
        print("/dev/vfio does NOT exist", flush=True)

    # Check jax is importable
    print("Importing jax...", flush=True)
    import jax

    print(f"JAX version: {jax.__version__}", flush=True)

    import jax.numpy as jnp

    # Verify TPU is available
    print("Calling jax.devices()...", flush=True)
    devices = jax.devices()
    tpu_devices = [d for d in devices if d.platform == "tpu"]
    if not tpu_devices:
        raise RuntimeError(f"No TPU devices found. Available: {[d.platform for d in devices]}")

    print(f"Found {len(tpu_devices)} TPU device(s): {tpu_devices}")

    # Simple computation to exercise TPU
    x = jnp.ones((1000, 1000))
    y = jnp.dot(x, x)
    result = float(y[0, 0])

    print(f"JAX TPU computation successful: {result}")
    return result


def _check_gpus_job(expected_count: int):
    """Verify that the expected number of GPUs are visible via nvidia-smi."""
    import subprocess as _sp

    result = _sp.run(
        ["nvidia-smi", "-L"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed (exit {result.returncode}): {result.stderr}")

    gpus = [line for line in result.stdout.splitlines() if line.startswith("GPU ")]
    print(f"Found {len(gpus)} GPU(s):")
    for gpu in gpus:
        print(f"  {gpu}")

    if len(gpus) < expected_count:
        raise RuntimeError(f"Expected {expected_count} GPUs, found {len(gpus)}")
    return len(gpus)


def _configure_logging():
    """Configure logging to show all iris module output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )
    # Ensure iris modules are visible
    logging.getLogger("iris").setLevel(logging.INFO)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SmokeTestConfig:
    """Configuration for the smoke test."""

    config_path: Path
    accelerator: AcceleratorConfig
    boot_timeout_seconds: int = DEFAULT_BOOT_TIMEOUT
    job_timeout_seconds: int = DEFAULT_JOB_TIMEOUT
    local: bool = False  # Run locally without GCP
    mode: Literal["full", "keep", "redeploy"] = "full"


# =============================================================================
# Test Result Tracking
# =============================================================================


@dataclass
class TestResult:
    """Result of a single test."""

    name: str
    passed: bool
    message: str
    duration_seconds: float


@dataclass
class SmokeTestCase:
    """Executable smoke test case."""

    label: str
    run: Callable[[IrisClient], TestResult]


# =============================================================================
# Main Test Runner
# =============================================================================


class SmokeTestRunner:
    """Orchestrates the smoke test lifecycle.

    Uses the `iris` CLI for cluster lifecycle (start, stop, cleanup, monitoring)
    and `IrisClient` for job submission. Background subprocesses (dashboard tunnel,
    controller log streaming) are tracked and cleaned up on exit.
    """

    def __init__(self, config: SmokeTestConfig):
        self.config = config
        self._accel = config.accelerator
        self._interrupted = False
        self._results: list[TestResult] = []
        self._failed = False
        self._cluster_config = load_config(config.config_path)
        # Unique run ID to avoid job name collisions with previous runs
        self._run_id = datetime.now().strftime("%H%M%S")
        # Background subprocesses to clean up
        self._background_procs: list[BackgroundProc] = []
        # Set once we have a controller connection (for RPC diagnostics at teardown)
        self._controller_url: str | None = None

    def run(self) -> bool:
        """Run the smoke test. Returns True if all tests pass."""
        _configure_logging()

        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

        self._configure_s3_env()

        try:
            self._print_header()

            # Cluster startup: always run in the background so both local
            # configs (which block on controller.wait()) and remote configs
            # work identically.
            controller_url: str | None = None

            if self.config.mode != "redeploy":
                if self.config.mode in ("full", "keep") and not self.config.local:
                    _log_section("PHASE 0: Clean Start")
                    self._cleanup_existing()
                    if self._interrupted:
                        return False

                _log_section("Starting Cluster")
                cluster_address = self._start_cluster()
                if self._interrupted:
                    return False

                if self.config.local:
                    controller_url = cluster_address

            if controller_url is None:
                # Remote mode (or redeploy): establish tunnel to get a
                # localhost URL regardless of whether we started the cluster
                # above or it was already running.
                _log_section("Connecting to Cluster")
                controller_url = self._connect_remote()

            self._controller_url = controller_url

            # Run tests
            _log_section("Running Tests")
            self._run_tests(controller_url)

            # Results
            _log_section("Results Summary")
            success = self._print_results()
            return success

        except Exception as e:
            logger.error("FATAL ERROR: %s", e)
            self._failed = True
            return False

        finally:
            self._cleanup()

    def _handle_interrupt(self, _signum: int, _frame: object):
        logger.warning("Interrupted! Cleaning up...")
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        self._interrupted = True

    def _configure_s3_env(self) -> None:
        """Set AWS_ENDPOINT_URL from the CoreWeave config if not already set.

        On the operator's laptop, AWS_ENDPOINT_URL is typically not set (it's
        injected inside K8s Pods). Without it, fsspec/s3fs can't reach CoreWeave
        Object Storage for reading logs.
        """
        if os.environ.get("AWS_ENDPOINT_URL"):
            return
        platform = self._cluster_config.platform
        if platform.HasField("coreweave") and platform.coreweave.object_storage_endpoint:
            endpoint = platform.coreweave.object_storage_endpoint
            os.environ["AWS_ENDPOINT_URL"] = endpoint
            logger.info("Set AWS_ENDPOINT_URL=%s from config", endpoint)

    def _print_header(self):
        logger.info("")
        logger.info("=" * 60)
        logger.info(" IRIS CLUSTER SMOKE TEST")
        logger.info("=" * 60)
        logger.info("")
        logger.info("Config: %s", self.config.config_path)
        logger.info("Boot timeout: %ds", self.config.boot_timeout_seconds)
        logger.info("Job timeout: %ds", self.config.job_timeout_seconds)
        logger.info("Accelerator: %s (%s)", self._accel.label(), self._accel.device_type)
        logger.info("Local: %s", self.config.local)

    # ----- Cluster lifecycle via CLI -----

    def _cleanup_existing(self):
        """Delete existing iris resources via `iris cluster stop`."""
        logger.info("Cleaning up existing resources...")
        try:
            result = _run_iris(
                "cluster", "stop", config_path=self.config.config_path, timeout=self.config.boot_timeout_seconds
            )
            for line in result.stdout.splitlines():
                logger.info("  %s", line)
            logger.info("Cleanup complete")
        except subprocess.CalledProcessError as e:
            logger.error("Cleanup failed: %s", e.stderr)
            raise

    def _start_cluster(self) -> str:
        """Start cluster via ``iris cluster start`` as a background subprocess.

        Always runs in the background so it works for both local and remote
        configs (local controllers block on ``controller.wait()``; remote exits
        after the controller VM is booted).  Adds ``--local`` when
        ``self.config.local`` is set.

        Returns the controller address from the "Controller started at" line.
        For local mode this is directly usable; for remote mode the caller
        should still establish a dashboard tunnel.
        """
        args = ["cluster", "start"]
        if self.config.local:
            args.append("--local")

        logger.info("Starting cluster (background)...")
        bg = _run_iris_background(*args, config_path=self.config.config_path)
        bg.name = "cluster-start"
        self._background_procs.append(bg)

        line = _wait_for_line(bg, r"Controller started at", timeout=self.config.boot_timeout_seconds)
        address = _parse_address_from_line(line, "Controller started at")
        logger.info("Controller started at: %s", address)
        return address

    def _connect_remote(self) -> str:
        """Establish tunnel via `iris cluster dashboard` as background subprocess.

        Parses the controller URL from the output line "Controller RPC: {url}".
        The subprocess blocks to keep the tunnel alive.
        """
        logger.info("Starting dashboard tunnel...")
        bg = _run_iris_background("cluster", "dashboard", config_path=self.config.config_path)
        bg.name = "dashboard-tunnel"
        self._background_procs.append(bg)

        line = _wait_for_line(bg, r"Controller RPC:", timeout=self.config.boot_timeout_seconds)
        controller_url = _parse_address_from_line(line, "Controller RPC:")
        logger.info("Controller URL: %s", controller_url)
        return controller_url

    # ----- Monitoring via CLI -----

    def _log_autoscaler_status(self):
        """Log current cluster state via `iris cluster status`."""
        if self.config.local:
            # `iris cluster status` resolves controller discovery through the config platform
            # (GCP/manual/coreweave). In local smoke mode we intentionally run a local
            # controller against the same config, so this command is not meaningful.
            return
        try:
            result = _run_iris("cluster", "status", config_path=self.config.config_path)
            for line in result.stdout.splitlines():
                logger.info("  %s", line)
        except Exception as e:
            logger.warning("  (Could not fetch cluster status: %s)", e)

    # ----- Test execution -----

    def _run_tests(self, controller_url: str):
        """Run test jobs against the cluster."""
        tests = self._build_test_cases()
        self._run_test_cases_parallel(controller_url, tests)

    def _build_test_cases(self) -> list[SmokeTestCase]:
        """Build the test list based on accelerator type and local mode."""
        a = self._accel
        local = self.config.local
        scheduling_only = local and a.is_tpu
        so = self._maybe_scheduling_only

        tests: list[SmokeTestCase] = []

        tests.append(SmokeTestCase(f"Simple job ({a.label()})", so(self._run_simple_job, scheduling_only)))
        tests.append(SmokeTestCase(f"Concurrent jobs (3x {a.label()})", so(self._run_concurrent_jobs, scheduling_only)))

        if a.is_tpu:
            tests.append(
                SmokeTestCase(
                    f"Coscheduled multi-task job ({a.variant})", so(self._run_coscheduled_job, scheduling_only)
                )
            )

        if a.is_tpu and not local:
            tests.append(SmokeTestCase(f"JAX TPU job ({a.variant})", self._run_jax_tpu_job))

        if a.is_gpu and not local:
            tests.append(SmokeTestCase(f"Multi-GPU device check ({a.label()})", self._run_gpu_check_job))

        if self._has_non_preemptible_cpu_group():
            tests.append(SmokeTestCase("Non-preemptible CPU job", self._run_non_preemptible_cpu_job))

        if a.region and not (a.is_gpu and local):
            tests.append(
                SmokeTestCase(
                    f"Region-constrained job ({a.region})", so(self._run_region_constrained_job, scheduling_only)
                )
            )
            tests.append(
                SmokeTestCase(
                    f"Nested constraint propagation ({a.region})", so(self._run_nested_constraint_job, scheduling_only)
                )
            )

        return tests

    def _maybe_scheduling_only(self, method, scheduling_only: bool):
        """Wrap a test method with scheduling_only=True when needed."""
        if scheduling_only:
            return partial(method, scheduling_only=True)
        return method

    def _run_test_cases_parallel(self, controller_url: str, tests: list[SmokeTestCase]) -> None:
        """Run all suite tests in parallel so resource demand is requested at once."""
        if not tests:
            return

        logger.info("Launching %d smoke tests in parallel", len(tests))

        def _run_case(case: SmokeTestCase) -> TestResult:
            client = IrisClient.remote(controller_url, workspace=IRIS_ROOT)
            logger.info("[START] %s", case.label)
            return case.run(client)

        with ThreadPoolExecutor(max_workers=len(tests), thread_name_prefix="smoke-test") as executor:
            future_pairs = [(case, executor.submit(_run_case, case)) for case in tests]
            for case, future in future_pairs:
                if self._interrupted:
                    future.cancel()
                    continue
                try:
                    result = future.result()
                except Exception as e:
                    logger.exception("  [FAIL] %s raised an exception", case.label)
                    result = TestResult(case.label, False, f"Exception: {e}", 0.0)
                self._results.append(result)
                self._log_autoscaler_status()

    def _run_gpu_check_job(self, client: IrisClient) -> TestResult:
        return self._run_job_test(
            client=client,
            test_name=f"Multi-GPU device check ({self._accel.label()})",
            entrypoint=Entrypoint.from_callable(_check_gpus_job, self._accel.count),
            job_name=f"smoke-gpu-check-{self._run_id}",
            resources=ResourceSpec(device=self._accel.make_device()),
        )

    def _has_non_preemptible_cpu_group(self) -> bool:
        """Return True when config contains an active non-preemptible CPU group."""
        for sg in self._cluster_config.scale_groups.values():
            if sg.HasField("max_slices") and sg.max_slices <= 0:
                continue
            if sg.accelerator_type != config_pb2.ACCELERATOR_TYPE_CPU:
                continue
            if sg.slice_template.preemptible:
                continue
            return True
        return False

    def _run_job_test(
        self,
        client: IrisClient,
        test_name: str,
        entrypoint: Entrypoint,
        job_name: str,
        resources: ResourceSpec,
        coscheduling: CoschedulingConfig | None = None,
        environment: EnvironmentSpec | None = None,
        constraints: list[Constraint] | None = None,
        replicas: int = 1,
        timeout: int | None = None,
        scheduling_only: bool = False,
    ) -> TestResult:
        """Generic job runner that handles submission, waiting, and result collection."""
        job_timeout = timeout or self.config.job_timeout_seconds
        start = time.monotonic()
        try:
            job = client.submit(
                entrypoint=entrypoint,
                name=job_name,
                resources=resources,
                environment=environment or EnvironmentSpec(),
                constraints=constraints,
                coscheduling=coscheduling,
                replicas=replicas,
            )
            logger.info("  Job submitted: %s", job.job_id)

            if scheduling_only:
                scheduled, detail, status = self._wait_for_scheduling(job, timeout=min(job_timeout, 30))
                duration = time.monotonic() - start
                if not is_job_finished(status.state):
                    try:
                        job.terminate()
                    except Exception as e:
                        logger.warning("  Could not terminate scheduled-only job %s: %s", job.job_id, e)

                if scheduled:
                    logger.info("  [PASS] %s in %.1fs", detail, duration)
                    return TestResult(test_name, True, detail, duration)

                logger.error("  [FAIL] %s", detail)
                return TestResult(test_name, False, detail, duration)

            status = job.wait(timeout=job_timeout, raise_on_failure=False, stream_logs=True)
            duration = time.monotonic() - start

            if status.state == cluster_pb2.JOB_STATE_SUCCEEDED:
                logger.info("  [PASS] Completed in %.1fs", duration)
                return TestResult(test_name, True, f"Completed in {duration:.1f}s", duration)
            else:
                state_name = cluster_pb2.JobState.Name(status.state)
                logger.error(
                    "  [FAIL] Job ended with state %s (use `iris job logs %s` to inspect)",
                    state_name,
                    job.job_id,
                )
                return TestResult(test_name, False, f"State: {state_name}, error: {status.error}", duration)

        except TimeoutError:
            duration = time.monotonic() - start
            logger.error("  [FAIL] Timed out after %ds", job_timeout)
            return TestResult(test_name, False, f"Timed out after {job_timeout}s", duration)

    def _submit_and_wait_multiple(
        self,
        client: IrisClient,
        jobs_config: list[tuple[Entrypoint, str, ResourceSpec]],
        test_name: str = "Concurrent jobs",
        scheduling_only: bool = False,
    ) -> tuple[float, list[str]]:
        """Submit multiple jobs and wait for all to complete.

        Returns:
            (duration, failed_job_descriptions)
        """
        start = time.monotonic()
        jobs = []

        for entrypoint, name, resources in jobs_config:
            job = client.submit(
                entrypoint=entrypoint,
                name=name,
                resources=resources,
                environment=EnvironmentSpec(),
            )
            jobs.append(job)
            logger.info("  Job submitted: %s", job.job_id)

        failed_jobs = []
        if scheduling_only:
            for job in jobs:
                scheduled, detail, status = self._wait_for_scheduling(
                    job, timeout=min(self.config.job_timeout_seconds, 30)
                )
                if not scheduled:
                    failed_jobs.append(f"{job.job_id}: {detail}")
                if not is_job_finished(status.state):
                    try:
                        job.terminate()
                    except Exception as e:
                        logger.warning("  Could not terminate scheduled-only job %s: %s", job.job_id, e)
        else:
            for job in jobs:
                status = job.wait(timeout=self.config.job_timeout_seconds, raise_on_failure=False, stream_logs=True)
                if status.state != cluster_pb2.JOB_STATE_SUCCEEDED:
                    state_name = cluster_pb2.JobState.Name(status.state)
                    failed_jobs.append(f"{job.job_id}: {state_name}")

        return time.monotonic() - start, failed_jobs

    def _wait_for_scheduling(self, job, timeout: int) -> tuple[bool, str, cluster_pb2.JobStatus]:
        """Wait until a job reaches a schedulable state or terminal failure."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = job.status()
            state_name = cluster_pb2.JobState.Name(status.state)
            if status.state in (
                cluster_pb2.JOB_STATE_PENDING,
                cluster_pb2.JOB_STATE_BUILDING,
                cluster_pb2.JOB_STATE_RUNNING,
                cluster_pb2.JOB_STATE_SUCCEEDED,
            ):
                return True, f"Scheduled (state={state_name})", status
            if status.state in (
                cluster_pb2.JOB_STATE_FAILED,
                cluster_pb2.JOB_STATE_KILLED,
                cluster_pb2.JOB_STATE_WORKER_FAILED,
                cluster_pb2.JOB_STATE_UNSCHEDULABLE,
            ):
                return False, f"Failed before scheduling (state={state_name}, error={status.error})", status
            time.sleep(0.5)

        status = job.status()
        state_name = cluster_pb2.JobState.Name(status.state)
        return False, f"Did not reach schedulable state in {timeout}s (state={state_name})", status

    def _run_simple_job(self, client: IrisClient, scheduling_only: bool = False) -> TestResult:
        """Run a simple job that just prints and returns."""
        return self._run_job_test(
            client=client,
            test_name=f"Simple job ({self._accel.label()})",
            entrypoint=Entrypoint.from_callable(_hello_job),
            job_name=f"smoke-simple-{self._run_id}",
            resources=ResourceSpec(device=self._accel.make_device()),
            scheduling_only=scheduling_only,
        )

    def _run_concurrent_jobs(self, client: IrisClient, scheduling_only: bool = False) -> TestResult:
        """Submit 3 concurrent jobs to test parallel provisioning and queueing."""
        resources = ResourceSpec(device=self._accel.make_device())
        test_name = f"Concurrent jobs (3x {self._accel.label()})"
        jobs_config = [
            (Entrypoint.from_callable(_quick_task_job, i), f"smoke-concurrent-{self._run_id}-{i}", resources)
            for i in range(3)
        ]

        try:
            duration, failed_jobs = self._submit_and_wait_multiple(
                client, jobs_config, test_name, scheduling_only=scheduling_only
            )

            if not failed_jobs:
                if scheduling_only:
                    logger.info("  [PASS] All 3 jobs scheduled in %.1fs", duration)
                    return TestResult(test_name, True, f"All scheduled in {duration:.1f}s", duration)
                logger.info("  [PASS] All 3 jobs completed in %.1fs", duration)
                return TestResult(test_name, True, f"All completed in {duration:.1f}s", duration)
            else:
                logger.error("  [FAIL] Some jobs failed: %s", ", ".join(failed_jobs))
                return TestResult(test_name, False, f"Failed: {', '.join(failed_jobs)}", duration)

        except TimeoutError:
            logger.error("  [FAIL] Timed out waiting for jobs")
            return TestResult(test_name, False, f"Timed out after {self.config.job_timeout_seconds}s", 0.0)

    def _run_coscheduled_job(self, client: IrisClient, scheduling_only: bool = False) -> TestResult:
        """Run a coscheduled multi-task job on TPU workers."""
        return self._run_job_test(
            client=client,
            test_name="Coscheduled multi-task job",
            entrypoint=Entrypoint.from_callable(_distributed_work_job),
            job_name=f"smoke-coscheduled-{self._run_id}",
            resources=ResourceSpec(device=self._accel.make_device()),
            coscheduling=CoschedulingConfig(group_by="tpu-name"),
            replicas=self._accel.num_vms,
            scheduling_only=scheduling_only,
        )

    def _run_jax_tpu_job(self, client: IrisClient) -> TestResult:
        """Run a JAX TPU job that initializes JAX and exercises the TPU.

        Uses coscheduling because multi-host TPU pods require all hosts to run
        JAX simultaneously for collective initialization.
        """
        return self._run_job_test(
            client=client,
            test_name=f"JAX TPU job ({self._accel.variant})",
            entrypoint=Entrypoint.from_callable(_jax_tpu_job),
            job_name=f"smoke-jax-tpu-{self._run_id}",
            resources=ResourceSpec(device=self._accel.make_device()),
            environment=EnvironmentSpec(pip_packages=["jax[tpu]"]),
            coscheduling=CoschedulingConfig(group_by="tpu-name"),
            replicas=self._accel.num_vms,
        )

    def _run_region_constrained_job(self, client: IrisClient, scheduling_only: bool = False) -> TestResult:
        """Run a CPU-only job with an explicit region constraint."""
        return self._run_job_test(
            client=client,
            test_name=f"Region-constrained job ({self._accel.region})",
            entrypoint=Entrypoint.from_callable(_hello_job),
            job_name=f"smoke-region-{self._run_id}",
            resources=ResourceSpec(cpu=1, memory="1GB", disk="1GB"),
            constraints=[region_constraint([self._accel.region])],
            scheduling_only=scheduling_only,
        )

    def _run_nested_constraint_job(self, client: IrisClient, scheduling_only: bool = False) -> TestResult:
        """Submit a parent job with a region constraint whose body submits a child.

        The child inherits the parent's region constraint and asserts it.
        """
        a = self._accel
        return self._run_job_test(
            client=client,
            test_name="Nested constraint propagation",
            entrypoint=Entrypoint.from_callable(_assert_region_child, a.region),
            job_name=f"smoke-nested-{self._run_id}",
            resources=ResourceSpec(cpu=1, memory="1GB", disk="1GB"),
            constraints=[region_constraint([a.region])],
            scheduling_only=scheduling_only,
        )

    def _run_non_preemptible_cpu_job(self, client: IrisClient, scheduling_only: bool = False) -> TestResult:
        """Run a CPU-only job constrained to non-preemptible workers."""
        return self._run_job_test(
            client=client,
            test_name="Non-preemptible CPU job",
            entrypoint=Entrypoint.from_callable(_quick_task_job, 1),
            job_name=f"smoke-non-preemptible-cpu-{self._run_id}",
            resources=ResourceSpec(cpu=1, memory="1GB", disk="1GB"),
            constraints=[preemptible_constraint(False)],
            scheduling_only=scheduling_only,
        )

    # ----- Diagnostics and cleanup -----

    def _dump_controller_diagnostics(self):
        """Fetch controller process logs, autoscaler status, and job list via CLI RPC.

        Called before teardown so diagnostics are captured even when tests fail.
        Only runs if we have a controller URL (i.e. we connected to the cluster).
        """
        if not self._controller_url:
            return

        _log_section("Controller Diagnostics")
        self._dump_process_logs()
        self._dump_autoscaler_status()
        self._dump_job_list()

    def _dump_worker_process_logs(self):
        assert self._controller_url
        log_prefix = self._cluster_config.storage.log_prefix
        if not log_prefix:
            logger.warning("Worker process logs unavailable: storage.log_prefix not configured")
            return
        try:
            result = _run_iris_rpc(self._controller_url, "controller", "list-workers")
            if result.returncode != 0:
                logger.warning("Failed to list workers: %s", result.stderr)
                return
            workers = json.loads(result.stdout).get("workers", [])
        except Exception as e:
            logger.warning("Could not list workers: %s", e)
            return
        if not workers:
            logger.warning("No workers reported by controller")
            return

        _log_section("Worker Process Logs")
        for worker in workers:
            worker_id = worker.get("worker_id") or "unknown"
            records = _load_worker_process_logs(log_prefix, worker_id, limit=500)
            if not records:
                logger.info("Worker %s: no process logs found", worker_id)
                continue
            warn_error = [r for r in records if r.get("level") in ("WARNING", "ERROR")]
            if not warn_error:
                logger.info("Worker %s: no WARNING/ERROR logs", worker_id)
                continue
            seen: dict[str, int] = {}
            for r in warn_error:
                msg = r.get("message", "")
                core = msg.split("] ", 1)[-1] if "] " in msg else msg
                seen[core] = seen.get(core, 0) + 1
            logger.warning(
                "Worker %s WARNING/ERROR logs (%d entries, %d unique):",
                worker_id,
                len(warn_error),
                len(seen),
            )
            for msg, count in seen.items():
                suffix = f" (x{count})" if count > 1 else ""
                logger.warning("  %s%s", msg, suffix)

    def _dump_process_logs(self):
        assert self._controller_url
        try:
            result = _run_iris_rpc(self._controller_url, "controller", "get-process-logs", "--limit", "0")
            if result.returncode != 0:
                logger.warning("Failed to fetch process logs: %s", result.stderr)
                return
            records = json.loads(result.stdout).get("records", [])
            warn_error = [r for r in records if r.get("level") in ("WARNING", "ERROR")]
            if not warn_error:
                logger.info("No WARNING/ERROR logs from controller")
                return
            # Deduplicate repeated messages (e.g. CAPACITY INSUFFICIENT every 10s)
            seen: dict[str, int] = {}
            for r in warn_error:
                msg = r["message"]
                # Strip the leading timestamp to group identical messages
                core = msg.split("] ", 1)[-1] if "] " in msg else msg
                seen[core] = seen.get(core, 0) + 1
            logger.warning("Controller WARNING/ERROR logs (%d entries, %d unique):", len(warn_error), len(seen))
            for msg, count in seen.items():
                suffix = f" (x{count})" if count > 1 else ""
                logger.warning("  %s%s", msg, suffix)
        except Exception as e:
            logger.warning("Could not fetch process logs: %s", e)

    def _dump_autoscaler_status(self):
        assert self._controller_url
        try:
            result = _run_iris_rpc(self._controller_url, "controller", "get-autoscaler-status")
            if result.returncode != 0:
                logger.warning("Failed to fetch autoscaler status: %s", result.stderr)
                return
            status = json.loads(result.stdout).get("status", {})
            _log_autoscaler_pretty(status)
        except Exception as e:
            logger.warning("Could not fetch autoscaler status: %s", e)

    def _dump_job_list(self):
        assert self._controller_url
        try:
            result = _run_iris_rpc(self._controller_url, "controller", "list-jobs")
            if result.returncode != 0:
                logger.warning("Failed to fetch job list: %s", result.stderr)
                return
            data = json.loads(result.stdout)
            jobs = data.get("jobs", [])
            _log_jobs_pretty(jobs)
        except Exception as e:
            logger.warning("Could not fetch job list: %s", e)

    def _print_results(self) -> bool:
        """Print final results and return True if all passed."""
        all_passed = True
        total_duration = 0.0

        for result in self._results:
            status = "PASS" if result.passed else "FAIL"
            logger.info("  [%s] %s: %s", status, result.name, result.message)
            total_duration += result.duration_seconds
            if not result.passed:
                all_passed = False

        logger.info("")
        passed_count = sum(1 for r in self._results if r.passed)
        total_count = len(self._results)

        if all_passed:
            logger.info("Results: %d/%d tests passed in %.1fs", passed_count, total_count, total_duration)
        else:
            logger.warning("Results: %d/%d tests passed in %.1fs", passed_count, total_count, total_duration)
            self._failed = True

        return all_passed

    def _cleanup(self):
        """Clean up cluster resources and background subprocesses.

        Each cleanup phase is wrapped in its own try/except so that a failure
        in one phase (e.g. terminating a background process) does not prevent
        the cluster from being torn down.
        """
        # Fetch diagnostics while the controller and dashboard tunnel are still alive.
        self._dump_controller_diagnostics()
        if self._failed and self._controller_url:
            self._dump_worker_process_logs()

        _log_section("CLEANUP")

        # Terminate all background subprocesses and close their owned file handles.
        # Failures here must not prevent cluster teardown below.
        for bg in self._background_procs:
            try:
                bg.terminate()
            except Exception as e:
                logger.warning("Error terminating %s: %s", bg.name, e)
        self._background_procs.clear()
        logger.info("Stopped background processes")

        # In redeploy mode, skip VM cleanup to preserve VMs for next run
        if self.config.mode == "redeploy":
            logger.info("Redeploy mode: keeping VMs running for next iteration")
            return

        if self.config.mode == "keep":
            logger.info("Skipping cleanup (--mode keep)")
            logger.info("VMs left running for debugging or redeploy iteration")
            return

        # Stop cluster via CLI
        if not self.config.local:
            logger.info("Stopping remote cluster...")
            try:
                result = _run_iris("cluster", "stop", config_path=self.config.config_path)
                for line in result.stdout.splitlines():
                    logger.info("  %s", line)
                logger.info("Remote cluster stopped")
            except Exception as e:
                logger.warning("Error stopping remote cluster: %s", e)

        logger.info("Done")


# =============================================================================
# CLI
# =============================================================================


@click.command()
@click.option(
    "--config",
    "config_path",
    default=str(DEFAULT_CONFIG_PATH),
    type=click.Path(exists=True, path_type=Path),
    show_default=True,
    help="Path to cluster config YAML",
)
@click.option(
    "--boot-timeout",
    "boot_timeout_seconds",
    default=DEFAULT_BOOT_TIMEOUT,
    show_default=True,
    help="Timeout in seconds for cluster startup and connection",
)
@click.option(
    "--job-timeout",
    "job_timeout_seconds",
    default=DEFAULT_JOB_TIMEOUT,
    show_default=True,
    help="Per-job timeout in seconds",
)
@click.option(
    "--mode",
    type=click.Choice(["full", "keep", "redeploy"]),
    default="full",
    show_default=True,
    help="Execution mode: 'full' (clean start + teardown), 'keep' (clean start + keep VMs), 'redeploy' (reuse VMs)",
)
@click.option(
    "--local",
    is_flag=True,
    help="Run locally without GCP (in-process controller and workers)",
)
def main(
    config_path: Path,
    boot_timeout_seconds: int,
    job_timeout_seconds: int,
    mode: str,
    local: bool,
):
    """Run Iris cluster autoscaling smoke test.

    Automatically detects whether the config uses GPU or TPU accelerators and
    runs the appropriate test suite. GPU configs run multi-GPU device checks;
    TPU configs run JAX TPU and coscheduling tests.

    Examples:

        # Basic smoke test (uses examples/smoke.yaml by default)
        uv run python scripts/smoke-test.py

        # CoreWeave GPU smoke test
        uv run python scripts/smoke-test.py --config examples/coreweave.yaml

        # Keep VMs running after test
        uv run python scripts/smoke-test.py --mode keep

        # Redeploy mode: reuse existing VMs (much faster for iteration)
        uv run python scripts/smoke-test.py --mode redeploy
    """
    config_path = config_path.resolve()

    accelerator = detect_accelerator(config_path)
    logger.info("Detected accelerator: %s (%s)", accelerator.label(), accelerator.device_type)

    config = SmokeTestConfig(
        config_path=config_path,
        accelerator=accelerator,
        boot_timeout_seconds=boot_timeout_seconds,
        job_timeout_seconds=job_timeout_seconds,
        mode=mode,  # type: ignore
        local=local,
    )

    runner = SmokeTestRunner(config)
    success = runner.run()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
