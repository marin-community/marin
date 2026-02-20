#!/usr/bin/env python3
# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris cluster autoscaling smoke test.

This script provides end-to-end validation of an Iris cluster by exercising
the `iris` CLI for cluster lifecycle and `IrisClient` for job submission:

1. Cleans up existing resources (`iris cluster debug cleanup`)
2. Starts the cluster (`iris cluster start`)
3. Establishes connection (`iris cluster dashboard` or local address)
4. Submits TPU jobs to exercise autoscaling via IrisClient
5. Logs results to stdout and structured log directory
6. Cleans up on success/failure/interrupt (`iris cluster stop`, cleanup)

Usage:
    # Basic smoke test (uses examples/smoke.yaml, logs to logs/smoke-test-{timestamp}/)
    uv run python scripts/smoke-test.py

    # With custom config and log directory
    uv run python scripts/smoke-test.py --config examples/marin.yaml \\
        --log-dir /path/to/logs

    # Custom timeout (45 min) for slow environments
    uv run python scripts/smoke-test.py --timeout 2700

    # Keep cluster running on failure for debugging
    uv run python scripts/smoke-test.py --mode keep

    # Redeploy mode: reuse existing VMs (much faster for iteration)
    uv run python scripts/smoke-test.py --mode redeploy
"""

import logging
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, TextIO

import click

from iris.client import IrisClient
from iris.cluster.types import (
    Constraint,
    CoschedulingConfig,
    Entrypoint,
    EnvironmentSpec,
    ResourceSpec,
    region_constraint,
    tpu_device,
)
from iris.rpc import cluster_pb2

IRIS_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG_PATH = IRIS_ROOT / "examples" / "smoke.yaml"

DEFAULT_JOB_TIMEOUT = 300  # 5 minutes; TPU slices are pre-warmed by earlier tests


# =============================================================================
# CLI Helpers
# =============================================================================


DEFAULT_CLI_TIMEOUT = 900  # 15 minutes; generous limit for image builds and cluster operations


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
    logging.info("Running (timeout=%ds): %s", timeout, " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(IRIS_ROOT), timeout=timeout)
    if result.returncode != 0:
        logging.error(
            "Command failed (exit %d): %s\nstdout: %s\nstderr: %s",
            result.returncode,
            " ".join(cmd),
            result.stdout,
            result.stderr,
        )
        result.check_returncode()
    return result


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
    logging.info("Starting background: %s", " ".join(cmd))

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


def _wait_for_line(bg: BackgroundProc, pattern: str, timeout: float = 300, drain_to: Path | None = None) -> str:
    """Read stdout from a background subprocess until a line matches `pattern`.

    Uses a reader thread so the timeout is enforced even if readline blocks on a
    partial line.  After the match is found, a drain thread continues consuming
    stdout to prevent the pipe buffer from filling and blocking the subprocess.

    Args:
        bg: Background process whose stdout is PIPE.
        pattern: Regex pattern to search for in each line.
        timeout: Maximum seconds to wait for a match.
        drain_to: If set, remaining stdout (including pre-match lines) is written
            to this file.  Otherwise drained and discarded.

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
    matched_line: str | None = None
    pre_match_lines: list[str] = []

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
        logging.info("  [bg] %s", line)
        pre_match_lines.append(line)
        if compiled.search(line):
            matched_line = line
            break

    if matched_line is None:
        raise TimeoutError(f"Timed out waiting for pattern '{pattern}' after {timeout}s")

    # Drain remaining stdout so the pipe never fills and blocks the subprocess.
    drain_file: TextIO | None = None
    if drain_to is not None:
        drain_file = open(drain_to, "w")
        bg.owned_fds.append(drain_file)
        for prev in pre_match_lines:
            drain_file.write(prev + "\n")

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
            if drain_file is not None:
                drain_file.write(remaining_line + "\n")
                drain_file.flush()

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
    logging.info("Terminating %s (pid=%d)...", name, proc.pid)
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        logging.warning("Process %s did not exit after SIGTERM, sending SIGKILL", name)
        proc.kill()
        proc.wait(timeout=5.0)


# =============================================================================
# Log Infrastructure
# =============================================================================


@dataclass
class LogArtifact:
    path: Path
    description: str


class LogTree:
    """Tracks log artifacts created during a smoke test run."""

    def __init__(self, root: Path):
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)
        self._artifacts: list[LogArtifact] = []

    @property
    def root(self) -> Path:
        return self._root

    def get_writer(self, name: str, description: str) -> Path:
        """Register and return path for a log artifact. Creates parent dirs."""
        path = self._root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        self._artifacts.append(LogArtifact(path=path, description=description))
        return path

    def get_dir(self, name: str, description: str) -> Path:
        """Register and return path for a directory artifact."""
        path = self._root / name
        path.mkdir(parents=True, exist_ok=True)
        self._artifacts.append(LogArtifact(path=path, description=description))
        return path

    def summary_lines(self) -> list[str]:
        lines = ["## Artifacts", ""]
        for artifact in self._artifacts:
            rel = artifact.path.relative_to(self._root)
            if artifact.path.is_dir():
                count = sum(1 for _ in artifact.path.iterdir()) if artifact.path.exists() else 0
                status = f"{count} files"
            else:
                exists = artifact.path.exists()
                size = artifact.path.stat().st_size if exists else 0
                status = f"{size:,} bytes" if exists else "not created"
            lines.append(f"- **{artifact.description}:** `{rel}` ({status})")
        return lines


# =============================================================================
# Test Job Definitions
# =============================================================================


def _hello_tpu_job():
    """Simple job that prints and returns."""
    print("Hello from TPU!")
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
        resources=RS(device=tpu_device("v5litepod-16")),
    )
    child.wait(timeout=300, raise_on_failure=True)
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
# Logging Infrastructure
# =============================================================================


class SmokeTestLogger:
    """Dual-output logger with timestamps and elapsed time.

    Writes both to stdout (with timestamps) and to a markdown file with
    structured sections for easy post-mortem analysis.
    """

    def __init__(self, log_tree: LogTree):
        self._start_time = time.monotonic()
        self._start_datetime = datetime.now()
        self._log_tree = log_tree
        summary_path = log_tree.get_writer("summary.md", "Execution summary")
        self._file: TextIO = open(summary_path, "w")

    def close(self):
        self._file.close()

    def write_header(self, config: "SmokeTestConfig"):
        """Write the markdown document header with run configuration."""
        self._file.write("# Smoke Test Execution Log\n\n")
        self._file.write(f"**Started:** {self._start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        self._file.write(f"**Config:** `{config.config_path}`\n\n")
        self._file.write(f"**TPU type:** `{config.tpu_type}`\n\n")
        self._file.write(f"**Log directory:** `{config.log_dir}`\n\n")
        self._file.write("---\n\n")
        self._file.flush()

    def write_artifacts(self, log_tree: LogTree):
        """Write the artifacts section from the log tree."""
        self._file.write("\n---\n\n")
        for line in log_tree.summary_lines():
            self._file.write(line + "\n")
        self._file.flush()

    def log(self, message: str, level: str = "INFO"):
        now = datetime.now()
        elapsed = time.monotonic() - self._start_time
        line = f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] [{elapsed:8.1f}s] [{level}] {message}"
        print(line, flush=True)
        self._file.write(line + "\n")
        self._file.flush()

    def section(self, title: str):
        """Write a markdown section header."""
        self.log("")
        self._file.write(f"\n## {title}\n\n")
        self._file.flush()
        print("=" * 60, flush=True)
        print(f" {title}", flush=True)
        print("=" * 60, flush=True)
        self.log("")


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SmokeTestConfig:
    """Configuration for the smoke test."""

    config_path: Path
    log_dir: Path
    timeout_seconds: int = 1800  # 30 min total
    job_timeout_seconds: int = DEFAULT_JOB_TIMEOUT
    tpu_type: str = "v5litepod-16"
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
        self.log_tree = LogTree(config.log_dir)
        self.logger = SmokeTestLogger(self.log_tree)
        self._task_logs_dir = self.log_tree.get_dir("task-logs", "Task logs from each job/task")
        self._interrupted = False
        self._deadline: float | None = None
        self._results: list[TestResult] = []
        # Unique run ID to avoid job name collisions with previous runs
        self._run_id = datetime.now().strftime("%H%M%S")
        # Background subprocesses to clean up
        self._background_procs: list[BackgroundProc] = []

    def run(self) -> bool:
        """Run the smoke test. Returns True if all tests pass."""
        _configure_logging()

        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)
        self._deadline = time.monotonic() + self.config.timeout_seconds

        try:
            self._print_header()

            # Cluster startup: always run in the background so both local
            # configs (which block on controller.wait()) and remote configs
            # work identically.
            controller_url: str | None = None

            if self.config.mode != "redeploy":
                if self.config.mode in ("full", "keep") and not self.config.local:
                    self.logger.section("PHASE 0: Clean Start")
                    self._cleanup_existing()
                    if self._interrupted or self._check_deadline():
                        return False

                self.logger.section("Starting Cluster")
                cluster_address = self._start_cluster()
                if self._interrupted or self._check_deadline():
                    return False

                if self.config.local:
                    controller_url = cluster_address

            if controller_url is None:
                # Remote mode (or redeploy): establish tunnel to get a
                # localhost URL regardless of whether we started the cluster
                # above or it was already running.
                self.logger.section("Connecting to Cluster")
                controller_url = self._connect_remote()

            # Start controller log streaming (remote GCP only)
            if not self.config.local:
                self._start_log_streaming()

            # Run tests
            self.logger.section("Running Tests")
            self._run_tests(controller_url)

            # Results
            self.logger.section("Results Summary")
            success = self._print_results()
            return success

        except Exception as e:
            self.logger.log(f"FATAL ERROR: {e}", level="ERROR")
            return False

        finally:
            self._cleanup()
            self.logger.close()

    def _handle_interrupt(self, _signum: int, _frame: object):
        self.logger.log("Interrupted! Cleaning up...", level="WARN")
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        self._interrupted = True

    def _check_deadline(self) -> bool:
        """Returns True if the global deadline has passed. Sets interrupted flag."""
        if self._deadline is not None and time.monotonic() > self._deadline:
            self.logger.log(
                f"Global timeout ({self.config.timeout_seconds}s) exceeded!",
                level="ERROR",
            )
            self._interrupted = True
            return True
        return False

    def _print_header(self):
        self.logger.write_header(self.config)
        self.logger.log("")
        self.logger.log("=" * 60)
        self.logger.log(" IRIS CLUSTER SMOKE TEST")
        self.logger.log("=" * 60)
        self.logger.log("")
        self.logger.log(f"Config: {self.config.config_path}")
        self.logger.log(f"Timeout: {self.config.timeout_seconds}s")
        self.logger.log(f"TPU type: {self.config.tpu_type}")
        self.logger.log(f"Log directory: {self.config.log_dir}")
        self.logger.log(f"Local: {self.config.local}")

    # ----- Cluster lifecycle via CLI -----

    def _cleanup_existing(self):
        """Delete existing iris resources via `iris cluster debug cleanup --no-dry-run`."""
        self.logger.log("Cleaning up existing resources...")
        try:
            result = _run_iris("cluster", "debug", "cleanup", "--no-dry-run", config_path=self.config.config_path)
            for line in result.stdout.splitlines():
                self.logger.log(f"  {line}")
            self.logger.log("Cleanup complete")
        except subprocess.CalledProcessError as e:
            self.logger.log(f"Cleanup failed: {e.stderr}", level="ERROR")
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

        self.logger.log("Starting cluster (background)...")
        bg = _run_iris_background(*args, config_path=self.config.config_path)
        bg.name = "cluster-start"
        self._background_procs.append(bg)

        drain_path = self.log_tree.get_writer("cluster-start.log", "Cluster start stdout")
        line = _wait_for_line(bg, r"Controller started at", timeout=DEFAULT_CLI_TIMEOUT, drain_to=drain_path)
        address = _parse_address_from_line(line, "Controller started at")
        self.logger.log(f"Controller started at: {address}")
        return address

    def _connect_remote(self) -> str:
        """Establish tunnel via `iris cluster dashboard` as background subprocess.

        Parses the controller URL from the output line "Controller RPC: {url}".
        The subprocess blocks to keep the tunnel alive.
        """
        self.logger.log("Starting dashboard tunnel...")
        bg = _run_iris_background("cluster", "dashboard", config_path=self.config.config_path)
        bg.name = "dashboard-tunnel"
        self._background_procs.append(bg)

        drain_path = self.log_tree.get_writer("dashboard-tunnel.log", "Dashboard tunnel stdout")
        line = _wait_for_line(bg, r"Controller RPC:", timeout=120, drain_to=drain_path)
        controller_url = _parse_address_from_line(line, "Controller RPC:")
        self.logger.log(f"Controller URL: {controller_url}")
        return controller_url

    def _start_log_streaming(self):
        """Stream controller logs via `iris cluster debug logs --follow` as background subprocess."""
        log_file = self.log_tree.get_writer("controller-logs.txt", "Controller docker logs")
        bg = _run_iris_background(
            "cluster",
            "debug",
            "logs",
            "--follow",
            config_path=self.config.config_path,
            stdout_path=log_file,
        )
        bg.name = "controller-logs"
        self._background_procs.append(bg)
        self.logger.log("Started controller log streaming")

    # ----- Monitoring via CLI -----

    def _log_autoscaler_status(self):
        """Log current autoscaler state via `iris cluster debug autoscaler-status`."""
        try:
            result = _run_iris("cluster", "debug", "autoscaler-status", config_path=self.config.config_path)
            for line in result.stdout.splitlines():
                self.logger.log(f"  {line}")
        except Exception as e:
            self.logger.log(f"  (Could not fetch autoscaler status: {e})", level="WARN")

    # ----- Test execution -----

    def _run_tests(self, controller_url: str):
        """Run test jobs against the cluster."""
        client = IrisClient.remote(controller_url, workspace=IRIS_ROOT)

        # Test 1: Simple TPU job
        self.logger.log(f"[Test 1/6] Simple TPU job ({self.config.tpu_type})")
        result = self._run_simple_tpu_job(client)
        self._results.append(result)
        self._log_autoscaler_status()

        if self._interrupted or self._check_deadline():
            return

        # Test 2: Concurrent TPU jobs
        self.logger.log(f"[Test 2/6] Concurrent TPU jobs (3x {self.config.tpu_type})")
        result = self._run_concurrent_tpu_jobs(client)
        self._results.append(result)
        self._log_autoscaler_status()

        if self._interrupted or self._check_deadline():
            return

        # Test 3: Coscheduled multi-task job
        self.logger.log(f"[Test 3/6] Coscheduled multi-task job ({self.config.tpu_type})")
        result = self._run_coscheduled_job(client)
        self._results.append(result)
        self._log_autoscaler_status()

        if self._interrupted or self._check_deadline():
            return

        # Test 4: JAX TPU job - validates JAX can initialize and use TPU
        self.logger.log(f"[Test 4/6] JAX TPU job ({self.config.tpu_type})")
        result = self._run_jax_tpu_job(client)
        self._results.append(result)
        self._log_autoscaler_status()

        if self._interrupted or self._check_deadline():
            return

        # Test 5: Region-constrained job - validates constraint-based routing
        self.logger.log(f"[Test 5/6] Region-constrained job ({self.config.tpu_type})")
        result = self._run_region_constrained_job(client)
        self._results.append(result)
        self._log_autoscaler_status()

        if self._interrupted or self._check_deadline():
            return

        # Test 6: Nested constraint propagation - child inherits parent region
        self.logger.log(f"[Test 6/6] Nested constraint propagation ({self.config.tpu_type})")
        result = self._run_nested_constraint_job(client)
        self._results.append(result)
        self._log_autoscaler_status()

    def _print_task_logs_on_failure(self, job, max_lines: int = 50):
        """Print task logs when a job fails, to show build errors."""
        try:
            for task in job.tasks():
                logs = task.logs(max_lines=max_lines)
                if logs:
                    self.logger.log(f"  Task {task.task_index} logs (last {len(logs)} lines):")
                    for entry in logs[-max_lines:]:
                        self.logger.log(f"    [{entry.source}] {entry.data}")
        except Exception as e:
            self.logger.log(f"  Failed to fetch task logs: {e}", level="WARN")

    def _write_task_logs(self, job, test_name: str):
        """Persist task logs for a job to the log directory."""
        job_dir = self._task_logs_dir / job.job_id.to_safe_token()
        job_dir.mkdir(parents=True, exist_ok=True)
        for task in job.tasks():
            task_file = job_dir / f"task-{task.task_index}.log"
            try:
                entries = task.logs()
                with open(task_file, "w") as handle:
                    handle.write(f"test_name={test_name}\n")
                    handle.write(f"job_id={job.job_id}\n")
                    handle.write(f"task_index={task.task_index}\n")
                    handle.write(f"task_state={cluster_pb2.TaskState.Name(task.state)}\n\n")
                    for entry in entries:
                        handle.write(f"[{entry.timestamp}] [{entry.source}] {entry.data}\n")
            except Exception as e:
                self.logger.log(
                    f"  Failed to write logs for job {job.job_id} task {task.task_index}: {e}",
                    level="WARN",
                )

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
            self.logger.log(f"  Job submitted: {job.job_id}")

            status = job.wait(timeout=job_timeout, raise_on_failure=False, stream_logs=True)
            duration = time.monotonic() - start
            self._write_task_logs(job, test_name)

            if status.state == cluster_pb2.JOB_STATE_SUCCEEDED:
                self.logger.log(f"  [PASS] Completed in {duration:.1f}s")
                return TestResult(test_name, True, f"Completed in {duration:.1f}s", duration)
            else:
                state_name = cluster_pb2.JobState.Name(status.state)
                self.logger.log(f"  [FAIL] Job ended with state {state_name}", level="ERROR")
                self._print_task_logs_on_failure(job)
                return TestResult(test_name, False, f"State: {state_name}, error: {status.error}", duration)

        except TimeoutError:
            duration = time.monotonic() - start
            self.logger.log(f"  [FAIL] Timed out after {job_timeout}s", level="ERROR")
            try:
                self._write_task_logs(job, test_name)
            except Exception:
                pass
            return TestResult(test_name, False, f"Timed out after {job_timeout}s", duration)

    def _submit_and_wait_multiple(
        self,
        client: IrisClient,
        jobs_config: list[tuple[Entrypoint, str, ResourceSpec]],
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
            self.logger.log(f"  Job submitted: {job.job_id}")

        failed_jobs = []
        for job in jobs:
            status = job.wait(timeout=self.config.job_timeout_seconds, raise_on_failure=False, stream_logs=True)
            if status.state != cluster_pb2.JOB_STATE_SUCCEEDED:
                state_name = cluster_pb2.JobState.Name(status.state)
                failed_jobs.append(f"{job.job_id}: {state_name}")
            self._write_task_logs(job, "Concurrent TPU jobs (3x)")

        return time.monotonic() - start, failed_jobs

    def _run_simple_tpu_job(self, client: IrisClient) -> TestResult:
        """Run a simple TPU job that just prints and returns."""
        return self._run_job_test(
            client=client,
            test_name=f"Simple TPU job ({self.config.tpu_type})",
            entrypoint=Entrypoint.from_callable(_hello_tpu_job),
            job_name=f"smoke-simple-{self._run_id}",
            resources=ResourceSpec(device=tpu_device(self.config.tpu_type)),
        )

    def _run_concurrent_tpu_jobs(self, client: IrisClient) -> TestResult:
        """Submit 3 concurrent TPU jobs to test parallel provisioning and queueing."""
        resources = ResourceSpec(device=tpu_device(self.config.tpu_type))
        jobs_config = [
            (Entrypoint.from_callable(_quick_task_job, i), f"smoke-concurrent-{self._run_id}-{i}", resources)
            for i in range(3)
        ]

        try:
            duration, failed_jobs = self._submit_and_wait_multiple(client, jobs_config)

            if not failed_jobs:
                self.logger.log(f"  [PASS] All 3 jobs completed in {duration:.1f}s")
                return TestResult("Concurrent TPU jobs (3x)", True, f"All completed in {duration:.1f}s", duration)
            else:
                self.logger.log(f"  [FAIL] Some jobs failed: {', '.join(failed_jobs)}", level="ERROR")
                return TestResult("Concurrent TPU jobs (3x)", False, f"Failed: {', '.join(failed_jobs)}", duration)

        except TimeoutError:
            self.logger.log("  [FAIL] Timed out waiting for jobs", level="ERROR")
            return TestResult(
                "Concurrent TPU jobs (3x)", False, f"Timed out after {self.config.job_timeout_seconds}s", 0.0
            )

    def _run_coscheduled_job(self, client: IrisClient) -> TestResult:
        """Run a coscheduled multi-task job on TPU workers."""
        return self._run_job_test(
            client=client,
            test_name="Coscheduled multi-task job",
            entrypoint=Entrypoint.from_callable(_distributed_work_job),
            job_name=f"smoke-coscheduled-{self._run_id}",
            resources=ResourceSpec(
                device=tpu_device(self.config.tpu_type),
            ),
            coscheduling=CoschedulingConfig(group_by="tpu-name"),
            replicas=4,
        )

    def _run_jax_tpu_job(self, client: IrisClient) -> TestResult:
        """Run a JAX TPU job that initializes JAX and exercises the TPU.

        Uses coscheduling with replicas=4 because multi-host TPU pods (e.g. v5litepod-16)
        require all hosts to run JAX simultaneously for collective initialization.
        """
        return self._run_job_test(
            client=client,
            test_name=f"JAX TPU job ({self.config.tpu_type})",
            entrypoint=Entrypoint.from_callable(_jax_tpu_job),
            job_name=f"smoke-jax-tpu-{self._run_id}",
            resources=ResourceSpec(
                device=tpu_device(self.config.tpu_type),
            ),
            environment=EnvironmentSpec(pip_packages=["jax[tpu]"]),
            coscheduling=CoschedulingConfig(group_by="tpu-name"),
            replicas=4,
            timeout=300,
        )

    def _run_region_constrained_job(self, client: IrisClient) -> TestResult:
        """Run a job with an explicit region constraint.

        Validates that constraint-based routing places the job on workers
        whose attributes match the requested region.
        """
        return self._run_job_test(
            client=client,
            test_name="Region-constrained job",
            entrypoint=Entrypoint.from_callable(_hello_tpu_job),
            job_name=f"smoke-region-{self._run_id}",
            resources=ResourceSpec(device=tpu_device(self.config.tpu_type)),
            constraints=[region_constraint(["europe-west4"])],
        )

    def _run_nested_constraint_job(self, client: IrisClient) -> TestResult:
        """Submit a parent job with a region constraint whose body submits a child.

        The parent uses IrisClient (via iris_ctx()) to submit a child without
        explicit constraints. The child inherits the parent's region constraint
        and asserts it via JobInfo.constraints.
        """
        return self._run_job_test(
            client=client,
            test_name="Nested constraint propagation",
            entrypoint=Entrypoint.from_callable(_assert_region_child, "europe-west4"),
            job_name=f"smoke-nested-{self._run_id}",
            resources=ResourceSpec(device=tpu_device(self.config.tpu_type)),
            constraints=[region_constraint(["europe-west4"])],
        )

    # ----- Results and cleanup -----

    def _print_results(self) -> bool:
        """Print final results and return True if all passed."""
        all_passed = True
        total_duration = 0.0

        for result in self._results:
            status = "PASS" if result.passed else "FAIL"
            self.logger.log(f"  [{status}] {result.name}: {result.message}")
            total_duration += result.duration_seconds
            if not result.passed:
                all_passed = False

        self.logger.log("")
        passed_count = sum(1 for r in self._results if r.passed)
        total_count = len(self._results)

        if all_passed:
            self.logger.log(f"Results: {passed_count}/{total_count} tests passed in {total_duration:.1f}s")
        else:
            self.logger.log(f"Results: {passed_count}/{total_count} tests passed in {total_duration:.1f}s", level="WARN")

        self.logger.write_artifacts(self.log_tree)
        return all_passed

    def _cleanup(self):
        """Clean up cluster resources and background subprocesses.

        Each cleanup phase is wrapped in its own try/except so that a failure
        in one phase (e.g. terminating a background process) does not prevent
        the cluster from being torn down.
        """
        self.logger.section("CLEANUP")

        # Terminate all background subprocesses and close their owned file handles.
        # Failures here must not prevent cluster teardown below.
        for bg in self._background_procs:
            try:
                bg.terminate()
            except Exception as e:
                self.logger.log(f"Error terminating {bg.name}: {e}", level="WARN")
        self._background_procs.clear()
        self.logger.log("Stopped background processes")

        # In redeploy mode, skip VM cleanup to preserve VMs for next run
        if self.config.mode == "redeploy":
            self.logger.log("Redeploy mode: keeping VMs running for next iteration")
            return

        if self.config.mode == "keep":
            self.logger.log("Skipping cleanup (--mode keep)")
            self.logger.log("VMs left running for debugging or redeploy iteration")
            return

        # Stop cluster via CLI
        if not self.config.local:
            self.logger.log("Stopping remote cluster...")
            try:
                result = _run_iris("cluster", "stop", config_path=self.config.config_path)
                for line in result.stdout.splitlines():
                    self.logger.log(f"  {line}")
                self.logger.log("Remote cluster stopped")
            except Exception as e:
                self.logger.log(f"Error stopping remote cluster: {e}", level="WARN")

            # Final resource cleanup — always attempted even if cluster stop failed
            self.logger.log("Running final resource cleanup...")
            try:
                result = _run_iris("cluster", "debug", "cleanup", "--no-dry-run", config_path=self.config.config_path)
                for line in result.stdout.splitlines():
                    self.logger.log(f"  {line}")
            except Exception as e:
                self.logger.log(f"Error during final cleanup: {e}", level="WARN")

        self.logger.log("Done")


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
    "--timeout",
    "timeout_seconds",
    default=1800,
    show_default=True,
    help="Total timeout in seconds",
)
@click.option(
    "--job-timeout",
    "job_timeout_seconds",
    default=DEFAULT_JOB_TIMEOUT,
    show_default=True,
    help="Per-job timeout in seconds",
)
@click.option(
    "--log-dir",
    type=click.Path(path_type=Path),
    help="Log directory path (default: logs/smoke-test-{timestamp})",
)
@click.option(
    "--tpu-type",
    default="v5litepod-16",
    show_default=True,
    help="TPU type for test jobs",
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
    timeout_seconds: int,
    job_timeout_seconds: int,
    log_dir: Path | None,
    tpu_type: str,
    mode: str,
    local: bool,
):
    """Run Iris cluster autoscaling smoke test.

    This script starts a cluster, submits TPU jobs to exercise autoscaling,
    and validates that everything works correctly. On completion (or failure),
    the cluster is cleaned up unless --mode keep or --mode redeploy is specified.

    Examples:

        # Basic smoke test (uses examples/smoke.yaml by default)
        uv run python scripts/smoke-test.py

        # Keep VMs running after test
        uv run python scripts/smoke-test.py --mode keep

        # Redeploy mode: reuse existing VMs (much faster for iteration)
        uv run python scripts/smoke-test.py --mode redeploy

        # With custom config and log directory
        uv run python scripts/smoke-test.py --config examples/marin.yaml \\
            --log-dir /path/to/logs
    """
    # Create default log directory with timestamp if not provided
    if log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = Path("logs") / f"smoke-test-{timestamp}"

    config = SmokeTestConfig(
        config_path=config_path,
        timeout_seconds=timeout_seconds,
        job_timeout_seconds=job_timeout_seconds,
        log_dir=log_dir,
        tpu_type=tpu_type,
        mode=mode,  # type: ignore
        local=local,
    )

    runner = SmokeTestRunner(config)
    success = runner.run()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
