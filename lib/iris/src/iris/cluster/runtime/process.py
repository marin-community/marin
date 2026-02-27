# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Subprocess-based container runtime for local execution.

ProcessRuntime implements ContainerRuntime using subprocess.Popen instead of
Docker, enabling lightweight local testing. ProcessContainerHandle runs commands
directly on the host, handling environment setup, mount path remapping, and
log capture.

Lifecycle management includes:
- Automatic cleanup on interpreter shutdown via atexit
- PR_SET_PDEATHSIG on Linux for orphan prevention
- Process group termination on Unix platforms
"""

from __future__ import annotations

import atexit
import ctypes
import ctypes.util
import logging
import os
import select
import signal
import shutil
import subprocess
import sys
import threading
import uuid
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from iris.cluster.runtime.env import build_device_env_vars
from iris.cluster.runtime.profile import (
    build_memray_attach_cmd,
    build_memray_transform_cmd,
    build_pyspy_cmd,
    resolve_cpu_spec,
    resolve_memory_spec,
)
from iris.cluster.runtime.types import ContainerConfig, ContainerStats, ContainerStatus, RuntimeLogReader
from iris.cluster.worker.worker_types import LogLine
from iris.managed_thread import ManagedThread, get_thread_container
from iris.rpc import cluster_pb2

logger = logging.getLogger(__name__)


# =============================================================================
# Subprocess cleanup on parent exit
# =============================================================================
# PR_SET_PDEATHSIG only works on Linux; on macOS, orphaned subprocesses survive
# the parent. We use a module-level atexit handler with weak references to
# ensure all ProcessRuntime instances kill their subprocesses when the
# Python interpreter shuts down (normal exit, sys.exit, unhandled exceptions).
# This does NOT cover SIGKILL of the parent.

_active_runtimes: weakref.WeakSet[ProcessRuntime] = weakref.WeakSet()


def _cleanup_all_runtimes() -> None:
    for runtime in list(_active_runtimes):
        runtime.cleanup()


atexit.register(_cleanup_all_runtimes)


# =============================================================================
# Process management utilities
# =============================================================================


def _read_proc_memory_mb(pid: int) -> int | None:
    """Read RSS memory usage for a process, in megabytes.

    On Linux, reads /proc/{pid}/statm directly (no external dependencies).
    On macOS, shells out to `ps` since /proc is not available.
    Returns None if the process doesn't exist or the read fails.
    """
    if sys.platform == "linux":
        try:
            with open(f"/proc/{pid}/statm") as f:
                pages = int(f.read().split()[1])  # resident set size in pages
            return (pages * 4096) // (1024 * 1024)
        except (FileNotFoundError, ProcessLookupError, ValueError, IndexError):
            return None
    else:
        try:
            result = subprocess.run(
                ["ps", "-o", "rss=", "-p", str(pid)],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return None
            rss_kb = int(result.stdout.strip())
            return rss_kb // 1024
        except (subprocess.TimeoutExpired, ValueError, OSError):
            return None


def _read_proc_cpu_percent(pid: int, prev_total: float, prev_utime: float) -> tuple[int, float, float]:
    """Read CPU usage percentage for a process since the last call.

    On Linux, computes delta CPU usage from /proc/{pid}/stat and /proc/stat.
    On macOS, returns 0 since /proc is not available.
    Returns (cpu_percent, new_total, new_utime).
    """
    if sys.platform != "linux":
        return (0, prev_total, prev_utime)

    try:
        with open(f"/proc/{pid}/stat") as f:
            fields = f.read().split()
        # utime (field 14) + stime (field 15), 1-indexed => 0-indexed 13, 14
        proc_time = int(fields[13]) + int(fields[14])

        with open("/proc/stat") as f:
            cpu_line = f.readline()
        total_time = sum(int(x) for x in cpu_line.split()[1:])

        dt = total_time - prev_total
        dp = proc_time - prev_utime
        if dt <= 0 or prev_total == 0:
            return (0, total_time, proc_time)

        cpu_pct = int((dp / dt) * 100)
        return (cpu_pct, total_time, proc_time)
    except (FileNotFoundError, ProcessLookupError, ValueError, IndexError):
        return (0, prev_total, prev_utime)


def set_pdeathsig_preexec():
    """Use prctl(PR_SET_PDEATHSIG, SIGKILL) to kill subprocess if parent dies.

    This is a Linux-specific feature that ensures container processes are
    automatically killed if the worker process dies unexpectedly. On other
    platforms, this is a no-op.
    """
    if sys.platform == "linux":
        PR_SET_PDEATHSIG = 1
        try:
            libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
            if libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL) != 0:
                errno = ctypes.get_errno()
                logger.warning(f"Failed to set parent death signal: errno {errno}")
        except Exception as e:
            logger.debug(f"Could not set parent death signal: {e}")


# =============================================================================
# Process-based container runtime
# =============================================================================


@dataclass
class ProcessContainer:
    """Container execution via subprocess (not thread).

    Uses subprocess.Popen to run commands, enabling hard termination and
    proper log capture. The command is provided at construction time with
    mount paths already remapped for local execution.
    """

    config: ContainerConfig
    command: list[str]  # Pre-computed command with remapped paths
    _process: subprocess.Popen | None = field(default=None, repr=False)
    _log_thread: ManagedThread | None = field(default=None, repr=False)
    _running: bool = False
    _exit_code: int | None = None
    _error: str | None = None
    _logs: list[LogLine] = field(default_factory=list)

    def start(self):
        """Start container as subprocess and begin streaming logs."""
        self._running = True
        cmd = list(self.command)
        if cmd and cmd[0] in {"python", "python3"}:
            # For Entrypoint.from_command("python", ...) — rewrite to the
            # current interpreter since "python" may not be on PATH.
            cmd = [sys.executable, *cmd[1:]]

        try:
            logger.info(
                "Starting local container (task_id=%s, job_id=%s, cmd=%s)",
                self.config.task_id,
                self.config.job_id,
                cmd,
            )
            env = {**os.environ, **self.config.env}
            iris_root = Path(__file__).resolve().parents[4]
            extra_paths = [str(iris_root / "src"), str(iris_root)]
            existing = env.get("PYTHONPATH", "")
            prefix = os.pathsep.join(p for p in extra_paths if p not in existing.split(os.pathsep))
            env["PYTHONPATH"] = f"{prefix}{os.pathsep}{existing}" if existing else prefix

            # Use process groups on Unix for clean termination
            # Set PR_SET_PDEATHSIG on Linux for automatic cleanup if parent dies
            popen_kwargs: dict[str, object] = {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "text": True,
                "env": env,
                "bufsize": 1,  # Line buffered
            }

            if sys.platform != "win32":
                # Create new process group for clean termination
                popen_kwargs["start_new_session"] = True
                # Set up automatic termination if parent dies (Linux only)
                popen_kwargs["preexec_fn"] = set_pdeathsig_preexec

            self._process = subprocess.Popen(cmd, **popen_kwargs)

            # Spawn thread to stream logs asynchronously
            name_suffix = self.config.task_id or self.config.job_id or "unnamed"
            self._log_thread = get_thread_container().spawn(
                target=self._stream_logs,
                name=f"logs-{name_suffix}",
            )
        except Exception as e:
            self._error = str(e)
            self._exit_code = 1
            self._running = False
            logger.exception("Failed to start container")

    def _stream_logs(self, stop_event: threading.Event):
        """Stream stdout/stderr from subprocess to log buffer.

        Runs in a separate thread to avoid blocking. Uses select() for
        non-blocking reads with timeout to respect stop_event.
        """
        if not self._process:
            return

        try:
            while self._process.poll() is None:
                if stop_event.is_set():
                    break

                # Non-blocking read with timeout
                assert self._process.stdout is not None
                assert self._process.stderr is not None
                ready, _, _ = select.select([self._process.stdout, self._process.stderr], [], [], 0.1)

                for stream in ready:
                    line = stream.readline()
                    if line:
                        source = "stdout" if stream == self._process.stdout else "stderr"
                        self._logs.append(
                            LogLine(
                                timestamp=datetime.now(timezone.utc),
                                source=source,
                                data=line.rstrip(),
                            )
                        )

            # Process exited - drain remaining output
            if self._process.stdout:
                for line in self._process.stdout:
                    self._logs.append(
                        LogLine(
                            timestamp=datetime.now(timezone.utc),
                            source="stdout",
                            data=line.rstrip(),
                        )
                    )
            if self._process.stderr:
                for line in self._process.stderr:
                    self._logs.append(
                        LogLine(
                            timestamp=datetime.now(timezone.utc),
                            source="stderr",
                            data=line.rstrip(),
                        )
                    )

            self._exit_code = self._process.returncode
            self._running = False

        except Exception as e:
            logger.exception("Error streaming logs from container")
            self._error = str(e)
            self._exit_code = 1
            self._running = False

    def kill(self):
        """Hard kill the subprocess immediately via SIGKILL.

        On Unix: kills the entire process group to ensure all children are terminated.
        On Windows: kills just the process.
        """
        if self._process and self._process.poll() is None:
            logger.debug("Killing container process %s", self._process.pid)
            try:
                if sys.platform == "win32":
                    self._process.kill()
                else:
                    # Kill entire process group
                    os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
            except ProcessLookupError:
                # Process already terminated
                pass
            except Exception as e:
                logger.warning("Failed to kill process %s: %s", self._process.pid, e)
                # Fall back to just killing the process itself
                self._process.kill()

            try:
                self._process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                logger.warning("Process did not terminate after SIGKILL")

        self._running = False
        if self._exit_code is None:
            self._exit_code = 137  # 128 + SIGKILL


def _read_proc_memory_mb(pid: int) -> int | None:
    """Read RSS memory in MB for a process.

    On Linux reads /proc/{pid}/statm directly. On macOS shells out to ps.
    Returns None if the process doesn't exist or the read fails.
    """
    if sys.platform == "linux":
        try:
            with open(f"/proc/{pid}/statm") as f:
                parts = f.read().split()
            resident_pages = int(parts[1])
            return (resident_pages * os.sysconf("SC_PAGE_SIZE")) // (1024 * 1024)
        except (FileNotFoundError, ProcessLookupError, IndexError, ValueError):
            return None
    else:
        try:
            result = subprocess.run(
                ["ps", "-o", "rss=", "-p", str(pid)],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return None
            rss_kb = int(result.stdout.strip())
            return rss_kb // 1024
        except (subprocess.TimeoutExpired, ValueError, OSError):
            return None


def _read_proc_cpu_percent(
    pid: int,
    prev_total: float,
    prev_utime: float,
) -> tuple[int, float, float]:
    """Compute delta CPU usage percentage between calls.

    On Linux reads /proc/{pid}/stat and /proc/stat. On other platforms returns 0.
    Returns (cpu_percent, new_total, new_utime).
    """
    if sys.platform != "linux":
        return (0, prev_total, prev_utime)
    try:
        with open(f"/proc/{pid}/stat") as f:
            fields = f.read().split()
        utime = int(fields[13]) + int(fields[14])

        with open("/proc/stat") as f:
            cpu_line = f.readline()
        total = sum(int(x) for x in cpu_line.split()[1:])

        delta_total = total - prev_total
        delta_utime = utime - prev_utime
        if delta_total <= 0 or prev_total == 0:
            return (0, total, utime)
        pct = int((delta_utime / delta_total) * 100)
        return (pct, total, utime)
    except (FileNotFoundError, ProcessLookupError, IndexError, ValueError):
        return (0, prev_total, prev_utime)


def _cpu_profile_stub(cpu_format: int) -> bytes:
    """Return a minimal stub CPU profile for when py-spy is unavailable."""
    if cpu_format == cluster_pb2.CpuProfile.FLAMEGRAPH:
        return (
            b'<svg xmlns="http://www.w3.org/2000/svg" width="400" height="50">'
            b'<text x="10" y="30" font-size="14">py-spy unavailable in local mode</text></svg>'
        )
    elif cpu_format == cluster_pb2.CpuProfile.SPEEDSCOPE:
        return (
            b'{"version":"0.1.0","$schema":"https://www.speedscope.app/file-format-schema.json",'
            b'"profiles":[],"shared":{"frames":[]}}'
        )
    else:  # RAW
        return b"py-spy unavailable in local mode\n"


def _memory_profile_stub(memory_format: int) -> bytes:
    """Return a minimal stub memory profile for when memray is unavailable."""
    if memory_format == cluster_pb2.MemoryProfile.FLAMEGRAPH:
        return (
            b"<!DOCTYPE html><html><head><title>Memory Profile</title></head><body>"
            b"<p>memray unavailable in local mode</p></body></html>"
        )
    elif memory_format == cluster_pb2.MemoryProfile.TABLE:
        return b"memray unavailable in local mode\n"
    else:  # STATS
        return b'{"error": "memray unavailable in local mode"}'


class ProcessLogReader:
    """Index-based incremental log reader for ProcessContainer._logs."""

    def __init__(self, logs: list[LogLine]) -> None:
        self._logs = logs
        self._index: int = 0

    def read(self) -> list[LogLine]:
        new_lines = self._logs[self._index :]
        self._index = len(self._logs)
        return new_lines

    def read_all(self) -> list[LogLine]:
        return list(self._logs)


@dataclass
class ProcessContainerHandle:
    """Process implementation of ContainerHandle.

    For local execution, build() is a no-op since the host is already configured.
    run() executes the command as a subprocess.
    """

    config: ContainerConfig
    runtime: ProcessRuntime
    _container: ProcessContainer | None = field(default=None, repr=False)
    _container_id: str | None = field(default=None, repr=False)
    _prev_cpu_total: float = field(default=0.0, repr=False)
    _prev_cpu_utime: float = field(default=0.0, repr=False)
    _prev_cpu_total: float = field(default=0.0, repr=False)
    _prev_cpu_utime: float = field(default=0.0, repr=False)

    @property
    def container_id(self) -> str | None:
        """Return the container ID, if any."""
        return self._container_id

    def build(self) -> list[LogLine]:
        """No-op for local execution - host is already configured.

        In local/test mode, the environment is already set up with the
        necessary dependencies. We skip the uv sync step.

        Returns:
            Empty list (no build logs in local mode).
        """
        return []

    def run(self) -> None:
        """Start the subprocess."""
        config = self.config

        # Remap container paths to host paths in env vars
        mount_map = {container_path: host_path for host_path, container_path, _ in config.mounts}
        env = {**build_device_env_vars(config), **dict(config.env)}
        for key, value in env.items():
            if value in mount_map:
                env[key] = mount_map[value]

        # In local mode, resolve IRIS_PYTHON to the current interpreter so that
        # bash -c "exec $IRIS_PYTHON ..." works even when "python" isn't on PATH.
        env["IRIS_PYTHON"] = sys.executable

        # Extract the command from the RuntimeEntrypoint proto
        cmd = list(config.entrypoint.run_command.argv)

        # Remap container mount paths in command args to host paths
        remapped_cmd = []
        for arg in cmd:
            for container_path, host_path in mount_map.items():
                if arg.startswith(container_path):
                    arg = host_path + arg[len(container_path) :]
                    break
            remapped_cmd.append(arg)

        # Create container with remapped environment and command
        from dataclasses import replace

        updated_config = replace(config, env=env)

        self._container_id = f"local-{uuid.uuid4().hex[:8]}"
        self._container = ProcessContainer(
            config=updated_config,
            command=remapped_cmd,
        )
        self._container.start()

        logger.info(
            "Started local process for task %s (container_id=%s)",
            config.task_id,
            self._container_id,
        )

    def stop(self, force: bool = False) -> None:
        """Stop the subprocess."""
        del force  # Local containers don't distinguish force vs graceful
        if self._container:
            self._container.kill()

    def status(self) -> ContainerStatus:
        """Check container status (running, exit code, error)."""
        if not self._container:
            return ContainerStatus(running=False, error="Container not started")
        return ContainerStatus(
            running=self._container._running,
            exit_code=self._container._exit_code,
            error=self._container._error,
        )

    def log_reader(self) -> RuntimeLogReader:
        """Create an incremental log reader for this container."""
        return ProcessLogReader(self._container._logs if self._container else [])

    def stats(self) -> ContainerStats:
        """Get resource usage statistics from the underlying subprocess."""
        if not self._container or not self._container._process or self._container._process.poll() is not None:
            return ContainerStats(memory_mb=0, cpu_percent=0, process_count=0, available=False)

        pid = self._container._process.pid
        memory_mb = _read_proc_memory_mb(pid)
        cpu_pct, self._prev_cpu_total, self._prev_cpu_utime = _read_proc_cpu_percent(
            pid, self._prev_cpu_total, self._prev_cpu_utime
        )
        return ContainerStats(
            memory_mb=memory_mb or 0,
            cpu_percent=cpu_pct,
            process_count=1,
            available=memory_mb is not None,
        )

    def profile(self, duration_seconds: int, profile_type: cluster_pb2.ProfileType) -> bytes:
        """Profile the running process using py-spy (CPU) or memray (memory), with fallback stubs."""

        if not self._container or not self._container._process:
            raise RuntimeError("Cannot profile: no running process")

        # Dispatch to CPU or memory profiling
        if profile_type.HasField("cpu"):
            return self._profile_cpu(duration_seconds, profile_type.cpu)
        elif profile_type.HasField("memory"):
            return self._profile_memory(duration_seconds, profile_type.memory)
        else:
            raise RuntimeError("ProfileType must specify either cpu or memory profiler")

    def _profile_cpu(self, duration_seconds: int, cpu_config: cluster_pb2.CpuProfile) -> bytes:
        """Profile CPU using py-spy, with fallback stub."""
        pid = self._container._process.pid
        spec = resolve_cpu_spec(cpu_config, duration_seconds, pid=str(pid))

        output_path = None
        try:
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=f".{spec.ext}", delete=False) as f:
                output_path = f.name

            cmd = build_pyspy_cmd(spec, py_spy_bin="py-spy", output_path=output_path)
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=duration_seconds + 30)
            if result.returncode == 0:
                return Path(output_path).read_bytes()
        except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError, OSError):
            logger.warning("py-spy profiling failed for PID %s; falling back to stub", pid, exc_info=True)
        finally:
            if output_path is not None:
                Path(output_path).unlink(missing_ok=True)

        return _cpu_profile_stub(cpu_config.format)

    def _profile_memory(self, duration_seconds: int, memory_config: cluster_pb2.MemoryProfile) -> bytes:
        """Profile memory using memray, with fallback stub."""
        pid = self._container._process.pid
        spec = resolve_memory_spec(memory_config, duration_seconds, pid=str(pid))

        trace_path = None
        output_path = None
        try:
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
                trace_path = f.name

            attach_cmd = build_memray_attach_cmd(spec, memray_bin="memray", trace_path=trace_path)
            result = subprocess.run(attach_cmd, capture_output=True, text=True, timeout=duration_seconds + 10)
            if result.returncode != 0:
                raise RuntimeError(f"memray attach failed: {result.stderr}")

            if spec.output_is_file:
                with tempfile.NamedTemporaryFile(suffix=f".{spec.ext}", delete=False) as f:
                    output_path = f.name

            transform_cmd = build_memray_transform_cmd(
                spec, memray_bin="memray", trace_path=trace_path, output_path=output_path or ""
            )
            result = subprocess.run(transform_cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                raise RuntimeError(f"memray {spec.reporter} failed: {result.stderr}")

            if spec.output_is_file:
                return Path(output_path).read_bytes()
            else:
                return result.stdout.encode("utf-8")

        except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError, OSError, RuntimeError):
            logger.warning("memray profiling failed for PID %s; falling back to stub", pid, exc_info=True)
        finally:
            if trace_path is not None:
                Path(trace_path).unlink(missing_ok=True)
            if output_path is not None:
                Path(output_path).unlink(missing_ok=True)

        return _memory_profile_stub(memory_config.format)

    def cleanup(self) -> None:
        """Kill the subprocess and clean up resources."""
        if self._container:
            self._container.kill()
        self._container = None
        self._container_id = None


class ProcessRuntime:
    """ContainerRuntime implementation using subprocess.Popen.

    Provides Docker-like interface for local testing without containers.
    Creates ProcessContainerHandle instances with the build/run lifecycle.
    """

    def __init__(self):
        self._handles: list[ProcessContainerHandle] = []
        _active_runtimes.add(self)

    def create_container(self, config: ContainerConfig) -> ProcessContainerHandle:
        """Create a container handle from config.

        The handle is not started - call handle.build() then handle.run()
        to execute the container.
        """
        handle = ProcessContainerHandle(config=config, runtime=self)
        self._handles.append(handle)
        return handle

    def stage_bundle(
        self,
        *,
        bundle_gcs_path: str,
        workdir: Path,
        workdir_files: dict[str, bytes],
        fetch_bundle: Callable[[str], Path],
    ) -> None:
        """Stage bundle and workdir files on worker-local filesystem."""
        bundle_path = fetch_bundle(bundle_gcs_path)
        shutil.copytree(bundle_path, workdir, dirs_exist_ok=True)
        for name, data in workdir_files.items():
            (workdir / name).write_bytes(data)

    def list_containers(self) -> list[ProcessContainerHandle]:
        """List all managed container handles."""
        return list(self._handles)

    def list_iris_containers(self, all_states: bool = True) -> list[str]:
        """List all container IDs."""
        del all_states
        return [h.container_id for h in self._handles if h.container_id]

    def remove_all_iris_containers(self) -> int:
        """Stop all containers. Returns count."""
        count = len(self._handles)
        for handle in self._handles:
            handle.cleanup()
        self._handles.clear()
        return count

    def cleanup(self) -> None:
        """Kill all tracked containers and clear state."""
        for handle in self._handles:
            handle.cleanup()
        self._handles.clear()
