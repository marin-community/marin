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
import subprocess
import sys
import threading
import uuid
import weakref
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from iris.cluster.runtime.types import ContainerConfig, ContainerStats, ContainerStatus
from iris.cluster.worker.worker_types import LogLine
from iris.managed_thread import ManagedThread, get_thread_container
from iris.time_utils import Timestamp

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
        cmd = self._build_command()
        if cmd and cmd[0] in {"python", "python3"}:
            # Ensure we use the current interpreter even when PATH lacks "python".
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

    def _build_command(self) -> list[str]:
        return self.command

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
        env = dict(config.env)
        for key, value in env.items():
            if value in mount_map:
                env[key] = mount_map[value]

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

    def logs(self, since: Timestamp | None = None) -> list[LogLine]:
        """Get container logs since timestamp."""
        if not self._container:
            return []
        if since:
            since_dt = datetime.fromtimestamp(since.epoch_seconds(), tz=timezone.utc)
            return [log for log in self._container._logs if log.timestamp > since_dt]
        return self._container._logs

    def stats(self) -> ContainerStats:
        """Get resource usage statistics."""
        return ContainerStats(memory_mb=100, cpu_percent=10, process_count=1, available=True)

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
