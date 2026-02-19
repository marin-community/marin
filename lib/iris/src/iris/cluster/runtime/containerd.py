# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Containerd runtime using crictl (CRI CLI) for containerd-based hosts.

Used on CoreWeave bare metal nodes where Docker is not available. The worker
Pod mounts the host containerd socket and uses crictl to create/manage task
containers inside CRI pod sandboxes. Each task gets its own sandbox for
namespace isolation.

Implements the ContainerHandle protocol with the same two-phase model as Docker:
- build(): Run setup commands in a temporary container sharing the workdir mount
- run(): Start the main command container

Both containers live inside the same pod sandbox, so they share a workdir
bind-mount and the same network namespace. CRI host-network mode
(namespace_options.network = 2 = NODE) preserves Iris's host-network
assumptions for endpoint registration and peer connectivity.
"""

import json
import logging
import os
import shlex
import shutil
import subprocess
import tempfile
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from iris.cluster.runtime.types import ContainerConfig, ContainerStats, ContainerStatus
from iris.cluster.worker.worker_types import LogLine
from iris.rpc import cluster_pb2
from iris.time_utils import Timestamp

logger = logging.getLogger(__name__)

# CRI NamespaceMode values (from k8s CRI API proto)
_CRI_NAMESPACE_MODE_NODE = 2  # Host network namespace

# Default log directory for CRI pod sandboxes. This must be writable by
# containerd (which runs as root). On CoreWeave bare metal, the worker Pod
# also runs as root so crictl can read the logs. For local development, the
# directory must exist and be root-writable.
_DEFAULT_LOG_DIR = "/var/lib/containerd/iris-logs"


def _run_crictl(
    args: list[str],
    *,
    endpoint: str | None = None,
    timeout: int = 30,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run a crictl command and return the result.

    All crictl calls go through this helper for consistent error handling
    and logging.
    """
    cmd = ["crictl"]
    if endpoint:
        cmd.extend(["--runtime-endpoint", endpoint])
    cmd.extend(args)
    logger.debug("Running: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"crictl {args[0]} failed (rc={result.returncode}): {result.stderr.strip()}")
    return result


def _write_json_file(data: dict, directory: Path, name: str) -> Path:
    """Write a JSON config file to a directory and return the path."""
    path = directory / name
    path.write_text(json.dumps(data, indent=2))
    return path


def _build_pod_sandbox_config(config: ContainerConfig, log_directory: str) -> dict:
    """Build a CRI pod sandbox config JSON for a task.

    The sandbox provides network namespace isolation and carries
    annotations for iris discoverability (CRI sandboxes use annotations
    rather than Docker-style labels).
    """
    annotations = {"iris.managed": "true"}
    if config.task_id:
        annotations["iris.task_id"] = config.task_id
    if config.job_id:
        annotations["iris.job_id"] = config.job_id

    # CRI requires name, uid, and namespace to all be non-empty in metadata.
    # Task IDs contain forward slashes (e.g. "/smoke-gpu-check/0") which are
    # not valid in CRI sandbox names, so we sanitize them to dashes.
    sanitized_name = (config.task_id or "unknown").replace("/", "-").strip("-")
    sandbox_config: dict = {
        "metadata": {
            "name": f"iris-task-{sanitized_name}",
            "uid": uuid.uuid4().hex,
            "namespace": "iris",
        },
        "annotations": annotations,
        "log_directory": log_directory,
    }

    linux_config = sandbox_config.setdefault("linux", {})

    # Host network mode: CRI namespace_options.network = NODE (2)
    if config.network_mode == "host":
        linux_config.setdefault("security_context", {}).setdefault("namespace_options", {})[
            "network"
        ] = _CRI_NAMESPACE_MODE_NODE

    cgroup_parent = os.environ.get("IRIS_CGROUP_PARENT", "").strip()
    if cgroup_parent:
        linux_config["cgroup_parent"] = cgroup_parent

    return sandbox_config


def _build_container_config(
    config: ContainerConfig,
    command: list[str],
    *,
    label_suffix: str = "",
    include_resources: bool = False,
) -> dict:
    """Build a CRI container config JSON.

    Maps ContainerConfig fields to the CRI container configuration format
    used by `crictl create`.
    """
    labels = {"iris.managed": "true"}
    if config.task_id:
        labels[f"iris.task_id{label_suffix}"] = config.task_id
    if config.job_id:
        labels["iris.job_id"] = config.job_id

    container_config: dict = {
        "metadata": {
            "name": f"iris-container-{config.task_id or 'unknown'}{label_suffix}",
        },
        "image": {"image": config.image},
        "command": command,
        "working_dir": config.workdir,
        "labels": labels,
        "log_path": f"iris-{config.task_id or 'unknown'}{label_suffix}.log",
    }

    # Environment variables
    env_vars = []
    for k, v in config.env.items():
        env_vars.append({"key": k, "value": v})
    if env_vars:
        container_config["envs"] = env_vars

    # Mounts
    mounts = []
    for host_path, container_path, mode in config.mounts:
        mount: dict[str, str | bool] = {
            "container_path": container_path,
            "host_path": host_path,
        }
        if "ro" in mode:
            mount["readonly"] = True
        mounts.append(mount)
    if mounts:
        container_config["mounts"] = mounts

    # Resource limits (only for run container)
    if include_resources and config.resources:
        linux = container_config.setdefault("linux", {})
        resources = linux.setdefault("resources", {})

        cpu_millicores = config.get_cpu_millicores()
        if cpu_millicores:
            # CRI uses cpu_quota/cpu_period for CPU limits
            # 1 CPU = period of 100000, quota of 100000
            resources["cpu_period"] = 100000
            resources["cpu_quota"] = cpu_millicores * 100

        memory_mb = config.get_memory_mb()
        if memory_mb:
            resources["memory_limit_in_bytes"] = memory_mb * 1024 * 1024

    return container_config


def _parse_crictl_log_line(line: str) -> tuple[datetime, str, str]:
    """Parse a CRI log line from the container log file.

    CRI log format: "<timestamp> <stream> <tag> <message>"
    e.g. "2024-01-01T00:00:00.000000000Z stdout F hello world"

    The timestamp may include nanoseconds which we truncate to microseconds
    for Python's datetime.fromisoformat().
    """
    parts = line.split(" ", 3)
    if len(parts) >= 3:
        ts_str = parts[0]
        stream = parts[1]
        # Tag is parts[2] (F=full, P=partial), message is parts[3] if present
        message = parts[3] if len(parts) == 4 else ""
        try:
            # Truncate nanoseconds to microseconds for fromisoformat
            if len(ts_str) > 27 and ts_str.endswith("Z"):
                ts_str = ts_str[:26] + "Z"
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            return ts, stream, message
        except ValueError:
            pass

    return datetime.now(timezone.utc), "stdout", line


@dataclass
class ContainerdContainerHandle:
    """ContainerHandle backed by crictl commands.

    Each task gets its own CRI pod sandbox for network namespace isolation.
    The build and run containers share the same sandbox (and thus the same
    workdir mount and network namespace). This mirrors DockerContainerHandle's
    two-phase lifecycle.
    """

    config: ContainerConfig
    runtime: "ContainerdRuntime"
    _sandbox_id: str = field(default="", repr=False)
    _run_container_id: str | None = field(default=None, repr=False)
    _config_dir: Path | None = field(default=None, repr=False)
    _log_directory: str = field(default=_DEFAULT_LOG_DIR, repr=False)
    _endpoint: str | None = field(default=None, repr=False)
    _cached_log_path: str | None = field(default=None, repr=False)

    @property
    def container_id(self) -> str | None:
        return self._run_container_id

    @property
    def sandbox_id(self) -> str:
        return self._sandbox_id

    def build(self) -> list[LogLine]:
        """Run setup_commands in a temporary container inside the same sandbox.

        Creates a temporary build container, runs setup commands, captures logs,
        and removes the build container. The workdir mount is shared with the
        run container, so .venv created during build is available to run().
        """
        if not self.config.entrypoint.setup_commands:
            logger.debug("No setup_commands, skipping build phase")
            return []

        setup_script = _generate_setup_script(self.config.entrypoint.setup_commands)
        _write_script_to_workdir(self.config, setup_script, "_setup_env.sh")

        build_config = _build_container_config(
            self.config,
            command=["sh", "/app/_setup_env.sh"],
            label_suffix="_build",
        )
        assert self._config_dir is not None
        build_config_path = _write_json_file(build_config, self._config_dir, "build-container.json")
        pod_config_path = self._config_dir / "pod-sandbox.json"

        result = _run_crictl(
            ["create", self._sandbox_id, str(build_config_path), str(pod_config_path)],
            endpoint=self._endpoint,
        )
        build_container_id = result.stdout.strip()
        logger.info("Created build container %s in sandbox %s", build_container_id, self._sandbox_id)

        build_logs: list[LogLine] = []
        try:
            _run_crictl(["start", build_container_id], endpoint=self._endpoint)

            last_log_time: Timestamp | None = None
            while True:
                status = _inspect_container(build_container_id, endpoint=self._endpoint)

                new_logs = _get_container_logs(
                    build_container_id, self._log_directory, since=last_log_time, endpoint=self._endpoint
                )
                if new_logs:
                    build_logs.extend(new_logs)
                    last_log_time = Timestamp.from_seconds(new_logs[-1].timestamp.timestamp()).add_ms(1)

                if not status.running:
                    break
                time.sleep(0.5)

            final_logs = _get_container_logs(
                build_container_id, self._log_directory, since=last_log_time, endpoint=self._endpoint
            )
            build_logs.extend(final_logs)

            if status.exit_code != 0:
                log_text = "\n".join(f"[{entry.source}] {entry.data}" for entry in build_logs[-50:])
                raise RuntimeError(f"Build failed with exit_code={status.exit_code}\nLast 50 log lines:\n{log_text}")

            logger.info("Build phase completed successfully for task %s", self.config.task_id)
            return build_logs

        finally:
            _run_crictl(["rm", "-f", build_container_id], endpoint=self._endpoint, check=False)

    def run(self) -> None:
        """Start the main command container.

        Non-blocking -- returns immediately after starting. Use status() to monitor.
        """
        quoted_cmd = " ".join(shlex.quote(arg) for arg in self.config.entrypoint.run_command.argv)

        if self.config.entrypoint.setup_commands:
            run_script = f"""#!/bin/bash
set -e
cd /app
source .venv/bin/activate
exec {quoted_cmd}
"""
            _write_script_to_workdir(self.config, run_script, "_run.sh")
            command = ["bash", "/app/_run.sh"]
        else:
            command = list(self.config.entrypoint.run_command.argv)

        run_config = _build_container_config(
            self.config,
            command=command,
            include_resources=True,
        )
        assert self._config_dir is not None
        run_config_path = _write_json_file(run_config, self._config_dir, "run-container.json")
        pod_config_path = self._config_dir / "pod-sandbox.json"

        result = _run_crictl(
            ["create", self._sandbox_id, str(run_config_path), str(pod_config_path)],
            endpoint=self._endpoint,
        )
        self._run_container_id = result.stdout.strip()
        self.runtime._track_container(self._run_container_id)

        _run_crictl(["start", self._run_container_id], endpoint=self._endpoint)
        logger.info(
            "Run phase started for task %s (container_id=%s)",
            self.config.task_id,
            self._run_container_id,
        )

    def stop(self, force: bool = False) -> None:
        """Stop the run container."""
        if not self._run_container_id:
            return
        timeout_seconds = 0 if force else 10
        _run_crictl(
            ["stop", f"--timeout={timeout_seconds}", self._run_container_id],
            endpoint=self._endpoint,
            check=False,
        )

    def status(self) -> ContainerStatus:
        if not self._run_container_id:
            return ContainerStatus(running=False, error="Container not started")
        return _inspect_container(self._run_container_id, endpoint=self._endpoint)

    def logs(self, since: Timestamp | None = None) -> list[LogLine]:
        if not self._run_container_id:
            return []
        return _get_container_logs(
            self._run_container_id,
            self._log_directory,
            since=since,
            endpoint=self._endpoint,
            cached_log_path=self._cached_log_path,
            cache_log_path_callback=self._set_cached_log_path,
        )

    def _set_cached_log_path(self, path: str) -> None:
        self._cached_log_path = path

    def stats(self) -> ContainerStats:
        if not self._run_container_id:
            return ContainerStats(memory_mb=0, cpu_percent=0, process_count=0, available=False)
        return _get_container_stats(self._run_container_id, endpoint=self._endpoint)

    def profile(self, duration_seconds: int, profile_type: cluster_pb2.ProfileType) -> bytes:
        raise NotImplementedError("Profiling is not yet supported on containerd runtime")

    def cleanup(self) -> None:
        """Remove the run container, its sandbox, and the temporary config directory."""
        if self._run_container_id:
            _run_crictl(["rm", "-f", self._run_container_id], endpoint=self._endpoint, check=False)
            self.runtime._untrack_container(self._run_container_id)
            self._run_container_id = None

        if self._sandbox_id:
            _run_crictl(["stopp", self._sandbox_id], endpoint=self._endpoint, check=False)
            _run_crictl(["rmp", self._sandbox_id], endpoint=self._endpoint, check=False)
            self._sandbox_id = ""

        if self._config_dir and self._config_dir.exists():
            shutil.rmtree(self._config_dir, ignore_errors=True)
            self._config_dir = None


# ---------------------------------------------------------------------------
# Top-level helper functions (functional style, no shared state mutation)
# ---------------------------------------------------------------------------


def _generate_setup_script(setup_commands: list[str]) -> str:
    lines = ["#!/bin/sh", "set -e"]
    lines.extend(setup_commands)
    return "\n".join(lines) + "\n"


def _write_script_to_workdir(config: ContainerConfig, script: str, filename: str) -> None:
    """Write a script file into the /app host mount directory."""
    for host_path, container_path, _mode in config.mounts:
        if container_path == "/app":
            (Path(host_path) / filename).write_text(script)
            return
    raise RuntimeError("No /app mount found in config")


def _inspect_container(container_id: str, *, endpoint: str | None = None) -> ContainerStatus:
    """Inspect a container via crictl and return its status.

    Note: crictl requires flags before positional args, so we use
    `crictl inspect -o json <id>` not `crictl inspect <id> -o json`.
    """
    result = _run_crictl(["inspect", "-o", "json", container_id], endpoint=endpoint, check=False)
    if result.returncode != 0:
        return ContainerStatus(running=False, error="Container not found")

    try:
        data = json.loads(result.stdout)
        status = data.get("status", {})
        state = status.get("state", "CONTAINER_UNKNOWN")

        running = state == "CONTAINER_RUNNING"
        exit_code = None
        error_msg = None
        oom_killed = False

        if not running:
            exit_code = status.get("exitCode")
            reason = status.get("reason", "")
            message = status.get("message", "")
            if reason == "OOMKilled" or "oom" in message.lower():
                oom_killed = True
            if message:
                error_msg = message

        return ContainerStatus(
            running=running,
            exit_code=exit_code,
            error=error_msg,
            oom_killed=oom_killed,
        )
    except (json.JSONDecodeError, KeyError) as e:
        return ContainerStatus(running=False, error=f"Failed to parse inspect output: {e}")


def _get_log_path_from_inspect(container_id: str, *, endpoint: str | None = None) -> str | None:
    """Get the log file path for a container from crictl inspect."""
    result = _run_crictl(["inspect", "-o", "json", container_id], endpoint=endpoint, check=False)
    if result.returncode != 0:
        return None
    try:
        data = json.loads(result.stdout)
        return data.get("status", {}).get("logPath")
    except (json.JSONDecodeError, KeyError):
        return None


def _get_container_logs(
    container_id: str,
    log_directory: str,
    *,
    since: Timestamp | None = None,
    endpoint: str | None = None,
    cached_log_path: str | None = None,
    cache_log_path_callback: Callable[[str], None] | None = None,
) -> list[LogLine]:
    """Fetch logs from a container.

    First tries `crictl logs` which reads from the CRI log file. If that fails
    (e.g. due to file permissions when running as non-root), falls back to
    reading the log file directly. The log path from inspect is cached via
    cached_log_path/cache_log_path_callback to avoid repeated crictl inspect calls.
    """
    cmd = ["logs"]
    if since:
        cmd.extend(["--since", since.as_formatted_date()])
    cmd.append(container_id)

    result = _run_crictl(cmd, endpoint=endpoint, check=False)

    raw_lines: str | None = None
    if result.returncode == 0 and result.stdout.strip():
        raw_lines = result.stdout
    else:
        log_path = cached_log_path
        if not log_path:
            log_path = _get_log_path_from_inspect(container_id, endpoint=endpoint)
            if log_path and cache_log_path_callback:
                cache_log_path_callback(log_path)
        if log_path:
            try:
                raw_lines = Path(log_path).read_text()
            except OSError:
                logger.debug("Could not read log file %s", log_path)

    if not raw_lines:
        return []

    logs: list[LogLine] = []
    for line in raw_lines.splitlines():
        if not line:
            continue
        timestamp, source, data = _parse_crictl_log_line(line)
        if since and timestamp.timestamp() < since.epoch_seconds():
            continue
        logs.append(LogLine(timestamp=timestamp, source=source, data=data))

    return logs


def _get_container_stats(container_id: str, *, endpoint: str | None = None) -> ContainerStats:
    """Fetch resource usage stats via crictl.

    Note: crictl requires flags before positional args.
    """
    result = _run_crictl(["stats", "-o", "json", "--id", container_id], endpoint=endpoint, check=False)
    if result.returncode != 0:
        return ContainerStats(memory_mb=0, cpu_percent=0, process_count=0, available=False)

    try:
        data = json.loads(result.stdout)
        stats_list = data.get("stats", [])
        if not stats_list:
            return ContainerStats(memory_mb=0, cpu_percent=0, process_count=0, available=False)

        s = stats_list[0]
        # CRI stats report memory in bytes under various paths depending on version
        memory_bytes = 0
        if "memory" in s:
            mem = s["memory"]
            if "workingSetBytes" in mem:
                memory_bytes = int(mem["workingSetBytes"].get("value", 0))
            elif "usageBytes" in mem:
                memory_bytes = int(mem["usageBytes"].get("value", 0))

        cpu_percent = 0
        if "cpu" in s:
            # CRI reports CPU in nanoseconds; percent approximation is not
            # directly available. We report 0 for now -- accurate CPU percent
            # requires sampling delta over time which is beyond a single stats call.
            pass

        return ContainerStats(
            memory_mb=memory_bytes // (1024 * 1024),
            cpu_percent=cpu_percent,
            process_count=0,
            available=True,
        )
    except (json.JSONDecodeError, ValueError, KeyError):
        return ContainerStats(memory_mb=0, cpu_percent=0, process_count=0, available=False)


def _ensure_log_directory(log_directory: str) -> None:
    """Ensure the CRI log directory exists.

    On CoreWeave nodes this is handled by the Pod spec (hostPath volume).
    For local development we attempt to create it but tolerate permission
    errors since containerd (running as root) may manage this directory.
    """
    try:
        path = Path(log_directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.warning("Cannot create log directory %s", log_directory)


class ContainerdRuntime:
    """Container runtime using crictl (CRI) for containerd-based hosts.

    Used on CoreWeave bare metal nodes where Docker is not available. The worker
    Pod mounts the host containerd socket and uses crictl to create/manage task
    containers. All images on ghcr.io are public, so crictl pull does not need
    auth credentials.

    Each create_container() call creates a CRI pod sandbox (for network namespace
    isolation) and a container inside it. The sandbox is the unit of cleanup.
    """

    def __init__(
        self,
        socket_path: str = "/run/containerd/containerd.sock",
        log_directory: str = _DEFAULT_LOG_DIR,
    ) -> None:
        self._endpoint = f"unix://{socket_path}"
        self._log_directory = log_directory
        self._handles: list[ContainerdContainerHandle] = []
        self._created_containers: set[str] = set()
        _ensure_log_directory(log_directory)

    def create_container(self, config: ContainerConfig) -> ContainerdContainerHandle:
        """Create a container handle: pull image, create sandbox, prepare configs.

        The handle is not started -- call handle.build() then handle.run().
        """
        logger.info("Pulling image: %s", config.image)
        _run_crictl(["pull", config.image], endpoint=self._endpoint, timeout=300)

        config_dir = Path(tempfile.mkdtemp(prefix="iris-cri-"))

        pod_config = _build_pod_sandbox_config(config, self._log_directory)
        _write_json_file(pod_config, config_dir, "pod-sandbox.json")

        pod_config_path = config_dir / "pod-sandbox.json"
        result = _run_crictl(["runp", str(pod_config_path)], endpoint=self._endpoint)
        sandbox_id = result.stdout.strip()
        logger.info("Created pod sandbox %s for task %s", sandbox_id, config.task_id)

        handle = ContainerdContainerHandle(
            config=config,
            runtime=self,
            _sandbox_id=sandbox_id,
            _config_dir=config_dir,
            _log_directory=self._log_directory,
            _endpoint=self._endpoint,
        )
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

    def _track_container(self, container_id: str) -> None:
        self._created_containers.add(container_id)

    def _untrack_container(self, container_id: str) -> None:
        self._created_containers.discard(container_id)

    def list_containers(self) -> list[ContainerdContainerHandle]:
        return list(self._handles)

    def list_iris_sandboxes(self) -> list[str]:
        """List all CRI pod sandboxes with iris.managed annotation."""
        result = _run_crictl(["pods", "-o", "json"], endpoint=self._endpoint, check=False)
        if result.returncode != 0:
            return []

        try:
            data = json.loads(result.stdout)
            sandbox_ids = []
            for item in data.get("items", []):
                annotations = item.get("annotations", {})
                if annotations.get("iris.managed") == "true":
                    sandbox_ids.append(item["id"])
            return sandbox_ids
        except (json.JSONDecodeError, KeyError):
            return []

    def remove_all_iris_containers(self) -> int:
        """Force remove all iris-managed sandboxes and their containers."""
        sandbox_ids = self.list_iris_sandboxes()
        if not sandbox_ids:
            return 0

        for sid in sandbox_ids:
            _run_crictl(["stopp", sid], endpoint=self._endpoint, check=False)
            _run_crictl(["rmp", sid], endpoint=self._endpoint, check=False)

        return len(sandbox_ids)

    def cleanup(self) -> None:
        """Clean up all containers and sandboxes managed by this runtime."""
        for handle in self._handles:
            handle.cleanup()
        self._handles.clear()

        for cid in list(self._created_containers):
            _run_crictl(["rm", "-f", cid], endpoint=self._endpoint, check=False)
        self._created_containers.clear()
