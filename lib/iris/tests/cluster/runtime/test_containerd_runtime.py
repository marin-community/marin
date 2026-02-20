# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for ContainerdRuntime.

These tests exercise real containerd via crictl. They require:
- containerd running locally with CRI plugin enabled
- crictl installed and accessible
- Permissions to access the containerd socket and CRI log files

Tests are skipped when containerd CRI is not available.
"""

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from iris.cluster.runtime.containerd import ContainerdRuntime
from iris.cluster.runtime.types import ContainerConfig
from iris.rpc import cluster_pb2
from iris.time_utils import Deadline, Duration

CONTAINERD_SOCKET = "/run/containerd/containerd.sock"
TEST_IMAGE = "docker.io/library/alpine:latest"
# containerd writes log files as root with mode 0640. The directory must exist
# and be writable by containerd (root). On CoreWeave the worker runs as root so
# both writing and reading work. For local dev testing, log reads may fail if
# the test process is non-root.
LOG_DIR = "/var/lib/containerd/iris-test-logs"

_containerd_available = False
_can_read_cri_logs = os.getuid() == 0

if Path(CONTAINERD_SOCKET).exists():
    try:
        result = subprocess.run(
            ["crictl", "version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        _containerd_available = result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass


pytestmark = [
    pytest.mark.containerd,
    pytest.mark.skipif(not _containerd_available, reason="containerd CRI not available"),
]


def _make_entrypoint(
    run_command: list[str],
    setup_commands: list[str] | None = None,
) -> cluster_pb2.RuntimeEntrypoint:
    """Build a RuntimeEntrypoint proto for tests."""
    ep = cluster_pb2.RuntimeEntrypoint()
    ep.run_command.CopyFrom(cluster_pb2.CommandEntrypoint(argv=run_command))
    if setup_commands:
        ep.setup_commands.extend(setup_commands)
    return ep


def _make_config(
    run_command: list[str],
    *,
    task_id: str = "test-task",
    job_id: str = "test-job",
    setup_commands: list[str] | None = None,
    workdir_mount: tuple[str, str, str] | None = None,
    env: dict[str, str] | None = None,
) -> ContainerConfig:
    """Build a ContainerConfig for tests."""
    mounts = []
    if workdir_mount:
        mounts.append(workdir_mount)
    return ContainerConfig(
        image=TEST_IMAGE,
        entrypoint=_make_entrypoint(run_command, setup_commands),
        env=env or {},
        task_id=task_id,
        job_id=job_id,
        mounts=mounts,
    )


@pytest.fixture
def runtime():
    """Create a ContainerdRuntime and clean up all iris sandboxes on teardown."""
    rt = ContainerdRuntime(socket_path=CONTAINERD_SOCKET, log_directory=LOG_DIR)
    yield rt
    rt.cleanup()
    rt.remove_all_iris_containers()


def _wait_for_exit(handle, timeout: Duration = Duration.from_seconds(15)):
    """Poll until the container exits or timeout."""
    deadline = Deadline.from_now(timeout)
    while not deadline.expired():
        status = handle.status()
        if not status.running:
            return status
        time.sleep(0.5)
    return handle.status()


def test_create_and_run_simple_command(runtime: ContainerdRuntime):
    """Create a container that echoes 'hello', verify it exits 0 and logs contain output."""
    config = _make_config(["echo", "hello"])
    handle = runtime.create_container(config)

    try:
        handle.run()
        status = _wait_for_exit(handle)

        assert not status.running
        assert status.exit_code == 0

        # CRI log files are owned by root with mode 0640. When running as
        # non-root, logs may not be readable. This is a local dev limitation;
        # on CoreWeave the worker runs as root.
        if _can_read_cri_logs:
            reader = handle.log_reader()
            logs = reader.read_all()
            log_text = " ".join(entry.data for entry in logs)
            assert "hello" in log_text
    finally:
        handle.cleanup()


def test_build_phase_creates_shared_workdir(runtime: ContainerdRuntime, tmp_path: Path):
    """Verify build() runs setup commands and writes output to the shared workdir mount.

    Alpine doesn't have bash, so the run() phase which activates a venv
    via bash won't work here. We test only the build phase and verify
    the setup script wrote files visible on the host mount.
    """
    workdir = tmp_path / "app"
    workdir.mkdir()

    config = _make_config(
        ["cat", "/app/build_marker"],
        setup_commands=["echo 'setup-done' > /app/build_marker"],
        workdir_mount=(str(workdir), "/app", "rw"),
    )
    handle = runtime.create_container(config)

    try:
        handle.build()
        # The marker file should exist on the host mount after setup
        assert (workdir / "build_marker").exists()
        content = (workdir / "build_marker").read_text().strip()
        assert "setup-done" in content
    finally:
        handle.cleanup()


def test_status_reports_exit_code(runtime: ContainerdRuntime):
    """A command that exits with code 1 should report exit_code=1."""
    config = _make_config(["sh", "-c", "exit 1"])
    handle = runtime.create_container(config)

    try:
        handle.run()
        status = _wait_for_exit(handle)

        assert not status.running
        assert status.exit_code == 1
    finally:
        handle.cleanup()


def test_cleanup_removes_sandbox(runtime: ContainerdRuntime):
    """After cleanup(), the sandbox should no longer exist."""
    config = _make_config(["echo", "cleanup-test"])
    handle = runtime.create_container(config)
    sandbox_id = handle.sandbox_id

    handle.run()
    _wait_for_exit(handle)
    handle.cleanup()

    result = subprocess.run(
        ["crictl", "inspectp", sandbox_id],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, "Sandbox should have been removed"


def test_host_network_mode(runtime: ContainerdRuntime):
    """Verify the sandbox was created with host network (NODE mode)."""
    config = _make_config(["echo", "network-test"])
    handle = runtime.create_container(config)

    try:
        sandbox_id = handle.sandbox_id

        # crictl requires flags before positional args
        result = subprocess.run(
            ["crictl", "inspectp", "-o", "json", sandbox_id],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        sandbox_info = json.loads(result.stdout)
        ns_options = sandbox_info.get("status", {}).get("linux", {}).get("namespaces", {}).get("options", {})
        network_mode = ns_options.get("network", "")
        assert network_mode == "NODE", f"Expected NODE network mode, got {network_mode}"
    finally:
        handle.cleanup()


def test_list_containers(runtime: ContainerdRuntime):
    """Creating two containers should make list_containers() return both."""
    config_a = _make_config(["sleep", "60"], task_id="list-task-a")
    config_b = _make_config(["sleep", "60"], task_id="list-task-b")

    handle_a = runtime.create_container(config_a)
    handle_b = runtime.create_container(config_b)

    try:
        handles = runtime.list_containers()
        assert len(handles) == 2
        sandbox_ids = {h.sandbox_id for h in handles}
        assert handle_a.sandbox_id in sandbox_ids
        assert handle_b.sandbox_id in sandbox_ids
    finally:
        handle_a.cleanup()
        handle_b.cleanup()


def test_runtime_cleanup(runtime: ContainerdRuntime):
    """runtime.cleanup() should remove all managed containers and sandboxes."""
    config_a = _make_config(["sleep", "60"], task_id="cleanup-a")
    config_b = _make_config(["sleep", "60"], task_id="cleanup-b")

    handle_a = runtime.create_container(config_a)
    handle_b = runtime.create_container(config_b)

    sandbox_a = handle_a.sandbox_id
    sandbox_b = handle_b.sandbox_id

    # Start both
    handle_a.run()
    handle_b.run()

    runtime.cleanup()

    # Both sandboxes should be gone
    for sid in [sandbox_a, sandbox_b]:
        result = subprocess.run(
            ["crictl", "inspectp", sid],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0, f"Sandbox {sid} should have been removed"

    assert len(runtime.list_containers()) == 0
