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

"""Tests for DockerRuntime.

These tests focus on observable behavior rather than Docker implementation details.
They verify that containers execute callables correctly, handle failures appropriately,
and can be managed through their lifecycle.
"""

import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from iris.cluster.types import Entrypoint
from iris.cluster.worker.docker import ContainerConfig, DockerRuntime
from iris.cluster.worker.env_probe import collect_workdir_size_mb
from iris.rpc import cluster_pb2

TEST_IMAGE = "iris-test-runtime:latest"


def build_test_image():
    """Build a minimal test image with cloudpickle installed.

    Uses the same Python version as the test environment to ensure cloudpickle
    bytecode compatibility.
    """
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    dockerfile = f"""\
FROM python:{py_version}-slim
RUN pip install --no-cache-dir cloudpickle
"""
    result = subprocess.run(
        ["docker", "build", "-t", TEST_IMAGE, "-"],
        input=dockerfile,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def wait_for_container_exit(runtime: DockerRuntime, container_id: str, max_wait: float = 5.0):
    """Wait for a container to stop running. Raises with logs if exit_code != 0."""
    start = time.time()
    while time.time() - start < max_wait:
        status = runtime.inspect(container_id)
        if not status.running:
            if status.exit_code != 0:
                logs = runtime.get_logs(container_id)
                log_lines = [f"{log.source}: {log.data}" for log in logs[-20:]]
                raise AssertionError(f"Container exited with code {status.exit_code}\n" + "\n".join(log_lines))
            return status
        time.sleep(0.1)
    raise TimeoutError(f"Container {container_id} did not exit within {max_wait}s")


@pytest.fixture(scope="module")
def test_image():
    """Build test image with cloudpickle."""
    if not build_test_image():
        pytest.skip("Failed to build test image")
    return TEST_IMAGE


@pytest.mark.slow
def test_create_and_start_container(docker_runtime, test_image):
    """Test creating and starting a container that executes a callable successfully."""

    def noop():
        pass

    config = ContainerConfig(
        image=test_image,
        entrypoint=Entrypoint.from_callable(noop),
        env={},
    )

    container_id = docker_runtime.create_container(config)
    docker_runtime.start_container(container_id)
    status = wait_for_container_exit(docker_runtime, container_id)

    assert not status.running
    assert status.exit_code == 0
    assert status.error is None

    docker_runtime.remove(container_id)


@pytest.mark.slow
def test_command_entrypoint(docker_runtime, test_image):
    """Test that command entrypoints execute correctly."""
    config = ContainerConfig(
        image=test_image,
        entrypoint=Entrypoint.from_command("python", "-c", "print('hello from command')"),
        env={},
    )

    container_id = docker_runtime.create_container(config)
    docker_runtime.start_container(container_id)
    status = wait_for_container_exit(docker_runtime, container_id)

    assert not status.running
    assert status.exit_code == 0
    assert status.error is None

    docker_runtime.remove(container_id)


@pytest.mark.slow
def test_container_with_failure_exit_code(docker_runtime, test_image):
    """Test container that exits with a non-zero code."""

    def exit_with_code(code: int):
        sys.exit(code)

    config = ContainerConfig(
        image=test_image,
        entrypoint=Entrypoint.from_callable(exit_with_code, 42),
        env={},
    )

    container_id = docker_runtime.create_container(config)
    docker_runtime.start_container(container_id)

    # Wait for container to exit (don't use wait_for_container_exit which raises on non-zero)
    status = None
    for _ in range(50):
        status = docker_runtime.inspect(container_id)
        if not status.running:
            break
        time.sleep(0.1)

    assert status is not None
    assert status.exit_code == 42
    assert status.error is None

    docker_runtime.remove(container_id)


@pytest.mark.slow
def test_kill_with_sigterm(docker_runtime, test_image):
    """Test killing container with SIGTERM allows graceful shutdown."""

    def sleep_and_handle_sigterm():
        def handler(signum, frame):
            sys.exit(0)

        signal.signal(signal.SIGTERM, handler)
        while True:
            time.sleep(1)

    config = ContainerConfig(
        image=test_image,
        entrypoint=Entrypoint.from_callable(sleep_and_handle_sigterm),
        env={},
    )

    container_id = docker_runtime.create_container(config)
    docker_runtime.start_container(container_id)

    time.sleep(0.5)

    docker_runtime.kill(container_id, force=False)
    time.sleep(1.0)

    status = docker_runtime.inspect(container_id)
    assert not status.running

    docker_runtime.remove(container_id)


@pytest.mark.slow
def test_kill_with_sigkill(docker_runtime, test_image):
    """Test killing container with SIGKILL (force) stops it immediately."""

    def sleep_forever():
        while True:
            time.sleep(1)

    config = ContainerConfig(
        image=test_image,
        entrypoint=Entrypoint.from_callable(sleep_forever),
        env={},
    )

    container_id = docker_runtime.create_container(config)
    docker_runtime.start_container(container_id)

    time.sleep(0.5)

    docker_runtime.kill(container_id, force=True)
    time.sleep(0.5)

    status = docker_runtime.inspect(container_id)
    assert not status.running

    docker_runtime.remove(container_id)


@pytest.mark.slow
def test_container_cleanup_with_remove(docker_runtime, test_image):
    """Test that remove() properly cleans up containers."""

    def noop():
        pass

    config = ContainerConfig(
        image=test_image,
        entrypoint=Entrypoint.from_callable(noop),
        env={},
    )

    container_id = docker_runtime.create_container(config)
    docker_runtime.start_container(container_id)
    wait_for_container_exit(docker_runtime, container_id)

    # Container should be inspectable before removal
    status = docker_runtime.inspect(container_id)
    assert status.error is None

    docker_runtime.remove(container_id)
    time.sleep(0.1)

    # After removal, inspect should indicate container not found
    status = docker_runtime.inspect(container_id)
    assert status.error is not None


@pytest.mark.slow
def test_inspect_running_container(docker_runtime, test_image):
    """Test inspect() reports running state correctly for an active container."""

    def sleep_forever():
        while True:
            time.sleep(1)

    config = ContainerConfig(
        image=test_image,
        entrypoint=Entrypoint.from_callable(sleep_forever),
        env={},
    )

    container_id = docker_runtime.create_container(config)
    docker_runtime.start_container(container_id)

    time.sleep(0.5)
    status = docker_runtime.inspect(container_id)
    assert status.running
    assert status.exit_code is None

    docker_runtime.kill(container_id, force=True)
    docker_runtime.remove(container_id)


def test_collect_workdir_size_mb_with_temp_directory(tmp_path):
    """Test workdir size calculation with a temporary directory."""
    (tmp_path / "file1.txt").write_text("x" * 1024 * 100)  # 100 KB
    (tmp_path / "file2.txt").write_text("y" * 1024 * 100)  # 100 KB

    size_mb = collect_workdir_size_mb(tmp_path)

    assert size_mb >= 1


def test_collect_workdir_size_mb_nonexistent_directory():
    """Test workdir size returns 0 for non-existent directory."""
    nonexistent = Path("/nonexistent/path/does/not/exist")

    size_mb = collect_workdir_size_mb(nonexistent)

    assert size_mb == 0


@pytest.mark.docker
def test_get_stats_invalid_container(docker_runtime):
    """Test that get_stats returns available=False for invalid container ID."""
    invalid_container_id = "nonexistent_container_12345"

    stats = docker_runtime.get_stats(invalid_container_id)

    assert stats.available is False


@pytest.mark.slow
def test_get_stats_from_running_container(docker_runtime, test_image):
    """Test get_stats returns positive values for a real running container."""

    def sleep_seconds(n: int):
        time.sleep(n)

    config = ContainerConfig(
        image=test_image,
        entrypoint=Entrypoint.from_callable(sleep_seconds, 5),
        env={},
    )

    container_id = docker_runtime.create_container(config)
    docker_runtime.start_container(container_id)

    time.sleep(0.5)

    try:
        stats = docker_runtime.get_stats(container_id)

        assert stats.available is True
        assert stats.memory_mb >= 0
        assert stats.process_count >= 1
        assert stats.cpu_percent >= 0

    finally:
        docker_runtime.kill(container_id, force=True)
        docker_runtime.remove(container_id)


@pytest.mark.slow
def test_get_stats_from_busy_container(docker_runtime, test_image):
    """Test get_stats returns higher values for a CPU-intensive container."""

    def busy_loop():
        while True:
            print("test")

    config = ContainerConfig(
        image=test_image,
        entrypoint=Entrypoint.from_callable(busy_loop),
        env={},
    )

    container_id = docker_runtime.create_container(config)
    docker_runtime.start_container(container_id)

    time.sleep(1.0)

    try:
        stats = docker_runtime.get_stats(container_id)

        assert stats.available is True
        assert stats.memory_mb >= 0
        assert stats.process_count >= 1
        assert stats.cpu_percent >= 0

    finally:
        docker_runtime.kill(container_id, force=True)
        docker_runtime.remove(container_id)


def test_tpu_metadata_flows_to_container_env():
    """Test that TPU worker metadata is correctly passed through to container env vars."""
    resources = cluster_pb2.ResourceSpecProto()
    resources.device.tpu.variant = "v5litepod-16"

    config = ContainerConfig(
        image="test",
        entrypoint=Entrypoint.from_command("echo", "hello"),
        env={},
        resources=resources,
    )

    config.worker_metadata = cluster_pb2.WorkerMetadata(
        tpu_name="iris-tpu-slice-001",
        tpu_worker_id="0",
        tpu_worker_hostnames="10.0.0.1,10.0.0.2",
        tpu_chips_per_host_bounds="2,2,1",
    )

    runtime = DockerRuntime()
    env = runtime._build_device_env_vars(config)

    assert env["TPU_NAME"] == "iris-tpu-slice-001"
    assert env["TPU_WORKER_ID"] == "0"
    assert env["TPU_WORKER_HOSTNAMES"] == "10.0.0.1,10.0.0.2"
    assert env["TPU_CHIPS_PER_HOST_BOUNDS"] == "2,2,1"
    assert env["JAX_COORDINATOR_ADDRESS"] == "10.0.0.1"
    assert env["JAX_NUM_PROCESSES"] == "2"
    assert env["JAX_PROCESS_ID"] == "0"
