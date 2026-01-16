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

"""Tests for DockerRuntime."""

import subprocess
import time

import pytest

from iris.rpc import cluster_pb2
from iris.cluster.worker.docker import ContainerConfig, DockerRuntime


def check_docker_available():
    """Check if Docker is available and running."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            check=True,
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


@pytest.fixture
def runtime():
    """Create DockerRuntime instance."""
    return DockerRuntime()


@pytest.mark.slow
def test_create_and_start_container(runtime):
    """Test creating and starting a simple container."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    config = ContainerConfig(
        image="alpine:latest",
        command=["echo", "Hello World"],
        env={},
    )

    # Create container
    container_id = runtime.create_container(config)
    assert container_id is not None
    assert len(container_id) > 0

    # Start container
    runtime.start_container(container_id)

    # Wait for completion
    max_wait = 5
    start = time.time()
    while time.time() - start < max_wait:
        status = runtime.inspect(container_id)
        if not status.running:
            break
        time.sleep(0.1)

    # Check final status
    status = runtime.inspect(container_id)
    assert not status.running
    assert status.exit_code == 0
    assert status.error is None

    # Cleanup
    runtime.remove(container_id)


@pytest.mark.slow
def test_container_with_failure_exit_code(runtime):
    """Test container that exits with non-zero code."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    config = ContainerConfig(
        image="alpine:latest",
        command=["sh", "-c", "exit 42"],
        env={},
    )

    container_id = runtime.create_container(config)
    runtime.start_container(container_id)

    # Wait for completion
    max_wait = 5
    start = time.time()
    while time.time() - start < max_wait:
        status = runtime.inspect(container_id)
        if not status.running:
            break
        time.sleep(0.1)

    # Check exit code
    status = runtime.inspect(container_id)
    assert status.exit_code == 42
    assert status.error is None

    # Cleanup
    runtime.remove(container_id)


@pytest.mark.slow
def test_resource_limits_cpu(runtime):
    """Test that CPU limits are applied to container."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    # Set CPU limit to 1 core (1000 millicores)
    resources = cluster_pb2.ResourceSpec(cpu=1)
    config = ContainerConfig(
        image="alpine:latest",
        command=["echo", "test"],
        env={},
        resources=resources,
    )

    container_id = runtime.create_container(config)

    # Inspect container to verify CPU limit was set
    result = subprocess.run(
        [
            "docker",
            "inspect",
            container_id,
            "--format",
            "{{.HostConfig.NanoCpus}}",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    nano_cpus = int(result.stdout.strip())

    # 1 core = 1000 millicores = 1000000000 nanocpus
    expected_nano_cpus = 1_000_000_000
    assert nano_cpus == expected_nano_cpus

    # Cleanup
    runtime.remove(container_id)


@pytest.mark.slow
def test_resource_limits_memory(runtime):
    """Test that memory limits are applied to container."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    # Set memory limit to 256 MB
    resources = cluster_pb2.ResourceSpec(memory="256m")
    config = ContainerConfig(
        image="alpine:latest",
        command=["echo", "test"],
        env={},
        resources=resources,
    )

    container_id = runtime.create_container(config)

    # Inspect container to verify memory limit was set
    result = subprocess.run(
        [
            "docker",
            "inspect",
            container_id,
            "--format",
            "{{.HostConfig.Memory}}",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    memory_bytes = int(result.stdout.strip())

    # 256 MB = 268435456 bytes
    expected_bytes = 256 * 1024 * 1024
    assert memory_bytes == expected_bytes

    # Cleanup
    runtime.remove(container_id)


@pytest.mark.slow
def test_resource_limits_combined(runtime):
    """Test that CPU and memory limits work together."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    resources = cluster_pb2.ResourceSpec(cpu=1, memory="512m")
    config = ContainerConfig(
        image="alpine:latest",
        command=["echo", "test"],
        env={},
        resources=resources,
    )

    container_id = runtime.create_container(config)

    # Verify both limits
    result = subprocess.run(
        [
            "docker",
            "inspect",
            container_id,
            "--format",
            "{{.HostConfig.NanoCpus}} {{.HostConfig.Memory}}",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    parts = result.stdout.strip().split()
    nano_cpus = int(parts[0])
    memory_bytes = int(parts[1])

    assert nano_cpus == 1_000_000_000
    assert memory_bytes == 512 * 1024 * 1024

    # Cleanup
    runtime.remove(container_id)


@pytest.mark.slow
def test_environment_variables(runtime):
    """Test that environment variables are passed to container."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    config = ContainerConfig(
        image="alpine:latest",
        command=["sh", "-c", "echo $TEST_VAR"],
        env={"TEST_VAR": "test_value_123"},
    )

    container_id = runtime.create_container(config)
    runtime.start_container(container_id)

    # Wait for completion
    max_wait = 5
    start = time.time()
    while time.time() - start < max_wait:
        status = runtime.inspect(container_id)
        if not status.running:
            break
        time.sleep(0.1)

    status = runtime.inspect(container_id)
    assert status.exit_code == 0

    # Cleanup
    runtime.remove(container_id)


@pytest.mark.slow
def test_multiple_environment_variables(runtime):
    """Test multiple environment variables."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    config = ContainerConfig(
        image="alpine:latest",
        command=["sh", "-c", "echo $VAR1 $VAR2 $VAR3"],
        env={
            "VAR1": "value1",
            "VAR2": "value2",
            "VAR3": "value3",
        },
    )

    container_id = runtime.create_container(config)
    runtime.start_container(container_id)

    # Wait for completion
    max_wait = 5
    start = time.time()
    while time.time() - start < max_wait:
        status = runtime.inspect(container_id)
        if not status.running:
            break
        time.sleep(0.1)

    status = runtime.inspect(container_id)
    assert status.exit_code == 0

    # Cleanup
    runtime.remove(container_id)


@pytest.mark.slow
def test_mounts(runtime, tmp_path):
    """Test that volume mounts work correctly."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    # Create a test file on host
    host_dir = tmp_path / "host_mount"
    host_dir.mkdir()
    test_file = host_dir / "test.txt"
    test_file.write_text("test content from host")

    config = ContainerConfig(
        image="alpine:latest",
        command=["sh", "-c", "cat /mnt/test.txt"],
        env={},
        mounts=[(str(host_dir), "/mnt", "ro")],
    )

    container_id = runtime.create_container(config)
    runtime.start_container(container_id)

    # Wait for completion
    max_wait = 5
    start = time.time()
    while time.time() - start < max_wait:
        status = runtime.inspect(container_id)
        if not status.running:
            break
        time.sleep(0.1)

    status = runtime.inspect(container_id)
    assert status.exit_code == 0

    # Cleanup
    runtime.remove(container_id)


@pytest.mark.slow
def test_mounts_writable(runtime, tmp_path):
    """Test writable volume mounts."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    # Create a directory for writable mount
    host_dir = tmp_path / "writable_mount"
    host_dir.mkdir()

    config = ContainerConfig(
        image="alpine:latest",
        command=["sh", "-c", "echo 'written from container' > /mnt/output.txt"],
        env={},
        mounts=[(str(host_dir), "/mnt", "rw")],
    )

    container_id = runtime.create_container(config)
    runtime.start_container(container_id)

    # Wait for completion
    max_wait = 5
    start = time.time()
    while time.time() - start < max_wait:
        status = runtime.inspect(container_id)
        if not status.running:
            break
        time.sleep(0.1)

    status = runtime.inspect(container_id)
    assert status.exit_code == 0

    # Verify file was written from container
    output_file = host_dir / "output.txt"
    assert output_file.exists()
    assert "written from container" in output_file.read_text()

    # Cleanup
    runtime.remove(container_id)


@pytest.mark.slow
def test_port_mapping(runtime):
    """Test port mapping configuration."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    # Note: We can't easily test actual network connectivity in unit tests,
    # but we can verify the port mapping configuration is applied
    config = ContainerConfig(
        image="alpine:latest",
        command=["echo", "test"],
        env={},
        ports={"http": 8080, "metrics": 9090},
    )

    container_id = runtime.create_container(config)
    runtime.start_container(container_id)

    # Wait for completion
    max_wait = 5
    start = time.time()
    while time.time() - start < max_wait:
        status = runtime.inspect(container_id)
        if not status.running:
            break
        time.sleep(0.1)

    status = runtime.inspect(container_id)
    assert status.exit_code == 0

    # Inspect container to verify port mappings were configured
    result = subprocess.run(
        [
            "docker",
            "inspect",
            container_id,
            "--format",
            "{{json .HostConfig.PortBindings}}",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    port_bindings = result.stdout.strip()

    # Should have port mappings (exact format depends on Docker version)
    # Just verify the ports appear in the output
    assert "8080" in port_bindings
    assert "9090" in port_bindings

    # Cleanup
    runtime.remove(container_id)


@pytest.mark.slow
def test_workdir(runtime):
    """Test custom working directory."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    config = ContainerConfig(
        image="alpine:latest",
        command=["sh", "-c", "pwd"],
        env={},
        workdir="/custom/workdir",
    )

    container_id = runtime.create_container(config)
    runtime.start_container(container_id)

    # Wait for completion
    max_wait = 5
    start = time.time()
    while time.time() - start < max_wait:
        status = runtime.inspect(container_id)
        if not status.running:
            break
        time.sleep(0.1)

    status = runtime.inspect(container_id)
    assert status.exit_code == 0

    # Cleanup
    runtime.remove(container_id)


@pytest.mark.slow
def test_container_cleanup_with_remove(runtime):
    """Test that remove() properly cleans up containers."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    config = ContainerConfig(
        image="alpine:latest",
        command=["echo", "test"],
        env={},
    )

    container_id = runtime.create_container(config)
    runtime.start_container(container_id)

    # Wait for completion
    max_wait = 5
    start = time.time()
    while time.time() - start < max_wait:
        status = runtime.inspect(container_id)
        if not status.running:
            break
        time.sleep(0.1)

    # Container should exist before removal
    result = subprocess.run(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            f"id={container_id}",
            "--format",
            "{{.ID}}",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert container_id[:12] in result.stdout

    # Remove container
    runtime.remove(container_id)

    # Wait a moment for removal to complete
    time.sleep(0.1)

    # Container should not exist after removal
    result = subprocess.run(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            f"id={container_id}",
            "--format",
            "{{.ID}}",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == ""


@pytest.mark.slow
def test_security_hardening_no_new_privileges(runtime):
    """Test that no-new-privileges security option is applied."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    config = ContainerConfig(
        image="alpine:latest",
        command=["echo", "test"],
        env={},
    )

    container_id = runtime.create_container(config)

    # Inspect container security options
    result = subprocess.run(
        [
            "docker",
            "inspect",
            container_id,
            "--format",
            "{{json .HostConfig.SecurityOpt}}",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    security_opts = result.stdout.strip()

    assert "no-new-privileges" in security_opts

    # Cleanup
    runtime.remove(container_id)


@pytest.mark.slow
def test_security_hardening_cap_drop_all(runtime):
    """Test that all capabilities are dropped."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    config = ContainerConfig(
        image="alpine:latest",
        command=["echo", "test"],
        env={},
    )

    container_id = runtime.create_container(config)

    # Inspect container capability drops
    result = subprocess.run(
        [
            "docker",
            "inspect",
            container_id,
            "--format",
            "{{json .HostConfig.CapDrop}}",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    cap_drop = result.stdout.strip()

    assert "ALL" in cap_drop

    # Cleanup
    runtime.remove(container_id)


@pytest.mark.slow
def test_kill_with_sigterm(runtime):
    """Test killing container with SIGTERM."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    # Start a long-running container with a trap to handle SIGTERM
    config = ContainerConfig(
        image="alpine:latest",
        command=["sh", "-c", "trap 'exit 0' TERM; while true; do sleep 1; done"],
        env={},
    )

    container_id = runtime.create_container(config)
    runtime.start_container(container_id)

    # Wait a moment for container to be running
    time.sleep(0.5)

    # Kill with SIGTERM
    runtime.kill(container_id, force=False)

    # Wait longer for graceful shutdown
    time.sleep(1.0)

    # Verify container is stopped
    status = runtime.inspect(container_id)
    assert not status.running

    # Cleanup
    runtime.remove(container_id)


@pytest.mark.slow
def test_kill_with_sigkill(runtime):
    """Test killing container with SIGKILL (force)."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    # Start a long-running container
    config = ContainerConfig(
        image="alpine:latest",
        command=["sleep", "30"],
        env={},
    )

    container_id = runtime.create_container(config)
    runtime.start_container(container_id)

    # Wait a moment for container to be running
    time.sleep(0.5)

    # Force kill with SIGKILL
    runtime.kill(container_id, force=True)

    # Wait for container to stop
    time.sleep(0.5)

    # Verify container is stopped
    status = runtime.inspect(container_id)
    assert not status.running

    # Cleanup
    runtime.remove(container_id)


@pytest.mark.slow
def test_inspect_running_container(runtime):
    """Test inspect() on a running container."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    config = ContainerConfig(
        image="alpine:latest",
        command=["sleep", "10"],
        env={},
    )

    container_id = runtime.create_container(config)
    runtime.start_container(container_id)

    # Check status while running
    time.sleep(0.5)
    status = runtime.inspect(container_id)
    assert status.running
    assert status.exit_code is None

    # Kill and cleanup
    runtime.kill(container_id, force=True)
    runtime.remove(container_id)


@pytest.mark.slow
def test_inspect_stopped_container(runtime):
    """Test inspect() on a stopped container."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    config = ContainerConfig(
        image="alpine:latest",
        command=["echo", "test"],
        env={},
    )

    container_id = runtime.create_container(config)
    runtime.start_container(container_id)

    # Wait for completion
    max_wait = 5
    start = time.time()
    while time.time() - start < max_wait:
        status = runtime.inspect(container_id)
        if not status.running:
            break
        time.sleep(0.1)

    # Check final status
    status = runtime.inspect(container_id)
    assert not status.running
    assert status.exit_code == 0

    # Cleanup
    runtime.remove(container_id)
