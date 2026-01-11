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

import asyncio
import subprocess

import pytest

from fluster.cluster.worker.runtime import ContainerConfig, DockerRuntime


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


@pytest.mark.asyncio
@pytest.mark.slow
async def test_run_simple_container_success(runtime):
    """Test running a simple container with exit code 0."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    config = ContainerConfig(
        image="alpine:latest",
        command=["echo", "Hello World"],
        env={},
    )

    result = await runtime.run(config)

    assert result.exit_code == 0
    assert result.container_id is not None
    assert len(result.container_id) > 0
    assert result.finished_at > result.started_at
    assert result.error is None

    # Cleanup
    await runtime.remove(result.container_id)


@pytest.mark.asyncio
@pytest.mark.slow
async def test_run_container_with_failure_exit_code(runtime):
    """Test container that exits with non-zero code."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    config = ContainerConfig(
        image="alpine:latest",
        command=["sh", "-c", "exit 42"],
        env={},
    )

    result = await runtime.run(config)

    assert result.exit_code == 42
    assert result.container_id is not None
    assert result.finished_at > result.started_at
    assert result.error is None

    # Cleanup
    await runtime.remove(result.container_id)


@pytest.mark.asyncio
@pytest.mark.slow
async def test_resource_limits_cpu(runtime):
    """Test that CPU limits are applied to container."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    # Set CPU limit to 500 millicores (0.5 CPU)
    config = ContainerConfig(
        image="alpine:latest",
        command=["echo", "test"],
        env={},
        cpu_millicores=500,
    )

    result = await runtime.run(config)

    assert result.exit_code == 0

    # Inspect container to verify CPU limit was set
    proc = await asyncio.create_subprocess_exec(
        "docker",
        "inspect",
        result.container_id,
        "--format",
        "{{.HostConfig.NanoCpus}}",
        stdout=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    nano_cpus = int(stdout.decode().strip())

    # 500 millicores = 0.5 CPUs = 500000000 nanocpus
    expected_nano_cpus = 500_000_000
    assert nano_cpus == expected_nano_cpus

    # Cleanup
    await runtime.remove(result.container_id)


@pytest.mark.asyncio
@pytest.mark.slow
async def test_resource_limits_memory(runtime):
    """Test that memory limits are applied to container."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    # Set memory limit to 256 MB
    config = ContainerConfig(
        image="alpine:latest",
        command=["echo", "test"],
        env={},
        memory_mb=256,
    )

    result = await runtime.run(config)

    assert result.exit_code == 0

    # Inspect container to verify memory limit was set
    proc = await asyncio.create_subprocess_exec(
        "docker",
        "inspect",
        result.container_id,
        "--format",
        "{{.HostConfig.Memory}}",
        stdout=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    memory_bytes = int(stdout.decode().strip())

    # 256 MB = 268435456 bytes
    expected_bytes = 256 * 1024 * 1024
    assert memory_bytes == expected_bytes

    # Cleanup
    await runtime.remove(result.container_id)


@pytest.mark.asyncio
@pytest.mark.slow
async def test_resource_limits_combined(runtime):
    """Test that CPU and memory limits work together."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    config = ContainerConfig(
        image="alpine:latest",
        command=["echo", "test"],
        env={},
        cpu_millicores=1000,
        memory_mb=512,
    )

    result = await runtime.run(config)

    assert result.exit_code == 0

    # Verify both limits
    proc = await asyncio.create_subprocess_exec(
        "docker",
        "inspect",
        result.container_id,
        "--format",
        "{{.HostConfig.NanoCpus}} {{.HostConfig.Memory}}",
        stdout=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    parts = stdout.decode().strip().split()
    nano_cpus = int(parts[0])
    memory_bytes = int(parts[1])

    assert nano_cpus == 1_000_000_000
    assert memory_bytes == 512 * 1024 * 1024

    # Cleanup
    await runtime.remove(result.container_id)


@pytest.mark.asyncio
@pytest.mark.slow
async def test_timeout_kills_container(runtime):
    """Test that timeout forces container to be killed."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    # Container that sleeps for 10 seconds but timeout is 1 second
    config = ContainerConfig(
        image="alpine:latest",
        command=["sleep", "10"],
        env={},
        timeout_seconds=1,
    )

    result = await runtime.run(config)

    # Should have timed out
    assert result.exit_code == -1
    assert result.error == "Timeout exceeded"
    assert result.finished_at > result.started_at

    # Execution time should be around 1 second (timeout)
    execution_time = result.finished_at - result.started_at
    assert 0.8 < execution_time < 2.0, f"Expected ~1s execution, got {execution_time}s"

    # Cleanup
    await runtime.remove(result.container_id)


@pytest.mark.asyncio
@pytest.mark.slow
async def test_log_streaming_captures_output(runtime):
    """Test that log streaming callback receives stdout and stderr."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    captured_logs = []

    def log_callback(stream: str, data: str):
        captured_logs.append((stream, data))

    # Use a longer-running container to ensure log streaming has time to capture output
    config = ContainerConfig(
        image="alpine:latest",
        command=["sh", "-c", "echo 'stdout message' && sleep 0.5 && echo 'stderr message' >&2 && sleep 0.5"],
        env={},
    )

    result = await runtime.run(config, log_callback=log_callback)

    assert result.exit_code == 0

    # Wait a moment for log streaming to flush
    await asyncio.sleep(0.2)

    # Verify we captured logs
    assert len(captured_logs) > 0

    # Check that we got both stdout and stderr
    stdout_messages = [data for stream, data in captured_logs if stream == "stdout"]
    stderr_messages = [data for stream, data in captured_logs if stream == "stderr"]

    # At least one message from each stream
    stdout_text = "".join(stdout_messages)
    stderr_text = "".join(stderr_messages)

    assert "stdout message" in stdout_text
    assert "stderr message" in stderr_text

    # Cleanup
    await runtime.remove(result.container_id)


@pytest.mark.asyncio
@pytest.mark.slow
async def test_log_streaming_multiple_lines(runtime):
    """Test log streaming with multiple lines of output."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    captured_logs = []

    def log_callback(stream: str, data: str):
        captured_logs.append((stream, data))

    # Add sleep to ensure log streaming has time to capture
    config = ContainerConfig(
        image="alpine:latest",
        command=[
            "sh",
            "-c",
            "for i in 1 2 3 4 5; do echo line $i; sleep 0.1; done",
        ],
        env={},
    )

    result = await runtime.run(config, log_callback=log_callback)

    assert result.exit_code == 0

    # Wait a moment for log streaming to flush
    await asyncio.sleep(0.2)

    assert len(captured_logs) > 0

    # Verify we got all lines
    stdout_text = "".join([data for stream, data in captured_logs if stream == "stdout"])
    for i in range(1, 6):
        assert f"line {i}" in stdout_text

    # Cleanup
    await runtime.remove(result.container_id)


@pytest.mark.asyncio
@pytest.mark.slow
async def test_environment_variables(runtime):
    """Test that environment variables are passed to container."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    config = ContainerConfig(
        image="alpine:latest",
        command=["sh", "-c", "echo $TEST_VAR && sleep 0.5"],
        env={"TEST_VAR": "test_value_123"},
    )

    captured_logs = []

    def log_callback(stream: str, data: str):
        captured_logs.append((stream, data))

    result = await runtime.run(config, log_callback=log_callback)

    assert result.exit_code == 0

    # Wait a moment for log streaming to flush
    await asyncio.sleep(0.2)

    # Verify environment variable was set
    stdout_text = "".join([data for stream, data in captured_logs if stream == "stdout"])
    assert "test_value_123" in stdout_text

    # Cleanup
    await runtime.remove(result.container_id)


@pytest.mark.asyncio
@pytest.mark.slow
async def test_multiple_environment_variables(runtime):
    """Test multiple environment variables."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    config = ContainerConfig(
        image="alpine:latest",
        command=["sh", "-c", "echo $VAR1 $VAR2 $VAR3 && sleep 0.5"],
        env={
            "VAR1": "value1",
            "VAR2": "value2",
            "VAR3": "value3",
        },
    )

    captured_logs = []

    def log_callback(stream: str, data: str):
        captured_logs.append((stream, data))

    result = await runtime.run(config, log_callback=log_callback)

    assert result.exit_code == 0

    # Wait a moment for log streaming to flush
    await asyncio.sleep(0.2)

    stdout_text = "".join([data for stream, data in captured_logs if stream == "stdout"])
    assert "value1" in stdout_text
    assert "value2" in stdout_text
    assert "value3" in stdout_text

    # Cleanup
    await runtime.remove(result.container_id)


@pytest.mark.asyncio
@pytest.mark.slow
async def test_mounts(runtime, tmp_path):
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
        command=["sh", "-c", "cat /mnt/test.txt && sleep 0.5"],
        env={},
        mounts=[(str(host_dir), "/mnt", "ro")],
    )

    captured_logs = []

    def log_callback(stream: str, data: str):
        captured_logs.append((stream, data))

    result = await runtime.run(config, log_callback=log_callback)

    assert result.exit_code == 0

    # Wait a moment for log streaming to flush
    await asyncio.sleep(0.2)

    # Verify file content was read from mount
    stdout_text = "".join([data for stream, data in captured_logs if stream == "stdout"])
    assert "test content from host" in stdout_text

    # Cleanup
    await runtime.remove(result.container_id)


@pytest.mark.asyncio
@pytest.mark.slow
async def test_mounts_writable(runtime, tmp_path):
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

    result = await runtime.run(config)

    assert result.exit_code == 0

    # Verify file was written from container
    output_file = host_dir / "output.txt"
    assert output_file.exists()
    assert "written from container" in output_file.read_text()

    # Cleanup
    await runtime.remove(result.container_id)


@pytest.mark.asyncio
@pytest.mark.slow
async def test_port_mapping(runtime):
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

    result = await runtime.run(config)

    assert result.exit_code == 0

    # Inspect container to verify port mappings were configured
    proc = await asyncio.create_subprocess_exec(
        "docker",
        "inspect",
        result.container_id,
        "--format",
        "{{json .HostConfig.PortBindings}}",
        stdout=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    port_bindings = stdout.decode().strip()

    # Should have port mappings (exact format depends on Docker version)
    # Just verify the ports appear in the output
    assert "8080" in port_bindings
    assert "9090" in port_bindings

    # Cleanup
    await runtime.remove(result.container_id)


@pytest.mark.asyncio
@pytest.mark.slow
async def test_workdir(runtime):
    """Test custom working directory."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    config = ContainerConfig(
        image="alpine:latest",
        command=["sh", "-c", "pwd && sleep 0.5"],
        env={},
        workdir="/custom/workdir",
    )

    captured_logs = []

    def log_callback(stream: str, data: str):
        captured_logs.append((stream, data))

    result = await runtime.run(config, log_callback=log_callback)

    assert result.exit_code == 0

    # Wait a moment for log streaming to flush
    await asyncio.sleep(0.2)

    # Verify working directory
    stdout_text = "".join([data for stream, data in captured_logs if stream == "stdout"])
    assert "/custom/workdir" in stdout_text

    # Cleanup
    await runtime.remove(result.container_id)


@pytest.mark.asyncio
@pytest.mark.slow
async def test_container_cleanup_with_remove(runtime):
    """Test that remove() properly cleans up containers."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    config = ContainerConfig(
        image="alpine:latest",
        command=["echo", "test"],
        env={},
    )

    result = await runtime.run(config)
    container_id = result.container_id

    # Container should exist before removal
    proc = await asyncio.create_subprocess_exec(
        "docker",
        "ps",
        "-a",
        "--filter",
        f"id={container_id}",
        "--format",
        "{{.ID}}",
        stdout=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    assert container_id[:12] in stdout.decode()

    # Remove container
    await runtime.remove(container_id)

    # Wait a moment for removal to complete
    await asyncio.sleep(0.1)

    # Container should not exist after removal
    proc = await asyncio.create_subprocess_exec(
        "docker",
        "ps",
        "-a",
        "--filter",
        f"id={container_id}",
        "--format",
        "{{.ID}}",
        stdout=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    assert stdout.decode().strip() == ""


@pytest.mark.asyncio
@pytest.mark.slow
async def test_security_hardening_no_new_privileges(runtime):
    """Test that no-new-privileges security option is applied."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    config = ContainerConfig(
        image="alpine:latest",
        command=["echo", "test"],
        env={},
    )

    result = await runtime.run(config)

    # Inspect container security options
    proc = await asyncio.create_subprocess_exec(
        "docker",
        "inspect",
        result.container_id,
        "--format",
        "{{json .HostConfig.SecurityOpt}}",
        stdout=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    security_opts = stdout.decode().strip()

    assert "no-new-privileges" in security_opts

    # Cleanup
    await runtime.remove(result.container_id)


@pytest.mark.asyncio
@pytest.mark.slow
async def test_security_hardening_cap_drop_all(runtime):
    """Test that all capabilities are dropped."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    config = ContainerConfig(
        image="alpine:latest",
        command=["echo", "test"],
        env={},
    )

    result = await runtime.run(config)

    # Inspect container capability drops
    proc = await asyncio.create_subprocess_exec(
        "docker",
        "inspect",
        result.container_id,
        "--format",
        "{{json .HostConfig.CapDrop}}",
        stdout=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    cap_drop = stdout.decode().strip()

    assert "ALL" in cap_drop

    # Cleanup
    await runtime.remove(result.container_id)


@pytest.mark.asyncio
@pytest.mark.slow
async def test_kill_with_sigterm(runtime):
    """Test killing container with SIGTERM."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    # Start a long-running container with a trap to handle SIGTERM
    config = ContainerConfig(
        image="alpine:latest",
        command=["sh", "-c", "trap 'exit 0' TERM; while true; do sleep 1; done"],
        env={},
    )

    # Run in background
    container_id = await runtime._create_container(config)
    await runtime._start_container(container_id)

    # Wait a moment for container to be running
    await asyncio.sleep(0.5)

    # Kill with SIGTERM
    await runtime.kill(container_id, force=False)

    # Wait longer for graceful shutdown
    await asyncio.sleep(1.0)

    # Verify container is stopped
    proc = await asyncio.create_subprocess_exec(
        "docker",
        "inspect",
        container_id,
        "--format",
        "{{.State.Running}}",
        stdout=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    is_running = stdout.decode().strip()

    assert is_running == "false"

    # Cleanup
    await runtime.remove(container_id)


@pytest.mark.asyncio
@pytest.mark.slow
async def test_kill_with_sigkill(runtime):
    """Test killing container with SIGKILL (force)."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    # Start a long-running container
    config = ContainerConfig(
        image="alpine:latest",
        command=["sleep", "30"],
        env={},
    )

    container_id = await runtime._create_container(config)
    await runtime._start_container(container_id)

    # Wait a moment for container to be running
    await asyncio.sleep(0.5)

    # Force kill with SIGKILL
    await runtime.kill(container_id, force=True)

    # Wait for container to stop
    await asyncio.sleep(0.5)

    # Verify container is stopped
    proc = await asyncio.create_subprocess_exec(
        "docker",
        "inspect",
        container_id,
        "--format",
        "{{.State.Running}}",
        stdout=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    is_running = stdout.decode().strip()

    assert is_running == "false"

    # Cleanup
    await runtime.remove(container_id)


@pytest.mark.asyncio
@pytest.mark.slow
async def test_no_timeout_runs_to_completion(runtime):
    """Test container with no timeout runs to natural completion."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    # Container that sleeps briefly with no timeout
    config = ContainerConfig(
        image="alpine:latest",
        command=["sh", "-c", "sleep 0.5 && echo 'completed'"],
        env={},
        timeout_seconds=None,
    )

    captured_logs = []

    def log_callback(stream: str, data: str):
        captured_logs.append((stream, data))

    result = await runtime.run(config, log_callback=log_callback)

    # Should complete successfully
    assert result.exit_code == 0
    assert result.error is None

    # Should have completion message
    stdout_text = "".join([data for stream, data in captured_logs if stream == "stdout"])
    assert "completed" in stdout_text

    # Cleanup
    await runtime.remove(result.container_id)


@pytest.mark.asyncio
@pytest.mark.slow
async def test_timeout_longer_than_execution(runtime):
    """Test that timeout doesn't interfere when longer than execution time."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    # Container finishes in 0.5s with 5s timeout
    config = ContainerConfig(
        image="alpine:latest",
        command=["sh", "-c", "sleep 0.5 && echo 'done'"],
        env={},
        timeout_seconds=5,
    )

    captured_logs = []

    def log_callback(stream: str, data: str):
        captured_logs.append((stream, data))

    result = await runtime.run(config, log_callback=log_callback)

    # Should complete successfully without timeout
    assert result.exit_code == 0
    assert result.error is None

    # Execution time should be ~0.5s, not 5s
    execution_time = result.finished_at - result.started_at
    assert execution_time < 2.0, f"Expected <2s execution, got {execution_time}s"

    stdout_text = "".join([data for stream, data in captured_logs if stream == "stdout"])
    assert "done" in stdout_text

    # Cleanup
    await runtime.remove(result.container_id)
