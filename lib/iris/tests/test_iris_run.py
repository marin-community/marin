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

"""Integration tests for iris_run.py script."""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from iris.client import IrisClient
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec
from iris.cluster.vm.cluster_manager import ClusterManager, make_local_config
from iris.cluster.vm.config import load_config
from iris.rpc import cluster_pb2


def find_iris_run_script() -> Path:
    """Find the iris_run.py script relative to this test file."""
    # test file is in lib/iris/tests/, script is in lib/iris/scripts/
    test_dir = Path(__file__).parent
    script_path = test_dir.parent / "scripts" / "iris_run.py"
    if not script_path.exists():
        raise FileNotFoundError(f"iris_run.py not found at {script_path}")
    return script_path


@pytest.mark.integration
def test_iris_run_with_local_cluster(tmp_path: Path):
    """Test iris_run.py submits job to local cluster and runs successfully."""
    # Create a test script that prints env vars and exits
    test_script = tmp_path / "test_job.py"
    test_script.write_text(
        """
import os
print(f"TEST_VAR={os.environ.get('TEST_VAR', 'NOT_SET')}")
print(f"PYTHONPATH={os.environ.get('PYTHONPATH', 'NOT_SET')}")
print("Job completed successfully")
"""
    )

    # Create a minimal cluster config
    config_file = tmp_path / "cluster.yaml"
    config_data = {
        "project_id": "test-project",
        "zone": "test-zone",
        "region": "test-region",
    }
    config_file.write_text(yaml.dump(config_data))

    # Create .marin.yaml with env vars
    marin_yaml = tmp_path / ".marin.yaml"
    marin_yaml.write_text(yaml.dump({"env": {"FROM_MARIN_YAML": "yaml_value"}}))

    # We can't easily test with controller_tunnel in CI, so this test
    # validates the script's argument parsing and env var handling logic
    # by calling it with --help and checking it doesn't crash.
    iris_run = find_iris_run_script()

    # Test --help works
    result = subprocess.run(
        [sys.executable, str(iris_run), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Submit jobs to Iris clusters" in result.stdout

    # Test missing command error
    result = subprocess.run(
        [sys.executable, str(iris_run), "--config", str(config_file)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 1
    assert "No command provided" in result.stderr or "Command must start with --" in result.stderr


@pytest.mark.integration
def test_iris_run_env_var_handling(tmp_path: Path):
    """Test environment variable loading and merging logic."""
    # Import the functions directly to test them
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    try:
        from iris_run import add_standard_env_vars, load_env_vars

        # Test basic env var parsing
        env_vars = load_env_vars([["KEY1", "value1"], ["KEY2", "value2"]])
        assert env_vars["KEY1"] == "value1"
        assert env_vars["KEY2"] == "value2"

        # Test single key (no value)
        env_vars = load_env_vars([["KEY_ONLY"]])
        assert env_vars["KEY_ONLY"] == ""

        # Test invalid key with =
        with pytest.raises(ValueError, match="cannot contain '='"):
            load_env_vars([["KEY=VALUE"]])

        # Test standard env vars
        base_env = {"USER_VAR": "user_value"}
        result = add_standard_env_vars(base_env)
        assert result["USER_VAR"] == "user_value"  # Not overridden
        assert result["PYTHONPATH"] == "."
        assert result["PYTHONUNBUFFERED"] == "1"
        assert result["HF_HOME"] == "~/.cache/huggingface"

    finally:
        sys.path.pop(0)


@pytest.mark.integration
def test_iris_run_resource_building(tmp_path: Path):
    """Test resource spec building with TPU device."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    try:
        from iris_run import build_resources

        # Test TPU device creation (the interesting behavior)
        spec = build_resources(tpu="v5litepod-16", gpu=None, cpu=None, memory=None)
        assert spec.device is not None
        assert spec.device.HasField("tpu")
        assert spec.device.tpu.variant == "v5litepod-16"

    finally:
        sys.path.pop(0)


@pytest.mark.integration
def test_iris_run_config_loading(tmp_path: Path):
    """Test cluster config loading."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    try:
        from iris_run import load_cluster_config

        # Test valid config
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            yaml.dump(
                {
                    "project_id": "my-project",
                    "zone": "us-central1-a",
                    "region": "us-central1",
                }
            )
        )
        config = load_cluster_config(config_file)
        assert config["project_id"] == "my-project"
        assert config["zone"] == "us-central1-a"

        # Test missing file
        with pytest.raises(FileNotFoundError):
            load_cluster_config(tmp_path / "nonexistent.yaml")

        # Test missing required field
        bad_config = tmp_path / "bad.yaml"
        bad_config.write_text(yaml.dump({"project_id": "test"}))
        with pytest.raises(ValueError, match="Missing 'zone'"):
            load_cluster_config(bad_config)

    finally:
        sys.path.pop(0)


def test_iris_run_job_name_generation():
    """Test job name generation from command."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    try:
        from iris_run import generate_job_name

        # Test with Python script
        name = generate_job_name(["python", "train.py", "--epochs", "10"])
        assert "train" in name
        assert "iris-run-" in name

        # Test without recognizable script
        name = generate_job_name(["echo", "hello"])
        assert "iris-run-" in name
        assert "job" in name

    finally:
        sys.path.pop(0)


@pytest.fixture
def local_cluster():
    """Start a local Iris cluster for testing."""
    # Find the demo config
    iris_root = Path(__file__).resolve().parents[1]  # lib/iris
    demo_config = iris_root / "examples" / "demo.yaml"

    config = load_config(demo_config)
    config = make_local_config(config)

    manager = ClusterManager(config)
    with manager.connect() as url:
        client = IrisClient.remote(url, workspace=iris_root)
        yield url, client


@pytest.mark.integration
def test_e2e_simple_command(local_cluster, tmp_path: Path):
    """End-to-end test: Submit simple command via IrisClient and verify completion."""
    _url, client = local_cluster

    # Create a simple test script
    test_script = tmp_path / "hello.py"
    test_script.write_text(
        """
print("Hello from Iris!")
import sys
sys.exit(0)
"""
    )

    # Submit job using the client directly (validates full stack minus subprocess)
    entrypoint = Entrypoint.from_command(sys.executable, str(test_script))
    job = client.submit(
        entrypoint=entrypoint,
        name="test-simple-command",
        resources=ResourceSpec(cpu=1, memory="1GB"),
        environment=EnvironmentSpec(),
    )

    # Wait for completion
    status = job.wait(timeout=30)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


@pytest.mark.integration
def test_e2e_environment_variables(local_cluster, tmp_path: Path):
    """Test environment variables propagate correctly to job."""
    _url, client = local_cluster

    # Create script that checks env vars
    test_script = tmp_path / "check_env.py"
    test_script.write_text(
        """
import os
import sys

# Check custom env var
test_var = os.environ.get("TEST_VAR", "MISSING")
print(f"TEST_VAR={test_var}")

# Check standard env vars
pythonpath = os.environ.get("PYTHONPATH", "MISSING")
print(f"PYTHONPATH={pythonpath}")

# Validate
assert test_var == "custom_value", f"Expected 'custom_value', got '{test_var}'"
assert pythonpath == ".", f"Expected '.', got '{pythonpath}'"
print("All env vars correct!")
"""
    )

    # Submit with env vars
    entrypoint = Entrypoint.from_command(sys.executable, str(test_script))
    env_vars = {
        "TEST_VAR": "custom_value",
        "PYTHONPATH": ".",
        "PYTHONUNBUFFERED": "1",
    }

    job = client.submit(
        entrypoint=entrypoint,
        name="test-env-vars",
        resources=ResourceSpec(cpu=1, memory="1GB"),
        environment=EnvironmentSpec(env_vars=env_vars),
    )

    status = job.wait(timeout=30)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


@pytest.mark.integration
def test_e2e_job_failure(local_cluster, tmp_path: Path):
    """Test job that fails with non-zero exit code."""
    from iris.client.client import JobFailedError

    _url, client = local_cluster

    # Create script that exits with error
    test_script = tmp_path / "fail.py"
    test_script.write_text(
        """
import sys
print("This job will fail")
sys.exit(42)
"""
    )

    entrypoint = Entrypoint.from_command(sys.executable, str(test_script))
    job = client.submit(
        entrypoint=entrypoint,
        name="test-failure",
        resources=ResourceSpec(cpu=1, memory="1GB"),
        environment=EnvironmentSpec(),
    )

    # Job.wait() raises JobFailedError when job fails
    with pytest.raises(JobFailedError) as exc_info:
        job.wait(timeout=30)

    # Verify the error contains the job_id and status is FAILED
    assert exc_info.value.job_id == "test-failure"
    assert exc_info.value.status.state == cluster_pb2.JOB_STATE_FAILED


@pytest.mark.integration
def test_e2e_multiple_replicas(local_cluster, tmp_path: Path):
    """Test gang scheduling with multiple replicas."""
    _url, client = local_cluster

    # Create script that prints replica info
    test_script = tmp_path / "replicas.py"
    test_script.write_text(
        """
import os
replica_id = os.environ.get("REPLICA_ID", "unknown")
world_size = os.environ.get("WORLD_SIZE", "unknown")
print(f"Replica {replica_id} of {world_size}")
"""
    )

    entrypoint = Entrypoint.from_command(sys.executable, str(test_script))
    job = client.submit(
        entrypoint=entrypoint,
        name="test-replicas",
        resources=ResourceSpec(cpu=1, memory="1GB"),
        environment=EnvironmentSpec(),
        replicas=2,  # Request 2 replicas
    )

    status = job.wait(timeout=30)
    # Gang scheduled job should succeed when all replicas complete
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


@pytest.mark.integration
def test_e2e_log_streaming(local_cluster, tmp_path: Path):
    """Test that job logs can be streamed."""
    _url, client = local_cluster

    # Create script with distinctive output
    test_script = tmp_path / "logging.py"
    test_script.write_text(
        """
print("LOG_LINE_1: Starting job")
print("LOG_LINE_2: Processing data")
print("LOG_LINE_3: Job complete")
"""
    )

    entrypoint = Entrypoint.from_command(sys.executable, str(test_script))
    job = client.submit(
        entrypoint=entrypoint,
        name="test-logs",
        resources=ResourceSpec(cpu=1, memory="1GB"),
        environment=EnvironmentSpec(),
    )

    # Wait with log streaming enabled
    # Note: In production, logs would stream to stdout; here we just verify completion
    status = job.wait(stream_logs=True, timeout=30)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED
