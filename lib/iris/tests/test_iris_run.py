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

import sys
from pathlib import Path

import pytest
import yaml

from iris.client import IrisClient
from iris.cluster.vm.cluster_manager import ClusterManager, make_local_config
from iris.cluster.vm.config import load_config
from iris.iris_run import (
    add_standard_env_vars,
    build_resources,
    load_cluster_config,
    load_env_vars,
    run_iris_job,
)

# Unit tests for pure functions (no mocks, no sys.path hacks)


def test_load_env_vars_basic():
    """Test env var parsing from flags."""
    result = load_env_vars([["KEY1", "val1"], ["KEY2", "val2"]])
    assert result["KEY1"] == "val1"
    assert result["KEY2"] == "val2"


def test_load_env_vars_single_key():
    """Test env var with no value (empty string)."""
    result = load_env_vars([["KEY_ONLY"]])
    assert result["KEY_ONLY"] == ""


def test_load_env_vars_invalid_key():
    """Test error on key with = sign."""
    with pytest.raises(ValueError, match="cannot contain '='"):
        load_env_vars([["KEY=VALUE"]])


def test_load_cluster_config_valid(tmp_path):
    """Test loading valid config."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump({"project_id": "test", "zone": "us-central1-a"}))
    config = load_cluster_config(config_file)
    assert config["project_id"] == "test"
    assert config["zone"] == "us-central1-a"


def test_load_cluster_config_missing_file(tmp_path):
    """Test error on missing config file."""
    with pytest.raises(FileNotFoundError):
        load_cluster_config(tmp_path / "nonexistent.yaml")


def test_load_cluster_config_missing_zone(tmp_path):
    """Test error on missing zone field."""
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text(yaml.dump({"project_id": "test"}))
    with pytest.raises(ValueError, match="Missing 'zone'"):
        load_cluster_config(bad_config)


def test_load_cluster_config_missing_project_id(tmp_path):
    """Test error on missing project_id field."""
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text(yaml.dump({"zone": "us-central1-a"}))
    with pytest.raises(ValueError, match="Missing 'project_id'"):
        load_cluster_config(bad_config)


def test_build_resources_with_tpu():
    """Test ResourceSpec creation with TPU device."""
    spec = build_resources(tpu="v5litepod-16", gpu=None, cpu=None, memory=None)
    assert spec.device.HasField("tpu")
    assert spec.device.tpu.variant == "v5litepod-16"


def test_build_resources_defaults():
    """Test default CPU and memory values."""
    spec = build_resources(tpu=None, gpu=None, cpu=None, memory=None)
    assert spec.cpu == 1
    assert spec.memory == "2GB"


def test_build_resources_gpu_not_supported():
    """Test that GPU raises error."""
    with pytest.raises(ValueError, match="GPU support not yet implemented"):
        build_resources(tpu=None, gpu=2, cpu=None, memory=None)


def test_add_standard_env_vars():
    """Test standard env vars are added without overriding."""
    base_env = {"USER_VAR": "user_value", "PYTHONPATH": "/custom/path"}
    result = add_standard_env_vars(base_env)

    # User values preserved
    assert result["USER_VAR"] == "user_value"
    assert result["PYTHONPATH"] == "/custom/path"

    # Defaults added
    assert result["PYTHONUNBUFFERED"] == "1"
    assert result["HF_HOME"] == "~/.cache/huggingface"
    assert result["HF_HUB_ENABLE_HF_TRANSFER"] == "1"


# Integration tests using local cluster


@pytest.fixture
def local_cluster_and_config(tmp_path):
    """Start local cluster and create config file for it."""
    iris_root = Path(__file__).resolve().parents[1]
    demo_config_path = iris_root / "examples" / "demo.yaml"

    config = load_config(demo_config_path)
    config = make_local_config(config)

    manager = ClusterManager(config)
    with manager.connect() as url:
        # Create a test config file with controller_address for local access
        test_config = tmp_path / "cluster.yaml"
        test_config.write_text(
            yaml.dump(
                {
                    "zone": config.zone,
                    "region": config.region,
                    "controller_address": url,  # Direct URL for local cluster
                }
            )
        )

        client = IrisClient.remote(url, workspace=iris_root)
        yield test_config, url, client


@pytest.mark.integration
def test_iris_run_cli_simple_job(local_cluster_and_config, tmp_path):
    """Test iris_run.py submits and runs a simple job successfully."""
    test_config, _url, _client = local_cluster_and_config

    # Create test script that prints and exits
    test_script = tmp_path / "test.py"
    test_script.write_text('print("SUCCESS"); exit(0)')

    exit_code = run_iris_job(
        config_path=test_config,
        command=[sys.executable, str(test_script)],
        env_vars={},
        wait=True,
    )

    assert exit_code == 0


@pytest.mark.integration
def test_iris_run_cli_env_vars_propagate(local_cluster_and_config, tmp_path):
    """Test environment variables reach the job."""
    test_config, _url, _client = local_cluster_and_config

    # Create script that checks env var
    test_script = tmp_path / "check_env.py"
    test_script.write_text(
        """
import os
import sys
val = os.environ.get("TEST_VAR", "MISSING")
print(f"TEST_VAR={val}")
sys.exit(0 if val == "test_value" else 1)
"""
    )

    env_vars = load_env_vars([["TEST_VAR", "test_value"]])

    exit_code = run_iris_job(
        config_path=test_config,
        command=[sys.executable, str(test_script)],
        env_vars=env_vars,
        wait=True,
    )

    assert exit_code == 0


@pytest.mark.integration
def test_iris_run_cli_job_failure(local_cluster_and_config, tmp_path):
    """Test iris_run.py returns non-zero on job failure."""
    test_config, _url, _client = local_cluster_and_config

    test_script = tmp_path / "fail.py"
    test_script.write_text("exit(1)")

    exit_code = run_iris_job(
        config_path=test_config,
        command=[sys.executable, str(test_script)],
        env_vars={},
        wait=True,
    )

    assert exit_code == 1
