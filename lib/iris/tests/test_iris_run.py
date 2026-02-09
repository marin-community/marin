# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for iris job CLI helpers."""

import sys
from pathlib import Path

import pytest
import yaml

from iris.client import IrisClient
from iris.cluster.vm.cluster_manager import ClusterManager
from iris.cluster.vm.config import IrisConfig, make_local_config, load_config
from iris.cli.job import (
    build_resources,
    load_env_vars,
    run_iris_job,
)

# Unit tests for error handling and edge cases (not trivial assertions)


def test_load_env_vars_single_key():
    """Test env var with no value (empty string)."""
    result = load_env_vars([["KEY_ONLY"]])
    assert result["KEY_ONLY"] == ""


def test_load_env_vars_invalid_key():
    """Test error on key with = sign."""
    with pytest.raises(ValueError, match="cannot contain '='"):
        load_env_vars([["KEY=VALUE"]])


def test_iris_config_missing_file(tmp_path):
    """Test error on missing config file."""
    with pytest.raises(FileNotFoundError):
        IrisConfig.load(tmp_path / "nonexistent.yaml")


def test_iris_config_empty_file(tmp_path):
    """Test error on empty config file."""
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text("")
    with pytest.raises(ValueError, match="Config file is empty"):
        IrisConfig.load(bad_config)


def test_build_resources_gpu_not_supported():
    """Test that GPU raises error."""
    with pytest.raises(ValueError, match="GPU support not yet implemented"):
        build_resources(tpu=None, gpu=2, cpu=None, memory=None)


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
        # Uses the local platform with controller_address set
        test_config = tmp_path / "cluster.yaml"
        test_config.write_text(
            yaml.dump(
                {
                    "platform": {"local": {}},
                    "defaults": {
                        "bootstrap": {"controller_address": url},
                    },
                    "scale_groups": {
                        "local-cpu": {
                            "min_slices": 1,
                            "max_slices": 1,
                            "accelerator_type": "cpu",
                            "vm_type": "local_vm",
                            "slice_size": 1,
                            "resources": {
                                "cpu": 1,
                                "ram": "1GB",
                                "disk": 0,
                                "gpu_count": 0,
                                "tpu_count": 0,
                            },
                        }
                    },
                }
            )
        )

        client = IrisClient.remote(url, workspace=iris_root)
        yield test_config, url, client


@pytest.mark.slow
def test_iris_run_cli_simple_job(local_cluster_and_config, tmp_path):
    """Test iris job submission runs a simple job successfully."""
    _test_config, url, _client = local_cluster_and_config

    # Create test script that prints and exits
    test_script = tmp_path / "test.py"
    test_script.write_text('print("SUCCESS"); exit(0)')

    exit_code = run_iris_job(
        controller_url=url,
        command=[sys.executable, str(test_script)],
        env_vars={},
        wait=True,
    )

    assert exit_code == 0


@pytest.mark.slow
def test_iris_run_cli_env_vars_propagate(local_cluster_and_config, tmp_path):
    """Test environment variables reach the job."""
    _test_config, url, _client = local_cluster_and_config

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
        controller_url=url,
        command=[sys.executable, str(test_script)],
        env_vars=env_vars,
        wait=True,
    )

    assert exit_code == 0


@pytest.mark.slow
def test_iris_run_cli_job_failure(local_cluster_and_config, tmp_path):
    """Test job submission returns non-zero on job failure."""
    _test_config, url, _client = local_cluster_and_config

    test_script = tmp_path / "fail.py"
    test_script.write_text("exit(1)")

    exit_code = run_iris_job(
        controller_url=url,
        command=[sys.executable, str(test_script)],
        env_vars={},
        wait=True,
    )

    assert exit_code == 1
