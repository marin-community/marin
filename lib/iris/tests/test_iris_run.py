# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for iris job CLI helpers."""

import sys
from pathlib import Path

import pytest
import yaml

from iris.client import IrisClient
from iris.cluster.config import IrisConfig, make_local_config, load_config
from iris.cluster.manager import connect_cluster
from iris.cli.job import (
    build_resources,
    load_env_vars,
    run_iris_job,
)
from iris.cluster.types import ConstraintOp

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


def test_build_resources_gpu():
    """Test GPU spec parsing in build_resources."""
    spec = build_resources(tpu=None, gpu="H100x8")
    assert spec.device.HasField("gpu")
    assert spec.device.gpu.variant == "H100"
    assert spec.device.gpu.count == 8

    # Bare count defaults to empty variant
    spec = build_resources(tpu=None, gpu="4")
    assert spec.device.gpu.variant == ""
    assert spec.device.gpu.count == 4

    # Bare variant defaults to count=1
    spec = build_resources(tpu=None, gpu="A100")
    assert spec.device.gpu.variant == "A100"
    assert spec.device.gpu.count == 1


# Integration tests using local cluster


@pytest.fixture
def local_cluster_and_config(tmp_path):
    """Start local cluster and create config file for it."""
    iris_root = Path(__file__).resolve().parents[1]
    demo_config_path = iris_root / "examples" / "demo.yaml"

    config = load_config(demo_config_path)
    config = make_local_config(config)

    with connect_cluster(config) as url:
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
                            "num_vms": 1,
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


def test_run_iris_job_adds_zone_constraint(monkeypatch):
    """run_iris_job forwards a zone placement constraint."""
    captured: dict[str, object] = {}

    def _fake_submit_and_wait_job(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr("iris.cli.job._submit_and_wait_job", _fake_submit_and_wait_job)

    exit_code = run_iris_job(
        controller_url="http://controller:10000",
        command=[sys.executable, "-c", "print('ok')"],
        env_vars={},
        wait=False,
        zone="us-central2-b",
    )

    assert exit_code == 0
    constraints = captured["constraints"]
    assert constraints is not None
    assert len(constraints) == 1
    assert constraints[0].key == "zone"
    assert constraints[0].op == ConstraintOp.EQ
    assert constraints[0].value == "us-central2-b"


def test_run_iris_job_adds_region_and_zone_constraints(monkeypatch):
    """run_iris_job combines region and zone constraints when both are set."""
    captured: dict[str, object] = {}

    def _fake_submit_and_wait_job(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr("iris.cli.job._submit_and_wait_job", _fake_submit_and_wait_job)

    exit_code = run_iris_job(
        controller_url="http://controller:10000",
        command=[sys.executable, "-c", "print('ok')"],
        env_vars={},
        wait=False,
        regions=("us-central2",),
        zone="us-central2-b",
    )

    assert exit_code == 0
    constraints = captured["constraints"]
    assert constraints is not None
    assert len(constraints) == 2

    region_constraints = [c for c in constraints if c.key == "region"]
    assert len(region_constraints) == 1
    assert region_constraints[0].op == ConstraintOp.EQ
    assert region_constraints[0].value == "us-central2"

    zone_constraints = [c for c in constraints if c.key == "zone"]
    assert len(zone_constraints) == 1
    assert zone_constraints[0].op == ConstraintOp.EQ
    assert zone_constraints[0].value == "us-central2-b"
