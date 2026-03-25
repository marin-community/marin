# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""E2E integration tests for iris job CLI helpers that boot a real local cluster."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

from iris.client import IrisClient
from iris.cli.job import load_env_vars, run_iris_job
from iris.cluster.config import connect_cluster, load_config, make_local_config

pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module")
def local_cluster_and_config(tmp_path_factory):
    """Start local cluster and create config file for it."""
    tmp_path = tmp_path_factory.mktemp("iris_run")
    iris_root = Path(__file__).resolve().parents[3]
    test_config_path = iris_root / "examples" / "test.yaml"

    config = load_config(test_config_path)
    config = make_local_config(config)

    with connect_cluster(config) as url:
        test_config = tmp_path / "cluster.yaml"
        test_config.write_text(
            yaml.dump(
                {
                    "platform": {"local": {}},
                    "defaults": {
                        "worker": {"controller_address": url},
                    },
                    "scale_groups": {
                        "local-cpu": {
                            "min_slices": 1,
                            "max_slices": 1,
                            "num_vms": 1,
                            "resources": {
                                "cpu": 1,
                                "ram": "1GB",
                                "disk": 0,
                                "device_type": "cpu",
                                "device_count": 0,
                                "preemptible": False,
                            },
                        }
                    },
                }
            )
        )

        client = IrisClient.remote(url, workspace=iris_root)
        yield test_config, url, client


@pytest.mark.timeout(120)
def test_iris_run_cli_simple_job(local_cluster_and_config, tmp_path):
    """Test iris job submission runs a simple job successfully."""
    _test_config, url, _client = local_cluster_and_config

    test_script = tmp_path / "test.py"
    test_script.write_text('print("SUCCESS"); exit(0)')

    exit_code = run_iris_job(
        controller_url=url,
        command=[sys.executable, str(test_script)],
        env_vars={},
        wait=True,
    )

    assert exit_code == 0


@pytest.mark.timeout(120)
def test_iris_run_cli_env_vars_propagate(local_cluster_and_config, tmp_path):
    """Test environment variables reach the job."""
    _test_config, url, _client = local_cluster_and_config

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


@pytest.mark.timeout(120)
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
