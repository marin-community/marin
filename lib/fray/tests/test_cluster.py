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

"""Tests for Cluster ABC implementations (LocalCluster and RayCluster).

Parameterized tests run for both implementations via fixtures.
Implementation-specific tests are kept as separate manual tests.
"""

import json
import logging
import os
import sys

import pytest
from fray.cluster import (
    Entrypoint,
    JobRequest,
    ResourceConfig,
    create_environment,
)


def test_cluster_launch(cluster):
    request = JobRequest(
        name="test-simple-job",
        entrypoint=Entrypoint(binary="python", args=["-m", "json.tool", "--help"]),
        environment=create_environment(),
    )

    job_id = cluster.launch(request)
    assert job_id is not None

    info = cluster.poll(job_id)
    assert info.job_id == job_id
    assert info.name == "test-simple-job"
    assert info.status in ["pending", "running", "succeeded", "failed"]


def test_cluster_list_jobs(cluster):
    request = JobRequest(
        name="list-test-job",
        entrypoint=Entrypoint(binary="python", args=["-m", "json.tool"]),
        environment=create_environment(),
    )

    job_id = cluster.launch(request)
    jobs = cluster.list_jobs()
    assert isinstance(jobs, list)
    assert len(jobs) >= 1
    assert any(job.job_id == job_id for job in jobs)


def test_cluster_terminate(cluster, cluster_type):
    request = JobRequest(
        name="terminate-test-job",
        entrypoint=Entrypoint(binary="python", args=["-m", "time", "sleep", "10"]),
        environment=create_environment(),
    )

    job_id = cluster.launch(request)
    cluster.terminate(job_id)
    info = cluster.poll(job_id)

    # ray... doesn't necessarily terminate jobs promptly
    if cluster_type != "ray":
        assert info.status in ["stopped", "failed", "succeeded"]


def test_cluster_job_success(cluster, cluster_type):
    request = JobRequest(
        name="success-test-job",
        entrypoint=Entrypoint(binary="python", args=["-m", "json.tool", "--help"]),
        environment=create_environment(),
    )

    job_id = cluster.launch(request)
    info = cluster.wait(job_id)

    assert info.status == "succeeded"
    assert info.error_message is None


def test_environment_integration(cluster, cluster_type, tmp_path):
    """Validate that environment variables are propogated into the job as expected."""

    def _check_env_closure(output_path: str):
        with open(output_path, "w") as f:
            json.dump(dict(os.environ), f)

    output_path = str(tmp_path / "env.json")

    request = JobRequest(
        name="env-integration-test",
        entrypoint=Entrypoint(
            callable=_check_env_closure,
            function_args={"output_path": output_path},
        ),
        environment=create_environment(
            env_vars={"TEST_INTEGRATION_VAR": "test_value_123"},
        ),
    )

    job_id = cluster.launch(request)
    info = cluster.wait(job_id)

    assert info.status == "succeeded"
    if not os.path.exists(output_path):
        pytest.fail(f"Output file {output_path} was not created by the job.")

    with open(output_path) as f:
        env_vars = json.load(f)

    # Verify our custom var
    assert (
        "TEST_INTEGRATION_VAR" in env_vars
    ), f"TEST_INTEGRATION_VAR should be set by the cluster: found {env_vars.keys()}"
    assert (
        env_vars["TEST_INTEGRATION_VAR"] == "test_value_123"
    ), f"TEST_INTEGRATION_VAR should be set to 'test_value_123': found {env_vars['TEST_INTEGRATION_VAR']}"
    assert "FRAY_CLUSTER_SPEC" in env_vars, f"FRAY_CLUSTER_SPEC should be set by the cluster: found {env_vars.keys()}"


def test_local_cluster_replica_integration(local_cluster, tmp_path, caplog):
    """Integration test for replica functionality: env vars, logs, status aggregation."""

    def replica_worker(output_path: str):
        replica_id = int(os.environ.get("FRAY_REPLICA_ID", "0"))
        replica_count = os.environ.get("FRAY_REPLICA_COUNT", "MISSING")

        print(f"Hello from replica {replica_id}")

        output_file = f"{output_path}_{replica_id}.json"
        with open(output_file, "w") as f:
            json.dump({"replica_id": str(replica_id), "replica_count": replica_count}, f)

        if replica_id == 1:
            print("Replica 1 intentionally failing")
            sys.exit(1)

    output_path = str(tmp_path / "replica_output")

    request = JobRequest(
        name="replica-integration-test",
        entrypoint=Entrypoint(callable=replica_worker, function_args={"output_path": output_path}),
        resources=ResourceConfig(replicas=3),
        environment=create_environment(),
    )

    job_id = local_cluster.launch(request)

    with caplog.at_level(logging.INFO):
        job_info = local_cluster.monitor(job_id)

    logs = caplog.text
    assert "[replica-0]" in logs and "Hello from replica 0" in logs, "No logs from replica-0"
    assert "[replica-1]" in logs and "Replica 1 intentionally failing" in logs, "No logs from replica-1"
    assert "[replica-2]" in logs and "Hello from replica 2" in logs, "No logs from replica-2"

    assert job_info.status == "failed"
    assert job_info.error_message is not None
    assert job_info.tasks[1].status == "failed"

    for replica_id in [0, 2]:
        output_file = f"{output_path}_{replica_id}.json"
        assert os.path.exists(output_file), f"Replica {replica_id} did not write output file"

        with open(output_file) as f:
            data = json.load(f)

        assert data["replica_id"] == str(replica_id)
        assert data["replica_count"] == "3"
