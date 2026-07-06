# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from click.testing import CliRunner
from ducky.deploy import _EFFECTIVELY_UNLIMITED_RETRIES, cli, submit_ducky
from iris.rpc import job_pb2


class _CapturingClient:
    """Captures the kwargs submit_ducky passes to client.submit."""

    def __init__(self):
        self.kwargs = None

    def submit(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(job_id="/rav/ducky")


def test_submit_ducky_sets_effectively_unlimited_budgets():
    client = _CapturingClient()
    # a task-level default 0 on max_task_failures would fail the job on the first hard failure,
    # so all three budgets must be raised for best-effort-always-up.
    submit_ducky(client, name="ducky", region="us-east5", tpu="", cpu=4, memory="8GB", env_vars={})  # pyrefly: ignore
    assert client.kwargs["max_retries_preemption"] == _EFFECTIVELY_UNLIMITED_RETRIES
    assert client.kwargs["max_retries_failure"] == _EFFECTIVELY_UNLIMITED_RETRIES
    assert client.kwargs["max_task_failures"] == _EFFECTIVELY_UNLIMITED_RETRIES
    assert client.kwargs["existing_job_policy"] == job_pb2.EXISTING_JOB_POLICY_RECREATE


def test_submit_ducky_keep_policy_is_passed_through():
    client = _CapturingClient()
    submit_ducky(  # pyrefly: ignore
        client,
        name="ducky",
        region="us-east5",
        tpu="",
        cpu=4,
        memory="8GB",
        env_vars={},
        existing_job_policy=job_pb2.EXISTING_JOB_POLICY_KEEP,
    )
    assert client.kwargs["existing_job_policy"] == job_pb2.EXISTING_JOB_POLICY_KEEP


def test_deploy_rejects_cluster_and_controller_url_together():
    result = CliRunner().invoke(cli, ["--cluster", "marin", "--controller-url", "http://x"])
    assert result.exit_code != 0
    assert "not both" in result.output


def test_deploy_requires_a_target(monkeypatch):
    monkeypatch.delenv("IRIS_CONTROLLER_URL", raising=False)
    result = CliRunner().invoke(cli, [])
    assert result.exit_code != 0
    assert "--cluster" in result.output
