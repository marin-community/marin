# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from contextlib import nullcontext
from pathlib import Path

import pytest
from click.testing import CliRunner
from iris.cluster.types import JobName
from iris.rpc import job_pb2
from rigging.redaction import REDACTED_VALUE

from scripts.ci import iris_monitor


class FakeIrisClient:
    def __init__(self, jobs: list[job_pb2.JobStatus]) -> None:
        self.jobs = jobs
        self.terminated: list[str] = []

    def list_jobs(self, *, prefix: str) -> list[job_pb2.JobStatus]:
        return [job for job in self.jobs if job.job_id.startswith(prefix)]

    def terminate(self, job_id: JobName) -> None:
        self.terminated.append(job_id.to_wire())


def _pod(name: str, *, phase: str = "Running", ready: bool = True, deleting: bool = False) -> dict:
    metadata = {"name": name}
    if deleting:
        metadata["deletionTimestamp"] = "2026-05-06T12:00:00Z"
    return {
        "metadata": metadata,
        "status": {
            "phase": phase,
            "conditions": [{"type": "Ready", "status": "True" if ready else "False"}],
        },
    }


def _statuses(*pods: dict) -> list[iris_monitor.K8sPodStatus]:
    return iris_monitor._controller_pods_from_json(json.dumps({"items": list(pods)}))


def test_settled_coreweave_controller_requires_exactly_one_ready_pod() -> None:
    assert iris_monitor._settled_controller_pod_name(_statuses(_pod("iris-controller-new"))) == "iris-controller-new"

    assert iris_monitor._settled_controller_pod_name(_statuses()) is None
    assert (
        iris_monitor._settled_controller_pod_name(
            _statuses(
                _pod("iris-controller-old", deleting=True),
                _pod("iris-controller-new"),
            )
        )
        is None
    )
    assert iris_monitor._settled_controller_pod_name(_statuses(_pod("iris-controller-new", ready=False))) is None
    assert iris_monitor._settled_controller_pod_name(_statuses(_pod("iris-controller-new", phase="Pending"))) is None


def test_redact_pod_doc_redacts_env_values_and_preserves_context() -> None:
    pod = {
        "metadata": {"name": "worker-0"},
        "spec": {
            "containers": [
                {
                    "name": "runner",
                    "image": "registry.example/iris-runner:sha",
                    "resources": {"limits": {"nvidia.com/gpu": "8"}},
                    "env": [
                        {"name": "AWS_ACCESS_KEY_ID", "value": "AKIA_TEST_ACCESS"},
                        # Low-entropy secret only caught via name-based lift.
                        {"name": "WANDB_API_KEY", "value": "wandb-test-secret"},
                        {
                            "name": "IRIS_JOB_ENV",
                            "value": json.dumps(
                                {
                                    "AWS_SECRET_ACCESS_KEY": "nested-secret-key",
                                    "HF_TOKEN": "nested-hf-token",
                                    "LOG_LEVEL": "debug",
                                }
                            ),
                        },
                        {"name": "NORMAL_ENV", "value": "normal-env-value"},
                        {
                            "name": "HF_TOKEN",
                            "valueFrom": {"secretKeyRef": {"name": "hf-token", "key": "HF_TOKEN"}},
                        },
                    ],
                }
            ]
        },
    }

    redacted = iris_monitor._redact_pod_doc(pod)
    env_by_name = {entry["name"]: entry for entry in redacted["spec"]["containers"][0]["env"]}

    assert env_by_name["AWS_ACCESS_KEY_ID"]["value"] == REDACTED_VALUE
    assert env_by_name["WANDB_API_KEY"]["value"] == REDACTED_VALUE
    assert env_by_name["NORMAL_ENV"]["value"] == "normal-env-value"

    nested = json.loads(env_by_name["IRIS_JOB_ENV"]["value"])
    assert nested == {
        "AWS_SECRET_ACCESS_KEY": REDACTED_VALUE,
        "HF_TOKEN": REDACTED_VALUE,
        "LOG_LEVEL": "debug",
    }

    # valueFrom entries pass through untouched and never gain a phantom `value`.
    assert "value" not in env_by_name["HF_TOKEN"]
    assert env_by_name["HF_TOKEN"]["valueFrom"]["secretKeyRef"]["name"] == "hf-token"

    # Non-env pod context stays intact.
    assert redacted["spec"]["containers"][0]["image"] == "registry.example/iris-runner:sha"
    assert redacted["spec"]["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] == "8"


@pytest.mark.parametrize(
    "pending_reason",
    [
        (
            "Scheduler: No worker matches constraints and has sufficient resources\n\n"
            "Autoscaler: Unsatisfied autoscaler demand: tier_blocked: "
            "1 matching group(s) blocked by quota-pool tier monotonicity"
        ),
        'There is no more capacity in the zone "us-east5-b"',
    ],
)
@pytest.mark.parametrize("wait_option", ["--timeout", "--child-wait-timeout"])
def test_wait_resource_exhaustion_shutdown_is_a_successful_warning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    wait_option: str,
    pending_reason: str,
) -> None:
    parent_job_id = "/runner/canary"
    client = FakeIrisClient(
        [
            job_pb2.JobStatus(job_id=parent_job_id, state=job_pb2.JOB_STATE_RUNNING, has_children=True),
            job_pb2.JobStatus(
                job_id=f"{parent_job_id}/train",
                state=job_pb2.JOB_STATE_PENDING,
                pending_reason=pending_reason,
            ),
        ]
    )
    github_output = tmp_path / "github-output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(github_output))
    monkeypatch.setattr(iris_monitor, "open_iris_client", lambda **_kwargs: nullcontext(client))

    result = CliRunner().invoke(
        iris_monitor.cli,
        [
            "wait",
            "--job-id",
            parent_job_id,
            "--controller-url",
            "http://iris.test",
            "--poll-interval",
            "0.01",
            wait_option,
            "60",
            "--resource-exhaustion-policy",
            "shutdown",
            "--github-output",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "::warning title=Canary resource exhaustion::" in result.output
    assert client.terminated == [parent_job_id]
    assert github_output.read_text().splitlines() == [
        f"job_id={parent_job_id}",
        "state=RESOURCE_EXHAUSTED",
        "succeeded=true",
        "resource_exhausted=true",
    ]


@pytest.mark.parametrize(
    ("resource_exhaustion_policy", "pending_reason"),
    [
        (
            "shutdown",
            "Autoscaler: Unsatisfied autoscaler demand: no_matching_group: need device=tpu:v7-8",
        ),
        (
            "escalate",
            "Autoscaler: Unsatisfied autoscaler demand: tier_blocked: quota-pool tier monotonicity",
        ),
    ],
)
def test_wait_non_silent_conditions_still_escalate(
    monkeypatch: pytest.MonkeyPatch,
    resource_exhaustion_policy: str,
    pending_reason: str,
) -> None:
    parent_job_id = "/runner/canary"
    client = FakeIrisClient(
        [
            job_pb2.JobStatus(job_id=parent_job_id, state=job_pb2.JOB_STATE_RUNNING, has_children=True),
            job_pb2.JobStatus(
                job_id=f"{parent_job_id}/train",
                state=job_pb2.JOB_STATE_PENDING,
                pending_reason=pending_reason,
            ),
        ]
    )
    monkeypatch.setattr(iris_monitor, "open_iris_client", lambda **_kwargs: nullcontext(client))

    result = CliRunner().invoke(
        iris_monitor.cli,
        [
            "wait",
            "--job-id",
            parent_job_id,
            "--controller-url",
            "http://iris.test",
            "--poll-interval",
            "0.01",
            "--timeout",
            "0",
            "--resource-exhaustion-policy",
            resource_exhaustion_policy,
        ],
    )

    assert result.exit_code != 0
    assert client.terminated == []
