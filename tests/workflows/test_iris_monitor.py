# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from rigging.redaction import REDACTED_VALUE

from scripts.workflows import iris_monitor


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


def _job(
    job_id: str,
    state: str,
    *,
    failure_count: int = 0,
    preemption_count: int = 0,
) -> dict:
    return {
        "job_id": job_id,
        "state": state,
        "failure_count": failure_count,
        "preemption_count": preemption_count,
    }


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


def test_wait_for_child_job_uses_runtime_timeout_after_child_start() -> None:
    parent_id = "/runner/parent"
    child_id = f"{parent_id}/grug-train-canary-tpu-1"
    polls = iter(
        [
            [
                _job(parent_id, "JOB_STATE_RUNNING"),
                _job(child_id, "JOB_STATE_PENDING", preemption_count=1),
            ],
            [
                _job(parent_id, "JOB_STATE_RUNNING"),
                _job(child_id, "JOB_STATE_RUNNING", preemption_count=1),
            ],
            [
                _job(parent_id, "JOB_STATE_SUCCEEDED"),
                _job(child_id, "JOB_STATE_SUCCEEDED", preemption_count=1),
            ],
        ]
    )
    monotonic = iter([0, 4000, 5001, 5010])

    status = iris_monitor.wait_for_child_job(
        parent_id,
        iris_config=None,
        poll_interval=0,
        queue_timeout=5000,
        run_timeout=60,
        repo_root=iris_monitor._REPO_ROOT,
        list_job_rows=lambda: next(polls),
        clock=lambda: next(monotonic),
    )

    assert status.state == iris_monitor.JOB_STATE_SUCCEEDED


def test_wait_for_child_job_reports_queue_timeout() -> None:
    parent_id = "/runner/parent"
    child_id = f"{parent_id}/grug-train-canary-tpu-1"
    monotonic = iter([0, 4000])

    with pytest.raises(TimeoutError, match="queue/start timeout") as exc:
        iris_monitor.wait_for_child_job(
            parent_id,
            iris_config=None,
            poll_interval=30,
            queue_timeout=3000,
            run_timeout=60,
            repo_root=iris_monitor._REPO_ROOT,
            list_job_rows=lambda: [
                _job(parent_id, "JOB_STATE_RUNNING", failure_count=2),
                _job(child_id, "JOB_STATE_PENDING", preemption_count=1),
            ],
            clock=lambda: next(monotonic),
        )

    message = str(exc.value)
    assert "parent state=JOB_STATE_RUNNING" in message
    assert "child state=JOB_STATE_PENDING" in message
    assert "parent failure_count=2" in message
    assert "child preemption_count=1" in message


def test_wait_for_child_job_reports_runtime_timeout() -> None:
    parent_id = "/runner/parent"
    child_id = f"{parent_id}/grug-train-canary-tpu-1"

    polls = iter(
        [
            [
                _job(parent_id, "JOB_STATE_RUNNING"),
                _job(child_id, "JOB_STATE_RUNNING", preemption_count=1),
            ],
            [
                _job(parent_id, "JOB_STATE_RUNNING"),
                _job(child_id, "JOB_STATE_RUNNING", preemption_count=1),
            ],
        ]
    )
    monotonic = iter([0, 10, 80])

    with pytest.raises(TimeoutError, match="runtime timeout") as exc:
        iris_monitor.wait_for_child_job(
            parent_id,
            iris_config=None,
            poll_interval=0,
            queue_timeout=3000,
            run_timeout=60,
            repo_root=iris_monitor._REPO_ROOT,
            list_job_rows=lambda: next(polls),
            clock=lambda: next(monotonic),
        )

    message = str(exc.value)
    assert "phase=child-running" in message
    assert "child state=JOB_STATE_RUNNING" in message
    assert "child preemption_count=1" in message


def test_redact_pod_doc_redacts_env_values_and_preserves_context():
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
